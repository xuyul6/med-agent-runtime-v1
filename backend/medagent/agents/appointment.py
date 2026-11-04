"""The booking role: its model sees only code-resolved fields and one candidate, and can only propose a draft."""
import json
import re

from ..appointments.preparation import Preparation, followup_for, prepare
from ..appointments.resolution import pending_from_issued
from ..appointments.store import AppointmentStore, subject_label
from ..appointments.params import WINDOW_NAMES
from ..core.contracts import Intent, Risk
from ..tools.audit import appointment_attempts, observe_model_input
from ..core.observability import event
from ..tools.registry import ToolBudget, ToolBudgetExceeded, ToolContext
from .contracts import AppointmentInput, AgentInput
from .execution import ROLE_DATA_PREFACE, AgentResult, HandoffRequired, RoleAgent, finish_result, model_blocks
from .output_validation import AgentOutputError
from .profiles import WorkflowRun

LIMITATION = "这是本地模拟医院系统，不接入真实医院。"
MISSING = {"no_slot": ["slot"], "no_booking": ["booking"], "ambiguous": ["choice"]}
CLARIFY_TEXT = {"missing": "生成预约草稿前，还需要补充下面的信息。", "invalid": "有一项信息没法用于预约，请看下面的说明。",
                "no_slot": "这个时段已经没有号了，请看下面的可选时段。", "no_booking": "没有找到符合条件的预约。",
                "ambiguous": "符合条件的不止一个，请选择一个。"}
TOOL = {"create": "propose_appointment", "cancel": "propose_cancellation"}


class AppointmentAgent(RoleAgent):
    def __init__(self, *args, appointments: AppointmentStore, timezone: str = "Asia/Shanghai", **options):
        super().__init__(*args, **options)
        self.appointments, self.timezone = appointments, timezone

    def _result(self, workflow: WorkflowRun, request: AppointmentInput, status: str, missing: list[str], text: str,
                questions=(), options=(), reason: str | None = None) -> AgentResult:
        data = {"role": self.role, "status": status, "missing_fields": missing,
                "answer": {"text": text, "questions": list(questions)[:6], "next_steps": [], "reference_keys": [], "limitations": [LIMITATION]},
                "payload": {"action": request.action, "draft_ids": [], "options": list(options)[:6]}}
        result = finish_result(self.profile, self.instance, workflow, data, {}, reason)
        result.model_dependency_exercised = False
        return result

    def _clarify(self, workflow: WorkflowRun, request: AppointmentInput, preparation: Preparation) -> AgentResult:
        missing = MISSING.get(preparation.kind) or list(preparation.awaiting)[:6]
        return self._result(workflow, request, "needs_clarification", missing, CLARIFY_TEXT[preparation.kind],
                            preparation.questions, preparation.options)

    async def run(self, text: str, intent: Intent, memory: dict, context_mode: str, *, subject: str = "self", risk: Risk | None = None,
                  details: dict | None = None, skill_snapshot=None, tool_budget: ToolBudget | None = None, original_request: str = "",
                  user: str = "", session: str = "", **_unused) -> AgentResult:
        snapshot = skill_snapshot or self.skills.snapshot()
        document = snapshot.documents[self.role]
        event("skill_binding", role=self.role, version=document.version, sha256=document.sha256,
              snapshot_id=snapshot.snapshot_id, common_hash=snapshot.common_hash)
        workflow = WorkflowRun(self.profile, self.instance)
        risk = risk or Risk()
        if risk.level == "CRITICAL":
            raise HandoffRequired("risk_handoff")
        details = details or {}
        if set(details) & set(AgentInput.model_fields):
            raise ValueError("cannot_override_agent_context")
        request = AppointmentInput.model_validate({"text": text, "intent": intent, "subject": subject, "risk": risk, **details})
        workflow.complete("validate_input", "input")
        human_requested = bool(re.search(r"转人工|人工客服|联系真人|医生接手|转接医护", original_request or text))
        workflow.complete("check_handoff", "handoff_checked", requested_human=human_requested)
        if human_requested:
            workflow.complete("prepare_evidence", "evidence", skipped=True)
            return self._result(workflow, request, "handoff_required", [],
                                "本服务没有接入人工队列，未实际转接。请通过你掌握的官方渠道联系医护人员。", reason="human_requested")
        context = ToolContext(self.profile, request, self.index, user=user, session=session, appointments=self.appointments)
        budget = tool_budget if tool_budget is not None else ToolBudget()
        preparation = await prepare(request, context, self.tools, budget)
        workflow.complete("prepare_evidence", "evidence", candidate_count=len(preparation.candidates), preparation=preparation.kind)
        if preparation.kind != "ready":
            record = followup_for(preparation, request, user=user, session=session, subject=subject,
                                  now=self.appointments.now(), ttl=self.appointments.followup_ttl)
            if record is not None:
                self.appointments.save_followup(record)
            return self._clarify(workflow, request, preparation)
        if self.model is None:
            return self._result(workflow, request, "needs_clarification", ["choice"], "离线预览仅展示执行结构，不生成预约草稿。")
        candidates = preparation.candidates
        data = json.dumps({"action": request.action, "subject_label": subject_label(subject),
                           "fields": [{"hospital": item.hospital, "department": item.department,
                                       "day": item.day, "window": WINDOW_NAMES[item.window]} for item in candidates],
                           "candidates": [item.model_dump(mode="json") for item in candidates]}, ensure_ascii=False)
        event("context", role=self.role, mode="structured", characters=len(data), reference_count=0)
        system = self._system_prompt(snapshot, "selective", intent, False)
        definitions = [self.tools.registry.specs[TOOL[request.action]].definition()]
        messages = [{"role": "user", "content": ROLE_DATA_PREFACE + "\n" + data}]
        # The tool round runs without the output schema: with it, the model wrote the final JSON and skipped the tool.
        observe_model_input(self.role, self.instance, 0, messages)
        blocks = model_blocks(await self.model.message("agent_appointment_action", system, messages, definitions, budget=1900))
        calls = [block for block in blocks if block.get("type") == "tool_use"]
        if not calls:
            raise AgentOutputError("appointment_draft_missing")
        with appointment_attempts(calls, self.role, self.instance, 0, request.action) as attempts:
            if len(calls) > 3:
                raise ToolBudgetExceeded("tool_round_or_batch_limit")
            # Extra calls get error results: the tool itself allows one draft per turn.
            messages += [{"role": "assistant", "content": blocks}, {"role": "user", "content": await self._tool_results(calls, context, budget, attempts)}]
        result = await self._generate(system, data, intent, request, context, budget, workflow, False, definitions=[], messages=messages)
        payload = result.execution.payload
        # With checked candidates the only valid outcome is one draft for each; a model clarification here is a failure.
        if result.execution.status != "answered" or len(context.issued) != len(candidates):
            raise AgentOutputError("appointment_draft_missing")
        if payload["action"] != request.action or payload["draft_ids"] != list(context.issued):
            raise AgentOutputError("appointment_draft_mismatch")
        result.pending_actions = pending_from_issued(context.issued, self.timezone)
        for draft, _ in context.issued.values():
            event("appointment_draft", role=self.role, draft_id=draft.draft_id, draft_kind=draft.kind, summary=draft.summary)
        return result
