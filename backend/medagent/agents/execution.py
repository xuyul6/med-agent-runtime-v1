import hashlib
import json
import re
from dataclasses import dataclass
from .contracts import AgentInput, ReportInput
from ..core.contracts import AgentAssignment, AgentExecution, Answer, Intent, Risk
from ..understanding.consultation_needs import is_fever_followup
from ..understanding.conversation import VisitFollowupEvidence, positive_request
from ..core.observability import event
from ..knowledge.medication_evidence import eligible_references, query_identity
from .output_validation import AgentOutputError
from .profiles import AgentProfile, DEFAULT_REGISTRY, ProfileRegistry, WorkflowRun
from ..knowledge.retrieval import has_query_overlap
from ..understanding.conversation import visit_query_has_context
from .skills import SkillSnapshot
from ..tools.registry import ToolBudget, ToolBudgetExceeded, ToolContext, ToolExecutor
from .checks import (
    REPORT_RANGE_DEFINITION_HASH, REPORT_RANGE_QUESTION, REPORT_RANGE_SOURCE_HASH,
    REPORT_STRUCTURE_SOURCE_HASH, REPORT_UNIT_SOURCE_HASH, check_boundaries, checked_role_output,
    explicit_reference_request, general_report_source_request, mixed_personal_report_without_item,
    personal_report_without_item, report_range_source_quotes, safe_report_status_repair, validate_draft
)


def model_blocks(result) -> list[dict]:
    if not isinstance(result,dict) or not isinstance(result.get("content"),list):
        raise AgentOutputError("invalid_model_envelope")
    blocks = result["content"]
    if any(not isinstance(block,dict) or block.get("type") == "text" and not isinstance(block.get("text"),str)
           or block.get("type") == "tool_use" and (not all(isinstance(block.get(key),str) and block[key] for key in ("id","name"))
              or not isinstance(block.get("input"),dict)) for block in blocks):
        raise AgentOutputError("invalid_model_envelope")
    return blocks


@dataclass
class AgentResult:
    answer: Answer
    references: list
    instance: str
    execution: AgentExecution
    model_dependency_exercised: bool = True


class HandoffRequired(RuntimeError):
    def __init__(self, target: str):
        self.target = target
        super().__init__(target)


def finish_result(profile: AgentProfile, instance: str, workflow: WorkflowRun, data: dict,
                  references: dict, reason: str | None = None) -> AgentResult:
    workflow.complete("compose_answer", "draft")
    output = checked_role_output(profile, data, references)
    if reason is None:
        reason = {"needs_clarification":"missing_information", "handoff_required":"professional_review"}.get(output.status)
    target = next((condition.target for condition in profile.handoff_conditions if condition.when == reason), None)
    workflow.complete("validate_output", "validated_output", output_status=output.status,
                      handoff_reason=reason, handoff_target=target)
    return AgentResult(output.answer, list(references.values()), instance,
        AgentExecution(role=profile.role, instance=instance, profile_version=profile.version, status=output.status,
                       workflow=workflow.completed, missing_fields=output.missing_fields,
                       handoff_reason=reason, handoff_target=target, payload=output.payload.model_dump(mode="json")))


class RoleAgent:
    def __init__(self, role, instance, model, index, skills, tools, registry: ProfileRegistry = DEFAULT_REGISTRY):
        self.profile = registry[role]
        if self.profile.deterministic:
            raise ValueError("deterministic_role_requires_risk_executor")
        self.role, self.instance = role, instance
        self.model, self.index, self.skills, self.tools = model, index, skills, tools

    async def run(self, text: str, intent: Intent, memory: dict, context_mode: str, *,
                  subject: str = "self", subject_changed: bool = False, risk: Risk | None = None, details: dict | None = None,
                  skill_snapshot: SkillSnapshot | None = None, tool_budget: ToolBudget | None = None,
                  reference_query: str = "", assignment: AgentAssignment | None = None,
                  original_request: str = "", visit_followup: VisitFollowupEvidence | None = None) -> AgentResult:
        profile = self.profile
        if assignment is not None and (assignment.role != self.role or assignment.intent != intent):
            raise ValueError("assignment_outside_role_scope")
        snapshot = skill_snapshot or self.skills.snapshot()
        document = snapshot.documents[self.role]
        event("skill_binding", role=self.role, version=document.version, sha256=document.sha256,
              snapshot_id=snapshot.snapshot_id, common_hash=snapshot.common_hash)
        workflow = WorkflowRun(profile, self.instance)
        risk = risk or Risk()
        if risk.level == "CRITICAL":
            raise HandoffRequired("risk_handoff")
        details = details or {}
        if set(details) & set(AgentInput.model_fields):
            raise ValueError("cannot_override_agent_context")
        request = profile.input_contract.model_validate({"text":text, "intent":intent, "subject":subject,
                                                        "risk":risk, **details})
        report_payload = ({"report_items":[item.model_dump(mode="json") for item in request.items]}
                          if isinstance(request, ReportInput) else None)
        workflow.complete("validate_input", "input")
        human_requested = bool(re.search(r"转人工|人工客服|联系真人|医生接手|转接医护", original_request or text))
        workflow.complete("check_handoff", "handoff_checked", requested_human=human_requested)
        permitted = profile.knowledge_domains
        evidence_query = original_request or text
        if visit_followup is not None:
            if intent != Intent.VISIT or subject_changed or visit_followup.current != original_request:
                raise ValueError("invalid_visit_followup_context")
            evidence_query += "\n"+visit_followup.prior_clause
            source_hash = hashlib.sha256(visit_followup.prior_turn_id.encode()).hexdigest()[:16]
            event("visit_context_recovered", origin="previous_user_turn", source_turn_hash=source_hash,
                  age_seconds=round(visit_followup.age_seconds, 1))
        context = ToolContext(profile, request, self.index, evidence_query=evidence_query)
        references = context.references
        budget = tool_budget if tool_budget is not None else ToolBudget()
        if human_requested:
            workflow.complete("prepare_evidence", "evidence", skipped=True)
            return self._limited_result(workflow, "handoff_required", [],
                "本服务没有接入人工队列，未实际转接。请通过你掌握的官方渠道联系医护人员。", references,
                reason="human_requested")
        if intent == Intent.SYMPTOMS and request.consultation_need:
            workflow.complete("prepare_evidence", "evidence", skipped=True, reason="consultation_focus")
            if request.consultation_need in {"clarify", "clarify_followup"}:
                event("consultation_focus", role=self.role, action="clarify")
                text = ("你提到：" + original_request.strip().rstrip("。！!？?") + "。"
                        if request.consultation_need == "clarify_followup" and is_fever_followup(original_request)
                        else "可以，我先确认你想了解的方向。")
                result = self._limited_result(workflow, "needs_clarification", ["consultation_focus"],
                    text, references)
            elif request.consultation_need == "food":
                event("consultation_focus", role=self.role, action="food")
                result = self._limited_result(workflow, "handoff_required", [],
                    "你想了解发烧时的饮食。当前没有适用的饮食资料，暂时无法提供具体饮食建议；可以向医师或营养师咨询。", references,
                    reason="professional_review", next_steps=["如方便，可记录实测体温、开始时间及是否有进食困难，向医师或营养师说明；不清楚的内容保留未知。"])
            else:
                event("consultation_focus", role=self.role, action="medicine")
                result = self._limited_result(workflow, "handoff_required", [],
                    "你想了解退烧药的信息。如果已有具体药名，可以告诉我以便查询公开资料；是否适合个人使用，请咨询医师或药师。", references,
                    reason="professional_review", next_steps=["如果已有药品包装，可核对通用名称后查询；拿不到或不确定时保留未知。",
                        "如需决定是否使用药物，请联系医师或药师核实；本服务未实际转接。"])
            result.model_dependency_exercised = False
            return result
        preflight_name = next(name for name in profile.tool_scope if self.tools.registry.specs[name].stage == "preflight")
        preflight = await self.tools.call(preflight_name, {}, context, budget)
        notes = {preflight_name:preflight.model_dump(mode="json")}
        missing = preflight.missing_fields
        mixed_missing = mixed_personal_report_without_item(original_request or text) if isinstance(request, ReportInput) else False
        personal_missing = personal_report_without_item(original_request or text) if isinstance(request, ReportInput) else False
        grounded = memory.get("grounded_fields", {})
        unknown_report_fields = {name for name in ("report_item", "value", "unit", "reference_range")
            if isinstance(grounded, dict) and isinstance(grounded.get(name), dict)
            and grounded[name].get("state") == "unknown"
            and isinstance(grounded[name].get("source"), dict)
            and grounded[name]["source"].get("origin") == "current_user"}
        explicitly_unknown = "report_item" in unknown_report_fields and len(unknown_report_fields) >= 2
        if isinstance(request, ReportInput) and not request.items and (mixed_missing or personal_missing or explicitly_unknown):
            # Evidence of a question is not a report item or a confirmed patient fact.
            missing = ["report_item", "value", "unit", "reference_range"]
            event("report_field_fallback", role=self.role,
                  reason=("mixed_personal_report_without_item" if mixed_missing else
                          "personal_report_without_item" if personal_missing else "explicit_unknown_report_fields"), fields=missing)
        if missing:
            # A general term question can be answered separately from an incomplete personal item.
            if intent == Intent.REPORT and positive_request(original_request or text, REPORT_RANGE_QUESTION):
                await self.tools.call("lookup_reference", {"query":"参考范围", "domain":"report"}, context, budget)
                reference = references.get("lab-context")
                trusted = reference is not None and reference.approved and reference.domain == Intent.REPORT and (
                    hashlib.sha256(json.dumps([reference.model_dump(mode="json")], ensure_ascii=False, sort_keys=True,
                                              separators=(",", ":"), allow_nan=False).encode()).hexdigest() == REPORT_RANGE_SOURCE_HASH)
                workflow.complete("prepare_evidence", "evidence", reference_count=1 if trusted else 0)
                if trusted:
                    # Only quote the reviewed source: no personal result or model inference is added.
                    excerpt = reference.text
                    check_boundaries(excerpt)
                    payload = {**report_payload, "term_explanations":[{"text":excerpt, "reference_keys":[reference.key]}]}
                    return self._limited_result(workflow, "needs_clarification", missing,
                        "以下是当前资料对报告阅读的通用说明："+excerpt+"\n个人报告的原始字段仍需核对。",
                        {reference.key:reference}, payload=payload, reference_keys=[reference.key])
                return self._limited_result(workflow, "needs_clarification", [*missing, "source"],
                    "个人报告的原始字段尚不完整；当前没有可用于通用参考范围说明的匹配资料。", {}, payload=report_payload)
            workflow.complete("prepare_evidence", "evidence", skipped=True, reason="missing_information")
            if intent == Intent.MEDICINE and subject_changed and "medication_name" in missing:
                event("clarification",role=self.role,reason="subject_changed_medication_name_missing",fields=missing)
                return self._limited_result(workflow,"needs_clarification",missing,
                    "你已切换咨询对象。为避免混用不同人的药品信息，请重新确认当前对象要查询的药品名称；不清楚时可以保留未知。",references,
                    next_steps=["请核对当前对象要查询药品的外包装或说明书，照原文提供“通用名称”或“药品名称”。",
                                "拿不到包装或仍不确定时，请保留“药名不清楚”；可先请药师结合实物核对，确认后再查询对应资料。"])
            return self._limited_result(workflow, "needs_clarification", missing,
                "已提供的结构化信息尚不完整，请补充必要字段；不清楚的内容可以保留未知。", references,
                payload=report_payload)
        if intent == Intent.VISIT and request.institution_specific:
            workflow.complete("prepare_evidence", "evidence", skipped=True, reason="institution_source_unavailable")
            return self._limited_result(workflow, "needs_clarification", ["source"],
                "本服务尚不能核验某家机构专属资料，无法确认该机构的流程或要求。", references,
                next_steps=["请自行通过目标机构官方渠道核对预约方式、流程和所需材料。",
                            "如需一般就医准备，可先整理已有资料和向医护人员确认的问题。"])
        if intent == Intent.VISIT and not visit_query_has_context(evidence_query):
            workflow.complete("prepare_evidence", "evidence", skipped=True, reason="visit_context_missing")
            return self._limited_result(workflow, "needs_clarification", ["visit_purpose"],
                "当前问题没有明确说明是否涉及就医准备，请先确认咨询目的。", references,
                next_steps=["如果想了解就医准备，可在问题中说明看医生、就诊或复诊的需求。"])
        # These intents cannot answer factual questions without a reference lookup.
        required = intent in {Intent.MEDICINE, Intent.REPORT, Intent.VISIT}
        if required:
            await self.tools.call("lookup_reference", {"query":reference_query or text,"domain":intent.value}, context, budget)
        for name in profile.tool_scope:
            if self.tools.registry.specs[name].stage == "context":
                result = await self.tools.call(name, {}, context, budget)
                notes[name] = result.model_dump(mode="json", exclude_none=True)
        exclude_unrequested_range_definition = False
        if intent == Intent.REPORT and isinstance(request, ReportInput) and not request.items:
            current = original_request or text
            has_prior_user = any(isinstance(row, dict) and row.get("role") == "user"
                                 for row in memory.get("messages", []))
            asks_range = re.search(r"参考|范围|正常值|区间|上下限|上界|下界|界限|百分位|阈值|统计|分布|\brange\b|reference\s*[-_]?\s*(?:range|interval)", current, re.I)
            # Remove only an explicit refusal of personal analysis from the gating
            # text. The original request remains intact in the model context.
            screening_text = re.sub(r"(?:不要|不需要|无需|不想)(?:你|做|进行)?(?:分析|判断|解读|解释|诊断)(?:我的|个人)?病情", "", current)
            workflow_only = (re.search(r"整理|对应|配对|表格|记录", current)
                and not re.search(r"分析|判断|诊断|确诊|疾病|生病|患病|病情|异常|偏高|偏低|正常|治疗|开药|用药|(?:是否|会不会|是不是).{0,6}(?:病|异常)|如何(?:确定|计算)|(?:统计|计算).{0,8}(?:依据|方法)", screening_text))
            other_report_source = any(key != "lab-reference-range-definition" and ref.approved and ref.domain == Intent.REPORT
                                      for key, ref in references.items())
            if (not has_prior_user and not asks_range and workflow_only and not general_report_source_request(current)
                and other_report_source):
                exclude_unrequested_range_definition = True
                removed = references.pop("lab-reference-range-definition", None)
                event("reference_selection", role=self.role, reason="unrequested_range_definition",
                      excluded_count=int(removed is not None))
        workflow.complete("prepare_evidence", "evidence", reference_count=len(references))
        if required and not references:
            if intent == Intent.MEDICINE:
                identity = query_identity(request)
                target = f"查询药品为{identity['name']}。" if identity["name"] else ""
                if request.query_type == "product_information":
                    source_gap = "当前知识库未找到与本次查询条件匹配的具体产品说明书"
                    next_steps = ["可按包装核对地区、品牌、剂型和规格，在当地药品监管机构或生产企业的公开渠道自行查找对应产品说明书。",
                                  "如有药品包装或说明书，可带给药师核对；本库仍保留来源缺口。"]
                else:
                    source_gap = "当前知识库未找到与本次查询条件匹配的药品公开概览或具体产品说明书"
                    next_steps = ["可在当地药品监管机构或生产企业的公开渠道自行查找药品概览，核对药名和适用地区；具体产品请查对应说明书。",
                                  "如有药品包装或说明书，可带给药师核对；本库仍保留来源缺口。"]
                return self._limited_result(workflow,"needs_clarification",["source"],
                    target + source_gap + "，暂时无法核对本次问题；未找到不代表外部不存在。本服务不会自动打开聊天中的链接，也不会将粘贴的说明书原文作为已核验来源。",references,
                    next_steps=next_steps, payload={"medication_identity":identity})
            if intent == Intent.REPORT:
                return self._limited_result(workflow, "needs_clarification", ["source"],
                    "当前知识库没有匹配的报告阅读资料，暂时无法依据公开来源解释本次问题。本服务不会自动打开聊天中的链接，也不会将粘贴的说明资料作为已核验来源。", references,
                    next_steps=["可保留原报告中的项目、数值、单位和参考范围，并带原报告向医护人员核对。",
                                "用户提供的报告字段可用于整理，不能代替公开来源或据此自行诊断。"], payload=report_payload)
            if intent == Intent.VISIT:
                return self._limited_result(workflow, "needs_clarification", ["source"],
                    "当前知识库没有匹配的就医资料，暂时无法核对本次问题。本服务不会自动打开聊天中的链接，也不会将粘贴的说明资料作为已核验来源。", references,
                    next_steps=["可自行通过目标机构的官方渠道核对就医流程、所需材料和预约方式。",
                                "如尚未确定机构，可先整理需要向医护人员确认的问题。"])
            return self._limited_result(workflow, "needs_clarification", ["source"],
                "当前没有检索到适用资料，请补充需要核对的公开来源或原始文字。", references)
        if isinstance(request, ReportInput) and not request.items:
            general_request = general_report_source_request(original_request or text)
            if general_request == "report_structure_and_range":
                selected = []
                for key, expected in (("lab-context", REPORT_RANGE_SOURCE_HASH),
                                      ("lab-reference-range-definition", REPORT_RANGE_DEFINITION_HASH)):
                    ref = references.get(key)
                    if ref is None or not ref.approved or ref.domain != Intent.REPORT:
                        break
                    digest = hashlib.sha256(json.dumps([ref.model_dump(mode="json")], ensure_ascii=False, sort_keys=True,
                                                       separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                    if digest != expected:
                        break
                    selected.append(ref)
                if len(selected) == 2:
                    event("report_source_direct", role=self.role, category=general_request, source_count=2)
                    organization = ("可以按项目名称、结果数值、单位和本报告参考范围逐项记录；"
                                    "如需解释个人结果，可交由医护人员结合实际情况核对。")
                    payload = {**report_payload, "term_explanations":[
                        {"text":organization, "reference_keys":[selected[0].key]},
                        {"text":selected[1].text, "reference_keys":[selected[1].key]}]}
                    result = self._limited_result(workflow, "answered", [], organization+"\n\n"+selected[1].text,
                        {ref.key:ref for ref in selected}, payload=payload, reference_keys=[ref.key for ref in selected])
                    result.answer.limitations.append("仅说明通用报告整理和参考范围，不判断个人结果。")
                    result.model_dependency_exercised = False
                    return result
                general_request = None
            if general_request == "report_unit_followup":
                recent_users = [row.get("content", "") for row in memory.get("messages", [])[-4:]
                                if isinstance(row, dict) and row.get("role") == "user"]
                if subject_changed or not any(isinstance(value, str) and "报告" in value for value in recent_users):
                    general_request = None
            structure_request = general_request in {"report_structure_mention", "report_unit_followup"}
            if general_request == "report_unit_followup":
                key, expected_hash = "lab-unit-overview", REPORT_UNIT_SOURCE_HASH
            elif general_request == "report_structure_mention":
                key, expected_hash = "lab-structure-overview", REPORT_STRUCTURE_SOURCE_HASH
            elif general_request == "report_reading_limitations":
                key, expected_hash = "lab-context", REPORT_RANGE_SOURCE_HASH
            else:
                key, expected_hash = "lab-reference-range-definition", REPORT_RANGE_DEFINITION_HASH
            reference = references.get(key)
            trusted = reference is not None and reference.approved and reference.domain == Intent.REPORT and (
                hashlib.sha256(json.dumps([reference.model_dump(mode="json")], ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":"), allow_nan=False).encode()).hexdigest() == expected_hash)
            if general_request and trusted:
                event("report_source_direct", role=self.role, source_key=reference.key, category=general_request)
                if general_request == "report_structure_mention":
                    excerpt = ("是的，报告中可以看到检验项目名称及相应结果。"
                               "阅读时可以先找到想了解的项目，再核对该项目在报告中的结果；"
                               "不同报告的排版可能不同。")
                elif general_request == "report_reading_limitations":
                    excerpt = "。".join(reference.text.split("。")[:2])+"。"
                else:
                    excerpt = reference.text
                payload = {**report_payload, "term_explanations":[{"text":excerpt, "reference_keys":[reference.key]}]}
                result = self._limited_result(workflow,"answered",[],excerpt,{reference.key:reference},
                                              payload=payload,reference_keys=[reference.key])
                result.model_dependency_exercised = False
                if general_request == "reference_range_mention":
                    result.answer.limitations.append("本次仅提供所提及参考范围的通用说明，如有其他问题请补充。")
                elif structure_request:
                    result.answer.limitations.append("这是通用整理提示，无需现在提供个人报告信息。")
                    if general_request == "report_structure_mention":
                        result.answer.questions.append("接下来你更想了解某个项目名称，还是学习如何查看报告？")
                elif general_request == "report_learning":
                    result.answer.limitations.append("本次仅介绍参考范围，不覆盖报告中的其他术语。")
                return result
        recent = memory.get("messages", []) if context_mode == "full" else memory.get("messages", [])[-4:]
        public = [ref.model_dump(mode="json") for ref in references.values()]
        if context_mode == "full":
            available = [r for r in self.index.documents.values() if r.approved and r.domain in permitted
                         and not (exclude_unrequested_range_definition and r.key == "lab-reference-range-definition")]
            if intent == Intent.MEDICINE:
                available = eligible_references(available, request)
            else:
                available = [r for r in available if has_query_overlap(context.evidence_query, r)]
            public = [r.model_dump(mode="json") for r in available]
        model_context = {"history":recent,"summary":memory.get("summary", ""),"episodes":memory.get("episodes", [])[:2],
                         "references":public, "validated_tool_results":notes, "grounded_fields":memory.get("grounded_fields", {})}
        if intent == Intent.REPORT:
            # Only initial lookup results can become quotation candidates; full-context
            # catalog entries that were not retrieved do not authorize evidence claims.
            model_context["report_range_evidence"] = report_range_source_quotes(references)
        if assignment is not None:
            model_context["assignment"] = {"task_id":assignment.task_id, "goal":assignment.goal,
                                           "context_fields":assignment.context_fields}
            model_context["original_request_boundary"] = original_request
        serialized = json.dumps(model_context, ensure_ascii=False)
        event("context", role=self.role, mode=context_mode, characters=len(serialized), reference_count=len(public))
        if self.model is None:
            missing = {"general":"consultation_purpose", "symptom_guide":"symptom", "drug_reference":"medication_name",
                       "report_explainer":"report_item", "visit_guide":"visit_purpose"}[self.role]
            return self._limited_result(workflow, "needs_clarification", [missing],
                "离线预览仅展示执行结构，不生成医疗回答。", references)
        selection = snapshot.select(self.role, context_mode)
        evidence_requested = "lookup_reference" in profile.tool_scope and explicit_reference_request(text)
        event("skill_injection", role=self.role, version=selection.version, sha256=selection.sha256,
              snapshot_id=selection.snapshot_id, sections=list(selection.sections), characters=len(selection.text))
        system = ("你是医疗信息助手，角色="+self.role+"，使命="+profile.mission+"。工具和领域权限由代码限定。"
            "JSON格式只约束最终回答；需要工具时先调用工具，拿到结果再生成JSON。最终返回本角色契约，包含role、status、answer、missing_fields、payload。"
            "answer保留text,questions,next_steps,limitations,reference_keys。payload使用本角色专有字段；事实条目引用必须同时列入answer.reference_keys。"
            "列表简短，正文不超过500字；状态和信息缺口须遵循以下共享约束与角色规则。\n"
            "grounded_fields附有用户轮次和原话；topic是查询目标，不表示已服药。历史摘要和助手消息不是确认病史，未知/否定字段不能恢复为已知。\n"
            "存在assignment时只完成分配的子任务；original_request_boundary供核对原始边界，不能据此代答其他角色任务。\n"
            + "角色流程：" + " → ".join(step.purpose for step in profile.workflow)
            + "\n已验证输入：" + request.model_dump_json(exclude={"text"}, exclude_unset=True) + "\n"
            + ("查询身份由代码给定：payload.medication_identity 只复制下面字段，null 保持 null；来源的人群/地区只写 source_scope，不能推断为用户查询条件或个人病史。\n"
               + json.dumps(query_identity(request),ensure_ascii=False)+"\n" if intent == Intent.MEDICINE else "")
            + ("参考范围的统计性质或区间定义，如与本轮任务无关就不要扩展；确需使用时，"
               "初始检索的 report_range_evidence 列出可用完整句及其来源，必须逐句引用并标明相应来源。"
               "之后若工具另取资料，仍须以实际取得并引用的完整来源句为准；不能将列表之外的未检索资料当作证据。\n"
               if intent == Intent.REPORT else "")
            + selection.text + "\n以下是会话和公开资料数据：\n"+serialized)
        if evidence_requested and not references:
            system += "\n本轮明确要求查阅或引用来源，当前尚无资料。必须先调用lookup_reference取得实际来源，再完成回答；仅描述模板或说来源待核实不算完成。检索无适用资料时明确说明无法完成，不编造来源。"
        messages = [{"role":"user", "content":text}]
        definitions = self.tools.registry.definitions(profile)
        repairing = False
        for turn in range(3):
            result = await self.model.message("agent_"+self.role, system, messages, definitions if turn < 2 and not repairing else None, budget=1900)
            try:
                blocks = model_blocks(result)
            except AgentOutputError as exc:
                event("agent_output_validation",role=self.role,instance=self.instance,model_round=turn+1,repair_requested=False,**exc.diagnostic())
                raise
            calls = [block for block in blocks if block.get("type") == "tool_use"]
            if calls:
                if repairing:
                    exc = AgentOutputError("unexpected_tools_during_repair")
                    event("agent_output_validation",role=self.role,instance=self.instance,model_round=turn+1,repair_requested=False,**exc.diagnostic())
                    raise exc
                if turn >= 2 or len(calls) > 3:
                    event("tool_batch", role=self.role, status="round_or_batch_limit", call_count=len(calls))
                    raise ToolBudgetExceeded("tool_round_or_batch_limit")
                messages.append({"role":"assistant", "content":blocks})
                outputs = []
                for call in calls:
                    try:
                        value = await self.tools.call(call["name"], call.get("input", {}), context, budget)
                        outputs.append({"type":"tool_result","tool_use_id":call["id"],"content":value.model_dump_json()})
                    except ToolBudgetExceeded:
                        raise
                    except Exception:
                        outputs.append({"type":"tool_result","tool_use_id":call["id"],"content":"工具不可用或参数不被允许。","is_error":True})
                messages.append({"role":"user", "content":outputs})
                continue
            text_result = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
            try:
                output = validate_draft(profile,text_result,request,references,evidence_requested)
                completed = finish_result(profile,self.instance,workflow,output,references)
            except AgentOutputError as exc:
                # This evidence error gets one narrowly scoped correction, not a blanket
                # permission to repair other evidence, safety, or schema failures.
                report_range_repair = (self.role == "report_explainer"
                    and exc.code == "report_range_claim_without_exact_source"
                    and any(ref.approved and ref.domain == Intent.REPORT for ref in references.values()))
                report_status_repair = (self.role == "report_explainer" and exc.code == "invalid_output_shape"
                    and exc.rules == ("missing_fields_require_clarification",)
                    and safe_report_status_repair(profile, text_result, request, references))
                repair = (exc.repairable or report_range_repair or report_status_repair) and not repairing and turn < 2
                strategy = ("report_range_evidence_only" if report_range_repair else
                            "report_status_only" if report_status_repair else None)
                event("agent_output_validation",role=self.role,instance=self.instance,model_round=turn+1,repair_requested=repair,
                      **({"repair_strategy":strategy} if repair and strategy else {}),
                      **exc.diagnostic())
                if not repair: raise
                repairing = True
                task = ("仅删除或改正不能与已引用资料完整原句对应的参考范围统计性质或区间定义；"
                        "如需此类表述，只能逐字使用 permitted_source_sentences 的完整句及其来源，列表为空时不要补写定义。"
                        "只能使用本次已有资料，不能补造事实或引用，保留用户已提供的信息及医学边界。"
                        "重新输出完整角色契约，不要调用工具。" if report_range_repair else
                        "只纠正 status 与 missing_fields 的冲突：当前问题不需要个人字段时，使用 answered 且 missing_fields 为空；"
                        "确有必要字段缺失时使用 needs_clarification 并只列实际缺口。不要编造用户信息、修改引用或越过医学边界；"
                        "重新输出完整角色契约，不要调用工具。" if report_status_repair else
                        "只纠正 JSON 格式或查询身份字段；引用、医学边界、状态和未知约束不变。不要调用工具。")
                feedback = {"task":task,
                            "validation":exc.diagnostic()}
                if report_range_repair:
                    feedback["repair_strategy"] = "report_range_evidence_only"
                    feedback["permitted_source_sentences"] = report_range_source_quotes(references)
                elif report_status_repair:
                    feedback["repair_strategy"] = "report_status_only"
                if intent == Intent.MEDICINE: feedback["medication_identity"] = query_identity(request)
                # Draft is returned only to this model context, never persisted in traces/logs.
                messages += [{"role":"assistant","content":[{"type":"text","text":text_result}]},
                             {"role":"user","content":json.dumps(feedback,ensure_ascii=False)}]
                continue
            if repairing: event("agent_output_repair",role=self.role,instance=self.instance,status="accepted",model_round=turn+1)
            return completed
        raise ValueError("tool_round_budget_exceeded")

    def _limited_result(self, workflow: WorkflowRun, status: str, missing: list[str], text: str,
                        references: dict, reason: str | None = None, *, next_steps: list[str] | None = None,
                        payload: dict | None = None, reference_keys: list[str] | None = None) -> AgentResult:
        payload_type = self.profile.output_contract.model_fields["payload"].annotation
        data = {"role":self.role, "status":status, "missing_fields":missing,
                "answer":{"text":text,"next_steps":next_steps or [], "reference_keys":reference_keys or [],
                          "limitations":["未作医学判断，也未执行医院操作或人工连接。"]},
                "payload":payload_type.model_validate(payload or {}).model_dump()}
        return finish_result(self.profile, self.instance, workflow, data, references, reason)


class RiskHandoffAgent:
    def __init__(self, registry: ProfileRegistry = DEFAULT_REGISTRY, tools: ToolExecutor | None = None):
        self.profile = registry["risk_handoff"]
        self.instance = "risk_handoff.deterministic"
        self.tools = tools if tools is not None else ToolExecutor()

    def run(self, text: str, risk: Risk, emergency_phone: str, subject: str = "self", tool_budget: ToolBudget | None = None) -> AgentResult:
        profile = self.profile
        request = profile.input_contract.model_validate({"text":text, "intent":Intent.URGENT, "risk":risk, "subject":subject})
        workflow = WorkflowRun(profile, self.instance)
        workflow.complete("validate_input", "input")
        workflow.complete("check_handoff", "handoff_checked", reason="critical_risk")
        summary = self.tools.call_local("create_handoff_summary", {}, ToolContext(profile, request),
                                        tool_budget if tool_budget is not None else ToolBudget(1))
        workflow.complete("prepare_evidence", "evidence", skipped=True)
        notice = f"你描述的信息包含需要立即线下评估的风险信号。请立即联系{emergency_phone}或就近急诊，不要等待在线问答。"
        next_steps = ["向医护说明症状开始时间、当前状态及已知用药。"]
        if "self_harm" in risk.reasons:
            next_steps.insert(0, "如果有伤害自己的想法，请让身边可信任的人陪着你，并联系当地心理危机干预热线。")
        return finish_result(profile, self.instance, workflow,
            {"role":"risk_handoff", "status":"handoff_required", "missing_fields":[],
             "answer":{"text":notice, "next_steps":next_steps,
                       "limitations":["本服务没有接入急救调度或人工队列；未实际转接。"]},
             "payload":{"risk_notice":notice, "known_facts":summary.known_facts,
                        "next_steps":next_steps, "handoff_connected":False}}, {}, "critical_risk")
