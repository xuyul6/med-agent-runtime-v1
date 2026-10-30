import asyncio
import math
import time
import uuid
from .understanding.classification import IntentRouter
from .core.contracts import (AgentAssignment, Answer, ExecutionAttempt, InstanceChoice, Intent, Question, Reply, RoutingDecision,
                             RoutingFallback, TaskUnderstanding, Understanding)
from .agents.execution import RoleAgent, RiskHandoffAgent
from .agents.checks import checked_role_output
from .agents.skills import Skills
from .tools.registry import ToolBudget, ToolExecutor
from .memory import scope_key
from .core.observability import CURRENT, RunLog, event
from .understanding.risk import RiskGate
from .agents.profiles import DEFAULT_REGISTRY, ProfileRegistry
from .understanding.conversation import SubjectTracker, institution_request, recover_visit_followup, role_details
from .understanding.consultation_needs import consultation_need
from .understanding.routing import RoutingPlanner
from .agents.collaboration import compose
from .agents.single import INSTANCE as SINGLE_INSTANCE, SINGLE_AGENT, SingleAgent
from .monitoring import Monitor, MonitorExcluded, MonitorPolicy
from .agents.output_validation import AgentOutputError


def role_input(assignment, understanding, question, context, clock):
    """Input for one assignment: only the fields routing granted to the role, plus role-specific flags."""
    role = assignment.role
    scoped = understanding.model_copy(update={"primary":assignment.intent,
        "tasks":[TaskUnderstanding(intent=assignment.intent, goal=assignment.input_text[:300])],
        "entities":{name:understanding.entities[name] for name in assignment.context_fields}})
    if role == "drug_reference" and not any(name == "medication_name" and fact.state in {"provided", "topic"}
                                           for name,fact in scoped.entities.items()):
        scoped.missing_fields = ["medication_name"]
    details = role_details(scoped, role)
    if role == "symptom_guide" and context.get("consultation_need") in {"clarify", "clarify_followup", "food", "medicine"}:
        details["consultation_need"] = context["consultation_need"]
    # The assignment excerpt is bounded; do not lose a later institution question.
    if role == "visit_guide" and institution_request(question.text):
        details["institution_specific"] = True
    context = {**context, "grounded_fields":{name:fact.model_dump(mode="json") for name,fact in scoped.entities.items()}}
    visit_followup = None
    if role == "visit_guide" and understanding.primary == Intent.VISIT and not understanding.subject_ambiguous:
        try:
            visit_followup = recover_visit_followup(question.text, context,
                subject_changed=understanding.subject_changed, now=clock())
        except (TypeError, ValueError, OverflowError):
            visit_followup = None
    target = scoped.entities.get("medication_name")
    query = (target.value+" 药品公开资料 "+assignment.input_text if role == "drug_reference" and target and target.state in {"provided", "topic"}
             else assignment.input_text)
    return details, context, visit_followup, query[:6000]


class MedicalRuntime:
    def __init__(self, settings, model, vectors, index, memory, profiles=None, skills=None, fallback_model=None, judge_model=None):
        self.settings, self.model, self.index, self.memory = settings, model, index, memory
        self.judge_model = judge_model
        self.monitor, self.risk = Monitor(MonitorPolicy.from_settings(settings)), RiskGate()
        self.subjects = SubjectTracker()
        self.vectors = vectors
        self.router = IntentRouter(model)
        self.registry = DEFAULT_REGISTRY if profiles is None else ProfileRegistry(profiles)
        self.skills = skills if skills is not None else Skills(settings.skill_directory, self.registry)
        self.tools = ToolExecutor(timeout=settings.tool_timeout, monitor=self.monitor)
        self.monitor.circuits = self.tools.circuit_snapshot
        for name,spec in self.tools.registry.specs.items(): self.monitor.register_tool(name, spec.dependency)
        if not 1 <= settings.tool_call_limit <= 100:
            raise ValueError("invalid_tool_budget")
        if not 1 <= settings.agent_concurrency <= 4 or not math.isfinite(settings.collaboration_timeout) or not 0 < settings.collaboration_timeout <= 600:
            raise ValueError("invalid_collaboration_budget")
        self.planner = RoutingPlanner(self.registry, self.tools.registry, settings.supporting_agent_limit)
        self.risk_agent = RiskHandoffAgent(self.registry, self.tools)
        # Slot 2 is a standby at another vendor when one is configured: it runs only after slot 1 failed or was excluded.
        self.agents = {role:[RoleAgent(role, f"{role}.{slot}", fallback_model if slot == 2 and fallback_model else model, index,
                                       self.skills, self.tools, self.registry, standby=slot == 2 and fallback_model is not None)
                             for slot in [1,2]]
                       for role, profile in self.registry.profiles.items() if not profile.deterministic}
        for role,agents in self.agents.items():
            for agent in agents: self.monitor.register_agent(agent.instance, role)
        self.monitor.register_agent(self.risk_agent.instance, "risk_handoff", deterministic=True)
        if settings.architecture not in {"multi_agent", SINGLE_AGENT}:
            raise ValueError("invalid_architecture")
        self.single = SingleAgent(model, index, self.tools, self.registry) if settings.architecture == SINGLE_AGENT else None
        if self.single:
            self.monitor.register_agent(SINGLE_INSTANCE, SINGLE_AGENT)

    async def _execute(self, assignment, decision, question, understanding, context, risk, skill_snapshot, tool_budget):
        role = assignment.role
        instances = {agent.instance:agent for agent in self.agents.get(role, [])}
        details, context, visit_followup, query = role_input(assignment, understanding, question, context,
                                                             lambda: getattr(self.memory, "context_clock", time.time)())
        for index, choice in enumerate(assignment.instances):
            started = time.perf_counter()
            probe = None
            try:
                probe = self.monitor.claim(choice.instance)
                agent = instances[choice.instance]
                if agent.role != role or assignment.intent not in agent.profile.knowledge_domains:
                    raise ValueError("instance_outside_assignment")
                result = await asyncio.wait_for(agent.run(assignment.input_text, assignment.intent, context, question.context,
                    subject=question.subject, subject_changed=understanding.subject_changed, risk=risk,
                    skill_snapshot=skill_snapshot, tool_budget=tool_budget,
                    details=details, reference_query=query, assignment=assignment, original_request=question.text,
                    visit_followup=visit_followup),
                    min(self.settings.agent_timeout, self.monitor.policy.probe_timeout) if probe is not None else self.settings.agent_timeout)
                if result.execution.role != role or result.instance != choice.instance or result.execution.instance != choice.instance:
                    raise ValueError("result_outside_assignment")
                checked_role_output(agent.profile, {"role":role, "status":result.execution.status,
                    "answer":result.answer.model_dump(), "missing_fields":result.execution.missing_fields,
                    "payload":result.execution.payload}, {ref.key:ref for ref in result.references})
                result.execution.task_id = assignment.task_id
                assignment.selected_instance, assignment.status = agent.instance, result.execution.status
                assignment.attempts.append(ExecutionAttempt(instance=agent.instance, status="ok"))
                self.monitor.record(agent.instance, True, (time.perf_counter()-started)*1000,
                                    recovery_evidence=result.execution.status == "answered" and result.model_dependency_exercised)
                event("agent", instance=agent.instance, task_id=assignment.task_id, status="ok")
                return result
            except MonitorExcluded:
                self.monitor.reject(choice.instance)
                assignment.attempts.append(ExecutionAttempt(instance=choice.instance, status="failed", error="MonitorExcluded"))
                decision.fallbacks.append(RoutingFallback(role=role, instance=choice.instance, reason="monitor_admission_changed", action="try_next_planned_instance"))
            except asyncio.CancelledError:
                assignment.status = "cancelled"
                assignment.attempts.append(ExecutionAttempt(instance=choice.instance, status="cancelled", error="CancelledError"))
                self.monitor.record(choice.instance, False, (time.perf_counter()-started)*1000, status="cancelled", error="CancelledError")
                event("agent", instance=choice.instance, task_id=assignment.task_id, status="cancelled")
                raise
            except Exception as exc:
                status = "timeout" if isinstance(exc, TimeoutError) else "failed"
                diagnostic = {"error_code":exc.code,"error_stage":exc.stage} if isinstance(exc,AgentOutputError) else {}
                assignment.attempts.append(ExecutionAttempt(instance=choice.instance, status=status, error=type(exc).__name__,**diagnostic))
                self.monitor.record(choice.instance, False, (time.perf_counter()-started)*1000, status=status, error=type(exc).__name__)
                event("agent", instance=choice.instance, task_id=assignment.task_id, status=status, error=type(exc).__name__,**diagnostic)
                if index+1 < len(assignment.instances):
                    decision.fallbacks.append(RoutingFallback(role=role, instance=choice.instance, reason=status, action="retry_same_role"))
            finally:
                self.monitor.release(choice.instance, probe)
        assignment.status = "failed"
        if assignment.attempts and all(attempt.error == "MonitorExcluded" for attempt in assignment.attempts):
            if role not in decision.monitor_fallback: decision.monitor_fallback.append(role)
            decision.routing_reason += ";monitor_fallback=["+role+"]"
        decision.fallbacks.append(RoutingFallback(role=role, reason="instances_exhausted", action="transparent_guidance"))
        return None

    async def _dispatch(self, decision, question, understanding, context, risk, skill_snapshot, tool_budget):
        outputs, semaphore = {}, asyncio.Semaphore(self.settings.agent_concurrency)
        async def execute(assignment):
            try:
                async with semaphore:
                    result = await self._execute(assignment, decision, question, understanding, context, risk, skill_snapshot, tool_budget)
                    if result is not None:
                        outputs[assignment.role] = result
            except asyncio.CancelledError:
                assignment.status = "cancelled"
                raise
            except Exception as exc:
                assignment.status = "failed"
                decision.fallbacks.append(RoutingFallback(role=assignment.role, reason="task_setup_or_dispatch_failed", action="transparent_guidance"))
                event("routing_task_error", task_id=assignment.task_id, error=type(exc).__name__)
        try:
            await asyncio.wait_for(asyncio.gather(*(execute(assignment) for assignment in decision.assignments)), self.settings.collaboration_timeout)
        except TimeoutError:
            decision.fallbacks.append(RoutingFallback(role=decision.primary_agent, reason="collaboration_deadline", action="preserve_completed_tasks"))
            event("collaboration_timeout", completed_roles=sorted(outputs))
        return outputs

    async def chat(self, question: Question) -> Reply:
        log = RunLog()
        token = CURRENT.set(log)
        session = question.session or str(uuid.uuid4())
        resolution = self.subjects.resolve(question, session)
        question = question.model_copy(update={"subject":resolution.subject})
        scope = scope_key(question.user, session, "__unresolved__" if resolution.ambiguous else question.subject)
        try:
            tool_budget = ToolBudget(self.settings.tool_call_limit)
            skill_snapshot = self.skills.snapshot()
            event("skills_snapshot", snapshot_id=skill_snapshot.snapshot_id, common_hash=skill_snapshot.common_hash)
            risk = self.risk.screen(question.text, scope)
            event("risk", level=risk.level, reasons=risk.reasons, version=risk.rule_version)
            if risk.level == "CRITICAL":
                understanding, decision, answer, refs, agent_results = self._urgent(question, risk, skill_snapshot, tool_budget, log)
                degraded, storage = False, "skipped"
            else:
                understanding, decision, answer, refs, agent_results, degraded, storage, risk = await self._consult(
                    question, resolution, scope, risk, skill_snapshot, tool_budget, log)
            self.monitor.observe_execution(decision, log.request_id)
            event("routing_execution", primary=decision.primary_agent, status=decision.collaboration_status, monitor_fallback=decision.monitor_fallback,
                assignments=[{"task_id":assignment.task_id, "role":assignment.role, "status":assignment.status,
                              "selected_instance":assignment.selected_instance, "attempts":[attempt.model_dump() for attempt in assignment.attempts]}
                             for assignment in decision.assignments], fallbacks=[fallback.model_dump() for fallback in decision.fallbacks])
            reply = Reply(request_id=log.request_id, session=session, intent=understanding, risk=risk,
                          primary_agent=decision.primary_agent, supporting_agents=decision.supporting_agents, answer=answer, references=refs,
                          degraded=degraded, storage=storage, usage=log.usage(), elapsed_ms=round(log.elapsed,1),
                          agent_results=agent_results, subject=question.subject, routing_decision=decision)
            self.monitor.finish(log, degraded)
            return reply
        finally:
            CURRENT.reset(token)

    def _urgent(self, question, risk, skill_snapshot, tool_budget, log):
        """A critical risk is answered by the deterministic handoff role before any model, retrieval or storage."""
        document = skill_snapshot.documents["risk_handoff"]
        event("skill_binding", role="risk_handoff", version=document.version, sha256=document.sha256,
              snapshot_id=skill_snapshot.snapshot_id, deterministic=True, injected=False)
        risk_started = time.perf_counter()
        result = self.risk_agent.run(question.text, risk, self.settings.emergency_phone, question.subject, tool_budget)
        self.monitor.record(result.instance, True, (time.perf_counter()-risk_started)*1000)
        understanding = Understanding(primary=Intent.URGENT, confidence=1, subject=question.subject, turn_id=log.request_id)
        decision = self.planner.plan(question.text, understanding, self.agents, self.monitor)
        event("routing", **self.planner.trace(decision))
        self.monitor.observe_routing(decision, log.request_id)
        assignment = decision.assignments[0]
        assignment.selected_instance, assignment.status = result.instance, result.execution.status
        assignment.attempts.append(ExecutionAttempt(instance=result.instance, status="ok"))
        result.execution.task_id = assignment.task_id
        decision.collaboration_status = "single"
        return understanding, decision, result.answer, [], [result.execution]

    async def _consult(self, question, resolution, scope, risk, skill_snapshot, tool_budget, log):
        """Memory read, understanding, routing, role execution, merge and memory write for a non-critical turn."""
        state, storage, degraded = await self._read_state(scope, question.text, resolution.ambiguous)
        if not self.settings.preview and self.index.collection is None:
            degraded = True
            event("storage_fallback", service="reference_index")
        understanding, need = await self._understand(question, resolution, state, storage, scope, log)
        if understanding.urgent_quote:
            risk = self.risk.escalate(scope)
            event("risk", level=risk.level, reasons=risk.reasons, version=risk.rule_version)
            return (*self._urgent(question, risk, skill_snapshot, tool_budget, log), False, "skipped", risk)
        state = {**state, "grounded_fields":{name:fact.model_dump(mode="json") for name,fact in understanding.entities.items()}}
        state["consultation_need"] = need[0]
        if self.single:
            decision, answer, refs, agent_results, outputs = await self._single_agent(
                question, understanding, state, risk, skill_snapshot, tool_budget, log)
            degraded = degraded or decision.collaboration_status == "failed"
        else:
            decision = self.planner.plan(question.text, understanding, self.agents, self.monitor, consultation_need=need[0])
            event("routing", **self.planner.trace(decision))
            self.monitor.observe_routing(decision, log.request_id)
            execution_started = time.perf_counter()
            outputs = await self._dispatch(decision, question, understanding, state, risk, skill_snapshot, tool_budget)
            remaining = self.settings.collaboration_timeout-(time.perf_counter()-execution_started)
            answer, refs, routing_degraded = await compose(decision, outputs, understanding,
                self.model if remaining > 0 else None, min(self.settings.model_timeout, max(.001, remaining)))
            degraded = degraded or routing_degraded
            agent_results = [outputs[assignment.role].execution for assignment in decision.assignments if assignment.role in outputs]
        if decision.primary_agent in outputs:
            understanding.missing_fields = outputs[decision.primary_agent].execution.missing_fields
        if risk.level == "CAUTION":
            answer.limitations = (answer.limitations+["涉及特殊人群或用药风险，需医师或药师结合具体情况核实。"])[:6]
        try:
            if not resolution.ambiguous:
                if not self.memory.available:
                    raise ConnectionError("memory_unavailable")
                await self._remember(scope, question, answer, understanding, decision, outputs, need, state, log.request_id)
        except Exception as exc:
            degraded, storage = True, "degraded"
            event("memory_error", stage="write", error=type(exc).__name__)
        return understanding, decision, answer, refs, agent_results, degraded, storage, risk

    async def _single_agent(self, question, understanding, state, risk, skill_snapshot, tool_budget, log):
        """Comparison baseline: one agent answers the whole turn; no routing, role contracts or merge."""
        assignment = AgentAssignment(task_id="task:single", role=SINGLE_AGENT, intent=understanding.primary,
            goal="单个 Agent 完成本轮全部信息任务。", input_text=question.text,
            instances=[InstanceChoice(instance=SINGLE_INSTANCE, health=1, samples=0)])
        decision = RoutingDecision(primary_agent=SINGLE_AGENT, routing_reason="single_agent_baseline", routing_confidence=1,
                                   assignments=[assignment], collaboration_status="single")
        event("routing", primary=SINGLE_AGENT, support=[], reason=decision.routing_reason, confidence=1,
              monitor_fallback=[], confidence_kind="engineering_score", candidates=[])
        started = time.perf_counter()
        try:
            answer, refs, execution = await asyncio.wait_for(self.single.run(question, understanding, state, risk,
                skill_snapshot, tool_budget), self.settings.agent_timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            status = "timeout" if isinstance(exc, TimeoutError) else "failed"
            diagnostic = {"error_code":exc.code, "error_stage":exc.stage} if isinstance(exc, AgentOutputError) else {}
            assignment.status = "failed"
            assignment.attempts.append(ExecutionAttempt(instance=SINGLE_INSTANCE, status=status, error=type(exc).__name__, **diagnostic))
            self.monitor.record(SINGLE_INSTANCE, False, (time.perf_counter()-started)*1000, status=status, error=type(exc).__name__)
            event("agent", instance=SINGLE_INSTANCE, task_id=assignment.task_id, status=status, error=type(exc).__name__, **diagnostic)
            decision.collaboration_status = "failed"
            decision.fallbacks.append(RoutingFallback(role=SINGLE_AGENT, reason=status, action="transparent_guidance"))
            answer = Answer(text="当前未能完成可靠回答。请稍后重试，或携带问题和相关资料咨询医护人员。",
                            limitations=["未作医学判断，也未执行医院操作或人工连接。"])
            return decision, answer, [], [], {}
        assignment.selected_instance, assignment.status = SINGLE_INSTANCE, execution.status
        assignment.attempts.append(ExecutionAttempt(instance=SINGLE_INSTANCE, status="ok"))
        self.monitor.record(SINGLE_INSTANCE, True, (time.perf_counter()-started)*1000)
        event("agent", instance=SINGLE_INSTANCE, task_id=assignment.task_id, status="ok")
        return decision, answer, refs, [execution], {}

    async def _read_state(self, scope, text, ambiguous):
        """Session state for this scope; a storage failure degrades to an empty state instead of failing the turn."""
        if ambiguous:
            return {"messages":[], "summary":""}, "skipped", False
        try:
            if not self.memory.available:
                raise ConnectionError("memory_unavailable")
            state = await asyncio.wait_for(self.memory.read(scope), timeout=3)
            if state.get("summary"):
                state["episodes"] = await asyncio.wait_for(self.memory.recall(scope, text), timeout=3)
            return state, "memory_preview" if self.settings.preview else "available", False
        except Exception as exc:
            event("memory_error", stage="read", error=type(exc).__name__)
            return {"messages":[], "summary":""}, "degraded", True

    async def _understand(self, question, resolution, state, storage, scope, log):
        """Fused intent and grounded fields; an open fever food/medicine question keeps the turn on symptoms."""
        if resolution.ambiguous:
            understanding = Understanding(primary=Intent.GENERAL, confidence=1, subject=question.subject,
                subject_ambiguous=True, missing_fields=["subject"], turn_id=log.request_id)
        else:
            understanding = await self.router.understand(question.text, state, subject=question.subject, turn_id=log.request_id)
            if storage == "degraded": understanding.fallbacks.append("memory_unavailable")
        understanding.subject_resolution, understanding.subject_changed = resolution.method, resolution.changed
        need_action, previous_need, need_now = None, None, None
        if not resolution.ambiguous:
            try:
                need_now = getattr(self.memory, "context_clock", time.time)()
                need_action, previous_need = consultation_need(question.text, state, subject=question.subject,
                    subject_changed=resolution.changed, now=need_now, recent_critical=scope in self.risk.hints)
            except (TypeError, ValueError, OverflowError):
                need_action, previous_need = None, None
        if need_action:
            understanding.primary = Intent.SYMPTOMS
            understanding.confidence = understanding.scores.get(Intent.SYMPTOMS.value, 0)
            understanding.tasks = [TaskUnderstanding(intent=Intent.SYMPTOMS, goal=question.text[:300], confidence=understanding.confidence)]
            event("classification_rule", rule="bounded_consultation_focus")
        return understanding, (need_action, previous_need, need_now)

    async def _remember(self, scope, question, answer, understanding, decision, outputs, need, state, request_id):
        need_action, previous_need, need_now = need
        pending = None
        if (need_action in {"clarify", "clarify_followup"} and decision.primary_agent == "symptom_guide"
            and "symptom_guide" in outputs and outputs["symptom_guide"].execution.status == "needs_clarification"
            and "consultation_focus" in outputs["symptom_guide"].execution.missing_fields):
            # The clarification stays open so the next short answer ("吃的") continues it.
            pending = {"kind":"food_or_medicine", "subject":question.subject,
                "created_at":previous_need["created_at"] if previous_need else need_now,
                "source_turn_id":previous_need["source_turn_id"] if previous_need else request_id,
                "continuations":previous_need["continuations"]+1 if previous_need else 0}
        await asyncio.wait_for(self.memory.remember(scope, question.text, answer.text, understanding,
            pending, state.get("last_intent_turn_id")), timeout=self.settings.memory_timeout)
