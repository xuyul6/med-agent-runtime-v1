"""Registered tool dispatch, permission checks, budgets, timeouts and circuits."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
import inspect
import re
import time
from types import MappingProxyType
from typing import Callable, Literal, TYPE_CHECKING
from pydantic import BaseModel, ValidationError
from ..agents.contracts import AgentInput
from ..core.contracts import Intent, Reference
from ..understanding.conversation import institution_request
from ..core.observability import event
from ..knowledge.medication_evidence import rejection_reason
from ..knowledge.retrieval import ReferenceIndex, has_query_overlap, visit_scope_rejection
from ..understanding.medications import canonical
from ..understanding.conversation import has_visit_context
from .contracts import (
    BookingSearchResult, ChecklistResult, ContextResult, DraftResult, EmptyArguments, FieldCheck, HandoffSummary, LookupArguments,
    LookupResult, MedicationScopeResult, ProposeAppointmentArguments, ProposeCancellationArguments, ReferenceScope,
    ReferenceSelection, SlotSearchResult, SlotView,
)
if TYPE_CHECKING:
    from ..agents.profiles import AgentProfile


class ToolInputError(ValueError):
    pass


class ToolResultError(ValueError):
    pass


class ToolBudgetExceeded(RuntimeError):
    pass


class UnknownToolError(KeyError):
    pass


class CircuitUnavailable(RuntimeError):
    pass


@dataclass
class ToolBudget:
    limit: int = 12
    used: int = 0

    def __post_init__(self):
        if not 1 <= self.limit <= 100:
            raise ValueError("invalid_tool_budget")

    def consume(self) -> None:
        if self.used >= self.limit:
            raise ToolBudgetExceeded("tool_call_budget_exceeded")
        self.used += 1


@dataclass
class ToolContext:
    profile: AgentProfile
    request: AgentInput
    index: object | None = None
    references: dict[str, Reference] = field(default_factory=dict)
    scope_filtered_count: int = 0
    evidence_query: str | None = None
    # Appointment tools only: the caller, the store, the resolved request and what this turn queried and drafted.
    user: str = ""
    session: str = ""
    appointments: object | None = None
    resolution: object | None = None
    slots: dict = field(default_factory=dict)
    bookings: dict = field(default_factory=dict)
    issued: dict = field(default_factory=dict)
    # One list of slot ids per requested slot; a draft must be the only slot of its group.
    slot_groups: list = field(default_factory=list)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_contract: type[BaseModel]
    output_contract: type[BaseModel]
    allowed_roles: frozenset[str]
    domains: frozenset[Intent]
    handler: Callable
    timeout: float = 20
    side_effect: Literal["compute", "read_external", "propose_write"] = "compute"
    dependency: bool = False
    stage: Literal["preflight", "retrieval", "context", "handoff", "action"] = "context"

    def definition(self) -> dict:
        return {"name":self.name, "description":self.description, "input_schema":self.input_contract.model_json_schema()}

    def describe(self) -> dict:
        return {**self.definition(), "output_schema":self.output_contract.model_json_schema(),
                "allowed_roles":sorted(self.allowed_roles), "domains":sorted(domain.value for domain in self.domains),
                "timeout":self.timeout, "side_effect":self.side_effect, "dependency":self.dependency, "stage":self.stage}


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]):
        registered = {}
        for spec in specs:
            if spec.name in registered or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", spec.name):
                raise ValueError("invalid_or_duplicate_tool:"+spec.name)
            if not spec.allowed_roles or not spec.domains or spec.timeout <= 0 or spec.side_effect not in {"compute", "read_external", "propose_write"}:
                raise ValueError("invalid_tool_spec:"+spec.name)
            if not all(isinstance(contract, type) and issubclass(contract, BaseModel)
                       for contract in [spec.input_contract, spec.output_contract]) or not callable(spec.handler):
                raise ValueError("invalid_tool_contract:"+spec.name)
            if spec.dependency != (spec.side_effect == "read_external") or spec.stage not in {"preflight", "retrieval", "context", "handoff", "action"}:
                raise ValueError("invalid_tool_execution_policy:"+spec.name)
            if spec.dependency and not inspect.iscoroutinefunction(spec.handler):
                raise ValueError("dependency_handler_must_be_async:"+spec.name)
            registered[spec.name] = spec
        self.specs = MappingProxyType(registered)

    def scope(self, role: str) -> tuple[str, ...]:
        return tuple(name for name, spec in self.specs.items() if role in spec.allowed_roles)

    def authorize(self, name: str, context: ToolContext) -> ToolSpec:
        spec = self.specs.get(name)
        if spec is None:
            raise UnknownToolError("unknown_tool")
        profile = context.profile
        if name not in profile.tool_scope or profile.role not in spec.allowed_roles:
            raise PermissionError("tool_not_allowed")
        domains = profile.knowledge_domains if not profile.deterministic else {Intent.URGENT}
        if context.request.intent not in spec.domains or context.request.intent not in domains:
            raise PermissionError("tool_domain_not_allowed")
        return spec

    def definitions(self, profile: AgentProfile) -> list[dict]:
        definitions = []
        for name in profile.tool_scope:
            spec = self.specs[name]
            if profile.role not in spec.allowed_roles:
                raise ValueError("profile_tool_not_allowed")
            definitions.append(spec.definition())
        return definitions


def check_fields(context: ToolContext, arguments: EmptyArguments) -> dict:
    request = context.request
    fields = [name for name in request.model_fields_set if name not in AgentInput.model_fields and getattr(request, name) is not None]
    return {"provided_fields":sorted(fields), "missing_fields":request.information_gaps(),
            "note":"只核对明确提供的结构化字段；未提取字段不等于原文没有说明，不补造事实。"}


def inspect_context(context: ToolContext, arguments: EmptyArguments) -> dict:
    checked = check_fields(context, arguments)
    return {"intent":context.request.intent, "subject":context.request.subject, "risk_level":context.request.risk.level,
            "provided_fields":checked["provided_fields"], "missing_fields":checked["missing_fields"]}


async def lookup_reference(context: ToolContext, arguments: LookupArguments) -> dict:
    if arguments.domain not in context.profile.knowledge_domains or arguments.domain != context.request.intent:
        raise PermissionError("knowledge_domain_not_allowed")
    query = arguments.query
    target = canonical(getattr(context.request, "medication_name", None))
    if arguments.domain == Intent.MEDICINE and target and target.casefold() not in query.casefold():
        query = target+" "+query
    if isinstance(context.index, ReferenceIndex):
        if arguments.domain == Intent.MEDICINE:
            found = await context.index.search(query, arguments.domain.value, medication_request=context.request)
        else:
            found = await context.index.search(query, arguments.domain.value,
                evidence_query=context.evidence_query if context.evidence_query is not None else context.request.text,
                visit_context=getattr(context.request, "visit_context", None))
    else:
        found = await context.index.search(query, arguments.domain.value)
    if arguments.domain == Intent.MEDICINE:
        matched = []
        for ref in found:
            # Malformed results still reach strict output validation, never the model.
            if not isinstance(ref, Reference):
                matched.append(ref)
                continue
            reason = rejection_reason(ref, context.request)
            if reason:
                event("medication_evidence_filtered", key=ref.key, reason=reason)
            else:
                matched.append(ref)
        context.scope_filtered_count += len(found)-len(matched)
        found = matched
    return {"documents":found}


def medication_scope(context: ToolContext, arguments: ReferenceSelection) -> dict:
    keys = arguments.keys or list(context.references)
    if set(keys)-set(context.references):
        raise ToolInputError("references_not_returned_to_this_role")
    sources = [ReferenceScope(key=ref.key, version=ref.version, scope=ref.scope, entities=ref.entities,
                             provenance=ref.provenance, medication_scope=ref.medication_scope)
               for key in keys for ref in [context.references[key]]]
    return {"medication_name":getattr(context.request, "medication_name", None), "sources":sources,
            "applicability":"needs_verification" if sources else "no_source",
            "note":"来源范围仅复用本次资料元数据；不自动确认剂型、人群或联合用药适用性。"}


def visit_checklist(context: ToolContext, arguments: EmptyArguments) -> dict:
    template = context.references.get("consultation-record")
    if template is None:
        return {"items":[], "reference_keys":[], "missing_fields":["source"], "needs_verification":[]}
    return {"items":["整理已提供的症状及时间变化。", "如有，整理现用药品与已知过敏情况。",
                     "如有，带上可自行查看的既往资料并列出待问问题。"],
            "reference_keys":[template.key], "missing_fields":[],
            "needs_verification":["机构具体证件、费用、号源和入口需在官方渠道核实。"]}


def handoff_summary(context: ToolContext, arguments: EmptyArguments) -> dict:
    return {"known_facts":["风险规则命中："+code for code in context.request.risk.reasons], "handoff_connected":False}


def slot_view(slot, remaining: int) -> dict:
    return SlotView(slot_id=slot.slot_id, hospital=slot.hospital, department=slot.department, day=slot.day.isoformat(),
                    window=slot.window, remaining=remaining).model_dump()


def search_slots(context: ToolContext, arguments: EmptyArguments) -> dict:
    resolution, store = context.resolution, context.appointments
    if resolution is None or store is None or not (resolution.hospital_id and resolution.department_id and resolution.day):
        context.slots = {}
        return {"slots":[]}
    found, groups = [], []
    for day, window in resolution.items or ((resolution.day, resolution.window),):
        part = store.search(resolution.hospital_id, resolution.department_id, day, window)
        found += part
        groups.append([slot.slot_id for slot, _ in part])
    context.slots, context.slot_groups = {slot.slot_id:slot for slot, _ in found}, groups
    return {"slots":[slot_view(slot, remaining) for slot, remaining in found]}


def find_bookings(context: ToolContext, arguments: EmptyArguments) -> dict:
    from ..appointments.resolution import booking_view
    resolution, store = context.resolution, context.appointments
    if resolution is None or store is None:
        context.bookings = {}
        return {"bookings":[]}
    found = [booking for booking in store.bookings(context.user, context.request.subject)
             if resolution.hospital_id in (None, booking.slot.hospital_id) and resolution.department_id in (None, booking.slot.department_id)
             and resolution.day in (None, booking.slot.day) and resolution.window in (None, booking.slot.window)
             # Same rule as booking: today's and past appointments cannot be cancelled here.
             and store.catalog.bookable(booking.slot.day, store.today())]
    context.bookings = {booking.appointment_id:booking for booking in found}
    return {"bookings":[booking_view(booking).model_dump() for booking in found]}


def _propose(context: ToolContext, action: str, candidate: str, candidates: dict, propose) -> dict:
    from ..appointments.store import AppointmentError
    if context.request.action != action:
        raise ToolInputError("action_mismatch")
    if candidate not in candidates:
        raise ToolInputError("slot_not_from_this_search" if action == "create" else "booking_not_from_this_search")
    groups = (context.slot_groups or [list(candidates)]) if action == "create" else [list(candidates)]
    if len(next(group for group in groups if candidate in group)) != 1:
        raise ToolInputError("candidate_not_unique")
    if len(context.issued) >= len(groups):
        raise ToolInputError("one_draft_per_turn" if len(groups) == 1 else "draft_limit_per_turn")
    if candidate in {draft.slot_id for draft, _ in context.issued.values()}:
        raise ToolInputError("slot_already_drafted")
    try:
        draft, token = propose(user=context.user, session=context.session, subject=context.request.subject)
    except AppointmentError as error:
        raise ToolInputError(error.code) from error
    context.issued[draft.draft_id] = (draft, token)
    return {"draft_id":draft.draft_id, "kind":draft.kind, "summary":draft.summary, "needs_user_confirmation":True}


def propose_appointment(context: ToolContext, arguments: ProposeAppointmentArguments) -> dict:
    return _propose(context, "create", arguments.slot_id, context.slots,
                    lambda **who: context.appointments.propose_create(**who, slot_id=arguments.slot_id))


def propose_cancellation(context: ToolContext, arguments: ProposeCancellationArguments) -> dict:
    return _propose(context, "cancel", arguments.appointment_id, context.bookings,
                    lambda **who: context.appointments.propose_cancel(**who, appointment_id=arguments.appointment_id))


SPECIALISTS = frozenset({"symptom_guide", "drug_reference", "report_explainer", "visit_guide"})
DEFAULT_TOOLS = ToolRegistry([
    ToolSpec("inspect_request_context", "核对当前已验证的咨询对象和输入字段，不读取其他会话。", EmptyArguments, ContextResult,
             frozenset({"general"}), frozenset({Intent.GENERAL}), inspect_context, timeout=1, stage="preflight"),
    *[ToolSpec(name, "检查本角色已提供的结构化字段与必要缺口，不解析或补造原文事实。", EmptyArguments, FieldCheck,
               frozenset({role}), frozenset({intent}), check_fields, timeout=1, stage="preflight")
      for name, role, intent in [("check_symptom_fields", "symptom_guide", Intent.SYMPTOMS),
                                 ("check_medication_fields", "drug_reference", Intent.MEDICINE),
                                 ("check_report_fields", "report_explainer", Intent.REPORT),
                                 ("check_visit_fields", "visit_guide", Intent.VISIT),
                                 ("check_appointment_fields", "appointment", Intent.APPOINTMENT)]],
    ToolSpec("lookup_reference", "查找当前角色允许的公开参考资料，不访问医院记录。", LookupArguments, LookupResult,
             SPECIALISTS, frozenset({Intent.SYMPTOMS, Intent.MEDICINE, Intent.REPORT, Intent.VISIT}), lookup_reference,
             side_effect="read_external", dependency=True, stage="retrieval"),
    ToolSpec("check_medication_scope", "核对本角色已取得来源的范围与未知项，不保证个人适用性。", ReferenceSelection, MedicationScopeResult,
             frozenset({"drug_reference"}), frozenset({Intent.MEDICINE}), medication_scope, timeout=1),
    ToolSpec("build_visit_checklist", "根据本次已取得的项目模板生成通用准备清单，不创建预约。", EmptyArguments, ChecklistResult,
             frozenset({"visit_guide"}), frozenset({Intent.VISIT}), visit_checklist, timeout=1),
    ToolSpec("search_slots", "按代码已解析的科室、日期和时段查询模拟医院号源，模型不能传查询条件。", EmptyArguments, SlotSearchResult,
             frozenset({"appointment"}), frozenset({Intent.APPOINTMENT}), search_slots, timeout=1, stage="retrieval"),
    ToolSpec("find_bookings", "查询当前用户、当前咨询对象在模拟医院的已有预约，按已解析字段过滤。", EmptyArguments, BookingSearchResult,
             frozenset({"appointment"}), frozenset({Intent.APPOINTMENT}), find_bookings, timeout=1, stage="retrieval"),
    ToolSpec("propose_appointment", "为本次查询得到的唯一号源生成待确认的预约草稿；不会写入，需用户点击确认。", ProposeAppointmentArguments, DraftResult,
             frozenset({"appointment"}), frozenset({Intent.APPOINTMENT}), propose_appointment, timeout=1, side_effect="propose_write", stage="action"),
    ToolSpec("propose_cancellation", "为本次查询得到的唯一已有预约生成待确认的取消草稿；不会写入，需用户点击确认。", ProposeCancellationArguments, DraftResult,
             frozenset({"appointment"}), frozenset({Intent.APPOINTMENT}), propose_cancellation, timeout=1, side_effect="propose_write", stage="action"),
    ToolSpec("create_handoff_summary", "同步整理已有风险规则标识，明确未连接人工，不调用外部依赖。", EmptyArguments, HandoffSummary,
             frozenset({"risk_handoff"}), frozenset({Intent.URGENT}), handoff_summary, timeout=1, stage="handoff"),
])


class ToolExecutor:
    def __init__(self, registry: ToolRegistry = DEFAULT_TOOLS, timeout: float = 20, cooldown: float = 15,
                 clock: Callable = time.monotonic, monitor=None):
        if timeout <= 0 or cooldown <= 0:
            raise ValueError("invalid_tool_timeout_or_cooldown")
        self.registry, self.timeout, self.cooldown, self.clock = registry, timeout, cooldown, clock
        self.failures, self.open_until, self.probes, self.generations = {}, {}, {}, {}
        self.monitor = monitor

    def circuit_snapshot(self, name: str) -> dict:
        """One tool's state, taking the worst of its per-domain circuits."""
        keys = {key for state in (self.failures, self.open_until, self.probes) for key in state
                if key == name or key.startswith(name+":")} or {name}
        until = max(self.open_until.get(key, 0) for key in keys)
        return {"circuit_state":"open" if until > self.clock() else "half_open" if until else "closed",
                "consecutive_fails":max(self.failures.get(key, 0) for key in keys), "probe_inflight":any(key in self.probes for key in keys),
                "cooldown_remaining_seconds":max(0, until-self.clock())}

    def _prepare(self, name: str, arguments: dict, context: ToolContext, budget: ToolBudget):
        budget.consume()
        spec = self.registry.authorize(name, context)
        try:
            request = spec.input_contract.model_validate(arguments)
        except ValidationError as exc:
            raise ToolInputError("invalid_tool_arguments") from exc
        if isinstance(request, LookupArguments) and request.domain != context.request.intent:
            raise PermissionError("knowledge_domain_not_allowed")
        return spec, request

    def _validate(self, spec: ToolSpec, value, context: ToolContext) -> BaseModel:
        try:
            result = spec.output_contract.model_validate(value, strict=True)
        except ValidationError as exc:
            raise ToolResultError("invalid_tool_result") from exc
        if isinstance(result, LookupResult):
            for ref in result.documents:
                if not ref.approved or ref.domain not in context.profile.knowledge_domains or ref.domain != context.request.intent:
                    raise ToolResultError("reference_outside_tool_scope")
                if ref.domain == Intent.MEDICINE:
                    if rejection_reason(ref, context.request):
                        raise ToolResultError("reference_outside_medication_scope")
                else:
                    evidence = context.evidence_query if context.evidence_query is not None else context.request.text
                    if ref.domain == Intent.VISIT and institution_request(evidence):
                        raise ToolResultError("institution_source_not_supported")
                    if reason := visit_scope_rejection(ref):
                        raise ToolResultError(reason)
                    judged = getattr(context.request, "visit_context", None)
                    if ref.domain == Intent.VISIT and not has_visit_context(evidence, judged):
                        raise ToolResultError("visit_context_missing")
                    if not has_query_overlap(evidence, ref, judged):
                        raise ToolResultError("reference_without_query_overlap")
            context.references.update({ref.key:ref for ref in result.documents})
        return result

    def _record(self, name: str, status: str, started: float, context: ToolContext, budget: ToolBudget, error=None,
                *, invoked=False, filtered=0) -> None:
        elapsed = max(0, (self.clock()-started)*1000)
        if self.monitor:
            self.monitor.record_tool(name, status, elapsed, invoked=invoked,
                quality=str(error) if isinstance(error, ToolResultError) else None, filtered=filtered)
        event("tool", name=name, role=context.profile.role, status=status,
              handler_invoked=invoked,
              elapsed_ms=round(elapsed, 1), budget_used=budget.used, budget_limit=budget.limit,
              error=type(error).__name__ if error else None)

    def _error_status(self, error: BaseException) -> str:
        if isinstance(error, ToolBudgetExceeded): return "budget_exhausted"
        if isinstance(error, UnknownToolError): return "unknown_tool"
        if isinstance(error, PermissionError): return "denied"
        if isinstance(error, ToolInputError): return "invalid_request"
        if isinstance(error, ToolResultError): return "invalid_result"
        if isinstance(error, CircuitUnavailable): return "circuit_open"
        if isinstance(error, TimeoutError): return "timeout"
        if isinstance(error, asyncio.CancelledError): return "cancelled"
        return "failed"

    async def call(self, name: str, arguments: dict, context: ToolContext, budget: ToolBudget) -> BaseModel:
        started, probe, generation, spec = self.clock(), None, None, None
        invoked, filtered_before = False, context.scope_filtered_count
        circuit = name
        try:
            spec, request = self._prepare(name, arguments, context, budget)
            # A domain's failures stay in that domain: a role retry counts twice, which must not block other roles' lookups.
            if isinstance(request, LookupArguments):
                circuit = f"{name}:{request.domain.value}"
            if spec.dependency:
                until = self.open_until.get(circuit, 0)
                if until > self.clock() or (until and circuit in self.probes):
                    raise CircuitUnavailable(name)
                if until:
                    probe = object()
                    self.probes[circuit] = probe
                generation = self.generations.get(circuit, 0)
            async def invoke():
                nonlocal invoked
                invoked = True
                value = spec.handler(context, request)
                return await value if inspect.isawaitable(value) else value
            value = await asyncio.wait_for(invoke(), timeout=min(self.timeout, spec.timeout))
            result = self._validate(spec, value, context)
            if spec.dependency and generation == self.generations.get(circuit, 0):
                self.failures[circuit] = 0
                self.open_until.pop(circuit, None)
            empty = isinstance(result, LookupResult) and not result.documents
            self._record(name, "empty_result" if empty else "ok", started, context, budget,
                         invoked=invoked, filtered=context.scope_filtered_count-filtered_before)
            return result
        except (Exception, asyncio.CancelledError) as exc:
            if spec is not None and spec.dependency and generation == self.generations.get(circuit, 0) and not isinstance(
                exc, (PermissionError, ToolInputError, ToolBudgetExceeded, CircuitUnavailable, asyncio.CancelledError)
            ):
                self.failures[circuit] = self.failures.get(circuit, 0)+1
                if self.failures[circuit] >= 2:
                    self.open_until[circuit] = self.clock()+self.cooldown
                    self.generations[circuit] = self.generations.get(circuit, 0)+1
            self._record(name, self._error_status(exc), started, context, budget, exc,
                         invoked=invoked, filtered=context.scope_filtered_count-filtered_before)
            raise
        finally:
            if probe is not None and self.probes.get(circuit) is probe:
                self.probes.pop(circuit, None)

    def call_local(self, name: str, arguments: dict, context: ToolContext, budget: ToolBudget) -> BaseModel:
        """Run only bounded local computation; deterministic risk cannot await I/O."""
        started = self.clock()
        invoked = False
        try:
            spec, request = self._prepare(name, arguments, context, budget)
            if spec.dependency or spec.side_effect != "compute" or inspect.iscoroutinefunction(spec.handler):
                raise PermissionError("local_dispatch_requires_sync_computation")
            invoked = True
            value = spec.handler(context, request)
            if inspect.isawaitable(value):
                if inspect.iscoroutine(value): value.close()
                raise ToolResultError("unexpected_async_local_result")
            if self.clock()-started > min(self.timeout, spec.timeout):
                raise TimeoutError("local_tool_timeout")
            result = self._validate(spec, value, context)
            self._record(name, "ok", started, context, budget, invoked=invoked)
            return result
        except Exception as exc:
            self._record(name, self._error_status(exc), started, context, budget, exc, invoked=invoked)
            raise
