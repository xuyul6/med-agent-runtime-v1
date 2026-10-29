from datetime import date
from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_serializer, model_validator


class Intent(str, Enum):
    SYMPTOMS = "symptoms"
    MEDICINE = "medicine"
    VISIT = "visit"
    REPORT = "report"
    GENERAL = "general"
    URGENT = "urgent"


class Risk(BaseModel):
    level: Literal["ROUTINE", "CAUTION", "CRITICAL"] = "ROUTINE"
    reasons: list[str] = Field(default_factory=list)
    rule_version: str = "screening-1"


class Question(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    user: str = Field(default="demo", min_length=1, max_length=100)
    session: str = Field(default="", max_length=100)
    subject: str = Field(default="self", min_length=1, max_length=40)
    context: Literal["selective", "full"] = "selective"


FieldName = Literal["medication_name", "formulation", "population", "product_name", "strength", "symptom", "onset", "change", "age_group",
                    "report_item", "value", "unit", "reference_range", "report_date", "institution", "region"]


class FieldSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn_id: str
    turn_index: int = Field(ge=0)
    subject: str
    quote: str = Field(min_length=1, max_length=600)
    origin: Literal["current_user", "history_user"]


class GroundedField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: FieldName
    value: str | None = Field(default=None, max_length=300)
    state: Literal["provided", "topic", "negated", "unknown"]
    source: FieldSource


class TaskUnderstanding(BaseModel):
    intent: Intent
    goal: str = Field(min_length=1, max_length=300)
    priority: Literal["primary", "support"] = "primary"
    confidence: float = Field(default=0, ge=0, le=1)


class Understanding(BaseModel):
    primary: Intent = Intent.GENERAL
    scores: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=0, ge=0, le=1)
    signals: dict[str, dict[str, float]] = Field(default_factory=dict)
    subject: str = "self"
    subject_resolution: str = "request"
    subject_changed: bool = False
    subject_ambiguous: bool = False
    turn_id: str = ""
    entities: dict[str, GroundedField] = Field(default_factory=dict)
    updates: list[GroundedField] = Field(default_factory=list, exclude=True)
    tasks: list[TaskUnderstanding] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    retrieval_query: str = ""
    context_used: bool = False
    fallbacks: list[str] = Field(default_factory=list)


class SourceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    organization: str = Field(min_length=1)
    document_type: Literal["public_overview", "product_label"]
    source_version: str = Field(min_length=1)
    updated_on: date
    published_on: date | None = None
    checked_on: date
    sections: list[str] = Field(min_length=1)
    use_conditions: str = Field(min_length=1)
    terms_url: HttpUrl


class MedicationEvidenceScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    region: str = Field(min_length=1)
    population: Literal["adults_18_plus"]
    label_population: str = Field(min_length=1)
    product_name: str | None = None
    formulation: str | None = None
    strength: str | None = None


class VisitEvidenceScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    purpose: Literal["general_preparation", "institution_specific"]


class Reference(BaseModel):
    key: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    domain: Intent
    text: str = Field(min_length=1, max_length=20000)
    url: HttpUrl
    version: str = Field(min_length=1, max_length=100)
    scope: str = Field(min_length=1, max_length=300)
    approved: bool = False
    entities: list[str] = Field(default_factory=list, max_length=20)
    provenance: SourceProvenance | None = None
    medication_scope: MedicationEvidenceScope | None = None
    visit_scope: VisitEvidenceScope | None = None

    @model_serializer(mode="wrap")
    def serialize_reference(self, handler):
        data = handler(self)
        # Historical records without a purpose keep their previous snapshot representation.
        if self.visit_scope is None:
            data.pop("visit_scope", None)
        return data

    @model_validator(mode="after")
    def validate_medication_evidence(self):
        if self.visit_scope is not None and self.domain != Intent.VISIT:
            raise ValueError("visit_scope_requires_visit_domain")
        if self.medication_scope is not None:
            if self.domain != Intent.MEDICINE or self.provenance is None or not self.entities:
                raise ValueError("medication_evidence_requires_provenance_and_entities")
            if self.provenance.document_type == "product_label" and not all(
                getattr(self.medication_scope, name) for name in ("product_name", "formulation", "strength")
            ):
                raise ValueError("product_label_requires_product_identity")
        if self.provenance and self.provenance.document_type == "product_label" and not self.medication_scope:
            raise ValueError("product_label_requires_medication_scope")
        return self


class Answer(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    questions: list[str] = Field(default_factory=list, max_length=6)
    next_steps: list[str] = Field(default_factory=list, max_length=6)
    limitations: list[str] = Field(default_factory=list, max_length=6)
    reference_keys: list[str] = Field(default_factory=list, max_length=8)


class AgentExecution(BaseModel):
    role: str
    instance: str
    profile_version: str
    status: Literal["answered", "needs_clarification", "handoff_required"]
    workflow: list[str]
    missing_fields: list[str]
    handoff_reason: str | None = None
    handoff_target: str | None = None
    payload: dict
    task_id: str = ""


class RoutingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InstanceChoice(RoutingRecord):
    instance: str
    health: float = Field(ge=0, le=1)
    samples: int = Field(ge=0)
    monitor_penalty: float = Field(default=0, ge=0, le=1)
    monitor_probe: bool = False


class RoutingCandidate(RoutingRecord):
    role: str
    intent: Intent
    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    components: dict[str, float] = Field(default_factory=dict)
    score: float = Field(default=0, ge=0, le=1)
    instances: list[InstanceChoice] = Field(default_factory=list)


class ExecutionAttempt(RoutingRecord):
    instance: str
    status: Literal["ok", "failed", "timeout", "cancelled"]
    error: str | None = None
    error_code: str | None = None
    error_stage: str | None = None


class AgentAssignment(RoutingRecord):
    task_id: str
    role: str
    intent: Intent
    goal: str
    input_text: str = Field(exclude=True)
    context_fields: list[str] = Field(default_factory=list)
    instances: list[InstanceChoice] = Field(default_factory=list)
    selected_instance: str | None = None
    attempts: list[ExecutionAttempt] = Field(default_factory=list)
    status: Literal["planned", "answered", "needs_clarification", "handoff_required", "failed", "cancelled"] = "planned"


class RoutingFallback(RoutingRecord):
    role: str
    reason: str
    action: str
    instance: str | None = None


class RoutingDecision(RoutingRecord):
    version: str = "routing-3"
    primary_agent: str
    supporting_agents: list[str] = Field(default_factory=list)
    routing_reason: str
    routing_confidence: float = Field(ge=0, le=1)
    confidence_kind: Literal["engineering_score"] = "engineering_score"
    candidates: list[RoutingCandidate] = Field(default_factory=list)
    assignments: list[AgentAssignment] = Field(default_factory=list)
    deferred_roles: list[str] = Field(default_factory=list)
    fallbacks: list[RoutingFallback] = Field(default_factory=list)
    conflicts: list[dict[str, str]] = Field(default_factory=list)
    collaboration_status: Literal["single", "planned", "merged", "partial", "conflict", "failed"] = "planned"
    monitor_fallback: list[str] = Field(default_factory=list)


class Reply(BaseModel):
    request_id: str
    session: str
    intent: Understanding
    risk: Risk
    primary_agent: str
    supporting_agents: list[str] = Field(default_factory=list)
    answer: Answer
    references: list[Reference] = Field(default_factory=list)
    degraded: bool = False
    storage: Literal["available", "degraded", "skipped", "memory_preview"]
    usage: dict
    elapsed_ms: float
    handoff_connected: bool = False
    agent_results: list[AgentExecution] = Field(default_factory=list)
    subject: str = "self"
    routing_decision: RoutingDecision | None = None


class DocumentBatch(BaseModel):
    documents: list[Reference] = Field(min_length=1, max_length=50)
