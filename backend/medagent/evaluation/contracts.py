"""Bounded, explicit evaluation input; omitted cases and empty arrays differ."""
import math
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, ValidationInfo
from ..agents.contracts import FIELD_QUESTIONS
from ..core.contracts import FieldName, Intent


Role = Literal["general", "symptom_guide", "drug_reference", "report_explainer", "visit_guide", "risk_handoff"]
NonUrgent = Literal[Intent.GENERAL, Intent.SYMPTOMS, Intent.MEDICINE, Intent.REPORT, Intent.VISIT]
RiskLevel = Literal["ROUTINE", "CAUTION", "CRITICAL"]


class EvalContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntentCase(EvalContract):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    message: str = Field(min_length=1, max_length=6000)
    expected: Intent
    classifier_expected: NonUrgent | None = None
    expected_risk: RiskLevel | None = None

    @model_validator(mode="after")
    def classifier_label(self):
        if self.expected == Intent.URGENT and self.expected_risk not in {None,"CRITICAL"} or self.expected != Intent.URGENT and self.expected_risk == "CRITICAL":
            raise ValueError("inconsistent_risk_gated_label")
        if self.expected == Intent.URGENT and self.classifier_expected is not None:
            raise ValueError("urgent_gate_bypasses_classifier")
        if self.expected != Intent.URGENT and self.classifier_expected is None:
            self.classifier_expected = self.expected
        return self


class ExpectedBehavior(EvalContract):
    role: Role | None = None
    intent: Intent | None = None
    risk: RiskLevel | None = None
    status: Literal["answered", "needs_clarification", "handoff_required"] | None = None
    references: bool | None = None
    degraded: bool = False
    subject: str | None = Field(default=None, max_length=40)
    entities: dict[FieldName, Annotated[str, Field(max_length=300)] | None] = Field(default_factory=dict)
    entity_states: dict[FieldName, Literal["provided", "topic", "unknown", "negated"]] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list, max_length=6)
    supporting_agents: list[Role] = Field(default_factory=list, max_length=3)

    @field_validator("missing_fields")
    @classmethod
    def known_missing_fields(cls, value):
        if set(value)-set(FIELD_QUESTIONS): raise ValueError("unknown_missing_field")
        return value


class DialogTurn(EvalContract):
    message: str = Field(min_length=1, max_length=6000)
    subject: str | None = Field(default=None, min_length=1, max_length=40)
    expected: ExpectedBehavior = Field(default_factory=ExpectedBehavior)


class FaultSpec(EvalContract):
    kind: Literal["memory_unavailable", "index_unavailable", "tool_timeout", "tool_invalid_result",
                  "agent_invalid_output", "synthesis_failure", "summary_failure", "judge_failure", "language_unavailable"]
    turn: int | None = Field(default=None, ge=1, le=8)


class ReferenceSnapshot(EvalContract):
    keys: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(max_length=100)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("keys")
    @classmethod
    def unique_keys(cls, value):
        if len(set(value)) != len(value): raise ValueError("duplicate_reference_key")
        return sorted(value)


class DialogCase(EvalContract):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    turns: list[DialogTurn] = Field(min_length=1, max_length=8)
    faults: list[FaultSpec] = Field(default_factory=list, max_length=2)
    reference_snapshot: ReferenceSnapshot | None = None

    @field_validator("turns", mode="before")
    @classmethod
    def string_turns(cls, value):
        return [{"message":turn} if isinstance(turn,str) else turn for turn in value] if isinstance(value,list) else value

    @model_validator(mode="after")
    def fault_turn_exists(self):
        if any(fault.turn and fault.turn > len(self.turns) for fault in self.faults):
            raise ValueError("fault_turn_outside_dialogue")
        return self


class EvalLimits(EvalContract):
    timeout_seconds: float = Field(default=300, gt=0, le=600)
    max_model_calls: int = Field(default=200, ge=1, le=400, strict=True)
    max_output_tokens: int = Field(default=240000, ge=1, le=400000, strict=True)
    per_call_output_tokens: int = Field(default=2500, ge=1, le=3000, strict=True)
    concurrency: Literal[1] = 1


class BaselineOptions(EvalContract):
    compare_to: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
    save_as: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


class EvaluationRequest(EvalContract):
    dataset_version: str | None = Field(default=None, min_length=1, max_length=80)
    intent_cases: list[IntentCase] | None = Field(default=None, max_length=50)
    dialog_cases: list[DialogCase] | None = Field(default=None, max_length=50)
    judge: bool = Field(default=True, strict=True)
    limits: EvalLimits = Field(default_factory=EvalLimits)
    baseline: BaselineOptions = Field(default_factory=BaselineOptions)

    @field_validator("intent_cases", "dialog_cases", mode="before")
    @classmethod
    def explicit_null_rejected(cls, value, info: ValidationInfo):
        if value is None: raise ValueError("omit_cases_for_defaults_or_use_empty_array")
        if not isinstance(value,list): return value
        prefix = "intent" if info.field_name == "intent_cases" else "dialog"
        return [{"id":f"{prefix}-{index+1}",**row} if isinstance(row,dict) else row for index,row in enumerate(value)]

    @model_validator(mode="after")
    def size_limits(self):
        cases = [*(self.intent_cases or []), *(self.dialog_cases or [])]
        if len(cases) > 50 or len({case.id for case in cases}) != len(cases):
            raise ValueError("too_many_or_duplicate_cases")
        turns = len(self.intent_cases or [])+sum(len(case.turns) for case in self.dialog_cases or [])
        if turns > 120: raise ValueError("too_many_total_turns")
        return self


class JudgeScore(EvalContract):
    relevance: float
    accuracy: float
    completeness: float
    helpfulness: float
    reason: str = Field(min_length=1, max_length=2000)

    @field_validator("relevance", "accuracy", "completeness", "helpfulness", mode="before")
    @classmethod
    def valid_number(cls, value):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid_judge_score")
        return value

    def result(self):
        result = self.model_dump()
        result["mean"] = sum(result[key] for key in ["relevance", "accuracy", "completeness", "helpfulness"])/4
        return result
