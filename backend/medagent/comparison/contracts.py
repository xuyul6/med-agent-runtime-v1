"""Preregistered inputs for the multi-agent / primary-only / single-agent comparison."""
import json
from typing import Literal
from pydantic import Field, field_validator, model_validator
from ..benchmark.contracts import BenchmarkLimits
from ..core.config import ROOT
from ..core.contracts import Intent
from ..evaluation.contracts import DialogTurn, EvalContract
from ..evaluation.judging import JUDGE_PROTOCOL_VERSION
from ..evaluation.metrics import METRIC_VERSION

ARMS = ("multi_agent", "primary_only", "single_agent")
# Each arm differs from multi_agent only in these runtime settings.
ARM_SETTINGS = {"multi_agent":{}, "primary_only":{"supporting_agent_limit":0}, "single_agent":{"architecture":"single_agent"}}
CATEGORIES = ("single_task", "compound", "multi_turn", "general", "urgent")
# These name what an orchestrated run must produce; they cannot be satisfied or compared by the single agent.
ARCHITECTURE_SPECIFIC = ("role", "status", "missing_fields", "supporting_agents")
DATASET = ROOT/"resources"/"comparison-cases-v1.json"


class ComparisonCase(EvalContract):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    category: Literal["single_task", "compound", "multi_turn", "general", "urgent"]
    preparation: list[DialogTurn] = Field(default_factory=list, max_length=6)
    target: DialogTurn
    subject: str = Field(default="self", min_length=1, max_length=40)
    # Domains whose approved sources the final answer has to cite; checked identically for every arm.
    reference_domains: list[Intent] = Field(default_factory=list, max_length=3)

    @field_validator("preparation", mode="before")
    @classmethod
    def string_turns(cls, value):
        return [{"message":item} if isinstance(item, str) else item for item in value] if isinstance(value, list) else value

    @model_validator(mode="after")
    def neutral_expectations(self):
        expected = self.target.expected
        if any(getattr(expected, name) for name in ARCHITECTURE_SPECIFIC):
            raise ValueError("architecture_specific_expectation")
        if set(self.reference_domains) - {Intent.MEDICINE, Intent.REPORT, Intent.VISIT, Intent.SYMPTOMS}:
            raise ValueError("unsupported_reference_domain")
        return self


class ComparisonRequest(EvalContract):
    cases: list[ComparisonCase] | None = Field(default=None, min_length=1, max_length=30)
    repetitions: int = Field(default=3, ge=3, le=9, strict=True)
    seed: int = Field(default=20261030, ge=0, le=2147483647, strict=True)
    judge: bool = Field(default=True, strict=True)
    limits: BenchmarkLimits = Field(default_factory=BenchmarkLimits)

    @field_validator("cases", mode="before")
    @classmethod
    def reject_explicit_null(cls, value):
        if value is None: raise ValueError("omit_cases_for_defaults")
        return value

    @model_validator(mode="after")
    def validate_design(self):
        if self.repetitions % len(ARMS): raise ValueError("repetitions_must_rotate_every_arm_first")
        if self.cases is not None and len({case.id for case in self.cases}) != len(self.cases): raise ValueError("duplicate_case_id")
        return self


PROTOCOL = {
    "version":"architecture-comparison-1",
    "behavior_check_version":METRIC_VERSION,
    "judge_protocol_version":JUDGE_PROTOCOL_VERSION,
    "arms":{"multi_agent":"routing_with_up_to_two_supporting_roles_and_merge",
            "primary_only":"same_routing_but_supporting_agent_limit_0",
            "single_agent":"one_prompt_with_all_role_rules; same_risk_gate_memory_understanding_retrieval_filters_tool_budget_and_answer_checks; no_routing_role_contracts_or_merge"},
    "shared":"same_model_skills_references_snapshot_and_output_cap; preparation_turns_run_once_under_multi_agent_then_frozen",
    "order":"seeded_rotation_so_each_arm_runs_first_equally_often_per_case",
    "judge_order":"reverse_target_order_after_all_arms; each_judge_sees_only_own_answer",
    "cache":"cold_only",
    "checks":"architecture_neutral_rules_and_safety; reference_domain_coverage_from_cited_keys; judge_quality",
    "metrics":"rule_pass_rate; coverage_rate; judge_dimensions; latency_p50_p95; input_output_tokens; model_calls; by_category",
    "minimum_cases_per_category":3,
    "missing":"retain_all_planned_arms; any_missing_failure_or_unknown_usage_blocks_claims",
    "stopping":"budget_deadline_safety_failure_or_infrastructure_failure_limit",
    "inference":"descriptive_engineering_comparison_on_synthetic_cases_only; no_statistical_superiority_or_clinical_generalization",
}


def resolve_comparison(request: ComparisonRequest):
    if request.cases is not None: return "custom", request.cases
    data = json.loads(DATASET.read_text())
    return data["version"], [ComparisonCase.model_validate(row) for row in data["cases"]]
