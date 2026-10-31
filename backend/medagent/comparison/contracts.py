"""Preregistered inputs for the multi-agent / primary-only / single-agent comparison."""
import json
import re
from typing import Literal
from pydantic import Field, field_validator, model_validator
from ..benchmark.contracts import BenchmarkLimits
from ..core.config import ROOT
from ..core.contracts import Intent
from ..evaluation.contracts import DialogTurn, EvalContract, FaultSpec
from ..evaluation.judging import JUDGE_PROTOCOL_VERSION
from ..evaluation.metrics import METRIC_VERSION

ARMS = ("multi_agent", "primary_only", "single_agent")
# Each arm differs from multi_agent only in these runtime settings.
ARM_SETTINGS = {"multi_agent":{}, "primary_only":{"supporting_agent_limit":0}, "single_agent":{"architecture":"single_agent"}}
CATEGORIES = ("single_task", "compound", "multi_turn", "general", "urgent", "tool_fault",
              "daily_sourced", "daily_gap", "compound_parallel", "compound_linked",
              "stability_once", "stability_always", "stability_tool", "stability_outage", "boundary", "injection")
TOOL_FAULTS = {"tool_timeout", "tool_invalid_result"}
# A fault category takes only its own faults; only an outage has no domain, it hits every generation.
FAULT_CATEGORIES = {"tool_fault":TOOL_FAULTS, "stability_tool":TOOL_FAULTS, "stability_once":{"generation_failure_once"},
                    "stability_always":{"generation_failure_always"}, "stability_outage":{"generation_failure_always"}}
SCORED_FAULT_CATEGORIES = {"stability_once", "stability_always", "stability_tool"}
# Layers are reported apart and never averaged into one score.
LAYERS = {"daily":{"dataset":"comparison-daily-v1.json", "arms":("multi_agent", "single_agent"), "judge":True},
          "compound":{"dataset":"comparison-compound-v1.json", "arms":ARMS, "judge":True},
          "stability":{"dataset":"comparison-stability-v1.json", "arms":("multi_agent", "single_agent"), "judge":True},
          "safety":{"dataset":"comparison-safety-v1.json", "arms":("multi_agent", "single_agent"), "judge":False}}
# These name what an orchestrated run must produce; they cannot be satisfied or compared by the single agent.
ARCHITECTURE_SPECIFIC = ("role", "status", "missing_fields", "supporting_agents")
DATASET = ROOT/"resources"/"comparison-cases-v1.json"


class ComparisonCase(EvalContract):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    category: Literal[*CATEGORIES]
    preparation: list[DialogTurn] = Field(default_factory=list, max_length=6)
    target: DialogTurn
    subject: str = Field(default="self", min_length=1, max_length=40)
    # Domains whose approved sources the final answer has to cite; checked identically for every arm.
    reference_domains: list[Intent] = Field(default_factory=list, max_length=3)
    # Injected on the target turn only, identically for every arm.
    faults: list[FaultSpec] = Field(default_factory=list, max_length=1)
    # Written before the run and matched on the visible answer, so these rules score the same for every arm.
    gap_pattern: str | None = Field(default=None, min_length=1, max_length=300)
    fill_pattern: str | None = Field(default=None, min_length=1, max_length=300)
    forbidden_pattern: str | None = Field(default=None, min_length=1, max_length=300)

    @field_validator("gap_pattern", "fill_pattern", "forbidden_pattern")
    @classmethod
    def compiled_pattern(cls, value):
        if value is not None:
            try: re.compile(value)
            except re.error: raise ValueError("invalid_pattern") from None
        return value

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
        allowed = FAULT_CATEGORIES.get(self.category, set())
        if bool(self.faults) != bool(allowed):
            raise ValueError("tool_fault_category_needs_exactly_its_fault")
        if any(fault.kind not in allowed or (fault.domain is None) != (self.category == "stability_outage") or fault.turn is not None
               for fault in self.faults):
            raise ValueError("comparison_faults_are_domain_scoped_tool_faults")
        if any(fault.domain in self.reference_domains for fault in self.faults):
            raise ValueError("faulted_domain_cannot_be_required")
        if self.category in SCORED_FAULT_CATEGORIES and not (self.gap_pattern and self.fill_pattern):
            raise ValueError("stability_case_needs_gap_and_fill_patterns")
        if self.category in {"boundary", "injection"} and not self.forbidden_pattern:
            raise ValueError("safety_case_needs_forbidden_pattern")
        if self.fill_pattern and re.search(self.fill_pattern, self.target.message):
            raise ValueError("fill_pattern_matches_the_question")
        return self


class ComparisonLimits(BenchmarkLimits):
    # Three arms over a full case set run past the benchmark's deadline.
    timeout_seconds: float = Field(default=600, gt=0, le=3600)


class ComparisonRequest(EvalContract):
    layer: Literal["daily", "compound", "stability", "safety"] | None = None
    cases: list[ComparisonCase] | None = Field(default=None, min_length=1, max_length=30)
    repetitions: int = Field(default=3, ge=3, le=9, strict=True)
    seed: int = Field(default=20261030, ge=0, le=2147483647, strict=True)
    judge: bool = Field(default=True, strict=True)
    limits: ComparisonLimits = Field(default_factory=ComparisonLimits)

    @property
    def arm_names(self) -> tuple[str, ...]:
        return LAYERS[self.layer]["arms"] if self.layer else ARMS

    @field_validator("cases", mode="before")
    @classmethod
    def reject_explicit_null(cls, value):
        if value is None: raise ValueError("omit_cases_for_defaults")
        return value

    @model_validator(mode="after")
    def validate_design(self):
        if self.repetitions % len(self.arm_names): raise ValueError("repetitions_must_rotate_every_arm_first")
        if self.layer and self.judge and not LAYERS[self.layer]["judge"]: raise ValueError("safety_layer_runs_without_judge")
        if self.cases is not None and len({case.id for case in self.cases}) != len(self.cases): raise ValueError("duplicate_case_id")
        return self


PROTOCOL = {
    "version":"architecture-comparison-3",
    "behavior_check_version":METRIC_VERSION,
    "judge_protocol_version":JUDGE_PROTOCOL_VERSION,
    "arms":{"multi_agent":"routing_with_up_to_two_supporting_roles_and_merge",
            "primary_only":"same_routing_but_supporting_agent_limit_0",
            "single_agent":"one_prompt_with_all_role_rules; same_risk_gate_memory_understanding_retrieval_filters_tool_budget_and_answer_checks; no_routing_role_contracts_or_merge"},
    "shared":"same_model_skills_references_snapshot_and_output_cap; preparation_turns_run_once_under_multi_agent_then_frozen",
    "order":"seeded_rotation_so_each_arm_runs_first_equally_often_per_case",
    "judge_order":"reverse_target_order_after_all_arms; each_judge_sees_only_own_answer",
    "cache":"cold_only",
    "faults":"one_lookup_domain_fails_on_the_target_turn_for_every_arm; reference_domains_list_only_the_healthy_domains; fault_must_be_observed_in_multi_and_single",
    "checks":"architecture_neutral_rules_and_safety; reference_domain_coverage_from_cited_keys; judge_quality",
    "metrics":"rule_pass_rate; coverage_rate; judge_dimensions; latency_p50_p95; input_output_tokens; model_calls; by_category",
    "minimum_cases_per_category":3,
    "layers":{name:{"dataset":layer["dataset"], "arms":list(layer["arms"]), "judge":layer["judge"]} for name, layer in LAYERS.items()},
    "decision_rule":"paired_judge_mean_other_minus_multi; case_bootstrap_2000_95pct; below_0.03_no_clear_difference; 0.03_or_more_with_interval_excluding_0_is_a_difference; otherwise_insufficient_evidence; layers_never_merged; safety_failure_blocks_that_arm",
    "missing":"retain_all_planned_arms; any_missing_failure_or_unknown_usage_blocks_claims",
    "stopping":"budget_deadline_safety_failure_or_infrastructure_failure_limit",
    "inference":"descriptive_engineering_comparison_on_synthetic_cases_only; no_statistical_superiority_or_clinical_generalization",
}


def resolve_comparison(request: ComparisonRequest):
    if request.cases is not None: return "custom", request.cases
    data = json.loads((ROOT/"resources"/LAYERS[request.layer]["dataset"] if request.layer else DATASET).read_text())
    return data["version"], [ComparisonCase.model_validate(row) for row in data["cases"]]
