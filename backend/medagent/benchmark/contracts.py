"""Preregistered, bounded paired-context experiment inputs."""
from typing import Literal
from pydantic import Field, field_validator, model_validator
from ..core.config import ROOT
from ..evaluation.contracts import DialogTurn, EvalContract
from ..evaluation.metrics import METRIC_VERSION
from ..evaluation.judging import JUDGE_PROTOCOL_VERSION
import json


class BenchmarkCase(EvalContract):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    preparation: list[DialogTurn] = Field(default_factory=list, max_length=7)
    target: DialogTurn
    subject: str = Field(default="self", min_length=1, max_length=40)

    @field_validator("preparation", mode="before")
    @classmethod
    def string_turns(cls, value):
        return [{"message":item} if isinstance(item,str) else item for item in value] if isinstance(value,list) else value


class BenchmarkLimits(EvalContract):
    timeout_seconds: float = Field(default=600, gt=0, le=1800)
    max_model_calls: int = Field(default=800, ge=1, le=2000, strict=True)
    max_output_tokens: int = Field(default=1200000, ge=1, le=4000000, strict=True)
    per_call_output_tokens: int = Field(default=2500, ge=950, le=3000, strict=True)
    max_infrastructure_failures: int = Field(default=4, ge=1, le=20, strict=True)


class BenchmarkRequest(EvalContract):
    cases: list[BenchmarkCase] | None = Field(default=None, min_length=1, max_length=10)
    repetitions: int = Field(default=4, ge=2, le=10, strict=True)
    cache_conditions: list[Literal["cold", "warm"]] = Field(default_factory=lambda:["cold","warm"], min_length=1, max_length=2)
    seed: int = Field(default=20261027, ge=0, le=2147483647, strict=True)
    judge: bool = Field(default=True, strict=True)
    limits: BenchmarkLimits = Field(default_factory=BenchmarkLimits)

    @field_validator("cases", mode="before")
    @classmethod
    def reject_explicit_null(cls, value):
        if value is None: raise ValueError("omit_cases_for_defaults")
        return value

    @model_validator(mode="after")
    def validate_design(self):
        if self.repetitions % 2: raise ValueError("even_repetitions_required_for_balanced_order")
        if len(set(self.cache_conditions)) != len(self.cache_conditions): raise ValueError("duplicate_cache_condition")
        if self.cases is not None and len({case.id for case in self.cases}) != len(self.cases): raise ValueError("duplicate_case_id")
        if self.cases is not None:
            signatures = [(case.subject,tuple(turn.message for turn in case.preparation),case.target.message) for case in self.cases]
            if len(set(signatures)) != len(signatures): raise ValueError("duplicate_scenario_content")
        return self


PROTOCOL = {
    "version":"paired-context-4",
    "behavior_check_version":METRIC_VERSION,
    "judge_protocol_version":JUDGE_PROTOCOL_VERSION,
    "preparation":"once_per_case_selective_then_freeze_user_assistant_state_entities_summary_episodes_risk",
    "arms":["selective","full"], "order":"seeded_case_shuffle_then_alternating_AB_BA_in_each_case_cache_cell",
    "judge_order":"reverse_target_order_after_both_targets; each_judge_sees_only_own_answer",
    "cache":"cold_empty; warm_shared_canonical_selective_replay_cache_with_90s_TTL_rebased_per_arm",
    "provider_cache":"not_controlled; record_base_cache_read_cache_creation_separately; no_cost_claim",
    "state":"fresh_branch_memory_episode_collection_monitor_tools_subjects_router; restore_same_snapshot_and_recent_risk",
    "context_freshness":"freeze_context_clock_at_snapshot_for_both_arms; not_a_live_latency_measurement",
    "comparability":"same_frozen_snapshot_and_known_consistent_provider_response_model; unique_request_ids_may_vary",
    "quality_threshold":.75, "quality_margin":.03,
    "quality_rule":"all_observed_arms_pass_rules_safety_and_quality; every_case_cache_cell_mean_delta_in_each_dimension_at_least_minus_margin",
    "minimum_scenarios":4, "minimum_repetitions_per_cell":4,
    "missing":"retain_all_planned_arms; any_missing_failure_or_unknown_usage_blocks_joint_claim",
    "stopping":"budget_deadline_cleanup_or_safety_failure; infrastructure_failure_limit; never_stop_for_a_desired_saving",
    "inference":"descriptive_engineering_noninferiority_on_observed_cases_only; no_statistical_equivalence_or_clinical_generalization",
    "input_metric":"input_tokens_plus_cache_read_input_tokens_plus_cache_creation_input_tokens; unknown_if_any_counter_missing",
}


def resolve_benchmark(request: BenchmarkRequest):
    if request.cases is not None: return "custom", request.cases
    data = json.loads((ROOT/"resources/benchmark.json").read_text())
    return data["version"], [BenchmarkCase.model_validate(row) for row in data["cases"]]
