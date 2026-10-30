"""Descriptive per-arm and per-category comparison; missing observations never become wins."""
import math
from ..benchmark.metrics import DIMENSIONS, distribution
from .contracts import ARMS, CATEGORIES, PROTOCOL


def percentile(values, share):
    values = sorted(values)
    return values[math.ceil(share*len(values))-1] if values else None


def arm_statistics(arms: list[dict]) -> dict:
    observed = [arm for arm in arms if arm["status"] != "not_run"]
    covered = [arm for arm in observed if arm.get("coverage") is not None]
    latency = [arm["elapsed_ms"] for arm in observed if "elapsed_ms" in arm]
    usage = [arm.get("usage", {}) for arm in observed]
    def tokens(key):
        values = [row.get(key) for row in usage]
        return distribution(values) if values and all(isinstance(value, int) for value in values) else None
    return {
        "planned":len(arms), "observed":len(observed), "completed":sum(arm["status"] == "completed" for arm in observed),
        "rule_pass_rate":sum(bool(arm.get("rule_passed")) for arm in observed)/len(observed) if observed else None,
        "coverage_rate":sum(arm["coverage"] for arm in covered)/len(covered) if covered else None,
        "partial_rate":sum(bool(arm.get("partial")) for arm in observed)/len(observed) if observed else None,
        "quality":{key:distribution([arm["quality"][key] for arm in observed if arm.get("quality")]) for key in DIMENSIONS},
        "judge_missing":sum(arm.get("quality") is None for arm in observed),
        "latency_ms":{"p50":percentile(latency, .5), "p95":percentile(latency, .95), "n":len(latency)},
        "input_tokens":tokens("total_input_tokens"), "output_tokens":tokens("output_tokens"),
        "model_calls":distribution([row.get("calls", 0) for row in usage]),
    }


def paired_quality(rows: list[dict], arm: str) -> dict:
    """Per-pair Judge mean difference against multi_agent; pairs missing either score are counted, not dropped silently."""
    deltas, missing = [], 0
    for row in rows:
        own, base = row["arms"][arm].get("quality"), row["arms"]["multi_agent"].get("quality")
        if own and base: deltas.append(own["mean"]-base["mean"])
        else: missing += 1
    return {"vs":"multi_agent", "difference":distribution(deltas), "missing_pairs":missing}


def summarize_comparison(report: dict, request) -> None:
    rows = report["pairs"]
    report["statistics"] = {
        "by_arm":{arm:arm_statistics([row["arms"][arm] for row in rows]) for arm in ARMS},
        "by_category":{category:{arm:arm_statistics([row["arms"][arm] for row in rows if row["category"] == category])
                                 for arm in ARMS} for category in CATEGORIES if any(row["category"] == category for row in rows)},
        "paired_quality":{arm:paired_quality(rows, arm) for arm in ARMS if arm != "multi_agent"},
    }
    reasons = []
    if report["mode"] != "online": reasons.append("offline_or_simulated_run")
    if not report["complete"]: reasons.append("incomplete_run")
    observed = [arm for row in rows for arm in row["arms"].values() if arm["status"] != "not_run"]
    if not request.judge or any(arm.get("quality") is None for arm in observed): reasons.append("judge_disabled_or_missing")
    counts = {category:len({row["case_id"] for row in rows if row["category"] == category}) for category in CATEGORIES}
    if any(0 < count < PROTOCOL["minimum_cases_per_category"] for count in counts.values()):
        reasons.append("too_few_cases_in_a_category")
    if any(arm.get("usage", {}).get("total_input_tokens") is None for arm in observed): reasons.append("unknown_token_usage")
    # primary_only may never reach a faulted supporting domain; the other two have to, or the case tested nothing.
    if any(row["category"] == "tool_fault" and name != "primary_only" and arm["status"] == "completed" and not arm.get("fault_triggered")
           for row in rows for name, arm in row["arms"].items()):
        reasons.append("fault_not_triggered")
    report["cases_per_category"] = counts
    report["claim_blockers"] = reasons
    report["conclusion"] = "descriptive_comparison_available" if not reasons else "not_established"
    report["not_run"] = [{"pair_id":row["id"], "arm":arm, "reason":value.get("reason", "not_reached")}
                         for row in rows for arm, value in row["arms"].items() if value["status"] == "not_run"]
