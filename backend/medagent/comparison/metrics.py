"""Descriptive per-arm and per-category comparison; missing observations never become wins."""
import math
import random
from ..benchmark.metrics import DIMENSIONS, distribution
from .contracts import CATEGORIES, FAULT_CATEGORIES, LAYERS, PROTOCOL

# Below this the difference is smaller than the Judge's own repeat noise (0.026).
NO_CLEAR_DIFFERENCE = .03


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


def stability_statistics(arms: list[dict]) -> dict:
    observed = [arm for arm in arms if arm["status"] != "not_run"]
    def rate(name):
        values = [arm[name] for arm in observed if arm.get(name) is not None]
        return sum(map(bool, values))/len(values) if values else None
    first_failed = [arm for arm in observed if arm.get("first_attempt_failed")]
    return {"observed":len(observed), "whole_failure_rate":rate("whole_failure"),
            # Without a retry a failed first attempt is the end of the turn.
            "no_retry_whole_failure_rate":sum(bool(arm.get("first_attempt_failed") or arm.get("whole_failure")) for arm in observed)/len(observed) if observed else None,
            "retry_recovered_rate":sum(not arm.get("whole_failure") for arm in first_failed)/len(first_failed) if first_failed else None,
            "healthy_kept_rate":rate("coverage"), "gap_disclosed_rate":rate("gap_disclosed"), "unsourced_fill_rate":rate("unsourced_fill")}


def case_bootstrap_ci(rows: list[dict], arm: str, seed: int, rounds: int = 2000) -> dict:
    """Paired Judge difference against multi_agent; repeats of one case are not independent, so cases are resampled."""
    by_case, missing = {}, 0
    for row in rows:
        own, base = row["arms"][arm].get("quality"), row["arms"]["multi_agent"].get("quality")
        if own and base: by_case.setdefault(row["case_id"], []).append(own["mean"]-base["mean"])
        else: missing += 1
    means = [sum(values)/len(values) for values in by_case.values()]
    if not means: return {"difference":None, "low":None, "high":None, "cases":0, "missing_pairs":missing}
    generator = random.Random(seed)
    samples = sorted(sum(generator.choices(means, k=len(means)))/len(means) for _ in range(rounds))
    return {"difference":sum(means)/len(means), "low":samples[int(.025*rounds)], "high":samples[math.ceil(.975*rounds)-1],
            "cases":len(means), "missing_pairs":missing}


def verdict(interval: dict, threshold: float = NO_CLEAR_DIFFERENCE) -> str:
    difference = interval["difference"]
    if difference is None: return "insufficient_evidence"
    if abs(difference) < threshold: return "no_clear_difference"
    if difference > 0 and interval["low"] > 0: return "other_better"
    if difference < 0 and interval["high"] < 0: return "multi_better"
    return "insufficient_evidence"


def verdicts(rows: list[dict], arms, seed: int) -> dict:
    groups = {"all":rows, **{category:[row for row in rows if row["category"] == category] for category in CATEGORIES
                             if any(row["category"] == category for row in rows)}}
    result = {}
    for name, group in groups.items():
        intervals = {arm:case_bootstrap_ci(group, arm, seed) for arm in arms if arm != "multi_agent"}
        result[name] = {arm:{**interval, "verdict":verdict(interval)} for arm, interval in intervals.items()}
    return result


def summarize_comparison(report: dict, request) -> None:
    rows, arms = report["pairs"], request.arm_names
    report["statistics"] = {
        "by_arm":{arm:arm_statistics([row["arms"][arm] for row in rows]) for arm in arms},
        "by_category":{category:{arm:arm_statistics([row["arms"][arm] for row in rows if row["category"] == category])
                                 for arm in arms} for category in CATEGORIES if any(row["category"] == category for row in rows)},
        "paired_quality":{arm:paired_quality(rows, arm) for arm in arms if arm != "multi_agent"},
        "failures":{arm:stability_statistics([row["arms"][arm] for row in rows]) for arm in arms},
        "rule_outcomes":{category:{arm:stability_statistics([row["arms"][arm] for row in rows if row["category"] == category]) for arm in arms}
                         for category in CATEGORIES if any(row["category"] == category for row in rows)},
    }
    for outcomes in [report["statistics"]["failures"], *report["statistics"]["rule_outcomes"].values()]:
        # Only the single agent's retry can be subtracted; a role's second instance is part of the architecture.
        for arm, stats in outcomes.items():
            if arm != "single_agent": stats["no_retry_whole_failure_rate"] = None
    if request.layer in {"daily", "compound"}:
        report["statistics"]["verdicts"] = verdicts(rows, arms, request.seed)
    if request.layer == "safety":
        report["statistics"]["safety"] = {arm:{"failures":sum(bool(row["arms"][arm].get("safety_failed")) for row in rows),
                                               "observed":sum(row["arms"][arm]["status"] != "not_run" for row in rows)} for arm in arms}
    reasons = []
    if report["mode"] != "online": reasons.append("offline_or_simulated_run")
    if not report["complete"]: reasons.append("incomplete_run")
    observed = [arm for row in rows for arm in row["arms"].values() if arm["status"] != "not_run"]
    judged = not request.layer or LAYERS[request.layer]["judge"]
    if judged and (not request.judge or any(arm.get("quality") is None for arm in observed)): reasons.append("judge_disabled_or_missing")
    if any(arm.get("safety_failed") for arm in observed): reasons.append("safety_failure")
    counts = {category:len({row["case_id"] for row in rows if row["category"] == category}) for category in CATEGORIES}
    if any(0 < count < PROTOCOL["minimum_cases_per_category"] for count in counts.values()):
        reasons.append("too_few_cases_in_a_category")
    if any(arm.get("usage", {}).get("total_input_tokens") is None for arm in observed): reasons.append("unknown_token_usage")
    # primary_only may never reach a faulted supporting domain; the other two have to, or the case tested nothing.
    if any(row["category"] in FAULT_CATEGORIES and name != "primary_only" and arm["status"] == "completed" and not arm.get("fault_triggered")
           for row in rows for name, arm in row["arms"].items()):
        reasons.append("fault_not_triggered")
    report["cases_per_category"] = counts
    report["claim_blockers"] = reasons
    report["conclusion"] = "descriptive_comparison_available" if not reasons else "not_established"
    report["not_run"] = [{"pair_id":row["id"], "arm":arm, "reason":value.get("reason", "not_reached")}
                         for row in rows for arm, value in row["arms"].items() if value["status"] == "not_run"]
