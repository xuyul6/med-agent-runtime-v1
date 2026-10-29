"""Descriptive paired metrics; missing observations never turn into savings."""
import math
import statistics
from collections import defaultdict
from .benchmark_contracts import PROTOCOL


DIMENSIONS = ("relevance","accuracy","completeness","helpfulness","mean")
COUNTERS = ("input_tokens","output_tokens","cache_read_input_tokens","cache_creation_input_tokens")


def measured_usage(entries):
    if not entries:
        return {"attempted_calls":0,"calls":0,"measurement":"not_used",**{key:0 for key in COUNTERS},"total_input_tokens":0}
    events = [event for entry in entries for event in entry["trace"]["events"] if event["event"] == "model"]
    # A failed call without a provider response makes the aggregate unknown.
    complete = all(entry["usage"]["measurement"] == "provider" for entry in entries)
    counters = {key:sum(event[key] for event in events) if complete and events and all(
        type(event.get(key)) is int and event[key] >= 0 for event in events) else None for key in COUNTERS}
    inputs = [counters[key] for key in ("input_tokens","cache_read_input_tokens","cache_creation_input_tokens")]
    return {"attempted_calls":len(entries),"calls":len(events),"measurement":"provider" if complete else "unavailable",
            **counters,"total_input_tokens":sum(inputs) if all(value is not None for value in inputs) else None}


def staged_usage(entries):
    groups = defaultdict(list)
    for entry in entries: groups[entry["stage"]].append(entry)
    return {stage:measured_usage(rows) for stage,rows in sorted(groups.items())}


def distribution(values):
    values = sorted(values)
    if not values: return {"n":0,"mean":None,"median":None,"p90":None,"min":None,"max":None,"stdev":None}
    return {"n":len(values),"mean":statistics.mean(values),"median":statistics.median(values),
            "p90":values[math.ceil(.9*len(values))-1],"min":values[0],"max":values[-1],
            "stdev":statistics.stdev(values) if len(values) > 1 else None}


def paired_statistics(rows):
    records, missing = [], []
    for pair in rows:
        arms = pair["arms"]
        values = {mode:arms[mode].get("usage",{}).get("total_input_tokens") for mode in ("selective","full")}
        if any(arms[mode]["status"] == "not_run" or values[mode] is None for mode in values):
            missing.append(pair["id"])
        else:
            full,selective = values["full"],values["selective"]
            records.append({"pair_id":pair["id"],"selective":selective,"full":full,"difference":selective-full,
                            "fraction":1-selective/full if full else None})
    means = {mode:distribution([row[mode] for row in records]) for mode in ("selective","full")}
    full_sum = sum(row["full"] for row in records)
    output_pairs = []
    missing_output = []
    for pair in rows:
        values = {mode:pair["arms"][mode].get("usage",{}).get("output_tokens") for mode in ("selective","full")}
        if any(pair["arms"][mode]["status"] == "not_run" or values[mode] is None for mode in values):
            missing_output.append(pair["id"])
        else:
            output_pairs.append({"pair_id":pair["id"],**values,"difference":values["selective"]-values["full"]})
    return {"planned_pairs":len(rows),"observed_pairs":len(records),"missing_pairs":missing,
            "input":means,"paired_difference":distribution([row["difference"] for row in records]),"pairs":records,
            "output":{mode:distribution([row[mode] for row in output_pairs]) for mode in ("selective","full")},
            "output_missing_pairs":missing_output,"output_paired_difference":distribution([row["difference"] for row in output_pairs]),
            "input_reduction":1-sum(row["selective"] for row in records)/full_sum if records and not missing and full_sum else None}


def summarize_benchmark(report, request):
    rows = report["pairs"]
    cells = defaultdict(list)
    for pair in rows: cells[(pair["case_id"],pair["cache"])].append(pair)
    by_cache = {cache:paired_statistics([pair for pair in rows if pair["cache"] == cache]) for cache in request.cache_conditions}
    quality_cells = []
    all_quality = True
    all_passed = True
    for (case_id,cache),pairs in sorted(cells.items()):
        deltas = {key:[] for key in DIMENSIONS}
        for pair in pairs:
            for mode,arm in pair["arms"].items():
                all_quality &= arm.get("quality") is not None
                all_passed &= bool(arm.get("rule_passed")) and bool(arm.get("quality")) and arm["quality"]["mean"] >= PROTOCOL["quality_threshold"]
            if all(pair["arms"][mode].get("quality") is not None for mode in ("selective","full")):
                for key in DIMENSIONS: deltas[key].append(pair["arms"]["selective"]["quality"][key]-pair["arms"]["full"]["quality"][key])
        stats = {key:distribution(values) for key,values in deltas.items()}
        within = len(deltas["mean"]) == len(pairs) and all(value["mean"] is not None and
            value["mean"] >= -PROTOCOL["quality_margin"]-1e-12 for value in stats.values())
        quality_cells.append({"case_id":case_id,"cache":cache,"planned":len(pairs),"deltas":stats,"within_margin":within})
    reasons = []
    if report["mode"] != "online": reasons.append("offline_or_simulated_run")
    if not report["complete"]: reasons.append("incomplete_run")
    if not request.judge or not all_quality: reasons.append("judge_disabled_or_missing")
    if not all_passed: reasons.append("rule_safety_or_quality_failure")
    if len({pair["case_id"] for pair in rows}) < PROTOCOL["minimum_scenarios"] or request.repetitions < PROTOCOL["minimum_repetitions_per_cell"]:
        reasons.append("insufficient_scenarios_or_repetitions")
    if any(not cell["within_margin"] for cell in quality_cells): reasons.append("quality_margin_not_met")
    if any(group["missing_pairs"] for group in by_cache.values()): reasons.append("unknown_or_missing_input_usage")
    if any(group["output_missing_pairs"] for group in by_cache.values()): reasons.append("unknown_or_missing_output_usage")
    if not all(report.get("comparability",{}).get(key) is True for key in ("snapshot_verified","actual_model_consistent")):
        reasons.append("snapshot_or_actual_model_not_verified")
    report["statistics"] = {"by_cache":by_cache,
        "by_case_cache":[{"case_id":case_id,"cache":cache,**paired_statistics(pairs)} for (case_id,cache),pairs in sorted(cells.items())],
        "quality_cells":quality_cells}
    report["cache_observations"] = {cache:{mode:{
        "seeded_entries":distribution([arm["seeded_cache_entries"] for pair in rows if pair["cache"] == cache
                                        for name,arm in pair["arms"].items() if name == mode and "seeded_cache_entries" in arm]),
        "observed_hits":distribution([arm["cache_hits"] for pair in rows if pair["cache"] == cache
                                       for name,arm in pair["arms"].items() if name == mode and "cache_hits" in arm])}
        for mode in ("selective","full")} for cache in request.cache_conditions}
    report["quality_noninferiority"] = "within_margin_on_observed_cases" if not reasons else "not_established"
    report["quality_equivalence"] = "not_established"
    report["claim_blockers"] = reasons
    report["supported_input_reduction"] = {cache:group["input_reduction"] if not reasons else None for cache,group in by_cache.items()}
    report["not_run"] = [{"pair_id":pair["id"],"mode":mode,"reason":arm.get("reason","not_reached")}
        for pair in rows for mode,arm in pair["arms"].items() if arm["status"] == "not_run"]
