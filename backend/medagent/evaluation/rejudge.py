"""Re-score saved comparison answers with the configured Judge.

Answers are not regenerated, so only the Judge changes. The rebuilt input is hashed against the one stored with the original
score, under the protocol version of that run; a turn whose input differs is skipped and counted instead of being compared.
"""
import argparse
import asyncio
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace
from ..comparison.contracts import LAYERS, ComparisonRequest, daily_reference_snapshot, resolve_comparison
from ..core.config import Settings
from ..core.contracts import AgentExecution, Answer, Reference
from ..core.provider import ClaudeGateway, vendor_gateway
from .budget import BudgetedModel, ModelBudget, aggregate_usage
from .contracts import EvalLimits
from .judging import JUDGE_PROTOCOL_VERSION, grade_result, judge_audit, judge_input, request_grade
from .metrics import digest
from .required_points import POINT_VALIDATOR_VERSION, required_points

DIMENSIONS = ("relevance", "accuracy", "completeness", "helpfulness")


def histories(report: dict) -> dict[str, list[dict]]:
    return {prep["case_id"]:[message for row in prep["turns"] for message in (
        {"role":"user", "content":row["message"]}, {"role":"assistant", "content":row["reply"]["answer"]["text"]})]
        for prep in report["preparation"]}


def sample(report: dict, per_category: int, seed: int) -> list[dict]:
    """A seeded, per-category sample of scored answers; urgent answers are fixed text, so they are left out."""
    grouped = defaultdict(list)
    for pair in report["pairs"]:
        if pair["category"] == "urgent":
            continue
        for name, arm in pair["arms"].items():
            if arm.get("quality") and arm.get("reply"):
                grouped[pair["category"]].append({"case_id":pair["case_id"], "pair":pair["id"], "arm":name, "observation":arm})
    picker = random.Random(seed)
    return [item for _, items in sorted(grouped.items()) for item in picker.sample(items, min(per_category, len(items)))]


def rebuilt_input(item: dict, cases: dict, history: dict) -> dict | None:
    arm = item["observation"]
    stored = arm["reply"]
    # Only what the Judge sees; stored replies leave out fields the API does not return.
    reply = SimpleNamespace(request_id=stored["request_id"], subject=stored["subject"], primary_agent=stored["primary_agent"],
                            degraded=stored.get("degraded"), answer=Answer.model_validate(stored["answer"]),
                            agent_results=[AgentExecution.model_validate(row) for row in stored["agent_results"]],
                            references=[Reference.model_validate(row) for row in stored["references"]])
    case = cases.get(item["case_id"])
    if case is None:
        return None
    policy = case.source_gap.judge_criteria() if case.source_gap else None
    context = arm.get("judge_context") or {}
    version = context.get("protocol_version")
    legacy = {"judge-context-3", "judge-context-4", "judge-context-5", "judge-context-6"}
    if version not in legacy | {JUDGE_PROTOCOL_VERSION}:
        return None
    params = {"trace":arm.get("trace"), "source_policy":policy}
    turns = history.get(item["case_id"], [])
    payload = judge_input(case.target, reply, turns, **params, points=required_points(case), hide_roles=context.get("hide_roles") is True)
    original = judge_input(case.target, reply, turns, **params) if version in legacy else deepcopy(payload)
    original["protocol_version"] = version
    if original["protocol_version"] in {"judge-context-3", "judge-context-4", "judge-context-5"}:
        original["current_turn"]["runtime_observation"].pop("lookup_outcomes")
        original["current_turn"]["expected_behavior"].pop("source_gap_policy", None)
    return payload if judge_audit(original)["input_sha256"] == context.get("input_sha256") else None


PILOT_KIND = "frozen_daily_v4_three_case_pilot"
PILOT_IDS = {"d-sourced-amoxicillin", "d-sourced-metformin", "d-sourced-tirads"}


def pilot_inputs(report: dict, *, controls: bool) -> list[tuple[dict, dict]]:
    """Preflight the complete saved pilot before allowing its first paid request."""
    version, cases = resolve_comparison(ComparisonRequest(layer="daily", repetitions=4, daily_version="comparison-daily-4"))
    by_id = {case.id:case for case in cases}
    pairs = report["pairs"]
    ids = [pair["case_id"] for pair in pairs]
    spec = daily_reference_snapshot(version)
    if (report.get("kind") != PILOT_KIND or report.get("complete") is not True or report.get("case_repetitions") != 1
            or report["metadata"]["dataset"] != version or sorted(ids) != sorted(PILOT_IDS)
            or report["metadata"]["cases_hash"] != digest([by_id[key].model_dump(mode="json") for key in ids])
            or report.get("reference_snapshot") != spec or report["metadata"].get("references_hash") != spec["sha256"]):
        raise ValueError("pilot_snapshot_mismatch")
    inputs = []
    multi = {}
    for pair in pairs:
        case = by_id[pair["case_id"]]
        if (case.preparation or case.target.message != pair["question"] or sorted(pair["evidence_key"]) != sorted(case.evidence_passages)
                or set(pair["arms"]) != {"multi_agent", "single_agent"} or not required_points(case)):
            raise ValueError(f"pilot_case_mismatch:{case.id}")
        for arm_name, arm in pair["arms"].items():
            item = {"case_id":case.id, "pair":case.id, "arm":arm_name, "observation":arm}
            payload = rebuilt_input(item, by_id, {})
            if payload is None or not arm.get("quality") or arm["judge_context"]["history_user_turns"] != 0:
                raise ValueError(f"pilot_original_input_mismatch:{case.id}:{arm_name}")
            inputs.append((item, payload))
            if arm_name == "multi_agent":
                multi[case.id] = payload
    if controls:
        case_id = "d-sourced-amoxicillin"
        original = multi[case_id]
        omission = deepcopy(original)
        answer = omission["current_turn"]["answer"]
        clause = "属于青霉素类抗生素，"
        if answer["text"].count(clause) != 1:
            raise ValueError("pilot_control_clause_mismatch")
        answer["text"] = answer["text"].replace(clause, "")
        wrong_source = deepcopy(original)
        current = wrong_source["current_turn"]
        other = multi["d-sourced-metformin"]["current_turn"]
        if ([ref["key"] for ref in other["references"]] != ["metformin-overview"]
                or current["answer"]["reference_keys"] != ["amoxicillin-overview"]):
            raise ValueError("pilot_control_reference_mismatch")
        current["references"] = deepcopy(other["references"])
        current["answer"]["reference_keys"] = ["metformin-overview"]
        for name, payload in (("omitted_class", omission), ("unrelated_source", wrong_source)):
            inputs.append(({"case_id":case_id, "pair":case_id, "arm":"multi_agent", "control":name}, payload))
    return inputs


def rank(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered.index(value)+(ordered.count(value)-1)/2 for value in values]


def summarize(rows: list[dict]) -> dict:
    scored = [row for row in rows if row["scores"]]
    if not scored:
        return {"answers":len(rows), "fully_scored":0}
    original = [row["original_mean"] for row in scored]
    current = [statistics.mean(score["mean"] for score in row["scores"]) for row in scored]
    gaps = [new-old for old, new in zip(original, current)]
    spreads = [max(score["mean"] for score in row["scores"])-min(score["mean"] for score in row["scores"]) for row in scored if len(row["scores"]) > 1]
    spearman = None
    if len(scored) > 2 and len(set(original)) > 1 and len(set(current)) > 1:
        spearman = round(statistics.correlation(rank(original), rank(current)), 3)
    return {"answers":len(rows), "fully_scored":sum(row["errors"] == [] for row in rows),
            "errors":sum(len(row["errors"]) for row in rows), "original_mean":round(statistics.mean(original), 3),
            "new_mean":round(statistics.mean(current), 3), "mean_difference":round(statistics.mean(gaps), 3),
            "mean_absolute_difference":round(statistics.mean(abs(gap) for gap in gaps), 3),
            "largest_differences":[{"pair":row["pair"], "arm":row["arm"], "original":row["original_mean"], "new":round(value, 3)}
                                   for row, value in sorted(zip(scored, current), key=lambda item:-abs(item[1]-item[0]["original_mean"]))[:5]],
            "rank_correlation":spearman, "mean_repeat_spread":round(statistics.mean(spreads), 3) if spreads else None,
            "max_repeat_spread":round(max(spreads), 3) if spreads else None,
            "by_dimension":{key:{"original":round(statistics.mean(row["original"][key] for row in scored), 3),
                                 "new":round(statistics.mean(score[key] for row in scored for score in row["scores"]), 3)} for key in DIMENSIONS}}


async def score(report: dict, model, limits: EvalLimits, *, per_category: int, repeats: int, seed: int, model_name: str | None = None,
                output_tokens: int = 950, thinking: bool = False, pilot_controls: bool = False) -> dict:
    pilot = report.get("kind") == PILOT_KIND
    if pilot_controls and not pilot:
        raise ValueError("controls_require_three_case_pilot")
    layer = "daily" if pilot else report.get("layer")
    repetitions = 2 * len(LAYERS[layer]["arms"]) if layer else 3
    request = ComparisonRequest(layer=layer, repetitions=repetitions, judge=layer not in {"safety", "permission"})
    dataset = report["metadata"].get("dataset")
    if layer == "daily" and dataset in {"comparison-daily-2", "comparison-daily-3"}:
        # Older daily runs sent the expanded dataset as custom cases. Rebuild their original question and policy.
        request = request.model_copy(update={"daily_version":"comparison-daily-3"})
    elif layer == "daily" and dataset == "comparison-daily-4":
        request = request.model_copy(update={"daily_version":dataset})
    _, cases = resolve_comparison(request)
    by_id = {case.id:case for case in cases}
    archived = {}
    if layer == "daily" and dataset == "custom":
        _, legacy_cases = resolve_comparison(request.model_copy(update={"daily_version":"comparison-daily-3"}))
        archived = {case.id:case for case in legacy_cases}
    budget = ModelBudget(limits)
    judge = BudgetedModel(model, budget, "judge")
    rows, control_rows, skipped = [], [], 0
    if pilot:
        inputs = pilot_inputs(report, controls=pilot_controls)
        if per_category < 6 or (pilot_controls and repeats != 1) or repeats < 1:
            raise ValueError("pilot_requires_all_six_originals_and_one_pass_with_controls")
        if len(inputs)*repeats > limits.max_model_calls:
            raise ValueError("pilot_exceeds_call_budget")
    else:
        inputs = []
        history = histories(report)
        for item in sample(report, per_category, seed):
            payload = rebuilt_input(item, by_id, history)
            if payload is None and item["case_id"] in archived:
                payload = rebuilt_input(item, archived, history)
            if payload is None:
                skipped += 1
            else:
                inputs.append((item, payload))
    for item, payload in inputs:
        row = {key:item[key] for key in ("case_id", "pair", "arm")}
        row.update(scores=[], errors=[], elapsed_ms=[], field_orders=[])
        if "control" in item:
            row["control"] = item["control"]
            control_rows.append(row)
        else:
            original = item["observation"]["quality"]
            row.update(original=original, original_mean=original["mean"])
            rows.append(row)
        if pilot:
            row.update(judge_input=payload, judge_context=judge_audit(payload))
            if "observation" in item:
                row["original_judge_context"] = item["observation"]["judge_context"]
        for _ in range(repeats):
            started = time.perf_counter()
            result = None
            try:
                result = await request_grade(judge, payload, output_tokens)
                row["field_orders"].append(list(result))
                row["scores"].append(grade_result(payload, result))
            except Exception as exc:
                row["errors"].append(str(exc) or type(exc).__name__)
                if result is not None:
                    row.setdefault("invalid_results", []).append(result)
            row["elapsed_ms"].append(round((time.perf_counter()-started)*1000))
    output = {"kind":"judge_rescore", "synthetic":True, "judge_model":model_name, "judge_protocol":JUDGE_PROTOCOL_VERSION, "thinking":thinking, "output_tokens":output_tokens,
            "source_code_hash":report["metadata"]["code_hash"],
            "source_run":report["run_id"], "repeats":repeats, "seed":seed, "skipped_input_mismatch":skipped,
            "note":"same stored answers, different Judge; scores are not comparable with any other Judge's",
            "usage":aggregate_usage(budget.entries), "budget_stop":budget.exhausted, "summary":summarize(rows), "cases":rows,
            "generated_at":datetime.now(timezone.utc).isoformat()}
    if pilot:
        output.update(controls=control_rows, call_log=budget.entries)
    if any("required_points" in p["current_turn"]["expected_behavior"] for _, p in inputs):
        output["point_validator"] = POINT_VALIDATOR_VERSION
    return output


async def run_online(path: Path, max_calls: int, per_category: int, repeats: int, seed: int, judge: str = "configured",
                     model: str = "", output_tokens: int = 950, thinking: bool = False, pilot_controls: bool = False) -> dict:
    settings = Settings.load()
    if thinking:
        # Thinking takes far longer than a direct answer.
        settings = replace(settings, thinking=True, model_timeout=max(settings.model_timeout, 180))
    name = settings.model if judge == "primary" else model or settings.judge_model
    gateway = ClaudeGateway(settings) if judge == "primary" else vendor_gateway(settings, name)
    if gateway is None:
        raise RuntimeError("judge_model_not_configured")
    try:
        return await score(json.loads(path.read_text()), gateway,
                           EvalLimits(max_model_calls=max_calls, max_output_tokens=max_calls*output_tokens, per_call_output_tokens=output_tokens),
                           per_category=per_category, repeats=repeats, seed=seed, model_name=name, output_tokens=output_tokens,
                           thinking=thinking, pilot_controls=pilot_controls)
    finally:
        await gateway.close()


def main():
    parser = argparse.ArgumentParser(description="Re-score saved comparison answers with the configured Judge (paid)")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-calls", type=int, required=True, help="hard cap on Judge calls")
    parser.add_argument("--per-category", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--judge", choices=["configured", "primary"], default="configured", help="primary scores with ANTHROPIC_MODEL")
    parser.add_argument("--model", default="", help="second-vendor model; overrides DEEPSEEK_JUDGE_MODEL")
    parser.add_argument("--thinking", action="store_true", help="enable the model's thinking mode")
    parser.add_argument("--pilot-controls", action="store_true", help="include two artificial controls for the frozen three-case pilot")
    parser.add_argument("--output-tokens", type=int, default=950, help="per-call output cap, thinking included")
    args = parser.parse_args()
    if not 1 <= args.max_calls <= 100:
        parser.error("--max-calls must be between 1 and 100")
    if not 1 <= args.output_tokens <= 3000:
        parser.error("--output-tokens must be between 1 and 3000")
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("preserve_existing_rejudge_report:"+str(target))
    report = asyncio.run(run_online(Path(args.report), args.max_calls, args.per_category, args.repeats, args.seed, args.judge,
                                    args.model, args.output_tokens, args.thinking, args.pilot_controls))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({key:report[key] for key in ("summary", "usage", "budget_stop", "skipped_input_mismatch")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
