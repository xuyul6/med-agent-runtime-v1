"""Re-score saved comparison answers with the configured Judge.

Answers are not regenerated, so only the Judge changes. The rebuilt input is hashed against the one stored with the original
score, under the protocol version of that run; a turn whose input differs is skipped and counted instead of being compared.
"""
import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace
from ..comparison.contracts import ComparisonRequest, resolve_comparison
from ..core.config import Settings
from ..core.contracts import AgentExecution, Answer, Reference
from ..core.provider import ClaudeGateway, judge_gateway
from .budget import BudgetedModel, ModelBudget, aggregate_usage
from .contracts import EvalLimits, JudgeScore
from .judging import JUDGE_INSTRUCTIONS, JUDGE_PROTOCOL_VERSION, judge_audit, judge_input

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


def rebuilt_input(item: dict, turns: dict, history: dict) -> dict | None:
    arm = item["observation"]
    stored = arm["reply"]
    # Only what the Judge sees; stored replies leave out fields the API does not return.
    reply = SimpleNamespace(request_id=stored["request_id"], subject=stored["subject"], primary_agent=stored["primary_agent"],
                            degraded=stored.get("degraded"), answer=Answer.model_validate(stored["answer"]),
                            agent_results=[AgentExecution.model_validate(row) for row in stored["agent_results"]],
                            references=[Reference.model_validate(row) for row in stored["references"]])
    payload = judge_input(turns[item["case_id"]], reply, history.get(item["case_id"], []), trace=arm.get("trace"))
    original = {**payload, "protocol_version":arm["judge_context"]["protocol_version"]}
    return payload if judge_audit(original)["input_sha256"] == arm["judge_context"]["input_sha256"] else None


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


async def score(report: dict, model, limits: EvalLimits, *, per_category: int, repeats: int, seed: int, model_name: str | None = None) -> dict:
    _, cases = resolve_comparison(ComparisonRequest())
    turns, history = {case.id:case.target for case in cases}, histories(report)
    budget = ModelBudget(limits)
    judge = BudgetedModel(model, budget, "judge")
    rows, skipped = [], 0
    for item in sample(report, per_category, seed):
        payload = rebuilt_input(item, turns, history)
        if payload is None:
            skipped += 1
            continue
        row = {key:item[key] for key in ("case_id", "pair", "arm")}
        row.update(original=item["observation"]["quality"], original_mean=item["observation"]["quality"]["mean"], scores=[], errors=[], elapsed_ms=[])
        for _ in range(repeats):
            started = time.perf_counter()
            try:
                result = await judge.json("evaluation_judge", JUDGE_INSTRUCTIONS, json.dumps(payload, ensure_ascii=False), 950)
                row["scores"].append(JudgeScore.model_validate(result).result())
            except Exception as exc:
                row["errors"].append(type(exc).__name__)
            row["elapsed_ms"].append(round((time.perf_counter()-started)*1000))
        rows.append(row)
    return {"kind":"judge_rescore", "synthetic":True, "judge_model":model_name, "judge_protocol":JUDGE_PROTOCOL_VERSION, "source_code_hash":report["metadata"]["code_hash"],
            "source_run":report["run_id"], "repeats":repeats, "seed":seed, "skipped_input_mismatch":skipped,
            "note":"same stored answers, different Judge; scores are not comparable with any other Judge's",
            "usage":aggregate_usage(budget.entries), "budget_stop":budget.exhausted, "summary":summarize(rows), "cases":rows,
            "generated_at":datetime.now(timezone.utc).isoformat()}


async def run_online(path: Path, max_calls: int, per_category: int, repeats: int, seed: int, judge: str = "configured") -> dict:
    settings = Settings.load()
    gateway = ClaudeGateway(settings) if judge == "primary" else judge_gateway(settings)
    if gateway is None:
        raise RuntimeError("judge_model_not_configured")
    try:
        return await score(json.loads(path.read_text()), gateway,
                           EvalLimits(max_model_calls=max_calls, max_output_tokens=max_calls*950, per_call_output_tokens=950),
                           per_category=per_category, repeats=repeats, seed=seed, model_name=settings.model if judge == "primary" else settings.judge_model)
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
    args = parser.parse_args()
    if not 1 <= args.max_calls <= 100:
        parser.error("--max-calls must be between 1 and 100")
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("preserve_existing_rejudge_report:"+str(target))
    report = asyncio.run(run_online(Path(args.report), args.max_calls, args.per_category, args.repeats, args.seed, args.judge))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({key:report[key] for key in ("summary", "usage", "budget_stop", "skipped_input_mismatch")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
