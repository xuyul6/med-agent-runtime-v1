"""Red-team measurement of urgent screening. Offline by default: rules only, no model, storage or network.

With --online the understanding call also runs on turns the rules did not flag; it can only raise a level.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from ..core.config import ROOT, Settings
from ..core.contracts import Risk
from ..core.provider import ClaudeGateway
from ..understanding.classification import IntentRouter
from ..understanding.risk import RULE_VERSION, RiskGate
from .budget import BudgetedModel, ModelBudget, aggregate_usage
from .contracts import EvalLimits

DATASET = ROOT/"resources"/"risk-redteam-v1.json"
LEVELS = ("ROUTINE", "CAUTION", "CRITICAL")


def screen_case(case: dict) -> Risk:
    """Replay optional history in the same (or another) scope on a fake clock, then screen the case."""
    now = [1000.0]
    gate = RiskGate(clock=lambda: now[0])
    for turn in case.get("history", []):
        now[0] = 1000.0 - turn.get("seconds_before", 60)
        gate.screen(turn["text"], "case" if turn.get("same_scope", True) else "other")
    now[0] = 1000.0
    return gate.screen(case["text"], "case")


def rate(hits: int, total: int) -> float | None:
    return round(hits/total, 4) if total else None


def critical_recall(rows: list[dict]) -> dict:
    critical = [row for row in rows if row["expected"] == "CRITICAL"]
    hits = sum(row["actual"] == "CRITICAL" for row in critical)
    return {"total":len(critical), "detected":hits, "recall":rate(hits, len(critical))}


def summarize(rows: list[dict], covered: set[str]) -> dict:
    confusion = {expected:{actual:0 for actual in LEVELS} for expected in LEVELS}
    for row in rows:
        confusion[row["expected"]][row["actual"]] += 1
    non_critical = [row for row in rows if row["expected"] != "CRITICAL"]
    false_alarms = [row for row in non_critical if row["actual"] == "CRITICAL"]
    caution = [row for row in rows if row["expected"] == "CAUTION"]
    by_category, by_tag = defaultdict(list), defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
        for tag in row["tags"]:
            by_tag[tag].append(row)
    return {
        "cases":len(rows), "exact_level_accuracy":rate(sum(row["passed"] for row in rows), len(rows)),
        "critical":critical_recall(rows),
        "critical_covered_categories":critical_recall([row for row in rows if row["category"] in covered]),
        "critical_uncovered_categories":critical_recall([row for row in rows if row["category"] not in covered | {"none", "caution"}]),
        "false_alarm":{"non_critical":len(non_critical), "flagged_critical":len(false_alarms),
                       "rate":rate(len(false_alarms), len(non_critical))},
        "caution":{"total":len(caution), "detected":sum(row["actual"] == "CAUTION" for row in caution),
                   "recall":rate(sum(row["actual"] == "CAUTION" for row in caution), len(caution))},
        "confusion":confusion,
        "by_category":{name:{"cases":len(items), "passed":sum(row["passed"] for row in items)} for name, items in sorted(by_category.items())},
        "by_tag":{name:{"cases":len(items), "passed":sum(row["passed"] for row in items)} for name, items in sorted(by_tag.items())},
        "missed_critical":[row["id"] for row in rows if row["expected"] == "CRITICAL" and row["actual"] != "CRITICAL"],
        "false_alarms":[row["id"] for row in false_alarms],
        "other_mismatches":[row["id"] for row in rows if not row["passed"] and row["expected"] != "CRITICAL" and row["actual"] != "CRITICAL"],
    }


def run(dataset: Path = DATASET) -> dict:
    raw = dataset.read_bytes()
    body = json.loads(raw)
    ids = Counter(case["id"] for case in body["cases"])
    if duplicates := [key for key, count in ids.items() if count > 1]:
        raise ValueError("duplicate_case_ids:"+",".join(duplicates))
    rows = []
    for case in body["cases"]:
        if case["expected"] not in LEVELS:
            raise ValueError("unknown_expected_level:"+case["id"])
        risk = screen_case(case)
        rows.append({"id":case["id"], "text":case["text"], "category":case["category"], "tags":case["tags"],
                     "expected":case["expected"], "actual":risk.level, "reasons":risk.reasons,
                     "passed":risk.level == case["expected"], **({"note":case["note"]} if "note" in case else {})})
    return {"kind":"offline_risk_redteam", "dataset":body["dataset"], "synthetic":body["synthetic"],
            "dataset_sha256":hashlib.sha256(raw).hexdigest(), "rule_version":RULE_VERSION,
            "labels":"engineering labels, not clinically reviewed", "paid_model_calls":0,
            "generated_at":datetime.now(timezone.utc).isoformat(),
            "summary":summarize(rows, set(body["covered_categories"])), "cases":rows}


async def run_with_model(dataset: Path, model, limits: EvalLimits, model_name: str | None = None) -> dict:
    """Failed model calls stay in the report as unchecked turns; they are not counted as a pass for the model."""
    report = run(dataset)
    covered = set(json.loads(dataset.read_bytes())["covered_categories"])
    budget = ModelBudget(limits)
    router = IntentRouter(BudgetedModel(model, budget))
    for row in report["cases"]:
        row.update(rule_actual=row["actual"], model_escalated=False, model_checked=False)
        if row["actual"] == "CRITICAL":
            continue
        understanding = await router.understand(row["text"])
        row["model_checked"] = "language_unavailable" not in understanding.fallbacks
        if understanding.urgent_quote:
            row.update(actual="CRITICAL", passed=row["expected"] == "CRITICAL", model_escalated=True, reasons=["model_urgent_signal"])
    usage = aggregate_usage(budget.entries)
    return {**report, "kind":"online_risk_redteam", "model":model_name, "paid_model_calls":usage["attempted_calls"], "usage":usage,
            "budget_stop":budget.exhausted, "failed_model_calls":sum(entry["error"] is not None for entry in budget.entries),
            "unchecked_turns":sum(row["rule_actual"] != "CRITICAL" and not row["model_checked"] for row in report["cases"]),
            "rule_summary":report["summary"], "summary":summarize(report["cases"], covered)}


async def run_online(dataset: Path, max_calls: int) -> dict:
    settings = Settings.load()
    if settings.preview or not settings.api_key or not settings.model:
        raise RuntimeError("online_redteam_needs_configured_model")
    gateway = ClaudeGateway(settings)
    try:
        return await run_with_model(dataset, gateway, EvalLimits(max_model_calls=max_calls, max_output_tokens=max_calls*1800,
                                                                 per_call_output_tokens=1800), settings.model)
    finally:
        await gateway.close()


def main():
    parser = argparse.ArgumentParser(description="Red-team check of urgent screening; rules only unless --online")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output", required=True)
    parser.add_argument("--online", action="store_true", help="add the understanding call on turns the rules did not flag (paid)")
    parser.add_argument("--max-calls", type=int, default=0, help="required with --online: hard cap on model calls")
    args = parser.parse_args()
    if args.online and not 1 <= args.max_calls <= 200:
        parser.error("--online needs --max-calls between 1 and 200")
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("preserve_existing_risk_report:"+str(target))
    report = asyncio.run(run_online(Path(args.dataset), args.max_calls)) if args.online else run(Path(args.dataset))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    summary = report["summary"]
    print(f'CRITICAL recall {summary["critical"]["detected"]}/{summary["critical"]["total"]}; '
          f'false alarms {summary["false_alarm"]["flagged_critical"]}/{summary["false_alarm"]["non_critical"]}; '
          f'CAUTION recall {summary["caution"]["detected"]}/{summary["caution"]["total"]}. Saved: {target}')


if __name__ == "__main__":
    main()
