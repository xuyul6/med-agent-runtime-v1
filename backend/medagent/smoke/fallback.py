"""Paid check of the standby instance: the primary model always fails, so every role runs on the standby vendor."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from ..core.config import ROOT, Settings
from ..core.contracts import Question, Reference
from ..core.provider import ModelFailure, fallback_gateway
from ..evaluation.budget import BudgetedModel, ModelBudget, aggregate_usage
from ..evaluation.contracts import EvalLimits
from ..knowledge.retrieval import ReferenceIndex, Vectorizer
from ..memory import SessionMemory
from ..runtime import MedicalRuntime
from .roles import CASES


class PrimaryDown:
    async def json(self, *args, **kwargs):
        raise ModelFailure("primary_down")

    async def message(self, *args, **kwargs):
        raise ModelFailure("primary_down")


async def score(standby, limits: EvalLimits, model_name: str | None = None) -> dict:
    budget = ModelBudget(limits)
    vectors = Vectorizer("hash")
    index = ReferenceIndex(vectors, None)
    await index.upsert([Reference.model_validate(x) for x in json.loads((ROOT/"resources"/"references.json").read_text())])
    runtime = MedicalRuntime(Settings(preview=True, embeddings="hash"), PrimaryDown(), vectors, index, SessionMemory(),
                             fallback_model=BudgetedModel(standby, budget))
    rows = []
    for case in CASES:
        reply = await runtime.chat(Question(text=case["text"], user="standby-"+case["id"]))
        assignments = reply.routing_decision.assignments
        rows.append({"id":case["id"], "expected_role":case["role"], "primary_agent":reply.primary_agent,
                     "statuses":[item.status for item in assignments], "selected":[item.selected_instance for item in assignments],
                     "errors":[attempt.error_code or attempt.error for item in assignments for attempt in item.attempts if attempt.status != "ok"
                               and attempt.instance.endswith(".2")],
                     # Deterministic paths finish on slot 1 without a model, so "ok" means no task failed, not that the standby ran.
                     "ok":bool(assignments) and all(item.status != "failed" for item in assignments),
                     "standby_used":any((item.selected_instance or "").endswith(".2") for item in assignments)})
    return {"kind":"online_standby_smoke", "synthetic":True, "standby_model":model_name, "cases":rows,
            "completed":sum(row["ok"] for row in rows), "served_by_standby":sum(row["standby_used"] for row in rows), "total":len(rows),
            "usage":aggregate_usage(budget.entries),
            "budget_stop":budget.exhausted, "generated_at":datetime.now(timezone.utc).isoformat()}


async def run_online(max_calls: int) -> dict:
    settings = Settings.load()
    standby = fallback_gateway(settings)
    if standby is None:
        raise RuntimeError("standby_model_not_configured")
    try:
        return await score(standby, EvalLimits(max_model_calls=max_calls, max_output_tokens=max_calls*1900, per_call_output_tokens=1900),
                           settings.fallback_model)
    finally:
        await standby.close()


def main():
    parser = argparse.ArgumentParser(description="Run the role smoke cases on the standby vendor (paid); the primary model is simulated as down")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-calls", type=int, default=40, help="hard cap on standby model calls")
    args = parser.parse_args()
    if not 1 <= args.max_calls <= 80:
        parser.error("--max-calls must be between 1 and 80")
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("preserve_existing_standby_report:"+str(target))
    report = asyncio.run(run_online(args.max_calls))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(f'completed {report["completed"]}/{report["total"]}; served by standby {report["served_by_standby"]}; usage {report["usage"]}; '
          f'budget stop {report["budget_stop"]}. Saved: {target}')


if __name__ == "__main__":
    main()
