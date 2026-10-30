"""Routing check against labelled requests.

Offline (default) uses the keyword and vector channels only. With a model, only the understanding stage calls it:
role execution stays in preview, so the paid calls are one per turn and are capped by an explicit budget.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from ..core.config import ROOT, Settings
from ..core.provider import ClaudeGateway
from ..core.contracts import Question, Reference
from ..knowledge.retrieval import ReferenceIndex, Vectorizer
from ..memory import SessionMemory
from ..runtime import MedicalRuntime
from .budget import BudgetedModel, ModelBudget, aggregate_usage
from .contracts import EvalLimits
from .runner import CODE_HASH

DATASET = ROOT/"resources"/"routing-labels-v1.json"
ROLES = ("general", "symptom_guide", "drug_reference", "report_explainer", "visit_guide", "risk_handoff")


async def preview_runtime(embeddings: str = "hash", language_model=None) -> MedicalRuntime:
    """Preview roles and storage; `language_model` feeds only the classifier's language channel."""
    vectors = Vectorizer(embeddings)
    index = ReferenceIndex(vectors, None)
    await index.upsert([Reference.model_validate(x) for x in json.loads((ROOT/"resources"/"references.json").read_text())])
    runtime = MedicalRuntime(Settings(preview=True, embeddings=embeddings), None, vectors, index, SessionMemory())
    runtime.router.model = language_model
    return runtime


async def route_case(runtime: MedicalRuntime, case: dict):
    """Earlier turns build session state; only the last turn is scored."""
    for text in case["turns"]:
        reply = await runtime.chat(Question(text=text, user="routing-"+case["id"], session="labels"))
    return reply


def rate(hits: int, total: int) -> float | None:
    return round(hits/total, 4) if total else None


def summarize(rows: list[dict]) -> dict:
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    confusion = {expected:{actual:0 for actual in ROLES} for expected in ROLES}
    for row in rows:
        if len(row["expected_primary"]) == 1:
            confusion[row["expected_primary"][0]][row["actual_primary"]] += 1
    expected_support = sum(len(row["expected_roles"])-1 for row in rows)
    found_support = sum(len(set(row["expected_roles"]) & set(row["actual_roles"]))-1
                        for row in rows if row["actual_primary"] in row["expected_roles"])
    return {
        "cases":len(rows),
        "primary_accuracy":rate(sum(row["primary_ok"] for row in rows), len(rows)),
        "role_set_accuracy":rate(sum(row["roles_ok"] for row in rows), len(rows)),
        "support_recall":{"expected":expected_support, "admitted":found_support, "rate":rate(found_support, expected_support)},
        "extra_roles":sum(len(set(row["actual_roles"]) - set(row["expected_roles"])) for row in rows),
        "by_category":{name:{"cases":len(items), "primary_ok":sum(row["primary_ok"] for row in items),
                             "roles_ok":sum(row["roles_ok"] for row in items)} for name, items in sorted(by_category.items())},
        "single_label_confusion":confusion,
        "mismatches":[row["id"] for row in rows if not row["roles_ok"] or not row["primary_ok"]],
    }


async def score(dataset: Path, embeddings: str = "hash", model=None, limits: EvalLimits | None = None,
                model_name: str | None = None) -> dict:
    raw = dataset.read_bytes()
    body = json.loads(raw)
    ids = Counter(case["id"] for case in body["cases"])
    if duplicates := [key for key, count in ids.items() if count > 1]:
        raise ValueError("duplicate_case_ids:"+",".join(duplicates))
    budget = ModelBudget(limits or EvalLimits(max_model_calls=120, max_output_tokens=120*900, per_call_output_tokens=900))
    runtime = await preview_runtime(embeddings, BudgetedModel(model, budget) if model is not None else None)
    rows = []
    for case in body["cases"]:
        if not set(case["primary"]) <= set(case["roles"]) <= set(ROLES):
            raise ValueError("invalid_case_labels:"+case["id"])
        reply = await route_case(runtime, case)
        actual = sorted([reply.primary_agent, *reply.supporting_agents])
        rows.append({"id":case["id"], "turns":case["turns"], "category":case["category"],
                     "expected_primary":case["primary"], "expected_roles":case["roles"],
                     "actual_primary":reply.primary_agent, "actual_roles":actual,
                     "intent":reply.intent.primary.value, "tasks":[task.intent.value for task in reply.intent.tasks],
                     "language_channel":"language" in reply.intent.signals,
                     "primary_ok":reply.primary_agent in case["primary"], "roles_ok":actual == case["roles"]})
    usage = aggregate_usage(budget.entries)
    online = model is not None
    # Turns where the language channel fell back are kept in the scores and counted, not dropped.
    missing = sum(not row["language_channel"] for row in rows if row["actual_primary"] != "risk_handoff")
    return {"kind":"online_routing_labels" if online else "offline_routing_labels", "dataset":body["version"],
            "dataset_sha256":hashlib.sha256(raw).hexdigest(), "code_sha256":CODE_HASH,
            "channels":["keywords", "embedding:"+runtime.router.vectors.identity, *(["language:"+(model_name or "model")] if online else [])],
            "language_channel":("scored_cases_without_language:"+str(missing)) if online else "not_used_offline",
            "labels":"engineering labels written before rule changes",
            "paid_model_calls":usage["attempted_calls"] if online else 0, "usage":usage, "budget_stop":budget.exhausted,
            "failed_model_calls":sum(entry["error"] is not None for entry in budget.entries),
            "generated_at":datetime.now(timezone.utc).isoformat(), "summary":summarize(rows), "cases":rows}


def run(dataset: Path = DATASET, embeddings: str = "hash") -> dict:
    return asyncio.run(score(dataset, embeddings))


async def run_online(dataset: Path, max_calls: int) -> dict:
    settings = Settings.load()
    if settings.preview or not settings.api_key or not settings.model:
        raise RuntimeError("online_routing_needs_configured_model")
    gateway = ClaudeGateway(settings)
    try:
        return await score(dataset, settings.embeddings, gateway,
                           EvalLimits(max_model_calls=max_calls, max_output_tokens=max_calls*900, per_call_output_tokens=900),
                           settings.model)
    finally:
        await gateway.close()


def main():
    parser = argparse.ArgumentParser(description="Routing check against labelled requests; offline unless --online")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output", required=True)
    parser.add_argument("--embeddings", choices=["hash", "onnx"], default="hash", help="offline vector channel")
    parser.add_argument("--online", action="store_true", help="add the configured model as the language channel (paid)")
    parser.add_argument("--max-calls", type=int, default=0, help="required with --online: hard cap on model calls")
    args = parser.parse_args()
    if args.online and not 1 <= args.max_calls <= 400:
        parser.error("--online needs --max-calls between 1 and 400")
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("preserve_existing_routing_report:"+str(target))
    report = asyncio.run(run_online(Path(args.dataset), args.max_calls)) if args.online else run(Path(args.dataset), args.embeddings)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    summary = report["summary"]
    print(f'primary {summary["primary_accuracy"]}; role set {summary["role_set_accuracy"]}; '
          f'support {summary["support_recall"]["admitted"]}/{summary["support_recall"]["expected"]}; '
          f'extra roles {summary["extra_roles"]}; model calls {report["paid_model_calls"]}; '
          f'budget stop {report["budget_stop"]}. Saved: {target}')


if __name__ == "__main__":
    main()
