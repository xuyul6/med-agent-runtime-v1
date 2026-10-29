"""Offline routing check against labelled requests: keyword and hash-vector channels only, no model or storage."""
import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from ..core.config import ROOT, Settings
from ..core.contracts import Question, Reference
from ..knowledge.retrieval import ReferenceIndex, Vectorizer
from ..memory import SessionMemory
from ..runtime import MedicalRuntime
from .runner import CODE_HASH

DATASET = ROOT/"resources"/"routing-labels-v1.json"
ROLES = ("general", "symptom_guide", "drug_reference", "report_explainer", "visit_guide", "risk_handoff")


async def preview_runtime() -> MedicalRuntime:
    vectors = Vectorizer("hash")
    index = ReferenceIndex(vectors, None)
    await index.upsert([Reference.model_validate(x) for x in json.loads((ROOT/"resources"/"references.json").read_text())])
    return MedicalRuntime(Settings(preview=True, embeddings="hash"), None, vectors, index, SessionMemory())


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


async def score(dataset: Path) -> dict:
    raw = dataset.read_bytes()
    body = json.loads(raw)
    ids = Counter(case["id"] for case in body["cases"])
    if duplicates := [key for key, count in ids.items() if count > 1]:
        raise ValueError("duplicate_case_ids:"+",".join(duplicates))
    runtime = await preview_runtime()
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
                     "primary_ok":reply.primary_agent in case["primary"], "roles_ok":actual == case["roles"]})
    return {"kind":"offline_routing_labels", "dataset":body["version"], "dataset_sha256":hashlib.sha256(raw).hexdigest(),
            "code_sha256":CODE_HASH, "channels":["keywords", "embedding:"+runtime.router.vectors.identity],
            "language_channel":"not_used_offline", "labels":"engineering labels written before rule changes",
            "paid_model_calls":0, "generated_at":datetime.now(timezone.utc).isoformat(),
            "summary":summarize(rows), "cases":rows}


def run(dataset: Path = DATASET) -> dict:
    return asyncio.run(score(dataset))


def main():
    parser = argparse.ArgumentParser(description="Offline routing check against labelled requests; no model calls")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("preserve_existing_routing_report:"+str(target))
    report = run(Path(args.dataset))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    summary = report["summary"]
    print(f'primary {summary["primary_accuracy"]}; role set {summary["role_set_accuracy"]}; '
          f'support {summary["support_recall"]["admitted"]}/{summary["support_recall"]["expected"]}; '
          f'extra roles {summary["extra_roles"]}. Saved: {target}')


if __name__ == "__main__":
    main()
