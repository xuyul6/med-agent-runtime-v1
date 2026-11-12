"""Offline synthetic multi-source retrieval and institution-rule observations."""
import argparse
import asyncio
import json
from pathlib import Path

from medagent.core.config import ROOT, Settings
from medagent.core.contracts import Intent, Reference, TaskUnderstanding, Understanding
from medagent.understanding.conversation import role_details
from medagent.evaluation.runner import CODE_HASH, reference_snapshot_hash, snapshot_preflight
from medagent.evaluation.budget import ModelBudget
from medagent.evaluation.contracts import DialogCase, EvalLimits
from medagent.evaluation.isolation import EvaluationIsolation
from medagent.evaluation.metrics import digest
from medagent.memory import SessionMemory
from medagent.knowledge.retrieval import ReferenceIndex, Vectorizer
from medagent.runtime import MedicalRuntime


async def run_check(dataset=None):
    path = Path(dataset) if dataset is not None else ROOT/"resources/source-relevance-expanded.json"
    data = json.loads(path.read_text())
    refs = [Reference.model_validate(item) for item in data["references"]]
    if len({ref.key for ref in refs}) != len(refs): raise ValueError("duplicate_reference_key")
    purposes = data["source_purposes"]
    if set(purposes) != {ref.key for ref in refs}: raise ValueError("reference_purposes_incomplete")
    vectors = Vectorizer("hash")
    source_index = ReferenceIndex(vectors, None)
    await source_index.upsert(refs)
    source = MedicalRuntime(Settings(preview=True, embeddings="hash"), None, vectors, source_index, SessionMemory())
    isolation = EvaluationIsolation(source, "source-relevance-expanded", ModelBudget(EvalLimits()))
    cases = data["retrieval_cases"]
    selections = [DialogCase.model_validate({"id":case["id"], "turns":[case["query"]],
                  "reference_snapshot":data["snapshots"][case["domain"]]}) for case in cases]
    snapshot_preflight(isolation, selections)
    retrieval = []
    try:
        await isolation.create()
        for case, selection in zip(cases, selections):
            keys = selection.reference_snapshot.keys
            if set(case["expected_keys"])-set(keys): raise ValueError("expected_reference_outside_snapshot")
            isolation.select_case_references(keys)
            actual = sorted(ref.key for ref in await isolation.index.search(case["query"],case["domain"]))
            expected = sorted(case["expected_keys"])
            retrieval.append({"id":case["id"], "domain":case["domain"], "query":case["query"],
                "expected_keys":expected, "actual_keys":actual, "matched_expectation":actual == expected,
                "actual_source_purposes":{key:purposes[key] for key in actual},
                "unexpected_keys":sorted(set(actual)-set(expected)), "missing_keys":sorted(set(expected)-set(actual)),
                "reference_snapshot":data["snapshots"][case["domain"]]})
    finally:
        await isolation.cleanup()
    if isolation.cleanup_errors: raise RuntimeError("controlled_reference_cleanup_failed")
    institution = []
    for case in data["institution_cases"]:
        understanding = Understanding(primary=Intent.VISIT, confidence=1, scores={"visit":1},
            tasks=[TaskUnderstanding(intent=Intent.VISIT, goal=case["query"], priority="primary")])
        decision = source.planner.plan(case["query"], understanding, source.agents, source.monitor)
        assignment = next(item for item in decision.assignments if item.role == "visit_guide")
        scoped = understanding.model_copy(update={"tasks":[TaskUnderstanding(intent=Intent.VISIT, goal=assignment.input_text[:300])]})
        actual = role_details(scoped, "visit_guide").get("institution_specific", False)
        institution.append({"id":case["id"], "query":case["query"], "expected_guard":case["expected_guard"],
                            "actual_guard":actual, "matched_expectation":actual == case["expected_guard"]})
    return {"schema_version":"source-relevance-expanded-result-1", "dataset_version":data["version"],
        "dataset_sha256":digest(data), "catalog_sha256":reference_snapshot_hash(refs), "code_hash":CODE_HASH,
        "model_calls":0, "judge_calls":0, "embedding":"hash_preview_v1", "kind":"synthetic_retrieval_and_rule_only",
        "labels_note":data["label_note"],
        "retrieval":{"total":len(retrieval), "matched_expectation":sum(row["matched_expectation"] for row in retrieval), "rows":retrieval},
        "institution":{"total":len(institution), "matched_expectation":sum(row["matched_expectation"] for row in institution), "rows":institution},
        "cleanup_errors":isolation.cleanup_errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise FileExistsError("preserve_existing_report")
    result = asyncio.run(run_check(args.dataset))
    with output.open("x") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")
    print(f'retrieval {result["retrieval"]["matched_expectation"]}/{result["retrieval"]["total"]}; '
          f'institution rule {result["institution"]["matched_expectation"]}/{result["institution"]["total"]}; '
          f'model calls=0; saved {output}')
