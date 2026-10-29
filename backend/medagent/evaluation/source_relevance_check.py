"""Reproducible offline retrieval observations with frozen public-reference fixtures."""
import argparse
import asyncio
import json
from pathlib import Path

from ..core.config import ROOT, Settings
from ..core.contracts import Reference
from .runner import CODE_HASH, reference_snapshot_hash, snapshot_preflight
from .budget import ModelBudget
from .contracts import DialogCase, EvalLimits
from .isolation import EvaluationIsolation
from .metrics import digest
from ..memory import SessionMemory
from ..knowledge.retrieval import ReferenceIndex, Vectorizer
from ..runtime import MedicalRuntime


async def run_check(dataset=None):
    dataset = Path(dataset) if dataset is not None else ROOT/"resources/source-relevance-controlled.json"
    data = json.loads(dataset.read_text())
    cases = data["cases"]
    records = [Reference.model_validate(item) for item in json.loads((ROOT/"resources/references.json").read_text())]
    vectors = Vectorizer("hash")
    source_index = ReferenceIndex(vectors, None)
    await source_index.upsert(records)
    source = MedicalRuntime(Settings(preview=True, embeddings="hash"), None, vectors, source_index, SessionMemory())
    isolation = EvaluationIsolation(source, "source-relevance-controlled", ModelBudget(EvalLimits()))
    selections = [DialogCase.model_validate({"id":case["id"], "turns":[case["query"]],
                  "reference_snapshot":case["reference_snapshot"]}) for case in cases]
    snapshot_preflight(isolation, selections)
    rows = []
    try:
        await isolation.create()
        for case, selection in zip(cases, selections):
            isolation.select_case_references(selection.reference_snapshot.keys)
            found = await isolation.index.search(case["query"], case["domain"])
            keys = [ref.key for ref in found]
            rows.append({"id":case["id"], "domain":case["domain"], "query":case["query"],
                         "expected_recall":case["expected_recall"], "actual_keys":keys,
                         "matched_expectation":bool(keys) == case["expected_recall"],
                         "reference_snapshot":case["reference_snapshot"]})
    finally:
        await isolation.cleanup()
    if isolation.cleanup_errors: raise RuntimeError("controlled_reference_cleanup_failed")
    return {"schema_version":"source-relevance-controlled-result-1", "dataset_version":data["version"],
            "dataset_sha256":digest(data), "catalog_sha256":reference_snapshot_hash(records), "code_hash":CODE_HASH,
            "model_calls":0, "judge_calls":0, "embedding":"hash_preview_v1", "kind":"isolated_retrieval_only",
            "total":len(rows), "matched_expectation":sum(row["matched_expectation"] for row in rows),
            "rows":rows, "cleanup_errors":isolation.cleanup_errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", help="optional local controlled-case JSON")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise FileExistsError("preserve_existing_report")
    result = asyncio.run(run_check(args.dataset))
    with output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f'{result["matched_expectation"]}/{result["total"]} retrieval expectations matched; model calls=0; saved {output}')
