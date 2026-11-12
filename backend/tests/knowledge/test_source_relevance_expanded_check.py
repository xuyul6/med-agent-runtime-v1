"""Test isolation and reporting, not the current relevance outcomes."""
import json

import pytest

from medagent.core.config import ROOT
from medagent.evaluation.isolation import EvaluationIsolation, ReferenceSnapshotError
from support.source_relevance_expanded_check import run_check
from support.harness import run


def test_expanded_check_preserves_all_cases_and_marks_unverified_sources():
    data = json.loads((ROOT/"resources/source-relevance-expanded.json").read_text())
    result = run(run_check())
    assert result["model_calls"] == result["judge_calls"] == 0 and not result["cleanup_errors"]
    assert len(result["retrieval"]["rows"]) == len(data["retrieval_cases"])
    assert len(result["institution"]["rows"]) == len(data["institution_cases"])
    assert result["retrieval"]["matched_expectation"] == sum(row["matched_expectation"] for row in result["retrieval"]["rows"])
    assert result["institution"]["matched_expectation"] == sum(row["matched_expectation"] for row in result["institution"]["rows"])
    for row in result["retrieval"]["rows"]:
        assert set(row["actual_keys"]) <= set(row["reference_snapshot"]["keys"])
        assert set(row["unexpected_keys"]) == set(row["actual_keys"])-set(row["expected_keys"])
        assert set(row["missing_keys"]) == set(row["expected_keys"])-set(row["actual_keys"])
        for key in row["actual_keys"]:
            assert row["actual_source_purposes"][key] == data["source_purposes"][key]
    assert "不代表资料真实、官方" in result["labels_note"]


def test_expanded_check_rejects_changed_snapshot_before_creating_collection(tmp_path, monkeypatch):
    data = json.loads((ROOT/"resources/source-relevance-expanded.json").read_text())
    data["references"][0]["text"] += " 已变更"
    path = tmp_path/"changed.json"
    path.write_text(json.dumps(data,ensure_ascii=False))
    async def forbidden(*args): raise AssertionError("collection_must_not_be_created")
    monkeypatch.setattr(EvaluationIsolation,"create",forbidden)
    with pytest.raises(ReferenceSnapshotError,match="snapshot_hash_mismatch"):
        run(run_check(path))
