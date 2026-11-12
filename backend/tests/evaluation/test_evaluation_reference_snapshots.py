"""Controlled reference sets exercise production retrieval without paid models."""
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from medagent.app import create_app
from medagent.core.config import Settings
from medagent.evaluation.runner import evaluate, reference_snapshot_hash
from medagent.evaluation.budget import ModelBudget
from medagent.evaluation.contracts import EvaluationRequest
from medagent.evaluation.isolation import EvaluationIsolation, ReferenceSnapshotError
from medagent.evaluation.metrics import BaselineError, BaselineStore
from medagent.evaluation.reference_snapshot import make_reference_snapshot
from medagent.core.contracts import Intent, Reference
from support.harness import fixture, run, role_data
from support.subject_switch import CountStages


def selected(records, keys):
    return {"keys":keys, "sha256":reference_snapshot_hash([records[key] for key in keys])}


def test_snapshot_builder_sorts_keys_and_validates_missing_or_duplicates():
    runtime = run(fixture())
    records = list(runtime.index.documents.values())
    keys = ["lab-context", "ibuprofen-overview"]
    snapshot = make_reference_snapshot(records, keys)
    assert snapshot == selected(runtime.index.documents, sorted(keys))
    assert make_reference_snapshot(records, []) == {"keys":[], "sha256":reference_snapshot_hash([])}
    with pytest.raises(ValueError, match="snapshot_reference_missing"):
        make_reference_snapshot(records, ["missing"])
    with pytest.raises(ValueError, match="duplicate_reference_key"):
        make_reference_snapshot(records, ["lab-context", "lab-context"])


def cases_body(runtime):
    refs = runtime.index.documents
    return EvaluationRequest(intent_cases=[], judge=False, dialog_cases=[
        {"id":"absent", "reference_snapshot":selected(refs, []), "turns":[
            {"message":"查询布洛芬公开资料。", "expected":{"role":"drug_reference", "missing_fields":["source"], "references":False}}]},
        {"id":"present", "reference_snapshot":selected(refs, ["ibuprofen-overview"]), "turns":[
            {"message":"查询布洛芬公开资料。", "expected":{"role":"drug_reference", "status":"answered", "references":True}}]},
        {"id":"default", "turns":[
            {"message":"查询布洛芬公开资料。", "expected":{"role":"drug_reference", "status":"answered", "references":True}}]},
    ])


def test_empty_selected_and_default_cases_are_isolated_without_touching_production():
    async def scenario():
        model = CountStages(); runtime = await fixture(model)
        before = set(runtime.index.documents)
        result = await evaluate(runtime, cases_body(runtime))
        assert result["complete"] and result["passed"] == 3
        assert result["metadata"]["case_reference_scope_complete"] is False
        assert result["metadata"]["case_reference_snapshots"]["absent"]["keys"] == []
        first, second, third = [case["turns"][0]["reply"] for case in result["cases"]]
        assert not first["references"] and first["agent_results"][0]["missing_fields"] == ["source"]
        assert [ref["key"] for ref in second["references"]] == ["ibuprofen-overview"]
        assert [ref["key"] for ref in third["references"]] == ["ibuprofen-overview"]
        assert set(runtime.index.documents) == before and not runtime.monitor.requests
        assert result["isolation"]["temporary_database_removed"] and not result["cleanup_errors"]
    run(scenario())


@pytest.mark.parametrize("fault", ["missing", "content_changed"])
def test_bad_snapshot_is_rejected_before_calls_or_isolation(fault):
    async def scenario():
        model = CountStages(); runtime = await fixture(model)
        body = cases_body(runtime)
        if fault == "missing": runtime.index.documents.pop("ibuprofen-overview")
        else: runtime.index.documents["ibuprofen-overview"].text += " 内容已变化"
        with pytest.raises(ReferenceSnapshotError, match="snapshot_reference_missing|snapshot_hash_mismatch"):
            await evaluate(runtime, body)
        assert model.stages == [] and not runtime.memory.local and not runtime.monitor.requests
    run(scenario())


def test_chroma_recall_is_restricted_to_selected_reference_keys():
    async def scenario():
        runtime = await fixture()
        budget = ModelBudget(EvaluationRequest().limits)
        isolation = EvaluationIsolation(runtime, "snapshot-chroma-check", budget)
        try:
            await isolation.create()
            isolation.select_case_references(["ibuprofen-overview"])
            original = isolation.index.collection
            class Spy:
                def query(self, **kwargs):
                    self.where = kwargs["where"]
                    self.result = original.query(**kwargs)
                    return self.result
            spy = Spy(); isolation.index.collection = spy
            docs = await isolation.index.search("布洛芬", "medicine")
            assert [doc.key for doc in docs] == ["ibuprofen-overview"]
            assert {"reference_key":{"$in":["ibuprofen-overview"]}} in spy.where["$and"]
            assert spy.result["ids"] == [["ibuprofen-overview"]]
        finally:
            await isolation.cleanup()
        assert not isolation.cleanup_errors
    run(scenario())


def test_snapshot_is_unchanged_if_production_catalog_changes_mid_run():
    async def scenario():
        runtime = await fixture()
        original = runtime.index.documents["lab-context"].text
        isolation = EvaluationIsolation(runtime, "snapshot-freeze-check", ModelBudget(EvaluationRequest().limits))
        try:
            await isolation.create()
            runtime.index.documents["lab-context"].text = "用户后来修改了这份资料"
            isolation.select_case_references(["lab-context"])
            assert isolation.index.documents["lab-context"].text == original
        finally:
            await isolation.cleanup()
        assert not isolation.cleanup_errors
    run(scenario())


def test_full_context_sees_only_the_selected_case_snapshot():
    class Capture(CountStages):
        async def message(self, stage, system, messages, tools=None, budget=1800):
            if stage == "agent_report_explainer":
                import json
                self.references = role_data(messages)["references"]
            return await super().message(stage, system, messages, tools, budget)
    async def scenario():
        model = Capture(); runtime = await fixture(model)
        await runtime.index.upsert([Reference(key="synthetic-second-report", title="报告参考范围补充", domain="report",
            text="参考范围的合成条目。", url="https://example.org/second", version="test", scope="合成测试", approved=True)])
        isolation = EvaluationIsolation(runtime, "snapshot-full-check", ModelBudget(EvaluationRequest().limits))
        try:
            await isolation.create()
            isolation.select_case_references(["lab-context"])
            case_runtime = isolation.new_case()
            result = await case_runtime.agents["report_explainer"][0].run("报告参考范围是什么？", Intent.REPORT, {}, "full")
            assert result.execution.status == "answered"
            assert [ref["key"] for ref in model.references] == ["lab-context"]
        finally:
            await isolation.cleanup()
        assert not isolation.cleanup_errors
    run(scenario())


@pytest.mark.parametrize("snapshot", [{"keys":["same", "same"], "sha256":"0"*64},
    {"keys":[], "sha256":"bad"}, {"keys":[], "sha256":"0"*64, "unknown":True}])
def test_invalid_snapshot_input_is_rejected(snapshot):
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate({"intent_cases":[], "dialog_cases":[{"turns":["你好"], "reference_snapshot":snapshot}]})


def test_unspecified_snapshot_round_trips_as_unrestricted_not_empty():
    request = EvaluationRequest(intent_cases=[], dialog_cases=[{"turns":["你好"]}])
    parsed = EvaluationRequest.model_validate(request.model_dump(mode="json"))
    assert parsed.dialog_cases[0].reference_snapshot is None


def test_snapshot_mismatch_http_409_with_no_model_calls():
    runtime = run(fixture(CountStages()))
    body = {"intent_cases":[], "judge":False, "dialog_cases":[{"turns":["你好"],
            "reference_snapshot":{"keys":[], "sha256":"0"*64}}]}
    with TestClient(create_app(Settings(developer_tools=True, preview=False, embeddings="hash"), runtime)) as client:
        response = client.post("/eval/run", json=body)
    assert response.status_code == 409
    assert response.json()["detail"] == {"code":"reference_snapshot_rejected", "reason":"snapshot_hash_mismatch"}
    assert runtime.model.stages == []


def test_uncontrolled_baseline_rejects_changed_catalog_before_new_calls(tmp_path):
    async def scenario():
        model = CountStages(); runtime = await fixture(model); store = BaselineStore(tmp_path)
        body = EvaluationRequest(intent_cases=[], judge=False, dialog_cases=[{"id":"general", "turns":["你好"]}],
                                 baseline={"save_as":"original"})
        result = await evaluate(runtime,body,baseline_store=store)
        assert result["baseline"]["saved"]
        runtime.index.documents["ibuprofen-overview"].text += " 更新"
        model.stages.clear()
        compared = body.model_copy(update={"baseline":body.baseline.model_copy(update={"save_as":None,"compare_to":"original"})})
        with pytest.raises(BaselineError,match="baseline_reference_catalog_incompatible"):
            await evaluate(runtime,compared,baseline_store=store)
        assert model.stages == []
    run(scenario())


def test_fully_controlled_baseline_survives_an_unselected_catalog_addition(tmp_path):
    async def scenario():
        model = CountStages(); runtime = await fixture(model); store = BaselineStore(tmp_path)
        body = EvaluationRequest(intent_cases=[], judge=False, dialog_cases=[{"id":"selected", "turns":["查询布洛芬公开资料。"],
            "reference_snapshot":selected(runtime.index.documents,["ibuprofen-overview"])}], baseline={"save_as":"original"})
        first = await evaluate(runtime,body,baseline_store=store)
        assert first["baseline"]["saved"] and first["metadata"]["case_reference_scope_complete"]
        extra = Reference(key="unselected-extra", title="合成资料", domain="report", text="合成文本", url="https://example.org/ref",
                          version="test", scope="仅测试", approved=True)
        await runtime.index.upsert([extra])
        compared = body.model_copy(update={"baseline":body.baseline.model_copy(update={"save_as":None,"compare_to":"original"})})
        result = await evaluate(runtime,compared,baseline_store=store)
        assert result["comparison"]["status"] == "compared"
        assert first["metadata"]["references_hash"] != result["metadata"]["references_hash"]
        assert first["metadata"]["case_reference_snapshots"] == result["metadata"]["case_reference_snapshots"]
    run(scenario())


def test_mixed_controlled_and_default_cases_reject_catalog_change(tmp_path):
    async def scenario():
        model = CountStages(); runtime = await fixture(model); store = BaselineStore(tmp_path)
        base = cases_body(runtime).model_copy(update={"baseline":EvaluationRequest().baseline.model_copy(update={"save_as":"mixed"})})
        first = await evaluate(runtime,base,baseline_store=store)
        assert first["baseline"]["saved"] and not first["metadata"]["case_reference_scope_complete"]
        await runtime.index.upsert([Reference(key="mixed-extra", title="合成资料", domain="report", text="合成文本",
            url="https://example.org/ref", version="test", scope="仅测试", approved=True)])
        model.stages.clear()
        again = base.model_copy(update={"baseline":base.baseline.model_copy(update={"save_as":None,"compare_to":"mixed"})})
        with pytest.raises(BaselineError, match="baseline_reference_catalog_incompatible"):
            await evaluate(runtime,again,baseline_store=store)
        assert not model.stages
    run(scenario())
