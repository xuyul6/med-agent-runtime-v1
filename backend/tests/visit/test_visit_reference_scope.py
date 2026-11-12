"""Purpose metadata is an explicit, unverified claim, not proof of provenance."""
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from medagent.app import create_app
from medagent.agents.contracts import VisitInput
from medagent.core.config import Settings
from medagent.core.contracts import Intent, Reference
from medagent.evaluation.runner import reference_snapshot_hash, snapshot_preflight
from medagent.evaluation.budget import ModelBudget
from medagent.evaluation.contracts import DialogCase, EvalLimits
from medagent.evaluation.isolation import EvaluationIsolation, ReferenceSnapshotError
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.knowledge.retrieval import ReferenceIndex, Vectorizer, catalog_updates
from medagent.tools.registry import ToolBudget, ToolContext, ToolResultError
from support.harness import Model, fixture, role_output, run, role_data


def visit_ref(key, purpose=None, approved=True):
    kwargs = {"visit_scope":{"purpose":purpose}} if purpose else {}
    return Reference(key=key, title="就医准备清单", domain=Intent.VISIT, text="虚构就医准备资料",
                     url="https://example.org/"+key, version="test", scope="虚构测试", approved=approved, **kwargs)


@pytest.mark.parametrize("scope", [
    {"purpose":"other"}, {"purpose":"general_preparation", "official":True}, {},
])
def test_invalid_visit_scope_is_rejected(scope):
    data = visit_ref("example").model_dump(mode="json")
    data["visit_scope"] = scope
    with pytest.raises(ValidationError):
        Reference.model_validate(data)


def test_visit_scope_cannot_be_attached_to_another_domain():
    data = visit_ref("example", "general_preparation").model_dump(mode="json")
    data["domain"] = "report"
    with pytest.raises(ValidationError, match="visit_scope_requires_visit_domain"):
        Reference.model_validate(data)


def test_old_reference_serialization_and_snapshot_remain_reproducible():
    runtime = run(fixture())
    current = runtime.index.documents["consultation-record"]
    legacy = current.model_copy(update={"visit_scope":None})
    assert "visit_scope" not in legacy.model_dump(mode="json")
    assert "visit_scope" not in json.loads(legacy.model_dump_json())
    assert reference_snapshot_hash([legacy]) == "4dd3686215bc5e7a92f82ebb672298af0f401f2bc43e6687aecc91b475d648ca"
    assert reference_snapshot_hash([current]) != reference_snapshot_hash([legacy])


def test_only_unchanged_approved_builtin_template_is_upgraded():
    runtime = run(fixture())
    current = runtime.index.documents["consultation-record"]
    legacy = current.model_copy(update={"visit_scope":None})
    assert catalog_updates([current], {current.key:legacy}) == [current]
    for patch in [{"text":legacy.text+"用户改动"}, {"version":"user-version"}, {"approved":False}]:
        assert catalog_updates([current], {current.key:legacy.model_copy(update=patch)}) == []
    marked = Reference.model_validate({**legacy.model_dump(), "visit_scope":{"purpose":"institution_specific"}})
    assert catalog_updates([current], {current.key:marked}) == []
    other = visit_ref("other", "general_preparation")
    assert catalog_updates([other], {other.key:other.model_copy(update={"visit_scope":None})}) == []


def test_stale_builtin_snapshot_is_rejected_before_model_or_isolation():
    async def scenario():
        runtime = await fixture(Model())
        isolation = EvaluationIsolation(runtime, "visit-scope-snapshot", ModelBudget(EvalLimits()))
        old = "4dd3686215bc5e7a92f82ebb672298af0f401f2bc43e6687aecc91b475d648ca"
        case = DialogCase.model_validate({"id":"stale-visit-snapshot", "turns":["就医准备"],
            "reference_snapshot":{"keys":["consultation-record"], "sha256":old}})
        with pytest.raises(ReferenceSnapshotError, match="snapshot_hash_mismatch"):
            snapshot_preflight(isolation, [case])
        assert not runtime.monitor.requests

    run(scenario())


def test_general_visit_search_filters_undeclared_institution_and_drafts():
    async def scenario():
        index = ReferenceIndex(Vectorizer("hash"), None)
        refs = [visit_ref("general", "general_preparation"), visit_ref("institution", "institution_specific"),
                visit_ref("unknown"), visit_ref("draft", "general_preparation", approved=False)]
        await index.upsert(refs)
        found = await index.search("就医准备清单", "visit")
        assert [ref.key for ref in found] == ["general"]
        assert "visit_scope" not in index.documents["unknown"].model_dump()

    run(scenario())


def test_changing_visit_purpose_invalidates_cached_results():
    async def scenario():
        index = ReferenceIndex(Vectorizer("hash"), None)
        await index.upsert([visit_ref("changed", "general_preparation")])
        assert [ref.key for ref in await index.search("就医准备清单", "visit")] == ["changed"]
        await index.upsert([visit_ref("changed", "institution_specific")])
        assert await index.search("就医准备清单", "visit") == []

    run(scenario())


def test_knowledge_api_round_trips_scope_and_search_respects_it():
    runtime = run(fixture())
    general = visit_ref("synthetic-general", "general_preparation")
    institution = visit_ref("synthetic-institution", "institution_specific")
    unknown = visit_ref("synthetic-unmarked")
    with TestClient(create_app(Settings(developer_tools=True, preview=False, embeddings="hash"), runtime)) as client:
        response = client.post("/knowledge", json={"documents":[ref.model_dump(mode="json") for ref in (general, institution, unknown)]})
        assert response.status_code == 200 and response.json()["saved"] == 3
        catalog = {ref["key"]:ref for ref in client.get("/knowledge").json()["documents"]}
        assert catalog[general.key]["visit_scope"]["purpose"] == "general_preparation"
        assert catalog[institution.key]["visit_scope"]["purpose"] == "institution_specific"
        assert "visit_scope" not in catalog[unknown.key]
    found = {ref.key for ref in run(runtime.index.search("就医准备清单", "visit"))}
    assert general.key in found and institution.key not in found and unknown.key not in found


def test_visit_purpose_survives_chroma_restore_and_filters_after_restart():
    async def scenario():
        runtime = await fixture()
        await runtime.index.upsert([visit_ref("stored-general", "general_preparation"),
                                    visit_ref("stored-institution", "institution_specific"), visit_ref("stored-unknown")])
        isolation = EvaluationIsolation(runtime, "visit-scope-persistence", ModelBudget(EvalLimits()))
        try:
            await isolation.create()
            restored = ReferenceIndex(Vectorizer("hash"), None, isolation.index.collection)
            await restored.restore()
            assert restored.documents["stored-general"].visit_scope.purpose == "general_preparation"
            assert restored.documents["stored-institution"].visit_scope.purpose == "institution_specific"
            assert restored.documents["stored-unknown"].visit_scope is None
            found = {ref.key for ref in await restored.search("就医准备清单", "visit")}
            assert "stored-general" in found
            assert not {"stored-institution", "stored-unknown"} & found
        finally:
            await isolation.cleanup()
        assert not isolation.cleanup_errors

    run(scenario())


@pytest.mark.parametrize("purpose,code", [(None, "visit_scope_missing"), ("institution_specific", "visit_scope_not_general")])
def test_untrusted_index_cannot_bypass_scope_and_reports_reason(purpose, code):
    class ForgedIndex:
        async def search(self, query, domain): return [visit_ref("other", purpose)]

    async def scenario():
        runtime = await fixture()
        context = ToolContext(DEFAULT_REGISTRY["visit_guide"],
            VisitInput(text="就医准备", intent=Intent.VISIT), ForgedIndex(), evidence_query="就医准备清单")
        with pytest.raises(ToolResultError, match=code):
            await runtime.tools.call("lookup_reference", {"query":"就医准备", "domain":"visit"}, context, ToolBudget())
        assert not context.references
        assert runtime.monitor.snapshot()["tool_stats"]["lookup_reference"]["quality_counts"] == {code:1}

    run(scenario())


def test_full_context_contains_only_declared_general_visit_references():
    class Capture(Model):
        async def message(self, stage, system, messages, tools=None, budget=1800):
            if stage == "agent_visit_guide":
                self.references = role_data(messages)["references"]
                return {"content":[{"type":"text", "text":json.dumps(role_output("visit_guide", ["consultation-record"]))}]}
            return await super().message(stage, system, messages, tools, budget)

    async def scenario():
        model = Capture()
        runtime = await fixture(model)
        await runtime.index.upsert([visit_ref("institution", "institution_specific"), visit_ref("unknown")])
        result = await runtime.agents["visit_guide"][0].run(
            "就医前要准备什么？", Intent.VISIT, {}, "full", original_request="就医前要准备什么？")
        assert result.execution.status == "answered"
        assert {ref["key"] for ref in model.references} == {"consultation-record"}

    run(scenario())
