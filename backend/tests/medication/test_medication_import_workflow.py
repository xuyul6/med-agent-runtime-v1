"""Synthetic API-to-tool medication import checks; no external services or model calls."""
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from medagent.agents.contracts import MedicationInput
from medagent.app import create_app
from medagent.core.config import Settings
from medagent.core.contracts import Intent
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.tools.registry import ToolBudget, ToolContext
from support.harness import fixture, run


def document(kind="public_overview"):
    label = kind == "product_label"
    return {
        "key": "synthetic-label" if label else "synthetic-overview",
        "title": "虚构药品资料", "domain": "medicine", "text": "测试专用来源摘要，无真实医疗内容。",
        "url": "https://example.org/synthetic", "version": "synthetic-v1", "scope": "测试范围",
        "approved": True, "entities": ["虚构药品甲"],
        "provenance": {
            "organization": "虚构来源机构", "document_type": kind, "source_version": "test-v1",
            "updated_on": "2026-01-01", "published_on": None, "checked_on": "2026-01-02",
            "sections": ["测试章节"], "use_conditions": "仅用于测试", "terms_url": "https://example.org/terms",
        },
        "medication_scope": {
            "region": "US", "population": "adults_18_plus", "label_population": "测试成人范围",
            "product_name": "虚构品牌" if label else None,
            "formulation": "胶囊" if label else None, "strength": "100mg" if label else None,
        },
    }


def portal_lookup(client, runtime, **details):
    async def lookup():
        request = MedicationInput(text="查询虚构药品甲", intent=Intent.MEDICINE, **details)
        context = ToolContext(DEFAULT_REGISTRY["drug_reference"], request, runtime.index)
        result = await runtime.tools.call("lookup_reference", {"query":"虚构药品甲", "domain":"medicine"},
                                          context, ToolBudget())
        scope = await runtime.tools.call("check_medication_scope", {}, context, ToolBudget())
        return result, scope
    return client.portal.call(lookup)


@pytest.fixture
def client_and_runtime():
    # The application is injected with an in-memory, hash-vector runtime with no model.
    runtime = run(fixture())
    runtime.index.documents.clear()
    with TestClient(create_app(Settings(developer_tools=True, preview=False, embeddings="hash"), runtime)) as client:
        yield client, runtime


def test_saved_overview_is_retrievable_but_not_individually_verified(client_and_runtime):
    client, runtime = client_and_runtime
    ref = document()
    response = client.post("/knowledge", json={"documents":[ref]})
    assert response.status_code == 200 and response.json() == {"saved":1}
    saved = client.get("/knowledge").json()["documents"]
    assert len(saved) == 1 and saved[0]["provenance"]["organization"] == "虚构来源机构"
    result, scope = portal_lookup(client, runtime, medication_name="虚构药品甲", query_type="overview")
    assert [item.key for item in result.documents] == ["synthetic-overview"]
    assert scope.applicability == "needs_verification"
    assert "不自动确认" in scope.note


def test_saved_product_label_requires_matching_scope(client_and_runtime):
    client, runtime = client_and_runtime
    assert client.post("/knowledge", json={"documents":[document("product_label")]}).status_code == 200
    details = {"medication_name":"虚构药品甲", "query_type":"product_information", "region":"美国",
               "population":"成人", "product_name":"虚构品牌", "formulation":"胶囊", "strength":"100毫克"}
    result, scope = portal_lookup(client, runtime, **details)
    assert [item.key for item in result.documents] == ["synthetic-label"]
    assert scope.applicability == "needs_verification"
    for field, incorrect in [("medication_name", "虚构药品乙"), ("region", "中国"), ("strength", "200mg")]:
        result, scope = portal_lookup(client, runtime, **{**details, field:incorrect})
        assert not result.documents and scope.applicability == "no_source"


@pytest.mark.parametrize("remove", [("provenance", "medication_scope"), ("medication_scope",)])
def test_saved_legacy_overview_without_metadata_is_filtered(client_and_runtime, remove):
    client, runtime = client_and_runtime
    ref = document()
    for field in remove:
        ref.pop(field)
    assert client.post("/knowledge", json={"documents":[ref]}).status_code == 200
    result, scope = portal_lookup(client, runtime, medication_name="虚构药品甲", query_type="overview")
    assert not result.documents and scope.applicability == "no_source"


def test_saved_draft_is_not_retrieved(client_and_runtime):
    client, runtime = client_and_runtime
    ref = document()
    ref["approved"] = False
    assert client.post("/knowledge", json={"documents":[ref]}).status_code == 200
    result, scope = portal_lookup(client, runtime, medication_name="虚构药品甲", query_type="overview")
    assert not result.documents and scope.applicability == "no_source"


@pytest.mark.parametrize("remove", [("provenance",), ("medication_scope",)])
def test_invalid_product_metadata_is_rejected_by_api(client_and_runtime, remove):
    client, runtime = client_and_runtime
    ref = deepcopy(document("product_label"))
    for field in remove:
        ref.pop(field)
    assert client.post("/knowledge", json={"documents":[ref]}).status_code == 422
    assert client.get("/knowledge").json()["documents"] == []


def test_product_label_without_strength_is_rejected_by_api(client_and_runtime):
    client, _ = client_and_runtime
    ref = document("product_label")
    ref["medication_scope"]["strength"] = None
    assert client.post("/knowledge", json={"documents":[ref]}).status_code == 422
    assert client.get("/knowledge").json()["documents"] == []
