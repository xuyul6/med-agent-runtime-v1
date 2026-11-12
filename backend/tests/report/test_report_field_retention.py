"""Deterministic report clarification retains only its validated input items."""
import json

import pytest
from fastapi.testclient import TestClient

from medagent.app import create_app
from medagent.core.config import Settings
from medagent.core.contracts import Intent
from support.harness import fixture, run
from support.multiturn import ContextModel, ask
from support.subject_switch import CountStages


def without_report_sources(runtime):
    runtime.index.documents = {key:ref for key,ref in runtime.index.documents.items() if ref.domain != Intent.REPORT}
    runtime.index.cache.clear()


@pytest.mark.parametrize("mode", ["selective", "full"])
def test_missing_source_retains_complete_input_without_explanations_or_citations(mode):
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        without_report_sources(runtime)
        item = {"name":"虚构项目A", "value":"10", "unit":"mmol/L", "reference_range":"3-5"}
        result = await runtime.agents["report_explainer"][0].run(
            "整理报告项目", Intent.REPORT, {}, mode, details={"items":[item]})
        assert result.execution.status == "needs_clarification" and result.execution.missing_fields == ["source"]
        assert result.execution.payload["report_items"] == [item]
        assert result.execution.payload["term_explanations"] == []
        assert not result.references and not result.answer.reference_keys
        assert "agent_report_explainer" not in model.stages

    run(scenario())


def test_field_preflight_retains_partial_values_and_does_not_retrieve():
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        result = await runtime.agents["report_explainer"][0].run(
            "整理报告项目", Intent.REPORT, {}, "selective", details={"items":[{"name":"虚构项目A", "value":"10"}]})
        assert result.execution.missing_fields == ["unit", "reference_range"]
        assert result.execution.payload["report_items"] == [
            {"name":"虚构项目A", "value":"10", "unit":None, "reference_range":None}]
        assert result.execution.payload["term_explanations"] == []
        assert runtime.monitor.snapshot()["tool_stats"]["lookup_reference"]["total"] == 0
        assert "agent_report_explainer" not in model.stages and not result.references

    run(scenario())


def report_updates(payload):
    text = payload["current"]["text"]
    values = [("report_item", "项目B"), ("value", "20")] if "项目B" in text else [
        ("report_item", "项目A"), ("value", "10"), ("unit", "mmol/L"), ("reference_range", "3-5")]
    return [{"name":name, "value":value, "state":"provided", "quote":value} for name,value in values]


def test_chat_keeps_grounded_fields_in_source_gap_without_logging_values():
    async def scenario():
        runtime = await fixture(ContextModel(report_updates))
        without_report_sources(runtime)
        reply = await ask(runtime, "报告项目A数值10，单位mmol/L，参考范围3-5。")
        result = reply.agent_results[0]
        assert reply.primary_agent == "report_explainer" and result.missing_fields == ["source"]
        assert result.payload["report_items"] == [{"name":"项目A", "value":"10", "unit":"mmol/L", "reference_range":"3-5"}]
        assert all(reply.intent.entities[name].source.turn_id == reply.request_id for name in ("report_item", "value", "unit", "reference_range"))
        assert not reply.references and not reply.answer.reference_keys and not result.payload["term_explanations"]
        # The random UUID can coincidentally contain the synthetic range "3-5".
        trace = json.dumps(runtime.monitor.traces[-1], ensure_ascii=False).replace(reply.request_id, "<request-id>")
        assert "项目A" not in trace and "mmol/L" not in trace and "3-5" not in trace

    run(scenario())


def test_chat_http_response_serializes_known_report_items_in_source_gap():
    runtime = run(fixture(ContextModel(report_updates)))
    without_report_sources(runtime)
    with TestClient(create_app(Settings(preview=True, embeddings="hash"), runtime)) as client:
        response = client.post("/chat", json={"text":"报告项目A数值10，单位mmol/L，参考范围3-5。"})
    assert response.status_code == 200
    body = response.json()
    result = body["agent_results"][0]
    assert result["missing_fields"] == ["source"]
    assert result["payload"]["report_items"] == [{"name":"项目A", "value":"10", "unit":"mmol/L", "reference_range":"3-5"}]
    assert not result["payload"]["term_explanations"] and not body["references"]


def test_correction_does_not_carry_old_item_unit_or_reference_range():
    async def scenario():
        runtime = await fixture(ContextModel(report_updates))
        without_report_sources(runtime)
        first = await ask(runtime, "报告项目A数值10，单位mmol/L，参考范围3-5。")
        assert first.agent_results[0].payload["report_items"][0]["name"] == "项目A"
        corrected = await ask(runtime, "改看项目B，数值20。")
        result = corrected.agent_results[0]
        assert result.missing_fields == ["unit", "reference_range"]
        assert result.payload["report_items"] == [{"name":"项目B", "value":"20", "unit":None, "reference_range":None}]
        assert "unit" not in corrected.intent.entities and "reference_range" not in corrected.intent.entities

    run(scenario())


def test_subject_switch_does_not_reuse_previous_person_report_fields():
    async def scenario():
        runtime = await fixture(ContextModel(report_updates))
        without_report_sources(runtime)
        await ask(runtime, "报告项目A数值10，单位mmol/L，参考范围3-5。")
        reply = await ask(runtime, "现在给我妈妈看报告项目B，数值20。")
        result = next(result for result in reply.agent_results if result.role == "report_explainer")
        assert reply.subject == "mother" and reply.intent.subject_changed
        assert result.payload["report_items"] == [{"name":"项目B", "value":"20", "unit":None, "reference_range":None}]
        assert result.missing_fields == ["unit", "reference_range"]

    run(scenario())


def test_explicit_unknown_fields_replace_previous_values_without_inference():
    async def scenario():
        runtime = await fixture(ContextModel(report_updates))
        without_report_sources(runtime)
        await ask(runtime, "报告项目A数值10，单位mmol/L，参考范围3-5。")
        reply = await ask(runtime, "报告项目A数值10，单位不清楚，参考范围不清楚。")
        result = reply.agent_results[0]
        assert result.payload["report_items"] == [{"name":"项目A", "value":"10", "unit":None, "reference_range":None}]
        assert result.missing_fields == ["unit", "reference_range"]
        assert reply.intent.entities["unit"].state == reply.intent.entities["reference_range"].state == "unknown"
        assert "unverified_entity_ignored" in reply.intent.fallbacks

    run(scenario())


def test_unverified_or_unattributed_fields_are_not_added_to_payload():
    def fabricated(_payload):
        return [{"name":"report_item", "value":"项目伪造", "state":"provided", "quote":"项目伪造"}]

    async def scenario():
        runtime = await fixture(ContextModel(fabricated))
        without_report_sources(runtime)
        reply = await ask(runtime, "报告中有项目A。")
        assert "unverified_entity_ignored" in reply.intent.fallbacks
        assert reply.agent_results[0].payload["report_items"] == []
        assert reply.agent_results[0].missing_fields == ["source"] and not reply.references

    run(scenario())
