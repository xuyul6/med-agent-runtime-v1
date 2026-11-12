import json
from dataclasses import replace
import pytest
from fastapi.testclient import TestClient
from medagent.app import create_app
from medagent.agents.contracts import GeneralInput
from medagent.core.config import Settings
from medagent.core.contracts import Intent, Question, Risk
from medagent.agents.execution import HandoffRequired, RoleAgent
from medagent.agents.checks import checked_role_output
from medagent.agents.profiles import DEFAULT_REGISTRY, STEP_NAMES, WorkflowRun, default_profiles
from medagent.core.provider import ClaudeGateway, json_schema
from medagent.runtime import MedicalRuntime
from support.harness import Model, fixture, role_output, run, role_data


@pytest.mark.parametrize("fault", ["duplicate", "unknown", "missing", "contract", "domain", "tools", "workflow", "handoff"])
def test_invalid_profiles_fail_during_runtime_construction(fault):
    async def scenario():
        source = await fixture()
        profiles = default_profiles()
        first = profiles[0]
        if fault == "duplicate": profiles.append(first)
        elif fault == "unknown": profiles[0] = replace(first, role="not_registered")
        elif fault == "missing": profiles.pop()
        elif fault == "contract": profiles[0] = replace(first, input_contract=None)
        elif fault == "domain": profiles[0] = replace(first, knowledge_domains=frozenset({Intent.MEDICINE}))
        elif fault == "tools": profiles[0] = replace(first, tool_scope=("lookup_reference",))
        elif fault == "workflow": profiles[0] = replace(first, workflow=first.workflow[::-1])
        elif fault == "handoff": profiles[0] = replace(first, handoff_conditions=())
        with pytest.raises(ValueError):
            MedicalRuntime(source.settings, None, source.vectors, source.index, source.memory, profiles=profiles)
    run(scenario())


def test_workflow_cannot_skip_required_steps():
    flow = WorkflowRun(DEFAULT_REGISTRY["drug_reference"], "test")
    with pytest.raises(ValueError, match="workflow_precondition_failed"):
        flow.complete("compose_answer", "draft")
    assert flow.completed == []


def test_agent_catalog_exposes_seven_real_contracts_including_deterministic_risk():
    runtime = run(fixture())
    with TestClient(create_app(Settings(developer_tools=True, preview=True), runtime)) as client:
        response = client.get("/agents")
        assert response.status_code == 200
        agents = {item["role"]:item for item in response.json()["agents"]}
        assert set(agents) == {"general", "symptom_guide", "drug_reference", "report_explainer", "visit_guide", "appointment", "risk_handoff"}
        assert agents["risk_handoff"]["deterministic"]
        assert agents["risk_handoff"]["tool_scope"] == ["create_handoff_summary"]
        assert agents["report_explainer"]["knowledge_domains"] == ["report"]
        assert set(agents["drug_reference"]["tool_scope"]) == {"lookup_reference", "check_medication_fields", "check_medication_scope"}
        for item in agents.values():
            assert item["mission"] and item["handoff_conditions"]
            assert item["input_contract"]["properties"] and item["output_contract"]["properties"]
            assert [step["name"] for step in item["workflow"]] == list(STEP_NAMES)


def test_role_input_rejects_wrong_domain_and_reserved_context_override_before_model():
    async def scenario():
        class NeverModel(Model):
            async def message(self, *args, **kwargs): raise AssertionError("model_must_not_run")
        runtime = await fixture(NeverModel())
        agent = runtime.agents["drug_reference"][0]
        with pytest.raises(ValueError):
            await agent.run("报告文字", Intent.REPORT, {}, "selective")
        with pytest.raises(ValueError, match="cannot_override_agent_context"):
            await agent.run("药品资料", Intent.MEDICINE, {}, "selective", details={"intent":Intent.REPORT})
        with pytest.raises(ValueError):
            await agent.run("药品资料", Intent.MEDICINE, {}, "selective", details={"unknown_field":"value"})
        assert GeneralInput.model_validate({"text":"你好", "intent":Intent.GENERAL}).subject == "self"
    run(scenario())


def test_missing_report_fields_return_clarification_through_chat_not_http_error():
    class MissingReport(Model):
        async def message(self, stage, system, messages, tools=None, budget=1800):
            output = {"role":"report_explainer", "status":"needs_clarification",
                      "answer":{"text":"需要原报告字段才能整理具体项目。"},
                      "missing_fields":["value", "unit", "reference_range"], "payload":{}}
            return {"content":[{"type":"text","text":json.dumps(output,ensure_ascii=False)}]}
    runtime = run(fixture(MissingReport()))
    with TestClient(create_app(Settings(developer_tools=True, preview=True), runtime)) as client:
        response = client.post("/chat", json={"text":"报告数值偏高，帮我整理一下。"})
        assert response.status_code == 200
        reply = response.json()
        assert reply["primary_agent"] == "report_explainer"
        assert not reply["degraded"]
        result = reply["agent_results"][0]
        assert result["status"] == "needs_clarification"
        assert result["missing_fields"] == ["value", "unit", "reference_range"]
        assert result["handoff_reason"] == "missing_information"
        assert result["handoff_target"] == "user"
        assert len(reply["answer"]["questions"]) == 3
        assert result["workflow"] == list(STEP_NAMES)
        trace = client.get("/traces/"+reply["request_id"]).json()
        steps = [event for event in trace["events"] if event["event"] == "agent_workflow"]
        assert [event["step"] for event in steps] == list(STEP_NAMES)
        assert all(event["profile_version"] == DEFAULT_REGISTRY["report_explainer"].version for event in steps)


def test_missing_medicine_source_is_clarification_without_factual_model_answer():
    async def scenario():
        class NoAnswer(Model):
            async def message(self, *args, **kwargs): raise AssertionError("no_source_no_answer")
        runtime = await fixture(NoAnswer())
        reply = await runtime.chat(Question(text="想查阿莫西林的药品说明书。"))
        assert not reply.degraded
        assert reply.agent_results[0].status == "needs_clarification"
        assert reply.agent_results[0].missing_fields == ["source"]
        assert not reply.references
    run(scenario())


@pytest.mark.parametrize("fault", ["wrong_role", "wrong_payload", "unknown_reference", "payload_reference", "boundary", "missing_but_answered", "unknown_missing"])
def test_bad_role_outputs_are_blocked(fault):
    async def scenario():
        runtime = await fixture()
        refs = {"consultation-record":runtime.index.documents["consultation-record"]}
        data = role_output("visit_guide", ["consultation-record"])
        if fault == "wrong_role": data["role"] = "drug_reference"
        elif fault == "wrong_payload": data["payload"] = {"symptom_summary":"字段错误"}
        elif fault == "unknown_reference": data["answer"]["reference_keys"] = ["invented"]
        elif fault == "payload_reference": data["payload"]["institution_facts"] = [{"text":"来源信息", "reference_keys":["invented"]}]
        elif fault == "boundary": data["payload"]["preparation_checklist"] = ["已为你预约门诊。"]
        elif fault == "missing_but_answered": data["missing_fields"] = ["institution"]
        elif fault == "unknown_missing": data["status"], data["missing_fields"] = "needs_clarification", ["password"]
        with pytest.raises(ValueError): checked_role_output(DEFAULT_REGISTRY["visit_guide"], data, refs)
    run(scenario())


def test_invalid_support_output_never_reaches_synthesis():
    async def scenario():
        class BadSupport(Model):
            def __init__(self): self.synthesis_calls = 0
            async def message(self, stage, system, messages, tools=None, budget=1800):
                if stage == "agent_symptom_guide":
                    return {"content":[{"type":"text","text":json.dumps(role_output("visit_guide"),ensure_ascii=False)}]}
                return await super().message(stage, system, messages, tools, budget)
            async def json(self, stage, *args, **kwargs):
                if stage == "synthesis": self.synthesis_calls += 1
                return await super().json(stage, *args, **kwargs)
        model = BadSupport()
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text="咳嗽三天，去看医生需要准备哪些资料？"))
        assert reply.degraded
        assert reply.primary_agent == "visit_guide"
        assert [result.role for result in reply.agent_results] == ["visit_guide"]
        assert model.synthesis_calls == 0
    run(scenario())


def test_primary_clarification_cannot_be_overwritten_by_synthesis():
    async def scenario():
        class ClarifyPrimary(Model):
            async def message(self, stage, system, messages, tools=None, budget=1800):
                if stage == "agent_visit_guide":
                    return {"content":[{"type":"text","text":json.dumps({"role":"visit_guide", "status":"needs_clarification",
                        "answer":{"text":"请说明希望了解的机构。"}, "missing_fields":["institution"], "payload":{}},ensure_ascii=False)}]}
                return await super().message(stage, system, messages, tools, budget)
            async def json(self, stage, *args, **kwargs):
                if stage == "synthesis": raise AssertionError("must_preserve_clarification")
                return await super().json(stage, *args, **kwargs)
        runtime = await fixture(ClarifyPrimary())
        reply = await runtime.chat(Question(text="咳嗽三天，去看医生需要准备哪些资料？"))
        assert not reply.degraded
        assert reply.agent_results[0].status == "needs_clarification"
        assert reply.answer.questions
    run(scenario())


def test_human_request_has_explicit_handoff_without_connection_or_agent_model():
    async def scenario():
        class NoAnswer(Model):
            async def message(self, *args, **kwargs): raise AssertionError("handoff_does_not_need_agent_model")
        runtime = await fixture(NoAnswer())
        result = await runtime.agents["general"][0].run("请转人工客服", Intent.GENERAL, {}, "selective")
        assert result.execution.status == "handoff_required"
        assert result.execution.handoff_reason == "human_requested"
        assert result.execution.handoff_target == "offline_professional"
        assert "未实际转接" in result.answer.text
    run(scenario())


def test_critical_role_contract_has_no_model_storage_or_index_dependency():
    async def scenario():
        runtime = await fixture(Model())
        class Forbidden:
            def __getattr__(self, name): raise AssertionError("critical_dependency_accessed:"+name)
        runtime.model = runtime.memory = runtime.index = runtime.router = Forbidden()
        reply = await runtime.chat(Question(text="我现在呼吸困难"))
        assert reply.primary_agent == "risk_handoff"
        assert reply.storage == "skipped"
        assert reply.usage["calls"] == 0
        assert not reply.handoff_connected
        result = reply.agent_results[0]
        assert result.status == "handoff_required"
        assert result.handoff_reason == "critical_risk"
        assert result.workflow == list(STEP_NAMES)
        assert result.payload["handoff_connected"] is False
        with pytest.raises(HandoffRequired):
            await runtime.agents["symptom_guide"][0].run("测试", Intent.SYMPTOMS, {}, "selective", risk=Risk(level="CRITICAL"))
        with pytest.raises(ValueError, match="deterministic_role"):
            RoleAgent("risk_handoff", "bad", None, None, None, None)
    run(scenario())


def test_provider_contract_schema_matches_each_role_shape():
    async def scenario():
        runtime = await fixture()
        key_for_role = {"general":[], "symptom_guide":["symptom-record"], "drug_reference":["ibuprofen-overview"],
                        "report_explainer":["lab-context"], "visit_guide":["consultation-record"]}
        expected_payloads = {
            "general":{"task_summary", "suggested_role"},
            "symptom_guide":{"symptom_summary", "timeline", "care_information"},
            "drug_reference":{"medication_identity", "source_scope", "evidence_items", "unresolved_questions"},
            "report_explainer":{"report_items", "term_explanations", "questions_for_clinician"},
            "visit_guide":{"preparation_checklist", "institution_facts", "needs_verification"},
        }
        for role, keys in key_for_role.items():
            profile = DEFAULT_REGISTRY[role]
            output = checked_role_output(profile, role_output(role, keys), runtime.index.documents)
            schema = json_schema("agent_"+role)
            assert schema["properties"]["role"]["enum"] == [role]
            assert schema["required"] == list(profile.output_contract.model_fields)
            payload_ref = schema["properties"]["payload"]["$ref"].removeprefix("#/$defs/")
            payload_schema = schema["$defs"][payload_ref]
            assert set(payload_schema["properties"]) == expected_payloads[role]
            for node in [schema, *schema["$defs"].values()]:
                if node.get("type") == "object":
                    assert node["additionalProperties"] is False
                    assert set(node["required"]) == set(node["properties"])
            assert output.role == role
        with pytest.raises(ValueError, match="deterministic_role"):
            json_schema("agent_risk_handoff")
    run(scenario())


@pytest.mark.parametrize("role,intent,details,expected", [
    ("symptom_guide", Intent.SYMPTOMS, {"age_group":"成年"}, ["onset"]),
    ("drug_reference", Intent.MEDICINE, {"medication_name":"布洛芬", "query_type":"product_information"}, ["formulation", "population", "product_name", "strength", "region"]),
    ("report_explainer", Intent.REPORT, {"items":[{"name":"虚构项目"}]}, ["value", "unit", "reference_range"]),
    ("visit_guide", Intent.VISIT, {"institution_specific":True}, ["source"]),
])
def test_explicit_structured_input_gaps_stop_before_model_and_retrieval(role, intent, details, expected):
    async def scenario():
        runtime = await fixture()
        class Forbidden:
            def __getattr__(self, name): raise AssertionError("clarification_dependency_accessed:"+name)
        agent = runtime.agents[role][0]
        agent.model = agent.index = Forbidden()
        result = await agent.run("虚构输入", intent, {}, "selective", details=details)
        assert result.execution.status == "needs_clarification"
        assert result.execution.missing_fields == expected
        assert len(result.answer.questions) == len(expected)
    run(scenario())


def test_gateway_sends_role_schema_in_native_output_config():
    async def scenario():
        import httpx
        captured = []
        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"content":[], "usage":{}, "stop_reason":"end_turn"})
        model = ClaudeGateway(Settings(api_key="fake", model="fake"), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            await model.message("agent_report_explainer", "test", [{"role":"user", "content":"虚构测试"}])
        finally:
            await model.close()
        assert captured[0]["output_config"]["format"]["schema"] == json_schema("agent_report_explainer")
    run(scenario())


def test_unextracted_optional_fields_are_not_presented_as_missing_patient_facts():
    async def scenario():
        class Capture(Model):
            async def message(self, stage, system, messages, tools=None, budget=1800):
                self.input_data = role_data(messages)["validated_input"]
                self.original_text = messages[0]["content"]
                return await super().message(stage, system, messages, tools, budget)
        model = Capture()
        runtime = await fixture(model)
        await runtime.agents["symptom_guide"][0].run("成年人咳嗽三天，想整理记录", Intent.SYMPTOMS, {}, "selective")
        assert "onset" not in model.input_data and "age_group" not in model.input_data
        assert "三天" in model.original_text
    run(scenario())
