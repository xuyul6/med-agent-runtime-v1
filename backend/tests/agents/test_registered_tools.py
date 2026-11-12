import asyncio
from dataclasses import replace
import json
import pytest
from fastapi.testclient import TestClient
from medagent.agents.contracts import MedicationInput, ReportInput, RiskInput, VisitInput
from medagent.app import create_app
from medagent.core.config import Settings
from medagent.core.contracts import Intent, Question, Risk
from medagent.core.observability import CURRENT, RunLog
from medagent.agents.checks import explicit_reference_request
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.tools.registry import CircuitUnavailable, DEFAULT_TOOLS, ToolBudget, ToolBudgetExceeded, ToolContext, ToolExecutor, ToolInputError, ToolRegistry, ToolResultError, UnknownToolError
from support.harness import Model, fixture, registered_dependency, role_data, role_output, run


def capture(coroutine, log):
    async def scenario():
        token = CURRENT.set(log)
        try: return await coroutine
        finally: CURRENT.reset(token)
    return scenario()


def test_tool_catalog_and_profile_whitelists_share_registered_definitions():
    runtime = run(fixture())
    with TestClient(create_app(Settings(developer_tools=True, preview=True), runtime)) as client:
        tools = client.get("/tools").json()["tools"]
        agents = client.get("/agents").json()["agents"]
        assert len(tools) == 18                     # four open_* tools belong to the comparison arm only
        for tool in tools:
            assert tool["input_schema"]["additionalProperties"] is False
            assert tool["output_schema"]["additionalProperties"] is False
            assert tool["timeout"] > 0
        for profile in agents:
            expected = {tool["name"] for tool in tools if profile["role"] in tool["allowed_roles"]}
            assert set(profile["tool_scope"]) == expected


@pytest.mark.parametrize("text,role,expected", [
    ("你好，介绍功能", "general", {"inspect_request_context"}),
    ("成年人咳嗽三天", "symptom_guide", {"check_symptom_fields"}),
    ("布洛芬药品资料", "drug_reference", {"check_medication_fields", "lookup_reference", "check_medication_scope"}),
    ("报告参考范围是什么意思", "report_explainer", {"check_report_fields", "lookup_reference"}),
    ("去看医生准备哪些资料", "visit_guide", {"check_visit_fields", "lookup_reference", "build_visit_checklist"}),
    ("我现在呼吸困难", "risk_handoff", {"create_handoff_summary"}),
])
def test_registered_role_tools_execute_in_actual_chat(text, role, expected):
    async def scenario():
        runtime = await fixture(Model())
        reply = await runtime.chat(Question(text=text))
        assert reply.primary_agent == role and not reply.degraded
        events = [item for item in runtime.monitor.traces[-1]["events"] if item["event"] == "tool"]
        assert {item["name"] for item in events if item["status"] == "ok"} == expected
        assert all(item["budget_used"] <= item["budget_limit"] for item in events)
        if role == "risk_handoff": assert reply.usage["calls"] == 0 and reply.storage == "skipped"
    run(scenario())


@pytest.mark.parametrize("fault,error,status", [
    ("role", PermissionError, "denied"),
    ("forged_scope", PermissionError, "denied"),
    ("domain", PermissionError, "denied"),
    ("unknown", UnknownToolError, "unknown_tool"),
    ("arguments", ToolInputError, "invalid_request"),
])
def test_rejected_tools_never_reach_dependency(fault, error, status):
    async def scenario():
        class Forbidden:
            async def search(self, *args): raise AssertionError("rejected_call_reached_handler")
        profile = DEFAULT_REGISTRY["drug_reference"]
        request = MedicationInput(text="布洛芬", intent=Intent.MEDICINE)
        name, arguments = "lookup_reference", {"query":"布洛芬", "domain":"medicine"}
        if fault == "role": profile = DEFAULT_REGISTRY["general"]
        elif fault == "forged_scope": profile = replace(DEFAULT_REGISTRY["general"], tool_scope=("lookup_reference",))
        elif fault == "domain": arguments["domain"] = "report"
        elif fault == "unknown": name = "hospital_booking"
        elif fault == "arguments": arguments["private_fact"] = "must-not-be-logged"
        executor, log = ToolExecutor(), RunLog()
        context, budget = ToolContext(profile, request, Forbidden()), ToolBudget()
        with pytest.raises(error): await capture(executor.call(name, arguments, context, budget), log)
        assert log.events[0]["status"] == status
        assert "must-not-be-logged" not in json.dumps(log.events)
        assert not executor.failures and not executor.open_until
    run(scenario())


@pytest.mark.parametrize("fault", ["schema", "domain", "unapproved", "wrong_drug"])
def test_invalid_lookup_result_is_not_published_to_role_context(fault):
    async def scenario():
        runtime = await fixture()
        ref = runtime.index.documents["ibuprofen-overview"]
        if fault == "schema": values = [{"unexpected":"shape"}]
        elif fault == "domain": values = [ref.model_copy(update={"domain":Intent.REPORT})]
        elif fault == "unapproved": values = [ref.model_copy(update={"approved":False})]
        else: values = [ref.model_copy(update={"entities":["阿莫西林"]})]
        class WrongIndex:
            async def search(self, *args): return values
        context = ToolContext(DEFAULT_REGISTRY["drug_reference"], MedicationInput(text="布洛芬资料", intent=Intent.MEDICINE), WrongIndex())
        if fault == "wrong_drug":
            result = await runtime.tools.call("lookup_reference", {"query":"布洛芬", "domain":"medicine"}, context, ToolBudget())
            assert not result.documents and context.scope_filtered_count == 1
        else:
            with pytest.raises(ToolResultError):
                await runtime.tools.call("lookup_reference", {"query":"布洛芬", "domain":"medicine"}, context, ToolBudget())
        assert not context.references
    run(scenario())


def test_empty_lookup_and_missing_scope_are_valid_results_not_dependency_failures():
    async def scenario():
        runtime = await fixture()
        context = ToolContext(DEFAULT_REGISTRY["drug_reference"], MedicationInput(text="未知药品", intent=Intent.MEDICINE), runtime.index)
        log, budget = RunLog(), ToolBudget()
        result = await capture(runtime.tools.call("lookup_reference", {"query":"未知药品", "domain":"medicine"}, context, budget), log)
        assert not result.documents and next(event for event in log.events if event["event"] == "tool")["status"] == "empty_result"
        scope = await runtime.tools.call("check_medication_scope", {}, context, budget)
        assert scope.applicability == "no_source" and not scope.sources
        assert runtime.tools.circuit_snapshot("lookup_reference")["consecutive_fails"] == 0 and not runtime.tools.open_until
        with pytest.raises(ToolInputError):
            await runtime.tools.call("check_medication_scope", {"keys":["another-agent-reference"]}, context, budget)
    run(scenario())


def test_field_tools_preserve_unknowns_and_checklist_requires_actual_template():
    async def scenario():
        runtime = await fixture()
        context = ToolContext(DEFAULT_REGISTRY["report_explainer"], ReportInput(text="报告", intent=Intent.REPORT, items=[{"name":"虚构项目"}]))
        result = await runtime.tools.call("check_report_fields", {}, context, ToolBudget())
        assert result.missing_fields == ["value", "unit", "reference_range"]
        assert context.request.items[0].value is None
        visit = ToolContext(DEFAULT_REGISTRY["visit_guide"], VisitInput(text="通用准备", intent=Intent.VISIT))
        empty = await runtime.tools.call("build_visit_checklist", {}, visit, ToolBudget())
        assert not empty.items and empty.missing_fields == ["source"]
        visit.references["consultation-record"] = runtime.index.documents["consultation-record"]
        filled = await runtime.tools.call("build_visit_checklist", {}, visit, ToolBudget())
        assert filled.items and filled.reference_keys == ["consultation-record"]
    run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_half_open_admits_only_one_probe_and_cleans_up_after_success_or_cancel(cancel):
    async def scenario():
        now, mode, called = [1.0], ["fail"], [0]
        entered, release = asyncio.Event(), asyncio.Event()
        async def handler(context, arguments):
            called[0] += 1
            if mode[0] == "fail": raise ConnectionError()
            if mode[0] == "wait":
                entered.set()
                await release.wait()
            return {"value":"recovered"}
        executor, context, budget = registered_dependency(handler, clock=lambda:now[0], cooldown=5)
        for _ in range(2):
            with pytest.raises(ConnectionError): await executor.call("lookup", {}, context, budget)
        now[0], mode[0] = 7, "wait"
        probe = asyncio.create_task(executor.call("lookup", {}, context, budget))
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(CircuitUnavailable): await executor.call("lookup", {}, context, budget)
        assert called[0] == 3
        if cancel:
            probe.cancel()
            with pytest.raises(asyncio.CancelledError): await probe
            assert not executor.probes
            mode[0] = "ok"
            await executor.call("lookup", {}, context, budget)
        else:
            release.set()
            await probe
        assert not executor.probes and not executor.open_until
        assert executor.failures["lookup"] == 0
    run(scenario())


def test_late_success_cannot_close_circuit_opened_by_other_inflight_failures():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = [0]
        async def handler(context, arguments):
            calls[0] += 1
            if calls[0] == 1:
                entered.set()
                await release.wait()
                return {"value":"late"}
            raise ConnectionError()
        executor, context, budget = registered_dependency(handler)
        late = asyncio.create_task(executor.call("lookup", {}, context, budget))
        await asyncio.wait_for(entered.wait(), 1)
        for _ in range(2):
            with pytest.raises(ConnectionError): await executor.call("lookup", {}, context, budget)
        opened = executor.open_until["lookup"]
        release.set()
        await late
        assert executor.open_until["lookup"] == opened and executor.failures["lookup"] == 2
    run(scenario())


def test_handler_key_error_is_dependency_failure_not_unknown_tool():
    async def scenario():
        async def broken(context, arguments): raise KeyError("private-upstream-field")
        executor, context, budget = registered_dependency(broken)
        log = RunLog()
        with pytest.raises(KeyError): await capture(executor.call("lookup", {}, context, budget), log)
        assert log.events[0]["status"] == "failed"
        assert "private-upstream-field" not in json.dumps(log.events)
    run(scenario())


def test_instance_retry_does_not_reset_tool_budget():
    async def scenario():
        class Invalid(Model):
            def __init__(self): self.messages = 0
            async def message(self, *args, **kwargs):
                self.messages += 1
                return {"content":[{"type":"text", "text":"{}"}]}
        model = Invalid()
        runtime = await fixture(model)
        runtime.settings = replace(runtime.settings, tool_call_limit=3)
        reply = await runtime.chat(Question(text="布洛芬药品资料"))
        events = [event for event in runtime.monitor.traces[-1]["events"] if event["event"] == "tool"]
        assert reply.degraded and model.messages == 1
        assert [event["status"] for event in events] == ["ok", "ok", "ok", "budget_exhausted"]
        assert max(event["budget_used"] for event in events) == 3
    run(scenario())


def test_primary_and_support_share_one_tool_budget():
    async def scenario():
        runtime = await fixture(Model())
        runtime.settings = replace(runtime.settings, tool_call_limit=4)
        reply = await runtime.chat(Question(text="咳嗽三天，去看医生准备哪些资料？"))
        events = [event for event in runtime.monitor.traces[-1]["events"] if event["event"] == "tool"]
        assert not reply.degraded and len(reply.agent_results) == 2
        assert len(events) == 4 and max(event["budget_used"] for event in events) == 4
        assert {event["role"] for event in events} == {"visit_guide", "symptom_guide"}
    run(scenario())


@pytest.mark.parametrize("mode", ["oversized_batch", "final_round"])
def test_disallowed_model_tool_round_or_batch_has_no_partial_execution(mode):
    async def scenario():
        class Calls(Model):
            async def message(self, stage, system, messages, tools=None, budget=1800):
                count = 4 if mode == "oversized_batch" else 1
                return {"content":[{"type":"tool_use", "id":str(i), "name":"inspect_request_context", "input":{}} for i in range(count)]}
        runtime = await fixture(Calls())
        log, budget = RunLog(), ToolBudget()
        with pytest.raises(ToolBudgetExceeded):
            await capture(runtime.agents["general"][0].run("介绍功能", Intent.GENERAL, {}, "selective", tool_budget=budget), log)
        expected = 1 if mode == "oversized_batch" else 3
        assert len([event for event in log.events if event["event"] == "tool"]) == expected
        assert budget.used == expected
    run(scenario())


@pytest.mark.parametrize("role, intent, text, details, offered", [
    ("general", Intent.GENERAL, "介绍功能", {}, set()),
    ("symptom_guide", Intent.SYMPTOMS, "成年人咳嗽三天，想整理记录", {}, {"lookup_reference"}),
    ("drug_reference", Intent.MEDICINE, "查询布洛芬药品资料", {"medication_name":"布洛芬"}, {"lookup_reference", "check_medication_scope"}),
    ("report_explainer", Intent.REPORT, "报告参考范围是什么？", {}, {"lookup_reference"}),
    ("visit_guide", Intent.VISIT, "就医前需要准备什么？", {}, {"lookup_reference"}),
])
def test_role_model_is_not_offered_a_no_argument_tool_the_code_already_ran(role, intent, text, details, offered):
    """Its result is already in validated_tool_results; calling it again only costs a model round and tool budget."""
    async def scenario():
        class Recording(Model):
            async def message(self, stage, system, messages, tools=None, budget=1800):
                self.offered = {tool["name"] for tool in tools or []}
                self.ran = set(role_data(messages)["validated_tool_results"])
                return await super().message(stage, system, messages, tools, budget)
        model = Recording()
        runtime = await fixture(model)
        result = await runtime.agents[role][0].run(text, intent, {}, "selective", details=details)
        assert result.execution.status == "answered"
        assert model.offered == offered
        # Nothing is withheld unless the code ran it; a tool that takes arguments stays available even after it ran.
        assert set(runtime.tools.registry.scope(role)) - model.offered <= model.ran
    run(scenario())


def test_local_risk_dispatch_cannot_execute_async_or_external_tool():
    runtime = run(fixture())
    context = ToolContext(DEFAULT_REGISTRY["drug_reference"], MedicationInput(text="布洛芬", intent=Intent.MEDICINE))
    with pytest.raises(PermissionError):
        runtime.tools.call_local("lookup_reference", {"query":"布洛芬", "domain":"medicine"}, context, ToolBudget())
    risk = ToolContext(DEFAULT_REGISTRY["risk_handoff"], RiskInput(text="虚构风险", intent=Intent.URGENT, risk=Risk(level="CRITICAL", reasons=["breathing"])))
    result = runtime.tools.call_local("create_handoff_summary", {}, risk, ToolBudget(1))
    assert result.known_facts == ["风险规则命中：breathing"] and result.handoff_connected is False


@pytest.mark.parametrize("fault", ["duplicate", "schema", "handler", "execution_policy"])
def test_invalid_tool_registration_is_rejected(fault):
    base = DEFAULT_TOOLS.specs["inspect_request_context"]
    if fault == "duplicate": specs = [base, base]
    elif fault == "schema": specs = [replace(base, output_contract=None)]
    elif fault == "handler": specs = [replace(base, handler=None)]
    else: specs = [replace(base, side_effect="read_external", dependency=False)]
    with pytest.raises(ValueError): ToolRegistry(specs)


@pytest.mark.parametrize("text,expected", [
    ("请先查阅症状模板，再给出来源", True),
    ("不要引用，只整理用户原述", False),
    ("不需要检索或引用", False),
    ("不用检索，但请给出已取得的来源", True),
])
def test_explicit_source_requirement_respects_local_negation(text, expected):
    assert explicit_reference_request(text) is expected


def test_source_task_cannot_succeed_without_reference_and_native_tool_result_is_validated():
    async def scenario():
        request = "成年人咳嗽三天，请查阅症状记录模板并说明来源。"
        runtime = await fixture(Model())
        with pytest.raises(ValueError, match="requested_evidence_not_returned"):
            await runtime.agents["symptom_guide"][0].run(request, Intent.SYMPTOMS, {}, "selective")
        class Retrieve(Model):
            def __init__(self): self.turn = 0
            async def message(self, stage, system, messages, tools=None, budget=1800):
                self.turn += 1
                if self.turn == 1:
                    return {"content":[{"type":"tool_use", "id":"source", "name":"lookup_reference",
                                        "input":{"query":"症状记录模板", "domain":"symptoms"}}]}
                result = json.loads(messages[-1]["content"][0]["content"])
                assert result["documents"][0]["key"] == "symptom-record"
                return {"content":[{"type":"text", "text":json.dumps(role_output("symptom_guide", ["symptom-record"]),ensure_ascii=False)}]}
        runtime = await fixture(Retrieve())
        result = await runtime.agents["symptom_guide"][0].run(request, Intent.SYMPTOMS, {}, "selective")
        assert result.answer.reference_keys == ["symptom-record"]
    run(scenario())
