import dataclasses
import asyncio
from dataclasses import replace
import json
import httpx
import pytest
from fastapi.testclient import TestClient
from medagent.app import create_app
from medagent.core.config import Settings
from medagent.core.contracts import Intent, Question
from medagent.monitoring import MonitorExcluded, MonitorPolicy
from medagent.core.observability import CURRENT, RunLog
from medagent.tools.registry import ToolBudget, ToolContext, ToolResultError
from medagent.core.provider import ClaudeGateway
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.agents.contracts import MedicationInput
from support.harness import Model, fixture, registered_dependency, run
from support.routing import COMPOUND_REQUEST, fixed_understanding, understanding
from support.monitoring import Clock, bind, monitor, poison


def test_agent_cumulative_total_window_and_unknown_metrics_have_distinct_meanings():
    monitored,clock = monitor(replace(MonitorPolicy(),window_size=3,min_samples=2))
    monitored.register_agent("a.1","a")
    empty = monitored.snapshot()["agent_stats"]["a.1"]
    assert empty["total"] == empty["samples"] == 0
    assert empty["success_rate"] is None and empty["avg_ms"] is None and empty["p95_ms"] is None
    for good,ms in [(True,100),(True,200),(False,300),(True,400)]: monitored.record("a.1",good,ms)
    stats = monitored.snapshot()["agent_stats"]["a.1"]
    assert stats["total"] == 4 and stats["samples"] == 3
    assert stats["success_rate"] == pytest.approx(2/3) and stats["avg_ms"] == 300
    assert stats["p50_ms"] == 300 and stats["p95_ms"] == 400
    clock.advance(301)
    stats = monitored.snapshot()["agent_stats"]["a.1"]
    assert stats["total"] == 4 and stats["samples"] == 0 and stats["success_rate"] is None


def test_cancellation_does_not_count_as_dependency_failure_or_recovery():
    monitored,_ = monitor()
    monitored.record("a.1",True,100)
    monitored.record("a.1",False,5000,status="cancelled",error="CancelledError")
    stats = monitored.snapshot()["agent_stats"]["a.1"]
    assert stats["total"] == 2 and stats["execution_samples"] == 1 and stats["success_rate"] == 1
    assert stats["avg_ms"] == 100 and stats["error_counts"] == {"cancelled":1}
    assert monitored.agent_series["a.1"].recovery_streak == 0


def test_min_samples_alert_dedup_and_penalty_hysteresis_are_separate():
    monitored,_ = monitor()
    for _ in range(4): monitored.record("a.1",False,100)
    monitored.sample()
    assert not monitored.snapshot()["active_alerts"]
    monitored.record("a.1",False,100)
    monitored.sample()
    assert len(monitored.alerts) == 1 and monitored.agent_series["a.1"].penalty == 0
    monitored.sample()
    assert monitored.agent_series["a.1"].penalty == .25
    for _ in range(5): monitored.sample()
    assert monitored.agent_series["a.1"].penalty == .25  # Old failures are not repeatedly charged.
    assert len(monitored.alert_changes) == 1
    monitored.record("a.1",False,100)
    monitored.sample()
    assert monitored.agent_series["a.1"].penalty == .5


def test_slow_successes_can_alert_without_claiming_operational_failure():
    monitored,_ = monitor()
    for _ in range(5): monitored.record("a.1",True,16000)
    monitored.sample()
    stats = monitored.snapshot()
    assert stats["agent_stats"]["a.1"]["success_rate"] == 1
    assert stats["active_alerts"][0]["reasons"] == ["high_latency"]
    assert stats["agent_stats"]["a.1"]["monitor_penalty"] == 0


def test_empty_tool_results_are_available_but_have_zero_nonempty_quality():
    monitored,_ = monitor()
    monitored.register_tool("lookup_reference",True)
    for _ in range(5): monitored.record_tool("lookup_reference","empty_result",50,invoked=True)
    monitored.record_tool("lookup_reference","denied",0)
    monitored.record_tool("lookup_reference","invalid_request",0)
    monitored.sample()
    stats = monitored.snapshot()["tool_stats"]["lookup_reference"]
    assert stats["total"] == 7 and stats["execution_samples"] == 5
    assert stats["success_rate"] == 1 and stats["quality_success_rate"] == 0
    assert stats["status_counts"] == {"empty_result":5,"denied":1,"invalid_request":1}
    assert any("low_nonempty_validated_result_rate" in alert["reasons"] for alert in monitored.alerts.values())


def test_scope_filter_schema_errors_and_unknown_tool_buckets_are_bounded():
    monitored,_ = monitor()
    monitored.register_tool("lookup_reference",True)
    monitored.record_tool("lookup_reference","ok",10,invoked=True,filtered=2)
    for code in ["invalid_tool_result","reference_for_another_medication","reference_outside_tool_scope"]:
        monitored.record_tool("lookup_reference","invalid_result",10,invoked=True,quality=code)
    for index in range(100): monitored.record_tool("private-name-"+str(index),"unknown_tool",0)
    stats = monitored.snapshot()["tool_stats"]
    assert set(stats) == {"lookup_reference","__unknown__"}
    assert stats["lookup_reference"]["scope_filtered_count"] == 2
    assert stats["lookup_reference"]["quality_counts"] == {code:1 for code in ["invalid_tool_result","reference_for_another_medication","reference_outside_tool_scope"]}
    assert stats["__unknown__"]["success_rate"] is None and stats["__unknown__"]["total"] == 100


def test_actual_tool_failures_circuit_block_and_half_open_probe_are_separate_metrics():
    async def scenario():
        clock = Clock()
        failing = [True]
        async def handler(context,args):
            if failing[0]: raise ConnectionError()
            return {"value":"ok"}
        executor,context,budget = registered_dependency(handler,clock=clock,cooldown=5)
        monitored,_ = monitor()
        monitored.register_tool("lookup",True)
        executor.monitor = monitored
        monitored.circuits = executor.circuit_snapshot
        for _ in range(2):
            with pytest.raises(ConnectionError): await executor.call("lookup",{},context,budget)
        with pytest.raises(Exception): await executor.call("lookup",{},context,budget)
        stats = monitored.snapshot()["tool_stats"]["lookup"]
        assert stats["status_counts"] == {"failed":2,"circuit_open":1}
        assert stats["execution_samples"] == 2 and stats["consecutive_fails"] == 2 and stats["circuit_state"] == "open"
        clock.advance(6)
        assert monitored.snapshot()["tool_stats"]["lookup"]["circuit_state"] == "half_open"
        failing[0] = False
        await executor.call("lookup",{},context,budget)
        stats = monitored.snapshot()["tool_stats"]["lookup"]
        assert stats["total"] == 4 and stats["consecutive_fails"] == 0 and stats["circuit_state"] == "closed"
    run(scenario())


def test_real_lookup_scope_filter_is_counted_without_exposing_values():
    async def scenario():
        runtime = await fixture(Model())
        context = ToolContext(DEFAULT_REGISTRY["drug_reference"],MedicationInput(text="synthetic",intent=Intent.MEDICINE,medication_name="阿莫西林"),runtime.index)
        reference = runtime.index.documents["ibuprofen-overview"]
        async def mismatched(*args, **kwargs): return [reference]
        runtime.index.search = mismatched
        result = await runtime.tools.call("lookup_reference",{"query":"公开资料","domain":"medicine"},context,ToolBudget())
        assert result.documents == []
        stats = runtime.monitor.snapshot()["tool_stats"]["lookup_reference"]
        assert stats["scope_filtered_count"] == 1 and stats["status_counts"] == {"empty_result":1}
    run(scenario())


def test_sampled_penalty_changes_next_real_instance_and_role_fallback_remains_scoped():
    async def scenario():
        runtime = await fixture(Model())
        monitored,_ = bind(runtime)
        poison(monitored,"drug_reference.1")
        reply = await runtime.chat(Question(text="布洛芬药品公开资料"))
        assert reply.agent_results[0].instance == "drug_reference.2"
        assert not reply.routing_decision.monitor_fallback
        poison(monitored,"drug_reference.2")
        reply = await runtime.chat(Question(text="布洛芬药品公开资料"))
        assert reply.primary_agent == "drug_reference" and not reply.agent_results and reply.degraded
        assert reply.routing_decision.monitor_fallback == ["drug_reference"]
        assert reply.routing_decision.monitor_fallback == ["drug_reference"]
        assert "monitor_fallback" not in reply.routing_decision.routing_reason
        assert monitored.snapshot()["monitor_fallbacks"] == 1
        assert not any(item.role == "general" for item in reply.routing_decision.assignments)
    run(scenario())


def test_alert_does_not_imply_actual_monitor_fallback():
    async def scenario():
        runtime = await fixture(Model())
        monitored,_ = bind(runtime)
        for name in ["drug_reference.1","drug_reference.2"]:
            for _ in range(5): monitored.record(name,False,100)
        monitored.sample()
        reply = await runtime.chat(Question(text="布洛芬公开资料"))
        assert monitored.snapshot()["active_alerts"]
        assert not reply.degraded and not reply.routing_decision.monitor_fallback
    run(scenario())


def test_compound_request_continues_only_other_admitted_task_when_role_excluded():
    async def scenario():
        runtime = await fixture(Model())
        monitored,_ = bind(runtime)
        for name in ["drug_reference.1","drug_reference.2"]: poison(monitored,name)
        fixed_understanding(runtime,understanding())
        reply = await runtime.chat(Question(text=COMPOUND_REQUEST))
        assert reply.primary_agent == "report_explainer" and reply.degraded
        assert reply.routing_decision.monitor_fallback == ["drug_reference"]
        assert [item.role for item in reply.agent_results] == ["report_explainer"]
        assert any("药品资料部分未完成" in value for value in reply.answer.limitations)
    run(scenario())


def test_recovery_probes_use_normal_requests_and_fresh_successes_reduce_penalty():
    async def scenario():
        runtime = await fixture(Model())
        monitored,clock = bind(runtime)
        poison(monitored,"drug_reference.1")
        clock.advance(31)
        first = await runtime.chat(Question(text="布洛芬公开资料"))
        assert first.agent_results[0].instance == "drug_reference.1"
        assert first.routing_decision.assignments[0].instances[0].monitor_probe
        assert not monitored.agent_series["drug_reference.1"].probe_token
        monitored.sample()
        assert monitored.agent_series["drug_reference.1"].penalty == .5
        second = await runtime.chat(Question(text="布洛芬公开资料"))
        assert second.agent_results[0].instance == "drug_reference.2"
        clock.advance(31)
        await runtime.chat(Question(text="布洛芬公开资料"))
        monitored.sample()
        assert monitored.agent_series["drug_reference.1"].penalty == .25
        assert not monitored.routing_state("drug_reference.1")["excluded"]
        for _ in range(3): monitored.sample()
        assert monitored.agent_series["drug_reference.1"].penalty == .25  # Historical failures cannot undo recovery.
    run(scenario())


def test_clarification_success_is_not_evidence_that_model_dependency_recovered():
    monitored,_ = monitor()
    poison(monitored,"a.1")
    for _ in range(3): monitored.record("a.1",True,10,recovery_evidence=False)
    monitored.sample()
    assert monitored.agent_series["a.1"].penalty == .5


def test_monitor_probe_has_single_role_admission_and_cancel_cleanup():
    async def scenario():
        runtime = await fixture(Model())
        monitored,clock = bind(runtime)
        poison(monitored,"drug_reference.1"); poison(monitored,"drug_reference.2")
        clock.advance(31)
        entered = asyncio.Event()
        async def delayed(*args,**kwargs):
            entered.set()
            await asyncio.sleep(10)
        runtime.agents["drug_reference"][0].run = delayed
        task = asyncio.create_task(runtime.chat(Question(text="布洛芬公开资料")))
        await entered.wait()
        assert monitored.agent_series["drug_reference.1"].probe_token is not None
        second = await runtime.chat(Question(text="布洛芬公开资料"))
        assert second.routing_decision.monitor_fallback == ["drug_reference"] and not second.agent_results
        with pytest.raises(MonitorExcluded): monitored.claim("drug_reference.2")
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert all(state.probe_token is None for state in monitored.agent_series.values())
    run(scenario())


def test_two_prepared_plans_cannot_execute_the_same_recovery_probe_concurrently():
    async def scenario():
        runtime = await fixture(Model())
        monitored,clock = bind(runtime)
        for name in ['drug_reference.1','drug_reference.2']: poison(monitored,name)
        clock.advance(31)
        ready, release, waiting = asyncio.Event(), asyncio.Event(), [0]
        dispatch = runtime._dispatch
        async def barrier(*args,**kwargs):
            waiting[0] += 1
            if waiting[0] == 2: ready.set()
            await ready.wait()
            return await dispatch(*args,**kwargs)
        runtime._dispatch = barrier
        original = runtime.agents['drug_reference'][0].run
        async def held(*args,**kwargs):
            await release.wait()
            return await original(*args,**kwargs)
        runtime.agents['drug_reference'][0].run = held
        tasks = [asyncio.create_task(runtime.chat(Question(text='布洛芬公开资料'))) for _ in range(2)]
        done,_ = await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED,timeout=1)
        assert len(done) == 1
        rejected = next(iter(done)).result()
        assert rejected.degraded and rejected.routing_decision.monitor_fallback == ['drug_reference']
        assert not rejected.agent_results and monitored.routing_rejections['drug_reference.1'] == 1
        release.set()
        results = await asyncio.gather(*tasks)
        assert sum(bool(reply.agent_results) for reply in results) == 1
        assert monitored.monitor_fallbacks == 1
        assert not monitored.agent_series['drug_reference.1'].probe_token
    run(scenario())


def test_recovery_probe_uses_short_timeout_then_healthy_instance_can_complete():
    async def scenario():
        runtime = await fixture(Model())
        monitored,clock = bind(runtime,replace(MonitorPolicy(),probe_timeout=.001))
        poison(monitored,'drug_reference.1')
        clock.advance(31)
        async def slow(*args,**kwargs): await asyncio.sleep(10)
        runtime.agents['drug_reference'][0].run = slow
        reply = await runtime.chat(Question(text='布洛芬公开资料'))
        assert not reply.degraded and reply.agent_results[0].instance == 'drug_reference.2'
        assert reply.routing_decision.assignments[0].attempts[0].status == 'timeout'
        assert not monitored.agent_series['drug_reference.1'].probe_token
    run(scenario())


def test_actual_rejected_tool_result_has_scope_quality_and_failure_counters():
    async def scenario():
        runtime = await fixture(Model())
        context = ToolContext(DEFAULT_REGISTRY['drug_reference'],MedicationInput(text='synthetic',intent=Intent.MEDICINE,medication_name='布洛芬'),runtime.index)
        reference = runtime.index.documents['ibuprofen-overview'].model_copy(update={'approved':False})
        async def unapproved(*args, **kwargs): return [reference]
        runtime.index.search = unapproved
        with pytest.raises(ToolResultError):
            await runtime.tools.call('lookup_reference',{'query':'资料','domain':'medicine'},context,ToolBudget())
        stats = runtime.monitor.snapshot()['tool_stats']['lookup_reference']
        assert stats['total'] == 1 and stats['success_rate'] == 0
        assert stats['quality_counts'] == {'reference_outside_tool_scope':1}
        assert not context.references
    run(scenario())


def test_model_trace_records_http_duration_separately_from_request_elapsed_time():
    async def scenario():
        async def respond(request):
            return httpx.Response(200,json={'content':[{'type':'text','text':'{}'}], 'usage':{'input_tokens':1,'output_tokens':1},'stop_reason':'end_turn'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            gateway = ClaudeGateway(Settings(api_key='synthetic-test-key',model='synthetic'),client)
            log = RunLog(); token = CURRENT.set(log)
            try: await gateway.message('understanding','synthetic',[{'role':'user','content':'synthetic'}])
            finally: CURRENT.reset(token)
            event = next(item for item in log.events if item['event'] == 'model')
            assert event['elapsed_ms'] >= 0 and event['at_ms'] >= 0
            assert 'synthetic-test-key' not in json.dumps(log.events)
    run(scenario())


def test_recovery_alert_clears_after_good_window_and_changes_are_deduplicated():
    monitored,clock = monitor(replace(MonitorPolicy(),window_size=5))
    poison(monitored,"a.1")
    clock.advance(301)
    assert monitored.snapshot()["agent_stats"]["a.1"]["success_rate"] is None
    monitored.sample()
    assert monitored.agent_series["a.1"].penalty == .5
    for _ in range(2): monitored.record("a.1",True,10)
    monitored.sample()
    for _ in range(3): monitored.record("a.1",True,10)
    monitored.sample()
    assert monitored.agent_series["a.1"].penalty == 0 and monitored.alerts
    monitored.sample()
    assert monitored.agent_series["a.1"].penalty == 0 and not monitored.alerts
    assert [item["state"] for item in monitored.alert_changes] == ["opened","resolved"]


def test_recent_routing_score_carries_request_context_and_is_not_global_constant():
    async def scenario():
        runtime = await fixture(Model())
        first = await runtime.chat(Question(text="布洛芬公开资料"))
        stats = runtime.monitor.snapshot()["agent_stats"]["drug_reference.1"]
        assert stats["routing_context"]["request_id"] == first.request_id and stats["routing_context"]["recorded_at"]
        assert stats["routing_score"] is not None
        second = await runtime.chat(Question(text="介绍支持哪些功能"))
        stats = runtime.monitor.snapshot()["agent_stats"]["drug_reference.1"]
        assert stats["routing_context"]["request_id"] == second.request_id
        assert stats["routing_score"] is None and not stats["routing_context"]["eligible"]
    run(scenario())


def test_risk_cannot_be_excluded_by_any_penalty_and_counts_deterministic_attempt():
    async def scenario():
        runtime = await fixture(Model())
        monitored,_ = bind(runtime)
        for name in monitored.agent_series: monitored.agent_series[name].penalty = 1
        runtime.memory.available = False
        reply = await runtime.chat(Question(text="现在呼吸困难"))
        assert reply.primary_agent == "risk_handoff" and reply.usage["calls"] == 0 and reply.storage == "skipped"
        monitored.sample()
        stats = monitored.snapshot()["agent_stats"]["risk_handoff.deterministic"]
        assert stats["total"] == 1 and stats["monitor_penalty"] == 0 and not stats["excluded"]
    run(scenario())


def test_sampler_lifespan_starts_once_updates_and_stops_on_app_shutdown():
    async def scenario():
        runtime = await fixture(Model())
        runtime.monitor.policy = replace(runtime.monitor.policy,interval=.001)
        return runtime
    runtime = run(scenario())
    app = create_app(runtime=runtime,settings=dataclasses.replace(runtime.settings, developer_tools=True))
    with TestClient(app) as client:
        assert client.get('/monitor').json()['sampler']['running']
        async def ticks():
            runtime.monitor.start()
            await asyncio.sleep(.01)
        client.portal.call(ticks)
        assert client.get('/monitor').json()['sampler']['revision'] >= 1
    assert runtime.monitor._task is None


def test_monitor_runtime_alias_and_metrics_units_bounded_labels_and_no_fake_unknown_rates():
    runtime = run(fixture(Model()))
    runtime.monitor.record("general.1",True,1000)
    runtime.monitor.register_tool("lookup_reference",True)
    runtime.monitor.record_tool("lookup_reference","timeout",2000,invoked=True)
    app = create_app(settings=dataclasses.replace(runtime.settings, developer_tools=True),runtime=runtime)
    with TestClient(app) as client:
        monitor_data = client.get('/monitor').json()
        runtime_data = client.get('/runtime').json()
        for key in ['agent_stats','tool_stats','agents','requests','degraded']: assert monitor_data[key] == runtime_data[key]
        text = client.get('/metrics').text
        assert 'mar_agent_latency_seconds_sum{instance="general.1",role="general"} 1.0' in text
        assert 'mar_tool_latency_seconds_sum{tool="lookup_reference"} 2.0' in text
        assert 'mar_agent_success_rate{instance="general.2"' not in text
        for label in ['user=','session=','request_id=','text=']: assert label not in text
        assert monitor_data['scope'] == 'single_process'


def test_monitor_policy_environment_loads_typed_values_without_reading_key_values(monkeypatch):
    monkeypatch.setenv('MAR_MONITOR_WINDOW_SIZE','12')
    monkeypatch.setenv('MAR_MONITOR_INTERVAL','0.5')
    monkeypatch.setenv('MAR_MONITOR_AGENT_LATENCY_MS','3000.0')
    settings = Settings.load()
    assert settings.monitor_window_size == 12 and settings.monitor_interval == .5 and settings.monitor_agent_latency_ms == 3000


@pytest.mark.parametrize('changes',[{'min_samples':41},{'window_size':0},{'agent_success_rate':0},{'max_penalty':.4},
    {'interval':float('inf')},{'window_seconds':0},{'probe_cooldown':-1},{'penalty_step':float('nan')}])
def test_invalid_monitor_policy_rejected(changes):
    with pytest.raises(ValueError): replace(MonitorPolicy(),**changes)
