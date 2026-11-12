import json
import httpx
from medagent.core.config import Settings
from medagent.core.contracts import Question
from medagent.core.observability import RunLog
from medagent.core.provider import ClaudeGateway, ModelFailure, fallback_gateway
from medagent.monitoring import Monitor, MonitorPolicy
from medagent.runtime import MedicalRuntime
from medagent.smoke.fallback import score as standby_score
from medagent.evaluation.contracts import EvalLimits
from support.harness import Model, fixture, run

QUESTION = "你好，请介绍一下你支持哪些能力。"


class Down(Model):
    async def message(self, stage, system, messages, tools=None, budget=1800):
        raise ModelFailure("model_request_failed")


class Counting(Model):
    def __init__(self):
        self.calls = 0

    async def message(self, stage, system, messages, tools=None, budget=1800):
        self.calls += 1
        return await super().message(stage, system, messages, tools, budget)


async def wired(primary, standby):
    runtime = await fixture(primary)
    return MedicalRuntime(runtime.settings, primary, runtime.vectors, runtime.index, runtime.memory, fallback_model=standby)


def captured_gateway(settings):
    seen = []
    def handler(request):
        seen.append((str(request.url), request.headers.get("x-api-key"), json.loads(request.content)))
        return httpx.Response(200, json={"model":"m", "stop_reason":"end_turn", "content":[{"type":"text","text":"{}"}],
                                         "usage":{"input_tokens":1, "output_tokens":1}})
    return settings, seen, httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_no_key_means_no_standby():
    assert fallback_gateway(Settings(api_key="primary", model="claude")) is None


def test_standby_uses_its_own_vendor_and_puts_the_schema_in_the_prompt():
    settings, seen, client = captured_gateway(Settings(api_key="primary", model="claude", fallback_api_key="standby"))
    gateway = fallback_gateway(settings, client)
    run(gateway.message("agent_general", "规则", [{"role":"user", "content":"问"}]))
    url, key, payload = seen[0]
    assert url == "https://api.deepseek.com/anthropic/v1/messages" and key == "standby"
    assert payload["model"] == "deepseek-flash" and payload["thinking"] == {"type":"disabled"}
    assert "output_config" not in payload and "JSON Schema" in payload["system"] and "reference_keys" in payload["system"]


def test_primary_keeps_provider_side_schema():
    settings, seen, client = captured_gateway(Settings(api_key="primary", model="claude", fallback_api_key="standby"))
    run(ClaudeGateway(settings, client).message("agent_general", "规则", [{"role":"user", "content":"问"}]))
    url, key, payload = seen[0]
    assert url.startswith("https://api.anthropic.com/") and key == "primary"
    assert "output_config" in payload and payload["system"] == "规则"


def test_standby_stays_idle_while_the_primary_works():
    primary, standby = Counting(), Counting()
    async def scenario():
        runtime = await wired(primary, standby)
        for index in range(6):
            reply = await runtime.chat(Question(text=QUESTION, session=f"s{index}"))
            assert reply.routing_decision.assignments[0].selected_instance == "general.1"
    run(scenario())
    assert primary.calls == 6 and standby.calls == 0


def test_without_a_standby_the_idle_instance_still_takes_traffic():
    """Documents why the standby flag exists: an unused instance outranks one that has latency samples."""
    monitor = Monitor(MonitorPolicy.from_settings(Settings()))
    monitor.register_agent("general.1", "general")
    monitor.register_agent("general.2", "general")
    monitor.record("general.1", True, 5000)
    assert monitor.routing_state("general.2")["health"] > monitor.routing_state("general.1")["health"]


def test_standby_answers_when_the_primary_fails():
    standby = Counting()
    async def scenario():
        runtime = await wired(Down(), standby)
        reply = await runtime.chat(Question(text=QUESTION))
        assignment = reply.routing_decision.assignments[0]
        assert assignment.selected_instance == "general.2" and reply.answer.text
        assert [attempt.status for attempt in assignment.attempts] == ["failed", "ok"]
    run(scenario())
    assert standby.calls == 1


def test_only_slot_two_moves_to_the_standby():
    primary, standby = Counting(), Counting()
    async def scenario():
        runtime = await wired(primary, standby)
        for agents in runtime.agents.values():
            assert agents[0].model is primary and not agents[0].standby
            assert agents[1].model is standby and agents[1].standby
    run(scenario())


def test_no_standby_leaves_routing_unchanged():
    primary = Counting()
    async def scenario():
        runtime = await wired(primary, None)
        assert all(agent.model is primary and not agent.standby for agents in runtime.agents.values() for agent in agents)
    run(scenario())


def test_usage_counts_cached_input():
    log = RunLog()
    log.events.append({"event":"model", "input_tokens":10, "output_tokens":2, "cache_read_input_tokens":90, "cache_creation_input_tokens":None})
    assert log.usage()["input_tokens"] == 100


def test_smoke_reports_roles_served_by_the_standby():
    report = run(standby_score(Counting(), EvalLimits(max_model_calls=30, max_output_tokens=30*1900, per_call_output_tokens=1900), "fake"))
    assert report["total"] == 7 and report["completed"] == 7 and report["served_by_standby"] >= 1
    assert all(row["selected"] for row in report["cases"]) and report["usage"]["attempted_calls"] >= 1
