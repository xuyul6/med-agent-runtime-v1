from medagent.core.contracts import Question
from medagent.core.provider import json_schema
from medagent.evaluation.contracts import EvalLimits
from medagent.evaluation.risk_redteam import run_with_model
from medagent.core.config import ROOT, Settings
from dataclasses import replace
from medagent.runtime import MedicalRuntime
from support.harness import Model, fixture, run

MESSAGE = "我爸喘气特别费劲，一句话都说不完整"
PARAPHRASE = ROOT/"resources"/"risk-redteam-paraphrase-v1.json"
LIMITS = EvalLimits(max_model_calls=30, max_output_tokens=30*1800, per_call_output_tokens=1800)


class Signal(Model):
    def __init__(self, quote):
        self.quote, self.roles = quote, 0

    async def json(self, stage, system, content, budget=900):
        result = await super().json(stage, system, content, budget)
        return {**result, "urgent_quote":self.quote} if stage == "understanding" else result

    async def message(self, stage, system, messages, tools=None, budget=1800):
        self.roles += 1
        return await super().message(stage, system, messages, tools, budget)


class Broken(Signal):
    async def json(self, stage, system, content, budget=900):
        if stage == "understanding":
            raise RuntimeError("model_down")
        return await super().json(stage, system, content, budget)


def test_schema_requires_the_signal():
    assert "urgent_quote" in json_schema("understanding")["required"]


def test_grounded_quote_goes_to_the_urgent_path_without_role_calls():
    model = Signal("喘气特别费劲")
    async def scenario():
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text=MESSAGE, session="a"))
        assert reply.risk.level == "CRITICAL" and reply.risk.reasons == ["model_urgent_signal"]
        assert reply.primary_agent == "risk_handoff" and reply.storage == "skipped" and "急" in reply.answer.text
        followup = await runtime.chat(Question(text="还是这样，越来越严重", session="a"))
        assert followup.risk.level == "CRITICAL"
    run(scenario())
    assert model.roles == 0


def test_the_quote_is_not_returned_to_clients():
    async def scenario():
        runtime = await fixture(Signal(""))
        reply = await runtime.chat(Question(text="你好"))
        assert "urgent_quote" not in reply.model_dump()["intent"]
    run(scenario())


def test_ungrounded_quote_is_ignored():
    model = Signal("胸口剧痛")
    async def scenario():
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text=MESSAGE))
        assert reply.risk.level != "CRITICAL" and reply.primary_agent != "risk_handoff"
        assert "unverified_urgent_ignored" in reply.intent.fallbacks
    run(scenario())


def test_empty_signal_changes_nothing():
    async def scenario():
        runtime = await fixture(Signal(""))
        reply = await runtime.chat(Question(text=MESSAGE))
        assert reply.risk.level == "ROUTINE" and reply.primary_agent != "risk_handoff"
    run(scenario())


def test_model_failure_keeps_the_rule_result():
    async def scenario():
        runtime = await fixture(Broken("喘气特别费劲"))
        assert (await runtime.chat(Question(text="呼吸很困难"))).risk.level == "CRITICAL"
        assert (await runtime.chat(Question(text=MESSAGE))).risk.level == "ROUTINE"
    run(scenario())


def test_rule_hit_skips_the_model_so_it_cannot_lower_a_level():
    class Refuses(Signal):
        async def json(self, stage, system, content, budget=900):
            assert stage != "understanding"
            return await super().json(stage, system, content, budget)
    async def scenario():
        runtime = await fixture(Refuses(""))
        assert (await runtime.chat(Question(text="呼吸很困难"))).risk.level == "CRITICAL"
    run(scenario())


def test_single_agent_arm_escalates_the_same_way():
    model = Signal("喘气特别费劲")
    async def scenario():
        base = await fixture(model)
        runtime = MedicalRuntime(replace(base.settings, architecture="single_agent"), model, base.vectors, base.index, base.memory)
        assert (await runtime.chat(Question(text=MESSAGE))).primary_agent == "risk_handoff"
    run(scenario())
    assert model.roles == 0


def test_online_report_keeps_rule_and_model_results_apart():
    class Echo(Model):
        async def json(self, stage, system, content, budget=900):
            result = await super().json(stage, system, content, budget)
            return {**result, "urgent_quote":"喘气特别费劲" if "喘气特别费劲" in content else ""}
    report = run(run_with_model(PARAPHRASE, Echo(), LIMITS, "fake"))
    row = next(row for row in report["cases"] if row["id"] == "p-c6")
    assert row["rule_actual"] == "ROUTINE" and row["actual"] == "CRITICAL" and row["model_escalated"]
    assert report["kind"] == "online_risk_redteam" and report["unchecked_turns"] == 0
    assert report["summary"]["critical"]["detected"] == report["rule_summary"]["critical"]["detected"]+1


def test_failed_model_calls_are_reported_as_unchecked():
    report = run(run_with_model(PARAPHRASE, Broken(""), LIMITS))
    assert report["failed_model_calls"] == report["unchecked_turns"] > 0 and report["usage"]["attempted_calls"] > 0
