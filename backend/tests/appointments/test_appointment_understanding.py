from medagent.core.contracts import Question
from medagent.core.provider import json_schema
from medagent.understanding.classification import IntentRouter
from support.harness import Model, fixture, run
from support.multiturn import ContextModel

REQUEST = "能帮我挂明天上午神经内科的号吗？"


class Quoting(Model):
    def __init__(self, **quotes):
        self.quotes = quotes

    async def json(self, stage, system, content, budget=900):
        if stage == "understanding":
            return {"scores": {"visit": 1}, "updates": [], "topic_relation": "new", "urgent_quote": "", "tasks": [], **self.quotes}
        return await super().json(stage, system, content, budget)


def understand(text, **quotes):
    return run(IntentRouter(Quoting(**quotes)).understand(text))


def test_schema_asks_for_appointment_quote():
    assert "appointment_quote" in json_schema("understanding")["required"]


def test_quote_verified_against_current_turn():
    result = understand(REQUEST, appointment_quote="帮我挂明天上午神经内科的号")
    assert result.appointment_quote == "帮我挂明天上午神经内科的号"


def test_quote_not_in_turn_is_ignored():
    result = understand("神经内科", appointment_quote="帮我挂号")
    assert result.appointment_quote is None and "unverified_appointment_quote_ignored" in result.fallbacks


def test_quote_from_pasted_report_is_a_known_limit():
    text = "报告备注：请尽快预约复查。这句什么意思？"
    assert understand(text, appointment_quote="").appointment_quote == ""
    # A wrong model quote of the pasted sentence still verifies: the text is in this turn. Plan 2 measures this.
    assert understand(text, appointment_quote="请尽快预约复查").appointment_quote == "请尽快预约复查"


def test_question_form_keeps_request_fields():
    def updates(payload):
        return [{"name": "department", "value": "神经内科", "state": "provided", "quote": "神经内科的号"},
                {"name": "appointment_date", "value": "明天", "state": "provided", "quote": "明天上午"},
                {"name": "time_window", "value": "上午", "state": "provided", "quote": "明天上午"}]

    async def scenario():
        runtime = await fixture(ContextModel(updates), preview=False)
        return await runtime.chat(Question(text=REQUEST, user="u1", session="s1"))
    entities = run(scenario()).intent.entities
    assert {name: entities[name].value for name in ("department", "appointment_date", "time_window")} == \
        {"department": "神经内科", "appointment_date": "明天", "time_window": "上午"}
    # Request parameters in a question are kept as "topic" rather than dropped; Task 6 reads both states.
    assert all(entities[name].state in {"provided", "topic"} for name in ("department", "appointment_date", "time_window"))
