"""Exercise both support-role thresholds through chat with controlled understanding-model scores."""
import pytest

from medagent.core.contracts import Question
from support.harness import fixture, run


class FixedUnderstanding:
    def __init__(self, scores, tasks):
        self.scores = scores
        self.tasks = tasks

    async def json(self, stage, system, content, budget):
        assert stage == "understanding"
        return {"scores": self.scores, "tasks": [
            {"intent": intent, "goal": quote, "quote": quote} for intent, quote in self.tasks
        ]}


def route(text, scores, tasks):
    async def scenario():
        runtime = await fixture()
        runtime.router.model = FixedUnderstanding(scores, tasks)
        return await runtime.chat(Question(text=text))
    return run(scenario())


@pytest.mark.parametrize("visit_score,expected_support", [
    (.119, []), (.120, ["visit_guide"]), (.121, ["visit_guide"]),
])
def test_support_with_its_own_request_at_point_12(visit_score, expected_support):
    symptom_request = "帮我整理头痛的症状"
    visit_request = "也想知道见医生那天得带些什么"
    # Only the model recognizes this wording for a visit. Give it the score needed
    # to put the fused visit score immediately below, at, or above the gate.
    language_visit = visit_score / (9 / 16)
    reply = route(symptom_request + "，" + visit_request + "。",
                  {"symptoms": 1 - language_visit, "visit": language_visit},
                  [("symptoms", symptom_request), ("visit", visit_request)])
    assert reply.intent.signals["keywords"]["visit"] == 0
    assert reply.intent.scores["visit"] == pytest.approx(visit_score)
    assert reply.primary_agent == "symptom_guide"
    assert reply.supporting_agents == expected_support


@pytest.mark.parametrize("report_score,expected_support", [
    (.299, []), (.300, ["report_explainer"]), (.301, ["report_explainer"]),
])
def test_strong_support_under_general_at_point_30(report_score, expected_support):
    general_request = "你好，周末穿什么合适"
    visit_request = "顺便帮我列出门诊复查前要带的东西"
    report_request = "也想弄懂那张单子的单位"
    # The model recognizes a report request whose wording misses the request patterns.
    # The visit has explicit lexical evidence, so it stays in the plan at all three scores.
    language_report = report_score / (9 / 16)
    reply = route("，".join([general_request, visit_request, report_request]) + "。",
                  {"general": .25, "report": language_report, "visit": .75 - language_report},
                  [("general", general_request), ("report", report_request)])
    assert reply.intent.primary.value == "general"
    assert reply.intent.signals["keywords"]["report"] == 0
    assert reply.intent.scores["report"] == pytest.approx(report_score)
    assert reply.primary_agent == "visit_guide"
    assert reply.supporting_agents == expected_support


def test_explicit_report_request_routes_even_below_the_keyword_candidate_threshold():
    symptom_request = "帮我整理头痛的症状"
    text = "你好，最近在吃布洛芬，" + symptom_request + "，顺便解释报告里的项目。"
    reply = route(text, {"symptoms": 1}, [("symptoms", symptom_request)])
    assert reply.intent.scores["report"] == pytest.approx(.109375)
    assert "report" not in [task.intent.value for task in reply.intent.tasks]
    assert reply.primary_agent == "symptom_guide"
    assert reply.supporting_agents == ["report_explainer"]
