"""General questions use the general model path without knowledge retrieval."""
import json

import pytest

from medagent.core.contracts import Question
from medagent.understanding.classification import IntentRouter
from medagent.core.contracts import Intent
from medagent.knowledge.retrieval import Vectorizer
from medagent.core.provider import json_schema
from support.harness import Model, fixture, run


class LifeModel(Model):
    async def json(self, stage, system, content, budget=900):
        if stage == "understanding" and any(term in json.loads(content)["current"]["text"] for term in ("书桌", "辣椒炒肉")):
            return {"scores":{"general":1}, "topic_relation":"new"}
        return await super().json(stage, system, content, budget)

    async def message(self, stage, system, messages, tools=None, budget=1800):
        if stage == "agent_general":
            assert "普通生活" in system
            data = {"role":"general", "status":"answered", "missing_fields":[],
                    "answer":{"text":"先清理桌面，再按使用频率收纳物品。", "reference_keys":[]},
                    "payload":{"task_summary":"回答书桌整理问题。", "suggested_role":None}}
            return {"content":[{"type":"text", "text":json.dumps(data, ensure_ascii=False)}]}
        return await super().message(stage, system, messages, tools, budget)


def test_understanding_schema_does_not_ask_for_an_unused_topic_relation():
    assert "topic_relation" not in json_schema("understanding")["properties"]


def test_general_life_question_uses_model_without_reference_lookup():
    async def scenario():
        runtime = await fixture(LifeModel())
        reply = await runtime.chat(Question(text="怎样整理书桌？", user="synthetic", session="life"))
        assert reply.primary_agent == "general"
        assert reply.agent_results[0].status == "answered"
        assert "收纳" in reply.answer.text and not reply.references
        assert not any(e.get("name") == "lookup_reference" for e in runtime.monitor.traces[-1]["events"])
    run(scenario())


def test_unmarked_life_question_after_fever_routes_to_general():
    async def scenario():
        runtime = await fixture(LifeModel())
        for text in ("发烧吃什么", "昨天晚上开始低烧"):
            await runtime.chat(Question(text=text, user="synthetic", session="unmarked"))
        reply = await runtime.chat(Question(text="辣椒炒肉怎么做", user="synthetic", session="unmarked"))
        assert reply.primary_agent == "general" and reply.agent_results[0].status == "answered"
        assert not reply.answer.questions and not reply.references
        assert not any(e.get("event") == "classification_rule" for e in runtime.monitor.traces[-1]["events"])
    run(scenario())


class ScoreModel(Model):
    def __init__(self, scores):
        self.scores = scores

    async def json(self, stage, system, content, budget=900):
        return {"scores":self.scores}


@pytest.mark.parametrize("text,scores,primary", [
    # A follow-up the model reads as medical stays medical, even when it is not fully sure.
    ("三天了", {"symptoms":.8, "general":.2}, Intent.SYMPTOMS),
    ("刚才说错了", {"symptoms":.7, "general":.3}, Intent.SYMPTOMS),
    # The model reads a new topic, and no rule pulls it back to the previous task.
    ("辣椒炒肉怎么做", {"general":1}, Intent.GENERAL),
    ("三天了", {"general":1}, Intent.GENERAL),
    # A turn without keywords leaves the keyword channel silent instead of voting general against the model.
    ("三天了", {"symptoms":.6, "general":.4}, Intent.SYMPTOMS),
    ("那单位呢", {"report":.55, "general":.45}, Intent.REPORT),
])
def test_followups_follow_the_fused_model_score(text, scores, primary):
    async def scenario():
        result = await IntentRouter(ScoreModel(scores)).understand(text, {"last_intent":"symptoms"}, turn_id="synthetic")
        assert result.primary == primary
    run(scenario())


@pytest.mark.parametrize("text", ["三天了", "刚才说错了", "那单位呢"])
def test_without_the_model_a_short_followup_keeps_the_previous_task(text):
    async def scenario():
        for prior, expected in (("symptoms", Intent.SYMPTOMS), ("report", Intent.REPORT)):
            result = await IntentRouter(None).understand(text, {"last_intent":prior}, turn_id="synthetic")
            assert result.primary == expected and "language_unavailable" in result.fallbacks
    run(scenario())


def test_critical_medical_request_still_preempts_general_model():
    async def scenario():
        runtime = await fixture(LifeModel())
        reply = await runtime.chat(Question(text="突然胸痛、呼吸困难", user="synthetic", session="urgent"))
        assert reply.primary_agent == "risk_handoff" and reply.risk.level == "CRITICAL"
        assert reply.usage["calls"] == 0
    run(scenario())


def test_explicit_new_topic_does_not_inherit_previous_medical_intent():
    async def scenario():
        runtime = await fixture(LifeModel())
        await runtime.chat(Question(text="我咳嗽三天，想整理症状记录。", user="synthetic", session="switch"))
        reply = await runtime.chat(Question(text="另外一个问题，怎样整理书桌？", user="synthetic", session="switch"))
        assert reply.primary_agent == "general"
        assert reply.agent_results[0].status == "answered" and "收纳" in reply.answer.text
        assert not reply.references
    run(scenario())


def test_a_symptom_followup_gets_no_support_task_from_the_previous_task_wording():
    async def scenario():
        result = await IntentRouter(ScoreModel({"symptoms":.8, "general":.2})).understand("三天了", {"last_intent":"symptoms"}, turn_id="synthetic")
        assert [task.intent for task in result.tasks] == [Intent.SYMPTOMS]
        assert "keywords" not in result.signals
    run(scenario())


def test_without_the_model_a_turn_without_keywords_is_general():
    async def scenario():
        result = await IntentRouter(None).understand("在吗", {}, turn_id="synthetic")
        assert result.primary == Intent.GENERAL and result.signals["keywords"]["general"] == 1
    run(scenario())
