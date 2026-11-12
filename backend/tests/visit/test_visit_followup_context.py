"""Limited continuation of an explicit same-subject visit task."""
import json
import hashlib

import pytest

from medagent.core.contracts import Question
from medagent.understanding.conversation import recover_visit_followup
from medagent.memory import scope_key
from support.harness import Model, fixture, run


def prior_state(text="就医前需要准备哪些资料？"):
    return {"last_intent":"visit", "last_intent_turn_id":"turn-1", "last_user_at":1000.0, "turn_counter":1,
            "messages":[{"role":"user", "content":text, "turn_id":"turn-1", "turn_index":1},
                        {"role":"assistant", "content":"合成回复", "turn_id":"turn-1", "turn_index":1}]}


@pytest.mark.parametrize("current", ["接下来准备哪些资料？", "还需要带什么材料？", "这些资料需要整理吗？"])
def test_recent_explicit_user_task_provides_one_bounded_clause(current):
    previous = "昨天看报告，就医前需要准备哪些资料？"
    found = recover_visit_followup(current, prior_state(previous), subject_changed=False, now=1060.0)
    assert found is not None and found.current == current
    assert found.prior_clause == "就医前需要准备哪些资料？"
    assert found.prior_turn_id == "turn-1" and found.age_seconds == 60


@pytest.mark.parametrize("current", [
    "出国旅行需要准备哪些资料？", "不需要准备资料了", "给我妈妈准备哪些资料？",
    "请给我推荐药品", "这些证件是必须的吗？", "医院地址在哪里？",
])
def test_topic_change_negation_and_specific_facts_cannot_reuse_visit_context(current):
    assert recover_visit_followup(current, prior_state(), subject_changed=False, now=1060) is None


@pytest.mark.parametrize("previous", [
    "不去医院，出国旅行需要准备资料。", "虚构青山医院挂号需要准备什么材料？",
    "以前去过医院，整理登机清单。", "只整理资料，其他不用问。", "看病历时整理材料。",
    "去宠物医院需要准备哪些资料？", "不是就医，只是准备资料。",
    "就医前需要准备资料，还想查询药品禁忌。",
    "别去看医生了，别准备资料。", "看医生前没必要准备资料。",
])
def test_negated_institution_specific_or_nonmedical_prior_is_not_admitted(previous):
    assert recover_visit_followup("接下来准备哪些资料？", prior_state(previous), subject_changed=False, now=1060) is None


@pytest.mark.parametrize("patch,now", [
    ({"last_user_at":None}, 1060), ({"last_user_at":float("nan")}, 1060),
    ({"last_user_at":1000.0}, 1301), ({"last_user_at":1100.0}, 1060),
    ({"last_intent":"report"}, 1060), ({"last_intent_turn_id":"other"}, 1060),
    ({"turn_counter":2}, 1060), ({"messages":[]}, 1060),
])
def test_stale_legacy_mismatched_or_invalid_state_fails_closed(patch, now):
    state = prior_state() | patch
    assert recover_visit_followup("接下来准备哪些资料？", state, subject_changed=False, now=now) is None


def test_same_source_must_be_immediately_previous_complete_turn_and_same_subject():
    state = prior_state()
    assert recover_visit_followup("接下来准备哪些资料？", state, subject_changed=True, now=1060) is None
    state["messages"][-1]["turn_id"] = "other"
    assert recover_visit_followup("接下来准备哪些资料？", state, subject_changed=False, now=1060) is None


def test_age_boundary_and_assistant_text_cannot_supply_a_user_anchor():
    assert recover_visit_followup("接下来准备哪些资料？", prior_state(), subject_changed=False, now=1300) is not None
    state = prior_state("介绍你的功能")
    state["messages"][-1]["content"] = "就医前需要准备哪些资料？"
    assert recover_visit_followup("接下来准备哪些资料？", state, subject_changed=False, now=1060) is None


@pytest.mark.parametrize("mode", ["selective", "full"])
def test_same_subject_recent_followup_uses_template_without_exposing_history_in_trace(mode):
    async def scenario():
        clock = [1000.0]
        runtime = await fixture(Model())
        runtime.memory.context_clock = lambda:clock[0]
        first = await runtime.chat(Question(text="就医前需要准备哪些资料？", user="synthetic", session="dialogue"))
        clock[0] = 1060.0
        reply = await runtime.chat(Question(text="接下来准备哪些资料？", user="synthetic", session="dialogue", context=mode))
        visit = next(item for item in reply.agent_results if item.role == "visit_guide")
        assert visit.status == "answered" and [ref.key for ref in reply.references] == ["consultation-record"]
        events = runtime.monitor.traces[-1]["events"]
        recovered = [event for event in events if event["event"] == "visit_context_recovered"]
        assert len(recovered) == 1 and recovered[0]["age_seconds"] == 60
        assert recovered[0]["source_turn_hash"] == hashlib.sha256(first.request_id.encode()).hexdigest()[:16]
        assert "就医前需要准备哪些资料" not in json.dumps(runtime.monitor.traces[-1], ensure_ascii=False)
        state = await runtime.memory.read(scope_key("synthetic", "dialogue", "self"))
        assert state["messages"][-4]["turn_id"] == first.request_id

    run(scenario())


def test_expired_context_and_second_implicit_followup_require_clarification():
    async def scenario():
        clock = [1000.0]
        runtime = await fixture(Model())
        runtime.memory.context_clock = lambda:clock[0]
        await runtime.chat(Question(text="就医前需要准备哪些资料？", user="synthetic", session="expired"))
        clock[0] = 1301.0
        expired = await runtime.chat(Question(text="接下来准备哪些资料？", user="synthetic", session="expired"))
        assert expired.agent_results[0].missing_fields == ["visit_purpose"] and not expired.references
        clock[0] = 2000.0
        await runtime.chat(Question(text="就医前需要准备哪些资料？", user="synthetic", session="chain"))
        second = await runtime.chat(Question(text="接下来准备哪些资料？", user="synthetic", session="chain"))
        assert second.agent_results[0].status == "answered"
        third = await runtime.chat(Question(text="还需要带什么材料？", user="synthetic", session="chain"))
        assert third.agent_results[0].missing_fields == ["visit_purpose"] and not third.references

    run(scenario())


def test_context_never_crosses_user_session_subject_or_storage_failure():
    async def scenario():
        runtime = await fixture(Model())
        await runtime.chat(Question(text="就医前需要准备哪些资料？", user="one", session="shared"))
        questions = [Question(text="接下来准备哪些资料？", user="two", session="shared"),
                     Question(text="接下来准备哪些资料？", user="one", session="other"),
                     Question(text="现在给我妈妈准备哪些资料？", user="one", session="shared")]
        for question in questions:
            reply = await runtime.chat(question)
            assert not reply.references
            assert not any(event["event"] == "visit_context_recovered" for event in runtime.monitor.traces[-1]["events"])
        runtime.memory.available = False
        reply = await runtime.chat(Question(text="接下来准备哪些资料？", user="one", session="shared", subject="self"))
        assert reply.storage == "degraded" and not reply.references
        assert not any(event["event"] == "visit_context_recovered" for event in runtime.monitor.traces[-1]["events"])

    run(scenario())


def test_recovery_survives_runtime_restart_with_same_scoped_redis_state():
    class Redis:
        def __init__(self): self.data = {}
        async def get(self, key): return self.data.get(key)
        async def set(self, key, value, ex):
            assert ex == 86400
            self.data[key] = value

    async def scenario():
        redis = Redis()
        first_runtime = await fixture(Model())
        first_runtime.memory.redis = redis
        first_runtime.memory.context_clock = lambda:1000.0
        await first_runtime.chat(Question(text="就医前需要准备哪些资料？", user="synthetic", session="restart", subject="self"))
        restarted = await fixture(Model())
        restarted.memory.redis = redis
        restarted.memory.context_clock = lambda:1060.0
        reply = await restarted.chat(Question(text="接下来准备哪些资料？", user="synthetic", session="restart", subject="self"))
        assert reply.agent_results[0].status == "answered" and [ref.key for ref in reply.references] == ["consultation-record"]
        assert any(event["event"] == "visit_context_recovered" for event in restarted.monitor.traces[-1]["events"])

    run(scenario())


def test_critical_risk_still_short_circuits_followup_memory_and_model():
    async def scenario():
        runtime = await fixture(Model())
        await runtime.chat(Question(text="就医前需要准备哪些资料？", user="synthetic", session="risk"))
        reply = await runtime.chat(Question(text="现在呼吸困难，接下来准备哪些资料？", user="synthetic", session="risk"))
        assert reply.primary_agent == "risk_handoff" and reply.storage == "skipped" and reply.usage["calls"] == 0
        assert not any(event["event"] == "visit_context_recovered" for event in runtime.monitor.traces[-1]["events"])

    run(scenario())
