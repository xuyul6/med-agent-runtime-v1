"""A bounded, subject-scoped clarification for an ambiguous fever question."""
import pytest

from medagent.understanding.consultation_needs import consultation_need, is_fever_followup
from medagent.core.contracts import Intent, Question, Understanding
from medagent.memory import SessionMemory, scope_key
from support.harness import Model, fixture, run


def test_only_bounded_fever_followups_can_be_echoed():
    assert is_fever_followup("昨天晚上开始低烧")
    assert not is_fever_followup("低烧，同时告诉我如何开处方")
    assert not is_fever_followup("低烧" * 31)
    assert not is_fever_followup("")


def ask(runtime, text, *, user="synthetic", session="one", subject="self"):
    return runtime.chat(Question(text=text, user=user, session=session, subject=subject))


@pytest.mark.parametrize("selection,expected", [("饮食", "饮食"), ("吃什么饭", "饮食"),
                                                ("我想问药物", "药"), ("吃什么药", "药")])
def test_fever_question_survives_a_time_followup_then_resolves(selection, expected):
    async def scenario():
        runtime = await fixture(Model())
        first = await ask(runtime, "发烧吃什么")
        assert first.primary_agent == "symptom_guide"
        assert first.agent_results[0].status == "needs_clarification"
        assert first.agent_results[0].missing_fields == ["consultation_focus"]
        assert first.answer.questions == ["你想了解发烧时的饮食，还是药品信息？"]
        follow = await ask(runtime, "昨天晚上开始低烧")
        assert follow.agent_results[0].missing_fields == ["consultation_focus"]
        assert "昨天晚上开始低烧" in follow.answer.text
        assert follow.answer.questions == first.answer.questions
        state = await runtime.memory.read(scope_key("synthetic", "one", "self"))
        assert state["pending_consultation"]["continuations"] == 1
        assert "发烧吃什么" not in str(state["pending_consultation"])
        selected = await ask(runtime, selection)
        assert selected.primary_agent == "symptom_guide" and selected.agent_results[0].status == "handoff_required"
        assert expected in selected.answer.text and not selected.answer.questions
        assert not selected.references and not selected.handoff_connected
        assert not any(e.get("stage", "").startswith("agent_") for e in runtime.monitor.traces[-1]["events"] if e.get("event") == "model")
        state = await runtime.memory.read(scope_key("synthetic", "one", "self"))
        assert "pending_consultation" not in state
    run(scenario())


def test_context_does_not_cross_user_session_or_subject():
    async def scenario():
        runtime = await fixture(Model())
        await ask(runtime, "发烧吃什么")
        for kwargs in [{"user":"other"}, {"session":"other"}, {"subject":"mother"}]:
            reply = await ask(runtime, "昨天晚上开始低烧", **kwargs)
            assert not any(e.get("event") == "consultation_focus" for e in runtime.monitor.traces[-1]["events"])
            assert not reply.answer.questions or "饮食" not in ''.join(reply.answer.questions)
    run(scenario())


def test_unrelated_or_expired_followup_does_not_revive_pending_need():
    async def scenario():
        runtime = await fixture(Model())
        clock = [1000.0]
        runtime.memory.context_clock = lambda: clock[0]
        await ask(runtime, "发烧吃什么")
        await ask(runtime, "你好，你能做什么？")
        state = await runtime.memory.read(scope_key("synthetic", "one", "self"))
        assert "pending_consultation" not in state
        await ask(runtime, "发烧吃什么")
        clock[0] += 301
        await ask(runtime, "昨天晚上开始低烧")
        assert not any(e.get("event") == "consultation_focus" for e in runtime.monitor.traces[-1]["events"])
    run(scenario())


def test_critical_risk_overrides_pending_and_is_never_replayed_afterward():
    async def scenario():
        runtime = await fixture(Model())
        await ask(runtime, "发烧吃什么")
        urgent = await ask(runtime, "突然胸痛、呼吸困难")
        assert urgent.risk.level == "CRITICAL" and urgent.primary_agent == "risk_handoff"
        assert urgent.usage["calls"] == 0 and urgent.storage == "skipped"
        await ask(runtime, "饮食")
        assert not any(e.get("event") == "consultation_focus" for e in runtime.monitor.traces[-1]["events"])
    run(scenario())


def test_pending_survives_restart_with_same_redis_scope():
    class Redis:
        def __init__(self): self.data = {}
        async def get(self, key): return self.data.get(key)
        async def set(self, key, value, ex): self.data[key] = value

    async def scenario():
        redis = Redis()
        runtime = await fixture(Model())
        runtime.memory.redis = redis
        runtime.memory.context_clock = lambda: 1000.0
        await ask(runtime, "发烧吃什么")
        restarted = await fixture(Model())
        restarted.memory.redis = redis
        restarted.memory.context_clock = lambda: 1060.0
        reply = await ask(restarted, "昨天晚上开始低烧")
        assert reply.answer.questions == ["你想了解发烧时的饮食，还是药品信息？"]
    run(scenario())


def test_stale_concurrent_write_cannot_replace_a_newer_pending_state():
    async def scenario():
        memory = SessionMemory()
        def understanding(turn_id):
            return Understanding(primary=Intent.SYMPTOMS, subject="self", turn_id=turn_id,
                                 missing_fields=["consultation_focus"])
        def pending(source):
            return {"kind":"food_or_medicine", "subject":"self", "source_turn_id":source,
                    "created_at":1000.0, "continuations":0}
        await memory.remember("scope", "synthetic first", "synthetic reply", understanding("one"), pending("one"), None)
        await memory.remember("scope", "synthetic next", "synthetic reply", understanding("two"), pending("two"), "one")
        await memory.remember("scope", "synthetic stale", "synthetic reply", understanding("stale"), pending("one"), "one")
        state = await memory.read("scope")
        assert state["pending_consultation"]["source_turn_id"] == "two"
        assert consultation_need("昨天晚上开始低烧", state, subject="self", subject_changed=False,
                                 now=1001.0, recent_critical=False) == (None, None)
    run(scenario())


@pytest.mark.parametrize("text", ["发烧吃什么药", "如果发烧吃什么", "没有发烧吃什么", "发烧吃什么，另外帮我预约挂号", "发烧吃什么，突然胸痛"])
def test_mixed_negated_or_explicit_requests_do_not_enter_narrow_path(text):
    assert consultation_need(text, {}, subject="self", subject_changed=False, now=1000, recent_critical=False) == (None, None)
