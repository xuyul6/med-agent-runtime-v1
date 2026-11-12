"""A user may correct a misread emergency: only a narrow, code-matched correction clears the 30-minute hint, never a dismissal."""
from medagent.core.contracts import Question
from medagent.understanding.risk import RiskGate
from support.comparison import RehearsalModel
from support.harness import fixture, run

URGENT = "胸口剧痛，喘不上气"


def gate_after_alarm():
    gate = RiskGate(clock=lambda: 0)
    assert gate.screen(URGENT, "s").level == "CRITICAL"
    return gate


def test_correction_clears_the_hint_and_follow_up_words_no_longer_alarm():
    gate = gate_after_alarm()
    risk = gate.screen("你误会了，我刚才是在引用小说里的一段话", "s")
    assert risk.level == "ROUTINE" and risk.reasons == ["urgent_hint_cleared:correction"]
    assert gate.screen("还是这样吗", "s").level != "CRITICAL"


def test_a_new_current_sign_still_wins_over_a_correction():
    gate = gate_after_alarm()
    assert gate.screen("我是在引用，不过我现在也胸口剧痛", "s").level == "CRITICAL"


def test_dismissal_present_tense_and_quoted_corrections_do_not_clear():
    for text in ("别管了，继续帮我预约", "不是我，是我妈现在这样", "朋友发来一句“我只是在引用”"):
        gate = gate_after_alarm()
        assert gate.screen(text, "s").reasons == [], text
        assert gate.screen("还是这样，越来越难受", "s").level == "CRITICAL", text


def test_correction_without_a_hint_changes_nothing_and_resolution_is_recorded():
    gate = RiskGate(clock=lambda: 0)
    assert gate.screen("我是在引用一段话", "fresh").reasons == []
    gate = gate_after_alarm()
    assert gate.screen("已经到医院了", "s").reasons == ["urgent_hint_cleared:resolved"]


def test_runtime_answers_a_correction_without_any_model_call():
    model = RehearsalModel()
    runtime = run(fixture(model))
    first = run(runtime.chat(Question(text=URGENT, user="u", session="s")))
    assert any("引用或讨论" in step for step in first.answer.next_steps)
    before = len(model.stages)
    reply = run(runtime.chat(Question(text="你误会了，我刚才是在引用小说里的一段话", user="u", session="s")))
    assert "已按你的说明解除" in reply.answer.text and reply.risk.reasons == ["urgent_hint_cleared:correction"]
    assert len(model.stages) == before and reply.storage == "skipped"


def test_self_harm_alarm_does_not_invite_a_correction():
    runtime = run(fixture(RehearsalModel()))
    reply = run(runtime.chat(Question(text="我不想活了", user="u", session="h")))
    assert reply.risk.reasons == ["self_harm"] and not any("引用或讨论" in step for step in reply.answer.next_steps)


def test_a_correction_never_lifts_a_self_harm_hint():
    gate = RiskGate(clock=lambda: 0)
    assert gate.screen("我不想活了", "s").level == "CRITICAL"
    assert gate.screen("我刚才是在引用小说里的一段话", "s").reasons == []
    assert "s" in gate.hints


def test_the_example_offered_in_the_alarm_clears_the_hint():
    import re
    runtime = run(fixture(RehearsalModel()))
    first = run(runtime.chat(Question(text=URGENT, user="u", session="e")))
    step = next(step for step in first.answer.next_steps if "引用或讨论" in step)
    example = re.search(r"“([^”]+)”", step).group(1)
    reply = run(runtime.chat(Question(text=example, user="u", session="e")))
    assert reply.risk.reasons == ["urgent_hint_cleared:correction"], example


def test_corrections_naming_the_speaker_clear_the_hint():
    for text in ("这是我在引用的文字", "这是我在引用小说里的话", "那是我在转述别人的经历", "刚才只是我在举例"):
        gate = gate_after_alarm()
        assert gate.screen(text, "s").reasons == ["urgent_hint_cleared:correction"], text


def test_a_denied_correction_does_not_clear_the_hint():
    for text in ("我不是在引用，是真的", "这不是我在引用的，是真的", "没在开玩笑，不是在举例"):
        gate = gate_after_alarm()
        assert gate.screen(text, "s").reasons == [], text
