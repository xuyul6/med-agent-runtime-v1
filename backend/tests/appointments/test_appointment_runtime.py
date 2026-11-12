import json
from types import SimpleNamespace as NS

import pytest

from medagent.benchmark.runner import observe
from medagent.core.contracts import Question
from medagent.evaluation.contracts import DialogCase, ExpectedBehavior
from medagent.evaluation.runner import run_dialog
from medagent.memory import scope_key
from support.appointments import BookingModel, Clock, booking_store, event_reason, trace_of
from support.harness import fixture, role_data, run

REQUEST = "能帮我挂明天上午神经内科的号吗？"


@pytest.fixture
def setup():
    clock, model = Clock(), BookingModel()
    runtime = run(fixture(model, preview=False, appointments=booking_store(clock)))
    return runtime, model, clock


def ask(runtime, text, session="", subject="self", user="u1"):
    return run(runtime.chat(Question(text=text, user=user, session=session, subject=subject)))


def roles(reply):
    return [reply.primary_agent, *reply.supporting_agents]


def test_booking_turn_returns_one_pending_action(setup):
    runtime, _, _ = setup
    [action] = ask(runtime, REQUEST).pending_actions
    assert action.kind == "appointment_create" and "神经内科" in action.summary and action.summary.endswith("就诊人：本人")


def test_appointment_model_sees_only_structured_data(setup):
    runtime, model, _ = setup
    ask(runtime, REQUEST)
    data = role_data(model.appointment_messages[0])
    assert not {"current_request", "history", "summary", "episodes", "references"} & set(data)
    assert data["candidates"][0]["slot_id"] == "demo-general:neurology:2026-11-04:am" and REQUEST not in json.dumps(data, ensure_ascii=False)


def test_token_never_leaves_the_reply(setup):
    runtime, model, _ = setup
    reply = ask(runtime, REQUEST)
    token = reply.pending_actions[0].token
    assert token not in json.dumps(trace_of(runtime, reply), ensure_ascii=False)
    assert token not in json.dumps(model.all_messages, ensure_ascii=False)
    assert token not in json.dumps(run(runtime.memory.read(scope_key("u1", reply.session, "self"))), ensure_ascii=False)


def test_report_rows_have_no_token(setup):
    runtime, _, _ = setup
    row = {}
    turn = NS(message=REQUEST, subject=None, expected=ExpectedBehavior())
    reply = run(observe(runtime, turn, "u1", "", "self", "selective", NS(entries=[]), row))
    assert reply.pending_actions and reply.pending_actions[0].token not in json.dumps(row, ensure_ascii=False)
    assert row["reply"]["pending_actions"][0]["token"] == "[redacted]"
    dialog = run(run_dialog(runtime, DialogCase(id="booking", turns=[REQUEST]), "run1", NS(judge=False), NS(entries=[], judge_model=None, exhausted=None)))
    saved = dialog["turns"][0]["reply"]["pending_actions"]
    assert saved and saved[0]["token"] == "[redacted]"


def test_three_turn_chain_end_to_end(setup):
    runtime, _, _ = setup
    first = ask(runtime, "帮我挂个号")
    second = ask(runtime, "神经内科", session=first.session)
    third = ask(runtime, "明天上午", session=first.session)
    assert first.pending_actions == second.pending_actions == [] and len(third.pending_actions) == 1
    assert first.primary_agent == second.primary_agent == third.primary_agent == "appointment"
    assert [event_reason(runtime, reply) for reply in (first, second, third)] == ["quote", "followup", "followup"]
    assert any("神经内科" in question for question in first.answer.questions)


def test_followup_dropped_on_topic_change(setup):
    runtime, _, _ = setup
    first = ask(runtime, "帮我挂个号")
    reply = ask(runtime, "不挂了，神经内科一般看什么？", session=first.session)
    assert "appointment" not in roles(reply) and event_reason(runtime, reply) == "abandoned"
    later = ask(runtime, "明天上午", session=first.session)
    assert "appointment" not in roles(later) and event_reason(runtime, later) == "none"


def test_followup_expires(setup):
    runtime, _, clock = setup
    first = ask(runtime, "帮我挂个号")
    clock.advance(601)
    reply = ask(runtime, "神经内科", session=first.session)
    assert event_reason(runtime, reply) == "expired" and reply.pending_actions == []


def test_followup_bound_to_subject_and_session(setup):
    runtime, _, _ = setup
    first = ask(runtime, "帮我挂个号")
    assert event_reason(runtime, ask(runtime, "神经内科", session=first.session, subject="mother")) == "subject_changed"
    ask(runtime, "帮我挂个号", session=first.session)
    assert event_reason(runtime, ask(runtime, "神经内科")) == "none"


def test_urgent_turn_drops_followup(setup):
    runtime, _, _ = setup
    first = ask(runtime, "帮我挂个号")
    assert event_reason(runtime, ask(runtime, "胸口剧痛喘不上气", session=first.session)) == "urgent"
    assert ask(runtime, "神经内科", session=first.session).pending_actions == []


def test_no_slot_then_ok_creates_nothing(setup):
    runtime, _, _ = setup
    first = ask(runtime, "帮我挂 11月8日下午神经内科的号")
    assert first.pending_actions == [] and first.answer.questions
    assert ask(runtime, "好的", session=first.session).pending_actions == []


def test_confirm_in_chat_creates_nothing(setup):
    runtime, _, _ = setup
    first = ask(runtime, REQUEST)
    reply = ask(runtime, "确认", session=first.session)
    assert reply.pending_actions == [] and event_reason(runtime, reply) == "none"


def test_family_subject_label(setup):
    runtime, _, _ = setup
    reply = ask(runtime, "帮我挂明天上午神经内科的号", subject="mother")
    assert reply.pending_actions[0].summary.endswith("就诊人：母亲")


def test_bad_slot_fails_both_instances():
    runtime = run(fixture(BookingModel(bad_slot=True), preview=False, appointments=booking_store(Clock())))
    reply = ask(runtime, REQUEST)
    assert reply.pending_actions == []
    [task] = [assignment for assignment in reply.routing_decision.assignments if assignment.role == "appointment"]
    assert task.status == "failed" and len(task.attempts) == 2


def test_critical_risk_no_booking(setup):
    runtime, _, _ = setup
    reply = ask(runtime, "胸口剧痛喘不上气，帮我挂个号")
    assert reply.primary_agent == "risk_handoff" and reply.pending_actions == []


def test_symptom_and_booking_together(setup):
    runtime, _, _ = setup
    reply = ask(runtime, "头疼一周了，帮我挂明天上午神经内科的号")
    assert reply.primary_agent == "symptom_guide" and "appointment" in reply.supporting_agents and len(reply.pending_actions) == 1


def test_ready_request_must_end_in_a_draft():
    runtime = run(fixture(BookingModel(skip_tool=True), preview=False, appointments=booking_store(Clock())))
    reply = ask(runtime, REQUEST)
    [task] = [assignment for assignment in reply.routing_decision.assignments if assignment.role == "appointment"]
    assert task.status == "failed" and "要不要换一天？" not in reply.answer.questions
    assert any(attempt.error_code == "appointment_draft_missing" for attempt in task.attempts)


def test_bad_slot_reports_its_error_code():
    runtime = run(fixture(BookingModel(bad_slot=True), preview=False, appointments=booking_store(Clock())))
    reply = ask(runtime, REQUEST)
    [task] = [assignment for assignment in reply.routing_decision.assignments if assignment.role == "appointment"]
    assert {attempt.error_code for attempt in task.attempts} == {"appointment_draft_missing"}


def test_tool_round_runs_without_the_output_schema(setup):
    from medagent.core.provider import json_schema
    runtime, model, _ = setup
    ask(runtime, REQUEST)
    # Structured output made the real model answer in JSON without calling the tool, so the tool round has no schema.
    assert model.appointment_stages == ["agent_appointment_action", "agent_appointment"]
    assert json_schema("agent_appointment_action") is None and json_schema("agent_appointment") is not None
