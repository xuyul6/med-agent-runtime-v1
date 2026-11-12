"""Several time slots in one booking request: one card per slot, all or nothing, and no draft beyond what was asked."""
from datetime import date

import pytest

from medagent.appointments.params import parse_items
from medagent.core.contracts import Question
from medagent.tools.registry import ToolInputError
from support.appointments import BOOKING_TURNS, BookingModel, Clock, SingleBookingModel, booking_store, single_runtime, trace_of
from support.harness import fixture, run
from test_appointment_tools import CATALOG, call, context_for, make_store, request
from medagent.tools.registry import ToolExecutor

TODAY = date(2026, 11, 3)
TWO = "帮我挂11月5日上午和11月6日上午神经内科的号"
SHARED = "帮我约11月5日和11月6日下午的呼吸内科"
PARTLY = "帮我挂11月5日上午和11月7日下午神经内科的号"
BOOKING_TURNS.update({
    TWO: ("visit", TWO, {"department":"神经内科", "appointment_date":"11月5日", "time_window":"上午"}),
    SHARED: ("visit", SHARED, {"department":"呼吸内科", "appointment_date":"11月5日", "time_window":"下午"}),
    PARTLY: ("visit", PARTLY, {"department":"神经内科", "appointment_date":"11月5日", "time_window":"上午"}),
})


def test_parse_items_splits_dates_and_shares_a_trailing_window():
    assert parse_items(TWO) == [("11月5日", "上午"), ("11月6日", "上午")]
    assert parse_items(SHARED) == [("11月5日", "下午"), ("11月6日", "下午")]
    assert parse_items("帮我挂11月5日上午和11月6日的号") == [("11月5日", "上午"), ("11月6日", None)]
    assert parse_items("帮我挂明天上午神经内科的号") == []                 # one date: the single-slot path is unchanged
    assert parse_items("挂明天上午的号，就明天上午") == []                  # the same slot twice is one slot


def multi_context(store, quote):
    return context_for(store, request(department="神经内科", appointment_date="11月5日", time_window="上午",
                                      items=[{"appointment_date":d, "time_window":w} for d, w in parse_items(quote)]))


def test_tools_allow_one_draft_per_requested_slot_and_no_more():
    store, executor = make_store(), ToolExecutor()
    context = multi_context(store, TWO)
    slots = call(executor, "search_slots", {}, context).slots
    assert [s.slot_id for s in slots] == ["demo-general:neurology:2026-11-05:am", "demo-general:neurology:2026-11-06:am"]
    for slot in slots:
        call(executor, "propose_appointment", {"slot_id":slot.slot_id}, context)
    assert len(context.issued) == 2
    with pytest.raises(ToolInputError, match="draft_limit_per_turn"):
        call(executor, "propose_appointment", {"slot_id":slots[0].slot_id}, context)


def test_the_same_slot_cannot_be_drafted_twice():
    store, executor = make_store(), ToolExecutor()
    context = multi_context(store, TWO)
    [first, _] = call(executor, "search_slots", {}, context).slots
    call(executor, "propose_appointment", {"slot_id":first.slot_id}, context)
    with pytest.raises(ToolInputError, match="slot_already_drafted"):
        call(executor, "propose_appointment", {"slot_id":first.slot_id}, context)


def test_more_than_three_slots_is_invalid():
    from medagent.appointments.resolution import resolve_request
    quote = "帮我挂11月5日上午、11月6日上午、11月9日上午和11月10日上午神经内科的号"
    req = request(department="神经内科", items=[{"appointment_date":d, "time_window":w} for d, w in parse_items(quote)])
    assert "appointment_date" in resolve_request(req, CATALOG, TODAY).invalid


def ask(runtime, text, session=""):
    return run(runtime.chat(Question(text=text, user="u1", session=session)))


@pytest.fixture(params=["multi_agent", "single_agent"])
def runtime(request):
    if request.param == "multi_agent":
        return run(fixture(BookingModel(), preview=False, appointments=booking_store(Clock())))
    return run(single_runtime(SingleBookingModel(), Clock()))


def test_two_slots_give_two_cards_in_both_arms(runtime):
    reply = ask(runtime, TWO)
    assert [a.kind for a in reply.pending_actions] == ["appointment_create"]*2
    assert "11月5日" in reply.pending_actions[0].summary and "11月6日" in reply.pending_actions[1].summary
    assert len(ask(runtime, SHARED).pending_actions) == 2


def test_one_slot_without_a_number_stops_the_whole_request(runtime):
    reply = ask(runtime, PARTLY, session="p")
    assert reply.pending_actions == [] and "11月7日" in reply.answer.text + "".join(reply.answer.questions)
    events = trace_of(runtime, reply)["events"]
    assert not any(e["event"] == "appointment_tool_attempt" for e in events)
    assert runtime.appointments.peek_followup("u1", "p") is None             # multi-slot requests write no follow-up in v1


def test_slots_from_the_quote_count_as_the_date():
    req = request(department="神经内科", items=[{"appointment_date":d, "time_window":w} for d, w in parse_items(TWO)])
    assert req.information_gaps() == []
