"""The open single-agent arm: every booking tool on every turn, its own searches, at most three drafts, writes only by confirmation."""
import json

from medagent.comparison.open_tools import OPEN_SPECS
from medagent.core.contracts import Question
from medagent.tools.registry import DEFAULT_TOOLS, ToolRegistry
from support.appointments import Clock, SingleBookingModel, single_runtime, trace_of
from support.harness import role_data, run

OPEN = ["open_search_slots", "open_find_bookings", "open_propose_appointment", "open_propose_cancellation"]


def test_open_tools_reach_only_the_runtime_of_the_open_arm():
    assert [spec.name for spec in OPEN_SPECS] == OPEN and not set(OPEN) & set(DEFAULT_TOOLS.specs)
    strict = run(single_runtime(SingleBookingModel(), Clock()))
    opened = run(single_runtime(SingleBookingModel(), Clock(), open_tools=True))
    assert not set(OPEN) & set(strict.tools.registry.specs) and not set(OPEN) & set(strict.monitor.tool_series)
    assert set(OPEN) <= set(opened.tools.registry.specs) and set(OPEN) <= set(opened.monitor.tool_series)
    assert opened.planner.tools is opened.tools.registry and not set(OPEN) & set(DEFAULT_TOOLS.specs)


class Scripted(SingleBookingModel):
    """Plays a list of tool rounds on the single-agent stages, then answers."""
    def __init__(self, rounds=()):
        super().__init__()
        self.rounds, self.seen = list(rounds), []

    async def message(self, stage, system, messages, tools=None, budget=1800):
        if not stage.startswith("agent_single_agent"):
            return await super().message(stage, system, messages, tools, budget)
        self.seen.append({"stage":stage, "tools":[tool["name"] for tool in tools or []], "messages":messages})
        if self.rounds:
            calls = self.rounds.pop(0)
            if callable(calls): calls = calls(messages)
            return {"content":[{"type":"tool_use", "id":f"c{i}", "name":name, "input":args} for i, (name, args) in enumerate(calls)]}
        text = {"text":"已处理。", "questions":[], "next_steps":[], "limitations":[], "reference_keys":[]}
        return {"content":[{"type":"text", "text":json.dumps(text, ensure_ascii=False)}]}


def ask(runtime, text):
    return run(runtime.chat(Question(text=text, user="u1", session="s")))


def results(messages):
    return [json.loads(block["content"]) for m in messages if isinstance(m["content"], list)
            for block in m["content"] if block.get("type") == "tool_result" and not block.get("is_error")]


def test_open_arm_has_every_booking_tool_without_a_booking_request():
    model = Scripted()
    runtime = run(single_runtime(model, Clock(), open_tools=True))
    ask(runtime, "第一次去医院看病要准备什么？")
    assert set(OPEN) <= set(model.seen[0]["tools"]) and model.seen[0]["stage"] == "agent_single_agent_action"
    strict = Scripted()
    ask(run(single_runtime(strict, Clock())), "第一次去医院看病要准备什么？")
    assert not set(OPEN) & set(strict.seen[0]["tools"])                     # the strict baseline is unchanged


def test_open_arm_can_search_and_draft_an_unrequested_slot_but_not_write():
    first = [("open_search_slots", {"department":"皮肤科", "appointment_date":"11月6日", "time_window":"上午", "institution":"演示总医院"})]
    second = lambda messages: [("open_propose_appointment", {"slot_id":results(messages)[-1]["slots"][0]["slot_id"]})]
    model = Scripted([first, second])
    runtime = run(single_runtime(model, Clock(), open_tools=True))
    reply = ask(runtime, "第一次去医院看病要准备什么？")
    [card] = reply.pending_actions
    assert "皮肤科" in card.summary
    attempts = [e for e in trace_of(runtime, reply)["events"] if e["event"] == "appointment_tool_attempt"]
    assert [(e["tool"], e["action"], e["target"]) for e in attempts] == [
        ("open_propose_appointment", "create", "demo-general:derm:2026-11-06:am")]
    assert runtime.appointments.booking_rows() == []                       # a draft is not a booking


def test_open_arm_stops_at_three_drafts_and_only_its_own_search():
    import pytest
    from medagent.agents.contracts import AppointmentInput
    from medagent.core.contracts import Intent
    from medagent.tools.registry import ToolContext, ToolExecutor, ToolInputError
    from dataclasses import replace
    from medagent.agents.profiles import DEFAULT_REGISTRY
    from test_appointment_tools import call, make_store
    store, executor = make_store(), ToolExecutor(ToolRegistry(OPEN_SPECS))
    profile = replace(DEFAULT_REGISTRY["appointment"], role="single_agent_open", tool_scope=tuple(OPEN))
    context = ToolContext(profile, AppointmentInput(text="你好", intent=Intent.APPOINTMENT, action="create"), user="u1", session="s",
                          appointments=store)
    with pytest.raises(ToolInputError, match="slot_not_from_this_search"):
        call(executor, "open_propose_appointment", {"slot_id":"demo-general:neurology:2026-11-05:am"}, context)
    slots = [call(executor, "open_search_slots", {"department":"神经内科", "appointment_date":day, "time_window":"上午"}, context).slots[0]
             for day in ["11月5日", "11月6日", "11月9日", "11月10日"]]
    for slot in slots[:3]:
        call(executor, "open_propose_appointment", {"slot_id":slot.slot_id}, context)
    with pytest.raises(ToolInputError, match="draft_limit_per_turn"):
        call(executor, "open_propose_appointment", {"slot_id":slots[3].slot_id}, context)
    assert store.booking_rows() == []


def test_open_arm_books_the_request_like_the_strict_arm():
    runtime = run(single_runtime(SingleBookingModel(), Clock(), open_tools=True))
    [card] = ask(runtime, "帮我挂明天上午神经内科的号").pending_actions
    assert "神经内科" in card.summary
