import pytest

from medagent.core.contracts import Question
from support.appointments import Clock, SingleBookingModel, event_reason, single_runtime, trace_of
from support.harness import fixture, run
from support.appointments import BookingModel, booking_store

REQUEST = "帮我挂明天上午神经内科的号"


def setup(**options):
    model = SingleBookingModel(**options)
    return run(single_runtime(model, Clock())), model


def ask(runtime, text, session=""):
    return run(runtime.chat(Question(text=text, user="u1", session=session)))


def test_single_gets_tool_only_with_ready_request():
    runtime, model = setup()
    ask(runtime, "ALT 偏高什么意思")
    first = model.single_calls[-1]
    assert first["tools"] == ["lookup_reference"] and "角色规则：appointment" not in first["system"] and "appointment" not in first["data"]
    ask(runtime, REQUEST)
    ready = model.single_calls[-2]
    assert "propose_appointment" in ready["tools"] and "角色规则：appointment" in ready["system"]
    assert ready["data"]["appointment"]["kind"] == "ready" and ready["data"]["current_request"] == REQUEST
    ask(runtime, "帮我挂明天的号")
    missing = model.single_calls[-1]
    assert "propose_appointment" not in missing["tools"] and missing["data"]["appointment"]["kind"] == "missing"


def test_single_shares_tool_checks():
    runtime, _ = setup(wrong_slot=True)
    reply = ask(runtime, REQUEST)
    assert reply.pending_actions == []
    tools = [event for event in trace_of(runtime, reply)["events"] if event["event"] == "tool" and event["name"] == "propose_appointment"]
    assert tools and tools[0]["status"] == "invalid_request"


def test_single_followup_chain():
    runtime, _ = setup()
    first = ask(runtime, "帮我挂个号")
    second = ask(runtime, "神经内科", session=first.session)
    third = ask(runtime, "明天上午", session=first.session)
    assert first.pending_actions == second.pending_actions == [] and len(third.pending_actions) == 1
    assert [event_reason(runtime, reply) for reply in (first, second, third)] == ["quote", "followup", "followup"]


def test_single_returns_same_summary_as_multi():
    runtime, _ = setup()
    single = ask(runtime, REQUEST).pending_actions
    multi_runtime = run(fixture(BookingModel(), preview=False, appointments=booking_store(Clock())))
    multi = run(multi_runtime.chat(Question(text=REQUEST, user="u1"))).pending_actions
    assert len(single) == len(multi) == 1 and single[0].summary == multi[0].summary and single[0].kind == multi[0].kind


# ---- Parity with the booking role: tool round without the output schema, and a missing draft is a failure ----

class Staged(SingleBookingModel):
    """Records every single-agent stage; with claim_only it answers "draft created" without calling the tool."""
    def __init__(self, claim_only=False):
        super().__init__()
        self.claim_only, self.stages = claim_only, []

    async def message(self, stage, system, messages, tools=None, budget=1800):
        if stage.startswith("agent_single_agent"):
            self.stages.append(stage)
            if self.claim_only:
                import json
                text = {"text":"已生成待确认的预约草稿，请在下方点击确认。", "questions":[], "next_steps":[], "limitations":[], "reference_keys":[]}
                return {"content":[{"type":"text", "text":json.dumps(text, ensure_ascii=False)}]}
        return await super().message(stage, system, messages, tools, budget)


def test_single_tool_round_runs_without_the_output_schema():
    from medagent.core.provider import json_schema
    assert json_schema("agent_single_agent_action") is None and json_schema("agent_single_agent") is not None
    model = Staged()
    runtime = run(single_runtime(model, Clock()))
    reply = ask(runtime, REQUEST)
    assert len(reply.pending_actions) == 1
    assert model.stages == ["agent_single_agent_action", "agent_single_agent"]
    ask(runtime, "ALT 偏高什么意思")
    assert model.stages[2:] == ["agent_single_agent"]          # no booking tool offered: one schema-bound round as before


def test_single_claiming_a_draft_it_never_made_fails():
    model = Staged(claim_only=True)
    runtime = run(single_runtime(model, Clock()))
    reply = ask(runtime, REQUEST)
    assert reply.pending_actions == [] and "已生成待确认的预约草稿" not in reply.answer.text
    errors = [e for e in trace_of(runtime, reply)["events"] if e["event"] == "agent_output_validation"]
    assert errors and errors[0]["code"] == "appointment_draft_missing"


def test_single_prompt_calls_tools_before_the_final_json():
    runtime, model = setup()
    ask(runtime, REQUEST)
    system = model.single_calls[0]["system"]
    assert "需要调用工具时先调用，拿到工具结果后再返回最终 JSON" in system and "角色规则：appointment" in system
