"""Which registered tools a role model is offered, and what happens when it calls one it was not offered.

The code runs field checks, the reference lookup and slot queries itself. A model is offered only the tools that
need its choice: the medication scope check and the appointment drafts. Everything else stays code-invoked, and a
model call to it is refused before the handler runs, even when the role's whitelist contains it.
"""
from dataclasses import replace
import json

import pytest

from medagent.core.contracts import Intent
from medagent.core.observability import CURRENT, RunLog
from medagent.tools.registry import DEFAULT_TOOLS, ToolRegistry
from support.harness import Model, fixture, role_output, run

MODEL_TOOLS = {"check_medication_scope", "propose_appointment", "propose_cancellation"}


def test_each_tool_says_whether_code_or_the_model_calls_it():
    described = {name:spec.describe()["invoked_by"] for name, spec in DEFAULT_TOOLS.specs.items()}
    assert {name for name, callers in described.items() if "model" in callers} == MODEL_TOOLS
    assert described["lookup_reference"] == ["code"]
    assert described["check_medication_scope"] == ["code", "model"]
    assert described["propose_appointment"] == ["model"]
    assert all(callers for callers in described.values())


def test_an_action_tool_nobody_can_call_is_rejected_at_registration():
    spec = DEFAULT_TOOLS.specs["propose_appointment"]
    with pytest.raises(ValueError, match="invalid_tool_execution_policy"):
        ToolRegistry([replace(spec, offered_to_model=False)])


class Retrieve(Model):
    """Asks for a second lookup, then answers citing what the code already found."""
    def __init__(self):
        self.turn, self.offered, self.results = 0, set(), []

    async def message(self, stage, system, messages, tools=None, budget=1800):
        if stage != "agent_symptom_guide":
            return await super().message(stage, system, messages, tools, budget)
        self.turn += 1
        if self.turn == 1:
            self.offered = {tool["name"] for tool in tools or []}
            return {"content":[{"type":"tool_use", "id":"again", "name":"lookup_reference",
                                "input":{"query":"发烧 咳嗽", "domain":"symptoms"}}]}
        self.results.append(messages[-1]["content"][0])
        return {"content":[{"type":"text", "text":json.dumps(role_output("symptom_guide", ["symptom-cough-nhs"]), ensure_ascii=False)}]}


def test_a_model_lookup_is_refused_and_the_code_lookup_still_stands():
    async def scenario():
        model, log = Retrieve(), RunLog()
        runtime = await fixture(model)
        token = CURRENT.set(log)
        try:
            result = await runtime.agents["symptom_guide"][0].run("这两天咳嗽，没有发烧，需要注意什么？", Intent.SYMPTOMS, {}, "selective")
        finally:
            CURRENT.reset(token)
        assert "lookup_reference" not in model.offered
        assert model.results[0]["is_error"] is True
        lookups = [e for e in log.events if e["event"] == "tool" and e["name"] == "lookup_reference"]
        # The code's lookup ran; the model's was refused before its handler, so the fever overview never arrived.
        assert [(e["status"], e["handler_invoked"]) for e in lookups] == [("ok", True), ("denied", False)]
        assert result.answer.reference_keys == ["symptom-cough-nhs"]
    run(scenario())


def test_a_source_request_with_no_source_found_is_answered_without_inventing_one():
    async def scenario():
        class Plain(Model):
            async def message(self, stage, system, messages, tools=None, budget=1800):
                if stage == "agent_symptom_guide":
                    self.system = system
                    return {"content":[{"type":"text", "text":json.dumps(role_output("symptom_guide"), ensure_ascii=False)}]}
                return await super().message(stage, system, messages, tools, budget)
        model = Plain()
        runtime = await fixture(model)
        async def nothing(*args, **kwargs):
            return []
        runtime.index.search = nothing
        result = await runtime.agents["symptom_guide"][0].run("最近总是失眠，请查阅资料并给出来源", Intent.SYMPTOMS, {}, "selective")
        assert result.execution.status == "answered" and result.answer.reference_keys == []
        assert "没有找到" in model.system and "调用lookup_reference" not in model.system
    run(scenario())
