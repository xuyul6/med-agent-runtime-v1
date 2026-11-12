"""The single agent retries a failed turn once, matching the second instance a role gets."""
import asyncio
from dataclasses import replace

import pytest

from support.comparison import RehearsalModel
from support.harness import fixture, run
from medagent.comparison.contracts import ARM_SETTINGS
from medagent.core.contracts import Question
from medagent.memory import scope_key
from medagent.runtime import MedicalRuntime


class Flaky(RehearsalModel):
    def __init__(self, failures, error=ConnectionError):
        super().__init__()
        self.failures, self.error = failures, error

    async def message(self, stage, *args, **kwargs):
        if stage == "agent_single_agent" and self.failures:
            self.failures -= 1
            raise self.error()
        return await super().message(stage, *args, **kwargs)


def single_runtime(attempts, fail_first, **kwargs):
    async def build():
        base = await fixture(Flaky(fail_first, **kwargs))
        settings = replace(base.settings, architecture="single_agent", single_agent_attempts=attempts)
        return MedicalRuntime(settings, base.model, base.vectors, base.index, base.memory)
    return run(build())


def ask(runtime):
    return run(runtime.chat(Question(text="想查成人布洛芬的公开资料", user="u", session="s")))


def attempts_of(reply):
    return [item.status for item in reply.routing_decision.assignments[0].attempts]


def test_one_failed_attempt_is_retried_and_recorded():
    reply = ask(single_runtime(2, 1))
    assert attempts_of(reply) == ["failed", "ok"]
    assert reply.routing_decision.collaboration_status == "single"
    assert reply.routing_decision.assignments[0].status == "answered"


def test_two_failed_attempts_end_in_transparent_guidance():
    reply = ask(single_runtime(2, 2))
    assert attempts_of(reply) == ["failed", "failed"]
    assert reply.routing_decision.collaboration_status == "failed"
    assert reply.routing_decision.assignments[0].status == "failed"
    assert len(reply.routing_decision.fallbacks) == 1


def test_default_keeps_a_single_attempt():
    assert attempts_of(ask(single_runtime(1, 1))) == ["failed"]


def test_cancellation_is_not_retried():
    runtime = single_runtime(2, 1, error=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        ask(runtime)
    assert runtime.model.failures == 0 and runtime.model.stages.count("agent_single_agent") == 0


def test_retry_writes_memory_once():
    runtime = single_runtime(2, 1)
    reply = ask(runtime)
    state = run(runtime.memory.read(scope_key("u", reply.session, reply.subject)))
    assert sum(item["role"] == "user" for item in state["messages"]) == 1


def test_retry_shares_the_tool_budget():
    runtime = single_runtime(2, 1)
    ask(runtime)
    trace = runtime.monitor.traces[-1]
    assert sum(item.get("event") == "tool" for item in trace["events"]) <= runtime.settings.tool_call_limit


def test_attempts_outside_one_or_two_are_rejected():
    with pytest.raises(ValueError, match="invalid_single_agent_attempts"):
        single_runtime(3, 0)


def test_the_comparison_arm_retries_once():
    assert ARM_SETTINGS["single_agent"]["single_agent_attempts"] == 2
