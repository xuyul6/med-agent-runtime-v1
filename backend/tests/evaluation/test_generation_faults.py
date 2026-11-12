"""A generation fault hits one domain: multi loses that part at most, the single agent's only generation carries every domain."""
from dataclasses import replace

import pytest

from support.comparison import RehearsalModel
from support.harness import fixture, run
from medagent.comparison.contracts import ARM_SETTINGS, ComparisonCase, ComparisonRequest
from medagent.comparison.metrics import summarize_comparison
from medagent.comparison.runner import attach_baseline
from medagent.core.contracts import Intent, Question
from medagent.evaluation.contracts import FaultSpec
from medagent.evaluation.isolation import injected_faults
from medagent.runtime import MedicalRuntime

COMPOUND = "下周要看医生，帮我列一下要带的东西，再查查成人布洛芬的公开资料。"


def faulted_chat(kind, domain, arm="multi_agent"):
    async def scenario():
        base = await fixture(RehearsalModel())
        runtime = MedicalRuntime(replace(base.settings, **ARM_SETTINGS[arm]), base.model, base.vectors, base.index, base.memory)
        async with injected_faults(runtime, [FaultSpec(kind=kind, domain=domain)], 1, baseline=attach_baseline(runtime, arm)):
            reply = await runtime.chat(Question(text=COMPOUND))
        return reply, next(row for row in reversed(runtime.monitor.traces) if row["request_id"] == reply.request_id)
    return run(scenario())


def assignment(reply, role):
    return next(item for item in reply.routing_decision.assignments if item.role == role)


def fired(trace):
    return [item for item in trace["events"] if item.get("event") == "synthetic_fault_triggered"]


@pytest.mark.parametrize("domain,role", [(Intent.VISIT, "visit_guide"), (Intent.MEDICINE, "drug_reference")])
def test_once_fails_one_attempt_and_the_second_instance_recovers(domain, role):
    reply, trace = faulted_chat("generation_failure_once", domain)
    assert [item.status for item in assignment(reply, role).attempts] == ["failed", "ok"]
    assert len(fired(trace)) == 1 and fired(trace)[0]["fault"] == "generation_failure_once"
    assert role in [item.role for item in reply.agent_results]


def test_once_leaves_the_other_role_alone():
    reply, _ = faulted_chat("generation_failure_once", Intent.VISIT)
    assert [item.status for item in assignment(reply, "drug_reference").attempts] == ["ok"]


def test_always_loses_only_the_faulted_part():
    reply, trace = faulted_chat("generation_failure_always", Intent.MEDICINE)
    assert "drug_reference" not in [item.role for item in reply.agent_results]
    assert "visit_guide" in [item.role for item in reply.agent_results]
    assert reply.routing_decision.collaboration_status == "partial"
    assert len(fired(trace)) == 2


def test_single_agent_generation_is_hit_by_any_domain():
    for domain in [Intent.VISIT, Intent.MEDICINE]:
        reply, trace = faulted_chat("generation_failure_always", domain, "single_agent")
        assert reply.routing_decision.collaboration_status == "failed" and fired(trace)


def test_outage_fails_every_arm_alike():
    for arm in ["multi_agent", "single_agent"]:
        reply, _ = faulted_chat("generation_failure_always", None, arm)
        assert reply.routing_decision.collaboration_status == "failed"
        assert not reply.agent_results


def stability_row(triggered):
    quality = {"mean":.8, "relevance":.8, "accuracy":.8, "completeness":.8, "helpfulness":.8}
    arms = {arm:{"arm":arm, "status":"completed", "quality":quality, "fault_triggered":triggered, "elapsed_ms":1,
                 "usage":{"total_input_tokens":1, "output_tokens":1, "calls":1}} for arm in ["multi_agent", "single_agent"]}
    return {"id":"s-1", "case_id":"s", "category":"stability_always", "repeat":1, "arms":arms}


def blockers(triggered):
    report = {"mode":"online", "complete":True, "pairs":[stability_row(triggered)]}
    request = ComparisonRequest(layer="stability", repetitions=4, cases=[ComparisonCase(
        id="s", category="compound", target={"message":"问题"})])
    summarize_comparison(report, request)
    return report["claim_blockers"]


def test_a_fault_on_a_role_never_routed_blocks_claims():
    assert "fault_not_triggered" in blockers(False)
    assert "fault_not_triggered" not in blockers(True)
