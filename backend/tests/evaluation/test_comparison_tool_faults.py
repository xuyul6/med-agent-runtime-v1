import json
from collections import Counter
from dataclasses import replace
import pytest
from pydantic import ValidationError
from support.comparison import RehearsalModel
from support.harness import fixture, run
from medagent.comparison.cli import build_plan
from medagent.comparison.contracts import CATEGORIES, ComparisonCase, ComparisonRequest
from medagent.comparison.metrics import summarize_comparison
from medagent.comparison.runner import run_comparison
from medagent.core.config import ROOT
from medagent.evaluation.contracts import FaultSpec
from medagent.evaluation.isolation import injected_faults
from medagent.core.contracts import Intent, Question

HOLDOUT = ROOT/"resources"/"comparison-cases-holdout-v1.json"
FAULT = [{"kind":"tool_timeout", "domain":"medicine"}]
COMPOUND = "下周要看医生，帮我列一下要带的东西，再查查成人布洛芬的公开资料。"


def case(**fields):
    body = {"id":"f", "category":"tool_fault", "target":{"message":COMPOUND}, "reference_domains":["visit"], "faults":FAULT}
    return ComparisonCase.model_validate({**body, **fields})


def online(base, model):
    from medagent.runtime import MedicalRuntime
    return MedicalRuntime(replace(base.settings, preview=False), model, base.vectors, base.index, base.memory)


def test_fault_category_and_fault_come_together():
    assert case().faults[0].domain == Intent.MEDICINE
    with pytest.raises(ValidationError, match="tool_fault_category"):
        case(faults=[])
    with pytest.raises(ValidationError, match="tool_fault_category"):
        case(category="compound")


@pytest.mark.parametrize("fault", [
    {"kind":"tool_timeout"}, {"kind":"agent_invalid_output", "domain":"medicine"}, {"kind":"tool_timeout", "domain":"medicine", "turn":1}])
def test_only_domain_scoped_tool_faults_are_allowed(fault):
    with pytest.raises(ValidationError, match="domain_scoped"):
        case(faults=[fault])


def test_a_failing_domain_cannot_be_required_coverage():
    with pytest.raises(ValidationError, match="faulted_domain"):
        case(reference_domains=["medicine"])


def test_scoped_fault_leaves_other_domains_working():
    model = RehearsalModel()
    async def scenario():
        runtime = await fixture(model)
        async with injected_faults(runtime, [FaultSpec.model_validate(FAULT[0])], 1):
            reply = await runtime.chat(Question(text=COMPOUND))
        statuses = {row.role:row.status for row in reply.agent_results}
        trace = next(row for row in reversed(runtime.monitor.traces) if row["request_id"] == reply.request_id)
        lookups = {(item["role"], item["status"]) for item in trace["events"] if item.get("event") == "tool" and item.get("name") == "lookup_reference"}
        assert ("drug_reference", "timeout") in lookups
        assert any(role == "visit_guide" and status in {"ok", "empty_result"} for role, status in lookups)
        assert "visit_guide" in statuses and statuses["visit_guide"] != "failed"
    run(scenario())


def test_fault_is_gone_after_the_target_turn():
    async def scenario():
        runtime = await fixture(RehearsalModel())
        registry = runtime.tools.registry
        async with injected_faults(runtime, [FaultSpec.model_validate(FAULT[0])], 1):
            assert runtime.tools.registry is not registry
        assert runtime.tools.registry is registry
    run(scenario())


def test_every_arm_gets_the_same_fault_and_it_is_recorded():
    async def scenario():
        base = await fixture(RehearsalModel())
        target = case(target={"message":"想知道看报告时项目和结果怎么对应，也想知道见医生那天该带什么。"},
                      faults=[{"kind":"tool_invalid_result", "domain":"report"}])
        return await run_comparison(online(base, base.model), ComparisonRequest(cases=[target]))
    report = run(scenario())
    arms = [row["arms"] for row in report["pairs"]]
    assert all(arm["multi_agent"]["fault_triggered"] and arm["single_agent"]["fault_triggered"] for arm in arms)
    assert report["statistics"]["by_category"]["tool_fault"]["multi_agent"]["coverage_rate"] == 1
    assert "fault_not_triggered" not in report["claim_blockers"]


def test_a_fault_that_never_fired_blocks_claims():
    arm = lambda **fields: {"status":"completed", "quality":{"mean":.8, "relevance":.8, "accuracy":.8, "completeness":.8, "helpfulness":.8},
                            "usage":{"total_input_tokens":1, "output_tokens":1, "calls":1}, "elapsed_ms":1, **fields}
    report = {"mode":"online", "complete":True, "pairs":[{"id":"a", "case_id":"a", "category":"tool_fault", "arms":{
        "multi_agent":arm(fault_triggered=False), "primary_only":arm(fault_triggered=False), "single_agent":arm(fault_triggered=True)}}]}
    summarize_comparison(report, ComparisonRequest())
    assert "fault_not_triggered" in report["claim_blockers"]


def test_primary_only_may_skip_the_faulted_supporting_domain():
    arm = lambda **fields: {"status":"completed", "quality":{"mean":.8, "relevance":.8, "accuracy":.8, "completeness":.8, "helpfulness":.8},
                            "usage":{"total_input_tokens":1, "output_tokens":1, "calls":1}, "elapsed_ms":1, **fields}
    report = {"mode":"online", "complete":True, "pairs":[{"id":"a", "case_id":"a", "category":"tool_fault", "arms":{
        "multi_agent":arm(fault_triggered=True), "primary_only":arm(fault_triggered=False), "single_agent":arm(fault_triggered=True)}}]}
    summarize_comparison(report, ComparisonRequest())
    assert "fault_not_triggered" not in report["claim_blockers"]


def test_holdout_covers_every_category_with_three_cases():
    body = json.loads(HOLDOUT.read_text())
    request = ComparisonRequest(cases=body["cases"], limits={"timeout_seconds":2700})
    assert set(Counter(item.category for item in request.cases)) == set(CATEGORIES[:6])
    assert min(Counter(item.category for item in request.cases).values()) >= 3
    assert build_plan(request)["planned_targets"] == 216


def test_circuit_of_one_domain_leaves_other_domains_open_and_snapshot_shows_it():
    async def scenario():
        runtime = await fixture(RehearsalModel())
        executor = runtime.tools
        async with injected_faults(runtime, [FaultSpec.model_validate(FAULT[0])], 1):
            reply = await runtime.chat(Question(text=COMPOUND))
            assert executor.circuit_snapshot("lookup_reference")["circuit_state"] == "open"
            assert executor.open_until.get("lookup_reference:medicine") and not executor.open_until.get("lookup_reference:visit")
        assert [row.status for row in reply.routing_decision.assignments if row.role == "drug_reference"] == ["failed"]
    run(scenario())
