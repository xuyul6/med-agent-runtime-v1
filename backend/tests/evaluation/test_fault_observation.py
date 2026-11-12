"""Synthetic injection must be exercised, not merely configured."""
import pytest

from medagent.evaluation.runner import evaluate, observed_fault_checks
from medagent.evaluation.contracts import EvaluationRequest
from support.evaluation import JudgeModel
from support.failure_contract import RegressionModel
from support.harness import fixture, run


FAULTS = ["memory_unavailable", "index_unavailable", "tool_timeout", "tool_invalid_result",
          "agent_invalid_output", "synthesis_failure", "summary_failure", "judge_failure", "language_unavailable"]


@pytest.mark.parametrize("fault", FAULTS)
def test_urgent_short_circuit_or_disabled_judge_cannot_claim_injection_occurred(fault):
    async def scenario():
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[{
            "id":"fault-not-reached", "turns":[{"message":"我现在呼吸困难。",
                "expected":{"role":"risk_handoff", "risk":"CRITICAL", "references":False}}],
            "faults":[{"kind":fault}]}])
        report = await evaluate(await fixture(JudgeModel()),body)
        turn = report["cases"][0]["turns"][0]
        assert turn["checks"]["role"] and turn["checks"]["risk"]
        assert turn["checks"][f"fault:{fault}_observed"] is False
        assert not turn["passed"] and not report["cases"][0]["passed"]
        assert turn["reply"]["usage"]["calls"] == 0

    run(scenario())


@pytest.mark.parametrize("fault", ["memory_unavailable", "index_unavailable", "tool_timeout", "tool_invalid_result", "agent_invalid_output"])
def test_actual_dependency_fault_has_evidence_and_is_isolated(fault):
    async def scenario():
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[{
            "id":"fault-reached", "turns":[{"message":"查询布洛芬公开药品资料。",
                "expected":{"role":"drug_reference", "degraded":True}}],"faults":[{"kind":fault}]}])
        runtime = await fixture(JudgeModel(),preview=fault != "index_unavailable")
        report = await evaluate(runtime,body)
        turn = report["cases"][0]["turns"][0]
        assert turn["checks"][f"fault:{fault}_observed"] is True
        assert turn["passed"] and not report["cleanup_errors"]
        assert not runtime.monitor.requests and not runtime.tools.failures

    run(scenario())


def test_language_fault_observed_when_understanding_is_called():
    async def scenario():
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[{
            "id":"language-reached", "turns":[{"message":"我想查药，但药名不清楚。",
                "expected":{"role":"drug_reference", "status":"needs_clarification"}}],
            "faults":[{"kind":"language_unavailable"}]}])
        report = await evaluate(await fixture(JudgeModel()),body)
        turn = report["cases"][0]["turns"][0]
        assert turn["checks"]["fault:language_unavailable_observed"] is True and turn["passed"]

    run(scenario())


def test_synthesis_fault_observed_when_both_roles_are_actually_merged():
    async def scenario():
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[{
            "id":"synthesis-reached", "turns":[{"message":"我最近反复头痛，想去看医生，需要整理哪些信息和资料？",
                "expected":{"role":"visit_guide", "supporting_agents":["symptom_guide"], "degraded":True}}],
            "faults":[{"kind":"synthesis_failure"}]}])
        report = await evaluate(await fixture(RegressionModel()),body)
        turn = report["cases"][0]["turns"][0]
        assert turn["checks"]["fault:synthesis_failure_observed"] is True and turn["passed"]

    run(scenario())


def test_summary_fault_observed_only_on_its_configured_turn():
    async def scenario():
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[{
            "id":"summary-reached", "turns":["查询布洛芬公开资料。"]+["这个药呢？"]*4+
                [{"message":"这个药呢？", "expected":{"degraded":True}}],
            "faults":[{"kind":"summary_failure", "turn":6}]}])
        report = await evaluate(await fixture(JudgeModel()),body)
        turns = report["cases"][0]["turns"]
        assert all("fault:summary_failure_observed" not in turn["checks"] for turn in turns[:-1])
        assert turns[-1]["checks"]["fault:summary_failure_observed"] is True and report["cases"][0]["passed"]

    run(scenario())


def test_judge_fault_is_recorded_without_fabricating_a_score():
    async def scenario():
        body = EvaluationRequest(intent_cases=[],judge=True,dialog_cases=[{"id":"judge-reached", "turns":["你好"],
            "faults":[{"kind":"judge_failure"}]}])
        report = await evaluate(await fixture(JudgeModel()),body)
        turn = report["cases"][0]["turns"][0]
        assert turn["checks"]["fault:judge_failure_observed"] is True
        assert turn["judge_error"] == "RuntimeError" and turn["quality"] is None and not turn["passed"]

    run(scenario())


def test_fault_evidence_must_match_the_injected_role_and_outcome():
    kinds = {"tool_timeout", "agent_invalid_output", "language_unavailable"}
    trace = {"events":[
        {"event":"synthetic_fault_triggered", "fault":"tool_timeout", "role":"drug_reference"},
        {"event":"tool", "name":"lookup_reference", "role":"visit_guide", "status":"timeout", "handler_invoked":True},
        {"event":"synthetic_fault_triggered", "fault":"agent_invalid_output", "role":"drug_reference"},
        {"event":"agent_output_validation", "role":"visit_guide", "code":"invalid_output_shape"},
        {"event":"classification_fallback", "channel":"language", "error":"RuntimeError"},
    ]}
    assert not any(observed_fault_checks(kinds,trace).values())


def test_overlapping_tool_faults_do_not_claim_both_handlers_executed():
    async def scenario():
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[{
            "id":"overlap", "turns":[{"message":"查询布洛芬公开药品资料。", "expected":{"degraded":True}}],
            "faults":[{"kind":"tool_timeout"},{"kind":"tool_invalid_result"}]}])
        report = await evaluate(await fixture(JudgeModel()),body)
        turn = report["cases"][0]["turns"][0]
        assert turn["checks"]["fault:tool_timeout_observed"] is True
        assert turn["checks"]["fault:tool_invalid_result_observed"] is False and not turn["passed"]
    run(scenario())


def test_preview_index_fault_requires_an_actual_eligible_search():
    async def scenario():
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[{
            "id":"preview-index", "turns":[{"message":"查询布洛芬公开药品资料。"}],
            "faults":[{"kind":"index_unavailable"}]}])
        report = await evaluate(await fixture(JudgeModel()),body)
        turn = report["cases"][0]["turns"][0]
        assert turn["checks"]["fault:index_unavailable_observed"] is True
        assert any(event["event"] == "retrieval_vector_unavailable" for event in turn["trace"]["events"])
    run(scenario())
