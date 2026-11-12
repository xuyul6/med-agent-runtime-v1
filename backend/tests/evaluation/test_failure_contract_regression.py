"""Independent strengthened regression cases; original evaluation set stays intact."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from medagent.core.contracts import Intent, Reference
from medagent.benchmark.contracts import PROTOCOL
from medagent.evaluation.runner import behavior_checks, evaluate, resolve_cases, snapshot_preflight
from medagent.evaluation.contracts import EvaluationRequest
from medagent.evaluation.isolation import ReferenceSnapshotError
from medagent.evaluation.metrics import BaselineError, METRIC_VERSION, compare_baseline
from medagent.agents.checks import VISIT_URGENCY_ATTRIBUTION_NOTICE
from support.harness import fixture, run
from support.failure_contract import RegressionModel, WARNING


RESOURCE = Path(__file__).parents[2]/"resources"/"failure-contract-v1.json"


def load_cases():
    return EvaluationRequest.model_validate_json(RESOURCE.read_text())


def test_new_contract_cases_keep_original_questions_and_strengthen_expectations():
    body = load_cases()
    _,intents,original = resolve_cases(EvaluationRequest())
    assert len(intents)+len(original) == 39
    originals = {c.id:c for c in original}
    mapping = {"report-context-structured-v1":"report-context", "report-context-no-source-v1":"report-context",
               "symptom-visit-support-v1":"symptom-and-visit", "drug-lookup-timeout-v1":"lookup-timeout"}
    assert len(body.dialog_cases) == 4 and not body.intent_cases
    for case in body.dialog_cases:
        assert [t.message for t in case.turns] == [t.message for t in originals[mapping[case.id]].turns]
        assert case.reference_snapshot is not None
    assert body.dialog_cases[0].turns[0].expected.status == "needs_clarification"
    assert body.dialog_cases[2].turns[0].expected.supporting_agents == ["symptom_guide"]
    assert [fault.kind for fault in body.dialog_cases[3].faults] == ["tool_timeout"]


def test_old_answered_without_fields_cannot_pass_new_report_contract():
    expected = load_cases().dialog_cases[0].turns[0].expected
    reply = SimpleNamespace(primary_agent="report_explainer",intent=SimpleNamespace(primary=Intent.REPORT,entities={}),
        risk=SimpleNamespace(level="ROUTINE"),subject="self",references=[object()],degraded=False,supporting_agents=[],
        agent_results=[SimpleNamespace(role="report_explainer",status="answered",missing_fields=[])])
    checks = behavior_checks(reply,expected)
    assert checks["role"] and checks["references"] and not checks["status"] and not checks["missing_fields"]


def test_missing_support_cannot_pass_new_collaboration_contract():
    expected = load_cases().dialog_cases[2].turns[0].expected
    reply = SimpleNamespace(primary_agent="visit_guide",intent=SimpleNamespace(primary=Intent.VISIT,entities={}),
        risk=SimpleNamespace(level="ROUTINE"),subject="self",references=[object()],degraded=False,supporting_agents=[],
        agent_results=[SimpleNamespace(role="visit_guide",status="answered",missing_fields=[])])
    checks = behavior_checks(reply,expected)
    assert checks["role"] and not checks["support"]


def test_selected_support_without_an_execution_result_is_not_treated_as_success():
    expected = load_cases().dialog_cases[2].turns[0].expected
    reply = SimpleNamespace(primary_agent="visit_guide",intent=SimpleNamespace(primary=Intent.VISIT,entities={}),
        risk=SimpleNamespace(level="ROUTINE"),subject="self",references=[object()],degraded=False,
        supporting_agents=["symptom_guide"],
        agent_results=[SimpleNamespace(role="visit_guide",status="answered",missing_fields=[])])
    assert not behavior_checks(reply,expected)["support"]
    reply.agent_results.append(SimpleNamespace(role="symptom_guide",status="answered",missing_fields=[]))
    assert behavior_checks(reply,expected)["support"]


def test_support_check_version_is_shared_and_old_metric_baseline_is_rejected():
    assert METRIC_VERSION == "eval-metrics-5" and PROTOCOL["behavior_check_version"] == METRIC_VERSION
    assert PROTOCOL["version"] == "paired-context-4"
    current = {"metadata":{"dataset_hash":"same", "metric_version":METRIC_VERSION, "policy_hash":"same"}}
    for version in ("eval-metrics-2", "eval-metrics-3"):
        old = {"metadata":{"dataset_hash":"same", "metric_version":version, "policy_hash":"same"}}
        with pytest.raises(BaselineError,match="baseline_dataset_or_policy_incompatible"):
            compare_baseline(current,old)


def test_injected_timeout_cannot_pass_when_risk_gate_skips_tool_execution():
    async def scenario():
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[{
            "id":"timeout-not-reached", "turns":[{"message":"我现在呼吸困难。",
                "expected":{"role":"risk_handoff", "risk":"CRITICAL", "references":False}}],
            "faults":[{"kind":"tool_timeout"}]}])
        report = await evaluate(await fixture(RegressionModel()),body)
        turn = report["cases"][0]["turns"][0]
        assert turn["checks"]["role"] and turn["checks"]["risk"]
        assert turn["checks"]["fault:tool_timeout_observed"] is False
        assert not turn["passed"] and not report["cases"][0]["passed"]
        assert turn["reply"]["usage"]["calls"] == 0

    run(scenario())


def test_timeout_observation_is_required_only_on_the_injected_turn():
    async def scenario():
        body = EvaluationRequest(intent_cases=[],judge=False,dialog_cases=[{
            "id":"timeout-second-turn", "turns":[
                {"message":"查询布洛芬公开药品资料。", "expected":{"role":"drug_reference"}},
                {"message":"这个药的资料呢？", "expected":{"role":"drug_reference", "degraded":True, "references":False}}],
            "faults":[{"kind":"tool_timeout", "turn":2}]}])
        report = await evaluate(await fixture(RegressionModel()),body)
        first,second = report["cases"][0]["turns"]
        assert "fault:tool_timeout_observed" not in first["checks"] and first["passed"]
        assert second["checks"]["fault:tool_timeout_observed"] and second["passed"]

    run(scenario())


def test_reference_change_is_rejected_before_running_contract_set():
    body = load_cases()
    records = [Reference.model_validate(x) for x in json.loads((RESOURCE.parent/"references.json").read_text())]
    snapshots = snapshot_preflight(SimpleNamespace(records=records),body.dialog_cases)
    assert len(snapshots) == 4
    changed = [r.model_copy(update={"text":r.text+" changed"}) if r.key == "lab-context" else r for r in records]
    with pytest.raises(ReferenceSnapshotError,match="snapshot_hash_mismatch"):
        snapshot_preflight(SimpleNamespace(records=changed),body.dialog_cases)


def test_strengthened_cases_execute_through_isolated_runtime_without_paid_model():
    async def scenario():
        body = load_cases().model_copy(update={"judge":False})
        report = await evaluate(await fixture(RegressionModel()),body)
        assert report["mode"] == "offline" and report["complete"] and report["total"] == 4
        assert report["passed"] == 4 and not report["cleanup_errors"] and not report["safety_failures"]
        rows = {row["id"]:row["turns"][0] for row in report["cases"]}
        report_row = rows["report-context-structured-v1"]
        result = report_row["reply"]["agent_results"][0]
        assert result["status"] == "needs_clarification" and set(result["missing_fields"]) >= {"value","unit","reference_range"}
        reference = report_row["reply"]["references"][0]
        assert result["payload"]["term_explanations"][0]["text"] == reference["text"]
        gap = rows["report-context-no-source-v1"]["reply"]
        assert not gap["references"] and "source" in gap["agent_results"][0]["missing_fields"]
        visit = rows["symptom-visit-support-v1"]["reply"]
        assert visit["primary_agent"] == "visit_guide" and "symptom_guide" in visit["supporting_agents"]
        assert WARNING in visit["answer"]["text"] and VISIT_URGENCY_ATTRIBUTION_NOTICE in "\n".join(visit["answer"]["limitations"])
        fault = rows["drug-lookup-timeout-v1"]["reply"]
        assert fault["degraded"] and not fault["references"] and "超时" in fault["answer"]["text"]
        assert rows["drug-lookup-timeout-v1"]["checks"]["fault:tool_timeout_observed"]

    run(scenario())
