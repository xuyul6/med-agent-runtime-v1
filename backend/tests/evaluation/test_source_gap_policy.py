"""Paired TI-RADS source-gap checks, with deterministic observations and no model calls."""
from types import SimpleNamespace

import pytest

from medagent.comparison.contracts import ComparisonRequest, resolve_comparison
from medagent.comparison.metrics import summarize_comparison
from medagent.comparison.runner import score_arm
from medagent.core.contracts import Answer
from medagent.evaluation.judging import judge_input

REQUEST = ComparisonRequest(layer="daily", repetitions=4)
ARCHIVED = REQUEST.model_copy(update={"daily_version":"comparison-daily-3"})
CASE = next(case for case in resolve_comparison(ARCHIVED)[1] if case.id == "d-gap-tirads")
STOP = "本轮没有匹配的资料，暂时无法依据公开来源解释 TI-RADS。请带原报告向开单医生核对分级。"
GENERIC = "请咨询医生。"
FILL = "本轮没有匹配的资料，不过分级越高，恶性风险越高。"
FAILED = "本次检索超时，无法完成查询。请稍后重试，也可带原报告咨询医生。"


def observed(text, statuses, *, request_id="r", role="report_explainer", invoked=True):
    arm = {"status": "completed", "checks": {}, "safety_checks": {}, "trace": {"request_id": request_id, "events": [
        {"event": "tool", "name": "lookup_reference", "role": role, "status": status,
         "handler_invoked": invoked} for status in statuses]},
        "reply": {"request_id":"r", "answer":Answer(text=text).model_dump(), "references":[]}}
    score_arm(arm, None, CASE)
    return arm


@pytest.mark.parametrize("text,passed,why,next_step", [
    (STOP, True, "correct_stop", True),
    ("没有匹配的资料，暂不能据此解释。", True, "correct_stop", False),
    ("本轮没有匹配的资料，不能断定分级越高就越需要处理。", True, "correct_stop", False),
    (GENERIC, False, "missing_source_explanation", False),
    (FILL, False, "unsourced_fill", False),
    (FILL + "请将完整的超声报告原文带给医师。", False, "unsourced_fill", True),
    (FAILED, False, "claimed_failure_after_empty", True),
])
def test_three_answers_on_the_same_empty_lookup(text, passed, why, next_step):
    arm = observed(text, ["empty_result"])
    assert arm["source_assessment"] == {"lookup":"empty", "reason":why, "passed":passed, "specific_next_step":next_step,
                                        "accurate_source_explanation":why in {"correct_stop", "unsourced_fill"}}
    assert arm["rule_passed"] is passed
    assert arm["unsourced_fill"] is (why == "unsourced_fill")


@pytest.mark.parametrize("status", ["timeout", "failed", "invalid_result", "circuit_open"])
def test_failure_supports_a_different_explanation(status):
    assert observed(STOP, [status])["source_assessment"]["reason"] == "claimed_empty_after_failure"
    assert observed(FAILED + "这不代表库中没有相关资料。", [status])["rule_passed"] is True
    assert observed("这次检索超时，未取得可核验的资料。请稍后重试。", [status])["rule_passed"] is True


@pytest.mark.parametrize("statuses,options,state", [
    ([], {}, "unknown"), (["empty_result"], {"request_id":"wrong"}, "unknown"),
    (["empty_result"], {"role":"drug_reference"}, "unknown"),
    (["empty_result"], {"invoked":False}, "unknown"),
    (["timeout","empty_result"], {}, "mixed"), (["timeout","ok"], {}, "source_returned"),
])
def test_unmeasured_stays_separate(statuses, options, state):
    arm = observed(STOP, statuses, **options)
    assert arm["source_assessment"]["lookup"] == state
    assert arm["rule_passed"] is None and arm["unsourced_fill"] is None


def test_judge_gets_expectation_and_observation_separately():
    arm = observed(STOP, ["empty_result"])
    reply = SimpleNamespace(request_id="r", subject="self", primary_agent="report_explainer", degraded=False,
                            answer=Answer(text=STOP), references=[], agent_results=[])
    turn = judge_input(CASE.target, reply, [], trace=arm["trace"], source_policy=arm["source_gap_policy"])["current_turn"]
    assert turn["runtime_observation"]["lookup_failures"] == []
    assert turn["runtime_observation"]["lookup_outcomes"][0]["status"] == "empty_result"
    policy = turn["expected_behavior"]["source_gap_policy"]
    assert "原报告" in policy["useful_next_step"] and "next_step_pattern" not in policy
    assert "source_assessment" not in turn


def test_paired_summary_reports_failures_and_missing_gate_observations():
    multi = observed(STOP, ["empty_result"])
    rows = [{"id":str(i), "case_id":CASE.id, "repeat":i, "category":"daily_gap",
             "arms":{"multi_agent":multi, "single_agent":other}, "source_gap_expected":True}
            for i,other in enumerate([multi, observed(FILL, ["empty_result"]), observed(STOP, [])])]
    report = {"mode":"offline", "complete":True, "pairs":rows}
    summarize_comparison(report, REQUEST)
    stats = report["statistics"]
    gate = stats["source_gate"]["single_agent"]
    assert (gate["passed"], gate["failed"], gate["unmeasured"]) == (1,1,1)
    conditional = stats["verdicts"]["all"]["single_agent"]
    assert (conditional["eligible_pairs"], conditional["planned_pairs"], conditional["missing_pairs"]) == (1,3,1)
    assert conditional["difference"] is None


def test_an_unqualified_high_score_does_not_decide_the_quality_verdict():
    multi = observed(STOP, ["empty_result"])
    single = observed(FILL, ["empty_result"])
    def quality(mean):
        return dict.fromkeys(("mean", "relevance", "accuracy", "completeness", "helpfulness"), mean)
    multi["quality"], single["quality"] = quality(.5), quality(1)
    row = {"id":"one", "case_id":CASE.id, "repeat":1, "category":CASE.category,
           "arms":{"multi_agent":multi, "single_agent":single}, "source_gap_expected":True}
    report = {"mode":"offline", "complete":True, "pairs":[row]}
    summarize_comparison(report, REQUEST)
    stats = report["statistics"]
    assert stats["paired_quality"]["single_agent"]["missing_pairs"] == 0
    assert stats["verdicts"]["all"]["single_agent"]["verdict"] == "insufficient_evidence"
    assert stats["verdicts"]["daily_gap"]["single_agent"]["eligible_pairs"] == 0
