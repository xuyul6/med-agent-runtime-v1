"""Corpus coverage, retrieval and answer evidence are independently measurable."""
import json

import pytest

from support.comparison import RehearsalModel
from support.harness import role_data, role_output
from medagent.comparison.cli import build_plan
from medagent.comparison.contracts import ComparisonRequest, resolve_comparison
from medagent.comparison.runner import attach_baseline, run_comparison, score_arm
from medagent.comparison.source_gap import bounded_quality
from medagent.core.config import ROOT
from medagent.core.contracts import Answer, Question
from medagent.evaluation.source_relevance_check import run_check
from support.harness import fixture, run


RESOURCES = ROOT / "resources"
REQUEST = ComparisonRequest(layer="daily", repetitions=4)
CASES = {c.id:c for c in resolve_comparison(REQUEST)[1]}
REFS = {r["key"]:r for r in json.loads((RESOURCES / "references.json").read_text())}


def assess(case_id, text, *, keys=(), cited=(), outcome="ok"):
    case = CASES[case_id]
    arm = {"status":"completed", "checks":{}, "safety_checks":{},
           "reply":{"request_id":"r", "answer":Answer(text=text, reference_keys=list(cited)).model_dump(),
                    "references":[REFS[k] for k in cited]},
           "trace":{"request_id":"r", "events":[
               {"event":"retrieval", "domain":"report", "ids":list(keys)},
               {"event":"tool", "role":"report_explainer", "name":"lookup_reference",
                "handler_invoked":True, "status":outcome}]}}
    score_arm(arm, None, case)
    return arm


def test_replay_plan_selects_the_same_catalog_and_case_labels():
    for version in ("comparison-daily-3", "comparison-daily-4"):
        plan = build_plan(REQUEST.model_copy(update={"daily_version":version}))
        sent = ComparisonRequest.model_validate(plan["request"])
        assert resolve_comparison(sent)[0] == version


def test_changed_frozen_reference_is_rejected_before_execution():
    async def changed():
        runtime = await fixture()
        runtime.index.documents["acr-tirads-overview"].text = "different"
        with pytest.raises(ValueError, match="daily_reference_snapshot_mismatch"):
            await run_comparison(runtime, REQUEST)
    run(changed())


def test_frozen_paraphrases_match_the_reviewed_keys():
    result = run(run_check(RESOURCES / "source-relevance-daily-v4.json"))
    assert result["matched_expectation"] == result["total"] == 16


def test_sourced_question_requires_both_the_right_reference_and_an_answer():
    key = "acr-tirads-overview"
    wrong = "lab-context"
    case = "d-sourced-tirads"
    text = REFS[key]["text"]
    assert assess(case, text, keys=[key], cited=[key])["rule_passed"]
    missed = assess(case, "没有查到匹配资料。", outcome="empty_result")
    assert missed["evidence"] == {"retrieved":False, "cited":[]}
    wrong_citation = assess(case, text, keys=[key,wrong], cited=[wrong])
    assert wrong_citation["evidence"] == {"retrieved":True, "cited":[]}
    assert not wrong_citation["rule_passed"]
    assert not assess(case, text, keys=[key,wrong], cited=[key,wrong])["rule_passed"]
    assert not assess(case, "请带原报告咨询医生。", keys=[key], cited=[key])["rule_passed"]


def test_general_report_documents_do_not_support_a_numeric_cutoff():
    case, keys = "d-gap-breath-c13", ["lab-context"]
    stop = assess(case, "没有找到这个界值的依据，请带原报告向医生核对。", keys=keys)
    assert stop["source_assessment"]["lookup"] == "no_applicable_source"
    assert stop["rule_passed"]
    wrong = assess(case, "按参考资料，超过 4 就是阳性。", keys=keys, cited=keys)
    assert wrong["unsourced_fill"] and wrong["inapplicable_citations"] == keys and not wrong["rule_passed"]
    assert assess(case, "没有查到资料。", keys=["unknown-new-document"])["rule_passed"] is None
    raw = {"relevance":1, "accuracy":1, "completeness":.7, "helpfulness":1}
    scored = bounded_quality(raw, stop["source_assessment"])
    assert scored["completeness"] == 1 and scored["helpfulness"] == .9 and raw["completeness"] == .7


def test_institution_time_needs_a_disclosure_and_an_official_channel():
    case = "d-institution-release-time"
    assert assess(case, "无法核实这家医院的实时放号时间，建议查看医院官网。")["rule_passed"]
    assert not assess(case, "每天 9 点放号，请看医院官网。")["rule_passed"]


class ExtractiveModel(RehearsalModel):
    """Return a passage actually available in the prompt, without seeing a gold key or question ID."""
    async def message(self, stage, system, messages, tools=None, budget=1800):
        if stage not in {"agent_report_explainer", "agent_drug_reference", "agent_single_agent"}:
            return await super().message(stage, system, messages, tools, budget)
        data = role_data(messages)
        if not data["references"]:
            return await super().message(stage, system, messages, tools, budget)
        ref = data["references"][0]
        answer = {"text":ref["text"], "reference_keys":[ref["key"]]}
        if stage == "agent_single_agent":
            output = answer
        else:
            output = role_output(stage.removeprefix("agent_"), [ref["key"]])
            output["answer"] = answer
            field = "term_explanations" if stage == "agent_report_explainer" else "evidence_items"
            output["payload"][field] = [answer]
            if stage == "agent_drug_reference":
                output["payload"].update(medication_identity=data["medication_identity"], source_scope=[ref["scope"]])
        return {"content":[{"type":"text", "text":json.dumps(output, ensure_ascii=False)}]}


async def answer_with_retrieved_passage(case, architecture):
    runtime = await fixture(ExtractiveModel())
    attach_baseline(runtime, architecture)
    reply = await runtime.chat(Question(text=case.target.message))
    trace = next(t for t in runtime.monitor.traces if t["request_id"] == reply.request_id)
    arm = {"status":"completed", "checks":{}, "safety_checks":{}, "trace":trace, "reply":reply.report_dump()}
    score_arm(arm, reply, case)
    return arm


@pytest.mark.parametrize("case_id", [k for k in CASES if k.startswith("d-sourced-")])
@pytest.mark.parametrize("architecture", ["multi_agent", "single_agent"])
def test_approved_passage_reaches_the_user_through_both_architectures(case_id, architecture):
    arm = run(answer_with_retrieved_passage(CASES[case_id], architecture))
    assert arm["rule_passed"] and arm["evidence"]["retrieved"]
