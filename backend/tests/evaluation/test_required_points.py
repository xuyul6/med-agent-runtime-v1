from copy import deepcopy
import json
from pathlib import Path

import pytest

from medagent.comparison.contracts import ComparisonCase, ComparisonRequest, resolve_comparison
from medagent.comparison.runner import score_arm
from medagent.core.provider import json_schema
from medagent.evaluation.contracts import EvalLimits
from medagent.evaluation.judging import grade_result, request_grade
from medagent.evaluation.rejudge import pilot_inputs, score
from medagent.evaluation.required_points import PointReviewRequired, required_points
from support.harness import run


PILOT = Path("artifacts/daily-v4-three-real-model-2026-11-22.json")
REGRESSION = json.loads((Path(__file__).parent / "fixtures/amoxicillin_class_judgments.json").read_text())


def grade(status, quote="", source="", key=""):
    return {"points":[{"point_id":"class", "status":status, "answer_quote":quote, "reference_key":key,
                       "source_quote":source}], "reason":"测试", "relevance":1, "accuracy":1, "helpfulness":1}


def class_payload(text, reference, question):
    return {"current_turn":{"question":question, "expected_behavior":{"required_points":{"class":"分类"}},
                            "answer":{"text":text, "reference_keys":[reference["key"]], "questions":[],
                                      "limitations":[], "next_steps":[]}, "references":[reference]}}


@pytest.mark.parametrize("row", REGRESSION["cases"], ids=lambda row:row["id"])
def test_archived_judgments_and_comparison_rule_agree_on_the_six_answers(row):
    ref = REGRESSION["reference"]
    payload = class_payload(row["text"], ref, REGRESSION["question"])
    score = grade_result(payload, row["raw_judge"])
    expected = row["expected"] == "met"
    assert score["completeness"] == int(expected)
    assert score["points"][0]["status"] == row["expected"]
    case = next(case for case in resolve_comparison(ComparisonRequest(layer="daily", repetitions=4))[1]
                if case.id == "d-sourced-amoxicillin")
    arm = {"status":"completed", "checks":{}, "safety_checks":{},
           "reply":{"answer":payload["current_turn"]["answer"], "references":[{**ref,"domain":"medicine"}]}}
    score_arm(arm, None, case)
    assert arm["checks"]["expected_point_answered"] is expected


@pytest.mark.parametrize("answer,quote", [
    ("这种药用于治疗某些细菌感染。", "用于治疗某些细菌感染"),
    ("我不能肯定，它属于青霉素类抗生素。", "它属于青霉素类抗生素"),
    ("阿莫西林并不属于青霉素类抗生素。", "青霉素类抗生素"),
    ("阿莫西林属于青霉素类抗生素吗？", "阿莫西林属于青霉素类抗生素"),
    ("阿莫西林属于青霉素类抗生素。对此我还不确定。", "青霉素类抗生素"),
    ("阿莫西林用于治疗细菌感染。另一种药是青霉素类抗生素。", "青霉素类抗生素"),
    ("阿莫西林用于治疗细菌感染。布洛芬属于青霉素类抗生素。", "青霉素类抗生素"),
    ("先看布洛芬。它属于青霉素类抗生素。", "它属于青霉素类抗生素"),
    ("举例：青霉素类抗生素。", "青霉素类抗生素"),
])
def test_false_positive_judge_label_is_held_for_review(answer, quote):
    ref = REGRESSION["reference"]
    payload = class_payload(answer, ref, REGRESSION["question"])
    with pytest.raises(PointReviewRequired):
        grade_result(payload, grade("met", quote, "青霉素类抗生素", ref["key"]))


def test_parent_excerpt_requires_an_adjacent_asserted_family_in_same_field():
    ref = REGRESSION["reference"]
    payload = class_payload("阿莫西林是一种抗生素。它属于青霉素这一族。", ref, REGRESSION["question"])
    raw = grade("met", "阿莫西林是一种抗生素", "青霉素类抗生素", ref["key"])
    assert grade_result(payload, raw)["completeness"] == 1
    payload["current_turn"]["answer"].update(text="阿莫西林是一种抗生素。", questions=["它属于青霉素类吗？"])
    with pytest.raises(PointReviewRequired):
        grade_result(payload, raw)


def test_metformin_name_or_other_drug_does_not_prove_the_family():
    ref = {"key":"metformin-overview", "approved":True, "text":"二甲双胍属于双胍类药物。"}
    question = "二甲双胍属于哪一类药物？"
    case = ComparisonCase(id="metformin-class", category="daily_sourced", target={"message":question},
                          reference_domains=["medicine"], evidence_passages={ref["key"]:ref["text"]}, answer_point_pattern="双胍类")
    assert required_points(case).keys() == {"class"}
    for text in ["二甲双胍帮助控制血糖。", "不清楚二甲双胍属于哪一类。", "二甲双胍属于双胍类药物。"]:
        # Even in a correct full answer, quoting just the drug name is insufficient.
        with pytest.raises(PointReviewRequired):
            grade_result(class_payload(text,ref,question), grade("met", "二甲双胍", "双胍类药物", ref["key"]))


@pytest.mark.parametrize("answer", [
    "二甲双胍是一种双胍类药物。",
    "二甲双胍是一种双胍类药物，其作用包括改善身体对胰岛素的反应。",
])
def test_reviewed_source_can_mention_another_substance_outside_the_classification(answer):
    case = next(c for c in resolve_comparison(ComparisonRequest(layer="daily", repetitions=4))[1]
                if c.id == "d-sourced-metformin")
    key = "metformin-overview"
    ref = {"key":key, "approved":True, "text":case.evidence_passages[key]}
    assert "胰岛素" in ref["text"]
    payload = class_payload(answer, ref, "二甲双胍属于哪一类药物？")
    raw = grade("met", "双胍类药物", "二甲双胍属于双胍类药物", key)
    assert grade_result(payload, raw)["completeness"] == 1


def test_rubric_only_attaches_to_scoped_questions():
    cases = resolve_comparison(ComparisonRequest(layer="daily", repetitions=4))[1]
    annotated = {case.id:required_points(case) for case in cases if required_points(case)}
    assert {key:len(value) for key, value in annotated.items()} == {
        "d-sourced-amoxicillin":1, "d-sourced-tirads":3, "d-sourced-metformin":2}
    case = next(case for case in cases if case.id == "d-sourced-amoxicillin")
    assert required_points(case.model_copy(update={"evidence_passages":{}})) is None


@pytest.mark.parametrize("text,correct", [
    ("从分类上说，青霉素类抗生素里就包括阿莫西林。", True),
    ("青霉素类抗生素里不包括阿莫西林。", False),
    ("我不确定，青霉素类抗生素里包括阿莫西林。", False),
    ("青霉素类抗生素里有另一种药。阿莫西林用于某些细菌感染。", False),
])
def test_family_first_still_requires_positive_membership(text, correct):
    ref = REGRESSION["reference"]
    payload = class_payload(text, ref, REGRESSION["question"])
    raw = grade("met", text, "青霉素类抗生素", ref["key"])
    if correct:
        assert grade_result(payload, raw)["completeness"] == 1
    else:
        with pytest.raises(PointReviewRequired):
            grade_result(payload, raw)


def test_inverse_metformin_class_requires_evidence_beyond_the_drug_name():
    ref = {"key":"metformin-overview", "approved":True, "text":"二甲双胍属于双胍类药物。"}
    text = "双胍类降糖药里就有二甲双胍。"
    payload = class_payload(text, ref, "二甲双胍属于哪一类药物？")
    assert grade_result(payload, grade("met", text, "双胍类药物", ref["key"]))["completeness"] == 1
    with pytest.raises(PointReviewRequired):
        grade_result(payload, grade("met", "二甲双胍", "双胍类药物", ref["key"]))


def test_point_scoring_requires_answer_and_cited_evidence_and_valid_status():
    key, claim = "amoxicillin-overview", "青霉素类抗生素"
    payload = {"current_turn":{"question":"阿莫西林是什么类型的药", "expected_behavior":{"required_points":{"class":"分类"}},
        "answer":{"text":claim, "reference_keys":[key], "questions":[], "limitations":[], "next_steps":[]},
        "references":[{"key":key, "approved":True, "text":claim}]}}
    raw = grade("met", claim, claim, key)
    assert grade_result(payload, raw)["completeness"] == 1
    usage = "用于治疗某些细菌感染"
    payload["current_turn"]["answer"]["text"] = claim + "，" + usage
    with pytest.raises(PointReviewRequired):
        grade_result(payload, grade("met", usage, claim, key))
    for alternative in [grade("missing"), grade("unsupported", claim)]:
        assert grade_result(payload, alternative)["completeness"] == 0
    for invalid in [grade("met", "杜撰", claim, key), grade("met", claim, "杜撰", key),
                    grade("missing", claim), {**raw, "points":[]}, {**raw, "points":raw["points"]*2}]:
        with pytest.raises(ValueError):
            grade_result(payload, invalid)
    payload["current_turn"]["answer"]["reference_keys"] = []
    with pytest.raises(ValueError, match="evidence_mismatch"):
        grade_result(payload, raw)
    payload["current_turn"]["answer"]["reference_keys"] = [key]
    payload["current_turn"]["references"][0]["approved"] = False
    with pytest.raises(ValueError, match="evidence_mismatch"):
        grade_result(payload, raw)


def test_point_schema_and_stage_are_used_by_real_grading_path():
    class Judge:
        async def json(self, stage, system, content, budget):
            assert stage == "evaluation_judge_points" and budget == 1500
            assert "points" in json_schema(stage)["required"]
            assert "required_points" in json.loads(content)["current_turn"]["expected_behavior"]
            return grade("missing")

    result = run(request_grade(Judge(), {"current_turn":{"expected_behavior":{"required_points":{"class":"分类"}}}}, 1500))
    assert result["points"][0]["status"] == "missing"


@pytest.mark.skipif(not PILOT.exists(), reason="local frozen pilot")
def test_replay_matches_all_hashes_and_controls_change_only_the_intended_variable():
    saved = json.loads(PILOT.read_text())
    inputs = pilot_inputs(saved, controls=True)
    assert len(inputs) == 8
    for _, payload in inputs:
        turn = payload["current_turn"]
        assert "agent_results" not in turn and "primary_agent" not in turn["runtime_observation"]
        assert "control" not in payload
    original = next(p for i, p in inputs if i["case_id"] == "d-sourced-amoxicillin" and i["arm"] == "multi_agent")
    omission, wrong = [deepcopy(p) for _, p in inputs[-2:]]
    assert "青霉素" not in omission["current_turn"]["answer"]["text"]
    omission["current_turn"]["answer"]["text"] = original["current_turn"]["answer"]["text"]
    assert omission == original
    assert wrong["current_turn"]["answer"]["reference_keys"] == ["metformin-overview"]
    wrong["current_turn"]["answer"]["reference_keys"] = original["current_turn"]["answer"]["reference_keys"]
    wrong["current_turn"]["references"] = original["current_turn"]["references"]
    assert wrong == original
    limits = EvalLimits(max_model_calls=8)
    with pytest.raises(ValueError, match="exceeds_call_budget"):
        run(score(saved, None, limits.model_copy(update={"max_model_calls":7}), per_category=6, repeats=1, seed=0, pilot_controls=True))
    saved["pairs"][-1]["arms"]["single_agent"]["reply"]["answer"]["text"] += "被改过"
    with pytest.raises(ValueError, match="original_input_mismatch"):
        run(score(saved, None, limits, per_category=6, repeats=1, seed=0, pilot_controls=True))
