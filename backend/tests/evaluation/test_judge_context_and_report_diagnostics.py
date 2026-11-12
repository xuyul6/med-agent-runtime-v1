"""Offline framing and contract regressions; no paid model or rescoring."""
import hashlib
import json
import pytest
from pydantic import BaseModel, ValidationError, model_validator
from pydantic_core import PydanticCustomError
from medagent.benchmark.contracts import PROTOCOL
from medagent.core.contracts import Question
from medagent.evaluation.runner import CODE_HASH, evaluate, judge_turn
from medagent.evaluation.contracts import EvaluationRequest
from medagent.evaluation.metrics import BaselineError, BaselineStore, POLICY, digest
from medagent.agents.checks import checked_role_output
from medagent.evaluation.judging import JUDGE_PROTOCOL_VERSION, judge_audit, judge_input
from medagent.agents.output_validation import AgentOutputError, schema_error
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.core.provider import role_json_schema
from support.evaluation import JudgeModel
from medagent.core.config import ROOT
from support.harness import Model, fixture, role_output, run
from support.judge_context import historical_subject_turn


SECRET = "SYNTHETIC_PRIVATE_SCHEMA_SENTINEL"


def test_original_failed_question_is_current_turn_with_prior_mother_only_in_history():
    turn,reply,history = historical_subject_turn()
    payload = judge_input(turn,reply,history)
    current = payload["current_turn"]
    assert current["index"] == 3 and current["question"] == "改回我自己，这个药的资料呢？"
    assert current["expected_behavior"]["subject"] == current["runtime_observation"]["subject"] == "self"
    assert payload["history"] == history and "妈妈" in history[-2]["content"]
    assert all(item["content"] != turn.message for item in payload["history"])
    assert current["answer"]["reference_keys"] == ["ibuprofen-overview"]


def test_runtime_subject_is_an_observation_and_cannot_replace_expected_subject():
    turn,reply,history = historical_subject_turn()
    reply.subject = "mother"
    payload = judge_input(turn,reply,history)
    assert payload["current_turn"]["expected_behavior"]["subject"] == "self"
    assert payload["current_turn"]["runtime_observation"]["subject"] == "mother"
    assert judge_audit(payload)["observed_subject_matches_expected"] is False


@pytest.mark.parametrize("question,subject", [("现在给我爸爸问，这个药呢？","father"),
    ("改回我自己，这个药呢？","self"),("更正刚才的持续时间，不是三天，是两天。","self")])
def test_current_switch_or_correction_is_not_replaced_by_prior_user_text(question,subject):
    turn,reply,history = historical_subject_turn()
    turn = turn.model_copy(update={"message":question,"expected":turn.expected.model_copy(update={"subject":subject})})
    reply.subject = subject
    payload = judge_input(turn,reply,history)
    assert payload["current_turn"]["question"] == question
    assert payload["current_turn"]["expected_behavior"]["subject"] == subject
    assert payload["history"][-2]["content"] == "现在给我妈妈问，这个药的资料呢？"


def test_judge_request_and_private_audit_do_not_override_a_low_score():
    class Capture:
        async def json(self, stage, system, content, budget):
            self.payload,self.instructions = json.loads(content),system
            assert stage == "evaluation_judge" and budget == 950
            return {**{key:.2625 for key in ("relevance","accuracy","completeness","helpfulness")},
                    "reason":"合成低分，仅核对不覆盖 Judge 返回值。"}
    turn,reply,history = historical_subject_turn()
    model,audit = Capture(),{}
    quality = run(judge_turn(model,turn,reply,history,audit=audit))
    assert quality["mean"] == .2625
    assert model.payload["current_turn"]["question"] == turn.message
    assert "当前原话优先" in model.instructions and "不是标准答案" in model.instructions
    assert audit["protocol_version"] == JUDGE_PROTOCOL_VERSION and audit["turn_index"] == 3
    assert turn.message not in json.dumps(audit,ensure_ascii=False)


def test_context_audit_is_finite_and_changes_with_the_current_question():
    turn,reply,history = historical_subject_turn()
    payload = judge_input(turn,reply,history)
    first = judge_audit(payload)
    payload["current_turn"]["question"] = SECRET
    second = judge_audit(payload)
    assert first["input_sha256"] != second["input_sha256"]
    assert SECRET not in json.dumps(second) and len(second["input_sha256"]) == 64


def test_dialog_evaluator_records_current_turn_context_for_each_real_runtime_reply():
    async def scenario():
        class Capture(JudgeModel):
            def __init__(self): super().__init__(score=.5); self.payloads = []
            async def json(self,stage,system,content,budget=900):
                if stage == "evaluation_judge": self.payloads.append(json.loads(content))
                return await super().json(stage,system,content,budget)
        model = Capture()
        body = EvaluationRequest(intent_cases=[],dialog_cases=[{"id":"core","turns":["你好","谢谢"]}])
        report = await evaluate(await fixture(model),body)
        turns = report["cases"][0]["turns"]
        assert [item["judge_context"]["turn_index"] for item in turns] == [1,2]
        assert [item["current_turn"]["question"] for item in model.payloads] == ["你好","谢谢"]
        assert model.payloads[1]["history"][0]["content"] == "你好"
        assert report["passed"] == 0 and all(item["quality"]["mean"] == .5 for item in turns)
        assert report["metadata"]["policy"]["judge_protocol_version"] == JUDGE_PROTOCOL_VERSION
    run(scenario())


def test_previous_judge_policy_is_rejected_before_any_new_model_call(tmp_path):
    async def scenario():
        class Count(JudgeModel):
            def __init__(self): super().__init__(); self.json_calls = 0
            async def json(self,*args,**kwargs):
                self.json_calls += 1
                return await super().json(*args,**kwargs)
        model = Count();runtime = await fixture(model)
        body = EvaluationRequest(intent_cases=[],dialog_cases=[{"id":"same","turns":["你好"]}])
        old = await evaluate(runtime,body)
        old["metadata"]["policy"].pop("judge_protocol_version")
        old["metadata"]["policy_hash"] = digest(old["metadata"]["policy"])
        store = BaselineStore(tmp_path);store.save("previous-policy",old)
        before = model.json_calls
        compared = body.model_copy(update={"baseline":body.baseline.model_copy(update={"compare_to":"previous-policy"})})
        with pytest.raises(BaselineError,match="baseline_dataset_or_policy_incompatible"):
            await evaluate(runtime,compared,baseline_store=store)
        assert model.json_calls == before
    run(scenario())


def test_paired_protocol_records_the_same_judge_revision():
    assert PROTOCOL["judge_protocol_version"] == POLICY["judge_protocol_version"] == JUDGE_PROTOCOL_VERSION


def test_paired_runner_uses_current_turn_audit_for_each_arm_without_altering_history():
    from medagent.benchmark.runner import run_benchmark
    from test_benchmark import CountingModel, small_request
    async def scenario():
        body = small_request(judge=True,cache_conditions=["cold"])
        report = await run_benchmark(await fixture(CountingModel()),body)
        assert report["complete"]
        for pair in report["pairs"]:
            for arm in pair["arms"].values():
                assert arm["judge_context"]["protocol_version"] == JUDGE_PROTOCOL_VERSION
                assert arm["judge_context"]["turn_index"] == 3
                assert arm["judge_context"]["history_user_turns"] == 2
    run(scenario())


@pytest.mark.parametrize("role,rule,fields", [
    ("general","task_summary_required",{"task_summary"}),
    ("symptom_guide","symptom_summary_required",{"symptom_summary"}),
    ("drug_reference","medication_evidence_required",{"medication_identity","source_scope","evidence_items"}),
    ("report_explainer","report_information_required",{"report_items","term_explanations"}),
    ("visit_guide","visit_checklist_required",{"preparation_checklist"}),
])
def test_root_business_rules_produce_safe_fields_without_exception_messages(role,rule,fields):
    data = role_output(role);data["payload"] = {};data["answer"]["text"] = SECRET
    with pytest.raises(AgentOutputError) as caught: checked_role_output(DEFAULT_REGISTRY[role],data,{})
    diagnostic = caught.value.diagnostic()
    assert diagnostic["code"] == "invalid_output_shape" and diagnostic["rules"] == [rule]
    assert set(diagnostic["fields"]) == fields and not diagnostic["repairable"]
    assert SECRET not in json.dumps(diagnostic) and SECRET not in str(caught.value)


@pytest.mark.parametrize("status,missing,rule", [
    ("needs_clarification",[],"clarification_requires_missing_fields"),
    ("answered",["value"],"missing_fields_require_clarification"),
])
def test_state_rules_have_specific_diagnostics_without_changing_acceptance(status,missing,rule):
    data = role_output("report_explainer");data.update(status=status,missing_fields=missing)
    with pytest.raises(AgentOutputError) as caught: checked_role_output(DEFAULT_REGISTRY["report_explainer"],data,{})
    assert caught.value.diagnostic()["rules"] == [rule]
    assert set(caught.value.fields) == {"status","missing_fields"}


def test_wrong_nested_report_type_reports_only_safe_path_and_rule():
    data = role_output("report_explainer");data["payload"]["term_explanations"] = [SECRET]
    with pytest.raises(AgentOutputError) as caught: checked_role_output(DEFAULT_REGISTRY["report_explainer"],data,{})
    diagnostic = caught.value.diagnostic()
    assert "term_explanations" in diagnostic["fields"] and diagnostic["rules"] == ["wrong_field_type"]
    assert SECRET not in json.dumps(diagnostic)


def test_unknown_schema_keys_and_custom_error_types_are_never_logged():
    data = role_output("report_explainer");data["payload"][SECRET] = SECRET
    with pytest.raises(AgentOutputError) as caught: checked_role_output(DEFAULT_REGISTRY["report_explainer"],data,{})
    assert "unknown_field" in caught.value.fields and caught.value.diagnostic()["rules"] == ["unexpected_field"]
    class Untrusted(BaseModel):
        @model_validator(mode="after")
        def fail(self): raise PydanticCustomError(SECRET,SECRET,{"private":SECRET})
    with pytest.raises(ValidationError) as error: Untrusted.model_validate({})
    diagnostic = schema_error(error.value).diagnostic()
    assert diagnostic["rules"] == ["schema_validation_failed"]
    assert SECRET not in json.dumps(diagnostic) and SECRET not in json.dumps(caught.value.diagnostic())


@pytest.mark.parametrize("persistent", [False,True])
def test_report_bad_shape_uses_same_role_fallback_and_never_repair_or_fabricate(persistent):
    async def scenario():
        class ReportDraft(Model):
            def __init__(self): self.report_calls = 0
            async def message(self,stage,system,messages,tools=None,budget=1800):
                if stage == "agent_report_explainer":
                    self.report_calls += 1
                    if persistent or self.report_calls == 1:
                        data = role_output("report_explainer",["lab-context"]);data["payload"] = {}
                        return {"content":[{"type":"text","text":json.dumps(data)}]}
                return await super().message(stage,system,messages,tools,budget)
        model = ReportDraft();runtime = await fixture(model)
        # This statement is outside the direct-answer grammar, so it exercises
        # model-output validation even with the reviewed summary available.
        reply = await runtime.chat(Question(text="报告上的参考范围与其他报告信息有什么联系？"))
        events = runtime.monitor.traces[-1]["events"]
        validations = [e for e in events if e["event"] == "agent_output_validation"]
        assert model.report_calls == 2 and reply.degraded is persistent
        assert all(e["rules"] == ["report_information_required"] and not e["repair_requested"] for e in validations)
        assert not any(e["event"] == "agent_output_repair" for e in events)
        assert len(reply.routing_decision.assignments[0].attempts) == 2
        if not persistent:
            assert reply.agent_results[0].payload["report_items"] == []
            assert reply.agent_results[0].payload["term_explanations"][0]["reference_keys"] == ["lab-context"]
        return reply,events
    run(scenario())


def test_report_schema_and_skill_explain_payload_without_relaxing_types():
    runtime = run(fixture())
    payload = role_json_schema("report_explainer")["$defs"]["ReportPayload"]
    assert payload["properties"]["term_explanations"]["type"] == "array"
    assert payload["properties"]["term_explanations"]["items"] == {"$ref":"#/$defs/EvidenceItem"}
    assert "正文不能代替此载荷" in payload["properties"]["term_explanations"]["description"]
    assert runtime.registry["report_explainer"].version == "role-contracts-6"
    assert runtime.skills.snapshot().documents["report_explainer"].version == "1.7.0"


async def offline_report():
    turn,reply,history = historical_subject_turn()
    framed = judge_input(turn,reply,history)
    original = ROOT/"artifacts/runtime-core-online-2026-10-27.json"
    baseline = ROOT/"artifacts/baselines/runtime-core-online-20261027.json"
    old = json.loads(original.read_text())
    old_baseline = json.loads(baseline.read_text())
    rules = {}
    for role in ("report_explainer","general","symptom_guide","drug_reference","visit_guide"):
        data = role_output(role);data["payload"] = {}
        try: checked_role_output(DEFAULT_REGISTRY[role],data,{})
        except AgentOutputError as exc: rules[role] = exc.diagnostic()
    checks = {
        "historical_third_question_is_current":framed["current_turn"]["question"] == "改回我自己，这个药的资料呢？",
        "historical_mother_question_remains_prior":framed["history"][-2]["content"] == "现在给我妈妈问，这个药的资料呢？",
        "current_expected_and_observed_self_not_overwritten":framed["current_turn"]["expected_behavior"]["subject"] == framed["current_turn"]["runtime_observation"]["subject"] == "self",
        "root_report_error_has_safe_rule_and_fields":rules["report_explainer"]["rules"] == ["report_information_required"] and set(rules["report_explainer"]["fields"]) == {"report_items","term_explanations"},
        "schema_failures_still_nonrepairable":all(not diagnostic["repairable"] for diagnostic in rules.values()),
        "new_judge_policy_distinct_from_old_baseline":POLICY["judge_protocol_version"] == JUDGE_PROTOCOL_VERSION and "judge_protocol_version" not in old_baseline["metadata"]["policy"],
        "old_failed_score_and_counts_preserved":old["passed"] == old_baseline["passed"] == 13 and next(row for row in old["cases"] if row["id"] == "consultation-subject-isolation")["turns"][2]["quality"]["mean"] == .2625,
    }
    assert all(checks.values())
    return {"kind":"offline_judge_framing_and_synthetic_contract_diagnostics","paid_model_calls":0,"quality_evaluation":False,
        "code_hash":CODE_HASH,"judge_protocol_version":JUDGE_PROTOCOL_VERSION,"report_profile_version":DEFAULT_REGISTRY["report_explainer"].version,
        "prior_real_report_sha256":hashlib.sha256(original.read_bytes()).hexdigest(),"prior_baseline_sha256":hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "judge_request_audit":judge_audit(framed),"judge_history_kind":"synthetic_prior_replies_with_original_checked_in_questions_not_an_online_replay",
        "synthetic_root_diagnostics":rules,"checks":checks,
        "online_retest":"not_run","historical_report_exact_field_cause":"unknown_original_draft_not_saved"}


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser();parser.add_argument("--output",required=True);args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise FileExistsError("preserve_existing_report")
    result = run(offline_report())
    with output.open("x") as handle:handle.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(f"{sum(result['checks'].values())}/{len(result['checks'])} checks passed; paid model calls=0. Saved: {output}")
