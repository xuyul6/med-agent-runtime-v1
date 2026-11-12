"""Fault-aware guidance and Judge framing based on bounded observed evidence."""
import json
from types import SimpleNamespace

import pytest

from medagent.benchmark.contracts import PROTOCOL
from medagent.benchmark.runner import score_observation
from medagent.core.contracts import Intent, Question
from medagent.evaluation.runner import evaluate
from medagent.evaluation.budget import ModelBudget
from medagent.evaluation.contracts import EvalLimits, EvaluationRequest, FaultSpec
from medagent.evaluation.isolation import injected_faults
from medagent.evaluation.metrics import BaselineError, POLICY, compare_baseline, digest
from medagent.evaluation.judging import JUDGE_PROTOCOL_VERSION, judge_input
from support.evaluation import JudgeModel
from support.harness import fixture, run
from support.judge_context import historical_subject_turn
from support.routing import field, fixed_understanding, understanding


class CapturingJudge(JudgeModel):
    def __init__(self, score=.5):
        super().__init__(score=score)
        self.judge_payloads = []

    async def json(self, stage, system, content, budget=900):
        if stage == "evaluation_judge":
            self.judge_payloads.append(json.loads(content))
        return await super().json(stage, system, content, budget)


def test_timeout_guidance_and_judge_use_observed_failure_without_changing_low_score():
    async def scenario():
        model = CapturingJudge()
        runtime = await fixture(model)
        body = EvaluationRequest(intent_cases=[], dialog_cases=[{
            "id":"lookup-timeout", "turns":[{"message":"查询布洛芬公开药品资料。",
                "expected":{"role":"drug_reference", "degraded":True, "references":False}}],
            "faults":[{"kind":"tool_timeout"}]}])
        report = await evaluate(runtime, body)
        row = report["cases"][0]["turns"][0]
        reply = row["reply"]
        assert row["rule_passed"] and not row["passed"] and row["quality"]["mean"] == .5
        assert reply["degraded"] and not reply["references"] and not reply["answer"]["reference_keys"]
        assert "超时" in reply["answer"]["text"] and len(reply["answer"]["next_steps"]) == 2
        assert "不代表相关资料不存在" in reply["answer"]["text"]
        assert all(item in reply["answer"]["next_steps"][0] for item in ["重试", "名称", "剂型", "规格", "地区", "未知"])
        assert "自行" in reply["answer"]["next_steps"][1] and "药师" in reply["answer"]["next_steps"][1]
        assert all(row["safety_checks"].values())
        observation = model.judge_payloads[0]["current_turn"]["runtime_observation"]
        assert observation["degraded"] is True
        assert observation["lookup_failures"] == [{"tool":"lookup_reference", "role":"drug_reference", "status":"timeout"}]
        assert model.judge_payloads[0]["current_turn"]["expected_behavior"]["degraded"] is True
        assert row["judge_context"]["lookup_observation_available"] is True
        assert row["judge_context"]["observed_lookup_failure_count"] == 1
        assert report["metadata"]["policy"]["judge_protocol_version"] == "judge-context-5"

    run(scenario())


def test_agent_output_failure_without_tool_timeout_does_not_claim_timeout():
    async def scenario():
        runtime = await fixture(CapturingJudge())
        body = EvaluationRequest(intent_cases=[], judge=False, dialog_cases=[{
            "id":"bad-output", "turns":[{"message":"查询布洛芬公开药品资料。",
                "expected":{"role":"drug_reference", "degraded":True}}],
            "faults":[{"kind":"agent_invalid_output"}]}])
        report = await evaluate(runtime,body)
        turn = report["cases"][0]["turns"][0]
        assert turn["reply"]["degraded"] and "超时" not in turn["reply"]["answer"]["text"]
        assert not any(item.get("event") == "tool" and item.get("name") == "lookup_reference"
                       and item.get("status") == "timeout" for item in turn["trace"]["events"])

    run(scenario())


def test_timeout_guidance_survives_successful_support_merge():
    class SupportJudge(CapturingJudge):
        async def message(self, stage, *args, **kwargs):
            result = await super().message(stage,*args,**kwargs)
            if stage == "agent_symptom_guide":
                payload = json.loads(result["content"][0]["text"])
                payload["answer"]["next_steps"] = ["补充症状时间线。"]
                result["content"][0]["text"] = json.dumps(payload,ensure_ascii=False)
            return result

    async def scenario():
        runtime = await fixture(SupportJudge())
        fixed_understanding(runtime, understanding(primary=Intent.MEDICINE, kinds=(Intent.MEDICINE,Intent.SYMPTOMS),
            entities={"medication_name":field("medication_name","布洛芬")}))
        async with injected_faults(runtime,[FaultSpec(kind="tool_timeout")],1):
            reply = await runtime.chat(Question(text="查询布洛芬药品资料，也整理头痛症状。"))
        assert reply.primary_agent == "drug_reference" and reply.supporting_agents == ["symptom_guide"]
        assert reply.degraded and "超时" in reply.answer.text and "已完成的部分" in reply.answer.text
        assert any("药品名称" in item for item in reply.answer.next_steps)
        assert "补充症状时间线。" in reply.answer.next_steps
        assert not reply.answer.reference_keys

    run(scenario())


def test_expected_degraded_does_not_become_observed_failure_without_trace():
    turn,reply,history = historical_subject_turn()
    turn = turn.model_copy(update={"expected":turn.expected.model_copy(update={"degraded":True})})
    reply.degraded = False
    current = judge_input(turn,reply,history)["current_turn"]
    assert current["expected_behavior"]["degraded"] is True
    assert current["runtime_observation"]["degraded"] is False
    assert current["runtime_observation"]["lookup_failures"] is None


def test_judge_only_receives_matching_redacted_and_bounded_actual_trace():
    turn,reply,history = historical_subject_turn()
    reply.request_id,reply.degraded = "current",True
    sensitive = "SYNTHETIC_SECRET_DO_NOT_SEND"
    trace = {"request_id":"current", "events":[
        {"event":"tool", "name":"lookup_reference", "role":"drug_reference", "status":"timeout",
         "handler_invoked":True, "arguments":sensitive, "error":sensitive},
        {"event":"tool", "name":"lookup_reference", "role":"drug_reference", "status":"timeout", "handler_invoked":True},
        {"event":"tool", "name":"lookup_reference", "role":"drug_reference", "status":"empty_result", "handler_invoked":True},
        {"event":"tool", "name":"lookup_reference", "role":"drug_reference", "status":"timeout", "handler_invoked":False},
        {"event":"tool", "name":"other", "role":"drug_reference", "status":"failed"},
        {"event":"tool", "name":"lookup_reference", "role":[], "status":"failed"},
    ]}
    payload = judge_input(turn,reply,history,trace=trace)
    observation = payload["current_turn"]["runtime_observation"]
    assert observation["lookup_failures"] == [{"tool":"lookup_reference", "role":"drug_reference", "status":"timeout"}]
    assert sensitive not in json.dumps(payload,ensure_ascii=False)
    assert judge_input(turn,reply,history,trace={**trace,"request_id":"another"})["current_turn"]["runtime_observation"]["lookup_failures"] is None
    assert judge_input(turn,reply,history,trace={"request_id":"current", "events":[]})["current_turn"]["runtime_observation"]["lookup_failures"] == []
    assert judge_input(turn,reply,history)["current_turn"]["runtime_observation"]["lookup_failures"] is None


def test_benchmark_judge_uses_its_own_request_trace():
    async def scenario():
        turn,reply,history = historical_subject_turn()
        reply.request_id,reply.degraded = "one",True
        model = CapturingJudge()
        row = {"trace":{"request_id":"one", "events":[{"event":"tool", "name":"lookup_reference",
            "role":"drug_reference", "status":"timeout", "handler_invoked":True}]}}
        await score_observation(reply,turn,SimpleNamespace(history=history),SimpleNamespace(judge_model=model),ModelBudget(EvalLimits()),row)
        observation = model.judge_payloads[0]["current_turn"]["runtime_observation"]
        assert observation["lookup_failures"] == [{"tool":"lookup_reference", "role":"drug_reference", "status":"timeout"}]
        assert row["judge_context"]["protocol_version"] == PROTOCOL["judge_protocol_version"] == JUDGE_PROTOCOL_VERSION

    run(scenario())


def test_old_judge_policy_cannot_be_compared_as_current():
    old_policy = {**POLICY, "judge_protocol_version":"judge-context-2"}
    metadata = {"dataset_hash":"same", "metric_version":"same", "policy_hash":digest(POLICY)}
    old_metadata = {**metadata, "policy_hash":digest(old_policy)}
    with pytest.raises(BaselineError,match="baseline_dataset_or_policy_incompatible"):
        compare_baseline({"metadata":metadata}, {"metadata":old_metadata})
