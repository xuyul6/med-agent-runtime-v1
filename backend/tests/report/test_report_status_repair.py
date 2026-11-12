"""A narrow retry for conflicting report status, with safety checks preserved."""
import json

import pytest

from medagent.core.contracts import Intent, Question
from medagent.evaluation.budget import BudgetedModel, EvaluationBudgetExceeded, ModelBudget
from medagent.evaluation.contracts import EvalLimits
from support.harness import Model, fixture, role_output, run, role_data


QUESTION = "我想整理化验报告的数值和单位，不需要分析个人病情。"
SECRET = "SYNTHETIC_REJECTED_STATUS_DRAFT"


class StatusModel(Model):
    def __init__(self, outcome="corrected"):
        self.outcome = outcome
        self.calls = []

    async def message(self, stage, system, messages, tools=None, budget=1800):
        if stage != "agent_report_explainer":
            return await super().message(stage, system, messages, tools, budget)
        repair = len(messages) > 1 and isinstance(messages[-1]["content"], str)
        self.calls.append({"repair":repair, "tools":tools is not None,
                           "feedback":json.loads(messages[-1]["content"]) if repair else None})
        if repair and self.outcome == "tool":
            return {"content":[{"type":"tool_use", "id":"forbidden", "name":"lookup_reference", "input":{}}]}
        context = role_data(messages)
        key = context["references"][0]["key"]
        data = role_output("report_explainer", [key])
        if not repair or self.outcome == "persistent":
            data["missing_fields"] = ["report_item"]
            data["answer"]["text"] += SECRET
        if not repair and self.outcome == "boundary":
            data["answer"]["text"] = "你患有某种疾病。" + SECRET
        if not repair and self.outcome == "unknown_reference":
            data["answer"]["reference_keys"] = ["not-an-approved-reference"]
        if not repair and self.outcome == "unsupported_range":
            data["answer"]["text"] = "参考范围是自己编造的统计区间。"
        if not repair and self.outcome == "invented_report_item":
            data["payload"]["report_items"] = [{"name":"虚构项目", "value":"1", "unit":"mg", "reference_range":"2-3"}]
        if repair and self.outcome == "repair_unknown_reference":
            data["answer"]["reference_keys"] = ["not-an-approved-reference"]
        return {"content":[{"type":"text", "text":json.dumps(data, ensure_ascii=False)}]}


def test_safe_status_conflict_gets_one_retry_and_complete_revalidation():
    async def scenario():
        model = StatusModel()
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text=QUESTION))
        assert not reply.degraded and reply.agent_results[0].status == "answered"
        assert [call["repair"] for call in model.calls] == [False, True]
        assert model.calls[0]["tools"] and not model.calls[1]["tools"]
        assert model.calls[1]["feedback"]["repair_strategy"] == "report_status_only"
        assert model.calls[1]["feedback"]["validation"]["rules"] == ["missing_fields_require_clarification"]
        events = runtime.monitor.traces[-1]["events"]
        assert any(event.get("repair_strategy") == "report_status_only" and event.get("repair_requested")
                   for event in events if event["event"] == "agent_output_validation")
        assert any(event["event"] == "agent_output_repair" and event["status"] == "accepted" for event in events)
        assert SECRET not in reply.answer.text and SECRET not in json.dumps(events, ensure_ascii=False)
    run(scenario())


@pytest.mark.parametrize("outcome", ["boundary", "unknown_reference", "unsupported_range", "invented_report_item"])
def test_status_retry_does_not_mask_other_invalid_draft_constraints(outcome):
    async def scenario():
        model = StatusModel(outcome)
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text=QUESTION))
        assert reply.degraded and not reply.agent_results
        assert len(model.calls) == 2 and all(not call["repair"] for call in model.calls)
        assert SECRET not in reply.answer.text and SECRET not in json.dumps(runtime.monitor.traces[-1], ensure_ascii=False)
    run(scenario())


def test_status_retry_respects_existing_model_budget():
    async def scenario():
        model = StatusModel()
        budget = ModelBudget(EvalLimits(max_model_calls=1))
        runtime = await fixture(BudgetedModel(model, budget))
        runtime.index.model = None
        with pytest.raises(EvaluationBudgetExceeded):
            await runtime.agents["report_explainer"][0].run(QUESTION, Intent.REPORT, {}, "selective", original_request=QUESTION)
        assert budget.calls == 1 and len(model.calls) == 1 and not model.calls[0]["repair"]
    run(scenario())


@pytest.mark.parametrize("outcome", ["persistent", "tool", "repair_unknown_reference"])
def test_status_retry_failure_still_falls_back_and_does_not_leak_drafts(outcome):
    async def scenario():
        model = StatusModel(outcome)
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text=QUESTION))
        assert reply.degraded and not reply.agent_results
        assert [call["repair"] for call in model.calls] == [False, True, False, True]
        assert all(not call["tools"] for call in model.calls if call["repair"])
        assert SECRET not in reply.answer.text and SECRET not in json.dumps(runtime.monitor.traces[-1], ensure_ascii=False)
    run(scenario())
