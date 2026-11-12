"""A synthetic reconstruction, not the unavailable rejected model drafts."""
import json

import pytest

from medagent.benchmark.contracts import BenchmarkRequest, resolve_benchmark
from medagent.benchmark.runner import run_benchmark
from medagent.core.contracts import Intent, Question
from medagent.evaluation.budget import BudgetedModel, EvaluationBudgetExceeded, ModelBudget
from medagent.evaluation.contracts import EvalLimits
from support.harness import fixture, run
from support.report_range import ReportRepairModel, SECRET, UNSUPPORTED


@pytest.fixture(autouse=True)
def no_insertion(monkeypatch):
    """The repair path below is reached when no source sentence can be inserted."""
    monkeypatch.setattr("medagent.agents.checks.insert_range_sentences", lambda text, keys, references: (text, keys, 0))


def test_range_error_can_be_corrected_once_with_full_validation_and_no_extra_tools():
    async def scenario():
        model = ReportRepairModel()
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text="报告上的参考范围与其他报告信息有什么联系？"))
        assert reply.primary_agent == "report_explainer" and not reply.degraded
        assert len(model.calls) == 2 and [call["repair"] for call in model.calls] == [False,True]
        assert model.calls[0]["tools"] and not model.calls[1]["tools"]
        assert model.calls[1]["feedback"]["validation"]["code"] == "report_range_claim_without_exact_source"
        assert model.calls[1]["feedback"]["repair_strategy"] == "report_range_evidence_only"
        assert reply.references and reply.agent_results[0].status == "answered"
        assert UNSUPPORTED not in reply.answer.text and SECRET not in reply.answer.text
        events = runtime.monitor.traces[-1]["events"]
        validation = [event for event in events if event["event"] == "agent_output_validation"]
        assert len(validation) == 1 and validation[0]["repair_requested"]
        assert validation[0]["repair_strategy"] == "report_range_evidence_only"
        assert any(event["event"] == "agent_output_repair" and event["status"] == "accepted" for event in events)
        assert SECRET not in json.dumps(events,ensure_ascii=False)
    run(scenario())


@pytest.mark.parametrize("outcome", ["persistent", "boundary", "tool", "unknown_reference"])
def test_failed_repair_preserves_validation_and_same_role_fallback(outcome):
    async def scenario():
        model = ReportRepairModel(outcome)
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text="报告上的参考范围与其他报告信息有什么联系？"))
        assert reply.degraded and not reply.references and not reply.agent_results
        assert len(model.calls) == 4 and [call["repair"] for call in model.calls] == [False,True,False,True]
        assert all(not call["tools"] for call in model.calls if call["repair"])
        assert SECRET not in reply.answer.text and UNSUPPORTED not in reply.answer.text
        events = runtime.monitor.traces[-1]["events"]
        assert SECRET not in json.dumps(events,ensure_ascii=False)
        assert not any(event["event"] == "agent_output_repair" and event["status"] == "accepted" for event in events)
        if outcome == "tool":
            assert any(event.get("code") == "unexpected_tools_during_repair" for event in events)
    run(scenario())


def test_original_report_preparation_reaches_targets_with_synthetic_repair():
    async def scenario():
        _,cases = resolve_benchmark(BenchmarkRequest())
        original = next(case for case in cases if case.id == "report-history")
        request = BenchmarkRequest(cases=[original], repetitions=2, cache_conditions=["cold"], judge=False)
        model = ReportRepairModel()
        report = await run_benchmark(await fixture(model),request)
        assert report["complete"] and not report["not_run"]
        preparation = report["preparation"][0]
        assert len(preparation["turns"]) == 5 and all(row["rule_passed"] for row in preparation["turns"])
        fourth = preparation["turns"][3]
        assert fourth["message"] == "我还看到化验报告上列有参考范围。"
        assert not any(event["event"] == "report_source_direct" for event in fourth["trace"]["events"])
        assert all(arm["status"] == "completed" and arm["rule_passed"] for pair in report["pairs"] for arm in pair["arms"].values())
        assert not report["cleanup_errors"] and report["quality_noninferiority"] == "not_established"
    run(scenario())


def test_correction_respects_the_existing_model_call_budget():
    async def scenario():
        model = ReportRepairModel()
        budget = ModelBudget(EvalLimits(max_model_calls=1))
        runtime = await fixture(BudgetedModel(model,budget))
        runtime.index.model = None
        agent = runtime.agents["report_explainer"][0]
        with pytest.raises(EvaluationBudgetExceeded):
            await agent.run("报告上的参考范围与其他报告信息有什么联系？",Intent.REPORT,{},"selective")
        assert budget.calls == 1 and len(model.calls) == 1 and not model.calls[0]["repair"]
    run(scenario())


def test_report_skill_scopes_unrequested_range_definition_in_actual_model_context():
    class Capture(ReportRepairModel):
        def __init__(self):
            super().__init__()
            self.systems = []

        async def message(self, stage, system, messages, tools=None, budget=1800):
            if stage == "agent_report_explainer":
                self.systems.append(system)
            return await super().message(stage, system, messages, tools, budget)

    async def scenario():
        model = Capture()
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text="报告里的项目名称需要如何结合其他报告信息理解？"))
        assert not reply.degraded and model.systems
        assert "不主动扩展参考范围的统计性质或区间定义" in model.systems[0]
        assert "不得省略完成当前问题所需的限制或安全提示" in model.systems[0]
        assert runtime.skills.snapshot().documents["report_explainer"].version == "1.7.0"
    run(scenario())
