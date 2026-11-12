"""Generic report questions go through the report model; no sentence-level shortcut answers them."""

import pytest

from medagent.benchmark.contracts import BenchmarkRequest, resolve_benchmark
from medagent.benchmark.runner import run_benchmark
from medagent.core.contracts import Intent, Question
from support.harness import fixture, run
from support.report_range import ReportRepairModel
from support.subject_switch import CountStages


@pytest.mark.parametrize("text", [
    # These were once answered by whole-sentence patterns copied from the benchmark; they now reach the model.
    "请解释化验报告参考范围的含义，以及为什么单个结果不能独立确定健康状态。",
    "我还看到化验报告上列有参考范围。",
    "我知道化验报告会列出项目名称。",
    "参考范围是什么意思？",
])
def test_generic_report_questions_reach_the_report_model(text):
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        result = await runtime.agents["report_explainer"][0].run(text, Intent.REPORT, {}, "selective", original_request=text)
        assert "agent_report_explainer" in model.stages
        assert result.model_dependency_exercised
    run(scenario())


def test_personal_report_without_item_still_asks_for_the_item():
    async def scenario():
        runtime = await fixture(CountStages())
        reply = await runtime.chat(Question(text="体检报告一个指标偏高，是不是一定有病？参考范围有什么意义？"))
        result = next(row for row in reply.agent_results if row.role == "report_explainer")
        assert result.status == "needs_clarification" and "report_item" in result.missing_fields
    run(scenario())


def test_persistent_invalid_model_in_the_report_scenario_remains_failed_not_hidden(monkeypatch):
    monkeypatch.setattr("medagent.agents.checks.insert_range_sentences", lambda text, keys, references: (text, keys, 0))
    async def scenario():
        _,cases = resolve_benchmark(BenchmarkRequest())
        case = next(item for item in cases if item.id == "report-history")
        report = await run_benchmark(await fixture(ReportRepairModel(outcome="persistent")),
            BenchmarkRequest(cases=[case],repetitions=2,cache_conditions=["cold"],judge=False))
        assert not report["complete"] and report["not_run"]
        assert any(not turn["rule_passed"] for turn in report["preparation"][0]["turns"])
        assert report["quality_noninferiority"] == "not_established"
    run(scenario())
