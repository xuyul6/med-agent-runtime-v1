"""Narrow report topic regressions using entirely synthetic reference text."""
import json

import pytest

from medagent.agents.contracts import ReportInput
from medagent.core.contracts import Intent, Question, Reference
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.knowledge.retrieval import ReferenceIndex, Vectorizer, has_query_overlap
from medagent.tools.registry import ToolBudget, ToolContext, ToolResultError
from support.harness import Model, fixture, role_output, run, role_data


def report(key, title, text):
    return Reference(key=key, title=title, text=text, domain=Intent.REPORT,
                     url="https://example.org/"+key, version="synthetic", scope="test only", approved=True)


@pytest.mark.parametrize("query", ["报告怎么理解？", "帮我看看我的报告", "解释检查报告"])
def test_generic_report_question_can_still_use_generic_report_reference(query):
    assert has_query_overlap(query, report("general", "报告阅读", "报告信息的通用整理"))


@pytest.mark.parametrize("query", ["影像报告术语", "血糖报告项目", "肝功能报告怎么理解"])
def test_domain_word_alone_does_not_make_a_different_topic_eligible(query):
    assert not has_query_overlap(query, report("unrelated", "报告阅读", "仅提供报告的一般记录示例"))


def test_mixed_report_topics_keep_both_topic_matches():
    async def scenario():
        index = ReferenceIndex(Vectorizer("hash"), None)
        await index.upsert([
            report("imaging", "影像报告词语", "影像记录"),
            report("glucose", "血糖报告记录", "血糖项目"),
            report("kidney", "肾功能报告", "肾功能记录"),
        ])
        result = await index.search("影像报告和血糖报告", "report")
        assert {ref.key for ref in result} == {"imaging", "glucose"}

    run(scenario())


def test_reference_interval_alias_is_not_expanded_inside_another_word():
    ref = report("range", "参考范围导读", "参考范围信息")
    assert has_query_overlap("REFERENCE interval", ref)
    assert not has_query_overlap("noreference interval", ref)


def test_specific_report_without_matching_source_reports_gap():
    async def scenario():
        runtime = await fixture(Model())
        reply = await runtime.chat(Question(text="肾功能报告怎么看？"))
        report_result = next(result for result in reply.agent_results if result.role == "report_explainer")
        assert report_result.status == "needs_clarification"
        assert "source" in report_result.missing_fields
        assert not reply.references

    run(scenario())


def test_tool_rejects_unrelated_topic_even_when_report_word_matches():
    class ForgedIndex:
        async def search(self, query, domain):
            return [report("other", "血糖报告", "血糖记录")]

    async def scenario():
        runtime = await fixture()
        context = ToolContext(DEFAULT_REGISTRY["report_explainer"],
            ReportInput(text="整理报告", intent=Intent.REPORT), ForgedIndex(), evidence_query="影像报告术语")
        with pytest.raises(ToolResultError, match="reference_without_query_overlap"):
            await runtime.tools.call("lookup_reference", {"query":"血糖报告", "domain":"report"}, context, ToolBudget())
        assert not context.references

    run(scenario())


def test_full_context_filters_unrelated_report_topic():
    class CapturingModel(Model):
        async def message(self, stage, system, messages, tools=None, budget=1800):
            if stage == "agent_report_explainer":
                self.context = role_data(messages)
                return {"content":[{"type":"text", "text":json.dumps(role_output("report_explainer", ["synthetic-imaging"]))}]}
            return await super().message(stage, system, messages, tools, budget)

    async def scenario():
        model = CapturingModel()
        runtime = await fixture(model)
        await runtime.index.upsert([
            report("synthetic-imaging", "影像报告术语", "影像记录"),
            report("synthetic-other", "血糖报告", "血糖记录"),
        ])
        await runtime.agents["report_explainer"][0].run(
            "整理影像报告术语", Intent.REPORT, {}, "full", original_request="影像报告术语")
        assert {ref["key"] for ref in model.context["references"]} == {"synthetic-imaging"}

    run(scenario())
