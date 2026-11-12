"""Mixed report questions retain incomplete personal fields and only quote reviewed evidence."""
import pytest

from medagent.core.contracts import Intent, Question
from medagent.agents.report_sources import mixed_personal_report_without_item, personal_report_without_item
from support.harness import Model, fixture, run
from support.multiturn import ContextModel
from support.subject_switch import CountStages


QUESTION = "体检报告一个指标偏高，是不是一定有病？参考范围有什么意义？"
ITEM = {"name":"一个指标", "value":None, "unit":None, "reference_range":None}


@pytest.mark.parametrize("text,expected", [
    (QUESTION, True),
    ("化验结果偏低，参考范围是什么意思？", True),
    ("参考范围是什么意思？", False),
    ("如果体检报告一个指标偏高，参考范围有什么意义？", False),
    ("体检报告没有指标异常，参考范围有什么意义？", False),
    ("体检报告一个指标偏高，不需要分析个人结果，只想问参考范围是什么意思？", False),
    ("体检报告一个指标偏高，不用分析这份报告，只问参考范围是什么？", False),
    ("体检报告一个指标偏高，不需要解释参考范围是什么。", False),
    ("体检报告一个指标偏高，只想整理记录。", False),
])
def test_missing_item_fallback_is_limited_to_explicit_mixed_questions(text,expected):
    assert mixed_personal_report_without_item(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("体检报告里有一个指标偏高，但项目名称和数值不清楚，可以判断我是否生病了吗？", True),
    ("化验报告一个项目偏低，这意味着什么？", True),
    ("如果体检报告一个指标偏高，是不是有病？", False),
    ("体检报告没有指标异常，是不是有病？", False),
    ("体检报告有一个指标偏高，不需要分析个人结果，只讲术语。", False),
    ("体检报告一个指标偏高，只想整理记录。", False),
    ("参考范围是什么意思？", False),
])
def test_personal_result_without_item_fallback_respects_request_boundaries(text,expected):
    assert personal_report_without_item(text) is expected


def test_unextracted_personal_abnormal_result_requires_clarification_without_range_question():
    async def scenario():
        runtime = await fixture(Model())
        text = "体检报告里有一个指标偏高，但项目名称和数值不清楚，可以判断我是否生病了吗？"
        reply = await runtime.chat(Question(text=text))
        result = next(row for row in reply.agent_results if row.role == "report_explainer")
        assert not reply.degraded and result.status == "needs_clarification"
        assert "report_item" in result.missing_fields and result.payload["report_items"] == []
        assert any(event.get("reason") == "personal_report_without_item"
                   for event in runtime.monitor.traces[-1]["events"] if event["event"] == "report_field_fallback")
    run(scenario())


@pytest.mark.parametrize("mode", ["selective", "full"])
def test_mixed_question_quotes_reviewed_source_and_keeps_personal_gaps(mode):
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        result = await runtime.agents["report_explainer"][0].run(
            QUESTION, Intent.REPORT, {}, mode, details={"items":[ITEM]}, original_request=QUESTION)
        assert result.execution.status == "needs_clarification"
        assert result.execution.missing_fields == ["value", "unit", "reference_range"]
        assert result.execution.payload["report_items"] == [ITEM]
        ref = runtime.index.documents["lab-context"]
        assert result.execution.payload["term_explanations"] == [{"text":ref.text, "reference_keys":[ref.key]}]
        assert ref.text in result.answer.text and result.answer.reference_keys == [ref.key]
        assert [x.key for x in result.references] == [ref.key]
        assert "agent_report_explainer" not in model.stages
        assert len(result.answer.questions) == 3

    run(scenario())


@pytest.mark.parametrize("change", ["absent", "unapproved", "text", "url"])
def test_mixed_question_does_not_quote_missing_or_modified_source(change):
    async def scenario():
        runtime = await fixture(CountStages())
        ref = runtime.index.documents["lab-context"]
        if change == "absent":
            del runtime.index.documents[ref.key]
        else:
            patch = {"unapproved":{"approved":False}, "text":{"text":ref.text+"变更"},
                     "url":{"url":"https://example.org/altered"}}[change]
            runtime.index.documents[ref.key] = ref.model_copy(update=patch)
        runtime.index.cache.clear()
        result = await runtime.agents["report_explainer"][0].run(
            QUESTION, Intent.REPORT, {}, "selective", details={"items":[ITEM]}, original_request=QUESTION)
        assert result.execution.status == "needs_clarification"
        assert result.execution.missing_fields == ["value", "unit", "reference_range", "source"]
        assert result.execution.payload["report_items"] == [ITEM]
        assert result.execution.payload["term_explanations"] == []
        assert result.references == [] and result.answer.reference_keys == []

    run(scenario())


@pytest.mark.parametrize("question", ["报告一个指标偏高，帮我整理。", "报告一个指标偏高，不需要解释参考范围是什么。"])
def test_personal_only_or_declined_general_question_does_not_trigger_lookup(question):
    async def scenario():
        runtime = await fixture(CountStages())
        result = await runtime.agents["report_explainer"][0].run(
            question, Intent.REPORT, {}, "selective", details={"items":[ITEM]}, original_request=question)
        assert result.execution.missing_fields == ["value", "unit", "reference_range"]
        assert result.references == [] and result.execution.payload["term_explanations"] == []
        assert runtime.monitor.snapshot()["tool_stats"]["lookup_reference"]["total"] == 0

    run(scenario())


def test_original_question_runs_through_chat_and_keeps_current_subject():
    def updates(_payload):
        return [{"name":"report_item", "value":"一个指标", "state":"provided", "quote":"一个指标"}]

    async def scenario():
        runtime = await fixture(ContextModel(updates))
        reply = await runtime.chat(Question(text=QUESTION, subject="mother"))
        result = next(row for row in reply.agent_results if row.role == "report_explainer")
        assert reply.subject == "mother" and result.status == "needs_clarification"
        assert result.payload["report_items"] == [ITEM]
        assert result.missing_fields == ["value", "unit", "reference_range"]
        assert reply.answer.reference_keys == ["lab-context"]
        assert {ref.key for ref in reply.references} == {"lab-context"}

    run(scenario())


@pytest.mark.parametrize("subject", ["self", "mother"])
def test_original_question_without_extracted_item_preserves_structured_clarification(subject):
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text=QUESTION, subject=subject))
        result = next(row for row in reply.agent_results if row.role == "report_explainer")
        assert "report_item" not in reply.intent.entities
        assert reply.subject == subject and result.status == "needs_clarification"
        assert result.missing_fields == ["report_item", "value", "unit", "reference_range"]
        assert result.payload["report_items"] == []
        ref = runtime.index.documents["lab-context"]
        assert result.payload["term_explanations"] == [{"text":ref.text, "reference_keys":[ref.key]}]
        assert reply.answer.reference_keys == [ref.key] and "统计区间" not in reply.answer.text
        assert "agent_report_explainer" not in model.stages
        trace = runtime.monitor.traces[-1]
        assert any(e["event"] == "report_field_fallback" and e["reason"] == "mixed_personal_report_without_item"
                   for e in trace["events"])
        assert QUESTION not in str(trace)

    run(scenario())


def test_mixed_question_without_extracted_item_and_without_source_keeps_all_gaps():
    async def scenario():
        runtime = await fixture(CountStages())
        del runtime.index.documents["lab-context"]
        result = await runtime.agents["report_explainer"][0].run(
            QUESTION, Intent.REPORT, {}, "selective", original_request=QUESTION)
        assert result.execution.status == "needs_clarification"
        assert result.execution.missing_fields == ["report_item", "value", "unit", "reference_range", "source"]
        assert result.execution.payload["report_items"] == []
        assert result.execution.payload["term_explanations"] == []
        assert result.references == [] and result.answer.reference_keys == []

    run(scenario())


def test_rejected_fabricated_item_does_not_bypass_structured_clarification():
    def invented(_payload):
        return [{"name":"report_item", "value":"虚构项目", "state":"provided", "quote":"虚构项目"}]

    async def scenario():
        runtime = await fixture(ContextModel(invented))
        reply = await runtime.chat(Question(text=QUESTION))
        result = next(row for row in reply.agent_results if row.role == "report_explainer")
        assert "report_item" not in reply.intent.entities
        assert result.status == "needs_clarification" and "report_item" in result.missing_fields
        assert result.payload["report_items"] == [] and "虚构项目" not in reply.answer.text
        assert reply.answer.reference_keys == ["lab-context"]
        assert any(event["event"] == "entity_rejected" for event in runtime.monitor.traces[-1]["events"])

    run(scenario())


def test_generic_question_without_personal_report_remains_answerable():
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text="报告里的参考范围有什么意义？"))
        result = next(row for row in reply.agent_results if row.role == "report_explainer")
        assert result.status == "answered" and not result.missing_fields
        assert "agent_report_explainer" in model.stages
        assert not any(e["event"] == "report_field_fallback" for e in runtime.monitor.traces[-1]["events"])

    run(scenario())
