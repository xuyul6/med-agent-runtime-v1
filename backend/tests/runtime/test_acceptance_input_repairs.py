"""Regressions for synthetic cases found during real-model acceptance."""
import pytest

from medagent.core.contracts import Question
from medagent.understanding.conversation import ground_update
from support.harness import fixture, run
from support.multiturn import ContextModel, ask


@pytest.mark.parametrize("value", ["人群", "适用人群", "哪些人", "年龄段"])
def test_population_question_is_not_a_specific_population(value):
    source = {"turn_id":"current", "turn_index":1, "subject":"self",
              "text":f"药品公开资料适用于{value}？", "origin":"current_user"}
    update = {"turn_id":"current", "name":"population", "value":value,
              "state":"topic", "quote":source["text"]}
    assert ground_update(update, {"current":source}) is None


def test_specific_population_question_can_still_be_a_query_scope():
    source = {"turn_id":"current", "turn_index":1, "subject":"self",
              "text":"这个药的资料适用于孕妇吗？", "origin":"current_user"}
    update = {"turn_id":"current", "name":"population", "value":"孕妇",
              "state":"provided", "quote":source["text"]}
    fact = ground_update(update, {"current":source})
    assert fact is not None and fact.value == "孕妇" and fact.state == "topic"


def test_generic_population_followup_keeps_the_prior_query_scope():
    def updates(payload):
        if "哪些人群" not in payload["current"]["text"]:
            return []
        return [{"name":"population", "value":"人群", "state":"topic", "quote":"人群"}]

    async def scenario():
        runtime = await fixture(ContextModel(updates))
        first = await ask(runtime, "查询成人布洛芬公开药品资料。")
        second = await ask(runtime, "这个药的公开资料适用于哪些人群？")
        assert second.intent.entities["population"].value == "成人"
        assert second.intent.entities["population"].source.turn_id == first.request_id
        assert [ref.key for ref in second.references] == ["ibuprofen-overview"]
        assert not second.degraded

    run(scenario())


def test_explicit_unknown_report_fields_return_a_clarification():
    async def scenario():
        runtime = await fixture(ContextModel())
        reply = await runtime.chat(Question(text="报告项目名、数值、单位和参考范围都没记住，先告诉我需要补充什么。"))
        result = next(row for row in reply.agent_results if row.role == "report_explainer")
        assert not reply.degraded and result.status == "needs_clarification"
        assert result.missing_fields == ["report_item", "value", "unit", "reference_range"]
        assert result.payload["report_items"] == [] and len(reply.answer.questions) == 4
        assert any(event.get("reason") == "explicit_unknown_report_fields"
                   for event in runtime.monitor.traces[-1]["events"])

    run(scenario())
