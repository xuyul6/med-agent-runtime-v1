"""A conservative current-message gate for generic visit evidence."""
import pytest

from medagent.agents.contracts import VisitInput
from medagent.core.contracts import Intent, Question, Reference
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.knowledge.retrieval import ReferenceIndex, Vectorizer, has_query_overlap
from medagent.understanding.conversation import visit_query_has_context
from medagent.tools.registry import ToolBudget, ToolContext, ToolResultError
from support.harness import Model, fixture, run


def template():
    return Reference(key="synthetic-general", title="就医准备清单", text="就医前的资料、证件和准备清单",
                     domain=Intent.VISIT, url="https://example.org/template", version="test", scope="合成测试",
                     approved=True, visit_scope={"purpose":"general_preparation"})


@pytest.mark.parametrize("text", [
    "就医前需要准备哪些资料？", "看医生之前整理什么？", "看病前要准备什么？",
    "门诊前的准备清单", "体检要准备哪些资料", "复诊时整理什么",
    "不查这家医院，只整理通用准备清单", "去社区卫生服务中心前要准备什么",
])
def test_explicit_visit_context_is_recognized(text):
    assert visit_query_has_context(text)


@pytest.mark.parametrize("text", [
    "火星旅行证件", "出国旅游需要准备哪些证件", "登机准备清单",
    "入职前需要准备哪些资料", "搬家要整理什么", "看病历", "看病房", "这个需要准备哪些资料",
])
def test_ambiguous_or_nonmedical_context_has_no_visit_anchor(text):
    assert not visit_query_has_context(text)


@pytest.mark.parametrize("query", ["火星旅行证件", "出国旅游需要准备哪些证件", "登机准备清单"])
def test_unrelated_query_cannot_match_generic_visit_source(query):
    assert not has_query_overlap(query, template())


def test_original_query_must_supply_context_even_when_rewrite_and_cache_do():
    class Rewrite(Model):
        calls = 0

        async def json(self, *args, **kwargs):
            self.calls += 1
            return {"query":"就医准备清单"}

    async def scenario():
        model = Rewrite()
        index = ReferenceIndex(Vectorizer("hash"), model)
        await index.upsert([template()])
        assert [ref.key for ref in await index.search("就医准备清单", "visit")] == ["synthetic-general"]
        calls = model.calls
        assert await index.search("就医准备清单", "visit", evidence_query="出国旅游准备证件") == []
        assert model.calls == calls
        assert [ref.key for ref in await index.search("就医准备清单", "visit")] == ["synthetic-general"]

    run(scenario())


def test_forged_tool_result_cannot_bypass_visit_context_gate():
    class ForgedIndex:
        async def search(self, query, domain): return [template()]

    async def scenario():
        runtime = await fixture()
        context = ToolContext(DEFAULT_REGISTRY["visit_guide"], VisitInput(text="就医准备", intent=Intent.VISIT),
                              ForgedIndex(), evidence_query="出国旅游需要准备哪些证件？")
        with pytest.raises(ToolResultError, match="visit_context_missing"):
            await runtime.tools.call("lookup_reference", {"query":"就医准备", "domain":"visit"}, context, ToolBudget())
        assert not context.references
        assert runtime.monitor.snapshot()["tool_stats"]["lookup_reference"]["quality_counts"] == {"visit_context_missing":1}

    run(scenario())


def test_full_context_does_not_publish_visit_source_for_unrelated_query():
    class NoAnswer(Model):
        async def message(self, *args, **kwargs):
            raise AssertionError("unrelated_request_reached_answer_model")

    async def scenario():
        runtime = await fixture(NoAnswer())
        result = await runtime.agents["visit_guide"][0].run(
            "就医准备", Intent.VISIT, {}, "full", original_request="出国旅游需要准备哪些证件？")
        assert result.execution.status == "needs_clarification"
        assert result.execution.missing_fields == ["visit_purpose"]
        assert not result.references and not result.answer.reference_keys

    run(scenario())


def test_recent_same_subject_followup_can_use_previous_user_context():
    async def scenario():
        runtime = await fixture(Model())
        first = await runtime.chat(Question(text="就医前需要准备哪些资料？"))
        assert first.primary_agent == "visit_guide" and first.agent_results[0].status == "answered"
        second = await runtime.chat(Question(text="接下来准备哪些资料？", session=first.session, subject=first.subject))
        visit = next(result for result in second.agent_results if result.role == "visit_guide")
        assert visit.status == "answered"
        assert [ref.key for ref in second.references] == ["consultation-record"]
        assert any(event["event"] == "visit_context_recovered" for event in runtime.monitor.traces[-1]["events"])

    run(scenario())
