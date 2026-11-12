"""Recognized institution requests cannot use unverified visit references."""
import pytest

from medagent.agents.contracts import VisitInput
from medagent.core.contracts import Intent, Question, Reference
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.knowledge.retrieval import ReferenceIndex, Vectorizer, has_query_overlap
from medagent.tools.registry import ToolBudget, ToolContext, ToolResultError
from support.harness import Model, fixture, run


def visit_ref(key, title, text):
    return Reference(key=key, title=title, domain=Intent.VISIT, text=text,
                     url="https://example.org/"+key, version="synthetic", scope="test only", approved=True,
                     visit_scope={"purpose":"general_preparation"})


@pytest.mark.parametrize("query", [
    "虚构青山医院的挂号入口在哪里？",
    "虚构青山医院的地址在哪？",
    "医院收费金额是多少？",
])
def test_approved_matching_institution_reference_is_not_verified_evidence(query):
    ref = visit_ref("unverified", "虚构青山医院挂号入口地址收费", "虚构占位资料")
    assert not has_query_overlap(query, ref)


def test_untrusted_rewrite_and_cached_general_query_cannot_enable_institution_reference():
    class Rewrite:
        calls = 0

        async def json(self, *args, **kwargs):
            self.calls += 1
            return {"query":"就医准备资料"}

    async def scenario():
        model = Rewrite()
        index = ReferenceIndex(Vectorizer("hash"), model)
        await index.upsert([visit_ref("general", "就医准备", "就医资料清单")])
        assert [ref.key for ref in await index.search("就医准备资料", "visit")] == ["general"]
        prior_calls = model.calls
        # A rewritten/tool query and a warm cache must not override the original request.
        assert await index.search("就医准备资料", "visit", evidence_query="虚构青山医院地址在哪？") == []
        assert model.calls == prior_calls
        assert [ref.key for ref in await index.search("就医准备资料", "visit")] == ["general"]

    run(scenario())


def test_fake_index_cannot_publish_a_matching_institution_document():
    class ForgedIndex:
        async def search(self, query, domain):
            return [visit_ref("unverified", "虚构青山医院挂号入口", "虚构占位资料")]

    async def scenario():
        runtime = await fixture()
        context = ToolContext(DEFAULT_REGISTRY["visit_guide"],
            VisitInput(text="就医准备", intent=Intent.VISIT), ForgedIndex(),
            evidence_query="虚构青山医院挂号入口在哪里？")
        with pytest.raises(ToolResultError, match="institution_source_not_supported"):
            await runtime.tools.call("lookup_reference", {"query":"就医准备", "domain":"visit"}, context, ToolBudget())
        assert not context.references
        stats = runtime.monitor.snapshot()["tool_stats"]["lookup_reference"]
        assert stats["quality_counts"] == {"institution_source_not_supported":1}

    run(scenario())


def test_general_preparation_and_explicit_negative_institution_remain_available():
    async def scenario():
        runtime = await fixture()
        for query in ["就医前要准备什么？", "不要查虚构青山医院地址，只整理就医准备。"]:
            assert [ref.key for ref in await runtime.index.search(query, "visit")] == ["consultation-record"]

    run(scenario())


def test_visit_role_without_preflight_flag_still_cannot_answer_institution_request():
    class NoAnswerModel(Model):
        async def message(self, *args, **kwargs):
            raise AssertionError("unverified_institution_reached_answer_model")

    async def scenario():
        runtime = await fixture(NoAnswerModel())
        result = await runtime.agents["visit_guide"][0].run(
            "就医准备", Intent.VISIT, {}, "full", original_request="虚构青山医院地址在哪里？")
        assert result.execution.status == "needs_clarification"
        assert result.execution.missing_fields == ["source"]
        assert not result.references and not result.answer.reference_keys

    run(scenario())


def test_critical_request_still_takes_risk_path_first():
    async def scenario():
        runtime = await fixture(Model())
        reply = await runtime.chat(Question(text="我现在呼吸困难，虚构青山医院地址在哪？"))
        assert reply.primary_agent == "risk_handoff"
        assert reply.usage["calls"] == 0

    run(scenario())
