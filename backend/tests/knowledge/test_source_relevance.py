"""Deterministic source-selection regressions; no network or paid models."""
import json
import pytest

from medagent.core.contracts import FieldSource, GroundedField, Intent, Question, Reference, TaskUnderstanding, Understanding
from medagent.knowledge.retrieval import relevance_terms
from medagent.understanding.conversation import role_details
from medagent.agents.checks import checked_role_output
from medagent.agents.output_validation import AgentOutputError
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.agents.contracts import ReportInput
from medagent.tools.registry import ToolBudget, ToolContext, ToolResultError
from support.harness import Model, fixture, role_output, run, role_data


@pytest.mark.parametrize("domain", [Intent.REPORT, Intent.VISIT, Intent.SYMPTOMS])
def test_no_lexical_overlap_does_not_return_same_domain_sources(domain):
    async def scenario():
        runtime = await fixture()
        # No Chroma collection: this checks lexical recall without constraining a semantic match.
        assert runtime.index.collection is None
        assert await runtime.index.search("zzzzsyntheticirrelevant", domain.value) == []
    run(scenario())


@pytest.mark.parametrize("domain,query,expected", [
    (Intent.REPORT, "报告参考范围是什么", "lab-context"),
    (Intent.VISIT, "就医前需要准备什么", "consultation-record"),
    (Intent.SYMPTOMS, "症状记录需要哪些信息", "symptom-record"),
])
def test_relevant_generic_queries_still_retrieve_own_domain(domain, query, expected):
    async def scenario():
        runtime = await fixture()
        assert expected in [ref.key for ref in await runtime.index.search(query, domain.value)]
    run(scenario())


@pytest.mark.parametrize("domain,query,expected", [
    (Intent.REPORT, "REFERENCE   RANGE", ["lab-reference-range-definition", "lab-context"]),
    (Intent.REPORT, "参考范围是什么？", ["lab-reference-range-definition", "lab-context"]),
    (Intent.REPORT, "结婚证", []),
    (Intent.REPORT, "结婚证件补办", []),
    (Intent.VISIT, "看医生前需要准备什么？", ["consultation-record"]),
    (Intent.VISIT, "看病前需要整理什么？", ["consultation-record"]),
    (Intent.VISIT, "火星旅行签证", []),
    (Intent.VISIT, "宠物航空运输", []),
])
def test_bounded_relevance_gate_handles_cases_without_single_character_matches(domain, query, expected):
    async def scenario():
        runtime = await fixture()
        assert [ref.key for ref in await runtime.index.search(query, domain.value)] == expected
    run(scenario())


@pytest.mark.parametrize("text", ["看病历", "看病理报告", "看病房", "noreference range"])
def test_relevance_aliases_do_not_expand_inside_different_words(text):
    normalized = relevance_terms(text)
    assert "就医" not in normalized and "参考" not in normalized


def test_original_single_character_tokenizer_for_other_uses_is_unchanged():
    from medagent.knowledge.retrieval import terms
    assert terms("结婚证") == {"结", "婚", "证"}


def test_model_rewrite_cannot_add_an_unrelated_source():
    class PoisonRewrite:
        async def json(self, stage, system, content, budget=900):
            assert stage == "retrieval_rewrite"
            return {"query":"报告参考范围"}
    async def scenario():
        runtime = await fixture()
        runtime.index.model = PoisonRewrite()
        assert await runtime.index.search("zzzzsyntheticirrelevant", "report") == []
    run(scenario())


def test_cache_keeps_the_original_evidence_query_separate():
    async def scenario():
        runtime = await fixture()
        result = await runtime.index.search("报告参考范围", "report", evidence_query="参考范围")
        assert {ref.key for ref in result} == {"lab-context", "lab-reference-range-definition"}
        assert await runtime.index.search("报告参考范围", "report", evidence_query="zzzzsyntheticirrelevant") == []
    run(scenario())


@pytest.mark.parametrize("mode", ["selective", "full"])
def test_role_instruction_text_cannot_make_unrelated_user_request_match(mode):
    async def scenario():
        model = Model()
        runtime = await fixture(model)
        result = await runtime.agents["report_explainer"][0].run(
            "仅整理报告文字与信息缺口。用户相关原文：zzzzsyntheticirrelevant",
            Intent.REPORT, {}, mode, original_request="zzzzsyntheticirrelevant")
        assert result.execution.missing_fields == ["source"]
        assert not result.references and not result.answer.reference_keys
    run(scenario())


def test_semantic_candidate_cannot_bypass_original_query_constraint():
    class SyntheticCollection:
        def query(self, **kwargs): return {"ids":[["lab-context"]]}
    async def scenario():
        runtime = await fixture()
        runtime.index.collection = SyntheticCollection()
        assert await runtime.index.search("zzzzsyntheticirrelevant", "report") == []
    run(scenario())


def test_untrusted_tool_index_cannot_publish_zero_overlap_reference():
    class UnrelatedIndex:
        def __init__(self, ref): self.ref = ref
        async def search(self, query, domain): return [self.ref]
    async def scenario():
        runtime = await fixture()
        context = ToolContext(DEFAULT_REGISTRY["report_explainer"],
            ReportInput(text="仅整理报告文字。", intent=Intent.REPORT),
            UnrelatedIndex(runtime.index.documents["lab-context"]), evidence_query="zzzzsyntheticirrelevant")
        with pytest.raises(ToolResultError, match="reference_without_query_overlap"):
            await runtime.tools.call("lookup_reference", {"query":"报告参考范围", "domain":"report"}, context, ToolBudget())
        assert not context.references
    run(scenario())


def test_full_context_cannot_include_unrelated_approved_sources():
    class CapturingModel(Model):
        async def message(self, stage, system, messages, tools=None, budget=1800):
            if stage == "agent_report_explainer":
                self.context = role_data(messages)
            return await super().message(stage, system, messages, tools, budget)
    async def scenario():
        model = CapturingModel()
        runtime = await fixture(model)
        await runtime.index.upsert([Reference(key="synthetic-unrelated", title="unrelatedalpha", domain=Intent.REPORT,
            text="unrelatedbeta", url="https://example.org/synthetic", version="test", scope="test", approved=True)])
        result = await runtime.agents["report_explainer"][0].run("报告上的参考范围与其他报告信息有什么联系？", Intent.REPORT, {}, "full")
        assert result.execution.status == "answered"
        assert {ref["key"] for ref in model.context["references"]} == {"lab-context", "lab-reference-range-definition"}
    run(scenario())


def test_specific_institution_without_verified_institution_source_clarifies():
    class NoAnswer(Model):
        async def message(self, *args, **kwargs):
            raise AssertionError("unsupported_institution_question_reached_answer_model")
    async def scenario():
        runtime = await fixture(NoAnswer())
        result = await runtime.agents["visit_guide"][0].run(
            "虚构星海医院在虚构地区挂号怎么收费？", Intent.VISIT, {}, "selective",
            details={"institution":"虚构星海医院","region":"虚构地区","institution_specific":True})
        assert result.execution.status == "needs_clarification" and result.execution.missing_fields == ["source"]
        assert not result.references and not result.answer.reference_keys
        assert not result.execution.payload["institution_facts"]
    run(scenario())


def test_named_institution_appointment_query_is_not_treated_as_general_preparation():
    goal = "虚构星海医院的预约入口在哪里？"
    understood = Understanding.model_validate({"primary":"visit", "entities":{
        "institution":{"name":"institution", "value":"虚构星海医院", "state":"topic", "source":{
            "turn_id":"synthetic", "turn_index":1, "subject":"self", "quote":"虚构星海医院", "origin":"current_user"}}},
        "tasks":[{"intent":"visit", "goal":goal, "priority":"primary"}]})
    details = role_details(understood, "visit_guide")
    assert details["institution"] == "虚构星海医院" and details["institution_specific"] is True


@pytest.mark.parametrize("goal,expected", [
    ("虚构星海医院的预约入口在哪里？", True),
    ("医院收费金额是多少？", True),
    ("不查这家医院，只整理通用准备清单", False),
    ("去医院看医生前要准备哪些资料？", False),
])
def test_specific_detection_and_negation_are_separate_from_general_preparation(goal, expected):
    understood = Understanding(primary=Intent.VISIT,
        tasks=[TaskUnderstanding(intent=Intent.VISIT, goal=goal, priority="primary")])
    assert role_details(understood, "visit_guide").get("institution_specific", False) is expected


def test_named_institution_is_blocked_through_runtime_before_answer_model():
    class NoAnswer(Model):
        async def message(self, *args, **kwargs):
            raise AssertionError("institution_request_reached_answer_model")
    async def scenario():
        runtime = await fixture(NoAnswer())
        text = "虚构星海医院在虚构地区的预约入口在哪里？"
        def fact(name, value):
            return GroundedField(name=name, value=value, state="topic" if name == "institution" else "provided",
                source=FieldSource(turn_id="current", turn_index=1, subject="self", quote=value, origin="current_user"))
        understood = Understanding(primary=Intent.VISIT, confidence=1, scores={"visit":1}, turn_id="current",
            entities={"institution":fact("institution", "虚构星海医院"), "region":fact("region", "虚构地区")},
            tasks=[TaskUnderstanding(intent=Intent.VISIT, goal=text, priority="primary")])
        async def fixed(*args, **kwargs): return understood.model_copy(deep=True)
        runtime.router.understand = fixed
        reply = await runtime.chat(Question(text=text))
        assert reply.primary_agent == "visit_guide" and reply.agent_results[0].missing_fields == ["source"]
        assert not reply.references and not reply.answer.reference_keys
        assert "官方渠道" in " ".join(reply.answer.next_steps)
    run(scenario())


def test_institution_fact_cannot_rely_on_generic_preparation_template():
    async def scenario():
        runtime = await fixture()
        template = runtime.index.documents["consultation-record"]
        draft = role_output("visit_guide", [template.key])
        draft["payload"]["institution_facts"] = [{"text":"虚构医院需提前付费。", "reference_keys":[template.key]}]
        with pytest.raises(AgentOutputError, match="institution_source_not_supported") as error:
            checked_role_output(DEFAULT_REGISTRY["visit_guide"], draft, {template.key:template})
        assert error.value.diagnostic() == {"code":"institution_source_not_supported", "stage":"evidence",
                                             "fields":["institution_facts"], "repairable":False}
    run(scenario())


def test_critical_risk_still_precedes_retrieval_and_models():
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError("critical_path_touched_dependency:" + name)
    async def scenario():
        runtime = await fixture()
        runtime.model = runtime.index = runtime.memory = runtime.router = Forbidden()
        reply = await runtime.chat(Question(text="现在胸痛而且呼吸困难。"))
        assert reply.primary_agent == "risk_handoff"
        assert reply.storage == "skipped" and reply.usage["calls"] == 0
        assert not reply.handoff_connected
    run(scenario())
