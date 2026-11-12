"""Narrow evidence checks for generated reference-range definition claims."""
import json

import pytest

from medagent.core.contracts import Intent, Question, Reference
from medagent.agents.checks import checked_role_output, check_report_range_claims
from medagent.agents.output_validation import AgentOutputError
from medagent.agents.profiles import DEFAULT_REGISTRY
from support.harness import Model, fixture, role_output, run


def source(text, key="synthetic-report"):
    return Reference(key=key, title="参考范围资料", text=text, domain=Intent.REPORT,
                     url="https://example.org/report", version="test", scope="合成数据", approved=True)


def output(text, term, key="synthetic-report"):
    data = role_output("report_explainer", [key])
    data["answer"]["text"] = text
    data["payload"]["term_explanations"] = [{"text":term, "reference_keys":[key]}]
    return data


@pytest.mark.parametrize("keys,cited_text,other_text,claim,expected", [
    ([], "报告说明。", None, "参考范围是统计区间。", "no_sources_cited"),
    (["synthetic-report"], "报告说明。", "参考范围是统计区间。", "参考范围是统计区间。", "exact_sentence_in_uncited_source"),
    (["synthetic-report"], "报告说明。", None, "参考范围是统计区间。", "no_range_sentence_in_cited_sources"),
    (["synthetic-report"], "参考范围是人群统计区间。", None, "参考范围是统计区间。", "not_exact_cited_sentence"),
])
def test_range_rejection_reason_is_bounded_and_contains_no_draft(keys,cited_text,other_text,claim,expected):
    cited = source(cited_text)
    refs = {cited.key:cited}
    if other_text:
        other = source(other_text, "other-report")
        refs[other.key] = other
    with pytest.raises(AgentOutputError) as caught:
        check_report_range_claims(claim, keys, refs, "text")
    diagnostic = caught.value.diagnostic()
    assert diagnostic["evidence_reason"] == expected
    assert claim not in json.dumps(diagnostic,ensure_ascii=False)
    assert set(diagnostic) == {"code", "stage", "fields", "repairable", "evidence_reason"}


def test_range_diagnostic_rejects_arbitrary_reason_text():
    with pytest.raises(ValueError, match="unknown_evidence_reason"):
        AgentOutputError("report_range_claim_without_exact_source", evidence_reason="user or model text")


@pytest.mark.parametrize("answer,term", [
    ("参考范围是统计得到的一般区间。", "资料仅提示需要核对原报告。"),
    ("请按原报告核对。", "参考范围是依据人群和方法得出的一般区间。"),
])
def test_generated_range_definition_not_supported_by_cited_source_is_rejected(answer,term):
    ref = source("参考范围可能因检测方法和人群不同而变化。")
    with pytest.raises(AgentOutputError,match="report_range_claim_without_exact_source") as error:
        checked_role_output(DEFAULT_REGISTRY["report_explainer"],output(answer,term),{ref.key:ref})
    assert error.value.stage == "evidence" and not error.value.repairable


def test_exact_source_sentence_is_accepted_without_claiming_semantic_proof():
    ref = source("参考范围是依据特定人群统计得到的一般区间。")
    text = "参考范围是依据特定人群统计得到的一般区间。"
    result = checked_role_output(DEFAULT_REGISTRY["report_explainer"],output(text,text),{ref.key:ref})
    assert result.answer.text == text and result.payload.term_explanations[0].text == text


@pytest.mark.parametrize("wrapper", ["- {}", "1. {}", "• {}", "“{}”", '"{}"'])
def test_exact_source_sentence_with_presentation_wrappers_is_accepted(wrapper):
    ref = source("参考范围是依据特定人群统计得到的一般区间。")
    text = wrapper.format("参考范围是依据特定人群统计得到的一般区间") + "。"
    result = checked_role_output(DEFAULT_REGISTRY["report_explainer"], output(text,text), {ref.key:ref})
    assert result.answer.text == text


@pytest.mark.parametrize("text", [
    "- 不应认为参考范围是依据特定人群统计得到的一般区间。",
    "“参考范围是依据特定人群统计得到的一般区间”适合所有人。",
    "1. 参考范围是依据特定人群统计得到的一般区间，且人人适用。",
])
def test_changed_or_negated_claims_are_replaced_by_the_source_sentence(text):
    ref = source("参考范围是依据特定人群统计得到的一般区间。")
    result = checked_role_output(DEFAULT_REGISTRY["report_explainer"],output(text,text),{ref.key:ref})
    assert result.answer.text == ref.text == result.payload.term_explanations[0].text


def test_exact_quote_from_an_uncited_reference_gets_its_citation_from_the_code():
    cited = source("参考范围可能因检测方法和人群不同而变化。")
    other = source("参考范围是依据特定人群统计得到的一般区间。", "other-report")
    text = "参考范围是依据特定人群统计得到的一般区间。"
    result = checked_role_output(DEFAULT_REGISTRY["report_explainer"],output(text,text),{cited.key:cited, other.key:other})
    assert result.answer.text == text and other.key in result.answer.reference_keys
    assert other.key in result.payload.term_explanations[0].reference_keys


def test_inserted_text_is_what_the_source_says_even_when_the_source_negates_the_claim():
    ref = source("不应认为参考范围是统计区间。")
    text = "参考范围是统计区间。"
    result = checked_role_output(DEFAULT_REGISTRY["report_explainer"],output(text,text),{ref.key:ref})
    assert result.answer.text == ref.text


def test_extra_claim_next_to_an_exact_sentence_is_dropped():
    ref = source("参考范围是依据特定人群统计得到的一般区间。")
    exact = "参考范围是依据特定人群统计得到的一般区间。"
    text = exact+"参考范围是通过其他统计方法得到的区间。"
    result = checked_role_output(DEFAULT_REGISTRY["report_explainer"],output(text,exact),{ref.key:ref})
    assert result.answer.text == exact


def test_long_sentence_cannot_bypass_range_claim_check():
    ref = source("参考范围可能因检测方法和人群不同而变化。")
    text = "参考范围"+"说明"*80+"是统计区间。"
    with pytest.raises(AgentOutputError,match="report_range_claim_without_exact_source"):
        checked_role_output(DEFAULT_REGISTRY["report_explainer"],output(text,text),{ref.key:ref})


def test_unrelated_general_text_is_not_treated_as_range_definition():
    ref = source("参考范围可能因检测方法和人群不同而变化。")
    text = "参考范围可能因检测方法和人群不同而变化。"
    result = checked_role_output(DEFAULT_REGISTRY["report_explainer"],output(text,text),{ref.key:ref})
    assert result.answer.reference_keys == [ref.key]


def test_unsupported_generated_claim_fails_closed_without_leaking_model_draft(monkeypatch):
    # No insertable source sentence this turn, so the original fail-closed path applies.
    monkeypatch.setattr("medagent.agents.checks.report_range_source_quotes", lambda references: [])
    class Unsupported(Model):
        async def message(self,stage,system,messages,tools=None,budget=1800):
            if stage == "agent_report_explainer":
                data = output("参考范围是统计得到的一般区间。", "参考范围是统计得到的一般区间。", "lab-context")
                return {"content":[{"type":"text", "text":json.dumps(data,ensure_ascii=False)}]}
            return await super().message(stage,system,messages,tools,budget)

    async def scenario():
        runtime = await fixture(Unsupported())
        reply = await runtime.chat(Question(text="报告上的参考范围与其他报告信息有什么联系？"))
        assert reply.degraded and not reply.agent_results and not reply.references
        assert "统计得到" not in reply.answer.text
        attempts = reply.routing_decision.assignments[0].attempts
        assert len(attempts) == 2 and all(a.error_code == "report_range_claim_without_exact_source" for a in attempts)
        events = runtime.monitor.traces[-1]["events"]
        assert all("统计得到" not in json.dumps(e,ensure_ascii=False) for e in events)

    run(scenario())


def range_references():
    import json
    from medagent.core.config import ROOT
    from medagent.core.contracts import Reference
    return {row["key"]:Reference.model_validate(row) for row in json.loads((ROOT/"resources"/"references.json").read_text())
            if row["key"] in {"lab-reference-range-definition", "lab-context"}}


RANGE_SENTENCE = "MedlinePlus介绍，参考范围是被视为正常的检验结果上下限区间，其建立依据是大量健康人群的检验结果"


def test_code_replaces_a_model_written_range_definition_with_the_source_sentence():
    from medagent.agents.checks import check_report_range_claims, insert_range_sentences
    references = range_references()
    text, keys, count = insert_range_sentences("先说结论。参考范围是根据统计得出的区间。请带报告咨询医护。", ["lab-context"], references)
    assert text == "先说结论。"+RANGE_SENTENCE+"。请带报告咨询医护。" and count == 1
    assert keys == ["lab-context", "lab-reference-range-definition"]
    check_report_range_claims(text, keys, references, "text")


def test_range_insertion_keeps_one_copy_and_leaves_other_text_alone():
    from medagent.agents.checks import insert_range_sentences
    references = range_references()
    # An exact quote stays; a second paraphrase next to it goes.
    text, keys, count = insert_range_sentences(RANGE_SENTENCE+"。参考范围是统计区间。其余内容。", ["lab-reference-range-definition"], references)
    assert text == RANGE_SENTENCE+"。其余内容。" and count == 1
    # Two paraphrases become one sentence.
    text, _, count = insert_range_sentences("参考范围是统计区间。另一种说法：参考范围是一个区间。", [], references)
    assert text.count(RANGE_SENTENCE) == 1 and count == 2
    # Nothing to replace.
    assert insert_range_sentences("请带报告咨询医护。", [], references) == ("请带报告咨询医护。", [], 0)


def test_range_insertion_uses_only_retrieved_approved_sources():
    from medagent.agents.checks import insert_range_sentences
    claim = "参考范围是统计区间。"
    assert insert_range_sentences(claim, [], {}) == (claim, [], 0)
    references = range_references()
    hidden = {key:ref.model_copy(update={"approved":False}) for key, ref in references.items()}
    assert insert_range_sentences(claim, [], hidden) == (claim, [], 0)
    # No room for another citation: leave it for the normal check.
    full = [f"key-{index}" for index in range(8)]
    assert insert_range_sentences(claim, full, references) == (claim, full, 0)


def test_failed_sentence_match_is_answered_from_the_source_through_the_model_path():
    """The pilot defect: "那参考范围…" misses the direct path; the model's own wording must not fail the answer."""
    class Paraphrase(Model):
        def __init__(self): self.calls = 0
        async def message(self,stage,system,messages,tools=None,budget=1800):
            if stage != "agent_report_explainer": return await super().message(stage,system,messages,tools,budget)
            self.calls += 1
            data = output("参考范围就是统计出来的区间。", "参考范围就是统计出来的区间。", "lab-context")
            return {"content":[{"type":"text", "text":json.dumps(data,ensure_ascii=False)}]}

    async def scenario():
        model = Paraphrase()
        runtime = await fixture(model)
        await runtime.chat(Question(text="我在学习阅读化验报告，只了解通用概念。", user="u", session="s"))
        calls = model.calls
        reply = await runtime.chat(Question(text="那参考范围是什么意思？", user="u", session="s"))
        assert model.calls == calls+1 and not reply.degraded and "统计出来" not in reply.answer.text
        assert RANGE_SENTENCE in reply.answer.text and "lab-reference-range-definition" in reply.answer.reference_keys
        assert any(event["event"] == "report_range_inserted" for event in runtime.monitor.traces[-1]["events"])
    run(scenario())
