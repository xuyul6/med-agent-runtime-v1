"""Replay the language scores and task excerpts from the two single-pass compound cases offline."""
import json

import pytest

from medagent.agents.collaboration import plain_answer
from medagent.core.contracts import Answer, Question, Understanding
from medagent.understanding.clinician_note import clinician_note
from support.harness import Model, fixture, role_data, role_output, run
from support.medication import references


REPORT_AND_DRUG = "报告上有一项超出了参考范围，我又在吃布洛芬，想把这两件事放在一起理清楚再问医生，资料上各怎么说？"
ACCOUNT = "头疼一周，吃过布洛芬，上周化验有一项偏高。"
NOTE_REQUEST = "帮我把这三件事串起来，整理成一份给医生看的说明。"
REWORDED_ACCOUNT = "头痛大概一星期了，布洛芬我吃过，上星期的化验有个指标比参考值高。"
SPLIT_REQUEST = "能不能替我写成一小段话，门诊时我可以直接给医生看？"
FALSE_DENIAL = "本次未对布洛芬做药品资料核查"
REAL_LIMIT = "本次未查询布洛芬的具体产品说明书"


class CompoundModel(Model):
    async def json(self, stage, system, content, budget=900):
        if stage != "understanding":
            return await super().json(stage, system, content, budget)
        text = json.loads(content)["current"]["text"]
        if text == REPORT_AND_DRUG:
            scores = {"medicine":8, "report":8, "visit":5, "symptoms":1, "general":1}
            tasks = [("medicine", "我又在吃布洛芬"), ("report", "报告上有一项超出了参考范围"),
                     ("visit", "想把这两件事放在一起理清楚再问医生")]
        else:
            scores = {"symptoms":9, "medicine":6, "report":7, "visit":5, "general":1}
            tasks = [("visit", NOTE_REQUEST.rstrip("。"))]
        return {"scores":{kind:value/sum(scores.values()) for kind, value in scores.items()},
                "tasks":[{"intent":kind, "quote":quote, "goal":quote} for kind, quote in tasks]}

    async def message(self, stage, system, messages, tools=None, budget=1800):
        context = role_data(messages)
        if context["original_request_boundary"] != REPORT_AND_DRUG:
            return await super().message(stage, system, messages, tools, budget)
        role = stage.removeprefix("agent_")
        key, field, gaps = {
            "drug_reference":("ibuprofen-overview", "evidence_items", ["formulation", "strength"]),
            "report_explainer":("lab-reference-range-definition", "term_explanations", ["report_item","value","unit","reference_range"]),
        }[role]
        reference = next(ref for ref in context["references"] if ref["key"] == key)
        excerpt = reference["text"].split("。")[0]+"。"
        draft = role_output(role, [key])
        draft.update(status="needs_clarification", missing_fields=gaps)
        draft["payload"][field][0]["text"] = excerpt
        # The report's validated explanation is present only in its structured output.
        draft["answer"]["text"] = excerpt if role == "drug_reference" else "需要补充具体检验项目才能解释个人结果。"
        if role == "report_explainer":
            draft["answer"]["limitations"] = [FALSE_DENIAL+"，药物相关问题请咨询药师或医生。", REAL_LIMIT+"。"]
            draft["answer"]["text"] += "关于布洛芬，这里只能整理报告文字，无法代为核对药品资料，相关问题请由药师或医生确认。"
        return {"content":[{"type":"text","text":json.dumps(draft, ensure_ascii=False)}]}


def test_two_clarifications_keep_each_roles_evidence_references_and_questions():
    async def scenario():
        reply = await (await fixture(CompoundModel())).chat(Question(text=REPORT_AND_DRUG))
        assert reply.primary_agent == "drug_reference"
        assert [(r.role,r.status) for r in reply.agent_results] == [
            ("drug_reference","needs_clarification"), ("report_explainer","needs_clarification")]
        assert reply.routing_decision.collaboration_status == "partial"
        assert "keep_cited_support" in [f.action for f in reply.routing_decision.fallbacks]
        for result, field in zip(reply.agent_results, ("evidence_items", "term_explanations"), strict=True):
            assert result.payload[field][0]["text"] in reply.answer.text
        assert set(reply.answer.reference_keys) == {r.key for r in reply.references} == {"ibuprofen-overview", "lab-reference-range-definition"}
        assert any("规格" in q for q in reply.answer.questions)
        assert any("单位" in q for q in reply.answer.questions)
        limits = " ".join(reply.answer.limitations)
        assert FALSE_DENIAL not in limits and "药物相关问题请咨询药师或医生" in limits
        assert REAL_LIMIT in limits
        assert "无法代为核对药品资料" not in reply.answer.text and "相关问题请由药师或医生确认" in reply.answer.text
    run(scenario())


def test_clinician_note_preserves_three_reported_facts_with_no_lab_item_or_visit_role():
    async def scenario():
        reply = await (await fixture(CompoundModel())).chat(Question(text=ACCOUNT+NOTE_REQUEST))
        assert reply.intent.scores["visit"] == pytest.approx(.10045, abs=.00001)
        assert [r.role for r in reply.agent_results] == ["symptom_guide", "drug_reference"]
        assert "report_item" not in reply.intent.entities
        assert reply.answer.text.split("\n\n")[0] == "【给医生的说明（本轮自述）】\n"+ACCOUNT
    run(scenario())


def test_drug_as_background_cannot_override_an_explicit_symptom_only_request():
    async def scenario():
        # This model ranks medicine first even though the user expressly narrowed the task.
        reply = await (await fixture(Model())).chat(Question(text="我吃过布洛芬，但这次只想记录头疼。"))
        assert reply.intent.primary == "medicine"
        assert [a.role for a in reply.routing_decision.assignments] == [reply.primary_agent] == ["symptom_guide"]
        assert "ibuprofen-overview" not in reply.answer.reference_keys
    run(scenario())


def test_note_can_precede_the_verbatim_account_and_preserves_negation():
    assert clinician_note("请把下面的情况整理成给医生看的说明：这周没有头疼，上周化验结果还没拿到。", Understanding()).endswith(
        "\n这周没有头疼，上周化验结果还没拿到。")


@pytest.mark.parametrize("request_text", [SPLIT_REQUEST, "请帮我写一段话。给医生看。",
                                          "帮我整理成一份说明。\n复诊时我想直接给大夫看。"])
def test_adjacent_writing_request_and_recipient_preserve_verbatim_account(request_text):
    async def scenario():
        reply = await (await fixture(Model())).chat(Question(text=REWORDED_ACCOUNT+request_text))
        assert reply.answer.text.split("\n\n")[0] == "【给医生的说明（本轮自述）】\n"+REWORDED_ACCOUNT
    run(scenario())


def test_reconcile_only_unqualified_lookup_denials_for_cited_approved_drug():
    refs = references()
    denial = "布洛芬的公开资料这次没有查询"
    answer = Answer(text=denial+"。", limitations=[FALSE_DENIAL+"，"+REAL_LIMIT+"。",
        "没有查询阿莫西林的药品资料。", "未核查布洛芬与其他药物的相互作用，不能判断个人适用性。",
        "关于阿莫西林，这里只能整理报告文字，无法代为核对药品资料。",
        "关于布洛芬，这里只能整理报告文字，无法代为核对该产品的完整说明书。"],
        questions=["之前没有查询布洛芬的资料吗？"], next_steps=["没有查阅 Ibuprofen 的药品资料。"],
        reference_keys=["ibuprofen-overview"])
    fixed = plain_answer(answer, refs)
    assert denial not in fixed.text and FALSE_DENIAL not in fixed.limitations[0]
    assert REAL_LIMIT in fixed.limitations[0] and fixed.limitations[1:] == answer.limitations[1:]
    assert fixed.questions == answer.questions and fixed.next_steps != answer.next_steps
    for keys in ([], ["lab-reference-range-definition"]):
        other = answer.model_copy(update={"reference_keys":keys})
        assert plain_answer(other, refs) == other
    unapproved = refs["ibuprofen-overview"].model_copy(update={"approved":False})
    assert plain_answer(answer, {unapproved.key:unapproved}) == answer


@pytest.mark.parametrize("text", [
    "我头疼三天，没有让你写给医生的说明。",
    "举例说头疼三天，吃过布洛芬。整理成给医生看的说明。",
    "我头疼三天，想知道参考范围的含义，之后再问医生。",
    "谢谢你。帮我写一份给医生看的说明。",
    "头痛三天。帮我写一段话。不要给医生看。",
    "头痛三天。帮我写一段话给同事。明天去门诊给医生看。",
    "头痛三天。请写一段话。我还要整理账单。给医生看。",
])
def test_only_explicit_requests_with_a_personal_account_produce_a_note(text):
    assert not clinician_note(text, Understanding())
