"""Offline replays of the report/visit, cross-section handover and report fallback cases."""
import json

import pytest

from medagent.agents.collaboration import missing_evidence_text
from medagent.core.contracts import Question
from medagent.understanding.conversation import followup_preparation_clause
from support.harness import Model, fixture, role_data, run


class Replay(Model):
    def __init__(self, scores, lines=None, invalid_report=False):
        self.scores = scores
        self.lines = lines or {}
        self.invalid_report = invalid_report
        self.inputs = {}

    async def json(self, stage, system, content, budget=900):
        if stage == "understanding":
            text = json.loads(content)["current"]["text"]
            medicine, _, report = text.partition("？")
            tasks = ([{"intent":kind, "goal":part, "quote":part} for kind, part in
                      [("medicine", medicine), ("report", report.rstrip("？。"))]]
                     if "布洛芬" in medicine and "报告" in report else [])
            return {"scores":self.scores, "tasks":tasks, "visit_quote":"去看牙" if "去看牙" in text else ""}
        return await super().json(stage, system, content, budget)

    async def message(self, stage, system, messages, tools=None, budget=1800):
        self.inputs[stage] = role_data(messages)
        if self.invalid_report and stage == "agent_report_explainer":
            return {"content":[{"type":"text", "text":"{}"}]}
        response = await super().message(stage, system, messages, tools, budget)
        if stage in self.lines:
            draft = json.loads(response["content"][0]["text"])
            draft["answer"]["text"] = self.lines[stage]
            draft["answer"]["limitations"] = [self.lines[stage]]
            response["content"][0]["text"] = json.dumps(draft, ensure_ascii=False)
        return response


def chat(text, model, *, empty=False):
    async def scenario():
        runtime = await fixture(model)
        if empty:
            runtime.index.documents.clear()
        return await runtime.chat(Question(text=text))
    return run(scenario())


@pytest.mark.parametrize("text, requested", [
    ("体检报告出来了，想知道参考范围怎么理解，还有下周复查要准备什么。", "复查要准备什么"),
    ("请解释报告的参考范围。下周要复诊，我需要带哪些资料？", "我需要带哪些资料"),
    ("我想知道参考范围怎么理解。最近不舒服，复诊前要提前准备哪些材料？", "提前准备哪些材料"),
    ("报告里的参考范围是什么意思？要带哪些资料去复诊？", "带哪些资料去复诊"),
    ("明天复查，想问医生的问题怎么理？", "想问医生的问题怎么理"),
    ("下周复查，帮我理一下要带什么，解释一下报告的参考范围，再查查成人布洛芬的剂型资料。", "要带什么"),
])
def test_visit_receives_the_explicit_request_even_with_an_empty_model_quote(text, requested):
    model = Replay({"report":.7, "visit":.3})
    reply = chat(text, model)
    visit = next(r for r in reply.agent_results if r.role == "visit_guide")
    inputs = model.inputs["agent_visit_guide"]
    assert requested in inputs["current_request"]
    assert inputs["validated_input"]["visit_context"] and visit.status == "answered"
    assert "consultation-record" in reply.answer.reference_keys == [ref.key for ref in reply.references]
    assert not any("是否涉及就医准备" in line for line in [reply.answer.text, *reply.answer.questions])


@pytest.mark.parametrize("text", [
    "上次复查带了报告。这次只想知道报告的参考范围是什么意思。",
    "我不需要问复诊要带什么，只想解释报告的参考范围。",
    "下周出差要带什么？之前复查过。",
    "这是例句：“复查要带哪些资料？”",
])
def test_background_or_negated_followup_is_not_preparation_evidence(text):
    assert not followup_preparation_clause(text)


@pytest.mark.parametrize("text", [
    "上次复查带了报告。这次只想知道参考范围是什么意思。",
    "上次复查带了报告。请解释一下参考范围的含义。",
    "我不需要问复诊要带什么，只想解释报告的参考范围。",
])
def test_background_visit_does_not_receive_a_checklist(text):
    model = Replay({"report":.9, "visit":.1})
    reply = chat(text, model)
    assert "visit_guide" not in [a.role for a in reply.routing_decision.assignments]
    assert "consultation-record" not in reply.answer.reference_keys


def test_named_institution_keeps_its_source_boundary():
    model = Replay({"visit":1})
    reply = chat("去这家医院复诊要带哪些材料？", model)
    assert "机构专属资料" in reply.answer.text and not reply.references
    assert "agent_visit_guide" not in model.inputs


@pytest.mark.parametrize("question, role, statement, refs", [
    ("成人布洛芬的公开资料是哪家机构写的？另外报告上的项目和结果怎么对应？", "drug_reference",
     "药品资料仅为概览。另外，报告项目与结果的对应属于另一部分任务，本部分未作解答。", {"report", "medicine"}),
    ("过两天去看牙，帮我想想要带什么；顺便查查成人布洛芬的公开资料。", "visit_guide",
     "具体证件、收费、号源和预约入口请以就诊机构官方渠道为准；布洛芬的公开资料由其他角色另行整理。", {"visit", "medicine"}),
    ("成人布洛芬的资料来自哪里？也说说化验报告里的项目和数值怎么对应。", "drug_reference",
     "药品资料仅为概览。报告项目和数值的对应关系，这里不再展开。", {"report", "medicine"}),
])
def test_handover_removed_only_when_other_section_survives(question, role, statement, refs):
    model = Replay({domain:.5 for domain in sorted(refs)}, {"agent_"+role:statement})
    reply = chat(question, model)
    visible = [reply.answer.text, *reply.answer.limitations]
    for handover in ("另一部分任务", "其他角色", "这里不再展开"):
        assert not any(handover in line for line in visible)
    assert statement.split("。")[0].split("；")[0] in reply.answer.text
    assert {ref.domain.value for ref in reply.references} == refs
    assert reply.answer.reference_keys == [ref.key for ref in reply.references]


def test_failed_sibling_does_not_erase_real_gap_or_export_its_citations():
    gap = "药品资料仅为概览。报告的整理这里未作解答。未核查布洛芬的具体产品禁忌。"
    reply = chat("成人布洛芬的公开资料是哪家机构写的？另外报告上的项目和结果怎么对应？",
                 Replay({"medicine":.85, "report":.15}, {"agent_drug_reference":gap}, invalid_report=True))
    assert gap in reply.answer.text and all(r.domain == "medicine" for r in reply.references)
    assert any("报告整理" in line and "未通过校验" in line for line in reply.answer.limitations)


def test_completed_visit_checklist_does_not_clear_a_specific_medication_gap():
    line = "本次未查询复诊时所用药品的具体产品禁忌。"
    reply = chat("过两天去看牙，帮我想想要带什么；顺便查查成人布洛芬的公开资料。",
                 Replay({"visit":.6, "medicine":.4}, {"agent_drug_reference":line}))
    assert line in reply.answer.text and line in reply.answer.limitations
    assert {ref.domain.value for ref in reply.references} == {"visit", "medicine"}


def test_three_task_clarification_adds_only_new_evidence_and_keeps_every_citation():
    body = ("目前检索到的只有 NHS 英国成人布洛芬公开概览的简短导读，它说明布洛芬是非甾体抗炎药，"
            "可用于缓解疼痛及炎症，并提示片剂、胶囊、外用等剂型不同，应以具体包装或说明书为准。"
            "这份资料没有列出各剂型的具体规格、用法或禁忌，因此无法据此回答某一剂型的详细信息。")
    evidence = ("NHS成人布洛芬页面面向18岁及以上人群，说明布洛芬属于非甾体抗炎药，可缓解疼痛及炎症；"
                "页面区分片剂、胶囊、外用等剂型，并提示按具体包装或说明书核对使用方法。")
    class ThreeTasks(Replay):
        async def message(self, stage, *args, **kwargs):
            response = await super().message(stage, *args, **kwargs)
            if stage == "agent_drug_reference":
                draft = json.loads(response["content"][0]["text"])
                draft.update(status="needs_clarification", missing_fields=["formulation"])
                draft["answer"]["text"] = body
                draft["payload"]["evidence_items"][0]["text"] = evidence
                response["content"][0]["text"] = json.dumps(draft)
            return response

    model = ThreeTasks({"visit":.5, "report":.3, "medicine":.2})
    reply = chat("下周复查，帮我理一下要带什么，解释一下报告的参考范围，再查查成人布洛芬的剂型资料。", model)
    assert {r.domain.value for r in reply.references} == {"visit", "report", "medicine"}
    assert reply.answer.reference_keys == [r.key for r in reply.references]
    assert body in reply.answer.text and "18岁及以上" in reply.answer.text and "核对使用方法" in reply.answer.text
    for fact in ("非甾体抗炎药", "缓解疼痛", "外用等剂型"):
        assert reply.answer.text.count(fact) == 1


@pytest.mark.parametrize("body, fact", [
    ("资料面向18岁及以上人群。", "资料面向12岁及以上人群。"),
    ("不能据此判断个人适用性。", "如果还有其他用药，不能据此判断个人适用性。"),
    ("该资料没有列出禁忌。", "该资料列出了禁忌。"),
    ("参考范围是一个区间。", "本资料说明，参考范围是一个区间。"),
])
def test_distinct_quantities_conditions_and_negations_survive(body, fact):
    assert missing_evidence_text(body, fact) == fact


def test_reworded_overview_keeps_the_new_scope_without_repeating_the_class():
    body = "该页面说明阿莫西林是抗生素类药物。"
    extra = missing_evidence_text(body, "本条为公开资料导读，说明阿莫西林属于抗生素类药物。")
    assert extra == "本条为公开资料导读。"


@pytest.mark.parametrize("question", [
    "看碳13尿素呼气试验报告时，DOB 数值达到多少才算阳性？",
    "碳13呼气试验的报告里，DOB 要到什么数值才判阳性？",
])
@pytest.mark.parametrize("empty", [False, True])
def test_report_validation_failure_and_empty_lookup_are_distinguishable(question, empty):
    model = Replay({"report":1}, invalid_report=True)
    reply = chat(question, model, empty=empty)
    assert not reply.references and not reply.answer.reference_keys
    if empty:
        assert "agent_report_explainer" not in model.inputs
        assert "没有匹配的报告阅读资料" in reply.answer.text and "校验" not in reply.answer.text
    else:
        attempts = reply.routing_decision.assignments[0].attempts
        assert len(attempts) == 2 and all(a.error_code == "invalid_output_shape" for a in attempts)
        assert "查到了资料" in reply.answer.text and "未通过校验" in reply.answer.text
        assert "没有匹配" not in reply.answer.text and any("原报告" in line for line in reply.answer.next_steps)


def test_successful_retry_does_not_label_the_final_answer_as_invalid():
    class Retry(Replay):
        async def message(self, stage, *args, **kwargs):
            response = await super().message(stage, *args, **kwargs)
            self.invalid_report = False
            return response

    reply = chat("看碳13尿素呼气试验报告时，DOB 数值达到多少才算阳性？", Retry({"report":1}, invalid_report=True))
    attempts = reply.routing_decision.assignments[0].attempts
    assert [a.status for a in attempts] == ["failed", "ok"]
    assert "校验" not in reply.answer.text and reply.answer.reference_keys
