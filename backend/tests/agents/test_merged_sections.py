"""Merged replies keep every completed part, and a part another section covers never reads as missing."""
import json
from types import SimpleNamespace as NS

import pytest

from medagent.agents.collaboration import removable_sentences, sections_answer
from medagent.core.contracts import Answer, Question
from medagent.understanding.conversation import visit_query_has_context
from support.harness import Model, fixture, role_data, role_output, run

COMPOUND = "咳嗽三天，去看医生需要准备哪些资料？"


class Clarifying(Model):
    """The named roles stop to ask; the others answer as usual."""
    def __init__(self, *roles):
        self.roles = roles

    async def message(self, stage, system, messages, tools=None, budget=1800):
        role = stage.removeprefix("agent_")
        if role not in self.roles:
            return await super().message(stage, system, messages, tools, budget)
        data = role_output(role)
        data.update(status="needs_clarification", missing_fields=["visit_purpose" if role == "visit_guide" else "symptom"])
        data["answer"]["text"] = role+"还需要补充信息。"
        return {"content":[{"type":"text","text":json.dumps(data, ensure_ascii=False)}]}


def test_primary_clarification_keeps_the_completed_support():
    async def scenario():
        reply = await (await fixture(Clarifying("visit_guide"))).chat(Question(text=COMPOUND))
        assert reply.primary_agent == "visit_guide"
        assert {item.role:item.status for item in reply.agent_results} == {"visit_guide":"needs_clarification", "symptom_guide":"answered"}
        assert reply.answer.text.index("visit_guide还需要补充信息。") < reply.answer.text.index("可按资料记录信息并向医护核实。")
        decision = reply.routing_decision
        assert decision.collaboration_status == "partial" and not reply.degraded
        assert [(item.reason, item.action) for item in decision.fallbacks] == [("primary_requires_clarification_or_handoff", "keep_completed_support")]
    run(scenario())


def test_primary_clarification_with_nothing_completed_stays_alone():
    async def scenario():
        reply = await (await fixture(Clarifying("visit_guide", "symptom_guide"))).chat(Question(text=COMPOUND))
        assert reply.answer.text == "visit_guide还需要补充信息。"
        assert [(item.reason, item.action) for item in reply.routing_decision.fallbacks] == [("primary_requires_clarification_or_handoff", "retain_primary_state")]
    run(scenario())


def test_oversized_support_falls_back_to_the_primary_clarification():
    class LongSupport(Clarifying):
        async def message(self, stage, system, messages, tools=None, budget=1800):
            if stage != "agent_symptom_guide":
                return await super().message(stage, system, messages, tools, budget)
            data = role_output("symptom_guide")
            data["answer"]["text"] = "记录"*3000
            return {"content":[{"type":"text","text":json.dumps(data)}]}

    async def scenario():
        reply = await (await fixture(LongSupport("visit_guide"))).chat(Question(text=COMPOUND))
        assert reply.degraded and reply.answer.text == "visit_guide还需要补充信息。"
        assert any("症状记录部分未完成" in item for item in reply.answer.limitations)
        assert any(item.reason == "composition_size_limit" for item in reply.routing_decision.fallbacks)
    run(scenario())


class Recording(Model):
    def __init__(self):
        self.seen = {}

    async def message(self, stage, system, messages, tools=None, budget=1800):
        self.seen[stage] = role_data(messages)
        return await super().message(stage, system, messages, tools, budget)


def test_each_role_is_told_which_other_parts_are_arranged():
    async def scenario():
        model = Recording()
        await (await fixture(model)).chat(Question(text=COMPOUND))
        assert model.seen["agent_visit_guide"]["assignment"]["other_goals"] == ["整理症状记录和信息缺口；发烧吃什么有歧义时确认饮食或药品方向。"]
        assert model.seen["agent_symptom_guide"]["assignment"]["other_goals"] == ["仅整理本次就医准备事项和需向机构核实的问题。"]
    run(scenario())


def test_a_lone_role_is_not_told_anything_is_arranged():
    async def scenario():
        model = Recording()
        await (await fixture(model)).chat(Question(text="看医生前要准备哪些资料？"))
        assert "other_goals" not in model.seen["agent_visit_guide"]["assignment"]
    run(scenario())


def test_role_prompt_separates_arranged_parts_from_real_omissions():
    async def scenario():
        class Prompt(Model):
            async def message(self, stage, system, messages, tools=None, budget=1800):
                self.system = system
                return await super().message(stage, system, messages, tools, budget)
        model = Prompt()
        await (await fixture(model)).chat(Question(text="看医生前要准备哪些资料？"))
        assert "assignment.other_goals" in model.system
    run(scenario())


class Speaking(Model):
    """Every role reports a text and limitations given per role."""
    def __init__(self, sentences):
        self.sentences = sentences

    async def message(self, stage, system, messages, tools=None, budget=1800):
        reply = await super().message(stage, system, messages, tools, budget)
        role = stage.removeprefix("agent_")
        if role in self.sentences:
            data = json.loads(reply["content"][0]["text"])
            data["answer"]["text"] = "这部分已整理。"+self.sentences[role]
            data["answer"]["limitations"] = [self.sentences[role]]
            return {"content":[{"type":"text","text":json.dumps(data, ensure_ascii=False)}]}
        return reply


@pytest.mark.parametrize("sentence", [
    "门诊准备清单这次没有查询，本次不涉及。",
    "就医准备不在本次整理范围内，这次没有查询。",
    "关于就医准备本次不在我的任务范围内，这里没有查询。",
    "本次未整理门诊准备清单，该部分不在本次查询范围内。",
    "本次只整理症状记录，未涉及复诊材料的整理，该部分需另行处理。",
    "关于见医生当天应携带的物品，本次未查询，此处不作说明。",
    "关于要带的材料，属于另一部分内容，本次未处理。",
    "本回答不涉及复诊准备事项，该部分由其他任务处理。",
])
def test_a_part_another_section_covers_is_not_reported_missing(sentence):
    async def scenario():
        reply = await (await fixture(Speaking({"symptom_guide":sentence}))).chat(Question(text=COMPOUND))
        assert reply.routing_decision.collaboration_status == "merged"
        assert sentence.rstrip("。") not in reply.answer.text
        assert sentence not in reply.answer.limitations
    run(scenario())


def test_a_section_keeps_its_own_gap():
    sentence = "具体证件、费用和号源本次未查询或确认，门诊准备清单里的内容请向机构核实。"
    async def scenario():
        reply = await (await fixture(Speaking({"visit_guide":sentence}))).chat(Question(text=COMPOUND))
        assert sentence in reply.answer.text and sentence in reply.answer.limitations
    run(scenario())


def test_gap_in_what_the_other_section_never_covers_stays():
    sentence = "未检索到适用的公开资料，本次不提供具体处理建议。"
    async def scenario():
        reply = await (await fixture(Speaking({"symptom_guide":sentence}))).chat(Question(text=COMPOUND))
        assert sentence in reply.answer.text and sentence in reply.answer.limitations
    run(scenario())


def test_alone_a_role_still_says_what_it_did_not_do():
    sentence = "布洛芬的公开资料这次没有查询。"
    async def scenario():
        reply = await (await fixture(Speaking({"visit_guide":sentence}))).chat(Question(text="看医生前要准备哪些资料？"))
        assert sentence in reply.answer.text
    run(scenario())


def test_part_of_a_failed_section_stays_reported_missing():
    sentence = "门诊准备清单这次没有查询。"
    class FailedVisit(Speaking):
        async def message(self, stage, *args, **kwargs):
            if stage == "agent_visit_guide": raise ConnectionError()
            return await super().message(stage, *args, **kwargs)

    async def scenario():
        reply = await (await fixture(FailedVisit({"symptom_guide":sentence}))).chat(Question(text=COMPOUND))
        assert sentence in reply.answer.text
    run(scenario())


def test_a_surviving_section_never_says_a_failed_part_was_handled():
    sentence = "本回答不涉及复诊准备事项，该部分由其他任务处理。"
    class FailedVisit(Speaking):
        async def message(self, stage, *args, **kwargs):
            if stage == "agent_visit_guide": raise ConnectionError()
            return await super().message(stage, *args, **kwargs)

    async def scenario():
        reply = await (await fixture(FailedVisit({"symptom_guide":sentence}))).chat(Question(text=COMPOUND))
        assert "其他任务" not in reply.answer.text and "本回答不涉及复诊准备事项。" in reply.answer.text
        assert any("就医准备部分未完成" in item for item in reply.answer.limitations)
    run(scenario())


def sections(*answers):
    return [NS(answer=Answer(text=text, limitations=limitations, next_steps=steps), execution=NS(role=role, task_id="task:"+role, status="answered", missing_fields=[]))
            for role, text, limitations, steps in answers]


def test_merged_limitations_keep_one_line_per_kind_of_notice():
    answer = sections_answer(sections(
        ("symptom_guide", "咳嗽三天。", ["本记录不构成诊断。", "“最近”为相对时间。"], []),
        ("visit_guide", "带上资料。", ["本回答不构成诊断或治疗建议。", "本清单为通用模板，不是医院规则。", "本清单为项目通用沟通模板，不是某家医院的规则。"], [])))
    assert answer.limitations == ["本记录不构成诊断。", "“最近”为相对时间。", "本清单为通用模板，不是医院规则。"]


def test_urgent_notice_is_never_folded_into_a_shared_line():
    urgent = "如症状加重，请立即就医，本回答不构成诊断。"
    answer = sections_answer(sections(("symptom_guide", "咳嗽。", ["本记录不构成诊断。"], []), ("visit_guide", "带资料。", [urgent], [])))
    assert urgent in answer.limitations


def test_merged_steps_drop_a_near_copy():
    answer = sections_answer(sections(
        ("symptom_guide", "咳嗽。", [], ["如有，整理现用药品与已知过敏情况。"]),
        ("visit_guide", "带资料。", [], ["如有，整理现用药品和已知过敏情况。", "向医院官方渠道核实费用。"])))
    assert answer.next_steps == ["如有，整理现用药品与已知过敏情况。", "向医院官方渠道核实费用。"]


class Dropping(Model):
    """Answers with a process sentence and a fact, then asks the merge step to drop whatever `pick` chooses."""
    def __init__(self, pick):
        self.pick, self.offered = pick, []

    async def json(self, stage, system, content, budget=900):
        if stage != "synthesis":
            return await super().json(stage, system, content, budget)
        source = json.loads(content)
        self.offered = [item for section in source["sections"] for item in section["removable"]]
        return {"section_ids":[section["id"] for section in source["sections"]], "drop_ids":self.pick(self.offered)}

    async def message(self, stage, system, messages, tools=None, budget=1800):
        result = await super().message(stage, system, messages, tools, budget)
        data = json.loads(result["content"][0]["text"])
        data["answer"]["text"] = "已整理您提供的情况。可按资料记录信息。如症状加重，请立即就医。"
        return {"content":[{"type":"text","text":json.dumps(data, ensure_ascii=False)}]}


def merged(pick):
    async def scenario():
        model = Dropping(pick)
        reply = await (await fixture(model)).chat(Question(text=COMPOUND))
        return model, reply
    return run(scenario())


def test_merge_can_drop_an_offered_process_sentence():
    model, reply = merged(lambda offered:[item["id"] for item in offered])
    assert model.offered and "已整理您提供的情况" not in reply.answer.text and "可按资料记录信息" in reply.answer.text
    assert not reply.degraded and reply.routing_decision.collaboration_status == "merged"


def test_merge_is_not_offered_facts_or_urgent_advice():
    model, _ = merged(lambda offered:[])
    assert model.offered and all(item["text"].startswith("已整理") for item in model.offered)


def test_dropping_an_unoffered_sentence_falls_back_to_the_full_sections():
    _, reply = merged(lambda offered:["task:visit#1"])
    assert "已整理您提供的情况" in reply.answer.text and reply.degraded
    assert any(item.reason == "synthesis_failed" for item in reply.routing_decision.fallbacks)


def test_a_section_never_empties():
    class Bare(Dropping):
        async def message(self, stage, system, messages, tools=None, budget=1800):
            result = await Model.message(self, stage, system, messages, tools, budget)
            data = json.loads(result["content"][0]["text"])
            data["answer"]["text"] = "已整理您提供的情况。"
            return {"content":[{"type":"text","text":json.dumps(data, ensure_ascii=False)}]}

    async def scenario():
        return await (await fixture(Bare(lambda offered:[item["id"] for item in offered]))).chat(Question(text=COMPOUND))
    assert "已整理您提供的情况" in run(scenario()).answer.text


def test_role_prompt_keeps_process_talk_out_of_the_text():
    class Prompts(Model):
        systems = []

        async def message(self, stage, system, messages, tools=None, budget=1800):
            self.systems.append(system)
            return await super().message(stage, system, messages, tools, budget)

    async def scenario():
        model = Prompts()
        await (await fixture(model)).chat(Question(text=COMPOUND))
        assert all("text只写用户要的内容" in system for system in model.systems)
    run(scenario())


def test_a_medication_notice_is_not_folded_into_a_diagnosis_notice():
    line = "本回答不构成个人用药、剂量或停换药决定。"
    answer = sections_answer(sections(("symptom_guide", "咳嗽。", ["本记录不构成诊断。"], []), ("drug_reference", "资料。", [line], [])))
    assert line in answer.limitations


def test_notice_kinds_apply_to_limitations_only():
    steps = ["记录用药情况，本次未查询医院规则，需向医院核实。", "本清单不构成诊断，请带给医生。", "另一份模板也带上。"]
    answer = sections_answer(sections(("symptom_guide", "咳嗽。", [], steps[:1]), ("visit_guide", "带资料。", [], steps[1:])))
    assert answer.next_steps == steps


@pytest.mark.parametrize("sentence", ["已整理您提供的记录：膝盖疼，约三天前开始。", "以下为通用准备清单：可整理已提供的症状。"])
def test_a_sentence_that_carries_content_is_not_offered(sentence):
    result = sections(("symptom_guide", sentence, [], []), ("visit_guide", "带资料。", [], []))
    assert removable_sentences(result, result[0]) == []


REPORT_FIELDS = ["report_item", "value", "unit", "reference_range"]


def asking(role, questions, status="answered", missing=()):
    return NS(answer=Answer(text=role+"。", questions=questions), execution=NS(role=role, task_id="task:"+role, status=status, missing_fields=list(missing)))


def test_a_role_waiting_on_several_fields_asks_for_them_once():
    waiting = asking("report_explainer", ["请提供希望了解的报告项目名称。", "该项目在报告上的原始结果是什么？", "该项目的原始单位是什么？", "报告为该项目列出的参考范围是什么？"],
                     "needs_clarification", REPORT_FIELDS)
    answer = sections_answer([asking("drug_reference", ["布洛芬是哪种剂型？"]), waiting])
    assert len(answer.questions) == 2
    assert all(name in answer.questions[0] for name in ("报告项目名称", "结果数值", "单位", "参考范围"))


def test_a_role_waiting_on_one_field_keeps_its_own_question():
    waiting = asking("visit_guide", ["你希望为哪类就医问题准备资料？"], "needs_clarification", ["visit_purpose"])
    assert sections_answer([asking("drug_reference", []), waiting]).questions == ["你希望为哪类就医问题准备资料？"]


def test_merged_questions_stop_at_three_and_every_role_gets_a_turn():
    answer = sections_answer([asking("drug_reference", ["剂型是什么？", "规格是多少？", "产品名称是什么？"]),
                              asking("visit_guide", ["复查哪一项？", "哪家医院？"]),
                              asking("symptom_guide", ["头疼在哪个部位？"])])
    assert answer.questions == ["剂型是什么？", "复查哪一项？", "头疼在哪个部位？"]


def test_a_role_waiting_on_the_user_asks_before_the_finished_ones():
    answer = sections_answer([asking("drug_reference", ["剂型是什么？", "规格是多少？"]), asking("visit_guide", ["复查哪一项？", "哪家医院？"]),
                              asking("report_explainer", ["请提供希望了解的报告项目名称。"], "needs_clarification", ["report_item"])])
    assert answer.questions[0] == "请提供希望了解的报告项目名称。" and len(answer.questions) == 3


def test_a_lone_role_keeps_every_field_question():
    questions = ["请提供希望了解的报告项目名称。", "该项目在报告上的原始结果是什么？", "该项目的原始单位是什么？", "报告为该项目列出的参考范围是什么？"]
    assert sections_answer([asking("report_explainer", questions, "needs_clarification", REPORT_FIELDS)]).questions == questions


def test_role_prompt_keeps_questions_for_the_user():
    class Prompts(Model):
        systems = []

        async def message(self, stage, system, messages, tools=None, budget=1800):
            self.systems.append(system)
            return await super().message(stage, system, messages, tools, budget)

    async def scenario():
        model = Prompts()
        await (await fixture(model)).chat(Question(text=COMPOUND))
        assert all("questions只问" in system and "放进next_steps" in system for system in model.systems)
    run(scenario())


def test_a_field_without_a_short_name_keeps_the_question_that_asks_for_it():
    questions = ["包装上标注的是哪种剂型？不清楚时可以保留未知。", "包装标注的每单位药品规格是什么？这不是建议服用剂量。", "是否还有需要一起向药师核实的其他药品？"]
    waiting = asking("drug_reference", questions, "needs_clarification", ["formulation", "strength", "other_medications"])
    answer = sections_answer([asking("visit_guide", []), waiting])
    assert answer.questions == questions


@pytest.mark.parametrize("line, opposite", [("资料说明需要停药。", "资料说明不需要停药。"), ("可以继续服用布洛芬。", "不可以继续服用布洛芬。"), ("复查前需要空腹。", "复查前无需空腹。")])
def test_opposite_lines_are_not_near_copies(line, opposite):
    answer = sections_answer(sections(("drug_reference", "资料。", [], [line]), ("visit_guide", "带资料。", [], [opposite])))
    assert answer.next_steps == [line, opposite]


@pytest.mark.parametrize("text, is_visit", [("过两天去看牙，要带什么", True), ("下周挂皮肤科", True), ("我看皮肤有点红", False), ("看眼睛为什么红", False)])
def test_seeing_a_body_part_is_not_a_visit(text, is_visit):
    assert visit_query_has_context(text) is is_visit
