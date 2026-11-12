"""Defects the layered evaluation found in the multi-role path: each stopped a role that could have answered."""
import json
from types import SimpleNamespace as NS

import pytest

from medagent.agents.contracts import VisitInput
from medagent.core.contracts import Intent, Question
from medagent.knowledge.retrieval import visit_relevance_terms
from medagent.understanding.conversation import role_details, visit_query_has_context
from support.harness import Model, fixture, role_data, role_output, run

FACT = NS(value="布洛芬", state="provided")


class QuotedTasks(Model):
    def __init__(self, tasks):
        self.tasks = tasks

    async def json(self, stage, system, content, budget=900):
        if stage == "understanding":
            kinds = [kind for kind, _ in self.tasks]
            return {"scores":{kind:(.6 if index == 0 else .3) for index, kind in enumerate(kinds)}, "updates":[], "topic_relation":"new",
                    "urgent_quote":"", "tasks":[{"intent":kind, "goal":"任务", "quote":quote} for kind, quote in self.tasks]}
        return await super().json(stage, system, content, budget)


def admitted(text, tasks):
    async def scenario():
        reply = await (await fixture(QuotedTasks(tasks))).chat(Question(text=text))
        return sorted(item.role for item in reply.routing_decision.assignments)
    return run(scenario())


def details(**entities):
    understanding = NS(entities={"medication_name":FACT, **{name:NS(value=value, state="provided") for name, value in entities.items()}},
                       tasks=[], missing_fields=[], subject_ambiguous=False, product_quote=None, visit_quote=None)
    return role_details(understanding, "drug_reference")


class Recording(QuotedTasks):
    def __init__(self, tasks=()):
        self.tasks, self.keys = tasks, []

    async def json(self, stage, system, content, budget=900):
        if stage == "understanding" and not self.tasks:
            return await Model.json(self, stage, system, content, budget)
        return await super().json(stage, system, content, budget)

    async def message(self, stage, system, messages, tools=None, budget=1800):
        self.keys += [ref["key"] for ref in role_data(messages)["references"]]
        return await super().message(stage, system, messages, tools, budget)


def chat(model, text):
    async def scenario():
        return await (await fixture(model)).chat(Question(text=text))
    return run(scenario())


def test_a_form_alone_is_a_question_about_the_drug_not_a_product_lookup():
    assert "query_type" not in details(formulation="胶囊")


@pytest.mark.parametrize("field,value", [("strength", "200 mg"), ("product_name", "Advil")])
def test_a_named_product_still_needs_its_label(field, value):
    assert details(**{field:value})["query_type"] == "product_information"


@pytest.mark.parametrize("text", ["成人布洛芬有片剂也有胶囊吧，资料上咋说的", "布洛芬的外用凝胶资料怎么说"])
def test_asking_about_forms_reaches_the_overview(text):
    model = Recording()
    chat(model, text)
    assert "ibuprofen-overview" in model.keys


def test_missing_text_names_what_is_needed():
    from medagent.agents.execution import missing_text
    text = missing_text(["unit", "reference_range"])
    assert "单位" in text and "参考范围" in text and "不清楚的内容可以保留未知" in text


def test_unknown_field_still_reads_as_a_request():
    from medagent.agents.execution import missing_text
    assert "补充" in missing_text(["never_seen"])


def test_product_lookup_says_what_is_missing():
    reply = chat(Model(), "查布洛芬 200 mg 的产品说明书")
    assert reply.answer.text.startswith("还缺") and "下面的信息" not in reply.answer.text


def test_a_named_institution_is_not_asked_for_its_region():
    request = VisitInput(text="协和医院挂号几点开始放号", intent=Intent.VISIT, subject="self", institution="协和医院", institution_specific=True)
    assert request.information_gaps() == []


def test_a_named_institution_gets_the_source_gap_instead_of_a_question():
    reply = chat(Recording([("visit", "协和医院挂号几点开始放号")]), "协和医院挂号几点开始放号")
    assert "无法确认该机构" in reply.answer.text and not any("地区" in item for item in reply.answer.questions)


@pytest.mark.parametrize("text", ["过两天去看牙，帮我想想要带什么", "明天看中医要准备什么", "下周去口腔科要带什么"])
def test_a_specialty_visit_counts_as_visit_context(text):
    assert visit_query_has_context(text)


def test_going_to_the_dentist_shares_visit_terms_with_the_checklist():
    assert visit_relevance_terms("过两天去看牙") & visit_relevance_terms("就医准备")


def test_dentist_trip_gets_the_checklist():
    text = "过两天去看牙，帮我想想要带什么"
    model = Recording([("visit", "帮我想想要带什么")])
    reply = chat(model, text)
    assert "visit_purpose" not in reply.agent_results[0].missing_fields and "consultation-record" in model.keys


class StatusConflict(Model):
    """The first draft is answered and lists gaps anyway; the correction clears them."""
    def __init__(self, role):
        self.role, self.feedback = role, []

    async def message(self, stage, system, messages, tools=None, budget=1800):
        if stage != "agent_"+self.role:
            return await super().message(stage, system, messages, tools, budget)
        keys = [ref["key"] for ref in role_data(messages)["references"]]
        data = role_output(self.role, keys)
        if isinstance(messages[-1]["content"], str) and len(messages) > 1: self.feedback.append(json.loads(messages[-1]["content"]))
        else: data["missing_fields"] = ["strength"]
        return {"content":[{"type":"text","text":json.dumps(data, ensure_ascii=False)}]}


def test_medicine_status_conflict_is_corrected_in_place():
    model = StatusConflict("drug_reference")
    reply = chat(model, "查询布洛芬药品资料")
    attempts = reply.routing_decision.assignments[0].attempts
    assert reply.agent_results[0].status == "answered" and [item.status for item in attempts] == ["ok"]
    assert [item["repair_strategy"] for item in model.feedback] == ["medicine_status_only"]


LINKED = "头疼一周，吃过布洛芬，上周化验有一项偏高。帮我把这三件事串起来，整理成一份给医生看的说明。"


def test_linking_request_admits_every_part_it_names():
    roles = admitted(LINKED, [("symptoms", "头疼一周"), ("report", "上周化验有一项偏高"), ("visit", "整理成一份给医生看的说明")])
    assert roles == ["drug_reference", "report_explainer", "symptom_guide"]


def test_putting_two_things_together_admits_both():
    text = "报告上有一项超出了参考范围，我又在吃布洛芬，想把这两件事放在一起理清楚再问医生，资料上各怎么说？"
    roles = admitted(text, [("report", "报告上有一项超出了参考范围"), ("medicine", "我又在吃布洛芬")])
    assert roles == ["drug_reference", "report_explainer"]


def test_background_medicine_stays_background_without_a_linking_request():
    roles = admitted("我在吃布洛芬，报告上的参考范围是什么意思？", [("report", "报告上的参考范围是什么意思"), ("medicine", "我在吃布洛芬")])
    assert roles == ["report_explainer"]


@pytest.mark.parametrize("line", ["本次未查询就医准备事项，相关内容由其他安排处理。", "本次未处理下周复查准备事项，该部分不在本次整理范围内。"])
def test_part_handed_over_or_out_of_scope_is_not_repeated_beside_its_section(line):
    from medagent.agents.collaboration import without_role_boundary
    from medagent.agents.collaboration import covered_elsewhere
    results = [NS(execution=NS(role="report_explainer")), NS(execution=NS(role="visit_guide"))]
    assert line not in without_role_boundary("已解释单位。"+line, covered_elsewhere(results, results[0]))


def test_a_failed_part_is_not_described_as_handled_elsewhere():
    from medagent.agents.collaboration import reads_as_missing
    assert reads_as_missing("本次未查询就医准备事项，相关内容由其他安排处理。")
