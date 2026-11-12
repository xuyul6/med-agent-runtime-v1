"""A model-named support task passes routing on its own quoted request, not on a domain word list."""
import pytest
from medagent.core.contracts import Question
from medagent.core.provider import json_schema
from support.harness import Model, fixture, run


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
        return reply, sorted(item.role for item in reply.routing_decision.assignments)
    return run(scenario())


def test_schema_requires_a_quote_on_every_task():
    assert "quote" in json_schema("understanding")["properties"]["tasks"]["items"]["required"]


# Wordings from the holdout run: the model named these tasks 18/18 times and the word list stopped all of them.
@pytest.mark.parametrize("text,tasks,roles", [
    ("想先弄明白报告上的项目和结果怎么对应，也想知道见医生那天该带些什么。",
     [("report", "想先弄明白报告上的项目和结果怎么对应"), ("visit", "也想知道见医生那天该带些什么")], ["report_explainer", "visit_guide"]),
    ("周五要去门诊，帮我列一下得准备的东西，再说说成人布洛芬的公开资料讲了什么。",
     [("visit", "帮我列一下得准备的东西"), ("medicine", "再说说成人布洛芬的公开资料讲了什么")], ["drug_reference", "visit_guide"]),
    ("这周总是没力气，想把情况记下来；另外化验单上的计量单位该怎么看？",
     [("report", "化验单上的计量单位该怎么看"), ("symptoms", "这周总是没力气，想把情况记下来")], ["report_explainer", "symptom_guide"]),
    ("查一下美国 Advil 200 mg 口服包衣片成人标签写的禁用情形，再解释化验结果为什么要结合病史来看。",
     [("report", "再解释化验结果为什么要结合病史来看"), ("medicine", "查一下美国 Advil 200 mg 口服包衣片成人标签写的禁用情形")], ["drug_reference", "report_explainer"]),
    ("胃胀断断续续两周了，帮我理一下要记哪些内容，再告诉我看门诊之前还要备点什么。",
     [("visit", "再告诉我看门诊之前还要备点什么"), ("symptoms", "帮我理一下要记哪些内容")], ["symptom_guide", "visit_guide"]),
    ("想知道看报告时项目和结果怎么对应，也想知道见医生那天该带什么。",
     [("visit", "也想知道见医生那天该带什么"), ("report", "想知道看报告时项目和结果怎么对应")], ["report_explainer", "visit_guide"]),
])
def test_quoted_request_admits_the_support_task(text, tasks, roles):
    _, actual = admitted(text, tasks)
    assert actual == roles


@pytest.mark.parametrize("text,tasks,roles", [
    # Background, not a request.
    ("我在吃二甲双胍，想整理一下最近头晕的症状", [("symptoms", "想整理一下最近头晕的症状"), ("medicine", "我在吃二甲双胍")], ["symptom_guide"]),
    # One request read as two tasks: the same words cannot justify both.
    ("复查之前要整理哪些资料", [("visit", "复查之前要整理哪些资料"), ("report", "复查之前要整理哪些资料")], ["visit_guide"]),
    ("我在吃二甲双胍，想整理一下最近头晕的症状",
     [("symptoms", "想整理一下最近头晕的症状"), ("medicine", "我在吃二甲双胍，想整理一下")], ["symptom_guide"]),
    # No quote from the model for the primary task leaves nothing to tell the two apart.
    ("帮我列一下要带的东西，再说说布洛芬的公开资料讲了什么", [("medicine", "再说说布洛芬的公开资料讲了什么")], ["drug_reference"]),
])
def test_context_or_shared_wording_stays_out(text, tasks, roles):
    _, actual = admitted(text, tasks)
    assert actual == roles


def test_declined_wording_stays_out():
    _, actual = admitted("帮我解释这个化验报告怎么读，不用查布洛芬的说明书",
                         [("report", "帮我解释这个化验报告怎么读"), ("medicine", "不用查布洛芬的说明书")])
    assert actual == ["report_explainer"]


def test_a_quote_the_user_never_wrote_is_ignored_and_recorded():
    reply, actual = admitted("想先弄明白报告上的项目怎么读",
                             [("report", "想先弄明白报告上的项目怎么读"), ("medicine", "帮我查布洛芬的资料")])
    assert actual == ["report_explainer"]
    assert "unverified_task_quote_ignored" in reply.intent.fallbacks


def test_word_list_still_admits_a_task_without_a_quote():
    _, actual = admitted("想整理一下最近头晕的症状，顺便查一下二甲双胍的公开资料", [("symptoms", ""), ("medicine", "")])
    assert actual == ["drug_reference", "symptom_guide"]
