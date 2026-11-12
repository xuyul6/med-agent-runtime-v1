"""Replies speak to the user: what was not done stays visible, the split of work does not."""
import json
import re
import pytest

from medagent.agents.collaboration import plain_wording, without_role_boundary
from medagent.core.contracts import Question
from support.harness import Model, fixture, role_output, run

INTERNAL = r"本角色|本任务|主任务|辅助任务|代答|分配|其他专业或通用角色|任务范围|任务未完成"


def visible_text(answer) -> str:
    return "\n".join([answer.text, *answer.questions, *answer.next_steps, *answer.limitations])


@pytest.mark.parametrize("text,topic", [
    ("布洛芬的公开资料查询不在本次分配任务范围内，未处理。", "布洛芬的公开资料查询"),
    ("布洛芬用药资料不在本任务范围内，未作查询或回答。", "布洛芬用药资料"),
    ("化验参考范围含义与布洛芬公开资料不在本次通用准备任务内。", "化验参考范围含义与布洛芬公开资料"),
    ("报告项目与结果的对应解读不在本次就医准备范围内，需由线下医护结合实际情况说明。", "报告项目与结果的对应解读"),
    ("报告项目与结果的对应解释不在本角色的任务范围内，需由线下医护说明。", "报告项目与结果的对应解释"),
    ("另外，您提到的 Advil 200 mg 禁用情形不在本角色的报告解释范围内，本次未作回答，请查阅药品说明书或咨询药师。", "Advil 200 mg 禁用情形"),
    ("药品禁用情形不属于本角色的服务范围，未检索或回答。", "药品禁用情形"),
    ("Advil 相关用药禁忌不在本角色权限内，请咨询药师或医护。", "Advil 相关用药禁忌"),
    ("Advil 相关用药问题超出本角色范围，未检索或回答。", "Advil 相关用药问题"),
    ("关于 Advil 禁用情形的问题不属于本角色的报告整理范围，本回复未作处理。", "关于 Advil 禁用情形的问题"),
    ("用户原请求中的布洛芬公开资料查询不在本角色范围内，未执行。", "布洛芬公开资料查询"),
    ("成人布洛芬公开资料查询不在本角色任务内，本次不处理。", "成人布洛芬公开资料查询"),
])
def test_an_unhandled_part_stays_named_in_plain_words(text, topic):
    plain = plain_wording(text)
    assert topic in plain and "这次没有处理" in plain
    assert not re.search(INTERNAL, plain)
    assert not re.search(r"这次没有处理，(?:本次|本回复)?[未不]", plain)


@pytest.mark.parametrize("text,expected", [
    ("本回答不涉及复诊准备事项，该部分由其他任务处理。", "本回答不涉及复诊准备事项。"),
    ("关于报告项目与结果如何对应，属于另一部分内容，本次未处理。", "关于报告项目与结果如何对应，本次未处理。"),
    ("关于 Advil 的禁用情形，本角色不负责查询用药标签，请以药品说明书为准。", "关于 Advil 的禁用情形，这里不负责查询用药标签，请以药品说明书为准。"),
    ("机构具体证件、费用、号源和入口无法由本角色确认。", "机构具体证件、费用、号源和入口无法在这里确认。"),
    ("这周总是没力气属于症状描述，本角色不做诊断。", "这周总是没力气属于症状描述，这里不做诊断。"),
])
def test_role_as_speaker_becomes_plain(text, expected):
    assert plain_wording(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("下周就诊资料清单未在本角色范围内单独整理。", "下周就诊资料清单未在这里单独整理。"),
    ("下周就诊的具体资料清单未在本角色范围内核实，仅提供报告相关整理建议。", "下周就诊的具体资料清单未在这里核实，仅提供报告相关整理建议。"),
    ("需由线下医护或合格角色说明。", "需由线下医护说明。"),
    ("需由药品资料相关角色或线下药师核实。", "需由线下药师核实。"),
])
def test_scope_wording_inside_a_clause_still_reads_as_a_sentence(text, expected):
    assert plain_wording(text) == expected


def test_a_following_safety_limit_is_not_taken_for_a_repeat():
    assert plain_wording("这部分不在本角色范围内，不作为诊断依据。") == "这部分这次没有处理，不作为诊断依据。"


@pytest.mark.parametrize("text", [
    "该项目不在本次报告范围内。", "空腹血糖不在本次检查项目里。", "参考范围不在报告上时，请问检验机构。",
    "本模板不提供这些医院业务事实。", "胸痛请立即拨打120。", "如有不适不在本地，可就近就医。",
])
def test_statements_about_the_users_material_are_left_alone(text):
    assert plain_wording(text) == text


@pytest.mark.parametrize("text", [
    "布洛芬资料不在本任务范围内，未作查询。", "报告解读不在本次就医准备范围内。", "这部分不在本次任务范围内。",
])
def test_merging_drops_the_wider_scope_wording(text):
    assert without_role_boundary("已整理好清单。"+text+"请带上现用药品。") == "已整理好清单。请带上现用药品。"


class Speaking(Model):
    """Answers with the wording a role wrote in the holdout replies."""

    async def message(self, stage, system, messages, tools=None, budget=1800):
        reply = await super().message(stage, system, messages, tools, budget)
        data = json.loads(reply["content"][0]["text"])
        data["answer"]["text"] = "已列出准备清单。布洛芬的公开资料不在本角色任务范围内，未作查询。"
        data["answer"]["limitations"] = ["药品禁用情形不属于本角色的服务范围，未检索或回答。"]
        return {"content":[{"type":"text","text":json.dumps(data, ensure_ascii=False)}]}


def test_single_role_reply_says_what_was_left_out_in_plain_words():
    async def scenario():
        runtime = await fixture(Speaking())
        reply = await runtime.chat(Question(text="看医生前要准备哪些资料？"))
        assert reply.primary_agent == "visit_guide"
        assert "布洛芬的公开资料这次没有处理" in reply.answer.text
        assert "药品禁用情形这次没有处理。" in reply.answer.limitations
        assert not re.search(INTERNAL, visible_text(reply.answer))

    run(scenario())


def test_failed_primary_keeps_support_and_names_the_failed_part():
    class FailedVisit(Model):
        async def message(self, stage, *args, **kwargs):
            if stage == "agent_visit_guide": raise ConnectionError()
            return await super().message(stage, *args, **kwargs)

    async def scenario():
        runtime = await fixture(FailedVisit())
        reply = await runtime.chat(Question(text="咳嗽三天，去看医生需要准备哪些资料？"))
        assert reply.answer.text.startswith("就医准备部分这次没能完成可靠回答。")
        assert "以下是已完成的部分：" in reply.answer.text
        assert any("就医准备部分未完成" in item for item in reply.answer.limitations)
        assert not re.search(INTERNAL, visible_text(reply.answer))

    run(scenario())


def test_failure_with_nothing_to_keep_has_no_internal_wording():
    class Failed(Model):
        async def message(self, stage, *args, **kwargs):
            raise ConnectionError()

    async def scenario():
        runtime = await fixture(Failed())
        reply = await runtime.chat(Question(text="看医生前要准备哪些资料？"))
        assert reply.answer.text.startswith("就医准备部分这次没能完成可靠回答。")
        assert not re.search(INTERNAL, visible_text(reply.answer))

    run(scenario())
