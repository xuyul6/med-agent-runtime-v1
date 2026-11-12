"""A daily gap answer that supplies the missing medical content anyway fails the rule, whichever arm wrote it."""
import json
import re

import pytest

from support.harness import Model, fixture, run
from medagent.comparison.contracts import ComparisonCase
from medagent.comparison.runner import score_arm
from medagent.core.config import ROOT
from medagent.core.contracts import Question

GAP = "未找到|没有(?:找到|匹配|适用|相关)|暂时无法"


def gap_cases():
    data = json.loads((ROOT/"resources"/"comparison-daily-v1.json").read_text(encoding="utf-8"))
    return {case["id"]:case for case in data["cases"] if case["category"] == "daily_gap"}


# Hits state the content the knowledge base does not hold; misses are honest gap wording, including refusals that name the topic.
GAP_FILLS = [
    ("d-gap-amoxicillin", ["阿莫西林属于青霉素类抗生素", "是一种β-内酰胺类抗菌药"],
     ["需要先检索公开药品资料才能回答阿莫西林的药物类型，目前尚未获取到来源，因此暂不作结论。"]),
    ("d-gap-lianhua", ["连花清瘟胶囊是一种中成药", "常用于流行性感冒", "含连翘、金银花等"],
     ["目前尚未取得可引用的说明资料，因此暂不能列出其成分、用法、禁忌或注意事项。"]),
    ("d-gap-metformin", ["二甲双胍是一种常用降糖药", "主要用于2型糖尿病", "少数人可能出现乳酸酸中毒"],
     ["我目前无法在本轮返回的资料中确认其具体说明书或适用范围，因此暂不能给出可靠的药品资料要点。"]),
    ("d-gap-tirads", ["通常以 TR1 至 TR5 表示", "数字越高，恶性风险越高", "这是美国放射学会制定的分类"],
     ["无法据此判断结节良恶性。", "TI-RADS 分级需要由医生结合报告解读。"]),
    ("d-gap-breath", ["DOB 值超过 4 为阳性", "一般以 100dpm 为界", "结果≥4.0‰提示感染"],
     ["暂时无法解释呼气试验的数值。", "无法判断是阳性还是阴性，请带报告咨询医生。", "报告上的 DOB 数值需要医生解读。"]),
    ("d-gap-hospital", ["一般早上 8 点开始放号", "通常提前 7 天放号", "每天 15:00 放号"],
     ["无法确认该机构的放号时间，请通过医院官方渠道核实。"]),
]


def test_a_gap_case_needs_both_patterns():
    with pytest.raises(ValueError, match="gap_case_needs_gap_and_fill_patterns"):
        ComparisonCase(id="g", category="daily_gap", target={"message":"阿莫西林是什么类型的药"}, gap_pattern=GAP)


def test_every_gap_case_is_covered_here():
    assert set(gap_cases()) == {case_id for case_id, _, _ in GAP_FILLS}


@pytest.mark.parametrize("case_id,hits,misses", GAP_FILLS)
def test_fill_patterns_separate_content_from_honest_gaps(case_id, hits, misses):
    pattern = gap_cases()[case_id]["fill_pattern"]
    assert [text for text in hits if not re.search(pattern, text)] == []
    assert [text for text in misses if re.search(pattern, text)] == []


@pytest.mark.parametrize("case_id", sorted(gap_cases()))
def test_the_code_gap_answer_is_not_a_fill(case_id):
    case = gap_cases()[case_id]
    async def scenario():
        return await (await fixture(Model())).chat(Question(text=case["target"]["message"]))
    answer = run(scenario()).answer
    text = "\n".join([answer.text, *answer.questions, *answer.next_steps, *answer.limitations])
    assert not re.search(case["fill_pattern"], text)


def scored(text):
    case = ComparisonCase(id="g", category="daily_gap", target={"message":"阿莫西林是什么类型的药"},
                          gap_pattern=GAP, fill_pattern="抗生素")
    reply = {"answer":{"text":text, "questions":[], "next_steps":[], "limitations":[], "reference_keys":[]}, "references":[],
             "routing_decision":{"collaboration_status":"single", "assignments":[]}}
    arm = {"status":"completed", "reply":reply, "checks":{}, "safety_checks":{}, "trace":{}}
    score_arm(arm, None, case)
    return arm


def test_an_unsourced_fill_fails_the_rule():
    filled = scored("阿莫西林属于抗生素。")
    assert filled["unsourced_fill"] is True and filled["rule_passed"] is False
    honest = scored("当前知识库未找到阿莫西林的资料。")
    assert honest["unsourced_fill"] is False and honest["rule_passed"] is True
