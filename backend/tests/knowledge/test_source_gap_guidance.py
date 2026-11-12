"""Role-specific missing-source guidance without paid models or external access."""
import json
import re

import pytest

from medagent.core.contracts import Intent, Question
from support.harness import fixture, run
from support.subject_switch import CountStages


@pytest.mark.parametrize("role,intent,message", [
    ("report_explainer", Intent.REPORT, "报告参考范围是什么？"),
    ("visit_guide", Intent.VISIT, "就医前需要准备什么？"),
])
def test_missing_source_guidance_reaches_chat_response(role, intent, message):
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        runtime.index.documents = {key: ref for key, ref in runtime.index.documents.items() if ref.domain != intent}
        reply = await runtime.chat(Question(text=message))
        assert reply.primary_agent == role and reply.agent_results[0].missing_fields == ["source"]
        assert not reply.references and not reply.answer.reference_keys
        assert "不会自动打开聊天中的链接" in reply.answer.text
        assert "agent_" + role not in model.stages
    run(scenario())


@pytest.mark.parametrize("role,intent,message,role_terms", [
    ("report_explainer", Intent.REPORT, "了解报告参考范围的含义。", ("报告", "医护")),
    ("visit_guide", Intent.VISIT, "整理就医前的准备事项。", ("就医", "官方渠道")),
])
@pytest.mark.parametrize("mode", ["selective", "full"])
def test_missing_source_guidance_is_role_specific_and_honest(role, intent, message, role_terms, mode):
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        runtime.index.documents = {key: ref for key, ref in runtime.index.documents.items() if ref.domain != intent}
        result = await runtime.agents[role][0].run(message, intent, {}, mode)
        answer = result.answer.model_dump()
        visible = json.dumps(answer, ensure_ascii=False)
        assert result.execution.status == "needs_clarification"
        assert result.execution.missing_fields == ["source"]
        assert not result.references and not result.answer.reference_keys
        assert all(term in visible for term in role_terms)
        assert "不会自动打开聊天中的链接" in visible
        assert "粘贴的说明资料" in visible and "已核验来源" in visible
        assert "能否提供需要核对的公开资料或原始文字" not in visible
        assert not re.search(r"(?:可|请|能否)提供[^。；\n]{0,20}(?:链接|原文)", visible)
        assert not re.search(r"(?:我|系统|本服务)(?:会|可|将|能)(?:读取|打开|访问|核验)[^。；\n]*(?:链接|网页|原文)", visible)
        assert "agent_" + role not in model.stages
        assert "已为你预约" not in visible and "已转接人工" not in visible
    run(scenario())


@pytest.mark.parametrize("role,intent,details,expected", [
    ("report_explainer", Intent.REPORT, {"items":[{"name":"测试项目", "value":"7"}]}, ["unit", "reference_range"]),
    ("visit_guide", Intent.VISIT, {"institution_specific":True}, ["source"]),
])
def test_field_preflight_remains_independent_of_missing_reference(role, intent, details, expected):
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        runtime.index.documents = {key: ref for key, ref in runtime.index.documents.items() if ref.domain != intent}
        result = await runtime.agents[role][0].run("使用虚构资料。", intent, {}, "selective", details=details)
        assert result.execution.missing_fields == expected
        assert len(result.answer.questions) == len(expected)
        assert not result.references and "agent_" + role not in model.stages
        assert "不会自动打开聊天中的链接" not in result.answer.text
    run(scenario())


@pytest.mark.parametrize("role,intent,message,reference", [
    ("report_explainer", Intent.REPORT, "报告参考范围是什么？", "lab-reference-range-definition"),
    ("visit_guide", Intent.VISIT, "就医前需要准备什么？", "consultation-record"),
])
def test_existing_reference_still_supports_normal_answer(role, intent, message, reference):
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        result = await runtime.agents[role][0].run(message, intent, {}, "selective")
        assert result.execution.status == "answered" and not result.execution.missing_fields
        assert reference in [ref.key for ref in result.references]
        assert set(result.answer.reference_keys) <= {ref.key for ref in result.references}
        assert "agent_" + role in model.stages
        assert "不会自动打开聊天中的链接" not in result.answer.text
    run(scenario())


def test_missing_report_source_does_not_remove_visit_source():
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        runtime.index.documents = {key: ref for key, ref in runtime.index.documents.items() if ref.domain != Intent.REPORT}
        report = await runtime.agents["report_explainer"][0].run("报告参考范围是什么？", Intent.REPORT, {}, "selective")
        visit = await runtime.agents["visit_guide"][0].run("就医前需要准备什么？", Intent.VISIT, {}, "selective")
        assert report.execution.missing_fields == ["source"] and not report.references
        assert visit.execution.status == "answered" and "consultation-record" in visit.answer.reference_keys
        assert "agent_report_explainer" not in model.stages and "agent_visit_guide" in model.stages
    run(scenario())


def test_urgent_request_remains_deterministic_and_does_not_read_storage():
    async def scenario():
        class Forbidden:
            def __getattr__(self, name):
                raise AssertionError("urgent_path_touched_dependency:" + name)
        runtime = await fixture(CountStages())
        runtime.model = runtime.memory = runtime.index = runtime.router = Forbidden()
        reply = await runtime.chat(Question(text="现在胸痛、呼吸困难。"))
        assert reply.primary_agent == "risk_handoff" and reply.storage == "skipped"
        assert reply.usage["calls"] == 0 and not reply.handoff_connected
    run(scenario())
