"""Offline tests for medication identity when no applicable source was found."""
import json
import re

import pytest

from medagent.core.contracts import Intent
from support.harness import fixture, run
from support.multiturn import ask
from support.subject_switch import CountStages, guidance_text


def test_corrected_drug_retains_current_identity_without_unsupported_assumptions():
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        # Keep the original three-turn scenario stable as the shared catalog grows.
        runtime.index.documents = {key: ref for key, ref in runtime.index.documents.items()
                                   if ref.domain != Intent.MEDICINE or key == "ibuprofen-overview"}
        first = await ask(runtime, "查询布洛芬公开资料。")
        await ask(runtime, "这个药的来源范围呢？")
        before = len(model.stages)
        reply = await ask(runtime, "更正，不是布洛芬，是阿莫西林。")
        fact = reply.intent.entities["medication_name"]
        assert fact.value == "阿莫西林" and fact.state == "topic"
        assert fact.source.turn_id == reply.request_id and fact.source.origin == "current_user"
        result = reply.agent_results[0]
        assert result.status == "needs_clarification" and result.missing_fields == ["source"]
        identity = result.payload["medication_identity"]
        assert identity["name"] == "阿莫西林"
        assert all(value is None for key, value in identity.items() if key != "name")
        assert not result.payload["source_scope"] and not result.payload["evidence_items"]
        assert not reply.references and not reply.answer.reference_keys
        visible = guidance_text(reply)
        assert "阿莫西林" in visible and "布洛芬" not in visible
        assert "禁用" not in visible and "相互作用" not in visible
        assert "药品概览" in visible and "具体产品" in visible and "说明书" in visible
        assert "未找到不代表外部不存在" in visible
        assert "自行查找" in visible and "药师" in visible
        assert "agent_drug_reference" not in model.stages[before:]
        assert first.references[0].key == "ibuprofen-overview"
        assert any(e.get("name") == "lookup_reference" and e.get("status") == "empty_result"
                   for e in runtime.monitor.traces[-1]["events"])
    run(scenario())


@pytest.mark.parametrize("details", [
    {"medication_name":"阿莫西林"},
    {"medication_name":"阿莫西林", "formulation":"胶囊", "population":"成人",
     "product_name":"虚构品牌", "strength":"250mg", "region":"中国",
     "query_type":"product_information"},
])
def test_missing_source_guidance_matches_actual_capabilities(details):
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        runtime.index.documents = {key: ref for key, ref in runtime.index.documents.items()
                                   if ref.domain != Intent.MEDICINE or key == "ibuprofen-overview"}
        result = await runtime.agents["drug_reference"][0].run(
            "查询阿莫西林资料。", Intent.MEDICINE, {}, "selective", details=details)
        visible = "\n".join([result.answer.text, *result.answer.questions, *result.answer.next_steps])
        assert result.execution.missing_fields == ["source"]
        assert "药师" in visible and "监管" in visible and "生产企业" in visible
        assert "药品概览" in visible and "具体产品" in visible and "说明书" in visible
        assert "未找到不代表外部不存在" in visible
        if details.get("query_type") == "product_information":
            assert "匹配的具体产品说明书" in result.answer.text
        else:
            assert "匹配的药品公开概览或具体产品说明书" in result.answer.text
        assert "不会自动打开聊天中的链接" in visible
        assert "不会将粘贴的说明书原文作为已核验来源" in visible
        assert "能否提供需要核对的公开资料或原始文字" not in visible
        assert not re.search(r"(?:可|请|能否)提供[^。；\n]{0,20}(?:链接|原文)", visible)
        assert not re.search(r"(?:我|系统|本服务)(?:会|可|将|能)(?:读取|打开|访问|核验)[^。；\n]*(?:链接|网页|原文)", visible)
        assert not result.references and not result.answer.reference_keys
        assert not result.execution.payload["source_scope"] and not result.execution.payload["evidence_items"]
        assert "agent_drug_reference" not in model.stages
    run(scenario())


def test_chat_link_does_not_become_a_verified_reference():
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        runtime.index.documents = {key: ref for key, ref in runtime.index.documents.items()
                                   if ref.domain != Intent.MEDICINE or key == "ibuprofen-overview"}
        await ask(runtime, "查询阿莫西林药品资料。")
        before = len(model.stages)
        reply = await ask(runtime, "这个药的说明书链接是 https://example.org/example-label")
        assert reply.agent_results[0].missing_fields == ["source"]
        assert not reply.references and not reply.answer.reference_keys
        assert "不会自动打开聊天中的链接" in reply.answer.text
        assert "agent_drug_reference" not in model.stages[before:]
    run(scenario())


@pytest.mark.parametrize("role,intent,question,expected", [
    ("report_explainer", Intent.REPORT, "查询缺失资料。", "是否方便带原报告向医护人员核对相关问题？"),
    ("visit_guide", Intent.VISIT, "就医前需要准备哪些资料？", "是否方便通过目标机构的官方渠道自行核对就医流程？"),
])
def test_other_roles_use_role_specific_missing_source_question(role, intent, question, expected):
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        runtime.index.documents = {key: ref for key, ref in runtime.index.documents.items() if ref.domain != intent}
        result = await runtime.agents[role][0].run(question, intent, {}, "selective")
        assert result.execution.status == "needs_clarification" and result.execution.missing_fields == ["source"]
        assert result.answer.questions == [expected]
        assert not result.references and not result.answer.reference_keys
        assert "agent_" + role not in model.stages
    run(scenario())


@pytest.mark.parametrize("mode", ["selective", "full"])
def test_product_query_keeps_only_known_identity_without_inventing_source(mode):
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        agent = runtime.agents["drug_reference"][0]
        details = {"medication_name":"阿莫西林", "formulation":"胶囊", "population":"成人",
                   "product_name":"虚构品牌", "strength":"250mg", "region":"中国",
                   "query_type":"product_information"}
        result = await agent.run("核对虚构产品公开资料。", Intent.MEDICINE, {}, mode, details=details)
        assert result.execution.status == "needs_clarification"
        assert result.execution.missing_fields == ["source"]
        identity = result.execution.payload["medication_identity"]
        assert identity == {"name":details["medication_name"],
                            **{field:details[field] for field in ("formulation", "population", "product_name", "strength", "region")}}
        assert not result.execution.payload["source_scope"] and not result.execution.payload["evidence_items"]
        assert not result.references and not result.answer.reference_keys
        assert "包装" in result.answer.next_steps[0] and "规格" in result.answer.next_steps[0]
        assert "对应产品说明书" in result.answer.next_steps[0]
        assert "agent_drug_reference" not in model.stages
    run(scenario())


def test_legacy_direct_call_with_no_known_name_does_not_claim_one():
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        result = await runtime.agents["drug_reference"][0].run("查询未知药品资料。", Intent.MEDICINE, {}, "selective")
        assert result.execution.missing_fields == ["source"]
        assert all(value is None for value in result.execution.payload["medication_identity"].values())
        assert "查询药品为" not in result.answer.text and "None" not in result.answer.text
        assert not result.references and "agent_drug_reference" not in model.stages
    run(scenario())


def test_missing_source_uses_the_current_subject_not_previous_subjects_drug():
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        own = await ask(runtime, "查询布洛芬公开资料。")
        before = len(model.stages)
        mother = await ask(runtime, "现在给我妈妈查询阿莫西林公开资料。")
        assert mother.subject == "mother" and mother.intent.subject_changed
        assert mother.intent.entities["medication_name"].source.subject == "mother"
        assert mother.agent_results[0].payload["medication_identity"]["name"] == "阿莫西林"
        assert "阿莫西林" in guidance_text(mother) and "布洛芬" not in guidance_text(mother)
        assert not mother.references and "agent_drug_reference" not in model.stages[before:]
        restored = await ask(runtime, "改回我自己，这个药的资料呢？")
        assert restored.intent.entities["medication_name"].source.turn_id == own.request_id
        assert restored.intent.entities["medication_name"].value == "布洛芬"
        assert restored.references[0].key == "ibuprofen-overview"
    run(scenario())


def test_unknown_name_on_object_switch_remains_unknown_without_lookup():
    async def scenario():
        model = CountStages()
        runtime = await fixture(model)
        await ask(runtime, "查询布洛芬公开资料。")
        before = len(model.stages)
        reply = await ask(runtime, "现在给我妈妈问，这个药的资料呢？")
        result = reply.agent_results[0]
        assert result.missing_fields == ["medication_name"]
        assert all(value is None for value in result.payload["medication_identity"].values())
        assert not reply.references and "布洛芬" not in json.dumps(reply.answer.model_dump(), ensure_ascii=False)
        assert model.stages[before:] == ["understanding"]
        assert not any(event.get("name") == "lookup_reference" for event in runtime.monitor.traces[-1]["events"])
    run(scenario())
