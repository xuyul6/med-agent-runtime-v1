"""Limit irrelevant range definitions without dropping requested or sole evidence."""
import json

import pytest

from medagent.core.contracts import Intent
from support.harness import Model, fixture, role_output, run, role_data


KEY = "lab-reference-range-definition"
GENERIC = "想学习怎样把化验报告中每一项的名称、结果和单位整理好，请给我通用步骤。"


class ContextCapture(Model):
    def __init__(self): self.context = None; self.input_text = None

    async def message(self, stage, system, messages, tools=None, budget=1800):
        if stage != "agent_report_explainer":
            return await super().message(stage, system, messages, tools, budget)
        self.context = role_data(messages)
        self.input_text = self.context["current_request"]
        key = self.context["references"][0]["key"]
        return {"content":[{"type":"text", "text":json.dumps(role_output("report_explainer", [key]), ensure_ascii=False)}]}


@pytest.mark.parametrize("mode", ["selective", "full"])
def test_unrequested_range_definition_is_excluded_from_first_generation(mode):
    async def scenario():
        model = ContextCapture()
        runtime = await fixture(model)
        assert KEY in runtime.index.documents
        result = await runtime.agents["report_explainer"][0].run(GENERIC, Intent.REPORT, {}, mode, original_request=GENERIC)
        assert result.execution.status == "answered"
        assert KEY not in {ref["key"] for ref in model.context["references"]}
        assert KEY not in {ref.key for ref in result.references}
        assert model.context["report_range_evidence"] == []
    run(scenario())


@pytest.mark.parametrize("text,memory", [
    ("报告上的参考范围与其他报告信息有什么联系？", {}),
    ("报告范围怎么整理？", {}),
    ("化验报告的上界是根据什么确定的？", {}),
    ("报告中项目数值的统计依据是什么？", {}),
    ("化验报告结果需要整理，是否可能生病？", {}),
    (GENERIC, {"messages":[{"role":"user", "content":"我刚提到了参考范围。"}]}),
])
def test_requested_range_or_followup_retains_the_definition(text,memory):
    class BothSources:
        def __init__(self, refs): self.refs = refs
        async def search(self, query, domain, **kwargs): return self.refs

    async def scenario():
        model = ContextCapture()
        runtime = await fixture(model)
        agent = runtime.agents["report_explainer"][0]
        agent.index = BothSources([runtime.index.documents[KEY], runtime.index.documents["lab-context"]])
        await agent.run(text, Intent.REPORT, memory, "selective", original_request=text)
        assert KEY in {ref["key"] for ref in model.context["references"]}
        assert KEY in {ref["reference_key"] for ref in model.context["report_range_evidence"]}
    run(scenario())


def test_sole_retrieved_source_is_not_discarded():
    class SingleSource:
        def __init__(self, ref): self.ref = ref
        async def search(self, query, domain, **kwargs): return [self.ref]

    async def scenario():
        model = ContextCapture()
        runtime = await fixture(model)
        agent = runtime.agents["report_explainer"][0]
        agent.index = SingleSource(runtime.index.documents[KEY])
        await agent.run(GENERIC, Intent.REPORT, {}, "selective", original_request=GENERIC)
        assert [ref["key"] for ref in model.context["references"]] == [KEY]
    run(scenario())


@pytest.mark.parametrize("text,expected_exclusion", [
    ("化验报告结果如何记录？不要分析病情。", True),
    ("化验报告结果如何记录？不需要判断个人病情。", True),
    ("化验报告结果如何记录？不要分析病情，但请判断是否有病。", False),
    ("化验报告结果如何记录？不需要诊断病情，但我想要治疗建议。", False),
    ("化验报告结果如何记录？不要分析病情，也请解释参考范围。", False),
])
def test_explicit_refusal_is_distinguished_from_an_additional_clinical_request(text,expected_exclusion):
    class BothSources:
        def __init__(self, refs): self.refs = refs
        async def search(self, query, domain, **kwargs): return self.refs

    async def scenario():
        model = ContextCapture()
        runtime = await fixture(model)
        agent = runtime.agents["report_explainer"][0]
        agent.index = BothSources([runtime.index.documents[KEY], runtime.index.documents["lab-context"]])
        result = await agent.run(text, Intent.REPORT, {}, "selective", original_request=text)
        if model.context is None:
            assert not expected_exclusion and KEY in result.answer.reference_keys
            assert not result.model_dependency_exercised
        else:
            assert (KEY not in {ref["key"] for ref in model.context["references"]}) is expected_exclusion
            assert model.input_text == text
    run(scenario())
