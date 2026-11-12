"""A bounded public-source summary and evidence-preserving correction."""
import json

from medagent.core.contracts import Intent, Question
from medagent.agents.checks import report_range_source_quotes
from support.harness import Model, fixture, role_output, run, role_data


KEY = "lab-reference-range-definition"


def test_summary_is_separate_from_historical_source_and_filters_unapproved():
    async def scenario():
        runtime = await fixture()
        old = runtime.index.documents["lab-context"]
        new = runtime.index.documents[KEY]
        assert old.version == "summary-2026-10-26" and new.version == "summary-2026-10-29"
        assert new.domain == Intent.REPORT and new.approved and new.url == old.url
        assert "不用于个人诊断" in new.scope
        results = await runtime.index.search("参考范围的含义是什么", "report")
        assert KEY in {row.key for row in results}
        snippets = report_range_source_quotes({new.key:new, old.key:old})
        assert snippets == [{"reference_key":KEY,"sentence":new.text.split("。")[0]}]
        assert report_range_source_quotes({new.key:new.model_copy(update={"approved":False})}) == []
    run(scenario())


def test_correction_receives_only_retrieved_source_sentence_and_revalidates_the_quote(monkeypatch):
    # Reached when the code cannot insert the sentence itself, for example when citations are full.
    monkeypatch.setattr("medagent.agents.checks.insert_range_sentences", lambda text, keys, references: (text, keys, 0))
    class QuotingModel(Model):
        def __init__(self): self.feedback = None
        async def message(self,stage,system,messages,tools=None,budget=1800):
            if stage != "agent_report_explainer": return await super().message(stage,system,messages,tools,budget)
            is_repair = len(messages) > 1 and isinstance(messages[-1]["content"],str)
            if is_repair:
                self.feedback = json.loads(messages[-1]["content"])
                quote = next(item for item in self.feedback["permitted_source_sentences"] if item["reference_key"] == KEY)
                data = role_output("report_explainer",[KEY])
                data["answer"]["text"] = quote["sentence"]+"。"
                data["payload"]["term_explanations"][0]["text"] = quote["sentence"]+"。"
            else:
                data = role_output("report_explainer",[KEY])
                data["answer"]["text"] = "参考范围是模型自己编造的统计区间。"
            return {"content":[{"type":"text","text":json.dumps(data,ensure_ascii=False)}]}
    async def scenario():
        model = QuotingModel()
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text="报告上的参考范围与其他报告信息有什么联系？"))
        assert not reply.degraded and reply.answer.reference_keys == [KEY]
        assert reply.answer.text.rstrip("。") == runtime.index.documents[KEY].text.split("。")[0]
        assert model.feedback["permitted_source_sentences"][0]["reference_key"] == KEY
        assert any(event["event"] == "agent_output_repair" and event["status"] == "accepted"
                   for event in runtime.monitor.traces[-1]["events"])
    run(scenario())


def test_first_generation_receives_only_sentences_from_the_actual_lookup():
    class Capture(Model):
        def __init__(self): self.context = None
        async def message(self,stage,system,messages,tools=None,budget=1800):
            if stage == "agent_report_explainer":
                self.context = role_data(messages)
                assert "确需使用时" in system
            return await super().message(stage,system,messages,tools,budget)

    class OldOnly:
        def __init__(self, ref): self.ref = ref
        async def search(self, query, domain, **kwargs): return [self.ref]

    async def scenario():
        model = Capture()
        runtime = await fixture(model)
        agent = runtime.agents["report_explainer"][0]
        text = "报告上的参考范围与其他报告信息有什么联系？"
        await agent.run(text, Intent.REPORT, {}, "selective", original_request=text)
        source = runtime.index.documents[KEY]
        assert {item["reference_key"] for item in model.context["report_range_evidence"]} == {KEY}
        assert model.context["report_range_evidence"][0]["sentence"] == source.text.split("。")[0]
        agent.index = OldOnly(runtime.index.documents["lab-context"])
        await agent.run(text, Intent.REPORT, {}, "selective", original_request=text)
        assert model.context["report_range_evidence"] == []
        assert KEY not in {item["key"] for item in model.context["references"]}
    run(scenario())
