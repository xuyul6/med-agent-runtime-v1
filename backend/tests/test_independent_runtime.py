import asyncio
import json
from dataclasses import replace
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from medagent.app import create_app
from medagent.classification import IntentRouter
from medagent.config import Settings, ROOT
from medagent.contracts import Answer, Intent, Question, Reference
from medagent.execution import CircuitUnavailable, Skills, ToolExecutor, RoleAgent, checked_answer
from medagent.memory import SessionMemory, scope_key
from medagent.observability import CURRENT, RunLog
from medagent.provider import ClaudeGateway, ModelFailure
from medagent.retrieval import ReferenceIndex, Vectorizer
from medagent.risk import RiskGate
from medagent.runtime import MedicalRuntime


def run(coroutine):
    return asyncio.run(coroutine)


class Model:
    async def json(self, stage, system, content, budget=900):
        if stage == "understanding":
            kind = "visit" if any(word in content for word in ["就医", "看医生", "挂号"]) else "medicine" if "药" in content or "布洛芬" in content else "report" if "报告" in content else "symptoms" if "咳嗽" in content else "general"
            return {"scores":{kind:1}}
        if stage == "retrieval_rewrite":
            return {"query":content}
        if stage == "retrieval_rerank":
            return {"keys":[]}
        if stage == "memory_summary":
            return {"summary":"用户在准备就诊，尚需整理已有资料。"}
        if stage == "synthesis":
            source = json.loads(content)
            return {"text":"可整理症状记录和已有资料，与医护沟通。", "reference_keys":source["allowed_references"]}
        raise RuntimeError("judge_unavailable")

    async def message(self, stage, system, messages, tools=None, budget=1800):
        keys = {"agent_visit_guide":"consultation-record", "agent_symptom_guide":"symptom-record", "agent_drug_reference":"ibuprofen-overview"}
        answer = {"text":"可按资料记录信息并向医护核实。", "reference_keys":[keys[stage]] if stage in keys and keys[stage] in system else []}
        return {"content":[{"type":"text","text":json.dumps(answer,ensure_ascii=False)}]}


async def fixture(model=None, preview=True):
    settings = Settings(preview=preview, embeddings="hash")
    vectors = Vectorizer("hash")
    index = ReferenceIndex(vectors, model)
    await index.upsert([Reference.model_validate(x) for x in json.loads((ROOT/"resources/references.json").read_text())])
    memory = SessionMemory(model=model)
    return MedicalRuntime(settings, model, vectors, index, memory)


@pytest.mark.parametrize("text,level", [
    ("我现在喘不过气", "CRITICAL"), ("没有胸痛，但现在呼吸困难", "CRITICAL"),
    ("去年胸痛，已经恢复", "ROUTINE"), ("科普一下胸痛的含义", "ROUTINE"),
    ("科普一下，不过我现在胸痛", "CRITICAL"), ("没有胸痛，也没有呼吸困难", "ROUTINE"),
    ("孕妇想查药品资料", "CAUTION"), ("突然一侧无力，口齿不清", "CRITICAL"),
])
def test_risk_qualifiers(text, level):
    assert RiskGate().screen(text,"a").level == level


def test_risk_continuity_is_scoped_and_expires():
    clock = [10]
    gate = RiskGate(lambda:clock[0])
    gate.screen("现在呼吸困难", "a")
    assert gate.screen("更严重了", "a").level == "CRITICAL"
    assert gate.screen("更严重了", "b").level == "ROUTINE"
    clock[0] = 311
    assert gate.screen("更严重了", "a").level == "ROUTINE"


def test_urgent_path_works_without_models_or_storage():
    async def scenario():
        runtime = await fixture()
        runtime.memory.available = False
        reply = await runtime.chat(Question(text="我现在胸痛，喘不过气"))
        assert reply.primary_agent == "risk_handoff"
        assert reply.intent.primary == Intent.URGENT
        assert reply.storage == "skipped"
        assert reply.usage["calls"] == 0
        assert not reply.handoff_connected
        assert not any(x["event"] == "understanding" for x in runtime.monitor.traces[-1]["events"])
    run(scenario())


def test_medicine_keyword_survives_classifier_provider_failure():
    async def scenario():
        router = IntentRouter(Vectorizer("hash"), None)
        understood = await router.understand("布洛芬药品说明书和用药禁忌")
        assert understood.primary == Intent.MEDICINE
        assert set(understood.signals) == {"keywords","embedding"}
    run(scenario())


@pytest.mark.parametrize("data,known", [
    ({"text":"参考资料。","reference_keys":["invented"]}, []),
    ({"text":"你患有肺炎。"}, []),
    ({"text":"已为你预约专家门诊。"}, []),
    ({"text":"每次服用200mg。"}, []),
])
def test_output_boundaries_reject_fabrication(data, known):
    with pytest.raises(ValueError): checked_answer(data, known)


def test_role_cannot_retrieve_another_domains_documents():
    async def scenario():
        class BadRequest(Model):
            def __init__(self): self.turn = 0; self.results = []
            async def message(self, stage, system, messages, tools=None, budget=1800):
                self.turn += 1
                if self.turn == 1:
                    return {"content":[{"type":"tool_use","id":"bad-call","name":"lookup_reference","input":{"query":"报告","domain":"report"}}]}
                self.results = messages[-1]["content"]
                return {"content":[{"type":"text","text":'{"text":"需要药师核实。"}'}]}
        model = BadRequest()
        runtime = await fixture(model)
        agent = runtime.agents["drug_reference"][0]
        await agent.run("布洛芬资料", Intent.MEDICINE, {}, "selective")
        assert model.results[0]["is_error"]
        assert all(value.domain == Intent.MEDICINE for value in (await runtime.index.search("布洛芬", "medicine")))
    run(scenario())


def test_circuit_blocks_repeated_failures_and_recovers_with_one_probe():
    async def scenario():
        now=[1.0]
        executor=ToolExecutor(clock=lambda:now[0],cooldown=5)
        async def fail(args):raise ConnectionError()
        async def ok(args):return "recovered"
        for _ in range(2):
            with pytest.raises(ConnectionError): await executor.call("lookup",{}, {"lookup"}, fail)
        with pytest.raises(CircuitUnavailable): await executor.call("lookup",{}, {"lookup"}, ok)
        now[0]=7
        assert await executor.call("lookup",{}, {"lookup"}, ok) == "recovered"
        assert "lookup" not in executor.open_until
    run(scenario())


def test_tool_timeout_is_a_failure_and_arguments_are_not_logged():
    async def scenario():
        log=RunLog()
        token=CURRENT.set(log)
        async def slow(args):await asyncio.sleep(.02)
        try:
            with pytest.raises(TimeoutError):
                await ToolExecutor(timeout=.001).call("slow",{"private":"fictional-private-text"},{"slow"},slow)
        finally:CURRENT.reset(token)
        assert "fictional-private-text" not in json.dumps(log.events)
        assert log.events[0]["status"] == "failed"
    run(scenario())


def test_memory_scope_subject_and_chronological_compression():
    async def scenario():
        memory=SessionMemory(model=Model())
        first=scope_key("a","s","self")
        other=scope_key("a","s","parent")
        for index in range(6):await memory.remember(first,f"u{index}",f"a{index}")
        state=await memory.read(first)
        assert state["summary"]
        assert [m["content"] for m in state["messages"]] == ["u4","a4","u5","a5"]
        assert not (await memory.read(other))["messages"]
    run(scenario())


def test_parallel_turns_are_not_lost():
    async def scenario():
        memory=SessionMemory()
        await asyncio.gather(*(memory.remember("s",f"u{i}",f"a{i}") for i in range(5)))
        messages=(await memory.read("s"))["messages"]
        assert len(messages)==10
        assert all(messages[i]["role"]=="user" and messages[i+1]["role"]=="assistant" for i in range(0,10,2))
    run(scenario())


def test_primary_and_support_role_collaborate_with_real_evidence_ids():
    async def scenario():
        runtime=await fixture(Model())
        reply=await runtime.chat(Question(text="咳嗽三天，去看医生需要准备哪些资料？"))
        assert reply.primary_agent == "visit_guide"
        assert reply.supporting_agents == ["symptom_guide"]
        assert not reply.degraded
        assert reply.references
        trace=runtime.monitor.traces[-1]
        assert not any("咳嗽三天" in json.dumps(event,ensure_ascii=False) for event in trace["events"])
    run(scenario())


def test_failed_primary_never_silently_switches_specialties():
    async def scenario():
        class Invalid(Model):
            async def message(self,*args,**kwargs):return {"content":[{"type":"text","text":'{"text":"资料","reference_keys":["invented"]}'}]}
        runtime=await fixture(Invalid())
        reply=await runtime.chat(Question(text="药品说明书"))
        assert reply.primary_agent == "drug_reference"
        assert reply.degraded
        events=runtime.monitor.traces[-1]["events"]
        assert len([event for event in events if event["event"]=="agent"])==2
        assert all(event["instance"].startswith("drug_reference.") for event in events if event["event"]=="agent")
    run(scenario())


def test_local_chroma_document_update_replaces_and_invalidates_cache(tmp_path):
    async def scenario():
        import chromadb
        from chromadb.config import Settings as CS
        collection=chromadb.PersistentClient(path=str(tmp_path),settings=CS(anonymized_telemetry=False)).get_or_create_collection("independent_test")
        index=ReferenceIndex(Vectorizer("hash"),None,collection)
        record=Reference(key="one",title="药品参考",domain=Intent.MEDICINE,text="药品资料概览",url="https://example.org/ref",version="1",scope="测试",approved=True,entities=["药品"])
        await index.upsert([record])
        assert len(await index.search("药品", "medicine"))==1
        await index.upsert([record.model_copy(update={"approved":False,"version":"2"})])
        assert await index.search("药品", "medicine")==[]
        assert await asyncio.to_thread(collection.count)==1
        other=ReferenceIndex(Vectorizer("hash"),None,collection)
        await other.restore()
        assert other.documents["one"].version=="2"
    run(scenario())


def test_preview_api_and_schema_errors():
    runtime=run(fixture())
    with TestClient(create_app(Settings(preview=True),runtime)) as client:
        assert client.get("/health").json()["preview"]
        assert client.post("/chat",json={"text":""}).status_code==422
        reply=client.post("/chat",json={"text":"我现在呼吸困难"}).json()
        assert reply["risk"]["level"]=="CRITICAL"
        assert client.get("/traces/"+reply["request_id"]).status_code==200
        assert client.post("/eval/run").status_code==409


def test_provider_headers_usage_and_disabled_thinking():
    async def scenario():
        import httpx
        captured=[]
        def handler(request):
            captured.append(request)
            return httpx.Response(200,json={"content":[{"type":"text","text":'{"scores":{"visit":1}}'}],"stop_reason":"end_turn","usage":{"input_tokens":7,"output_tokens":4}})
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        model=ClaudeGateway(Settings(api_key="fake-key",model="configured-model"),client)
        log=RunLog();token=CURRENT.set(log)
        try:assert (await model.json("understanding","system","input"))["scores"]["visit"]==1
        finally:CURRENT.reset(token);await model.close()
        payload=json.loads(captured[0].content)
        assert payload["thinking"]=={"type":"disabled"}
        assert "temperature" not in payload
        assert payload["output_config"]["format"]["schema"]["additionalProperties"] is False
        assert log.usage()["input_tokens"]==7
        assert "fake-key" not in json.dumps(log.events)
    run(scenario())


def test_invalid_tool_arguments_do_not_open_dependency_circuit():
    async def scenario():
        executor=ToolExecutor()
        async def invalid(args):raise PermissionError("invalid_domain")
        for _ in range(3):
            with pytest.raises(PermissionError):await executor.call("lookup",{}, {"lookup"}, invalid)
        assert "lookup" not in executor.open_until
    run(scenario())


def test_explicit_visit_information_task_overrides_symptom_label():
    async def scenario():
        class SymptomBiased(Model):
            async def json(self, stage, system, content, budget=900):
                if stage=="understanding":return {"scores":{"symptoms":1}}
                return await super().json(stage,system,content,budget)
        runtime=await fixture(SymptomBiased())
        reply=await runtime.chat(Question(text="反复头痛，想去看医生，需要整理哪些信息和资料？"))
        assert reply.primary_agent=="visit_guide"
        assert reply.supporting_agents==["symptom_guide"]
    run(scenario())


def test_medicine_retrieval_does_not_substitute_another_drug():
    async def scenario():
        runtime=await fixture()
        assert await runtime.index.search("阿莫西林说明资料", "medicine")==[]
        assert (await runtime.index.search("布洛芬公开资料", "medicine"))[0].key=="ibuprofen-overview"
    run(scenario())


def test_judge_failure_is_not_a_passing_quality_score():
    from medagent.evaluation import evaluate
    async def scenario():
        result=await evaluate(await fixture(Model()))
        assert result["passed"]==0
        assert all(row["quality"] is None and row["judge_error"] for row in result["cases"])
    run(scenario())
