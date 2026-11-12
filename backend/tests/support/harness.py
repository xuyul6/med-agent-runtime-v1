"""Shared offline runtime fixture, deterministic fake model and role output builders."""
import asyncio
import json
import re
from dataclasses import replace
from medagent.core.config import Settings, ROOT
from medagent.core.contracts import Intent, Reference
from medagent.memory import SessionMemory
from medagent.knowledge.retrieval import ReferenceIndex, Vectorizer
from medagent.runtime import MedicalRuntime
from medagent.understanding.medications import NAMES
from medagent.agents.contracts import Contract, GeneralInput
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.tools.contracts import EmptyArguments
from medagent.tools.registry import ToolBudget, ToolContext, ToolExecutor, ToolRegistry, ToolSpec


def run(coroutine):
    return asyncio.run(coroutine)


def role_data(messages) -> dict:
    """The JSON data object a role model receives after the fixed preface of its first user message."""
    return json.loads(messages[0]["content"].split("\n", 1)[1])


def captured_deadlines(monkeypatch) -> list:
    """Collect the deadlines a runner opens with asyncio.timeout, so a fake model can end one with expire().

    A short real deadline also has to outlast the temporary store setup before the first call, and that takes
    several times longer on a cold or slow machine: the test then fails although the runner behaved correctly."""
    opened, real = [], asyncio.timeout
    def recorded(delay):
        opened.append(real(delay))
        return opened[-1]
    monkeypatch.setattr(asyncio, "timeout", recorded)
    return opened


def expire(deadline) -> None:
    """Make an open deadline pass now; the task is cancelled at its next await."""
    deadline.reschedule(asyncio.get_running_loop().time())


class ToolValue(Contract):
    value: str


def registered_dependency(handler, name="lookup", input_contract=EmptyArguments, **options):
    spec = ToolSpec(name, "虚构依赖工具", input_contract, ToolValue, frozenset({"general"}), frozenset({Intent.GENERAL}),
                    handler, side_effect="read_external", dependency=True)
    context = ToolContext(replace(DEFAULT_REGISTRY["general"], tool_scope=(name,)), GeneralInput(text="虚构测试", intent=Intent.GENERAL))
    return ToolExecutor(ToolRegistry([spec]), **options), context, ToolBudget()


def role_output(role, keys=()):
    evidence = [{"text":"本次来源提供的信息导读。", "reference_keys":list(keys)}] if keys else []
    payloads = {
        "general":{"task_summary":"介绍医疗信息整理能力。"},
        "symptom_guide":{"symptom_summary":"用户希望整理症状信息。"},
        "drug_reference":{"medication_identity":{"name":"布洛芬"}, "source_scope":["成人公开概览"], "evidence_items":evidence},
        "report_explainer":{"term_explanations":evidence},
        "visit_guide":{"preparation_checklist":["整理已有就医资料。"]},
    }
    return {"role":role, "status":"answered", "answer":{"text":"可按资料记录信息并向医护核实。", "reference_keys":list(keys)},
            "missing_fields":[], "payload":payloads[role]}


# The fake model stands in for a working language channel: it names the domain the current turn talks about and
# falls back to the earlier user turns for a follow-up, as the real model does with history and known fields.
MODEL_DOMAINS = (
    ("visit", r"就医|看医生|挂号|医院|诊所|复诊|复查|门诊|就诊|接诊|出诊"),
    ("medicine", r"药|说明书|禁忌|" + NAMES),
    ("report", r"报告|化验|指标|参考范围|血常规"),
    ("symptoms", r"咳嗽|头痛|头晕|发烧|发热|疼|痛|不舒服|症状"),
)


def model_domain(content: str) -> str:
    try:
        payload = json.loads(content)
        turns = [payload["current"]["text"], " ".join(item["text"] for item in payload.get("history", []))]
    except (ValueError, KeyError, TypeError):
        turns = [content]
    for text in turns:
        for kind, pattern in MODEL_DOMAINS:
            if re.search(pattern, text, re.I):
                return kind
    return "general"


class Model:
    async def json(self, stage, system, content, budget=900):
        if stage == "understanding":
            return {"scores":{model_domain(content):1}}
        if stage == "retrieval_rewrite":
            return {"query":content}
        if stage == "retrieval_rerank":
            return {"keys":[]}
        if stage == "memory_summary":
            return {"summary":"用户在准备就诊，尚需整理已有资料。"}
        if stage == "synthesis":
            source = json.loads(content)
            return {"section_ids":[section["id"] for section in source["sections"]]}
        raise RuntimeError("judge_unavailable")

    async def message(self, stage, system, messages, tools=None, budget=1800):
        keys = {"agent_visit_guide":"consultation-record", "agent_symptom_guide":"symptom-record", "agent_drug_reference":"ibuprofen-overview",
                "agent_report_explainer":"lab-context"}
        cited = stage in keys and keys[stage] in {ref["key"] for ref in role_data(messages)["references"]}
        answer = role_output(stage.removeprefix("agent_"), [keys[stage]] if cited else [])
        return {"content":[{"type":"text","text":json.dumps(answer,ensure_ascii=False)}]}


async def fixture(model=None, preview=True, **runtime_options):
    settings = Settings(preview=preview, embeddings="hash")
    vectors = Vectorizer("hash")
    index = ReferenceIndex(vectors, model)
    await index.upsert([Reference.model_validate(x) for x in json.loads((ROOT/"resources/references.json").read_text())])
    memory = SessionMemory(model=model)
    return MedicalRuntime(settings, model, vectors, index, memory, **runtime_options)
