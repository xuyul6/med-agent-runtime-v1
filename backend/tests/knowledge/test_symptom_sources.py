"""Symptom care sources: matched by symptom name, looked up by code, and never required for the record itself."""
import json
import re

from medagent.core.config import ROOT
from medagent.core.contracts import Intent
from medagent.understanding.medications import NAMES
from support.harness import Model, fixture, role_data, run

NHS_KEYS = {"symptom-cough-nhs", "symptom-fever-adult-nhs", "symptom-headache-nhs", "symptom-sore-throat-nhs",
            "symptom-diarrhoea-vomiting-nhs"}


def catalog():
    return {record["key"]:record for record in json.loads((ROOT/"resources/references.json").read_text())}


def test_symptom_summaries_carry_provenance_and_leave_medicines_out():
    records = catalog()
    assert NHS_KEYS <= set(records)
    for key in NHS_KEYS:
        record = records[key]
        assert record["domain"] == "symptoms" and record["approved"] and record["entities"]
        assert record["provenance"]["organization"] == "NHS website (England)"
        assert "Open Government Licence v3.0" in record["provenance"]["use_conditions"]
        assert not re.search(NAMES, record["text"], re.I), key
        assert "药师" in record["text"]


def search(text):
    async def scenario():
        runtime = await fixture()
        return [ref.key for ref in await runtime.index.search(text, Intent.SYMPTOMS.value, evidence_query=text)]
    return run(scenario())


def test_a_symptom_overview_is_found_only_for_its_own_symptom():
    assert "symptom-cough-nhs" in search("这两天咳嗽，晚上更厉害，需要注意什么？")
    assert "symptom-headache-nhs" not in search("这两天咳嗽，晚上更厉害，需要注意什么？")
    assert "symptom-headache-nhs" in search("头疼了一下午怎么办")
    assert "symptom-sore-throat-nhs" in search("嗓子疼，吞东西难受")
    assert not NHS_KEYS & set(search("最近总是失眠，需要注意什么？"))


class Recording(Model):
    def __init__(self):
        self.references = []

    async def message(self, stage, system, messages, tools=None, budget=1800):
        if stage == "agent_symptom_guide":
            self.references.append([ref["key"] for ref in role_data(messages)["references"]])
        return await super().message(stage, system, messages, tools, budget)


def test_symptom_role_looks_up_sources_before_the_model_writes():
    async def scenario():
        model = Recording()
        runtime = await fixture(model)
        await runtime.agents["symptom_guide"][0].run("这两天咳嗽，晚上更厉害，需要注意什么？", Intent.SYMPTOMS, {}, "selective")
        assert "symptom-cough-nhs" in model.references[-1]
    run(scenario())


def test_a_follow_up_without_the_symptom_word_uses_the_known_symptom():
    async def scenario():
        model = Recording()
        runtime = await fixture(model)
        await runtime.agents["symptom_guide"][0].run("有痰，没有其他不适", Intent.SYMPTOMS, {}, "selective",
                                                     details={"symptom":"咳嗽"})
        assert "symptom-cough-nhs" in model.references[-1]
    run(scenario())


def test_a_failed_lookup_still_leaves_the_record():
    async def scenario():
        runtime = await fixture(Model())
        async def broken(*args, **kwargs):
            raise TimeoutError("index down")
        runtime.index.search = broken
        result = await runtime.agents["symptom_guide"][0].run("这两天咳嗽，需要注意什么？", Intent.SYMPTOMS, {}, "selective")
        assert result.execution.status == "answered"
        assert result.answer.reference_keys == []
    run(scenario())


def test_symptom_summaries_pass_the_answer_boundary_checks():
    from medagent.agents.checks import check_boundaries
    for key in NHS_KEYS:
        check_boundaries({"text":catalog()[key]["text"]})


def test_a_denied_symptom_does_not_pull_in_its_overview():
    found = search("这两天咳嗽，晚上更厉害，没有发烧，需要注意什么？")
    assert "symptom-cough-nhs" in found and "symptom-fever-adult-nhs" not in found
    assert "symptom-fever-adult-nhs" in search("发烧一天了，没有咳嗽")
    assert "symptom-cough-nhs" not in search("发烧一天了，没有咳嗽")
    assert "symptom-cough-nhs" in search("这两天不停地咳嗽")
    assert "symptom-cough-nhs" in search("没有其他不适 咳嗽")


def test_the_role_task_text_does_not_decide_which_overview_applies():
    from medagent.understanding.routing import GOALS
    async def scenario():
        model = Recording()
        runtime = await fixture(model)
        text = "昨天开始拉肚子，一天四五次，需要注意什么？"
        await runtime.agents["symptom_guide"][0].run(text, Intent.SYMPTOMS, {}, "selective",
            reference_query=GOALS["symptom_guide"]+"\n用户相关原文：\n"+text, original_request=text)
        assert "symptom-fever-adult-nhs" not in model.references[-1]
        assert "symptom-diarrhoea-vomiting-nhs" in model.references[-1]
    run(scenario())


def test_the_symptom_task_text_does_not_suggest_a_fever_question():
    from medagent.understanding.routing import GOALS
    assert "发烧" not in GOALS["symptom_guide"]
