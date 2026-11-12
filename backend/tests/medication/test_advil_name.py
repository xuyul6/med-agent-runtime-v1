"""Advil has a label in the reference set, so the brand is queried as its ingredient; nothing maps the other way."""
import re

from medagent.core.contracts import Question
from support.harness import Model, fixture, role_data, run

LABEL = "查一下美国 Advil 200 mg 口服包衣片成人标签写的禁用情形"


def chat(text):
    async def scenario():
        return await (await fixture(Model())).chat(Question(text=text))
    return run(scenario())


def test_advil_counts_as_a_named_medication_and_keeps_the_product():
    reply = chat(LABEL)
    assert reply.intent.entities["medication_name"].value == "Advil"
    assert reply.intent.entities["product_name"].value == "Advil"
    assert "medication_name" not in reply.intent.missing_fields


def test_advil_question_reaches_its_label():
    class Recording(Model):
        keys = []

        async def message(self, stage, system, messages, tools=None, budget=1800):
            self.keys = [ref["key"] for ref in role_data(messages)["references"]]
            return await super().message(stage, system, messages, tools, budget)

    async def scenario():
        model = Recording()
        await (await fixture(model)).chat(Question(text=LABEL))
        assert model.keys == ["advil-us-200mg-tablet-label"]
    run(scenario())


def test_ibuprofen_does_not_become_advil():
    assert "product_name" not in chat("想查成人布洛芬的公开资料").intent.entities


def test_missing_field_wording_is_plain():
    reply = chat("这个药有什么要注意的")
    assert reply.intent.primary.value == "medicine"
    assert not re.search("结构化|必要字段", reply.answer.text)
