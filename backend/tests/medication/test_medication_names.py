import pytest
from medagent.understanding.medications import CANONICAL, LEXICON_PATH, NAME_PATTERN, canonical, load
from support.harness import run
from support.medication_probe import probe
from support.multiturn import ContextModel, ask
from support.harness import fixture


def test_lexicon_has_unique_aliases_and_rejects_ambiguous_ones(tmp_path):
    assert len(set(CANONICAL.values())) >= 60 and canonical("Aspirin") == "阿司匹林" and canonical("扑热息痛") == "对乙酰氨基酚"
    path = tmp_path/"names.json"
    path.write_text('{"medications":[{"name":"甲","aliases":["x"]},{"name":"乙","aliases":["X"]}]}')
    with pytest.raises(ValueError, match="ambiguous_medication_alias"):
        load(path)
    assert LEXICON_PATH.exists()


@pytest.mark.parametrize("text,expected", [
    ("想查地氯雷他定的资料", ["地氯雷他定"]),
    ("左氧氟沙星说明书", ["左氧氟沙星"]),
    ("aspirin label", ["aspirin"]),
    ("folic acid 资料", ["folic acid"]),
    ("aspirinx and metforminy", []),
    ("胰岛素抵抗是什么意思", []),
    ("叶酸含量高的食物", []),
    ("胰岛素怎么保存", ["胰岛素"]),
    ("艾司奥美拉唑说明书", ["艾司奥美拉唑"]),
    ("复方对乙酰氨基酚片", []),
    ("我左手疼吃了布洛芬", ["布洛芬"]),
    ("阿莫西林克拉维酸钾片", ["阿莫西林克拉维酸钾"]),
])
def test_name_pattern_prefers_longest_whole_names_and_skips_concepts(text, expected):
    assert [match.group() for match in NAME_PATTERN.finditer(text)] == expected


def test_every_lexicon_name_is_extracted_and_routed_in_common_phrasings():
    rows = run(probe())
    misses = [row["text"] for row in rows if row["extracted"] is None or row["role"] != "drug_reference"]
    assert misses == []
    assert all(canonical(row["extracted"]) == row["canonical"] for row in rows)


@pytest.mark.parametrize("text,expected,missing", [
    ("不是布洛芬，是对乙酰氨基酚", "对乙酰氨基酚", None),
    ("华法林和阿司匹林能一起吃吗", None, "medication_name"),
    ("我妈在吃二甲双胍，想查公开资料", "二甲双胍", None),
    ("胰岛素抵抗是什么意思", None, None),
])
def test_corrections_and_multiple_drugs_never_pick_the_wrong_target(text, expected, missing):
    async def scenario():
        reply = await ask(await fixture(ContextModel()), text, user="names")
        fact = reply.intent.entities.get("medication_name")
        assert (fact.value if fact else None) == expected
        if missing:
            assert missing in reply.agent_results[0].missing_fields
        if text.startswith("胰岛素抵抗"):
            assert reply.primary_agent != "drug_reference"
    run(scenario())


def test_names_outside_the_lexicon_still_come_from_grounded_model_proposals():
    async def scenario():
        model = ContextModel(lambda payload: [{"name":"medication_name", "value":"瑞巴派特", "state":"topic",
                                               "quote":payload["current"]["text"]}])
        reply = await ask(await fixture(model), "想查瑞巴派特的公开资料", user="outside")
        assert reply.intent.entities["medication_name"].value == "瑞巴派特"
        invented = ContextModel(lambda payload: [{"name":"medication_name", "value":"瑞巴派特", "state":"topic",
                                                  "quote":payload["current"]["text"]}])
        other = await ask(await fixture(invented), "想查一个胃药的公开资料", user="invented")
        assert "medication_name" not in other.intent.entities
    run(scenario())


def test_cross_language_names_compare_as_the_same_drug():
    from medagent.understanding.medications import canonical as name
    assert name("ibuprofen") == name("布洛芬") and name(" Paracetamol ") == name("对乙酰氨基酚")
    assert name("瑞巴派特") == "瑞巴派特" and name(None) is None


def test_multiple_drugs_are_named_back_instead_of_asking_for_a_name():
    async def scenario():
        reply = await ask(await fixture(ContextModel()), "布洛芬和对乙酰氨基酚有什么区别", user="pair")
        assert reply.agent_results[0].missing_fields == ["medication_name"]
        assert "布洛芬、对乙酰氨基酚" in reply.answer.text and "先查哪一个" in reply.answer.text
        assert any("相互作用" in step for step in reply.answer.next_steps)
        single = await ask(await fixture(ContextModel()), "这个药的用量怎么看说明书", user="none")
        assert "多个药品" not in single.answer.text
    run(scenario())
