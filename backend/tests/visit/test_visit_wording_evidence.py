"""Visit wording the routing gate accepts must also reach the general template."""
import json
import pytest

from medagent.core.config import ROOT
from medagent.core.contracts import Intent, Question, Reference
from medagent.knowledge.retrieval import ReferenceIndex, Vectorizer, has_query_overlap
from medagent.understanding.conversation import visit_query_has_context
from support.harness import Model, fixture, run


def catalog_template():
    records = json.loads((ROOT/"resources"/"references.json").read_text())
    return next(Reference.model_validate(item) for item in records if item["key"] == "consultation-record")


@pytest.mark.parametrize("text", [
    "周五要去门诊，帮我列一下得准备的东西", "再告诉我看门诊之前还要备点什么", "复诊前要整理什么",
    "就诊要带什么", "看诊时要说些什么", "问诊时该说什么", "下周复查要带什么",
])
def test_visit_wording_shares_evidence_with_the_catalog_template(text):
    assert visit_query_has_context(text)
    assert has_query_overlap(text, catalog_template())


def test_catalog_template_is_retrieved_for_the_outpatient_wording():
    async def scenario():
        index = ReferenceIndex(Vectorizer("hash"), None)
        await index.upsert([catalog_template()])
        found = await index.search("门诊准备", "visit", evidence_query="周五要去门诊，帮我列一下得准备的东西。")
        assert [ref.key for ref in found] == ["consultation-record"]

    run(scenario())


@pytest.mark.parametrize("text", ["门诊大楼几点关门外的火星", "复查一下这段代码"])
def test_visit_mapping_does_not_reach_other_domains(text):
    report = Reference(key="synthetic-report", title="报告导读", text="就医时可带上这份报告", domain=Intent.REPORT,
                       url="https://example.org/report", version="test", scope="合成测试", approved=True)
    assert not has_query_overlap(text, report)




@pytest.mark.parametrize("text", ["周五要去门诊，帮我列一下得准备的东西。", "看门诊之前要备点什么？帮我列个准备清单。"])
def test_outpatient_request_reaches_the_visit_role_with_the_template(text):
    async def scenario():
        runtime = await fixture(Model())
        reply = await runtime.chat(Question(text=text))
        assert reply.primary_agent == "visit_guide"
        assert "consultation-record" in [ref.key for ref in reply.references]

    run(scenario())
