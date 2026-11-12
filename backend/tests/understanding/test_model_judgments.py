"""Two semantic judgments come from the understanding model's verified quotes; code rules stand in only when no model judged."""
import json
from types import SimpleNamespace as NS

from medagent.core.contracts import Intent, Question
from medagent.core.provider import json_schema
from medagent.understanding.classification import IntentRouter
from medagent.understanding.conversation import role_details
from support.harness import Model, fixture, role_data, run

DENTIST = "过两天去拔牙，帮我想想要带什么"
HELD_PACK = "我手上这盒布洛芬缓释胶囊的说明书写了什么"


class Judging(Model):
    """Understanding returns the given tasks plus judgment quotes; role models record the references they receive."""
    def __init__(self, tasks, **quotes):
        self.tasks, self.quotes, self.keys = tasks, quotes, []

    async def json(self, stage, system, content, budget=900):
        if stage == "understanding":
            kinds = [kind for kind, _ in self.tasks]
            return {"scores":{kind:(.6 if index == 0 else .3) for index, kind in enumerate(kinds)}, "updates":[],
                    "topic_relation":"new", "urgent_quote":"",
                    "tasks":[{"intent":kind, "goal":"任务", "quote":quote} for kind, quote in self.tasks], **self.quotes}
        return await super().json(stage, system, content, budget)

    async def message(self, stage, system, messages, tools=None, budget=1800):
        self.keys += [ref["key"] for ref in role_data(messages)["references"]]
        return await super().message(stage, system, messages, tools, budget)


def chat(model, text):
    async def scenario():
        return await (await fixture(model)).chat(Question(text=text))
    return run(scenario())


def understand(text, **quotes):
    return run(IntentRouter(Judging([("visit", text)], **quotes)).understand(text))


def understanding(goal="", **quotes):
    return NS(entities={"medication_name":NS(value="布洛芬", state="provided")}, tasks=[NS(goal=goal)] if goal else [],
              missing_fields=[], subject_ambiguous=False, product_quote=quotes.get("product_quote"),
              visit_quote=quotes.get("visit_quote"))


def test_understanding_schema_asks_for_both_quotes():
    schema = json_schema("understanding")
    assert {"product_quote", "visit_quote"} <= set(schema["required"])


def test_a_verified_quote_is_kept_and_an_absent_one_is_unjudged():
    kept = understand(DENTIST, visit_quote="过两天去拔牙", product_quote="")
    assert kept.visit_quote == "过两天去拔牙" and kept.product_quote == ""
    unjudged = understand(DENTIST)
    assert unjudged.visit_quote is None and unjudged.product_quote is None


def test_a_quote_outside_the_turn_is_unjudged():
    result = understand(DENTIST, visit_quote="下周去医院复查")
    assert result.visit_quote is None and "unverified_visit_quote_ignored" in result.fallbacks


def test_no_model_leaves_both_judgments_to_the_rules():
    result = run(IntentRouter(None).understand(DENTIST))
    assert result.visit_quote is None and result.product_quote is None


def test_the_product_judgment_follows_the_model_over_the_wording():
    wording = "布洛芬的产品说明书一般有哪些部分"
    assert "query_type" not in role_details(understanding(wording, product_quote=""), "drug_reference")
    quoted = understanding("我手上这盒布洛芬的说明书写了什么", product_quote="这盒布洛芬的说明书")
    assert role_details(quoted, "drug_reference")["query_type"] == "product_information"
    assert role_details(understanding(wording), "drug_reference")["query_type"] == "product_information"


def test_a_named_strength_still_needs_the_label():
    facts = understanding(product_quote="")
    facts.entities["strength"] = NS(value="200mg", state="provided")
    assert role_details(facts, "drug_reference")["query_type"] == "product_information"


def test_the_visit_judgment_reaches_the_role_input():
    assert role_details(understanding(visit_quote="过两天去拔牙"), "visit_guide")["visit_context"] is True
    assert role_details(understanding(visit_quote=""), "visit_guide")["visit_context"] is False
    assert "visit_context" not in role_details(understanding(), "visit_guide")


def test_a_quoted_visit_gets_the_checklist_without_a_listed_word():
    model = Judging([("visit", "帮我想想要带什么")], visit_quote="过两天去拔牙", product_quote="")
    reply = chat(model, DENTIST)
    visit = next(item for item in reply.agent_results if item.role == "visit_guide")
    assert "visit_purpose" not in visit.missing_fields and "consultation-record" in model.keys


def test_a_model_no_overrides_a_listed_word():
    async def scenario():
        runtime = await fixture(Judging([]))
        agent = runtime.agents["visit_guide"][0]
        return await agent.run("我是医生，下周出差帮我想想要带什么", Intent.VISIT, {}, "selective", details={"visit_context":False})
    result = run(scenario())
    assert result.execution.status == "needs_clarification" and result.execution.missing_fields == ["visit_purpose"]
    assert not result.references


def test_a_quoted_product_request_waits_for_the_label_fields():
    model = Judging([("medicine", HELD_PACK)], product_quote="我手上这盒布洛芬缓释胶囊的说明书", visit_quote="")
    reply = chat(model, HELD_PACK)
    medicine = next(item for item in reply.agent_results if item.role == "drug_reference")
    assert medicine.status == "needs_clarification" and "strength" in medicine.missing_fields
    assert "ibuprofen-overview" not in model.keys


class FollowupJudging(Model):
    """The first turn quotes its visit wording; the short follow-up shows none of its own."""
    async def json(self, stage, system, content, budget=900):
        if stage == "understanding":
            current = json.loads(content)["current"]["text"]
            return {"scores":{"visit":1}, "product_quote":"", "visit_quote":"" if current.startswith("接下来") else "就医前"}
        return await super().json(stage, system, content, budget)


def test_a_recovered_visit_turn_outweighs_an_empty_quote_of_the_followup():
    async def scenario():
        clock = [1000.0]
        runtime = await fixture(FollowupJudging())
        runtime.memory.context_clock = lambda:clock[0]
        await runtime.chat(Question(text="就医前需要准备哪些资料？", user="synthetic", session="dialogue"))
        clock[0] = 1060.0
        return await runtime.chat(Question(text="接下来准备哪些资料？", user="synthetic", session="dialogue"))
    reply = run(scenario())
    visit = next(item for item in reply.agent_results if item.role == "visit_guide")
    assert visit.status == "answered" and [ref.key for ref in reply.references] == ["consultation-record"]
