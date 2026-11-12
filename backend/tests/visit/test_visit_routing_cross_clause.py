"""Cross-clause visit preparation wording, and visit routing that never bypasses role admission or health."""
import pytest

from medagent.core.contracts import Intent, Question
from medagent.understanding.routing import cross_clause_visit_preparation
from medagent.tools.registry import ToolRegistry
from support.harness import Model, fixture, run
from support.monitoring import bind, poison
from support.routing import field, fixed_understanding, understanding


ORIGINAL = "我最近反复头痛，想去看医生，需要整理哪些信息和资料？"


@pytest.mark.parametrize("text", [
    ORIGINAL,
    "计划就医,需要准备哪些材料？",
    "看医生前，需要整理已有报告。",
])
def test_recognizes_explicit_adjacent_visit_preparation(text):
    assert cross_clause_visit_preparation(text)


@pytest.mark.parametrize("text", [
    "不想去看医生，需要整理资料。",
    "没计划去看医生，需要整理资料。",
    "计划去看医生，不需要整理资料。",
    "计划去看医生，不是让我准备资料。",
    "计划去看医生，别整理资料。",
    "去年去看医生，需要整理资料。",
    "过去想去看医生，需要整理资料。",
    "计划看医生，需要整理旅行资料。",
    "计划看医生，需要整理签证资料。",
    "计划看医生。需要整理信息资料。",
    "计划看医生，今天休息，需要整理资料。",
    "头痛，需要整理资料。",
])
def test_does_not_link_negated_historical_unrelated_or_distant_clauses(text):
    assert not cross_clause_visit_preparation(text)


def original_understanding():
    return understanding(primary=Intent.VISIT, kinds=(Intent.VISIT, Intent.SYMPTOMS),
        entities={"symptom":field("symptom", "头痛"), "change":field("change", "反复")},
        scores={"visit":0.5177142857142857, "symptoms":0.292, "medicine":0.04057142857142858,
                "report":0.0662857142857143, "general":0.08342857142857144})


def test_original_compound_request_leads_with_the_visit_primary_and_keeps_support():
    async def scenario():
        runtime = await fixture(Model())
        fixed_understanding(runtime, original_understanding())
        reply = await runtime.chat(Question(text=ORIGINAL))
        assert reply.primary_agent == "visit_guide" and reply.supporting_agents == ["symptom_guide"]
        assert reply.routing_decision.version == "routing-4"
        assert {row.role for row in reply.agent_results} == {"visit_guide", "symptom_guide"}
        assert reply.routing_decision.collaboration_status == "merged"
        visit = next(c for c in reply.routing_decision.candidates if c.role == "visit_guide")
        symptom = next(c for c in reply.routing_decision.candidates if c.role == "symptom_guide")
        assert visit.score > symptom.score and set(visit.components) == {"intent"}
        assert not reply.degraded

    run(scenario())


def test_visit_primary_does_not_bypass_tool_admission():
    async def scenario():
        runtime = await fixture(Model())
        runtime.planner.tools = ToolRegistry([spec for name,spec in runtime.tools.registry.specs.items()
                                               if name != "build_visit_checklist"])
        decision = runtime.planner.plan(ORIGINAL, original_understanding(), runtime.agents, runtime.monitor)
        visit = next(c for c in decision.candidates if c.role == "visit_guide")
        assert not visit.eligible and "required_tool_unavailable_or_out_of_scope" in visit.reasons
        assert decision.primary_agent == "symptom_guide"

    run(scenario())


def test_visit_primary_does_not_bypass_monitor_exclusion():
    async def scenario():
        runtime = await fixture(Model())
        monitored,_ = bind(runtime)
        for agent in runtime.agents["visit_guide"]:
            poison(monitored, agent.instance)
        decision = runtime.planner.plan(ORIGINAL, original_understanding(), runtime.agents, monitored)
        visit = next(c for c in decision.candidates if c.role == "visit_guide")
        assert not visit.eligible and "monitor_penalty_threshold" in visit.reasons
        assert decision.primary_agent == "symptom_guide" and decision.monitor_fallback == ["visit_guide"]

    run(scenario())
