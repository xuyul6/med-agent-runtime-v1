"""Institution questions must reach the source guard without paid model calls."""
import pytest

from medagent.core.contracts import Intent, Question
from medagent.understanding.conversation import institution_request
from support.harness import Model, fixture, run


@pytest.mark.parametrize("text,expected", [
    ("虚构星海医院的医保政策是什么？", True),
    ("虚构晨光医院周末几点接诊？", True),
    ("虚构星海医院地址在哪？", True),
    ("虚构晨光医院的医生出诊时间？", True),
    ("请问虚构青山诊所的营业时间", True),
    ("虚构蓝天医院可以报销吗", True),
    ("解释报告参考范围，并查询虚构青山医院的地址", True),
    ("看医生前要准备什么？", False),
    ("上次去了医院，现在只要通用准备清单。", False),
    ("不查这家医院，只整理通用准备清单", False),
    ("不要查虚构晨光医院的地址，只整理通用准备", False),
    ("没有问虚构星海医院的医保政策，请整理就诊资料", False),
])
def test_institution_request_boundary(text, expected):
    assert institution_request(text) is expected


@pytest.mark.parametrize("text", [
    "虚构星海医院的医保政策是什么？",
    "虚构晨光医院周末几点接诊？",
    "虚构星海医院地址在哪？",
    "虚构晨光医院的医生出诊时间？",
    "不需要预约，请查虚构星海医院地址。",
])
def test_chat_routes_unrecognized_institution_queries_to_guard(text):
    class NoAnswerModel(Model):
        async def message(self, *args, **kwargs):
            raise AssertionError("institution_question_reached_answer_model")

    async def scenario():
        runtime = await fixture(NoAnswerModel())
        reply = await runtime.chat(Question(text=text))
        assert reply.intent.primary == Intent.VISIT
        assert reply.primary_agent == "visit_guide"
        visit = next(result for result in reply.agent_results if result.role == "visit_guide")
        assert visit.status == "needs_clarification" and visit.missing_fields
        assert not visit.payload["institution_facts"]
        assert not reply.references and not reply.answer.reference_keys

    run(scenario())


def test_mixed_report_and_institution_questions_keep_separate_tasks():
    class NoVisitAnswerModel(Model):
        async def message(self, stage, *args, **kwargs):
            assert stage != "agent_visit_guide"
            return await super().message(stage, *args, **kwargs)

    async def scenario():
        runtime = await fixture(NoVisitAnswerModel())
        reply = await runtime.chat(Question(text="解释报告参考范围，查询虚构星海医院的地址在哪里？"))
        roles = {result.role:result for result in reply.agent_results}
        assert {"visit_guide", "report_explainer"} <= roles.keys()
        assert roles["visit_guide"].status == "needs_clarification"
        assert not roles["visit_guide"].payload["institution_facts"]

    run(scenario())


@pytest.mark.parametrize("text", [
    "看医生前要准备什么？",
    "不查虚构晨光医院地址，只整理看医生前要准备什么。",
])
def test_general_preparation_is_not_blocked_by_institution_guard(text):
    async def scenario():
        runtime = await fixture(Model())
        reply = await runtime.chat(Question(text=text))
        visit = next(result for result in reply.agent_results if result.role == "visit_guide")
        assert visit.status == "answered"
        assert not {"institution", "region"} & set(visit.missing_fields)

    run(scenario())


def test_emergency_still_preempts_institution_request():
    async def scenario():
        runtime = await fixture(Model())
        reply = await runtime.chat(Question(text="我现在呼吸困难，虚构星海医院地址在哪？"))
        assert reply.primary_agent == "risk_handoff"
        assert reply.intent.primary == Intent.URGENT
        assert reply.usage["calls"] == 0

    run(scenario())
