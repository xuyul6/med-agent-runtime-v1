from medagent.core.contracts import Intent
from support.harness import fixture, run
from support.routing import FOUR_TASK_REQUEST, understanding

CREATE = {"intent": "appointment", "action": "create", "origin": "quote"}


def plan(text, primary, appointment, kinds=(), subject_ambiguous=False):
    runtime = run(fixture())
    value = understanding(primary=primary, kinds=(primary, *kinds)).model_copy(update={"subject_ambiguous": subject_ambiguous})
    return runtime.planner.plan(text, value, runtime.agents, runtime.monitor, appointment=appointment)


def roles(decision):
    return [assignment.role for assignment in decision.assignments]


def test_booking_only_turn():
    decision = plan("帮我挂明天上午神经内科的号", Intent.VISIT, CREATE)
    assert decision.primary_agent == "appointment" and decision.supporting_agents == []


def test_booking_with_symptoms():
    decision = plan("头疼一周了，帮我挂个号", Intent.SYMPTOMS, CREATE)
    assert decision.primary_agent == "symptom_guide" and decision.supporting_agents == ["appointment"]


def test_booking_keeps_admitted_report_task():
    decision = plan("帮我挂号，顺便解释报告里的 ALT", Intent.VISIT, CREATE, kinds=(Intent.REPORT,))
    assert decision.primary_agent == "report_explainer" and decision.supporting_agents == ["appointment"]


def test_visit_preparation_kept_with_booking():
    decision = plan("帮我挂明天的号，顺便告诉我看医生前要准备什么", Intent.VISIT, CREATE)
    assert {decision.primary_agent, *decision.supporting_agents} == {"visit_guide", "appointment"}


def test_no_request_no_dispatch():
    assert "appointment" not in roles(plan("帮我挂明天的号", Intent.VISIT, None))


def test_ambiguous_subject_no_dispatch():
    assert "appointment" not in roles(plan("帮我挂明天的号", Intent.VISIT, CREATE, subject_ambiguous=True))


def test_capacity_keeps_booking():
    text = FOUR_TASK_REQUEST + "顺便帮我挂明天神经内科的号"
    decision = plan(text, Intent.MEDICINE, CREATE, kinds=(Intent.REPORT, Intent.SYMPTOMS, Intent.VISIT))
    assert decision.supporting_agents[0] == "appointment" and decision.deferred_roles


def test_assignment_excerpt():
    runtime = run(fixture())
    value = understanding(primary=Intent.VISIT, kinds=(Intent.VISIT,)).model_copy(update={"appointment_quote": "帮我挂个号"})
    decision = runtime.planner.plan("头疼，帮我挂个号", value, runtime.agents, runtime.monitor,
                                    appointment=CREATE | {"department": "神经内科"})
    [task] = [assignment for assignment in decision.assignments if assignment.role == "appointment"]
    assert "帮我挂个号" in task.input_text and task.context_fields == ["department"]


def test_booking_keeps_institution_question():
    decision = plan("帮我挂明天上午演示总医院神经内科的号，这家医院几点开门？", Intent.VISIT, CREATE)
    assert {decision.primary_agent, *decision.supporting_agents} == {"visit_guide", "appointment"}
    assert plan("帮我挂明天上午演示总医院神经内科的号", Intent.VISIT, CREATE).supporting_agents == []
