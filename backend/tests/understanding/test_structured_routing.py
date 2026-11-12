import asyncio
from dataclasses import replace
import json
import pytest
from fastapi.testclient import TestClient
from medagent.app import create_app
from medagent.core.config import Settings
from medagent.core.contracts import Intent, Question
from medagent.core.provider import json_schema
from medagent.runtime import MedicalRuntime
from medagent.tools.registry import ToolRegistry
from support.harness import Model, fixture, role_output, run
from support.routing import COMPOUND_REQUEST, FOUR_TASK_REQUEST, field, fixed_understanding, understanding


def test_extracted_fields_do_not_reorder_roles_against_the_primary_intent():
    async def scenario():
        runtime = await fixture(Model())
        base = understanding()
        fixed_understanding(runtime, base)
        first = await runtime.chat(Question(text=COMPOUND_REQUEST))
        assert first.primary_agent == "drug_reference"
        base.entities = {name:field(name,value) for name,value in [("report_item","项目A"),("value","10"),("unit","U"),("reference_range","3-5")]}
        second = await runtime.chat(Question(text=COMPOUND_REQUEST))
        assert second.intent.primary == first.intent.primary == Intent.MEDICINE
        assert second.primary_agent == "drug_reference" and second.supporting_agents == ["report_explainer"]
        primary = next(item for item in second.routing_decision.candidates if item.role == "drug_reference")
        assert primary.components == {"intent":primary.score}
    run(scenario())


def test_slow_but_admitted_instances_do_not_reorder_roles():
    async def scenario():
        runtime = await fixture(Model())
        fixed_understanding(runtime, understanding())
        first = await runtime.chat(Question(text=COMPOUND_REQUEST))
        assert first.primary_agent == "drug_reference"
        for agent in runtime.agents["drug_reference"]:
            for _ in range(40): runtime.monitor.record(agent.instance, False, 1000)
        second = await runtime.chat(Question(text=COMPOUND_REQUEST))
        # Health excludes instances through the monitor penalty; it does not move the primary task to another role.
        assert second.primary_agent == "drug_reference"
        assert second.supporting_agents == ["report_explainer"]
    run(scenario())


def test_high_scoring_unrequested_domains_and_general_cannot_win_single_specialty_task():
    async def scenario():
        runtime = await fixture(Model())
        fixed_understanding(runtime, understanding(kinds=(Intent.MEDICINE,), scores={"medicine":.01,"report":.99,"general":1}))
        for agent in runtime.agents["drug_reference"]: runtime.monitor.record(agent.instance, False, 10000)
        reply = await runtime.chat(Question(text="查布洛芬资料"))
        assert reply.primary_agent == "drug_reference" and not reply.supporting_agents
        assert all(not candidate.eligible for candidate in reply.routing_decision.candidates if candidate.role != "drug_reference")
    run(scenario())


def test_instance_health_and_ties_control_real_attempt_order_and_snapshot():
    async def scenario():
        runtime = await fixture(Model())
        fixed_understanding(runtime, understanding(kinds=(Intent.MEDICINE,)))
        runtime.monitor.record("drug_reference.1", False, 1000)
        reply = await runtime.chat(Question(text="药品资料"))
        assignment = reply.routing_decision.assignments[0]
        assert assignment.selected_instance == "drug_reference.2"
        assert [choice.instance for choice in assignment.instances] == ["drug_reference.2","drug_reference.1"]
        assert assignment.instances[0].samples == 0 and assignment.instances[0].health == 1
        assert assignment.attempts[0].instance == "drug_reference.2"
        # Request snapshot remains unchanged by the completed call's latency sample.
        assert assignment.instances[0].samples == 0
        runtime.monitor.samples.clear()
        decision = runtime.planner.plan("药", understanding(kinds=(Intent.MEDICINE,)), runtime.agents, runtime.monitor)
        assert [choice.instance for choice in decision.assignments[0].instances] == ["drug_reference.1","drug_reference.2"]
    run(scenario())


def test_missing_tool_excludes_role_before_execution_and_does_not_select_general():
    async def scenario():
        runtime = await fixture(Model())
        runtime.planner.tools = ToolRegistry([spec for name,spec in runtime.tools.registry.specs.items() if name != "lookup_reference"])
        reply = await runtime.chat(Question(text="布洛芬公开药品资料"))
        assert reply.primary_agent == "drug_reference" and reply.degraded and not reply.agent_results
        decision = reply.routing_decision
        assert decision.routing_reason == "no_eligible_role" and not decision.assignments
        assert "required_tool_unavailable_or_out_of_scope" in next(c for c in decision.candidates if c.role == "drug_reference").reasons
        assert not any(event["event"] == "agent" for event in runtime.monitor.traces[-1]["events"])
    run(scenario())


def test_no_instance_exclusion_reports_unfinished_task_in_compound_request():
    async def scenario():
        runtime = await fixture(Model())
        fixed_understanding(runtime, understanding())
        runtime.agents["drug_reference"] = []
        reply = await runtime.chat(Question(text=COMPOUND_REQUEST))
        assert reply.primary_agent == "report_explainer" and reply.degraded
        assert reply.routing_decision.deferred_roles == ["drug_reference"]
        assert any("药品资料部分未完成" in item for item in reply.answer.limitations)
    run(scenario())


def test_report_and_medication_receive_different_tasks_fields_and_retrieval_domains():
    async def scenario():
        runtime = await fixture(Model())
        value = understanding(entities={"medication_name":field("medication_name","布洛芬"),"report_item":field("report_item","项目A")})
        fixed_understanding(runtime, value)
        seen = []
        for role in ["drug_reference","report_explainer"]:
            agent = runtime.agents[role][0]
            original = agent.run
            async def capture(text, domain, context, mode, *, _original=original, _role=role, **kwargs):
                seen.append((_role, text, domain, context["grounded_fields"], kwargs))
                return await _original(text, domain, context, mode, **kwargs)
            agent.run = capture
        reply = await runtime.chat(Question(text="查布洛芬的公开资料；整理报告项目A。"))
        assert {item.role for item in reply.agent_results} == {"drug_reference","report_explainer"}
        assert len({item[1] for item in seen}) == 2
        for role, text, domain, fields, kwargs in seen:
            assert domain == (Intent.MEDICINE if role == "drug_reference" else Intent.REPORT)
            assert set(fields) == ({"medication_name"} if role == "drug_reference" else {"report_item"})
            assert kwargs["assignment"].role == role
        assert all(item.task_id for item in reply.agent_results)
    run(scenario())


def test_primary_failure_preserves_support_as_partial_and_retries_only_same_role():
    async def scenario():
        class FailedVisit(Model):
            async def message(self, stage, *args, **kwargs):
                if stage == "agent_visit_guide": raise ConnectionError()
                return await super().message(stage, *args, **kwargs)
        runtime = await fixture(FailedVisit())
        reply = await runtime.chat(Question(text="咳嗽三天，去看医生需要准备哪些资料？"))
        assert reply.primary_agent == "visit_guide" and reply.degraded
        assert [result.role for result in reply.agent_results] == ["symptom_guide"]
        assert "就医准备部分这次没能完成" in reply.answer.text and "已完成的部分" in reply.answer.text
        decision = reply.routing_decision
        assert decision.collaboration_status == "partial"
        assert [attempt.instance for attempt in decision.assignments[0].attempts] == ["visit_guide.1","visit_guide.2"]
        assert any(item.reason == "primary_unavailable" for item in decision.fallbacks)
    run(scenario())


def test_support_failure_retains_primary_and_explicitly_reports_missing_part():
    async def scenario():
        class FailedSupport(Model):
            async def message(self, stage, *args, **kwargs):
                if stage == "agent_symptom_guide": raise ConnectionError()
                return await super().message(stage, *args, **kwargs)
        runtime = await fixture(FailedSupport())
        reply = await runtime.chat(Question(text="咳嗽三天，去看医生需要准备哪些资料？"))
        assert reply.primary_agent == "visit_guide" and reply.degraded
        assert [result.role for result in reply.agent_results] == ["visit_guide"]
        assert any("症状记录部分未完成" in item for item in reply.answer.limitations)
        assert reply.routing_decision.collaboration_status == "partial"
    run(scenario())


@pytest.mark.parametrize("selection", [None, [], ["task:visit"], ["task:visit","task:visit"], ["task:symptoms","task:visit"], ["task:visit","unknown"]])
def test_invalid_synthesis_selection_falls_back_to_validated_sections(selection):
    async def scenario():
        class BadMerge(Model):
            async def json(self, stage, *args, **kwargs):
                if stage == "synthesis":
                    if selection is None: raise ConnectionError()
                    return {"section_ids":selection}
                return await super().json(stage, *args, **kwargs)
        runtime = await fixture(BadMerge())
        reply = await runtime.chat(Question(text="咳嗽三天，去看医生需要准备哪些资料？"))
        assert reply.degraded and len(reply.agent_results) == 2
        assert "【就医准备】" in reply.answer.text and "【症状记录】" in reply.answer.text
        assert any(item.reason == "synthesis_failed" for item in reply.routing_decision.fallbacks)
        assert reply.references
    run(scenario())


def test_synthesis_cannot_add_claims_or_citations_even_with_valid_section_ids():
    async def scenario():
        class Injection(Model):
            async def json(self, stage, system, content, budget=900):
                result = await super().json(stage, system, content, budget)
                if stage == "synthesis": result.update(text="伪造的新结论", reference_keys=["invented"])
                return result
        runtime = await fixture(Injection())
        reply = await runtime.chat(Question(text="咳嗽三天，去看医生需要准备哪些资料？"))
        assert reply.degraded and "伪造的新结论" not in reply.answer.text
        assert "invented" not in reply.answer.reference_keys
        assert json_schema("synthesis")["required"] == ["section_ids", "drop_ids"]
    run(scenario())


@pytest.mark.parametrize("primary_fails", [False,True])
def test_oversized_combination_retains_bounded_primary_or_failure_notice(primary_fails):
    async def scenario():
        class LongSupport(Model):
            async def message(self, stage, *args, **kwargs):
                if stage == "agent_visit_guide" and primary_fails:
                    raise ConnectionError()
                if stage == "agent_symptom_guide":
                    data = role_output("symptom_guide")
                    data["answer"]["text"] = "记录"*3000
                    return {"content":[{"type":"text","text":json.dumps(data)}]}
                return await super().message(stage, *args, **kwargs)
        runtime = await fixture(LongSupport())
        reply = await runtime.chat(Question(text="咳嗽三天，去看医生需要准备哪些资料？"))
        assert reply.degraded and len(reply.answer.text) <= 6000
        assert reply.routing_decision.collaboration_status == "partial"
        assert any(item.reason == "composition_size_limit" for item in reply.routing_decision.fallbacks)
        if primary_fails: assert "这次没能完成可靠回答" in reply.answer.text
    run(scenario())


def test_structured_conflict_preserves_uncertainty_and_skips_model_merge():
    async def scenario():
        class WrongField(Model):
            async def message(self, stage, *args, **kwargs):
                if stage == "agent_report_explainer":
                    data = role_output("report_explainer")
                    data["payload"] = {"report_items":[{"name":"项目A","value":"20","unit":"U","reference_range":"3-5"}]}
                    return {"content":[{"type":"text","text":json.dumps(data)}]}
                return await super().message(stage, *args, **kwargs)
            async def json(self, stage, *args, **kwargs):
                if stage == "synthesis": raise AssertionError("conflict_cannot_be_resolved_by_model")
                return await super().json(stage, *args, **kwargs)
        runtime = await fixture(WrongField())
        value = understanding(primary=Intent.REPORT, entities={name:field(name,value) for name,value in [
            ("report_item","项目A"),("value","10"),("unit","U"),("reference_range","3-5"),("medication_name","布洛芬")]})
        fixed_understanding(runtime, value)
        reply = await runtime.chat(Question(text="整理两个信息任务"))
        assert reply.degraded and reply.routing_decision.collaboration_status == "conflict"
        assert reply.routing_decision.conflicts == [{"role":"report_explainer","field":"value","reason":"differs_from_user_field"}]
        assert "不一致" in reply.answer.limitations[0]
    run(scenario())


def test_support_limit_and_request_concurrency_are_enforced():
    async def scenario():
        runtime = await fixture(Model())
        fixed_understanding(runtime, understanding(kinds=(Intent.MEDICINE,Intent.REPORT,Intent.SYMPTOMS,Intent.VISIT)))
        active, peak = [0], [0]
        for agents in runtime.agents.values():
            for agent in agents:
                original = agent.run
                async def delayed(*args, _original=original, **kwargs):
                    active[0] += 1; peak[0] = max(peak[0],active[0])
                    try:
                        await asyncio.sleep(.01)
                        return await _original(*args, **kwargs)
                    finally: active[0] -= 1
                agent.run = delayed
        reply = await runtime.chat(Question(text=FOUR_TASK_REQUEST))
        assert len(reply.supporting_agents) == 2 and len(reply.routing_decision.assignments) == 3
        assert len(reply.routing_decision.deferred_roles) == 1 and reply.degraded
        assert peak[0] == 2 and active[0] == 0
    run(scenario())


def test_execution_deadline_cancels_pending_work_and_keeps_completed_support():
    async def scenario():
        runtime = await fixture(Model())
        runtime.settings = replace(runtime.settings, collaboration_timeout=.02, agent_timeout=1)
        cancelled = []
        for agent in runtime.agents["visit_guide"]:
            async def slow(*args, **kwargs):
                try: await asyncio.sleep(10)
                finally: cancelled.append(True)
            agent.run = slow
        reply = await runtime.chat(Question(text="咳嗽三天，去看医生需要准备哪些资料？"))
        assert reply.degraded and cancelled
        assert [result.role for result in reply.agent_results] == ["symptom_guide"]
        assert reply.routing_decision.assignments[0].status == "cancelled"
        assert any(item.reason == "collaboration_deadline" for item in reply.routing_decision.fallbacks)
    run(scenario())


def test_instance_timeout_is_recorded_and_next_planned_instance_runs():
    async def scenario():
        runtime = await fixture(Model())
        runtime.settings = replace(runtime.settings, agent_timeout=.01)
        async def slow(*args, **kwargs): await asyncio.sleep(10)
        runtime.agents["general"][0].run = slow
        reply = await runtime.chat(Question(text="介绍功能"))
        assert not reply.degraded
        assignment = reply.routing_decision.assignments[0]
        assert [attempt.status for attempt in assignment.attempts] == ["timeout","ok"]
        assert assignment.selected_instance == "general.2"
    run(scenario())


def test_critical_routing_ignores_health_missing_agents_and_memory():
    async def scenario():
        runtime = await fixture(Model())
        runtime.agents.clear()
        runtime.memory.available = False
        def forbidden(*args): raise AssertionError("risk_must_not_read_health")
        runtime.monitor.health = forbidden
        reply = await runtime.chat(Question(text="现在呼吸困难"))
        assert reply.primary_agent == "risk_handoff" and reply.usage["calls"] == 0 and reply.storage == "skipped"
        assert reply.routing_decision.routing_reason == "critical_risk_override"
        assert reply.routing_decision.assignments[0].selected_instance == "risk_handoff.deterministic"
    run(scenario())


def test_capability_question_does_not_activate_all_mentioned_specialties():
    async def scenario():
        runtime = await fixture(Model())
        fixed_understanding(runtime, understanding(primary=Intent.GENERAL, kinds=(Intent.GENERAL,)))
        reply = await runtime.chat(Question(text="请介绍你支持哪些功能，例如药品资料和报告整理"))
        assert reply.primary_agent == "general" and not reply.supporting_agents
    run(scenario())


def test_explicitly_declined_task_cannot_be_reactivated_by_semantic_support_score():
    async def scenario():
        runtime = await fixture(Model())
        fixed_understanding(runtime, understanding(primary=Intent.REPORT))
        reply = await runtime.chat(Question(text="不查药品，只整理报告文字。"))
        assert reply.primary_agent == "report_explainer" and not reply.supporting_agents
    run(scenario())


def test_mentioning_medication_history_does_not_automatically_create_drug_query():
    async def scenario():
        runtime = await fixture(Model())
        value = understanding(primary=Intent.SYMPTOMS, kinds=(Intent.SYMPTOMS,))
        fixed_understanding(runtime, value)
        reply = await runtime.chat(Question(text="咳嗽三天，我没有服用药物，只整理症状记录。"))
        assert reply.primary_agent == "symptom_guide" and not reply.supporting_agents
    run(scenario())


def test_request_cancellation_does_not_leave_agent_tasks_running():
    async def scenario():
        runtime = await fixture(Model())
        entered, cleaned = asyncio.Event(), []
        async def slow(*args, **kwargs):
            entered.set()
            try: await asyncio.sleep(10)
            finally: cleaned.append(True)
        runtime.agents["general"][0].run = slow
        pending = asyncio.create_task(runtime.chat(Question(text="介绍功能")))
        await entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError): await pending
        assert cleaned == [True]
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
    run(scenario())


def test_routing_trace_contains_scores_and_outcomes_without_user_text_or_entity_values():
    async def scenario():
        runtime = await fixture(Model())
        fixed_understanding(runtime, understanding(entities={"medication_name":field("medication_name","布洛芬")}))
        reply = await runtime.chat(Question(text="虚构私密标记ABC"))
        events = runtime.monitor.traces[-1]["events"]
        trace = json.dumps(events,ensure_ascii=False)
        assert "虚构私密标记ABC" not in trace and "布洛芬" not in trace
        assert any(item["event"] == "routing" and item["candidates"] for item in events)
        assert any(item["event"] == "routing_execution" and item["assignments"] for item in events)
        assert "input_text" not in reply.model_dump()["routing_decision"]["assignments"][0]
    run(scenario())


@pytest.mark.parametrize("changes", [{"supporting_agent_limit":4},{"agent_concurrency":0},{"collaboration_timeout":0},{"collaboration_timeout":float("inf")}])
def test_invalid_collaboration_settings_fail_during_construction(changes):
    async def scenario():
        original = await fixture()
        with pytest.raises(ValueError):
            MedicalRuntime(replace(original.settings, **changes), None, original.vectors, original.index, original.memory)
    run(scenario())


def test_http_preserves_public_fields_and_adds_real_execution_decision():
    app = create_app(Settings(preview=True,embeddings="hash"))
    with TestClient(app) as client:
        response = client.post("/chat",json={"text":"现在胸痛"})
        assert response.status_code == 200
        data = response.json()
        assert {"answer","references","primary_agent","supporting_agents","subject","routing_decision"} <= set(data)
        assert data["routing_decision"]["assignments"][0]["selected_instance"] == data["agent_results"][0]["instance"]
        assert "input_text" not in data["routing_decision"]["assignments"][0]


@pytest.mark.parametrize("text,roles", [
    # A resolved, past episode is context for the visit task, not a symptom task of its own.
    ("去年有过胸痛，已经恢复；现在只想整理已有就医资料。", ["visit_guide"]),
    ("之前腰疼过，早就好了，想问下复诊要带什么", ["visit_guide"]),
    # "只想…" limits support to tasks requested inside that clause.
    ("我头痛两天了，只想知道下周复诊需要准备什么。", ["visit_guide"]),
    # Current or recurring symptoms keep the symptom support.
    ("我头痛两天了，下周复诊需要准备什么？", ["visit_guide", "symptom_guide"]),
    ("去年开始头痛，现在又痛了，下周复诊需要准备什么？", ["visit_guide", "symptom_guide"]),
    ("只想整理头痛的情况和复诊要带的资料。", ["visit_guide", "symptom_guide"]),
])
def test_symptom_support_needs_a_current_symptom_inside_the_requested_scope(text, roles):
    async def scenario():
        reply = await (await fixture(Model())).chat(Question(text=text))
        assert [assignment.role for assignment in reply.routing_decision.assignments] == roles
    run(scenario())


@pytest.mark.parametrize("text,roles", [
    # A request in another clause widens "只想…"; background, negated or past mentions do not.
    ("只想整理一下头痛的情况，顺便查布洛芬说明书", {"symptom_guide", "drug_reference"}),
    ("只想整理一下头痛的情况，另外帮我查一下布洛芬的禁忌", {"symptom_guide", "drug_reference"}),
    ("只想整理一下头痛的情况，我在吃布洛芬", {"symptom_guide"}),
    ("只想整理一下头痛的情况，不用查布洛芬说明书", {"symptom_guide"}),
    ("只想整理一下头痛的情况，之前查过布洛芬说明书", {"symptom_guide"}),
])
def test_scope_keeps_tasks_explicitly_requested_in_another_clause(text, roles):
    async def scenario():
        reply = await (await fixture()).chat(Question(text=text))
        assert {assignment.role for assignment in reply.routing_decision.assignments} == roles
    run(scenario())


class TaskModel(Model):
    """Language channel that names an extra task, as the real model did in the online label run."""
    def __init__(self, primary, extra):
        self.primary, self.extra = primary, extra

    async def json(self, stage, system, content, budget=900):
        if stage == "understanding":
            return {"scores":{self.primary:.7, self.extra:.3}, "updates":[], "topic_relation":"new",
                    "tasks":[{"intent":self.primary, "goal":"主任务"}, {"intent":self.extra, "goal":"模型给出的辅助任务"}]}
        return await super().json(stage, system, content, budget)


@pytest.mark.parametrize("text,primary,extra,roles", [
    # Background mentions: the model names a task, but the turn does not ask for it.
    ("我在吃二甲双胍，想整理一下最近头晕的症状", "symptoms", "medicine", ["symptom_guide"]),
    ("复查之前要整理哪些资料", "visit", "report", ["visit_guide"]),
    # The same model task is admitted when the turn asks for it.
    ("想整理一下最近头晕的症状，顺便查一下二甲双胍的公开资料", "symptoms", "medicine", ["symptom_guide", "drug_reference"]),
])
def test_model_named_support_tasks_need_request_wording_in_the_current_turn(text, primary, extra, roles):
    async def scenario():
        model = TaskModel(primary, extra)
        reply = await (await fixture(model)).chat(Question(text=text))
        assert any(task.source == "model" for task in reply.intent.tasks)
        # Which admitted role leads is the scorer's call; this test is about admission.
        assert sorted(assignment.role for assignment in reply.routing_decision.assignments) == sorted(roles)
    run(scenario())


def merged(*sections):
    from types import SimpleNamespace
    from medagent.agents.collaboration import sections_answer
    from medagent.core.contracts import Answer
    return sections_answer([SimpleNamespace(answer=Answer(**fields), execution=SimpleNamespace(role=role, task_id="task:"+role, status="answered", missing_fields=[])) for role, fields in sections])


def test_merge_removes_role_boundary_text_from_every_visible_field():
    answer = merged(
        ("visit_guide", {"text":"已整理就医准备清单。化验参考范围解释不属于本次分配任务，未处理。具体证件需向机构核实。",
                         "limitations":["原文中的化验报告参考范围解释不属于本次分配任务，未处理。", "本清单为通用准备建议。"],
                         "next_steps":["把报告带上。", "报告解读不在本角色范围内。"]}),
        ("report_explainer", {"text":"参考范围需结合医护解释。布洛芬用药资料不属于本角色的报告阅读范围，本次未检索。",
                              "limitations":["本角色不处理布洛芬等药物资料查询。", "下周就诊资料清单未在本角色范围内单独整理。"],
                              "questions":["是否有具体项目？"]}))
    assert answer.text == "【就医准备】\n已整理就医准备清单。具体证件需向机构核实。\n\n【报告整理】\n参考范围需结合医护解释。"
    assert answer.limitations == ["本清单为通用准备建议。"]
    assert answer.next_steps == ["把报告带上。"] and answer.questions == ["是否有具体项目？"]


def test_merge_keeps_safety_advice_and_never_empties_a_section():
    answer = merged(
        ("symptom_guide", {"text":"用药问题不属于本角色任务，如胸痛加重请立即拨打急救电话。"}),
        ("drug_reference", {"text":"这部分不属于本次分配任务。"}))
    assert "立即拨打急救电话" in answer.text and "这部分不属于本次分配任务" in answer.text


def test_merge_keeps_ordinary_limitations_and_a_lone_role_is_untouched():
    limits = ["本服务不作诊断。", "本次未提供具体报告项目。", "不涉及个人诊断。"]
    answer = merged(("visit_guide", {"text":"甲。", "limitations":limits}), ("report_explainer", {"text":"乙。"}))
    assert answer.limitations == limits
    text = "布洛芬不属于本角色任务。"
    assert merged(("report_explainer", {"text":text})).text == text


def test_a_lone_role_may_still_say_what_it_did_not_handle():
    # With no other section the sentence is the user's only hint that part of the request was left out.
    limits = ["布洛芬的公开资料查询不属于本角色任务，请另行咨询药师。"]
    assert merged(("symptom_guide", {"text":"已整理头痛记录。", "limitations":limits})).limitations == limits
