import json
from collections import Counter
from dataclasses import fields
import pytest
from pydantic import ValidationError
from support.comparison import RehearsalModel
from support.harness import Model, fixture, role_data, run
from support.multiturn import ask
from medagent.comparison.cli import build_plan
from medagent.comparison.contracts import ARM_SETTINGS, ARMS, ComparisonCase, ComparisonRequest, resolve_comparison
from medagent.comparison.runner import ArchitectureIsolation, run_comparison, schedule
from medagent.comparison.single_agent import SingleAgent
from medagent.core.contracts import Question
from medagent.runtime import MedicalRuntime

MARK = "INJECTION-single-9c1"


def single(runtime, model=None):
    arm = MedicalRuntime(runtime.settings, model or runtime.model, runtime.vectors, runtime.index, runtime.memory)
    arm.answerer = SingleAgent(arm)
    return arm


class SingleModel(Model):
    def __init__(self, answer=None, tools=()):
        self.answer, self.tools, self.calls = answer, list(tools), []

    async def message(self, stage, system, messages, tools=None, budget=1800):
        self.calls.append({"stage":stage, "system":system, "data":role_data(messages), "tools":[t["name"] for t in tools or []],
                           "messages":messages})
        if self.tools:
            return {"content":[{"type":"tool_use", "id":"t1", **self.tools.pop(0)}]}
        keys = [ref["key"] for ref in role_data(messages)["references"]][:2]
        body = self.answer(keys) if self.answer else {"text":"按资料整理，个人情况请咨询药师。", "reference_keys":keys}
        return {"content":[{"type":"text", "text":json.dumps(body, ensure_ascii=False)}]}


def test_default_case_set_is_neutral_and_has_three_cases_per_category():
    version, cases = resolve_comparison(ComparisonRequest())
    assert version == "comparison-cases-1"
    assert min(Counter(case.category for case in cases).values()) >= 3
    with pytest.raises(ValidationError, match="architecture_specific_expectation"):
        ComparisonCase.model_validate({"id":"x", "category":"single_task", "target":{"message":"布洛芬", "expected":{"role":"drug_reference"}}})
    with pytest.raises(ValidationError, match="repetitions_must_rotate_every_arm_first"):
        ComparisonRequest(repetitions=4)


def test_schedule_puts_every_arm_first_equally_often_for_each_case():
    _, cases = resolve_comparison(ComparisonRequest())
    rows = schedule(ComparisonRequest(), cases)
    for case in cases:
        firsts = Counter(row["order"][0] for row in rows if row["case_id"] == case.id)
        assert firsts == {arm:1 for arm in ARMS}
    assert all(row["judge_order"] == list(reversed(row["order"])) for row in rows)
    plan = build_plan(ComparisonRequest())
    assert plan["planned_targets"] == 3*len(rows) and plan["online_status"] == "not_run"


def test_arm_runtimes_differ_from_the_base_only_by_arm_settings_or_the_answerer():
    async def scenario():
        source = await fixture(RehearsalModel(), preview=False)
        isolation = ArchitectureIsolation(source, "arms", None)
        await isolation.create()
        try:
            for arm in ARMS:
                runtime = await isolation.arm_branch(arm, None)
                changed = {f.name for f in fields(source.settings) if getattr(runtime.settings, f.name) != getattr(source.settings, f.name)}
                assert changed == set(ARM_SETTINGS[arm]) and isinstance(runtime.answerer, SingleAgent) == (arm == "single_agent")
            assert isolation.settings == source.settings
        finally:
            await isolation.cleanup()
    run(scenario())


def test_single_agent_keeps_data_out_of_the_system_prompt_and_uses_role_retrieval_filters():
    async def scenario():
        model = SingleModel()
        runtime = single(await fixture(model, preview=False), model)
        reply = await runtime.chat(Question(text=f"{MARK} 想查成人布洛芬的公开资料"))
        call = model.calls[0]
        assert call["stage"] == "agent_single_agent" and call["tools"] == ["lookup_reference"]
        assert MARK not in call["system"] and MARK in call["data"]["current_request"]
        assert "# 角色规则：drug_reference" in call["system"] and "# 角色规则：visit_guide" in call["system"]
        assert reply.primary_agent == "single_agent" and reply.agent_results[0].role == "single_agent"
        assert reply.answer.reference_keys == ["ibuprofen-overview"]
        # The pre-lookup ran under the drug role's profile: the US product label is filtered for an overview query.
        assert "advil-us-200mg-tablet-label" not in {ref.key for ref in reply.references}
    run(scenario())


def test_single_agent_tool_calls_are_limited_to_lookup_in_known_domains():
    async def scenario():
        model = SingleModel(tools=[{"name":"create_handoff_summary", "input":{}},
                                   {"name":"lookup_reference", "input":{"query":"参考范围", "domain":"report"}}])
        runtime = single(await fixture(model, preview=False), model)
        reply = await runtime.chat(Question(text="你好"))
        results = [block for message in model.calls[-1]["messages"] if message["role"] == "user" and isinstance(message["content"], list)
                   for block in message["content"]]
        assert results[0]["is_error"] is True and "documents" in results[1]["content"] and not results[1].get("is_error")
        # The report role's relevance filter also applies here: "你好" does not ask about a report.
        assert reply.references == []
    run(scenario())


@pytest.mark.parametrize("answer,error", [
    (lambda keys: {"text":"建议每次吃两片。", "reference_keys":keys}, "AgentOutputError"),
    (lambda keys: {"text":"资料如下。", "reference_keys":["made-up"]}, "AgentOutputError"),
])
def test_single_agent_answers_pass_the_same_boundary_and_citation_checks(answer, error):
    async def scenario():
        model = SingleModel(answer)
        reply = await single(await fixture(model, preview=False), model).chat(Question(text="想查成人布洛芬的公开资料"))
        attempt = reply.routing_decision.assignments[0].attempts[0]
        assert reply.degraded and attempt.error == error and "两片" not in reply.answer.text
        assert reply.routing_decision.collaboration_status == "failed"
    run(scenario())


def test_single_agent_still_short_circuits_on_critical_risk():
    async def scenario():
        model = SingleModel()
        reply = await single(await fixture(model, preview=False), model).chat(Question(text="胸口压着疼，喘不上气"))
        assert reply.primary_agent == "risk_handoff" and model.calls == []
    run(scenario())


def test_the_product_has_no_switch_that_turns_the_baseline_on():
    """The comparison attaches the baseline to a runtime; settings and environment cannot."""
    async def scenario():
        runtime = await fixture()
        assert not {"architecture", "single_agent_attempts", "single_agent_open_tools"} & {f.name for f in fields(runtime.settings)}
        assert not hasattr(runtime, "single") and not isinstance(runtime.answerer, SingleAgent)
    run(scenario())


def test_offline_rehearsal_runs_every_planned_arm_and_blocks_claims():
    async def scenario():
        return await run_comparison(await fixture(RehearsalModel(), preview=False), ComparisonRequest())
    report = run(scenario())
    assert report["complete"] and report["infrastructure_failures"] == 0 and not report["not_run"]
    assert report["mode"] == "offline" and report["conclusion"] == "not_established"
    assert {"offline_or_simulated_run", "unknown_token_usage"} <= set(report["claim_blockers"])
    assert report["isolation"]["temporary_database_removed"] and not report["cleanup_errors"]
    compound = report["statistics"]["by_category"]["compound"]
    # Dropping supporting roles shows up as partial answers and lower coverage, not as rule or infrastructure failures.
    assert compound["primary_only"]["partial_rate"] == 1.0 and compound["primary_only"]["coverage_rate"] < 1.0
    assert compound["multi_agent"]["coverage_rate"] == 1.0 and compound["primary_only"]["rule_pass_rate"] == 1.0


@pytest.mark.parametrize("text,subject", [
    ("不对，是我自己想查成人布洛芬的公开资料。", "self"),
    ("现在是我自己想查成人布洛芬的公开资料。", "self"),
    ("她还说是我自己吃的药让她担心。", "mother"),
])
def test_correcting_the_subject_back_to_self(text, subject):
    async def scenario():
        runtime = await fixture()
        await ask(runtime, "现在帮我妈妈问，她想查布洛芬。", user="u", session="s")
        assert (await ask(runtime, text, user="u", session="s")).subject == subject
    run(scenario())


def test_single_agent_runs_through_the_real_gateway_with_its_own_output_schema():
    # The pilot failed here: the fake model never asked the gateway for a schema, the real one did.
    import httpx
    from medagent.core.config import Settings
    from medagent.core.provider import ClaudeGateway
    sent = []
    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        properties = body.get("output_config", {}).get("format", {}).get("schema", {}).get("properties", {})
        data = json.loads(body["messages"][0]["content"].split("\n", 1)[1]) if "text" in properties else None
        output = ({"scores":{"symptoms":0, "medicine":1, "visit":0, "report":0, "general":0}, "tasks":[], "updates":[],
                   "topic_relation":"new"} if "scores" in properties
                  else {"query":"布洛芬"} if "query" in properties else {"keys":[]} if "keys" in properties
                  else {"text":"按资料整理，个人情况请咨询药师。", "questions":[], "next_steps":[], "limitations":[],
                        "reference_keys":[ref["key"] for ref in data["references"]][:1]})
        return httpx.Response(200, json={"content":[{"type":"text", "text":json.dumps(output, ensure_ascii=False)}],
                                         "usage":{"input_tokens":10, "output_tokens":5}, "stop_reason":"end_turn"})
    async def scenario():
        base = await fixture(preview=False)
        gateway = ClaudeGateway(Settings(api_key="fake", model="fake"), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            return await single(base, gateway).chat(Question(text="想查成人布洛芬的公开资料"))
        finally:
            await gateway.close()
    reply = run(scenario())
    # The fixture has no Chroma collection, so `degraded` is set by storage; the agent's own attempt must succeed.
    attempt = reply.routing_decision.assignments[0].attempts[0]
    assert attempt.status == "ok" and attempt.error is None and reply.routing_decision.collaboration_status == "single"
    assert reply.primary_agent == "single_agent" and reply.answer.reference_keys
    schema = [body["output_config"]["format"]["schema"] for body in sent if "text" in body["output_config"]["format"]["schema"]["properties"]][0]
    assert schema["required"] == ["text", "questions", "next_steps", "limitations", "reference_keys"] and not schema["additionalProperties"]


RANGE_QUESTION = "化验报告上的参考范围是什么意思？"


def range_source_sentence():
    from medagent.agents.checks import report_range_source_quotes
    from medagent.core.config import ROOT
    from medagent.core.contracts import Reference
    references = {row["key"]:Reference.model_validate(row) for row in json.loads((ROOT/"resources"/"references.json").read_text())}
    return report_range_source_quotes({"lab-reference-range-definition":references["lab-reference-range-definition"]})[0]["sentence"]


class ScriptedSingle(SingleModel):
    """Returns one answer per generation round, citing the range definition when it was retrieved."""
    def __init__(self, *texts):
        super().__init__()
        self.texts = list(texts)

    async def message(self, stage, system, messages, tools=None, budget=1800):
        self.calls.append({"stage":stage, "system":system, "messages":messages})
        known = [ref["key"] for ref in role_data(messages)["references"]]
        text = self.texts.pop(0)
        return {"content":[{"type":"text", "text":json.dumps({"text":text, "reference_keys":known}, ensure_ascii=False)}]}


def test_single_agent_gets_the_role_range_evidence_and_passes_with_the_exact_sentence():
    async def scenario():
        sentence = range_source_sentence()
        model = ScriptedSingle(sentence+"。")
        reply = await single(await fixture(model, preview=False), model).chat(Question(text=RANGE_QUESTION))
        assert role_data(model.calls[0]["messages"])["report_range_evidence"][0]["sentence"] == sentence
        assert "report_range_evidence" in model.calls[0]["system"] and len(model.calls) == 1
        assert reply.routing_decision.assignments[0].attempts[0].status == "ok" and sentence in reply.answer.text
    run(scenario())


def test_single_agent_range_claim_is_replaced_by_the_source_sentence_without_a_repair_round():
    async def scenario():
        model = ScriptedSingle("参考范围是统计得到的区间，大家都差不多。")
        reply = await single(await fixture(model, preview=False), model).chat(Question(text=RANGE_QUESTION))
        assert len(model.calls) == 1 and reply.routing_decision.assignments[0].attempts[0].status == "ok"
        assert range_source_sentence() in reply.answer.text and "大家都差不多" not in reply.answer.text
        assert "lab-reference-range-definition" in reply.answer.reference_keys
    run(scenario())


def test_single_agent_range_claim_is_repaired_once_with_permitted_sentences_like_the_role_path(monkeypatch):
    monkeypatch.setattr("medagent.comparison.single_agent.insert_range_sentences", lambda text, keys, references: (text, keys, 0))
    async def scenario():
        model = ScriptedSingle("参考范围是统计得到的区间，大家都差不多。", "参考范围需要由医护结合情况解释。")
        reply = await single(await fixture(model, preview=False), model).chat(Question(text=RANGE_QUESTION))
        feedback = json.loads(model.calls[1]["messages"][-1]["content"])
        assert feedback["repair_strategy"] == "report_range_evidence_only"
        assert feedback["permitted_source_sentences"][0]["sentence"] == range_source_sentence()
        assert feedback["validation"]["code"] == "report_range_claim_without_exact_source"
        assert reply.routing_decision.assignments[0].attempts[0].status == "ok" and "统计得到" not in reply.answer.text
    run(scenario())


def test_single_agent_range_claim_fails_when_the_repair_also_breaks_the_rule(monkeypatch):
    monkeypatch.setattr("medagent.comparison.single_agent.insert_range_sentences", lambda text, keys, references: (text, keys, 0))
    async def scenario():
        bad = "参考范围是统计得到的区间，大家都差不多。"
        model = ScriptedSingle(bad, bad)
        reply = await single(await fixture(model, preview=False), model).chat(Question(text=RANGE_QUESTION))
        attempt = reply.routing_decision.assignments[0].attempts[0]
        assert attempt.status == "failed" and attempt.error_code == "report_range_claim_without_exact_source"
        assert len(model.calls) == 2 and "统计得到" not in reply.answer.text
    run(scenario())


def test_single_answer_checks_match_the_role_path(monkeypatch):
    from medagent.agents.checks import VISIT_URGENCY_ATTRIBUTION_NOTICE
    from medagent.comparison.single_agent import checked_single_answer
    from medagent.agents.output_validation import AgentOutputError
    from medagent.core.config import ROOT
    from medagent.core.contracts import Reference
    rows = {row["key"]:Reference.model_validate(row) for row in json.loads((ROOT/"resources"/"references.json").read_text())}
    known = {"consultation-record":rows["consultation-record"]}
    # An explicit source request needs a citation when something was retrieved, and may say so honestly when nothing was.
    with pytest.raises(AgentOutputError, match="requested_evidence_not_returned"):
        checked_single_answer({"text":"这是通用说明。"}, known, visit_task=False, evidence_requested=True)
    assert checked_single_answer({"text":"没有查到适用资料。"}, {}, visit_task=False, evidence_requested=True).reference_keys == []
    # Unapproved references cannot be cited.
    hidden = {"consultation-record":rows["consultation-record"].model_copy(update={"approved":False})}
    with pytest.raises(AgentOutputError, match="reference_outside_role_scope"):
        checked_single_answer({"text":"这是通用说明。", "reference_keys":["consultation-record"]}, hidden, visit_task=False, evidence_requested=False)
    # Specific urgent advice outside the cited sources gets the visit role's attribution notice.
    urgent = {"text":"如果胸痛请立即就医。", "reference_keys":["consultation-record"]}
    assert VISIT_URGENCY_ATTRIBUTION_NOTICE in checked_single_answer(dict(urgent), known, visit_task=True, evidence_requested=False).limitations
    assert not checked_single_answer(dict(urgent), known, visit_task=False, evidence_requested=False).limitations
