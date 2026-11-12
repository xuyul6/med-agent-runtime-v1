"""Synthetic output faults, never a replay or quality score of the lost real drafts."""
import json
import pytest
from medagent.agents.contracts import MedicationOutput
from medagent.core.contracts import Intent, Question
from medagent.evaluation.runner import recommendations_for
from medagent.evaluation.budget import BudgetedModel, EvaluationBudgetExceeded, ModelBudget
from medagent.evaluation.contracts import EvalLimits
from medagent.agents.execution import model_blocks
from medagent.agents.checks import checked_role_output, validate_draft
from medagent.knowledge.medication_evidence import query_identity, validate_output_identity
from medagent.core.observability import CURRENT, RunLog
from medagent.agents.output_validation import AgentOutputError
from medagent.agents.profiles import DEFAULT_REGISTRY
from support.harness import Model, fixture, role_output, run, role_data
from support.medication import SCOPED, references, request


SECRET = "SYNTHETIC_PRIVATE_SENTINEL_NEVER_LOG"


def draft(messages):
    context = role_data(messages)
    inputs = role_data(messages)["validated_input"]
    ref = context["references"][0]
    output = role_output("drug_reference",[ref["key"]])
    output["payload"]["medication_identity"] = {"name":inputs.get("medication_name"),
        **{key:inputs.get(key) for key in ("formulation","population","product_name","strength","region")}}
    output["payload"]["source_scope"] = [ref["scope"]]
    return MedicationOutput.model_validate(output).model_dump(mode="json")


class DraftModel(Model):
    def __init__(self, fault=None, *, persistent=False, tool_rounds=0, tool_on_repair=False):
        self.fault,self.persistent,self.tool_rounds,self.tool_on_repair = fault,persistent,tool_rounds,tool_on_repair
        self.calls,self.feedback,self.tool_flags = 0,[],[]

    async def message(self,stage,system,messages,tools=None,budget=1800):
        self.calls += 1
        self.tool_flags.append(tools is not None)
        if self.calls <= self.tool_rounds or self.tool_on_repair and self.calls > 1:
            return {"content":[{"type":"tool_use","id":"scope-call","name":"check_medication_scope","input":{}}]}
        is_repair = len(messages) > 1 and isinstance(messages[-1]["content"],str)
        if is_repair: self.feedback.append(json.loads(messages[-1]["content"]))
        data = draft(messages)
        if self.fault and (not is_repair or self.persistent):
            data = self.fault(data)
        text = data if isinstance(data,str) else json.dumps(data,ensure_ascii=False)
        return {"content":[{"type":"text","text":text}]}


def source_scope_as_query(data):
    data["payload"]["medication_identity"].update(population="成人",region="英国")
    return data


def test_source_metadata_cannot_fill_unknown_query_fields_and_can_be_corrected_once():
    async def scenario():
        model = DraftModel(source_scope_as_query)
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text="查询布洛芬药品资料"))
        assert not reply.degraded and model.calls == 2
        identity = reply.agent_results[0].payload["medication_identity"]
        assert identity["name"] == "布洛芬" and identity["population"] is None and identity["region"] is None
        assert reply.references[0].medication_scope.region == "UK"
        assert model.feedback[0]["validation"] == {"code":"unconfirmed_medication_identity","stage":"identity","fields":["population"],"repairable":True}
        assert model.feedback[0]["medication_identity"]["population"] is None
        assert model.tool_flags == [True,False]
        events = runtime.monitor.traces[-1]["events"]
        assert any(event["event"] == "agent_output_repair" and event["status"] == "accepted" for event in events)
        assert len(reply.routing_decision.assignments[0].attempts) == 1
    run(scenario())


def test_repeated_invalid_identity_still_fails_with_safe_structured_diagnostics():
    async def scenario():
        model = DraftModel(source_scope_as_query,persistent=True)
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text="查询布洛芬药品资料"))
        assert reply.degraded and model.calls == 4
        assert not reply.answer.reference_keys
        attempts = reply.routing_decision.assignments[0].attempts
        assert len(attempts) == 2 and all(attempt.error_code == "unconfirmed_medication_identity" and attempt.error_stage == "identity" for attempt in attempts)
        stats = runtime.monitor.snapshot()["agent_stats"]["drug_reference.1"]
        assert stats["error_counts"].get("contract_error") == 1
        rows = [{"id":"synthetic","passed":False,"turns":[{"turn":1,"checks":{"degraded":False},"trace":runtime.monitor.traces[-1]}]}]
        suggestions = recommendations_for(rows)
        assert any("output_validation:identity:unconfirmed_medication_identity" in row["evidence"] for row in suggestions)
    run(scenario())


@pytest.mark.parametrize("field,rendered", [("region","美国（US）"),("population","成人（18岁及以上）"),
    ("formulation","口服包衣片（coated tablet）"),("strength","200毫克"),("product_name","advil")])
def test_whole_alias_renderings_normalize_to_confirmed_input_only(field, rendered):
    data = {"payload":{"medication_identity":query_identity(request())}}
    data["payload"]["medication_identity"][field] = rendered
    changed = validate_output_identity(data,request(),references().values())
    assert data["payload"]["medication_identity"][field] == SCOPED[field] and field in changed


@pytest.mark.parametrize("field,value", [("region","美国（中国）"),("population","成人及儿童"),
    ("formulation","缓释胶囊"),("strength","200 mg，每日一次"),("product_name","Advil Plus")])
def test_matching_never_accepts_partial_aliases_or_expanded_scope(field,value):
    data = {"payload":{"medication_identity":{field:value}}}
    with pytest.raises(AgentOutputError) as error: validate_output_identity(data,request(),references().values())
    assert error.value.code == "unconfirmed_medication_identity" and error.value.fields == (field,)


@pytest.mark.parametrize("marker", ["未知","不详","未提供","未明确","不清楚","unknown","unspecified"])
def test_unknown_markers_remain_null_in_unprovided_query_fields(marker):
    data = {"payload":{"medication_identity":{"region":marker}}}
    validate_output_identity(data,request(region=None),references().values())
    assert data["payload"]["medication_identity"]["region"] is None
    data["payload"]["medication_identity"]["region"] = "英国"
    with pytest.raises(AgentOutputError): validate_output_identity(data,request(region=None),references().values())


def test_bilingual_name_must_consist_entirely_of_same_reference_aliases():
    data = {"payload":{"medication_identity":{"name":"布洛芬（Ibuprofen）"}}}
    validate_output_identity(data,request(),references().values())
    assert data["payload"]["medication_identity"]["name"] == "布洛芬"
    data["payload"]["medication_identity"]["name"] = "布洛芬（阿莫西林）"
    with pytest.raises(AgentOutputError,match="medication_identity_target_mismatch"):
        validate_output_identity(data,request(),references().values())


def test_only_trusted_positive_input_is_copied_into_missing_model_metadata():
    data = {"payload":{"medication_identity":{"name":"布洛芬"}}}
    validate_output_identity(data,request(),references().values())
    assert data["payload"]["medication_identity"]["region"] == "美国"
    unknown = {"payload":{"medication_identity":{"name":"布洛芬","region":None}}}
    validate_output_identity(unknown,request(region=None),references().values())
    assert unknown["payload"]["medication_identity"]["region"] is None


def test_invalid_json_can_be_corrected_without_persisting_raw_draft_or_private_values():
    async def scenario():
        model = DraftModel(lambda _:"{not-json:"+SECRET)
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text="查询布洛芬药品资料"))
        assert not reply.degraded and model.calls == 2
        assert model.feedback[0]["validation"]["code"] == "invalid_output_json"
        public = json.dumps({"trace":runtime.monitor.traces[-1],"reply":reply.model_dump(mode="json")},ensure_ascii=False)
        assert SECRET not in public
        assert model.feedback[0]["validation"]["fields"] == []
    run(scenario())


@pytest.mark.parametrize("fault,code", [("boundary","medical_boundary_violation"),("dose","individual_dosing_not_supported"),
    ("references","unsupported_reference"),("payload_refs","payload_references_not_in_answer"),("schema","invalid_output_shape")])
def test_evidence_boundary_and_shape_failures_are_never_relaxed_for_repair(fault,code):
    def corrupt(data):
        if fault == "boundary": data["answer"]["text"] = "你患有某种疾病。"
        elif fault == "dose": data["answer"]["text"] = "建议每次200mg。"
        elif fault == "references": data["answer"]["reference_keys"] = [SECRET]
        elif fault == "payload_refs": data["answer"]["reference_keys"] = []
        else: data["payload"][SECRET] = SECRET
        return data
    async def scenario():
        model = DraftModel(corrupt,persistent=True)
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text="查询布洛芬药品资料"))
        assert reply.degraded and model.calls == 2 and not model.feedback
        assert all(attempt.error_code == code for attempt in reply.routing_decision.assignments[0].attempts)
        assert SECRET not in json.dumps(runtime.monitor.traces[-1],ensure_ascii=False)
    run(scenario())


def test_invalid_json_with_medical_boundary_content_is_not_sent_for_repair():
    async def scenario():
        model = DraftModel(lambda _:"{你患有某病",persistent=True)
        runtime = await fixture(model)
        reply = await runtime.chat(Question(text="查询布洛芬药品资料"))
        assert reply.degraded and model.calls == 2 and not model.feedback
        assert reply.routing_decision.assignments[0].attempts[0].error_code == "medical_boundary_violation"
    run(scenario())


def test_repair_cannot_execute_a_tool_even_if_provider_returns_one():
    async def scenario():
        model = DraftModel(source_scope_as_query,tool_on_repair=True)
        runtime = await fixture(model)
        agent = runtime.agents["drug_reference"][0]
        log = RunLog();token = CURRENT.set(log)
        try:
            with pytest.raises(AgentOutputError,match="unexpected_tools_during_repair"):
                await agent.run("查询布洛芬药品资料",Intent.MEDICINE,{},"selective",details={"medication_name":"布洛芬"})
        finally: CURRENT.reset(token)
        assert model.calls == 2
        assert len([event for event in log.events if event["event"] == "tool"]) == 3
        assert model.tool_flags[-1] is False
    run(scenario())


@pytest.mark.parametrize("tools,expected_calls,recovers", [(1,3,True),(2,3,False)])
def test_correction_uses_existing_three_model_rounds(tools,expected_calls,recovers):
    async def scenario():
        model = DraftModel(source_scope_as_query,tool_rounds=tools)
        runtime = await fixture(model)
        agent = runtime.agents["drug_reference"][0]
        if recovers:
            result = await agent.run("查询布洛芬药品资料",Intent.MEDICINE,{},"selective",details={"medication_name":"布洛芬"})
            assert result.execution.status == "answered"
        else:
            with pytest.raises(AgentOutputError,match="unconfirmed_medication_identity"):
                await agent.run("查询布洛芬药品资料",Intent.MEDICINE,{},"selective",details={"medication_name":"布洛芬"})
        assert model.calls == expected_calls
    run(scenario())


def test_correction_is_charged_to_same_model_call_budget():
    async def scenario():
        runtime = await fixture()
        raw = DraftModel(source_scope_as_query)
        budget = ModelBudget(EvalLimits(max_model_calls=1))
        agent = runtime.agents["drug_reference"][0]
        agent.model = BudgetedModel(raw,budget)
        with pytest.raises(EvaluationBudgetExceeded,match="max_model_calls"):
            await agent.run("查询布洛芬药品资料",Intent.MEDICINE,{},"selective",details={"medication_name":"布洛芬"})
        assert raw.calls == budget.calls == len(budget.entries) == 1
        assert budget.exhausted == "max_model_calls"
    run(scenario())


@pytest.mark.parametrize("value", [None,[],{"content":None},{"content":[3]},
    {"content":[{"type":"text","text":3}]},{"content":[{"type":"tool_use","id":"x","name":"x","input":None}]}])
def test_invalid_model_envelopes_have_a_safe_stage_code(value):
    with pytest.raises(AgentOutputError) as error: model_blocks(value)
    assert error.value.stage == "envelope" and error.value.code == "invalid_model_envelope"


def test_pydantic_diagnostics_discard_unknown_keys_values_and_error_messages():
    output = role_output("drug_reference",["ibuprofen-overview"])
    output["payload"][SECRET] = SECRET
    with pytest.raises(AgentOutputError) as error:
        checked_role_output(DEFAULT_REGISTRY["drug_reference"],output,references())
    diagnostic = error.value.diagnostic()
    assert diagnostic["code"] == "invalid_output_shape" and "unknown_field" in diagnostic["fields"]
    assert SECRET not in json.dumps(diagnostic) and SECRET not in str(error.value)


def test_missing_source_provides_operational_next_steps_without_cross_region_evidence():
    async def scenario():
        model = DraftModel()
        runtime = await fixture(model)
        result = await runtime.agents["drug_reference"][0].run("核对中国制剂",Intent.MEDICINE,{},"selective",details=SCOPED|{"region":"中国"})
        assert result.execution.status == "needs_clarification" and result.execution.missing_fields == ["source"]
        assert len(result.answer.next_steps) == 2 and not result.answer.reference_keys
        assert model.calls == 0
        assert "包装" in result.answer.next_steps[0]
    run(scenario())


def test_valid_json_array_is_schema_failure_without_echoing_contents():
    with pytest.raises(AgentOutputError) as error:
        validate_draft(DEFAULT_REGISTRY["drug_reference"],json.dumps([SECRET]),request(),references(),False)
    assert error.value.code == "invalid_output_shape" and not error.value.repairable
    assert SECRET not in json.dumps(error.value.diagnostic())


def test_recovered_correction_is_not_misattributed_as_an_unresolved_case_failure():
    row = {"id":"synthetic","passed":False,"turns":[{"turn":1,"checks":{},"quality":{
        "mean":.7,"relevance":.9,"accuracy":.9,"completeness":.5,"helpfulness":.5},"trace":{"events":[
            {"event":"agent_output_validation","role":"drug_reference","instance":"drug_reference.1","stage":"identity","code":"unconfirmed_medication_identity"},
            {"event":"agent_output_repair","role":"drug_reference","instance":"drug_reference.1","status":"accepted"}]}}]}
    recommendations = recommendations_for([row])
    assert recommendations and all(not entry.startswith("output_validation:") for entry in recommendations[0]["evidence"])


async def offline_report():
    import hashlib
    from medagent.evaluation.runner import evaluate
    from medagent.evaluation.contracts import EvaluationRequest
    from test_independent_runtime import ROOT
    source = await fixture(DraftModel(source_scope_as_query))
    body = json.loads((ROOT/"resources/medication-regression.json").read_text())
    body["judge"] = False
    report = await evaluate(source,EvaluationRequest.model_validate(body))
    events = [event for case in report["cases"] for turn in case["turns"] for event in turn.get("trace",{}).get("events",[])]
    diagnostics = [event for event in events if event["event"] == "agent_output_validation"]
    recovered = [event for event in events if event["event"] == "agent_output_repair"]
    original = ROOT/"artifacts/medication-online-2026-10-27.json"
    checks = {
        "original_4_groups_5_turns_executed":report["complete"] and report["executed"] == 4 and sum(len(case["turns"]) for case in report["cases"]) == 5,
        "all_engineering_checks_pass":report["rule_passed"] == report["passed"] == 4,
        "three_synthetic_identity_errors_observed":len(diagnostics) == 3 and all(event["code"] == "unconfirmed_medication_identity" for event in diagnostics),
        "each_synthetic_error_corrected_once":len(recovered) == 3 and all(event["model_round"] == 2 for event in recovered),
        "no_schema_or_raw_values_in_diagnostics":all(set(event["fields"]) <= {"population","region"} for event in diagnostics),
        "no_extra_instance_retry":all(len(turn["reply"]["routing_decision"]["assignments"][0]["attempts"]) == 1 for case in report["cases"] for turn in case["turns"]),
        "source_missing_has_actionable_steps":len(report["cases"][3]["turns"][0]["reply"]["answer"]["next_steps"]) == 2,
        "judge_scores_remain_missing":all(case["quality"] is None for case in report["cases"]),
        "cleanup_completed":not report["cleanup_errors"] and report["isolation"]["temporary_database_removed"],
    }
    report["verification"] = {"kind":"synthetic_contract_fault_regression","paid_model_calls":0,"online_retest":"not_run",
        "historical_root_cause":"not_proven_no_original_draft_saved","historical_report_sha256":hashlib.sha256(original.read_bytes()).hexdigest(),
        "injected_fault":"synthetic model places reference population/region in unknown query identity; not a replay of historical output",
        "checks":checks}
    return report


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Offline injected contract fault regression; no real model or Judge")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists(): raise FileExistsError("preserve_existing_validation_report")
    report = run(offline_report())
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x") as handle: handle.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report["verification"]["checks"],ensure_ascii=False))
    if not all(report["verification"]["checks"].values()): raise SystemExit(1)
