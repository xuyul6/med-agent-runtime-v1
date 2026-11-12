"""Deterministic clarification guidance without cross-subject facts or live models."""
import json
import pytest
from medagent.core.contracts import Intent, Question
from medagent.evaluation.runner import evaluate
from medagent.evaluation.contracts import EvaluationRequest
from medagent.core.config import ROOT
from support.harness import fixture, run
from support.multiturn import ask
from support.subject_switch import CountStages, guidance_text


class Forbidden:
    def __getattr__(self, name): raise AssertionError("clarification_touched_dependency:"+name)


def test_original_subject_switch_clarifies_without_inheriting_or_generating_drug_information():
    async def scenario():
        model = CountStages();runtime = await fixture(model)
        first = await ask(runtime,"查询布洛芬公开资料。")
        begin = len(model.stages)
        mother = await ask(runtime,"现在给我妈妈问，这个药的资料呢？")
        assert mother.subject == "mother" and mother.intent.subject_changed
        assert not mother.intent.entities and mother.references == []
        result = mother.agent_results[0]
        assert result.status == "needs_clarification" and result.missing_fields == ["medication_name"]
        assert all(value is None for value in result.payload["medication_identity"].values())
        visible = guidance_text(mother)
        assert "切换咨询对象" in visible and "避免混用" in visible
        assert "通用名称" in visible and "药名不清楚" in visible and "药师" in visible
        assert "布洛芬" not in visible and len(mother.answer.next_steps) == 2
        assert model.stages[begin:] == ["understanding"]
        trace = runtime.monitor.traces[-1]
        assert not any(event["event"] == "tool" and event.get("name") == "lookup_reference" for event in trace["events"])
        clarification = next(event for event in trace["events"] if event["event"] == "clarification")
        assert clarification["reason"] == "subject_changed_medication_name_missing" and clarification["fields"] == ["medication_name"]
        restored = await ask(runtime,"改回我自己，这个药的资料呢？")
        assert restored.intent.entities["medication_name"].source.turn_id == first.request_id
        assert restored.intent.entities["medication_name"].source.subject == "self"
        assert restored.agent_results[0].status == "answered" and restored.references[0].key == "ibuprofen-overview"
        return runtime,mother
    run(scenario())


@pytest.mark.parametrize("subject", ["mother","father","PRIVATE_SUBJECT_SENTINEL"])
def test_explicit_subject_metadata_uses_guidance_without_echoing_subject_values(subject):
    async def scenario():
        runtime = await fixture(CountStages())
        await ask(runtime,"查询布洛芬公开资料。")
        reply = await ask(runtime,"这个药的资料呢？",subject=subject)
        assert reply.subject == subject and reply.intent.subject_changed
        assert reply.agent_results[0].missing_fields == ["medication_name"]
        assert "切换咨询对象" in reply.answer.text and subject not in guidance_text(reply)
        assert "布洛芬" not in guidance_text(reply) and not reply.references
        events = [event for event in runtime.monitor.traces[-1]["events"] if event["event"] == "clarification"]
        assert len(events) == 1 and subject not in json.dumps(events)
    run(scenario())


def test_existing_medication_for_current_subject_is_not_asked_again_after_switching_back():
    async def scenario():
        runtime = await fixture(CountStages())
        named = await ask(runtime,"现在给我妈妈查询阿莫西林公开资料。")
        await ask(runtime,"改回我自己，查询布洛芬公开资料。")
        restored = await ask(runtime,"现在给我妈妈问，这个药的资料呢？")
        fact = restored.intent.entities["medication_name"]
        assert fact.value == "阿莫西林" and fact.source.subject == "mother" and fact.source.turn_id == named.request_id
        assert restored.agent_results[0].missing_fields == ["source"]
        assert "medication_name" not in restored.agent_results[0].missing_fields
        assert "布洛芬" not in guidance_text(restored)
        assert not any(e["event"] == "clarification" for e in runtime.monitor.traces[-1]["events"])
    run(scenario())


def test_new_subject_with_explicit_name_does_not_trigger_name_clarification():
    async def scenario():
        runtime = await fixture(CountStages())
        await ask(runtime,"查询布洛芬公开资料。")
        reply = await ask(runtime,"现在给我妈妈查询阿莫西林公开资料。")
        assert reply.subject == "mother" and reply.intent.entities["medication_name"].value == "阿莫西林"
        assert reply.agent_results[0].missing_fields == ["source"]
        assert "重新确认当前对象" not in reply.answer.text
    run(scenario())


def test_same_subject_unknown_followup_does_not_falsely_claim_another_switch():
    async def scenario():
        runtime = await fixture(CountStages())
        await ask(runtime,"查询布洛芬公开资料。")
        await ask(runtime,"现在给我妈妈问，这个药的资料呢？")
        reply = await ask(runtime,"药名还是不清楚。")
        assert reply.subject == "mother" and not reply.intent.subject_changed
        assert reply.agent_results[0].missing_fields == ["medication_name"]
        assert "切换咨询对象" not in reply.answer.text
        assert "布洛芬" not in guidance_text(reply)
    run(scenario())


@pytest.mark.parametrize("details", [{"medication_name":None},
    {"medication_name":None,"query_type":"product_information"}])
def test_switch_guidance_needs_no_model_or_retrieval_and_preserves_all_missing_fields(details):
    async def scenario():
        runtime = await fixture();agent = runtime.agents["drug_reference"][0]
        agent.model = agent.index = Forbidden()
        reply = await agent.run("为当前对象查询药品资料。",Intent.MEDICINE,{},"selective",
            subject_changed=True,details=details)
        assert "medication_name" in reply.execution.missing_fields
        assert len(reply.answer.questions) == len(reply.execution.missing_fields)
        assert len(reply.answer.next_steps) == 2 and not reply.references
        assert reply.execution.payload["medication_identity"]["name"] is None
        if details.get("query_type") == "product_information": assert len(reply.execution.missing_fields) == 6
    run(scenario())


def test_untrusted_memory_cannot_enable_switch_guidance_and_other_roles_keep_their_own_gaps():
    async def scenario():
        runtime = await fixture()
        drug = runtime.agents["drug_reference"][0];drug.model = drug.index = Forbidden()
        unchanged = await drug.run("药名不清楚。",Intent.MEDICINE,{"subject_changed":True},"selective",
            details={"medication_name":None})
        assert "切换咨询对象" not in unchanged.answer.text and not unchanged.answer.next_steps
        report = runtime.agents["report_explainer"][0];report.model = report.index = Forbidden()
        result = await report.run("整理虚构项目。",Intent.REPORT,{},"selective",subject_changed=True,
            details={"items":[{"name":"虚构项目"}]})
        assert result.execution.missing_fields == ["value","unit","reference_range"]
        assert "药品" not in result.answer.text and not result.answer.next_steps
    run(scenario())


def test_critical_subject_switch_still_short_circuits_before_model_or_memory():
    async def scenario():
        runtime = await fixture();runtime.router.model = runtime.memory = Forbidden()
        reply = await runtime.chat(Question(text="现在给我妈妈问，她现在胸痛、呼吸困难。"))
        assert reply.subject == "mother" and reply.primary_agent == "risk_handoff"
        assert reply.storage == "skipped" and reply.usage["calls"] == 0
        assert not any(event["event"] == "clarification" for event in runtime.monitor.traces[-1]["events"])
    run(scenario())


async def offline_report():
    body = json.loads((ROOT/"resources/evaluation.json").read_text())
    case = next(row for row in body["dialog_cases"] if row["id"] == "consultation-subject-isolation")
    request = EvaluationRequest(intent_cases=[],dialog_cases=[case],judge=False)
    report = await evaluate(await fixture(CountStages()),request)
    turns = report["cases"][0]["turns"]
    middle = turns[1]["reply"]
    old_summary = json.loads((ROOT/"artifacts/judge-report-online-recheck-2026-10-27-summary.json").read_text())
    import hashlib
    checks = {
        "original_three_turns_completed":report["complete"] and len(turns) == 3,
        "all_engineering_rules_pass":report["rule_passed"] == 1,
        "subjects_remain_isolated":[item["reply"]["subject"] for item in turns] == ["self","mother","self"] and not middle["intent"]["entities"],
        "switch_reason_and_packaging_steps_present":"切换咨询对象" in middle["answer"]["text"] and "避免混用" in middle["answer"]["text"] and len(middle["answer"]["next_steps"]) == 2,
        "no_previous_drug_or_reference_in_clarification":"布洛芬" not in json.dumps(middle["answer"],ensure_ascii=False) and not middle["references"],
        "original_owner_recovers_own_name":turns[2]["reply"]["intent"]["entities"]["medication_name"]["source"]["subject"] == "self",
        "judge_quality_remains_missing":all(item["quality"] is None for item in turns),
        "old_real_failure_preserved":hashlib.sha256((ROOT/old_summary["raw_report"]).read_bytes()).hexdigest() == old_summary["raw_report_sha256"],
        "isolated_resources_cleaned":not report["cleanup_errors"] and report["isolation"]["temporary_database_removed"],
    }
    assert all(checks.values())
    report["verification"] = {"kind":"offline_synthetic_runtime_clarification_regression","paid_model_calls":0,
        "checks":checks,"prior_real_report_sha256":old_summary["raw_report_sha256"],"online_retest":"not_run"}
    return report


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser();parser.add_argument("--output",required=True);args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise FileExistsError("preserve_existing_report")
    result = run(offline_report())
    with output.open("x") as handle:handle.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(f"{sum(result['verification']['checks'].values())}/{len(result['verification']['checks'])} checks passed; paid model calls=0. Saved: {output}")
