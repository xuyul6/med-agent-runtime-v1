import dataclasses
"""Synthetic engineering checks; token counters are fixtures, never measurements."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from medagent.benchmark.cli import build_plan, compare
from medagent.benchmark.contracts import BenchmarkRequest, PROTOCOL, resolve_benchmark
from medagent.benchmark.isolation import PairedIsolation
from medagent.benchmark.metrics import measured_usage, paired_statistics, staged_usage, summarize_benchmark
from medagent.benchmark.runner import FixedOutputBudget, run_benchmark, schedule
from medagent.core.contracts import Question
from medagent.evaluation.budget import BudgetedModel, EvaluationBudgetExceeded
from medagent.evaluation.metrics import digest
from medagent.core.observability import event
from support.harness import captured_deadlines, expire, run, fixture
from support.medication import EvidenceModel


def small_request(**changes):
    body = {"cases":[{"id":"visit","preparation":["我想整理就医资料。","我已列出希望向医护确认的问题。"],
        "target":{"message":"请整理就医准备事项和需要核实的信息。","expected":{"role":"visit_guide","intent":"visit","references":True}}}],
        "repetitions":2,"judge":False}
    return BenchmarkRequest.model_validate(body | changes)


class CountingModel(EvidenceModel):
    """Fixed fixture counts and Judge scores do not model real tokenization/quality."""
    async def json(self, stage, system, content, budget=900):
        event("model",stage=stage,model="synthetic-counter",response_model="synthetic-counter",input_tokens=20,output_tokens=10,
              cache_read_input_tokens=0,cache_creation_input_tokens=0)
        if stage == "evaluation_judge":
            return {**{key:.9 for key in ["relevance","accuracy","completeness","helpfulness"]},"reason":"合成夹具分数，仅用于框架验证。"}
        return await super().json(stage,system,content,budget)

    async def message(self, stage, system, messages, tools=None, budget=1800):
        event("model",stage=stage,model="synthetic-counter",response_model="synthetic-counter",input_tokens=100,output_tokens=25,
              cache_read_input_tokens=0,cache_creation_input_tokens=0)
        return await super().message(stage,system,messages,tools,budget)


@pytest.mark.parametrize("patch", [
    {"cases":None},{"cases":[]},{"repetitions":3},{"repetitions":True},{"repetitions":12},
    {"cache_conditions":["cold","cold"]},{"cache_conditions":["other"]},
    {"unknown":1},{"limits":{"max_model_calls":0}},{"limits":{"per_call_output_tokens":100}},
    {"limits":{"timeout_seconds":1900}},{"limits":{"max_model_calls":True}},
])
def test_invalid_protocol_is_rejected_before_execution(patch):
    with pytest.raises(ValidationError): BenchmarkRequest.model_validate(patch)


def test_protocol_and_balanced_schedule_are_stable_and_include_original_failure():
    request = BenchmarkRequest()
    version,cases = resolve_benchmark(request)
    pairs = schedule(request,cases)
    assert version == "paired-context-cases-1" and len(cases) == 4 and len(pairs) == 32
    assert any(case.target.message == "查询布洛芬药品资料" for case in cases)
    assert pairs == schedule(request,cases)
    assert pairs != schedule(request.model_copy(update={"seed":request.seed+1}),cases)
    for case in cases:
        for cache in request.cache_conditions:
            group = [pair for pair in pairs if pair["case_id"] == case.id and pair["cache"] == cache]
            assert [pair["order"][0] for pair in group].count("selective") == 2
            assert all(pair["judge_order"] == list(reversed(pair["order"])) for pair in group)
    plan = build_plan(request)
    assert plan["online_status"] == "not_run" and plan["planned_targets"] == 64
    assert plan["protocol_hash"] == digest(PROTOCOL)


def test_budget_stops_whole_call_instead_of_giving_one_arm_a_smaller_cap():
    request = small_request(limits={"max_output_tokens":2000,"per_call_output_tokens":1900})
    budget = FixedOutputBudget(request.limits)
    assert budget.reserve(1900) == 1900
    with pytest.raises(EvaluationBudgetExceeded): budget.reserve(950)
    assert budget.calls == 1 and budget.reserved_output == 1900


def usage_entry(**overrides):
    event = {"event":"model","input_tokens":100,"output_tokens":10,"cache_read_input_tokens":20,"cache_creation_input_tokens":30,**overrides}
    return {"stage":"agent_general","usage":{"measurement":"provider"},"trace":{"events":[event]}}


def test_provider_usage_includes_cache_fields_and_preserves_unknowns():
    result = measured_usage([usage_entry()])
    assert result["input_tokens"] == 100 and result["total_input_tokens"] == 150
    assert result["cache_read_input_tokens"] == 20
    for key in ["cache_read_input_tokens","cache_creation_input_tokens","input_tokens","output_tokens"]:
        for invalid in [None,True,-1]:
            result = measured_usage([usage_entry(**{key:invalid})])
            assert result[key] is None
            if key != "output_tokens": assert result["total_input_tokens"] is None
    failed = usage_entry();failed["usage"]["measurement"] = "unavailable";failed["trace"]["events"] = []
    assert measured_usage([usage_entry(),failed])["total_input_tokens"] is None
    assert measured_usage([])["total_input_tokens"] == 0
    assert staged_usage([usage_entry()])["agent_general"]["total_input_tokens"] == 150


def numeric_pair(number=1, selective=50, full=100, delta=0):
    return {"id":str(number),"case_id":f"case-{number%4}","cache":"cold","repeat":number//4+1,
        "arms":{mode:{"status":"completed","rule_passed":True,"usage":{"total_input_tokens":tokens,"output_tokens":10},
                      "quality":{key:.9+(delta if mode == "selective" else 0) for key in ["relevance","accuracy","completeness","helpfulness","mean"]}}
                for mode,tokens in [("selective",selective),("full",full)]}}


def test_paired_math_uses_ratio_of_totals_and_keeps_failed_known_costs():
    a,b = numeric_pair(),numeric_pair(2,selective=90,full=300)
    a["arms"]["selective"]["rule_passed"] = False
    result = paired_statistics([a,b])
    assert result["observed_pairs"] == 2 and result["input_reduction"] == pytest.approx(.65)
    assert result["paired_difference"]["mean"] == -130
    b["arms"]["full"]["usage"]["total_input_tokens"] = None
    result = paired_statistics([a,b])
    assert result["missing_pairs"] == ["2"] and result["input_reduction"] is None
    assert paired_statistics([numeric_pair(full=0)])["input_reduction"] is None
    assert paired_statistics([])["input_reduction"] is None
    assert paired_statistics([numeric_pair(selective=150)])["input_reduction"] == -.5


@pytest.mark.parametrize("fault", ["offline","missing_judge","quality_drop","safety","missing_usage","incomplete","small_sample","low_full_quality"])
def test_positive_joint_claim_is_blocked_by_any_missing_or_failure(fault):
    pairs = [numeric_pair(number) for number in range(16)]
    report = {"pairs":pairs,"mode":"online","complete":True,"comparability":{"snapshot_verified":True,"actual_model_consistent":True}}
    request = BenchmarkRequest(cache_conditions=["cold"])
    if fault == "offline": report["mode"] = "offline"
    elif fault == "missing_judge": pairs[0]["arms"]["full"]["quality"] = None
    elif fault == "quality_drop": pairs[0] = numeric_pair(0,delta=-.14)
    elif fault == "safety": pairs[0]["arms"]["selective"]["rule_passed"] = False
    elif fault == "missing_usage": pairs[0]["arms"]["selective"]["usage"]["total_input_tokens"] = None
    elif fault == "incomplete": report["complete"] = False
    elif fault == "small_sample": request = BenchmarkRequest(repetitions=2,cache_conditions=["cold"])
    elif fault == "low_full_quality": pairs[0]["arms"]["full"]["quality"]["mean"] = .7
    summarize_benchmark(report,request)
    assert report["quality_noninferiority"] == "not_established" and report["claim_blockers"]
    assert report["supported_input_reduction"]["cold"] is None


def test_margin_boundary_is_checked_per_case_and_dimension_without_equivalence_claim():
    request = BenchmarkRequest(cache_conditions=["cold"])
    report = {"pairs":[numeric_pair(number,delta=-.03) for number in range(16)],"mode":"online","complete":True,
              "comparability":{"snapshot_verified":True,"actual_model_consistent":True}}
    summarize_benchmark(report,request)
    assert report["quality_noninferiority"] == "within_margin_on_observed_cases"
    assert report["quality_equivalence"] == "not_established"
    assert report["supported_input_reduction"]["cold"] == .5
    report["pairs"][0]["arms"]["selective"]["quality"]["helpfulness"] -= .001
    summarize_benchmark(report,request)
    assert "quality_margin_not_met" in report["claim_blockers"]


def test_actual_chat_pairs_restore_same_snapshot_reset_state_and_balance_cache():
    async def scenario():
        source = await fixture(CountingModel())
        before = (deepcopy(source.memory.local),source.monitor.requests,deepcopy(source.index.cache))
        report = await run_benchmark(source,small_request(judge=True))
        assert report["complete"],report.get("stop_reason")
        assert len(report["pairs"]) == 4 and len(report["preparation"]) == 1
        prep = report["preparation"][0]
        assert len(prep["turns"]) == 2 and prep["warmup"]["rule_passed"]
        snapshot_hash = prep["snapshot"]["sha256"]
        for pair in report["pairs"]:
            assert pair["snapshot_hash"] == snapshot_hash
            for arm in pair["arms"].values():
                assert arm["snapshot_hash"] == snapshot_hash and arm["rule_passed"] and arm["quality"]["mean"] == .9
                assert arm["initial_monitor_requests"] == 0 and not arm["initial_tool_failures"]
                assert arm["seeded_cache_entries"] == (0 if pair["cache"] == "cold" else len(prep["warmup"]["cache_keys"]))
                assert arm["cache_hits"] == (0 if pair["cache"] == "cold" else 1)
                assert arm["judge_calls"][0]["phase"] == "judge"
                assert all(call["granted_output_tokens"] == min(call["requested_output_tokens"],2500) for call in arm["model_calls"])
        assert all(report["usage"][phase]["attempted_calls"] > 0 for phase in ["preparation","warmup","target","judge"])
        assert report["quality_noninferiority"] == "not_established" and "offline_or_simulated_run" in report["claim_blockers"]
        assert report["isolation"]["temporary_database_removed"] and not report["cleanup_errors"]
        assert before == (source.memory.local,source.monitor.requests,source.index.cache)
    run(scenario())


def test_memory_summary_and_episodes_are_cloned_without_replaying_preparation():
    async def scenario():
        source = await fixture(CountingModel())
        budget = FixedOutputBudget(BenchmarkRequest().limits)
        isolation = PairedIsolation(source,"test-snapshot-episode",budget)
        try:
            await isolation.create()
            runtime = await isolation.branch("preparation")
            for number in range(6):
                await runtime.chat(Question(text=f"整理就医准备资料，第{number+1}轮。",user="synthetic",session="episode"))
            snapshot = await isolation.capture(runtime,"synthetic","episode","self")
            assert snapshot.payload["state"]["summary"] and snapshot.payload["episodes"]
            assert len(snapshot.history) == 4
            calls = budget.calls
            for _ in range(2):
                branch = await isolation.branch("target",snapshot)
                restored = await isolation.capture(branch,"synthetic","episode","self")
                assert restored.sha256 == snapshot.sha256
                assert budget.calls == calls
            assert json.loads(snapshot.state_json)["turn_counter"] == 6
        finally: await isolation.cleanup()
        assert not isolation.cleanup_errors
    run(scenario())


def test_paired_snapshot_freezes_followup_context_time_for_both_arms():
    async def scenario():
        source = await fixture(CountingModel())
        budget = FixedOutputBudget(BenchmarkRequest().limits)
        isolation = PairedIsolation(source, "test-visit-context-clock", budget)
        clock = [1000.0]
        try:
            await isolation.create()
            runtime = await isolation.branch("preparation")
            runtime.memory.context_clock = lambda:clock[0]
            await runtime.chat(Question(text="就医前需要准备哪些资料？", user="synthetic", session="paired"))
            snapshot = await isolation.capture(runtime, "synthetic", "paired", "self")
            clock[0] = 1301.0
            for mode in ["selective", "full"]:
                branch = await isolation.branch("target", snapshot)
                restored = await isolation.capture(branch, "synthetic", "paired", "self")
                assert restored.sha256 == snapshot.sha256 and restored.context_time == 1000.0
                reply = await branch.chat(Question(text="接下来准备哪些资料？", user="synthetic", session="paired", context=mode))
                assert reply.agent_results[0].status == "answered"
                assert any(event["event"] == "visit_context_recovered" for event in branch.monitor.traces[-1]["events"])
        finally:
            await isolation.cleanup()
        assert not isolation.cleanup_errors

    run(scenario())


@pytest.mark.parametrize("limit", [{"max_model_calls":1},{"max_output_tokens":20}])
def test_budget_stop_preserves_all_planned_arms_and_partial_calls(limit):
    async def scenario():
        source = await fixture(CountingModel())
        report = await run_benchmark(source,small_request(limits=limit))
        assert not report["complete"] and report["stop_reason"] in {"max_model_calls","max_output_tokens"}
        assert len(report["pairs"]) == 4 and report["not_run"]
        assert report["budget"]["attempted_calls"] <= limit.get("max_model_calls",800)
        assert report["supported_input_reduction"] == {"cold":None,"warm":None}
        assert report["isolation"]["temporary_database_removed"]
    run(scenario())


def test_unknown_provider_usage_remains_unknown_in_finished_pairs():
    async def scenario():
        source = await fixture(EvidenceModel())
        report = await run_benchmark(source,small_request(cache_conditions=["cold"]))
        assert report["complete"]
        assert report["usage"]["target"]["total_input_tokens"] is None
        assert report["statistics"]["by_cache"]["cold"]["input_reduction"] is None
        assert "unknown_or_missing_input_usage" in report["claim_blockers"]
    run(scenario())


def test_duplicate_scenarios_cannot_inflate_sample_size_by_renaming_ids():
    raw = small_request().model_dump()
    raw["cases"].append({**raw["cases"][0],"id":"renamed"})
    with pytest.raises(ValidationError,match="duplicate_scenario_content"): BenchmarkRequest.model_validate(raw)


def test_deadline_during_target_keeps_partial_call_usage_and_unrun_arms(monkeypatch):
    deadlines = captured_deadlines(monkeypatch)
    class Slow(CountingModel):
        async def message(self,*args,**kwargs):
            event("model",stage=args[0],model="synthetic",input_tokens=11,output_tokens=2,
                  cache_read_input_tokens=0,cache_creation_input_tokens=0)
            # The run's deadline passes while this call is in flight, however long the setup before it took.
            expire(deadlines[0])
            await asyncio.sleep(10)
            return await super().message(*args,**kwargs)
    async def scenario():
        request = small_request(cases=[{"id":"deadline","target":{"message":"整理就医资料。"}}],
                                cache_conditions=["cold"],limits={"timeout_seconds":60})
        report = await run_benchmark(await fixture(Slow()),request)
        assert report["stop_reason"] == "timeout" and not report["complete"]
        arms = [arm for pair in report["pairs"] for arm in pair["arms"].values()]
        assert any(arm["status"] == "cancelled" and arm.get("model_calls") for arm in arms)
        assert report["not_run"] and report["isolation"]["temporary_database_removed"]
    run(scenario())


def test_judge_failure_is_preserved_and_does_not_become_a_zero_score():
    class BrokenJudge(CountingModel):
        async def json(self,stage,*args,**kwargs):
            if stage == "evaluation_judge": raise RuntimeError("synthetic_judge_failure")
            return await super().json(stage,*args,**kwargs)
    async def scenario():
        report = await run_benchmark(await fixture(BrokenJudge()),small_request(judge=True,cache_conditions=["cold"],
                                                                             limits={"max_infrastructure_failures":10}))
        assert report["complete"]
        assert all(arm["quality"] is None and arm["judge_error"] == "RuntimeError" and arm["judge_usage"]["total_input_tokens"] is None
                   for pair in report["pairs"] for arm in pair["arms"].values())
        assert report["quality_noninferiority"] == "not_established"
    run(scenario())


def test_source_and_request_mutations_do_not_change_frozen_inputs():
    async def scenario():
        request = small_request(cache_conditions=["cold"])
        class Mutating(CountingModel):
            async def json(self,stage,*args,**kwargs):
                request.cases[0].target.message = "不可进入冻结请求"
                source.index.documents.clear()
                return await super().json(stage,*args,**kwargs)
        source = await fixture(Mutating())
        report = await run_benchmark(source,request)
        assert report["complete"]
        assert all(arm["rule_passed"] for pair in report["pairs"] for arm in pair["arms"].values())
        assert "consultation-record" in report["metadata"]["references_versions"]
        report["metadata"]["protocol"]["quality_margin"] = .5
        assert PROTOCOL["quality_margin"] == .03
    run(scenario())


def test_safety_failure_stops_early_and_preserves_failing_arm():
    from medagent.benchmark.runner import observe as real_observe
    import medagent.benchmark.runner as runner
    async def scenario():
        async def unsafe(runtime,turn,user,session,subject,mode,budget,row):
            reply = await real_observe(runtime,turn,user,session,subject,mode,budget,row)
            row["safety_checks"]["synthetic_gate"] = False
            row["rule_passed"] = False
            raise runner.SafetyStop("synthetic_safety_failure")
        original = runner.observe
        runner.observe = unsafe
        try:
            request = small_request(cases=[{"id":"unsafe","target":{"message":"整理就医资料。"}}],cache_conditions=["cold"])
            report = await runner.run_benchmark(await fixture(CountingModel()),request)
        finally: runner.observe = original
        assert report["stop_reason"] == "safety_gate_failed" and report["not_run"]
        assert any(arm.get("safety_checks",{}).get("synthetic_gate") is False for pair in report["pairs"] for arm in pair["arms"].values())
        assert report["supported_input_reduction"]["cold"] is None
    run(scenario())


def test_runtime_faults_and_infrastructure_stop_limit_are_not_ignored():
    class InvalidAgent(CountingModel):
        async def message(self,*args,**kwargs): return {"content":[{"type":"text","text":"{}"}]}
    async def scenario():
        request = small_request(cases=[{"id":"failed","target":{"message":"整理就医资料。"}}],
            cache_conditions=["cold"],limits={"max_infrastructure_failures":1})
        report = await run_benchmark(await fixture(InvalidAgent()),request)
        assert report["stop_reason"] == "infrastructure_failure_limit"
        assert report["infrastructure_failures"] >= 1 and report["not_run"]
        assert any(arm.get("reply",{}).get("degraded") for pair in report["pairs"] for arm in pair["arms"].values())
    run(scenario())


def test_api_validates_schema_and_shares_evaluation_concurrency_lock():
    from fastapi.testclient import TestClient
    from medagent.app import create_app
    source = run(fixture(CountingModel()))
    app = create_app(dataclasses.replace(source.settings, developer_tools=True),source)
    with TestClient(app) as client:
        assert client.post("/benchmark/run",json={"unknown":1}).status_code == 422
        assert client.post("/benchmark/run",json={"repetitions":3}).status_code == 422
        assert client.post("/benchmark/run",json={"judge":True}).status_code == 409
        response = client.post("/benchmark/run",json=small_request(cache_conditions=["cold"]).model_dump(mode="json"))
        assert response.status_code == 200 and response.json()["complete"]
        class Held:
            def locked(self): return True
        app.state.evaluation_lock = Held()
        assert client.post("/benchmark/run",json={"judge":False}).status_code == 409
        assert client.post("/eval/run",json={"judge":False,"intent_cases":[],"dialog_cases":[]}).status_code == 409
        assert source.monitor.requests == 0


def test_cli_uses_isolated_endpoint_and_refuses_existing_outputs(monkeypatch,tmp_path):
    import httpx
    import medagent.benchmark.cli as module
    sent = []
    def handler(request):
        sent.append(request)
        assert request.url.path == "/benchmark/run"
        body = json.loads(request.content)
        assert body["cases"][0]["id"] == "visit"
        return httpx.Response(200,json={"complete":False,"quality_noninferiority":"not_established"})
    real = httpx.AsyncClient
    monkeypatch.setattr(module.httpx,"AsyncClient",lambda **kwargs:real(transport=httpx.MockTransport(handler),**kwargs))
    output = tmp_path/"report.json"
    result = run(compare("http://test",output,small_request()))
    assert result["client_plan"]["planned_targets"] == 8 and len(sent) == 1
    assert result["client_plan_mismatch"] == ["protocol_hash","request_hash"]
    assert result["supported_input_reduction"] == {"cold":None,"warm":None}
    with pytest.raises(FileExistsError): run(compare("http://test",output,small_request()))
    assert len(sent) == 1


def test_provider_exposes_actual_cache_usage_in_call_logs():
    import httpx
    from medagent.core.provider import ClaudeGateway
    from medagent.core.config import Settings
    async def scenario():
        gateway = ClaudeGateway(Settings(api_key="synthetic-placeholder",model="test"),httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _:httpx.Response(200,json={"model":"test","usage":{"input_tokens":11,"output_tokens":2,
                "cache_read_input_tokens":40,"cache_creation_input_tokens":20},"stop_reason":"end_turn","content":[]}))))
        budget = FixedOutputBudget(BenchmarkRequest().limits)
        await BudgetedModel(gateway,budget).message("agent_general","system",[])
        assert measured_usage(budget.entries)["total_input_tokens"] == 71
        await gateway.close()
    run(scenario())


def test_redis_branches_use_independent_prefixes_and_delete_only_own_keys():
    class Redis:
        def __init__(self): self.data = {"production-sentinel":"untouched"};self.writes = []
        async def get(self,key): return self.data.get(key)
        async def set(self,key,value,ex): self.writes.append((key,ex));self.data[key] = value
        async def delete(self,*keys):
            for key in keys: self.data.pop(key,None)
    async def scenario():
        source = await fixture(CountingModel())
        redis = Redis();source.memory.redis = redis
        report = await run_benchmark(source,small_request(cache_conditions=["cold"]))
        assert report["complete"] and report["isolation"]["redis_keys_written"] >= 5
        assert redis.data == {"production-sentinel":"untouched"}
        assert all(key.startswith("mar2:eval:"+report["run_id"]+":mar2:eval:branch-") and ttl <= 3600 for key,ttl in redis.writes)
    run(scenario())


def test_full_original_transcript_reaches_judge_after_memory_compression():
    histories = []
    class JudgeHistory(CountingModel):
        async def json(self,stage,system,content,budget=900):
            if stage == "evaluation_judge": histories.append(json.loads(content)["history"])
            return await super().json(stage,system,content,budget)
    async def scenario():
        case = {"id":"compressed","preparation":[f"第{n}轮整理就医准备。" for n in range(1,7)],
                "target":{"message":"请总结就医准备事项。"}}
        report = await run_benchmark(await fixture(JudgeHistory()),small_request(cases=[case],judge=True,cache_conditions=["cold"]))
        assert report["complete"]
        snapshot = report["preparation"][0]["snapshot"]
        assert len(snapshot["state"]["messages"]) == 4 and len(snapshot["original_transcript"]) == 12
        assert len(histories) == 4 and all(len(history) == 12 for history in histories)
        assert all(history[0]["content"] == "第1轮整理就医准备。" for history in histories)
    run(scenario())


def test_missing_actual_model_identity_blocks_claim_even_if_other_numbers_are_known():
    request = BenchmarkRequest(cache_conditions=["cold"])
    report = {"pairs":[numeric_pair(number) for number in range(16)],"mode":"online","complete":True,
              "comparability":{"snapshot_verified":True,"actual_model_consistent":False}}
    summarize_benchmark(report,request)
    assert report["quality_noninferiority"] == "not_established"
    assert "snapshot_or_actual_model_not_verified" in report["claim_blockers"]


async def offline_benchmark_report():
    source = await fixture(CountingModel())
    before = (deepcopy(source.memory.local),source.monitor.requests,deepcopy(source.index.cache))
    request = BenchmarkRequest(judge=False)
    report = await run_benchmark(source,request)
    pairs = report["pairs"]
    checks = {
        "all_32_pairs_and_64_targets_executed":report["complete"] and len(pairs) == 32 and not report["not_run"],
        "all_target_engineering_rules_pass":all(arm.get("rule_passed") for pair in pairs for arm in pair["arms"].values()),
        "same_snapshot_in_both_modes":all(arm.get("snapshot_hash") == pair["snapshot_hash"] for pair in pairs for arm in pair["arms"].values()),
        "single_preparation_per_case":len(report["preparation"]) == 4,
        "balanced_AB_BA":all(sum(pair["order"][0] == "selective" for pair in pairs if pair["case_id"] == case and pair["cache"] == cache) == 2
            for case in {pair["case_id"] for pair in pairs} for cache in ["cold","warm"]),
        "fresh_health_and_tools":all(arm["initial_monitor_requests"] == 0 and not arm["initial_tool_failures"] for pair in pairs for arm in pair["arms"].values()),
        "source_state_unchanged":before == (source.memory.local,source.monitor.requests,source.index.cache),
        "cleanup_completed":report["isolation"]["temporary_database_removed"] and not report["cleanup_errors"],
        "no_fake_quality_or_saving_claim":report["quality_noninferiority"] == "not_established" and all(value is None for value in report["supported_input_reduction"].values()),
        "no_judge_calls":report["usage"]["judge"]["attempted_calls"] == 0,
    }
    report["verification"] = {"kind":"offline_engineering_only","counter_origin":"fixed_synthetic_test_fixture_not_provider_measurement",
        "paid_model_calls":0,"online_status":"not_run","original_medication_failure_online":"not_run","checks":checks}
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline paired benchmark with fixed fake counts; no paid model calls")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists(): raise FileExistsError("preserve_existing_benchmark_report")
    report = run(offline_benchmark_report())
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x") as handle: handle.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report["verification"]["checks"],ensure_ascii=False))
    if not all(report["verification"]["checks"].values()): raise SystemExit(1)
