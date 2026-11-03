"""Paired selective/full experiment through the production chat pipeline."""
import asyncio
from copy import deepcopy
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
import random
import time
import uuid
from .contracts import BenchmarkRequest, PROTOCOL, resolve_benchmark
from .isolation import PairedIsolation
from .metrics import measured_usage, staged_usage, summarize_benchmark
from ..core.config import UNHASHED_SETTINGS
from ..core.contracts import Question
from ..evaluation.runner import CODE_HASH, behavior_checks, hard_checks, judge_turn
from ..evaluation.budget import BudgetedModel, EvaluationBudgetExceeded, ModelBudget, judge_model_name
from ..evaluation.metrics import digest


class SafetyStop(RuntimeError):
    pass


class FixedOutputBudget(ModelBudget):
    def reserve(self, requested):
        if self.limits.max_output_tokens-self.reserved_output < min(requested,self.limits.per_call_output_tokens):
            self.exhausted = "max_output_tokens"
            raise EvaluationBudgetExceeded(self.exhausted)
        return super().reserve(requested)


def schedule(request, cases):
    shuffled = list(cases)
    random.Random(request.seed).shuffle(shuffled)
    rows = []
    for repeat in range(request.repetitions):
        for case_index,case in enumerate(shuffled):
            for cache_index,cache in enumerate(request.cache_conditions):
                order = ["selective","full"] if (repeat+case_index+cache_index+request.seed) % 2 == 0 else ["full","selective"]
                rows.append({"id":f"{case.id}-{cache}-{repeat+1}","case_id":case.id,"repeat":repeat+1,"cache":cache,
                    "order":order,"judge_order":list(reversed(order)),
                    "arms":{mode:{"mode":mode,"status":"not_run","quality":None,"judge_error":None,
                                  "rule_passed":False} for mode in ["selective","full"]}})
    return rows


def experiment_metadata(source, isolation, request, version, cases, pairs):
    settings = {field.name:getattr(isolation.settings,field.name) for field in fields(isolation.settings)
        if field.name not in UNHASHED_SETTINGS}
    return {"code_hash":CODE_HASH,"protocol":deepcopy(PROTOCOL),"protocol_hash":digest(PROTOCOL),"dataset":version,
        "cases_hash":digest([case.model_dump(mode="json") for case in cases]),
        "request_hash":digest(request.model_dump(mode="json")),"schedule_hash":digest(pairs),
        "model":isolation.settings.model,"judge_model":judge_model_name(source),"provider":type(isolation.base_model).__name__ if isolation.base_model else None,
        "skills_hash":isolation.skills.snapshot_id,"skills_versions":{role:doc.version for role,doc in isolation.skills.documents.items()},
        "references_hash":digest([ref.model_dump(mode="json") for ref in sorted(isolation.records,key=lambda ref:ref.key)]),
        "references_versions":{ref.key:ref.version for ref in isolation.records},
        "profiles_and_tools_hash":digest({"profiles":[profile.describe() for profile in isolation.profiles],
                                         "tools":[spec.describe() for spec in isolation.specs]}),
        "runtime_settings_hash":digest(settings),"output_cap":request.limits.per_call_output_tokens,
        "cache_TTL_seconds":90,"online_authorization":"execution_requires_separate_user_authorization"}


async def observe(runtime, turn, user, session, subject, mode, budget, row):
    begin,started = len(budget.entries),time.perf_counter()
    row.update(status="running",checks={},safety_checks={},quality=None,judge_error=None)
    reply = None
    runtime.monitor.start()
    try:
        reply = await runtime.chat(Question(text=turn.message,user=user,session=session,subject=turn.subject or subject,context=mode))
        trace = next((trace for trace in reversed(runtime.monitor.traces) if trace["request_id"] == reply.request_id),{})
        row.update(status="completed",reply=reply.report_dump(),trace=trace,
                   checks=behavior_checks(reply,turn.expected),safety_checks=hard_checks(reply,trace,runtime,turn.expected))
        row["rule_passed"] = all(row["checks"].values()) and all(row["safety_checks"].values())
        row["cache_hits"] = sum(event["event"] == "retrieval" and event.get("cache") == "hit" for event in trace.get("events",[]))
        row["cache_misses"] = sum(event["event"] == "retrieval" and event.get("cache") == "miss" for event in trace.get("events",[]))
        if not all(row["safety_checks"].values()): raise SafetyStop("safety_gate_failed")
    except asyncio.CancelledError:
        row.update(status="cancelled",error="deadline_or_cancellation",rule_passed=False)
        raise
    except SafetyStop: raise
    except Exception as exc:
        row.update(status="failed",error=type(exc).__name__,rule_passed=False)
    finally:
        entries = budget.entries[begin:]
        row.update(usage=measured_usage(entries),stages=staged_usage(entries),model_calls=entries,
                   elapsed_ms=round((time.perf_counter()-started)*1000,1))
        await runtime.monitor.stop()
    return reply


def observed_judge_models(entries) -> list[str]:
    return sorted({event["response_model"] for entry in entries if entry["phase"] == "judge" for event in entry["trace"]["events"]
                   if event["event"] == "model" and isinstance(event.get("response_model"),str) and event["response_model"]})


async def score_observation(reply, turn, snapshot, isolation, budget, row):
    begin = len(budget.entries)
    try:
        if reply is None: raise RuntimeError("target_reply_unavailable")
        judge = BudgetedModel(isolation.judge_model,budget,"judge") if isolation.judge_model else None
        row["judge_context"] = {}
        row["quality"] = await judge_turn(judge,turn,reply,snapshot.history,audit=row["judge_context"],trace=row.get("trace"))
    except asyncio.CancelledError:
        row["judge_error"] = "deadline_or_cancellation"
        raise
    except Exception as exc:
        row["judge_error"] = type(exc).__name__
    finally:
        entries = budget.entries[begin:]
        row.update(judge_usage=measured_usage(entries),judge_calls=entries)


def infrastructure_failed(row):
    return row.get("status") in {"failed","cancelled"} or bool(row.get("judge_error")) or bool(row.get("reply",{}).get("degraded"))


async def run_benchmark(source, request: BenchmarkRequest | None=None, *, isolation_factory=PairedIsolation):
    request = (request or BenchmarkRequest()).model_copy(deep=True)
    version,cases = resolve_benchmark(request)
    pairs = schedule(request,cases)
    identity = uuid.uuid4().hex
    budget = FixedOutputBudget(request.limits)
    isolation = isolation_factory(source,identity,budget)
    metadata = experiment_metadata(source,isolation,request,version,cases,pairs)
    report = {"schema_version":"paired-benchmark-1","run_id":identity,"started_at":datetime.now(timezone.utc).isoformat(),
        "mode":"offline" if source.settings.preview or source.model is None else "online","metadata":metadata,
        "limits":request.limits.model_dump(),"pairs":pairs,"preparation":[],"complete":False,"stop_reason":None,
        "quality_equivalence":"not_established"}
    prepared,caches,blocked,warm_failed = {},{},{},set()
    case_map = {case.id:case for case in cases}
    failures,started = 0,time.perf_counter()
    try:
        async with asyncio.timeout(request.limits.timeout_seconds):
            await isolation.create()
            for pair in pairs:
                if budget.exhausted:
                    report["stop_reason"] = budget.exhausted
                    break
                if failures >= request.limits.max_infrastructure_failures:
                    report["stop_reason"] = "infrastructure_failure_limit"
                    break
                case = case_map[pair["case_id"]]
                if case.id not in prepared and case.id not in blocked:
                    prep = {"case_id":case.id,"turns":[],"warmup":None,"snapshot":None}
                    report["preparation"].append(prep)
                    runtime = await isolation.branch("preparation")
                    user,session,subject = "benchmark-"+identity,case.id,case.subject
                    for number,turn in enumerate(case.preparation,1):
                        row = {"turn":number,"message":turn.message}
                        prep["turns"].append(row)
                        reply = await observe(runtime,turn,user,session,subject,"selective",budget,row)
                        if reply is not None: subject = reply.subject
                        if budget.exhausted or not row.get("rule_passed"):
                            blocked[case.id] = budget.exhausted or "preparation_rule_failure"
                            failures += int(infrastructure_failed(row))
                            break
                    prep["not_run_turns"] = list(range(len(prep["turns"])+1,len(case.preparation)+1))
                    if case.id not in blocked:
                        transcript = [message for row in prep["turns"] for message in (
                            {"role":"user","content":row["message"]},
                            {"role":"assistant","content":row["reply"]["answer"]["text"]})]
                        snapshot = await isolation.capture(runtime,user,session,subject,transcript)
                        prepared[case.id] = snapshot
                        prep["snapshot"] = {"sha256":snapshot.sha256,"user":user,"session":session,"risk_seconds":snapshot.risk_seconds,**snapshot.payload}
                        if "warm" in request.cache_conditions:
                            warm = {"mode":"selective","purpose":"common_cache_seed"}
                            prep["warmup"] = warm
                            runtime = await isolation.branch("warmup",snapshot)
                            await observe(runtime,case.target,user,session,subject,"selective",budget,warm)
                            failures += int(infrastructure_failed(warm))
                            if warm.get("rule_passed") and not budget.exhausted:
                                caches[case.id] = isolation.cache_snapshot(runtime)
                                warm["cache_keys"] = [key for key,_ in caches[case.id]]
                            else: warm_failed.add(case.id)
                if case.id in blocked or pair["cache"] == "warm" and case.id in warm_failed:
                    for arm in pair["arms"].values(): arm["reason"] = blocked.get(case.id,"warmup_unavailable")
                    continue
                if budget.exhausted:
                    report["stop_reason"] = budget.exhausted
                    break
                if failures >= request.limits.max_infrastructure_failures:
                    report["stop_reason"] = "infrastructure_failure_limit"
                    break
                snapshot = prepared[case.id]
                pair["snapshot_hash"] = snapshot.sha256
                replies = {}
                counted_failures = set()
                for mode in pair["order"]:
                    if budget.exhausted: break
                    arm = pair["arms"][mode]
                    runtime = await isolation.branch("target",snapshot,caches.get(case.id,()) if pair["cache"] == "warm" else ())
                    restored = await isolation.capture(runtime,snapshot.user,snapshot.session,snapshot.subject,snapshot.history)
                    if restored.sha256 != snapshot.sha256: raise ValueError("snapshot_restore_mismatch")
                    arm.update(snapshot_hash=restored.sha256,seeded_cache_entries=len(runtime.index.cache),
                        initial_monitor_requests=runtime.monitor.requests,initial_tool_failures=dict(runtime.tools.failures))
                    replies[mode] = await observe(runtime,case.target,snapshot.user,snapshot.session,snapshot.subject,mode,budget,arm)
                    if infrastructure_failed(arm):
                        failures += 1
                        counted_failures.add(mode)
                    if failures >= request.limits.max_infrastructure_failures:
                        report["stop_reason"] = "infrastructure_failure_limit"
                        break
                if report["stop_reason"]: break
                if request.judge:
                    for mode in pair["judge_order"]:
                        if budget.exhausted: break
                        if pair["arms"][mode]["status"] != "not_run":
                            await score_observation(replies.get(mode),case.target,snapshot,isolation,budget,pair["arms"][mode])
                            if mode not in counted_failures and infrastructure_failed(pair["arms"][mode]):
                                failures += 1
                                counted_failures.add(mode)
                            if failures >= request.limits.max_infrastructure_failures:
                                report["stop_reason"] = "infrastructure_failure_limit"
                                break
                if report["stop_reason"]: break
            if budget.exhausted: report["stop_reason"] = budget.exhausted
    except TimeoutError:
        report["stop_reason"] = "timeout"
    except SafetyStop:
        report["stop_reason"] = "safety_gate_failed"
    except asyncio.CancelledError: raise
    except Exception as exc:
        report["stop_reason"] = "isolation_or_execution_error:"+type(exc).__name__
    finally:
        try: await asyncio.wait_for(isolation.cleanup(),10)
        except Exception as exc: isolation.cleanup_errors.append(type(exc).__name__)
    report["cleanup_errors"] = isolation.cleanup_errors
    for pair in pairs:
        for arm in pair["arms"].values():
            if arm["status"] == "not_run": arm.setdefault("reason",report["stop_reason"] or "not_reached")
    report["complete"] = not report["stop_reason"] and not isolation.cleanup_errors and all(
        arm["status"] in {"completed","failed"} for pair in pairs for arm in pair["arms"].values())
    report["infrastructure_failures"] = failures
    report["isolation"] = {"branches":isolation.serial,"redis_keys_written":len(isolation.redis.written) if isolation.redis else 0,
        "temporary_database_removed":isolation.directory is None or not Path(isolation.directory.name).exists(),
        "production_state_written":False}
    report["usage"] = {phase:measured_usage([entry for entry in budget.entries if entry["phase"] == phase])
                       for phase in ["preparation","warmup","target","judge"]}
    report["stages"] = {phase:staged_usage([entry for entry in budget.entries if entry["phase"] == phase]) for phase in report["usage"]}
    report["budget"] = {"attempted_calls":budget.calls,"reserved_max_output_tokens":budget.reserved_output,"stop_reason":budget.exhausted}
    report["elapsed_ms"] = round((time.perf_counter()-started)*1000,1)
    # The Judge may be another vendor by design; only the models under test have to agree.
    model_events = [event for entry in budget.entries if entry["phase"] != "judge" for event in entry["trace"]["events"] if event["event"] == "model"]
    observed_models = sorted({event["response_model"] for event in model_events if isinstance(event.get("response_model"),str) and event["response_model"]})
    report["observed_models"] = observed_models
    report["observed_judge_models"] = observed_judge_models(budget.entries)
    report["comparability"] = {
        "snapshot_verified":all(arm.get("snapshot_hash") == pair.get("snapshot_hash") and bool(pair.get("snapshot_hash"))
            for pair in pairs for arm in pair["arms"].values() if arm["status"] != "not_run"),
        "actual_model_consistent":bool(model_events) and len(observed_models) == 1 and all(event.get("response_model") in observed_models for event in model_events)}
    summarize_benchmark(report,request)
    return report
