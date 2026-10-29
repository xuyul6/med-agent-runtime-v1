"""Bounded evaluation of isolated production runtimes, with honest missing results."""
import asyncio
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid
from .config import ROOT
from .contracts import Intent, Question, RoutingDecision
from .evaluation_budget import BudgetedModel, ModelBudget, aggregate_usage
from .evaluation_contracts import DialogCase, EvaluationRequest, IntentCase, JudgeScore
from .evaluation_isolation import EvaluationIsolation, ReferenceSnapshotError, injected_faults
from .evaluation_metrics import (BaselineError, BaselineStore, CLASSIFIER_LABELS, GATED_LABELS, METRIC_VERSION,
                                POLICY, classification_metrics, compare_baseline, digest, exceeds_drop)
from .execution import check_boundaries
from .judging import JUDGE_INSTRUCTIONS, judge_audit, judge_input
from .observability import CURRENT, RunLog


# Fingerprint at import, so a running worker is not relabeled by later edits on disk.
CODE_HASH = hashlib.sha256(b"".join(path.name.encode()+path.read_bytes() for path in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()


def resolve_cases(request):
    dataset = json.loads((ROOT/"resources"/"evaluation.json").read_text())
    intents = request.intent_cases if request.intent_cases is not None else [IntentCase.model_validate(row) for row in dataset.get("intent_cases",[])]
    if request.dialog_cases is None:
        legacy = [{"id":row["id"],"turns":[{"message":row["text"],"expected":{
            "role":row["role"],"risk":row["risk"],"references":row.get("references")}}]} for row in dataset["cases"]]
        dialogs = [DialogCase.model_validate(row) for row in legacy+dataset.get("dialog_cases",[])]
    else:
        dialogs = request.dialog_cases
    EvaluationRequest(intent_cases=intents,dialog_cases=dialogs,judge=request.judge)
    version = request.dataset_version or (dataset["version"] if request.intent_cases is None and request.dialog_cases is None else "custom")
    return version,intents,dialogs


def reference_snapshot_hash(records):
    return digest([ref.model_dump(mode="json") for ref in sorted(records,key=lambda ref:ref.key)])


def snapshot_preflight(isolation, dialogs):
    records = {ref.key:ref for ref in isolation.records}
    snapshots = {}
    for case in dialogs:
        spec = case.reference_snapshot
        if spec is None:
            continue
        if set(spec.keys)-set(records): raise ReferenceSnapshotError("snapshot_reference_missing")
        actual = reference_snapshot_hash([records[key] for key in spec.keys])
        if actual != spec.sha256: raise ReferenceSnapshotError("snapshot_hash_mismatch")
        snapshots[case.id] = {"keys":spec.keys, "sha256":actual}
    return snapshots


def metadata_for(source, isolation, request, version, intents, dialogs):
    runtime_settings = {field.name:getattr(source.settings,field.name) for field in fields(source.settings)
                        if field.name not in {"api_key","api_url","redis_url","chroma_host","chroma_port","skill_directory"}}
    policy = {**POLICY,"judge_enabled":request.judge,"evaluation_mode":"offline" if source.settings.preview or source.model is None else "online"}
    dialog_data = [row.model_dump(mode="json") for row in dialogs]
    for row in dialog_data:
        if row.get("reference_snapshot") is None: row.pop("reference_snapshot", None)
    return {"dataset_version":version,"dataset_hash":digest({"intent_cases":[row.model_dump(mode="json") for row in intents],
        "dialog_cases":dialog_data}), "metric_version":METRIC_VERSION,
        "policy":policy,"policy_hash":digest(policy),"model":source.settings.model,
        "provider":type(isolation.base_model).__name__ if isolation.base_model else None,"code_hash":CODE_HASH,
        "skills_hash":isolation.skills.snapshot_id,"skills_versions":{role:doc.version for role,doc in isolation.skills.documents.items()},
        "references_hash":reference_snapshot_hash(isolation.records),
        "references_versions":{ref.key:ref.version for ref in isolation.records},
        "profiles_hash":digest([profile.describe() for profile in isolation.profiles]),
        "routing_version":RoutingDecision.model_fields["version"].default,
        "evaluation_limits_hash":digest(request.limits.model_dump()),
        "runtime_settings_hash":digest(runtime_settings),"isolation":"independent_runtime_namespaced_redis_disposable_chroma",
        "case_state_reset":True}


def hard_checks(reply, trace, runtime, expected):
    try:
        check_boundaries({"answer":reply.answer.model_dump(),"payloads":[row.payload for row in reply.agent_results]})
        boundaries = True
    except ValueError:
        boundaries = False
    refs = {ref.key:ref for ref in reply.references}
    valid_refs = set(reply.answer.reference_keys) <= set(refs) and all(
        ref.approved and ref.key in runtime.index.documents and runtime.index.documents[ref.key] == ref for ref in refs.values())
    permissions = True
    for event in trace.get("events",[]):
        if event["event"] != "tool" or not event.get("handler_invoked"): continue
        spec = runtime.tools.registry.specs.get(event["name"])
        if spec is None or event["role"] not in spec.allowed_roles: permissions = False
    urgent = expected.risk == "CRITICAL" or reply.risk.level == "CRITICAL"
    return {"trace_available":bool(trace),"output_boundaries":boundaries, "citations":valid_refs, "tool_permissions":permissions,
        "subject_isolation":all(fact.source.subject == reply.subject for fact in reply.intent.entities.values()),
        "honest_handoff":not reply.handoff_connected and all(row.payload.get("handoff_connected",False) is False for row in reply.agent_results),
        "urgent_short_circuit":not urgent or reply.risk.level == "CRITICAL" and reply.primary_agent == "risk_handoff"
            and reply.usage["calls"] == 0 and reply.storage == "skipped"}


def behavior_checks(reply, expected):
    primary = next((row for row in reply.agent_results if row.role == reply.primary_agent),None)
    checks = {"degraded":reply.degraded == expected.degraded}
    for name,actual in [("role",reply.primary_agent),("intent",reply.intent.primary),("risk",reply.risk.level),
                        ("subject",reply.subject),("references",bool(reply.references)),("status",primary.status if primary else None)]:
        target = getattr(expected,name)
        if target is not None: checks[name] = actual == target
    if expected.missing_fields: checks["missing_fields"] = bool(primary) and set(expected.missing_fields) <= set(primary.missing_fields)
    if expected.supporting_agents:
        selected = set(reply.supporting_agents)-{reply.primary_agent}
        completed = {row.role for row in reply.agent_results if row.role != reply.primary_agent}
        checks["support"] = set(expected.supporting_agents) <= (selected & completed)
    for name,target in expected.entities.items():
        fact = reply.intent.entities.get(name)
        checks["entity:"+name] = fact is not None and fact.value == target
    for name,target in expected.entity_states.items():
        fact = reply.intent.entities.get(name)
        checks["entity_state:"+name] = fact is not None and fact.state == target
    return checks


def observed_fault_checks(kinds: set[str], trace: dict) -> dict[str, bool]:
    """Require evidence of the exercised fault and the matching runtime response."""
    events = trace.get("events", [])

    def has(event_name, **fields):
        return any(entry.get("event") == event_name and all(entry.get(key) == value for key,value in fields.items())
                   for entry in events)

    def marked(fault):
        return has("synthetic_fault_triggered", fault=fault)

    def matched_role(fault, event_name, **fields):
        return any(entry.get("event") == "synthetic_fault_triggered" and entry.get("fault") == fault
                   and entry.get("role") and has(event_name, role=entry["role"], **fields) for entry in events)

    observed = {
        "memory_unavailable": has("memory_error", stage="read", error="ConnectionError"),
        "index_unavailable": has("storage_fallback", service="reference_index") or has("retrieval_vector_unavailable"),
        "tool_timeout": matched_role("tool_timeout", "tool", name="lookup_reference", status="timeout", handler_invoked=True),
        "tool_invalid_result": matched_role("tool_invalid_result", "tool", name="lookup_reference", status="invalid_result", handler_invoked=True),
        "agent_invalid_output": matched_role("agent_invalid_output", "agent_output_validation", code="invalid_output_shape"),
        "synthesis_failure": marked("synthesis_failure") and has("synthesis_error", error="RuntimeError"),
        "summary_failure": marked("summary_failure") and has("memory_error", stage="write", error="RuntimeError"),
        "language_unavailable": marked("language_unavailable") and has("classification_fallback", channel="language", error="RuntimeError"),
        # Judge runs outside the chat trace; run_dialog updates this only when its injected exception is raised.
        "judge_failure": False,
    }
    return {f"fault:{fault}_observed": observed[fault] for fault in kinds}


def recommendations_for(rows):
    suggestions = []
    for row in rows:
        if row.get("passed"): continue
        for turn in row.get("turns",[]) or [row]:
            failed = [name for name,passed in turn.get("checks",{}).items() if not passed]
            failed += ["safety:"+name for name,passed in turn.get("safety_checks",{}).items() if not passed]
            trace = turn.get("trace",{})
            recovered = {(event["role"],event["instance"]) for event in trace.get("events",[])
                         if event["event"] == "agent_output_repair" and event.get("status") == "accepted"}
            validations = [event for event in trace.get("events",[]) if event["event"] == "agent_output_validation"
                           and (event["role"],event["instance"]) not in recovered]
            for event in validations:
                failed.append("output_validation:"+event["stage"]+":"+event["code"])
            for event in trace.get("events",[]):
                if event["event"] in {"classification_fallback","memory_error","agent","tool","synthesis_error"} and (
                    event["event"] not in {"agent","tool"} or event.get("status") not in {"ok","empty_result"}):
                    failed.append(event["event"]+":"+str(event.get("status",event.get("stage",event.get("channel","error")))))
            if turn.get("judge_error"): failed.append("judge:"+turn["judge_error"])
            if turn.get("error"): failed.append("execution:"+turn["error"])
            if turn.get("quality") and turn["quality"]["mean"] < .75:
                weakest = min(["relevance","accuracy","completeness","helpfulness"],key=turn["quality"].get)
                failed.append("judge_below_threshold:"+weakest)
            if failed:
                if any(item.startswith("safety:") for item in failed): action = "优先核对失败的权限、风险或输出硬门槛，不能用平均高分抵消；保留原题回归。"
                elif any(item.startswith("judge:") for item in failed): action = "核对 Judge 调用与结构化解析；先恢复评分覆盖，不能用模拟分补齐缺测。"
                elif validations: action = "按本轮脱敏输出错误码和字段定位解析、身份或载荷问题；核对一次纠正是否成功，保持证据和医学边界后定向复测原题。"
                elif any(item.startswith("judge_below_threshold:") for item in failed): action = "结合本轮引用和预期行为检查最低评分维度，定向复测原题。"
                elif any("intent" in item or "classification" in item for item in failed): action = "核对本轮三路信号、风险门控和期望标签，检查历史是否压过了新任务。"
                elif any(item.startswith("entity") for item in failed): action = "核对用户原话出处、指代、更正和咨询对象作用域，不从摘要补事实。"
                else: action = "按本案 Trace 核对字段、来源、记忆与依赖故障，保留原题定向回归。"
                suggestions.append({"case_id":row["id"],"turn":turn.get("turn",1),"trace_id":trace.get("request_id"),
                    "evidence":list(dict.fromkeys(failed)),"action":action})
    return suggestions


async def run_intent(runtime, case, run_id, row=None):
    log = RunLog(); token = CURRENT.set(log)
    row = row if row is not None else {}
    row.update(id=case.id,kind="intent",expected=case.expected.value,classifier_expected=case.classifier_expected,
               predicted=None,classifier_predicted=None,classifier_attempted=False,quality=None,judge_error=None)
    try:
        risk = runtime.risk.screen(case.message,run_id+case.id)
        row["risk"] = risk.level
        if risk.level == "CRITICAL": row["predicted"] = "urgent"
        else:
            row["classifier_attempted"] = True
            result = await runtime.router.understand(case.message,subject="self",turn_id=log.request_id)
            row["predicted"] = row["classifier_predicted"] = result.primary.value
            row["signals"],row["fallbacks"] = result.signals,result.fallbacks
        row["checks"] = {"intent":row["predicted"] == case.expected.value}
        if case.expected_risk: row["checks"]["risk"] = risk.level == case.expected_risk
        row["safety_checks"] = {"urgent_gate":case.expected != Intent.URGENT or risk.level == "CRITICAL" and log.usage()["calls"] == 0}
        row["rule_passed"] = row["passed"] = all(row["checks"].values()) and all(row["safety_checks"].values())
    except asyncio.CancelledError: raise
    except Exception as exc:
        row.update(error=type(exc).__name__,checks={"intent":False},safety_checks={},rule_passed=False,passed=False)
    finally:
        CURRENT.reset(token)
        row["trace"] = {"request_id":log.request_id,"events":log.events}
    return row


async def judge_turn(model, turn, reply, history, *, audit=None, trace=None):
    if model is None: raise RuntimeError("judge_model_unavailable")
    payload = judge_input(turn,reply,history,trace=trace)
    if audit is not None: audit.update(judge_audit(payload))
    result = await model.json("evaluation_judge",JUDGE_INSTRUCTIONS,json.dumps(payload,ensure_ascii=False),950)
    return JudgeScore.model_validate(result).result()


async def run_dialog(runtime, case, run_id, request, budget, row=None):
    row = row if row is not None else {}
    row.update(id=case.id,kind="dialog",turns=[],total_turns=len(case.turns),quality=None,judge_error=None,checks={},passed=False,rule_passed=False)
    session,subject,history = "","self",[]
    judge = BudgetedModel(budget.base_model,budget,"judge") if budget.base_model else None
    for index,turn in enumerate(case.turns,1):
        usage_begin = len(budget.entries)
        item = {"turn":index,"checks":{},"safety_checks":{},"quality":None,"judge_error":None,"passed":False}
        row["turns"].append(item)
        try:
            async with injected_faults(runtime,case.faults,index) as kinds:
                question = Question(text=turn.message,user="eval-"+run_id+"-"+digest(case.id)[:16],session=session,subject=turn.subject or subject)
                reply = await runtime.chat(question)
                session,subject = reply.session,reply.subject
                trace = next((record for record in reversed(runtime.monitor.traces) if record["request_id"] == reply.request_id),{})
                item.update(reply=reply.model_dump(mode="json"),trace=trace,checks=behavior_checks(reply,turn.expected),
                            safety_checks=hard_checks(reply,trace,runtime,turn.expected))
                item["checks"].update(observed_fault_checks(kinds, trace))
                if request.judge:
                    try:
                        if "judge_failure" in kinds:
                            item["checks"]["fault:judge_failure_observed"] = True
                            raise RuntimeError("synthetic_judge_failure")
                        item["judge_context"] = {}
                        item["quality"] = await judge_turn(judge,turn,reply,history,audit=item["judge_context"],trace=trace)
                    except asyncio.CancelledError: raise
                    except Exception as exc: item["judge_error"] = type(exc).__name__
                item["rule_passed"] = all(item["checks"].values()) and all(item["safety_checks"].values())
                item["passed"] = item["rule_passed"] and (not request.judge or item["quality"] is not None and item["quality"]["mean"] >= .75)
                history += [{"role":"user","content":turn.message},{"role":"assistant","content":reply.answer.text}]
        except asyncio.CancelledError:
            item["error"] = "evaluation_cancelled_or_deadline"
            raise
        except Exception as exc:
            item.update(error=type(exc).__name__,checks={"execution":False})
        finally:
            entries = budget.entries[usage_begin:]
            item["runtime_usage"] = aggregate_usage([entry for entry in entries if entry["phase"] == "runtime"])
            item["judge_usage"] = aggregate_usage([entry for entry in entries if entry["phase"] == "judge"])
            item["judge_traces"] = [entry["trace"] for entry in entries if entry["phase"] == "judge"]
        if budget.exhausted:
            item.update(error="evaluation_budget_exhausted",passed=False)
            break
    row["rule_passed"] = all(item.get("rule_passed",False) for item in row["turns"])
    row["passed"] = all(item["passed"] for item in row["turns"])
    for item in row["turns"]:
        for key,value in item["checks"].items(): row["checks"][f'turn_{item["turn"]}:{key}'] = value
        for key,value in item["safety_checks"].items(): row["checks"][f'turn_{item["turn"]}:safety:{key}'] = value
    qualities = [item["quality"] for item in row["turns"] if item["quality"] is not None]
    if qualities:
        row["quality"] = {key:sum(value[key] for value in qualities)/len(qualities) for key in ["relevance","accuracy","completeness","helpfulness","mean"]}
        row["quality"]["reason"] = "逐轮 Judge 有效分的均值；缺测数量另列，不替代整组通过条件。"
    row["judge_error"] = ";".join(sorted({item["judge_error"] for item in row["turns"] if item["judge_error"]})) or None
    row["not_run_turns"] = list(range(len(row["turns"])+1,len(case.turns)+1))
    return row


def summarize(report, request):
    rows = report["cases"]
    for row in rows:
        if row["kind"] == "dialog": row["not_run_turns"] = list(range(len(row.get("turns",[]))+1,row.get("total_turns",0)+1))
    intents = [row for row in rows if row["kind"] == "intent"]
    eligible_classifier = [row for row in intents if row.get("classifier_expected") is not None]
    classifier = [row for row in eligible_classifier if row.get("classifier_attempted")]
    metrics = {"classifier":classification_metrics([row["classifier_expected"] for row in classifier],[row["classifier_predicted"] for row in classifier],CLASSIFIER_LABELS),
               "risk_gated":classification_metrics([row["expected"] for row in intents if "expected" in row],[row["predicted"] for row in intents if "expected" in row],GATED_LABELS)}
    metrics["classifier"]["eligible_nonurgent"] = len(eligible_classifier)
    metrics["classifier"]["skipped_by_risk_gate"] = sum(row.get("risk") == "CRITICAL" for row in eligible_classifier)
    metrics["classifier"]["coverage"] = len(classifier)/len(eligible_classifier) if eligible_classifier else None
    turns = [item for row in rows if row["kind"] == "dialog" for item in row.get("turns",[])]
    quality = [item["quality"] for item in turns if item["quality"] is not None]
    report.update(executed=len(rows),passed=sum(row.get("passed",False) for row in rows),
        rule_passed=sum(row.get("rule_passed",False) for row in rows),classification=metrics,
        judge={"enabled":request.judge,"total":len(turns) if request.judge else 0,"scored":len(quality),
               "errors":sum(bool(item.get("judge_error")) for item in turns),
               "missing":len(turns)-len(quality) if request.judge else 0})
    report["pass_rate"] = report["passed"]/report["total"] if report["total"] else None
    report["metrics"] = {"intent_accuracy":metrics["classifier"]["accuracy"],"intent_macro_f1":metrics["classifier"]["macro_f1"],
        "classifier_coverage":metrics["classifier"]["coverage"],
        "risk_gated_accuracy":metrics["risk_gated"]["accuracy"],"risk_gated_macro_f1":metrics["risk_gated"]["macro_f1"],
        "rule_pass_rate":report["rule_passed"]/report["total"] if report["total"] else None,
        "combined_pass_rate":report["pass_rate"],
        "judge_coverage":len(quality)/len(turns) if request.judge and turns else None,
        **{"quality_"+key:sum(value[key] for value in quality)/len(quality) if quality else None for key in ["relevance","accuracy","completeness","helpfulness","mean"]}}
    report["avg_scores"] = report["metrics"]
    report["safety_failures"] = [{"case_id":row["id"],"turn":item.get("turn",1),"gate":key}
        for row in rows for item in (row.get("turns",[]) or [row]) for key,passed in item.get("safety_checks",{}).items() if not passed]
    report["recommendations"] = recommendations_for(rows)


async def evaluate(source, request: EvaluationRequest | None = None, *, baseline_store=None, isolation_factory=EvaluationIsolation):
    request = request or EvaluationRequest()
    version,intents,dialogs = resolve_cases(request)
    budget = ModelBudget(request.limits)
    budget.base_model = source.model
    identity = uuid.uuid4().hex
    isolation = isolation_factory(source,identity,budget)
    snapshots = snapshot_preflight(isolation,dialogs)
    metadata = metadata_for(source,isolation,request,version,intents,dialogs)
    if snapshots:
        metadata["case_reference_snapshots"] = snapshots
        metadata["case_reference_scope_complete"] = len(snapshots) == len(dialogs)
    store = baseline_store or BaselineStore()
    baseline = store.preflight(request.baseline,metadata)
    selected = [("intent",case) for case in intents]+[("dialog",case) for case in dialogs]
    report = {"schema_version":"evaluation-2","run_id":identity,"dataset":version,"implementation":"independent-v1",
              "mode":"offline" if source.settings.preview or source.model is None else "online",
              "quality_evaluation":request.judge,"model":source.settings.model,"started_at":datetime.now(timezone.utc).isoformat(),
              "metadata":metadata,"limits":request.limits.model_dump(),"total":len(selected),"cases":[],"complete":False}
    started = time.perf_counter()
    current = None
    try:
        async with asyncio.timeout(request.limits.timeout_seconds):
            if selected: await isolation.create()
            for kind,case in selected:
                if budget.exhausted: break
                isolation.select_case_references(case.reference_snapshot.keys if kind == "dialog" and case.reference_snapshot else None)
                runtime = isolation.new_case()
                runtime.monitor.start()
                begin = len(budget.entries)
                current = {"id":case.id,"kind":kind,"quality":None,"judge_error":None,"checks":{},"safety_checks":{},"passed":False,"rule_passed":False}
                if kind == "intent": current.update(expected=case.expected.value,classifier_expected=case.classifier_expected,predicted=None,classifier_predicted=None)
                report["cases"].append(current)
                try:
                    row = await run_intent(runtime,case,identity,current) if kind == "intent" else await run_dialog(runtime,case,identity,request,budget,current)
                    current.update(row)
                    if budget.exhausted: current.update(error="evaluation_budget_exhausted",passed=False)
                finally:
                    entries = budget.entries[begin:]
                    current["runtime_usage"] = aggregate_usage([entry for entry in entries if entry["phase"] == "runtime"])
                    current["judge_usage"] = aggregate_usage([entry for entry in entries if entry["phase"] == "judge"])
                    await runtime.monitor.stop()
            report["stop_reason"] = budget.exhausted
    except TimeoutError:
        report["stop_reason"] = "timeout"
        if current: current.update(error="evaluation_deadline",passed=False,rule_passed=False)
    except asyncio.CancelledError: raise
    except Exception as exc:
        report["stop_reason"] = "isolation_or_execution_error:"+type(exc).__name__
        if current: current.update(error=type(exc).__name__,passed=False,rule_passed=False)
    finally:
        try: await asyncio.wait_for(isolation.cleanup(),5)
        except Exception as exc: isolation.cleanup_errors.append(type(exc).__name__)
    report["not_run"] = [case.id for _,case in selected[len(report["cases"]):]]
    report["complete"] = not report["not_run"] and not report.get("stop_reason") and not isolation.cleanup_errors
    report["cleanup_errors"] = isolation.cleanup_errors
    report["isolation"] = {"session_backend":"namespaced_redis" if isolation.redis else "local" if isolation.runtime else "not_created",
        "redis_keys_written":len(isolation.redis.written) if isolation.redis else 0,
        "temporary_database_removed":isolation.directory is None or not Path(isolation.directory.name).exists(),
        "case_state_reset":True}
    report["elapsed_ms"] = round((time.perf_counter()-started)*1000,1)
    report["usage"] = {phase:aggregate_usage([entry for entry in budget.entries if entry["phase"] == phase]) for phase in ["runtime","judge"]}
    report["budget"] = {"attempted_calls":budget.calls,"reserved_max_output_tokens":budget.reserved_output,"stop_reason":budget.exhausted}
    summarize(report,request)
    if not report["complete"]:
        report["recommendations"].append({"case_id":None,"evidence":[report.get("stop_reason"),*report["cleanup_errors"]],
                                          "action":"先解决预算、执行或清理问题，再完成同一数据集；未完成运行不用于普通基线回归。"})
    report["comparison"] = compare_baseline(report,baseline)
    report["regressions"] = report["comparison"]["regressions"]
    if baseline:
        previous = {row["id"]:row for row in baseline.get("cases",[])}
        for row in report["cases"]:
            old_turns = {item["turn"]:item for item in previous.get(row["id"],{}).get("turns",[])}
            for turn in row.get("turns",[]):
                old_quality = old_turns.get(turn["turn"],{}).get("quality")
                if not turn.get("quality") or not old_quality: continue
                drops = [change["metric"] for change in report["regressions"] if change["kind"] == "quality"
                         and change["metric"].startswith("quality_") and old_quality[change["metric"].removeprefix("quality_")] > 0
                         and exceeds_drop(old_quality[change["metric"].removeprefix("quality_")],turn["quality"][change["metric"].removeprefix("quality_")])]
                if drops:
                    report["recommendations"].append({"case_id":row["id"],"turn":turn["turn"],"trace_id":turn.get("trace",{}).get("request_id"),
                        "evidence":drops,"action":"本轮虽可能仍达通过阈值，但相对基线下降；结合实验变量和该轮来源核对下降维度。"})
    if request.baseline.save_as:
        try: report["baseline"] = store.save(request.baseline.save_as,report)
        except BaselineError as exc: report["baseline"] = {"saved":False,"reason":str(exc)}
    return report
