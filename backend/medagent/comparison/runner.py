"""Run every case under the three architecture arms from the same frozen conversation snapshot."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import random
import time
import uuid
from ..benchmark.isolation import PairedIsolation
from ..core.provider import ClaudeGateway
from ..benchmark.metrics import measured_usage, staged_usage
from ..benchmark.runner import (FixedOutputBudget, SafetyStop, experiment_metadata, infrastructure_failed, observe, observed_judge_models,
                                score_observation)
from ..evaluation.isolation import injected_faults
from ..evaluation.metrics import digest
from .contracts import ARM_SETTINGS, ARMS, PROTOCOL, ComparisonRequest, resolve_comparison
from .metrics import summarize_comparison


class ArchitectureIsolation(PairedIsolation):
    async def arm_branch(self, arm: str, snapshot, phase: str = "target"):
        """A fresh runtime whose settings differ from the base only by the arm's settings."""
        base = self.settings
        self.settings = replace(base, **ARM_SETTINGS[arm])
        try:
            return await self.branch(phase, snapshot)
        finally:
            self.settings = base


def schedule(request: ComparisonRequest, cases) -> list[dict]:
    shuffled = list(cases)
    random.Random(request.seed).shuffle(shuffled)
    orders = [list(ARMS[index:]+ARMS[:index]) for index in range(len(ARMS))]
    rows = []
    for repeat in range(request.repetitions):
        for case_index, case in enumerate(shuffled):
            order = orders[(repeat+case_index+request.seed) % len(orders)]
            rows.append({"id":f"{case.id}-{repeat+1}", "case_id":case.id, "category":case.category, "repeat":repeat+1,
                         "order":order, "judge_order":list(reversed(order)),
                         "arms":{arm:{"arm":arm, "status":"not_run", "quality":None, "judge_error":None, "rule_passed":False,
                                      "coverage":None} for arm in ARMS}})
    return rows


def broken(arm: dict) -> bool:
    """Infrastructure failure only. A partial (degraded) answer is an outcome the comparison measures."""
    return arm.get("status") in {"failed", "cancelled"} or bool(arm.get("judge_error"))


def score_arm(arm: dict, reply, case) -> None:
    arm["coverage"] = reference_coverage(reply, case)
    arm["partial"] = bool(reply.degraded) if reply is not None else None
    if case.faults:
        arm["fault_triggered"] = any(item.get("event") == "synthetic_fault_triggered" for item in arm.get("trace", {}).get("events", []))
    if reply is not None:
        arm["roles"] = [reply.primary_agent, *reply.supporting_agents]
    # The shared "degraded" check would count primary_only's honest report of an unfinished task as a rule failure.
    arm["rule_passed"] = (arm.get("status") == "completed" and all(value for name, value in arm.get("checks", {}).items()
                          if name != "degraded") and all(arm.get("safety_checks", {}).values()))


def reference_coverage(reply, case) -> bool | None:
    """Whether the answer cites an approved source from every domain the case requires."""
    if not case.reference_domains:
        return None
    if reply is None:
        return False
    cited = {ref.domain for ref in reply.references if ref.key in reply.answer.reference_keys and ref.approved}
    return set(case.reference_domains) <= cited


async def run_comparison(source, request: ComparisonRequest | None = None, *, isolation_factory=ArchitectureIsolation):
    request = (request or ComparisonRequest()).model_copy(deep=True)
    version, cases = resolve_comparison(request)
    pairs = schedule(request, cases)
    identity = uuid.uuid4().hex
    budget = FixedOutputBudget(request.limits)
    isolation = isolation_factory(source, identity, budget)
    metadata = experiment_metadata(source, isolation, request, version, cases, pairs)
    metadata.update(protocol=deepcopy(PROTOCOL), protocol_hash=digest(PROTOCOL))
    report = {"schema_version":"architecture-comparison-1", "run_id":identity, "started_at":datetime.now(timezone.utc).isoformat(),
              # Only the real provider gateway counts as online; a stand-in model is a rehearsal.
              "mode":"online" if not source.settings.preview and isinstance(source.model, ClaudeGateway) else "offline", "metadata":metadata,
              "limits":request.limits.model_dump(), "pairs":pairs, "preparation":[], "complete":False, "stop_reason":None}
    case_map = {case.id:case for case in cases}
    prepared, blocked, failures, started = {}, {}, 0, time.perf_counter()
    try:
        async with asyncio.timeout(request.limits.timeout_seconds):
            await isolation.create()
            for pair in pairs:
                if budget.exhausted or failures >= request.limits.max_infrastructure_failures:
                    report["stop_reason"] = budget.exhausted or "infrastructure_failure_limit"
                    break
                case = case_map[pair["case_id"]]
                if case.id not in prepared and case.id not in blocked:
                    prep = {"case_id":case.id, "turns":[], "snapshot":None}
                    report["preparation"].append(prep)
                    runtime = await isolation.arm_branch("multi_agent", None, "preparation")
                    user, session, subject = "comparison-"+identity, case.id, case.subject
                    for number, turn in enumerate(case.preparation, 1):
                        row = {"turn":number, "message":turn.message}
                        prep["turns"].append(row)
                        reply = await observe(runtime, turn, user, session, subject, "selective", budget, row)
                        if reply is not None: subject = reply.subject
                        if budget.exhausted or not row.get("rule_passed"):
                            blocked[case.id] = budget.exhausted or "preparation_rule_failure"
                            failures += int(infrastructure_failed(row))
                            break
                    if case.id not in blocked:
                        transcript = [message for row in prep["turns"] for message in (
                            {"role":"user", "content":row["message"]}, {"role":"assistant", "content":row["reply"]["answer"]["text"]})]
                        prepared[case.id] = await isolation.capture(runtime, user, session, subject, transcript)
                        prep["snapshot"] = {"sha256":prepared[case.id].sha256, **prepared[case.id].payload}
                if case.id in blocked:
                    for arm in pair["arms"].values(): arm["reason"] = blocked[case.id]
                    continue
                snapshot = prepared[case.id]
                pair["snapshot_hash"] = snapshot.sha256
                replies = {}
                for name in pair["order"]:
                    if budget.exhausted: break
                    arm = pair["arms"][name]
                    runtime = await isolation.arm_branch(name, snapshot)
                    restored = await isolation.capture(runtime, snapshot.user, snapshot.session, snapshot.subject, snapshot.history)
                    if restored.sha256 != snapshot.sha256: raise ValueError("snapshot_restore_mismatch")
                    arm["snapshot_hash"] = restored.sha256
                    async with injected_faults(runtime, case.faults, 1):
                        replies[name] = await observe(runtime, case.target, snapshot.user, snapshot.session, snapshot.subject,
                                                      "selective", budget, arm)
                    score_arm(arm, replies[name], case)
                    failures += int(broken(arm))
                if request.judge:
                    for name in pair["judge_order"]:
                        if budget.exhausted: break
                        if pair["arms"][name]["status"] != "not_run":
                            await score_observation(replies.get(name), case.target, snapshot, isolation, budget, pair["arms"][name])
                            failures += int(bool(pair["arms"][name].get("judge_error")))
            if budget.exhausted: report["stop_reason"] = budget.exhausted
    except TimeoutError:
        report["stop_reason"] = "timeout"
    except SafetyStop:
        report["stop_reason"] = "safety_gate_failed"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        report["stop_reason"] = "isolation_or_execution_error:"+type(exc).__name__
    finally:
        try: await asyncio.wait_for(isolation.cleanup(), 10)
        except Exception as exc: isolation.cleanup_errors.append(type(exc).__name__)
    report["cleanup_errors"] = isolation.cleanup_errors
    for pair in pairs:
        for arm in pair["arms"].values():
            if arm["status"] == "not_run": arm.setdefault("reason", report["stop_reason"] or "not_reached")
    report["complete"] = not report["stop_reason"] and not isolation.cleanup_errors and all(
        arm["status"] in {"completed", "failed"} for pair in pairs for arm in pair["arms"].values())
    report["infrastructure_failures"] = failures
    report["isolation"] = {"branches":isolation.serial, "temporary_database_removed":isolation.directory is None
                           or not Path(isolation.directory.name).exists(), "production_state_written":False}
    report["usage"] = {phase:measured_usage([entry for entry in budget.entries if entry["phase"] == phase])
                       for phase in ["preparation", "target", "judge"]}
    report["stages"] = {phase:staged_usage([entry for entry in budget.entries if entry["phase"] == phase]) for phase in report["usage"]}
    report["budget"] = {"attempted_calls":budget.calls, "reserved_max_output_tokens":budget.reserved_output, "stop_reason":budget.exhausted}
    report["observed_judge_models"] = observed_judge_models(budget.entries)
    report["elapsed_ms"] = round((time.perf_counter()-started)*1000, 1)
    summarize_comparison(report, request)
    return report
