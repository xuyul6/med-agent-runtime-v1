"""Process-local metrics and sampled agent governance; no model or network work."""
import asyncio
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import time


AGENT_STATUSES = {"ok", "failed", "timeout", "cancelled"}
TOOL_STATUSES = {"ok", "empty_result", "denied", "invalid_request", "invalid_result", "unknown_tool",
                 "budget_exhausted", "circuit_open", "timeout", "cancelled", "failed"}
QUALITY_CODES = {"invalid_tool_result", "reference_outside_tool_scope", "reference_for_another_medication",
                 "institution_source_not_supported", "visit_scope_missing", "visit_scope_not_general", "visit_context_missing"}


@dataclass(frozen=True)
class MonitorPolicy:
    interval: float = 10
    window_size: int = 40
    window_seconds: float = 300
    min_samples: int = 5
    agent_success_rate: float = .90
    agent_latency_ms: float = 15000
    tool_success_rate: float = .90
    tool_latency_ms: float = 10000
    local_tool_latency_ms: float = 1000
    penalty_step: float = .25
    max_penalty: float = .8
    fallback_penalty: float = .5
    bad_ticks: int = 2
    recovery_successes: int = 2
    recovery_ticks: int = 2
    probe_cooldown: float = 30
    probe_timeout: float = 15

    def __post_init__(self):
        for name in ["interval", "window_seconds", "agent_latency_ms", "tool_latency_ms", "local_tool_latency_ms", "probe_cooldown", "probe_timeout"]:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid_monitor_"+name)
        for name in ["window_size", "min_samples", "bad_ticks", "recovery_successes", "recovery_ticks"]:
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 10000:
                raise ValueError("invalid_monitor_"+name)
        if self.min_samples > self.window_size:
            raise ValueError("monitor_min_samples_exceeds_window")
        for name in ["agent_success_rate", "tool_success_rate", "penalty_step", "max_penalty", "fallback_penalty"]:
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError("invalid_monitor_"+name)
        if self.fallback_penalty > self.max_penalty:
            raise ValueError("monitor_fallback_exceeds_max_penalty")

    @classmethod
    def from_settings(cls, settings):
        return cls(**{name:getattr(settings, "monitor_"+name) for name in cls.__dataclass_fields__})


@dataclass
class Observation:
    at: float
    status: str
    ms: float
    invoked: bool = True
    quality: str | None = None
    filtered: int = 0


@dataclass
class Series:
    total: int = 0
    counts: Counter = field(default_factory=Counter)
    errors: Counter = field(default_factory=Counter)
    quality: Counter = field(default_factory=Counter)
    filtered: int = 0
    latency_sum_ms: float = 0
    latency_count: int = 0
    penalty: float = 0
    bad_ticks: int = 0
    good_ticks: int = 0
    action_revision: int = 0
    bad_revision: int = 0
    recovery_streak: int = 0
    next_probe: float = 0
    probe_token: object | None = None
    routing: dict | None = None


class MonitorExcluded(RuntimeError):
    pass


def error_kind(error):
    if error in {"TimeoutError", "CancelledError"}: return error.removesuffix("Error").lower()
    if error in {"ValueError", "ValidationError", "AgentOutputError"}: return "contract_error"
    if error in {"ConnectionError", "ModelFailure"}: return "dependency_error"
    if error in {"PermissionError", "ToolInputError", "ToolBudgetExceeded", "CircuitUnavailable", "MonitorExcluded"}: return "execution_rejected"
    return "other"


class Monitor:
    def __init__(self, policy: MonitorPolicy | None = None, clock=time.monotonic, wall_clock=time.time):
        self.policy, self.clock, self.wall_clock = policy or MonitorPolicy(), clock, wall_clock
        self.samples, self.tool_samples = {}, {}
        self.agent_series, self.tool_series, self.roles, self.tool_dependencies = {}, {}, {}, {}
        self.deterministic = set()
        self.traces = deque(maxlen=256)
        self.alerts, self.alert_changes = {}, deque(maxlen=128)
        self.requests = self.degraded = self.monitor_fallbacks = 0
        self.request_latencies = deque(maxlen=self.policy.window_size)
        self.request_latency_sum_ms = 0
        self.sample_revision, self.sampled_at, self.sampler_error = 0, None, None
        self.circuits = None
        self.routing_rejections = Counter()
        self._task = None

    def timestamp(self):
        return datetime.fromtimestamp(self.wall_clock(), timezone.utc).isoformat()

    def register_agent(self, name: str, role: str, deterministic=False):
        self.roles[name] = role
        self.agent_series.setdefault(name, Series())
        self.samples.setdefault(name, deque(maxlen=self.policy.window_size))
        if deterministic: self.deterministic.add(name)

    def register_tool(self, name: str, dependency=False):
        self.tool_dependencies[name] = dependency
        self.tool_series.setdefault(name, Series())
        self.tool_samples.setdefault(name, deque(maxlen=self.policy.window_size))

    def _rows(self, name, tool=False):
        rows = (self.tool_samples if tool else self.samples).setdefault(name, deque(maxlen=self.policy.window_size))
        while rows and rows[0].at < self.clock()-self.policy.window_seconds:
            rows.popleft()
        return list(rows)

    def _append(self, name, observation, tool=False):
        self._rows(name, tool)
        (self.tool_samples if tool else self.samples)[name].append(observation)
        state = (self.tool_series if tool else self.agent_series)[name]
        state.total += 1
        state.counts[observation.status] += 1
        if observation.quality: state.quality[observation.quality] += 1
        state.filtered += observation.filtered
        if observation.invoked and observation.status != "cancelled":
            state.latency_sum_ms += observation.ms
            state.latency_count += 1
        return state

    def record(self, name: str, ok: bool, milliseconds: float, *, status=None, error=None, recovery_evidence=True):
        if name not in self.agent_series: self.register_agent(name, name.rsplit(".",1)[0])
        status = status or ("ok" if ok else "failed")
        if status not in AGENT_STATUSES or not math.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError("invalid_agent_observation")
        state = self._append(name, Observation(self.clock(), status, milliseconds))
        if error: state.errors[error_kind(error)] += 1
        if status in {"failed", "timeout"} or status == "ok" and milliseconds > self.policy.agent_latency_ms:
            state.bad_revision, state.recovery_streak = state.total, 0
        elif status == "ok" and recovery_evidence and milliseconds <= self.policy.agent_latency_ms*.8:
            state.recovery_streak += 1
        else:
            state.recovery_streak = 0

    def record_tool(self, name, status, milliseconds, *, invoked=False, quality=None, filtered=0):
        # Unknown model tool names share one bounded metric bucket.
        if name not in self.tool_series:
            name = "__unknown__"
            if name not in self.tool_series: self.register_tool(name)
        if status not in TOOL_STATUSES or not math.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError("invalid_tool_observation")
        self._append(name, Observation(self.clock(), status, milliseconds, invoked,
            quality if quality in QUALITY_CODES else None, max(0, filtered)), tool=True)

    def _metrics(self, name, tool=False):
        rows = self._rows(name, tool)
        counts = Counter(row.status for row in rows)
        executed = [row for row in rows if row.invoked and row.status != "cancelled"]
        durations = sorted(row.ms for row in executed)
        successes = sum(row.status in {"ok", "empty_result"} for row in executed)
        returned = sum(counts[key] for key in ["ok", "empty_result", "invalid_result"])
        return {"samples":len(rows), "execution_samples":len(executed),
                "success_rate":successes/len(executed) if executed else None,
                "avg_ms":sum(durations)/len(durations) if durations else None,
                "p50_ms":durations[math.ceil(len(durations)*.5)-1] if durations else None,
                "p95_ms":durations[math.ceil(len(durations)*.95)-1] if durations else None,
                "window_counts":dict(counts), "quality_success_rate":counts["ok"]/returned if returned else None}

    def health(self, name):
        metrics = self._metrics(name)
        if metrics["success_rate"] is None: return 1.0
        return max(.1, metrics["success_rate"]/(1+metrics["avg_ms"]/15000))

    def routing_state(self, name):
        state = self.agent_series.get(name, Series())
        excluded = name not in self.deterministic and state.penalty >= self.policy.fallback_penalty
        role_busy = any(other.probe_token is not None and self.roles.get(key) == self.roles.get(name)
                        for key,other in self.agent_series.items())
        return {"health":self.health(name)*(1-state.penalty), "monitor_penalty":state.penalty,
                "excluded":excluded, "probe_available":excluded and not role_busy and self.clock() >= state.next_probe,
                "probe_inflight":state.probe_token is not None}

    def claim(self, name):
        state = self.agent_series[name]
        admission = self.routing_state(name)
        if not admission["excluded"]: return None
        if not admission["probe_available"]: raise MonitorExcluded("monitor_excluded_or_probe_busy")
        token = object()
        state.probe_token = token
        return token

    def release(self, name, token):
        state = self.agent_series[name]
        if token is not None and state.probe_token is token:
            state.probe_token = None
            state.next_probe = self.clock()+self.policy.probe_cooldown

    def reject(self, name):
        self.routing_rejections[name] += 1

    def _alert(self, kind, name, reasons):
        key = kind+":"+name
        if reasons:
            if key not in self.alerts:
                self.alerts[key] = {"id":key, "kind":kind, "name":name, "first_seen":self.timestamp()}
                self.alert_changes.append({"id":key, "state":"opened", "at":self.timestamp()})
            self.alerts[key].update(reasons=reasons, last_seen=self.timestamp())
        elif key in self.alerts:
            self.alerts.pop(key)
            self.alert_changes.append({"id":key, "state":"resolved", "at":self.timestamp()})

    def sample(self):
        for name,state in self.agent_series.items():
            if name in self.deterministic:
                state.penalty = 0
                continue
            metrics = self._metrics(name)
            sufficient = metrics["execution_samples"] >= self.policy.min_samples
            reasons = []
            if sufficient:
                if metrics["success_rate"] < self.policy.agent_success_rate: reasons.append("low_success_rate")
                if metrics["avg_ms"] > self.policy.agent_latency_ms: reasons.append("high_latency")
            state.bad_ticks = state.bad_ticks+1 if reasons else 0
            state.good_ticks = state.good_ticks+1 if sufficient and not reasons else 0
            if state.penalty and state.recovery_streak >= self.policy.recovery_successes and state.total > state.action_revision:
                state.penalty = max(0, round(state.penalty-self.policy.penalty_step, 6))
                state.action_revision, state.recovery_streak, state.bad_ticks = state.total, 0, 0
            elif state.bad_ticks >= self.policy.bad_ticks and state.bad_revision > state.action_revision:
                previous = state.penalty
                state.penalty = min(self.policy.max_penalty, round(state.penalty+self.policy.penalty_step, 6))
                state.action_revision, state.recovery_streak = state.total, 0
                if previous < self.policy.fallback_penalty <= state.penalty:
                    state.next_probe = self.clock()+self.policy.probe_cooldown
            if not reasons and state.penalty: reasons = ["awaiting_recovery_evidence"]
            if not reasons and "agent:"+name in self.alerts and state.good_ticks < self.policy.recovery_ticks:
                reasons = ["awaiting_recovery_confirmation"]
            self._alert("agent", name, reasons)
        for name,state in self.tool_series.items():
            metrics = self._metrics(name, tool=True)
            reasons = []
            if metrics["execution_samples"] >= self.policy.min_samples:
                limit = self.policy.tool_latency_ms if self.tool_dependencies[name] else self.policy.local_tool_latency_ms
                if metrics["success_rate"] < self.policy.tool_success_rate: reasons.append("low_success_rate")
                if metrics["avg_ms"] > limit: reasons.append("high_latency")
                if name == "lookup_reference" and metrics["quality_success_rate"] is not None and metrics["quality_success_rate"] < .5:
                    reasons.append("low_nonempty_validated_result_rate")
            circuit = self.circuits(name) if self.circuits else {}
            if circuit.get("circuit_state") == "open": reasons.append("circuit_open")
            state.good_ticks = state.good_ticks+1 if metrics["execution_samples"] >= self.policy.min_samples and not reasons else 0
            if not reasons and "tool:"+name in self.alerts and state.good_ticks < self.policy.recovery_ticks:
                reasons = ["awaiting_recovery_confirmation"]
            self._alert("tool", name, reasons)
        self.sample_revision += 1
        self.sampled_at, self.sampler_error = self.timestamp(), None

    def observe_routing(self, decision, request_id):
        now = self.timestamp()
        for candidate in decision.candidates:
            ranked = {choice.instance:(index,choice) for index,choice in enumerate(candidate.instances)}
            for name,state in self.agent_series.items():
                if self.roles[name] != candidate.role: continue
                position, choice = ranked.get(name, (None,None))
                score = candidate.score-.15*candidate.components["health"]+.15*choice.health if choice and candidate.eligible else None
                state.routing = {"request_id":request_id, "recorded_at":now, "role_score":candidate.score,
                    "routing_score":round(score,6) if score is not None else None,
                    "eligible":candidate.eligible and choice is not None, "instance_rank":position,
                    "primary":candidate.role == decision.primary_agent, "reasons":candidate.reasons}

    def observe_execution(self, decision, request_id):
        for assignment in decision.assignments:
            for name,state in self.agent_series.items():
                if self.roles[name] != assignment.role: continue
                if state.routing is None or state.routing["request_id"] != request_id: continue
                state.routing["selected"] = assignment.selected_instance == name
                state.routing["attempt_status"] = next((attempt.status for attempt in assignment.attempts if attempt.instance == name), None)

    def snapshot(self):
        while self.request_latencies and self.request_latencies[0][0] < self.clock()-self.policy.window_seconds:
            self.request_latencies.popleft()
        durations = sorted(value for _,value in self.request_latencies)
        agents, tools = {}, {}
        for name,state in self.agent_series.items():
            metrics = self._metrics(name)
            agents[name] = {**metrics, "role":self.roles[name], "total":state.total, "status_counts":dict(state.counts),
                "error_counts":dict(state.errors), "health":self.health(name), "monitor_penalty":state.penalty,
                "routing_rejections":self.routing_rejections[name],
                "excluded":state.penalty >= self.policy.fallback_penalty and name not in self.deterministic,
                "probe_inflight":state.probe_token is not None,
                "probe_after_seconds":max(0,state.next_probe-self.clock()) if state.penalty >= self.policy.fallback_penalty else None,
                "routing_score":state.routing["routing_score"] if state.routing else None, "routing_context":state.routing}
        for name,state in self.tool_series.items():
            circuit = self.circuits(name) if self.circuits else {"circuit_state":"closed", "consecutive_fails":0}
            tools[name] = {**self._metrics(name, tool=True), "total":state.total, "status_counts":dict(state.counts),
                           "quality_counts":dict(state.quality), "scope_filtered_count":state.filtered, **circuit}
            tools[name]["avg_latency_ms"] = tools[name]["avg_ms"]
        actions = {"low_success_rate":"核对错误分类、超时和实际调用记录。", "high_latency":"区分完整 Agent、工具及模型 HTTP 耗时后检查瓶颈。",
            "low_nonempty_validated_result_rate":"核对资料覆盖、药名别名和适用范围，不能用其他资料填补空检索。",
            "circuit_open":"检查依赖失败并等待冷却后的单次工具探测。", "awaiting_recovery_evidence":"通过正常请求获得新鲜的完整回答恢复证据。",
            "awaiting_recovery_confirmation":"等待足够近期样本及连续健康采样确认，当前不声明依赖仍在失败。"}
        return {"schema_version":"monitor-1", "scope":"single_process", "requests":self.requests, "degraded":self.degraded,
            "request_stats":{"total":self.requests, "samples":len(durations), "avg_ms":sum(durations)/len(durations) if durations else None,
                             "p95_ms":durations[math.ceil(len(durations)*.95)-1] if durations else None},
            "monitor_fallbacks":self.monitor_fallbacks, "agent_stats":agents, "tool_stats":tools,
            "active_alerts":list(self.alerts.values()), "alert_changes":list(self.alert_changes),
            "suggestions":[{"target":alert["id"], "reasons":alert["reasons"], "action":"".join(actions[reason] for reason in alert["reasons"])} for alert in self.alerts.values()],
            "sampler":{"running":self._task is not None and not self._task.done(), "revision":self.sample_revision,
                       "sampled_at":self.sampled_at, "error":self.sampler_error, "policy":asdict(self.policy)},
            "agents":{name:{"health":data["health"], "samples":data["samples"], "success_rate":data["success_rate"],
                             "mean_ms":data["avg_ms"]} for name,data in agents.items()}}

    def finish(self, log, degraded):
        self.requests += 1
        self.degraded += int(degraded)
        self.request_latencies.append((self.clock(),log.elapsed))
        self.request_latency_sum_ms += self.request_latencies[-1][1]
        self.monitor_fallbacks += int(any(event["event"] in {"routing", "routing_execution"} and event.get("monitor_fallback") for event in log.events))
        self.traces.append({"request_id":log.request_id, "elapsed_ms":round(log.elapsed,1), "events":log.events,
                            "usage":log.usage(), "degraded":degraded})

    async def _loop(self):
        while True:
            await asyncio.sleep(self.policy.interval)
            try: self.sample()
            except Exception as exc: self.sampler_error = type(exc).__name__

    def start(self):
        if self._task is None or self._task.done(): self._task = asyncio.create_task(self._loop(), name="medagent-monitor")

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
            self._task = None

    def prometheus(self):
        snapshot = self.snapshot()
        lines = ["# TYPE mar_requests_total counter", "# TYPE mar_degraded_total counter",
                 "# TYPE mar_monitor_fallbacks_total counter", "# TYPE mar_active_alerts gauge",
                 "# TYPE mar_request_latency_seconds summary", "# TYPE mar_request_p95_seconds gauge",
                 f"mar_request_latency_seconds_sum {self.request_latency_sum_ms/1000}",
                 f"mar_request_latency_seconds_count {self.requests}",
                 f"mar_requests_total {self.requests}", f"mar_degraded_total {self.degraded}",
                 f"mar_monitor_fallbacks_total {self.monitor_fallbacks}", f"mar_active_alerts {len(self.alerts)}"]
        if snapshot["request_stats"]["p95_ms"] is not None:
            lines.append("mar_request_p95_seconds "+str(snapshot["request_stats"]["p95_ms"]/1000))
        def emit(metric, value, **labels):
            if value is None: return
            suffix = ",".join(key+"="+json.dumps(value,ensure_ascii=False) for key,value in labels.items())
            lines.append(metric+"{"+suffix+"} "+str(value))
        for kind, series, stats in [("agent",self.agent_series,snapshot["agent_stats"]),("tool",self.tool_series,snapshot["tool_stats"])]:
            lines += ["# TYPE mar_"+kind+"_calls_total counter", "# TYPE mar_"+kind+"_latency_seconds summary"]
            for field in ["window_samples", "success_rate", "avg_seconds", "p95_seconds"]:
                lines.append("# TYPE mar_"+kind+"_"+field+" gauge")
            lines += ["# TYPE mar_agent_monitor_penalty gauge", "# TYPE mar_agent_excluded gauge",
                      "# TYPE mar_agent_routing_rejections_total counter"] if kind == "agent" else [
                      "# TYPE mar_tool_quality_success_rate gauge", "# TYPE mar_tool_scope_filtered_total counter",
                      "# TYPE mar_tool_consecutive_fails gauge", "# TYPE mar_tool_circuit_state gauge"]
            for name,state in series.items():
                labels = {"instance":name, "role":self.roles[name]} if kind == "agent" else {"tool":name}
                for status,count in state.counts.items(): emit("mar_"+kind+"_calls_total",count,**labels,status=status)
                emit("mar_"+kind+"_latency_seconds_sum",state.latency_sum_ms/1000,**labels)
                emit("mar_"+kind+"_latency_seconds_count",state.latency_count,**labels)
                emit("mar_"+kind+"_window_samples",stats[name]["samples"],**labels)
                emit("mar_"+kind+"_success_rate",stats[name]["success_rate"],**labels)
                for field in ["avg_ms", "p95_ms"]:
                    value = stats[name][field]
                    emit("mar_"+kind+"_"+field.removesuffix("_ms")+"_seconds",value/1000 if value is not None else None,**labels)
                if kind == "agent":
                    emit("mar_agent_monitor_penalty",state.penalty,**labels)
                    emit("mar_agent_excluded",int(stats[name]["excluded"]),**labels)
                    emit("mar_agent_routing_rejections_total",self.routing_rejections[name],**labels)
                else:
                    emit("mar_tool_quality_success_rate",stats[name]["quality_success_rate"],**labels)
                    emit("mar_tool_scope_filtered_total",state.filtered,**labels)
                    emit("mar_tool_consecutive_fails",stats[name]["consecutive_fails"],**labels)
                    for circuit in ["closed","open","half_open"]:
                        emit("mar_tool_circuit_state",int(stats[name]["circuit_state"] == circuit),**labels,state=circuit)
        return "\n".join(lines)+"\n"
