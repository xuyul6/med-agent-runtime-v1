import contextvars
import time
import uuid
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RunLog:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started: float = field(default_factory=time.perf_counter)
    events: list[dict] = field(default_factory=list)

    def add(self, event: str, **fields):
        self.events.append({"event": event, "at_ms": round(self.elapsed, 2), **fields})

    @property
    def elapsed(self):
        return (time.perf_counter() - self.started) * 1000

    def usage(self):
        calls = [x for x in self.events if x["event"] == "model"]
        known = bool(calls) and all(isinstance(x.get("input_tokens"), int) and isinstance(x.get("output_tokens"), int) for x in calls)
        return {"calls": len(calls), "measurement": "provider" if known else "unavailable",
                "input_tokens": sum(x["input_tokens"] for x in calls) if known else None,
                "output_tokens": sum(x["output_tokens"] for x in calls) if known else None}


CURRENT = contextvars.ContextVar("medical_run", default=None)


def event(kind, **fields):
    log = CURRENT.get()
    if log is not None:
        log.add(kind, **fields)


class Monitor:
    def __init__(self):
        self.samples = {}
        self.traces = deque(maxlen=256)
        self.requests = 0
        self.degraded = 0

    def record(self, name, ok, milliseconds):
        self.samples.setdefault(name, deque(maxlen=40)).append((bool(ok), milliseconds))

    def health(self, name):
        rows = self.samples.get(name, [])
        if not rows:
            return 1.0
        success = sum(ok for ok, _ in rows) / len(rows)
        latency = sum(ms for _, ms in rows) / len(rows)
        return max(.1, success * (1 / (1 + latency / 15000)))

    def snapshot(self):
        return {"requests": self.requests, "degraded": self.degraded,
                "agents": {key: {"health": round(self.health(key), 3), "samples": len(rows),
                                  "success_rate": sum(ok for ok, _ in rows) / len(rows),
                                  "mean_ms": round(sum(ms for _, ms in rows) / len(rows), 1)}
                           for key, rows in self.samples.items()}}

    def finish(self, log, degraded):
        self.requests += 1
        self.degraded += int(degraded)
        self.traces.append({"request_id": log.request_id, "elapsed_ms": round(log.elapsed, 1),
                            "events": log.events, "usage": log.usage(), "degraded": degraded})
