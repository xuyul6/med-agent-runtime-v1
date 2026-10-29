import contextvars
import time
import uuid
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
        known = bool(calls) and all(type(x.get("input_tokens")) is int and x["input_tokens"] >= 0
                                   and type(x.get("output_tokens")) is int and x["output_tokens"] >= 0 for x in calls)
        return {"calls": len(calls), "measurement": "provider" if known else "unavailable",
                "input_tokens": sum(x["input_tokens"] for x in calls) if known else None,
                "output_tokens": sum(x["output_tokens"] for x in calls) if known else None}


CURRENT = contextvars.ContextVar("medical_run", default=None)


def event(kind, **fields):
    log = CURRENT.get()
    if log is not None:
        log.add(kind, **fields)
