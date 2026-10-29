"""Per-run model budgets and per-call logs preserve concurrent provider accounting."""
from ..core.observability import CURRENT, RunLog


class EvaluationBudgetExceeded(RuntimeError):
    pass


def aggregate_usage(entries):
    if not entries: return {"calls":0,"attempted_calls":0,"measurement":"not_used","input_tokens":0,"output_tokens":0}
    known = all(item["usage"]["measurement"] == "provider" for item in entries)
    return {"calls":sum(item["usage"]["calls"] for item in entries),"attempted_calls":len(entries),
            "measurement":"provider" if known else "unavailable",
            "input_tokens":sum(item["usage"]["input_tokens"] for item in entries) if known else None,
            "output_tokens":sum(item["usage"]["output_tokens"] for item in entries) if known else None}


class ModelBudget:
    def __init__(self, limits):
        self.limits, self.calls, self.reserved_output, self.entries = limits,0,0,[]
        self.exhausted = None

    def reserve(self, requested):
        if self.calls >= self.limits.max_model_calls:
            self.exhausted = "max_model_calls"
            raise EvaluationBudgetExceeded(self.exhausted)
        remaining = self.limits.max_output_tokens-self.reserved_output
        if remaining <= 0:
            self.exhausted = "max_output_tokens"
            raise EvaluationBudgetExceeded(self.exhausted)
        granted = min(requested,self.limits.per_call_output_tokens,remaining)
        self.calls += 1
        self.reserved_output += granted
        return granted


class BudgetedModel:
    def __init__(self, base, budget, phase="runtime"):
        self.base,self.budget,self.phase = base,budget,phase

    async def _call(self, method, stage, args, requested):
        granted = self.budget.reserve(requested)
        parent,log = CURRENT.get(),RunLog()
        token = CURRENT.set(log)
        error = None
        try:
            return await getattr(self.base,method)(stage,*args,budget=granted)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            CURRENT.reset(token)
            if parent is not None:
                offset = (log.started-parent.started)*1000
                parent.events.extend({**item,"at_ms":round(item["at_ms"]+offset,2)} for item in log.events)
            self.budget.entries.append({"phase":self.phase,"stage":stage,"usage":log.usage(),"error":error,
                                        "requested_output_tokens":requested,"granted_output_tokens":granted,
                                        "trace":{"request_id":log.request_id,"events":log.events}})

    async def json(self, stage, system, content, budget=900):
        return await self._call("json",stage,(system,content),budget)

    async def message(self, stage, system, messages, tools=None, budget=1800):
        return await self._call("message",stage,(system,messages,tools),budget)
