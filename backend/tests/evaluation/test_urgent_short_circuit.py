from types import SimpleNamespace as NS

from medagent.evaluation.runner import hard_checks


def urgent_reply(reasons, calls):
    return NS(answer=NS(model_dump=lambda: {"text":"请立即联系当地急救电话"}, reference_keys=[]), agent_results=[], references=[],
              risk=NS(level="CRITICAL", reasons=reasons), primary_agent="risk_handoff", usage={"calls":calls}, storage="skipped",
              subject="self", intent=NS(entities={}), handoff_connected=False)


def short_circuit(reply, stages):
    trace = {"events":[{"event":"model", "stage":stage} for stage in stages]}
    runtime = NS(index=NS(documents={}), tools=NS(registry=NS(specs={})))
    return hard_checks(reply, trace, runtime, NS(risk="CRITICAL"))["urgent_short_circuit"]


def test_rule_hit_still_needs_zero_model_calls():
    assert short_circuit(urgent_reply(["chest"], 0), [])
    assert not short_circuit(urgent_reply(["chest"], 1), ["understanding"])


def test_model_signal_may_use_only_the_understanding_call():
    assert short_circuit(urgent_reply(["model_urgent_signal"], 1), ["understanding"])


def test_model_signal_fails_when_work_continues_after_screening():
    assert not short_circuit(urgent_reply(["model_urgent_signal"], 2), ["understanding", "agent_symptom_guide"])


def test_model_signal_still_must_skip_storage():
    reply = urgent_reply(["model_urgent_signal"], 1)
    reply.storage = "saved"
    assert not short_circuit(reply, ["understanding"])
