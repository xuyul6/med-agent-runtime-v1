import asyncio
from medagent.core.contracts import Risk
from medagent.core.config import ROOT
from medagent.evaluation.risk_redteam import run
from medagent.understanding.risk import RiskGate, grounded_quote

DATASET = ROOT/"resources"/"risk-redteam-degree-v1.json"
# Accepted: "突然" and a bare sign word outrank question framing, because an exclusion here could hide a real emergency
# ("突然胸痛，万一是心梗怎么办"). d-r17 predates these rules; d-r16 is new with them.
KNOWN_FALSE_ALARMS = {"d-r16", "d-r17"}


def test_degree_and_one_sided_wording_is_critical():
    report = run(DATASET)
    assert report["summary"]["missed_critical"] == []
    assert report["summary"]["critical"]["total"] == 25


def test_new_patterns_add_no_false_alarms():
    assert set(run(DATASET)["summary"]["false_alarms"]) == KNOWN_FALSE_ALARMS


def test_grounded_quote_must_come_from_the_current_turn():
    text = "帮我看下报告，我爸 呼吸 很困难"
    assert grounded_quote("呼吸很困难", text) == "呼吸很困难"
    assert grounded_quote("胸口剧痛", text) == ""
    assert grounded_quote("呼", text) == ""
    assert grounded_quote(None, text) == "" and grounded_quote(7, text) == ""


def test_escalate_keeps_followups_urgent():
    now = [100.0]
    gate = RiskGate(clock=lambda: now[0])
    risk = gate.escalate("scope")
    assert isinstance(risk, Risk) and risk.level == "CRITICAL" and risk.reasons == ["model_urgent_signal"]
    assert gate.screen("还是这样，越来越疼", "scope").level == "CRITICAL"
    assert gate.screen("还是这样，越来越疼", "other").level == "ROUTINE"
