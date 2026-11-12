import json

import pytest
from pydantic import ValidationError

from medagent.agents.checks import checked_answer
from medagent.agents.contracts import AppointmentInput, AppointmentOutput
from medagent.agents.execution import AgentOutputError
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.agents.skills import Skills
from medagent.core.config import Settings
from medagent.core.contracts import Intent, PendingAction
from medagent.understanding.classification import CLASSES, distribution
from support.appointments import sample_reply

DRAFT_TEXT = "已生成待确认的预约草稿，请在下方点击确认后才会提交。"


def answer(text):
    return {"text": text, "questions": [], "next_steps": [], "limitations": [], "reference_keys": []}


def test_appointment_intent_outside_classification():
    assert Intent.APPOINTMENT not in CLASSES and set(distribution({}).keys()) == {"general", "symptoms", "medicine", "visit", "report"}


def test_registry_has_appointment_role():
    profile = DEFAULT_REGISTRY["appointment"]
    assert profile.knowledge_domains == frozenset({Intent.APPOINTMENT}) and not profile.deterministic


def test_answered_requires_one_draft():
    with pytest.raises(ValidationError):
        AppointmentOutput.model_validate({"role": "appointment", "status": "answered", "answer": {"text": DRAFT_TEXT},
                                          "missing_fields": [], "payload": {"action": "create", "draft_ids": []}})
    AppointmentOutput.model_validate({"role": "appointment", "status": "answered", "answer": {"text": DRAFT_TEXT},
                                      "missing_fields": [], "payload": {"action": "create", "draft_ids": ["draft-1"]}})


def test_gaps():
    assert AppointmentInput(text="x", intent=Intent.APPOINTMENT, action="create").information_gaps() == ["department", "appointment_date"]
    assert AppointmentInput(text="x", intent=Intent.APPOINTMENT, action="cancel").information_gaps() == []


def test_report_dump_redacts_token():
    reply = sample_reply(pending_actions=[PendingAction(kind="appointment_create", draft_id="d1", token="secret-token", summary="s",
                                                        expires_at="2026-11-04T09:10:00+08:00")])
    dumped = reply.report_dump()
    assert "secret-token" not in json.dumps(dumped) and dumped["pending_actions"][0]["draft_id"] == "d1"
    assert reply.pending_actions[0].token == "secret-token"


def test_draft_wording_passes_boundary_check():
    checked_answer(answer(DRAFT_TEXT), {})
    with pytest.raises(AgentOutputError):
        checked_answer(answer("已为你预约好明天上午的号。"), {})


def test_skills_load_and_agree():
    documents = Skills(Settings().skill_directory).snapshot().documents
    assert "appointment" in documents
    assert "模拟预约由预约角色负责" in documents["visit_guide"].body and "模拟预约由预约角色负责" in documents["general"].body
