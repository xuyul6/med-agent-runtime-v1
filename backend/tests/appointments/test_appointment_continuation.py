from datetime import date, datetime
from zoneinfo import ZoneInfo

from medagent.agents.contracts import AppointmentInput
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.appointments.catalog import HospitalCatalog
from medagent.appointments.continuation import appointment_request
from medagent.appointments.preparation import followup_for, prepare
from medagent.appointments.store import AppointmentStore, Followup
from medagent.core.config import ROOT
from medagent.core.contracts import Intent
from medagent.tools.registry import ToolBudget, ToolContext, ToolExecutor
from support.appointments import understanding_for
from support.harness import run

CATALOG = HospitalCatalog.load(ROOT/"resources"/"mock-hospital.json")
NOW = datetime(2026, 11, 3, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
TODAY = date(2026, 11, 3)


def store():
    return AppointmentStore(CATALOG, clock=lambda: NOW)


def followup(**update):
    values = dict(user="u1", session="s1", subject="self", action="create", kind="missing", fields={}, awaiting=("department",),
                  options=(), expires_at=NOW + 600)
    return Followup(**(values | update))


def ask(text, record=None, primary=None, quote=None, fields=None, field_turn="current", subject="self"):
    return appointment_request(text=text, understanding=understanding_for(text, primary, quote, fields, field_turn), followup=record,
                               subject=subject, now=NOW, catalog=CATALOG, today=TODAY)


def prepared(details, appointments=None, user="u1"):
    appointments = appointments or store()
    request = AppointmentInput(text="预约", subject="self", **details)
    context = ToolContext(DEFAULT_REGISTRY["appointment"], request, user=user, session="s1", appointments=appointments)
    preparation = run(prepare(request, context, ToolExecutor(), ToolBudget()))
    return request, preparation


def record_of(details, preparation):
    request = AppointmentInput(text="预约", subject="self", **details)
    return followup_for(preparation, request, user="u1", session="s1", subject="self", now=NOW, ttl=600)


def test_three_turn_followup_chain():
    details, reason = ask("帮我挂个号", quote="帮我挂个号")
    assert reason == "quote" and details["origin"] == "quote" and details["action"] == "create"
    _, preparation = prepared(details)
    assert preparation.kind == "missing" and preparation.awaiting == ("department", "appointment_date")
    first = record_of(details, preparation)
    assert first.kind == "missing" and first.awaiting == ("department", "appointment_date")

    details, reason = ask("神经内科", first)
    assert reason == "followup" and details["department"] == "神经内科" and details["origin"] == "followup"
    _, preparation = prepared(details)
    assert preparation.kind == "missing" and preparation.awaiting == ("appointment_date",)
    second = record_of(details, preparation)
    assert second.fields["department"] == "神经内科"

    details, reason = ask("明天上午", second)
    assert reason == "followup" and (details["appointment_date"], details["time_window"], details["department"]) == ("2026-11-04", "上午", "神经内科")
    _, preparation = prepared(details)
    assert preparation.kind == "ready" and [c.slot_id for c in preparation.candidates] == ["demo-general:neurology:2026-11-04:am"]
    assert record_of(details, preparation) is None


def test_followup_rejects_unrelated_turns():
    for text, reason in [("不挂了，神经内科一般看什么？", "abandoned"), ("神经内科一般看什么？", "new_task"), ("好的", "no_progress"), ("确认", "no_progress")]:
        assert ask(text, followup()) == (None, reason)
    assert ask("神经内科，另外我头疼三天了", followup(), primary=Intent.SYMPTOMS) == (None, "new_task")
    assert ask("神经内科") == (None, "none")


def test_followup_expiry_and_subject():
    assert ask("神经内科", followup(expires_at=NOW - 1)) == (None, "expired")
    assert ask("神经内科", followup(subject="mother")) == (None, "subject_changed")


def test_followup_invalid_value_not_replaced_by_old():
    record = followup(fields={"department": "神经内科", "appointment_date": "2026-11-04"}, awaiting=("time_window",))
    details, reason = ask("晚上", record)
    assert reason == "followup_invalid" and details["time_window"] == "晚上"
    _, preparation = prepared(details)
    assert preparation.kind == "invalid" and preparation.awaiting == ("time_window",)
    assert record_of(details, preparation) is None
    assert any("完整" in question for question in preparation.questions)
    details, reason = ask("11月31日", followup(fields={"department": "神经内科", "appointment_date": "2026-11-04"}, awaiting=("appointment_date",)))
    assert reason == "followup_invalid" and details["appointment_date"] == "11月31日"


def test_no_slot_followup_needs_replacement():
    details, _ = ask("帮我挂11月8日下午神经内科的号", quote="帮我挂11月8日下午神经内科的号",
                     fields={"department": "神经内科", "appointment_date": "11月8日", "time_window": "下午"})
    _, preparation = prepared(details)
    assert preparation.kind == "no_slot" and preparation.options and preparation.awaiting == ("appointment_date", "time_window")
    record = record_of(details, preparation)
    assert record.kind == "no_slot" and record.fields["appointment_date"] == "2026-11-08"
    assert ask("好的", record) == (None, "no_progress")
    details, reason = ask("后天下午", record)
    assert reason == "followup" and (details["appointment_date"], details["time_window"], details["department"]) == ("2026-11-05", "下午", "神经内科")


def test_new_quote_overrides_followup():
    record = followup(fields={"appointment_date": "2026-11-04"})
    details, reason = ask("帮我挂消化内科的号", record, quote="帮我挂消化内科的号", fields={"department": "消化内科"})
    assert reason == "quote" and details["department"] == "消化内科" and "appointment_date" not in details


def test_fresh_request_ignores_stale_fields():
    details, reason = ask("帮我挂神经内科的号", quote="帮我挂神经内科的号", fields={"appointment_date": "明天"}, field_turn="earlier")
    assert reason == "quote" and "appointment_date" not in details


def test_fresh_invalid_value_records_followup():
    details, _ = ask("帮我挂骨科的号", quote="帮我挂骨科的号", fields={"department": "骨科", "appointment_date": "明天"})
    _, preparation = prepared(details)
    assert preparation.kind == "invalid" and "神经内科" in preparation.options
    record = record_of(details, preparation)
    assert record.kind == "missing" and record.awaiting == ("department",) and record.fields["appointment_date"] == "2026-11-04"


def test_ambiguous_slots_ask_window():
    details, _ = ask("帮我挂明天神经内科的号", quote="帮我挂明天神经内科的号", fields={"department": "神经内科", "appointment_date": "明天"})
    _, preparation = prepared(details)
    assert preparation.kind == "ambiguous" and len(preparation.options) == 2 and preparation.awaiting == ("time_window",)
    assert record_of(details, preparation).kind == "ambiguous"


def test_ambiguous_bookings_ask():
    appointments = store()
    for slot_id in ("demo-general:derm:2026-11-04:am", "demo-community:derm:2026-11-05:pm"):
        draft, token = appointments.propose_create(user="u1", session="s", subject="self", slot_id=slot_id)
        appointments.confirm(user="u1", draft_id=draft.draft_id, token=token)
    details, _ = ask("帮我取消皮肤科的预约", quote="帮我取消皮肤科的预约", fields={"department": "皮肤科"})
    _, preparation = prepared(details, appointments)
    assert preparation.kind == "ambiguous" and len(preparation.candidates) == 2
    assert set(preparation.awaiting) == {"institution", "appointment_date", "time_window"}
    record = record_of(details, preparation)
    assert record.kind == "ambiguous" and record.action == "cancel"
    details, reason = ask("社区医院那个", record)
    assert reason == "followup" and details["institution"] == "演示社区医院"
    _, preparation = prepared(details, appointments)
    assert preparation.kind == "ready" and preparation.candidates[0].hospital == "演示社区医院（模拟）"


def test_followup_answer_phrased_as_question_or_correction():
    record = followup(fields={"department": "神经内科"}, awaiting=("appointment_date",))
    details, reason = ask("明天上午可以吗", record)
    assert reason == "followup" and details["appointment_date"] == "2026-11-04"
    record = followup(kind="no_slot", fields={"department": "神经内科", "appointment_date": "2026-11-08", "time_window": "下午"},
                      awaiting=("appointment_date", "time_window"))
    details, reason = ask("不约11月8日了，改后天下午", record)
    assert reason == "followup" and details["appointment_date"] == "2026-11-05"
    assert ask("可以吗", record) == (None, "new_task")


def test_date_question_has_no_nested_brackets():
    details, _ = ask("帮我挂个号", quote="帮我挂个号")
    _, preparation = prepared(details)
    assert all("））" not in question for question in preparation.questions)
