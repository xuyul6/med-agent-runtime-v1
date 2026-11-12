from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from medagent.agents.contracts import AppointmentInput
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.appointments.catalog import HospitalCatalog
from medagent.appointments.resolution import pending_from_issued, resolve_request
from medagent.appointments.store import AppointmentStore
from medagent.core.config import ROOT
from medagent.core.contracts import Intent
from medagent.tools.contracts import DraftResult
from medagent.tools.registry import DEFAULT_TOOLS, ToolBudget, ToolContext, ToolExecutor, ToolInputError
from support.harness import run

CATALOG = HospitalCatalog.load(ROOT/"resources"/"mock-hospital.json")
TODAY = date(2026, 11, 3)


def make_store():
    return AppointmentStore(CATALOG, clock=lambda: datetime(2026, 11, 3, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())


def request(action="create", **fields):
    return AppointmentInput(text="预约", intent=Intent.APPOINTMENT, action=action, **fields)


def context_for(store, req, user="u1"):
    return ToolContext(DEFAULT_REGISTRY["appointment"], req, user=user, session="s1", appointments=store,
                       resolution=resolve_request(req, CATALOG, store.today()))


def call(executor, name, arguments, context):
    return run(executor.call(name, arguments, context, ToolBudget()))


@pytest.mark.parametrize("role", ["general", "symptom_guide", "drug_reference", "report_explainer", "visit_guide", "risk_handoff"])
@pytest.mark.parametrize("tool", ["search_slots", "find_bookings", "propose_appointment", "propose_cancellation"])
def test_other_roles_cannot_use_appointment_tools(role, tool):
    with pytest.raises(PermissionError):
        DEFAULT_TOOLS.authorize(tool, ToolContext(DEFAULT_REGISTRY[role], request()))


def test_appointment_role_scope():
    assert set(DEFAULT_REGISTRY["appointment"].tool_scope) == {"check_appointment_fields", "search_slots", "find_bookings",
                                                               "propose_appointment", "propose_cancellation"}


def test_propose_requires_slot_from_this_search():
    store, executor = make_store(), ToolExecutor()
    context = context_for(store, request(department="神经内科", appointment_date="明天", time_window="上午"))
    [slot] = call(executor, "search_slots", {}, context).slots
    with pytest.raises(ToolInputError, match="slot_not_from_this_search"):
        call(executor, "propose_appointment", {"slot_id": "demo-general:neurology:2026-11-05:am"}, context)
    result = call(executor, "propose_appointment", {"slot_id": slot.slot_id}, context)
    assert isinstance(result, DraftResult) and result.kind == "create"
    with pytest.raises(ToolInputError, match="one_draft_per_turn"):
        call(executor, "propose_appointment", {"slot_id": slot.slot_id}, context)


def test_propose_rejects_non_unique_candidate():
    store, executor = make_store(), ToolExecutor()
    context = context_for(store, request(department="神经内科", appointment_date="明天"))
    slots = call(executor, "search_slots", {}, context).slots
    assert [slot.window for slot in slots] == ["am", "pm"]
    with pytest.raises(ToolInputError, match="candidate_not_unique"):
        call(executor, "propose_appointment", {"slot_id": slots[0].slot_id}, context)


def test_draft_result_has_no_token():
    store, executor = make_store(), ToolExecutor()
    context = context_for(store, request(department="神经内科", appointment_date="明天", time_window="上午"))
    [slot] = call(executor, "search_slots", {}, context).slots
    result = call(executor, "propose_appointment", {"slot_id": slot.slot_id}, context)
    draft, token = context.issued[result.draft_id]
    assert token not in result.model_dump_json() and "token" not in DraftResult.model_fields
    [action] = pending_from_issued(context.issued, "Asia/Shanghai")
    assert action.token == token and action.kind == "appointment_create" and action.expires_at == "2026-11-03T09:10:00+08:00"


def test_action_mismatch():
    store, executor = make_store(), ToolExecutor()
    context = context_for(store, request("cancel", department="神经内科"))
    with pytest.raises(ToolInputError, match="action_mismatch"):
        call(executor, "propose_appointment", {"slot_id": "demo-general:neurology:2026-11-04:am"}, context)


def book(store, slot_id, subject="self"):
    draft, token = store.propose_create(user="u1", session="s", subject=subject, slot_id=slot_id)
    return store.confirm(user="u1", draft_id=draft.draft_id, token=token)


def test_find_bookings_filters_hospital():
    store, executor = make_store(), ToolExecutor()
    book(store, "demo-general:derm:2026-11-04:am")
    community = book(store, "demo-community:derm:2026-11-04:am")
    context = context_for(store, request("cancel", institution="社区医院", department="皮肤科"))
    [found] = call(executor, "find_bookings", {}, context).bookings
    assert found.appointment_id == community.appointment_id and found.hospital == "演示社区医院（模拟）"
    result = call(executor, "propose_cancellation", {"appointment_id": community.appointment_id}, context)
    assert result.kind == "cancel" and result.summary.startswith("取消 演示社区医院")


def test_find_bookings_only_current_subject():
    store, executor = make_store(), ToolExecutor()
    book(store, "demo-general:derm:2026-11-04:am", subject="mother")
    assert call(executor, "find_bookings", {}, context_for(store, request("cancel"))).bookings == []


def test_resolve_request():
    derm = resolve_request(request(department="皮肤科", appointment_date="明天"), CATALOG, TODAY)
    assert derm.missing == ("institution",) and set(derm.options) == {"演示总医院（模拟）", "演示社区医院（模拟）"}
    ortho = resolve_request(request(department="骨科", appointment_date="明天"), CATALOG, TODAY)
    assert ortho.invalid == ("department",) and "神经内科" in ortho.options
    ready = resolve_request(request(department="神经科", appointment_date="明天", time_window="上午"), CATALOG, TODAY)
    assert (ready.department_id, ready.day, ready.window, ready.missing, ready.invalid) == ("neurology", date(2026, 11, 4), "am", (), ())
    assert resolve_request(request(department="神经科", appointment_date="11月31日"), CATALOG, TODAY).invalid == ("appointment_date",)
    assert resolve_request(request(department="神经科", appointment_date="今天"), CATALOG, TODAY).invalid == ("appointment_date",)
    assert resolve_request(request(department="神经科", appointment_date="明天", time_window="晚上"), CATALOG, TODAY).invalid == ("time_window",)
    cancel = resolve_request(request("cancel", appointment_date="11月31日"), CATALOG, TODAY)
    assert cancel.invalid == ("appointment_date",) and cancel.day is None
    assert resolve_request(request(), CATALOG, TODAY).missing == ("department", "appointment_date")
    both = resolve_request(request("cancel", department="皮肤科"), CATALOG, TODAY)
    assert both.missing == () and both.department_id == "derm" and both.hospital_id is None
