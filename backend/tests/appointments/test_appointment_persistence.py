from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from medagent.agents.contracts import AppointmentInput
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.app import create_app
from medagent.appointments.catalog import HospitalCatalog
from medagent.appointments.preparation import prepare
from medagent.appointments.store import AppointmentStore
from medagent.core.config import ROOT, Settings
from medagent.core.contracts import Intent
from medagent.tools.registry import ToolBudget, ToolContext, ToolExecutor
from support.appointments import Clock, booking_store
from support.harness import fixture, run

CATALOG = HospitalCatalog.load(ROOT/"resources"/"mock-hospital.json")
SHANGHAI = ZoneInfo("Asia/Shanghai")


def book(store, slot_id, user="u1"):
    draft, token = store.propose_create(user=user, session="s", subject="self", slot_id=slot_id)
    return store.confirm(user=user, draft_id=draft.draft_id, token=token)


def test_store_persists_across_instances(tmp_path):
    path = str(tmp_path/"sub"/"a.sqlite3")
    clock = Clock()
    book(AppointmentStore(CATALOG, path, clock=clock), "demo-general:neurology:2026-11-04:am")
    assert [b.slot.slot_id for b in AppointmentStore(CATALOG, path, clock=clock).bookings("u1")] == ["demo-general:neurology:2026-11-04:am"]


def test_upcoming_hides_ended_slots():
    clock = Clock()
    store = booking_store(clock)
    book(store, "demo-general:neurology:2026-11-04:pm")
    book(store, "demo-general:neurology:2026-11-04:am")
    windows = lambda: [b.slot.window for b in store.upcoming("u1")]
    clock.value = datetime(2026, 11, 4, 11, 59, tzinfo=SHANGHAI).timestamp()
    assert windows() == ["am", "pm"]
    clock.value = datetime(2026, 11, 4, 12, 0, tzinfo=SHANGHAI).timestamp()
    assert windows() == ["pm"]
    clock.value = datetime(2026, 11, 4, 18, 0, tzinfo=SHANGHAI).timestamp()
    assert windows() == [] and len(store.bookings("u1")) == 2


def test_find_bookings_skips_today():
    clock = Clock()
    store = booking_store(clock)
    book(store, "demo-general:neurology:2026-11-04:am")
    clock.value = datetime(2026, 11, 4, 8, 0, tzinfo=SHANGHAI).timestamp()
    request = AppointmentInput(text="取消", intent=Intent.APPOINTMENT, action="cancel")
    context = ToolContext(DEFAULT_REGISTRY["appointment"], request, user="u1", session="s", appointments=store)
    assert run(prepare(request, context, ToolExecutor(), ToolBudget())).kind == "no_booking"


def test_settings_load_defaults_to_file(monkeypatch):
    monkeypatch.delenv("MAR_APPOINTMENT_DB", raising=False)
    monkeypatch.setattr("medagent.core.config.load_dotenv", lambda *args, **kwargs: None)
    assert Settings.load().appointment_db.endswith("data/appointments.sqlite3")
    assert Settings().appointment_db == ":memory:"


def test_list_endpoint_hides_ended():
    clock = Clock()
    store = booking_store(clock)
    book(store, "demo-general:neurology:2026-11-04:am")
    book(store, "demo-general:neurology:2026-11-05:am")
    clock.value = datetime(2026, 11, 4, 13, 0, tzinfo=SHANGHAI).timestamp()
    runtime = run(fixture(appointments=store))
    with TestClient(create_app(Settings(preview=True), runtime)) as client:
        assert [b["day"] for b in client.get("/appointments", params={"user": "u1"}).json()] == ["2026-11-05"]
