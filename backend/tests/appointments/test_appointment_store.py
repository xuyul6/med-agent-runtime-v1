import threading
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from medagent.appointments.catalog import HospitalCatalog
from medagent.appointments.store import AppointmentError, AppointmentStore, Followup
from medagent.core.config import ROOT

SHANGHAI = ZoneInfo("Asia/Shanghai")
SLOT = "demo-general:neurology:2026-11-04:am"


class Clock:
    def __init__(self, moment: datetime):
        self.value = moment.timestamp()

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def make_store(moment=datetime(2026, 11, 3, 9, 0, tzinfo=SHANGHAI)):
    clock = Clock(moment)
    return AppointmentStore(HospitalCatalog.load(ROOT/"resources"/"mock-hospital.json"), clock=clock), clock


@pytest.fixture
def store():
    return make_store()[0]


def code(call):
    with pytest.raises(AppointmentError) as caught:
        call()
    return caught.value.code


def booked(store, user, subject="self", slot_id=SLOT):
    draft, token = store.propose_create(user=user, session="s", subject=subject, slot_id=slot_id)
    return store.confirm(user=user, draft_id=draft.draft_id, token=token)


def followup(store, user="u1", session="s1", **update):
    values = dict(user=user, session=session, subject="self", action="create", kind="missing", fields={"appointment_date": "2026-11-04"},
                  awaiting=("department",), options=(), expires_at=store.now() + 600)
    return Followup(**(values | update))


def test_create_confirm_once(store):
    draft, token = store.propose_create(user="u1", session="s1", subject="self", slot_id=SLOT)
    assert draft.summary == "演示总医院（模拟） · 神经内科 · 11月4日（周三）上午 · 就诊人：本人"
    booking = store.confirm(user="u1", draft_id=draft.draft_id, token=token)
    assert booking.status == "booked" and [b.appointment_id for b in store.bookings("u1")] == [booking.appointment_id]
    assert code(lambda: store.confirm(user="u1", draft_id=draft.draft_id, token=token)) == "draft_not_pending"


def test_confirm_rejections():
    store, clock = make_store()
    draft, token = store.propose_create(user="u1", session="s1", subject="self", slot_id=SLOT)
    assert code(lambda: store.confirm(user="u1", draft_id=draft.draft_id, token="wrong")) == "token_invalid"
    assert code(lambda: store.confirm(user="u2", draft_id=draft.draft_id, token=token)) == "user_mismatch"
    assert code(lambda: store.confirm(user="u1", draft_id="draft-missing", token=token)) == "draft_not_found"
    clock.advance(601)
    assert code(lambda: store.confirm(user="u1", draft_id=draft.draft_id, token=token)) == "draft_expired"


def test_confirm_after_midnight_rejected():
    store, clock = make_store(datetime(2026, 11, 3, 23, 58, tzinfo=SHANGHAI))
    draft, token = store.propose_create(user="u1", session="s1", subject="self", slot_id=SLOT)
    clock.advance(180)
    assert code(lambda: store.confirm(user="u1", draft_id=draft.draft_id, token=token)) == "slot_out_of_window"
    assert store.bookings("u1") == []


def test_timezone_is_fixed():
    store, _ = make_store(datetime(2026, 11, 3, 16, 30, tzinfo=ZoneInfo("UTC")))
    assert store.today() == date(2026, 11, 4)


def race(*calls):
    barrier, results = threading.Barrier(len(calls)), []

    def worker(call):
        barrier.wait()
        try:
            results.append(call().status)
        except AppointmentError as error:
            results.append(error.code)

    threads = [threading.Thread(target=worker, args=(call,)) for call in calls]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(results)


def test_concurrent_double_confirm(store):
    draft, token = store.propose_create(user="u1", session="s1", subject="self", slot_id=SLOT)
    confirm = lambda: store.confirm(user="u1", draft_id=draft.draft_id, token=token)
    assert race(confirm, confirm) == ["booked", "draft_not_pending"]
    assert len(store.bookings("u1")) == 1


def test_concurrent_last_slot(store):
    booked(store, "a"), booked(store, "b")
    drafts = [store.propose_create(user=user, session="s", subject="self", slot_id=SLOT) for user in ("u2", "u3")]
    calls = [lambda user=user, draft=draft, token=token: store.confirm(user=user, draft_id=draft.draft_id, token=token)
             for user, (draft, token) in zip(("u2", "u3"), drafts)]
    assert race(*calls) == ["booked", "slot_full"]
    assert sum(len(store.bookings(user)) for user in ("a", "b", "u2", "u3")) == 3


def test_token_stored_hashed(store):
    draft, token = store.propose_create(user="u1", session="s1", subject="self", slot_id=SLOT)
    raw = store._connection.execute("select * from drafts").fetchall()
    assert token not in repr(raw)


def test_cancel_only_own_subject(store):
    booking = booked(store, "u1", subject="mother")
    assert code(lambda: store.propose_cancel(user="u1", session="s", subject="self", appointment_id=booking.appointment_id)) == "appointment_not_found"
    draft, token = store.propose_cancel(user="u1", session="s", subject="mother", appointment_id=booking.appointment_id)
    assert draft.summary.startswith("取消 ") and draft.summary.endswith("就诊人：母亲")
    assert store.confirm(user="u1", draft_id=draft.draft_id, token=token).status == "cancelled"
    assert store.bookings("u1") == []


def test_supersede_and_already_booked(store):
    first, first_token = store.propose_create(user="u1", session="s1", subject="self", slot_id=SLOT)
    second, second_token = store.propose_create(user="u1", session="s1", subject="self", slot_id=SLOT)
    assert code(lambda: store.confirm(user="u1", draft_id=first.draft_id, token=first_token)) == "draft_not_pending"
    store.confirm(user="u1", draft_id=second.draft_id, token=second_token)
    assert code(lambda: store.propose_create(user="u1", session="s1", subject="self", slot_id=SLOT)) == "already_booked"


def test_propose_rejects_unknown_and_out_of_window(store):
    assert code(lambda: store.propose_create(user="u1", session="s", subject="self", slot_id="demo-general:ortho:2026-11-04:am")) == "slot_not_found"
    assert code(lambda: store.propose_create(user="u1", session="s", subject="self", slot_id="demo-general:neurology:2026-11-03:am")) == "slot_out_of_window"


def test_followup_take_once(store):
    store.save_followup(followup(store))
    assert store.take_followup("u1", "s1").awaiting == ("department",)
    assert store.take_followup("u1", "s1") is None
    assert store.take_followup("u1", "s2") is None


def test_followup_latest_wins(store):
    store.save_followup(followup(store))
    store.save_followup(followup(store, awaiting=("appointment_date",), fields={"department": "神经内科"}))
    taken = store.take_followup("u1", "s1")
    assert taken.awaiting == ("appointment_date",) and taken.fields == {"department": "神经内科"}
