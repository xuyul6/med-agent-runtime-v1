from fastapi.testclient import TestClient

from medagent.app import create_app
from medagent.core.config import Settings
from support.appointments import Clock, booking_store, trace_of
from support.harness import fixture, run

SLOT = "demo-general:neurology:2026-11-04:am"


def client_for(store):
    runtime = run(fixture(appointments=store))
    return TestClient(create_app(Settings(preview=True), runtime)), runtime


def draft(client, user="u1", **body):
    return client.post("/appointments/drafts", json={"user": user, "subject": "self", **body})


def test_manual_create_then_confirm():
    store = booking_store(Clock())
    client, runtime = client_for(store)
    with client:
        catalog = client.get("/appointments/catalog").json()
        assert [h["name"] for h in catalog["hospitals"]] == ["演示总医院（模拟）", "演示社区医院（模拟）"]
        assert {s["id"]: s["label"] for s in catalog["subjects"]}["mother"] == "母亲"
        assert (catalog["first_day"], catalog["last_day"]) == ("2026-11-04", "2026-11-17")
        slots = client.get("/appointments/slots", params={"hospital_id": "demo-general", "department_id": "neurology",
                                                          "day": "2026-11-04", "window": "am"}).json()
        assert [s["slot_id"] for s in slots] == [SLOT] and slots[0]["remaining"] == 3
        response = draft(client, action="create", slot_id=SLOT)
        assert response.status_code == 200
        action = response.json()
        assert action["kind"] == "appointment_create" and action["summary"].endswith("就诊人：本人")
        assert action["token"] not in str(runtime.monitor.traces)
        assert client.post("/appointments/confirm", json={"user": "u1", "draft_id": action["draft_id"], "token": action["token"]}).status_code == 200
        assert [b["department"] for b in client.get("/appointments", params={"user": "u1"}).json()] == ["神经内科"]


def test_manual_cancel_only_own_booking():
    store = booking_store(Clock())
    first, token = store.propose_create(user="u1", session="s", subject="self", slot_id=SLOT)
    booking = store.confirm(user="u1", draft_id=first.draft_id, token=token)
    client, _ = client_for(store)
    with client:
        assert draft(client, user="u2", action="cancel", appointment_id=booking.appointment_id).json() == {"detail": "appointment_not_found"}
        action = draft(client, action="cancel", appointment_id=booking.appointment_id).json()
        assert action["kind"] == "appointment_cancel"
        client.post("/appointments/confirm", json={"user": "u1", "draft_id": action["draft_id"], "token": action["token"]})
        assert client.get("/appointments", params={"user": "u1"}).json() == []


def test_manual_errors():
    store = booking_store(Clock())
    for user in ("a", "b", "c"):
        made, token = store.propose_create(user=user, session="s", subject="self", slot_id=SLOT)
        store.confirm(user=user, draft_id=made.draft_id, token=token)
    client, _ = client_for(store)
    with client:
        full = draft(client, action="create", slot_id=SLOT)
        assert full.status_code == 409 and full.json() == {"detail": "slot_full"}
        today = draft(client, action="create", slot_id="demo-general:neurology:2026-11-03:am")
        assert today.status_code == 410
        assert draft(client, action="create").status_code == 422
        assert draft(client, action="cancel", slot_id=SLOT).status_code == 422
        assert client.post("/appointments/drafts", json={"user": "u1", "subject": "boss", "action": "create", "slot_id": SLOT}).status_code == 422
        assert client.get("/appointments/slots", params={"hospital_id": "demo-general", "department_id": "ortho", "day": "2026-11-04"}).status_code == 404
        assert client.get("/appointments/slots", params={"hospital_id": "demo-general", "department_id": "neurology", "day": "2026-11-03"}).status_code == 410
        assert client.get("/appointments/slots", params={"hospital_id": "demo-general", "department_id": "neurology", "day": "2026-11-04", "window": "am"}).json() == []


def test_list_carries_subject_id_for_cancel():
    store = booking_store(Clock())
    made, token = store.propose_create(user="u1", session="s", subject="mother", slot_id=SLOT)
    store.confirm(user="u1", draft_id=made.draft_id, token=token)
    client, _ = client_for(store)
    with client:
        [booking] = client.get("/appointments", params={"user": "u1"}).json()
        assert (booking["subject"], booking["subject_label"]) == ("mother", "母亲")
        assert draft(client, action="cancel", appointment_id=booking["appointment_id"], subject=booking["subject"]).status_code == 200
