from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from medagent.app import create_app
from medagent.core.config import Settings
from support.appointments import BookingModel, Clock, booking_store
from support.harness import fixture, run

REQUEST = "能帮我挂明天上午神经内科的号吗？"


def client_for(clock):
    runtime = run(fixture(BookingModel(), preview=False, appointments=booking_store(clock)))
    return TestClient(create_app(Settings(preview=True), runtime))


def draft(client, user="u1"):
    return client.post("/chat", json={"text": REQUEST, "user": user}).json()["pending_actions"][0]


def confirm(client, action, user="u1", token=None):
    return client.post("/appointments/confirm", json={"user": user, "draft_id": action["draft_id"], "token": token or action["token"]})


def test_chat_then_confirm():
    with client_for(Clock()) as client:
        action = draft(client)
        ok = confirm(client, action)
        assert ok.status_code == 200 and ok.json()["status"] == "booked" and ok.json()["appointment"]["department"] == "神经内科"
        again = confirm(client, action)
        assert again.status_code == 409 and again.json() == {"detail": "draft_not_pending"}
        assert [b["department"] for b in client.get("/appointments", params={"user": "u1"}).json()] == ["神经内科"]
        assert client.get("/appointments", params={"user": "u2"}).json() == []


def test_confirm_errors():
    with client_for(Clock()) as client:
        action = draft(client)
        assert confirm(client, action, user="u2").json() == {"detail": "user_mismatch"}
        assert confirm(client, action, user="u2").status_code == 403
        wrong = confirm(client, action, token="wrong-token")
        assert wrong.status_code == 403 and wrong.json() == {"detail": "token_invalid"}
        missing = confirm(client, {"draft_id": "draft-missing", "token": "x"})
        assert missing.status_code == 404 and missing.json() == {"detail": "draft_not_found"}
        assert client.post("/appointments/confirm", json={"user": "u1", "draft_id": "", "token": "x"}).status_code == 422


def test_confirm_after_midnight_is_gone():
    clock = Clock(datetime(2026, 11, 3, 23, 58, tzinfo=ZoneInfo("Asia/Shanghai")))
    with client_for(clock) as client:
        action = draft(client)
        clock.advance(180)
        late = confirm(client, action)
        assert late.status_code == 410 and late.json() == {"detail": "slot_out_of_window"}
