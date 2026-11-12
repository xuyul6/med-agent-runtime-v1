import pytest
from fastapi.testclient import TestClient
from medagent.app import create_app
from medagent.core.config import UNHASHED_SETTINGS, Settings
from support.harness import fixture, run

# Every developer route, with a body that would otherwise fail validation, so the 404 must come before it.
DEVELOPER_ROUTES = [
    ("get", "/runtime", None), ("get", "/monitor", None), ("get", "/agents", None), ("get", "/tools", None),
    ("get", "/traces/any-request", None), ("get", "/knowledge", None), ("post", "/knowledge", {"documents": "invalid"}),
    ("get", "/skills", None), ("post", "/skills/reload", {}), ("post", "/eval/run", {"unknown": 1}),
    ("post", "/benchmark/run", {"unknown": 1}), ("post", "/comparison/run", {"unknown": 1}),
]


def client_for(developer_tools: bool) -> TestClient:
    return TestClient(create_app(Settings(preview=True, developer_tools=developer_tools), run(fixture())))


def call(client, method, path, body):
    return client.post(path, json=body) if method == "post" else client.get(path)


@pytest.mark.parametrize("method,path,body", DEVELOPER_ROUTES)
def test_developer_routes_look_absent_by_default(method, path, body):
    with client_for(False) as client:
        response = call(client, method, path, body)
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_user_routes_and_metrics_stay_open_when_developer_tools_are_off():
    with client_for(False) as client:
        assert client.get("/health").json()["developer_tools"] is False
        reply = client.post("/chat", json={"text": "介绍一下你的功能", "user": "alice"})
        assert reply.status_code == 200
        assert client.get("/appointments/catalog").status_code == 200
        assert client.get("/appointments", params={"user": "alice"}).status_code == 200
        assert client.get("/metrics").status_code == 200
        # A real request id still gives nothing away.
        assert client.get("/traces/" + reply.json()["request_id"]).status_code == 404


def test_developer_routes_answer_when_switched_on():
    with client_for(True) as client:
        assert client.get("/health").json()["developer_tools"] is True
        for path in ["/runtime", "/monitor", "/agents", "/tools", "/knowledge", "/skills"]:
            assert client.get(path).status_code == 200, path
        reply = client.post("/chat", json={"text": "介绍一下你的功能", "user": "alice"}).json()
        assert client.get("/traces/" + reply["request_id"]).status_code == 200
        # Past the switch, the routes apply their own rules: preview refuses paid runs, and bodies are validated.
        assert client.post("/eval/run").status_code == 409
        assert client.post("/knowledge", json={"documents": "invalid"}).status_code == 422


@pytest.mark.parametrize("value,expected", [(None, False), ("false", False), ("true", True), ("TRUE", True), ("1", False)])
def test_switch_is_read_from_environment(value, expected):
    env = {} if value is None else {"MAR_DEVELOPER_TOOLS": value}
    assert Settings._from(env).developer_tools is expected


def test_switch_is_off_by_default_and_not_part_of_report_settings_hash():
    assert Settings().developer_tools is False
    assert "developer_tools" in UNHASHED_SETTINGS
