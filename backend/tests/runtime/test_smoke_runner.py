import json
import httpx
import pytest
import medagent.smoke.common as common
from medagent.smoke import context, roles, routing
from support.harness import run

WORKFLOW = ["validate_input", "check_handoff", "prepare_evidence", "compose_answer", "validate_output"]


def serve(monkeypatch, chat, health=None):
    sent = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json=health or {"status":"ok", "preview":False, "developer_tools":True})
        if request.url.path == "/chat":
            body = json.loads(request.content)
            sent.append(body)
            return httpx.Response(200, json={"request_id":"r"+str(len(sent)), **chat(body, len(sent))})
        return httpx.Response(200, json={"events":[]})

    real = httpx.AsyncClient
    monkeypatch.setattr(common.httpx, "AsyncClient", lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs))
    return sent


def general_reply(_body, _count):
    return {"primary_agent":"general", "degraded":False, "storage":"available", "handoff_connected":False,
            "usage":{"measurement":"provider", "input_tokens":10},
            "agent_results":[{"role":"general", "status":"answered", "workflow":WORKFLOW}]}


def test_existing_report_is_preserved_without_network(monkeypatch, tmp_path):
    output = tmp_path/"smoke.json"
    output.write_text("{}")
    sent = serve(monkeypatch, general_reply)
    with pytest.raises(FileExistsError):
        run(routing.run("http://test", str(output)))
    assert output.read_text() == "{}" and not sent


def test_preview_runtime_is_rejected_before_any_chat(monkeypatch, tmp_path):
    sent = serve(monkeypatch, general_reply, health={"status":"ok", "preview":True})
    with pytest.raises(ValueError, match="healthy_online_runtime_required"):
        run(roles.run("http://test", str(tmp_path/"smoke.json")))
    assert not sent and not (tmp_path/"smoke.json").exists()


def test_user_mode_backend_is_rejected_before_any_chat(monkeypatch, tmp_path):
    sent = serve(monkeypatch, general_reply, health={"status":"ok", "preview":False, "developer_tools":False})
    with pytest.raises(ValueError, match="developer_tools_required"):
        run(roles.run("http://test", str(tmp_path/"smoke.json")))
    assert not sent and not (tmp_path/"smoke.json").exists()


def test_evaluation_cli_refuses_a_user_mode_backend_before_any_run(monkeypatch, tmp_path):
    import medagent.evaluate as cli
    posted = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status":"ok", "preview":False, "developer_tools":False})
        posted.append(request.url.path)
        return httpx.Response(404, json={"detail":"Not Found"})

    real = httpx.AsyncClient
    monkeypatch.setattr(cli.httpx, "AsyncClient", lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs))
    with pytest.raises(ValueError, match="developer_tools_required"):
        run(cli.run("http://test", str(tmp_path/"evaluation.json")))
    assert not posted and not (tmp_path/"evaluation.json").exists()


def test_routing_stops_after_two_consecutive_failures_and_keeps_partial_report(monkeypatch, tmp_path):
    output = tmp_path/"routing.json"
    sent = serve(monkeypatch, lambda body, count: {"degraded":True})
    report = run(routing.run("http://test", str(output)))
    saved = json.loads(output.read_text())
    assert len(sent) == 2 and saved["executed"] == 2 and saved["passed"] == 0 and saved["total"] == len(routing.CASES)
    assert report["environment"]["preview"] is False and all("checks" in row for row in saved["cases"])


def test_role_contract_smoke_records_passing_case_and_not_run_items(monkeypatch, tmp_path):
    output = tmp_path/"roles.json"
    serve(monkeypatch, general_reply)
    report = run(roles.run("http://test", str(output), only=["general"]))
    assert report["dataset"] == "role-contract-smoke-v1" and report["passed"] == report["executed"] == 1
    assert report["not_run"] == [] and report["quality_evaluation"] is False
    with pytest.raises(ValueError):
        run(roles.run("http://test", str(tmp_path/"other.json"), only=["unknown"]))


def test_context_smoke_carries_session_within_dialogue_only(monkeypatch, tmp_path):
    sent = serve(monkeypatch, lambda body, count: {"session":"s-"+body["user"], "degraded":True})
    run(context.run("http://test", str(tmp_path/"context.json")))
    assert len(sent) == 2 and sent[0]["session"] == "" and sent[1]["session"] == "s-"+sent[0]["user"]


def test_transport_errors_record_only_the_exception_type(monkeypatch, tmp_path):
    def broken(_body, _count):
        raise httpx.ConnectError("synthetic detail must not be stored")
    serve(monkeypatch, broken)
    report = run(routing.run("http://test", str(tmp_path/"errors.json")))
    assert [row["error"] for row in report["cases"]] == ["ConnectError", "ConnectError"]
    assert "synthetic detail" not in (tmp_path/"errors.json").read_text()
