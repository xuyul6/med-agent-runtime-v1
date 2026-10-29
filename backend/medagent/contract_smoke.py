"""Small, explicitly invoked online contract check; no Judge or quality score."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import httpx


CASES = [
    {"id":"general", "text":"你好，请介绍一下你支持哪些能力。", "role":"general", "status":"answered"},
    {"id":"symptom", "text":"我是一名成年人，咳嗽三天，最近没有变化，没有胸痛，也没有呼吸困难。我想整理症状时间线，不需要判断疾病。", "role":"symptom_guide", "status":"answered"},
    {"id":"medicine", "text":"想查成人布洛芬公开资料，请仅介绍来源与适用范围，不提供个体剂量。", "role":"drug_reference", "status":"answered", "references":True},
    {"id":"report", "text":"我在学习体检报告的参考范围，想了解“参考范围”这个通用术语，不分析个人数值。", "role":"report_explainer", "status":"answered", "references":True},
    {"id":"visit", "text":"第一次去看医生，想整理就医资料和要问医生的问题，请给通用准备清单。", "role":"visit_guide", "status":"answered", "references":True},
    {"id":"report-missing", "text":"我的报告里有一个指标偏高，但我没记住项目名、数值、单位和参考范围。请先告诉我需要补充哪些原始信息。", "role":"report_explainer", "status":"needs_clarification"},
    {"id":"collaboration", "text":"我是一名成年人，咳嗽三天，最近没有变化，想整理症状时间线，并准备看医生要带的已有资料，请分别整理。", "role":"visit_guide", "status":"answered", "support":"symptom_guide", "references":True},
]
REQUIRED_TOOLS = {
    "general":{"inspect_request_context"},
    "symptom":{"check_symptom_fields"},
    "medicine":{"check_medication_fields", "lookup_reference", "check_medication_scope"},
    "report":{"check_report_fields", "lookup_reference"},
    "visit":{"check_visit_fields", "lookup_reference", "build_visit_checklist"},
    "report-missing":{"check_report_fields", "lookup_reference"},
    "collaboration":{"check_symptom_fields", "check_visit_fields", "lookup_reference", "build_visit_checklist"},
}


def contract_checks(case: dict, reply: dict, trace: dict) -> dict[str, bool]:
    results = reply.get("agent_results", [])
    primary = next((item for item in results if item["role"] == case["role"]), {})
    usage = reply.get("usage", {})
    checks = {
        "primary_role":reply.get("primary_agent") == case["role"],
        "primary_status":primary.get("status") == case["status"],
        "not_degraded":reply.get("degraded") is False,
        "storage":reply.get("storage") == "available",
        "no_fake_handoff":reply.get("handoff_connected") is False,
        "validated_workflows":bool(results) and all(item.get("workflow") == ["validate_input", "check_handoff", "prepare_evidence", "compose_answer", "validate_output"] for item in results),
        "provider_usage":usage.get("measurement") == "provider" and isinstance(usage.get("input_tokens"), int),
        "no_provider_errors":not any(event["event"] == "model_error" for event in trace.get("events", [])),
    }
    if case.get("references"):
        checks["references"] = bool(reply.get("references"))
    if case["status"] == "needs_clarification":
        checks["clarification"] = bool(primary.get("missing_fields")) and bool(reply.get("answer", {}).get("questions"))
    if case.get("support"):
        checks["support"] = any(item["role"] == case["support"] and item["status"] == "answered" for item in results)
        checks["synthesis"] = any(event["event"] == "model" and event.get("stage") == "synthesis" for event in trace.get("events", []))
    if case.get("required_tools"):
        tool_events = [event for event in trace.get("events", []) if event["event"] == "tool"]
        checks["registered_tools"] = set(case["required_tools"]) <= {event["name"] for event in tool_events if event["status"] == "ok"}
        checks["no_tool_errors"] = all(event["status"] in {"ok", "empty_result"} for event in tool_events)
        checks["tool_budget"] = bool(tool_events) and all(event["budget_used"] <= event["budget_limit"] for event in tool_events)
    return checks


async def run(url: str, output: str, only: list[str] | None = None, require_tools: bool = False) -> dict:
    target = Path(output)
    if target.exists():
        raise FileExistsError("preserve_existing_smoke_report:"+output)
    selected = [case for case in CASES if only is None or case["id"] in only]
    if not selected or only and set(only)-{case["id"] for case in CASES}:
        raise ValueError("unknown_or_empty_smoke_cases")
    if require_tools:
        selected = [{**case, "required_tools":sorted(REQUIRED_TOOLS[case["id"]])} for case in selected]
    async with httpx.AsyncClient(timeout=240) as client:
        health_response = await client.get(url.rstrip("/")+"/health")
        health_response.raise_for_status()
        health = health_response.json()
        if health.get("preview") is not False or health.get("status") != "ok":
            raise ValueError("online_healthy_runtime_required")
        report = {"dataset":"registered-tool-smoke-v1" if require_tools else "role-contract-smoke-v1", "started_at":datetime.now(timezone.utc).isoformat(),
                  "environment":health, "judge":False, "quality_evaluation":False, "cases":[]}
        target.parent.mkdir(parents=True, exist_ok=True)
        failures = 0
        for case in selected:
            row = {"id":case["id"], "case":case, "passed":False}
            try:
                response = await client.post(url.rstrip("/")+"/chat", json={"text":case["text"], "user":"contract-smoke-"+str(uuid.uuid4())})
                response.raise_for_status()
                row["reply"] = response.json()
                trace_response = await client.get(url.rstrip("/")+"/traces/"+row["reply"]["request_id"])
                trace_response.raise_for_status()
                row["trace"] = trace_response.json()
                row["checks"] = contract_checks(case, row["reply"], row["trace"])
                row["passed"] = all(row["checks"].values())
            except Exception as exc:
                row["error"] = type(exc).__name__
            report["cases"].append(row)
            failures = 0 if row["passed"] else failures+1
            report.update(total=len(selected), executed=len(report["cases"]), passed=sum(item["passed"] for item in report["cases"]))
            report["not_run"] = [item["id"] for item in selected[len(report["cases"]):]]
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
            print(json.dumps({"id":case["id"], "passed":row["passed"], "failed_checks":[key for key, value in row.get("checks", {}).items() if not value],
                              "error":row.get("error"), "usage":row.get("reply", {}).get("usage"), "elapsed_ms":row.get("reply", {}).get("elapsed_ms")},ensure_ascii=False), flush=True)
            if failures >= 2:
                print("Stopped after two consecutive failures; original results preserved.", flush=True)
                break
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18001")
    parser.add_argument("--output", required=True)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--require-tools", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.output, args.only, args.require_tools))
