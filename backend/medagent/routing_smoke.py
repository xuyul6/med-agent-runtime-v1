"""Explicit online routing smoke checks with synthetic requests; no Judge."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import httpx


CASES = [
    {"id":"general", "text":"你好，请介绍你支持哪些能力。", "roles":["general"]},
    {"id":"symptom-visit", "text":"我是一名成年人，咳嗽三天，最近没有变化，想整理症状时间线，并准备看医生要带的已有资料，请分别整理。", "roles":["symptom_guide","visit_guide"]},
    {"id":"report-drug", "text":"请分别解释报告中的“参考范围”这个通用术语，并查询成人布洛芬公开资料的来源范围；不分析个人数值，不提供个人剂量。", "roles":["report_explainer","drug_reference"]},
]


def checks(case, reply, trace):
    decision = reply.get("routing_decision", {})
    assignments, results = decision.get("assignments", []), reply.get("agent_results", [])
    result = {"roles":set(case["roles"]) == {item["role"] for item in assignments} == {item["role"] for item in results},
        "answered":bool(results) and all(item["status"] == "answered" for item in results),
        "not_degraded":reply.get("degraded") is False, "storage":reply.get("storage") == "available",
        "primary_matches_decision":reply.get("primary_agent") == decision.get("primary_agent"),
        "support_matches_decision":reply.get("supporting_agents") == decision.get("supporting_agents"),
        "engineering_confidence":decision.get("confidence_kind") == "engineering_score",
        "actual_instances":bool(assignments) and all(item.get("selected_instance") and item.get("attempts") and
            any(output["instance"] == item["selected_instance"] and output["task_id"] == item["task_id"] for output in results) for item in assignments),
        "no_provider_errors":not any(item["event"] == "model_error" for item in trace.get("events", []))}
    if len(case["roles"]) > 1:
        result["merged"] = decision.get("collaboration_status") == "merged"
        result["references"] = bool(reply.get("references"))
    return result


async def run(url: str, output: str):
    path = Path(output)
    if path.exists(): raise FileExistsError("preserve_existing_routing_report")
    report = {"dataset":"routing-smoke-v1", "synthetic":True, "judge":False,
              "started_at":datetime.now(timezone.utc).isoformat(), "total":len(CASES), "cases":[]}
    async with httpx.AsyncClient(timeout=240) as client:
        health = await client.get(url.rstrip("/")+"/health")
        health.raise_for_status()
        report["environment"] = health.json()
        if report["environment"].get("preview") or report["environment"].get("status") != "ok":
            raise ValueError("healthy_online_runtime_required")
        failures = 0
        for case in CASES:
            row = {"case":case, "passed":False}
            try:
                response = await client.post(url.rstrip("/")+"/chat",json={"text":case["text"], "user":"routing-smoke-"+str(uuid.uuid4())})
                response.raise_for_status()
                row["reply"] = response.json()
                traced = await client.get(url.rstrip("/")+"/traces/"+row["reply"]["request_id"])
                traced.raise_for_status()
                row["trace"] = traced.json()
                row["checks"] = checks(case,row["reply"],row["trace"])
                row["passed"] = all(row["checks"].values())
            except Exception as exc:
                row["error"] = type(exc).__name__
            report["cases"].append(row)
            report.update(executed=len(report["cases"]),passed=sum(item["passed"] for item in report["cases"]))
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
            print(json.dumps({"id":case["id"], "passed":row["passed"],
                "failed_checks":[key for key,value in row.get("checks",{}).items() if not value],
                "error":row.get("error"), "usage":row.get("reply",{}).get("usage")},ensure_ascii=False),flush=True)
            failures = 0 if row["passed"] else failures+1
            if failures >= 2: break
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:18001")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    asyncio.run(run(args.url,args.output))
