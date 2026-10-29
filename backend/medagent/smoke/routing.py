"""Explicit online routing smoke checks with synthetic requests; no Judge."""
import asyncio
import uuid
from .common import OnlineSmoke, arguments, new_report


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
    report = new_report("routing-smoke-v1", total=len(CASES))
    async with OnlineSmoke(url, output, report) as smoke:
        for case in CASES:
            row = await smoke.attempt({"case":case}, {"text":case["text"], "user":"routing-smoke-"+str(uuid.uuid4())},
                                      lambda reply, trace: checks(case, reply, trace))
            if not smoke.record(row, {"id":case["id"]}):
                break
    return report


if __name__ == "__main__":
    args = arguments().parse_args()
    asyncio.run(run(args.url, args.output))
