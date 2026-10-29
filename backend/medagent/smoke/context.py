"""Explicitly invoked synthetic multi-turn integration checks; no Judge."""
import asyncio
import uuid
from .common import OnlineSmoke, arguments, new_report


DIALOGUES = [
    {"id":"drug-correction", "turns":[
        {"text":"我想查成人布洛芬的公开资料，请介绍来源范围。", "role":"drug_reference", "field":"medication_name", "value":"布洛芬", "references":True},
        {"text":"这个药的说明资料呢？", "role":"drug_reference", "field":"medication_name", "value":"布洛芬", "references":True, "history_source":True},
        {"text":"更正，不是布洛芬，是阿莫西林。请只核对现有资料。", "role":"drug_reference", "field":"medication_name", "value":"阿莫西林", "references":False, "missing":"source"},
    ]},
    {"id":"subject-isolation", "turns":[
        {"text":"请查布洛芬的公开资料。", "role":"drug_reference", "field":"medication_name", "value":"布洛芬", "subject":"self"},
        {"text":"现在给我妈妈问，这个药的资料呢？", "role":"drug_reference", "absent":"medication_name", "missing":"medication_name", "subject":"mother"},
        {"text":"她想查阿莫西林的药品资料。", "role":"drug_reference", "field":"medication_name", "value":"阿莫西林", "subject":"mother", "references":False},
        {"text":"改回我自己，这个药来源呢？", "role":"drug_reference", "field":"medication_name", "value":"布洛芬", "subject":"self", "references":True, "history_source":True},
    ]},
    {"id":"symptom-correction", "turns":[
        {"text":"我是成年人，咳嗽三天，最近没有变化，只帮我整理记录。", "role":"symptom_guide", "field":"onset", "value":"三天"},
        {"text":"刚才说错了，不是三天，是两天。", "role":"symptom_guide", "field":"onset", "value":"两天"},
        {"text":"我现在没有咳嗽了。", "role":"symptom_guide", "field":"symptom", "state":"negated"},
    ]},
    {"id":"report-item-change", "turns":[
        {"text":"报告项目A，结果10，单位U，参考范围3-5，只整理原始字段，不做判断。", "role":"report_explainer", "field":"report_item", "value":"项目A"},
        {"text":"改看项目B，结果20，单位和范围不清楚。", "role":"report_explainer", "field":"report_item", "value":"项目B", "missing":"unit", "no_old_report_scope":True},
    ]},
]


def checks(case: dict, reply: dict, trace: dict) -> dict:
    understanding = reply.get("intent", {})
    entities = understanding.get("entities", {})
    primary = next((item for item in reply.get("agent_results", []) if item["role"] == case["role"]), {})
    result = {"role":reply.get("primary_agent") == case["role"], "not_degraded":reply.get("degraded") is False,
              "storage":reply.get("storage") == "available", "no_provider_errors":not any(item["event"] == "model_error" for item in trace.get("events", [])),
              "grounded_subject":all(item["source"]["subject"] == reply.get("subject") for item in entities.values())}
    if "field" in case:
        fact = entities.get(case["field"], {})
        result["field"] = bool(fact)
        if "value" in case: result["value"] = fact.get("value") == case["value"]
        if "state" in case: result["state"] = fact.get("state") == case["state"]
        if case.get("history_source"): result["history_source"] = fact.get("source", {}).get("origin") == "history_user"
    if "absent" in case: result["no_subject_leak"] = case["absent"] not in entities
    if "missing" in case: result["clarification"] = case["missing"] in primary.get("missing_fields", [])
    if "subject" in case: result["subject"] = reply.get("subject") == case["subject"]
    if "references" in case: result["references"] = bool(reply.get("references")) == case["references"]
    if case.get("no_old_report_scope"):
        result["no_old_report_scope"] = all(entities.get(name, {}).get("value") is None for name in ["unit", "reference_range"])
    return result


async def run(url: str, output: str) -> dict:
    report = new_report("context-smoke-v1", total=sum(len(dialogue["turns"]) for dialogue in DIALOGUES))
    async with OnlineSmoke(url, output, report) as smoke:
        for dialogue in DIALOGUES:
            session, user = "", "context-smoke-"+str(uuid.uuid4())
            for turn_index, case in enumerate(dialogue["turns"]):
                row = {"dialogue":dialogue["id"], "turn":turn_index+1, "case":case}
                await smoke.attempt(row, {"text":case["text"], "user":user, "session":session},
                                    lambda reply, trace: checks(case, reply, trace))
                session = row.get("reply", {}).get("session", session)
                if not smoke.record(row, {"dialogue":row["dialogue"], "turn":row["turn"]}):
                    return report
    return report


if __name__ == "__main__":
    args = arguments().parse_args()
    asyncio.run(run(args.url, args.output))
