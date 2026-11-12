"""Fake model for report reference-range repair scenarios."""
import json
from support.harness import Model, role_output


UNSUPPORTED = "参考范围是统计得到的一般区间。"


SECRET = "SYNTHETIC_REJECTED_DRAFT_MARKER"


class ReportRepairModel(Model):
    def __init__(self, outcome="corrected"):
        self.outcome = outcome
        self.calls = []

    async def message(self, stage, system, messages, tools=None, budget=1800):
        if stage != "agent_report_explainer":
            return await super().message(stage,system,messages,tools,budget)
        repair = len(messages) > 1 and isinstance(messages[-1]["content"],str)
        self.calls.append({"repair":repair,"tools":tools is not None,
                           "feedback":json.loads(messages[-1]["content"]) if repair else None})
        if repair and self.outcome == "tool":
            return {"content":[{"type":"tool_use", "id":"forbidden", "name":"lookup_reference", "input":{}}]}
        data = role_output("report_explainer",["lab-context"])
        text = "参考范围可能因检测方法与人群不同而变化。"
        if not repair or self.outcome == "persistent": text = UNSUPPORTED+SECRET
        if repair and self.outcome == "boundary": text = "你患有某种疾病。"
        data["answer"]["text"] = text
        data["payload"]["term_explanations"][0]["text"] = text
        if repair and self.outcome == "unknown_reference": data["answer"]["reference_keys"] = ["not-an-approved-reference"]
        return {"content":[{"type":"text","text":json.dumps(data,ensure_ascii=False)}]}
