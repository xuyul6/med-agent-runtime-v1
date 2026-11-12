"""Fake model that reproduces the reinforced failure-contract scenarios."""
import json
from support.harness import role_output
from support.multiturn import ContextModel


WARNING = "如出现剧烈头痛、伴随肢体无力或意识改变，请立即就近急诊或拨打急救电话。"


class RegressionModel(ContextModel):
    async def message(self,stage,system,messages,tools=None,budget=1800):
        if stage == "agent_visit_guide":
            data = role_output("visit_guide",["consultation-record"])
            data["answer"]["text"] = "请整理就医资料。"+WARNING
            return {"content":[{"type":"text", "text":json.dumps(data,ensure_ascii=False)}]}
        return await super().message(stage,system,messages,tools,budget)
