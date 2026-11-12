"""Offline rehearsal model for the architecture comparison: answers every arm and returns fixed Judge scores."""
import json
from support.harness import Model, role_data, role_output

FIXED_SCORES = {"relevance":.8, "accuracy":.8, "completeness":.7, "helpfulness":.8, "reason":"离线演练的固定分数，不代表回答质量。"}


class RehearsalModel(Model):
    def __init__(self): self.stages = []

    async def json(self, stage, system, content, budget=900):
        self.stages.append(stage)
        if stage == "evaluation_judge":
            return dict(FIXED_SCORES)
        return await super().json(stage, system, content, budget)

    async def message(self, stage, system, messages, tools=None, budget=1800):
        self.stages.append(stage)
        if stage.startswith("agent_single_agent"):
            keys = [ref["key"] for ref in role_data(messages)["references"]][:3]
            answer = {"text":"按检索到的资料整理如下，个人情况请咨询医生或药师。", "reference_keys":keys}
            return {"content":[{"type":"text", "text":json.dumps(answer, ensure_ascii=False)}]}
        if stage == "agent_drug_reference":
            data = role_data(messages)
            refs = data["references"][:1]
            answer = role_output("drug_reference", [ref["key"] for ref in refs])
            # Copy the code-given query identity, as the role rules require of a real model.
            answer["payload"]["medication_identity"] = data["medication_identity"]
            answer["payload"]["source_scope"] = [ref["scope"] for ref in refs]
            if not refs:
                answer.update(status="needs_clarification", missing_fields=["source"])
            return {"content":[{"type":"text", "text":json.dumps(answer, ensure_ascii=False)}]}
        return await super().message(stage, system, messages, tools, budget)
