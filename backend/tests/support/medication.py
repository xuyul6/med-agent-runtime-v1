"""Medication evidence fixtures and fake model."""
import json
from medagent.agents.contracts import MedicationInput
from medagent.core.contracts import Intent, Reference
from medagent.core.config import ROOT
from support.harness import Model, role_data


LABEL_KEY = "advil-us-200mg-tablet-label"


SCOPED = {"medication_name":"布洛芬", "product_name":"Advil", "strength":"200 mg",
          "formulation":"口服包衣片", "population":"成人", "region":"美国", "query_type":"product_information"}


def request(**changes):
    return MedicationInput(text="核对布洛芬产品说明书", intent=Intent.MEDICINE, **{**SCOPED, **changes})


def references():
    return {ref.key:ref for raw in json.loads((ROOT/"resources/references.json").read_text()) for ref in [Reference.model_validate(raw)]}


class EvidenceModel(Model):
    """Builds bounded, deterministic contract output from actual supplied data."""
    def __init__(self): self.contexts = []

    async def message(self, stage, system, messages, tools=None, budget=1800):
        if stage != "agent_drug_reference":
            return await super().message(stage, system, messages, tools, budget)
        context = role_data(messages)
        self.contexts.append(context)
        data = role_data(messages)["validated_input"]
        ref = context["references"][0]
        is_label = ref["key"] == LABEL_KEY
        identity = {"name":data.get("medication_name")}
        for key in ["formulation", "population", "product_name", "strength", "region"]:
            identity[key] = data.get(key)
        missing = [] if is_label else [key for key in SCOPED if key not in {"query_type", "medication_name"} and not data.get(key)]
        answer = {"role":"drug_reference", "status":"answered" if is_label else "needs_clarification",
                  "missing_fields":missing, "answer":{"text":ref["text"], "reference_keys":[ref["key"]],
                  "limitations":["仅核对来源范围，不判断个人适用；非完整相互作用资料。"]},
                  "payload":{"medication_identity":identity, "source_scope":[ref["scope"]],
                             "evidence_items":[{"text":ref["text"], "reference_keys":[ref["key"]]}],
                             "unresolved_questions":[] if is_label else ["需要匹配具体制剂及地区才能核对标签。"]}}
        return {"content":[{"type":"text", "text":json.dumps(answer,ensure_ascii=False)}]}
