"""Historical subject-switch turn used to check Judge framing."""
import copy
import json
from types import SimpleNamespace
from medagent.evaluation.contracts import DialogTurn
from medagent.core.config import ROOT
from support.harness import role_output


class PublicDump:
    def __init__(self, value): self.value = value
    def model_dump(self, **kwargs): return copy.deepcopy(self.value)


def historical_subject_turn():
    # Tests use checked-in synthetic questions, not a local online baseline or paid replay.
    body = json.loads((ROOT/"resources/evaluation.json").read_text())
    case = next(row for row in body["dialog_cases"] if row["id"] == "consultation-subject-isolation")
    history = []
    for question in case["turns"][:2]:
        history += [{"role":"user","content":question["message"]},
                    {"role":"assistant","content":"合成历史回复，只用于区分当前轮与此前背景。"}]
    public = role_output("drug_reference",["ibuprofen-overview"])
    reply = SimpleNamespace(subject="self",primary_agent="drug_reference",
        answer=PublicDump(public["answer"]),agent_results=[PublicDump(public)],references=[])
    return DialogTurn.model_validate(case["turns"][2]),reply,history
