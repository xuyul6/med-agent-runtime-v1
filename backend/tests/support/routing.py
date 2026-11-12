"""Structured understanding builders for routing tests."""
from medagent.core.contracts import FieldSource, GroundedField, Intent, TaskUnderstanding, Understanding


# Support tasks are admitted only when the turn asks for them, so compound fixtures need real request wording.
COMPOUND_REQUEST = "请核对布洛芬的公开资料，再解释一下化验报告的参考范围。"
FOUR_TASK_REQUEST = COMPOUND_REQUEST + "另外想整理头痛的症状，也想知道复诊要准备什么。"


def field(name, value):
    return GroundedField(name=name, value=value, state="topic" if name == "medication_name" else "provided",
        source=FieldSource(turn_id="current", turn_index=1, subject="self", quote=value, origin="current_user"))


def understanding(primary=Intent.MEDICINE, kinds=(Intent.MEDICINE, Intent.REPORT), entities=None, scores=None):
    return Understanding(primary=primary, confidence=.5, scores=scores or {kind.value:1/len(kinds) for kind in kinds},
        tasks=[TaskUnderstanding(intent=kind, goal="合成任务", confidence=.5, priority="primary" if kind == primary else "support") for kind in kinds],
        entities=entities or {}, turn_id="current")


def fixed_understanding(runtime, value):
    async def understand(*args, **kwargs):
        return value.model_copy(deep=True)
    runtime.router.understand = understand
