"""Shared helpers for appointment tests: offline runtime replies and trace readers."""
from medagent.core.contracts import Question
from support.harness import fixture, run


def sample_reply(**update):
    async def ask():
        runtime = await fixture()
        return await runtime.chat(Question(text="你好"))
    return run(ask()).model_copy(update=update)


def trace_of(runtime, reply) -> dict:
    return next(trace for trace in runtime.monitor.traces if trace["request_id"] == reply.request_id)


def judgments(runtime, reply) -> dict:
    return next(event for event in trace_of(runtime, reply)["events"] if event["event"] == "understanding")["judgments"]


def understanding_for(text, primary=None, quote=None, fields=None, field_turn="current"):
    """An understanding of `text` with an optional verified appointment quote and grounded fields from `field_turn`."""
    from medagent.core.contracts import FieldSource, GroundedField, Intent, Understanding
    entities = {name: GroundedField(name=name, value=value, state="provided",
                                    source=FieldSource(turn_id=field_turn, turn_index=1, subject="self", quote=value, origin="current_user"))
                for name, value in (fields or {}).items()}
    return Understanding(primary=primary or Intent.GENERAL, turn_id="current", entities=entities, appointment_quote=quote)


class Clock:
    """A controllable clock for the appointment store, starting at 2026-11-03 09:00 in Shanghai."""
    def __init__(self, moment=None):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        self.value = (moment or datetime(2026, 11, 3, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))).timestamp()

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def booking_store(clock):
    from medagent.appointments.catalog import HospitalCatalog
    from medagent.appointments.store import AppointmentStore
    from medagent.core.config import ROOT
    return AppointmentStore(HospitalCatalog.load(ROOT/"resources"/"mock-hospital.json"), clock=clock)


# Message → (intent, appointment_quote, grounded fields). Messages outside the table get no quote and no fields.
BOOKING_TURNS = {
    "能帮我挂明天上午神经内科的号吗？": ("visit", "帮我挂明天上午神经内科的号", {"department": "神经内科", "appointment_date": "明天", "time_window": "上午"}),
    "帮我挂明天上午神经内科的号": ("visit", "帮我挂明天上午神经内科的号", {"department": "神经内科", "appointment_date": "明天", "time_window": "上午"}),
    "帮我挂个号": ("visit", "帮我挂个号", {}),
    "帮我挂明天上午的号": ("visit", "帮我挂明天上午的号", {"appointment_date": "明天", "time_window": "上午"}),
    "帮我挂明天的号": ("visit", "帮我挂明天的号", {"appointment_date": "明天"}),
    "帮我挂 11月8日下午神经内科的号": ("visit", "帮我挂 11月8日下午神经内科的号", {"department": "神经内科", "appointment_date": "11月8日", "time_window": "下午"}),
    "头疼一周了，帮我挂明天上午神经内科的号": ("symptoms", "帮我挂明天上午神经内科的号", {"department": "神经内科", "appointment_date": "明天", "time_window": "上午"}),
    "神经内科一般看什么？": ("general", "", {}),
    "帮我把明天的预约取消": ("visit", "帮我把明天的预约取消", {"appointment_date": "明天"}),
    "不挂了，神经内科一般看什么？": ("general", "", {}),
}


class BookingModel:
    """Offline model: understanding follows BOOKING_TURNS; the appointment role proposes its candidate, then answers."""
    def __init__(self, bad_slot=False, skip_tool=False):
        from support.harness import Model
        self.base, self.bad_slot, self.skip_tool = Model(), bad_slot, skip_tool
        self.appointment_messages, self.all_messages, self.appointment_stages = [], [], []

    async def json(self, stage, system, content, budget=900):
        import json
        if stage != "understanding":
            return await self.base.json(stage, system, content, budget)
        payload = json.loads(content)
        self.all_messages.append(content)
        text, turn = payload["current"]["text"], payload["current"]["turn_id"]
        kind, quote, fields = BOOKING_TURNS.get(text, (None, "", {}))
        scores = {kind: 1} if kind else (await self.base.json(stage, system, text, budget))["scores"]
        return {"scores": scores, "tasks": [], "topic_relation": "new", "urgent_quote": "", "product_quote": "", "visit_quote": "",
                "appointment_quote": quote,
                "updates": [{"name": name, "value": value, "state": "provided", "quote": value, "turn_id": turn} for name, value in fields.items()]}

    async def message(self, stage, system, messages, tools=None, budget=1800):
        import json
        self.all_messages.append(json.dumps(messages, ensure_ascii=False))
        if stage not in {"agent_appointment", "agent_appointment_action"}:
            return await self.base.message(stage, system, messages, tools, budget)
        from support.harness import role_data
        self.appointment_messages.append(messages)
        self.appointment_stages.append(stage)
        data = role_data(messages)
        results = [block for message in messages if isinstance(message["content"], list) for block in message["content"]
                   if block.get("type") == "tool_result"]
        if self.skip_tool:
            output = {"role": "appointment", "status": "needs_clarification", "missing_fields": ["choice"],
                      "answer": {"text": "请再确认一下。", "questions": ["要不要换一天？"], "next_steps": [], "limitations": [], "reference_keys": []},
                      "payload": {"action": data["action"], "draft_ids": [], "options": []}}
            return {"content": [{"type": "text", "text": json.dumps(output, ensure_ascii=False)}]}
        if not results:
            calls = []
            for number, candidate in enumerate(data["candidates"], 1):
                key = "slot_id" if "slot_id" in candidate else "appointment_id"
                name = "propose_appointment" if key == "slot_id" else "propose_cancellation"
                value = "demo-general:ortho:2026-11-04:am" if self.bad_slot else candidate[key]
                calls.append({"type": "tool_use", "id": f"call-{number}", "name": name, "input": {key: value}})
            return {"content": calls}
        drafts = [json.loads(item["content"]).get("draft_id") if not item.get("is_error") else "draft-unknown" for item in results]
        output = {"role": "appointment", "status": "answered", "missing_fields": [],
                  "answer": {"text": "已生成待确认的预约草稿，请在下方点击确认后才会提交。", "questions": [], "next_steps": [], "limitations": [], "reference_keys": []},
                  "payload": {"action": data["action"], "draft_ids": drafts, "options": []}}
        return {"content": [{"type": "text", "text": json.dumps(output, ensure_ascii=False)}]}


def event_reason(runtime, reply):
    return next(event for event in trace_of(runtime, reply)["events"] if event["event"] == "appointment_followup")["reason"]


class SingleBookingModel(BookingModel):
    """BookingModel whose single-agent stage proposes the offered candidate, or a slot outside the search with wrong_slot."""
    def __init__(self, wrong_slot=False):
        super().__init__()
        self.wrong_slot, self.single_calls = wrong_slot, []

    async def message(self, stage, system, messages, tools=None, budget=1800):
        import json
        if not stage.startswith("agent_single_agent"):
            return await super().message(stage, system, messages, tools, budget)
        from support.harness import role_data
        self.all_messages.append(json.dumps(messages, ensure_ascii=False))
        self.single_calls.append({"system": system, "tools": [tool["name"] for tool in tools or []], "data": role_data(messages)})
        data, names = role_data(messages), [tool["name"] for tool in tools or []]
        answered = any(isinstance(message["content"], list) and any(block.get("type") == "tool_result" for block in message["content"])
                       for message in messages)
        prefix = "open_" if "open_propose_appointment" in names else ""
        candidates = data.get("appointment", {}).get("candidates", [])
        if prefix + "propose_appointment" in names and not answered and (prefix == "" or candidates and "slot_id" in candidates[0]):
            slots = (["demo-general:neurology:2026-11-05:am"] if self.wrong_slot else [item["slot_id"] for item in candidates])
            return {"content": [{"type": "tool_use", "id": f"call-{n}", "name": prefix + "propose_appointment", "input": {"slot_id": slot}}
                                for n, slot in enumerate(slots, 1)]}
        if prefix + "propose_cancellation" in names and not answered and candidates and "appointment_id" in candidates[0]:
            target = candidates[0]["appointment_id"]
            return {"content": [{"type": "tool_use", "id": "call-1", "name": prefix + "propose_cancellation", "input": {"appointment_id": target}}]}
        text = "已生成待确认的预约草稿，请在下方点击确认后才会提交。" if answered else "可以先说明想补充的信息。"
        questions = data.get("appointment", {}).get("questions", []) if not answered else []
        return {"content": [{"type": "text", "text": json.dumps({"text": text, "questions": questions[:6], "next_steps": [], "limitations": [],
                                                                "reference_keys": []}, ensure_ascii=False)}]}


async def single_runtime(model, clock, open_tools=False):
    import json
    from medagent.core.config import ROOT, Settings
    from medagent.core.contracts import Reference
    from medagent.knowledge.retrieval import ReferenceIndex, Vectorizer
    from medagent.memory import SessionMemory
    from medagent.runtime import MedicalRuntime
    settings = Settings(preview=False, embeddings="hash", architecture="single_agent", single_agent_open_tools=open_tools)
    vectors = Vectorizer("hash")
    index = ReferenceIndex(vectors, model)
    await index.upsert([Reference.model_validate(x) for x in json.loads((ROOT/"resources/references.json").read_text())])
    return MedicalRuntime(settings, model, vectors, index, SessionMemory(model=model), appointments=booking_store(clock))
