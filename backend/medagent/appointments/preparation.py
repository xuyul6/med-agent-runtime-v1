"""Check an appointment request and query candidates before any model sees it; both arms ask the same follow-up questions."""
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from ..agents.contracts import AppointmentInput
from ..tools.contracts import BookingView, SlotView
from ..tools.registry import ToolBudget, ToolContext, ToolExecutor
from .params import WINDOW_NAMES
from .resolution import Resolution, resolve_request
from .store import Followup

Kind = Literal["ready", "missing", "invalid", "no_slot", "no_booking", "ambiguous"]
EXAMPLE = "请重新说一次完整的预约请求，例如“帮我挂明天上午神经内科的号”。"


@dataclass(frozen=True)
class Preparation:
    kind: Kind
    resolution: Resolution
    candidates: tuple[SlotView | BookingView, ...] = ()
    questions: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    awaiting: tuple[str, ...] = ()


def day_text(iso: str) -> str:
    year, month, day = iso.split("-")
    return f"{int(month)}月{int(day)}日"


def slot_text(slot: SlotView) -> str:
    return f"{day_text(slot.day)}{WINDOW_NAMES[slot.window]}（剩 {slot.remaining} 个）"


def booking_text(booking: BookingView) -> str:
    return f"{booking.hospital} · {booking.department} · {day_text(booking.day)}{WINDOW_NAMES[booking.window]}"


def _range(store) -> str:
    today, horizon = store.today(), store.catalog.horizon_days
    return f"可约明天起 {horizon} 天内（{day_text((today + timedelta(days=1)).isoformat())}至{day_text((today + timedelta(days=horizon)).isoformat())}）"


def _invalid_questions(request: AppointmentInput, resolution: Resolution, store) -> list[str]:
    questions = {
        "institution": f"没有找到“{request.institution}”，模拟医院可选：{'、'.join(h.name for h in store.catalog.hospitals)}。",
        "department": f"模拟医院没有“{request.department}”，可选：{'、'.join(store.catalog.department_names())}。",
        "appointment_date": f"“{request.appointment_date}”不是可约的日期，{_range(store)}。",
        "time_window": f"模拟医院只有上午和下午的号，“{request.time_window}”约不了，请说上午或下午。",
    }
    return [questions[name] for name in resolution.invalid]


def _missing_questions(missing: tuple[str, ...], resolution: Resolution, store) -> list[str]:
    questions = {
        "institution": f"这个科室两家模拟医院都有，想去哪家？可选：{'、'.join(resolution.options)}。",
        "department": f"想挂哪个科室？可选：{'、'.join(store.catalog.department_names())}。",
        "appointment_date": f"想约哪一天？上午还是下午？（{_range(store)}）",
    }
    return [questions[name] for name in missing if name in questions]


def _next_slots(store, resolution: Resolution, limit: int = 3) -> list[SlotView]:
    found, day = [], resolution.day
    while len(found) < limit and day < store.today() + timedelta(days=store.catalog.horizon_days):
        day += timedelta(days=1)
        found += [SlotView(slot_id=slot.slot_id, hospital=slot.hospital, department=slot.department, day=slot.day.isoformat(),
                           window=slot.window, remaining=remaining)
                  for slot, remaining in store.search(resolution.hospital_id, resolution.department_id, day, None)]
    return found[:limit]


def _distinguishing(bookings: list[BookingView]) -> tuple[str, ...]:
    attributes = (("institution", "hospital"), ("department", "department"), ("appointment_date", "day"), ("time_window", "window"))
    return tuple(name for name, attribute in attributes if len({getattr(item, attribute) for item in bookings}) > 1)


async def prepare(request: AppointmentInput, context: ToolContext, tools: ToolExecutor, budget: ToolBudget) -> Preparation:
    store = context.appointments
    resolution = resolve_request(request, store.catalog, store.today())
    context.resolution = resolution
    await tools.call("check_appointment_fields", {}, context, budget)
    if resolution.invalid:
        questions = _invalid_questions(request, resolution, store) + ([EXAMPLE] if request.origin == "followup" else [])
        return Preparation("invalid", resolution, questions=tuple(questions), options=resolution.options, awaiting=resolution.invalid)
    missing = tuple(dict.fromkeys(resolution.missing + tuple(request.information_gaps())))
    if missing:
        return Preparation("missing", resolution, questions=tuple(_missing_questions(missing, resolution, store)),
                           options=resolution.options, awaiting=missing)
    if request.action == "create":
        slots = (await tools.call("search_slots", {}, context, budget)).slots
        if not slots:
            options = tuple(slot_text(slot) for slot in _next_slots(store, resolution))
            when = day_text(resolution.day.isoformat()) + (WINDOW_NAMES[resolution.window] if resolution.window else "")
            question = f"{when}已经没有号了。" + (f"之后有号的时段：{'、'.join(options)}。" if options else "") + "请直接说想换的日期或时段。"
            return Preparation("no_slot", resolution, questions=(question,), options=options, awaiting=("appointment_date", "time_window"))
        if len(slots) > 1:
            options = tuple(slot_text(slot) for slot in slots)
            return Preparation("ambiguous", resolution, tuple(slots), (f"{'、'.join(options)}都有号，想约哪个时段？",), options, ("time_window",))
        return Preparation("ready", resolution, tuple(slots))
    bookings = (await tools.call("find_bookings", {}, context, budget)).bookings
    if not bookings:
        return Preparation("no_booking", resolution, questions=("没有找到符合条件的预约，请说出要取消的预约的医院、科室、日期或时段。",),
                           awaiting=("institution", "department", "appointment_date", "time_window"))
    if len(bookings) > 1:
        options = tuple(booking_text(booking) for booking in bookings)
        return Preparation("ambiguous", resolution, tuple(bookings), (f"找到多个预约：{'；'.join(options)}。想取消哪一个？请说出医院、日期或时段。",),
                           options, _distinguishing(bookings))
    return Preparation("ready", resolution, tuple(bookings))


def followup_for(preparation: Preparation, request: AppointmentInput, *, user: str, session: str, subject: str, now: float,
                 ttl: float) -> Followup | None:
    if preparation.kind == "ready" or preparation.kind == "invalid" and request.origin == "followup":
        return None
    resolution = preparation.resolution
    fields = {"institution": request.institution if resolution.hospital_id and "institution" not in resolution.invalid else None,
              "department": request.department if resolution.department_id else None,
              "appointment_date": resolution.day.isoformat() if resolution.day else None,
              "time_window": WINDOW_NAMES[resolution.window] if resolution.window else None}
    kind = {"invalid": "missing", "no_booking": "no_slot"}.get(preparation.kind, preparation.kind)
    return Followup(user, session, subject, request.action, kind, {name: value for name, value in fields.items() if value},
                    preparation.awaiting, preparation.options, now + ttl)
