"""Booking tools of the open comparison arm: the single agent holds them on every turn, without request admission.

No product role holds them and the product registry does not list them. The open arm adds them to the registry
of the runtime it answers for; writes still happen only when the user confirms a draft.
"""
from pydantic import Field
from ..appointments.params import parse_day, parse_window
from ..appointments.resolution import booking_view
from ..appointments.store import AppointmentError
from ..core.contracts import Intent
from ..tools.contracts import (BookingSearchResult, Contract, DraftResult, EmptyArguments, ProposeAppointmentArguments,
                               ProposeCancellationArguments, SlotSearchResult)
from ..tools.registry import ToolContext, ToolInputError, ToolSpec, slot_view

OPEN_DRAFT_LIMIT = 3
# The audit records attempts on these like the product's own proposal tools: (action, argument that names the target).
OPEN_PROPOSALS = {"open_propose_appointment":("create", "slot_id"), "open_propose_cancellation":("cancel", "appointment_id")}


class OpenSlotQuery(Contract):
    department: str = Field(min_length=1, max_length=40)
    appointment_date: str = Field(min_length=1, max_length=40)
    time_window: str | None = Field(default=None, max_length=20)
    institution: str | None = Field(default=None, max_length=40)


def open_search_slots(context: ToolContext, arguments: OpenSlotQuery) -> dict:
    """The model chooses what to search; every slot found this turn becomes proposable."""
    store = context.appointments
    catalog, today = store.catalog, store.today()
    hospitals = catalog.match_hospital(arguments.institution) if arguments.institution else []
    day, window = parse_day(arguments.appointment_date, today), parse_window(arguments.time_window)
    if day.state != "valid" or window.state == "invalid" or not catalog.bookable(day.value, today):
        return {"slots":[]}
    found = [item for hospital, dept in catalog.match_department(arguments.department, hospitals[0].id if len(hospitals) == 1 else None)
             for item in store.search(hospital.id, dept.id, day.value, window.value)]
    context.slots.update({slot.slot_id:slot for slot, _ in found})
    return {"slots":[slot_view(slot, remaining) for slot, remaining in found]}


def open_find_bookings(context: ToolContext, arguments: EmptyArguments) -> dict:
    store = context.appointments
    found = [booking for booking in store.bookings(context.user, context.request.subject) if store.catalog.bookable(booking.slot.day, store.today())]
    context.bookings.update({booking.appointment_id:booking for booking in found})
    return {"bookings":[booking_view(booking).model_dump() for booking in found]}


def _open_propose(context: ToolContext, candidate: str, candidates: dict, propose, already: set) -> dict:
    if candidate not in candidates:
        raise ToolInputError("slot_not_from_this_search" if candidates is context.slots else "booking_not_from_this_search")
    if len(context.issued) >= OPEN_DRAFT_LIMIT:
        raise ToolInputError("draft_limit_per_turn")
    if candidate in already:
        raise ToolInputError("slot_already_drafted")
    try:
        draft, token = propose(user=context.user, session=context.session, subject=context.request.subject)
    except AppointmentError as error:
        raise ToolInputError(error.code) from error
    context.issued[draft.draft_id] = (draft, token)
    return {"draft_id":draft.draft_id, "kind":draft.kind, "summary":draft.summary, "needs_user_confirmation":True}


def open_propose_appointment(context: ToolContext, arguments: ProposeAppointmentArguments) -> dict:
    return _open_propose(context, arguments.slot_id, context.slots,
                         lambda **who: context.appointments.propose_create(**who, slot_id=arguments.slot_id),
                         {draft.slot_id for draft, _ in context.issued.values() if draft.kind == "create"})


def open_propose_cancellation(context: ToolContext, arguments: ProposeCancellationArguments) -> dict:
    return _open_propose(context, arguments.appointment_id, context.bookings,
                         lambda **who: context.appointments.propose_cancel(**who, appointment_id=arguments.appointment_id),
                         {draft.appointment_id for draft, _ in context.issued.values() if draft.kind == "cancel"})


OPEN_ROLE = "single_agent_open"
OPEN_SPECS = [
    ToolSpec("open_search_slots", "按科室、日期和时段查询模拟医院号源，可选医院。", OpenSlotQuery, SlotSearchResult,
             frozenset({OPEN_ROLE}), frozenset({Intent.APPOINTMENT}), open_search_slots, timeout=1, stage="retrieval"),
    ToolSpec("open_find_bookings", "查询当前用户、当前咨询对象在模拟医院的全部可取消预约。", EmptyArguments, BookingSearchResult,
             frozenset({OPEN_ROLE}), frozenset({Intent.APPOINTMENT}), open_find_bookings, timeout=1, stage="retrieval"),
    ToolSpec("open_propose_appointment", "为本轮查到的号源生成待确认的预约草稿，每轮最多 3 份；不会写入，需用户点击确认。", ProposeAppointmentArguments, DraftResult,
             frozenset({OPEN_ROLE}), frozenset({Intent.APPOINTMENT}), open_propose_appointment, timeout=1, side_effect="propose_write", stage="action"),
    ToolSpec("open_propose_cancellation", "为本轮查到的已有预约生成待确认的取消草稿，每轮最多 3 份；不会写入，需用户点击确认。", ProposeCancellationArguments, DraftResult,
             frozenset({OPEN_ROLE}), frozenset({Intent.APPOINTMENT}), open_propose_cancellation, timeout=1, side_effect="propose_write", stage="action"),
]
OPEN_TOOLS = tuple(spec.name for spec in OPEN_SPECS)
