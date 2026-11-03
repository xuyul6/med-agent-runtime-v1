"""Resolve an appointment request against the catalog; shared by the multi-agent role and the single-agent baseline."""
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

from ..agents.contracts import AppointmentInput
from ..core.contracts import PendingAction
from ..tools.contracts import BookingView
from .catalog import HospitalCatalog
from .params import parse_day, parse_window
from .store import Booking, Draft, subject_label


@dataclass(frozen=True)
class Resolution:
    hospital_id: str | None
    department_id: str | None
    day: date | None
    window: Literal["am", "pm"] | None
    missing: tuple[str, ...]
    invalid: tuple[str, ...]
    options: tuple[str, ...]


def resolve_request(request: AppointmentInput, catalog: HospitalCatalog, today: date) -> Resolution:
    missing, invalid, options = [], [], []
    hospital_id = department_id = None
    if request.institution:
        hospitals = catalog.match_hospital(request.institution)
        if not hospitals:
            invalid.append("institution")
            options = [hospital.name for hospital in catalog.hospitals]
        elif len(hospitals) == 1:
            hospital_id = hospitals[0].id
    if request.department:
        matched = catalog.match_department(request.department, hospital_id)
        if not matched:
            invalid.append("department")
            options = catalog.department_names()
        elif len({hospital.id for hospital, _ in matched}) > 1:
            department_id = matched[0][1].id
            # Cancelling searches both hospitals; a booking needs one.
            if request.action == "create":
                missing.append("institution")
                options = [hospital.name for hospital, _ in matched]
        else:
            hospital_id, department_id = matched[0][0].id, matched[0][1].id
    elif request.action == "create":
        missing.append("department")
        options = catalog.department_names()
    day = parse_day(request.appointment_date, today)
    if day.state == "invalid" or day.state == "valid" and not catalog.bookable(day.value, today):
        invalid.append("appointment_date")
    elif day.state == "absent" and request.action == "create":
        missing.append("appointment_date")
    window = parse_window(request.time_window)
    if window.state == "invalid":
        invalid.append("time_window")
    return Resolution(hospital_id, department_id, day.value if "appointment_date" not in invalid and day.state == "valid" else None,
                      window.value if window.state == "valid" else None, tuple(missing), tuple(invalid), tuple(options))


def booking_view(booking: Booking) -> BookingView:
    slot = booking.slot
    return BookingView(appointment_id=booking.appointment_id, hospital=slot.hospital, department=slot.department, day=slot.day.isoformat(),
                       window=slot.window, subject=booking.subject, subject_label=subject_label(booking.subject))


def pending_from_issued(issued: dict[str, tuple[Draft, str]], timezone: str) -> list[PendingAction]:
    zone = ZoneInfo(timezone)
    return [PendingAction(kind="appointment_" + draft.kind, draft_id=draft.draft_id, token=token, summary=draft.summary,
                          expires_at=datetime.fromtimestamp(draft.expires_at, zone).isoformat(timespec="seconds"))
            for draft, token in issued.values()]
