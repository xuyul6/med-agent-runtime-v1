"""Decide this turn's appointment request: a verified request of this turn, or a valid answer to the previous turn's follow-up."""
from datetime import date

from ..core.contracts import Intent, Understanding
from .catalog import HospitalCatalog
from .params import WINDOW_NAMES, abandons, appointment_action, asks_question, parse_day, parse_window
from .store import Followup

FIELDS = ("institution", "department", "appointment_date", "time_window")


def _normal_day(value: str | None, today: date) -> str | None:
    parsed = parse_day(value, today)
    return parsed.value.isoformat() if parsed.state == "valid" else value


def _fresh(action: str, understanding: Understanding, today: date) -> dict:
    fields = {name: entity.value for name in FIELDS if (entity := understanding.entities.get(name)) and entity.value
              and entity.state in {"provided", "topic"} and entity.source.turn_id == understanding.turn_id}
    if "appointment_date" in fields:
        fields["appointment_date"] = _normal_day(fields["appointment_date"], today)
    return {"intent": Intent.APPOINTMENT, "action": action, "origin": "quote", **fields}


def _reply_fields(text: str, followup: Followup, catalog: HospitalCatalog, today: date) -> tuple[dict, set[str]]:
    """Values this reply gives, read by code from the user's own words, and which of them cannot be used."""
    given, invalid = {}, set()
    hospitals = catalog.match_hospital(text)
    if len(hospitals) == 1:
        given["institution"] = hospitals[0].short_name
    hospital = hospitals[0].id if len(hospitals) == 1 else next(
        (item.id for item in catalog.match_hospital(followup.fields.get("institution", "")) if followup.fields.get("institution")), None)
    departments = catalog.match_department(text, hospital)
    if departments:
        given["department"] = departments[0][1].name
    day = parse_day(text, today)
    if day.state == "valid" and catalog.bookable(day.value, today):
        given["appointment_date"] = day.value.isoformat()
    elif day.state != "absent":
        given["appointment_date"] = day.value.isoformat() if day.state == "valid" else day.value
        invalid.add("appointment_date")
    window = parse_window(text)
    if window.state != "absent":
        given["time_window"] = WINDOW_NAMES[window.value] if window.state == "valid" else window.value
        if window.state == "invalid":
            invalid.add("time_window")
    return given, invalid


def appointment_request(*, text: str, understanding: Understanding, followup: Followup | None, subject: str, now: float,
                        catalog: HospitalCatalog, today: date) -> tuple[dict | None, str]:
    action = appointment_action(understanding.appointment_quote)
    if action is not None:
        return _fresh(action, understanding, today), "quote"
    if followup is None:
        return None, "none"
    if followup.expires_at <= now:
        return None, "expired"
    if followup.subject != subject:
        return None, "subject_changed"
    if abandons(text):
        return None, "abandoned"
    if understanding.primary not in {Intent.GENERAL, Intent.VISIT} or asks_question(text):
        return None, "new_task"
    given, invalid = _reply_fields(text, followup, catalog, today)
    details = {"intent": Intent.APPOINTMENT, "action": followup.action, "origin": "followup", **followup.fields, **given}
    if invalid:
        return details, "followup_invalid"
    if any(name in given for name in followup.awaiting):
        return details, "followup"
    return None, "no_progress"
