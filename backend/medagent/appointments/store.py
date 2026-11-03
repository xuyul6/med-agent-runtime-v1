"""Appointment storage for the mock hospital. Tools only create drafts; confirm() with a one-time token is the only write."""
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time
from pathlib import Path
from typing import Callable, Literal
from zoneinfo import ZoneInfo

from .catalog import HospitalCatalog, Slot
from .params import WINDOW_NAMES

SUBJECT_LABELS = {"self": "本人", "mother": "母亲", "father": "父亲", "parent": "其他长辈", "child": "孩子", "other": "其他人"}
WEEKDAY_NAMES = "一二三四五六日"
# A booking stops being listed once its half-day is over.
WINDOW_END = {"am": clock_time(12, 0), "pm": clock_time(18, 0)}
SCHEMA = """
create table if not exists bookings(appointment_id text primary key, user text, subject text, slot_id text, status text, created_at real);
create table if not exists drafts(draft_id text primary key, kind text, user text, session text, subject text, slot_id text,
    appointment_id text, summary text, token_sha256 text, expires_at real, status text, created_at real);
create table if not exists followups(user text, session text, subject text, action text, kind text, fields_json text,
    awaiting_json text, options_json text, expires_at real, primary key (user, session));
"""


class AppointmentError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Draft:
    draft_id: str
    kind: Literal["create", "cancel"]
    user: str
    session: str
    subject: str
    slot_id: str
    appointment_id: str | None
    summary: str
    expires_at: float


@dataclass(frozen=True)
class Booking:
    appointment_id: str
    user: str
    subject: str
    slot: Slot
    status: Literal["booked", "cancelled"]


@dataclass(frozen=True)
class Followup:
    user: str
    session: str
    subject: str
    action: Literal["create", "cancel"]
    kind: Literal["missing", "no_slot", "ambiguous"]
    fields: dict[str, str]
    awaiting: tuple[str, ...]
    options: tuple[str, ...]
    expires_at: float


def subject_label(subject: str) -> str:
    return SUBJECT_LABELS.get(subject, "其他人")


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class AppointmentStore:
    def __init__(self, catalog: HospitalCatalog, path: str = ":memory:", clock: Callable[[], float] = time.time, ttl: float = 600,
                 followup_ttl: float = 600, timezone: str = "Asia/Shanghai"):
        self.catalog, self.clock, self.ttl, self.followup_ttl, self.zone = catalog, clock, ttl, followup_ttl, ZoneInfo(timezone)
        self._lock = threading.Lock()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._connection.executescript(SCHEMA)

    def now(self) -> float:
        return self.clock()

    def today(self) -> date:
        return datetime.fromtimestamp(self.clock(), self.zone).date()

    def _taken(self, slot_id: str) -> int:
        return self._connection.execute("select count(*) from bookings where slot_id = ? and status = 'booked'", (slot_id,)).fetchone()[0]

    def _slot(self, slot_id: str) -> Slot:
        slot = self.catalog.slot(slot_id)
        if slot is None:
            raise AppointmentError("slot_not_found")
        return slot

    def _check_window(self, slot: Slot) -> None:
        if not self.catalog.bookable(slot.day, self.today()):
            raise AppointmentError("slot_out_of_window")

    def summary(self, slot: Slot, subject: str) -> str:
        return (f"{slot.hospital} · {slot.department} · {slot.day.month}月{slot.day.day}日（周{WEEKDAY_NAMES[slot.day.weekday()]}）"
                f"{WINDOW_NAMES[slot.window]} · 就诊人：{subject_label(subject)}")

    def search(self, hospital_id: str, department_id: str, day: date, window: Literal["am", "pm"] | None) -> list[tuple[Slot, int]]:
        with self._lock:
            found = [(slot, slot.capacity - self._taken(slot.slot_id))
                     for slot in self.catalog.slots(hospital_id, department_id, day, window, self.today())]
        return [(slot, remaining) for slot, remaining in found if remaining > 0]

    def _booking(self, row) -> Booking:
        return Booking(row[0], row[1], row[2], self._slot(row[3]), row[4])

    def bookings(self, user: str, subject: str | None = None) -> list[Booking]:
        with self._lock:
            rows = self._connection.execute("select appointment_id, user, subject, slot_id, status from bookings where user = ? "
                                            "and status = 'booked' and (? is null or subject = ?) order by created_at",
                                            (user, subject, subject)).fetchall()
        return [self._booking(row) for row in rows]

    def upcoming(self, user: str) -> list[Booking]:
        now = datetime.fromtimestamp(self.clock(), self.zone)
        found = [booking for booking in self.bookings(user)
                 if datetime.combine(booking.slot.day, WINDOW_END[booking.slot.window], self.zone) > now]
        return sorted(found, key=lambda booking: (booking.slot.day, booking.slot.window))

    def _draft(self, kind, user, session, subject, slot: Slot, appointment_id) -> tuple[Draft, str]:
        token, now = secrets.token_urlsafe(24), self.clock()
        draft = Draft("draft-" + secrets.token_hex(6), kind, user, session, subject, slot.slot_id, appointment_id,
                      ("取消 " if kind == "cancel" else "") + self.summary(slot, subject), now + self.ttl)
        self._connection.execute("insert into drafts values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                                 (draft.draft_id, kind, user, session, subject, slot.slot_id, appointment_id, draft.summary,
                                  digest(token), draft.expires_at, now))
        return draft, token

    def propose_create(self, *, user: str, session: str, subject: str, slot_id: str) -> tuple[Draft, str]:
        with self._lock:
            slot = self._slot(slot_id)
            self._check_window(slot)
            if self._taken(slot_id) >= slot.capacity:
                raise AppointmentError("slot_full")
            if self._connection.execute("select 1 from bookings where user = ? and subject = ? and slot_id = ? and status = 'booked'",
                                        (user, subject, slot_id)).fetchone():
                raise AppointmentError("already_booked")
            self._connection.execute("update drafts set status = 'superseded' where kind = 'create' and user = ? and subject = ? "
                                     "and slot_id = ? and status = 'pending'", (user, subject, slot_id))
            return self._draft("create", user, session, subject, slot, None)

    def propose_cancel(self, *, user: str, session: str, subject: str, appointment_id: str) -> tuple[Draft, str]:
        with self._lock:
            row = self._connection.execute("select slot_id from bookings where appointment_id = ? and user = ? and subject = ? "
                                           "and status = 'booked'", (appointment_id, user, subject)).fetchone()
            if row is None:
                raise AppointmentError("appointment_not_found")
            slot = self._slot(row[0])
            self._check_window(slot)
            return self._draft("cancel", user, session, subject, slot, appointment_id)

    def confirm(self, *, user: str, draft_id: str, token: str) -> Booking:
        with self._lock:
            self._connection.execute("begin immediate")
            try:
                booking = self._confirm(user, draft_id, token)
            except BaseException:
                self._connection.execute("rollback")
                raise
            self._connection.execute("commit")
            return booking

    def _confirm(self, user: str, draft_id: str, token: str) -> Booking:
        row = self._connection.execute("select kind, user, subject, slot_id, appointment_id, token_sha256, expires_at, status "
                                       "from drafts where draft_id = ?", (draft_id,)).fetchone()
        if row is None:
            raise AppointmentError("draft_not_found")
        kind, owner, subject, slot_id, appointment_id, stored, expires_at, status = row
        if owner != user:
            raise AppointmentError("user_mismatch")
        if status != "pending":
            raise AppointmentError("draft_not_pending")
        if self.clock() >= expires_at:
            raise AppointmentError("draft_expired")
        if not hmac.compare_digest(stored, digest(token or "")):
            raise AppointmentError("token_invalid")
        slot = self._slot(slot_id)
        self._check_window(slot)
        if kind == "create":
            if self._taken(slot_id) >= slot.capacity:
                raise AppointmentError("slot_full")
            appointment_id = "apt-" + secrets.token_hex(6)
            self._connection.execute("insert into bookings values (?, ?, ?, ?, 'booked', ?)", (appointment_id, user, subject, slot_id, self.clock()))
        else:
            changed = self._connection.execute("update bookings set status = 'cancelled' where appointment_id = ? and status = 'booked'",
                                               (appointment_id,)).rowcount
            if not changed:
                raise AppointmentError("appointment_not_found")
        self._connection.execute("update drafts set status = 'confirmed' where draft_id = ?", (draft_id,))
        return Booking(appointment_id, user, subject, slot, "booked" if kind == "create" else "cancelled")

    def save_followup(self, followup: Followup) -> None:
        with self._lock:
            self._connection.execute("insert or replace into followups values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                     (followup.user, followup.session, followup.subject, followup.action, followup.kind,
                                      json.dumps(followup.fields, ensure_ascii=False), json.dumps(list(followup.awaiting)),
                                      json.dumps(list(followup.options), ensure_ascii=False), followup.expires_at))

    def take_followup(self, user: str, session: str) -> Followup | None:
        with self._lock:
            row = self._connection.execute("select * from followups where user = ? and session = ?", (user, session)).fetchone()
            self._connection.execute("delete from followups where user = ? and session = ?", (user, session))
        if row is None:
            return None
        return Followup(row[0], row[1], row[2], row[3], row[4], json.loads(row[5]), tuple(json.loads(row[6])), tuple(json.loads(row[7])), row[8])
