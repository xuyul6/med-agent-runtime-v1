"""Typed tool data; models cannot supply new patient facts through tool arguments."""
from typing import Literal
from pydantic import Field
from ..agents.contracts import Contract
from ..core.contracts import Intent, MedicationEvidenceScope, Reference, SourceProvenance


class EmptyArguments(Contract):
    pass


class LookupArguments(Contract):
    query: str = Field(min_length=1, max_length=6000)
    domain: Intent


class LookupResult(Contract):
    documents: list[Reference] = Field(max_length=10)


class ContextResult(Contract):
    intent: Intent
    subject: str
    risk_level: Literal["ROUTINE", "CAUTION", "CRITICAL"]
    provided_fields: list[str]
    missing_fields: list[str]


class FieldCheck(Contract):
    provided_fields: list[str]
    missing_fields: list[str]
    note: str


class ReferenceSelection(Contract):
    keys: list[str] = Field(default_factory=list, max_length=10)


class ReferenceScope(Contract):
    key: str
    version: str
    scope: str
    entities: list[str]
    provenance: SourceProvenance | None = None
    medication_scope: MedicationEvidenceScope | None = None


class MedicationScopeResult(Contract):
    medication_name: str | None
    sources: list[ReferenceScope]
    applicability: Literal["needs_verification", "no_source"]
    note: str


class ChecklistResult(Contract):
    items: list[str]
    reference_keys: list[str]
    missing_fields: list[str]
    needs_verification: list[str]


class SlotView(Contract):
    slot_id: str
    hospital: str
    department: str
    day: str
    window: Literal["am", "pm"]
    remaining: int


class SlotSearchResult(Contract):
    slots: list[SlotView]


class BookingView(Contract):
    appointment_id: str
    hospital: str
    department: str
    day: str
    window: Literal["am", "pm"]
    subject_label: str


class BookingSearchResult(Contract):
    bookings: list[BookingView]


class ProposeAppointmentArguments(Contract):
    slot_id: str = Field(min_length=1, max_length=120)


class ProposeCancellationArguments(Contract):
    appointment_id: str = Field(min_length=1, max_length=64)


class DraftResult(Contract):
    """What the model sees after proposing: no confirmation token."""
    draft_id: str
    kind: Literal["create", "cancel"]
    summary: str
    needs_user_confirmation: Literal[True] = True


class HandoffSummary(Contract):
    known_facts: list[str]
    handoff_connected: Literal[False] = False
