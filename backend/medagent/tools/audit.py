"""Credential-free tool attempt audit and scoped synthetic exposure observation."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import re
import uuid
from ..core.observability import event

# Tools that propose a booking change: (action, argument that names the target).
PROPOSALS = {'propose_appointment': ('create', 'slot_id'), 'propose_cancellation': ('cancel', 'appointment_id')}
PATTERNS = {'create': r'[a-z0-9-]+:[a-z0-9-]+:\d{4}-\d{2}-\d{2}:(?:am|pm)', 'cancel': r'apt-[0-9a-f]{12}'}
MATERIALS = ContextVar('synthetic_exposure_materials', default=())
SAFE_ERRORS = frozenset({'action_mismatch', 'slot_not_from_this_search', 'booking_not_from_this_search',
    'candidate_not_unique', 'one_draft_per_turn', 'draft_limit_per_turn', 'slot_already_drafted', 'invalid_tool_arguments', 'slot_not_found', 'slot_full',
    'already_booked', 'appointment_not_found', 'slot_out_of_window'})


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def outcome(error: BaseException | None) -> str:
    if error is None: return 'drafted'
    if isinstance(error, asyncio.CancelledError): return 'cancelled'
    if isinstance(error, TimeoutError): return 'timeout'
    if isinstance(error, PermissionError): return 'permission_denied'
    # Import at runtime to keep the registry free to use this module.
    from .registry import ToolBudgetExceeded, ToolInputError
    if isinstance(error, ToolBudgetExceeded): return 'budget_or_batch_limit'
    if isinstance(error, ToolInputError):
        return str(error) if str(error) in SAFE_ERRORS else 'invalid_arguments'
    return 'failed'


class AppointmentAttempts:
    def __init__(self, calls: list[dict], role: str, instance: str, model_round: int, request_action: str | None,
                 proposals: dict = PROPOSALS):
        self.pending = {}
        for index, call in enumerate(calls):
            if call.get('name') not in proposals: continue
            action, key = proposals[call['name']]
            arguments = call.get('input')
            target = arguments.get(key) if isinstance(arguments, dict) else None
            valid = isinstance(target, str) and len(target) <= 120 and re.fullmatch(PATTERNS[action], target) is not None
            attempt_id = uuid.uuid4().hex
            self.pending[index] = attempt_id
            event('appointment_tool_attempt', attempt_id=attempt_id, role=role, instance=instance, model_round=model_round,
                  tool=call['name'], action=action, target=target if valid else None, target_valid=valid,
                  request_action=request_action if request_action in {'create', 'cancel'} else None)

    def result(self, index: int, error: BaseException | None = None) -> None:
        identity = self.pending.pop(index, None)
        if identity:
            event('appointment_tool_result', attempt_id=identity, outcome=outcome(error))


@contextmanager
def appointment_attempts(calls: list[dict], role: str, instance: str, model_round: int, request_action: str | None,
                         proposals: dict = PROPOSALS):
    """proposals: the proposal tools the caller holds; a caller with tools of its own passes them too."""
    attempts = AppointmentAttempts(calls, role, instance, model_round, request_action, proposals)
    error = None
    try:
        yield attempts
    except BaseException as exc:
        error = exc
        raise
    finally:
        for index in list(attempts.pending):
            attempts.result(index, error or RuntimeError('unprocessed'))


@contextmanager
def exposure_scope(materials: list[dict]):
    token = MATERIALS.set(tuple(dict(item) for item in materials))
    try:
        yield
    finally:
        MATERIALS.reset(token)


def texts(value: object):
    """Every string in what is sent; role data arrives as JSON already serialized inside a message."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values(): yield from texts(item)
    elif isinstance(value, (list, tuple)):
        for item in value: yield from texts(item)


def observe_model_input(role: str, instance: str, model_round: int, content: object) -> None:
    if not MATERIALS.get(): return
    # Inspect only the message content actually sent, never add test instructions to it.
    sent = list(texts(content))
    for material in MATERIALS.get():
        attack = material['attack']
        # The attack appears as written, or escaped once when it travels inside serialized role data.
        forms = {attack, json.dumps(attack, ensure_ascii=False)[1:-1]}
        if any(form in text for text in sent for form in forms):
            event('injection_exposure', reference_key=material['key'], carrier=material['carrier'],
                  material_hash=fingerprint(material['material']), fragment_hash=fingerprint(attack),
                  role=role, instance=instance, model_round=model_round)
