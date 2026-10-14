"""Turns a status-write request plus the stored row into a refusal or a write plan.

Contract: plans/deferral-flip-condition-prd.md §5.3 and decisions 3 to 5.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, get_args

from shared import process_identity
from shared.process_identity import ProcessIdentity, ProcessIdentityUnreadable
from shared.task_deferral import (
    CALLER_KINDS,
    CarriedByRequest,
    Deferral,
    DeferralInvalid,
    DeferralInvalidReason,
    DeferralRefusalCode,
    DeferralRequest,
    HeldBySessionRequest,
    InvalidRecord,
    LegacyUnknownRequest,
    parse_record,
    parse_request,
    parse_utc_timestamp,
)
from shared.task_metadata_wire import coerce_task_metadata
from shared.task_statuses import TaskStatus

from fused_memory.backends.task_backend_errors import TaskNotFoundError

__all__ = [
    'ACCEPTED_KINDS',
    'ALTERNATIVES',
    'HOLDER_COMMAND',
    'NO_WRITE',
    'DeferralCall',
    'DeferralChangedRefusal',
    'DeferralInvalidRefusal',
    'DeferralRequiredRefusal',
    'DeferralWrite',
    'GateRefusal',
    'StatusWriteScope',
    'corroborate',
    'plan',
    'read_holder',
]

DEFERRED: Final = TaskStatus.DEFERRED.value
HOLDER_COMMAND: Final = 'claude'
ACCEPTED_KINDS: Final = tuple(sorted(kind.value for kind in CALLER_KINDS))
ALTERNATIVES: Final = (
    {
        'shape': "status='pending' plus add_dependency",
        'use': 'the hold is only another task finishing first',
    },
    {
        'shape': "status='pending' plus metadata.milestone {mode: 'dated', at: <ISO-8601>}",
        'use': 'the hold is only a time',
    },
    {
        'shape': (
            "an escalate-on-dispatch pure gate (metadata.execution_class='operational': "
            'a born-at-L2 escalation with a supervised consumer)'
        ),
        'use': 'the hold is a human judgment needed at dispatch',
    },
)


class StatusWriteScope(StrEnum):
    """Who is asking: a status write, or commit_planning releasing a planning batch."""

    STATUS = 'status'
    COMMIT_PLANNING = 'commit_planning'


def _caller_kind_shapes() -> str:
    request_models = get_args(get_args(DeferralRequest)[0])
    shapes = []
    for model in request_models:
        kind = model.model_fields['kind'].default
        if kind not in CALLER_KINDS:
            continue
        fields = [
            name if info.is_required() else f'{name}?'
            for name, info in model.model_fields.items()
            if name != 'kind'
        ]
        shapes.append(f'{kind.value} {{{", ".join(fields)}}}')
    return '; '.join(sorted(shapes))


_KINDS_AND_ALTERNATIVES: Final = (
    f'Accepted deferral kinds: {_caller_kind_shapes()} (? = optional; holder_pid is your '
    "session's $CLAUDE_PID). Not a deferral: "
    + '; '.join(f'{alt["use"]}: {alt["shape"]}' for alt in ALTERNATIVES)
    + '.'
)
_ECHO_CHECK: Final = (
    'A successful write echoes `deferral` in its response. No echo means the server '
    'predates the gate: treat the hold as unrecorded and re-stamp it after the server '
    "restarts. Recipe: docs/task-authoring.md §9 'Deferring a task'."
)
REQUIRED_HINT: Final = (
    "Entering 'deferred' needs a `deferral` saying what ends the hold, e.g. "
    "set_task_status(id=<id>, status='deferred', deferral={'kind': 'until_condition', "
    "'condition': '<what must become true>'}). "
    f'{_KINDS_AND_ALTERNATIVES} {_ECHO_CHECK}'
)
INVALID_HINT: Final = (
    'The `deferral` request was refused: reason_code names the rule, field the offending '
    f'field and value what was sent. {_KINDS_AND_ALTERNATIVES} {_ECHO_CHECK}'
)


def _json_safe(value: object) -> object:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


@dataclass(frozen=True)
class DeferralRequiredRefusal:
    task_id: str
    from_status: str

    def to_error_dict(self) -> dict[str, Any]:
        return {
            'success': False,
            'error': DeferralRefusalCode.REQUIRED.value,
            'task_id': self.task_id,
            'from_status': self.from_status,
            'accepted_kinds': list(ACCEPTED_KINDS),
            'alternatives': [dict(alt) for alt in ALTERNATIVES],
            'hint': REQUIRED_HINT,
        }

    def journal_text(self) -> str:
        return f'{DeferralRefusalCode.REQUIRED.value} from_status={self.from_status}'


@dataclass(frozen=True)
class DeferralInvalidRefusal:
    task_id: str
    reason: DeferralInvalidReason
    field: str
    value: object

    @classmethod
    def from_exception(cls, task_id: str, exc: DeferralInvalid) -> DeferralInvalidRefusal:
        return cls(task_id, exc.reason, exc.field, exc.value)

    def to_error_dict(self) -> dict[str, Any]:
        return {
            'success': False,
            'error': DeferralRefusalCode.INVALID.value,
            'task_id': self.task_id,
            'reason_code': self.reason.value,
            'field': self.field,
            'value': _json_safe(self.value),
            'hint': INVALID_HINT,
        }

    def journal_text(self) -> str:
        return (
            f'{DeferralRefusalCode.INVALID.value} reason_code={self.reason.value} '
            f'field={self.field}'
        )


@dataclass(frozen=True)
class DeferralChangedRefusal:
    task_id: str
    expected_stamped_at: str | None
    current_stamped_at: str | None
    current_status: str

    def to_error_dict(self) -> dict[str, Any]:
        return {
            'success': False,
            'error': DeferralRefusalCode.CHANGED.value,
            'task_id': self.task_id,
            'expected_stamped_at': self.expected_stamped_at,
            'current_stamped_at': self.current_stamped_at,
            'current_status': self.current_status,
        }

    def journal_text(self) -> str:
        return (
            f'{DeferralRefusalCode.CHANGED.value} expected={self.expected_stamped_at} '
            f'current={self.current_stamped_at} status={self.current_status}'
        )


GateRefusal = DeferralRequiredRefusal | DeferralInvalidRefusal | DeferralChangedRefusal


@dataclass(frozen=True)
class DeferralCall:
    """One status-write call's deferral arguments, validated and corroborated."""

    request: DeferralRequest | None
    holder: ProcessIdentity | None
    expected_stamped_at: datetime | None
    expected_stamped_at_text: str | None
    restamp_only: bool
    scope: StatusWriteScope
    now: datetime


@dataclass(frozen=True)
class DeferralWrite:
    """What a status write must also do to metadata.deferral, and what it reports."""

    audit_fields: Mapping[str, Any] = field(default_factory=dict)
    clear_keys: frozenset[str] = frozenset()
    event_fields: Mapping[str, Any] = field(default_factory=dict)
    restamp_old: object = None
    restamp_new: Mapping[str, Any] | None = None

    @property
    def is_restamp(self) -> bool:
        return self.restamp_new is not None

    @property
    def echo(self) -> dict[str, Any]:
        if self.restamp_new is not None:
            return {'deferral': self.restamp_new}
        return dict(self.event_fields)


NO_WRITE: Final = DeferralWrite()


def read_holder(pid: int, proc_root: Path) -> ProcessIdentity:
    """The live Claude CLI process at *pid*, or DeferralInvalid saying why it is not one."""
    try:
        identity = process_identity.capture(pid, proc_root=proc_root)
        name = process_identity.command_name(pid, proc_root=proc_root)
    except ProcessIdentityUnreadable as exc:
        raise DeferralInvalid(DeferralInvalidReason.HOLDER_UNVERIFIABLE, 'holder_pid', pid) from exc
    if identity is None or name is None:
        raise DeferralInvalid(DeferralInvalidReason.HOLDER_NOT_LIVE, 'holder_pid', pid)
    if name != HOLDER_COMMAND:
        raise DeferralInvalid(DeferralInvalidReason.HOLDER_NOT_SESSION, 'holder_pid', pid)
    return identity


def _request_shape_error(
    *,
    task_id: str,
    status: str,
    raw_deferral: object,
    expected_stamped_at: str | None,
    restamp_only: bool,
) -> DeferralInvalid | None:
    if ',' in task_id and (restamp_only or expected_stamped_at is not None):
        return DeferralInvalid(DeferralInvalidReason.CAS_NEEDS_SINGLE_ID, 'id', task_id)
    if restamp_only and status != DEFERRED:
        return DeferralInvalid(DeferralInvalidReason.FIELD_NOT_ALLOWED, 'restamp_only', status)
    if raw_deferral is not None and status != DEFERRED:
        return DeferralInvalid(DeferralInvalidReason.STATUS_NOT_DEFERRED, 'status', status)
    return None


def _parse_expected(text: str | None) -> datetime | None:
    if text is None:
        return None
    try:
        return parse_utc_timestamp(text)
    except ValueError as exc:
        raise DeferralInvalid(
            DeferralInvalidReason.INVALID_VALUE, 'expected_stamped_at', text,
        ) from exc


async def _corroborate_request(
    request: DeferralRequest | None,
    *,
    read_task: Callable[[str], Awaitable[object]],
    proc_root: Path,
) -> ProcessIdentity | None:
    if isinstance(request, HeldBySessionRequest):
        return await asyncio.to_thread(read_holder, request.holder_pid, proc_root)
    if isinstance(request, CarriedByRequest):
        try:
            await read_task(str(request.carrier_task_id))
        except TaskNotFoundError as exc:
            raise DeferralInvalid(
                DeferralInvalidReason.CARRIER_NOT_FOUND, 'carrier_task_id', request.carrier_task_id,
            ) from exc
    return None


async def corroborate(
    *,
    task_id: str,
    status: str,
    raw_deferral: object,
    expected_stamped_at: str | None,
    restamp_only: bool,
    scope: StatusWriteScope,
    read_task: Callable[[str], Awaitable[object]],
    proc_root: Path,
    now: datetime,
) -> DeferralCall | DeferralInvalidRefusal:
    """The row-independent checks, run once per call before the write lock.

    *task_id* is the caller's raw id argument (a CSV for a batch); *read_task* reads a
    carrier row and raises TaskNotFoundError when it does not exist.
    """
    try:
        shape_error = _request_shape_error(
            task_id=task_id,
            status=status,
            raw_deferral=raw_deferral,
            expected_stamped_at=expected_stamped_at,
            restamp_only=restamp_only,
        )
        if shape_error is not None:
            raise shape_error
        expected = _parse_expected(expected_stamped_at)
        request = None if raw_deferral is None else parse_request(raw_deferral, now=now)
        holder = await _corroborate_request(request, read_task=read_task, proc_root=proc_root)
    except DeferralInvalid as exc:
        return DeferralInvalidRefusal.from_exception(task_id, exc)
    return DeferralCall(
        request=request,
        holder=holder,
        expected_stamped_at=expected,
        expected_stamped_at_text=expected_stamped_at,
        restamp_only=restamp_only,
        scope=scope,
        now=now,
    )


@dataclass(frozen=True)
class _StoredRecord:
    present: bool
    raw: object
    parsed: Deferral | InvalidRecord | None

    @property
    def valid(self) -> Deferral | None:
        return None if isinstance(self.parsed, InvalidRecord) else self.parsed


def _stored_record(stored_metadata: object) -> _StoredRecord:
    metadata = coerce_task_metadata(stored_metadata)
    if metadata is None or 'deferral' not in metadata:
        return _StoredRecord(present=False, raw=None, parsed=None)
    raw = metadata['deferral']
    return _StoredRecord(present=True, raw=raw, parsed=parse_record(raw))


def _needs_request(old_status: str, call: DeferralCall) -> bool:
    return (
        old_status != DEFERRED
        or call.restamp_only
        or call.scope is StatusWriteScope.COMMIT_PLANNING
    )


def _kind_refusal(
    request: DeferralRequest, *, old_status: str, stored: _StoredRecord, task_id: str,
) -> DeferralInvalidRefusal | None:
    if isinstance(request, LegacyUnknownRequest) and (
        old_status != DEFERRED or stored.valid is not None
    ):
        return DeferralInvalidRefusal(
            task_id, DeferralInvalidReason.SERVER_ONLY_KIND, 'kind', request.kind.value,
        )
    if isinstance(request, CarriedByRequest) and str(request.carrier_task_id) == task_id.strip():
        return DeferralInvalidRefusal(
            task_id, DeferralInvalidReason.CARRIER_IS_SELF, 'carrier_task_id',
            request.carrier_task_id,
        )
    return None


def plan(
    *,
    old_status: str,
    stored_metadata: object,
    status: str,
    call: DeferralCall,
    task_id: str,
    agent_id: str | None,
) -> GateRefusal | DeferralWrite:
    """One id's refusal or write plan, decided under the write lock from the stored row."""
    stored = _stored_record(stored_metadata)
    request = call.request
    if status == DEFERRED and request is None and _needs_request(old_status, call):
        return DeferralRequiredRefusal(task_id, old_status)
    if request is not None:
        refusal = _kind_refusal(request, old_status=old_status, stored=stored, task_id=task_id)
        if refusal is not None:
            return refusal
    return NO_WRITE
