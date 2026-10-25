"""A deferred row's hold, worded as the summary the Tasks tab draws, plus the holder liveness it needs.

The kinds, the records and the needs-human verdict are ``shared.task_deferral``'s;
this module only words them. The texts are
``plans/deferral-flip-condition-prd.md`` decision 10's.

A hold whose referent was not read is drawn UNKNOWN. Only an unread holder
also needs a human, because only it has a shared-model reason
(LIVENESS_UNKNOWN). An unread carrier has none, and an unmeasured status map
must not read as 'carrier missing', so it gets no verdict.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, assert_never

from shared.process_identity import (
    PROC_ROOT,
    Liveness,
    ProcessIdentity,
    ProcessIdentityUnreadable,
    liveness,
)
from shared.task_deferral import (
    CarriedByDeferral,
    Deferral,
    HeldBySessionDeferral,
    InvalidRecord,
    LegacyUnknownDeferral,
    NeedsHumanReason,
    PlanningDeferral,
    UntilConditionDeferral,
    needs_human,
    parse_record,
)
from shared.task_statuses import TaskStatus

__all__ = [
    'UNREAD',
    'HoldReads',
    'HoldState',
    'deferral_row_fields',
    'read_liveness',
    'session_holders',
]


class HoldState(StrEnum):
    """How a hold is drawn. UNKNOWN means a referent could not be read, never that it is lost."""

    HELD = 'held'
    NEEDS_HUMAN = 'needs_human'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class HoldReads:
    """What the holds' referents read as when the rows were shaped.

    Attributes:
        carrier_statuses: ``{id: status}`` over the whole tree, or ``None`` when
            the map was not measured, so that no carrier reads as missing.
        holder_liveness: Each held_by_session row's holder liveness, by row id.
            An absent id or ``None`` means the holder was not read.
    """

    carrier_statuses: Mapping[int, str] | None
    holder_liveness: Mapping[int, Liveness | None]


UNREAD: Final = HoldReads(carrier_statuses=None, holder_liveness=MappingProxyType({}))

_CARRIER_QUALIFIERS: Final[Mapping[NeedsHumanReason | None, str]] = {
    None: '',
    NeedsHumanReason.CLOSABLE: ' (closable)',
    NeedsHumanReason.ORPHANED: ' (orphaned)',
    NeedsHumanReason.CARRIER_MISSING: ' (carrier missing)',
}
_CARRIER_NOT_READ_QUALIFIER: Final = ' (carrier status unknown)'
_LIVENESS_WORDS: Final[Mapping[Liveness | None, str]] = {
    Liveness.ALIVE: 'live',
    Liveness.DEAD: 'lost, releasing',
    Liveness.OTHER_HOST: 'on another host',
    None: 'liveness unknown',
}
_INVALID_DETAIL_CHARS: Final = 300


@dataclass(frozen=True)
class HoldSummary:
    label: str
    state: HoldState
    age_seconds: int | None
    needs_human: NeedsHumanReason | None
    detail: str

    def to_wire(self) -> dict[str, object]:
        return {
            'label': self.label,
            'state': self.state.value,
            'age_seconds': self.age_seconds,
            'needs_human': None if self.needs_human is None else self.needs_human.value,
            'detail': self.detail,
        }


class _NotRead(Enum):
    NOT_READ = 'not_read'


_CARRIER_NOT_READ: Final = _NotRead.NOT_READ


@dataclass(frozen=True)
class _Referents:
    """``needs_human``'s two referent arguments for one hold; each kind ignores the other one.

    Attributes:
        carrier_status: ``None`` when the carrier is absent from a measured
            status map; ``_CARRIER_NOT_READ`` when the map was not measured.
        liveness: ``None`` when the holder was not read, as the shared model
            spells it.
    """

    carrier_status: TaskStatus | _NotRead | None = None
    liveness: Liveness | None = None


def _is_deferred(task: Mapping[str, Any]) -> bool:
    return task.get('status') == TaskStatus.DEFERRED


def _record_of(task: Mapping[str, Any]) -> Deferral | InvalidRecord | None:
    return parse_record((task.get('metadata') or {}).get('deferral'))


def session_holders(rows: Iterable[Mapping[str, Any]]) -> dict[int, ProcessIdentity]:
    """The holder of every DEFERRED row held by a session, by row id."""
    holders: dict[int, ProcessIdentity] = {}
    for row in rows:
        record = _record_of(row) if _is_deferred(row) else None
        if isinstance(record, HeldBySessionDeferral):
            holders[row['id']] = record.holder
    return holders


def _read_one(holder: ProcessIdentity, proc_root: Path) -> Liveness | None:
    try:
        return liveness(holder, proc_root=proc_root)
    except ProcessIdentityUnreadable:
        return None


def read_liveness(
    holders: Mapping[int, ProcessIdentity], *, proc_root: Path = PROC_ROOT
) -> dict[int, Liveness | None]:
    """Each holder's liveness from /proc; ``None`` where /proc could not answer."""
    return {task_id: _read_one(holder, proc_root) for task_id, holder in holders.items()}


def _carrier_referents(record: CarriedByDeferral, reads: HoldReads) -> _Referents:
    if reads.carrier_statuses is None:
        return _Referents(carrier_status=_CARRIER_NOT_READ)
    status = reads.carrier_statuses.get(record.carrier_task_id)
    return _Referents(carrier_status=None if status is None else TaskStatus(status))


def _referents(task_id: int, record: Deferral | InvalidRecord, reads: HoldReads) -> _Referents:
    if isinstance(record, CarriedByDeferral):
        return _carrier_referents(record, reads)
    if isinstance(record, HeldBySessionDeferral):
        return _Referents(liveness=reads.holder_liveness.get(task_id))
    return _Referents()


def _verdict(
    record: Deferral | InvalidRecord, referents: _Referents, now: datetime
) -> NeedsHumanReason | None:
    if referents.carrier_status is _CARRIER_NOT_READ:
        return None
    return needs_human(record, now, referents.carrier_status, referents.liveness)


def _state(verdict: NeedsHumanReason | None, referents: _Referents) -> HoldState:
    holder_unread = verdict is NeedsHumanReason.LIVENESS_UNKNOWN
    if referents.carrier_status is _CARRIER_NOT_READ or holder_unread:
        return HoldState.UNKNOWN
    return HoldState.HELD if verdict is None else HoldState.NEEDS_HUMAN


def _utc_minute(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')


def _carrier_qualifier(verdict: NeedsHumanReason | None, referents: _Referents) -> str:
    if referents.carrier_status is _CARRIER_NOT_READ:
        return _CARRIER_NOT_READ_QUALIFIER
    return _CARRIER_QUALIFIERS[verdict]


def _lapse_clause(record: UntilConditionDeferral) -> str:
    return '' if record.expires_at is None else f' (lapses {_utc_minute(record.expires_at)})'


def _label(
    record: Deferral | InvalidRecord, verdict: NeedsHumanReason | None, referents: _Referents
) -> str:
    match record:
        case CarriedByDeferral():
            qualifier = _carrier_qualifier(verdict, referents)
            return f'parked: carried by #{record.carrier_task_id}{qualifier}'
        case UntilConditionDeferral():
            return f'parked until {record.condition}{_lapse_clause(record)}'
        case HeldBySessionDeferral():
            return f'held by session {record.holder.pid} ({_LIVENESS_WORDS[referents.liveness]})'
        case PlanningDeferral():
            return 'planning batch'
        case LegacyUnknownDeferral():
            return 'legacy: no recorded reason'
        case InvalidRecord():
            return 'invalid deferral record'
        case _:
            assert_never(record)


def _age_seconds(record: Deferral | InvalidRecord, now: datetime) -> int | None:
    if isinstance(record, InvalidRecord) or record.deferred_at is None:
        return None
    return max(0, int((now - record.deferred_at).total_seconds()))


def _one_line(text: str) -> str:
    flat = ' '.join(text.split())
    return flat if len(flat) <= _INVALID_DETAIL_CHARS else f'{flat[:_INVALID_DETAIL_CHARS]}…'


def _detail(record: Deferral | InvalidRecord) -> str:
    if isinstance(record, InvalidRecord):
        return _one_line(record.detail)
    stamped = f'stamped by {record.stamped_by or "unattributed"} at {_utc_minute(record.stamped_at)}'
    return f'{stamped}; reason: {record.reason}' if record.reason else stamped


def _summarize(
    task_id: int, record: Deferral | InvalidRecord, *, now: datetime, reads: HoldReads
) -> HoldSummary:
    referents = _referents(task_id, record, reads)
    verdict = _verdict(record, referents, now)
    return HoldSummary(
        label=_label(record, verdict, referents),
        state=_state(verdict, referents),
        age_seconds=_age_seconds(record, now),
        needs_human=verdict,
        detail=_detail(record),
    )


def deferral_row_fields(
    task: Mapping[str, Any], *, now: datetime, reads: HoldReads
) -> dict[str, dict[str, object] | None]:
    """The ``deferral`` row key: present only on a deferred row, ``None`` when it has no record."""
    if not _is_deferred(task):
        return {}
    record = _record_of(task)
    if record is None:
        return {'deferral': None}
    return {'deferral': _summarize(task['id'], record, now=now, reads=reads).to_wire()}
