"""Every deferred row's hold, classified with the shared deferral model.

Contract: plans/deferral-flip-condition-prd.md §4 decision 10 ("Census") and §9 ε.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final

from deferral_store import TaskRow
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
    NeedsHumanReason,
    needs_human,
)
from shared.task_statuses import TaskStatus

# The spawn_context the flake ledger stamps on the de-flake owners it files:
# orchestrator/src/orchestrator/flake_ledger.py::build_deflake_task_arguments
DEFLAKE_SPAWN_CONTEXT: Final = 'flake_ledger_debt'
NO_RECORD: Final = 'no record'
INVALID_RECORD: Final = 'invalid record'

_FAILS_CHECK: Final = frozenset(
    {NeedsHumanReason.MISSING_RECORD, NeedsHumanReason.INVALID_RECORD, NeedsHumanReason.EXPIRY_OVERDUE}
)


@dataclass(frozen=True)
class RowVerdict:
    task_id: int
    title: str
    held: str
    needs_human: NeedsHumanReason | None
    deflake_owner: bool


@dataclass(frozen=True)
class Referents:
    """What the recorded holds point at: carrier task ids, and each held row's holder."""

    carrier_ids: frozenset[int]
    holders: Mapping[int, ProcessIdentity]


def _held(record: Deferral | InvalidRecord | None) -> str:
    match record:
        case None:
            return NO_RECORD
        case InvalidRecord():
            return INVALID_RECORD
        case _:
            return record.kind.value


def _is_deflake_owner(row: TaskRow) -> bool:
    return row.metadata is not None and row.metadata.get('spawn_context') == DEFLAKE_SPAWN_CONTEXT


def referents(rows: Iterable[TaskRow]) -> Referents:
    carrier_ids: set[int] = set()
    holders: dict[int, ProcessIdentity] = {}
    for row in rows:
        record = row.deferral
        if isinstance(record, CarriedByDeferral):
            carrier_ids.add(record.carrier_task_id)
        elif isinstance(record, HeldBySessionDeferral):
            holders[row.task_id] = record.holder
    return Referents(carrier_ids=frozenset(carrier_ids), holders=MappingProxyType(holders))


def _verdict(
    row: TaskRow,
    now: datetime,
    carrier_statuses: Mapping[int, TaskStatus],
    holder_liveness: Mapping[int, Liveness | None],
) -> RowVerdict:
    record = row.deferral
    carrier_status = (
        carrier_statuses.get(record.carrier_task_id) if isinstance(record, CarriedByDeferral) else None
    )
    holder = holder_liveness.get(row.task_id) if isinstance(record, HeldBySessionDeferral) else None
    return RowVerdict(
        task_id=row.task_id,
        title=row.title,
        held=_held(record),
        needs_human=needs_human(record, now, carrier_status, holder),
        deflake_owner=_is_deflake_owner(row),
    )


def classify_deferred(
    rows: Iterable[TaskRow],
    *,
    now: datetime,
    carrier_statuses: Mapping[int, TaskStatus],
    holder_liveness: Mapping[int, Liveness | None],
) -> tuple[RowVerdict, ...]:
    """One verdict per row; a carrier absent from *carrier_statuses* is missing, a holder absent from
    *holder_liveness* is unread."""
    return tuple(_verdict(row, now, carrier_statuses, holder_liveness) for row in rows)


def fails_check(verdict: RowVerdict) -> bool:
    return verdict.needs_human in _FAILS_CHECK


def holder_liveness(identity: ProcessIdentity, *, proc_root: Path = PROC_ROOT) -> Liveness | None:
    """The holder's liveness, or None when /proc could not answer (never read as dead)."""
    try:
        return liveness(identity, proc_root=proc_root)
    except ProcessIdentityUnreadable:
        return None
