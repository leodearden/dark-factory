"""Is an INV-12 debt owner alive, and through whom.

The owner sweep (``scripts/exception_owner_sweep.py``) asks this module one
question per cited owner.  A task is read from tasks.db and classified by
``shared.task_statuses.owner_liveness``; a closed task may still own its debt
through a successor, followed in this order:

* ``metadata.deferral`` as a ``carried_by`` record — its home is
  ``shared/src/shared/task_deferral.py``;
* the legacy forward pointer ``metadata.x_coalesced_into``, written by the
  2026-09-09 backlog sweep;
* for a ticket owner, the task its ticket resolved to (``created`` or
  ``combined``), read through fused-memory's ``resolve_ticket``.

This module files nothing and holds no transport: the store path and the
``resolve_ticket`` lookup arrive as arguments, so every answer is a function
of what the caller hands in.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import _checkout as _checkout
from _task_db_scan import connect_ro, decode_metadata
from shared.task_deferral import CarriedByDeferral, parse_record
from shared.task_statuses import OwnerLiveness, owner_liveness

_SELECT_ROWS = "SELECT id, status, title, metadata FROM tasks WHERE tag = 'master'"


def _positive_id(value: object) -> int | None:
    """*value* as a task id when it is a positive int (never a bool) or its decimal text."""
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is int and value > 0:
        return value
    return None


@dataclass(frozen=True)
class TaskRow:
    """One master-tag task as the sweep needs it, read-only."""

    id: int
    status: str
    title: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))

    def liveness(self) -> OwnerLiveness:
        return owner_liveness(self.status)

    def successor(self) -> int | None:
        """The task this one hands its work to, or None when it names none."""
        deferral = parse_record(self.metadata.get('deferral'))
        if isinstance(deferral, CarriedByDeferral):
            return deferral.carrier_task_id
        return _positive_id(self.metadata.get('x_coalesced_into'))


def _task_row(raw: tuple[int, str, str, object]) -> TaskRow:
    task_id, status, title, metadata = raw
    return TaskRow(id=task_id, status=status, title=title, metadata=decode_metadata(metadata))


def read_tasks(db: Path, ids: Collection[int]) -> Mapping[int, TaskRow]:
    """The master-tag rows among *ids*; an absent id is simply missing.

    Store refusals propagate as ``_task_db_scan.UNREADABLE_STORE_ERRORS``.
    """
    wanted = sorted(set(ids))
    if not wanted:
        return MappingProxyType({})
    placeholders = ', '.join('?' for _ in wanted)
    conn = connect_ro(db)
    try:
        raw = conn.execute(f'{_SELECT_ROWS} AND id IN ({placeholders})', wanted).fetchall()
    finally:
        conn.close()
    return MappingProxyType({row.id: row for row in map(_task_row, raw)})


def read_tasks_carrying(db: Path, field: str) -> tuple[TaskRow, ...]:
    """Every master-tag row whose decoded metadata holds a non-None *field*, by id.

    The ``LIKE`` is only a byte prefilter; the decoded value is the authority.
    """
    conn = connect_ro(db)
    try:
        raw = conn.execute(
            f'{_SELECT_ROWS} AND metadata LIKE ? ORDER BY id', (f'%"{field}"%',)
        ).fetchall()
    finally:
        conn.close()
    rows = map(_task_row, raw)
    return tuple(row for row in rows if row.metadata.get(field) is not None)


class TicketState(enum.Enum):
    """What a ticket says about its owner: still curating, a task, or nothing."""

    PENDING = 'pending'
    RESOLVED_TO_TASK = 'resolved_to_task'
    DEAD = 'dead'


@dataclass(frozen=True)
class TicketOutcome:
    """``resolve_ticket``'s answer, parsed once: the state plus the wire facts behind it."""

    state: TicketState
    status: str
    task_id: int | None
    reason: str | None


class TicketUnreadable(Exception):
    """``resolve_ticket`` answered something that says nothing about the ticket.

    Never a dead owner: an outage must not read as death (INV-11).

    Attributes:
        result: The answer, verbatim.
    """

    def __init__(self, message: str, *, result: object) -> None:
        self.result = result
        super().__init__(message)


UNKNOWN_TICKET_REASON = 'unknown_ticket'

_UNPERSISTED_FAILURES: Mapping[str, TicketState | None] = MappingProxyType(
    {
        'timeout': TicketState.PENDING,
        UNKNOWN_TICKET_REASON: TicketState.DEAD,
        'server_closed': None,
        'ticket_store not configured': None,
    }
)
"""The server's ``failed`` answers that no ticket row carries, by reason.

Their home is
``fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor._resolve_ticket_raw``.
``timeout`` is a ticket still being curated; ``unknown_ticket`` names no
ticket at all.  None marks a server that could not look: the answer is
unreadable, not a verdict."""

_RESOLVED_STATUSES = frozenset({'created', 'combined'})
_DEAD_STATUSES = frozenset({'failed', 'refused', 'cancelled'})


def _ticket_state(status: str, reason: str | None, task_id: int | None) -> TicketState | None:
    if status == 'pending':
        return TicketState.PENDING
    if status in _RESOLVED_STATUSES:
        return TicketState.RESOLVED_TO_TASK if task_id is not None else None
    if status == 'failed' and reason is not None and reason in _UNPERSISTED_FAILURES:
        return _UNPERSISTED_FAILURES[reason]
    if status in _DEAD_STATUSES:
        return TicketState.DEAD
    return None


def ticket_outcome(result: object) -> TicketOutcome:
    """Parse ``resolve_ticket``'s wire dict, or raise :class:`TicketUnreadable`."""
    status = result.get('status') if isinstance(result, Mapping) else None
    if not isinstance(result, Mapping) or 'error' in result or not isinstance(status, str):
        raise TicketUnreadable(f'resolve_ticket answered {result!r}', result=result)
    reason = result.get('reason')
    reason = reason if isinstance(reason, str) else None
    task_id = _positive_id(result.get('task_id'))
    state = _ticket_state(status, reason, task_id)
    if state is None:
        raise TicketUnreadable(
            f'resolve_ticket answered {result!r}, which names no ticket state', result=result
        )
    return TicketOutcome(state=state, status=status, task_id=task_id, reason=reason)
