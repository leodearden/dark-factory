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
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import _checkout as _checkout
from _task_db_scan import connect_ro, decode_metadata
from shared.governed_exceptions import TaskRef, TicketRef
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


class DeathCause(enum.StrEnum):
    """Why a cited owner no longer owns its debt."""

    OWNER_ABSENT = 'owner_absent'
    OWNER_CLOSED = 'owner_closed'
    SUCCESSOR_UNRESOLVED = 'successor_unresolved'
    TICKET_ABSENT = 'ticket_absent'
    TICKET_FAILED = 'ticket_failed'
    TICKET_REFUSED = 'ticket_refused'
    TICKET_CANCELLED = 'ticket_cancelled'


class TicketState(enum.Enum):
    """What a ticket says about its owner: still curating, a task, or nothing."""

    PENDING = 'pending'
    RESOLVED_TO_TASK = 'resolved_to_task'
    DEAD = 'dead'


@dataclass(frozen=True)
class TicketOutcome:
    """``resolve_ticket``'s answer, parsed once: the state plus the wire facts behind it.

    ``task_id`` is set exactly when the ticket resolved to a task.
    """

    state: TicketState
    status: str
    task_id: int | None
    reason: str | None

    def __post_init__(self) -> None:
        if (self.task_id is not None) != (self.state is TicketState.RESOLVED_TO_TASK):
            raise ValueError(
                f'TicketOutcome: state={self.state} with task_id={self.task_id!r} -- '
                'a ticket names a task exactly when it resolved to one'
            )


class TicketUnreadable(Exception):
    """``resolve_ticket`` answered something that says nothing about the ticket.

    Never a dead owner: an outage must not read as death (INV-11).

    Attributes:
        result: The answer, verbatim.
    """

    def __init__(self, message: str, *, result: object) -> None:
        self.result = result
        super().__init__(message)


_UNKNOWN_TICKET_REASON = 'unknown_ticket'

_UNPERSISTED_FAILURES: Mapping[str, TicketState | None] = MappingProxyType(
    {
        'timeout': TicketState.PENDING,
        _UNKNOWN_TICKET_REASON: TicketState.DEAD,
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

_TICKET_DEATH_CAUSES: Mapping[str, DeathCause] = MappingProxyType(
    {
        'failed': DeathCause.TICKET_FAILED,
        'refused': DeathCause.TICKET_REFUSED,
        'cancelled': DeathCause.TICKET_CANCELLED,
    }
)
"""The persisted terminal ticket statuses that yield no task, and what each means."""


def _ticket_state(status: str, reason: str | None, task_id: int | None) -> TicketState | None:
    if status == 'pending':
        return TicketState.PENDING
    if status in _RESOLVED_STATUSES:
        return TicketState.RESOLVED_TO_TASK if task_id is not None else None
    if status == 'failed' and reason is not None and reason in _UNPERSISTED_FAILURES:
        return _UNPERSISTED_FAILURES[reason]
    if status in _TICKET_DEATH_CAUSES:
        return TicketState.DEAD
    return None


def _ticket_death_cause(outcome: TicketOutcome) -> DeathCause:
    if outcome.reason == _UNKNOWN_TICKET_REASON:
        return DeathCause.TICKET_ABSENT
    return _TICKET_DEATH_CAUSES[outcome.status]


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
    resolved = state is TicketState.RESOLVED_TO_TASK
    return TicketOutcome(
        state=state, status=status, task_id=task_id if resolved else None, reason=reason
    )


class Liveness(enum.StrEnum):
    """Whether a cited owner still owns its debt, and whether directly."""

    LIVE = 'live'
    LIVE_VIA_SUCCESSOR = 'live_via_successor'
    DEAD = 'dead'


MAX_SUCCESSOR_HOPS = 10
"""How many successor links one owner may be followed through before giving up."""


@dataclass(frozen=True)
class OwnerVerdict:
    """One owner's liveness, with the facts the sweep renders; it carries no prose.

    Attributes:
        owning_task: The task that owns the debt now, if any.
        path: The task ids read, in order: the owner (or its ticket's task),
            then each successor followed.
        cause: Why the owner is dead; set exactly when it is.
        closed_status: The first task's status when it is not open.
        ticket: A ticket owner's outcome.
    """

    owner: TaskRef | TicketRef
    liveness: Liveness
    owning_task: int | None
    path: tuple[int, ...]
    cause: DeathCause | None
    closed_status: str | None
    ticket: TicketOutcome | None

    def __post_init__(self) -> None:
        if (self.cause is not None) != (self.liveness is Liveness.DEAD):
            raise ValueError(
                f'OwnerVerdict: liveness={self.liveness!s} with cause={self.cause!s} -- '
                'a verdict names a death cause exactly when it is dead'
            )


ReadTasks = Callable[[Collection[int]], Mapping[int, TaskRow]]
ResolveTicket = Callable[[TicketRef], TicketOutcome]


def _follow(
    owner: TaskRef | TicketRef, first: int, read_tasks: ReadTasks, ticket: TicketOutcome | None
) -> OwnerVerdict:
    """Read *first*, then its successors, until a live task, a dead end or the bound."""
    path: list[int] = []
    closed_status: str | None = None

    def verdict(
        liveness: Liveness, owning_task: int | None = None, cause: DeathCause | None = None
    ) -> OwnerVerdict:
        return OwnerVerdict(
            owner=owner,
            liveness=liveness,
            owning_task=owning_task,
            path=tuple(path),
            cause=cause,
            closed_status=closed_status,
            ticket=ticket,
        )

    task_id: int | None = first
    while task_id is not None:
        if task_id in path or len(path) > MAX_SUCCESSOR_HOPS:
            return verdict(Liveness.DEAD, cause=DeathCause.SUCCESSOR_UNRESOLVED)
        path.append(task_id)
        is_owner = len(path) == 1
        row = read_tasks((task_id,)).get(task_id)
        if row is None:
            absent = DeathCause.OWNER_ABSENT if is_owner else DeathCause.SUCCESSOR_UNRESOLVED
            return verdict(Liveness.DEAD, cause=absent)
        if row.liveness() is OwnerLiveness.OPEN:
            live = Liveness.LIVE if is_owner else Liveness.LIVE_VIA_SUCCESSOR
            return verdict(live, owning_task=task_id)
        if is_owner:
            closed_status = row.status
        task_id = row.successor()
    return verdict(Liveness.DEAD, cause=DeathCause.OWNER_CLOSED)


def resolve_owner(
    owner: TaskRef | TicketRef, *, read_tasks: ReadTasks, resolve_ticket: ResolveTicket
) -> OwnerVerdict:
    """Whether *owner* still owns its debt, given the two lookups.

    A ticket owner maps through its outcome first: still curating is live with
    no task yet, a ticket that yields no task is dead, and a resolved ticket is
    followed as its task.  Store errors and :class:`TicketUnreadable` propagate.
    """
    if isinstance(owner, TaskRef):
        return _follow(owner, owner.id, read_tasks, ticket=None)
    outcome = resolve_ticket(owner)
    if outcome.task_id is not None:
        return _follow(owner, outcome.task_id, read_tasks, ticket=outcome)
    pending = outcome.state is TicketState.PENDING
    return OwnerVerdict(
        owner=owner,
        liveness=Liveness.LIVE if pending else Liveness.DEAD,
        owning_task=None,
        path=(),
        cause=None if pending else _ticket_death_cause(outcome),
        closed_status=None,
        ticket=outcome,
    )
