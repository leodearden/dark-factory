"""The task_lookup datum: a few tasks' rows by id, never the whole tree.

Declared by ``plans/dashboard-one-datum-one-path-prd.md``, decision 12. A
surface that names tasks it did not list — a merge row, an escalation — needs
each one's dashboard row (its title, its status). It used to fetch every
project's whole task tree per request to find a handful of titles.

:func:`lookup_tasks` answers each :class:`TaskRef` with a ``Datum[dict]``
whose value is the ``tasks.fetch_task`` row, read from the first of three
sources that holds it:

1. the snapshot unit's ACTIVE rows (``task_snapshot.acquire_snapshot``), which
   the Tasks tab already pays for — served with that rows datum's own
   provenance, so an active id is never older than the unit;
2. a per-id cache holding only answers that cannot change under it — a
   TERMINAL row, or "no such task" — for :data:`TERMINAL_ROW_TTL_SECONDS`;
3. one ``get_task`` per remaining id, through ``tasks.fetch_task``.

The whole lookup is bounded as ONE operation by :data:`LOOKUP_BUDGET_SECONDS`.
Within it, at most :data:`LOOKUP_MISS_CAP` misses are read per call, newest
(highest) id first, at most :data:`LOOKUP_CONCURRENCY` at a time. An id past
the cap or the deadline is ``unknown`` with a reason starting
:data:`LOOKUP_BUDGET_REASON`, and the per-id cache lets the next poll read the
next batch. Answers are recorded as they land, so ids resolved before the
deadline survive it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

import httpx
from shared.task_statuses import TERMINAL

from dashboard.config import DashboardConfig
from dashboard.data.datum import Datum, DatumState
from dashboard.data.mcp_fanout import TTLCache
from dashboard.data.task_snapshot import acquire_snapshot
from dashboard.data.tasks import (
    DEFAULT_PER_CALL_TIMEOUT,
    DEFAULT_WHOLE_OPERATION_BUDGET,
    TaskNotFound,
    TaskReadOffline,
    TaskRowRead,
    fetch_task,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, order=True)
class TaskRef:
    """One task, named by the project root that holds it and its id."""

    project_root: str
    task_id: int


LOOKUP_BUDGET_SECONDS = DEFAULT_WHOLE_OPERATION_BUDGET
"""The whole lookup's deadline. Bound by reference: a call site may only tighten it."""

LOOKUP_CONCURRENCY = 4
"""How many ``get_task`` misses one call reads at once."""

LOOKUP_MISS_CAP = 64
"""How many ``get_task`` misses one call reads at most, highest ids first."""

TERMINAL_ROW_TTL_SECONDS = 600.0
"""How long a terminal row or a not-found answer is held per id."""

FETCHED_ROW_FRESHNESS_BOUND_SECONDS = int(2 * TERMINAL_ROW_TTL_SECONDS)
"""Twice the TTL, so a held row served at any age inside it is still fresh.

The snapshot unit's rule (``task_snapshot.FRESHNESS_BOUND_SECONDS``): a bound
below the TTL would make ``validate_datum`` refuse this module's own correct
cached output.
"""

LOOKUP_BUDGET_REASON = 'lookup budget'
"""The prefix of every reason an id was left unread by this call's budget."""


@dataclass(frozen=True, slots=True)
class _Fetched:
    """One ``get_task`` answer and the instant it was read."""

    outcome: TaskRowRead
    fetched_at: datetime

    def holds_still(self) -> bool:
        """A terminal row or a not-found answer; nothing live, nothing offline."""
        if isinstance(self.outcome, TaskNotFound):
            return True
        return isinstance(self.outcome, dict) and self.outcome.get('status') in TERMINAL


_lookup_cache: TTLCache[_Fetched, TaskRef] = TTLCache(
    ttl_seconds=lambda: TERMINAL_ROW_TTL_SECONDS
)


def _lookup_cache_clear() -> None:
    """Clear the per-id cache (test/admin hook)."""
    _lookup_cache.clear()


def _unknown(reason: str) -> Datum[dict]:
    return Datum(None, None, DatumState.UNKNOWN, reason, FETCHED_ROW_FRESHNESS_BOUND_SECONDS)


def _datum_of(ref: TaskRef, fetched: _Fetched) -> Datum[dict]:
    outcome = fetched.outcome
    if isinstance(outcome, TaskNotFound):
        return _unknown(f'task {ref.task_id} is not in {ref.project_root}: {outcome.detail}')
    if isinstance(outcome, TaskReadOffline):
        return _unknown(outcome.detail or 'no fused-memory URL is configured')
    return Datum(
        outcome, fetched.fetched_at, DatumState.FRESH, None,
        FETCHED_ROW_FRESHNESS_BOUND_SECONDS,
    )


async def _active_rows(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    roots: Sequence[str],
    *,
    now: datetime,
) -> dict[TaskRef, Datum[dict]]:
    """Every active row the snapshot units of *roots* hold, as its own Datum."""
    snapshots = await asyncio.gather(
        *(acquire_snapshot(client, config, root, now=now) for root in roots),
    )
    served: dict[TaskRef, Datum[dict]] = {}
    for root, snapshot in zip(roots, snapshots, strict=True):
        rows = snapshot.rows
        for row in rows.value or ():
            served[TaskRef(root, row['id'])] = Datum(
                row, rows.as_of, rows.state, rows.reason, rows.freshness_bound_seconds,
            )
    return served


async def _resolve(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    wanted: Sequence[TaskRef],
    resolved: dict[TaskRef, Datum[dict]],
    *,
    now: datetime,
) -> None:
    """Record into *resolved* each ref's answer as soon as it is known.

    A held answer needs no I/O, so it is recorded BEFORE the snapshot read: a
    deadline spent there cannot blank it. An active row read afterwards still
    wins, because a held terminal id may since have been reopened.
    """
    held = {
        ref: fetched for ref in wanted
        if (fetched := _lookup_cache.get_fresh(ref)) is not None
    }
    for ref, fetched in held.items():
        resolved[ref] = _datum_of(ref, fetched)
    active = await _active_rows(
        client, config, sorted({ref.project_root for ref in wanted}), now=now,
    )
    misses: list[TaskRef] = []
    for ref in wanted:
        if ref in active:
            resolved[ref] = active[ref]
        elif ref not in held:
            misses.append(ref)

    for ref in misses[LOOKUP_MISS_CAP:]:
        resolved[ref] = _unknown(
            f"{LOOKUP_BUDGET_REASON}: beyond this request's {LOOKUP_MISS_CAP}-id miss cap"
        )

    # Per call: a module-level asyncio primitive binds to the first event loop
    # that contends it.
    width = asyncio.Semaphore(LOOKUP_CONCURRENCY)

    async def _read(ref: TaskRef) -> None:
        async def _refresh() -> _Fetched:
            async with width:
                outcome = await fetch_task(
                    client, config, ref.project_root, ref.task_id,
                    timeout=DEFAULT_PER_CALL_TIMEOUT,
                )
            return _Fetched(outcome, now)

        fetched = await _lookup_cache.get_or_refresh(
            ref, _refresh, cache_ok=_Fetched.holds_still,
        )
        resolved[ref] = _datum_of(ref, fetched)

    await asyncio.gather(*(_read(ref) for ref in misses[:LOOKUP_MISS_CAP]))


async def lookup_tasks(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    refs: Iterable[TaskRef],
    *,
    now: datetime,
) -> dict[TaskRef, Datum[dict]]:
    """Each distinct ref's dashboard row as a ``Datum``, within one deadline.

    *now* is the caller's resolved instant: a row read here is stamped with
    it, and the snapshot unit is acquired at it. Every distinct ref gets an
    entry; one this call could not read is ``unknown`` and says why.
    """
    wanted = sorted(set(refs), key=lambda ref: (-ref.task_id, ref.project_root))
    resolved: dict[TaskRef, Datum[dict]] = {}
    try:
        await asyncio.wait_for(
            _resolve(client, config, wanted, resolved, now=now),
            timeout=LOOKUP_BUDGET_SECONDS,
        )
    except TimeoutError:
        unread = [ref for ref in wanted if ref not in resolved]
        logger.warning(
            'lookup_tasks: exceeded the %.1fs whole-operation budget — %d of %d '
            'task id(s) are UNKNOWN for this poll',
            LOOKUP_BUDGET_SECONDS, len(unread), len(wanted),
        )
        for ref in unread:
            resolved[ref] = _unknown(
                f'{LOOKUP_BUDGET_REASON}: the {LOOKUP_BUDGET_SECONDS}s lookup '
                f'deadline expired before task {ref.task_id} was read'
            )
    return resolved
