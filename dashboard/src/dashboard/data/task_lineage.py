"""The task lineage index: every task's PRD provenance and lineage link, per root.

ACCESS (PRD ``plans/dashboard-prd-lanes-prd.md`` §4.2, §5.3), and the one
caller of :func:`dashboard.data.tasks.fetch_metadata_projection`. Each root's
index is refreshed in the background, one read at a time. A request waits for
a refresh at most :data:`LINEAGE_REQUEST_WAIT_SECONDS`, then is served what
the store holds: the newest good read, stale once a later read failed or once
it is older than its bound at the request's *now*, or unknown before any read
succeeded. No request is held past that wait.

No clock read: every instant is the caller's injected *now*, and a read's
``as_of`` is the *now* of the request that started it, so the index never
claims a measurement newer than it is.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType

import httpx

from dashboard.config import DashboardConfig
from dashboard.data.datum import Datum, DatumState, aged_at, unknown_datum
from dashboard.data.prd_attribution import LINEAGE_METADATA_KEYS, LineageEntry, lineage_entry
from dashboard.data.tasks import TaskReadOffline, fetch_metadata_projection

logger = logging.getLogger(__name__)

LINEAGE_TTL_SECONDS = 300.0
"""Age below which a root's last good read is served without a refresh.

Lineage moves only when tasks are filed or relinked, so five minutes of lag
costs a PRD view little, while one unpaged projected read moves ~1.2 MB
(measured 2026-11-04: reify 9,734 tasks, 1.24 MB; dark-factory 8,347 tasks,
1.19 MB) — not a read to repeat on every 3 s poll.
"""

LINEAGE_FRESHNESS_BOUND_SECONDS = 900
"""The lineage Datum's freshness bound, three TTLs.

It must exceed :data:`LINEAGE_TTL_SECONDS`, because a value inside its TTL
is served without a refresh. A failed read does not wait for it: the failure
is served ``stale`` at once, whatever the last good read's age, and is
retried after :data:`LINEAGE_RETRY_SECONDS`. So this bound only ages a last
good read that nothing has refreshed: after an idle spell with no request,
or while a refresh started past an old ``as_of`` is still in flight.
"""

LINEAGE_PER_CALL_TIMEOUT = 6.0
"""Per-HTTP-request budget of the projected read.

About 2.6x the 2.27 s maximum measured for one unpaged projected read on
2026-11-04 (reify 1.85–1.93 s; dark-factory 1.34–2.27 s; warm session, live
server contended by the production dashboard). The read runs in the
background, so generosity costs no request latency, while a bound too tight
would leave lineage permanently unknown.
"""

LINEAGE_READ_TIMEOUT = 20.0
"""Whole-read bound, applied with ``asyncio.wait_for`` around the projected read.

At least ``len(tasks.COLD_SESSION_POSTS)`` (3) x :data:`LINEAGE_PER_CALL_TIMEOUT`
(6.0) = 18.0 plus headroom, and far below :data:`LINEAGE_TTL_SECONDS`, so a
hung read frees its slot before the next refresh is due. Only this enclosing
bound ends an httpcore connection-lock hang (the 19.8 h incident); the
per-request timeout does not.
"""

LINEAGE_REQUEST_WAIT_SECONDS = 5.0
"""Longest a request waits on an in-flight refresh before it is served from
the store: it covers a cold first read on the common path."""

LINEAGE_RETRY_SECONDS = 30.0
"""Least interval between the start of a failed read and the next attempt,
so a down substrate is not asked again by every poll."""

TaskLineage = Mapping[int, LineageEntry]


@dataclass(frozen=True, slots=True)
class _Measured:
    value: TaskLineage
    as_of: datetime


@dataclass(frozen=True, slots=True)
class _RootState:
    last_good: _Measured | None = None
    failure: str | None = None
    refresh: asyncio.Task[None] | None = None
    attempt_started: datetime | None = None


_store: dict[str, _RootState] = {}
_generation = 0


async def acquire_lineage(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | os.PathLike[str],
    *,
    now: datetime,
) -> Datum[TaskLineage]:
    """*project_root*'s lineage index as it reads at *now*, refreshed when due."""
    root = str(project_root)
    state = _store.get(root, _RootState())
    if state.last_good is not None and _age(state.last_good.as_of, now) < LINEAGE_TTL_SECONDS:
        return aged_at(_served(state), now)
    refresh = _live_refresh(state)
    if refresh is None and not _retry_deferred(state, now):
        refresh = _start_refresh(client, config, root, now=now)
    if refresh is not None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(refresh), LINEAGE_REQUEST_WAIT_SECONDS)
    return aged_at(_served(_store.get(root, _RootState())), now)


def clear_lineage_state() -> None:
    """Empty every root's store; a read still in flight is abandoned, never written.

    It is not cancelled: it ends within :data:`LINEAGE_READ_TIMEOUT`, and the
    generation it was started under no longer matches, so its result is
    discarded.
    """
    global _generation
    _store.clear()
    _generation += 1


def _age(as_of: datetime, now: datetime) -> float:
    return (now - as_of).total_seconds()


def _live_refresh(state: _RootState) -> asyncio.Task[None] | None:
    refresh = state.refresh
    if refresh is None or refresh.done() or refresh.get_loop() is not asyncio.get_running_loop():
        return None
    return refresh


def _retry_deferred(state: _RootState, now: datetime) -> bool:
    return (
        state.failure is not None
        and state.attempt_started is not None
        and _age(state.attempt_started, now) < LINEAGE_RETRY_SECONDS
    )


def _start_refresh(
    client: httpx.AsyncClient, config: DashboardConfig, root: str, *, now: datetime,
) -> asyncio.Task[None]:
    refresh = asyncio.create_task(
        _refresh(client, config, root, started=now, generation=_generation),
    )
    _store[root] = replace(_store.get(root, _RootState()), refresh=refresh, attempt_started=now)
    return refresh


async def _refresh(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    root: str,
    *,
    started: datetime,
    generation: int,
) -> None:
    outcome = await _read(client, config, root, started=started)
    if generation != _generation:
        return
    state = _store.get(root, _RootState())
    if isinstance(outcome, _Measured):
        _store[root] = replace(state, last_good=outcome, failure=None)
    else:
        _store[root] = replace(state, failure=outcome)


async def _read(
    client: httpx.AsyncClient, config: DashboardConfig, root: str, *, started: datetime,
) -> _Measured | str:
    """One bounded projected read: the measured index, or why there is none.

    A timeout or an offline substrate is an outage and is answered quietly.
    Any other exception, from the read or from parsing its rows, is our own
    bug, and is logged with its traceback as well.
    """
    try:
        rows = await asyncio.wait_for(
            fetch_metadata_projection(
                client, config, root,
                metadata_keys=LINEAGE_METADATA_KEYS, timeout=LINEAGE_PER_CALL_TIMEOUT,
            ),
            LINEAGE_READ_TIMEOUT,
        )
        if isinstance(rows, TaskReadOffline):
            return rows.detail
        index = {row.id: lineage_entry(row.metadata, root) for row in rows}
    except TimeoutError:
        return f'lineage read exceeded its {LINEAGE_READ_TIMEOUT}s budget'
    except Exception as exc:
        logger.warning(
            'project %s: unexpected error while reading its lineage index, which keeps '
            'its last good read (or none); this is a BUG, not an outage',
            root, exc_info=True,
        )
        return f'lineage read raised {type(exc).__name__}: {exc}'
    return _Measured(MappingProxyType(index), as_of=started)


def _served(state: _RootState) -> Datum[TaskLineage]:
    good = state.last_good
    if good is None:
        reason = state.failure or (
            f'first lineage read still in flight after {LINEAGE_REQUEST_WAIT_SECONDS}s'
        )
        return unknown_datum(reason, LINEAGE_FRESHNESS_BOUND_SECONDS)
    return Datum(
        value=good.value,
        as_of=good.as_of,
        state=DatumState.FRESH if state.failure is None else DatumState.STALE,
        reason=state.failure,
        freshness_bound_seconds=LINEAGE_FRESHNESS_BOUND_SECONDS,
    )
