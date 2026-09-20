"""ONE task snapshot per project root: two reads, two ``Datum``s, one TTL.

Declared by ``plans/dashboard-one-datum-one-path-prd.md``. Before this module
the Tasks tab acquired a project's state through three separately-cached reads
at two different TTLs, and reported the result as a row list, a bare count and
three parallel project lists — none of which carried the instant it was
measured, so a zero could equally mean "measured zero", "the read failed" or
"the budget expired".

The unit here is the fix. Both halves — the ACTIVE ROWS and the compact
``{id: status}`` map that becomes the census — are read under one TTL and
stamped with one instant, and each crosses the wire inside a
:class:`~dashboard.data.datum.Datum` that says how fresh it is and, when it is
not fresh, why. A consumer never has to infer provenance from a value.

WHAT LIVES HERE AND WHY. This module owns the two reads, the budgets that
bound them, and the state machine that turns a read outcome into a ``Datum``.
It does NOT shape task rows: ``active_tasks`` does that, consuming
``snapshot.rows.value`` and ``snapshot.status_map``. The split is the point —
row shaping is pure given a snapshot, and acquiring a snapshot is pure I/O.
"""

from __future__ import annotations

import asyncio
import enum
import os
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Generic, TypeVar

import httpx
from shared.task_statuses import ACTIVE

from dashboard.config import DashboardConfig
from dashboard.data.census import CensusVocabularyError, TaskCensus, build_census
from dashboard.data.datum import Datum, DatumState, validate_datum
from dashboard.data.mcp_fanout import TTLCache
from dashboard.data.tasks import fetch_statuses, fetch_tasks, task_is_stranded

# --- Budget constants -------------------------------------------------------

PER_PROJECT_MCP_CALLS: tuple[str, ...] = (
    'get_tasks[active]',
    'get_statuses[walk]',
    'get_tasks[terminal]',
)
"""The bounded OPERATIONS one project root can cost, as a roster.

OPERATIONS, not HTTP requests, and the distinction is load-bearing: the status
map is walked in :data:`~dashboard.data.tasks.STATUSES_SAFE_PAGE_SIZE` pages,
so ``get_statuses[walk]`` is ``ceil(N / 2000)`` requests on the wire — three on
this repo's own tree. It is nonetheless ONE entry here because it is bounded as
one, by a single ``asyncio.wait_for(PER_CALL_TIMEOUT)`` around the whole walk.
Counting requests instead would make ``test_tasks_budget.py``'s arithmetic read
``4.4 * 4 > 14.0`` and invite a bound-raising that no measurement supports.

A named tuple rather than a literal ``3``: the invariant then tracks reality,
so adding a fourth bounded operation without raising the budget fails a
structural test instead of silently overrunning in production.

The DEFAULT render spends only the first two. ``get_tasks[terminal]`` is
reserved for the ``?terminal=`` request, which is the only one that asks for
terminal rows.
"""

PER_CALL_TIMEOUT = 4.4
"""Budget for ONE bounded operation, applied at both layers it has.

As the ``timeout=`` keyword it bounds each HTTP request; as the
``asyncio.wait_for`` around a half-read it bounds the whole operation, which
for the paged map walk is the only place its total cost is capped. The two
layers are complementary rather than redundant, the same idiom
``task_runtime.py`` and ``collect_tasks_with_counts`` already use.

MEASURED 2026-09-07 against the live fused-memory (localhost:8002), 9
configured roots, caches cleared per root, per-call timeout temporarily raised
to 10.0 so a slow root reported its real latency instead of a ReadTimeout:
per-CALL max 2.696 s, per-ROOT wall max 2.876 s (dark-factory 5128 tasks,
reify 7279), p95 across roots 1.695 s. Everything else was under 0.1 s — the
distribution is two big trees and seven small ones, not a uniform cost.

4.4 = 1.5 * the 2.876 s per-root wall max, rounded up to one decimal.

WHY 2.0 IS TOO SMALL, on evidence rather than on principle: at
``tasks.DEFAULT_PER_CALL_TIMEOUT`` the same measurement's truly-cold render
marked dark-factory, reify AND autopilot-video OFFLINE and shipped 208 of 3045
active rows — the Tasks tab reporting the three largest projects unreachable
while fused-memory was serving them fine. Offline is a claim that a read
demonstrably failed; a per-call budget below the honest service time turns
that claim into a lie on every cold render.

WHY THE SERVER COST SCALES WITH TREE SIZE even though the row read is
status-narrowed: there is no field projection at any layer and the backend
query is ``SELECT *`` feeding a fixed 14-key row — see ``fetch_tasks``'
docstring, and task 4390, the open follow-up to add projection. Until that
lands, narrowing the STATUSES does not narrow the WORK.

WHY THIS IS TASKS-TAB-LOCAL rather than a bump of the shared
``tasks.DEFAULT_PER_CALL_TIMEOUT``: that constant feeds
``tasks.DEFAULT_WHOLE_OPERATION_BUDGET``, which
``orchestrator._ORCHESTRATORS_PER_ROOT_BUDGET``,
``merge_queue._TASK_TITLES_BUDGET`` and ``app._TASK_CARDS_BUDGET`` all bind BY
REFERENCE (task 4788). Raising the shared default to fix the Tasks tab would
silently widen three unrelated route budgets, none of which fetches a
5 000-task tree. ``test_tasks_budget.py`` assertion (e) pins both halves.
"""

SNAPSHOT_TTL_SECONDS = 15.0
"""How long one acquired unit is served before both halves are re-read.

THE ONLY TTL ON THIS PATH. The 5 s ``fetch_statuses`` cache was removed and
the row read passes ``cached=False``, so nothing underneath holds a value that
could be older than the ``as_of`` this unit stamps on it.

15 s sits inside the PRD's 15-30 s staleness window for a monitoring view, and
is five times ``data.js``'s 3 s poll, so a browser polling two endpoints that
both reach this unit costs one pair of reads per root per 15 s rather than one
per poll.
"""

FRESHNESS_BOUND_SECONDS = 30
"""The age past which a half of this unit stops calling itself fresh.

Twice :data:`SNAPSHOT_TTL_SECONDS`, and derived from it rather than from the
browser's poll interval. ``validate_datum`` raises ``DatumContractError`` when
a ``FRESH`` datum is older than this bound, and a unit legitimately served
from its own cache at age 14.9 s is fresh by its producer's own refresh
contract — so any bound below the TTL would make this layer raise on its own
correct output, on the routine path. The browser's poll cannot make a value
newer than the cache allows, so its cadence is not the one that governs here.
"""

_RETENTION_BOUND_SECONDS = 24 * 60 * 60
"""How old a last-good value may be and still be served as ``stale``."""


class SnapshotFailure(enum.StrEnum):
    """WHY a unit is not wholly measured — structured, never parsed from prose.

    The three project lists ``/api/v2/dashboard/tasks`` emits encode facts the
    codebase insists must not be merged: offline means a read DEMONSTRABLY
    failed, degraded means a budget expired and the state is simply unknown.
    ``Datum.reason`` is by contract the producer's VERBATIM failure text, so
    routing on it would be an ad-hoc parser over a meaningful string and one
    reworded fan-out message would move a project between banners.

    The kind reported on a unit is the ROWS half's, because that is the half
    that decides whether a PROJECT is offline. The map's own fate is carried
    by the census ``Datum``'s state: rows fresh beside a non-fresh census is
    exactly the count-unknown case, and needs no third enum member.
    """

    NONE = 'none'
    UNREACHABLE = 'unreachable'
    BUDGET = 'budget'


T = TypeVar('T')


@dataclass(frozen=True, slots=True)
class _HalfRead(Generic[T]):
    """One half's outcome: what it measured, or why it measured nothing.

    ``value is None`` and ``failure is not NONE`` are the same condition, kept
    as two fields because the CALLER needs the kind and the reason separately —
    one routes the banner, the other is shown verbatim.
    """

    value: T | None
    failure: SnapshotFailure
    reason: str | None


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    """One project root's task state, measured once and stamped once.

    Attributes:
        census: Every status's population, as a ``Datum``.
        rows: The project's ACTIVE rows, as a ``Datum``.
        in_progress_live: In-progress rows with a live claimant, or ``None``
            when the rows were never measured — never a fabricated zero.
        in_progress_stranded: The complement of the above, by the same
            predicate.
        skew_seconds: How far apart the two halves' ``as_of`` instants are.
            ``None`` when either half has none: a gap between one measured
            instant and no instant is not zero.
        status_map: The raw ``{id: status}`` map. The unit's RAW MATERIAL, not
            part of its wire shape — ``active_tasks._resolve_deps`` needs it
            in-process as its only bounded fallback for a dependency outside
            the fetched rows, and skipping that fallback silently drops
            dependency chips.
        failure: Why this unit is not wholly measured; see
            :class:`SnapshotFailure`.
    """

    census: Datum[TaskCensus]
    rows: Datum[list[dict]]
    in_progress_live: int | None
    in_progress_stranded: int | None
    skew_seconds: int | None
    status_map: Mapping[int, str]
    failure: SnapshotFailure

    def to_wire(self) -> dict[str, object]:
        """Render the five contract keys; the raw map stops here."""
        return {
            'census': self.census.to_wire(),
            'rows': self.rows.to_wire(),
            'in_progress_live': self.in_progress_live,
            'in_progress_stranded': self.in_progress_stranded,
            'skew_seconds': self.skew_seconds,
        }


# One unit per project root. ``TTLCache`` rather than a hand-rolled lock: it
# carries the per-key single-flight that keeps two concurrent endpoint polls
# from each issuing their own pair of reads, the bounded lock bypass, and the
# ``_evict_expired`` reclamation that eight other call sites already depend on.
# The key space — project-root strings — is small and bounded.
_snapshot_cache: TTLCache[TaskSnapshot, str] = TTLCache(
    ttl_seconds=lambda: SNAPSHOT_TTL_SECONDS
)


def _snapshot_cache_clear() -> None:
    """Clear the snapshot unit cache (test/admin hook)."""
    _snapshot_cache.clear()


async def _bounded(coro: Awaitable[Any], *, label: str) -> _HalfRead[Any]:
    """Run one half-read under its whole-operation budget, total by contract.

    Every outcome becomes a ``_HalfRead`` rather than an exception, because the
    two halves are gathered together and one half's failure must never cancel
    or blank the other — ``rows`` fresh beside a failed map is a supported,
    user-visible state, not an error.

    The two failure kinds are kept apart at the point they are DISTINGUISHABLE:
    an offline marker means the read demonstrably failed, while a ``wait_for``
    expiry means this process ran out of budget and the server may be perfectly
    healthy. Collapsing them is what made the 2026-07-30 event get misdiagnosed
    as an orchestrator outage.
    """
    try:
        result = await asyncio.wait_for(coro, PER_CALL_TIMEOUT)
    except TimeoutError:
        return _HalfRead(
            None, SnapshotFailure.BUDGET,
            f'{label} read exceeded its {PER_CALL_TIMEOUT}s budget',
        )
    except Exception as exc:  # noqa: BLE001 — a failed half degrades, never raises
        return _HalfRead(
            None, SnapshotFailure.UNREACHABLE,
            f'{label} read raised {type(exc).__name__}: {exc}',
        )
    if isinstance(result, dict) and result.get('offline'):
        # The fan-out's own marker text, carried VERBATIM: it names the urls
        # tried and what each said, which is what an operator needs and what
        # no reworded copy of it could say.
        return _HalfRead(None, SnapshotFailure.UNREACHABLE, str(result.get('error')))
    return _HalfRead(result, SnapshotFailure.NONE, None)


def _datum(half: _HalfRead[T], *, now: datetime) -> Datum[T]:
    """Build the ``Datum`` for one half — THE one place a state is chosen.

    Routing every half through one constructor is what makes the envelope's
    ``unknown <=> value is None <=> as_of is None`` triad hold by construction
    rather than by each branch remembering to.
    """
    if half.failure is SnapshotFailure.NONE:
        return Datum(half.value, now, DatumState.FRESH, None, FRESHNESS_BOUND_SECONDS)
    return Datum(
        None, None, DatumState.UNKNOWN, half.reason or 'read failed',
        FRESHNESS_BOUND_SECONDS,
    )


def _census_half(map_half: _HalfRead[Mapping[int, str]]) -> _HalfRead[TaskCensus]:
    """Tally a measured status map, degrading a vocabulary drift into visibility.

    ``build_census`` is pure and cannot degrade itself; its docstring assigns
    that job to this layer. Catching the drift here is what turns it from fatal
    into a visible ``Datum`` carrying the producer's message verbatim.

    Keyed on whether there is a MAP TO TALLY rather than on the failure kind —
    the same condition by the record's own invariant, but this is the half the
    tally actually needs, and it carries the map's kind and reason through
    unchanged when there is none.
    """
    if map_half.value is None:
        return _HalfRead(None, map_half.failure, map_half.reason)
    try:
        return _HalfRead(build_census(map_half.value), SnapshotFailure.NONE, None)
    except CensusVocabularyError as drift:
        return _HalfRead(None, SnapshotFailure.UNREACHABLE, str(drift))


def _skew_seconds(census: Datum[Any], rows: Datum[Any]) -> int | None:
    """How far apart the two halves were measured, or None if one was not."""
    if census.as_of is None or rows.as_of is None:
        return None
    return int(abs((census.as_of - rows.as_of).total_seconds()))


def _strand_split(
    rows: Datum[list[dict]], *, now: datetime
) -> tuple[int | None, int | None]:
    """Partition the in-progress ROWS into live and stranded.

    The ROWS, not the census: the split needs ``claimant_run_id`` and
    ``heartbeat_at``, which only a row carries — a census counts a population
    and would have to invent them.

    Through ``tasks.task_is_stranded`` rather than ``shared.task_claimant``
    directly: that wrapper is the single dashboard-side strand predicate,
    binding ``STRANDED_HEARTBEAT_TTL`` and the request-scoped clock onto the
    shared function so the task-row badge and this split cannot disagree.
    """
    if rows.value is None:
        return None, None
    in_progress = [row for row in rows.value if row.get('status') == 'in-progress']
    stranded = sum(1 for row in in_progress if task_is_stranded(row, now))
    return len(in_progress) - stranded, stranded


async def _read_unit(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str,
    *,
    now: datetime,
) -> TaskSnapshot:
    """Read both halves concurrently and assemble one stamped unit."""
    rows_half, map_half = await asyncio.gather(
        _bounded(
            fetch_tasks(
                client, config, project_root,
                statuses=sorted(ACTIVE),
                timeout=PER_CALL_TIMEOUT,
                # UNCACHED, deliberately. ``_fetch_tasks_cache`` would serve
                # rows up to 20 s old under an ``as_of`` this unit stamps at
                # the present instant — the envelope's one unforgivable lie.
                # The unit's own TTL is what bounds the read cost instead.
                cached=False,
            ),
            label='active rows',
        ),
        _bounded(
            fetch_statuses(client, config, project_root, timeout=PER_CALL_TIMEOUT),
            label='status map',
        ),
    )

    rows: Datum[list[dict]] = _datum(rows_half, now=now)
    census: Datum[TaskCensus] = _datum(_census_half(map_half), now=now)
    live, stranded = _strand_split(rows, now=now)
    snapshot = TaskSnapshot(
        census=census,
        rows=rows,
        in_progress_live=live,
        in_progress_stranded=stranded,
        skew_seconds=_skew_seconds(census, rows),
        status_map=MappingProxyType(
            dict(map_half.value) if isinstance(map_half.value, dict) else {}
        ),
        failure=rows_half.failure,
    )
    # Check the contract HERE, where the producer still has the context to
    # name what it built, rather than letting a break surface as a shaping
    # crash in the HTTP layer.
    validate_datum(snapshot.census, now)
    validate_datum(snapshot.rows, now)
    return snapshot


async def acquire_snapshot(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | bytes | os.PathLike[str],
    *,
    now: datetime,
) -> TaskSnapshot:
    """Return *project_root*'s task snapshot, re-reading at most once per TTL.

    *now* is the caller's single resolved instant, threaded in rather than read
    here: both halves' ``as_of`` and every row's derived age must share one
    clock read, and a test that cannot say what the instant is cannot check it.
    A unit served from cache carries the instant it was MEASURED, which is the
    whole point — its age is then visible rather than implied.
    """
    key = str(project_root)

    async def _refresh() -> TaskSnapshot:
        return await _read_unit(client, config, key, now=now)

    return await _snapshot_cache.get_or_refresh(key, _refresh)
