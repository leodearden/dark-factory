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

TWO STRUCTURES, ONE OWNER. The TTL cache decides WHETHER to re-read; the
last-good store decides WHAT to serve when a read fails. ``TTLCache`` alone
cannot do both — ``get_or_refresh`` only ever serves a FRESH entry, so it has
no previous value to fall back on — and one structure doing both would have to
conflate "this value is still current enough to reuse" with "this value is the
best evidence we have". They are not the same question and they expire on
different clocks: the TTL is 15 s, the last good is held for
:data:`_RETENTION_BOUND_SECONDS`.

A FULLY-STALE UNIT IS ITSELF CACHED for the TTL (``cache_ok`` is left at its
always-true default). That is what replaces ``_FETCH_TASKS_NEGATIVE_TTL_SECONDS``'
retry suppression on this path, and it is strictly longer: a wedged root costs
one attempt per 15 s instead of one per 3 s browser poll. The price is up to
15 s of recovery latency — a recovered root is noticed on the next refresh,
not on the next poll — and that is deliberate.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Any, Generic, TypeVar

import httpx
from shared.task_statuses import ACTIVE, TERMINAL

from dashboard.config import DashboardConfig
from dashboard.data.census import (
    CensusVocabularyError,
    TaskCensus,
    TaskView,
    build_census,
)
from dashboard.data.datum import Datum, DatumState, validate_datum
from dashboard.data.mcp_fanout import TTLCache
from dashboard.data.tasks import (
    fetch_statuses,
    fetch_task_page,
    fetch_tasks,
    task_is_stranded,
)

logger = logging.getLogger(__name__)

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
``merge_queue._TASK_TITLES_BUDGET`` and ``escalations._TASK_CARDS_BUDGET``
both bind BY REFERENCE (task 4788). Raising the shared default to fix the
Tasks tab would silently widen unrelated route budgets, neither of which
fetches a 5 000-task tree. ``test_tasks_budget.py`` assertion (e) pins both
halves.
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
        rows: The project's ACTIVE rows, as a ``Datum``, in one of two shapes
            depending on which unit holds it. The unit :func:`acquire_snapshot`
            returns and caches holds the RAW MCP rows. The unit
            ``active_tasks.collect_tasks_with_counts`` returns holds the shaped
            ``TaskRow`` list, the same row dicts as ``ACTIVE_TASKS``, and that
            one crosses the wire. The two cannot be one list. Shaping reads the
            integer ids, ``dependencies`` and ``metadata`` that only a raw row
            carries, so the cache must keep raw rows for the next render to
            shape. And a render's shaped rows belong to that render: they are
            joined to its runtime probe and its clock, and its external-dep
            tail overwrites them in place, so a cache shared across renders
            must never hold them. Raw rows on the wire would ship the whole
            ``metadata`` blob a second time beside ``ACTIVE_TASKS`` and break
            the PRD's ``Datum[list[TaskRow]]`` contract.
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


@dataclass(frozen=True, slots=True)
class _LastGood:
    """The most recent successful read of one half, and when it happened.

    ``value`` is ``Any`` because the store is keyed by half NAME and so holds
    a census beside a row list; the per-half type is recovered where it is
    read, by the ``_HalfRead[T]`` that half arrived in.
    """

    value: Any
    as_of: datetime


# The last successful read of each half, per project root. Deliberately NOT
# the TTL cache: that structure expires an entry at 15 s, and an expired entry
# is exactly the one a failed refresh needs. Written only on success, so a
# failure can never overwrite the evidence it is about to fall back on.
_last_good: dict[str, dict[str, _LastGood]] = {}


def _snapshot_cache_clear() -> None:
    """Clear the snapshot unit cache AND the last-good store (test/admin hook).

    Both, because a test that clears only the TTL would still be served the
    previous test's values as ``stale`` — the one thing a clear hook exists to
    prevent.
    """
    _snapshot_cache.clear()
    _last_good.clear()


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


def _datum(
    half: _HalfRead[T], *, now: datetime, project_root: str, name: str
) -> Datum[T]:
    """Build the ``Datum`` for one half-outcome — THE one place its state is chosen.

    Three outcomes, in one place so the envelope's
    ``unknown <=> value is None <=> as_of is None`` triad holds by construction
    rather than by each branch remembering to honour it:

    * measured now — ``fresh``, and the last good is updated;
    * failed, with a last good inside the retention bound — ``stale``, carrying
      that value at ITS original instant so the consumer can see how old it is,
      and the producer's failure text verbatim as the reason. A last good
      stamped AFTER *now* is inside the bound too. The store is shared across
      renders, so a render with a later instant can refresh this root while
      this one is still waiting on it. That value is the newest evidence there
      is, and its age is negative only against this render's own instant. On
      the wire it precedes ``served_at``, which ``api_tasks`` resolves after
      its fan-out, so no retention bound applies to it;
    * failed with nothing to fall back on — ``unknown``, carrying the
      producer's failure VERBATIM and nothing else: there is no previous value
      and so nothing more to explain;
    * failed with a last good PAST the bound — ``unknown`` too, but the reason
      gains a clause naming the bound that disqualified it. A discarded value
      is a fact the consumer cannot otherwise account for — it saw that census
      a minute ago and needs to know why it is gone — whereas a value that
      never existed needs no such account. Verbatim is the default and the
      clause is the exception, so ``CensusVocabularyError``'s own contract
      ("carrying this message verbatim as its ``reason``") holds on the path
      that class actually travels.

    *name* identifies the half within a root, and is what keeps the rows' last
    good from ever being served as the census's.
    """
    if half.value is not None:
        _last_good.setdefault(project_root, {})[name] = _LastGood(half.value, now)
        return Datum(half.value, now, DatumState.FRESH, None, FRESHNESS_BOUND_SECONDS)

    reason = half.reason or 'read failed'
    previous = _last_good.get(project_root, {}).get(name)
    if previous is None:
        return Datum(
            None, None, DatumState.UNKNOWN, reason, FRESHNESS_BOUND_SECONDS,
        )
    age = (now - previous.as_of).total_seconds()
    if age > _RETENTION_BOUND_SECONDS:
        # Past the bound a last good stops being evidence about the present.
        # Serving it would put a day-old census behind a stale badge that
        # reads the same as a twenty-second-old one.
        return Datum(
            None, None, DatumState.UNKNOWN,
            f'{reason} (last good is {int(age)}s old, past the '
            f'{_RETENTION_BOUND_SECONDS}s retention bound)',
            FRESHNESS_BOUND_SECONDS,
        )
    return Datum(
        previous.value, previous.as_of, DatumState.STALE, reason,
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


def _strand_split(rows: Datum[list[dict]]) -> tuple[int | None, int | None]:
    """Partition the in-progress ROWS into live and stranded.

    The ROWS, not the census: the split needs ``claimant_run_id`` and
    ``heartbeat_at``, which only a row carries — a census counts a population
    and would have to invent them.

    Through ``tasks.task_is_stranded`` rather than ``shared.task_claimant``
    directly: that wrapper is the single dashboard-side strand predicate,
    binding ``STRANDED_HEARTBEAT_TTL`` and the request-scoped clock onto the
    shared function so the task-row badge and this split cannot disagree.

    Judged at ``rows.as_of`` — the instant the rows were MEASURED — not at the
    serving instant. A heartbeat beating when the rows were read was live then,
    and a stale half re-judged against a later clock would manufacture strands
    that never happened; for a last good near the retention bound, every claim
    would read as abandoned. The Datum's own ``as_of`` is what discloses the
    age, so the split does not have to lie about it as well.
    """
    if rows.value is None or rows.as_of is None:
        return None, None
    in_progress = [row for row in rows.value if row.get('status') == 'in-progress']
    stranded = sum(1 for row in in_progress if task_is_stranded(row, rows.as_of))
    return len(in_progress) - stranded, stranded


def _assemble(
    rows: Datum[list[dict]],
    map_half: _HalfRead[Mapping[int, str]],
    *,
    failure: SnapshotFailure,
    now: datetime,
    project_root: str,
) -> TaskSnapshot:
    """Turn a rows ``Datum`` and a map outcome into one stamped, contract-checked unit.

    THE one place a ``TaskSnapshot`` is built, so a unit standing in for a
    root this render never reached is assembled by exactly the code that
    assembles a measured one — including the census's last-good fallback,
    which is what lets a budget expiry still show the previous census, aged,
    rather than a bare unknown.

    The rows arrive as a finished ``Datum``, not as a half-outcome, because
    the two callers choose them differently: a read's rows go through
    :func:`_datum` like the census does, and :func:`unmeasured_snapshot`'s
    are unknown outright. *failure* is the rows half's kind, for the reason
    :class:`SnapshotFailure` gives.
    """
    census: Datum[TaskCensus] = _datum(
        _census_half(map_half), now=now, project_root=project_root, name='census',
    )
    live, stranded = _strand_split(rows)
    snapshot = TaskSnapshot(
        census=census,
        rows=rows,
        in_progress_live=live,
        in_progress_stranded=stranded,
        skew_seconds=_skew_seconds(census, rows),
        status_map=MappingProxyType(
            dict(map_half.value) if isinstance(map_half.value, Mapping) else {}
        ),
        failure=failure,
    )
    # Check the contract HERE, where the producer still has the context to
    # name what it built, rather than letting a break surface as a shaping
    # crash in the HTTP layer.
    validate_datum(snapshot.census, now)
    validate_datum(snapshot.rows, now)
    return snapshot


async def _read_unit(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | bytes | os.PathLike[str],
    *,
    now: datetime,
) -> TaskSnapshot:
    """Read both halves concurrently and assemble one stamped unit.

    *project_root* reaches the two reads EXACTLY as the caller spelled it —
    both accept ``str | bytes | PathLike`` — while the last-good store is
    keyed by its string form. Normalising before the read would hand a fake
    substrate a different type from the one production passes it, which is
    how a call-site regression passes its tests.
    """
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
    root = str(project_root)
    return _assemble(
        _datum(rows_half, now=now, project_root=root, name='rows'),
        map_half, failure=rows_half.failure, now=now, project_root=root,
    )


def unmeasured_snapshot(
    project_root: str | bytes | os.PathLike[str],
    *,
    now: datetime,
    reason: str,
    failure: SnapshotFailure,
) -> TaskSnapshot:
    """A unit for a root this render did not measure at all.

    A caller that bounds the WHOLE per-root share — ``collect_tasks_with_counts``
    does, on top of the per-operation bound applied here — can run out of
    budget before, or instead of, either read. Every configured root still owes
    the wire an entry, and this is it: the failure the caller can name, with
    the census routed through the same state machine a failed read takes, so a
    root whose budget expired still shows its last good census aged rather
    than a bare unknown.

    *failure* is the caller's own structured verdict — ``BUDGET`` for a
    deadline, ``UNREACHABLE`` for a share that raised — and never inferred
    from *reason*, which stays free to be prose.

    ITS ROWS DO NOT FALL BACK, unlike its census, and are ``unknown`` with
    *reason*. The unit a caller returns carries the rows that caller SHAPED
    (see :attr:`TaskSnapshot.rows`), and for a root it never measured it
    shaped none. So a last-good row list here could only reach the wire raw,
    or be swapped for an empty shaped list that claims a measured zero at the
    last good's instant. Shaping the last good instead would run outside the
    per-root guard, where a row that already broke shaping would take the
    whole render down with it. The census is a count and needs no shaping,
    so its last good is still honest evidence.
    """
    return _assemble(
        Datum(None, None, DatumState.UNKNOWN, reason, FRESHNESS_BOUND_SECONDS),
        _HalfRead(None, failure, reason),
        failure=failure, now=now, project_root=str(project_root),
    )


class SnapshotHealth(enum.StrEnum):
    """How a consumer must present one root, from its unit alone.

    The four members are the four banners ``/api/v2/dashboard/tasks`` can put
    a project under, and the codebase insists they stay distinct:

    * ``OK`` — both halves measured.
    * ``OFFLINE`` — the row read DEMONSTRABLY failed. Go look at the server.
    * ``DEGRADED`` — a budget expired first, so this root's state is simply
      UNKNOWN. Nothing was proven unreachable. Merging this into ``OFFLINE``
      tells an operator to restart a healthy service.
    * ``COUNT_UNKNOWN`` — the rows are current but the census is not. Without
      a name of its own such a root renders as healthy with a confident
      "0 done", which is the invisible failure the envelope exists to remove.
    """

    OK = 'ok'
    OFFLINE = 'offline'
    DEGRADED = 'degraded'
    COUNT_UNKNOWN = 'count_unknown'


def classify(snapshot: TaskSnapshot) -> SnapshotHealth:
    """Route one unit to its banner, from its kind and its states.

    THE single place the routing is decided, so the handler's project lists,
    ``collect_active_tasks``' offline labels and every test that asserts a
    partition all read the same rule. Keyed on the STRUCTURED failure kind and
    the census's state, never on ``reason`` — that field is by contract the
    producer's verbatim failure text, so keying on it would be an ad-hoc
    parser over a meaningful string and one reworded fan-out message would
    silently move a project between banners.
    """
    if snapshot.failure is SnapshotFailure.UNREACHABLE:
        return SnapshotHealth.OFFLINE
    if snapshot.failure is SnapshotFailure.BUDGET:
        return SnapshotHealth.DEGRADED
    if snapshot.census.state is not DatumState.FRESH:
        return SnapshotHealth.COUNT_UNKNOWN
    return SnapshotHealth.OK


def _aged(datum: Datum[T], served_at: datetime) -> Datum[T]:
    """*datum* as it reads at *served_at*: ``stale`` once past its freshness bound."""
    if datum.state is not DatumState.FRESH or datum.as_of is None:
        return datum
    age = (served_at - datum.as_of).total_seconds()
    if age <= datum.freshness_bound_seconds:
        return datum
    return replace(
        datum,
        state=DatumState.STALE,
        reason=(
            f'measured {int(age)}s before it was served, past the '
            f'{datum.freshness_bound_seconds}s freshness bound'
        ),
    )


def as_served(snapshot: TaskSnapshot, served_at: datetime) -> TaskSnapshot:
    """*snapshot* re-read at the instant a payload carrying it is served.

    A unit is stamped when it is MEASURED and shared across every render that
    reaches the cache within its TTL, so the age a consumer sees is the unit's
    time in the cache plus however long that consumer's own fan-out ran after
    reading it. Past :data:`FRESHNESS_BOUND_SECONDS` a half is honestly
    ``stale``: same value, same ``as_of``, and a reason naming its age. That
    is a fact about the serving instant, not a producer bug, so it must not
    reach ``validate_datum`` still claiming to be fresh.
    """
    return replace(
        snapshot,
        census=_aged(snapshot.census, served_at),
        rows=_aged(snapshot.rows, served_at),
    )


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
    async def _refresh() -> TaskSnapshot:
        return await _read_unit(client, config, project_root, now=now)

    return await _snapshot_cache.get_or_refresh(str(project_root), _refresh)


# ---------------------------------------------------------------------------
# The on-demand terminal window
# ---------------------------------------------------------------------------

_TERMINAL_FETCH_WINDOW = 400
"""How many terminal (done + cancelled) rows ONE ``?terminal=`` request reads.

A ceiling, and the reason one is needed: unbounded, this read pulls every done
row in the tree — ~4000 rows / ~40 MB on dark-factory. Even this window
measured 9.4 MB to READ across nine roots on 2026-09-18. The payload never
carried all of it: the default render shipped a capped slice, 2.2 MB on
2026-09-23. It used to be paid on EVERY render; now only the request that
asks for terminal rows pays it, and only up to here.

SELECTED BY DESCENDING TASK ID, which is the only ordering the substrate
offers: ``SqliteTaskBackend._get_tasks_internal`` is ``ORDER BY id`` and
``page_size``/``offset`` slice that ascending list, with no ``ORDER BY
updated_at`` anywhere. Tasks are filed and completed in roughly id order, so
the common case matches recency; the divergent case — a long-parked low-id
task completing late — is real, which is why this is 8x the 50-row cap the
retired default-render buckets used, why exceeding it WARNS, and why the
answer crosses the wire as ``LOWER_BOUND`` rather than as a plain list.
"""


def measured_terminal_total(snapshot: TaskSnapshot) -> int | None:
    """*snapshot*'s terminal population, or ``None`` if it did not measure one.

    The window's offset is ``n_terminal - window``, so this count is what makes
    the window POSITIONABLE. A non-fresh census has no count to give — and a
    fabricated zero would collapse the offset to 0, which slices the
    ascending-id list at its OLDEST end. ``None`` is the honest answer and
    :func:`acquire_terminal_window` is what refuses to guess past it.
    """
    census = snapshot.census
    if census.state is not DatumState.FRESH or census.value is None:
        return None
    return census.value.views[TaskView.TERMINAL]


async def acquire_terminal_window(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | bytes | os.PathLike[str],
    *,
    now: datetime,
    terminal_total: int | None,
) -> Datum[list[dict]]:
    """Read the newest :data:`_TERMINAL_FETCH_WINDOW` terminal rows, or say why not.

    Lives beside the unit because it is the same datum family read through the
    same access implementation, under the third slot of
    :data:`PER_PROJECT_MCP_CALLS` — the one the default render deliberately
    does not spend.

    Three outcomes, and the state is the whole disclosure:

    * positioned and read — ``LOWER_BOUND``, because rows outside the window
      were never fetched and the value is therefore known to under-report.
      ``reason`` names the window so a consumer can tell "all of them" from
      "the newest N". UNIFORMLY ``LOWER_BOUND``, even when the population fits:
      *terminal_total* comes from a DIFFERENT read, so a task completing
      between the two can leave a row outside a window that looked roomy, and
      a state that flipped on a count the consumer cannot see would make the
      rare truncation the one case nobody's code path had exercised;
    * *terminal_total* is ``None`` — ``UNKNOWN``. Without the census the offset
      would collapse to ``max(0, 0 - window) == 0``, and since
      ``page_size``/``offset`` slice an ASCENDING-id list that serves the
      OLDEST terminal rows, which a recency-ordered tab then presents as its
      newest. Showing months-old rows as the newest is a worse failure than
      showing none, so an unpositionable window is not fetched at all;
    * the read failed — ``UNKNOWN`` carrying the producer's failure verbatim.

    NO last-good fallback, unlike the unit: this window is acquired per
    request rather than on a refresh cycle, so there is no "previous value at
    its own instant" to age — only the value this request asked for and got,
    or did not.

    Benign race: *terminal_total* was measured by a different read, so a task
    completing between the two shifts the window by a row.
    """
    if terminal_total is None:
        return Datum(
            None, None, DatumState.UNKNOWN,
            'the terminal window cannot be positioned without a measured '
            'terminal count, and an unpositioned window serves the OLDEST '
            'rows rather than the newest',
            FRESHNESS_BOUND_SECONDS,
        )

    window = _TERMINAL_FETCH_WINDOW
    if terminal_total > window:
        logger.warning(
            'project root %s: %d terminal (done+cancelled) tasks exceed the '
            '%d-row fetch window — only the %d highest-id terminal rows are '
            'read, so a low-id task completed long after it was filed can be '
            'missing',
            project_root, terminal_total, window, window,
        )
    half = await _bounded(
        fetch_task_page(
            client, config, project_root,
            statuses=sorted(TERMINAL),
            page_size=window,
            # A COMPUTED offset, not a LIMIT: the slice is over ascending ids,
            # so reaching the high-id end is arithmetic rather than an option.
            offset=max(0, terminal_total - window),
            timeout=PER_CALL_TIMEOUT,
        ),
        label='terminal window',
    )
    if half.value is None:
        return Datum(
            None, None, DatumState.UNKNOWN, half.reason or 'read failed',
            FRESHNESS_BOUND_SECONDS,
        )
    return Datum(
        half.value, now, DatumState.LOWER_BOUND,
        (
            f'the newest {window} of {terminal_total} terminal rows; older '
            'ones were never read'
            if terminal_total > window
            else f'all {terminal_total} terminal rows, read under a '
                 f'{window}-row window'
        ),
        FRESHNESS_BOUND_SECONDS,
    )
