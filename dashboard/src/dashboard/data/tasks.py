"""Async fetchers for task state via fused-memory MCP HTTP endpoint.

Replaces the legacy ``.taskmaster/tasks/tasks.json`` readers after the
2026-05-02 SQLite cutover made fused-memory the sole owner of task state.

The dashboard's per-task wire shape is preserved here so consumers
(``active_tasks``, ``orchestrator``, ``burndown``, ``merge_queue``)
do not need to be re-keyed.

Network errors are caught and surfaced as ``{'offline': True, 'error': ...}``;
the caller turns that into a per-project skip plus a Tasks-tab banner.

Note: the three failover loops below raise ``ValueError`` from within their
``_call`` closures on a "soft failure" (malformed/errored MCP result), which
``mcp_fanout.first_success`` treats the same as a transport error — including
invalidating that URL's cached session. Previously a soft failure here fell
through with a bare ``continue`` and no session teardown; see
``mcp_fanout``'s module docstring for why this normalization is intentional.

Two of those three loops (``fetch_tasks``, ``fetch_statuses``) are
parameterized by ``project_root``, so they compose their ``log_label``
through ``mcp_fanout.fanout_label`` to keep each root's failure streak on its
own throttle key — one fused-memory URL serves every root, so a fixed literal
label would let a healthy root's success clear a broken root's streak and
re-arm its opening WARNING every poll cycle. ``fetch_external_statuses`` is
parameterized by a ``deps`` list rather than a root, so its fixed label is
already a correct single key.

Caching: ``fetch_tasks`` alone is cached, at a 20 s TTL. Its key is a
:class:`_TasksRead` RECORD — (project_root, statuses, mode) — rather than the
root alone. Four of its five callers need the whole tree while ``active_tasks``
narrows, so a root-only key would let one caller's status-filtered result be
served to the others for up to the TTL window. The record is also the single
source of the WIRE arguments (:meth:`_TasksRead.wire_arguments`), so the key
and the request it stands for cannot drift apart. A caller that must not ride
that TTL passes ``cached=False``.
``fetch_statuses`` and ``fetch_external_statuses`` are uncached and return
live data; ``fetch_statuses`` held a 5 s cache until task 5587, removed
because its one consumer now owns a longer TTL over the unit above it.
"""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx
from shared.task_claimant import is_stranded

from dashboard.config import DashboardConfig
from dashboard.data.mcp_fanout import TTLCache, fanout_label, first_success
from dashboard.data.memory import mcp_tool_call
from dashboard.data.utils import resolve_now

# ---------------------------------------------------------------------------
# Per-project_root TTL cache for fetch_tasks
# (mirrors app._load_task_cards / merge_queue.load_task_titles pattern)
#
# Code-duplication note: fetch_tasks's own copy of the {TTL constant, store,
# _clear() hook, store-only-on-success, list()-copy} pattern is now extracted
# into dashboard.data.mcp_fanout.TTLCache (this task).  app._task_cards_cache
# and merge_queue._task_titles_cache still implement the pattern inline —
# extracting a shared helper there would require changes to those modules,
# which fall outside this task's module lock.  Both caller caches are now
# primarily redundant for MCP de-duplication (the 20 s inner TTL handles it);
# they remain for legacy shaping-cost avoidance and are outside this task's
# scope to remove.
#
# Keyed by project_root_str — a small, bounded set in practice — which
# satisfies TTLCache's documented "bounded key space" assumption (it never
# evicts individual store/lock entries short of a blanket .clear()).
# ---------------------------------------------------------------------------

# Within the PRD's recommended 15-30 s staleness window.  Slightly longer than
# the 10 s caller caches (_TASK_CARDS_TTL_SECONDS / _TASK_TITLES_TTL_SECONDS)
# because fetch_tasks is the dominant full-tree seam — a monitoring view
# tolerates brief staleness; the inner TTL dominates net MCP cadence.
#
# Caller-cache stacking: app._load_task_cards (10 s) and
# merge_queue.load_task_titles (10 s) both cache fetch_tasks output on top of
# this inner cache.  Worst-case combined staleness ≈ caller TTL + inner TTL
# ≈ 10 s + 20 s = 30 s — at the PRD's upper bound; intentional for a
# monitoring view where brief staleness is preferable to MCP hammering.
DEFAULT_PER_CALL_TIMEOUT = 2.0
"""Per-HTTP-request budget for every ``fetch_tasks`` MCP call.

Deliberately shares its public name with
:data:`dashboard.data.task_runtime.DEFAULT_PER_CALL_TIMEOUT` so the idiom is
recognisable across probe callers, and is the single source of the
per-request term that ``active_tasks``'s whole-handler budget invariant
reads (it must never be restated as a literal there).

Strictly tighter than :func:`dashboard.data.memory.mcp_tool_call`'s own 10 s
default: this seam only ever narrows a budget, never widens one.
"""

COLD_SESSION_POSTS: tuple[str, ...] = (
    'initialize',
    'notifications/initialized',
    'tools/call',
)
"""The JSON-RPC posts a COLD MCP session performs against ONE URL.

The fact is :func:`dashboard.data.memory.mcp_tool_call`'s own — its docstring
states that a cold session performs these three posts, which is why its
*timeout* bounds each request and not the operation.

A named tuple rather than a literal ``3``, for the same reason
``task_snapshot.PER_PROJECT_MCP_CALLS`` is one: if the session handshake ever
gains a fourth post, the structural invariant in
``tests/test_fetch_tasks_whole_operation_budget.py`` fails a test instead of
silently overrunning in production.
"""

DEFAULT_WHOLE_OPERATION_BUDGET = 7.0
"""Whole-operation bound for ONE ``fetch_tasks`` call, enforced by callers.

Nothing in ``fetch_tasks`` applies this itself — it is the value a caller
hands to ``asyncio.wait_for``, which is the only construct that bounds this
operation as a whole (``timeout`` is per-HTTP-request; see the "Per-request
budget" note on :func:`fetch_tasks`).

Derivation: ``DEFAULT_PER_CALL_TIMEOUT`` (2.0) * ``len(COLD_SESSION_POSTS)``
(3) = 6.0 <= 7.0, leaving 1.0 s of slack so the bound is a real backstop for
non-MCP overhead (JSON decode, row shaping, event-loop scheduling) rather
than coinciding exactly with the sum of its parts. It deliberately equals
``active_tasks._TASKS_PER_PROJECT_BUDGET`` because both fall out of the same
arithmetic — the two are not coupled in code, so either may be tightened
independently.

**What that sum does and does NOT claim.** The 6.0 figure is PER URL.
``fetch_tasks``' ``_refresh`` delegates to
``first_success(config.fused_memory_urls, ...)``, and
``mcp_fanout.first_success`` walks its URLs strictly IN ORDER, falling
through to the next only after the current one fails — so an N-URL
deployment's cold worst case is ``N * 6.0``, not 6.0. As SHIPPED the fit is
exact: ``config.DEFAULT_FUSED_MEMORY_URLS`` is
``('http://localhost:8002',)``, exactly one URL. ``fused_memory_urls`` is
nevertheless operator-configurable (a comma-separated env var), and for that
case this budget deliberately CAPS the fan-out rather than accommodating it.
Capping exactly this residual is why the caller-side ``wait_for`` layer
exists: the two layers are complementary, not redundant — the same
disclosure ``active_tasks._TASKS_PER_PROJECT_BUDGET`` carries for its own
budget. Without this note, ``6.0 <= 7.0`` reads as a total-worst-case
guarantee it is not.
"""

# ---------------------------------------------------------------------------
# The fetch_tasks cache key: a structured record, not an encoded string
# ---------------------------------------------------------------------------
# These replace the hand-rolled ``_fetch_tasks_cache_key`` encoder (a ``*``
# sentinel for None, ``|`` field separators and a ``\x1f`` unit separator).
# That encoding produced two measured defects, and the shape below makes both
# UNREPRESENTABLE rather than merely fixed:
#
#   OFFSET DRIFT.  The encoder rendered ``|o={offset}`` unconditionally while
#   ``offset`` only reached the wire alongside ``page_size``, so two reads with
#   a byte-identical wire request minted two entries.  Here the only read that
#   HAS an offset is :class:`_OnePage`, and ``wire_arguments`` sends THAT
#   field rather than a window handed to it, so the bytes on the wire are the
#   bytes in the key by construction.
#
#   STATUSES ORDER.  ``['a','b']`` and ``['b','a']`` encoded differently while
#   naming one order-insensitive SQL ``IN`` list.  ``frozenset`` collapses them.
#
# The mode is a UNION rather than three flat fields because flat fields would
# permit `page_size` set with `offset` unset, and a read that is somehow both a
# page and a walk — invalid states that would then need runtime validation.
# No field carries a DEFAULT, so a future read mode that forgets to enter one
# is a pyright construction error AND a runtime TypeError, rather than a
# valid-but-wrong key that collides silently.


@dataclass(frozen=True, slots=True)
class _OnePage:
    """One explicit slice of the ascending-id task list: a PARTIAL answer."""

    page_size: int
    offset: int


@dataclass(frozen=True, slots=True)
class _CompleteRead:
    """The COMPLETE task set. *chunk_size* selects transport, not the contract.

    ``None`` means one unpaginated request; an int means walk the tree that
    many rows at a time.  Both yield the same set.

    *chunk_size* is nevertheless part of the cache key, and removing it is a
    SILENT PRODUCTION REGRESSION rather than a simplification.  The positive
    cache is chunk-INsensitive (both transports agree), but the NEGATIVE cache
    is chunk-SENSITIVE: ``burndown._fetch_snapshot_tasks`` probes UNPAGINATED
    and falls back to the chunked walk only when the probe returns the offline
    marker — which is exactly what an oversize tree produces.  Both caches
    share this one key (keying them separately is how they drift apart), so a
    chunk-insensitive key would make probe and fallback ONE key: the fallback
    would be suppressed by the probe's own failure and never reach the server,
    and precisely the large projects pagination exists to serve would write no
    snapshot row — a permanent hole in an append-only table with no backfill.
    """

    chunk_size: int | None


@dataclass(frozen=True, slots=True)
class _TasksRead:
    """One ``fetch_tasks`` read: simultaneously the cache key AND the wire args.

    Being both is the point.  The key and the request dict used to be built by
    two separate encoders that already disagreed about ``offset``; deriving
    both from this one record means they cannot disagree again.  For a PAGE
    read that is enforced rather than merely intended: :meth:`wire_arguments`
    reads the window off :attr:`mode` — the very field the key hashes — and
    REFUSES a second one, so no call site can hand the wire a window the key
    does not carry.

    The RETURN CONTRACT is the *mode*'s TYPE — ``type(read.mode)`` answers
    "one page or the whole set?" without parsing anything out of a string.
    """

    project_root: str
    statuses: frozenset[str] | None
    mode: _OnePage | _CompleteRead

    def wire_arguments(self, window: _OnePage | None) -> dict:
        """Build the MCP ``get_tasks`` arguments for this read.

        *window* exists for the WALK, which re-uses ONE record across every
        page and varies only the window — that is the whole reason this is a
        parameter at all.  A :class:`_OnePage` read has no such freedom: its
        window IS :attr:`mode`, so it is read from there and a caller-supplied
        one is a ``TypeError`` rather than a silent override.  Both halves
        matter.  Deriving it is what makes key/wire OFFSET DRIFT structurally
        impossible instead of a discipline the one page call site happens to
        keep; refusing the redundant argument is what stops a future edit
        (clamping an offset, retrying a short page) from reintroducing the
        disagreement quietly.

        ``statuses`` is guarded on ``is not None``, never truthiness:
        ``frozenset()`` is falsy but means "no tasks at all", the opposite of
        ``None``'s "whole tree", so a truthiness guard would silently widen it.
        It crosses the wire ``sorted()`` — a frozenset has no iteration order,
        and deterministic bytes beat nondeterministic ones for logs and mocks.
        """
        if isinstance(self.mode, _OnePage) and window is not None:
            raise TypeError(
                f'{self.mode!r} already fixes this read\'s window; passing '
                f'{window!r} as well is the key/wire disagreement this record '
                f'exists to make unrepresentable'
            )
        page = self.mode if isinstance(self.mode, _OnePage) else window
        arguments: dict = {'project_root': self.project_root}
        if self.statuses is not None:
            arguments['statuses'] = sorted(self.statuses)
        if page is not None:
            arguments['page_size'] = page.page_size
            arguments['offset'] = page.offset
        return arguments


_FETCH_TASKS_TTL_SECONDS = 20.0
_fetch_tasks_cache: TTLCache[list[dict] | dict, _TasksRead] = TTLCache(
    ttl_seconds=lambda: _FETCH_TASKS_TTL_SECONDS
)

# Negative (offline-marker) cache.  ``cache_ok`` on the positive cache stores
# successes ONLY, which made failure the expensive path: a healthy root rides
# the 20 s TTL while a broken one re-walks its whole tree on every UI poll.
#
# 5.0 s is picked against the two real clocks either side of it:
#   * SHORTER than _FETCH_TASKS_TTL_SECONDS (20 s), so an outage is re-probed
#     several times per positive-cache window and recovery is noticed quickly;
#   * LONGER than data.js's POLL_INTERVAL_MS (3 s), so a broken root costs at
#     most one tree-walk attempt per two polls instead of one per poll.
#
# A SECOND TTLCache instance rather than a change to TTLCache itself: the
# class carries exactly one TTL per instance and sits on eight other call
# sites, so parameterising it would be the larger and less obviously correct
# change.  No mcp_fanout change is required.
#
# The two caches CAN both hold a fresh entry for one key, so the read order
# between them is load-bearing.  The negative lookup sits outside the positive
# cache's per-key lock, and ``TTLCache`` documents that a ``cache_ok``-rejected
# value stores nothing and lets "the next lock-queued waiter run its own
# refresh in turn" — so with two concurrent callers for the same key (the
# routine case: app._load_task_cards, data.orchestrator, data.merge_queue and
# data.burndown all fetch the same unnarrowed key on the same poll) waiter A's
# failure can write a 5 s marker while waiter B's success writes a 20 s
# positive entry.  ``fetch_tasks`` therefore prefers a fresh POSITIVE entry
# over a fresh marker: a demonstrated success outranks a retry-suppression
# hint, and serving the marker there would put a false offline banner over
# rows that had already loaded.  Both caches are keyed by the IDENTICAL
# :class:`_TasksRead` record, so a recovered root also repopulates the positive
# cache on its next attempt.  Keying them differently is exactly how the two
# drift apart, which is why the record is built once per read and shared.
_FETCH_TASKS_NEGATIVE_TTL_SECONDS = 5.0
_fetch_tasks_negative_cache: TTLCache[dict, _TasksRead] = TTLCache(
    ttl_seconds=lambda: _FETCH_TASKS_NEGATIVE_TTL_SECONDS
)


# ---------------------------------------------------------------------------
# Safe page size for the fetch_statuses walk
# ---------------------------------------------------------------------------
#
# NO CACHE sits here, deliberately, and that is a change: fetch_statuses held
# a 5 s TTL cache until task 5587.  Its only consumer now is
# ``task_snapshot.acquire_snapshot``, which owns a 15 s TTL over the whole
# snapshot unit — a second, shorter TTL layered under that one is the
# duplicated staleness the one-datum-one-path PRD removes, and it would let
# the unit stamp an ``as_of`` newer than the map it is stamping.  The snapshot
# reads its rows uncached for the same reason; see ``fetch_tasks``' ``cached``
# parameter.

STATUSES_SAFE_PAGE_SIZE = 2000
"""How many statuses one ``get_statuses`` page may carry.

The server's own documented bound (``_STATUSES_AUTO_PAGE_LIMIT``), and a
measured one rather than a round number: 2 000 entries is ~46 KB against the
~62 KB documented-safe MCP tool-response envelope, while 3 000 is ~69 KB —
past the wall.  An oversized page is NOT clamped server-side; it is rejected
WHOLESALE, which is the incident paging exists to close.

Restated here rather than imported because the dashboard does not depend on
the fused-memory package — this is the wire contract's value, held at the
seam that speaks it.
"""


def _fetch_tasks_cache_clear() -> None:
    """Clear BOTH fetch_tasks TTL caches, positive and negative (test/admin hook)."""
    _fetch_tasks_cache.clear()
    _fetch_tasks_negative_cache.clear()


def _shape_task(task: dict) -> dict | None:
    """Trim an MCP get_tasks row to the dashboard's persistent shape.

    MCP returns top-level ids as strings and includes testStrategy/subtasks
    that the dashboard does not render. Cast id at the boundary; drop those.
    ``updatedAt`` is preserved as ``updated_at`` — it is the recency key for
    ordering done tasks and the ``completed`` display timestamp.

    ``claimant_run_id`` and ``heartbeat_at`` are carried through for the
    STRANDED projection (task 3543 / PRD ι): they are the two columns
    :func:`task_is_stranded` reads, and dropping them here is what previously
    made a strand invisible on every dashboard surface.  Both are read with
    ``.get`` so a pre-migration row (or an older fused-memory that does not
    emit them) surfaces ``None`` rather than raising — the shaped dict always
    carries the keys, so consumers never have to guard for their absence.

    Mutation warning: :func:`fetch_tasks` caches these dicts by reference and
    hands the same objects to every caller within the TTL window, so callers
    must NOT mutate ``claimant_run_id``/``heartbeat_at`` (or any other field)
    in place — build a fresh row instead.
    """
    raw_id = task.get('id')
    if raw_id is None:
        return None
    try:
        tid = int(raw_id)
    except (TypeError, ValueError):
        return None

    raw_deps = task.get('dependencies') or []
    deps: list[int] = []
    for d in raw_deps:
        try:
            deps.append(int(d))
        except (TypeError, ValueError):
            continue

    metadata = task.get('metadata')
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        'id': tid,
        'title': task.get('title') or '',
        'description': task.get('description') or '',
        'details': task.get('details') or '',
        'status': task.get('status'),
        'priority': task.get('priority'),
        'dependencies': deps,
        'metadata': metadata,
        'updated_at': task.get('updatedAt'),
        'claimant_run_id': task.get('claimant_run_id'),
        'heartbeat_at': task.get('heartbeat_at'),
    }


# ---------------------------------------------------------------------------
# Stranded-task projection (task 3543 / PRD ι, spec S8)
# ---------------------------------------------------------------------------

# Mirrors the orchestrator's ``harness._RECONCILE_HEARTBEAT_TTL`` (10 minutes):
# a claim whose heartbeat has not advanced within this window is treated as
# abandoned.  The dashboard deliberately does NOT import the orchestrator
# package — it is a separate deployable and the dashboard's dependency set is
# intentionally narrow — so the value is restated here.  If the orchestrator's
# TTL moves, this constant must move with it; the two are a documented pair,
# not an accident.
STRANDED_HEARTBEAT_TTL = timedelta(minutes=10)


@dataclass(frozen=True, slots=True)
class _Page:
    """One page as it came off the wire.

    *rows* are :func:`_shape_task`-shaped and may be SHORTER than *delivered*:
    ``_shape_task`` drops a row whose id is missing or non-integer. That gap is
    a NAMED FIELD rather than a comment because the walk's ``returned``
    cross-check and its offset advance must both use *delivered* — the count
    the server actually sent. Checking ``len(rows)`` instead would turn ONE
    unparseable task id into a whole-project offline marker, and would advance
    the walk short so the next page re-read rows already held.
    """

    rows: list[dict]
    delivered: int
    pagination: dict | None


async def _fetch_page(
    client: httpx.AsyncClient,
    url: str,
    read: _TasksRead,
    window: _OnePage | None,
    timeout: float,
) -> _Page:
    """Issue ONE ``get_tasks`` call against ONE url and shape what comes back.

    The pagination envelope is PRESERVED rather than discarded: the walk needs
    ``returned``/``total``, and this is the only layer that sees them.

    Raises ``ValueError`` for a structured MCP error — ``first_success``'s
    documented soft-failure signal, so the fan-out falls through to the next
    URL. The guard is ``'error' in result and 'tasks' not in result``
    specifically: an ``error`` alongside ``tasks`` is a partial-warning payload
    and must still be served.
    """
    result = await mcp_tool_call(
        client, url, 'get_tasks', read.wire_arguments(window), timeout=timeout,
    )
    if 'error' in result and 'tasks' not in result:
        raise ValueError(str(result.get('error')))
    raw = result.get('tasks') or []
    rows = [shaped for shaped in (_shape_task(task) for task in raw) if shaped is not None]
    meta = result.get('pagination')
    return _Page(rows, len(raw), meta if isinstance(meta, dict) else None)


async def _walk_pages(
    page_fn: Callable[[_OnePage], Awaitable[_Page]],
    project_root: str,
    chunk_size: int,
) -> list[dict]:
    """Assemble the COMPLETE task set from repeated *page_fn* calls.

    *page_fn* is injected and already bound to ONE url. That binding is
    load-bearing rather than stylistic: ``first_success`` tries urls in order,
    so a walk free to fan out mid-walk would assemble pages from DIFFERENT
    servers and silently invalidate the changed-``total`` coherence check below —
    pages from two different states of the world, with every counter still
    self-consistent. Hence no url/config parameter here.

    The walk starts at offset 0. A complete read starting mid-tree is not
    complete, and after task 5018 the cache key for a complete read carries no
    offset, so honouring one would put two different answers under one key.

    TRUNCATION IS LOUD. Every failure below raises ``ValueError`` and DISCARDS
    whatever rows were accumulated. ``fetch_tasks``' contract distinguishes only
    ``list`` (a complete success) from the ``{'offline': True}`` marker, so a
    truncated list is indistinguishable from a complete one at EVERY call site:
    ``collect_snapshot`` triages on ``isinstance(result, list)`` and would write
    a confident undercount into the APPEND-ONLY ``snapshots`` table. A plausible
    dip in an append-only chart is unfalsifiable after the fact, whereas a gap
    is visible. ``ValueError`` is ``first_success``'s soft-failure signal, so it
    tries the next URL and, on exhaustion, yields the marker
    ``collect_snapshot`` already skips on.
    """
    shaped: list[dict] = []
    walk_offset = 0
    pages = 0
    page_budget: int | None = None
    first_total: int | None = None
    while True:
        # Same record, different window — which is why wire_arguments takes a
        # window at all.  Only a `_CompleteRead` may supply one; a page read's
        # window is its own mode, and offering a second raises.
        # `statuses` therefore reaches EVERY page: the tool applies the status
        # filter BEFORE its in-memory slice, so `total` is the FILTERED count and
        # an unfiltered page would both over-read and desynchronise the walk's
        # terminator.
        page = await page_fn(_OnePage(chunk_size, walk_offset))
        shaped.extend(page.rows)
        pages += 1

        meta = page.pagination
        if meta is None:
            # An older fused-memory ignores page_size and answers with the
            # whole bare list.  There is no `total` to page against, so this
            # response IS the answer — take it and stop.  Looping blind
            # would either spin or re-request the same rows forever.
            break
        returned = meta.get('returned')
        total = meta.get('total')
        if not isinstance(returned, int) or not isinstance(total, int):
            # No usable counters: completeness is UNVERIFIABLE here, and an
            # unverifiable read must not be reported as a complete one.
            raise ValueError(
                f'get_tasks pagination for {project_root} truncated at '
                f'{len(shaped)} row(s), total={total!r}'
            )
        if returned != page.delivered:
            # The SELF-REPORTED counter is what the walk advances on, so it
            # must be cross-checked against what actually arrived — this is
            # the one remaining way a bounded-but-incomplete read could be
            # handed back as a plain list.  A server (or a proxy/serialiser
            # that clips a page) claiming returned=10 while shipping 4 rows
            # would otherwise skip 6 rows per page silently, terminate
            # normally at offset >= total, and hand collect_snapshot a
            # confident undercount to write into an append-only table.  The
            # mirror case (under-reporting) re-requests rows already held
            # and inflates the counts instead.  Neither is verifiable after
            # the fact, so refuse the read.
            #
            # `page.delivered`, NOT `len(page.rows)`: _shape_task drops a row
            # with an unparseable id, so a legitimately-shaped page can be
            # shorter than what arrived.
            raise ValueError(
                f'get_tasks pagination for {project_root} inconsistent '
                f'at offset {walk_offset}: server claims returned={returned} '
                f'but sent {page.delivered} row(s)'
            )
        if first_total is None:
            first_total = total
            # BOUND THE WALK.  `total` is the loop's only terminator, and it
            # is server-reported: a stale count, a bad merge of tag scopes,
            # or a tree being written concurrently makes ceil(total/P)
            # sequential round trips on the SAME httpx client the 2 s render
            # polls share — the PoolTimeout hazard the burndown size probe
            # exists to bound.  So derive one budget from the FIRST response
            # and refuse to exceed it.  The +2 covers the final partial page
            # plus one page of slack; a server that keeps handing back short
            # pages is misbehaving in exactly the way that amplifies the
            # walk, and is caught here rather than paid for.
            page_budget = math.ceil(total / max(chunk_size, 1)) + 2
        elif total != first_total:
            # A `total` that CHANGES mid-walk — in EITHER direction — means the
            # tree changed underneath the read: the pages in hand are from
            # different states of the world, so the assembled list is not a
            # coherent snapshot of either.
            #
            # The check is on inequality rather than growth because the
            # SHRINKING case is the worse of the two.  A growing `total`
            # un-bounds the budget derived above and so trips something; a
            # shrinking one terminates the loop EARLY on `walk_offset >= total`
            # with every counter still self-consistent, and hands the assembled
            # prefix back as a plain `list` — which `collect_snapshot` triages
            # on `isinstance(result, list)` and writes into the append-only
            # `snapshots` table as fact.  The server re-slices from the now
            # shorter list, so the pages after the change skip rows outright.
            raise ValueError(
                f'get_tasks pagination for {project_root} raced a write '
                f'at offset {walk_offset}: total changed from {first_total} to '
                f'{total} mid-walk'
            )
        if returned <= 0:
            # Guard the COMPLETE cases first.  An empty page with no rows
            # still owed (total <= 0, or offset already past total) is a
            # complete read of an empty or exhausted tree — it must keep
            # returning [], or every empty project becomes a permanent
            # burndown hole and loses its legitimate all-zero row.
            if total <= 0 or walk_offset >= total:
                break
            # The server claims rows remain but hands back an empty page, so
            # it cannot be paged past.  Detail lives in the exception rather
            # than a second WARNING for the same event.
            raise ValueError(
                f'get_tasks pagination for {project_root} truncated at '
                f'{len(shaped)} row(s) — empty page at offset {walk_offset}, '
                f'total={total}'
            )
        walk_offset += page.delivered
        if walk_offset >= total:
            break
        if page_budget is not None and pages >= page_budget:
            raise ValueError(
                f'get_tasks pagination for {project_root} exceeded its '
                f'{page_budget}-page budget at offset {walk_offset} '
                f'(total={total}, page_size={chunk_size}); refusing to keep '
                f'walking'
            )
    return shaped


@dataclass(frozen=True, slots=True)
class _StatusPage:
    """One ``get_statuses`` response: the slice it carried, and its envelope.

    ``pagination`` is None when the response carried no envelope, which the
    tool's contract defines as a COMPLETE answer rather than a missing field.
    """

    statuses: dict[int, str]
    pagination: dict | None


async def _walk_statuses(
    page_fn: Callable[[int], Awaitable[_StatusPage]],
    project_root: str,
) -> dict[int, str]:
    """Assemble the COMPLETE status map from repeated *page_fn* calls.

    *page_fn* is injected and already bound to ONE url, for the same reason
    :func:`_walk_pages` binds its own: ``first_success`` tries urls in order,
    and a walk free to fan out mid-walk would tile pages from two different
    states of the world.

    ADVANCE BY WHAT WAS SERVED.  ``get_statuses`` does not clamp an oversized
    ``page_size``; a server may answer with fewer entries than asked and says
    so in ``pagination['page_size']``.  Advancing by the REQUESTED size would
    skip the difference — trading a loud transport rejection for a silently
    incomplete census, which is the strictly worse bargain.

    TRUNCATION IS LOUD, as in :func:`_walk_pages`: every failure below raises
    ``ValueError`` and discards the pages in hand.  ``fetch_statuses``'
    contract distinguishes only a map (complete) from the offline marker, so a
    short map is indistinguishable from a full one at the call site, and
    ``census.build_census`` would tally it into a confident under-report of
    ``total`` — the uniform lie that module exists to refuse.  ``ValueError``
    is ``first_success``'s soft-failure signal, so the fan-out tries the next
    url and, on exhaustion, yields the marker.

    Unlike ``_walk_pages`` there is no changed-``total`` coherence check,
    because the terminator is different: this walk stops on the SERVER's
    ``has_more``, computed against the population at the instant of each page,
    where ``_walk_pages`` stops on a client-side ``offset >= total`` comparison
    that a shrinking total silently satisfies early.
    """
    merged: dict[int, str] = {}
    offset = 0
    pages = 0
    page_budget: int | None = None
    while True:
        page = await page_fn(offset)
        merged.update(page.statuses)
        pages += 1

        meta = page.pagination
        if meta is None:
            # No envelope: the response IS the whole map.  Looping blind here
            # would re-request the same entries forever.
            break

        has_more = meta.get('has_more')
        served = meta.get('page_size')
        total = meta.get('total')
        if not isinstance(has_more, bool) or not isinstance(served, int):
            # Without these two, completeness is UNVERIFIABLE — and an
            # unverifiable read must not be reported as a complete one.
            raise ValueError(
                f'get_statuses pagination for {project_root} is unverifiable at '
                f'offset {offset}: has_more={has_more!r}, page_size={served!r}'
            )
        if not has_more:
            break
        if served <= 0:
            # More entries are owed, but the page that would carry them cannot
            # advance the offset, so this server cannot be paged past.
            raise ValueError(
                f'get_statuses pagination for {project_root} cannot advance at '
                f'offset {offset}: more entries remain but page_size={served}'
            )

        if page_budget is None:
            # BOUND THE WALK.  ``has_more`` is the loop's only terminator and
            # it is server-reported, so a server that never lowers it would
            # spin here on the same httpx client the render polls share.
            # Derived from the FIRST page's SERVED size because that is the
            # rate the walk actually advances at; the +2 covers the final
            # partial page plus one page of slack.
            if not isinstance(total, int):
                raise ValueError(
                    f'get_statuses pagination for {project_root} reports no usable '
                    f'total at offset {offset}: total={total!r}'
                )
            page_budget = math.ceil(total / served) + 2
        if pages >= page_budget:
            raise ValueError(
                f'get_statuses pagination for {project_root} exceeded its '
                f'{page_budget}-page budget at offset {offset} '
                f'(total={total}); refusing to keep walking'
            )
        offset += served
    return merged


async def _cached_fanout(
    config: DashboardConfig,
    read: _TasksRead,
    strategy: Callable[[str], Awaitable[list[dict]]],
    label: str,
    *,
    cached: bool = True,
) -> list[dict] | dict:
    """Fan out *strategy* across the configured urls, through BOTH task caches.

    THE one place the fan-out / positive cache / negative cache / marker
    policy lives.  Every public read routes through it and they differ only in
    the *read* record they build and the *strategy* they bind — so there is no
    second copy of any of this to drift out of step (INV-5).

    *strategy* is invoked with ONE url at a time and is already bound to it, so
    whatever it does (a single page, or a whole pinned-url walk) stays below
    the cache: a per-page public call would mint one entry and one lock per
    page, and let a failed page write a 5 s marker served mid-walk.

    The bound *strategy* carries the HTTP client and the per-request timeout
    budget a read actually costs; this layer neither needs nor receives them.

    *label* NAMES THE CALLING READ, and is a parameter rather than a literal
    because :func:`mcp_fanout.first_success` keys its per-url failure streak on
    ``(log_label, url)``.  A shared literal would throttle both public reads as
    one stream and log either failure under the other's name — the operator
    debugging a page read would be told ``fetch_tasks`` had failed.

    *cached* gates the two cache LOOKUPS and nothing else.  The fan-out, the
    offline-marker policy and the copy isolation stay on the one path they
    already live on, so a caller opting out of staleness does not also opt out
    of everything else this layer decides.  A failed uncached read still
    RECORDS its marker: knowing a root is down is shared knowledge worth
    suppressing the next cached caller's retry with, whoever discovered it.
    """
    async def _refresh() -> list[dict] | dict:
        return await first_success(
            config.fused_memory_urls,
            strategy,
            log_label=fanout_label(label, read.project_root),
            offline_result=lambda errs: {'offline': True, 'error': '; '.join(errs)},
        )

    # A fresh negative entry short-circuits the attempt.  The marker is still
    # RETURNED, so degradation stays exactly as visible to the caller as it was
    # before — only the retry is suppressed.
    #
    # UNLESS a fresh positive entry also exists (see the negative-cache note
    # above: a concurrent failure+success pair leaves both fresh).  Then fall
    # through and serve the data: the marker exists to suppress a RETRY, not to
    # withhold a result already in hand, and reporting a root offline while
    # holding fresh rows for it is the false-banner failure this whole seam is
    # meant to avoid.  Falling through costs no MCP call — ``get_or_refresh``
    # returns the same fresh entry this check just saw.  If it expires in the
    # gap the worst case is one extra attempt, which is strictly better than a
    # wrong answer.
    if cached:
        suppressed = _fetch_tasks_negative_cache.get_fresh(read)
        if suppressed is not None and _fetch_tasks_cache.get_fresh(read) is None:
            return suppressed

    result = (
        await _fetch_tasks_cache.get_or_refresh(
            read, _refresh, cache_ok=lambda v: isinstance(v, list),
        )
        if cached else await _refresh()
    )
    if not isinstance(result, list):
        # Record the offline marker, under the SAME record the positive cache
        # is keyed by — keying the two differently is exactly how they drift.
        # ``get_or_refresh`` is the store path because ``TTLCache`` exposes no
        # bare setter and this needs no ``mcp_fanout`` change; the default
        # always-true ``cache_ok`` keeps it, and its per-key lock makes a
        # concurrent second failure reuse the first marker rather than race it.
        async def _mark() -> dict:
            return result

        await _fetch_tasks_negative_cache.get_or_refresh(read, _mark)
        return result
    # Shallow copy: list-level mutation by a caller is isolated from the cached
    # entry, the inner dicts are shared.  Both halves are documented on the
    # public reads.
    return list(result)


def task_is_stranded(task: Mapping[str, Any], now: datetime | None = None) -> bool:
    """Return True when *task* is an in-progress task with no live claimant.

    THE single dashboard-side strand predicate.  A thin wrapper binding
    :data:`STRANDED_HEARTBEAT_TTL` and the request-scoped clock onto
    :func:`shared.task_claimant.is_stranded` (Table C4 of the
    task-status-authority contract), so every dashboard surface that renders a
    strand — the task-row badge, the burndown live/stranded split — resolves
    it through one function and the surfaces cannot disagree (INV-5).

    ``is_stranded`` specifically, NOT its siblings:

    * ``has_live_claimant`` carries neither the ``status == 'in-progress'``
      gate nor the ``metadata.infra_hold`` carve-out, so
      ``not has_live_claimant(...)`` is a different — and, for this projection,
      wrong — predicate that would flag every pending/done task as stranded.
    * ``is_stranded_blocked`` is the blocked-status variant, out of scope here.

    Args:
        task: A dashboard-shaped task row (or a raw MCP row) — reads
            ``status``, ``claimant_run_id``, ``heartbeat_at``, ``metadata``.
        now: Request-scoped reference timestamp.  Resolved through
            :func:`dashboard.data.utils.resolve_now`, never a bare clock read.
            Callers doing a batch pass should resolve once and thread the
            concrete value through rather than passing None per row.

    Returns:
        True when the task is stranded, else False.
    """
    return is_stranded(task, resolve_now(now), STRANDED_HEARTBEAT_TTL)


async def fetch_task_page(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | bytes | os.PathLike[str],
    *,
    page_size: int,
    offset: int,
    statuses: list[str] | None = None,
    timeout: float = DEFAULT_PER_CALL_TIMEOUT,
) -> list[dict] | dict:
    """Fetch ONE explicit page of *project_root*'s task list via MCP.

    A PARTIAL answer, and the name says so.  Returns the ``list[dict]`` of
    rows in the window ``[offset, offset + page_size)`` of the ascending-``id``
    task list, or an offline marker ``{'offline': True, 'error': str}`` if
    every configured server fails.

    *page_size* and *offset* are both REQUIRED.  A page read with an implicit
    window is the defect this split exists to remove: defaulting either would
    let a caller ask for "a page" without saying WHICH page and silently get
    the first, which is how a page and a whole tree came to look alike
    (esc-4360-7).  For the complete set call :func:`fetch_tasks` — a different
    function because it is a different contract, and the two can no longer
    compute the same cache key.

    **Ascending id is the only ordering key.**  *page_size*/*offset* are a
    POST-FETCH in-memory slice in ``server/tools.py::get_tasks``, over a list
    already ordered by ascending ``id``.  They cut wire bytes but not backend
    work.  Two substrate gaps bound what any caller can do here and are
    recorded as fused-memory-side follow-up rather than faked client-side:
    ``get_tasks`` offers NO field/column projection (the backend is a
    hardcoded ``SELECT *`` feeding a fixed 14-key row, so the heavy
    ``description``/``details``/``testStrategy``/``metadata`` fields cannot be
    dropped), and NO ``ORDER BY updated_at`` (so "the N most recently updated"
    is not expressible server-side).  That second gap is why reaching the
    high-id end takes a COMPUTED *offset* rather than a ``LIMIT``.

    **The key space is NOT a small fixed set.**  ``active_tasks``' terminal
    window passes ``offset=max(0, n_terminal - window)``, computed from a live
    task count that grows every time a task completes, so a fresh key is
    minted on every completion.  Each retired key held a 400-row list — rows
    carrying description/details/metadata — plus an ``asyncio.Lock``, forever
    (task 3857 review).  Quantizing the offset does NOT fix this and was
    rejected: ``n_terminal`` grows monotonically, so quantized offsets do too
    — that slows the leak by the quantum, it does not bound it.
    :meth:`TTLCache._evict_expired` evicts entries past a multiple of the TTL,
    which bounds the resident set to "keys requested within the eviction
    horizon" no matter how many distinct keys are ever used.  That is the real
    invariant, it lives where the store lives, and it holds for every
    ``TTLCache`` caller.

    Caching, *statuses* narrowing, copy isolation, graceful degradation,
    negative caching and the per-request *timeout* budget behave IDENTICALLY
    here and are documented once on :func:`fetch_tasks`.  Both reads go
    through the single :func:`_cached_fanout` core, so there is deliberately
    no second copy of that policy — or of its description — to drift.
    """
    read = _TasksRead(
        str(project_root),
        None if statuses is None else frozenset(statuses),
        _OnePage(page_size, offset),
    )

    async def _call(url: str) -> list[dict]:
        """Read the whole answer from ONE url: for a page read, that is a page."""
        # No window is passed: `read.mode` IS this read's window, so the wire
        # request is derived from the cache key rather than from a second copy
        # of *page_size*/*offset* that could drift from it.
        #
        # The pagination envelope is deliberately DISCARDED. It bounds a WALK;
        # here the requested window IS the contract, and a short final page is
        # a correct answer rather than a truncation to detect.
        return (await _fetch_page(client, url, read, None, timeout)).rows

    return await _cached_fanout(config, read, _call, 'fetch_task_page')


async def fetch_tasks(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | bytes | os.PathLike[str],
    *,
    statuses: list[str] | None = None,
    chunk_size: int | None = None,
    timeout: float = DEFAULT_PER_CALL_TIMEOUT,
    cached: bool = True,
) -> list[dict] | dict:
    """Fetch the COMPLETE dashboard-shaped task set for *project_root* via MCP.

    ALWAYS the whole set (subject to *statuses*): there is no window, and no
    way to ask for one.  Returns a ``list[dict]`` on success, or an offline
    marker ``{'offline': True, 'error': str}`` if every configured server
    fails.  For one explicit page call :func:`fetch_task_page`.

    **Chunking (``chunk_size``) selects TRANSPORT, never the contract.**
    ``None`` issues one unpaginated request; an int WALKS the tree that many
    rows at a time and returns the assembled complete set.  Both yield the
    same rows.  It exists for the burndown collector, whose whole-tree read on
    a large project can exceed the MCP transport's response limit; that
    response is rejected wholesale, so the collector's cycle writes no row at
    all into an append-only history table — a permanent hole no later cycle
    backfills (the hole is logged; no backfill exists).

    It is spelled ``chunk_size`` and not ``page_size`` on purpose.  Since the
    public split this module holds BOTH meanings side by side — the slice
    :func:`fetch_task_page` takes, and the chunk this one walks in — and one
    spelling for two opposite meanings, distinguishable only by which function
    you are inside, is precisely the ambiguity that produced esc-4360-7.  The
    MCP WIRE argument is still ``page_size``; only the Python parameter differs.

    **``chunk_size`` is nevertheless PART OF THE CACHE KEY, and removing it is
    a silent production regression rather than a simplification.**  Why, in
    full, is stated once on :class:`_CompleteRead`, which carries the field —
    read it before "tidying" a transport detail out of a key.

    The RETURN CONTRACT is what discriminates this read from a page read, and
    it does so structurally: the cache key's ``mode`` is a
    :class:`_CompleteRead` rather than an :class:`_OnePage`, so a walk at
    ``chunk_size=10`` and a genuine first-page-of-10 cannot compute one key and
    be served to each other — a complete tree handed to a caller that asked for
    one page, or a 10-row page handed to the burndown collector as the whole
    tree.

    *statuses* and *timeout* are threaded into EVERY page request of a walk.
    For *statuses* that is required for coherence: ``server/tools.py::get_tasks``
    applies the status filter BEFORE its in-memory slice, so ``total`` under a
    filter is the FILTERED count and the walk's termination condition is only
    meaningful if every page carries the same filter.  For *timeout* it follows
    the per-request contract below — a walk multiplies that per-request budget
    by the page count, so a caller wanting a hard bound on the whole walk must
    size its own ``asyncio.wait_for`` accordingly.

    Chunking bounds the per-RESPONSE size ONLY.  It is not a way to do less
    work: MCP ``get_tasks`` has no field projection at any layer and the
    backend query is ``SELECT *``, so the same rows are read either way — they
    simply arrive in chunks the caller has sized to be likely to cross the
    wire.  It is a probability reduction, not a guarantee: no chunk size can be
    unconditionally safe because a SINGLE dense task row can exceed the
    transport envelope on its own.  Nor is it free — the server materialises
    the whole list and slices in memory per request, so a chunk size of P on an
    N-task project costs ``ceil(N/P)`` SEQUENTIAL round trips and O(N**2/P)
    server-side row builds.  The row-density measurement and the derivation of
    the only chunk size in the tree today live in one place:
    ``burndown._SNAPSHOT_PAGE_SIZE``.

    **A chunked read is all-or-nothing.**  If a page cannot be verified as
    complete, the partial rows are DISCARDED and the offline marker is returned
    instead.  Five ways a page fails that check: a non-int ``pagination``
    envelope; an empty page while the server still claims rows remain; a
    ``returned`` counter disagreeing with the number of rows actually delivered
    (the walk advances on that counter, so an unchecked one skips or re-reads
    rows silently); a ``total`` that CHANGES mid-walk in EITHER direction (the
    tree changed underneath the read, so the assembled pages are not one
    coherent snapshot — and a SHRINKING total ends the walk early with the
    prefix looking complete); and a walk exceeding the page budget derived from
    the first response's ``total``.  A
    truncated ``list`` would be indistinguishable from a complete one at every
    call site, and ``collect_snapshot`` would write that undercount into an
    append-only history table as fact.  An empty page with no rows still owed
    (``total <= 0``, or ``offset >= total``) is a complete read of an empty or
    exhausted tree and still returns ``[]`` — a true zero, not a truncation.

    **Caching.**  Results are cached under a :class:`_TasksRead` record —
    (project_root, statuses, mode) — for ``_FETCH_TASKS_TTL_SECONDS`` (~20 s)
    to avoid hammering the MCP server on every render.  Whatever a given
    narrowing returns is cached unchanged, and an unnarrowed entry and a
    narrowed entry for the same root are INDEPENDENT — which is what keeps a
    narrowed read from serving a status-filtered subset to the full-tree
    callers.  Offline/error markers never enter the POSITIVE cache, so a
    transient failure does not pin empty results for the TTL window.  The
    policy itself lives once, in :func:`_cached_fanout`; this section and
    the three below describe it for BOTH public reads.

    Pass ``cached=False`` to bypass both cache LOOKUPS — everything else on
    this path is unchanged, including the marker this read records on failure.
    ``task_snapshot.acquire_snapshot`` is the caller that needs it: it stamps
    the rows with an ``as_of`` at the present instant, and a cached read could
    hand it rows up to a full TTL older than that stamp claims.  It pays no
    duplicate-read cost for the bypass because it holds a longer TTL of its
    own over the whole unit.

    **Copy isolation (list-level only):** returns a shallow ``list()`` copy on
    every call, so list-level mutations (``result.clear()``, ``result.append()``)
    do not affect the cached entry.  Inner task dicts are shared references —
    mutating a field in place (e.g. ``result[0]['status'] = 'x'``) WILL corrupt
    the cached entry and other callers' views within the TTL window.  Current
    callers (active_tasks, shape_escalations) build fresh rows and do not mutate
    source dicts.  Switch to ``copy.deepcopy(cached[1])`` if element-level
    isolation becomes necessary.

    **Graceful degradation:** during an MCP outage that begins while a valid
    cache entry exists, callers receive the stale cached list (not the offline
    marker) for up to ``_FETCH_TASKS_TTL_SECONDS`` before the entry expires and
    a fresh attempt is made.  This delays outage detection by up to ~20 s —
    intentional for a monitoring view (stale data preferable to a blank tab).

    **Negative caching:** once that entry does expire and the attempt fails,
    the resulting offline marker is held for
    ``_FETCH_TASKS_NEGATIVE_TTL_SECONDS`` (~5 s) under the SAME key, so a
    broken root stops being the expensive path.  The marker is still returned
    to every caller in the window — the retry is suppressed, the degradation
    signal is not — and the negative entry is per (project_root, narrowing,
    mode), so one failing read never blinds a healthy sibling root or a
    differently narrowed read of the same root.  A fresh POSITIVE entry
    outranks the marker, so a success that raced a failure is served rather
    than shadowed; the marker still suppresses the retry either way.

    **Data consistency:** a caller combining a CACHED task tree (this
    function) with a status map in the same render may observe transiently
    inconsistent rows for up to ~20 s — a task listed as in-progress in the
    tree but already done per the map.  ``fetch_statuses`` no longer caches
    (task 5587), so the skew runs one way only: the map is always the fresher
    half.  The caller that must not carry that skew at all is
    ``task_snapshot.acquire_snapshot``, which passes ``cached=False`` and
    measures both halves live under its own TTL; its ``skew_seconds`` reports
    whatever gap remains rather than hiding it.

    **Server-side narrowing.**  *statuses* is forwarded to the ``get_tasks``
    MCP tool and is added to the arguments dict only when actually requested,
    so a caller that narrows nothing sends a dict byte-identical to the
    pre-narrowing shape — the four full-tree callers
    (``app._load_task_cards``, ``data.orchestrator``, ``data.merge_queue``,
    ``data.burndown``) are unaffected.  It is a REAL server-side row filter —
    it becomes ``WHERE tag = ? AND status IN (...)`` in SQL, so narrowing with
    it cuts backend work, not just wire bytes.  ``None`` (the default) means
    "no filter"; an EMPTY LIST is a valid, distinct "return nothing" request
    and is therefore SENT rather than dropped — which is why every guard on
    this path tests ``is not None`` and never truthiness, ``frozenset()``
    being falsy.  A bare string is rejected server-side with a
    ``ValidationError``.  Because the ``IN`` list is order-insensitive,
    *statuses* is held as a ``frozenset`` and crosses the wire ``sorted()``:
    two calls differing only in order are ONE read, and the wire bytes stay
    deterministic.

    **Per-request budget.** *timeout* is threaded into
    :func:`dashboard.data.memory.mcp_tool_call`, whose docstring is the
    authority: it is a PER-HTTP-REQUEST budget bounding connect/read/write and
    pool acquisition, NOT a whole-operation bound. A cold session performs
    three posts (``initialize``, ``notifications/initialized``,
    ``tools/call``), so the worst case here is roughly ``3 * timeout`` plus
    the server's think time — and that is before the fan-out tries a second
    URL. A caller needing a hard bound must still wrap this in
    ``asyncio.wait_for``; every route caller now does, and the two layers are
    complementary rather than redundant.
    ``active_tasks.collect_tasks_with_counts`` was first, with its own
    per-project budget; ``merge_queue.load_task_titles`` and
    ``app._load_task_cards`` follow it, each binding a named module constant
    to :data:`DEFAULT_WHOLE_OPERATION_BUDGET`. ``discover_orchestrators`` used
    to be a fourth and is no longer a caller at all — task 5587 removed its
    task fetch, which is the one resolution a budget cannot beat.

    EVERY caller is now whole-operation bounded, including the background
    ones (task 4884 / #4424 closed the last gap). ``burndown.collect_snapshot``
    binds ``burndown._SNAPSHOT_PER_ROOT_BUDGET`` around each root's read rather
    than :data:`DEFAULT_WHOLE_OPERATION_BUDGET`, and that discrepancy is
    DELIBERATE — do not "fix" it. Its read can fall back to
    ``paginate=True``, ONE call that internally walks ``ceil(N/page_size)``
    SEQUENTIAL round trips (measured ~209 s for one root of this repo's size),
    so a 7.0 s bound would time out every big root on every cycle and hole an
    APPEND-ONLY chart that no later cycle backfills. It shares this
    convention's SHAPE — a named module constant, ``asyncio.wait_for``, expiry
    surfacing as a handled per-root exception — and differs only in its value;
    the derivation is on that constant.
    """
    read = _TasksRead(
        str(project_root),
        None if statuses is None else frozenset(statuses),
        _CompleteRead(chunk_size),
    )

    async def _call(url: str) -> list[dict]:
        """Read the whole answer from ONE url, pinned for the duration."""
        page_fn = functools.partial(_fetch_page, client, url, read, timeout=timeout)

        if chunk_size is not None:
            # The walk binds a SINGLE url per first_success attempt — the
            # coherence checks assume one server — and sits BELOW the cache, so
            # a failed page cannot mint a per-page marker served mid-walk.
            # Both constraints now hold BY CONSTRUCTION: `_walk_pages` takes no
            # url/config and is reachable only from inside a bound strategy.
            return await _walk_pages(page_fn, read.project_root, chunk_size)

        # One unpaginated request: no chunk to walk, and the whole tree is the
        # answer.  The envelope, if any, is irrelevant — nothing is being paged.
        return (await page_fn(None)).rows

    return await _cached_fanout(config, read, _call, 'fetch_tasks', cached=cached)


async def fetch_external_statuses(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    deps: list[str],
) -> dict[str, str | bool]:
    """Fetch a ``{dep_id: status}`` map for a list of external dep strings via MCP.

    Calls ``get_external_statuses`` which returns a BARE ``{dep: status}`` map
    (NOT wrapped in a ``'statuses'`` key, unlike ``get_statuses``).

    Short-circuits to ``{}`` when *deps* is empty (no MCP call issued).
    Returns ``{}`` on any network/parse failure (fail-safe: leaves entries at
    the ``'unknown'`` sentinel rather than fabricating statuses or crashing).
    """
    if not deps:
        return {}

    async def _call(url: str) -> dict[str, str | bool]:
        result = await mcp_tool_call(
            client, url, 'get_external_statuses', {'deps': deps},
        )
        if not isinstance(result, dict):
            raise ValueError(f'unexpected result type {type(result).__name__}')
        if 'error' in result or not result:
            # Structured error dict or empty result (e.g. parse failure) — try next URL.
            raise ValueError(str(result.get('error', 'empty result')))
        return result  # bare {dep: status} map

    return await first_success(
        config.fused_memory_urls,
        _call,
        log_label='fetch_external_statuses',
        offline_result=lambda errs: {'offline': True, 'error': '; '.join(errs)},
    )


async def fetch_statuses(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | bytes | os.PathLike[str],
    *,
    timeout: float = DEFAULT_PER_CALL_TIMEOUT,
) -> Mapping[Any, Any]:
    """Fetch a compact ``{int(id): status}`` map for *project_root* via MCP.

    ~95% smaller than ``fetch_tasks``, so it is the right seam for a
    status-only caller.  The sole consumer is
    ``task_snapshot.acquire_snapshot``, which tallies the map into a
    ``TaskCensus`` and hands ``active_tasks`` the raw map as ``_resolve_deps``'
    bounded fallback for a dependency outside the fetched rows.

    NOTE: the burndown collector used to be the headline consumer and is NOT
    one any more — ``burndown.collect_snapshot`` moved to ``fetch_tasks`` in
    task 3543 because this compact map carries no claimant columns, so the
    live/stranded split is physically underivable from it.

    Returns ``{'offline': True, 'error': str}`` if every server fails, or if
    any page of the walk does — a partial map is never served as a complete
    one (see :func:`_walk_statuses`).

    **Paging.** The map is assembled by walking
    :data:`STATUSES_SAFE_PAGE_SIZE`-sized pages, because an unpaginated read
    of a large population is rejected wholesale by the MCP transport.  Each
    page is one HTTP request; the WALK is one operation, and the caller bounds
    it as one — see ``task_snapshot.PER_PROJECT_MCP_CALLS``.

    **No cache.** Reads are live.  The consumer above owns a 15 s TTL over the
    whole snapshot unit, and a second TTL under it would let that unit serve a
    map older than the ``as_of`` it stamps.  The returned mapping is freshly
    built by the walk, so a caller may mutate it freely.

    **Per-request budget.** *timeout* bounds ONE page request and is threaded
    into :func:`dashboard.data.memory.mcp_tool_call` exactly as ``fetch_tasks``
    threads its own, defaulting to the same shared
    ``DEFAULT_PER_CALL_TIMEOUT``.  Left on ``mcp_tool_call``'s 10 s default a
    single page could alone exceed the per-project budget its caller's
    arithmetic claims to bound.  It is deliberately NOT a whole-walk bound:
    that belongs to the caller, which wraps the walk in one
    ``asyncio.wait_for`` (``task_snapshot``, design decision 2), and the two
    layers are complementary rather than redundant.
    """
    project_root_str = str(project_root)

    async def _page(url: str, offset: int) -> _StatusPage:
        """Read ONE page from *url*, shaped and with its envelope kept."""
        result = await mcp_tool_call(
            client, url, 'get_statuses',
            {
                'project_root': project_root_str,
                'page_size': STATUSES_SAFE_PAGE_SIZE,
                'offset': offset,
            },
            timeout=timeout,
        )
        if 'error' in result and 'statuses' not in result:
            raise ValueError(str(result.get('error')))

        raw = result.get('statuses') or {}
        out: dict[int, str] = {}
        for raw_id, status in raw.items():
            try:
                out[int(raw_id)] = status
            except (TypeError, ValueError):
                continue
        meta = result.get('pagination')
        return _StatusPage(out, meta if isinstance(meta, dict) else None)

    async def _call(url: str) -> dict[int, str]:
        """Read the whole map from ONE url, pinned for the duration of the walk."""
        return await _walk_statuses(
            functools.partial(_page, url), project_root_str,
        )

    return await first_success(
        config.fused_memory_urls,
        _call,
        log_label=fanout_label('fetch_statuses', project_root_str),
        offline_result=lambda errs: {'offline': True, 'error': '; '.join(errs)},
    )
