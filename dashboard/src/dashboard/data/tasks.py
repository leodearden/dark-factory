"""Async fetchers for task state via fused-memory MCP HTTP endpoint.

Replaces the legacy ``.taskmaster/tasks/tasks.json`` readers after the
2026-05-02 SQLite cutover made fused-memory the sole owner of task state.

The dashboard's per-task wire shape is preserved here so consumers
(``active_tasks``, ``orchestrator``, ``burndown``, ``merge_queue``)
do not need to be re-keyed.

Network errors are caught and surfaced as ``{'offline': True, 'error': ...}``;
the caller turns that into a per-project skip plus a Tasks-tab banner.

Note: the four failover loops below raise ``ValueError`` from within their
``_call`` closures on a "soft failure" (malformed/errored MCP result), which
``mcp_fanout.first_success`` treats the same as a transport error — including
invalidating that URL's cached session. Previously a soft failure here fell
through with a bare ``continue`` and no session teardown; see
``mcp_fanout``'s module docstring for why this normalization is intentional.
``fetch_task_prose`` RETURNS one structured error instead of raising it: a
missing task is a definitive answer, not a soft failure.

Three of those four loops (``fetch_tasks``, ``fetch_statuses``,
``fetch_task_prose``) are parameterized by ``project_root``, so they compose
their ``log_label`` through ``mcp_fanout.fanout_label`` to keep each root's
failure streak on its own throttle key — one fused-memory URL serves every
root, so a fixed literal label would let a healthy root's success clear a
broken root's streak and re-arm its opening WARNING every poll cycle.
``fetch_external_statuses`` is parameterized by a ``deps`` list rather than a
root, so its fixed label is already a correct single key.

Caching: none. Every read here is live; staleness is owned by
``task_snapshot``'s unit TTL, the one cache a task datum has
(one-datum-one-path PRD decision 20). ``fetch_task``'s caller owns its cache
(``task_lookup``), because only the caller knows which answers may be held.
"""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar

import httpx
from shared.task_claimant import DEFAULT_CLAIMANT_HEARTBEAT_TTL, is_stranded

from dashboard.config import DashboardConfig
from dashboard.data.mcp_fanout import fanout_label, first_success
from dashboard.data.memory import mcp_tool_call
from dashboard.data.utils import resolve_now

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
``fetch_tasks`` delegates (through ``_fanout_read``) to
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
# The task read record: structured, not an encoded string
# ---------------------------------------------------------------------------
# A read's MCP wire arguments are derived from one record, so no second copy
# of a window can disagree with the read it stands for.  The only read that
# HAS an offset is :class:`_OnePage`, and ``wire_arguments`` sends THAT field
# rather than a window handed to it.
#
# The mode is a UNION rather than three flat fields because flat fields would
# permit `page_size` set with `offset` unset, and a read that is somehow both a
# page and a walk — invalid states that would then need runtime validation.
# No field carries a DEFAULT, so a future read mode that forgets to enter one
# is a pyright construction error AND a runtime TypeError.


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
    """

    chunk_size: int | None


@dataclass(frozen=True, slots=True)
class _TasksRead:
    """One task read: the record its MCP wire arguments are derived from.

    The request dict is derived from this record, never assembled beside it,
    so a read and the request it sends cannot disagree about ``offset``.
    For a PAGE read that is enforced
    rather than merely intended: :meth:`wire_arguments` reads the window off
    :attr:`mode` and REFUSES a second one, so no call site can hand the wire a
    window the read does not name.

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
        matter.  Deriving it is what makes read/wire OFFSET DRIFT structurally
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
                f'{window!r} as well is the read/wire disagreement this record '
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


# ---------------------------------------------------------------------------
# Safe page size for the fetch_statuses walk
# ---------------------------------------------------------------------------
#
# NO CACHE sits here, deliberately.  The only consumer,
# ``task_snapshot.acquire_snapshot``, owns a 15 s TTL over the whole snapshot
# unit — a second, shorter TTL layered under that one is the duplicated
# staleness the one-datum-one-path PRD removes, and it would let the unit
# stamp an ``as_of`` newer than the map it is stamping.  The row reads are
# uncached for the same reason.

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


def _shape_task(task: dict) -> dict | None:
    """Trim an MCP get_tasks row to the dashboard's persistent shape.

    MCP returns top-level ids as strings and includes testStrategy/subtasks
    that the dashboard does not render. Cast id at the boundary; drop those.
    ``updatedAt`` is preserved as ``updated_at`` — it is the recency key for
    ordering done tasks and the ``completed`` display timestamp.

    ``description`` stays because ``redux_api.shape_escalations`` embeds this
    whole dict as each escalation row's ``task`` and the Escalations drawer
    renders it; ``details`` is dropped because only the Task Detail pane
    renders it, and that pane reads it through :func:`fetch_task_prose`.

    ``claimant_run_id`` and ``heartbeat_at`` are carried through for the
    STRANDED projection (task 3543 / PRD ι): they are the two columns
    :func:`task_is_stranded` reads, and dropping them here is what previously
    made a strand invisible on every dashboard surface.  Both are read with
    ``.get`` so a pre-migration row (or an older fused-memory that does not
    emit them) surfaces ``None`` rather than raising — the shaped dict always
    carries the keys, so consumers never have to guard for their absence.

    Mutation warning: ``task_snapshot``'s unit cache holds these dicts by
    reference and hands the same objects to every reader within its TTL, so
    callers must NOT mutate ``claimant_run_id``/``heartbeat_at`` (or any other
    field) in place — build a fresh row instead.
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
        'status': task.get('status'),
        'priority': task.get('priority'),
        'dependencies': deps,
        'metadata': metadata,
        'updated_at': task.get('updatedAt'),
        'claimant_run_id': task.get('claimant_run_id'),
        'heartbeat_at': task.get('heartbeat_at'),
    }


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
    complete, which is why a :class:`_CompleteRead` carries no offset to
    honour.

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

    *statuses* may be SHORTER than *delivered*: an entry whose id is not an
    integer is dropped. The walk's ``returned`` cross-check must count
    *delivered*, what the server actually sent, for the reason
    :class:`_Page` gives: counting the parsed entries would turn one
    unparseable id into an offline marker for the whole project.
    """

    statuses: dict[int, str]
    delivered: int
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

    ADVANCE BY ``returned``, THE COUNT SERVED.  The envelope comes from
    ``fused_memory/server/tools.py::_pagination_meta``.  It passes the
    caller's REQUESTED ``page_size`` through verbatim, sets ``returned`` to
    the number of entries it actually sliced, and derives
    ``has_more = offset + returned < total``.  That identity is why
    ``offset + returned`` is the only advance that tiles the population
    without gaps: a walk that stops on the server's ``has_more`` has to step
    exactly as the server counted.  Advancing by the requested size would skip
    the difference whenever a server serves fewer entries than asked, which
    produces a silently incomplete census.

    ``get_statuses``' own docstring says the opposite: "Advance by
    ``pagination['page_size']`` (what was actually served)".  Its
    implementation does not do that.  Do not switch this walk back to
    ``page_size`` because of that docstring.  :func:`_walk_pages` reads
    ``returned`` off the same envelope, so both walkers key on one field.

    ``returned`` is cross-checked against the entries the page actually
    delivered, because it is the counter the walk advances on.  A page
    clipped in flight that still claims its full count would skip the clipped
    entries and terminate normally.  A page that under-reports would re-read
    entries already held.  Neither is detectable afterwards, so the read is
    refused.

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
        returned = meta.get('returned')
        total = meta.get('total')
        if not isinstance(has_more, bool) or not isinstance(returned, int):
            # Without these two, completeness is UNVERIFIABLE — and an
            # unverifiable read must not be reported as a complete one.
            raise ValueError(
                f'get_statuses pagination for {project_root} is unverifiable at '
                f'offset {offset}: has_more={has_more!r}, returned={returned!r}'
            )
        if returned != page.delivered:
            raise ValueError(
                f'get_statuses pagination for {project_root} inconsistent at '
                f'offset {offset}: server claims returned={returned} but sent '
                f'{page.delivered} status(es)'
            )
        if not has_more:
            break
        if returned <= 0:
            # More entries are owed, but the page that would carry them cannot
            # advance the offset, so this server cannot be paged past.
            raise ValueError(
                f'get_statuses pagination for {project_root} cannot advance at '
                f'offset {offset}: more entries remain but returned={returned}'
            )

        if page_budget is None:
            # BOUND THE WALK.  ``has_more`` is the loop's only terminator and
            # it is server-reported, so a server that never lowers it would
            # spin here on the same httpx client the render polls share.
            # Derived from the FIRST page's ``returned`` because that is the
            # rate the walk actually advances at; the +2 covers the final
            # partial page plus one page of slack.
            if not isinstance(total, int):
                raise ValueError(
                    f'get_statuses pagination for {project_root} reports no usable '
                    f'total at offset {offset}: total={total!r}'
                )
            page_budget = math.ceil(total / returned) + 2
        if pages >= page_budget:
            raise ValueError(
                f'get_statuses pagination for {project_root} exceeded its '
                f'{page_budget}-page budget at offset {offset} '
                f'(total={total}); refusing to keep walking'
            )
        offset += returned
    return merged




async def _fanout_read(
    config: DashboardConfig,
    read: _TasksRead,
    strategy: Callable[[str], Awaitable[list[dict]]],
    label: str,
) -> list[dict] | dict:
    """Fan *strategy* out across the configured urls: rows, or the offline marker.

    THE one place the fan-out and offline-marker policy of the two row reads
    lives.  Both public reads route through it and differ only in the *read*
    record they build and the *strategy* they bind, so there is no second copy
    of this to drift out of step (INV-5).  Nothing is stored: every call
    reaches the substrate, and a failure is returned to the caller that met it
    and remembered nowhere, so the very next call retries.

    *strategy* is invoked with ONE url at a time and is already bound to it, so
    whatever it does (a single page, or a whole pinned-url walk) is one attempt
    against one server.  It carries the HTTP client and the per-request timeout
    budget a read actually costs; this layer neither needs nor receives them.

    *label* NAMES THE CALLING READ, and is a parameter rather than a literal
    because :func:`mcp_fanout.first_success` keys its per-url failure streak on
    ``(log_label, url)``.  A shared literal would throttle both public reads as
    one stream and log either failure under the other's name — the operator
    debugging a page read would be told ``fetch_tasks`` had failed.
    """
    return await first_success(
        config.fused_memory_urls,
        strategy,
        log_label=fanout_label(label, read.project_root),
        offline_result=lambda errs: {'offline': True, 'error': '; '.join(errs)},
    )


def task_is_stranded(task: Mapping[str, Any], now: datetime | None = None) -> bool:
    """Return True when *task* is an in-progress task with no live claimant.

    THE single dashboard-side strand predicate.  A thin wrapper binding
    :data:`shared.task_claimant.DEFAULT_CLAIMANT_HEARTBEAT_TTL` and the
    request-scoped clock onto :func:`shared.task_claimant.is_stranded` (Table
    C4 of the task-status-authority contract), so every dashboard surface
    that renders a strand — the task-row badge, the burndown live/stranded
    split — resolves it through one function and the surfaces cannot
    disagree (INV-5).

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
    return is_stranded(task, resolve_now(now), DEFAULT_CLAIMANT_HEARTBEAT_TTL)


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
    function because it is a different contract, carried by a different read
    mode.

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

    *statuses* narrowing, the offline marker, the absence of any cache and the
    per-request *timeout* budget behave IDENTICALLY here and are documented
    once on :func:`fetch_tasks`.  Both reads go through the single
    :func:`_fanout_read` core, so there is deliberately no second copy of that
    policy — or of its description — to drift.
    """
    read = _TasksRead(
        str(project_root),
        None if statuses is None else frozenset(statuses),
        _OnePage(page_size, offset),
    )

    async def _call(url: str) -> list[dict]:
        """Read the whole answer from ONE url: for a page read, that is a page."""
        # No window is passed: `read.mode` IS this read's window, so the wire
        # request is derived from the read record rather than from a second
        # copy of *page_size*/*offset* that could drift from it.
        #
        # The pagination envelope is deliberately DISCARDED. It bounds a WALK;
        # here the requested window IS the contract, and a short final page is
        # a correct answer rather than a truncation to detect.
        return (await _fetch_page(client, url, read, None, timeout)).rows

    return await _fanout_read(config, read, _call, 'fetch_task_page')


async def fetch_tasks(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | bytes | os.PathLike[str],
    *,
    statuses: list[str] | None = None,
    chunk_size: int | None = None,
    timeout: float = DEFAULT_PER_CALL_TIMEOUT,
) -> list[dict] | dict:
    """Fetch the COMPLETE dashboard-shaped task set for *project_root* via MCP.

    ALWAYS the whole set (subject to *statuses*): there is no window, and no
    way to ask for one.  Returns a ``list[dict]`` on success, or an offline
    marker ``{'offline': True, 'error': str}`` if every configured server
    fails.  For one explicit page call :func:`fetch_task_page`.

    **Chunking (``chunk_size``) selects TRANSPORT, never the contract.**
    ``None`` issues one unpaginated request; an int WALKS the tree that many
    rows at a time and returns the assembled complete set.  Both yield the
    same rows.  It was built for the burndown collector, whose whole-tree read
    on a large project can exceed the MCP transport's response limit (that
    response is rejected wholesale).  The collector now reads through
    ``task_snapshot.acquire_snapshot`` instead, and no production caller passes
    *chunk_size* today.

    It is spelled ``chunk_size`` and not ``page_size`` on purpose.  Since the
    public split this module holds BOTH meanings side by side — the slice
    :func:`fetch_task_page` takes, and the chunk this one walks in — and one
    spelling for two opposite meanings, distinguishable only by which function
    you are inside, is precisely the ambiguity that produced esc-4360-7.  The
    MCP WIRE argument is still ``page_size``; only the Python parameter differs.

    The RETURN CONTRACT is what discriminates this read from a page read, and
    it does so structurally: this read's record carries a
    :class:`_CompleteRead` mode, never an :class:`_OnePage`, so a walk at
    ``chunk_size=10`` and a genuine first-page-of-10 are different reads with
    different answers rather than one read with two meanings.

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
    server-side row builds.

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
    call site, and a caller writing history would record that undercount as
    fact.  An empty page with no rows still owed
    (``total <= 0``, or ``offset >= total``) is a complete read of an empty or
    exhausted tree and still returns ``[]`` — a true zero, not a truncation.

    **No cache.**  Every call reaches the substrate; staleness is owned by
    ``task_snapshot``'s unit TTL, the one cache a task datum has (PRD decision
    20).  An offline marker is returned to the caller that met the failure and
    stored nowhere, so the very next call retries.  Every call shapes its rows
    afresh, so no caller shares a list with another through this read.  A
    caller pairing this read with :func:`fetch_statuses` measures both halves
    live, and ``task_snapshot``'s ``skew_seconds`` reports the gap between
    them.

    **Server-side narrowing.**  *statuses* is forwarded to the ``get_tasks``
    MCP tool and is added to the arguments dict only when actually requested,
    so a caller that narrows nothing sends a dict byte-identical to the
    pre-narrowing shape — the one full-tree caller,
    ``app._fanout_probe_completion``, is unaffected.  It is a REAL server-side
    row filter — it becomes ``WHERE tag = ? AND status IN (...)`` in SQL, so
    narrowing with it cuts backend work, not just wire bytes.  ``None`` (the
    default) means "no filter"; an EMPTY LIST is a valid, distinct "return
    nothing" request
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
    ``asyncio.wait_for``, and the two layers are complementary rather than
    redundant. ``task_snapshot._bounded`` does so around each of the snapshot
    unit's reads. The one caller deliberately left unbounded is
    ``app._fanout_probe_completion``: the /healthz probe keeps its read
    outstanding so a wedge stays observable, and
    ``app._MCP_PROBE_OUTSTANDING_LIMIT`` turns that read's age into a verdict.
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
            # coherence checks assume one server.  That holds BY CONSTRUCTION:
            # `_walk_pages` takes no url/config and is reachable only from
            # inside a bound strategy.
            return await _walk_pages(page_fn, read.project_root, chunk_size)

        # One unpaginated request: no chunk to walk, and the whole tree is the
        # answer.  The envelope, if any, is irrelevant — nothing is being paged.
        return (await page_fn(None)).rows

    return await _fanout_read(config, read, _call, 'fetch_tasks')


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
        return _StatusPage(out, len(raw), meta if isinstance(meta, dict) else None)

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


@dataclass(frozen=True, slots=True)
class TaskProse:
    """One task's description and details, as the Task Detail pane renders them."""

    description: str
    details: str

    def to_wire(self) -> dict[str, str]:
        """The pair's JSON spelling; the one place it is written down."""
        return {'description': self.description, 'details': self.details}


@dataclass(frozen=True, slots=True)
class TaskNotFound:
    """fused-memory answered definitively: the root holds no task with that id."""

    detail: str


@dataclass(frozen=True, slots=True)
class TaskReadOffline:
    """No fused-memory URL answered; *detail* names each URL's failure."""

    detail: str


TaskProseRead = TaskProse | TaskNotFound | TaskReadOffline
TaskRowRead = dict | TaskNotFound | TaskReadOffline

_Projected = TypeVar('_Projected')


async def _read_task(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | os.PathLike[str],
    task_id: int,
    *,
    timeout: float,
    log_name: str,
    project: Callable[[dict], _Projected],
) -> _Projected | TaskNotFound | TaskReadOffline:
    """Read ONE task through fused-memory's ``get_task``, projected by *project*.

    The single place the ``get_task`` wire answer is classified:

    - fused-memory's ``TaskNotFoundError``, matched on its structured
      ``error_type``, is RETURNED as :class:`TaskNotFound` from the per-URL
      call, not raised: ``first_success`` would treat a raised ``ValueError``
      as a soft failure and ask the next URL, reporting an absent task as an
      outage;
    - any other tool error, or an empty result, raises ``ValueError`` — a soft
      failure, so the next URL is asked; *project* may raise it too;
    - every URL failing is :class:`TaskReadOffline` naming each failure.

    *timeout* is per HTTP request; the caller bounds the whole read.
    """
    root = str(project_root)

    async def _call(url: str) -> _Projected | TaskNotFound:
        result = await mcp_tool_call(
            client, url, 'get_task', {'id': str(task_id), 'project_root': root},
            timeout=timeout,
        )
        if result.get('error_type') == 'TaskNotFoundError':
            return TaskNotFound(str(result['error']))
        if 'error' in result or not result:
            raise ValueError(str(result.get('error', 'empty result')))
        return project(result)

    return await first_success(
        config.fused_memory_urls,
        _call,
        log_label=fanout_label(log_name, root),
        offline_result=lambda errs: TaskReadOffline('; '.join(errs)),
    )


def _prose_of(result: dict) -> TaskProse:
    """Missing prose normalises to ``''``, as :func:`_shape_task` does for ``description``."""
    return TaskProse(result.get('description') or '', result.get('details') or '')


def _row_of(result: dict) -> dict:
    """The :func:`_shape_task` row; an id it cannot parse is a soft failure."""
    row = _shape_task(result)
    if row is None:
        raise ValueError(f'get_task answered with an unparseable id {result.get("id")!r}')
    return row


async def fetch_task_prose(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | os.PathLike[str],
    task_id: int,
    *,
    timeout: float = DEFAULT_PER_CALL_TIMEOUT,
) -> TaskProseRead:
    """Read ONE task's description/details for the Tasks tab's Task Detail pane.

    The ACTIVE_TASKS rows omit both fields; the pane fetches them here for the
    selected task only. Each outcome is its own type because the route answers
    each with a different status: :class:`TaskProse`, :class:`TaskNotFound`
    or :class:`TaskReadOffline`, classified by :func:`_read_task`.

    *timeout* is per HTTP request; the caller bounds the whole read.
    Uncached, deliberately: a primary-key lookup fetched only on selection.
    """
    return await _read_task(
        client, config, project_root, task_id,
        timeout=timeout, log_name='fetch_task_prose', project=_prose_of,
    )


async def fetch_task(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | os.PathLike[str],
    task_id: int,
    *,
    timeout: float = DEFAULT_PER_CALL_TIMEOUT,
) -> TaskRowRead:
    """Read ONE task's dashboard row — the :func:`_shape_task` row ``fetch_tasks`` serves.

    The per-id read for a caller that needs a few tasks, not the tree:
    :class:`TaskNotFound` and :class:`TaskReadOffline` are classified by
    :func:`_read_task`. *timeout* is per HTTP request; the caller bounds the
    whole read. Uncached here; its caller owns the cache.
    """
    return await _read_task(
        client, config, project_root, task_id,
        timeout=timeout, log_name='fetch_task', project=_row_of,
    )
