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

Caching: ``fetch_tasks`` and ``fetch_statuses`` are both cached, at
deliberately different TTLs (20 s and 5 s). ``fetch_tasks``'s key is the pair
**(project_root, narrowing)** rather than the root alone — see
:func:`_fetch_tasks_cache_key`. Four of its five callers need the whole tree
while ``active_tasks`` narrows, so a root-only key would let one caller's
status-filtered result be served to the others for up to the TTL window.
``fetch_statuses`` takes no narrowing arguments, so the root alone IS its
whole key. ``fetch_external_statuses`` is uncached and returns live data.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
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

_FETCH_TASKS_TTL_SECONDS = 20.0
_fetch_tasks_cache: TTLCache[list[dict] | dict] = TTLCache(
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
# rows that had already loaded.  Both caches share
# :func:`_fetch_tasks_cache_key`, so a recovered root also repopulates the
# positive cache on its next attempt.
_FETCH_TASKS_NEGATIVE_TTL_SECONDS = 5.0
_fetch_tasks_negative_cache: TTLCache[dict] = TTLCache(
    ttl_seconds=lambda: _FETCH_TASKS_NEGATIVE_TTL_SECONDS
)


# ---------------------------------------------------------------------------
# Per-project_root TTL cache for fetch_statuses
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS.  ``fetch_statuses`` was the one per-project MCP call in the
# post-narrowing design with no cache, and ``active_tasks._shape_one_project``
# issues it UNCONDITIONALLY (it is ``_resolve_deps``' only bounded fallback,
# not merely done_count's source).  Because BOTH ``/api/v2/dashboard/tasks``
# and ``/api/v2/dashboard/scheduler`` route through
# ``collect_tasks_with_counts``, and ``data.js`` polls both every 3 s, that
# meant two full-population ``get_statuses`` reads per root per poll — on a
# nine-root config ~360 uncached status reads a minute.  The narrowing cut
# wire BYTES dramatically and would have raised backend QUERY count, which
# cuts against this change's own goal of bounding dashboard->MCP cost.
#
# 5.0 s is picked against the same two clocks as the negative cache above:
#   * STRICTLY SHORTER than _FETCH_TASKS_TTL_SECONDS (20 s), so the status map
#     is always the fresher half of any row+map pair (done_count can be newer
#     than the rows, never staler) — see fetch_tasks' "Data consistency" note;
#   * LONGER than data.js's POLL_INTERVAL_MS (3 s), so the two endpoints'
#     duplicate read within one poll collapses to one call, and consecutive
#     polls of one endpoint collapse too.
#
# NO negative cache, deliberately, unlike fetch_tasks: a failed get_statuses
# costs one bounded DEFAULT_PER_CALL_TIMEOUT per URL rather than a full
# tree-walk, so failure is not the expensive path here; and its offline marker
# is what puts a root in TASKS_COUNT_UNKNOWN_PROJECTS, a user-visible
# degradation that should clear on the first poll after recovery rather than
# up to 5 s later.
_FETCH_STATUSES_TTL_SECONDS = 5.0
_fetch_statuses_cache: TTLCache[dict] = TTLCache(
    ttl_seconds=lambda: _FETCH_STATUSES_TTL_SECONDS
)


def _fetch_tasks_cache_clear() -> None:
    """Clear BOTH fetch_tasks TTL caches, positive and negative (test/admin hook)."""
    _fetch_tasks_cache.clear()
    _fetch_tasks_negative_cache.clear()


def _fetch_statuses_cache_clear() -> None:
    """Clear the fetch_statuses TTL cache (test/admin hook).

    Deliberately NOT folded into :func:`_fetch_tasks_cache_clear`: that hook's
    contract ("both fetch_tasks caches") is asserted by name in the suite, and
    a caller clearing one seam should not silently reach into the other.
    """
    _fetch_statuses_cache.clear()


def _fetch_tasks_cache_key(
    project_root_str: str,
    statuses: list[str] | None,
    page_size: int | None,
    offset: int,
) -> str:
    """Compose the ``_fetch_tasks_cache`` key for one (root, narrowing) pair.

    The key must cover the narrowing arguments, not just the root: only ONE
    of ``fetch_tasks``' five callers narrows, so a root-only key would let
    ``active_tasks``' status-filtered entry be served to the four full-tree
    callers (``app._load_task_cards``, ``data.orchestrator``,
    ``data.merge_queue``, ``data.burndown``) — silently truncating them for
    up to the TTL window, and doing so non-deterministically depending on
    which caller raced in first.

    ``statuses=None`` renders as ``*`` and is therefore distinct from
    ``statuses=[]``, which renders as the empty string: the tool treats them
    as opposite requests (whole tree vs no tasks at all), so collapsing them
    onto one key would serve an empty list as if it were the full tree.

    A ``\x1f`` (ASCII unit separator) delimits the statuses so a status
    string containing the field separators cannot forge another key.
    """
    statuses_part = '*' if statuses is None else '\x1f'.join(statuses)
    page_part = '*' if page_size is None else str(page_size)
    return f'{project_root_str}|s={statuses_part}|p={page_part}|o={offset}'


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


async def fetch_tasks(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: str | bytes | os.PathLike[str],
    *,
    statuses: list[str] | None = None,
    page_size: int | None = None,
    offset: int = 0,
    timeout: float = DEFAULT_PER_CALL_TIMEOUT,
) -> list[dict] | dict:
    """Fetch the dashboard-shaped task list for *project_root* via MCP.

    Returns a ``list[dict]`` on success, or an offline marker
    ``{'offline': True, 'error': str}`` if every configured server fails.

    Results are cached per **(project_root, narrowing)** — see
    :func:`_fetch_tasks_cache_key` — for ``_FETCH_TASKS_TTL_SECONDS`` (~20 s)
    to avoid hammering the MCP server on every render.  Whatever a given
    narrowing returns is cached unchanged, and an unnarrowed entry and a
    narrowed entry for the same root are INDEPENDENT — which is what keeps a
    narrowed read from serving a status-filtered subset to the full-tree
    callers.  Offline/error markers are never cached so a transient failure
    does not pin empty results for the TTL window.

    The key space is NOT a small fixed set, and this docstring previously
    claimed it was.  ``active_tasks``' terminal-window call passes
    ``offset=max(0, n_terminal - window)``, computed from a live task count
    that grows every time a task completes, so a fresh key is minted on every
    completion (``...|p=400|o=3601``, then ``o=3602``, ...).  Each retired key
    held a 400-row list — rows carrying description/details/metadata — plus an
    ``asyncio.Lock``, forever (task 3857 review).

    Quantizing the offset does NOT fix this and was rejected: ``n_terminal``
    grows monotonically, so quantized offsets do too — that slows the leak by
    the quantum, it does not bound it.  ``TTLCache`` now evicts entries past
    a multiple of the TTL (see :meth:`TTLCache._evict_expired`), which bounds
    the resident set to "keys requested within the eviction horizon" no matter
    how many distinct keys are ever used.  That is the real invariant, it lives
    where the store lives, and it holds for every ``TTLCache`` caller.

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
    signal is not — and the negative entry is per (project_root, narrowing),
    so one failing read never blinds a healthy sibling root or a differently
    narrowed read of the same root.  A fresh POSITIVE entry outranks the
    marker, so a success that raced a failure is served rather than shadowed;
    the marker still suppresses the retry either way.

    **Data consistency:** ``fetch_statuses`` is cached too, but at a much
    shorter TTL (``_FETCH_STATUSES_TTL_SECONDS``, ~5 s), so callers that
    combine a cached task tree (this function) with a status map in the same
    render may observe transiently inconsistent rows for up to ~20 s (e.g. a
    task listed as in-progress in the tree but already done per the status
    map).  The status map is the FRESHER of the two by construction — its TTL
    is strictly shorter — so the skew never runs the other way.  The
    pre-existing 10 s caller caches had this property at a narrower window;
    the 20 s inner cache widens it uniformly across all callers.

    **Server-side narrowing.** *statuses*, *page_size* and *offset* are
    forwarded to the ``get_tasks`` MCP tool. Each is added to the arguments
    dict only when actually requested, so a caller that narrows nothing sends
    a dict byte-identical to the pre-narrowing shape — the four full-tree
    callers (``app._load_task_cards``, ``data.orchestrator``,
    ``data.merge_queue``, ``data.burndown``) are unaffected.

    What the substrate does with each, established by tracing
    ``server/tools.py::get_tasks`` → ``TaskInterceptor.get_tasks`` →
    ``SqliteTaskBackend._get_tasks_internal`` for task 3857:

    * *statuses* is a REAL server-side row filter — it becomes
      ``WHERE tag = ? AND status IN (...)`` in SQL, so narrowing with it cuts
      backend work, not just wire bytes. ``None`` (the default) means "no
      filter"; an EMPTY LIST is a valid, distinct "return nothing" request and
      is therefore sent rather than dropped. A bare string is rejected
      server-side with a ``ValidationError``.
    * *page_size*/*offset* are a POST-FETCH in-memory slice in the tool body,
      over a list already ordered by ascending ``id``. They cut wire bytes but
      not backend work, and ascending id is their only ordering key — which is
      why reaching the high-id end requires a computed *offset* rather than a
      ``LIMIT``. *offset* is meaningless on its own and is omitted from the
      wire unless *page_size* is set.

    Two substrate gaps bound what any caller here can do, and are recorded as
    fused-memory-side follow-up rather than faked client-side: ``get_tasks``
    offers NO field/column projection (the backend is a hardcoded
    ``SELECT *`` feeding a fixed 14-key row, so the heavy
    ``description``/``details``/``testStrategy``/``metadata`` fields cannot be
    dropped from the dashboard), and NO ``ORDER BY updated_at`` (so "the N
    most recently updated" is not expressible server-side).

    **Per-request budget.** *timeout* is threaded into
    :func:`dashboard.data.memory.mcp_tool_call`, whose docstring is the
    authority: it is a PER-HTTP-REQUEST budget bounding connect/read/write and
    pool acquisition, NOT a whole-operation bound. A cold session performs
    three posts (``initialize``, ``notifications/initialized``,
    ``tools/call``), so the worst case here is roughly ``3 * timeout`` plus
    the server's think time — and that is before the fan-out tries a second
    URL. A caller needing a hard bound must still wrap this in
    ``asyncio.wait_for``; ``active_tasks.collect_tasks_with_counts`` does, and
    the two layers are complementary rather than redundant.
    """
    project_root_str = str(project_root)

    arguments: dict = {'project_root': project_root_str}
    if statuses is not None:
        arguments['statuses'] = statuses
    if page_size is not None:
        arguments['page_size'] = page_size
        arguments['offset'] = offset

    async def _call(url: str) -> list[dict]:
        result = await mcp_tool_call(
            client, url, 'get_tasks', arguments, timeout=timeout,
        )
        if 'error' in result and 'tasks' not in result:
            raise ValueError(str(result.get('error')))

        raw_tasks = result.get('tasks') or []
        shaped: list[dict] = []
        for task in raw_tasks:
            row = _shape_task(task)
            if row is not None:
                shaped.append(row)
        return shaped

    async def _refresh() -> list[dict] | dict:
        return await first_success(
            config.fused_memory_urls,
            _call,
            log_label=fanout_label('fetch_tasks', project_root_str),
            offline_result=lambda errs: {'offline': True, 'error': '; '.join(errs)},
        )

    key = _fetch_tasks_cache_key(project_root_str, statuses, page_size, offset)

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
    suppressed = _fetch_tasks_negative_cache.get_fresh(key)
    if suppressed is not None and _fetch_tasks_cache.get_fresh(key) is None:
        return suppressed

    result = await _fetch_tasks_cache.get_or_refresh(
        key, _refresh, cache_ok=lambda v: isinstance(v, list),
    )
    if not isinstance(result, list):
        # Record the offline marker.  ``get_or_refresh`` is the store path
        # because ``TTLCache`` exposes no bare setter and this needs no
        # ``mcp_fanout`` change; the default always-true ``cache_ok`` keeps it,
        # and its per-key lock makes a concurrent second failure reuse the
        # first marker rather than race it.
        async def _mark() -> dict:
            return result

        await _fetch_tasks_negative_cache.get_or_refresh(key, _mark)
        return result
    return list(result)


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
    status-only caller.  The sole consumer today is
    ``active_tasks.collect_done_counts``, which needs nothing but a per-status
    count.

    NOTE: the burndown collector used to be the headline consumer and is NOT
    one any more — ``burndown.collect_snapshot`` moved to ``fetch_tasks`` in
    task 3543 because this compact map carries no claimant columns, so the
    live/stranded split is physically underivable from it.  (The mirror of
    this note lives on ``collect_done_counts``; keep the two in step.)

    Returns ``{'offline': True, 'error': str}`` if every server fails.

    **Caching.** Successful reads are held for
    ``_FETCH_STATUSES_TTL_SECONDS`` (~5 s) in a per-``project_root`` TTL cache
    — see that constant for why this call, of all of them, needs one and why
    5 s is the number.  The key is the root ALONE: unlike ``fetch_tasks`` this
    function takes no narrowing arguments, so there is no second dimension to
    collapse.  *timeout* is deliberately NOT part of the key — it is a
    per-request budget, not a narrowing argument, so two callers passing
    different budgets are asking for the identical map and either may serve
    the other.  Offline markers are NOT cached (``cache_ok``), so a broken
    root is re-probed on every poll and recovery is noticed immediately.  The
    returned mapping is a shallow COPY, so a caller mutating it cannot poison
    the entry for the next one.

    **Per-request budget.** *timeout* is threaded into
    :func:`dashboard.data.memory.mcp_tool_call` exactly as ``fetch_tasks``
    threads its own, and shares ``fetch_tasks``'s default so the two agree by
    construction.  This is not decoration: ``get_statuses`` is one of the
    three calls enumerated in ``active_tasks._PER_PROJECT_MCP_CALLS``, whose
    budget invariant (``DEFAULT_PER_CALL_TIMEOUT * 3 <=
    _TASKS_PER_PROJECT_BUDGET``) is only a true statement about the shipped
    system if every enumerated call actually carries the term.  Left on
    ``mcp_tool_call``'s 10 s default, this one call could alone exceed the
    per-project budget the arithmetic claims to bound.
    """
    project_root_str = str(project_root)

    async def _call(url: str) -> dict[int, str]:
        result = await mcp_tool_call(
            client, url, 'get_statuses', {'project_root': project_root_str},
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
        return out

    async def _refresh() -> dict:
        return await first_success(
            config.fused_memory_urls,
            _call,
            log_label=fanout_label('fetch_statuses', project_root_str),
            offline_result=lambda errs: {'offline': True, 'error': '; '.join(errs)},
        )

    result = await _fetch_statuses_cache.get_or_refresh(
        project_root_str,
        _refresh,
        cache_ok=lambda v: isinstance(v, dict) and not v.get('offline'),
    )
    return dict(result) if isinstance(result, dict) else result
