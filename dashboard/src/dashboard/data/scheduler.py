"""Scheduler state aggregator for the Dark Factory dashboard.

Collects scheduler state from fused-memory MCP tools and active-task lock sets
to produce a unified view of contention, parks, overrides, and event sparklines.

Shape contract (returned by collect_scheduler_state):
    rows: list[dict]   — one per pending/active task, fields:
        task_id         str    raw numeric task id (snapshot key)
        title           str
        declared_priority  str | None  from task metadata
        effective_priority str | None  from snapshot.effective_priorities
        priority_differs   bool        True when effective != declared
        skip_count         int         from snapshot.skip_counts
        park_state         dict | None from snapshot.parks[task_id]
        age_seconds        int         seconds since park install or started*60
        lock_set           list[str]   normalized module locks (depth from snapshot)
        pinned             bool        from snapshot.overrides
        reserve_now        bool        from snapshot.overrides
        boost_tier         str | None  from snapshot.overrides
        ttl_until          str | None  from snapshot.overrides
        project            str | None  display label for the owning project
        project_root       str | None  absolute path to the owning project root

    modules: list[dict]  — one per unique module path that appears in any lock set:
        path        str
        holder      str | None   task_id of in-flight holder, or None
        contention  int          count of tasks whose lock_set includes this path
        sorted by contention desc, ties broken alphabetically by path

    pin_queue: list[dict]  — from snapshot.pin_queue: [{task_id, order}, ...]

    events_by_task: dict[str, dict]  — {task_id: {labels: [...], values: [...]}}
        sparklines from get_scheduler_events; empty dict when no history

    offline_projects: list[str]  — project labels that failed to respond

    paused_projects: list[dict]  — projects whose scheduler is park-stop paused.
        Each entry: {'project': str, 'reason': str | None}
        Empty list when no project is paused or snapshots lack the is_paused key
        (safe degradation for pre-upgrade on-disk snapshots).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta

import httpx
from shared.locking import files_to_modules, modules_conflict

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import _all_project_roots, _project_label, collect_active_tasks
from dashboard.data.memory import invalidate_session, mcp_tool_call

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Short-TTL cache for get_scheduler_snapshot
#
# The frontend polls /scheduler every 3 s (data.js:143/151) unconditionally,
# regardless of the active tab.  Without a cache, every poll re-runs the full
# N-project collection.  A TTL >= the poll interval keeps the cache warm so
# switching to the Scheduler tab renders instantly from already-populated
# DF_DATA.SCHEDULER.  The 5 s window allows a few seconds of staleness, which
# the task explicitly accepts.
#
# _scheduler_cache_clear() is a test hook that resets both the cache tuple and
# the asyncio.Lock so pytest-asyncio's function-scoped event loops don't hit
# "got Future attached to a different loop" on the second cache test.
# ---------------------------------------------------------------------------

_SCHEDULER_TTL_SECONDS: float = 5.0
# Explicit event-history bound passed to the get_scheduler_events MCP tool.
# The MCP tool defaults to limit=200, but spelling it out here hardens the
# reader→producer contract so the dashboard stays bounded if the MCP default
# ever changes (mirrors the producer→reader hardening from task-1188).
# 200 matches the current effective cap so no sparkline data is lost.
_SCHEDULER_EVENTS_LIMIT: int = 200
# Cache entry: (monotonic_ts: float, six_tuple: tuple, snapshot_at: str) | None
_scheduler_cache: tuple[float, tuple, str] | None = None
# Single-flight lock: only one cold-cache caller runs the collection;
# waiters re-check the cache after acquiring the lock and return early if
# another waiter already filled it (double-checked locking pattern).
_scheduler_refresh_lock: asyncio.Lock = asyncio.Lock()


def _scheduler_cache_clear() -> None:
    """Clear the scheduler snapshot TTL cache and reset the refresh lock.

    Test/admin hook.  Resetting the Lock creates a fresh instance bound to
    no event loop, preventing "got Future attached to a different loop" when
    pytest-asyncio's function-scoped loops reuse the module-level Lock across
    successive test functions.
    """
    global _scheduler_cache, _scheduler_refresh_lock
    _scheduler_cache = None
    _scheduler_refresh_lock = asyncio.Lock()


async def get_scheduler_snapshot(
    client: httpx.AsyncClient,
    config: DashboardConfig,
) -> tuple[tuple, str]:
    """Return cached scheduler snapshot ``(six_tuple, snapshot_at)``.

    Calls :func:`collect_scheduler_state` at most once per
    ``_SCHEDULER_TTL_SECONDS`` interval (default 5 s).  Because the
    frontend polls every 3 s, a warm cache means every steady-state poll
    returns instantly without re-running the N-project MCP fan-out.

    Single-flight: when the cache is cold, only the first concurrent caller
    runs the collection; others await the lock and then return from the
    freshly-filled cache (double-checked locking).

    ``snapshot_at`` is an ISO-8601 wall-clock string stamped at cache-fill
    time.  The value propagates through ``api_scheduler`` → ``shape_scheduler``
    → the JS envelope so the UI can show a truthful "data as of" timestamp.

    Single-config assumption: the cache is unkeyed (ignores ``client`` and
    ``config``).  This is correct for the dashboard process, which is always
    started with a single process-wide :class:`DashboardConfig`.  If the
    process ever served multiple configs (multi-tenant), callers with a
    different ``config`` would silently receive another config's snapshot.
    """
    global _scheduler_cache  # must precede first use of the name

    # Fast path — no lock needed for a warm cache hit.
    now = time.monotonic()
    cached = _scheduler_cache
    if cached is not None and (now - cached[0]) < _SCHEDULER_TTL_SECONDS:
        return cached[1], cached[2]

    # Slow path — acquire the lock; re-check after acquiring (double-check).
    async with _scheduler_refresh_lock:
        now = time.monotonic()
        cached = _scheduler_cache
        if cached is not None and (now - cached[0]) < _SCHEDULER_TTL_SECONDS:
            return cached[1], cached[2]

        # We are the one caller responsible for filling the cache.
        # collect_scheduler_state never raises (per-project errors surface as
        # offline_projects), so we always cache the result — even a degraded
        # snapshot where some projects are offline.  Caching a partial result
        # avoids re-running a multi-second fan-out that would just report the
        # same offline state again; ≤5 s of stale-offline is acceptable.
        six_tuple = await collect_scheduler_state(client, config)
        snapshot_at = datetime.now(UTC).isoformat()
        _scheduler_cache = (time.monotonic(), six_tuple, snapshot_at)
        return six_tuple, snapshot_at


# ---------------------------------------------------------------------------
# Pure helpers (no I/O, fully unit-testable)
# ---------------------------------------------------------------------------


def _module_contention_counts(
    rows: list[dict],
    current_holders: dict[str, str],
    *,
    project: str | None = None,
) -> list[dict]:
    """Build a sorted module-contention list from task rows and current holders.

    For each module path that appears in any task's ``lock_set``, count how many
    tasks include it.  Attach the current holder (from *current_holders*) if any.

    The holder for a module is resolved via the shared hierarchical prefix rule
    (``modules_conflict``), not dict-equality, so a holder key that is a
    sub-``lock_depth`` parent of the module still matches.  Both *rows*'
    ``lock_set`` and *current_holders* keys are normalized module locks.

    Args:
        rows:            Task row dicts, each with ``task_id`` and ``lock_set``
                         (normalized module locks).
        current_holders: ``{module_path: task_id}`` from the scheduler snapshot
                         (module keys normalized to lock_depth).
        project:         Optional project label to tag each module entry with.
                         Module paths are NOT globally unique (two projects can
                         each have ``src/utils.py``), so we propagate the
                         owning project label on every module so the merge
                         step can key by ``(project, path)`` without conflating
                         unrelated contention.  When set and a holder exists,
                         ``holder_project`` is also populated — the snapshot's
                         ``current_holders`` map only contains task_ids from
                         the same project root.

    Returns:
        ``[{path, holder, contention, project, holder_project}, ...]`` sorted
        by contention descending, ties broken alphabetically by path.  Each
        entry's ``holder`` is the task_id string from *current_holders*, or
        ``None`` if no task currently holds that module.
    """
    counts: dict[str, int] = {}
    for row in rows:
        for path in row.get('lock_set') or []:
            counts[path] = counts.get(path, 0) + 1

    result: list[dict] = []
    for path, count in counts.items():
        # Resolve the holder via the shared hierarchical prefix rule rather than
        # dict-equality.  Both *path* (a normalized module key) and the
        # current_holders keys are normalized to lock_depth, so the common case
        # is an exact match — but a holder key with fewer than lock_depth
        # components (a sub-depth parent) still conflicts with this child
        # module, and modules_conflict catches it where `.get(path)` would miss.
        holder = next(
            (tid for mod, tid in current_holders.items() if modules_conflict(mod, path)),
            None,
        )
        result.append({
            'path': path,
            'holder': holder,
            'contention': count,
            # `project` scopes this module entry; `holder_project` qualifies
            # the holder task_id since taskmaster ids are project-scoped.
            'project': project,
            'holder_project': project if holder else None,
        })
    return sorted(result, key=lambda m: (-m['contention'], m['path']))


def _compose_rows(
    active_tasks: list[dict],
    snapshot: dict,
    depth: int = 2,
) -> list[dict]:
    """Join active-task metadata with a scheduler snapshot into displayable rows.

    Each active task in *active_tasks* must carry a ``task_id`` field (the raw
    numeric string used as key in the snapshot dicts).  The ``started`` field is
    the task age in minutes (from ``collect_active_tasks``).

    The row's ``lock_set`` is the task's footprint **normalized to module locks**
    at *depth* (the scheduler's ``lock_depth``) via ``files_to_modules`` — so the
    values match the snapshot's ``current_holders`` keys.  The footprint is
    sourced exclusively from ``meta_files`` (taskmaster ``metadata.files``, the
    file footprint the scheduler actually locks on).  Tasks that carry no
    ``meta_files`` produce an empty ``lock_set`` and are correctly invisible to
    the contention view — they hold no scheduler locks.

    Snapshot keys consumed: ``skip_counts``, ``parks``, ``effective_priorities``,
    ``overrides``.

    Returns one dict per task with all fields needed by the React scheduler tab.
    """
    skip_counts: dict[str, int] = snapshot.get('skip_counts') or {}
    parks: dict[str, dict] = snapshot.get('parks') or {}
    effective_priorities: dict[str, str] = snapshot.get('effective_priorities') or {}
    overrides: dict[str, dict] = snapshot.get('overrides') or {}

    result: list[dict] = []
    for task in active_tasks:
        tid = task.get('task_id') or ''
        declared = task.get('priority')
        effective = effective_priorities.get(tid)
        priority_differs = (effective is not None and effective != declared)

        park_state = parks.get(tid)
        if park_state:
            # Age from when parks were installed.  Coerce to str up front
            # — `or ''` handles None, but a number/dict would raise
            # AttributeError on `.replace(...)` and was not previously caught.
            installed_at_str = str(park_state.get('installed_at') or '')
            try:
                installed_at = datetime.fromisoformat(
                    installed_at_str.replace('Z', '+00:00')
                )
                age_seconds = int(
                    (datetime.now(UTC) - installed_at).total_seconds()
                )
                age_seconds = max(age_seconds, 0)
            except (ValueError, TypeError):
                logger.warning(
                    'park_state.installed_at unparseable for %s: %r', tid, installed_at_str
                )
                age_seconds = 0
        else:
            # Age from task started (minutes → seconds)
            age_seconds = int(task.get('started') or 0) * 60

        ov: dict = overrides.get(tid) or {}
        result.append({
            'task_id': tid,
            'title': task.get('title') or '',
            'declared_priority': declared,
            'effective_priority': effective,
            'priority_differs': priority_differs,
            'skip_count': skip_counts.get(tid, 0),
            'park_state': park_state,
            'age_seconds': age_seconds,
            'lock_set': files_to_modules(task.get('meta_files') or [], depth),
            'pinned': bool(ov.get('pinned')),
            'reserve_now': bool(ov.get('reserve_now')),
            'boost_tier': ov.get('boost_tier'),
            'ttl_until': ov.get('ttl_until'),
            # project/project_root are copied verbatim so the React drawer's
            # ActionsPanel.submit and ActivePinsStrip unpin handlers can supply
            # the correct project_root in override/clear-override POSTs.
            'project': task.get('project'),
            'project_root': task.get('project_root'),
        })
    return result


def _skip_event_sparkline(
    events: list[dict],
    *,
    since: datetime,
    until: datetime | None = None,
    bin_seconds: int = 300,
) -> dict:
    """Bin ``task_skipped`` events into fixed-width time buckets.

    Args:
        events:      List of event dicts (may include events of any type).
                     Each dict must have ``event_type`` and ``timestamp`` fields.
        since:       Start of the time window (inclusive).
        until:       End of the time window (exclusive).  Defaults to
                     ``datetime.now(UTC)`` when omitted.
        bin_seconds: Bucket width in seconds (default 5 minutes).

    Returns:
        ``{labels: [iso_str, ...], values: [int, ...]}`` — one entry per bucket
        from *since* to *until*.  Returns ``{labels: [], values: []}`` when
        *events* is empty (no PRNG fallback; the caller renders no sparkline).
    """
    if not events:
        return {'labels': [], 'values': []}

    # Filter to task_skipped events only
    skipped = [e for e in events if e.get('event_type') == 'task_skipped']

    now = until if until is not None else datetime.now(UTC)
    # Normalise since to UTC
    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    window_seconds = int((now - since).total_seconds())
    n_bins = max(1, (window_seconds + bin_seconds - 1) // bin_seconds)

    labels: list[str] = []
    values: list[int] = [0] * n_bins
    for i in range(n_bins):
        bucket_start = since + timedelta(seconds=i * bin_seconds)
        labels.append(bucket_start.isoformat())

    for event in skipped:
        ts_str = event.get('timestamp') or ''
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        offset = (ts - since).total_seconds()
        idx = int(offset // bin_seconds)
        if 0 <= idx < n_bins:
            values[idx] += 1

    return {'labels': labels, 'values': values}


# ---------------------------------------------------------------------------
# Public I/O collector
# ---------------------------------------------------------------------------


async def collect_scheduler_state(
    client: httpx.AsyncClient,
    config: DashboardConfig,
) -> tuple[list[dict], list[dict], list[dict], dict[str, dict], list[str], list[dict]]:
    """Fan out over all configured project roots and aggregate scheduler state.

    For each project root:
      - Calls ``get_scheduler_state`` and ``get_scheduler_events`` via
        ``mcp_tool_call``.
      - Joins with ``collect_active_tasks`` output for task metadata and lock sets.
      - Composes rows, module contention list, and per-task event sparklines.

    Returns:
        ``(rows, modules, pin_queue, events_by_task, offline_projects, paused_projects)``

        rows:             Composed task rows (see module docstring).
        modules:          Sorted module-contention list.
        pin_queue:        Pin-queue list from snapshot.
        events_by_task:   ``{task_id: sparkline_dict}``; empty when no history.
        offline_projects: Project labels that failed to respond.
        paused_projects:  List of ``{'project': label, 'reason': str|None}`` dicts
                          for projects whose scheduler is park-stop paused.
                          Empty list when no project is paused or the snapshot
                          lacks ``is_paused`` (pre-upgrade on-disk degradation).
    """
    since = datetime.now(UTC) - timedelta(hours=1)

    # Fetch active tasks (for lock sets and titles)
    all_active, active_offline = await collect_active_tasks(client, config)

    offline_projects: list[str] = list(active_offline)

    # Determine which project roots still need querying
    roots_to_query = [
        root for root in _all_project_roots(config)
        if _project_label(root) not in offline_projects
    ]

    async def _one_project(root) -> tuple[str, dict | None, list[dict]]:
        """Fetch snapshot + events for one project root.

        URLs are tried sequentially (first-success wins).  On success the
        function returns immediately from inside the ``try`` block, so there is
        no risk of bleed between a partial success on url1 and a failure on url2.

        Simplicity trade-off: if the snapshot call succeeds on url1 but the
        events call raises, the whole pair is re-issued against url2 rather
        than retrying only the events leg.  This amplifies load slightly on
        the surviving URL but keeps the retry logic single-axis; both calls
        are idempotent and small, so the re-fetch is cheap.

        Returns ``(label, snapshot, events)`` on success — ``snapshot`` may be
        ``{}`` for a project with no scheduler state yet (online but empty),
        and is normalised to ``{}`` if the MCP server returns a non-dict
        (defensive — older/buggy servers should never crash the aggregator).
        Returns ``(label, None, [])`` when every URL fails; the ``None``
        sentinel lets the caller distinguish *offline* from *online-but-empty*
        so a legitimately quiescent project is never misclassified as offline.
        """
        label = _project_label(root)

        for url in config.fused_memory_urls:
            try:
                snapshot = await mcp_tool_call(
                    client,
                    url,
                    'get_scheduler_state',
                    {'project_root': str(root)},
                )
                # Normalise: a buggy/older MCP server returning a list or
                # None must not propagate as AttributeError on downstream
                # `.get(...)` calls.  Treat as online-but-empty.
                if not isinstance(snapshot, dict):
                    snapshot = {}
                events_raw = await mcp_tool_call(
                    client,
                    url,
                    'get_scheduler_events',
                    {
                        'project_root': str(root),
                        'since': since.isoformat(),
                        'event_types': ['task_skipped'],
                        'limit': _SCHEDULER_EVENTS_LIMIT,
                    },
                )
                if isinstance(events_raw, list):
                    events: list[dict] = events_raw
                elif isinstance(events_raw, dict):
                    # events may be wrapped: {'events': [...]}
                    events = events_raw.get('events') or []
                else:
                    events = []
                return label, snapshot, events  # first-success wins
            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.HTTPStatusError,
                ValueError,
            ) as exc:
                logger.debug('get_scheduler_state failed for %s / %s: %s', label, url, exc)
                invalidate_session(url)
                continue

        return label, None, []  # all URLs failed; None signals offline

    # Build a label→root map for O(1) lookup after asyncio.gather.
    # (A linear next(...) search over roots_to_query would be O(n²) in the
    # number of projects — trivially avoidable with a single pre-pass.)
    label_to_root = {_project_label(r): r for r in roots_to_query}

    # Fan out to all project roots concurrently (mirrors metrics.py pattern)
    per_project_results = await asyncio.gather(
        *(_one_project(root) for root in roots_to_query),
        return_exceptions=True,
    )

    all_rows: list[dict] = []
    all_modules: list[dict] = []
    all_pin_queue: list[dict] = []
    # NOTE: events_by_task is keyed by composite '{label}/{task_id}' to prevent
    # silent overwrite when the same numeric task_id exists in two project roots
    # (Taskmaster numbers are project-scoped, so collisions are expected in
    # multi-project dashboards).  The React lookup in tab_scheduler.jsx uses
    # `${row.project}/${row.task_id}` to match this key.
    all_events_by_task: dict[str, dict] = {}
    all_paused_projects: list[dict] = []

    for result in per_project_results:
        if isinstance(result, BaseException):
            logger.warning('_one_project raised unexpectedly: %s', result)
            continue
        label, snapshot, events = result

        # None snapshot means every URL failed (offline); {} means online-but-empty.
        if snapshot is None:
            offline_projects.append(label)
            continue

        # Aggregate pause state — reads defensively so pre-upgrade snapshots
        # (missing is_paused key) degrade to "not paused" without error.
        if snapshot.get('is_paused'):
            all_paused_projects.append({
                'project': label,
                'reason': snapshot.get('pause_reason'),
            })

        # O(1) lookup via pre-built map (avoids O(n²) linear search per project)
        root = label_to_root.get(label)
        if root is None:
            continue

        # Build task lookup from active_tasks for this project
        project_active = [t for t in all_active if t.get('project') == label]

        # Enrich active tasks with raw task_id, project label, and project_root
        # for snapshot lookups and downstream override POSTs.
        enriched: list[dict] = []
        for task in project_active:
            uid = task.get('id') or ''
            # uid format: 'project/T-NNN' → raw_id = 'NNN'
            try:
                raw_id = uid.split('/T-', 1)[1]
            except (IndexError, AttributeError):
                raw_id = uid
            enriched.append({
                **task,
                'task_id': raw_id,
                # Propagate project context so _compose_rows can fill
                # project/project_root on every row — without these the React
                # drawer sends empty project_root and every override call 400s.
                'project': label,
                'project_root': str(root),
            })

        # Compose rows — normalize footprints to the scheduler's lock_depth so
        # row lock_set values match the snapshot's current_holders keys.  Older
        # snapshots without lock_depth degrade to the default depth of 2.
        depth = snapshot.get('lock_depth', 2)
        rows = _compose_rows(enriched, snapshot, depth)
        all_rows.extend(rows)

        # Module contention from composed rows + snapshot holders.
        # Tag each module with the owning `project` so _merge_module_lists can
        # key by (project, path) — two projects sharing the same file path
        # (e.g. `src/utils.py`) must not collide into a single heatmap column.
        current_holders: dict[str, str] = snapshot.get('current_holders') or {}
        modules = _module_contention_counts(rows, current_holders, project=label)
        all_modules = _merge_module_lists(all_modules, modules)

        # Pin queue — tag every pin entry with project/project_root before
        # extending.  Taskmaster task_ids are project-scoped, so the same
        # numeric id can appear in two projects' pin queues; without these
        # fields the React strip would collapse rows under a duplicate key
        # and the per-project reorder call would be ambiguous.
        pin_queue_raw = snapshot.get('pin_queue') or []
        for pin in pin_queue_raw:
            if not isinstance(pin, dict):
                continue
            all_pin_queue.append({
                **pin,
                'project': label,
                'project_root': str(root),
            })

        # Per-task event sparklines — keyed by composite '{label}/{tid}' to
        # avoid silent overwrite when the same numeric task_id appears in two
        # project roots (see NOTE above).
        events_list = events if isinstance(events, list) else []
        by_task: dict[str, list] = {}
        for ev in events_list:
            tid = str(ev.get('task_id') or '')
            by_task.setdefault(tid, []).append(ev)
        for tid, task_events in by_task.items():
            composite_key = f'{label}/{tid}'
            all_events_by_task[composite_key] = _skip_event_sparkline(task_events, since=since)

    return all_rows, all_modules, all_pin_queue, all_events_by_task, offline_projects, all_paused_projects


def _merge_module_lists(
    existing: list[dict],
    incoming: list[dict],
) -> list[dict]:
    """Merge two module-contention lists, summing contention counts.

    Modules are keyed by ``(project, path)`` rather than ``path`` alone — two
    different project roots can both contain a file with the same relative
    path (e.g. ``src/utils.py``) and their contention must remain distinct,
    otherwise the heatmap would conflate unrelated locks and the holder
    task_id (project-scoped) would point at a phantom row.

    When the same ``(project, path)`` key appears twice (single project
    aggregation), counts sum and a later holder wins.

    Returns a new sorted list: contention desc, path asc.
    """
    merged: dict[tuple[str | None, str], dict] = {}
    for m in existing + incoming:
        key = (m.get('project'), m['path'])
        if key in merged:
            merged[key]['contention'] += m['contention']
            if m.get('holder'):
                merged[key]['holder'] = m['holder']
                # holder_project must travel with holder so the React side
                # can build the composite ${holder_project}/${holder} key
                # used to look up sparklines and joined row metadata.
                if m.get('holder_project'):
                    merged[key]['holder_project'] = m['holder_project']
        else:
            merged[key] = dict(m)
    return sorted(merged.values(), key=lambda m: (-m['contention'], m['path']))
