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
        lock_set           list[str]   declared file locks from plan.json
        pinned             bool        from snapshot.overrides
        reserve_now        bool        from snapshot.overrides
        boost_tier         str | None  from snapshot.overrides
        ttl_until          str | None  from snapshot.overrides

    modules: list[dict]  — one per unique module path that appears in any lock set:
        path        str
        holder      str | None   task_id of in-flight holder, or None
        contention  int          count of tasks whose lock_set includes this path
        sorted by contention desc, ties broken alphabetically by path

    pin_queue: list[dict]  — from snapshot.pin_queue: [{task_id, order}, ...]

    events_by_task: dict[str, dict]  — {task_id: {labels: [...], values: [...]}}
        sparklines from get_scheduler_events; empty dict when no history

    offline_projects: list[str]  — project labels that failed to respond
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import _all_project_roots, _project_label, collect_active_tasks
from dashboard.data.memory import invalidate_session, mcp_tool_call

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers (no I/O, fully unit-testable)
# ---------------------------------------------------------------------------


def _module_contention_counts(
    rows: list[dict],
    current_holders: dict[str, str],
) -> list[dict]:
    """Build a sorted module-contention list from task rows and current holders.

    For each module path that appears in any task's ``lock_set``, count how many
    tasks include it.  Attach the current holder (from *current_holders*) if any.

    Args:
        rows:            Task row dicts, each with ``task_id`` and ``lock_set``.
        current_holders: ``{module_path: task_id}`` from the scheduler snapshot.

    Returns:
        ``[{path, holder, contention}, ...]`` sorted by contention descending,
        ties broken alphabetically by path.  Each entry's ``holder`` is the
        task_id string from *current_holders*, or ``None`` if no task currently
        holds that module.
    """
    counts: dict[str, int] = {}
    for row in rows:
        for path in row.get('lock_set') or []:
            counts[path] = counts.get(path, 0) + 1

    return sorted(
        [
            {
                'path': path,
                'holder': current_holders.get(path),
                'contention': count,
            }
            for path, count in counts.items()
        ],
        key=lambda m: (-m['contention'], m['path']),
    )


def _compose_rows(
    active_tasks: list[dict],
    snapshot: dict,
) -> list[dict]:
    """Join active-task metadata with a scheduler snapshot into displayable rows.

    Each active task in *active_tasks* must carry a ``task_id`` field (the raw
    numeric string used as key in the snapshot dicts).  The ``started`` field is
    the task age in minutes (from ``collect_active_tasks``).

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
            # Age from when parks were installed
            installed_at_str = park_state.get('installed_at') or ''
            try:
                installed_at = datetime.fromisoformat(
                    installed_at_str.replace('Z', '+00:00')
                )
                age_seconds = int(
                    (datetime.now(UTC) - installed_at).total_seconds()
                )
                age_seconds = max(age_seconds, 0)
            except (ValueError, TypeError):
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
            'lock_set': list(task.get('locks') or []),
            'pinned': bool(ov.get('pinned')),
            'reserve_now': bool(ov.get('reserve_now')),
            'boost_tier': ov.get('boost_tier'),
            'ttl_until': ov.get('ttl_until'),
        })
    return result


def _skip_event_sparkline(
    events: list[dict],
    *,
    since: datetime,
    bin_seconds: int = 300,
) -> dict:
    """Bin ``task_skipped`` events into fixed-width time buckets.

    Args:
        events:      List of event dicts (may include events of any type).
                     Each dict must have ``event_type`` and ``timestamp`` fields.
        since:       Start of the time window (inclusive).
        bin_seconds: Bucket width in seconds (default 5 minutes).

    Returns:
        ``{labels: [iso_str, ...], values: [int, ...]}`` — one entry per bucket
        from *since* to ``now``.  Returns ``{labels: [], values: []}`` when
        *events* is empty (no PRNG fallback; the caller renders no sparkline).
    """
    if not events:
        return {'labels': [], 'values': []}

    # Filter to task_skipped events only
    skipped = [e for e in events if e.get('event_type') == 'task_skipped']

    now = datetime.now(UTC)
    # Normalise since to UTC
    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)

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
) -> tuple[list[dict], list[dict], list[dict], dict[str, dict], list[str]]:
    """Fan out over all configured project roots and aggregate scheduler state.

    For each project root:
      - Calls ``get_scheduler_state`` and ``get_scheduler_events`` via
        ``mcp_tool_call``.
      - Joins with ``collect_active_tasks`` output for task metadata and lock sets.
      - Composes rows, module contention list, and per-task event sparklines.

    Returns:
        ``(rows, modules, pin_queue, events_by_task, offline_projects)``

        rows:             Composed task rows (see module docstring).
        modules:          Sorted module-contention list.
        pin_queue:        Pin-queue list from snapshot.
        events_by_task:   ``{task_id: sparkline_dict}``; empty when no history.
        offline_projects: Project labels that failed to respond.
    """
    since = datetime.now(UTC) - timedelta(hours=1)

    # Fetch active tasks (for lock sets and titles)
    all_active, _file_locks, active_offline = await collect_active_tasks(client, config)

    all_rows: list[dict] = []
    all_modules: list[dict] = []
    all_pin_queue: list[dict] = []
    all_events_by_task: dict[str, dict] = {}
    offline_projects: list[str] = list(active_offline)

    for root in _all_project_roots(config):
        label = _project_label(root)
        if label in offline_projects:
            continue  # already failed during collect_active_tasks

        snapshot: dict = {}
        events: list[dict] = []

        for url in config.fused_memory_urls:
            try:
                snapshot = await mcp_tool_call(
                    client,
                    url,
                    'get_scheduler_state',
                    {'project_root': str(root)},
                )
                events = await mcp_tool_call(
                    client,
                    url,
                    'get_scheduler_events',
                    {
                        'project_root': str(root),
                        'since': since.isoformat(),
                        'event_types': ['task_skipped'],
                    },
                )
                if isinstance(events, list):
                    break
                # events may be wrapped: {'events': [...]}
                if isinstance(events, dict):
                    events = events.get('events') or []
                break
            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.HTTPStatusError,
                ValueError,
            ) as exc:
                logger.debug('get_scheduler_state failed for %s / %s: %s', label, url, exc)
                invalidate_session(url)
                continue

        if not snapshot:
            offline_projects.append(label)
            continue

        # Build task lookup from active_tasks for this project
        project_active = [t for t in all_active if t.get('project') == label]

        # Enrich active tasks with raw task_id for snapshot lookups
        enriched: list[dict] = []
        for task in project_active:
            uid = task.get('id') or ''
            # uid format: 'project/T-NNN' → raw_id = 'NNN'
            try:
                raw_id = uid.split('/T-', 1)[1]
            except (IndexError, AttributeError):
                raw_id = uid
            enriched.append({**task, 'task_id': raw_id})

        # Compose rows
        rows = _compose_rows(enriched, snapshot)
        all_rows.extend(rows)

        # Module contention from composed rows + snapshot holders
        current_holders: dict[str, str] = snapshot.get('current_holders') or {}
        modules = _module_contention_counts(rows, current_holders)
        all_modules = _merge_module_lists(all_modules, modules)

        # Pin queue
        pin_queue_raw = snapshot.get('pin_queue') or []
        all_pin_queue.extend(pin_queue_raw)

        # Per-task event sparklines
        events_list = events if isinstance(events, list) else []
        by_task: dict[str, list] = {}
        for ev in events_list:
            tid = str(ev.get('task_id') or '')
            by_task.setdefault(tid, []).append(ev)
        for tid, task_events in by_task.items():
            all_events_by_task[tid] = _skip_event_sparkline(task_events, since=since)

    return all_rows, all_modules, all_pin_queue, all_events_by_task, offline_projects


def _merge_module_lists(
    existing: list[dict],
    incoming: list[dict],
) -> list[dict]:
    """Merge two module-contention lists, summing contention counts.

    Returns a new sorted list: contention desc, path asc.
    """
    merged: dict[str, dict] = {}
    for m in existing + incoming:
        path = m['path']
        if path in merged:
            merged[path]['contention'] += m['contention']
            if m.get('holder'):
                merged[path]['holder'] = m['holder']
        else:
            merged[path] = dict(m)
    return sorted(merged.values(), key=lambda m: (-m['contention'], m['path']))
