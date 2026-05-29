"""Aggregate active tasks across all known projects for the redux dashboard.

Joins three sources — task tree (via fused-memory MCP), worktree artifacts,
and optional burst state from reconciliation — into the ``ACTIVE_TASKS``
shape consumed by the React dashboard's tasks tab.

Output shape (per task) matches ``data.js`` mock fixtures:

    {
        'id': 'dark_factory/T-19',
        'project': 'dark_factory',
        'title': '...',
        'description': '...',
        'details': '...',         # may be empty; many tasks have none
        'status': 'in-progress',
        'agent': 'claude-task-19',  # or None if no worktree
        'started': 14,              # minutes since metadata.created_at, 0 if unknown
        'loops': 2,                 # iterations.jsonl line count
        'attempts': 3,              # review files count
        'deps': [{'id': 'dark_factory/T-15', 'title': '...', 'done': True}, ...],
        'meta_files': ['src/...py', ...],  # taskmaster metadata.files; retained on API for
                                           # debugging/tooling — no frontend UI reads it directly
        'train': {'id': 'demo', 'order': 0},  # present when metadata.train is set; None otherwise
    }

Lock state is surfaced via the scheduler endpoint (see /api/v2/dashboard/scheduler).
The bespoke FILE_LOCKS derivation has been removed; all lock display routes through
``D.SCHEDULER.{rows,modules}`` on the frontend.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx

from dashboard.config import DashboardConfig
from dashboard.data.orchestrator import _scan_worktrees
from dashboard.data.tasks import fetch_statuses, fetch_tasks

_ACTIVE_STATUSES = {'in-progress', 'blocked', 'pending', 'merge-deferred'}

# Maximum done tasks to include per project when the caller opts in via
# ``max_done_per_project``.  Kept at module level so app.py can import it.
_MAX_DONE_PER_PROJECT = 50


def _project_label(root: Path) -> str:
    """Display label for a project root path: the directory's basename."""
    return root.name or str(root)


def _all_project_roots(config: DashboardConfig) -> list[Path]:
    """All known project roots, deduped, primary first."""
    seen: set[Path] = {config.project_root}
    roots: list[Path] = [config.project_root]
    for r in config.known_project_roots:
        if r not in seen:
            seen.add(r)
            roots.append(r)
    return roots


def _task_uid(project: str, task_id: int) -> str:
    """Project-scoped unique id used by the React tasks tab as a map key."""
    return f'{project}/T-{task_id}'


def _minutes_since(iso: str | None) -> int:
    """Whole minutes between *iso* and now (UTC). 0 on parse failure / future."""
    if not iso:
        return 0
    try:
        ts = datetime.fromisoformat(iso.replace('Z', '+00:00'))
    except ValueError:
        return 0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - ts
    minutes = int(delta.total_seconds() // 60)
    return max(minutes, 0)


def _attempts_from_review_summary(summary: str) -> int:
    """Total review attempts from a 'N/M passed' string. 0 on miss/dash."""
    if not summary or '/' not in summary:
        return 0
    head = summary.split('/', 1)[1].split(' ', 1)[0]
    try:
        return int(head)
    except ValueError:
        return 0


def _build_task_row(
    project: str,
    task: dict,
    task_id: int,
    wt: dict,
    uid: str,
) -> dict:
    """Build the common row fields shared by active and done task rows.

    Returns a dict with all fields that are identical regardless of task
    status.  Callers add status-specific fields afterwards:
    active rows add ``started`` (minutes) and ``deps``; done rows add
    ``started: 0``, ``deps: []``, and ``completed`` (ISO timestamp or '').
    """
    meta_files = list((task.get('metadata') or {}).get('files') or [])
    train_meta = (task.get('metadata') or {}).get('train')
    train = (
        {'id': train_meta['id'], 'order': train_meta.get('order', 0)}
        if isinstance(train_meta, dict) and train_meta.get('id')
        else None
    )
    return {
        'id': uid,
        'project': project,
        'title': task.get('title') or '',
        'description': task.get('description') or '',
        'details': task.get('details') or '',
        'status': task.get('status'),
        'agent': f'claude-task-{task_id}' if wt else None,
        'loops': int(wt.get('iteration_count') or 0),
        'attempts': _attempts_from_review_summary(wt.get('review_summary') or ''),
        'meta_files': meta_files,
        'train': train,
    }


async def _shape_one_project(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: Path,
    *,
    max_done_per_project: int = 0,
) -> tuple[list[dict], bool, int]:
    """Build ``(active_tasks, offline, done_count)`` for a single project root.

    *offline* is True when the MCP fetch failed for this project; the
    caller surfaces that in the API payload so the React Tasks tab can
    show an offline banner.

    *done_count* is the total number of ``'done'`` tasks in the fetched
    list, **before** the *max_done_per_project* cap is applied.  Callers
    that need the authoritative count for display should use this value
    rather than counting the (capped) emitted rows.

    When *max_done_per_project* > 0, the most-recent N done tasks
    (sorted by ``updated_at`` descending, then ``id`` descending) are
    appended to the returned list.  Each done row carries a ``completed``
    field (the ``updated_at`` ISO string or ``''``).  Active rows are
    unaffected.
    """
    project = _project_label(project_root)
    fetched = await fetch_tasks(client, config, project_root)
    if isinstance(fetched, dict) and fetched.get('offline'):
        return [], True, 0
    tasks = fetched if isinstance(fetched, list) else []
    if not tasks:
        return [], False, 0

    # Count ALL done tasks before any cap — this is the authoritative figure
    # for the DONE_COUNTS payload key.
    done_count = sum(1 for t in tasks if t.get('status') == 'done')

    worktrees = await asyncio.to_thread(_scan_worktrees, project_root / '.worktrees')

    # Lookup table for dep title/status resolution within the same project.
    by_id: dict[int, dict] = {t['id']: t for t in tasks if isinstance(t.get('id'), int)}

    active: list[dict] = []

    for task in tasks:
        status = task.get('status')
        if status not in _ACTIVE_STATUSES:
            continue

        task_id = task['id']
        wt = worktrees.get(task_id) or {}
        meta = wt.get('metadata') or {}

        deps: list[dict] = []
        for dep_id in task.get('dependencies') or []:
            dep_task = by_id.get(dep_id)
            if dep_task is None:
                continue
            deps.append({
                'id': _task_uid(project, dep_id),
                'title': dep_task.get('title') or '',
                'done': dep_task.get('status') == 'done',
            })

        uid = _task_uid(project, task_id)
        row = _build_task_row(project, task, task_id, wt, uid)
        # active rows: started from worktree creation time; deps from task tree.
        row['started'] = _minutes_since(meta.get('created_at'))
        row['deps'] = deps
        active.append(row)

    if max_done_per_project > 0:
        done_tasks = [t for t in tasks if t.get('status') == 'done']
        # Sort by updated_at descending; id descending as tie-breaker.
        done_tasks.sort(
            key=lambda t: (t.get('updated_at') or '', t.get('id') or 0),
            reverse=True,
        )
        for task in done_tasks[:max_done_per_project]:
            task_id = task['id']
            uid = _task_uid(project, task_id)
            wt = worktrees.get(task_id) or {}
            row = _build_task_row(project, task, task_id, wt, uid)
            # done rows: no meaningful start time; no deps surfaced.
            row['started'] = 0
            row['deps'] = []
            row['completed'] = task.get('updated_at') or ''
            active.append(row)

    return active, False, done_count


async def collect_tasks_with_counts(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    *,
    max_done_per_project: int = 0,
) -> tuple[list[dict], list[str], dict[str, int]]:
    """Aggregate active tasks and per-project done counts in a single MCP pass.

    Returns ``(active_tasks, offline_projects, done_counts)`` where:

    - *active_tasks* is the list of active (and optionally bounded done) rows
    - *offline_projects* lists project labels whose MCP fetch failed
    - *done_counts* maps project label → total done task count (pre-cap)

    Prefer this over calling ``collect_active_tasks`` and
    ``collect_done_counts`` concurrently: it halves per-project MCP
    round-trips and guarantees that DONE_COUNTS matches the same snapshot
    as the ACTIVE_TASKS rows.
    """
    all_active: list[dict] = []
    offline_projects: list[str] = []
    done_counts: dict[str, int] = {}
    for root in _all_project_roots(config):
        active, offline, done_count = await _shape_one_project(
            client, config, root, max_done_per_project=max_done_per_project,
        )
        label = _project_label(root)
        if offline:
            offline_projects.append(label)
        else:
            done_counts[label] = done_count
        all_active.extend(active)
    return all_active, offline_projects, done_counts


async def collect_active_tasks(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    *,
    max_done_per_project: int = 0,
) -> tuple[list[dict], list[str]]:
    """Collect active tasks across all known projects.

    Returns ``(active_tasks, offline_projects)`` where *offline_projects* is
    the list of project labels whose MCP fetch failed.  The handler turns a
    non-empty *offline_projects* into ``offline: True`` on the dashboard payload.

    When *max_done_per_project* > 0, the most-recent N done tasks per project
    are appended to the returned list (each with a ``completed`` field).
    Default 0 leaves the return shape unchanged — scheduler.py is unaffected.

    Lock state is surfaced via the scheduler endpoint — see
    /api/v2/dashboard/scheduler.

    Note: callers that also need per-project done counts should use
    ``collect_tasks_with_counts`` to avoid a second MCP round-trip.
    """
    active, offline, _ = await collect_tasks_with_counts(
        client, config, max_done_per_project=max_done_per_project,
    )
    return active, offline


async def collect_done_counts(
    client: httpx.AsyncClient,
    config: DashboardConfig,
) -> dict[str, int]:
    """Return a ``{project_label: done_count}`` map for all reachable projects.

    Uses the compact ``fetch_statuses`` (the same source the burndown collector
    uses) so the count agrees with the burndown snapshot.

    All projects are fetched concurrently to minimise latency.  Projects whose
    ``fetch_statuses`` returns an offline marker are silently omitted.
    """
    roots = _all_project_roots(config)
    results = await asyncio.gather(
        *(fetch_statuses(client, config, r) for r in roots)
    )
    counts: dict[str, int] = {}
    for root, result in zip(roots, results):
        # Skip offline markers (dict with 'offline' key).
        if isinstance(result, dict) and result.get('offline'):
            continue
        label = _project_label(root)
        counts[label] = sum(1 for s in result.values() if s == 'done')
    return counts
