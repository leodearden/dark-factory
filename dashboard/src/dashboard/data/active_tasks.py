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
from dashboard.data.tasks import fetch_tasks

_ACTIVE_STATUSES = {'in-progress', 'blocked', 'pending'}


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


async def _shape_one_project(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: Path,
) -> tuple[list[dict], bool]:
    """Build (active_tasks, offline) for a single project root.

    *offline* is True when the MCP fetch failed for this project; the
    caller surfaces that in the API payload so the React Tasks tab can
    show an offline banner.
    """
    project = _project_label(project_root)
    fetched = await fetch_tasks(client, config, project_root)
    if isinstance(fetched, dict) and fetched.get('offline'):
        return [], True
    tasks = fetched if isinstance(fetched, list) else []
    if not tasks:
        return [], False

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

        # Footprint the scheduler actually locks on: taskmaster metadata.files.
        # This is the same source the scheduler derives its module locks from;
        # lock display is routed through D.SCHEDULER.{rows,modules} on the frontend.
        meta_files = list((task.get('metadata') or {}).get('files') or [])

        uid = _task_uid(project, task_id)
        agent = f'claude-task-{task_id}' if wt else None
        active.append({
            'id': uid,
            'project': project,
            'title': task.get('title') or '',
            'description': task.get('description') or '',
            'details': task.get('details') or '',
            'status': status,
            'agent': agent,
            'started': _minutes_since(meta.get('created_at')),
            'loops': int(wt.get('iteration_count') or 0),
            'attempts': _attempts_from_review_summary(wt.get('review_summary') or ''),
            'deps': deps,
            'meta_files': meta_files,
        })

    return active, False


async def collect_active_tasks(
    client: httpx.AsyncClient,
    config: DashboardConfig,
) -> tuple[list[dict], list[str]]:
    """Collect active tasks across all known projects.

    Returns ``(active_tasks, offline_projects)`` where *offline_projects* is
    the list of project labels whose MCP fetch failed.  The handler turns a
    non-empty *offline_projects* into ``offline: True`` on the dashboard payload.

    Lock state is surfaced via the scheduler endpoint — see
    /api/v2/dashboard/scheduler.
    """
    all_active: list[dict] = []
    offline_projects: list[str] = []
    for root in _all_project_roots(config):
        active, offline = await _shape_one_project(client, config, root)
        if offline:
            offline_projects.append(_project_label(root))
        all_active.extend(active)
    return all_active, offline_projects
