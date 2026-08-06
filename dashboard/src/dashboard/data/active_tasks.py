"""Aggregate active tasks across all known projects for the redux dashboard.

Joins three sources — task tree (via fused-memory MCP), per-task runtime
state (via the orchestrator's escalation MCP, ``get_task_runtime_state``),
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
        'agent': 'claude-task-19',  # TaskRuntimeEntry.has_worktree; None if no worktree
        'started': 14,              # minutes since TaskRuntimeEntry.started (runtime snapshot)
        'loops': 2,                 # TaskRuntimeEntry.loops (runtime snapshot)
        'attempts': 3,              # TaskRuntimeEntry.attempts (runtime snapshot)
        'lane': '_lane-7',          # TaskRuntimeEntry.lane, or None
        'phase': 'EXECUTE',         # TaskRuntimeEntry.phase, or None
        'lane_state': 'assigned',   # TaskRuntimeEntry.lane_state, or None
        'runtime_offline': False,   # True iff this project's runtime snapshot is unreachable —
                                     # loops/attempts/started/agent/lane/phase/lane_state are then
                                     # ALL None (never a fabricated 0). A task absent from an
                                     # online snapshot instead gets honest zeros/None with
                                     # runtime_offline False; a per-task read failure on an online
                                     # snapshot yields None fields with runtime_offline still False
                                     # (honest error != offline). See _runtime_fields.
        'runtime_status': 'ok',     # WHY runtime_offline is what it is — see RuntimeStatus.
                                     # runtime_offline alone cannot tell an operator whether the
                                     # orchestrator is down ('unreachable'), the dashboard was too
                                     # starved to ask within its own probe budget
                                     # ('deadline_exceeded'), or no orchestrator is configured for
                                     # this root at all ('not_configured') — three cases that
                                     # demand opposite responses but render identically as blank
                                     # cells. Collapsing them is what made the 2026-07-30 event get
                                     # misdiagnosed as an orchestrator outage. runtime_offline is
                                     # UNCHANGED: True for every non-'ok' member.
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
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot

from dashboard.config import DashboardConfig
from dashboard.data.task_runtime import fetch_task_runtime
from dashboard.data.tasks import fetch_external_statuses, fetch_statuses, fetch_tasks
from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = {'in-progress', 'blocked', 'pending', 'merge-deferred', 'deferred'}

# Why a task row's runtime fields are what they are — the FAULT DOMAIN of the
# runtime probe, not the task (task 3517). Mirrors the ``status``-discriminator
# convention already used by ``app._probe_db`` and ``redux_api._shape_wal_status``.
#
# - 'ok'                the probe succeeded. Fields are real (or honest zeros
#                       for a task absent from the snapshot, or honest Nones on
#                       a PER-TASK read failure — the probe was still fine).
# - 'not_configured'    no escalation URL for this project root, so nothing was
#                       ever probed. Expected and permanent; NOT a fault.
# - 'unreachable'       connect refused / HTTP error / malformed payload. The
#                       ORCHESTRATOR is the fault domain; go look at it.
# - 'deadline_exceeded' the probe's own budget fired. Likely the DASHBOARD's
#                       fault under a starved event loop — the orchestrator may
#                       be perfectly healthy (2026-07-30).
# - 'unknown'           the snapshot said offline with no reason. Out-of-contract
#                       for a dashboard-synthesized snapshot; the honest sentinel.
#                       NEVER fabricate a diagnosis to fill this in.
RuntimeStatus = Literal[
    'ok', 'not_configured', 'unreachable', 'deadline_exceeded', 'unknown',
]

# Maximum done / cancelled tasks to include per project when the caller opts in
# via ``max_done_per_project`` / ``max_cancelled_per_project``.
# Kept at module level so app.py can import them.
_MAX_DONE_PER_PROJECT = 50
_MAX_CANCELLED_PER_PROJECT = 50

# Defensive-visibility threshold: the live-PRD terminal-member exemption (see
# _shape_one_project) never drops rows — "all live members" is the contract,
# not "up to N" — but a PRD with an unusually large number of live done/
# cancelled members beyond the per-bucket cap logs a warning so a
# pathological case is visible rather than silently inflating the payload.
_LIVE_PRD_EXEMPTION_WARN_THRESHOLD = 200


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


def _minutes_since(iso: str | None, *, now: datetime | None = None) -> int:
    """Whole minutes between *iso* and *now* (UTC). 0 on parse failure / future.

    *now* defaults to the live clock via :func:`dashboard.data.utils.resolve_now`;
    pass an explicit value for deterministic results or to share one instant
    across multiple rows in an aggregation.
    """
    if not iso:
        return 0
    try:
        ts = datetime.fromisoformat(iso.replace('Z', '+00:00'))
    except ValueError:
        return 0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    delta = resolve_now(now) - ts
    minutes = int(delta.total_seconds() // 60)
    return max(minutes, 0)


def _coalesce_prd(metadata: dict) -> str | None:
    """Coalesce PRD provenance from *metadata* into a single normalized string.

    Checks ``prd_path``, then ``prd``, then ``prd_ref`` (in that precedence
    order); the first value that is a non-empty string after stripping a
    trailing ``#anchor`` or ``§section`` suffix and surrounding whitespace
    wins. Non-string values are skipped. A value that cleans to ``''`` (e.g.
    it was only a suffix) falls through to the next key. Returns ``None``
    when no key yields a non-empty result.
    """
    for key in ('prd_path', 'prd', 'prd_ref'):
        raw = metadata.get(key)
        if not isinstance(raw, str):
            continue
        cleaned = raw.split('#', 1)[0].split('§', 1)[0].strip()
        if cleaned:
            return cleaned
    return None


def _build_task_row(
    project: str,
    task: dict,
    task_id: int,
    rt: dict,
    uid: str,
    *,
    prd: str | None = None,
) -> dict:
    """Build the common row fields shared by active and done task rows.

    Returns a dict with all fields that are identical regardless of task
    status.  Callers add status-specific fields afterwards:
    active rows add ``started`` (minutes) and ``deps``; done rows add
    ``started: 0``, ``deps: []``, and ``completed`` (ISO timestamp or '').

    *rt* is the runtime-fields dict produced by :func:`_runtime_fields`
    (``agent``/``loops``/``attempts``/``lane``/``phase``/``lane_state``/
    ``runtime_offline``/``runtime_status`` — ``started`` is handled separately by the caller,
    since active rows use ``rt['started']`` while terminal rows hard-code
    ``0``). Missing keys default to ``None``/``False`` so a bare ``{}`` (used
    by direct unit tests of this function) is still valid.

    *prd*, if given, is used verbatim as the row's ``prd`` value instead of
    re-deriving it from *task*'s metadata via ``_coalesce_prd`` — callers
    that already computed it (e.g. the terminal-bucket loop, to decide
    live-PRD membership) can pass it through to avoid doing the same
    split/strip work twice. Omitting it (or passing ``None``, the actual
    no-provenance value) falls back to deriving it from metadata, which is
    safe because re-deriving a true ``None`` is idempotent.
    """
    metadata = task.get('metadata') or {}
    meta_files = list(metadata.get('files') or [])
    train_meta = metadata.get('train')
    train = (
        {'id': train_meta['id'], 'order': train_meta.get('order', 0)}
        if isinstance(train_meta, dict) and train_meta.get('id')
        else None
    )
    raw_ext = metadata.get('external_deps')
    external_deps = (
        [{'id': dep, 'status': 'unknown'}
         for dep in raw_ext
         if isinstance(dep, str) and dep]
        if isinstance(raw_ext, list) else []
    )
    return {
        'id': uid,
        'project': project,
        'title': task.get('title') or '',
        'description': task.get('description') or '',
        'details': task.get('details') or '',
        'status': task.get('status'),
        'agent': rt.get('agent'),
        'loops': rt.get('loops'),
        'attempts': rt.get('attempts'),
        'lane': rt.get('lane'),
        'phase': rt.get('phase'),
        'lane_state': rt.get('lane_state'),
        'runtime_offline': rt.get('runtime_offline', False),
        'runtime_status': rt.get('runtime_status', 'ok'),
        'meta_files': meta_files,
        'train': train,
        'external_deps': external_deps,
        'prd': prd if prd is not None else _coalesce_prd(metadata),
    }


def _probe_status(runtime: TaskRuntimeSnapshot | None) -> RuntimeStatus:
    """Classify a project's runtime probe outcome — see :data:`RuntimeStatus`.

    ``None`` means the project label never appeared in the fan-out result at
    all, i.e. no escalation URL is configured for it and no probe was ever
    attempted — distinct from a probe that was attempted and failed.

    An ``offline=True`` snapshot with no ``offline_reason`` is out-of-contract
    for one the dashboard synthesized, so it degrades to ``'unknown'`` rather
    than being assigned a plausible-sounding reason we did not measure.
    """
    if runtime is None:
        return 'not_configured'
    if not runtime.offline:
        return 'ok'
    if runtime.offline_reason == 'deadline_exceeded':
        return 'deadline_exceeded'
    if runtime.offline_reason == 'unreachable':
        return 'unreachable'
    return 'unknown'


def _runtime_fields(
    index: dict[int, TaskRuntimeEntry],
    status: RuntimeStatus,
    task_id: int,
    *,
    now: datetime | None = None,
) -> dict:
    """Derive a task row's runtime-sourced fields from the project's runtime snapshot.

    *status* is this project's probe outcome from :func:`_probe_status`, and is
    emitted verbatim as the row's ``runtime_status``. ``runtime_offline`` is
    derived from it here as ``status != 'ok'`` — one source of truth — and keeps
    its EXACT prior meaning, so no downstream consumer's semantics shift.

    Three cases:

    - *status* is any non-``'ok'`` member (``'not_configured'``,
      ``'unreachable'``, ``'deadline_exceeded'``, ``'unknown'`` — we have no
      usable snapshot, whatever the cause): every field is ``None`` — never a
      fabricated ``0`` — and ``runtime_offline`` is ``True``.
    - *task_id* absent from *index* (the project IS online, the task just has
      no entry in its snapshot): honest zeros (``loops``/``attempts``/
      ``started`` are ``0``; ``agent``/``lane``/``phase``/``lane_state`` are
      ``None``) and ``runtime_offline`` is ``False``.
    - *task_id* present in *index*: real fields from the ``TaskRuntimeEntry``,
      which may themselves be ``None`` on a per-task artifact read failure —
      an honest per-task error, not an offline project, so ``runtime_offline``
      stays ``False`` and ``runtime_status`` stays ``'ok'`` either way (the
      PROBE succeeded; only this one task's artifact read did not).
    """
    if status != 'ok':
        return {
            'agent': None, 'loops': None, 'attempts': None, 'started': None,
            'lane': None, 'phase': None, 'lane_state': None,
            'runtime_offline': True, 'runtime_status': status,
        }
    entry = index.get(task_id)
    if entry is None:
        return {
            'agent': None, 'loops': 0, 'attempts': 0, 'started': 0,
            'lane': None, 'phase': None, 'lane_state': None,
            'runtime_offline': False, 'runtime_status': status,
        }
    return {
        'agent': f'claude-task-{task_id}' if entry.has_worktree else None,
        'loops': entry.loops,
        'attempts': entry.attempts,
        'started': _minutes_since(entry.started, now=now),
        'lane': entry.lane,
        'phase': entry.phase,
        'lane_state': entry.lane_state,
        'runtime_offline': False,
        'runtime_status': status,
    }


def _resolve_deps(task: dict, by_id: dict[int, dict], project: str) -> list[dict]:
    """Resolve *task*'s ``dependencies`` ids into ``{id, title, done}`` dicts.

    *by_id* is the full project task lookup (built over every fetched task,
    regardless of status bucket), so a dependency resolves whether it lives
    in an active status or a terminal one.
    """
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
    return deps


async def _shape_one_project(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: Path,
    *,
    max_done_per_project: int = 0,
    max_cancelled_per_project: int = 0,
    now: datetime | None = None,
    runtime: TaskRuntimeSnapshot | None = None,
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

    When *max_cancelled_per_project* > 0, the most-recent N cancelled tasks
    are similarly appended (same sort key, same ``completed`` field, same
    ``started: 0`` / ``deps: []`` treatment as done rows).

    *runtime* is this project's ``TaskRuntimeSnapshot`` (resolved ONCE by the
    caller — see ``collect_tasks_with_counts`` — via ``fetch_task_runtime``).
    ``None`` (no escalation URL configured for this project) is treated
    identically to ``runtime.offline``: every row's runtime-sourced fields
    degrade to an honest ``None`` via :func:`_runtime_fields`, distinct from
    the task-tree ``offline`` return value above.
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

    runtime_status = _probe_status(runtime)
    runtime_index: dict[int, TaskRuntimeEntry] = (
        {e.task_id: e for e in runtime.tasks} if runtime is not None and not runtime.offline else {}
    )

    # Lookup table for dep title/status resolution within the same project.
    by_id: dict[int, dict] = {t['id']: t for t in tasks if isinstance(t.get('id'), int)}

    active: list[dict] = []

    for task in tasks:
        status = task.get('status')
        if status not in _ACTIVE_STATUSES:
            continue

        task_id = task['id']
        rt = _runtime_fields(runtime_index, runtime_status, task_id, now=now)

        uid = _task_uid(project, task_id)
        row = _build_task_row(project, task, task_id, rt, uid)
        # active rows: started from the runtime entry; deps from task tree.
        row['started'] = rt['started']
        row['deps'] = _resolve_deps(task, by_id, project)
        active.append(row)

    # PRDs with at least one member still in an active status. Done/cancelled
    # members of these "live" PRDs are exempt from the terminal-bucket cap —
    # see Contract: task-row prd field in plans/dashboard-taskgraph-legibility-prd.md.
    live_prds = {row['prd'] for row in active if row.get('prd')}

    # Bounded terminal buckets: iterate over (status, cap) pairs. Rows within
    # the top-N cap keep the original done/cancelled shape; done/cancelled
    # members of a still-live PRD are additionally emitted beyond the cap
    # (with populated deps) per the live-PRD terminal-member exemption above.
    for _bkt_status, _bkt_cap in (
        ('done', max_done_per_project),
        ('cancelled', max_cancelled_per_project),
    ):
        if _bkt_cap <= 0:
            continue
        bucket_tasks = [t for t in tasks if t.get('status') == _bkt_status]
        # Sort by updated_at descending; id descending as tie-breaker.
        bucket_tasks.sort(
            key=lambda t: (t.get('updated_at') or '', t.get('id') or 0),
            reverse=True,
        )
        capped_ids = {t['id'] for t in bucket_tasks[:_bkt_cap]}
        exempted_count = 0
        for task in bucket_tasks:
            task_id = task['id']
            prd = _coalesce_prd(task.get('metadata') or {})
            is_live_member = prd is not None and prd in live_prds
            beyond_cap = task_id not in capped_ids
            if beyond_cap and not is_live_member:
                continue
            if beyond_cap:
                exempted_count += 1
            uid = _task_uid(project, task_id)
            rt = _runtime_fields(runtime_index, runtime_status, task_id, now=now)
            row = _build_task_row(project, task, task_id, rt, uid, prd=prd)
            # terminal rows: no meaningful start time; deps only for live-PRD
            # members (the terminal-member exemption), else unsurfaced.
            row['started'] = 0
            row['deps'] = _resolve_deps(task, by_id, project) if is_live_member else []
            row['completed'] = task.get('updated_at') or ''
            active.append(row)
        if exempted_count > _LIVE_PRD_EXEMPTION_WARN_THRESHOLD:
            logger.warning(
                'project %s: live-PRD exemption emitted %d %s rows beyond the '
                'cap (max=%d) — a PRD may have an unusually large number of '
                'live terminal members',
                project, exempted_count, _bkt_status, _bkt_cap,
            )

    return active, False, done_count


async def collect_tasks_with_counts(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    *,
    max_done_per_project: int = 0,
    max_cancelled_per_project: int = 0,
    resolve_external: bool = False,
    now: datetime | None = None,
) -> tuple[list[dict], list[str], dict[str, int]]:
    """Aggregate active tasks and per-project done counts in a single MCP pass.

    Returns ``(active_tasks, offline_projects, done_counts)`` where:

    - *active_tasks* is the list of active (and optionally bounded done) rows
    - *offline_projects* lists project labels whose MCP fetch failed
    - *done_counts* maps project label → total done task count (pre-cap)

    When *resolve_external* is ``True``, gathers the deduped union of every
    row's ``external_deps`` ids, issues **one** batched
    ``fetch_external_statuses`` call (skipped when the union is empty), and
    overwrites each entry's status.  Deps absent from the map keep the honest
    ``'unknown'`` sentinel.  Defaults to ``False`` so the scheduler-page path
    (``collect_active_tasks``) issues no extra MCP round-trip.

    *now* is resolved ONCE (via :func:`dashboard.data.utils.resolve_now`) at
    this aggregation boundary and threaded into every project's
    ``_shape_one_project`` call, so every returned row's ``started`` shares
    the same instant regardless of which project it came from.

    Per-task runtime state (loops/attempts/started/agent/lane/phase/
    lane_state) is likewise fetched ONCE here — a single concurrent fan-out
    via :func:`dashboard.data.task_runtime.fetch_task_runtime` over
    ``config.escalation_urls`` — and each project's snapshot is threaded into
    its ``_shape_one_project`` call, mirroring the single-``now`` threading.

    Prefer this over calling ``collect_active_tasks`` and
    ``collect_done_counts`` concurrently: it halves per-project MCP
    round-trips and guarantees that DONE_COUNTS matches the same snapshot
    as the ACTIVE_TASKS rows.
    """
    effective_now = resolve_now(now)
    runtime_by_label = await fetch_task_runtime(client, config.escalation_urls)
    all_active: list[dict] = []
    offline_projects: list[str] = []
    done_counts: dict[str, int] = {}
    for root in _all_project_roots(config):
        label = _project_label(root)
        active, offline, done_count = await _shape_one_project(
            client, config, root,
            max_done_per_project=max_done_per_project,
            max_cancelled_per_project=max_cancelled_per_project,
            now=effective_now,
            runtime=runtime_by_label.get(label),
        )
        if offline:
            offline_projects.append(label)
        else:
            done_counts[label] = done_count
        all_active.extend(active)

    if resolve_external:
        # Gather the deduped union of external dep ids for ACTIVE (non-done) rows only.
        # Done rows' external deps are no longer actionable; skipping them avoids
        # needless MCP load and prevents 'External dependencies' chips appearing on
        # completed tasks in the dashboard (where they would be noise, not signal).
        dep_ids: set[str] = set()
        for row in all_active:
            if 'completed' in row:
                continue  # skip bounded done rows
            for entry in row.get('external_deps') or []:
                dep_ids.add(entry['id'])
        if dep_ids:
            status_map = await fetch_external_statuses(
                client, config, sorted(dep_ids),
            )
            map_offline = bool(status_map.get('offline'))
            for row in all_active:
                if 'completed' in row:
                    continue  # skip bounded done rows
                for entry in row.get('external_deps') or []:
                    if map_offline:
                        entry['status'] = 'offline'
                    else:
                        entry['status'] = status_map.get(entry['id'], 'unknown')

    return all_active, offline_projects, done_counts


async def collect_active_tasks(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    *,
    max_done_per_project: int = 0,
    max_cancelled_per_project: int = 0,
    now: datetime | None = None,
) -> tuple[list[dict], list[str]]:
    """Collect active tasks across all known projects.

    Returns ``(active_tasks, offline_projects)`` where *offline_projects* is
    the list of project labels whose MCP fetch failed.  The handler turns a
    non-empty *offline_projects* into ``offline: True`` on the dashboard payload.

    When *max_done_per_project* > 0, the most-recent N done tasks per project
    are appended to the returned list (each with a ``completed`` field).
    Default 0 leaves the return shape unchanged — scheduler.py is unaffected.

    *now* is forwarded to ``collect_tasks_with_counts`` so every row's
    ``started`` derives from a single shared instant; see that function's
    docstring for details. Defaults to the live clock.

    Lock state is surfaced via the scheduler endpoint — see
    /api/v2/dashboard/scheduler.

    Note: callers that also need per-project done counts should use
    ``collect_tasks_with_counts`` to avoid a second MCP round-trip.
    """
    active, offline, _ = await collect_tasks_with_counts(
        client, config,
        max_done_per_project=max_done_per_project,
        max_cancelled_per_project=max_cancelled_per_project,
        now=now,
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
    for root, result in zip(roots, results, strict=False):
        # Skip offline markers (dict with 'offline' key).
        if isinstance(result, dict) and result.get('offline'):
            continue
        label = _project_label(root)
        counts[label] = sum(1 for s in result.values() if s == 'done')
    return counts
