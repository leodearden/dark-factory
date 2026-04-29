"""Pure shape adapters that reshape existing aggregator outputs into the
``window.DF_DATA`` keys consumed by the redux React dashboard.

These functions are deliberately I/O-free and FastAPI-free so they can be
exercised in unit tests with synthetic aggregator output.  Routes in
``dashboard.app`` call the existing async aggregators, then pass their results
through the matching ``shape_*`` function before serialising as JSON.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ORCHESTRATORS + PROJECTS
# ---------------------------------------------------------------------------


def _project_label(value: str | Path) -> str:
    """Display label for a project root: directory basename, fallback to str."""
    s = str(value)
    name = s.rstrip('/').rsplit('/', 1)[-1]
    return name or s


def _current_task_for_orchestrator(orchestrator: Mapping[str, Any]) -> str:
    """Pick a representative current task title from an orchestrator's worktrees.

    Prefers a worktree in EXECUTE phase, then PLAN, then any.  Falls back to
    the en-dash placeholder when no worktrees exist.
    """
    worktrees = orchestrator.get('worktrees') or {}
    if not worktrees:
        return '—'

    def _entry_title(entry: Mapping[str, Any]) -> str:
        meta = entry.get('metadata') or {}
        tid = meta.get('task_id')
        title = meta.get('title') or ''
        if tid is not None:
            return f'T-{tid}: {title}' if title else f'T-{tid}'
        return title or '—'

    for desired in ('EXECUTE', 'PLAN'):
        for entry in worktrees.values():
            if entry.get('phase') == desired:
                return _entry_title(entry)
    # Fall back to any worktree.
    return _entry_title(next(iter(worktrees.values())))


def shape_orchestrators(
    orchestrators: Iterable[Mapping[str, Any]],
    *,
    known_project_roots: Iterable[str | Path] = (),
) -> dict[str, list]:
    """Return ``{ORCHESTRATORS: [...], PROJECTS: [...]}`` for the API.

    ``orchestrators`` is the raw list from
    :func:`dashboard.data.orchestrator.discover_orchestrators`.  ``known_project_roots``
    is the union of configured roots; projects without a running orchestrator
    are still surfaced (with ``active: False``).
    """
    orchestrators = list(orchestrators)
    out_orchs: list[dict] = []
    active_roots: set[str] = set()
    for o in orchestrators:
        pids = o.get('pids') or []
        primary_pid = pids[0] if pids else None
        project_root = o.get('project_root') or ''
        active_roots.add(str(project_root))
        out_orchs.append({
            'pid': primary_pid,
            'pids': list(pids),
            'label': o.get('label') or _project_label(project_root),
            'project': _project_label(project_root),
            'project_root': str(project_root),
            'running': bool(o.get('running')),
            'started': o.get('started') or '',
            'summary': dict(o.get('summary') or {}),
            'current_task': _current_task_for_orchestrator(o),
        })

    # PROJECTS: union of active roots + known roots.  ``active`` reflects
    # whether an orchestrator is currently running for that root.
    seen: set[str] = set()
    out_projects: list[dict] = []
    for root in (*active_roots, *(str(r) for r in known_project_roots)):
        if root in seen:
            continue
        seen.add(root)
        name = _project_label(root)
        out_projects.append({
            'id': name,
            'name': name,
            'active': root in active_roots,
        })

    return {'ORCHESTRATORS': out_orchs, 'PROJECTS': out_projects}


# ---------------------------------------------------------------------------
# MEMORY_STATUS  (also surfaces queue stats inline)
# ---------------------------------------------------------------------------


def shape_memory(status: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``{MEMORY_STATUS: {...}}`` matching the redux UI's read sites."""
    if status.get('offline'):
        return {'MEMORY_STATUS': {
            'graphiti': {'connected': False, 'node_count': 0, 'edge_count': 0, 'episode_count': 0},
            'mem0': {'connected': False, 'memory_count': 0},
            'taskmaster': {'connected': False},
            'queue': {'counts': dict(queue.get('counts') or {}),
                      'oldest_pending_age_seconds': queue.get('oldest_pending_age_seconds')},
            'projects': {},
            'offline': True,
            'error': status.get('error'),
        }}

    graphiti = dict(status.get('graphiti') or {})
    graphiti.setdefault('connected', True)
    for _key in ('node_count', 'edge_count', 'episode_count'):
        graphiti.setdefault(_key, 0)
    mem0 = dict(status.get('mem0') or {})
    mem0.setdefault('connected', True)
    mem0.setdefault('memory_count', 0)
    taskmaster = dict(status.get('taskmaster') or {})
    taskmaster.setdefault('connected', True)

    queue_counts = dict(queue.get('counts') or {})
    for _ckey in ('pending', 'retry', 'dead'):
        queue_counts.setdefault(_ckey, 0)
    queue_block = {
        'counts': queue_counts,
        'oldest_pending_age_seconds': queue.get('oldest_pending_age_seconds'),
    }
    if queue.get('offline'):
        queue_block['offline'] = True

    return {'MEMORY_STATUS': {
        'graphiti': graphiti,
        'mem0': mem0,
        'taskmaster': taskmaster,
        'queue': queue_block,
        'projects': dict(status.get('projects') or {}),
    }}


# ---------------------------------------------------------------------------
# MEMORY_TIMESERIES + MEMORY_OPS_BREAKDOWN
# ---------------------------------------------------------------------------


def shape_memory_graphs(
    timeseries: Mapping[str, Any], ops: Mapping[str, Any],
) -> dict[str, Any]:
    """Return ``{MEMORY_TIMESERIES, MEMORY_OPS_BREAKDOWN}``.

    ``timeseries`` is already in DF_DATA shape; ``ops`` is reshaped from
    ``{labels, values}`` to ``[{label, value}, ...]``.
    """
    ts_labels = list(timeseries.get('labels') or [])
    breakdown = [
        {'label': lbl, 'value': val}
        for lbl, val in zip(
            ops.get('labels') or [], ops.get('values') or [], strict=False,
        )
    ]
    return {
        'MEMORY_TIMESERIES': {
            'labels': ts_labels,
            'reads': list(timeseries.get('reads') or []),
            'writes': list(timeseries.get('writes') or []),
        },
        'MEMORY_OPS_BREAKDOWN': breakdown,
    }


# ---------------------------------------------------------------------------
# RECON_STATE + AGENTS
# ---------------------------------------------------------------------------


def shape_recon(
    *,
    buffer_stats: Mapping[str, Any],
    burst_state: Iterable[Mapping[str, Any]],
    watermarks: Iterable[Mapping[str, Any]],
    verdict: Mapping[str, Any] | None,
    runs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return ``{RECON_STATE: {...}, AGENTS: [...]}``.

    ``watermarks`` is converted from a list-of-dicts to a project-keyed dict.
    ``runs`` and ``burst_state`` are passed through as lists.  ``AGENTS`` is
    the distinct sorted set of ``agent_id`` values from ``burst_state``.
    """
    burst_list = list(burst_state)
    wm_map: dict[str, dict] = {}
    for wm in watermarks:
        pid = wm.get('project_id')
        if not pid:
            continue
        wm_map[pid] = {k: v for k, v in wm.items() if k != 'project_id'}

    runs_list = [dict(r) for r in runs]
    agents = sorted({b['agent_id'] for b in burst_list if b.get('agent_id')})

    return {
        'RECON_STATE': {
            'buffer': dict(buffer_stats),
            'burst_state': [dict(b) for b in burst_list],
            'watermarks': wm_map,
            'verdict': dict(verdict) if verdict else None,
            'runs': runs_list,
        },
        'AGENTS': agents,
    }


# ---------------------------------------------------------------------------
# MERGE_QUEUE
# ---------------------------------------------------------------------------


def shape_merge_queue(per_project: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return ``{MERGE_QUEUE: {project_label: {...}}}``.

    The aggregator already returns one entry per project; we relabel the keys
    from absolute project_root paths to short basenames so the React side can
    match against ``PROJECTS[].name``.  The shape per-project follows the
    DF_DATA mock: ``depth`` (renamed from ``depth_timeseries``), ``outcomes``,
    ``latency``, ``recent``, ``speculative``, ``active``.
    """
    out: dict[str, dict] = {}
    for pid, data in per_project.items():
        out[_project_label(pid)] = {
            'depth': dict(data.get('depth_timeseries') or {'labels': [], 'values': []}),
            'outcomes': dict(data.get('outcomes') or {'labels': [], 'values': []}),
            'latency': dict(data.get('latency') or {}),
            'recent': [dict(r) for r in (data.get('recent') or [])],
            'speculative': dict(data.get('speculative') or {}),
            'active': [dict(a) for a in (data.get('active') or [])],
        }
    return {'MERGE_QUEUE': out}


# ---------------------------------------------------------------------------
# COSTS
# ---------------------------------------------------------------------------


def _sum_models(models: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Reduce ``[{model, total}, ...]`` to ``{model: total}``."""
    acc: dict[str, float] = {}
    for m in models:
        name = m.get('model') or 'unknown'
        acc[name] = acc.get(name, 0.0) + float(m.get('total') or 0.0)
    return acc


def shape_costs(
    *,
    summary: Mapping[str, Mapping[str, Any]],
    by_project: Mapping[str, Iterable[Mapping[str, Any]]],
    by_account: Mapping[str, Mapping[str, Any]],
    by_role: Mapping[str, Mapping[str, Mapping[str, float]]],
    trend: Mapping[str, Iterable[Mapping[str, Any]]],
    events: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce per-project aggregator output to the flat ``COSTS`` shape.

    The mock data in ``data.js`` flattens summaries to a single org-wide row
    plus per-project / per-account / per-role lists.  We do the same:
    ``summary`` and ``trend`` sum across projects; ``by_project``,
    ``by_account``, and ``by_role`` become sorted lists ready for chart
    rendering.
    """
    # ── summary ──
    total = sum(float(p.get('total_spend') or 0.0) for p in summary.values())
    runs = sum(int(p.get('task_count') or 0) for p in summary.values())
    by_project_list: list[dict] = []
    for pid, models in by_project.items():
        models_list = list(models)
        sums = _sum_models(models_list)
        by_project_list.append({
            'project': _project_label(pid),
            'total': round(sum(sums.values()), 4),
            **{k: round(v, 4) for k, v in sums.items()},
        })
    by_project_list.sort(key=lambda r: r['total'], reverse=True)

    # ── by_account ──
    grand_account_total = sum(float(v.get('spend') or 0.0) for v in by_account.values()) or 1.0
    by_account_list = [
        {
            'account': acct,
            'total': round(float(info.get('spend') or 0.0), 4),
            'status': info.get('status') or 'active',
            'resets_at': info.get('resets_at'),
            'share': round(float(info.get('spend') or 0.0) / grand_account_total * 100, 1),
        }
        for acct, info in by_account.items()
    ]
    by_account_list.sort(key=lambda r: r['total'], reverse=True)

    # ── by_role ── (sum across projects + models)
    role_totals: dict[str, float] = {}
    for project_roles in by_role.values():
        for role, models in project_roles.items():
            role_totals[role] = role_totals.get(role, 0.0) + sum(
                float(v) for v in (models or {}).values()
            )
    grand_role_total = sum(role_totals.values()) or 1.0
    by_role_list = [
        {
            'role': role,
            'total': round(t, 4),
            'share': round(t / grand_role_total * 100, 1),
        }
        for role, t in role_totals.items()
    ]
    by_role_list.sort(key=lambda r: r['total'], reverse=True)

    # ── trend ── (sum per day across projects, preserve chronological order)
    day_totals: dict[str, float] = {}
    for series in trend.values():
        for entry in series:
            day = entry.get('day')
            if not day:
                continue
            day_totals[day] = day_totals.get(day, 0.0) + float(entry.get('total') or 0.0)
    sorted_days = sorted(day_totals)
    trend_values = [round(day_totals[d], 4) for d in sorted_days]
    trend_block = {
        'labels': sorted_days,
        'values': trend_values,
    }
    # Day-over-day delta from the tail of the trend.  Returns None when there
    # isn't enough history to compute it (consumers render '—').
    if len(trend_values) >= 2 and trend_values[-2] > 0:
        delta_pct = round(
            (trend_values[-1] - trend_values[-2]) / trend_values[-2] * 100, 1,
        )
    else:
        delta_pct = None

    # ── events ── (already a flat list, normalise field names)
    events_list = [
        {
            'ts': ev.get('created_at') or '',
            'account': ev.get('account_name') or '',
            'event': ev.get('event_type') or '',
            'detail': ev.get('details') or '',
            'project': ev.get('project_id'),
        }
        for ev in events
    ]

    # tokens / p95_run_cost: not aggregated server-side yet.  Returned as None
    # so the UI can render '—' rather than a misleading 0.  delta_pct comes
    # from the trend tail above.
    summary_block = {
        'total': round(total, 4),
        'runs': runs,
        'today': trend_values[-1] if trend_values else 0.0,
        'tokens': None,
        'p95_run_cost': None,
        'delta_pct': delta_pct,
    }

    return {'COSTS': {
        'summary': summary_block,
        'by_project': by_project_list,
        'by_account': by_account_list,
        'by_role': by_role_list,
        'trend': trend_block,
        'events': events_list,
    }}


# ---------------------------------------------------------------------------
# PERFORMANCE
# ---------------------------------------------------------------------------


def shape_performance(
    *,
    paths: Mapping[str, Iterable[Mapping[str, Any]]],
    escalations: Mapping[str, Mapping[str, Any]],
    histograms: Mapping[str, Mapping[str, Any]],
    ttc: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine the four performance aggregators into the per-project shape.

    Output: ``{PERFORMANCE: {project_label: {paths, escalation, hist_outer,
    hist_inner, ttc}}}``.
    """
    project_ids = set(paths) | set(escalations) | set(histograms) | set(ttc)
    out: dict[str, dict] = {}
    for pid in project_ids:
        hist = histograms.get(pid) or {}
        out[_project_label(pid)] = {
            'paths': [dict(e) for e in (paths.get(pid) or [])],
            'escalation': dict(escalations.get(pid) or {}),
            'hist_outer': dict(hist.get('outer') or {'labels': [], 'values': []}),
            'hist_inner': dict(hist.get('inner') or {'labels': [], 'values': []}),
            'ttc': dict(ttc.get(pid) or {}),
        }
    return {'PERFORMANCE': out}


# ---------------------------------------------------------------------------
# BURNDOWN + BURNDOWN_BY_PROJECT
# ---------------------------------------------------------------------------


_BURNDOWN_KEYS = ('done', 'in_progress', 'blocked', 'pending')


def shape_burndown(
    series_by_project: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build ``{BURNDOWN, BURNDOWN_BY_PROJECT}`` from per-project series.

    ``BURNDOWN`` is the sum across projects, aligned by label.
    ``BURNDOWN_BY_PROJECT`` is keyed by project basename.
    """
    by_project: dict[str, dict] = {}
    label_set: set[str] = set()
    for pid, series in series_by_project.items():
        labels = list(series.get('labels') or [])
        label_set.update(labels)
        by_project[_project_label(pid)] = {
            'labels': labels,
            **{k: list(series.get(k) or []) for k in _BURNDOWN_KEYS},
        }

    sorted_labels = sorted(label_set)
    aggregate: dict[str, list] = {'labels': sorted_labels, **{k: [0] * len(sorted_labels) for k in _BURNDOWN_KEYS}}
    for series in series_by_project.values():
        labels = list(series.get('labels') or [])
        index_map = {lbl: i for i, lbl in enumerate(sorted_labels)}
        for k in _BURNDOWN_KEYS:
            for lbl, val in zip(labels, series.get(k) or [], strict=False):
                if lbl in index_map:
                    aggregate[k][index_map[lbl]] += int(val or 0)

    return {'BURNDOWN': aggregate, 'BURNDOWN_BY_PROJECT': by_project}
