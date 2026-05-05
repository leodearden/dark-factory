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

from dashboard.data.stats_utils import percentile


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
            return f'{tid}: {title}' if title else f'{tid}'
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
    running_spark: Mapping[str, list] | None = None,
) -> dict[str, list]:
    """Return ``{ORCHESTRATORS: [...], PROJECTS: [...]}`` for the API.

    ``orchestrators`` is the raw list from
    :func:`dashboard.data.orchestrator.discover_orchestrators`.  ``known_project_roots``
    is the union of configured roots; projects without a running orchestrator
    are still surfaced (with ``active: False``).  ``running_spark`` is an
    optional ``{labels, values}`` time-series of the total running-orchestrator
    count over the last day, surfaced as ``ORCHESTRATORS_SPARK``.
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

    spark_block = (
        {'labels': list(running_spark.get('labels') or []),
         'values': list(running_spark.get('values') or [])}
        if running_spark else {'labels': [], 'values': []}
    )

    return {
        'ORCHESTRATORS': out_orchs,
        'PROJECTS': out_projects,
        'ORCHESTRATORS_SPARK': spark_block,
    }


# ---------------------------------------------------------------------------
# MEMORY_STATUS  (also surfaces queue stats inline)
# ---------------------------------------------------------------------------


_EMPTY_SERIES: dict[str, list] = {'labels': [], 'values': []}


def _empty_series_dict() -> dict[str, list]:
    return {'labels': [], 'values': []}


def shape_memory(
    status: Mapping[str, Any],
    queue: Mapping[str, Any],
    *,
    sparks: Mapping[str, Mapping[str, list]] | None = None,
    queue_spark: Mapping[str, list] | None = None,
    delta_24h: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return ``{MEMORY_STATUS: {...}}`` matching the redux UI's read sites.

    ``sparks`` (optional) carries ``{graphiti_nodes, mem0_memories}`` time
    series; ``queue_spark`` carries the pending-queue series; ``delta_24h``
    carries per-project ``{graphiti_nodes, mem0_memories}`` snapshots from
    ~24h ago for delta rendering.  All three are populated from
    metrics.db; absent / empty when history is sparse.
    """
    spark_g = (sparks or {}).get('graphiti_nodes') or _EMPTY_SERIES
    spark_m = (sparks or {}).get('mem0_memories') or _EMPTY_SERIES
    spark_q = queue_spark or _EMPTY_SERIES
    delta_map = delta_24h or {}

    def _project_block(pid: str, payload: Mapping[str, Any]) -> dict:
        before = delta_map.get(pid) or {}
        return {
            **dict(payload),
            'graphiti_nodes_24h_ago': before.get('graphiti_nodes'),
            'mem0_memories_24h_ago': before.get('mem0_memories'),
        }

    if status.get('offline'):
        return {'MEMORY_STATUS': {
            'graphiti': {'connected': False, 'node_count': 0, 'edge_count': 0, 'episode_count': 0,
                         'spark': _empty_series_dict()},
            'mem0': {'connected': False, 'memory_count': 0, 'spark': _empty_series_dict()},
            'taskmaster': {'connected': False},
            'queue': {'counts': dict(queue.get('counts') or {}),
                      'oldest_pending_age_seconds': queue.get('oldest_pending_age_seconds'),
                      'spark': _empty_series_dict()},
            'projects': {},
            'offline': True,
            'error': status.get('error'),
        }}

    graphiti = dict(status.get('graphiti') or {})
    graphiti.setdefault('connected', True)
    for _key in ('node_count', 'edge_count', 'episode_count'):
        graphiti.setdefault(_key, 0)
    graphiti['spark'] = {
        'labels': list(spark_g.get('labels') or []),
        'values': list(spark_g.get('values') or []),
    }
    mem0 = dict(status.get('mem0') or {})
    mem0.setdefault('connected', True)
    mem0.setdefault('memory_count', 0)
    mem0['spark'] = {
        'labels': list(spark_m.get('labels') or []),
        'values': list(spark_m.get('values') or []),
    }
    taskmaster = dict(status.get('taskmaster') or {})
    taskmaster.setdefault('connected', True)

    queue_counts = dict(queue.get('counts') or {})
    for _ckey in ('pending', 'retry', 'dead'):
        queue_counts.setdefault(_ckey, 0)
    queue_block = {
        'counts': queue_counts,
        'oldest_pending_age_seconds': queue.get('oldest_pending_age_seconds'),
        'spark': {
            'labels': list(spark_q.get('labels') or []),
            'values': list(spark_q.get('values') or []),
        },
    }
    if queue.get('offline'):
        queue_block['offline'] = True

    raw_projects = dict(status.get('projects') or {})
    enriched_projects = {pid: _project_block(pid, payload) for pid, payload in raw_projects.items()}

    return {'MEMORY_STATUS': {
        'graphiti': graphiti,
        'mem0': mem0,
        'taskmaster': taskmaster,
        'queue': queue_block,
        'projects': enriched_projects,
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
    sparks: Mapping[str, Mapping[str, list]] | None = None,
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

    sparks = sparks or {}
    buffered_spark = sparks.get('buffered_count') or _EMPTY_SERIES
    agents_spark = sparks.get('active_agents') or _EMPTY_SERIES

    buffer_block = {**dict(buffer_stats), 'spark': {
        'labels': list(buffered_spark.get('labels') or []),
        'values': list(buffered_spark.get('values') or []),
    }}

    return {
        'RECON_STATE': {
            'buffer': buffer_block,
            'burst_state': [dict(b) for b in burst_list],
            'watermarks': wm_map,
            'verdict': dict(verdict) if verdict else None,
            'runs': runs_list,
            'agents_spark': {
                'labels': list(agents_spark.get('labels') or []),
                'values': list(agents_spark.get('values') or []),
            },
        },
        'AGENTS': agents,
    }


# ---------------------------------------------------------------------------
# MERGE_QUEUE
# ---------------------------------------------------------------------------


def shape_merge_queue(
    per_project: Mapping[str, Mapping[str, Any]],
    *,
    active_sparks: Mapping[str, Mapping[str, list]] | None = None,
    halt_status: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return ``{MERGE_QUEUE: {project_label: {...}}}``.

    The aggregator already returns one entry per project; we relabel the keys
    from absolute project_root paths to short basenames so the React side can
    match against ``PROJECTS[].name``.  The shape per-project follows the
    DF_DATA mock: ``depth`` (renamed from ``depth_timeseries``), ``outcomes``,
    ``latency``, ``recent``, ``speculative``, ``active``, ``active_spark``,
    ``halt``.

    ``active_sparks`` (optional) carries true active-queue depth over time
    keyed by absolute project_root path; surfaced as ``active_spark`` per
    project label so the UI tile no longer falls back to attempt-count.

    ``halt_status`` (optional) is keyed by project basename and carries
    ``{wired, halted, owner_esc_id, offline}`` per orchestrator. Missing
    projects fall back to ``{offline: True}`` so the UI can render an Offline
    pill on every panel.
    """
    sparks = active_sparks or {}
    halts = halt_status or {}
    out: dict[str, dict] = {}
    for pid, data in per_project.items():
        spark = sparks.get(pid) or _EMPTY_SERIES
        label = _project_label(pid)
        out[label] = {
            'depth': dict(data.get('depth_timeseries') or {'labels': [], 'values': []}),
            'outcomes': dict(data.get('outcomes') or {'labels': [], 'values': []}),
            'latency': dict(data.get('latency') or {}),
            'recent': [dict(r) for r in (data.get('recent') or [])],
            'speculative': dict(data.get('speculative') or {}),
            'active': [dict(a) for a in (data.get('active') or [])],
            'active_spark': {
                'labels': list(spark.get('labels') or []),
                'values': list(spark.get('values') or []),
            },
            'halt': dict(halts.get(label) or {'offline': True}),
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

    # Global tokens: component-wise sum across projects.  Returns None when no
    # project reports tokens (UI renders '—' rather than a misleading 0).
    global_tokens: dict[str, int] = {
        'input': 0, 'output': 0, 'cache_read': 0, 'cache_create': 0, 'total': 0,
    }
    saw_tokens = False
    for p in summary.values():
        t = p.get('tokens') if isinstance(p, Mapping) else None
        if not isinstance(t, Mapping):
            continue
        saw_tokens = True
        for key in global_tokens:
            global_tokens[key] += int(t.get(key) or 0)
    tokens_block = global_tokens if saw_tokens else None

    # Global p95: concatenate per-project run_costs and recompute, so
    # cross-project distribution drives the org-wide percentile rather than
    # averaging per-project p95s.
    all_run_costs: list[float] = []
    for p in summary.values():
        rc = p.get('run_costs') if isinstance(p, Mapping) else None
        if isinstance(rc, list):
            all_run_costs.extend(float(c) for c in rc if c is not None)
    if all_run_costs:
        all_run_costs.sort()
        p95_run_cost = round(percentile(all_run_costs, 95), 4)
    else:
        p95_run_cost = None

    # Hint surfaces the cause of a missing delta so the UI's '—' is
    # interpretable.  Empty when delta_pct is computable.
    delta_hint = (
        'need ≥2 days for delta'
        if delta_pct is None and len(trend_values) <= 1 and trend_values
        else None
    )

    summary_block = {
        'total': round(total, 4),
        'runs': runs,
        'today': trend_values[-1] if trend_values else 0.0,
        'tokens': tokens_block,
        'p95_run_cost': p95_run_cost,
        'delta_pct': delta_pct,
        'delta_hint': delta_hint,
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
    history: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Combine the four performance aggregators into the per-project shape.

    Output: ``{PERFORMANCE: {project_label: {paths, escalation, hist_outer,
    hist_inner, ttc, time_centiles_history, one_pass_history,
    escalation_history}}}``.

    ``history`` (optional) carries per-project hour-bucketed histories from
    :func:`dashboard.data.performance.aggregate_performance_history`. When
    absent or sparse, the history blocks render as empty {labels, values}.
    """
    project_ids = set(paths) | set(escalations) | set(histograms) | set(ttc) | set(history or {})
    history = history or {}
    empty_pair = {'labels': [], 'values': []}
    empty_centiles = {'labels': [], 'p50': [], 'p95': []}
    out: dict[str, dict] = {}
    for pid in project_ids:
        hist = histograms.get(pid) or {}
        h = history.get(pid) or {}
        out[_project_label(pid)] = {
            'paths': [dict(e) for e in (paths.get(pid) or [])],
            'escalation': dict(escalations.get(pid) or {}),
            'hist_outer': dict(hist.get('outer') or {'labels': [], 'values': []}),
            'hist_inner': dict(hist.get('inner') or {'labels': [], 'values': []}),
            'ttc': dict(ttc.get(pid) or {}),
            'time_centiles_history': dict(h.get('time_centiles_history') or empty_centiles),
            'one_pass_history': dict(h.get('one_pass_history') or empty_pair),
            'escalation_history': dict(h.get('escalation_history') or empty_pair),
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
    ``BURNDOWN_BY_PROJECT`` is keyed by project basename.  Both blocks
    carry ``forecast_low`` / ``forecast_high`` (None when <7 days history).
    """
    # Local import avoids circular import: burndown.py imports stats and
    # this module imports config/no-burndown.
    from dashboard.data.burndown import compute_forecast_confidence

    by_project: dict[str, dict] = {}
    label_set: set[str] = set()
    for pid, series in series_by_project.items():
        labels = list(series.get('labels') or [])
        label_set.update(labels)
        forecast = compute_forecast_confidence(series)
        by_project[_project_label(pid)] = {
            'labels': labels,
            **{k: list(series.get(k) or []) for k in _BURNDOWN_KEYS},
            **forecast,
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

    aggregate.update(compute_forecast_confidence(aggregate))

    return {'BURNDOWN': aggregate, 'BURNDOWN_BY_PROJECT': by_project}
