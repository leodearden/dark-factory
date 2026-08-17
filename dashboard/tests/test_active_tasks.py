"""Tests for the ACTIVE_TASKS aggregator that joins task tree + worktrees."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import (
    _build_task_row,
    _minutes_since,
    collect_active_tasks,
    collect_done_counts,
    collect_tasks_with_counts,
)

# ---------------------------------------------------------------------------
# helpers used inside the aggregator
# ---------------------------------------------------------------------------


def test_minutes_since_handles_z_suffix_and_naive_iso():
    one_hour_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace('+00:00', 'Z')
    assert 59 <= _minutes_since(one_hour_ago) <= 61


def test_minutes_since_returns_none_on_missing_and_zero_on_bad():
    """A MISSING start time is an honest unknown (None); a bad one keeps 0.

    ``None``/``''`` is the per-task artifact-read-failure signal on
    ``TaskRuntimeEntry.started`` (see ``shared/src/shared/task_runtime_state.py``
    — "never a fabricated 0"), so the helper must propagate the unknown rather
    than render it as '0m running'. A present-but-unparseable timestamp is a
    different failure (upstream data damage, no known producer) and
    deliberately keeps the existing 0 — out of scope for task 4055, see the
    plan's design decisions.
    """
    assert _minutes_since(None) is None
    assert _minutes_since('') is None
    assert _minutes_since('not-a-date') == 0


def test_minutes_since_uses_provided_now():
    """_minutes_since(iso, now=fixed) derives its result from the passed now, not the clock."""
    fixed = datetime(2026, 4, 11, 12, 30, 0, tzinfo=UTC)
    ts = fixed - timedelta(minutes=37, seconds=10)
    expected = int((fixed - ts).total_seconds() // 60)
    assert _minutes_since(ts.isoformat(), now=fixed) == expected


def test_minutes_since_no_now_resolves_via_clock():
    """Without now, _minutes_since still resolves via resolve_now (the live clock).

    Brackets the real clock read with before/after captures rather than
    patching a module-level ``datetime`` symbol, mirroring
    ``Test_Cutoff.test_cutoff_no_now_uses_current_time`` in test_costs_data.py.
    """
    ts = datetime.now(UTC) - timedelta(minutes=10)
    before = datetime.now(UTC)
    result = _minutes_since(ts.isoformat())
    after = datetime.now(UTC)

    lower = int((before - ts).total_seconds() // 60)
    upper = int((after - ts).total_seconds() // 60)
    assert lower <= result <= upper


# ---------------------------------------------------------------------------
# collect_active_tasks against in-memory MCP-shaped fixture
# ---------------------------------------------------------------------------


def _shape_task(task: dict) -> dict:
    """Build a row in the dashboard's per-task wire shape.

    Mirrors tasks.py::_shape_task — must include ``updated_at`` so that
    done-task ordering tests work correctly.
    """
    return {
        'id': int(task['id']),
        'title': task.get('title') or '',
        'description': task.get('description') or '',
        'details': task.get('details') or '',
        'status': task.get('status'),
        'priority': task.get('priority'),
        'dependencies': list(task.get('dependencies') or []),
        'metadata': task.get('metadata', {}),
        'updated_at': task.get('updated_at'),
    }


def _make_project(root, *, project_dir, tasks):
    """Create a project root dir and return ``(project_root, shaped_tasks)``.

    The tasks themselves no longer live on disk — fused-memory MCP owns task
    state — so we return them in their dashboard-shaped form for the caller
    to register against ``fetch_tasks`` via monkeypatch. Per-task runtime
    state (loops/attempts/agent/lane/phase/lane_state) is likewise sourced
    over MCP now, not read from a ``.worktrees/.task`` artifact tree — see
    ``_runtime_entry``/``_register_runtime`` below.
    """
    project_root = root / project_dir
    project_root.mkdir(parents=True, exist_ok=True)
    return project_root, [_shape_task(t) for t in tasks]


def _runtime_entry(task_id: int, **overrides) -> TaskRuntimeEntry:
    """A ``TaskRuntimeEntry`` with sane defaults; ``overrides`` replace fields."""
    base: dict[str, Any] = dict(
        task_id=task_id,
        has_worktree=True,
        loops=0,
        attempts=0,
        started=None,
        lane=None,
        phase=None,
        lane_state=None,
        error=None,
    )
    base.update(overrides)
    return TaskRuntimeEntry(**base)


def _register_runtime(monkeypatch, mapping: dict[str, list[TaskRuntimeEntry]]) -> None:
    """Monkeypatch ``fetch_task_runtime`` to return a fixed ``{label: TaskRuntimeSnapshot}``.

    *mapping* maps project label -> list of ``TaskRuntimeEntry``. A label
    absent from *mapping* is simply absent from the returned dict (mirroring
    a real fan-out that only covers configured ``escalation_urls``) — tests
    that need an explicit "online but empty" snapshot for a label must
    include it with an empty list.
    """
    snapshots = {label: TaskRuntimeSnapshot(tasks=entries) for label, entries in mapping.items()}

    async def _fake_fetch_task_runtime(client, escalation_urls):
        return dict(snapshots)

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_task_runtime', _fake_fetch_task_runtime)


def _register_fetch_tasks(monkeypatch, fetch) -> None:
    """Register a full-tree *fetch* as BOTH narrowed ``fetch_tasks`` and ``fetch_statuses``.

    ``_shape_one_project`` no longer issues one unnarrowed fetch (task 3857).
    It asks for active rows and, separately, a bounded window of terminal
    rows, and it reads its ``done_count`` from the compact ``fetch_statuses``
    map.  A fake that ignored ``statuses`` would hand the whole tree to BOTH
    ``fetch_tasks`` calls and duplicate every row; one that left
    ``fetch_statuses`` unpatched would reach for the network.

    So the wrapper emulates exactly what the substrate does — a ``statuses``
    row filter, then a ``page_size``/``offset`` slice over an ASCENDING-id
    list — and derives the compact map from the same canned tree.  Tests here
    are about SHAPING; the wire contract itself is asserted against a canned
    ``mcp_tool_call`` in ``TestShapeOneProjectNarrowing``.

    *fetch* keeps its original ``(client, config, project_root)`` signature and
    may still return an offline marker dict, which is propagated unchanged.
    """

    async def _narrowed(
        client, config, project_root, *,
        statuses=None, page_size=None, offset=0, timeout=None,
    ):
        rows = await fetch(client, config, project_root)
        if not isinstance(rows, list):
            return rows
        if statuses is not None:
            rows = [r for r in rows if r.get('status') in statuses]
        rows = sorted(rows, key=lambda r: r.get('id') or 0)  # ORDER BY id ASC
        if page_size is not None:
            rows = rows[offset:offset + page_size]
        return rows

    async def _statuses(client, config, project_root):
        rows = await fetch(client, config, project_root)
        if not isinstance(rows, list):
            return {'offline': True, 'error': 'task fetch offline'}
        return {
            r['id']: r.get('status') for r in rows if isinstance(r.get('id'), int)
        }

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _narrowed)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_statuses', _statuses)


@pytest.fixture()
def two_project_config(tmp_path, monkeypatch):
    """Two-project layout with shaped task lists registered against fetch_tasks."""
    started = (datetime.now(UTC) - timedelta(minutes=14)).isoformat()
    df_root, df_tasks = _make_project(
        tmp_path,
        project_dir='dark-factory',
        tasks=[
            {'id': 19, 'title': 'consolidation retry', 'status': 'in-progress',
             'dependencies': [15, 17],
             'metadata': {'files': ['src/agents/consolidation.py', 'src/store/graphiti_adapter.py']}},
            {'id': 17, 'title': 'pre-filter', 'status': 'done', 'dependencies': []},
            {'id': 15, 'title': 'partitioning', 'status': 'done', 'dependencies': []},
            {'id': 23, 'title': 'collision', 'status': 'pending', 'dependencies': [21]},
            {'id': 21, 'title': 'dedup index', 'status': 'in-progress',
             'dependencies': []},
        ],
    )
    reify_root, reify_tasks = _make_project(
        tmp_path,
        project_dir='reify',
        tasks=[{'id': 8, 'title': 'parser recovery', 'status': 'blocked',
                'dependencies': []}],
    )

    by_root = {df_root.resolve(): df_tasks, reify_root.resolve(): reify_tasks}

    async def _fake_fetch_tasks(client, config, project_root):
        return list(by_root.get(project_root.resolve(), []))

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    _register_runtime(monkeypatch, {
        'dark-factory': [
            # 1/3 reviews passed -> attempts == 3 (total review count, not pass count)
            _runtime_entry(19, loops=2, attempts=3, started=started),
            _runtime_entry(21, loops=1, attempts=1, started=started),
        ],
        'reify': [
            _runtime_entry(8, loops=0, attempts=0, started=started),
        ],
    })

    return DashboardConfig(project_root=df_root, known_project_roots=[reify_root])


@pytest.mark.asyncio
async def test_collect_active_tasks_filters_to_active_statuses(two_project_config, dummy_client):
    active, _ = await collect_active_tasks(client=dummy_client, config=two_project_config)
    statuses = {t['status'] for t in active}
    assert statuses <= {'in-progress', 'blocked', 'pending'}
    # Done tasks (17, 15) should not appear.
    ids = {t['id'] for t in active}
    assert 'dark-factory/T-17' not in ids
    assert 'dark-factory/T-15' not in ids


@pytest.mark.asyncio
async def test_collect_active_tasks_resolves_deps_with_done_flags(two_project_config, dummy_client):
    active, _ = await collect_active_tasks(client=dummy_client, config=two_project_config)
    by_id = {t['id']: t for t in active}
    t19 = by_id['dark-factory/T-19']
    assert {d['id']: d['done'] for d in t19['deps']} == {
        'dark-factory/T-15': True,
        'dark-factory/T-17': True,
    }
    t23 = by_id['dark-factory/T-23']
    # T-21 is in-progress, not done
    assert t23['deps'] == [{'id': 'dark-factory/T-21', 'title': 'dedup index', 'done': False}]


@pytest.mark.asyncio
async def test_collect_active_tasks_pulls_metadata_and_loops(two_project_config, dummy_client):
    active, _ = await collect_active_tasks(client=dummy_client, config=two_project_config)
    t19 = next(t for t in active if t['id'] == 'dark-factory/T-19')
    assert t19['agent'] == 'claude-task-19'
    assert t19['loops'] == 2
    assert t19['attempts'] == 3
    # `started` is the minutes-since difference, allow a small slack vs 14.
    assert 13 <= t19['started'] <= 15
    # meta_files is the module lock source used by the scheduler pipeline.
    assert 'src/agents/consolidation.py' in t19['meta_files']


@pytest.mark.asyncio
async def test_collect_active_tasks_started_uses_provided_now(tmp_path, monkeypatch, dummy_client):
    """collect_active_tasks(now=fixed) computes every row's `started` against that one instant."""
    fixed = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
    created_at = (fixed - timedelta(minutes=42)).isoformat()
    root, shaped = _make_project(
        tmp_path,
        project_dir='fixedclock',
        tasks=[{'id': 1, 'title': 'a', 'status': 'in-progress', 'dependencies': []}],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    _register_runtime(monkeypatch, {'fixedclock': [_runtime_entry(1, started=created_at)]})
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg, now=fixed)
    assert len(active) == 1
    assert active[0]['started'] == 42


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_started_uses_provided_now_across_projects(
    tmp_path, monkeypatch, dummy_client,
):
    """collect_tasks_with_counts(now=fixed) shares ONE now across every project's rows.

    Two projects, each with a task started at a different known offset from the
    same fixed instant — proves `now` is resolved once at the aggregation
    boundary and threaded down, not re-read per project.
    """
    fixed = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
    df_created = (fixed - timedelta(minutes=10)).isoformat()
    reify_created = (fixed - timedelta(minutes=25)).isoformat()
    df_root, df_tasks = _make_project(
        tmp_path,
        project_dir='df',
        tasks=[{'id': 1, 'title': 'a', 'status': 'in-progress', 'dependencies': []}],
    )
    reify_root, reify_tasks = _make_project(
        tmp_path,
        project_dir='reify',
        tasks=[{'id': 2, 'title': 'b', 'status': 'pending', 'dependencies': []}],
    )
    by_root = {df_root.resolve(): df_tasks, reify_root.resolve(): reify_tasks}

    async def _fake_fetch_tasks(client, config, project_root):
        return list(by_root.get(project_root.resolve(), []))

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    _register_runtime(monkeypatch, {
        'df': [_runtime_entry(1, started=df_created)],
        'reify': [_runtime_entry(2, started=reify_created)],
    })
    cfg = DashboardConfig(project_root=df_root, known_project_roots=[reify_root])

    active, _, _, _, _ = await collect_tasks_with_counts(client=dummy_client, config=cfg, now=fixed)
    started_by_id = {t['id']: t['started'] for t in active}
    assert started_by_id == {'df/T-1': 10, 'reify/T-2': 25}


@pytest.mark.asyncio
async def test_collect_active_tasks_handles_missing_worktree_metadata(tmp_path, monkeypatch, dummy_client):
    """A pending task absent from an ONLINE runtime map still appears, with honest zeros."""
    root, shaped = _make_project(
        tmp_path, project_dir='solo',
        tasks=[{'id': 1, 'title': 'lonely', 'status': 'pending', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    # Project is online (a snapshot is registered for its label) but the
    # snapshot carries no entry for task 1 — the honest-zero case, distinct
    # from an offline project (see the runtime-offline tests below).
    _register_runtime(monkeypatch, {'solo': []})
    cfg = DashboardConfig(project_root=root)
    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert active == [{
        'id': 'solo/T-1', 'project': 'solo', 'title': 'lonely',
        'description': '', 'details': '', 'status': 'pending', 'agent': None,
        'started': 0, 'loops': 0, 'attempts': 0, 'deps': [],
        'meta_files': [], 'train': None, 'external_deps': [], 'prd': None,
        'lane': None, 'phase': None, 'lane_state': None, 'runtime_offline': False,
        # Claim projection (task 3543): carried on every row. A 'pending' task
        # is never stranded — the shared predicate gates on 'in-progress'.
        'claimant_run_id': None, 'heartbeat_at': None, 'stranded': False,
    }]


@pytest.mark.asyncio
async def test_collect_active_tasks_surfaces_offline_projects(tmp_path, monkeypatch, dummy_client):
    """A project whose MCP fetch returns an offline marker is reported."""
    root = tmp_path / 'offline-project'
    root.mkdir()

    async def _fake_fetch_tasks(client, config, project_root):
        return {'offline': True, 'error': 'connection refused'}

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    cfg = DashboardConfig(project_root=root)
    active, offline_projects = await collect_active_tasks(client=dummy_client, config=cfg)
    assert active == []
    assert offline_projects == ['offline-project']


# ---------------------------------------------------------------------------
# get_task_runtime_state MCP join (task 2636 step-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_active_tasks_runtime_join_populates_lane_phase_lane_state(
    tmp_path, monkeypatch, dummy_client,
):
    """An online runtime entry's loops/attempts/lane/phase/lane_state/agent all
    join onto the row; started is computed via _minutes_since against `now`.
    """
    fixed = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)
    entry_started = (fixed - timedelta(minutes=7)).isoformat()
    root, shaped = _make_project(
        tmp_path, project_dir='warmlane',
        tasks=[{'id': 42, 'title': 'warm task', 'status': 'in-progress', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    _register_runtime(monkeypatch, {
        'warmlane': [_runtime_entry(
            42, loops=3, attempts=1, started=entry_started,
            lane='_lane-7', phase='EXECUTE', lane_state='assigned',
        )],
    })
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg, now=fixed)
    assert len(active) == 1
    row = active[0]
    assert row['loops'] == 3
    assert row['attempts'] == 1
    assert row['agent'] == 'claude-task-42'
    assert row['started'] == 7
    assert row['lane'] == '_lane-7'
    assert row['phase'] == 'EXECUTE'
    assert row['lane_state'] == 'assigned'
    assert row['runtime_offline'] is False


@pytest.mark.asyncio
async def test_collect_active_tasks_runtime_offline_snapshot_yields_all_none(
    tmp_path, monkeypatch, dummy_client,
):
    """A project whose runtime snapshot reports offline=True gets honest None
    fields (never a fabricated 0), with runtime_offline=True.
    """
    root, shaped = _make_project(
        tmp_path, project_dir='downlane',
        tasks=[{'id': 5, 'title': 'stuck task', 'status': 'in-progress', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    async def _offline_fetch_task_runtime(client, escalation_urls):
        return {'downlane': TaskRuntimeSnapshot(offline=True)}

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_task_runtime', _offline_fetch_task_runtime)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert len(active) == 1
    row = active[0]
    for key in ('agent', 'loops', 'attempts', 'started', 'lane', 'phase', 'lane_state'):
        assert row[key] is None, f'expected {key}=None when runtime offline, got {row[key]!r}'
    assert row['runtime_offline'] is True


@pytest.mark.asyncio
async def test_collect_active_tasks_no_escalation_url_treated_as_offline(
    tmp_path, monkeypatch, dummy_client,
):
    """A project absent from the runtime map entirely (no escalation URL
    configured for it) is treated identically to an explicit offline=True
    snapshot — we genuinely have no runtime source for it either way.
    """
    root, shaped = _make_project(
        tmp_path, project_dir='nourl',
        tasks=[{'id': 6, 'title': 'no url task', 'status': 'pending', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    _register_runtime(monkeypatch, {})  # no label registered for 'nourl' at all
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert len(active) == 1
    row = active[0]
    for key in ('agent', 'loops', 'attempts', 'started', 'lane', 'phase', 'lane_state'):
        assert row[key] is None, f'expected {key}=None when no escalation URL, got {row[key]!r}'
    assert row['runtime_offline'] is True


@pytest.mark.asyncio
async def test_collect_active_tasks_runtime_per_task_read_failure_stays_online(
    tmp_path, monkeypatch, dummy_client,
):
    """A per-task artifact read failure (loops/attempts/started/phase=None,
    error set) is honest but distinct from project-offline: runtime_offline
    stays False.
    """
    root, shaped = _make_project(
        tmp_path, project_dir='flaky',
        tasks=[{'id': 9, 'title': 'flaky task', 'status': 'in-progress', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    _register_runtime(monkeypatch, {
        'flaky': [_runtime_entry(
            9, loops=None, attempts=None, started=None, phase=None,
            lane_state=None, error='wire-contract violation: bad enum value',
        )],
    })
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert len(active) == 1
    row = active[0]
    assert row['loops'] is None
    assert row['attempts'] is None
    assert row['phase'] is None
    assert row['started'] is None, (
        "a per-task read failure must not fabricate '0m running'"
    )
    assert row['runtime_offline'] is False, (
        'a per-task read failure is an honest error, not an offline project'
    )


@pytest.mark.asyncio
async def test_collect_active_tasks_includes_merge_deferred_and_train_field(
    tmp_path, monkeypatch, dummy_client
):
    """merge-deferred tasks survive the active filter and carry the `train` field.

    Task 101 has metadata.train set; the output dict must have train={'id', 'order'}
    (members[] is intentionally omitted from the projected wire shape).
    Task 102 has no train metadata; the output dict must have train=None.
    """
    root, shaped = _make_project(
        tmp_path,
        project_dir='trainyard',
        tasks=[
            {
                'id': 101,
                'title': 'train task with metadata',
                'status': 'merge-deferred',
                'dependencies': [],
                'metadata': {
                    'train': {'id': 'demo', 'order': 0, 'members': ['T-101', 'T-102']},
                    'files': [],
                },
            },
            {
                'id': 102,
                'title': 'merge-deferred without train metadata',
                'status': 'merge-deferred',
                'dependencies': [],
                'metadata': {'files': []},
            },
        ],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    cfg = DashboardConfig(project_root=root)
    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)

    ids = {t['id'] for t in active}
    assert 'trainyard/T-101' in ids, (
        "merge-deferred task 101 was dropped by the active-status filter — "
        "add 'merge-deferred' to _ACTIVE_STATUSES"
    )
    assert 'trainyard/T-102' in ids, (
        "merge-deferred task 102 was dropped by the active-status filter — "
        "add 'merge-deferred' to _ACTIVE_STATUSES"
    )

    by_id = {t['id']: t for t in active}
    assert by_id['trainyard/T-101']['train'] == {'id': 'demo', 'order': 0}, (
        "task 101 with train metadata should have train={'id': 'demo', 'order': 0} "
        "(members[] is intentionally omitted from the projected wire shape)"
    )
    assert by_id['trainyard/T-102']['train'] is None, (
        "task 102 without train metadata should have train=None"
    )


# ---------------------------------------------------------------------------
# Bounded done emission (step-3/step-4)
# ---------------------------------------------------------------------------


def _make_done_project(root, *, project_dir, active_tasks, done_tasks):
    """Layout a project with active + done tasks carrying updated_at.

    ``active_tasks`` and ``done_tasks`` are raw dicts; done tasks MUST include
    ``updated_at`` so the ordering / completed-field assertions work.
    """
    project_root = root / project_dir
    project_root.mkdir(parents=True, exist_ok=True)
    shaped = [_shape_task(t) for t in (active_tasks + done_tasks)]
    return project_root, shaped


@pytest.mark.asyncio
async def test_collect_active_tasks_bounded_done_appends_done_rows(tmp_path, monkeypatch, dummy_client):
    """max_done_per_project=2 appends the 2 most-recent done rows, most-recent first."""
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='df',
        active_tasks=[
            {'id': 1, 'title': 'active one', 'status': 'in-progress', 'dependencies': []},
        ],
        done_tasks=[
            {'id': 50, 'title': 'done oldest', 'status': 'done', 'dependencies': [],
             'updated_at': '2026-05-29T10:00:00+00:00'},
            {'id': 51, 'title': 'done middle', 'status': 'done', 'dependencies': [],
             'updated_at': '2026-05-29T11:00:00+00:00'},
            {'id': 52, 'title': 'done newest', 'status': 'done', 'dependencies': [],
             'updated_at': '2026-05-29T12:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    from dashboard.config import DashboardConfig
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg,
                                            max_done_per_project=2)

    done_rows = [t for t in active if t['status'] == 'done']
    assert len(done_rows) == 2, f'expected 2 done rows, got {len(done_rows)}'

    # Most-recent two: id 52 (12:00) and id 51 (11:00)
    done_ids = [t['id'] for t in done_rows]
    assert 'df/T-52' in done_ids
    assert 'df/T-51' in done_ids
    assert 'df/T-50' not in done_ids, 'oldest done task should be excluded by N=2 cap'


@pytest.mark.asyncio
async def test_collect_active_tasks_bounded_done_completed_field(tmp_path, monkeypatch, dummy_client):
    """Each done row must have 'completed' == its updated_at ISO string."""
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='df',
        active_tasks=[],
        done_tasks=[
            {'id': 10, 'title': 'done task', 'status': 'done', 'dependencies': [],
             'updated_at': '2026-05-29T09:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    from dashboard.config import DashboardConfig
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg,
                                            max_done_per_project=5)

    done_rows = [t for t in active if t['status'] == 'done']
    assert len(done_rows) == 1
    row = done_rows[0]
    assert row['completed'] == '2026-05-29T09:00:00+00:00', (
        f"expected completed == '2026-05-29T09:00:00+00:00', got {row.get('completed')!r}"
    )


@pytest.mark.asyncio
async def test_collect_active_tasks_active_rows_unchanged_no_completed_key(tmp_path, monkeypatch, dummy_client):
    """Active rows must NOT have a 'completed' key even when max_done_per_project>0."""
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='df',
        active_tasks=[
            {'id': 1, 'title': 'active', 'status': 'in-progress', 'dependencies': []},
        ],
        done_tasks=[
            {'id': 2, 'title': 'done', 'status': 'done', 'dependencies': [],
             'updated_at': '2026-05-29T10:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    from dashboard.config import DashboardConfig
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg,
                                            max_done_per_project=5)

    active_rows = [t for t in active if t['status'] != 'done']
    assert active_rows, 'expected at least one active row'
    for row in active_rows:
        assert 'completed' not in row, (
            f"active row {row['id']!r} must not have 'completed' key"
        )


@pytest.mark.asyncio
async def test_collect_active_tasks_default_excludes_done_rows(tmp_path, monkeypatch, dummy_client):
    """Default (no max_done_per_project) must return NO done rows — scheduler back-compat."""
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='df',
        active_tasks=[
            {'id': 1, 'title': 'active', 'status': 'pending', 'dependencies': []},
        ],
        done_tasks=[
            {'id': 2, 'title': 'done', 'status': 'done', 'dependencies': [],
             'updated_at': '2026-05-29T10:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    from dashboard.config import DashboardConfig
    cfg = DashboardConfig(project_root=root)

    # Call with NO max_done_per_project — default behaviour
    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)

    done_rows = [t for t in active if t['status'] == 'done']
    assert done_rows == [], (
        'Default collect_active_tasks must NOT return done rows — '
        'scheduler.py must not receive done tasks'
    )


@pytest.mark.asyncio
async def test_collect_active_tasks_done_ordering_tie_broken_by_id(tmp_path, monkeypatch, dummy_client):
    """When updated_at is identical, higher id wins the tie-break."""
    same_ts = '2026-05-29T10:00:00+00:00'
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='df',
        active_tasks=[],
        done_tasks=[
            {'id': 10, 'title': 'done low id', 'status': 'done', 'dependencies': [],
             'updated_at': same_ts},
            {'id': 20, 'title': 'done high id', 'status': 'done', 'dependencies': [],
             'updated_at': same_ts},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    from dashboard.config import DashboardConfig
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg,
                                            max_done_per_project=1)

    done_rows = [t for t in active if t['status'] == 'done']
    assert len(done_rows) == 1
    assert done_rows[0]['id'] == 'df/T-20', (
        'tie-break: higher id (20) should win over lower id (10)'
    )


# ---------------------------------------------------------------------------
# collect_done_counts (step-5/step-6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_done_counts_returns_per_project_done_count(tmp_path, monkeypatch, dummy_client):
    """collect_done_counts counts 'done' entries per project label."""
    df_root = tmp_path / 'dark-factory'
    df_root.mkdir()
    reify_root = tmp_path / 'reify'
    reify_root.mkdir()

    # dark-factory: 3 done out of mixed statuses
    df_statuses = {1: 'done', 2: 'done', 3: 'in-progress', 4: 'done', 5: 'pending'}
    # reify: 1 done
    reify_statuses = {10: 'done', 11: 'in-progress', 12: 'pending'}

    async def _fake_fetch_statuses(client, config, project_root):
        resolved = project_root.resolve()
        if resolved == df_root.resolve():
            return dict(df_statuses)
        if resolved == reify_root.resolve():
            return dict(reify_statuses)
        return {'offline': True, 'error': 'not found'}

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_statuses', _fake_fetch_statuses)

    cfg = DashboardConfig(project_root=df_root, known_project_roots=[reify_root])
    counts = await collect_done_counts(client=dummy_client, config=cfg)

    assert counts == {'dark-factory': 3, 'reify': 1}


@pytest.mark.asyncio
async def test_collect_done_counts_skips_offline_projects(tmp_path, monkeypatch, dummy_client):
    """collect_done_counts omits projects whose fetch_statuses returns an offline marker."""
    online_root = tmp_path / 'online-project'
    online_root.mkdir()
    offline_root = tmp_path / 'offline-project'
    offline_root.mkdir()

    async def _fake_fetch_statuses(client, config, project_root):
        if project_root.resolve() == offline_root.resolve():
            return {'offline': True, 'error': 'connection refused'}
        return {1: 'done', 2: 'in-progress'}

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_statuses', _fake_fetch_statuses)

    cfg = DashboardConfig(project_root=online_root, known_project_roots=[offline_root])
    counts = await collect_done_counts(client=dummy_client, config=cfg)

    assert 'offline-project' not in counts
    assert counts.get('online-project') == 1


@pytest.mark.asyncio
async def test_collect_done_counts_all_done_zero(tmp_path, monkeypatch, dummy_client):
    """A project with no done tasks returns 0, not a missing key."""
    root = tmp_path / 'empty-project'
    root.mkdir()

    async def _fake_fetch_statuses(client, config, project_root):
        return {1: 'in-progress', 2: 'pending'}

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_statuses', _fake_fetch_statuses)
    cfg = DashboardConfig(project_root=root)
    counts = await collect_done_counts(client=dummy_client, config=cfg)

    assert counts == {'empty-project': 0}


# ---------------------------------------------------------------------------
# external_deps field on task rows (step-1 / step-2)
# ---------------------------------------------------------------------------


def test_build_task_row_external_deps_from_metadata():
    """_build_task_row carries external_deps as [{'id','status':'unknown'}] per entry."""
    task = {
        'id': 42,
        'title': 'cross-project waiter',
        'description': '',
        'details': '',
        'status': 'pending',
        'metadata': {'external_deps': ['dark_factory:13', 'reify:8']},
    }
    row = _build_task_row('myproject', task, 42, {}, 'myproject/T-42')
    assert row['external_deps'] == [
        {'id': 'dark_factory:13', 'status': 'unknown'},
        {'id': 'reify:8', 'status': 'unknown'},
    ]


def test_build_task_row_external_deps_empty_when_absent():
    """_build_task_row yields external_deps=[] when metadata.external_deps is absent."""
    task = {'id': 1, 'title': 'no ext', 'status': 'pending', 'metadata': {}}
    row = _build_task_row('myproject', task, 1, {}, 'myproject/T-1')
    assert row['external_deps'] == []


def test_build_task_row_external_deps_empty_when_non_list():
    """_build_task_row yields external_deps=[] when metadata.external_deps is not a list."""
    for bad_value in [None, 'foo:1', 123, {'a': 'b'}]:
        task = {'id': 1, 'title': 'bad', 'status': 'pending',
                'metadata': {'external_deps': bad_value}}
        row = _build_task_row('p', task, 1, {}, 'p/T-1')
        assert row['external_deps'] == [], (
            f'expected [] for external_deps={bad_value!r}'
        )


def test_build_task_row_external_deps_ignores_non_str_and_empty():
    """_build_task_row ignores empty strings and non-str items in external_deps."""
    task = {'id': 1, 'title': 'x', 'status': 'pending',
            'metadata': {'external_deps': ['', 'dark_factory:13', '', None, 42]}}
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['external_deps'] == [{'id': 'dark_factory:13', 'status': 'unknown'}]


# ---------------------------------------------------------------------------
# prd field coalescing on task rows (step-1 / step-2)
# ---------------------------------------------------------------------------


def test_build_task_row_prd_field_from_prd_path():
    """_build_task_row coalesces metadata.prd_path into row['prd'] verbatim."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': 'plans/dashboard-taskgraph-legibility-prd.md'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/dashboard-taskgraph-legibility-prd.md'


def test_build_task_row_prd_field_strips_anchor_suffix():
    """A trailing '#anchor' fragment on prd_path is stripped."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': 'plans/foo-prd.md#implementation-notes'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/foo-prd.md'


def test_build_task_row_prd_field_strips_section_suffix():
    """A trailing '§section' fragment on prd_path is stripped."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': 'plans/foo-prd.md§Contract'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/foo-prd.md'


def test_build_task_row_prd_field_trims_whitespace():
    """Surrounding whitespace on prd_path is trimmed."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': '  plans/foo-prd.md  '},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/foo-prd.md'


def test_build_task_row_prd_field_legacy_prd_key():
    """Legacy 'prd' key is coalesced when prd_path is absent."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd': 'docs/legacy-prd.md'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'docs/legacy-prd.md'


def test_build_task_row_prd_field_legacy_prd_ref_key():
    """Legacy 'prd_ref' key is coalesced when prd_path and prd are absent."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_ref': 'docs/legacy-ref-prd.md'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'docs/legacy-ref-prd.md'


def test_build_task_row_prd_field_precedence_prd_path_over_prd_and_ref():
    """When multiple provenance keys are present, prd_path wins over prd and prd_ref."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {
            'prd_path': 'plans/winner-prd.md',
            'prd': 'plans/loser-prd.md',
            'prd_ref': 'plans/loser-ref-prd.md',
        },
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/winner-prd.md'


def test_build_task_row_prd_field_empty_prd_path_falls_through_to_prd():
    """An empty-string prd_path is skipped in favor of a non-empty prd."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': '', 'prd': 'plans/fallback-prd.md'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/fallback-prd.md'


def test_build_task_row_prd_field_suffix_only_prd_path_falls_through():
    """A prd_path that is ONLY a suffix (cleans to '') falls through to the next key."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': '#just-an-anchor', 'prd': 'plans/fallback-prd.md'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/fallback-prd.md'


def test_build_task_row_prd_field_none_when_no_provenance_keys():
    """row['prd'] is None when no prd_path/prd/prd_ref keys are present."""
    task = {'id': 1, 'title': 'x', 'status': 'pending', 'metadata': {}}
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] is None


def test_build_task_row_prd_field_non_string_values_skipped():
    """Non-string prd_path values (int, None) are skipped, yielding None."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': 123},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] is None

    task_none = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': None},
    }
    row_none = _build_task_row('p', task_none, 1, {}, 'p/T-1')
    assert row_none['prd'] is None


def test_build_task_row_prd_kwarg_overrides_metadata_coalescing():
    """An explicit prd= kwarg wins verbatim over metadata coalescing.

    Callers that already ran _coalesce_prd (e.g. the terminal-bucket loop, to
    decide live-PRD membership) pass the result through instead of paying for
    the split/strip work a second time; this proves the passthrough is used
    rather than silently re-deriving from metadata.
    """
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': 'plans/metadata-derived-prd.md'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1', prd='plans/explicit-prd.md')
    assert row['prd'] == 'plans/explicit-prd.md', (
        'an explicit prd kwarg must win over metadata coalescing'
    )


@pytest.mark.asyncio
async def test_collect_active_tasks_includes_external_deps_with_unknown_sentinel(
    tmp_path, monkeypatch, dummy_client,
):
    """A task carrying metadata.external_deps surfaces external_deps with 'unknown' sentinels."""
    root, shaped = _make_project(
        tmp_path,
        project_dir='xdeps',
        tasks=[
            {
                'id': 5,
                'title': 'waits on upstream',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['dark_factory:13', 'reify:8']},
            },
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert len(active) == 1
    row = active[0]
    assert row['external_deps'] == [
        {'id': 'dark_factory:13', 'status': 'unknown'},
        {'id': 'reify:8', 'status': 'unknown'},
    ]


@pytest.mark.asyncio
async def test_collect_active_tasks_external_deps_empty_when_absent(
    tmp_path, monkeypatch, dummy_client,
):
    """Tasks without external_deps carry external_deps=[] (no KeyError)."""
    root, shaped = _make_project(
        tmp_path,
        project_dir='nodeps',
        tasks=[
            {'id': 1, 'title': 'plain task', 'status': 'pending',
             'dependencies': [], 'metadata': {}},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert len(active) == 1
    assert active[0]['external_deps'] == []


# ---------------------------------------------------------------------------
# collect_tasks_with_counts resolve_external (step-5 / step-6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_overwrites_status(
    tmp_path, monkeypatch, dummy_client,
):
    """With resolve_external=True, statuses from fetch_external_statuses overwrite 'unknown'.

    'dark_factory:13' resolves to 'done'; 'reify:8' is absent from the map
    and keeps the honest 'unknown' sentinel (no fabricated status).
    """
    root, shaped = _make_project(
        tmp_path,
        project_dir='xdeps',
        tasks=[
            {
                'id': 5,
                'title': 'waits on upstream',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['dark_factory:13', 'reify:8']},
            },
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    async def _fake_ext_statuses(client, config, deps):
        # Returns only 'dark_factory:13'; 'reify:8' is absent (simulates partial map).
        return {'dark_factory:13': 'done'}

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _fake_ext_statuses)

    cfg = DashboardConfig(project_root=root)
    active, _, _, _, _ = await collect_tasks_with_counts(
        client=dummy_client, config=cfg, resolve_external=True,
    )
    assert len(active) == 1
    row = active[0]
    assert row['external_deps'] == [
        {'id': 'dark_factory:13', 'status': 'done'},      # resolved
        {'id': 'reify:8', 'status': 'unknown'},            # absent from map → stays 'unknown'
    ]


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_false_skips_mcp(
    tmp_path, monkeypatch, dummy_client,
):
    """With resolve_external=False (default), fetch_external_statuses is NOT called."""
    root, shaped = _make_project(
        tmp_path,
        project_dir='xdeps',
        tasks=[
            {
                'id': 5,
                'title': 'waits on upstream',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['dark_factory:13']},
            },
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    async def _must_not_be_called(*args, **kwargs):
        raise AssertionError('fetch_external_statuses must NOT be called when resolve_external=False')

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _must_not_be_called)

    cfg = DashboardConfig(project_root=root)
    # Default resolve_external=False — must NOT call fetch_external_statuses.
    active, _, _, _, _ = await collect_tasks_with_counts(client=dummy_client, config=cfg)
    # Rows keep 'unknown' sentinel (unresolved).
    assert active[0]['external_deps'] == [{'id': 'dark_factory:13', 'status': 'unknown'}]


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_single_batched_call(
    tmp_path, monkeypatch, dummy_client,
):
    """resolve_external=True issues exactly ONE batched fetch_external_statuses call
    covering the deduped union of ALL rows' external dep ids.
    """
    root, shaped = _make_project(
        tmp_path,
        project_dir='multi',
        tasks=[
            {
                'id': 1, 'title': 'A', 'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['proj:10', 'proj:20']},
            },
            {
                'id': 2, 'title': 'B', 'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['proj:20', 'proj:30']},  # 'proj:20' deduped
            },
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    calls = []

    async def _record_call(client, config, deps):
        calls.append(sorted(deps))
        return {}

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _record_call)

    cfg = DashboardConfig(project_root=root)
    await collect_tasks_with_counts(client=dummy_client, config=cfg, resolve_external=True)

    assert len(calls) == 1, f'expected 1 batched call, got {len(calls)}: {calls}'
    # Deduped union: proj:10, proj:20, proj:30
    assert calls[0] == ['proj:10', 'proj:20', 'proj:30']


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_skips_call_when_no_deps(
    tmp_path, monkeypatch, dummy_client,
):
    """resolve_external=True skips the fetch call when no rows have external deps."""
    root, shaped = _make_project(
        tmp_path,
        project_dir='nodeps',
        tasks=[
            {'id': 1, 'title': 'plain', 'status': 'pending',
             'dependencies': [], 'metadata': {}},
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    async def _must_not_be_called(*args, **kwargs):
        raise AssertionError('fetch_external_statuses must NOT be called when union is empty')

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _must_not_be_called)

    cfg = DashboardConfig(project_root=root)
    active, _, _, _, _ = await collect_tasks_with_counts(
        client=dummy_client, config=cfg, resolve_external=True,
    )
    assert active[0]['external_deps'] == []


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_skips_done_rows(
    tmp_path, monkeypatch, dummy_client,
):
    """resolve_external=True must NOT include done or cancelled rows' external dep ids.

    Done and cancelled tasks' external deps are no longer actionable. Their ids must not
    bloat the MCP request, and their rows must keep the 'unknown' sentinel (not re-stamped).
    This covers both the done bounded bucket and the cancelled bounded bucket, since both
    rows carry the 'completed' sentinel key that the skip guard checks.
    """
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='xdeps',
        active_tasks=[
            {
                'id': 5,
                'title': 'active waiter',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['proj:10']},
            },
        ],
        done_tasks=[
            {
                'id': 6,
                'title': 'done with external dep',
                'status': 'done',
                'dependencies': [],
                'updated_at': '2026-05-29T10:00:00+00:00',
                'metadata': {'external_deps': ['proj:99']},  # must NOT enter the batched call
            },
            {
                'id': 7,
                'title': 'cancelled with external dep',
                'status': 'cancelled',
                'dependencies': [],
                'updated_at': '2026-05-29T11:00:00+00:00',
                'metadata': {'external_deps': ['proj:88']},  # must NOT enter the batched call
            },
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    calls: list[list[str]] = []

    async def _record_call(client, config, deps):
        calls.append(sorted(deps))
        return {'proj:10': 'done'}

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _record_call)

    cfg = DashboardConfig(project_root=root)
    active, _, _, _, _ = await collect_tasks_with_counts(
        client=dummy_client, config=cfg,
        max_done_per_project=5, max_cancelled_per_project=5, resolve_external=True,
    )

    # Only the active row's dep id should be in the batched call — NOT 'proj:99' or 'proj:88'.
    assert calls == [['proj:10']], (
        f'done/cancelled row deps must not appear in the batched call; got {calls}'
    )

    by_id = {r['id']: r for r in active}
    # Active row's dep was resolved.
    assert by_id['xdeps/T-5']['external_deps'] == [{'id': 'proj:10', 'status': 'done'}]
    # Done row's dep kept 'unknown' (was not re-stamped).
    assert by_id['xdeps/T-6']['external_deps'] == [{'id': 'proj:99', 'status': 'unknown'}]
    # Cancelled row's dep also kept 'unknown' (was not re-stamped).
    assert by_id['xdeps/T-7']['external_deps'] == [{'id': 'proj:88', 'status': 'unknown'}]


# ---------------------------------------------------------------------------
# bounded cancelled emission (step-3 / step-4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_active_tasks_bounded_cancelled_appends_rows(
    tmp_path, monkeypatch, dummy_client,
):
    """max_cancelled_per_project=2 appends the 2 most-recent cancelled rows, most-recent first.

    Project has 1 active task + 3 cancelled tasks (ids 60/61/62 ascending updated_at).
    - Expected: exactly 2 cancelled rows, the 2 most-recent (T-62, T-61); T-60 excluded.
    - Each cancelled row: status=='cancelled', started==0, deps==[], completed==its updated_at.

    RED today: max_cancelled_per_project kwarg does not exist (TypeError) and no cancelled
    emission occurs.
    """
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='df',
        active_tasks=[
            {'id': 1, 'title': 'active one', 'status': 'in-progress', 'dependencies': []},
        ],
        done_tasks=[
            {'id': 60, 'title': 'cancelled oldest', 'status': 'cancelled', 'dependencies': [],
             'updated_at': '2026-05-29T10:00:00+00:00'},
            {'id': 61, 'title': 'cancelled middle', 'status': 'cancelled', 'dependencies': [],
             'updated_at': '2026-05-29T11:00:00+00:00'},
            {'id': 62, 'title': 'cancelled newest', 'status': 'cancelled', 'dependencies': [],
             'updated_at': '2026-05-29T12:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    from dashboard.config import DashboardConfig
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(
        client=dummy_client, config=cfg, max_cancelled_per_project=2,
    )

    cancelled_rows = [t for t in active if t['status'] == 'cancelled']
    assert len(cancelled_rows) == 2, f'expected 2 cancelled rows, got {len(cancelled_rows)}'

    # Most-recent two: id 62 (12:00) and id 61 (11:00)
    cancelled_ids = [t['id'] for t in cancelled_rows]
    assert 'df/T-62' in cancelled_ids
    assert 'df/T-61' in cancelled_ids
    assert 'df/T-60' not in cancelled_ids, 'oldest cancelled task should be excluded by N=2 cap'

    # Each row must have the bounded-bucket sentinel fields
    for row in cancelled_rows:
        assert row['started'] == 0, f"cancelled row {row['id']}: expected started==0, got {row['started']}"
        assert row['deps'] == [], f"cancelled row {row['id']}: expected deps==[], got {row['deps']}"
        assert 'completed' in row, f"cancelled row {row['id']}: must have 'completed' key"

    # completed == updated_at for each row
    by_id = {t['id']: t for t in cancelled_rows}
    assert by_id['df/T-62']['completed'] == '2026-05-29T12:00:00+00:00'
    assert by_id['df/T-61']['completed'] == '2026-05-29T11:00:00+00:00'


@pytest.mark.asyncio
async def test_collect_active_tasks_default_excludes_cancelled(
    tmp_path, monkeypatch, dummy_client,
):
    """Default (no max_cancelled_per_project) must return ZERO cancelled rows."""
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='df',
        active_tasks=[
            {'id': 1, 'title': 'active', 'status': 'pending', 'dependencies': []},
        ],
        done_tasks=[
            {'id': 60, 'title': 'cancelled', 'status': 'cancelled', 'dependencies': [],
             'updated_at': '2026-05-29T10:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    from dashboard.config import DashboardConfig
    cfg = DashboardConfig(project_root=root)

    # Call with NO max_cancelled_per_project — default behaviour
    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)

    cancelled_rows = [t for t in active if t['status'] == 'cancelled']
    assert cancelled_rows == [], (
        'Default collect_active_tasks must NOT return cancelled rows'
    )


@pytest.mark.asyncio
async def test_collect_active_tasks_cancelled_ordering_tie_broken_by_id(
    tmp_path, monkeypatch, dummy_client,
):
    """When updated_at is identical for cancelled tasks, higher id wins the tie-break."""
    same_ts = '2026-05-29T10:00:00+00:00'
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='df',
        active_tasks=[],
        done_tasks=[
            {'id': 10, 'title': 'cancelled low id', 'status': 'cancelled', 'dependencies': [],
             'updated_at': same_ts},
            {'id': 20, 'title': 'cancelled high id', 'status': 'cancelled', 'dependencies': [],
             'updated_at': same_ts},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    from dashboard.config import DashboardConfig
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(
        client=dummy_client, config=cfg, max_cancelled_per_project=1,
    )

    cancelled_rows = [t for t in active if t['status'] == 'cancelled']
    assert len(cancelled_rows) == 1
    assert cancelled_rows[0]['id'] == 'df/T-20', (
        'tie-break: higher id (20) should win over lower id (10)'
    )


@pytest.mark.asyncio
async def test_collect_active_tasks_both_done_and_cancelled_buckets_independent(
    tmp_path, monkeypatch, dummy_client,
):
    """Done and cancelled bucket caps are applied independently and don't bleed into each other.

    Project: 1 active + 2 done (T-50 older, T-51 newer) + 2 cancelled (T-60 older, T-61 newer).
    Call with max_done_per_project=1, max_cancelled_per_project=1.
    Expected:
    - Exactly 1 done row: the most-recent T-51 (T-50 excluded).
    - Exactly 1 cancelled row: the most-recent T-61 (T-60 excluded).
    - Done rows carry status=='done' only; cancelled rows carry status=='cancelled' only.
    - Total: 3 rows (1 active + 1 done + 1 cancelled).

    A regression that mixes the two buckets or shares a single cap would surface here:
    e.g. if the cap were 1 shared across both, only one terminal row would appear instead of two.
    If the buckets bleed, a done row might carry status=='cancelled' or vice-versa.
    """
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='df',
        active_tasks=[
            {'id': 1, 'title': 'active one', 'status': 'in-progress', 'dependencies': []},
        ],
        done_tasks=[
            {'id': 50, 'title': 'done older', 'status': 'done', 'dependencies': [],
             'updated_at': '2026-05-29T10:00:00+00:00'},
            {'id': 51, 'title': 'done newer', 'status': 'done', 'dependencies': [],
             'updated_at': '2026-05-29T12:00:00+00:00'},
            {'id': 60, 'title': 'cancelled older', 'status': 'cancelled', 'dependencies': [],
             'updated_at': '2026-05-29T09:00:00+00:00'},
            {'id': 61, 'title': 'cancelled newer', 'status': 'cancelled', 'dependencies': [],
             'updated_at': '2026-05-29T11:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    from dashboard.config import DashboardConfig
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(
        client=dummy_client, config=cfg,
        max_done_per_project=1,
        max_cancelled_per_project=1,
    )

    done_rows = [t for t in active if t['status'] == 'done']
    cancelled_rows = [t for t in active if t['status'] == 'cancelled']

    # Each bucket is capped independently at 1 (shared cap would leave only 1 terminal row).
    assert len(done_rows) == 1, f'expected 1 done row, got {len(done_rows)}'
    assert len(cancelled_rows) == 1, f'expected 1 cancelled row, got {len(cancelled_rows)}'
    assert len(active) == 3, f'expected 3 total rows (1 active + 1 done + 1 cancelled), got {len(active)}'

    # Most-recent row from each bucket wins.
    assert done_rows[0]['id'] == 'df/T-51', (
        f"expected most-recent done T-51, got {done_rows[0]['id']}"
    )
    assert cancelled_rows[0]['id'] == 'df/T-61', (
        f"expected most-recent cancelled T-61, got {cancelled_rows[0]['id']}"
    )

    # Cross-bucket purity: done bucket contains only done rows; cancelled only cancelled.
    assert all(r['status'] == 'done' for r in done_rows), (
        f"done bucket contains non-done rows: {[r['status'] for r in done_rows]}"
    )
    assert all(r['status'] == 'cancelled' for r in cancelled_rows), (
        f"cancelled bucket contains non-cancelled rows: {[r['status'] for r in cancelled_rows]}"
    )

    # Older rows in each bucket are excluded by the cap.
    ids = {t['id'] for t in active}
    assert 'df/T-50' not in ids, 'older done T-50 should be excluded by max_done_per_project=1'
    assert 'df/T-60' not in ids, 'older cancelled T-60 should be excluded by max_cancelled_per_project=1'


# ---------------------------------------------------------------------------
# deferred via active path (step-1 / step-2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_active_tasks_includes_deferred_via_active_path(
    tmp_path, monkeypatch, dummy_client,
):
    """deferred tasks must flow through the ACTIVE path (with resolved deps + started).

    A deferred task (id 30, deps=[31]) plus a done dep (id 31).
    Expected:
    (a) 'proj/T-30' appears in the returned rows — deferred survives the active filter.
    (b) The deferred row's deps == [{'id': 'proj/T-31', 'title': ..., 'done': True}] —
        proves it flows through the active path with resolved deps, NOT the stripped
        bounded-bucket path.
    (c) The row has a 'started' key and does NOT have a 'completed' key.

    RED today: 'deferred' is not in _ACTIVE_STATUSES, so the task is dropped.
    """
    root, shaped = _make_project(
        tmp_path,
        project_dir='proj',
        tasks=[
            {
                'id': 30,
                'title': 'parked work',
                'status': 'deferred',
                'dependencies': [31],
                'metadata': {},
            },
            {
                'id': 31,
                'title': 'finished dep',
                'status': 'done',
                'dependencies': [],
                'metadata': {},
            },
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)

    ids = {t['id'] for t in active}
    assert 'proj/T-30' in ids, (
        "deferred task T-30 was dropped by the active-status filter — "
        "add 'deferred' to _ACTIVE_STATUSES"
    )

    by_id = {t['id']: t for t in active}
    row = by_id['proj/T-30']

    # (b) resolved deps via active path — done flag on the dep.
    #
    # Task 3857: this is the scheduler path (both terminal caps 0), which by
    # design now fetches NO terminal rows at all, so the done dep's full row
    # is not available and its title degrades to ''. The done flag — the
    # load-bearing half of the chip — still resolves, via the compact status
    # map. Resolving the title would cost an extra whole-tree read, which is
    # the unbounded fetch this design removed.
    assert row['deps'] == [{'id': 'proj/T-31', 'title': '', 'done': True}], (
        f"expected deferred row deps with done=True, got: {row.get('deps')}"
    )

    # (c) active-path fields present / absent
    assert 'started' in row, "deferred row must have 'started' key (active path)"
    assert 'completed' not in row, (
        "deferred row must NOT have 'completed' key (that is the bounded-bucket sentinel)"
    )


# ---------------------------------------------------------------------------
# resolve_external + offline marker (step-11 / step-12)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_offline_marker(
    tmp_path, monkeypatch, dummy_client,
):
    """When fetch_external_statuses returns the offline marker, each dep entry gets
    status=='offline' (distinct from 'unknown'=task-not-found).

    Fails today because status_map.get(id, 'unknown') yields 'unknown' for all deps
    when the map is the offline marker {'offline':True,'error':...}.
    """
    from dashboard.config import DashboardConfig
    from dashboard.data.active_tasks import collect_tasks_with_counts

    root, shaped = _make_project(
        tmp_path,
        project_dir='offline_ext',
        tasks=[
            {
                'id': 99,
                'title': 'blocked on upstream',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['dark_factory:42', 'reify:7']},
            },
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    async def _offline_ext_statuses(client, config, deps):
        return {'offline': True, 'error': 'down'}

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _offline_ext_statuses)

    cfg = DashboardConfig(project_root=root)
    active, _, _, _, _ = await collect_tasks_with_counts(
        client=dummy_client, config=cfg, resolve_external=True,
    )

    assert len(active) == 1
    row = active[0]
    for entry in row['external_deps']:
        assert entry['status'] == 'offline', (
            f"expected status='offline' for dep {entry['id']!r} when MCP is offline, "
            f"got: {entry['status']!r}"
        )


# ---------------------------------------------------------------------------
# live-PRD terminal-member exemption (step-3 / step-4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_active_tasks_live_prd_member_beyond_cap_is_exempted(
    tmp_path, monkeypatch, dummy_client,
):
    """A done member of a live (still-active) PRD is emitted even beyond max_done_per_project.

    Project has:
    - 1 active task (id=1) tagged metadata.prd_path='plans/x-prd.md' (keeps the PRD "live").
    - A done x-prd member (id=2) OLDER than a no-prd done task (id=4), with a done
      dependency (id=3) present in the task list.
    - A no-prd done task (id=4), NEWER than id=2, which is the top-1 cap pick.

    With max_done_per_project=1, the cap alone would only keep id=4. The x-prd
    done member (id=2) must ALSO appear because its prd is still live, with
    populated deps (resolved via by_id) and the usual terminal-row fields. The
    no-prd top-1 row (id=4) is unaffected: deps==[].
    """
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='xprd',
        active_tasks=[
            {'id': 1, 'title': 'active x-prd member', 'status': 'in-progress',
             'dependencies': [], 'metadata': {'prd_path': 'plans/x-prd.md'}},
        ],
        done_tasks=[
            {'id': 2, 'title': 'x-prd done member', 'status': 'done',
             'dependencies': [3], 'metadata': {'prd_path': 'plans/x-prd.md'},
             'updated_at': '2026-05-29T09:00:00+00:00'},
            {'id': 3, 'title': 'x-prd done dep', 'status': 'done',
             'dependencies': [], 'metadata': {},
             'updated_at': '2026-05-29T08:00:00+00:00'},
            {'id': 4, 'title': 'no-prd done newest', 'status': 'done',
             'dependencies': [], 'metadata': {},
             'updated_at': '2026-05-29T12:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg,
                                            max_done_per_project=1)

    by_id = {t['id']: t for t in active}
    assert 'xprd/T-2' in by_id, (
        'live-PRD done member (id=2) must be emitted despite being beyond the top-1 cap'
    )
    row = by_id['xprd/T-2']
    assert row['started'] == 0
    assert row['completed'] == '2026-05-29T09:00:00+00:00'
    assert row['deps'] == [{'id': 'xprd/T-3', 'title': 'x-prd done dep', 'done': True}]

    # The no-prd top-1 row (id=4) is unaffected: still capped-in, still deps==[].
    assert 'xprd/T-4' in by_id
    assert by_id['xprd/T-4']['deps'] == []

    # The dep task itself (id=3): no prd metadata, older than id=4 → stays excluded.
    assert 'xprd/T-3' not in by_id, (
        'the dep task itself (id=3, no prd, older, beyond cap) should stay excluded'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('active_status', ['blocked', 'pending', 'merge-deferred', 'deferred'])
async def test_collect_active_tasks_live_prd_member_beyond_cap_exempted_for_other_active_statuses(
    tmp_path, monkeypatch, dummy_client, active_status,
):
    """The live-PRD exemption keys on ANY _ACTIVE_STATUSES member, not just 'in-progress'.

    Same shape as test_collect_active_tasks_live_prd_member_beyond_cap_is_exempted, but
    the task keeping the PRD "live" is parametrized across the *other* members of
    _ACTIVE_STATUSES. Guards against a regression that narrows "live" to just
    'in-progress'.
    """
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='statprd',
        active_tasks=[
            {'id': 1, 'title': 'active status-prd member', 'status': active_status,
             'dependencies': [], 'metadata': {'prd_path': 'plans/status-prd.md'}},
        ],
        done_tasks=[
            {'id': 2, 'title': 'status-prd done member', 'status': 'done',
             'dependencies': [], 'metadata': {'prd_path': 'plans/status-prd.md'},
             'updated_at': '2026-05-29T09:00:00+00:00'},
            {'id': 3, 'title': 'no-prd done newest', 'status': 'done',
             'dependencies': [], 'metadata': {},
             'updated_at': '2026-05-29T12:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg,
                                            max_done_per_project=1)

    ids = {t['id'] for t in active}
    assert 'statprd/T-2' in ids, (
        f'live-PRD done member must be exempted from the cap when kept live by a '
        f'{active_status!r} member, not just "in-progress"'
    )
    assert 'statprd/T-3' in ids


@pytest.mark.asyncio
async def test_collect_active_tasks_fully_done_prd_not_exempted(
    tmp_path, monkeypatch, dummy_client,
):
    """A done member of a PRD with NO active member stays subject to the cap.

    'plans/y-prd.md' has no task in _ACTIVE_STATUSES, so it is not a "live" PRD:
    its done member does not get the terminal-member exemption and, being older
    than the capped-in row, is excluded entirely.
    """
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='yprd',
        active_tasks=[],
        done_tasks=[
            {'id': 10, 'title': 'y-prd done member', 'status': 'done',
             'dependencies': [], 'metadata': {'prd_path': 'plans/y-prd.md'},
             'updated_at': '2026-05-29T09:00:00+00:00'},
            {'id': 11, 'title': 'other done newest', 'status': 'done',
             'dependencies': [], 'metadata': {},
             'updated_at': '2026-05-29T12:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg,
                                            max_done_per_project=1)

    ids = {t['id'] for t in active}
    assert 'yprd/T-10' not in ids, (
        'done member of a fully-done PRD (no active member) must NOT be exempted from the cap'
    )
    assert 'yprd/T-11' in ids


@pytest.mark.asyncio
async def test_collect_active_tasks_live_prd_member_within_cap_no_duplicate(
    tmp_path, monkeypatch, dummy_client,
):
    """A live-PRD done member that is ALSO within the top-N cap appears exactly once.

    Single-pass emission must not double-emit a task that satisfies both the
    capped_ids membership AND the is_live_member exemption, and it must carry
    the populated-deps treatment either way.
    """
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='zprd',
        active_tasks=[
            {'id': 1, 'title': 'active z-prd member', 'status': 'in-progress',
             'dependencies': [], 'metadata': {'prd_path': 'plans/z-prd.md'}},
        ],
        done_tasks=[
            {'id': 2, 'title': 'z-prd done member newest', 'status': 'done',
             'dependencies': [3], 'metadata': {'prd_path': 'plans/z-prd.md'},
             'updated_at': '2026-05-29T12:00:00+00:00'},
            {'id': 3, 'title': 'z-prd done dep', 'status': 'done',
             'dependencies': [], 'metadata': {},
             'updated_at': '2026-05-29T08:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg,
                                            max_done_per_project=1)

    matches = [t for t in active if t['id'] == 'zprd/T-2']
    assert len(matches) == 1, f'expected exactly 1 row for the live-PRD member, got {len(matches)}'
    assert matches[0]['deps'] == [{'id': 'zprd/T-3', 'title': 'z-prd done dep', 'done': True}]


@pytest.mark.asyncio
async def test_collect_active_tasks_live_prd_cancelled_member_beyond_cap_is_exempted(
    tmp_path, monkeypatch, dummy_client,
):
    """Cancelled-bucket symmetry: a cancelled member of a live PRD is exempted from
    max_cancelled_per_project the same way a done member is exempted from
    max_done_per_project.
    """
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='cprd',
        active_tasks=[
            {'id': 1, 'title': 'active c-prd member', 'status': 'in-progress',
             'dependencies': [], 'metadata': {'prd_path': 'plans/c-prd.md'}},
        ],
        done_tasks=[
            {'id': 2, 'title': 'c-prd cancelled member', 'status': 'cancelled',
             'dependencies': [3], 'metadata': {'prd_path': 'plans/c-prd.md'},
             'updated_at': '2026-05-29T09:00:00+00:00'},
            {'id': 3, 'title': 'c-prd cancelled dep', 'status': 'done',
             'dependencies': [], 'metadata': {},
             'updated_at': '2026-05-29T08:00:00+00:00'},
            {'id': 4, 'title': 'no-prd cancelled newest', 'status': 'cancelled',
             'dependencies': [], 'metadata': {},
             'updated_at': '2026-05-29T12:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg,
                                            max_cancelled_per_project=1)

    by_id = {t['id']: t for t in active}
    assert 'cprd/T-2' in by_id, (
        'live-PRD cancelled member (id=2) must be emitted despite being beyond the top-1 cap'
    )
    row = by_id['cprd/T-2']
    assert row['started'] == 0
    assert row['completed'] == '2026-05-29T09:00:00+00:00'
    assert row['deps'] == [{'id': 'cprd/T-3', 'title': 'c-prd cancelled dep', 'done': True}]

    assert 'cprd/T-4' in by_id
    assert by_id['cprd/T-4']['deps'] == []


@pytest.mark.asyncio
async def test_collect_active_tasks_no_provenance_terminal_row_stays_capped(
    tmp_path, monkeypatch, dummy_client,
):
    """A terminal task with no prd metadata (sharing no live PRD) stays subject to
    the cap and keeps deps==[] — the live-PRD exemption must not apply to it.
    """
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='noprd',
        active_tasks=[
            {'id': 1, 'title': 'active, no prd', 'status': 'in-progress',
             'dependencies': [], 'metadata': {}},
        ],
        done_tasks=[
            {'id': 2, 'title': 'done newest, no prd', 'status': 'done',
             'dependencies': [], 'metadata': {},
             'updated_at': '2026-05-29T12:00:00+00:00'},
            {'id': 3, 'title': 'done oldest, no prd', 'status': 'done',
             'dependencies': [], 'metadata': {},
             'updated_at': '2026-05-29T09:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg,
                                            max_done_per_project=1)

    ids = {t['id'] for t in active}
    assert 'noprd/T-3' not in ids, (
        'no-provenance done row beyond the cap must stay excluded (no exemption applies)'
    )
    assert 'noprd/T-2' in ids
    row = next(t for t in active if t['id'] == 'noprd/T-2')
    assert row['deps'] == []


@pytest.mark.asyncio
async def test_collect_active_tasks_live_prd_exemption_warns_when_unusually_large(
    tmp_path, monkeypatch, dummy_client, caplog,
):
    """A pathological PRD with far more live terminal members than the warn
    threshold logs a warning — but still emits every member; the exemption
    never silently drops rows, it only gains defensive visibility for an
    unusually large count.
    """
    import dashboard.data.active_tasks as active_tasks_mod

    n = active_tasks_mod._LIVE_PRD_EXEMPTION_WARN_THRESHOLD + 5
    done_tasks = [
        {'id': 100 + i, 'title': f'huge-prd done member {i}', 'status': 'done',
         'dependencies': [], 'metadata': {'prd_path': 'plans/huge-prd.md'},
         'updated_at': f'2026-05-29T09:{i % 60:02d}:00+00:00'}
        for i in range(n)
    ]
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='hugeprd',
        active_tasks=[
            {'id': 1, 'title': 'active huge-prd member', 'status': 'in-progress',
             'dependencies': [], 'metadata': {'prd_path': 'plans/huge-prd.md'}},
        ],
        done_tasks=done_tasks,
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    cfg = DashboardConfig(project_root=root)

    with caplog.at_level('WARNING', logger='dashboard.data.active_tasks'):
        active, _ = await collect_active_tasks(client=dummy_client, config=cfg,
                                                max_done_per_project=1)

    done_rows = [t for t in active if t['status'] == 'done']
    assert len(done_rows) == n, (
        'the exemption must still emit every live member even when the count is huge'
    )
    assert any(
        'live-PRD exemption' in rec.message and 'hugeprd' in rec.message
        for rec in caplog.records
    ), 'an unusually large exemption count should log a warning'


@pytest.mark.asyncio
async def test_collect_active_tasks_live_prd_exemption_no_warning_under_threshold(
    tmp_path, monkeypatch, dummy_client, caplog,
):
    """A normal-sized live-PRD exemption (well under the threshold) logs nothing."""
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='smallprd',
        active_tasks=[
            {'id': 1, 'title': 'active small-prd member', 'status': 'in-progress',
             'dependencies': [], 'metadata': {'prd_path': 'plans/small-prd.md'}},
        ],
        done_tasks=[
            {'id': 2, 'title': 'small-prd done member', 'status': 'done',
             'dependencies': [], 'metadata': {'prd_path': 'plans/small-prd.md'},
             'updated_at': '2026-05-29T09:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    cfg = DashboardConfig(project_root=root)

    with caplog.at_level('WARNING', logger='dashboard.data.active_tasks'):
        active, _ = await collect_active_tasks(client=dummy_client, config=cfg,
                                                max_done_per_project=1)

    assert any(t['id'] == 'smallprd/T-2' for t in active)
    assert not any('live-PRD exemption' in rec.message for rec in caplog.records), (
        'a small exemption count must not trigger the pathological-case warning'
    )


# ---------------------------------------------------------------------------
# TestShapeOneProjectNarrowing — _shape_one_project must request only what it
# renders, and derive counts from the compact seam (task 3857 step-7)
# ---------------------------------------------------------------------------


def _canned_mcp(rows, status_map):
    """Return ``(mcp_tool_call_fake, calls)`` emulating the fused-memory substrate.

    Faithful to what was traced for task 3857, because the whole point of the
    narrowing work is that the SERVER does the filtering:

    * ``get_tasks`` applies ``statuses`` as a row filter (SQL ``status IN``),
      then slices ``page_size``/``offset`` over a list ordered by ASCENDING
      ``id`` — so reaching the high-id end requires a computed offset.
    * ``get_statuses`` returns the compact ``{id: status}`` map.

    *rows* are raw MCP rows (string ids); *status_map* is ``{int id: status}``.
    """
    calls: list[dict] = []

    async def _mcp(client, url, tool, args, **_kw):
        # ``kwargs`` is recorded too so the per-request budget (``timeout=``,
        # which rides as a keyword and never inside ``args``) is assertable at
        # the wire — see test_every_per_project_call_carries_the_per_request_budget.
        calls.append({'tool': tool, 'args': dict(args), 'kwargs': dict(_kw)})
        if tool == 'get_statuses':
            return {'statuses': {str(k): v for k, v in status_map.items()}}
        if tool == 'get_tasks':
            statuses = args.get('statuses')
            selected = [
                r for r in rows
                if statuses is None or r.get('status') in statuses
            ]
            selected.sort(key=lambda r: int(r['id']))  # ORDER BY id ASC
            page_size = args.get('page_size')
            if page_size is not None:
                start = args.get('offset', 0)
                selected = selected[start:start + page_size]
            return {'tasks': selected}
        raise AssertionError(f'unexpected tool {tool!r}')

    return _mcp, calls


def _raw_row(task_id, status, *, title=None, updated_at=None):
    return {
        'id': str(task_id),
        'title': title or f'task {task_id}',
        'status': status,
        'dependencies': [],
        'metadata': {},
        'updatedAt': updated_at or f'2026-01-01T00:00:{task_id % 60:02d}+00:00',
    }


class TestShapeOneProjectNarrowing:
    """The Tasks-tab fetch must have a ceiling that does not grow with the tree.

    Asserted at the MCP wire, through the real ``fetch_tasks`` /
    ``fetch_statuses``, because "the dashboard discards the done rows
    afterwards" is precisely the defect — what matters is which rows the
    server was asked for.
    """

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        import dashboard.data.tasks as tasks_mod
        tasks_mod._fetch_tasks_cache_clear()
        _register_runtime(monkeypatch, {})
        yield
        tasks_mod._fetch_tasks_cache_clear()

    @staticmethod
    def _one_project_config(tmp_path):
        root = tmp_path / 'dark-factory'
        root.mkdir(parents=True, exist_ok=True)
        return DashboardConfig(project_root=root)

    @staticmethod
    def _get_tasks_calls(calls):
        return [c for c in calls if c['tool'] == 'get_tasks']

    async def test_scheduler_path_never_asks_for_a_terminal_row(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(a) caps 0/0 → exactly two calls, and 'done' never crosses the wire."""
        from dashboard.data.active_tasks import _ACTIVE_STATUSES, _shape_one_project

        rows = [_raw_row(1, 'in-progress'), _raw_row(2, 'pending')]
        rows += [_raw_row(i, 'done') for i in range(10, 60)]
        status_map = {int(r['id']): r['status'] for r in rows}
        mcp, calls = _canned_mcp(rows, status_map)
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        config = self._one_project_config(tmp_path)
        active, offline, done_count = await _shape_one_project(
            dummy_client, config, config.project_root,
            max_done_per_project=0, max_cancelled_per_project=0,
        )

        assert offline is False
        assert len(calls) == 2, f'expected exactly 2 MCP calls, got {calls}'
        tools = sorted(c['tool'] for c in calls)
        assert tools == ['get_statuses', 'get_tasks']

        get_tasks_call = self._get_tasks_calls(calls)[0]
        assert get_tasks_call['args'].get('statuses') == sorted(_ACTIVE_STATUSES)
        for call in calls:
            requested = call['args'].get('statuses') or []
            assert 'done' not in requested and 'cancelled' not in requested, (
                f'the scheduler path must never request terminal rows: {call}'
            )
        assert {r['title'] for r in active} == {'task 1', 'task 2'}
        assert done_count == 50

    async def test_every_per_project_call_carries_the_per_request_budget(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """The budget ROSTER must describe the shipped calls, not merely count them.

        ``test_tasks_budget.py`` machine-checks
        ``DEFAULT_PER_CALL_TIMEOUT * len(_PER_PROJECT_MCP_CALLS) <=
        _TASKS_PER_PROJECT_BUDGET``.  That arithmetic is only a true statement
        ABOUT THIS SYSTEM if every enumerated call actually threads the term.
        ``fetch_statuses`` shipped without it, so one of the three ran on
        ``mcp_tool_call``'s 10 s default and could alone overrun the 7 s
        per-project budget the roster claims to bound — a constants-only test
        cannot see that, which is why this one asserts at the WIRE.

        Driving the full three-call path (caps > 0) also means adding a fourth
        per-project call without the keyword fails here, rather than silently
        widening the budget.
        """
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.active_tasks import (
            _PER_PROJECT_MCP_CALLS,
            _shape_one_project,
        )

        rows = [_raw_row(1, 'in-progress')]
        rows += [_raw_row(i, 'done') for i in range(10, 20)]
        status_map = {int(r['id']): r['status'] for r in rows}
        mcp, calls = _canned_mcp(rows, status_map)
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        config = self._one_project_config(tmp_path)
        await _shape_one_project(
            dummy_client, config, config.project_root,
            max_done_per_project=50, max_cancelled_per_project=50,
        )

        assert len(calls) == len(_PER_PROJECT_MCP_CALLS), (
            f'the roster enumerates {len(_PER_PROJECT_MCP_CALLS)} per-project '
            f'calls {_PER_PROJECT_MCP_CALLS} but {len(calls)} were issued: '
            f'{[c["tool"] for c in calls]}'
        )
        for call in calls:
            assert call['kwargs'].get('timeout') == tasks_mod.DEFAULT_PER_CALL_TIMEOUT, (
                f"{call['tool']} was issued without the per-request budget "
                f"(timeout={call['kwargs'].get('timeout')!r}) — it falls back to "
                "mcp_tool_call's 10s default, so the per-project budget "
                'arithmetic in test_tasks_budget.py does not describe it'
            )

    async def test_tasks_tab_path_issues_a_bounded_terminal_window(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(b) caps 50/50 → three calls, the third a bounded high-id window."""
        import dashboard.data.active_tasks as at_mod
        from dashboard.data.active_tasks import _ACTIVE_STATUSES, _shape_one_project

        rows = [_raw_row(1, 'in-progress')]
        rows += [_raw_row(i, 'done') for i in range(100, 120)]
        rows += [_raw_row(i, 'cancelled') for i in range(200, 205)]
        status_map = {int(r['id']): r['status'] for r in rows}
        n_terminal = sum(1 for s in status_map.values() if s in ('done', 'cancelled'))
        mcp, calls = _canned_mcp(rows, status_map)
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        config = self._one_project_config(tmp_path)
        await _shape_one_project(
            dummy_client, config, config.project_root,
            max_done_per_project=50, max_cancelled_per_project=50,
        )

        assert len(calls) == 3, f'expected exactly 3 MCP calls, got {calls}'
        get_tasks_calls = self._get_tasks_calls(calls)
        assert len(get_tasks_calls) == 2
        active_call, terminal_call = get_tasks_calls
        assert active_call['args'].get('statuses') == sorted(_ACTIVE_STATUSES)
        assert terminal_call['args'].get('statuses') == ['cancelled', 'done']
        window = at_mod._TERMINAL_FETCH_WINDOW
        assert terminal_call['args'].get('page_size') == window
        assert terminal_call['args'].get('offset') == max(0, n_terminal - window)

    async def test_done_count_comes_from_the_compact_map_not_the_rows(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(c) The status map has MORE done tasks than the window can return.

        A ``done_count`` still derived from returned rows is provably wrong
        here — that is the whole reason it moved to the compact seam.
        """
        import dashboard.data.active_tasks as at_mod
        from dashboard.data.active_tasks import _shape_one_project

        monkeypatch.setattr(at_mod, '_TERMINAL_FETCH_WINDOW', 4)
        rows = [_raw_row(1, 'in-progress')]
        rows += [_raw_row(i, 'done') for i in range(100, 120)]  # 20 done rows
        status_map = {int(r['id']): r['status'] for r in rows}
        mcp, _calls = _canned_mcp(rows, status_map)
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        config = self._one_project_config(tmp_path)
        active, offline, done_count = await _shape_one_project(
            dummy_client, config, config.project_root,
            max_done_per_project=50, max_cancelled_per_project=50,
        )

        assert offline is False
        emitted_done = [r for r in active if r.get('status') == 'done']
        assert len(emitted_done) == 4, 'sanity: the window really did bound the rows'
        assert done_count == 20, (
            f'done_count must come from the compact status map (20), not the '
            f'{len(emitted_done)} rows the window returned'
        )

    async def test_window_reaches_the_high_id_end(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """The offset must select the HIGHEST ids, not the oldest ones.

        ``page_size``/``offset`` slice an ASCENDING-id list, so a naive
        ``offset=0`` would return the oldest terminal rows — the opposite of
        what the tab renders.
        """
        import dashboard.data.active_tasks as at_mod
        from dashboard.data.active_tasks import _shape_one_project

        monkeypatch.setattr(at_mod, '_TERMINAL_FETCH_WINDOW', 3)
        rows = [_raw_row(1, 'in-progress')]
        rows += [_raw_row(i, 'done') for i in range(100, 110)]
        status_map = {int(r['id']): r['status'] for r in rows}
        mcp, _calls = _canned_mcp(rows, status_map)
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        config = self._one_project_config(tmp_path)
        active, _offline, _done = await _shape_one_project(
            dummy_client, config, config.project_root,
            max_done_per_project=50, max_cancelled_per_project=50,
        )

        emitted = sorted(
            int(r['id'].rsplit('T-', 1)[-1])
            for r in active if r.get('status') == 'done'
        )
        assert emitted == [107, 108, 109], (
            f'the window must reach the high-id end, got {emitted}'
        )

    async def test_no_truncation_when_population_fits_the_window(
        self, monkeypatch, tmp_path, dummy_client, caplog
    ):
        """(d) n_terminal <= window → offset 0, every terminal row present, no WARNING."""
        import logging

        import dashboard.data.active_tasks as at_mod
        from dashboard.data.active_tasks import _shape_one_project

        monkeypatch.setattr(at_mod, '_TERMINAL_FETCH_WINDOW', 10)
        rows = [_raw_row(1, 'in-progress')]
        rows += [_raw_row(i, 'done') for i in range(100, 104)]
        rows += [_raw_row(i, 'cancelled') for i in range(200, 202)]
        status_map = {int(r['id']): r['status'] for r in rows}
        mcp, calls = _canned_mcp(rows, status_map)
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        config = self._one_project_config(tmp_path)
        with caplog.at_level(logging.WARNING, logger='dashboard.data.active_tasks'):
            active, _offline, _done = await _shape_one_project(
                dummy_client, config, config.project_root,
                max_done_per_project=50, max_cancelled_per_project=50,
            )

        terminal_call = self._get_tasks_calls(calls)[1]
        assert terminal_call['args'].get('offset') == 0
        terminal_rows = [r for r in active if r.get('status') in ('done', 'cancelled')]
        assert len(terminal_rows) == 6, 'every terminal row must be present'
        assert not [
            r for r in caplog.records
            if r.name == 'dashboard.data.active_tasks' and r.levelno >= logging.WARNING
        ], 'no truncation WARNING may fire when the population fits'

    async def test_truncation_warns_naming_project_and_counts(
        self, monkeypatch, tmp_path, dummy_client, caplog
    ):
        """(e) n_terminal > window → a WARNING naming the project, count and window.

        No silent cap: the window is a real behaviour change (selection by
        descending id rather than updated_at), so a reader has to be able to
        see when it bit.
        """
        import logging

        import dashboard.data.active_tasks as at_mod
        from dashboard.data.active_tasks import _shape_one_project

        monkeypatch.setattr(at_mod, '_TERMINAL_FETCH_WINDOW', 3)
        rows = [_raw_row(1, 'in-progress')]
        rows += [_raw_row(i, 'done') for i in range(100, 112)]  # 12 terminal
        status_map = {int(r['id']): r['status'] for r in rows}
        mcp, _calls = _canned_mcp(rows, status_map)
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        config = self._one_project_config(tmp_path)
        with caplog.at_level(logging.WARNING, logger='dashboard.data.active_tasks'):
            await _shape_one_project(
                dummy_client, config, config.project_root,
                max_done_per_project=50, max_cancelled_per_project=50,
            )

        messages = [
            r.getMessage() for r in caplog.records
            if r.name == 'dashboard.data.active_tasks' and r.levelno >= logging.WARNING
        ]
        assert any(
            'dark-factory' in m and '12' in m and '3' in m for m in messages
        ), f'expected a truncation WARNING naming project/count/window, got {messages}'

    async def test_offline_active_fetch_still_reports_the_project_offline(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(f) The existing offline contract is preserved by the new call shape."""
        import httpx

        from dashboard.data.active_tasks import _shape_one_project

        async def _refuse(client, url, tool, args, **_kw):
            raise httpx.ConnectError('refused')

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _refuse)

        config = self._one_project_config(tmp_path)
        active, offline, done_count = await _shape_one_project(
            dummy_client, config, config.project_root,
            max_done_per_project=50, max_cancelled_per_project=50,
        )

        assert offline is True
        assert active == []
        assert done_count == 0

    async def test_status_map_offline_degrades_the_count_not_the_project(
        self, monkeypatch, tmp_path, dummy_client, caplog
    ):
        """A failed compact-map read must not declare an otherwise-healthy project offline."""
        import logging

        import httpx

        from dashboard.data.active_tasks import _shape_one_project

        rows = [_raw_row(1, 'in-progress'), _raw_row(100, 'done')]

        async def _statuses_fail(client, url, tool, args, **_kw):
            if tool == 'get_statuses':
                raise httpx.ConnectError('refused')
            statuses = args.get('statuses')
            selected = [
                r for r in rows
                if statuses is None or r.get('status') in statuses
            ]
            return {'tasks': selected}

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _statuses_fail)

        config = self._one_project_config(tmp_path)
        with caplog.at_level(logging.WARNING, logger='dashboard.data.active_tasks'):
            active, offline, done_count = await _shape_one_project(
                dummy_client, config, config.project_root,
                max_done_per_project=50, max_cancelled_per_project=50,
            )

        assert offline is False, 'the active fetch succeeded — the project is not offline'
        assert [r['title'] for r in active if r.get('status') == 'in-progress'] == ['task 1']
        assert done_count is None, (
            'done_count must be UNKNOWN, not a fabricated 0: the terminal window '
            'is skipped when the map is unavailable, so counting the fetched rows '
            'would report zero done tasks for a project that has them'
        )
        assert any(
            r.name == 'dashboard.data.active_tasks'
            and r.levelno >= logging.WARNING
            and 'dark-factory' in r.getMessage()
            for r in caplog.records
        ), 'the degraded count must be logged, not silent'

    async def test_offline_status_map_omits_the_project_from_done_counts(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """An UNKNOWN done_count must not reach the payload as an authoritative 0.

        The front end treats a MISSING ``DONE_COUNTS`` entry as "no
        authoritative count" and falls back to its own row count
        (``DF_T.DONE_COUNTS[p.id] != null`` in tab_tasks.jsx).  Writing a 0
        would instead assert, with authority, that the project has completed
        nothing.
        """
        import httpx

        from dashboard.data.active_tasks import collect_tasks_with_counts

        rows = [_raw_row(1, 'in-progress'), _raw_row(100, 'done')]

        async def _statuses_fail(client, url, tool, args, **_kw):
            if tool == 'get_statuses':
                raise httpx.ConnectError('refused')
            statuses = args.get('statuses')
            return {'tasks': [
                r for r in rows
                if statuses is None or r.get('status') in statuses
            ]}

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _statuses_fail)

        config = self._one_project_config(tmp_path)
        _active, offline_projects, done_counts, degraded, _unknown = await collect_tasks_with_counts(
            dummy_client, config,
            max_done_per_project=50, max_cancelled_per_project=50,
        )

        assert offline_projects == [], 'the active fetch succeeded — not offline'
        assert degraded == [], 'the budget was not exceeded — not degraded'
        assert 'dark-factory' not in done_counts, (
            'a project whose count is UNKNOWN must be OMITTED from DONE_COUNTS, '
            f'not written as a fabricated value; got {done_counts!r}'
        )

    async def test_offline_status_map_never_emits_the_oldest_terminal_rows(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """A failed compact-map read must not repopulate the tab with ANCIENT rows.

        Regression for the review finding on the task-3857 branch.  The
        terminal window is positioned by ``offset = n_terminal - window``,
        and ``n_terminal`` comes from the compact map.  When that read failed,
        ``status_map`` was reset to ``{}`` so ``n_terminal`` was 0 and the
        offset collapsed to ``max(0, 0 - window) == 0``.  Because
        ``page_size``/``offset`` slice an ASCENDING-id list, offset 0 selects
        the OLDEST terminal rows — which were then sorted by ``updated_at``
        descending and emitted as the Tasks tab's *most recent* done list.

        The previous test at this seam used two rows, so ``n_terminal <=
        window`` held either way and the case could not fire.  This one uses
        strictly MORE terminal rows than the window.
        """
        import httpx

        import dashboard.data.active_tasks as at_mod
        from dashboard.data.active_tasks import _shape_one_project

        monkeypatch.setattr(at_mod, '_TERMINAL_FETCH_WINDOW', 3)
        # 10 done rows; ids 100..109 ascend with age (100 = oldest completion).
        rows = [_raw_row(1, 'in-progress')]
        rows += [_raw_row(i, 'done') for i in range(100, 110)]

        calls: list[dict] = []

        async def _statuses_fail(client, url, tool, args, **_kw):
            calls.append({'tool': tool, 'args': args})
            if tool == 'get_statuses':
                raise httpx.ConnectError('refused')
            statuses = args.get('statuses')
            selected = [
                r for r in rows
                if statuses is None or r.get('status') in statuses
            ]
            offset = args.get('offset') or 0
            page_size = args.get('page_size')
            selected = selected[offset:offset + page_size] if page_size else selected[offset:]
            return {'tasks': selected}

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _statuses_fail)

        config = self._one_project_config(tmp_path)
        active, offline, done_count = await _shape_one_project(
            dummy_client, config, config.project_root,
            max_done_per_project=50, max_cancelled_per_project=50,
        )

        assert offline is False, 'the active fetch succeeded — the project is not offline'
        assert done_count is None, 'the count is UNKNOWN without the map'

        # (1) No unpositionable terminal fetch is issued at all.
        terminal_calls = [
            c for c in calls
            if c['tool'] == 'get_tasks'
            and 'done' in (c['args'].get('statuses') or [])
        ]
        assert terminal_calls == [], (
            'the terminal window cannot be positioned without the compact map, '
            f'so it must not be fetched; got {terminal_calls!r}'
        )

        # (2) And therefore no ancient row is presented as recent.
        emitted_done = [r for r in active if r.get('status') == 'done']
        assert emitted_done == [], (
            'omitting done rows is honest; showing the OLDEST rows as the '
            f'newest is not. Got ids {[r.get("id") for r in emitted_done]}'
        )

        # (3) The healthy active row still renders — this is a partial
        # degradation, not an offline project.
        assert [r['title'] for r in active if r.get('status') == 'in-progress'] == ['task 1']


# ---------------------------------------------------------------------------
# TestDepsOutsideTheTerminalWindow — dependency chips must not silently vanish
# (task 3857 step-9)
# ---------------------------------------------------------------------------


class TestDepsOutsideTheTerminalWindow:
    """A bounded terminal fetch must not silently delete dependency chips.

    ``_resolve_deps`` used to read a ``by_id`` built over the WHOLE tree, so
    every dep id resolved.  After the terminal window bounds what is fetched,
    a done dependency outside the window is no longer in ``by_id`` — and the
    ``continue`` would drop a chip that renders today.  The compact status map
    is the bounded source that keeps the load-bearing half of the chip (the
    ``done`` flag) honest.
    """

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        import dashboard.data.tasks as tasks_mod
        tasks_mod._fetch_tasks_cache_clear()
        _register_runtime(monkeypatch, {})
        yield
        tasks_mod._fetch_tasks_cache_clear()

    @staticmethod
    def _config(tmp_path):
        root = tmp_path / 'proj'
        root.mkdir(parents=True, exist_ok=True)
        return DashboardConfig(project_root=root)

    async def _shape(self, monkeypatch, tmp_path, dummy_client, *, window=2):
        """One active task depending on ids inside, outside and beyond the tree."""
        import dashboard.data.active_tasks as at_mod
        from dashboard.data.active_tasks import _shape_one_project

        monkeypatch.setattr(at_mod, '_TERMINAL_FETCH_WINDOW', window)

        rows = [
            _raw_row(1, 'in-progress', title='the active one'),
            _raw_row(5, 'pending', title='an active dep'),
            # Low-id done dep: present in the status map, pushed OUT of the
            # high-id terminal window by the ids below.
            _raw_row(10, 'done', title='long-parked dep'),
            _raw_row(90, 'done', title='recent dep'),
            _raw_row(91, 'done', title='recenter dep'),
        ]
        rows[0]['dependencies'] = ['5', '10', '90', '999']
        status_map = {int(r['id']): r['status'] for r in rows}
        mcp, _calls = _canned_mcp(rows, status_map)
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        config = self._config(tmp_path)
        active, _offline, _done = await _shape_one_project(
            dummy_client, config, config.project_root,
            max_done_per_project=50, max_cancelled_per_project=50,
        )
        row = next(r for r in active if r.get('status') == 'in-progress')
        return {int(d['id'].rsplit('T-', 1)[-1]): d for d in row['deps']}

    async def test_done_dep_outside_the_window_is_still_emitted(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(a) An honest partial entry beats a dropped chip."""
        deps = await self._shape(monkeypatch, tmp_path, dummy_client)

        assert 10 in deps, (
            'a done dependency outside the terminal window must still render a '
            f'chip — got only {sorted(deps)}'
        )
        assert deps[10]['done'] is True, 'the done flag comes from the status map'
        assert deps[10]['title'] == '', (
            'the title is unresolvable without an extra whole-tree read, so it '
            'degrades to the empty string the shape already allows'
        )

    async def test_dep_present_in_the_window_keeps_its_real_title(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(b) No regression for deps whose full row was actually fetched."""
        deps = await self._shape(monkeypatch, tmp_path, dummy_client)

        assert deps[90]['title'] == 'recent dep'
        assert deps[90]['done'] is True

    async def test_dep_absent_from_rows_and_map_is_still_dropped(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(c) The id does not exist — fabricating a chip is worse than omitting it."""
        deps = await self._shape(monkeypatch, tmp_path, dummy_client)

        assert 999 not in deps, (
            'an id absent from both the rows and the status map must be dropped'
        )

    async def test_active_dep_resolves_with_done_false(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(d) A non-done dep is emitted with done False, however it resolved."""
        deps = await self._shape(monkeypatch, tmp_path, dummy_client)

        assert deps[5]['done'] is False
        assert deps[5]['title'] == 'an active dep'

    async def test_active_dep_outside_the_rows_yields_done_false(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(d) The status-map fallback must not assume 'not fetched' means done."""
        import dashboard.data.active_tasks as at_mod
        from dashboard.data.active_tasks import _shape_one_project

        monkeypatch.setattr(at_mod, '_TERMINAL_FETCH_WINDOW', 1)
        rows = [
            _raw_row(1, 'in-progress', title='the active one'),
            _raw_row(7, 'blocked', title='a blocked dep'),
            _raw_row(80, 'done'),
            _raw_row(81, 'done'),
        ]
        rows[0]['dependencies'] = ['7', '80']
        status_map = {int(r['id']): r['status'] for r in rows}

        async def _mcp(client, url, tool, args, **_kw):
            if tool == 'get_statuses':
                return {'statuses': {str(k): v for k, v in status_map.items()}}
            statuses = args.get('statuses')
            # Deliberately omit the blocked row from the ACTIVE rows so its
            # only source is the compact map.
            selected = [
                r for r in rows
                if (statuses is None or r.get('status') in statuses)
                and r['id'] != '7'
            ]
            selected.sort(key=lambda r: int(r['id']))
            page_size = args.get('page_size')
            if page_size is not None:
                start = args.get('offset', 0)
                selected = selected[start:start + page_size]
            return {'tasks': selected}

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _mcp)

        config = self._config(tmp_path)
        active, _offline, _done = await _shape_one_project(
            dummy_client, config, config.project_root,
            max_done_per_project=50, max_cancelled_per_project=50,
        )
        row = next(r for r in active if r.get('status') == 'in-progress')
        deps = {int(d['id'].rsplit('T-', 1)[-1]): d for d in row['deps']}

        assert deps[7]['done'] is False, 'a blocked dep must never render as done'
        assert deps[7]['title'] == ''


# ---------------------------------------------------------------------------
# collect_tasks_with_counts whole-handler budget (task 3857 steps 13/14)
# ---------------------------------------------------------------------------


# Far beyond any budget exercised below, so a project that sleeps this long can
# only ever end by being CUT OFF. Picking a number near the budget instead would
# make "did the deadline fire?" a race rather than a fact.
_BUDGET_SLOW = 5.0


def _register_shaper(monkeypatch, delays, *, offline=(), done_counts=None,
                     external_deps=None, raises=None):
    """Patch ``_shape_one_project`` with a per-project coroutine that sleeps.

    *delays* maps project label -> seconds to sleep before returning; a label
    absent from it returns immediately. Labels in *offline* return the offline
    marker triple ``([], True, 0)`` — a fetch that demonstrably FAILED, which
    must stay distinguishable from a project the budget never reached.

    *raises* maps project label -> an exception INSTANCE to raise instead of
    returning. Models the unexpected-failure path (a shape bug, a decode
    error, an ``httpx`` transport error escaping the fan-out) as distinct from
    both the timeout and the offline marker.

    *external_deps* is an optional list of external dep ids stamped onto every
    emitted row (in the ``_build_task_row`` shape, each on the ``'unknown'``
    sentinel), so the batched ``fetch_external_statuses`` leg of the handler is
    reachable from this harness — without it ``dep_ids`` is empty and that leg
    short-circuits.

    Returns the list of labels ``_shape_one_project`` was actually INVOKED
    with, in order. That record is what makes "never got its turn" a checkable
    fact rather than an inference from an absence in the output.
    """
    invoked: list[str] = []
    counts = done_counts or {}
    explode = raises or {}

    async def _fake_shape(
        client, config, project_root, *,
        max_done_per_project=0, max_cancelled_per_project=0,
        now=None, runtime=None,
    ):
        label = project_root.name
        invoked.append(label)
        delay = delays.get(label, 0.0)
        if delay:
            await asyncio.sleep(delay)
        if label in explode:
            raise explode[label]
        if label in offline:
            return [], True, 0
        row = {'id': f'{label}/T-1', 'project': label, 'status': 'in-progress'}
        if external_deps:
            row['external_deps'] = [
                {'id': dep, 'status': 'unknown'} for dep in external_deps
            ]
        return [row], False, counts.get(label, 0)

    monkeypatch.setattr('dashboard.data.active_tasks._shape_one_project', _fake_shape)
    return invoked


def _budget_config(tmp_path, labels):
    """A DashboardConfig whose project roots are *labels*, primary first."""
    roots = []
    for label in labels:
        root = tmp_path / label
        root.mkdir(parents=True, exist_ok=True)
        roots.append(root)
    return DashboardConfig(project_root=roots[0], known_project_roots=roots[1:])


class TestCollectTasksBudget:
    """The Tasks-tab aggregation must be bounded as a WHOLE, and degrade honestly.

    ``collect_tasks_with_counts`` walks every configured project root
    sequentially with no deadline anywhere, so its worst case is the SUM of
    every project's worst case — unbounded in the number of roots. The fix is
    the ``/healthz`` shape: one ``loop.time()`` deadline for the handler, one
    ``asyncio.wait_for`` per project, and — the part that is easy to get wrong
    — an explicit marker for every project the budget did not reach.

    That last part is the real contract here. A truncated-but-confident payload
    (rows for the projects that finished, silence for the rest) renders as "no
    active work" on those projects, which is the same invisible-failure class
    the fan-out logging policy was raised to WARNING to close. *degraded*
    (budget expired — state UNKNOWN) is a strictly different fact from
    *offline* (fetch demonstrably failed), and the two must never be merged.
    """

    @pytest.fixture(autouse=True)
    def _no_runtime_fanout(self, monkeypatch):
        """Runtime fan-out returns instantly, so every measured second is the loop's."""
        _register_runtime(monkeypatch, {})

    @staticmethod
    def _tighten(monkeypatch, *, total, per_project):
        monkeypatch.setattr('dashboard.data.active_tasks._TASKS_TOTAL_BUDGET', total)
        monkeypatch.setattr(
            'dashboard.data.active_tasks._TASKS_PER_PROJECT_BUDGET', per_project
        )

    async def test_returns_five_element_tuple_with_degraded_and_unknown(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(a) the return shape carries degraded_projects AND count_unknown."""
        _register_shaper(monkeypatch, {}, done_counts={'alpha': 3, 'beta': 5})
        config = _budget_config(tmp_path, ['alpha', 'beta'])

        result = await collect_tasks_with_counts(client=dummy_client, config=config)

        assert len(result) == 5, (
            'expected (active, offline, done_counts, degraded, count_unknown), '
            f'got {len(result)} elements — a project the budget never reached '
            'has nowhere to be reported without the fourth list, and one whose '
            'status map failed has nowhere without the fifth'
        )
        _active, offline, counts, degraded, count_unknown = result
        assert degraded == []
        assert offline == []
        assert count_unknown == []
        assert counts == {'alpha': 3, 'beta': 5}

    async def test_deadline_expiry_marks_unreached_projects_degraded(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(b) projects the handler never reached are named, not silently dropped.

        Budgets are chosen so the arithmetic is one-directional rather than a
        race: ``alpha`` returns instantly, then ``beta`` and ``gamma`` each
        sleep far past their per-project budget and so consume 0.2s + 0.1s =
        the entire 0.3s handler budget. Timers overshoot and never undershoot,
        so ``delta`` and ``epsilon`` are guaranteed to find a non-positive
        remaining budget — they can only be reached by the deadline branch.
        """
        invoked = _register_shaper(
            monkeypatch,
            {'beta': _BUDGET_SLOW, 'gamma': _BUDGET_SLOW},
            done_counts={'alpha': 7},
        )
        self._tighten(monkeypatch, total=0.3, per_project=0.2)
        config = _budget_config(
            tmp_path, ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        )

        active, offline, counts, degraded, _unknown = await collect_tasks_with_counts(
            client=dummy_client, config=config,
        )

        # The project that completed still contributes its rows and its count.
        assert [row['project'] for row in active] == ['alpha']
        assert counts == {'alpha': 7}

        # Everything the budget did not deliver is NAMED.
        assert set(degraded) == {'beta', 'gamma', 'delta', 'epsilon'}

        # ...and the two never-reached projects are provably never-reached:
        # they were not invoked at all, so their degraded marker cannot have
        # come from a per-project timeout.
        assert 'delta' not in invoked and 'epsilon' not in invoked, (
            f'expected the handler deadline to skip delta/epsilon, but it '
            f'invoked {invoked}'
        )

        # Never proven unreachable -> never reported offline.
        assert offline == []
        # No count was measured -> none is fabricated (not even a 0, which
        # would render as a real "this project has zero done tasks").
        assert 'delta' not in counts and 'epsilon' not in counts

    async def test_slow_project_is_cut_off_and_the_next_one_still_runs(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(c) one slow project must not starve its neighbours of their turn.

        The handler budget is left generous here so the ONLY thing that can
        cut ``beta`` short is its own per-project budget — which is what makes
        ``gamma`` completing a fact about per-project containment rather than
        a coincidence of the total.
        """
        invoked = _register_shaper(
            monkeypatch,
            {'beta': _BUDGET_SLOW},
            done_counts={'alpha': 1, 'gamma': 2},
        )
        self._tighten(monkeypatch, total=10.0, per_project=0.2)
        config = _budget_config(tmp_path, ['alpha', 'beta', 'gamma'])

        started = time.monotonic()
        active, offline, counts, degraded, _unknown = await collect_tasks_with_counts(
            client=dummy_client, config=config,
        )
        elapsed = time.monotonic() - started

        assert degraded == ['beta']
        assert offline == []
        assert 'gamma' in invoked, 'the project after the slow one never got its turn'
        assert {row['project'] for row in active} == {'alpha', 'gamma'}
        assert counts == {'alpha': 1, 'gamma': 2}
        assert elapsed < 1.0, (
            f'elapsed {elapsed:.3f}s — beta sleeps {_BUDGET_SLOW}s, so anything '
            'near that means the per-project budget did not fire'
        )

    async def test_degraded_and_offline_are_disjoint(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(d) a project is either demonstrably offline or unknown — never both.

        Merging the lists would let an operator read "the budget ran out" as
        "fused-memory is down", which sends them to restart a healthy service.
        """
        _register_shaper(
            monkeypatch,
            {'gamma': _BUDGET_SLOW},
            offline=('beta',),
            done_counts={'alpha': 4},
        )
        self._tighten(monkeypatch, total=10.0, per_project=0.2)
        config = _budget_config(tmp_path, ['alpha', 'beta', 'gamma'])

        _active, offline, counts, degraded, _unknown = await collect_tasks_with_counts(
            client=dummy_client, config=config,
        )

        assert offline == ['beta']
        assert degraded == ['gamma']
        assert set(offline).isdisjoint(degraded)
        # An offline project already had no count; a degraded one must not
        # acquire a fabricated one either.
        assert counts == {'alpha': 4}

    async def test_one_projects_unexpected_error_cannot_blank_the_whole_tab(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(e) an UNEXPECTED exception from one root must not 500 the handler.

        The per-project ``try`` caught ``TimeoutError`` only, so any other
        exception escaping ``_shape_one_project`` — a decode error, a shape
        bug, an ``httpx`` transport error not converted to an offline marker by
        the fan-out — propagated out of the whole aggregation and 500'd
        ``/api/v2/dashboard/tasks``, discarding every HEALTHY project's rows
        that had already been collected.

        That is the same "one bad root blanks the whole tab" failure the
        ``TASKS_OFFLINE`` fix exists to close, relocated from the banner to the
        handler.  The fan-out normally converts failures into offline markers,
        so this is defense-in-depth rather than a demonstrated crash — which is
        exactly why it needs a test: nothing else exercises the path.

        The failing root is marked OFFLINE, not degraded: the read demonstrably
        failed, which is what *offline* means.  *degraded* is reserved for
        "the budget never let us find out".
        """
        invoked = _register_shaper(
            monkeypatch,
            {},
            done_counts={'alpha': 4, 'gamma': 7},
            raises={'beta': ValueError('malformed get_tasks payload')},
        )
        config = _budget_config(tmp_path, ['alpha', 'beta', 'gamma'])

        active, offline, counts, degraded, _unknown = await collect_tasks_with_counts(
            client=dummy_client, config=config,
        )

        # The walk CONTINUED past the exploding root rather than unwinding.
        assert invoked == ['alpha', 'beta', 'gamma'], (
            f'invoked {invoked} — gamma never got its turn, so the exception '
            'aborted the aggregation instead of being contained to beta'
        )
        assert offline == ['beta']
        assert degraded == [], 'the budget did not expire — nothing is UNKNOWN'
        # Every healthy project still renders, and beta contributes no
        # fabricated count.
        assert {row['project'] for row in active} == {'alpha', 'gamma'}
        assert counts == {'alpha': 4, 'gamma': 7}

    async def test_unexpected_error_is_logged_at_warning_with_the_project(
        self, monkeypatch, tmp_path, dummy_client, caplog
    ):
        """(f) ...and the swallowed exception must not be silent.

        Containing the failure is only half the fix: an exception absorbed into
        an offline marker with no log is a bug that renders as a routine
        outage forever.  The record must name the project and carry the
        traceback, so the next reader can tell "fused-memory is down" from
        "our shaping code raised".
        """
        _register_shaper(
            monkeypatch, {}, raises={'beta': ValueError('malformed get_tasks payload')},
        )
        config = _budget_config(tmp_path, ['alpha', 'beta'])

        with caplog.at_level(logging.WARNING, logger='dashboard.data.active_tasks'):
            await collect_tasks_with_counts(client=dummy_client, config=config)

        records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'beta' in r.getMessage()
        ]
        assert records, (
            'an unexpected per-project exception was swallowed with no WARNING'
        )
        assert any(r.exc_info for r in records), (
            'the WARNING carries no traceback — the exception type and origin '
            'are exactly what distinguishes this from a routine outage'
        )

    async def test_total_wall_time_is_bounded_by_the_handler_budget(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(e) the whole call is bounded, not merely each project within it."""
        import dashboard.data.active_tasks as active_tasks_mod

        delays = {label: _BUDGET_SLOW for label in ('beta', 'gamma', 'delta')}
        _register_shaper(monkeypatch, delays)
        self._tighten(monkeypatch, total=0.5, per_project=0.2)
        config = _budget_config(tmp_path, ['alpha', 'beta', 'gamma', 'delta'])

        started = time.monotonic()
        await collect_tasks_with_counts(client=dummy_client, config=config)
        elapsed = time.monotonic() - started

        sum_of_sleeps = sum(delays.values())
        assert elapsed < sum_of_sleeps / 2, (
            f'elapsed {elapsed:.3f}s is not well under the {sum_of_sleeps}s sum '
            'of per-project sleeps — the walk is still additive in the number '
            'of roots'
        )
        # +0.5s of tolerance for event-loop scheduling, the same convention as
        # test_healthz_deadline.py's elapsed assertions.
        assert elapsed < active_tasks_mod._TASKS_TOTAL_BUDGET + 0.5, (
            f'elapsed {elapsed:.3f}s exceeded the whole-handler budget of '
            f'{active_tasks_mod._TASKS_TOTAL_BUDGET}s'
        )

    async def test_external_status_fetch_cannot_overrun_the_handler_budget(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(e2) the batched external-dep call is BOUNDED, not merely deadline-CHECKED.

        It runs AFTER the per-project walk, guarded only by an
        ``ext_remaining <= 0`` skip.  With a small POSITIVE remainder the call
        still proceeded unbounded on ``mcp_tool_call``'s 10 s default — and a
        cold MCP session performs three posts, per fan-out URL — so the handler
        could exceed ``_TASKS_TOTAL_BUDGET`` by ~30 s and blow past ``data.js``'s
        30 000 ms fetch abort.  That is precisely the "the degraded payload is
        aborted before it can be rendered" failure ``test_tasks_budget.py``
        exists to prevent and structurally cannot see: it checks constants, and
        this leg simply did not honour them.

        Expiry leaves every entry on its honest ``'unknown'`` sentinel — the
        same treatment the ``ext_remaining <= 0`` skip and the per-project
        ``TimeoutError`` branch already give.
        """
        import dashboard.data.active_tasks as active_tasks_mod

        _register_shaper(
            monkeypatch, {}, external_deps=['dark_factory:13', 'reify:8'],
        )

        async def _slow_ext(client, config, deps):
            await asyncio.sleep(_BUDGET_SLOW)
            return {'dark_factory:13': 'done'}

        monkeypatch.setattr(
            'dashboard.data.active_tasks.fetch_external_statuses', _slow_ext
        )
        # A positive remainder when the external leg is reached: the
        # ext_remaining <= 0 skip must NOT be what saves us here, or the test
        # would pass against the unbounded code.
        self._tighten(monkeypatch, total=1.0, per_project=0.5)
        config = _budget_config(tmp_path, ['alpha'])

        started = time.monotonic()
        active, offline, _counts, _degraded, _unknown = await collect_tasks_with_counts(
            client=dummy_client, config=config, resolve_external=True,
        )
        elapsed = time.monotonic() - started

        assert elapsed < _BUDGET_SLOW / 2, (
            f'elapsed {elapsed:.3f}s is not well under the {_BUDGET_SLOW}s '
            'external-status sleep — the batched external-dep call is still '
            'unbounded, so the handler budget does not bound the handler'
        )
        assert elapsed < active_tasks_mod._TASKS_TOTAL_BUDGET + 0.5, (
            f'elapsed {elapsed:.3f}s exceeded the whole-handler budget of '
            f'{active_tasks_mod._TASKS_TOTAL_BUDGET}s'
        )
        # Degrade honestly: no status was read, so none is fabricated, and the
        # project is NOT declared offline (its rows loaded fine).
        assert offline == []
        assert active[0]['external_deps'] == [
            {'id': 'dark_factory:13', 'status': 'unknown'},
            {'id': 'reify:8', 'status': 'unknown'},
        ]

    async def test_happy_path_is_unchanged_by_the_budget(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(f) with fast projects, nothing degrades and the payload is identical."""
        _register_shaper(monkeypatch, {}, done_counts={'alpha': 11, 'beta': 22})
        # Shipped constants deliberately NOT tightened here: the happy path
        # must hold under the values that actually ship.
        config = _budget_config(tmp_path, ['alpha', 'beta'])

        active, offline, counts, degraded, _unknown = await collect_tasks_with_counts(
            client=dummy_client, config=config,
        )

        assert degraded == []
        assert offline == []
        assert counts == {'alpha': 11, 'beta': 22}
        assert [row['id'] for row in active] == ['alpha/T-1', 'beta/T-1']

    async def test_collect_active_tasks_still_returns_two_elements(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(g) the scheduler's caller keeps its two-element contract.

        ``data/scheduler.py`` unpacks ``(active, offline)``; the fourth element
        is absorbed by ``collect_active_tasks``, not leaked to it.
        """
        _register_shaper(monkeypatch, {'beta': _BUDGET_SLOW}, done_counts={'alpha': 1})
        self._tighten(monkeypatch, total=10.0, per_project=0.2)
        config = _budget_config(tmp_path, ['alpha', 'beta'])

        result = await collect_active_tasks(client=dummy_client, config=config)

        assert len(result) == 2, (
            f'collect_active_tasks must keep its (active, offline) shape, got '
            f'{len(result)} elements'
        )
        active, offline = result
        assert [row['project'] for row in active] == ['alpha']
        # A degraded project is NOT offline here either — the marker is
        # dropped by this narrower contract, not silently reclassified.
        assert offline == []


class TestActiveAndTerminalReadsAreDeduped:
    """Splitting one fetch into two must not double-emit a task.

    Regression for the task-3857 review finding. The active and terminal
    reads are separately cached, so a task completing between them appears
    in BOTH — and both loops emit a row sharing one ``_task_uid``, the id
    the React tab uses as its map key and selection identity.
    """

    @staticmethod
    def _one_project_config(tmp_path):
        return TestShapeOneProjectNarrowing._one_project_config(tmp_path)

    async def test_a_task_in_both_reads_emits_exactly_one_row(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """id 7 is 'pending' per the active read and 'done' per the terminal read."""
        from dashboard.data.active_tasks import _shape_one_project

        async def _skewed(client, url, tool, args, **_kw):
            if tool == 'get_statuses':
                return {'statuses': {1: 'in-progress', 7: 'done'}}
            statuses = args.get('statuses') or []
            if 'done' in statuses:
                # The terminal read is the NEWER snapshot: 7 has completed.
                return {'tasks': [_raw_row(7, 'done')]}
            # The active read is served from a cache entry predating that.
            return {'tasks': [_raw_row(1, 'in-progress'), _raw_row(7, 'pending')]}

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _skewed)

        config = self._one_project_config(tmp_path)
        active, offline, _done = await _shape_one_project(
            dummy_client, config, config.project_root,
            max_done_per_project=50, max_cancelled_per_project=50,
        )

        assert offline is False
        ids = [r['id'] for r in active]
        assert len(set(ids)) == len(ids), (
            f'a task present in both reads must emit ONE row; got {ids}'
        )
        seven = [r for r in active if r['id'].endswith('T-7')]
        assert len(seven) == 1, f'expected exactly one row for task 7, got {seven}'
        assert seven[0]['status'] == 'done', (
            'the terminal read is the newer snapshot and must win the dedup'
        )


class TestCountUnknownProjectsAreNamedOnTheWire:
    """A project whose count is UNKNOWN must not look healthy-with-zero."""

    @staticmethod
    def _one_project_config(tmp_path):
        return TestShapeOneProjectNarrowing._one_project_config(tmp_path)

    async def test_status_map_offline_project_is_named_count_unknown(
        self, monkeypatch, tmp_path, dummy_client
    ):
        import httpx

        from dashboard.data.active_tasks import collect_tasks_with_counts

        rows = [_raw_row(1, 'in-progress'), _raw_row(100, 'done')]

        async def _statuses_fail(client, url, tool, args, **_kw):
            if tool == 'get_statuses':
                raise httpx.ConnectError('refused')
            statuses = args.get('statuses')
            return {'tasks': [
                r for r in rows
                if statuses is None or r.get('status') in statuses
            ]}

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _statuses_fail)

        config = self._one_project_config(tmp_path)
        (
            _active, offline_projects, done_counts,
            degraded, count_unknown,
        ) = await collect_tasks_with_counts(
            dummy_client, config,
            max_done_per_project=50, max_cancelled_per_project=50,
        )

        assert offline_projects == [], 'the active fetch succeeded — not offline'
        assert degraded == [], 'nothing timed out — not degraded'
        assert 'dark-factory' not in done_counts, 'no fabricated count'
        assert count_unknown == ['dark-factory'], (
            'a project that is neither offline nor degraded but whose count was '
            'never measured must still be NAMED, or it renders as a healthy '
            f'project with a confident "0 done"; got {count_unknown!r}'
        )
