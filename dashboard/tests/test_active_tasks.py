"""Tests for the ACTIVE_TASKS aggregator that joins task tree + worktrees."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import (
    _attempts_from_review_summary,
    _build_task_row,
    _minutes_since,
    collect_active_tasks,
    collect_done_counts,
    collect_tasks_with_counts,
)

# ---------------------------------------------------------------------------
# helpers used inside the aggregator
# ---------------------------------------------------------------------------


def test_attempts_from_review_summary_parses_passed_string():
    assert _attempts_from_review_summary('2/5 passed') == 5
    assert _attempts_from_review_summary('0/3 passed') == 3


def test_attempts_from_review_summary_handles_dash_and_empty():
    assert _attempts_from_review_summary('—') == 0
    assert _attempts_from_review_summary('') == 0


def test_minutes_since_handles_z_suffix_and_naive_iso():
    one_hour_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace('+00:00', 'Z')
    assert 59 <= _minutes_since(one_hour_ago) <= 61


def test_minutes_since_returns_zero_on_missing_or_bad():
    assert _minutes_since(None) == 0
    assert _minutes_since('not-a-date') == 0


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


def _make_project(root, *, project_dir, tasks, worktrees=None):
    """Lay down per-worktree .task/ artifacts and return ``(project_root, shaped_tasks)``.

    The tasks themselves no longer live on disk — fused-memory MCP owns task
    state — so we return them in their dashboard-shaped form for the caller
    to register against ``fetch_tasks`` via monkeypatch.
    """
    project_root = root / project_dir
    project_root.mkdir(parents=True, exist_ok=True)

    if worktrees:
        worktrees_dir = project_root / '.worktrees'
        worktrees_dir.mkdir()
        for task_id, metadata, files, iteration_lines, review_files in worktrees:
            wt = worktrees_dir / str(task_id)
            wt.mkdir()
            task_dir = wt / '.task'
            task_dir.mkdir()
            if metadata is not None:
                (task_dir / 'metadata.json').write_text(json.dumps(metadata))
            if files is not None:
                (task_dir / 'plan.json').write_text(json.dumps({'steps': [], 'files': files}))
            if iteration_lines is not None:
                (task_dir / 'iterations.jsonl').write_text(
                    '\n'.join('{}' for _ in range(iteration_lines)) + ('\n' if iteration_lines else ''),
                )
            if review_files is not None:
                reviews = task_dir / 'reviews'
                reviews.mkdir()
                for i, verdict in enumerate(review_files):
                    (reviews / f'r{i}.json').write_text(json.dumps({'verdict': verdict}))

    return project_root, [_shape_task(t) for t in tasks]


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
        worktrees=[
            (19,
             {'task_id': '19', 'title': 'consolidation retry', 'created_at': started},
             ['src/agents/consolidation.py', 'src/store/graphiti_adapter.py'],
             2,  # iterations.jsonl lines
             ['PASS', 'FAIL', 'FAIL']),  # 1/3 passed → attempts == 3
            (21,
             {'task_id': '21', 'title': 'dedup index', 'created_at': started},
             ['src/store/dedup.py'],
             1,
             ['PASS']),
        ],
    )
    reify_root, reify_tasks = _make_project(
        tmp_path,
        project_dir='reify',
        tasks=[{'id': 8, 'title': 'parser recovery', 'status': 'blocked',
                'dependencies': []}],
        worktrees=[(8, {'task_id': '8', 'title': 'parser recovery',
                        'created_at': started},
                    ['parser/recovery.rs'], 0, [])],
    )

    by_root = {df_root.resolve(): df_tasks, reify_root.resolve(): reify_tasks}

    async def _fake_fetch_tasks(client, config, project_root):
        return list(by_root.get(project_root.resolve(), []))

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake_fetch_tasks)

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
async def test_collect_active_tasks_handles_missing_worktree_metadata(tmp_path, monkeypatch, dummy_client):
    """A pending task with no worktree should still appear with empty fields."""
    root, shaped = _make_project(
        tmp_path, project_dir='solo',
        tasks=[{'id': 1, 'title': 'lonely', 'status': 'pending', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake_fetch_tasks)
    cfg = DashboardConfig(project_root=root)
    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert active == [{
        'id': 'solo/T-1', 'project': 'solo', 'title': 'lonely',
        'description': '', 'details': '', 'status': 'pending', 'agent': None,
        'started': 0, 'loops': 0, 'attempts': 0, 'deps': [],
        'meta_files': [], 'train': None, 'external_deps': [],
    }]


@pytest.mark.asyncio
async def test_collect_active_tasks_surfaces_offline_projects(tmp_path, monkeypatch, dummy_client):
    """A project whose MCP fetch returns an offline marker is reported."""
    root = tmp_path / 'offline-project'
    root.mkdir()

    async def _fake_fetch_tasks(client, config, project_root):
        return {'offline': True, 'error': 'connection refused'}

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake_fetch_tasks)
    cfg = DashboardConfig(project_root=root)
    active, offline_projects = await collect_active_tasks(client=dummy_client, config=cfg)
    assert active == []
    assert offline_projects == ['offline-project']


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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake_fetch_tasks)
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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake)
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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake)
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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake)
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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake)
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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake)
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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake)
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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake)
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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _fake_ext_statuses)

    cfg = DashboardConfig(project_root=root)
    active, _, _ = await collect_tasks_with_counts(
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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _must_not_be_called)

    cfg = DashboardConfig(project_root=root)
    # Default resolve_external=False — must NOT call fetch_external_statuses.
    active, _, _ = await collect_tasks_with_counts(client=dummy_client, config=cfg)
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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake_fetch)
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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _must_not_be_called)

    cfg = DashboardConfig(project_root=root)
    active, _, _ = await collect_tasks_with_counts(
        client=dummy_client, config=cfg, resolve_external=True,
    )
    assert active[0]['external_deps'] == []


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_skips_done_rows(
    tmp_path, monkeypatch, dummy_client,
):
    """resolve_external=True must NOT include done rows' external dep ids in the batched call.

    Done tasks' external deps are no longer actionable. Their ids must not bloat the MCP
    request, and their rows must keep the 'unknown' sentinel (not get re-stamped).
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
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    calls: list[list[str]] = []

    async def _record_call(client, config, deps):
        calls.append(sorted(deps))
        return {'proj:10': 'done'}

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _record_call)

    cfg = DashboardConfig(project_root=root)
    active, _, _ = await collect_tasks_with_counts(
        client=dummy_client, config=cfg,
        max_done_per_project=5, resolve_external=True,
    )

    # Only the active row's dep id should be in the batched call — NOT 'proj:99'.
    assert calls == [['proj:10']], (
        f'done row dep "proj:99" must not appear in the batched call; got {calls}'
    )

    by_id = {r['id']: r for r in active}
    # Active row's dep was resolved.
    assert by_id['xdeps/T-5']['external_deps'] == [{'id': 'proj:10', 'status': 'done'}]
    # Done row's dep kept 'unknown' (was not re-stamped).
    assert by_id['xdeps/T-6']['external_deps'] == [{'id': 'proj:99', 'status': 'unknown'}]


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

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_tasks', _fake_fetch)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)

    ids = {t['id'] for t in active}
    assert 'proj/T-30' in ids, (
        "deferred task T-30 was dropped by the active-status filter — "
        "add 'deferred' to _ACTIVE_STATUSES"
    )

    by_id = {t['id']: t for t in active}
    row = by_id['proj/T-30']

    # (b) resolved deps via active path — done flag on the dep
    assert row['deps'] == [{'id': 'proj/T-31', 'title': 'finished dep', 'done': True}], (
        f"expected deferred row deps with done=True, got: {row.get('deps')}"
    )

    # (c) active-path fields present / absent
    assert 'started' in row, "deferred row must have 'started' key (active path)"
    assert 'completed' not in row, (
        "deferred row must NOT have 'completed' key (that is the bounded-bucket sentinel)"
    )
