"""Tests for the ACTIVE_TASKS aggregator that joins task tree + worktrees."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import (
    _attempts_from_review_summary,
    _minutes_since,
    collect_active_tasks,
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
        'meta_files': [], 'train': None,
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
