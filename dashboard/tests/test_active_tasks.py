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
# collect_active_tasks against synthetic on-disk fixture
# ---------------------------------------------------------------------------


def _make_project(root, *, project_dir, tasks, worktrees=None):
    """Lay out a synthetic project with .taskmaster/tasks.json + optional worktrees.

    ``worktrees`` is a list of (task_id, metadata_dict, plan_modules,
    iteration_lines, review_files) tuples.  Any element after metadata may be
    None / [] to skip writing that artefact.
    """
    project_root = root / project_dir
    project_root.mkdir(parents=True, exist_ok=True)
    tasks_dir = project_root / '.taskmaster' / 'tasks'
    tasks_dir.mkdir(parents=True)
    (tasks_dir / 'tasks.json').write_text(json.dumps({'tasks': tasks}))

    if not worktrees:
        return project_root

    worktrees_dir = project_root / '.worktrees'
    worktrees_dir.mkdir()
    for task_id, metadata, modules, iteration_lines, review_files in worktrees:
        wt = worktrees_dir / str(task_id)
        wt.mkdir()
        task_dir = wt / '.task'
        task_dir.mkdir()
        if metadata is not None:
            (task_dir / 'metadata.json').write_text(json.dumps(metadata))
        if modules is not None:
            (task_dir / 'plan.json').write_text(json.dumps({'steps': [], 'modules': modules}))
        if iteration_lines is not None:
            (task_dir / 'iterations.jsonl').write_text(
                '\n'.join('{}' for _ in range(iteration_lines)) + ('\n' if iteration_lines else ''),
            )
        if review_files is not None:
            reviews = task_dir / 'reviews'
            reviews.mkdir()
            for i, verdict in enumerate(review_files):
                (reviews / f'r{i}.json').write_text(json.dumps({'verdict': verdict}))
    return project_root


@pytest.fixture()
def two_project_config(tmp_path):
    """Lay down dark-factory + reify projects with a mix of tasks/statuses."""
    started = (datetime.now(UTC) - timedelta(minutes=14)).isoformat()
    df_root = _make_project(
        tmp_path,
        project_dir='dark-factory',
        tasks=[
            {'id': 19, 'title': 'consolidation retry', 'status': 'in-progress',
             'dependencies': [15, 17]},
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
    reify_root = _make_project(
        tmp_path,
        project_dir='reify',
        tasks=[{'id': 8, 'title': 'parser recovery', 'status': 'blocked',
                'dependencies': []}],
        worktrees=[(8, {'task_id': '8', 'title': 'parser recovery',
                        'created_at': started},
                    ['parser/recovery.rs'], 0, [])],
    )
    return DashboardConfig(project_root=df_root, known_project_roots=[reify_root])


def test_collect_active_tasks_filters_to_active_statuses(two_project_config):
    active, _ = collect_active_tasks(two_project_config)
    statuses = {t['status'] for t in active}
    assert statuses <= {'in-progress', 'blocked', 'pending'}
    # Done tasks (17, 15) should not appear.
    ids = {t['id'] for t in active}
    assert 'dark-factory/T-17' not in ids
    assert 'dark-factory/T-15' not in ids


def test_collect_active_tasks_resolves_deps_with_done_flags(two_project_config):
    active, _ = collect_active_tasks(two_project_config)
    by_id = {t['id']: t for t in active}
    t19 = by_id['dark-factory/T-19']
    assert {d['id']: d['done'] for d in t19['deps']} == {
        'dark-factory/T-15': True,
        'dark-factory/T-17': True,
    }
    t23 = by_id['dark-factory/T-23']
    # T-21 is in-progress, not done
    assert t23['deps'] == [{'id': 'dark-factory/T-21', 'title': 'dedup index', 'done': False}]


def test_collect_active_tasks_pulls_metadata_and_locks(two_project_config):
    active, _ = collect_active_tasks(two_project_config)
    t19 = next(t for t in active if t['id'] == 'dark-factory/T-19')
    assert t19['agent'] == 'claude-task-19'
    assert t19['loops'] == 2
    assert t19['attempts'] == 3
    assert 'src/agents/consolidation.py' in t19['locks']
    # `started` is the minutes-since difference, allow a small slack vs 14.
    assert 13 <= t19['started'] <= 15


def test_collect_active_tasks_handles_missing_worktree_metadata(tmp_path):
    """A pending task with no worktree should still appear with empty fields."""
    root = _make_project(
        tmp_path, project_dir='solo',
        tasks=[{'id': 1, 'title': 'lonely', 'status': 'pending', 'dependencies': []}],
    )
    cfg = DashboardConfig(project_root=root)
    active, _ = collect_active_tasks(cfg)
    assert active == [{
        'id': 'solo/T-1', 'project': 'solo', 'title': 'lonely',
        'status': 'pending', 'agent': None, 'started': 0, 'loops': 0,
        'attempts': 0, 'deps': [], 'locks': [],
    }]


def test_collect_active_tasks_inverts_locks_into_file_locks(two_project_config):
    _, locks = collect_active_tasks(two_project_config)
    df = locks['dark-factory']
    assert df['src/agents/consolidation.py'] == {'holder': 'dark-factory/T-19'}
    assert df['src/store/dedup.py'] == {'holder': 'dark-factory/T-21'}
    # blocked tasks count as holders too
    reify = locks['reify']
    assert reify['parser/recovery.rs'] == {'holder': 'reify/T-8'}
