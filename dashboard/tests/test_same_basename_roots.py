"""Two configured roots sharing a basename stay distinct on every per-project surface."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot

import dashboard.data.active_tasks as active_tasks
import dashboard.data.task_snapshot as task_snapshot
from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import (
    collect_census_snapshots,
    collect_tasks_with_counts,
    shape_terminal_rows,
)
from dashboard.data.datum import Datum, DatumState
from dashboard.data.task_snapshot import FRESHNESS_BOUND_SECONDS

A_LABEL = 'a/proj'
B_LABEL = 'b/proj'


@pytest.fixture(autouse=True)
def _fresh_collector_state():
    active_tasks._reset_root_rotation()
    task_snapshot._snapshot_cache_clear()
    yield
    active_tasks._reset_root_rotation()
    task_snapshot._snapshot_cache_clear()


@pytest.fixture
def roots(tmp_path) -> tuple[Path, Path]:
    a = tmp_path / 'a' / 'proj'
    b = tmp_path / 'b' / 'proj'
    for root in (a, b):
        root.mkdir(parents=True)
    return a.resolve(), b.resolve()


@pytest.fixture
def config(roots) -> DashboardConfig:
    a, b = roots
    config = DashboardConfig(project_root=a, known_project_roots=[b])
    assert (config.project_labels.of(a), config.project_labels.of(b)) == (A_LABEL, B_LABEL)
    return config


def _substrate_with_one_task_per_root(titles: dict[Path, str]):
    """A fused-memory stand-in whose one in-progress task 1 is titled per root."""

    async def _mcp(client, url, tool, args, **_kw):
        title = titles[Path(args['project_root'])]
        if tool == 'get_statuses':
            return {'statuses': {'1': 'in-progress'}}
        if tool == 'get_tasks':
            return {'tasks': [{
                'id': '1', 'title': title, 'status': 'in-progress',
                'dependencies': [], 'metadata': {},
                'updatedAt': '2026-01-01T00:00:01+00:00',
            }]}
        raise AssertionError(f'unexpected tool {tool!r}')

    return _mcp


def _runtime_for_b_only(monkeypatch) -> None:
    snapshot = TaskRuntimeSnapshot(tasks=[TaskRuntimeEntry(
        task_id=1, has_worktree=True, loops=3, attempts=1, started=None,
        lane='lane-b', phase=None, lane_state=None,
    )])

    async def _fetch_task_runtime(client, escalation_urls):
        return {B_LABEL: snapshot}

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_task_runtime', _fetch_task_runtime)


class TestTasksCollector:
    @pytest.fixture(autouse=True)
    def _canned_reads(self, roots, monkeypatch):
        a, b = roots
        monkeypatch.setattr(
            'dashboard.data.tasks.mcp_tool_call',
            _substrate_with_one_task_per_root({a: 'from-a', b: 'from-b'}),
        )
        _runtime_for_b_only(monkeypatch)

    async def test_each_root_contributes_its_own_row_once(self, config, dummy_client):
        rows, _snapshots = await collect_tasks_with_counts(dummy_client, config)

        assert sorted((row['id'], row['project'], row['title']) for row in rows) == [
            ('a/proj/T-1', A_LABEL, 'from-a'),
            ('b/proj/T-1', B_LABEL, 'from-b'),
        ]

    async def test_each_root_keeps_its_own_snapshot(self, config, dummy_client):
        _rows, snapshots = await collect_tasks_with_counts(dummy_client, config)

        assert set(snapshots) == {A_LABEL, B_LABEL}

    async def test_a_roots_runtime_reaches_only_that_roots_rows(self, config, dummy_client):
        rows, _snapshots = await collect_tasks_with_counts(dummy_client, config)
        by_project = {row['project']: row for row in rows}

        assert by_project[B_LABEL]['lane'] == 'lane-b'
        assert by_project[B_LABEL]['runtime_status'] == 'ok'
        assert by_project[A_LABEL]['lane'] is None
        assert by_project[A_LABEL]['runtime_status'] == 'not_configured'

    async def test_the_census_keeps_each_root(self, config, dummy_client):
        snapshots = await collect_census_snapshots(dummy_client, config)

        assert set(snapshots) == {A_LABEL, B_LABEL}


def test_terminal_rows_are_identified_by_the_label_they_are_shaped_under():
    raw = {
        'id': 7, 'title': 'finished', 'status': 'done', 'dependencies': [],
        'metadata': {}, 'updated_at': '2026-01-01T00:00:07+00:00',
    }

    (row,) = shape_terminal_rows(B_LABEL, [raw], now=datetime(2026, 1, 2, tzinfo=UTC))

    assert (row['id'], row['project']) == ('b/proj/T-7', B_LABEL)


def test_the_terminal_window_reads_the_root_its_label_names(client, roots):
    a, b = roots
    client.app.state.config = dataclasses.replace(
        client.app.state.config, project_root=a, known_project_roots=[b],
    )
    window = AsyncMock(return_value=Datum(
        None, None, DatumState.UNKNOWN, 'canned', FRESHNESS_BOUND_SECONDS,
    ))
    with patch(
        'dashboard.api.tasks.collect_tasks_with_counts',
        new=AsyncMock(return_value=([], {})),
    ), patch('dashboard.api.tasks.acquire_terminal_window', new=window):
        resp = client.get('/api/v2/dashboard/tasks?terminal=b%2Fproj')

    assert resp.status_code == 200, resp.text
    assert f'TASKS_TERMINAL:{B_LABEL}' in resp.json()
    (acquisition,) = window.await_args_list
    assert acquisition.args[2] == b
