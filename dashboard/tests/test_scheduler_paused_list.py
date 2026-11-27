"""The Scheduler tab's paused list reads each root's own files through the shared reader.

The pause fact's home is the runs.db ``scheduler_state`` row and
scheduler_state.json is its cache (plans/instruments-honesty-prd.md decision 5,
boundary row 4). These tests seed both files under real tmp roots and drive
``collect_scheduler_state`` through its public seam; the fused-memory side is
mocked to say "not paused", so only the local read can list a project.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from _task_units import units_by_label
from shared.project_layout import runs_db, scheduler_state_json
from shared.scheduler_state import SchedulerStateSnapshot

from dashboard.config import DashboardConfig
from dashboard.data.scheduler import clear_scheduler_state, collect_scheduler_state

_NOW = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
_SNAPSHOT_AT = _NOW - timedelta(minutes=90)
_ROW_REASON = 'cost_ceiling_orch_exceeded'

# A minimal transcription of the scheduler_state table in
# orchestrator/src/orchestrator/run_store.py (the dashboard cannot import orchestrator).
_SCHEDULER_STATE_DDL = (
    'CREATE TABLE scheduler_state ('
    ' project_id TEXT PRIMARY KEY,'
    ' pause_reason TEXT NOT NULL,'
    ' pause_at TEXT NOT NULL,'
    ' set_by_run_id TEXT NOT NULL,'
    ' pause_class TEXT)'
)


@pytest.fixture(autouse=True)
def _forget_scheduler_state():
    clear_scheduler_state()
    yield
    clear_scheduler_state()


def _write_snapshot(root: Path, body: dict) -> None:
    path = scheduler_state_json(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def _seed_pause_row(root: Path, reason: str = _ROW_REASON, pause_class: str = 'cost_ceiling') -> None:
    path = runs_db(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(_SCHEDULER_STATE_DDL)
        conn.execute(
            'INSERT INTO scheduler_state '
            '(project_id, pause_reason, pause_at, set_by_run_id, pause_class) '
            'VALUES (?, ?, ?, ?, ?)',
            ('dark_factory', reason, (_NOW - timedelta(hours=2)).isoformat(), 'run-1', pause_class),
        )
        conn.commit()
    finally:
        conn.close()


def _roots(tmp_path: Path, *names: str) -> list[Path]:
    roots = [tmp_path / name for name in names]
    for root in roots:
        root.mkdir()
    return roots


def _config(roots: list[Path]) -> DashboardConfig:
    return DashboardConfig(project_root=roots[0], known_project_roots=roots[1:])


async def _paused_projects(config: DashboardConfig, *, offline: tuple[str, ...] = ()) -> list[dict]:
    async def not_paused_over_mcp(client, url, tool, args):
        if tool == 'get_scheduler_state':
            return SchedulerStateSnapshot(snapshot_at=_NOW).to_json()
        return []

    async with httpx.AsyncClient() as client:
        with (
            patch('dashboard.data.scheduler.mcp_tool_call', side_effect=not_paused_over_mcp),
            patch(
                'dashboard.data.scheduler.collect_tasks_with_counts',
                AsyncMock(return_value=units_by_label([], offline=offline)),
            ),
        ):
            result = await collect_scheduler_state(client, config, now=_NOW)
    return result[5]


async def test_a_pause_row_under_a_running_snapshot_is_listed(tmp_path: Path) -> None:
    (root,) = _roots(tmp_path, 'p1')
    _write_snapshot(root, SchedulerStateSnapshot(is_paused=False, snapshot_at=_SNAPSHOT_AT).to_json())
    _seed_pause_row(root)
    config = _config([root])

    assert await _paused_projects(config) == [{
        'project': config.project_labels.of(root),
        'reason': _ROW_REASON,
        'pause_class': 'cost_ceiling',
        'snapshot_at': _SNAPSHOT_AT.isoformat(),
        'snapshot_age_s': (_NOW - _SNAPSHOT_AT).total_seconds(),
        'pause_disagreement': True,
    }]


async def test_a_paused_snapshot_without_runs_db_is_listed_in_agreement(tmp_path: Path) -> None:
    (root,) = _roots(tmp_path, 'p1')
    _write_snapshot(root, SchedulerStateSnapshot(
        is_paused=True, pause_reason='park-stop: 5 tasks parked in 1h',
        pause_class='park_stop', snapshot_at=_SNAPSHOT_AT,
    ).to_json())
    config = _config([root])

    assert await _paused_projects(config) == [{
        'project': config.project_labels.of(root),
        'reason': 'park-stop: 5 tasks parked in 1h',
        'pause_class': 'park_stop',
        'snapshot_at': _SNAPSHOT_AT.isoformat(),
        'snapshot_age_s': (_NOW - _SNAPSHOT_AT).total_seconds(),
        'pause_disagreement': False,
    }]


@pytest.mark.parametrize(
    'body',
    [
        SchedulerStateSnapshot(is_paused=False, snapshot_at=_SNAPSHOT_AT).to_json(),
        {'current_holders': {}, 'snapshot_at': _SNAPSHOT_AT.isoformat()},
    ],
    ids=['running', 'legacy-without-is_paused'],
)
async def test_a_project_neither_file_calls_paused_is_not_listed(tmp_path: Path, body: dict) -> None:
    (root,) = _roots(tmp_path, 'p1')
    _write_snapshot(root, body)

    assert await _paused_projects(_config([root])) == []


async def test_only_the_paused_root_of_two_is_listed(tmp_path: Path) -> None:
    paused, running = _roots(tmp_path, 'p1', 'p2')
    _write_snapshot(paused, SchedulerStateSnapshot(is_paused=False, snapshot_at=_SNAPSHOT_AT).to_json())
    _seed_pause_row(paused)
    _write_snapshot(running, SchedulerStateSnapshot(is_paused=False, snapshot_at=_SNAPSHOT_AT).to_json())
    config = _config([paused, running])

    listed = await _paused_projects(config)

    assert [entry['project'] for entry in listed] == [config.project_labels.of(paused)]


async def test_a_root_offline_to_fused_memory_is_still_listed_from_its_files(tmp_path: Path) -> None:
    (root,) = _roots(tmp_path, 'p1')
    _seed_pause_row(root)
    config = _config([root])
    label = config.project_labels.of(root)

    listed = await _paused_projects(config, offline=(label,))

    assert [(entry['project'], entry['reason']) for entry in listed] == [(label, _ROW_REASON)]
