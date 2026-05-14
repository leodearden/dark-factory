"""Tests for get_scheduler_state and get_scheduler_events MCP tools.

All tools read from orchestrator-owned files under
``<project_root>/data/orchestrator/``.  Tests use a ``tmp_path``-rooted
project_root with the ``passthrough_main_checkout`` autouse fixture so the
path isn't rejected by the git-worktree validator.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server


# ---------------------------------------------------------------------------
# Fixtures (mirror test_scheduler_overrides_tools.py pattern)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    These tests use synthetic project_root values rooted in tmp_path that
    aren't real git working trees; the real resolver would reject them.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


@pytest.fixture
def memory_service():
    """Mocked MemoryService with all methods as AsyncMocks."""
    svc = AsyncMock()
    svc.add_memory = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def task_interceptor():
    """Mocked task interceptor."""
    ti = AsyncMock()
    ti.set_task_status = AsyncMock(return_value={'success': True})
    return ti


@pytest.fixture
def mcp_server(memory_service, task_interceptor):
    """MCP server with mocked MemoryService and task interceptor."""
    return create_mcp_server(memory_service, task_interceptor=task_interceptor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY_SKELETON = {
    'skip_counts': {},
    'parks': {},
    'effective_priorities': {},
    'pin_queue': [],
    'overrides': {},
    'current_holders': {},
    'snapshot_at': None,
}


def _write_snapshot(project_root: Path, data: dict) -> Path:
    """Write a synthetic scheduler_state.json under project_root."""
    path = project_root / 'data' / 'orchestrator' / 'scheduler_state.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def _runs_db_path(project_root: Path) -> Path:
    return project_root / 'data' / 'orchestrator' / 'runs.db'


# ===========================================================================
# Step-19: get_scheduler_state tool
# ===========================================================================


class TestGetSchedulerStateTool:
    """get_scheduler_state reads the on-disk JSON snapshot."""

    @pytest.mark.asyncio
    async def test_returns_snapshot_from_disk(self, tmp_path, mcp_server):
        """Tool round-trips a synthetic snapshot dict written to disk."""
        synthetic = {
            'skip_counts': {'T1': 3},
            'parks': {'T2': {'modules': ['m/src'], 'installed_at': '2026-01-01T00:00:00+00:00'}},
            'effective_priorities': {'T1': 'high'},
            'pin_queue': [{'task_id': 'T3', 'order': 1}],
            'overrides': {'T1': {'boost_tier': 'high', 'pinned': False,
                                 'reserve_now': False, 'ttl_until': None}},
            'current_holders': {'m/src': 'T4'},
            'snapshot_at': '2026-05-14T12:00:00+00:00',
        }
        _write_snapshot(tmp_path, synthetic)

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_state',
            {'project_root': str(tmp_path)},
        )
        assert result == synthetic, (
            f'Expected round-tripped snapshot, got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_returns_empty_skeleton_when_file_missing(
        self, tmp_path, mcp_server
    ):
        """Tool returns the empty skeleton when scheduler_state.json is missing."""
        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_state',
            {'project_root': str(tmp_path)},
        )
        assert result == _EMPTY_SKELETON, (
            f'Expected empty skeleton, got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_invalid_project_root_returns_validation_error(
        self, tmp_path, mcp_server
    ):
        """Empty string project_root triggers the ValidationError envelope."""
        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_state',
            {'project_root': ''},
        )
        assert isinstance(result, dict), f'Expected dict, got {type(result)}'
        assert result.get('error_type') == 'ValidationError', (
            f'Expected ValidationError, got {result!r}'
        )
        assert 'error' in result


# ===========================================================================
# Step-21: get_scheduler_events tool
# ===========================================================================

_EVENTS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL,
    duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""


def _seed_runs_db(project_root: Path, rows: list[dict]) -> Path:
    """Create runs.db and insert rows with controlled timestamps.

    Each row dict supports keys: timestamp, run_id, task_id, event_type, data.
    """
    db_path = project_root / 'data' / 'orchestrator' / 'runs.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_EVENTS_SCHEMA)
        for row in rows:
            conn.execute(
                'INSERT INTO events '
                '(timestamp, run_id, task_id, event_type, data) '
                'VALUES (?, ?, ?, ?, ?)',
                (
                    row['timestamp'],
                    row.get('run_id', 'run-1'),
                    row.get('task_id'),
                    row['event_type'],
                    json.dumps(row.get('data', {})),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


class TestGetSchedulerEventsTool:
    """get_scheduler_events reads events from runs.db."""

    @pytest.mark.asyncio
    async def test_returns_events_from_runs_db(self, tmp_path, mcp_server):
        """Tool returns three events in newest-first order."""
        rows = [
            {'timestamp': '2026-05-14T10:00:00+00:00', 'event_type': 'task_skipped',
             'task_id': 'T1', 'data': {'reason': 'held'}},
            {'timestamp': '2026-05-14T11:00:00+00:00', 'event_type': 'lock_acquired',
             'task_id': 'T2', 'data': {'module': 'm/src'}},
            {'timestamp': '2026-05-14T12:00:00+00:00', 'event_type': 'lock_released',
             'task_id': 'T3', 'data': {}},
        ]
        _seed_runs_db(tmp_path, rows)

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': str(tmp_path)},
        )
        assert isinstance(result, dict), f'Expected dict, got {type(result)!r}'
        events = result['events']
        assert result['count'] == 3, f'Expected count=3, got {result["count"]}'
        # newest-first order
        assert events[0]['event_type'] == 'lock_released'
        assert events[1]['event_type'] == 'lock_acquired'
        assert events[2]['event_type'] == 'task_skipped'
        # data is a parsed dict, not a string
        assert isinstance(events[2]['data'], dict)
        assert events[2]['data'] == {'reason': 'held'}

    @pytest.mark.asyncio
    async def test_event_types_filter(self, tmp_path, mcp_server):
        """Tool returns only rows matching the requested event_types."""
        rows = [
            {'timestamp': '2026-05-14T10:00:00+00:00', 'event_type': 'task_skipped',
             'task_id': 'T1'},
            {'timestamp': '2026-05-14T11:00:00+00:00', 'event_type': 'lock_acquired',
             'task_id': 'T2'},
            {'timestamp': '2026-05-14T12:00:00+00:00', 'event_type': 'task_skipped',
             'task_id': 'T3'},
        ]
        _seed_runs_db(tmp_path, rows)

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': str(tmp_path), 'event_types': ['task_skipped']},
        )
        assert result['count'] == 2, f'Expected count=2, got {result["count"]}'
        assert all(e['event_type'] == 'task_skipped' for e in result['events'])

    @pytest.mark.asyncio
    async def test_since_filter(self, tmp_path, mcp_server):
        """Tool returns only events at or after the `since` timestamp."""
        rows = [
            {'timestamp': '2026-05-14T09:00:00+00:00', 'event_type': 'task_skipped',
             'task_id': 'T1'},
            {'timestamp': '2026-05-14T11:00:00+00:00', 'event_type': 'lock_acquired',
             'task_id': 'T2'},
            {'timestamp': '2026-05-14T12:00:00+00:00', 'event_type': 'lock_released',
             'task_id': 'T3'},
        ]
        _seed_runs_db(tmp_path, rows)

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': str(tmp_path), 'since': '2026-05-14T10:00:00+00:00'},
        )
        assert result['count'] == 2, f'Expected count=2, got {result["count"]}'
        # Only T2 and T3 are >= 10:00
        returned_tasks = {e['task_id'] for e in result['events']}
        assert returned_tasks == {'T2', 'T3'}

    @pytest.mark.asyncio
    async def test_limit_caps_results(self, tmp_path, mcp_server):
        """Tool caps results to the `limit` parameter."""
        rows = [
            {'timestamp': f'2026-05-14T0{i}:00:00+00:00', 'event_type': 'task_skipped',
             'task_id': f'T{i}'}
            for i in range(5)
        ]
        _seed_runs_db(tmp_path, rows)

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': str(tmp_path), 'limit': 2},
        )
        assert result['count'] == 2, f'Expected count=2, got {result["count"]}'
        assert len(result['events']) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_when_db_missing(self, tmp_path, mcp_server):
        """Tool returns empty result when runs.db does not exist."""
        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': str(tmp_path)},
        )
        assert result == {'events': [], 'count': 0}, (
            f'Expected empty result, got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_invalid_project_root_returns_validation_error(
        self, tmp_path, mcp_server
    ):
        """Empty string project_root triggers the ValidationError envelope."""
        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': ''},
        )
        assert isinstance(result, dict), f'Expected dict, got {type(result)}'
        assert result.get('error_type') == 'ValidationError', (
            f'Expected ValidationError, got {result!r}'
        )
