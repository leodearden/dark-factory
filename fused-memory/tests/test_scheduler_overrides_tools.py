"""Tests for MCP scheduler-override tool behavior.

Covers four tools:
- get_pin_queue         (read-only, no audit)
- set_task_priority_override  (write, emits audit)
- clear_task_priority_override (write, emits audit)
- reorder_pin_queue     (write, emits audit)

All tools open ``<project_root>/data/orchestrator/scheduler_overrides.db``
via aiosqlite; these tests use a ``tmp_path``-rooted project_root with the
``passthrough_main_checkout`` autouse fixture so the path isn't rejected by
the git-worktree validator.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from fused_memory.models.scope import resolve_project_id
from fused_memory.server.tools import create_mcp_server


# ---------------------------------------------------------------------------
# Fixtures
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
def mcp_server(memory_service):
    """MCP server with a mocked MemoryService and a mocked task interceptor."""
    return create_mcp_server(memory_service, task_interceptor=AsyncMock())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_path(project_root: str | Path) -> Path:
    """Return the canonical override DB path for a project_root."""
    return Path(project_root) / 'data' / 'orchestrator' / 'scheduler_overrides.db'


def _open_db(project_root: str | Path) -> sqlite3.Connection:
    """Open the override DB with sqlite3 (for assertion reads)."""
    return sqlite3.connect(str(_db_path(project_root)))


def _row_count(project_root: str | Path) -> int:
    conn = _open_db(project_root)
    try:
        try:
            row = conn.execute('SELECT COUNT(*) FROM overrides').fetchone()
            return row[0]
        except sqlite3.OperationalError:
            return 0
    finally:
        conn.close()


# ===========================================================================
# get_pin_queue — read-only, no audit emit
# ===========================================================================


@pytest.mark.asyncio
async def test_get_pin_queue_empty_returns_empty_list(tmp_path, mcp_server, memory_service):
    """get_pin_queue against a fresh project_root returns {'pin_queue': []}.

    Regression lock: read-only tool must NOT emit an audit add_memory call.
    """
    result = await mcp_server._tool_manager.call_tool(
        'get_pin_queue',
        {'project_root': str(tmp_path)},
    )
    assert result == {'pin_queue': []}
    memory_service.add_memory.assert_not_called()


# ===========================================================================
# set_task_priority_override — write + audit
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize('extra_kwargs,extra_row_checks', [
    ({}, {'pinned': 0, 'reserve_now': 0}),
    ({'pinned': True}, {'pinned': 1}),
    ({'reserve_now': True}, {'reserve_now': 1}),
])
async def test_set_task_priority_override_writes_row_and_emits_audit(
    tmp_path, mcp_server, memory_service, extra_kwargs, extra_row_checks,
):
    """Happy-path: a row is written to SQLite and an audit add_memory is emitted."""
    memory_service.add_memory.reset_mock()
    result = await mcp_server._tool_manager.call_tool(
        'set_task_priority_override',
        {'project_root': str(tmp_path), 'task_id': '5', 'boost_tier': 'high', **extra_kwargs},
    )
    assert result.get('success') is True or 'error' not in result

    conn = _open_db(tmp_path)
    try:
        row = conn.execute(
            'SELECT project_root, task_id, boost_tier, pinned, reserve_now '
            'FROM overrides WHERE project_root=? AND task_id=?',
            (str(tmp_path), '5'),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == str(tmp_path)
    assert row[1] == '5'
    assert row[2] == 'high'
    for col, val in extra_row_checks.items():
        idx = {'pinned': 3, 'reserve_now': 4}[col]
        assert row[idx] == val, f'{col} mismatch: expected {val}, got {row[idx]}'

    memory_service.add_memory.assert_called_once()
    _, audit_kwargs = memory_service.add_memory.call_args
    assert audit_kwargs['category'] == 'decisions_and_rationale'
    assert audit_kwargs['project_id'] == resolve_project_id(str(tmp_path))
    assert audit_kwargs['agent_id'] == 'scheduler-overrides'
    assert audit_kwargs['metadata']['task_id'] == '5'
    assert audit_kwargs['metadata']['fields']['boost_tier'] == 'high'
