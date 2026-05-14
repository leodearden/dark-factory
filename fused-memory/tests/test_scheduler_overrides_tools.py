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
