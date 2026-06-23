"""Tests for the request_park_eviction MCP tool.

Covers the thin producer-side enqueue tool that writes to
``<project_root>/data/orchestrator/park_eviction_requests.db``.

The scheduler-tick drain (task δ/1871) owns the authoritative D4 live-owner
guard; this tool performs NO eviction and NO guard — it is a pure enqueue.

All tests use a ``tmp_path``-rooted project_root with the
``passthrough_main_checkout`` autouse fixture so the path isn't rejected by
the git-worktree validator.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

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
    """MCP server with a mocked MemoryService (no task interceptor needed)."""
    return create_mcp_server(memory_service)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_path(project_root: str | Path) -> Path:
    """Return the canonical park_eviction_requests.db path for a project_root.

    Mirrors _park_eviction_db_path in tools.py — same resolved path the
    scheduler drain opens when looking for rows to consume.
    """
    return Path(project_root).resolve() / 'data' / 'orchestrator' / 'park_eviction_requests.db'


def _open_db(project_root: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the park eviction DB with sqlite3 (for assertion reads)."""
    p = _db_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(p))


def _row_count(project_root: str | Path) -> int:
    """Return row count from park_eviction_requests, or 0 if the table/DB doesn't exist."""
    p = _db_path(project_root)
    if not p.exists():
        return 0
    conn = sqlite3.connect(str(p))
    try:
        try:
            row = conn.execute('SELECT COUNT(*) FROM park_eviction_requests').fetchone()
            return row[0]
        except sqlite3.OperationalError:
            return 0
    finally:
        conn.close()


# ===========================================================================
# Happy path — well-formed row
# ===========================================================================


@pytest.mark.asyncio
async def test_request_park_eviction_happy_path(tmp_path, mcp_server):
    """Happy path: a row is written and the return dict matches the contract.

    Asserts:
    - Return value is exactly {'requested': True, 'task_id': '42'}.
    - Exactly one row exists in park_eviction_requests.db.
    - task_id column equals '42'.
    - project_root column equals str(Path(tmp_path).resolve()) — this is the
      CANONICAL form that δ's drain matches on via
      ``WHERE project_root = str(Path(config.project_root).resolve())``.
      Storing any other form (e.g. un-resolved, with trailing slash, or the
      worktree path) would cause the scheduler to silently never drain the row.
    - requested_at is parseable by datetime.fromisoformat into a tz-aware datetime.
    """
    result = await mcp_server._tool_manager.call_tool(
        'request_park_eviction',
        {'task_id': '42', 'project_root': str(tmp_path)},
    )
    assert result == {'requested': True, 'task_id': '42'}

    conn = _open_db(tmp_path)
    try:
        rows = conn.execute(
            'SELECT task_id, project_root, requested_at FROM park_eviction_requests',
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1, f'expected exactly 1 row, got {len(rows)}'
    task_id_col, project_root_col, requested_at_col = rows[0]

    assert task_id_col == '42'

    # Critical interop contract with δ drain: stored project_root MUST equal
    # str(Path(tmp_path).resolve()) — exact string equality with drain's WHERE clause.
    assert project_root_col == str(Path(tmp_path).resolve()), (
        f'project_root column {project_root_col!r} != canonical resolved form '
        f'{str(Path(tmp_path).resolve())!r}; '
        "δ drain matches WHERE project_root=str(Path(config.project_root).resolve()) "
        '— a non-canonical stored value would cause silent drain failure'
    )

    # requested_at must parse as a tz-aware datetime (UTC).
    parsed = datetime.fromisoformat(requested_at_col)
    assert parsed.tzinfo is not None, (
        f'requested_at {requested_at_col!r} is not tz-aware; '
        'δ drain relies on ISO8601 UTC timestamps for ordering'
    )
