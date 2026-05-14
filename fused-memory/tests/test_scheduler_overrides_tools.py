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
    """Open (creating if needed) the override DB with sqlite3 (for assertion reads)."""
    p = _db_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(p))


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


# ===========================================================================
# set_task_priority_override — validation errors
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_tier', ['urgent', 'urgentish', '', 'critical-ish'])
async def test_set_task_priority_override_rejects_unknown_boost_tier(
    tmp_path, mcp_server, memory_service, bad_tier,
):
    """Unknown boost_tier returns ValidationError; no row is written, no audit emitted."""
    memory_service.add_memory.reset_mock()
    result = await mcp_server._tool_manager.call_tool(
        'set_task_priority_override',
        {'project_root': str(tmp_path), 'task_id': '5', 'boost_tier': bad_tier},
    )
    assert result.get('error_type') == 'ValidationError'
    assert bad_tier in result.get('error', '')

    # DB may not exist at all (validation fires before any DB open).
    assert _row_count(tmp_path) == 0

    memory_service.add_memory.assert_not_called()


@pytest.mark.asyncio
async def test_set_task_priority_override_pin_order_collision_returns_structured_error(
    tmp_path, mcp_server, memory_service,
):
    """A second pin at the same pin_order returns a structured collision error."""
    memory_service.add_memory.reset_mock()

    r1 = await mcp_server._tool_manager.call_tool(
        'set_task_priority_override',
        {'project_root': str(tmp_path), 'task_id': 'A', 'pinned': True, 'pin_order': 1},
    )
    assert 'error' not in r1

    memory_service.add_memory.reset_mock()

    r2 = await mcp_server._tool_manager.call_tool(
        'set_task_priority_override',
        {'project_root': str(tmp_path), 'task_id': 'B', 'pinned': True, 'pin_order': 1},
    )
    assert r2 == {'error': 'pin_order_collision', 'conflicting_task_id': 'A', 'pin_order': 1}

    # task B must have no row
    conn = _open_db(tmp_path)
    try:
        row = conn.execute(
            'SELECT task_id FROM overrides WHERE project_root=? AND task_id=?',
            (str(tmp_path), 'B'),
        ).fetchone()
        assert row is None
    finally:
        conn.close()

    memory_service.add_memory.assert_not_called()


@pytest.mark.asyncio
async def test_set_task_priority_override_ttl_secs_converts_to_absolute_iso(
    tmp_path, mcp_server, memory_service,
):
    """ttl_secs is converted to an absolute UTC ISO8601 ttl_until in the DB."""
    from datetime import timezone
    before = datetime.now(UTC)
    await mcp_server._tool_manager.call_tool(
        'set_task_priority_override',
        {'project_root': str(tmp_path), 'task_id': 'X', 'ttl_secs': 3600},
    )
    after = datetime.now(UTC)

    conn = _open_db(tmp_path)
    try:
        row = conn.execute(
            'SELECT ttl_until FROM overrides WHERE project_root=? AND task_id=?',
            (str(tmp_path), 'X'),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None and row[0] is not None
    parsed = datetime.fromisoformat(row[0])
    assert parsed.tzinfo is not None
    low = before + timedelta(seconds=3600) - timedelta(seconds=5)
    high = after + timedelta(seconds=3600) + timedelta(seconds=5)
    assert low <= parsed <= high

    _, audit_kwargs = memory_service.add_memory.call_args
    assert audit_kwargs['metadata']['fields']['ttl_secs'] == 3600


@pytest.mark.asyncio
@pytest.mark.parametrize('kwargs,expected_pin_orders', [
    # (a) empty DB → first pin gets pin_order=1
    ([{'task_id': 'A', 'pinned': True}], {'A': 1}),
    # (b) second pin auto-assigns pin_order=2
    (
        [{'task_id': 'A', 'pinned': True}, {'task_id': 'B', 'pinned': True}],
        {'A': 1, 'B': 2},
    ),
])
async def test_set_task_priority_override_pinned_true_auto_assigns_pin_order(
    tmp_path, mcp_server, memory_service, kwargs, expected_pin_orders,
):
    """Pinning without explicit pin_order auto-assigns next available position."""
    for kw in kwargs:
        await mcp_server._tool_manager.call_tool(
            'set_task_priority_override',
            {'project_root': str(tmp_path), **kw},
        )

    conn = _open_db(tmp_path)
    try:
        for task_id, expected_order in expected_pin_orders.items():
            row = conn.execute(
                'SELECT pin_order FROM overrides WHERE project_root=? AND task_id=?',
                (str(tmp_path), task_id),
            ).fetchone()
            assert row is not None
            assert row[0] == expected_order, f'{task_id}: expected {expected_order}, got {row[0]}'
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_set_task_priority_override_re_pin_preserves_existing_pin_order(
    tmp_path, mcp_server, memory_service,
):
    """Re-pinning an already-pinned task preserves its existing pin_order (idempotency)."""
    await mcp_server._tool_manager.call_tool(
        'set_task_priority_override',
        {'project_root': str(tmp_path), 'task_id': 'A', 'pinned': True},
    )
    await mcp_server._tool_manager.call_tool(
        'set_task_priority_override',
        {'project_root': str(tmp_path), 'task_id': 'B', 'pinned': True},
    )
    # Re-pin A with a boost_tier change — pin_order must stay 1
    await mcp_server._tool_manager.call_tool(
        'set_task_priority_override',
        {'project_root': str(tmp_path), 'task_id': 'A', 'pinned': True, 'boost_tier': 'high'},
    )

    conn = _open_db(tmp_path)
    try:
        row = conn.execute(
            'SELECT pin_order FROM overrides WHERE project_root=? AND task_id=?',
            (str(tmp_path), 'A'),
        ).fetchone()
        assert row[0] == 1
    finally:
        conn.close()
