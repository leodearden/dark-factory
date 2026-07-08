"""MCP-layer tests for claimant_run_id/heartbeat_at plumbing (task 2188).

PRD plans/task-status-authority-prd.md contract C4/D4 — omega1 exposes the
already-built backend/interceptor claimant write over MCP so the orchestrator
(which only reaches tasks.db via dispatch_tool) can stamp/refresh/clear it.

Covers:
  - step-5/6: set_task_status tool forwards claimant_run_id/heartbeat_at to
    interceptor.set_task_status only when supplied (legacy calls unaffected).
  - step-7/8: new set_task_claimant tool routes to
    interceptor.set_task_claimant, preserving tri-state over the wire (a
    wire-sentinel default is untouched; explicit JSON null clears).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    These tests use synthetic project_root values like ``/project`` that
    aren't real git working trees; the real resolver would reject them.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


@pytest.fixture
def task_interceptor():
    ti = AsyncMock()
    ti.set_task_status = AsyncMock(return_value={'success': True})
    ti.set_task_claimant = AsyncMock(return_value={'id': '1', 'message': 'ok'})
    return ti


@pytest.fixture
def mcp_server_with_tasks(task_interceptor):
    """MCP server with a mocked task interceptor."""
    mock_service = AsyncMock()
    return create_mcp_server(mock_service, task_interceptor=task_interceptor)


# ---------------------------------------------------------------------------
# step-5/6: set_task_status forwards claimant kwargs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_task_status_forwards_supplied_claimant_kwargs(
    mcp_server_with_tasks, task_interceptor,
):
    """Explicit claimant_run_id/heartbeat_at reach interceptor.set_task_status."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_status',
        {
            'id': '1',
            'status': 'in-progress',
            'project_root': '/project',
            'claimant_run_id': 'run-x/session-y/pid=123',
            'heartbeat_at': '2026-07-08T00:00:00+00:00',
        },
    )
    task_interceptor.set_task_status.assert_called_once()
    kwargs = task_interceptor.set_task_status.call_args.kwargs
    assert kwargs.get('claimant_run_id') == 'run-x/session-y/pid=123'
    assert kwargs.get('heartbeat_at') == '2026-07-08T00:00:00+00:00'


@pytest.mark.asyncio
async def test_set_task_status_without_claimant_kwargs_is_byte_identical(
    mcp_server_with_tasks, task_interceptor,
):
    """No claimant args supplied -> interceptor call carries neither kwarg.

    Legacy callers (every caller today) must be unaffected by this change.
    """
    await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_status',
        {'id': '1', 'status': 'in-progress', 'project_root': '/project'},
    )
    task_interceptor.set_task_status.assert_called_once()
    kwargs = task_interceptor.set_task_status.call_args.kwargs
    assert 'claimant_run_id' not in kwargs
    assert 'heartbeat_at' not in kwargs


# ---------------------------------------------------------------------------
# step-7/8: new set_task_claimant tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_task_claimant_tool_forwards_string_kwargs(
    mcp_server_with_tasks, task_interceptor,
):
    """A string claimant_run_id/heartbeat_at reaches interceptor.set_task_claimant."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_claimant',
        {
            'id': '1',
            'project_root': '/project',
            'claimant_run_id': 'run-x/session-y/pid=123',
            'heartbeat_at': '2026-07-08T00:00:00+00:00',
        },
    )
    task_interceptor.set_task_claimant.assert_called_once()
    kwargs = task_interceptor.set_task_claimant.call_args.kwargs
    assert kwargs.get('claimant_run_id') == 'run-x/session-y/pid=123'
    assert kwargs.get('heartbeat_at') == '2026-07-08T00:00:00+00:00'


@pytest.mark.asyncio
async def test_set_task_claimant_tool_heartbeat_only_refresh_leaves_run_id_untouched(
    mcp_server_with_tasks, task_interceptor,
):
    """Heartbeat-only refresh: claimant_run_id omitted -> untouched (not forwarded)."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_claimant',
        {
            'id': '1',
            'project_root': '/project',
            'heartbeat_at': '2026-07-08T00:00:00+00:00',
        },
    )
    task_interceptor.set_task_claimant.assert_called_once()
    kwargs = task_interceptor.set_task_claimant.call_args.kwargs
    assert 'claimant_run_id' not in kwargs
    assert kwargs.get('heartbeat_at') == '2026-07-08T00:00:00+00:00'


@pytest.mark.asyncio
async def test_set_task_claimant_tool_release_clear_forwards_explicit_none(
    mcp_server_with_tasks, task_interceptor,
):
    """Release clear: explicit JSON null for both fields forwards as None."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_claimant',
        {
            'id': '1',
            'project_root': '/project',
            'claimant_run_id': None,
            'heartbeat_at': None,
        },
    )
    task_interceptor.set_task_claimant.assert_called_once()
    kwargs = task_interceptor.set_task_claimant.call_args.kwargs
    assert 'claimant_run_id' in kwargs
    assert kwargs['claimant_run_id'] is None
    assert 'heartbeat_at' in kwargs
    assert kwargs['heartbeat_at'] is None


@pytest.mark.asyncio
async def test_set_task_claimant_tool_no_kwargs_supplied_omits_both(
    mcp_server_with_tasks, task_interceptor,
):
    """Neither field supplied -> neither key reaches the interceptor call."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_claimant',
        {'id': '1', 'project_root': '/project'},
    )
    task_interceptor.set_task_claimant.assert_called_once()
    kwargs = task_interceptor.set_task_claimant.call_args.kwargs
    assert 'claimant_run_id' not in kwargs
    assert 'heartbeat_at' not in kwargs
