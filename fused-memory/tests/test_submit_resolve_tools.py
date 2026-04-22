"""Tests for the submit_task and resolve_ticket MCP tool registrations."""

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    These tests use synthetic project_root values like ``/project`` that
    aren't real git working trees.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


@pytest.fixture
def task_interceptor():
    ti = AsyncMock()
    ti.submit_task = AsyncMock(return_value={'ticket': 'tkt_ABCDEFGHIJKLMNOPQRSTUVWXYZ'})
    ti.resolve_ticket = AsyncMock(return_value={'status': 'created', 'task_id': '5'})
    return ti


@pytest.fixture
def mcp_server(task_interceptor):
    """MCP server with a mocked task interceptor."""
    mock_service = AsyncMock()
    return create_mcp_server(mock_service, task_interceptor=task_interceptor)


# ------------------------------------------------------------------
# submit_task MCP tool
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_task_mcp_tool_signature_and_forwarding(mcp_server, task_interceptor):
    """submit_task MCP tool forwards all documented args to interceptor.submit_task
    and the returned {ticket: ...} dict flows back unchanged.
    """
    result = await mcp_server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': '/project',
            'title': 'My Task',
            'description': 'A description',
            'details': 'Some details',
            'dependencies': '1,2',
            'priority': 'high',
            'metadata': {'escalation_id': 'e1'},
            'tag': 'mytag',
        },
    )

    task_interceptor.submit_task.assert_called_once()
    call_kwargs = task_interceptor.submit_task.call_args.kwargs
    assert call_kwargs['project_root'] == '/project'
    assert call_kwargs.get('title') == 'My Task'
    assert call_kwargs.get('description') == 'A description'
    assert call_kwargs.get('details') == 'Some details'
    assert call_kwargs.get('dependencies') == '1,2'
    assert call_kwargs.get('priority') == 'high'
    assert call_kwargs.get('tag') == 'mytag'

    # Return value flows back unchanged.
    assert result == {'ticket': 'tkt_ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
