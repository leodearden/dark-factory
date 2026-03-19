"""Tests for the /health endpoint and MCP tool-level behavior."""

from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from fused_memory.server.tools import create_mcp_server


@pytest.fixture
def mcp_server():
    """Create an MCP server with a mocked MemoryService."""
    mock_service = AsyncMock()
    return create_mcp_server(mock_service)


@pytest.fixture
def task_interceptor():
    ti = AsyncMock()
    ti.update_task = AsyncMock(return_value={'success': True})
    return ti


@pytest.fixture
def mcp_server_with_tasks(task_interceptor):
    """MCP server with a mocked task interceptor."""
    mock_service = AsyncMock()
    return create_mcp_server(mock_service, task_interceptor=task_interceptor)


@pytest.fixture
def health_app(mcp_server):
    """Get the Starlette app that includes the /health route."""
    return mcp_server.streamable_http_app()


def test_health_returns_200(health_app):
    """GET /health should return 200 with {"status": "ok"}."""
    client = TestClient(health_app)
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.json() == {'status': 'ok'}


def test_health_post_not_allowed(health_app):
    """POST /health should be rejected (405 Method Not Allowed)."""
    client = TestClient(health_app)
    resp = client.post('/health')
    assert resp.status_code == 405


# ------------------------------------------------------------------
# update_task metadata coercion
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_metadata_dict_coerced_to_json_string(
    mcp_server_with_tasks, task_interceptor,
):
    """When metadata is passed as a dict (as MCP callers naturally do),
    the tool should JSON-serialize it before forwarding to the interceptor."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project', 'metadata': {'key': 'value'}},
    )
    task_interceptor.update_task.assert_called_once()
    _, kwargs = task_interceptor.update_task.call_args
    assert kwargs['metadata'] == '{"key": "value"}'


@pytest.mark.asyncio
async def test_update_task_metadata_string_passed_through(
    mcp_server_with_tasks, task_interceptor,
):
    """When metadata is already a JSON string, it should pass through unchanged."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project', 'metadata': '{"key": "value"}'},
    )
    task_interceptor.update_task.assert_called_once()
    _, kwargs = task_interceptor.update_task.call_args
    assert kwargs['metadata'] == '{"key": "value"}'


@pytest.mark.asyncio
async def test_update_task_metadata_none_passed_through(
    mcp_server_with_tasks, task_interceptor,
):
    """When metadata is None/omitted, it should pass through as None."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project'},
    )
    task_interceptor.update_task.assert_called_once()
    _, kwargs = task_interceptor.update_task.call_args
    assert kwargs['metadata'] is None
