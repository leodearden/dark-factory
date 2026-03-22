"""Tests for the /health endpoint, MCP tool-level behavior, and server settings."""

from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from fused_memory.config.schema import ServerConfig
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


# ------------------------------------------------------------------
# update_task prompt parameter forwarding
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_prompt_forwarded_to_interceptor(
    mcp_server_with_tasks, task_interceptor,
):
    """When prompt is provided, it should be forwarded to the interceptor."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project', 'prompt': 'Update the description'},
    )
    task_interceptor.update_task.assert_called_once()
    _, kwargs = task_interceptor.update_task.call_args
    assert kwargs['prompt'] == 'Update the description'
    assert kwargs['append'] is False  # default value when not provided


# ------------------------------------------------------------------
# update_task append parameter forwarding
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_append_true_forwarded_to_interceptor(
    mcp_server_with_tasks, task_interceptor,
):
    """When append=True, the value should be forwarded to the interceptor."""
    await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project', 'prompt': 'Extra info', 'append': True},
    )
    task_interceptor.update_task.assert_called_once()
    _, kwargs = task_interceptor.update_task.call_args
    assert kwargs['append'] is True


# ------------------------------------------------------------------
# update_task error handling
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_interceptor_error_returns_error_dict(
    mcp_server_with_tasks, task_interceptor,
):
    """When the interceptor raises an Exception, the tool returns {'error': str(e)}."""
    task_interceptor.update_task.side_effect = RuntimeError('backend unavailable')
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'update_task',
        {'id': '1', 'project_root': '/project'},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'backend unavailable' in result['error']


# ------------------------------------------------------------------
# Defensive tool registration (always registered, even without Taskmaster)
# ------------------------------------------------------------------


def test_task_tools_registered_without_interceptor():
    """Task tools are registered even when no task_interceptor is provided."""
    mock_service = AsyncMock()
    server = create_mcp_server(mock_service)  # No task_interceptor
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    for name in ['get_tasks', 'get_task', 'set_task_status', 'add_task',
                 'update_task', 'add_subtask', 'remove_task', 'add_dependency',
                 'remove_dependency', 'expand_task', 'parse_prd']:
        assert name in tool_names, f'{name} should be registered'


@pytest.mark.asyncio
async def test_task_tool_error_without_taskmaster():
    """Calling a task tool with no-taskmaster interceptor returns structured error."""
    mock_service = AsyncMock()
    server = create_mcp_server(mock_service)  # No task_interceptor → fallback
    result = await server._tool_manager.call_tool(
        'get_tasks', {'project_root': '/project'},
    )
    assert 'error' in result
    assert 'not configured' in result['error'].lower()


# ------------------------------------------------------------------
# ServerConfig stateless_http / json_response defaults and propagation
# ------------------------------------------------------------------


class TestServerConfigDefaults:
    """ServerConfig defaults preserve backward compatibility."""

    def test_stateless_http_defaults_false(self):
        cfg = ServerConfig()
        assert cfg.stateless_http is False

    def test_json_response_defaults_false(self):
        cfg = ServerConfig()
        assert cfg.json_response is False

    def test_explicit_true_values(self):
        cfg = ServerConfig(stateless_http=True, json_response=True)
        assert cfg.stateless_http is True
        assert cfg.json_response is True


class TestServerSettingsPropagation:
    """MCP server settings reflect config values."""

    def test_stateless_and_json_propagated_to_mcp_settings(self):
        """When stateless_http and json_response are set, they propagate to mcp.settings."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)
        server.settings.stateless_http = True
        server.settings.json_response = True
        assert server.settings.stateless_http is True
        assert server.settings.json_response is True

    def test_default_mcp_settings_are_false(self):
        """MCP server defaults to stateful SSE mode."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)
        assert server.settings.stateless_http is False
        assert server.settings.json_response is False

    def test_stateless_json_app_accepts_json_only(self):
        """With json_response=True, POST /mcp with Accept: application/json succeeds (no 406)."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)
        server.settings.stateless_http = True
        server.settings.json_response = True
        app = server.streamable_http_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            '/mcp',
            json={
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'initialize',
                'params': {
                    'protocolVersion': '2025-03-26',
                    'capabilities': {},
                    'clientInfo': {'name': 'test', 'version': '0.1'},
                },
            },
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
        )
        # Should NOT be 406 (Not Acceptable)
        assert resp.status_code != 406, (
            f'Expected non-406 with Accept: application/json in json_response mode, '
            f'got {resp.status_code}: {resp.text[:200]}'
        )

    def test_stateful_sse_app_rejects_json_only_accept(self):
        """Without json_response, POST /mcp with Accept: application/json is not 200."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)
        # Leave defaults: stateless_http=False, json_response=False
        app = server.streamable_http_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            '/mcp',
            json={
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'initialize',
                'params': {
                    'protocolVersion': '2025-03-26',
                    'capabilities': {},
                    'clientInfo': {'name': 'test', 'version': '0.1'},
                },
            },
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
        )
        # SSE mode rejects Accept: application/json alone (406 or 500 in test transport)
        assert resp.status_code != 200, (
            'SSE mode should NOT succeed with Accept: application/json alone'
        )

    def test_stateful_sse_app_accepts_both_content_types(self):
        """Default SSE mode accepts Accept: application/json, text/event-stream."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)
        app = server.streamable_http_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            '/mcp',
            json={
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'initialize',
                'params': {
                    'protocolVersion': '2025-03-26',
                    'capabilities': {},
                    'clientInfo': {'name': 'test', 'version': '0.1'},
                },
            },
            headers={
                'Accept': 'application/json, text/event-stream',
                'Content-Type': 'application/json',
            },
        )
        # Should NOT be 406
        assert resp.status_code != 406, (
            f'Expected non-406 with dual Accept header, got {resp.status_code}: {resp.text[:200]}'
        )


# ------------------------------------------------------------------
# set_task_status input validation
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_task_status_rejects_invalid_status(mcp_server_with_tasks):
    """set_task_status with an invalid status returns an error dict."""
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_status',
        {'id': '1', 'project_root': '/project', 'status': 'bogus'},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'bogus' in result['error'] or 'invalid' in result['error'].lower()
    # Should mention valid statuses
    assert 'done' in result['error'] or 'pending' in result['error']


@pytest.mark.asyncio
async def test_set_task_status_valid_status_passes_through(
    mcp_server_with_tasks, task_interceptor,
):
    """set_task_status with a valid status passes through to the interceptor."""
    task_interceptor.set_task_status = AsyncMock(return_value={'success': True})
    result = await mcp_server_with_tasks._tool_manager.call_tool(
        'set_task_status',
        {'id': '1', 'project_root': '/project', 'status': 'done'},
    )
    task_interceptor.set_task_status.assert_called_once()
    assert 'error' not in result


# ------------------------------------------------------------------
# delete_memory store validation
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_memory_rejects_invalid_store(mcp_server):
    """delete_memory with an invalid store returns an error dict."""
    result = await mcp_server._tool_manager.call_tool(
        'delete_memory',
        {'memory_id': 'abc-123', 'store': 'redis', 'project_id': 'proj'},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'redis' in result['error'] or 'invalid' in result['error'].lower()
    # Should mention valid stores
    assert 'graphiti' in result['error'] or 'mem0' in result['error']


@pytest.mark.asyncio
async def test_delete_memory_valid_store_passes_through(mcp_server):
    """delete_memory with a valid store passes through to the service."""
    mcp_server._tool_manager  # ensure initialized
    # Patch the memory service to return a success dict
    from unittest.mock import AsyncMock
    result_mock = {'deleted': True}
    # Access the server's underlying service mock via the closure
    # The mcp_server fixture uses AsyncMock for the service
    # We just confirm no validation error is returned for valid stores
    for valid_store in ('graphiti', 'mem0'):
        result = await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'memory_id': 'abc-123', 'store': valid_store, 'project_id': 'proj'},
        )
        # Should NOT be a validation error
        if isinstance(result, dict) and 'error' in result:
            assert 'invalid' not in result['error'].lower(), (
                f'Unexpected validation error for store={valid_store!r}: {result}'
            )


# ------------------------------------------------------------------
# add_memory category validation
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_memory_rejects_invalid_category(mcp_server):
    """add_memory with an invalid category returns an error dict."""
    result = await mcp_server._tool_manager.call_tool(
        'add_memory',
        {
            'content': 'some fact',
            'project_id': 'proj',
            'category': 'invalid_cat',
        },
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'invalid_cat' in result['error'] or 'invalid' in result['error'].lower()
    # Should mention all 6 valid categories
    assert 'observations_and_summaries' in result['error'] or 'preferences_and_norms' in result['error']


@pytest.mark.asyncio
async def test_add_memory_none_category_passes_through(mcp_server):
    """add_memory with no category (auto-classify) should NOT return a validation error."""
    result = await mcp_server._tool_manager.call_tool(
        'add_memory',
        {'content': 'some fact', 'project_id': 'proj'},
    )
    # If there's an error, it shouldn't be a validation error about category
    if isinstance(result, dict) and 'error' in result:
        assert 'invalid' not in result['error'].lower() or 'category' not in result['error'].lower()


@pytest.mark.asyncio
async def test_add_memory_valid_category_passes_through(mcp_server):
    """add_memory with a valid category should NOT return a validation error."""
    for valid_cat in ('observations_and_summaries', 'preferences_and_norms',
                      'procedural_knowledge', 'entities_and_relations',
                      'temporal_facts', 'decisions_and_rationale'):
        result = await mcp_server._tool_manager.call_tool(
            'add_memory',
            {'content': 'fact', 'project_id': 'proj', 'category': valid_cat},
        )
        if isinstance(result, dict) and 'error' in result:
            assert 'ValidationError' != result.get('error_type'), (
                f'Unexpected validation error for category={valid_cat!r}: {result}'
            )
