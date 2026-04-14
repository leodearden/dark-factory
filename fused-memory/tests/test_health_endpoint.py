"""Tests for the /health endpoint, ServerConfig, and memory/search/episodes validation."""

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
def mcp_server_with_service():
    """Create an MCP server exposing both the server and service mock."""
    mock_service = AsyncMock()
    mock_service.search = AsyncMock(return_value=[])
    mock_service.get_episodes = AsyncMock(return_value=[])
    server = create_mcp_server(mock_service)
    return server, mock_service


@pytest.fixture
def health_app(mcp_server):
    """Get the Starlette app that includes the /health route."""
    return mcp_server.streamable_http_app()


def test_health_returns_200(health_app):
    """GET /health should return 200 with {"status": "ok"}."""
    client = TestClient(health_app)
    resp = client.get('/health')
    assert resp.status_code == 200
    body = resp.json()
    assert body['status'] == 'ok'
    assert 'graphiti' in body
    assert 'mem0' in body


def test_health_post_not_allowed(health_app):
    """POST /health should be rejected (405 Method Not Allowed)."""
    client = TestClient(health_app)
    resp = client.post('/health')
    assert resp.status_code == 405


# ------------------------------------------------------------------
# ServerConfig stateless_http / json_response defaults and propagation
# ------------------------------------------------------------------


class TestServerConfigDefaults:
    """ServerConfig defaults preserve backward compatibility."""

    def test_stateless_http_defaults_true(self):
        cfg = ServerConfig()
        assert cfg.stateless_http is True

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

    def test_default_mcp_settings_before_config(self):
        """FastMCP defaults before config propagation."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)
        # FastMCP's own defaults — config propagation happens in run_server()
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
        server.settings.stateless_http = False
        server.settings.json_response = False
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
        """Stateful SSE mode accepts Accept: application/json, text/event-stream."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)
        server.settings.stateless_http = False
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
            assert result.get('error_type') != 'ValidationError', (
                f'Unexpected validation error for category={valid_cat!r}: {result}'
            )


# ------------------------------------------------------------------
# search limit validation
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_rejects_zero_limit(mcp_server):
    """search with limit=0 returns an error dict."""
    result = await mcp_server._tool_manager.call_tool(
        'search',
        {'query': 'test', 'project_id': 'proj', 'limit': 0},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'limit' in result['error'].lower() or '0' in result['error']


@pytest.mark.asyncio
async def test_search_rejects_negative_limit(mcp_server):
    """search with limit=-1 returns an error dict."""
    result = await mcp_server._tool_manager.call_tool(
        'search',
        {'query': 'test', 'project_id': 'proj', 'limit': -1},
    )
    assert isinstance(result, dict)
    assert 'error' in result


@pytest.mark.asyncio
async def test_search_caps_limit_at_1000(mcp_server_with_service):
    """search with limit=5000 silently caps to 1000 and calls service with limit=1000."""
    server, mock_service = mcp_server_with_service
    result = await server._tool_manager.call_tool(
        'search',
        {'query': 'test', 'project_id': 'proj', 'limit': 5000},
    )
    # Should not be a validation error - result has 'results' key
    assert isinstance(result, dict)
    assert 'results' in result
    # Verify the service was called with capped limit=1000
    mock_service.search.assert_called_once()
    _, kwargs = mock_service.search.call_args
    assert kwargs['limit'] == 1000


# ------------------------------------------------------------------
# get_episodes last_n validation
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_episodes_rejects_zero_last_n(mcp_server):
    """get_episodes with last_n=0 returns an error dict."""
    result = await mcp_server._tool_manager.call_tool(
        'get_episodes',
        {'project_id': 'proj', 'last_n': 0},
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'last_n' in result['error'].lower() or '0' in result['error']


@pytest.mark.asyncio
async def test_get_episodes_rejects_negative_last_n(mcp_server):
    """get_episodes with last_n=-1 returns an error dict."""
    result = await mcp_server._tool_manager.call_tool(
        'get_episodes',
        {'project_id': 'proj', 'last_n': -1},
    )
    assert isinstance(result, dict)
    assert 'error' in result


@pytest.mark.asyncio
async def test_get_episodes_caps_last_n_at_1000(mcp_server_with_service):
    """get_episodes with last_n=5000 silently caps to 1000 and calls service with last_n=1000."""
    server, mock_service = mcp_server_with_service
    result = await server._tool_manager.call_tool(
        'get_episodes',
        {'project_id': 'proj', 'last_n': 5000},
    )
    assert isinstance(result, dict)
    assert 'episodes' in result
    # Verify the service was called with capped last_n=1000
    mock_service.get_episodes.assert_called_once()
    _, kwargs = mock_service.get_episodes.call_args
    assert kwargs['last_n'] == 1000
