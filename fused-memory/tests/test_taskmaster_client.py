"""Tests for Taskmaster MCP client backend."""

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import TextContent

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.config.schema import TaskmasterConfig


@pytest.fixture
def config():
    return TaskmasterConfig(
        transport='stdio',
        command='node',
        args=['server.js'],
        project_root='/project',
    )


@pytest.fixture
def client(config):
    c = TaskmasterBackend(config)
    # Mock the session
    mock_session = AsyncMock()
    c._session = mock_session
    return c, mock_session


def _mock_tool_result(data: dict):
    """Create a mock MCP tool result."""
    result = MagicMock()
    text_block = TextContent(type='text', text=json.dumps(data))
    result.content = [text_block]
    return result


@pytest.mark.asyncio
async def test_call_tool(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_mock_tool_result({'tasks': []}))

    result = await c.call_tool('get_tasks', {'projectRoot': '/project'})
    assert result == {'tasks': []}
    session.call_tool.assert_called_once_with('get_tasks', {'projectRoot': '/project'})


@pytest.mark.asyncio
async def test_base_args_with_tag(client):
    c, session = client
    args = c._base_args('/project', 'my-tag')
    assert args['projectRoot'] == '/project'
    assert args['tag'] == 'my-tag'


def test_base_args_rejects_dot_project_root():
    """_base_args must reject '.' — not an absolute path."""
    cfg = TaskmasterConfig(
        transport='stdio', command='node', args=['server.js'], project_root='.',
    )
    c = TaskmasterBackend(cfg)
    with pytest.raises(ValueError, match='project_root is required'):
        c._base_args(project_root='.')


def test_base_args_rejects_empty_project_root():
    """_base_args must reject empty string."""
    cfg = TaskmasterConfig(
        transport='stdio', command='node', args=['server.js'], project_root='',
    )
    c = TaskmasterBackend(cfg)
    with pytest.raises(ValueError, match='project_root is required'):
        c._base_args(project_root='')


@pytest.mark.asyncio
async def test_require_session_raises_without_init(config):
    c = TaskmasterBackend(config)
    with pytest.raises(RuntimeError, match='not connected'):
        c._require_session()


@pytest.mark.asyncio
async def test_initialize_sets_metadata_updates_env(config):
    """TASK_MASTER_ALLOW_METADATA_UPDATES must be set in subprocess env."""
    import contextlib
    from unittest.mock import patch

    c = TaskmasterBackend(config)

    captured_params = {}

    @contextlib.asynccontextmanager
    async def fake_ctx(params):
        captured_params['params'] = params
        yield (AsyncMock(), AsyncMock())

    with patch(
        'fused_memory.backends.taskmaster_client.stdio_client',
        side_effect=lambda p: fake_ctx(p),
    ):
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        # Patch ClientSession to return our mock
        with patch(
            'fused_memory.backends.taskmaster_client.ClientSession',
            return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_session),
                __aexit__=AsyncMock(return_value=False),
            ),
        ):
            await c.initialize()

    assert 'params' in captured_params
    env = captured_params['params'].env
    assert env is not None
    assert env.get('TASK_MASTER_ALLOW_METADATA_UPDATES') == 'true'
    assert env.get('TASK_MASTER_TOOLS') == config.tool_mode


@pytest.mark.asyncio
async def test_call_tool_non_json_response(client):
    """Non-JSON response should be wrapped."""
    c, session = client
    result = MagicMock()
    text_block = TextContent(type='text', text='Not JSON')
    result.content = [text_block]
    session.call_tool = AsyncMock(return_value=result)

    data = await c.call_tool('some_tool', {})
    assert data == {'text': 'Not JSON'}


# ── connected property and ensure_connected tests ──────────────────


def test_connected_false_before_init(config):
    """connected is False on a fresh instance with no session."""
    c = TaskmasterBackend(config)
    assert c.connected is False


def test_connected_true_with_session(client):
    """connected is True when _session is set."""
    c, _ = client
    assert c.connected is True


@pytest.mark.asyncio
async def test_ensure_connected_noop_when_connected(client):
    """ensure_connected returns immediately when already connected."""
    c, session = client
    await c.ensure_connected()
    # initialize should NOT have been called (session was already set)
    assert c._session is session


@pytest.mark.asyncio
async def test_ensure_connected_triggers_reconnect(config):
    """ensure_connected calls initialize() when disconnected and cooldown elapsed."""
    c = TaskmasterBackend(config)
    assert c.connected is False

    mock_session = AsyncMock()
    # Patch initialize to just set the session
    async def fake_init():
        c._session = mock_session

    c.initialize = fake_init
    await c.ensure_connected()
    assert c.connected is True
    assert c._session is mock_session


@pytest.mark.asyncio
async def test_ensure_connected_cooldown_raises(config):
    """ensure_connected raises RuntimeError when cooldown hasn't elapsed."""
    c = TaskmasterBackend(config, reconnect_cooldown_seconds=60.0)
    # Simulate a recent failed reconnect attempt
    c._last_reconnect_attempt = time.monotonic()

    with pytest.raises(RuntimeError, match='cooldown'):
        await c.ensure_connected()


@pytest.mark.asyncio
async def test_call_tool_broken_pipe_resets_session(client):
    """BrokenPipeError from session.call_tool resets _session to None."""
    c, session = client
    session.call_tool = AsyncMock(side_effect=BrokenPipeError('pipe gone'))
    # Stub out _cleanup_contexts so it just clears _session
    c._cleanup_contexts = AsyncMock(side_effect=lambda: setattr(c, '_session', None))

    with pytest.raises(BrokenPipeError):
        await c.call_tool('get_tasks', {})

    c._cleanup_contexts.assert_called_once()
    assert c._session is None


# ── transport-dead exception handling in call_tool ─────────────────


@pytest.mark.asyncio
async def test_call_tool_closed_resource_error_resets_session(client):
    """anyio.ClosedResourceError must trigger cleanup — this is the
    exact exception class seen in esc-1956-44 that the narrow except
    list used to miss.
    """
    import anyio

    c, session = client
    session.call_tool = AsyncMock(side_effect=anyio.ClosedResourceError())
    c._cleanup_contexts = AsyncMock(side_effect=lambda: setattr(c, '_session', None))

    with pytest.raises(anyio.ClosedResourceError):
        await c.call_tool('get_tasks', {})

    c._cleanup_contexts.assert_called_once()
    assert c._session is None


@pytest.mark.asyncio
async def test_call_tool_broken_resource_error_resets_session(client):
    """anyio.BrokenResourceError must trigger cleanup."""
    import anyio

    c, session = client
    session.call_tool = AsyncMock(side_effect=anyio.BrokenResourceError())
    c._cleanup_contexts = AsyncMock(side_effect=lambda: setattr(c, '_session', None))

    with pytest.raises(anyio.BrokenResourceError):
        await c.call_tool('get_tasks', {})

    c._cleanup_contexts.assert_called_once()
    assert c._session is None


@pytest.mark.asyncio
async def test_call_tool_mcp_connection_closed_resets_session(client):
    """McpError with CONNECTION_CLOSED code must trigger cleanup."""
    from mcp.shared.exceptions import McpError
    from mcp.types import CONNECTION_CLOSED, ErrorData

    c, session = client
    err = McpError(ErrorData(code=CONNECTION_CLOSED, message='closed'))
    session.call_tool = AsyncMock(side_effect=err)
    c._cleanup_contexts = AsyncMock(side_effect=lambda: setattr(c, '_session', None))

    with pytest.raises(McpError):
        await c.call_tool('get_tasks', {})

    c._cleanup_contexts.assert_called_once()
    assert c._session is None


@pytest.mark.asyncio
async def test_call_tool_mcp_request_timeout_resets_session(client):
    """McpError with REQUEST_TIMEOUT code (408) must trigger cleanup —
    this is how send_request surfaces a hung stdio subprocess."""
    import httpx
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    c, session = client
    err = McpError(ErrorData(code=int(httpx.codes.REQUEST_TIMEOUT), message='timeout'))
    session.call_tool = AsyncMock(side_effect=err)
    c._cleanup_contexts = AsyncMock(side_effect=lambda: setattr(c, '_session', None))

    with pytest.raises(McpError):
        await c.call_tool('get_tasks', {})

    c._cleanup_contexts.assert_called_once()
    assert c._session is None


@pytest.mark.asyncio
async def test_call_tool_passes_through_tool_level_mcp_errors(client):
    """Tool-level McpError (INTERNAL_ERROR etc.) must NOT tear down
    the session — the proxy is alive, Taskmaster just said no."""
    from mcp.shared.exceptions import McpError
    from mcp.types import INTERNAL_ERROR, ErrorData

    c, session = client
    err = McpError(ErrorData(code=INTERNAL_ERROR, message='tool refused'))
    session.call_tool = AsyncMock(side_effect=err)
    c._cleanup_contexts = AsyncMock()

    with pytest.raises(McpError):
        await c.call_tool('some_tool', {})

    c._cleanup_contexts.assert_not_called()
    assert c._session is session


# ── is_alive() probe ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_is_alive_returns_false_when_disconnected(config):
    """is_alive reports (False, 'not connected') and does not probe
    when _session is None."""
    c = TaskmasterBackend(config)
    alive, err = await c.is_alive()
    assert alive is False
    assert err == 'not connected'


@pytest.mark.asyncio
async def test_is_alive_probes_via_get_tasks(client):
    """Healthy probe calls get_tasks with configured projectRoot and
    reports (True, None)."""
    c, session = client
    session.call_tool = AsyncMock(return_value=_mock_tool_result({'tasks': []}))

    alive, err = await c.is_alive()

    assert alive is True
    assert err is None
    call_args = session.call_tool.call_args[0]
    assert call_args[0] == 'get_tasks'
    assert call_args[1]['projectRoot'] == '/project'


@pytest.mark.asyncio
async def test_is_alive_reports_false_on_closed_resource_error(client):
    """ClosedResourceError from the probe → (False, ...) and _session
    is torn down so next mutating call reconnects."""
    import anyio

    c, session = client
    session.call_tool = AsyncMock(side_effect=anyio.ClosedResourceError())
    c._cleanup_contexts = AsyncMock(side_effect=lambda: setattr(c, '_session', None))

    alive, err = await c.is_alive()

    assert alive is False
    assert err is not None and 'ClosedResourceError' in err
    assert c._session is None


@pytest.mark.asyncio
async def test_is_alive_treats_tool_level_mcp_error_as_healthy(client):
    """A tool-level McpError response still proves the stdio round-trip
    completed — that is proof-of-life for the proxy."""
    from mcp.shared.exceptions import McpError
    from mcp.types import INTERNAL_ERROR, ErrorData

    c, session = client
    session.call_tool = AsyncMock(
        side_effect=McpError(ErrorData(code=INTERNAL_ERROR, message='bad project')),
    )
    c._cleanup_contexts = AsyncMock()

    alive, err = await c.is_alive()

    assert alive is True
    assert err is None
    c._cleanup_contexts.assert_not_called()


@pytest.mark.asyncio
async def test_is_alive_caches_within_ttl(client):
    """Two rapid is_alive calls hit session.call_tool only once."""
    c, session = client
    session.call_tool = AsyncMock(return_value=_mock_tool_result({'tasks': []}))

    await c.is_alive()
    await c.is_alive()

    assert session.call_tool.call_count == 1


@pytest.mark.asyncio
async def test_is_alive_reprobes_after_ttl(config):
    """After cache TTL expires, is_alive probes again."""
    c = TaskmasterBackend(config, alive_cache_ttl_seconds=0.0)
    session = AsyncMock()
    session.call_tool = AsyncMock(return_value=_mock_tool_result({'tasks': []}))
    c._session = session

    await c.is_alive()
    await c.is_alive()

    assert session.call_tool.call_count == 2


# ── close() lifecycle tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_suppresses_session_ctx_aexit_exception(config):
    """close() must not raise when _session_ctx.__aexit__ raises, and must
    still tear down _stdio_ctx and clear all three state attributes."""
    c = TaskmasterBackend(config)
    session_ctx = MagicMock()
    session_ctx.__aexit__ = AsyncMock(side_effect=RuntimeError('boom'))
    stdio_ctx = MagicMock()
    stdio_ctx.__aexit__ = AsyncMock(return_value=False)
    c._session_ctx = session_ctx
    c._stdio_ctx = stdio_ctx
    c._session = AsyncMock()

    # Must not raise despite session_ctx.__aexit__ raising
    await c.close()

    # stdio_ctx.__aexit__ must still have been awaited
    stdio_ctx.__aexit__.assert_awaited_once()
    # All three state attributes must be cleared
    assert c._session is None
    assert c._session_ctx is None
    assert c._stdio_ctx is None


@pytest.mark.asyncio
async def test_close_suppresses_stdio_ctx_aexit_exception(config):
    """close() must not raise when _stdio_ctx.__aexit__ raises, and must
    clear all three state attributes."""
    c = TaskmasterBackend(config)
    session_ctx = MagicMock()
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    stdio_ctx = MagicMock()
    stdio_ctx.__aexit__ = AsyncMock(side_effect=RuntimeError('stdio gone'))
    c._session_ctx = session_ctx
    c._stdio_ctx = stdio_ctx
    c._session = AsyncMock()

    # Must not raise despite stdio_ctx.__aexit__ raising
    await c.close()

    # All three state attributes must be cleared
    assert c._session is None
    assert c._session_ctx is None
    assert c._stdio_ctx is None


@pytest.mark.asyncio
async def test_close_clears_state_attributes_on_normal_teardown(config):
    """close() must await both __aexit__ calls and clear all three
    state attributes on a clean (non-raising) teardown."""
    c = TaskmasterBackend(config)
    session_ctx = MagicMock()
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    stdio_ctx = MagicMock()
    stdio_ctx.__aexit__ = AsyncMock(return_value=False)
    c._session_ctx = session_ctx
    c._stdio_ctx = stdio_ctx
    c._session = AsyncMock()

    await c.close()

    session_ctx.__aexit__.assert_awaited_once()
    stdio_ctx.__aexit__.assert_awaited_once()
    assert c._session is None
    assert c._session_ctx is None
    assert c._stdio_ctx is None
