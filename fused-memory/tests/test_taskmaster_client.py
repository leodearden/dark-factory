"""Tests for Taskmaster MCP client backend.

Covers the public surface of :class:`TaskmasterBackend` — ``call_tool``,
``is_alive``, ``connected``, and the convenience-method helpers — using a
session mock plumbed in past the supervisor. The supervisor lifecycle
itself (spawn, respawn, lock, escalation, close) lives in
``test_taskmaster_supervisor.py``.
"""

import json
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
    """Backend with a mock session installed and ``_session_ready`` set,
    bypassing the supervisor for direct tool-dispatch tests."""
    c = TaskmasterBackend(config)
    mock_session = AsyncMock()
    c._session = mock_session
    c._session_ready.set()
    return c, mock_session


def _mock_tool_result(data: dict):
    """Create a mock MCP tool result wrapping ``data`` as JSON text."""
    result = MagicMock()
    text_block = TextContent(type='text', text=json.dumps(data))
    result.content = [text_block]
    return result


# ── call_tool dispatch ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_tool(client):
    c, session = client
    session.call_tool = AsyncMock(return_value=_mock_tool_result({'tasks': []}))

    result = await c.call_tool('get_tasks', {'projectRoot': '/project'})
    assert result == {'tasks': []}
    session.call_tool.assert_called_once_with('get_tasks', {'projectRoot': '/project'})


@pytest.mark.asyncio
async def test_call_tool_non_json_response(client):
    """Non-JSON response should be wrapped in {'text': ...}."""
    c, session = client
    result = MagicMock()
    text_block = TextContent(type='text', text='Not JSON')
    result.content = [text_block]
    session.call_tool = AsyncMock(return_value=result)

    data = await c.call_tool('some_tool', {})
    assert data == {'text': 'Not JSON'}


@pytest.mark.asyncio
async def test_call_tool_empty_content_returns_empty_dict(client):
    c, session = client
    result = MagicMock()
    result.content = []
    session.call_tool = AsyncMock(return_value=result)

    data = await c.call_tool('some_tool', {})
    assert data == {}


# ── transport-dead exception handling in call_tool ────────────────────


@pytest.mark.asyncio
async def test_call_tool_broken_pipe_clears_ready(client):
    """BrokenPipeError clears _session_ready so subsequent callers block
    until the supervisor respawns."""
    c, session = client
    session.call_tool = AsyncMock(side_effect=BrokenPipeError('pipe gone'))

    with pytest.raises(BrokenPipeError):
        await c.call_tool('get_tasks', {})

    assert c._session_ready.is_set() is False


@pytest.mark.asyncio
async def test_call_tool_closed_resource_error_clears_ready(client):
    """anyio.ClosedResourceError — the exact class seen in esc-1956-44 —
    must trigger the same ``_session_ready`` teardown path."""
    import anyio

    c, session = client
    session.call_tool = AsyncMock(side_effect=anyio.ClosedResourceError())

    with pytest.raises(anyio.ClosedResourceError):
        await c.call_tool('get_tasks', {})

    assert c._session_ready.is_set() is False


@pytest.mark.asyncio
async def test_call_tool_broken_resource_error_clears_ready(client):
    """anyio.BrokenResourceError must clear the ready event."""
    import anyio

    c, session = client
    session.call_tool = AsyncMock(side_effect=anyio.BrokenResourceError())

    with pytest.raises(anyio.BrokenResourceError):
        await c.call_tool('get_tasks', {})

    assert c._session_ready.is_set() is False


@pytest.mark.asyncio
async def test_call_tool_mcp_connection_closed_clears_ready(client):
    """McpError with CONNECTION_CLOSED code must clear the ready event."""
    from mcp.shared.exceptions import McpError
    from mcp.types import CONNECTION_CLOSED, ErrorData

    c, session = client
    err = McpError(ErrorData(code=CONNECTION_CLOSED, message='closed'))
    session.call_tool = AsyncMock(side_effect=err)

    with pytest.raises(McpError):
        await c.call_tool('get_tasks', {})

    assert c._session_ready.is_set() is False


@pytest.mark.asyncio
async def test_call_tool_mcp_request_timeout_clears_ready(client):
    """McpError with REQUEST_TIMEOUT (408) — how send_request surfaces a
    hung stdio subprocess — must clear the ready event."""
    import httpx
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    c, session = client
    err = McpError(ErrorData(code=int(httpx.codes.REQUEST_TIMEOUT), message='timeout'))
    session.call_tool = AsyncMock(side_effect=err)

    with pytest.raises(McpError):
        await c.call_tool('get_tasks', {})

    assert c._session_ready.is_set() is False


@pytest.mark.asyncio
async def test_call_tool_passes_through_tool_level_mcp_errors(client):
    """Tool-level McpError (INTERNAL_ERROR etc.) must NOT clear the ready
    event — the proxy is alive, Taskmaster just said no."""
    from mcp.shared.exceptions import McpError
    from mcp.types import INTERNAL_ERROR, ErrorData

    c, session = client
    err = McpError(ErrorData(code=INTERNAL_ERROR, message='tool refused'))
    session.call_tool = AsyncMock(side_effect=err)

    with pytest.raises(McpError):
        await c.call_tool('some_tool', {})

    assert c._session_ready.is_set() is True


# ── connected property and ensure_connected ──────────────────────────


def test_connected_false_before_start(config):
    """connected is False on a fresh instance with no supervisor running."""
    c = TaskmasterBackend(config)
    assert c.connected is False


def test_connected_true_when_ready(client):
    """connected reflects ``_session_ready``."""
    c, _ = client
    assert c.connected is True


@pytest.mark.asyncio
async def test_ensure_connected_noop_when_ready(client):
    """ensure_connected returns immediately when ``_session_ready`` is set.

    The fixture set ``_session_ready`` directly without spawning a
    supervisor; ensure_connected should still see the event and return.
    """
    c, _ = client
    c._supervisor_task = MagicMock()  # any non-None placeholder
    await c.ensure_connected()  # should not raise


@pytest.mark.asyncio
async def test_ensure_connected_raises_without_supervisor(config):
    """ensure_connected raises RuntimeError when start() was never called."""
    c = TaskmasterBackend(config)
    with pytest.raises(RuntimeError, match='not started'):
        await c.ensure_connected()


# ── _base_args validation ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_base_args_with_tag(client):
    c, _ = client
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
async def test_require_session_raises_without_session(config):
    c = TaskmasterBackend(config)
    with pytest.raises(RuntimeError, match='not connected'):
        c._require_session()


# ── is_alive() probe ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_is_alive_returns_false_when_disconnected(config):
    """is_alive reports (False, 'not connected') when ``_session_ready``
    is unset and does not attempt a probe."""
    c = TaskmasterBackend(config)
    alive, err = await c.is_alive()
    assert alive is False
    assert err == 'not connected'


@pytest.mark.asyncio
async def test_is_alive_probes_via_get_tasks(client):
    """Healthy probe calls get_tasks with the configured projectRoot and
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
    """ClosedResourceError from the probe → (False, ...) and ready event
    cleared so next mutating call blocks."""
    import anyio

    c, session = client
    session.call_tool = AsyncMock(side_effect=anyio.ClosedResourceError())

    alive, err = await c.is_alive()

    assert alive is False
    assert err is not None and 'ClosedResourceError' in err
    assert c._session_ready.is_set() is False


@pytest.mark.asyncio
async def test_is_alive_treats_tool_level_mcp_error_as_healthy(client):
    """A tool-level McpError still proves the stdio round-trip completed —
    proof-of-life for the proxy."""
    from mcp.shared.exceptions import McpError
    from mcp.types import INTERNAL_ERROR, ErrorData

    c, session = client
    session.call_tool = AsyncMock(
        side_effect=McpError(ErrorData(code=INTERNAL_ERROR, message='bad project')),
    )

    alive, err = await c.is_alive()

    assert alive is True
    assert err is None


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
    c._session_ready.set()

    await c.is_alive()
    await c.is_alive()

    assert session.call_tool.call_count == 2
