"""Tests for MCP session retry logic in mcp_lifecycle.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from orchestrator.mcp_lifecycle import (
    _MCP_BACKOFF_BASE,
    _MCP_MAX_RETRIES,
    McpLifecycle,
    McpSession,
)


def _make_response(
    status_code: int = 200,
    json_body: dict | None = None,
    session_id: str | None = None,
) -> httpx.Response:
    """Build a fake httpx.Response."""
    body = json_body or {'jsonrpc': '2.0', 'id': 1, 'result': {'ok': True}}
    resp = httpx.Response(
        status_code=status_code,
        json=body,
        headers={'content-type': 'application/json', **(
            {'mcp-session-id': session_id} if session_id else {}
        )},
        request=httpx.Request('POST', 'http://localhost:8002/mcp'),
    )
    return resp


class TestRawCallRetry:
    """_raw_call retries on transient failures."""

    @pytest.mark.asyncio
    async def test_succeeds_first_attempt(self):
        session = McpSession('http://localhost:8002')
        ok = _make_response(session_id='s1')
        mock_client = AsyncMock()
        mock_client.post.return_value = ok

        with patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await session._raw_call('tools/call', {'name': 'test', 'arguments': {}})

        assert result == {'jsonrpc': '2.0', 'id': 1, 'result': {'ok': True}}
        assert mock_client.post.call_count == 1
        assert session._session_id == 's1'

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(self):
        session = McpSession('http://localhost:8002')
        session._session_id = 'old-session'

        ok = _make_response(session_id='new-session')
        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            httpx.ConnectError('Connection refused'),
            httpx.ConnectError('Connection refused'),
            ok,
        ]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await session._raw_call('initialize')

        assert result['result'] == {'ok': True}
        assert mock_client.post.call_count == 3
        # Session ID should be cleared on retry then set from response
        assert session._session_id == 'new-session'
        # Backoff: sleep(1), sleep(2)
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0].args[0] == pytest.approx(_MCP_BACKOFF_BASE * 1)
        assert mock_sleep.call_args_list[1].args[0] == pytest.approx(_MCP_BACKOFF_BASE * 2)

    @pytest.mark.asyncio
    async def test_retries_on_503(self):
        session = McpSession('http://localhost:8002')

        err_resp = _make_response(status_code=503)
        ok_resp = _make_response()
        mock_client = AsyncMock()
        mock_client.post.side_effect = [err_resp, ok_resp]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await session._raw_call('tools/call', {'name': 't', 'arguments': {}})

        assert result['result'] == {'ok': True}
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        session = McpSession('http://localhost:8002')

        err_resp = _make_response(status_code=400)
        mock_client = AsyncMock()
        mock_client.post.return_value = err_resp

        with patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(httpx.HTTPStatusError):
                await session._raw_call('tools/call', {'name': 't', 'arguments': {}})

        # Only one attempt — no retry on 4xx
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        session = McpSession('http://localhost:8002')

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError('Connection refused')

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(httpx.ConnectError):
                await session._raw_call('tools/call', {'name': 't', 'arguments': {}})

        assert mock_client.post.call_count == _MCP_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_clears_initialized_on_retry(self):
        session = McpSession('http://localhost:8002')
        session._initialized = True
        session._session_id = 'old'

        ok = _make_response()
        mock_client = AsyncMock()
        mock_client.post.side_effect = [httpx.ConnectError('refused'), ok]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await session._raw_call('tools/call', {'name': 't', 'arguments': {}})

        # _initialized cleared during retry so call_tool will re-init
        assert session._initialized is False


class TestRawNotifyRetry:
    """_raw_notify retries on transient connection failures."""

    @pytest.mark.asyncio
    async def test_succeeds_first_attempt(self):
        session = McpSession('http://localhost:8002')

        mock_client = AsyncMock()
        mock_client.post.return_value = _make_response(status_code=202)

        with patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await session._raw_notify('notifications/initialized')

        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(self):
        session = McpSession('http://localhost:8002')

        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            httpx.ConnectError('refused'),
            _make_response(status_code=200),
        ]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await session._raw_notify('notifications/initialized')

        assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# McpLifecycle process-group tests (step-19 / step-20)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMcpLifecycleProcessGroup:
    """McpLifecycle must spawn its subprocess in a fresh process group."""

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> MagicMock:
        """Minimal config-shaped mock for McpLifecycle."""
        cfg = MagicMock()
        cfg.fused_memory.server_command = ['echo', 'ok']
        cfg.fused_memory.url = 'http://localhost:8002'
        cfg.fused_memory.config_path = 'fused-memory/config/config.yaml'
        cfg.project_root = tmp_path
        return cfg

    async def test_mcp_lifecycle_starts_subprocess_in_new_session(
        self, mock_config: MagicMock
    ) -> None:
        """create_subprocess_exec is called with start_new_session=True.

        Failing test — mcp_lifecycle.start() does not pass that kwarg yet.
        """
        captured_kwargs: dict = {}

        async def fake_exec(*args: object, **kwargs: object) -> MagicMock:
            captured_kwargs.update(kwargs)
            proc = MagicMock()
            proc.returncode = None
            proc.stdout = MagicMock()
            proc.stderr = MagicMock()
            return proc

        with (
            patch(
                'orchestrator.mcp_lifecycle.asyncio.create_subprocess_exec',
                side_effect=fake_exec,
            ),
            patch.object(
                McpLifecycle, '_wait_for_health', new=AsyncMock(return_value=True)
            ),
            patch('orchestrator.mcp_lifecycle.McpSession') as mock_session_cls,
        ):
            mock_session_cls.return_value.initialize = AsyncMock()
            mcp = McpLifecycle(mock_config)
            await mcp.start()

        assert captured_kwargs.get('start_new_session') is True, (
            'create_subprocess_exec must be called with start_new_session=True'
        )

    async def test_mcp_lifecycle_stop_uses_terminate_process_group(
        self, mock_config: MagicMock
    ) -> None:
        """stop() must delegate to terminate_process_group, not bare terminate/kill.

        Failing test — mcp_lifecycle.stop() uses proc.terminate()/proc.kill() directly.
        """
        proc = MagicMock()
        proc.returncode = None

        with patch(
            'orchestrator.mcp_lifecycle.terminate_process_group',
            new_callable=AsyncMock,
        ) as mock_tpg:
            mcp = McpLifecycle(mock_config)
            mcp._process = proc
            mcp._pgid = 12345  # captured at spawn in production; stub here
            await mcp.stop()

        mock_tpg.assert_awaited_once()
