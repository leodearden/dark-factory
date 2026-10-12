"""The orchestrator's fused-memory session wiring in mcp_lifecycle.py.

The MCP transport itself (retry loop, client_op_id, timeout predicate) is
shared.mcp_session.McpSession, tested in shared/tests/test_mcp_session.py.
What is pinned here is the orchestrator's policy on top of it: fm_session's
identity and injected fm-restart retry window, the D8 null sentinel's
fail-fast, and mcp_call reaching the shared client.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import shared.mcp_session
from _orch_helpers import pydantic_spec
from shared.mcp_session import McpCallError
from shared.testing_mcp_stub import StubMcpServer

import orchestrator.mcp_lifecycle
from orchestrator.config import OrchestratorConfig
from orchestrator.fm_retry import FM_NULL_SENTINEL_URL, fm_retry_backoffs
from orchestrator.mcp_lifecycle import (
    ORCHESTRATOR_MCP_IDENTITY,
    McpLifecycle,
    fm_session,
    get_session,
    mcp_call,
)

OK_BODY = {'jsonrpc': '2.0', 'id': 1, 'result': {'ok': True}}


def _response(status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=OK_BODY,
        headers={'content-type': 'application/json'},
        request=httpx.Request('POST', 'http://localhost:8002/mcp'),
    )


@contextmanager
def _scripted(answers: Any) -> Iterator[tuple[AsyncMock, AsyncMock]]:
    """Patch the httpx client seam so each post() takes the next scripted answer."""
    client = AsyncMock()
    client.post.side_effect = answers
    with (
        patch('httpx.AsyncClient') as client_cls,
        patch('asyncio.sleep', new_callable=AsyncMock) as sleep,
    ):
        client_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        yield client, sleep


class TestFmSessionRetrySchedule:
    """fm_session rides fm's restart window (task 2706) except at the D8 null sentinel (task 2880)."""

    @pytest.mark.asyncio
    async def test_sentinel_tool_call_fails_after_one_attempt(self):
        with (
            _scripted(httpx.ConnectError('refused')) as (client, sleep),
            pytest.raises(McpCallError),
        ):
            await fm_session(FM_NULL_SENTINEL_URL).call_tool('get_tasks', {})

        assert client.post.call_count == 1
        assert sleep.call_count == 0

    @pytest.mark.asyncio
    async def test_sentinel_handshake_notification_fails_after_one_attempt(self):
        with (
            _scripted([_response(), httpx.ConnectError('refused')]) as (client, sleep),
            pytest.raises(McpCallError),
        ):
            await fm_session(FM_NULL_SENTINEL_URL).initialize()

        assert client.post.call_count == 2
        assert sleep.call_count == 0

    @pytest.mark.asyncio
    async def test_a_real_endpoint_spans_the_fm_restart_window(self):
        """random.uniform is pinned to the max draw while fm_session draws its
        schedule, so this test's own fm_retry_backoffs() call agrees with it."""
        expected = fm_retry_backoffs(rng=lambda lo, hi: hi)
        with patch('random.uniform', side_effect=lambda lo, hi: hi):
            session = fm_session('http://localhost:8002')

        with (
            _scripted(httpx.ConnectError('refused')) as (client, sleep),
            pytest.raises(McpCallError),
        ):
            await session.call_tool('get_tasks', {})

        assert len(expected) + 1 > 3
        assert client.post.call_count == len(expected) + 1
        assert [c.args[0] for c in sleep.call_args_list] == expected

    @pytest.mark.asyncio
    async def test_a_real_endpoint_recovers_after_transient_errors(self):
        answers = [
            httpx.ConnectError('refused'),
            httpx.ConnectError('refused'),
            _response(),
            _response(status_code=202),
            _response(),
        ]
        with _scripted(answers) as (client, _sleep):
            result = await fm_session('http://localhost:8002').call_tool('get_tasks', {})

        assert result == OK_BODY
        assert client.post.call_count == 5


@pytest.mark.asyncio
async def test_fm_session_identifies_as_the_orchestrator():
    with StubMcpServer(stateful=True) as server:
        await fm_session(server.url).call_tool('get_statuses', {'project_root': '/p'})
        requests = server.requests
        minted = server.minted_session_ids

    assert requests[0].body['params']['clientInfo']['name'] == ORCHESTRATOR_MCP_IDENTITY
    assert ORCHESTRATOR_MCP_IDENTITY == 'orchestrator'
    assert [r.headers.get('mcp-session-id') for r in requests[1:]] == [minted[0]] * 2


@pytest.mark.asyncio
async def test_mcp_call_one_shot_lands_a_write_through_the_shared_session():
    """With no McpLifecycle singleton (the unit-test default), mcp_call builds a
    one-shot fm_session and the write reaches the server."""
    with pytest.raises(RuntimeError):
        get_session()
    arguments = {'content': 'a fact', 'project_id': 'dark_factory'}

    with StubMcpServer(stateful=False, payload={'success': True}) as server:
        body = await mcp_call(
            f'{server.url}/mcp', 'tools/call', {'name': 'add_memory', 'arguments': arguments},
        )
        await mcp_call(f'{server.url}/mcp', 'tools/list')
        requests = server.requests

    tool_calls = [r for r in requests if r.body['method'] == 'tools/call']
    assert len(tool_calls) == 1
    assert tool_calls[0].path == '/mcp'
    assert tool_calls[0].body['params'] == {'name': 'add_memory', 'arguments': arguments}
    assert body['result']['structuredContent'] == {'success': True}
    assert [r.body['method'] for r in requests[3:]] == [
        'initialize', 'notifications/initialized', 'tools/list',
    ]


def test_no_session_copy_remains():
    assert orchestrator.mcp_lifecycle.McpSession is shared.mcp_session.McpSession


# ---------------------------------------------------------------------------
# McpLifecycle process-group tests (step-19 / step-20)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMcpLifecycleProcessGroup:
    """McpLifecycle must spawn its subprocess in a fresh process group."""

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> MagicMock:
        """Minimal config-shaped mock for McpLifecycle."""
        cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
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

    async def test_stop_logs_warning_when_terminate_raises(
        self, mock_config: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """stop() must log a warning (not re-raise) when terminate_process_group raises.

        Failing test — current code swallows silently with `except Exception: pass`.
        """
        with patch(
            'orchestrator.mcp_lifecycle.terminate_process_group',
            new_callable=AsyncMock,
            side_effect=RuntimeError('boom'),
        ):
            mcp = McpLifecycle(mock_config)
            mcp._process = MagicMock(returncode=None, pid=12345)
            mcp._pgid = 12345
            with caplog.at_level(logging.WARNING, logger='orchestrator.mcp_lifecycle'):
                # stop() must not re-raise — reaching the next line proves it
                await mcp.stop()

        # Warning must mention the prescribed phrase and the exception message
        assert 'Failed to terminate fused-memory server' in caplog.text
        assert 'boom' in caplog.text
        # State cleaned up regardless
        assert mcp._process is None
        assert mcp._pgid is None

    async def test_stop_falls_back_to_process_pid_when_pgid_missing(
        self, mock_config: MagicMock
    ) -> None:
        """stop() must use _process.pid as fallback pgid when _pgid is None.

        Simulates a crash in start() between create_subprocess_exec and the
        `self._pgid = self._process.pid` assignment, leaving _process set but
        _pgid None. Currently fails because the `if self._pgid is not None:`
        guard short-circuits and terminate_process_group is never called.
        """
        fake_proc = MagicMock(returncode=None, pid=54321)

        with patch(
            'orchestrator.mcp_lifecycle.terminate_process_group',
            new_callable=AsyncMock,
        ) as mock_tpg:
            mcp = McpLifecycle(mock_config)
            mcp._process = fake_proc
            mcp._pgid = None  # crash between spawn and pgid assignment
            await mcp.stop()

        # terminate_process_group must have been called exactly once
        mock_tpg.assert_awaited_once()
        # pgid fallback must be _process.pid
        call_args = mock_tpg.call_args
        assert call_args.args[1] == 54321, (
            f'Expected pgid=54321 (from _process.pid), got {call_args.args[1]}'
        )
        # proc argument must be the process object
        assert call_args.args[0] is fake_proc
        # State cleaned up
        assert mcp._process is None
        assert mcp._pgid is None

