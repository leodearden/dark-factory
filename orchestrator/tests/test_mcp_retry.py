"""Tests for MCP session retry logic in mcp_lifecycle.py."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.fm_retry import fm_retry_backoffs
from orchestrator.mcp_lifecycle import (
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
        fixed = [1.0, 2.0]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch(
                'orchestrator.mcp_lifecycle.fm_retry_backoffs',
                return_value=fixed,
            ),
            patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await session._raw_call('initialize')

        assert result['result'] == {'ok': True}
        assert mock_client.post.call_count == 3
        # Session ID should be cleared on retry then set from response
        assert session._session_id == 'new-session'
        # Backoff: sleep(fixed[0]), sleep(fixed[1]) — from the shared schedule
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0].args[0] == pytest.approx(fixed[0])
        assert mock_sleep.call_args_list[1].args[0] == pytest.approx(fixed[1])

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
        fixed = [0.01, 0.01]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch(
                'orchestrator.mcp_lifecycle.fm_retry_backoffs',
                return_value=fixed,
            ),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            # Exhausted retries now raise RuntimeError carrying the original
            # exception type + message so downstream loggers produce something
            # diagnostic even when str(exc) is empty (e.g. httpx.ReadTimeout).
            with pytest.raises(RuntimeError) as excinfo:
                await session._raw_call('tools/call', {'name': 't', 'arguments': {}})

        assert mock_client.post.call_count == len(fixed) + 1
        assert 'ConnectError' in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, httpx.ConnectError)

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

    @pytest.mark.asyncio
    async def test_retries_on_read_timeout(self):
        """Motivating fix (2026-04-20): httpx.ReadTimeout is now in the
        retry except clause. Before this fix, a ReadTimeout propagated
        directly out of _raw_call with empty str(exc), producing empty
        'Failed to set task X status to Y:' error lines in orchestrator
        logs when fused-memory was briefly slow (e.g. curator LLM call
        inside add_task held the project lock for 30s).
        """
        session = McpSession('http://localhost:8002')
        ok = _make_response()
        mock_client = AsyncMock()
        # First two attempts time out; third succeeds.
        mock_client.post.side_effect = [
            httpx.ReadTimeout(''),
            httpx.ReadTimeout(''),
            ok,
        ]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await session._raw_call(
                'tools/call', {'name': 't', 'arguments': {}},
            )

        assert mock_client.post.call_count == 3
        assert result['result']['ok'] is True

    @pytest.mark.asyncio
    async def test_retries_on_oserror(self):
        """Raw OSError (e.g. FileNotFoundError) surfaces when httpx's
        map_httpcore_exceptions() doesn't rewrap a lower-level failure.

        Reproduces 2026-04-20 orchestrator hang where the scheduler logged
        "Failed to fetch tasks: [Errno 2] No such file or directory" every
        15s for >1.5h with zero outgoing HTTP requests — the exception was
        escaping the retry loop because OSError wasn't in the except set.
        Now OSError is retryable.
        """
        session = McpSession('http://localhost:8002')
        ok = _make_response()
        mock_client = AsyncMock()
        # First two attempts raise FileNotFoundError (subclass of OSError);
        # third succeeds.
        mock_client.post.side_effect = [
            FileNotFoundError(2, 'No such file or directory'),
            FileNotFoundError(2, 'No such file or directory'),
            ok,
        ]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await session._raw_call(
                'tools/call', {'name': 't', 'arguments': {}},
            )

        assert mock_client.post.call_count == 3
        assert result['result']['ok'] is True

    @pytest.mark.asyncio
    async def test_exhausted_oserror_wrapped_in_runtimeerror(self):
        """FileNotFoundError's str() is '[Errno 2] No such file or directory'
        with no filename; the wrapped RuntimeError must carry the class name
        so log lines are not indistinguishable from a real file-open error.
        """
        session = McpSession('http://localhost:8002')
        mock_client = AsyncMock()
        mock_client.post.side_effect = FileNotFoundError(
            2, 'No such file or directory',
        )
        fixed = [0.01, 0.01]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch(
                'orchestrator.mcp_lifecycle.fm_retry_backoffs',
                return_value=fixed,
            ),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError) as excinfo:
                await session._raw_call(
                    'tools/call', {'name': 't', 'arguments': {}},
                )

        assert mock_client.post.call_count == len(fixed) + 1
        assert 'FileNotFoundError' in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, FileNotFoundError)

    @pytest.mark.asyncio
    async def test_exhausted_read_timeout_wrapped_in_runtimeerror(self):
        """When retries exhaust on ReadTimeout (empty str), the wrapped
        RuntimeError MUST carry the exception type name so log lines
        produce diagnostic output instead of an empty string."""
        session = McpSession('http://localhost:8002')
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ReadTimeout('')
        fixed = [0.01, 0.01]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch(
                'orchestrator.mcp_lifecycle.fm_retry_backoffs',
                return_value=fixed,
            ),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError) as excinfo:
                await session._raw_call(
                    'tools/call', {'name': 't', 'arguments': {}},
                )

        assert mock_client.post.call_count == len(fixed) + 1
        # The wrapped message must contain the type name so that downstream
        # loggers produce something useful even when str(exc) is empty.
        assert 'ReadTimeout' in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, httpx.ReadTimeout)


class TestClientOpIdInjection:
    """_raw_call injects a per-logical-call client_op_id for mutating task
    tools so the transport-level retry loop is safe (task 2712).

    The id is generated ONCE before the retry loop, so every transport retry
    of the same logical call reuses the same key — a retried write dedupes
    server-side instead of double-applying.
    """

    @pytest.mark.asyncio
    async def test_injects_client_op_id_for_mutating_tool(self):
        session = McpSession('http://localhost:8002')
        ok = _make_response()
        mock_client = AsyncMock()
        mock_client.post.return_value = ok

        with patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await session._raw_call(
                'tools/call', {'name': 'update_task', 'arguments': {'id': '1'}},
            )

        posted = mock_client.post.call_args.kwargs['json']
        op_id = posted['params']['arguments'].get('client_op_id')
        assert isinstance(op_id, str) and op_id, (
            'a mutating tool call must get a non-empty client_op_id injected'
        )

    @pytest.mark.asyncio
    async def test_same_client_op_id_across_transport_retries(self):
        """The injected key is stable across all transport retries of one
        logical call — generated once BEFORE the retry loop.

        The payload dict is reused by reference across retries, so we snapshot
        (deep-copy) the posted body at each post() to compare the key values
        as they were actually sent, not their final in-place state.
        """
        session = McpSession('http://localhost:8002')
        ok = _make_response()
        posted_bodies: list[dict] = []

        async def capture_post(*args, **kwargs):
            posted_bodies.append(copy.deepcopy(kwargs['json']))
            if len(posted_bodies) == 1:
                raise httpx.ConnectError('x')
            return ok

        mock_client = AsyncMock()
        mock_client.post.side_effect = capture_post

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch(
                'orchestrator.mcp_lifecycle.fm_retry_backoffs',
                return_value=[0.0],
            ),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await session._raw_call(
                'tools/call', {'name': 'update_task', 'arguments': {'id': '1'}},
            )

        assert len(posted_bodies) == 2
        op_ids = [
            b['params']['arguments']['client_op_id'] for b in posted_bodies
        ]
        assert op_ids[0] and op_ids[0] == op_ids[1], (
            'the same client_op_id must be posted on every transport retry'
        )

    @pytest.mark.asyncio
    async def test_no_injection_for_read_tool(self):
        session = McpSession('http://localhost:8002')
        ok = _make_response()
        mock_client = AsyncMock()
        mock_client.post.return_value = ok

        with patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await session._raw_call(
                'tools/call', {'name': 'get_tasks', 'arguments': {}},
            )

        posted = mock_client.post.call_args.kwargs['json']
        assert 'client_op_id' not in posted['params']['arguments'], (
            'read tools must not get a client_op_id injected'
        )

    @pytest.mark.asyncio
    async def test_preserves_caller_supplied_client_op_id(self):
        session = McpSession('http://localhost:8002')
        ok = _make_response()
        mock_client = AsyncMock()
        mock_client.post.return_value = ok

        with patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await session._raw_call(
                'tools/call',
                {
                    'name': 'update_task',
                    'arguments': {'id': '1', 'client_op_id': 'caller-set'},
                },
            )

        posted = mock_client.post.call_args.kwargs['json']
        assert posted['params']['arguments']['client_op_id'] == 'caller-set', (
            'a caller-supplied client_op_id must be preserved, not overwritten'
        )


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


class TestSharedFmRetrySchedule:
    """_raw_call/_raw_notify consume the shared orchestrator.fm_retry schedule
    (task 2706) instead of the old, too-small _MCP_MAX_RETRIES/_MCP_BACKOFF_BASE
    budget.
    """

    @pytest.mark.asyncio
    async def test_raw_call_consumes_shared_schedule_until_recovery(self):
        session = McpSession('http://localhost:8002')
        ok = _make_response(session_id='s1')
        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            httpx.ConnectError('refused'),
            httpx.ConnectError('refused'),
            httpx.ConnectError('refused'),
            httpx.ConnectError('refused'),
            ok,
        ]
        fixed = [0.5, 1.5, 3.0, 6.0]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch(
                'orchestrator.mcp_lifecycle.fm_retry_backoffs',
                return_value=fixed,
            ),
            patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await session._raw_call(
                'tools/call', {'name': 't', 'arguments': {}},
            )

        assert result['result'] == {'ok': True}
        assert mock_client.post.call_count == 5
        assert [c.args[0] for c in mock_sleep.call_args_list] == fixed

    @pytest.mark.asyncio
    async def test_raw_call_exhausts_shared_schedule_and_wraps_runtimeerror(self):
        session = McpSession('http://localhost:8002')
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError('refused')
        fixed = [0.5, 1.5, 3.0, 6.0]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch(
                'orchestrator.mcp_lifecycle.fm_retry_backoffs',
                return_value=fixed,
            ),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError) as excinfo:
                await session._raw_call(
                    'tools/call', {'name': 't', 'arguments': {}},
                )

        assert mock_client.post.call_count == 5
        assert '5 attempts' in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_raw_notify_consumes_shared_schedule(self):
        session = McpSession('http://localhost:8002')
        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            httpx.ConnectError('refused'),
            httpx.ConnectError('refused'),
            httpx.ConnectError('refused'),
            httpx.ConnectError('refused'),
            _make_response(status_code=202),
        ]
        fixed = [0.5, 1.5, 3.0, 6.0]

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch(
                'orchestrator.mcp_lifecycle.fm_retry_backoffs',
                return_value=fixed,
            ),
            patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await session._raw_notify('notifications/initialized')

        assert mock_client.post.call_count == 5
        assert [c.args[0] for c in mock_sleep.call_args_list] == fixed

    @pytest.mark.asyncio
    async def test_default_path_uses_shared_schedule_and_exceeds_old_budget(self):
        """With fm_retry_backoffs left unpatched (real import wiring), a
        never-recovering ConnectError must exhaust len(fm_retry_backoffs())+1
        attempts — more than the old _MCP_MAX_RETRIES=3 — proving _raw_call
        consumes the shared schedule rather than a hardcoded retry count.

        random.uniform is pinned to the max-draw boundary so the test's own
        fm_retry_backoffs() call and the SUT's internal (unpatched) call are
        guaranteed to agree — both are pure functions of the same entropy
        source (see test_fm_retry.py's purity test).
        """
        session = McpSession('http://localhost:8002')
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError('refused')

        with (
            patch('orchestrator.mcp_lifecycle.httpx.AsyncClient') as mock_cls,
            patch('asyncio.sleep', new_callable=AsyncMock),
            patch('random.uniform', side_effect=lambda lo, hi: hi),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            expected_attempts = len(fm_retry_backoffs(rng=lambda lo, hi: hi)) + 1
            with pytest.raises(RuntimeError):
                await session._raw_call(
                    'tools/call', {'name': 't', 'arguments': {}},
                )

        assert expected_attempts > 3
        assert mock_client.post.call_count == expected_attempts


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

