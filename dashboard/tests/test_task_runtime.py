"""Tests for dashboard.data.task_runtime — fan-out of get_task_runtime_state."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot


def _mcp_response(inner: dict, request_id: int = 1) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            'jsonrpc': '2.0',
            'id': request_id,
            'result': {
                'content': [{'type': 'text', 'text': json.dumps(inner)}],
            },
        },
        headers={'mcp-session-id': 'test-session-id'},
    )


def _init_response(request_id: int = 1) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            'jsonrpc': '2.0',
            'id': request_id,
            'result': {
                'protocolVersion': '2025-03-26',
                'capabilities': {'tools': {}},
                'serverInfo': {'name': 'test', 'version': '0.1'},
            },
        },
        headers={'mcp-session-id': 'test-session-id'},
    )


class _PerPortHandler:
    """Mock httpx handler that dispatches per-port behaviour.

    ``responses`` maps port -> TaskRuntimeSnapshot-shaped dict (i.e.
    ``TaskRuntimeSnapshot(...).model_dump(mode='json')``) to return on
    tools/call. ``fail_ports`` raise httpx.ConnectError. ``slow_ports``
    sleep before responding (to drive the timeout path).
    """

    def __init__(
        self,
        responses: dict[int, dict] | None = None,
        *,
        fail_ports: set[int] | None = None,
        slow_ports: dict[int, float] | None = None,
    ):
        self.responses = responses or {}
        self.fail_ports = fail_ports or set()
        self.slow_ports = slow_ports or {}

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        port = request.url.port
        assert port is not None
        if port in self.fail_ports:
            raise httpx.ConnectError('refused')
        if port in self.slow_ports:
            await asyncio.sleep(self.slow_ports[port])
        body = json.loads(request.content)
        method = body.get('method', '')
        request_id = body.get('id', 1)
        if method == 'initialize':
            return _init_response(request_id)
        if method.startswith('notifications/'):
            return httpx.Response(202, headers={'mcp-session-id': 'test-session-id'})
        # tools/call
        inner = self.responses.get(port, TaskRuntimeSnapshot().model_dump(mode='json'))
        return _mcp_response(inner, request_id)


@pytest.fixture(autouse=True)
def _clean_sessions():
    from dashboard.data.memory import reset_sessions
    reset_sessions()
    yield
    reset_sessions()


def _urls(*ports: int) -> dict[str, str]:
    return {f'proj{p}': f'http://127.0.0.1:{p}/mcp' for p in ports}


def _entry_kwargs(task_id: int | str = 1, **overrides) -> dict:
    """Valid TaskRuntimeEntry constructor kwargs, with overrides applied.

    ``task_id`` accepts ``str`` too so callers can deliberately build an
    out-of-contract wire payload (e.g. ``task_id='not-an-int'``) to exercise
    the malformed-payload decode path — this dict is a raw JSON-shaped
    payload, not passed through TaskRuntimeEntry's own validation until the
    code under test decodes it.
    """
    base = dict(
        task_id=task_id,
        has_worktree=True,
        loops=3,
        attempts=1,
        started='2026-07-16T10:00:00+00:00',
        lane='_lane-1',
        phase='EXECUTE',
        lane_state='assigned',
        error=None,
    )
    base.update(overrides)
    return base


class TestFetchTaskRuntime:
    """End-to-end tests for the fan-out helper fetch_task_runtime."""

    async def test_empty_urls_returns_empty(self):
        from dashboard.data.task_runtime import fetch_task_runtime
        async with httpx.AsyncClient() as client:
            assert await fetch_task_runtime(client, {}) == {}

    async def test_all_succeed_decodes_snapshot_and_survives_task_fields(self):
        from dashboard.data.task_runtime import fetch_task_runtime
        snapshot = TaskRuntimeSnapshot(
            tasks=[TaskRuntimeEntry(**_entry_kwargs(task_id=7))],
        )
        handler = _PerPortHandler({8200: snapshot.model_dump(mode='json')})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_task_runtime(client, _urls(8200))

        assert set(result) == {'proj8200'}
        got = result['proj8200']
        assert isinstance(got, TaskRuntimeSnapshot)
        assert got.offline is False
        assert len(got.tasks) == 1
        entry = got.tasks[0]
        assert entry.task_id == 7
        assert entry.loops == 3
        assert entry.attempts == 1
        assert entry.lane == '_lane-1'
        assert entry.phase == 'EXECUTE'
        assert entry.lane_state == 'assigned'

    async def test_connect_error_yields_offline_snapshot_others_unaffected(self):
        from dashboard.data.task_runtime import fetch_task_runtime
        snapshot = TaskRuntimeSnapshot(tasks=[TaskRuntimeEntry(**_entry_kwargs())])
        handler = _PerPortHandler(
            {8100: snapshot.model_dump(mode='json')},
            fail_ports={8102},
        )
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_task_runtime(client, _urls(8100, 8102))

        assert result['proj8100'].offline is False
        assert len(result['proj8100'].tasks) == 1
        assert result['proj8102'] == TaskRuntimeSnapshot(offline=True)

    async def test_timeout_yields_offline_snapshot_for_slow_project_only(self):
        from dashboard.data.task_runtime import fetch_task_runtime
        handler = _PerPortHandler(
            {8100: TaskRuntimeSnapshot().model_dump(mode='json')},
            slow_ports={8105: 0.5},
        )
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_task_runtime(
                client, _urls(8100, 8105), per_call_timeout=0.05,
            )

        assert result['proj8100'].offline is False
        assert result['proj8105'] == TaskRuntimeSnapshot(offline=True)

    async def test_malformed_payload_degrades_only_that_project(self, caplog):
        """An out-of-contract payload (task_id not coercible to int) degrades
        just that project to offline=True, with a WARNING logged; the other
        project still decodes normally."""
        from dashboard.data.task_runtime import fetch_task_runtime
        good = TaskRuntimeSnapshot(tasks=[TaskRuntimeEntry(**_entry_kwargs(task_id=1))])
        bad = {'offline': False, 'tasks': [_entry_kwargs(task_id='not-an-int')]}
        handler = _PerPortHandler({8100: good.model_dump(mode='json'), 8103: bad})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with caplog.at_level('WARNING'):
                result = await fetch_task_runtime(client, _urls(8100, 8103))

        assert result['proj8100'].offline is False
        assert result['proj8103'] == TaskRuntimeSnapshot(offline=True)
        assert any('proj8103' in rec.message or '8103' in rec.message for rec in caplog.records), (
            'expected a WARNING naming the offending project/URL'
        )

    async def test_all_known_projects_present_on_mixed_failures(self):
        from dashboard.data.task_runtime import fetch_task_runtime
        handler = _PerPortHandler(
            {8100: TaskRuntimeSnapshot().model_dump(mode='json')},
            fail_ports={8102, 8105},
        )
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_task_runtime(client, _urls(8100, 8102, 8105))

        assert set(result) == {'proj8100', 'proj8102', 'proj8105'}


class TestProbeOneTimeoutBudget:
    """The probe's budget must reach client.post, not just the wait_for.

    Twin of ``test_merge_halt.TestProbeOneTimeoutBudget`` — this module
    clones merge_halt's fan-out pattern, so it clones its bound too. The
    per-request budget matters because ``client.post(timeout=...)`` also
    governs **pool acquisition** on the shared client; wait_for alone left
    each post able to wait httpx's 10s default for a connection slot.

    AsyncMock rather than MockTransport deliberately: MockTransport never
    surfaces the ``timeout`` kwarg to its handler.
    """

    @staticmethod
    def _cold_session_responses(inner: dict, url: str) -> list[httpx.Response]:
        responses = [
            _init_response(),
            httpx.Response(202, headers={'mcp-session-id': 'test-session-id'}),
            _mcp_response(inner),
        ]
        for resp in responses:
            resp.request = httpx.Request('POST', f'{url.rstrip("/")}/mcp')
        return responses

    async def test_budget_reaches_every_post(self):
        from unittest.mock import AsyncMock

        from dashboard.data.task_runtime import _probe_one

        url = 'http://127.0.0.1:8100'
        mock_client = AsyncMock()
        mock_client.post.side_effect = self._cold_session_responses(
            TaskRuntimeSnapshot().model_dump(mode='json'), url,
        )

        result = await _probe_one(mock_client, url, 0.05)

        assert result.offline is False, f'probe should have succeeded: {result}'
        timeouts = [c.kwargs['timeout'] for c in mock_client.post.call_args_list]
        assert timeouts == [0.05, 0.05, 0.05], (
            f"the probe budget must reach every post, not httpx's 10s "
            f'default, got {timeouts}'
        )
