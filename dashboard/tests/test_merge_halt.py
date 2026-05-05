"""Tests for dashboard.data.merge_halt — fan-out of get_merge_halt_status."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest


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

    ``responses`` maps port → halt-status dict to return on tools/call.
    ``fail_ports`` raise httpx.ConnectError.
    ``slow_ports`` sleep before responding (to drive the timeout path).
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
        inner = self.responses.get(port, {'wired': False, 'halted': False})
        return _mcp_response(inner, request_id)


@pytest.fixture(autouse=True)
def _clean_sessions():
    from dashboard.data.memory import reset_sessions
    reset_sessions()
    yield
    reset_sessions()


def _urls(*ports: int) -> dict[str, str]:
    return {f'proj{p}': f'http://127.0.0.1:{p}/mcp' for p in ports}


class TestGetMergeHaltStatus:
    """End-to-end tests for the fan-out helper."""

    async def test_empty_urls_returns_empty(self):
        from dashboard.data.merge_halt import get_merge_halt_status
        async with httpx.AsyncClient() as client:
            assert await get_merge_halt_status(client, {}) == {}

    async def test_all_succeed_returns_per_project_status(self):
        from dashboard.data.merge_halt import get_merge_halt_status
        handler = _PerPortHandler({
            8100: {'wired': True, 'halted': False, 'owner_esc_id': None},
            8105: {'wired': True, 'halted': True, 'owner_esc_id': 'esc-7'},
        })
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_merge_halt_status(client, _urls(8100, 8105))
        assert set(result) == {'proj8100', 'proj8105'}
        assert result['proj8100']['halted'] is False
        assert result['proj8100']['wired'] is True
        assert result['proj8100']['offline'] is False
        assert result['proj8105']['halted'] is True
        assert result['proj8105']['owner_esc_id'] == 'esc-7'

    async def test_connect_error_yields_offline_entry(self):
        from dashboard.data.merge_halt import get_merge_halt_status
        handler = _PerPortHandler(
            {8100: {'wired': True, 'halted': False}},
            fail_ports={8102},
        )
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_merge_halt_status(client, _urls(8100, 8102))
        assert result['proj8100']['wired'] is True
        assert result['proj8100']['offline'] is False
        assert result['proj8102']['offline'] is True
        assert result['proj8102']['wired'] is False
        assert result['proj8102']['halted'] is False
        assert 'error' in result['proj8102']

    async def test_timeout_yields_offline_entry(self):
        from dashboard.data.merge_halt import get_merge_halt_status
        handler = _PerPortHandler(
            {8100: {'wired': True, 'halted': False}},
            slow_ports={8105: 0.5},
        )
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_merge_halt_status(
                client, _urls(8100, 8105), per_call_timeout=0.05,
            )
        assert result['proj8100']['offline'] is False
        assert result['proj8105']['offline'] is True
        assert 'error' in result['proj8105']

    async def test_all_known_projects_present_on_mixed_failures(self):
        from dashboard.data.merge_halt import get_merge_halt_status
        handler = _PerPortHandler(
            {8100: {'wired': True, 'halted': False}},
            fail_ports={8102, 8105},
        )
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await get_merge_halt_status(client, _urls(8100, 8102, 8105))
        # All projects must appear regardless of per-call failure.
        assert set(result) == {'proj8100', 'proj8102', 'proj8105'}
