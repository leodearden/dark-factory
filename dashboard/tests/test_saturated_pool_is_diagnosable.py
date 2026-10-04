"""With the shared httpx pool saturated, fused-memory failures name PoolTimeout.

The user-observable signal: a dashboard failure log for a saturated pool names
``PoolTimeout`` — never a content-free line, never a generic timeout that reads
like a dead endpoint. The fake saturates at the transport layer, so the real
fan-out seam (``first_success``, the shielded exchange, ``safe_gather_result``)
is exercised rather than stubbed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

import httpx
import pytest

from dashboard.data import memory as memory_data

_SATURATED_URL = 'http://saturated-pool.test'
_FANOUT_LOGGER = 'dashboard.data.mcp_fanout'
_GATHER_LOGGER = 'dashboard.data.utils'


class _SaturatedPoolTransport(httpx.AsyncBaseTransport):
    """httpcore on a full pool: wait out THIS request's pool budget, then raise.

    The message is empty because that is how httpcore raises it.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(request.extensions['timeout']['pool'])
        raise httpx.PoolTimeout('')


@pytest.fixture()
def saturated_app(client, monkeypatch):
    """The isolated-lifespan TestClient, pointed at a saturated pool.

    The URL is unique to this module so no background sampler shares a
    failure-streak key with it and demotes the asserted WARNING to DEBUG.
    """
    memory_data.reset_sessions()
    monkeypatch.setattr(
        client.app.state,
        'config',
        dataclasses.replace(client.app.state.config, fused_memory_urls=[_SATURATED_URL]),
    )
    monkeypatch.setattr(
        client.app.state,
        'http_client',
        httpx.AsyncClient(transport=_SaturatedPoolTransport()),
    )
    yield client
    memory_data.reset_sessions()


def _warnings(caplog: pytest.LogCaptureFixture, logger_name: str) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == logger_name and r.levelno == logging.WARNING
    ]


def test_curator_state_leg_names_pool_timeout(saturated_app, caplog, monkeypatch):
    monkeypatch.setattr('dashboard.app._CURATOR_ENDPOINT_TIMEOUT_SECONDS', 0.05)

    with caplog.at_level(logging.WARNING):
        resp = saturated_app.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    fanout = _warnings(caplog, _FANOUT_LOGGER)
    assert any('get_curator_state' in m and 'PoolTimeout' in m for m in fanout), (
        f'no fan-out WARNING named PoolTimeout for get_curator_state; got {fanout}'
    )
    gather = _warnings(caplog, _GATHER_LOGGER)
    assert not [m for m in gather if m.startswith('Error gathering curator/state')], (
        f'the leg hit its outer deadline instead of degrading through the '
        f"fan-out's own offline sentinel; got {gather}"
    )


def test_memory_route_names_pool_timeout(saturated_app, caplog, monkeypatch):
    monkeypatch.setattr('dashboard.api.memory._MEMORY_ENDPOINT_TIMEOUT_SECONDS', 0.05)

    with caplog.at_level(logging.WARNING):
        resp = saturated_app.get('/api/v2/dashboard/memory')

    assert resp.status_code == 200
    fanout = _warnings(caplog, _FANOUT_LOGGER)
    assert any('get_status' in m and 'PoolTimeout' in m for m in fanout), (
        f'no fan-out WARNING named PoolTimeout for get_status; got {fanout}'
    )
