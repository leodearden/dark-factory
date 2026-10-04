"""With the shared httpx pool saturated, fused-memory failures name PoolTimeout.

The user-observable signal: a dashboard failure log for a saturated pool names
``PoolTimeout`` — never a content-free line, never a generic timeout that reads
like a dead endpoint. The fake saturates at the transport layer, so the real
fan-out seam (``first_success``, the shielded exchange, ``safe_gather_result``)
is exercised rather than stubbed.

Each per-request budget is a private module constant with no config seam (the
same constants ``test_app_http_limits.py`` and ``test_metrics_curator.py``
patch), so the dotted paths below are the one place these tests name them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

import aiosqlite
import httpx
import pytest

from dashboard.config import DashboardConfig
from dashboard.data import memory as memory_data
from dashboard.data.metrics import METRICS_SCHEMA, collect_metrics_snapshot

_SATURATED_URL = 'http://saturated-pool.test'
_FANOUT_LOGGER = 'dashboard.data.mcp_fanout'
_GATHER_LOGGER = 'dashboard.data.utils'
_METRICS_LOGGER = 'dashboard.data.metrics'
_CURATOR_BUDGET = 'dashboard.app._CURATOR_ENDPOINT_TIMEOUT_SECONDS'
_MEMORY_BUDGET = 'dashboard.api.memory._MEMORY_ENDPOINT_TIMEOUT_SECONDS'
_SAMPLER_BUDGET = 'dashboard.data.metrics._HTTP_SAMPLER_TIMEOUT_SECONDS'
_FAST_BUDGET_SECONDS = 0.05


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


def test_curator_legs_name_pool_timeout(saturated_app, caplog, monkeypatch):
    monkeypatch.setattr(_CURATOR_BUDGET, _FAST_BUDGET_SECONDS)

    with caplog.at_level(logging.WARNING):
        resp = saturated_app.get('/api/v2/dashboard/curator')

    assert resp.status_code == 200
    fanout = _warnings(caplog, _FANOUT_LOGGER)

    assert any('get_curator_state' in m and 'PoolTimeout' in m for m in fanout), (
        f'no fan-out WARNING named PoolTimeout for get_curator_state; got {fanout}'
    )
    gather = _warnings(caplog, _GATHER_LOGGER)
    assert not [m for m in gather if m.startswith('Error gathering curator/state')], (
        f'the state leg hit its outer deadline instead of degrading through the '
        f"fan-out's own offline sentinel; got {gather}"
    )

    list_tickets = [m for m in fanout if 'list_tickets' in m]
    assert list_tickets, 'no fan-out WARNING was logged for list_tickets'
    assert all('PoolTimeout' in m for m in list_tickets), (
        f'pool saturation must be named, not reported as a timeout; got {list_tickets}'
    )
    assert not [m for m in list_tickets if 'timed out' in m], (
        f'pool saturation was mislabelled as an endpoint timeout; got {list_tickets}'
    )


def test_memory_route_names_pool_timeout(saturated_app, caplog, monkeypatch):
    monkeypatch.setattr(_MEMORY_BUDGET, _FAST_BUDGET_SECONDS)

    with caplog.at_level(logging.WARNING):
        resp = saturated_app.get('/api/v2/dashboard/memory')

    assert resp.status_code == 200
    fanout = _warnings(caplog, _FANOUT_LOGGER)
    assert any('get_status' in m and 'PoolTimeout' in m for m in fanout), (
        f'no fan-out WARNING named PoolTimeout for get_status; got {fanout}'
    )


@pytest.mark.asyncio
async def test_metrics_memory_and_queue_samplers_name_pool_timeout(
    tmp_path, caplog, monkeypatch,
):
    monkeypatch.setattr(_SAMPLER_BUDGET, _FAST_BUDGET_SECONDS)
    monkeypatch.setattr('dashboard.data.metrics.find_running_orchestrators', lambda: [])

    async def _gate_unread(*_args, **_kwargs):
        # The curator sampler's gate read carries mcp_tool_call's 10s default
        # budget, which would stall this test; it is not what is asserted here.
        return {}

    monkeypatch.setattr('dashboard.data.metrics.get_curator_state', _gate_unread)
    config = DashboardConfig(project_root=tmp_path, fused_memory_urls=[_SATURATED_URL])

    memory_data.reset_sessions()
    conn = await aiosqlite.connect(':memory:')
    try:
        await conn.executescript(METRICS_SCHEMA)
        async with httpx.AsyncClient(transport=_SaturatedPoolTransport()) as http_client:
            with caplog.at_level(logging.WARNING):
                await collect_metrics_snapshot(
                    conn, config, http_client, recon_db=None, merge_dbs=[],
                )
    finally:
        await conn.close()
        memory_data.reset_sessions()

    fanout = _warnings(caplog, _FANOUT_LOGGER)
    for tool in ('get_status', 'get_queue_stats'):
        assert any(m.startswith(f'{tool} failed') and 'PoolTimeout' in m for m in fanout), (
            f'no fan-out WARNING named PoolTimeout for {tool}; got {fanout}'
        )
    sampler_failures = [
        m for m in _warnings(caplog, _METRICS_LOGGER)
        if m in ('memory sampler failed', 'queue sampler failed')
    ]
    assert not sampler_failures, (
        f'a sampler hit its outer deadline instead of degrading through the '
        f"fan-out's own offline sentinel; got {sampler_failures}"
    )
