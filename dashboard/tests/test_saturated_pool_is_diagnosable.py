"""With the shared httpx pool saturated, dashboard MCP failures (fused-memory
fan-outs and the escalation-URL probes) name PoolTimeout.

The user-observable signal: a dashboard failure log for a saturated pool names
``PoolTimeout`` — never a content-free line, never a generic timeout that reads
like a dead endpoint. The fake saturates at the transport layer, so the real
fan-out seam (``first_success``, the shielded exchange, ``safe_gather_result``)
is exercised rather than stubbed.

Each per-request budget is a private module constant with no config seam (the
same constants ``test_app_http_limits.py`` and ``test_metrics_curator.py``
patch), so the dotted paths below are the one place these tests name them.

One site is the deliberate exception: the Tasks-tab snapshot reports a
saturated pool as its own budget running out, so its test pins that rather
than a named ``PoolTimeout``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from datetime import UTC, datetime

import aiosqlite
import httpx
import pytest

from dashboard.config import DashboardConfig
from dashboard.data import escalations, merge_halt, merge_queue, task_runtime, task_snapshot
from dashboard.data import memory as memory_data
from dashboard.data.metrics import METRICS_SCHEMA, collect_metrics_snapshot

_SATURATED_URL = 'http://saturated-pool.test'
_FANOUT_LOGGER = 'dashboard.data.mcp_fanout'
_GATHER_LOGGER = 'dashboard.data.utils'
_METRICS_LOGGER = 'dashboard.data.metrics'
_TASK_RUNTIME_LOGGER = 'dashboard.data.task_runtime'
_ESCALATIONS_LOGGER = 'dashboard.data.escalations'
_CURATOR_BUDGET = 'dashboard.app._CURATOR_ENDPOINT_TIMEOUT_SECONDS'
_MEMORY_BUDGET = 'dashboard.api.memory._MEMORY_ENDPOINT_TIMEOUT_SECONDS'
_SAMPLER_BUDGET = 'dashboard.data.metrics._HTTP_SAMPLER_TIMEOUT_SECONDS'
_SNAPSHOT_BUDGET = 'dashboard.data.task_snapshot.PER_CALL_TIMEOUT'
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


@pytest.fixture()
async def saturated_client():
    """An ``httpx.AsyncClient`` over ``_SaturatedPoolTransport``.

    The MCP session cache is process-global and keyed by URL, so it is reset on
    both sides: no cached session lets a test skip the saturated handshake, and
    none outlives the test.
    """
    memory_data.reset_sessions()
    async with httpx.AsyncClient(transport=_SaturatedPoolTransport()) as client:
        yield client
    memory_data.reset_sessions()


@pytest.fixture()
def fresh_task_runtime_log():
    """Reset task_runtime's process-global WARNING dedupe on both sides."""
    task_runtime.reset_probe_log_state()
    yield
    task_runtime.reset_probe_log_state()


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
    saturated_client, tmp_path, caplog, monkeypatch,
):
    monkeypatch.setattr(_SAMPLER_BUDGET, _FAST_BUDGET_SECONDS)
    monkeypatch.setattr('dashboard.data.metrics.find_running_orchestrators', lambda: [])

    async def _gate_unread(*_args, **_kwargs):
        # The curator sampler's gate read carries mcp_tool_call's 10s default
        # budget, which would stall this test; it is not what is asserted here.
        return {}

    monkeypatch.setattr('dashboard.data.metrics.get_curator_state', _gate_unread)
    config = DashboardConfig(project_root=tmp_path, fused_memory_urls=[_SATURATED_URL])

    conn = await aiosqlite.connect(':memory:')
    try:
        await conn.executescript(METRICS_SCHEMA)
        with caplog.at_level(logging.WARNING):
            await collect_metrics_snapshot(
                conn, config, saturated_client, recon_db=None, merge_dbs=[],
            )
    finally:
        await conn.close()

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


@pytest.mark.asyncio
async def test_task_runtime_probe_names_pool_timeout_in_the_dashboard_fault_domain(
    saturated_client, fresh_task_runtime_log, caplog,
):
    with caplog.at_level(logging.WARNING):
        result = await task_runtime.fetch_task_runtime(
            saturated_client,
            {'saturated': f'{_SATURATED_URL}/mcp'},
            per_call_timeout=_FAST_BUDGET_SECONDS,
        )

    assert result['saturated'].offline_reason == 'deadline_exceeded', (
        "a saturated local pool is the DASHBOARD's fault domain (task 3517), "
        f"never the orchestrator's; got {result['saturated'].offline_reason!r}"
    )
    probe = _warnings(caplog, _TASK_RUNTIME_LOGGER)
    assert any('deadline_exceeded' in m and 'PoolTimeout' in m for m in probe), (
        f'no task_runtime WARNING named PoolTimeout; got {probe}'
    )


@pytest.mark.asyncio
async def test_merge_halt_probe_names_pool_timeout(saturated_client):
    result = await merge_halt.get_merge_halt_status(
        saturated_client,
        {'saturated': f'{_SATURATED_URL}/mcp'},
        per_call_timeout=_FAST_BUDGET_SECONDS,
    )

    entry = result['saturated']
    assert entry['offline'] is True
    assert 'PoolTimeout' in entry['error'], (
        f'the halt probe error must name pool saturation; got {entry["error"]!r}'
    )


@pytest.mark.asyncio
async def test_live_merge_queue_probe_names_pool_timeout(saturated_client):
    result = await merge_queue.fetch_live_merge_queues(
        saturated_client,
        {'saturated': f'{_SATURATED_URL}/mcp'},
        per_call_timeout=_FAST_BUDGET_SECONDS,
    )

    live = result['saturated']
    assert live['reachable'] is False
    assert 'PoolTimeout' in live['error'], (
        "this string becomes the 'In queue now' unknown reason; got "
        f'{live["error"]!r}'
    )


@pytest.mark.asyncio
async def test_pins_probe_names_pool_timeout(saturated_client, caplog):
    with caplog.at_level(logging.DEBUG, logger=_ESCALATIONS_LOGGER):
        result = await escalations.fetch_pins_recovery(
            saturated_client,
            {'saturated': f'{_SATURATED_URL}/mcp'},
            per_call_timeout=_FAST_BUDGET_SECONDS,
        )

    assert result == {'saturated': None}, 'an unread project is UNKNOWN, never an empty map'
    # This probe's only diagnostic is its DEBUG line, so that is what is read.
    lines = [r.getMessage() for r in caplog.records if r.name == _ESCALATIONS_LOGGER]
    assert any(
        'get_pending_escalations failed' in m and 'PoolTimeout' in m for m in lines
    ), f'no pins-probe line named PoolTimeout; got {lines}'


@pytest.mark.asyncio
async def test_task_snapshot_reports_a_saturated_pool_as_budget_not_offline(
    saturated_client, tmp_path, monkeypatch,
):
    """A saturated local pool is this process running out of budget, so the
    Tasks tab must show it degraded, never offline. Converting
    ``task_snapshot._bounded`` to ``enclosing_deadline`` would let
    ``first_success`` fold the PoolTimeout into an offline marker and flip this
    to UNREACHABLE.

    The two timers are deliberately EQUAL, as in production: the fake's pool
    wait reads the same patched ``PER_CALL_TIMEOUT`` through the per-request
    budget. The outcome is BUDGET only because ``_bounded``'s ``wait_for`` arms
    its deadline before the request reaches the pool. That ordering is
    load-bearing: arming the per-request timer first, or moving the per-request
    budget off ``PER_CALL_TIMEOUT``, would also flip this result.
    """
    monkeypatch.setattr(_SNAPSHOT_BUDGET, _FAST_BUDGET_SECONDS)
    config = DashboardConfig(project_root=tmp_path, fused_memory_urls=[_SATURATED_URL])

    snapshot = await task_snapshot.acquire_snapshot(
        saturated_client, config, tmp_path, now=datetime.now(UTC),
    )

    assert snapshot.failure is task_snapshot.SnapshotFailure.BUDGET, (
        f'a saturated pool must read as BUDGET (degraded); got {snapshot.failure!r}'
    )
