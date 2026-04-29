"""Integration tests for the redux dashboard app.

These tests focus on the route-level contract: the SPA index, the static
asset mount, and the JSON API under ``/api/v2/dashboard/*``.  Most endpoints
read from per-project DBs that don't exist in the temp fixture, so they are
expected to return empty-but-well-formed JSON — the assertion is on the
response *shape*, not the contents, which exercises the shape adapters.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from dashboard.app import _parse_window


# ---------------------------------------------------------------------------
# _parse_window helper
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in for FastAPI's Request.query_params used by _parse_window."""

    def __init__(self, **params):
        self.query_params = params


@pytest.mark.parametrize('value, expected', [
    ('24h', 1),
    ('7d', 7),
    ('30d', 30),
    ('all', 3650),
    ('weird', 30),  # default
    (None, 30),
])
def test_parse_window_known_and_unknown(value, expected):
    request = _FakeRequest()
    if value is not None:
        request.query_params['window'] = value
    assert _parse_window(request) == expected


# ---------------------------------------------------------------------------
# SPA + health
# ---------------------------------------------------------------------------


def test_index_serves_redux_html(client):
    resp = client.get('/')
    assert resp.status_code == 200
    body = resp.text
    assert '<div id="root">' in body
    assert '/static/redux/app.jsx' in body


def test_static_redux_assets_load(client):
    for name in ('app.jsx', 'data.js', 'shell.jsx', 'tabs.jsx', 'styles.css'):
        resp = client.get(f'/static/redux/{name}')
        assert resp.status_code == 200, f'expected 200 for /static/redux/{name}'


def test_health_endpoint(client):
    resp = client.get('/api/health')
    assert resp.status_code == 200
    assert resp.json() == {'status': 'ok'}


# ---------------------------------------------------------------------------
# JSON API: shape contracts
# ---------------------------------------------------------------------------


def test_orchestrators_returns_orchestrators_and_projects(client):
    """Even with no running orchestrators the response carries both keys."""
    with patch(
        'dashboard.app.discover_orchestrators', return_value=[],
    ):
        resp = client.get('/api/v2/dashboard/orchestrators')
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {'ORCHESTRATORS', 'PROJECTS', 'ORCHESTRATORS_SPARK'}
    assert isinstance(body['ORCHESTRATORS'], list)
    assert isinstance(body['PROJECTS'], list)
    assert isinstance(body['ORCHESTRATORS_SPARK'], dict)
    assert 'labels' in body['ORCHESTRATORS_SPARK']
    assert 'values' in body['ORCHESTRATORS_SPARK']


def test_tasks_returns_active_tasks_and_file_locks(client):
    with patch(
        'dashboard.app.collect_active_tasks', return_value=([], {}),
    ):
        resp = client.get('/api/v2/dashboard/tasks')
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {'ACTIVE_TASKS', 'FILE_LOCKS'}
    assert isinstance(body['ACTIVE_TASKS'], list)
    assert isinstance(body['FILE_LOCKS'], dict)


def test_memory_returns_memory_status(client):
    """memory endpoint composes status + queue stats into a MEMORY_STATUS block."""
    with patch(
        'dashboard.data.memory.get_memory_status',
        new=AsyncMock(return_value={'offline': True, 'error': 'no fused-memory'}),
    ), patch(
        'dashboard.data.memory.get_queue_stats',
        new=AsyncMock(return_value={'counts': {}, 'oldest_pending_age_seconds': None}),
    ):
        resp = client.get('/api/v2/dashboard/memory')
    assert resp.status_code == 200
    body = resp.json()
    assert 'MEMORY_STATUS' in body
    ms = body['MEMORY_STATUS']
    for key in ('graphiti', 'mem0', 'taskmaster', 'queue'):
        assert key in ms


def test_memory_graphs_returns_timeseries_and_breakdown(client):
    resp = client.get('/api/v2/dashboard/memory-graphs')
    assert resp.status_code == 200
    body = resp.json()
    assert {'MEMORY_TIMESERIES', 'MEMORY_OPS_BREAKDOWN'} <= set(body)
    ts = body['MEMORY_TIMESERIES']
    assert {'labels', 'reads', 'writes'} <= set(ts)
    assert isinstance(body['MEMORY_OPS_BREAKDOWN'], list)


def test_recon_returns_recon_state_and_agents(client):
    resp = client.get('/api/v2/dashboard/recon')
    assert resp.status_code == 200
    body = resp.json()
    assert {'RECON_STATE', 'AGENTS'} <= set(body)
    rs = body['RECON_STATE']
    for key in ('buffer', 'burst_state', 'watermarks', 'verdict', 'runs'):
        assert key in rs


def test_merge_queue_returns_merge_queue(client):
    resp = client.get('/api/v2/dashboard/merge-queue')
    assert resp.status_code == 200
    body = resp.json()
    assert 'MERGE_QUEUE' in body
    assert isinstance(body['MERGE_QUEUE'], dict)


def test_costs_returns_full_costs_block(client):
    resp = client.get('/api/v2/dashboard/costs?window=7d')
    assert resp.status_code == 200
    body = resp.json()
    assert 'COSTS' in body
    costs = body['COSTS']
    for key in ('summary', 'by_project', 'by_account', 'by_role', 'trend', 'events'):
        assert key in costs, f'COSTS missing {key}'


def test_performance_returns_performance(client):
    resp = client.get('/api/v2/dashboard/performance')
    assert resp.status_code == 200
    body = resp.json()
    assert 'PERFORMANCE' in body
    assert isinstance(body['PERFORMANCE'], dict)


def test_burndown_returns_aggregate_and_per_project(client):
    resp = client.get('/api/v2/dashboard/burndown?window=30d')
    assert resp.status_code == 200
    body = resp.json()
    assert {'BURNDOWN', 'BURNDOWN_BY_PROJECT'} <= set(body)
    aggregate = body['BURNDOWN']
    assert {'labels', 'done', 'in_progress', 'blocked', 'pending'} <= set(aggregate)
