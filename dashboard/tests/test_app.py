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


@pytest.mark.parametrize('value, expected', [
    ('24h', 1),
    ('7d', 7),
    ('30d', 30),
    ('all', 3650),
    ('weird', 30),  # default
    (None, 30),
])
def test_parse_window_known_and_unknown(value, expected):
    query_params: dict[str, str] = {}
    if value is not None:
        query_params['window'] = value
    assert _parse_window(query_params) == expected


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
        'dashboard.app.discover_orchestrators',
        new=AsyncMock(return_value=[]),
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


def test_tasks_endpoint_omits_file_locks_and_returns_active_only(client):
    with patch(
        'dashboard.app.collect_active_tasks',
        new=AsyncMock(return_value=([], [])),
    ):
        resp = client.get('/api/v2/dashboard/tasks')
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {'ACTIVE_TASKS', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS'}
    assert 'FILE_LOCKS' not in body
    assert isinstance(body['ACTIVE_TASKS'], list)
    assert body['TASKS_OFFLINE'] is False
    assert body['TASKS_OFFLINE_PROJECTS'] == []


def test_tasks_surfaces_offline_marker_when_mcp_unreachable(client):
    """When collect_active_tasks reports offline projects, the payload sets ``offline=True``."""
    with patch(
        'dashboard.app.collect_active_tasks',
        new=AsyncMock(return_value=([], ['dark-factory'])),
    ):
        resp = client.get('/api/v2/dashboard/tasks')
    assert resp.status_code == 200
    body = resp.json()
    assert body['TASKS_OFFLINE'] is True
    assert body['TASKS_OFFLINE_PROJECTS'] == ['dark-factory']


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


# ---------------------------------------------------------------------------
# Escalations endpoint
# ---------------------------------------------------------------------------

_EMPTY_SUMMARY = {
    'by_level': {0: 0, 1: 0, 2: 0},
    'by_status': {'pending': 0, 'resolved': 0, 'dismissed': 0},
}

_EMPTY_QUEUES = {
    'subsections': [],
    'summary': _EMPTY_SUMMARY,
}


def test_escalations_endpoint_returns_escalations_block(client):
    """GET /api/v2/dashboard/escalations returns 200 with ESCALATIONS key."""
    with patch(
        'dashboard.app.build_escalation_queues',
        return_value=_EMPTY_QUEUES,
    ), patch(
        'dashboard.app.fetch_tasks',
        new=AsyncMock(return_value=[]),
    ):
        resp = client.get('/api/v2/dashboard/escalations')
    assert resp.status_code == 200
    body = resp.json()
    assert 'ESCALATIONS' in body
    esc = body['ESCALATIONS']
    assert 'subsections' in esc
    assert 'summary' in esc
    assert esc['subsections'] == []


def test_escalations_endpoint_attaches_task_cards_and_resolves_recon(client, tmp_path):
    """Full endpoint→shaper integration: task attachment + reconciliation resolution."""
    from dashboard.app import _task_cards_cache_clear
    _task_cards_cache_clear()

    proj_a = tmp_path / 'projA'
    task_dict = {
        'id': 11, 'title': 'wired', 'description': '', 'details': '',
        'status': 'pending', 'priority': 'med', 'dependencies': [], 'metadata': {},
    }
    sub_summary = {
        'by_level': {0: 1, 1: 1, 2: 0},
        'by_status': {'pending': 2, 'resolved': 0, 'dismissed': 0},
    }
    queues = {
        'subsections': [
            {
                'id': str(proj_a),
                'label': 'projA',
                'kind': 'orchestrator',
                'escalations': [{'id': 'e1', 'task_id': 11, 'level': 0, 'status': 'pending', 'summary': 'oops'}],
                'summary': sub_summary,
            },
            {
                'id': 'reconciliation',
                'label': 'fused-memory',
                'kind': 'reconciliation',
                'escalations': [{
                    'id': 'er1', 'task_id': 11,
                    'worktree': str(proj_a / '.worktrees' / '11'),
                    'level': 1, 'status': 'pending',
                }],
                'summary': sub_summary,
            },
        ],
        'summary': {
            'by_level': {0: 1, 1: 1, 2: 0},
            'by_status': {'pending': 2, 'resolved': 0, 'dismissed': 0},
        },
    }

    with patch(
        'dashboard.app.build_escalation_queues',
        return_value=queues,
    ), patch(
        'dashboard.app.fetch_tasks',
        new=AsyncMock(return_value=[task_dict]),
    ):
        resp = client.get('/api/v2/dashboard/escalations')

    assert resp.status_code == 200
    body = resp.json()
    subs = body['ESCALATIONS']['subsections']
    assert len(subs) == 2

    orch_sub = next(s for s in subs if s['kind'] == 'orchestrator')
    assert len(orch_sub['escalations']) == 1
    orch_row = orch_sub['escalations'][0]
    assert orch_row['project'] == 'projA'
    assert orch_row['task']['title'] == 'wired'
    assert orch_row['task_unresolved'] is False

    recon_sub = next(s for s in subs if s['kind'] == 'reconciliation')
    assert len(recon_sub['escalations']) == 1
    recon_row = recon_sub['escalations'][0]
    assert recon_row['project'] == 'projA'
    assert recon_row['task']['title'] == 'wired'
    assert recon_row['task_unresolved'] is False


def test_load_task_cards_caches_within_ttl(client, tmp_path):
    """_load_task_cards: cache hit within TTL + offline result not cached."""
    from dashboard.app import _task_cards_cache_clear

    proj_a = tmp_path / 'projA'
    orch_sub = {
        'id': str(proj_a),
        'label': 'projA',
        'kind': 'orchestrator',
        'escalations': [],
        'summary': _EMPTY_SUMMARY,
    }
    recon_sub = {
        'id': 'reconciliation',
        'label': 'fused-memory',
        'kind': 'reconciliation',
        'escalations': [],
        'summary': _EMPTY_SUMMARY,
    }
    one_orch_queues = {
        'subsections': [orch_sub, recon_sub],
        'summary': _EMPTY_SUMMARY,
    }
    task_list = [{'id': 1, 'title': 't', 'description': '', 'details': '', 'status': 'pending',
                  'priority': 'low', 'dependencies': [], 'metadata': {}}]

    # Case 1: cache hit — second request should NOT call fetch_tasks again.
    _task_cards_cache_clear()
    mock_ft = AsyncMock(return_value=task_list)
    with patch('dashboard.app.build_escalation_queues', return_value=one_orch_queues), \
         patch('dashboard.app.fetch_tasks', new=mock_ft):
        client.get('/api/v2/dashboard/escalations')
        client.get('/api/v2/dashboard/escalations')
    assert mock_ft.call_count == 1, f'expected 1 fetch_tasks call, got {mock_ft.call_count}'

    # Case 2: offline result NOT cached — each request should call fetch_tasks.
    _task_cards_cache_clear()
    mock_offline = AsyncMock(return_value={'offline': True, 'error': 'x'})
    with patch('dashboard.app.build_escalation_queues', return_value=one_orch_queues), \
         patch('dashboard.app.fetch_tasks', new=mock_offline):
        r1 = client.get('/api/v2/dashboard/escalations')
        r2 = client.get('/api/v2/dashboard/escalations')
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert mock_offline.call_count == 2, f'expected 2 fetch_tasks calls, got {mock_offline.call_count}'
