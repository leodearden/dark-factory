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
        'dashboard.app.collect_tasks_with_counts',
        new=AsyncMock(return_value=([], [], {})),
    ):
        resp = client.get('/api/v2/dashboard/tasks')
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {'ACTIVE_TASKS', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS', 'DONE_COUNTS'}
    assert 'FILE_LOCKS' not in body
    assert isinstance(body['ACTIVE_TASKS'], list)
    assert body['TASKS_OFFLINE'] is False
    assert body['TASKS_OFFLINE_PROJECTS'] == []
    assert body['DONE_COUNTS'] == {}


def test_tasks_endpoint_includes_done_counts(client):
    """DONE_COUNTS payload carries the per-project done count from collect_tasks_with_counts."""
    with patch(
        'dashboard.app.collect_tasks_with_counts',
        new=AsyncMock(return_value=([], [], {'dark-factory': 7})),
    ):
        resp = client.get('/api/v2/dashboard/tasks')
    assert resp.status_code == 200
    body = resp.json()
    assert body['DONE_COUNTS'] == {'dark-factory': 7}


def test_tasks_surfaces_offline_marker_when_mcp_unreachable(client):
    """When the tasks collector reports offline projects, the payload sets ``offline=True``."""
    with patch(
        'dashboard.app.collect_tasks_with_counts',
        new=AsyncMock(return_value=([], ['dark-factory'], {})),
    ):
        resp = client.get('/api/v2/dashboard/tasks')
    assert resp.status_code == 200
    body = resp.json()
    assert body['TASKS_OFFLINE'] is True
    assert body['TASKS_OFFLINE_PROJECTS'] == ['dark-factory']


def test_tasks_endpoint_passes_resolve_external_true_and_forwards_external_deps(client):
    """api_tasks must call collect_tasks_with_counts with resolve_external=True
    and forward the external_deps field in ACTIVE_TASKS rows unchanged.

    Asserts:
    (a) collect_tasks_with_counts is called with resolve_external=True
    (b) ACTIVE_TASKS[0]['external_deps'] contains the resolved dep
    (c) Top-level key set is unchanged (non-breaking)
    """
    mock_row = {
        'id': 'dark-factory/T-5',
        'project': 'dark-factory',
        'title': 'waits on upstream',
        'status': 'pending',
        'external_deps': [{'id': 'dark_factory:13', 'status': 'done'}],
    }
    mock = AsyncMock(return_value=([mock_row], [], {}))

    with patch('dashboard.app.collect_tasks_with_counts', new=mock):
        resp = client.get('/api/v2/dashboard/tasks')

    assert resp.status_code == 200
    body = resp.json()

    # (a) resolve_external=True was passed
    call_kwargs = mock.call_args.kwargs
    assert call_kwargs.get('resolve_external') is True, (
        f'expected resolve_external=True in call_args.kwargs, got: {call_kwargs}'
    )

    # (b) external_deps passes through unmodified
    assert body['ACTIVE_TASKS'][0]['external_deps'] == [
        {'id': 'dark_factory:13', 'status': 'done'}
    ]

    # (c) top-level key set unchanged
    assert set(body) == {'ACTIVE_TASKS', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS', 'DONE_COUNTS'}


def test_tasks_endpoint_passes_max_cancelled_per_project(client):
    """api_tasks must call collect_tasks_with_counts with max_cancelled_per_project=_MAX_CANCELLED_PER_PROJECT.

    Asserts:
    (a) max_cancelled_per_project == _MAX_CANCELLED_PER_PROJECT is passed in call kwargs
    (b) existing kwargs still present: max_done_per_project == _MAX_DONE_PER_PROJECT,
        resolve_external == True
    (c) top-level payload key-set remains exactly
        {'ACTIVE_TASKS', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS', 'DONE_COUNTS'}
        (no new key added for cancelled)

    RED today: app.py does not yet pass max_cancelled_per_project.
    """
    from dashboard.data.active_tasks import _MAX_CANCELLED_PER_PROJECT, _MAX_DONE_PER_PROJECT

    mock = AsyncMock(return_value=([], [], {}))

    with patch('dashboard.app.collect_tasks_with_counts', new=mock):
        resp = client.get('/api/v2/dashboard/tasks')

    assert resp.status_code == 200
    body = resp.json()

    call_kwargs = mock.call_args.kwargs

    # (a) max_cancelled_per_project
    assert call_kwargs.get('max_cancelled_per_project') == _MAX_CANCELLED_PER_PROJECT, (
        f'expected max_cancelled_per_project={_MAX_CANCELLED_PER_PROJECT} in call kwargs, '
        f'got: {call_kwargs}'
    )

    # (b) existing kwargs unchanged
    assert call_kwargs.get('max_done_per_project') == _MAX_DONE_PER_PROJECT, (
        f'expected max_done_per_project={_MAX_DONE_PER_PROJECT} in call kwargs, got: {call_kwargs}'
    )
    assert call_kwargs.get('resolve_external') is True, (
        f'expected resolve_external=True in call kwargs, got: {call_kwargs}'
    )

    # (c) payload key-set unchanged
    assert set(body) == {'ACTIVE_TASKS', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS', 'DONE_COUNTS'}


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


def test_load_task_cards_ttl_expiry(client, tmp_path):
    """_load_task_cards: after TTL expires, fetch_tasks is called again."""
    import dashboard.app as app_module
    from dashboard.app import _task_cards_cache_clear

    proj_a = tmp_path / 'projA'
    one_orch_queues = {
        'subsections': [
            {'id': str(proj_a), 'label': 'projA', 'kind': 'orchestrator',
             'escalations': [], 'summary': _EMPTY_SUMMARY},
            {'id': 'reconciliation', 'label': 'fused-memory', 'kind': 'reconciliation',
             'escalations': [], 'summary': _EMPTY_SUMMARY},
        ],
        'summary': _EMPTY_SUMMARY,
    }
    task_list = [{'id': 1, 'title': 't', 'description': '', 'details': '',
                  'status': 'pending', 'priority': 'low', 'dependencies': [], 'metadata': {}}]

    _task_cards_cache_clear()
    original_ttl = app_module._TASK_CARDS_TTL_SECONDS
    mock_ft = AsyncMock(return_value=task_list)
    try:
        with patch('dashboard.app.build_escalation_queues', return_value=one_orch_queues), \
             patch('dashboard.app.fetch_tasks', new=mock_ft):
            # First request: cache miss — fetch_tasks called once, result cached.
            client.get('/api/v2/dashboard/escalations')
            assert mock_ft.call_count == 1

            # Zero out TTL so the cached entry is immediately treated as expired.
            app_module._TASK_CARDS_TTL_SECONDS = 0.0

            # Second request: TTL expired — fetch_tasks called again.
            resp = client.get('/api/v2/dashboard/escalations')
    finally:
        app_module._TASK_CARDS_TTL_SECONDS = original_ttl

    assert resp.status_code == 200
    assert mock_ft.call_count == 2, (
        f'expected 2 fetch_tasks calls after TTL expiry, got {mock_ft.call_count}'
    )


def test_escalations_endpoint_multi_root_gather(client, tmp_path):
    """Endpoint fetches each orchestrator root separately and maps tasks to the right subsection."""
    from dashboard.app import _task_cards_cache_clear

    _task_cards_cache_clear()

    proj_a = tmp_path / 'projA'
    proj_b = tmp_path / 'projB'

    task_a = {'id': 11, 'title': 'task-A', 'description': '', 'details': '',
              'status': 'pending', 'priority': 'low', 'dependencies': [], 'metadata': {}}
    task_b = {'id': 22, 'title': 'task-B', 'description': '', 'details': '',
              'status': 'pending', 'priority': 'high', 'dependencies': [], 'metadata': {}}

    queues = {
        'subsections': [
            {'id': str(proj_a), 'label': 'projA', 'kind': 'orchestrator',
             'escalations': [{'id': 'esc-a1', 'task_id': 11, 'level': 0,
                              'status': 'pending', 'summary': 'a-issue'}],
             'summary': _EMPTY_SUMMARY},
            {'id': str(proj_b), 'label': 'projB', 'kind': 'orchestrator',
             'escalations': [{'id': 'esc-b1', 'task_id': 22, 'level': 1,
                              'status': 'pending', 'summary': 'b-issue'}],
             'summary': _EMPTY_SUMMARY},
        ],
        'summary': _EMPTY_SUMMARY,
    }

    async def fetch_side_effect(_client, _config, root_id):
        if 'projA' in root_id:
            return [task_a]
        if 'projB' in root_id:
            return [task_b]
        return []

    with patch('dashboard.app.build_escalation_queues', return_value=queues), \
         patch('dashboard.app.fetch_tasks', side_effect=fetch_side_effect):
        resp = client.get('/api/v2/dashboard/escalations')

    assert resp.status_code == 200
    body = resp.json()
    subs = body['ESCALATIONS']['subsections']
    assert len(subs) == 2

    sub_a = next(s for s in subs if s['label'] == 'projA')
    sub_b = next(s for s in subs if s['label'] == 'projB')

    # projA subsection gets task-A (id=11), not task-B.
    assert len(sub_a['escalations']) == 1
    row_a = sub_a['escalations'][0]
    assert row_a['project'] == 'projA'
    assert row_a['task']['id'] == 11
    assert row_a['task']['title'] == 'task-A'
    assert row_a['task_unresolved'] is False

    # projB subsection gets task-B (id=22), not task-A.
    assert len(sub_b['escalations']) == 1
    row_b = sub_b['escalations'][0]
    assert row_b['project'] == 'projB'
    assert row_b['task']['id'] == 22
    assert row_b['task']['title'] == 'task-B'
    assert row_b['task_unresolved'] is False


# ---------------------------------------------------------------------------
# /api/load — host load card endpoint
# ---------------------------------------------------------------------------


def test_load_endpoint_returns_known_metric_shape(client) -> None:
    """/api/load returns all 9 known metrics with the expected per-metric shape.

    The test client creates a temp project_root where load-samples.db does not
    exist, so every metric must degrade to the all-placeholders shape (current=None,
    sparkline=[]).  This exercises the missing-DB degradation path end-to-end.
    """
    from dashboard.data.load import KNOWN_METRICS

    resp = client.get('/api/load')
    assert resp.status_code == 200

    body = resp.json()
    assert isinstance(body, dict)
    assert set(body.keys()) == set(KNOWN_METRICS)

    # Every key must carry exactly these four sub-keys
    expected_sub_keys = {'current', 'sparkline', 'window_mean', 'window_max'}
    for metric in KNOWN_METRICS:
        assert set(body[metric].keys()) == expected_sub_keys, (
            f'{metric} has unexpected keys: {set(body[metric].keys())}'
        )

    # Spot-check one metric for the placeholder values (DB absent → all None/[])
    oq = body['occt_queue_depth']
    assert oq['current'] is None
    assert oq['sparkline'] == []
    assert oq['window_mean'] is None
    assert oq['window_max'] is None
