"""Pure-function tests for the redux JSON API shape adapters."""

from __future__ import annotations

from dashboard.data import redux_api


# ---------------------------------------------------------------------------
# shape_orchestrators / PROJECTS
# ---------------------------------------------------------------------------


def test_shape_orchestrators_picks_first_pid_and_basename_project():
    raw = [{
        'pids': [482103, 482104],
        'prd': '/home/leo/src/dark-factory/prd.md',
        'label': 'dark-factory/main',
        'project_root': '/home/leo/src/dark-factory',
        'running': True,
        'started': '2h ago',
        'tasks': [],
        'worktrees': {},
        'summary': {'total': 0, 'done': 0, 'in_progress': 0, 'blocked': 0, 'pending': 0},
    }]

    body = redux_api.shape_orchestrators(raw)
    [orch] = body['ORCHESTRATORS']
    assert orch['pid'] == 482103
    assert orch['pids'] == [482103, 482104]
    assert orch['project'] == 'dark-factory'
    assert orch['running'] is True
    assert orch['summary']['total'] == 0
    assert orch['current_task'] == '—'


def test_shape_orchestrators_picks_execute_phase_for_current_task():
    raw = [{
        'pids': [42],
        'project_root': '/p',
        'running': True,
        'summary': {},
        'started': 'now',
        'worktrees': {
            7: {'phase': 'PLAN', 'metadata': {'task_id': '7', 'title': 'plan-task'}},
            9: {'phase': 'EXECUTE', 'metadata': {'task_id': '9', 'title': 'doing-this'}},
        },
    }]
    body = redux_api.shape_orchestrators(raw)
    assert body['ORCHESTRATORS'][0]['current_task'] == 'T-9: doing-this'


def test_shape_orchestrators_marks_inactive_known_projects():
    raw = [{
        'pids': [1], 'project_root': '/a/dark-factory',
        'running': True, 'summary': {}, 'started': '', 'worktrees': {},
    }]
    body = redux_api.shape_orchestrators(
        raw, known_project_roots=['/a/dark-factory', '/b/reify'],
    )
    by_id = {p['id']: p for p in body['PROJECTS']}
    assert by_id['dark-factory']['active'] is True
    assert by_id['reify']['active'] is False


# ---------------------------------------------------------------------------
# shape_memory
# ---------------------------------------------------------------------------


def test_shape_memory_offline_keeps_required_keys():
    body = redux_api.shape_memory(
        {'offline': True, 'error': 'unreachable'},
        {'counts': {'pending': 0}, 'oldest_pending_age_seconds': None},
    )
    ms = body['MEMORY_STATUS']
    assert ms['graphiti']['connected'] is False
    assert ms['mem0']['connected'] is False
    assert ms['taskmaster']['connected'] is False
    assert ms['queue']['counts'] == {'pending': 0}
    assert ms['offline'] is True


def test_shape_memory_online_passes_through_plus_defaults():
    body = redux_api.shape_memory(
        {'graphiti': {'node_count': 100}, 'mem0': {'memory_count': 50},
         'projects': {'dark_factory': {'graphiti_nodes': 100}}},
        {'counts': {'pending': 4}, 'oldest_pending_age_seconds': 12.5},
    )
    ms = body['MEMORY_STATUS']
    assert ms['graphiti']['connected'] is True
    assert ms['graphiti']['node_count'] == 100
    assert ms['mem0']['connected'] is True
    assert ms['queue']['counts']['pending'] == 4
    assert ms['queue']['oldest_pending_age_seconds'] == 12.5
    assert ms['projects']['dark_factory']['graphiti_nodes'] == 100


# ---------------------------------------------------------------------------
# shape_memory_graphs
# ---------------------------------------------------------------------------


def test_shape_memory_graphs_zips_ops_into_label_value_list():
    body = redux_api.shape_memory_graphs(
        {'labels': ['00:00', '01:00'], 'reads': [3, 7], 'writes': [1, 2]},
        {'labels': ['add_memory', 'search'], 'values': [10, 25]},
    )
    assert body['MEMORY_TIMESERIES']['labels'] == ['00:00', '01:00']
    assert body['MEMORY_OPS_BREAKDOWN'] == [
        {'label': 'add_memory', 'value': 10},
        {'label': 'search', 'value': 25},
    ]


# ---------------------------------------------------------------------------
# shape_recon
# ---------------------------------------------------------------------------


def test_shape_recon_keys_watermarks_by_project_and_extracts_agents():
    body = redux_api.shape_recon(
        buffer_stats={'buffered_count': 5, 'oldest_event_age_seconds': 10.0},
        burst_state=[
            {'agent_id': 'claude-task-7', 'state': 'bursting', 'last_write_at': 'x'},
            {'agent_id': 'claude-interactive', 'state': 'cooling', 'last_write_at': 'y'},
        ],
        watermarks=[
            {'project_id': 'p1', 'last_full_run_completed': 't1'},
            {'project_id': 'p2', 'last_full_run_completed': 't2'},
        ],
        verdict={'severity': 'minor', 'action_taken': 'repair'},
        runs=[{'id': 'R-1', 'status': 'success'}],
    )
    rs = body['RECON_STATE']
    assert rs['buffer']['buffered_count'] == 5
    assert set(rs['watermarks']) == {'p1', 'p2'}
    assert rs['watermarks']['p1']['last_full_run_completed'] == 't1'
    assert body['AGENTS'] == ['claude-interactive', 'claude-task-7']
    assert rs['runs'][0]['id'] == 'R-1'


def test_shape_recon_no_verdict_returns_none():
    body = redux_api.shape_recon(
        buffer_stats={}, burst_state=[], watermarks=[], verdict=None, runs=[],
    )
    assert body['RECON_STATE']['verdict'] is None


# ---------------------------------------------------------------------------
# shape_merge_queue
# ---------------------------------------------------------------------------


def test_shape_merge_queue_relabels_and_renames_depth():
    raw = {
        '/home/leo/src/dark-factory': {
            'depth_timeseries': {'labels': [0, 1], 'values': [3, 4]},
            'outcomes': {'labels': ['done'], 'values': [12]},
            'latency': {'p50': 6000},
            'recent': [{'task_id': '17'}],
            'speculative': {'hit_rate': 0.75},
            'active': [],
        },
    }
    body = redux_api.shape_merge_queue(raw)
    assert 'dark-factory' in body['MERGE_QUEUE']
    section = body['MERGE_QUEUE']['dark-factory']
    assert section['depth'] == {'labels': [0, 1], 'values': [3, 4]}
    assert section['recent'] == [{'task_id': '17'}]


# ---------------------------------------------------------------------------
# shape_costs
# ---------------------------------------------------------------------------


def test_shape_costs_flattens_summary_and_sums_by_role():
    body = redux_api.shape_costs(
        summary={
            'p1': {'total_spend': 12.0, 'task_count': 3},
            'p2': {'total_spend': 8.0, 'task_count': 2},
        },
        by_project={
            'p1': [{'model': 'sonnet', 'total': 10.0}, {'model': 'haiku', 'total': 2.0}],
            'p2': [{'model': 'sonnet', 'total': 8.0}],
        },
        by_account={
            'anthropic-pri': {'spend': 15.0, 'status': 'active', 'resets_at': None},
            'anthropic-sec': {'spend': 5.0, 'status': 'capped', 'resets_at': '2026-01-01T00:00:00'},
        },
        by_role={
            'p1': {'planner': {'sonnet': 6.0}, 'coder': {'sonnet': 4.0}},
            'p2': {'coder': {'sonnet': 8.0}},
        },
        trend={
            'p1': [{'day': '2026-04-28', 'total': 4.0}],
            'p2': [{'day': '2026-04-28', 'total': 1.5}, {'day': '2026-04-29', 'total': 3.0}],
        },
        events=[
            {'created_at': '2026-04-29T01:00', 'account_name': 'a', 'event_type': 'cap_hit',
             'details': 'caps until 06:42', 'project_id': 'p1', 'run_id': 'r1'},
        ],
    )
    costs = body['COSTS']
    assert costs['summary']['total'] == 20.0
    assert costs['summary']['runs'] == 5
    # by_project entries sorted desc by total, with model totals included
    assert costs['by_project'][0]['project'] == 'p1'
    assert costs['by_project'][0]['sonnet'] == 10.0
    # by_role sums coder across projects (4 + 8 = 12)
    by_role = {r['role']: r for r in costs['by_role']}
    assert by_role['coder']['total'] == 12.0
    # by_account share computed against total spend
    by_account_share = {a['account']: a['share'] for a in costs['by_account']}
    assert by_account_share['anthropic-pri'] == 75.0
    # trend collapses days across projects
    assert costs['trend']['labels'] == ['2026-04-28', '2026-04-29']
    assert costs['trend']['values'] == [5.5, 3.0]
    # events normalised
    assert costs['events'][0]['account'] == 'a'
    assert costs['events'][0]['event'] == 'cap_hit'


# ---------------------------------------------------------------------------
# shape_performance
# ---------------------------------------------------------------------------


def test_shape_performance_unions_project_keys():
    body = redux_api.shape_performance(
        paths={'p1': [{'path': 'one-pass', 'count': 10, 'pct': 100.0}]},
        escalations={'p2': {'steward_rate': 5.0, 'interactive_rate': 0.0}},
        histograms={'p1': {'outer': {'labels': ['1'], 'values': [10]},
                            'inner': {'labels': ['1'], 'values': [10]}}},
        ttc={'p1': {'p50': 60_000}},
    )
    assert set(body['PERFORMANCE']) == {'p1', 'p2'}
    assert body['PERFORMANCE']['p1']['paths'][0]['path'] == 'one-pass'
    assert body['PERFORMANCE']['p2']['escalation']['steward_rate'] == 5.0


# ---------------------------------------------------------------------------
# shape_burndown
# ---------------------------------------------------------------------------


def test_shape_burndown_aggregates_and_keeps_per_project():
    series = {
        'dark_factory': {'labels': ['D-1', 'D-0'], 'done': [3, 4], 'in_progress': [1, 2],
                         'blocked': [0, 0], 'pending': [10, 9]},
        'reify':        {'labels': ['D-1', 'D-0'], 'done': [1, 2], 'in_progress': [0, 1],
                         'blocked': [0, 0], 'pending': [4, 3]},
    }
    body = redux_api.shape_burndown(series)
    aggregate = body['BURNDOWN']
    assert aggregate['labels'] == ['D-0', 'D-1']  # sorted
    # D-0 done: 4 + 2; D-1 done: 3 + 1
    assert aggregate['done'] == [6, 4]
    assert set(body['BURNDOWN_BY_PROJECT']) == {'dark_factory', 'reify'}
