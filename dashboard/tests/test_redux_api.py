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
    assert 'current_task' not in orch


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


def test_shape_orchestrators_copies_last_update_when_present():
    """last_update ISO string from the raw dict is copied to the ORCHESTRATORS entry."""
    raw = [{
        'pids': [1000],
        'prd': '/home/leo/src/dark-factory/prd.md',
        'label': 'dark-factory/main',
        'project_root': '/home/leo/src/dark-factory',
        'running': True,
        'started': '2h ago',
        'last_update': '2026-06-14T20:00:00',
        'tasks': [],
        'worktrees': {},
        'summary': {'total': 0, 'done': 0, 'in_progress': 0, 'blocked': 0, 'pending': 0},
    }]
    body = redux_api.shape_orchestrators(raw)
    [orch] = body['ORCHESTRATORS']
    assert orch['last_update'] == '2026-06-14T20:00:00'


def test_shape_orchestrators_last_update_none_when_absent():
    """last_update is None on the ORCHESTRATORS entry when the raw dict omits the field."""
    raw = [{
        'pids': [2000],
        'prd': None,
        'label': 'proj',
        'project_root': '/home/leo/src/proj',
        'running': False,
        'started': '',
        'tasks': [],
        'worktrees': {},
        'summary': {'total': 0, 'done': 0, 'in_progress': 0, 'blocked': 0, 'pending': 0},
    }]
    body = redux_api.shape_orchestrators(raw)
    [orch] = body['ORCHESTRATORS']
    assert orch['last_update'] is None


def test_shape_orchestrators_propagates_offline_marker():
    """shape_orchestrators must copy offline=True / error from the raw entry to the wire dict.

    Fails today because the out_orchs.append() block does not include offline/error keys.
    """
    raw = [{
        'pids': [7777],
        'prd': '/home/leo/src/dark-factory/prd.md',
        'label': 'dark-factory/main',
        'project_root': '/home/leo/src/dark-factory',
        'running': True,
        'started': 'Mar18',
        'last_update': None,
        'tasks': [],
        'worktrees': {},
        'summary': {'total': 0, 'done': 0, 'in_progress': 0, 'blocked': 0, 'pending': 0},
        'offline': True,
        'error': 'boom',
    }]

    body = redux_api.shape_orchestrators(raw)
    [orch] = body['ORCHESTRATORS']
    assert orch.get('offline') is True, f'expected offline=True in ORCHESTRATORS entry, got: {orch}'
    assert orch.get('error') == 'boom', f'expected error=boom in ORCHESTRATORS entry, got: {orch}'


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


def test_shape_memory_uptime_threaded_when_present():
    """online status with uptime fields → both appear in MEMORY_STATUS."""
    body = redux_api.shape_memory(
        {
            'graphiti': {'node_count': 10},
            'mem0': {'memory_count': 5},
            'uptime_seconds': 277020,
            'started_at': '2026-06-12T10:00:00+00:00',
        },
        {'counts': {'pending': 0}, 'oldest_pending_age_seconds': None},
    )
    ms = body['MEMORY_STATUS']
    assert ms['uptime_seconds'] == 277020
    assert ms['started_at'] == '2026-06-12T10:00:00+00:00'


def test_shape_memory_uptime_none_when_absent():
    """online status missing uptime fields → keys present but None."""
    body = redux_api.shape_memory(
        {'graphiti': {'node_count': 1}, 'mem0': {'memory_count': 1}},
        {'counts': {'pending': 0}, 'oldest_pending_age_seconds': None},
    )
    ms = body['MEMORY_STATUS']
    assert ms['uptime_seconds'] is None
    assert ms['started_at'] is None


def test_shape_memory_offline_uptime_keys_none():
    """offline status → uptime_seconds and started_at present and None."""
    body = redux_api.shape_memory(
        {'offline': True, 'error': 'unreachable'},
        {'counts': {'pending': 0}, 'oldest_pending_age_seconds': None},
    )
    ms = body['MEMORY_STATUS']
    assert 'uptime_seconds' in ms
    assert ms['uptime_seconds'] is None
    assert 'started_at' in ms
    assert ms['started_at'] is None


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
# shape_memory — WAL block
# ---------------------------------------------------------------------------


def _basic_status_and_queue():
    return (
        {'graphiti': {}, 'mem0': {}, 'projects': {}},
        {'counts': {}, 'oldest_pending_age_seconds': None},
    )


def test_shape_memory_wal_offline_when_wal_missing():
    status, queue = _basic_status_and_queue()
    body = redux_api.shape_memory(status, queue, wal=None)
    wal = body['MEMORY_STATUS']['wal']
    assert wal['status'] == 'offline'
    assert wal['rows'] == []


def test_shape_memory_wal_offline_payload_propagates_error():
    status, queue = _basic_status_and_queue()
    body = redux_api.shape_memory(
        status, queue, wal={'offline': True, 'error': 'unreachable'},
    )
    wal = body['MEMORY_STATUS']['wal']
    assert wal['status'] == 'offline'
    assert wal['reason'] == 'unreachable'


def test_shape_memory_wal_ok_when_all_rows_healthy():
    from datetime import UTC, datetime
    now_iso = datetime.now(UTC).isoformat()
    status, queue = _basic_status_and_queue()
    body = redux_api.shape_memory(status, queue, wal={
        'stores': {
            'http://srv': {
                'task_backend': {'ts': now_iso, 'busy': 0, 'log': 12, 'checkpointed': 12,
                                 'detail': '1 project(s)'},
                'recon_journal': {'ts': now_iso, 'busy': 0, 'log': 4, 'checkpointed': 4,
                                  'detail': None},
            },
        },
    })
    wal = body['MEMORY_STATUS']['wal']
    assert wal['status'] == 'ok'
    assert wal['reason'] is None
    assert {r['store'] for r in wal['rows']} == {'task_backend', 'recon_journal'}
    for row in wal['rows']:
        assert row['status'] == 'ok'


def test_shape_memory_wal_red_on_busy_row():
    from datetime import UTC, datetime
    now_iso = datetime.now(UTC).isoformat()
    status, queue = _basic_status_and_queue()
    body = redux_api.shape_memory(status, queue, wal={
        'stores': {'http://srv': {
            'recon_journal': {'ts': now_iso, 'busy': 1, 'log': 200, 'checkpointed': 0,
                              'detail': None},
        }},
    })
    wal = body['MEMORY_STATUS']['wal']
    assert wal['status'] == 'red'
    assert 'recon_journal' in (wal['reason'] or '')
    assert wal['rows'][0]['status'] == 'red'


def test_shape_memory_wal_warn_on_log_frames_overflow():
    from datetime import UTC, datetime
    now_iso = datetime.now(UTC).isoformat()
    status, queue = _basic_status_and_queue()
    body = redux_api.shape_memory(status, queue, wal={
        'stores': {'http://srv': {
            'event_buffer': {'ts': now_iso, 'busy': 0, 'log': 10_000, 'checkpointed': 10_000,
                             'detail': None},
        }},
    })
    wal = body['MEMORY_STATUS']['wal']
    assert wal['status'] == 'warn'
    assert 'log=' in (wal['reason'] or '')


def test_shape_memory_wal_red_on_stale_ts():
    from datetime import UTC, datetime, timedelta
    old_iso = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    status, queue = _basic_status_and_queue()
    body = redux_api.shape_memory(status, queue, wal={
        'stores': {'http://srv': {
            'write_journal': {'ts': old_iso, 'busy': 0, 'log': 5, 'checkpointed': 5,
                              'detail': None},
        }},
    })
    wal = body['MEMORY_STATUS']['wal']
    assert wal['status'] == 'red'
    assert 'stale' in (wal['reason'] or '')


def test_shape_wal_status_red_on_corrupt_ts(caplog):
    """A corrupt ts string causes row status='red' with reason 'corrupt ts',
    panel status escalates to 'red', and a WARNING is emitted (via parse_timestamp_or_warn).

    Fails today because the except-ValueError path sets ts_dt=None -> age_s=None ->
    stale guard skipped, so row stays 'ok' and no WARNING is emitted.
    """
    import logging

    status, queue = _basic_status_and_queue()
    with caplog.at_level(logging.WARNING):
        body = redux_api.shape_memory(status, queue, wal={
            'stores': {'http://srv': {
                'task_backend': {
                    'ts': 'not-a-date',
                    'busy': 0,
                    'log': 5,
                    'checkpointed': 5,
                    'detail': None,
                },
            }},
        })
    wal = body['MEMORY_STATUS']['wal']
    # Row must be red with a 'corrupt ts' reason.
    assert len(wal['rows']) == 1
    row = wal['rows'][0]
    assert row['status'] == 'red', (
        f"expected row status='red' for corrupt ts, got {row['status']!r}; "
        f"row reason: {row.get('reason')!r}"
    )
    assert 'corrupt' in (row['reason'] or '').lower(), (
        f"expected 'corrupt' in row reason, got {row.get('reason')!r}"
    )
    # Panel-level status must escalate.
    assert wal['status'] == 'red', (
        f"expected panel status='red', got {wal['status']!r}"
    )
    # A WARNING must have been emitted (from shared.timestamps.parse_timestamp_or_warn).
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_records, (
        'expected at least one WARNING for corrupt ts, but no warnings were emitted'
    )


def test_shape_wal_status_ok_on_valid_recent_ts():
    """A store with a valid recent ts stays 'ok' — no regression from the fix."""
    from datetime import UTC, datetime
    now_iso = datetime.now(UTC).isoformat()
    status, queue = _basic_status_and_queue()
    body = redux_api.shape_memory(status, queue, wal={
        'stores': {'http://srv': {
            'task_backend': {'ts': now_iso, 'busy': 0, 'log': 0, 'checkpointed': 0,
                             'detail': None},
        }},
    })
    wal = body['MEMORY_STATUS']['wal']
    assert wal['status'] == 'ok'
    assert wal['rows'][0]['status'] == 'ok'


def test_shape_wal_status_benign_on_missing_ts(caplog):
    """A store where ts is None stays benign — missing ts is not an error."""
    import logging

    status, queue = _basic_status_and_queue()
    with caplog.at_level(logging.WARNING):
        body = redux_api.shape_memory(status, queue, wal={
            'stores': {'http://srv': {
                'task_backend': {'ts': None, 'busy': 0, 'log': 0, 'checkpointed': 0,
                                 'detail': None},
            }},
        })
    wal = body['MEMORY_STATUS']['wal']
    row = wal['rows'][0]
    assert row['status'] == 'ok', (
        f"expected row status='ok' for missing ts (benign), got {row['status']!r}"
    )
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warning_records, (
        f'expected no WARNINGs for missing ts, got: {[r.message for r in warning_records]}'
    )


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
    # Default: no halt_status passed → offline fallback per project.
    assert section['halt'] == {'offline': True}


def test_shape_merge_queue_injects_halt_status_per_project():
    raw = {
        '/home/leo/src/reify': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {'hit_rate': 0.0},
            'active': [],
        },
        '/home/leo/src/know-live': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {'hit_rate': 0.0},
            'active': [],
        },
        '/home/leo/src/dark-factory': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {'hit_rate': 0.0},
            'active': [],
        },
    }
    halt_status = {
        'reify': {'wired': True, 'halted': True, 'owner_esc_id': 'esc-42', 'offline': False},
        'know-live': {'wired': True, 'halted': False, 'owner_esc_id': None, 'offline': False},
        # dark-factory deliberately absent → offline fallback
    }
    body = redux_api.shape_merge_queue(raw, halt_status=halt_status)
    mq = body['MERGE_QUEUE']
    assert mq['reify']['halt']['halted'] is True
    assert mq['reify']['halt']['owner_esc_id'] == 'esc-42'
    assert mq['know-live']['halt']['halted'] is False
    assert mq['know-live']['halt']['wired'] is True
    assert mq['dark-factory']['halt'] == {'offline': True}


def test_shape_merge_queue_includes_train_events():
    """shape_merge_queue exposes train_events list per project."""
    raw = {
        '/home/leo/src/dark-factory': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {'hit_rate': 0.0},
            'active': [],
            'train_events': [
                {
                    'event_type': 'train_started',
                    'task_id': 'trn-a',
                    'run_id': 'run-1',
                    'timestamp': '2026-05-28T12:00:00+00:00',
                    'data': {'train_id': 't1', 'member_count': 3},
                }
            ],
        },
    }
    body = redux_api.shape_merge_queue(raw)
    section = body['MERGE_QUEUE']['dark-factory']
    assert 'train_events' in section, f'Missing train_events in section keys: {list(section.keys())}'
    assert isinstance(section['train_events'], list)
    assert len(section['train_events']) == 1
    assert section['train_events'][0]['event_type'] == 'train_started'

    # When 'train_events' key is absent, result should default to []
    raw_no_train = {
        '/home/leo/src/dark-factory': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {'hit_rate': 0.0},
            'active': [],
            # no 'train_events' key
        },
    }
    body2 = redux_api.shape_merge_queue(raw_no_train)
    assert body2['MERGE_QUEUE']['dark-factory']['train_events'] == []


# ---------------------------------------------------------------------------
# shape_merge_queue — outcomes.colors producer→reader contract (step-7)
# ---------------------------------------------------------------------------

# The five reify codes in alphabetical order (non-canonical → sorted).
_REIFY_LABELS = [
    'cas_retry',
    'dropped_plan_targets',
    'plan_files_narrowed',
    'plan_files_not_touched',
    'post_merge_equivalence_failed',
]


def test_shape_merge_queue_attaches_outcome_colors():
    """shape_merge_queue adds outcomes['colors'] parallel to labels."""
    from dashboard.data.outcome_colors import assign_outcome_colors

    raw = {
        '/home/leo/src/dark-factory': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': _REIFY_LABELS, 'values': [3, 2, 1, 4, 2]},
            'latency': {},
            'recent': [],
            'speculative': {},
            'active': [],
        },
    }
    body = redux_api.shape_merge_queue(raw)
    outcomes = body['MERGE_QUEUE']['dark-factory']['outcomes']

    assert 'colors' in outcomes, f"Expected 'colors' key in outcomes; got keys: {list(outcomes)}"
    colors = outcomes['colors']
    assert isinstance(colors, list)
    assert len(colors) == len(_REIFY_LABELS), (
        f"Expected {len(_REIFY_LABELS)} colors, got {len(colors)}"
    )
    assert colors == assign_outcome_colors(_REIFY_LABELS), (
        "outcomes['colors'] does not match assign_outcome_colors(labels)"
    )


def test_shape_merge_queue_empty_outcomes_yields_empty_colors():
    """shape_merge_queue with empty outcomes attaches colors == []."""
    raw = {
        '/home/leo/src/dark-factory': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {},
            'active': [],
        },
    }
    body = redux_api.shape_merge_queue(raw)
    outcomes = body['MERGE_QUEUE']['dark-factory']['outcomes']
    assert outcomes.get('colors') == [], (
        f"Expected colors == [] for empty outcomes, got {outcomes.get('colors')!r}"
    )


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


def test_shape_burndown_emits_completed_and_velocity():
    """shape_burndown emits completed/velocity per-project and aggregated correctly.

    Per-project completed = delta(done[-1] - done[0]).
    Aggregate completed = sum of per-project completeds (NOT delta on aggregate series).
    Aggregate velocity = aggregate_completed / distinct_days(sorted_labels).
    """
    series = {
        'dark_factory': {
            'labels': ['2026-05-20T00:00:00', '2026-05-21T00:00:00'],
            'done': [3, 5], 'in_progress': [1, 1], 'blocked': [0, 0], 'pending': [10, 8],
        },
        'reify': {
            'labels': ['2026-05-20T00:00:00', '2026-05-21T00:00:00'],
            'done': [10, 12], 'in_progress': [0, 1], 'blocked': [0, 0], 'pending': [4, 3],
        },
    }
    body = redux_api.shape_burndown(series)

    df = body['BURNDOWN_BY_PROJECT']['dark_factory']
    assert df['completed'] == 2       # 5 - 3
    assert df['velocity'] == 1.0      # 2 / 2 distinct days
    assert df['window_days'] == 2     # 2026-05-20 and 2026-05-21

    ri = body['BURNDOWN_BY_PROJECT']['reify']
    assert ri['completed'] == 2       # 12 - 10
    assert ri['velocity'] == 1.0      # 2 / 2 distinct days
    assert ri['window_days'] == 2     # same label range

    agg = body['BURNDOWN']
    assert agg['completed'] == 4      # sum(2, 2) — not delta on aggregate series
    assert agg['velocity'] == 2.0     # 4 / 2 aggregate distinct days
    assert agg['window_days'] == 2    # union of all labels = 2 distinct days


def test_shape_burndown_completed_ignores_snapshot_frequency():
    """100 flat snapshots in one day must not inflate completed/velocity.

    Regression: the buggy frontend summed all snapshot done-counts.
    The correct delta-based answer is max(0, 7-7) = 0.
    """
    labels = [f'2026-05-20T{h:02d}:{m:02d}:00' for h in range(10) for m in range(10)]
    series = {
        'proj_x': {
            'labels': labels,
            'done': [7] * 100,
            'in_progress': [0] * 100,
            'blocked': [0] * 100,
            'pending': [5] * 100,
        },
    }
    body = redux_api.shape_burndown(series)
    assert body['BURNDOWN']['completed'] == 0
    assert body['BURNDOWN']['velocity'] == 0.0


# ---------------------------------------------------------------------------
# shape_escalations
# ---------------------------------------------------------------------------

_EMPTY_SUMMARY = {
    'by_level': {0: 0, 1: 0, 2: 0},
    'by_status': {'pending': 0, 'resolved': 0, 'dismissed': 0},
}


class TestShapeEscalations:
    """Tests for redux_api.shape_escalations."""

    def test_shape_escalations_basic_envelope(self):
        """Empty queues → correct envelope with passthrough summary."""
        queues = {'subsections': [], 'summary': _EMPTY_SUMMARY}
        body = redux_api.shape_escalations(queues=queues, task_maps={})
        assert set(body.keys()) == {'ESCALATIONS'}
        esc = body['ESCALATIONS']
        assert set(esc.keys()) == {'subsections', 'summary'}
        assert esc['subsections'] == []
        assert esc['summary'] == _EMPTY_SUMMARY

    def test_shape_escalations_preserves_subsection_metadata(self):
        """Orchestrator subsection metadata passes through unchanged."""
        subsection = {
            'id': '/p/projA',
            'label': 'projA',
            'kind': 'orchestrator',
            'escalations': [],
            'summary': _EMPTY_SUMMARY,
        }
        queues = {'subsections': [subsection], 'summary': _EMPTY_SUMMARY}
        body = redux_api.shape_escalations(queues=queues, task_maps={})
        subsections = body['ESCALATIONS']['subsections']
        assert len(subsections) == 1
        out = subsections[0]
        assert out['id'] == '/p/projA'
        assert out['label'] == 'projA'
        assert out['kind'] == 'orchestrator'
        assert out['summary'] == _EMPTY_SUMMARY
        assert out['escalations'] == []

    def test_shape_escalations_orchestrator_attaches_task_card(self):
        """Orchestrator row gets project label and resolved task card."""
        task = {
            'id': 42, 'title': 'task-42-title', 'description': 'd',
            'details': 'D', 'status': 'pending', 'priority': 'high',
            'dependencies': [], 'metadata': {},
        }
        subsection = {
            'id': '/p/projA',
            'label': 'projA',
            'kind': 'orchestrator',
            'escalations': [{'id': 'esc-1', 'task_id': '42', 'level': 0, 'status': 'pending', 'summary': 'oops'}],
            'summary': _EMPTY_SUMMARY,
        }
        queues = {'subsections': [subsection], 'summary': _EMPTY_SUMMARY}
        task_maps = {'/p/projA': [task]}
        body = redux_api.shape_escalations(queues=queues, task_maps=task_maps)
        rows = body['ESCALATIONS']['subsections'][0]['escalations']
        assert len(rows) == 1
        row = rows[0]
        # original esc fields preserved
        assert row['id'] == 'esc-1'
        assert row['summary'] == 'oops'
        # new fields
        assert row['project'] == 'projA'
        assert row['task'] == task
        assert row['task_unresolved'] is False

    def test_shape_escalations_orchestrator_unresolved_task(self):
        """Orchestrator row with unknown task_id → task=None, task_unresolved=True."""
        subsection = {
            'id': '/p/projA',
            'label': 'projA',
            'kind': 'orchestrator',
            'escalations': [
                {'id': 'esc-99', 'task_id': '999', 'level': 1, 'status': 'pending', 'summary': 'gone'},
            ],
            'summary': _EMPTY_SUMMARY,
        }
        queues = {'subsections': [subsection], 'summary': _EMPTY_SUMMARY}
        # task_maps has no task with id=999
        task_maps = {'/p/projA': [{'id': 1, 'title': 'other', 'description': '', 'details': '',
                                    'status': 'done', 'priority': 'low', 'dependencies': [], 'metadata': {}}]}
        body = redux_api.shape_escalations(queues=queues, task_maps=task_maps)
        rows = body['ESCALATIONS']['subsections'][0]['escalations']
        assert len(rows) == 1
        row = rows[0]
        assert row['project'] == 'projA'
        assert row['task'] is None
        assert row['task_unresolved'] is True
        # original esc fields preserved
        assert row['id'] == 'esc-99'
        assert row['summary'] == 'gone'

    def test_shape_escalations_reconciliation_resolves_via_worktree(self, tmp_path):
        """Reconciliation row: worktree under projA root → project='projA', task resolved."""
        task = {
            'id': 7, 'title': 'recon-task-7', 'description': 'x',
            'details': '', 'status': 'pending', 'priority': 'medium',
            'dependencies': [], 'metadata': {},
        }
        projA_root = tmp_path / 'projA'
        projA_root.mkdir()
        worktree_path = str(projA_root / '.worktrees' / '7')
        subsection = {
            'id': 'reconciliation',
            'label': 'fused-memory',
            'kind': 'reconciliation',
            'escalations': [
                {
                    'id': 'esc-r1',
                    'task_id': '7',
                    'worktree': worktree_path,
                    'level': 1,
                    'status': 'pending',
                },
            ],
            'summary': _EMPTY_SUMMARY,
        }
        queues = {'subsections': [subsection], 'summary': _EMPTY_SUMMARY}
        task_maps = {str(projA_root): [task]}
        body = redux_api.shape_escalations(queues=queues, task_maps=task_maps)
        rows = body['ESCALATIONS']['subsections'][0]['escalations']
        assert len(rows) == 1
        row = rows[0]
        assert row['project'] == 'projA'
        assert row['task'] == task
        assert row['task_unresolved'] is False

    def test_shape_escalations_reconciliation_resolves_via_task_map_probe(self, tmp_path):
        """Reconciliation row without worktree resolves via task-id probe."""
        task = {
            'id': 42, 'title': 'probe-task', 'description': '',
            'details': '', 'status': 'pending', 'priority': 'low',
            'dependencies': [], 'metadata': {},
        }
        projB_root = tmp_path / 'projB'
        projB_root.mkdir()
        subsection = {
            'id': 'reconciliation',
            'label': 'fused-memory',
            'kind': 'reconciliation',
            'escalations': [
                {
                    'id': 'esc-probe', 'task_id': '42',
                    # no 'worktree' field
                    'level': 0, 'status': 'pending',
                },
            ],
            'summary': _EMPTY_SUMMARY,
        }
        queues = {'subsections': [subsection], 'summary': _EMPTY_SUMMARY}
        task_maps = {str(projB_root): [task]}
        body = redux_api.shape_escalations(queues=queues, task_maps=task_maps)
        rows = body['ESCALATIONS']['subsections'][0]['escalations']
        row = rows[0]
        assert row['project'] == 'projB'
        assert row['task'] == task
        assert row['task_unresolved'] is False

    def test_shape_escalations_reconciliation_unresolvable(self, tmp_path):
        """Reconciliation row that can't be resolved → project=None, task=None, task_unresolved=True."""
        projC_root = tmp_path / 'projC'
        projC_root.mkdir()
        # worktree is under a completely unrelated path
        unrelated_worktree = str(tmp_path / 'other_project' / '.worktrees' / '5')
        subsection = {
            'id': 'reconciliation',
            'label': 'fused-memory',
            'kind': 'reconciliation',
            'escalations': [
                {
                    'id': 'esc-unres',
                    'task_id': '99',  # not in any task_map
                    'worktree': unrelated_worktree,
                    'level': 2,
                    'status': 'pending',
                    'summary': 'unresolvable',
                },
            ],
            'summary': _EMPTY_SUMMARY,
        }
        queues = {'subsections': [subsection], 'summary': _EMPTY_SUMMARY}
        # task_maps only has projC with no matching task
        task_maps = {str(projC_root): [{'id': 1, 'title': 't', 'description': '', 'details': '',
                                         'status': 'done', 'priority': 'low', 'dependencies': [], 'metadata': {}}]}
        body = redux_api.shape_escalations(queues=queues, task_maps=task_maps)
        rows = body['ESCALATIONS']['subsections'][0]['escalations']
        assert len(rows) == 1
        row = rows[0]
        assert row['project'] is None
        assert row['task'] is None
        assert row['task_unresolved'] is True
        # original esc fields preserved
        assert row['id'] == 'esc-unres'
        assert row['summary'] == 'unresolvable'

    def test_shape_escalations_top_level_summary_passthrough(self):
        """Non-trivial top-level summary passes through verbatim (not recomputed)."""
        top_summary = {
            'by_level': {0: 2, 1: 1, 2: 1},
            'by_status': {'pending': 3, 'resolved': 1, 'dismissed': 0},
        }
        queues = {'subsections': [], 'summary': top_summary}
        body = redux_api.shape_escalations(queues=queues, task_maps={})
        assert body['ESCALATIONS']['summary'] == top_summary


# ---------------------------------------------------------------------------
# shape_merge_queue — active_approximate pass-through (task-1606 step-9)
# ---------------------------------------------------------------------------


def _mq_raw(label: str, active_approximate: bool | None = None) -> dict:
    """Build a minimal per_project entry keyed by absolute path matching label."""
    data: dict = {
        'depth_timeseries': {'labels': [], 'values': []},
        'outcomes': {'labels': [], 'values': []},
        'latency': {},
        'recent': [],
        'speculative': {},
        'active': [{'task_id': '1', 'branch': 'task/1', 'state': 'queued'}],
        'train_events': [],
    }
    if active_approximate is not None:
        data['active_approximate'] = active_approximate
    # Key by a fake abs-path whose basename == label
    return {f'/proj/{label}': data}


class TestShapeMergeQueueActiveApproximate:
    """Tests that shape_merge_queue surfaces active_approximate per project."""

    def test_active_approximate_true_surfaces_in_output(self):
        """active_approximate=True in per_project → MERGE_QUEUE[label]['active_approximate'] is True."""
        raw = _mq_raw('myproj', active_approximate=True)
        body = redux_api.shape_merge_queue(raw)
        mq = body['MERGE_QUEUE']
        assert 'myproj' in mq
        assert mq['myproj']['active_approximate'] is True

    def test_active_approximate_false_surfaces_in_output(self):
        """active_approximate=False explicitly set → surfaces as False."""
        raw = _mq_raw('myproj', active_approximate=False)
        body = redux_api.shape_merge_queue(raw)
        assert body['MERGE_QUEUE']['myproj']['active_approximate'] is False

    def test_active_approximate_absent_defaults_false(self):
        """active_approximate absent from per_project data → defaults to False."""
        raw = _mq_raw('myproj', active_approximate=None)
        body = redux_api.shape_merge_queue(raw)
        assert body['MERGE_QUEUE']['myproj']['active_approximate'] is False

    def test_active_approximate_per_project_isolated(self):
        """Two projects with different active_approximate values are kept isolated."""
        per_project = {
            '/proj/alpha': {
                'depth_timeseries': {'labels': [], 'values': []},
                'outcomes': {'labels': [], 'values': []}, 'latency': {},
                'recent': [], 'speculative': {}, 'train_events': [],
                'active': [], 'active_approximate': True,
            },
            '/proj/beta': {
                'depth_timeseries': {'labels': [], 'values': []},
                'outcomes': {'labels': [], 'values': []}, 'latency': {},
                'recent': [], 'speculative': {}, 'train_events': [],
                'active': [],
                # active_approximate absent → defaults False
            },
        }
        body = redux_api.shape_merge_queue(per_project)
        mq = body['MERGE_QUEUE']
        assert mq['alpha']['active_approximate'] is True
        assert mq['beta']['active_approximate'] is False

    def test_existing_active_list_unchanged(self):
        """shape_merge_queue adding active_approximate does not break the active list."""
        raw = _mq_raw('myproj', active_approximate=True)
        body = redux_api.shape_merge_queue(raw)
        active = body['MERGE_QUEUE']['myproj']['active']
        assert isinstance(active, list)
        assert len(active) == 1
        assert active[0]['task_id'] == '1'


# ---------------------------------------------------------------------------
# shape_merge_queue — train_throughput passthrough (step-14 RED / step-15 GREEN)
# ---------------------------------------------------------------------------


def test_shape_merge_queue_includes_train_throughput():
    """shape_merge_queue exposes train_throughput dict per project.

    When per-project data contains 'train_throughput', it must appear in the
    shaped output alongside 'train_events' and 'speculative'.
    When 'train_throughput' is absent, the shaped output defaults to {}.
    """
    throughput_payload = {
        'trains_landed': 2,
        'tasks_landed_via_trains': 4,
        'train_verifies_per_landed_task': 0.5,
        'baseline_solo_landed': 3,
        'baseline_verifies_per_landed_task': 1.0,
        'verifies_per_landed_task_delta': 0.5,
        'train_cas_retry_rate': 0.25,
        'baseline_cas_retry_rate': 0.5,
        'cas_retry_rate_delta': 0.25,
        'improved': True,
    }
    raw = {
        '/home/leo/src/dark-factory': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {'hit_rate': 0.0},
            'active': [],
            'train_events': [],
            'train_throughput': throughput_payload,
        },
    }
    body = redux_api.shape_merge_queue(raw)
    section = body['MERGE_QUEUE']['dark-factory']

    assert 'train_throughput' in section, (
        f"expected 'train_throughput' in shaped output; keys: {list(section.keys())}"
    )
    tt = section['train_throughput']
    assert tt['trains_landed'] == 2
    assert tt['tasks_landed_via_trains'] == 4
    assert tt['improved'] is True

    # When 'train_throughput' key is absent, defaults to {}.
    raw_no_throughput = {
        '/home/leo/src/dark-factory': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {'hit_rate': 0.0},
            'active': [],
            'train_events': [],
            # no 'train_throughput' key
        },
    }
    body2 = redux_api.shape_merge_queue(raw_no_throughput)
    assert body2['MERGE_QUEUE']['dark-factory']['train_throughput'] == {}
