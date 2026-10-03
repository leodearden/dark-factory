"""Pure-function tests for the redux JSON API shape adapters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from dashboard import loops
from dashboard.data import burndown, census, redux_api
from dashboard.data.datum import (
    Datum,
    DatumContractError,
    DatumInvariant,
    DatumState,
    unknown_datum,
)
from dashboard.data.escalation_corpus import EscalationView
from dashboard.data.memory import WRITE_QUEUE_FRESHNESS_BOUND_SECONDS, write_queue_datum
from dashboard.data.performance import PerformanceCards
from dashboard.data.write_journal import MemoryOps

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
    assert 'summary' not in orch, (
        'nothing measures a task count on this path any more, so the shaper '
        'must not project one — a fabricated all-zero summary would read as a '
        f'measured "this orchestrator has no tasks": {orch}'
    )
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
    assert orch.get('degraded') is False, (
        'this fetch was attempted and demonstrably failed, so the root is '
        'proven unreachable; reporting it as merely unmeasured understates a '
        f'real outage, got: {orch}'
    )


def test_shape_orchestrators_projects_degraded():
    """A root the budget starved reaches the wire as degraded, NOT as offline.

    The pair matters, not either field alone: a degraded root's state is
    UNKNOWN, while an offline root is proven down.  Collapsing them here would
    re-merge on the wire what the raw entry keeps apart.  Discovery has set
    neither flag since task 5587; the pair is the shaper's contract for any
    caller that supplies it.
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
        'offline': False,
        'degraded': True,
        'error': (
            'exceeded its 7.0s share of the 20.0s orchestrators budget; its '
            'task tree is UNKNOWN for this render (not zero)'
        ),
    }]

    body = redux_api.shape_orchestrators(raw)
    [orch] = body['ORCHESTRATORS']
    assert orch.get('degraded') is True, f'expected degraded=True in ORCHESTRATORS entry, got: {orch}'
    assert orch.get('offline') is False, (
        'the budget expired before this root was measured; nothing proved it '
        f'unreachable, and saying so sends an operator to a healthy service: {orch}'
    )


def test_shape_orchestrators_degraded_defaults_false_when_absent():
    """An entry with no ``degraded`` key shapes to False, never a missing key.

    Every wire entry must carry the field: the orchestrators tab reads
    ``o.degraded`` directly, so a well-formed payload may never hand it
    ``undefined``.
    """
    raw = [{
        'pids': [2000],
        'prd': None,
        'label': 'proj',
        'project_root': '/home/leo/src/proj',
        'running': True,
        'started': 'Mar18',
        'last_update': None,
        'tasks': [],
        'worktrees': {},
        'summary': {'total': 0, 'done': 0, 'in_progress': 0, 'blocked': 0, 'pending': 0},
    }]

    body = redux_api.shape_orchestrators(raw)
    [orch] = body['ORCHESTRATORS']
    assert orch.get('degraded') is False, (
        f'degraded must be present and False on every entry, got: {orch}'
    )


# ---------------------------------------------------------------------------
# shape_memory
# ---------------------------------------------------------------------------

_MEMORY_SERVED_AT = datetime(2026, 10, 3, 12, 0, 30, tzinfo=UTC)
_ONLINE_STATUS = {'graphiti': {}, 'mem0': {}, 'projects': {}}
_OFFLINE_STATUS = {'offline': True, 'error': 'unreachable'}


def _queue_datum(*, pending=0, retry=0, dead=0, oldest=None, measured_at=_MEMORY_SERVED_AT):
    """A reachable write-queue reading, built by its real producer."""
    return write_queue_datum(
        {
            'counts': {'pending': pending, 'retry': retry, 'dead': dead},
            'oldest_pending_age_seconds': oldest,
        },
        measured_at=measured_at,
    )


def _shape_memory(status, queue=None, **kwargs):
    return redux_api.shape_memory(
        status,
        _queue_datum() if queue is None else queue,
        served_at=_MEMORY_SERVED_AT,
        **kwargs,
    )


def test_shape_memory_offline_keeps_required_keys():
    body = _shape_memory(_OFFLINE_STATUS)
    ms = body['MEMORY_STATUS']
    assert ms['graphiti']['connected'] is False
    assert ms['mem0']['connected'] is False
    assert ms['taskmaster']['connected'] is False
    assert ms['queue']['stats']['value']['pending'] == 0
    assert ms['offline'] is True


def test_shape_memory_offline_status_still_renders_a_measured_queue():
    """get_status offline / get_queue_stats online: the queue's state, not zeros."""
    body = _shape_memory(_OFFLINE_STATUS, _queue_datum(pending=4, retry=1))
    stats = body['MEMORY_STATUS']['queue']['stats']
    assert stats['state'] == 'fresh'
    assert stats['value']['pending'] == 4
    assert stats['value']['retry'] == 1
    assert stats['as_of'] == _MEMORY_SERVED_AT.isoformat()


def test_shape_memory_online_status_renders_an_unmeasured_queue_as_unknown():
    queue = unknown_datum(
        'http://localhost:8002: ConnectError: refused', WRITE_QUEUE_FRESHNESS_BOUND_SECONDS,
    )
    body = _shape_memory(_ONLINE_STATUS, queue)
    block = body['MEMORY_STATUS']['queue']
    assert block['stats']['state'] == 'unknown'
    assert block['stats']['value'] is None
    assert block['stats']['reason'] == 'http://localhost:8002: ConnectError: refused'
    assert 'counts' not in block
    assert 'offline' not in block, 'the Datum state carries the offline fact now'


@pytest.mark.parametrize('status', [_ONLINE_STATUS, _OFFLINE_STATUS], ids=['online', 'offline'])
def test_shape_memory_queue_block_is_one_normaliser_for_both_branches(status):
    queue = _queue_datum(pending=2, oldest=3.0)
    spark = {'labels': ['2026-10-03T11:00:00+00:00'], 'values': [2]}
    online = _shape_memory(_ONLINE_STATUS, queue, queue_spark=spark)
    shaped = _shape_memory(status, queue, queue_spark=spark)
    assert shaped['MEMORY_STATUS']['queue'] == online['MEMORY_STATUS']['queue']
    assert shaped['MEMORY_STATUS']['queue']['spark'] == spark


def test_shape_memory_refuses_a_queue_that_is_not_a_datum():
    with pytest.raises(DatumContractError) as raised:
        _shape_memory(_ONLINE_STATUS, {'counts': {'pending': 0}})
    assert raised.value.invariant is DatumInvariant.DATUM_REQUIRED


def test_shape_memory_ages_a_queue_reading_past_its_bound():
    measured_at = _MEMORY_SERVED_AT - timedelta(seconds=WRITE_QUEUE_FRESHNESS_BOUND_SECONDS + 30)
    body = _shape_memory(_ONLINE_STATUS, _queue_datum(pending=1, measured_at=measured_at))
    stats = body['MEMORY_STATUS']['queue']['stats']
    assert stats['state'] == 'stale'
    assert stats['value']['pending'] == 1
    assert 'freshness bound' in stats['reason']


def test_shape_memory_payload_carries_served_at():
    body = _shape_memory(_ONLINE_STATUS)
    assert body['served_at'] == _MEMORY_SERVED_AT.isoformat()


def test_shape_memory_uptime_threaded_when_present():
    """online status with uptime fields → both appear in MEMORY_STATUS."""
    body = _shape_memory(
        {
            'graphiti': {'node_count': 10},
            'mem0': {'memory_count': 5},
            'uptime_seconds': 277020,
            'started_at': '2026-06-12T10:00:00+00:00',
        },
    )
    ms = body['MEMORY_STATUS']
    assert ms['uptime_seconds'] == 277020
    assert ms['started_at'] == '2026-06-12T10:00:00+00:00'


def test_shape_memory_uptime_none_when_absent():
    """online status missing uptime fields → keys present but None."""
    body = _shape_memory({'graphiti': {'node_count': 1}, 'mem0': {'memory_count': 1}})
    ms = body['MEMORY_STATUS']
    assert ms['uptime_seconds'] is None
    assert ms['started_at'] is None


def test_shape_memory_offline_uptime_keys_none():
    """offline status → uptime_seconds and started_at present and None."""
    body = _shape_memory(_OFFLINE_STATUS)
    ms = body['MEMORY_STATUS']
    assert 'uptime_seconds' in ms
    assert ms['uptime_seconds'] is None
    assert 'started_at' in ms
    assert ms['started_at'] is None


def test_shape_memory_online_passes_through_plus_defaults():
    body = _shape_memory(
        {'graphiti': {'node_count': 100}, 'mem0': {'memory_count': 50},
         'projects': {'dark_factory': {'graphiti_nodes': 100}}},
        _queue_datum(pending=4, oldest=12.5),
    )
    ms = body['MEMORY_STATUS']
    assert ms['graphiti']['connected'] is True
    assert ms['graphiti']['node_count'] == 100
    assert ms['mem0']['connected'] is True
    assert ms['queue']['stats']['value']['pending'] == 4
    assert ms['queue']['stats']['value']['oldest_pending_age_seconds'] == 12.5
    assert ms['projects']['dark_factory']['graphiti_nodes'] == 100


# ---------------------------------------------------------------------------
# shape_memory — WAL block
# ---------------------------------------------------------------------------


def _basic_status_and_queue():
    return {'graphiti': {}, 'mem0': {}, 'projects': {}}, _queue_datum()


def test_shape_memory_wal_offline_when_wal_missing():
    status, queue = _basic_status_and_queue()
    body = redux_api.shape_memory(status, queue, served_at=_MEMORY_SERVED_AT, wal=None)
    wal = body['MEMORY_STATUS']['wal']
    assert wal['status'] == 'offline'
    assert wal['rows'] == []


def test_shape_memory_wal_offline_payload_propagates_error():
    status, queue = _basic_status_and_queue()
    body = redux_api.shape_memory(
        status, queue, served_at=_MEMORY_SERVED_AT, wal={'offline': True, 'error': 'unreachable'},
    )
    wal = body['MEMORY_STATUS']['wal']
    assert wal['status'] == 'offline'
    assert wal['reason'] == 'unreachable'


def test_shape_memory_wal_ok_when_all_rows_healthy():
    from datetime import UTC, datetime
    now_iso = datetime.now(UTC).isoformat()
    status, queue = _basic_status_and_queue()
    body = redux_api.shape_memory(status, queue, served_at=_MEMORY_SERVED_AT, wal={
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
    body = redux_api.shape_memory(status, queue, served_at=_MEMORY_SERVED_AT, wal={
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
    body = redux_api.shape_memory(status, queue, served_at=_MEMORY_SERVED_AT, wal={
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
    body = redux_api.shape_memory(status, queue, served_at=_MEMORY_SERVED_AT, wal={
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
        body = redux_api.shape_memory(status, queue, served_at=_MEMORY_SERVED_AT, wal={
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
    body = redux_api.shape_memory(status, queue, served_at=_MEMORY_SERVED_AT, wal={
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
        body = redux_api.shape_memory(status, queue, served_at=_MEMORY_SERVED_AT, wal={
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
# _shape_wal_status now-threading (task 2281)
# ---------------------------------------------------------------------------


def test_shape_wal_status_now_threading_age_seconds():
    """_shape_wal_status(wal, now=fixed) computes age_seconds against the passed now."""
    from datetime import UTC, datetime, timedelta

    fixed = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
    ts = fixed - timedelta(minutes=10)
    wal = {'stores': {'http://srv': {
        'task_backend': {'ts': ts.isoformat(), 'busy': 0, 'log': 0, 'checkpointed': 0,
                          'detail': None},
    }}}

    body = redux_api._shape_wal_status(wal, now=fixed)

    assert body['rows'][0]['age_seconds'] == 600


def test_shape_wal_status_now_threading_stale_boundary():
    """Stale/red status flips exactly at _WAL_STALE_SECONDS relative to the passed now."""
    from datetime import UTC, datetime, timedelta

    fixed = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
    just_inside = fixed - timedelta(seconds=redux_api._WAL_STALE_SECONDS)
    just_outside = fixed - timedelta(seconds=redux_api._WAL_STALE_SECONDS + 1)

    def _wal_at(ts):
        return {'stores': {'http://srv': {
            'task_backend': {'ts': ts.isoformat(), 'busy': 0, 'log': 0, 'checkpointed': 0,
                              'detail': None},
        }}}

    ok_body = redux_api._shape_wal_status(_wal_at(just_inside), now=fixed)
    red_body = redux_api._shape_wal_status(_wal_at(just_outside), now=fixed)

    assert ok_body['status'] == 'ok'
    assert red_body['status'] == 'red'
    assert 'stale' in (red_body['reason'] or '')


def test_shape_wal_status_no_now_brackets_real_clock():
    """Without now, _shape_wal_status still derives age_seconds from the current UTC clock.

    Brackets the real clock read with before/after captures (rather than patching
    a module-level ``datetime`` symbol) because the no-now branch resolves through
    ``resolve_now`` in ``dashboard.data.utils``.
    """
    from datetime import UTC, datetime, timedelta

    before = datetime.now(UTC)
    ts = before - timedelta(minutes=5)
    wal = {'stores': {'http://srv': {
        'task_backend': {'ts': ts.isoformat(), 'busy': 0, 'log': 0, 'checkpointed': 0,
                          'detail': None},
    }}}

    body = redux_api._shape_wal_status(wal)
    after = datetime.now(UTC)

    age = body['rows'][0]['age_seconds']
    lower = int((before - ts).total_seconds())
    upper = int((after - ts).total_seconds()) + 1
    assert lower <= age <= upper


# ---------------------------------------------------------------------------
# shape_memory_graphs
# ---------------------------------------------------------------------------


def _memory_ops() -> MemoryOps:
    return MemoryOps(
        labels=('11:00', '12:00'),
        reads=(3, 7),
        writes=(1, 2),
        other=(0, 2),
        by_operation=(('search', 10), ('add_memory', 3), ('compact', 2)),
    )


def test_shape_memory_graphs_serves_one_memory_ops_key():
    body = redux_api.shape_memory_graphs(_memory_ops())

    assert list(body) == ['MEMORY_OPS'], (
        'one query, one datum, one key: MEMORY_TIMESERIES and '
        'MEMORY_OPS_BREAKDOWN are retired'
    )
    ops = body['MEMORY_OPS']
    assert set(ops) == {
        'labels', 'reads', 'writes', 'other', 'total', 'totals', 'by_operation',
    }
    assert ops['labels'] == ['11:00', '12:00']
    assert ops['reads'] == [3, 7]
    assert ops['writes'] == [1, 2]
    assert ops['other'] == [0, 2]


def test_shape_memory_graphs_hourly_total_sums_the_three_series():
    ops = redux_api.shape_memory_graphs(_memory_ops())['MEMORY_OPS']

    assert ops['total'] == [
        r + w + o for r, w, o in zip(ops['reads'], ops['writes'], ops['other'], strict=True)
    ]
    assert ops['total'] == [4, 11]


def test_shape_memory_graphs_window_totals_reconcile_with_by_operation():
    """PRD sketch #11: the caption's three numbers sum to the donut's total."""
    ops = redux_api.shape_memory_graphs(_memory_ops())['MEMORY_OPS']
    totals = ops['totals']

    assert totals == {'reads': 10, 'writes': 3, 'other': 2, 'total': 15}
    assert totals['reads'] == sum(ops['reads'])
    assert totals['writes'] == sum(ops['writes'])
    assert totals['other'] == sum(ops['other'])
    assert totals['total'] == totals['reads'] + totals['writes'] + totals['other']
    assert totals['total'] == sum(row['value'] for row in ops['by_operation'])
    assert totals['total'] == sum(ops['total'])


def test_shape_memory_graphs_by_operation_keeps_memory_ops_order():
    ops = redux_api.shape_memory_graphs(_memory_ops())['MEMORY_OPS']

    assert ops['by_operation'] == [
        {'label': 'search', 'value': 10},
        {'label': 'add_memory', 'value': 3},
        {'label': 'compact', 'value': 2},
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

MQ_SERVED_AT = datetime(2026, 10, 1, 12, 0, 30, tzinfo=UTC)
"""The serving instant every shape_merge_queue case ages and validates against."""

_MQ_MEASURED_QUEUE = {
    'in_queue': Datum(0, MQ_SERVED_AT, DatumState.FRESH, None, 30),
    'live_probe_configured': True,
}
"""The queue fields the route resolves for every project: here, a measured empty queue."""


def _mq_titled(task_id: str) -> dict:
    """A merge row whose title the route already looked up."""
    title = Datum(f'task {task_id}', MQ_SERVED_AT, DatumState.FRESH, None, 1200)
    return {'task_id': task_id, 'title': title}


def _task_ids(rows):
    return [row['task_id'] for row in rows]


def test_shape_merge_queue_relabels_and_renames_depth():
    raw = {
        '/home/leo/src/dark-factory': {
            'depth_timeseries': {'labels': [0, 1], 'values': [3, 4]},
            'outcomes': {'labels': ['done'], 'values': [12]},
            'latency': {'p50': 6000},
            'recent': [_mq_titled('17')],
            'speculative': {'hit_rate': 0.75},
            'active': [],
            **_MQ_MEASURED_QUEUE,
        },
    }
    body = redux_api.shape_merge_queue(raw, served_at=MQ_SERVED_AT)
    assert 'dark-factory' in body['MERGE_QUEUE']
    section = body['MERGE_QUEUE']['dark-factory']
    assert section['depth'] == {'labels': [0, 1], 'values': [3, 4]}
    assert _task_ids(section['recent']) == ['17']
    # Default: no halt_status passed → offline fallback per project.
    assert section['halt'] == {'offline': True}


def test_shape_merge_queue_carries_the_recent_window_total():
    """recent_total is the window's merge count, which the capped recent rows may not reach."""
    recent = [_mq_titled(str(i)) for i in range(200)]
    raw = {
        '/home/leo/src/dark-factory': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': recent,
            'recent_total': 228,
            'speculative': {},
            'active': [],
            **_MQ_MEASURED_QUEUE,
        },
        '/home/leo/src/reify': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {},
            'active': [],
            **_MQ_MEASURED_QUEUE,
        },
    }
    mq = redux_api.shape_merge_queue(raw, served_at=MQ_SERVED_AT)['MERGE_QUEUE']
    assert mq['dark-factory']['recent_total'] == 228
    assert _task_ids(mq['dark-factory']['recent']) == _task_ids(recent)
    assert mq['reify']['recent_total'] == 0


def test_shape_merge_queue_injects_halt_status_per_project():
    raw = {
        '/home/leo/src/reify': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {'hit_rate': 0.0},
            'active': [],
            **_MQ_MEASURED_QUEUE,
        },
        '/home/leo/src/know-live': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {'hit_rate': 0.0},
            'active': [],
            **_MQ_MEASURED_QUEUE,
        },
        '/home/leo/src/dark-factory': {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {'hit_rate': 0.0},
            'active': [],
            **_MQ_MEASURED_QUEUE,
        },
    }
    halt_status = {
        'reify': {'wired': True, 'halted': True, 'owner_esc_id': 'esc-42', 'offline': False},
        'know-live': {'wired': True, 'halted': False, 'owner_esc_id': None, 'offline': False},
        # dark-factory deliberately absent → offline fallback
    }
    body = redux_api.shape_merge_queue(raw, halt_status=halt_status, served_at=MQ_SERVED_AT)
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
            **_MQ_MEASURED_QUEUE,
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
    body = redux_api.shape_merge_queue(raw, served_at=MQ_SERVED_AT)
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
            **_MQ_MEASURED_QUEUE,
            # no 'train_events' key
        },
    }
    body2 = redux_api.shape_merge_queue(raw_no_train, served_at=MQ_SERVED_AT)
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
            **_MQ_MEASURED_QUEUE,
        },
    }
    body = redux_api.shape_merge_queue(raw, served_at=MQ_SERVED_AT)
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
            **_MQ_MEASURED_QUEUE,
        },
    }
    body = redux_api.shape_merge_queue(raw, served_at=MQ_SERVED_AT)
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


_PERF_SERVED_AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
_PERF_WINDOW_SECONDS = 7 * 86400


def _perf_cards(
    as_of: datetime, state: DatumState = DatumState.FRESH, reason: str | None = None,
) -> Datum[PerformanceCards]:
    return Datum(
        value=PerformanceCards(
            paths=[{'path': 'one-pass', 'count': 10, 'pct': 100.0}],
            escalation={'total_tasks': 10, 'steward_rate': 5.0, 'interactive_rate': 0.0},
            hist_outer={'labels': ['0'], 'values': [10]},
            hist_inner={'labels': ['0'], 'values': [10]},
            ttc={'p50': 60_000, 'p75': 70_000, 'p90': 80_000, 'p95': 90_000, 'count': 10},
        ),
        as_of=as_of,
        state=state,
        reason=reason,
        freshness_bound_seconds=_PERF_WINDOW_SECONDS,
    )


_PERF_HISTORY = {
    'time_centiles_history': {'labels': ['2026-09-30T11:00'], 'p50': [60_000], 'p95': [90_000]},
    'one_pass_history': {'labels': ['2026-09-30T11:00'], 'values': [100.0]},
    'escalation_history': {'labels': ['2026-09-30T11:00'], 'values': [0.0]},
}


def test_shape_performance_entry_is_the_cards_datum_beside_its_histories():
    plus_two = timezone(timedelta(hours=2))
    cards = _perf_cards((_PERF_SERVED_AT - timedelta(hours=1)).astimezone(plus_two))
    body = redux_api.shape_performance(
        cards={'/home/leo/src/p1': cards},
        history={'/home/leo/src/p1': _PERF_HISTORY},
        served_at=_PERF_SERVED_AT,
    )
    entry = body['PERFORMANCE']['p1']
    assert set(entry) == {'cards', 'time_centiles_history', 'one_pass_history', 'escalation_history'}
    assert entry['cards'] == cards.to_wire()
    assert entry['cards']['as_of'] == '2026-09-30T11:00:00+00:00'
    assert entry['time_centiles_history'] == _PERF_HISTORY['time_centiles_history']


def test_shape_performance_project_without_history_gets_empty_blocks():
    body = redux_api.shape_performance(
        cards={'p1': _perf_cards(_PERF_SERVED_AT - timedelta(hours=1))},
        served_at=_PERF_SERVED_AT,
    )
    entry = body['PERFORMANCE']['p1']
    assert entry['time_centiles_history'] == {'labels': [], 'p50': [], 'p95': []}
    assert entry['one_pass_history'] == {'labels': [], 'values': []}
    assert entry['escalation_history'] == {'labels': [], 'values': []}


def test_shape_performance_lists_exactly_the_projects_with_cards():
    body = redux_api.shape_performance(
        cards={
            'active': _perf_cards(_PERF_SERVED_AT - timedelta(hours=1)),
            'idle': _perf_cards(
                _PERF_SERVED_AT - timedelta(days=20), DatumState.STALE,
                'no completions in the 7d window; last completion 2026-09-10T12:00:00+00:00',
            ),
        },
        history={'active': _PERF_HISTORY, 'history-only': _PERF_HISTORY},
        served_at=_PERF_SERVED_AT,
    )
    performance_by_label = body['PERFORMANCE']
    assert set(performance_by_label) == {'active', 'idle'}
    for entry in performance_by_label.values():
        assert entry['cards']['state'] in {'fresh', 'stale'}
        assert entry['cards']['value'] is not None


def test_shape_performance_serves_the_instant_it_validated_against():
    body = redux_api.shape_performance(
        cards={'p1': _perf_cards(_PERF_SERVED_AT - timedelta(hours=1))},
        served_at=_PERF_SERVED_AT,
    )
    assert body['served_at'] == '2026-09-30T12:00:00+00:00'


def test_shape_performance_serves_an_unread_project_as_an_unknown_cards_datum():
    reason = 'the loop histograms of this project could not be read'
    unread = Datum(
        value=None, as_of=None, state=DatumState.UNKNOWN, reason=reason,
        freshness_bound_seconds=_PERF_WINDOW_SECONDS,
    )
    body = redux_api.shape_performance(cards={'p1': unread}, served_at=_PERF_SERVED_AT)
    assert body['PERFORMANCE']['p1']['cards'] == {
        'value': None, 'as_of': None, 'state': 'unknown', 'reason': reason,
        'freshness_bound_seconds': _PERF_WINDOW_SECONDS,
    }


def test_shape_performance_propagates_a_broken_cards_datum():
    """A FRESH Datum older than its own bound is a shaper bug, not a state."""
    overdue = _perf_cards(_PERF_SERVED_AT - timedelta(days=8))
    with pytest.raises(DatumContractError):
        redux_api.shape_performance(cards={'p1': overdue}, served_at=_PERF_SERVED_AT)


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

    df = body['BURNDOWN_BY_PROJECT']['dark_factory']['latest']['value']
    assert df['completed'] == 2       # 5 - 3
    assert df['velocity'] == 1.0      # 2 / 2 distinct days
    assert df['window_days'] == 2     # 2026-05-20 and 2026-05-21

    ri = body['BURNDOWN_BY_PROJECT']['reify']['latest']['value']
    assert ri['completed'] == 2       # 12 - 10
    assert ri['velocity'] == 1.0      # 2 / 2 distinct days
    assert ri['window_days'] == 2     # same label range

    agg = body['BURNDOWN']['latest']['value']
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
    assert body['BURNDOWN']['latest']['value']['completed'] == 0
    assert body['BURNDOWN']['latest']['value']['velocity'] == 0.0


# --- divergent per-project label rows -------------------------------------
#
# The three tests above all give every project an IDENTICAL label row (or use a
# single project), which is exactly why the tabs.jsx mispairing of the aggregate
# union row with per-project values was invisible to the suite.  These pin the
# shape contract the consumer-side fix rests on, using divergent rows.

_DIVERGENT_SERIES = {
    'p1': {
        'labels': ['2026-05-20T00:00:00', '2026-05-22T00:00:00'],
        'done': [3, 7], 'in_progress': [1, 2], 'blocked': [0, 1], 'pending': [10, 6],
    },
    'p2': {
        'labels': ['2026-05-20T00:00:00', '2026-05-21T00:00:00', '2026-05-22T00:00:00'],
        'done': [100, 200, 300], 'in_progress': [10, 20, 30],
        'blocked': [1, 2, 3], 'pending': [50, 40, 30],
    },
}

_BURNDOWN_SERIES_KEYS = ('done', 'in_progress', 'blocked', 'pending')


def test_shape_burndown_per_project_labels_are_that_projects_own_row():
    """Each per-project block keeps its OWN snapshot row, not the union.

    Passes against current server code — this is a regression pin, not a RED
    step.  It fires if a future change densifies per-project rows onto the
    union row, which would silently re-break the tabs.jsx consumer.
    """
    body = redux_api.shape_burndown(_DIVERGENT_SERIES)

    assert body['BURNDOWN_BY_PROJECT']['p1']['labels'] == [
        '2026-05-20T00:00:00', '2026-05-22T00:00:00',
    ]
    assert body['BURNDOWN_BY_PROJECT']['p2']['labels'] == [
        '2026-05-20T00:00:00', '2026-05-21T00:00:00', '2026-05-22T00:00:00',
    ]
    # The aggregate carries the sorted union — strictly longer than p1's row.
    assert body['BURNDOWN']['labels'] == [
        '2026-05-20T00:00:00', '2026-05-21T00:00:00', '2026-05-22T00:00:00',
    ]
    assert len(body['BURNDOWN']['labels']) > len(body['BURNDOWN_BY_PROJECT']['p1']['labels'])


def test_shape_burndown_per_project_series_are_co_length_with_own_labels():
    """shape_burndown PRESERVES the co-length of each project's own row.

    Passes against current server code — a regression pin.  Note what it does
    and does not say: co-length is established UPSTREAM by get_burndown_series,
    which appends one value per key per snapshot row, and shape_burndown copies
    each block verbatim rather than normalizing it.  So this pins that the copy
    does not re-align or truncate anything, not that shape_burndown would
    repair a ragged input — see
    test_shape_burndown_ragged_input_is_passed_through_unnormalized for the
    contract on that.

    Given a co-length input, this is the invariant that makes `labels={pb.labels}`
    sound at the tabs.jsx consumer: a per-project block is internally co-indexed,
    so it needs no hole semantics.  Densifying these onto the union row would
    require making deriveVelocitySeries, the summary-table last-value reads, and
    compute_window_completion hole-aware first.
    """
    body = redux_api.shape_burndown(_DIVERGENT_SERIES)

    for pid, block in body['BURNDOWN_BY_PROJECT'].items():
        for key in _BURNDOWN_SERIES_KEYS:
            assert len(block[key]) == len(block['labels']), (
                f'{pid}.{key} has {len(block[key])} values for '
                f'{len(block["labels"])} labels'
            )


def test_shape_burndown_aggregate_series_are_co_length_with_union_labels():
    """Aggregate series are a label-indexed carry-last sum over the union row.

    At every union label each project contributes its last measured row at or
    before that label — neither a positional sum (which would fold p1's 05-22
    values into the 05-21 slot) nor a zero-fill (which would read p1 as
    having nothing at 05-21, only because its collector ticked elsewhere).
    """
    body = redux_api.shape_burndown(_DIVERGENT_SERIES)
    agg = body['BURNDOWN']
    n = len(agg['labels'])

    for key in _BURNDOWN_SERIES_KEYS:
        assert len(agg[key]) == n, f'aggregate {key} has {len(agg[key])} values for {n} labels'

    mid = agg['labels'].index('2026-05-21T00:00:00')
    # Only p2 measured this timestamp; p1 is carried from its 05-20 row.
    assert agg['done'][mid] == 203      # 200 + 3
    assert agg['pending'][mid] == 50    # 40 + 10
    # Timestamps both projects reported sum across them.
    assert agg['done'][agg['labels'].index('2026-05-20T00:00:00')] == 103   # 3 + 100
    assert agg['done'][agg['labels'].index('2026-05-22T00:00:00')] == 307   # 7 + 300


def test_shape_burndown_ragged_input_is_passed_through_unnormalized():
    """A ragged upstream row survives into BURNDOWN_BY_PROJECT unchanged.

    shape_burndown neither normalizes nor rejects a block whose series are not
    co-length with its labels — it copies them verbatim.  Ragged input is
    reachable in principle (compute_window_completion and
    compute_forecast_confidence both treat a length mismatch as a bail-out
    condition rather than an impossibility), so this pins what actually
    happens today rather than leaving it to be rediscovered:

      * the per-project block stays ragged (labels 3, done 2), so the co-length
        property the tabs.jsx `labels={pb.labels}` pairing relies on comes from
        get_burndown_series upstream, NOT from this function;
      * completed / velocity / window_days are zeroed and the forecast is None,
        because both helpers bail out on the mismatch;
      * the aggregate reads the missing tail as a HOLE (None), never 0: a
        value missing from a measured row is not a measured zero.

    If a future change makes shape_burndown normalize (pad/truncate) or raise
    on ragged input, that is a deliberate contract change and this test should
    be updated to the new behaviour, not deleted.
    """
    body = redux_api.shape_burndown({
        'p1': {
            'labels': [
                '2026-05-20T00:00:00', '2026-05-21T00:00:00', '2026-05-22T00:00:00',
            ],
            'done': [3, 7],  # ragged: one short of its own label row
            'in_progress': [1, 2, 3], 'blocked': [0, 0, 0], 'pending': [10, 9, 8],
        },
    })

    block = body['BURNDOWN_BY_PROJECT']['p1']
    assert len(block['labels']) == 3
    assert block['done'] == [3, 7]          # verbatim — not padded, not truncated
    assert block['in_progress'] == [1, 2, 3]
    completion = block['latest']['value']
    assert completion['completed'] == 0
    assert completion['velocity'] == 0.0
    assert completion['window_days'] == 0
    assert block['forecast']['state'] == 'unknown'
    assert block['forecast']['value'] is None

    agg = body['BURNDOWN']
    assert agg['labels'] == [
        '2026-05-20T00:00:00', '2026-05-21T00:00:00', '2026-05-22T00:00:00',
    ]
    # The unreported third slot is a hole, not a 0.
    assert agg['done'] == [3, 7, None]
    assert agg['in_progress'] == [1, 2, 3]
    assert agg['latest']['value']['completed'] == 0   # sum of per-project completeds


def test_shape_burndown_aggregate_endpoint_is_the_sum_of_each_projects_last_measured_row():
    """Sketch #6: the newest point sums A's t3 row with B's t2 row, never A alone.

    A measured t1 and t3; B measured only t2. Before B's first row B has
    nothing to carry, so t1 is A alone.
    """
    t1, t2, t3 = '2026-05-20T00:00:00', '2026-05-20T00:10:00', '2026-05-20T00:20:00'
    a = {'labels': [t1, t3], 'done': [3, 5], 'in_progress': [2, 4],
         'blocked': [1, 0], 'pending': [9, 7]}
    b = {'labels': [t2], 'done': [50], 'in_progress': [6], 'blocked': [2], 'pending': [30]}

    agg = redux_api.shape_burndown({'A': a, 'B': b})['BURNDOWN']

    assert agg['labels'] == [t1, t2, t3]
    for key in ('in_progress', 'blocked', 'pending', 'done'):
        assert agg[key][0] == a[key][0], key
        assert agg[key][1] == a[key][0] + b[key][0], key
        assert agg[key][-1] == a[key][1] + b[key][0], key


def test_shape_burndown_aggregate_carries_a_hole_where_the_project_contributes():
    """B's one row predates review (NULL): every label B contributes to is a hole.

    Members both projects measured still sum; at t1, before B's first row,
    A's review stands alone.
    """
    labels = ['2026-05-20T00:00:00', '2026-05-21T00:00:00', '2026-05-22T00:00:00']
    a = {'labels': labels, 'done': [1, 2, 3], 'pending': [9, 8, 7], 'review': [1, 2, 3]}
    b = {'labels': [labels[1]], 'done': [5], 'pending': [4], 'review': [None]}

    agg = redux_api.shape_burndown({'A': a, 'B': b})['BURNDOWN']

    assert agg['review'] == [1, None, None]
    assert agg['done'] == [1, 7, 8]
    assert agg['pending'] == [9, 12, 11]


_SPLIT_SERIES_KEYS = ('in_progress_live', 'in_progress_stranded', 'in_progress_rows')


def _nine_member_series(labels: list[str], base: int) -> dict:
    """A series carrying every census member plus the in-progress split."""
    n = len(labels)
    series: dict = {'labels': labels}
    for offset, key in enumerate(census.SERIES_KEYS.values()):
        series[key] = [base + offset * 10 + i for i in range(n)]
    for offset, key in enumerate(_SPLIT_SERIES_KEYS):
        series[key] = [base + 100 + offset * 10 + i for i in range(n)]
    return series


def test_shape_burndown_carries_all_nine_members_and_the_split_to_the_wire():
    """Cancelled, deferred and delta-1's three members are no longer dropped at the seam."""
    labels = ['2026-05-20T00:00:00', '2026-05-21T00:00:00']
    a = _nine_member_series(labels, base=1)
    b = _nine_member_series(labels, base=1000)

    body = redux_api.shape_burndown({'A': a, 'B': b})

    for key in (*census.SERIES_KEYS.values(), *_SPLIT_SERIES_KEYS):
        for pid, fixture in (('A', a), ('B', b)):
            assert body['BURNDOWN_BY_PROJECT'][pid][key] == fixture[key], (pid, key)
        assert body['BURNDOWN'][key] == [x + y for x, y in zip(a[key], b[key], strict=True)], key
    assert body['BURNDOWN_BY_PROJECT']['A']['cancelled'] == a['cancelled']
    assert body['BURNDOWN_BY_PROJECT']['A']['deferred'] == a['deferred']


def test_shape_burndown_aggregate_forecast_is_the_measured_rows_fold():
    """B entering on day 6 with done 4000 is where B starts, not 4000 completions.

    A forecast over the summed ``done`` series would read that entry as a jump
    and forecast ~0 days; the fold over per-project measured series does not.
    """
    labels = [f'2026-04-{d:02d}T00:00:00' for d in range(1, 11)]
    a = {'labels': labels, 'done': list(range(10)), 'pending': [20] * 10}
    b = {'labels': labels[5:], 'done': [4000] * 5, 'pending': [5] * 5}

    agg = redux_api.shape_burndown({'A': a, 'B': b})['BURNDOWN']

    assert burndown.aggregate_forecast_confidence([a, b]) == {
        'forecast_low': 27.8,
        'forecast_high': 29.2,
    }
    assert agg['forecast']['value'] == {'forecast_low': 27.8, 'forecast_high': 29.2}


# ---------------------------------------------------------------------------
# shape_burndown — every block serves its own provenance (task 5592)
# ---------------------------------------------------------------------------

_BOUND = 2 * loops._SAMPLE_INTERVAL_SECONDS
_BASE = datetime(2026, 5, 20, tzinfo=UTC)
_WIRE_SERIES_KEYS = (*census.SERIES_KEYS.values(), *_SPLIT_SERIES_KEYS)
_DATUM_KEYS = {'value', 'as_of', 'state', 'reason', 'freshness_bound_seconds'}
_FLAT_FIELDS_NOW_IN_DATUMS = ('completed', 'velocity', 'window_days', 'forecast_low', 'forecast_high')
_EMPTY = {'labels': []}


def _label(minutes: float) -> str:
    """A UTC snapshot label *minutes* after the fixtures' base instant."""
    return (_BASE + timedelta(minutes=minutes)).isoformat()


def _served(minutes: float) -> datetime:
    return _BASE + timedelta(minutes=minutes)


_ALPHA = _nine_member_series([_label(0), _label(10), _label(20)], base=1)
_BRAVO = _nine_member_series([_label(0), _label(10)], base=1000)
_ZULU_UNPARSEABLE = _nine_member_series(['not-a-timestamp'], base=7)


def _eight_daily(first_done: int) -> dict:
    labels = [(_BASE + timedelta(days=d)).isoformat() for d in range(8)]
    return {'labels': labels, 'done': [first_done + 2 * d for d in range(8)],
            'pending': [30 - d for d in range(8)]}


def _served_datums(body: dict) -> dict:
    """Every Datum a burndown payload serves, keyed by where it sits."""
    blocks = {'BURNDOWN': body['BURNDOWN']}
    blocks.update({f'BURNDOWN_BY_PROJECT[{pid}]': block
                   for pid, block in body['BURNDOWN_BY_PROJECT'].items()})
    return {f'{where}.{field}': block[field]
            for where, block in blocks.items() for field in ('latest', 'forecast')}


_PAYLOADS_OF_EVERY_STATE = {
    'fresh': ({'alpha': _ALPHA}, _served(25)),
    'carried': ({'alpha': _ALPHA, 'bravo': _BRAVO}, _served(25)),
    'empty-project': ({'alpha': _ALPHA, 'echo': _EMPTY}, _served(25)),
    'past-the-bound': ({'alpha': _ALPHA}, _served(20) + timedelta(seconds=_BOUND + 1)),
    'clock-skew': ({'alpha': _ALPHA}, _served(10)),
    'unparseable-label': ({'alpha': _ALPHA, 'zulu': _ZULU_UNPARSEABLE}, _served(25)),
    'forecastable': ({'alpha': _eight_daily(0), 'bravo': _eight_daily(50)}, _served(7 * 24 * 60)),
    'nothing-measured': ({'echo': _EMPTY}, _served(25)),
    'no-projects': ({}, _served(25)),
}


def test_shape_burndown_latest_is_fresh_for_a_project_measured_at_the_newest_sample():
    block = redux_api.shape_burndown({'alpha': _ALPHA}, served_at=_served(25))[
        'BURNDOWN_BY_PROJECT']['alpha']
    completion = burndown.compute_window_completion(_ALPHA)
    assert block['latest'] == {
        'value': {
            'counts': {key: _ALPHA[key][-1] for key in _WIRE_SERIES_KEYS},
            'completed': completion['completed'],
            'velocity': completion['velocity'],
            'window_days': completion['window_days'],
        },
        'as_of': _label(20),
        'state': 'fresh',
        'reason': None,
        'freshness_bound_seconds': _BOUND,
    }


def test_shape_burndown_latest_is_stale_for_a_project_carried_to_the_newest_sample():
    """bravo last measured at +10min; the newest sample is alpha's at +20min."""
    body = redux_api.shape_burndown({'alpha': _ALPHA, 'bravo': _BRAVO}, served_at=_served(25))
    latest = body['BURNDOWN_BY_PROJECT']['bravo']['latest']
    assert latest['state'] == 'stale'
    assert latest['as_of'] == _label(10)
    assert _label(20) in latest['reason']
    assert '600s' in latest['reason']
    assert latest['value']['counts'] == {key: _BRAVO[key][-1] for key in _WIRE_SERIES_KEYS}


def test_shape_burndown_latest_is_unknown_for_a_project_with_no_measured_row():
    body = redux_api.shape_burndown({'alpha': _ALPHA, 'echo': _EMPTY}, served_at=_served(25))
    latest = body['BURNDOWN_BY_PROJECT']['echo']['latest']
    assert latest['state'] == 'unknown'
    assert latest['value'] is None
    assert latest['as_of'] is None
    assert 'no measured burndown sample in this window' in latest['reason']


def test_shape_burndown_aggregate_latest_headline_is_the_spark_endpoint():
    """Sketch #6: the tile's number and the spark's last point are one number."""
    body = redux_api.shape_burndown({'alpha': _ALPHA, 'bravo': _BRAVO}, served_at=_served(25))
    agg = body['BURNDOWN']
    value = agg['latest']['value']
    assert value['counts'] == {key: agg[key][-1] for key in _WIRE_SERIES_KEYS}
    fold = burndown.aggregate_window_completion(
        {'alpha': burndown.compute_window_completion(_ALPHA),
         'bravo': burndown.compute_window_completion(_BRAVO)},
        agg['labels'],
    )
    assert {key: value[key] for key in ('completed', 'velocity', 'window_days')} == fold


def test_shape_burndown_aggregate_latest_is_stale_as_of_its_oldest_contribution():
    body = redux_api.shape_burndown({'alpha': _ALPHA, 'bravo': _BRAVO}, served_at=_served(25))
    latest = body['BURNDOWN']['latest']
    assert latest['state'] == 'stale'
    assert latest['as_of'] == _label(10)
    assert 'bravo' in latest['reason']
    assert 'alpha' not in latest['reason']


def test_shape_burndown_aggregate_latest_is_a_lower_bound_when_a_project_is_unmeasured():
    body = redux_api.shape_burndown({'alpha': _ALPHA, 'echo': _EMPTY}, served_at=_served(25))
    latest = body['BURNDOWN']['latest']
    assert latest['state'] == 'lower_bound'
    assert latest['value'] is not None
    assert 'echo' in latest['reason']


def test_shape_burndown_aggregate_latest_is_fresh_when_every_project_is_measured_at_the_newest():
    both = {'alpha': _ALPHA, 'bravo': _nine_member_series([_label(5), _label(20)], base=500)}
    latest = redux_api.shape_burndown(both, served_at=_served(25))['BURNDOWN']['latest']
    assert latest['state'] == 'fresh'
    assert latest['reason'] is None
    assert latest['as_of'] == _label(20)


def test_shape_burndown_latest_goes_stale_past_the_freshness_bound():
    body = redux_api.shape_burndown(
        {'alpha': _ALPHA}, served_at=_served(20) + timedelta(seconds=_BOUND + 1),
    )
    for latest in (body['BURNDOWN']['latest'], body['BURNDOWN_BY_PROJECT']['alpha']['latest']):
        assert latest['state'] == 'stale'
        assert f'{_BOUND}s' in latest['reason']
        assert _label(20) in latest['reason']


def test_shape_burndown_latest_is_stale_when_the_newest_sample_postdates_the_serving_instant():
    """A sample stamped after served_at is doubted, never served as FRESH from the future."""
    body = redux_api.shape_burndown({'alpha': _ALPHA}, served_at=_served(10))
    for latest in (body['BURNDOWN']['latest'], body['BURNDOWN_BY_PROJECT']['alpha']['latest']):
        assert latest['state'] == 'stale'
        assert latest['as_of'] == _label(20)
        assert 'clock skew' in latest['reason']
        assert _label(20) in latest['reason']


def test_shape_burndown_an_unparseable_label_makes_every_measured_block_unknown():
    """With one newest label unreadable no block can say how old it is, so none claims to know."""
    body = redux_api.shape_burndown(
        {'alpha': _ALPHA, 'zulu': _ZULU_UNPARSEABLE}, served_at=_served(25),
    )
    for where, datum in _served_datums(body).items():
        assert datum['state'] == 'unknown', where
        assert datum['value'] is None, where
        assert "'not-a-timestamp'" in datum['reason'], where


def test_shape_burndown_aggregate_latest_is_unknown_when_nothing_was_measured():
    for series in ({'echo': _EMPTY}, {}):
        latest = redux_api.shape_burndown(series, served_at=_served(25))['BURNDOWN']['latest']
        assert latest['state'] == 'unknown'
        assert latest['value'] is None
        assert latest['reason']


def test_shape_burndown_forecast_datum_carries_its_blocks_latest_provenance():
    series = {'alpha': _eight_daily(0), 'bravo': _eight_daily(50)}
    body = redux_api.shape_burndown(series, served_at=_served(7 * 24 * 60))

    blocks = [(body['BURNDOWN'], burndown.aggregate_forecast_confidence(series.values()))]
    blocks += [(body['BURNDOWN_BY_PROJECT'][pid], burndown.compute_forecast_confidence(s))
               for pid, s in series.items()]
    for block, expected in blocks:
        forecast, latest = block['forecast'], block['latest']
        assert expected['forecast_low'] is not None
        assert forecast['value'] == expected
        assert (forecast['as_of'], forecast['state'], forecast['reason']) == (
            latest['as_of'], latest['state'], latest['reason'],
        )


def test_shape_burndown_forecast_datum_is_unknown_on_sparse_history():
    body = redux_api.shape_burndown({'alpha': _ALPHA}, served_at=_served(25))
    for block in (body['BURNDOWN'], body['BURNDOWN_BY_PROJECT']['alpha']):
        assert block['forecast']['state'] == 'unknown'
        assert block['forecast']['value'] is None
        assert block['forecast']['reason']


@pytest.mark.parametrize('case', _PAYLOADS_OF_EVERY_STATE)
def test_shape_burndown_every_datum_declares_two_sample_intervals(case):
    """PRD open question 1: burndown data is sampled, so its bound is two intervals."""
    series, served_at = _PAYLOADS_OF_EVERY_STATE[case]
    for where, datum in _served_datums(redux_api.shape_burndown(series, served_at=served_at)).items():
        assert set(datum) == _DATUM_KEYS, where
        assert datum['freshness_bound_seconds'] == _BOUND, where


@pytest.mark.parametrize('case', _PAYLOADS_OF_EVERY_STATE)
def test_shape_burndown_every_datum_keeps_the_wire_triad(case):
    """unknown <=> no value <=> no as_of; anything but fresh says why."""
    series, served_at = _PAYLOADS_OF_EVERY_STATE[case]
    for where, datum in _served_datums(redux_api.shape_burndown(series, served_at=served_at)).items():
        unknown = datum['state'] == 'unknown'
        assert unknown == (datum['value'] is None) == (datum['as_of'] is None), where
        if datum['state'] != 'fresh':
            assert isinstance(datum['reason'], str) and datum['reason'].strip(), where


def test_shape_burndown_reasons_do_not_move_with_the_serving_instant():
    """A payload over data already past the bound is identical one second later."""
    served_at = _served(20) + timedelta(seconds=_BOUND + 60)
    series = {'alpha': _ALPHA, 'bravo': _BRAVO, 'echo': _EMPTY}
    assert redux_api.shape_burndown(series, served_at=served_at) == redux_api.shape_burndown(
        series, served_at=served_at + timedelta(seconds=1),
    )


def test_shape_burndown_per_project_blocks_carry_completed_per_day():
    series = {'alpha': _eight_daily(0), 'bravo': _BRAVO}
    body = redux_api.shape_burndown(series, served_at=_served(25))
    for pid, s in series.items():
        assert body['BURNDOWN_BY_PROJECT'][pid]['completed_per_day'] == (
            burndown.compute_window_completion(s)['completed_per_day']
        )


def test_shape_burndown_blocks_carry_no_flat_copy_of_a_datum_value():
    body = redux_api.shape_burndown({'alpha': _ALPHA, 'bravo': _BRAVO}, served_at=_served(25))
    for block in (body['BURNDOWN'], *body['BURNDOWN_BY_PROJECT'].values()):
        assert not set(_FLAT_FIELDS_NOW_IN_DATUMS) & set(block)


# ---------------------------------------------------------------------------
# shape_burndown — live/stranded split + parity alarm (task 3543)
# ---------------------------------------------------------------------------


def _split_series(
    labels: list[str],
    in_progress: list[int],
    live: list[int],
    stranded: list[int],
    caps: list[int | None],
) -> dict:
    """A burndown series in the post-split shape ``get_burndown_series`` emits."""
    n = len(labels)
    return {
        'labels': labels,
        'done': [0] * n,
        'in_progress': in_progress,
        'in_progress_live': live,
        'in_progress_stranded': stranded,
        'blocked': [0] * n,
        'pending': [0] * n,
        'concurrency_cap': caps,
    }


def test_burndown_keys_include_the_live_stranded_split():
    """The split rides the summed-key loop, so both blocks carry it.

    ``concurrency_cap`` must stay OUT of that loop — it is a per-project
    scalar series, not an additive count, and summing caps across projects
    produces a denominator no single orchestrator ever enforced.
    """
    assert 'in_progress_live' in redux_api._BURNDOWN_KEYS
    assert 'in_progress_stranded' in redux_api._BURNDOWN_KEYS
    assert 'concurrency_cap' not in redux_api._BURNDOWN_KEYS


def test_shape_burndown_carries_split_and_conserves_in_progress():
    """live + stranded == in_progress on every label, per-project AND aggregate.

    This is the load-bearing invariant: the split exists to be trusted, and a
    stacked chart whose two bands do not add up to the band they replaced is
    worse than no split at all.
    """
    labels = ['2026-08-01T00:00:00', '2026-08-02T00:00:00']
    series = {
        'dark_factory': _split_series(labels, [5, 4], [3, 4], [2, 0], [24, 24]),
        'reify': _split_series(labels, [2, 7], [2, 5], [0, 2], [12, 12]),
    }
    body = redux_api.shape_burndown(series)

    df = body['BURNDOWN_BY_PROJECT']['dark_factory']
    assert df['in_progress_live'] == [3, 4]
    assert df['in_progress_stranded'] == [2, 0]

    agg = body['BURNDOWN']
    assert agg['in_progress'] == [7, 11]
    assert agg['in_progress_live'] == [5, 9]
    assert agg['in_progress_stranded'] == [2, 2]
    for live, stranded, total in zip(
        agg['in_progress_live'], agg['in_progress_stranded'], agg['in_progress'], strict=True
    ):
        assert live + stranded == total


def test_shape_burndown_split_defaults_to_all_live_when_absent():
    """A series with no split (un-migrated DB) degrades to all-live, not to zeros.

    ``get_burndown_series`` already degrades an un-migrated peer DB this way;
    shaping must not re-break conservation by contributing 0 to both bands
    while the ``in_progress`` band it splits stays fully populated.
    """
    series = {
        'legacy': {
            'labels': ['D-1', 'D-0'],
            'done': [1, 2], 'in_progress': [4, 6], 'blocked': [0, 0], 'pending': [3, 2],
        },
    }
    body = redux_api.shape_burndown(series)

    legacy = body['BURNDOWN_BY_PROJECT']['legacy']
    assert legacy['in_progress_live'] == [4, 6]
    assert legacy['in_progress_stranded'] == [0, 0]

    agg = body['BURNDOWN']
    assert agg['in_progress_live'] == [6, 4]  # labels sort to ['D-0', 'D-1']
    assert agg['in_progress_stranded'] == [0, 0]
    for live, stranded, total in zip(
        agg['in_progress_live'], agg['in_progress_stranded'], agg['in_progress'], strict=True
    ):
        assert live + stranded == total


def test_shape_burndown_per_project_carries_parity_block():
    """Every per-project block carries compute_parity_alarm's four fields."""
    labels = ['2026-08-01T00:00:00', '2026-08-02T00:00:00']
    series = {
        'dark_factory': _split_series(labels, [36, 20], [33, 20], [3, 0], [24, 24]),
        'reify': _split_series(labels, [2, 3], [2, 3], [0, 0], [100, 100]),
    }
    body = redux_api.shape_burndown(series)

    df = body['BURNDOWN_BY_PROJECT']['dark_factory']
    assert df['parity_alarm'] is True
    assert df['parity_cap'] == 24
    assert df['parity_peak'] == 33
    assert df['parity_breach_count'] == 1

    ri = body['BURNDOWN_BY_PROJECT']['reify']
    assert ri['parity_alarm'] is False
    assert ri['parity_breach_count'] == 0


def test_shape_burndown_aggregate_parity_ors_projects_not_summed_counts():
    """A real breach must survive aggregation — summing would hide it.

    dark_factory peaks at 33 against a cap of 24 (the live E12 shape).  reify
    idles at 3 against a cap of 100.  Summed, that is 35 in-progress against a
    124 "cap" — comfortably healthy, and the breach vanishes.  The aggregate
    alarm is therefore an OR over per-project alarms, and it names which
    project breached so an operator is not left hunting.
    """
    labels = ['2026-08-01T00:00:00', '2026-08-02T00:00:00']
    series = {
        'dark_factory': _split_series(labels, [36, 20], [33, 20], [3, 0], [24, 24]),
        'reify': _split_series(labels, [2, 3], [2, 3], [0, 0], [100, 100]),
    }
    agg = redux_api.shape_burndown(series)['BURNDOWN']

    assert agg['parity_alarm'] is True
    assert agg['parity_projects'] == ['dark_factory']
    assert agg['parity_breach_count'] == 1
    # peak and cap travel as a matched pair from the breaching project — a peak
    # from one project beside a cap from another explains nothing.
    assert agg['parity_peak'] == 33
    assert agg['parity_cap'] == 24


def test_shape_burndown_parity_ignores_a_series_with_no_split():
    """The alarm reads the RAW series, never ``_with_split``'s census-filled copy.

    The display block still fills the missing split with the census so the
    stacked chart conserves, but that fill is not a live measurement and must
    not be compared against the cap.
    """
    labels = ['2026-08-01T00:00:00', '2026-08-02T00:00:00']
    series = {
        'legacy': {
            'labels': labels,
            'done': [0, 0], 'blocked': [0, 0], 'pending': [0, 0],
            'in_progress': [30, 30],
            'concurrency_cap': [24, 24],
        },
    }
    body = redux_api.shape_burndown(series)

    legacy = body['BURNDOWN_BY_PROJECT']['legacy']
    assert legacy['parity_alarm'] is False
    assert legacy['parity_peak'] is None
    assert legacy['in_progress_live'] == [30, 30]
    assert body['BURNDOWN']['parity_alarm'] is False


def test_shape_burndown_aggregate_parity_ignores_capless_projects():
    """A NULL-cap project must not deflate a summed denominator into a fake alarm.

    Nothing breaches here: dark_factory runs 20 under a cap of 24, and reify's
    30 is uncapped (unknown, not a breach).  A summed comparison would read 50
    in-progress against the only known cap, 24, and alarm on fiction.
    """
    labels = ['2026-08-01T00:00:00']
    series = {
        'dark_factory': _split_series(labels, [20], [20], [0], [24]),
        'reify': _split_series(labels, [30], [30], [0], [None]),
    }
    body = redux_api.shape_burndown(series)

    assert body['BURNDOWN_BY_PROJECT']['reify']['parity_cap'] is None
    assert body['BURNDOWN_BY_PROJECT']['reify']['parity_alarm'] is False

    agg = body['BURNDOWN']
    assert agg['parity_alarm'] is False
    assert agg['parity_projects'] == []
    assert agg['parity_breach_count'] == 0
    # No breach to describe — peak/cap are None rather than a healthy-looking
    # pair that implies the aggregate was measured against a cap it never had.
    assert agg['parity_peak'] is None
    assert agg['parity_cap'] is None


def test_shape_burndown_aggregate_parity_reports_the_worst_breach():
    """With two breaching projects the aggregate names both and shows the worst."""
    labels = ['2026-08-01T00:00:00', '2026-08-02T00:00:00']
    series = {
        'dark_factory': _split_series(labels, [33, 30], [33, 30], [0, 0], [24, 24]),
        'reify': _split_series(labels, [9, 4], [9, 4], [0, 0], [8, 8]),
        'quiet': _split_series(labels, [1, 1], [1, 1], [0, 0], [16, 16]),
    }
    agg = redux_api.shape_burndown(series)['BURNDOWN']

    assert agg['parity_alarm'] is True
    assert agg['parity_projects'] == ['dark_factory', 'reify']  # sorted, quiet excluded
    assert agg['parity_breach_count'] == 3  # 2 from dark_factory + 1 from reify
    assert agg['parity_peak'] == 33  # widest margin (33 - 24 = 9, vs reify's 9 - 8 = 1)
    assert agg['parity_cap'] == 24


def test_shape_burndown_aggregate_has_no_summed_concurrency_cap():
    """The aggregate must not publish a summed cap — no orchestrator enforces it."""
    labels = ['2026-08-01T00:00:00']
    series = {
        'dark_factory': _split_series(labels, [1], [1], [0], [24]),
        'reify': _split_series(labels, [1], [1], [0], [12]),
    }
    agg = redux_api.shape_burndown(series)['BURNDOWN']
    assert 'concurrency_cap' not in agg


# ---------------------------------------------------------------------------
# shape_escalations — task cards and views as served Datums (task 5596)
# ---------------------------------------------------------------------------

ESC_SERVED_AT = datetime(2026, 10, 1, 12, 0, 30, tzinfo=UTC)
ESC_MEASURED_AT = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)

_EMPTY_SUMMARY = {
    'by_level': {0: 0, 1: 0, 2: 0},
    'by_status': {'pending': 0, 'resolved': 0, 'dismissed': 0},
    'skipped_count': 0,
}


def _count(value: int, *, state: DatumState = DatumState.FRESH,
           reason: str | None = None, bound: int = 120) -> Datum:
    return Datum(value, ESC_MEASURED_AT, state, reason, bound)


def _views(queue_pending: int = 0, open_in_history: int = 0) -> dict:
    return {
        EscalationView.QUEUE_PENDING: _count(queue_pending),
        EscalationView.OPEN_IN_HISTORY: _count(open_in_history),
    }


def _esc_row(esc_id: str = 'esc-1-1', **extra) -> dict:
    return {
        'id': esc_id, 'task_id': '1', 'level': 0, 'status': 'pending',
        'summary': 'oops', 'project': 'projA', 'project_root': '/p/projA', **extra,
    }


def _queues(*rows: dict, skipped: list | None = None) -> dict:
    return {
        'subsections': [{
            'id': '/p/projA', 'label': 'projA', 'kind': 'orchestrator',
            'escalations': list(rows), 'skipped': skipped or [],
            'summary': dict(_EMPTY_SUMMARY), 'views': _views(1, 3),
        }],
        'summary': dict(_EMPTY_SUMMARY),
        'views': _views(1, 3),
    }


def _card(value: dict | None = None, **kwargs) -> Datum:
    return Datum(value or {'id': 1, 'title': 'one', 'status': 'pending'},
                 ESC_MEASURED_AT, DatumState.FRESH, None, kwargs.get('bound', 1200))


class TestShapeEscalations:
    """Rows carry their task card, subsections and the top level carry views — all wired Datums."""

    def test_the_payload_carries_served_at(self):
        body = redux_api.shape_escalations(_queues(), {}, served_at=ESC_SERVED_AT)

        assert set(body) == {'ESCALATIONS', 'served_at'}
        assert body['served_at'] == ESC_SERVED_AT.isoformat()

    def test_a_row_task_is_its_wired_card(self):
        card = _card()
        body = redux_api.shape_escalations(
            _queues(_esc_row()), {('/p/projA', 'esc-1-1'): card}, served_at=ESC_SERVED_AT,
        )
        (row,) = body['ESCALATIONS']['subsections'][0]['escalations']

        assert row['task'] == card.to_wire()
        assert row['id'] == 'esc-1-1' and row['summary'] == 'oops'
        assert row['project'] == 'projA'
        assert 'task_unresolved' not in row
        assert 'project_root' not in row

    def test_a_fresh_card_past_its_bound_arrives_stale(self):
        card = _card(bound=10)
        body = redux_api.shape_escalations(
            _queues(_esc_row()), {('/p/projA', 'esc-1-1'): card}, served_at=ESC_SERVED_AT,
        )
        (row,) = body['ESCALATIONS']['subsections'][0]['escalations']

        assert row['task']['state'] == 'stale'
        assert row['task']['value'] == card.value
        assert row['task']['reason']

    def test_an_unknown_card_renders_its_reason(self):
        card = unknown_datum('no task id', 1200)
        body = redux_api.shape_escalations(
            _queues(_esc_row()), {('/p/projA', 'esc-1-1'): card}, served_at=ESC_SERVED_AT,
        )
        (row,) = body['ESCALATIONS']['subsections'][0]['escalations']

        assert row['task']['value'] is None
        assert row['task']['state'] == 'unknown'
        assert row['task']['reason'] == 'no task id'

    def test_a_row_without_a_card_is_a_wiring_bug(self):
        with pytest.raises(DatumContractError) as raised:
            redux_api.shape_escalations(_queues(_esc_row()), {}, served_at=ESC_SERVED_AT)

        assert raised.value.invariant is DatumInvariant.DATUM_REQUIRED

    def test_subsection_and_top_level_views_are_wired(self):
        body = redux_api.shape_escalations(_queues(), {}, served_at=ESC_SERVED_AT)
        esc = body['ESCALATIONS']

        for views in (esc['views'], esc['subsections'][0]['views']):
            assert set(views) == {'queue_pending', 'open_in_history'}
            assert views['queue_pending'] == _count(1).to_wire()
            assert views['open_in_history'] == _count(3).to_wire()

    def test_a_lower_bound_view_keeps_its_reason(self):
        queues = _queues()
        queues['views'][EscalationView.OPEN_IN_HISTORY] = _count(
            3, state=DatumState.LOWER_BOUND, reason='partial scan',
        )
        body = redux_api.shape_escalations(queues, {}, served_at=ESC_SERVED_AT)

        assert body['ESCALATIONS']['views']['open_in_history']['state'] == 'lower_bound'
        assert body['ESCALATIONS']['views']['open_in_history']['reason'] == 'partial scan'

    def test_a_missing_view_is_a_wiring_bug(self):
        queues = _queues()
        queues['subsections'][0]['views'] = {}

        with pytest.raises(DatumContractError) as raised:
            redux_api.shape_escalations(queues, {}, served_at=ESC_SERVED_AT)

        assert raised.value.invariant is DatumInvariant.DATUM_REQUIRED

    def test_summary_skipped_and_metadata_pass_through_as_copies(self):
        skipped = [{'path': '/p/projA/data/escalations/esc-bad.json', 'error': 'boom',
                    'location': 'root'}]
        queues = _queues(skipped=skipped)
        queues['summary'] = {**_EMPTY_SUMMARY, 'skipped_count': 1}
        body = redux_api.shape_escalations(queues, {}, served_at=ESC_SERVED_AT)
        sub = body['ESCALATIONS']['subsections'][0]

        assert (sub['id'], sub['label'], sub['kind']) == ('/p/projA', 'projA', 'orchestrator')
        assert sub['skipped'] == skipped
        assert sub['skipped'] is not skipped and sub['skipped'][0] is not skipped[0]
        assert sub['summary'] == _EMPTY_SUMMARY
        assert body['ESCALATIONS']['summary']['skipped_count'] == 1

    def test_one_esc_id_in_two_queues_reads_two_cards(self):
        queues = _queues(_esc_row('esc-101-1'))
        queues['subsections'].append({
            'id': 'reconciliation', 'label': 'fused-memory', 'kind': 'reconciliation',
            'escalations': [_esc_row('esc-101-1', project=None, project_root=None)],
            'skipped': [], 'summary': dict(_EMPTY_SUMMARY), 'views': _views(),
        })
        cards = {
            ('/p/projA', 'esc-101-1'): _card(),
            ('reconciliation', 'esc-101-1'): unknown_datum('no owning project', 1200),
        }
        body = redux_api.shape_escalations(queues, cards, served_at=ESC_SERVED_AT)
        subs = body['ESCALATIONS']['subsections']

        assert subs[0]['escalations'][0]['task']['state'] == 'fresh'
        assert subs[1]['escalations'][0]['task']['reason'] == 'no owning project'


class TestShapeEscalationAnalytics:
    """The analytics payload passes through, its views wired at served_at."""

    def _payload(self) -> dict:
        return {
            'generated_at': ESC_MEASURED_AT.isoformat(),
            'parse_failures': 0,
            'regime_markers': [],
            'per_project': [{'project': 'projA', 'terminal': 0, 'origin': {}, 'lifespan': {},
                             'workflow': {}, 'views': _views(2, 5)}],
            'archives_present': True,
            'archives_reached': True,
            'views': _views(2, 5),
        }

    def test_views_are_wired_and_the_rest_passes_through(self):
        payload = self._payload()

        body = redux_api.shape_escalation_analytics(payload, served_at=ESC_SERVED_AT)
        shaped = body['ESCALATION_ANALYTICS']

        assert body['served_at'] == ESC_SERVED_AT.isoformat()
        for views in (shaped['views'], shaped['per_project'][0]['views']):
            assert views == {
                'queue_pending': _count(2).to_wire(),
                'open_in_history': _count(5).to_wire(),
            }
        assert shaped['per_project'][0]['project'] == 'projA'
        for key in ('generated_at', 'parse_failures', 'regime_markers',
                    'archives_present', 'archives_reached'):
            assert shaped[key] == payload[key]

    def test_a_missing_view_is_a_wiring_bug(self):
        payload = self._payload()
        payload['per_project'][0]['views'] = {}

        with pytest.raises(DatumContractError) as raised:
            redux_api.shape_escalation_analytics(payload, served_at=ESC_SERVED_AT)

        assert raised.value.invariant is DatumInvariant.DATUM_REQUIRED


# ---------------------------------------------------------------------------
# shape_merge_queue — served Datums: "In queue now" and row titles (task 5595)
# ---------------------------------------------------------------------------


def _mq_project(**overrides) -> dict:
    """A minimal per-project aggregate entry; *overrides* replace its keys."""
    data: dict = {
        'depth_timeseries': {'labels': [], 'values': []},
        'outcomes': {'labels': [], 'values': []},
        'latency': {},
        'recent': [],
        'speculative': {},
        'active': [],
        **_MQ_MEASURED_QUEUE,
        'train_events': [],
    }
    data.update(overrides)
    return data


def _shaped(**overrides) -> dict:
    body = redux_api.shape_merge_queue(
        {'/proj/myproj': _mq_project(**overrides)}, served_at=MQ_SERVED_AT,
    )
    return body['MERGE_QUEUE']['myproj']


class TestShapeMergeQueueServedDatums:
    """Every datum on /merge-queue is aged, validated and rendered at served_at."""

    def test_the_payload_carries_served_at(self):
        body = redux_api.shape_merge_queue(
            {'/proj/myproj': _mq_project()}, served_at=MQ_SERVED_AT,
        )

        assert body['served_at'] == MQ_SERVED_AT.isoformat()

    def test_a_fresh_in_queue_is_a_wire_datum(self):
        in_queue = Datum(2, MQ_SERVED_AT - timedelta(seconds=5), DatumState.FRESH, None, 30)

        assert _shaped(in_queue=in_queue)['in_queue'] == in_queue.to_wire()

    def test_a_fresh_in_queue_past_its_bound_is_served_stale(self):
        as_of = MQ_SERVED_AT - timedelta(seconds=45)

        wire = _shaped(in_queue=Datum(2, as_of, DatumState.FRESH, None, 30))['in_queue']

        assert wire['state'] == 'stale'
        assert (wire['value'], wire['as_of']) == (2, as_of.isoformat())
        assert '30s freshness bound' in wire['reason']

    def test_a_stale_in_queue_keeps_its_own_reason(self):
        in_queue = Datum(1, MQ_SERVED_AT - timedelta(minutes=10), DatumState.STALE,
                         'connect refused', 30)

        wire = _shaped(in_queue=in_queue)['in_queue']

        assert (wire['state'], wire['reason']) == ('stale', 'connect refused')

    def test_a_project_without_an_in_queue_datum_is_a_wiring_bug(self):
        project = _mq_project()
        del project['in_queue']

        with pytest.raises(DatumContractError) as excinfo:
            redux_api.shape_merge_queue({'/proj/myproj': project}, served_at=MQ_SERVED_AT)

        assert excinfo.value.invariant is DatumInvariant.DATUM_REQUIRED
        assert 'myproj' in str(excinfo.value)

    def test_every_row_title_is_a_wire_datum(self):
        found = Datum('Fix X', MQ_SERVED_AT - timedelta(seconds=5), DatumState.FRESH, None, 1200)
        unread = Datum(None, None, DatumState.UNKNOWN, 'lookup budget: past the cap', 1200)

        section = _shaped(
            recent=[{'task_id': '7', 'title': found}],
            active=[{'task_id': '8', 'title': unread}],
        )

        assert section['recent'] == [{'task_id': '7', 'title': found.to_wire()}]
        assert section['active'] == [{'task_id': '8', 'title': unread.to_wire()}]

    @pytest.mark.parametrize('table', ['recent', 'active'])
    @pytest.mark.parametrize('row', [{'task_id': '7'}, {'task_id': '7', 'title': ''}])
    def test_a_row_without_a_title_datum_is_a_wiring_bug(self, table, row):
        with pytest.raises(DatumContractError) as excinfo:
            _shaped(**{table: [row]})

        assert excinfo.value.invariant is DatumInvariant.DATUM_REQUIRED
        assert table in str(excinfo.value)

    def test_latency_carries_the_timed_and_untimed_split(self):
        latency = {'p50': 100, 'p95': 200, 'p99': 300, 'mean_ms': 150.0,
                   'with_duration': 3, 'without_duration': 2}

        assert _shaped(latency=latency)['latency'] == latency

    @pytest.mark.parametrize('configured', [True, False])
    def test_live_probe_configured_is_carried_verbatim(self, configured):
        assert _shaped(live_probe_configured=configured)['live_probe_configured'] is configured

    def test_there_is_no_active_approximate_key(self):
        assert 'active_approximate' not in _shaped(active_approximate=True)

    def test_a_broken_datum_is_a_shaper_bug_not_a_degraded_payload(self):
        broken = Datum(None, MQ_SERVED_AT, DatumState.FRESH, None, 30)

        with pytest.raises(DatumContractError):
            _shaped(in_queue=broken)


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
            **_MQ_MEASURED_QUEUE,
            'train_events': [],
            'train_throughput': throughput_payload,
        },
    }
    body = redux_api.shape_merge_queue(raw, served_at=MQ_SERVED_AT)
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
            **_MQ_MEASURED_QUEUE,
            'train_events': [],
            # no 'train_throughput' key
        },
    }
    body2 = redux_api.shape_merge_queue(raw_no_throughput, served_at=MQ_SERVED_AT)
    assert body2['MERGE_QUEUE']['dark-factory']['train_throughput'] == {}


# ---------------------------------------------------------------------------
# shape_merge_queue — live metrics passthrough (step-09 RED / step-10 GREEN)
# ---------------------------------------------------------------------------


def _mq_project_base() -> dict:
    """Minimal per-project data dict for shape_merge_queue tests."""
    return {
        'depth_timeseries': {'labels': [], 'values': []},
        'outcomes': {'labels': [], 'values': []},
        'latency': {},
        'recent': [],
        'speculative': {'hit_rate': 0.0},
        'active': [],
        **_MQ_MEASURED_QUEUE,
    }


_LIVE_METRICS = {
    'retries_per_landing': 1.5,
    'drift_at_detection': {'count': 2, 'last': 3, 'mean': 2.5, 'max': 3},
    'landings_total': 2,
    'retries_total': 3,
}


def test_shape_merge_queue_surfaces_live_metrics():
    """shape_merge_queue emits 'metrics' from live_metrics per project.

    RED until step-10 GREEN adds 'metrics': dict(data.get('live_metrics') or {})
    to each out[label] in shape_merge_queue.
    """
    proj = _mq_project_base()
    proj['live_metrics'] = _LIVE_METRICS
    raw = {'/home/leo/src/dark-factory': proj}
    body = redux_api.shape_merge_queue(raw, served_at=MQ_SERVED_AT)
    section = body['MERGE_QUEUE']['dark-factory']
    assert 'metrics' in section, (
        f"shape_merge_queue must emit 'metrics' key per project; "
        f"got keys: {list(section.keys())}"
    )
    assert section['metrics'] == _LIVE_METRICS


def test_shape_merge_queue_metrics_defaults_to_empty_when_absent():
    """When live_metrics is absent, 'metrics' defaults to {} (no KeyError)."""
    proj = _mq_project_base()
    # no 'live_metrics' key
    raw = {'/home/leo/src/dark-factory': proj}
    body = redux_api.shape_merge_queue(raw, served_at=MQ_SERVED_AT)
    section = body['MERGE_QUEUE']['dark-factory']
    assert 'metrics' in section, (
        f"'metrics' key must always be present in shaped output; "
        f"got keys: {list(section.keys())}"
    )
    assert section['metrics'] == {}


def test_shape_merge_queue_metrics_defaults_to_empty_when_none():
    """When live_metrics is None, 'metrics' defaults to {} (safe default)."""
    proj = _mq_project_base()
    proj['live_metrics'] = None
    raw = {'/home/leo/src/dark-factory': proj}
    body = redux_api.shape_merge_queue(raw, served_at=MQ_SERVED_AT)
    section = body['MERGE_QUEUE']['dark-factory']
    assert section['metrics'] == {}


def test_shape_merge_queue_metrics_rpl_value():
    """retries_per_landing value is threaded through correctly."""
    proj = _mq_project_base()
    proj['live_metrics'] = {'retries_per_landing': 2.0, 'drift_at_detection': {'last': 5}}
    raw = {'/home/leo/src/dark-factory': proj}
    body = redux_api.shape_merge_queue(raw, served_at=MQ_SERVED_AT)
    assert body['MERGE_QUEUE']['dark-factory']['metrics']['retries_per_landing'] == 2.0
