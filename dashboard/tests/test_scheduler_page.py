"""Tests for dashboard /scheduler page — data layer and HTTP routes.

Step-by-step TDD (matching plan.json):
  step-1   test_module_contention_counts_sorted_desc
  step-3   test_compose_rows_joins_active_tasks_with_snapshot
  step-5   test_skip_event_sparkline_bins / test_skip_event_sparkline_empty
  step-7   test_collect_scheduler_state_happy_path
  step-9   test_collect_scheduler_state_surfaces_offline
  step-11  test_shape_scheduler_envelope
  step-13  test_scheduler_endpoint_returns_envelope_shape
  step-15  test_override_endpoint_rejects_invalid_body
  step-17  test_override_endpoint_proxies_verbatim / _returns_502
  step-19  test_clear_override_endpoint_validation_and_proxy
  step-21  test_reorder_pin_queue_endpoint_validation_and_proxy
  step-23  test_index_html_references_new_jsx_files
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest


# ---------------------------------------------------------------------------
# step-1: pure-function test for _module_contention_counts
# ---------------------------------------------------------------------------


def test_module_contention_counts_sorted_desc():
    """_module_contention_counts returns list sorted by contention desc, ties alpha.

    Three tasks with overlapping lock sets {a,b}, {b,c}, {b,d}.
    'b' appears in all three → contention 3.
    'a', 'c', 'd' appear once each → contention 1, ties broken alphabetically.
    Expected order: b(3), a(1), c(1), d(1).
    """
    from dashboard.data.scheduler import _module_contention_counts

    rows = [
        {'task_id': 'T1', 'lock_set': ['a', 'b']},
        {'task_id': 'T2', 'lock_set': ['b', 'c']},
        {'task_id': 'T3', 'lock_set': ['b', 'd']},
    ]
    current_holders = {'b': 'T1'}  # T1 holds 'b'

    result = _module_contention_counts(rows, current_holders)

    assert isinstance(result, list)
    assert len(result) == 4

    # b must be first with contention 3 and the correct holder
    assert result[0]['path'] == 'b'
    assert result[0]['contention'] == 3
    assert result[0]['holder'] == 'T1'

    # a, c, d follow with contention 1 (alphabetical for ties), no holder
    assert [r['path'] for r in result[1:]] == ['a', 'c', 'd']
    assert all(r['contention'] == 1 for r in result[1:])
    assert all(r['holder'] is None for r in result[1:])


# ---------------------------------------------------------------------------
# step-3: _compose_rows joins active tasks with snapshot
# ---------------------------------------------------------------------------


def test_compose_rows_joins_active_tasks_with_snapshot():
    """_compose_rows produces correctly joined rows with priority_differs flag.

    Feed a synthetic active-tasks list (with ``locks`` and ``priority``) and a
    synthetic snapshot dict; assert each row has all expected fields, and that
    ``priority_differs=True`` when effective_priority != declared priority.
    """
    from dashboard.data.scheduler import _compose_rows

    now_iso = '2026-01-01T12:00:00+00:00'

    active_tasks = [
        {
            'id': 'proj/T-1',
            'task_id': '1',
            'title': 'Task One',
            'priority': 'high',
            'status': 'in-progress',
            'started': 10,  # minutes
            'locks': ['src/a.py', 'src/b.py'],
        },
        {
            'id': 'proj/T-2',
            'task_id': '2',
            'title': 'Task Two',
            'priority': 'medium',
            'status': 'pending',
            'started': 5,
            'locks': ['src/c.py'],
        },
    ]

    snapshot = {
        'skip_counts': {'1': 3, '2': 0},
        'parks': {
            '1': {
                'modules': ['src/a.py'],
                'installed_at': now_iso,
            }
        },
        'effective_priorities': {
            '1': 'critical',  # differs from 'high'
            '2': 'medium',    # same as declared
        },
        'overrides': {
            '1': {
                'boost_tier': 'critical',
                'pinned': False,
                'reserve_now': True,
                'ttl_until': None,
            }
        },
        'current_holders': {'src/a.py': '1'},
        'pin_queue': [],
    }

    rows = _compose_rows(active_tasks, snapshot)

    assert len(rows) == 2

    # Required fields on every row
    required_fields = {
        'task_id', 'title', 'declared_priority', 'effective_priority',
        'priority_differs', 'skip_count', 'park_state', 'age_seconds',
        'lock_set', 'pinned', 'reserve_now', 'boost_tier', 'ttl_until',
    }
    for row in rows:
        assert required_fields <= set(row.keys()), (
            f'Row missing fields: {required_fields - set(row.keys())}'
        )

    # Row 1: priority_differs=True (critical vs high), parked, skip_count=3
    r1 = rows[0]
    assert r1['task_id'] == '1'
    assert r1['title'] == 'Task One'
    assert r1['declared_priority'] == 'high'
    assert r1['effective_priority'] == 'critical'
    assert r1['priority_differs'] is True
    assert r1['skip_count'] == 3
    assert r1['park_state'] is not None
    assert r1['age_seconds'] >= 0        # installed_at = now, so ~0
    assert r1['lock_set'] == ['src/a.py', 'src/b.py']
    assert r1['reserve_now'] is True
    assert r1['boost_tier'] == 'critical'
    assert r1['pinned'] is False

    # Row 2: priority_differs=False (medium == medium), not parked, age from started
    r2 = rows[1]
    assert r2['task_id'] == '2'
    assert r2['declared_priority'] == 'medium'
    assert r2['effective_priority'] == 'medium'
    assert r2['priority_differs'] is False
    assert r2['skip_count'] == 0
    assert r2['park_state'] is None
    assert r2['age_seconds'] == 5 * 60  # 5 minutes → 300 seconds
    assert r2['lock_set'] == ['src/c.py']
    assert r2['pinned'] is False
    assert r2['reserve_now'] is False
    assert r2['boost_tier'] is None


# ---------------------------------------------------------------------------
# step-5: _skip_event_sparkline — binning and empty-history guard
# ---------------------------------------------------------------------------


def test_skip_event_sparkline_bins_into_five_minute_buckets():
    """_skip_event_sparkline bins task_skipped events into 5-minute buckets.

    Setup: 60-minute window starting at ``since``, three task_skipped events
    for T1 placed at minutes 5, 15, and 35.  Expect exactly 12 buckets (one
    per 5-minute slot), the three occupied buckets have count > 0, and a
    different event_type is ignored.
    """
    from dashboard.data.scheduler import _skip_event_sparkline

    since = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    until = since + timedelta(hours=1)  # fixed window → exactly 12 buckets

    def make_event(event_type: str, minutes_offset: float) -> dict:
        ts = since + timedelta(minutes=minutes_offset)
        return {'event_type': event_type, 'task_id': 'T1', 'timestamp': ts.isoformat()}

    events = [
        make_event('task_skipped', 5),    # bucket 1
        make_event('task_skipped', 15),   # bucket 3
        make_event('task_scheduled', 20), # should be ignored (wrong type)
        make_event('task_skipped', 35),   # bucket 7
    ]

    result = _skip_event_sparkline(events, since=since, until=until, bin_seconds=300)

    assert 'labels' in result
    assert 'values' in result
    assert len(result['labels']) == len(result['values'])
    # 60-minute window / 5-minute buckets = exactly 12 buckets
    assert len(result['labels']) == 12

    # Buckets at minutes 5, 15, 35 → bucket indices 1, 3, 7
    assert result['values'][1] == 1   # 5-min mark falls in bucket 1
    assert result['values'][3] == 1   # 15-min mark falls in bucket 3
    assert result['values'][7] == 1   # 35-min mark falls in bucket 7

    # Other buckets are zero
    other_buckets = [i for i in range(12) if i not in (1, 3, 7)]
    assert all(result['values'][i] == 0 for i in other_buckets)

    # Total skipped events counted
    assert sum(result['values']) == 3


def test_skip_event_sparkline_returns_empty_when_no_history():
    """Empty events list → {labels: [], values: []} (no PRNG fallback)."""
    from dashboard.data.scheduler import _skip_event_sparkline

    since = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    result = _skip_event_sparkline([], since=since)

    assert result == {'labels': [], 'values': []}


# ---------------------------------------------------------------------------
# step-7: collect_scheduler_state happy path
# ---------------------------------------------------------------------------


async def test_collect_scheduler_state_happy_path(dummy_client, dummy_config):
    """collect_scheduler_state returns correct shapes and joined fields.

    Patches:
      - mcp_tool_call → returns snapshot then events list
      - collect_active_tasks → returns one synthetic in-progress task
    Asserts the 5-tuple (rows, modules, pin_queue, events_by_task, offline_projects)
    has the right shapes and a couple of joined field values.
    """
    from unittest.mock import AsyncMock, patch

    from dashboard.data.scheduler import collect_scheduler_state

    project = dummy_config.project_root.name  # _project_label result

    snapshot = {
        'skip_counts': {'1': 2},
        'parks': {},
        'effective_priorities': {'1': 'high'},
        'pin_queue': [{'task_id': '1', 'order': 0}],
        'overrides': {
            '1': {
                'boost_tier': 'high',
                'pinned': True,
                'reserve_now': False,
                'ttl_until': None,
            }
        },
        'current_holders': {'src/a.py': '1'},
        'snapshot_at': '2026-01-01T00:00:00+00:00',
    }
    now_iso = datetime.now(UTC).isoformat()
    events = [
        {'event_type': 'task_skipped', 'task_id': '1', 'timestamp': now_iso},
    ]

    active_tasks = [
        {
            'id': f'{project}/T-1',
            'project': project,
            'title': 'Task One',
            'priority': 'medium',
            'status': 'in-progress',
            'started': 5,
            'locks': ['src/a.py', 'src/b.py'],
        }
    ]

    mock_mcp = AsyncMock(side_effect=[snapshot, events])
    mock_active = AsyncMock(return_value=(active_tasks, {}, []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', mock_mcp),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active),
    ):
        rows, modules, pin_queue, events_by_task, offline_projects = \
            await collect_scheduler_state(dummy_client, dummy_config)

    assert offline_projects == []
    assert len(rows) == 1

    # Joined fields
    assert rows[0]['lock_set'] == ['src/a.py', 'src/b.py']
    assert rows[0]['task_id'] == '1'
    assert rows[0]['pinned'] is True
    assert rows[0]['skip_count'] == 2

    # Module contention: both src/a.py and src/b.py appear once;
    # src/a.py sorts first (alpha tie-break) and carries the holder
    assert len(modules) == 2
    assert modules[0]['holder'] == '1'   # src/a.py is held by task '1'

    assert len(pin_queue) == 1
    assert pin_queue[0]['task_id'] == '1'

    # events_by_task for task '1' has a sparkline
    assert '1' in events_by_task
    assert 'labels' in events_by_task['1']
    assert 'values' in events_by_task['1']


# ---------------------------------------------------------------------------
# step-9: collect_scheduler_state surfaces offline when MCP unreachable
# ---------------------------------------------------------------------------


async def test_collect_scheduler_state_surfaces_offline_when_mcp_unreachable(
    dummy_client, dummy_config
):
    """When every fused-memory URL is unreachable, offline_projects is non-empty.

    Mirrors test_collect_active_tasks_surfaces_offline_projects.
    All other tuple members must be empty.
    """
    import httpx
    from unittest.mock import AsyncMock, patch

    from dashboard.data.scheduler import collect_scheduler_state

    project = dummy_config.project_root.name

    active_tasks = [
        {
            'id': f'{project}/T-1',
            'project': project,
            'title': 'Task One',
            'priority': 'medium',
            'status': 'in-progress',
            'started': 5,
            'locks': ['src/a.py'],
        }
    ]

    mock_mcp = AsyncMock(side_effect=httpx.ConnectError('refused'))
    mock_active = AsyncMock(return_value=(active_tasks, {}, []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', mock_mcp),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active),
    ):
        rows, modules, pin_queue, events_by_task, offline_projects = \
            await collect_scheduler_state(dummy_client, dummy_config)

    assert rows == []
    assert modules == []
    assert pin_queue == []
    assert events_by_task == {}
    assert offline_projects == [project]


# ---------------------------------------------------------------------------
# step-11: shape_scheduler envelope
# ---------------------------------------------------------------------------


def test_shape_scheduler_envelope():
    """shape_scheduler wraps inputs in SCHEDULER key with offline flag."""
    from dashboard.data.redux_api import shape_scheduler

    rows = [{'task_id': '1', 'title': 'T1'}]
    modules = [{'path': 'src/a.py', 'holder': '1', 'contention': 1}]
    pin_queue = [{'task_id': '1', 'order': 0}]
    events_by_task = {'1': {'labels': [], 'values': []}}
    snapshot_at = '2026-01-01T00:00:00+00:00'

    # Non-empty offline_projects → offline=True
    result_offline = shape_scheduler(
        rows=rows,
        modules=modules,
        pin_queue=pin_queue,
        events_by_task=events_by_task,
        offline_projects=['proj-a'],
        snapshot_at=snapshot_at,
    )
    assert 'SCHEDULER' in result_offline
    inner = result_offline['SCHEDULER']
    assert set(inner.keys()) == {
        'rows', 'modules', 'pin_queue', 'events_by_task',
        'snapshot_at', 'offline', 'offline_projects',
    }
    assert inner['offline'] is True
    assert inner['offline_projects'] == ['proj-a']
    assert inner['rows'] == rows
    assert inner['modules'] == modules
    assert inner['pin_queue'] == pin_queue
    assert inner['events_by_task'] == events_by_task
    assert inner['snapshot_at'] == snapshot_at

    # Empty offline_projects → offline=False
    result_online = shape_scheduler(
        rows=rows,
        modules=modules,
        pin_queue=pin_queue,
        events_by_task=events_by_task,
        offline_projects=[],
        snapshot_at=snapshot_at,
    )
    assert result_online['SCHEDULER']['offline'] is False


# ---------------------------------------------------------------------------
# step-13: GET /api/v2/dashboard/scheduler envelope shape
# ---------------------------------------------------------------------------


def test_scheduler_endpoint_returns_envelope_shape(client):
    """GET /scheduler returns 200 with SCHEDULER key containing the inner keys."""
    from unittest.mock import AsyncMock, patch

    empty_5tuple = ([], [], [], {}, [])
    with patch('dashboard.app.collect_scheduler_state', new=AsyncMock(return_value=empty_5tuple)):
        resp = client.get('/api/v2/dashboard/scheduler')

    assert resp.status_code == 200
    data = resp.json()
    assert 'SCHEDULER' in data
    inner = data['SCHEDULER']
    assert 'rows' in inner
    assert 'modules' in inner
    assert 'pin_queue' in inner
    assert 'events_by_task' in inner
    assert 'snapshot_at' in inner
    assert 'offline' in inner
    assert 'offline_projects' in inner
