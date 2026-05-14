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
