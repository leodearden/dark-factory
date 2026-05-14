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
