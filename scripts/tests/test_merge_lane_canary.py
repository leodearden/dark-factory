"""Tests for scripts/merge_lane_canary.py, the pure layer of the task-8532 canary readout.

Every input is an event dict in merge_lane_throughput.load_events' shape,
``{'timestamp': <iso +00:00>, 'task_id': ..., 'data': {...}}``, built by
:func:`ev` below, or one of the module's small frozen records. Every expected
number is worked out by hand from the fixture beside it.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import merge_lane_canary as canary
import pytest

T0 = datetime(2026, 11, 20, 0, 0, tzinfo=UTC)


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def ev(minutes: float, task_id: str | None = None, **data: Any) -> dict[str, Any]:
    return {'timestamp': at(minutes).isoformat(), 'task_id': task_id, 'data': data}


# ---------------------------------------------------------------------------
# Predicate: one D12 row, its comparison and how an unmeasured value reads.
# ---------------------------------------------------------------------------


def predicate(value: float | None, limit: float, comparison: canary.Comparison) -> canary.Predicate:
    return canary.Predicate(
        key='k', label='a figure', value=value, limit=limit, comparison=comparison,
        rule='the counting rule', baseline='0.84', chain_traced=False,
    )


def test_lt_holds_strictly_below_the_limit():
    assert predicate(0.25, 0.3, canary.Comparison.LT).holds
    assert not predicate(0.3, 0.3, canary.Comparison.LT).holds


def test_le_holds_at_equality():
    assert predicate(10800, 10800, canary.Comparison.LE).holds
    assert not predicate(10800.5, 10800, canary.Comparison.LE).holds


def test_eq0_holds_only_at_zero():
    assert predicate(0, 0, canary.Comparison.EQ0).holds
    assert not predicate(1, 0, canary.Comparison.EQ0).holds


@pytest.mark.parametrize('comparison', list(canary.Comparison))
def test_an_unmeasured_predicate_never_holds_and_says_so(comparison):
    row = predicate(None, 0.3, comparison)
    assert row.holds is False
    assert 'not measured' in row.rendered_value


def test_a_measured_value_renders_as_its_number():
    assert predicate(0.25, 0.3, canary.Comparison.LT).rendered_value == '0.25'


def test_comparison_is_parsed_from_its_wire_spelling():
    assert canary.Comparison('le') is canary.Comparison.LE


# ---------------------------------------------------------------------------
# missing_alpha_fields: the refusal test for a window that predates alpha.
# ---------------------------------------------------------------------------


def alpha_window() -> tuple[list[dict[str, Any]], ...]:
    verify = [ev(1, '1', runner_reason='only_host', base_relation='main_tip', passed=True)]
    cancelled = [ev(2, '2', elapsed_ms=1000, fraction=0.5, cause='void', host='local')]
    voided = [ev(3, '3', item_commit='abc', killer={'site': 'head_fail', 'task_id': '9'})]
    finalized = [ev(4, '1', state='done', queue_to_land_secs=600.0)]
    return verify, cancelled, voided, finalized


def test_no_field_is_missing_when_each_is_carried_by_some_row():
    verify, cancelled, voided, finalized = alpha_window()
    pre_alpha = ev(0, '0', passed=True)
    assert canary.missing_alpha_fields(
        [pre_alpha, *verify], cancelled, [ev(0, '0'), *voided], finalized,
    ) == ()


def test_a_pre_alpha_window_names_every_absent_field_in_order():
    verify = [ev(1, '1', passed=True, runner_reason=None)]
    voided = [ev(3, '3', item_commit=None)]
    finalized = [
        ev(4, '1', state='done'),
        ev(5, '2', state='blocked', queue_to_land_secs=60.0),
    ]
    assert canary.missing_alpha_fields(verify, [], voided, finalized) == (
        'merge_verify.runner_reason',
        'merge_verify.base_relation',
        'merge_verify_cancelled',
        'verdict_voided.item_commit',
        'verdict_voided.killer',
        'merge_finalized.queue_to_land_secs',
    )


def test_a_window_without_cancelled_rows_names_only_that_event():
    verify, _cancelled, voided, finalized = alpha_window()
    assert canary.missing_alpha_fields(verify, [], voided, finalized) == (
        'merge_verify_cancelled',
    )
