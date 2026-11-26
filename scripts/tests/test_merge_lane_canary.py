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


# ---------------------------------------------------------------------------
# The landing ledger and the gate/void ratios (merge-throughput report,
# B-ledger.md population rules and P5).
# ---------------------------------------------------------------------------


def test_landings_are_done_finalizes_union_train_members_in_time_order():
    finalized = [
        ev(30, 'b', state='done', merge_sha='sha-b', landed_via_chain=1),
        ev(10, 'a', state='done', merge_sha='sha-a', landed_via_chain=None),
        ev(20, 'x', state='blocked', merge_sha=None),
    ]
    train_merged = [ev(25, 't', member_task_ids=['m1', 'm2'], merge_commit_sha='sha-t')]

    rows = canary.landings(finalized, train_merged)

    assert [(r.task_id, r.at, r.merge_sha) for r in rows] == [
        ('a', at(10), 'sha-a'),
        ('m1', at(25), 'sha-t'),
        ('m2', at(25), 'sha-t'),
        ('b', at(30), 'sha-b'),
    ]
    assert [r.via_chain for r in rows] == [False, False, False, True]


def test_a_train_member_with_its_own_done_finalize_lands_once():
    finalized = [ev(10, 'm1', state='done', merge_sha='sha-t')]
    train_merged = [ev(10, 't', member_task_ids=['m1', 'm2'], merge_commit_sha='sha-t')]
    assert [r.task_id for r in canary.landings(finalized, train_merged)] == ['m1', 'm2']


@pytest.mark.parametrize(('raw', 'via_chain'), [(1, True), (2.0, True), (None, False), (True, False), ('1', False)])
def test_via_chain_is_true_only_for_a_non_bool_number(raw, via_chain):
    finalized = [ev(1, 'a', state='done', merge_sha='s', landed_via_chain=raw)]
    assert canary.landings(finalized, [])[0].via_chain is via_chain


def two_landings() -> tuple[canary.Landing, ...]:
    return canary.landings(
        [ev(10, 'a', state='done', merge_sha='sa'), ev(20, 'b', state='done', merge_sha='sb')], [],
    )


def test_gates_per_landing_counts_only_reverifying_gate_rows():
    gates = [ev(5, 'a', reverify=True), ev(15, 'b', reverify=False), ev(16, 'b', reverify=True)]
    assert canary.gates_per_landing(gates, two_landings()) == 1.0


def test_gates_per_landing_is_unmeasured_without_a_landing():
    assert canary.gates_per_landing([ev(5, 'a', reverify=True)], ()) is None


def test_voids_per_landing_splits_dispatch_from_adoption_voids():
    voided = [ev(1, 'a', point='dispatch'), ev(2, 'b', point='dispatch'), ev(3, 'c', point='adoption')]

    tally = canary.voids_per_landing(voided, two_landings())

    assert (tally.total, tally.before_host, tally.at_adoption) == (3, 2, 1)
    assert tally.per_landing == 1.5


def test_voids_per_landing_is_unmeasured_without_a_landing():
    assert canary.voids_per_landing([ev(1, 'a', point='dispatch')], ()).per_landing is None


def gated_ledger(pattern: str) -> tuple[tuple[canary.Landing, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    """One landing per letter, ten minutes apart; 'G' lands behind a re-verifying gate."""
    finalized, gates, queued = [], [], []
    for i, letter in enumerate(pattern):
        task = f't{i}'
        queued.append(ev(10 * i + 1, task))
        if letter == 'G':
            gates.append(ev(10 * i + 5, task, reverify=True))
        finalized.append(ev(10 * i + 9, task, state='done', merge_sha=f's{i}'))
    return canary.landings(finalized, []), gates, queued


def test_gate_conditionals_over_a_known_sequence():
    ledger, gates, queued = gated_ledger('GGNGGG')

    result = canary.gate_conditionals(ledger, gates, queued)

    assert result.gated == (True, True, False, True, True, True)
    assert result.p_gate_after_gate == 0.75
    assert result.p_gate_after_ungated == 1.0


def test_a_gate_from_an_earlier_submission_does_not_gate_the_landing():
    queued = [ev(1, 'a'), ev(20, 'a')]
    gates = [ev(5, 'a', reverify=True), ev(25, 'a', reverify=False)]
    ledger = canary.landings([ev(30, 'a', state='done', merge_sha='s')], [])

    assert canary.gate_conditionals(ledger, gates, queued).gated == (False,)


def test_a_gate_stamped_after_the_landing_does_not_gate_it():
    ledger = canary.landings([ev(30, 'a', state='done', merge_sha='s')], [])
    assert canary.gate_conditionals(ledger, [ev(31, 'a', reverify=True)], [ev(1, 'a')]).gated == (False,)


def test_gate_conditionals_are_unmeasured_when_no_gated_landing_has_a_successor():
    ledger, gates, queued = gated_ledger('NG')

    result = canary.gate_conditionals(ledger, gates, queued)

    assert result.p_gate_after_gate is None
    assert result.p_gate_after_ungated == 1.0


# ---------------------------------------------------------------------------
# Time-based figures: heartbeat-weighted shares and verify-interval unions
# (merge-throughput report: C-laptop, B-ledger §4b, D-regimes §3.2).
# ---------------------------------------------------------------------------


def heartbeat(minutes: float, *leases: tuple[str, str]) -> dict[str, Any]:
    return ev(minutes, None, occupancy={
        'leases': [{'host': host, 'task_id': 'x', 'activity': activity} for host, activity in leases],
    })


def test_finished_share_weights_each_heartbeat_by_the_gap_to_the_next():
    beats = [
        heartbeat(0, ('laptop', 'verifying')),
        heartbeat(5, ('laptop', 'finished')),
        heartbeat(10, ('local', 'finished')),
    ]
    assert canary.host_finished_share(beats, at(15), host='laptop') == pytest.approx(1 / 3)


def test_finished_share_caps_a_gap_at_600_seconds():
    beats = [heartbeat(0, ('laptop', 'finished')), heartbeat(120, ('laptop', 'verifying'))]
    assert canary.host_finished_share(beats, at(125), host='laptop') == pytest.approx(600 / 900)


def test_the_last_heartbeat_is_carried_to_hi_under_the_same_cap():
    beats = [heartbeat(0, ('laptop', 'verifying')), heartbeat(5, ('laptop', 'finished'))]
    assert canary.host_finished_share(beats, at(600), host='laptop') == pytest.approx(600 / 900)


def test_another_hosts_finished_lease_does_not_count():
    beats = [heartbeat(0, ('local', 'finished')), heartbeat(5, ('laptop', 'verifying'))]
    assert canary.host_finished_share(beats, at(10), host='laptop') == 0.0


def test_finished_share_is_unmeasured_without_heartbeats():
    assert canary.host_finished_share([], at(10), host='laptop') is None


def completed(end_minutes: float, runner: str, duration_minutes: float, **extra: Any) -> dict[str, Any]:
    return ev(end_minutes, extra.pop('task_id', 'v'), runner=runner,
              duration_ms=duration_minutes * 60_000, **extra)


def cancelled_verify(end_minutes: float, host: str, elapsed_minutes: float) -> dict[str, Any]:
    return ev(end_minutes, 'c', host=host, elapsed_ms=elapsed_minutes * 60_000, cause='void')


def test_host_verify_hours_is_the_clipped_union_of_one_hosts_intervals():
    verify = [
        completed(120, 'local', 60),   # [60, 120]
        completed(150, 'local', 60),   # [90, 150], overlaps: union [60, 150] = 90 min
        completed(300, 'laptop', 60),  # another host
        completed(30, 'local', 60),    # [-30, 30], clipped to [0, 30] = 30 min
    ]
    cancelled = [cancelled_verify(400, 'local', 30), cancelled_verify(500, 'laptop', 30)]

    result = canary.host_verify_hours(verify, cancelled, at(0), at(600), host='local')

    assert result.hours == pytest.approx(2.5)
    assert result.per_week == pytest.approx(2.5 * 168 / 10)


def test_union_busy_hours_counts_two_hosts_verifying_at_once_once():
    verify = [completed(120, 'local', 60), completed(150, 'laptop', 60)]
    assert canary.union_busy_hours(verify, at(0), at(600)) == pytest.approx(1.5)


def test_queue_to_land_is_the_median_of_done_landings_in_hours():
    finalized = [
        ev(1, 'a', state='done', queue_to_land_secs=3600.0),
        ev(2, 'b', state='done', queue_to_land_secs=10800.0),
        ev(3, 'c', state='done', queue_to_land_secs=7200.0),
        ev(4, 'd', state='done'),
        ev(5, 'e', state='blocked', queue_to_land_secs=99999.0),
    ]

    result = canary.queue_to_land(finalized)

    assert result.p50_hours == pytest.approx(2.0)
    assert (result.n, result.n_missing) == (3, 1)


def test_queue_to_land_is_unmeasured_without_a_landing():
    assert canary.queue_to_land([]).p50_hours is None
