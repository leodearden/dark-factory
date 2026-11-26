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


# ---------------------------------------------------------------------------
# Chain figures over merge_verify payloads: chain_items, chain_task_ids,
# chain_depth_reason {basis, pass_estimate, row: [threshold, depth], queue_depth}.
# ---------------------------------------------------------------------------


def chain_verify(
    end_minutes: float, task_id: str, items: Any, *, secs: float = 600.0, passed: bool = True,
    members: list[str] | None = None, reason: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ev(end_minutes, task_id, runner='local', chain_items=items, duration_ms=secs * 1000,
              passed=passed, chain_task_ids=members or [task_id], chain_depth_reason=reason)


def sized(depth: int, queue_depth: int, basis: str = 'pass_estimate') -> dict[str, Any]:
    return {'basis': basis, 'pass_estimate': 0.9, 'row': [0.876, depth], 'queue_depth': queue_depth}


def test_deep_chain_p90_is_the_nearest_rank_p90_of_deep_verifies():
    verify = [chain_verify(i, 't', 2, secs=100.0 * (i + 1)) for i in range(10)]
    verify += [chain_verify(20, 'solo', 1, secs=99999.0), chain_verify(21, 'odd', True, secs=99999.0)]
    assert canary.deep_chain_p90_secs(verify) == 900.0


def test_deep_chain_p90_is_unmeasured_without_a_deep_verify():
    assert canary.deep_chain_p90_secs([chain_verify(1, 'solo', 1)]) is None


def test_chain_items_distribution_tallies_every_verify():
    verify = [chain_verify(1, 'a', 1), chain_verify(2, 'b', 4), chain_verify(3, 'c', 4)]
    assert canary.chain_items_distribution(verify) == {1: 1, 4: 2}


def test_depth_reason_tally_buckets_by_basis_and_unsized():
    verify = [
        chain_verify(1, 'a', 3, reason=sized(4, 9)),
        chain_verify(2, 'b', 1, reason=sized(4, 9, basis='no_estimate')),
        chain_verify(3, 'c', 1),
        chain_verify(4, 'd', 4, reason=sized(4, 9)),
    ]
    assert canary.depth_reason_tally(verify) == {'pass_estimate': 2, 'no_estimate': 1, 'unsized': 1}


def test_depth_starved_splits_too_few_waiting_from_other():
    verify = [
        chain_verify(1, 'a', 2, reason=sized(4, 2)),  # starved, too few waiting
        chain_verify(2, 'b', 2, reason=sized(4, 9)),  # starved, other
        chain_verify(3, 'c', 4, reason=sized(4, 9)),  # full depth
        chain_verify(4, 'd', 1),                      # unsized
    ]

    result = canary.depth_starved(verify)

    assert (result.sized, result.starved, result.too_few_waiting, result.other) == (3, 2, 1, 1)
    assert result.share_of_verifies == 0.5


def test_post_red_chain_solos_count_unretired_members_re_verified_alone():
    verify = [
        chain_verify(100, 'a', 3, secs=1800, passed=False, members=['a', 'b', 'c']),  # red, predates lo
        chain_verify(105, 'e', 2, secs=600, members=['e', 'f']),                      # green chain
        chain_verify(150, 'a', 1, secs=1200),                 # counted
        chain_verify(160, 'b', 1, secs=1200, passed=False),   # b landed in between: not counted
        chain_verify(170, 'd', 1, secs=1200),                 # never in a red chain
        chain_verify(180, 'e', 1, secs=1800),                 # its chain was green
    ]
    finalized = [ev(120, 'b', state='done'), ev(155, 'a', state='blocked'), ev(190, 'a', state='done')]

    result = canary.post_red_chain_solos(verify, finalized, at(110), at(300))

    assert result.count == 1
    assert result.hours == pytest.approx(1 / 3)
    assert result.share_of_verify_hours == pytest.approx(20 / 90)
    assert result.next_finalize_blocked == 1


def test_a_solo_running_beside_its_red_chain_is_not_a_post_red_solo():
    verify = [
        chain_verify(100, 'a', 2, secs=1800, passed=False, members=['a', 'b']),
        chain_verify(110, 'a', 1, secs=1200),  # started at 90, before the red chain ended
    ]
    assert canary.post_red_chain_solos(verify, [], at(0), at(300)).count == 0


def test_per_item_chain_wall_over_deep_enough_chains():
    verify = [
        chain_verify(1, 'a', 8, secs=800.0),
        chain_verify(2, 'b', 10, secs=3000.0),
        chain_verify(3, 'c', 9, secs=1800.0),
        chain_verify(4, 'd', 5, secs=99999.0),
    ]

    wall = canary.per_item_chain_wall(verify, min_items=8)

    assert wall is not None
    assert (wall.n, wall.p50_secs, wall.max_secs) == (3, 200.0, 300.0)


def test_per_item_chain_wall_is_none_without_a_deep_enough_chain():
    assert canary.per_item_chain_wall([chain_verify(1, 'a', 5)], min_items=8) is None


# ---------------------------------------------------------------------------
# The two chain-traced invariant predicates whose producers are not in runs.db.
# ---------------------------------------------------------------------------


def anchor(minutes: float, task_id: str, base: str | None, reason: str = 'live_tip', **extra: Any) -> dict[str, Any]:
    return ev(minutes, task_id, anchor_reason=reason, anchor_base=base, point='dispatch', **extra)


def test_a_live_tip_anchor_onto_an_already_dead_tip_is_a_new_class_violation():
    voided = [
        anchor(1, 'k', None, reason='tip_dead', dead_link='dead-link', item_commit='dead-item'),
        anchor(5, 'x', 'dead-link'),
        anchor(8, 'w', 'fresh'),
        anchor(9, 'q', None, reason='no_tip', dead_link='fresh'),
        anchor(10, 'v', 'dead-link', reason='tip_vacated'),
        anchor(11, 'u', 'healthy'),
    ]
    discards = [anchor(6, 'y', 'red-tip'), anchor(7, 'z', 'dead-item')]
    verify = [ev(2, 'r', merge_sha='red-tip', passed=False), ev(3, 'g', merge_sha='healthy', passed=True)]

    result = canary.frozen_tip_violations(voided, discards, verify)

    assert result.anchors == 5
    assert result.violations == (
        ('x', at(5), 'dead-link'),
        ('y', at(6), 'red-tip'),
        ('z', at(7), 'dead-item'),
    )
    assert result.died_after_anchor == 1


def test_a_tip_recorded_dead_at_the_anchor_instant_is_not_earlier():
    voided = [anchor(5, 'x', 'tip'), anchor(5, 'k', None, reason='tip_dead', dead_link='tip')]
    result = canary.frozen_tip_violations(voided, [], [])
    assert (result.anchors, result.violations, result.died_after_anchor) == (1, (), 1)


def test_an_anchor_without_a_string_base_is_not_examined():
    result = canary.frozen_tip_violations([anchor(5, 'x', None)], [], [])
    assert result.anchors == 0


def audit(minutes: float, *texts: str, level: str = 'WARNING') -> canary.AuditRecord:
    return canary.AuditRecord(at=at(minutes), level=canary.AuditLevel(level), violations=texts)


LEDGER_TEXT = 'unregistered on-disk merge worktree /w/_merge-1 (age 9001s > grace 9000s) absent from owned ledger'


def test_accounting_violations_count_distinct_texts_of_the_four_prefixes():
    records = [
        audit(1, 'speculation-slot conservation violated: slot_available=0', LEDGER_TEXT),
        audit(2, 'speculation-slot conservation violated: slot_available=0'),
        audit(3, 'merge-ahead-cap conservation violated: held=2 cap=1'),
        audit(4, 'merge-ahead-cap handoff violated: RealMergeItem in x'),
        audit(5, 'speculation_accounting_violations: merge-ahead-cap check raised: boom'),
        audit(6, LEDGER_TEXT),
        audit(7, level='INFO'),
    ]

    tally = canary.accounting_violations(records)

    assert tally.distinct_texts == 4
    assert tally.records_with_any == 5
    assert tally.worktree_ledger_records == 2


def test_worktree_ledger_records_alone_are_not_accounting_violations():
    tally = canary.accounting_violations([audit(1, LEDGER_TEXT), audit(2, LEDGER_TEXT)])
    assert (tally.distinct_texts, tally.records_with_any, tally.worktree_ledger_records) == (0, 0, 2)


# ---------------------------------------------------------------------------
# Main's history: which commits the queue landed, the gate re-verify split,
# red-main episodes from the main-tip sweep, and what started on a red main.
# ---------------------------------------------------------------------------


def main_history() -> tuple[canary.MainUpdate, ...]:
    return (
        canary.MainUpdate(at=at(10), new_sha='d1', commits=('d1',)),
        canary.MainUpdate(at=at(20), new_sha='m2', commits=('m2',)),
        canary.MainUpdate(at=at(30), new_sha='f3', commits=('f1', 'f2', 'f3')),
        canary.MainUpdate(at=at(40), new_sha='d5', commits=('d4', 'd5')),
    )


def test_every_commit_of_a_queue_landing_is_queue_and_the_rest_direct():
    ledger = canary.classify_commits(main_history(), {'m2', 'f3'})

    queue, direct = canary.CommitClass.QUEUE, canary.CommitClass.DIRECT
    assert ledger.classes == {
        'd1': direct, 'm2': queue, 'f1': queue, 'f2': queue, 'f3': queue, 'd4': direct, 'd5': direct,
    }
    assert ledger.landed_at['f1'] == at(30)
    assert ledger.landed_at['d4'] == at(40)


def gate(minutes: float, rebased_from: str, reverify: bool = True) -> dict[str, Any]:
    return ev(minutes, 'g', reverify=reverify, rebased_from=rebased_from, rebased_onto='onto')


def test_gate_reverify_split_buckets_each_gate_by_what_landed_under_it():
    ledger = canary.classify_commits(main_history(), {'m2', 'f3'})
    ranges = {
        ('q', 'onto'): ('m2',), ('d', 'onto'): ('d1',), ('b', 'onto'): ('m2', 'd4'),
        ('e', 'onto'): (), ('u', 'onto'): ('m2', 'zz'),
    }
    gates = [
        gate(1, 'q'), gate(2, 'd'), gate(3, 'b'), gate(4, 'e'), gate(5, 'q', reverify=False),
        gate(6, 'u'), gate(7, 'missing'),
    ]

    split = canary.gate_reverify_split(gates, ranges, ledger.classes)

    assert split == {
        canary.GateRange.QUEUE_ONLY: 1, canary.GateRange.DIRECT_ONLY: 1, canary.GateRange.BOTH: 1,
        canary.GateRange.EMPTY: 1, canary.GateRange.UNCLASSIFIED: 2,
    }


def full(letter: str) -> str:
    return letter * 40


def sweep(start: float, letter: str, passed: bool, filed: bool | None = None) -> canary.SweepVerdict:
    return canary.SweepVerdict(
        started_at=at(start), verdict_at=at(start + 5), sha12=full(letter)[:12], passed=passed, filed=filed,
    )


def test_red_episodes_open_only_at_a_filed_failure_and_close_at_the_next_green():
    sweeps = [
        sweep(100, 'a', True), sweep(110, 'b', False, False), sweep(120, 'c', False, True),
        sweep(130, 'd', False, False), sweep(140, 'e', True), sweep(150, 'f', False, False),
        sweep(160, '1', True), sweep(170, '2', False, True),
    ]
    landed_at = {full(x): at(t) for x, t in [('a', 10), ('b', 20), ('c', 30), ('d', 40), ('e', 50), ('f', 60), ('1', 70)]}

    result = canary.red_episodes(sweeps, landed_at)

    closed, still_open = result.episodes
    assert (closed.first_red_sha12, closed.opening_sha12, closed.green_sha12) == (
        full('b')[:12], full('c')[:12], full('e')[:12],
    )
    assert (closed.candidate_after, closed.start, closed.end, closed.red_verdicts) == (at(10), at(20), at(50), 3)
    assert (still_open.first_red_sha12, still_open.green_sha12) == (full('2')[:12], None)
    assert (still_open.candidate_after, still_open.start, still_open.end) == (at(70), at(170), None)
    assert result.unconfirmed == 3


def test_an_episode_whose_shas_never_landed_falls_back_to_the_sweep_clock():
    sweeps = [sweep(100, 'b', False, True), sweep(140, 'e', True)]

    (episode,) = canary.red_episodes(sweeps, {}).episodes

    assert (episode.candidate_after, episode.start, episode.end) == (None, at(100), at(145))


def episode(start: float, end: float | None, after: float | None = None) -> canary.RedEpisode:
    return canary.RedEpisode(
        first_red_sha12='r', opening_sha12='r', green_sha12=None if end is None else 'g',
        candidate_after=None if after is None else at(after), start=at(start),
        end=None if end is None else at(end), red_verdicts=1,
    )


def test_verifies_started_on_red_counts_starts_inside_an_episode_by_kind():
    verify = [
        completed(150, 'local', 30, passed=False),  # started 120: on red, failed
        completed(250, 'local', 30, passed=True),   # started 220: after the episode
        completed(190, 'local', 30, passed=True),   # started 160: on red
    ]
    cancelled = [cancelled_verify(130, 'laptop', 20)]  # started 110: on red
    tasks = [
        canary.TaskVerifyStart(task_id='t1', at=at(150), passed=False),
        canary.TaskVerifyStart(task_id='t2', at=at(199), passed=True),
        canary.TaskVerifyStart(task_id='t3', at=at(200), passed=None),
    ]

    result = canary.verifies_started_on_red([episode(100, 200)], verify, cancelled, tasks)

    assert (result.merge.started, result.merge.share, result.merge.failed) == (3, 0.75, 1)
    assert (result.task.started, result.task.share, result.task.failed) == (2, pytest.approx(2 / 3), 1)


def test_an_open_episode_covers_every_later_start():
    tasks = [canary.TaskVerifyStart(task_id='t', at=at(10_000), passed=True)]
    result = canary.verifies_started_on_red([episode(100, None)], [], [], tasks)
    assert result.task.started == 1
    assert result.merge.share is None


def test_chain_attributed_episodes_need_a_chain_landing_in_the_candidate_range():
    ledger = canary.CommitLedger(
        classes={},
        landed_at={'in': at(15), 'at-red': at(20), 'at-green': at(10), 'later': at(95)},
    )
    first, second = episode(20, 50, after=10), episode(90, None, after=70)

    assert canary.chain_attributed_episodes([first, second], ledger, ['in']) == 1
    assert canary.chain_attributed_episodes([first], ledger, ['at-red']) == 1
    assert canary.chain_attributed_episodes([first], ledger, ['at-green']) == 0
    assert canary.chain_attributed_episodes([second], ledger, ['later', 'unlanded']) == 0
    assert canary.chain_attributed_episodes([], ledger, ['in']) == 0


# ---------------------------------------------------------------------------
# evaluate: the ten D12 rows, and what a false row recommends.
# ---------------------------------------------------------------------------

D12_KEYS = (
    'gates_per_landing', 'voids_per_landing', 'p_gate_after_gate', 'laptop_finished_share',
    'frozen_tip_violations', 'speculation_accounting_violations', 'chain_attributed_red_main',
    'queue_to_land_p50_hours', 'local_verify_hours_per_week', 'deep_chain_p90_secs',
)


def healthy_figures(**overrides: Any) -> canary.CanaryFigures:
    values: dict[str, Any] = {
        'gates_per_landing': 0.2, 'voids_per_landing': 0.05, 'p_gate_after_gate': 0.1,
        'laptop_finished_share': 0.05, 'frozen_tip_violations': 0,
        'speculation_accounting_violations': 0, 'chain_attributed_red_main': 0,
        'queue_to_land_p50_hours': 0.5, 'local_verify_hours_per_week': 40.0,
        'deep_chain_p90_secs': 10800.0,
    }
    return canary.CanaryFigures(**{**values, **overrides})


def test_evaluate_returns_the_ten_d12_rows_in_order_with_their_values():
    rows = canary.evaluate(healthy_figures())

    assert tuple(row.key for row in rows) == D12_KEYS
    assert [row.value for row in rows] == [0.2, 0.05, 0.1, 0.05, 0, 0, 0, 0.5, 40.0, 10800.0]
    assert all(row.holds for row in rows)
    assert all(row.rule and row.label for row in rows)


def test_evaluate_carries_each_limit_comparison_and_baseline():
    by_key = {row.key: row for row in canary.evaluate(healthy_figures())}
    lt, le, eq0 = canary.Comparison.LT, canary.Comparison.LE, canary.Comparison.EQ0

    assert {key: (row.limit, row.comparison, row.baseline) for key, row in by_key.items()} == {
        'gates_per_landing': (0.3, lt, '0.84'),
        'voids_per_landing': (0.1, lt, '0.78'),
        'p_gate_after_gate': (0.3, lt, '0.79'),
        'laptop_finished_share': (0.1, lt, '33%'),
        'frozen_tip_violations': (0, eq0, '—'),
        'speculation_accounting_violations': (0, eq0, '—'),
        'chain_attributed_red_main': (0, eq0, '—'),
        'queue_to_land_p50_hours': (1.0, lt, '5.4 h'),
        'local_verify_hours_per_week': (60.0, lt, '~100 h'),
        'deep_chain_p90_secs': (10800.0, le, '—'),
    }


def test_exactly_the_three_invariant_rows_are_chain_traced():
    traced = {row.key for row in canary.evaluate(healthy_figures()) if row.chain_traced}
    assert traced == {'frozen_tip_violations', 'speculation_accounting_violations', 'chain_attributed_red_main'}


def test_false_predicates_include_unmeasured_rows():
    rows = canary.evaluate(healthy_figures(gates_per_landing=0.84, deep_chain_p90_secs=None))
    assert [row.key for row in canary.false_predicates(rows)] == ['gates_per_landing', 'deep_chain_p90_secs']


def test_a_false_gate_row_alone_does_not_recommend_chain_cap_zero():
    rows = canary.evaluate(healthy_figures(gates_per_landing=0.84, voids_per_landing=0.78))
    assert canary.recommends_chain_cap_zero(rows) is False


@pytest.mark.parametrize('key', ['frozen_tip_violations', 'speculation_accounting_violations', 'chain_attributed_red_main'])
def test_a_false_chain_traced_row_recommends_chain_cap_zero(key):
    assert canary.recommends_chain_cap_zero(canary.evaluate(healthy_figures(**{key: 1}))) is True


def test_an_unmeasured_chain_traced_row_recommends_chain_cap_zero():
    rows = canary.evaluate(healthy_figures(chain_attributed_red_main=None))
    assert canary.recommends_chain_cap_zero(rows) is True


def test_all_rows_holding_recommend_nothing():
    rows = canary.evaluate(healthy_figures())
    assert canary.false_predicates(rows) == ()
    assert canary.recommends_chain_cap_zero(rows) is False
