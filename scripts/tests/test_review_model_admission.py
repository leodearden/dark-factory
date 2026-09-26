"""Tests for scripts/review_model_admission.py.

Hermetic: the pure summaries run over InvocationRecord values built here, and
the scans run against the synthetic runs.db from conftest.py's ``runs_db``
fixture and escalation records written into tmp_path.
"""
import json
from datetime import UTC, datetime, timedelta

import audit_model_admission
import escalation_ladder
import pytest
import review_model_admission

FABLE = 'claude-fable-5-1'
APPLY = datetime(2026, 9, 12, 6, 43, 16, tzinfo=UTC)


def _at(**offset):
    """An ISO-8601 timestamp *offset* from the D6 apply time, spelled as the store does."""
    return (APPLY + timedelta(**offset)).isoformat()


def _merged(state):
    return audit_model_admission.MergeOutcome(
        timestamp=_at(hours=3), state=state, merge_sha=None, reason=None,
    )


def _record(**overrides):
    """An InvocationRecord with keyword defaults: a successful Fable merger run."""
    fields = {
        'task_id': '4377',
        'project_id': 'dark_factory',
        'role': 'merger',
        'account_name': 'max-b',
        'cost_usd': 1.0,
        'duration_ms': 60_000,
        'capped': False,
        'started_at': _at(hours=2),
        'completed_at': _at(hours=2, minutes=1),
        'turns': 10,
        'succeeded': True,
        'timed_out': False,
        'end_event_model': FABLE,
        'merge_outcome': None,
        'at_or_over_flat_role_ceiling': False,
        'subtype': 'success',
        'escalation_id': None,
        'routing_tier': 0,
        'dispatch_max_turns': 100,
    }
    return audit_model_admission.InvocationRecord(**{**fields, **overrides})


# --- nearest_rank ---


def test_nearest_rank_of_nothing_is_none():
    assert review_model_admission.nearest_rank([], 50) is None


def test_nearest_rank_of_one_value_is_that_value():
    assert review_model_admission.nearest_rank([7.5], 95) == 7.5


@pytest.mark.parametrize(('pct', 'expected'), [(50, 10), (95, 19)])
def test_nearest_rank_over_one_to_twenty(pct, expected):
    assert review_model_admission.nearest_rank(list(range(1, 21)), pct) == expected


def test_nearest_rank_p95_of_eighteen_is_the_maximum():
    """ceil(0.95 * 18) = 18: on the D6 merger arm's n, p95 IS the worst run."""
    assert review_model_admission.nearest_rank(list(range(1, 19)), 95) == 18


def test_nearest_rank_does_not_depend_on_input_order():
    assert review_model_admission.nearest_rank([5, 1, 4, 2, 3], 50) == 3


# --- summarize_outcomes ---


def _five_runs():
    return [
        _record(cost_usd=2.0, turns=20, duration_ms=100_000,
                merge_outcome=_merged('done')),
        _record(succeeded=False, subtype='error_max_turns', turns=100, cost_usd=8.0,
                duration_ms=700_000, merge_outcome=_merged('blocked'),
                at_or_over_flat_role_ceiling=True),
        _record(succeeded=False, subtype='error_max_budget_usd', turns=60, cost_usd=5.1,
                duration_ms=300_000, dispatch_max_turns=50),
        _record(succeeded=False, subtype='error', turns=30, timed_out=True, cost_usd=3.0,
                duration_ms=7_200_000, merge_outcome=_merged('done'),
                at_or_over_flat_role_ceiling=True),
        _record(succeeded=None, subtype=None, turns=None, timed_out=None, cost_usd=0.5,
                duration_ms=1_000, dispatch_max_turns=None, end_event_model=None),
    ]


def test_a_worked_five_run_summary_reports_every_field():
    summary = review_model_admission.summarize_outcomes(_five_runs())

    assert summary.runs == 5
    assert summary.succeeded == 1
    assert summary.turn_cap_kills == 1
    assert summary.budget_kills == 1
    assert summary.timed_out == 1
    assert summary.no_end_event == 1
    assert summary.over_flat_ceiling == 2
    assert summary.cost_total_usd == pytest.approx(18.6)
    assert (summary.cost_usd_p50, summary.cost_usd_p95) == (3.0, 8.0)
    # the no-end-event run's turns are unknown, not zero: left out
    assert (summary.turns_p50, summary.turns_p95) == (30, 100)
    assert (summary.duration_ms_p50, summary.duration_ms_p95) == (300_000, 7_200_000)
    assert summary.duration_ms_max == 7_200_000
    assert summary.merge_states == (('blocked', 1), ('done', 2), ('-', 2))
    assert summary.dispatch_caps == ((50, 1), (100, 3), ('-', 1))


def test_resolved_is_the_done_count_of_the_merge_breakdown():
    summary = review_model_admission.summarize_outcomes(_five_runs())

    assert summary.resolved == dict(summary.merge_states)['done'] == 2


def test_resolved_is_zero_when_no_merge_finished_done():
    summary = review_model_admission.summarize_outcomes(
        [_record(merge_outcome=_merged('blocked'))],
    )

    assert summary.resolved == 0


def test_an_empty_arm_summarizes_to_zeros_and_unknown_percentiles():
    summary = review_model_admission.summarize_outcomes([])

    assert (summary.runs, summary.succeeded, summary.turn_cap_kills, summary.budget_kills,
            summary.timed_out, summary.no_end_event, summary.over_flat_ceiling) == (
        0, 0, 0, 0, 0, 0, 0,
    )
    assert summary.cost_total_usd == 0.0
    assert (summary.cost_usd_p50, summary.cost_usd_p95, summary.turns_p50,
            summary.turns_p95, summary.duration_ms_p50, summary.duration_ms_p95,
            summary.duration_ms_max) == (None,) * 7
    assert summary.merge_states == ()
    assert summary.dispatch_caps == ()
    assert summary.resolved == 0


# --- merger integrity: the drop guard and conflict re-opens ---

UNTIL = APPLY + timedelta(days=1)
DROP_REASON = 'Merge commit is missing plan target files: scripts/foo.py'


def _finalized(runs_db, timestamp, *, task_id, state, reason=None):
    runs_db.seed_event(
        timestamp, 'merge_finalized', task_id=task_id,
        data={'branch': task_id, 'state': state, 'merge_sha': None, 'reason': reason},
    )


def test_the_drop_guard_is_read_from_both_of_its_witnesses(runs_db):
    runs_db.seed_event(_at(hours=1), 'merge_attempt', task_id='1058',
                       data={'outcome': 'dropped_plan_targets'})
    _finalized(runs_db, _at(hours=2), task_id='1071', state='blocked', reason=DROP_REASON)

    events = review_model_admission.scan_drop_guard(runs_db, since=APPLY, until=UNTIL)

    assert [(e.timestamp, e.task_id, e.source) for e in events] == [
        (_at(hours=1), '1058', 'merge_attempt'),
        (_at(hours=2), '1071', 'merge_finalized'),
    ]


def test_other_merge_outcomes_and_out_of_window_drops_are_not_drop_guard_events(runs_db):
    _finalized(runs_db, _at(hours=1), task_id='1', state='blocked',
               reason='Post-merge verification failed: pytest exited 1')
    runs_db.seed_event(_at(hours=2), 'merge_attempt', task_id='2', data={'outcome': 'conflict'})
    runs_db.seed_event(_at(hours=-1), 'merge_attempt', task_id='3',
                       data={'outcome': 'dropped_plan_targets'})
    runs_db.seed_event(UNTIL.isoformat(), 'merge_attempt', task_id='4',
                       data={'outcome': 'dropped_plan_targets'})
    _finalized(runs_db, UNTIL.isoformat(), task_id='5', state='blocked', reason=DROP_REASON)

    assert review_model_admission.scan_drop_guard(runs_db, since=APPLY, until=UNTIL) == ()


def _resolved_merger_run(task_id='4377', **overrides):
    return _record(task_id=task_id, completed_at=_at(hours=2), **overrides)


def test_a_conflict_after_a_successful_merger_run_is_a_reopen(runs_db):
    _finalized(runs_db, _at(hours=1), task_id='4377', state='conflict')  # the one it resolved
    _finalized(runs_db, _at(hours=5), task_id='4377', state='conflict')
    _finalized(runs_db, _at(hours=6), task_id='4377', state='conflict')
    _finalized(runs_db, _at(hours=5), task_id='9999', state='conflict')  # another task
    _finalized(runs_db, _at(hours=25), task_id='4377', state='conflict')  # after until
    _finalized(runs_db, _at(hours=7), task_id='4377', state='done')       # not a conflict

    reopens = review_model_admission.conflict_reopens(
        runs_db, [_resolved_merger_run()], until=UNTIL,
    )

    assert [(r.task_id, r.run_completed_at, r.reopened_at) for r in reopens] == [
        ('4377', _at(hours=2), _at(hours=5)),
        ('4377', _at(hours=2), _at(hours=6)),
    ]


@pytest.mark.parametrize(
    'record',
    [
        _resolved_merger_run(succeeded=False),
        _resolved_merger_run(succeeded=None),
        _resolved_merger_run(role='steward'),
    ],
    ids=['failed-run', 'no-end-event', 'non-merger'],
)
def test_only_a_successful_merger_run_can_be_reopened(runs_db, record):
    _finalized(runs_db, _at(hours=5), task_id='4377', state='conflict')

    assert review_model_admission.conflict_reopens(runs_db, [record], until=UNTIL) == ()


def test_no_records_means_no_reopens(runs_db):
    _finalized(runs_db, _at(hours=5), task_id='4377', state='conflict')

    assert review_model_admission.conflict_reopens(runs_db, [], until=UNTIL) == ()


# --- cost and caps: daily slices, the peak trailing 24 h, rejections, scoped hits ---


def _spend(runs_db, cost_usd, **offset):
    runs_db.seed_invocation(
        model=FABLE, role='merger', task_id='4377', cost_usd=cost_usd,
        started_at=_at(**offset), completed_at=_at(**offset),
    )


def test_daily_spend_is_exactly_days_half_open_slices(runs_db):
    _spend(runs_db, 5.0, hours=1)
    _spend(runs_db, 7.0, hours=24)   # on the slice-0/1 boundary: the LATER slice
    _spend(runs_db, 3.0, hours=47)
    _spend(runs_db, 99.0, hours=72)  # the end of the last slice: out

    slices = review_model_admission.daily_spend(
        runs_db, model=FABLE, since=APPLY, days=3, ceiling_usd=150.0,
    )

    assert all(isinstance(s, audit_model_admission.SpendInWindow) for s in slices)
    assert [(s.window_start, s.window_end) for s in slices] == [
        (_at(days=k), _at(days=k + 1)) for k in range(3)
    ]
    assert [s.total_usd for s in slices] == [5.0, 10.0, 0.0]
    assert [s.invocation_count for s in slices] == [1, 2, 0]
    assert slices[0].headroom_usd == 145.0


def test_daily_spend_without_a_ceiling_leaves_the_ceiling_cells_unknown(runs_db):
    _spend(runs_db, 5.0, hours=1)

    (only,) = review_model_admission.daily_spend(
        runs_db, model=FABLE, since=APPLY, days=1, ceiling_usd=None,
    )

    assert (only.ceiling_usd, only.headroom_usd, only.at_or_over_ceiling) == (None, None, None)


def test_the_peak_trailing_24h_can_exceed_a_ceiling_no_daily_slice_reaches(runs_db):
    """The resolver sums a TRAILING window, so it can trip on spend that two
    adjacent daily slices split between them."""
    _spend(runs_db, 60.0, hours=14)
    _spend(runs_db, 50.0, hours=22)
    _spend(runs_db, 45.0, hours=37, minutes=59)

    peak = review_model_admission.peak_trailing_24h(
        runs_db, model=FABLE, since=APPLY, until=APPLY + timedelta(days=2),
        ceiling_usd=150.0,
    )
    slices = review_model_admission.daily_spend(
        runs_db, model=FABLE, since=APPLY, days=2, ceiling_usd=150.0,
    )

    assert peak is not None
    assert peak.total_usd == pytest.approx(155.0)
    assert peak.at == _at(hours=37, minutes=59)
    assert peak.ceiling_usd == 150.0
    assert peak.at_or_over_ceiling is True
    assert not any(s.at_or_over_ceiling for s in slices)


def test_the_peak_window_is_closed_at_both_ends_like_the_resolver_s(runs_db):
    """shared/src/shared/cost_store.py::CostStore.model_cost_in_window sums
    with BETWEEN: a run exactly 24 h before t is still inside t's window."""
    _spend(runs_db, 100.0, hours=1)
    _spend(runs_db, 60.0, hours=25)

    peak = review_model_admission.peak_trailing_24h(
        runs_db, model=FABLE, since=APPLY, until=APPLY + timedelta(days=2),
        ceiling_usd=None,
    )

    assert peak is not None
    assert (peak.total_usd, peak.at) == (pytest.approx(160.0), _at(hours=25))
    assert peak.at_or_over_ceiling is None


def test_the_peak_is_none_when_nothing_ran(runs_db):
    _spend(runs_db, 100.0, hours=-1)  # before since

    assert review_model_admission.peak_trailing_24h(
        runs_db, model=FABLE, since=APPLY, until=APPLY + timedelta(days=2),
        ceiling_usd=150.0,
    ) is None


def _rejection(reason, resolved_model):
    return audit_model_admission.RoutingRejection(
        timestamp=_at(hours=1), task_id='4377', role='merger',
        resolved_model=resolved_model, reasons=(reason,),
    )


def test_ceiling_trips_are_the_ceiling_rejections_with_their_fall_through():
    rejections = (
        _rejection('config:model-not-in-allowlist', 'opus'),
        _rejection('config:model-ceiling-exhausted', 'opus'),
        _rejection('policy_rule:model-capacity-exhausted', 'sonnet'),
    )
    scan = audit_model_admission.RoutingScan(
        selections=(), rejections=rejections, skipped_rows=0,
    )

    assert review_model_admission.model_rejections(scan) == rejections
    trips = review_model_admission.ceiling_trips(scan)
    assert [(t.reasons, t.resolved_model) for t in trips] == [
        (('config:model-ceiling-exhausted',), 'opus'),
    ]


def test_scoped_hits_are_counted_per_account_with_each_first_hit():
    hits = tuple(
        audit_model_admission.ScopedCapHit(created_at=at, account_name=account, reason='limit')
        for at, account in (
            (_at(days=2), 'max-e'), (_at(days=3), 'max-c'), (_at(days=4), 'max-e'),
        )
    )
    scan = audit_model_admission.ScopedCapScan(
        scoped_hits=hits, unscoped_cap_hit_count=0, restarts=(),
    )

    assert review_model_admission.scoped_hits_by_account(scan) == (
        ('max-c', 1, _at(days=3)),
        ('max-e', 2, _at(days=2)),
    )


# --- assembly: review(), render_markdown() and the CLI over one scenario ---

END = APPLY + timedelta(days=14)
BASE = APPLY - timedelta(days=30)
DAYS, BASELINE_DAYS = 14, 30


def _decision_payload(*, role, model, routing_tier=0, max_turns=100, rejected=()):
    """The `routing_decision` keys the scans read, as the producer spells them."""
    return {
        'role': role, 'model': model, 'source_layer': 'config', 'rule_id': None,
        'rejected': list(rejected), 'routing_tier': routing_tier, 'max_turns': max_turns,
        'budget_usd': 8.0,
    }


def _run(conn, *, model, role, task_id, start, minutes, cost_usd, turns=20,
         success=True, subtype='success', routing_tier=0, max_turns=100,
         merge_state=None, escalation_id=None):
    """Seed one run with its own dispatch decision, end event and merge outcome.

    The decision sits where its producer records it: 10 ms before a steward
    run starts, 10 ms after any other run completes.
    """
    started, completed = APPLY + start, APPLY + start + timedelta(minutes=minutes)
    decided = (started - timedelta(milliseconds=10) if role == 'steward'
               else completed + timedelta(milliseconds=10))
    conn.seed_event(
        decided.isoformat(), 'routing_decision', task_id=task_id, role=role,
        data=_decision_payload(role=role, model=model, routing_tier=routing_tier,
                               max_turns=max_turns),
    )
    conn.seed_invocation(
        model=model, role=role, task_id=task_id, cost_usd=cost_usd,
        started_at=started.isoformat(), completed_at=completed.isoformat(),
        duration_ms=minutes * 60_000,
    )
    conn.seed_event(
        completed.isoformat(), 'invocation_end', task_id=task_id, role=role,
        data={'turns': turns, 'success': success, 'subtype': subtype, 'model': model,
              'timed_out': False, 'escalation_id': escalation_id},
    )
    if merge_state is not None:
        _finalized(conn, completed.isoformat(), task_id=task_id, state=merge_state)


def _escalation(directory, esc_id, *, level=0, agent_role='implementer', at,
                resolved_at=None, resolved_by=None, action=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{esc_id}.json').write_text(json.dumps({
        'id': esc_id, 'task_id': esc_id.removeprefix('esc-').rsplit('-', 1)[0],
        'level': level, 'agent_role': agent_role,
        'status': 'pending' if resolved_at is None else 'resolved',
        'timestamp': at.isoformat(),
        'resolved_at': None if resolved_at is None else resolved_at.isoformat(),
        'resolved_by': resolved_by, 'resolution_action': action,
    }))


@pytest.fixture
def scenario(runs_db, tmp_path):
    """One store pair exercising every section; returns the escalations dir.

    Fable merger x2 (one re-opened by a later conflict), opus merger x2 before
    the apply (one turn-cap kill at max_turns 50), Fable steward x2 at tier 1
    (one resolved in place, one promoted), opus steward at tier 0 in the
    window and at tier 1 before it, a Fable implementer run (leakage), and a
    Fable merger completing exactly at the window end (out everywhere).
    """
    day, hour = timedelta(days=1), timedelta(hours=1)
    _run(runs_db, model=FABLE, role='merger', task_id='101', start=hour, minutes=5,
         cost_usd=4.0, merge_state='done')
    _run(runs_db, model=FABLE, role='merger', task_id='102', start=day, minutes=100,
         cost_usd=6.0, merge_state='done')
    _finalized(runs_db, _at(days=2), task_id='102', state='conflict')
    _run(runs_db, model='opus', role='merger', task_id='201', start=-3 * day, minutes=9,
         cost_usd=5.0, turns=50, success=False, subtype='error_max_turns', max_turns=50,
         merge_state='blocked')
    _run(runs_db, model='opus', role='merger', task_id='202', start=-2 * day, minutes=8,
         cost_usd=3.0, max_turns=50, merge_state='done')
    _run(runs_db, model=FABLE, role='steward', task_id='301', start=3 * day, minutes=20,
         cost_usd=2.0, routing_tier=1, max_turns=80, escalation_id='esc-301-1')
    _run(runs_db, model=FABLE, role='steward', task_id='302', start=4 * day, minutes=20,
         cost_usd=3.0, routing_tier=1, max_turns=80, escalation_id='esc-302-1')
    _run(runs_db, model='opus', role='steward', task_id='303', start=5 * day, minutes=10,
         cost_usd=1.5, escalation_id='esc-303-1')
    _run(runs_db, model='opus', role='steward', task_id='304', start=-5 * day, minutes=10,
         cost_usd=1.0, routing_tier=1, escalation_id='esc-304-1')
    _run(runs_db, model=FABLE, role='implementer', task_id='401', start=6 * day,
         minutes=10, cost_usd=1.0)
    _run(runs_db, model=FABLE, role='merger', task_id='501',
         start=14 * day - timedelta(minutes=10), minutes=10, cost_usd=50.0,
         merge_state='done')

    runs_db.seed_event(_at(days=-10), 'merge_attempt', task_id='901',
                       data={'outcome': 'dropped_plan_targets'})
    runs_db.seed_event(
        _at(days=6), 'routing_decision', task_id='305', role='steward',
        data=_decision_payload(role='steward', model='opus',
                               rejected=['config:model-ceiling-exhausted']),
    )
    runs_db.seed_account_event(account_name='max-e', event_type='cap_hit',
                               created_at=_at(days=2),
                               details={'reason': 'limit', 'scope': FABLE})
    runs_db.seed_event(_at(days=1), 'service_restart', task_id='102',
                       data={'service': 'orchestrator', 'reason': 'merge landed'})

    escalations = tmp_path / 'escalations'
    archived = escalations / 'archive' / '2026-09-01'
    _escalation(escalations, 'esc-301-1', at=APPLY + 3 * day,
                resolved_at=APPLY + 3 * day + 20 * timedelta(minutes=1),
                resolved_by='claude-task-301-steward')
    promoted_at = APPLY + 4 * day + timedelta(minutes=20)
    _escalation(escalations, 'esc-302-1', at=APPLY + 4 * day, resolved_at=promoted_at,
                resolved_by='steward')
    _escalation(escalations, 'esc-302-2', level=1, agent_role='steward',
                at=promoted_at - timedelta(milliseconds=8))
    _escalation(escalations, 'esc-303-1', at=APPLY + 5 * day,
                resolved_at=APPLY + 5 * day + timedelta(minutes=10),
                resolved_by='auto-dismissed')
    _escalation(archived, 'esc-304-1', at=APPLY - 5 * day,
                resolved_at=APPLY - 5 * day + timedelta(minutes=10),
                resolved_by='claude-task-304-steward')
    _escalation(escalations, 'esc-601-1', level=2, agent_role='escalation-watcher-auto',
                at=APPLY + 7 * day, resolved_at=APPLY + 7 * day + hour, action='close_only')
    _escalation(archived, 'esc-602-1', level=2, agent_role='steward',
                at=APPLY - 7 * day, resolved_at=APPLY - 7 * day + hour,
                action='code_change')
    return escalations


def _review(runs_db, escalations, **overrides):
    kwargs = {
        'model': FABLE, 'baseline_model': 'opus', 'expected_roles': ('merger', 'steward'),
        'apply': APPLY, 'days': DAYS, 'baseline_days': BASELINE_DAYS, 'ceiling_usd': 150.0,
        **overrides,
    }
    corpus = escalation_ladder.load_escalation_corpus(escalations)
    return review_model_admission.review(runs_db, corpus, **kwargs)


def test_the_merger_arms_are_candidate_baseline_before_and_baseline_in_window(
    runs_db, scenario
):
    candidate, before, in_window = _review(runs_db, scenario).merger_arms

    assert [(a.model, a.role, a.window_start, a.window_end, a.tier_filter)
            for a in (candidate, before, in_window)] == [
        (FABLE, 'merger', APPLY.isoformat(), END.isoformat(), 'any'),
        ('opus', 'merger', BASE.isoformat(), APPLY.isoformat(), 'any'),
        ('opus', 'merger', APPLY.isoformat(), END.isoformat(), 'any'),
    ]
    assert len({a.label for a in (candidate, before, in_window)}) == 3
    # the run completing exactly at the window end is out
    assert (candidate.outcomes.runs, candidate.outcomes.cost_total_usd) == (2, 10.0)
    assert candidate.outcomes.merge_states == (('conflict', 1), ('done', 1))
    assert candidate.outcomes.over_flat_ceiling == 1
    assert (before.outcomes.runs, before.outcomes.turn_cap_kills) == (2, 1)
    assert before.outcomes.dispatch_caps == ((50, 2),)
    assert in_window.outcomes.runs == 0
    assert all(a.dispositions is None for a in (candidate, before, in_window))


def test_the_steward_arms_split_each_window_by_tier_and_carry_dispositions(
    runs_db, scenario
):
    candidate, window_low, before_like, before_low = _review(runs_db, scenario).steward_arms

    assert [(a.model, a.window_start, a.window_end, a.tier_filter)
            for a in (candidate, window_low, before_like, before_low)] == [
        (FABLE, APPLY.isoformat(), END.isoformat(), 'tier >= 1'),
        ('opus', APPLY.isoformat(), END.isoformat(), 'tier < 1'),
        ('opus', BASE.isoformat(), APPLY.isoformat(), 'tier >= 1'),
        ('opus', BASE.isoformat(), APPLY.isoformat(), 'tier < 1'),
    ]
    assert [a.outcomes.runs for a in (candidate, window_low, before_like, before_low)] == [
        2, 1, 1, 0,
    ]
    Disposition = escalation_ladder.StewardDisposition
    assert candidate.dispositions.count(Disposition.PROMOTED_TO_L1) == 1
    assert candidate.dispositions.count(Disposition.RESOLVED_IN_PLACE) == 1
    assert window_low.dispositions.count(Disposition.AUTO_DISMISSED) == 1
    assert before_like.dispositions.count(Disposition.RESOLVED_IN_PLACE) == 1
    assert before_low.dispositions.decided == 0


def test_a_steward_run_no_arm_covers_is_reported_rather_than_dropped(runs_db, scenario):
    """An opus steward at tier >= 1 inside the window is the fall-through a
    spent Fable ceiling would produce: it belongs to no fixed arm, so it must
    surface on its own."""
    _run(runs_db, model='opus', role='steward', task_id='306', start=timedelta(days=8),
         minutes=5, cost_usd=0.5, routing_tier=1)

    result = _review(runs_db, scenario)

    assert [(model, r.task_id, r.routing_tier)
            for model, r in result.steward_runs_in_no_arm] == [('opus', '306', 1)]


def test_the_review_carries_integrity_ladder_cost_and_leakage_sections(runs_db, scenario):
    result = _review(runs_db, scenario)

    assert (result.window_start, result.window_end, result.baseline_start) == (
        APPLY.isoformat(), END.isoformat(), BASE.isoformat(),
    )
    assert result.steward_runs_in_no_arm == ()
    assert result.drop_guard_window == ()
    assert [e.task_id for e in result.drop_guard_baseline] == ['901']
    assert [(r.task_id, r.reopened_at) for r in result.conflict_reopens] == [
        ('102', _at(days=2)),
    ]
    assert (result.l2_window.filed, result.l2_window.watcher_filed,
            result.l2_window.close_only) == (1, 1, 1)
    assert (result.l2_baseline.filed, result.l2_baseline.close_only) == (1, 0)
    assert (result.escalations_loaded, result.escalations_skipped,
            result.oldest_archive_date) == (7, 0, '2026-09-01')
    assert len(result.daily_spend) == DAYS
    assert sum(s.total_usd for s in result.daily_spend) == pytest.approx(16.0)
    assert result.peak_24h is not None
    assert (result.peak_24h.total_usd, result.peak_24h.at_or_over_ceiling) == (6.0, False)
    assert [r.task_id for r in result.model_rejections] == ['305']
    assert [(t.task_id, t.resolved_model) for t in result.ceiling_trips] == [('305', 'opus')]
    assert result.scoped_hits == (('max-e', 1, _at(days=2)),)
    assert [r.service for r in result.restarts] == ['orchestrator']
    assert result.containment.unexpected_roles == ('implementer',)
    assert dict((u.role, u.count) for u in result.containment.by_role)['merger'] == 2


def _cells(line):
    return [cell.strip() for cell in line.strip().strip('|').split('|')]


def _table_rows(markdown):
    """Every markdown table row in *markdown*, as a {header: cell} dict."""
    lines = markdown.splitlines()
    rows = []
    for i, line in enumerate(lines[:-1]):
        if line.startswith('|') and lines[i + 1].startswith('|---'):
            header = _cells(line)
            for body in lines[i + 2:]:
                if not body.startswith('|'):
                    break
                rows.append(dict(zip(header, _cells(body), strict=True)))
    return rows


def _arm_cells(markdown, label):
    """Every cell the rendered tables give *label*'s arm, merged across tables."""
    merged = {}
    for row in _table_rows(markdown):
        if row.get('arm') == label:
            merged.update(row)
    return merged


def test_the_markdown_has_one_section_per_item_each_stating_its_windows(runs_db, scenario):
    markdown = review_model_admission.render_markdown(_review(runs_db, scenario))

    headings = [line for line in markdown.splitlines() if line.startswith('### ')]
    assert len(headings) == 4   # merger, steward and L2, cost and caps, leakage
    assert all(APPLY.isoformat() in h and END.isoformat() in h for h in headings)
    assert all(BASE.isoformat() in h for h in headings[:2])


def test_the_markdown_renders_the_measured_rates_with_their_denominators(runs_db, scenario):
    result = _review(runs_db, scenario)
    markdown = review_model_admission.render_markdown(result)

    before = _arm_cells(markdown, result.merger_arms[1].label)
    assert before['turn-cap kills'] == '1/2 (50.0%)'
    candidate = _arm_cells(markdown, result.merger_arms[0].label)
    assert candidate['duration max s'] == '6000'
    steward = _arm_cells(markdown, result.steward_arms[0].label)
    assert steward['promoted to L1'] == '1'
    assert steward['promoted share'] == '1/2 (50.0%)'
    roles = {row['role']: row for row in _table_rows(markdown) if 'admitted' in row}
    assert roles['implementer']['admitted'] == 'False'


def _cli(runs_db_path, escalations, *extra):
    return review_model_admission.main([
        '--model', FABLE, '--expect-roles', 'merger,steward',
        '--apply', APPLY.isoformat(), '--days', str(DAYS),
        '--baseline-days', str(BASELINE_DAYS), '--ceiling', '150',
        '--runs-db', str(runs_db_path), '--escalations-dir', str(escalations), *extra,
    ])


def test_main_prints_the_review_and_leaves_the_store_unchanged(
    runs_db, runs_db_path, scenario, capsys
):
    expected = review_model_admission.render_markdown(_review(runs_db, scenario))
    before = (runs_db_path.read_bytes(), runs_db_path.stat().st_mtime_ns)

    assert _cli(runs_db_path, scenario) == 0

    assert capsys.readouterr().out.strip() == expected.strip()
    assert (runs_db_path.read_bytes(), runs_db_path.stat().st_mtime_ns) == before


def test_main_fails_loudly_on_a_missing_escalations_dir(runs_db_path, tmp_path):
    with pytest.raises(FileNotFoundError):
        _cli(runs_db_path, tmp_path / 'no-such-dir')


@pytest.mark.parametrize('flag', ['--days', '--baseline-days'])
@pytest.mark.parametrize('value', ['0', '-1', 'x'])
def test_a_non_positive_window_length_is_rejected_at_the_boundary(
    runs_db_path, scenario, flag, value, capsys
):
    with pytest.raises(SystemExit) as exit_info:
        _cli(runs_db_path, scenario, flag, value)

    assert exit_info.value.code != 0
