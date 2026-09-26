"""Tests for scripts/review_model_admission.py.

Hermetic: the pure summaries run over InvocationRecord values built here, and
the scans run against the synthetic runs.db from conftest.py's ``runs_db``
fixture and escalation records written into tmp_path.
"""
from datetime import UTC, datetime, timedelta

import audit_model_admission
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
