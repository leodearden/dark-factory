"""Tests for verify_admission_cancel_on_red_census.py — which admission legs a cancel-on-red would avoid.

A leg is avoidable when it was granted after the end of a red sibling. Under
BY_MODULE a module is judged by its final attempt, and a module is never its
own sibling. BY_LEG is the literal one-leg-one-unit reading. Times are epoch
floats; a leg ends at granted_at + held_secs.
"""
from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import datetime

import pytest
import verify_admission_cancel_on_red_census as mod


def _leg(module, granted_at, held_secs, *, rc=0, timed_out=False, arrived_at=0.0):
    return mod.Leg(
        module=module, arrived_at=arrived_at, granted_at=granted_at, held_secs=held_secs, rc=rc, timed_out=timed_out,
    )


def _plan(*legs, role='task', plan_id='p1'):
    return mod.Plan(plan_id=plan_id, role=role, legs=tuple(legs))


def test_leg_granted_after_a_red_sibling_ends_is_avoidable():
    a = _leg('a', 0.0, 40.0, rc=1)
    b = _leg('b', 41.0, 10.0)

    assert mod.avoidable_legs(_plan(a, b)) == (mod.AvoidableLeg(leg=b, cutoff=40.0),)


def test_leg_granted_exactly_as_the_red_sibling_ends_is_not_avoidable():
    a = _leg('a', 0.0, 40.0, rc=1)
    b = _leg('b', 40.0, 10.0)

    assert mod.avoidable_legs(_plan(a, b)) == ()


def test_leg_already_running_when_the_sibling_goes_red_is_not_avoidable():
    a = _leg('a', 0.0, 40.0, rc=1)
    b = _leg('b', 20.0, 10.0)

    assert mod.avoidable_legs(_plan(a, b)) == ()


def test_a_timed_out_leg_is_red_even_with_rc_zero():
    timed_out_with_rc_zero = _leg('a', 0.0, 1200.0, timed_out=True)
    b = _leg('b', 1201.0, 10.0)

    assert mod.avoidable_legs(_plan(timed_out_with_rc_zero, b)) == (mod.AvoidableLeg(leg=b, cutoff=1200.0),)


def test_a_module_whose_retry_passed_is_green_and_not_its_own_sibling():
    a_timeout = _leg('a', 0.0, 1200.0, timed_out=True)
    a_retry_green = _leg('a', 1201.0, 1199.0)
    b = _leg('b', 2401.0, 10.0)

    assert mod.avoidable_legs(_plan(a_timeout, a_retry_green, b)) == ()


def test_a_module_red_on_every_attempt_cuts_off_at_its_final_attempts_end():
    a_timeout = _leg('a', 0.0, 1200.0, timed_out=True)
    a_retry_red = _leg('a', 1201.0, 1199.0, rc=1)
    b = _leg('b', 2401.0, 10.0)

    assert mod.avoidable_legs(_plan(a_timeout, a_retry_red, b)) == (mod.AvoidableLeg(leg=b, cutoff=2400.0),)


def test_queued_at_red_tells_a_leg_already_waiting_from_one_that_arrived_later():
    a = _leg('a', 0.0, 40.0, rc=1)
    arrived_after_red = _leg('b', 200.0, 10.0, arrived_at=100.0)
    queued_before_red = _leg('c', 300.0, 10.0, arrived_at=30.0)

    found = mod.avoidable_legs(_plan(a, arrived_after_red, queued_before_red))

    assert [(item.leg, item.queued_at_red) for item in found] == [
        (arrived_after_red, False),
        (queued_before_red, True),
    ]


def test_every_later_leg_is_measured_against_the_earliest_red_sibling():
    a = _leg('a', 0.0, 40.0, rc=1)
    b = _leg('b', 41.0, 59.0, rc=1)
    c = _leg('c', 101.0, 10.0)

    assert mod.avoidable_legs(_plan(a, b, c)) == (
        mod.AvoidableLeg(leg=b, cutoff=40.0),
        mod.AvoidableLeg(leg=c, cutoff=40.0),
    )


def test_a_single_module_red_plan_has_nothing_to_avoid():
    assert mod.avoidable_legs(_plan(_leg('a', 0.0, 40.0, rc=1))) == ()


def test_an_all_green_multi_module_plan_has_nothing_to_avoid():
    assert mod.avoidable_legs(_plan(_leg('a', 0.0, 40.0), _leg('b', 41.0, 10.0))) == ()


def test_by_leg_is_the_literal_reading_that_counts_a_passing_retry():
    a_timeout = _leg('a', 0.0, 1200.0, timed_out=True)
    a_retry_green = _leg('a', 1201.0, 1199.0)
    b = _leg('b', 2401.0, 10.0)

    assert mod.avoidable_legs(_plan(a_timeout, a_retry_green, b), group=mod.BY_LEG) == (
        mod.AvoidableLeg(leg=a_retry_green, cutoff=1200.0),
        mod.AvoidableLeg(leg=b, cutoff=1200.0),
    )


def test_a_none_module_is_one_module_so_its_legs_are_attempts_of_one_unit():
    whole_project_timeout = _leg(None, 0.0, 1200.0, timed_out=True)
    whole_project_retry_green = _leg(None, 1201.0, 1199.0)

    assert mod.avoidable_legs(_plan(whole_project_timeout, whole_project_retry_green)) == ()


def test_a_none_module_is_a_sibling_of_a_named_module():
    whole_project = _leg(None, 0.0, 40.0, rc=1)
    named = _leg('a', 41.0, 10.0)

    assert mod.avoidable_legs(_plan(whole_project, named)) == (mod.AvoidableLeg(leg=named, cutoff=40.0),)


def _epoch(iso):
    return datetime.fromisoformat(iso).timestamp()


def _leg_payload(plan_id, module, arrived_at, granted_at, held_secs, *, rc=0, timed_out=False, order='arrival'):
    return {
        'plan_id': plan_id, 'module': module, 'scope_kind': 'full_suite', 'order': order,
        'arrived_at': arrived_at, 'granted_at': granted_at, 'waited_secs': 1.0, 'depth_at_arrival': 0,
        'slot_held': True, 'flock_wait_secs': 0.01, 'held_secs': held_secs, 'rc': rc, 'timed_out': timed_out,
    }


def _plan_event(timestamp, plan_id, legs, *, role='task', order: str | None = 'arrival', outcome='returned'):
    return {
        'timestamp': timestamp, 'event_type': 'verify_admission_plan', 'task_id': '42', 'phase': 'verify',
        'role': role,
        'data': {
            'plan_id': plan_id, 'task_id': '42', 'role': role, 'outcome': outcome, 'order': order,
            'plan_arrival_at': timestamp, 'completed_at': timestamp, 'legs': legs,
        },
    }


ARRIVED = '2026-10-17T06:00:00+00:00'
GRANTED = '2026-10-17T06:00:10+00:00'


@pytest.fixture
def mixed_runs_db(make_runs_db):
    return make_runs_db([
        _plan_event('2026-10-16T20:00:00+00:00', 'old', [
            _leg_payload('old', 'a', ARRIVED, GRANTED, 5.0, order='unordered'),
        ], order='unordered'),
        _plan_event('2026-10-17T06:07:41+00:00', 'first', [
            _leg_payload('first', 'a', ARRIVED, GRANTED, 30.5, rc=1),
            _leg_payload('first', 'b', ARRIVED, '2026-10-17T06:00:41+00:00', 1200.0, rc=0, timed_out=True),
            _leg_payload('first', 'c', ARRIVED, '2026-10-17T06:20:41+00:00', 7.0),
        ]),
        _plan_event('2026-10-18T00:00:00+00:00', 'raised', [], order=None, outcome='raised'),
        {'timestamp': '2026-10-18T00:00:01+00:00', 'event_type': 'workflow_verify', 'data': {'passed': True}},
        _plan_event('2026-10-19T06:07:41+00:00', 'second', [
            _leg_payload('second', None, ARRIVED, GRANTED, 2.0),
        ], role='background'),
    ])


def test_loader_keeps_only_returned_arrival_plans_in_row_order(mixed_runs_db):
    sample = mod.load_arrival_plans(mixed_runs_db)

    assert [(plan.plan_id, plan.role) for plan in sample.plans] == [('first', 'task'), ('second', 'background')]


def test_loader_parses_legs_to_epoch_floats_keeping_rc_and_timed_out_apart(mixed_runs_db):
    first = mod.load_arrival_plans(mixed_runs_db).plans[0]

    assert first.legs == (
        mod.Leg(
            module='a', arrived_at=_epoch(ARRIVED), granted_at=_epoch(GRANTED),
            held_secs=30.5, rc=1, timed_out=False,
        ),
        mod.Leg(
            module='b', arrived_at=_epoch(ARRIVED), granted_at=_epoch('2026-10-17T06:00:41+00:00'),
            held_secs=1200.0, rc=0, timed_out=True,
        ),
        mod.Leg(
            module='c', arrived_at=_epoch(ARRIVED), granted_at=_epoch('2026-10-17T06:20:41+00:00'),
            held_secs=7.0, rc=0, timed_out=False,
        ),
    )


def test_loader_window_spans_the_first_and_last_kept_rows(mixed_runs_db):
    sample = mod.load_arrival_plans(mixed_runs_db)

    assert (sample.window_start, sample.window_end, sample.window_days) == (
        '2026-10-17T06:07:41+00:00', '2026-10-19T06:07:41+00:00', 2.0,
    )


def test_loader_opens_read_only_so_a_missing_db_is_an_error_not_a_new_file(tmp_path):
    missing = tmp_path / 'absent.db'

    with pytest.raises(sqlite3.OperationalError):
        mod.load_arrival_plans(missing)
    assert not missing.exists()


def test_loader_never_creates_a_file_when_the_path_holds_a_uri_fragment_mark(tmp_path):
    missing = tmp_path / 'runs#1.db'

    with pytest.raises(sqlite3.OperationalError):
        mod.load_arrival_plans(missing)
    assert list(tmp_path.iterdir()) == []


def test_loader_reads_a_db_whose_path_holds_uri_metacharacters(make_runs_db, tmp_path):
    odd = tmp_path / 'odd?mode=rwc#dir%20'
    odd.mkdir()
    db = make_runs_db([
        _plan_event('2026-10-17T06:07:41+00:00', 'only', [_leg_payload('only', 'a', ARRIVED, GRANTED, 5.0)]),
    ], directory=odd)

    assert [plan.plan_id for plan in mod.load_arrival_plans(db).plans] == ['only']


def test_loader_on_a_store_without_arrival_rows_returns_an_empty_window(make_runs_db):
    db = make_runs_db([
        _plan_event('2026-10-16T20:00:00+00:00', 'old', [
            _leg_payload('old', 'a', ARRIVED, GRANTED, 5.0, order='unordered'),
        ], order='unordered'),
    ])

    sample = mod.load_arrival_plans(db)

    assert (sample.plans, sample.window_days) == ((), 0.0)



# measure() fixture. Busy = 150 + 1550 + 180 + 10 + 50 = 1940 s over 2 days.
#   red:        task  a red 0-40; b 50-150 (queued at red); d red 160-170 (arrived 100) -> b, d avoidable = 110 s
#   retry:      task  fm timed out 1000-2200, fm retry green 2201-2501, c 2600-2650 -> BY_LEG only: 300 + 50
#   sweep:      background x red 3000-3020; y 3030-3100; r red twice (rc, not a timeout) 3100-3150, 3151-3191
#               -> 160 s, out of headline
#   single:     task  one green leg 4000-4010
#   waiter:     task  w waits 80-120, w2 waits 100-150: their union covers 70 s of b's 50-150 hold
RED = _plan(
    _leg('a', 0.0, 40.0, rc=1),
    _leg('b', 50.0, 100.0),
    _leg('d', 160.0, 10.0, rc=1, arrived_at=100.0),
    plan_id='red',
)
RETRY = _plan(
    _leg('fm', 1000.0, 1200.0, timed_out=True, arrived_at=1000.0),
    _leg('fm', 2201.0, 300.0, arrived_at=1000.0),
    _leg('c', 2600.0, 50.0, arrived_at=1000.0),
    plan_id='retry',
)
SWEEP = _plan(
    _leg('x', 3000.0, 20.0, rc=1, arrived_at=3000.0),
    _leg('y', 3030.0, 70.0, arrived_at=3000.0),
    _leg('r', 3100.0, 50.0, rc=1, arrived_at=3000.0),
    _leg('r', 3151.0, 40.0, rc=1, arrived_at=3000.0),
    role='background',
    plan_id='sweep',
)
SINGLE = _plan(_leg('z', 4000.0, 10.0, arrived_at=4000.0), plan_id='single')
WAITER = _plan(
    _leg('w', 120.0, 30.0, arrived_at=80.0),
    _leg('w2', 150.0, 20.0, arrived_at=100.0),
    plan_id='waiter',
)
SAMPLE = mod.Sample(
    plans=(RED, RETRY, SWEEP, SINGLE, WAITER),
    window_start='2026-10-17T00:00:00+00:00',
    window_end='2026-10-19T00:00:00+00:00',
    window_days=2.0,
)
BUSY = 1940.0


def _measure(sample=SAMPLE, resamples=500, seed=1):
    return mod.measure('alpha', sample, resamples=resamples, seed=seed)


def test_measure_copies_the_window_and_counts_plans_by_role():
    m = _measure()

    assert (m.project, m.window_start, m.window_end, m.window_days) == (
        'alpha', SAMPLE.window_start, SAMPLE.window_end, 2.0,
    )
    assert m.plans == mod.PlanCounts(total=5, task=4, background=1)


def test_n_counts_task_plans_whose_red_module_is_red_on_its_final_attempt():
    assert _measure().multi_module_red_plans == 1


def test_busy_is_every_leg_of_every_role():
    assert _measure().busy_secs == BUSY


def test_headline_is_task_role_by_module_and_background_is_reported_apart():
    m = _measure()

    assert (m.avoidable_secs, m.avoidable_share) == (110.0, 110.0 / BUSY)
    assert m.background_avoidable_secs == 160.0


def test_literal_figure_adds_exactly_the_passing_retry_and_its_later_sibling():
    m = _measure()

    assert m.literal_avoidable_secs - m.avoidable_secs == 300.0 + 50.0
    assert m.literal_avoidable_share == m.literal_avoidable_secs / BUSY


def test_share_interval_is_a_seeded_bootstrap_around_the_point():
    first, second = _measure(seed=3), _measure(seed=3)
    low, high = first.share_interval

    assert first == second
    assert low <= first.avoidable_share <= high
    assert low < high


def test_share_interval_collapses_when_every_plan_contributes_alike():
    alike = mod.Sample(plans=(RED, RED, RED), window_start='s', window_end='e', window_days=1.0)
    m = _measure(alike)

    assert m.share_interval == (m.avoidable_share, m.avoidable_share)


def test_imposed_wait_is_the_union_of_other_plans_waiting_during_avoidable_holds():
    assert _measure().imposed_wait_secs == 70.0


def test_cost_side_counts_what_a_cancel_would_have_thrown_away():
    assert _measure().cost == mod.CostSide(
        avoidable_legs=2,
        avoidable_legs_queued_at_red=1,
        avoidable_modules_red=1,
        plans_losing_a_red=1,
    )


def test_retried_attempts_decompose_the_non_final_attempts_of_every_role_and_count_the_timeouts_apart():
    assert _measure().retried_attempts == mod.RetriedAttempts(
        attempts=2,
        timed_out=1,
        held_secs=1250.0,
        share_of_busy=1250.0 / BUSY,
        modules=2,
        modules_final_green=1,
    )


def test_avoidable_hours_per_day_spreads_the_headline_over_the_window():
    assert _measure().avoidable_hours_per_day == 110.0 / 3600 / 2.0


def test_an_empty_sample_measures_zero_without_dividing_by_zero():
    m = _measure(mod.Sample(plans=(), window_start='', window_end='', window_days=0.0))

    assert (m.busy_secs, m.avoidable_share, m.literal_avoidable_share, m.share_interval) == (0.0, 0.0, 0.0, (0.0, 0.0))
    assert (m.avoidable_hours_per_day, m.retried_attempts.share_of_busy) == (0.0, 0.0)



STRADDLING = (0.0, 0.1)


def _judged(*, window_days, n, share=0.01, interval=STRADDLING):
    m = dataclasses.replace(
        _measure(), window_days=window_days, multi_module_red_plans=n, avoidable_share=share, share_interval=interval,
    )
    return mod.verdict(m)


def test_decision_constants_are_the_tasks_rule():
    assert (mod.DECISION_SHARE, mod.MIN_WINDOW_DAYS, mod.MIN_MULTI_MODULE_RED_PLANS) == (0.03, 14.0, 40)


def test_a_full_window_decides_on_the_point_estimate():
    assert _judged(window_days=14.0, n=40, share=0.030) is mod.Verdict.FILE
    assert _judged(window_days=14.0, n=40, share=0.0299) is mod.Verdict.DECLINE


def test_too_few_multi_module_red_plans_is_thin_whatever_the_share_or_window():
    assert _judged(window_days=14.0, n=39, share=0.5) is mod.Verdict.THIN_DATA
    assert _judged(window_days=13.7, n=39, interval=(0.035, 0.074)) is mod.Verdict.THIN_DATA
    assert _judged(window_days=13.7, n=39, interval=(0.0015, 0.023)) is mod.Verdict.THIN_DATA


def test_a_short_window_decides_only_when_the_whole_interval_is_on_one_side():
    assert _judged(window_days=13.7, n=120, interval=(0.0015, 0.023)) is mod.Verdict.DECLINE
    assert _judged(window_days=13.7, n=120, interval=(0.035, 0.074)) is mod.Verdict.FILE
    assert _judged(window_days=13.7, n=120, interval=(0.02, 0.04)) is mod.Verdict.THIN_DATA


def test_a_zero_length_window_is_thin_even_with_a_decisive_interval():
    assert _judged(window_days=0.0, n=40, interval=(0.0015, 0.023)) is mod.Verdict.THIN_DATA


def test_verdict_values_are_the_json_spellings():
    assert [v.value for v in mod.Verdict] == ['file', 'decline', 'thin_data']



def _two_project_dbs(make_runs_db):
    alpha = make_runs_db([
        _plan_event('2026-10-17T06:07:41+00:00', 'a1', [
            _leg_payload('a1', 'a', ARRIVED, GRANTED, 30.0, rc=1),
            _leg_payload('a1', 'b', ARRIVED, '2026-10-17T06:00:41+00:00', 12.0),
        ]),
    ], name='alpha.db')
    beta = make_runs_db([
        _plan_event('2026-10-17T06:07:41+00:00', 'b1', [_leg_payload('b1', None, ARRIVED, GRANTED, 9.0)]),
    ], name='beta.db')
    return alpha, beta


def test_cli_prints_one_measurement_and_verdict_per_db_in_argument_order(make_runs_db, capsys):
    alpha, beta = _two_project_dbs(make_runs_db)

    rc = mod.main(['--db', f'alpha={alpha}', '--db', f'beta={beta}', '--resamples', '200', '--seed', '7'])

    assert rc == 0
    reports = json.loads(capsys.readouterr().out)
    assert [report['project'] for report in reports] == ['alpha', 'beta']
    measurement_fields = {field.name for field in dataclasses.fields(mod.Measurement)}
    assert all(measurement_fields <= report.keys() for report in reports)
    assert [report['verdict'] for report in reports] == ['thin_data', 'thin_data']
    assert reports[0]['avoidable_share'] == 12.0 / 42.0
    assert reports[0]['cost']['avoidable_legs'] == 1
    assert reports[0]['retried_attempts']['attempts'] == 0


def test_cli_rejects_a_db_without_a_project_name(tmp_path):
    with pytest.raises(SystemExit) as exited:
        mod.main(['--db', str(tmp_path / 'runs.db')])

    assert exited.value.code == 2


def test_cli_reports_an_unreadable_db_on_stderr_without_creating_it(tmp_path, capsys):
    missing = tmp_path / 'absent.db'

    rc = mod.main(['--db', f'gone={missing}'])

    assert rc != 0
    assert str(missing) in capsys.readouterr().err
    assert not missing.exists()
