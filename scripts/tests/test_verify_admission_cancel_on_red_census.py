"""Tests for verify_admission_cancel_on_red_census.py — which admission legs a cancel-on-red would avoid.

A leg is avoidable when it was granted after the end of a red sibling. Under
BY_MODULE a module is judged by its final attempt, and a module is never its
own sibling. BY_LEG is the literal one-leg-one-unit reading. Times are epoch
floats; a leg ends at granted_at + held_secs.
"""
from __future__ import annotations

import verify_admission_cancel_on_red_census as mod


def _leg(module, granted_at, held_secs, *, red=False, arrived_at=0.0):
    return mod.Leg(module=module, arrived_at=arrived_at, granted_at=granted_at, held_secs=held_secs, red=red)


def _plan(*legs, role='task', plan_id='p1'):
    return mod.Plan(plan_id=plan_id, role=role, legs=tuple(legs))


def test_leg_granted_after_a_red_sibling_ends_is_avoidable():
    a = _leg('a', 0.0, 40.0, red=True)
    b = _leg('b', 41.0, 10.0)

    assert mod.avoidable_legs(_plan(a, b)) == (mod.AvoidableLeg(leg=b, cutoff=40.0),)


def test_leg_already_running_when_the_sibling_goes_red_is_not_avoidable():
    a = _leg('a', 0.0, 40.0, red=True)
    b = _leg('b', 20.0, 10.0)

    assert mod.avoidable_legs(_plan(a, b)) == ()


def test_red_flag_alone_decides_redness_so_a_timeout_counts():
    timed_out_with_rc_zero = _leg('a', 0.0, 1200.0, red=True)
    b = _leg('b', 1201.0, 10.0)

    assert mod.avoidable_legs(_plan(timed_out_with_rc_zero, b)) == (mod.AvoidableLeg(leg=b, cutoff=1200.0),)


def test_a_module_whose_retry_passed_is_green_and_not_its_own_sibling():
    a_timeout = _leg('a', 0.0, 1200.0, red=True)
    a_retry_green = _leg('a', 1201.0, 1199.0)
    b = _leg('b', 2401.0, 10.0)

    assert mod.avoidable_legs(_plan(a_timeout, a_retry_green, b)) == ()


def test_a_module_red_on_every_attempt_cuts_off_at_its_final_attempts_end():
    a_timeout = _leg('a', 0.0, 1200.0, red=True)
    a_retry_red = _leg('a', 1201.0, 1199.0, red=True)
    b = _leg('b', 2401.0, 10.0)

    assert mod.avoidable_legs(_plan(a_timeout, a_retry_red, b)) == (mod.AvoidableLeg(leg=b, cutoff=2400.0),)


def test_queued_at_red_tells_a_leg_already_waiting_from_one_that_arrived_later():
    a = _leg('a', 0.0, 40.0, red=True)
    arrived_after_red = _leg('b', 200.0, 10.0, arrived_at=100.0)
    queued_before_red = _leg('c', 300.0, 10.0, arrived_at=30.0)

    found = mod.avoidable_legs(_plan(a, arrived_after_red, queued_before_red))

    assert [(item.leg, item.queued_at_red) for item in found] == [
        (arrived_after_red, False),
        (queued_before_red, True),
    ]


def test_every_later_leg_is_measured_against_the_earliest_red_sibling():
    a = _leg('a', 0.0, 40.0, red=True)
    b = _leg('b', 41.0, 59.0, red=True)
    c = _leg('c', 101.0, 10.0)

    assert mod.avoidable_legs(_plan(a, b, c)) == (
        mod.AvoidableLeg(leg=b, cutoff=40.0),
        mod.AvoidableLeg(leg=c, cutoff=40.0),
    )


def test_a_single_module_red_plan_has_nothing_to_avoid():
    assert mod.avoidable_legs(_plan(_leg('a', 0.0, 40.0, red=True))) == ()


def test_an_all_green_multi_module_plan_has_nothing_to_avoid():
    assert mod.avoidable_legs(_plan(_leg('a', 0.0, 40.0), _leg('b', 41.0, 10.0))) == ()


def test_by_leg_is_the_literal_reading_that_counts_a_passing_retry():
    a_timeout = _leg('a', 0.0, 1200.0, red=True)
    a_retry_green = _leg('a', 1201.0, 1199.0)
    b = _leg('b', 2401.0, 10.0)

    assert mod.avoidable_legs(_plan(a_timeout, a_retry_green, b), group=mod.BY_LEG) == (
        mod.AvoidableLeg(leg=a_retry_green, cutoff=1200.0),
        mod.AvoidableLeg(leg=b, cutoff=1200.0),
    )


def test_a_none_module_is_one_module_so_its_legs_are_attempts_of_one_unit():
    whole_project_timeout = _leg(None, 0.0, 1200.0, red=True)
    whole_project_retry_green = _leg(None, 1201.0, 1199.0)

    assert mod.avoidable_legs(_plan(whole_project_timeout, whole_project_retry_green)) == ()


def test_a_none_module_is_a_sibling_of_a_named_module():
    whole_project = _leg(None, 0.0, 40.0, red=True)
    named = _leg('a', 41.0, 10.0)

    assert mod.avoidable_legs(_plan(whole_project, named)) == (mod.AvoidableLeg(leg=named, cutoff=40.0),)
