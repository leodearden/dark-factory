"""scripts/effort_experiment_look.py: the effort experiment's weekly sequential look.

Each look is placed by an injected clock. Every root is a synthetic cohort
ledger built in tmp_path through the throughput_cohort_fixtures builders; no
test reads a live store.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import effort_experiment_look as look
import pytest
import routing_experiment_readout as readout
from throughput_cohort_fixtures import (
    CONTROL_EFFORT,
    TREATED_RULE_ID,
    CohortTask,
    Dispatch,
    arm_task_ids,
    build_cohort_root,
    planted_cohort,
    seed_tasks,
)

SALT = 'effort-exp-look-test'
PREFIX = 'effort-impl-high'
SINCE = datetime(2026, 11, 2, tzinfo=UTC)
WEEK = timedelta(days=7)
LAG = timedelta(days=7)
DECIDING_N_PER_ARM = 20
SMALL_N_PER_ARM = 5
PLANTED_CUT = 0.6
PLANTED_LOSS = 1 / 0.6
IDENTICAL_ARMS = 1.0
NOT_DUE = 75
UNDECIDED = 75


def _now_at_week(k: int) -> datetime:
    """The instant whose look week, floor((now - lag - SINCE) / 7 days), is *k*."""
    return SINCE + LAG + k * WEEK


def _argv(root: Path, *extra: str) -> list[str]:
    return [
        '--salt', SALT, '--rule-prefix', PREFIX, '--roots', str(root),
        '--resamples', '1000', '--seed', '7', *extra,
    ]


def _look(argv: list[str], now: datetime) -> int:
    """The look's exit status, whether main returns it or a parser raises it."""
    try:
        return look.main(argv, clock=lambda: now)
    except SystemExit as exc:
        return exc.code


def _last_json(out: str) -> dict:
    return json.loads(out.splitlines()[-1])


def _verdict_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith('VERDICT')]


@pytest.fixture
def cohort_root(tmp_path, make_seeding_runs_db):
    """Factory: a root holding planted_cohort at *factor*, *n_per_arm* tasks per arm, and its connection."""
    def _build(factor: float, n_per_arm: int):
        root = tmp_path / f'df-{factor:.4f}-{n_per_arm}'
        conn = build_cohort_root(root, make_seeding_runs_db, planted_cohort(
            SALT, SINCE, n_per_arm=n_per_arm, treated_cost_factor=factor,
        ))
        return root, conn

    return _build


@pytest.fixture
def small_aa_root(cohort_root) -> Path:
    root, _ = cohort_root(IDENTICAL_ARMS, SMALL_N_PER_ARM)
    return root


def test_a_look_before_the_first_look_week_is_not_due(small_aa_root, capsys):
    assert _look(_argv(small_aa_root), SINCE + timedelta(days=14)) == NOT_DUE

    out = capsys.readouterr().out
    assert _verdict_lines(out) == []
    summary = _last_json(out)
    assert summary['look_week'] == 1
    assert summary['verdict'] == 'not-due'
    assert summary['since'] == SINCE.isoformat()


def test_a_planted_cut_at_week_2_is_a_gain_with_the_readouts_json_line(cohort_root, capsys):
    root, _ = cohort_root(PLANTED_CUT, DECIDING_N_PER_ARM)
    now = _now_at_week(2)

    assert _look(_argv(root), now) == 0

    out = capsys.readouterr().out
    assert _verdict_lines(out) == ['VERDICT gain']
    summary = _last_json(out)
    assert readout.main([
        '--throughput', *_argv(root), '--read-at', now.isoformat(), '--cohort-days', '14',
        '--interval-level', '0.97',
    ]) == 0
    readout_summary = _last_json(capsys.readouterr().out)
    assert summary == {**readout_summary, 'look_week': 2, 'verdict': 'gain'}
    assert summary['cohort_days'] == 14
    assert summary['interval_level'] == 0.97


def test_a_planted_loss_at_week_2_is_a_loss(cohort_root, capsys):
    root, _ = cohort_root(PLANTED_LOSS, DECIDING_N_PER_ARM)

    assert _look(_argv(root), _now_at_week(2)) == 0

    out = capsys.readouterr().out
    assert _verdict_lines(out) == ['VERDICT loss']
    assert _last_json(out)['verdict'] == 'loss'


def test_identical_arms_at_the_last_look_week_are_no_difference_at_cap(small_aa_root, capsys):
    assert _look(_argv(small_aa_root), _now_at_week(8)) == 0

    out = capsys.readouterr().out
    assert _verdict_lines(out) == ['VERDICT no-difference-at-cap']
    summary = _last_json(out)
    assert (summary['look_week'], summary['cohort_days']) == (8, 56)


def test_a_look_past_the_last_week_keeps_the_cohort_capped_at_56_days(small_aa_root, capsys):
    assert _look(_argv(small_aa_root), _now_at_week(10)) == 0

    out = capsys.readouterr().out
    assert _verdict_lines(out) == ['VERDICT no-difference-at-cap']
    summary = _last_json(out)
    assert (summary['look_week'], summary['cohort_days']) == (10, 56)


def test_identical_arms_before_the_cap_are_undecided(small_aa_root, capsys):
    assert _look(_argv(small_aa_root), _now_at_week(3)) == UNDECIDED

    out = capsys.readouterr().out
    assert _verdict_lines(out) == []
    summary = _last_json(out)
    assert (summary['look_week'], summary['verdict'], summary['cohort_days']) == (3, 'undecided', 21)


def test_a_treated_dispatch_at_control_effort_fails_the_look_instead_of_waiting(
    cohort_root, capsys,
):
    root, conn = cohort_root(IDENTICAL_ARMS, SMALL_N_PER_ARM)
    (treated, *_), _ = arm_task_ids(SALT, 2, 1, SMALL_N_PER_ARM)
    seed_tasks(conn, [CohortTask(task_id=treated, dispatches=(
        Dispatch(at=SINCE + timedelta(days=3), effort=CONTROL_EFFORT, rule_id=TREATED_RULE_ID),
    ))])

    assert _look(_argv(root), _now_at_week(2)) == 1

    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line.startswith('COMPLIANCE FAILURE')]
    assert _verdict_lines(out) == []
    assert _last_json(out)['verdict'] == 'compliance-failure'


def test_a_ledger_without_a_prefixed_dispatch_fails_the_look_naming_the_prefix(
    tmp_path, make_seeding_runs_db, capsys,
):
    root = tmp_path / 'df'
    build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=1, dispatches=(Dispatch(at=SINCE, effort=CONTROL_EFFORT),)),
    ])

    assert _look(_argv(root), _now_at_week(2)) == 1

    err = capsys.readouterr().err
    assert 'no routing_decision carries the rule prefix' in err
    assert PREFIX in err


@pytest.mark.parametrize('bad', [('--treated-bucket', '5'), ('--lag-days', 'x')])
def test_a_bad_option_fails_the_look_rather_than_exiting_as_usage(tmp_path, bad):
    assert _look(_argv(tmp_path, *bad), _now_at_week(2)) == 1


def test_a_passthrough_read_at_is_overridden_by_the_clock(small_aa_root, capsys):
    now = _now_at_week(3)

    _look(_argv(small_aa_root, '--read-at', '2026-01-01T00:00:00+00:00'), now)

    assert _last_json(capsys.readouterr().out)['read_at'] == now.isoformat()
