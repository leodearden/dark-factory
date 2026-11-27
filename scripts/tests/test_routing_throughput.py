"""scripts/routing_throughput.py: landed-per-dollar by arm over a first-dispatch cohort.

Every root is a synthetic ledger built in tmp_path through the
throughput_cohort_fixtures builders; no test reads a live store.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import routing_throughput
from routing_throughput import (
    Arm,
    ArmTotals,
    IntervalSide,
    ThroughputOptions,
    ThroughputUnavailable,
    Unavailable,
    resolve_since,
    throughput_report,
)
from shared.routing_split import split_bucket
from throughput_cohort_fixtures import (
    CONTROL_EFFORT,
    TREATED_EFFORT,
    TREATED_RULE_ID,
    CohortTask,
    Dispatch,
    arm_task_ids,
    build_cohort_root,
    land_branch,
    planted_cohort,
    revert_merge,
    seed_tasks,
)

SALT = 'effort-exp-test'
PREFIX = 'effort-impl-high'
SINCE = datetime(2026, 11, 2, tzinfo=UTC)
COHORT_DAYS = 14
COHORT_END = SINCE + timedelta(days=COHORT_DAYS)
READ_AT = SINCE + timedelta(days=20)


BASE_OPTIONS = ThroughputOptions(
    salt=SALT,
    buckets=2,
    treated_bucket=1,
    roles=('implementer', 'debugger'),
    rule_prefix=PREFIX,
    treated_effort=TREATED_EFFORT,
    control_effort=CONTROL_EFFORT,
    since=None,
    cohort_days=COHORT_DAYS,
    read_at=READ_AT,
    roots=(),
    interval_level=0.90,
    seed=7,
    resamples=200,
)


def _options(*roots: Path, **overrides) -> ThroughputOptions:
    return replace(BASE_OPTIONS, roots=tuple(roots), **overrides)


def _treated(at: datetime, role: str = 'implementer') -> Dispatch:
    return Dispatch(at=at, role=role, effort=TREATED_EFFORT, rule_id=TREATED_RULE_ID)


def _control(at: datetime, role: str = 'implementer') -> Dispatch:
    return Dispatch(at=at, role=role, effort=CONTROL_EFFORT)


def _hours(n: float) -> datetime:
    return SINCE + timedelta(hours=n)


def _by_task(report) -> dict[str, routing_throughput.CohortRecord]:
    return {record.task_id: record for record in report.records}


# The known-ratio ledger: treated 3 tasks, 2 landed, $4.00 (an architect
# invocation before since included); control 2 tasks, 1 landed, $4.00. R = 2.0.
def _known_ratio_tasks() -> list[CohortTask]:
    (t0, t1, t2), (c0, c1, _) = arm_task_ids(SALT, 2, 1, 3)
    return [
        CohortTask(
            task_id=t0, dispatches=(_treated(_hours(1)),),
            invocations=(('architect', 1.0, SINCE - timedelta(days=2)),
                         ('implementer', 0.5, _hours(1))),
            landed_at=_hours(48),
        ),
        CohortTask(
            task_id=t1, dispatches=(_treated(_hours(2)),),
            invocations=(('implementer', 1.5, _hours(2)),), landed_at=_hours(72),
        ),
        CohortTask(task_id=t2, dispatches=(_treated(_hours(3)),),
                   invocations=(('implementer', 1.0, _hours(3)),)),
        CohortTask(
            task_id=c0, dispatches=(_control(_hours(1)),),
            invocations=(('implementer', 3.0, _hours(1)),), landed_at=_hours(48),
        ),
        CohortTask(task_id=c1, dispatches=(_control(_hours(2)),),
                   invocations=(('implementer', 1.0, _hours(2)),)),
    ]


@pytest.fixture
def known_root(tmp_path, make_seeding_runs_db) -> Path:
    root = tmp_path / 'df'
    build_cohort_root(root, make_seeding_runs_db, _known_ratio_tasks())
    return root


def test_the_known_ratio_ledger_reads_as_landed_per_dollar_by_arm(known_root):
    report = throughput_report(_options(known_root), SINCE)

    assert report.totals[Arm.TREATED] == ArmTotals(n=3, landed=2, cost_usd=4.0)
    assert report.totals[Arm.CONTROL] == ArmTotals(n=2, landed=1, cost_usd=4.0)
    assert report.ratio == 2.0
    assert report.since == SINCE
    assert report.cohort_end == COHORT_END


def test_each_cohort_task_sits_in_the_arm_its_split_bucket_names(known_root):
    report = throughput_report(_options(known_root), SINCE)

    assert len(report.records) == 5
    for record in report.records:
        treated = split_bucket(SALT, record.task_id, 2) == 1
        assert record.arm is (Arm.TREATED if treated else Arm.CONTROL)


def test_cost_counts_every_role_with_no_lower_bound(known_root):
    (t0, _, _), _ = arm_task_ids(SALT, 2, 1, 3)

    record = _by_task(throughput_report(_options(known_root), SINCE))[str(t0)]

    assert record.cost_usd == 1.5
    assert dict(record.cost_by_role) == {'architect': 1.0, 'implementer': 0.5}
    assert record.first_dispatch_at == _hours(1)
    assert record.landed_at == _hours(48)


def test_tasks_outside_the_cohort_are_in_neither_arm(tmp_path, make_seeding_runs_db):
    root = tmp_path / 'df'
    tasks = [
        *_known_ratio_tasks(),
        CohortTask(task_id=9001, dispatches=(
            _control(SINCE - timedelta(days=1)), _control(SINCE + timedelta(days=1)),
        )),
        CohortTask(task_id=9002, dispatches=(_control(COHORT_END),)),
        CohortTask(task_id=9003, dispatches=(_control(_hours(5), role='architect'),)),
    ]
    build_cohort_root(root, make_seeding_runs_db, tasks)

    report = throughput_report(_options(root), SINCE)

    assert {'9001', '9002', '9003'}.isdisjoint(_by_task(report))
    assert report.ratio == 2.0


def test_a_debugger_only_first_dispatch_inside_the_window_is_in_the_cohort(
    tmp_path, make_seeding_runs_db,
):
    root = tmp_path / 'df'
    build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=9004, dispatches=(_control(_hours(4), role='debugger'),)),
    ])

    report = throughput_report(_options(root), SINCE)

    assert set(_by_task(report)) == {'9004'}


def test_an_invocation_started_at_or_after_read_at_adds_no_cost(tmp_path, make_seeding_runs_db):
    root = tmp_path / 'df'
    build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=9005, dispatches=(_control(_hours(4)),), invocations=(
            ('implementer', 2.0, READ_AT - timedelta(seconds=1)),
            ('implementer', 100.0, READ_AT),
            ('reviewer', 100.0, READ_AT + timedelta(days=1)),
        )),
    ])

    record = _by_task(throughput_report(_options(root), SINCE))['9005']

    assert record.cost_usd == 2.0


def test_a_landing_at_or_after_read_at_does_not_count_as_landed(tmp_path, make_seeding_runs_db):
    root = tmp_path / 'df'
    build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=9006, dispatches=(_control(_hours(4)),), landed_at=READ_AT),
    ])

    record = _by_task(throughput_report(_options(root), SINCE))['9006']

    assert record.landed_at is None
    assert record.merge_sha is None


def test_a_train_member_superseded_in_merge_finalized_counts_as_landed(
    tmp_path, make_seeding_runs_db,
):
    root = tmp_path / 'df'
    conn = build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=9007, dispatches=(_control(_hours(4)),)),
    ])
    train_at = _hours(30)
    conn.seed_event(train_at.isoformat(), 'merge_finalized', task_id='9007', data={
        'state': 'superseded', 'merge_sha': None, 'superseded_by': '9008',
    })
    conn.seed_event(train_at.isoformat(), 'train_merged', task_id='9008', data={
        'train_id': 'coalesce-9008', 'member_task_ids': ['9007', '9008'],
        'merge_commit_sha': 'train-sha', 'base_sha': 'base-sha',
    })

    record = _by_task(throughput_report(_options(root), SINCE))['9007']

    assert (record.landed_at, record.merge_sha) == (train_at, 'train-sha')


def test_a_cohort_without_cohort_days_runs_to_read_at(known_root):
    report = throughput_report(_options(known_root, cohort_days=None), SINCE)

    assert report.cohort_end == READ_AT


def test_a_cohort_longer_than_the_read_window_ends_at_read_at(known_root):
    report = throughput_report(_options(known_root, cohort_days=60), SINCE)

    assert report.cohort_end == READ_AT


def test_since_resolves_to_the_first_prefixed_dispatch_of_any_role_across_roots(
    tmp_path, make_seeding_runs_db,
):
    first = tmp_path / 'a' / 'df'
    second = tmp_path / 'b' / 'reify'
    build_cohort_root(first, make_seeding_runs_db, [
        CohortTask(task_id=1, dispatches=(
            _control(SINCE - timedelta(days=2)), _treated(_hours(1)),
        )),
    ])
    build_cohort_root(second, make_seeding_runs_db, [
        CohortTask(task_id=2, dispatches=(
            Dispatch(at=SINCE, role='architect', effort='high', rule_id=f'{PREFIX}-architect'),
        )),
    ])

    assert resolve_since(_options(first, second)) == SINCE


def test_an_explicit_since_is_returned_as_given(known_root):
    explicit = SINCE + timedelta(days=1)

    assert resolve_since(_options(known_root, since=explicit)) == explicit


def test_a_ledger_without_a_prefixed_dispatch_cannot_resolve_since(
    tmp_path, make_seeding_runs_db,
):
    root = tmp_path / 'df'
    build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=1, dispatches=(_control(_hours(1)),)),
        CohortTask(task_id=2, dispatches=(
            Dispatch(at=_hours(2), effort='high', rule_id='other-rule'),
        )),
    ])

    with pytest.raises(ThroughputUnavailable) as excinfo:
        resolve_since(_options(root))

    assert excinfo.value.kind is Unavailable.NO_TREATED_DISPATCH
    assert PREFIX in str(excinfo.value)


def test_a_root_without_runs_db_is_an_unreadable_store_naming_the_root(tmp_path):
    root = tmp_path / 'empty'
    root.mkdir()

    with pytest.raises(ThroughputUnavailable) as excinfo:
        throughput_report(_options(root), SINCE)

    assert excinfo.value.kind is Unavailable.STORE_UNREADABLE
    assert str(root) in str(excinfo.value)


@pytest.mark.parametrize(('overrides', 'named'), [
    ({'treated_bucket': 2}, '2'),
    ({'treated_bucket': -1}, '-1'),
    ({'interval_level': 0.0}, '0.0'),
    ({'interval_level': 1.0}, '1.0'),
    ({'interval_level': 1.5}, '1.5'),
    ({'resamples': 0}, '0'),
    ({'roles': ()}, '()'),
])
def test_options_refuse_an_out_of_range_value_naming_it(tmp_path, overrides, named):
    with pytest.raises(ValueError, match=named.replace('(', r'\(').replace(')', r'\)')):
        _options(tmp_path, **overrides)


def test_options_refuse_fewer_than_two_buckets(tmp_path):
    with pytest.raises(ValueError, match='buckets'):
        _options(tmp_path, buckets=1, treated_bucket=0)


PLANTED_CUT = 0.6
PLANTED_LOSS = 1 / 0.6
IDENTICAL_ARMS = 1.0


@pytest.fixture
def planted_root(tmp_path, make_seeding_runs_db):
    """Factory: a root holding planted_cohort(n_per_arm=40) at *factor*."""
    def _build(factor: float) -> Path:
        root = tmp_path / f'df-{factor:.4f}'
        build_cohort_root(
            root, make_seeding_runs_db,
            planted_cohort(SALT, SINCE, n_per_arm=40, treated_cost_factor=factor),
        )
        return root

    return _build


def _planted_report(root: Path, **overrides):
    return throughput_report(_options(root, resamples=2000, seed=20261117, **overrides), SINCE)


def test_a_planted_cost_cut_is_an_interval_wholly_above_1(planted_root):
    report = _planted_report(planted_root(PLANTED_CUT))

    assert report.ratio == pytest.approx(1 / PLANTED_CUT)
    assert report.interval is not None
    assert report.interval[0] > 1
    assert report.interval_side is IntervalSide.ABOVE_1


def test_a_planted_cost_rise_is_an_interval_wholly_below_1(planted_root):
    report = _planted_report(planted_root(PLANTED_LOSS))

    assert report.interval is not None
    assert report.interval[1] < 1
    assert report.interval_side is IntervalSide.BELOW_1


def test_identical_arms_give_a_ratio_of_exactly_1_inside_an_interval_including_1(planted_root):
    report = _planted_report(planted_root(IDENTICAL_ARMS))

    assert report.ratio == 1.0
    assert report.interval is not None
    low, high = report.interval
    assert low < 1 < high
    assert report.interval_side is IntervalSide.INCLUDES_1


def test_a_wider_level_gives_a_strictly_wider_interval_containing_the_narrower(planted_root):
    root = planted_root(IDENTICAL_ARMS)

    narrow = _planted_report(root, interval_level=0.80).interval
    wide = _planted_report(root, interval_level=0.97).interval

    assert narrow is not None and wide is not None
    assert wide[0] < narrow[0] < narrow[1] < wide[1]


def test_the_a_a_null_is_two_ordered_magnitudes_of_the_ratio(planted_root):
    report = _planted_report(planted_root(PLANTED_CUT))

    assert report.null is not None
    p95, p99 = report.null
    assert 1.0 <= p95 <= p99


def test_the_same_options_give_an_identical_report(planted_root):
    root = planted_root(PLANTED_CUT)

    assert _planted_report(root) == _planted_report(root)


def test_an_undefined_ratio_has_no_interval_and_an_undefined_side(tmp_path, make_seeding_runs_db):
    root = tmp_path / 'df'
    (treated,), (control,) = arm_task_ids(SALT, 2, 1, 1)
    build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=treated, dispatches=(_treated(_hours(1)),),
                   invocations=(('implementer', 1.0, _hours(1)),), landed_at=_hours(5)),
        CohortTask(task_id=control, dispatches=(_control(_hours(1)),),
                   invocations=(('implementer', 1.0, _hours(1)),)),
    ])

    report = throughput_report(_options(root), SINCE)

    assert report.ratio is None
    assert report.interval is None
    assert report.interval_side is IntervalSide.UNDEFINED


def test_verify_attempts_and_review_cycles_are_counted_per_landing(tmp_path, make_seeding_runs_db):
    root = tmp_path / 'df'
    (treated,), (control,) = arm_task_ids(SALT, 2, 1, 1)
    after_read = READ_AT + timedelta(hours=1)
    build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(
            task_id=treated, dispatches=(_treated(_hours(1)),), landed_at=_hours(48),
            results=(('done', _hours(40), 2, 1), ('done', _hours(47), 1, 2),
                     ('done', after_read, 10, 10)),
            merge_attempts=((_hours(45), {'outcome': 'verify_failed'}),
                            (_hours(46), {'outcome': 'merged'}),
                            (after_read, {'outcome': 'verify_failed'})),
        ),
        CohortTask(task_id=control, dispatches=(_control(_hours(1)),), landed_at=_hours(48)),
    ])

    report = throughput_report(_options(root), SINCE)

    treated_secondary = report.secondary[Arm.TREATED]
    assert treated_secondary.n == 1
    assert treated_secondary.verify_attempts_per_landing == 4
    assert treated_secondary.review_cycles_per_landing == 3
    assert report.secondary[Arm.CONTROL].verify_attempts_per_landing == 0


def test_a_per_landing_figure_is_none_when_nothing_landed(tmp_path, make_seeding_runs_db):
    root = tmp_path / 'df'
    (treated,), _ = arm_task_ids(SALT, 2, 1, 1)
    build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=treated, dispatches=(_treated(_hours(1)),),
                   results=(('blocked', _hours(40), 1, 1),)),
    ])

    treated_secondary = throughput_report(_options(root), SINCE).secondary[Arm.TREATED]

    assert treated_secondary.verify_attempts_per_landing is None
    assert treated_secondary.review_cycles_per_landing is None
    assert treated_secondary.failures_after_merge_per_landing is None


def test_the_blocked_or_escalated_share_counts_any_outcome_before_read_at(
    tmp_path, make_seeding_runs_db,
):
    root = tmp_path / 'df'
    (t0, t1), (c0, _) = arm_task_ids(SALT, 2, 1, 2)
    build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=t0, dispatches=(_treated(_hours(1)),),
                   results=(('blocked', _hours(10), 0, 0), ('done', _hours(20), 0, 0))),
        CohortTask(task_id=t1, dispatches=(_treated(_hours(2)),),
                   results=(('escalated', READ_AT + timedelta(hours=1), 0, 0),)),
        CohortTask(task_id=c0, dispatches=(_control(_hours(1)),),
                   results=(('done', _hours(10), 0, 0),)),
    ])

    report = throughput_report(_options(root), SINCE)

    assert report.secondary[Arm.TREATED].blocked_or_escalated_share == 0.5
    assert report.secondary[Arm.CONTROL].blocked_or_escalated_share == 0.0
    assert _by_task(report)[str(t0)].blocked_or_escalated is True


def test_cost_per_task_by_role_divides_each_roles_arm_cost_by_arm_n(
    tmp_path, make_seeding_runs_db,
):
    root = tmp_path / 'df'
    (t0, t1), _ = arm_task_ids(SALT, 2, 1, 2)
    build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=t0, dispatches=(_treated(_hours(1)),), invocations=(
            ('implementer', 2.0, _hours(1)), ('reviewer', 1.0, _hours(2)),
        )),
        CohortTask(task_id=t1, dispatches=(_treated(_hours(2)),), invocations=(
            ('implementer', 4.0, _hours(2)),
        )),
    ])

    treated_secondary = throughput_report(_options(root), SINCE).secondary[Arm.TREATED]

    assert dict(treated_secondary.cost_per_task_by_role) == {'implementer': 3.0, 'reviewer': 0.5}


LANDED = _hours(48)
UNRELATED_SHA = 'f' * 40


def _skew(conn, at: datetime, implicated: list[str]) -> None:
    conn.seed_event(at.isoformat(), 'merge_attempt', task_id='777', data={
        'outcome': 'verify_failed', 'disposition': 'integration_skew',
        'implicated_commits': implicated,
    })


def _skew_two_days_after(root, conn, merge_sha, branch_commit):
    _skew(conn, LANDED + timedelta(days=2), [branch_commit])


def _skew_eight_days_after(root, conn, merge_sha, branch_commit):
    _skew(conn, LANDED + timedelta(days=8), [branch_commit])


def _skew_implicating_an_unrelated_sha(root, conn, merge_sha, branch_commit):
    _skew(conn, LANDED + timedelta(days=2), [UNRELATED_SHA])


def _revert_three_days_after(root, conn, merge_sha, branch_commit):
    revert_merge(root, merge_sha, LANDED + timedelta(days=3))


def _revert_of_another_merge(root, conn, merge_sha, branch_commit):
    other_merge, _ = land_branch(root, 778, LANDED + timedelta(days=1))
    revert_merge(root, other_merge, LANDED + timedelta(days=3))


@pytest.mark.parametrize(('plant', 'failures'), [
    (_skew_two_days_after, 1),
    (_skew_eight_days_after, 0),
    (_skew_implicating_an_unrelated_sha, 0),
    (_revert_three_days_after, 1),
    (_revert_of_another_merge, 0),
])
def test_failures_after_merge_count_skews_and_reverts_within_seven_days_of_landing(
    tmp_path, make_seeding_runs_db, plant, failures,
):
    root = tmp_path / 'df'
    conn = build_cohort_root(root, make_seeding_runs_db, [])
    (treated,), (control,) = arm_task_ids(SALT, 2, 1, 1)
    merge_sha, branch_commit = land_branch(root, treated, LANDED)
    seed_tasks(conn, [
        CohortTask(task_id=treated, dispatches=(_treated(_hours(1)),),
                   landed_at=LANDED, merge_sha=merge_sha),
        CohortTask(task_id=control, dispatches=(_control(_hours(1)),)),
    ])
    plant(root, conn, merge_sha, branch_commit)

    report = throughput_report(_options(root), SINCE)

    assert _by_task(report)[str(treated)].failures_after_merge == failures
    assert report.secondary[Arm.TREATED].failures_after_merge_per_landing == failures


def test_an_unreadable_main_history_is_refused_naming_the_root(tmp_path, make_seeding_runs_db):
    root = tmp_path / 'not-a-repo'
    conn = build_cohort_root(root, make_seeding_runs_db, [], git=False)
    (treated,), _ = arm_task_ids(SALT, 2, 1, 1)
    seed_tasks(conn, [CohortTask(task_id=treated, dispatches=(_treated(_hours(1)),),
                                 landed_at=LANDED)])
    _skew(conn, LANDED + timedelta(days=2), [UNRELATED_SHA])

    with pytest.raises(ThroughputUnavailable) as excinfo:
        throughput_report(_options(root), SINCE)

    assert excinfo.value.kind is Unavailable.HISTORY_UNREADABLE
    assert str(root) in str(excinfo.value)
