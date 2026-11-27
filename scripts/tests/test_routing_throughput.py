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
    build_cohort_root(root, make_seeding_runs_db, _known_ratio_tasks(), git=False)
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
    build_cohort_root(root, make_seeding_runs_db, tasks, git=False)

    report = throughput_report(_options(root), SINCE)

    assert {'9001', '9002', '9003'}.isdisjoint(_by_task(report))
    assert report.ratio == 2.0


def test_a_debugger_only_first_dispatch_inside_the_window_is_in_the_cohort(
    tmp_path, make_seeding_runs_db,
):
    root = tmp_path / 'df'
    build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=9004, dispatches=(_control(_hours(4), role='debugger'),)),
    ], git=False)

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
    ], git=False)

    record = _by_task(throughput_report(_options(root), SINCE))['9005']

    assert record.cost_usd == 2.0


def test_a_landing_at_or_after_read_at_does_not_count_as_landed(tmp_path, make_seeding_runs_db):
    root = tmp_path / 'df'
    build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=9006, dispatches=(_control(_hours(4)),), landed_at=READ_AT),
    ], git=False)

    record = _by_task(throughput_report(_options(root), SINCE))['9006']

    assert record.landed_at is None
    assert record.merge_sha is None


def test_a_train_member_superseded_in_merge_finalized_counts_as_landed(
    tmp_path, make_seeding_runs_db,
):
    root = tmp_path / 'df'
    conn = build_cohort_root(root, make_seeding_runs_db, [
        CohortTask(task_id=9007, dispatches=(_control(_hours(4)),)),
    ], git=False)
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
    ], git=False)
    build_cohort_root(second, make_seeding_runs_db, [
        CohortTask(task_id=2, dispatches=(
            Dispatch(at=SINCE, role='architect', effort='high', rule_id=f'{PREFIX}-architect'),
        )),
    ], git=False)

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
    ], git=False)

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
