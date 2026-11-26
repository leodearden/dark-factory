"""Tests for scripts/routing_experiment_readout.py over synthetic fleet roots built in tmp_path.

They never read the live ledgers. To run just this suite:

    uv run --project shared pytest scripts/tests/test_routing_experiment_readout.py -q
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
import routing_experiment_readout as readout
from shared.routing_split import split_bucket

UNTIL = '2026-11-20T00:00:00+00:00'
IN_WINDOW = '2026-11-10T12:00:00+00:00'
BEFORE_WINDOW = '2026-10-01T00:00:00+00:00'
SALT = 'routing-exp-test'
WINDOW = readout.Window.trailing(datetime.fromisoformat(UNTIL), 30)


def _at(day: int, hour: int = 0) -> str:
    return f'2026-11-{day:02d}T{hour:02d}:00:00+00:00'


@dataclass(frozen=True)
class ResultRow:
    outcome: str
    completed_at: str = IN_WINDOW
    review_cycles: int = 0
    execute_iterations: int = 0


@dataclass(frozen=True)
class Invocation:
    role: str
    model: str
    cost_usd: float
    started_at: str = IN_WINDOW


@dataclass(frozen=True)
class Event:
    event_type: str
    data: dict | str | None = None
    timestamp: str = IN_WINDOW
    role: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    """One synthetic task across both stores; ``in_tasks_db=False`` leaves it out of tasks.db."""

    task_id: int
    results: tuple[ResultRow, ...] = (ResultRow('done'),)
    invocations: tuple[Invocation, ...] = ()
    events: tuple[Event, ...] = ()
    metadata: dict | None = None
    candidate_key: str | None = None
    in_tasks_db: bool = True


@pytest.fixture
def fleet_root(tmp_path, make_seeding_runs_db, make_tasks_db):
    """Factory: build one synthetic project root named *name* holding *tasks*, return its Path."""

    def _build(name: str, tasks, *, with_tasks_db: bool = True) -> Path:
        root = tmp_path / name
        orchestrator_dir = root / 'data' / 'orchestrator'
        orchestrator_dir.mkdir(parents=True)
        conn = make_seeding_runs_db(orchestrator_dir)
        conn.execute('PRAGMA synchronous = OFF')  # every seed_* commits; skip the fsyncs
        for spec in tasks:
            for run_number, row in enumerate(spec.results):
                conn.seed_task_result(
                    task_id=spec.task_id,
                    outcome=row.outcome,
                    completed_at=row.completed_at,
                    run_id=f'run-{run_number}',
                    review_cycles=row.review_cycles,
                    execute_iterations=row.execute_iterations,
                )
            for invocation in spec.invocations:
                conn.seed_invocation(
                    task_id=str(spec.task_id),
                    role=invocation.role,
                    model=invocation.model,
                    cost_usd=invocation.cost_usd,
                    started_at=invocation.started_at,
                    completed_at=invocation.started_at,
                )
            for event in spec.events:
                conn.seed_event(
                    event.timestamp,
                    event.event_type,
                    task_id=str(spec.task_id),
                    role=event.role,
                    data=event.data,
                )
        if with_tasks_db:
            tasks_dir = root / '.taskmaster' / 'tasks'
            tasks_dir.mkdir(parents=True)
            make_tasks_db(
                [
                    {
                        'id': spec.task_id,
                        'metadata': spec.metadata,
                        'candidate_key': spec.candidate_key,
                    }
                    for spec in tasks
                    if spec.in_tasks_db
                ],
                directory=tasks_dir,
            )
        return root

    return _build


def _unit_must_not_be_read():
    pytest.fail('the fused-memory unit was read although an earlier source named roots')


def _no_unit():
    return None


# --- root resolution -------------------------------------------------------


def test_cli_roots_win_over_an_environment_that_also_names_roots():
    selection = readout.resolve_roots(
        '/a, /b,',
        {'DASHBOARD_KNOWN_PROJECT_ROOTS': '/c'},
        _unit_must_not_be_read,
    )

    assert selection.roots == (Path('/a'), Path('/b'))
    assert selection.source is readout.RootSource.CLI


def test_environment_roots_win_over_the_unit():
    selection = readout.resolve_roots(
        None, {'DASHBOARD_KNOWN_PROJECT_ROOTS': '/c,/d'}, _unit_must_not_be_read
    )

    assert selection.roots == (Path('/c'), Path('/d'))
    assert selection.source is readout.RootSource.ENVIRONMENT


def test_roots_fall_back_to_the_installed_fused_memory_unit_environment():
    selection = readout.resolve_roots(
        None, {}, lambda: 'PATH=/bin DASHBOARD_KNOWN_PROJECT_ROOTS=/a,/b\n'
    )

    assert selection.roots == (Path('/a'), Path('/b'))
    assert selection.source is readout.RootSource.UNIT


@pytest.mark.parametrize('unit_text', [None, '', 'PATH=/bin', 'DASHBOARD_KNOWN_PROJECT_ROOTS='])
def test_no_source_naming_a_root_resolves_no_roots(unit_text):
    selection = readout.resolve_roots(None, {}, lambda: unit_text)

    assert selection.roots == ()


# --- CLI -------------------------------------------------------------------


def test_main_with_no_resolvable_root_exits_nothing_read_naming_both_remedies(capsys):
    status = readout.main(
        ['--salt', 's', '--days', '30', '--until', UNTIL],
        environ={},
        unit_environment=_no_unit,
    )

    assert status == readout.EXIT_NOTHING_READ == 3
    err = capsys.readouterr().err
    assert 'DASHBOARD_KNOWN_PROJECT_ROOTS' in err
    assert '--roots' in err


@pytest.mark.parametrize(
    'bad_arguments',
    [
        ['--buckets', '1'],
        ['--days', '0'],
        ['--until', '2026-11-20T00:00:00'],
        ['--until', 'yesterday'],
    ],
    ids=['buckets-below-two', 'zero-days', 'until-without-offset', 'until-unparseable'],
)
def test_invalid_arguments_are_usage_errors(bad_arguments):
    with pytest.raises(SystemExit) as raised:
        readout.main(['--salt', 's', *bad_arguments], environ={}, unit_environment=_no_unit)

    assert raised.value.code == 2


# --- ledger load -----------------------------------------------------------


def _routing_decision(timestamp, rule_id=None, role='implementer', **plan_shape):
    return Event(
        'routing_decision',
        {'role': role, 'model': 'sonnet', 'rule_id': rule_id, **plan_shape},
        timestamp=timestamp,
        role=role,
    )


_RICH_TASK = TaskSpec(
    1,
    results=(
        ResultRow('done', _at(2), review_cycles=0, execute_iterations=1),
        ResultRow('blocked', _at(3), review_cycles=1, execute_iterations=2),
        ResultRow('done', _at(4), review_cycles=0, execute_iterations=3),
    ),
    invocations=(
        Invocation('implementer', 'sonnet', 2.0, _at(2)),
        Invocation('implementer', 'opus', 3.0, _at(3)),
        Invocation('architect', 'opus', 1.0, _at(2)),
        Invocation('implementer', 'haiku', 5.0, BEFORE_WINDOW),
    ),
    events=(
        _routing_decision(_at(2, 1)),
        _routing_decision(_at(2, 2), 'large-plan-modules', plan_steps=5, plan_modules=3),
        _routing_decision(_at(3, 1), 'large-plan-modules', plan_steps=7, plan_modules=3),
        Event('workflow_verify', {'passed': True, 'duration_ms': 60_000}, _at(2, 3)),
        Event('workflow_verify', {'passed': True, 'duration_ms': 120_000}, _at(3, 3)),
        Event('merge_verify', {'duration_ms': 30_000}, _at(4, 1)),
        Event('merge_verify', {'duration_ms': 90_000}, _at(4, 2)),
        Event('review_completed', {'blocking_count': 1}, _at(3, 4)),
        Event('review_completed', {'blocking_count': 2}, _at(3, 5)),
    ),
    metadata={'complexity': 'simple', 'x_curator_complexity_hint': 'simple'},
)
_PRE_PRODUCER_TASK = TaskSpec(
    2,
    events=(
        _routing_decision(_at(5), 'retry-tier-up'),
        Event('workflow_verify', {'passed': True}, _at(5, 1)),
        Event('merge_verify', {'passed': True}, _at(5, 2)),
        Event('review_completed', {'cycle': 1}, _at(5, 3)),
    ),
)
_QUIET_TASK = TaskSpec(3)
_COMPLETED_BEFORE_WINDOW = TaskSpec(4, results=(ResultRow('done', BEFORE_WINDOW),))
_COMPLETED_AT_UNTIL = TaskSpec(5, results=(ResultRow('done', UNTIL),))
_LEDGER_TASKS = (
    _RICH_TASK, _PRE_PRODUCER_TASK, _QUIET_TASK, _COMPLETED_BEFORE_WINDOW, _COMPLETED_AT_UNTIL,
)


@pytest.fixture
def ledger_root(fleet_root):
    return fleet_root('proj-a', _LEDGER_TASKS)


def _load(roots, *, buckets=2, key=readout.KeyKind.TASK_ID):
    return readout.load_fleet(tuple(roots), WINDOW, SALT, buckets, key)


def _by_task(load):
    return {record.task_id: record for record in load.records}


def test_outcomes_take_the_final_row_and_the_maxima_over_every_in_window_row(ledger_root):
    record = _by_task(_load([ledger_root]))['1']

    assert record.project == 'proj-a'
    assert record.blocked_or_escalated is True
    assert record.final_outcome == 'done'
    assert record.max_review_cycles == 1
    assert record.max_execute_iterations == 3


def test_cost_sums_every_in_window_invocation_and_models_come_from_the_implementer(
    ledger_root,
):
    record = _by_task(_load([ledger_root]))['1']

    assert record.cost_usd == 6.0
    assert record.implementer_models == frozenset({'sonnet', 'opus'})


def test_verify_minutes_sum_present_durations_and_are_none_when_a_duration_is_missing(
    ledger_root,
):
    records = _by_task(_load([ledger_root]))

    assert (records['1'].workflow_verify_minutes, records['1'].merge_verify_minutes) == (3.0, 2.0)
    assert (records['2'].workflow_verify_minutes, records['2'].merge_verify_minutes) == (None, None)
    assert (records['3'].workflow_verify_minutes, records['3'].merge_verify_minutes) == (0.0, 0.0)


def test_reviewer_blocking_count_sums_and_is_none_when_a_review_lacks_the_field(ledger_root):
    records = _by_task(_load([ledger_root]))

    assert records['1'].reviewer_blocking_count == 3.0
    assert records['2'].reviewer_blocking_count is None
    assert records['3'].reviewer_blocking_count == 0.0


def test_plan_shape_comes_from_the_earliest_implementer_decision_carrying_it(ledger_root):
    records = _by_task(_load([ledger_root]))

    assert (records['1'].plan_steps, records['1'].plan_modules) == (5, 3)
    assert records['1'].large_plan_rule_fired is True
    assert (records['2'].plan_steps, records['2'].plan_modules) == (None, None)
    assert records['2'].large_plan_rule_fired is False


def test_task_metadata_surfaces_complexity_and_the_curator_hint(ledger_root):
    records = _by_task(_load([ledger_root]))

    assert (records['1'].complexity, records['1'].curator_complexity_hint) == ('simple', 'simple')
    assert (records['2'].complexity, records['2'].curator_complexity_hint) == (None, None)


def test_every_record_is_bucketed_by_the_shared_split_bucket(ledger_root):
    load = _load([ledger_root], buckets=3)

    assert load.records
    for record in load.records:
        assert record.key == record.task_id
        assert record.bucket == split_bucket(SALT, record.task_id, 3)


def test_population_is_the_tasks_with_a_row_completed_inside_the_half_open_window(
    ledger_root,
):
    load = _load([ledger_root])

    assert sorted(_by_task(load)) == ['1', '2', '3']
    assert load.roots[0].runs_db_problem is None


def test_producer_evidence_counts_rule_ids_plan_shapes_and_hinted_tasks(ledger_root):
    evidence = _load([ledger_root]).evidence

    assert dict(evidence.rule_id_counts) == {'large-plan-modules': 2, 'retry-tier-up': 1}
    assert evidence.plan_shape_decisions == 2
    assert evidence.hinted_tasks == 1


def test_loading_leaves_both_stores_byte_identical_and_creates_no_sibling_files(ledger_root):
    stores = [
        ledger_root / 'data' / 'orchestrator' / 'runs.db',
        ledger_root / '.taskmaster' / 'tasks' / 'tasks.db',
    ]
    before = {store: hashlib.sha256(store.read_bytes()).hexdigest() for store in stores}
    listing_before = {store.parent: sorted(store.parent.iterdir()) for store in stores}

    _load([ledger_root])

    assert {store: hashlib.sha256(store.read_bytes()).hexdigest() for store in stores} == before
    assert {store.parent: sorted(store.parent.iterdir()) for store in stores} == listing_before


# --- keys, several roots, skipped roots ------------------------------------


def _run_main(roots, *extra):
    return readout.main(
        ['--roots', ','.join(str(root) for root in roots), '--salt', SALT, '--until', UNTIL,
         *extra],
        environ={},
        unit_environment=_unit_must_not_be_read,
    )


def _rule_line(text):
    (line,) = [line for line in text.splitlines() if line.startswith('rule:')]
    return line


def test_candidate_key_buckets_by_the_stored_key_and_counts_tasks_without_one(fleet_root):
    root = fleet_root(
        'proj-a',
        [
            TaskSpec(1, candidate_key='ck-1'),
            TaskSpec(2, candidate_key=None),
            TaskSpec(3, in_tasks_db=False),
        ],
    )

    load = _load([root], key=readout.KeyKind.CANDIDATE_KEY)

    (record,) = load.records
    assert (record.task_id, record.key) == ('1', 'ck-1')
    assert record.bucket == split_bucket(SALT, 'ck-1', 2)
    assert load.exclusions == {'no candidate_key': 2}


def test_records_carry_their_project_so_one_task_id_in_two_projects_is_two_records(
    fleet_root,
):
    roots = [
        fleet_root('proj-a', [TaskSpec(1)]),
        fleet_root('proj-b', [TaskSpec(1), TaskSpec(2)]),
    ]

    load = _load(roots)

    assert sorted((record.project, record.task_id) for record in load.records) == [
        ('proj-a', '1'), ('proj-b', '1'), ('proj-b', '2'),
    ]


@pytest.fixture
def unreadable_roots(tmp_path):
    absent = tmp_path / 'no-runs'
    absent.mkdir()
    garbage = tmp_path / 'garbage'
    (garbage / 'data' / 'orchestrator').mkdir(parents=True)
    (garbage / 'data' / 'orchestrator' / 'runs.db').write_bytes(b'not a sqlite database\n' * 64)
    return [absent, garbage]


def test_main_names_each_skipped_root_and_its_reason_on_the_rule_line(
    fleet_root, unreadable_roots, capsys,
):
    roots = [
        *unreadable_roots,
        fleet_root('no-tasks', [TaskSpec(7)], with_tasks_db=False),
        fleet_root('good', [TaskSpec(1)]),
    ]

    status = _run_main(roots)

    assert status == readout.EXIT_OK
    out, err = capsys.readouterr()
    rule = _rule_line(out)
    assert 'read no-tasks, good' in rule
    assert 'no-runs (runs.db absent)' in rule
    assert 'garbage (runs.db unreadable: file is not a database)' in rule
    assert 'no-tasks (tasks.db absent)' in rule
    for root in unreadable_roots:
        assert str(root) in err


def test_main_over_only_unreadable_roots_exits_nothing_read(unreadable_roots, capsys):
    assert _run_main(unreadable_roots) == readout.EXIT_NOTHING_READ
    assert 'no runs.db was readable' in capsys.readouterr().err


# --- per-arm comparison and its A/A null ----------------------------------

_BASE_RECORD = readout.TaskRecord(
    project='proj-a',
    task_id='0',
    key='0',
    bucket=0,
    implementer_models=frozenset(),
    plan_steps=None,
    plan_modules=None,
    large_plan_rule_fired=False,
    complexity=None,
    curator_complexity_hint=None,
    final_outcome='done',
    blocked_or_escalated=False,
    max_review_cycles=0,
    max_execute_iterations=0,
    cost_usd=0.0,
    workflow_verify_minutes=0.0,
    merge_verify_minutes=0.0,
    reviewer_blocking_count=0.0,
)


def _arm_records(arm_values):
    """Records from {bucket: [overrides, ...]}, task ids numbered across arms."""
    records = []
    for bucket, overrides in arm_values.items():
        for override in overrides:
            task_id = str(len(records) + 1)
            records.append(
                dataclasses.replace(_BASE_RECORD, task_id=task_id, key=task_id, bucket=bucket,
                                    **override)
            )
    return records


def _compare(records, *, buckets=2, resamples=400, seed=readout.DEFAULT_SEED):
    return readout.compare_arms(
        records, readout.METRICS, buckets, resamples, random.Random(seed)
    )


def _row(comparison, name):
    (row,) = [row for row in comparison.rows if row.metric == name]
    return row


def test_a_hand_checkable_fixture_gives_arm_means_and_their_absolute_difference():
    records = _arm_records({
        0: [{'cost_usd': cost, 'blocked_or_escalated': cost <= 2} for cost in (1.0, 2.0, 3.0, 4.0)],
        1: [{'cost_usd': cost} for cost in (2.0, 3.0, 4.0, 5.0)],
    })

    (comparison,) = _compare(records)

    assert comparison.arm == 1
    cost = _row(comparison, 'cost_usd')
    assert (cost.base_n, cost.arm_n) == (4, 4)
    assert (cost.base_mean, cost.arm_mean, cost.diff) == (2.5, 3.5, 1.0)
    blocked = _row(comparison, 'blocked_or_escalated')
    assert (blocked.base_mean, blocked.arm_mean, blocked.diff) == (0.5, 0.0, -0.5)


def test_one_comparison_per_treated_arm_against_arm_zero():
    records = _arm_records({0: [{}], 1: [{}], 2: [{}]})

    assert [comparison.arm for comparison in _compare(records, buckets=3)] == [1, 2]


def test_a_column_with_no_spread_has_a_zero_null_band():
    records = _arm_records({0: [{'cost_usd': 7.0}] * 25, 1: [{'cost_usd': 7.0}] * 25})

    (comparison,) = _compare(records)

    assert _row(comparison, 'cost_usd').null == readout.NullBand(p95=0.0, p99=0.0, mde=0.0)


def _balanced_overrides(n_per_arm):
    return [
        {
            'cost_usd': float(i),
            'workflow_verify_minutes': float(i % 7),
            'blocked_or_escalated': i % 4 == 0,
            'max_review_cycles': i % 2,
        }
        for i in range(n_per_arm)
    ]


def test_a_balanced_by_construction_split_has_zero_difference_inside_a_nonzero_null():
    overrides = _balanced_overrides(25)
    records = _arm_records({0: overrides, 1: overrides})

    (comparison,) = _compare(records)

    cost = _row(comparison, 'cost_usd')
    assert cost.diff == 0.0
    assert cost.null is not None
    assert cost.null.p99 > 0.0
    assert cost.null.p95 <= cost.null.p99
    assert cost.null.mde > 0.0


def test_means_count_only_eligible_tasks_and_a_thin_metric_gets_no_null():
    overrides = _balanced_overrides(25)
    thin_arm = [
        {**override, 'workflow_verify_minutes': None if i >= 10 else 2.0}
        for i, override in enumerate(overrides)
    ]
    records = _arm_records({0: overrides, 1: thin_arm})

    (comparison,) = _compare(records)

    verify = _row(comparison, 'workflow_verify_minutes')
    assert (verify.base_n, verify.arm_n, verify.arm_mean) == (25, 10, 2.0)
    assert readout.MIN_ELIGIBLE_PER_ARM == 20
    assert verify.null is None
    assert _row(comparison, 'cost_usd').null is not None


def test_the_same_records_and_seed_give_identical_comparisons():
    overrides = _balanced_overrides(25)
    records = _arm_records({0: overrides, 1: list(reversed(overrides))})

    assert _compare(records, seed=7) == _compare(records, seed=7)


def _arm_task_ids(n_per_arm=25, *, salt=SALT):
    """The first *n_per_arm* task ids the real split_bucket puts in each of two arms."""
    arms = {0: [], 1: []}
    task_id = 0
    while min(len(ids) for ids in arms.values()) < n_per_arm:
        task_id += 1
        bucket = split_bucket(salt, str(task_id), 2)
        if len(arms[bucket]) < n_per_arm:
            arms[bucket].append(task_id)
    return arms


def _fleet_task(task_id, i, *, outcome=None, cost=None, verify_duration=True):
    """The i-th task of an arm; arms built from the same i carry identical values."""
    verify_data = {'duration_ms': 60_000 * (i % 7)} if verify_duration else {'passed': True}
    return TaskSpec(
        task_id,
        results=(
            ResultRow(outcome or ('blocked' if i % 4 == 0 else 'done'), review_cycles=i % 2,
                      execute_iterations=i % 3),
        ),
        invocations=(Invocation('implementer', 'sonnet', float(i) if cost is None else cost),),
        events=(
            Event('workflow_verify', verify_data),
            Event('merge_verify', {'duration_ms': 30_000 * (i % 5)}),
            Event('review_completed', {'blocking_count': i % 2}),
        ),
    )


def _balanced_fleet_specs(n_per_arm=25):
    return [
        _fleet_task(task_id, i)
        for ids in _arm_task_ids(n_per_arm).values()
        for i, task_id in enumerate(ids)
    ]


@pytest.fixture
def balanced_root(fleet_root):
    return fleet_root('proj-a', _balanced_fleet_specs())


def test_main_prints_the_rule_producers_arm_sizes_and_one_row_per_metric(
    balanced_root, capsys,
):
    assert _run_main([balanced_root], '--resamples', '400') == readout.EXIT_OK

    out = capsys.readouterr().out
    rule = _rule_line(out)
    for fragment in (
        '[2026-10-21T00:00:00+00:00, 2026-11-20T00:00:00+00:00)',
        'population = the 50 tasks',
        'read proj-a',
        'key task_id',
        f"salt '{SALT}' buckets 2",
        'subset all',
        '400 permutations',
        f'seed {readout.DEFAULT_SEED}',
    ):
        assert fragment in rule
    lines = out.splitlines()
    assert any(line.startswith('producers:') for line in lines)
    assert 'arm 0: 25 tasks; arm 1: 25 tasks' in out
    (header,) = [line for line in lines if 'mean0' in line]
    assert header.split()[1:9] == ['n0', 'n1', 'mean0', 'mean1', 'diff', 'null_p95', 'null_p99',
                                  'mde']
    for metric in readout.METRICS:
        (row,) = [line for line in lines if line.split()[:2] in (['*', metric.name],
                                                                  ['-', metric.name])]
        assert row.split()[0] == ('*' if metric.headline else '-')


def test_two_runs_with_one_seed_print_byte_identical_output(balanced_root, capsys):
    _run_main([balanced_root], '--resamples', '400')
    first = capsys.readouterr().out
    _run_main([balanced_root], '--resamples', '400')

    assert capsys.readouterr().out == first


def test_json_carries_the_per_task_records_and_the_same_comparisons(balanced_root, capsys):
    assert _run_main([balanced_root], '--resamples', '400', '--json') == readout.EXIT_OK

    report = json.loads(capsys.readouterr().out)
    load = _load([balanced_root])
    expected = _compare(load.records, resamples=400)
    assert sorted(task['task_id'] for task in report['tasks']) == sorted(
        record.task_id for record in load.records
    )
    assert report['comparisons'] == json.loads(
        json.dumps([dataclasses.asdict(comparison) for comparison in expected])
    )


# --- the A/A verdict --------------------------------------------------------

SCRIPT = Path(readout.__file__)


def _skewed_fleet_specs(n_per_arm=25):
    """Arm 1 all blocked at cost 10, arm 0 all done at cost 1; the other headline metrics balanced."""
    arms = _arm_task_ids(n_per_arm)
    return [
        *(_fleet_task(task_id, i, outcome='done', cost=1.0) for i, task_id in enumerate(arms[0])),
        *(_fleet_task(task_id, i, outcome='blocked', cost=10.0)
          for i, task_id in enumerate(arms[1])),
    ]


def _table_line(out, metric):
    (line,) = [line for line in out.splitlines() if line.split()[1:2] == [metric]]
    return line


def test_aa_on_a_balanced_split_passes(balanced_root, capsys):
    assert _run_main([balanced_root], '--aa', '--resamples', '400') == readout.EXIT_OK

    out = capsys.readouterr().out
    assert 'A/A check passed' in out
    assert 'SKEWED' not in out
    assert '--aa fails when a headline metric has |diff| > null p99' in _rule_line(out)


def test_aa_on_a_skewed_split_exits_one_from_a_real_child_process(fleet_root):
    root = fleet_root('proj-a', _skewed_fleet_specs())

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), '--aa', '--roots', str(root), '--salt', SALT,
         '--buckets', '2', '--days', '30', '--until', UNTIL],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == readout.EXIT_AA_SKEWED, completed.stderr
    skewed = [
        line for line in (completed.stdout + completed.stderr).splitlines() if 'SKEWED' in line
    ]
    blocked = [line for line in skewed if 'blocked_or_escalated' in line]
    cost = [line for line in skewed if 'cost_usd' in line]
    assert blocked and cost
    assert all('|diff| 1.000' in line and 'null p99' in line for line in blocked)
    assert all('|diff| 9.000' in line and 'null p99' in line for line in cost)
    assert not [line for line in skewed if 'review_cycles_ge1' in line]


def test_without_aa_a_skewed_split_exits_zero_and_marks_the_rows_over_p99(fleet_root, capsys):
    root = fleet_root('proj-a', _skewed_fleet_specs())

    assert _run_main([root], '--resamples', '400') == readout.EXIT_OK

    out = capsys.readouterr().out
    assert _table_line(out, 'blocked_or_escalated').endswith('>p99')
    assert _table_line(out, 'cost_usd').endswith('>p99')
    assert not _table_line(out, 'review_cycles_ge1').endswith('>p99')
    assert 'A/A check' not in out


def test_aa_with_a_headline_metric_below_the_eligibility_floor_exits_unproven(
    fleet_root, capsys,
):
    arms = _arm_task_ids()
    root = fleet_root(
        'proj-a',
        [
            *(_fleet_task(task_id, i) for i, task_id in enumerate(arms[0])),
            *(_fleet_task(task_id, i, verify_duration=i < 5) for i, task_id in enumerate(arms[1])),
        ],
    )

    assert _run_main([root], '--aa', '--resamples', '400') == readout.EXIT_UNPROVEN

    err = capsys.readouterr().err
    (line,) = [line for line in err.splitlines() if 'workflow_verify_minutes' in line]
    assert 'eligible n0=25 n1=5' in line
    assert f'below {readout.MIN_ELIGIBLE_PER_ARM}' in line
