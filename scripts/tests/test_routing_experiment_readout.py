"""Tests for scripts/routing_experiment_readout.py over synthetic fleet roots built in tmp_path.

They never read the live ledgers. To run just this suite:

    uv run --project shared pytest scripts/tests/test_routing_experiment_readout.py -q
"""
from __future__ import annotations

import hashlib
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
    assert 'read good, no-tasks' in rule
    assert 'no-runs (runs.db absent)' in rule
    assert 'garbage (runs.db unreadable: file is not a database)' in rule
    assert 'no-tasks (tasks.db absent)' in rule
    for root in unreadable_roots:
        assert str(root) in err


def test_main_over_only_unreadable_roots_exits_nothing_read(unreadable_roots, capsys):
    assert _run_main(unreadable_roots) == readout.EXIT_NOTHING_READ
    assert 'no runs.db was readable' in capsys.readouterr().err
