"""Tests for scripts/routing_experiment_readout.py over synthetic fleet roots built in tmp_path.

They never read the live ledgers. To run just this suite:

    uv run --project shared pytest scripts/tests/test_routing_experiment_readout.py -q
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import routing_experiment_readout as readout

UNTIL = '2026-11-20T00:00:00+00:00'
IN_WINDOW = '2026-11-10T12:00:00+00:00'
BEFORE_WINDOW = '2026-10-01T00:00:00+00:00'


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
