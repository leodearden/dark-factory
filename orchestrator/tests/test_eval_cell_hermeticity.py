"""Boundary row 26: an eval cell is never built around a production writer.

S1 boundary row 26, Contract C11 of
plans/review-all-s1-one-escalation-filing-entry-point-prd.md. The eval cell
factory, ``evals/runner.py::_cell_workflow``, refuses every collaborator
through which a ``TaskWorkflow`` writes production state (the escalation
queue, and with it the filer the workflow derives from it; the steward
factory; the event store; the cost store) with an ``EvalHermeticityError``
that names the field and the offending object, before any workflow is built.
It refuses the built workflow too, if it holds a writer it was never handed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from escalation.queue import EscalationQueue
from shared.cost_store import CostStore
from shared.project_layout import escalation_queue_dir, runs_db

from orchestrator.config import OrchestratorConfig
from orchestrator.evals import runner
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.hermeticity import EvalHermeticityError, ProductionWriters
from orchestrator.evals.runner import _cell_workflow, build_eval_orch_config
from orchestrator.event_store import EventStore
from orchestrator.workflow import TaskWorkflow

# Each field holding its real production object, rooted at a project's own paths.
_PRODUCTION_WRITERS: dict[str, Callable[[Path], ProductionWriters]] = {
    'escalation_queue': lambda proj: ProductionWriters(
        escalation_queue=EscalationQueue(escalation_queue_dir(proj)),
    ),
    'steward_factory': lambda proj: ProductionWriters(
        steward_factory=lambda worktree, config_dir: None,
    ),
    'event_store': lambda proj: ProductionWriters(
        event_store=EventStore(runs_db(proj), run_id='r'),
    ),
    'cost_store': lambda proj: ProductionWriters(cost_store=CostStore(runs_db(proj))),
}


class _Cell(NamedTuple):
    project_root: Path
    task: dict
    orch_config: OrchestratorConfig
    worktree: Path


@pytest.fixture
def cell(tmp_path: Path, code_default_config: None) -> _Cell:
    proj = tmp_path / 'proj'
    task = {
        'id': 'df_task_9845',
        'project_root': str(proj),
        'pre_task_commit': 'deadbeef',
        'modules': ['lib'],
    }
    orch_config = build_eval_orch_config(
        EvalConfig('t', 'claude', 'sonnet', 'high'),
        task,
        OrchestratorConfig(project_root=proj),
    )
    return _Cell(proj, task, orch_config, tmp_path / 'wt')


def _build(cell: _Cell, **writers: ProductionWriters) -> TaskWorkflow:
    return _cell_workflow(
        cell.task, cell.orch_config, cell.worktree, None,
        replay_framed=False, usage_gate=None, **writers,
    )


@pytest.mark.parametrize('field', list(_PRODUCTION_WRITERS))
def test_each_production_writer_handed_to_the_cell_is_refused(cell: _Cell, field: str) -> None:
    writers = _PRODUCTION_WRITERS[field](cell.project_root)
    writer = getattr(writers, field)

    with pytest.raises(EvalHermeticityError) as refused:
        _build(cell, writers=writers)

    assert refused.value.field == field
    assert field in str(refused.value)
    assert repr(writer) in str(refused.value)


def test_two_production_writers_name_the_first_declared_field(cell: _Cell) -> None:
    writers = ProductionWriters(
        escalation_queue=EscalationQueue(escalation_queue_dir(cell.project_root)),
        cost_store=CostStore(runs_db(cell.project_root)),
    )

    with pytest.raises(EvalHermeticityError) as refused:
        _build(cell, writers=writers)

    assert refused.value.field == 'escalation_queue'


@pytest.mark.parametrize('writers', [{}, {'writers': ProductionWriters()}], ids=['default', 'explicit'])
def test_with_no_writers_the_cell_builds_as_today(
    cell: _Cell, writers: dict[str, ProductionWriters],
) -> None:
    workflow = _build(cell, **writers)

    assert isinstance(workflow, TaskWorkflow)
    assert workflow.escalation_queue is None
    assert workflow.event_store is None
    assert workflow.cost_store is None
    assert workflow.worktree == cell.worktree


@pytest.mark.parametrize('field', ['escalation_queue', 'event_store', 'cost_store'])
def test_a_built_workflow_holding_a_writer_it_was_not_handed_is_refused(
    cell: _Cell, field: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = getattr(_PRODUCTION_WRITERS[field](cell.project_root), field)
    real_build_workflow = runner.build_workflow

    def build_defaulting_a_writer(**kwargs: Any) -> TaskWorkflow:
        workflow = real_build_workflow(**kwargs)
        setattr(workflow, field, writer)
        return workflow

    monkeypatch.setattr(runner, 'build_workflow', build_defaulting_a_writer)

    with pytest.raises(EvalHermeticityError) as refused:
        _build(cell)

    assert refused.value.field == field
    assert repr(writer) in str(refused.value)
