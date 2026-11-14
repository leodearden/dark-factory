"""Boundary row 26: an eval cell is never built around a production store.

S1 boundary row 26, Contract C11 of
plans/review-all-s1-one-escalation-filing-entry-point-prd.md. The eval cell
factory, ``evals/runner.py::_cell_workflow``, refuses every store through
which a ``TaskWorkflow`` writes production state (the escalation queue, and
with it the filer the workflow derives from it; the steward factory; the
event store; the cost store) with an ``EvalHermeticityError`` that names the
field and the offending object, before any workflow is built.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest
from escalation.queue import EscalationQueue
from shared.cost_store import CostStore
from shared.project_layout import escalation_queue_dir, runs_db

from orchestrator.config import OrchestratorConfig
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.hermeticity import EvalHermeticityError, WorkflowStores
from orchestrator.evals.runner import _cell_workflow, build_eval_orch_config
from orchestrator.event_store import EventStore
from orchestrator.workflow import TaskWorkflow

# Each field holding its real production object, rooted at a project's own paths.
_PRODUCTION_STORES: dict[str, Callable[[Path], WorkflowStores]] = {
    'escalation_queue': lambda proj: WorkflowStores(
        escalation_queue=EscalationQueue(escalation_queue_dir(proj)),
    ),
    'steward_factory': lambda proj: WorkflowStores(
        steward_factory=lambda worktree, config_dir: None,
    ),
    'event_store': lambda proj: WorkflowStores(
        event_store=EventStore(runs_db(proj), run_id='r'),
    ),
    'cost_store': lambda proj: WorkflowStores(cost_store=CostStore(runs_db(proj))),
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


def _build(cell: _Cell, **stores: WorkflowStores) -> TaskWorkflow:
    return _cell_workflow(
        cell.task, cell.orch_config, cell.worktree, None,
        replay_framed=False, usage_gate=None, **stores,
    )


@pytest.mark.parametrize('field', list(_PRODUCTION_STORES))
def test_each_production_store_handed_to_the_cell_is_refused(cell: _Cell, field: str) -> None:
    stores = _PRODUCTION_STORES[field](cell.project_root)
    store = getattr(stores, field)

    with pytest.raises(EvalHermeticityError) as refused:
        _build(cell, stores=stores)

    assert refused.value.field == field
    assert field in str(refused.value)
    assert repr(store) in str(refused.value)


def test_two_production_stores_name_the_first_declared_field(cell: _Cell) -> None:
    stores = WorkflowStores(
        escalation_queue=EscalationQueue(escalation_queue_dir(cell.project_root)),
        cost_store=CostStore(runs_db(cell.project_root)),
    )

    with pytest.raises(EvalHermeticityError) as refused:
        _build(cell, stores=stores)

    assert refused.value.field == 'escalation_queue'


@pytest.mark.parametrize('stores', [{}, {'stores': WorkflowStores()}], ids=['default', 'explicit'])
def test_with_no_stores_the_cell_builds_as_today(
    cell: _Cell, stores: dict[str, WorkflowStores],
) -> None:
    workflow = _build(cell, **stores)

    assert isinstance(workflow, TaskWorkflow)
    assert workflow.escalation_queue is None
    assert workflow.event_store is None
    assert workflow.cost_store is None
    assert workflow.worktree == cell.worktree
