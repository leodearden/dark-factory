"""The wall-clock ceiling an eval cell runs under: where it comes from, and what the persisted cell records."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from _campaign_gate_helpers import eval_base_config
from shared.testing_virtual_clock import virtual_clock_test

from orchestrator.evals import runner
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.metrics import EvalMetrics
from orchestrator.evals.results import ResultStore
from orchestrator.workflow import WorkflowOutcome

_TASK_PATH = Path('/fake/task.json')
_ARCHITECT = EvalConfig('architect-opus-high', 'claude', 'opus', 'high')
_IMPLEMENTER = EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max')


def _task(tmp_path: Path, **extra) -> dict:
    task = {
        'id': 'df_task_t1',
        'project_root': str(tmp_path),
        'pre_task_commit': 'basecommit',
        'task_definition': {'title': 'W', 'description': 'd'},
        'modules': ['pkg/mod.py'],
        'plan': {'steps': [{'id': 's1', 'status': 'pending'}]},
    }
    task.update(extra)
    return task


class _CutOffWorkflow:
    """Runs until cancelled, then returns a CANCELLED report, as TaskWorkflow.run does on a hard cancel."""

    def __init__(self) -> None:
        self.started_at: float | None = None
        self.cut_off_at: float | None = None

    async def run(self) -> SimpleNamespace:
        loop = asyncio.get_running_loop()
        self.started_at = loop.time()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cut_off_at = loop.time()
        return SimpleNamespace(outcome=WorkflowOutcome.CANCELLED)

    @property
    def minutes_run(self) -> float:
        assert self.started_at is not None and self.cut_off_at is not None
        return (self.cut_off_at - self.started_at) / 60


async def _fake_create_wt(*_a, **_k) -> tuple[Path, str]:
    return Path('/fake/wt'), 'run-t1'


async def _fake_collect(_wf, _wt, _task, *, role, replay_frame=None, sandbox_backend=None):
    return EvalMetrics(
        role_under_test=role, replay_frame=replay_frame, sandbox_backend=sandbox_backend,
    )


def _stub_cell(monkeypatch: pytest.MonkeyPatch, task: dict, workflow) -> None:
    monkeypatch.setattr(runner, 'create_eval_worktree', _fake_create_wt)
    monkeypatch.setattr(runner, 'build_workflow', lambda **_kw: workflow)
    monkeypatch.setattr(runner, 'collect_metrics', AsyncMock(side_effect=_fake_collect))
    monkeypatch.setattr(runner, 'load_task', lambda _p: task)


def _run_executor(executor: str, tmp_path: Path, store: ResultStore, **kwargs):
    if executor == 'run_end_to_end':
        return runner.run_end_to_end(
            _TASK_PATH, _ARCHITECT, _IMPLEMENTER, eval_base_config(tmp_path),
            usage_gate=None, store=store, **kwargs,
        )
    return runner.run_eval(
        _TASK_PATH, _IMPLEMENTER, eval_base_config(tmp_path),
        usage_gate=None, store=store, **kwargs,
    )


@pytest.mark.parametrize('executor', ['run_eval', 'run_end_to_end'])
@virtual_clock_test
async def test_a_workflow_cell_without_a_fixture_ceiling_is_cut_off_at_three_hours(
    executor: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    workflow = _CutOffWorkflow()
    store = ResultStore(tmp_path / 'results')
    _stub_cell(monkeypatch, _task(tmp_path), workflow)

    result = await _run_executor(executor, tmp_path, store)

    assert workflow.minutes_run == pytest.approx(180)
    assert result.timeout_minutes == 180
    [persisted] = store.load_all()
    assert persisted.timeout_minutes == 180


@virtual_clock_test
async def test_a_fixture_ceiling_overrides_the_workflow_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    workflow = _CutOffWorkflow()
    _stub_cell(monkeypatch, _task(tmp_path, timeout_minutes=45), workflow)

    result = await _run_executor('run_eval', tmp_path, ResultStore(tmp_path / 'results'))

    assert workflow.minutes_run == pytest.approx(45)
    assert result.timeout_minutes == 45


@virtual_clock_test
async def test_the_callers_override_beats_the_fixture_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    workflow = _CutOffWorkflow()
    _stub_cell(monkeypatch, _task(tmp_path, timeout_minutes=45), workflow)

    result = await _run_executor(
        'run_eval', tmp_path, ResultStore(tmp_path / 'results'), timeout_override=90,
    )

    assert workflow.minutes_run == pytest.approx(90)
    assert result.timeout_minutes == 90
