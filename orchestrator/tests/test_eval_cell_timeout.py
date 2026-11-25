"""The wall-clock ceiling an eval cell runs under: where it comes from, and what the persisted cell records."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from _architect_eval_harness import run_architect_eval_hermetic
from _campaign_gate_helpers import eval_base_config
from _eval_workflow_cell import executor_task, stub_workflow_cell
from shared.testing_virtual_clock import virtual_clock_test

from orchestrator.evals import runner
from orchestrator.evals.configs import EvalConfig
from orchestrator.evals.results import ResultStore
from orchestrator.workflow import WorkflowOutcome

_TASK_PATH = Path('/fake/task.json')
_ARCHITECT = EvalConfig('architect-opus-high', 'claude', 'opus', 'high')
_IMPLEMENTER = EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max')


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
    stub_workflow_cell(monkeypatch, executor_task(tmp_path), workflow)

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
    stub_workflow_cell(monkeypatch, executor_task(tmp_path, timeout_minutes=45), workflow)

    result = await _run_executor('run_eval', tmp_path, ResultStore(tmp_path / 'results'))

    assert workflow.minutes_run == pytest.approx(45)
    assert result.timeout_minutes == 45


@virtual_clock_test
async def test_the_callers_override_beats_the_fixture_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    workflow = _CutOffWorkflow()
    stub_workflow_cell(monkeypatch, executor_task(tmp_path, timeout_minutes=45), workflow)

    result = await _run_executor(
        'run_eval', tmp_path, ResultStore(tmp_path / 'results'), timeout_override=90,
    )

    assert workflow.minutes_run == pytest.approx(90)
    assert result.timeout_minutes == 90


class _HangingInvoke:
    """An architect invoke that never answers; records when it started and when it was cut off."""

    def __init__(self) -> None:
        self.started_at: float | None = None
        self.cut_off_at: float | None = None

    async def invoke(self, **_kw) -> None:
        loop = asyncio.get_running_loop()
        self.started_at = loop.time()
        try:
            await asyncio.Event().wait()
        finally:
            self.cut_off_at = loop.time()

    @property
    def minutes_run(self) -> float:
        assert self.started_at is not None and self.cut_off_at is not None
        return (self.cut_off_at - self.started_at) / 60


@virtual_clock_test
async def test_an_architect_cell_without_a_fixture_ceiling_keeps_its_one_hour_default(
    tmp_path: Path,
):
    hang = _HangingInvoke()
    store = ResultStore(tmp_path / 'results')

    result, _ = await run_architect_eval_hermetic(
        _ARCHITECT, produced_plan={'steps': []}, invoke_side_effect=hang.invoke, store=store,
    )

    assert hang.minutes_run == pytest.approx(60)
    assert result.timeout_minutes == 60
    [persisted] = store.load_all()
    assert persisted.timeout_minutes == 60


@virtual_clock_test
async def test_a_fixture_ceiling_overrides_the_architect_default(tmp_path: Path):
    hang = _HangingInvoke()

    result, _ = await run_architect_eval_hermetic(
        _ARCHITECT, produced_plan={'steps': []}, invoke_side_effect=hang.invoke,
        task_timeout_minutes=45, store=ResultStore(tmp_path / 'results'),
    )

    assert hang.minutes_run == pytest.approx(45)
    assert result.timeout_minutes == 45


class _PropagatingWorkflow:
    """Runs until cancelled and lets the cancellation propagate."""

    async def run(self) -> SimpleNamespace:
        await asyncio.Event().wait()
        raise AssertionError('the event is never set')


class _EndsCancelledWorkflow:
    """Ends CANCELLED at once, long before any ceiling."""

    async def run(self) -> SimpleNamespace:
        return SimpleNamespace(outcome=WorkflowOutcome.CANCELLED)


class _FailsWhileCancelledWorkflow:
    """Runs until cancelled, then raises while finalising the cancellation."""

    async def run(self) -> SimpleNamespace:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError('finalising the cancellation failed') from None
        raise AssertionError('the event is never set')


class _InnerBoundExpiresWorkflow:
    """Hits a bound of its own, far short of the cell's ceiling."""

    async def run(self) -> SimpleNamespace:
        async with asyncio.timeout(60):
            await asyncio.Event().wait()
        raise AssertionError('the event is never set')


@virtual_clock_test
async def test_a_workflow_that_absorbs_the_deadline_reports_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    store = ResultStore(tmp_path / 'results')
    stub_workflow_cell(monkeypatch, executor_task(tmp_path), _CutOffWorkflow())

    result = await _run_executor('run_eval', tmp_path, store)

    assert result.outcome == 'timeout'
    [persisted] = store.load_all()
    assert persisted.outcome == 'timeout'


@virtual_clock_test
async def test_a_workflow_that_propagates_the_deadline_reports_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    stub_workflow_cell(monkeypatch, executor_task(tmp_path), _PropagatingWorkflow())

    result = await _run_executor('run_eval', tmp_path, ResultStore(tmp_path / 'results'))

    assert result.outcome == 'timeout'


@virtual_clock_test
async def test_a_workflow_that_ends_cancelled_before_its_ceiling_stays_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    stub_workflow_cell(monkeypatch, executor_task(tmp_path), _EndsCancelledWorkflow())

    result = await _run_executor('run_eval', tmp_path, ResultStore(tmp_path / 'results'))

    assert result.outcome == 'cancelled'


@virtual_clock_test
async def test_a_workflow_that_fails_after_its_ceiling_fired_reports_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    stub_workflow_cell(monkeypatch, executor_task(tmp_path), _FailsWhileCancelledWorkflow())

    result = await _run_executor('run_eval', tmp_path, ResultStore(tmp_path / 'results'))

    assert result.outcome == 'timeout'


@virtual_clock_test
async def test_a_bound_of_the_workflows_own_expiring_before_the_ceiling_reports_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    stub_workflow_cell(monkeypatch, executor_task(tmp_path), _InnerBoundExpiresWorkflow())

    result = await _run_executor('run_eval', tmp_path, ResultStore(tmp_path / 'results'))

    assert result.outcome == 'blocked'
