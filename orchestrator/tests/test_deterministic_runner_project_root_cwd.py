"""A before_done script runs under the configured project_root (task 6464).

The SCP-838 shape: the orchestrator PROCESS runs from one checkout while its
config names another project_root. A relative ``before_done.script`` and an
absent ``cwd`` must resolve against the configured project_root — the root
fused-memory's submit guard validated the script under — never against the
process cwd. Every case drives the public ``DeterministicRunner.run()`` with the
real default script runner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.queue import EscalationQueue

from orchestrator.deterministic_runner import DeterministicRunner
from orchestrator.proc_supervision import RestartDisposition, RestartOutcome, RestartPlan
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome

RELATIVE_SCRIPT = 'data/ops/deploy.sh'
WRITE_CWD_TO_MARKER = '#!/bin/sh\npwd -P > "$MARKER"\n'

BASELINE_UNIT_STATE: dict = {
    'MainPID': 100,
    'ActiveState': 'active',
    'ActiveEnterTimestamp': 'Mon 2026-06-23 10:00:00 UTC',
    'ActiveEnterTimestampMonotonic': 1_000_000,
}
FRESH_UNIT_STATE: dict = {
    'MainPID': 200,
    'ActiveState': 'active',
    'ActiveEnterTimestamp': 'Mon 2026-06-23 10:01:00 UTC',
    'ActiveEnterTimestampMonotonic': 2_000_000,
}


def _install_script(root: Path) -> None:
    script = root / RELATIVE_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(WRITE_CWD_TO_MARKER)
    script.chmod(0o755)


def _task(task_id: str, before_done: dict) -> dict:
    return {
        'id': task_id,
        'title': f'before_done task {task_id}',
        'description': 'Runs a repo-relative before_done script',
        'metadata': {
            'task_kind': 'deterministic',
            'always_escalates': False,
            'before_done': before_done,
        },
    }


def _scheduler(task: dict, project_root: Path) -> MagicMock:
    scheduler = MagicMock()
    scheduler.config.project_root = project_root
    scheduler.set_task_status = AsyncMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_task = AsyncMock(return_value=task)
    return scheduler


class _SchedulerWithoutConfig:
    def __init__(self, task: dict) -> None:
        self.set_task_status = AsyncMock()
        self.update_task = AsyncMock(return_value=True)
        self.get_task = AsyncMock(return_value=task)


def _degraded_scheduler(shape: str, task: dict):
    if shape == 'mock-config':
        scheduler = MagicMock()
        scheduler.set_task_status = AsyncMock()
        scheduler.update_task = AsyncMock(return_value=True)
        scheduler.get_task = AsyncMock(return_value=task)
        assert not isinstance(scheduler.config.project_root, (str, Path))
        return scheduler
    scheduler = _SchedulerWithoutConfig(task)
    assert not hasattr(scheduler, 'config')
    return scheduler


def _assignment(task: dict) -> TaskAssignment:
    return TaskAssignment(task_id=str(task['id']), task=task, modules=[])


def _done_provenance_kind(scheduler) -> str:
    scheduler.set_task_status.assert_awaited_once()
    call = scheduler.set_task_status.call_args
    assert call.args[1] == 'done'
    return call.kwargs['done_provenance']['kind']


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / 'project'
    root.mkdir()
    _install_script(root)
    return root


@pytest.fixture
def elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    process_cwd = tmp_path / 'elsewhere'
    process_cwd.mkdir()
    monkeypatch.chdir(process_cwd)
    return process_cwd


@pytest.fixture
def marker(tmp_path: Path) -> Path:
    return tmp_path / 'cwd-marker'


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


@pytest.mark.asyncio
class TestBeforeDoneRunsUnderConfiguredProjectRoot:
    async def test_targetless_deploy_runs_relative_script_in_project_root(
        self, project_root: Path, elsewhere: Path, marker: Path, queue: EscalationQueue,
    ) -> None:
        task = _task('6464', {
            'script': RELATIVE_SCRIPT,
            'timeout_secs': 10,
            'env': {'MARKER': str(marker)},
        })
        scheduler = _scheduler(task, project_root)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)

        outcome = await runner.run(_assignment(task))

        assert outcome == WorkflowOutcome.DONE
        assert marker.read_text().strip() == str(project_root.resolve())
        assert _done_provenance_kind(scheduler) == 'deterministic-deploy'

    async def test_predicate_runs_relative_script_in_project_root(
        self, project_root: Path, elsewhere: Path, marker: Path, queue: EscalationQueue,
    ) -> None:
        task = _task('6465', {
            'script': RELATIVE_SCRIPT,
            'timeout_secs': 10,
            'env': {'MARKER': str(marker)},
            'kind': 'predicate',
            'target_unit': None,
        })
        scheduler = _scheduler(task, project_root)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)

        outcome = await runner.run(_assignment(task))

        assert outcome == WorkflowOutcome.DONE
        assert marker.read_text().strip() == str(project_root.resolve())

    async def test_cross_unit_deploy_runs_relative_script_in_project_root(
        self, project_root: Path, elsewhere: Path, marker: Path, queue: EscalationQueue,
    ) -> None:
        task = _task('6466', {
            'script': RELATIVE_SCRIPT,
            'timeout_secs': 10,
            'env': {'MARKER': str(marker)},
            'target_unit': 'orchestrator-other.service',
        })
        scheduler = _scheduler(task, project_root)
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            own_unit_resolver=lambda: 'orchestrator.service',
            unit_inspector=AsyncMock(side_effect=[BASELINE_UNIT_STATE, FRESH_UNIT_STATE]),
        )

        outcome = await runner.run(_assignment(task))

        assert outcome == WorkflowOutcome.DONE
        assert marker.read_text().strip() == str(project_root.resolve())

    async def test_self_target_detached_restart_plan_uses_project_root(
        self, project_root: Path, elsewhere: Path, queue: EscalationQueue,
    ) -> None:
        own_unit = 'orchestrator-self.service'
        task = _task('6467', {
            'script': 'scripts/restart.sh',
            'timeout_secs': 10,
            'target_unit': own_unit,
        })
        scheduler = _scheduler(task, project_root)
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            own_unit_resolver=lambda: own_unit,
        )
        captured_plans: list[RestartPlan] = []

        async def _fake_execute(self, *, runner=None, inspector=None):
            captured_plans.append(self)
            return RestartOutcome(disposition=RestartDisposition.SCHEDULED)

        with patch.object(RestartPlan, 'execute', _fake_execute):
            outcome = await runner.run(_assignment(task))

        assert outcome == WorkflowOutcome.DONE
        [plan] = captured_plans
        assert plan.cwd == project_root.resolve()
        assert plan.script == project_root.resolve() / 'scripts/restart.sh'

    async def test_explicit_cwd_is_honoured_while_script_resolves_under_project_root(
        self, tmp_path: Path, project_root: Path, elsewhere: Path, marker: Path,
        queue: EscalationQueue,
    ) -> None:
        other_dir = tmp_path / 'other'
        other_dir.mkdir()
        task = _task('6468', {
            'script': RELATIVE_SCRIPT,
            'cwd': str(other_dir),
            'timeout_secs': 10,
            'env': {'MARKER': str(marker)},
        })
        scheduler = _scheduler(task, project_root)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)

        outcome = await runner.run(_assignment(task))

        assert outcome == WorkflowOutcome.DONE
        assert marker.read_text().strip() == str(other_dir.resolve())


@pytest.mark.asyncio
class TestNoConfiguredProjectRoot:
    @pytest.mark.parametrize('shape', ['mock-config', 'no-config'])
    async def test_falls_back_to_process_cwd_with_warning(
        self, shape: str, elsewhere: Path, marker: Path, queue: EscalationQueue,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _install_script(elsewhere)
        task = _task('6469', {
            'script': RELATIVE_SCRIPT,
            'timeout_secs': 10,
            'env': {'MARKER': str(marker)},
        })
        runner = DeterministicRunner(
            scheduler=_degraded_scheduler(shape, task), escalation_queue=queue,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.deterministic_runner'):
            outcome = await runner.run(_assignment(task))

        assert outcome == WorkflowOutcome.DONE
        assert marker.read_text().strip() == str(elsewhere.resolve())
        warnings = [
            r for r in caplog.records
            if r.name == 'orchestrator.deterministic_runner'
            and r.levelno == logging.WARNING
            and 'no configured project_root' in r.getMessage()
        ]
        assert warnings, [r.getMessage() for r in caplog.records]
