"""The runner's default script runner forwards what was configured into ``run_script``.

That is ``before_done``'s ``args`` and ``env``, and the constructor's
``reap_grace_secs``. Every case drives the public ``DeterministicRunner.run()``
with ``script_runner=None``, so the real adapter and the real subprocess run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.deterministic_runner import DeterministicRunner
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome


def _install_script(directory: Path, body: str) -> Path:
    script = directory / 'deploy.sh'
    script.write_text(f'#!/bin/sh\n{body}\n')
    script.chmod(0o755)
    return script


def _targetless_deploy(task_id: str, before_done: dict) -> dict:
    return {
        'id': task_id,
        'title': f'targetless deploy {task_id}',
        'description': 'Runs a before_done deploy script with no target_unit',
        'metadata': {
            'task_kind': 'deterministic',
            'always_escalates': False,
            'before_done': before_done,
        },
    }


def _runner(task: dict, project_root: Path, **kwargs) -> DeterministicRunner:
    scheduler = MagicMock()
    scheduler.config.project_root = project_root
    scheduler.set_task_status = AsyncMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_task = AsyncMock(return_value=task)
    return DeterministicRunner(
        scheduler=scheduler,
        escalation_queue=EscalationQueue(project_root / 'escalations'),
        script_runner=None,
        **kwargs,
    )


def _assignment(task: dict) -> TaskAssignment:
    return TaskAssignment(task_id=str(task['id']), task=task, modules=[])


@pytest.mark.asyncio
class TestDefaultRunnerForwardsBeforeDone:
    async def test_args_and_env_reach_the_script(self, tmp_path: Path) -> None:
        marker = tmp_path / 'marker'
        script = _install_script(tmp_path, 'printf "%s %s" "$1" "$GREETING" > "$MARKER"')
        task = _targetless_deploy('6488a', {
            'script': str(script),
            'args': ['first'],
            'env': {'GREETING': 'hello', 'MARKER': str(marker)},
            'timeout_secs': 10,
        })

        outcome = await _runner(task, tmp_path).run(_assignment(task))

        assert outcome == WorkflowOutcome.DONE
        assert marker.read_text() == 'first hello'


@pytest.mark.asyncio
class TestDefaultRunnerForwardsReapGrace:
    """``DeterministicRunner(reap_grace_secs=...)`` bounds the timed-out
    deploy script's reap, not the module default.

    The bounded reap itself is
    ``orchestrator/src/orchestrator/before_done_script.py::terminate_process_tree``,
    whose own tests pass the bound directly.  This pins the runner's half:
    the default script runner forwards the constructor value.
    """

    async def test_constructor_reap_grace_bounds_the_timeout_reap(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A zero bound abandons the reap, and the warning names that bound.

        ``asyncio.wait_for`` with a timeout of 0 raises before it starts the
        awaitable, so the abandoned-reap warning is deterministic here.  The
        5s module default would have reaped the SIGKILLed script in time and
        logged nothing.
        """
        script = _install_script(tmp_path, 'sleep 30')
        task = _targetless_deploy('6488', {'script': str(script), 'timeout_secs': 1})
        runner = _runner(task, tmp_path, unit_inspector=AsyncMock(), reap_grace_secs=0)

        with caplog.at_level('WARNING', logger='orchestrator.before_done_script'):
            outcome = await asyncio.wait_for(runner.run(_assignment(task)), timeout=20)

        assert outcome == WorkflowOutcome.BLOCKED
        reap_warnings = [
            r.getMessage() for r in caplog.records
            if r.name == 'orchestrator.before_done_script'
        ]
        assert len(reap_warnings) == 1, reap_warnings
        assert 'reap_grace_secs=0 after termination signal' in reap_warnings[0], (
            reap_warnings[0]
        )
