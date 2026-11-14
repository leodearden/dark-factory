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
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.queue import EscalationQueue
from shared.repo_paths import DARK_FACTORY_ROOT_ENV

from orchestrator.before_done_script import ProcessTeardown, ScriptTimeout
from orchestrator.deterministic_runner import DeterministicRunner
from orchestrator.proc_supervision import RestartDisposition, RestartOutcome, RestartPlan
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome

RELATIVE_SCRIPT = 'data/ops/deploy.sh'
_ABSENT = object()
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


async def _run_capturing_restart_plan(
    runner: DeterministicRunner, task: dict,
) -> tuple[WorkflowOutcome, RestartPlan]:
    captured_plans: list[RestartPlan] = []

    async def _fake_execute(self, *, runner=None, inspector=None):
        captured_plans.append(self)
        return RestartOutcome(disposition=RestartDisposition.SCHEDULED)

    with patch.object(RestartPlan, 'execute', _fake_execute):
        outcome = await runner.run(_assignment(task))
    [plan] = captured_plans
    return outcome, plan


def _self_target_task(task_id: str, own_unit: str) -> dict:
    return _task(task_id, {
        'script': 'scripts/restart.sh',
        'timeout_secs': 10,
        'target_unit': own_unit,
    })


def _assert_warned_no_project_root(caplog: pytest.LogCaptureFixture) -> None:
    warnings = [
        r for r in caplog.records
        if r.name == 'orchestrator.deterministic_runner'
        and r.levelno == logging.WARNING
        and 'no configured project_root' in r.getMessage()
    ]
    assert warnings, [r.getMessage() for r in caplog.records]


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
        task = _self_target_task('6467', own_unit)
        runner = DeterministicRunner(
            scheduler=_scheduler(task, project_root),
            escalation_queue=queue,
            own_unit_resolver=lambda: own_unit,
        )

        outcome, plan = await _run_capturing_restart_plan(runner, task)

        assert outcome == WorkflowOutcome.DONE
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
        _assert_warned_no_project_root(caplog)

    @pytest.mark.parametrize('shape', ['mock-config', 'no-config'])
    async def test_detached_restart_falls_back_to_process_cwd_with_warning(
        self, shape: str, elsewhere: Path, queue: EscalationQueue,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        own_unit = 'orchestrator-self.service'
        task = _self_target_task('6470', own_unit)
        runner = DeterministicRunner(
            scheduler=_degraded_scheduler(shape, task),
            escalation_queue=queue,
            own_unit_resolver=lambda: own_unit,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.deterministic_runner'):
            outcome, plan = await _run_capturing_restart_plan(runner, task)

        assert outcome == WorkflowOutcome.DONE
        assert plan.cwd == elsewhere.resolve()
        assert plan.script == elsewhere.resolve() / 'scripts/restart.sh'
        _assert_warned_no_project_root(caplog)


@pytest.mark.asyncio
class TestScriptTimeoutEscalationNamesTheScriptThatRan:
    @pytest.mark.parametrize('target_unit', [None, 'orchestrator-other.service'])
    async def test_survivor_search_key_is_the_resolved_absolute_script(
        self, target_unit: str | None, project_root: Path, elsewhere: Path,
        queue: EscalationQueue,
    ) -> None:
        task = _task('6471', {
            'script': RELATIVE_SCRIPT,
            'timeout_secs': 10,
            'target_unit': target_unit,
        })
        runner = DeterministicRunner(
            scheduler=_scheduler(task, project_root),
            escalation_queue=queue,
            own_unit_resolver=lambda: 'orchestrator.service',
            unit_inspector=AsyncMock(side_effect=[BASELINE_UNIT_STATE, FRESH_UNIT_STATE]),
            script_runner=AsyncMock(
                side_effect=ScriptTimeout(
                    10, exit_code=0, teardown=ProcessTeardown.NOT_SIGNALLED,
                ),
            ),
        )

        outcome = await runner.run(_assignment(task))

        assert outcome == WorkflowOutcome.BLOCKED
        [esc] = queue.get_by_task('6471', status='pending')
        assert esc.summary.startswith('Deploy script timed out'), esc.summary
        assert str(project_root.resolve() / RELATIVE_SCRIPT) in esc.detail, esc.detail


FACTORY_SCRIPT = 'scripts/report-cwd.sh'


def _factory_predicate_task(task_id: str, marker: Path) -> dict:
    return _task(task_id, {
        'script': FACTORY_SCRIPT,
        'script_root': 'factory',
        'kind': 'predicate',
        'timeout_secs': 10,
        'env': {'MARKER': str(marker)},
    })


def _never_consulted() -> Path | None:
    raise AssertionError('the factory root resolver must not run for a project-rooted script')


@pytest.fixture
def factory_root(tmp_path: Path) -> Path:
    root = tmp_path / 'factory'
    (root / 'scripts').mkdir(parents=True)
    (root / 'scripts' / 'watcher-rearm.sh').touch()
    script = root / FACTORY_SCRIPT
    script.write_text(WRITE_CWD_TO_MARKER)
    script.chmod(0o755)
    return root


@pytest.mark.asyncio
class TestFactoryRootedPredicate:
    async def test_factory_predicate_execs_factory_script_with_cwd_at_project_root(
        self, project_root: Path, factory_root: Path, elsewhere: Path, marker: Path,
        queue: EscalationQueue, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, str(factory_root))
        task = _factory_predicate_task('6386', marker)
        scheduler = _scheduler(task, project_root)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)

        outcome = await runner.run(_assignment(task))

        assert outcome == WorkflowOutcome.DONE
        assert marker.read_text().strip() == str(project_root.resolve())
        assert _done_provenance_kind(scheduler) == 'deterministic-milestone'

    async def test_unresolvable_factory_root_is_a_no_verdict_leg(
        self, project_root: Path, elsewhere: Path, marker: Path, queue: EscalationQueue,
    ) -> None:
        task = _factory_predicate_task('6387', marker)
        scheduler = _scheduler(task, project_root)
        runner = DeterministicRunner(
            scheduler=scheduler, escalation_queue=queue, factory_root_resolver=lambda: None,
        )

        outcome = await runner.run(_assignment(task))

        assert outcome == WorkflowOutcome.BLOCKED
        [esc] = queue.get_by_task('6387', status='pending')
        assert esc.summary == 'Predicate script root unresolvable', esc.summary
        assert esc.category == 'infra_issue'
        assert DARK_FACTORY_ROOT_ENV in esc.detail, esc.detail
        assert FACTORY_SCRIPT in esc.detail, esc.detail
        statuses = [c.args[1] for c in scheduler.set_task_status.await_args_list]
        assert 'done' not in statuses

    async def test_factory_rooted_deploy_is_refused_without_running_the_script(
        self, project_root: Path, factory_root: Path, elsewhere: Path, marker: Path,
        queue: EscalationQueue,
    ) -> None:
        resolver_calls: list[None] = []

        def _resolver() -> Path:
            resolver_calls.append(None)
            return factory_root

        task = _task('6389', {
            'script': FACTORY_SCRIPT,
            'script_root': 'factory',
            'timeout_secs': 10,
            'env': {'MARKER': str(marker)},
        })
        scheduler = _scheduler(task, project_root)
        runner = DeterministicRunner(
            scheduler=scheduler, escalation_queue=queue, factory_root_resolver=_resolver,
        )

        outcome = await runner.run(_assignment(task))

        assert outcome == WorkflowOutcome.BLOCKED
        assert not marker.exists()
        assert resolver_calls == []
        [esc] = queue.get_by_task('6389', status='pending')
        assert "requires kind='predicate'" in esc.detail, esc.detail

    async def test_project_rooted_deploy_never_consults_the_factory_resolver(
        self, project_root: Path, elsewhere: Path, marker: Path, queue: EscalationQueue,
    ) -> None:
        task = _task('6388', {
            'script': RELATIVE_SCRIPT,
            'timeout_secs': 10,
            'env': {'MARKER': str(marker)},
        })
        runner = DeterministicRunner(
            scheduler=_scheduler(task, project_root),
            escalation_queue=queue,
            factory_root_resolver=_never_consulted,
        )

        outcome = await runner.run(_assignment(task))

        assert outcome == WorkflowOutcome.DONE
        assert marker.read_text().strip() == str(project_root.resolve())


@dataclass(frozen=True)
class _RefusalLeg:
    fields: dict
    summary: str
    own_unit: str | None = None


REFUSAL_LEGS = [
    pytest.param(
        _RefusalLeg(
            {'kind': 'predicate', 'target_unit': None},
            'Predicate before_done refused (check not run)',
        ),
        id='predicate',
    ),
    pytest.param(
        _RefusalLeg({}, 'Deploy refused before_done (no target_unit)'),
        id='targetless-deploy',
    ),
    pytest.param(
        _RefusalLeg(
            {'target_unit': 'orchestrator-other.service'},
            'Deploy refused before_done: orchestrator-other.service',
            own_unit='orchestrator.service',
        ),
        id='cross-unit-deploy',
    ),
    pytest.param(
        _RefusalLeg(
            {'target_unit': 'orchestrator-self.service'},
            'Deploy refused before_done: orchestrator-self.service',
            own_unit='orchestrator-self.service',
        ),
        id='self-target-restart',
    ),
]


def _stamped_keys(scheduler) -> set[str]:
    return {key for call in scheduler.update_task.await_args_list for key in call.args[1]}

BEFORE_DONE_REFUSALS = [
    pytest.param({'cwd': 5}, ('before_done.cwd', "'int'"), id='cwd-not-a-string'),
    pytest.param({'script_root': 'elsewhere'}, ("'elsewhere'",), id='unknown-script-root'),
    pytest.param({'script': 5}, ('before_done.script', 'int'), id='script-not-a-string'),
    pytest.param({'script': _ABSENT}, ('before_done.script',), id='script-missing'),
]


@pytest.mark.asyncio
class TestRefusedBeforeDoneIsANamedInfraIssue:
    @pytest.mark.parametrize(('refusal_fields', 'detail_fragments'), BEFORE_DONE_REFUSALS)
    @pytest.mark.parametrize('leg', REFUSAL_LEGS)
    async def test_refusal_files_the_legs_own_summary_without_running_the_script(
        self, leg: _RefusalLeg, refusal_fields: dict, detail_fragments: tuple[str, ...],
        project_root: Path, elsewhere: Path, marker: Path, queue: EscalationQueue,
        request: pytest.FixtureRequest,
    ) -> None:
        task_id = f'9116-{request.node.callspec.id}'
        record = {
            'script': RELATIVE_SCRIPT,
            'timeout_secs': 10,
            'env': {'MARKER': str(marker)},
            **leg.fields,
            **refusal_fields,
        }
        task = _task(task_id, {key: value for key, value in record.items() if value is not _ABSENT})
        scheduler = _scheduler(task, project_root)
        runner_kwargs: dict = {}
        if leg.own_unit is not None:
            runner_kwargs['own_unit_resolver'] = lambda: leg.own_unit
        runner = DeterministicRunner(
            scheduler=scheduler, escalation_queue=queue, **runner_kwargs,
        )
        executed_plans: list[RestartPlan] = []

        async def _record_execute(self, *, runner=None, inspector=None):
            executed_plans.append(self)
            return RestartOutcome(disposition=RestartDisposition.SCHEDULED)

        with patch.object(RestartPlan, 'execute', _record_execute):
            outcome = await runner.run(_assignment(task))

        assert outcome == WorkflowOutcome.BLOCKED
        assert not marker.exists()
        assert executed_plans == []
        [esc] = queue.get_by_task(task_id, status='pending')
        assert esc.category == 'infra_issue'
        assert esc.summary == leg.summary, esc.summary
        for fragment in detail_fragments:
            assert fragment in esc.detail, esc.detail
        statuses = [c.args[1] for c in scheduler.set_task_status.await_args_list]
        assert 'done' not in statuses
        assert {'before_done_ran_at', 'deploy_state'}.isdisjoint(_stamped_keys(scheduler))

    async def test_fixed_record_runs_the_deploy_on_the_next_dispatch(
        self, project_root: Path, elsewhere: Path, marker: Path, queue: EscalationQueue,
    ) -> None:
        fixed = {'script': RELATIVE_SCRIPT, 'timeout_secs': 10, 'env': {'MARKER': str(marker)}}
        refused_task = _task('9117', {**fixed, 'cwd': 5})
        runner = DeterministicRunner(
            scheduler=_scheduler(refused_task, project_root), escalation_queue=queue,
        )
        assert await runner.run(_assignment(refused_task)) == WorkflowOutcome.BLOCKED
        [esc] = queue.get_by_task('9117', status='pending')
        queue.resolve(esc.id, 'cwd fixed via update_task', resolved_by='human')

        fixed_task = _task('9117', fixed)
        scheduler = _scheduler(fixed_task, project_root)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(_assignment(fixed_task))

        assert outcome == WorkflowOutcome.DONE
        assert marker.read_text().strip() == str(project_root.resolve())
        assert _done_provenance_kind(scheduler) == 'deterministic-deploy'
