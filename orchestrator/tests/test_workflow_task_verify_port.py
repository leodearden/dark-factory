"""δ (task 6670): the workflow's task-verify port and the green's runner.

``build_workflow(..., task_verify=port)`` routes the implement-phase verify
through *port* with the argument list ``run_scoped_verification`` gets today,
still inside the task lane's lease; an un-injected workflow keeps calling the
module-level name (PRD ``plans/remote-task-leg-verify-prd.md`` §5.6, rows 16
and 17).  Every ``workflow_verify`` row names the host whose verdict it
records: the port's recorded decision for the green attempt, else ``local``.

Imports orchestrator.workflow only, never orchestrator.merge_lane.
"""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _git_fixtures import seed_repo
from _orch_helpers import MOCK_WORKFLOW_PROJECT_ROOT, pydantic_spec

from orchestrator.artifacts import ReviewAggregation, TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.task_verify_dispatch import DispatchDecision
from orchestrator.verify import VerifyResult, run_scoped_verification
from orchestrator.verify_checkpoint import green_checkpoint_at_tip
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, WorkflowState, build_workflow

_TASK = '6670'
_TIP = 'a' * 40
_BASE = 'b' * 40
_WORKTREE = Path('/tmp/fake-worktree-6670')


def _passed() -> VerifyResult:
    return VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary='All checks passed',
    )


def _config() -> Any:
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.git = GitConfig()
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = MOCK_WORKFLOW_PROJECT_ROOT
    config.max_consecutive_infra_resumes = 3
    config.verify_infra_retry_max_attempts = 3
    config.verify_infra_retry_backoff_secs = 0.01
    config.verify_infra_retry_max_backoff_secs = 0.1
    config.max_verify_attempts = 5
    config.max_failure_signature_repeat = 3
    config.max_opaque_timeout_attempts = 3
    config.rebase_before_verify = False
    config.escalate_preexisting_main_break = False
    return config


class _LeaseLog:
    """The task lane's lease and in-use span, recording when each is entered and left."""

    def __init__(self) -> None:
        self.log: list[str] = []

    @contextlib.asynccontextmanager
    async def lease(self, worktree: Path) -> AsyncIterator[None]:
        self.log.append('lease_enter')
        try:
            yield
        finally:
            self.log.append('lease_exit')

    @contextlib.asynccontextmanager
    async def in_use(self, worktree: Path) -> AsyncIterator[None]:
        yield


def _workflow(
    *, task_verify: Any = None, event_store: EventStore | None = None,
    lease: _LeaseLog | None = None,
) -> TaskWorkflow:
    assignment = MagicMock()
    assignment.task_id = _TASK
    assignment.task = {'id': _TASK, 'title': 'T', 'description': 'd', 'metadata': {}}
    assignment.modules = ['mod_a']
    scheduler = MagicMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')
    git_ops = MagicMock()
    git_ops.merge_base_with_main = AsyncMock(return_value=_BASE)
    if lease is not None:
        git_ops.task_verify_lease = lease.lease
        git_ops.lane_in_use = lease.in_use
    queue = MagicMock()
    queue.get_by_task = MagicMock(return_value=[])
    wf = build_workflow(
        assignment=assignment, config=_config(), git_ops=git_ops, scheduler=scheduler,
        briefing=MagicMock(), mcp=MagicMock(), escalation_queue=queue,
        event_store=event_store, task_verify=task_verify,
    )
    wf.worktree = _WORKTREE
    wf.artifacts = MagicMock()
    wf.artifacts.read_iteration_log = MagicMock(return_value=([], []))
    return wf


def _expected_call(wf: TaskWorkflow, event_store: EventStore | None) -> tuple[tuple, dict]:
    return (_WORKTREE, wf.config, wf._module_configs), {
        'task_files': wf._task_files,
        'attempt_id': 1,
        'task_id': _TASK,
        'archive_root': MOCK_WORKFLOW_PROJECT_ROOT / 'data' / 'verify-logs',
        'force_workspace': False,
        'role': 'task',
        'event_store': event_store,
    }


class _RecordingPort:
    """A task-verify port that records its decisions, as the dispatcher does."""

    def __init__(
        self, decisions: dict[tuple[str, int], DispatchDecision] | None = None,
        lease: _LeaseLog | None = None,
    ) -> None:
        self.decisions = decisions or {}
        self.lease = lease
        self.calls: list[tuple[tuple, dict]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> VerifyResult:
        self.calls.append((args, kwargs))
        if self.lease is not None:
            self.lease.log.append('port')
        return _passed()

    def decision_for(self, task_id: str, attempt_id: int) -> DispatchDecision | None:
        return self.decisions.get((task_id, attempt_id))


def _laptop_decision() -> DispatchDecision:
    return DispatchDecision(
        runner='laptop', reason='remote', tip_sha=_TIP, base_sha=_BASE,
        remote_wall_secs=1.0, lease_held_secs=1.5, cross_check=None,
        red_confirmed_locally=None, remote_wait_secs=0.5,
    )


async def _verify_then_review(wf: TaskWorkflow) -> None:
    """Drive the verify loop to its green, then the VERIFY->REVIEW edge that emits it."""
    assert await wf._verify_debugfix_loop() == WorkflowOutcome.DONE
    wf._verify_green_tip_sha = _TIP
    wf._base_commit = _BASE
    wf.state = WorkflowState.VERIFY
    wf._enter_phase(WorkflowState.REVIEW)


def _done_plan_artifacts(wf: TaskWorkflow) -> TaskArtifacts:
    """Real artifacts in the worktree whose plan is done, so EXECUTE passes straight through."""
    assert wf.worktree is not None
    artifacts = TaskArtifacts(wf.worktree)
    artifacts.init(_TASK, 'T', 'd', base_commit=_BASE)
    artifacts.write_plan({
        'task_id': _TASK, 'title': 'T', 'files': [], 'analysis': 'a', 'prerequisites': [],
        'steps': [{
            'id': 'step-1', 'type': 'impl', 'description': 'd', 'status': 'done', 'commit': _TIP,
        }],
    })
    artifacts.stamp_plan_provenance(wf.session_id)
    return artifacts


def _verify_rows(store: EventStore) -> list[dict[str, Any]]:
    return [row['data'] for row in store.fetch_events_by_type(EventType.workflow_verify)]


@pytest.mark.asyncio
class TestRouting:
    async def test_an_uninjected_workflow_calls_the_module_name_unchanged(
        self, tmp_path: Path,
    ) -> None:
        """Row 16: byte-identical to today."""
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        wf = _workflow(event_store=store)

        with patch(
            'orchestrator.workflow.run_scoped_verification', new=AsyncMock(return_value=_passed()),
        ) as module_verify:
            assert await wf._verify_debugfix_loop() == WorkflowOutcome.DONE

        [call] = module_verify.await_args_list
        assert (call.args, call.kwargs) == _expected_call(wf, store)

    async def test_an_injected_port_gets_the_same_call_inside_the_lease(
        self, tmp_path: Path,
    ) -> None:
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        lease = _LeaseLog()
        port = _RecordingPort(lease=lease)
        wf = _workflow(task_verify=port, event_store=store, lease=lease)

        with patch(
            'orchestrator.workflow.run_scoped_verification', new=AsyncMock(return_value=_passed()),
        ) as module_verify:
            assert await wf._verify_debugfix_loop() == WorkflowOutcome.DONE

        assert port.calls == [_expected_call(wf, store)]
        module_verify.assert_not_awaited()
        assert lease.log == ['lease_enter', 'port', 'lease_exit']

    async def test_an_async_mock_port_is_awaited_like_the_module_name(self) -> None:
        port = AsyncMock(spec=run_scoped_verification, return_value=_passed())
        wf = _workflow(task_verify=port)

        with patch('orchestrator.workflow.run_scoped_verification') as module_verify:
            assert await wf._verify_debugfix_loop() == WorkflowOutcome.DONE

        [call] = port.await_args_list
        assert (call.args, call.kwargs) == _expected_call(wf, None)
        module_verify.assert_not_called()


@pytest.mark.asyncio
class TestGreenRunner:
    async def test_the_green_names_the_port_s_recorded_runner(self, tmp_path: Path) -> None:
        """Row 17: the runner rides beside the tip, and the checkpoint still hits."""
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        port = _RecordingPort({(_TASK, 1): _laptop_decision()})
        wf = _workflow(task_verify=port, event_store=store)

        await _verify_then_review(wf)

        [row] = _verify_rows(store)
        assert (row['runner'], row['tip_sha'], row['site']) == ('laptop', _TIP, 'implement')
        assert green_checkpoint_at_tip(store, EventType.workflow_verify, _TASK, _TIP) is True

    @pytest.mark.parametrize('port', [
        pytest.param(_RecordingPort(), id='port-without-a-decision'),
        pytest.param(None, id='uninjected'),
    ])
    async def test_an_unrecorded_green_is_local(self, tmp_path: Path, port: Any) -> None:
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        wf = _workflow(task_verify=port, event_store=store)

        with patch(
            'orchestrator.workflow.run_scoped_verification', new=AsyncMock(return_value=_passed()),
        ):
            await _verify_then_review(wf)

        [row] = _verify_rows(store)
        assert row['runner'] == 'local'

    async def test_the_verify_review_loop_pairs_the_runner_with_the_real_tip(
        self, tmp_path: Path,
    ) -> None:
        """Row 17 through the loop that sequences VERIFY->REVIEW, not a hand-built edge.

        The loop resets the runner at VERIFY entry, captures the green tip
        from the worktree's real HEAD, then emits at REVIEW entry.
        """
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        port = _RecordingPort({(_TASK, 1): _laptop_decision()})
        wf = _workflow(task_verify=port, event_store=store)
        wf.worktree = seed_repo(tmp_path / 'wt')
        head = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=wf.worktree, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        wf.artifacts = _done_plan_artifacts(wf)
        wf.config.max_execute_iterations = 5
        wf.config.inter_iteration_rebase = False
        wf.config.max_review_cycles = 2
        wf.config.max_amendment_rounds = 1
        passing_review = ReviewAggregation(
            has_blocking_issues=False, blocking_issues=[], suggestions=[], reviews={'analyst': {}},
        )

        with patch.object(wf, '_review', AsyncMock(return_value=passing_review)), \
             patch.object(wf, '_write_suggestions_to_memory', AsyncMock()):
            assert await wf._execute_verify_review_loop() == WorkflowOutcome.DONE

        assert len(port.calls) == 1
        [row] = _verify_rows(store)
        assert (row['runner'], row['tip_sha'], row['site']) == ('laptop', head, 'implement')

    async def test_the_pre_merge_reverify_stays_local(self, tmp_path: Path) -> None:
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        port = _RecordingPort({(_TASK, 1): _laptop_decision()})
        wf = _workflow(task_verify=port, event_store=store)
        wf._get_head_commit = AsyncMock(return_value=_TIP)

        with patch(
            'orchestrator.workflow.run_scoped_verification', new=AsyncMock(return_value=_passed()),
        ) as module_verify:
            assert await wf._post_rebase_reverify_passed() is True

        module_verify.assert_awaited_once()
        assert port.calls == []
        [row] = _verify_rows(store)
        assert (row['site'], row['runner']) == ('pre_merge', 'local')
