"""A real workflow that wedges after DONE gives back its slot and locks at hard-cancel (task 6660).

Reify tasks 4289 and 6151 logged ``Task DONE``, ignored three soft cancels and
the hard-cancel, and held a workflow slot and their module locks until the next
restart: an entry of the terminal cleanup was stuck in an await that absorbed
the single ``cancel()`` it was sent. This drives a real ``TaskWorkflow`` to DONE
through ``Harness._run_slot`` with a claimant-heartbeat write that survives one
cancel (the shape of an MCP call whose connection strands a cancel), so
``stop_claimant_heartbeat`` wedges exactly that way, then hard-cancels it as the
terminal-status watcher does.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT, wait_responsive
from _workflow_helpers import (
    AgentStub,
    FakeScheduler,
    _build_harness,
    _build_workflow,
    _derive_meta_root_like_production,
    _init_repo,
)

from orchestrator.config import GitConfig, OrchestratorConfig, SandboxConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps
from orchestrator.harness import TaskReport
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import WorkflowState

# Autouse; named here so the agent-side doubles resolve the meta_root production does.
pytestmark = pytest.mark.usefixtures(_derive_meta_root_like_production.__name__)

TID = '42'
GRACE = 0.5
BARRIER = 45.0
WEDGE_ALONE_OR_WITH_THE_GIT_CLEANUP = (
    ['stop_claimant_heartbeat'],
    ['stop_claimant_heartbeat', 'cleanup_done_worktree'],
)


class _HeartbeatSurvivesOneCancel(FakeScheduler):
    """A claimant-heartbeat refresh that never returns and absorbs the first cancel."""

    def __init__(self) -> None:
        super().__init__()
        self.cancel_absorbed = asyncio.Event()

    async def set_task_claimant(
        self,
        task_id: str,
        *,
        claimant_run_id: str | None = None,
        heartbeat_at: str | None = None,
    ) -> None:
        if claimant_run_id is not None or heartbeat_at is None:
            await super().set_task_claimant(task_id, claimant_run_id=claimant_run_id, heartbeat_at=heartbeat_at)
            return
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancel_absorbed.set()
            await asyncio.Event().wait()


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=1,
        git=GitConfig(main_branch='main', branch_prefix='task/', remote='origin', worktree_dir='.worktrees'),
        sandbox=SandboxConfig(enabled=False),
        claimant_heartbeat_interval_secs=0.01,
        terminal_cleanup_cancel_grace_secs=GRACE,
    )


@dataclass(frozen=True)
class _Unwound:
    slot: asyncio.Task[TaskReport | None]
    sem: asyncio.Semaphore
    scheduler: Scheduler
    events: MagicMock


class _WorktreeCleanupNeverReturns(GitOps):
    """A worktree removal that ends only when cancelled, so the entry after the wedge overruns its grace."""

    async def cleanup_worktree(self, worktree: Path, branch: str) -> None:
        await asyncio.Event().wait()


async def _hard_cancel_a_slot_wedged_after_done(
    config: OrchestratorConfig, monkeypatch: pytest.MonkeyPatch, git_ops: GitOps,
) -> _Unwound:
    assignment = TaskAssignment(
        task_id=TID,
        task={
            'id': TID,
            'title': 'Add farewell function',
            'description': 'Add a farewell(name) function to lib.py with tests',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )
    stub = AgentStub()
    monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
    monkeypatch.setattr(
        'orchestrator.workflow.run_scoped_verification',
        AsyncMock(return_value=VerifyResult(
            passed=True, test_output='OK', lint_output='', type_output='', summary='All checks passed',
        )),
    )
    workflow, fake_scheduler = _build_workflow(config, git_ops, assignment, stub)
    wedging_scheduler = _HeartbeatSurvivesOneCancel()
    monkeypatch.setattr(workflow, 'scheduler', wedging_scheduler)

    harness = _build_harness(config)
    scheduler = Scheduler(config)
    monkeypatch.setattr(scheduler, 'dispatch_tool', AsyncMock(return_value={}))
    harness.scheduler = scheduler
    events = MagicMock()
    harness.event_store = events
    workflow.event_store = events
    assert scheduler.lock_table.try_acquire(TID, ['lib'])
    sem = asyncio.Semaphore(0)

    with patch('orchestrator.harness.build_workflow', return_value=workflow):
        slot = asyncio.create_task(harness._run_slot(assignment, sem), name=f'workflow-{TID}')
        await wait_responsive(
            wedging_scheduler.cancel_absorbed.wait(), timeout=BARRIER,
            label='DONE-phase cleanup wedges on the heartbeat',
        )
        assert workflow.state == WorkflowState.DONE
        assert not slot.done(), 'terminal cleanup should be wedged on the heartbeat'

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 3 * GRACE + 5.0
        while not slot.done() and loop.time() < deadline:
            harness.hard_cancel_workflow(TID, reason='terminal_status_cancel')
            await asyncio.wait({slot}, timeout=GRACE / 5)

    return _Unwound(slot=slot, sem=sem, scheduler=scheduler, events=events)


def _assert_released_with_the_wedge_abandoned(unwound: _Unwound) -> list[str]:
    """The wedge is always abandoned; cleanup_done_worktree, the only later entry here that waits on real git, may legitimately overrun the test's short grace too.

    The other entries are instant in this harness: no steward, no escalation queue, no warm-lane pool so release_lane returns at once, and a synchronous cleanup_config_dir.
    """
    assert unwound.slot.done(), 'the hard-cancelled slot never unwound'
    assert not unwound.slot.cancelled()
    assert not unwound.sem.locked(), 'the slot semaphore was not released'
    assert not unwound.scheduler.lock_table.is_held(TID), 'the module locks were not released'
    emitted = [c.args[0] for c in unwound.events.emit.call_args_list if c.args]
    assert EventType.task_completed in emitted
    abandoned = [
        c.kwargs['data']['entries'] for c in unwound.events.emit.call_args_list
        if c.args and c.args[0] is EventType.workflow_cleanup_abandoned
    ]
    assert len(abandoned) == 1
    assert abandoned[0] in WEDGE_ALONE_OR_WITH_THE_GIT_CLEANUP
    return abandoned[0]


@pytest.mark.asyncio
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
async def test_real_workflow_blocking_after_done_releases_slot_and_locks_at_hard_cancel(
    config: OrchestratorConfig, monkeypatch: pytest.MonkeyPatch,
):
    unwound = await _hard_cancel_a_slot_wedged_after_done(
        config, monkeypatch, GitOps(config.git, config.project_root),
    )
    _assert_released_with_the_wedge_abandoned(unwound)


@pytest.mark.asyncio
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
async def test_a_cleanup_entry_that_also_overruns_its_grace_is_abandoned_with_the_wedge(
    config: OrchestratorConfig, monkeypatch: pytest.MonkeyPatch,
):
    unwound = await _hard_cancel_a_slot_wedged_after_done(
        config, monkeypatch, _WorktreeCleanupNeverReturns(config.git, config.project_root),
    )
    assert _assert_released_with_the_wedge_abandoned(unwound) == ['stop_claimant_heartbeat', 'cleanup_done_worktree']
