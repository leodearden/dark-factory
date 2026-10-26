"""What a resumed implementer is told after ``run()``'s ESCALATED wait (task 3754).

The wait hands ``run()`` the attributed resolutions it resumes on, refuses to
resume past a record only a human can dispose of, and the resume briefing
names which escalations the resolutions answer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from test_workflow_escalated_steward_stall import (
    _build_workflow,
    _make_advanced_worktree,
    _wire_resolve_callback,
)
from test_workflow_escalated_steward_stall import config as config
from test_workflow_escalated_steward_stall import git_ops as git_ops
from test_workflow_escalated_steward_stall import git_repo as git_repo
from test_workflow_escalated_steward_stall import task_assignment as task_assignment

from orchestrator.agents.roles import STEWARD
from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import (
    TaskWorkflow,
    _HumanHandoff,
    _NoStewardApplicable,
    _StewardReescalated,
)

TASK_ID = '42'


def _submit(
    queue: EscalationQueue,
    summary: str,
    *,
    severity: str = 'blocking',
    level: int = 0,
    agent_role: str = 'implementer',
) -> Escalation:
    esc = Escalation(
        id=queue.make_id(TASK_ID),
        task_id=TASK_ID,
        agent_role=agent_role,
        severity=severity,
        category='task_failure',
        summary=summary,
        detail=f'detail for {summary}',
        level=level,
    )
    queue.submit(esc)
    return esc


def _make_follow_on_filing_steward(
    queue: EscalationQueue, *, agent_role: str, severity: str, level: int,
) -> type:
    """A steward that resolves the pending L0 and files ONE follow-on record.

    Same shape as ``test_workflow_merge_gating_strand.py::_make_chained_l2_steward``:
    the chain runs in the background after a short delay, so the follow-on lands
    after the wait's entry gate, and resolve-then-submit has no await between
    them, so the waiter wakes to the L0 gone and the follow-on open.
    """

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self.chain_task: asyncio.Task | None = None

        def set_outcome_channel(self, channel) -> None:
            pass

        def set_wip_probe(self, probe) -> None:
            pass

        async def _chain(self) -> None:
            await asyncio.sleep(0.05)
            for esc in queue.get_by_task(TASK_ID, status='pending', level=0):
                queue.resolve(esc.id, 'resolved by the fake steward', resolved_by='steward')
            _submit(
                queue, 'follow-on filed mid-wait',
                severity=severity, level=level, agent_role=agent_role,
            )

        async def start(self) -> None:
            self.chain_task = asyncio.create_task(self._chain())

        async def stop(self) -> None:
            if self.chain_task is not None:
                await self.chain_task

    return _FakeSteward


async def _wait_while_steward_files(
    workflow: TaskWorkflow, queue: EscalationQueue, *, agent_role: str, severity: str, level: int,
):
    workflow._steward_factory = _make_follow_on_filing_steward(
        queue, agent_role=agent_role, severity=severity, level=level,
    )
    _wire_resolve_callback(queue, workflow)
    _submit(queue, 'blocking L0 the steward resolves')
    await workflow._ensure_steward_started()
    return await asyncio.wait_for(workflow._wait_for_resolution(), 10)


def _the_follow_on(queue: EscalationQueue) -> Escalation:
    (follow_on,) = queue.get_by_task(TASK_ID, status='pending')
    return follow_on


@pytest.mark.asyncio
class TestWaitReturnsTheResolutionsItResumesOn:
    async def test_returns_resolved_records_in_resolution_order(
        self,
        config: OrchestratorConfig,
        git_ops: GitOps,
        task_assignment: TaskAssignment,
        tmp_path: Path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        worktree = await _make_advanced_worktree(git_ops, TASK_ID)
        workflow, _ = _build_workflow(config, git_ops, task_assignment, queue, worktree)
        first = _submit(queue, 'first blocker')
        second = _submit(queue, 'second blocker')
        dismissed = _submit(queue, 'third blocker')
        queue.resolve(second.id, 'answer to second', resolved_by='steward')
        queue.resolve(first.id, 'answer to first', resolved_by='interactive')
        queue.resolve(
            dismissed.id, 'Auto-dismissed: superseded', dismiss=True,
            resolved_by='auto-dismissed',
        )
        _submit(queue, 'informational note', severity='info', level=1)

        carried = await asyncio.wait_for(workflow._wait_for_resolution(), 10)

        assert isinstance(carried, tuple)
        assert [(c.escalation_id, c.resolution) for c in carried] == [
            (second.id, 'answer to second'),
            (first.id, 'answer to first'),
        ]

    async def test_without_a_queue_carries_nothing(
        self,
        config: OrchestratorConfig,
        git_ops: GitOps,
        task_assignment: TaskAssignment,
        tmp_path: Path,
    ):
        workflow, _ = _build_workflow(
            config, git_ops, task_assignment, EscalationQueue(tmp_path / 'queue'), tmp_path,
        )
        workflow.escalation_queue = None

        assert await workflow._wait_for_resolution() == ()


@pytest.mark.asyncio
class TestTailGuardCoversRecordsFiledDuringTheWait:
    @pytest_asyncio.fixture
    async def workflow_and_queue(
        self,
        config: OrchestratorConfig,
        git_ops: GitOps,
        task_assignment: TaskAssignment,
        tmp_path: Path,
    ) -> tuple[TaskWorkflow, EscalationQueue]:
        queue = EscalationQueue(tmp_path / 'queue')
        worktree = await _make_advanced_worktree(git_ops, TASK_ID)
        workflow, _ = _build_workflow(config, git_ops, task_assignment, queue, worktree)
        return workflow, queue

    async def test_steward_filed_born_at_l2_is_a_steward_reescalation(
        self, workflow_and_queue: tuple[TaskWorkflow, EscalationQueue],
    ):
        workflow, queue = workflow_and_queue

        with pytest.raises(_StewardReescalated) as raised:
            await _wait_while_steward_files(
                workflow, queue, agent_role=STEWARD.name, severity='critical', level=2,
            )

        assert [e.id for e in raised.value.escalations] == [_the_follow_on(queue).id]

    async def test_level_2_queue_handoff_from_elsewhere_needs_a_human(
        self, workflow_and_queue: tuple[TaskWorkflow, EscalationQueue],
    ):
        workflow, queue = workflow_and_queue

        with pytest.raises(_NoStewardApplicable) as raised:
            await _wait_while_steward_files(
                workflow, queue, agent_role='escalation-watcher', severity='blocking', level=2,
            )

        follow_on = _the_follow_on(queue)
        assert raised.value.block_reason == _NoStewardApplicable([follow_on], 'L2').block_reason

    async def test_born_at_l2_severity_at_level_1_needs_a_human(
        self, workflow_and_queue: tuple[TaskWorkflow, EscalationQueue],
    ):
        workflow, queue = workflow_and_queue

        with pytest.raises(_HumanHandoff) as raised:
            await _wait_while_steward_files(
                workflow, queue, agent_role='escalation-watcher', severity='critical', level=1,
            )

        follow_on = _the_follow_on(queue)
        assert [e.id for e in raised.value.escalations] == [follow_on.id]
        assert raised.value.block_reason == _NoStewardApplicable([follow_on], 'L2').block_reason

    async def test_info_record_at_level_2_does_not_stop_the_resume(
        self, workflow_and_queue: tuple[TaskWorkflow, EscalationQueue],
    ):
        workflow, queue = workflow_and_queue

        carried = await _wait_while_steward_files(
            workflow, queue, agent_role='escalation-watcher', severity='info', level=2,
        )

        assert [c.resolution for c in carried] == ['resolved by the fake steward']
