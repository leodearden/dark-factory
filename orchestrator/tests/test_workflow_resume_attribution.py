"""What a resumed implementer is told after ``run()``'s ESCALATED wait (task 3754).

The wait hands ``run()`` the attributed resolutions it resumes on, refuses to
resume past a record only a human can dispose of, and the resume briefing
names which escalations the resolutions answer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _workflow_helpers import FakeBriefing
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from test_workflow_escalated_steward_stall import (
    _build_workflow,
    _make_advanced_worktree,
    _make_evrl_returner,
    _wire_resolve_callback,
)
from test_workflow_escalated_steward_stall import config as config
from test_workflow_escalated_steward_stall import git_ops as git_ops
from test_workflow_escalated_steward_stall import git_repo as git_repo
from test_workflow_escalated_steward_stall import task_assignment as task_assignment

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import IMPLEMENTER, STEWARD
from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.resume_resolution import MISMATCH_INSTRUCTION, STILL_PENDING_HEADING
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import (
    TaskWorkflow,
    WorkflowOutcome,
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


def _make_background_resolving_steward(
    queue: EscalationQueue, *, then_file: Callable[[], object] = lambda: None,
) -> type:
    """A steward that resolves the pending L0s in the background, as a live one does.

    Same shape as ``test_workflow_merge_gating_strand.py::_make_chained_l2_steward``:
    the resolve runs after a short delay, so it lands after the wait's entry,
    where a production ``TaskSteward.start()`` has only scheduled its loop.
    *then_file* runs right after the resolve with no await between them, so a
    record it files is open when the waiter wakes to the L0 gone.
    """

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):
            self.chain_task: asyncio.Task | None = None

        def set_outcome_channel(self, channel) -> None:
            pass

        def set_wip_probe(self, probe) -> None:
            pass

        async def _chain(self) -> None:
            await asyncio.sleep(0.05)
            for esc in queue.get_by_task(TASK_ID, status='pending', level=0):
                queue.resolve(esc.id, 'resolved by the fake steward', resolved_by='steward')
            then_file()

        async def start(self) -> None:
            self.chain_task = asyncio.create_task(self._chain())

        async def stop(self) -> None:
            if self.chain_task is not None:
                await self.chain_task

    return _FakeSteward


async def _wait_while_steward_resolves(
    workflow: TaskWorkflow, queue: EscalationQueue, *, then_file: Callable[[], object] = lambda: None,
):
    workflow._steward_factory = _make_background_resolving_steward(queue, then_file=then_file)
    _wire_resolve_callback(queue, workflow)
    _submit(queue, 'blocking L0 the steward resolves')
    await workflow._ensure_steward_started()
    return await asyncio.wait_for(workflow._wait_for_resolution(), 10)


async def _wait_while_steward_files(
    workflow: TaskWorkflow, queue: EscalationQueue, *, agent_role: str, severity: str, level: int,
):
    return await _wait_while_steward_resolves(
        workflow, queue,
        then_file=lambda: _submit(
            queue, 'follow-on filed mid-wait',
            severity=severity, level=level, agent_role=agent_role,
        ),
    )


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

    async def test_only_records_resolved_during_this_wait_answer_this_round(
        self,
        config: OrchestratorConfig,
        git_ops: GitOps,
        task_assignment: TaskAssignment,
        tmp_path: Path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        worktree = await _make_advanced_worktree(git_ops, TASK_ID)
        workflow, _ = _build_workflow(config, git_ops, task_assignment, queue, worktree)
        earlier = _submit(queue, 'blocker from an earlier round')
        queue.resolve(earlier.id, 'RESUME with step 20', resolved_by='steward')

        carried = await _wait_while_steward_resolves(workflow, queue)

        (current,) = (e for e in queue.get_by_task(TASK_ID) if e.id != earlier.id)
        assert [(c.escalation_id, c.answered_this_round) for c in carried] == [
            (earlier.id, False),
            (current.id, True),
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
        assert raised.value.block_reason == _NoStewardApplicable([follow_on]).block_reason

    async def test_block_reason_names_each_records_own_tier(
        self, workflow_and_queue: tuple[TaskWorkflow, EscalationQueue],
    ):
        workflow, queue = workflow_and_queue
        open_l1 = _submit(queue, 'hand-off left open elsewhere', level=1, agent_role='reviewer')

        with pytest.raises(_NoStewardApplicable) as raised:
            await _wait_while_steward_files(
                workflow, queue, agent_role='escalation-watcher', severity='blocking', level=2,
            )

        (open_l2,) = queue.get_by_task(TASK_ID, status='pending', level=2)
        assert raised.value.block_reason == (
            f'Gated by open L2 {open_l2.id} and L1 {open_l1.id}; no steward applicable'
        )

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
        assert raised.value.block_reason == _NoStewardApplicable([follow_on]).block_reason

    async def test_info_record_at_level_2_does_not_stop_the_resume(
        self, workflow_and_queue: tuple[TaskWorkflow, EscalationQueue],
    ):
        workflow, queue = workflow_and_queue

        carried = await _wait_while_steward_files(
            workflow, queue, agent_role='escalation-watcher', severity='info', level=2,
        )

        assert [c.resolution for c in carried] == ['resolved by the fake steward']


class _ResumeRecordingBriefing(FakeBriefing):
    def __init__(self) -> None:
        self.resume_calls: list[tuple[str, str]] = []

    async def build_resume_prompt(
        self, task: dict, plan: dict, escalation_summary: str, resolution: str, worktree=None,
        *, scope_drift=None,
    ) -> str:
        self.resume_calls.append((escalation_summary, resolution))
        return await super().build_resume_prompt(
            task, plan, escalation_summary, resolution, worktree, scope_drift=scope_drift,
        )


def _make_prompt_building_invoke(before_building_the_prompt=lambda: None) -> AsyncMock:
    """An ``_invoke`` that builds the resumed implementer's prompt as the real one does.

    The hook runs before the prompt is built, which is where a record filed in
    the window between the wait's end and the prompt build lands.
    """

    async def _invoke(role, prompt_builder, *_args, **_kwargs):
        if role == IMPLEMENTER:
            before_building_the_prompt()
            await prompt_builder()
        return AgentResult(success=True, output='')

    return AsyncMock(side_effect=_invoke)


async def _resume_once(
    config: OrchestratorConfig,
    git_ops: GitOps,
    task_assignment: TaskAssignment,
    queue: EscalationQueue,
    invoke: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_ResumeRecordingBriefing, Escalation]:
    """Drive ``run()`` through one ESCALATED wait and one implementer resume."""
    worktree = await _make_advanced_worktree(git_ops, TASK_ID)
    workflow, _ = _build_workflow(config, git_ops, task_assignment, queue, worktree)
    briefing = _ResumeRecordingBriefing()
    monkeypatch.setattr(workflow, 'briefing', briefing)
    _wire_resolve_callback(queue, workflow)
    workflow._steward_factory = _make_background_resolving_steward(queue)
    evrl, _ = _make_evrl_returner([WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE])
    monkeypatch.setattr(workflow, '_execute_verify_review_loop', evrl)
    monkeypatch.setattr(workflow, '_invoke', invoke)
    answered = _submit(queue, 'verify red at step 19')

    await asyncio.wait_for(workflow.run(), 30)

    return briefing, answered


@pytest.mark.asyncio
class TestResumeBriefingNamesItsEscalations:
    async def test_record_filed_before_the_prompt_is_built_is_named_as_still_pending(
        self,
        config: OrchestratorConfig,
        git_ops: GitOps,
        task_assignment: TaskAssignment,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        late: list[Escalation] = []
        invoke = _make_prompt_building_invoke(
            lambda: late.append(_submit(queue, 'verify still red at step 20')),
        )

        briefing, answered = await _resume_once(
            config, git_ops, task_assignment, queue, invoke, monkeypatch,
        )

        ((issue, resolution),) = briefing.resume_calls
        assert resolution.index(answered.id) < resolution.index('resolved by the fake steward')
        heading = issue.index(STILL_PENDING_HEADING)
        assert issue.index(answered.id) < heading < issue.index(late[0].id)
        assert MISMATCH_INSTRUCTION in issue

    async def test_issue_section_names_the_answered_escalation(
        self,
        config: OrchestratorConfig,
        git_ops: GitOps,
        task_assignment: TaskAssignment,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        briefing, answered = await _resume_once(
            config, git_ops, task_assignment, EscalationQueue(tmp_path / 'queue'),
            _make_prompt_building_invoke(), monkeypatch,
        )

        ((issue, _resolution),) = briefing.resume_calls
        assert answered.id in issue

    async def test_an_open_info_note_is_not_named_as_awaiting_an_answer(
        self,
        config: OrchestratorConfig,
        git_ops: GitOps,
        task_assignment: TaskAssignment,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        notes: list[Escalation] = []
        invoke = _make_prompt_building_invoke(
            lambda: notes.append(_submit(queue, 'fyi: flaky host', severity='info', level=1)),
        )

        briefing, answered = await _resume_once(
            config, git_ops, task_assignment, queue, invoke, monkeypatch,
        )

        ((issue, _resolution),) = briefing.resume_calls
        assert answered.id in issue
        assert notes[0].id not in issue
        assert STILL_PENDING_HEADING not in issue
        assert MISMATCH_INSTRUCTION not in issue
