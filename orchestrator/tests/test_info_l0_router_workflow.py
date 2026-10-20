"""The info-L0 router's workflow call sites (plans/info-l0-disposition-router-prd.md).

D10: the five blanket level-0 dismissals a workflow makes on its way out leave
an info note the router owns pending for it, and dismiss everything else
exactly as before.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from _workflow_helpers import AgentStub, _build_workflow_with_escalation
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig, TimeoutsConfig, UnblockAutoConfig
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome
from orchestrator.workflow_types import StewardInterrupted, StewardReescalatedL1

TASK_ID = '42'


def _config(project_root: Path, *, router_enabled: bool) -> OrchestratorConfig:
    """Steward waits short enough to expire inside a test; no dry-run unblock."""
    return OrchestratorConfig(
        project_root=project_root,
        max_concurrent_tasks=1,
        steward_completion_timeout=0.25,
        timeouts=TimeoutsConfig(steward=0.25),
        unblock_auto=UnblockAutoConfig(enabled=False),
        info_l0_router_enabled=router_enabled,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id=TASK_ID,
        task={
            'id': TASK_ID, 'title': 'X', 'description': 'Y', 'status': 'pending',
            'metadata': {'files': ['lib']}, 'dependencies': [],
        },
        modules=['lib'],
    )


def _workflow(tmp_path: Path, *, router_enabled: bool) -> tuple[TaskWorkflow, EscalationQueue]:
    config = _config(tmp_path, router_enabled=router_enabled)
    workflow, _scheduler, queue = _build_workflow_with_escalation(
        config, GitOps(config.git, config.project_root), _assignment(), AgentStub(), tmp_path,
        spawn_merge_worker=False,
    )
    return workflow, queue


def _submit(queue: EscalationQueue, *, severity: str, level: int = 0) -> Escalation:
    esc = Escalation(
        id=queue.make_id(TASK_ID), task_id=TASK_ID, agent_role='implementer',
        severity=severity, category='task_failure' if severity != 'info' else 'design_concern',
        summary=f'{severity} L{level} note', detail=f'{severity} detail',
        suggested_action='create_followup_task', level=level,
    )
    queue.submit(esc)
    return esc


def _status(queue: EscalationQueue, esc: Escalation) -> str:
    current = queue.get(esc.id)
    assert current is not None
    return current.status


def _wip_interrupting_steward() -> type:
    """A steward that publishes a wip-present interruption as soon as it starts."""

    class _Steward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002 — factory signature
            self._channel: asyncio.Queue | None = None

        def set_outcome_channel(self, channel: asyncio.Queue) -> None:
            self._channel = channel

        def set_wip_probe(self, probe) -> None:  # noqa: ARG002 — steward interface
            pass

        async def start(self) -> None:
            assert self._channel is not None
            self._channel.put_nowait(StewardInterrupted('attempt_cap', wip_commits_present=True))

        async def stop(self) -> None:
            pass

    return _Steward


async def _blocked_fall_through(workflow: TaskWorkflow, queue: EscalationQueue, tmp_path: Path) -> None:  # noqa: ARG001
    assert await workflow._mark_blocked('synthetic failure') == WorkflowOutcome.BLOCKED


async def _skip_escalation_sweep(workflow: TaskWorkflow, queue: EscalationQueue, tmp_path: Path) -> None:  # noqa: ARG001
    await workflow._mark_blocked('synthetic failure', skip_escalation=True)


async def _wip_interruption(workflow: TaskWorkflow, queue: EscalationQueue, tmp_path: Path) -> None:  # noqa: ARG001
    worktree = tmp_path / 'wt'
    worktree.mkdir()
    workflow.worktree = worktree
    workflow._steward_factory = _wip_interrupting_steward()
    assert await workflow._mark_blocked('synthetic failure') == WorkflowOutcome.REQUEUED


async def _open_l1_override(workflow: TaskWorkflow, queue: EscalationQueue, tmp_path: Path) -> None:  # noqa: ARG001
    _submit(queue, severity='blocking', level=1)
    channel: asyncio.Queue = asyncio.Queue()
    channel.put_nowait(StewardInterrupted('timeout', wip_commits_present=True))
    workflow._steward_outcome_channel = channel
    assert isinstance(await workflow._await_steward_completion(), StewardReescalatedL1)


async def _steward_wait_timeout(workflow: TaskWorkflow, queue: EscalationQueue, tmp_path: Path) -> None:  # noqa: ARG001
    await asyncio.wait_for(workflow._wait_for_resolution(), 10)
    assert workflow._steward_wait_expired


DISMISSAL_SITES = {
    'blocked-fall-through': _blocked_fall_through,
    'skip-escalation-sweep': _skip_escalation_sweep,
    'wip-interruption': _wip_interruption,
    'open-l1-override': _open_l1_override,
    'steward-wait-timeout': _steward_wait_timeout,
}


@pytest.mark.asyncio
class TestBlanketDismissalsSpareRouterOwnedNotes:
    """D10 (boundary #14 is the BLOCKED fall-through)."""

    @pytest.mark.parametrize('site', DISMISSAL_SITES.values(), ids=DISMISSAL_SITES.keys())
    async def test_the_info_note_survives_and_the_blocking_l0_is_dismissed(
        self, tmp_path: Path, site,
    ):
        workflow, queue = _workflow(tmp_path, router_enabled=True)
        blocking = _submit(queue, severity='blocking')
        info = _submit(queue, severity='info')

        await site(workflow, queue, tmp_path)

        assert _status(queue, blocking) == 'dismissed'
        assert _status(queue, info) == 'pending'

    @pytest.mark.parametrize('site', DISMISSAL_SITES.values(), ids=DISMISSAL_SITES.keys())
    async def test_with_the_router_disabled_the_info_note_is_dismissed_too(
        self, tmp_path: Path, site,
    ):
        workflow, queue = _workflow(tmp_path, router_enabled=False)
        blocking = _submit(queue, severity='blocking')
        info = _submit(queue, severity='info')

        await site(workflow, queue, tmp_path)

        assert _status(queue, blocking) == 'dismissed'
        assert _status(queue, info) == 'dismissed'
