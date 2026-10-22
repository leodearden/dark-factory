"""``_mark_blocked`` dismisses a pending L0 only while an open handoff covers it.

``skip_escalation`` means the caller has already filed the task's
human-facing level>=1 record, so ``_mark_blocked`` files nothing itself.
A pending L0 is dismissed on the way out only when the open level>=1
records pin the blocked task (``recovery_pins.records_pin_blocked_recovery``,
the stranded sweep's own veto); otherwise it stays pending for the
orphan-L0 reaper, so a finding is never dismissed into nowhere.
``escalate_to_human`` cannot be combined with ``skip_escalation``.

Driven against a real ``EscalationQueue`` and a real ``OrchestratorConfig``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig, UnblockAutoConfig
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow
from orchestrator.workflow_types import WorkflowOutcome

TASK_ID = '7096'


def _make_workflow(tmp_path: Path) -> tuple[TaskWorkflow, EscalationQueue, FakeScheduler]:
    assignment = TaskAssignment(
        task_id=TASK_ID,
        task={
            'id': TASK_ID, 'title': 'L0 coverage', 'description': '',
            'status': 'in-progress', 'metadata': {}, 'dependencies': [],
        },
        modules=[],
    )
    config = OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main', branch_prefix='task/',
            remote='origin', worktree_dir='.worktrees',
        ),
        unblock_auto=UnblockAutoConfig(enabled=False),
    )
    queue = EscalationQueue(tmp_path / 'escalations')
    scheduler = FakeScheduler()
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=scheduler,
        briefing=FakeBriefing(),    # type: ignore[arg-type]
        mcp=FakeMcp(),              # type: ignore[arg-type]
        escalation_queue=queue,
    )
    workflow.worktree = tmp_path / 'wt'
    return workflow, queue, scheduler


def _submit(
    queue: EscalationQueue, *, level: int, severity: str = 'blocking',
) -> Escalation:
    record = Escalation(
        id=queue.make_id(TASK_ID),
        task_id=TASK_ID,
        agent_role='architect' if level == 0 else 'orchestrator-workflow',
        severity=severity,
        category='design_concern',
        summary=f'level-{level} {severity} finding',
        detail='what was measured',
        level=level,
    )
    queue.submit(record)
    return record


def _status(queue: EscalationQueue, record: Escalation) -> str:
    stored = queue.get(record.id)
    assert stored is not None
    return stored.status


def _pending(queue: EscalationQueue) -> list[Escalation]:
    return queue.get_by_task(TASK_ID, status='pending')


@pytest.mark.asyncio
async def test_escalate_to_human_with_skip_escalation_is_refused(tmp_path: Path):
    workflow, queue, scheduler = _make_workflow(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        await workflow._mark_blocked(
            'r', escalate_to_human=True, skip_escalation=True,
        )

    assert 'escalate_to_human' in str(excinfo.value)
    assert 'skip_escalation' in str(excinfo.value)
    assert scheduler.statuses.get(TASK_ID, []) == []
    assert queue.get_by_task(TASK_ID) == []


@pytest.mark.asyncio
async def test_uncovered_l0_stays_pending_and_is_named(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    workflow, queue, scheduler = _make_workflow(tmp_path)
    l0 = _submit(queue, level=0)

    with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
        outcome = await workflow._mark_blocked('r', skip_escalation=True)

    assert outcome == WorkflowOutcome.BLOCKED
    assert scheduler.statuses[TASK_ID][-1] == 'blocked'
    assert _status(queue, l0) == 'pending'
    assert any(
        r.levelno == logging.WARNING and l0.id in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_info_l1_does_not_cover_an_l0(tmp_path: Path):
    workflow, queue, _ = _make_workflow(tmp_path)
    l0 = _submit(queue, level=0)
    _submit(queue, level=1, severity='info')

    await workflow._mark_blocked('r', skip_escalation=True)

    assert _status(queue, l0) == 'pending'


@pytest.mark.asyncio
async def test_blocking_l1_covers_an_l0(tmp_path: Path):
    workflow, queue, _ = _make_workflow(tmp_path)
    l0 = _submit(queue, level=0)
    l1 = _submit(queue, level=1)

    await workflow._mark_blocked('r', skip_escalation=True)

    dismissed = queue.get(l0.id)
    assert dismissed is not None
    assert dismissed.status == 'dismissed'
    assert dismissed.resolution is not None and l1.id in dismissed.resolution
    assert queue.get(l1.id) == l1


@pytest.mark.asyncio
async def test_open_l2_handoff_covers_an_l0(tmp_path: Path):
    workflow, queue, _ = _make_workflow(tmp_path)
    l0 = _submit(queue, level=0)
    _submit(queue, level=2)

    await workflow._mark_blocked('r', skip_escalation=True)

    assert _status(queue, l0) == 'dismissed'


@pytest.mark.asyncio
async def test_unskipped_block_files_its_own_l1_and_dismisses_its_l0(tmp_path: Path):
    workflow, queue, _ = _make_workflow(tmp_path)

    outcome = await workflow._mark_blocked('r')

    assert outcome == WorkflowOutcome.BLOCKED
    pending = _pending(queue)
    assert [e.level for e in pending] == [1], [(e.id, e.level) for e in pending]
    filed = queue.get_by_task(TASK_ID, level=0)
    assert [e.status for e in filed] == ['dismissed']
