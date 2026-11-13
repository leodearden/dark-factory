"""``_mark_blocked``'s steward wait is scoped to the escalation it filed (task 5563).

The steward works pending level-0 escalations oldest-first.  When it is still
busy with an older L0 A as ``_mark_blocked`` files B, A's outcome lands on the
shared channel first.  Before task 5563 that outcome ended B's wait, and
``_requeue()`` fired while B was still pending and untouched (reify 6615).

The invariant pinned here, through the real ``_mark_blocked`` with a real
``EscalationQueue`` and a real worktree: an outcome published for escalation X
never resolves the wait for escalation Y.  Only B's own outcome, B's own
record found closed by a non-sweep resolver, the grace deadline, or a cancel
ends it.
"""

from __future__ import annotations

import asyncio
import contextlib
from enum import Enum, auto
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from _workflow_helpers import (
    GIT_REPO_SEED,
    AgentStub,
    FakeScheduler,
    _build_workflow_with_escalation,
    _config,
    _make_git_ops,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import UnblockAutoConfig
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome
from orchestrator.workflow_types import PublishedStewardOutcome, StewardResolved

TASK_ID = '42'
FOLD_CATEGORY = 'preexisting_main_break'
FOLD_FINGERPRINT = 'fp-5563'


class NewL0Handling(Enum):
    """What the fake steward does about the L0 ``_mark_blocked`` files while
    it is still working the older one."""

    UNTOUCHED = auto()
    # Closed after the older L0, with its own envelope.
    RESOLVED_AFTER_OLDER = auto()
    # Closed alongside the older L0; only the older L0's envelope is published.
    CLOSED_WITH_OLDER = auto()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo', GIT_REPO_SEED)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id=TASK_ID,
        task={
            'id': TASK_ID,
            'title': 'X',
            'description': 'Y',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


def _submit_l0(
    queue: EscalationQueue,
    *,
    category: str,
    summary: str,
    dedupe_fingerprint: str | None = None,
) -> Escalation:
    esc = Escalation(
        id=queue.make_id(TASK_ID),
        task_id=TASK_ID,
        agent_role='implementer',
        severity='blocking',
        category=category,
        summary=summary,
        detail=summary,
        dedupe_fingerprint=dedupe_fingerprint,
    )
    queue.submit(esc)
    return esc


def _make_busy_on_older_l0_steward(
    queue: EscalationQueue,
    older_id: str,
    *,
    new_l0: NewL0Handling,
    new_l0_closer: str = 'fake-steward',
    new_l0_dismiss: bool = False,
) -> type:
    """A steward class still working the OLDER L0 when ``_mark_blocked`` waits.

    It resolves *older_id* and publishes that outcome tagged with *older_id*,
    as ``TaskSteward`` does.  *new_l0* says what it does about every other
    pending L0 (see :class:`NewL0Handling`); whenever it closes one, it calls
    ``resolve()`` with *new_l0_closer* as ``resolved_by`` and *new_l0_dismiss*
    as ``dismiss``.

    Except for ``RESOLVED_AFTER_OLDER``, all of that happens synchronously in
    ``start()``, which ``_mark_blocked`` awaits after filing the new L0 and
    before its wait begins, so the older L0's envelope is on the channel when
    the wait starts — by construction, not by timing.

    The class is itself the ``steward_factory``; every built instance lands in
    ``instances``, so teardown can stop them without reaching into the workflow.
    """

    class _FakeSteward:
        instances: list[_FakeSteward] = []
        published: list[str] = []

        def __init__(self, wt_path: Path, cfg_dir: Path | None) -> None:
            self._outcome_channel: asyncio.Queue[PublishedStewardOutcome] | None = None
            self._work: asyncio.Task | None = None
            _FakeSteward.instances.append(self)

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            pass

        def _publish(self, esc_id: str, text: str) -> None:
            assert self._outcome_channel is not None
            self._outcome_channel.put_nowait(
                PublishedStewardOutcome(esc_id, StewardResolved(text)),
            )
            _FakeSteward.published.append(esc_id)

        def _close_other_pending(self) -> list[str]:
            closed = [
                esc.id
                for esc in queue.get_by_task(TASK_ID, status='pending', level=0)
                if esc.id != older_id
            ]
            for esc_id in closed:
                queue.resolve(
                    esc_id, f'fixed {esc_id}',
                    dismiss=new_l0_dismiss, resolved_by=new_l0_closer,
                )
            return closed

        def _finish_older(self) -> None:
            if new_l0 is NewL0Handling.CLOSED_WITH_OLDER:
                self._close_other_pending()
            queue.resolve(older_id, 'older fixed', resolved_by='fake-steward')
            self._publish(older_id, 'older fixed')

        async def _finish_older_then_new(self) -> None:
            await asyncio.sleep(0.05)
            self._finish_older()
            await asyncio.sleep(0.2)
            for esc_id in self._close_other_pending():
                self._publish(esc_id, f'fixed {esc_id}')

        async def start(self) -> None:
            if new_l0 is NewL0Handling.RESOLVED_AFTER_OLDER:
                self._work = asyncio.create_task(self._finish_older_then_new())
            else:
                self._finish_older()

        async def stop(self) -> None:
            if self._work is not None:
                self._work.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._work

    return _FakeSteward


async def _build_workflow(
    git_repo: Path,
    assignment: TaskAssignment,
    queue: EscalationQueue,
    steward_factory: type,
    tmp_path: Path,
    *,
    completion_timeout: float,
) -> tuple[TaskWorkflow, FakeScheduler]:
    config = _config(
        git_repo,
        steward_completion_timeout=completion_timeout,
        unblock_auto=UnblockAutoConfig(enabled=False),
    )
    git_ops = _make_git_ops(git_repo)
    workflow, scheduler, _queue = _build_workflow_with_escalation(
        config, git_ops, assignment, AgentStub(), tmp_path,
        spawn_merge_worker=False,
        steward_factory=steward_factory,
        queue=queue,
    )
    workflow.worktree = (await git_ops.create_worktree(TASK_ID)).path
    assert await workflow._worktree_has_wip_commits() is False, (
        'precondition: a fresh, clean worktree must derive no WIP, or the '
        'grace-timeout outcome would take the task-2060 resume-plan branch'
    )
    return workflow, scheduler


def _new_l0_ids(queue: EscalationQueue, *known: str) -> list[str]:
    """Ids of every level-0 record for the task, pending or archived, not in *known*."""
    return [
        esc.id for esc in queue.get_by_task(TASK_ID, level=0) if esc.id not in known
    ]


async def _stop_all(steward_cls: type) -> None:
    for steward in steward_cls.instances:
        await steward.stop()


@pytest.mark.asyncio
class TestMarkBlockedWaitIsScopedToItsOwnEscalation:

    @pytest.mark.parametrize('merge_phase', [False, True])
    async def test_older_l0_resolution_does_not_requeue_the_new_l0_wait(
        self, git_repo, task_assignment, tmp_path, merge_phase,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        older = _submit_l0(queue, category='design_concern', summary='older blocker')
        steward_cls = _make_busy_on_older_l0_steward(
            queue, older.id, new_l0=NewL0Handling.UNTOUCHED,
        )
        grace = 0.5
        workflow, scheduler = await _build_workflow(
            git_repo, task_assignment, queue, steward_cls, tmp_path,
            completion_timeout=grace,
        )

        try:
            outcome = await asyncio.wait_for(
                workflow._mark_blocked(
                    'Suffix item needs rebase onto frozen-prefix tip',
                    merge_phase=merge_phase,
                ),
                10,
            )
        finally:
            await _stop_all(steward_cls)

        assert steward_cls.published == [older.id], (
            "precondition: the older L0's outcome must be on the channel when "
            'the wait begins, or BLOCKED proves nothing about scoping'
        )
        assert outcome == WorkflowOutcome.BLOCKED, (
            "the older L0's StewardResolved must not end the wait for the new L0"
        )
        assert 'pending' not in scheduler.statuses.get(TASK_ID, [])
        [new_id] = _new_l0_ids(queue, older.id)
        new = queue.get(new_id)
        assert new is not None
        assert new.status == 'dismissed'
        assert new.resolved_by == 'auto-dismissed', (
            'the new L0 was never touched by the steward; only the BLOCKED '
            f'fall-through may close it (resolved_by={new.resolved_by!r})'
        )

    async def test_new_l0_own_resolution_requeues_after_it_is_resolved(
        self, git_repo, task_assignment, tmp_path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        older = _submit_l0(queue, category='design_concern', summary='older blocker')
        steward_cls = _make_busy_on_older_l0_steward(
            queue, older.id, new_l0=NewL0Handling.RESOLVED_AFTER_OLDER,
        )
        workflow, _scheduler = await _build_workflow(
            git_repo, task_assignment, queue, steward_cls, tmp_path, completion_timeout=5.0,
        )

        try:
            outcome = await asyncio.wait_for(
                workflow._mark_blocked('Suffix item needs rebase onto frozen-prefix tip'),
                10,
            )
            [new_id] = _new_l0_ids(queue, older.id)
            new = queue.get(new_id)
            assert new is not None
            new_status_at_return = new.status
        finally:
            await _stop_all(steward_cls)

        assert outcome == WorkflowOutcome.REQUEUED
        assert new_status_at_return == 'resolved', (
            'the requeue must follow the new L0 being resolved, not the older one'
        )

    async def test_same_task_dedupe_fold_scopes_the_wait_to_the_parent(
        self, git_repo, task_assignment, tmp_path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        older = _submit_l0(queue, category='design_concern', summary='older blocker')
        parent = _submit_l0(
            queue, category=FOLD_CATEGORY, summary='inherited break',
            dedupe_fingerprint=FOLD_FINGERPRINT,
        )
        steward_cls = _make_busy_on_older_l0_steward(
            queue, older.id, new_l0=NewL0Handling.RESOLVED_AFTER_OLDER,
        )
        workflow, _scheduler = await _build_workflow(
            git_repo, task_assignment, queue, steward_cls, tmp_path, completion_timeout=5.0,
        )

        try:
            outcome = await asyncio.wait_for(
                workflow._mark_blocked(
                    'inherited break',
                    category=FOLD_CATEGORY,
                    dedupe_fingerprint=FOLD_FINGERPRINT,
                ),
                10,
            )
            parent_record = queue.get(parent.id)
            assert parent_record is not None
            parent_status_at_return = parent_record.status
        finally:
            await _stop_all(steward_cls)

        assert _new_l0_ids(queue, older.id, parent.id) == [], (
            'precondition: the filing must have folded into the parent'
        )
        assert outcome == WorkflowOutcome.REQUEUED
        assert parent_status_at_return == 'resolved', (
            'the wait must be scoped to the record the filing landed in (the '
            'parent), so only its resolution may requeue'
        )

    async def test_new_l0_closed_while_steward_handles_older_requeues_before_the_timeout(
        self, git_repo, task_assignment, tmp_path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        older = _submit_l0(queue, category='design_concern', summary='older blocker')
        steward_cls = _make_busy_on_older_l0_steward(
            queue, older.id,
            new_l0=NewL0Handling.CLOSED_WITH_OLDER,
            new_l0_closer='claude-task-42-steward',
        )
        workflow, _scheduler = await _build_workflow(
            git_repo, task_assignment, queue, steward_cls, tmp_path, completion_timeout=30.0,
        )

        try:
            outcome = await asyncio.wait_for(
                workflow._mark_blocked('Suffix item needs rebase onto frozen-prefix tip'),
                5,
            )
        finally:
            await _stop_all(steward_cls)

        assert outcome == WorkflowOutcome.REQUEUED, (
            'the steward closed the new L0 while handling the older one; its '
            'closed record must settle the wait without an envelope of its own'
        )
        [new_id] = _new_l0_ids(queue, older.id)
        new = queue.get(new_id)
        assert new is not None
        assert new.status == 'resolved'
        assert new.resolved_by == 'claude-task-42-steward', (
            'no fall-through dismissal may overwrite the steward\'s close '
            f'(resolved_by={new.resolved_by!r})'
        )
        assert queue.get_by_task(TASK_ID, level=1) == [], (
            'no human-facing L1 for an escalation the steward already resolved'
        )

    async def test_new_l0_swept_while_steward_handles_older_does_not_requeue(
        self, git_repo, task_assignment, tmp_path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        older = _submit_l0(queue, category='design_concern', summary='older blocker')
        steward_cls = _make_busy_on_older_l0_steward(
            queue, older.id,
            new_l0=NewL0Handling.CLOSED_WITH_OLDER,
            new_l0_closer='auto-dismissed',
            new_l0_dismiss=True,
        )
        grace = 0.5
        workflow, scheduler = await _build_workflow(
            git_repo, task_assignment, queue, steward_cls, tmp_path,
            completion_timeout=grace,
        )

        try:
            outcome = await asyncio.wait_for(
                workflow._mark_blocked('Suffix item needs rebase onto frozen-prefix tip'),
                10,
            )
        finally:
            await _stop_all(steward_cls)

        assert steward_cls.published == [older.id], (
            'precondition: the sweep close and the older L0\'s outcome must '
            'precede the wait, or BLOCKED is a plain timeout and proves nothing '
            'about the sweep close'
        )
        assert outcome == WorkflowOutcome.BLOCKED, (
            'a sweep close carries no finding, so it must not settle the wait'
        )
        assert 'pending' not in scheduler.statuses.get(TASK_ID, [])
