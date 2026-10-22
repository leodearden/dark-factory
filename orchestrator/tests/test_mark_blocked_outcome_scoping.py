"""``_mark_blocked``'s steward wait is scoped to the escalation it filed (task 5563).

The steward works pending level-0 escalations oldest-first.  When it is still
busy with an older L0 A as ``_mark_blocked`` files B, A's outcome lands on the
shared channel first.  Before task 5563 that outcome ended B's wait, and
``_requeue()`` fired while B was still pending and untouched (reify 6615).

The invariant pinned here, through the real ``_mark_blocked`` with a real
``EscalationQueue`` and a real worktree: an outcome published for escalation X
never resolves the wait for escalation Y.  Only B's own outcome, the grace
deadline, or a cancel ends it.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from _workflow_helpers import (
    GIT_REPO_SEED,
    FakeBriefing,
    FakeMcp,
    FakeScheduler,
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
    queue: EscalationQueue, older_id: str, *, then_resolve_new: bool,
) -> type:
    """A steward class still working the OLDER L0 when ``_mark_blocked`` waits.

    It resolves *older_id* first and publishes that outcome tagged with
    *older_id*, as ``TaskSteward`` does.  With *then_resolve_new* it then
    resolves every other pending L0, publishing one envelope per record tagged
    with that record's own id.

    The class is itself the ``steward_factory``; every built instance lands in
    ``instances``, so teardown can stop them without reaching into the workflow.
    """

    class _FakeSteward:
        instances: list[_FakeSteward] = []

        def __init__(self, wt_path: Path, cfg_dir: Path | None) -> None:  # noqa: ARG002
            self._outcome_channel: asyncio.Queue[PublishedStewardOutcome] | None = None
            self._work: asyncio.Task | None = None
            _FakeSteward.instances.append(self)

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:  # noqa: ARG002
            pass

        def _resolve_and_publish(self, esc_id: str, text: str) -> None:
            queue.resolve(esc_id, text, resolved_by='fake-steward')
            assert self._outcome_channel is not None
            self._outcome_channel.put_nowait(
                PublishedStewardOutcome(esc_id, StewardResolved(text)),
            )

        async def _run(self) -> None:
            await asyncio.sleep(0.05)
            self._resolve_and_publish(older_id, 'older fixed')
            if not then_resolve_new:
                return
            await asyncio.sleep(0.2)
            for esc in queue.get_by_task(TASK_ID, status='pending', level=0):
                self._resolve_and_publish(esc.id, f'fixed {esc.id}')

        async def start(self) -> None:
            self._work = asyncio.create_task(self._run())

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
    *,
    completion_timeout: float,
) -> tuple[TaskWorkflow, FakeScheduler]:
    config = _config(
        git_repo,
        steward_completion_timeout=completion_timeout,
        unblock_auto=UnblockAutoConfig(enabled=False),
    )
    git_ops = _make_git_ops(git_repo)
    scheduler = FakeScheduler()
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
        escalation_queue=queue,
        steward_factory=steward_factory,
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
            queue, older.id, then_resolve_new=False,
        )
        workflow, scheduler = await _build_workflow(
            git_repo, task_assignment, queue, steward_cls, completion_timeout=0.5,
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
            queue, older.id, then_resolve_new=True,
        )
        workflow, _scheduler = await _build_workflow(
            git_repo, task_assignment, queue, steward_cls, completion_timeout=5.0,
        )

        try:
            outcome = await asyncio.wait_for(
                workflow._mark_blocked('Suffix item needs rebase onto frozen-prefix tip'),
                10,
            )
            [new_id] = _new_l0_ids(queue, older.id)
            new_status_at_return = queue.get(new_id).status  # type: ignore[union-attr]
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
            queue, older.id, then_resolve_new=True,
        )
        workflow, _scheduler = await _build_workflow(
            git_repo, task_assignment, queue, steward_cls, completion_timeout=5.0,
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
            parent_status_at_return = queue.get(parent.id).status  # type: ignore[union-attr]
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
