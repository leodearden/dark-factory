"""Tests for the run()-ESCALATED steward wait (task 3170).

``run()``'s ESCALATED branch waits via :meth:`TaskWorkflow._wait_for_resolution`,
which is escalation-queue-only: it is woken by ``EscalationQueue.resolve`` →
``_resolve_callback`` → ``harness._on_escalation_resolved`` →
``_escalation_events[task_id].set()``.  It never reads the steward outcome
channel that its sibling waiter (:meth:`_await_steward_completion`, reached via
``_mark_blocked``) consumes.

Task 2248 introduced two steward give-up branches that publish a typed
``StewardInterrupted`` outcome WITHOUT dismissing their own L0.  On the
``_mark_blocked`` path that is fine (the channel carries the signal); on the
``run()``-ESCALATED path the publish goes to an unread channel and the still-
pending L0 keeps the waiter blocked forever, while the steward re-handles the
same capped escalation at loop speed.

The converged contract these tests pin:

    A steward give-up ALWAYS dismisses its own L0 before publishing an
    outcome.  No pending L0 survives a steward give-up.

Both waiters observe that single producer invariant, and the ESCALATED wait is
additionally bounded by ``steward_completion_timeout`` so a future producer bug
degrades into a loud, bounded unblock instead of a permanent strand.

Modelled on ``test_workflow_status_on_resume.py`` — the only existing module
that drives ``run()``'s ESCALATED branch deterministically.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import IMPLEMENTER
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome
from orchestrator.workflow_types import StewardInterrupted

# ---------------------------------------------------------------------------
# Fixtures (local — mirrors test_workflow_status_on_resume.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'lib.py').write_text('def greet(name): return name\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    """Config with a SHORT ``steward_completion_timeout``.

    Only ``timeouts.steward >= steward_completion_timeout`` is validated, and
    the stock ``timeouts.steward`` (1800s) satisfies 0.5s comfortably.  Tests
    that must NOT be satisfiable by the timeout (the cross-component
    regression) raise it back up locally.
    """
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        steward_completion_timeout=0.5,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'X',
            'description': 'Y',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


PLAN = {
    'task_id': '42',
    'title': 'X',
    'files': ['lib.py'],
    'analysis': '',
    'prerequisites': [],
    'steps': [
        {
            'id': 'step-1',
            'type': 'impl',
            'description': '',
            'status': 'pending',
            'commit': None,
        },
    ],
    'design_decisions': [],
    'reuse': [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_advanced_worktree(git_ops: GitOps, task_id: str) -> Path:
    """Create a worktree on a task branch ONE commit ahead of main.

    Two things depend on this: the already-on-main short-circuit in run()'s
    ESCALATED branch must not fire (it would mask the wait entirely), and
    ``_worktree_has_wip_commits`` must derive True (it is the wip probe the
    production ``_ensure_steward_started`` injects into the steward, and the
    give-up branches under test are wip-gated).
    """
    wt_info = await git_ops.create_worktree(task_id)
    wt = wt_info.path
    (wt / 'precommit.txt').write_text('test marker\n')
    await _run(['git', 'add', 'precommit.txt'], cwd=wt)
    await _run(['git', 'commit', '-m', 'test marker commit'], cwd=wt)
    return wt


def _build_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    queue: EscalationQueue,
    worktree: Path,
) -> tuple[TaskWorkflow, FakeScheduler]:
    """Wire a TaskWorkflow with all fakes for these tests.

    ``escalation_event`` is passed at construction exactly as the harness does
    at dispatch time (``_register_escalation_event``), so
    :func:`_wire_resolve_callback` always has a live event to set.

    Worktree is pre-set (eval-mode external path) so run() skips
    create_worktree and the MERGE block.
    """
    scheduler = FakeScheduler()
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
        escalation_queue=queue,
        escalation_event=asyncio.Event(),
        initial_plan=dict(PLAN),
    )
    workflow.worktree = worktree
    return workflow, scheduler


def _wire_resolve_callback(queue: EscalationQueue, workflow: TaskWorkflow) -> None:
    """Wire the queue's resolve callback to the workflow's escalation event.

    Mirrors ``harness._on_escalation_resolved`` (harness.py:11164-11174), which
    does exactly ``self._escalation_events[escalation.task_id].set()``.  This is
    the ONLY thing that wakes ``_wait_for_resolution`` in production, so a test
    that omits it does not reproduce the production wake path at all.
    """
    def _on_resolved(esc) -> None:  # noqa: ARG001 — signature parity with harness
        event = workflow._escalation_event
        if event is not None:
            event.set()

    queue.set_resolve_callback(_on_resolved)


def _submit_l0(
    queue: EscalationQueue, task_id: str, *, category: str = 'task_failure',
) -> Escalation:
    """Submit a pending level-0 escalation and return the model (not just the id).

    The steward-side tests need the whole ``Escalation`` to hand to a real
    ``TaskSteward._handle_escalation``.
    """
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='implementer',
        severity='blocking',
        category=category,
        summary='synthetic blocker',
        detail='synthetic detail',
    )
    queue.submit(esc)
    return esc


def _make_evrl_returner(returns: list[WorkflowOutcome]):
    """Return an AsyncMock that pops successive WorkflowOutcomes per call.

    The last value is reused if the list is exhausted, so a misconfigured test
    does not hang the workflow's outer while-true.
    """
    state = {'count': 0, 'queue': list(returns)}

    async def fake_evrl():
        state['count'] += 1
        q = state['queue']
        if len(q) > 1:
            return q.pop(0)
        return q[0] if q else WorkflowOutcome.DONE

    return AsyncMock(side_effect=fake_evrl), state


def _make_silent_steward() -> type:
    """A steward that publishes NOTHING and leaves its L0 pending.

    The general "producer went silent" shape — the failure mode fix C must
    bound REGARDLESS of which producer bug caused it. Deliberately does not
    model the task-2248 bug specifically: fix C has to hold for any future
    early return that forgets to dismiss, not just the two known ones.
    """

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            self._wip_probe = probe

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    return _FakeSteward


def _make_dismiss_and_publish_steward(
    queue: EscalationQueue, task_id: str, *, delay: float = 0.05,
) -> type:
    """A steward that behaves exactly like the post-fix-A producer.

    It DISMISSES its own pending L0 — which fires the queue's resolve callback
    and wakes ``_wait_for_resolution`` — and THEN publishes the typed
    ``StewardInterrupted('attempt_cap', wip_commits_present=True)`` that its
    OTHER consumer (``_await_steward_completion``, reached via
    ``_mark_blocked``) would read.

    Dismiss-then-publish with no await between the two mirrors
    ``TaskSteward._dismiss_capped_l0`` → ``_publish_outcome`` exactly, so the
    waiter cannot resume between them: by the time it runs again, both the
    dismissal and the publish have landed.  That is precisely the situation
    this hygiene step exists for — on the run()-ESCALATED path nobody consumes
    that publish.

    The give-up runs as a background task after a short delay so the waiter
    genuinely blocks and is genuinely woken by the resolve callback, rather
    than finding an already-empty pending list and never exercising the wake
    path at all.
    """

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None
            self.give_up_task: asyncio.Task | None = None

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            self._wip_probe = probe

        async def _give_up(self) -> None:
            await asyncio.sleep(delay)
            for esc in queue.get_by_task(task_id, status='pending', level=0):
                queue.resolve(
                    esc.id,
                    'Auto-dismissed: steward interrupted (attempt_cap) with WIP '
                    'present — resuming plan, not escalating',
                    dismiss=True,
                    resolved_by='auto-dismissed',
                )
            if self._outcome_channel is not None:
                self._outcome_channel.put_nowait(
                    StewardInterrupted('attempt_cap', wip_commits_present=True),
                )

        async def start(self) -> None:
            self.give_up_task = asyncio.create_task(self._give_up())

        async def stop(self) -> None:
            if self.give_up_task is not None:
                await self.give_up_task

    return _FakeSteward


def _submit_l1(queue: EscalationQueue, task_id: str) -> Escalation:
    """Submit a pending LEVEL-1 escalation (severity 'blocking', not born-at-L2).

    Severity matters: a critical/urgent L1 would trip
    ``_wait_for_resolution``'s born-at-L2 stop-the-line check BEFORE the wait
    loop, so the test would pass without exercising the bounded wait at all.
    """
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='steward',
        severity='blocking',
        category='task_failure',
        summary='steward gave up',
        detail='pre-existing L1',
        level=1,
    )
    queue.submit(esc)
    return esc


@pytest.mark.asyncio
class TestEscalatedWaitIsBounded:
    """Fix C: the run()-ESCALATED wait must not be strandable.

    ``_wait_for_resolution``'s loop is ``while True: ... await
    self._escalation_event.wait()`` with no deadline, so a producer that
    leaves an L0 pending and never fires the resolve callback parks the
    workflow forever. Its sibling waiter ``_await_steward_completion``
    already bounds itself by ``steward_completion_timeout``; this pins the
    same knob, the same semantics, on this path.

    Each test wraps ``run()`` in ``asyncio.wait_for`` so a regression fails
    as a bounded test failure rather than hanging the whole suite.
    """

    async def test_bounded_wait_dismisses_orphan_l0_and_resumes(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """No L1 open → the task LEAVES phase escalated and resumes."""
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _submit_l0(queue, task_assignment.task_id)
        workflow, _scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        _wire_resolve_callback(queue, workflow)
        workflow._steward_factory = _make_silent_steward()
        evrl_mock, state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        t0 = asyncio.get_running_loop().time()
        await asyncio.wait_for(workflow.run(), 10)
        elapsed = asyncio.get_running_loop().time() - t0

        assert elapsed < 5, (
            f'the wait must be bounded by steward_completion_timeout '
            f'({config.steward_completion_timeout}s), not by the test\'s own '
            f'10s backstop; run() took {elapsed:.1f}s. (run() swallows the '
            f'wait_for cancellation and returns, so an unbounded wait shows '
            f'up here rather than as a TimeoutError.)'
        )
        assert queue.get_by_task(
            task_assignment.task_id, status='pending', level=0,
        ) == [], (
            'the orphan L0 must be dismissed on the way out — leaving it '
            'pending would re-strand the next ESCALATED entry on the same '
            'record for another full timeout window'
        )
        archived = queue.get(esc.id)
        assert archived is not None, 'the record must still be readable'
        assert archived.resolved_by == 'auto-dismissed', (
            f'the dismissal must be attributed so archived records stay '
            f'greppable by the same signature as every other auto-dismissal; '
            f'got resolved_by={archived.resolved_by!r}'
        )
        assert not queue.has_open_l1(task_assignment.task_id), (
            'a pure wait-timeout must not file an L1 by itself'
        )
        roles = [c.args[0] for c in invoke_mock.await_args_list]
        assert IMPLEMENTER in roles, (
            f'the implementer must be resumed — i.e. the task LEFT phase '
            f'escalated rather than parking there; invoked roles: '
            f'{[r.name for r in roles]}'
        )
        assert state['count'] >= 1

    async def test_bounded_wait_with_open_l1_blocks_without_duplicating_it(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """An L1 already open → BLOCKED via the EXISTING tail, no duplicate L1.

        This is the other half of the acceptance disjunction, and it is why
        the timeout disposition falls through to the unchanged
        ``has_open_l1`` check rather than calling ``_mark_blocked`` directly:
        ``_mark_blocked`` would file a fresh L0 and could add a SECOND full
        ``steward_completion_timeout`` grace window.
        """
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _submit_l0(queue, task_assignment.task_id)
        _submit_l1(queue, task_assignment.task_id)
        workflow, _scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        _wire_resolve_callback(queue, workflow)
        workflow._steward_factory = _make_silent_steward()
        evrl_mock, _state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        result = await asyncio.wait_for(workflow.run(), 10)

        assert result.outcome == WorkflowOutcome.BLOCKED, (
            f'an open L1 must route through _StewardReescalated → '
            f'_mark_blocked(skip_escalation=True); got {result.outcome!r}'
        )
        assert queue.get_by_task(
            task_assignment.task_id, status='pending', level=0,
        ) == [], 'the orphan L0 must be dismissed even on the blocking path'
        assert queue.get(esc.id).resolved_by == 'auto-dismissed'
        assert len(
            queue.get_by_task(task_assignment.task_id, status='pending', level=1),
        ) == 1, 'skip_escalation=True must not duplicate the existing L1'
        roles = [c.args[0] for c in invoke_mock.await_args_list]
        assert IMPLEMENTER not in roles, (
            'the implementer must NOT be resumed when an L1 is open'
        )


@pytest.mark.asyncio
class TestStaleOutcomeHygiene:
    """The ESCALATED wait must leave no unconsumed steward outcome behind.

    Making the run()-ESCALATED path viable (fixes A + C) means outcomes
    published while ``_wait_for_resolution`` is the active waiter are never
    consumed — that waiter is escalation-queue-only by design.  The outcome
    then sits on ``_steward_outcome_channel`` indefinitely, and a LATER
    ``_mark_blocked`` in the same workflow pops it out of
    ``_await_steward_completion`` as if the steward had just published it.

    A stale ``StewardInterrupted(wip_commits_present=True)`` is exactly the
    task-2060 resume-plan outcome, so the consequence is a spurious
    ``_requeue()`` driven by an outcome from a completed, already-dispositioned
    escalation cycle.  (The hazard pre-dates task 3170 — ``StewardResolved`` is
    published on every steward success and the run() path never drained either
    — but fix A makes it reachable far more often.)
    """

    async def test_escalated_wait_drains_the_steward_outcome_channel(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """Post-fix-A producer: dismiss + publish → the wait consumes both."""
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _submit_l0(queue, task_assignment.task_id)
        workflow, _scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        _wire_resolve_callback(queue, workflow)
        workflow._steward_factory = _make_dismiss_and_publish_steward(
            queue, task_assignment.task_id,
        )

        await workflow._ensure_steward_started()
        await asyncio.wait_for(workflow._wait_for_resolution(), 10)

        # Precondition: the producer really did run the post-fix-A sequence.
        assert queue.get(esc.id).resolved_by == 'auto-dismissed', (
            'the fake steward must have dismissed its own L0 — otherwise this '
            'test is measuring fix C\'s timeout, not the drain'
        )
        channel = workflow._steward_outcome_channel
        assert channel is not None, 'the steward wiring must have created a channel'
        assert channel.empty(), (
            'the ESCALATED wait must drain the outcome channel on the way out; '
            f'{channel.qsize()} outcome(s) were left for a later _mark_blocked '
            f'to pop as if freshly published'
        )

        await workflow._steward.stop()

    async def test_later_mark_blocked_cannot_pop_the_stale_outcome(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """The consequence, asserted directly on the sibling waiter.

        With the scheduler status non-terminal, a subsequent
        ``_await_steward_completion`` must synthesize a fresh grace-timeout
        outcome — NOT hand back the ``attempt_cap`` interruption the run()
        path already implicitly consumed a full cycle earlier.
        """
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        _submit_l0(queue, task_assignment.task_id)
        workflow, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, wt,
        )
        _wire_resolve_callback(queue, workflow)
        workflow._steward_factory = _make_dismiss_and_publish_steward(
            queue, task_assignment.task_id,
        )

        await workflow._ensure_steward_started()
        await asyncio.wait_for(workflow._wait_for_resolution(), 10)
        await workflow._steward.stop()

        assert await scheduler.get_status(task_assignment.task_id) not in (
            'done', 'cancelled', 'deferred',
        ), 'a terminal status would short-circuit _await_steward_completion'
        outcome = await asyncio.wait_for(workflow._await_steward_completion(), 10)

        assert isinstance(outcome, StewardInterrupted), (
            f'nothing is published in this second cycle, so the grace period '
            f'must elapse into a synthesized StewardInterrupted; got {outcome!r}'
        )
        assert outcome.reason == 'timeout', (
            f'the second cycle must synthesize its OWN grace-timeout outcome, '
            f'not replay the stale attempt_cap give-up the run()-ESCALATED wait '
            f'already dispositioned — a stale wip-present interruption drives '
            f'_mark_blocked straight into a spurious resume-plan requeue; got '
            f'reason={outcome.reason!r}'
        )
