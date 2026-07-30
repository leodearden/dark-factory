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

Both waiters observe that single producer invariant.  The ESCALATED wait is
additionally bounded, so a future producer bug degrades into a loud, bounded
unblock instead of a permanent strand — by an IDLE window of
``timeouts.steward + steward_completion_timeout``, refreshed while the steward
is observably working and stopping the steward before it resumes anything.
(The two waiters share the PRODUCER invariant but deliberately diverge on their
bound: the sibling's ``steward_completion_timeout`` is a post-completion drain
grace, which is far too short for a steward that is actively working.)

Modelled on ``test_workflow_status_on_resume.py`` — the only existing module
that drives ``run()``'s ESCALATED branch deterministically.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec, stamp_stock_routing_config
from _recording_event_store import _RecordingEventStore
from _workflow_helpers import (
    FakeBriefing,
    FakeMcp,
    FakeScheduler,
    _make_resolving_steward,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import IMPLEMENTER
from orchestrator.config import GitConfig, OrchestratorConfig, TimeoutsConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.steward import TaskSteward
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome
from orchestrator.workflow_types import StewardInterrupted, StewardReescalatedL1

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
    """Config with a SHORT ESCALATED-wait window: 0.5 + 0.5 = 1.0s.

    BOTH terms must be shrunk, because the ESCALATED wait is bounded by the
    DERIVED window ``timeouts.steward + steward_completion_timeout`` (review
    fix D1) — not by ``steward_completion_timeout`` alone.  Leaving
    ``timeouts.steward`` at its stock 1800s would give the tests that assert
    the backstop FIRES a ~1800s window, so they would blow their own
    ``asyncio.wait_for`` guard instead of exercising expiry.

    Only ``timeouts.steward >= steward_completion_timeout`` is validated, and
    0.5 >= 0.5 satisfies it (the invariant is ``>=``, not strict ``>``).
    Tests that must NOT be satisfiable by the backstop (the cross-component
    regression) raise ``steward_completion_timeout`` back up locally, which
    widens the derived window a fortiori.
    """
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        steward_completion_timeout=0.5,
        timeouts=TimeoutsConfig(steward=0.5),
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
    event_store: _RecordingEventStore | None = None,
) -> tuple[TaskWorkflow, FakeScheduler]:
    """Wire a TaskWorkflow with all fakes for these tests.

    ``escalation_event`` is passed at construction exactly as the harness does
    at dispatch time (``_register_escalation_event``), so
    :func:`_wire_resolve_callback` always has a live event to set.

    Worktree is pre-set (eval-mode external path) so run() skips
    create_worktree and the MERGE block.

    *event_store* is optional because only the tests that assert on the
    ``steward_wait_timeout`` emission need one; ``_wait_for_resolution``'s
    emit is guarded by ``if self.event_store``, so ``None`` is the normal
    no-telemetry shape rather than a degraded one.
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
        event_store=event_store,  # type: ignore[arg-type]
    )
    workflow.worktree = worktree
    return workflow, scheduler


def _steward_wait_timeouts(store: _RecordingEventStore) -> list[dict]:
    """Every ``escalation_resolved`` event carrying the give-up marker.

    ``_wait_for_resolution``'s expiry handler is the ONLY emitter of
    ``outcome='steward_wait_timeout'`` (it reuses the existing
    ``escalation_resolved`` type rather than adding an EventType member), so
    this list being empty is the precise, greppable signal that the backstop
    did not fire.
    """
    return [
        payload['data']
        for name, payload in store.events
        if name == 'escalation_resolved'
        and payload['data'].get('outcome') == 'steward_wait_timeout'
    ]


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


def _time_the_wait(workflow: TaskWorkflow) -> dict[str, float]:
    """Record how long ``_wait_for_resolution`` itself blocks for.

    Returns a dict the caller reads AFTER ``run()`` returns; ``'elapsed'`` is
    the cumulative time spent inside the wait (summed, so a hypothetical
    second ESCALATED entry cannot hide a second unbounded wait behind a short
    first one) and is ABSENT if the wait was never entered at all.

    The bounded-wait assertions are about the WAIT, not about ``run()``.
    Everything ``run()`` does after the wait — the resumed implementer, real
    git subprocesses in a real worktree, and the DONE tail's real ``httpx``
    POST to ``FakeMcp.url`` (nothing listens on :9999, so this is a genuine
    connect failure, not a stub) — is unbounded-by-design wall-clock that
    scales with machine load.  Measured on the verify box under its 32-worker
    xdist fan-out that tail alone reached multiple seconds, which is what made
    ``assert total_run_elapsed < 5`` fail while the wait it named was in fact
    bounded to ~1.0s exactly as designed.

    Timing the wait preserves the property verbatim — the wait expires on its
    own derived window instead of riding the test's ``asyncio.wait_for``
    backstop — while excluding work that property says nothing about.  A
    regression to an unbounded wait is still caught: ``run()`` swallows the
    backstop's cancellation, and the ``finally`` here records the full
    rode-the-backstop duration on the way out.
    """
    timing: dict[str, float] = {}
    original = workflow._wait_for_resolution

    async def _timed() -> str:
        started = asyncio.get_running_loop().time()
        try:
            return await original()
        finally:
            timing['elapsed'] = timing.get('elapsed', 0.0) + (
                asyncio.get_running_loop().time() - started
            )

    workflow._wait_for_resolution = _timed  # type: ignore[method-assign]
    return timing


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

        timing = _time_the_wait(workflow)

        await asyncio.wait_for(workflow.run(), 10)

        waited = timing.get('elapsed')
        assert waited is not None, 'the ESCALATED wait was never entered'
        assert waited < 5, (
            f'the wait must be bounded by the derived window (timeouts.steward '
            f'{config.timeouts.steward}s + steward_completion_timeout '
            f'{config.steward_completion_timeout}s), not by the test\'s own 10s '
            f'backstop; the wait took {waited:.1f}s. (run() swallows the '
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
        archived = queue.get(esc.id)
        assert archived is not None, 'the record must still be readable'
        assert archived.resolved_by == 'auto-dismissed'
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
        archived = queue.get(esc.id)
        assert archived is not None, 'the record must still be readable'
        assert archived.resolved_by == 'auto-dismissed', (
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

        assert workflow._steward is not None, (
            '_ensure_steward_started() must have wired a steward to tear down'
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
        assert workflow._steward is not None, (
            '_ensure_steward_started() must have wired a steward to tear down'
        )
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


def _make_steward_config() -> MagicMock:
    """A MagicMock ``OrchestratorConfig`` for a REAL ``TaskSteward``.

    Same stamping recipe as ``test_steward.py``'s ``mock_config`` fixture so
    the integration test does not introduce a second config shape.  Only the
    fields the attempt-cap path actually reads are load-bearing
    (``steward_max_attempts``, ``steward_lifetime_budget``); the routing stamp
    is kept because the class's other paths reach ``resolve_route``, and a
    ``spec_set`` MagicMock must not be the reason a future assertion moves.

    Deliberately SEPARATE from the workflow's own ``OrchestratorConfig``: the
    steward-side cap and the workflow-side ``steward_completion_timeout`` are
    independent knobs here, which is what lets the test set the latter high
    enough that fix C cannot be what satisfies it.
    """
    cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    cfg.project_root = Path('/tmp/fake-project')
    cfg.models.steward = 'opus'
    cfg.budgets.steward = 5.0
    cfg.max_turns.steward = 100
    cfg.effort.steward = 'high'
    cfg.backends.steward = 'claude'
    stamp_stock_routing_config(cfg)
    cfg.escalation.host = 'localhost'
    cfg.escalation.port = 8102
    cfg.fused_memory.url = 'http://localhost:8002'
    cfg.fused_memory.project_id = 'dark_factory'
    cfg.steward_lifetime_budget = 12.0
    cfg.steward_max_attempts = 1
    cfg.steward_completion_timeout = 300.0
    cfg.steward_max_timeouts_per_escalation = 3
    cfg.steward_max_empty_outputs_per_escalation = 2
    cfg.timeouts.steward = 1800.0
    return cfg


def _make_real_steward_factory(
    queue: EscalationQueue, esc: Escalation, task_id: str,
):
    """A ``_steward_factory`` that builds a GENUINE ``TaskSteward``.

    Only ``start``/``stop`` are overridden, and only to replace ``_run_loop``
    (which would spawn the inotify watcher subprocess and invoke a real agent)
    with a single, deterministic ``_handle_escalation(esc)`` — seeded so the
    per-escalation attempt cap fires on the first pass.  EVERYTHING the
    contract depends on stays real: the cap guard, the wip gate, the
    dismissal (``_dismiss_capped_l0`` → ``EscalationQueue.resolve`` → the
    resolve callback), the publish, and the terminal-state memory.

    That composition is the point.  A steward-only unit test passes with fix A
    but never proves the workflow wakes; a workflow-only test with a fake
    steward passes against a fake contract.  Task 2248 shipped the strand
    precisely because only the ``_mark_blocked`` half was ever tested.
    """
    steward_config = _make_steward_config()

    class _CapFiringSteward(TaskSteward):
        async def start(self) -> None:
            # Seed the retry counter so the FIRST _handle_escalation trips the
            # per-escalation retry guard (steward_max_attempts=1) — exactly
            # the state a steward is in after one failed resolution attempt.
            self._retry_counts[esc.id] = steward_config.steward_max_attempts
            self._give_up_task = asyncio.create_task(self._handle_escalation(esc))

        async def stop(self) -> None:
            give_up = getattr(self, '_give_up_task', None)
            if give_up is not None:
                await give_up
            self._stopped = True

    def _factory(wt_path: Path, cfg_dir):
        return _CapFiringSteward(
            task_id=task_id,
            task={'id': task_id, 'title': 'X', 'description': 'Y'},
            worktree=wt_path,
            config=steward_config,
            mcp=MagicMock(),
            escalation_queue=queue,
            briefing=AsyncMock(),
            config_dir=cfg_dir,
        )

    return _factory


@pytest.mark.asyncio
class TestRealStewardGiveUpUnblocksEscalatedRun:
    """The cross-component regression whose absence let task 2248 ship.

    Composes the REAL ``TaskSteward`` attempt-cap + WIP give-up with the REAL
    ``run()``-ESCALATED wait and the harness-shaped resolve callback.  The
    steward publishes its typed outcome to a channel nobody on this path
    reads, so the ONLY thing that can unblock the workflow is the producer-side
    dismissal — the converged contract:

        A steward give-up ALWAYS dismisses its own L0 before publishing an
        outcome.  No pending L0 survives a steward give-up.
    """

    async def test_real_attempt_cap_give_up_wakes_the_escalated_waiter(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _submit_l0(queue, task_assignment.task_id)
        # LOAD-BEARING: a large completion timeout makes it impossible for fix
        # C's bounded-wait disposition to be what satisfies this test.  With
        # the fixture's 0.5s the assertions below would still pass with fix A
        # fully reverted — the strand would just be auto-dismissed 0.5s later,
        # and this regression would silently stop regressing.
        long_wait = 60.0
        workflow, _scheduler = _build_workflow(
            config.model_copy(update={'steward_completion_timeout': long_wait}),
            git_ops, task_assignment, queue, wt,
        )
        _wire_resolve_callback(queue, workflow)
        workflow._steward_factory = _make_real_steward_factory(
            queue, esc, task_assignment.task_id,
        )
        evrl_mock, _state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        t0 = asyncio.get_running_loop().time()
        await asyncio.wait_for(workflow.run(), 15)
        elapsed = asyncio.get_running_loop().time() - t0

        assert elapsed < long_wait / 2, (
            f'the give-up dismissal must wake the waiter promptly; run() took '
            f'{elapsed:.1f}s against a {long_wait:.0f}s completion timeout, so '
            f'the timeout path — not the dismissal — is what unblocked it'
        )
        archived = queue.get(esc.id)
        assert archived is not None, 'the record must still be readable'
        assert archived.status == 'dismissed', (
            f'the wip-gated give-up must dismiss its own L0 rather than leave '
            f'it pending for a waiter that never reads the outcome channel; '
            f'status={archived.status!r}'
        )
        assert archived.resolved_by == 'auto-dismissed', (
            f'got resolved_by={archived.resolved_by!r}'
        )
        assert queue.get_by_task(
            task_assignment.task_id, status='pending', level=1,
        ) == [], (
            'the task-2060 resume-plan semantics must be preserved: a wip-'
            'present interruption is resumable, so no L1 may be filed'
        )
        roles = [c.args[0] for c in invoke_mock.await_args_list]
        assert IMPLEMENTER in roles, (
            f'the workflow must LEAVE phase escalated and resume the '
            f'implementer; invoked roles: {[r.name for r in roles]}'
        )
        # The steward's typed outcome went to a channel this path never reads
        # — the drain (step-14) must have consumed it on the way out.
        channel = workflow._steward_outcome_channel
        assert channel is not None and channel.empty(), (
            'the unconsumed give-up outcome must not survive the wait'
        )


def _short_window_config(
    base: OrchestratorConfig, *, completion: float, invocation: float,
) -> OrchestratorConfig:
    """A freshly-VALIDATED config with both steward timeouts pinned.

    Constructed rather than ``model_copy``d because the pair is exactly what
    the ``timeouts.steward >= steward_completion_timeout`` validator
    (config.py:4071-4081) guards, and these tests turn on that relationship —
    a silently-invalid pair would make them meaningless.
    """
    return OrchestratorConfig(
        project_root=base.project_root,
        max_concurrent_tasks=1,
        steward_completion_timeout=completion,
        timeouts=TimeoutsConfig(steward=invocation),
        git=base.git,
    )


def _make_slow_healthy_steward(
    queue: EscalationQueue,
    task_id: str,
    *,
    delay: float,
    markers: list[str],
    resolved_by: str = 'steward-auto-dismissed',
) -> type:
    """A HEALTHY steward that is merely SLOW: silent for *delay*, then gives up.

    It upholds the fix-A producer contract exactly — dismiss its own L0 through
    ``EscalationQueue.resolve`` (which fires the resolve callback and wakes
    ``_wait_for_resolution``), then publish the typed outcome — but does so
    only after *delay*, which the caller sets PAST
    ``steward_completion_timeout`` and INSIDE the steward's own per-invocation
    ceiling ``timeouts.steward``.  That is the normal-operation window at stock
    config, not an exotic one: stock is 900s vs 1800s, and the validator at
    config.py:4071-4081 GUARANTEES the ceiling exceeds the completion timeout.

    ``resolved_by`` is deliberately NOT ``'auto-dismissed'``: that is the
    workflow's own force-dismissal signature, so a distinct value is what lets
    the test tell "the steward finished its work" apart from "the workflow gave
    up on it and stamped the record on the way past".

    Appends ``'steward-stop'`` to *markers* on every ``stop()`` await, so the
    caller can assert on ORDER against the implementer-resume marker rather
    than merely on occurrence.
    """

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None
            self.give_up_task: asyncio.Task | None = None
            self.stop_count = 0

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
                    resolved_by=resolved_by,
                )
            if self._outcome_channel is not None:
                self._outcome_channel.put_nowait(
                    StewardInterrupted('attempt_cap', wip_commits_present=True),
                )

        async def start(self) -> None:
            self.give_up_task = asyncio.create_task(self._give_up())

        async def stop(self) -> None:
            self.stop_count += 1
            markers.append('steward-stop')
            if self.give_up_task is not None:
                self.give_up_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.give_up_task

    return _FakeSteward


def _make_marking_invoke(markers: list[str]) -> AsyncMock:
    """An ``_invoke`` double that records WHEN the implementer was resumed.

    Ordering against ``'steward-stop'`` is the safety property under test:
    the steward runs its agent with ``cwd = self.worktree`` (steward.py:542,
    590) — the same worktree the resumed implementer edits and commits in — so
    "both happened" is not enough, only "stopped BEFORE resumed" is.
    """

    async def _invoke(role, *_args, **_kwargs):
        if role is IMPLEMENTER:
            markers.append('implementer-resumed')
        return AgentResult(success=True, output='')

    return AsyncMock(side_effect=_invoke)


@pytest.mark.asyncio
class TestHealthyStewardIsNotForceDismissed:
    """Fix D1: the ESCALATED-wait backstop must not fire on a HEALTHY steward.

    Step-12 bounded the wait by ``config.steward_completion_timeout`` so the
    two waiters would share one knob.  That symmetry was itself the defect:
    a SINGLE steward ``invoke_agent`` call is budgeted
    ``config.timeouts.steward`` (1800s stock) while the wait was bounded by
    ``steward_completion_timeout`` (900s stock) — and the only validator
    (config.py:4071-4081) enforces ``timeouts.steward >=
    steward_completion_timeout``, i.e. it GUARANTEES the per-invocation
    ceiling exceeds the wait bound.  ``TimeoutsConfig``'s own docstring
    (config.py:211-222) calls ``steward_completion_timeout`` the
    post-completion *drain grace window*, "intentionally decoupled" from
    per-invocation work.

    So at stock config the backstop is a NORMAL-operation trigger, not the
    rare future-producer-bug backstop its docstring claims — and firing it
    force-dismisses the L0 out from under a live steward and resumes the
    implementer beside it in the same worktree.
    """

    async def test_slow_but_working_steward_keeps_ownership_of_its_l0(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """Silent past the completion timeout, inside the invocation ceiling."""
        # 0.3s completion timeout + 1.0s per-invocation ceiling.  The steward
        # gives up at 0.6s: past the OLD bound (0.3s) — so this fails today —
        # and inside the derived window (1.0 + 0.3 = 1.3s).
        local_config = _short_window_config(config, completion=0.3, invocation=1.0)
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _submit_l0(queue, task_assignment.task_id)
        store = _RecordingEventStore()
        workflow, _scheduler = _build_workflow(
            local_config, git_ops, task_assignment, queue, wt, event_store=store,
        )
        _wire_resolve_callback(queue, workflow)
        markers: list[str] = []
        workflow._steward_factory = _make_slow_healthy_steward(
            queue, task_assignment.task_id, delay=0.6, markers=markers,
        )
        evrl_mock, _state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = _make_marking_invoke(markers)
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        await asyncio.wait_for(workflow.run(), 10)

        archived = queue.get(esc.id)
        assert archived is not None, 'the record must still be readable'
        assert archived.resolved_by == 'steward-auto-dismissed', (
            f'the L0 must still carry the STEWARD\'s own attribution: this '
            f'steward was merely slow — silent for 0.6s, well inside its '
            f'{local_config.timeouts.steward}s per-invocation ceiling — and '
            f'dismissed its own L0 exactly as the fix-A producer contract '
            f'requires.  resolved_by={archived.resolved_by!r} means the '
            f'workflow force-dismissed the record out from under a live '
            f'steward after only {local_config.steward_completion_timeout}s'
        )
        assert _steward_wait_timeouts(store) == [], (
            'the give-up backstop must not fire on a healthy steward; it '
            'exists for a genuinely SILENT producer'
        )
        assert 'implementer-resumed' in markers, (
            f'the steward\'s own dismissal must still unblock the wait and '
            f'resume the implementer; markers={markers}'
        )
        # NOT "stop() was never awaited": run()'s terminal-cleanup hook
        # (_on_terminal_cleanups → _stop_steward, workflow.py:2995-2997)
        # legitimately stops the steward once on the way out of every run().
        # The property under test is that the GIVE-UP path did not stop it —
        # i.e. no stop precedes the resume.
        assert markers.index('implementer-resumed') < markers.index('steward-stop'), (
            f'the workflow must not have stopped the steward before resuming '
            f'the implementer — that only happens on the give-up path, which '
            f'must not have fired here; markers={markers}'
        )
        assert workflow._steward is not None, (
            'the give-up path\'s stop-then-clear must not have run: a cleared '
            '_steward is the tell that the workflow gave up on this steward'
        )


def _make_progressing_steward(
    queue: EscalationQueue,
    task_id: str,
    *,
    tick: float,
    ticks: int,
    markers: list[str],
    advance: bool = True,
    counter: str = 'invocations',
    resolved_by: str = 'steward-auto-dismissed',
) -> type:
    """A steward that publishes an observable liveness signal while it works.

    Exposes ``metrics.invocations`` — the same public counter the real
    ``TaskSteward`` bumps after EVERY invocation returns (steward.py:597, and
    :948 on the auto-escalate path), including each timeout-kill retry — and
    advances it once per *tick* for *ticks* ticks before dismissing its L0.

    That spread is the point: with ``steward_max_attempts`` (1) plus
    ``steward_max_timeouts_per_escalation`` (3), a HEALTHY steward can
    legitimately occupy ~4 full invocation ceilings on ONE escalation, because
    the timeout-kill path explicitly loops (steward.py:399-412 increments
    ``_timeout_counts`` and leaves the record pending for re-handling).  A
    single fixed window — even the derived one — still fires mid-retry.

    *advance=False* is the negative control: byte-identical timing and
    behaviour, but the counter never moves.  That is the genuinely-SILENT
    producer the backstop exists for, and it must still be given up on —
    otherwise "refresh on progress" would be trivially satisfiable by never
    expiring at all.

    *counter* selects WHICH of the real ``StewardMetrics`` liveness fields
    advances.  ``'subprocess_attempts'`` models the all-accounts-capped steward
    (task 3170, review fix D4): ``invocations`` stays frozen for the whole
    cap-retry loop — potentially hours — while each subprocess attempt still
    ticks.  Both fields are always exposed, matching the real dataclass, so the
    two variants differ only in which one moves.
    """

    class _Metrics:
        def __init__(self) -> None:
            self.invocations = 0
            self.subprocess_attempts = 0

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None
            self.metrics = _Metrics()
            self.work_task: asyncio.Task | None = None
            self.stop_count = 0

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            self._wip_probe = probe

        async def _work(self) -> None:
            for _ in range(ticks):
                await asyncio.sleep(tick)
                if advance:
                    setattr(
                        self.metrics, counter,
                        getattr(self.metrics, counter) + 1,
                    )
            for esc in queue.get_by_task(task_id, status='pending', level=0):
                queue.resolve(
                    esc.id,
                    'Auto-dismissed: steward interrupted (attempt_cap) with WIP '
                    'present — resuming plan, not escalating',
                    dismiss=True,
                    resolved_by=resolved_by,
                )
            if self._outcome_channel is not None:
                self._outcome_channel.put_nowait(
                    StewardInterrupted('attempt_cap', wip_commits_present=True),
                )

        async def start(self) -> None:
            self.work_task = asyncio.create_task(self._work())

        async def stop(self) -> None:
            self.stop_count += 1
            markers.append('steward-stop')
            if self.work_task is not None:
                self.work_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.work_task

    return _FakeSteward


@pytest.mark.asyncio
class TestObservableProgressRefreshesTheWait:
    """Fix D2: a steward that is visibly WORKING must not be given up on.

    A fixed window — even the derived one from fix D1 — is still wrong at the
    tail, because the steward legitimately RETRIES and each retry gets its own
    full ``timeouts.steward`` invocation.  The fix is to refresh the deadline
    whenever the steward is observably still working, so only a GENUINELY
    SILENT producer trips the backstop.  Both tests share one window and one
    fake shape, differing ONLY in whether ``metrics.invocations`` advances.
    """

    async def test_steady_progress_extends_the_wait_past_a_full_window(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """~1.2s of steady progress against a 0.5s window."""
        local_config = _short_window_config(config, completion=0.2, invocation=0.3)
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _submit_l0(queue, task_assignment.task_id)
        store = _RecordingEventStore()
        workflow, _scheduler = _build_workflow(
            local_config, git_ops, task_assignment, queue, wt, event_store=store,
        )
        _wire_resolve_callback(queue, workflow)
        markers: list[str] = []
        workflow._steward_factory = _make_progressing_steward(
            queue, task_assignment.task_id, tick=0.3, ticks=4, markers=markers,
        )
        evrl_mock, _state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        workflow._invoke = _make_marking_invoke(markers)  # type: ignore[method-assign]

        await asyncio.wait_for(workflow.run(), 10)

        window = (
            local_config.timeouts.steward + local_config.steward_completion_timeout
        )
        archived = queue.get(esc.id)
        assert archived is not None, 'the record must still be readable'
        assert archived.resolved_by == 'steward-auto-dismissed', (
            f'the steward advanced metrics.invocations every 0.3s for ~1.2s — '
            f'more than two full {window:.1f}s windows of visible progress — so '
            f'the wait must have been refreshed rather than expired.  '
            f'resolved_by={archived.resolved_by!r} means a fixed window fired '
            f'mid-retry on a steward that was demonstrably working'
        )
        assert _steward_wait_timeouts(store) == [], (
            'the backstop must trip only on a genuinely SILENT producer'
        )
        assert 'implementer-resumed' in markers, (
            f'the steward\'s own dismissal must still unblock the wait; '
            f'markers={markers}'
        )
        # See the sibling class: run()'s terminal-cleanup hook legitimately
        # stops the steward once on the way out, so ORDER is the property.
        assert markers.index('implementer-resumed') < markers.index('steward-stop'), (
            f'the give-up path must not have stopped the steward; markers={markers}'
        )

    async def test_a_cap_waiting_steward_is_not_mistaken_for_a_silent_one(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """Fix D4: ``invocations`` frozen + ``subprocess_attempts`` advancing.

        This is the all-accounts-capped window, which is a routine designed-for
        condition in this system, not an exotic one.  ``_invoke_with_session``
        delegates to ``invoke_with_cap_retry``, which runs up to 16 subprocess
        attempts with cooldowns between them behind ONE return — and
        ``metrics.invocations`` is bumped only after that return (steward.py:597).
        So a perfectly healthy steward patiently waiting out a cap shows a FROZEN
        ``invocations`` for the whole wait.

        If the waiter's progress signal reads ``invocations`` alone, it fires on
        that steward: ``stop()`` cancels the loop and kills the in-flight agent's
        process group, the L0 is dismissed, and the implementer resumes into the
        same cap.  That is exactly the outcome fix D2 says must happen ONLY on a
        genuinely SILENT producer, so the per-attempt counter has to count.
        """
        local_config = _short_window_config(config, completion=0.2, invocation=0.3)
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _submit_l0(queue, task_assignment.task_id)
        store = _RecordingEventStore()
        workflow, _scheduler = _build_workflow(
            local_config, git_ops, task_assignment, queue, wt, event_store=store,
        )
        _wire_resolve_callback(queue, workflow)
        markers: list[str] = []
        workflow._steward_factory = _make_progressing_steward(
            queue, task_assignment.task_id, tick=0.3, ticks=4, markers=markers,
            counter='subprocess_attempts',
        )
        evrl_mock, _state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        workflow._invoke = _make_marking_invoke(markers)  # type: ignore[method-assign]

        await asyncio.wait_for(workflow.run(), 10)

        window = (
            local_config.timeouts.steward + local_config.steward_completion_timeout
        )
        archived = queue.get(esc.id)
        assert archived is not None, 'the record must still be readable'
        assert archived.resolved_by == 'steward-auto-dismissed', (
            f'the steward ticked metrics.subprocess_attempts every 0.3s for '
            f'~1.2s — more than two full {window:.1f}s windows of visible '
            f'cap-retry progress — with metrics.invocations frozen throughout, '
            f'exactly as an all-accounts-capped steward looks.  '
            f'resolved_by={archived.resolved_by!r} means the backstop fired on '
            f'a working producer'
        )
        assert _steward_wait_timeouts(store) == [], (
            'a steward waiting out a usage cap is not a silent producer'
        )
        assert 'steward-stop' not in markers[:markers.index('implementer-resumed')], (
            f'the give-up path must not have killed the cap-waiting steward '
            f'mid-work; markers={markers}'
        )

    async def test_a_never_advancing_counter_is_still_given_up_on(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """Negative control: identical fake, counter frozen → backstop fires.

        Without this, "refresh on progress" would be satisfiable by an
        implementation that simply never expires — which would re-open the
        permanent strand this whole task exists to close.
        """
        local_config = _short_window_config(config, completion=0.2, invocation=0.3)
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _submit_l0(queue, task_assignment.task_id)
        store = _RecordingEventStore()
        workflow, _scheduler = _build_workflow(
            local_config, git_ops, task_assignment, queue, wt, event_store=store,
        )
        _wire_resolve_callback(queue, workflow)
        markers: list[str] = []
        workflow._steward_factory = _make_progressing_steward(
            queue, task_assignment.task_id, tick=0.3, ticks=4, markers=markers,
            advance=False,
        )
        evrl_mock, _state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        workflow._invoke = _make_marking_invoke(markers)  # type: ignore[method-assign]

        timing = _time_the_wait(workflow)

        await asyncio.wait_for(workflow.run(), 10)

        archived = queue.get(esc.id)
        assert archived is not None, 'the record must still be readable'
        assert archived.resolved_by == 'auto-dismissed', (
            f'a producer that never advances its liveness counter is exactly '
            f'what the backstop is for; got resolved_by={archived.resolved_by!r}'
        )
        assert [e['escalation_ids'] for e in _steward_wait_timeouts(store)] == [
            [esc.id],
        ], (
            'the give-up must stay loud and greppable — one escalation_resolved '
            'event carrying outcome=steward_wait_timeout and the orphan id'
        )
        waited = timing.get('elapsed')
        assert waited is not None, 'the ESCALATED wait was never entered'
        assert waited < 10 / 2, (
            f'the silent producer must still be given up on promptly, not '
            f'ride the test\'s own backstop; the wait took {waited:.1f}s'
        )


def _make_silent_marking_steward(
    markers: list[str], instances: list, *, stop_delay: float = 0.0,
) -> type:
    """The genuinely-SILENT producer, instrumented for ORDER assertions.

    Publishes nothing, dismisses nothing, and never advances
    ``metrics.invocations`` — so the idle window legitimately expires and the
    give-up path fires.  Records every ``stop()`` await into the shared
    *markers* list alongside :func:`_make_marking_invoke`'s implementer marker,
    and every constructed instance into *instances* so the caller can still
    inspect one after the workflow drops its reference.

    *stop_delay* makes ``stop()`` genuinely await, so an implementation that
    fired it as a background task (rather than awaiting it before resuming)
    would lose the ordering race and be caught.
    """

    class _Metrics:
        def __init__(self) -> None:
            self.invocations = 0

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None
            self.metrics = _Metrics()
            self.stop_count = 0
            instances.append(self)

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            self._wip_probe = probe

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            if stop_delay:
                await asyncio.sleep(stop_delay)
            self.stop_count += 1
            markers.append('steward-stop')

    return _FakeSteward


@pytest.mark.asyncio
class TestGiveUpStopsTheStewardBeforeResuming:
    """Fix D3: the backstop must STOP the steward before it resumes anything.

    ``TaskSteward`` invokes its agent with ``cwd = self.worktree``
    (steward.py:542, 590) — the SAME worktree the resumed implementer edits and
    commits in.  Before this fix the only ``_steward.stop()`` sites were the
    terminal-exit teardown hook (workflow.py:2995) and the unrelated
    unactionable/false-premise paths, so when the wait backstop fired the
    workflow force-dismissed the L0 out from under a LIVE steward agent and
    then resumed the implementer beside it: two agents committing in one git
    worktree.  That is a corruption/lost-work hazard, not a benign slow path.
    """

    async def test_expiry_stops_the_steward_before_resuming_the_implementer(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        local_config = _short_window_config(config, completion=0.2, invocation=0.3)
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        esc = _submit_l0(queue, task_assignment.task_id)
        store = _RecordingEventStore()
        workflow, _scheduler = _build_workflow(
            local_config, git_ops, task_assignment, queue, wt, event_store=store,
        )
        _wire_resolve_callback(queue, workflow)
        markers: list[str] = []
        instances: list = []
        workflow._steward_factory = _make_silent_marking_steward(
            markers, instances, stop_delay=0.05,
        )
        evrl_mock, _state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        workflow._invoke = _make_marking_invoke(markers)  # type: ignore[method-assign]

        await asyncio.wait_for(workflow.run(), 10)

        assert len(instances) == 1, (
            f'exactly one steward must have been constructed; got {len(instances)}'
        )
        assert markers.count('steward-stop') == 1, (
            f'the give-up path must stop the steward exactly once — and having '
            f'cleared the reference, run()\'s terminal-cleanup hook must then '
            f'find nothing left to stop; markers={markers}'
        )
        assert 'implementer-resumed' in markers, (
            f'the disposition is unchanged: the wait still unblocks and '
            f'resumes; markers={markers}'
        )
        assert markers.index('steward-stop') < markers.index('implementer-resumed'), (
            f'ORDER is the safety property, not mere occurrence: the steward '
            f'runs its agent in the SAME worktree the resumed implementer '
            f'commits in, so a stop that lands after the resume leaves two '
            f'agents writing one worktree; markers={markers}'
        )
        assert workflow._steward is None, (
            'the give-up must clear the reference too, so a later '
            '_mark_blocked builds a FRESH steward via _ensure_steward_started '
            'rather than awaiting a cancelled loop that can never publish'
        )
        # The step-12 disposition is preserved, not replaced.
        archived = queue.get(esc.id)
        assert archived is not None, 'the record must still be readable'
        assert archived.resolved_by == 'auto-dismissed', (
            f'the orphan L0 must still be dismissed on the way out; '
            f'resolved_by={archived.resolved_by!r}'
        )
        assert [e['escalation_ids'] for e in _steward_wait_timeouts(store)] == [
            [esc.id],
        ], 'the steward_wait_timeout event must still be emitted'


def _make_reescalating_steward(queue: EscalationQueue, task_id: str) -> type:
    """A steward that gives up to a human, exactly as ``_auto_escalate_to_human``.

    Dismisses its own L0, files a level-1, and publishes ``StewardReescalatedL1``
    — the in-cycle hand-off that ``_wait_for_resolution`` turns into
    ``_StewardReescalated`` → ``run()`` → ``_mark_blocked(skip_escalation=True)``,
    whose ``_await_steward_completion`` is the intended reader of that publish.
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
            l1 = _submit_l1(queue, task_id)
            for esc in queue.get_by_task(task_id, status='pending', level=0):
                queue.resolve(
                    esc.id, 'Auto-dismissed: re-escalated to human',
                    dismiss=True, resolved_by='auto-dismissed',
                )
            if self._outcome_channel is not None:
                self._outcome_channel.put_nowait(StewardReescalatedL1(esc_id=l1.id))

        async def stop(self) -> None:
            pass

    return _FakeSteward


@pytest.mark.asyncio
class TestEscalatedWaitDoesNotStallItsSuccessors:
    """Neither continuation of the ESCALATED wait may burn a grace window.

    Both call sites of ``_await_steward_completion`` sit DOWNSTREAM of
    ``_wait_for_resolution`` and, before task 3170, returned instantly only
    because a stale outcome was still sitting on the channel.  Draining that
    channel (step-14) without accounting for them turns each into a full
    ``steward_completion_timeout`` stall — 15 minutes on stock config, on the
    two most common post-escalation routes there are.

    Both tests use a LARGE completion timeout and assert on elapsed wall-clock:
    a stall regression must show up as a bounded failure here, not as an
    invisible slowdown that only the 900s production value makes painful.
    """

    async def test_resolved_escalation_reaches_done_without_a_grace_window(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """Success tail: steward resolved the L0, so nothing is outstanding."""
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        _submit_l0(queue, task_assignment.task_id)
        long_wait = 60.0
        workflow, _scheduler = _build_workflow(
            config.model_copy(update={'steward_completion_timeout': long_wait}),
            git_ops, task_assignment, queue, wt,
        )
        _wire_resolve_callback(queue, workflow)
        workflow._steward_factory = _make_resolving_steward(
            queue, task_assignment.task_id,
        )
        evrl_mock, _state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        workflow._invoke = AsyncMock(  # type: ignore[method-assign]
            return_value=AgentResult(success=True, output=''),
        )

        t0 = asyncio.get_running_loop().time()
        result = await asyncio.wait_for(workflow.run(), 30)
        elapsed = asyncio.get_running_loop().time() - t0

        assert result.outcome == WorkflowOutcome.DONE
        assert elapsed < long_wait / 4, (
            f'the post-merge success tail waits for the steward to finish any '
            f'PENDING work; with the L0 already resolved there is none, so it '
            f'must not burn the grace window. run() took {elapsed:.1f}s against '
            f'a {long_wait:.0f}s steward_completion_timeout'
        )

    async def test_reescalated_l1_blocks_without_a_grace_window(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """_StewardReescalated hand-off: _mark_blocked must read that publish.

        The steward's ``StewardReescalatedL1`` is an IN-CYCLE hand-off to
        ``_mark_blocked(skip_escalation=True)``, not a stale leftover — draining
        it in ``_wait_for_resolution`` would leave that consumer waiting a full
        grace window for an outcome that was already published and thrown away.
        """
        wt = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        queue = EscalationQueue(tmp_path / 'queue')
        _submit_l0(queue, task_assignment.task_id)
        long_wait = 60.0
        workflow, _scheduler = _build_workflow(
            config.model_copy(update={'steward_completion_timeout': long_wait}),
            git_ops, task_assignment, queue, wt,
        )
        _wire_resolve_callback(queue, workflow)
        workflow._steward_factory = _make_reescalating_steward(
            queue, task_assignment.task_id,
        )
        evrl_mock, _state = _make_evrl_returner(
            [WorkflowOutcome.ESCALATED, WorkflowOutcome.DONE],
        )
        workflow._execute_verify_review_loop = evrl_mock  # type: ignore[method-assign]
        invoke_mock = AsyncMock(return_value=AgentResult(success=True, output=''))
        workflow._invoke = invoke_mock  # type: ignore[method-assign]

        t0 = asyncio.get_running_loop().time()
        result = await asyncio.wait_for(workflow.run(), 30)
        elapsed = asyncio.get_running_loop().time() - t0

        assert result.outcome == WorkflowOutcome.BLOCKED
        assert elapsed < long_wait / 4, (
            f'the steward already published StewardReescalatedL1 for this very '
            f'cycle; _mark_blocked(skip_escalation=True) must read it rather '
            f'than wait out the grace window. run() took {elapsed:.1f}s against '
            f'a {long_wait:.0f}s steward_completion_timeout'
        )
        assert len(
            queue.get_by_task(task_assignment.task_id, status='pending', level=1),
        ) == 1, 'skip_escalation=True must not duplicate the L1'
        assert IMPLEMENTER not in [c.args[0] for c in invoke_mock.await_args_list], (
            'the implementer must NOT be resumed when the steward handed off to a human'
        )
