"""The MERGE-entry gating bail enters the bounded Path-A escalated wait (γ₁).

Before this task, ``_run_merge_phase``'s gating bail did
``return WorkflowOutcome.ESCALATED`` with **no** steward started, **no**
``_wait_for_resolution``, and **no** task-status write.  That outcome
propagates ``_run_merge_phase`` → ``_merge_and_finalise`` → ``_drive`` →
``run()`` verbatim, and ``harness._run_slot``'s ``finally``
(``harness.py:7949-7972``) then clears the claimant unconditionally while
writing no status.  The result is the strand shape spec E1/S1 names:

    task row ``in-progress`` + NULL claimant + an open gating escalation,
    with nothing left running to advance it.

The contract these tests pin (PRD ``plans/task-escalation-state-graph-prd.md``
D1; spec ``docs/task-escalation-state-spec.md`` E1/S1/S5):

    At MERGE entry with gating escalations open the workflow enters the SAME
    bounded ESCALATED machinery every other phase uses — ``_enter_phase(
    ESCALATED)`` → ``_ensure_steward_started()`` → ``_wait_for_resolution()``
    — and every terminal exit writes its successor status BEFORE returning.
    There is no steward-less ESCALATED exit.  Uniformity is the point: one
    ESCALATED semantics, not two.

Three boundaries, one per disposition:

* **#1 — steward consumes the L0.**  The wait returns normally, a single-shot
  gate re-check finds nothing open, and ``_run_merge_phase`` falls through to
  the merge retry loop **in-slot**.  No re-dispatch is burned and the row
  stays ``in-progress`` under the live claimant for the whole wait (spec §4:
  ``in-progress`` covers Path-A escalated-waiting).
* **#2 — born-at-L2 / level≥2 open.**  ``_wait_for_resolution``'s
  stop-the-line short-circuit (``workflow.py:12206-12217``) raises
  ``_StewardReescalated`` BEFORE the wait loop, so a human-gated record is
  never waited on in-slot; the handler parks the row ``blocked`` via
  ``_mark_blocked(..., skip_escalation=True)`` and leaves the L2 untouched.
* **#3 — the wait expires (silent producer).**  The existing give-up
  machinery stops the steward, emits ``escalation_resolved{outcome=
  'steward_wait_timeout'}`` and dismisses the orphan L0s; the merge-entry
  handler must then BLOCK with an L1 rather than merge, because an expired
  wait means nobody ADJUDICATED the gating record and repend-PRD D4
  stop-the-line forbids merging past it.

Plus the bounded-by-construction property: the post-wait gate re-check is
SINGLE-SHOT.  A still-open or newly-filed gating record parks the task
instead of re-entering the wait, so the whole merge-entry gate path costs at
most ONE derived idle window (spec S5).

Gate policy itself is untouched: ``_is_gating_escalation`` and the ``gating =
[...]`` comprehension are byte-identical (PRD out-of-scope fence), which is
why these tests drive a REAL ``EscalationQueue`` and never neuter
``_check_escalations`` — the real predicate stays in the loop.

This is the module ``task 3423``'s metadata promised and never delivered
(3423 is now cancelled, its ``metadata.files`` emptied per PRD D11).

Modelled on ``test_workflow_escalated_steward_stall.py`` (task 3170 — the only
module that drives the bounded steward wait deterministically) crossed with
``test_workflow_merge_phase_liveness.py``'s ``_run_merge_phase`` stub preamble.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _recording_event_store import _RecordingEventStore
from _workflow_helpers import (
    FakeBriefing,
    FakeMcp,
    FakeScheduler,
    _make_resolving_steward,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import (
    GitConfig,
    OrchestratorConfig,
    SandboxConfig,
    TimeoutsConfig,
)
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

# ---------------------------------------------------------------------------
# Fixtures (local — mirrors test_workflow_escalated_steward_stall.py:70-118)
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
    DERIVED window ``timeouts.steward + steward_completion_timeout`` — not by
    ``steward_completion_timeout`` alone.  Leaving ``timeouts.steward`` at its
    stock 1800s would give the expiry test (boundary #3) a ~1800s window, so
    it would blow its own ``asyncio.wait_for`` guard instead of exercising
    expiry.  (Same rationale, verbatim, as
    ``test_workflow_escalated_steward_stall.py:91-103``.)

    Only ``timeouts.steward >= steward_completion_timeout`` is validated, and
    0.5 >= 0.5 satisfies it (the invariant is ``>=``, not strict ``>``).

    ``max_merge_retries`` / ``max_pre_merge_retries`` are pinned to 1 so a
    boundary-#1 fall-through executes exactly ONE merge attempt, making the
    stubbed ``_submit_to_merge_queue``'s await count an exact
    "did the merge proceed in-slot?" signal rather than a lower bound.

    Sandbox is off: these tests drive ``_run_merge_phase`` against a
    bare-minimum worktree and exercise gate logic, not sandboxing.
    """
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        max_merge_retries=1,
        max_pre_merge_retries=1,
        steward_completion_timeout=0.5,
        timeouts=TimeoutsConfig(steward=0.5),
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
        sandbox=SandboxConfig(enabled=False),
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
            'status': 'in-progress',
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


_PASSING_VERIFY_RESULT = VerifyResult(
    passed=True,
    test_output='',
    lint_output='',
    type_output='',
    summary='all green',
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_advanced_worktree(git_ops: GitOps, task_id: str) -> Path:
    """Create a worktree on a task branch ONE commit ahead of main.

    ``_worktree_has_wip_commits`` must derive True — it is the wip probe the
    production ``_ensure_steward_started`` injects into the steward, and the
    steward give-up branches these tests model are wip-gated.
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
    """Wire a TaskWorkflow with all fakes, around a REAL ``EscalationQueue``.

    The real queue is deliberate: the merge-entry gate calls the real
    ``_is_gating_escalation`` over the real ``get_by_task`` result, so a fake
    queue would let the gate policy drift out of the test's reach.

    ``escalation_event`` is passed at construction exactly as the harness does
    at dispatch time (``_register_escalation_event``), so
    :func:`_wire_resolve_callback` always has a live event to set.

    *event_store* is optional because only boundary #3 asserts on the
    ``steward_wait_timeout`` emission; ``_wait_for_resolution``'s emit is
    guarded by ``if self.event_store``, so ``None`` is the normal
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


def _wire_resolve_callback(queue: EscalationQueue, workflow: TaskWorkflow) -> None:
    """Wire the queue's resolve callback to the workflow's escalation event.

    Mirrors ``harness._on_escalation_resolved``, which does exactly
    ``self._escalation_events[escalation.task_id].set()``.  This is the ONLY
    thing that wakes ``_wait_for_resolution`` in production, so a test that
    omits it does not reproduce the production wake path at all.
    """
    def _on_resolved(esc) -> None:  # noqa: ARG001 — signature parity with harness
        event = workflow._escalation_event
        if event is not None:
            event.set()

    queue.set_resolve_callback(_on_resolved)


def _time_the_wait(workflow: TaskWorkflow) -> dict[str, float]:
    """Record how long ``_wait_for_resolution`` itself blocks for.

    Returns a dict the caller reads AFTER ``_run_merge_phase`` returns.
    ``'elapsed'`` is the CUMULATIVE time spent inside the wait (summed, so a
    hypothetical second entry into the machinery cannot hide a second
    unbounded wait behind a short first one) and is ABSENT if the wait was
    never entered at all.  ``'calls'`` counts entries, which is what the
    single-shot property (spec S5) reads directly.

    The bounded-wait assertions are about the WAIT, not about
    ``_run_merge_phase``: everything after the wait (real git subprocesses in
    a real worktree) is unbounded-by-design wall-clock that scales with
    machine load, and folding it in is what made the equivalent assertion in
    test_workflow_escalated_steward_stall.py fail spuriously under xdist.

    A regression to an unbounded wait is still caught: the caller's
    ``asyncio.wait_for`` backstop cancels, and the ``finally`` here records
    the full rode-the-backstop duration on the way out.
    """
    timing: dict[str, float] = {}
    original = workflow._wait_for_resolution

    async def _timed() -> str:
        started = asyncio.get_running_loop().time()
        timing['calls'] = timing.get('calls', 0) + 1
        try:
            return await original()
        finally:
            timing['elapsed'] = timing.get('elapsed', 0.0) + (
                asyncio.get_running_loop().time() - started
            )

    workflow._wait_for_resolution = _timed  # type: ignore[method-assign]
    return timing


def _steward_wait_timeouts(store: _RecordingEventStore) -> list[dict]:
    """Every ``escalation_resolved`` event carrying the give-up marker.

    ``_wait_for_resolution``'s expiry handler is the ONLY emitter of
    ``outcome='steward_wait_timeout'`` (it reuses the existing
    ``escalation_resolved`` type rather than adding an EventType member), so
    this list being non-empty is the precise, greppable proof that boundary #3
    went through the EXISTING task-3170 give-up machinery rather than some new
    merge-entry-only timeout path.
    """
    return [
        payload['data']
        for name, payload in store.events
        if name == 'escalation_resolved'
        and payload['data'].get('outcome') == 'steward_wait_timeout'
    ]


def _fake_git_run(*, head: str = 'BRANCH-HEAD-SHA'):
    """Async stand-in for ``orchestrator.workflow._run`` (the merge-entry
    ``git rev-parse HEAD``)."""

    async def _stub(cmd, cwd=None, **kwargs):  # noqa: ARG001
        return 0, head + '\n', ''

    return _stub


def _stub_merge_internals(
    wf: TaskWorkflow,
    monkeypatch,
    *,
    submit_outcome: WorkflowOutcome = WorkflowOutcome.DONE,
) -> AsyncMock:
    """Stub everything DOWNSTREAM of the merge-entry gate, and nothing else.

    Lifted from ``test_workflow_merge_phase_liveness.py:465-500`` but
    deliberately WITHOUT that module's ``wf._check_escalations = MagicMock(
    return_value=[])`` line: the real ``_is_gating_escalation`` predicate must
    stay in the loop, since the awaited/not-awaited state of the returned
    ``_submit_to_merge_queue`` mock is precisely the "did the merge proceed
    in-slot?" signal for boundaries #1/#2/#3.

    ``_check_scope_invariant`` is likewise left REAL: ``FakeScheduler.get_task``
    returns ``None`` for an untracked task, which the tripwire's documented
    fail-safe treats as "cannot check" and skips — so it files no escalation of
    its own and cannot contaminate the gate under test.

    Returns the ``_submit_to_merge_queue`` mock.
    """
    monkeypatch.setattr('orchestrator.workflow._run', _fake_git_run())
    monkeypatch.setattr(
        'orchestrator.workflow.run_scoped_verification',
        AsyncMock(return_value=_PASSING_VERIFY_RESULT),
    )
    wf._maybe_defer_as_train_member = AsyncMock(return_value=None)  # type: ignore[method-assign]
    wf._recover_before_merge = AsyncMock(return_value=None)  # type: ignore[method-assign]
    wf.git_ops.get_main_sha = AsyncMock(return_value='MAIN-SHA')  # type: ignore[method-assign]
    wf.git_ops.rebase_onto_main = AsyncMock(return_value=True)  # type: ignore[method-assign]
    submit = AsyncMock(return_value=submit_outcome)
    wf._submit_to_merge_queue = submit  # type: ignore[method-assign]
    return submit


def _submit_gating_l0(queue: EscalationQueue, task_id: str) -> Escalation:
    """Submit a pending blocking level-0 record — the task-2505 tripwire shape.

    ``severity='blocking'`` + ``level=0`` is disjunct 1 of
    ``_is_gating_escalation``, and ``category='infra_issue'`` matches what
    ``_escalate_scope_invariant_violation`` files 12 lines above the gate.
    """
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='orchestrator',
        severity='blocking',
        category='infra_issue',
        summary='plan.files/metadata.files divergence at MERGE entry',
        detail='synthetic tripwire L0 — gates the merge (task 2505 shape)',
        level=0,
    )
    queue.submit(esc)
    return esc


def _submit_born_at_l2(queue: EscalationQueue, task_id: str) -> Escalation:
    """Submit a pending born-at-L2 record (critical / level=2).

    Hits disjuncts 2 AND 3 of ``_is_gating_escalation``, and trips
    ``_wait_for_resolution``'s stop-the-line short-circuit BEFORE the wait
    loop — which is exactly the fast, never-waited-on disposition boundary #2
    pins.
    """
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='implementer',
        severity='critical',
        category='risk_identified',
        summary='Critical risk requiring human review before merge',
        detail='Born at L2 — routes straight to a human',
        level=2,
    )
    queue.submit(esc)
    return esc


def _make_silent_steward() -> type:
    """A steward that publishes NOTHING and leaves its L0 pending.

    The general "producer went silent" shape — the bound must survive it
    REGARDLESS of which producer bug caused it.  Deliberately does not model
    any specific known bug: the bound has to hold for any future early return
    that forgets to dismiss.  Mirrors
    ``test_workflow_escalated_steward_stall.py:329-357``.

    ``instances`` counts constructions, which the single-shot property reads
    to prove the handler did not loop back into the machinery.
    """

    class _FakeSteward:
        instances: list[_FakeSteward] = []

        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None
            self.started = False
            type(self).instances.append(self)

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            self._wip_probe = probe

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            pass

    _FakeSteward.instances = []
    return _FakeSteward


def _make_chained_gating_steward(queue: EscalationQueue, task_id: str) -> type:
    """A steward that resolves its own L0 then files a FRESH blocking L0.

    Adapted from ``_make_resolve_then_dismiss_chained_steward``
    (test_workflow_e2e.py:1568) — a REAL observed steward behaviour, not a
    synthetic edge case: ``TaskSteward`` routinely chains a follow-on
    ``infra_issue`` while resolving a ``task_failure``.

    The resolve fires the queue's resolve callback and wakes the waiter, so by
    the time the merge-entry handler re-checks the gate there is a NEW gating
    record open that it never waited on.  That is the shape the single-shot
    bound (spec S5) exists to cap.
    """

    class _FakeSteward:
        instances: list[_FakeSteward] = []

        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None
            self.started = False
            type(self).instances.append(self)

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            self._wip_probe = probe

        async def start(self) -> None:
            self.started = True
            pending = queue.get_by_task(task_id, status='pending', level=0)
            assert pending, 'expected a pending L0 to chain off'
            for esc in pending:
                queue.resolve(
                    esc.id, 'Resolved by chained FakeSteward',
                    resolved_by='fake-steward',
                )
            chained = Escalation(
                id=queue.make_id(task_id),
                task_id=task_id,
                agent_role='steward',
                severity='blocking',
                category='infra_issue',
                summary='Chained follow-on L0 from steward',
                detail='steward found a follow-on infra issue while resolving',
                level=0,
            )
            queue.submit(chained)

        async def stop(self) -> None:
            pass

    _FakeSteward.instances = []
    return _FakeSteward


# ---------------------------------------------------------------------------
# Boundary #1 — steward consumes the L0 → the merge retries IN-SLOT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeGateBailEntersBoundedWait:
    """Boundary #1: a gating L0 at MERGE entry enters the bounded wait, and a
    steward that consumes it lets the merge proceed IN-SLOT.

    Pre-γ₁ the gate did ``return WorkflowOutcome.ESCALATED`` with no steward
    started and no status write: the task exited the slot, the harness nulled
    its claimant, and the row sat ``in-progress`` forever (spec E1).  The
    corrected disposition burns no re-dispatch at all — the row stays
    ``in-progress`` under the live claimant for the whole wait (spec §4:
    ``in-progress`` covers Path-A escalated-waiting) and the merge simply
    continues once the record is adjudicated.
    """

    async def test_resolved_l0_falls_through_to_the_merge_in_slot(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        wf, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree,
        )
        _wire_resolve_callback(queue, wf)
        submit = _stub_merge_internals(wf, monkeypatch)

        _submit_gating_l0(queue, task_id)
        wf._steward_factory = _make_resolving_steward(queue, task_id)

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 20)

        # Fell through to the SUCCESS tail rather than bailing out of the slot.
        assert outcome is None, (
            f'a consumed gating L0 must fall through to the merge, not exit '
            f'the slot; got {outcome}'
        )
        # ...and the merge genuinely ran, in this same slot.
        assert submit.await_count == 1, (
            f'expected exactly one in-slot merge submission; got '
            f'{submit.await_count}'
        )
        # The steward was actually started — _ensure_steward_started is on
        # this path, not skipped (the resolving fake asserts a pending L0
        # exists when its start() runs, so this also pins the ordering).
        assert wf._steward is not None, (
            'the merge-entry gate must start a steward, exactly as run()\'s '
            'ESCALATED branch does'
        )
        # The steward consumed its own L0 — nothing left pending to re-strand
        # the next entry.
        assert queue.get_by_task(task_id, status='pending') == []
        # No re-dispatch burned: the row was never re-pended and never parked.
        written = scheduler.statuses.get(task_id, [])
        assert 'pending' not in written, (
            f'the in-slot wait must not re-pend the task; got {written}'
        )
        assert 'blocked' not in written, (
            f'a resolved gate must not park the task; got {written}'
        )


# ---------------------------------------------------------------------------
# Boundary #2 — born-at-L2 short-circuits to a visible ``blocked`` park
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeGateL2ShortCircuits:
    """Boundary #2: a born-at-L2 (or level≥2) record open at MERGE entry parks
    the row ``blocked`` immediately — it is never waited on in-slot.

    ``_wait_for_resolution``'s stop-the-line check (``workflow.py:12206-12217``)
    raises ``_StewardReescalated`` BEFORE the wait loop, so the handler's
    ``except`` clause routes straight to ``_mark_blocked(...,
    skip_escalation=True)``.  Two properties matter equally: the row IS
    written (no strand, spec S1) and the human-facing L2 is NOT consumed
    (repend-PRD boundary B1 stop-the-line — ``_mark_blocked``'s straggler
    cleanup dismisses level-0 only).
    """

    async def test_born_at_l2_blocks_without_merging_or_consuming_the_record(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        wf, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree,
        )
        _wire_resolve_callback(queue, wf)
        submit = _stub_merge_internals(wf, monkeypatch)

        _submit_born_at_l2(queue, task_id)
        # Must never be needed: with no pending L0, _ensure_steward_started
        # returns without constructing anything, and the short-circuit fires
        # before the wait loop.
        silent = _make_silent_steward()
        wf._steward_factory = silent

        # The short-circuit sits BEFORE the wait loop, so a human-gated record
        # is never waited on in-slot.  5s is far below the 1.0s-window-times-
        # anything a regression to "wait on the L2" would cost, and far above
        # the real path's cost.
        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 5)

        assert outcome is not None, (
            'a born-at-L2 record must terminate the slot, not fall through '
            'to the merge'
        )
        # The row is written BEFORE the return (spec S1: a status write
        # precedes every terminal return).
        assert scheduler.statuses.get(task_id, [])[-1:] == ['blocked'], (
            f'expected the row parked blocked before the return; got '
            f'{scheduler.statuses.get(task_id)}'
        )
        # D4 stop-the-line: nothing merges past an open gating record.
        assert submit.await_count == 0, (
            'the merge must not be attempted while a born-at-L2 record is open'
        )
        # The gate does not consume the human's record.
        still_pending = queue.get_by_task(task_id, status='pending', level=2)
        assert len(still_pending) == 1, (
            f'the pending L2 must survive the gate; got {still_pending}'
        )
        assert silent.instances == [], (
            'no steward should be constructed for a record no steward can resolve'
        )


__all__ = [
    'PLAN',
    '_build_workflow',
    '_make_advanced_worktree',
    '_make_chained_gating_steward',
    '_make_resolving_steward',
    '_make_silent_steward',
    '_steward_wait_timeouts',
    '_stub_merge_internals',
    '_submit_born_at_l2',
    '_submit_gating_l0',
    '_time_the_wait',
    '_wire_resolve_callback',
]
