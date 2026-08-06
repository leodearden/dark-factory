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
import re
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
from orchestrator.task_status import WORKFLOW_PRESERVE_STATUSES
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome
from orchestrator.workflow_types import (
    StewardResolved,
    WorkflowState,
    outcome_allows_status,
)

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
    its own and cannot contaminate the gate under test.  That default is what
    lets every OTHER test hand-submit its own record; populating
    ``scheduler.task_data[task_id]`` with a divergent ``metadata.files`` turns
    the tripwire back on, which is exactly what
    :class:`TestScopeInvariantTripwireGatesTheMerge` does to pin the
    escalate-then-gate coupling itself.

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


def _make_terminal_decision_steward(
    queue: EscalationQueue,
    task_id: str,
    scheduler,
    status: str,
    *,
    resolve: bool = True,
) -> type:
    """A steward that takes a TERMINAL DECISION on the row while resolving.

    The steward is the ONLY workflow role holding
    ``mcp__fused-memory__set_task_status`` (``roles.py:1131-1133``), and its
    prompt (``roles.py:1277+``) explicitly directs it to mark tasks ``done``
    via ``kind='found_on_main'``.  ``run()``'s own guard names the wider case
    verbatim: the steward "may have set the task to done / cancelled /
    deferred / blocked while resolving the L0 (e.g. queued a follow-up task
    that is now the durable fix and deferred this one onto it)"
    (``workflow.py:2685-2696``).

    Such a decision IS an adjudication of the gate, and honouring it costs
    nothing.  Ignoring it at MERGE entry is strictly worse than the
    wasted-dispatch consequence ``run()`` guards against: the branch LANDS ON
    MAIN, and for a non-terminal park like ``deferred`` / ``merge-deferred``
    (which are NOT in ``TERMINAL_STATUSES``, so no server rejection fires)
    ``_finalise_merged_done`` then silently overwrites the row to ``done`` —
    a park erased unrecoverably, since re-dispatch cannot un-merge a branch.

    *resolve* controls whether the gating record is ALSO consumed.  With
    ``resolve=False`` the decision is taken but the record is left open, so the
    only wake is the wait's own expiry — which is what makes the guard's
    PLACEMENT observable: both sibling branches that case would otherwise
    reach call ``_mark_blocked(..., escalate_to_human=True)``, filing an L1 for
    a task the steward deliberately parked.

    The work runs in a background ``_chain()`` task after a short delay, for
    the same reason :func:`_make_chained_gating_steward` does: a synchronous
    ``start()`` would leave it already done at wait entry, so the waiter would
    never block and the production wake path would go unexercised.
    """

    class _FakeSteward:
        instances: list[_FakeSteward] = []

        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None
            self.started = False
            self.chain_task: asyncio.Task | None = None
            type(self).instances.append(self)

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            self._wip_probe = probe

        async def _chain(self) -> None:
            await asyncio.sleep(0.05)
            # The row write comes FIRST and through the scheduler, exactly as a
            # real steward's set_task_status MCP call does — not through any
            # workflow-side helper.
            await scheduler.set_task_status(task_id, status)
            if not resolve:
                return
            pending = queue.get_by_task(task_id, status='pending', level=0)
            assert pending, 'expected a pending L0 to adjudicate'
            for esc in pending:
                queue.resolve(
                    esc.id, f'Steward parked the task {status}',
                    resolved_by='fake-steward',
                )

        async def start(self) -> None:
            self.started = True
            self.chain_task = asyncio.create_task(self._chain())

        async def stop(self) -> None:
            if self.chain_task is not None:
                await self.chain_task

    _FakeSteward.instances = []
    return _FakeSteward


def _make_chained_gating_steward(queue: EscalationQueue, task_id: str) -> type:
    """A steward that resolves its own L0 then files a FRESH blocking L0.

    Adapted from ``_make_resolve_then_dismiss_chained_steward``
    (test_workflow_e2e.py:1568) — a REAL observed steward behaviour, not a
    synthetic edge case: ``TaskSteward`` routinely chains a follow-on
    ``infra_issue`` while resolving a ``task_failure``.

    The resolve fires the queue's resolve callback and wakes the waiter, which
    then finds a NEW gating record in place of the one it was waiting on.  A
    fresh level-0 is still within the WAIT's reach — its loop re-polls level-0
    on every wake, inside the SAME deadline — so this shape costs exactly one
    window and then parks via the expiry branch.  Contrast
    :func:`_make_chained_l2_steward`, whose follow-on is invisible to the wait
    entirely.  Between them they cover both halves of the single-shot bound
    (spec S5).

    Runs in the background after a short delay for the same reason
    :func:`_make_chained_l2_steward` does: a synchronous ``start()`` would
    leave the chained record already pending at wait entry, so the waiter
    would never block and the production wake path would go unexercised.
    """

    class _FakeSteward:
        instances: list[_FakeSteward] = []

        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None
            self.started = False
            self.chain_task: asyncio.Task | None = None
            type(self).instances.append(self)

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            self._wip_probe = probe

        async def _chain(self) -> None:
            await asyncio.sleep(0.05)
            pending = queue.get_by_task(task_id, status='pending', level=0)
            assert pending, 'expected a pending L0 to chain off'
            for esc in pending:
                queue.resolve(
                    esc.id, 'Resolved by chained FakeSteward',
                    resolved_by='fake-steward',
                )
            queue.submit(Escalation(
                id=queue.make_id(task_id),
                task_id=task_id,
                agent_role='steward',
                severity='blocking',
                category='infra_issue',
                summary='Chained follow-on L0 from steward',
                detail='steward found a follow-on infra issue while resolving',
                level=0,
            ))

        async def start(self) -> None:
            self.started = True
            self.chain_task = asyncio.create_task(self._chain())

        async def stop(self) -> None:
            if self.chain_task is not None:
                await self.chain_task

    _FakeSteward.instances = []
    return _FakeSteward


def _make_journalling_resolving_steward(
    queue: EscalationQueue, task_id: str, events: list[str],
) -> type:
    """``_make_resolving_steward`` plus a lifecycle journal.

    Same behaviour as the shared helper — ``start()`` resolves the pending L0
    and publishes ``StewardResolved`` — but both lifecycle edges append to
    *events*, so a test can assert the stop edge lands BEFORE the first
    worktree-touching merge step rather than merely observing the end state.

    The ordering is the whole point: a real ``TaskSteward.stop()`` cancels a
    persistent poll loop that invokes its agent with ``cwd = worktree``
    (``steward.py:209-217``, ``:542``, ``:590``), the same worktree the merge
    tail rebases and verifies in.
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
            events.append('steward:start')
            pending = queue.get_by_task(task_id, status='pending', level=0)
            assert pending, 'expected at least one pending L0 to resolve'
            for esc in pending:
                queue.resolve(
                    esc.id, 'Resolved by journalling FakeSteward',
                    resolved_by='fake-steward',
                )
            if self._outcome_channel is not None:
                self._outcome_channel.put_nowait(
                    StewardResolved(resolution_text='Resolved by FakeSteward'),
                )

        async def stop(self) -> None:
            events.append('steward:stop')

    return _FakeSteward


def _make_chained_l2_steward(queue: EscalationQueue, task_id: str) -> type:
    """A steward that resolves its own L0 then files a fresh born-at-L2.

    The same chained shape as :func:`_make_chained_gating_steward`, but the
    follow-on record is one the WAIT provably never waits on — which is what
    makes it the load-bearing case for the post-wait re-check:

    * ``_wait_for_resolution``'s loop polls ``get_by_task(..., level=0)``, so a
      level-2 record is invisible to it;
    * its entry-time ``pending_high_sev`` stop-the-line check already ran,
      before this record existed;
    * its tail's ``has_open_l1`` matches level-1 only.

    So the wait returns NORMALLY with no expiry, and without a re-check the
    handler resumes the merge straight past an open, human-gated record — a D4
    stop-the-line violation.  (A chained plain-blocking L0, by contrast, IS
    caught by the wait loop, which re-polls it inside the SAME deadline; see
    :func:`_make_chained_gating_steward`.)

    Mirrors ``TaskSteward._auto_escalate_to_human``'s real behaviour of filing
    a human-facing record while handing off.

    The chain runs as a BACKGROUND task after a short delay, not inside
    ``start()``: a real steward works while the workflow waits, so a record it
    files lands AFTER ``_wait_for_resolution``'s entry-time checks have already
    run.  A synchronous ``start()`` would file it before the wait is even
    entered, where the stop-the-line check catches it and the post-wait gap is
    never exercised at all.  (Same reasoning, and the same delay-then-wake
    shape, as ``_make_dismiss_and_publish_steward`` in
    test_workflow_escalated_steward_stall.py.)
    """

    class _FakeSteward:
        instances: list[_FakeSteward] = []

        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None
            self.started = False
            self.chain_task: asyncio.Task | None = None
            type(self).instances.append(self)

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            self._wip_probe = probe

        async def _chain(self) -> None:
            await asyncio.sleep(0.05)
            pending = queue.get_by_task(task_id, status='pending', level=0)
            assert pending, 'expected a pending L0 to chain off'
            # resolve-then-submit with NO await between: the waiter cannot
            # resume in the gap, so by the time it runs again the L0 is gone
            # and the L2 is open — exactly the state a real steward hand-off
            # leaves behind.
            for esc in pending:
                queue.resolve(
                    esc.id, 'Resolved by chained-L2 FakeSteward',
                    resolved_by='fake-steward',
                )
            queue.submit(Escalation(
                id=queue.make_id(task_id),
                task_id=task_id,
                agent_role='steward',
                severity='critical',
                category='risk_identified',
                summary='Chained born-at-L2 filed while resolving',
                detail='steward escalated straight to a human mid-resolution',
                level=2,
            ))

        async def start(self) -> None:
            self.started = True
            self.chain_task = asyncio.create_task(self._chain())

        async def stop(self) -> None:
            if self.chain_task is not None:
                await self.chain_task

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
        # this path, not skipped.  Asserted through its EFFECT rather than
        # `wf._steward is not None`, which the stop-before-merge below
        # deliberately falsifies: the resolving fake asserts a pending L0
        # exists when its start() runs and resolves it, so an empty pending
        # set can only mean start() ran.
        assert queue.get_by_task(task_id, status='pending', level=0) == [], (
            'the merge-entry gate must start a steward, exactly as run()\'s '
            'ESCALATED branch does'
        )
        # ...and it was STOPPED AND CLEARED before the merge tail ran.  A
        # TaskSteward poll loop invokes its agent with cwd = the worktree the
        # merge tail rebases/verifies/commits in, so leaving it live here is
        # the two-agents-one-worktree hazard _wait_for_resolution's give-up
        # branch calls out as a safety requirement.  Sibling assertion in
        # test_steward_is_stopped_before_the_merge_tail_runs below pins the
        # ORDERING; this one pins the end state.
        assert wf._steward is None, (
            'the gate-cleared fall-through must stop and clear the steward '
            'before the merge tail touches the worktree'
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

    async def test_steward_is_stopped_before_the_merge_tail_runs(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        """No steward agent can overlap the merge tail's work in the worktree.

        ``TaskSteward._loop`` (``steward.py:259-273``) is a PERSISTENT poll
        loop: once started it keeps picking up fresh L0s and invoking its agent
        with ``cwd = self.worktree``.  Everything downstream of the gate-cleared
        fall-through — ``_recover_before_merge``, ``git rev-parse HEAD``,
        ``rebase_onto_main``, ``run_scoped_verification``,
        ``_submit_to_merge_queue`` — runs in that same worktree.  Leaving the
        loop live across the fall-through is therefore the
        two-agents-one-git-worktree hazard ``_wait_for_resolution``'s own
        give-up branch calls out as "a safety requirement, not tidiness"
        (``workflow.py:12546-12565``): the rebase either fails on a dirty tree
        / ``index.lock``, or unreviewed steward commits ride the branch onto
        main.

        The exposure is NEW with the bounded wait and worse here than anywhere
        else: pre-γ₁ the gate returned ESCALATED without ever constructing a
        steward, and unlike ``run()``'s ESCALATED branch (which only re-invokes
        the implementer and re-gates later) a bad merge is irrecoverable — no
        re-dispatch un-merges a branch.

        Pins the ORDERING, not just the end state that the sibling test above
        asserts: ``steward:stop`` must precede the FIRST worktree-touching
        merge step.
        """
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        wf, _scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree,
        )
        _wire_resolve_callback(queue, wf)
        _stub_merge_internals(wf, monkeypatch)

        events: list[str] = []

        # Re-stub the merge steps that actually touch ``wf.worktree`` so their
        # invocation order is journalled against the steward's lifecycle.
        async def _recover(*_a, **_k) -> None:
            events.append('merge:_recover_before_merge')

        async def _rebase(*_a, **_k) -> bool:
            events.append('merge:rebase_onto_main')
            return True

        async def _submit(*_a, **_k) -> WorkflowOutcome:
            events.append('merge:_submit_to_merge_queue')
            return WorkflowOutcome.DONE

        wf._recover_before_merge = _recover  # type: ignore[method-assign]
        wf.git_ops.rebase_onto_main = _rebase  # type: ignore[method-assign]
        wf._submit_to_merge_queue = _submit  # type: ignore[method-assign]

        _submit_gating_l0(queue, task_id)
        wf._steward_factory = _make_journalling_resolving_steward(
            queue, task_id, events,
        )

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 20)

        assert outcome is None, (
            f'the gate cleared, so the merge must run in-slot; got {outcome}'
        )
        assert 'steward:start' in events, 'the gate must start a steward'
        assert 'steward:stop' in events, (
            f'the gate-cleared fall-through must STOP the steward before '
            f'handing the worktree to the merge tail; journal: {events}'
        )
        merge_steps = [i for i, e in enumerate(events) if e.startswith('merge:')]
        assert merge_steps, f'the merge tail never ran; journal: {events}'
        assert events.index('steward:stop') < merge_steps[0], (
            f'the steward was still live when the merge tail began touching '
            f'the worktree — two agents, one git worktree; journal: {events}'
        )
        # And the reference is cleared, so a later _mark_blocked on the merge
        # tail builds a FRESH steward rather than awaiting a cancelled loop
        # that can never publish.
        assert wf._steward is None, (
            'stop-then-CLEAR is the established idiom (workflow.py:6807-6809 '
            '/ :6851-6853 / :12579-12582)'
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


# ---------------------------------------------------------------------------
# Boundary #3 — the wait expires (silent producer) → blocked + L1, no strand
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeGateWaitExpiry:
    """Boundary #3: the gating L0 is never consumed within the idle window.

    The EXISTING task-3170 give-up machinery does the cleanup — stop the
    steward, emit ``escalation_resolved{outcome='steward_wait_timeout'}``,
    dismiss the orphan L0(s) — and then ``_wait_for_resolution`` returns
    normally, indistinguishable from a genuine resolution by its ``str``
    return.  The merge-entry gate must NOT read that as an adjudication: an
    expired wait means nobody DECIDED anything, and repend-PRD D4
    stop-the-line forbids merging past the record.  The disposition is a
    visible ``blocked`` park with an L1 for a human.

    This is the failure mode the bound exists for at all — a producer that
    went silent — so the assertions cover both halves: that the shared
    machinery really ran (not some new merge-entry-only timeout path), and
    that the merge-entry handler drew the right conclusion from it.
    """

    async def test_expired_wait_blocks_with_l1_instead_of_merging(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        store = _RecordingEventStore()
        wf, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree, event_store=store,
        )
        _wire_resolve_callback(queue, wf)
        submit = _stub_merge_internals(wf, monkeypatch)
        timing = _time_the_wait(wf)

        gating = _submit_gating_l0(queue, task_id)
        # Publishes nothing, dismisses nothing — the general "producer went
        # silent" shape, deliberately not modelling any one known bug.
        wf._steward_factory = _make_silent_steward()

        # 10s backstop against a regression to an unbounded wait; the derived
        # window is 0.5 + 0.5 = 1.0s, so the real path never approaches it.
        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 10)

        # ── the EXISTING give-up machinery ran ──────────────────────────
        assert timing.get('elapsed', 0.0) < 5, (
            f'the wait itself must be bounded by roughly one derived window '
            f'(0.5 + 0.5 = 1.0s), not ride the test backstop; took '
            f'{timing.get("elapsed")}s'
        )
        timeouts = _steward_wait_timeouts(store)
        assert timeouts, (
            'expiry must go through _wait_for_resolution\'s give-up branch — '
            'the only emitter of escalation_resolved{outcome='
            "'steward_wait_timeout'} — not through a merge-entry-only path"
        )
        assert gating.id in timeouts[0]['escalation_ids']
        # Its orphan-dismissal ran, so the next entry cannot re-strand on the
        # same record.
        assert queue.get_by_task(task_id, status='pending', level=0) == []
        archived = queue.get(gating.id)
        assert archived is not None
        assert archived.resolved_by == 'auto-dismissed'

        # ── the merge-entry disposition ─────────────────────────────────
        assert outcome is not None, (
            'an expired wait means the gating record was never adjudicated — '
            'the merge must not resume'
        )
        assert submit.await_count == 0, (
            'D4 stop-the-line: merging past a never-adjudicated gating record '
            'is exactly what the gate exists to prevent'
        )
        assert scheduler.statuses.get(task_id, [])[-1:] == ['blocked']
        assert queue.has_open_l1(task_id), (
            'a silent steward has proved it cannot help — the park must be '
            'visible to a human as an L1, not a fresh L0 for the same steward'
        )

        # ── the no-strand shape, stated explicitly ──────────────────────
        # harness._run_slot's finally (harness.py:7949-7972) clears the
        # claimant unconditionally and writes NO status, so whatever run()
        # left behind survives.  A status write happened here, so the row
        # cannot be left `in-progress` with a nulled claimant.
        assert scheduler.statuses.get(task_id), (
            'no exit of the merge-entry gate may reach the harness slot-finally '
            'without having written its successor status'
        )


# ---------------------------------------------------------------------------
# Boundary #4 — a steward TERMINAL DECISION is honoured, not merged past
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardTerminalDecisionAtMergeGate:
    """Boundary #4: a terminal decision the steward takes WHILE resolving the
    merge-entry gate is an adjudication of that gate — honour it, never merge
    past it.

    The merge-entry handler is documented as a deliberate mirror of ``run()``'s
    ESCALATED branch, and ``run()`` ends its wait with exactly this guard
    (``workflow.py:2685-2709``): read ``scheduler.get_status`` and return DONE
    on ``'done'`` / BLOCKED on ``WORKFLOW_PRESERVE_STATUSES``.  The steward is
    the only workflow role holding ``set_task_status``, so this is not a
    hypothetical race — it is the documented, prompted behaviour of the very
    producer this gate waits on.

    Checking only whether gating RECORDS are open is not enough: a steward that
    resolves the record AND parks the row leaves ``still_gating`` empty and
    ``_steward_wait_expired`` False, so the gate falls straight through to
    ``_recover_before_merge`` → the retry loop → ``_submit_to_merge_queue``.
    The measured consequences, per park status:

    * ``deferred`` / ``merge-deferred`` — not in ``TERMINAL_STATUSES``, so no
      server rejection fires: the branch lands and ``_finalise_merged_done``
      overwrites the row to ``done``.  The steward's park is silently ERASED,
      and unlike ``run()``'s wasted-dispatch consequence this is not
      recoverable by re-dispatch — the branch is already on main.
    * ``cancelled`` — the branch lands, ``set_task_status(..., 'done')`` is
      rejected ``terminal_exit_rejected``, but ``scheduler.py:2303-2306``
      swallows the rejection for terminal targets, so the row stays
      ``cancelled`` while ``run()`` returns DONE — and the run()-exit SM-2
      check (``workflow.py:3248``) then raises ``AssertionError``, because
      ``outcome_allows_status(DONE, 'cancelled')`` is False.  A post-merge
      crash exit.
    * ``done`` — the branch is merged a SECOND time, for work the steward
      already declared covered.

    ``merge_queue.py``'s ``WORKFLOW_PRESERVE_STATUSES`` guard does not catch
    any of this: it sits inside ``reconcile_landed_row`` (the crash-recovery
    outbox path), not on the enqueue path.  Nothing downstream catches it.
    """

    @pytest.mark.parametrize('status', ['cancelled', 'deferred', 'done'])
    async def test_terminal_decision_is_honoured_not_merged_past(
        self, status, config, git_ops, task_assignment, monkeypatch, tmp_path,
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
        wf._steward_factory = _make_terminal_decision_steward(
            queue, task_id, scheduler, status,
        )

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 20)

        # The headline: the branch was NOT merged.
        assert submit.await_count == 0, (
            f'the steward parked the row {status!r} while resolving the gate — '
            f'that is an adjudication, and merging past it lands the branch on '
            f'main irrecoverably'
        )
        assert outcome is not None, (
            'a steward terminal decision must terminate the slot, not fall '
            'through to the merge'
        )
        expected = (
            WorkflowOutcome.DONE if status == 'done' else WorkflowOutcome.BLOCKED
        )
        assert outcome == expected, f'expected {expected}; got {outcome}'
        # The steward's decision survives verbatim — in particular the
        # cancelled/deferred rows are NOT overwritten to 'done'.
        assert scheduler.statuses[task_id][-1] == status, (
            f'the steward\'s {status!r} decision must survive; got '
            f'{scheduler.statuses[task_id]}'
        )
        # run()'s exit assertion `report.phase == self.machine.state`
        # (workflow.py:3228) requires the phase to be entered before the
        # terminal return.
        assert wf.machine.state == (
            WorkflowState.DONE if status == 'done' else WorkflowState.BLOCKED
        ), f'phase not entered before the terminal return; at {wf.machine.state}'
        # The run()-exit SM-2 consistency check (workflow.py:3248), pinned
        # directly and cheaply without driving full run().  This is what makes
        # the guard's branch ORDER load-bearing: WORKFLOW_PRESERVE_STATUSES
        # CONTAINS 'done', so a guard testing membership first would return
        # BLOCKED for a done row — and outcome_allows_status(BLOCKED, 'done')
        # is False, which crashes run() on the way out.
        assert outcome_allows_status(outcome, scheduler.statuses[task_id][-1]), (
            f'{outcome} is not a legal pairing with row status '
            f'{scheduler.statuses[task_id][-1]!r} — run()\'s SM-2 exit check '
            f'would raise'
        )

    async def test_terminal_decision_wins_over_a_still_open_gating_record(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        """The guard's PLACEMENT, pinned: it must precede BOTH block branches.

        Here the steward parks the row and then goes silent on the queue, so
        the wake comes from the wait's own expiry.  Without the guard running
        FIRST, this lands in either the ``_steward_wait_expired`` branch or the
        ``still_gating`` re-check — both of which call ``_mark_blocked(...,
        escalate_to_human=True)``, filing an L1 for a task the steward
        deliberately parked and attempting ``set_task_status('blocked')`` over
        a terminal row (routing into ``_handle_terminal_exit_rejection``'s
        bypass-done detection and risking a spurious ``bypass_done`` L1).
        """
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        wf, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree,
        )
        _wire_resolve_callback(queue, wf)
        submit = _stub_merge_internals(wf, monkeypatch)

        _submit_gating_l0(queue, task_id)
        wf._steward_factory = _make_terminal_decision_steward(
            queue, task_id, scheduler, 'cancelled', resolve=False,
        )

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 20)

        assert submit.await_count == 0, (
            'D4 stop-the-line: the record is still open AND the steward parked '
            'the row — the merge must not proceed on either count'
        )
        assert outcome == WorkflowOutcome.BLOCKED, f'got {outcome}'
        assert scheduler.statuses[task_id][-1] == 'cancelled', (
            f'the steward\'s cancel must not be overwritten by the gate\'s own '
            f'park; got {scheduler.statuses[task_id]}'
        )
        assert queue.has_open_l1(task_id) is False, (
            'no L1 may be filed for a task the steward deliberately parked — '
            'the decision IS the adjudication, so there is nothing to ask a '
            'human about'
        )


# ---------------------------------------------------------------------------
# The no-strand exit property, across every disposition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'disposition',
    ['resolving_steward', 'born_at_l2', 'silent_steward', 'terminal_decision'],
)
class TestNoStrandExitProperty:
    """Spec S1, stated as one property over every boundary:

        ``_run_merge_phase`` either returns ``None`` — it fell through to the
        in-slot merge with the slot still held and the claimant still live —
        or the row carries a status the workflow must not overwrite
        (``WORKFLOW_PRESERVE_STATUSES``) before the terminal return.

    That status is written either by ``_mark_blocked`` on this path (the
    ``blocked`` parks of boundaries #2/#3) or by the STEWARD and deliberately
    preserved (boundary #4's terminal decision).  Both satisfy the property
    the strand is the absence of: something durable is on the row.

    Never a terminal return with the row left at ``in-progress``.  That pair
    is the strand: ``harness._run_slot``'s ``finally``
    (harness.py:7949-7972) clears the claimant unconditionally and writes NO
    status, so whatever ``run()`` left behind survives — and a steward-less
    ESCALATED left ``in-progress``.  (That harness test's parametrize list
    covers DONE/CANCELLED/BLOCKED/REQUEUED but not ESCALATED, which is why
    the property is pinned workflow-side rather than by extending it.)

    Deliberately phrased as "a status write precedes every terminal return"
    and NOT as "``WorkflowOutcome.ESCALATED`` is never returned":
    ``_mark_blocked`` legitimately returns ESCALATED on its
    ``StewardReescalatedL1`` branch (workflow.py:14077) AFTER
    ``_enter_phase(BLOCKED)`` and ``set_task_status('blocked')`` have run, and
    spec §5 explicitly sanctions that shape ("a run() exit with ESCALATED
    means the wait concluded in re-escalation → ``_mark_blocked`` ran").  A
    test that banned the outcome outright would fight a correct path.
    """

    async def test_status_write_precedes_every_terminal_return(
        self, disposition, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        wf, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree,
        )
        _wire_resolve_callback(queue, wf)
        _stub_merge_internals(wf, monkeypatch)

        if disposition == 'born_at_l2':
            _submit_born_at_l2(queue, task_id)
            wf._steward_factory = _make_silent_steward()
        elif disposition == 'terminal_decision':
            _submit_gating_l0(queue, task_id)
            wf._steward_factory = _make_terminal_decision_steward(
                queue, task_id, scheduler, 'cancelled',
            )
        else:
            _submit_gating_l0(queue, task_id)
            wf._steward_factory = (
                _make_resolving_steward(queue, task_id)
                if disposition == 'resolving_steward'
                else _make_silent_steward()
            )

        before = list(scheduler.statuses.get(task_id, []))
        assert before == [], 'precondition: nothing written yet'

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 20)
        after = scheduler.statuses.get(task_id, [])

        if outcome is None:
            # Fell through to the in-slot merge: the slot is still held and the
            # claimant is still live, so there is nothing for the harness
            # slot-finally to strand.
            assert 'blocked' not in after
        else:
            assert after, (
                f'{disposition}: terminal return {outcome} with NO status ever '
                f'written — the harness slot-finally would null the claimant '
                f'and leave the row in-progress (spec E1 strand)'
            )
            assert after[-1] in WORKFLOW_PRESERVE_STATUSES, (
                f'{disposition}: a terminal exit must leave a status the '
                f'workflow must not overwrite — written by _mark_blocked here, '
                f'or by the steward and preserved; got {after}'
            )

        # The exact defect, named as a negative: ESCALATED + no status write is
        # the strand shape spec E1 describes, and it must be unreachable.
        assert not (outcome == WorkflowOutcome.ESCALATED and not after), (
            f'{disposition}: returned ESCALATED with no status write — the '
            f'steward-less ESCALATED exit is back'
        )

        # A surviving orphan L0 re-strands the NEXT entry on the same record
        # for another full window, so no exit may leave one pending.  Exempt
        # for `terminal_decision` only: a steward that parks the row terminally
        # has ended the workflow, so any record it leaves belongs to the
        # orphan-L0 reaper — exactly as run()'s own guard leaves it when it
        # bypasses _mark_blocked (and thus _mark_blocked's straggler cleanup).
        if disposition != 'terminal_decision':
            assert queue.get_by_task(task_id, status='pending', level=0) == [], (
                f'{disposition}: a pending level-0 record survived the exit'
            )


# ---------------------------------------------------------------------------
# The gate re-fire is SINGLE-SHOT, not a loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGateReFireIsBounded:
    """A gating record filed WHILE the steward resolves must park the task,
    not buy another idle window.

    ``_make_chained_gating_steward`` models a real observed steward behaviour:
    ``TaskSteward`` routinely resolves the escalation it was given and chains
    a follow-on ``infra_issue`` of its own.  The resolve wakes the waiter, so
    by the time the merge-entry handler looks again there is a NEW gating
    record open that nothing has waited on.

    Both properties matter:

    * D4 stop-the-line still holds — nothing merges past an open gating
      record, INCLUDING one filed during the resolution.
    * Spec S5 boundedness — the whole merge-entry gate path costs at most ONE
      derived idle window.  Looping back into the machinery would let a
      steward that keeps chaining fresh L0s hold the workflow slot for an
      unbounded number of windows, which is exactly the unbounded-hold defect
      this task exists to remove.  ``run()``'s ESCALATED branch legitimately
      loops because each pass re-invokes the implementer and can make
      progress; the merge-entry gate does no work between passes, so a second
      wait is pure latency with no new information.
    """

    async def test_chained_l2_is_not_merged_past(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        """The load-bearing case: a chained record the WAIT never waits on.

        ``_wait_for_resolution`` polls level-0 records only, ran its
        stop-the-line check before this record existed, and matches level-1 in
        its ``has_open_l1`` tail — so a level-2 filed mid-resolution slips
        through all three and the wait returns normally with no expiry.  Only
        a post-wait re-check of the SAME ``_is_gating_escalation`` predicate
        catches it.
        """
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        wf, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree,
        )
        _wire_resolve_callback(queue, wf)
        submit = _stub_merge_internals(wf, monkeypatch)
        timing = _time_the_wait(wf)

        _submit_gating_l0(queue, task_id)
        chained = _make_chained_l2_steward(queue, task_id)
        wf._steward_factory = chained

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 20)

        assert submit.await_count == 0, (
            'D4 stop-the-line: a born-at-L2 filed during the resolution is an '
            'open gating record — the merge must not proceed past it'
        )
        assert outcome is not None
        assert scheduler.statuses.get(task_id, [])[-1:] == ['blocked'], (
            f'the task must park visibly for the human the L2 is addressed to; '
            f'got {scheduler.statuses.get(task_id)}'
        )
        # The human's record survives — the re-check parks, it does not consume.
        assert len(queue.get_by_task(task_id, status='pending', level=2)) == 1
        # Single-shot: the re-check parks rather than buying a second window.
        assert timing.get('calls') == 1, (
            f'the merge-entry gate must be single-shot; entered the steward '
            f'wait {timing.get("calls")} time(s)'
        )
        assert len(chained.instances) <= 1

    async def test_chained_gating_record_parks_instead_of_re_waiting(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        """The plain-L0 chain: caught by the wait loop's own re-poll, inside
        the SAME deadline, then parked by the expiry branch.  Pinned here so
        the two chained shapes stay legibly distinct — this one costs one
        window and needs no re-check; the L2 one above needs the re-check and
        costs none."""
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        wf, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree,
        )
        _wire_resolve_callback(queue, wf)
        submit = _stub_merge_internals(wf, monkeypatch)
        timing = _time_the_wait(wf)

        _submit_gating_l0(queue, task_id)
        chained = _make_chained_gating_steward(queue, task_id)
        wf._steward_factory = chained

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 20)

        assert submit.await_count == 0, (
            'D4 stop-the-line: a gating record filed during the resolution '
            'still gates — nothing merges past it'
        )
        assert outcome is not None
        assert scheduler.statuses.get(task_id, [])[-1:] == ['blocked'], (
            f'the task must park visibly rather than merge; got '
            f'{scheduler.statuses.get(task_id)}'
        )

        # Boundedness: exactly ONE wait was entered.  _time_the_wait sums
        # elapsed across calls precisely so a second wait cannot hide behind a
        # short first one.
        assert timing.get('calls') == 1, (
            f'the merge-entry gate must be single-shot; entered the steward '
            f'wait {timing.get("calls")} time(s)'
        )
        assert timing.get('elapsed', 0.0) < 5, (
            f'cumulative wait must stay inside one derived window (0.5 + 0.5 '
            f'= 1.0s); got {timing.get("elapsed")}s'
        )
        assert len(chained.instances) <= 1, (
            f'the handler must not loop back into the machinery and build a '
            f'second steward; built {len(chained.instances)}'
        )


# ---------------------------------------------------------------------------
# The task-2505 scope tripwire's OWN L0 gates the merge (not observational)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScopeInvariantTripwireGatesTheMerge:
    """The escalate-then-gate coupling the merge-entry comments now assert.

    ``workflow.py``'s merge-entry comment and ``_check_scope_invariant``'s
    docstring were corrected by this task to say the tripwire is **NOT**
    observational: ``_escalate_scope_invariant_violation`` files
    ``severity='blocking'``, ``level=0``, which ``_is_gating_escalation``
    classifies as GATING, and the gate sits ~20 lines below the call.

    Every other test in this module hand-submits an equivalent L0 via
    :func:`_submit_gating_l0`, so the coupling itself went unpinned: if the
    tripwire's severity/level ever changed (making it non-gating), or the
    check moved BELOW the gate, both docs would be wrong and nothing would
    fail.  This drives the real chain end to end — real
    ``_check_scope_invariant``, real ``_escalate_scope_invariant_violation``,
    real ``_is_gating_escalation`` — with NO hand-submitted record at all.
    """

    async def test_divergent_metadata_files_gates_this_dispatch(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        store = _RecordingEventStore()
        wf, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree, event_store=store,
        )
        _wire_resolve_callback(queue, wf)
        submit = _stub_merge_internals(wf, monkeypatch)
        timing = _time_the_wait(wf)

        # The ONLY setup: make the backend's metadata.files derive a DIFFERENT
        # lock-module set from PLAN['files'].  `_stamp_merge_phase_entered`
        # reads this blob and threads it straight into _check_scope_invariant,
        # so this is the same read production performs.
        assert PLAN['files'] == ['lib.py']
        scheduler.task_data[task_id] = {
            'id': task_id,
            'metadata': {'files': ['other_pkg/src/other_pkg/thing.py']},
        }
        # No _submit_gating_l0 here — the record under test is the one the
        # tripwire files for itself.
        assert queue.get_by_task(task_id, status='pending') == []
        wf._steward_factory = _make_silent_steward()

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 20)

        # ── the tripwire fired, and what it filed is a GATING record ────
        timeouts = _steward_wait_timeouts(store)
        assert timeouts, (
            'the tripwire\'s own blocking L0 must reach the merge-entry gate '
            'and enter the bounded steward wait — if it did not, the "NOT '
            'observational" comments at workflow.py:3447+ and the '
            '_check_scope_invariant docstring are lying again'
        )
        assert timing.get('calls') == 1
        # The record the wait timed out on is the tripwire's, by content.
        orphan_ids = timeouts[0]['escalation_ids']
        assert len(orphan_ids) == 1
        tripwire = queue.get(orphan_ids[0])
        assert tripwire is not None
        assert tripwire.severity == 'blocking'
        assert tripwire.level == 0
        assert tripwire.category == 'infra_issue'
        assert 'plan.files/metadata.files divergence' in tripwire.summary

        # ── and it stopped the line for this dispatch ───────────────────
        assert submit.await_count == 0, (
            'a scope divergence stops the line: the merge must not be '
            'submitted on the dispatch that detected it'
        )
        assert outcome is not None
        assert scheduler.statuses.get(task_id, [])[-1:] == ['blocked']


# ---------------------------------------------------------------------------
# _steward_wait_expired is per-wait state, not a sticky workflow flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStewardWaitExpiredDoesNotLeak:
    """``_steward_wait_expired`` is reset by EVERY wait, so a stale True from
    an earlier one cannot park a later, legitimately-cleared gate.

    The flag is a mutable cross-method signal: ``_wait_for_resolution`` sets
    it, and only the merge-entry gate reads it.  Its correctness rests entirely
    on the reset landing after the no-queue early return (``workflow.py:12489``)
    — a property a later refactor of that method's early returns could drop
    silently.  The consequence would be specific and bad: a task whose gate the
    steward genuinely cleared parks as ``blocked`` with a spurious L1, because
    a wait that expired back in EXECUTE/VERIFY left the flag True.
    """

    async def test_expiry_in_an_earlier_wait_does_not_park_a_cleared_gate(
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

        # (1) An EARLIER wait expires — the execute/verify-phase shape, where
        # run()'s ESCALATED branch deliberately ignores the flag and resumes.
        # No steward is wired, so _steward_progress_counter() is None and the
        # wait degrades to a plain deadline, then dismisses its orphan.
        stale = _submit_gating_l0(queue, task_id)
        await asyncio.wait_for(wf._wait_for_resolution(), 10)
        assert wf._steward_wait_expired is True, (
            'precondition: the first wait must really have expired, otherwise '
            'this test proves nothing about a stale True'
        )
        assert queue.get_by_task(task_id, status='pending', level=0) == []
        assert queue.get(stale.id).resolved_by == 'auto-dismissed'

        # (2) A LATER merge entry whose gate the steward genuinely clears.
        _submit_gating_l0(queue, task_id)
        wf._steward_factory = _make_resolving_steward(queue, task_id)

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 20)

        assert wf._steward_wait_expired is False, (
            'the second wait must report its OWN outcome — it resolved'
        )
        assert outcome is None, (
            'the gate cleared, so the handler must fall through to the merge '
            'retry loop in-slot; a terminal outcome here means the stale True '
            'leaked into the expiry branch'
        )
        assert submit.await_count == 1
        assert 'blocked' not in scheduler.statuses.get(task_id, []), (
            'a leaked stale expiry would park a task whose gate was cleared, '
            'with an L1 for a human who has nothing to decide'
        )
        assert not queue.has_open_l1(task_id)

    async def test_flag_is_reset_by_a_wait_that_resolves(
        self, config, git_ops, task_assignment, tmp_path,
    ):
        """The reset itself, isolated from the merge phase: any wait that
        returns without expiring must leave the flag False, whatever it was
        beforehand.  This is the property the reset's PLACEMENT (after the
        no-queue early return, before the loop) exists to deliver."""
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        wf, _scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree,
        )
        _wire_resolve_callback(queue, wf)

        wf._steward_wait_expired = True  # as if an earlier wait had expired
        esc = _submit_gating_l0(queue, task_id)
        queue.resolve(esc.id, 'resolved before the wait', resolved_by='test')

        await asyncio.wait_for(wf._wait_for_resolution(), 10)

        assert wf._steward_wait_expired is False


# ---------------------------------------------------------------------------
# The expiry park reports MEASURED elapsed time, not the configured window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExpiryDetailReportsMeasuredElapsed:
    """A refreshed idle window makes "the gate waited <window>s" a lie.

    ``_wait_for_resolution``'s deadline is an IDLE one: it is extended by
    another full window every time ``_steward_progress_counter()`` advances
    (``workflow.py:12578-12586``), so a steward that keeps invoking its agent
    without resolving holds the slot for N windows.  The L1 the expiry park
    files is what a human triaging the hold reasons from, so it must report the
    time that actually elapsed — with the configured window kept as a secondary
    figure, since the two answer different questions.

    Every other test in this module shrinks both terms to 0.5s and wires fakes
    with no progress counter, so the extension branch never runs; this is the
    one that exercises it.
    """

    async def test_refreshed_window_still_terminates_and_reports_real_elapsed(
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
        timing = _time_the_wait(wf)

        _submit_gating_l0(queue, task_id)
        wf._steward_factory = _make_silent_steward()

        # Advance the progress counter exactly ONCE: read at wait entry (0),
        # then 1 at the first expiry — an advance, so the deadline refreshes —
        # then 1 again at the second, which is no advance and gives up.  Two
        # windows of ~1.0s each, so ~2s of real wait against a detail line that
        # would otherwise claim 1s.
        counter = iter([0, 1, 1, 1, 1])
        wf._steward_progress_counter = lambda: next(counter, 1)  # type: ignore[method-assign]

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 30)

        # It still TERMINATES — a refreshed window is bounded by the steward's
        # own attempt ceiling, not unbounded.
        assert outcome is not None
        assert submit.await_count == 0
        assert scheduler.statuses.get(task_id, [])[-1:] == ['blocked']

        # The refresh really happened: one window is 1.0s, so a wait that did
        # NOT extend could not have taken this long.
        elapsed = timing.get('elapsed', 0.0)
        assert elapsed >= 1.5, (
            f'expected the deadline to be refreshed for a second window '
            f'(~2.0s total); the wait took {elapsed}s, so the extension branch '
            f'never ran and this test is not exercising what it claims'
        )

        # And the L1 a human reads reports the MEASURED time, not the 1s
        # configured window.
        l1s = [
            e for e in queue.get_by_task(task_id, status='pending') if e.level == 1
        ]
        assert l1s, 'the expiry park must file an L1'
        match = re.search(r'waited ([0-9.]+)s of wall clock', l1s[0].detail)
        assert match, (
            f'the expiry detail must report measured wall clock; got:\n'
            f'{l1s[0].detail}'
        )
        reported = float(match.group(1))
        assert reported >= 1.5, (
            f'the detail reported {reported}s but the wait really took '
            f'{elapsed}s — a human triaging this park would under-estimate the '
            f'hold by a whole multiple of the configured window'
        )
        assert 'configured idle window' in l1s[0].detail, (
            'the configured window stays in the detail as a SECONDARY figure — '
            'it is what an operator would tune, just not what elapsed'
        )


__all__ = [
    'PLAN',
    '_build_workflow',
    '_make_advanced_worktree',
    '_make_chained_gating_steward',
    '_make_resolving_steward',
    '_make_silent_steward',
    '_make_terminal_decision_steward',
    '_steward_wait_timeouts',
    '_stub_merge_internals',
    '_submit_born_at_l2',
    '_submit_gating_l0',
    '_time_the_wait',
    '_wire_resolve_callback',
]
