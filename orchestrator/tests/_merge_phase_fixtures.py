"""Merge-phase scaffolding shared by the merge-entry gate test modules.

Drives ``TaskWorkflow._run_merge_phase`` around a REAL ``EscalationQueue``,
with a short 0.5 + 0.5s escalated-wait window and every merge internal
downstream of the gate stubbed.  Consumed by
``test_workflow_merge_gating_strand.py`` and ``test_info_l2_never_gates.py``.

A uniquely named module rather than ``conftest.py`` for the reason
``_briefing_helpers.py`` gives: it imports by name from a test file without
colliding with a sibling subproject's conftest under ``sys.modules['conftest']``.
A consumer imports all four fixtures by redundant alias
(``from _merge_phase_fixtures import config as config``), ``git_repo`` included
even when no test names it, because pytest resolves a fixture's dependencies
only from the requesting module's namespace.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _git_fixtures import LIB_FUNCTION_SEED, seed_repo
from _recording_event_store import _RecordingEventStore
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import (
    GitConfig,
    OrchestratorConfig,
    SandboxConfig,
    TimeoutsConfig,
)
from orchestrator.git_ops import GitOps, _run
from orchestrator.resume_resolution import CarriedResolution
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo', LIB_FUNCTION_SEED)


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    """Config with a SHORT ESCALATED-wait window: 0.5 + 0.5 = 1.0s.

    BOTH terms must be shrunk, because the ESCALATED wait is bounded by the
    DERIVED window ``timeouts.steward + steward_completion_timeout`` — not by
    ``steward_completion_timeout`` alone.  Leaving ``timeouts.steward`` at its
    stock 1800s would give the expiry test
    (``orchestrator/tests/test_workflow_merge_gating_strand.py::TestMergeGateWaitExpiry``) a ~1800s window, so
    it would blow its own ``asyncio.wait_for`` guard instead of exercising
    expiry.  (Same rationale, verbatim, as
    ``orchestrator/tests/test_workflow_escalated_steward_stall.py::config``.)

    Only ``timeouts.steward >= steward_completion_timeout`` is validated, and
    0.5 >= 0.5 satisfies it (the invariant is ``>=``, not strict ``>``).

    ``max_merge_retries`` / ``max_pre_merge_retries`` are pinned to 1 so a
    fall-through past the gate
    (``orchestrator/tests/test_workflow_merge_gating_strand.py::TestMergeGateBailEntersBoundedWait``)
    executes exactly ONE merge attempt, making the
    stubbed ``_submit_to_merge_queue``'s await count an exact
    "did the merge proceed in-slot?" signal rather than a lower bound.

    Sandbox is off: the consuming tests drive ``_run_merge_phase`` against a
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


async def _make_advanced_worktree(git_ops: GitOps, task_id: str) -> Path:
    """Create a worktree on a task branch ONE commit ahead of main.

    ``_worktree_has_wip_commits`` must derive True — it is the wip probe the
    production ``_ensure_steward_started`` injects into the steward, and the
    steward give-up branches the consuming tests model are wip-gated.
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

    *event_store* is optional because only the wait-expiry tests, such as
    ``orchestrator/tests/test_workflow_merge_gating_strand.py::TestMergeGateWaitExpiry``, assert on the
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
    def _on_resolved(esc) -> None:  # signature parity with harness
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
    single-shot property (spec S5,
    ``orchestrator/tests/test_workflow_merge_gating_strand.py::TestGateReFireIsBounded``) reads directly.

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

    async def _timed() -> tuple[CarriedResolution, ...]:
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


def _fake_git_run(*, head: str = 'BRANCH-HEAD-SHA'):
    """Async stand-in for ``orchestrator.workflow._run`` (the merge-entry
    ``git rev-parse HEAD``)."""

    async def _stub(cmd, cwd=None, **kwargs):
        return 0, head + '\n', ''

    return _stub


def _stub_merge_internals(
    wf: TaskWorkflow,
    monkeypatch,
    *,
    submit_outcome: WorkflowOutcome = WorkflowOutcome.DONE,
) -> AsyncMock:
    """Stub everything DOWNSTREAM of the merge-entry gate, and nothing else.

    Lifted from the merge-internals stub preamble of
    ``orchestrator/tests/test_workflow_merge_phase_liveness.py::TestEntryOrdering``
    but deliberately WITHOUT that preamble's ``wf._check_escalations = MagicMock(
    return_value=[])`` line: the real ``_is_gating_escalation`` predicate must
    stay in the loop, since the awaited/not-awaited state of the returned
    ``_submit_to_merge_queue`` mock is precisely the "did the merge proceed
    in-slot?" signal for the three gate boundaries
    (``orchestrator/tests/test_workflow_merge_gating_strand.py::TestMergeGateBailEntersBoundedWait``,
    ``::TestMergeGateL2ShortCircuits`` and ``::TestMergeGateWaitExpiry``).

    ``_check_scope_invariant`` is likewise left REAL: ``FakeScheduler.get_task``
    returns ``None`` for an untracked task, which the tripwire's documented
    fail-safe treats as "cannot check" and skips — so it files no escalation of
    its own and cannot contaminate the gate under test.  That default is what
    lets every OTHER test hand-submit its own record; populating
    ``scheduler.task_data[task_id]`` with a divergent ``metadata.files`` turns
    the tripwire back on, which is exactly what
    ``orchestrator/tests/test_workflow_merge_gating_strand.py::TestScopeInvariantTripwireGatesTheMerge``
    does to pin the
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


def _make_silent_steward() -> type:
    """A steward that publishes NOTHING and leaves its L0 pending.

    The general "producer went silent" shape — the bound must survive it
    REGARDLESS of which producer bug caused it.  Deliberately does not model
    any specific known bug: the bound has to hold for any future early return
    that forgets to dismiss.  Mirrors
    ``orchestrator/tests/test_workflow_escalated_steward_stall.py::_make_silent_steward``.

    ``instances`` counts constructions, which the single-shot property
    (``orchestrator/tests/test_workflow_merge_gating_strand.py::TestGateReFireIsBounded``) reads
    to prove the handler did not loop back into the machinery.
    """

    class _FakeSteward:
        instances: list[_FakeSteward] = []

        def __init__(self, wt_path, cfg_dir):
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
