"""Regression tests for task 1448: merge-queue halt owner released on workflow cancellation.

Tests that each of the four _handle_* workflow methods releases the MergeWorker
halt-owner registration when the workflow task is cancelled mid-await, preventing
a deadlock where the merger loop blocks forever on _wip_halt.wait() with no
escalation owner to trigger _on_escalation_resolved → unhalt_wip().

Each handler follows the pattern:
    escalation_queue.submit(esc)
    merge_worker.set_halt_owner(esc.id)
    await self._escalation_event.wait()

Without a try/except around the await, a CancelledError leaves halt_owner_esc_id
set with no live workflow to clear it, blocking the merge queue permanently.

See: task 1448 (Protect merge-queue halt owner registration from workflow cancellation)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from escalation.queue import EscalationQueue

from orchestrator.merge_queue import MergeOutcome
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow


class _FakeMergeWorker:
    """Minimal halt-owner state machine — same contract as MergeWorker.

    Mirrors test_halt_owner._FakeMergeWorker and extends it to:
    - accept an optional ``reason`` keyword on ``unhalt_wip``
    - record the most-recent reason as ``last_unhalt_reason`` for assertion
    """

    def __init__(self) -> None:
        self._halted = False
        self._owner: str | None = None
        self.last_unhalt_reason: str | None = None

    @property
    def is_wip_halted(self) -> bool:
        return self._halted

    @property
    def halt_owner_esc_id(self) -> str | None:
        return self._owner

    def halt_for_wip(self, reason: str) -> None:
        self._halted = True
        self._owner = None

    def set_halt_owner(self, esc_id: str) -> None:
        assert self._owner is None, (
            f'halt owner already set to {self._owner!r}, '
            f'refusing to overwrite with {esc_id!r}'
        )
        self._owner = esc_id

    def is_halt_owner(self, esc_id: str) -> bool:
        return self._owner is not None and self._owner == esc_id

    def unhalt_wip(self, reason: str | None = None) -> None:
        self.last_unhalt_reason = reason
        self._halted = False
        self._owner = None


@pytest.fixture
def fake_worker() -> _FakeMergeWorker:
    return _FakeMergeWorker()


@pytest.fixture
def workflow(
    tmp_path: Path,
    fake_worker: _FakeMergeWorker,
    mock_orch_config: MagicMock,
) -> TaskWorkflow:
    """Minimal TaskWorkflow wired for halt-owner cancellation tests.

    Uses a real EscalationQueue (rooted at tmp_path) so submit/make_id work
    end-to-end.  Everything else is a no-op stub — these tests exercise only
    the four _handle_* methods, not the full workflow.run() cycle.
    """
    queue = EscalationQueue(tmp_path / 'escalations')
    assignment = TaskAssignment(
        task_id='1448-test',
        task={
            'id': '1448-test',
            'title': 'Test task for halt-owner regression',
            'description': 'Halt owner release on cancellation',
            'status': 'pending',
            'metadata': {},
            'dependencies': [],
        },
        modules=[],  # empty → _resolve_module_configs returns [] without calling config.for_module
    )
    git_ops = MagicMock()
    escalation_event = asyncio.Event()
    return TaskWorkflow(
        assignment=assignment,
        config=mock_orch_config,
        git_ops=git_ops,
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        briefing=FakeBriefing(),    # type: ignore[arg-type]
        mcp=FakeMcp(),              # type: ignore[arg-type]
        escalation_queue=queue,
        escalation_event=escalation_event,
        merge_worker=fake_worker,
    )


# ---------------------------------------------------------------------------
# Helper: poll until a condition is met or raise on timeout
# ---------------------------------------------------------------------------

async def _poll_until(condition, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    """Yield to the event loop until condition() is truthy or timeout expires."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            pytest.fail(f'Timed out after {timeout}s waiting for condition')
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Step 1: _handle_wip_conflict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_wip_conflict_releases_halt_on_cancel(
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """Cancelling mid-await releases halt owner and un-halts the merge queue.

    Without the fix, CancelledError escapes _handle_wip_conflict leaving
    halt_owner_esc_id set, so _on_escalation_resolved never fires unhalt_wip
    and the merger loop blocks forever.

    Must fail today (pre-fix): the handler has no try/except around the await.
    """
    # Simulate the state the merge worker enters before calling the handler:
    # the queue is halted and the handler is invoked to register the escalation.
    fake_worker.halt_for_wip('wip_conflict')

    task = asyncio.create_task(
        workflow._handle_wip_conflict(
            MergeOutcome(status='wip_halted', overlap_files=['x.py']),
            'task/1448',
        )
    )

    # Wait for the handler to submit the escalation and register the halt owner
    # before we cancel — ensures we're testing the cancel-mid-await scenario,
    # not a cancel-before-await race.
    await _poll_until(lambda: fake_worker.halt_owner_esc_id is not None)

    # Cancel the workflow task — simulates harness tearing down a stalled workflow.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # (a) Halt owner must be cleared — no orphaned registration.
    assert fake_worker.halt_owner_esc_id is None, (
        'halt_owner_esc_id must be cleared after workflow cancellation '
        '(orphan halt blocks the merge queue forever)'
    )
    # (b) Merge queue must be un-halted.
    assert fake_worker.is_wip_halted is False, (
        'is_wip_halted must be False after workflow cancellation'
    )
    # (c) The un-halt must carry the workflow_cancelled reason for observability.
    assert fake_worker.last_unhalt_reason == 'workflow_cancelled', (
        f'expected reason="workflow_cancelled", got {fake_worker.last_unhalt_reason!r}'
    )


# ---------------------------------------------------------------------------
# Step 3: _handle_wip_recovery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_wip_recovery_releases_halt_on_cancel(
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """Cancel mid-await in _handle_wip_recovery releases halt owner.

    _handle_wip_recovery fires when the merge landed on main but the stash pop
    conflicted.  It submits an escalation and waits — same risk as wip_conflict.

    Must fail today (pre-fix): the handler has no try/except around the await.
    """
    fake_worker.halt_for_wip('done_wip_recovery')

    task = asyncio.create_task(
        workflow._handle_wip_recovery(
            MergeOutcome(
                status='done_wip_recovery',
                recovery_branch='wip/r1',
                merge_sha='deadbeef',
            ),
        )
    )

    await _poll_until(lambda: fake_worker.halt_owner_esc_id is not None)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_worker.halt_owner_esc_id is None, (
        'halt_owner_esc_id must be cleared after workflow cancellation'
    )
    assert fake_worker.is_wip_halted is False, (
        'is_wip_halted must be False after workflow cancellation'
    )
    assert fake_worker.last_unhalt_reason == 'workflow_cancelled', (
        f'expected reason="workflow_cancelled", got {fake_worker.last_unhalt_reason!r}'
    )


# ---------------------------------------------------------------------------
# Step 5: _handle_wip_recovery_no_advance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_wip_recovery_no_advance_releases_halt_on_cancel(
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """Cancel mid-await in _handle_wip_recovery_no_advance releases halt owner.

    This handler fires on the CAS-failure path where main was NOT advanced.
    Same escalation-submit + await pattern, same cancellation risk.

    Must fail today (pre-fix): the handler has no try/except around the await.
    """
    fake_worker.halt_for_wip('wip_recovery_no_advance')

    task = asyncio.create_task(
        workflow._handle_wip_recovery_no_advance(
            MergeOutcome(
                status='wip_recovery_no_advance',
                recovery_branch='wip/r2',
            ),
        )
    )

    await _poll_until(lambda: fake_worker.halt_owner_esc_id is not None)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_worker.halt_owner_esc_id is None, (
        'halt_owner_esc_id must be cleared after workflow cancellation'
    )
    assert fake_worker.is_wip_halted is False, (
        'is_wip_halted must be False after workflow cancellation'
    )
    assert fake_worker.last_unhalt_reason == 'workflow_cancelled', (
        f'expected reason="workflow_cancelled", got {fake_worker.last_unhalt_reason!r}'
    )


# ---------------------------------------------------------------------------
# Step 7: _handle_unmerged_state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_unmerged_state_releases_halt_on_cancel(
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """Cancel mid-await in _handle_unmerged_state releases halt owner.

    _handle_unmerged_state fires when project_root already had pre-existing
    UU/AA/DD markers before the merge attempt.  Same pattern, same risk.

    Must fail today (pre-fix): the handler has no try/except around the await.
    """
    fake_worker.halt_for_wip('unmerged_state')

    task = asyncio.create_task(
        workflow._handle_unmerged_state(
            MergeOutcome(status='unmerged_state'),
            'task/1448',
        )
    )

    await _poll_until(lambda: fake_worker.halt_owner_esc_id is not None)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_worker.halt_owner_esc_id is None, (
        'step-7: halt_owner_esc_id must be cleared after workflow cancellation'
    )
    assert fake_worker.is_wip_halted is False, (
        'step-7: is_wip_halted must be False after workflow cancellation'
    )
    assert fake_worker.last_unhalt_reason == 'workflow_cancelled', (
        f'step-7: expected reason="workflow_cancelled", got {fake_worker.last_unhalt_reason!r}'
    )


# ---------------------------------------------------------------------------
# Step 9: submit failure must not register a halt owner
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_failure_does_not_register_halt_owner(
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """If escalation_queue.submit() raises, set_halt_owner must never be called.

    The helper orders: submit → set_halt_owner.  A registered owner with no
    pending escalation cannot be resolved by _on_escalation_resolved, so the
    halt would be permanent.  This pin ensures submit-first ordering is
    preserved through any future refactor of _submit_halt_escalation_and_wait.

    The test replaces submit with a function that raises RuntimeError and calls
    _handle_wip_conflict directly (no asyncio.Task wrapper) — the RuntimeError
    must propagate and halt_owner_esc_id must remain None.

    This test must pass immediately after step-2 (helper preserves the
    submit-before-set_halt_owner ordering that gives this property for free).
    """
    # Inject a failing submit — no halt_for_wip call needed, we're only testing
    # that a submit failure cannot corrupt the halt-owner state.
    assert workflow.escalation_queue is not None
    original_submit = workflow.escalation_queue.submit

    def _raise_on_submit(esc):
        raise RuntimeError('disk full')

    workflow.escalation_queue.submit = _raise_on_submit  # type: ignore[method-assign]

    try:
        with pytest.raises(RuntimeError, match='disk full'):
            await workflow._handle_wip_conflict(
                MergeOutcome(status='wip_halted', overlap_files=['x.py']),
                'task/1448',
            )
    finally:
        workflow.escalation_queue.submit = original_submit  # type: ignore[method-assign]

    # (a) No halt owner must have been registered — submit raised before set_halt_owner.
    assert fake_worker.halt_owner_esc_id is None, (
        'halt_owner_esc_id must remain None when submit() raises — '
        'a registered owner with no pending escalation blocks the merge queue forever'
    )
    # (b) Merge queue must remain un-halted — unhalt_wip was never called.
    assert fake_worker.is_wip_halted is False, (
        'is_wip_halted must be False when submit() raises — '
        'no escalation was registered so the halt state must be clean'
    )


# ---------------------------------------------------------------------------
# Task 1765: HALT-WITHOUT-OWNER guard — submit failure releases orphan halt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_halt_owning_escalation_releases_orphan_halt_on_submit_failure(
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """If submit() raises while a merger halt is engaged (ownerless), the helper
    must release the orphan halt before re-raising.

    Without the guard (pre-fix), submit raises and propagates before
    set_halt_owner; the halt stays engaged with no owner — silently blocking
    all merges on the lane until force_unhalt_merge_queue.

    Must FAIL today (pre-fix): _submit_halt_owning_escalation has no guard.
    """
    from escalation.models import Escalation

    # Pre-engage the merger halt (ownerless — mirrors _map_advance_failure).
    fake_worker.halt_for_wip('wip_overlap')
    assert fake_worker.is_wip_halted
    assert fake_worker.halt_owner_esc_id is None

    # Monkeypatch submit to raise.
    assert workflow.escalation_queue is not None
    original_submit = workflow.escalation_queue.submit

    def _raise_on_submit(esc):
        raise RuntimeError('disk full')

    workflow.escalation_queue.submit = _raise_on_submit  # type: ignore[method-assign]

    # Build a minimal Escalation to pass to the helper.
    esc = Escalation(
        id=workflow.escalation_queue.make_id(workflow.task_id),
        task_id=workflow.task_id,
        agent_role='orchestrator',
        severity='blocking',
        category='wip_conflict',
        summary='WIP overlap test',
        level=1,
    )

    try:
        with pytest.raises(RuntimeError, match='disk full'):
            workflow._submit_halt_owning_escalation(esc)
    finally:
        workflow.escalation_queue.submit = original_submit  # type: ignore[method-assign]

    # (a) Halt must be released — no silent orphan halt.
    assert fake_worker.is_wip_halted is False, (
        'is_wip_halted must be False after submit() raises — '
        'orphan halt guard must release it before re-raising'
    )
    # (b) Owner must still be None (set_halt_owner was never reached).
    assert fake_worker.halt_owner_esc_id is None, (
        'halt_owner_esc_id must remain None — set_halt_owner was never reached'
    )
    # (c) Unhalt reason must be set (guard fired).
    assert fake_worker.last_unhalt_reason is not None, (
        'last_unhalt_reason must be set — guard must call unhalt_wip with a reason'
    )


@pytest.mark.asyncio
async def test_handle_wip_conflict_releases_orphan_halt_on_submit_failure(
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """A pre-engaged merger halt is released when submit() raises inside
    _handle_wip_conflict, via the new guard in _submit_halt_owning_escalation.

    _submit_halt_escalation_and_wait's existing try/except does NOT cover this
    case: it wraps emit+await (lines after the _submit_halt_owning_escalation
    call), so a submit failure escapes before the try. The guard in the shared
    helper closes the gap for both the single-task path and the train path.

    Must FAIL today (pre-fix): no guard in _submit_halt_owning_escalation.
    """
    # Pre-engage the merger halt (ownerless).
    fake_worker.halt_for_wip('wip_overlap')
    assert fake_worker.is_wip_halted

    # Monkeypatch submit to raise.
    assert workflow.escalation_queue is not None
    original_submit = workflow.escalation_queue.submit

    def _raise_on_submit(esc):
        raise RuntimeError('disk full')

    workflow.escalation_queue.submit = _raise_on_submit  # type: ignore[method-assign]

    try:
        with pytest.raises(RuntimeError, match='disk full'):
            await workflow._handle_wip_conflict(
                MergeOutcome(status='wip_halted', overlap_files=['x.py']),
                'task/1765',
            )
    finally:
        workflow.escalation_queue.submit = original_submit  # type: ignore[method-assign]

    # Orphan halt must be released.
    assert fake_worker.is_wip_halted is False, (
        'is_wip_halted must be False after submit() raises — orphan halt released'
    )
    assert fake_worker.halt_owner_esc_id is None, (
        'halt_owner_esc_id must remain None — set_halt_owner was never reached'
    )
    assert fake_worker.last_unhalt_reason is not None, (
        'last_unhalt_reason must be set — guard fired on ownerless orphan halt'
    )


@pytest.mark.asyncio
async def test_submit_failure_does_not_release_foreign_owned_halt(
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """Guard is a no-op when the halt is already owned by a FOREIGN escalation.

    Pins the ``halt_owner_esc_id is None`` conjunct: the guard must only release
    genuine ownerless orphans — never steal or clear a foreign owner's halt.

    Mirrors test_consumer_halt_already_owned_skips_escalate (which guards the
    same probe predicate in the consumer).

    This test is a green pin — it passes before and after the fix because the
    guard must not fire when the halt is already owned.
    """
    from escalation.models import Escalation

    # Pre-engage the halt AND set a FOREIGN owner (bypass the assertion in set_halt_owner).
    fake_worker.halt_for_wip('wip_overlap')
    fake_worker._owner = 'esc-other-1'  # type: ignore[attr-defined]
    assert fake_worker.is_wip_halted
    assert fake_worker.halt_owner_esc_id == 'esc-other-1'

    # Monkeypatch submit to raise.
    assert workflow.escalation_queue is not None
    original_submit = workflow.escalation_queue.submit

    def _raise_on_submit(esc):
        raise RuntimeError('disk full')

    workflow.escalation_queue.submit = _raise_on_submit  # type: ignore[method-assign]

    esc = Escalation(
        id=workflow.escalation_queue.make_id(workflow.task_id),
        task_id=workflow.task_id,
        agent_role='orchestrator',
        severity='blocking',
        category='wip_conflict',
        summary='Foreign owner test',
        level=1,
    )

    try:
        with pytest.raises(RuntimeError, match='disk full'):
            workflow._submit_halt_owning_escalation(esc)
    finally:
        workflow.escalation_queue.submit = original_submit  # type: ignore[method-assign]

    # (a) Foreign owner must be preserved — guard must NOT fire.
    assert fake_worker.halt_owner_esc_id == 'esc-other-1', (
        f'halt_owner_esc_id must remain "esc-other-1", got {fake_worker.halt_owner_esc_id!r}'
    )
    # (b) Halt must remain engaged.
    assert fake_worker.is_wip_halted is True, (
        'is_wip_halted must be True — guard must NOT release a foreign-owned halt'
    )
    # (c) Unhalt must NOT have fired.
    assert fake_worker.last_unhalt_reason is None, (
        'last_unhalt_reason must be None — guard was a no-op for foreign-owned halt'
    )
