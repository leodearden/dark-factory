"""Halt-owner contracts for the single-task ``_handle_*`` merge-halt handlers.

The handlers split into TWO shapes, and this module pins both.

WAITING shape — ``_handle_wip_conflict`` (REQUEUED: the wait IS its retry
mechanism) and ``_handle_wip_recovery`` (DONE: the merge landed).  These follow::

    escalation_queue.submit(esc)
    merge_worker.set_halt_owner(esc.id)
    await self._escalation_event.wait()

Without a try/except around that await, a CancelledError leaves
halt_owner_esc_id set with no live workflow to clear it, blocking the merge
queue permanently — so task 1448 hardened them to release the halt on
cancellation, and ``*_releases_halt_on_cancel`` pins that.

ESCALATE-AND-BLOCK shape — the merge-halt TRIO ``_handle_stash_failed`` /
``_handle_unmerged_state`` / ``_handle_wip_recovery_no_advance``, i.e. exactly
the handlers that return BLOCKED.  Task 3537 (spec
``docs/task-escalation-state-spec.md`` §7.9 / §8-E3, INV-6 + INV-7) rewrote them
to file the halt-owning L1 and return immediately, so the slot/lane/queue are
never held on an unbounded wait and the task row is parked ``blocked`` rather
than left ``in-progress`` with no claimant.  For these the cancellation contract
is INVERTED: there is no in-handler await to interrupt and no cleanup to run, so
an exit must NEVER unhalt — the sole unhalt edge is the durable record's
resolution.  See the tombstone comment mid-module and
``test_trio_cancellation_never_unhalts``.

See: task 1448 (Protect merge-queue halt owner registration from workflow
cancellation), task 1765 (orphan-halt guards), task 2758 (stash_failed),
task 3537 (merge-halt trio escalates-and-blocks).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from escalation.queue import EscalationQueue

from orchestrator.merge_queue import MergeOutcome
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import _ORPHAN_HALT_NO_QUEUE_TOKENS, TaskWorkflow, WorkflowOutcome


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


@pytest.fixture
def workflow_no_esc(
    tmp_path: Path,
    fake_worker: _FakeMergeWorker,
    mock_orch_config: MagicMock,
) -> TaskWorkflow:
    """Same as workflow but with escalation_queue=None (config-absent deployment).

    Mirrors test_workflow_train_halt_owner.workflow_no_esc for the single-task
    handler paths.
    """
    assignment = TaskAssignment(
        task_id='1765-noesc',
        task={
            'id': '1765-noesc',
            'title': 'Single-task tip — no escalation queue',
            'description': 'Degrade gracefully when esc_queue=None',
            'status': 'in-progress',
            'metadata': {},
            'dependencies': [],
        },
        modules=[],
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
        escalation_queue=None,
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
# task 2758: stash_failed halt-escalation text (_build_wip_halt_escalation_text)
# ---------------------------------------------------------------------------

def test_build_wip_halt_escalation_text_stash_failed(workflow: TaskWorkflow) -> None:
    """stash_failed renders a DEDICATED main-checkout-hygiene halt escalation:
    category=='stash_failed', with a summary + detail that name the dirty
    tracked files and frame the incident as a project_root park/hygiene fault
    requiring manual intervention.

    RED: the builder has no stash_failed branch today, so status falls through
    to the else (wip_halted) arm → category=='wip_conflict' and the dirty
    files are never rendered.
    """
    category, summary, detail = workflow._build_wip_halt_escalation_text(
        'stash_failed',
        MergeOutcome(status='stash_failed', dirty_files=['README.md', 'write_queue.db']),
        branch_name='task/x',
    )

    assert category == 'stash_failed'

    # Summary names at least the first dirty file.
    assert 'README.md' in summary, f'summary should name the dirty file(s): {summary!r}'

    # Detail lists every dirty file and frames it as a main-checkout hygiene fault.
    assert 'README.md' in detail and 'write_queue.db' in detail
    detail_l = detail.lower()
    assert 'park' in detail_l or 'stash' in detail_l, (
        f'detail should describe the failed park/stash: {detail!r}'
    )
    assert 'project_root' in detail
    # Infrastructure / hygiene incident → route to a human.
    assert 'manual' in detail_l


def test_build_wip_halt_escalation_text_stash_failed_empty_dirty_files(
    workflow: TaskWorkflow,
) -> None:
    """An empty dirty_files list (side channel was unset) renders the
    stash_failed escalation text without crashing."""
    category, summary, detail = workflow._build_wip_halt_escalation_text(
        'stash_failed',
        MergeOutcome(status='stash_failed', dirty_files=[]),
        branch_name='task/x',
    )

    assert category == 'stash_failed'
    assert isinstance(summary, str) and summary
    assert isinstance(detail, str) and detail


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
# Task 3537: the INVERSE of the two deleted `*_releases_halt_on_cancel` tests.
#
# `test_handle_wip_recovery_no_advance_releases_halt_on_cancel` and
# `test_handle_unmerged_state_releases_halt_on_cancel` used to live here.  They
# cancelled the handler mid-await and asserted
# `last_unhalt_reason == 'workflow_cancelled'` with the owner cleared — task
# 1448's cancellation-orphan hardening of the WAITING shape, and the precise
# OPPOSITE of spec §7.9, which requires the rewritten handlers to exit WITHOUT
# ever running that cleanup.  Once these two handlers no longer wait there is no
# in-handler await to interrupt and no cleanup path to run, so keeping them
# would force the very unhalt edge the spec forbids.
#
# The two SIBLING cancel tests above (`_handle_wip_conflict`,
# `_handle_wip_recovery`) are deliberately left in place: those handlers still
# wait and still need the 1448 cleanup, which is what keeps this from being a
# blanket weakening of cancellation coverage.
#
# Their replacement — the INVERSE pin, `test_trio_cancellation_never_unhalts` —
# lives with the other boundary-#13 trio tests at the bottom of this module.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task 1765 (fold-in): orphan-halt warning when escalation_queue=None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize('handler_id,expected_outcome', [
    ('wip_conflict', WorkflowOutcome.REQUEUED),
    ('wip_recovery', WorkflowOutcome.DONE),
    ('wip_recovery_no_advance', WorkflowOutcome.BLOCKED),
    ('unmerged_state', WorkflowOutcome.BLOCKED),
    ('stash_failed', WorkflowOutcome.BLOCKED),
], ids=['wip_conflict', 'wip_recovery', 'wip_recovery_no_advance', 'unmerged_state', 'stash_failed'])
async def test_handler_warns_on_orphan_halt_when_no_escalation_queue(
    handler_id: str,
    expected_outcome: WorkflowOutcome,
    workflow_no_esc: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each of the four single-task handlers logs a distinct orphan-halt WARNING
    when escalation_queue=None and the merger pre-engaged the halt.

    The train path's ``_escalate_train_halt`` already logs this diagnostic
    (workflow.py:5752-5758). The four single-task handlers currently have
    ``if self.escalation_queue:`` with NO else, so a merger-engaged halt is
    left ownerless and silent when escalation_queue=None.

    Must FAIL today (pre-fix): no ``else:`` branch in the four handlers.

    After step-4 impl, each handler emits a WARNING containing both:
      - 'orphan halt'  (distinguishes from the pre-existing 'creating level-1
                        escalation' warning each handler already emits before the if)
      - 'unhalt_merge_queue'
    Control flow is unchanged: each handler still returns its existing outcome.
    """
    assert workflow_no_esc.escalation_queue is None

    # Pre-engage the merger halt (ownerless — mirrors _map_advance_failure precondition).
    fake_worker.halt_for_wip('wip_overlap')
    assert fake_worker.is_wip_halted
    assert fake_worker.halt_owner_esc_id is None

    with caplog.at_level(logging.WARNING):
        if handler_id == 'wip_conflict':
            outcome = await workflow_no_esc._handle_wip_conflict(
                MergeOutcome(status='wip_halted', overlap_files=['x.py']),
                'task/1765',
            )
        elif handler_id == 'wip_recovery':
            outcome = await workflow_no_esc._handle_wip_recovery(
                MergeOutcome(
                    status='done_wip_recovery',
                    recovery_branch='wip/r',
                    merge_sha='deadbeef',
                ),
            )
        elif handler_id == 'wip_recovery_no_advance':
            outcome = await workflow_no_esc._handle_wip_recovery_no_advance(
                MergeOutcome(
                    status='wip_recovery_no_advance',
                    recovery_branch='wip/r',
                ),
            )
        elif handler_id == 'stash_failed':
            outcome = await workflow_no_esc._handle_stash_failed(
                MergeOutcome(status='stash_failed', dirty_files=['README.md']),
                'task/1765',
            )
        else:  # unmerged_state
            outcome = await workflow_no_esc._handle_unmerged_state(
                MergeOutcome(status='unmerged_state'),
                'task/1765',
            )

    # (a) A WARNING must be emitted containing the orphan-halt diagnostic tokens.
    # Assert against _ORPHAN_HALT_NO_QUEUE_TOKENS (the stable constant defined
    # alongside _warn_orphan_halt_no_queue) so a prose change in the helper
    # is caught at the constant definition rather than silently breaking tests.
    warning_messages = [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING
    ]
    orphan_warnings = [
        msg for msg in warning_messages
        if all(tok in msg for tok in _ORPHAN_HALT_NO_QUEUE_TOKENS)
    ]
    assert orphan_warnings, (
        f'handler {handler_id!r}: expected a WARNING containing all tokens in '
        f'_ORPHAN_HALT_NO_QUEUE_TOKENS={_ORPHAN_HALT_NO_QUEUE_TOKENS!r}, '
        f'but got: {warning_messages!r}'
    )

    # (b) Control flow unchanged — returns the expected outcome.
    assert outcome == expected_outcome, (
        f'handler {handler_id!r}: expected {expected_outcome!r}, got {outcome!r}'
    )

    # (c) Halt remains engaged — documented degraded behavior when esc_queue is None.
    # No escalation was filed, no owner was registered, and only
    # force_unhalt_merge_queue can recover the lane. The handler must NOT
    # silently clear the halt here; that would hide the production queue blockage.
    assert fake_worker.is_wip_halted, (
        f'handler {handler_id!r}: is_wip_halted must stay True when escalation_queue '
        'is None — no esc was filed and no auto-recovery path exists; '
        'operator must run force_unhalt_merge_queue'
    )


# ---------------------------------------------------------------------------
# Task 3537 / boundary #13: the merge-halt trio ESCALATES AND BLOCKS.
#
# Spec docs/task-escalation-state-spec.md §7.9 / §8-E3, INV-6
# (status-matches-liveness) and INV-7 (holds-owned-and-bounded).
#
# The three BLOCKED-returning halt handlers used to file an L1 and then
# `await self._submit_halt_escalation_and_wait(esc)` — an UNBOUNDED wait while
# holding the slot, the locks, the lane and the merge queue (INV-7/S5), tailed
# by a bare `return WorkflowOutcome.BLOCKED` that never wrote the task row
# (INV-6/S1).  They now transplant `_escalate_train_halt`'s landed non-waiting
# shape: defensive owner re-check -> `_submit_halt_owning_escalation` ->
# `_mark_blocked(..., skip_escalation=True)`.
#
# The two SIBLING handlers keep waiting and are deliberately out of scope:
# `_handle_wip_conflict` returns REQUEUED (the wait IS its retry mechanism) and
# `_handle_wip_recovery` returns DONE (the merge landed).  Their
# `*_releases_halt_on_cancel` tests above must stay untouched.
# ---------------------------------------------------------------------------

def _invoke_trio_handler(workflow: TaskWorkflow, handler_id: str):
    """Return the un-awaited coroutine for one merge-halt trio handler."""
    if handler_id == 'stash_failed':
        return workflow._handle_stash_failed(
            MergeOutcome(status='stash_failed', dirty_files=['README.md']),
            'task/x',
        )
    if handler_id == 'unmerged_state':
        return workflow._handle_unmerged_state(
            MergeOutcome(status='unmerged_state'), 'task/1448',
        )
    if handler_id == 'wip_recovery_no_advance':
        return workflow._handle_wip_recovery_no_advance(
            MergeOutcome(
                status='wip_recovery_no_advance', recovery_branch='wip/r2',
            ),
        )
    raise AssertionError(f'unknown trio handler {handler_id!r}')


#: ``(handler_id, expected escalation category, token required in the summary)``
#:
#: The category is NOT the handler name: ``_build_wip_halt_escalation_text``
#: maps ``wip_recovery_no_advance`` to ``'wip_conflict'``.  Assert what the
#: shared builder actually produces — the rehydration filter and
#: ``_on_escalation_resolved`` both key on that category, not on the status.
_TRIO = [
    ('stash_failed', 'stash_failed', 'README.md'),
    ('unmerged_state', 'unmerged_state', 'UU/AA/DD'),
    ('wip_recovery_no_advance', 'wip_conflict', 'wip/r2'),
]

_TRIO_IDS = [row[0] for row in _TRIO]

#: The categories ``harness._rehydrate_merge_halt`` selects a halt owner from,
#: and that ``_on_escalation_resolved``'s owner check can therefore unhalt on.
_HALT_CATEGORIES = {'wip_conflict', 'unmerged_state', 'stash_failed'}

#: A representative ``_build_train_state()`` payload (escalation.models.TrainState).
_TRAIN_STATE = {
    'id': 'T7',
    'order': 2,
    'parked_members': ['alpha', 'beta'],
    'failing_member': '1448',
}


def _forbid_waiting_helper(workflow: TaskWorkflow) -> None:
    """§7.9 constraint (a), enforced STRUCTURALLY.

    ``_submit_halt_escalation_and_wait`` is the only thing on these paths that
    owns an ``except BaseException -> unhalt_wip('workflow_cancelled')``
    cleanup.  A rewrite that merely BOUNDED the wait would keep that cleanup one
    stray cancellation away from un-halting the merge queue over a dirty tree.
    Not calling the helper at all is what makes "never unhalts" a property of
    the code's shape rather than of its timing — so pin the absence of the call,
    not just the absence of its effect.
    """
    async def _never(*_args, **_kwargs):
        raise AssertionError(
            'merge-halt trio must NOT route through '
            '_submit_halt_escalation_and_wait (spec §7.9 constraint (a)): its '
            'except-BaseException cleanup unhalts the merge queue over a dirty '
            'project_root'
        )

    workflow._submit_halt_escalation_and_wait = _never  # type: ignore[method-assign]


@pytest.mark.asyncio
@pytest.mark.parametrize('handler_id,expected_category,summary_token', _TRIO, ids=_TRIO_IDS)
async def test_trio_escalates_and_blocks_without_waiting(
    handler_id: str,
    expected_category: str,
    summary_token: str,
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """Boundary #13, full contract, for each merge-halt trio handler.

    RED today: the handler parks forever on ``_escalation_event.wait()``, so it
    never returns and never writes the row.  ``asyncio.wait_for`` makes a
    regression back to the unbounded wait fail loudly as a timeout rather than
    hanging the suite.
    """
    # Merger pre-engaged the (ownerless) halt, mirroring _map_advance_failure.
    fake_worker.halt_for_wip(handler_id)
    assert fake_worker.is_wip_halted
    assert fake_worker.halt_owner_esc_id is None
    _forbid_waiting_helper(workflow)

    # (a) bounded — returns BLOCKED without anything releasing an escalation event
    outcome = await asyncio.wait_for(
        _invoke_trio_handler(workflow, handler_id), timeout=2,
    )
    assert outcome == WorkflowOutcome.BLOCKED
    assert workflow._escalation_event is not None
    assert not workflow._escalation_event.is_set(), (
        'the handler must return on its own, not because something released '
        'the escalation event (INV-7: no unbounded hold of slot/lane/queue)'
    )

    # (b) exactly ONE human-facing L1, at the category the shared builder emits
    assert workflow.escalation_queue is not None
    pending = workflow.escalation_queue.get_pending()
    assert len(pending) == 1, (
        f'{handler_id}: expected exactly one escalation (the handler owns the '
        f'human-facing record; _mark_blocked(skip_escalation=True) must not '
        f'double-file), got {pending!r}'
    )
    esc = pending[0]
    assert esc.level == 1
    assert esc.category == expected_category
    assert summary_token in esc.summary, (
        f'{handler_id}: summary must name {summary_token!r}: {esc.summary!r}'
    )

    # (c) halt still asserted and owned by that escalation — the sole unhalt
    #     edge is now the durable record's resolution (harness
    #     _on_escalation_resolved -> is_halt_owner -> unhalt_wip).
    assert fake_worker.is_wip_halted is True, (
        f'{handler_id}: the halt must survive the handler returning — the tree '
        f'is still dirty and only a human resolving the L1 may clear it'
    )
    assert fake_worker.halt_owner_esc_id == esc.id

    # (d) INV-6: the slot is freed, so the row must say so
    assert isinstance(workflow.scheduler, FakeScheduler)
    assert 'blocked' in workflow.scheduler.statuses.get(workflow.task_id, []), (
        f'{handler_id}: returning BLOCKED exits the slot, so the task row must '
        f'be parked blocked; got '
        f'{workflow.scheduler.statuses.get(workflow.task_id, [])!r}'
    )

    # (e) §7.9 constraint (a): nothing on this exit path unhalts
    assert fake_worker.last_unhalt_reason is None, (
        f'{handler_id}: the exit path must NEVER unhalt (got reason '
        f'{fake_worker.last_unhalt_reason!r}) — the halt is released only by '
        f'the durable record being resolved'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('handler_id,expected_category,summary_token', _TRIO, ids=_TRIO_IDS)
async def test_trio_never_refiles_sibling_halt_category(
    handler_id: str,
    expected_category: str,
    summary_token: str,
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """§7.9 constraint (c): recovery never re-files a sibling HALT-CATEGORY
    record — but it does still file a plain one for THIS task.

    Rehydration selects the halt owner most-recent-wins across the halt
    categories, so a second record IN A HALT CATEGORY would make a restart
    re-own the WRONG escalation.  Mirrors the defensive re-check
    ``_escalate_train_halt`` used to carry inline (now shared: it delegates to
    ``_file_halt_owning_l1`` too).

    Filing NOTHING is not the answer either (review amendment): the caller
    tails into ``_mark_blocked(skip_escalation=True)``, so the task would be
    parked ``blocked`` with ZERO escalations referencing it — a LEAVE-shaped
    row for the ground-truth sweep, while resolving the SIBLING's L1 unhalts
    the queue with no edge back to this task.  That is a silent permanent
    strand.  So the branch files a NON-OWNING ``task_failure`` record naming
    the owner: invisible to rehydration's filter and to the owner check, but
    discoverable and resolvable for this task.

    RED today: the handler unconditionally submits and calls ``set_halt_owner``,
    tripping ``_FakeMergeWorker``'s owner-collision assertion.
    """
    fake_worker.halt_for_wip(handler_id)
    fake_worker._owner = 'esc-foreign-1'  # type: ignore[attr-defined]
    _forbid_waiting_helper(workflow)

    outcome = await asyncio.wait_for(
        _invoke_trio_handler(workflow, handler_id), timeout=2,
    )

    # (a) still a truthful slot exit
    assert outcome == WorkflowOutcome.BLOCKED
    assert isinstance(workflow.scheduler, FakeScheduler)
    assert 'blocked' in workflow.scheduler.statuses.get(workflow.task_id, [])

    # (b) exactly ONE record, and NOT in a halt category
    assert workflow.escalation_queue is not None
    pending = workflow.escalation_queue.get_pending()
    assert len(pending) == 1, (
        f'{handler_id}: expected exactly one non-owning record for this task '
        f'(zero would strand it blocked-with-no-escalation; two would be the '
        f'duplicate filing), got {pending!r}'
    )
    record = pending[0]
    assert record.level == 1, f'{handler_id}: the record must be human-facing'
    assert record.category not in _HALT_CATEGORIES, (
        f'{handler_id}: a sibling already owns the halt — a second record in '
        f'halt category {record.category!r} would join rehydration\'s '
        f'most-recent-wins candidate set and make a restart re-own the wrong '
        f'escalation'
    )
    assert 'esc-foreign-1' in record.detail, (
        f'{handler_id}: the record must name the owning escalation so a human '
        f'can see what actually releases the halt: {record.detail!r}'
    )

    # (c) the foreign owner is untouched, and the halt is neither stolen nor released
    assert fake_worker.halt_owner_esc_id == 'esc-foreign-1'
    assert fake_worker.is_wip_halted is True
    assert fake_worker.last_unhalt_reason is None


@pytest.mark.parametrize('train_state', [None, _TRAIN_STATE], ids=['single_task', 'train'])
def test_file_halt_owning_l1_threads_train_state(
    train_state,
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """The shared helper is ALSO the train path's filing block.

    ``_escalate_train_halt``'s inline copy of the owner-re-check ->
    ``Escalation(...)`` -> ``_submit_halt_owning_escalation`` sequence was
    deleted in favour of this helper (review amendment: two copies of a
    load-bearing ordering contract drift).  The only thing the train copy added
    was the ``train_state=`` payload, so pin that it survives the delegation —
    the train-path tests assert ownership and outcome, but never this field.
    """
    fake_worker.halt_for_wip('stash_failed')

    workflow._file_halt_owning_l1(
        'stash_failed', 'summary', 'detail', train_state=train_state,
    )

    assert workflow.escalation_queue is not None
    pending = workflow.escalation_queue.get_pending()
    assert len(pending) == 1
    assert pending[0].train_state == train_state
    # ...and the ordering contract still holds: submitted, THEN owned.
    assert fake_worker.halt_owner_esc_id == pending[0].id


@pytest.mark.asyncio
@pytest.mark.parametrize('handler_id,expected_category,summary_token', _TRIO, ids=_TRIO_IDS)
async def test_trio_cancellation_never_unhalts(
    handler_id: str,
    expected_category: str,
    summary_token: str,
    workflow: TaskWorkflow,
    fake_worker: _FakeMergeWorker,
) -> None:
    """Cancelling AROUND a rewritten trio handler can never un-halt the queue.

    The INVERSE of the two deleted ``*_releases_halt_on_cancel`` tests (see the
    tombstone comment above ``test_submit_failure_does_not_register_halt_owner``).
    The halt is engaged because project_root is dirty / unmergeable, so
    releasing it on a teardown would let the whole fleet merge over that tree.
    Whether the handler completed first or was interrupted, the halt must stay
    engaged and ``unhalt_wip`` must never have been called.

    RED today: the handler parks on ``_escalation_event.wait()``, so the cancel
    lands mid-await and ``_submit_halt_escalation_and_wait``'s
    ``except BaseException`` cleanup fires ``unhalt_wip('workflow_cancelled')``.
    """
    fake_worker.halt_for_wip(handler_id)

    task = asyncio.create_task(_invoke_trio_handler(workflow, handler_id))
    # Let the handler get as far as owning the halt, so the cancel lands at or
    # after the point where the OLD shape would have been parked on its wait.
    await _poll_until(lambda: fake_worker.halt_owner_esc_id is not None)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert fake_worker.last_unhalt_reason is None, (
        f'{handler_id}: cancellation must NEVER unhalt the merge queue over a '
        f'dirty project_root (got reason {fake_worker.last_unhalt_reason!r}) — '
        f'the only unhalt edge is the durable L1 being resolved (spec §7.9)'
    )
    assert fake_worker.is_wip_halted is True, (
        f'{handler_id}: the halt must remain engaged after cancellation'
    )
    assert fake_worker.halt_owner_esc_id is not None, (
        f'{handler_id}: the halt owner registration must survive cancellation — '
        f'clearing it would strand the halt with nothing to resolve it'
    )
