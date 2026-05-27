"""Tests for ``Harness.cancel_workflow`` — Step 4 of zombie-escalation fix.

Verifies the registry-and-set behaviour without spinning up a real workflow:

* ``cancel_workflow`` returns ``False`` when the task has no slot.
* ``cancel_workflow`` sets the registered Event when the slot exists.
* ``is_workflow_active`` mirrors registry presence.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from _orch_helpers import _init_harness_state_for_test

from orchestrator.harness import Harness
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome


@pytest.fixture
def harness() -> Harness:
    # cancel_workflow / is_workflow_active read only ``_workflow_cancel_events``.
    # Building a full Harness requires a real OrchestratorConfig; bypass __init__
    # and populate only the fields we exercise.
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)  # task 1449: initialise digest counters
    h._workflow_cancel_events = {}
    # cancel_workflow stamps wall-clock for the reconcile sweep's grace
    # window (R3 race guard).  Initialise here so the bypass-__init__
    # fixture matches the real Harness attribute set.
    h._workflow_cancel_at = {}
    # hard_cancel_workflow reads _workflow_slot_tasks (task 1491, step-6).
    h._workflow_slot_tasks = {}
    return h


def test_cancel_workflow_returns_false_when_no_slot(harness: Harness):
    assert harness.cancel_workflow('does-not-exist') is False
    assert harness.is_workflow_active('does-not-exist') is False


def test_cancel_workflow_sets_event_when_slot_exists(harness: Harness):
    event = asyncio.Event()
    harness._workflow_cancel_events['42'] = event

    assert harness.is_workflow_active('42') is True
    assert event.is_set() is False

    assert harness.cancel_workflow('42') is True
    assert event.is_set() is True


def test_cancel_workflow_idempotent(harness: Harness):
    event = asyncio.Event()
    event.set()
    harness._workflow_cancel_events['42'] = event

    # Already-set event is fine; cancel still reports active=True.
    assert harness.cancel_workflow('42') is True
    assert event.is_set() is True


# ---------------------------------------------------------------------------
# hard_cancel_workflow — task 1491, ITEM 2
# ---------------------------------------------------------------------------


def test_hard_cancel_workflow_returns_false_when_no_slot(harness: Harness):
    """Returns False when no slot task is registered for the task_id."""
    assert harness.hard_cancel_workflow('does-not-exist') is False


@pytest.mark.asyncio
async def test_hard_cancel_workflow_cancels_registered_task(harness: Harness):
    """When a live asyncio.Task is registered, hard_cancel_workflow cancels it
    and returns True."""
    slot_task = asyncio.ensure_future(asyncio.sleep(3600))
    harness._workflow_slot_tasks['42'] = slot_task

    result = harness.hard_cancel_workflow('42')
    assert result is True

    # Allow the event loop to propagate the cancellation.
    with pytest.raises(asyncio.CancelledError):
        await slot_task
    assert slot_task.cancelled()


@pytest.mark.asyncio
async def test_hard_cancel_workflow_returns_false_for_done_task(harness: Harness):
    """Returns False when the registered task is already done."""
    slot_task = asyncio.ensure_future(asyncio.sleep(0))
    await slot_task  # let it complete

    harness._workflow_slot_tasks['42'] = slot_task
    assert slot_task.done()

    result = harness.hard_cancel_workflow('42')
    assert result is False


# ---------------------------------------------------------------------------
# _run_slot hard-cancel integration — task 1491, step-9 / step-10
# ---------------------------------------------------------------------------


@pytest.fixture
def harness_for_run_slot() -> Harness:
    """Harness with all attributes needed to drive _run_slot directly."""
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)
    # cancel / slot registries
    h._workflow_cancel_events = {}
    h._workflow_cancel_at = {}
    h._workflow_slot_tasks = {}
    h._terminal_cancel_counts = {}
    h._escalation_events = {}
    h._escalation_queue = None
    # run-slot supporting state
    h._recovered_plans = {}
    h._recovered_sessions = {}
    h._preserved_worktrees = set()
    h.event_store = None
    # TaskWorkflow constructor kwarg sources — all None since TaskWorkflow is
    # patched in the test body; kwargs are evaluated before passing to the mock
    # so the attribute accesses must not raise AttributeError.
    h.config = None  # type: ignore[assignment]
    h.git_ops = None  # type: ignore[assignment]
    h.briefing = None  # type: ignore[assignment]
    h.mcp = None  # type: ignore[assignment]
    h.usage_gate = None
    h._merge_queue = None  # type: ignore[assignment]
    h._merge_worker = None
    h.cost_store = None
    # _collect_done_reports uses these
    h._run_store = None
    h._run_id = None
    h.review_checkpoint = None
    # scheduler — only release() is called from _run_slot's finally
    h.scheduler = MagicMock()
    h.scheduler.release = MagicMock()
    return h


@pytest.mark.asyncio
async def test_run_slot_returns_cancelled_report_when_hard_cancelled(
    harness_for_run_slot: Harness,
) -> None:
    """RED: CancelledError escapes _run_slot → wrapper_task.cancelled() is True;
    _collect_done_reports' t.result() re-raises CancelledError past its own
    `except Exception` guard, unwinding the harness main loop.

    GREEN (step-10): _run_slot catches asyncio.CancelledError before
    `except Exception`, returns a synthetic TaskReport(outcome=CANCELLED) so:
    (a) wrapper_task completes normally (cancelled() is False);
    (b) wrapper_task.result() is a TaskReport(outcome=CANCELLED);
    (c) finally cleanup runs (registries cleared, semaphore released);
    (d) _collect_done_reports appends the report without raising.
    """
    h = harness_for_run_slot
    tid = '42'
    assignment = TaskAssignment(
        task_id=tid,
        task={'title': 'wedged task'},
        modules=[],
    )
    # Semaphore starts at 0 (as if the slot was already acquired by the harness
    # main loop).  _run_slot's finally calls sem.release() → value becomes 1.
    sem = asyncio.Semaphore(0)

    with patch('orchestrator.harness.TaskWorkflow') as mock_wf_cls:
        # Stub workflow.run() to wedge until hard-cancelled.
        async def _wedge() -> None:
            await asyncio.sleep(3600)

        mock_wf = MagicMock()
        mock_wf.run = _wedge
        mock_wf_cls.return_value = mock_wf

        wrapper_task = asyncio.create_task(h._run_slot(assignment, sem))

        # Poll until _run_slot registers itself in _workflow_slot_tasks.
        for _ in range(50):
            if tid in h._workflow_slot_tasks:
                break
            await asyncio.sleep(0.01)
        assert tid in h._workflow_slot_tasks, (
            '_run_slot did not register itself in _workflow_slot_tasks'
        )

        h.hard_cancel_workflow(tid)

        # Wait for wrapper_task to finish (cancelled or returned normally).
        done, _ = await asyncio.wait({wrapper_task}, timeout=5.0)
        assert wrapper_task in done, 'wrapper_task did not finish within 5 s'

    # (a) RED: fails because wrapper_task.cancelled() is True — CancelledError
    # escaped _run_slot, setting the asyncio.Task to CANCELLED state.
    assert not wrapper_task.cancelled(), (
        'Expected _run_slot to return a synthetic CANCELLED TaskReport, '
        'but the Task ended up in CANCELLED state — CancelledError escaped.'
    )

    # (b) wrapper_task.result() must return TaskReport(outcome=CANCELLED).
    report = wrapper_task.result()
    assert report is not None, 'Expected _run_slot to return a TaskReport, got None'
    assert report.outcome == WorkflowOutcome.CANCELLED, (
        f'Expected outcome=CANCELLED, got {report.outcome!r}'
    )

    # (c) Finally cleanup ran: registries cleared, semaphore released.
    assert tid not in h._workflow_slot_tasks, 'slot task not cleaned up in finally'
    assert tid not in h._workflow_cancel_events, 'cancel event not cleaned up in finally'
    assert sem._value == 1, f'semaphore not released by finally (value={sem._value})'

    # (c2) The soft-cancel grace stamp set by hard_cancel_workflow must PERSIST
    # past the finally — it is no longer popped there (popping it defeated the
    # R3 reconcile grace window, leaving the post-exit window uncovered).  It
    # is cleared only at (re)dispatch or lazily by the sweep once it expires.
    assert tid in h._workflow_cancel_at, (
        'grace stamp must persist past the finally (popped only at re-dispatch)'
    )

    # (d) Regression: _collect_done_reports must handle the wrapper_task cleanly.
    task_reports: list = []
    h._collect_done_reports({wrapper_task}, task_reports)
    assert len(task_reports) == 1, (
        f'Expected _collect_done_reports to append 1 report, got {len(task_reports)}'
    )
    assert task_reports[0].outcome == WorkflowOutcome.CANCELLED


@pytest.mark.asyncio
async def test_run_slot_clears_stale_grace_stamp_at_dispatch(
    harness_for_run_slot: Harness,
) -> None:
    """A stale ``_workflow_cancel_at`` stamp from a prior incarnation is
    cleared when the task is (re)dispatched.

    The finally no longer pops the stamp, so a freshly re-dispatched run must
    clear it at the top of ``_run_slot`` (right after registering the new
    cancel_event) — otherwise a leftover stamp would grace-skip the sweep for
    a task that is genuinely running again.
    """
    h = harness_for_run_slot
    tid = '42'
    # Stale stamp left over from a prior incarnation's soft-cancel.
    h._workflow_cancel_at[tid] = 123.0
    assignment = TaskAssignment(task_id=tid, task={'title': 't'}, modules=[])
    sem = asyncio.Semaphore(0)

    with patch('orchestrator.harness.TaskWorkflow') as mock_wf_cls:
        async def _wedge() -> None:
            await asyncio.sleep(3600)

        mock_wf = MagicMock()
        mock_wf.run = _wedge
        mock_wf_cls.return_value = mock_wf

        wrapper_task = asyncio.create_task(h._run_slot(assignment, sem))

        # Poll until the slot registers.  The stamp pop happens at dispatch,
        # just before the slot-task registration, so once the slot is present
        # the stale stamp must already be gone.
        for _ in range(50):
            if tid in h._workflow_slot_tasks:
                break
            await asyncio.sleep(0.01)
        assert tid in h._workflow_slot_tasks, (
            '_run_slot did not register itself in _workflow_slot_tasks'
        )

        assert tid not in h._workflow_cancel_at, (
            'stale grace stamp not cleared at (re)dispatch'
        )

        # Cleanup: hard-cancel the wedged slot and drain.
        h.hard_cancel_workflow(tid)
        await asyncio.wait({wrapper_task}, timeout=5.0)
