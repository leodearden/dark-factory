"""Tests for ``Harness.cancel_workflow`` — Step 4 of zombie-escalation fix.

Verifies the registry-and-set behaviour without spinning up a real workflow:

* ``cancel_workflow`` returns ``False`` when the task has no slot.
* ``cancel_workflow`` sets the registered Event when the slot exists.
* ``is_workflow_active`` mirrors registry presence.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import _init_harness_state_for_test, wire_scheduler_liveness_mock

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome
from orchestrator.workflow_types import WorkflowState


@pytest.fixture
def harness() -> Harness:
    # cancel_workflow / is_workflow_active read only ``_workflow_cancel_events``.
    # Building a full Harness requires a real OrchestratorConfig; bypass __init__
    # and populate only the fields we exercise.
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)  # task 1449: initialise digest counters
    h._workflow_cancel_events = {}
    # cancel_workflow/hard_cancel_workflow stamp wall-clock for the reconcile
    # sweep's grace window (R3 race guard) via the Scheduler (task 2235: the
    # grace-stamp dict now lives on the Scheduler, not the Harness) — wire a
    # scheduler mock with real liveness-accessor behaviour so
    # note_workflow_cancelled() and the _workflow_cancel_at back-compat shim
    # both work instead of AttributeError'ing on a missing `scheduler`.
    h.scheduler = MagicMock()
    wire_scheduler_liveness_mock(h.scheduler)
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
    # scheduler must exist before any _workflow_cancel_at access below — the
    # grace-stamp dict now lives on the Scheduler (task 2235) and the Harness
    # `_workflow_cancel_at` property forwards reads/writes to
    # self.scheduler._workflow_cancel_at.
    h.scheduler = MagicMock()
    wire_scheduler_liveness_mock(h.scheduler)
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
    h._recovered_session_config_dirs = {}
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
    h._merge_inflight_registry = None  # type: ignore[assignment]
    h.cost_store = None
    # _collect_done_reports uses these
    h._run_store = None
    h._run_id = None
    h.review_checkpoint = None
    # scheduler (wired above) — only release() is additionally called from
    # _run_slot's finally
    h.scheduler.release = MagicMock()
    # carries_substrate_probe was added after these tests were written; mock it
    # False so _run_slot does not call _run_substrate_gate (which dereferences
    # git_ops.worktree_base, left None because TaskWorkflow is fully patched).
    h.scheduler.carries_substrate_probe = MagicMock(return_value=False)
    # is_deterministic gate (added after these tests): mock False so _run_slot
    # follows the normal TaskWorkflow path instead of misrouting to
    # _run_deterministic_slot on a truthy MagicMock (task 1907).
    h.scheduler.is_deterministic = MagicMock(return_value=False)
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

    with patch('orchestrator.harness.build_workflow') as mock_wf_cls:
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

    with patch('orchestrator.harness.build_workflow') as mock_wf_cls:
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


async def _drive_cancelled_slot(
    h: Harness,
    tid: str,
    *,
    reason: str | None = None,
    state: WorkflowState | None = WorkflowState.EXECUTE,
    cancel_during_setup: bool = False,
    capture_task: list | None = None,
):
    """Run a real _run_slot to its hard-cancel and return the synthetic report.

    Reuses the driver shape of test_run_slot_returns_cancelled_report_when_hard
    _cancelled: a live slot task with build_workflow patched to a wedging run(),
    then hard_cancel_workflow.

    ``capture_task`` receives the wrapper asyncio.Task itself, for callers that
    need to feed it back through ``_collect_done_reports`` (the runs.db half).
    """
    assignment = TaskAssignment(task_id=tid, task={'title': 'wedged task'}, modules=[])
    sem = asyncio.Semaphore(0)

    with patch('orchestrator.harness.build_workflow') as mock_wf_cls:
        if cancel_during_setup:
            # A cancel landing BEFORE `workflow` is bound.  The except handler
            # must not touch an unbound local.
            mock_wf_cls.side_effect = asyncio.CancelledError()
            wrapper_task = asyncio.create_task(h._run_slot(assignment, sem))
            done, _ = await asyncio.wait({wrapper_task}, timeout=5.0)
            assert wrapper_task in done
            return wrapper_task.result()

        async def _wedge() -> None:
            await asyncio.sleep(3600)

        mock_wf = MagicMock()
        mock_wf.run = _wedge
        mock_wf.state = state
        mock_wf_cls.return_value = mock_wf

        wrapper_task = asyncio.create_task(h._run_slot(assignment, sem))
        if capture_task is not None:
            capture_task.append(wrapper_task)
        for _ in range(50):
            if tid in h._workflow_slot_tasks:
                break
            await asyncio.sleep(0.01)
        assert tid in h._workflow_slot_tasks

        if reason is None:
            h.hard_cancel_workflow(tid)
        else:
            h.hard_cancel_workflow(tid, reason=reason)

        done, _ = await asyncio.wait({wrapper_task}, timeout=5.0)
        assert wrapper_task in done, 'wrapper_task did not finish within 5 s'
        return wrapper_task.result()


@pytest.mark.asyncio
class TestHardCancelReportCarriesCause:
    """A hard-cancelled slot's synthetic report must say WHY and WHERE (task 3172).

    Before this, block_reason/block_phase both fell back to '', so a
    drain-cancelled task was indistinguishable in runs.db from a
    terminal-status cancel or an escalation-action teardown.
    """

    async def test_report_carries_the_attributed_cancel_reason(
        self, harness_for_run_slot: Harness
    ):
        """The reason stamped by hard_cancel_workflow reaches the report."""
        report = await _drive_cancelled_slot(
            harness_for_run_slot, '42', reason='terminal_status_cancel'
        )

        assert report.outcome == WorkflowOutcome.CANCELLED
        assert report.block_reason == 'terminal_status_cancel'

    async def test_report_carries_the_live_workflow_phase(
        self, harness_for_run_slot: Harness
    ):
        """block_phase reads the live workflow's own state, not a guess."""
        report = await _drive_cancelled_slot(
            harness_for_run_slot,
            '42',
            reason='terminal_status_cancel',
            state=WorkflowState.VERIFY,
        )

        assert report.block_phase == WorkflowState.VERIFY.value

    async def test_setup_phase_cancel_reports_dispatch_setup(
        self, harness_for_run_slot: Harness
    ):
        """A cancel landing before build_workflow must not raise UnboundLocalError."""
        report = await _drive_cancelled_slot(
            harness_for_run_slot, '42', cancel_during_setup=True
        )

        assert report.outcome == WorkflowOutcome.CANCELLED
        assert report.block_phase == 'dispatch_setup'


@pytest.mark.asyncio
class TestHardCancelReasonFallbackLadder:
    """With no attributed cause, the reason is derived — never guessed (task 3172)."""

    async def test_drain_cancel_reports_shutdown_drain(self, harness_for_run_slot: Harness):
        """A slot drained by run()'s finally is separable from every other cancel.

        _draining is a POSITIVE signal: it is set at the top of run()'s finally,
        before the only loop that cancels slot tasks on SIGTERM/SIGINT.
        """
        harness_for_run_slot._draining = True

        report = await _drive_cancelled_slot(harness_for_run_slot, '42')

        assert report.block_reason == 'shutdown_drain'

    async def test_unattributed_cancel_is_labelled_honestly(
        self, harness_for_run_slot: Harness
    ):
        """No cause and no drain gets a residue label, not an invented one."""
        harness_for_run_slot._draining = False

        report = await _drive_cancelled_slot(harness_for_run_slot, '42')

        assert report.block_reason == 'cancelled_unattributed'

    async def test_stamped_cause_is_consumed_once(self, harness_for_run_slot: Harness):
        """A re-dispatch of the same task cannot inherit a stale reason."""
        harness_for_run_slot._draining = False

        await _drive_cancelled_slot(harness_for_run_slot, '42', reason='terminal_status_cancel')

        assert '42' not in harness_for_run_slot._workflow_cancel_causes

    async def test_stamped_cause_wins_over_the_drain_flag(
        self, harness_for_run_slot: Harness
    ):
        """An explicit attribution outranks the drain inference."""
        harness_for_run_slot._draining = True

        report = await _drive_cancelled_slot(
            harness_for_run_slot, '42', reason='action_teardown:park'
        )

        assert report.block_reason == 'action_teardown:park'


# ---------------------------------------------------------------------------
# Cancel-source attribution — task 3172, step-19
#
# The fallback ladder above only stays honest if every in-process cancel
# SOURCE registers itself.  A source that silently omits `reason` lands in the
# `cancelled_unattributed` residue bucket, which is exactly the flattening
# this task exists to remove.  These tests pin the two in-process sources.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTerminalStatusWatcherAttributesItsCancel:
    """The terminal-status watcher names itself when it hard-cancels a slot."""

    async def test_hard_cancel_passes_terminal_status_cancel_reason(
        self, harness: Harness
    ):
        """Driving the scan past the threshold attributes the cancel.

        Reuses the watcher's own drive shape (two scans with threshold=2, the
        second crossing it) from test_harness_terminal_status_watcher.py, but
        patches ``hard_cancel_workflow`` so the assertion is on the CALL, not
        on the downstream report — the report side is covered above.
        """
        h = harness
        h.config = OrchestratorConfig(terminal_status_hard_cancel_polls=2)
        h._terminal_cancel_counts = {}
        h._workflow_cancel_events['6'] = asyncio.Event()
        h.scheduler.get_statuses = AsyncMock(return_value=({'6': 'cancelled'}, None))
        h.hard_cancel_workflow = MagicMock(return_value=True)

        # Poll 1: below threshold → soft cancel only.
        await h._scan_for_terminal_active_tasks()
        h.hard_cancel_workflow.assert_not_called()  # type: ignore[attr-defined]

        # Poll 2: at threshold → hard cancel, attributed.
        await h._scan_for_terminal_active_tasks()

        h.hard_cancel_workflow.assert_called_once()  # type: ignore[attr-defined]
        kwargs = h.hard_cancel_workflow.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs.get('reason') == 'terminal_status_cancel', (
            'terminal-status watcher must attribute its own hard cancel; '
            f'got reason={kwargs.get("reason")!r}'
        )
        # The pre-existing restamp contract must survive the new kwarg: only
        # the threshold-CROSSING call restamps the R3 grace window.
        assert kwargs.get('restamp') is True


@pytest.mark.asyncio
class TestCancelledReportReachesRunStore:
    """The runs.db half of the acceptance query (task 3172, step-21).

    events.db gets the new task_completed emit; runs.db gets the same two
    fields via ``_collect_done_reports`` → ``save_task_result``.  Pinning both
    keeps the two stores answerable by the SAME question.
    """

    async def test_save_task_result_receives_block_reason_and_phase(
        self, harness_for_run_slot: Harness
    ):
        h = harness_for_run_slot
        captured: list = []

        await _drive_cancelled_slot(
            h,
            '42',
            reason='action_teardown:restart',
            state=WorkflowState.EXECUTE,
            capture_task=captured,
        )

        # _collect_done_reports persists whatever the slot returned.  Wire the
        # run store only now: _run_slot itself must not need it.
        h._run_store = MagicMock()
        h._run_id = 'run-1'
        h.config = MagicMock()
        h.config.fused_memory.project_id = 'dark_factory'

        h._collect_done_reports(set(captured), [])

        h._run_store.save_task_result.assert_called_once()
        saved_report = h._run_store.save_task_result.call_args.args[1]
        assert saved_report.outcome == WorkflowOutcome.CANCELLED
        assert saved_report.block_reason == 'action_teardown:restart'
        assert saved_report.block_phase == WorkflowState.EXECUTE.value
