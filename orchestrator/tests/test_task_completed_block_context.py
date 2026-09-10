"""Task 3068 / Part A — ``EventType.task_completed`` carries the block context.

Origin incident (reify esc-5556-1): a 46h warm-lane requeue loop was
forensically unqueryable because ``events.db``'s ``task_completed`` payload
was counters-only — it recorded THAT a dispatch requeued but never WHY.
``TaskReport.block_reason`` / ``.block_phase`` were both already in scope at
the emit site and simply not passed through.

These tests pin the fixed contract:

* the REQUEUED case carries the classified reason and the PRE-block working
  phase (``blocked_from_phase``, not the terminal machine state);
* the DONE case carries BOTH keys as empty strings — always present, never
  omitted — so ``json_extract(data,'$.reason') IS NULL`` unambiguously means
  "this event predates task 3068" rather than "clean exit";
* the seven pre-existing counter keys are unchanged (regression guard).

Fixture idiom copied from ``test_harness_deterministic_dispatch.py:26-53``
(Harness(mock_orch_config) under patch of McpLifecycle/OverrideStore/
Scheduler/BriefingAssembler, then a MagicMock scheduler) plus the
``build_workflow``-patching ``_run_slot`` driver from
``test_crash_recovery.py:169-196``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _recording_event_store import _RecordingEventStore

from orchestrator.event_store import EventType
from orchestrator.harness import Harness, TaskReport
from orchestrator.workflow import TerminalReport, WorkflowOutcome, WorkflowState

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

#: The counter keys the emit carried BEFORE task 3068.  Pinned as a literal
#: set so dropping or renaming one is a test failure, not a silent telemetry
#: regression for every existing events.db consumer.
_LEGACY_COUNTER_KEYS = frozenset({
    'outcome',
    'agent_invocations',
    'execute_iterations',
    'verify_attempts',
    'review_cycles',
    'steward_cost_usd',
    'steward_invocations',
})


@pytest.fixture
def harness(mock_orch_config) -> Harness:
    """Harness with mocked internals, wired with a recording event store."""
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler.is_deterministic = MagicMock(return_value=False)
    h.scheduler.get_status = AsyncMock(return_value='in-progress')
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.get_task = AsyncMock(return_value={'id': '3068', 'metadata': {}})
    h.scheduler.update_task = AsyncMock(return_value=True)
    h.scheduler.release = MagicMock()

    h.event_store = _RecordingEventStore()  # type: ignore[assignment]
    return h


def _terminal(
    outcome: WorkflowOutcome,
    *,
    reason: str = '',
    detail: str = '',
    blocked_from_phase: WorkflowState | None = None,
) -> TerminalReport:
    """Build a ``TerminalReport`` for ``workflow.run()`` to return."""
    return TerminalReport(
        outcome=outcome,
        reason=reason,
        phase=WorkflowState.BLOCKED if reason else WorkflowState.DONE,
        detail=detail,
        category=None,
        blocked_from_phase=blocked_from_phase,
    )


async def _drive_slot(
    harness: Harness,
    terminal: TerminalReport | None,
    task_id: str = '3068',
    *,
    cancel: bool = False,
    state: WorkflowState = WorkflowState.EXECUTE,
) -> TaskReport:
    """Run ``_run_slot`` with ``build_workflow`` patched to return ``terminal``.

    ``cancel=True`` (task 3172) instead makes ``workflow.run()`` raise
    ``asyncio.CancelledError``, driving the hard-cancel branch of ``_run_slot``.
    ``state`` is the live workflow phase the cancel lands in, which that branch
    reads for ``block_phase``.
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'title': f'task {task_id}'}
    sem = MagicMock()
    sem.release = MagicMock()

    with patch('orchestrator.harness.build_workflow') as MockWorkflow:
        mock_wf = AsyncMock()
        if cancel:
            mock_wf.run.side_effect = asyncio.CancelledError()
        else:
            mock_wf.run.return_value = terminal
        mock_wf.state = state
        mock_wf.metrics = MagicMock(
            total_cost_usd=1.25,
            total_duration_ms=4200,
            agent_invocations=0,
            execute_iterations=0,
            verify_attempts=0,
            review_cycles=0,
        )
        # No steward => steward_cost / steward_invocations stay 0.
        mock_wf._steward = None
        MockWorkflow.return_value = mock_wf
        report = await harness._run_slot(assignment, sem)
    # ``_run_slot`` is typed ``TaskReport | None``; every path this driver
    # exercises returns a report (terminal or synthetic-CANCELLED), so this
    # states an existing precondition rather than weakening a check.
    assert report is not None, 'expected _run_slot to return a TaskReport, got None'
    return report


def _task_completed_payloads(harness: Harness) -> list[dict]:
    """Return the ``data`` dict of every recorded ``task_completed`` event."""
    return [
        entry['data']
        for (etype, entry) in harness.event_store.events  # type: ignore[attr-defined]
        if etype == EventType.task_completed
    ]


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTaskCompletedBlockContext:
    """``task_completed`` must carry the WHY, not just the counters."""

    async def test_requeued_emits_reason_and_block_phase(self, harness: Harness):
        """A REQUEUED dispatch records its classified reason + pre-block phase."""
        terminal = _terminal(
            WorkflowOutcome.REQUEUED,
            reason='warm_lane_pool_hard_down',
            detail='pool exhausted; no warm lane could be claimed',
            blocked_from_phase=WorkflowState.PLAN,
        )

        await _drive_slot(harness, terminal)

        payloads = _task_completed_payloads(harness)
        assert len(payloads) == 1, f'expected exactly one task_completed, got {payloads}'
        data = payloads[0]
        assert data['reason'] == 'warm_lane_pool_hard_down'
        assert data['block_phase'] == WorkflowState.PLAN.value

    async def test_done_emits_both_keys_as_empty_strings(self, harness: Harness):
        """A clean DONE emits BOTH keys present-and-empty, never omitted.

        This is the load-bearing half of the contract: after this lands, a
        NULL from ``json_extract(data,'$.reason')`` means "pre-3068 event",
        and ``''`` means "clean exit".  Conditional omission would make those
        two cases indistinguishable — the exact ambiguity that made the
        origin incident unqueryable.
        """
        await _drive_slot(harness, _terminal(WorkflowOutcome.DONE))

        payloads = _task_completed_payloads(harness)
        assert len(payloads) == 1
        data = payloads[0]
        assert 'reason' in data, 'reason key must be present even on a clean exit'
        assert 'block_phase' in data, 'block_phase key must be present even on a clean exit'
        assert data['reason'] == ''
        assert data['block_phase'] == ''

    async def test_legacy_counter_keys_unchanged(self, harness: Harness):
        """The seven pre-3068 counter keys survive the additive change."""
        terminal = _terminal(
            WorkflowOutcome.REQUEUED,
            reason='warm_lane_pool_hard_down',
            blocked_from_phase=WorkflowState.EXECUTE,
        )

        await _drive_slot(harness, terminal)

        data = _task_completed_payloads(harness)[0]
        assert set(data) >= _LEGACY_COUNTER_KEYS, (
            f'missing pre-existing counter keys: {_LEGACY_COUNTER_KEYS - set(data)}'
        )
        assert data['outcome'] == WorkflowOutcome.REQUEUED.value
        assert data['agent_invocations'] == 0
        assert data['execute_iterations'] == 0
        assert data['verify_attempts'] == 0
        assert data['review_cycles'] == 0
        assert data['steward_cost_usd'] == 0.0
        assert data['steward_invocations'] == 0
        # Additive only: exactly the legacy keys plus the two new ones.
        assert set(data) == _LEGACY_COUNTER_KEYS | {'reason', 'block_phase'}


# ---------------------------------------------------------------------------
# Task 3172 — the hard-cancel path emits too
#
# Discovered while verifying this task's premises: the emit above sits on the
# SUCCESS path only, so a hard-cancelled slot emitted NO task_completed at all
# and its sole durable trace was the runs.db task_results row.  events.db had
# nothing to GROUP BY, which is precisely the query the acceptance names
# ("separate drain-cancelled tasks from other cancellations").
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTaskCompletedOnHardCancel:
    """A hard-cancelled slot must be visible in events.db, with the same keys."""

    async def test_hard_cancel_emits_exactly_one_task_completed(self, harness: Harness):
        """Outcome, reason and phase all reach the payload."""
        harness._workflow_cancel_causes['3068'] = 'terminal_status_cancel'

        report = await _drive_slot(harness, None, cancel=True, state=WorkflowState.VERIFY)

        assert report.outcome == WorkflowOutcome.CANCELLED
        payloads = _task_completed_payloads(harness)
        assert len(payloads) == 1, (
            f'expected exactly one task_completed on the cancel path, got {payloads}'
        )
        data = payloads[0]
        assert data['outcome'] == WorkflowOutcome.CANCELLED.value
        # The reason must be the RESOLVED cancel cause — the same value the
        # synthetic report carries into runs.db, so the two stores agree.
        assert data['reason'] == 'terminal_status_cancel'
        assert data['reason'] == report.block_reason
        assert data['block_phase'] == WorkflowState.VERIFY.value
        assert data['block_phase'] == report.block_phase

    async def test_drain_cancel_is_separable_in_the_payload(self, harness: Harness):
        """The acceptance query: drain-cancelled vs any other cancellation."""
        harness._draining = True

        await _drive_slot(harness, None, cancel=True)

        data = _task_completed_payloads(harness)[0]
        assert data['outcome'] == WorkflowOutcome.CANCELLED.value
        assert data['reason'] == 'shutdown_drain'

    async def test_cancel_payload_keeps_the_legacy_key_vocabulary(
        self, harness: Harness
    ):
        """Same seven counter keys as DONE/REQUEUED — a uniform row shape.

        The synthetic report has no metrics (the workflow never returned one),
        so the counters are zeros rather than omitted: DF 3068's contract is
        that these keys are ALWAYS present, so a consumer can read every
        task_completed row with one schema regardless of outcome.
        """
        harness._draining = False

        await _drive_slot(harness, None, cancel=True)

        data = _task_completed_payloads(harness)[0]
        assert set(data) >= _LEGACY_COUNTER_KEYS, (
            f'missing pre-existing counter keys: {_LEGACY_COUNTER_KEYS - set(data)}'
        )
        assert set(data) == _LEGACY_COUNTER_KEYS | {'reason', 'block_phase'}
        assert data['agent_invocations'] == 0
        assert data['execute_iterations'] == 0
        assert data['verify_attempts'] == 0
        assert data['review_cycles'] == 0
        assert data['steward_cost_usd'] == 0.0
        assert data['steward_invocations'] == 0
        # Honest residue, not a guess: no stamped cause and not draining.
        assert data['reason'] == 'cancelled_unattributed'
