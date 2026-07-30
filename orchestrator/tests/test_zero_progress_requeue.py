"""Task 3068 / Part C — the zero-progress-requeue backstop.

Origin incident (reify esc-5556-1): ~24 tasks requeued ~349 times over 46h
without invoking a single agent, and NOTHING alarmed.  That is by design, and
that is the bug: ``workflow_types._disposition_table()`` sets
``counts_against_requeue_cap=False`` for the warm-lane dispositions, so
``Scheduler.record_requeue(counts_against_cap=False)`` is history-only —
neither ``_requeue_counts`` nor ``_transient_requeue_counts`` moves, so neither
ceiling in ``Harness._apply_retry_cap`` can ever trip.  A hard fault
masquerading as transient backpressure requeues forever, invisibly.

This module's detector is the only backstop for that class, so these tests pin
its shape tightly: a per-task CONSECUTIVE streak of zero-agent-invocation
requeues, reset by any other outcome, alarming once per unresolved incident.
"""

from __future__ import annotations

import pytest

from orchestrator.workflow import WorkflowOutcome

# ---------------------------------------------------------------------------
# Part C.1 — the pure streak tracker
# ---------------------------------------------------------------------------


class TestZeroProgressRequeueTracker:
    """Per-task consecutive zero-progress requeue counting, no I/O."""

    def _tracker(self):
        from orchestrator.zero_progress_requeue import ZeroProgressRequeueTracker

        return ZeroProgressRequeueTracker()

    def test_consecutive_zero_progress_requeues_accumulate(self):
        """Three zero-invocation requeues in a row return 1, 2, 3."""
        tracker = self._tracker()
        streaks = [
            tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
            for _ in range(3)
        ]
        assert streaks == [1, 2, 3]
        assert tracker.streak('3068') == 3

    def test_requeue_with_real_work_resets(self):
        """A requeue that DID invoke an agent is progress — reset the streak.

        This is the false-positive guard: genuine slow progress (a task that
        requeues after doing real work) must never accumulate toward an alarm.
        """
        tracker = self._tracker()
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)

        assert tracker.record(
            '3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=1
        ) == 0
        assert tracker.streak('3068') == 0

        assert tracker.record(
            '3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0
        ) == 1

    def test_done_resets(self):
        """A DONE outcome clears the streak."""
        tracker = self._tracker()
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)

        assert tracker.record(
            '3068', outcome=WorkflowOutcome.DONE, agent_invocations=0
        ) == 0
        assert tracker.streak('3068') == 0

    def test_blocked_resets(self):
        """A BLOCKED outcome clears the streak — it is a terminal, visible state."""
        tracker = self._tracker()
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)

        assert tracker.record(
            '3068', outcome=WorkflowOutcome.BLOCKED, agent_invocations=0
        ) == 0
        assert tracker.streak('3068') == 0

    def test_streaks_are_per_task(self):
        """Two tasks keep independent streaks and do not interfere."""
        tracker = self._tracker()
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        tracker.record('4000', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)

        assert tracker.streak('3068') == 2
        assert tracker.streak('4000') == 1

        # Resetting one must not touch the other.
        tracker.record('4000', outcome=WorkflowOutcome.DONE, agent_invocations=0)
        assert tracker.streak('3068') == 2
        assert tracker.streak('4000') == 0

    def test_streak_of_unknown_task_is_zero(self):
        """Reading a never-seen task is 0, not a KeyError."""
        assert self._tracker().streak('never-seen') == 0

    def test_reset_pops_the_key_rather_than_storing_zero(self):
        """A reset must POP, not store 0.

        The orchestrator runs for weeks; storing a zero per task would leak one
        dict entry per task forever.  Popping keeps the tracker's footprint
        proportional to the number of tasks CURRENTLY looping, which for a
        healthy fleet is zero.
        """
        tracker = self._tracker()
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        tracker.record('4000', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        assert set(tracker.tracked_task_ids()) == {'3068', '4000'}

        tracker.record('3068', outcome=WorkflowOutcome.DONE, agent_invocations=0)
        assert set(tracker.tracked_task_ids()) == {'4000'}
        assert tracker.streak('3068') == 0

        # A DONE for a task that was never tracked must also not create an entry.
        tracker.record('9999', outcome=WorkflowOutcome.DONE, agent_invocations=0)
        assert set(tracker.tracked_task_ids()) == {'4000'}

    @pytest.mark.parametrize(
        'outcome',
        [
            WorkflowOutcome.DONE,
            WorkflowOutcome.PLANNED,
            WorkflowOutcome.BLOCKED,
            WorkflowOutcome.ESCALATED,
            WorkflowOutcome.CANCELLED,
            WorkflowOutcome.MERGE_DEFERRED,
            WorkflowOutcome.SOFT_CANCELLED,
        ],
    )
    def test_every_non_requeue_outcome_resets(self, outcome):
        """Only REQUEUED can ever accumulate — every other outcome resets."""
        tracker = self._tracker()
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        assert tracker.record('3068', outcome=outcome, agent_invocations=0) == 0
        assert tracker.streak('3068') == 0
