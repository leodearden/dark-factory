"""Zero-progress requeue detection — the backstop the requeue cap cannot be.

Task 3068. Origin incident: reify esc-5556-1, in which ~24 tasks requeued ~349
times over 46 hours without invoking a single agent, and nothing alarmed.

WHY THIS MODULE EXISTS
----------------------
Nothing alarmed *by design*, and that is precisely the gap.
``workflow_types._disposition_table()`` sets ``counts_against_requeue_cap=False``
for the warm-lane dispositions.  ``Harness._apply_retry_cap`` threads that
straight into ``Scheduler.record_requeue(counts_against_cap=False)``, whose
``False`` route is history-only: neither ``_requeue_counts`` nor
``_transient_requeue_counts`` moves, so neither ceiling in ``_apply_retry_cap``
can ever trip.  That is correct for genuine transient backpressure — a warm
lane briefly unavailable should not burn a task's retry budget — but it means a
HARD fault masquerading as transient backpressure requeues forever, invisibly,
consuming a dispatch slot on every pass.

The requeue cap therefore structurally cannot see this class of failure.  This
module is the only backstop for it, which is why the alert it files is
``blocking``/level-1 rather than an advisory note nobody routes: an
informational signal here would reproduce the original 46h silence.

SHAPE
-----
Two deliberately separate halves, following ``merge_skew_tripwire.py``:

* :class:`ZeroProgressRequeueTracker` — pure per-task streak counting.  No I/O,
  no config, no escalation dependency, trivially unit-testable.
* ``emit_zero_progress_requeue_alert`` (added by task 3068 step-10) — the
  fail-open escalation filer, with every collaborator passed as an explicit
  keyword argument.

The signal is a per-task CONSECUTIVE streak rather than a wall-clock window or
a per-run-loop-cycle count: the harness has no clean cycle boundary to hang a
counter on (``--until-idle`` breaks out before ever reaching run()'s per-cycle
reset), while a consecutive streak is exactly the incident's measured shape,
needs no clock, and is deterministic.  Any non-zero-progress outcome — DONE,
BLOCKED, or a requeue that DID invoke an agent — resets it, so genuine slow
progress never accumulates a false alarm.
"""

from __future__ import annotations

from collections.abc import Iterable

from orchestrator.workflow_types import WorkflowOutcome


class ZeroProgressRequeueTracker:
    """Counts CONSECUTIVE zero-agent-invocation requeues, per task.

    Pure in-memory state with no I/O — the harness owns one instance for the
    life of a run and calls :meth:`record` from the single per-report
    chokepoint (``Harness._apply_retry_cap``) that already sees every outcome.
    """

    def __init__(self) -> None:
        #: task_id -> current consecutive zero-progress requeue count.  Entries
        #: are POPPED on reset rather than set to 0, so the dict stays
        #: proportional to the number of tasks CURRENTLY looping (zero, for a
        #: healthy fleet) instead of growing one entry per task ever dispatched
        #: over a weeks-long run.
        self._streaks: dict[str, int] = {}

    def record(
        self,
        task_id: str,
        *,
        outcome: WorkflowOutcome,
        agent_invocations: int,
    ) -> int:
        """Fold one completed dispatch into ``task_id``'s streak.

        Args:
            task_id: The task the dispatch belonged to.
            outcome: The dispatch's terminal :class:`WorkflowOutcome`.
            agent_invocations: How many agents the dispatch actually invoked.

        Returns:
            The task's NEW consecutive zero-progress streak — incremented when
            the dispatch both requeued and invoked no agent (a burned slot with
            nothing to show for it), and ``0`` for every other combination.
        """
        if outcome is WorkflowOutcome.REQUEUED and agent_invocations == 0:
            streak = self._streaks.get(task_id, 0) + 1
            self._streaks[task_id] = streak
            return streak

        # Any other outcome is progress (or at least a visible terminal state
        # that other machinery already alarms on) — forget the task entirely.
        self._streaks.pop(task_id, None)
        return 0

    def streak(self, task_id: str) -> int:
        """Return ``task_id``'s current streak — ``0`` if never seen."""
        return self._streaks.get(task_id, 0)

    def tracked_task_ids(self) -> Iterable[str]:
        """Return the task ids currently holding a non-zero streak.

        Introspection hook: exists so tests (and any future operator dump) can
        assert the pop-on-reset footprint contract directly.
        """
        return tuple(self._streaks)
