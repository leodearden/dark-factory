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

import logging
from collections.abc import Iterable
from typing import Any

from orchestrator.workflow_types import WorkflowOutcome

logger = logging.getLogger(__name__)

#: Sentinel task_id prefix for the alert.  A synthetic id is required because
#: no per-task steward can be dispatched for this alarm — it is a monitor
#: signal ABOUT a task, not work ON one.  Same shape as
#: merge_liveness's ``_verify_host_unreachable_sentinel``.
ZERO_PROGRESS_SENTINEL_PREFIX = '__zero_progress_requeue__'

#: Category used for both the filed escalation and the has_open_l1 signature
#: filter.  Filtering on category (task 2757's rationale) keeps an UNRELATED
#: open L1 from silently suppressing this signal.
_CATEGORY = 'risk_identified'


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


def emit_zero_progress_requeue_alert(
    *,
    escalation_queue: Any,
    event_store: Any,
    task_id: str,
    streak: int,
    threshold: int,
    block_reason: str,
    block_phase: str,
) -> bool:
    """File ONE blocking L1 when ``task_id``'s zero-progress streak hits the bar.

    Fully fail-open: every external call is individually guarded AND the whole
    body is wrapped again, so this function NEVER raises into
    ``Harness._apply_retry_cap``.  A bug or a full disk here must not disturb
    requeue-cap accounting on the dispatch hot path.

    Args:
        escalation_queue: The :class:`EscalationQueue`, or ``None`` (no-op).
        event_store: Optional event store for the paired telemetry event.
        task_id: The REAL task id that is looping.
        streak: Its current consecutive zero-progress requeue count.
        threshold: ``config.zero_progress_requeue.threshold``.
        block_reason: ``TaskReport.block_reason`` for the most recent requeue —
            the WHY that task 3068 parts A/B make durable, reported inline here
            so the operator does not have to go digging in a rotated log.
        block_phase: ``TaskReport.block_phase`` for the most recent requeue.

    Returns:
        ``True`` when a NEW escalation was filed; ``False`` when below
        threshold, deduped against a still-open alert, or suppressed by a
        failure.  A telemetry (event-emit) failure does NOT flip this to
        ``False`` — the escalation is the load-bearing output, and reporting
        ``False`` after actually filing one would be a lie.
    """
    if escalation_queue is None:
        return False

    # Fire on >= rather than ==: if an operator resolves the alert while the
    # loop is still running, the very next zero-progress requeue re-files it,
    # so the signal cannot be permanently silenced by a premature resolve.
    # The streak is deliberately NOT cleared on fire — dedup below, not a
    # counter reset, is what prevents a sawtooth re-file every N requeues.
    if streak < threshold:
        return False

    try:
        sentinel = f'{ZERO_PROGRESS_SENTINEL_PREFIX}{task_id}'

        # Dedup: one alert per unresolved incident.  The category filter keeps
        # an unrelated open L1 on this sentinel from suppressing us.
        if escalation_queue.has_open_l1(sentinel, category=_CATEGORY):
            return False

        from escalation.models import Escalation  # noqa: PLC0415 — optional dep

        summary = (
            f'Task {task_id} has requeued {streak} consecutive times without '
            f'invoking a single agent (threshold {threshold}) — zero progress, '
            f'burning a dispatch slot each pass'
        )
        detail = (
            f'Task: {task_id}\n'
            f'Consecutive zero-agent-invocation requeues: {streak}\n'
            f'Alert threshold: {threshold}\n'
            f'Most recent block_reason: {block_reason or "(none recorded)"}\n'
            f'Most recent block_phase: {block_phase or "(none recorded)"}\n'
            '\n'
            f'Every one of the last {streak} dispatches of task {task_id} '
            'consumed a workflow slot and returned REQUEUED having invoked no '
            'agent at all.  That is not slow progress — it is no progress.\n'
            '\n'
            'This alert exists because the per-task requeue cap CANNOT see '
            'this class of failure.  workflow_types._disposition_table() sets '
            'counts_against_requeue_cap=False for the warm-lane dispositions, '
            'so Scheduler.record_requeue takes its history-only route: neither '
            '_requeue_counts nor _transient_requeue_counts moves, and neither '
            'ceiling in Harness._apply_retry_cap can ever trip.  That is '
            'correct for genuine transient backpressure, but it means a HARD '
            'fault masquerading as backpressure requeues forever, invisibly '
            '(origin incident: ~24 tasks x ~349 requeues over 46h).'
        )

        esc = Escalation(
            id=escalation_queue.make_id(sentinel),
            task_id=sentinel,
            agent_role='orchestrator-zero-progress-requeue',
            severity='blocking',
            level=1,
            category=_CATEGORY,
            summary=summary,
            detail=detail,
            suggested_action=(
                f'Diagnose why task {task_id} cannot start: the block_reason '
                f'above ({block_reason or "unrecorded"}) names the disposition '
                'that keeps requeueing it.  Check whether the underlying '
                'resource (e.g. the warm-lane pool) is genuinely down rather '
                'than momentarily busy; if it is, the disposition is '
                'mis-classified as transient in '
                'workflow_types._disposition_table() and should count against '
                'the requeue cap.  To stop the bleeding now, block or cancel '
                f'task {task_id} so it stops consuming dispatch slots.  '
                'Retune or silence this detector live via the green-tier '
                'config section zero_progress_requeue.{enabled,threshold}.'
            ),
        )
    except Exception as exc:  # noqa: BLE001 — fail-open backstop
        logger.warning(
            'zero-progress-requeue alert for task %s could not be built '
            '(non-fatal): %s', task_id, exc,
        )
        return False

    # The escalation and the event are submitted under INDIVIDUAL try/excepts:
    # the escalation is the load-bearing output and the event is telemetry, so
    # a failure of the second must not cost us the first.
    try:
        escalation_queue.submit(esc)
    except Exception as exc:  # noqa: BLE001 — fail-open backstop
        logger.warning(
            'zero-progress-requeue escalation for task %s could not be filed '
            '(non-fatal): %s', task_id, exc,
        )
        return False

    logger.warning(
        'Zero-progress requeue alert filed for task %s: %d consecutive '
        'zero-agent-invocation requeues (threshold %d, reason=%r, phase=%r)',
        task_id, streak, threshold, block_reason, block_phase,
    )

    if event_store is not None:
        try:
            from orchestrator.event_store import EventType  # noqa: PLC0415

            event_store.emit(
                EventType.zero_progress_requeue,
                task_id=task_id,  # the REAL id, not the sentinel — stays
                                  # joinable against task_completed
                data={
                    'streak': streak,
                    'threshold': threshold,
                    # Same key vocabulary as the task_completed payload (part A)
                    # so one query shape serves both.
                    'reason': block_reason,
                    'block_phase': block_phase,
                },
            )
        except Exception as exc:  # noqa: BLE001 — telemetry is best-effort
            logger.warning(
                'zero-progress-requeue event for task %s could not be emitted '
                '(non-fatal, escalation WAS filed): %s', task_id, exc,
            )

    return True
