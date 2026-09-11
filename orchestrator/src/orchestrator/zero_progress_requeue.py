"""Zero-progress requeue detection — the backstop the requeue cap cannot be.

Task 3068. Origin incident: reify esc-5556-1, in which ~24 tasks requeued ~349
times over 46 hours without invoking a single agent, and nothing alarmed.

WHY THIS MODULE EXISTS
----------------------
**This docstring is the single canonical explanation.**  Every other site that
touches this detector (``config.ZeroProgressRequeueConfig``, the
``zero_progress_requeue`` stanza in ``defaults.yaml``, the inline comment in
``Harness._apply_retry_cap``, the ``EventType.zero_progress_requeue`` member,
the escalation body built below, and the test module) carries a ONE-LINE
pointer here instead of a copy — task 2988 already changed the disposition
table once, and seven near-verbatim copies of the causal chain would have gone
stale six ways.

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

THE TWO-DIMENSIONAL SIGNAL
--------------------------
A streak alone is not enough.  The dispositions this watches are non-counting
*because they represent normal busy-fleet backpressure*: on a saturated fleet
where ``max_concurrent_tasks`` exceeds the warm-lane pool size, a low-priority
task can genuinely lose the warm-lane race several dispatches in a row within
seconds.  Alarming ``blocking``/L1 on that would page a human for healthy
contention.

So the alarm requires BOTH:

* ``streak >= threshold`` — N consecutive requeues that invoked no agent, and
* ``span >= min_span_seconds`` — that streak spanning real wall-clock time.

The span is what separates "lost the pool race five times in eight seconds"
(normal, silent) from "has not started in fifteen minutes" (the incident).
Both halves are green-tier hot-tunable, so a noisy detector can be retuned live.

SHAPE
-----
Three deliberately separate pieces, following ``merge_skew_tripwire.py`` for
the pure/injectable split and ``merge_liveness.py`` for the alarm/recovery pair:

* :class:`ZeroProgressRequeueTracker` — pure per-task streak + span tracking.
  No I/O, no config, no escalation dependency, trivially unit-testable.
* :func:`emit_zero_progress_requeue_alert` — the fail-open escalation filer.
* :func:`resolve_zero_progress_requeue_alert` — its recovery half.  Without it
  the filed L1 would sit ``pending`` forever after the loop self-healed, and —
  worse — the ``has_open_l1`` dedup would then permanently suppress the alarm
  for a genuine LATER recurrence on the same task, silencing the detector after
  one incident per task for the whole life of the queue.

Every collaborator is passed as an explicit keyword argument.

The signal is a per-task CONSECUTIVE streak rather than a per-run-loop-cycle
count: the harness has no clean cycle boundary to hang a counter on
(``--until-idle`` breaks out before ever reaching run()'s per-cycle reset),
while a consecutive streak is exactly the incident's measured shape and is
deterministic.  Any non-zero-progress outcome — DONE, BLOCKED, or a requeue
that DID invoke an agent — resets it, so genuine slow progress never
accumulates a false alarm.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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

#: agent_role stamped on both the alarm and its resolution.
_ROLE = 'orchestrator-zero-progress-requeue'


@dataclass
class _Streak:
    """One task's live zero-progress streak."""

    count: int
    #: Clock reading at the FIRST zero-progress requeue of this streak.  Held
    #: on the streak (not per-record) so the span measures the whole loop, and
    #: dropped with it on reset.
    started_at: float


class ZeroProgressRequeueTracker:
    """Counts CONSECUTIVE zero-agent-invocation requeues, per task, with span.

    Pure in-memory state with no I/O — the harness owns one instance for the
    life of a run and calls :meth:`record` from the single per-report
    chokepoint (``Harness._apply_retry_cap``) that already sees every outcome.

    Args:
        clock: Monotonic seconds source, injected for testability.
            ``time.monotonic`` (not wall clock) so a span cannot be corrupted
            by an NTP step or a DST jump during a multi-hour loop.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        #: task_id -> live streak.  Entries are POPPED on reset rather than set
        #: to 0, so the dict stays proportional to the number of tasks
        #: CURRENTLY looping (zero, for a healthy fleet) instead of growing one
        #: entry per task ever dispatched over a weeks-long run.
        self._streaks: dict[str, _Streak] = {}

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
            live = self._streaks.get(task_id)
            if live is None:
                # First of a new streak — stamp the span origin.
                self._streaks[task_id] = _Streak(count=1, started_at=self._clock())
                return 1
            live.count += 1
            return live.count

        # Any other outcome is progress (or at least a visible terminal state
        # that other machinery already alarms on) — forget the task entirely.
        self._streaks.pop(task_id, None)
        return 0

    def streak(self, task_id: str) -> int:
        """Return ``task_id``'s current streak — ``0`` if never seen."""
        live = self._streaks.get(task_id)
        return live.count if live is not None else 0

    def span(self, task_id: str) -> float:
        """Return seconds elapsed since ``task_id``'s streak began.

        ``0.0`` when the task holds no live streak.  This is the second half of
        the alarm predicate (see the module docstring): a streak that accrued
        in seconds is fleet contention, not a stuck task.
        """
        live = self._streaks.get(task_id)
        if live is None:
            return 0.0
        return max(0.0, self._clock() - live.started_at)

    def tracked_task_ids(self) -> Iterable[str]:
        """Return the task ids currently holding a non-zero streak.

        Introspection hook: exists so tests (and any future operator dump) can
        assert the pop-on-reset footprint contract directly.
        """
        return tuple(self._streaks)


def _fmt_duration(seconds: float) -> str:
    """Render a span the way an operator reads it."""
    if seconds < 90.0:
        return f'{seconds:.0f}s'
    if seconds < 5400.0:
        return f'{seconds / 60.0:.1f} min'
    return f'{seconds / 3600.0:.1f} h'


def emit_zero_progress_requeue_alert(
    *,
    escalation_queue: Any,
    event_store: Any,
    task_id: str,
    streak: int,
    threshold: int,
    span_seconds: float,
    min_span_seconds: float,
    block_reason: str,
    block_phase: str,
    filed_at: dict[str, int] | None = None,
) -> bool:
    """File ONE blocking L1 when ``task_id``'s zero-progress loop clears the bar.

    Both halves of the predicate must hold — ``streak >= threshold`` AND
    ``span_seconds >= min_span_seconds``.  See the module docstring for why a
    streak alone would page a human for ordinary warm-lane contention.

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
        span_seconds: Wall-clock seconds the streak has spanned
            (``ZeroProgressRequeueTracker.span``).
        min_span_seconds: ``config.zero_progress_requeue.min_span_seconds``.
            ``0`` disables the time half of the predicate.
        block_reason: ``TaskReport.block_reason`` for the most recent requeue —
            the WHY that task 3068 parts A/B make durable, reported inline here
            so the operator does not have to go digging in a rotated log.
        block_phase: ``TaskReport.block_phase`` for the most recent requeue.
        filed_at: Optional caller-owned memo (``task_id -> streak at last disk
            check``) used to keep this off the filesystem on the dispatch hot
            path — see the dedup comment below.  ``None`` means "always ask the
            queue", which is correct but does a full pending-queue scan per
            call.

    Returns:
        ``True`` when a NEW escalation was filed; ``False`` when below either
        half of the threshold, deduped against a still-open alert, or
        suppressed by a failure.  A telemetry (event-emit) failure does NOT
        flip this to ``False`` — the escalation is the load-bearing output, and
        reporting ``False`` after actually filing one would be a lie.
    """
    if escalation_queue is None:
        return False

    # Fire on >= rather than ==: if an operator resolves the alert while the
    # loop is still running, a later re-check re-files it, so the signal cannot
    # be permanently silenced by a premature resolve.  The streak is
    # deliberately NOT cleared on fire — dedup below, not a counter reset, is
    # what prevents a sawtooth re-file every N requeues.
    if streak < threshold:
        return False

    # Second half of the predicate: a streak that accrued in seconds is fleet
    # contention losing the warm-lane race, not a stuck task.  Checked BEFORE
    # any filesystem access so the common busy-fleet case stays free.
    if span_seconds < min_span_seconds:
        return False

    try:
        sentinel = f'{ZERO_PROGRESS_SENTINEL_PREFIX}{task_id}'

        # Dedup: one alert per unresolved incident.  ``has_open_l1`` globs and
        # JSON-parses every pending escalation in the queue, and this runs on
        # every completed dispatch of every looping task — i.e. hardest during
        # exactly the many-tasks-looping incident this exists to catch.  So the
        # caller's memo answers it for free, and we only go to disk once every
        # ``threshold`` further dispatches.  That bounds the scan rate while
        # still re-filing (within `threshold` dispatches) if an operator
        # resolves the alarm prematurely while the loop grinds on.
        if filed_at is not None:
            last_checked = filed_at.get(task_id)
            if last_checked is not None and streak - last_checked < threshold:
                return False

        # The category filter keeps an unrelated open L1 on this sentinel from
        # suppressing us.
        if escalation_queue.has_open_l1(sentinel, category=_CATEGORY):
            if filed_at is not None:
                filed_at[task_id] = streak
            return False

        from escalation.models import Escalation  # noqa: PLC0415 — optional dep

        span_str = _fmt_duration(span_seconds)
        summary = (
            f'Task {task_id} has requeued {streak} consecutive times over '
            f'{span_str} without invoking a single agent '
            f'(threshold {threshold}) — zero progress, burning a dispatch '
            'slot each pass'
        )
        detail = (
            f'Task: {task_id}\n'
            f'Consecutive zero-agent-invocation requeues: {streak}\n'
            f'Elapsed since the streak began: {span_str}\n'
            f'Alert thresholds: {threshold} requeues AND '
            f'{_fmt_duration(min_span_seconds)} elapsed\n'
            f'Most recent block_reason: {block_reason or "(none recorded)"}\n'
            f'Most recent block_phase: {block_phase or "(none recorded)"}\n'
            '\n'
            f'Every one of the last {streak} dispatches of task {task_id} '
            'consumed a workflow slot and returned REQUEUED having invoked no '
            'agent at all.  That is not slow progress — it is no progress.\n'
            '\n'
            'The per-task requeue cap CANNOT see this class of failure: the '
            'warm-lane dispositions are non-counting, so neither ceiling in '
            'Harness._apply_retry_cap can trip for them.  The full causal '
            'chain, and why this detector is the only backstop, is documented '
            'in orchestrator/src/orchestrator/zero_progress_requeue.py '
            '(module docstring).  Origin incident: reify esc-5556-1, ~24 tasks '
            'x ~349 requeues over 46h.'
        )

        esc = Escalation(
            id=escalation_queue.make_id(sentinel),
            task_id=sentinel,
            agent_role=_ROLE,
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
                'config section zero_progress_requeue.{enabled,threshold,'
                'min_span_seconds}.'
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

    if filed_at is not None:
        filed_at[task_id] = streak

    logger.warning(
        'Zero-progress requeue alert filed for task %s: %d consecutive '
        'zero-agent-invocation requeues over %s (threshold %d, reason=%r, '
        'phase=%r)',
        task_id, streak, _fmt_duration(span_seconds), threshold,
        block_reason, block_phase,
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
                    'span_seconds': round(span_seconds, 3),
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


def resolve_zero_progress_requeue_alert(
    *,
    escalation_queue: Any,
    event_store: Any,
    task_id: str,
    recovered_streak: int,
    threshold: int,
    filed_at: dict[str, int] | None = None,
) -> bool:
    """Resolve a filed alarm once ``task_id`` starts making progress again.

    The recovery half of the ``merge_liveness`` alarm/recovery pair, and NOT
    optional polish: ``emit_zero_progress_requeue_alert`` dedups on
    ``has_open_l1``, so an L1 left ``pending`` after the loop self-healed would
    (a) leave an operator holding a blocking alarm for a cleared condition and
    (b) permanently suppress the alarm for a genuine LATER recurrence on the
    same task — the detector would silence itself after one incident per task,
    for the life of the queue.

    Called on the streak-reset transition, i.e. whenever
    ``ZeroProgressRequeueTracker.record`` returns 0 after a non-zero streak.
    Deliberately NOT gated on ``config.zero_progress_requeue.enabled``:
    disabling the detector must not strand an already-filed blocking alarm.

    Unlike ``merge_liveness``, recovery files no paired info escalation — a
    fleet clearing many simultaneous loops would churn the queue for a
    condition the ``zero_progress_requeue_recovered`` event already records.

    Args:
        escalation_queue: The :class:`EscalationQueue`, or ``None`` (no-op).
        event_store: Optional event store for the recovery event.
        task_id: The REAL task id that just made progress.
        recovered_streak: The streak immediately BEFORE the reset.
        threshold: ``config.zero_progress_requeue.threshold``.
        filed_at: The same caller-owned memo passed to the emitter.  Its entry
            for ``task_id`` is dropped here, so the next incident on this task
            re-files rather than short-circuiting on a stale memo.

    Returns:
        ``True`` when at least one pending L1 was resolved.
    """
    if escalation_queue is None:
        return False

    # Cheap gate FIRST — a healthy fleet calls this on every DONE, and touching
    # the filesystem there would be a per-dispatch pending-queue scan.  We only
    # look at disk when this process actually filed (memo hit), or when the
    # streak that just broke was long enough that a PRIOR process could have
    # filed for it before a restart.
    was_filed = filed_at is not None and task_id in filed_at
    if not was_filed and recovered_streak < threshold:
        return False

    if filed_at is not None:
        filed_at.pop(task_id, None)

    try:
        sentinel = f'{ZERO_PROGRESS_SENTINEL_PREFIX}{task_id}'
        pending = escalation_queue.get_by_task(sentinel, status='pending')
        resolved = 0
        for esc in pending:
            if getattr(esc, 'category', None) != _CATEGORY:
                continue
            escalation_queue.resolve(
                esc.id,
                (
                    f'Task {task_id} resumed making progress — the '
                    f'{recovered_streak}-requeue zero-agent-invocation streak '
                    'broke (the task reached a terminal outcome, or requeued '
                    'having actually invoked an agent).  Auto-resolved by the '
                    'zero-progress requeue detector; it will re-file if the '
                    'loop recurs.'
                ),
                resolved_by=_ROLE,
            )
            resolved += 1
    except Exception as exc:  # noqa: BLE001 — fail-open backstop
        logger.warning(
            'zero-progress-requeue alert for task %s could not be resolved '
            '(non-fatal): %s', task_id, exc,
        )
        return False

    if not resolved:
        return False

    logger.info(
        'Zero-progress requeue alert resolved for task %s after a %d-requeue '
        'streak broke', task_id, recovered_streak,
    )

    if event_store is not None:
        try:
            from orchestrator.event_store import EventType  # noqa: PLC0415

            event_store.emit(
                EventType.zero_progress_requeue_recovered,
                task_id=task_id,
                data={'streak': recovered_streak, 'resolved': resolved},
            )
        except Exception as exc:  # noqa: BLE001 — telemetry is best-effort
            logger.warning(
                'zero-progress-requeue recovery event for task %s could not be '
                'emitted (non-fatal, alert WAS resolved): %s', task_id, exc,
            )

    return True
