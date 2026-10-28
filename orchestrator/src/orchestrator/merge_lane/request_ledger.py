"""Request-liveness ledger: detects dequeued MergeRequests whose Future is
never resolved (MQ-invariants eta / task 1992).

A dequeued :class:`~orchestrator.merge_lane.types.MergeRequest` is expected to
eventually resolve its ``result`` Future — either via a normal outcome
(``set_result``) or via abandonment (``cancel``).  If neither ever happens
(a bug in some future code path, a wedged verify that never finalizes, or a
request that silently falls out of every pipeline structure) the caller
hangs forever with no signal.  :class:`RequestLedger` arms an entry for
every dequeued request and :func:`report_stuck_requests`, called from the
merge worker's heartbeat, periodically sweeps it, raising a bounded ladder of
loud alarms (a WARNING plus an L1 escalation) for any entry that has aged past
the stuck threshold without resolving.

This module is entirely OBSERVATION + ESCALATION — it never resolves a
Future, mutates queue/inflight state, or halts the pipeline (PRD design
decision 4: invariants escalate loudly, degrade never).  Enforcement stays
with the existing halt machinery.

Lifecycle (see :class:`RequestLedger`):

* ``on_dequeue`` arms an entry when a request is dequeued by the merger
  loop.  Idempotent — a second dequeue of the same (already-armed)
  request_id keeps the EARLIEST ``dequeued_at``.
* ``on_requeued`` removes the entry when a request is deliberately put back
  on the queue with its Future still pending (operator halt / cascade
  re-queue).  Without this, a legitimately parked request would eventually
  false-alarm.  The next ``on_dequeue`` re-arms it with a fresh
  ``dequeued_at``, restarting the age clock.
* Resolution is detected PASSIVELY: an entry whose ``request.result.done()``
  is True (set via any of the ~20 ``set_result`` call sites, the zeta
  chokepoint, or Future cancellation on abandonment) is swept on the next
  ``stuck_entries`` call.  No per-site instrumentation is required.
* ``mark_alarmed`` counts the stuck alarms raised for an entry's current
  dequeue episode; ``stuck_entries`` surfaces the count as
  ``StuckRequest.alarms_raised`` so ``report_stuck_requests`` walks the
  :data:`STUCK_ALARM_THRESHOLD_MULTIPLES` ladder once per episode instead of
  re-alarming on every heartbeat poll. A fresh episode (``on_requeued``
  followed by a re-``on_dequeue``) starts the count at zero.
* ``discard`` unconditionally drops an entry regardless of Future state — a
  manual/operator-driven "stop tracking this" for a request that will never
  resolve on its own and was never requeued (the genuine silent-hang case);
  see :class:`RequestLedger`'s docstring for the unbounded-lifetime note.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from escalation.models import Escalation

from orchestrator.event_store import EventType

if TYPE_CHECKING:
    from orchestrator.merge_lane.types import MergeRequest

# Shared with merge_queue.py and its sibling split modules (merge_liveness,
# merge_gates, merge_shadow, merge_drift, suffix_graph) so every merge-queue
# log line — regardless of which physical module emits it — appears under
# one logger namespace for caplog / journalctl filtering.
logger = logging.getLogger('orchestrator.merge_lane')

# One alarm when a dequeue first passes the stuck threshold, ONE final re-alarm past 3x
# (13.5h at the 16200s default), then silence until the request resolves or is re-dequeued.
# A sweep raises at most one alarm per episode, so an episode first swept already past 3x
# (a heartbeat stalled for hours) raises both alarms on consecutive heartbeats.
STUCK_ALARM_THRESHOLD_MULTIPLES: tuple[float, ...] = (1.0, 3.0)


@dataclass(frozen=True)
class StuckRequest:
    """A read-only snapshot of a ledger entry that has aged past the stuck threshold.

    ``phase`` defaults to ``'unowned'`` — :func:`report_stuck_requests`
    annotates it from its pipeline-phase lookup when the request_id is still
    visible somewhere in the pipeline (e.g. ``'verifying'``); a request absent
    from every pipeline structure (leaked) keeps the ``'unowned'`` default.

    ``alarms_raised`` counts the alarms already raised for this request's
    current dequeue episode.
    """

    request_id: str
    task_id: str
    branch: str
    age_secs: float
    phase: str = 'unowned'
    alarms_raised: int = 0


@dataclass
class _LedgerEntry:
    """Internal bookkeeping record for one armed (dequeued, unresolved) request."""

    request_id: str
    task_id: str
    branch: str
    request: Any  # MergeRequest — Any to avoid a runtime import of merge_types
    dequeued_at: float
    # Alarms raised in this dequeue episode (see mark_alarmed). The entry IS
    # the episode, so only a brand-new entry starts back at zero.
    alarms_raised: int = 0


class RequestLedger:
    """Tracks dequeued-but-unresolved MergeRequests by ``request_id``.

    Pure in-memory bookkeeping; holds strong references to the tracked
    ``MergeRequest`` objects (needed to read ``request.result.done()``
    passively) for as long as they remain armed.  Process-lifetime only —
    not persisted, and intentionally so: a restart re-dequeues whatever is
    still on-disk/in-queue, which re-arms the ledger from scratch.

    Unbounded-lifetime note: a request whose Future is never resolved
    (``set_result``/``cancel``) AND never explicitly requeued (``on_requeued``)
    — the genuine silent-hang case this ledger exists to catch — is NOT
    automatically evicted. ``sweep_resolved`` only drops ``done()`` entries;
    nothing else removes an armed entry on a timer. Such an entry (and its
    strong reference to the ``MergeRequest``) persists for the life of the
    process and keeps surfacing via ``stuck_entries()``, but raises at most
    ``len(STUCK_ALARM_THRESHOLD_MULTIPLES)`` alarms per dequeue episode. Call
    ``discard(request_id)`` explicitly once an operator has finished
    investigating to stop tracking it (this does not touch escalation-queue
    state).
    """

    def __init__(self) -> None:
        self._entries: dict[str, _LedgerEntry] = {}

    def on_dequeue(self, request: MergeRequest, *, now: float) -> None:
        """Arm the ledger for a freshly-dequeued request.

        Idempotent: if an entry for ``request.request_id`` already exists,
        it is left untouched (earliest ``dequeued_at`` wins) rather than
        duplicated or reset.  A request with no ``request_id`` (should not
        happen in practice — ``MergeRequest.request_id`` always has a
        default factory) is silently skipped rather than raising.
        """
        request_id = getattr(request, 'request_id', None)
        if not request_id:
            return
        if request_id in self._entries:
            return
        self._entries[request_id] = _LedgerEntry(
            request_id=request_id,
            task_id=request.task_id,
            branch=request.branch.bare_id,
            request=request,
            dequeued_at=now,
        )

    def on_requeued(self, request_id: str) -> None:
        """Remove the entry for a request that was deliberately requeued.

        Idempotent: removing an absent (or already-removed) request_id is a
        silent no-op, never a ``KeyError``.  The request's Future is left
        pending by the requeue caller; the NEXT ``on_dequeue`` re-arms it
        with a fresh ``dequeued_at``, restarting the age clock.
        """
        self._entries.pop(request_id, None)

    def discard(self, request_id: str) -> None:
        """Unconditionally drop the entry for *request_id*, regardless of Future state.

        For operator/programmatic eviction of a request that will never
        resolve on its own (the genuine silent-hang case) once its stuck
        escalation has been inspected — see :class:`RequestLedger`'s
        docstring for the unbounded-lifetime note this addresses. Unlike
        ``on_requeued`` (which implies the request will be re-armed by a
        subsequent ``on_dequeue``), ``discard`` is a terminal "stop tracking
        this": the ledger will not report on this request_id again unless a
        future ``on_dequeue`` re-arms it from scratch. Idempotent:
        discarding an absent request_id is a silent no-op, never a
        ``KeyError``. Does not touch escalation-queue state (e.g.
        ``has_open_l1``) — that is resolved independently via
        ``resolve_issue``.
        """
        self._entries.pop(request_id, None)

    def mark_alarmed(self, request_id: str) -> None:
        """Count one more stuck alarm raised for this entry's current dequeue episode.

        A silent no-op when *request_id* is absent (already resolved,
        requeued or never armed), like ``on_requeued``.
        """
        entry = self._entries.get(request_id)
        if entry is not None:
            entry.alarms_raised += 1

    def sweep_resolved(self) -> None:
        """Drop every entry whose request's Future is already ``done()``.

        Covers both a normal outcome (``set_result`` at any of the resolve
        sites, including the zeta chokepoint) and abandonment
        (``cancel()``) — both make ``Future.done()`` True.
        """
        resolved_ids = [
            request_id
            for request_id, entry in self._entries.items()
            if entry.request.result.done()
        ]
        for request_id in resolved_ids:
            del self._entries[request_id]

    def stuck_entries(self, now: float, threshold_s: float) -> list[StuckRequest]:
        """Sweep resolved entries, then return those older than *threshold_s*.

        ``age_secs`` on each returned :class:`StuckRequest` is ``now -
        dequeued_at``.  Sweeping first ensures a request that resolved
        between the previous check and this one is never misreported as
        stuck.
        """
        self.sweep_resolved()
        stuck: list[StuckRequest] = []
        for entry in self._entries.values():
            age_secs = now - entry.dequeued_at
            if age_secs > threshold_s:
                stuck.append(
                    StuckRequest(
                        request_id=entry.request_id,
                        task_id=entry.task_id,
                        branch=entry.branch,
                        age_secs=age_secs,
                        alarms_raised=entry.alarms_raised,
                    )
                )
        return stuck

    def open_request_ids(self) -> set[str]:
        """Return the set of currently-armed request_ids."""
        return set(self._entries)

    def is_empty(self) -> bool:
        """True when no request is currently armed."""
        return not self._entries

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Stuck-request alarms — report_stuck_requests walks each dequeue episode's
# alarm ladder from the merge worker's heartbeat and files each alarm through
# _alarm_merge_request_stuck.
# ---------------------------------------------------------------------------


def _alarm_due(stuck: StuckRequest, threshold_s: float) -> bool:
    """True when *stuck*'s episode has passed the ladder tier of its next alarm."""
    raised = stuck.alarms_raised
    return (
        raised < len(STUCK_ALARM_THRESHOLD_MULTIPLES)
        and stuck.age_secs > STUCK_ALARM_THRESHOLD_MULTIPLES[raised] * threshold_s
    )


def report_stuck_requests(
    ledger: RequestLedger,
    *,
    now: float,
    threshold_s: float,
    pipeline_phases: Callable[[], Mapping[str, str]],
    escalation_queue: Any,
    event_store: Any = None,
) -> None:
    """Raise every due stuck alarm: a WARNING plus an L1 filing per alarm.

    Alarm k of a dequeue episode is due once the request's age passes
    ``STUCK_ALARM_THRESHOLD_MULTIPLES[k] * threshold_s``, so an episode raises
    at most ``len(STUCK_ALARM_THRESHOLD_MULTIPLES)`` alarms however long it
    stays stuck and however its escalations are closed. A new dequeue of the
    request is a new episode with a fresh ladder.

    *pipeline_phases* maps request_id to its pipeline state; it is called at
    most once per sweep, and only when an alarm is due.

    An alarm counts as raised once its WARNING is logged, whether or not its
    L1 filing succeeds. A filing that raises is logged with its traceback, and
    the sweep goes on to the next due entry. The failed filing is not retried:
    the episode's next tier, if any, is its next chance. Retrying on every
    heartbeat would repeat the WARNING on every heartbeat, which is the
    repetition this ladder bounds.

    OBSERVATION + ESCALATION only — never resolves a Future, mutates
    pipeline state, or halts the pipeline (PRD design decision 4).
    """
    due = [
        stuck for stuck in ledger.stuck_entries(now, threshold_s)
        if _alarm_due(stuck, threshold_s)
    ]
    if not due:
        return
    phases = pipeline_phases()
    for stuck in due:
        _raise_stuck_alarm(
            ledger,
            replace(stuck, phase=phases.get(stuck.request_id, 'unowned')),
            escalation_queue=escalation_queue,
            event_store=event_store,
        )


def _raise_stuck_alarm(
    ledger: RequestLedger,
    stuck: StuckRequest,
    *,
    escalation_queue: Any,
    event_store: Any,
) -> None:
    """WARN, count, then file one alarm; a failed filing is logged, never raised."""
    alarm_number = _ladder_alarm_number(stuck)
    logger.warning(
        'merge request stuck: request_id=%s task=%s branch=%s age=%.0fs '
        'phase=%s alarm %d/%d — dequeued but never resolved (silent-hang failure mode)',
        stuck.request_id, stuck.task_id, stuck.branch, stuck.age_secs, stuck.phase,
        alarm_number, len(STUCK_ALARM_THRESHOLD_MULTIPLES),
    )
    ledger.mark_alarmed(stuck.request_id)
    try:
        _alarm_merge_request_stuck(escalation_queue, stuck, event_store=event_store)
    except Exception:
        logger.exception(
            'merge request stuck: filing alarm %d for request_id=%s failed; it still '
            'counts, so this dequeue files again only at its next ladder tier',
            alarm_number, stuck.request_id,
        )


def _ladder_alarm_number(stuck: StuckRequest) -> int:
    """Number *stuck*'s next alarm, refusing any number that is not a tier of the ladder."""
    alarm_number = stuck.alarms_raised + 1
    ladder_size = len(STUCK_ALARM_THRESHOLD_MULTIPLES)
    if not 1 <= alarm_number <= ladder_size:
        raise ValueError(
            f'stuck alarm {alarm_number} for request {stuck.request_id!r} is outside the '
            f'{ladder_size}-alarm ladder STUCK_ALARM_THRESHOLD_MULTIPLES '
            f'(alarms_raised={stuck.alarms_raised})'
        )
    return alarm_number


_MERGE_REQUEST_STUCK_SENTINEL_PREFIX = '__merge_request_stuck__'


def _merge_request_stuck_sentinel(request_id: str) -> str:
    """Return the per-request dedup sentinel task_id for stuck-request alarms."""
    return f'{_MERGE_REQUEST_STUCK_SENTINEL_PREFIX}{request_id}'


def _stuck_alarm_summary(stuck: StuckRequest, alarm_number: int) -> str:
    age_int = int(stuck.age_secs)
    if alarm_number == 1:
        return (
            f'MergeRequest {stuck.request_id!r} (branch {stuck.branch!r}) has been '
            f'dequeued for {age_int}s without resolving (phase={stuck.phase!r}).'
        )
    multiple = STUCK_ALARM_THRESHOLD_MULTIPLES[alarm_number - 1]
    ladder_size = len(STUCK_ALARM_THRESHOLD_MULTIPLES)
    which = (
        'final stuck alarm' if alarm_number == ladder_size
        else f'stuck alarm {alarm_number} of {ladder_size}'
    )
    return (
        f'MergeRequest {stuck.request_id!r} (branch {stuck.branch!r}) is still '
        f'unresolved {age_int}s after dequeue, past {multiple:g}x the stuck threshold, '
        f'after an earlier stuck alarm (phase={stuck.phase!r}); {which} for this dequeue.'
    )


def _stuck_alarm_close_note(alarm_number: int) -> str:
    if alarm_number < len(STUCK_ALARM_THRESHOLD_MULTIPLES):
        next_multiple = STUCK_ALARM_THRESHOLD_MULTIPLES[alarm_number]
        return (
            'Closing this escalation suppresses further stuck alarms for this '
            'dequeue; the next follows only if the request is still unresolved '
            f'past {next_multiple:g}x the stuck threshold.'
        )
    return (
        'This is the final stuck alarm for this dequeue: none follow until the '
        'request resolves or is re-dequeued.'
    )


def _alarm_merge_request_stuck(
    escalation_queue: Any,
    stuck: StuckRequest,
    *,
    event_store: Any = None,
) -> None:
    """Submit an L1 escalation for one alarm of a request stuck past the liveness threshold.

    Modeled on
    :func:`orchestrator.merge_lane.liveness._alarm_verify_host_unreachable`.
    Per-dequeue-episode dedup is the alarm ladder, enforced by
    :func:`report_stuck_requests`; this function files the alarm numbered
    ``stuck.alarms_raised + 1``. ``has_open_l1`` remains only as the guard for
    an L1 left open across a process restart, whose ledger starts empty.
    This is OBSERVATION + ESCALATION only — it never resolves the stuck
    request's Future, mutates queue/inflight state, or halts the pipeline
    (PRD design decision 4: invariants escalate loudly, degrade never).

    * ``level=1`` (L1 blocking) — loud (steward -> auto-watcher -> human
      ladder), but not itself a merge-halt trigger; the merge-halt gate keys
      off the ``wip_conflict`` / ``unmerged_state`` / ``stash_failed``
      categories only.
    * ``category='merge_request_stuck'``
    * ``task_id=_merge_request_stuck_sentinel(stuck.request_id)``

    None-safe: returns immediately when *escalation_queue* is None.
    Returns immediately when an open L1 already exists for the sentinel.

    Args:
        escalation_queue: Live escalation queue or ``None``.
        stuck: The :class:`StuckRequest` snapshot to alarm on.
        event_store: Optional event store; when provided an
            ``EventType.escalation_created`` event is emitted.

    Raises:
        ValueError: ``stuck.alarms_raised + 1`` is not an alarm of the
            :data:`STUCK_ALARM_THRESHOLD_MULTIPLES` ladder, even when
            *escalation_queue* is None.
    """
    alarm_number = _ladder_alarm_number(stuck)
    if escalation_queue is None:
        return

    sentinel = _merge_request_stuck_sentinel(stuck.request_id)

    # Dedup: don't re-alarm while an open L1 already exists for this request.
    if escalation_queue.has_open_l1(sentinel):
        return

    age_int = int(stuck.age_secs)
    detail = (
        f'request_id: {stuck.request_id}\n'
        f'task_id: {stuck.task_id}\n'
        f'branch: {stuck.branch}\n'
        f'phase: {stuck.phase}\n'
        f'age_secs: {age_int}\n'
        f'alarm: {alarm_number} of {len(STUCK_ALARM_THRESHOLD_MULTIPLES)} for this dequeue\n'
        '\n'
        'This request was dequeued by the merger loop but its result Future '
        'has neither resolved (set_result) nor been abandoned (cancel) within '
        'the stuck threshold. This is the silent-hang failure mode: some code '
        'path is holding the request without ever finalizing it. The '
        'orchestrator has NOT halted or mutated any pipeline state — this is '
        'observation-only. '
        f'{_stuck_alarm_close_note(alarm_number)}'
    )

    esc = Escalation(
        id=escalation_queue.make_id(sentinel),
        task_id=sentinel,
        agent_role='orchestrator-merge-liveness-monitor',
        severity='blocking',
        level=1,
        category='merge_request_stuck',
        summary=_stuck_alarm_summary(stuck, alarm_number),
        detail=detail,
        suggested_action=(
            f'Inspect the merge worker for request {stuck.request_id!r} '
            f'(branch {stuck.branch!r}, phase {stuck.phase!r}). Determine '
            'whether the request is wedged in-flight (e.g. a hung verify) or '
            'has leaked out of every pipeline structure, then resolve or '
            'abandon it manually.'
        ),
    )
    escalation_queue.submit(esc)

    if event_store is not None:
        event_store.emit(
            EventType.escalation_created,
            data={
                'request_id': stuck.request_id,
                'task_id': stuck.task_id,
                'branch': stuck.branch,
                'age_secs': stuck.age_secs,
                'phase': stuck.phase,
                'alarm_number': alarm_number,
            },
        )
