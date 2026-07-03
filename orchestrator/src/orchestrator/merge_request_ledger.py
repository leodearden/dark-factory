"""Request-liveness ledger: detects dequeued MergeRequests whose Future is
never resolved (MQ-invariants eta / task 1992).

A dequeued :class:`~orchestrator.merge_types.MergeRequest` is expected to
eventually resolve its ``result`` Future — either via a normal outcome
(``set_result``) or via abandonment (``cancel``).  If neither ever happens
(a bug in some future code path, a wedged verify that never finalizes, or a
request that silently falls out of every pipeline structure) the caller
hangs forever with no signal.  :class:`RequestLedger` arms an entry for
every dequeued request and :class:`SpeculativeMergeWorker` (see
``merge_queue.py``'s ``_check_request_liveness``) periodically sweeps it,
firing a loud, dedup'd L1 escalation for any entry that has aged past the
stuck threshold without resolving.

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
* ``mark_warned`` records that the caller has already logged a WARNING for
  an entry's current episode; ``stuck_entries`` surfaces this via
  ``StuckRequest.already_warned`` so ``_check_request_liveness`` logs at
  most one WARNING per open episode (mirroring the escalation's
  ``has_open_l1`` dedup) instead of re-logging on every heartbeat poll. A
  fresh episode (``on_requeued`` followed by a re-``on_dequeue``) resets
  ``warned`` back to False.
* ``discard`` unconditionally drops an entry regardless of Future state — a
  manual/operator-driven "stop tracking this" for a request that will never
  resolve on its own and was never requeued (the genuine silent-hang case);
  see :class:`RequestLedger`'s docstring for the unbounded-lifetime note.

``merge_queue.py`` re-exports every public name here through its top-level
shim so existing importers (``from orchestrator.merge_queue import X``)
keep working unchanged — see that module's re-export shim block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestrator.merge_types import MergeRequest

# Shared with merge_queue.py and its sibling split modules (merge_liveness,
# merge_gates, merge_shadow, merge_drift, suffix_graph) so every merge-queue
# log line — regardless of which physical module emits it — appears under
# one logger namespace for caplog / journalctl filtering.
logger = logging.getLogger('orchestrator.merge_queue')


@dataclass(frozen=True)
class StuckRequest:
    """A read-only snapshot of a ledger entry that has aged past the stuck threshold.

    ``phase`` defaults to ``'unowned'`` — the caller (``_check_request_liveness``)
    annotates it from ``snapshot()['entries']`` when the request_id is still
    visible somewhere in the pipeline (e.g. ``'verifying'``); a request absent
    from every pipeline structure (leaked) keeps the ``'unowned'`` default.

    ``already_warned`` mirrors the ledger entry's ``warned`` flag — True once
    ``_check_request_liveness`` has already logged a WARNING for this
    request's CURRENT open episode, so the caller logs at most once per
    episode instead of re-logging on every heartbeat poll. The dedup'd L1
    escalation is unaffected by this flag — it continues to follow its own
    independent ``has_open_l1`` contract.
    """

    request_id: str
    task_id: str
    branch: str
    age_secs: float
    phase: str = 'unowned'
    already_warned: bool = False


@dataclass
class _LedgerEntry:
    """Internal bookkeeping record for one armed (dequeued, unresolved) request."""

    request_id: str
    task_id: str
    branch: str
    request: Any  # MergeRequest — Any to avoid a runtime import of merge_types
    dequeued_at: float
    # True once a WARNING has been logged for this episode (see mark_warned).
    # Reset to False only by creating a brand-new entry (on_dequeue when no
    # entry is already present) — i.e. after a prior entry was cleared by
    # on_requeued or swept as resolved.
    warned: bool = False


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
    process, continuing to surface via ``stuck_entries()`` /
    ``_check_request_liveness`` on every heartbeat poll (subject to the
    ``warned`` log-dedup contract) even after an operator has inspected and
    resolved the corresponding L1 escalation — ``has_open_l1`` flipping back
    to False (via ``resolve_issue``) makes the NEXT sweep resubmit. Call
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
            branch=request.branch,
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

    def mark_warned(self, request_id: str) -> None:
        """Record that a stuck WARNING has been logged for this entry's current episode.

        Consulted via ``StuckRequest.already_warned`` (populated by
        ``stuck_entries``) so callers (``_check_request_liveness``) log the
        stuck WARNING at most once per open episode instead of on every
        heartbeat poll — mirroring the escalation's ``has_open_l1`` dedup.
        A fresh episode begins the next time ``on_dequeue`` creates a
        brand-new entry (i.e. after the previous one was cleared by
        ``on_requeued`` or swept as resolved), which resets ``warned`` back
        to False. Idempotent and a silent no-op when *request_id* is absent
        (already resolved/requeued/never armed) — mirrors ``on_requeued``'s
        tolerance.
        """
        entry = self._entries.get(request_id)
        if entry is not None:
            entry.warned = True

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
                        already_warned=entry.warned,
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
# Dedup'd L1 escalation — fires when a request has aged past the stuck
# threshold without resolving.  See ``SpeculativeMergeWorker._check_request_liveness``
# in merge_queue.py, which calls this from the heartbeat loop.
# ---------------------------------------------------------------------------

_MERGE_REQUEST_STUCK_SENTINEL_PREFIX = '__merge_request_stuck__'


def _merge_request_stuck_sentinel(request_id: str) -> str:
    """Return the per-request dedup sentinel task_id for stuck-request alarms."""
    return f'{_MERGE_REQUEST_STUCK_SENTINEL_PREFIX}{request_id}'


def _alarm_merge_request_stuck(
    escalation_queue: Any,
    stuck: StuckRequest,
    *,
    event_store: Any = None,
) -> None:
    """Submit a dedup'd L1 escalation for a request stuck past the liveness threshold.

    Modeled verbatim on
    :func:`orchestrator.merge_liveness._alarm_verify_host_unreachable`. Fires at
    most ONCE per open episode per request_id (dedup'd via ``has_open_l1``).
    This is OBSERVATION + ESCALATION only — it never resolves the stuck
    request's Future, mutates queue/inflight state, or halts the pipeline
    (PRD design decision 4: invariants escalate loudly, degrade never).

    * ``level=1`` (L1 blocking) — loud (steward -> auto-watcher -> human
      ladder), but not itself a merge-halt trigger; the merge-halt gate keys
      off the ``wip_conflict`` / ``unmerged_state`` categories only.
    * ``category='merge_request_stuck'``
    * ``task_id=_merge_request_stuck_sentinel(stuck.request_id)``

    None-safe: returns immediately when *escalation_queue* is None.
    Dedup: returns immediately when an open L1 already exists for the sentinel.

    Args:
        escalation_queue: Live escalation queue or ``None``.
        stuck: The :class:`StuckRequest` snapshot to alarm on.
        event_store: Optional event store; when provided an
            ``EventType.escalation_created`` event is emitted.
    """
    if escalation_queue is None:
        return

    sentinel = _merge_request_stuck_sentinel(stuck.request_id)

    # Dedup: don't re-alarm while an open L1 already exists for this request.
    if escalation_queue.has_open_l1(sentinel):
        return

    from escalation.models import Escalation  # local import — escalation optional dep

    age_int = int(stuck.age_secs)
    summary = (
        f'MergeRequest {stuck.request_id!r} (branch {stuck.branch!r}) has been '
        f'dequeued for {age_int}s without resolving (phase={stuck.phase!r}).'
    )
    detail = (
        f'request_id: {stuck.request_id}\n'
        f'task_id: {stuck.task_id}\n'
        f'branch: {stuck.branch}\n'
        f'phase: {stuck.phase}\n'
        f'age_secs: {age_int}\n'
        '\n'
        'This request was dequeued by the merger loop but its result Future '
        'has neither resolved (set_result) nor been abandoned (cancel) within '
        'the stuck threshold. This is the silent-hang failure mode: some code '
        'path is holding the request without ever finalizing it. The '
        'orchestrator has NOT halted or mutated any pipeline state — this is '
        'observation-only.'
    )

    esc = Escalation(
        id=escalation_queue.make_id(sentinel),
        task_id=sentinel,
        agent_role='orchestrator-merge-liveness-monitor',
        severity='blocking',
        level=1,
        category='merge_request_stuck',
        summary=summary,
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
        from orchestrator.event_store import EventType

        event_store.emit(
            EventType.escalation_created,
            data={
                'request_id': stuck.request_id,
                'task_id': stuck.task_id,
                'branch': stuck.branch,
                'age_secs': stuck.age_secs,
                'phase': stuck.phase,
            },
        )
