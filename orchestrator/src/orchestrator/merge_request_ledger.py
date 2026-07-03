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
    """

    request_id: str
    task_id: str
    branch: str
    age_secs: float
    phase: str = 'unowned'


@dataclass
class _LedgerEntry:
    """Internal bookkeeping record for one armed (dequeued, unresolved) request."""

    request_id: str
    task_id: str
    branch: str
    request: Any  # MergeRequest — Any to avoid a runtime import of merge_types
    dequeued_at: float


class RequestLedger:
    """Tracks dequeued-but-unresolved MergeRequests by ``request_id``.

    Pure in-memory bookkeeping; holds strong references to the tracked
    ``MergeRequest`` objects (needed to read ``request.result.done()``
    passively) for as long as they remain armed.  Process-lifetime only —
    not persisted, and intentionally so: a restart re-dequeues whatever is
    still on-disk/in-queue, which re-arms the ledger from scratch.
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
# Escalation (step-3/step-4): dedup'd L1 alarm for a stuck request.
# ---------------------------------------------------------------------------

# Prefix for the per-request stuck-liveness escalation sentinel.  Mirrors
# merge_liveness._VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX: a distinct dedup
# namespace per request_id, keyed off a value guaranteed unique per request
# (MergeRequest.request_id), so distinct stuck requests never collide and a
# single request re-alarms at most once per open-L1 episode.
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
    """Submit a dedup'd L1 escalation for a MergeRequest stuck past the liveness threshold.

    Fires at most ONCE per open-episode per request (dedup'd via
    ``has_open_l1(sentinel)``), modeled verbatim on
    :func:`orchestrator.merge_liveness._alarm_verify_host_unreachable`.

    The escalation is:

    * ``level=1`` (L1 blocking) — loud (steward -> auto-watcher -> human
      ladder) but NON-halting: this is observation-only per PRD design
      decision 4 — it never resolves the Future, mutates queue/inflight
      state, or halts the merge pipeline.
    * ``category='merge_request_stuck'``
    * ``task_id=_merge_request_stuck_sentinel(stuck.request_id)``

    None-safe: returns immediately when *escalation_queue* is None.
    Dedup: returns immediately when an open L1 already exists for the sentinel.

    Args:
        escalation_queue: Live escalation queue or ``None``.
        stuck: The :class:`StuckRequest` snapshot to alarm on.
        event_store: Optional event store; when provided an
            ``EventType.escalation_created`` event is emitted (mirrors the
            harness.py / workflow.py escalation_created emission
            convention — this module has no dedicated EventType of its
            own).
    """
    if escalation_queue is None:
        return

    sentinel = _merge_request_stuck_sentinel(stuck.request_id)

    if escalation_queue.has_open_l1(sentinel):
        return

    from escalation.models import Escalation  # local import — escalation optional dep

    age_secs_int = int(stuck.age_secs)
    age_str = (
        f'{stuck.age_secs / 3600:.1f} h' if stuck.age_secs >= 3600
        else f'{age_secs_int}s'
    )
    summary = (
        f'Merge request {stuck.request_id!r} (task {stuck.task_id}, branch '
        f'{stuck.branch!r}) has been dequeued for {age_str} '
        f'({age_secs_int}s) without its Future resolving (phase={stuck.phase!r}).'
    )
    detail = (
        f'request_id: {stuck.request_id}\n'
        f'task_id: {stuck.task_id}\n'
        f'branch: {stuck.branch}\n'
        f'age_secs: {age_secs_int}\n'
        f'phase: {stuck.phase}\n'
        '\n'
        'This request was dequeued by the merge worker but its outcome '
        'Future has not been resolved (set_result or cancelled) for longer '
        'than the stuck-request liveness threshold.  This is '
        'observation-only: the merge queue keeps running and this request '
        'has NOT been requeued, resolved, or halted automatically.\n'
        '\n'
        f"phase={stuck.phase!r} distinguishes a request still nominally "
        "owned by an in-flight verify slot (e.g. 'verifying', "
        "'gate_reverify', 'finalizing') from one that has fallen out of "
        "every pipeline structure ('unowned')."
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
            f'Inspect merge request {stuck.request_id} (task {stuck.task_id}, '
            f'branch {stuck.branch}).  Check worker logs for the dequeue and '
            'any verify/finalize activity for this request_id.  If the '
            'worker process is wedged, a restart requeues outstanding work; '
            'if the request has genuinely leaked out of every pipeline '
            'structure, manual intervention is needed to resolve or '
            'resubmit it.'
        ),
    )
    escalation_queue.submit(esc)

    if event_store is not None:
        from orchestrator.event_store import EventType
        event_store.emit(
            EventType.escalation_created,
            task_id=sentinel,
            data={
                'escalation_id': esc.id,
                'category': 'merge_request_stuck',
                'severity': 'blocking',
                'level': 1,
                'request_id': stuck.request_id,
                'branch': stuck.branch,
                'age_secs': stuck.age_secs,
                'phase': stuck.phase,
            },
        )
