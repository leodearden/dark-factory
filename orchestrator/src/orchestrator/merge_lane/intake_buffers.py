"""The merger's lane buffers, its redispatch park and the suffix-conflict state.

``SpeculativeMergeWorker`` composes one :class:`MergeIntake`, the sole writer of
all three, and hands it the ports it reads. Requests reach the queue it drains
through :mod:`orchestrator.merge_lane.intake`.
"""
from __future__ import annotations

import asyncio
import collections
import dataclasses
import logging
from collections.abc import Callable, Collection, Iterable
from typing import Any, Literal, Protocol

from escalation.models import Escalation

from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.halt import HaltState
from orchestrator.merge_lane.intake import emit_merge_coalesced, normalize_lane
from orchestrator.merge_lane.types import (
    GroupMergeRequest,
    ItemLifecycleState,
    MergeBounceRegistry,
    MergeOutcome,
    MergeRequest,
    SpeculativeItem,
)
from orchestrator.merge_pool_facts import PickRecords
from orchestrator.suffix_graph import SuffixConflictGraph, SuffixConflictTracker
from orchestrator.task_priority import TaskPriorityLookup

logger = logging.getLogger('orchestrator.merge_lane')

MERGE_WORKER_SHUTDOWN_REASON = 'Merge worker shutting down'
"""Reason string set on MergeOutcome('blocked') by stop() and the _merger_loop
finally block.  Used by MergeIntake.buffer_owned_request._on_terminal to
distinguish a graceful-stop / crash path (keep in journal for recovery) from a
deterministic terminal failure (remove — not retried on restart)."""

_COALESCE_LIVE_ORIGINAL_SENTINEL_PREFIX = '__merge_coalesce_live_original__'


class LifecycleHop(Protocol):
    """A forward registry hop for *request_id*, re-indexing *live_obj* when given."""

    def __call__(
        self,
        request_id: str,
        from_state: ItemLifecycleState,
        to_state: ItemLifecycleState,
        /,
        *,
        live_obj: MergeRequest | None = None,
    ) -> object: ...


@dataclasses.dataclass(frozen=True)
class IntakeLifecycle:
    """The five lifecycle-registry operations the intake drives, as the worker's
    chokepoint helpers (``_register_item``, ``_advance_if_at``,
    ``_note_transition``) implement them. The worker passes callables that
    re-read those helpers on every call, so a helper replaced on the instance
    still sees the intake's hops."""

    current: Callable[[str], ItemLifecycleState | None]
    register: Callable[[MergeRequest, ItemLifecycleState], object]
    advance_if_at: LifecycleHop
    note_transition: LifecycleHop
    live_request: Callable[[str], MergeRequest | None]
    """The request behind *request_id*'s live registry object, whatever its
    pipeline shape, or None when nothing is live under that id."""


def _alarm_coalesce_live_original(
    escalation_queue: Any,
    request: MergeRequest,
    current_state: ItemLifecycleState,
    *,
    event_store: Any = None,
) -> None:
    """Submit a dedup'd L1 escalation for a re-entrant drain of the LIVE
    ORIGINAL request object (task 3082).

    Modeled verbatim on the worker's ``_alarm_illegal_lifecycle_transition``
    dedup'd ``has_open_l1``/``make_id`` idiom. Fires at most ONCE per open
    episode per request_id. OBSERVATION + ESCALATION only: never mutates
    queue/inflight state (PRD design decision 4: invariants escalate loudly,
    degrade never).

    Distinct from the benign task-2852 journal-recovery twin, which is a
    DIFFERENT object carrying its own fresh, unobserved future. When the
    arriving item IS the live original, some requeue site put it back on
    ``_queue`` without returning the registry to QUEUED, so the drain
    chokepoint mistook a genuinely live request for a duplicate — a
    requeue-wiring bug.

    * ``level=1`` (L1 blocking).
    * ``category='merge_coalesce_live_original'``

    None-safe: returns immediately when *escalation_queue* is None.
    Dedup: returns immediately when an open L1 already exists for the sentinel.
    """
    if escalation_queue is None:
        return

    sentinel = f'{_COALESCE_LIVE_ORIGINAL_SENTINEL_PREFIX}{request.request_id}'
    if escalation_queue.has_open_l1(sentinel):
        return

    summary = (
        f'Re-entrant merge drain of the LIVE ORIGINAL request '
        f'{request.request_id!r} (branch={request.branch.bare_id}, registry at '
        f'{current_state!r}) — a requeue-wiring bug, not a benign duplicate twin'
    )
    detail = (
        f'request_id: {request.request_id}\n'
        f'task_id: {request.task_id}\n'
        f'branch: {request.branch.bare_id}\n'
        f'registry current state: {current_state!r}\n'
        '\n'
        'The merge drain chokepoint received the SAME MergeRequest object that '
        'is already live in the lifecycle registry (matched by result-future '
        'identity), not a distinct journal-recovery twin. That means a requeue '
        'site put this request back on _queue WITHOUT returning the registry to '
        'QUEUED, so MergeIntake.buffer_owned_request classified a genuinely '
        'live request as a duplicate and coalesced it away.\n'
        '\n'
        'The request was NOT buffered and its Future was deliberately left '
        'PENDING — resolving it would hand the real waiter a fabricated '
        "'already_merged' AND silence the merge_request_stuck watchdog (the "
        'request ledger drops any resolved entry before the stuck sweep sees '
        'it). Expect a merge_request_stuck alarm for this request_id as the '
        'slower backstop if it is never re-dispatched.\n'
        '\n'
        'This condition is unreachable in normal operation: every requeue site '
        'pairs on_requeued with _note_requeue, so it fires only if a requeue '
        'site has regressed. Start there: find the on_requeued call that put '
        'this request_id back on _queue and check it also called _note_requeue.'
    )

    esc = Escalation(
        id=escalation_queue.make_id(sentinel),
        task_id=sentinel,
        agent_role='orchestrator-merge-lifecycle-monitor',
        severity='blocking',
        level=1,
        category='merge_coalesce_live_original',
        summary=summary,
        detail=detail,
        suggested_action=(
            f'Find the requeue site that re-queued request_id '
            f'{request.request_id!r} without calling _note_requeue — check the '
            'abort/defer branches of _run_inflight_verify, _dispatch_item, and '
            'the head-failure cascade. The registry read '
            f'{current_state!r} instead of QUEUED at the drain.'
        ),
    )
    escalation_queue.submit(esc)

    if event_store is not None:
        event_store.emit(
            EventType.escalation_created,
            task_id=request.task_id,
            data={
                'request_id': request.request_id,
                'current_state': str(current_state),
                'category': 'merge_coalesce_live_original',
            },
        )


def _aging_key(req: MergeRequest) -> tuple[float, str]:
    """Aging sort key for a merge request (ζ=1891).

    Returns a ``(wall_clock, request_id)`` tuple:

    * **wall_clock** — ``merge_first_enqueued_at`` (persisted epoch of the
      *first* merge submission, survives restart) with ``enqueued_at`` as
      the legacy fallback for requests created before α=1886 was deployed
      (field is ``None``).  Smaller = older = higher priority.

    * **request_id** — lexical tie-break for equal timestamps (PRD §5.4 /
      §11.3).  Lexically smaller wins.

    Used by :meth:`MergeIntake.pop_next_pickable` to identify the
    clique-minimal item within a footprint clique — the buffered item whose
    footprint-clique peers all have an equal-or-larger aging key.
    """
    return (req.first_enqueued_at, req.request_id)


def answers_to(req: MergeRequest, request_id: str | None, task_id: str | None) -> bool:
    """Whether *req* is the queue entry an operator named: by its own request id,
    or by a task id it carries — its own, or a coalesced member's (task 5246)."""
    if request_id is not None:
        return req.request_id == request_id
    return req.task_id == task_id or (
        isinstance(req, GroupMergeRequest) and task_id in req.member_task_ids
    )


class MergeIntake:
    """The merger's per-lane buffers, the front-priority redispatch park and the
    two-layer suffix-conflict tracker over the buffered suffix.

    Sole writer of all three: every other worker concern reaches them through
    the request and read methods below. The verifier's chain-link walk and
    ``stop()``'s drains are legitimate non-owner writers, and say so where
    they write.
    """

    def __init__(
        self,
        *,
        queue: asyncio.Queue[MergeRequest],
        lanes: tuple[str, ...],
        lifecycle: IntakeLifecycle,
        halt: HaltState,
        admission_halted: Callable[[], bool],
        merge_store: Any,
        event_store: EventStore | None,
        escalation_queue: Any,
        git_ops: Callable[[], GitOps],
        frozen_requests: Callable[[], tuple[MergeRequest, ...]],
        frozen_prefix_tip: Callable[[str], str],
        retire_item: Callable[[str], None],
        assert_merger_owns: Callable[[str], None],
        is_reserved: Callable[[str], bool],
        task_priority: TaskPriorityLookup | None = None,
    ) -> None:
        self._queue = queue
        self._is_reserved = is_reserved
        self._lanes = lanes
        self._lifecycle = lifecycle
        self._halt = halt
        self._admission_halted = admission_halted
        # Durable journal (task 1772): records worker-owned requests on accept,
        # removes them on terminal.  None-safe so bare-worker tests (no store)
        # are unaffected.
        self._merge_store: Any = merge_store
        self._event_store = event_store
        self._escalation_queue: Any = escalation_queue
        self._assert_merger_owns = assert_merger_owns
        # Set of request_ids for which a done-callback has already been registered
        # (avoids duplicate removal callbacks on redispatch / chain of the same id).
        self._journaled_request_ids: set[str] = set()
        # Per-lane FIFO buffers — items are drained from _queue into these so
        # pick-order can prefer high over normal.  Each deque preserves FIFO
        # within a lane.  Accessed only from the merger coroutine.
        self._lane_buffers: dict[str, collections.deque[MergeRequest]] = {
            ln: collections.deque() for ln in lanes
        }
        # γ front-priority re-dispatch deque: re-merged/re-dispatched items (chain-
        # invalidation + RunnerUnavailable) go here and are drained before the
        # verifier queue so they are re-verified in submission order ahead of
        # newer arrivals.
        self._redispatch: collections.deque[SpeculativeItem] = collections.deque()
        # δ=1988 — the two-layer suffix-conflict graph + bounce state (conflict
        # graph over the unfrozen suffix, debounce signature, cached main SHA,
        # per-branch bounce counter). git_ops/frozen_requests/frozen_prefix_tip/
        # retire_item are narrow callables that RE-READ the worker's live attribute on every
        # call, so the tracker stays correct if the worker's `_git_ops` is ever
        # reassigned after construction. See suffix_graph.py's module docstring
        # + SuffixConflictTracker docstring for the full
        # debounce/fail-open/TOCTOU/cap contract.
        self._suffix_tracker = SuffixConflictTracker(
            git_ops=git_ops,
            lane_buffers=lambda: self._lane_buffers,
            frozen_requests=frozen_requests,
            frozen_prefix_tip=frozen_prefix_tip,
            retire_item=retire_item,
            is_reserved=is_reserved,
        )
        self._pick_records = PickRecords(
            lanes=lanes, is_reserved=is_reserved, is_lane_halted=halt.is_lane_halted,
            task_priority=task_priority,
        )

    # ── reads ──────────────────────────────────────────────────────────────

    def buffered(self, lane: str) -> tuple[MergeRequest, ...]:
        """The requests waiting in *lane*, FIFO."""
        return tuple(self._lane_buffers[lane])

    def queued_arrivals(self) -> list[MergeRequest]:
        """Requests still on the external queue, not yet drained into a lane buffer.

        Peeks CPython's ``asyncio.Queue`` item deque, read-only: the queue has no
        public peek, and a side-list kept at every put/get site would be a
        second copy to keep in step.
        """
        pending: collections.deque[MergeRequest | None] = vars(self._queue)['_queue']
        return [req for req in pending if req is not None]

    def pending_redispatch(self) -> tuple[SpeculativeItem, ...]:
        """The parked redispatch items, in the order they will be dispatched."""
        return tuple(self._redispatch)

    def take_pick_record(self, request_id: str) -> dict[str, object]:
        """What the buffers held when *request_id* was last picked, once; empty if unrecorded."""
        return self._pick_records.take(request_id)

    @property
    def has_pending_redispatch(self) -> bool:
        return bool(self._redispatch)

    @property
    def suffix_conflict_graph(self) -> SuffixConflictGraph:
        return self._suffix_tracker.graph

    @property
    def suffix_conflict_signature(self) -> tuple[tuple[str, ...], str] | None:
        return self._suffix_tracker.signature

    @property
    def last_known_main_sha(self) -> str | None:
        return self._suffix_tracker.last_known_main_sha

    @property
    def bounce_registry(self) -> MergeBounceRegistry:
        return self._suffix_tracker.bounce_registry

    # ── the drain and the pick ─────────────────────────────────────────────

    def buffer_owned_request(self, item: MergeRequest) -> None:
        """Append *item* to its lane buffer and record it in the durable journal.

        Registers a once-only done-callback on ``item.result`` so the journal
        entry is removed on ANY terminal outcome (done/error/abandoned/
        superseded).  The ``_journaled_request_ids`` set prevents duplicate
        callbacks when the same request_id is re-dispatched (e.g. CAS retry).

        Fail-open: store errors are logged and never propagate so a broken
        journal never stalls the merge pipeline.

        MQ-reliability kappa (task 2169): this is the worker's first sighting
        of *item* for a FRESH request — its three callers
        (:meth:`drain_into_lanes`, the worker's ``_acquire_next_request``
        harvest branch, and the merger loop's look-ahead harvest) all pull an
        item straight off the external ``_queue``/``_pending_get``. That item
        is USUALLY a brand-new request_id, but it may also be a request
        previously put back on ``_queue`` by a requeue site (the worker's
        ``_note_requeue`` — operator-halt / cascade self-requeue), which
        leaves it registered at QUEUED rather than removing it
        (``ItemLifecycle`` has no deregister op).

        task 2852 (SHARPER RCA mr-8cafbaf0 / original incident mr-c69ca480):
        an already-tracked rid does NOT unconditionally skip straight to the
        QUEUED -> LANE_BUFFERED transition — that assumption is exactly what
        the incident disproved (a duplicate/re-entrant *item*, e.g. from the
        journal-recovery ``reconstruct_merge_request`` path, can arrive for a
        request_id that has already advanced well past QUEUED, such as
        VERIFYING). The registry's CURRENT state discriminates three cases:

          * unregistered (``None``)   — register fresh at QUEUED, then fall
            through.
          * ``QUEUED`` (fresh-just-registered, OR a legit requeue left here
            by ``_note_requeue``) — fall through as-is; this is the ONLY
            case where re-``register()``-ing would raise ``ValueError`` on
            the live duplicate (``ItemLifecycle.register``'s
            precondition), which is exactly why registration is skipped here.
          * anything else (already advanced past QUEUED) — a duplicate/
            re-entrant *item* for a request_id already deep in the pipeline.
            Coalesced via :meth:`_coalesce_reentrant_drain` and dropped
            BEFORE the transition, the lane-buffer append, and the journal
            block below — never re-``register()``-ed, never buffered, never
            re-recorded (which would otherwise clobber the live original's
            durable journal entry under the shared request_id).

        The fall-through path routes the QUEUED -> LANE_BUFFERED hop through
        :attr:`IntakeLifecycle.advance_if_at` (current is always QUEUED at
        that point, so it always fires), matching this method's own
        postcondition (the item is now sitting in a lane buffer).
        """
        lane = normalize_lane(item.lane)
        self._assert_merger_owns('_lane_buffers')
        rid = item.request_id
        current = self._lifecycle.current(rid)
        if current is not None and current != ItemLifecycleState.QUEUED:
            self._coalesce_reentrant_drain(item, current)
            return
        if current is None:
            self._lifecycle.register(item, ItemLifecycleState.QUEUED)
        self._lifecycle.advance_if_at(
            rid, ItemLifecycleState.QUEUED, ItemLifecycleState.LANE_BUFFERED, live_obj=item,
        )
        self._lane_buffers[lane].append(item)

        if self._merge_store is not None:
            try:
                self._merge_store.record(item)
            except Exception:  # noqa: BLE001
                logger.warning(
                    'merge_queue: _buffer_owned_request: store.record failed for %s',
                    item.request_id,
                    exc_info=True,
                )

            if item.request_id not in self._journaled_request_ids:
                self._journaled_request_ids.add(item.request_id)

                def _on_terminal(fut: asyncio.Future, *, _rid: str = item.request_id) -> None:  # type: ignore[type-arg]
                    # Decide whether to remove from the durable journal.
                    #
                    # KEEP in journal (return early) when the outcome is from the
                    # graceful-stop / crash path — branch is still live, recovery
                    # should retry:
                    #   • fut.cancelled() — explicitly cancelled future; keep.
                    #   • result.status == 'blocked' AND reason == SHUTDOWN_REASON
                    #     — stop() and the _merger_loop finally block both set this
                    #     exact reason; a true process crash leaves the future
                    #     unresolved so the callback never fires regardless.
                    #
                    # REMOVE from journal for ALL other terminals, including other
                    # 'blocked' outcomes (drop-guard, ENOSPC, merge conflict, merger
                    # error, verification error, chain-generation bound) — those
                    # branches are still live but the failure is deterministic, so
                    # keep-for-recovery would cause indefinite cross-restart retries
                    # and unbounded merge_queue.json growth.
                    #
                    # Prune _journaled_request_ids on removal so the set does not
                    # grow without bound and a later re-dispatch with a fresh Future
                    # can register a new cleanup callback.
                    try:
                        if self._merge_store is None:
                            return
                        if fut.cancelled():
                            return  # explicitly cancelled: keep in journal
                        try:
                            result = fut.result()
                            if (
                                result.status == 'blocked'
                                and result.reason == MERGE_WORKER_SHUTDOWN_REASON
                            ):
                                return  # crash / graceful-stop path: keep for recovery
                        except Exception:  # noqa: BLE001
                            pass  # exception-set future: fall through to remove
                        self._merge_store.remove(_rid)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            'merge_queue: _on_terminal: store.remove failed for %s',
                            _rid,
                            exc_info=True,
                        )
                    # Prune regardless of whether store.remove succeeded: if the
                    # write failed the entry is still semantically terminal, and a
                    # future re-dispatch with a new Future will re-register cleanup.
                    self._journaled_request_ids.discard(_rid)

                item.result.add_done_callback(_on_terminal)

    def _coalesce_reentrant_drain(
        self, item: MergeRequest, current_state: ItemLifecycleState,
    ) -> None:
        """Coalesce a duplicate/re-entrant drain of *item* onto a request_id
        already live in the registry at *current_state* (task 2852 /
        SHARPER RCA option B — mr-8cafbaf0, original incident mr-c69ca480).

        Called by :meth:`buffer_owned_request` BEFORE the QUEUED ->
        LANE_BUFFERED transition, the lane-buffer append, and the
        ``_merge_store.record``/``_on_terminal`` journal block — dropping
        *item* here (rather than after) prevents both a divergent second
        pipeline item under the shared request_id (which would otherwise
        keep firing rejected transitions for the rest of the live
        original's lifetime, exactly the reported incident) and a journal
        clobber of the live original's durable entry under that same
        request_id.

        Two cases, discriminated by RESULT-FUTURE IDENTITY against the live
        registry object (task 3082) — see :attr:`IntakeLifecycle.live_request`:

        **A genuine TWIN** (a distinct object with its own fresh, unobserved
        future — the confirmed journal-recovery shape). Logged loudly at
        WARNING and deliberately NOT escalated: once recognized here the
        divergence is HANDLED and benign, and escalating would re-introduce
        the exact ``merge_lifecycle_transition_rejected`` noise task 2852
        removes (PRD design decision 4: invariants escalate loudly, degrade
        never — satisfied via the WARNING without silently degrading). Its
        ``result`` Future is resolved (if not already done) to a benign
        ``MergeOutcome(status='already_merged')`` so no hypothetical waiter
        hangs forever. For THIS case only, that resolution is robustness
        rather than a correctness dependency.

        **The LIVE ORIGINAL** (the same ``result`` future the live registry
        object carries). Logged at ERROR and ESCALATED via
        :func:`_alarm_coalesce_live_original`, and its Future is deliberately
        left PENDING. Here the resolution is load-bearing and WRONG in both
        directions: it lies to a REAL waiter (an ``already_merged`` that
        escalation's server maps to ``'done'``), and it silences the only
        watchdog that would catch the wedge — the request ledger's
        ``sweep_resolved`` drops any ``done()`` entry before the stuck sweep
        sees it, so ``merge_request_stuck`` never fires. Leaving it pending
        re-arms that alarm as the slower backstop, so the two signals are
        complementary rather than duplicative. Unreachable in normal
        operation: every requeue site pairs ``on_requeued`` with
        ``_note_requeue``, so it fires only if a requeue site regresses.

        In BOTH cases a ``merge_coalesced`` observability event is emitted via
        the shared :func:`emit_merge_coalesced` helper (None-safe when no
        event store is wired) — observability must not depend on which branch
        was taken. The ``source`` DISCRIMINATES them, because the two
        conditions are not the same event: ``'duplicate_submission'`` for a
        twin, which carries the established meaning at every other emission
        site (attached to an in-flight merge, will receive that merge's
        outcome), versus ``'live_original_redrain'`` for the live original,
        where nothing is coalesced at all — the request is dropped, its
        future stays PENDING and no outcome will ever arrive. Consumers
        aggregating coalesces must exclude the latter rather than count a
        wedge as a benign attach.
        """
        # task 3082: is the arriving item the LIVE ORIGINAL, or a genuine twin?
        # Discriminated by FUTURE identity, deliberately NOT by object identity
        # of the item: the live registry object legitimately changes SHAPE
        # across the pipeline (MergeRequest -> SpeculativeItem -> InflightEntry)
        # while carrying the SAME `request.result`, so an `is`-on-the-item check
        # would fail open exactly when the entry is deepest in the pipeline —
        # which is when this bug bites.
        _live_req = self._lifecycle.live_request(item.request_id)
        _is_live_original = _live_req is not None and _live_req.result is item.result

        if _is_live_original:
            logger.error(
                'merge_queue: _buffer_owned_request: re-entrant drain of the LIVE '
                'ORIGINAL request_id=%s branch=%s; registry at %s. NOT buffering '
                'and NOT resolving its Future (a real waiter is attached) — this '
                'is a requeue-wiring bug, not a benign duplicate twin. Escalated.',
                item.request_id, item.branch.bare_id, current_state,
            )
            _alarm_coalesce_live_original(
                self._escalation_queue, item, current_state,
                event_store=self._event_store,
            )
        else:
            logger.warning(
                'merge_queue: _buffer_owned_request: dropping duplicate/re-entrant '
                'merge submission for request_id=%s branch=%s; registry already at '
                '%s. Coalescing as a benign no-op — NOT buffering a divergent '
                'twin, NOT escalated.',
                item.request_id, item.branch.bare_id, current_state,
            )
            if not item.result.done():
                item.result.set_result(
                    MergeOutcome(
                        status='already_merged',
                        reason=(
                            'duplicate/re-entrant merge submission coalesced onto '
                            'in-flight/landed request'
                        ),
                    ),
                )
        # Emitted UNCONDITIONALLY in both branches: observability must not depend
        # on which one was taken — but with a DISCRIMINATING source, since only
        # the twin was actually coalesced onto a merge that will deliver it an
        # outcome (see the docstring).
        emit_merge_coalesced(
            self._event_store, item,
            source='live_original_redrain' if _is_live_original else 'duplicate_submission',
            eta=None,
        )

    def drain_into_lanes(self) -> bool:
        """Non-blocking drain of the queue into the per-lane buffers.

        Returns True when the shutdown sentinel (None) was consumed, after
        buffering every item that arrived before it.
        """
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if item is None:
                return True  # sentinel consumed — all prior items already buffered
            self.buffer_owned_request(item)
        return False

    def pop_next_pickable(self) -> MergeRequest | None:
        """Return the next pickable request using clique-scoped aging (ζ=1891).

        Lane priority is unchanged: high beats normal; halted lanes are skipped.
        Within the first non-empty non-halted lane, pick and remove the
        **clique-minimal** item: the FIFO-earliest buffered item *x* such that
        no footprint-neighbor of *x* (per :attr:`suffix_conflict_graph`) in
        the same lane has a strictly smaller :func:`_aging_key`.

        Aging key = ``(merge_first_enqueued_at or enqueued_at, request_id)`` —
        older (smaller timestamp) wins; lexically-smaller request_id breaks ties.

        Degrades to pure FIFO when the conflict graph is empty or the
        request_id is absent from the graph (``footprint_neighbors`` returns
        the empty set → item is vacuously minimal).

        Returns None when every non-empty lane is halted or all buffers are
        empty.  Pure/synchronous so unit tests run without an event loop.

        MQ-reliability kappa (task 2169): every item in a lane buffer got
        there via :meth:`buffer_owned_request` (fresh drain) or the coalesce
        QUEUE SURGERY (the worker's ``_maybe_coalesce_waiting_singles``
        GroupMergeRequest, registered directly at LANE_BUFFERED and placed by
        :meth:`replace_lane`) — both leave the registry at LANE_BUFFERED, so a
        pop here is always a LANE_BUFFERED->MERGING transition. This is the
        ONLY place that picks from the lane buffers (``_acquire_next_request``'s
        regular pick and the merger loop's speculative look-ahead both call
        this method), so wiring the transition here — rather than at each
        caller — covers both paths in one chokepoint; by the time either caller
        resumes processing *req* as the merger loop's dequeued item, the
        registry already reads MERGING for it (task 2435 kappa-b: formerly
        illustrated by the now-deleted ``self._inflight_req = req``
        assignment).

        Returns None too while admission is halted (the worker's
        ``halt_admission``).

        A request a deep chain reserves (task 7720) is invisible here, as a
        candidate and as a neighbour: it belongs to that chain's round, whose
        walk takes it through :meth:`take_buffered`. A lane holding only
        reserved requests falls through to the next lane.
        """
        if self._admission_halted():
            return None
        self._rebucket_relaned_requests()
        for lane in self._lanes:  # high → normal
            if self._halt.is_lane_halted(lane):
                continue
            picked = self._clique_minimal_visible(lane)
            if picked is None:
                continue
            self._assert_merger_owns('_lane_buffers')
            buf = self._lane_buffers[lane]
            del buf[next(i for i, req in enumerate(buf) if req is picked)]
            self._lifecycle.note_transition(
                picked.request_id, ItemLifecycleState.LANE_BUFFERED, ItemLifecycleState.MERGING,
                live_obj=picked,
            )
            self._pick_records.record(picked, lane, self._lane_buffers, self._suffix_tracker.graph)
            return picked
        return None

    def _clique_minimal_visible(self, lane: str) -> MergeRequest | None:
        """The FIFO-earliest unreserved request in *lane* that no unreserved
        footprint-neighbour in *lane* out-ages; None when nothing is visible.

        Items with no visible footprint-neighbour are vacuously minimal, so
        disjoint items pick in pure FIFO, and the pick degrades to FIFO when the
        graph is empty or a request is absent from it. The aging-minimal
        visible item always qualifies, so the FIFO head is only a defensive
        fallback.
        """
        visible = [
            req for req in self._lane_buffers[lane] if not self._is_reserved(req.request_id)
        ]
        if not visible:
            return None
        graph = self._suffix_tracker.graph
        by_rid = {req.request_id: req for req in visible}
        for x in visible:
            neighbours = graph.footprint_neighbors(x.request_id) & by_rid.keys()
            key_x = _aging_key(x)
            if not any(_aging_key(by_rid[rid]) < key_x for rid in neighbours):
                return x
        return visible[0]

    def _rebucket_relaned_requests(self) -> None:
        """Move each buffered request whose ``lane`` was changed by
        :meth:`relane_waiting` into that lane's buffer, at its aging position
        (task 5246).

        Runs on the merger task, the buffers' owner, so it never lands inside
        an await that holds a view of them (the coalescing pass does).  The
        aging position, not the tail, because a moved request has no arrival
        slot in its new lane: placing it by first submission (the key
        :meth:`pop_next_pickable` orders conflicting requests by) means a
        promoted old fix is not queued behind younger high-lane work.
        """
        for lane in self._lanes:
            buf = self._lane_buffers[lane]
            strays = [req for req in buf if normalize_lane(req.lane) != lane]
            if not strays:
                continue
            self._assert_merger_owns('_lane_buffers')
            for req in strays:
                buf.remove(req)
                target = self._lane_buffers[normalize_lane(req.lane)]
                key = _aging_key(req)
                idx = next(
                    (i for i, queued in enumerate(target) if _aging_key(queued) > key),
                    len(target),
                )
                target.insert(idx, req)

    # ── lane-buffer requests from the other concerns ───────────────────────

    def relane_waiting(
        self, lane: Literal['normal', 'high'], request_id: str | None, task_id: str | None,
    ) -> tuple[MergeRequest, Literal['normal', 'high']] | None:
        """Set *lane* on the undecided request still waiting for the merger that
        :func:`answers_to` (*request_id*, *task_id*); return it with its former lane.

        Searches the lane buffers, then the not-yet-drained arrivals.  Only
        ``req.lane`` changes here, plus a solo request's journal record so a
        restart keeps it (a train is never journaled).  The move between
        buffers is the merger's own, at the top of :meth:`pop_next_pickable`.
        Returns None when nothing waiting answers to that id.  No await: lookup
        and write are one atomic step on the loop.
        """
        waiting = [
            *(req for ln in self._lanes for req in self._lane_buffers[ln]),
            *self.queued_arrivals(),
        ]
        target = next(
            (req for req in waiting
             if not req.result.done() and answers_to(req, request_id, task_id)),
            None,
        )
        if target is None:
            return None
        from_lane = target.lane
        target.lane = lane
        if self._merge_store is not None and target.request_id in self._journaled_request_ids:
            try:
                self._merge_store.record(target)
            except Exception:
                logger.warning(
                    'merge_queue: relane: store.record failed for %s',
                    target.request_id, exc_info=True,
                )
        return target, from_lane


    def take_buffered(self, task_id: str) -> tuple[str, int, MergeRequest] | None:
        """Remove and return the first lane-buffered request for *task_id*.

        Returns ``(lane, index, request)`` — the coordinates
        :meth:`restore_buffered` needs to put it back EXACTLY where
        it was if the landing it was taken for does not happen.

        The queue half of δ's walk (task 3186): a chain-landed link must leave
        the buffers in the same synchronous beat it is resolved, or the merger
        re-picks a request that is already on main and merges it a second time.

        Search order is ``chain_snapshot``'s — high lane first, FIFO within
        lane — because that is the order ``build_chain`` itself snapshotted, so
        a duplicate ``task_id`` (a regenerated request) resolves to the SAME
        object the chain was built from rather than a later twin.

        Returns ``None`` when the request is no longer buffered.  That is a
        REAL state, not a defensive branch: ``_run_inflight_verify`` awaits a
        whole verify between the build and this walk.  A reserved link cannot
        be popped (task 7720), but once its reservation has ended (expiry, or a
        release) the merger loop may :meth:`pop_next_pickable` it, and a
        cancel or a relane can take it at any time.  The caller MUST
        stop the walk on ``None`` — the pipeline has taken ownership of that
        item elsewhere, and landing it from here would be the double-land the
        removal above exists to prevent.

        **Cross-writer note.**  The lane buffers' single-writer owner is the
        worker's ``_merger_task`` (``assert_merger_owns``), and this runs on
        its ``_verifier_task``.  The mutation is nonetheless safe and deliberate,
        for the same reason ``stop()``'s drain is (merge_queue.py's "legitimate
        non-owner drain" comment): the scan and the delete are one synchronous
        statement pair with no ``await`` between them, so no other coroutine
        can observe a torn buffer.  ``assert_merger_owns`` is deliberately
        NOT called — it would fire on this legitimate second writer — and the
        None-return contract above is what covers the interleaving the guard
        would otherwise be protecting.
        """
        for lane in self._lanes:
            buf = self._lane_buffers[lane]
            for idx, buffered in enumerate(buf):
                if buffered.task_id == task_id:
                    del buf[idx]
                    return lane, idx, buffered
        return None

    def restore_buffered(self, lane: str, index: int, req: MergeRequest) -> None:
        """Put a taken-but-not-landed chain link back at its exact position.

        The undo half of :meth:`take_buffered`, and the reason δ's
        stale-CAS abort really is "the remaining links stay queued UNTOUCHED"
        (PRD decision #9) rather than merely "not failed".

        Deliberately NOT the worker's ``_requeue_request``: that recipe is for an item
        that was genuinely IN FLIGHT, and its three effects are all wrong here.
        It would move the link onto ``_queue`` (losing its lane position and
        its place in submission order), fire ``on_requeued`` for a round the
        link never entered, and drive ``_note_requeue`` to attempt
        ``LANE_BUFFERED -> QUEUED`` — an edge absent from
        :data:`_LEGAL_TRANSITIONS`, so it would escalate a rejected transition
        on every deep round that loses a CAS race.  A link that never left its
        buffer needs no re-arming at all; it needs its slot back.
        """
        self._lane_buffers[lane].insert(index, req)

    def remove_buffered(self, request_ids: Collection[str]) -> None:
        """Drop every buffered request whose request_id is in *request_ids*."""
        self._assert_merger_owns('_lane_buffers')
        for lane in self._lanes:
            self._lane_buffers[lane] = collections.deque(
                req for req in self._lane_buffers[lane] if req.request_id not in request_ids
            )

    def replace_lane(self, lane: str, requests: Iterable[MergeRequest]) -> None:
        """Rebuild *lane*'s buffer as *requests*, in order (the coalesce queue surgery)."""
        self._assert_merger_owns('_lane_buffers')
        self._lane_buffers[lane] = collections.deque(requests)

    def drain_lanes_for_shutdown(self) -> tuple[MergeRequest, ...]:
        """Empty every lane, returning its requests in pick order.

        ``stop()``'s legitimate non-owner drain, covered by the worker's
        single-writer assert exempting a stopped worker (task 1999 I7).
        """
        drained = tuple(req for lane in self._lanes for req in self._lane_buffers[lane])
        for lane in self._lanes:
            self._lane_buffers[lane].clear()
        return drained

    # ── the redispatch park ────────────────────────────────────────────────

    def request_redispatch(self, item: SpeculativeItem, *, front: bool = False) -> None:
        """Park *item* for re-dispatch: at the back, or ahead of everything with *front*."""
        park = self._redispatch.appendleft if front else self._redispatch.append
        park(item)

    def next_redispatch(self) -> SpeculativeItem:
        """Pop the next parked item.  Raises IndexError when nothing is parked."""
        return self._redispatch.popleft()

    # ── the suffix-conflict state ──────────────────────────────────────────

    def note_main_sha(self, sha: str | None) -> None:
        """Cache the real main SHA last seen landing or finalizing."""
        self._suffix_tracker.last_known_main_sha = sha

    async def recompute_suffix_conflict_graph(self) -> None:
        """Thin delegator (task δ=1988) — see :meth:`SuffixConflictTracker.recompute`
        (orchestrator.suffix_graph) for the full debounce / fail-open /
        textual-pruning contract."""
        await self._suffix_tracker.recompute()

    async def bounce_conflicting_suffix_items(self) -> None:
        """Thin delegator (task δ=1988) — see
        :meth:`SuffixConflictTracker.bounce_conflicting_suffix_items`
        (orchestrator.suffix_graph) for the full cap/escalation/TOCTOU
        contract.  Called by the worker's ``_acquire_next_request`` immediately
        after :meth:`recompute_suffix_conflict_graph` and before
        :meth:`pop_next_pickable`."""
        await self._suffix_tracker.bounce_conflicting_suffix_items()
