"""Speculation state machine: explicit permit ownership for the merger loop
(MQ-refactor theta / task 1993).

Reifies the merger loop's implicit speculation state machine — previously
five loop-locals (``spec_base``, ``prefetched``, ``held_spec_permit``,
``pending_spec_base``, ``pending_predecessor``) threaded through
``_merger_loop`` — into a single :class:`SpeculationController` object with
EXPLICIT permit-ownership-transfer semantics. ``exited_via_sentinel`` stays a
``_merger_loop`` local (it governs finally-block sentinel forwarding,
orthogonal to permit/speculation state) and is NOT owned by this class.

Permit lifecycle
-----------------
The Merger and Verifier coroutines share one ``_speculation_slot``
(``asyncio.Semaphore``, depth K). :class:`SpeculationController` holds a
REFERENCE to that shared semaphore (injected — it never owns/creates its
own) and drives only the MERGER side of its lifecycle:

* ``acquire_for_lookahead()`` — acquire one permit before peeking for a
  speculative look-ahead item (depth-K cap). ``held_by_merger`` becomes 1.
* ``on_lookahead_found`` — a pickable item was found; the permit is
  RETAINED (not released) because the Verifier will release it on drain of
  the resulting speculative item.
* ``on_lookahead_pending`` — nothing pickable yet but the predecessor is
  still in-flight; the permit is RETAINED for a possible late-arrival
  ATTACH (task 1862) at the next dequeue.
* ``on_dequeue`` — the ATTACH/FALLBACK four-condition decision for a
  freshly dequeued request: ATTACH retains the permit (it transfers to the
  verifier when the attached item is put); FALLBACK releases it
  immediately.
* ``on_transfer`` — the merger has handed a speculative item to the
  verifier via ``_verifier_queue.put()``; the verifier now owns the permit,
  so this clears ``held_by_merger`` WITHOUT releasing the semaphore.
* ``on_abort`` — a guard/failure/exception/train short-circuit before a
  put; releases the permit if held (covers the request_abandoned/train/
  WorktreeMissing/Exception in-body releases).
* ``on_shutdown`` — releases any held permit and clears all five fields
  (covers the shutdown-after-lookahead branch and the outer ``finally``).

The verifier-side releases (``_resolve_and_release``'s ``was_speculative``/
``speculative`` release, ``_finalize_inflight``'s finally release, and the
cascade re-merge release) are UNCHANGED by this class — they keep releasing
the same shared semaphore directly. This controller owns only the
merger-side acquire/transfer/retain/release lifecycle.

Plain-Semaphore double-release tolerance
-----------------------------------------
``_slot`` MUST remain a plain ``asyncio.Semaphore`` — never a
``BoundedSemaphore``. A narrow ``CancelledError``-after-``put()`` race can
cause the same permit to be released twice (once by the merger's outer
``finally`` and once by the verifier's drain) — see ``merge_queue.py``'s
speculative-queue-put site comment. A plain ``Semaphore`` tolerates this
(its internal counter simply increments past the original bound); a
``BoundedSemaphore`` would raise ``ValueError`` and crash the shutdown
path. Every release call in this module is therefore an unguarded
``self._slot.release()`` — never wrapped in extra bookkeeping to prevent
over-release.

``merge_queue.py`` re-exports :class:`SpeculationController` through its
top-level shim so existing importers (``from orchestrator.merge_queue
import SpeculationController``) keep working unchanged — see that module's
re-export shim block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio

    from orchestrator.merge_types import MergeRequest


class SpeculationController:
    """Owns the merger loop's speculation state + merger-side permit lifecycle.

    Constructed with a REFERENCE to the worker's shared ``_speculation_slot``
    semaphore (never its own) so the verifier-side releases keep operating
    on the same object. See the module docstring for the full lifecycle.
    """

    def __init__(self, slot: asyncio.Semaphore, depth: int) -> None:
        self._slot = slot
        self._depth = depth
        # True exactly while the merger holds one _slot permit that has NOT
        # yet been handed off to the verifier (on_transfer) or released
        # (on_abort / on_shutdown / on_dequeue's FALLBACK branch).
        self._held: bool = False
        # SHA to use as base for the CURRENT request's merge; None → merge
        # against actual main HEAD (non-speculative).
        self.spec_base: str | None = None
        # Pre-fetched next request grabbed speculatively from the main queue.
        self.prefetched: MergeRequest | None = None
        # Late-arrival attach state (task 1862): predecessor's merge commit,
        # recorded when the look-ahead peek finds nothing pickable but the
        # predecessor is still in-flight. Consumed by on_dequeue's ATTACH/
        # FALLBACK decision at the next dequeue.
        self.pending_spec_base: str | None = None
        self.pending_predecessor: MergeRequest | None = None

    @property
    def held_by_merger(self) -> int:
        """1 if the merger currently holds a speculation permit, else 0.

        An int (not bool) — this is the queryable operand the downstream
        conservation-audit task (iota) sums against ``slot_available`` and
        ``inflight_speculative`` to check the identity ``== depth``.
        """
        return 1 if self._held else 0

    def is_idle(self) -> bool:
        """True when there is no in-flight speculation state at all.

        Mirrors the merger loop's DD1 coalesce gate: ``spec_base``,
        ``prefetched``, and ``pending_spec_base`` all None.
        ``pending_predecessor`` is not checked separately — it is always
        set/cleared in lockstep with ``pending_spec_base``.
        """
        return (
            self.spec_base is None
            and self.prefetched is None
            and self.pending_spec_base is None
        )

    def take_prefetched(self) -> MergeRequest | None:
        """Consume and return the pre-fetched request, or None if absent."""
        req = self.prefetched
        self.prefetched = None
        return req

    def base_for(self, actual_main: str) -> str:
        """Return the SHA the current request should merge against.

        ``spec_base`` when speculating, else the caller-supplied
        ``actual_main`` (plain non-speculative merge).
        """
        return self.spec_base if self.spec_base is not None else actual_main

    def on_dequeue(self, req: MergeRequest) -> str | None:
        """Decide ATTACH vs FALLBACK for a freshly dequeued request.

        ATTACH — all four conditions hold: ``pending_spec_base`` is
        recorded, a permit is held, ``pending_predecessor`` is known, and
        the predecessor is still in-flight (its result Future is not yet
        done). The held permit is RETAINED — it transfers to the verifier
        when the attached item is eventually put via ``on_transfer`` — and
        ``spec_base`` becomes the predecessor's recorded commit so this
        late arrival merges against it instead of plain main.

        FALLBACK — any condition fails: release the held permit (if any)
        and merge against plain main (``spec_base`` cleared to None).

        Either way, ``pending_spec_base``/``pending_predecessor`` are
        cleared — consumed by ATTACH or dropped by FALLBACK. Mirrors
        ``_merger_loop``'s ATTACH/FALLBACK decision (task 1862).

        ``req`` (the freshly dequeued request) does not affect the
        decision itself — it is accepted for call-site symmetry with the
        original loop-local logic and future extension (e.g. logging).
        """
        if (
            self.pending_spec_base is not None
            and self._held
            and self.pending_predecessor is not None
            and not self.pending_predecessor.result.done()
        ):
            self.spec_base = self.pending_spec_base  # ATTACH
        else:
            # FALLBACK — release retained permit (if any); over-release is
            # avoided by the `if self._held` guard, never by the semaphore
            # type (see module docstring's double-release tolerance note).
            if self._held:
                self._slot.release()
                self._held = False
            self.spec_base = None
        # Clear pending locals — consumed by ATTACH or dropped by FALLBACK.
        self.pending_spec_base = None
        self.pending_predecessor = None
        return self.spec_base

    async def acquire_for_lookahead(self) -> None:
        """Acquire one speculation permit before peeking for a look-ahead item.

        The depth-K cap acquire (mirrors ``_merger_loop``'s look-ahead site,
        ``merge_queue.py:6936-6937``). Blocks until a permit is available;
        sets ``held_by_merger`` to 1 once acquired.
        """
        await self._slot.acquire()
        self._held = True

    def on_lookahead_found(self, next_req: MergeRequest, merge_commit: str) -> None:
        """Record a pickable look-ahead item found while a permit is held.

        The permit STAYS held (not released) — the Verifier releases it on
        drain of the resulting speculative item. Mirrors
        ``merge_queue.py:6951-6952``.
        """
        self.prefetched = next_req
        self.spec_base = merge_commit

    def on_lookahead_pending(self, merge_commit: str, predecessor: MergeRequest) -> None:
        """Record a late-arrival-pending state: nothing pickable yet, but the
        predecessor is still in-flight.

        The permit STAYS held (retained) for a possible ATTACH at the next
        ``on_dequeue``. Mirrors ``merge_queue.py:6975-6977``.
        """
        self.pending_spec_base = merge_commit
        self.pending_predecessor = predecessor

    def on_transfer(self) -> None:
        """The merger has handed a speculative item to the verifier.

        Clears ``held_by_merger`` WITHOUT releasing the semaphore — the
        verifier now owns the permit and will release it itself on drain
        (the zeta chokepoint). Mirrors ``merge_queue.py:6926``.
        """
        self._held = False

    def on_abort(self) -> None:
        """A guard/failure/exception/train short-circuit before a put.

        Releases the held permit (if any — guarded against over-release,
        never by the semaphore type) and clears ``spec_base``. Covers the
        merger's request_abandoned/train/WorktreeMissing/Exception in-body
        releases (``merge_queue.py:6540/6645/7013/7045``).
        """
        if self._held:
            self._slot.release()
            self._held = False
        self.spec_base = None

    def on_shutdown(self) -> None:
        """Worker shutdown: release any held permit and clear all state.

        Releases the held permit (if any) and clears all five fields
        (``spec_base``, ``prefetched``, ``pending_spec_base``,
        ``pending_predecessor``, and — via the release — ``held_by_merger``).
        Idempotent when already idle. Covers the shutdown-after-lookahead
        branch and the outer ``finally`` (``merge_queue.py:6958-6962`` and
        ``:7052-7053``).
        """
        if self._held:
            self._slot.release()
            self._held = False
        self.spec_base = None
        self.prefetched = None
        self.pending_spec_base = None
        self.pending_predecessor = None

    def snapshot(self) -> dict[str, Any]:
        """Return a synchronous read-only snapshot of the controller's state.

        Additive-value dict consumed by ``SpeculativeMergeWorker.snapshot()``'s
        ``'speculation'`` key (task 1993 step-10). ``slot_available`` reads
        the shared semaphore's internal counter directly — safe here because
        both this read and every mutation happen on the single asyncio event
        loop (no lock needed; mirrors ``snapshot()``'s existing non-blocking
        contract).
        """
        return {
            'depth': self._depth,
            'held_by_merger': self.held_by_merger,
            'spec_base': self.spec_base,
            'prefetched_task_id': (
                self.prefetched.task_id if self.prefetched is not None else None
            ),
            'pending_spec_base': self.pending_spec_base,
            'pending_predecessor_task_id': (
                self.pending_predecessor.task_id
                if self.pending_predecessor is not None else None
            ),
            # Internal Semaphore counter — no public accessor exists; the
            # controller and worker both run on the single asyncio event
            # loop so this read is never torn.
            'slot_available': self._slot._value,  # type: ignore[attr-defined]
        }
