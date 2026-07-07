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
  so this clears ``held_by_merger`` WITHOUT releasing the semaphore. Used
  ONLY at the single post-merge-success look-ahead site — ``spec_base`` is
  deliberately left untouched because the immediately-following look-ahead
  call (``on_lookahead_found`` / ``on_lookahead_pending`` / ``on_shutdown``)
  re-derives or clears it.
* ``on_transfer_terminal`` — the same held-by-merger clear as ``on_transfer``,
  used at the seven early-continue sites (already_merged / conflict /
  merge-fail / abandoned / revparse-fail / drop / branch-presence-guard)
  where the loop ``continue``s immediately with no subsequent look-ahead to
  re-derive state. ALSO clears ``spec_base`` — mirrors the original
  loop-locals, which set both ``held_spec_permit = False`` AND
  ``spec_base = None`` at these sites.
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

from orchestrator.merge_types import SpecPermit

if TYPE_CHECKING:
    import asyncio

    from orchestrator.merge_types import MergeRequest


class PermitLedger:
    """Single owner of a shared speculation-slot semaphore (MQ-refactor zeta
    / task 2159).

    Wraps an externally-owned ``asyncio.Semaphore`` (injected — never owns or
    creates its own, mirroring :class:`SpeculationController`'s existing
    contract) and mediates acquire/release through a :class:`SpecPermit`
    token registered in ``live``. Conservation is structural: ``slot_available
    + len(live) == depth`` holds after every acquire/release call because
    neither mutates across an ``await`` boundary — on the single asyncio
    event loop, each pair (semaphore + ``live``) is therefore atomic from
    every other coroutine's perspective.

    In task zeta this ledger WRAPS the worker's existing
    ``_speculation_slot`` while the verifier side continues to release that
    same semaphore directly (raw, not through this ledger) — task eta
    migrates those call sites to ``ledger.release(item.permit)``. Until then,
    ``live`` is not yet the pipeline's sole source of truth for in-flight
    permits; only the acquire/release paths already routed through this
    ledger (currently just :class:`SpeculationController`'s merger-side
    lifecycle) are covered by the identity above.
    """

    def __init__(self, slot: asyncio.Semaphore, depth: int) -> None:
        self._slot = slot
        self._depth = depth
        self._live: set[SpecPermit] = set()

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def slot_available(self) -> int:
        """Read the wrapped semaphore's internal counter directly.

        Safe without a lock: both this read and every mutation happen on the
        single asyncio event loop (mirrors
        ``SpeculationController.snapshot()``'s existing non-blocking
        contract).
        """
        return self._slot._value  # type: ignore[attr-defined]

    @property
    def live(self) -> frozenset[SpecPermit]:
        """Read-only view of the currently-outstanding permits."""
        return frozenset(self._live)

    async def acquire(self) -> SpecPermit:
        """Acquire one permit, returning a fresh token registered in ``live``.

        No ``await`` between the semaphore acquire and the ``live``
        registration below, so the pair is atomic from every other
        coroutine's perspective.
        """
        await self._slot.acquire()
        permit = SpecPermit()
        self._live.add(permit)
        return permit

    def release(self, permit: SpecPermit) -> None:
        """Release *permit*, restoring one slot. Idempotent + live-checked.

        Checks ``permit.released`` FIRST so a double-release is a silent
        no-op — never an over-release, never an assertion failure. Only past
        that guard does it assert the token is actually live, catching a
        genuinely-unknown token as a programming error.
        """
        if permit.released:
            return
        assert permit in self._live, (
            f'PermitLedger.release: permit {permit!r} is not live (a token '
            'this ledger never issued, or already discarded)'
        )
        permit.released = True
        self._live.discard(permit)
        self._slot.release()


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

        Used ONLY at the single post-merge-success look-ahead site:
        ``spec_base`` is deliberately left as-is because the
        immediately-following look-ahead call (``on_lookahead_found`` /
        ``on_lookahead_pending`` / ``on_shutdown``) re-derives or clears it.
        For the seven early-continue sites (no subsequent look-ahead ever
        runs for this permit), use ``on_transfer_terminal`` instead so
        ``spec_base`` does not leak stale between requests.
        """
        self._held = False

    def on_transfer_terminal(self) -> None:
        """The merger has handed a TERMINAL (early-continue) item to the verifier.

        Like ``on_transfer``, clears ``held_by_merger`` WITHOUT releasing the
        semaphore — the verifier now owns the permit. UNLIKE ``on_transfer``,
        ALSO clears ``spec_base``: this is for the seven early-continue sites
        (already_merged / conflict / merge-fail / abandoned / revparse-fail /
        drop / branch-presence-guard) where the loop ``continue``s
        immediately afterward, so no subsequent look-ahead call will ever
        run to re-derive ``spec_base`` for this permit. Mirrors the original
        loop-locals, which set BOTH ``held_spec_permit = False`` AND
        ``spec_base = None`` at these sites (``merge_queue.py``'s Step
        0/0.5/1/2/3 short-circuit ``continue``s).

        Without this, ``spec_base`` would leak the just-completed request's
        base SHA across the ``continue``, wrongly reporting ``is_idle()`` as
        False and ``snapshot()['speculation']['spec_base']`` as non-None
        until the next ``on_dequeue`` happens to overwrite or clear it.
        """
        self._held = False
        self.spec_base = None

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
