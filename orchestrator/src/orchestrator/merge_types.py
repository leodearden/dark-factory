"""Merge-queue data types and registries (MQ-refactor task α).

Extracted verbatim from :mod:`orchestrator.merge_queue`: the request /
outcome / item / entry dataclasses and the registries that own them.
``merge_queue`` re-exports every name here through a top-level shim so
existing importers (``from orchestrator.merge_queue import MergeRequest``,
etc.) keep working unchanged.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from orchestrator.git_ops import MergeResult
from orchestrator.verify import VerifyResult

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig


class MainHealthAutoHealRegistry:
    """Monotonic per-signature attempt counter for main-health auto-heal.

    Keyed by sha-INDEPENDENT failure signatures (workflow._merge_outcome_signature),
    so a recurrence at a new main SHA (after a fix advanced main) is detected and
    the attempt cap trips correctly.

    Thread safety: no synchronisation needed — the registry is owned by the merge
    worker and accessed only from asyncio tasks in the same event loop.
    """

    def __init__(self) -> None:
        self._attempts: dict[str, int] = {}

    def attempts(self, sig: str) -> int:
        """Return the number of auto-heal attempts recorded for *sig* (0 if none)."""
        return self._attempts.get(sig, 0)

    def record_attempt(self, sig: str) -> int:
        """Increment the attempt counter for *sig* and return the new count."""
        count = self._attempts.get(sig, 0) + 1
        self._attempts[sig] = count
        return count


class MergeBounceRegistry:
    """Monotonic per-branch bounce counter for the η=1892 needs-rebase bounce cap.

    Keyed by branch NAME (sha-independent), so the counter survives the
    HEAD-SHA churn of repeated rebases.  This is the same robustness principle
    as the 1688 thrash signature (which is sha-independent), realized here by
    mirroring :class:`MainHealthAutoHealRegistry`.

    Thread safety: no synchronisation needed — the registry is owned by the
    merge worker and accessed only from asyncio tasks in the same event loop.
    """

    def __init__(self) -> None:
        self._bounces: dict[str, int] = {}

    def count(self, branch: str) -> int:
        """Return the number of bounces recorded for *branch* (0 if none)."""
        return self._bounces.get(branch, 0)

    def record_bounce(self, branch: str) -> int:
        """Increment the bounce counter for *branch* and return the new count."""
        n = self._bounces.get(branch, 0) + 1
        self._bounces[branch] = n
        return n

    def clear(self, branch: str) -> None:
        """Remove the bounce counter for *branch*.

        Call when a branch is escalated (removed from the lane buffer) so that
        a later resubmission of the same branch name starts from zero rather
        than inheriting the old count.  This prevents a new PR on a recycled
        branch name from being prematurely cap-escalated.

        For the successful-merge path the counter is also moot once the branch
        has landed; wiring a clear() there would require hooking into
        advance_main / verifier completion — deferred as a future improvement.
        """
        self._bounces.pop(branch, None)


@dataclass
class _InFlightEntry:
    """Registry slot for a single in-flight merge branch.

    γ1 extended fields (all have defaults — existing callers unaffected):
    - ``branch`` / ``snapshot_tip`` / ``generation`` / ``verifying``:
      substrate for the PRD §7.2 multi-waiter coalescing model.
    - ``waiters``: ordered list of :class:`WaiterRecord` objects; the
      dispatcher is seeded as waiter #1 by :meth:`InFlightMergeRegistry.acquire`.
    - ``primary_future``: the future owned by the dispatcher (waiter #1).
      Stored so :meth:`~InFlightMergeRegistry.attach` can mirror its
      terminal outcome onto all later waiters' futures.
    """

    task_id: str
    enqueued_monotonic: float  # time.monotonic() at acquire time
    request_id: str | None = None
    """Stable identity of the dispatched MergeRequest (e.g. 'mr-a1b2c3d4').
    Set at acquire time from the MergeRequest.request_id; None for legacy
    callers that don't pass a request_id (back-compat)."""
    # ── γ1 multi-waiter substrate ─────────────────────────────────────────
    branch: str | None = None
    """Branch name (e.g. '591'), set at acquire time."""
    snapshot_tip: str | None = None
    """Git SHA of the branch tip at acquire time (PRD §7.2 snapshot_tip)."""
    generation: int = 1
    """Monotonically increasing generation counter; incremented on each
    re-snapshot that triggers a new merge attempt (γ2)."""
    verifying: bool = False
    """True once the merge worker has entered the post-merge verify phase.
    Used by :func:`decide_attach_action` to choose ATTACH_AND_CHAIN vs
    RESNAPSHOT for SUPERSET incoming submissions (γ2)."""
    waiters: list[WaiterRecord] = field(default_factory=list)
    """Ordered list of waiters; dispatcher is waiter #1 (seeded by
    :meth:`InFlightMergeRegistry.acquire`).  Empty ⟺ released."""
    primary_future: asyncio.Future | None = field(default=None, repr=False)
    """The dispatcher's future (waiter #1).  Resolved by the worker;
    its done-callbacks fan the terminal outcome out to all attached waiters."""


_INFLIGHT_MERGE_ETA_ESTIMATE_SECS: int = 600
"""Coarse estimate (seconds) for how long a full post-merge verify takes.
Used ONLY to compute a best-effort ETA for coalesced callers.  NOT a
guaranteed bound — cold npm+cargo builds vary widely.  Tests must NOT
assert a specific numeric value for ETA."""


@dataclass
class TerminalOutcomeRecord:
    """Immutable record of a MergeRequest's terminal outcome.

    Stored in the TerminalOutcomeRetention ring for O(1) hot-path lookups.
    The event store (merge_finalized rows) is the durable tier, so ring
    eviction is lossless — evicted ids fall through to event-store queries.
    """

    request_id: str
    task_id: str
    branch: str
    state: str
    """Terminal state: MergeOutcome.status value, 'abandoned' (cancelled), or 'error'."""
    snapshot_tip: str | None = None
    merge_sha: str | None = None
    finished_at: float = field(default_factory=time.time, kw_only=True)
    superseded_by: str | None = field(default=None, kw_only=True)
    """request_id of the gen-(n+1) request that supersedes this one (α1/γ2 provenance)."""
    generation: int = field(default=1, kw_only=True)
    """Generation of the merge request that produced this record (γ2 provenance)."""


class TerminalOutcomeRetention:
    """Bounded in-memory ring of recent terminal merge outcomes.

    Backed by a ``collections.deque(maxlen=maxlen)`` for eviction and three
    dict indexes for O(1) lookups:

    * ``_index`` — keyed by ``request_id`` (primary).
    * ``_by_branch`` — keyed by ``branch`` (secondary, newest-wins).
    * ``_by_task`` — keyed by ``task_id`` (secondary, newest-wins).

    When the ring is full, the oldest entry is evicted from all three indexes
    atomically using an identity guard — so evicting an old record never drops
    a secondary-index entry that is now owned by a newer live record (see
    ``record()`` docstring).

    A fourth ``_aliases`` ordered dict maps coalesced/absorbed request_ids to
    the primary request_id whose terminal record they should resolve to.
    Aliases resolve lazily through ``_index``; a dangling alias (primary
    evicted or not-yet-recorded) returns None — matching the ring's existing
    lossless eviction contract.  ``_aliases`` is capped at ``maxlen * 4``
    entries; when the cap is exceeded the oldest alias is dropped first
    (FIFO), as oldest entries are the most likely to be dangling and the
    least likely to be polled (see ``record_alias()`` docstring).

    The event store is the durable tier so eviction is lossless — α3's
    merge_status falls through to event-store queries for evicted ids.
    """

    def __init__(self, maxlen: int = 200) -> None:
        self._maxlen = maxlen
        self._ring: collections.deque[TerminalOutcomeRecord] = collections.deque(maxlen=maxlen)
        self._index: dict[str, TerminalOutcomeRecord] = {}
        self._by_branch: dict[str, TerminalOutcomeRecord] = {}
        self._by_task: dict[str, TerminalOutcomeRecord] = {}
        # OrderedDict so oldest alias can be FIFO-trimmed when the cap is hit.
        self._aliases: collections.OrderedDict[str, str] = collections.OrderedDict()

    def record(self, rec: TerminalOutcomeRecord) -> None:
        """Append *rec* to the ring, evicting the oldest entry from all indexes if full.

        Eviction uses an identity guard on each index: ``if index[key] is evicted``
        before deleting, so a newer record that already claimed the same
        branch/task_id key is never accidentally removed (Case B in the
        eviction-discipline tests).  The secondary-index prune runs BEFORE the
        new record is indexed, so a new record with the same branch/task_id
        replaces rather than loses its secondary entry.
        """
        if len(self._ring) == self._ring.maxlen:
            # Capture the about-to-be-evicted entry before appending.
            evicted = self._ring[0]
            self._ring.append(rec)
            # Only remove each index entry if it still points to *evicted* — a
            # newer record that already claimed the same key must be preserved.
            if self._index.get(evicted.request_id) is evicted:
                del self._index[evicted.request_id]
            if self._by_branch.get(evicted.branch) is evicted:
                del self._by_branch[evicted.branch]
            if self._by_task.get(evicted.task_id) is evicted:
                del self._by_task[evicted.task_id]
        else:
            self._ring.append(rec)
        self._index[rec.request_id] = rec
        self._by_branch[rec.branch] = rec
        self._by_task[rec.task_id] = rec

    def get(self, request_id: str) -> TerminalOutcomeRecord | None:
        """Return the record for *request_id*, or None if evicted / not yet recorded.

        When *request_id* is not in ``_index``, falls back to alias resolution:
        ``_aliases[request_id]`` → ``_index[primary]``.  A direct ``_index`` hit
        always takes precedence over an alias for the same id.  A dangling alias
        (primary evicted or not-yet-recorded) returns None — lossless fall-through.
        """
        rec = self._index.get(request_id)
        if rec is not None:
            return rec
        primary = self._aliases.get(request_id)
        if primary is not None:
            return self._index.get(primary)
        return None

    def record_alias(self, alias_id: str, primary_request_id: str) -> None:
        """Register *alias_id* as an alias for *primary_request_id*.

        Aliases resolve lazily through ``_index`` in ``get()``, so the primary
        record need not be recorded before the alias is registered — useful when
        a coalesced request_id is registered at coalesce time before the primary
        finalises.  Dangling aliases (primary evicted or never recorded) return
        None, matching the ring's lossless eviction contract.

        ``_aliases`` is capped at ``maxlen * 4`` entries (default 800).  When
        the cap is exceeded the oldest alias is dropped first (FIFO); oldest
        aliases are most likely dangling and least likely to be polled.  The
        cap prevents unbounded growth proportional to total coalesce traffic
        for long-running orchestrator processes.
        """
        self._aliases[alias_id] = primary_request_id
        _cap = self._maxlen * 4
        while len(self._aliases) > _cap:
            self._aliases.popitem(last=False)  # drop oldest alias FIFO

    def get_by_branch(self, branch: str) -> TerminalOutcomeRecord | None:
        """Return the most-recently recorded record for *branch*, or None if unknown."""
        return self._by_branch.get(branch)

    def get_by_task(self, task_id: str) -> TerminalOutcomeRecord | None:
        """Return the most-recently recorded record for *task_id*, or None if unknown."""
        return self._by_task.get(task_id)


class InFlightMergeRegistry:
    """Per-branch in-flight de-dup registry for the merge-request chokepoint.

    Tracks at most one in-flight merge request per branch (keyed by the bare
    branch name, e.g. ``"591"`` not ``"task/591"``).  The slot is acquired at
    dispatch and auto-released when the request's future resolves — via
    ``Future.add_done_callback`` — so neither MergeWorker nor
    SpeculativeMergeWorker needs any change.

    Thread safety: all callers run in the same asyncio event loop; the
    ``acquire`` check-and-set is synchronous so it is race-free within the
    loop (no ``await`` between the presence check and the dict write).
    """

    def __init__(self) -> None:
        self._slots: dict[str, _InFlightEntry] = {}

    def acquire(
        self,
        branch: str,
        task_id: str,
        future: asyncio.Future,
        *,
        request_id: str | None = None,
        source: str = 'mcp',
        submitted_tip: str | None = None,
        snapshot_tip: str | None = None,
    ) -> bool:
        """Atomic check-and-set: claim *branch* for *task_id*.

        Returns True if the slot was free and has been claimed; False if
        *branch* was already in-flight (caller should coalesce).

        On success, registers a ``done_callback`` on *future* so that
        ``_release_if_current(branch, entry)`` fires automatically on every
        terminal path (result set, exception set, or cancellation).  The
        identity guard ensures that a late callback from a cancelled stale
        future does NOT clobber a subsequently re-acquired slot for the same
        branch (see ``_release_if_current`` for details).

        *request_id* is the stable per-instance identity of the dispatched
        :class:`MergeRequest` (e.g. ``'mr-a1b2c3d4'``).  Stored on the
        :class:`_InFlightEntry` so β1 can re-source the ``'attached'``
        response from the existing entry's id rather than the submitting
        request's id (PRD D8).

        γ1 additions (keyword-only, all defaulted for back-compat):
        - *source*: origin of the dispatcher (``'mcp'`` or ``'workflow'``).
        - *submitted_tip*: branch tip SHA at submit time; used for
          tip-relation classification downstream.
        - *snapshot_tip*: branch tip SHA snapshotted at dispatch time
          (often the same as *submitted_tip*); stored on the entry for
          re-snapshot / gen-2 chaining (γ2).

        The dispatcher is seeded as waiter #1 in ``entry.waiters``; the
        entry invariant is ``in-flight ⟺ len(waiters) ≥ 1``.
        """
        if branch in self._slots:
            return False
        entry = _InFlightEntry(
            task_id=task_id,
            enqueued_monotonic=time.monotonic(),
            request_id=request_id,
            branch=branch,
            snapshot_tip=snapshot_tip,
            generation=1,
            primary_future=future,
        )
        entry.waiters = [WaiterRecord(
            request_id=request_id or '',
            future=future,
            source=source,
            submitted_tip=submitted_tip,
        )]
        self._slots[branch] = entry
        future.add_done_callback(lambda _: self._release_if_current(branch, entry))
        return True

    def is_inflight(self, branch: str) -> bool:
        """True when *branch* has an active in-flight slot."""
        return branch in self._slots

    def entry(self, branch: str) -> _InFlightEntry | None:
        """Return the in-flight entry for *branch*, or None if free."""
        return self._slots.get(branch)

    def eta_seconds(self, branch: str) -> int | None:
        """Best-effort ETA (seconds) for the in-flight merge of *branch*.

        Returns ``max(0, ESTIMATE - elapsed)`` as a coarse hint for pollers.
        This is NOT a guaranteed bound — cold builds vary widely.  Returns
        None when *branch* is not in-flight.
        """
        e = self._slots.get(branch)
        if e is None:
            return None
        elapsed = time.monotonic() - e.enqueued_monotonic
        remaining = _INFLIGHT_MERGE_ETA_ESTIMATE_SECS - elapsed
        if remaining <= 0:
            # Estimate exceeded — return None so callers fall back to a fixed
            # backoff rather than busy-polling with a saturated 0 estimate.
            # (ETA window is 600 s; liveness window is 10800 s — a worktree can
            # legitimately be in-flight long after the ETA estimate runs out.)
            return None
        return int(remaining)

    def attach(self, branch: str, waiter: WaiterRecord) -> bool:
        """Attach *waiter* as a peer on the in-flight entry for *branch*.

        Returns True if the branch is in-flight and the waiter was appended;
        False if the branch is free (caller must dispatch independently).

        Fan-out: registers a guarded done-callback on ``entry.primary_future``
        that mirrors its terminal outcome (result / exception / cancel) onto
        ``waiter.future``.  The guard skips ``waiter.future`` when it is
        already done (soft-cancel / detach race), so the callback never raises
        ``InvalidStateError``.

        Synchronous (no ``await``) — I10 race-freedom guaranteed within a
        single asyncio event loop tick.
        """
        entry = self._slots.get(branch)
        if entry is None:
            return False
        entry.waiters.append(waiter)
        primary = entry.primary_future
        if primary is not None and primary is not waiter.future:
            target = waiter.future

            def _mirror(pf: asyncio.Future) -> None:
                if target.done():
                    return  # guard: pre-resolved / detached future — skip
                if pf.cancelled():
                    target.cancel()
                elif (exc := pf.exception()) is not None:
                    target.set_exception(exc)
                else:
                    target.set_result(pf.result())

            primary.add_done_callback(_mirror)
        return True

    def detach(self, branch: str, request_id: str) -> int:
        """Remove the waiter identified by *request_id* from *branch*.

        Returns the remaining waiter count (0 when the last waiter detaches).

        When the count reaches zero the entry is considered abandoned:
        ``primary_future`` is cancelled (if not already done), which fires
        the existing acquire-time release done-callback (slot freed) and
        makes the existing ``_request_abandoned`` checkpoint return True at
        the next worker poll — dropping the queued work with ZERO worker code
        change.

        While ≥ 1 waiter remains the entry proceeds normally (boundary test
        10 substrate).

        Returns 0 for a branch not in-flight (no-op, safe to call).
        Unknown *request_id* is a no-op returning the unchanged count.

        **Resolution race guard (γ2/task 1640 wiring invariant):**
        ``primary_future`` may be cancelled by ``detach()`` while the worker
        has already passed an ``_request_abandoned`` checkpoint (returning
        False) but has not yet called ``req.result.set_result()`` /
        ``set_exception()``.  Calling either on a cancelled future raises
        ``InvalidStateError``.

        The established codebase convention — ``if not req.result.done():
        req.result.set_result(...)`` — is already present at every
        production resolution site (e.g. lines 3054-3055, 3065-3066,
        4456-4457) and degrades a detach-cancel race to a safe no-op.
        **All new worker resolution paths introduced by γ2 (task 1640) MUST
        use the same guard.**  No ``await`` may appear between an
        ``_request_abandoned`` checkpoint returning False and the subsequent
        resolution call.
        """
        entry = self._slots.get(branch)
        if entry is None:
            return 0
        entry.waiters = [w for w in entry.waiters if w.request_id != request_id]
        if not entry.waiters:
            pf = entry.primary_future
            if pf is not None and not pf.done():
                pf.cancel()
        return len(entry.waiters)

    def re_snapshot(self, branch: str, new_tip: str) -> bool:
        """Update the snapshot tip for *branch* to *new_tip*.

        Returns True on success; False if *branch* is not in-flight.
        The generation counter is NOT incremented here — the caller (γ2
        worker wiring) increments it when triggering a new merge attempt.
        """
        entry = self._slots.get(branch)
        if entry is None:
            return False
        entry.snapshot_tip = new_tip
        return True

    def set_verifying(self, branch: str, verifying: bool = True) -> None:
        """Set the ``verifying`` flag on the in-flight entry for *branch*.

        No-op when *branch* is not in-flight (safe to call on release race).
        """
        entry = self._slots.get(branch)
        if entry is not None:
            entry.verifying = verifying

    def release(self, branch: str, *, detach_waiters: bool = False) -> None:
        """Remove *branch* from the in-flight registry.

        Public surface for callers that need to release a slot on an
        exceptional path (e.g. enqueue failure before the worker can ever
        resolve the future).  The ``done_callback`` registered inside
        :meth:`acquire` is the normal release path; this method is the
        explicit fallback used by the slot-leak guards in
        :func:`register_and_enqueue_merge_request` and
        :func:`coalesce_or_enqueue_merge_request`.

        *detach_waiters* (keyword-only, default False):
        When True, every still-pending waiter future (including the primary,
        which is waiter #1) is cancelled atomically before the slot is
        popped.  Use this on the **abnormal / stale-reap path** where the
        primary is dead and any attached waiters would otherwise hang
        forever.

        Keep *detach_waiters=False* (the default) for the **normal
        terminal resolution path**: the acquire-time done-callback
        (``_release`` alias) fires BEFORE the attach() ``_mirror``
        callbacks; cancelling waiters here would make ``_mirror`` see them
        as already-done and skip delivery — regressing coalesced waiters
        from receiving the real outcome to being cancelled instead.
        """
        if detach_waiters:
            entry = self._slots.get(branch)
            if entry is not None:
                for w in entry.waiters:
                    if not w.future.done():
                        w.future.cancel()
                entry.waiters.clear()
        self._slots.pop(branch, None)

    def _release_if_current(self, branch: str, entry: _InFlightEntry) -> None:
        """Identity-aware acquire-time done-callback.

        Pops *branch* from ``_slots`` only when the stored entry is still
        *entry* (object-identity check via ``is``).

        **Why identity matters.**  ``Future.cancel()`` / ``Future.set_result()``
        schedule done-callbacks via ``loop.call_soon``, so they run on a LATER
        event-loop turn — not synchronously.  The stale-reap path in
        :func:`coalesce_or_enqueue_merge_request` calls
        ``release(branch, detach_waiters=True)``, which cancels the stale
        primary future and immediately re-acquires the slot for the same branch
        with a fresh request.  A branch-keyed ``_release(branch)`` would then
        pop the FRESH entry on the next loop turn, leaving the branch with no
        registry slot for the rest of the freshly-dispatched merge — so a
        concurrent :func:`merge_request` would not coalesce and would
        double-dispatch, the exact failure the registry exists to prevent.

        The identity guard makes the late callback a no-op once the slot has
        moved on (``self._slots.get(branch)`` returns the FRESH entry, not
        *entry*).  On the NORMAL terminal-resolution path the slot still IS the
        entry, so it pops exactly as before.
        """
        if self._slots.get(branch) is entry:
            self._slots.pop(branch, None)

    # Keep the private alias so callers that hold a reference to ``_release``
    # continue to work, and for the legacy path.  The acquire-time done-callback
    # now uses ``_release_if_current(branch, entry)`` instead (identity-aware).
    _release = release


@dataclass
class MergeDispatchResult:
    """Structured return value from :func:`coalesce_or_enqueue_merge_request`.

    Attributes:
        dispatched: True when the request was enqueued (new work item added).
        in_flight: True when a merge for *branch* was already in-flight and
            the caller was coalesced rather than enqueued.
        branch: The bare branch name (e.g. ``"591"``).
        inflight_task_id: task_id of the existing in-flight merger, or None
            when dispatched=True.
        eta_seconds: Best-effort ETA (coarse heuristic, NOT a bound), or None
            when dispatched=True or eta is unavailable.
        source: ``'registry'`` (in-memory) or ``'worktree'`` (disk scan),
            indicating which source detected the in-flight merger.  None when
            dispatched=True.
    """

    dispatched: bool
    in_flight: bool
    branch: str
    inflight_task_id: str | None = None
    eta_seconds: int | None = None
    source: str | None = None
    inflight_request_id: str | None = None
    """request_id of the already-in-flight MergeRequest for *branch* (D8).
    Set on coalesce from the _InFlightEntry's stored request_id; None when
    dispatched=True or the entry predates request_id tracking."""


@dataclass
class WaiterRecord:
    """Server-side durable-intent waiter record keyed by request_id.

    Registered in the ``merge_request`` MCP tool's closure dict
    ``_waiters[request_id]`` at dispatch time, and cleaned up via a
    ``future.add_done_callback``.  The ``asyncio.shield`` on every awaiting
    path (β1) ensures that cancelling the tool coroutine (MCP client
    disconnect) does NOT cancel ``future`` — only an explicit
    ``merge_cancel`` call (β2) may do so, by looking up this record and
    cancelling ``future`` directly.

    Lifecycle:
      - Created in the dispatched-path of ``merge_request``.
      - Cleaned up automatically when the future resolves/errors/cancels
        (done_callback removes it from ``_waiters``).
      - Consumed by β2 (``merge_cancel``) which looks up ``request_id``
        here to cancel the future explicitly.
    """

    request_id: str
    """Stable per-instance identity of the MergeRequest (e.g. 'mr-a1b2c3d4')."""
    future: asyncio.Future = field(repr=False)
    """The MergeRequest.result future — the one shielded from MCP disconnect."""
    source: str = 'mcp'
    """Origin of the waiter: 'mcp' (via merge_request tool) or 'workflow'."""
    submitted_tip: str | None = None
    """Git SHA of the branch tip at submit time (snapshot_tip), or None if unavailable."""


@dataclass
class MergeRequest:
    """A request to merge a task branch into main."""

    task_id: str
    branch: str  # e.g. "591" — without the task/ prefix
    worktree: Path
    pre_rebased: bool
    task_files: list[str] | None
    module_configs: list[ModuleConfig]
    config: OrchestratorConfig
    result: asyncio.Future[MergeOutcome] = field(repr=False)
    enqueued_at: float = field(default_factory=time.time, kw_only=True)
    merge_first_enqueued_at: float | None = field(default=None, kw_only=True)
    """Wall-clock epoch of the FIRST merge submission of this branch's lineage.

    Sourced from ``metadata.merge_first_enqueued_at`` (persisted, survives
    restart) and populated by :meth:`TaskWorkflow._stamp_first_merge_enqueue`
    at the per-task merge-submit chokepoint (workflow.py α-substrate, task 1886).

    Contrast with :attr:`enqueued_at`, which is re-stamped on every resubmission
    and held in-memory only (lost on restart).  ζ's aging comparator at
    ``_pop_next_pickable`` reads this field; ``None`` for legacy in-flight requests
    created before α was deployed (ζ owns the ``enqueued_at`` fallback for those).
    """
    request_id: str = field(
        default_factory=lambda: f'mr-{uuid.uuid4().hex[:8]}',
        kw_only=True,
    )
    """Stable per-instance identity for this merge request (e.g. 'mr-a1b2c3d4')."""
    snapshot_tip: str | None = field(default=None, kw_only=True)
    """Optional git ref / SHA of the snapshot tip used by α3 merge-status lookups."""
    generation: int = field(default=1, kw_only=True)
    """Generation counter for auto-chained merges (γ2).  Gen-1 is the original;
    each auto-chain increments by 1.  Bounded by MAX_AUTO_CHAINED_GENERATIONS."""
    lane: Literal['normal', 'high'] = field(default='normal', kw_only=True)
    """Priority lane for this request.  'high' requests are picked before all
    'normal' requests; within a lane FIFO order is preserved."""


@dataclass(frozen=True)
class TrainCallbacks:
    """Scheduler-backed callbacks for a single train, built in the harness.

    Holds async callables captured over a live scheduler + train_id so
    the merge worker (a pure git engine with no scheduler import) can flip
    member tasks done after a train advance.  Built by
    :func:`harness.build_train_callback_factory` and consumed by task γ when
    it constructs :class:`GroupMergeRequest` inside SpeculativeMergeWorker.
    """

    status_check: Callable[[list[str]], Awaitable[dict[str, str]]]
    """Async callback: given member_task_ids, return {task_id: status}."""

    mark_member_done: Callable[[str, str], Awaitable[None]]
    """Async callback: mark a single member task done with the merge SHA."""

    redrive_member: Callable[[str, bool, str | None], Awaitable[None]] | None = None
    """Async callback: re-drive an absorbed member that is still merge-deferred
    after a coalesce-train derail.  Signature: (mid, found_on_main, sha) -> None.
    found_on_main=True → mark done with found_on_main provenance (double-landing
    guard); False → flip to pending so the scheduler re-dispatches a solo merge.
    None when the callback is not available (back-compat with existing callers
    that build TrainCallbacks with only status_check/mark_member_done)."""


# Type alias for the factory that produces per-train callbacks.
# Called with a train_id str; returns a TrainCallbacks for that train.
TrainCallbackFactory = Callable[[str], TrainCallbacks]


# Type alias for the injectable merge-ready predicate (δ/1720 confidence gate).
# Called with a MergeRequest; returns an exclusion REASON string (truthy →
# exclude from train) or None (eligible).  Returning the reason (not a bool)
# lets the same value flow uniformly into the log line and the event
# data['exclusions'] entry, for both built-in and injected predicates.
MergeReadyPredicate = Callable[['MergeRequest'], 'str | None']


@dataclass
class GroupMergeRequest(MergeRequest):
    """A request to atomically merge a linear-stacked train of task branches.

    Extends :class:`MergeRequest` with train-specific fields and async
    callbacks.  The base ``task_id`` / ``branch`` / ``worktree`` are set to
    the TIP task's values so existing logging, event emission, and CAS
    bookkeeping all behave normally.

    The callbacks are populated by the enqueuer (δ₂) and capture the
    scheduler + train_id, keeping the merge worker a pure git engine with no
    direct scheduler dependency.
    """

    train_id: str
    """Unique identifier for this train (e.g. the scheduler's train UUID)."""

    member_task_ids: list[str]
    """Ordered list of task IDs from root to tip (inclusive)."""

    tip_branch: str
    """Branch name of the tip task (alias of ``branch``)."""

    tip_task_id: str
    """Task ID of the tip member (alias of ``task_id``)."""

    status_check: Callable[[list[str]], Awaitable[dict[str, str]]]
    """Async callback: given *member_task_ids*, return ``{task_id: status}``."""

    mark_member_done: Callable[[str, str], Awaitable[None]]
    """Async callback: mark a single member task done with the merge SHA."""

    redrive_member: Callable[[str, bool, str | None], Awaitable[None]] | None = field(
        default=None, kw_only=True,
    )
    """Optional async callback: re-drive an absorbed member still merge-deferred
    after a coalesce-train derail.  Signature: (mid, found_on_main, sha) -> None.
    None when not wired (back-compat with existing test constructions)."""


@dataclass
class MergeOutcome:
    """Result delivered to the caller via the Future."""

    status: Literal['done', 'conflict', 'blocked', 'already_merged', 'wip_halted', 'done_wip_recovery', 'wip_recovery_no_advance', 'unmerged_state', 'unknown_branch', 'superseded', 'error']
    reason: str = ''
    conflict_details: str = ''
    recovery_branch: str | None = None
    overlap_files: list[str] | None = None
    merge_sha: str | None = None
    push_status: str | None = None
    failure_diagnostic: dict[str, str] | None = None
    failure_category: str = ''
    """Structured post-merge VerifyResult category (e.g. 'gui_tsc') for the
    workflow merge-thrash signature.  Empty when no VerifyResult was produced."""
    failure_cause_hint: str = ''
    """Structured post-merge VerifyResult cause_hint for the workflow
    merge-thrash signature.  Empty when no VerifyResult was produced."""
    verify_skipped: bool = False
    """True when the disk guard fired and ``run_scoped_verification`` was never
    called.  Lets callers distinguish a disk-guard short-circuit from an actual
    verification failure in log messages."""
    superseded_by: str | None = None
    """request_id of the gen-(n+1) request that supersedes this one (γ2)."""
    dedupe_fingerprint: str = ''
    """Carries the preexisting-main-break dedupe fingerprint so the workflow
    can fold N concurrent failing merges into one parent escalation via
    ``submit_or_dedupe``.  Empty for all non-main-health outcomes."""


@dataclass
class SoloVerifyResult:
    """Result of verifying a single train member's un-stacked delta in isolation.

    Returned by :func:`reverify_member_solo`.  ``passed=True`` means the
    member's own delta passes the post-merge verify in a fresh solo worktree;
    ``passed=False`` means it failed (or the solo worktree could not be created
    / the rebase conflicted — treated as a failer by the attribution logic).

    ``merge_sha`` is the rebased tip SHA of the solo branch (used by
    ``advance_main`` when the member passes and needs to land); None on failure.

    ``solo_wt`` and ``solo_branch`` carry the isolated worktree path and branch
    name so ``_attribute_train_failure`` can call ``advance_main`` for passers
    without re-materialising the worktree (keeps the verify cost at ≤N+1).
    """
    member_id: str
    passed: bool
    merge_sha: str | None
    reason: str = ''
    solo_wt: Path | None = None
    solo_branch: str | None = None


@dataclass
class SpeculativeItem:
    """Internal message passed from Merger coroutine to Verifier coroutine.

    Holds everything the Verifier needs to run verification and CAS-advance
    main, or to immediately resolve a Future (for conflict/already_merged).
    """

    request: MergeRequest
    merge_result: MergeResult | None  # None means already_merged or conflict
    merge_wt: Path | None             # Merge worktree (if merge succeeded)
    base_sha: str                      # main SHA at merge time (actual or speculative)
    speculative: bool                  # True → merged against pending N's SHA
    skip_verify: bool                  # Retained but always False for single-task items (task-1724); trains still set True
    immediate_outcome: MergeOutcome | None = None  # Set for conflict/already_merged
    already_delivered: bool = False  # True → merger resolved req.result OOB; verifier skips set_result but still runs n_failed/slot bookkeeping
    started_monotonic: float | None = None  # time.monotonic() at entry; None → unset, _elapsed_ms returns None
    failure_diagnostic: dict[str, str] | None = None  # Populated on non-conflict merge failure
    merged_branch_tip: str | None = None  # γ2: branch HEAD rev-parsed by the merger; passed to _finalize_advanced_merge
    counts_against_cap: bool = False  # True for non-speculative, non-train successful merges (Mechanism 1)

    def __post_init__(self) -> None:
        """Enforce the I2 shape invariants (task 1990 / MQ-invariants ε).

        A SpeculativeItem is either REAL (a merge actually happened; carries
        merge_result + merge_wt) or DECIDED (conflict/already_merged/skip;
        carries immediate_outcome) — never both, never neither.  Structurally
        retires the task-1928 bug class where a hand-rebuilt item silently
        dropped a field into an inconsistent shape.
        """
        if (self.merge_result is None) == (self.immediate_outcome is None):
            raise ValueError(
                'SpeculativeItem requires exactly one of merge_result or '
                f'immediate_outcome to be set (REAL xor DECIDED); got '
                f'merge_result={self.merge_result!r}, immediate_outcome={self.immediate_outcome!r}',
            )
        if (self.merge_wt is not None) != (self.merge_result is not None):
            raise ValueError(
                'SpeculativeItem.merge_wt must be set iff merge_result is set; got '
                f'merge_wt={self.merge_wt!r}, merge_result={self.merge_result!r}',
            )
        if self.already_delivered and self.immediate_outcome is None:
            raise ValueError(
                'SpeculativeItem.already_delivered=True requires immediate_outcome to be set',
            )


class InflightStatus(StrEnum):
    """Sentinel status values for :class:`InflightEntry` / :class:`InflightVerifyResult`.

    A single str-compatible Enum (task 1990 / MQ-invariants ε) shared by both
    dataclasses' ``status`` fields — ``InflightEntry.status`` may carry any of
    the five members; ``InflightVerifyResult.status`` carries only the first
    three (DROPPED / REQUEUED / RUNNER_UNAVAILABLE).  Members are ``str``
    instances (mirrors ``event_store.EventType`` / ``verify_runner.DriftVerdict``),
    so every existing ``==`` / ``in`` comparison against the raw sentinel
    strings keeps working unchanged.
    """

    DROPPED = 'DROPPED'
    REQUEUED = 'REQUEUED'
    RUNNER_UNAVAILABLE = 'RUNNER_UNAVAILABLE'
    ABANDONED_PREDISPATCH = 'ABANDONED_PREDISPATCH'
    REQUEUED_PREDISPATCH = 'REQUEUED_PREDISPATCH'


@dataclass
class InflightEntry:
    """An in-flight verify entry held in SpeculativeMergeWorker._inflight deque.

    One entry per item that has been dispatched to a host and has a background
    asyncio verify task running (or a passthrough/sentinel that needs serial
    finalization in submission order).

    Ordering invariant
    ------------------
    The deque head is always finalized before the next entry is processed.
    This guarantees that main is advanced in SUBMISSION ORDER regardless of
    which background verify task finishes first.

    This invariant covers main-advancement ordering; it does NOT guarantee
    that result-Future delivery is strictly ordered across item types.
    Passthrough entries (verify_task=None, immediate_outcome set) are finalized
    INLINE during DISPATCH-FILL — meaning a later passthrough can resolve its
    Future before an earlier real-verify entry resolves its Future.  Because
    passthroughs never advance main (they are conflict / already_merged / skip),
    this does not violate the main-advancement order guarantee.  No known
    consumer depends on strict cross-item Future-resolution ordering.

    Fields
    ------
    item           : the SpeculativeItem being verified
    lease          : the HostLease held for this verify (None for passthroughs)
    verify_task    : the asyncio.Task wrapping _run_inflight_verify (None for passthroughs)
    merge_wt       : the merge worktree path (may have been warm-swapped by _run_inflight_verify)
    was_speculative: True if item.speculative was True at dispatch time (for slot release)
    phase          : current phase string for snapshot() observability (per-entry source of
                     truth for multi-host; _verify_phase is the single-host compat field)
    passthrough_outcome: set for immediate-outcome entries (conflict/already_merged/skip_verify)
                         that are enqueued without a real verify task so finalize can deliver
                         them in submission order
    verify_result  : set when the verify has completed (pass=None; fail=VerifyResult)
    status         : optional sentinel string ('DROPPED', 'REQUEUED', 'RUNNER_UNAVAILABLE')
                     returned by _run_inflight_verify to signal special handling by _finalize_inflight
    """

    item: SpeculativeItem
    lease: Any | None                       # HostLease | None
    verify_task: asyncio.Task | None        # type: ignore[type-arg]
    merge_wt: Path | None
    was_speculative: bool
    phase: str
    passthrough_outcome: MergeOutcome | None = None
    verify_result: VerifyResult | None = None  # None = pass; VerifyResult = fail/skip
    status: InflightStatus | None = None    # sentinel: DROPPED / REQUEUED / RUNNER_UNAVAILABLE / ABANDONED_PREDISPATCH / REQUEUED_PREDISPATCH
    started_at: float | None = None         # time.time() at dispatch construction (≈ verify start)

    def __post_init__(self) -> None:
        """Enforce the I2-shadow invariant (task 1990 / MQ-invariants ε).

        A passthrough entry (immediate-outcome delivery, no real verify) must
        wrap a DECIDED item — i.e. passthrough_outcome is set only when the
        wrapped item's own immediate_outcome is also set.
        """
        if self.passthrough_outcome is not None and self.item.immediate_outcome is None:
            raise ValueError(
                'InflightEntry.passthrough_outcome requires item.immediate_outcome '
                f'to be set; got passthrough_outcome={self.passthrough_outcome!r} on '
                f'an item with immediate_outcome=None (merge_result={self.item.merge_result!r})',
            )


@dataclass
class _HostUnavailability:
    """Per-host RunnerUnavailable streak tracker entry (task 1795).

    Tracks consecutive failures so the worker can fire a dedup'd escalation
    once the streak threshold is reached, and can clear state on recovery.

    Fields
    ------
    streak              : consecutive RunnerUnavailable count for this host.
    first_unavailable_at: time.time() of the first failure in this episode.
    reason              : str(exc) from the most recent RunnerUnavailable.
    """

    streak: int
    first_unavailable_at: float
    reason: str


@dataclass
class InflightVerifyResult:
    """Result returned by SpeculativeMergeWorker._run_inflight_verify.

    Fields
    ------
    outcome     : None if verification passed; MergeOutcome if it failed/was skipped.
    merge_wt    : the (possibly warm-swapped) merge worktree path; may be None if
                  the verify was aborted/dropped before starting.
    warm_results: dict[str, bool] of per-test results from warm verify (for shadow compare);
                  empty dict if the warm path was not taken.
    status      : None on normal completion; sentinel string for special cases:
                  'DROPPED'           — sole-waiter abandoned; merge_wt cleaned
                  'REQUEUED'          — operator halt; req re-queued on _queue
                  'RUNNER_UNAVAILABLE' — remote runner raised RunnerUnavailable;
                                        merge_wt NOT cleaned (will be re-dispatched)
    reason      : str(exc) from the RunnerUnavailable exception when status is
                  'RUNNER_UNAVAILABLE'; None on all other paths.  Used by the
                  unavailability tracker + alarm to name the actual failure cause
                  in escalation summaries.
    """

    outcome: MergeOutcome | None
    merge_wt: Path | None
    warm_results: dict[str, str] = dataclasses.field(default_factory=dict)
    status: InflightStatus | None = None  # None | 'DROPPED' | 'REQUEUED' | 'RUNNER_UNAVAILABLE'
    reason: str | None = None  # str(RunnerUnavailable exc) when status='RUNNER_UNAVAILABLE'
    spec_warm: bool = False   # True when merge_wt is a warm _spec- lane (not an ephemeral wt)
