"""The verify dispatcher: the lane's one seam for running a verify.

:class:`VerifyDispatcher` dispatches a merged item onto a free verify host
and launches its in-flight verify on the held lease
(``orchestrator/src/orchestrator/merge_lane/inflight_verify.py::InflightVerifyRun``),
binds every post-merge verify to the lane's shared context, and runs a parked
gate's history leg. It owns the verify tuning, the verify-attempt and
dead-verify-abort counts and the chain-invalidation flag, and nothing else;
the concerns it drives but does not own are reached through the ports in
``orchestrator/src/orchestrator/merge_lane/dispatch_ports.py``.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, ClassVar

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.dead_base_ledger import DeadBaseSite, VoidPoint
from orchestrator.merge_lane.dispatch_ports import (
    DispatchCustody,
    DispatchSpeculation,
    DispatchWorktrees,
    LaneVerifyContext,
    LeaseContention,
)
from orchestrator.merge_lane.halt import HaltState
from orchestrator.merge_lane.inflight_verify import (
    InflightVerifyCall,
    InflightVerifyCounters,
    InflightVerifyLane,
    InflightVerifyRun,
)
from orchestrator.merge_lane.ports import ClockPort, VerifyPort
from orchestrator.merge_lane.post_merge_verify import (
    placeholder_local_runner,
    run_post_merge_verify,
)
from orchestrator.merge_lane.remerge_base import RemergeClaim
from orchestrator.merge_lane.types import (
    ChainResult,
    DecidedItem,
    GroupMergeRequest,
    InflightEntry,
    InflightEntrySlot,
    InflightStatus,
    InflightVerifyResult,
    ItemLifecycleState,
    MergeOutcome,
    MergeRequest,
    NoHost,
    RealMergeItem,
    RemergeBase,
    SpeculativeItem,
    VerifyBaseDivergence,
    VerifyWorktreeHandle,
    item_merge_wt,
)
from orchestrator.merge_pool_facts import with_chain_members
from orchestrator.merge_verify_facts import (
    BaseRelation,
    MergeVerifyDispatchFacts,
    RunnerReason,
    classify_base_relation,
)
from orchestrator.verify_runner import HostAllocator, HostLease, LocalRunner

logger = logging.getLogger('orchestrator.merge_lane')

GATE_HISTORY_LEG_FACTS = MergeVerifyDispatchFacts(
    runner_reason=RunnerReason.LOCAL_ONLY_RULE, base_relation=BaseRelation.MAIN,
)
"""A gate's history leg verifies the head rebased onto main, on the local-only runner."""


@dataclasses.dataclass(frozen=True)
class _DeadBase:
    """INV-3 at dispatch: the item's base is a known-dead commit, *dead_link*."""

    dead_link: str
    site: ClassVar[DeadBaseSite] = DeadBaseSite.DISPATCH_VOID

    def announce(
        self, speculation: DispatchSpeculation, item: RealMergeItem, anchor: RemergeBase,
    ) -> None:
        speculation.emit_dead_base_discard(
            item.request, item, self.dead_link, VoidPoint.DISPATCH, anchor,
        )
        logger.info(
            'Task %s: dead-base straggler at dispatch (dead link %s) — '
            're-merging (%s) instead of burning a verify',
            item.request.task_id, self.dead_link, anchor.reason.value,
        )


@dataclasses.dataclass(frozen=True)
class _StaleMerge:
    """Mechanism 2: the item was merged onto a base that is no longer the one to verify on."""

    reason: str
    site: ClassVar[DeadBaseSite] = DeadBaseSite.STALE_DISCARD

    def announce(
        self, speculation: DispatchSpeculation, item: RealMergeItem, anchor: RemergeBase,
    ) -> None:
        speculation.emit_speculative(
            EventType.speculative_discard, item.request.task_id,
            reason=self.reason, **anchor.to_wire(),
        )
        logger.info(
            'Task %s: discarding stale merge (%s), re-merging (%s)',
            item.request.task_id, self.reason, anchor.reason.value,
        )


@dataclasses.dataclass(frozen=True)
class _Placement:
    """Where one held-lease dispatch verifies: its facts, labels and any deep chain."""

    facts: MergeVerifyDispatchFacts
    depth: int
    probe_base: str | None
    chain: ChainResult | None
    divergence: VerifyBaseDivergence | None


def _passthrough(
    item: DecidedItem, was_speculative: bool, started_at: float | None = None,
) -> InflightEntry:
    return InflightEntry(
        item=item, lease=None, verify_task=None, merge_wt=None,
        was_speculative=was_speculative, passthrough_outcome=item.immediate_outcome,
        started_at=started_at, permit=item.permit,
    )


def _predispatch(item: SpeculativeItem, status: InflightStatus) -> InflightEntry:
    return InflightEntry(
        item=item, lease=None, verify_task=None, merge_wt=None,
        was_speculative=item.speculative, status=status, permit=item.permit,
    )


class VerifyDispatcher:
    """Runs the lane's verifies and owns their tuning and counts.

    The tuning is class-level with per-instance overrides read live, so a
    test sets a small value on the instance the worker exposes as
    ``verify_dispatch``.
    """

    VERIFY_ABANDON_POLL_SECS: float = 10.0
    """How often an in-flight verify, and the verifier loop's waits, re-check
    for an abandoned waiter or an operator halt; negligible over a verify."""

    INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS: float = 5400.0
    """No-progress budget before an in-flight verify is aborted (task 2420).

    Progress is a live remote dispatch, CPU in the verify's process tree, or a
    newer content mtime under the worktree, on both lease kinds (tasks 2566,
    5290). INVARIANT: this exceeds the longest per-command verify timeout a
    target can authorise (3600s on dark-factory), since a command inside its
    own sanctioned runtime is healthy by definition and the CPU signal can be
    absent; raise it if any per-command budget passes 5400s (incident: task
    4659, 2e58a8fc28). A genuinely hung verify holds its host
    budget x MAX_INFLIGHT_DEAD_VERIFY_ABORTS = 4.5h before resolving blocked.
    """

    INFLIGHT_VERIFY_PROGRESS_PROBE_SECS: float = 60.0
    """Throttle on the progress probes, independent of the abort-poll cadence.
    A probe reads the tree's CPU fingerprint and walks the worktree for its
    newest content mtime only while that is silent, so the default assumes
    thousands, not millions, of entries outside ``.git``."""

    MAX_INFLIGHT_DEAD_VERIFY_ABORTS: int = 3
    """Consecutive no-progress aborts of one task before it resolves 'blocked'
    instead of re-queuing a deterministically hanging verify forever."""

    CONTENDED_LEASE_REQUEUE_WARN_STREAK: int = 5
    """Consecutive contended-lane defers at which the per-defer WARNING is
    raised once to an ERROR (task 2828): ~25 min of continuous contention,
    past any short legitimate holder."""

    CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS: float = 30.0
    """Minimum period between contended-lane defers of one task (task 3003).

    The sleep is ``max(0, period - exc.wait_secs)``, so the bounded-wait
    raisers (reset 30s, lease 300s) add nothing; it exists for
    MergeVerifyLeaseHeld, whose immediate refusal would otherwise spin
    merge -> refuse -> requeue at git speed for the holder's whole window.
    0.0 opts out.
    """

    MAX_CONTENDED_LEASE_DEFER_SECS: float = 14400.0
    """Elapsed-time cap on an unbroken contended-lane defer streak, after which
    the defer resolves 'blocked' (task 3003). It must stay above the longest
    legitimate holder (a 1-2h merge-ahead train), and is elapsed time rather
    than a count because each raiser re-attempts at a different period."""

    ORPHAN_REAP_TIMEOUT_SECS: float = 5.0
    """Bound on an in-flight verify's orphan-guard reap (task 3043): both
    external cancellers await the outer verify, so an unbounded reap against
    an unreachable host would stall shutdown or the head-failure cascade."""

    def __init__(
        self,
        *,
        git_ops: Callable[[], GitOps],
        host_allocator: Callable[[OrchestratorConfig], HostAllocator],
        clock: ClockPort,
        verifier: VerifyPort,
        halt: HaltState,
        lane_context: LaneVerifyContext,
        custody: DispatchCustody,
        speculation: DispatchSpeculation,
        worktrees: DispatchWorktrees,
        contention: LeaseContention,
    ) -> None:
        self._git_ops = git_ops
        self._host_allocator = host_allocator
        self._clock = clock
        self._verifier = verifier
        self._halt = halt
        self._lane_context = lane_context
        self._custody = custody
        self._speculation = speculation
        self._worktrees = worktrees
        self._chain_invalidated = False
        self._counters = InflightVerifyCounters()
        self._inflight_lane = InflightVerifyLane(
            git_ops=git_ops,
            clock=clock,
            halt=halt,
            tuning=self,
            counters=self._counters,
            custody=custody,
            speculation=speculation,
            worktrees=worktrees,
            contention=contention,
            event_store=lane_context.event_store,
            escalation_queue=lane_context.escalation_queue,
            retry_budgets=lane_context.retry_budgets,
            post_merge_verify=lambda *args, **kwargs: self.post_merge_verify(*args, **kwargs),
        )

    def note_chain_invalidated(self) -> None:
        """A head failure tore down its followers: the next speculative dispatch re-merges."""
        self._chain_invalidated = True

    def chain_invalidated(self) -> bool:
        """Whether the next speculative dispatch re-merges as ``chain_invalidated``.

        Raised by :meth:`note_chain_invalidated` and by a dispatch that
        re-merged; cleared by any other dispatch that reaches a verdict path.
        """
        return self._chain_invalidated

    def verify_attempts(self) -> int:
        """Local verify attempts that reached a verify, over the lane's lifetime."""
        return self._counters.verify_attempts

    def dead_verify_aborts(self, task_id: str) -> int:
        """*task_id*'s consecutive no-progress aborts so far."""
        return self._counters.dead_verify_aborts(task_id)

    async def run_inflight_verify(
        self,
        item: RealMergeItem,
        lease: Any,  # HostLease
        depth: int | None = None,
        probe_base: str | None = None,
        *,
        chain_items: int = 1,
        chain: ChainResult | None = None,
        verify_wt: VerifyWorktreeHandle | None = None,
        entry_slot: InflightEntrySlot | None = None,
        dispatch_facts: MergeVerifyDispatchFacts | None = None,
    ) -> InflightVerifyResult:
        """Verify *item* on *lease*; see :class:`InflightVerifyCall` for the arguments.

        The positional arity stays four so the ``(item, lease, depth=None,
        probe_base=None, **_kw)`` stand-ins across the suite stay compatible.
        A local lease swaps onto the warm lane and counts the attempt; a
        remote one dispatches on ``lease.runner``. A RunnerUnavailable benches
        the host and returns RUNNER_UNAVAILABLE for the caller to re-dispatch.
        """
        call = InflightVerifyCall(
            item=item, lease=lease, depth=depth, probe_base=probe_base,
            chain_items=chain_items, chain=chain, verify_wt=verify_wt,
            entry_slot=entry_slot, dispatch_facts=dispatch_facts,
        )
        return await InflightVerifyRun(self._inflight_lane, call).run()

    def post_merge_verify(
        self, git_ops: GitOps, req: MergeRequest, merge_wt: Path, **per_call: Any,
    ) -> Coroutine[Any, Any, MergeOutcome | None]:
        """The single place that decides which lane handles a single-branch merge verify inherits."""
        context = self._lane_context
        budgets = context.retry_budgets()
        return run_post_merge_verify(
            git_ops, req, merge_wt,
            max_narrowed=budgets.max_narrowed,
            narrowed_retries=budgets.narrowed_retries,
            event_store=context.event_store(),
            quarantine_divergent=context.quarantine_divergent,
            escalation_queue=context.escalation_queue(),
            halt_hook=context.halt_hook,
            dry_run_handles=context.dry_run_handles,
            main_health_probe_handles=context.main_health_probe(),
            verifier=self._verifier,
            **per_call,
        )

    async def gate_history_leg(
        self, git_ops: GitOps, req: MergeRequest, merge_wt: Path, **gate_kwargs: Any,
    ) -> MergeOutcome | None:
        """Run a parked gate's history leg, the only verify a single-branch gate runs.

        A full re-verify leaves the gate as a dispatch of the rebased head
        (task 8525), so what a parked gate still verifies is a declared
        history leg on a same tree. Its runner is local-only, so it runs on
        local whatever host the head holds, and acquires no slot.
        """
        return await self.post_merge_verify(
            git_ops, req, merge_wt,
            dispatch_facts=with_chain_members(GATE_HISTORY_LEG_FACTS, req, None), **gate_kwargs,
        )

    async def dispatch(self, item: SpeculativeItem) -> InflightEntry | NoHost:
        """Dispatch *item* to a verify host.

        An :class:`InflightEntry` for the caller's frontier, which is a
        passthrough with no lease for an abandoned, halted or decided item, or
        :class:`NoHost` when no slot is free, holding the item this dispatch
        produced (a dispatch-time re-merge's replacement when there was one)
        for the redispatch park. No return carries a cap permit.
        """
        item = self._release_cap_permit(item)
        req = item.request
        if self._custody.request_abandoned(req):
            return await self._abandon_predispatch(item)
        halted = self._halt.is_operator_halted or self._custody.admission_halted()
        if halted and isinstance(item, RealMergeItem):
            return await self._requeue_predispatch(item)
        if isinstance(item, DecidedItem):
            self._chain_invalidated = False
            return _passthrough(item, item.speculative)
        allocator = self._host_allocator(req.config)
        if allocator.free_host_count() == 0:
            return NoHost(item)
        return await self._dispatch_on_free_host(item, allocator)

    def _release_cap_permit(self, item: SpeculativeItem) -> SpeculativeItem:
        """Mechanism 1: hand the merger-ahead cap back on drain, before any branching."""
        if isinstance(item, RealMergeItem) and item.cap_permit is not None:
            self._speculation.release_cap_permit(item.cap_permit)
            return dataclasses.replace(item, cap_permit=None)
        return item

    async def _abandon_predispatch(self, item: SpeculativeItem) -> InflightEntry:
        """The waiter gave up before dispatch: no Future resolves, so retirement is the exit."""
        abandoned_wt = item_merge_wt(item)
        if abandoned_wt is not None:
            with contextlib.suppress(BaseException):
                await self._worktrees.cleanup_owned(abandoned_wt)
        self._chain_invalidated = False
        self._custody.record_unlanded_rebased_head(item)
        self._custody.retire_item(item.request.request_id)
        return _predispatch(item, InflightStatus.ABANDONED_PREDISPATCH)

    async def _requeue_predispatch(self, item: RealMergeItem) -> InflightEntry:
        """A halt holds the built item back; its Future stays pending for finalize to skip."""
        with contextlib.suppress(BaseException):
            await self._worktrees.cleanup_owned(item.merge_wt)
        self._custody.requeue_request(item.request)
        self._chain_invalidated = False
        return _predispatch(item, InflightStatus.REQUEUED_PREDISPATCH)

    async def _dispatch_on_free_host(
        self, item: RealMergeItem, allocator: HostAllocator,
    ) -> InflightEntry | NoHost:
        """Re-merge a stale or dead-based item first, then verify what is current.

        The entry keeps the ORIGINAL speculative flag; the permit follows the
        item, so a replacement carries the permit its claim decided.
        """
        was_speculative = item.speculative
        in_verify = self._custody.has_inflight_verify()
        main_sha = await self._dispatch_main_sha(item, in_verify)
        cause = self._remerge_cause(item, main_sha, in_verify)
        if cause is not None:
            replacement = await self._remerge_stale(item, main_sha, cause)
            if isinstance(replacement, DecidedItem):
                self._chain_invalidated = True
                return _passthrough(replacement, was_speculative, self._clock.now())
            item = replacement
        self._chain_invalidated = cause is not None
        try:
            return await self._launch(item, allocator, was_speculative)
        except BaseException:
            if cause is not None:
                await self._discard_unlaunched_replacement(item)
            raise

    async def _dispatch_main_sha(self, item: RealMergeItem, in_verify: bool) -> str:
        """Main, read only when a consumer needs it; '' fails the checks open.

        The dead-base check needs it only with a non-empty ledger, and
        Mechanism 2 only for a non-speculative, non-group item with no verify
        in flight, so the steady speculative path reads no main at all.
        """
        needs_main = self._speculation.has_dead_bases() or (
            not in_verify and not item.speculative
            and not isinstance(item.request, GroupMergeRequest)
        )
        if not needs_main:
            return ''
        return await self._speculation.main_sha_or_empty('dispatch chain check')

    def _remerge_cause(
        self, item: RealMergeItem, main_sha: str, in_verify: bool,
    ) -> _DeadBase | _StaleMerge | None:
        """Why *item* must be re-merged before it is verified, if it must.

        INV-3 (plans/merge-verdict-integrity-prd.md §3.3) is checked
        regardless of in-flight verifies: a straggler built on a commit
        already re-merged away would burn a whole verify (the 5260 gap).
        Mechanism 2 waits for an empty verify frontier, since a speculative
        item stacked on an in-flight predecessor is the head-failure
        cascade's to re-merge.
        """
        dead_link = self._speculation.chain_dead_link(item, main_sha) if main_sha else None
        if dead_link is not None:
            return _DeadBase(dead_link)
        if in_verify:
            return None
        reason = self._stale_merge_reason(item, main_sha)
        return None if reason is None else _StaleMerge(reason)

    def _stale_merge_reason(self, item: RealMergeItem, main_sha: str) -> str | None:
        """Mechanism 2's reason, or None.

        A speculative item's base is its predecessor's commit by design, so
        main moving never makes it stale (task 1862 step-4): only a failed
        predecessor or an invalidated chain does. A failed main read skips
        the ``main_advanced`` check rather than comparing against ''.
        """
        if item.speculative:
            if self._speculation.previous_failed():
                return 'previous_failed'
            return 'chain_invalidated' if self._chain_invalidated else None
        if isinstance(item.request, GroupMergeRequest) or not main_sha:
            return None
        return 'main_advanced' if item.base_sha != main_sha else None

    async def _remerge_stale(
        self, item: RealMergeItem, main_sha: str, cause: _DeadBase | _StaleMerge,
    ) -> SpeculativeItem:
        """Discard *item*'s build and re-merge it onto its claimed base.

        The registry holds the request at MERGING for the re-merge window.
        The discarded build's own commit is recorded dead, so a straggler
        stacked on it is caught by this same dispatch-time check.
        """
        req = item.request
        self._custody.note_transition(
            req.request_id, ItemLifecycleState.DISPATCHING, ItemLifecycleState.MERGING, req,
        )
        await self._worktrees.cleanup_owned(item.merge_wt)
        claim = self._speculation.claim_remerge_base(item, main_sha, item.permit)
        cause.announce(self._speculation, item, claim.anchor)
        self._speculation.record_item_commit_dead(item, cause.site, False)
        replacement = await self._remerge_at_dispatch(item, claim)
        self._custody.note_transition(
            req.request_id, ItemLifecycleState.MERGING, ItemLifecycleState.DISPATCHING,
            replacement,
        )
        return replacement

    async def _remerge_at_dispatch(
        self, item: RealMergeItem, claim: RemergeClaim,
    ) -> SpeculativeItem:
        """Re-merge *item* onto *claim* and tell the merger, which may hold
        *item*'s commit (D2): it re-points at the replacement, or at the live
        tip when the re-merge raises."""
        try:
            replacement = await self._speculation.remerge(
                item.request, item.started_monotonic, claim,
            )
        except BaseException:
            self._speculation.notify_merger_of_dead((item.merge_result.merge_commit or '',))
            raise
        self._speculation.reanchor_merger_on(item, replacement)
        return replacement

    async def _discard_unlaunched_replacement(self, replacement: RealMergeItem) -> None:
        """Free the permit and worktree of a re-merge's replacement no entry
        ever held, and record its commit dead so a merger re-pointed onto it moves off."""
        if replacement.permit is not None:
            with contextlib.suppress(BaseException):
                self._speculation.release_permit(replacement.permit)
        with contextlib.suppress(BaseException):
            await self._worktrees.cleanup_owned(replacement.merge_wt)
        self._speculation.record_item_commit_dead(replacement, DeadBaseSite.DISPATCH_VOID, True)

    async def _launch(
        self, item: RealMergeItem, allocator: HostAllocator, was_speculative: bool,
    ) -> InflightEntry | NoHost:
        """Acquire a host for *item* and launch its verify; :class:`NoHost` when
        the slot seen free was taken meanwhile (the drift check takes slots too).

        LEASE/PERMIT LEAK GUARD (task 3185): from the acquire until the verify
        task exists, the lease and the speculation permit are held by this
        frame alone, across a chain build that lets an external cancel
        propagate. Any raise hands both back (``release``, not
        ``cancel_and_release``: nothing is dispatched on the lease yet) and
        re-raises unchanged. A ``return`` in that window would leak the lease.
        """
        lease = await allocator.acquire(
            self._local_runner_factory(item), policy=item.request.config.verify_host_policy,
        )
        if lease is None:
            return NoHost(item)
        try:
            placement = await self._place(item, lease)
        except BaseException:
            with contextlib.suppress(BaseException):
                await allocator.release(lease)
            if item.permit is not None:
                with contextlib.suppress(BaseException):
                    self._speculation.release_permit(item.permit)
            raise
        return self._start_verify(item, lease, placement, was_speculative)

    def _local_runner_factory(self, item: RealMergeItem) -> Callable[[], LocalRunner]:
        """The local runner, built only when the local host is selected; the
        in-flight verify warm-swaps its worktree."""
        return lambda: placeholder_local_runner(item.merge_wt, item.request, self._verifier)

    async def _place(self, item: RealMergeItem, lease: HostLease) -> _Placement:
        """Decide this held-lease dispatch's facts, depth label and deep chain.

        A firing probe relabels only the depth and attributed base, never the
        tree verified (``worker.py::ProbePlacement``). A built chain holds a
        scratch lane the verify's ``finally`` releases, so the chain is the
        last thing bound before the verify task exists; a cancel inside the
        build is released by the builder, before any chain reaches this frame.
        """
        divergence, guard_main = await self._observe_verify_base(item)
        stacked_on, rebased_heads = self._speculation.frontier_base_commits()
        facts = MergeVerifyDispatchFacts(
            runner_reason=lease.reason,
            base_relation=classify_base_relation(
                item.base_sha, main_sha=guard_main,
                stacked_on=stacked_on, rebased_heads=rebased_heads,
            ),
        )
        probe = self._speculation.probe_placement(item)
        depth = probe.depth if probe is not None else self._speculation.frontier_depth()
        deep = await self._speculation.deep_chain_placement(item, depth, lease.is_local)
        return _Placement(
            facts=facts.with_chain_depth_reason(deep.depth_reason),
            depth=depth,
            probe_base=probe.base if probe is not None else None,
            chain=deep.chain,
            divergence=divergence,
        )

    async def _observe_verify_base(
        self, item: RealMergeItem,
    ) -> tuple[VerifyBaseDivergence | None, str | None]:
        """The PRD §5.3 frozen-tip detector, fail-open on any git error.

        A detector, never a refusal (esc-3206-3): INV-3 enforces soundness at
        adoption. Making the detector raise would be swallowed here, and a
        refusal returning from this window would leak the lease.
        """
        divergence: VerifyBaseDivergence | None = None
        guard_main: str | None = None
        with contextlib.suppress(Exception):
            guard_main = await self._git_ops().get_main_sha()
            self._speculation.note_main_sha(guard_main)
            divergence = self._speculation.verify_base_divergence(item, guard_main)
        return divergence, guard_main

    def _start_verify(
        self, item: RealMergeItem, lease: HostLease, placement: _Placement,
        was_speculative: bool,
    ) -> InflightEntry:
        """Start the verify task and build the entry that owns it.

        ``chain_items`` is 1 on both arms: the deep arm counts its own links,
        and the frontier is the base, not the chain (``chain_items >= 2`` is
        the deep-verify discriminator). The entry's ``merge_wt`` is the item's
        own ephemeral even on the chain arm, where the verify disposes of it.
        """
        verify_wt = VerifyWorktreeHandle(merge_wt=item.merge_wt, spec_warm=False)
        entry_slot = InflightEntrySlot()
        verify_task = asyncio.ensure_future(
            self._custody.release_at_verdict(
                functools.partial(
                    self.run_inflight_verify,
                    item, lease, depth=placement.depth, probe_base=placement.probe_base,
                    chain_items=1, chain=placement.chain, verify_wt=verify_wt,
                    entry_slot=entry_slot, dispatch_facts=placement.facts,
                ),
                entry_slot,
            ),
        )
        entry = InflightEntry(
            item=item,
            lease=lease,
            verify_task=verify_task,
            merge_wt=item.merge_wt,
            was_speculative=was_speculative,
            started_at=self._clock.now(),
            permit=item.permit,
            chain=placement.chain,
            verify_wt=verify_wt,
            verify_base_divergence=placement.divergence,
        )
        entry_slot.entry = entry
        return entry
