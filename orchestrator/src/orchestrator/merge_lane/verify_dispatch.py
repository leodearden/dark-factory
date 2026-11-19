"""The verify dispatcher: the lane's one seam for running a verify.

:class:`VerifyDispatcher` launches an in-flight verify on a held lease
(``orchestrator/src/orchestrator/merge_lane/inflight_verify.py::InflightVerifyRun``),
binds every post-merge verify to the lane's shared context, and runs a parked
gate's history leg. It owns the verify tuning, the verify-attempt and
dead-verify-abort counts, and nothing else; the concerns it drives but does
not own are reached through the ports in
``orchestrator/src/orchestrator/merge_lane/dispatch_ports.py``.
"""
from __future__ import annotations

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from orchestrator.git_ops import GitOps
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
from orchestrator.merge_lane.post_merge_verify import run_post_merge_verify
from orchestrator.merge_lane.types import (
    ChainResult,
    InflightEntrySlot,
    InflightVerifyResult,
    MergeOutcome,
    MergeRequest,
    RealMergeItem,
    VerifyWorktreeHandle,
)
from orchestrator.merge_pool_facts import with_chain_members
from orchestrator.merge_verify_facts import (
    BaseRelation,
    MergeVerifyDispatchFacts,
    RunnerReason,
)

GATE_HISTORY_LEG_FACTS = MergeVerifyDispatchFacts(
    runner_reason=RunnerReason.LOCAL_ONLY_RULE, base_relation=BaseRelation.MAIN,
)
"""A gate's history leg verifies the head rebased onto main, on the local-only runner."""


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
        clock: ClockPort,
        verifier: VerifyPort,
        halt: HaltState,
        lane_context: LaneVerifyContext,
        custody: DispatchCustody,
        speculation: DispatchSpeculation,
        worktrees: DispatchWorktrees,
        contention: LeaseContention,
    ) -> None:
        self._verifier = verifier
        self._lane_context = lane_context
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
