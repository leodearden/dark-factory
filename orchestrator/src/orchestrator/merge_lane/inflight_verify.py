"""One dispatched item's verify, run on the host lease it already holds.

An :class:`InflightVerifyRun` lives for exactly one verify. It resolves what
is verified and where (the item's own merge commit, or a deep chain's tip in
the chain's pooled lane), swaps a local verify onto the warm merge-verify
lane, launches the post-merge verify and polls it for a verdict or for one of
the three ways it is abandoned mid-run (a waiter that gave up, an operator
halt, no verify progress for a full budget). It turns a remote host going
away, a contended merge-verify lane and any other error into their results,
reaps the inner verify task on every exit, and settles the verdict: a chain
tip adopts or requeues, a single verify passes, fails or is skipped.

It is the VERIFY half of an in-flight entry; the CAS half is the worker's
finalize. It resolves ``req.result`` itself only on a terminal error, so the
item never stalls the queue.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, MergeVerifyLeaseContended, MergeVerifyLeaseHeld, _run
from orchestrator.merge_lane.dispatch_ports import (
    ChainPlanning,
    DispatchCustody,
    DispatchWorktrees,
    LeaseContention,
    VerifyRetryBudgets,
)
from orchestrator.merge_lane.halt import HaltState
from orchestrator.merge_lane.liveness import (
    VerifyProgressWatch,
    _acquire_warm_verify_worktree,
    _safety_valve_due,
    release_chain_build_lane,
)
from orchestrator.merge_lane.no_progress_abort import NoProgressAbort, emit_no_progress_abort
from orchestrator.merge_lane.ports import ClockPort
from orchestrator.merge_lane.shadow import (
    _alarm_warm_shadow_unparseable,
    build_warm_shadow_results,
)
from orchestrator.merge_lane.types import (
    ChainResult,
    InflightEntrySlot,
    InflightStatus,
    InflightVerifyResult,
    MergeOutcome,
    RealMergeItem,
    VerifyBaseFacts,
    VerifyWorktreeHandle,
    remote_dispatch_live,
)
from orchestrator.merge_lane.verify_cancellation import VerifyCancel, VerifyCancelCause
from orchestrator.merge_pool_facts import with_chain_members
from orchestrator.merge_verify_facts import MergeVerifyDispatchFacts
from orchestrator.verify import VerifyResult
from orchestrator.verify_cancel import elide_middle
from orchestrator.verify_runner import HostLease, RunnerUnavailable

logger = logging.getLogger('orchestrator.merge_lane')


async def resolve_dispatch_time_merge_base(
    repo_root: Path,
    main_sha: str,
    branch_tip: str | None,
) -> str | None:
    """Best-effort ``git merge-base(main_sha, branch_tip)`` for the
    merge-skew classifier's dispatch-time base facts (task 2383 β, 2357).

    Both *main_sha* and *branch_tip* are caller-supplied frozen values —
    *main_sha* is the production caller's ``item.base_sha`` (never a fresh
    ``get_main_sha()`` re-read) and *branch_tip* is ``item.merged_branch_tip``
    (never a fresh branch-ref re-read) — this helper only computes their
    common ancestor via a read-only ``git merge-base`` call.

    Returns ``None`` — degrading classification to INDETERMINATE via
    ``run_post_merge_verify``'s None-default (I3, fail-open) — when
    *branch_tip* is unavailable or the git call fails for any reason
    (non-git *repo_root*, unresolvable SHA, subprocess error).
    """
    if not branch_tip:
        return None
    try:
        rc, out, err = await _run(
            ['git', 'merge-base', main_sha, branch_tip], cwd=repo_root,
        )
        if rc != 0:
            logger.warning(
                'resolve_dispatch_time_merge_base: git merge-base failed '
                '(rc=%d) for main_sha=%s branch_tip=%s: %s; degrading to '
                'None (fail-safe, I3)',
                rc, main_sha, branch_tip, err.strip(),
            )
            return None
        return out.strip() or None
    except Exception:
        logger.warning(
            'resolve_dispatch_time_merge_base: git merge-base raised for '
            'main_sha=%s branch_tip=%s; degrading to None (fail-safe, I3)',
            main_sha, branch_tip,
            exc_info=True,
        )
        return None


def _slot_cancel(
    cause: VerifyCancelCause, entry_slot: InflightEntrySlot | None, lease: HostLease | None,
) -> VerifyCancel:
    """The cancel of the verify *entry_slot* wraps, running on *lease*."""
    return VerifyCancel.of(cause, entry_slot.entry if entry_slot is not None else None, lease)


class InflightVerifyTuning(Protocol):
    """The verify tuning an in-flight verify reads, live, from its dispatcher."""

    VERIFY_ABANDON_POLL_SECS: float
    INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS: float
    INFLIGHT_VERIFY_PROGRESS_PROBE_SECS: float
    MAX_INFLIGHT_DEAD_VERIFY_ABORTS: int
    CONTENDED_LEASE_REQUEUE_WARN_STREAK: int
    CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS: float
    MAX_CONTENDED_LEASE_DEFER_SECS: float
    ORPHAN_REAP_TIMEOUT_SECS: float


class InflightVerifyCounters:
    """The two counts the lane keeps across in-flight verifies.

    The verify-attempt count drives the periodic cold-verify safety valve and
    covers the worker's whole lifetime; it counts only attempts whose warm
    acquire succeeded, so a deferred attempt cannot advance the valve (task
    3003). The per-task dead-verify-abort count is the no-progress busy-loop
    guard (task 2420): consecutive no-progress aborts, forgotten on every
    other exit so a later hang counts fresh and the map holds only live tasks.
    """

    def __init__(self) -> None:
        self._verify_attempts = 0
        self._dead_verify_aborts: dict[str, int] = {}

    @property
    def verify_attempts(self) -> int:
        return self._verify_attempts

    def note_verify_attempt(self, attempt: int) -> None:
        self._verify_attempts = attempt

    def dead_verify_aborts(self, task_id: str) -> int:
        return self._dead_verify_aborts.get(task_id, 0)

    def strike_dead_verify(self, task_id: str) -> int:
        strike = self._dead_verify_aborts.get(task_id, 0) + 1
        self._dead_verify_aborts[task_id] = strike
        return strike

    def forget_dead_verify(self, task_id: str) -> None:
        self._dead_verify_aborts.pop(task_id, None)


@dataclass(frozen=True)
class InflightVerifyLane:
    """What every in-flight verify of one lane shares."""

    git_ops: Callable[[], GitOps]
    clock: ClockPort
    halt: HaltState
    tuning: InflightVerifyTuning
    counters: InflightVerifyCounters
    custody: DispatchCustody
    chain_planning: ChainPlanning
    worktrees: DispatchWorktrees
    contention: LeaseContention
    event_store: Callable[[], EventStore | None]
    escalation_queue: Callable[[], Any]
    retry_budgets: Callable[[], VerifyRetryBudgets]
    post_merge_verify: Callable[..., Coroutine[Any, Any, MergeOutcome | None]]


@dataclass(frozen=True)
class InflightVerifyCall:
    """One dispatch's verify arguments.

    *depth* (task 2340) and *probe_base* (task 2359) are attribution labels for
    the merge-skew classifier: a firing probe relabels the verify onto a deeper
    built stack it never exercises, and *item* is never mutated. *chain_items*
    is a fact about the tree that actually runs, in chain-item units (the
    dispatching item is #1); ``chain_items >= 2`` is the deep-verify
    discriminator scripts/merge-deep-canary-predicate.sh keys on. *chain*
    (task 3185) REDIRECTS the verify onto its cumulative tip, inside its lane,
    whose release is this verify's obligation. *verify_wt* receives the
    post-swap worktree; *entry_slot* is the entry every requeue or drop exit
    vacates and the RunnerUnavailable exit benches.
    """

    item: RealMergeItem
    lease: HostLease
    depth: int | None = None
    probe_base: str | None = None
    chain_items: int = 1
    chain: ChainResult | None = None
    verify_wt: VerifyWorktreeHandle | None = None
    entry_slot: InflightEntrySlot | None = None
    dispatch_facts: MergeVerifyDispatchFacts | None = None


@dataclass(frozen=True)
class _Verdict:
    out: MergeOutcome | None


@dataclass(frozen=True)
class _VerifyTarget:
    """What an in-flight verify verifies and where, as resolved at construction."""

    merge_wt: Path
    merge_commit: str
    chain_items: int
    chain_build_ms: int | None
    is_warm_path: bool


def _requeued() -> InflightVerifyResult:
    return InflightVerifyResult(outcome=None, merge_wt=None, status=InflightStatus.REQUEUED)


class InflightVerifyRun:
    """One in-flight verify, from target to settled result."""

    def __init__(self, lane: InflightVerifyLane, call: InflightVerifyCall) -> None:
        self._lane = lane
        self._call = call
        self._item = call.item
        self._req = call.item.request
        self._lease = call.lease
        self._chain = call.chain
        self._slot = call.entry_slot
        self._dispatch_facts = with_chain_members(call.dispatch_facts, self._req, call.chain)
        target = self._resolve_target()
        self._merge_wt: Path = target.merge_wt
        self._merge_commit: str = target.merge_commit
        self._chain_items = target.chain_items
        self._chain_build_ms: int | None = target.chain_build_ms
        self._is_warm_path = target.is_warm_path
        self._warm_capture: list[VerifyResult] = []
        self._attempt0_shadow: dict[str, str] = {}
        self._spec_warm = False
        self._verify_task: asyncio.Task[MergeOutcome | None] | None = None
        self._ru_owned_by_handler = False
        self._chain_lane_released = False
        self._chain_item_wt_dropped = False
        self._chain_adopting = False

    def _resolve_target(self) -> _VerifyTarget:
        """Choose WHAT is verified and WHERE, once, so no later phase consults
        ``chain`` again. On the chain arm both, and the item count and build
        cost, are read off the ChainResult that actually ran: computed, never
        accumulated onto the caller's count."""
        chain = self._chain
        if chain is not None:
            assert chain.lane is not None, (
                'a non-empty ChainResult always HOLDS a lane (merge_types.py, '
                '"Lane ownership") — a None lane here means _deep_chain_placement '
                'returned an empty result it should have declined'
            )
            return _VerifyTarget(
                merge_wt=chain.lane,
                merge_commit=chain.tip.strip(),
                chain_items=1 + len(chain.links),
                chain_build_ms=chain.build_ms,
                is_warm_path=chain.lane_warm,
            )
        assert self._item.merge_wt is not None
        merge_commit = self._item.merge_result.merge_commit
        assert merge_commit is not None
        return _VerifyTarget(
            merge_wt=self._item.merge_wt,
            merge_commit=merge_commit.strip(),
            chain_items=self._call.chain_items,
            chain_build_ms=None,
            is_warm_path=False,
        )

    async def run(self) -> InflightVerifyResult:
        try:
            verdict = await self._verify()
            if not isinstance(verdict, _Verdict):
                return verdict
        except RunnerUnavailable as exc:
            return await self._on_runner_unavailable(exc)
        except (MergeVerifyLeaseContended, MergeVerifyLeaseHeld) as exc:
            return await self._on_lane_unavailable(exc)
        except Exception as exc:
            return await self._on_verify_error(exc)
        finally:
            await self._reap_orphaned_verify()
            await self._release_chain_resources()
        self._close_per_task_streaks()
        if self._chain is not None:
            return await self._settle_chain(verdict.out)
        return self._settle_single(verdict.out)

    # ── the verify ──────────────────────────────────────────────────────

    async def _verify(self) -> _Verdict | InflightVerifyResult:
        if self._lease.is_local and self._chain is None:
            await self._swap_to_warm_lane()
        logger.info(
            f'Task {self._req.task_id}: verify start (merge={self._merge_commit[:8]}, '
            f'worktree={self._merge_wt.name})'
        )
        verify_base = await self._publish_verify_base()
        self._verify_task = asyncio.ensure_future(self._launch(verify_base))
        return await self._await_verdict(self._verify_task)

    async def _swap_to_warm_lane(self) -> None:
        """Swap a local verify onto the persistent warm merge-verify lane.

        The attempt is counted only once the acquire has succeeded, and the
        safety valve still receives the 1-based count it documents. The swap
        is published on *verify_wt*, the one place the verify's worktree
        changes, so an adopted head lands from the live lane (task 3186).
        """
        lane, req, item = self._lane, self._req, self._item
        attempt = lane.counters.verify_attempts + 1
        due = _safety_valve_due(
            attempt, req.config.git.persistent_merge_worktree_safety_valve_every_n,
        )
        merge_wt, self._spec_warm = await _acquire_warm_verify_worktree(
            lane.git_ops(), req, self._merge_wt, self._merge_commit,
            safety_valve_due=due,
            speculative=item.speculative,
        )
        lane.counters.note_verify_attempt(attempt)
        assert merge_wt is not None
        self._merge_wt = merge_wt
        if merge_wt is not item.merge_wt:
            lane.worktrees.deregister_owned(item.merge_wt)
        verify_wt = self._call.verify_wt
        if verify_wt is not None:
            verify_wt.merge_wt = merge_wt
            verify_wt.spec_warm = self._spec_warm
        self._is_warm_path = (
            (req.config.git.persistent_merge_worktree and not due)
            or (req.config.git.merge_spec_warm_lane_pool and self._spec_warm)
        )

    async def _publish_verify_base(self) -> VerifyBaseFacts:
        """The merge-skew attribution base: see merge_lane/types.py::VerifyBaseFacts."""
        item, probe_base = self._item, self._call.probe_base
        main_sha = item.base_sha if probe_base is None else probe_base
        verify_base = VerifyBaseFacts(
            main_sha=main_sha,
            merge_base_sha=await resolve_dispatch_time_merge_base(
                self._req.config.project_root, main_sha, item.merged_branch_tip,
            ),
        )
        if self._slot is not None:
            self._slot.publish_verify_base(verify_base)
        return verify_base

    def _launch(self, verify_base: VerifyBaseFacts) -> Coroutine[Any, Any, MergeOutcome | None]:
        lane, call = self._lane, self._call
        budgets = lane.retry_budgets()
        warm = self._is_warm_path
        return lane.post_merge_verify(
            lane.git_ops(), self._req, self._merge_wt,
            timeouts=budgets.timeouts,
            enospc_retries=budgets.enospc_retries,
            max_timeouts=budgets.max_timeouts,
            max_enospc=budgets.max_enospc,
            merge_sha=self._merge_commit,
            on_result=self._warm_capture.append if warm else None,
            shadow_baseline_sink=self._attempt0_shadow if warm else None,
            keep_worktrees=lane.worktrees.owned(),
            runner=None if self._lease.is_local else self._lease.runner,
            depth=call.depth,
            speculative=self._item.speculative,
            chain_items=self._chain_items,
            chain_build_ms=self._chain_build_ms,
            merge_base_sha=verify_base.merge_base_sha,
            main_sha=verify_base.main_sha,
            dispatch_facts=self._dispatch_facts,
        )

    def _progress_probe_secs(self) -> float:
        """The progress-probe interval, clamped below the budget (tasks 2420, 4579).

        A probe interval at or past the budget would trip abort trigger 3
        before a single probe ran, false-aborting every verify on either lease
        kind; the clamp is local and never mutates the tuning.
        """
        tuning = self._lane.tuning
        probe_secs = tuning.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS
        if probe_secs >= tuning.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS:
            logger.warning(
                'Task %s: INFLIGHT_VERIFY_PROGRESS_PROBE_SECS (%.3fs) >= '
                'INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS (%.3fs) -- would '
                'false-abort every verify before a single progress '
                'probe could run; clamping the effective probe interval '
                'to budget/2 for this verify',
                self._req.task_id, probe_secs,
                tuning.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS,
            )
            probe_secs = tuning.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS / 2
        return probe_secs

    async def _await_verdict(
        self, verify_task: asyncio.Task[MergeOutcome | None],
    ) -> _Verdict | InflightVerifyResult:
        """Poll the verify every VERIFY_ABANDON_POLL_SECS for its verdict or an
        abort trigger: abandon before halt before no progress, so a waiter
        that gave up wins over a simultaneous halt.

        The progress watch is seeded before the first await on both lease
        kinds, so a coasting remote lease gets no free budget window; what
        counts as progress lives once, at
        orchestrator/src/orchestrator/merge_lane/liveness.py::VerifyProgressWatch.
        """
        lane, clock = self._lane, self._lane.clock
        watch = VerifyProgressWatch.start(
            clock.monotonic(),
            cpu_ticks=clock.verify_cpu_ticks(self._merge_wt),
            content_mtime=clock.newest_content_mtime(self._merge_wt),
        )
        probe_secs = self._progress_probe_secs()
        while True:
            done = await clock.wait_for_any(
                {verify_task}, timeout=lane.tuning.VERIFY_ABANDON_POLL_SECS,
            )
            if verify_task in done:
                return self._take_verdict(verify_task)
            if lane.custody.request_abandoned(self._req):
                return await self._abort_abandoned()
            if lane.halt.is_operator_halted:
                return await self._abort_operator_halt()
            now = clock.monotonic()
            watch = watch.observe(
                now,
                dispatch_live=remote_dispatch_live(self._lease),
                probe_every_secs=probe_secs,
                cpu_ticks=lambda: clock.verify_cpu_ticks(self._merge_wt),
                content_mtime=lambda: clock.newest_content_mtime(self._merge_wt),
            )
            no_progress_secs = watch.no_progress_secs(now)
            if no_progress_secs >= lane.tuning.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS:
                return await self._abort_no_progress(no_progress_secs, watch)

    def _take_verdict(self, verify_task: asyncio.Task[MergeOutcome | None]) -> _Verdict:
        """Record the verdict and decide adoption now, because the ``finally``
        runs before the chain exit and must know whether to keep the item's
        ephemeral worktree. Only a pass adopts: a low-disk skip does not, and
        an exception never reaches here."""
        out = verify_task.result()
        self._lane.custody.note_verify_completed(out, self._slot, self._lease)
        self._chain_adopting = self._chain is not None and out is None
        return _Verdict(out)

    # ── the three abort triggers ────────────────────────────────────────

    async def _tear_down(self, cause: VerifyCancelCause) -> None:
        assert self._verify_task is not None
        await self._lane.custody.teardown_verify_task(
            self._lease, self._verify_task, self._req.task_id,
            cancel=_slot_cancel(cause, self._slot, self._lease),
        )
        await self._dispose_verify_worktree()

    async def _abort_abandoned(self) -> InflightVerifyResult:
        """Trigger 1: the sole waiter gave up, so DROP the request, retiring it
        at this site (task 3082) since nothing re-enters for its id."""
        lane, task_id = self._lane, self._req.task_id
        await self._tear_down(VerifyCancelCause.ABANDONED)
        lane.counters.forget_dead_verify(task_id)
        lane.contention.clear_streak(task_id)
        await lane.custody.drop_from_verify(self._item, self._slot)
        return InflightVerifyResult(outcome=None, merge_wt=None, status=InflightStatus.DROPPED)

    async def _abort_operator_halt(self) -> InflightVerifyResult:
        """Trigger 2: an operator halt re-queues the merge for re-verify after
        un-halt; ``req.result`` stays pending and retry counters untouched."""
        task_id = self._req.task_id
        logger.warning(
            'Task %s: operator halt — aborting in-flight verify '
            'and re-queuing merge for re-verify after un-halt',
            task_id,
        )
        await self._tear_down(VerifyCancelCause.OPERATOR_HALT)
        await self._lane.custody.requeue_from_verify(self._item, self._slot)
        self._lane.contention.clear_streak(task_id)
        return _requeued()

    async def _abort_no_progress(
        self, no_progress_secs: float, watch: VerifyProgressWatch,
    ) -> InflightVerifyResult:
        """Trigger 3: no verify progress for a full budget (tasks 2420, 2566)
        re-queues, and the MAX_INFLIGHT_DEAD_VERIFY_ABORTS-th consecutive one
        resolves 'blocked'; each firing is recorded as a strike per
        orchestrator/src/orchestrator/merge_lane/no_progress_abort.py::NoProgressAbort."""
        lane, req = self._lane, self._req
        abort = NoProgressAbort(
            task_id=req.task_id,
            request_id=req.request_id,
            runner=self._lease.name,
            is_local=self._lease.is_local,
            no_progress_secs=no_progress_secs,
            budget_secs=lane.tuning.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS,
            strike=lane.counters.strike_dead_verify(req.task_id),
            max_strikes=lane.tuning.MAX_INFLIGHT_DEAD_VERIFY_ABORTS,
            dispatch_seen_in_flight=watch.dispatch_seen_in_flight,
        )
        logger.warning(
            'Task %s: no in-flight verify progress for %.0fs '
            '(budget=%.0fs) — %s (%d/%d consecutive dead aborts)',
            abort.task_id,
            abort.no_progress_secs,
            abort.budget_secs,
            'abandoning without re-queue' if abort.capped
            else 'aborting and re-queuing merge for re-verify',
            abort.strike,
            abort.max_strikes,
        )
        emit_no_progress_abort(lane.event_store(), abort)
        await self._tear_down(VerifyCancelCause.NO_PROGRESS)
        lane.contention.clear_streak(req.task_id)
        if abort.capped:
            lane.counters.forget_dead_verify(req.task_id)
            return self._resolve_blocked(MergeOutcome('blocked', reason=abort.terminal_reason()))
        await lane.custody.requeue_from_verify(self._item, self._slot)
        return _requeued()

    def _resolve_blocked(self, outcome: MergeOutcome) -> InflightVerifyResult:
        if not self._req.result.done():
            self._req.result.set_result(outcome)
        return InflightVerifyResult(outcome=outcome, merge_wt=None)

    # ── exits without a verdict ─────────────────────────────────────────

    async def _on_runner_unavailable(self, exc: RunnerUnavailable) -> InflightVerifyResult:
        """A dead remote transport: bench the host and free its slot now (task
        7266), and hand the worktree back un-cleaned for finalize to dispose of
        beside the re-merge that replaces it (task 3251). The local fallback is
        the caller's decision. On the chain arm the lane is the ``finally``'s
        to release, so nothing is handed back."""
        req = self._req
        logger.warning(
            'Task %s: remote runner unavailable (merge=%s) on host %s — '
            'will re-dispatch on another host: %s',
            req.task_id, self._merge_commit[:8], self._lease.name, elide_middle(str(exc)))
        self._lane.contention.clear_streak(req.task_id)
        self._ru_owned_by_handler = True
        await self._lane.custody.bench_from_verify(self._slot, exc)
        chained = self._chain is not None
        return InflightVerifyResult(
            outcome=None,
            merge_wt=None if chained else self._merge_wt,
            status=InflightStatus.RUNNER_UNAVAILABLE,
            reason=str(exc),
            spec_warm=False if chained else self._spec_warm,
            kind=exc.kind,
            identity=exc.identity,
        )

    async def _on_lane_unavailable(
        self, exc: MergeVerifyLeaseContended | MergeVerifyLeaseHeld,
    ) -> InflightVerifyResult:
        """The shared merge-verify lane lock was unavailable: DEFER by
        re-queuing, never a 'blocked' verdict whose fixed reason would feed
        workflow.py's consecutive_merge_thrash ladder (reify 5354/5300/5328).

        Three raisers land here (tasks 2828, 3003): the verify's lease
        acquire, and the warm-lane reset's contended acquire and
        foreign-holder pre-check, both before any verify task exists. A defer
        is transient but bounded: throttled to CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS
        between attempts, alarmed once at CONTENDED_LEASE_REQUEUE_WARN_STREAK,
        and capped terminally after MAX_CONTENDED_LEASE_DEFER_SECS of unbroken
        contention. The requeue runs only on the non-cancelled path: a
        cancelling caller owns the request, and re-queuing from a ``finally``
        would double-file it at every cancel site.
        """
        lane, task_id = self._lane, self._req.task_id
        waited = float(getattr(exc, 'wait_secs', 0.0) or 0.0)
        streak, elapsed = lane.contention.advance_streak(task_id, exc)
        if elapsed >= lane.tuning.MAX_CONTENDED_LEASE_DEFER_SECS:
            return await self._cap_out_contended_lane(exc, streak, elapsed)
        self._log_contended_defer(exc, streak)
        await self._dispose_verify_worktree()
        backoff = max(0.0, lane.tuning.CONTENDED_LEASE_DEFER_MIN_PERIOD_SECS - waited)
        if backoff:
            await lane.clock.sleep(backoff)
        await lane.custody.requeue_from_verify(self._item, self._slot)
        return _requeued()

    async def _cap_out_contended_lane(
        self, exc: BaseException, streak: int, elapsed: float,
    ) -> InflightVerifyResult:
        """Resolve 'blocked' past the contention budget, forgetting every
        per-task count so a re-submission starts fresh. The reason carries a
        strictly increasing cap-out ordinal so no two cap-outs share a
        merge_outcome_signature; never round its figures coarser."""
        lane, task_id = self._lane, self._req.task_id
        lane.contention.clear_streak(task_id)
        lane.counters.forget_dead_verify(task_id)
        cap_out = lane.contention.next_cap_out()
        await self._dispose_verify_worktree()
        logger.error(
            'Task %s: merge-verify lane has been unavailable for %.0fs '
            'across %d consecutive deferred attempts (%s) — resolving '
            'BLOCKED instead of deferring again (lane cap-out #%d). '
            'This is a genuine lane pathology: inspect the merge-verify '
            'lane lock holder.',
            task_id, elapsed, streak, exc,
            cap_out,
        )
        return self._resolve_blocked(MergeOutcome(
            'blocked',
            reason=(
                f'merge-verify lane unavailable for '
                f'{elapsed:.0f}s across {streak} '
                f'consecutive deferred attempts '
                f'(lane cap-out #{cap_out}): {exc}'
            ),
        ))

    def _log_contended_defer(self, exc: BaseException, streak: int) -> None:
        """ERROR once at the warn-streak crossing, a WARNING heartbeat on every
        other defer. Only a bounded-wait raiser carries ``wait_secs``, so the
        duration clause is left out rather than printed as zero."""
        task_id = self._req.task_id
        if streak == self._lane.tuning.CONTENDED_LEASE_REQUEUE_WARN_STREAK:
            wait_secs = getattr(exc, 'wait_secs', None)
            duration_clause = (
                f' (~{streak * wait_secs / 60.0:.0f} min of '
                f'continuous contention)'
                if wait_secs is not None
                else ''
            )
            logger.error(
                'Task %s: merge-verify lane unavailable (%s) — DEFERRED %d '
                'times in a row%s. A long-running or wedged lane holder is '
                'blocking this verify; re-queuing again rather than running '
                'unprotected. If this persists, inspect the merge-verify '
                'lane lock holder.',
                task_id, exc, streak, duration_clause,
            )
            return
        logger.warning(
            'Task %s: merge-verify lane unavailable (%s) — re-queuing '
            'merge for re-verify rather than proceeding unprotected '
            '(consecutive contended-lane requeue #%d)',
            task_id, exc, streak,
        )

    async def _on_verify_error(self, exc: Exception) -> InflightVerifyResult:
        """Any other error resolves 'blocked', except on the chain arm: an
        error out of a cumulative tree is not attributable to this item, so it
        re-queues and its next dispatch verifies its own subset tree
        un-chained, which ends in 'blocked' if that raises too."""
        lane, req = self._lane, self._req
        logger.info(
            f'Task {req.task_id}: verify end '
            f'(merge={self._merge_commit[:8]}, error)'
        )
        await self._dispose_verify_worktree()
        lane.counters.forget_dead_verify(req.task_id)
        lane.contention.clear_streak(req.task_id)
        if self._chain is None:
            return self._resolve_blocked(MergeOutcome('blocked', reason=f'Verification error: {exc}'))
        lane.chain_planning.mark_next_dispatch_unchained([req.task_id])
        logger.warning(
            'Task %s: deep tip verify RAISED against tip=%s '
            '(%s) -- NOT attributed to this task (the tree was '
            'cumulative); re-queuing, and the next dispatch for it '
            'will verify its own subset tree un-chained',
            req.task_id, self._merge_commit[:8], exc,
        )
        await lane.custody.requeue_from_verify(self._item, self._slot)
        return _requeued()

    # ── every exit ──────────────────────────────────────────────────────

    async def _reap_orphaned_verify(self) -> None:
        """The orphan guard (task 3043, INV-C): the inner verify task never
        outlives this run, on any exit route, including an external cancel
        that none of the handlers catch. It must never return a value or
        raise its own error."""
        verify_task = self._verify_task
        if verify_task is None:
            return
        if not verify_task.done():
            await self._bounded_reap(verify_task)
        orphan_exc: BaseException | None = None
        if verify_task.done() and not verify_task.cancelled():
            orphan_exc = verify_task.exception()
        if (
            isinstance(orphan_exc, RunnerUnavailable)
            and not self._lease.is_local
            and not self._ru_owned_by_handler
        ):
            self._bench_escaped_runner_unavailable(orphan_exc)

    async def _bounded_reap(self, verify_task: asyncio.Task[MergeOutcome | None]) -> None:
        """Cancel and reap through the sanctioned teardown, bounded by
        ORPHAN_REAP_TIMEOUT_SECS so a slow unwind against a dead host cannot
        wedge the canceller.

        ``lease=None``: every route reaching here has already aborted the
        remote, and a second ``cancel_verify()`` is observable. ``asyncio.shield``
        is load-bearing: without it the timeout's cancel chains into the inner
        task and the bound is a no-op against exactly the inner that motivates it.
        """
        timeout = self._lane.tuning.ORPHAN_REAP_TIMEOUT_SECS
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(
                    self._lane.custody.teardown_verify_task(
                        None, verify_task, self._req.task_id,
                        cancel=None, shutdown_defensive=True,
                    )
                ),
                timeout=timeout,
            )
        if not verify_task.done():
            with contextlib.suppress(Exception):
                logger.warning(
                    'Task %s: inner verify task on host %r did not '
                    'unwind within %.1fs of cancellation — abandoning '
                    'the reap so the cancelling path is not wedged; '
                    'the task is left cancelled-but-pending',
                    self._req.task_id, self._lease.name,
                    timeout,
                )

    def _bench_escaped_runner_unavailable(self, exc: RunnerUnavailable) -> None:
        """Bench a host whose RunnerUnavailable never reached the handler, or it
        is stranded. The bench runs first in its own suppress so a failing log
        line cannot cost the state change."""
        with contextlib.suppress(Exception):
            self._lane.custody.bench_unavailable_runner(self._lease.name, exc)
        with contextlib.suppress(Exception):
            logger.warning(
                'Task %s: in-flight verify unwound while its remote '
                'verify task failed with RunnerUnavailable on host '
                '%r — quarantined + recorded the host so the '
                'reprobe sweep can re-adopt it: %s',
                self._req.task_id, self._lease.name, exc,
            )

    async def _release_chain_resources(self) -> None:
        """Return the chain's resources on every exit, after the orphan guard
        has proved the inner verify dead. A round that will not adopt gives
        its reserved links back synchronously, before the await (task 7720),
        and drops the item's ephemeral worktree; the adopting exit hands that
        worktree to finalize instead (task 3186)."""
        if self._chain is not None and not self._chain_adopting:
            self._lane.chain_planning.end_chain_reservation(self._req.request_id)
        await self._release_chain_lane()
        if not self._chain_adopting:
            await self._drop_chain_item_ephemeral_wt()

    async def _release_chain_lane(self) -> None:
        """Return the chain's pooled lane at most once, whichever exits reach it."""
        chain = self._chain
        if chain is None or self._chain_lane_released:
            return
        self._chain_lane_released = True
        await release_chain_build_lane(self._lane.git_ops(), chain.lane, warm=chain.lane_warm)

    async def _drop_chain_item_ephemeral_wt(self) -> None:
        """Drop the chain-dispatched item's ephemeral ``_merge-<uuid>``, once.

        Left registered it would be immortal; left on disk unregistered it
        trips the I6 worktree-ledger audit. Its merge commit stays reachable
        from the chain lane, and the request is re-queued on every exit that
        reaches here, so it is re-merged rather than reused.
        """
        if self._chain is None or self._chain_item_wt_dropped:
            return
        self._chain_item_wt_dropped = True
        await self._lane.worktrees.cleanup_owned(self._item.merge_wt)

    async def _dispose_verify_worktree(self) -> None:
        """Give back the worktree this verify ran in: a chain's pooled lane goes
        back to the pool, never through ``release_or_cleanup``, which would
        ``git worktree remove`` it and lose the slot."""
        if self._chain is not None:
            await self._release_chain_lane()
            return
        await self._lane.worktrees.release_or_cleanup(self._merge_wt, self._spec_warm)

    def _close_per_task_streaks(self) -> None:
        """A verify that returned at all proves it was not hung, and that the
        lane lock was acquired: neither count may span it."""
        task_id = self._req.task_id
        self._lane.counters.forget_dead_verify(task_id)
        self._lane.contention.clear_streak(task_id)

    # ── settling a verdict ──────────────────────────────────────────────

    async def _settle_chain(self, out: MergeOutcome | None) -> InflightVerifyResult:
        """A tip verdict proves the CUMULATIVE tree, never one member's subset.

        A green tip licenses landing the contiguous prefix in order, which
        finalize's CAS walk does; this exit only says "adopt", tearing the
        head's now-redundant verify down here rather than waiting out its
        verdict (task 3186). A red tip names no culprit, so it re-queues and
        every chained item next verifies un-chained; REQUEUED emits no
        merge_attempt, so deep fails cannot feed consecutive_merge_thrash.
        The pass estimate is fed the BUILT depth on both arms.
        """
        lane, req, chain = self._lane, self._req, self._chain
        assert chain is not None
        chain_depth = 1 + len(chain.links)
        lane.chain_planning.note_tip_result(self._dispatch_facts.chain_depth_reason, chain_depth, out)
        if self._chain_adopting:
            logger.info(
                'Task %s: deep tip verify end (tip=%s, items=%d, passed=True) '
                '— ADOPTING: handing the prefix to the in-order CAS walk in '
                '_finalize_inflight',
                req.task_id, self._merge_commit[:8], chain_depth,
            )
            await lane.custody.adopt_head_on_tip_authority(self._item)
            return InflightVerifyResult(outcome=None, merge_wt=self._item.merge_wt, spec_warm=False)
        logger.info(
            'Task %s: deep tip verify end (tip=%s, items=%d, passed=False) '
            '— re-queuing; a red tip names no culprit, so every chained '
            'item next verifies un-chained',
            req.task_id, self._merge_commit[:8], chain_depth,
        )
        lane.chain_planning.mark_next_dispatch_unchained(
            [req.task_id, *(task_id for task_id, _ in chain.links)],
        )
        await lane.custody.requeue_from_verify(self._item, self._slot)
        return _requeued()

    def _settle_single(self, out: MergeOutcome | None) -> InflightVerifyResult:
        """Feed the pass estimate (a round the deep policy sized but did not
        chain still counts), then pass with the warm shadow baseline, fail,
        or report a low-disk skip."""
        self._lane.chain_planning.note_tip_result(self._dispatch_facts.chain_depth_reason, 1, out)
        task_id, merge = self._req.task_id, self._merge_commit[:8]
        if out is None:
            logger.info(f'Task {task_id}: verify end (merge={merge}, passed=True)')
            return InflightVerifyResult(
                outcome=None, merge_wt=self._merge_wt,
                warm_results=self._warm_shadow_results(), spec_warm=self._spec_warm,
            )
        if out.verify_skipped:
            logger.info(f'Task {task_id}: verify skipped: low disk (merge={merge})')
        else:
            logger.info(f'Task {task_id}: verify end (merge={merge}, passed=False)')
        return InflightVerifyResult(outcome=out, merge_wt=self._merge_wt, spec_warm=self._spec_warm)

    def _warm_shadow_results(self) -> dict[str, str]:
        """The warm shadow baseline: the attempt-0 verdicts of a corroborated
        narrowed retry unioned with the possibly partial warm output, so a
        partial map never reads as a phantom only_cold divergence. An empty
        parse stays empty, so the unparseable alarm stays reachable."""
        if not self._warm_capture:
            return {}
        test_output = self._warm_capture[0].test_output or ''
        warm_results = build_warm_shadow_results(test_output, self._attempt0_shadow)
        if not warm_results and self._req.config.git.warm_verify_shadow_compare:
            _alarm_warm_shadow_unparseable(
                self._lane.escalation_queue(), self._merge_commit, test_output,
            )
        return warm_results
