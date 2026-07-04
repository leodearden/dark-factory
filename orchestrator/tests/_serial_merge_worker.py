"""Test-local reference copy of the retired serial ``MergeWorker`` (task 1998 / R7b).

PRD: plans/merge-queue-modularization-invariants-prd.md, task nu.

The production package's sole merge worker is now
``orchestrator.merge_queue.SpeculativeMergeWorker``.  The legacy serial
worker below is preserved here VERBATIM (byte-identical method bodies) so
the ~89 existing test constructions across the suite keep exercising the
same serial-specific surface (``_dequeue``/``_process``/``_do_merge``,
front-of-queue ``_urgent`` CAS re-enqueue) without being rewritten against
the two-coroutine speculative pipeline, which has different semantics.

This module lives in orchestrator/tests/ and is imported by bare module name
(``from _serial_merge_worker import MergeWorker``), matching the existing
flat-test-helper convention (see ``_orch_helpers.py``, ``_workflow_helpers.py``
— orchestrator/tests/ has no ``__init__.py``).

Do not add new behavior here.  This is a frozen reference fixture, not a
production class — all imports below are the exact set the class body
(originally orchestrator/src/orchestrator/merge_queue.py:3243-3567) reads as
module globals; the moved code itself is unchanged.
"""

from __future__ import annotations

import asyncio
import collections
import time

from orchestrator.merge_queue import (
    MAX_AUTO_CHAINED_GENERATIONS,
    MERGE_LANES,
    MERGE_WORKER_SHUTDOWN_REASON,
    WORKTREE_MISSING_REASON_PREFIX,
    Decided,
    EventStore,
    EventType,
    GitOps,
    GroupMergeRequest,
    MainHealthAutoHealRegistry,
    MergeOutcome,
    MergeRequest,
    WorktreeMissing,
    _acquire_warm_verify_worktree,
    _do_train_merge,
    _elapsed_ms,
    _emit_merge_attempt,
    _emit_merge_queued,
    _finalize_advanced_merge,
    _GenerationChainContext,
    _HALT_ADVANCE_RESULTS,
    _map_advance_failure,
    _run_post_merge_verify,
    _safety_valve_due,
    _WipHaltMixin,
    classify_and_merge,
    logger,
)


class MergeWorker(_WipHaltMixin):
    """Single coroutine that processes merge requests serially.

    Owns all main-branch advancement via CAS ``update-ref``.  The harness
    creates one instance and passes the same ``asyncio.Queue`` to every
    ``TaskWorkflow``.

    .. note:: Legacy path.
        This class inherits the per-lane halt *state machine* from
        ``_WipHaltMixin`` and honours ``halt_for_wip`` / ``unhalt_wip``
        (all-lanes) correctly.  However its ``_dequeue`` picks from the
        FIFO queue without consulting ``req.lane``, so it does **not**
        implement lane-priority ordering.  Use
        :class:`SpeculativeMergeWorker` (the production worker) when lane
        priority matters.
    """

    MAX_CAS_RETRIES = 5
    # After this many consecutive post-merge verify TIMEOUTS for the same
    # task, the merge queue stops trying and returns an 'abandoned' blocked
    # outcome.  Caps the verify-timeout / re-enqueue oscillation (two tasks
    # alternating on the merge queue for hours, each dying at the 30-min
    # warm timeout).  Counter resets on any successful merge for that task.
    MAX_POST_MERGE_VERIFY_TIMEOUTS: int = 2
    # After a post-merge verify fails with an ENOSPC signature, prune stale
    # _merge-* worktrees and retry the verify at most this many times before
    # escalating as transient infra.  Disk pressure is often self-healing, so
    # one retry-after-prune is the conservative middle ground between blindly
    # blocking and looping.  Resets on any successful merge for that task.
    MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES: int = 1

    def __init__(
        self,
        git_ops: GitOps,
        queue: asyncio.Queue[MergeRequest],
        event_store: EventStore | None = None,
    ):
        self._git_ops = git_ops
        self._queue = queue
        self._event_store = event_store
        # Front-of-queue buffer for CAS-failure re-enqueue (processed first)
        self._urgent: collections.deque[MergeRequest] = collections.deque()
        self._running = True
        # Per-task CAS re-enqueue counter — prevents infinite loops
        self._cas_retries: dict[str, int] = {}
        # Per-task consecutive post-merge-verify-timeout counter.  Bumped
        # when a verify times out, cleared on a successful merge.  Keyed by
        # task_id; lives across submissions (re-submits of the same task
        # after an orchestrator re-queue also feed this counter).
        self._post_merge_verify_timeouts: dict[str, int] = {}
        # Per-task ENOSPC prune-and-retry counter (see
        # MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES).  Same lifetime semantics as
        # the timeout counter: persists across submissions, reset on success.
        self._post_merge_verify_enospc_retries: dict[str, int] = {}
        # γ2 per-branch generation auto-chain counter.  Incremented on each
        # consecutive tip-advance equivalence failure; popped on a clean 'done'
        # landing or bound-exceeded escalation.  Mirrors _cas_retries shape.
        self._generation_chain_counts: dict[str, int] = {}
        # Persistent warm merge-verify worktree: counts verifying attempts so
        # _safety_valve_due can fire the periodic cold-verify (PRD §10 invariant 6).
        # Only incremented when not skip_verify; never reset so the counter
        # covers the full worker lifetime (cross-submission).
        self._verify_attempt_count: int = 0
        # Per-lane halt: each event is set (running) by default
        self._lane_halt = {ln: asyncio.Event() for ln in MERGE_LANES}
        for ln in MERGE_LANES:
            self._lane_halt[ln].set()
        self._lane_halt_owner: dict[str, str | None] = {ln: None for ln in MERGE_LANES}
        # Operator-halt signal (see _WipHaltMixin.operator_halt). Initially
        # clear = no operator halt. MergeWorker has no verifier abort-poll loop,
        # so this is for API parity (operator_halt/unhalt_all_lanes reference it).
        self._operator_halt = asyncio.Event()
        # Cross-workflow auto-heal attempt counter (shared via self.merge_worker
        # on TaskWorkflow instances).  Lives on the worker so the counter persists
        # across the heal→re-break cycle without any harness.py change.
        self.auto_heal_registry: MainHealthAutoHealRegistry = MainHealthAutoHealRegistry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop — runs until ``stop()`` is called."""
        while self._running:
            await self._wait_until_any_lane_runnable()  # blocks if all lanes halted
            req = await self._dequeue()
            if req is None:
                break  # shutdown sentinel

            if self._event_store is not None:
                self._event_store.emit(
                    EventType.merge_dequeued,
                    task_id=req.task_id,
                    phase='merge',
                    data={'branch': req.branch, 'queue_depth': self._queue.qsize()},
                )

            outcome = await self._process(req)
            # outcome is None when the request was re-enqueued (CAS failure)
            if outcome is not None and not req.result.done():
                req.result.set_result(outcome)

    async def stop(self) -> None:
        """Graceful shutdown: drain queues and resolve all pending Futures."""
        self._running = False
        shutdown = MergeOutcome('blocked', reason=MERGE_WORKER_SHUTDOWN_REASON)

        # Drain urgent buffer
        while self._urgent:
            req = self._urgent.popleft()
            if not req.result.done():
                req.result.set_result(shutdown)

        # Drain main queue
        while not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                if not req.result.done():
                    req.result.set_result(shutdown)
            except asyncio.QueueEmpty:
                break

        # Unblock the run() loop if it's waiting on an empty queue
        await self._queue.put(None)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _dequeue(self) -> MergeRequest | None:
        """Get the next request — urgent buffer first, then main queue."""
        if self._urgent:
            return self._urgent.popleft()

        item = await self._queue.get()
        if item is None:
            return None  # shutdown sentinel
        return item

    async def _process(self, req: MergeRequest) -> MergeOutcome | None:
        """Process one merge request.  Returns None if re-enqueued."""
        # Drop-on-detection: if the workflow that submitted this request has
        # cancelled its result future (workflow soft-cancel), don't even
        # start the merge.  Skipping here avoids the orphan-halt window
        # entirely for the common case (workflow exited before dequeue).
        if self._request_abandoned(req):
            return None
        try:
            return await self._do_merge(req)
        except WorktreeMissing as exc:
            # Worktree removed out-of-band (e.g. human cleanup after marking
            # the task done).  Surface with a recognisable prefix so
            # ``TaskWorkflow`` can re-check status.
            logger.info(
                f'Merge worker for task {req.task_id}: missing worktree '
                f'{exc.path} — surfacing as blocked'
            )
            return MergeOutcome(
                'blocked',
                reason=f'{WORKTREE_MISSING_REASON_PREFIX}: {exc.path}',
            )
        except Exception as exc:
            logger.exception(
                f'Merge worker error for task {req.task_id}: {exc}'
            )
            return MergeOutcome('blocked', reason=f'Merge worker error: {exc}')

    async def _do_merge(self, req: MergeRequest) -> MergeOutcome | None:
        # Train dispatch: GroupMergeRequest is handled by the shared
        # _do_train_merge pipeline which owns rebase + merge + verify + advance
        # + member callbacks.  The rest of _do_merge is the single-task path.
        if isinstance(req, GroupMergeRequest):
            return await _do_train_merge(self, req)

        t0 = time.monotonic()

        # Loop-breaker: refuse to process tasks that have already timed out
        # in post-merge verify MAX_POST_MERGE_VERIFY_TIMEOUTS times in a
        # row.  Short-circuits before any git work so a stuck task can't
        # keep burning merge-queue capacity (30+ minutes per attempt).
        prior_timeouts = self._post_merge_verify_timeouts.get(req.task_id, 0)
        if prior_timeouts >= self.MAX_POST_MERGE_VERIFY_TIMEOUTS:
            logger.warning(
                'Task %s: abandoning merge — %d consecutive post-merge '
                'verify timeouts (threshold=%d)',
                req.task_id, prior_timeouts,
                self.MAX_POST_MERGE_VERIFY_TIMEOUTS,
            )
            _emit_merge_attempt(
                self._event_store, req.task_id, 'abandoned_verify_timeouts',
                attempt=prior_timeouts, duration_ms=_elapsed_ms(t0),
            )
            return self._abandon_outcome(req.task_id, prior_timeouts)

        # Guard + merge + drop-guard pipeline (MQ-refactor kappa): shared with
        # SpeculativeMergeWorker's _merger_loop/_remerge.  main_sha is read
        # here (not just inside classify_and_merge) because it is also
        # needed below for advance_main's expected_main and
        # _finalize_advanced_merge's base_sha — classify_and_merge reads its
        # own actual_main for the already-merged check (benign double-read;
        # advance_main's CAS tolerates any drift between the two reads).
        main_sha = await self._git_ops.get_main_sha()
        result = await classify_and_merge(
            self, req, main_sha, speculative=False, started_monotonic=t0,
        )
        if isinstance(result, Decided):
            return result.outcome
        merge_result = result.merge_result
        branch_head = result.branch_tip
        # classify_and_merge only returns MergedOk after its own drop-guard
        # step asserted merge_commit is set, and after rev-parsing branch
        # HEAD successfully — both are guaranteed non-None here.
        assert merge_result.merge_commit is not None
        assert branch_head is not None

        # 4. Verify — task-1724: unconditional, skip_verify path removed
        merge_wt = merge_result.merge_worktree
        assert merge_wt is not None
        # ── Persistent warm merge-verify worktree swap (PRD §10 κ) ──────
        # Parity with SpeculativeMergeWorker._verify_and_advance: increment the
        # per-worker verify counter and compute the safety-valve predicate so
        # every Nth verifying attempt runs a from-scratch cold verify in a
        # throwaway worktree (PRD §10 invariant 6).
        # merge_result.merge_commit is non-None (asserted above).
        self._verify_attempt_count += 1
        _due = _safety_valve_due(
            self._verify_attempt_count,
            req.config.git.persistent_merge_worktree_safety_valve_every_n,
        )
        merge_wt, _ = await _acquire_warm_verify_worktree(
            self._git_ops, req, merge_wt,
            merge_result.merge_commit,  # non-None; assert at 4044 above
            safety_valve_due=_due,
            speculative=False,  # MergeWorker always handles non-speculative items
        )
        assert merge_wt is not None  # input was non-None; warm or unchanged
        out = await _run_post_merge_verify(
            self._git_ops, req, merge_wt,
            timeouts=self._post_merge_verify_timeouts,
            enospc_retries=self._post_merge_verify_enospc_retries,
            max_timeouts=self.MAX_POST_MERGE_VERIFY_TIMEOUTS,
            max_enospc=self.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES,
            event_store=self._event_store,
            merge_sha=merge_result.merge_commit or '',
        )
        if out is not None:
            return out

        # 5. CAS advance_main
        assert merge_result.merge_commit is not None
        outcome = await self._git_ops.advance_main(
            merge_result.merge_commit,
            merge_wt,
            branch=req.branch,
            max_attempts=req.config.max_advance_attempts,
            expected_main=main_sha,
        )
        result = outcome.result
        await self._git_ops.cleanup_merge_worktree(merge_wt)

        if result == 'advanced':
            return await _finalize_advanced_merge(
                self._git_ops, req, self._event_store,
                merge_commit_fallback=merge_result.merge_commit,
                base_sha=main_sha,
                started_monotonic=t0,
                cas_retries=self._cas_retries,
                timeouts=self._post_merge_verify_timeouts,
                enospc_retries=self._post_merge_verify_enospc_retries,
                chain_ctx=_GenerationChainContext(
                    queue=self._queue,
                    counts=self._generation_chain_counts,
                    max_auto_generations=MAX_AUTO_CHAINED_GENERATIONS,
                ),
                merged_branch_tip=branch_head.strip(),
                advanced_sha=outcome.advanced_sha,
            )

        if result in _HALT_ADVANCE_RESULTS and self._request_abandoned(req):
            # Workflow soft-cancelled mid-merge: dropping the request
            # prevents the orphan-halt window where no escalation owner
            # is registered (2026-05-04 incident).
            return None
        if result != 'cas_failed':
            return await _map_advance_failure(
                self._git_ops, result,
                task_id=req.task_id,
                merge_commit_fallback=merge_result.merge_commit,
                halt=self.halt_for_wip,
                unhalt=self.unhalt_wip,
                cas_retries=self._cas_retries,
                advanced_sha=outcome.advanced_sha,
            )

        # result == 'cas_failed' — transient, re-enqueue with limit
        retries = self._cas_retries.get(req.task_id, 0) + 1
        self._cas_retries[req.task_id] = retries
        if retries > self.MAX_CAS_RETRIES:
            self._cas_retries.pop(req.task_id, None)
            logger.warning(
                f'Task {req.task_id}: CAS retry limit exhausted '
                f'({self.MAX_CAS_RETRIES} attempts)'
            )
            _emit_merge_attempt(self._event_store, req.task_id, 'cas_exhausted', attempt=retries, duration_ms=_elapsed_ms(t0))
            return MergeOutcome(
                'blocked',
                reason=(
                    f'CAS retry limit exhausted after '
                    f'{self.MAX_CAS_RETRIES} attempts for task {req.task_id}'
                ),
            )

        logger.info(
            f'Task {req.task_id}: CAS failed (attempt {retries}/'
            f'{self.MAX_CAS_RETRIES}), re-enqueueing at front'
        )
        _emit_merge_attempt(self._event_store, req.task_id, 'cas_retry', attempt=retries, duration_ms=_elapsed_ms(t0))
        _emit_merge_queued(
            self._event_store, req, reason='cas_retry',
            queue_depth=self._queue.qsize() + len(self._urgent) + 1,
            position=0,
        )
        self._urgent.append(req)
        return None  # don't resolve Future — will be reprocessed
