"""Merge queue: single worker owns all main-branch advancement.

Replaces the previous asyncio.Lock serialization with a queue + dedicated
worker coroutine.  Tasks submit MergeRequest objects and await a Future.
The worker merges, verifies, and CAS-advances main one request at a time.

Conflicts are rejected immediately — the caller resolves them outside the
queue (in its own worktree) and re-submits.  CAS failures (external actor
moved main) trigger front-of-queue re-enqueue for lower conflict risk.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.verify import run_scoped_verification

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig

logger = logging.getLogger(__name__)


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


@dataclass
class MergeOutcome:
    """Result delivered to the caller via the Future."""

    status: Literal['done', 'conflict', 'blocked', 'already_merged']
    reason: str = ''
    conflict_details: str = ''


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
    skip_verify: bool                  # True → pre_rebased and main unchanged
    immediate_outcome: MergeOutcome | None = None  # Set for conflict/already_merged


class MergeWorker:
    """Single coroutine that processes merge requests serially.

    Owns all main-branch advancement via CAS ``update-ref``.  The harness
    creates one instance and passes the same ``asyncio.Queue`` to every
    ``TaskWorkflow``.
    """

    MAX_CAS_RETRIES = 5

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

    def _emit_merge(
        self, task_id: str, outcome: str, *, attempt: int | None = None,
    ) -> None:
        if self._event_store:
            data: dict = {'outcome': outcome}
            if attempt is not None:
                data['attempt'] = attempt
            self._event_store.emit(
                EventType.merge_attempt, task_id=task_id, phase='merge',
                data=data,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop — runs until ``stop()`` is called."""
        while self._running:
            req = await self._dequeue()
            if req is None:
                break  # shutdown sentinel

            outcome = await self._process(req)
            # outcome is None when the request was re-enqueued (CAS failure)
            if outcome is not None and not req.result.done():
                req.result.set_result(outcome)

    async def stop(self) -> None:
        """Graceful shutdown: drain queues and resolve all pending Futures."""
        self._running = False
        shutdown = MergeOutcome('blocked', reason='Merge worker shutting down')

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
        try:
            return await self._do_merge(req)
        except Exception as exc:
            logger.exception(
                f'Merge worker error for task {req.task_id}: {exc}'
            )
            return MergeOutcome('blocked', reason=f'Merge worker error: {exc}')

    async def _do_merge(self, req: MergeRequest) -> MergeOutcome | None:
        # 1. Already-merged detection (ghost-loop fix)
        _, branch_head, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=req.worktree,
        )
        main_sha = await self._git_ops.get_main_sha()
        if await self._git_ops.is_ancestor(branch_head.strip(), main_sha):
            # Guard: if worktree has uncommitted changes, an agent may
            # have started work — don't skip.
            if await self._git_ops.has_uncommitted_work(req.worktree):
                logger.warning(
                    f'Task {req.task_id}: branch is ancestor of main but '
                    f'worktree has uncommitted changes — not skipping merge'
                )
            else:
                logger.info(
                    f'Task {req.task_id}: branch already on main — skipping merge'
                )
                self._emit_merge(req.task_id, 'already_merged')
                return MergeOutcome('already_merged')

        # 2. Merge in a temporary worktree
        merge_result = await self._git_ops.merge_to_main(
            req.worktree, req.branch,
        )

        # 3. Conflict → reject immediately (caller resolves outside queue)
        if merge_result.conflicts:
            logger.info(f'Task {req.task_id}: merge conflicts detected')
            self._emit_merge(req.task_id, 'conflict')
            if merge_result.merge_worktree:
                await self._git_ops.cleanup_merge_worktree(
                    merge_result.merge_worktree,
                )
            return MergeOutcome(
                'conflict', conflict_details=merge_result.details,
            )

        if not merge_result.success:
            if merge_result.merge_worktree:
                await self._git_ops.cleanup_merge_worktree(
                    merge_result.merge_worktree,
                )
            return MergeOutcome('blocked', reason=merge_result.details)

        # 4. Verify (skip if pre-rebased and main unchanged)
        merge_wt = merge_result.merge_worktree
        assert merge_wt is not None
        skip_verify = (
            req.pre_rebased
            and merge_result.pre_merge_sha is not None
            and merge_result.pre_merge_sha == main_sha
        )
        if skip_verify:
            logger.info(
                f'Task {req.task_id}: skipping re-verification '
                f'(pre-rebased, main unchanged)'
            )
        if not skip_verify:
            verify = await run_scoped_verification(
                merge_wt, req.config, req.module_configs,
                task_files=req.task_files,
            )
            if not verify.passed:
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                return MergeOutcome(
                    'blocked',
                    reason=f'Post-merge verification failed: {verify.summary}',
                )

        # 5. CAS advance_main
        assert merge_result.merge_commit is not None
        result = await self._git_ops.advance_main(
            merge_result.merge_commit,
            merge_wt,
            branch=req.branch,
            max_attempts=req.config.max_advance_attempts,
            expected_main=main_sha,
        )
        await self._git_ops.cleanup_merge_worktree(merge_wt)

        if result == 'advanced':
            self._cas_retries.pop(req.task_id, None)
            logger.info(f'Task {req.task_id}: merged to main successfully')
            self._emit_merge(req.task_id, 'done')
            return MergeOutcome('done')

        if result in ('not_descendant', 'contaminated', 'stash_failed'):
            # Permanent failure — do NOT re-enqueue
            self._cas_retries.pop(req.task_id, None)
            return MergeOutcome(
                'blocked',
                reason=f'advance_main failed ({result}) for task {req.task_id}',
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
            self._emit_merge(req.task_id, 'cas_exhausted', attempt=retries)
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
        self._emit_merge(req.task_id, 'cas_retry', attempt=retries)
        self._urgent.append(req)
        return None  # don't resolve Future — will be reprocessed


class SpeculativeMergeWorker:
    """Two-coroutine speculative merge-verify pipeline.

    The Merger coroutine creates merge commits; the Verifier coroutine runs
    verification and CAS-advances main.  While the Verifier processes merge N,
    the Merger speculatively merges N+1 against N's merge SHA.  If N succeeds,
    N+1 is already a descendant and its CAS works immediately.  If N fails,
    the Verifier re-merges N+1 against actual main.

    Speculation depth is capped at 1: the Merger waits on ``_speculation_slot``
    before grabbing N+2 speculatively, which the Verifier sets after completing
    the item preceding the speculation.
    """

    MAX_CAS_RETRIES = 5

    def __init__(
        self,
        git_ops: GitOps,
        queue: asyncio.Queue[MergeRequest],
        event_store: EventStore | None = None,
    ):
        self._git_ops = git_ops
        self._queue = queue
        self._event_store = event_store
        # Internal pipeline: Merger → Verifier
        self._verifier_queue: asyncio.Queue[SpeculativeItem | None] = asyncio.Queue()
        self._running = True
        self._cas_retries: dict[str, int] = {}
        # Depth-1 cap: cleared when a speculative merge is in flight,
        # set by the Verifier when it finishes the item before the speculation.
        self._speculation_slot = asyncio.Event()
        self._speculation_slot.set()  # initially free
        # Internal tasks created by run()
        self._merger_task: asyncio.Task | None = None
        self._verifier_task: asyncio.Task | None = None
        # In-flight request being processed by the merger loop. Set after
        # dequeue, cleared after the SpeculativeItem is pushed to the verifier
        # queue. Used by stop() to resolve Futures for requests that were
        # mid-processing when shutdown was initiated.
        self._inflight_req: MergeRequest | None = None

    # ------------------------------------------------------------------
    # Public API (same interface as MergeWorker)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start merger and verifier coroutines and wait for both to finish."""
        self._merger_task = asyncio.create_task(self._merger_loop())
        self._verifier_task = asyncio.create_task(self._verifier_loop())
        try:
            await asyncio.gather(self._merger_task, self._verifier_task)
        except BaseException:
            for t in (self._merger_task, self._verifier_task):
                if t and not t.done():
                    t.cancel()
            await asyncio.gather(
                self._merger_task, self._verifier_task, return_exceptions=True,
            )
            raise

    async def stop(self) -> None:
        """Graceful shutdown: drain queues and resolve all pending Futures."""
        self._running = False
        shutdown = MergeOutcome('blocked', reason='Merge worker shutting down')
        # Release speculation slot so merger doesn't hang waiting
        self._speculation_slot.set()

        # Drain main queue
        while not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                if req is not None and not req.result.done():
                    req.result.set_result(shutdown)
            except asyncio.QueueEmpty:
                break

        # Drain verifier queue — also clean up orphaned merge worktrees.
        # cleanup_merge_worktree is wrapped in suppress(Exception): filesystem
        # errors during shutdown must not abort the drain loop and leave
        # remaining Futures unresolved (callers would hang forever).
        while not self._verifier_queue.empty():
            try:
                item = self._verifier_queue.get_nowait()
                if item is not None:
                    if item.merge_wt is not None:
                        with contextlib.suppress(Exception):
                            await self._git_ops.cleanup_merge_worktree(item.merge_wt)
                    if not item.request.result.done():
                        item.request.result.set_result(shutdown)
            except asyncio.QueueEmpty:
                break

        # Send sentinels to unblock both loops
        await self._queue.put(None)  # type: ignore[arg-type]
        await self._verifier_queue.put(None)  # type: ignore[arg-type]

        # Allow worker tasks to exit gracefully via sentinels before the
        # harness cancels them, preventing unresolved mid-flight Futures.
        # _shutdown_timeout can be overridden in tests for fast shutdown.
        tasks_to_wait = [
            t for t in (self._merger_task, self._verifier_task)
            if t is not None and not t.done()
        ]
        if tasks_to_wait:
            timeout = getattr(self, '_shutdown_timeout', 5.0)
            await asyncio.wait(tasks_to_wait, timeout=timeout)

        # Re-drain the verifier queue: the merger may have pushed SpeculativeItems
        # after the initial drain above (e.g., after completing its in-flight merge
        # while asyncio.wait() was running). Use the same suppress pattern so
        # cleanup failures don't prevent Future resolution.
        while not self._verifier_queue.empty():
            try:
                item = self._verifier_queue.get_nowait()
                if item is not None:
                    if item.merge_wt is not None:
                        with contextlib.suppress(Exception):
                            await self._git_ops.cleanup_merge_worktree(item.merge_wt)
                    if not item.request.result.done():
                        item.request.result.set_result(shutdown)
            except asyncio.QueueEmpty:
                break

        # Check _inflight_req: if the merger was still blocked inside merge_to_main
        # when asyncio.wait() timed out, it still holds _inflight_req.  Resolve the
        # Future now so the caller doesn't hang forever.
        if self._inflight_req is not None and not self._inflight_req.result.done():
            self._inflight_req.result.set_result(shutdown)

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    def _emit_merge(
        self, task_id: str, outcome: str, *, attempt: int | None = None,
    ) -> None:
        if self._event_store:
            data: dict = {'outcome': outcome}
            if attempt is not None:
                data['attempt'] = attempt
            self._event_store.emit(
                EventType.merge_attempt, task_id=task_id, phase='merge', data=data,
            )

    def _emit_speculative(
        self, event_type: EventType, task_id: str, **data: object,
    ) -> None:
        if self._event_store:
            self._event_store.emit(
                event_type, task_id=task_id, phase='merge',
                data={k: str(v) for k, v in data.items()},
            )

    # ------------------------------------------------------------------
    # Merger coroutine
    # ------------------------------------------------------------------

    async def _merger_loop(self) -> None:
        """Dequeue requests, create merge commits, feed Verifier.

        Tracks a speculative base SHA: after a successful merge of N, the
        Merger grabs N+1 non-blocking and merges it against N's merge commit
        (rather than current main).  Both N and N+1 are pushed to the
        Verifier queue in order.
        """
        # SHA to use as base for the CURRENT request's merge.
        # None  → merge against actual main HEAD (non-speculative).
        # str   → merge against this commit (speculative, N+1 case).
        spec_base: str | None = None
        # Pre-fetched next request grabbed speculatively from main queue.
        prefetched: MergeRequest | None = None

        try:
            while self._running:
                # Get next request: use pre-fetched item if available, else block.
                if prefetched is not None:
                    req = prefetched
                    prefetched = None
                else:
                    req = await self._queue.get()
                    if req is None:
                        break  # shutdown sentinel
                    spec_base = None  # fresh dequeue resets speculation chain

                self._inflight_req = req  # track for stop() race resolution
                try:
                    speculative = spec_base is not None
                    actual_main = await self._git_ops.get_main_sha()
                    base_for_merge = spec_base if spec_base else actual_main

                    # ── Step 1: already-merged detection ──────────────────────
                    rc, branch_head, err = await _run(
                        ['git', 'rev-parse', 'HEAD'], cwd=req.worktree,
                    )
                    if rc != 0:
                        logger.warning(
                            f'Task {req.task_id}: rev-parse HEAD failed: {err.strip()}'
                        )
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=actual_main, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=MergeOutcome(
                                'blocked',
                                reason=f'rev-parse HEAD failed: {err.strip()}',
                            ),
                        ))
                        spec_base = None
                        self._inflight_req = None
                        continue
                    branch_head = branch_head.strip()
                    if await self._git_ops.is_ancestor(branch_head, actual_main) and not await self._git_ops.has_uncommitted_work(req.worktree):
                        logger.info(
                            f'Task {req.task_id}: branch already on main — skipping'
                        )
                        self._emit_merge(req.task_id, 'already_merged')
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=actual_main, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=MergeOutcome('already_merged'),
                        ))
                        spec_base = None
                        self._inflight_req = None
                        continue

                    # ── Step 2: merge (speculative or normal) ─────────────────
                    if speculative:
                        self._emit_speculative(
                            EventType.speculative_merge, req.task_id,
                            base_sha=base_for_merge,
                        )
                    merge_result = await self._git_ops.merge_to_main(
                        req.worktree, req.branch, base_sha=base_for_merge if speculative else None,
                    )

                    # ── Step 3: conflict or non-conflict failure ───────────────
                    if merge_result.conflicts:
                        logger.info(f'Task {req.task_id}: merge conflicts')
                        self._emit_merge(req.task_id, 'conflict')
                        if merge_result.merge_worktree:
                            await self._git_ops.cleanup_merge_worktree(
                                merge_result.merge_worktree,
                            )
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=base_for_merge, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=MergeOutcome(
                                'conflict', conflict_details=merge_result.details,
                            ),
                        ))
                        spec_base = None
                        self._inflight_req = None
                        continue

                    if not merge_result.success:
                        if merge_result.merge_worktree:
                            await self._git_ops.cleanup_merge_worktree(
                                merge_result.merge_worktree,
                            )
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=base_for_merge, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=MergeOutcome(
                                'blocked', reason=merge_result.details,
                            ),
                        ))
                        spec_base = None
                        self._inflight_req = None
                        continue

                    # ── Merge succeeded ────────────────────────────────────────
                    merge_commit = merge_result.merge_commit
                    assert merge_commit is not None
                    merge_commit = merge_commit.strip()

                    skip_verify = (
                        req.pre_rebased
                        and merge_result.pre_merge_sha is not None
                        and merge_result.pre_merge_sha == base_for_merge
                    )
                    await self._verifier_queue.put(SpeculativeItem(
                        request=req, merge_result=merge_result,
                        merge_wt=merge_result.merge_worktree,
                        base_sha=base_for_merge, speculative=speculative,
                        skip_verify=skip_verify,
                    ))
                    self._inflight_req = None  # item is now owned by verifier

                    # ── Speculative look-ahead (depth-1 cap) ──────────────────
                    # Non-blocking peek: if N+1 is already queued, grab it and
                    # merge it against N's commit so the Verifier can CAS it
                    # immediately after N succeeds.
                    await self._speculation_slot.wait()  # depth-1 cap
                    try:
                        next_req = self._queue.get_nowait()
                        if next_req is None:
                            # Shutdown sentinel — stop.
                            break
                        self._speculation_slot.clear()  # claim the slot
                        prefetched = next_req
                        spec_base = merge_commit  # N+1 will merge against N's commit
                        logger.debug(
                            f'Task {req.task_id}: speculative look-ahead for '
                            f'{next_req.task_id} (base={merge_commit[:8]})'
                        )
                    except asyncio.QueueEmpty:
                        spec_base = None  # no next item, no speculation
                except Exception as exc:
                    logger.exception(
                        f'Task {req.task_id}: unexpected merger error: {exc}'
                    )
                    if self._inflight_req is not None and not self._inflight_req.result.done():
                        self._inflight_req.result.set_result(
                            MergeOutcome('blocked', reason=f'Merger error: {exc}')
                        )
                    spec_base = None
                    self._inflight_req = None
        finally:
            # Resolve any in-flight request not yet handed to the verifier.
            # Covers BaseException paths (e.g. CancelledError) that bypass
            # the inner except clause above.
            if self._inflight_req is not None and not self._inflight_req.result.done():
                self._inflight_req.result.set_result(
                    MergeOutcome('blocked', reason='Merge worker shutting down')
                )
            # Always send shutdown sentinel so the verifier exits cleanly,
            # even if an unexpected exception propagates from the loop body.
            await self._verifier_queue.put(None)

    # ------------------------------------------------------------------
    # Verifier coroutine
    # ------------------------------------------------------------------

    async def _verifier_loop(self) -> None:
        """Verify and CAS-advance for each SpeculativeItem from the Merger.

        When N's verification/advance fails and N+1 was speculatively merged,
        the Verifier discards N+1's stale worktree and re-merges it against
        actual main before re-verifying.

        Chain invalidation: if N+1 was re-merged (because N failed), N+2 was
        speculatively built on N+1's stale commit — it must ALSO be re-merged.
        ``remerge_occurred`` propagates this through the chain automatically.
        """
        # True when the previous non-speculative item failed verification
        # or CAS, meaning any following speculative item is invalid.
        n_failed = False
        # True when the previous iteration performed a discard+re-merge.
        # Causes subsequent speculative items to also be discarded and re-merged,
        # because they were built on the stale pre-re-merge commit chain.
        remerge_occurred = False

        while True:
            item = await self._verifier_queue.get()
            if item is None:
                break  # shutdown sentinel

            req = item.request
            # Track whether THIS iteration performs a re-merge so we can
            # propagate the chain-invalidation flag to the next iteration.
            iteration_did_remerge = False

            try:
                # ── Discard stale speculative merge when chain is invalidated ─
                # Two cases: (1) N failed directly (n_failed=True); (2) a prior
                # iteration re-merged, meaning the Merger's spec_base for this
                # item descended from a commit that never reached main.
                if item.speculative and (n_failed or remerge_occurred):
                    # Set flag early so an exception during cleanup/_remerge still
                    # propagates chain invalidation to the next iteration.
                    iteration_did_remerge = True
                    # Clean up the stale merge worktree (merged against a commit
                    # that never reached main).
                    if item.merge_wt:
                        await self._git_ops.cleanup_merge_worktree(item.merge_wt)
                    discard_reason = 'previous_failed' if n_failed else 'chain_invalidated'
                    self._emit_speculative(
                        EventType.speculative_discard, req.task_id,
                        reason=discard_reason,
                    )
                    logger.info(
                        f'Task {req.task_id}: discarding stale speculative merge '
                        f'({discard_reason}), re-merging against actual main'
                    )
                    item = await self._remerge(req)

                # ── Immediate outcome (already_merged / conflict / blocked) ─
                if item.immediate_outcome is not None:
                    if not req.result.done():
                        req.result.set_result(item.immediate_outcome)
                    n_failed = item.immediate_outcome.status not in ('done', 'already_merged')
                    continue  # finally will call _speculation_slot.set()

                n_succeeded = await self._verify_and_advance(item)
                n_failed = not n_succeeded

            except Exception as exc:
                logger.exception(f'Task {req.task_id}: unexpected verifier error')
                if not req.result.done():
                    req.result.set_result(MergeOutcome(
                        'blocked', reason=f'Verifier error: {exc}',
                    ))
                n_failed = True
            except BaseException:
                # CancelledError or other fatal — resolve the in-flight Future
                # and clean up the merge worktree so callers don't hang forever.
                if item.merge_wt is not None:
                    with contextlib.suppress(Exception):
                        await self._git_ops.cleanup_merge_worktree(item.merge_wt)
                if not req.result.done():
                    req.result.set_result(MergeOutcome(
                        'blocked', reason='Merge worker cancelled',
                    ))
                raise
            finally:
                # Propagate chain-invalidation state BEFORE releasing the slot
                # so the Merger's next speculative item sees the updated flag.
                remerge_occurred = iteration_did_remerge
                self._speculation_slot.set()

    async def _remerge(self, req: MergeRequest) -> SpeculativeItem:
        """Re-merge a request against actual main after speculation invalidation."""
        actual_main = await self._git_ops.get_main_sha()
        merge_result = await self._git_ops.merge_to_main(
            req.worktree, req.branch, base_sha=None,
        )
        if merge_result.conflicts:
            self._emit_merge(req.task_id, 'conflict')
            if merge_result.merge_worktree:
                await self._git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
            return SpeculativeItem(
                request=req, merge_result=None, merge_wt=None,
                base_sha=actual_main, speculative=False, skip_verify=False,
                immediate_outcome=MergeOutcome(
                    'conflict', conflict_details=merge_result.details,
                ),
            )
        if not merge_result.success:
            if merge_result.merge_worktree:
                await self._git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
            return SpeculativeItem(
                request=req, merge_result=None, merge_wt=None,
                base_sha=actual_main, speculative=False, skip_verify=False,
                immediate_outcome=MergeOutcome('blocked', reason=merge_result.details),
            )
        skip_verify = (
            req.pre_rebased
            and merge_result.pre_merge_sha is not None
            and merge_result.pre_merge_sha == actual_main
        )
        return SpeculativeItem(
            request=req, merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=actual_main, speculative=False, skip_verify=skip_verify,
        )

    async def _verify_and_advance(self, item: SpeculativeItem) -> bool:
        """Run verification + CAS advance for one item.

        Returns True if the item advanced main successfully, False otherwise.
        Resolves item.request.result in all cases.
        """
        req = item.request
        merge_wt = item.merge_wt
        assert merge_wt is not None
        assert item.merge_result is not None
        merge_commit = item.merge_result.merge_commit
        assert merge_commit is not None
        merge_commit = merge_commit.strip()

        # ── Step 4: verify ────────────────────────────────────────────
        if not item.skip_verify:
            try:
                verify = await run_scoped_verification(
                    merge_wt, req.config, req.module_configs,
                    task_files=req.task_files,
                )
            except Exception as exc:
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                if not req.result.done():
                    req.result.set_result(MergeOutcome(
                        'blocked', reason=f'Verification error: {exc}',
                    ))
                return False

            if not verify.passed:
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                if not req.result.done():
                    req.result.set_result(MergeOutcome(
                        'blocked',
                        reason=f'Post-merge verification failed: {verify.summary}',
                    ))
                return False
        else:
            logger.info(
                f'Task {req.task_id}: skipping re-verification '
                f'(pre-rebased, main unchanged)'
            )

        # ── Step 5: CAS advance_main ──────────────────────────────────
        retries = 0
        while True:
            result = await self._git_ops.advance_main(
                merge_commit, merge_wt,
                branch=req.branch,
                max_attempts=req.config.max_advance_attempts,
                expected_main=item.base_sha,
            )

            if result == 'advanced':
                self._cas_retries.pop(req.task_id, None)
                logger.info(f'Task {req.task_id}: merged to main successfully')
                self._emit_merge(req.task_id, 'done')
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                if not req.result.done():
                    req.result.set_result(MergeOutcome('done'))
                return True

            if result in ('not_descendant', 'contaminated', 'stash_failed'):
                self._cas_retries.pop(req.task_id, None)
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                if not req.result.done():
                    req.result.set_result(MergeOutcome(
                        'blocked',
                        reason=f'advance_main failed ({result}) for task {req.task_id}',
                    ))
                return False

            # result == 'cas_failed' — transient, retry with limit
            retries += 1
            total = self._cas_retries.get(req.task_id, 0) + 1
            self._cas_retries[req.task_id] = total
            if total > self.MAX_CAS_RETRIES:
                self._cas_retries.pop(req.task_id, None)
                logger.warning(
                    f'Task {req.task_id}: CAS retry limit exhausted '
                    f'({self.MAX_CAS_RETRIES} attempts)'
                )
                self._emit_merge(req.task_id, 'cas_exhausted', attempt=total)
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                if not req.result.done():
                    req.result.set_result(MergeOutcome(
                        'blocked',
                        reason=(
                            f'CAS retry limit exhausted after '
                            f'{self.MAX_CAS_RETRIES} attempts for task {req.task_id}'
                        ),
                    ))
                return False

            # Update base_sha to current main for retry
            item = SpeculativeItem(
                request=item.request,
                merge_result=item.merge_result,
                merge_wt=item.merge_wt,
                base_sha=await self._git_ops.get_main_sha(),
                speculative=item.speculative,
                skip_verify=item.skip_verify,
            )
            logger.info(
                f'Task {req.task_id}: CAS failed (attempt {total}/'
                f'{self.MAX_CAS_RETRIES}), retrying'
            )
            self._emit_merge(req.task_id, 'cas_retry', attempt=total)
