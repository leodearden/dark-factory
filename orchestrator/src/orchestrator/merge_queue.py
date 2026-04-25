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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from orchestrator.artifacts import TaskArtifacts
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.verify import run_scoped_verification

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig

logger = logging.getLogger(__name__)


@dataclass
class UnresolvedStep:
    """A done plan-step whose ``git diff-tree`` query returned rc != 0.

    Recorded on :class:`DropGuardResult` when the drop-guard cannot fully
    determine whether a planned-file deletion was intentional.  The
    ``object_missing`` flag distinguishes a pruned/never-existed object
    (``True``) from some other transient failure (``False``), giving the
    steward the data needed to investigate the next misfire.
    """

    step_idx: int
    step_id: str | None
    commit: str
    rc: int
    stderr: str
    object_missing: bool


@dataclass
class DropGuardResult:
    """Structured return value from :func:`_check_plan_targets_in_tree`.

    Replaces the previous ``list[str]`` return so that unresolved-step
    diagnostics can be propagated to the ``MergeOutcome.reason`` text
    without side channels.

    Attributes:
        dropped: Plan-file paths that are absent from the merge commit and
            could not be explained by a trusted done-step deletion.  Empty
            list means no drops detected.
        unresolved_steps: Done plan-steps for which the ``git diff-tree``
            probe returned rc != 0.  A non-empty list means the guard may
            have missed some expected-absent files — the steward should
            investigate whether a real deletion was silently discarded.
    """

    dropped: list[str] = field(default_factory=list)
    unresolved_steps: list[UnresolvedStep] = field(default_factory=list)


async def _check_plan_targets_in_tree(
    merge_commit_sha: str,
    task_worktree: Path,
    git_ops: GitOps,
    *,
    task_id: str | None = None,
) -> DropGuardResult:
    """Return a :class:`DropGuardResult` describing plan files dropped by the merge.

    A "drop" means the file was on the task branch tip but is absent from
    the merge commit — i.e. conflict resolution discarded work the task
    actually produced. Paths that are absent from *both* task HEAD and the
    merge commit are not drops: the task branch itself intentionally
    doesn't carry them (e.g. a plan step that adds a scaffold and a later
    step that deletes it per review). Flagging those would trip on plans
    whose terminal state legitimately removes a listed file.

    Reads the plan from the task worktree (plan.json lives in gitignored
    .task/, so it's only in the source worktree — not the merge worktree).

    Returns a DropGuardResult with empty ``dropped`` when:
    - no plan.json exists (architect never ran — nothing to check)
    - plan['files'] is empty or missing
    - every planned file is present in the merge commit, or is explained
      by a trusted done-step deletion, or (only when plan has no structured
      steps) is absent from task HEAD as well.
    """
    artifacts = TaskArtifacts(task_worktree)
    plan = artifacts.read_plan()
    files = plan.get('files') or []
    if not files:
        return DropGuardResult()

    # Narrow-against-task-HEAD filter only applies when plan has no
    # structured steps. When steps exist, the done-step cross-reference below
    # is the authoritative intent signal: files absent from task HEAD without
    # a trusted done-step deletion must fail closed as real drops rather than
    # be silently dismissed by the narrow filter.
    steps_raw = plan.get('steps') or []
    has_structured_steps = any(isinstance(s, dict) for s in steps_raw)

    if has_structured_steps:
        task_head = ''
    else:
        rc_head, task_head_out, head_err = await _run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=task_worktree,
        )
        if rc_head == 0:
            task_head = task_head_out.strip()
        else:
            # rev-parse HEAD on a valid worktree essentially always succeeds;
            # a non-zero rc signals an unexpected problem (corrupt worktree,
            # detached state with no HEAD). Log visibly so regressions surface
            # in operations rather than as mystery false positives — with an
            # empty task_head the loop below falls back to the pre-narrow
            # behaviour of flagging every missing planned file as a drop.
            logger.warning(
                'drop-guard: git rev-parse HEAD failed in %s (rc=%d, stderr=%s); '
                'falling back to flag-every-missing behaviour. '
                'task_id=%s merge_commit_sha=%s',
                task_worktree, rc_head, head_err.strip(),
                task_id or '<unknown>', merge_commit_sha,
            )
            task_head = ''

    missing: list[str] = []
    for f in files:
        rc_merge, _, _ = await _run(
            ['git', 'cat-file', '-e', f'{merge_commit_sha}:{f}'],
            cwd=git_ops.project_root,
        )
        if rc_merge == 0:
            continue
        # When structured steps are present, skip the narrow filter — the
        # done-step cross-reference below decides expected-absence.
        if has_structured_steps:
            missing.append(f)
            continue
        if not task_head:
            missing.append(f)
            continue
        rc_head_f, _, _ = await _run(
            ['git', 'cat-file', '-e', f'{task_head}:{f}'],
            cwd=git_ops.project_root,
        )
        if rc_head_f == 0:
            missing.append(f)

    if not missing:
        return DropGuardResult()

    # Cross-reference missing files against intentional deletions performed
    # by done plan-steps.  A file absent from the merge tree but deleted by
    # a done step is "expected absent" and should NOT be flagged as a drop.
    #
    # Uses `git diff-tree --root -m --no-commit-id --no-renames -r <sha>`
    # rather than `git show` or the two-arg `<sha>^ <sha>` form to:
    #   (a) surface deletions in merge commits: `git show` defaults to
    #       combined-diff which silences deletions; the two-arg `<sha>^ <sha>`
    #       form forces a two-way diff against the first parent but fails for
    #       root commits; `-m` shows per-parent diffs for merge commits so
    #       files deleted from any parent are captured, and
    #   (b) disable rename detection so renamed plan files correctly surface
    #       as expected_absent (git show with renames converts D+A to R,
    #       which --diff-filter=D does not match).
    #   (c) `--root` makes root commits (no parents) diff against the empty
    #       tree rather than failing — they can't delete files, so the output
    #       is empty and the step is silently skipped.
    #   (d) `--no-commit-id` suppresses the leading SHA line that
    #       `git diff-tree <commit>` (single-arg form) emits per parent.
    #
    # All diff-tree queries run concurrently via asyncio.gather — each is
    # independent and the sequential loop would otherwise scale O(N) in
    # done-step count on the merge hot-path.  Multiple done steps may point
    # at the same commit SHA; queries are deduplicated before gather to avoid
    # redundant subprocess launches.
    #
    # Trade-off: if a done-step commit is itself a merge commit that dropped
    # a planned file during conflict resolution, `-m` surfaces that deletion
    # against at least one parent and we silence it as expected_absent — the
    # exact false-negative this guard was designed to catch.  In the current
    # flow done-step commits are produced by individual coding agents and are
    # overwhelmingly unlikely to be merge commits; if that ever changes, gate
    # `-m` on `git rev-list --parents -n 1 <sha> | wc -w > 2` and skip
    # done-step commits that are themselves merges.

    # ── Orphaned-commit failure mode (task 1068) ─────────────────────────────
    #
    # Background (task 1058 escalations esc-1058-170..173):
    #   After `git commit --amend`, the pre-amend SHA is orphaned — it is no
    #   longer reachable from any branch ref — but it remains queryable via the
    #   shared ODB until the next `git gc` / pack prune runs.  If the plan.json
    #   records the pre-amend SHA as a done-step commit (because the plan was
    #   written before the amend happened), `git diff-tree` can still resolve
    #   the deletion stored in that orphan commit.  This is the HAPPY PATH:
    #   orphan-but-not-pruned is fine and does not trigger a false positive.
    #
    # Latent failure mode:
    #   If `git diff-tree` returns rc != 0 for a done-step commit for ANY
    #   reason (ODB lock during pack rewrite, environment anomaly, host
    #   contention, or the commit has been pruned after `git gc`), the
    #   silent `continue` below drops that step's expected-absent contribution.
    #   Any file the step legitimately deleted is then mis-flagged as a real
    #   drop — the original symptom in task 1058 was
    #   `orchestrator/tests/test_conftest_helpers.py` being flagged as dropped.
    #
    # Rejected alternatives:
    #   (a) Search-by-commit-message / step-id: fragile — depends on message
    #       conventions not enforced in the codebase; could match wrong commits
    #       and would mask real bugs by silently substituting similar commits.
    #   (b) Rewrite plan-step commit SHAs after amend: iteration-loop concern
    #       explicitly noted in task 1068 as out of scope for the drop-guard.
    #
    # Diagnostic strategy (implemented below):
    #   When diff-tree fails, run `git cat-file -e <sha>^{commit}` to determine
    #   whether the object is missing/pruned (rc != 0 → object_missing=True) or
    #   failed for a different reason (rc == 0 → some other transient error).
    #   Record the failure as an UnresolvedStep on the returned DropGuardResult
    #   so the merge worker can surface it in the MergeOutcome.reason and the
    #   steward can distinguish the two failure modes.  See task 1068 for full
    #   rationale.
    # ─────────────────────────────────────────────────────────────────────────

    # Pass 1: collect (step_idx, commit) for every qualifying done step.
    steps_to_query: list[tuple[int, str]] = []
    for step_idx, step in enumerate(plan.get('steps') or []):
        if not isinstance(step, dict):
            continue
        if step.get('status') != 'done':
            continue
        commit = step.get('commit')
        if not isinstance(commit, str) or not commit:
            continue
        steps_to_query.append((step_idx, commit))

    # Pass 2: fire all diff-tree queries concurrently, then process results.
    # Deduplicate commit SHAs (preserving first-occurrence order) so each
    # unique SHA is queried only once even when multiple done steps share it.
    expected_absent: set[str] = set()
    # Build a mapping of step info for diagnostics (step_idx → step dict)
    step_info: dict[int, dict] = {
        step_idx: step
        for step_idx, step in enumerate(plan.get('steps') or [])
        if isinstance(step, dict)
    }
    unresolved: list[UnresolvedStep] = []
    if steps_to_query:
        unique_commits: list[str] = list(dict.fromkeys(c for _, c in steps_to_query))
        commit_results = await asyncio.gather(*[
            _run(
                [
                    'git', 'diff-tree',
                    '--root',
                    '-m',
                    '--no-commit-id',
                    '--no-renames',
                    '--diff-filter=D',
                    '--name-only',
                    '-r',
                    commit,
                ],
                cwd=git_ops.project_root,
            )
            for commit in unique_commits
        ])
        commit_to_result: dict[str, tuple[int, str, str]] = dict(
            zip(unique_commits, commit_results, strict=True),
        )
        for step_idx, commit in steps_to_query:
            rc, stdout, stderr = commit_to_result[commit]
            if rc != 0:
                logger.warning(
                    'git diff-tree --diff-filter=D failed for step %d commit %s: %s',
                    step_idx, commit, stderr.strip(),
                )
                # Probe whether the failure is due to a missing/pruned object
                # or some other transient error (ODB lock, env issue, etc.).
                # `git cat-file -e <sha>^{commit}` exits 0 if the object exists
                # as a commit, non-zero if it is absent from the ODB.
                rc_cat, _, _ = await _run(
                    ['git', 'cat-file', '-e', f'{commit}^{{commit}}'],
                    cwd=git_ops.project_root,
                )
                unresolved.append(UnresolvedStep(
                    step_idx=step_idx,
                    step_id=step_info.get(step_idx, {}).get('id'),
                    commit=commit,
                    rc=rc,
                    stderr=stderr[:500],
                    object_missing=(rc_cat != 0),
                ))
                continue
            for line in stdout.splitlines():
                path = line.strip()
                if path:
                    expected_absent.add(path)
                    logger.debug(
                        'File %s expected absent (deleted in step %d: %s)',
                        path, step_idx, commit,
                    )

    return DropGuardResult(
        dropped=[f for f in missing if f not in expected_absent],
        unresolved_steps=unresolved,
    )


ABANDONED_REASON_PREFIX = 'Post-merge verify timed out'
"""Prefix of the ``MergeOutcome.reason`` string emitted by the merge-queue
loop-breaker.  Downstream classifiers (task steward, dashboard) use this to
recognise a task that has been abandoned after repeated post-merge verify
timeouts rather than a first-time verify failure.  Kept as a module-level
constant so tests and any future callers share a single source of truth."""


def _elapsed_ms(start: float | None) -> int | None:
    """Milliseconds since *start* (a ``time.monotonic()`` value).

    Returns ``None`` when *start* is ``None`` so callers can safely forward
    the result to ``event_store.emit(duration_ms=...)`` without special-casing.
    """
    if start is None:
        return None
    return round((time.monotonic() - start) * 1000)


def _emit_merge_attempt(
    event_store: EventStore | None,
    task_id: str,
    outcome: str,
    *,
    attempt: int | None = None,
    duration_ms: int | None = None,
) -> None:
    """Emit a ``merge_attempt`` event for the given outcome.

    Note: certain terminal outcomes are intentionally NOT emitted here —
    specifically ``blocked`` outcomes from ``not merge_result.success`` paths
    (e.g. merge infrastructure failures unrelated to conflicts) and from
    ``advance_main`` non-CAS failure codes (``not_descendant``, ``contaminated``,
    ``stash_failed``).  These are rare infrastructure errors rather than
    normal merge-latency outcomes and omitting them keeps dashboard latency
    percentiles free of unbounded outliers from external failures.

    ``blocked`` outcomes that carry a specific diagnostic outcome code
    (e.g. ``dropped_plan_targets``, ``cas_exhausted``) ARE emitted here;
    only ``blocked`` outcomes from infrastructure failures are not.
    """
    if event_store is not None:
        data: dict = {'outcome': outcome}
        if attempt is not None:
            data['attempt'] = attempt
        event_store.emit(
            EventType.merge_attempt, task_id=task_id, phase='merge',
            data=data, duration_ms=duration_ms,
        )


def _emit_merge_queued(
    event_store: EventStore | None,
    req: MergeRequest,
    reason: str | None = None,
) -> None:
    """Emit a merge_queued event.  No-op when *event_store* is None.

    Centralises the emit payload so both :func:`enqueue_merge_request` and
    the ``MergeWorker`` CAS-retry path use an identical record shape.  If
    *reason* is provided (e.g. ``'cas_retry'``) it is stored in ``data``.
    """
    if event_store is None:
        return
    data: dict = {'branch': req.branch}
    if reason is not None:
        data['reason'] = reason
    event_store.emit(
        EventType.merge_queued,
        task_id=req.task_id,
        phase='merge',
        data=data,
    )


async def enqueue_merge_request(
    queue: asyncio.Queue,
    req: MergeRequest,
    event_store: EventStore | None,
) -> None:
    """Enqueue a MergeRequest and emit a merge_queued event.

    Puts the request on *queue* first so that a cancellation between put and
    emit (or any emit error) does not leave a dangling ``merge_queued`` row
    with no corresponding worker pickup.  Losing the event is less confusing
    than a stale "queued" row that persists until the TTL expires.

    If ``event_store`` is None the request is still enqueued; emission is
    silently skipped (mirrors the None-safe pattern used by
    ``_emit_merge_attempt``).
    """
    await queue.put(req)
    _emit_merge_queued(event_store, req)


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

    status: Literal['done', 'conflict', 'blocked', 'already_merged', 'wip_halted', 'done_wip_recovery', 'wip_recovery_no_advance', 'unmerged_state']
    reason: str = ''
    conflict_details: str = ''
    recovery_branch: str | None = None
    overlap_files: list[str] | None = None
    merge_sha: str | None = None
    push_status: str | None = None


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
    started_monotonic: float | None = None  # time.monotonic() at entry; None → unset, _elapsed_ms returns None


class MergeWorker:
    """Single coroutine that processes merge requests serially.

    Owns all main-branch advancement via CAS ``update-ref``.  The harness
    creates one instance and passes the same ``asyncio.Queue`` to every
    ``TaskWorkflow``.
    """

    MAX_CAS_RETRIES = 5
    # After this many consecutive post-merge verify TIMEOUTS for the same
    # task, the merge queue stops trying and returns an 'abandoned' blocked
    # outcome.  Caps the verify-timeout / re-enqueue oscillation (two tasks
    # alternating on the merge queue for hours, each dying at the 30-min
    # warm timeout).  Counter resets on any successful merge for that task.
    MAX_POST_MERGE_VERIFY_TIMEOUTS = 2

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
        # WIP halt: cleared when halted, set when running
        self._wip_halt = asyncio.Event()
        self._wip_halt.set()  # not halted initially
        # ID of the escalation that owns the current halt. Registered by the
        # workflow handler after it submits the L1 escalation. Single source
        # of truth for the resolve-callback un-halt path.
        self._halt_owner_esc_id: str | None = None

    def _abandon_outcome(self, task_id: str, count: int) -> MergeOutcome:
        """Build the terminal MergeOutcome for the loop-breaker.

        Kept as a method so tests can assert against the reason string via a
        single source.  Uses ``ABANDONED_REASON_PREFIX`` so downstream
        classifiers (task steward, dashboard) can recognise the outcome.
        """
        return MergeOutcome(
            'blocked',
            reason=(
                f'{ABANDONED_REASON_PREFIX} {count} times for task '
                f'{task_id} — manual investigation required. '
                'The merge queue has stopped retrying this task to avoid '
                'starving the queue behind a deterministic verify hang.'
            ),
        )

    def halt_for_wip(self, reason: str) -> None:
        """Halt the merge queue due to a WIP conflict."""
        logger.warning('Merge queue halted for WIP: %s', reason)
        self._wip_halt.clear()
        self._halt_owner_esc_id = None

    def set_halt_owner(self, esc_id: str) -> None:
        """Register the escalation that owns the current halt.

        The workflow calls this right after submitting its halt-triggering
        escalation. Asserts owner is currently None — a double-register
        indicates a double-halt bug that should fail loudly.
        """
        assert self._halt_owner_esc_id is None, (
            f'halt owner already set to {self._halt_owner_esc_id!r}, '
            f'refusing to overwrite with {esc_id!r}'
        )
        self._halt_owner_esc_id = esc_id

    def is_halt_owner(self, esc_id: str) -> bool:
        """True iff esc_id is the currently registered halt owner."""
        return (
            self._halt_owner_esc_id is not None
            and self._halt_owner_esc_id == esc_id
        )

    def unhalt_wip(self) -> None:
        """Resume the merge queue after WIP conflict resolution."""
        logger.info('Merge queue un-halted (WIP conflict resolved)')
        self._wip_halt.set()
        self._halt_owner_esc_id = None

    @property
    def is_wip_halted(self) -> bool:
        return not self._wip_halt.is_set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop — runs until ``stop()`` is called."""
        while self._running:
            await self._wip_halt.wait()  # blocks if halted for WIP conflict
            req = await self._dequeue()
            if req is None:
                break  # shutdown sentinel

            if self._event_store is not None:
                self._event_store.emit(
                    EventType.merge_dequeued,
                    task_id=req.task_id,
                    phase='merge',
                    data={'branch': req.branch},
                )

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
                _emit_merge_attempt(self._event_store, req.task_id, 'already_merged', duration_ms=_elapsed_ms(t0))
                return MergeOutcome('already_merged')

        # 2. Merge in a temporary worktree
        merge_result = await self._git_ops.merge_to_main(
            req.worktree, req.branch,
        )

        # 3. Conflict → reject immediately (caller resolves outside queue)
        if merge_result.conflicts:
            logger.info(f'Task {req.task_id}: merge conflicts detected')
            _emit_merge_attempt(self._event_store, req.task_id, 'conflict', duration_ms=_elapsed_ms(t0))
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

        # 3b. Drop-guard: every file the task planned must survive the merge.
        # Catches "accept origin" conflict resolutions that silently drop
        # planned work from the task branch.
        assert merge_result.merge_commit is not None
        drop_result = await _check_plan_targets_in_tree(
            merge_result.merge_commit, req.worktree, self._git_ops,
            task_id=req.task_id,
        )
        dropped = drop_result.dropped
        if dropped:
            if merge_result.merge_worktree:
                await self._git_ops.cleanup_merge_worktree(
                    merge_result.merge_worktree,
                )
            logger.warning(
                f'Task {req.task_id}: merge dropped plan targets: {dropped}'
            )
            _emit_merge_attempt(self._event_store, req.task_id, 'dropped_plan_targets', duration_ms=_elapsed_ms(t0))
            return MergeOutcome(
                'blocked',
                reason=(
                    f'Merge commit is missing plan target files: '
                    f'{", ".join(dropped)}. '
                    f'Conflict resolution likely dropped planned work. '
                    f'Review the merge commit and restore missing files.'
                ),
            )

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
            # max_retries=0: post-merge verify hangs are usually deterministic
            # (e.g. a deadlocked test); retrying just multiplies queue-wide stall.
            # is_merge_verify=True: merge worktrees are freshly created per
            # merge (no `.task/` dir and no warm cargo cache), so they need
            # the cold timeout despite `_is_verify_cold`'s filesystem
            # heuristic classifying them as warm.
            verify = await run_scoped_verification(
                merge_wt, req.config, req.module_configs,
                task_files=req.task_files,
                max_retries=0,
                is_merge_verify=True,
            )
            if not verify.passed:
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                detail = verify.failure_report()
                reason = f'Post-merge verification failed: {verify.summary}'
                if detail:
                    reason = f'{reason}\n\n{detail}'
                # Loop-breaker bookkeeping: bump only when the failure was a
                # pure timeout.  Real test/lint/type failures already bubble
                # up to the steward and don't drive the re-queue oscillation
                # the loop-breaker is designed to catch.
                if verify.timed_out:
                    new_count = prior_timeouts + 1
                    self._post_merge_verify_timeouts[req.task_id] = new_count
                    if new_count >= self.MAX_POST_MERGE_VERIFY_TIMEOUTS:
                        logger.warning(
                            'Task %s: post-merge verify timed out %d times in a '
                            'row — next submission will be abandoned',
                            req.task_id, new_count,
                        )
                return MergeOutcome('blocked', reason=reason)

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
            # Loop-breaker counter: a successful merge means whatever caused
            # the earlier timeouts has cleared (e.g. test was flaky, host
            # contention eased).  Reset so future timeouts start from 0.
            self._post_merge_verify_timeouts.pop(req.task_id, None)
            logger.info(f'Task {req.task_id}: merged to main successfully')
            _emit_merge_attempt(self._event_store, req.task_id, 'done', duration_ms=_elapsed_ms(t0))
            push_status = await self._git_ops.push_main()
            return MergeOutcome('done', merge_sha=merge_result.merge_commit, push_status=push_status)

        if result in ('wip_overlap', 'pop_conflict'):
            # Halt the queue globally — no more merges until resolved
            self.halt_for_wip(f'advance_main: {result}')
            if result == 'pop_conflict':
                # Main was advanced — push origin even though stash pop failed.
                push_status = await self._git_ops.push_main()
                recovery = getattr(self._git_ops, '_last_recovery_branch', None)
                return MergeOutcome(
                    'done_wip_recovery',
                    reason=f'Merge advanced but stash pop conflicted. Recovery branch: {recovery}',
                    recovery_branch=recovery,
                    push_status=push_status,
                )
            else:
                overlap = getattr(self._git_ops, '_last_overlap_files', None)
                return MergeOutcome(
                    'wip_halted',
                    reason=f'WIP overlaps merge diff: {", ".join(overlap or [])}',
                    overlap_files=overlap,
                )

        if result == 'unmerged_state':
            # Permanent block — pre-existing UU markers in project_root.
            # Halt the queue and route to human escalation (not steward).
            self.halt_for_wip(
                'advance_main: unmerged_state — project_root has unresolved merge '
                'conflicts. Manual investigation required before any retry.'
            )
            self._cas_retries.pop(req.task_id, None)
            return MergeOutcome(
                'unmerged_state',
                reason=(
                    f'advance_main returned unmerged_state: project_root has '
                    f'unresolved (UU/AA/DD) merge conflicts — halting queue; '
                    f'manual investigation required before any retry. '
                    f'(task {req.task_id})'
                ),
            )

        if result == 'pop_conflict_no_advance':
            # Stash pop conflicted during CAS-failure recovery — merge did NOT land.
            # Halt queue and return distinct outcome for human-level escalation.
            self.halt_for_wip('advance_main: pop_conflict_no_advance')
            recovery = getattr(self._git_ops, '_last_recovery_branch', None)
            self._cas_retries.pop(req.task_id, None)
            return MergeOutcome(
                'wip_recovery_no_advance',
                reason=(
                    f'Merge did not advance AND WIP stash pop conflicted. '
                    f'Recovery branch: {recovery}. '
                    f'Manual intervention required — do not retry automatically. '
                    f'(task {req.task_id})'
                ),
                recovery_branch=recovery,
            )

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
        _emit_merge_queued(self._event_store, req, reason='cas_retry')
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
    # Mirror of MergeWorker.MAX_POST_MERGE_VERIFY_TIMEOUTS — see that class
    # for rationale.  Kept as a class attribute so tests can monkeypatch
    # per-class if the two workers ever diverge.
    MAX_POST_MERGE_VERIFY_TIMEOUTS = 2

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
        # Per-task consecutive post-merge-verify-timeout counter.  Bumped by
        # the Verifier when a post-merge verify finishes with timed_out=True,
        # cleared on a successful CAS advance.  Keyed by task_id; lives
        # across submissions so an orchestrator re-queue of the same task
        # continues to feed the same counter.
        self._post_merge_verify_timeouts: dict[str, int] = {}
        # Depth-1 cap: cleared when a speculative merge is in flight,
        # set by the Verifier when it finishes the item before the speculation.
        self._speculation_slot = asyncio.Event()
        self._speculation_slot.set()  # initially free
        # WIP halt: cleared when halted, set when running
        self._wip_halt = asyncio.Event()
        self._wip_halt.set()  # not halted initially
        # ID of the escalation that owns the current halt. Registered by the
        # workflow handler after it submits the L1 escalation. Single source
        # of truth for the resolve-callback un-halt path.
        self._halt_owner_esc_id: str | None = None
        # Internal tasks created by run()
        self._merger_task: asyncio.Task | None = None
        self._verifier_task: asyncio.Task | None = None
        # In-flight request being processed by the merger loop. Set after
        # dequeue, cleared after the SpeculativeItem is pushed to the verifier
        # queue. Used by stop() to resolve Futures for requests that were
        # mid-processing when shutdown was initiated.
        self._inflight_req: MergeRequest | None = None
        # Can be overridden in tests for fast shutdown (see stop()).
        self._shutdown_timeout: float = 5.0

    def _abandon_outcome(self, task_id: str, count: int) -> MergeOutcome:
        """Build the terminal MergeOutcome for the loop-breaker.

        Mirror of ``MergeWorker._abandon_outcome`` — kept in sync so
        downstream classifiers (steward, dashboard) see the same reason
        prefix regardless of which worker served the request.
        """
        return MergeOutcome(
            'blocked',
            reason=(
                f'{ABANDONED_REASON_PREFIX} {count} times for task '
                f'{task_id} — manual investigation required. '
                'The merge queue has stopped retrying this task to avoid '
                'starving the queue behind a deterministic verify hang.'
            ),
        )

    # ------------------------------------------------------------------
    # Public API (same interface as MergeWorker)
    # ------------------------------------------------------------------

    def halt_for_wip(self, reason: str) -> None:
        """Halt the merge queue due to a WIP conflict."""
        logger.warning('Merge queue halted for WIP: %s', reason)
        self._wip_halt.clear()
        self._halt_owner_esc_id = None

    def set_halt_owner(self, esc_id: str) -> None:
        """Register the escalation that owns the current halt.

        The workflow calls this right after submitting its halt-triggering
        escalation. Asserts owner is currently None — a double-register
        indicates a double-halt bug that should fail loudly.
        """
        assert self._halt_owner_esc_id is None, (
            f'halt owner already set to {self._halt_owner_esc_id!r}, '
            f'refusing to overwrite with {esc_id!r}'
        )
        self._halt_owner_esc_id = esc_id

    def is_halt_owner(self, esc_id: str) -> bool:
        """True iff esc_id is the currently registered halt owner."""
        return (
            self._halt_owner_esc_id is not None
            and self._halt_owner_esc_id == esc_id
        )

    def unhalt_wip(self) -> None:
        """Resume the merge queue after WIP conflict resolution."""
        logger.info('Merge queue un-halted (WIP conflict resolved)')
        self._wip_halt.set()
        self._halt_owner_esc_id = None

    @property
    def is_wip_halted(self) -> bool:
        return not self._wip_halt.is_set()

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
        # Release speculation slot and WIP halt so merger doesn't hang waiting
        self._speculation_slot.set()
        self._wip_halt.set()

        # Drain main queue
        while not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                if req is not None and not req.result.done():
                    req.result.set_result(shutdown)
            except asyncio.QueueEmpty:
                break

        # Drain verifier queue — also clean up orphaned merge worktrees.
        # cleanup_merge_worktree is wrapped in suppress(BaseException) so that
        # CancelledError mid-drain (cancellation is propagating from SIGTERM)
        # does not abort the drain loop and leave remaining Futures unresolved
        # (callers would hang forever) and leaked merge worktrees on disk.
        while not self._verifier_queue.empty():
            try:
                item = self._verifier_queue.get_nowait()
                if item is not None:
                    if item.merge_wt is not None:
                        with contextlib.suppress(BaseException):
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
            timeout = self._shutdown_timeout
            await asyncio.wait(tasks_to_wait, timeout=timeout)

        # Re-drain the verifier queue: the merger may have pushed SpeculativeItems
        # after the initial drain above (e.g., after completing its in-flight merge
        # while asyncio.wait() was running). Use the same suppress(BaseException)
        # pattern so cleanup failures (including CancelledError mid-cleanup) don't
        # prevent Future resolution.
        while not self._verifier_queue.empty():
            try:
                item = self._verifier_queue.get_nowait()
                if item is not None:
                    if item.merge_wt is not None:
                        with contextlib.suppress(BaseException):
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

    def _emit_speculative(
        self, event_type: EventType, task_id: str, **data: object,
    ) -> None:
        # Stays a method (not _emit_merge_attempt) because it emits
        # speculative-specific event types — not generic merge_attempt rows.
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
                await self._wip_halt.wait()  # blocks if halted for WIP conflict
                # Get next request: use pre-fetched item if available, else block.
                if prefetched is not None:
                    req = prefetched
                    prefetched = None
                else:
                    req = await self._queue.get()
                    if req is None:
                        break  # shutdown sentinel
                    spec_base = None  # fresh dequeue resets speculation chain
                    # Re-check halt after blocking on queue.get() — the halt
                    # may have been triggered while we were waiting.
                    await self._wip_halt.wait()

                self._inflight_req = req  # track for stop() race resolution
                if self._event_store is not None:
                    self._event_store.emit(
                        EventType.merge_dequeued,
                        task_id=req.task_id,
                        phase='merge',
                        data={'branch': req.branch},
                    )
                t0 = time.monotonic()
                merge_result_local: MergeResult | None = None
                try:
                    speculative = spec_base is not None
                    actual_main = await self._git_ops.get_main_sha()
                    base_for_merge = spec_base if spec_base else actual_main

                    # ── Step 0: loop-breaker short-circuit ────────────────────
                    # If this task has already timed out in post-merge verify
                    # MAX_POST_MERGE_VERIFY_TIMEOUTS times in a row, abandon
                    # without doing any git work.  The outcome rides through
                    # the verifier queue as an ``immediate_outcome`` so the
                    # usual resolution path (including speculation bookkeeping
                    # via ``n_failed``) stays consistent.
                    prior_timeouts = self._post_merge_verify_timeouts.get(req.task_id, 0)
                    if prior_timeouts >= self.MAX_POST_MERGE_VERIFY_TIMEOUTS:
                        logger.warning(
                            'Task %s: abandoning merge — %d consecutive '
                            'post-merge verify timeouts (threshold=%d)',
                            req.task_id, prior_timeouts,
                            self.MAX_POST_MERGE_VERIFY_TIMEOUTS,
                        )
                        _emit_merge_attempt(
                            self._event_store, req.task_id,
                            'abandoned_verify_timeouts',
                            attempt=prior_timeouts, duration_ms=_elapsed_ms(t0),
                        )
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=actual_main, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=self._abandon_outcome(
                                req.task_id, prior_timeouts,
                            ),
                            started_monotonic=t0,
                        ))
                        spec_base = None
                        self._inflight_req = None
                        continue

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
                            started_monotonic=t0,
                        ))
                        spec_base = None
                        self._inflight_req = None
                        continue
                    branch_head = branch_head.strip()
                    if await self._git_ops.is_ancestor(branch_head, actual_main) and not await self._git_ops.has_uncommitted_work(req.worktree):
                        logger.info(
                            f'Task {req.task_id}: branch already on main — skipping'
                        )
                        _emit_merge_attempt(self._event_store, req.task_id, 'already_merged', duration_ms=_elapsed_ms(t0))
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=actual_main, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=MergeOutcome('already_merged'),
                            started_monotonic=t0,
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
                    merge_result_local = merge_result  # track for cleanup on post-merge exception

                    # ── Step 3: conflict or non-conflict failure ───────────────
                    if merge_result.conflicts:
                        logger.info(f'Task {req.task_id}: merge conflicts')
                        _emit_merge_attempt(self._event_store, req.task_id, 'conflict', duration_ms=_elapsed_ms(t0))
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
                            started_monotonic=t0,
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
                            started_monotonic=t0,
                        ))
                        spec_base = None
                        self._inflight_req = None
                        continue

                    # ── Merge succeeded ────────────────────────────────────────
                    merge_commit = merge_result.merge_commit
                    assert merge_commit is not None
                    merge_commit = merge_commit.strip()

                    # Drop-guard: every file the task planned must survive.
                    drop_result = await _check_plan_targets_in_tree(
                        merge_commit, req.worktree, self._git_ops,
                        task_id=req.task_id,
                    )
                    dropped = drop_result.dropped
                    if dropped:
                        if merge_result.merge_worktree:
                            await self._git_ops.cleanup_merge_worktree(
                                merge_result.merge_worktree,
                            )
                        logger.warning(
                            f'Task {req.task_id}: merge dropped plan '
                            f'targets: {dropped}'
                        )
                        _emit_merge_attempt(self._event_store, req.task_id, 'dropped_plan_targets', duration_ms=_elapsed_ms(t0))
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=base_for_merge, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=MergeOutcome(
                                'blocked',
                                reason=(
                                    f'Merge commit is missing plan target '
                                    f'files: {", ".join(dropped)}. '
                                    f'Conflict resolution likely dropped '
                                    f'planned work. Review the merge commit '
                                    f'and restore missing files.'
                                ),
                            ),
                            started_monotonic=t0,
                        ))
                        spec_base = None
                        self._inflight_req = None
                        continue

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
                        started_monotonic=t0,
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
                    # Clean up any merge worktree created by merge_to_main
                    # before the exception was raised (e.g. AssertionError on
                    # merge_commit or queue.put failure).  Use suppress so a
                    # cleanup failure never masks the original exception or
                    # prevents Future resolution.
                    if (
                        merge_result_local is not None
                        and merge_result_local.merge_worktree
                    ):
                        logger.debug(
                            f'Task {req.task_id}: cleaning up merge worktree after post-merge error'
                        )
                        with contextlib.suppress(Exception):
                            await self._git_ops.cleanup_merge_worktree(
                                merge_result_local.merge_worktree
                            )
                    merge_result_local = None
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
                    item = await self._remerge(req, item.started_monotonic)

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
                if item.merge_wt is not None:
                    with contextlib.suppress(BaseException):
                        await self._git_ops.cleanup_merge_worktree(item.merge_wt)
                if not req.result.done():
                    req.result.set_result(MergeOutcome(
                        'blocked', reason=f'Verifier error: {exc}',
                    ))
                n_failed = True
            except BaseException:
                # CancelledError or other fatal — resolve the in-flight Future
                # and clean up the merge worktree so callers don't hang forever.
                if item.merge_wt is not None:
                    with contextlib.suppress(BaseException):
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

    async def _remerge(self, req: MergeRequest, started_monotonic: float | None) -> SpeculativeItem:
        """Re-merge a request against actual main after speculation invalidation."""
        actual_main = await self._git_ops.get_main_sha()
        merge_result = await self._git_ops.merge_to_main(
            req.worktree, req.branch, base_sha=None,
        )
        if merge_result.conflicts:
            _emit_merge_attempt(self._event_store, req.task_id, 'conflict', duration_ms=_elapsed_ms(started_monotonic))
            if merge_result.merge_worktree:
                await self._git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
            return SpeculativeItem(
                request=req, merge_result=None, merge_wt=None,
                base_sha=actual_main, speculative=False, skip_verify=False,
                immediate_outcome=MergeOutcome(
                    'conflict', conflict_details=merge_result.details,
                ),
                started_monotonic=started_monotonic,
            )
        if not merge_result.success:
            if merge_result.merge_worktree:
                await self._git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
            return SpeculativeItem(
                request=req, merge_result=None, merge_wt=None,
                base_sha=actual_main, speculative=False, skip_verify=False,
                immediate_outcome=MergeOutcome('blocked', reason=merge_result.details),
                started_monotonic=started_monotonic,
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
            started_monotonic=started_monotonic,
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
            logger.info(
                f'Task {req.task_id}: verify start (merge={merge_commit[:8]}, '
                f'worktree={merge_wt.name})'
            )
            try:
                # max_retries=0: a hung post-merge verify is almost always a
                # deterministic failure (e.g. deadlocked test); retries just
                # multiply queue-wide stall.
                # is_merge_verify=True: merge worktrees are freshly created
                # per merge (no `.task/` dir and no warm cargo cache), so
                # they need the cold timeout despite `_is_verify_cold`'s
                # filesystem heuristic classifying them as warm.
                verify = await run_scoped_verification(
                    merge_wt, req.config, req.module_configs,
                    task_files=req.task_files,
                    max_retries=0,
                    is_merge_verify=True,
                )
            except Exception as exc:
                logger.info(
                    f'Task {req.task_id}: verify end '
                    f'(merge={merge_commit[:8]}, error)'
                )
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                if not req.result.done():
                    req.result.set_result(MergeOutcome(
                        'blocked', reason=f'Verification error: {exc}',
                    ))
                return False

            logger.info(
                f'Task {req.task_id}: verify end (merge={merge_commit[:8]}, '
                f'passed={verify.passed})'
            )
            if not verify.passed:
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                # Loop-breaker bookkeeping: bump only when the failure was a
                # pure timeout.  Real test/lint/type failures already bubble
                # up to the steward via the ``blocked`` outcome and do not
                # drive the verify-timeout / re-queue oscillation this
                # counter is designed to detect.
                if verify.timed_out:
                    new_count = self._post_merge_verify_timeouts.get(req.task_id, 0) + 1
                    self._post_merge_verify_timeouts[req.task_id] = new_count
                    if new_count >= self.MAX_POST_MERGE_VERIFY_TIMEOUTS:
                        logger.warning(
                            'Task %s: post-merge verify timed out %d times in '
                            'a row — next submission will be abandoned',
                            req.task_id, new_count,
                        )
                if not req.result.done():
                    detail = verify.failure_report()
                    reason = f'Post-merge verification failed: {verify.summary}'
                    if detail:
                        reason = f'{reason}\n\n{detail}'
                    req.result.set_result(MergeOutcome(
                        'blocked', reason=reason,
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
                # Loop-breaker counter reset on success — see MergeWorker.
                self._post_merge_verify_timeouts.pop(req.task_id, None)
                logger.info(f'Task {req.task_id}: merged to main successfully')
                _emit_merge_attempt(self._event_store, req.task_id, 'done', duration_ms=_elapsed_ms(item.started_monotonic))
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                push_status = await self._git_ops.push_main()
                if not req.result.done():
                    req.result.set_result(MergeOutcome('done', merge_sha=merge_commit, push_status=push_status))
                return True

            if result in ('wip_overlap', 'pop_conflict'):
                # Halt the queue globally — no more merges until resolved
                self.halt_for_wip(f'advance_main: {result}')
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                if result == 'pop_conflict':
                    # Main was advanced — push origin even though stash pop failed.
                    push_status = await self._git_ops.push_main()
                    recovery = getattr(self._git_ops, '_last_recovery_branch', None)
                    if not req.result.done():
                        req.result.set_result(MergeOutcome(
                            'done_wip_recovery',
                            reason=f'Merge advanced but stash pop conflicted. Recovery branch: {recovery}',
                            recovery_branch=recovery,
                            push_status=push_status,
                        ))
                else:
                    overlap = getattr(self._git_ops, '_last_overlap_files', None)
                    if not req.result.done():
                        req.result.set_result(MergeOutcome(
                            'wip_halted',
                            reason=f'WIP overlaps merge diff: {", ".join(overlap or [])}',
                            overlap_files=overlap,
                        ))
                return False

            if result == 'unmerged_state':
                # Pre-existing UU markers — halt queue, human escalation.
                self.halt_for_wip(
                    'advance_main: unmerged_state — project_root has unresolved '
                    'merge conflicts. Manual investigation required before any retry.'
                )
                self._cas_retries.pop(req.task_id, None)
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                if not req.result.done():
                    req.result.set_result(MergeOutcome(
                        'unmerged_state',
                        reason=(
                            f'advance_main returned unmerged_state: project_root has '
                            f'unresolved (UU/AA/DD) merge conflicts — halting queue; '
                            f'manual investigation required before any retry. '
                            f'(task {req.task_id})'
                        ),
                    ))
                return False

            if result == 'pop_conflict_no_advance':
                # Stash pop conflicted during CAS-failure recovery — merge did NOT land.
                self.halt_for_wip('advance_main: pop_conflict_no_advance')
                recovery = getattr(self._git_ops, '_last_recovery_branch', None)
                self._cas_retries.pop(req.task_id, None)
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                if not req.result.done():
                    req.result.set_result(MergeOutcome(
                        'wip_recovery_no_advance',
                        reason=(
                            f'Merge did not advance AND WIP stash pop conflicted. '
                            f'Recovery branch: {recovery}. '
                            f'Manual intervention required — do not retry automatically. '
                            f'(task {req.task_id})'
                        ),
                        recovery_branch=recovery,
                    ))
                return False

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
                _emit_merge_attempt(self._event_store, req.task_id, 'cas_exhausted', attempt=total, duration_ms=_elapsed_ms(item.started_monotonic))
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
                started_monotonic=item.started_monotonic,
            )
            logger.info(
                f'Task {req.task_id}: CAS failed (attempt {total}/'
                f'{self.MAX_CAS_RETRIES}), retrying'
            )
            _emit_merge_attempt(self._event_store, req.task_id, 'cas_retry', attempt=total, duration_ms=_elapsed_ms(item.started_monotonic))
