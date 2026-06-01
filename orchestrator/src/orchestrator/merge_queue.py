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
import dataclasses
import logging
import posixpath
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, MergeResult, WorktreeMissing, _run
from orchestrator.verify import VerifyResult, run_scoped_verification, run_verification

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig

logger = logging.getLogger(__name__)


@dataclass
class DropGuardResult:
    """Structured return value from :func:`_check_plan_targets_in_tree`.

    Attributes:
        dropped: Paths present on task HEAD but absent from the merge
            commit — i.e. files the merger discarded.  Empty list means
            the merge preserved everything the task branch produced.
    """

    dropped: list[str] = field(default_factory=list)


@dataclass
class PlanFilesTouchedResult:
    """Structured return value from :func:`_check_plan_files_touched_in_branch`.

    Attributes:
        not_touched: Plan-file entries that the branch's history did NOT
            touch.  Non-empty means the architect declared work that the
            branch never actually delivered.  Empty list means every plan
            entry is covered by some commit on the branch.
    """

    not_touched: list[str] = field(default_factory=list)


DROPPED_PLAN_TARGETS_REASON_PREFIX = 'Merge commit is missing plan target files'
"""Prefix of the ``MergeOutcome.reason`` string emitted when the drop-guard
detects work on the task tip that the merge commit dropped.  Workflow-side
short-circuits use this prefix to route the outcome straight to L1 without
invoking the steward (the gate fires only on real merger drops post-rewrite,
which is the human-judgement case the gate was built for)."""


PLAN_FILES_NOT_TOUCHED_REASON_PREFIX = 'Plan files not touched by branch'
"""Prefix of the ``MergeOutcome.reason`` string emitted by the pre-merge
Decision-1 check.  When the architect declared specific plan files but
the branch's history (``base..HEAD``) doesn't touch them, the implementation
hasn't actually delivered against the plan — short-circuit straight to L1
without involving the steward (mutating plan.json to silence the gate
would defeat its purpose)."""


POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX = 'Post-merge content equivalence failed'
"""Prefix of the ``MergeOutcome.reason`` string emitted by the post-merge
Decision-2 check.  After ``advance_main`` succeeds, we verify that
``branch_HEAD`` and the advanced main SHA have the same tree (modulo
``.task/``).  Any divergence indicates conflict resolution dropped or
rewrote work and needs human judgement, not a steward retry."""


TRANSIENT_INFRA_REASON_PREFIX = 'Transient infrastructure failure (disk pressure)'
"""Prefix of the ``MergeOutcome.reason`` emitted when a post-merge verify
fails with a no-space-left/ENOSPC signature that PERSISTS after one
prune-and-retry.  Disk pressure is frequently a self-healing host condition
(a concurrent build finishes, a worktree is reaped), so the merge worker
prunes stale ``_merge-*`` worktrees and retries the verify once before
surfacing this.  The workflow routes this prefix to
``_mark_blocked(escalate_to_human=True, category='infra_issue')`` — the
infra_issue category plus the durable ref lets the escalation-watcher
auto-resolve if the disk has recovered by read-time."""


TRAIN_INCOMPLETE_REASON_PREFIX = 'Train merge rejected: not all members are merge-deferred'
"""Prefix of the ``MergeOutcome.reason`` string emitted when a
``GroupMergeRequest`` is dispatched but one or more member tasks are not
in the ``merge-deferred`` status.  Workflow-side classifiers pattern-match
this prefix to distinguish a pre-condition failure (no git work done, safe
to retry after the offending member catches up) from a post-merge block."""

TRAIN_REBASE_CONFLICT_REASON_PREFIX = 'Train merge rejected: tip branch rebase conflict'
"""Prefix of the ``MergeOutcome.reason`` string emitted when the tip-branch
rebase-onto-main step of a ``GroupMergeRequest`` fails with conflicts.
``rebase_onto_main`` already ran ``git rebase --abort``, so the worktree is
left clean.  The downstream classifier surfaces this to the enqueuer so the
tip task can be re-rebased in its own worktree before the train is
re-submitted."""

TRAIN_PARTIAL_FLIP_REASON_PREFIX = 'Train partially flipped'
"""Prefix of the ``MergeOutcome.reason`` string emitted when a train lands
(main advances successfully) but one or more ``mark_member_done`` callbacks
raise after the advance.  The outcome status STAYS ``'done'`` — the git state
is correct — but this prefix signals downstream reconciliation / the steward
that some member tasks still need their status flipped.  Downstream classifiers
pattern-match this prefix to distinguish a fully-clean landing
(``reason=None``) from a partial-flip that requires manual / automated
cleanup."""


POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX = 'Post-merge unscoped type-check failed'
"""Prefix of the ``MergeOutcome.reason`` string emitted when the post-merge
unscoped pyright check detects a cross-PR union break.  After
``advance_main`` succeeds, a full package-wide type-check is run against the
advanced main SHA for each subproject that declares a ``type_check_command``
in its ``ModuleConfig``.  Unlike per-PR scoped verify, this catches the case
where PR A widens a Protocol and PR B (verified against pre-A main) adds a
conformer satisfying the OLD Protocol — after both land the union has a
conformer missing the new method; each PR verified clean, but only the
post-merge whole-package check sees it.

This is a SIGNAL, not a gate: the merge has already landed (``update-ref``
ran), so we do NOT revert.  We skip ``push_main`` (same as the equivalence
check), emit an L1 blocked outcome, and route to human / auto-watcher for a
fix-forward task.  Consistent with ``POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX``:
a landed merge must never be blocked by a flaky hang or infra error, so the
check fails open on timeouts and worktree-create exceptions."""

_HALT_ADVANCE_RESULTS: tuple[str, ...] = (
    'wip_overlap', 'pop_conflict', 'unmerged_state', 'pop_conflict_no_advance',
)
"""``advance_main`` result codes that can trigger a WIP halt.

Shared between :class:`MergeWorker` and :class:`SpeculativeMergeWorker` to
avoid silent divergence: if the set of halt-triggering results ever changes,
updating this single constant propagates to both workers automatically."""


@dataclass
class PostMergePyrightResult:
    """Structured return value from :func:`_check_post_merge_pyright`.

    Attributes:
        failing_subprojects: Prefixes of subprojects whose unscoped
            type-check command exited non-zero (genuine type failure, not
            a timeout or infra error).  Empty list means clean.
        detail: Bounded human-readable detail from the first failing
            subproject's output, for inclusion in the escalation reason.
    """

    failing_subprojects: list[str] = field(default_factory=list)
    detail: str = ''

    @property
    def broken(self) -> bool:
        """True when at least one subproject's type-check genuinely failed."""
        return bool(self.failing_subprojects)


_ENOSPC_MARKERS = ('no space left on device', 'os error 28', 'enospc')
"""Substrings (matched case-insensitively) that mark a disk-full failure.
There is no structured ENOSPC field on ``VerifyResult`` — the signal only
appears in captured tool output — so detection is a string match."""


def _verify_hit_enospc(verify: VerifyResult) -> bool:
    """True when a failed ``VerifyResult`` bears a disk-full signature.

    Scans the human-readable failure report and each captured output stream
    (test/lint/type) for any ``_ENOSPC_MARKERS`` entry.
    """
    haystack = ' '.join(
        s for s in (
            verify.failure_report(),
            verify.test_output,
            verify.lint_output,
            verify.type_output,
        ) if isinstance(s, str)
    ).lower()
    return any(marker in haystack for marker in _ENOSPC_MARKERS)


_SPECULATION_RACE_MARKER = 'not something we can merge'
"""LOAD-BEARING exact substring match on git porcelain output.

Do NOT paraphrase this string. Git emits it verbatim when a ref cannot be
resolved to a mergeable object (e.g. the branch was force-pushed away between
the speculative merge build and the re-merge attempt). The exact phrase is what
distinguishes a speculation-race failure from other git non-conflict errors.
"""


def _is_speculation_race(stderr: str) -> bool:
    """True when *stderr* from a failed merge contains the speculation-race signature.

    Uses a load-bearing exact substring match against ``_SPECULATION_RACE_MARKER``
    (the git porcelain phrase ``not something we can merge``).  Mirrors the
    ``_verify_hit_enospc`` / ``_ENOSPC_MARKERS`` pattern.
    """
    return _SPECULATION_RACE_MARKER in stderr


async def _ensure_verify_disk_space(
    git_ops: GitOps,
    merge_wt: Path,
    min_free_bytes: int,
    task_id: str,
) -> str | None:
    """Pre-verify disk guard.  Returns a blocked-reason string (prefixed with
    ``TRANSIENT_INFRA_REASON_PREFIX``) when free space on *merge_wt*'s volume
    is still below *min_free_bytes* after pruning stale ``_merge-*`` worktrees
    — the caller should clean up the merge worktree and short-circuit to the
    L1 infra_issue path WITHOUT running a doomed build.  Returns ``None`` to
    proceed with verify (either there was already enough space, or pruning
    freed enough).

    Fails open: any ``shutil.disk_usage`` ``OSError`` returns ``None``.  A
    merge must never be blocked because we couldn't stat the volume — the
    post-verify ENOSPC retry/escalate path is the backstop for builds that
    pass this guard (≥ threshold free) but still exhaust the disk mid-run.

    Scope: post-merge verify only.  The non-merge task-verify path
    (``run_verification`` during implement→verify) still classifies a
    disk-full failure by return code and can route to the steward; guarding
    that path is deliberately out of scope here.
    """
    gib = 1024**3
    try:
        free = shutil.disk_usage(merge_wt).free
    except OSError as exc:
        logger.warning(
            'Task %s: pre-verify disk guard: disk_usage(%s) failed (%s); '
            'failing open (proceeding with verify)',
            task_id, merge_wt, exc,
        )
        return None
    if free >= min_free_bytes:
        return None
    logger.warning(
        'Task %s: pre-verify disk guard: %.2f GiB free < %.2f GiB threshold; '
        'pruning stale _merge-* worktrees before verify',
        task_id, free / gib, min_free_bytes / gib,
    )
    pruned = await git_ops.prune_stale_merge_worktrees(keep=merge_wt)
    try:
        free = shutil.disk_usage(merge_wt).free
    except OSError as exc:
        logger.warning(
            'Task %s: pre-verify disk guard: post-prune disk_usage(%s) failed '
            '(%s); failing open (proceeding with verify)',
            task_id, merge_wt, exc,
        )
        return None
    if free >= min_free_bytes:
        logger.info(
            'Task %s: pre-verify disk guard: pruning %d stale merge '
            'worktree(s) freed enough space (%.2f GiB free); proceeding',
            task_id, len(pruned), free / gib,
        )
        return None
    return (
        f'{TRANSIENT_INFRA_REASON_PREFIX}: pre-verify disk guard found only '
        f'{free / gib:.2f} GiB free (threshold {min_free_bytes / gib:.2f} GiB) '
        f'after pruning {len(pruned)} stale merge worktree(s); skipping '
        f'post-merge verify to avoid a doomed build under disk pressure.'
    )


async def _run_post_merge_verify(
    git_ops: GitOps,
    req: MergeRequest,
    merge_wt: Path,
    *,
    timeouts: dict[str, int],
    enospc_retries: dict[str, int],
    max_timeouts: int,
    max_enospc: int,
) -> MergeOutcome | None:
    """Run post-merge verification for a single task.

    Shared by :class:`MergeWorker` and :class:`SpeculativeMergeWorker`.

    Returns ``None`` when verification passes; returns a ``MergeOutcome``
    (and cleans up *merge_wt*) when it fails via a controlled path (disk
    guard, verify-not-passed).  Does **not** contain a ``try/except`` — any
    exception from ``run_scoped_verification`` propagates to the caller.
    ``MergeWorker`` calls this bare (exceptions reach ``_process``);
    ``SpeculativeMergeWorker`` wraps the call in its existing ``try/except``
    that maps a raised verify to a ``'Verification error: ...'`` outcome.
    """
    # Pre-verify disk guard: if free space is low, prune stale merge
    # worktrees; if still low, skip the build and escalate as transient
    # infra rather than entering a doomed multi-minute ENOSPC build.
    disk_reason = await _ensure_verify_disk_space(
        git_ops, merge_wt,
        req.config.merge_verify_min_free_disk_bytes, req.task_id,
    )
    if disk_reason is not None:
        await git_ops.cleanup_merge_worktree(merge_wt)
        return MergeOutcome('blocked', reason=disk_reason, verify_skipped=True)
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
        force_workspace=req.config.merge_verify_workspace,
        role='merge',
    )
    # Transient-infra (disk pressure) retry: an ENOSPC failure is
    # often a self-healing host condition.  Prune stale _merge-*
    # worktrees (never task worktrees) and retry the verify once in
    # the same merge_wt before escalating.
    if not verify.passed and _verify_hit_enospc(verify):
        prior_enospc = enospc_retries.get(req.task_id, 0)
        if prior_enospc < max_enospc:
            enospc_retries[req.task_id] = prior_enospc + 1
            pruned = await git_ops.prune_stale_merge_worktrees(keep=merge_wt)
            logger.warning(
                'Task %s: post-merge verify hit ENOSPC; pruned %d '
                'stale merge worktree(s), retrying verify once',
                req.task_id, len(pruned),
            )
            verify = await run_scoped_verification(
                merge_wt, req.config, req.module_configs,
                task_files=req.task_files,
                max_retries=0,
                is_merge_verify=True,
                force_workspace=req.config.merge_verify_workspace,
                role='merge',
            )
    if not verify.passed:
        await git_ops.cleanup_merge_worktree(merge_wt)
        # Persistent ENOSPC after the prune-and-retry → transient infra.
        if _verify_hit_enospc(verify):
            detail = verify.failure_report()
            reason = (
                f'{TRANSIENT_INFRA_REASON_PREFIX}: post-merge verify '
                f'still reports no space left on device after pruning '
                f'stale merge worktrees and retrying. {verify.summary}'
            )
            if detail:
                reason = f'{reason}\n\n{detail}'
            return MergeOutcome('blocked', reason=reason)
        detail = verify.failure_report()
        reason = f'Post-merge verification failed: {verify.summary}'
        if detail:
            reason = f'{reason}\n\n{detail}'
        # Loop-breaker bookkeeping: bump only when the failure was a
        # pure timeout.  Real test/lint/type failures already bubble
        # up to the steward and don't drive the re-queue oscillation
        # the loop-breaker is designed to catch.
        if verify.timed_out:
            new_count = timeouts.get(req.task_id, 0) + 1
            timeouts[req.task_id] = new_count
            if new_count >= max_timeouts:
                logger.warning(
                    'Task %s: post-merge verify timed out %d times in a '
                    'row — next submission will be abandoned',
                    req.task_id, new_count,
                )
        return MergeOutcome('blocked', reason=reason)
    return None


async def _finalize_advanced_merge(
    git_ops: GitOps,
    req: MergeRequest,
    event_store: EventStore | None,
    *,
    merge_commit_fallback: str,
    base_sha: str,
    started_monotonic: float | None,
    cas_retries: dict[str, int],
    timeouts: dict[str, int],
    enospc_retries: dict[str, int],
    log_label: str = '',
    train_id: str | None = None,
    member_task_ids: list[str] | None = None,
) -> MergeOutcome:
    """Post-advance success block shared by MergeWorker and SpeculativeMergeWorker.

    Pops per-task counters, resolves the advanced SHA, runs the
    equivalence and pyright gates, and returns a :class:`MergeOutcome`.
    Does **not** touch *merge_wt* — callers must clean it up before
    calling this function (MergeWorker already does so right after
    ``advance_main``; SpeculativeMergeWorker moves its pre-pyright
    cleanup to before this call).

    *log_label* is interpolated into warning log messages so each
    worker can retain its current log prefix
    (``''`` for MergeWorker, ``' (speculative)'`` for
    SpeculativeMergeWorker).

    *train_id* and *member_task_ids* are optional train-correlation tags
    forwarded to every ``_emit_merge_attempt`` call inside this function.
    MergeWorker and SpeculativeMergeWorker pass ``None`` (no behavior
    change); ``_do_train_merge`` passes the request's train metadata so
    ``merge_attempt`` events stay tagged for downstream reconciliation.
    """
    cas_retries.pop(req.task_id, None)
    timeouts.pop(req.task_id, None)
    enospc_retries.pop(req.task_id, None)
    # Use the post-rebase SHA actually placed on main (advance_main
    # rebases on CAS retry; merge_commit_fallback is the stale
    # pre-rebase SHA and would fail done_provenance ancestor check).
    advanced_sha = getattr(git_ops, '_last_advanced_sha', None) or merge_commit_fallback

    # Decision-2 post-merge content-equivalence check.
    equiv_failed = await _check_post_merge_equivalence(
        req.worktree, advanced_sha, git_ops, base_sha,
        task_id=req.task_id,
    )
    if equiv_failed:
        logger.warning(
            'Task %s%s: post-merge equivalence failed — '
            'branch HEAD and advanced main %s diverge in: %r',
            req.task_id, log_label, advanced_sha[:12], equiv_failed,
        )
        _emit_merge_attempt(
            event_store, req.task_id,
            'post_merge_equivalence_failed',
            duration_ms=_elapsed_ms(started_monotonic),
            train_id=train_id,
            member_task_ids=member_task_ids,
        )
        return MergeOutcome(
            'blocked',
            reason=(
                f'{POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX}: '
                f'branch and main diverge in '
                f'{", ".join(equiv_failed)}. '
                f'Conflict resolution likely dropped or rewrote '
                f'work; review {advanced_sha[:12]} against the '
                f'task branch tip.'
            ),
        )

    # Decision-3 post-merge unscoped type-check.
    pyright_result = await _check_post_merge_pyright(
        advanced_sha, git_ops, req.config, req.module_configs,
        task_id=req.task_id,
    )
    if pyright_result.broken:
        logger.warning(
            'Task %s%s: post-merge unscoped type-check failed for %s on %s',
            req.task_id, log_label,
            ', '.join(pyright_result.failing_subprojects),
            advanced_sha[:12],
        )
        _emit_merge_attempt(
            event_store, req.task_id,
            'post_merge_pyright_broken',
            duration_ms=_elapsed_ms(started_monotonic),
            train_id=train_id,
            member_task_ids=member_task_ids,
        )
        return MergeOutcome(
            'blocked',
            reason=(
                f'{POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX}: '
                f'post-merge unscoped type-check failed for '
                f'{", ".join(pyright_result.failing_subprojects)} '
                f'on {advanced_sha[:12]}. {pyright_result.detail}'
            ),
        )

    logger.info(f'Task {req.task_id}: merged to main successfully')
    _emit_merge_attempt(
        event_store, req.task_id, 'done',
        duration_ms=_elapsed_ms(started_monotonic),
        train_id=train_id,
        member_task_ids=member_task_ids,
    )
    push_status = await git_ops.push_main()
    return MergeOutcome('done', merge_sha=advanced_sha, push_status=push_status)


async def _map_advance_failure(
    git_ops: GitOps,
    result: str,
    *,
    task_id: str,
    merge_commit_fallback: str,
    halt: Callable[[str], None],
    cas_retries: dict[str, int],
) -> MergeOutcome:
    """Advance-failure result → MergeOutcome mapping shared by both workers.

    Handles ``wip_overlap``, ``pop_conflict``, ``unmerged_state``,
    ``pop_conflict_no_advance``, ``not_descendant``, ``contaminated``,
    and ``stash_failed``.  Does **not** handle ``cas_failed``
    (per-worker retry orchestration is a preserved difference) or the
    request-abandoned early-out (kept per-worker because the two workers
    differ on what to clean up).  Does **not** touch *merge_wt* — callers
    must clean it up separately.

    *halt* is the worker's ``halt_for_wip`` callable; *cas_retries* is
    mutated for the terminal result codes (popped) but NOT for
    ``wip_overlap`` (which is a recoverable halt, not a terminal outcome).
    """
    if result in ('wip_overlap', 'pop_conflict'):
        halt(f'advance_main: {result}')
        if result == 'pop_conflict':
            # Main was advanced — push origin even though stash pop failed.
            push_status = await git_ops.push_main()
            recovery = getattr(git_ops, '_last_recovery_branch', None)
            # Main IS on the post-rebase SHA — propagate it so workflow's
            # _handle_wip_recovery → set_task_status('done') has valid
            # done_provenance (otherwise the call hits "kind required").
            advanced_sha = (
                getattr(git_ops, '_last_advanced_sha', None) or merge_commit_fallback
            )
            return MergeOutcome(
                'done_wip_recovery',
                reason=f'Merge advanced but stash pop conflicted. Recovery branch: {recovery}',
                recovery_branch=recovery,
                push_status=push_status,
                merge_sha=advanced_sha,
            )
        else:
            overlap = getattr(git_ops, '_last_overlap_files', None)
            return MergeOutcome(
                'wip_halted',
                reason=f'WIP overlaps merge diff: {", ".join(overlap or [])}',
                overlap_files=overlap,
            )

    if result == 'unmerged_state':
        # Permanent block — pre-existing UU markers in project_root.
        # Halt the queue and route to human escalation (not steward).
        halt(
            'advance_main: unmerged_state — project_root has unresolved merge '
            'conflicts. Manual investigation required before any retry.'
        )
        cas_retries.pop(task_id, None)
        return MergeOutcome(
            'unmerged_state',
            reason=(
                f'advance_main returned unmerged_state: project_root has '
                f'unresolved (UU/AA/DD) merge conflicts — halting queue; '
                f'manual investigation required before any retry. '
                f'(task {task_id})'
            ),
        )

    if result == 'pop_conflict_no_advance':
        # Stash pop conflicted during CAS-failure recovery — merge did NOT land.
        # Halt queue and return distinct outcome for human-level escalation.
        halt('advance_main: pop_conflict_no_advance')
        recovery = getattr(git_ops, '_last_recovery_branch', None)
        cas_retries.pop(task_id, None)
        return MergeOutcome(
            'wip_recovery_no_advance',
            reason=(
                f'Merge did not advance AND WIP stash pop conflicted. '
                f'Recovery branch: {recovery}. '
                f'Manual intervention required — do not retry automatically. '
                f'(task {task_id})'
            ),
            recovery_branch=recovery,
        )

    # not_descendant / contaminated / stash_failed — permanent failure
    cas_retries.pop(task_id, None)
    return MergeOutcome(
        'blocked',
        reason=f'advance_main failed ({result}) for task {task_id}',
    )


async def _check_plan_targets_in_tree(
    merge_commit_sha: str,
    task_worktree: Path,
    git_ops: GitOps,
    main_sha: str,
    *,
    task_id: str | None = None,
) -> DropGuardResult:
    """Return a :class:`DropGuardResult` listing files dropped by the merger.

    Compares ``task_HEAD`` (the source worktree's HEAD) to the merge commit
    directly.  A "drop" means the file is on the task tip but absent from
    the merge commit — i.e. conflict resolution discarded work the branch
    actually produced.  Plan-vs-tip mismatches (gitignored files listed in
    ``plan['files']``, prereq-deleted files, amend-deleted files) are out
    of scope for this gate; catching those belongs to verify/review.

    The raw ``task_HEAD``-minus-``merge_commit`` diff over-flags: a clean
    merge legitimately drops a path that a *sibling* moved or deleted on
    main, even though this branch carried the old copy and never touched
    it.  To subtract main-side change, we intersect the drop set with the
    files the branch itself ADDED or MODIFIED since the shared merge-base
    (``merge-base(task_HEAD, main_sha)``).  ``main_sha`` is the pre-merge
    main tip the merge was computed against (actual or speculative), not
    the post-merge advanced SHA — using it keeps the subtraction robust to
    ``advance_main``'s CAS-retry rebase.  ``--no-renames`` is deliberate:
    a sibling rename appears as a delete of the old path on main, which is
    absent from the branch's add/modify set and therefore dropped here.

    Fail-open on rc != 0: post-merge verify is the next safety net, and
    flagging a phantom drop on a transient git error is worse than missing
    a real one.  Loud-log so regressions surface in ops.
    """
    rc, head_out, head_err = await _run(
        ['git', 'rev-parse', 'HEAD'], cwd=task_worktree,
    )
    if rc != 0:
        logger.warning(
            'drop-guard: git rev-parse HEAD failed in %s (rc=%d, stderr=%s); '
            'failing open. task_id=%s merge_commit_sha=%s',
            task_worktree, rc, head_err.strip(),
            task_id or '<unknown>', merge_commit_sha,
        )
        return DropGuardResult()
    task_head = head_out.strip()

    # Shared baseline: what the branch and main diverged from.  Subtracting
    # main-side change below is anchored here.
    rc, base_out, base_err = await _run(
        ['git', 'merge-base', task_head, main_sha],
        cwd=git_ops.project_root,
    )
    if rc != 0:
        logger.warning(
            'drop-guard: merge-base %s %s failed (rc=%d, stderr=%s); '
            'failing open. task_id=%s',
            task_head, main_sha, rc, base_err.strip(),
            task_id or '<unknown>',
        )
        return DropGuardResult()
    base = base_out.strip()

    # Files the branch itself ADDED or MODIFIED since the merge-base.  A
    # legitimately-dropped path the branch never touched (e.g. sibling-moved
    # on main) is absent here, so the intersection below excludes it.
    rc, changed_out, changed_err = await _run(
        [
            'git', 'diff', '--name-only', '--no-renames',
            '--diff-filter=AM', base, task_head,
        ],
        cwd=git_ops.project_root,
    )
    if rc != 0:
        logger.warning(
            'drop-guard: branch-changed diff %s..%s failed (rc=%d, stderr=%s); '
            'failing open. task_id=%s',
            base, task_head, rc, changed_err.strip(),
            task_id or '<unknown>',
        )
        return DropGuardResult()
    branch_changed = {ln.strip() for ln in changed_out.splitlines() if ln.strip()}

    rc, out, err = await _run(
        [
            'git', 'diff', '--name-only', '--no-renames',
            '--diff-filter=D', task_head, merge_commit_sha,
        ],
        cwd=git_ops.project_root,
    )
    if rc != 0:
        logger.warning(
            'drop-guard: git diff %s..%s failed (rc=%d, stderr=%s); '
            'failing open. task_id=%s',
            task_head, merge_commit_sha, rc, err.strip(),
            task_id or '<unknown>',
        )
        return DropGuardResult()

    dropped_in_merge = [ln.strip() for ln in out.splitlines() if ln.strip()]
    # Subtract main-side change: only a path the branch actually produced
    # AND the merge discarded is a real drop.  Preserve merge-diff order.
    real_drops = [p for p in dropped_in_merge if p in branch_changed]
    if real_drops:
        logger.warning(
            'drop-guard: dropped_plan_targets '
            'task_id=%s merge_commit_sha=%s task_head=%s dropped=%r',
            task_id or '<unknown>', merge_commit_sha, task_head, real_drops,
        )
    return DropGuardResult(dropped=real_drops)


def _normalize_plan_path(entry: str) -> str:
    """Return a git-canonical form of a declared plan path for comparison.

    Strips leading ``./`` and collapses redundant separators via
    ``posixpath.normpath`` (always POSIX-style, regardless of host OS, so
    the canonical form always matches git's forward-slash output).  Guards
    the degenerate case where normpath maps ``'.'`` / ``'./'`` to ``'.'``
    (the repo root), which would spuriously prefix-match every touched path —
    in that case the original string is returned so the entry falls through
    and is correctly flagged.

    Examples::

        './.jcodemunch.jsonc' → '.jcodemunch.jsonc'
        './src/a.py'          → 'src/a.py'
        './src/pkg'           → 'src/pkg'
        './src/pkg/'          → 'src/pkg'   (trailing slash collapsed)
        'src//pkg'            → 'src/pkg'   (redundant separator collapsed)
        'src/b.py'            → 'src/b.py'  (unchanged)
    """
    norm = posixpath.normpath(entry)
    # normpath maps '' / '.' / './' → '.', which denotes the repo root.
    # Empty entries are already filtered by the caller; guard '.' so it
    # can't spuriously match the repo root prefix.
    return entry if norm == '.' else norm


async def _check_plan_files_touched_in_branch(
    plan_files: list[str],
    base_sha: str,
    branch_head: str,
    git_ops: GitOps,
    *,
    task_id: str | None = None,
) -> PlanFilesTouchedResult:
    """Pre-merge Decision-1 check: every plan file must be touched on the branch.

    For each entry in ``plan_files``, classify as touched if either:
        (a) the entry appears verbatim in
            ``git log --name-only base..branch_head`` (file path), OR
        (b) the entry resolves to a directory in the branch tree (via
            ``git ls-tree``) and at least one touched file path has it as
            a path-prefix (directory entries are valid plan targets when
            an agent stages multiple files inside).

    Empty ``plan_files`` returns no entries — vacuously satisfied.

    Fail-open on git error (matches :func:`_check_plan_targets_in_tree`):
    return an empty ``PlanFilesTouchedResult`` so a transient diff error
    doesn't block a real merge.  Loud-log so regressions surface in ops.
    """
    if not plan_files:
        return PlanFilesTouchedResult()

    touched = await git_ops.get_files_touched_in_branch(base_sha, branch_head)
    touched_set = set(touched)

    not_touched: list[str] = []
    for entry in plan_files:
        if not entry:
            continue
        # Normalize the declared path to git-canonical form (strip leading ./)
        # before comparison.  The touched_set is already canonical git output.
        # Keep the original `entry` for diagnostics so the escalation message
        # reflects exactly what the architect wrote in plan.json.
        norm = _normalize_plan_path(entry)
        if norm in touched_set:
            continue

        # Directory match: ask the branch tree what kind of object the
        # entry names.  ``git ls-tree`` prints "<mode> tree <sha>\t<path>"
        # for directories and "<mode> blob <sha>\t<path>" for files.
        rc, ls_out, ls_err = await _run(
            ['git', 'ls-tree', branch_head, '--', norm],
            cwd=git_ops.project_root,
        )
        if rc == 0 and ls_out.strip() and ' tree ' in ls_out:
            # Directory: prefix-match against the touched set.
            prefix = norm.rstrip('/') + '/'
            if any(t.startswith(prefix) for t in touched_set):
                continue

        not_touched.append(entry)

    if not_touched:
        logger.warning(
            'plan-files-touched: not_touched task_id=%s '
            'base=%s head=%s entries=%r',
            task_id or '<unknown>', base_sha, branch_head, not_touched,
        )
    return PlanFilesTouchedResult(not_touched=not_touched)


async def _check_post_merge_equivalence(
    task_worktree: Path,
    advanced_sha: str,
    git_ops: GitOps,
    main_sha: str,
    *,
    task_id: str | None = None,
) -> list[str]:
    """Return branch-touched paths whose ``advanced_sha`` blob differs from ``branch_HEAD``.

    Decision-2 post-merge gate: every file the branch touched must appear
    in the advanced main commit with identical content.  Files the branch
    did NOT touch are excluded — main legitimately includes work from
    siblings or earlier merges that the branch never saw.

    Scope (compare set): the merge-base of branch and ``main_sha`` (the
    pre-merge main tip, NOT ``advanced_sha``) is the pre-branch baseline;
    ``git diff --name-only base..branch_head`` lists every path the branch
    produced.  We then subtract the paths main *also* changed since that
    baseline (``base..main_sha``): a clean 3-way merge legitimately combines
    the branch's and a sibling's edits to a shared path (e.g. ``Cargo.lock``),
    so merged main differs from the branch tip there without anything being
    dropped.  Anchoring the base on ``main_sha`` rather than ``advanced_sha``
    keeps the gate robust to ``advance_main``'s CAS-retry rebase.

    The surviving compare set is the branch's own work that main did not
    touch; we ask git whether any of those paths differ between
    ``branch_HEAD`` and ``advanced_sha`` — non-empty = the merge dropped or
    rewrote that work.

    Empty list = clean preservation (ff-merge, --no-ff with no conflicts,
    clean rebase).  Non-empty = caller treats as a hard failure.

    Fail-open on git error: returns an empty list and logs a WARNING.
    The call is a defense-in-depth check; a transient git error must
    not block a successful merge from being recorded.
    """
    rc, head_out, head_err = await _run(
        ['git', 'rev-parse', 'HEAD'], cwd=task_worktree,
    )
    if rc != 0:
        logger.warning(
            'post-merge-equiv: git rev-parse HEAD failed in %s '
            '(rc=%d, stderr=%s); failing open. task_id=%s advanced_sha=%s',
            task_worktree, rc, head_err.strip(),
            task_id or '<unknown>', advanced_sha,
        )
        return []
    branch_head = head_out.strip()

    # Determine the branch's touched set against the merge-base with the
    # PRE-merge main tip (main_sha).  Using main_sha rather than advanced_sha
    # lets us subtract main-side change below and stays rebase-robust.
    rc, mb_out, mb_err = await _run(
        ['git', 'merge-base', branch_head, main_sha],
        cwd=git_ops.project_root,
    )
    if rc != 0:
        logger.warning(
            'post-merge-equiv: merge-base failed for %s..%s '
            '(rc=%d, stderr=%s); failing open. task_id=%s',
            branch_head, main_sha, rc, mb_err.strip(),
            task_id or '<unknown>',
        )
        return []
    base_sha = mb_out.strip()

    rc, touched_out, touched_err = await _run(
        [
            'git', 'diff', '--name-only', '--no-renames',
            base_sha, branch_head, '--', ':!.task/',
        ],
        cwd=git_ops.project_root,
    )
    if rc != 0:
        logger.warning(
            'post-merge-equiv: branch-touched diff failed for %s..%s '
            '(rc=%d, stderr=%s); failing open. task_id=%s',
            base_sha, branch_head, rc, touched_err.strip(),
            task_id or '<unknown>',
        )
        return []
    branch_touched = [ln.strip() for ln in touched_out.splitlines() if ln.strip()]
    if not branch_touched:
        return []

    # Paths main independently changed since the shared baseline.  A clean
    # merge combining the branch's and a sibling's edits to such a path makes
    # merged main differ from the branch tip there with nothing dropped, so
    # subtract them from the compare set.
    #
    # Edge: when base == main_sha (speculative merge against a base that is
    # itself the pre-merge tip), this diff is empty and we degrade to strict
    # equivalence — the correct conservative fallback.  When a CAS re-merge
    # advanced main past main_sha, main_touched may be a subset of what main
    # really changed, making the gate slightly more conservative (a rare
    # re-introduced FP) but never masking a drop — the safe direction.
    rc, main_touched_out, main_touched_err = await _run(
        [
            'git', 'diff', '--name-only', '--no-renames',
            base_sha, main_sha, '--', ':!.task/',
        ],
        cwd=git_ops.project_root,
    )
    if rc != 0:
        logger.warning(
            'post-merge-equiv: main-touched diff failed for %s..%s '
            '(rc=%d, stderr=%s); failing open. task_id=%s',
            base_sha, main_sha, rc, main_touched_err.strip(),
            task_id or '<unknown>',
        )
        return []
    main_touched = {ln.strip() for ln in main_touched_out.splitlines() if ln.strip()}

    compare_set = [p for p in branch_touched if p not in main_touched]
    if not compare_set:
        # Empty pathspec on ``git diff -- `` means *all files*, not none, so
        # short-circuit rather than running an unscoped diff.
        return []

    # Compare branch_head vs advanced_sha restricted to the surviving paths.
    rc, out, err = await _run(
        [
            'git', 'diff', '--name-only', '--no-renames',
            branch_head, advanced_sha, '--', *compare_set,
        ],
        cwd=git_ops.project_root,
    )
    if rc != 0:
        logger.warning(
            'post-merge-equiv: scoped diff %s..%s failed (rc=%d, stderr=%s); '
            'failing open. task_id=%s',
            branch_head, advanced_sha, rc, err.strip(),
            task_id or '<unknown>',
        )
        return []

    return [ln.strip() for ln in out.splitlines() if ln.strip()]


# Sentinel returned by _rebase_delta_touched_overlap when a git error makes
# the overlap status unknowable.  Any non-empty list causes the caller to
# re-verify (fail-CLOSED policy).
_OVERLAP_GIT_ERROR_SENTINEL = ['<git-error: re-verify required>']


async def _rebase_delta_touched_overlap(
    task_worktree: Path,
    rebased_from: str,
    rebased_onto: str,
    git_ops: GitOps,
    *,
    task_id: str | None = None,
) -> list[str]:
    """Return the sorted intersection of branch-touched and intervening-delta files.

    Determines whether the post-rebase tree needs re-verification by checking
    whether the intervening main churn (``rebased_from`` → ``rebased_onto``)
    overlaps with the files the branch itself changed.

    Algorithm
    ---------
    1. Resolve ``branch_head`` from ``task_worktree`` HEAD.
    2. Compute ``base = merge-base(branch_head, rebased_from)`` — the common
       ancestor of the branch and the pre-churn main.
    3. ``branch_touched = diff(base..branch_head) --no-renames :!.task/``
    4. ``intervening   = diff(rebased_from..rebased_onto) --no-renames :!.task/``
    5. Return ``sorted(set(branch_touched) & set(intervening))``.

    Fail-CLOSED policy
    ------------------
    Any non-zero git return code causes the function to return the module-level
    ``_OVERLAP_GIT_ERROR_SENTINEL`` (a non-empty list), so the caller
    re-verifies rather than skipping the gate on uncertainty.  This is the
    opposite of ``_check_post_merge_equivalence``, which fails open, because
    here a false-skip of re-verify risks landing a broken main (catastrophic),
    whereas a false-trigger of re-verify costs only throughput.

    Parameters
    ----------
    task_worktree:
        Worktree whose HEAD is the branch tip.
    rebased_from:
        The main SHA the branch was originally merged against (original
        ``expected_main`` before main moved).
    rebased_onto:
        The current main SHA the branch was rebased onto.
    git_ops:
        Provides ``project_root`` for cross-worktree git operations.
    task_id:
        Optional task identifier for log messages.
    """
    label = task_id or '<unknown>'

    # Fail-CLOSED guard: rebased_from is required to compute the intervening
    # delta.  If it is falsy (e.g. a future caller passes
    # reverify_on_rebase=True with expected_main=None), we cannot safely
    # determine whether the rebase is disjoint, so return the sentinel to
    # force re-verification rather than silently passing None to git.
    if not rebased_from:
        logger.warning(
            'rebase-delta-overlap: rebased_from is falsy (%r); failing '
            'closed. task_id=%s',
            rebased_from, label,
        )
        return _OVERLAP_GIT_ERROR_SENTINEL

    # 1. Resolve branch HEAD
    rc, head_out, head_err = await _run(
        ['git', 'rev-parse', 'HEAD'], cwd=task_worktree,
    )
    if rc != 0:
        logger.warning(
            'rebase-delta-overlap: rev-parse HEAD failed in %s '
            '(rc=%d, err=%s); failing closed. task_id=%s',
            task_worktree, rc, head_err.strip(), label,
        )
        return _OVERLAP_GIT_ERROR_SENTINEL
    branch_head = head_out.strip()

    # 2. merge-base(branch_head, rebased_from) → common fork point
    rc, mb_out, mb_err = await _run(
        ['git', 'merge-base', branch_head, rebased_from],
        cwd=git_ops.project_root,
    )
    if rc != 0:
        logger.warning(
            'rebase-delta-overlap: merge-base %s %s failed '
            '(rc=%d, err=%s); failing closed. task_id=%s',
            branch_head[:8], rebased_from[:8] if len(rebased_from) >= 8 else rebased_from,
            rc, mb_err.strip(), label,
        )
        return _OVERLAP_GIT_ERROR_SENTINEL
    base = mb_out.strip()

    # 3. branch_touched: files the branch changed since the fork point
    rc, bt_out, bt_err = await _run(
        [
            'git', 'diff', '--name-only', '--no-renames',
            base, branch_head, '--', ':!.task/',
        ],
        cwd=git_ops.project_root,
    )
    if rc != 0:
        logger.warning(
            'rebase-delta-overlap: branch-touched diff %s..%s failed '
            '(rc=%d, err=%s); failing closed. task_id=%s',
            base[:8], branch_head[:8], rc, bt_err.strip(), label,
        )
        return _OVERLAP_GIT_ERROR_SENTINEL
    branch_touched = {ln.strip() for ln in bt_out.splitlines() if ln.strip()}

    if not branch_touched:
        # Branch touched nothing (e.g. pure .task/ commit) — no overlap possible.
        return []

    # 4. intervening: files changed on main from rebased_from to rebased_onto
    rc, iv_out, iv_err = await _run(
        [
            'git', 'diff', '--name-only', '--no-renames',
            rebased_from, rebased_onto, '--', ':!.task/',
        ],
        cwd=git_ops.project_root,
    )
    if rc != 0:
        logger.warning(
            'rebase-delta-overlap: intervening diff %s..%s failed '
            '(rc=%d, err=%s); failing closed. task_id=%s',
            rebased_from[:8] if len(rebased_from) >= 8 else rebased_from,
            rebased_onto[:8], rc, iv_err.strip(), label,
        )
        return _OVERLAP_GIT_ERROR_SENTINEL
    intervening = {ln.strip() for ln in iv_out.splitlines() if ln.strip()}

    # 5. Intersection → sorted for deterministic output
    return sorted(branch_touched & intervening)


async def _reverify_rebased_tree(
    git_ops: GitOps,
    req: MergeRequest,
    merge_wt: Path,
    *,
    rebased_from: str,
    rebased_onto: str,
    timeouts: dict[str, int],
    enospc_retries: dict[str, int],
    max_timeouts: int,
    max_enospc: int,
) -> MergeOutcome | None:
    """Shared gate for the disjoint-delta re-verify check.

    Called by the SpeculativeMergeWorker CAS loop after
    ``advance_main`` returns ``'rebased_pending_reverify'``.

    Algorithm
    ---------
    1. Call ``_rebase_delta_touched_overlap`` to compute the intersection of
       the branch-touched file set and the intervening main delta.
    2. **Disjoint** (empty intersection): return ``None``.  The caller can
       advance immediately — the intervening churn cannot interact with the
       branch's changes.  No extra verify call is made.
    3. **Overlapping** (non-empty intersection): log a warning and delegate to
       ``_run_post_merge_verify``, which runs the full post-merge scoped
       verification against the rebased *merge_wt*.

       * ``None`` return → verification passed; caller may advance.
       * ``MergeOutcome`` return → verification failed or disk-guard fired;
         *merge_wt* has already been cleaned up by ``_run_post_merge_verify``.

    Parameters
    ----------
    git_ops:
        GitOps instance.
    req:
        The MergeRequest whose worktree and task_id are inspected.
    merge_wt:
        The merge worktree, currently positioned at the rebased SHA.
    rebased_from:
        The original ``expected_main`` SHA (pre-churn main tip).
    rebased_onto:
        The current main SHA the branch was rebased onto.
    timeouts / enospc_retries / max_timeouts / max_enospc:
        Forwarded verbatim to ``_run_post_merge_verify``.
    """
    overlap = await _rebase_delta_touched_overlap(
        req.worktree, rebased_from, rebased_onto, git_ops,
        task_id=req.task_id,
    )

    if not overlap:
        # Disjoint: the intervening main churn does not intersect the branch's
        # touched files — the rebased tree is safe to advance without re-verify.
        logger.debug(
            'Task %s: rebased tree disjoint from intervening delta (%s..%s) '
            '— skipping re-verify',
            req.task_id, rebased_from[:8], rebased_onto[:8],
        )
        return None

    logger.warning(
        'Task %s: rebased tree overlaps intervening delta (%s..%s) '
        'on %d file(s) [%s] — triggering re-verify',
        req.task_id, rebased_from[:8], rebased_onto[:8],
        len(overlap), ', '.join(overlap[:5]),
    )
    return await _run_post_merge_verify(
        git_ops, req, merge_wt,
        timeouts=timeouts,
        enospc_retries=enospc_retries,
        max_timeouts=max_timeouts,
        max_enospc=max_enospc,
    )


# Maximum number of characters to include in the detail field of a
# ``PostMergePyrightResult`` — keeps the blocked reason string in the
# escalation payload under reasonable size limits.
_POST_MERGE_PYRIGHT_MAX_DETAIL = 2000


async def _check_post_merge_pyright(
    advanced_sha: str,
    git_ops: GitOps,
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig],
    *,
    task_id: str | None = None,
) -> PostMergePyrightResult:
    """Run an unscoped, package-wide type-check against the post-merge main SHA.

    Second post-merge "equivalence" signal: after ``advance_main`` succeeds,
    runs each subproject's ``type_check_command`` VERBATIM (unscoped —
    ``scope_module_config``/``_scope_command`` are NOT called) in a fresh
    detached worktree at ``advanced_sha``.

    Returns a :class:`PostMergePyrightResult` whose ``broken`` property is
    ``True`` when at least one subproject's type-check reports a genuine
    failure (``not passed AND not timed_out``).

    Early returns:
    - Empty or no-type-check-command module_configs → clean (no-op).

    Fail-open:
    - Timeouts (``verify.timed_out``) → skip, log WARNING, treat as clean.
    - Any exception creating the worktree or during verify → log WARNING,
      treat as clean.

    The merge has already landed via ``update-ref`` before this runs; we NEVER
    block a landed merge on a flaky hang or transient infra error.

    .. note::
        **Pre-condition**: This check has no baseline. If a subproject's
        unscoped ``type_check_command`` is already failing on main *before*
        the merge, every subsequent merge will be blocked for that subproject.
        Opting into ``type_check_command`` requires the unscoped check to pass
        on main as a standing pre-condition.  Fix the pre-existing failure
        forward to unblock the queue.
    """
    # Quick-exit: if no module defines a type_check_command there is nothing to check.
    active = [mc for mc in module_configs if mc.type_check_command is not None]
    if not active:
        return PostMergePyrightResult()

    merge_wt: Path | None = None
    try:
        # NOTE: _create_merge_worktree is a private GitOps method.  This helper
        # is its only cross-module caller; promoting it to public (drop the
        # leading underscore) is tracked as a follow-up task.
        merge_wt, _ = await git_ops._create_merge_worktree(advanced_sha)

        async def _run_one(mc: ModuleConfig) -> tuple[ModuleConfig, VerifyResult]:
            # Run only the type-check command verbatim (unscoped).
            # Null out test/lint so run_verification skips them (None => skip).
            type_only_mc = dataclasses.replace(mc, test_command=None, lint_command=None)
            return mc, await run_verification(
                merge_wt, config, type_only_mc,
                max_retries=0, is_merge_verify=True, role='merge',
            )

        # Run all subproject type-checks concurrently to minimise wall-clock
        # impact on the merge queue's push_main delay.
        pairs = await asyncio.gather(*(_run_one(mc) for mc in active))

        failing_subprojects: list[str] = []
        detail_parts: list[str] = []

        for mc, verify in pairs:
            if verify.timed_out:
                logger.warning(
                    'post-merge-pyright: type-check for %r timed out on %s; '
                    'failing open. task_id=%s',
                    mc.prefix, advanced_sha[:12], task_id or '<unknown>',
                )
                continue  # fail open on timeout

            if not verify.passed:
                failing_subprojects.append(mc.prefix)
                # Collect bounded detail from the FIRST failing subproject only
                # (matches PostMergePyrightResult.detail docstring).
                if not detail_parts:
                    raw = verify.failure_report() or verify.type_output or ''
                    if isinstance(raw, str) and raw:
                        detail_parts.append(raw[:_POST_MERGE_PYRIGHT_MAX_DETAIL])

        detail = '\n'.join(detail_parts)
        return PostMergePyrightResult(
            failing_subprojects=failing_subprojects,
            detail=detail,
        )

    except Exception as exc:
        logger.warning(
            'post-merge-pyright: infra error checking %s — failing open. '
            'task_id=%s error=%s',
            advanced_sha[:12], task_id or '<unknown>', exc,
        )
        return PostMergePyrightResult()

    finally:
        if merge_wt is not None:
            await git_ops.cleanup_merge_worktree(merge_wt)


ABANDONED_REASON_PREFIX = 'Post-merge verify timed out'
"""Prefix of the ``MergeOutcome.reason`` string emitted by the merge-queue
loop-breaker.  Downstream classifiers (task steward, dashboard) use this to
recognise a task that has been abandoned after repeated post-merge verify
timeouts rather than a first-time verify failure.  Kept as a module-level
constant so tests and any future callers share a single source of truth."""


WORKTREE_MISSING_REASON_PREFIX = 'Worktree missing'
"""Prefix of the ``MergeOutcome.reason`` string emitted when the task worktree
has been removed out-of-band (typically by a human marking the task ``done``
and cleaning up).  ``TaskWorkflow._submit_to_merge_queue`` recognises this
prefix and re-checks task status: if terminal, it short-circuits to
``WorkflowOutcome.DONE`` instead of cascading into ``_mark_blocked``."""


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
    train_id: str | None = None,
    member_task_ids: list[str] | None = None,
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

    When called from ``_do_train_merge``, *train_id* and *member_task_ids* are
    set so downstream reconciliation can correlate ``merge_attempt`` rows with
    the specific train — not just the tip task_id.
    """
    if event_store is not None:
        data: dict = {'outcome': outcome}
        if attempt is not None:
            data['attempt'] = attempt
        if train_id is not None:
            data['train_id'] = train_id
        if member_task_ids is not None:
            data['member_task_ids'] = member_task_ids
        event_store.emit(
            EventType.merge_attempt, task_id=task_id, phase='merge',
            data=data, duration_ms=duration_ms,
        )


async def _classify_branch_presence(
    git_ops: GitOps,
    event_store: EventStore | None,
    task_id: str,
    branch: str,
    t0: float | None,
) -> MergeOutcome | None:
    """Terminal outcome when *branch* has no live ref in this repo, else None.

    Classifies a queued *branch* (the bare name, e.g. ``"4011"``) by whether
    its ``{branch_prefix}{branch}`` ref still exists:

        ref present            -> None  (proceed with normal merge)
        ref absent + marker    -> already_merged   (merged then cleaned up)
        ref absent + no marker -> unknown_branch    (never existed; likely a misroute)

    The ``unknown_branch`` case fires when a ``merge_request`` for one repo's
    branch is mis-routed to another repo's escalation MCP (e.g. a reify
    ``task/4011`` reaching the dark-factory queue): the branch was never created
    here, so its ref never existed and no merge marker can be on main.

    A bare ``merge_dequeued`` left as the latest event for a task makes the
    dashboard render it as ``in_flight`` (a phantom).  Emitting the matching
    terminal ``merge_attempt`` here means the latest event is terminal, so the
    dashboard excludes it and the mis-routing caller gets a clear signal.
    ``unknown_branch`` is a diagnostic terminal outcome, so emitting it is
    consistent with ``_emit_merge_attempt``'s contract (only infrastructure
    ``blocked`` outcomes are intentionally suppressed there).
    """
    full_branch = f'{git_ops.config.branch_prefix}{branch}'  # bare name -> task/<branch>
    if await git_ops.resolve_branch_sha(full_branch) is not None:
        return None  # common case: ref present, one rev-parse
    if await git_ops.find_merge_marker(full_branch) is not None:
        _emit_merge_attempt(
            event_store, task_id, 'already_merged', duration_ms=_elapsed_ms(t0),
        )
        return MergeOutcome('already_merged')
    _emit_merge_attempt(
        event_store, task_id, 'unknown_branch', duration_ms=_elapsed_ms(t0),
    )
    return MergeOutcome(
        'unknown_branch', reason=f'branch {full_branch!r} not found in repo',
    )


def _emit_train_event(
    event_store: EventStore | None,
    event_type: EventType,
    *,
    task_id: str,
    train_id: str,
    member_task_ids: list[str] | None = None,
    data: dict | None = None,
) -> None:
    """Emit a train lifecycle event.  No-op when *event_store* is None."""
    if event_store is None:
        return
    payload: dict = {'train_id': train_id}
    if member_task_ids is not None:
        payload['member_task_ids'] = member_task_ids
    if data:
        payload.update(data)
    event_store.emit(event_type, task_id=task_id, phase='merge', data=payload)


INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS: int = 3600
"""Maximum age (seconds, wall-clock mtime) for an on-disk ``_merge-*`` worktree
to be considered actively in-flight rather than abandoned.

The evidence shows a cold npm+cargo verify can run for 10–20 minutes on the
first run; the scheduler's /unblock backoff is ~1 hour.  A 1-hour window:
- Covers any plausible in-progress verify (cold build < 1 h in practice).
- Matches the /unblock backoff so a backoff-fired re-request still coalesces
  rather than races if the original verify is still running.
- Flags as abandoned any ``_merge-*`` worktree that has not been touched for
  over an hour — safe to reap and replace with a fresh merger.

Adjusted at module level or injected via `liveness_secs` in tests."""


@dataclass
class _InFlightEntry:
    """Registry slot for a single in-flight merge branch."""

    task_id: str
    enqueued_monotonic: float  # time.monotonic() at acquire time


_INFLIGHT_MERGE_ETA_ESTIMATE_SECS: int = 600
"""Coarse estimate (seconds) for how long a full post-merge verify takes.
Used ONLY to compute a best-effort ETA for coalesced callers.  NOT a
guaranteed bound — cold npm+cargo builds vary widely.  Tests must NOT
assert a specific numeric value for ETA."""


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

    def acquire(self, branch: str, task_id: str, future: asyncio.Future) -> bool:
        """Atomic check-and-set: claim *branch* for *task_id*.

        Returns True if the slot was free and has been claimed; False if
        *branch* was already in-flight (caller should coalesce).

        On success, registers a ``done_callback`` on *future* so that
        ``_release(branch)`` fires automatically on every terminal path
        (result set, exception set, or cancellation).
        """
        if branch in self._slots:
            return False
        self._slots[branch] = _InFlightEntry(
            task_id=task_id,
            enqueued_monotonic=time.monotonic(),
        )
        future.add_done_callback(lambda _: self._release(branch))
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
            # (ETA window is 600 s; liveness window is 3600 s — a worktree can
            # legitimately be in-flight long after the ETA estimate runs out.)
            return None
        return int(remaining)

    def _release(self, branch: str) -> None:
        """Remove *branch* from the in-flight registry.  Called by done_callback."""
        self._slots.pop(branch, None)


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


def _emit_merge_coalesced(
    event_store: EventStore | None,
    req: MergeRequest,
    source: str,
    eta: int | None,
) -> None:
    """Emit a merge_coalesced event.  No-op when *event_store* is None.

    Mirrors the ``_emit_merge_queued`` helper so the coalesced path emits
    an identical-shape record but with ``merge_coalesced`` as the event type.
    """
    if event_store is None:
        return
    data: dict = {
        'branch': req.branch,
        'source': source,
    }
    if eta is not None:
        data['eta_seconds'] = eta
    event_store.emit(
        EventType.merge_coalesced,
        task_id=req.task_id,
        phase='merge',
        data=data,
    )


class _FindInflightWorktreeP(Protocol):
    """Narrow protocol for the disk-scan used by coalesce_or_enqueue_merge_request.

    Only :meth:`find_inflight_merge_worktree` is called on *git_ops* inside
    :func:`coalesce_or_enqueue_merge_request`.  Declaring the parameter with
    this Protocol (rather than the concrete :class:`~orchestrator.git_ops.GitOps`)
    lets test stubs that implement only this method satisfy the type-checker
    without inheriting from the full ``GitOps`` class.
    """

    async def find_inflight_merge_worktree(self, branch: str) -> Path | None: ...
    async def cleanup_merge_worktree(self, merge_wt: Path) -> None: ...


async def coalesce_or_enqueue_merge_request(
    queue: asyncio.Queue,
    req: MergeRequest,
    event_store: EventStore | None,
    registry: InFlightMergeRegistry,
    git_ops: _FindInflightWorktreeP | None = None,
    *,
    liveness_secs: int = INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS,
) -> MergeDispatchResult:
    """De-dup gate for the merge_request MCP chokepoint.

    Consults two sources of truth to detect an already-in-flight merge for
    *req.branch*:

    1. **In-memory registry fast-path** — O(1) dict lookup, covers within-
       process dispatch races (``merge_request`` /unblock spam).
    2. **On-disk worktree scan** (when *git_ops* is not None) — detects the
       workflow's in-flight merger whose ``_merge-*`` worktree exists on disk
       even after a process restart.  Alive worktrees (mtime ≤ liveness_secs)
       are coalesced; stale/abandoned worktrees are reaped via
       ``cleanup_merge_worktree`` and a fresh merge is dispatched (step-10).

    Returns a :class:`MergeDispatchResult` with ``dispatched=True`` if the
    request was enqueued, or ``in_flight=True`` if it was coalesced.

    On coalesce a ``merge_coalesced`` event is emitted so observers can track
    the de-dup rate without polling the queue depth.

    **NOT** used by workflow.py's single-task or train paths — those call
    :func:`enqueue_merge_request` directly.  This function is the
    ``merge_request`` MCP tool's entry point only.
    """
    branch = req.branch

    # ── 1. Registry fast-path ──────────────────────────────────────────
    if registry.is_inflight(branch):
        eta = registry.eta_seconds(branch)
        entry = registry.entry(branch)
        _emit_merge_coalesced(event_store, req, source='registry', eta=eta)
        return MergeDispatchResult(
            dispatched=False,
            in_flight=True,
            branch=branch,
            inflight_task_id=entry.task_id if entry else None,
            eta_seconds=eta,
            source='registry',
        )

    # ── 2. On-disk worktree scan (crash-safety / cross-actor) ──────────
    if git_ops is not None:
        wt = await git_ops.find_inflight_merge_worktree(branch)
        if wt is not None:
            try:
                age = time.time() - wt.stat().st_mtime
            except OSError:
                age = 0.0  # stat failed — treat as alive to be safe
            if age <= liveness_secs:
                # ALIVE: coalesce without enqueuing or reaping
                _emit_merge_coalesced(event_store, req, source='worktree', eta=None)
                return MergeDispatchResult(
                    dispatched=False,
                    in_flight=True,
                    branch=branch,
                    source='worktree',
                )
            else:
                # STALE/ABANDONED: reap the abandoned worktree so a fresh merger
                # can be dispatched.  Foreign-process killing is deliberately out
                # of scope — there is no cwd-based kill utility in the repo
                # (terminate_process_group only handles procs the orchestrator
                # spawned), and git worktree remove --force removes the tree so
                # any orphaned build procs fail when their cwd vanishes.
                logger.warning(
                    'coalesce_or_enqueue_merge_request: reaping stale '
                    '_merge-* worktree %s for branch %r (age=%.0fs > liveness=%ss)',
                    wt, branch, age, liveness_secs,
                )
                await git_ops.cleanup_merge_worktree(wt)
                if event_store is not None:
                    event_store.emit(
                        EventType.worktree_reaped,
                        task_id=req.task_id,
                        phase='merge',
                        data={'branch': branch, 'path': str(wt), 'reason': 'stale_inflight'},
                    )
                # Fall through to acquire-and-enqueue below

    # ── 3. Atomic acquire-and-enqueue ─────────────────────────────────
    if registry.acquire(branch, req.task_id, req.result):
        try:
            await enqueue_merge_request(queue, req, event_store)
        except BaseException:
            # Slot leak guard: if the enqueue raises (e.g. queue closed,
            # cancellation) before the worker can ever resolve req.result,
            # the done_callback will never fire.  Release the slot explicitly
            # so a future merge_request for this branch can proceed.
            registry._release(branch)
            raise
        return MergeDispatchResult(
            dispatched=True,
            in_flight=False,
            branch=branch,
        )

    # Concurrent dispatch won the race during the (currently no-op) scan await
    eta = registry.eta_seconds(branch)
    entry = registry.entry(branch)
    _emit_merge_coalesced(event_store, req, source='registry', eta=eta)
    return MergeDispatchResult(
        dispatched=False,
        in_flight=True,
        branch=branch,
        inflight_task_id=entry.task_id if entry else None,
        eta_seconds=eta,
        source='registry',
    )


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


@dataclass
class MergeOutcome:
    """Result delivered to the caller via the Future."""

    status: Literal['done', 'conflict', 'blocked', 'already_merged', 'wip_halted', 'done_wip_recovery', 'wip_recovery_no_advance', 'unmerged_state', 'unknown_branch']
    reason: str = ''
    conflict_details: str = ''
    recovery_branch: str | None = None
    overlap_files: list[str] | None = None
    merge_sha: str | None = None
    push_status: str | None = None
    failure_diagnostic: dict[str, str] | None = None
    verify_skipped: bool = False
    """True when the disk guard fired and ``run_scoped_verification`` was never
    called.  Lets callers distinguish a disk-guard short-circuit from an actual
    verification failure in log messages."""


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
    failure_diagnostic: dict[str, str] | None = None  # Populated on non-conflict merge failure


class _TrainMergeHost(Protocol):
    """Narrow Protocol exposing per-worker state required by ``_do_train_merge``.

    Both :class:`MergeWorker` and :class:`SpeculativeMergeWorker` inherit
    :class:`_WipHaltMixin` and define every attribute / constant listed here;
    the Protocol lets ``_do_train_merge`` accept either worker type without
    creating a coupling to the concrete class hierarchy.

    The surface is intentionally narrow — only the state that the shared
    train-merge pipeline actually touches.  Adding new attributes here does
    NOT require touching ``_WipHaltMixin``; both concrete workers already
    define them in their own ``__init__``.
    """

    # ── Git / event dependencies ──────────────────────────────────────────
    _git_ops: GitOps
    _event_store: EventStore | None

    # ── Per-task counters (mutated by shared helpers) ─────────────────────
    _post_merge_verify_timeouts: dict[str, int]
    _post_merge_verify_enospc_retries: dict[str, int]
    _cas_retries: dict[str, int]

    # ── Class-level thresholds ────────────────────────────────────────────
    MAX_POST_MERGE_VERIFY_TIMEOUTS: int
    MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES: int

    # ── WIP halt / abandon helpers ────────────────────────────────────────
    def halt_for_wip(self, reason: str) -> None: ...
    def _abandon_outcome(self, task_id: str, count: int) -> MergeOutcome: ...


async def _do_train_merge(
    worker: _TrainMergeHost,
    req: GroupMergeRequest,
) -> MergeOutcome:
    """Atomic train-merge pipeline shared by MergeWorker and SpeculativeMergeWorker.

    Implements PRD δ₁ spec §9.6:
    (a) Status pre-check — all members must be ``merge-deferred``.
    (b) Tip rebase — ``rebase_onto_main`` rebases the tip onto current main;
        on conflict it aborts the rebase (worktree left clean) and returns False.
    (c) Merge — ``merge_to_main`` performs the --no-ff merge of the TIP branch
        (which, by stacking, already carries all member commits).
    (d) Workspace verify — ``run_scoped_verification`` with ``is_merge_verify=True``
        enforces the workspace-wide post-merge green gate (scenario 5).
    (e) CAS advance — ``advance_main`` atomically updates the main ref.
    (f) Member callbacks — ``req.mark_member_done`` is called for each member
        ONLY after advance succeeds (invariant: members flip iff main lands).

    Pre-checks (a)+(b) are added in steps 6 and 8; this initial implementation
    (step 4) covers the happy path where main is unmoved and all members are
    already merge-deferred.
    """
    # Unpack worker state so the rest of the function reads like the single-task path.
    git_ops = worker._git_ops
    event_store = worker._event_store

    t0 = time.monotonic()
    logger.info(
        'Train %s: starting atomic merge of %d members via tip branch %s',
        req.train_id, len(req.member_task_ids), req.branch,
    )

    _train_emit_kwargs: dict = {
        'train_id': req.train_id,
        'member_task_ids': req.member_task_ids,
    }

    # Telemetry: emit train_started once per *attempt*.  Because the workflow
    # re-parks an incomplete train (MERGE_DEFERRED) for retry, the same
    # train_id can fire train_started on each scheduler iteration until all
    # members are ready.  Consumers should treat train_started as
    # "merge-attempt started" and correlate by (train_id, timestamp) rather
    # than expecting a single occurrence per train lifecycle.
    #
    # base_sha_t0 is a best-effort telemetry read (main at the START of this
    # attempt); it is NOT the CAS expected_main used for the actual advance
    # (that is read after rebase at the (c) anchor below).  A transient git
    # failure here must not abort the core merge path, hence the try/except.
    try:
        base_sha_t0: str = await git_ops.get_main_sha()
    except Exception:
        base_sha_t0 = ''
    _emit_train_event(
        event_store, EventType.train_started,
        task_id=req.task_id, train_id=req.train_id,
        member_task_ids=req.member_task_ids,
        data={'member_count': len(req.member_task_ids), 'base_sha': base_sha_t0},
    )

    # Loop-breaker: if this train's tip has timed out in post-merge verify
    # MAX_POST_MERGE_VERIFY_TIMEOUTS times in a row, abandon without any git
    # work — mirrors MergeWorker._do_merge:2286-2298.  A stuck verify
    # can otherwise burn merge-queue capacity for 30+ minutes per attempt.
    prior_timeouts = worker._post_merge_verify_timeouts.get(req.task_id, 0)
    if prior_timeouts >= worker.MAX_POST_MERGE_VERIFY_TIMEOUTS:
        logger.warning(
            'Train %s: abandoning merge — %d consecutive post-merge '
            'verify timeouts (threshold=%d)',
            req.train_id, prior_timeouts,
            worker.MAX_POST_MERGE_VERIFY_TIMEOUTS,
        )
        _emit_merge_attempt(
            event_store, req.task_id, 'abandoned_verify_timeouts',
            attempt=prior_timeouts, duration_ms=_elapsed_ms(t0),
            **_train_emit_kwargs,
        )
        return worker._abandon_outcome(req.task_id, prior_timeouts)

    # (a) Status pre-check: all members must be 'merge-deferred'.
    statuses = await req.status_check(req.member_task_ids)
    incomplete = [
        (mid, statuses.get(mid, '<missing>'))
        for mid in req.member_task_ids
        if statuses.get(mid) != 'merge-deferred'
    ]
    if incomplete:
        # Emit train_member_deferred for each incomplete member (retryable, NOT a derail).
        for mid, status in incomplete:
            deferred_reason = f"status is {status!r}, expected merge-deferred"
            remaining = [m for m in req.member_task_ids if m != mid]
            _emit_train_event(
                event_store, EventType.train_member_deferred,
                task_id=req.task_id, train_id=req.train_id,
                data={
                    'deferred_task_id': mid,
                    'deferred_reason': deferred_reason,
                    'remaining_members': remaining,
                },
            )
        first_id, first_status = incomplete[0]
        reason = (
            f'{TRAIN_INCOMPLETE_REASON_PREFIX}: member {first_id!r} is '
            f'{first_status!r} (expected merge-deferred)'
        )
        logger.info('Train %s: %s', req.train_id, reason)
        _emit_merge_attempt(event_store, req.task_id, 'train_incomplete', duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
        return MergeOutcome('blocked', reason=reason)

    # (b) Rebase tip onto current main so the --no-ff merge is clean.
    ok = await git_ops.rebase_onto_main(req.worktree)
    if not ok:
        reason = (
            f'{TRAIN_REBASE_CONFLICT_REASON_PREFIX}: tip branch '
            f'{req.branch!r} conflicts with current main; rebase aborted '
            f'— resolve in the tip worktree'
        )
        logger.info('Train %s: %s', req.train_id, reason)
        _emit_train_event(
            event_store, EventType.train_derailed,
            task_id=req.task_id, train_id=req.train_id,
            member_task_ids=req.member_task_ids,
            data={'derail_reason': reason},
        )
        _emit_merge_attempt(event_store, req.task_id, 'train_rebase_conflict', duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
        return MergeOutcome('blocked', reason=reason)

    # (c) Read current main HEAD AFTER rebase (so CAS expected_main is fresh).
    main_sha = await git_ops.get_main_sha()

    # (d) --no-ff merge of the tip branch (carries all member commits by stacking).
    merge_result = await git_ops.merge_to_main(req.worktree, req.branch)
    if merge_result.conflicts or not merge_result.success:
        if merge_result.merge_worktree:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        reason = merge_result.details or 'Merge failed'
        logger.info('Train %s: merge failed: %s', req.train_id, reason)
        _emit_train_event(
            event_store, EventType.train_derailed,
            task_id=req.task_id, train_id=req.train_id,
            member_task_ids=req.member_task_ids,
            data={'derail_reason': reason},
        )
        _emit_merge_attempt(event_store, req.task_id, 'conflict' if merge_result.conflicts else 'merge_failed', duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
        return MergeOutcome('blocked', reason=reason)

    # Enforce invariants explicitly — plain assert is stripped under python -O,
    # and a None merge_commit would silently pass the wrong SHA to member callbacks.
    if merge_result.merge_commit is None:
        if merge_result.merge_worktree:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        raise RuntimeError(
            f'Train {req.train_id}: merge_to_main reported success '
            f'but merge_commit is None'
        )
    merge_commit = merge_result.merge_commit.strip()
    merge_wt = merge_result.merge_worktree
    if merge_wt is None:
        raise RuntimeError(
            f'Train {req.train_id}: merge_to_main reported success '
            f'but merge_worktree is None'
        )

    # (e) Workspace-wide post-merge verify via the shared helper.
    # _run_post_merge_verify runs the pre-verify disk guard, the scoped
    # verification (with ENOSPC prune-retry), and the timeout loop-breaker
    # bookkeeping.  Returns None when verify passes; returns a MergeOutcome
    # (and cleans up merge_wt) when any controlled failure fires.
    verify_outcome = await _run_post_merge_verify(
        git_ops, req, merge_wt,
        timeouts=worker._post_merge_verify_timeouts,
        enospc_retries=worker._post_merge_verify_enospc_retries,
        max_timeouts=worker.MAX_POST_MERGE_VERIFY_TIMEOUTS,
        max_enospc=worker.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES,
    )
    if verify_outcome is not None:
        reason = verify_outcome.reason
        logger.info('Train %s: verify gate blocked: %s', req.train_id, reason)
        _emit_train_event(
            event_store, EventType.train_derailed,
            task_id=req.task_id, train_id=req.train_id,
            member_task_ids=req.member_task_ids,
            data={'derail_reason': reason},
        )
        _emit_merge_attempt(event_store, req.task_id, 'verify_failed', duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
        return verify_outcome

    # (f) CAS-advance main.
    adv = await git_ops.advance_main(
        merge_commit, merge_wt,
        branch=req.branch,
        max_attempts=req.config.max_advance_attempts,
        expected_main=main_sha,
    )

    await git_ops.cleanup_merge_worktree(merge_wt)

    if adv != 'advanced':
        logger.info('Train %s: advance_main returned %r', req.train_id, adv)
        _emit_train_event(
            event_store, EventType.train_derailed,
            task_id=req.task_id, train_id=req.train_id,
            member_task_ids=req.member_task_ids,
            data={'derail_reason': f'Train merge advance failed: {adv}'},
        )
        _emit_merge_attempt(event_store, req.task_id, 'advance_failed', duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
        return MergeOutcome('blocked', reason=f'Train merge advance failed: {adv}')

    # advance succeeded — run equivalence + pyright + push via shared finalize.
    # merge_wt has already been cleaned up above (caller contract for finalize).
    # t0 is passed as started_monotonic so finalize's duration_ms is accurate
    # for the whole train attempt, not just the post-advance window.
    outcome = await _finalize_advanced_merge(
        git_ops, req, event_store,
        merge_commit_fallback=merge_commit,
        base_sha=main_sha,
        started_monotonic=t0,
        cas_retries=worker._cas_retries,
        timeouts=worker._post_merge_verify_timeouts,
        enospc_retries=worker._post_merge_verify_enospc_retries,
        log_label=' (train)',
        train_id=req.train_id,
        member_task_ids=req.member_task_ids,
    )
    if outcome.status != 'done':
        # Equivalence or pyright gate fired — main landed but post-merge gates
        # failed.  Emit derailed and return without flipping members.
        logger.info('Train %s: finalize blocked: %s', req.train_id, outcome.reason)
        _emit_train_event(
            event_store, EventType.train_derailed,
            task_id=req.task_id, train_id=req.train_id,
            member_task_ids=req.member_task_ids,
            data={'derail_reason': outcome.reason},
        )
        return outcome  # no member flips

    # outcome.merge_sha: post-rebase advanced SHA resolved by finalize (rebase-robust).
    advanced_sha: str = outcome.merge_sha  # type: ignore[assignment]

    _emit_train_event(
        event_store, EventType.train_merged,
        task_id=req.task_id, train_id=req.train_id,
        member_task_ids=req.member_task_ids,
        data={'merge_commit_sha': advanced_sha, 'base_sha': main_sha},
    )

    # (g) Flip all members done — ONLY after advance + finalize succeed.
    # Each callback is wrapped in try/except so a single scheduler blip does
    # NOT abort the remaining flips (split-brain prevention) and does NOT
    # surface as a 'blocked' outcome that contradicts the landed git state.
    failed: list[tuple[str, str]] = []
    for member_id in req.member_task_ids:
        try:
            await req.mark_member_done(member_id, advanced_sha)
        except Exception as exc:
            failed.append((member_id, repr(exc)))
            logger.exception(
                'Train %s: mark_member_done failed for member %s after main '
                'advanced to %s',
                req.train_id, member_id, advanced_sha,
            )

    if failed:
        # Cap detail to first 3 failures; a wide train with many failed callbacks
        # would otherwise produce an unboundedly large reason string.
        _MAX_DETAIL = 3
        detail_items = [f'{mid}: {err}' for mid, err in failed[:_MAX_DETAIL]]
        overflow = len(failed) - _MAX_DETAIL
        if overflow > 0:
            detail_items.append(f'... and {overflow} more')
        reason = (
            f'{TRAIN_PARTIAL_FLIP_REASON_PREFIX}: train landed at '
            f'{advanced_sha[:12]} but {len(failed)}/{len(req.member_task_ids)} '
            f'member(s) failed to flip — manual cleanup required: '
            + '; '.join(detail_items)
        )
        logger.warning('Train %s: %s', req.train_id, reason)
        _emit_merge_attempt(event_store, req.task_id, 'train_partial_flip', duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
        return MergeOutcome('done', merge_sha=advanced_sha, reason=reason)

    # _finalize_advanced_merge already emitted merge_attempt 'done'; return its
    # outcome directly so push_status is propagated to the caller.
    logger.info(
        'Train %s: landed at %s; %d members marked done',
        req.train_id, advanced_sha[:12], len(req.member_task_ids),
    )
    return outcome


class _WipHaltMixin:
    """Shared WIP-halt machinery and request-abandoned helper.

    Provides the byte-identical halt-owner methods that both
    :class:`MergeWorker` and :class:`SpeculativeMergeWorker` expose as
    public API to ``workflow.py`` and ``harness.py``.

    Methods-only: each concrete worker's ``__init__`` is responsible for
    creating the instance attributes::

        self._wip_halt = asyncio.Event(); self._wip_halt.set()
        self._halt_owner_esc_id: str | None = None
    """

    # Class-level annotations so pyright sees the attributes without an
    # __init__ on the mixin itself.
    _wip_halt: asyncio.Event
    _halt_owner_esc_id: str | None

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

    def unhalt_wip(self, reason: str | None = None) -> None:
        """Resume the merge queue after WIP conflict resolution."""
        logger.info(
            'Merge queue un-halted (WIP conflict resolved%s)',
            f', reason={reason!r}' if reason else '',
        )
        self._wip_halt.set()
        self._halt_owner_esc_id = None

    @property
    def is_wip_halted(self) -> bool:
        return not self._wip_halt.is_set()

    @property
    def halt_owner_esc_id(self) -> str | None:
        """Read-only public view of the current halt-owner escalation id."""
        return self._halt_owner_esc_id

    def _request_abandoned(self, req: MergeRequest) -> bool:
        """True iff the requester cancelled the result future — drop the request."""
        if req.result.cancelled():
            logger.info(
                'Task %s: merge request abandoned by waiter '
                '(future cancelled) — dropping request without halting queue',
                req.task_id,
            )
            return True
        return False


class MergeWorker(_WipHaltMixin):
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
    # After a post-merge verify fails with an ENOSPC signature, prune stale
    # _merge-* worktrees and retry the verify at most this many times before
    # escalating as transient infra.  Disk pressure is often self-healing, so
    # one retry-after-prune is the conservative middle ground between blindly
    # blocking and looping.  Resets on any successful merge for that task.
    MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES = 1

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
        # WIP halt: cleared when halted, set when running
        self._wip_halt = asyncio.Event()
        self._wip_halt.set()  # not halted initially
        # ID of the escalation that owns the current halt. Registered by the
        # workflow handler after it submits the L1 escalation. Single source
        # of truth for the resolve-callback un-halt path.
        self._halt_owner_esc_id: str | None = None

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

        # 0. Branch-presence guard: a missing branch ref resolves terminally
        # (unknown_branch on a never-existed ref — typically a mis-routed
        # merge_request — or already_merged when the ref was cleaned up after
        # merge).  Runs before the rev-parse HEAD below so a misroute can't be
        # born as a bare merge_dequeued (phantom in_flight on the dashboard).
        guard = await _classify_branch_presence(
            self._git_ops, self._event_store, req.task_id, req.branch, t0,
        )
        if guard is not None:
            return guard

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
            merge_result.merge_commit, req.worktree, self._git_ops, main_sha,
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
            reason = (
                f'{DROPPED_PLAN_TARGETS_REASON_PREFIX}: '
                f'{", ".join(dropped)}. '
                f'Conflict resolution likely dropped planned work. '
                f'Review the merge commit and restore missing files.'
            )
            return MergeOutcome('blocked', reason=reason)

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
            out = await _run_post_merge_verify(
                self._git_ops, req, merge_wt,
                timeouts=self._post_merge_verify_timeouts,
                enospc_retries=self._post_merge_verify_enospc_retries,
                max_timeouts=self.MAX_POST_MERGE_VERIFY_TIMEOUTS,
                max_enospc=self.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES,
            )
            if out is not None:
                return out

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
            return await _finalize_advanced_merge(
                self._git_ops, req, self._event_store,
                merge_commit_fallback=merge_result.merge_commit,
                base_sha=main_sha,
                started_monotonic=t0,
                cas_retries=self._cas_retries,
                timeouts=self._post_merge_verify_timeouts,
                enospc_retries=self._post_merge_verify_enospc_retries,
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
                cas_retries=self._cas_retries,
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


class SpeculativeMergeWorker(_WipHaltMixin):
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
    # Mirror of MergeWorker.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES.
    MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES = 1

    def __init__(
        self,
        git_ops: GitOps,
        queue: asyncio.Queue[MergeRequest],
        event_store: EventStore | None = None,
        on_merge_landed: Callable[[str, str, str], Awaitable[object]] | None = None,
    ):
        self._git_ops = git_ops
        self._queue = queue
        self._event_store = event_store
        # Post-merge notification hook — called with (task_id, base_sha,
        # advanced_sha) after each 'done' merge.  Wrapped in try/except so a
        # coordinator bug never blocks or fails the merge.  See task 1592.
        self._on_merge_landed = on_merge_landed
        # Internal pipeline: Merger → Verifier
        self._verifier_queue: asyncio.Queue[SpeculativeItem | None] = asyncio.Queue()
        self._running = True
        self._cas_retries: dict[str, int] = {}
        # Per-task gate-iteration counter for rebased_pending_reverify results.
        # Separate from _cas_retries so that disjoint rebases (where no extra
        # verify is run) do not consume the CAS-failure budget.  Bounded by
        # MAX_CAS_RETRIES to prevent runaway gate loops on a perpetually-moving
        # main; cleared on exhaustion or when the task leaves the queue.
        self._gate_retries: dict[str, int] = {}
        # Per-task consecutive post-merge-verify-timeout counter.  Bumped by
        # the Verifier when a post-merge verify finishes with timed_out=True,
        # cleared on a successful CAS advance.  Keyed by task_id; lives
        # across submissions so an orchestrator re-queue of the same task
        # continues to feed the same counter.
        self._post_merge_verify_timeouts: dict[str, int] = {}
        # Per-task ENOSPC prune-and-retry counter (mirror of MergeWorker's;
        # see MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES).  Persists across
        # submissions, reset on a successful CAS advance.
        self._post_merge_verify_enospc_retries: dict[str, int] = {}
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
                # Drop-on-detection: workflow soft-cancelled before worker
                # dequeued.  Skipping merge work avoids the orphan-halt
                # window where no escalation owner is registered.
                if self._request_abandoned(req):
                    spec_base = None
                    self._inflight_req = None
                    continue
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

                    # ── Train dispatch: GroupMergeRequest bypasses speculative pipeline ──
                    # A train changes main topology atomically; the next item must not be
                    # pre-merged against an unverified train commit, so NO speculative
                    # look-ahead is performed.  _do_train_merge owns its own rebase /
                    # merge / verify / advance / cleanup pipeline; the outcome rides the
                    # verifier queue via immediate_outcome so the standard future-resolution
                    # path handles it.
                    #
                    # Pipeline-ordering contract: trains should be enqueued (by δ₂) only
                    # when the merge pipeline is idle.  If spec_base is set here, the
                    # previous regular request's merge commit is on the verifier queue but
                    # its advance_main has not yet run — the train will rebase and CAS
                    # against a temporarily stale main.  advance_main's internal retry loop
                    # absorbs the resulting CAS race, but adds latency and event noise.
                    if isinstance(req, GroupMergeRequest):
                        if spec_base is not None:
                            logger.warning(
                                'Train %s: dequeued while speculative merge is '
                                'in-flight (spec_base=%s); advance_main retries '
                                'will absorb the CAS race — enqueuer should wait '
                                'for an idle pipeline before submitting a train',
                                req.train_id, spec_base[:12],
                            )
                        outcome = await _do_train_merge(self, req)
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=actual_main, speculative=False,
                            skip_verify=False, immediate_outcome=outcome,
                            started_monotonic=t0,
                        ))
                        spec_base = None
                        self._inflight_req = None
                        continue

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

                    # ── Step 0.5: branch-presence guard ───────────────────────
                    # A missing branch ref resolves terminally before the
                    # rev-parse HEAD below: unknown_branch on a never-existed
                    # ref (typically a mis-routed merge_request) or
                    # already_merged when the ref was cleaned up post-merge.
                    # Riding it through the verifier queue as an immediate
                    # outcome means the latest event is the terminal
                    # merge_attempt _classify_branch_presence emitted, not a
                    # bare merge_dequeued (phantom in_flight on the dashboard).
                    guard = await _classify_branch_presence(
                        self._git_ops, self._event_store, req.task_id,
                        req.branch, t0,
                    )
                    if guard is not None:
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=actual_main, speculative=speculative,
                            skip_verify=False, immediate_outcome=guard,
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
                        # failure_diagnostic is NOT populated here: this failure
                        # occurs before a merge attempt (git cannot even read the
                        # worktree HEAD), so the merge_to_main diagnostic fields
                        # (base_sha, branch_ref_in_worktree, etc.) are meaningless.
                        # failure_diagnostic is only set on genuine merge_to_main
                        # non-conflict failures downstream.
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
                        _diag = await self._build_merge_failure_diagnostic(
                            req,
                            base_sha=merge_result.pre_merge_sha or base_for_merge,
                            base_label='speculative' if speculative else 'main_head',
                            git_stderr=merge_result.details,
                        )
                        _rendered = self._render_failure_diagnostic(_diag)
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=base_for_merge, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=MergeOutcome(
                                'blocked',
                                reason=f'{merge_result.details}\n{_rendered}',
                                failure_diagnostic=_diag,
                            ),
                            failure_diagnostic=_diag,
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
                    # Pass base_for_merge (the pre-merge main tip the merge
                    # was computed against — actual or speculative) so the
                    # subtraction is rebase-robust.
                    drop_result = await _check_plan_targets_in_tree(
                        merge_commit, req.worktree, self._git_ops, base_for_merge,
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
                        reason = (
                            f'{DROPPED_PLAN_TARGETS_REASON_PREFIX}: '
                            f'{", ".join(dropped)}. '
                            f'Conflict resolution likely dropped '
                            f'planned work. Review the merge commit '
                            f'and restore missing files.'
                        )
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=base_for_merge, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=MergeOutcome('blocked', reason=reason),
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
                except WorktreeMissing as exc:
                    # The task worktree was removed out-of-band (typical
                    # cause: a human marked the task done and cleaned up
                    # while we were processing it).  Surface as ``blocked``
                    # with a recognisable reason; ``TaskWorkflow`` re-checks
                    # task status and short-circuits to DONE if terminal.
                    logger.info(
                        f'Task {req.task_id}: merger detected missing '
                        f'worktree {exc.path} — surfacing as blocked'
                    )
                    if (
                        merge_result_local is not None
                        and merge_result_local.merge_worktree
                    ):
                        with contextlib.suppress(Exception):
                            await self._git_ops.cleanup_merge_worktree(
                                merge_result_local.merge_worktree
                            )
                    merge_result_local = None
                    if (
                        self._inflight_req is not None
                        and not self._inflight_req.result.done()
                    ):
                        self._inflight_req.result.set_result(
                            MergeOutcome(
                                'blocked',
                                reason=(
                                    f'{WORKTREE_MISSING_REASON_PREFIX}: '
                                    f'{exc.path}'
                                ),
                            )
                        )
                    spec_base = None
                    self._inflight_req = None
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

            # Drop-on-detection: if the workflow that submitted this request
            # cancelled its result future after the merger handed the item
            # off, skip verify+CAS and any halt sites entirely.  Cleans up
            # the merge worktree to avoid leaks.
            if self._request_abandoned(req):
                if item.merge_wt is not None:
                    with contextlib.suppress(BaseException):
                        await self._git_ops.cleanup_merge_worktree(item.merge_wt)
                # Treat as failed for chain-invalidation: any speculative
                # item built on this one's commit is now stale.
                n_failed = True
                self._speculation_slot.set()
                continue

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

    async def _build_merge_failure_diagnostic(
        self,
        req: MergeRequest,
        base_sha: str,
        base_label: str,
        git_stderr: str,
    ) -> dict[str, str]:
        """Build the four-key failure diagnostic dict for a non-conflict merge failure.

        ``branch_ref_in_worktree`` is resolved via ``resolve_branch_sha`` which runs
        ``git rev-parse`` against ``project_root`` (the primary repo), not the temporary
        merge worktree.  The value is equivalent: git ref namespaces (``refs/heads/*``)
        are shared across all worktrees of a repository, so project_root and the merge
        worktree always agree on the branch HEAD SHA.  The key name uses the worktree
        framing because that is the conceptually meaningful question ("what SHA would the
        merge worktree have seen for this branch?").
        """
        full_branch = f'{self._git_ops.config.branch_prefix}{req.branch}'
        resolved = await self._git_ops.resolve_branch_sha(full_branch)
        return {
            'base_sha': base_sha,
            'base_label': base_label,
            'branch_ref_in_worktree': resolved or '<unresolved>',
            'git_stderr': git_stderr,
        }

    @staticmethod
    def _render_failure_diagnostic(diag: dict[str, str]) -> str:
        """Render the diagnostic dict as a labelled key=value line for inclusion in reason.

        ``git_stderr`` is intentionally omitted from the rendered line: the full git
        output is already prepended to ``reason`` via ``merge_result.details``, so
        repeating even a truncated first line would be redundant noise in the human-
        facing field.  The structured ``git_stderr`` value remains accessible via the
        ``failure_diagnostic`` dict.
        """
        return (
            f"[merge-failure] "
            f"base_sha={diag['base_sha']} "
            f"base_label={diag['base_label']} "
            f"branch_ref_in_worktree={diag['branch_ref_in_worktree']}"
        )

    async def _remerge(self, req: MergeRequest, started_monotonic: float | None) -> SpeculativeItem:
        """Re-merge a request against actual main after speculation invalidation."""
        actual_main = await self._git_ops.get_main_sha()
        merge_result = await self._git_ops.merge_to_main(
            req.worktree, req.branch, base_sha=None,
        )

        # ── Speculation-race retry ─────────────────────────────────────────────
        # When the first attempt fails with the load-bearing git porcelain phrase
        # ``not something we can merge`` (detected by _is_speculation_race) AND
        # the merge ran against a stale base (pre_merge_sha != actual_main), main
        # advanced between our get_main_sha() read and merge_to_main's own read,
        # so the worktree was built against a commit no longer on main.  Retry
        # exactly once against a freshly-read main HEAD to clear the stale-base
        # environment.
        #
        # Design note: git emits this phrase when the merge argument (the branch
        # ref) cannot be resolved to a commit — e.g. a stale ref cache after
        # rapid concurrent pushes.  The pre_merge_sha != actual_main gate pins
        # the retry to cases where the base genuinely drifted; if the branch ref
        # was deleted or force-pushed between the two calls, the retry will fail
        # identically.  The full stderr is attached to the warning log below for
        # post-hoc diagnosis.
        #
        # merge_to_main self-cleans its worktree on non-conflict failure, so
        # merge_result.merge_worktree is None here — no pre-retry cleanup needed.
        if (
            not merge_result.success
            and not merge_result.conflicts
            and _is_speculation_race(merge_result.details)
            and merge_result.pre_merge_sha is not None
            and merge_result.pre_merge_sha != actual_main
        ):
            retry_main = await self._git_ops.get_main_sha()
            logger.warning(
                'Task %s: speculation-race detected (first_base=%s, stderr=%r) '
                '— retrying against main %s',
                req.task_id, merge_result.pre_merge_sha[:8],
                merge_result.details[:120], retry_main[:8],
            )
            retry_result = await self._git_ops.merge_to_main(
                req.worktree, req.branch, base_sha=retry_main,
            )
            if retry_result.success:
                logger.info(
                    'Task %s: merge_retry_after_speculation_race succeeded '
                    '(retry_base=%s)',
                    req.task_id, retry_main[:8],
                )
                # skip_verify is UNCONDITIONALLY False on the race-retry path.
                #
                # merge_to_main pins pre_merge_sha to the explicit base_sha=retry_main,
                # so the old expression (req.pre_rebased and pre_merge_sha==retry_main)
                # was a tautology that degenerated to skip_verify=req.pre_rebased.
                # However, this branch is reached ONLY after the gate confirmed main
                # advanced (merge_result.pre_merge_sha != actual_main): the branch was
                # pre-rebased onto the OLD main while the retry merges it against the
                # newer retry_main, integrating main commits the branch never
                # incorporated.  The documented skip_verify invariant
                # ('pre_rebased AND main unchanged', SpeculativeItem.skip_verify) does
                # NOT hold; skipping verification would let semantically-unverified
                # main commits land on the protected branch.  Always verify.
                return SpeculativeItem(
                    request=req, merge_result=retry_result,
                    merge_wt=retry_result.merge_worktree,
                    base_sha=retry_main, speculative=False, skip_verify=False,
                    started_monotonic=started_monotonic,
                )
            if retry_result.conflicts:
                _emit_merge_attempt(
                    self._event_store, req.task_id, 'conflict',
                    duration_ms=_elapsed_ms(started_monotonic),
                )
                if retry_result.merge_worktree:
                    await self._git_ops.cleanup_merge_worktree(retry_result.merge_worktree)
                return SpeculativeItem(
                    request=req, merge_result=None, merge_wt=None,
                    base_sha=retry_main, speculative=False, skip_verify=False,
                    immediate_outcome=MergeOutcome(
                        'conflict', conflict_details=retry_result.details,
                    ),
                    started_monotonic=started_monotonic,
                )
            # Retry non-conflict failure — build μ diagnostics for BOTH attempts
            # and surface them together in reason and failure_diagnostic.
            if retry_result.merge_worktree:
                await self._git_ops.cleanup_merge_worktree(retry_result.merge_worktree)
            first_diag = await self._build_merge_failure_diagnostic(
                req,
                base_sha=merge_result.pre_merge_sha or actual_main,
                base_label='main_head',
                git_stderr=merge_result.details,
            )
            retry_diag = await self._build_merge_failure_diagnostic(
                req,
                base_sha=retry_result.pre_merge_sha or retry_main,
                base_label='main_head',
                git_stderr=retry_result.details,
            )
            first_rendered = self._render_failure_diagnostic(first_diag)
            retry_rendered = self._render_failure_diagnostic(retry_diag)
            # Combined failure_diagnostic: retry (final) attempt's μ 4 keys
            # plus first-attempt extras under prefixed keys.
            combined_diag: dict[str, str] = {
                **retry_diag,
                'first_attempt_base_sha': first_diag['base_sha'],
                'first_attempt_git_stderr': first_diag['git_stderr'],
            }
            combined_reason = (
                f'Attempt 1: {merge_result.details}\n{first_rendered}\n'
                f'Attempt 2 (retry against main {retry_main[:8]}): '
                f'{retry_result.details}\n{retry_rendered}'
            )
            retry_outcome = MergeOutcome(
                'blocked',
                reason=combined_reason,
                failure_diagnostic=combined_diag,
            )
            return SpeculativeItem(
                request=req, merge_result=None, merge_wt=None,
                base_sha=retry_main, speculative=False, skip_verify=False,
                immediate_outcome=retry_outcome,
                failure_diagnostic=combined_diag,
                started_monotonic=started_monotonic,
            )
        # ── END speculation-race retry ─────────────────────────────────────────

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
            diag = await self._build_merge_failure_diagnostic(
                req,
                base_sha=merge_result.pre_merge_sha or actual_main,
                base_label='main_head',
                git_stderr=merge_result.details,
            )
            rendered = self._render_failure_diagnostic(diag)
            outcome = MergeOutcome(
                'blocked',
                reason=f'{merge_result.details}\n{rendered}',
                failure_diagnostic=diag,
            )
            return SpeculativeItem(
                request=req, merge_result=None, merge_wt=None,
                base_sha=actual_main, speculative=False, skip_verify=False,
                immediate_outcome=outcome,
                failure_diagnostic=diag,
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
                out = await _run_post_merge_verify(
                    self._git_ops, req, merge_wt,
                    timeouts=self._post_merge_verify_timeouts,
                    enospc_retries=self._post_merge_verify_enospc_retries,
                    max_timeouts=self.MAX_POST_MERGE_VERIFY_TIMEOUTS,
                    max_enospc=self.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES,
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
            if out is None:
                logger.info(
                    f'Task {req.task_id}: verify end (merge={merge_commit[:8]}, '
                    f'passed=True)'
                )
            elif out.verify_skipped:
                # Disk guard fired — run_scoped_verification was never called;
                # log 'skipped' rather than 'passed=False' to avoid misleading
                # post-mortem triage of merge-queue stalls (2026-06-01).
                logger.info(
                    f'Task {req.task_id}: verify skipped: low disk '
                    f'(merge={merge_commit[:8]})'
                )
                if not req.result.done():
                    req.result.set_result(out)
                return False
            else:
                logger.info(
                    f'Task {req.task_id}: verify end (merge={merge_commit[:8]}, '
                    f'passed=False)'
                )
                if not req.result.done():
                    req.result.set_result(out)
                return False
        else:
            logger.info(
                f'Task {req.task_id}: skipping re-verification '
                f'(pre-rebased, main unchanged)'
            )

        # ── Step 5: CAS advance_main ──────────────────────────────────
        # current_sha tracks the merge SHA to pass to advance_main.  After a
        # clean rebase (rebased_pending_reverify), the gate clears it to the
        # post-rebase SHA so the next advance_main call lands the verified tree.
        current_sha = merge_commit
        retries = 0
        while True:
            result = await self._git_ops.advance_main(
                current_sha, merge_wt,
                branch=req.branch,
                max_attempts=req.config.max_advance_attempts,
                expected_main=item.base_sha,
                reverify_on_rebase=True,
            )

            if result == 'advanced':
                # Cleanup merge_wt BEFORE the finalize gate: neither the
                # equivalence check (uses req.worktree) nor the pyright check
                # (builds its own detached worktree) reads merge_wt — cleaning
                # it here lowers peak worktree count under disk pressure and
                # mirrors MergeWorker which already cleans merge_wt right after
                # advance_main.
                self._gate_retries.pop(req.task_id, None)
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                outcome = await _finalize_advanced_merge(
                    self._git_ops, req, self._event_store,
                    merge_commit_fallback=merge_commit,
                    base_sha=item.base_sha,
                    started_monotonic=item.started_monotonic,
                    cas_retries=self._cas_retries,
                    timeouts=self._post_merge_verify_timeouts,
                    enospc_retries=self._post_merge_verify_enospc_retries,
                    log_label=' (speculative)',
                )
                if not req.result.done():
                    req.result.set_result(outcome)
                # SMW-only post-merge notification hook (task 1592).  Fires only
                # on a 'done' landing: _finalize_advanced_merge may instead
                # return a 'blocked' outcome (equivalence/pyright gate), and
                # main's pre-refactor inline code reached this hook only after
                # those gates passed (they returned early on failure).  Guard on
                # status == 'done' to preserve that semantics; outcome.merge_sha
                # carries the advanced SHA that the old inline code passed.
                # MergeWorker deliberately has no such hook.
                if (
                    outcome.status == 'done'
                    and outcome.merge_sha is not None
                    and self._on_merge_landed is not None
                ):
                    try:
                        await self._on_merge_landed(
                            req.task_id, item.base_sha, outcome.merge_sha
                        )
                    except Exception:
                        logger.warning(
                            'on_merge_landed hook raised for task %s; ignoring (fail-open)',
                            req.task_id,
                            exc_info=True,
                        )
                return True

            if result == 'rebased_pending_reverify':
                # advance_main rebased merge_wt onto the new main but did NOT
                # update-ref.  Read side channels immediately before any further
                # advance_main call could overwrite them.
                # Use getattr with None defaults so a missing attribute raises a
                # clear AssertionError rather than a bare AttributeError, turning
                # a silent contract violation into an observable failure.
                rebased_sha = getattr(self._git_ops, '_last_advanced_sha', None)
                rebased_from = getattr(self._git_ops, '_rebased_from', None)
                rebased_onto = getattr(self._git_ops, '_rebased_onto', None)
                if rebased_sha is None or rebased_from is None or rebased_onto is None:
                    raise AssertionError(
                        f'advance_main returned rebased_pending_reverify but '
                        f'side-channel attributes are not all set (task '
                        f'{req.task_id}): _last_advanced_sha={rebased_sha!r}, '
                        f'_rebased_from={rebased_from!r}, '
                        f'_rebased_onto={rebased_onto!r}'
                    )

                gate = await _reverify_rebased_tree(
                    self._git_ops, req, merge_wt,
                    rebased_from=rebased_from,
                    rebased_onto=rebased_onto,
                    timeouts=self._post_merge_verify_timeouts,
                    enospc_retries=self._post_merge_verify_enospc_retries,
                    max_timeouts=self.MAX_POST_MERGE_VERIFY_TIMEOUTS,
                    max_enospc=self.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES,
                )
                if gate is not None:
                    # Overlapping delta, verify failed (or disk guard fired).
                    # _run_post_merge_verify already cleaned up merge_wt.
                    if not req.result.done():
                        req.result.set_result(gate)
                    return False

                # Disjoint, or overlap+green: advance with the verified rebased
                # SHA.  Use the dedicated _gate_retries counter (not _cas_retries)
                # so benign disjoint rebases do not draw from the CAS-failure
                # budget.  Both counters are bounded by MAX_CAS_RETRIES to
                # prevent runaway loops; they are tracked independently.
                current_sha = rebased_sha
                gate_total = self._gate_retries.get(req.task_id, 0) + 1
                self._gate_retries[req.task_id] = gate_total
                if gate_total > self.MAX_CAS_RETRIES:
                    self._gate_retries.pop(req.task_id, None)
                    logger.warning(
                        'Task %s: gate retry limit exhausted after gate '
                        'cleared (%d attempts)',
                        req.task_id, self.MAX_CAS_RETRIES,
                    )
                    _emit_merge_attempt(
                        self._event_store, req.task_id, 'cas_exhausted',
                        attempt=gate_total,
                        duration_ms=_elapsed_ms(item.started_monotonic),
                    )
                    await self._git_ops.cleanup_merge_worktree(merge_wt)
                    if not req.result.done():
                        req.result.set_result(MergeOutcome(
                            'blocked',
                            reason=(
                                f'Gate retry limit exhausted after '
                                f'{self.MAX_CAS_RETRIES} attempts for task '
                                f'{req.task_id}'
                            ),
                        ))
                    return False

                # Rebuild item with base_sha = rebased_onto so the NEXT call to
                # advance_main uses rebased_onto as expected_main.  On a second
                # rebase, _rebased_from will be set to rebased_onto (not the
                # original fork point), so _rebase_delta_touched_overlap only
                # computes the INCREMENTAL new delta — not the full interval
                # from the original base.  This ensures repeated gate verifies
                # on a hot main are scoped to new churn only.
                item = SpeculativeItem(
                    request=item.request,
                    merge_result=item.merge_result,
                    merge_wt=item.merge_wt,
                    base_sha=rebased_onto,
                    speculative=item.speculative,
                    skip_verify=item.skip_verify,
                    started_monotonic=item.started_monotonic,
                )
                logger.info(
                    'Task %s: gate cleared (disjoint or green re-verify); '
                    'advancing with rebased SHA %s (gate attempt %d/%d)',
                    req.task_id, rebased_sha[:8],
                    gate_total, self.MAX_CAS_RETRIES,
                )
                _emit_merge_attempt(
                    self._event_store, req.task_id, 'gate_retry',
                    attempt=gate_total,
                    duration_ms=_elapsed_ms(item.started_monotonic),
                )
                continue

            if result in _HALT_ADVANCE_RESULTS and self._request_abandoned(req):
                # Workflow soft-cancelled mid-merge: dropping the request
                # prevents the orphan-halt window where no escalation
                # owner is registered (2026-05-04 incident).
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                if result in ('unmerged_state', 'pop_conflict_no_advance'):
                    self._cas_retries.pop(req.task_id, None)
                    self._gate_retries.pop(req.task_id, None)
                return False
            if result != 'cas_failed':
                self._gate_retries.pop(req.task_id, None)
                outcome = await _map_advance_failure(
                    self._git_ops, result,
                    task_id=req.task_id,
                    merge_commit_fallback=merge_commit,
                    halt=self.halt_for_wip,
                    cas_retries=self._cas_retries,
                )
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                if not req.result.done():
                    req.result.set_result(outcome)
                return False

            # result == 'cas_failed' — transient, retry with limit
            retries += 1
            total = self._cas_retries.get(req.task_id, 0) + 1
            self._cas_retries[req.task_id] = total
            if total > self.MAX_CAS_RETRIES:
                self._cas_retries.pop(req.task_id, None)
                self._gate_retries.pop(req.task_id, None)
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
