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
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, MergeResult, WorktreeMissing, _run
from orchestrator.verify import (
    VerifyResult,
    _resolve_verify_timeout,
    run_scoped_verification,
    run_verification,
)

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

MAX_AUTO_CHAINED_GENERATIONS = 2
"""Maximum number of consecutive auto-chained generations per branch lineage (γ2 / PRD D3).

When the branch tip advances during verify (tip-advanced pathology), the worker
auto-chains a gen-(n+1) MergeRequest for the delta.  After this many consecutive
advances the chain is broken and the request is escalated to humans via a 'blocked'
outcome using the :data:`POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX`.  Not configurable."""

_MERGE_AHEAD_BOUND = 1
"""Maximum number of counted (non-speculative, non-train) items that may sit in
the SpeculativeMergeWorker verifier queue simultaneously (Mechanism 1, task 1646).

.. note:: This constant is an input to the startup liveness-margin guard.
   See :func:`check_merge_liveness_margin` for the coupling between this bound,
   the verify timeout, and :data:`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS`.

With BOUND=1 the Merger runs at most one non-speculative merge ahead of the
Verifier: after enqueuing a counted item the Merger blocks at
``_merge_ahead_cap.acquire()`` until the Verifier drains that item, at which
point it re-reads a fresh main HEAD for the next merge.  Values in [1, 2] are
safe; higher values allow more build-ahead but increase staleness risk.

Cap invariants (all verified by integration tests):
- Acquired at the single success-enqueue site in _merger_loop for non-speculative
  blocking-path items (trains continue before this site; speculative items are
  governed by _speculation_slot instead).
- Released ON-DRAIN in _verifier_loop, immediately after ``_verifier_queue.get()``
  returns a non-None item, before any branching or item reassignment.  This
  uniform placement covers all drain paths (normal verify, immediate_outcome,
  chain-invalidation discard+_remerge, abandoned early-continue) with a single
  release point and no risk of double-release (each counted item has exactly one
  drain in the FIFO).
- Released by stop() (over-release of a plain Semaphore is safe) so a merger
  blocked at acquire() unblocks cleanly at shutdown."""

_HEARTBEAT_POLL_S: float = 30.0
"""How often _heartbeat_loop wakes up to call _maybe_log_queue_heartbeat.

The heartbeat loop polls this frequently; the actual emission rate is governed
by the per-instance _heartbeat_interval_s (default 300 s).  Keeping the poll
period short (30 s) means the first heartbeat fires within ~30 s of startup
when depth > 0 (because _last_heartbeat_at is initialised to 0.0, making the
rate-limit check pass immediately on the first poll), then subsequently no
more often than _heartbeat_interval_s, without adding measurable overhead."""

AUTO_CHAIN_GENERATIONS_ENABLED: bool = False
"""Kill-switch for the γ2 generation auto-chaining producer.

MUST remain False until γ3 lands BOTH:
  1. The workflow.py 'superseded' consumer handler (so the workflow.py:3963-4071
     single-task merge consumer has a branch for 'superseded' outcomes instead of
     falling through to _mark_blocked with an empty reason).
  2. The gen-(n+1) registry slot handoff via ATTACH_AND_CHAIN (re-acquiring the
     branch slot in InFlightMergeRegistry for the chained request without tripping
     the gen-1 done-callback double-release, and threading TerminalOutcomeRetention
     from the harness into the workers — harness.py:3238 omits both today).

While False, _finalize_advanced_merge ignores chain_ctx on equivalence failures
and returns 'blocked' exactly as before γ2, so no 'superseded' outcome can reach
the workflow consumer."""


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
        timed_out_subprojects: Prefixes of subprojects whose unscoped
            type-check command timed out.  A subproject can appear in both
            this list and ``failing_subprojects`` when ``block_on_timeout=True``
            (pre-advance gate), or only here when ``block_on_timeout=False``
            (post-advance fail-open path).  Empty list means no timeouts.
        detail: Bounded human-readable detail from the first failing
            subproject's output, for inclusion in the escalation reason.
    """

    failing_subprojects: list[str] = field(default_factory=list)
    timed_out_subprojects: list[str] = field(default_factory=list)
    detail: str = ''

    @property
    def broken(self) -> bool:
        """True when at least one subproject's type-check genuinely failed."""
        return bool(self.failing_subprojects)

    @property
    def timed_out(self) -> bool:
        """True when at least one subproject's type-check timed out."""
        return bool(self.timed_out_subprojects)


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
    # merge (no `.task/` dir and no warm cargo cache), so the cold
    # timeout applies despite `_is_verify_cold`'s filesystem heuristic
    # classifying them as warm.  The per-command cold timeout used here
    # is `merge_verify_cold_command_timeout_secs` (config default 7200 s)
    # if set, falling back to `verify_cold_command_timeout_secs` then warm.
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

    # Pre-advance, fail-closed unscoped type-check gate.
    # Runs ONLY type_check_command unscoped against the already-created merge_wt
    # (before advance_main).  No-op when no module declares a type_check_command.
    gate = await _run_unscoped_typechecks(
        merge_wt, req.config, req.module_configs,
        block_on_timeout=True, task_id=req.task_id,
    )
    if gate.broken:
        await git_ops.cleanup_merge_worktree(merge_wt)
        failing = ', '.join(gate.failing_subprojects)
        if gate.timed_out_subprojects:
            reason = (
                f'Post-merge verification failed: unscoped type-check timed out '
                f'for {failing}.'
            )
            new_count = timeouts.get(req.task_id, 0) + 1
            timeouts[req.task_id] = new_count
            if new_count >= max_timeouts:
                logger.warning(
                    'Task %s: post-merge unscoped type-check timed out %d times in a '
                    'row — next submission will be abandoned',
                    req.task_id, new_count,
                )
        else:
            reason = (
                f'Post-merge verification failed: unscoped type-check failed '
                f'for {failing}.'
            )
            if gate.detail:
                reason = f'{reason}\n\n{gate.detail}'
        return MergeOutcome('blocked', reason=reason)

    return None


@dataclasses.dataclass(frozen=True)
class _GenerationChainContext:
    """Bundle passed from a worker into _finalize_advanced_merge for γ2 auto-chaining.

    Carries the worker's queue, the per-branch generation counter dict, and
    the configured maximum so _finalize_advanced_merge can delegate to
    _maybe_auto_chain_generation without a direct worker reference.

    Defaulting to None in _finalize_advanced_merge preserves the function's
    behaviour for trains (_do_train_merge passes None, per PRD D9) and all
    existing callers.
    """

    queue: asyncio.Queue
    counts: dict  # dict[str, int] — per-branch chain counter
    max_auto_generations: int
    retention: TerminalOutcomeRetention | None = None
    """Retention ring to pass to enqueue_merge_request for the chained gen-(n+1)
    request so its terminal outcome is recorded (provenance: superseded_by resolves).
    The in-flight-registry SLOT HANDOFF for gen-(n+1) belongs to γ3's ATTACH_AND_CHAIN
    scope and is the second precondition guarded by AUTO_CHAIN_GENERATIONS_ENABLED."""


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
    chain_ctx: _GenerationChainContext | None = None,
    merged_branch_tip: str | None = None,
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

    *chain_ctx* + *merged_branch_tip* enable γ2 generation auto-chaining
    when :data:`AUTO_CHAIN_GENERATIONS_ENABLED` is ``True``.  While the
    kill-switch is ``False`` (the default, pending γ3), equivalence failures
    always return ``'blocked'`` and no ``'superseded'`` outcome is produced.
    When enabled and *chain_ctx* is not None, :func:`_maybe_auto_chain_generation`
    is consulted: if the branch tip advanced during verify a gen-(n+1) request
    is enqueued and a ``'superseded'`` outcome is returned instead of ``'blocked'``.
    On a clean ``'done'`` landing, chain_ctx.counts pops the branch key
    (resetting the lineage counter).  Passing ``None`` (the default)
    preserves all existing behaviour — trains pass ``None`` (PRD D9).
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
        # γ2: if chain_ctx is wired AND the kill-switch is on, try to
        # discriminate whether the branch tip advanced mid-verify and
        # auto-chain gen-(n+1).  The switch is OFF by default until γ3
        # lands the workflow 'superseded' consumer handler + slot handoff.
        if chain_ctx is not None and AUTO_CHAIN_GENERATIONS_ENABLED:
            chained = await _maybe_auto_chain_generation(
                req, advanced_sha, git_ops, event_store,
                merged_branch_tip=merged_branch_tip,
                counts=chain_ctx.counts,
                queue=chain_ctx.queue,
                max_auto_generations=chain_ctx.max_auto_generations,
                retention=chain_ctx.retention,
            )
            if chained is not None:
                _emit_merge_attempt(
                    event_store, req.task_id,
                    'post_merge_generation_chained',
                    duration_ms=_elapsed_ms(started_monotonic),
                    train_id=train_id,
                    member_task_ids=member_task_ids,
                )
                return chained
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
    # γ2: clean landing — reset the per-branch generation chain counter
    # so consecutive tip-advances count is cleared for this branch.
    if chain_ctx is not None:
        chain_ctx.counts.pop(req.branch, None)
    push_status = await git_ops.push_main()
    return MergeOutcome('done', merge_sha=advanced_sha, push_status=push_status)


async def _map_advance_failure(
    git_ops: GitOps,
    result: str,
    *,
    task_id: str,
    merge_commit_fallback: str,
    halt: Callable[[str], None],
    unhalt: Callable[[str], None],
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

    *halt* is the worker's ``halt_for_wip`` callable; *unhalt* is the
    worker's ``unhalt_wip`` callable.  *unhalt* is invoked in an
    ``except BaseException`` guard around the post-halt ``push_main`` call
    in the ``pop_conflict`` branch so that a push failure (including
    ``CancelledError``) never leaves the queue silently halted with no
    escalation owner registered (task 1671 orphan-halt window).
    *cas_retries* is mutated for the terminal result codes (popped) but
    NOT for ``wip_overlap`` (which is a recoverable halt, not a terminal
    outcome).
    """
    if result in ('wip_overlap', 'pop_conflict'):
        halt(f'advance_main: {result}')
        if result == 'pop_conflict':
            # Main was advanced — push origin even though stash pop failed.
            # Guard: if push_main raises (git failure, CancelledError) AFTER
            # halt() above, the done_wip_recovery outcome is never produced and
            # no halt-owner escalation is registered downstream — leaving the
            # queue silently halted with owner=None (task 1671 orphan-halt
            # window).  unhalt-on-raise restores the queue before re-raising.
            try:
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
            except BaseException:
                # pop_conflict push_main raised — unhalt to avoid orphan halt
                # (task 1671): the done_wip_recovery outcome was never returned
                # so no halt-owner escalation will be registered; un-halt here
                # so the queue can accept new work without requiring
                # force_unhalt_merge_queue.
                unhalt('pop_conflict push_main raised — unhalting to avoid orphan halt')
                raise
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


class TipRelation(Enum):
    """Topological relationship between two branch-tip SHAs.

    Used by the γ1 multi-waiter coalescing substrate to decide how to handle
    a new submission when a merge is already in-flight for the same branch.

    Values:
    - SAME: both tips are the same SHA.
    - SUPERSET: *new_tip* is a strict ancestor-descendant of *old_tip*
      (new has all of old's commits plus more).
    - SUBSET: *new_tip* is an ancestor of *old_tip* (new is strictly behind).
    - DIVERGENT: neither tip is an ancestor of the other; must be resolved via
      :func:`resolve_divergent` before passing to :func:`decide_attach_action`.

    Consumer wiring (pending as of γ1):
    - γ2 (task 1640): wires :func:`classify_tip_relation` /
      :func:`decide_attach_action` into the worker's post-merge-equivalence
      path to trigger generation auto-chaining; flips ``entry.verifying`` via
      :meth:`InFlightMergeRegistry.set_verifying`.
    - γ3 (task 1641): wires :func:`classify_tip_relation` /
      :func:`decide_attach_action` at the workflow merge-phase submission
      site to attach as a peer waiter.
    """

    SAME = 'same'
    SUPERSET = 'superset'
    SUBSET = 'subset'
    DIVERGENT = 'divergent'


async def classify_tip_relation(
    new_tip: str,
    old_tip: str,
    git_ops: GitOps,
) -> TipRelation:
    """Return the topological relationship between *new_tip* and *old_tip*.

    Checks performed in order (cheap first):

    1. SHA equality → SAME.
    2. ``is_ancestor(old_tip, new_tip)`` → SUPERSET (old is strictly behind new).
    3. ``is_ancestor(new_tip, old_tip)`` → SUBSET (new is strictly behind old).
    4. Neither → DIVERGENT (no ancestor relation; caller must use
       :func:`resolve_divergent` to determine the patch-id relationship).

    Consumed by γ2 (worker wiring) and γ3 (workflow caller) via the pure
    :func:`decide_attach_action` mapping.
    """
    if new_tip == old_tip:
        return TipRelation.SAME
    if await git_ops.is_ancestor(old_tip, new_tip):
        return TipRelation.SUPERSET
    if await git_ops.is_ancestor(new_tip, old_tip):
        return TipRelation.SUBSET
    return TipRelation.DIVERGENT


async def patch_content_contained(
    head: str,
    upstream: str,
    git_ops: GitOps,
) -> bool:
    """Return True if every commit in *head* is already present (by patch-id) in *upstream*.

    Uses ``git cherry upstream head``: lines without a leading ``+`` are
    commits already applied; ``+`` lines are commits NOT yet in *upstream*.
    An empty ``+``-line set → fully contained → True.

    Fail-open: returns False on any git error (``rc != 0``) so that the
    caller falls through to a full merge attempt rather than incorrectly
    declaring the branch "already merged".

    This is also the α2/D6 submit-time fast-path machinery: a branch whose
    content is fully cherry-picked/rebased into main is "already merged" even
    when its tip is not a literal ancestor.  The ``is_ancestor``-only
    fast-path was wired in task 1629; the patch-id extension (this helper)
    for escalation/server.py is not yet scheduled as a separate task.
    """
    rc, out, _ = await _run(
        ['git', 'cherry', upstream, head],
        cwd=git_ops.project_root,
    )
    if rc != 0:
        return False  # fail-open
    return not any(line.startswith('+') for line in out.splitlines())


async def resolve_divergent(
    new_tip: str,
    old_tip: str,
    git_ops: GitOps,
) -> TipRelation:
    """Resolve a DIVERGENT tip-relation using patch-id content comparison.

    Called when :func:`classify_tip_relation` returns DIVERGENT (neither tip
    is a topological ancestor of the other).

    If :func:`patch_content_contained` determines that *new_tip*'s commits are
    all already present in *old_tip* (by patch-id — covers rebase/cherry-pick
    rewrites), returns SUBSET (the new submission is content-equivalent to
    what's already in-flight, just rebased).

    Otherwise returns SUPERSET (the new submission has genuinely new content).
    """
    if await patch_content_contained(new_tip, old_tip, git_ops):
        return TipRelation.SUBSET
    return TipRelation.SUPERSET


class AttachAction(Enum):
    """Action to take when a new submission arrives for an already in-flight branch.

    Derived from the PRD §7.2 dispatch table.  Computed by the pure
    :func:`decide_attach_action` function given a resolved :class:`TipRelation`
    and the current ``verifying`` flag.

    Values:
    - COALESCE: tip identical — attach as peer, resolve with same outcome.
    - RESNAPSHOT: new tip is a superset and merge hasn't started verifying —
      update snapshot_tip and re-try the merge with the new tip.
    - ATTACH_AND_CHAIN: new tip is a superset but verify is already running —
      attach as peer and set up gen-2 chaining (γ2 worker wiring).
    - ATTACH_CONTAINMENT: new tip is a subset — attach as peer; at finalize
      time the worker resolves via containment logic (boundary test 13).

    Consumer wiring (pending as of γ1): γ2 (task 1640) and γ3 (task 1641).
    See :class:`TipRelation` for per-phase details.
    """

    COALESCE = 'coalesce'
    RESNAPSHOT = 'resnapshot'
    ATTACH_AND_CHAIN = 'attach_and_chain'
    ATTACH_CONTAINMENT = 'attach_containment'


def decide_attach_action(
    relation: TipRelation,
    *,
    verifying: bool,
) -> AttachAction:
    """Return the coalescing action for *relation* given the *verifying* flag.

    Pure function (no git I/O) — maps the PRD §7.2 table exactly:

    +------------------+----------------+---------------------+
    | relation         | verifying=False | verifying=True      |
    +==================+================+=====================+
    | SAME             | COALESCE        | COALESCE            |
    +------------------+----------------+---------------------+
    | SUPERSET         | RESNAPSHOT      | ATTACH_AND_CHAIN    |
    +------------------+----------------+---------------------+
    | SUBSET           | ATTACH_CONTAINMENT | ATTACH_CONTAINMENT |
    +------------------+----------------+---------------------+
    | DIVERGENT        | ValueError      | ValueError          |
    +------------------+----------------+---------------------+

    DIVERGENT must be resolved via :func:`resolve_divergent` before calling
    this function; it raises :class:`ValueError` otherwise to enforce the
    "patch-id compare first" contract.
    """
    if relation is TipRelation.SAME:
        return AttachAction.COALESCE
    if relation is TipRelation.SUBSET:
        return AttachAction.ATTACH_CONTAINMENT
    if relation is TipRelation.SUPERSET:
        return AttachAction.ATTACH_AND_CHAIN if verifying else AttachAction.RESNAPSHOT
    # TipRelation.DIVERGENT
    raise ValueError(
        'DIVERGENT must be resolved via resolve_divergent() first before '
        'decide_attach_action() can map it to an AttachAction.'
    )


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


async def _run_unscoped_typechecks(
    worktree: Path,
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig],
    *,
    block_on_timeout: bool,
    task_id: str | None = None,
) -> PostMergePyrightResult:
    """Run each module's ``type_check_command`` unscoped against a caller-owned worktree.

    This helper operates on a worktree supplied by the caller — it does **not**
    create or clean up any worktree itself.

    Args:
        worktree: Path to the worktree to run type-checks in.  Caller owns it.
        config: Orchestrator config (passed through to ``run_verification``).
        module_configs: List of module configs.  Modules without a
            ``type_check_command`` are silently skipped.
        block_on_timeout: When ``True``, a timed-out module is appended to
            *both* ``timed_out_subprojects`` and ``failing_subprojects``, making
            the result ``broken`` (fail-closed, for the pre-advance gate).
            When ``False``, a timed-out module is appended only to
            ``timed_out_subprojects`` (fail-open, for the post-advance check).
        task_id: Optional task identifier for log messages.

    Returns:
        :class:`PostMergePyrightResult` with the classification of each
        module's type-check outcome.
    """
    active = [mc for mc in module_configs if mc.type_check_command is not None]
    if not active:
        return PostMergePyrightResult()

    async def _run_one(mc: ModuleConfig) -> tuple[ModuleConfig, VerifyResult]:
        # Run only the type-check command verbatim (unscoped).
        # Null out test/lint so run_verification skips them (None => skip).
        # is_merge_verify=True forces cold semantics; the per-command timeout
        # used is `merge_verify_cold_command_timeout_secs` (config default 7200 s)
        # if set, falling back to `verify_cold_command_timeout_secs` then warm.
        type_only_mc = dataclasses.replace(mc, test_command=None, lint_command=None)
        return mc, await run_verification(
            worktree, config, type_only_mc,
            max_retries=0, is_merge_verify=True, role='merge',
        )

    # Run all subproject type-checks concurrently to minimise wall-clock
    # impact on the merge queue's push_main delay.
    pairs = await asyncio.gather(*(_run_one(mc) for mc in active))

    failing_subprojects: list[str] = []
    timed_out_subprojects: list[str] = []
    detail_parts: list[str] = []

    for mc, verify in pairs:
        if verify.timed_out:
            timed_out_subprojects.append(mc.prefix)
            if block_on_timeout:
                failing_subprojects.append(mc.prefix)
            continue

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
        timed_out_subprojects=timed_out_subprojects,
        detail=detail,
    )


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

    Delegates the per-module classification loop to
    :func:`_run_unscoped_typechecks` with ``block_on_timeout=False`` —
    the merge has already landed; we never block a landed merge on a flaky
    hang or transient infra error (fail-open on timeout).

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

        result = await _run_unscoped_typechecks(
            merge_wt, config, module_configs,
            block_on_timeout=False, task_id=task_id,
        )
        # Log timeouts explicitly (fail-open path: not in failing_subprojects).
        for prefix in result.timed_out_subprojects:
            logger.warning(
                'post-merge-pyright: type-check for %r timed out on %s; '
                'failing open. task_id=%s',
                prefix, advanced_sha[:12], task_id or '<unknown>',
            )
        return result

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

.. note:: Raising this value (or raising :data:`_MERGE_AHEAD_BOUND`, or
   increasing the merge-verify cold timeout) affects the safety margin computed
   by :func:`check_merge_liveness_margin`.  Run that guard after changing any
   of these three values.

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

    Backed by a ``collections.deque(maxlen=maxlen)`` for eviction and a
    ``dict`` index keyed by ``request_id`` for O(1) lookups.  When the ring
    is full, the oldest entry is evicted from both structures atomically.

    The event store is the durable tier so eviction is lossless — α3's
    merge_status falls through to event-store queries for evicted ids.
    """

    def __init__(self, maxlen: int = 200) -> None:
        self._ring: collections.deque[TerminalOutcomeRecord] = collections.deque(maxlen=maxlen)
        self._index: dict[str, TerminalOutcomeRecord] = {}

    def record(self, rec: TerminalOutcomeRecord) -> None:
        """Append *rec* to the ring, evicting the oldest entry from the index if full."""
        if len(self._ring) == self._ring.maxlen:
            # Capture the about-to-be-evicted entry before appending.
            evicted = self._ring[0]
            self._ring.append(rec)
            # Only remove the index entry if it still points to *evicted* — a
            # duplicate request_id (pathological case) should not evict the
            # newer entry.
            if self._index.get(evicted.request_id) is evicted:
                del self._index[evicted.request_id]
        else:
            self._ring.append(rec)
        self._index[rec.request_id] = rec

    def get(self, request_id: str) -> TerminalOutcomeRecord | None:
        """Return the record for *request_id*, or None if evicted / not yet recorded."""
        return self._index.get(request_id)


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
        ``_release(branch)`` fires automatically on every terminal path
        (result set, exception set, or cancellation).

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

    def release(self, branch: str) -> None:
        """Remove *branch* from the in-flight registry.

        Public surface for callers that need to release a slot on an
        exceptional path (e.g. enqueue failure before the worker can ever
        resolve the future).  The ``done_callback`` registered inside
        :meth:`acquire` is the normal release path; this method is the
        explicit fallback used by the slot-leak guards in
        :func:`register_and_enqueue_merge_request` and
        :func:`coalesce_or_enqueue_merge_request`.
        """
        self._slots.pop(branch, None)

    # Keep the private alias so existing done_callbacks installed by acquire()
    # continue to fire correctly without any change to those lambda closures.
    _release = release


def _emit_merge_queued(
    event_store: EventStore | None,
    req: MergeRequest,
    reason: str | None = None,
    *,
    queue_depth: int | None = None,
    position: int | None = None,
) -> None:
    """Emit a merge_queued event.  No-op when *event_store* is None.

    Centralises the emit payload so both :func:`enqueue_merge_request` and
    the ``MergeWorker`` CAS-retry path use an identical record shape.  If
    *reason* is provided (e.g. ``'cas_retry'``) it is stored in ``data``.

    *queue_depth* (when provided) records how deep the main queue was at the
    moment of enqueue — O(1) qsize() from the call site.  *position* (when
    provided) records the front-of-line position for urgent re-inserts (0 ==
    head).  Each key is omitted when None so the shape remains backward-
    compatible with existing consumers.
    """
    if event_store is None:
        return
    data: dict = {'branch': req.branch}
    if reason is not None:
        data['reason'] = reason
    if queue_depth is not None:
        data['queue_depth'] = queue_depth
    if position is not None:
        data['position'] = position
    event_store.emit(
        EventType.merge_queued,
        task_id=req.task_id,
        phase='merge',
        data=data,
    )


async def _maybe_auto_chain_generation(
    req: MergeRequest,
    advanced_sha: str,
    git_ops: GitOps,
    event_store: EventStore | None,
    *,
    merged_branch_tip: str | None,
    counts: dict[str, int],
    queue: asyncio.Queue,
    max_auto_generations: int,
    retention: TerminalOutcomeRetention | None = None,
) -> MergeOutcome | None:
    """Check whether a post-merge equivalence failure was caused by a tip advance,
    and if so, enqueue a gen-(n+1) MergeRequest for the delta (γ2).

    Returns:
    - ``MergeOutcome('superseded', ...)`` if the tip advanced and the chain is
      within the bound — a gen-(n+1) request has been placed on *queue*.
    - ``MergeOutcome('blocked', ...)`` if the chain-generation bound is exceeded
      (consecutive tip advances > *max_auto_generations*) — the branch counter
      is reset.  Uses POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX so the
      existing workflow blocked-handler routes to _mark_blocked(escalate_to_human=True).
    - ``None`` if the tip has NOT advanced (genuine drop; caller should block
      as before).

    The *counts* dict is mutated in-place: incremented on a chain, reset on
    bound-exceeded.  A clean 'done' landing pops the branch key (handled by
    the caller: _finalize_advanced_merge).

    NOTE: this function is only reachable when :data:`AUTO_CHAIN_GENERATIONS_ENABLED`
    is ``True`` (the kill-switch in _finalize_advanced_merge gates the call site).
    The in-flight-registry SLOT HANDOFF for gen-(n+1) (re-acquiring the branch
    slot without tripping the gen-1 done-callback double-release, plus threading
    InFlightMergeRegistry and TerminalOutcomeRetention from the harness into the
    workers — harness.py:3238 omits both) belongs to γ3's ATTACH_AND_CHAIN scope
    and is the second precondition guarded by the kill-switch.
    """
    if not merged_branch_tip:
        return None

    # Rev-parse the branch's current HEAD in its worktree.
    rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=req.worktree)
    if rc != 0:
        # Fail-open: treat as genuine drop so the caller blocks as today.
        return None
    current_head = head_out.strip()

    if current_head == merged_branch_tip:
        # Tip has not moved — genuine drop.
        return None

    # Classify the topological relationship.
    rel = await classify_tip_relation(current_head, merged_branch_tip, git_ops)
    if rel is TipRelation.DIVERGENT:
        rel = await resolve_divergent(current_head, merged_branch_tip, git_ops)

    if rel is not TipRelation.SUPERSET:
        # SUBSET (patch-contained) or SAME: no new content — genuine drop.
        return None

    # Tip advanced.  Enforce the per-branch generation bound.
    new_count = counts.get(req.branch, 0) + 1
    if new_count > max_auto_generations:
        counts.pop(req.branch, None)
        return MergeOutcome(
            'blocked',
            reason=(
                f'{POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX}: '
                f'branch tip advanced past the merged snapshot for the '
                f'{max_auto_generations}-generation auto-chain bound '
                f'(consecutive tip advances). Manual review required.'
            ),
        )

    # Within bound — build the gen-(n+1) request and enqueue it.
    gen_next = dataclasses.replace(
        req,
        result=asyncio.get_running_loop().create_future(),
        request_id=f'mr-{uuid.uuid4().hex[:8]}',
        generation=req.generation + 1,
        snapshot_tip=current_head,
        pre_rebased=False,
    )
    await enqueue_merge_request(queue, gen_next, event_store, retention=retention)
    counts[req.branch] = new_count
    return MergeOutcome('superseded', superseded_by=gen_next.request_id, merge_sha=advanced_sha)


async def enqueue_merge_request(
    queue: asyncio.Queue,
    req: MergeRequest,
    event_store: EventStore | None,
    *,
    retention: TerminalOutcomeRetention | None = None,
) -> None:
    """Enqueue a MergeRequest and emit a merge_queued event.

    Puts the request on *queue* first so that a cancellation between put and
    emit (or any emit error) does not leave a dangling ``merge_queued`` row
    with no corresponding worker pickup.  Losing the event is less confusing
    than a stale "queued" row that persists until the TTL expires.

    If ``event_store`` is None the request is still enqueued; emission is
    silently skipped (mirrors the None-safe pattern used by
    ``_emit_merge_attempt``).

    Registers a single ``req.result.add_done_callback`` that, when the future
    reaches its terminal state (resolved, cancelled, or exception), emits a
    ``merge_finalized`` event and records the outcome into *retention* (when
    provided).  The callback is fire-and-forget: any exception is logged as a
    warning and never propagates.
    """
    def _on_finalized(fut: asyncio.Future) -> None:  # noqa: ANN001
        # --- derive terminal state -------------------------------------------
        superseded_by: str | None = None
        try:
            if fut.cancelled():
                state: str = 'abandoned'
                merge_sha: str | None = None
            elif fut.exception() is not None:
                state = 'error'
                merge_sha = None
            else:
                outcome: MergeOutcome = fut.result()
                state = outcome.status
                merge_sha = outcome.merge_sha
                superseded_by = outcome.superseded_by
        except Exception:  # noqa: BLE001
            logger.warning(
                'enqueue_merge_request: _on_finalized could not derive terminal '
                'state for request_id=%s task_id=%s',
                req.request_id, req.task_id, exc_info=True,
            )
            return
        # --- in-memory hot tier (recorded before durable tier so a DB failure
        #     does not also degrade the O(1) lookup ring) --------------------
        if retention is not None:
            try:
                retention.record(TerminalOutcomeRecord(
                    request_id=req.request_id,
                    task_id=req.task_id,
                    branch=req.branch,
                    state=state,
                    snapshot_tip=req.snapshot_tip,
                    merge_sha=merge_sha,
                    superseded_by=superseded_by,
                    generation=req.generation,
                ))
            except Exception:  # noqa: BLE001
                logger.warning(
                    'enqueue_merge_request: _on_finalized retention.record failed '
                    'for request_id=%s task_id=%s',
                    req.request_id, req.task_id, exc_info=True,
                )
        # --- durable tier ----------------------------------------------------
        if event_store is not None:
            try:
                event_store.emit(
                    EventType.merge_finalized,
                    task_id=req.task_id,
                    phase='merge',
                    data={
                        'request_id': req.request_id,
                        'branch': req.branch,
                        'state': state,
                        'snapshot_tip': req.snapshot_tip,
                        'merge_sha': merge_sha,
                        'superseded_by': superseded_by,
                        'generation': req.generation,
                    },
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    'enqueue_merge_request: _on_finalized event_store.emit failed '
                    'for request_id=%s task_id=%s',
                    req.request_id, req.task_id, exc_info=True,
                )

    req.result.add_done_callback(_on_finalized)
    await queue.put(req)
    _emit_merge_queued(event_store, req, queue_depth=queue.qsize())


async def register_and_enqueue_merge_request(
    queue: asyncio.Queue,
    req: MergeRequest,
    event_store: EventStore | None,
    registry: InFlightMergeRegistry | None,
    *,
    retention: TerminalOutcomeRetention | None = None,
) -> bool:
    """Workflow-path enqueue that registers the branch in the in-flight registry.

    Unlike the MCP path (``coalesce_or_enqueue_merge_request``), the workflow
    MUST always enqueue — it blocks on ``req.result`` for its outcome, so
    skipping the enqueue when the slot is already held would leave the future
    unresolved and deadlock the caller.

    When *registry* is not None and the slot is free, ``registry.acquire`` is
    called with ``req.result`` before enqueuing.  The existing
    ``Future.add_done_callback`` registered inside ``acquire`` releases the
    slot automatically on result, exception, or cancellation.  If the enqueue
    itself raises (e.g. queue closed or cancellation), the slot is released
    explicitly to avoid a leak — mirroring the guard in
    ``coalesce_or_enqueue_merge_request`` (merge_queue.py:1794-1803).

    When *retention* is provided it is forwarded to the single
    :func:`enqueue_merge_request` chokepoint so the dominant workflow path
    populates the in-memory ring alongside the MCP path (see
    :func:`coalesce_or_enqueue_merge_request`).  Existing call sites in
    workflow.py pass only positional args so this keyword-only param is
    backwards-compatible.

    Returns True if the registry slot was newly acquired (caller's branch was
    free); False if the slot was already held by another task or *registry* is
    None.  The return value is informational — the request is always enqueued.
    """
    acquired = (
        registry.acquire(req.branch, req.task_id, req.result, request_id=req.request_id)
        if registry is not None
        else False
    )
    try:
        await enqueue_merge_request(queue, req, event_store, retention=retention)
    except BaseException:
        # Slot-leak guard: if the enqueue raises before the worker can ever
        # resolve req.result, the done_callback will never fire.  Release the
        # slot explicitly so a future merge for this branch can proceed.
        if acquired:
            registry.release(req.branch)  # type: ignore[union-attr]
        raise
    return acquired


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
    retention: TerminalOutcomeRetention | None = None,
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

    When *retention* is provided it is forwarded to the single
    :func:`enqueue_merge_request` chokepoint so the MCP path populates
    the ring alongside the workflow path (see
    :func:`register_and_enqueue_merge_request`).  Coalesced requests
    are NOT recorded — their terminal outcome is owned by the in-flight
    entry's callback.
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
            inflight_request_id=entry.request_id if entry else None,
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
    if registry.acquire(branch, req.task_id, req.result, request_id=req.request_id):
        try:
            await enqueue_merge_request(queue, req, event_store, retention=retention)
        except BaseException:
            # Slot leak guard: if the enqueue raises (e.g. queue closed,
            # cancellation) before the worker can ever resolve req.result,
            # the done_callback will never fire.  Release the slot explicitly
            # so a future merge_request for this branch can proceed.
            registry.release(branch)
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
        inflight_request_id=entry.request_id if entry else None,
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
    enqueued_at: float = field(default_factory=time.time, kw_only=True)
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

    status: Literal['done', 'conflict', 'blocked', 'already_merged', 'wip_halted', 'done_wip_recovery', 'wip_recovery_no_advance', 'unmerged_state', 'unknown_branch', 'superseded']
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
    superseded_by: str | None = None
    """request_id of the gen-(n+1) request that supersedes this one (γ2)."""


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
    already_delivered: bool = False  # True → merger resolved req.result OOB; verifier skips set_result but still runs n_failed/slot bookkeeping
    started_monotonic: float | None = None  # time.monotonic() at entry; None → unset, _elapsed_ms returns None
    failure_diagnostic: dict[str, str] | None = None  # Populated on non-conflict merge failure
    merged_branch_tip: str | None = None  # γ2: branch HEAD rev-parsed by the merger; passed to _finalize_advanced_merge
    counts_against_cap: bool = False  # True for non-speculative, non-train successful merges (Mechanism 1)


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
    def unhalt_wip(self, reason: str | None = None) -> None: ...
    def _abandon_outcome(self, task_id: str, count: int) -> MergeOutcome: ...


async def _do_train_merge(
    worker: _TrainMergeHost,
    req: GroupMergeRequest,
) -> MergeOutcome:
    """Atomic train-merge pipeline shared by MergeWorker and SpeculativeMergeWorker.

    BEHAVIOUR-ADDING (task 1596): trains now inherit the full shared post-merge
    core at PARITY with MergeWorker._do_merge — specifically:
      • disk-guard pre-verify short-circuit (_run_post_merge_verify)
      • verify-timeout loop-breaker (worker._post_merge_verify_timeouts)
      • post-merge content-equivalence gate (_finalize_advanced_merge)
      • unscoped pyright gate (_finalize_advanced_merge)
      • push_main (outcome.push_status propagated from _finalize_advanced_merge)
      • wip-halt + escalation routing for non-'advanced' advance results
        (_map_advance_failure via worker.halt_for_wip)

    DEFENSIBLE DELTAS — behaviours intentionally absent from the train path
    despite being present in MergeWorker or SpeculativeMergeWorker:

    1. No ``reverify_on_rebase``: the 1595 disjoint-delta gate lives ONLY in
       SpeculativeMergeWorker._verify_and_advance's CAS loop.  MergeWorker (the
       readable serial reference) also does NOT pass it.  "Parity" means parity
       with MergeWorker, not the spec-worker CAS loop.  Adding the gate here
       would duplicate speculative-worker logic that has no business in the train.

    2. Pyright redundant-but-harmless: the train always rebases its tip onto
       main before merging, so the type surface is fresh.  The post-merge
       unscoped pyright run is therefore redundant in the common case but is
       kept for uniformity and because the cost is low relative to the safety
       guarantee.

    3. No drop-guard (_check_plan_targets_in_tree): the drop guard targets
       conflict-RESOLUTION drops (a human resolved and accidentally removed work).
       Trains abort cleanly on rebase conflict (TRAIN_REBASE_CONFLICT), so there
       is no conflict-resolution step from which targets can be dropped.  The
       post-merge equivalence check covers rebase-rewrite.

    4. No ``skip_verify``: the train always rebases its tip and needs the full
       workspace-green gate.  The single-task ``skip_verify`` path is triggered
       by a pre-rebased branch; trains always start from a fresh rebase.

    5. ``cas_failed`` → ``blocked``: ``advance_main`` already retried internally
       up to ``max_advance_attempts``; the workflow re-parks an incomplete train
       for retry.  Mapping a residual ``cas_failed`` to ``blocked`` (not
       re-enqueuing directly) is correct — the orchestrator will re-dispatch on
       the next scheduler tick.

    6. ``wip_halted`` / ``done_wip_recovery`` / ``wip_recovery_no_advance`` /
       ``unmerged_state`` → halt-owning L1 escalation for trains (task 1599):
       ``_map_advance_failure`` can return any of these four statuses.  For a
       *single* task they trigger automatic in-place recovery in ``workflow.py``
       (``_handle_wip_conflict`` / ``_handle_wip_recovery`` / etc.) — which
       awaits escalation resolution before retrying.  For a train, the
       GroupMergeRequest consumer gates on the orphan-halt probe:

           merge_worker.is_wip_halted AND halt_owner_esc_id is None

       When the probe fires, ``_escalate_train_halt`` builds a per-status L1
       (category='wip_conflict' or 'unmerged_state'), calls
       ``_submit_halt_owning_escalation`` (submit → set_halt_owner), and
       returns ``_mark_blocked(skip_escalation=True)`` — the tip stays BLOCKED
       and re-dispatches once the L1 is resolved and the queue is unhalted.

       This closes the asymmetry from the original implementation:
       • Resolving the L1 now auto-unhalts the queue via
         ``harness._on_escalation_resolved → unhalt_wip()`` (because
         ``set_halt_owner`` was called).
       • ``harness._rehydrate_merge_halt`` re-owns the L1 (category in
         {wip_conflict, unmerged_state}) across restarts — restart-survival
         parity with single tasks, with no harness change.

       Two intentional remaining differences from the single-task path:
       (i)  The train returns BLOCKED + re-dispatches rather than inline
            wait-and-retry.  Inline-waiting reintroduces the cancellation-orphan
            surface task 1448 had to harden; trains are multi-member so the
            coroutine must not block waiting for human resolution.
       (ii) ``done_wip_recovery`` escalates rather than returning DONE — the
            merge landed on main but members were NOT flipped, a split-brain
            that cannot safely map to the single-task DONE return.

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
        ONLY after advance + _finalize_advanced_merge succeed (invariant: members
        flip iff main lands AND post-merge gates pass).
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
        _emit_merge_attempt(event_store, req.task_id, 'advance_failed', duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
        if adv == 'cas_failed':
            # advance_main already retried internally; workflow re-parks the
            # train.  Return a simple blocked rather than routing through
            # _map_advance_failure (which does not handle cas_failed).
            _emit_train_event(
                event_store, EventType.train_derailed,
                task_id=req.task_id, train_id=req.train_id,
                member_task_ids=req.member_task_ids,
                data={'derail_reason': f'Train merge advance failed: {adv}'},
            )
            return MergeOutcome('blocked', reason=f'Train merge advance failed: {adv}')
        # All other codes (wip_overlap, pop_conflict, unmerged_state, …) route
        # through the shared failure mapper so the train gets the same
        # wip-halt + escalation routing as the single-task path.
        #
        # NOTE — behaviour-asymmetry (documented in docstring delta #6):
        # _map_advance_failure can return wip_halted (wip_overlap) or
        # done_wip_recovery (pop_conflict).  For single tasks workflow.py
        # handles these with _handle_wip_conflict / _handle_wip_recovery.
        # For trains the GroupMergeRequest consumer falls through to
        # _mark_blocked(escalate_to_human=True) in both cases — intentional
        # safe-by-escalation behaviour; auto-recovery parity is a follow-up
        # (requires changes to workflow.py, outside task 1596 scope).
        outcome = await _map_advance_failure(
            git_ops, adv,
            task_id=req.task_id,
            merge_commit_fallback=merge_commit,
            halt=worker.halt_for_wip,
            unhalt=worker.unhalt_wip,
            cas_retries=worker._cas_retries,
        )
        _emit_train_event(
            event_store, EventType.train_derailed,
            task_id=req.task_id, train_id=req.train_id,
            member_task_ids=req.member_task_ids,
            data={'derail_reason': f'Train merge advance failed: {adv}'},
        )
        return outcome

    # advance succeeded — run equivalence + pyright + push via shared finalize.
    # merge_wt has already been cleaned up above (caller contract for finalize).
    # t0 is passed as started_monotonic so finalize's duration_ms is accurate
    # for the whole train attempt, not just the post-advance window.
    #
    # PRD D9: trains are bit-identical, multi-waiter merges — γ2 auto-chaining
    # applies ONLY to single-branch MergeRequest paths (MergeWorker /
    # SpeculativeMergeWorker).  chain_ctx=None is passed explicitly here so the
    # invariant is visible at the call site and not left implicit.
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
        chain_ctx=None,  # PRD D9: trains never auto-chain
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
                    data={'branch': req.branch, 'queue_depth': self._queue.qsize()},
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
                chain_ctx=_GenerationChainContext(
                    queue=self._queue,
                    counts=self._generation_chain_counts,
                    max_auto_generations=MAX_AUTO_CHAINED_GENERATIONS,
                ),
                merged_branch_tip=branch_head.strip(),
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
    MAX_POST_MERGE_VERIFY_TIMEOUTS: int = 2
    # Mirror of MergeWorker.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES.
    MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES: int = 1

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
        # γ2 per-branch generation auto-chain counter (mirrors MergeWorker).
        # Incremented on each consecutive tip-advance equivalence failure;
        # popped on a clean 'done' landing or bound-exceeded escalation.
        self._generation_chain_counts: dict[str, int] = {}
        # Depth-1 cap: cleared when a speculative merge is in flight,
        # set by the Verifier when it finishes the item before the speculation.
        self._speculation_slot = asyncio.Event()
        self._speculation_slot.set()  # initially free
        # Merger-ahead cap (Mechanism 1, task 1646): limits non-speculative
        # build-ahead to _MERGE_AHEAD_BOUND items in the verifier queue.
        # Plain Semaphore (not BoundedSemaphore) so stop() may over-release
        # without raising.  Released ON-DRAIN (right after verifier_queue.get()
        # for a counted item) so the slot is free while verify runs.
        self._merge_ahead_cap = asyncio.Semaphore(_MERGE_AHEAD_BOUND)
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
        self._heartbeat_task: asyncio.Task | None = None
        # In-flight request being processed by the merger loop. Set after
        # dequeue, cleared after the SpeculativeItem is pushed to the verifier
        # queue. Used by stop() to resolve Futures for requests that were
        # mid-processing when shutdown was initiated.
        self._inflight_req: MergeRequest | None = None
        # Verifier sub-state: current item and its phase within _verify_and_advance.
        # Set by _verifier_loop (early, before _remerge, to cover the remerge blind
        # spot); cleared in the loop's finally.
        self._verify_item: SpeculativeItem | None = None
        self._verify_phase: str | None = None
        # Timestamp (wall clock) when _verify_phase first entered 'verifying'.
        # Separate from enqueued_at so verify_in_progress can report pure verify
        # time rather than total queue-wait time (useful for triage of stuck verifies).
        self._verify_started_at: float | None = None
        # Can be overridden in tests for fast shutdown (see stop()).
        self._shutdown_timeout: float = 5.0
        # Heartbeat: wall-clock time of last emission; initialised to 0.0 so the
        # first emission fires within one poll period after startup when depth > 0
        # (the very-large now - 0.0 gap always exceeds _heartbeat_interval_s).
        self._last_heartbeat_at: float = 0.0
        # Default interval ~5 min; override in tests for deterministic rate-limit checks.
        # Mirrors the _shutdown_timeout override precedent.
        self._heartbeat_interval_s: float = 300.0

    def snapshot(self) -> dict:
        """Return a synchronous read-only snapshot of the merge worker pipeline state.

        Safe to call from any context (no await, no lock) because asyncio's
        single-loop model ensures in-memory reads are non-interleaved.

        Returns a dict with:
          entries: list of entry dicts, head-of-line first.
          depth: total number of entries.
          head_of_line: task_id of the first entry, or None.
          verify_in_progress: {task_id, age_secs} when verifier is active, else None.
          is_wip_halted: bool.
          halt_owner_esc_id: str or None.

        Each entry dict contains:
          task_id, branch, state, enqueued_at, age_secs, position,
          waiter_alive, worktree, pre_rebased, request_id.
        State values: queued, merging, awaiting_verify, verifying,
          gate_reverify, finalizing.
        """
        entries: list[dict] = []
        now = time.time()

        def _entry(req: MergeRequest, state: str, worktree_path=None, position: int = 0) -> dict:
            return {
                'task_id': req.task_id,
                'branch': req.branch,
                'state': state,
                'enqueued_at': req.enqueued_at,
                'age_secs': max(0.0, now - req.enqueued_at),
                'position': position,
                'waiter_alive': not req.result.cancelled(),
                'worktree': str(worktree_path) if worktree_path is not None else None,
                'pre_rebased': req.pre_rebased,
                'request_id': req.request_id,
            }

        # 1. Verifier-current item (head-of-line)
        if self._verify_item is not None:
            item = self._verify_item
            state = self._verify_phase or 'verifying'
            entries.append(_entry(
                item.request, state,
                worktree_path=item.merge_wt,
                position=len(entries),
            ))

        # 2. Awaiting-verify items from the verifier queue (skip None sentinel)
        # Accessing asyncio.Queue._queue (the internal deque) directly — a CPython
        # implementation detail.  Safe here: snapshot() is synchronous, runs under
        # the single asyncio event loop, and never mutates the deque; the only
        # alternative would be maintaining a separate side-list at every put/get
        # call site.  A # type: ignore[attr-defined] suppresses the attr check.
        for item in list(self._verifier_queue._queue):  # type: ignore[attr-defined]
            if item is None:
                continue
            entries.append(_entry(
                item.request, 'awaiting_verify',
                worktree_path=item.merge_wt,
                position=len(entries),
            ))

        # 3. Merging (in-flight with the merger)
        if self._inflight_req is not None:
            entries.append(_entry(
                self._inflight_req, 'merging',
                worktree_path=None,
                position=len(entries),
            ))

        # 4. Queued (waiting for the merger; the incident blind spot)
        # Same CPython internal-deque access as above — read-only, no lock needed.
        for req in list(self._queue._queue):  # type: ignore[attr-defined]
            if req is None:
                continue
            entries.append(_entry(req, 'queued', worktree_path=None, position=len(entries)))

        verify_in_progress = None
        if self._verify_item is not None:
            vi = self._verify_item
            verify_age: float | None = (
                max(0.0, now - self._verify_started_at)
                if self._verify_started_at is not None else None
            )
            verify_in_progress = {
                'task_id': vi.request.task_id,
                # age_secs: total time since this request was enqueued (queue wait
                # + verify time).  Use verify_age_secs for pure verification time.
                'age_secs': max(0.0, now - vi.request.enqueued_at),
                # verify_age_secs: time elapsed since 'verifying' phase started
                # (None when still remerging / before first verify call).
                'verify_age_secs': verify_age,
            }

        return {
            'entries': entries,
            'depth': len(entries),
            'head_of_line': entries[0]['task_id'] if entries else None,
            'verify_in_progress': verify_in_progress,
            'is_wip_halted': self.is_wip_halted,
            'halt_owner_esc_id': self.halt_owner_esc_id,
        }

    def _maybe_log_queue_heartbeat(self, now: float) -> bool:
        """Emit a queue-depth heartbeat log line and event if conditions are met.

        Synchronous and clock-injectable (``now`` parameter) so tests can drive
        firing/rate-limiting deterministically without relying on real sleep.

        Returns True when a heartbeat was emitted, False otherwise.

        No-ops when:
          - ``snapshot()['depth'] == 0`` (idle pipeline — no journal spam)
          - ``now - self._last_heartbeat_at < self._heartbeat_interval_s``
            (rate-limit — respects the overridable interval)
        """
        snap = self.snapshot()
        if snap['depth'] == 0:
            return False
        if now - self._last_heartbeat_at < self._heartbeat_interval_s:
            return False

        entries = snap['entries']
        oldest_age = max(e['age_secs'] for e in entries)
        head = entries[0]
        head_of_line = {
            'task_id': head['task_id'],
            'state': head['state'],
            'age_secs': head['age_secs'],
        }

        logger.info(
            'merge queue heartbeat: %d in pipeline, oldest age=%.0fs, '
            'head=task %s (state=%s, age=%.0fs)',
            snap['depth'], oldest_age,
            head['task_id'], head['state'], head['age_secs'],
        )

        if self._event_store is not None:
            self._event_store.emit(
                EventType.merge_heartbeat,
                task_id=None,
                phase='merge',
                data={
                    'depth': snap['depth'],
                    'oldest_age_secs': oldest_age,
                    'head_of_line': head_of_line,
                    'verify_in_progress': snap['verify_in_progress'],
                },
            )

        self._last_heartbeat_at = now
        return True

    async def _heartbeat_loop(self) -> None:
        """Periodically emit merge-queue depth heartbeats while the worker runs.

        Runs independently of the merger and verifier loops so it continues to
        fire even when those are blocked on ``queue.get()`` or semaphores (the
        exact silence window that caused the 2026-06-04 dead-slot misdiagnosis).

        Wakes every ``_HEARTBEAT_POLL_S`` seconds and delegates the
        fire/rate-limit/format/emit decision to the synchronous, clock-injectable
        :meth:`_maybe_log_queue_heartbeat`.  Any unexpected exception is logged
        and swallowed so a heartbeat bug can never crash the worker.
        """
        while self._running:
            await asyncio.sleep(_HEARTBEAT_POLL_S)
            try:
                self._maybe_log_queue_heartbeat(time.time())
            except Exception:
                logger.exception('merge queue heartbeat: unexpected error')

    async def run(self) -> None:
        """Start merger, verifier, and heartbeat coroutines; wait for merge tasks."""
        self._merger_task = asyncio.create_task(self._merger_loop())
        self._verifier_task = asyncio.create_task(self._verifier_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        try:
            await asyncio.gather(self._merger_task, self._verifier_task)
        except BaseException:
            for t in (self._merger_task, self._verifier_task, self._heartbeat_task):
                if t and not t.done():
                    t.cancel()
            await asyncio.gather(
                self._merger_task, self._verifier_task, self._heartbeat_task,
                return_exceptions=True,
            )
            raise
        finally:
            # Cancel the heartbeat on both normal and exceptional exit so its
            # lifetime is self-contained regardless of why the merge loops exit.
            # On the exception path the except block already cleaned it up, so
            # _heartbeat_task.done() is True and this is a no-op.
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                await asyncio.gather(self._heartbeat_task, return_exceptions=True)

    async def stop(self) -> None:
        """Graceful shutdown: drain queues and resolve all pending Futures."""
        self._running = False
        shutdown = MergeOutcome('blocked', reason='Merge worker shutting down')
        # Release speculation slot, WIP halt, and merge-ahead cap so the merger
        # doesn't hang waiting at any of the three synchronisation points.
        # Over-releasing a plain Semaphore is safe (it just increments the counter).
        self._speculation_slot.set()
        self._wip_halt.set()
        for _ in range(_MERGE_AHEAD_BOUND + 1):
            self._merge_ahead_cap.release()

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

        # Cancel the heartbeat task — it loops independently (no sentinel path).
        # _running is already False so the loop will not re-enter after the
        # cancellation; we await it to ensure the task is done before stop() returns.
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

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
    # Out-of-band delivery helper
    # ------------------------------------------------------------------

    def _oob_deliver(
        self,
        req: MergeRequest,
        outcome: MergeOutcome,
        *,
        speculative: bool,
    ) -> bool:
        """Resolve req.result out-of-band at detection time when safe.

        PREDICATE: not GroupMergeRequest AND not speculative AND
        outcome.status not in {'done','already_merged'}.

        Speculative items can be discarded and re-merged by the verifier
        (which may flip 'conflict' → 'done'), so they must never be
        early-delivered.  Trains are excluded by the isinstance guard.
        'done'/'already_merged' are excluded because they are either never
        produced here or are already handled at the door.

        Returns True when the OOB precondition holds (the verifier must skip
        set_result), regardless of whether set_result was actually invoked
        here — the result may already be resolved by a peer/door.  The caller
        passes the return value as already_delivered on the SpeculativeItem
        ordering token so the verifier skips set_result but still runs
        n_failed/slot bookkeeping.
        """
        if (
            isinstance(req, GroupMergeRequest)
            or speculative
            or outcome.status in ('done', 'already_merged')
        ):
            return False
        if not req.result.done():
            req.result.set_result(outcome)
        return True

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
                        data={'branch': req.branch, 'queue_depth': self._queue.qsize()},
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
                        _abandon = self._abandon_outcome(req.task_id, prior_timeouts)
                        _already = self._oob_deliver(req, _abandon, speculative=speculative)
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=actual_main, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=_abandon,
                            already_delivered=_already,
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
                        _already = self._oob_deliver(req, guard, speculative=speculative)
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=actual_main, speculative=speculative,
                            skip_verify=False, immediate_outcome=guard,
                            already_delivered=_already,
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
                        _revparse_fail = MergeOutcome(
                            'blocked',
                            reason=f'rev-parse HEAD failed: {err.strip()}',
                        )
                        _already = self._oob_deliver(req, _revparse_fail, speculative=speculative)
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=actual_main, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=_revparse_fail,
                            already_delivered=_already,
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
                        _conflict = MergeOutcome(
                            'conflict', conflict_details=merge_result.details,
                        )
                        _already = self._oob_deliver(req, _conflict, speculative=speculative)
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=base_for_merge, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=_conflict,
                            already_delivered=_already,
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
                        _merge_fail = MergeOutcome(
                            'blocked',
                            reason=f'{merge_result.details}\n{_rendered}',
                            failure_diagnostic=_diag,
                        )
                        _already = self._oob_deliver(req, _merge_fail, speculative=speculative)
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=base_for_merge, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=_merge_fail,
                            already_delivered=_already,
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
                        _drop = MergeOutcome('blocked', reason=reason)
                        _already = self._oob_deliver(req, _drop, speculative=speculative)
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=base_for_merge, speculative=speculative,
                            skip_verify=False,
                            immediate_outcome=_drop,
                            already_delivered=_already,
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
                    # Mechanism 1: cap non-speculative build-ahead.
                    # Trains (continue before this) and immediate-outcome guards
                    # (all return above) never reach this site, so `not speculative`
                    # is the exact predicate for blocking-path items.
                    counts_against_cap = not speculative
                    if counts_against_cap:
                        await self._merge_ahead_cap.acquire()
                    try:
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=merge_result,
                            merge_wt=merge_result.merge_worktree,
                            base_sha=base_for_merge, speculative=speculative,
                            skip_verify=skip_verify,
                            started_monotonic=t0,
                            merged_branch_tip=branch_head,  # γ2: branch tip at merge time
                            counts_against_cap=counts_against_cap,
                        ))
                    except BaseException:
                        # put() failed — the verifier will never drain this item
                        # and release the cap.  Release it here to prevent the
                        # merger from deadlocking at the next acquire.
                        #
                        # Double-release edge case: if CancelledError is raised
                        # at the `await` boundary AFTER put() already enqueued
                        # the item (a narrow asyncio race), this release fires
                        # AND the verifier releases again on drain — two releases
                        # for one acquire.  This is intentionally tolerated:
                        # asyncio.Semaphore allows over-release (its counter
                        # simply increments past the original bound).
                        # Do NOT replace with asyncio.BoundedSemaphore, which
                        # raises ValueError on over-release and would crash here.
                        if counts_against_cap:
                            self._merge_ahead_cap.release()
                        raise
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

            # Mechanism 1: release the merger-ahead cap ON-DRAIN — immediately
            # after get(), before any branching or item reassignment, so every
            # drain path (normal verify, immediate_outcome, discard/_remerge,
            # abandoned early-continue) is covered uniformly and the flag is
            # captured before _remerge could reassign `item`.
            if item.counts_against_cap:
                self._merge_ahead_cap.release()

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
                # ── Unified re-merge site (Mechanism 2 + existing chain-invalidation) ─
                #
                # remerge_reason is set when the item must be re-merged:
                #   'previous_failed'    — N failed verify (n_failed=True)
                #   'chain_invalidated'  — a prior iteration re-merged; this item's
                #                          spec_base descends from a stale commit
                #   'main_advanced'      — real (non-speculative, non-train) item whose
                #                          base_sha != current main (Mechanism 2,
                #                          task 1646: freshness re-base at verify-pickup)
                #
                # The elif ordering guarantees:
                #   • chain-invalidation is evaluated first so a speculative item that
                #     is ALSO stale-base is not double-re-merged;
                #   • 'main_advanced' only fires for real items (immediate_outcome is
                #     None, merge_result is not None, not a train) that are not already
                #     covered by chain-invalidation above.
                remerge_reason: str | None = None
                if item.speculative and (n_failed or remerge_occurred):
                    remerge_reason = 'previous_failed' if n_failed else 'chain_invalidated'
                elif (
                    item.immediate_outcome is None
                    and item.merge_result is not None
                    and not isinstance(req, GroupMergeRequest)
                ):
                    # Mechanism 2: check staleness at pickup for real items.
                    # Reading main once per non-speculative pickup adds one git
                    # rev-parse per item — negligible vs. the merge/verify cost.
                    #
                    # Train exemption (D9/I6, boundary test 12):
                    # GroupMergeRequest trains are exempt via two independent guards:
                    #   1. `item.immediate_outcome is None` — trains always set
                    #      immediate_outcome from _do_train_merge, so this is False
                    #      and the elif is skipped for every train.
                    #   2. `not isinstance(req, GroupMergeRequest)` — explicit
                    #      defense-in-depth so the exemption is clear at the
                    #      call site independent of the immediate_outcome contract.
                    # Trains are also structurally exempt from Mechanism 1: the
                    # GroupMergeRequest `continue` in _merger_loop executes before
                    # the _merge_ahead_cap.acquire() site, so trains never acquire
                    # the cap (counts_against_cap defaults to False).
                    current_main = await self._git_ops.get_main_sha()
                    if item.base_sha != current_main:
                        remerge_reason = 'main_advanced'

                if remerge_reason is not None:
                    # Set flag early so an exception during cleanup/_remerge still
                    # propagates chain invalidation to the next iteration.
                    iteration_did_remerge = True
                    # Make the item visible in snapshot() during the re-merge so it
                    # is never in-flight-but-invisible.  _remerge can be slow
                    # (a full merge operation); without this the item is popped from
                    # _verifier_queue but not yet in _verify_item — the exact
                    # "genuinely queued but invisible" window the tool was built to
                    # surface (reify 3112).  Use 'remerging' as a distinct phase so
                    # operators can distinguish it from normal verification.
                    self._verify_item = item
                    self._verify_phase = 'remerging'
                    # Clean up the stale merge worktree.
                    if item.merge_wt:
                        await self._git_ops.cleanup_merge_worktree(item.merge_wt)
                    self._emit_speculative(
                        EventType.speculative_discard, req.task_id,
                        reason=remerge_reason,
                    )
                    logger.info(
                        f'Task {req.task_id}: discarding stale merge '
                        f'({remerge_reason}), re-merging against actual main'
                    )
                    # force_verify for 'main_advanced': same precondition as the
                    # speculation-race retry — main advanced since the branch was
                    # pre-rebased, so the skip_verify invariant ('pre_rebased AND
                    # main unchanged') does not hold.  Always verify.
                    # chain-invalidation triggers ('previous_failed' /
                    # 'chain_invalidated') pass force_verify=False.  Those
                    # triggers fire when a PRIOR item failed, meaning main has
                    # NOT advanced since this branch was pre-rebased — the
                    # skip_verify invariant genuinely holds for those cases.
                    item = await self._remerge(
                        req, item.started_monotonic,
                        force_verify=(remerge_reason == 'main_advanced'),
                    )
                    # Update _verify_item to the freshly re-merged item; phase stays
                    # 'remerging' until _verify_and_advance transitions it.
                    self._verify_item = item

                # ── Immediate outcome (already_merged / conflict / blocked) ─
                if item.immediate_outcome is not None:
                    if not item.already_delivered and not req.result.done():
                        req.result.set_result(item.immediate_outcome)
                    # immediate_outcome is always identical to the OOB-delivered
                    # outcome at every _oob_deliver call site; no divergence today.
                    n_failed = item.immediate_outcome.status not in ('done', 'already_merged')
                    continue  # finally will call _speculation_slot.set()

                self._verify_item = item
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
                self._verify_item = None
                self._verify_phase = None
                self._verify_started_at = None
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

    async def _remerge(
        self,
        req: MergeRequest,
        started_monotonic: float | None,
        *,
        force_verify: bool = False,
    ) -> SpeculativeItem:
        """Re-merge a request against actual main after speculation invalidation.

        ``force_verify`` overrides the normal skip_verify computation in the
        normal-success return.  Set it to True when the re-merge is triggered
        by 'main_advanced': the branch was pre-rebased onto an old main while
        _remerge merges it against the current (newer) main, integrating commits
        the branch never incorporated.  The documented skip_verify invariant
        ('pre_rebased AND main unchanged') does NOT hold; skipping verification
        would let semantically-unverified main commits land on the protected
        branch.  Always verify — same reasoning as the speculation-race retry
        success return in this method (see the 'Always verify' comment below).

        Passing force_verify=False (the default) preserves the existing
        computation for chain-invalidation re-merges ('previous_failed' /
        'chain_invalidated'), keeping their skip_verify semantics unchanged
        (invariant 3).
        """
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
        # When force_verify is set (main_advanced re-merge), skip_verify is
        # unconditionally False — same 'Always verify' rule as the race-retry
        # success return above:  main advanced since the branch was pre-rebased,
        # so the invariant ('pre_rebased AND main unchanged') does not hold and
        # verification must run.
        if force_verify:
            skip_verify = False
        else:
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
            self._verify_phase = 'verifying'
            self._verify_started_at = time.time()  # wall-clock verify start for triage
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
        self._verify_phase = 'finalizing'
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
                    chain_ctx=_GenerationChainContext(
                        queue=self._queue,
                        counts=self._generation_chain_counts,
                        max_auto_generations=MAX_AUTO_CHAINED_GENERATIONS,
                    ),
                    merged_branch_tip=item.merged_branch_tip,
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

                self._verify_phase = 'gate_reverify'
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
                # Gate cleared — restore 'finalizing' so the next advance_main
                # call in the loop reports the correct phase, not 'gate_reverify'.
                self._verify_phase = 'finalizing'
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
                # Cleanup BEFORE _map_advance_failure so a cleanup raise
                # propagates before halt_for_wip is ever called -- mirroring
                # the serial MergeWorker path (cleanup_merge_worktree before
                # _map_advance_failure in MergeWorker._process_one) and the
                # abandoned-request short-circuit just above this block.
                # Without this order, a cleanup raise strands the queue
                # halted with no escalation owner on the single-task workflow
                # path, which routes 'blocked' to _mark_blocked with no
                # is_wip_halted probe (task 1598).
                await self._git_ops.cleanup_merge_worktree(merge_wt)
                outcome = await _map_advance_failure(
                    self._git_ops, result,
                    task_id=req.task_id,
                    merge_commit_fallback=merge_commit,
                    halt=self.halt_for_wip,
                    unhalt=self.unhalt_wip,
                    cas_retries=self._cas_retries,
                )
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


# ---------------------------------------------------------------------------
# Startup liveness-margin guard (task 1674)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MergeLivenessAssessment:
    """Return value from :func:`check_merge_liveness_margin`.

    All fields are informational; callers should treat :attr:`safe` as the
    primary decision bit and log/display the numeric fields for triage.

    Attributes:
        worst_case_secs: Computed worst-case time (seconds) a counted
            ``_merge-*`` worktree can legitimately remain queued without
            having its mtime updated, under the given config.
        threshold_secs: Safety threshold (``safety_factor * liveness_secs``);
            the guard fires when ``worst_case_secs >= threshold_secs``.
        liveness_secs: The reaper's liveness window passed to the guard.
        timeout_secs: Effective per-command merge-verify cold timeout resolved
            from the config (the primary knob that drives the worst-case).
        merge_ahead_bound: Injected (or defaulted) merge-ahead cap value.
        max_verify_timeouts: Injected (or defaulted) consecutive-timeout
            limit; included for informational display in log messages.
        safe: True iff ``worst_case_secs < threshold_secs``.
    """

    worst_case_secs: float
    threshold_secs: float
    liveness_secs: float
    timeout_secs: float
    merge_ahead_bound: int
    max_verify_timeouts: int
    safe: bool


def check_merge_liveness_margin(
    config: OrchestratorConfig,
    *,
    merge_ahead_bound: int = _MERGE_AHEAD_BOUND,
    max_verify_timeouts: int = SpeculativeMergeWorker.MAX_POST_MERGE_VERIFY_TIMEOUTS,
    liveness_secs: float = INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS,
    safety_factor: float = 0.75,
    logger: logging.Logger = logger,
) -> MergeLivenessAssessment:
    """Evaluate whether the merge-verify cold timeout fits safely within the
    reaper's liveness window and emit a WARNING when it does not.

    Called once at orchestrator startup (from ``Harness._start_merge_worker``)
    against the live per-project :class:`OrchestratorConfig`.

    **Physical model**

    With ``_MERGE_AHEAD_BOUND=N``, up to *N* counted ``_merge-*`` worktrees
    can sit in the :class:`SpeculativeMergeWorker` verifier queue while the
    verifier is busy with an earlier item.  Those queued worktrees have their
    mtime frozen at merge time and are NOT updated until the verifier picks
    them up.  The worst-case stale interval for a queued worktree is therefore
    bounded by the longest a single verify can run — the cold merge-verify
    timeout resolved from *config*.

    The reaper in :func:`coalesce_or_enqueue_merge_request` treats any
    ``_merge-*`` worktree whose mtime age exceeds
    :data:`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` as abandoned and reaps it.
    A queued-but-legitimate worktree is indistinguishable from an abandoned
    one once its mtime age exceeds that window.

    **Formula**

    .. code-block:: text

        worst_case = merge_ahead_bound * timeout
        threshold  = safety_factor * liveness_secs
        safe       = worst_case < threshold

    The formula uses a multiplier of 1 (not ``max_verify_timeouts + 1``)
    because each verify attempt creates and cleans up its own ``_merge-*``
    worktree; the per-task timeout-retry counter does not extend any single
    worktree's on-disk lifetime.  ``max_verify_timeouts`` is carried in the
    return value for operator context.

    **Default calibration**

    With ``safety_factor=0.75`` and ``liveness_secs=3600``:
    - threshold = 2700 s
    - A warm-only deployment (``verify_command_timeout_secs=1800``, no cold
      overrides) resolves timeout=1800 s → worst_case=1800 s < 2700 s → safe.
    - The shipped ``defaults.yaml`` (``merge_verify_cold_command_timeout_secs
      =7200``) resolves timeout=7200 s → worst_case=7200 s ≥ 2700 s → WARN.

    Args:
        config: Live per-project orchestrator config.
        merge_ahead_bound: Override for :data:`_MERGE_AHEAD_BOUND` (injectable
            for tests and future tuning).
        max_verify_timeouts: Override for
            :attr:`SpeculativeMergeWorker.MAX_POST_MERGE_VERIFY_TIMEOUTS`
            (informational; not part of the worst-case formula).
        liveness_secs: Override for :data:`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS`.
        safety_factor: Fraction of *liveness_secs* that constitutes the
            "comfortably below" threshold.  Default 0.75.
        logger: Logger to use for the WARNING (default: module logger, captured
            by ``pytest caplog`` as ``orchestrator.merge_queue``).

    Returns:
        :class:`MergeLivenessAssessment` with all computed values and the
        ``safe`` verdict.
    """
    # Resolve the effective per-command merge-verify cold timeout.
    # Merge worktrees are always cold (merge_queue.py:379-384), so we pass
    # is_cold=True and is_merge_verify=True to get the full cold cascade:
    #   merge_verify_cold_command_timeout_secs
    #   → verify_cold_command_timeout_secs
    #   → verify_command_timeout_secs (warm fallback)
    # module_config=None: no per-module override exists at startup.
    timeout_secs = _resolve_verify_timeout(
        config, None, is_cold=True, is_merge_verify=True,
    )

    worst_case_secs = merge_ahead_bound * timeout_secs
    threshold_secs = safety_factor * liveness_secs
    safe = worst_case_secs < threshold_secs

    assessment = MergeLivenessAssessment(
        worst_case_secs=worst_case_secs,
        threshold_secs=threshold_secs,
        liveness_secs=liveness_secs,
        timeout_secs=timeout_secs,
        merge_ahead_bound=merge_ahead_bound,
        max_verify_timeouts=max_verify_timeouts,
        safe=safe,
    )

    if not safe:
        logger.warning(
            'check_merge_liveness_margin: queued _merge-* worktree worst-case '
            'age (%.0fs) is not comfortably below the reaper liveness window '
            '(%.0fs, threshold=%.0fs, factor=%.2f). '
            'Reduce merge_verify_cold_command_timeout_secs (currently %.0fs) '
            'or raise INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS '
            '(merge_ahead_bound=%d, max_verify_timeouts=%d).',
            worst_case_secs,
            liveness_secs,
            threshold_secs,
            safety_factor,
            timeout_secs,
            merge_ahead_bound,
            max_verify_timeouts,
        )

    return assessment
