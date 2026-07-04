"""Post-merge gates, finalize, and reason-prefix constants (MQ-refactor task β).

Extracted verbatim from :mod:`orchestrator.merge_queue`: the plan-target /
plan-files-touched / post-merge-equivalence / post-merge-pyright gates, the
advance-finalize and advance-failure-mapping functions, and their supporting
types and reason-prefix constants.  ``merge_queue`` re-exports every name
here through a top-level shim so existing importers
(``from orchestrator.merge_queue import X``, etc.) keep working unchanged.

Verify EXECUTION (``_run_post_merge_verify`` and its cluster) stays in
``merge_queue`` — this module owns gate POLICY, not verify execution (PRD
Open Q1).  A moved function that calls a merge_queue-resident sibling —
whether that sibling stays permanently or co-moved here but is monkeypatched
by the existing test suite via the string path
``orchestrator.merge_queue.<name>`` — resolves it through a function-local
(deferred) import from :mod:`orchestrator.merge_queue` rather than a direct
intra-module reference.  This mirrors the existing
``_main_health_fingerprint`` convention in ``merge_queue.py`` and keeps this
module free of any top-level import of ``merge_queue`` (which would deadlock
module load, since merge_queue's shim needs this module fully defined
first).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import posixpath
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_types import MergeOutcome, MergeRequest, TerminalOutcomeRetention

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig

logger = logging.getLogger('orchestrator.merge_queue')


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


async def _resolve_second_parent(git_ops: GitOps, sha: str) -> str | None:
    """Return the second parent (``sha^2``) of a merge commit, or None.

    Runs ``git rev-parse <sha>^2`` in *git_ops.project_root*.  Returns the
    stripped SHA string on success (rc==0) or ``None`` if the commit has no
    second parent or the command fails (e.g. fast-forward commit, transient
    git error, or missing worktree).

    Used by :func:`_finalize_advanced_merge` to derive the branch tip that was
    actually merged into the ``--no-ff`` merge commit so the post-merge
    equivalence gate compares the right tree (drift-proof vs a lane whose
    worktree HEAD has been hijacked or rebased after snapshotting).

    Discoverability note: ``_finalize_advanced_merge`` never calls this
    definition directly — it reaches back through
    ``orchestrator.merge_queue._resolve_second_parent`` (see the module
    docstring's reach-back convention) so the existing suite's string-path
    monkeypatches stay effective.  This function body is nonetheless the
    canonical implementation that ``merge_queue`` re-exports.

    Fail-open: any exception returns ``None`` so the caller falls back to the
    next tier (``merged_branch_tip`` or ``req.snapshot_tip``).
    """
    # Reach-back (deferred import): the existing test suite patches the git
    # subprocess runner by string path at `orchestrator.merge_queue._run`
    # (this function used to live in merge_queue.py).  Resolving `_run`
    # dynamically from merge_queue's namespace at call time keeps those
    # patches effective post-extraction.
    from orchestrator.merge_queue import _run

    try:
        rc, out, _err = await _run(
            ['git', 'rev-parse', f'{sha}^2'],
            cwd=git_ops.project_root,
        )
    except Exception:
        # Abnormal failure (subprocess spawn error etc.) — the routine
        # "commit has no second parent" case returns rc != 0 below, NOT here.
        # Log so the fail-open to the next tip-resolution tier is observable
        # (task 1917 added this fail-open silently; greens the shared
        # silent-fallthrough gate, which requires WARN+ logging or re-raise).
        logger.warning(
            '_resolve_second_parent: git rev-parse %s^2 raised — '
            'falling open to None (caller uses next tip tier)',
            sha,
            exc_info=True,
        )
        return None
    if rc != 0:
        return None
    return out.strip() or None


async def _commit_is_linear(git_ops: GitOps, sha: str) -> bool:
    """Return ``True`` iff *sha* is a commit with at most one parent (linear/rebase).

    Runs ``git rev-list --parents -n 1 <sha>`` in *git_ops.project_root*.  The
    output is a single line of the form ``<sha> [<parent1> [<parent2> ...]]``.
    A linear (rebase-flattened) commit has ≤1 parent token after its own SHA,
    so the split yields ≤2 tokens total.

    Returns ``False`` on any git error or if the commit has ≥2 parents (a real
    ``--no-ff`` merge commit).  Fail-closed: returns ``False`` on error so the
    caller does NOT suppress the worktree-HEAD fallback when the linearity check
    is inconclusive (defense-in-depth — never masks a drop).

    Used by :func:`_finalize_advanced_merge` as the gating condition for the
    task-1928 fail-safe: the fail-safe (``allow_worktree_head_fallback=False``)
    is applied ONLY when (a) no trusted branch tip is recoverable via the
    three-term chain AND (b) the advanced SHA is *positively* linear.  This
    distinguishes a genuinely rebase-flattened landing (safe to skip the
    redundant gate — the tree was already re-verified by ``_reverify_rebased_tree``)
    from a transient ``^2`` git error on a real merge commit (must keep the
    legacy HEAD check to avoid masking a real drop).

    Discoverability note: like :func:`_resolve_second_parent`,
    ``_finalize_advanced_merge`` calls this via the
    ``orchestrator.merge_queue._commit_is_linear`` reach-back rather than
    this local definition, so the existing suite's monkeypatches on that
    string path stay effective.
    """
    # Reach-back (deferred import): see _resolve_second_parent — same
    # `orchestrator.merge_queue._run` string-path patch convention applies.
    from orchestrator.merge_queue import _run

    try:
        rc, out, _err = await _run(
            ['git', 'rev-list', '--parents', '-n', '1', sha],
            cwd=git_ops.project_root,
        )
    except Exception:
        logger.warning(
            '_commit_is_linear: git rev-list raised for %s — '
            'returning False (fail-closed: keep legacy HEAD check)',
            sha,
            exc_info=True,
        )
        return False
    if rc != 0:
        return False
    # Output: "<sha> [parent1] [parent2] ..."
    # Linear commit → ≤1 parent → ≤2 tokens.
    tokens = out.strip().split()
    return len(tokens) <= 2


@dataclasses.dataclass(frozen=True)
class GateVerdict:
    """Outcome of a single post-advance gate: pass, or block with a reason.

    A 2-state value object — either the gate passed (``passed=True``, every
    other field ``None``) or it blocked (``passed=False`` with ``reason``,
    ``merge_sha``, and ``emit_subtype`` all populated).  Construct via the
    :meth:`ok` / :meth:`block` factories rather than the constructor directly.
    """

    passed: bool
    reason: str | None = None
    merge_sha: str | None = None
    emit_subtype: str | None = None

    @classmethod
    def ok(cls) -> GateVerdict:
        """The gate passed."""
        return cls(passed=True)

    @classmethod
    def block(cls, *, reason: str, merge_sha: str | None, emit_subtype: str) -> GateVerdict:
        """The gate blocked the merge.

        *reason* becomes the terminal ``MergeOutcome.reason``, *merge_sha*
        its ``merge_sha`` (the merge already landed on main), and
        *emit_subtype* the ``merge_attempt`` subtype the caller emits.
        """
        return cls(passed=False, reason=reason, merge_sha=merge_sha, emit_subtype=emit_subtype)


@dataclasses.dataclass(frozen=True)
class Gate:
    """A single post-advance gate: a name, a run callable, and an optional
    failure interceptor.

    *run* takes a :class:`_PostAdvanceContext` and returns a
    :class:`GateVerdict`.  *on_blocked*, consulted only when *run* blocks,
    may return a substitute :class:`MergeOutcome` (e.g. the γ2 auto-chain's
    ``'superseded'``) to short-circuit the default outcome, or ``None`` to
    fall through to the ``'blocked'`` outcome built from the verdict.
    ``None`` (the default) means "no interceptor" — a pure registration.
    """

    name: str
    run: Callable[[_PostAdvanceContext], Awaitable[GateVerdict]]
    on_blocked: Callable[[_PostAdvanceContext], Awaitable[MergeOutcome | None]] | None = None


@dataclasses.dataclass(frozen=True)
class _PostAdvanceContext:
    """Everything a post-advance gate (or its on_blocked hook) needs.

    Bundles the post-``advance_main`` state :func:`_finalize_advanced_merge`
    computes once — notably ``resolved_merged_tip`` and
    ``allow_worktree_head_fallback``, each derived via a single git
    subprocess call — so every gate that consults them shares the same
    value instead of re-deriving it.
    """

    git_ops: GitOps
    req: MergeRequest
    event_store: EventStore | None
    advanced_sha: str
    base_sha: str
    resolved_merged_tip: str | None
    allow_worktree_head_fallback: bool
    started_monotonic: float | None
    train_id: str | None
    member_task_ids: list[str] | None
    chain_ctx: _GenerationChainContext | None
    merged_branch_tip: str | None
    log_label: str


async def _run_equivalence_gate(ctx: _PostAdvanceContext) -> GateVerdict:
    """Decision-2 post-merge content-equivalence gate.

    Discoverability note: reaches back through
    ``orchestrator.merge_queue._check_post_merge_equivalence`` (see the
    module docstring's reach-back convention) rather than calling the
    co-located definition directly, so the existing suite's string-path
    monkeypatches stay effective.
    """
    from orchestrator.merge_queue import _check_post_merge_equivalence

    equiv_failed = await _check_post_merge_equivalence(
        ctx.req.worktree, ctx.advanced_sha, ctx.git_ops, ctx.base_sha,
        task_id=ctx.req.task_id,
        merged_tip=ctx.resolved_merged_tip,
        allow_worktree_head_fallback=ctx.allow_worktree_head_fallback,
    )
    if not equiv_failed:
        return GateVerdict.ok()

    logger.warning(
        'Task %s%s: post-merge equivalence failed — '
        'branch HEAD and advanced main %s diverge in: %r',
        ctx.req.task_id, ctx.log_label, ctx.advanced_sha[:12], equiv_failed,
    )
    return GateVerdict.block(
        reason=(
            f'{POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX}: '
            f'branch and main diverge in '
            f'{", ".join(equiv_failed)}. '
            f'Conflict resolution likely dropped or rewrote '
            f'work; review {ctx.advanced_sha[:12]} against the '
            f'task branch tip.'
        ),
        merge_sha=ctx.advanced_sha,
        emit_subtype='post_merge_equivalence_failed',
    )


async def _auto_chain_on_equivalence_blocked(ctx: _PostAdvanceContext) -> MergeOutcome | None:
    """γ2: on equivalence-gate failure, try to auto-chain a gen-(n+1) request.

    Returns the chained ``'superseded'`` (or bound-exceeded ``'blocked'``)
    outcome when the kill-switch is on and the branch tip advanced mid-verify;
    ``None`` when auto-chaining does not apply (no *chain_ctx*, kill-switch
    off, or the tip genuinely did not advance), so the caller falls through
    to the default ``'blocked'`` outcome built from the gate's verdict.
    """
    from orchestrator.merge_queue import (
        AUTO_CHAIN_GENERATIONS_ENABLED,
        _elapsed_ms,
        _emit_merge_attempt,
        _maybe_auto_chain_generation,
    )

    if ctx.chain_ctx is None or not AUTO_CHAIN_GENERATIONS_ENABLED:
        return None

    chained = await _maybe_auto_chain_generation(
        ctx.req, ctx.advanced_sha, ctx.git_ops, ctx.event_store,
        merged_branch_tip=ctx.merged_branch_tip or ctx.resolved_merged_tip,
        counts=ctx.chain_ctx.counts,
        queue=ctx.chain_ctx.queue,
        max_auto_generations=ctx.chain_ctx.max_auto_generations,
        retention=ctx.chain_ctx.retention,
    )
    if chained is None:
        return None

    _emit_merge_attempt(
        ctx.event_store, ctx.req.task_id,
        'post_merge_generation_chained',
        duration_ms=_elapsed_ms(ctx.started_monotonic),
        train_id=ctx.train_id,
        member_task_ids=ctx.member_task_ids,
    )
    return chained


async def _run_pyright_gate(ctx: _PostAdvanceContext) -> GateVerdict:
    """Decision-3 post-merge unscoped type-check gate.

    Discoverability note: reaches back through
    ``orchestrator.merge_queue._check_post_merge_pyright`` rather than
    calling the co-located definition directly, so the existing suite's
    string-path monkeypatches stay effective.
    """
    from orchestrator.merge_queue import _check_post_merge_pyright

    pyright_result = await _check_post_merge_pyright(
        ctx.advanced_sha, ctx.git_ops, ctx.req.config, ctx.req.module_configs,
        task_id=ctx.req.task_id,
    )
    if not pyright_result.broken:
        return GateVerdict.ok()

    logger.warning(
        'Task %s%s: post-merge unscoped type-check failed for %s on %s',
        ctx.req.task_id, ctx.log_label,
        ', '.join(pyright_result.failing_subprojects),
        ctx.advanced_sha[:12],
    )
    return GateVerdict.block(
        reason=(
            f'{POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX}: '
            f'post-merge unscoped type-check failed for '
            f'{", ".join(pyright_result.failing_subprojects)} '
            f'on {ctx.advanced_sha[:12]}. {pyright_result.detail}'
        ),
        merge_sha=ctx.advanced_sha,
        emit_subtype='post_merge_pyright_broken',
    )


POST_ADVANCE_GATES: list[Gate] = [
    Gate('equivalence', _run_equivalence_gate, on_blocked=_auto_chain_on_equivalence_blocked),
    Gate('pyright', _run_pyright_gate),
]
"""Declarative post-advance gate chain, run in order by
:func:`_finalize_advanced_merge`.  Registering a new gate is an append to
this list — "a registration, not surgery" — rather than an edit to
``_finalize_advanced_merge``'s body.  ``merge_queue`` re-exports this exact
list object (not a copy) so a test — or a future gate author — can register
a gate by replacing/monkeypatching this module attribute."""


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
    advanced_sha: str | None = None,
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

    *advanced_sha* is the post-rebase SHA actually placed on main, threaded
    from :class:`~orchestrator.git_ops.AdvanceOutcome`.advanced_sha (task
    1997) — callers must pass ``outcome.advanced_sha`` from the
    ``advance_main`` call that preceded this one.  ``None`` (e.g. no rebase
    occurred) falls back to *merge_commit_fallback*.
    """
    from orchestrator.merge_queue import (
        _commit_is_linear,
        _elapsed_ms,
        _emit_merge_attempt,
        _resolve_second_parent,
    )

    cas_retries.pop(req.task_id, None)
    timeouts.pop(req.task_id, None)
    enospc_retries.pop(req.task_id, None)
    # Use the post-rebase SHA actually placed on main (advance_main
    # rebases on CAS retry; merge_commit_fallback is the stale
    # pre-rebase SHA and would fail done_provenance ancestor check).
    advanced_sha = advanced_sha or merge_commit_fallback

    # Resolve the branch tip that was actually merged (second parent of the
    # --no-ff merge commit).  Preferred over the live worktree HEAD so the
    # equivalence gate is drift-proof when a warm lane's worktree HEAD has
    # been rebased or hijacked after the snapshot was taken.
    # Fallback chain: advanced_sha^2 → merged_branch_tip (caller) → req.snapshot_tip.
    resolved_merged_tip = (
        (await _resolve_second_parent(git_ops, advanced_sha))
        or merged_branch_tip
        or getattr(req, 'snapshot_tip', None)
    )

    # Task-1928 fail-safe: if no trusted branch tip is recoverable via any of the
    # three terms AND advanced_sha is positively linear (confirmed single-parent),
    # suppress the live worktree-HEAD fallback.  A positively-linear advanced_sha
    # means the landed tree is a fast-forward or rebase-flattened projection of the
    # branch tip, so the HEAD-drift comparison can only produce phantom failures —
    # the tree is already equivalent by construction.  This holds for both rebase
    # landings (where _reverify_rebased_tree also ran) and ff-style landings (where
    # it did not): in both cases a genuine file drop would require the branch itself
    # to have differed from what was merged, which the linear commit topology rules
    # out.  The positive-linearity check (_commit_is_linear) distinguishes a confirmed
    # linear commit from a transient ^2 git error on a real merge commit —
    # the latter must keep the legacy HEAD check to avoid masking a real drop.
    allow_worktree_head_fallback = True
    if resolved_merged_tip is None and await _commit_is_linear(git_ops, advanced_sha):
        allow_worktree_head_fallback = False

    ctx = _PostAdvanceContext(
        git_ops=git_ops,
        req=req,
        event_store=event_store,
        advanced_sha=advanced_sha,
        base_sha=base_sha,
        resolved_merged_tip=resolved_merged_tip,
        allow_worktree_head_fallback=allow_worktree_head_fallback,
        started_monotonic=started_monotonic,
        train_id=train_id,
        member_task_ids=member_task_ids,
        chain_ctx=chain_ctx,
        merged_branch_tip=merged_branch_tip,
        log_label=log_label,
    )

    # Declarative post-advance gate chain (task λ): iterate the registered
    # gates in order.  Registering a new gate is an append to
    # POST_ADVANCE_GATES — "a registration, not surgery" — not an edit to
    # this loop.
    ran = []
    terminal = None
    for gate in POST_ADVANCE_GATES:
        ran.append(gate.name)
        verdict = await gate.run(ctx)
        if verdict.passed:
            continue
        # GateVerdict.block() always populates reason/emit_subtype; the
        # field type is Optional only because it's shared with the
        # all-None .ok() case.  Bind to locals and narrow once for pyright.
        # An explicit guard (not `assert`) so a mis-constructed verdict —
        # e.g. a future gate building GateVerdict(passed=False, ...)
        # directly instead of via .block() — fails loudly even under `-O`,
        # rather than silently emitting a None subtype.
        emit_subtype = verdict.emit_subtype
        reason = verdict.reason
        if emit_subtype is None or reason is None:
            raise AssertionError(
                f'gate {gate.name!r} returned a blocked GateVerdict with '
                f'reason/emit_subtype unset (reason={reason!r}, '
                f'emit_subtype={emit_subtype!r}); construct blocked verdicts '
                f'via GateVerdict.block(), which always populates both'
            )
        _emit_merge_attempt(
            event_store, req.task_id, emit_subtype,
            duration_ms=_elapsed_ms(started_monotonic),
            train_id=train_id,
            member_task_ids=member_task_ids,
        )
        # A gate's on_blocked hook (e.g. the equivalence gate's γ2
        # auto-chain) may substitute a chained outcome; None means "no
        # interceptor" or "interceptor declined" — fall through to the
        # default blocked outcome built from the verdict.
        if gate.on_blocked is not None:
            terminal = await gate.on_blocked(ctx)
        if terminal is None:
            terminal = MergeOutcome(
                'blocked', merge_sha=verdict.merge_sha, reason=reason,
            )
        break

    logger.info(
        'Task %s%s: post-advance gates run: %s',
        req.task_id, log_label, ', '.join(ran),
    )
    if terminal is not None:
        return terminal

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
    advanced_sha: str | None = None,
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

    *advanced_sha* is the post-rebase SHA actually placed on main, threaded
    from :class:`~orchestrator.git_ops.AdvanceOutcome`.advanced_sha (task
    1997) — only consulted on the ``pop_conflict`` branch (main did land in
    that case).  ``None`` falls back to *merge_commit_fallback*.
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
                advanced_sha = advanced_sha or merge_commit_fallback
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


async def _check_post_merge_equivalence(
    task_worktree: Path,
    advanced_sha: str,
    git_ops: GitOps,
    main_sha: str,
    *,
    task_id: str | None = None,
    merged_tip: str | None = None,
    allow_worktree_head_fallback: bool = True,
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

    *merged_tip* — when provided, it is used directly as the branch tip to
    compare against ``advanced_sha``, bypassing the ``git rev-parse HEAD``
    read in ``task_worktree``.  Pass the tip *actually merged* (e.g. the
    second parent of the merge commit, ``advanced_sha^2``) so that the gate
    is drift-proof when the worktree HEAD has been rebased or hijacked after
    the snapshot was taken.  When ``None`` (the default) the existing
    worktree-HEAD behaviour is preserved for back-compat.

    *allow_worktree_head_fallback* — when ``False`` and ``merged_tip`` is
    ``None``, return ``[]`` immediately WITHOUT reading the live worktree
    HEAD.  This is set to ``False`` by :func:`_finalize_advanced_merge`
    when no trusted branch tip is recoverable AND ``advanced_sha`` is
    positively linear (confirmed single-parent commit).  A linear
    ``advanced_sha`` means the landed tree is a fast-forward or
    rebase-flattened projection of the branch tip, so the HEAD-drift
    comparison can only produce phantom failures — the tree is equivalent
    by construction regardless of whether ``_reverify_rebased_tree`` ran on
    this path.  Default ``True`` preserves byte-for-byte back-compat for
    all existing callers.
    """
    if merged_tip is not None:
        branch_head = merged_tip
    elif not allow_worktree_head_fallback:
        # No trusted tip and caller explicitly suppressed the worktree-HEAD
        # fallback (advanced_sha is a positively-linear commit — the landed
        # tree is equivalent to the branch tip by construction).  Return []
        # rather than reading a potentially drifted live HEAD.
        logger.debug(
            'post-merge-equiv: allow_worktree_head_fallback=False with '
            'merged_tip=None — skipping gate (no HEAD read). '
            'task_id=%s advanced_sha=%s',
            task_id or '<unknown>', advanced_sha,
        )
        return []
    else:
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

    Discoverability note: :func:`_reverify_rebased_tree` calls this via the
    ``orchestrator.merge_queue._rebase_delta_touched_overlap`` reach-back
    rather than this local definition, so the existing suite's monkeypatches
    on that string path stay effective.
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
    merge_sha: str = '',
    keep_worktrees: Collection[Path] | None = None,
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
    keep_worktrees:
        Additional worktrees to protect from disk-guard pruning — forwarded
        verbatim to ``_run_post_merge_verify``.  Default ``None`` produces
        legacy single-keep behaviour (only ``merge_wt`` is protected).
    """
    from orchestrator.merge_queue import _rebase_delta_touched_overlap, _run_post_merge_verify

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
        merge_sha=merge_sha,
        keep_worktrees=keep_worktrees,
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
    from orchestrator.merge_queue import _run_unscoped_typechecks

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
