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
import json
import logging
import math
import os
import posixpath
import re
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import (
    PERSISTENT_MERGE_WORKTREE_NAME,
    GitOps,
    MergeResult,
    WorktreeMissing,
    _run,
)
from orchestrator.verify import (
    PREEXISTING_BREAK_SKIP_CATEGORIES,
    VerifyResult,
    _derive_task_files_from_git,
    run_scoped_verification,
    run_verification,
    verify_failure_is_preexisting_on_main,
)
from orchestrator.verify_runner import (
    UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY,
    DriftDetector,
    LocalRunner,
    RemoteRunner,
    VerifyRunnerPool,
    build_merge_verify_spec,
    is_unscoped_gate_failure,
    unscoped_gate_failing_subprojects,
)

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig

logger = logging.getLogger(__name__)


def _build_remote_runners(
    config: OrchestratorConfig,
    cwd: str | Path,
    *,
    quarantine: set[str] | None = None,
) -> list[RemoteRunner]:
    """Build the list of RemoteRunner instances from operator config (Lever C).

    Returns REMOTES ONLY — the LocalRunner trust anchor is prepended by callers
    since it needs call-specific arguments (worktree path, module configs, etc.).

    Filters out disabled runners (enabled=False) and any runner whose name is in
    the quarantine set (in-memory worker-level quarantine from DriftDetector).
    quarantine=None is treated as an empty set (no quarantine).

    _build_verify_runners passes main_branch=config.git.main_branch so the
    remote host receives a freshness push before the merge-sha transport.
    """
    return [
        RemoteRunner(
            name=r.name,
            ssh_host=r.ssh_host,
            git_remote=r.git_remote,
            cwd=cwd,
            config_path=r.config_path,
            main_branch=config.git.main_branch,
        )
        for r in config.enabled_verify_runners
        if quarantine is None or r.name not in quarantine
    ]


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
more often than _heartbeat_interval_s, without adding measurable overhead.

This constant is also a multiplicand of the liveness-margin guard's heartbeat
floor (see :data:`TOUCH_MISS_TOLERANCE`)."""

TOUCH_MISS_TOLERANCE: int = 20
"""Maximum number of consecutive _HEARTBEAT_POLL_S ticks a live worker's
owned ``_merge-*`` worktrees are permitted to miss before their mtime would
falsely age into the reaper window.

**Derivation**

The α owner-heartbeat (task 1728) touches every owned ``_merge-*`` worktree's
mtime every :data:`_HEARTBEAT_POLL_S` seconds.  Under normal operation a
worktree's mtime age is bounded by ~1 poll period (30 s).  A sustained
event-loop stall (GIL contention, heavy I/O, OS scheduling jitter) can delay
the heartbeat, but the stall budget is:

    floor_secs = _HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE = 30 × 20 = 600 s

**Margin calibration** (PRD §9, 10–60 band)

    default_threshold = 0.75 × INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS
                      = 0.75 × 10800 = 8100 s
    margin = 8100 / 600 = 13.5× ≥ 3× (PRD minimum)

20 is the PRD's running example ("e.g. 20 → 600 s floor").  The constant is
read in the body of :func:`check_merge_liveness_margin` (not as a default
argument) so that a harness-level test can monkeypatch it to force an
over-budget verdict without requiring a low-liveness config."""

AUTO_CHAIN_GENERATIONS_ENABLED: bool = False
"""Kill-switch for the γ2 generation auto-chaining producer.

MUST remain False until the sole remaining γ3 precondition lands:
  1. ✓ DONE (task/1717): The workflow.py 'superseded' consumer handler — the
     single-task merge consumer in ``_submit_to_merge_queue`` now parks
     'superseded' outcomes as merge-deferred via ``_handle_superseded`` instead
     of falling through to ``_mark_blocked`` with an empty reason.
  2. The gen-(n+1) registry slot handoff via ATTACH_AND_CHAIN (re-acquiring the
     branch slot in InFlightMergeRegistry for the chained request without tripping
     the gen-1 done-callback double-release, and threading TerminalOutcomeRetention
     from the harness into the workers — harness.py:3238 omits both today).

While False, _finalize_advanced_merge ignores chain_ctx on equivalence failures
and returns 'blocked' exactly as before γ2, so no 'superseded' outcome can reach
the workflow consumer."""


MERGE_LANES: tuple[str, ...] = ('high', 'normal')
"""Priority-ordered lane names (high first).  Used by SpeculativeMergeWorker to
drain and pick requests: the first non-halted non-empty lane wins."""


def _normalize_lane(lane: str) -> str:
    """Map an unrecognised lane value to 'normal' (defensive; prevents starvation)."""
    return lane if lane in MERGE_LANES else 'normal'


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

TRAIN_VERIFY_FAILED_REASON_PREFIX = (
    'Train union verify failed (re-verifying members as singles)'
)
"""Prefix tagged onto ``MergeOutcome.reason`` when the post-merge verify of a
train (union of all member commits) fails with a structured failure_category
AND the failure is NOT a pre-existing main-health-red break.

This precise set of conditions identifies an "interaction candidate" — a
verify-red that *could* be caused by a cross-member interaction rather than a
single broken member.  Workflow δ intercepts this prefix and falls back to
re-verifying each member as a solo branch before blocking or escalating.

Outcomes that must NOT be tagged (and are NOT):
  - main-health-red (``reason`` starts with ``MAIN_HEALTH_RED_REASON_PREFIX``):
    the break pre-exists on bare main; per-member re-verify would be spurious.
  - transient-infra / disk-guard (``failure_category == ''``): no structured
    failure; re-verify would be unreliable.
  - rebase-conflict (``reason`` starts with ``TRAIN_REBASE_CONFLICT_REASON_PREFIX``):
    git-level conflict, not a test failure.
"""

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

MAIN_HEALTH_RED_REASON_PREFIX = 'Main-health red: merge verify failure reproduces on bare main'
"""Prefix of the ``MergeOutcome.reason`` string emitted when the post-merge
verify failure is classified as a pre-existing break already present on bare
main HEAD — i.e. not introduced by this task's merge.

The workflow ``_submit_to_merge_queue`` pattern-matches this prefix and routes
the outcome to a single dedup'd L1 escalation (``escalate_to_human=True``,
``category='preexisting_main_break'``), skipping the steward entirely.  The
steward operates in-branch and cannot fix a broken main; routing each
concurrent failing merge to the steward would cause an ~85 min-per-task
livelock for every task whose verify runs against a red main."""


# ---------------------------------------------------------------------------
# Auto-heal eligibility gate
# ---------------------------------------------------------------------------

# Conservative allowlist of verify.py category strings that are mechanical
# (small diff, deterministic cause) and therefore safe to auto-spawn a fix
# task for.  The category strings are those emitted by
# ``verify._classify_failure``; compile_error covers the tsc/type/lint class.
#
# Deliberately EXCLUDED — not eligible for auto-heal:
#   test_failure:         multi-file; needs human judgement on test expectations
#   unknown_test_failure: ambiguous signal; may be flaky
#   infra_timeout:        infra problem; auto-spawning cannot fix infra
#   flock_error:          transient lock contention; auto-spawning cannot fix
AUTO_HEAL_MECHANICAL_CATEGORIES: frozenset[str] = frozenset({'compile_error'})


def is_auto_heal_eligible(
    category: str | None, cause_hint: str | None,  # noqa: ARG001
) -> bool:
    """Return True when the failure class is safe for automated fix spawning.

    The allowlist is intentionally conservative — start with compile_error
    (the tsc/type/lint class, typically a small bounded diff) and expand
    only after production evidence of clean auto-heals for each new class.

    ``cause_hint`` is accepted for future per-hint filtering but is not
    currently used; the gate is purely category-based.
    """
    return (category or '') in AUTO_HEAL_MECHANICAL_CATEGORIES


def lane_for_task_metadata(metadata: dict | None) -> Literal['normal', 'high']:
    """Return the merge lane to use based on task metadata.

    Reads ``metadata['merge_lane']`` and validates via :func:`_normalize_lane`
    (unknown/missing → 'normal').  Passing ``None`` or an empty dict returns
    the default 'normal' lane.
    """
    value = (metadata or {}).get('merge_lane', '')
    return _normalize_lane(value or '')  # type: ignore[return-value]


_FIX_BRIEF_TITLE_MAX = 100
"""Maximum length for a compose_fix_main_brief title."""
_FIX_BRIEF_DETAIL_MAX = 800
"""Maximum bytes of detail to embed in a compose_fix_main_brief description."""


def compose_fix_main_brief(
    category: str, cause_hint: str, detail: str = '',
) -> tuple[str, str]:
    """Build a task title and description for a main-health fix task.

    The title follows the pattern ``'fix main: <cause_hint-or-category>'``,
    truncated to :data:`_FIX_BRIEF_TITLE_MAX` characters.  The description
    states that this is an automated main-health fix and includes ``category``,
    ``cause_hint``, and a bounded slice of ``detail``.

    Pure function — no I/O, no side effects.
    """
    # Title: use cause_hint when non-empty; fall back to category
    label = cause_hint.strip() if cause_hint and cause_hint.strip() else category
    raw_title = f'fix main: {label}'
    title = raw_title[:_FIX_BRIEF_TITLE_MAX]

    # Description: automated context header + structured fields + truncated detail
    detail_snippet = (detail or '').strip()[:_FIX_BRIEF_DETAIL_MAX]
    description = (
        f'Automated main-health fix task.\n'
        f'Category: {category}\n'
        f'Cause: {cause_hint}\n'
    )
    if detail_snippet:
        description += f'\nDetail:\n{detail_snippet}'

    return title, description


MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS: int = 1
"""Maximum number of auto-heal attempts allowed per sha-independent failure signature.

A value of 1 means: attempt the first auto-heal, but if the same signature
recurs afterwards (heal → re-break loop), hard-escalate instead of spawning
another fix task.  Promotes to config if tuning is needed.
"""


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
    keep_worktrees: Collection[Path] | None = None,
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
    keep = {merge_wt, *(keep_worktrees or ())}
    pruned = await git_ops.prune_stale_merge_worktrees(keep=keep)
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


def _main_health_fingerprint(category: str, cause_hint: str, probe_sha: str) -> str:
    """Compose a dedupe fingerprint for a preexisting-main-break outcome.

    Lazy-imports compute_preexisting_main_break_fingerprint from workflow so
    merge_queue→workflow is a deferred import (no module-level cycle).
    Fail-safe: returns '' when the import or composition raises.
    """
    try:
        from orchestrator.workflow import compute_preexisting_main_break_fingerprint
        return compute_preexisting_main_break_fingerprint(category, cause_hint, probe_sha)
    except Exception:
        return ''


async def _classify_main_health_red(
    git_ops: GitOps,
    req: MergeRequest,
    verify: VerifyResult,
    event_store: EventStore | None = None,
) -> MergeOutcome | None:
    """Probe whether *verify* is a pre-existing break already on bare main HEAD.

    Returns a :class:`MergeOutcome` with reason starting with
    :data:`MAIN_HEALTH_RED_REASON_PREFIX` when the break is confirmed
    pre-existing.  Returns ``None`` to fall through to the normal task-fault
    outcome.

    Guards (short-circuit to None before calling the probe):
    - ``req.config.escalate_preexisting_main_break`` is False
    - ``verify.timed_out`` is True (non-deterministic; re-probing is wasteful)
    - ``verify.category`` is in :data:`PREEXISTING_BREAK_SKIP_CATEGORIES`
      (infra_timeout / flock_error — inherently flaky)

    Latency: the first failing merge per ``(main_sha, category, cause_hint)``
    tuple pays a full probe build/test cost, bounded by the same
    ``merge_verify_cold_command_timeout_secs`` / ``verify_cold_command_timeout_secs``
    timeout that ``_run_post_merge_verify`` uses (inherited via
    ``run_scoped_verification``'s config lookup — the probe runs with
    ``max_retries=0`` so a hung probe cannot stall the worker beyond one
    timeout window).  Subsequent merges with the same signature hit the
    ``_PROBE_CACHE`` and pay only a ``get_main_sha()`` round-trip.
    """
    if not req.config.escalate_preexisting_main_break:
        return None
    if verify.timed_out:
        return None
    if (verify.category or '') in PREEXISTING_BREAK_SKIP_CATEGORIES:
        return None
    try:
        is_preexisting, probe_sha = await verify_failure_is_preexisting_on_main(
            req.worktree, req.config, req.module_configs, req.task_files,
            verify, git_ops,
        )
    except Exception:
        return None
    if not is_preexisting:
        return None
    detail = verify.failure_report()
    suffix = (verify.cause_hint or verify.summary or '')[:160]
    reason = (
        f'{MAIN_HEALTH_RED_REASON_PREFIX} '
        f'(category={verify.category!r}): {suffix}'
    )
    if detail:
        reason = f'{reason}\n\n{detail}'
    outcome = MergeOutcome(
        'blocked',
        reason=reason,
        failure_category=verify.category,
        failure_cause_hint=verify.cause_hint,
        dedupe_fingerprint=_main_health_fingerprint(
            verify.category or '', verify.cause_hint, probe_sha,
        ),
    )
    _emit_merge_attempt(event_store, req.task_id, 'main_health_red')
    return outcome


async def _run_post_merge_verify(
    git_ops: GitOps,
    req: MergeRequest,
    merge_wt: Path,
    *,
    timeouts: dict[str, int],
    enospc_retries: dict[str, int],
    max_timeouts: int,
    max_enospc: int,
    event_store: EventStore | None = None,
    merge_sha: str = '',
    on_result: Callable[[VerifyResult], None] | None = None,
    quarantine: set[str] | None = None,
    keep_worktrees: Collection[Path] | None = None,
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

    Args:
        on_result: Optional callback invoked with the final :class:`~orchestrator.verify.VerifyResult`
            BEFORE the pass/fail branch — additive, default ``None`` keeps
            :class:`MergeWorker` call sites byte-identical.  Used by
            :class:`SpeculativeMergeWorker` to capture warm per-test results
            for PRD §10 invariant 6(b) shadow compare.
        keep_worktrees: Additional worktrees to protect during any disk-pressure
            prune triggered inside this verify run (pre-verify guard and ENOSPC
            retry).  Default ``None`` keeps the legacy single-keep behaviour
            (only *merge_wt* is protected).  Pass
            ``set(worker._owned_merge_worktrees)`` from
            :class:`SpeculativeMergeWorker` to protect all in-flight + queued
            speculative worktrees (PRD §5 decision 5).

            **Snapshot timing:** The caller's keep-set is a one-shot snapshot
            taken at dispatch.  Worktrees registered into the ledger *after*
            this snapshot is taken are not protected by prunes triggered inside
            this call — they rely on the heartbeat mtime / grace-period
            mechanism to avoid premature removal in the residual window between
            snapshot capture and prune execution.
    """
    # Pre-verify disk guard: if free space is low, prune stale merge
    # worktrees; if still low, skip the build and escalate as transient
    # infra rather than entering a doomed multi-minute ENOSPC build.
    disk_reason = await _ensure_verify_disk_space(
        git_ops, merge_wt,
        req.config.merge_verify_min_free_disk_bytes, req.task_id,
        keep_worktrees=keep_worktrees,
    )
    if disk_reason is not None:
        await git_ops.cleanup_merge_worktree(merge_wt)
        return MergeOutcome('blocked', reason=disk_reason, verify_skipped=True)

    # Build the spec (carried for forward-compat with γ/δ remote runners;
    # the LocalRunner does not use it to drive execution).
    task_files_tuple = tuple(req.task_files) if req.task_files is not None else None

    # Mechanism 5b (dispatching-host scope derivation): when Lever C is on and
    # task_files was not supplied, derive the scope on the DISPATCHING host
    # (which has a fresh main) before building the spec.  This ships a faithful
    # non-None task_files in the spec so the remote's run_scoped_verification
    # never calls _derive_task_files_from_git against its own possibly-stale main.
    # Gated on enabled_verify_runners so the local-only path is byte-identical
    # when Lever C is off (zero behaviour change, trivially revertible).
    if task_files_tuple is None and req.config.enabled_verify_runners:
        derived = await _derive_task_files_from_git(merge_wt, req.config)
        if derived:
            task_files_tuple = tuple(derived)

    spec = build_merge_verify_spec(req.config, req.module_configs, task_files_tuple)

    # β decision 6: LOCAL-ONLY pool for all direct callers of _run_post_merge_verify
    # (MergeWorker._do_merge, _reverify_rebased_tree, reverify_member_solo,
    # _do_train_merge — recovery/train paths stay on the trust anchor and out of
    # slot accounting).  Remote dispatch is handled by γ's concurrent acquire/release
    # path via HostAllocator.  The `quarantine` parameter is reserved/unused here;
    # it remains in the signature so existing call sites stay byte-identical.
    # _run_cold_shadow_verify was already LOCAL-ONLY (cold trust-anchor design decision).
    pool = VerifyRunnerPool(
        [LocalRunner(
            merge_wt, req.config, req.module_configs, task_files_tuple,
            run_scoped=run_scoped_verification,
            run_unscoped=_run_unscoped_typechecks,
            task_id=req.task_id,
        )],
        event_store=event_store,
        task_id=req.task_id,
    )

    # max_retries=0: post-merge verify hangs are usually deterministic
    # (e.g. a deadlocked test); retrying just multiplies queue-wide stall.
    # is_merge_verify=True: merge worktrees are freshly created per
    # merge (no `.task/` dir and no warm cargo cache), so the cold
    # timeout applies despite `_is_verify_cold`'s filesystem heuristic
    # classifying them as warm.  The per-command cold timeout used here
    # is `merge_verify_cold_command_timeout_secs` (config default 7200 s)
    # if set, falling back to `verify_cold_command_timeout_secs` then warm.
    verify = await pool.dispatch(merge_sha, spec)

    # Transient-infra (disk pressure) retry: an ENOSPC failure is
    # often a self-healing host condition.  Prune stale _merge-*
    # worktrees (never task worktrees) and retry the verify once in
    # the same merge_wt before escalating.
    # ENOSPC always comes from the scoped phase (before the unscoped gate
    # runs), so the sentinel check is not needed here.
    if not verify.passed and _verify_hit_enospc(verify):
        prior_enospc = enospc_retries.get(req.task_id, 0)
        if prior_enospc < max_enospc:
            enospc_retries[req.task_id] = prior_enospc + 1
            enospc_keep = {merge_wt, *(keep_worktrees or ())}
            pruned = await git_ops.prune_stale_merge_worktrees(keep=enospc_keep)
            logger.warning(
                'Task %s: post-merge verify hit ENOSPC; pruned %d '
                'stale merge worktree(s), retrying verify once',
                req.task_id, len(pruned),
            )
            verify = await pool.dispatch(merge_sha, spec, attempt=1)

    # Invoke the optional result-capture callback (PRD §10 invariant 6(b)):
    # called with the FINAL VerifyResult (after any ENOSPC retry) so the
    # warm per-test results are always the last-observed verify for this commit.
    # Default None keeps MergeWorker call sites byte-identical.
    if on_result is not None:
        on_result(verify)

    if not verify.passed:
        await git_ops.cleanup_merge_worktree(merge_wt)

        # Unscoped-gate sentinel: check this BEFORE the ENOSPC guard because the
        # sentinel's type_output field carries type-check output (gate.detail) that
        # could contain disk-full markers and be misclassified as transient-infra.
        # Returns blocked directly without running the ENOSPC retry or main-health probe.
        if is_unscoped_gate_failure(verify):
            failing = ', '.join(unscoped_gate_failing_subprojects(verify))
            if verify.category == UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY:
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
                detail = verify.type_output or ''
                if detail:
                    reason = f'{reason}\n\n{detail}'
            return MergeOutcome('blocked', reason=reason)

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

        # Main-health probe: classify whether this failure is pre-existing on
        # bare main HEAD rather than introduced by this merge.  Inserted after
        # the ENOSPC early-return (ENOSPC is always task-side infra) and before
        # the generic task-fault build so all 4 merge paths are covered uniformly.
        # merge_wt is already cleaned up; the probe builds its own _mainprobe-
        # worktree and always cleans it in a finally block.
        main_health_outcome = await _classify_main_health_red(
            git_ops, req, verify, event_store,
        )
        if main_health_outcome is not None:
            return main_health_outcome
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
        return MergeOutcome(
            'blocked', reason=reason,
            failure_category=verify.category,
            failure_cause_hint=verify.cause_hint,
        )

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


async def _resolve_commit_tree(git_ops: GitOps, commit: str) -> str | None:
    """Return the tree SHA for *commit* (``git rev-parse <commit>^{tree}``).

    Returns the tree SHA string on success, or None on any git error.
    Callers that use this to gate a skip-verify decision must treat None as
    fail-closed (i.e., verify rather than skip).
    """
    if not commit:
        return None
    rc, out, err = await _run(
        ['git', 'rev-parse', f'{commit}^{{tree}}'],
        cwd=git_ops.project_root,
    )
    if rc != 0:
        logger.warning(
            '_resolve_commit_tree: git rev-parse %s^{tree} failed '
            '(rc=%d, stderr=%s); failing closed',
            commit[:8], rc, err.strip(),
        )
        return None
    return out.strip() or None


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


INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS: int = 10800
"""Maximum age (seconds, wall-clock mtime) for an on-disk ``_merge-*`` worktree
to be considered actively in-flight rather than abandoned.

.. note:: Raising this value (or raising :data:`_MERGE_AHEAD_BOUND`, or
   increasing the merge-verify cold timeout) affects the safety margin computed
   by :func:`check_merge_liveness_margin`.  Run that guard after changing any
   of these three values.

Set to 3 hours (10800 s) so the liveness window comfortably exceeds the cold
merge-verify budget shipped in ``defaults.yaml``
(``merge_verify_cold_command_timeout_secs: 7200``).  With ``safety_factor=0.75``
the guard threshold is ``0.75 × 10800 = 8100 s > 7200 s``, so the shipped
config is silent and in-flight verifies are never reaped mid-run.

**Tradeoff**: a genuinely-abandoned ``_merge-*`` worktree (e.g. from a crashed
worker) now lingers up to ~3 hours before the reaper reclaims it.  The cost is
disk space only — no correctness impact — and is the deliberate, accepted cost
of closing the merge-verify-vs-reaper race (esc-1674-34 Option 2).

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

    # Set the counter BEFORE enqueue so the cleanup callback can safely
    # pop it on any terminal path — including if the worker finalizes
    # gen_next before the post-await resumes (closes the set-after-pop race).
    counts[req.branch] = new_count

    # Register a fire-and-forget cleanup callback that pops the per-branch
    # lineage counter on EVERY terminal outcome EXCEPT 'superseded'.
    # 'superseded' means _maybe_auto_chain_generation already enqueued a
    # gen-(n+2) successor and incremented counts[branch] for it; popping
    # there would reset the MAX_AUTO_CHAINED_GENERATIONS bound to 0 every
    # generation.  This is a SECOND, independent add_done_callback alongside
    # the retention _on_finalized registered by enqueue_merge_request — both
    # coexist on gen_next.result.  The callback fires regardless of which
    # worker (MergeWorker or SpeculativeMergeWorker) finalizes gen_next.
    _branch = req.branch  # close over the branch name
    def _cleanup_chain_counter(fut: asyncio.Future) -> None:  # noqa: ANN001
        try:
            if fut.cancelled():
                # Cancellation — lineage ends; pop the counter.
                counts.pop(_branch, None)
            elif fut.exception() is not None:
                # Unhandled exception — lineage ends; pop the counter.
                counts.pop(_branch, None)
            else:
                outcome: MergeOutcome = fut.result()
                if outcome.status != 'superseded':
                    # Terminal that doesn't hand off to a gen-(n+2) successor
                    # (e.g. 'done', 'blocked', 'conflict', 'cancelled', …):
                    # release the lineage counter.  'done' is also caught here
                    # but is already popped by _finalize_advanced_merge — the
                    # pop() is idempotent so both callbacks are harmless.
                    counts.pop(_branch, None)
                # 'superseded': lineage continues; keep the counter.
        except Exception:  # noqa: BLE001
            logger.warning(
                '_maybe_auto_chain_generation: _cleanup_chain_counter failed '
                'for branch=%s', _branch, exc_info=True,
            )

    gen_next.result.add_done_callback(_cleanup_chain_counter)
    await enqueue_merge_request(queue, gen_next, event_store, retention=retention)
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
    lane: Literal['normal', 'high'] = field(default='normal', kw_only=True)
    """Priority lane for this request.  'high' requests are picked before all
    'normal' requests; within a lane FIFO order is preserved."""


@dataclass(frozen=True)
class TrainCallbacks:
    """Scheduler-backed callbacks for a single train, built in the harness.

    Holds two async callables captured over a live scheduler + train_id so
    the merge worker (a pure git engine with no scheduler import) can flip
    member tasks done after a train advance.  Built by
    :func:`harness.build_train_callback_factory` and consumed by task γ when
    it constructs :class:`GroupMergeRequest` inside SpeculativeMergeWorker.
    """

    status_check: Callable[[list[str]], Awaitable[dict[str, str]]]
    """Async callback: given member_task_ids, return {task_id: status}."""

    mark_member_done: Callable[[str, str], Awaitable[None]]
    """Async callback: mark a single member task done with the merge SHA."""


# Type alias for the factory that produces per-train callbacks.
# Called with a train_id str; returns a TrainCallbacks for that train.
TrainCallbackFactory = Callable[[str], TrainCallbacks]

# δ/1720 merge-ready confidence gate constants.
# Prefix used when constructing coalesce-formed train IDs in
# _maybe_coalesce_waiting_singles.  Used as the single source of truth
# so the merger-loop recording hook can identify coalesce-formed trains
# without a separate flag field on GroupMergeRequest.
_COALESCE_TRAIN_ID_PREFIX = 'coalesce-'

# Terminal merge outcomes in this set are treated as "risky" by the default
# predicate — a waiting single whose branch's most-recent merge_finalized
# event has one of these states is excluded from train formation.
_COALESCE_RISKY_TERMINAL_STATES: frozenset[str] = frozenset({'blocked', 'error'})

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


async def reverify_member_solo(
    git_ops: GitOps,
    member_id: str,
    solo_wt: Path,
    solo_branch: str,
    tip_sha: str,
    config: OrchestratorConfig,
    task_files: list[str] | None,
    module_configs: list[ModuleConfig],
    event_store: EventStore | None = None,
) -> SoloVerifyResult:
    """Run post-merge verification on a single train member's un-stacked solo branch.

    Wraps :func:`_run_post_merge_verify` (which provides disk-guard, ENOSPC
    prune-retry, and timeout loop-breaker semantics) but does NOT advance main.
    Fresh per-call ``timeouts`` / ``enospc_retries`` dicts are used so solo
    attempts do not count against the tip's existing timeout budgets.

    Returns a :class:`SoloVerifyResult`:
      - ``passed=True``  when ``_run_post_merge_verify`` returns ``None``.
        The solo worktree and branch are **left intact** so that
        ``_attribute_train_failure`` can call ``advance_main`` against the live
        worktree without re-materialising it (keeps verify cost ≤N+1).
        ``solo_wt`` and ``solo_branch`` are populated in the result for handoff.
      - ``passed=False`` when it returns a :class:`MergeOutcome`.
        Both the worktree (``cleanup_merge_worktree``) and the bare branch
        (``git_ops.delete_solo_branch``) are torn down before returning.
        A failer is never landed, so its solo is no longer needed.

    Args:
        git_ops:        GitOps instance for the current repo.
        member_id:      The member's task id (used for logging and the result).
        solo_wt:        Path to the isolated solo worktree (from
                        :meth:`~GitOps.materialize_member_solo`).
        solo_branch:    Name of the temporary solo branch (e.g. ``_solo-b2``).
        tip_sha:        Rebased tip SHA of the solo branch (the "merge SHA"
                        for the verify run and for the returned result).
        config:         OrchestratorConfig carrying verify command / disk limits.
        task_files:     Task-scoped files list for scoped verify (may be None).
        module_configs: Module-level configs for multi-module projects.
        event_store:    Optional EventStore for telemetry (may be None).

    Returns:
        :class:`SoloVerifyResult` with passed/failed verdict and reason.
    """
    req = MergeRequest(
        task_id=member_id,
        branch=solo_branch,
        worktree=solo_wt,
        pre_rebased=True,
        task_files=task_files,
        module_configs=module_configs,
        config=config,
        result=asyncio.get_running_loop().create_future(),
    )

    outcome = await _run_post_merge_verify(
        git_ops, req, solo_wt,
        timeouts={},
        enospc_retries={},
        max_timeouts=3,
        max_enospc=3,
        event_store=event_store,
        merge_sha=tip_sha,
    )
    if outcome is None:
        # Pass: hand off the live worktree+branch to _attribute_train_failure.
        return SoloVerifyResult(
            member_id=member_id,
            passed=True,
            merge_sha=tip_sha,
            reason='',
            solo_wt=solo_wt,
            solo_branch=solo_branch,
        )

    # Fail: tear down the solo worktree and branch — failer is never landed.
    try:
        await git_ops.cleanup_merge_worktree(solo_wt)
    except Exception:  # noqa: BLE001
        logger.warning(
            'reverify_member_solo: cleanup_merge_worktree failed for member %s',
            member_id, exc_info=True,
        )
    try:
        await git_ops.delete_solo_branch(solo_branch)
    except Exception:  # noqa: BLE001
        logger.warning(
            'reverify_member_solo: delete_solo_branch failed for member %s',
            member_id, exc_info=True,
        )
    return SoloVerifyResult(
        member_id=member_id,
        passed=False,
        merge_sha=None,
        reason=outcome.reason,
        # solo_wt and solo_branch are set to None to reflect that they
        # have already been torn down above.  A caller that iterated
        # failer results and trusted these fields would double-clean or
        # operate on a stale path.
        solo_wt=None,
        solo_branch=None,
    )


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
        data={
            'member_count': len(req.member_task_ids),
            'base_sha': base_sha_t0,
            'train_scope': 'workspace' if req.config.merge_verify_workspace else 'union',
        },
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
        event_store=event_store,
        merge_sha=merge_commit,
    )
    if verify_outcome is not None:
        reason = verify_outcome.reason
        # Tag the reason with TRAIN_VERIFY_FAILED_REASON_PREFIX when this is an
        # "interaction candidate" — a structured verify-red that could be caused
        # by a cross-member interaction rather than a single broken member.
        # Conditions: failure_category is non-empty (structured failure) AND the
        # reason does NOT already start with MAIN_HEALTH_RED_REASON_PREFIX (which
        # indicates a pre-existing break on bare main, not an interaction).
        # Rebase-conflict, disk-guard, transient-infra, and unscoped-pyright
        # failures all leave failure_category='' and therefore are NOT tagged.
        _is_interaction_candidate = (
            verify_outcome.failure_category != ''
            and not reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX)
        )
        if _is_interaction_candidate:
            tagged_reason = f'{TRAIN_VERIFY_FAILED_REASON_PREFIX}: {reason}'
            logger.info(
                'Train %s: verify gate interaction-candidate — tagging reason for δ attribution',
                req.train_id,
            )
            verify_outcome = MergeOutcome(
                verify_outcome.status,
                reason=tagged_reason,
                failure_category=verify_outcome.failure_category,
                failure_cause_hint=verify_outcome.failure_cause_hint,
            )
            reason = tagged_reason
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

    Provides the halt-owner methods that both :class:`MergeWorker` and
    :class:`SpeculativeMergeWorker` expose as public API to ``workflow.py``
    and ``harness.py``.

    Per-lane halt state: each lane in MERGE_LANES has an independent
    asyncio.Event (set = not halted; cleared = halted) and an optional owner
    esc_id.  Legacy all-lanes methods (halt_for_wip, unhalt_wip, …) are
    preserved as backward-compatible shims.

    Methods-only: each concrete worker's ``__init__`` is responsible for
    building the dicts::

        self._lane_halt = {l: asyncio.Event() for l in MERGE_LANES}
        for l in MERGE_LANES: self._lane_halt[l].set()
        self._lane_halt_owner: dict[str, str | None] = {l: None for l in MERGE_LANES}
    """

    # Class-level annotations so pyright sees the attributes.
    _lane_halt: dict[str, asyncio.Event]
    _lane_halt_owner: dict[str, str | None]
    # Dedicated operator-halt signal — set ONLY by operator_halt() (the outward
    # halt_merge_queue tool path), never by the automatic per-lane WIP halt. The
    # verifier checks THIS (not is_wip_halted) to decide whether to abort an
    # in-flight verify, so halt_for_wip's drain-the-verifier behaviour is left
    # untouched. set() = operator halt active.
    _operator_halt: asyncio.Event

    # ── per-lane public API ────────────────────────────────────────────────

    def halt_lane(self, lane: str, reason: str, *, owner_esc_id: str | None = None) -> None:
        """Halt a specific lane."""
        logger.warning('Merge queue lane %r halted: %s', lane, reason)
        self._lane_halt[lane].clear()
        if owner_esc_id is not None:
            self._lane_halt_owner[lane] = owner_esc_id

    def unhalt_lane(self, lane: str, reason: str | None = None) -> None:
        """Resume a specific lane."""
        logger.info(
            'Merge queue lane %r un-halted%s',
            lane,
            f' ({reason})' if reason else '',
        )
        self._lane_halt[lane].set()
        self._lane_halt_owner[lane] = None
        self._signal_resume()

    def is_lane_halted(self, lane: str) -> bool:
        """True iff the given lane is currently halted."""
        return not self._lane_halt[lane].is_set()

    # ── per-lane owner API (step-12 extensions live here too) ─────────────

    def set_lane_halt_owner(self, lane: str, esc_id: str) -> None:
        """Register the escalation that owns the halt on *lane*.

        Asserts no different owner is already set (mirrors ``set_halt_owner``
        single-owner invariant).
        """
        current = self._lane_halt_owner[lane]
        assert current is None or current == esc_id, (
            f'lane {lane!r} halt owner already set to {current!r}, '
            f'refusing to overwrite with {esc_id!r}'
        )
        self._lane_halt_owner[lane] = esc_id

    def lane_owned_by(self, esc_id: str) -> str | None:
        """Return the lane owned by *esc_id*, or None."""
        for lane, owner in self._lane_halt_owner.items():
            if owner == esc_id:
                return lane
        return None

    def unhalt_lanes_owned_by(self, esc_id: str, reason: str | None = None) -> list[str]:
        """Un-halt every lane owned by *esc_id*.  Returns list of lanes resumed."""
        resumed = []
        for lane in MERGE_LANES:
            if self._lane_halt_owner.get(lane) == esc_id:
                # Call lane halt setter directly to avoid double-signal; signal once after.
                self._lane_halt[lane].set()
                self._lane_halt_owner[lane] = None
                logger.info('Merge queue lane %r un-halted (owner %r resolved)', lane, esc_id)
                resumed.append(lane)
        if resumed:
            self._signal_resume()
        return resumed

    def _signal_resume(self) -> None:
        """Set the resume signal if the concrete worker has one (SpeculativeMergeWorker).

        The mixin is shared with MergeWorker which has no _resume_signal; the
        hasattr guard makes the call a no-op on MergeWorker.
        """
        sig = getattr(self, '_resume_signal', None)
        if sig is not None:
            sig.set()

    # ── legacy all-lanes shims (backward-compatible) ──────────────────────

    def halt_for_wip(self, reason: str) -> None:
        """Halt the merge queue due to a WIP conflict (all lanes)."""
        logger.warning('Merge queue halted for WIP: %s', reason)
        for lane in MERGE_LANES:
            self._lane_halt[lane].clear()
            self._lane_halt_owner[lane] = None

    def operator_halt(self, reason: str) -> None:
        """Operator-initiated halt: stop the merger AND abort the in-flight verify.

        Backs the ``halt_merge_queue`` escalation tool.  Unlike
        :meth:`halt_for_wip` (the automatic WIP-conflict halt, which intentionally
        lets the verifier keep draining the items behind it), this ALSO raises the
        dedicated ``_operator_halt`` signal.  The verifier's abort-poll loop and
        verifier-loop top check that signal: the in-flight verify is cancelled
        (killing its subprocess via the existing CancelledError seam) and its merge
        is re-queued so it re-verifies after un-halt — the waiter's future is left
        pending and per-task retry counters are untouched (a halt is not a failure).

        Halts every lane with no owner (like ``halt_for_wip``), so
        ``is_wip_halted`` reports True and ``halt_owner_esc_id`` is None; the
        existing ``unhalt_merge_queue`` / ``force_unhalt_merge_queue`` path
        (→ ``unhalt_all_lanes``, which clears ``_operator_halt``) cleanly reverses
        it without tripping the active-owner refusal.  Synchronous: the verify
        abort happens asynchronously in the verifier within one
        ``VERIFY_ABANDON_POLL_SECS`` interval.
        """
        logger.warning('Merge queue operator-halted: %s', reason)
        self._operator_halt.set()
        for lane in MERGE_LANES:
            self._lane_halt[lane].clear()
            self._lane_halt_owner[lane] = None

    def set_halt_owner(self, esc_id: str) -> None:
        """Register the escalation that owns the current halt.

        The workflow calls this right after submitting its halt-triggering
        escalation.  Sets the owner on all currently-halted lanes.
        Asserts no owner is already set — a double-register indicates a
        double-halt bug that should fail loudly.
        """
        current = self._halt_owner_esc_id
        assert current is None, (
            f'halt owner already set to {current!r}, '
            f'refusing to overwrite with {esc_id!r}'
        )
        halted = [ln for ln in MERGE_LANES if not self._lane_halt[ln].is_set()]
        assert halted, (
            f'set_halt_owner({esc_id!r}) called when no lane is halted — '
            'halt must be active before registering an owner'
        )
        for lane in halted:
            self._lane_halt_owner[lane] = esc_id

    def is_halt_owner(self, esc_id: str) -> bool:
        """True iff esc_id owns any currently-halted lane."""
        for lane in MERGE_LANES:
            if (not self._lane_halt[lane].is_set()
                    and self._lane_halt_owner.get(lane) == esc_id):
                return True
        return False

    def unhalt_all_lanes(self, reason: str | None = None) -> None:
        """Resume ALL lanes unconditionally, clearing every owner.

        This is the global 'resume-all' backstop — e.g. operator
        ``force_unhalt_merge_queue`` or a signal that all active halts should
        be cleared regardless of which escalation owned them.
        """
        logger.info(
            'Merge queue: all lanes un-halted%s',
            f' ({reason})' if reason else '',
        )
        for lane in MERGE_LANES:
            self._lane_halt[lane].set()
            self._lane_halt_owner[lane] = None
        # Also clear any operator halt — this global resume-all backstop (reached
        # via unhalt_wip ← force_unhalt_merge_queue ← unhalt_merge_queue) is the
        # reversal path for operator_halt().
        self._operator_halt.clear()
        self._signal_resume()

    def unhalt_wip(self, reason: str | None = None) -> None:
        """Resume the merge queue after WIP conflict resolution (all lanes).

        Delegates to :meth:`unhalt_all_lanes` so that the operator
        ``force_unhalt_merge_queue`` path (which calls this) also clears any
        orphaned per-lane halt.
        """
        self.unhalt_all_lanes(reason=reason)

    @property
    def is_wip_halted(self) -> bool:
        """True iff at least one lane is halted."""
        return any(not self._lane_halt[ln].is_set() for ln in MERGE_LANES)

    @property
    def _halt_owner_esc_id(self) -> str | None:
        """Owner of any currently-halted lane (for backward compat attr access)."""
        for lane in MERGE_LANES:
            if not self._lane_halt[lane].is_set():
                owner = self._lane_halt_owner.get(lane)
                if owner is not None:
                    return owner
        return None

    @property
    def halt_owner_esc_id(self) -> str | None:
        """Read-only public view of the current halt-owner escalation id."""
        return self._halt_owner_esc_id

    # ── wait helper ───────────────────────────────────────────────────────

    async def _wait_until_any_lane_runnable(self) -> None:
        """Block until at least one lane is not halted.

        Returns immediately if any lane is already running.  When all lanes
        are halted, waits for the first lane event to be set (un-halted) and
        cancels the remaining waiters before returning.
        """
        if any(self._lane_halt[ln].is_set() for ln in MERGE_LANES):
            return
        tasks = [asyncio.ensure_future(self._lane_halt[ln].wait()) for ln in MERGE_LANES]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()

    # ── misc ──────────────────────────────────────────────────────────────

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

        # 4. Verify — task-1724: unconditional, skip_verify path removed
        merge_wt = merge_result.merge_worktree
        assert merge_wt is not None
        # ── Persistent warm merge-verify worktree swap (PRD §10 κ) ──────
        # Parity with SpeculativeMergeWorker._verify_and_advance: increment the
        # per-worker verify counter and compute the safety-valve predicate so
        # every Nth verifying attempt runs a from-scratch cold verify in a
        # throwaway worktree (PRD §10 invariant 6).
        # merge_result.merge_commit is non-None (asserted at line 4044 above).
        self._verify_attempt_count += 1
        _due = _safety_valve_due(
            self._verify_attempt_count,
            req.config.git.persistent_merge_worktree_safety_valve_every_n,
        )
        merge_wt = await _acquire_warm_verify_worktree(
            self._git_ops, req, merge_wt,
            merge_result.merge_commit,  # non-None; assert at 4044 above
            safety_valve_due=_due,
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

    Speculation depth is configurable via ``speculation_depth`` (K = number of
    verify runners, default ``_MERGE_AHEAD_BOUND`` = 1).  One K value sizes
    both ``_speculation_slot`` and ``_merge_ahead_cap`` so the two caps remain
    in sync as runner count grows.  The default (K=1) reproduces the original
    depth-1 behaviour byte-identically.
    """

    MAX_CAS_RETRIES = 5
    # Mirror of MergeWorker.MAX_POST_MERGE_VERIFY_TIMEOUTS — see that class
    # for rationale.  Kept as a class attribute so tests can monkeypatch
    # per-class if the two workers ever diverge.
    MAX_POST_MERGE_VERIFY_TIMEOUTS: int = 2
    # Mirror of MergeWorker.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES.
    MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES: int = 1
    # Poll interval (seconds) used in the _verify_and_advance abort-loop that
    # checks whether a sole-waiter detach() cancelled req.result mid-verify.
    # Default ~10 s is negligible over a 10-40 min verify; kept as a class
    # attribute so tests can monkeypatch (e.g. worker.VERIFY_ABANDON_POLL_SECS
    # = 0.01) for fast, deterministic abort-path coverage.  Mirrors the MAX_*
    # monkeypatch convention above.
    VERIFY_ABANDON_POLL_SECS: float = 10.0

    def __init__(
        self,
        git_ops: GitOps,
        queue: asyncio.Queue[MergeRequest],
        event_store: EventStore | None = None,
        on_merge_landed: Callable[[str, str, str], Awaitable[object]] | None = None,
        speculation_depth: int = _MERGE_AHEAD_BOUND,
        escalation_queue: Any = None,
        train_callback_factory: TrainCallbackFactory | None = None,
        merge_ready_predicate: MergeReadyPredicate | None = None,
    ):
        self._git_ops = git_ops
        self._queue = queue
        self._event_store = event_store
        # Post-merge notification hook — called with (task_id, base_sha,
        # advanced_sha) after each 'done' merge.  Wrapped in try/except so a
        # coordinator bug never blocks or fails the merge.  See task 1592.
        self._on_merge_landed = on_merge_landed
        # K = number of verify runners; sizes both caps so speculation depth
        # and merger-ahead bound track runner count as a single knob (PRD D4).
        # Default = _MERGE_AHEAD_BOUND (1) → byte-identical to prior behaviour.
        self._speculation_depth: int = speculation_depth
        # PRD §10 invariant 6(b): born-at-L2 shadow compare escalation queue.
        # None-safe so bare-worker/bare-harness tests stay green without wiring.
        self._escalation_queue: Any = escalation_queue
        # Opaque factory for building per-train GroupMergeRequest callbacks.
        # Built by harness.build_train_callback_factory(self.scheduler) and
        # injected here so task γ can construct GroupMergeRequests without the
        # worker importing the scheduler (pure-git-engine layering preserved).
        # The worker itself does NOT call this factory in this task; γ uses it.
        self._train_callback_factory: TrainCallbackFactory | None = train_callback_factory
        # δ/1720 merge-ready confidence gate: injectable predicate for excluding
        # known-risky candidates from coalescing.  None → use the built-in
        # _default_coalesce_exclusion_reason (event-store history + one-strike).
        # The injectable seam keeps the worker a pure git engine while letting
        # the harness thread richer closures (flakiness counters, dry-run
        # proposals) later without re-opening the worker layering.
        self._merge_ready_predicate: MergeReadyPredicate | None = merge_ready_predicate
        # Tracks in-flight shadow compare asyncio.Tasks (single-in-flight guard).
        self._shadow_compare_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]
        # Persisted cadence state path — under project_root/data/orchestrator/
        # so it survives orchestrator restarts and lives next to other data files.
        # None-safe (mirrors escalation_queue None-safety above) so bare-worker/
        # bare-harness tests stay green without wiring project_root onto the mock.
        # Production GitOps always has project_root, so the Path branch is always
        # taken in production.
        _root = getattr(git_ops, 'project_root', None)
        self._shadow_state_path: Path | None = (
            _root / 'data' / 'orchestrator' / 'warm_verify_shadow.json'
        ) if _root is not None else None
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
        # Speculation-depth cap: one permit consumed by the Merger when it
        # prefetches a speculative item; released by the Verifier when it drains
        # that speculative item.  Symmetric accounting: acquire=prefetch,
        # release=drain so in-flight speculations are bounded at
        # _speculation_depth (K) at all times.  Plain Semaphore (not Bounded)
        # so stop() may over-release without raising.
        self._speculation_slot = asyncio.Semaphore(self._speculation_depth)
        # Merger-ahead cap (Mechanism 1, task 1646): limits non-speculative
        # build-ahead to speculation_depth items in the verifier queue.
        # Plain Semaphore (not BoundedSemaphore) so stop() may over-release
        # without raising.  Released ON-DRAIN (right after verifier_queue.get()
        # for a counted item) so the slot is free while verify runs.
        self._merge_ahead_cap = asyncio.Semaphore(self._speculation_depth)
        # Per-lane halt: each event is set (running) by default
        self._lane_halt = {ln: asyncio.Event() for ln in MERGE_LANES}
        for ln in MERGE_LANES:
            self._lane_halt[ln].set()
        self._lane_halt_owner: dict[str, str | None] = {ln: None for ln in MERGE_LANES}
        # Operator-halt signal (see _WipHaltMixin.operator_halt). Initially
        # clear = no operator halt. set() by operator_halt() to ALSO abort the
        # in-flight verify and drain the verifier pipeline, re-queuing affected
        # merges; cleared by unhalt_all_lanes(). Distinct from the per-lane halt
        # state so the automatic WIP-halt path (halt_for_wip) never aborts a verify.
        self._operator_halt = asyncio.Event()
        # Cross-workflow auto-heal attempt counter (mirrors MergeWorker; shared
        # via self.merge_worker on TaskWorkflow instances).
        self.auto_heal_registry: MainHealthAutoHealRegistry = MainHealthAutoHealRegistry()
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
        # Persistent warm merge-verify worktree: counts verifying attempts so
        # _safety_valve_due can fire the periodic cold-verify (PRD §10 invariant 6).
        # Only incremented when not item.skip_verify; never reset so the counter
        # covers the full worker lifetime (cross-submission).
        self._verify_attempt_count: int = 0
        # Lever C drift detective: land counter and in-memory quarantine set.
        # _drift_land_count increments on every 'done' land; _runner_quarantine
        # is the set of remote runner names that have been quarantined by
        # DriftDetector.check in _run_drift_check.  Both are additive across
        # submissions; the dedup'd L1 escalation in DriftDetector is the
        # durable cross-restart protection.  None-safe: initialised here so
        # bare-worker tests that don't set up remotes stay green.
        #
        # RESTART RE-TRUST WINDOW: _runner_quarantine resets to empty on worker
        # restart.  A remote that diverged (remote PASS / local FAIL) becomes
        # re-eligible immediately after restart.  The dedup'd L1 blocking
        # escalation (submitted by DriftDetector before quarantine) is the
        # cross-restart guard: an open verify_drift_divergence L1 escalation
        # halts the merge queue across restart via the normal EscalationQueue
        # gate, so the re-trusted remote's verdict is never authoritative for
        # real merges until the escalation is resolved.  Rely on that coupling
        # rather than persisting the quarantine set (operational simplicity
        # tradeoff accepted; see plan.json §Mechanism-4 design notes).
        self._drift_land_count: int = 0
        self._runner_quarantine: set[str] = set()
        # In-flight drift-detective asyncio.Tasks.  asyncio keeps only a WEAK
        # reference to running tasks, so without a strong ref here the drift
        # detective can be GC'd mid-run and a remote-PASS / local-FAIL
        # divergence would go undetected — defeating the safety control.
        # Mirrors the _shadow_compare_tasks pattern (see :func:`_maybe_schedule_shadow_compare`).
        self._drift_check_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]
        # Can be overridden in tests for fast shutdown (see stop()).
        self._shutdown_timeout: float = 5.0
        # Heartbeat: wall-clock time of last emission; initialised to 0.0 so the
        # first emission fires within one poll period after startup when depth > 0
        # (the very-large now - 0.0 gap always exceeds _heartbeat_interval_s).
        self._last_heartbeat_at: float = 0.0
        # Default interval ~5 min; override in tests for deterministic rate-limit checks.
        # Mirrors the _shutdown_timeout override precedent.
        self._heartbeat_interval_s: float = 300.0
        # Per-lane FIFO buffers — items are drained from _queue into these so
        # pick-order can prefer high over normal.  Each deque preserves FIFO
        # within a lane.  Accessed only from the merger coroutine.
        self._lane_buffers: dict[str, collections.deque[MergeRequest]] = {
            ln: collections.deque() for ln in MERGE_LANES
        }
        # Resume signal: set by every unhalt method so a blocked merger
        # (waiting with no pickable item) wakes up to re-check lanes.
        # Cleared by the merger before each wait; never cancelled.
        self._resume_signal = asyncio.Event()
        # Persistent queue.get() future — kept alive across iterations to
        # avoid the lost-item hazard of cancelling an in-flight get().
        # The merger always has at most one of these outstanding.
        self._pending_get: asyncio.Task | None = None
        # Set True when the shutdown sentinel (None) has been dequeued so
        # _acquire_next_request can drain remaining lane-buffer items before
        # returning None.  Cleared by stop() on full reset.
        self._shutdown_signaled: bool = False
        # γ/1719 debounce signature — frozenset of candidate request_ids from
        # the last _maybe_coalesce_waiting_singles attempt (incl. no-viable-train
        # attempts).  Short-circuits when the waiting set is unchanged.
        self._last_coalesce_signature: frozenset[str] | None = None
        # δ/1720 one-strike registry — task_ids of members whose coalesce-formed
        # train derailed (MergeOutcome('blocked') on a train with train_id
        # startswith _COALESCE_TRAIN_ID_PREFIX).  Keyed by task_id so the marker
        # survives re-dispatch as a new MergeRequest with the same task_id.
        # Process-lifetime (cleared only on worker restart); injectable predicate
        # is the seam for richer decay/flakiness policies later.
        self._coalesce_derailed_task_ids: set[str] = set()
        # α liveness ledger: ephemeral _merge-<uuid> worktrees owned by THIS
        # SpeculativeMergeWorker instance.  Touched every _heartbeat_loop tick
        # so the stale-worktree reaper (coalesce_or_enqueue_merge_request) never
        # reaps a live-owner worktree under Lever C prefer-remote.
        #
        # Scope on current main:
        #   (a) PERSISTENT_MERGE_WORKTREE_NAME ('_merge-verify') is reset-in-place
        #       per verify and already exempt from prune/find_inflight
        #       (git_ops.py:2075) — guard in _register_owned_merge_worktree keeps
        #       it out.
        #   (b) Serial MergeWorker (:4062) holds ≤1 worktree whose build activity
        #       refreshes mtime — out of ledger scope.
        #   (c) Cold-shadow (_run_cold_shadow_verify :7670) and drift-check
        #       (_run_drift_check :7912) _merge-* creators are short-lived local
        #       executions — out of ledger scope.
        #   (d) reverify_member_solo's _solo-* worktrees (git_ops.materialize_
        #       member_solo) use a different prefix the reaper never scans
        #       (git_ops.py:2069) — out of ledger scope.
        #   (e) Coalesced GroupMergeRequest merge worktrees ARE in scope;
        #       registered automatically at _merger_loop handoff (:5703).
        self._owned_merge_worktrees: set[Path] = set()
        # One-warning-per-path set: added when a non-ENOENT OSError is logged
        # for a path; cleared on next successful touch or on ENOENT-drop so
        # each new failure episode emits exactly one WARNING.
        self._merge_wt_touch_warned: set[Path] = set()

    # ── owned-worktree liveness ledger ───────────────────────────────────

    def _register_owned_merge_worktree(self, wt: Path | None) -> None:
        """Add *wt* to the liveness ledger.

        No-ops for None and for the persistent warm worktree
        (PERSISTENT_MERGE_WORKTREE_NAME = '_merge-verify'): the persistent
        worktree is reset-in-place and already exempt from reaper scans
        (git_ops.py:2075), so touching it would be meaningless.
        The guard is also defence-in-depth against accidental registration.
        """
        if wt is None or wt.name == PERSISTENT_MERGE_WORKTREE_NAME:
            return
        self._owned_merge_worktrees.add(wt)

    def _deregister_owned_merge_worktree(self, wt: Path | None) -> None:
        """Remove *wt* from both ledger sets (idempotent)."""
        if wt is None:
            return
        self._owned_merge_worktrees.discard(wt)
        self._merge_wt_touch_warned.discard(wt)

    async def _cleanup_owned_merge_worktree(self, wt: Path | None) -> None:
        """Deregister *wt* then clean it up from disk.

        Deregister-before-cleanup ensures a failed git cleanup cannot
        immortalise a ledger entry (PRD §3 / design decision).
        """
        self._deregister_owned_merge_worktree(wt)
        if wt is not None:
            await self._git_ops.cleanup_merge_worktree(wt)

    def _touch_owned_merge_worktrees(self) -> int:
        """Touch (os.utime) every ledger path to refresh its mtime.

        Called unconditionally every _heartbeat_loop tick so a live owner's
        worktrees never age past ~1 poll period (~30 s; 360× inside the 10800 s
        liveness window).  A dead owner stops calling this; its worktrees age
        and are reaped exactly as before.

        Mirrors the sync, clock-injectable pattern of _maybe_log_queue_heartbeat
        so unit tests can drive it directly without running the async loop.

        Returns:
            Number of worktrees successfully touched this tick.
        """
        touched = 0
        for p in list(self._owned_merge_worktrees):
            try:
                os.utime(p, None)
            except FileNotFoundError:
                self._deregister_owned_merge_worktree(p)
                logger.info(
                    'owned merge worktree %s gone (ENOENT) — dropped from liveness ledger', p
                )
                continue
            except OSError as exc:
                if p not in self._merge_wt_touch_warned:
                    logger.warning(
                        'failed to touch owned merge worktree %s for liveness heartbeat: %s',
                        p, exc,
                    )
                    self._merge_wt_touch_warned.add(p)
                continue
            self._merge_wt_touch_warned.discard(p)
            touched += 1
        if touched:
            logger.debug('touched %d owned merge worktree(s)', touched)
        return touched

    # ── lane-buffer helpers ───────────────────────────────────────────────

    def _drain_queue_into_lanes(self) -> None:
        """Non-blocking drain of _queue into per-lane buffers.

        When the shutdown sentinel (None) is encountered it sets
        ``_shutdown_signaled`` so the caller knows shutdown was requested
        after draining any items that arrived before the sentinel.
        """
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item is None:
                self._shutdown_signaled = True
                return  # sentinel consumed — all prior items already buffered
            lane = _normalize_lane(item.lane)
            self._lane_buffers[lane].append(item)

    def _pop_next_pickable(self) -> MergeRequest | None:
        """Return the next pickable request (highest-priority non-halted lane, FIFO).

        Returns None if every non-empty lane is halted, or all buffers are
        empty.  Pure/synchronous so unit tests run without an event loop.
        """
        for lane in MERGE_LANES:  # high → normal
            if self.is_lane_halted(lane):
                continue
            buf = self._lane_buffers[lane]
            if buf:
                return buf.popleft()
        return None

    async def _acquire_next_request(self) -> MergeRequest | None:
        """Return the next pickable request, blocking if nothing is available.

        Drains _queue into lane buffers, then picks the highest-priority
        non-halted item.  When nothing is pickable it waits on FIRST_COMPLETED
        of (a) the persistent _pending_get future that will fire when a new
        item arrives, and (b) the _resume_signal that fires when a lane is
        un-halted.  Only the _resume_signal.wait() task is ever cancelled;
        the queue.get() task (_pending_get) persists to avoid the lost-item
        hazard of cancelling a pending get().

        Returns None when all lane-buffer items have been returned and the
        shutdown sentinel has been seen.
        """
        while True:
            # Drain any newly arrived items into lane buffers.
            # Sets _shutdown_signaled if the sentinel is encountered.
            self._drain_queue_into_lanes()

            req = self._pop_next_pickable()
            if req is not None:
                return req

            # Lane buffers empty — check shutdown before blocking.
            if self._shutdown_signaled:
                return None

            # Nothing pickable — start (or reuse) the persistent queue getter
            # and wait for EITHER a new arrival OR a lane resume.
            if self._pending_get is None or self._pending_get.done():
                self._pending_get = asyncio.ensure_future(self._queue.get())

            self._resume_signal.clear()
            resume_task = asyncio.ensure_future(self._resume_signal.wait())
            try:
                done, _ = await asyncio.wait(
                    [self._pending_get, resume_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not resume_task.done():
                    resume_task.cancel()

            # If _pending_get finished, harvest its result into lane buffers.
            if self._pending_get in done:
                item = self._pending_get.result()
                self._pending_get = None
                if item is None:
                    self._shutdown_signaled = True
                else:
                    self._lane_buffers[_normalize_lane(item.lane)].append(item)
            # Loop to try _pop_next_pickable again (maybe the resume unblocked a lane).

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

        def _entry(
            req: MergeRequest,
            state: str,
            worktree_path=None,
            position: int = 0,
            lane: str | None = None,
        ) -> dict:
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
                'lane': lane if lane is not None else req.lane,
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
        # Items may be in the external _queue OR already drained into _lane_buffers.
        # Enumerate _lane_buffers first (priority-ordered; items here are already
        # waiting), then _queue for any not-yet-drained arrivals.
        # Same CPython internal-deque access as _queue above — read-only, no lock.
        for lane in MERGE_LANES:
            for req in list(self._lane_buffers[lane]):
                entries.append(_entry(
                    req, 'queued', worktree_path=None,
                    position=len(entries), lane=lane,
                ))
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
        """Periodically emit merge-queue depth heartbeats and touch owned worktrees.

        Runs independently of the merger and verifier loops so it continues to
        fire even when those are blocked on ``queue.get()`` or semaphores (the
        exact silence window that caused the 2026-06-04 dead-slot misdiagnosis).

        Wakes every ``_HEARTBEAT_POLL_S`` seconds and:
          1. Touches every owned ephemeral _merge-* worktree in the liveness
             ledger (α mechanism) so the stale-worktree reaper never reaps a
             live-owner worktree.  Touch runs FIRST, unconditionally, and
             swallows per-path errors internally (ENOENT/OSError) so it can
             never starve the heartbeat log.
          2. Delegates the fire/rate-limit/format/emit decision to the
             synchronous, clock-injectable :meth:`_maybe_log_queue_heartbeat`.

        Any unexpected exception from either call is logged and swallowed so a
        heartbeat bug can never crash the worker.
        """
        while self._running:
            await asyncio.sleep(_HEARTBEAT_POLL_S)
            try:
                self._touch_owned_merge_worktrees()
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
        # Release speculation-depth permits, all lane halts, and merge-ahead cap
        # so the merger doesn't hang waiting at any synchronisation point.
        # Over-releasing a plain Semaphore is safe (it just increments the counter).
        for _ in range(self._speculation_depth + 1):
            self._speculation_slot.release()
        for ln in MERGE_LANES:
            self._lane_halt[ln].set()
        for _ in range(self._speculation_depth + 1):
            self._merge_ahead_cap.release()

        # Drain per-lane buffers (items already removed from _queue by the merger)
        for lane in MERGE_LANES:
            while self._lane_buffers[lane]:
                req = self._lane_buffers[lane].popleft()
                if not req.result.done():
                    req.result.set_result(shutdown)

        # Drain main queue (items not yet drained into lane buffers)
        while not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                if req is not None and not req.result.done():
                    req.result.set_result(shutdown)
            except asyncio.QueueEmpty:
                break

        # Cancel the persistent _pending_get future if live (it holds a
        # queue.get() that will never fire because we're shutting down).
        if self._pending_get is not None and not self._pending_get.done():
            self._pending_get.cancel()
            self._pending_get = None

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
                            await self._cleanup_owned_merge_worktree(item.merge_wt)
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
                            await self._cleanup_owned_merge_worktree(item.merge_wt)
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

        # Cancel in-flight drift-check detective tasks so their finally blocks run
        # (i.e. _run_drift_check's cleanup_merge_worktree call executes during
        # CancelledError unwinding rather than leaking the throwaway verify worktree).
        # Take a snapshot before iterating: the done-callback mutates the set.
        for _dt in list(self._drift_check_tasks):
            if not _dt.done():
                _dt.cancel()
                with contextlib.suppress(BaseException):
                    await _dt

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
    # Retroactive coalescing pass (γ/1719)
    # ------------------------------------------------------------------

    def _mark_coalesce_derailed(self, member_task_ids: list[str]) -> None:
        """Record the members of a derailed coalesce-formed train as one-strike.

        Called by _merger_loop after _do_train_merge returns a blocked outcome
        for a train whose train_id startswith _COALESCE_TRAIN_ID_PREFIX.  Adds
        each task_id to self._coalesce_derailed_task_ids so the next coalescing
        pass (default predicate) excludes them from train formation.
        """
        if not member_task_ids:
            return
        self._coalesce_derailed_task_ids.update(member_task_ids)
        logger.info(
            'Coalesce one-strike: marked %d task(s) after coalesce-train derail: %s',
            len(member_task_ids), member_task_ids,
        )

    def _default_coalesce_exclusion_reason(self, req: MergeRequest) -> str | None:
        """Built-in merge-ready predicate (δ/1720 confidence gate).

        Returns an exclusion REASON string (truthy → exclude from coalescing) or
        None (eligible).  Signals implemented here use only worker-reachable
        substrate (no scheduler import):

          1. One-strike registry (cheapest, in-memory first):
             If req.task_id is in self._coalesce_derailed_task_ids (a prior
             coalesce-formed train that included this task derailed), exclude.
             Filled in step-6; always empty until then.

          2. Event-store blocked history:
             If the branch's most-recent terminal merge outcome was 'blocked' or
             'error', exclude.  Filled in step-4.
        """
        # Signal 1: one-strike registry (cheapest, in-memory — check first).
        if req.task_id in self._coalesce_derailed_task_ids:
            return 'coalesce_derailed_one_strike'
        # Signal 2: event-store blocked history.
        if self._event_store is not None:
            rec = self._event_store.latest_merge_finalized(branch=req.branch)
            if rec is not None and rec.get('state') in _COALESCE_RISKY_TERMINAL_STATES:
                return f'recent_terminal_{rec["state"]}'
        return None

    async def _maybe_coalesce_waiting_singles(self) -> bool:
        """Attempt to coalesce waiting single MergeRequests into one GroupMergeRequest.

        Called from _merger_loop at the pre-dequeue point when the pipeline is
        idle (spec_base is None and prefetched is None) so a train is never
        enqueued behind an unverified speculative merge commit (pipeline-ordering
        contract, :5239 warning comment).

        Returns True when a GroupMergeRequest was formed and appended to
        _lane_buffers['normal']; False in all no-op / guard-exit cases.

        Config is read from the first candidate MergeRequest's .config field
        (OrchestratorConfig), mirroring the pattern in _do_train_merge (req.config).
        """
        # Guard: factory required to build callbacks.
        if self._train_callback_factory is None:
            return False

        # Drain any newly arrived items so the candidate list is current.
        self._drain_queue_into_lanes()

        # Build candidate list: single MergeRequests in the normal lane whose
        # futures are still live (not done and not cancelled).
        #
        # STRUCTURAL EXCLUSIONS (no new MergeRequest field needed):
        #  · GroupMergeRequest: a pre-existing or re-formed train — excluded by
        #    isinstance check; it stays in the buffer untouched and its future is
        #    never resolved here (idempotency guarantee).
        #  · req.result.done() / req.result.cancelled(): an absorbed request has
        #    its future resolved ('superseded') — excluded; a detached/cancelled
        #    waiter has its future cancelled — excluded AND never receives
        #    set_result (so it stays cancelled, not overwritten with 'superseded').
        #  · In-flight / verifying request: lives in self._inflight_req or is
        #    carried by a SpeculativeItem in self._verify_item — structurally
        #    absent from self._lane_buffers, so it is excluded without any
        #    explicit filter.  No buffer scan of _inflight_req/_verify_item is
        #    required or performed.
        candidates = [
            req for req in self._lane_buffers['normal']
            if not isinstance(req, GroupMergeRequest)
            and not req.result.done()
            and not req.result.cancelled()
        ]

        # Guard: need at least 2 candidates to form a train.
        if len(candidates) < 2:
            return False

        # Read OrchestratorConfig from the first candidate (all requests in the
        # queue share the same config since the worker is per-project).
        orch_config = candidates[0].config

        # Guard: feature knob (OFF by default — fold-the-decision norm).
        if not orch_config.merge_train_coalesce_enabled:
            return False
        # Guard: max_members ≥ 2 (the ge=2 Pydantic constraint should catch this,
        # but guard defensively so the invariant is local to this method).
        if orch_config.merge_train_max_members < 2:
            return False

        # ── Step-12: debounce ──────────────────────────────────────────────────
        # Compute the candidate-set signature.  Short-circuit on an unchanged set
        # so a steady stream of waiting singles does not re-run get_changed_line_ranges
        # + stack_train_branches every merger tick.  Setting the signature BEFORE the
        # selection/stack work means both the no-viable-train AND the successful-coalesce
        # paths record the attempt — a composition change re-arms (new request_id in set
        # → different frozenset → inequality → re-runs).
        sig: frozenset[str] = frozenset(c.request_id for c in candidates)
        if sig == self._last_coalesce_signature:
            return False
        self._last_coalesce_signature = sig

        # ── δ/1720: merge-ready exclusion filter ──────────────────────────────
        # Split structural candidates into eligible + exclusions.  The predicate
        # returns a REASON string (truthy → exclude) or None (eligible).
        # Run AFTER recording the debounce signature (keyed on the structural
        # waiting set) so risk signals — which are monotonic within a process —
        # only re-evaluate when the structural set changes, avoiding per-tick
        # event-store reads.
        predicate = self._merge_ready_predicate or self._default_coalesce_exclusion_reason
        eligible: list[MergeRequest] = []
        exclusions: list[dict[str, str]] = []
        for _c in candidates:
            try:
                _reason = predicate(_c)
            except Exception:  # noqa: BLE001
                # An injected predicate raised — degrade gracefully: log the
                # error and treat the candidate as eligible so a buggy closure
                # does not kill the merger loop.  The default predicate is
                # fire-safe (latest_merge_finalized swallows its own errors).
                logger.exception(
                    'Coalesce predicate raised for request_id=%s task_id=%s '
                    'branch=%s; treating as eligible (safe-degrade)',
                    _c.request_id, _c.task_id, _c.branch,
                )
                _reason = None
            if _reason:
                exclusions.append({'request_id': _c.request_id, 'reason': _reason})
                logger.info(
                    'Coalesce exclusion: request_id=%s task_id=%s branch=%s reason=%s',
                    _c.request_id, _c.task_id, _c.branch, _reason,
                )
            else:
                eligible.append(_c)
        candidates = eligible
        if len(candidates) < 2:
            # The exclusion gate reduced eligible candidates below the 2-member
            # minimum needed to form a train.  Log so the decision is auditable
            # even though no event is emitted (a train_coalesced event only fires
            # when a train actually forms).  The excluded tasks will merge solo.
            if exclusions:
                logger.info(
                    'Coalesce near-train suppressed: exclusion gate left only %d '
                    'eligible candidate(s) (need ≥2); tasks will merge solo. '
                    'exclusions=%r',
                    len(candidates),
                    exclusions,
                )
            return False

        # ── Step-6: core pass body ─────────────────────────────────────────────
        # SELECTION: fan out line-range fetches concurrently (one git subprocess
        # per candidate), then delegate to the greedy mutually-stackable selector.
        # Function-local import avoids a load-time circular import: workflow.py
        # imports merge_queue only lazily, so merge_queue→workflow is safe here.
        from orchestrator.workflow import _select_train_members  # noqa: PLC0415

        anchor = candidates[0]
        other_ids = [c.task_id for c in candidates[1:]]
        all_ids = [c.task_id for c in candidates]

        range_results = await asyncio.gather(*[
            self._git_ops.get_changed_line_ranges(task_id) for task_id in all_ids
        ])
        ranges_by_id: dict[str, dict] = {
            tid: r for tid, r in zip(all_ids, range_results, strict=True)
        }

        selected = _select_train_members(
            anchor.task_id, other_ids, ranges_by_id,
            orch_config.merge_train_max_members,
        )
        if len(selected) < 2:
            # Signature already recorded above; no further work needed.
            return False

        # STACK: rebase successors onto the previous survivor's branch.  Rebase
        # conflicts or missing worktrees produce ejected members that stay solo.
        stack_result = await self._git_ops.stack_train_branches(selected)
        survivors: list[str] = stack_result.survivors
        ejected: list[str] = stack_result.ejected

        if len(survivors) < 2:
            # Abort cleanly: sig already recorded; no buffer mutation, no future
            # resolution, no event emitted.  Non-selected and ejected members
            # remain in _lane_buffers with unresolved futures (they may yet join a
            # future train when a new stackable partner arrives and re-arms the sig).
            return False

        # BUILD GroupMergeRequest from the tip (last survivor) request.
        req_by_task_id: dict[str, MergeRequest] = {c.task_id: c for c in candidates}
        tip_id = survivors[-1]
        tip_req = req_by_task_id[tip_id]
        survivor_reqs = [req_by_task_id[sid] for sid in survivors]

        # Union scope: if workspace-wide verify, use tip's scope (no per-module
        # constraint); otherwise union task_files + module_configs across survivors.
        if orch_config.merge_verify_workspace:
            union_task_files = tip_req.task_files
            union_module_configs = tip_req.module_configs
        else:
            # task_files=None means "all files"; any None in the set wins.
            if any(r.task_files is None for r in survivor_reqs):
                union_task_files = None
            else:
                _seen_files: set[str] = set()
                union_task_files = []
                for sr in survivor_reqs:
                    for f in (sr.task_files or []):
                        if f not in _seen_files:
                            _seen_files.add(f)
                            union_task_files.append(f)
            # module_configs: deduplicate by mc.prefix (semantic identity),
            # matching workflow._union_train_scope.  Using id(mc) would retain
            # duplicate configs for equal-but-distinct objects and produce
            # redundant verify-scope entries.
            union_module_configs = []
            _seen_mc: set[str] = set()
            for sr in survivor_reqs:
                for mc in sr.module_configs:
                    if mc.prefix not in _seen_mc:
                        _seen_mc.add(mc.prefix)
                        union_module_configs.append(mc)

        train_id = f'{_COALESCE_TRAIN_ID_PREFIX}{tip_id}-{uuid.uuid4().hex[:8]}'
        callbacks = self._train_callback_factory(train_id)  # type: ignore[misc]
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()

        group_req = GroupMergeRequest(
            task_id=tip_req.task_id,
            branch=tip_req.branch,
            worktree=tip_req.worktree,
            pre_rebased=False,
            task_files=union_task_files,
            module_configs=union_module_configs,
            config=tip_req.config,
            result=future,
            train_id=train_id,
            member_task_ids=list(survivors),
            tip_branch=tip_req.branch,
            tip_task_id=tip_id,
            status_check=callbacks.status_check,
            mark_member_done=callbacks.mark_member_done,
        )

        # QUEUE SURGERY: rebuild _lane_buffers['normal'] preserving FIFO order for
        # non-survivor + ejected solos, then append the new train at the tail.
        survivor_set = set(survivors)
        new_buffer: collections.deque[MergeRequest] = collections.deque()
        for buf_req in self._lane_buffers['normal']:
            # Keep GroupMergeRequests and non-survivor singles intact.
            if isinstance(buf_req, GroupMergeRequest) or buf_req.task_id not in survivor_set:
                new_buffer.append(buf_req)
        new_buffer.append(group_req)
        self._lane_buffers['normal'] = new_buffer

        # RESOLVE absorbed futures: park each absorbed workflow as merge-deferred.
        # The existing workflow._handle_superseded consumer (α/1717) transitions
        # the task; mark_member_done flips it done after the train lands.
        # Detached/cancelled requests were filtered from candidates in step 1, so
        # no set_result is ever called on a cancelled future here.
        for s_req in survivor_reqs:
            if not s_req.result.done():
                s_req.result.set_result(
                    MergeOutcome('superseded', superseded_by=train_id)
                )

        # EMIT train_coalesced lifecycle event.
        _emit_train_event(
            self._event_store,
            EventType.train_coalesced,
            task_id=tip_id,
            train_id=train_id,
            member_task_ids=list(survivors),
            data={
                'absorbed_request_ids': [r.request_id for r in survivor_reqs],
                'tip_task_id': tip_id,
                'ejected': ejected,
                'size': len(survivors),
                'exclusions': exclusions,
            },
        )

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
        # True exactly while the merger holds one _speculation_slot permit that
        # has NOT yet been handed off to the verifier via _verifier_queue.put().
        # Set True immediately after _speculation_slot.acquire() (:5003); set
        # False after every _verifier_queue.put() hand-off (the verifier then
        # owns the permit and releases it on drain), and on the no-pickable /
        # shutdown release branches.  Every exit path that bypasses the verifier
        # (abandoned-at-top, except WorktreeMissing, except Exception, outer
        # finally) releases the permit and clears this flag.
        held_spec_permit: bool = False

        try:
            while self._running:
                # γ/1719 retroactive coalescing pass — design decisions summary:
                # • DD1: runs at the pre-dequeue point, gated on a clean pipeline
                #   (spec_base=None and prefetched=None) so a train is never enqueued
                #   behind an unverified speculative merge commit (:5239 warning).
                # • DD2: idempotency via candidate filter (excludes GroupMergeRequests,
                #   done/cancelled futures); no new MergeRequest field required.
                # • DD3: debounce via _last_coalesce_signature prevents re-stacking an
                #   unchanged waiting set on every tick.  NOTE: the signature is set
                #   before the stack attempt, so a transient stack/worktree error
                #   (survivors<2 due to env issue, not a deterministic rebase conflict)
                #   permanently skips the set until the candidate composition changes.
                #   Low priority given feature ships OFF; tracked for δ/ζ follow-up.
                # • DD6 (timing): absorbed members park merge-deferred asynchronously;
                #   _do_train_merge's status pre-check may return TRAIN_INCOMPLETE on a
                #   prematurely-dequeued train — same retryable 'blocked' the β/δ path
                #   uses; full retry is left to δ/ζ.  Feature is OFF by default.
                # When merge_train_coalesce_enabled=False (default) the call has
                # near-zero overhead: it drains the queue (already done at acquire
                # time), builds the candidate list, reads the knob, and returns False.
                if spec_base is None and prefetched is None:
                    await self._maybe_coalesce_waiting_singles()

                # Get next request: use pre-fetched (speculative) item if available,
                # otherwise acquire from the lane-priority pick system.
                if prefetched is not None:
                    req = prefetched
                    prefetched = None
                else:
                    req = await self._acquire_next_request()
                    if req is None:
                        break  # shutdown sentinel
                    spec_base = None  # fresh dequeue resets speculation chain

                self._inflight_req = req  # track for stop() race resolution
                # Drop-on-detection: workflow soft-cancelled before worker
                # dequeued.  Skipping merge work avoids the orphan-halt
                # window where no escalation owner is registered.
                if self._request_abandoned(req):
                    if held_spec_permit:
                        self._speculation_slot.release()
                        held_spec_permit = False
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
                        # δ/1720: if a coalesce-formed train derailed with a
                        # risky terminal outcome, mark all its members one-strike
                        # so they are excluded from the next coalescing pass and
                        # left to merge solo.  We use _COALESCE_RISKY_TERMINAL_STATES
                        # (frozenset{'blocked','error'}) to align the recording
                        # trigger with the exclusion predicate — both 'blocked'
                        # (verify failure) and 'error' (unexpected git/infra error)
                        # are deterministic enough to warrant solo-merge.
                        # 'conflict' and 'wip_halted' are intentionally excluded:
                        # a conflict may resolve after the partner task lands, and
                        # wip_halted is policy-driven rather than a task failure.
                        if (
                            outcome.status in _COALESCE_RISKY_TERMINAL_STATES
                            and req.train_id.startswith(_COALESCE_TRAIN_ID_PREFIX)
                        ):
                            self._mark_coalesce_derailed(req.member_task_ids)
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=None, merge_wt=None,
                            base_sha=actual_main, speculative=False,
                            skip_verify=False, immediate_outcome=outcome,
                            started_monotonic=t0,
                        ))
                        # Train is put with speculative=False so the verifier
                        # will NOT release the slot on drain.  Release explicitly
                        # if the train was prefetched as a speculative item.
                        if held_spec_permit:
                            self._speculation_slot.release()
                            held_spec_permit = False
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
                        held_spec_permit = False  # verifier releases if speculative
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
                        held_spec_permit = False  # verifier releases if speculative
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
                        held_spec_permit = False  # verifier releases if speculative
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
                        held_spec_permit = False  # verifier releases if speculative
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
                        held_spec_permit = False  # verifier releases if speculative
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
                        held_spec_permit = False  # verifier releases if speculative
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
                        held_spec_permit = False  # verifier releases if speculative
                        spec_base = None
                        self._inflight_req = None
                        continue

                    # Mechanism 1: cap non-speculative build-ahead.
                    # Trains (continue before this) and immediate-outcome guards
                    # (all return above) never reach this site, so `not speculative`
                    # is the exact predicate for blocking-path items.
                    counts_against_cap = not speculative
                    if counts_against_cap:
                        await self._merge_ahead_cap.acquire()
                    try:
                        self._register_owned_merge_worktree(merge_result.merge_worktree)
                        await self._verifier_queue.put(SpeculativeItem(
                            request=req, merge_result=merge_result,
                            merge_wt=merge_result.merge_worktree,
                            base_sha=base_for_merge, speculative=speculative,
                            skip_verify=False,  # task-1724: always run merge-gate verify
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
                        #
                        # The same double-release tolerance applies to
                        # _speculation_slot: if this is a speculative item and
                        # CancelledError fires after the put succeeded,
                        # held_spec_permit remains True so the outer finally
                        # releases the slot — and the verifier also releases it
                        # on drain.  Both are plain Semaphores so over-release
                        # is safe here too.
                        if counts_against_cap:
                            self._merge_ahead_cap.release()
                        raise
                    # The put succeeded — verifier now owns the speculation
                    # permit for this item (released on drain if speculative).
                    held_spec_permit = False
                    self._inflight_req = None  # item is now owned by verifier

                    # ── Speculative look-ahead (depth-K cap) ──────────────────
                    # Acquire one speculation permit before the look-ahead peek.
                    # If an item is found and prefetched, the permit stays held;
                    # the Verifier releases it when draining this speculative item.
                    # If nothing is pickable (or shutdown), the permit is released
                    # immediately — symmetric accounting keeps in-flight speculations
                    # bounded at self._speculation_depth (K).
                    await self._speculation_slot.acquire()  # depth-K cap
                    held_spec_permit = True  # permit held until verifier drains or exception releases
                    # Harvest any item already delivered to the persistent
                    # getter so the look-ahead can see it via _pop_next_pickable.
                    if self._pending_get is not None and self._pending_get.done():
                        _item = self._pending_get.result()
                        self._pending_get = None
                        if _item is None:
                            self._shutdown_signaled = True
                        else:
                            self._lane_buffers[_normalize_lane(_item.lane)].append(_item)
                    self._drain_queue_into_lanes()
                    next_req = self._pop_next_pickable()
                    if next_req is not None:
                        # Permit stays held — verifier will release on drain.
                        prefetched = next_req
                        spec_base = merge_commit  # N+1 will merge against N's commit
                        logger.debug(
                            f'Task {req.task_id}: speculative look-ahead for '
                            f'{next_req.task_id} (base={merge_commit[:8]})'
                        )
                    elif self._shutdown_signaled:
                        self._speculation_slot.release()  # return unused permit
                        held_spec_permit = False
                        break  # shutdown sentinel and nothing left to speculate
                    else:
                        self._speculation_slot.release()  # no pickable item — return permit
                        held_spec_permit = False
                        spec_base = None  # no pickable item, no speculation
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
                    # Release any speculation permit held for a prefetched item
                    # that failed before being put on the verifier queue.
                    if held_spec_permit:
                        self._speculation_slot.release()
                        held_spec_permit = False
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
                    # Release any speculation permit held for a prefetched item
                    # that failed before being put on the verifier queue.
                    if held_spec_permit:
                        self._speculation_slot.release()
                        held_spec_permit = False
                    spec_base = None
                    self._inflight_req = None
        finally:
            # Release any speculation permit still held — covers BaseException
            # paths (e.g. CancelledError) that bypass the inner except clauses.
            if held_spec_permit:
                self._speculation_slot.release()
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
                        await self._cleanup_owned_merge_worktree(item.merge_wt)
                # Treat as failed for chain-invalidation: any speculative
                # item built on this one's commit is now stale.
                n_failed = True
                if item.speculative:
                    self._speculation_slot.release()
                continue

            # Operator halt: bounce real verify candidates back to the merger
            # input queue, draining the build-ahead pipeline to empty while
            # halted.  Keyed on _operator_halt (NOT is_wip_halted) so the
            # automatic WIP-halt path (halt_for_wip) is unaffected — it
            # intentionally lets the verifier keep draining.  immediate_outcome
            # items (trains / already-decided conflict/already_merged) run no
            # verify subprocess, so they fall through and resolve normally below.
            # Mirror the abandoned drain above: clean the merge worktree, mark
            # n_failed for chain-invalidation, release the speculation slot — but
            # re-queue req (result left pending) instead of dropping it.  The
            # merger is halted, so nothing re-feeds _verifier_queue; the loop then
            # blocks on an empty get() until un-halt.
            if self._operator_halt.is_set() and item.immediate_outcome is None:
                if item.merge_wt is not None:
                    with contextlib.suppress(BaseException):
                        await self._cleanup_owned_merge_worktree(item.merge_wt)
                n_failed = True
                if item.speculative:
                    self._speculation_slot.release()
                self._queue.put_nowait(req)
                continue

            try:
                # Capture the original speculative flag BEFORE any _remerge
                # reassignment so the finally can release the slot for exactly
                # the items that consumed a permit (speculation_slot.acquire()
                # was called by the Merger for every speculative prefetch).
                # _remerge reassigns `item` to a non-speculative remapped item,
                # so reading item.speculative in finally would miss the release.
                item_was_speculative = item.speculative

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
                    # Clean up the stale merge worktree (deregister-before-cleanup).
                    if item.merge_wt:
                        await self._cleanup_owned_merge_worktree(item.merge_wt)
                    self._emit_speculative(
                        EventType.speculative_discard, req.task_id,
                        reason=remerge_reason,
                    )
                    logger.info(
                        f'Task {req.task_id}: discarding stale merge '
                        f'({remerge_reason}), re-merging against actual main'
                    )
                    # task-1724: all re-merges always verify (skip_verify removed).
                    item = await self._remerge(req, item.started_monotonic)
                    # Update _verify_item to the freshly re-merged item; phase stays
                    # 'remerging' until _verify_and_advance transitions it.
                    self._verify_item = item

                # ── Immediate outcome (already_merged / conflict / blocked) ─
                # GroupMergeRequest/train items (immediate_outcome set) always
                # reach this branch; they never enter _run_post_merge_verify, so
                # the sole-waiter mid-verify orphan window fixed in task 1681
                # does not apply to trains.  (skip_verify is retained on the
                # SpeculativeItem dataclass but is always False for single-task
                # items after task-1724; it is not honoured by _verify_and_advance.)
                # A soft-cancel on the group-merge consumer falls to the blanket
                # fut.cancel() via workflow.py:675 _await_cancellable (no
                # on_soft_cancel detach hook attached), which is the accepted
                # PRD D9 decision documented at workflow.py:6522 ('trains stay
                # on the direct path; blanket cancel untouched').  Residual:
                # accepted observability-only gap, not a wasted-verify orphan.
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
                        await self._cleanup_owned_merge_worktree(item.merge_wt)
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
                        await self._cleanup_owned_merge_worktree(item.merge_wt)
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
                # Release one speculation permit only if the ORIGINAL item
                # consumed one (i.e. it was a speculative prefetch).  Use
                # item_was_speculative (captured before any _remerge reassignment)
                # because _remerge replaces `item` with a non-speculative
                # remapped item, so `item.speculative` would be False even for
                # originally-speculative items that were re-merged.
                if item_was_speculative:
                    self._speculation_slot.release()

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
    ) -> SpeculativeItem:
        """Re-merge a request against actual main after speculation invalidation.

        task-1724: skip_verify is unconditionally False on every success path —
        the merge gate always runs before advance_main regardless of pre_rebased
        or tree-SHA equality.  The force_verify/prev_skip_verify/prev_merge_tree
        parameters and tree-equality cascade are removed.
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
                self._register_owned_merge_worktree(retry_result.merge_worktree)
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
        # task-1724: merge gate always runs — skip_verify is unconditionally False.
        self._register_owned_merge_worktree(merge_result.merge_worktree)
        return SpeculativeItem(
            request=req, merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=actual_main, speculative=False, skip_verify=False,
            started_monotonic=started_monotonic,
        )

    def _resolve_or_drop_abandoned(
        self, req: MergeRequest, outcome: MergeOutcome,
    ) -> None:
        """Resolve *req.result* with *outcome*, or drop silently if abandoned.

        Used at the three task-1681 resolution sites (disk-skip 5053,
        verify-fail 5061, advance-success 5111) to emit the canonical
        'abandoned by waiter' INFO log when detach() has already cancelled
        req.result, instead of performing a silent no-op.

        Synchronous — no ``await`` between the abandonment check and
        ``set_result``, preserving the detach resolution-race guard invariant
        documented at merge_queue.py:2105-2119.
        """
        if self._request_abandoned(req):
            return
        if not req.result.done():
            req.result.set_result(outcome)

    async def _verify_and_advance(self, item: SpeculativeItem) -> bool:
        """Run verification + CAS advance for one item.

        Returns True if the item advanced main successfully, False otherwise.
        Resolves item.request.result in all cases, except when detach() has
        already cancelled req.result (mid-verify abort or post-verify
        abandonment short-circuit) — in those cases the future is
        intentionally left cancelled; req.result.cancelled() is True on
        return.
        """
        req = item.request
        merge_wt = item.merge_wt
        assert merge_wt is not None
        assert item.merge_result is not None
        merge_commit = item.merge_result.merge_commit
        assert merge_commit is not None
        merge_commit = merge_commit.strip()

        # ── Persistent warm merge-verify worktree swap (PRD §10 κ) ──────
        # When the knob is ON and the safety valve isn't due, swap the
        # ephemeral merge_wt for the fixed _merge-verify path BEFORE the
        # verify-start log so the log shows worktree=_merge-verify (the
        # user-observable persistence signal) and verify, advance_main, and
        # all cleanup_merge_worktree calls use the warm path.
        # PRD §10 invariant 6: every Nth verifying attempt bypasses the swap
        # and runs a cold verify in the throwaway ephemeral worktree.
        # task-1724: verification is unconditional (skip_verify is never honored
        # here — the merge gate always runs before advance_main).
        self._verify_attempt_count += 1
        _due = _safety_valve_due(
            self._verify_attempt_count,
            req.config.git.persistent_merge_worktree_safety_valve_every_n,
        )
        merge_wt = await _acquire_warm_verify_worktree(
            self._git_ops, req, merge_wt, merge_commit,
            safety_valve_due=_due,
        )
        assert merge_wt is not None  # input was non-None; warm or unchanged
        # Warm-swap: if _acquire_warm_verify_worktree returned the
        # persistent _merge-verify path instead of the ephemeral
        # item.merge_wt, the helper already removed item.merge_wt from
        # disk.  Deregister it from the liveness ledger now so the
        # ghost is cleared immediately (no need to wait for the touch
        # loop's ENOENT self-heal).  If no swap occurred, merge_wt IS
        # item.merge_wt and the cleanup calls below deregister it via
        # _cleanup_owned_merge_worktree.
        if merge_wt is not item.merge_wt:
            self._deregister_owned_merge_worktree(item.merge_wt)

        # ── Step 4: verify ────────────────────────────────────────────
        # PRD §10 invariant 6(b): warm per-test results captured here for the
        # same-candidate shadow compare scheduled in the 'done' block below.
        # Initialised empty so the shadow compare scheduler sees {} if the warm
        # path was not taken (safety-valve or knob off) and short-circuits
        # without scheduling a cold leg.
        _warm_results: dict[str, bool] = {}
        self._verify_phase = 'verifying'
        self._verify_started_at = time.time()  # wall-clock verify start for triage
        logger.info(
            f'Task {req.task_id}: verify start (merge={merge_commit[:8]}, '
            f'worktree={merge_wt.name})'
        )
        # Capture the VerifyResult via on_result callback on the genuine warm
        # path (persistent_merge_worktree on, safety valve not due) to provide
        # per-test results to the shadow compare scheduler.
        _warm_capture: list[VerifyResult] = []
        _is_warm_path = (
            req.config.git.persistent_merge_worktree
            and not _due
        )
        try:
            # Wrap _run_post_merge_verify in an abort-poll loop so that a
            # sole-waiter detach() (pf.cancel() → req.result.cancelled())
            # landing mid-verify aborts the wasted compute instead of
            # burning one full 10-40 min cycle (task 1681 fix-2).
            # Poll cost: one cheap req.result.cancelled() check per interval.
            verify_task = asyncio.ensure_future(_run_post_merge_verify(
                self._git_ops, req, merge_wt,
                timeouts=self._post_merge_verify_timeouts,
                enospc_retries=self._post_merge_verify_enospc_retries,
                max_timeouts=self.MAX_POST_MERGE_VERIFY_TIMEOUTS,
                max_enospc=self.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES,
                event_store=self._event_store,
                merge_sha=merge_commit,
                on_result=_warm_capture.append if _is_warm_path else None,
                quarantine=self._runner_quarantine,
                keep_worktrees=set(self._owned_merge_worktrees),
            ))
            while True:
                done, _ = await asyncio.wait(
                    {verify_task},
                    timeout=self.VERIFY_ABANDON_POLL_SECS,
                )
                if verify_task in done:
                    out = verify_task.result()
                    break
                # Abort trigger 1 — sole-waiter gave up (future cancelled):
                # DROP the request (checked first so a gave-up waiter wins
                # over the operator-halt re-queue below when both hold).
                if self._request_abandoned(req):
                    verify_task.cancel()
                    with contextlib.suppress(BaseException):
                        await verify_task
                    await self._cleanup_owned_merge_worktree(merge_wt)
                    return False
                # Abort trigger 2 — operator halt: terminate the in-flight
                # verify (CancelledError propagates into _run_cmd, which kills
                # the verify subprocess) and RE-QUEUE the merge for re-verify
                # after un-halt.  req.result is left pending so the waiting
                # workflow keeps waiting; per-task retry counters are untouched
                # (a transient operator halt is not a verify failure).
                if self._operator_halt.is_set():
                    logger.warning(
                        'Task %s: operator halt — aborting in-flight verify '
                        'and re-queuing merge for re-verify after un-halt',
                        req.task_id,
                    )
                    verify_task.cancel()
                    with contextlib.suppress(BaseException):
                        await verify_task
                    await self._cleanup_owned_merge_worktree(merge_wt)
                    self._queue.put_nowait(req)
                    return False
        except Exception as exc:
            logger.info(
                f'Task {req.task_id}: verify end '
                f'(merge={merge_commit[:8]}, error)'
            )
            await self._cleanup_owned_merge_worktree(merge_wt)
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
            # Parse per-test results from the warm verify for shadow compare.
            # Only populated when _is_warm_path and on_result captured a result.
            if _warm_capture:
                _warm_results = parse_per_test_results(_warm_capture[0].test_output or '')
                if not _warm_results and req.config.git.warm_verify_shadow_compare:
                    # Fail-closed: if tests ran but we parsed nothing, the
                    # shadow-compare detective is silently inert — raise a
                    # born-at-L2 alarm instead of silently skipping.
                    _alarm_warm_shadow_unparseable(
                        self._escalation_queue,
                        merge_commit,
                        _warm_capture[0].test_output or '',
                    )
        elif out.verify_skipped:
            # Disk guard fired — run_scoped_verification was never called;
            # log 'skipped' rather than 'passed=False' to avoid misleading
            # post-mortem triage of merge-queue stalls (2026-06-01).
            logger.info(
                f'Task {req.task_id}: verify skipped: low disk '
                f'(merge={merge_commit[:8]})'
            )
            self._resolve_or_drop_abandoned(req, out)
            return False
        else:
            logger.info(
                f'Task {req.task_id}: verify end (merge={merge_commit[:8]}, '
                f'passed=False)'
            )
            self._resolve_or_drop_abandoned(req, out)
            return False

        # Short-circuit: if abandonment landed while (or just as) verify
        # completed, skip the expensive advance-main CAS loop and
        # _finalize_advanced_merge work.  req.result is already cancelled by
        # detach(); _request_abandoned emits the canonical log once and we
        # clean up merge_wt before returning (task 1681, reviewer suggestion 1).
        if self._request_abandoned(req):
            await self._cleanup_owned_merge_worktree(merge_wt)
            return False

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
                await self._cleanup_owned_merge_worktree(merge_wt)
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
                self._resolve_or_drop_abandoned(req, outcome)
                # SMW-only post-merge notification hook (task 1592).  Fires only
                # on a 'done' landing: _finalize_advanced_merge may instead
                # return a 'blocked' outcome (equivalence/pyright gate), and
                # main's pre-refactor inline code reached this hook only after
                # those gates passed (they returned early on failure).  Guard on
                # status == 'done' to preserve that semantics; outcome.merge_sha
                # carries the advanced SHA that the old inline code passed.
                # MergeWorker deliberately has no such hook.
                if outcome.status == 'done':
                    if (
                        outcome.merge_sha is not None
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
                    # PRD §10 invariant 6(b): schedule shadow compare on the
                    # same-candidate merge commit — off the serial lane.
                    # _maybe_schedule_shadow_compare returns IMMEDIATELY (spawns a
                    # task); _warm_results empty → no-op inside the scheduler.
                    await _maybe_schedule_shadow_compare(
                        self, self._git_ops, req, merge_commit,
                        _warm_results, self._escalation_queue, self._event_store,
                    )
                    # Lever C drift detective: cadence-gated multi-host parity
                    # check.  Returns IMMEDIATELY when cadence not met or no
                    # enabled runners; spawns asyncio.create_task off the serial
                    # lane when due.  Worker quarantine propagates into subsequent
                    # _run_post_merge_verify dispatches via _runner_quarantine.
                    await _maybe_run_drift_check(
                        self, self._git_ops, req, merge_commit,
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
                    merge_sha=rebased_sha,
                    keep_worktrees=set(self._owned_merge_worktrees),
                )
                if gate is not None:
                    # Overlapping delta, verify failed (or disk guard fired).
                    # _run_post_merge_verify already cleaned up merge_wt.
                    # NOTE: uses the bare guard (not _resolve_or_drop_abandoned)
                    # because this rebase-gate site is outside the three
                    # task-1681 resolution sites (disk-skip, verify-fail,
                    # advance-success); see plan design decision.  A detach
                    # landing this deep (after advance_main returned
                    # rebased_pending_reverify) is a very narrow window; the
                    # bare guard is left intentionally to keep the regression
                    # surface minimal.
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
                    await self._cleanup_owned_merge_worktree(merge_wt)
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
                await self._cleanup_owned_merge_worktree(merge_wt)
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
                await self._cleanup_owned_merge_worktree(merge_wt)
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
                await self._cleanup_owned_merge_worktree(merge_wt)
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

    **Heartbeat-floor model (task 1729 / β)**

    Since the α owner-heartbeat (task 1728) touches every owned ``_merge-*``
    worktree every :data:`_HEARTBEAT_POLL_S` seconds, the worst frozen age
    for a *live* worker's worktrees is bounded by the stall budget
    ``_HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE``, independent of K,
    cold timeout, or num_hosts.  The old ``timeout_secs / merge_ahead_bound /
    num_hosts / max_verify_timeouts`` fields are therefore removed.

    Attributes:
        worst_case_secs: Heartbeat floor — ``_HEARTBEAT_POLL_S ×
            TOUCH_MISS_TOLERANCE`` (seconds); invariant across config.
        threshold_secs: Safety threshold (``safety_factor * liveness_secs``);
            the guard fires when ``worst_case_secs >= threshold_secs``.
        liveness_secs: The reaper's liveness window passed to the guard.
        heartbeat_poll_secs: Value of :data:`_HEARTBEAT_POLL_S` at call time
            (for diagnostics and monkeypatch verification).
        touch_miss_tolerance: Value of :data:`TOUCH_MISS_TOLERANCE` at call
            time (for diagnostics and monkeypatch verification).
        safety_factor: Fraction of *liveness_secs* used as the threshold.
        safe: True iff ``worst_case_secs < threshold_secs``.
    """

    worst_case_secs: float
    threshold_secs: float
    liveness_secs: float
    heartbeat_poll_secs: float
    touch_miss_tolerance: int
    safety_factor: float
    safe: bool


def check_merge_liveness_margin(
    config: OrchestratorConfig,
    *,
    liveness_secs: float = INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS,
    safety_factor: float = 0.75,
    logger: logging.Logger = logger,
) -> MergeLivenessAssessment:
    """Evaluate whether the heartbeat floor fits safely within the reaper's
    liveness window and emit a WARNING when it does not.

    Called once at orchestrator startup (from ``Harness._start_merge_worker``)
    against the live per-project :class:`OrchestratorConfig`.

    **Physical model (task 1729 / β — heartbeat-floor)**

    The α owner-heartbeat (task 1728) touches every owned ``_merge-*``
    worktree's mtime every :data:`_HEARTBEAT_POLL_S` seconds.  Under normal
    operation a live worker's worktrees never age past ~1 poll period.  A
    sustained event-loop stall (GIL contention, heavy I/O, OS scheduling
    jitter) can delay the heartbeat, but the stall budget is:

    .. code-block:: text

        floor_secs = _HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE

    K (verify pool size), cold timeout, and num_hosts drop out of the formula
    because the heartbeat model makes them irrelevant: a live worker touches
    its owned worktrees regardless of how many verifiers are in flight or how
    long each verify takes.

    **Formula**

    .. code-block:: text

        worst_case_secs = _HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE  # read as module globals
        threshold_secs  = safety_factor × liveness_secs
        safe            = worst_case_secs < threshold_secs

    Both :data:`_HEARTBEAT_POLL_S` and :data:`TOUCH_MISS_TOLERANCE` are read
    in the function body (not as default-argument values) so that a
    harness-level test can monkeypatch either constant and have the change
    reflected at call time.

    **Default calibration**

    With ``_HEARTBEAT_POLL_S=30``, ``TOUCH_MISS_TOLERANCE=20``,
    ``safety_factor=0.75``, and ``liveness_secs=10800``:

    - floor = 600 s
    - threshold = 8100 s
    - margin = 13.5× ≥ 3× (PRD minimum)
    - All shipped configs (including ``merge_verify_cold_command_timeout_secs
      =9000``) are now safe because cold timeout no longer feeds the formula.

    Args:
        config: Live per-project orchestrator config.  Kept as the first
            positional parameter for call-signature stability; the heartbeat
            floor does not read config fields.
        liveness_secs: Override for :data:`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS`.
            The corrective operator lever: raise this to widen the safety
            threshold.
        safety_factor: Fraction of *liveness_secs* that constitutes the
            "comfortably below" threshold.  Default 0.75.
        logger: Logger to use for the WARNING (default: module logger, captured
            by ``pytest caplog`` as ``orchestrator.merge_queue``).

    Returns:
        :class:`MergeLivenessAssessment` with all computed values and the
        ``safe`` verdict.
    """
    # Heartbeat-floor formula.  Read _HEARTBEAT_POLL_S and TOUCH_MISS_TOLERANCE
    # as module globals here (not as default-arg values) so that monkeypatching
    # either constant inside a test is reflected at call time.
    _poll = _HEARTBEAT_POLL_S
    _tolerance = TOUCH_MISS_TOLERANCE

    worst_case_secs = _poll * _tolerance
    threshold_secs = safety_factor * liveness_secs
    safe = worst_case_secs < threshold_secs

    assessment = MergeLivenessAssessment(
        worst_case_secs=worst_case_secs,
        threshold_secs=threshold_secs,
        liveness_secs=liveness_secs,
        heartbeat_poll_secs=_poll,
        touch_miss_tolerance=_tolerance,
        safety_factor=safety_factor,
        safe=safe,
    )

    if not safe:
        logger.warning(
            'check_merge_liveness_margin: heartbeat floor (%.0fs = '
            'heartbeat_poll=%.0fs × touch_miss_tolerance=%d) is not '
            'comfortably below the reaper liveness threshold '
            '(%.0fs, factor=%.2f, liveness=%.0fs). '
            'Raise INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS or lower '
            'TOUCH_MISS_TOLERANCE to reduce the floor.',
            worst_case_secs,
            _poll,
            _tolerance,
            threshold_secs,
            safety_factor,
            liveness_secs,
        )

    return assessment


class MergeLivenessConfigError(Exception):
    """Raised by :func:`enforce_merge_liveness_margin` when the heartbeat
    floor (``_HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE``) is not comfortably
    below the reaper liveness threshold, indicating that startup should be
    refused.

    The exception message names the heartbeat model and the corrective levers
    (raise :data:`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` or lower
    :data:`TOUCH_MISS_TOLERANCE`) for operator triage.  Cold timeout and
    merge-ahead bound are NOT named because the heartbeat model decouples them.
    """


def enforce_merge_liveness_margin(
    config: OrchestratorConfig,
    *,
    liveness_secs: float = INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS,
    safety_factor: float = 0.75,
    logger: logging.Logger = logger,
) -> MergeLivenessAssessment:
    """Fail-closed wrapper around :func:`check_merge_liveness_margin`.

    Delegates entirely to :func:`check_merge_liveness_margin` (which remains
    WARNING-only / fail-open) and raises :exc:`MergeLivenessConfigError` when
    the assessment is ``not safe``, causing startup to be refused.

    Args:
        config: Live per-project orchestrator config.  Forwarded verbatim to
            :func:`check_merge_liveness_margin`.
        liveness_secs: Override for :data:`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS`.
            The corrective operator lever: raise this to widen the safety
            threshold.
        safety_factor: Fraction of *liveness_secs* that constitutes the
            threshold.  Default 0.75.
        logger: Logger forwarded to :func:`check_merge_liveness_margin`.

    Returns:
        :class:`MergeLivenessAssessment` when the config is safe.

    Raises:
        :exc:`MergeLivenessConfigError`: When ``worst_case_secs >= threshold_secs``
            (i.e. heartbeat floor ≥ threshold).
    """
    assessment = check_merge_liveness_margin(
        config,
        liveness_secs=liveness_secs,
        safety_factor=safety_factor,
        logger=logger,
    )
    if not assessment.safe:
        raise MergeLivenessConfigError(
            f'enforce_merge_liveness_margin: startup refused — heartbeat floor '
            f'({assessment.worst_case_secs:.0f}s = heartbeat_poll='
            f'{assessment.heartbeat_poll_secs:.0f}s × touch_miss_tolerance='
            f'{assessment.touch_miss_tolerance}) is not below the reaper '
            f'liveness threshold ({assessment.threshold_secs:.0f}s, '
            f'factor={safety_factor:.2f}, liveness={assessment.liveness_secs:.0f}s). '
            f'Raise INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS or lower '
            f'TOUCH_MISS_TOLERANCE to reduce the floor.'
        )
    return assessment


# ---------------------------------------------------------------------------
# Persistent warm merge-verify worktree — serial-lane startup guard
# ---------------------------------------------------------------------------


class PersistentWorktreeConfigError(Exception):
    """Raised by :func:`enforce_persistent_worktree_serial_lane` when the
    persistent warm merge-verify worktree is enabled but the per-host
    in-flight verify count would exceed 1, which would risk concurrent cargo
    invocations on a single shared ``target/`` directory.

    The exception message names the bound, host count, and per-host in-flight
    count so the operator knows what to change (lower merge_ahead_bound,
    increase num_hosts, or disable ``git.persistent_merge_worktree``).
    """


def _safety_valve_due(attempt_count: int, every_n: int) -> bool:
    """Return True when the periodic cold-verify safety valve should fire.

    The safety valve (PRD §10 invariant 6) bypasses the warm-worktree swap on
    every Nth verifying serial attempt so that a true from-scratch cold verify
    runs in a throwaway ephemeral worktree (target NOT retained).  A cold
    failure on the serial lane surfaces through the existing verify-failure
    escalation path.

    Args:
        attempt_count: The 1-based count of verifying attempts on this worker
            (incremented in ``_verify_and_advance`` before calling this).
        every_n: From ``config.git.persistent_merge_worktree_safety_valve_every_n``.
            0 or negative → disabled (always returns False).

    Returns:
        True when the valve is enabled (``every_n > 0``) and
        ``attempt_count`` is a positive multiple of ``every_n``.
    """
    return every_n > 0 and attempt_count > 0 and attempt_count % every_n == 0


# ---------------------------------------------------------------------------
# PRD §10 invariant 6(b): warm-vs-cold SHADOW compare cadence
# ---------------------------------------------------------------------------

@dataclass
class ShadowCompareState:
    """Persisted cadence state for the warm-vs-cold shadow compare.

    Stored as JSON at ``config.project_root/data/orchestrator/warm_verify_shadow.json``
    so both cadence conditions survive orchestrator restarts.

    Fields:
        merges_since_last_shadow: Count of warm-verified lands since the last
            shadow compare run.  Reset to 0 when a shadow compare is triggered.
        last_shadow_run_at: Unix timestamp (float) of the last shadow compare
            trigger.  0.0 when no shadow compare has ever run.
    """

    merges_since_last_shadow: int = 0
    last_shadow_run_at: float = 0.0


# ---------------------------------------------------------------------------
# Per-test result parsers for the warm-vs-cold shadow compare.
#
# Two formats are supported so that the shadow compare works regardless of
# which test runner the project's verify command uses:
#
#   cargo-nextest (reify's default):
#       "        PASS [   0.045s] reify-core some::mod::test_a"
#       "        FAIL [   1.200s] reify-eval other::test_b"
#       "        TIMEOUT [  5.000s] crate slow::test"
#       "        LEAK [   0.100s] crate leaky::test"
#       "        SIGSEGV [  0.001s] crate crash::test"
#   Groups: (1) status, (2) crate, (3) test_path
#
#   Rust libtest (plain `cargo test` output):
#       "test some::mod::test_name ... ok"
#       "test some::mod::test_name ... FAILED"
#   Groups: (1) test_path, (2) status
#
# SKIP / ignored lines are intentionally excluded: a skipped/ignored test
# is not "run" so treating it as present-but-failed would create spurious
# only_warm / only_cold presence divergences between warm and cold runs.
# ---------------------------------------------------------------------------

# Matches cargo-nextest human-output test result lines.
# Capture groups: (1) status, (2) crate/package::binary, (3) test path (rest of line)
#
# Real cargo-nextest 0.9.136 output (reify's merge-verify runner) inserts an
# OPTIONAL parenthesized progress counter such as '(  1/250)' between the timing
# bracket and the package::binary id, e.g.:
#
#     PASS [   0.130s] (  1/250) reify-cli::cli_affine_eval eval_x
#
# The non-capturing optional group ``(?:\(\s*\d+/\s*\d+\)\s+)?`` consumes and
# DISCARDS the counter so it does not appear in the stable key
# ``"pkg::bin test_path"``.  Without this group the regex captures the open-paren
# '(' as the crate and folds the counter remainder into the test path, producing
# run-specific garbage keys that break warm/cold shadow comparison.
#
# Backward-compatible: the group is optional, so old no-counter format and the
# libtest branch are unaffected.
_NEXTEST_TEST_LINE_RE = re.compile(
    r'^\s*(PASS|FAIL|TIMEOUT|LEAK|SIGSEGV)\s+\[[^\]]*\]\s+'
    r'(?:\(\s*\d+/\s*\d+\)\s+)?'  # optional N/M progress counter — consumed, not captured
    r'(\S+)\s+(\S.*?)\s*$'
)

# Matches plain `cargo test` (libtest) result lines.
# Capture groups: (1) test_path, (2) status ("ok" or "FAILED")
_LIBTEST_TEST_LINE_RE = re.compile(
    r'^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED)\s*$'
)


def parse_per_test_results(test_output: str) -> dict[str, bool]:
    """Parse test runner output into a per-test pass/fail map.

    Supports two formats:

    * **cargo-nextest** (reify's default merge-verify runner)::

          <whitespace> PASS|FAIL|TIMEOUT|LEAK|SIGSEGV [<timing>] [(<N>/<M>)] <pkg::bin> <path>

      Real cargo-nextest 0.9.136 output inserts an optional parenthesized progress
      counter ``(  N/M)`` (with internal whitespace padding) between the timing
      bracket and the ``package::binary`` id.  The counter is consumed and
      **excluded** from the key so that warm and cold runs (which have different
      N/M indices) produce identical stable keys.

      Key: ``"<pkg::bin> <test::path>"``, value: ``True`` iff status is ``PASS``.
      TIMEOUT / LEAK / SIGSEGV are treated as failures (``False``).

    * **libtest** (plain ``cargo test``)::

          test <test::path> ... ok|FAILED

      Key: ``"<test::path>"``, value: ``True`` iff status is ``ok``.

    SKIP / ignored lines are excluded from both formats so they do not
    introduce spurious presence-divergences in the shadow compare diff.

    All other lines (build output, summary footer, blank lines) are ignored.

    Used by the warm-vs-cold shadow compare (PRD §10 invariant 6(b)) to
    capture per-test granularity so divergences can be named in the L2 alarm.

    Args:
        test_output: Raw string output from a verify run.

    Returns:
        ``dict[str, bool]`` mapping test id to pass status.  Empty dict for
        empty/blank input or when no test lines are present.  A caller that
        receives an empty dict from a genuine verify run should log a warning
        — the parser may not match the project's verify command output format.
    """
    result: dict[str, bool] = {}
    for line in test_output.splitlines():
        m = _NEXTEST_TEST_LINE_RE.match(line)
        if m:
            status, crate, test_path = m.group(1), m.group(2), m.group(3)
            result[f"{crate} {test_path}"] = (status == 'PASS')
            continue
        m = _LIBTEST_TEST_LINE_RE.match(line)
        if m:
            test_path, status = m.group(1), m.group(2)
            result[test_path] = (status == 'ok')
    return result


# Matches cargo-nextest Summary footer lines, e.g.:
#   Summary [   1.25s] 250 tests run: 249 passed, 1 failed, 0 skipped
#   Summary [   0.13s]   1 test run: 1 passed, 0 failed, 0 skipped   (N==1 → singular)
#   (leading whitespace tolerated: nextest may indent the Summary footer)
# Capture group: (1) total test count N from 'N tests run:' / 'N test run:'
_NEXTEST_SUMMARY_LINE_RE = re.compile(
    r'^\s*Summary\s+\[[^\]]*\]\s+(\d+)\s+tests?\s+run:',
    re.MULTILINE,
)


def _nextest_reported_test_count(output: str) -> int | None:
    """Return the total number of tests reported in nextest Summary footer line(s).

    Scans all lines in *output* for the cargo-nextest human-format footer::

        Summary [<timing>] N tests run: P passed, F failed, S skipped

    Returns the **sum** of N across all matched Summary lines (to cover
    multi-pass debug+release aggregate runs), or ``None`` when no Summary
    line is found in the output.

    A return value of ``0`` is distinct from ``None``:  ``0`` means a Summary
    was found but reported zero tests run (e.g. legitimately test-free crate);
    ``None`` means no nextest pass occurred at all (pure build noise or empty
    output).

    Used by :func:`_alarm_warm_shadow_unparseable` to discriminate between
    a genuinely test-free merge (no alarm) and a parser failure (alarm).

    Args:
        output: Raw string from a verify run.

    Returns:
        Sum of reported test counts, or ``None`` if no Summary line present.
    """
    matches = _NEXTEST_SUMMARY_LINE_RE.findall(output)
    if not matches:
        return None
    return sum(int(n) for n in matches)


@dataclass
class ShadowCompareDiff:
    """Per-test divergence between a warm and a cold verify run.

    Produced by :func:`diff_per_test_results` for PRD §10 invariant 6(b).
    The ``diverging`` dict contains every test whose warm/cold verdicts differ;
    the list buckets partition diverging tests by direction for easy alarming.

    Presence divergences (a test in only one result set) are also recorded
    because they indicate structural differences between the two runs.

    Attributes:
        diverging: Maps test_id → (warm_passed, cold_passed) for every
            diverging test.
        warm_pass_cold_fail: Test ids that passed warm but failed cold
            (the dangerous class: warm landed OK, cold reveals a real fail).
        warm_fail_cold_pass: Test ids that failed warm but passed cold
            (less dangerous; warm was conservative).
        only_warm: Test ids present in the warm result but absent from cold
            (structural difference; may indicate a cold build failure).
        only_cold: Test ids present in the cold result but absent from warm.
    """

    diverging: dict[str, tuple[bool, bool]]
    warm_pass_cold_fail: list[str]
    warm_fail_cold_pass: list[str]
    only_warm: list[str]
    only_cold: list[str]

    @property
    def has_divergence(self) -> bool:
        """True iff any divergence bucket is non-empty."""
        return bool(
            self.diverging
            or self.only_warm
            or self.only_cold
        )


def diff_per_test_results(
    warm: dict[str, bool],
    cold: dict[str, bool],
) -> ShadowCompareDiff:
    """Compute the per-test divergence between warm and cold verify results.

    Classifies every test in the union of both result sets into a divergence
    bucket.  Tests whose warm verdict equals their cold verdict are omitted.

    Args:
        warm: Per-test results from the warm (in-place) verify run,
            as returned by :func:`parse_per_test_results`.
        cold: Per-test results from the cold (throwaway-worktree) verify run.

    Returns:
        A :class:`ShadowCompareDiff` with buckets populated for diverging
        tests.  ``has_divergence`` is False iff all buckets are empty.
    """
    diverging: dict[str, tuple[bool, bool]] = {}
    warm_pass_cold_fail: list[str] = []
    warm_fail_cold_pass: list[str] = []
    only_warm: list[str] = []
    only_cold: list[str] = []

    all_tests = warm.keys() | cold.keys()
    for test_id in sorted(all_tests):
        in_warm = test_id in warm
        in_cold = test_id in cold
        if in_warm and in_cold:
            w, c = warm[test_id], cold[test_id]
            if w != c:
                diverging[test_id] = (w, c)
                if w and not c:
                    warm_pass_cold_fail.append(test_id)
                else:
                    warm_fail_cold_pass.append(test_id)
        elif in_warm:
            only_warm.append(test_id)
        else:
            only_cold.append(test_id)

    return ShadowCompareDiff(
        diverging=diverging,
        warm_pass_cold_fail=warm_pass_cold_fail,
        warm_fail_cold_pass=warm_fail_cold_pass,
        only_warm=only_warm,
        only_cold=only_cold,
    )


def _load_shadow_compare_state(path: Path) -> ShadowCompareState:
    """Load the shadow compare cadence state from a JSON file.

    Fail-safe: returns a default ``ShadowCompareState()`` on any error
    (file not found, unreadable, unparseable JSON, or missing keys) so the
    orchestrator never fails to start due to a corrupt state file.

    Args:
        path: Path to the JSON state file (typically
            ``config.project_root/data/orchestrator/warm_verify_shadow.json``).

    Returns:
        The persisted state, or ``ShadowCompareState(0, 0.0)`` on any failure.
    """
    try:
        data = json.loads(path.read_text())
        return ShadowCompareState(
            merges_since_last_shadow=int(data['merges_since_last_shadow']),
            last_shadow_run_at=float(data['last_shadow_run_at']),
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return ShadowCompareState()


def _save_shadow_compare_state(path: Path, state: ShadowCompareState) -> None:
    """Persist the shadow compare cadence state to a JSON file.

    Creates parent directories as needed.

    Args:
        path: Destination path for the JSON state file.
        state: The :class:`ShadowCompareState` to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(state)))


def _shadow_compare_due(
    state: ShadowCompareState,
    now: float,
    *,
    every_n_merges: int,
    nightly_interval_secs: float,
) -> bool:
    """Return True when a shadow compare should be triggered.

    Implements PRD §10 invariant 6(b) "whichever sooner" = OR cadence:
    a shadow compare fires when EITHER the merge-count leg OR the nightly-timer
    leg is satisfied.

    Count leg: fires when ``state.merges_since_last_shadow >= every_n_merges``
        (provided ``every_n_merges > 0``; 0 disables the count leg entirely).

    Nightly leg: fires when ``now - state.last_shadow_run_at >= nightly_interval_secs``
        (provided ``nightly_interval_secs > 0``; 0 disables the timer leg).

    Args:
        state: Current persisted cadence state.
        now: Current Unix timestamp (time.time()).
        every_n_merges: From ``config.git.warm_verify_shadow_compare_every_n_merges``.
            0 → count leg disabled.
        nightly_interval_secs: From
            ``config.git.warm_verify_shadow_compare_nightly_interval_secs``.
            0 → timer leg disabled.

    Returns:
        True when at least one trigger condition is met.
    """
    count_due = (
        every_n_merges > 0
        and state.merges_since_last_shadow >= every_n_merges
    )
    timer_due = (
        nightly_interval_secs > 0
        and (now - state.last_shadow_run_at) >= nightly_interval_secs
    )
    return count_due or timer_due


# Sentinel task_id used for dedup on the warm/cold shadow divergence escalation.
# Mirrors ``_DRIFT_SENTINEL`` in verify_runner.py.
_WARM_COLD_SHADOW_SENTINEL = '__warm_cold_shadow__'

# Sentinel task_id for the fail-closed unparseable-format escalation.
# Kept DISTINCT from _WARM_COLD_SHADOW_SENTINEL so a divergence alarm and an
# unparseable-format alarm dedup independently.
_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL = '__warm_cold_shadow_unparseable__'


def _submit_shadow_divergence_escalation(
    escalation_queue: Any,
    merge_commit: str,
    diff: ShadowCompareDiff,
    warm_results: dict[str, bool],
    cold_results: dict[str, bool],
) -> None:
    """Submit a born-at-L2 escalation for a warm/cold shadow divergence.

    Implements PRD §10 invariant 6(b) L2 alarm.  The escalation is:

    * ``severity='critical'`` (in ``BORN_AT_L2_SEVERITIES``) → born at L2
    * ``level=2``
    * ``agent_role='orchestrator-warm-cold-shadow'`` (``orchestrator-`` prefix →
      harness sentinel → not downgraded by the escalation server)
    * ``category='risk_identified'``
    * ``task_id=_WARM_COLD_SHADOW_SENTINEL`` (dedup key)

    The detail explicitly states that the warm merge has ALREADY LANDED via the
    shadow/async lane and that the commit may be bad on main.

    None-safe: if *escalation_queue* is None the function is a no-op.
    Dedup: if an open escalation for the sentinel already exists (checked via
    ``escalation_queue.has_open_l1``), no second submission is made.

    Args:
        escalation_queue: Live escalation queue (``EscalationQueue`` instance),
            or ``None`` when escalation is unavailable.
        merge_commit: Full or abbreviated SHA of the just-landed merge commit.
        diff: Per-test divergence bucket summary from :func:`diff_per_test_results`.
        warm_results: Per-test pass/fail map from the warm verify run.
        cold_results: Per-test pass/fail map from the cold verify run.
    """
    if escalation_queue is None:
        return

    # Dedup: don't fire again while an open/pending alarm already exists.
    # Global-dedup is intentional — matches DriftDetector's _DRIFT_SENTINEL pattern.
    # A single open escalation suppresses ALL subsequent shadow-divergence alarms
    # while it is unresolved.  The expectation is that divergences are investigated
    # in sequence; a rollback recommendation implicitly covers subsequent same-area
    # divergences.  If per-commit independent alarms are ever needed, incorporate
    # the commit into the dedup key (e.g. make_id(f'{_WARM_COLD_SHADOW_SENTINEL}
    # :{merge_commit[:8]}')) — but that change is out of scope for this task.
    if escalation_queue.has_open_l1(_WARM_COLD_SHADOW_SENTINEL):
        return

    from escalation.models import Escalation  # local import — escalation optional dep

    n_diverging = len(diff.diverging) + len(diff.only_warm) + len(diff.only_cold)
    short_sha = merge_commit[:8]

    # Build summary (must name commit[:8] and diverging test count).
    summary = (
        f'Warm/cold shadow divergence on {short_sha}: '
        f'{n_diverging} diverging test(s)'
    )

    # Build detail: list diverging tests + both result sets + "already landed" statement.
    lines: list[str] = [
        f'Commit: {merge_commit}',
        f'Diverging tests ({n_diverging}):',
    ]
    for test_id, (w, c) in sorted(diff.diverging.items()):
        lines.append(f'  warm={w} cold={c}  {test_id}')
    if diff.only_warm:
        lines.append('Tests present only in warm run (absent cold):')
        lines.extend(f'  {t}' for t in diff.only_warm)
    if diff.only_cold:
        lines.append('Tests present only in cold run (absent warm):')
        lines.extend(f'  {t}' for t in diff.only_cold)
    lines.append('')
    lines.append('Warm results: ' + repr(warm_results))
    lines.append('Cold results: ' + repr(cold_results))
    lines.append('')
    lines.append(
        'The warm merge has ALREADY LANDED via the shadow/async lane — '
        'this commit may be bad on main.  '
        'Investigate the diverging tests and consider a potential rollback.'
    )
    detail = '\n'.join(lines)

    esc = Escalation(
        id=escalation_queue.make_id(_WARM_COLD_SHADOW_SENTINEL),
        task_id=_WARM_COLD_SHADOW_SENTINEL,
        agent_role='orchestrator-warm-cold-shadow',
        severity='critical',
        level=2,
        category='risk_identified',
        summary=summary,
        detail=detail,
        suggested_action=(
            'Investigate the diverging tests on the landed merge commit; '
            'roll back main if the cold leg reveals a real failure.'
        ),
    )
    escalation_queue.submit(esc)


def _alarm_warm_shadow_unparseable(
    escalation_queue: Any,
    merge_commit: str,
    test_output: str,
) -> None:
    """Submit a born-at-L2 critical escalation when the warm verify is unparseable.

    Fail-closed guard for the warm/cold shadow-compare detective: when the warm
    verify output shows that tests actually RAN (a nextest Summary footer with
    N > 0) yet :func:`parse_per_test_results` returned an empty dict, the
    detective is silently inert for that landing — a dangerous invisible failure
    mode.  This function converts that silent failure to an L2 alarm.

    The escalation is modelled on :func:`_submit_shadow_divergence_escalation`:

    * ``severity='critical'`` (in ``BORN_AT_L2_SEVERITIES``) → born at L2
    * ``level=2``
    * ``agent_role='orchestrator-warm-cold-shadow-unparseable'``
      (``orchestrator-`` prefix → harness sentinel → not downgraded)
    * ``category='risk_identified'``
    * ``task_id=_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL`` (separate dedup key,
      does not collide with ``_WARM_COLD_SHADOW_SENTINEL``)

    The alarm is suppressed (no false positive) when *test_output* contains no
    nextest Summary line or the Summary reports 0 tests — that case represents a
    legitimately test-free merge.

    None-safe: if *escalation_queue* is None the function is a no-op.
    Dedup: if an open escalation for the unparseable sentinel already exists
    (checked via ``escalation_queue.has_open_l1``), no second submission is made.

    Args:
        escalation_queue: Live escalation queue or ``None``.
        merge_commit: Full or abbreviated SHA of the just-landed merge commit.
        test_output: Raw ``test_output`` string from the warm :class:`VerifyResult`.
    """
    if escalation_queue is None:
        return

    # Discriminate: did tests actually run in this output?
    # NOTE: _nextest_reported_test_count is nextest-only (reads cargo-nextest
    # "Summary [..] N tests run:" footers).  A libtest-format verify run whose
    # per-test parse fails will not match here, so the alarm is suppressed.
    # Warm verify is expected to use cargo-nextest; libtest is not a supported
    # warm-verify format and would fall through as reported=None (no false alarm).
    reported = _nextest_reported_test_count(test_output)
    if reported is None or reported == 0:
        # Legitimately test-free merge (no nextest pass, or zero tests reported).
        # No alarm — would be a false positive.
        if reported is None:
            # Leave a low-severity breadcrumb so suppressed alarms are diagnosable
            # in the field even when no escalation is raised.
            logger.debug(
                'warm shadow-compare: no nextest Summary line found in warm verify '
                'output — unparseable alarm suppressed '
                '(legitimately test-free or non-nextest run)'
            )
        return

    # Dedup: don't fire again while an open/pending alarm already exists.
    if escalation_queue.has_open_l1(_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL):
        return

    from escalation.models import Escalation  # local import — escalation optional dep

    short_sha = merge_commit[:8]
    summary = (
        f'Warm/cold shadow-compare INERT on {short_sha}: '
        f'verify output format could not be parsed ({reported} tests ran, 0 parsed)'
    )
    detail = (
        f'Commit: {merge_commit}\n'
        f'Tests reported by nextest Summary: {reported}\n'
        f'Tests parsed by parse_per_test_results: 0\n'
        '\n'
        'The warm verify ran tests successfully but the per-test parser produced '
        'an empty result map.  The warm/cold shadow-compare detective is INERT '
        'for this landing — divergence detection is disabled.\n'
        '\n'
        'This is a fail-closed alarm: a format mismatch between the verify output '
        'and _NEXTEST_TEST_LINE_RE (or _LIBTEST_TEST_LINE_RE) is silently '
        'disabling the shadow compare.  Fix the per-test parser to match the '
        'actual verify command output format.'
    )

    esc = Escalation(
        id=escalation_queue.make_id(_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL),
        task_id=_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL,
        agent_role='orchestrator-warm-cold-shadow-unparseable',
        severity='critical',
        level=2,
        category='risk_identified',
        summary=summary,
        detail=detail,
        suggested_action=(
            'Fix the per-test result parser (_NEXTEST_TEST_LINE_RE or '
            '_LIBTEST_TEST_LINE_RE in merge_queue.py) to match the actual verify '
            'command output format so the shadow-compare detective can resume.'
        ),
    )
    escalation_queue.submit(esc)


async def _run_cold_shadow_verify(
    git_ops: GitOps,
    req: MergeRequest,
    merge_commit: str,
    event_store: EventStore | None,
) -> dict[str, bool]:
    """Run a from-scratch cold verify on *merge_commit* in a throwaway worktree.

    Creates an ephemeral ``_merge-<uuid>`` worktree at *merge_commit* via
    :meth:`~orchestrator.git_ops.GitOps.create_throwaway_verify_worktree`,
    runs the full merge verify (build_merge_verify_spec + VerifyRunnerPool
    dispatch — the same execution path as ``_run_post_merge_verify``), parses
    the per-test results from the output, and removes the throwaway worktree
    in a ``finally`` block.

    The throwaway worktree is NEVER the persistent warm ``_merge-verify`` path
    — it has no retained ``target/`` warmth — ensuring a true from-scratch
    cold verify (PRD §10 invariant 6(b)).

    Args:
        git_ops: Live :class:`~orchestrator.git_ops.GitOps` instance.
        req: The :class:`MergeRequest` that just warm-landed (provides config,
            module_configs, task_files, task_id).
        merge_commit: The merge commit SHA to verify cold.
        event_store: Optional event store (passed to VerifyRunnerPool; None-safe).

    Returns:
        Per-test pass/fail map as returned by :func:`parse_per_test_results`.
        Empty dict if the cold verify produced no parseable test output.
    """
    wt = await git_ops.create_throwaway_verify_worktree(merge_commit)
    try:
        task_files_tuple = (
            tuple(req.task_files) if req.task_files is not None else None
        )
        spec = build_merge_verify_spec(req.config, req.module_configs, task_files_tuple)
        # LOCAL-ONLY by design: this is the from-scratch cold trust-anchor detective
        # control (PRD §10 invariant 6(b)).  Adding remotes here would (a) defeat the
        # from-scratch-cold guarantee (a remote may have a warm sccache/target) and
        # (b) reintroduce remote scope-derivation concerns into the very control whose
        # purpose is to BE the local ground truth.  See design decision in plan.json.
        pool = VerifyRunnerPool(
            [LocalRunner(
                wt, req.config, req.module_configs, task_files_tuple,
                run_scoped=run_scoped_verification,
                run_unscoped=_run_unscoped_typechecks,
                task_id=req.task_id,
            )],
            event_store=event_store,
            task_id=req.task_id,
        )
        verify = await pool.dispatch(merge_commit, spec)
        return parse_per_test_results(verify.test_output or '')
    finally:
        await git_ops.cleanup_merge_worktree(wt)


async def _run_shadow_compare(
    git_ops: GitOps,
    req: MergeRequest,
    merge_commit: str,
    warm_results: dict[str, bool],
    escalation_queue: Any,
    event_store: EventStore | None,
) -> None:
    """Compare warm vs cold verify results for *merge_commit* and alarm on divergence.

    Implements PRD §10 invariant 6(b) DETECTIVE control:

    1. Runs a cold verify on *merge_commit* via :func:`_run_cold_shadow_verify`
       in a throwaway ``_merge-<uuid>`` worktree (off the serial lane).
    2. Diffs the cold results against *warm_results* via :func:`diff_per_test_results`.
    3. On per-test divergence: submits a born-at-L2 critical escalation via
       :func:`_submit_shadow_divergence_escalation` naming the diverging tests
       and explicitly stating the warm merge has ALREADY LANDED.
    4. On agreement (no divergence): emits an :attr:`~orchestrator.event_store.EventType.verdict_parity_ok`
       event (mirrors :class:`~orchestrator.verify_runner.DriftDetector`).

    **Exception handling**: any exception from the cold leg is logged at WARNING
    level and swallowed.  A shadow/detective control must never crash or stall
    the merge worker — it runs off the critical serial lane via
    ``asyncio.create_task`` (see :func:`_maybe_schedule_shadow_compare`).

    Args:
        git_ops: Live :class:`~orchestrator.git_ops.GitOps` instance.
        req: The :class:`MergeRequest` that warm-landed (provides config +
             module_configs for the cold verify spec).
        merge_commit: The just-landed merge commit SHA.
        warm_results: Per-test pass/fail map captured from the warm verify run.
        escalation_queue: Live escalation queue, or ``None`` (None-safe).
        event_store: Optional event store for parity-ok event emission.
    """
    try:
        cold_results = await _run_cold_shadow_verify(
            git_ops, req, merge_commit, event_store
        )
    except Exception:
        logger.warning(
            'Shadow compare cold leg failed for %s — swallowing exception',
            merge_commit[:8],
            exc_info=True,
        )
        return

    # Inconclusive guard: if the cold leg produced NO test results but the warm
    # run had results, treat this as inconclusive rather than divergence.
    # An empty cold result usually signals a build/compile failure, OOM, or
    # infra hiccup — not a genuine warm-pass/cold-fail flip.
    # diff_per_test_results({warm tests…}, {}) would classify every warm test as
    # only_warm (has_divergence=True), producing a false-positive born-at-L2 alarm
    # that states "warm merge may be bad" when the cold side simply didn't run.
    # This mirrors DriftDetector's INCONCLUSIVE path (avoids alarming on transport
    # failure).  Neither alarm nor parity-ok event is emitted on inconclusive.
    if not cold_results and warm_results:
        logger.warning(
            'Shadow compare inconclusive for %s: cold leg produced no parseable '
            'test results (possible build/compile/infra failure in the throwaway '
            'worktree); not alarming',
            merge_commit[:8],
        )
        return

    diff = diff_per_test_results(warm_results, cold_results)

    if diff.has_divergence:
        _submit_shadow_divergence_escalation(
            escalation_queue, merge_commit, diff, warm_results, cold_results
        )
    else:
        # Parity OK — emit event (mirrors DriftDetector.check verdict_parity_ok)
        if event_store is not None:
            event_store.emit(
                EventType.verdict_parity_ok,
                task_id=req.task_id,
                data={
                    'merge_commit': merge_commit,
                    'shadow_compare': True,
                    'warm_test_count': len(warm_results),
                    'cold_test_count': len(cold_results),
                },
            )


async def _maybe_schedule_shadow_compare(
    worker: SpeculativeMergeWorker,
    git_ops: GitOps,
    req: MergeRequest,
    merge_commit: str,
    warm_results: dict[str, bool],
    escalation_queue: Any,
    event_store: EventStore | None,
) -> None:
    """Non-blocking scheduler for the warm-vs-cold SHADOW compare (PRD §10 invariant 6(b)).

    Called from :meth:`SpeculativeMergeWorker._verify_and_advance` on every
    successful warm-verified land.  Returns **immediately** without awaiting
    the cold leg — the shadow/detective control must never block or occupy the
    serial merge lane.

    Cadence (whichever sooner = OR):

    * Every *N* merges (``warm_verify_shadow_compare_every_n_merges``).
    * Once per nightly window (``warm_verify_shadow_compare_nightly_interval_secs``).

    State is persisted to ``worker._shadow_state_path`` so the cadence
    survives orchestrator restarts.

    Single-in-flight guard: if a shadow compare task is already running (tracked in
    ``worker._shadow_compare_tasks``), the new trigger is silently skipped so the
    cold leg never piles up behind the serial lane.

    Args:
        worker: The live :class:`SpeculativeMergeWorker` instance (provides
            ``_shadow_compare_tasks`` set and ``_shadow_state_path``).
        git_ops: Live :class:`~orchestrator.git_ops.GitOps` instance.
        req: The :class:`MergeRequest` that just landed (provides config).
        merge_commit: The just-landed merge commit SHA (same-candidate guarantee).
        warm_results: Per-test pass/fail map from the warm verify run.
        escalation_queue: Live escalation queue, or ``None`` (None-safe).
        event_store: Optional event store for parity-ok event emission.
    """
    # Early exits: knob off or no warm results to compare against
    if not req.config.git.warm_verify_shadow_compare:
        return
    if not warm_results:
        return
    # None-safe: _shadow_state_path is None on bare-harness workers (mirrors the
    # escalation_queue None-safety / bare-harness contract in __init__).
    # _load_shadow_compare_state(None) raises AttributeError — not in its except
    # tuple — so guard here to keep the Path|None type sound at call sites.
    if worker._shadow_state_path is None:
        return

    # Load persisted cadence state (fail-safe: returns default on missing/corrupt)
    state = _load_shadow_compare_state(worker._shadow_state_path)

    # Increment the merge counter (counts this landing)
    state = ShadowCompareState(
        merges_since_last_shadow=state.merges_since_last_shadow + 1,
        last_shadow_run_at=state.last_shadow_run_at,
    )

    now = time.time()
    due = _shadow_compare_due(
        state, now,
        every_n_merges=req.config.git.warm_verify_shadow_compare_every_n_merges,
        nightly_interval_secs=req.config.git.warm_verify_shadow_compare_nightly_interval_secs,
    )

    if not due:
        # Save incremented counter and return without scheduling a task
        _save_shadow_compare_state(worker._shadow_state_path, state)
        return

    # In-flight guard: skip if a shadow compare task is already running.
    # Persist the incremented counter even on early-return so merges that
    # land during an in-flight cold leg are still counted (amendment: fix
    # cadence_counter_loss where the due-but-in-flight path did not persist).
    in_flight = [t for t in worker._shadow_compare_tasks if not t.done()]
    if in_flight:
        _save_shadow_compare_state(worker._shadow_state_path, state)
        return

    # Due and no in-flight task: reset state + persist
    state = ShadowCompareState(merges_since_last_shadow=0, last_shadow_run_at=now)
    _save_shadow_compare_state(worker._shadow_state_path, state)

    # Spawn the shadow compare OFF the serial lane — this call returns IMMEDIATELY
    # without awaiting the cold verify (detective/async control, PRD §10 invariant 6(b)).
    t = asyncio.create_task(
        _run_shadow_compare(
            git_ops, req, merge_commit, warm_results, escalation_queue, event_store
        )
    )

    def _discard_task(task: asyncio.Task) -> None:  # type: ignore[type-arg]
        worker._shadow_compare_tasks.discard(task)

    t.add_done_callback(_discard_task)
    worker._shadow_compare_tasks.add(t)


async def _run_drift_check(
    git_ops: GitOps,
    req: MergeRequest,
    merge_commit: str,
    escalation_queue: Any,
    event_store: EventStore | None,
    quarantine_set: set[str],
) -> None:
    """Drift detective control: run DriftDetector.check in a throwaway worktree.

    Mirrors ``_run_cold_shadow_verify`` / ``_run_shadow_compare`` as an
    off-serial-lane detective control (spawned via :func:`_maybe_run_drift_check`).
    Creates a throwaway verify worktree, builds a 2-host pool (local trust-anchor
    + eligible remote from config), runs :meth:`~orchestrator.verify_runner.DriftDetector.check`,
    and propagates any quarantined remote name into *quarantine_set* (the worker-
    level shared set consulted by ``_run_post_merge_verify``).

    Exceptions are caught and logged so this detective control never crashes the
    worker.  ``cleanup_merge_worktree`` is called in a ``finally`` block.

    Args:
        git_ops: Live :class:`~orchestrator.git_ops.GitOps` instance.
        req: The :class:`MergeRequest` that just landed (config, task_files, …).
        merge_commit: The just-landed merge commit SHA.
        escalation_queue: Live escalation queue (None-safe: passed to DriftDetector).
        event_store: Optional event store (None-safe: passed to DriftDetector).
        quarantine_set: Worker-level mutable set; quarantined remote names are
            added here so subsequent ``_run_post_merge_verify`` dispatches
            skip the diverged remote (in-memory protection until restart).
    """
    # wt is initialised before the try so the finally guard (`if wt is not None`)
    # is safe even when create_throwaway_verify_worktree itself raises.  Moving the
    # creation inside the try ensures the docstring contract ("Exceptions are caught
    # and logged") holds for all failure modes — disk-full / git errors included.
    wt = None
    try:
        wt = await git_ops.create_throwaway_verify_worktree(merge_commit)
        task_files_tuple = tuple(req.task_files) if req.task_files is not None else None
        # Derive task_files on the dispatching host (fresh main) when not supplied
        # and Lever C is on — mirrors the same gate in _run_post_merge_verify.
        if task_files_tuple is None and req.config.enabled_verify_runners:
            derived = await _derive_task_files_from_git(wt, req.config)
            if derived:
                task_files_tuple = tuple(derived)
        spec = build_merge_verify_spec(req.config, req.module_configs, task_files_tuple)
        pool = VerifyRunnerPool(
            [LocalRunner(
                wt, req.config, req.module_configs, task_files_tuple,
                run_scoped=run_scoped_verification,
                run_unscoped=_run_unscoped_typechecks,
                task_id=req.task_id,
            ), *_build_remote_runners(req.config, wt, quarantine=quarantine_set)],
            event_store=event_store,
            task_id=req.task_id,
        )
        # Cadence is already enforced upstream in _maybe_run_drift_check
        # (_drift_land_count % every_n gate).  DriftDetector.should_sample is
        # never called on this path, so omitting every_n_lands here keeps a
        # single source of truth and avoids a dead duplicate cadence value.
        detector = DriftDetector(
            pool,
            event_store=event_store,
            escalation_queue=escalation_queue,
            task_id=req.task_id,
        )
        # Capture the eligible remote BEFORE check() so we can propagate its
        # name even after the pool quarantines it (eligible_remote() returns
        # None after quarantine).
        eligible_before = pool.eligible_remote()
        result = await detector.check(merge_commit, spec)
        if result.quarantined and eligible_before is not None:
            quarantine_set.add(eligible_before.name)
    except Exception:
        logger.warning(
            'Drift check failed for task %s / commit %s; ignoring (detective control)',
            req.task_id, merge_commit,
            exc_info=True,
        )
    finally:
        if wt is not None:
            await git_ops.cleanup_merge_worktree(wt)


async def _maybe_run_drift_check(
    worker: SpeculativeMergeWorker,
    git_ops: GitOps,
    req: MergeRequest,
    merge_commit: str,
) -> None:
    """Cadence gate + off-serial-lane spawn for the Lever C drift detective.

    Called from :meth:`SpeculativeMergeWorker._verify_and_advance` on every
    successful ('done') land, immediately after ``_maybe_schedule_shadow_compare``.
    Returns **immediately** — spawns a background task when cadence is met,
    otherwise is a no-op.

    Cadence: every ``config.verify_drift_check_every_n_lands`` successful lands.
    No-op when ``config.enabled_verify_runners`` is empty (Lever C off).

    State is tracked in worker-level attrs ``_drift_land_count`` (incremented
    here) and ``_runner_quarantine`` (propagated from :func:`_run_drift_check`).

    Args:
        worker: Live :class:`SpeculativeMergeWorker` instance.
        git_ops: Live :class:`~orchestrator.git_ops.GitOps` instance.
        req: The :class:`MergeRequest` that just landed.
        merge_commit: The just-landed merge commit SHA.
    """
    if not req.config.enabled_verify_runners:
        return
    every_n = req.config.verify_drift_check_every_n_lands
    worker._drift_land_count += 1
    if worker._drift_land_count % every_n == 0:
        _coro = _run_drift_check(
            git_ops, req, merge_commit,
            worker._escalation_queue, worker._event_store,
            worker._runner_quarantine,
        )
        _task = asyncio.create_task(_coro)
        if not isinstance(_task, asyncio.Task):
            # create_task was intercepted (e.g. by a test mock) and did not
            # schedule _coro.  Close it here to prevent
            # "coroutine was never awaited" RuntimeWarning from leaking into
            # unrelated tests (pyproject.toml converts those warnings to errors).
            _coro.close()
        else:
            # Hold a strong reference so the event loop cannot GC the drift
            # detective mid-run (asyncio keeps only a weak ref to Tasks).
            # Mirrors the _shadow_compare_tasks pattern at :func:`_maybe_schedule_shadow_compare`.
            worker._drift_check_tasks.add(_task)
            _task.add_done_callback(
                lambda t: worker._drift_check_tasks.discard(t)
            )


async def _acquire_warm_verify_worktree(
    git_ops: GitOps,
    req: MergeRequest,
    merge_wt: Path | None,
    merge_commit: str,
    *,
    safety_valve_due: bool,
) -> Path | None:
    """Swap the ephemeral merge worktree for the persistent warm worktree.

    Called at the top of ``_verify_and_advance`` (and ``MergeWorker._process``)
    when the persistent warm merge-verify worktree feature is enabled.  Resets
    the fixed ``_merge-verify`` worktree to *merge_commit* (retaining
    ``target/``), cleans up the now-redundant ephemeral *merge_wt*, and returns
    the warm path so that the verify, advance_main, and cleanup_merge_worktree
    calls all run on the stable fixed path.

    When the knob is OFF or the safety valve is due (PRD §10 invariant 6),
    *merge_wt* is returned unchanged so the caller proceeds with the original
    ephemeral worktree — byte-identical to today's behaviour.

    Args:
        git_ops: Live :class:`GitOps` instance.
        req: The :class:`MergeRequest` being processed (provides config).
        merge_wt: The ephemeral merge worktree path produced by
            ``merge_to_main``, or ``None`` if unavailable.
        merge_commit: The SHA of the merge commit to check out in the warm wt.
        safety_valve_due: ``True`` on the Nth verifying attempt (PRD §10
            invariant 6): bypass the swap and run a cold verify in the
            throwaway ephemeral worktree instead.

    Returns:
        The warm ``_merge-verify`` path when the swap is performed, or the
        original *merge_wt* when the knob is off / safety valve is due.
    """
    if not req.config.git.persistent_merge_worktree or safety_valve_due:
        return merge_wt

    warm = await git_ops.reset_persistent_merge_worktree(merge_commit)
    # The merge commit is already a reachable git object; the ephemeral worktree
    # is no longer needed — drop it immediately to free the worktree slot.
    if merge_wt is not None and merge_wt.resolve() != warm.resolve():
        await git_ops.cleanup_merge_worktree(merge_wt)
    return warm


def enforce_persistent_worktree_serial_lane(
    config: OrchestratorConfig,
    *,
    merge_ahead_bound: int = _MERGE_AHEAD_BOUND,
    num_hosts: int = 1,
) -> None:
    """Fail-closed startup guard for the persistent warm merge-verify worktree.

    The warm worktree feature is SERIAL-LANE-ONLY per host (PRD §A invariant 4):
    a single shared ``target/`` directory is only safe when exactly one verify
    attempt runs at a time on that host.  The guard now computes the worst-case
    per-host in-flight count as ``ceil(merge_ahead_bound / num_hosts)`` and
    rejects only when that exceeds 1.

    At ``num_hosts=1`` (the default, matching the harness call site) the logic
    reduces exactly to the original ``bound != 1`` check so all pre-existing
    guard tests and the harness call site are unaffected.

    This guard is called immediately after :func:`enforce_merge_liveness_margin`
    in :meth:`Harness._start_merge_worker` so that any misconfiguration is
    caught at startup (fail-closed) rather than at the first concurrent verify.

    Args:
        config: Live per-project orchestrator config.
        merge_ahead_bound: The effective merge-ahead bound (defaults to the
            module-level :data:`_MERGE_AHEAD_BOUND`).
        num_hosts: Number of verify hosts sharing the workload (default 1).
            Set to the number of RemoteRunner hosts when operating in a
            multi-host configuration so that the per-host ceiling is computed
            correctly (e.g. K=2 across 2 hosts → per_host=1 → no raise).

    Returns:
        ``None`` when the configuration is safe (knob off OR per-host
        in-flight ≤ 1).

    Raises:
        :exc:`PersistentWorktreeConfigError`: When
            ``config.git.persistent_merge_worktree is True`` and the per-host
            in-flight count ``ceil(merge_ahead_bound / num_hosts) > 1``.
    """
    if not config.git.persistent_merge_worktree:
        return None
    # max(1, ...) clamps degenerate inputs: merge_ahead_bound is always >= 1
    # in practice (harness pins _k = _MERGE_AHEAD_BOUND = 1; callers never pass
    # 0 or negative).  The clamp ensures we fail-safe (per_host_inflight stays
    # >= 1 → guard still raises for any positive bound with a single host) rather
    # than silently allowing division-by-zero or a spuriously permissive result.
    per_host_inflight = math.ceil(max(1, merge_ahead_bound) / max(1, num_hosts))
    if per_host_inflight > 1:
        raise PersistentWorktreeConfigError(
            f'enforce_persistent_worktree_serial_lane: startup refused — '
            f'git.persistent_merge_worktree is enabled but the per-host '
            f'in-flight verify count is {per_host_inflight} '
            f'(merge_ahead_bound={merge_ahead_bound}, num_hosts={num_hosts}). '
            f'The persistent warm _merge-verify worktree is serial-lane-only '
            f'per host (PRD §A invariant 4): a shared target/ is unsafe under '
            f'concurrent verify attempts on the same host. '
            f'Lower merge_ahead_bound, increase num_hosts, or disable '
            f'git.persistent_merge_worktree.'
        )
    return None
