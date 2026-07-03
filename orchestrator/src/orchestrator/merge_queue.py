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
import os
import shutil
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable, Collection
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
from orchestrator.merge_drift import (  # noqa: F401  re-export shim
    _maybe_run_drift_check,
    _run_drift_check,
)
from orchestrator.merge_gates import (  # noqa: F401  re-export shim
    _OVERLAP_GIT_ERROR_SENTINEL,
    DROPPED_PLAN_TARGETS_REASON_PREFIX,
    PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
    POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
    POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
    DropGuardResult,
    PlanFilesTouchedResult,
    PostMergePyrightResult,
    _check_plan_files_touched_in_branch,
    _check_plan_targets_in_tree,
    _check_post_merge_equivalence,
    _check_post_merge_pyright,
    _commit_is_linear,
    _finalize_advanced_merge,
    _GenerationChainContext,
    _map_advance_failure,
    _normalize_plan_path,
    _rebase_delta_touched_overlap,
    _resolve_second_parent,
    _reverify_rebased_tree,
)
from orchestrator.merge_liveness import (  # noqa: F401  re-export shim
    _MERGE_WORKER_LOOP_DIED_SENTINEL,
    _VERIFY_HOST_RECOVERED_SENTINEL_PREFIX,
    _VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX,
    MergeLivenessAssessment,
    MergeLivenessConfigError,
    PersistentWorktreeConfigError,
    _acquire_warm_verify_worktree,
    _alarm_verify_host_unreachable,
    _clear_verify_host_unreachable,
    _safety_valve_due,
    _verify_host_unreachable_sentinel,
    check_merge_liveness_margin,
    enforce_merge_liveness_margin,
    enforce_persistent_worktree_serial_lane,
)
from orchestrator.merge_request_ledger import (  # noqa: F401  re-export shim
    RequestLedger,
    StuckRequest,
    _alarm_merge_request_stuck,
    _merge_request_stuck_sentinel,
)
from orchestrator.merge_shadow import (  # noqa: F401  re-export shim
    _LIBTEST_TEST_LINE_RE,
    _NEXTEST_SUMMARY_LINE_RE,
    _NEXTEST_TEST_LINE_RE,
    _WARM_COLD_SHADOW_SENTINEL,
    _WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL,
    ShadowCompareDiff,
    ShadowCompareState,
    _alarm_warm_shadow_unparseable,
    _classify_test_status,
    _load_shadow_compare_state,
    _maybe_schedule_shadow_compare,
    _nextest_reported_test_count,
    _persistent_alarm_tests,
    _run_cold_shadow_verify,
    _run_shadow_compare,
    _save_shadow_compare_state,
    _shadow_compare_due,
    _submit_shadow_divergence_escalation,
    diff_per_test_results,
    parse_per_test_results,
)
from orchestrator.merge_types import (  # noqa: F401  re-export shim
    _INFLIGHT_MERGE_ETA_ESTIMATE_SECS,
    GroupMergeRequest,
    InflightEntry,
    InFlightMergeRegistry,
    InflightStatus,
    InflightVerifyResult,
    MainHealthAutoHealRegistry,
    MergeBounceRegistry,
    MergeDispatchResult,
    MergeOutcome,
    MergeReadyPredicate,
    MergeRequest,
    SoloVerifyResult,
    SpeculativeItem,
    TerminalOutcomeRecord,
    TerminalOutcomeRetention,
    TrainCallbackFactory,
    TrainCallbacks,
    WaiterRecord,
    _HostUnavailability,
    _InFlightEntry,
)
from orchestrator.overlap_footprint import (  # noqa: F401  re-export seam for δ/ζ consumers
    DEFAULT_OVERLAP_DETECTOR,
    DefaultPathOverlapDetector,
    Footprint,
    OverlapFootprintDetector,
    changesets_overlap,
    get_overlap_detector,
    register_overlap_detector,
)
from orchestrator.suffix_graph import (  # noqa: F401  re-export shim
    EMPTY_SUFFIX_CONFLICT_GRAPH,
    SuffixConflictGraph,
    SuffixConflictTracker,
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
    HostAllocator,
    LocalRunner,
    RemoteRunner,
    RunnerUnavailable,
    VerifyRunner,
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


NEEDS_REBASE_REASON_PREFIX = 'Suffix item needs rebase onto frozen-prefix tip'
"""Prefix of the ``MergeOutcome.reason`` string emitted by the η=1892
graph-time bounce logic when a suffix item conflicts with the frozen-prefix
tip (see :meth:`SpeculativeMergeWorker._bounce_conflicting_suffix_items`).

Escalation outcomes carry ``status='blocked'`` with this prefix so they ride
the existing generic ``blocked`` → ``_mark_blocked(escalate_to_human)``
steward route.  The prefix distinguishes this category from other blocked
outcomes for operators and λ analytics — no workflow.py change needed.

Two cases produce this prefix:
* **Real rebase conflict** — the mechanical rebase onto the frozen tip fails;
  the item is removed from the lane buffer and escalated to the steward.
* **Bounce-cap exceeded** — the branch has been bounced
  :data:`MERGE_BOUNCE_CAP` times already; escalated WITHOUT attempting a
  further rebase (prevents an A↔B flapping conflict from burning rebase
  attempts without limit, mirroring the 1688 thrash-signature pattern).
"""

MERGE_BOUNCE_CAP = 3
"""Maximum number of times a branch may be mechanically rebased before the
bounce logic escalates instead of rebasing again (η=1892 cap backstop).

Mirrors the 1688 thrash-signature pattern (:data:`MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS`):
a flapping A↔B conflict where both branches keep rebasing onto each other
cannot become an unbounded agent/rebase fire — after this many bounces the
branch is escalated to the steward.

The count is sha-independent (keyed by branch name in
:class:`MergeBounceRegistry`) so a successful rebase that advances the HEAD
still counts toward the cap for this branch lineage.
"""

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

MERGE_WORKER_SHUTDOWN_REASON = 'Merge worker shutting down'
"""Reason string set on MergeOutcome('blocked') by stop() and the _merger_loop
finally block.  Used by _buffer_owned_request._on_terminal to distinguish a
graceful-stop / crash path (keep in journal for recovery) from a deterministic
terminal failure (remove — not retried on restart)."""


def _normalize_lane(lane: str) -> str:
    """Map an unrecognised lane value to 'normal' (defensive; prevents starvation)."""
    return lane if lane in MERGE_LANES else 'normal'


def _aging_key(req: MergeRequest) -> tuple[float, str]:
    """Aging sort key for a merge request (ζ=1891).

    Returns a ``(wall_clock, request_id)`` tuple:

    * **wall_clock** — ``merge_first_enqueued_at`` (persisted epoch of the
      *first* merge submission, survives restart) with ``enqueued_at`` as
      the legacy fallback for requests created before α=1886 was deployed
      (field is ``None``).  Smaller = older = higher priority.

    * **request_id** — lexical tie-break for equal timestamps (PRD §5.4 /
      §11.3).  Lexically smaller wins.

    Used by ``SpeculativeMergeWorker._pop_next_pickable`` to identify the
    clique-minimal item within a footprint clique — the buffered item whose
    footprint-clique peers all have an equal-or-larger aging key.
    """
    ts = req.merge_first_enqueued_at if req.merge_first_enqueued_at is not None else req.enqueued_at
    return (ts, req.request_id)


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


_HALT_ADVANCE_RESULTS: tuple[str, ...] = (
    'wip_overlap', 'pop_conflict', 'unmerged_state', 'pop_conflict_no_advance',
)
"""``advance_main`` result codes that can trigger a WIP halt.

Shared between :class:`MergeWorker` and :class:`SpeculativeMergeWorker` to
avoid silent divergence: if the set of halt-triggering results ever changes,
updating this single constant propagates to both workers automatically."""


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
        logger.warning(
            'main-health fingerprint composition failed '
            '(category=%s cause_hint=%s probe_sha=%s) — '
            'returning empty fingerprint (dedupe disabled for this outcome)',
            category, cause_hint, probe_sha,
            exc_info=True,
        )
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
    runner: VerifyRunner | None = None,
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

    # γ decision 4: additive runner= param selects the verify host.
    # runner=None (default) → LOCAL-ONLY pool, byte-identical to β (all legacy
    # callers: MergeWorker._do_merge, _reverify_rebased_tree, reverify_member_solo,
    # _do_train_merge, _run_cold_shadow_verify — recovery/train paths stay on the
    # trust anchor and out of slot accounting).
    # runner=<RemoteRunner> → pool=[runner] (no LocalRunner); warm-swap is skipped
    # (per-host persistent worktree — handled by γ's _run_inflight_verify caller).
    # The `quarantine` parameter is reserved/unused here; it remains in the signature
    # so existing call sites stay byte-identical.
    _ = quarantine  # reserved/unused
    if runner is not None:
        # Remote path: build a single-runner pool from the injected runner.
        # Warm-swap runs only for LOCAL leases (caller's responsibility).
        # archive_root mirrors the local path (910-920) so remote stderr lands
        # beside local merge-verify logs for side-by-side operator triage
        # (task 1920, sibling of 1768).
        pool = VerifyRunnerPool(
            [runner],
            event_store=event_store,
            task_id=req.task_id,
            archive_root=req.config.project_root / 'data' / 'verify-logs',
        )
    else:
        # Local path: sole production site for human-facing local merge verify.
        # archive_root mirrors the task-verify convention (workflow.py:4105):
        # durable logs land under data/verify-logs/<task_id>/ so merge-verify
        # and task-verify archives are co-located for operator triage.
        # Cold-shadow (merge_queue.py:9112) and drift (9375/9402) are detective
        # controls that intentionally leave archive_root=None — they are never
        # constructed here and are auto-excluded by LocalRunner's default.
        pool = VerifyRunnerPool(
            [LocalRunner(
                merge_wt, req.config, req.module_configs, task_files_tuple,
                run_scoped=run_scoped_verification,
                run_unscoped=_run_unscoped_typechecks,
                task_id=req.task_id,
                archive_root=req.config.project_root / 'data' / 'verify-logs',
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
        # Append the failure category so the timeout-vs-real-failure
        # distinction is visible in the surviving human-facing signal
        # without requiring log spelunking.  Append-only (after the
        # summary, before the detail block) keeps all existing
        # prefix/substring reason assertions green.
        if verify.category:
            reason = f'{reason} [category: {verify.category}]'
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


async def resolve_attach_action(
    new_tip: str,
    old_tip: str,
    *,
    verifying: bool,
    git_ops: GitOps,
) -> AttachAction:
    """Classify and decide the attach action for a new tip vs an in-flight tip.

    Shared helper used by BOTH the workflow path (workflow.py) and the MCP
    coalesce path (coalesce_or_enqueue_merge_request) so the two paths share
    the same tip-recency *classification decision* and cannot diverge on which
    :class:`AttachAction` class to return.

    Note: the two consumers handle the resulting action differently.  The
    workflow path on RESNAPSHOT re-snapshots and then coalesces the new request
    as a peer waiter onto the in-flight entry (the peer future resolves with the
    in-flight outcome).  The MCP path on RESNAPSHOT re-snapshots and then
    independent-enqueues via :func:`register_and_enqueue_merge_request` — the
    new submission gets its own merge+verify pass; the old in-flight is not
    cancelled.  Both paths on ATTACH_AND_CHAIN independent-enqueue (identical
    behaviour).

    Composes :func:`classify_tip_relation` → :func:`resolve_divergent` →
    :func:`decide_attach_action` in sequence:

    1. Classify the topological relationship between *new_tip* and *old_tip*.
    2. If DIVERGENT, resolve via patch-id content comparison
       (:func:`resolve_divergent`) to either SUBSET or SUPERSET.
    3. Map the resolved relation + *verifying* to an :class:`AttachAction`.

    Returns one of:
    - :attr:`AttachAction.COALESCE` — tips are identical (SAME).
    - :attr:`AttachAction.RESNAPSHOT` — new tip is a SUPERSET and not verifying.
    - :attr:`AttachAction.ATTACH_AND_CHAIN` — new tip is a SUPERSET and verifying.
    - :attr:`AttachAction.ATTACH_CONTAINMENT` — new tip is a SUBSET.
    """
    relation = await classify_tip_relation(new_tip, old_tip, git_ops)
    if relation is TipRelation.DIVERGENT:
        relation = await resolve_divergent(new_tip, old_tip, git_ops)
    return decide_attach_action(relation, verifying=verifying)


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
    *,
    worktree: Path | None = None,
) -> MergeOutcome | None:
    """Terminal outcome when *branch* has no live ref in this repo, else None.

    Classifies a queued *branch* (the bare name, e.g. ``"4011"``) by whether
    its ``{branch_prefix}{branch}`` ref still exists:

        ref present            -> None  (proceed with normal merge)
        ref absent + marker    -> already_merged   (merged then cleaned up)
        ref absent + no marker + worktree HEAD beyond main
                               -> None  (proceed; merge_to_main uses HEAD fallback)
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

    *branch* is resolved via :meth:`GitOps.resolve_queued_branch_ref` which
    tolerates three shapes: bare task id (``'4778'``), already-prefixed
    (``'task/4778'``), and full non-task names (``'cost-min-prd'``).  When a
    live ref is found the method returns ``None`` (proceed).

    For the already-merged check, both candidate forms are tried in priority
    order — ``{branch_prefix}{branch}`` then ``branch`` as-is — because a
    merged+deleted branch has no live ref and the merge-commit subject on main
    reflects whichever form ``merge_to_main`` used at merge time.  Trying both
    ensures prefixed-input callers (``branch='task/4778'``) still classify
    as ``already_merged`` rather than collapsing to ``unknown_branch``.

    The ``unknown_branch`` reason reports both candidate forms tried
    (``{prefixed!r}`` and ``{branch!r}``) so operators can see exactly what
    was submitted and what was attempted during triage.

    **Worktree-HEAD fallback**: when *worktree* is supplied (keyword-only) and
    the ref is absent with no merge marker, BEFORE emitting unknown_branch this
    function checks whether the worktree's HEAD carries commits beyond main via
    :meth:`GitOps.worktree_head_beyond_main` (the shared helper also used by
    :meth:`GitOps.merge_to_main`).  When that helper returns a non-None SHA
    (i.e. HEAD is not an ancestor of main), this function returns ``None``
    (proceed) without emitting a merge_attempt, allowing ``merge_to_main``'s
    worktree-HEAD fallback to land those commits.
    If *worktree* is None, the worktree is unreadable, or HEAD is already an
    ancestor of main, the function falls through to the original
    ``unknown_branch`` outcome — failing safe to today's behavior.
    """
    # ── Live-ref check via canonical resolver ─────────────────────────────
    if await git_ops.resolve_queued_branch_ref(branch) is not None:
        return None  # common case: ref present (bare, prefixed, or full name)

    # ── No live ref: check for an already-merged marker on main ───────────
    # Try both candidate forms in priority order (prefixed wins tie-break).
    # A merged+deleted branch has no live ref, so the resolver returned None;
    # the merge-commit subject is under whichever form merge_to_main used.
    prefixed = f'{git_ops.config.branch_prefix}{branch}'
    for candidate in (prefixed, branch):
        if await git_ops.find_merge_marker(candidate) is not None:
            _emit_merge_attempt(
                event_store, task_id, 'already_merged', duration_ms=_elapsed_ms(t0),
            )
            return MergeOutcome('already_merged')

    # ── Worktree-HEAD fallback: ref absent, no marker, but work is present ─
    # If a worktree was provided and its HEAD carries commits beyond main,
    # this is NOT a misroute — the ref was lost while the work is still live.
    # Return None (proceed) so merge_to_main can merge via the worktree HEAD.
    #
    # Misroute guard: when the worktree HEAD is on an ATTACHED branch (i.e.
    # not in detached-HEAD state), we verify the branch name matches the
    # expected branch before proceeding.  A different attached branch means
    # this is a genuine misroute (wrong worktree was passed) — fall through to
    # unknown_branch in that case.  A detached HEAD indicates the ref was
    # deleted while the work sat as a bare commit (the intended scenario), so
    # it skips the name check and trusts the SHA directly.
    if worktree is not None and worktree.exists():
        rc_sym, sym_ref, _ = await _run(
            ['git', 'symbolic-ref', 'HEAD'], cwd=worktree,
        )
        sym_ref = sym_ref.strip()
        is_detached = (rc_sym != 0)

        # For an attached HEAD, the symbolic ref must end with the expected
        # branch (prefixed or bare).  A mismatch is a misroute — skip fallback.
        branch_matches = is_detached or (
            sym_ref.endswith('/' + prefixed) or sym_ref.endswith('/' + branch)
        )

        # Use the shared helper so the proceed-decision here and the
        # merge-source-selection in merge_to_main share one source of truth
        # for "worktree HEAD carries commits beyond main".  The branch_matches
        # guard short-circuits before the async helper on a misrouted worktree.
        if branch_matches and await git_ops.worktree_head_beyond_main(worktree) is not None:
            # HEAD carries commits beyond main — this is not a misroute.
            # Return None (proceed); merge_to_main uses the worktree-HEAD
            # fallback as the merge source.
            return None

    # ── Genuine misroute ───────────────────────────────────────────────────
    _emit_merge_attempt(
        event_store, task_id, 'unknown_branch', duration_ms=_elapsed_ms(t0),
    )
    return MergeOutcome(
        'unknown_branch',
        reason=f'branch not found in repo (tried {prefixed!r} and {branch!r})',
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

    KNOWN RACE (accepted until γ3 slot-handoff lands):
    :func:`_inflight_entry_is_stale` has a sub-tick window in this code path.
    After gen-1's future resolves as ``'superseded'`` and gen_next is enqueued,
    the registry slot still holds gen-1's request_id until the done-callback
    ``_release`` fires (scheduled via ``call_soon``).  An inbound
    ``merge_request`` arriving in that window sees a registry slot pointing at
    gen-1's id (absent from the live snapshot, which now shows gen_next's id),
    causing :func:`_inflight_entry_is_stale` to judge the slot stale and reap
    it — dispatching a third concurrent request alongside *gen_next* (double
    dispatch).  Resolving this requires γ3's ATTACH_AND_CHAIN to update the
    registry entry's ``request_id`` to gen_next atomically with the enqueue,
    so the slot always tracks the live generation.  While the kill-switch is
    OFF (the production default) this code path is unreachable, so the race is
    accepted.
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


def _inflight_entry_is_stale(
    entry: _InFlightEntry | None,
    branch: str,
    live_snapshot: Callable[[], dict] | None,
) -> bool:
    """Return True when *entry* appears stale relative to the live worker snapshot.

    Liveness is determined by matching ``entry.request_id`` against the
    ``request_id`` fields of snapshot entries.  When ``entry.request_id`` is
    None (legacy entries predating the request_id field) the match falls back
    to branch-name comparison.

    Returns **False** (not stale → keep coalescing) in four cases:

    1. *live_snapshot* is None — no provider wired (back-compat default; all
       existing callers without a worker).
    2. ``live_snapshot()`` raises — transient error (fail-safe to coalesce;
       avoids double-dispatch storms on transient worker hiccups).
    3. The snapshot is not a dict or lacks the ``'entries'`` key — malformed
       response (fail-safe to coalesce; mirrors case 2's philosophy: when the
       snapshot cannot be trusted, assume live rather than risk a double-dispatch).
    4. The entry's ``request_id`` IS present in the snapshot — slot is live.
    """
    if live_snapshot is None:
        return False
    try:
        snap = live_snapshot()
    except Exception:
        return False  # fail-safe: transient error → not stale → coalesce
    if not isinstance(snap, dict) or 'entries' not in snap:
        return False  # malformed snapshot → fail-safe: not stale → coalesce
    entries = snap['entries']
    rid = entry.request_id if entry is not None else None
    if rid is not None:
        return not any(e.get('request_id') == rid for e in entries)
    else:
        # Legacy entry without a request_id: fall back to branch matching.
        return not any(e.get('branch') == branch for e in entries)


async def coalesce_or_enqueue_merge_request(
    queue: asyncio.Queue,
    req: MergeRequest,
    event_store: EventStore | None,
    registry: InFlightMergeRegistry,
    git_ops: _FindInflightWorktreeP | None = None,
    *,
    liveness_secs: int = INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS,
    retention: TerminalOutcomeRetention | None = None,
    live_snapshot: Callable[[], dict] | None = None,
    classifier_git_ops: GitOps | None = None,
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
    entry's callback.  However, when *retention* is supplied and an in-flight
    registry entry is available (and the slot is not reaped as stale; see
    *live_snapshot* below), the coalesced request's ``request_id`` is
    registered as an alias onto the in-flight entry's ``request_id`` via
    :meth:`TerminalOutcomeRetention.record_alias`, so callers polling the
    coalesced id will resolve to the primary outcome once it is recorded.

    *live_snapshot* (keyword-only, default None): a zero-argument callable
    that returns the live worker snapshot dict (same shape as
    ``SpeculativeMergeWorker.snapshot()``).  When provided, the registry
    fast-path **reconciles** the in-memory slot against the live snapshot
    before coalescing: if the slot's ``request_id`` is absent from the
    snapshot the slot is considered stale (the request finalized but its
    slot was not auto-released), it is reaped via
    ``registry.release(branch, detach_waiters=True)``, and the call falls
    through to the acquire-and-enqueue block dispatching a fresh request.
    When absent (None) or when ``live_snapshot()`` raises, the gate
    behaves exactly as today — trust the registry.

    *classifier_git_ops* (keyword-only, default None): when provided AND both
    ``entry.snapshot_tip`` and ``req.snapshot_tip`` are set, the registry
    fast-path classifies the tip relation via :func:`resolve_attach_action`
    before coalescing.  On SUPERSET (the new submission is strictly ahead of
    the in-flight snapshot), the request is independent-enqueued via
    :func:`register_and_enqueue_merge_request` (which enqueues even when the
    slot is held) and ``dispatched=True`` is returned — mirroring workflow.py's
    ATTACH_AND_CHAIN path.  On RESNAPSHOT, :meth:`InFlightMergeRegistry.re_snapshot`
    is also called before enqueueing.  SAME/SUBSET fall through to the existing
    unconditional coalesce.  When absent or either snapshot_tip is None, the
    gate is a no-op and the path preserves current behaviour (back-compat).
    The disk-scan cross-process coalesce branch is intentionally left untouched.
    """
    branch = req.branch

    # ── 1. Registry fast-path ──────────────────────────────────────────
    if registry.is_inflight(branch):
        entry = registry.entry(branch)
        if _inflight_entry_is_stale(entry, branch, live_snapshot):
            # Slot points at a dead request_id (finalize path left the slot
            # un-released).  Reap it so the caller gets a fresh dispatch.
            logger.warning(
                'coalesce_or_enqueue_merge_request: reaping STALE in-flight slot '
                'for branch %r (request_id=%r not present in live worker snapshot); '
                'dispatching fresh',
                branch, entry.request_id if entry else None,
            )
            registry.release(branch, detach_waiters=True)
            # Fall through to the acquire-and-enqueue block below.
        else:
            # ── Tip-recency check (γ2/γ3 consumer wiring) ─────────────────
            # When classifier_git_ops is wired AND both snapshot tips are
            # known, classify the relation using the shared resolve_attach_action
            # helper (same classification decision as workflow.py — the two
            # paths cannot diverge on *which* action class is returned).
            #
            # Action handling is intentionally different from the workflow path:
            #   • RESNAPSHOT: re_snapshot + independent-enqueue via
            #     register_and_enqueue_merge_request (dispatched=True, no alias).
            #     The old in-flight is NOT cancelled or replaced; it continues
            #     and typically lands first.  The second item (the new tip) will
            #     then resolve to already_merged — a redundant but benign extra
            #     merge+verify pass.  The non-blocking MCP path cannot attach the
            #     new request as a peer waiter (as the workflow path does) because
            #     that requires a live Future reference not available in the MCP
            #     call stack.
            #   • ATTACH_AND_CHAIN: independent-enqueue (same as workflow path).
            #   • COALESCE / ATTACH_CONTAINMENT (SAME/SUBSET): fall through to
            #     the unconditional coalesce below (back-compat).
            if (
                classifier_git_ops is not None
                and entry is not None
                and entry.snapshot_tip is not None
                and req.snapshot_tip is not None
            ):
                attach_action = await resolve_attach_action(
                    req.snapshot_tip, entry.snapshot_tip,
                    verifying=entry.verifying,
                    git_ops=classifier_git_ops,
                )
                if attach_action is AttachAction.RESNAPSHOT:
                    registry.re_snapshot(branch, req.snapshot_tip)
                    logger.info(
                        'coalesce_or_enqueue_merge_request: RESNAPSHOT — '
                        'new tip %s is SUPERSET of in-flight %s for branch %r; '
                        'independent-enqueue with snapshot update.',
                        req.snapshot_tip[:12], entry.snapshot_tip[:12], branch,
                    )
                    await register_and_enqueue_merge_request(
                        queue, req, event_store, registry, retention=retention,
                    )
                    return MergeDispatchResult(
                        dispatched=True, in_flight=False, branch=branch,
                    )
                elif attach_action is AttachAction.ATTACH_AND_CHAIN:
                    logger.info(
                        'coalesce_or_enqueue_merge_request: ATTACH_AND_CHAIN — '
                        'new tip %s is SUPERSET of verifying in-flight %s for branch %r; '
                        'independent-enqueue for own merge+verify.',
                        req.snapshot_tip[:12], entry.snapshot_tip[:12], branch,
                    )
                    await register_and_enqueue_merge_request(
                        queue, req, event_store, registry, retention=retention,
                    )
                    return MergeDispatchResult(
                        dispatched=True, in_flight=False, branch=branch,
                    )
                # COALESCE or ATTACH_CONTAINMENT → fall through to coalesce.

            eta = registry.eta_seconds(branch)
            # Coalescing onto a LIVE in-flight entry: register the caller's
            # request_id as an alias onto the primary entry's request_id, so a
            # poll on the coalesced id resolves to the primary terminal outcome
            # (the coalesced request never gets its own terminal record).  Only
            # in this branch — the stale path above reaps the slot and dispatches
            # a fresh request that gets its own terminal record.
            if retention is not None and entry is not None and entry.request_id is not None:
                retention.record_alias(req.request_id, entry.request_id)
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
                # ALIVE: coalesce without enqueuing or reaping.
                # No alias is registered here: the primary request_id belongs to
                # a different process, so there is no in-process registry entry
                # to alias onto.  Callers polling the coalesced id fall through
                # to the event-store / git-authority tiers on a ring miss.
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
    if retention is not None and entry is not None and entry.request_id is not None:
        retention.record_alias(req.request_id, entry.request_id)
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

        # 0. Branch-presence guard: a missing branch ref resolves terminally
        # (unknown_branch on a never-existed ref — typically a mis-routed
        # merge_request — or already_merged when the ref was cleaned up after
        # merge).  Runs before the rev-parse HEAD below so a misroute can't be
        # born as a bare merge_dequeued (phantom in_flight on the dashboard).
        # Pass req.worktree so _classify_branch_presence can distinguish a
        # genuine misroute (no unmerged commits) from a lost-ref-but-work-present
        # scenario where the worktree HEAD carries commits beyond main.
        guard = await _classify_branch_presence(
            self._git_ops, self._event_store, req.task_id, req.branch, t0,
            worktree=req.worktree,
        )
        if guard is not None:
            return guard

        # 1. Already-merged detection (ghost-loop fix)
        _, branch_head, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=req.worktree,
        )
        main_sha = await self._git_ops.get_main_sha()
        # Use snapshot_tip when set: it is the ref THIS request intends to
        # merge, which is drift-proof vs a shared or hijacked lane whose
        # worktree HEAD may have been rebased to an orphaned lineage after
        # snapshotting.  Workflow-path requests leave snapshot_tip=None and
        # retain the existing worktree-HEAD basis (back-compat).
        effective_tip = req.snapshot_tip or branch_head.strip()
        if await self._git_ops.is_ancestor(effective_tip, main_sha):
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


class MergeMetrics:
    """Pure in-memory accumulator for ι=1894 operator metrics.

    Tracks two metrics surfaced on the heartbeat snapshot and dashboard:

    * **retries_per_landing** — (merge re-attempts) / (clean landings).
      Returns ``None`` while ``landings == 0`` to avoid division by zero.
      A "retry" is any CAS round that did not immediately advance main:
      ``cas_failed`` (transient atomic-swap loss) and
      ``rebased_pending_reverify`` (main advanced, fresh re-verify needed)
      both count.  See ``_note_merge_retry`` for the rationale.
    * **drift_at_detection** — how far main advanced (in landed-merge units)
      between when a request was based and when its conflict was caught.
      Stored as a bounded ring of samples; summary exposes count/last/mean/max.

    ``main_position`` is the landings counter — each clean landing advances
    main by at least one merge, so it is a deterministic in-process measure of
    "how far main advanced" without any git calls in the hot path.

    Thread / concurrency safety: SpeculativeMergeWorker runs single-threaded
    under asyncio; all methods are synchronous and side-effect-free in the
    sense that they produce no I/O — safe to call from any asyncio coroutine.

    Args:
        drift_window: Maximum number of drift samples to retain (FIFO ring).
            Oldest samples are dropped when the buffer exceeds this size.
            Default 50 (about one working hour of continuous merge activity).
    """

    def __init__(self, drift_window: int = 50) -> None:
        self._landings: int = 0
        self._retries: int = 0
        self._drift_window = drift_window
        self._drift_samples: collections.deque[int] = collections.deque(
            maxlen=drift_window
        )

    # ── counters ─────────────────────────────────────────────────────────────

    @property
    def landings(self) -> int:
        """Number of clean landings recorded so far."""
        return self._landings

    @property
    def retries(self) -> int:
        """Number of merge retries recorded so far."""
        return self._retries

    @property
    def main_position(self) -> int:
        """Current main position in units of clean landings.

        Equivalent to ``landings`` — each clean landing advances main by at
        least one merge, making this a deterministic proxy for "how far main
        has advanced" without any git calls in the hot path.
        """
        return self._landings

    @property
    def retries_per_landing(self) -> float | None:
        """Retries ÷ landings.  ``None`` when ``landings == 0``."""
        if self._landings == 0:
            return None
        return self._retries / self._landings

    # ── mutators ─────────────────────────────────────────────────────────────

    def record_landing(self) -> None:
        """Increment the clean-landing counter (advances main_position by 1)."""
        self._landings += 1

    def record_retry(self) -> None:
        """Increment the retry counter."""
        self._retries += 1

    def record_drift(self, n: int) -> None:
        """Append one drift sample.  Oldest sample is dropped when buffer full."""
        self._drift_samples.append(n)

    # ── summaries ────────────────────────────────────────────────────────────

    def drift_summary(self) -> dict:
        """Return ``{count, last, mean, max}`` from the retained drift samples.

        All values are ``None`` when no drift has been recorded.
        """
        samples = list(self._drift_samples)
        if not samples:
            return {'count': 0, 'last': None, 'mean': None, 'max': None}
        return {
            'count': len(samples),
            'last': samples[-1],
            'mean': sum(samples) / len(samples),
            'max': max(samples),
        }

    def as_snapshot(self) -> dict:
        """Return the serialisable snapshot dict for embedding in worker.snapshot().

        Shape::

            {
                'retries_per_landing': float | None,
                'drift_at_detection':  {count, last, mean, max},
                'landings_total':      int,
                'retries_total':       int,
            }
        """
        return {
            'retries_per_landing': self.retries_per_landing,
            'drift_at_detection': self.drift_summary(),
            'landings_total': self._landings,
            'retries_total': self._retries,
        }


# ── No-landings circuit-breaker (θ=1893, PRD §5.5) ──────────────────────────

# PROVISIONAL thresholds — PRD §11: calibrate from ι's landings_total metric.
# These are module-level defaults that the constructor can override (tests
# inject small values).  A follow-up task will calibrate from production data.
_NO_LANDINGS_WINDOW_SAMPLES: int = 30  # 30 samples × 60 s = 30 min > worst-case ~25 min
# serialized reify verify, so a healthy slow pipeline registers ≥1 landing in
# the window; only a true 0-landing spiral (with falling disk) trips.
_NO_LANDINGS_DISK_FREE_FLOOR_BYTES: int = 50 * 1024 * 1024 * 1024  # 50 GiB absolute floor


@dataclasses.dataclass(frozen=True)
class BreakerTrip:
    """Immutable decision record returned by NoLandingsCircuitBreaker.observe().

    ``action`` is either ``'halt'`` (trip just fired) or ``'resume'`` (breaker
    just cleared).  The remaining fields carry the window context used to build
    the operator escalation message.

    Fields
    ------
    action          : ``'halt'`` or ``'resume'``
    window_samples  : configured window size (number of samples)
    landings_in_window : **Dual meaning** — on ``action='halt'``: always 0
                        (landings were flat across the window, which is the trip
                        condition); on ``action='resume'``: the delta since trip
                        time (``landings_total − landings_at_trip``), which may
                        span much more than one window when disk-recovery (not a
                        clean landing) triggers resume.  Consumers of resume
                        records should treat this as ``landings_since_trip``.
    free_start      : free-bytes at the START of the evaluated window (on halt)
                      or the free-bytes recorded at trip time (on resume).
    free_end        : free-bytes at the END of the evaluated window (last sample)
    reason          : human-readable one-liner for log/escalation messages
    """

    action: str
    window_samples: int
    landings_in_window: int
    free_start: int
    free_end: int
    reason: str


class NoLandingsCircuitBreaker:
    """Detect and contain the residual merge-churn spiral (PRD §5.5).

    Accumulates a rolling window of (landings_total, free_bytes) samples.
    ``observe()`` returns a ``BreakerTrip`` decision when conditions change,
    or ``None`` when no state transition occurs.

    **Trip conditions (AND)**:
    * rolling ``landing-rate == 0``: ``last.landings == first.landings`` over
      the full ``window_samples``
    * disk strictly falling: every consecutive free_bytes is strictly less than
      its predecessor across the full window
    * disk under pressure: window-end ``free_bytes < disk_free_floor_bytes``
      (trip only fires below the floor; mirrors resume's ``>= floor`` so a
      trip above the floor is impossible and self-resume cannot occur)

    **Resume conditions (OR)**:
    * a clean landing: ``landings_total`` increased above the value at trip time
    * disk recovered: ``free_bytes >= disk_free_floor_bytes`` (absolute floor,
      independent of the trip-level free_bytes)

    **Anti-flap**:
    * sample buffer is CLEARED on every resume — a fresh full window of
      flat+falling samples is required before the next trip can fire

    Args
    ----
    window_samples       : Number of consecutive samples needed to trip.
    disk_free_floor_bytes: Absolute disk-free floor in bytes.  Dispatch
                           auto-resumes once ``free_bytes`` rises back above
                           this floor, regardless of the free-bytes level at
                           which the trip occurred.  Defaults to 50 GiB,
                           aligned with ``warm_lane_min_free_gib`` (50 GiB)
                           admission threshold.

    Notes
    -----
    * Pure / synchronous — no I/O, no asyncio.  Fully unit-testable.
    * Thread safety: SpeculativeMergeWorker runs single-threaded under asyncio;
      all methods here are synchronous and produce no I/O.
    * Tripped state is *sticky*: once ``_tripped`` is True, ``observe()`` only
      returns a ``BreakerTrip`` for the resume transition — not for every call.
    """

    def __init__(
        self,
        window_samples: int = _NO_LANDINGS_WINDOW_SAMPLES,
        disk_free_floor_bytes: int = _NO_LANDINGS_DISK_FREE_FLOOR_BYTES,
    ) -> None:
        self._window_samples = window_samples
        self._disk_free_floor_bytes = disk_free_floor_bytes
        # Rolling buffer of (landings_total, free_bytes) tuples, bounded to
        # window_samples entries (oldest drops automatically via maxlen).
        self._samples: collections.deque[tuple[int, int]] = collections.deque(
            maxlen=window_samples
        )
        self._tripped: bool = False
        # Stashed at trip time for resume comparison
        self._landings_at_trip: int = 0
        self._free_at_trip: int = 0

    # ── public interface ─────────────────────────────────────────────────────

    @property
    def is_tripped(self) -> bool:
        """True when the breaker has fired and dispatch is expected to be halted."""
        return self._tripped

    @property
    def disk_free_floor_bytes(self) -> int:
        """Absolute disk-free floor in bytes.

        The breaker trips only when free-bytes falls strictly below this value
        and auto-resumes once free-bytes rises back to or above it.  Exposed as
        a public property so callers (e.g. Harness escalation messages, capacity
        sanity checks) do not need to reach into the private ``_disk_free_floor_bytes``
        attribute.
        """
        return self._disk_free_floor_bytes

    def reset(self) -> None:
        """Clear tripped state and sample buffer so the breaker re-arms from scratch.

        Used by the harness when it detects that the scheduler was externally
        resumed (i.e. the scheduler is no longer paused but the breaker still
        thinks it is tripped).  After ``reset()``, a fresh full window of
        flat+falling samples is required before the breaker trips again.
        """
        self._tripped = False
        self._landings_at_trip = 0
        self._free_at_trip = 0
        self._samples.clear()

    def observe(self, landings_total: int, free_bytes: int) -> BreakerTrip | None:
        """Feed one sample; return a BreakerTrip decision or None.

        Args
        ----
        landings_total : monotonically non-decreasing landing counter from
                         ``worker.snapshot()['metrics']['landings_total']``.
        free_bytes     : current disk free bytes on the warm-lane volume
                         (``shutil.disk_usage(worktree_base).free``).

        Returns
        -------
        ``BreakerTrip(action='halt', ...)``
            Trip just fired — caller should halt dispatch.
        ``BreakerTrip(action='resume', ...)``
            Breaker just cleared — caller should resume dispatch.
        ``None``
            No state change.
        """
        self._samples.append((landings_total, free_bytes))

        if self._tripped:
            return self._check_resume(landings_total, free_bytes)
        else:
            return self._check_trip()

    # ── private helpers ──────────────────────────────────────────────────────

    def _check_trip(self) -> BreakerTrip | None:
        """Evaluate trip conditions over the last window_samples entries.

        Returns a BreakerTrip(action='halt') when ALL THREE conditions hold:
          1. Landing-rate == 0: landings_total is identical across the window.
          2. Disk strictly falling: every consecutive pair of free_bytes values
             is strictly decreasing (no plateau allowed).
          3. Disk under pressure: the most recent free_bytes is strictly less
             than disk_free_floor_bytes (trip only when disk is already below
             the floor, never while disk is healthy above it).

        The trip/resume floor symmetry is intentional: trip fires when
        ``last_free < disk_free_floor_bytes`` and resume fires when
        ``free_bytes >= disk_free_floor_bytes``, so a trip can only occur
        below the floor and self-resume is structurally impossible (the
        halted host cannot transition from tripped-below to untripped-above
        without passing through the floor).  This eliminates the
        halt+escalate+auto-resume flap that occurs when disk is healthy.

        Returns None otherwise.  Sets internal ``_tripped`` state on first trip.
        """
        if len(self._samples) < self._window_samples:
            return None  # not enough data yet

        # Evaluate over the last window_samples entries
        window = list(self._samples)[-self._window_samples:]
        first_landings, first_free = window[0]
        last_landings, last_free = window[-1]

        # Condition 1: landing-rate == 0 (landings count unchanged)
        landings_flat = (last_landings == first_landings)

        # Condition 2: disk strictly falling (every pair strictly less)
        #
        # Calibration note: the warm-lane admission gate (warm_lane_disk_guard /
        # warm_lane_min_free_gib, default 50 GiB, same as the floor default) throttles
        # new warm-lane worktree acquisitions below the floor.  Because acquisitions are
        # throttled near the floor, the disk fall may plateau around the boundary rather
        # than continuing to fall monotonically, which can break this strict-fall
        # condition.  If production telemetry shows that genuine spirals below 50 GiB
        # fail to produce 30 consecutive strictly-falling samples (because the admission
        # gate keeps disk oscillating at the floor), consider relaxing to
        # non-increasing-with-net-decline: ``window[0][1] > window[-1][1]`` with no
        # individual pair required to be strictly less.  See PRD §11 calibration.
        disk_falling = all(
            window[i + 1][1] < window[i][1]
            for i in range(len(window) - 1)
        )

        # Condition 3: disk under pressure — only trip when window-end free is
        # strictly below the absolute floor (mirrors _check_resume's >= floor).
        # A host with healthy disk above the floor must never trip so it can
        # never immediately self-resume (the flap elimination).
        disk_pressure = last_free < self._disk_free_floor_bytes

        if not (landings_flat and disk_falling and disk_pressure):
            return None

        # All three conditions met — trip
        self._tripped = True
        self._landings_at_trip = last_landings
        self._free_at_trip = last_free

        reason = (
            f'No-landings circuit-breaker tripped: 0 landings over '
            f'{self._window_samples} samples; disk free fell '
            f'{first_free:,} → {last_free:,} bytes'
            f'; free below floor {self._disk_free_floor_bytes:,}'
        )
        return BreakerTrip(
            action='halt',
            window_samples=self._window_samples,
            landings_in_window=0,
            free_start=first_free,
            free_end=last_free,
            reason=reason,
        )

    def _check_resume(self, landings_total: int, free_bytes: int) -> BreakerTrip | None:
        """Check resume conditions (called only while _tripped is True).

        Resume when EITHER (OR semantics, PRD §5.5):
          * clean landing: landings_total > landings-at-trip
          * disk recovered: free_bytes >= disk_free_floor_bytes  (absolute floor)

        The absolute floor is independent of the trip-level free_bytes — it
        resumes once disk is back above the configured floor regardless of where
        the trip occurred.  Anti-flap is provided by the buffer-clear-on-resume:
        a fresh full window of flat+falling samples is required before the next
        trip can fire.

        **Corner case**: if ``disk_free_floor_bytes`` exceeds the volume's total
        capacity, ``free_bytes`` can never reach the floor, so the disk-recovery
        branch is structurally unreachable.  In that case only a clean landing
        can resume.  The Harness logs a one-shot WARNING when it detects this
        condition (floor > total) via ``shutil.disk_usage``.

        On resume, _tripped is cleared so ``observe()`` re-enters the
        trip-detection path.
        """
        clean_landing = landings_total > self._landings_at_trip
        disk_recovered = free_bytes >= self._disk_free_floor_bytes

        if not (clean_landing or disk_recovered):
            return None  # neither condition met — stay tripped

        # Resume — clear the sample buffer so a fresh full window is required
        # before the next trip can fire.  This is the cross-cycle anti-flap
        # guarantee: without the clear, stale pre-resume samples would still be
        # in the deque, reducing the effective window for the next trip
        # evaluation.  (PRD §5.5 "hysteresis to prevent flap".)
        self._tripped = False
        self._samples.clear()
        reason = (
            'No-landings circuit-breaker cleared: '
            + (
                f'landing detected (landings_total={landings_total} > {self._landings_at_trip})'
                if clean_landing
                else (
                    f'disk recovered ({free_bytes:,} >= '
                    f'absolute floor {self._disk_free_floor_bytes:,})'
                )
            )
        )
        return BreakerTrip(
            action='resume',
            window_samples=self._window_samples,
            landings_in_window=landings_total - self._landings_at_trip,
            free_start=self._free_at_trip,
            free_end=free_bytes,
            reason=reason,
        )


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
        merge_store: Any = None,
    ):
        self._git_ops = git_ops
        self._queue = queue
        self._event_store = event_store
        # Post-merge notification hook — called with (task_id, base_sha,
        # advanced_sha) after each 'done' merge.  Wrapped in try/except so a
        # coordinator bug never blocks or fails the merge.  See task 1592.
        #
        # This is also the PRIMARY in-process trigger for the offline
        # deep-test lane (PRD docs/prds/offline-deep-test-lane-worker.md §5
        # C1): harness.py wires this to _note_merge_all, which fans out to
        # _note_offline_lane (task 1951, β1).  The FALLBACK trigger, used for
        # orchestrator-down landings via scripts/land.sh, is reify's
        # hooks/reference-transaction main-move log, which yields the same
        # (base, head) pair.
        self._on_merge_landed = on_merge_landed
        # Durable journal (task 1772): records worker-owned requests on accept,
        # removes them on terminal.  None-safe so bare-worker tests (no store) are
        # unaffected.  Typed Any to keep merge_queue.py free of an import from
        # merge_queue_store.py (one-directional dependency).
        self._merge_store: Any = merge_store
        # Set of request_ids for which a done-callback has already been registered
        # (avoids duplicate removal callbacks on redispatch / chain of the same id).
        self._journaled_request_ids: set[str] = set()
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
        self._reprobe_task: asyncio.Task | None = None
        # In-flight request being processed by the merger loop. Set after
        # dequeue, cleared after the SpeculativeItem is pushed to the verifier
        # queue. Used by stop() to resolve Futures for requests that were
        # mid-processing when shutdown was initiated.
        self._inflight_req: MergeRequest | None = None
        # Vestigial single-host observability fields — write-only after ε.
        # snapshot() no longer reads these; all observability derives from
        # self._inflight (InflightEntry.phase) and self._remerging_item.
        # Retained to avoid a large diff; may be removed in a later cleanup.
        self._verify_item: SpeculativeItem | None = None
        self._verify_phase: str | None = None
        self._verify_started_at: float | None = None
        # Remerge-window observability: set to the MergeRequest being remerged
        # so snapshot() can surface it between queue-pop and _inflight append.
        # Cleared to None immediately after _remerge() returns.
        self._remerging_item: MergeRequest | None = None
        # Finalize-head window observability: set to the InflightEntry that was
        # popped from _inflight by popleft() and passed to _finalize_inflight().
        # Without this, the head is invisible to snapshot() for the entire
        # duration of `await entry.verify_task` inside _finalize_inflight —
        # the same transient-window pattern as _remerging_item and _inflight_req.
        # Set at the top of _finalize_inflight; cleared in its finally clause.
        self._finalizing_head: InflightEntry | None = None
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
        # Per-host RunnerUnavailable streak tracker (task 1795).  Keyed by
        # runner name; entries created on first failure, popped on recovery.
        # Used by _record_runner_unavailable / _record_runner_recovered to fire
        # a dedup'd escalation and auto-reprobe on recovery.  None-safe at call
        # sites: only populated when remote runners are configured.
        self._runner_unavailable: dict[str, _HostUnavailability] = {}
        # Thresholds for firing / probing (copies of config knobs set by
        # _ensure_host_allocator; defaults keep bare-worker tests green).
        # Override in tests to drive threshold logic synchronously (mirrors
        # _heartbeat_interval_s precedent).
        self._unreachable_escalate_after_n: int = 3
        self._unreachable_escalate_after_secs: float = 600.0
        self._reprobe_interval_s: float = 120.0
        # In-flight drift-detective asyncio.Tasks.  asyncio keeps only a WEAK
        # reference to running tasks, so without a strong ref here the drift
        # detective can be GC'd mid-run and a remote-PASS / local-FAIL
        # divergence would go undetected — defeating the safety control.
        # Mirrors the _shadow_compare_tasks pattern (see :func:`_maybe_schedule_shadow_compare`).
        self._drift_check_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]
        # β worker-lifetime host allocator (one slot per host, prefer-local).
        # None until first _ensure_host_allocator(config) call — lazily built
        # because config arrives per-MergeRequest, not at __init__ time.
        self._host_allocator: HostAllocator | None = None
        # γ in-flight deque: ordered list of InflightEntry objects dispatched to hosts.
        # Finalized strictly in submission order (head-first).  Empty when single-host
        # (finalize drains before the next dispatch) → byte-identical serial behaviour.
        self._inflight: collections.deque[InflightEntry] = collections.deque()
        # γ front-priority re-dispatch deque: re-merged/re-dispatched items (chain-
        # invalidation + RunnerUnavailable) go here and are drained before _verifier_queue
        # so they are re-verified in submission order ahead of newer arrivals.
        self._redispatch: collections.deque[SpeculativeItem] = collections.deque()
        # γ cross-iteration state promoted from loop-locals: set by finalize after
        # each head result; read by dispatch to decide chain re-merge.
        # Single-host: byte-identical (deque is always empty at dispatch point).
        self._n_failed: bool = False
        self._remerge_occurred: bool = False
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
        # δ=1988 — the two-layer suffix-conflict graph + bounce state (conflict
        # graph over the unfrozen suffix, debounce signature, cached main SHA,
        # per-branch bounce counter) now live on a SuffixConflictTracker
        # (orchestrator.suffix_graph), owned here as self._suffix_tracker.
        # git_ops/lane_buffers/frozen_prefix/frozen_prefix_tip are ALL injected
        # as narrow callables that RE-READ the live attribute on `self` on
        # every call (not `self` itself, and not a value/bound-method
        # snapshot captured once here) so the tracker is unit-testable
        # without a worker AND stays correct if this worker's `_git_ops` /
        # `frozen_prefix` / `frozen_prefix_tip` is ever reassigned after
        # construction (amend: reviewer robustness_stale_reference +
        # consistency — previously git_ops was a direct snapshot and
        # frozen_prefix/frozen_prefix_tip were bound-method snapshots, an
        # asymmetry with lane_buffers' existing re-reading lambda). Must be
        # constructed AFTER self._lane_buffers above. The 4 @property
        # delegators below (_suffix_conflict_graph, _suffix_conflict_signature,
        # _last_known_main_sha, _bounce_registry) preserve the original
        # attribute names for existing callers/tests;
        # recompute_suffix_conflict_graph() and _bounce_conflicting_suffix_items()
        # are thin delegators to self._suffix_tracker. See suffix_graph.py's
        # module docstring + SuffixConflictTracker docstring for the full
        # debounce/fail-open/TOCTOU/cap contract.
        self._suffix_tracker = SuffixConflictTracker(
            git_ops=lambda: self._git_ops,
            lane_buffers=lambda: self._lane_buffers,
            frozen_prefix=lambda: self.frozen_prefix(),
            frozen_prefix_tip=lambda main_sha: self.frozen_prefix_tip(main_sha),
        )
        # Resume signal: set by every unhalt method so a blocked merger
        # (waiting with no pickable item) wakes up to re-check lanes.
        # Cleared by the merger before each wait; never cancelled.
        self._resume_signal = asyncio.Event()
        # Persistent queue.get() future — kept alive across iterations to
        # avoid the lost-item hazard of cancelling an in-flight get().
        # The merger always has at most one of these outstanding.
        self._pending_get: asyncio.Task | None = None
        # γ persistent getter on _verifier_queue, mirroring _pending_get for the
        # merger queue.  Kept alive across DISPATCH-FILL iterations so the verifier
        # loop can race the next queue item against running verify tasks
        # (asyncio.wait FIRST_COMPLETED) without the lost-item hazard of cancelling
        # a pending get().  At most one outstanding.  Cancelled in stop().
        self._pending_verifier_get: asyncio.Task | None = None
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
        # ── supervisor fields (task 1857) ────────────────────────────────
        # Max restarts allowed per loop within the rolling window before the
        # worker halts and emits a terminal born-at-L2 escalation.
        self._max_loop_restarts: int = 3
        # Rolling window in seconds for counting restarts (time.monotonic).
        self._loop_restart_window_s: float = 300.0
        # Injectable clock for restart-window accounting; mirrors the
        # _maybe_log_queue_heartbeat clock-injection convention.
        self._restart_clock: Callable[[], float] = time.monotonic
        # Per-loop restart timestamp rings (pruned to the rolling window).
        self._loop_restart_times: dict[str, collections.deque[float]] = {
            'merger': collections.deque(),
            'verifier': collections.deque(),
        }
        # Set of currently live loop names ('merger', 'verifier').  The
        # supervisor retires entries from here; when empty (or halted)
        # _loops_finished is set so run() can return.
        self._live_loops: set[str] = set()
        # Event that run() awaits instead of asyncio.gather(): set by the
        # supervisor when all loops have retired normally OR a loop has
        # permanently failed (restart cap exceeded).
        self._loops_finished: asyncio.Event = asyncio.Event()
        # Set True when the supervisor has halted the worker after exhausting
        # the restart cap; run() returns immediately when this is set.
        self._supervisor_halted: bool = False
        # Human-readable reason for the halt; set alongside _supervisor_halted.
        self._supervisor_halt_reason: str | None = None
        # ── ι=1894 live operator metrics ─────────────────────────────────────
        # Pure in-memory accumulator: retries-per-landing + drift-at-detection.
        # Consumed in-process by θ (circuit-breaker) and surfaced on the
        # dashboard via snapshot()['metrics'].  No SQLite persistence needed —
        # θ reads the live readout and the dashboard shows the live snapshot.
        self._merge_metrics: MergeMetrics = MergeMetrics()
        # Per-request drift base: request_id → main_position (landing count) at
        # the moment the request entered the merger.  Popped on clean landing
        # or conflict detection.  Mirrors the _cas_retries / _gate_retries
        # per-task counter-dict lifecycle idiom (:5225-5245).
        self._drift_base: dict[str, int] = {}
        # MQ-invariants eta (task 1992): request-liveness ledger.  Arms an
        # entry per dequeued request and detects a never-resolved Future
        # (silent-hang failure mode) for a loud, dedup'd L1 escalation.
        # None-safe/no external deps, so bare-worker tests are unaffected.
        # See merge_request_ledger.py's module docstring for the full
        # lifecycle contract.
        self._request_ledger = RequestLedger()

    # ── δ=1988 SuffixConflictTracker delegation ─────────────────────────────
    # Data-descriptor properties forwarding the original attribute names to
    # self._suffix_tracker's fields, so existing read/write call sites (incl.
    # tests that do `worker._suffix_conflict_graph = g`) keep working unchanged.

    @property
    def _suffix_conflict_graph(self) -> SuffixConflictGraph:
        return self._suffix_tracker.graph

    @_suffix_conflict_graph.setter
    def _suffix_conflict_graph(self, value: SuffixConflictGraph) -> None:
        self._suffix_tracker.graph = value

    @property
    def _suffix_conflict_signature(self) -> tuple[tuple[str, ...], str] | None:
        return self._suffix_tracker.signature

    @_suffix_conflict_signature.setter
    def _suffix_conflict_signature(self, value: tuple[tuple[str, ...], str] | None) -> None:
        self._suffix_tracker.signature = value

    @property
    def _last_known_main_sha(self) -> str | None:
        return self._suffix_tracker.last_known_main_sha

    @_last_known_main_sha.setter
    def _last_known_main_sha(self, value: str | None) -> None:
        self._suffix_tracker.last_known_main_sha = value

    @property
    def _bounce_registry(self) -> MergeBounceRegistry:
        return self._suffix_tracker.bounce_registry

    @_bounce_registry.setter
    def _bounce_registry(self, value: MergeBounceRegistry) -> None:
        self._suffix_tracker.bounce_registry = value

    # ── β host allocator ──────────────────────────────────────────────────

    def _ensure_host_allocator(self, config: OrchestratorConfig) -> HostAllocator:
        """Lazily build and cache the worker-lifetime HostAllocator.

        Called on first use (config arrives per-MergeRequest; not available at
        __init__ time).  The allocator is built once and reused for the worker's
        lifetime so RemoteRunner instances are CACHED — enabling the
        ``_last_pushed_main_sha`` dedup across consecutive calls.

        Stable cwd: objects are shared across worktrees so the merge-sha push
        is valid from ``git_ops.project_root`` regardless of which worktree
        built the merge commit.

        Caching assumption: ``config.enabled_verify_runners`` (and therefore
        the resolved remote set) is assumed stable for the worker's lifetime.
        If a later MergeRequest carries a different runner list the cached
        allocator silently uses the first-seen set — this is acceptable because
        worker lifetime maps to a single scheduler dispatch session, within
        which runner configuration does not change.

        None-safe: when ``git_ops`` has no ``project_root`` the allocator is
        built transiently (no cache) with an empty remote list so that a later
        call, once ``project_root`` is available, produces the fully-populated
        cached allocator rather than re-using an empty one.

        The allocator shares ``self._runner_quarantine`` by reference so
        HostAllocator-driven (RunnerUnavailable) and DriftDetector-driven
        quarantines share one source of truth.
        """
        if self._host_allocator is not None:
            return self._host_allocator
        project_root = getattr(self._git_ops, 'project_root', None)
        if project_root is not None:
            remotes = _build_remote_runners(
                config, project_root, quarantine=self._runner_quarantine
            )
            self._host_allocator = HostAllocator(
                remotes, quarantine=self._runner_quarantine
            )
            # Copy config knobs onto worker attrs so operator config drives
            # production while __init__ defaults keep bare-worker tests green
            # (mirrors _heartbeat_interval_s precedent).
            self._unreachable_escalate_after_n = (
                config.verify_host_unreachable_escalate_after_n
            )
            self._unreachable_escalate_after_secs = (
                config.verify_host_unreachable_escalate_after_secs
            )
            self._reprobe_interval_s = config.verify_host_reprobe_interval_s
            return self._host_allocator
        # project_root unavailable: return a transient empty-remote allocator
        # without caching so a subsequent call with project_root set builds the
        # fully-populated instance.
        return HostAllocator([], quarantine=self._runner_quarantine)

    # ── per-host RunnerUnavailable streak tracker (task 1795) ────────────

    def _record_runner_unavailable(
        self, host: str, reason: str, now: float
    ) -> bool:
        """Record one RunnerUnavailable failure for *host* and return whether to escalate.

        Creates a new ``_HostUnavailability`` entry on the first call, then
        increments ``streak`` on each subsequent call.  ``first_unavailable_at``
        is fixed to the first-call timestamp so duration can be computed later.
        ``reason`` is refreshed to the most-recent exception message.

        Returns ``True`` when ``streak >= self._unreachable_escalate_after_n``
        (the caller should fire ``_alarm_verify_host_unreachable``), ``False``
        otherwise.  The streak is persistent until ``_record_runner_recovered``
        clears it, so consecutive calls beyond the threshold continue returning
        ``True`` (the alarm helper's ``has_open_l1`` dedup prevents re-submitting).
        """
        if host in self._runner_unavailable:
            entry = self._runner_unavailable[host]
            entry.streak += 1
            entry.reason = reason
        else:
            self._runner_unavailable[host] = _HostUnavailability(
                streak=1,
                first_unavailable_at=now,
                reason=reason,
            )
        return self._runner_unavailable[host].streak >= self._unreachable_escalate_after_n

    def _record_runner_recovered(self, host: str) -> None:
        """Clear the unavailability tracker entry for *host* (idempotent).

        After this call, a subsequent ``_record_runner_unavailable`` starts a
        fresh episode with ``streak=1`` and a new ``first_unavailable_at``.
        """
        self._runner_unavailable.pop(host, None)

    async def _reprobe_quarantined_hosts(self, now: float) -> None:
        """Probe each RU-quarantined remote host and clear on recovery.

        Called periodically by :meth:`_reprobe_loop`.  For each host that is
        **both** in the allocator's quarantine set **and** tracked as
        RunnerUnavailable (``self._runner_unavailable``):

        1. Probes the host via ``runner.health()`` (cheap SSH reachability check).
        2. **On success** (host recovered): clears quarantine, resets the tracker,
           and calls :func:`_clear_verify_host_unreachable` to resolve the open
           L1 and emit a recovery event.
        3. **On failure** (host still unreachable): if
           ``self._unreachable_escalate_after_secs > 0`` **and**
           ``now - entry.first_unavailable_at >= self._unreachable_escalate_after_secs``
           fires the time-based variant of the unreachability alarm (dedup'd via
           ``has_open_l1``).  When *escalate_after_secs* is **0** the time-based
           trip is disabled (streak-only mode); see
           ``OrchestratorConfig.verify_host_unreachable_escalate_after_secs``.
           Probing before alarming avoids a spurious open→immediate-resolve L1
           churn for a host that recovers in the same cycle it would have tripped
           the time-based threshold.

        **Correctness invariant**: hosts quarantined by :class:`DriftDetector`
        for verdict divergence are intentionally skipped — they are in the
        allocator quarantine but absent from ``self._runner_unavailable``.
        Clearing them on mere SSH reachability would bypass the drift parity
        gate (Invariant 5).

        Per-host exceptions are caught so one host's failure cannot abort the
        sweep for the remaining hosts.
        """
        if self._host_allocator is None:
            return

        for name, runner in self._host_allocator.quarantined_remote_runners():
            # Skip divergence-quarantined hosts — not tracked as RunnerUnavailable.
            entry = self._runner_unavailable.get(name)
            if entry is None:
                continue

            try:
                downtime_s = now - entry.first_unavailable_at

                # Probe health first so a host that recovers in the same cycle
                # it would have tripped the time-based threshold goes directly
                # to the recovery path, avoiding a spurious open→immediate-
                # resolve L1 notification.
                if await runner.health():
                    self._host_allocator.clear_quarantine(name)
                    self._record_runner_recovered(name)
                    _clear_verify_host_unreachable(
                        self._escalation_queue,
                        self._event_store,
                        name,
                        downtime_s=downtime_s,
                    )
                else:
                    # Host still unreachable: fire the time-based alarm path
                    # (dedup'd — no-op if already open).
                    # `> 0` guard: secs=0 disables the time-based trip
                    # (streak-only).
                    if self._unreachable_escalate_after_secs > 0 and downtime_s >= self._unreachable_escalate_after_secs:
                        _alarm_verify_host_unreachable(
                            self._escalation_queue,
                            name,
                            entry.reason,
                            streak=entry.streak,
                            duration_s=downtime_s,
                            event_store=self._event_store,
                        )
            except Exception:
                logger.exception(
                    'reprobe_quarantined_hosts: unexpected error probing %r; skipping',
                    name,
                )

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

    async def _release_or_cleanup(
        self, merge_wt: Path | None, *, spec_warm: bool
    ) -> None:
        """Route worktree cleanup: release a warm _spec- lane or run cold cleanup.

        A single shared helper for every cleanup site that may hold a warm
        spec lane, preventing drift.

        When ``spec_warm=True`` and ``merge_wt`` is not None, the lane is a
        member of ``spec_warm_lane_pool`` and must be RELEASED back to FREE
        (retaining target/ for the next assignment) rather than removed from
        disk.  Cold/ephemeral fallback paths (``spec_warm=False``) delegate to
        ``_cleanup_owned_merge_worktree`` which deregisters the ledger entry
        and calls ``git worktree remove``.
        """
        if spec_warm and merge_wt is not None:
            await self._git_ops.release_spec_lane(merge_wt, warm=True)
        else:
            await self._cleanup_owned_merge_worktree(merge_wt)

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

    def _buffer_owned_request(self, item: MergeRequest) -> None:
        """Append *item* to its lane buffer and record it in the durable journal.

        Registers a once-only done-callback on ``item.result`` so the journal
        entry is removed on ANY terminal outcome (done/error/abandoned/
        superseded).  The ``_journaled_request_ids`` set prevents duplicate
        callbacks when the same request_id is re-dispatched (e.g. CAS retry).

        Fail-open: store errors are logged and never propagate so a broken
        journal never stalls the merge pipeline.
        """
        lane = _normalize_lane(item.lane)
        self._lane_buffers[lane].append(item)

        if self._merge_store is not None:
            try:
                self._merge_store.record(item)
            except Exception:  # noqa: BLE001
                logger.warning(
                    'merge_queue: _buffer_owned_request: store.record failed for %s',
                    item.request_id,
                    exc_info=True,
                )

            if item.request_id not in self._journaled_request_ids:
                self._journaled_request_ids.add(item.request_id)

                def _on_terminal(fut: asyncio.Future, *, _rid: str = item.request_id) -> None:  # type: ignore[type-arg]
                    # Decide whether to remove from the durable journal.
                    #
                    # KEEP in journal (return early) when the outcome is from the
                    # graceful-stop / crash path — branch is still live, recovery
                    # should retry:
                    #   • fut.cancelled() — explicitly cancelled future; keep.
                    #   • result.status == 'blocked' AND reason == SHUTDOWN_REASON
                    #     — stop() and the _merger_loop finally block both set this
                    #     exact reason; a true process crash leaves the future
                    #     unresolved so the callback never fires regardless.
                    #
                    # REMOVE from journal for ALL other terminals, including other
                    # 'blocked' outcomes (drop-guard, ENOSPC, merge conflict, merger
                    # error, verification error, chain-generation bound) — those
                    # branches are still live but the failure is deterministic, so
                    # keep-for-recovery would cause indefinite cross-restart retries
                    # and unbounded merge_queue.json growth.
                    #
                    # Prune _journaled_request_ids on removal so the set does not
                    # grow without bound and a later re-dispatch with a fresh Future
                    # can register a new cleanup callback.
                    try:
                        if self._merge_store is None:
                            return
                        if fut.cancelled():
                            return  # explicitly cancelled: keep in journal
                        try:
                            result = fut.result()
                            if (
                                result.status == 'blocked'
                                and result.reason == MERGE_WORKER_SHUTDOWN_REASON
                            ):
                                return  # crash / graceful-stop path: keep for recovery
                        except Exception:  # noqa: BLE001
                            pass  # exception-set future: fall through to remove
                        self._merge_store.remove(_rid)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            'merge_queue: _on_terminal: store.remove failed for %s',
                            _rid,
                            exc_info=True,
                        )
                    # Prune regardless of whether store.remove succeeded: if the
                    # write failed the entry is still semantically terminal, and a
                    # future re-dispatch with a new Future will re-register cleanup.
                    self._journaled_request_ids.discard(_rid)

                item.result.add_done_callback(_on_terminal)

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
            self._buffer_owned_request(item)

    def _pop_next_pickable(self) -> MergeRequest | None:
        """Return the next pickable request using clique-scoped aging (ζ=1891).

        Lane priority is unchanged: high beats normal; halted lanes are skipped.
        Within the first non-empty non-halted lane, pick and remove the
        **clique-minimal** item: the FIFO-earliest buffered item *x* such that
        no footprint-neighbor of *x* (per ``self._suffix_conflict_graph``) in
        the same lane has a strictly smaller :func:`_aging_key`.

        Aging key = ``(merge_first_enqueued_at or enqueued_at, request_id)`` —
        older (smaller timestamp) wins; lexically-smaller request_id breaks ties.

        Degrades to pure FIFO when the conflict graph is empty or the
        request_id is absent from the graph (``footprint_neighbors`` returns
        the empty set → item is vacuously minimal).

        Returns None when every non-empty lane is halted or all buffers are
        empty.  Pure/synchronous so unit tests run without an event loop.
        """
        graph = self._suffix_conflict_graph
        for lane in MERGE_LANES:  # high → normal
            if self.is_lane_halted(lane):
                continue
            buf = self._lane_buffers[lane]
            if not buf:
                continue
            # Clique-minimal selection: scan FIFO order; pick the first item x
            # such that no footprint-neighbor of x in the same buffer has a
            # strictly smaller aging key than x.  Items with no buffered
            # footprint-neighbor are vacuously minimal → pure FIFO for disjoint
            # items.  Degrades to FIFO when graph is empty or rid is absent
            # from the graph (footprint_neighbors returns frozenset()).
            buf_rids: frozenset[str] = frozenset(item.request_id for item in buf)
            for i, x in enumerate(buf):
                neighbors = graph.footprint_neighbors(x.request_id) & buf_rids
                key_x = _aging_key(x)
                # x is clique-minimal iff no same-lane neighbor has key < key_x
                if not any(_aging_key(buf[j]) < key_x
                           for j, item in enumerate(buf)
                           if item.request_id in neighbors):
                    del buf[i]
                    return x
            # Defensive fallback (unreachable: the aging-minimal item is always
            # minimal by the above criterion, so the loop always returns).
            return buf.popleft()
        return None

    # ── ε=1890 frozen-prefix / verify-frontier partition ─────────────────────

    def _frozen_inflight_entries(self) -> list[InflightEntry]:
        """Return ordered list of frozen InflightEntry objects (ε=1890).

        Frozen = currently verifying.  The frozen prefix is the immutable
        head of the pipeline; items here must never be reordered or re-based
        while verification is in flight.

        Definition (§5.3):
          frozen = (self._finalizing_head if its phase is a verify/finalize phase)
                   + [e for e in self._inflight if e.verify_task is not None]

        Passthroughs (verify_task=None: conflict/already_merged/skip) are NOT
        frozen — they carry no merge_commit and are not part of the verify
        frontier.

        Pure/synchronous — mirrors _pop_next_pickable.
        """
        entries: list[InflightEntry] = []
        if (
            self._finalizing_head is not None
            and self._finalizing_head.phase in {'verifying', 'gate_reverify', 'finalizing'}
        ):
            entries.append(self._finalizing_head)
        for e in self._inflight:
            if e.verify_task is not None:
                entries.append(e)
        return entries

    def frozen_prefix(self) -> tuple[str, ...]:
        """Return request_ids of the frozen prefix in submission order (ε=1890).

        The frozen prefix is the immutable head of the pipeline: items that
        are currently in a verify / finalize phase.  Submission order matches
        the finalizing_head (if any) followed by _inflight left-to-right.

        Pure/synchronous (no await).
        """
        return tuple(e.item.request.request_id for e in self._frozen_inflight_entries())

    def unfrozen_suffix(self) -> tuple[str, ...]:
        """Return request_ids of the unfrozen suffix in pick order (ε=1890).

        The unfrozen suffix is the lane-buffer region: items in _lane_buffers
        (high lane before normal, FIFO within each lane).  These may be
        reordered / inserted freely; any reorder triggers a suffix-only
        recompute (recompute_suffix_conflict_graph).

        Pure/synchronous (no await).
        """
        rids: list[str] = []
        for lane in MERGE_LANES:  # high → normal, FIFO within lane
            rids.extend(req.request_id for req in self._lane_buffers[lane])
        return tuple(rids)

    def _newest_frozen_commit(self) -> str | None:
        """Return the newest frozen entry's merge_commit, or None (ε=1890).

        Iterates _frozen_inflight_entries() in reverse (newest first) and
        returns the first non-None merge_commit found.  Entries with no
        merge_result (passthroughs) are skipped.

        Pure/synchronous.
        """
        for entry in reversed(self._frozen_inflight_entries()):
            mr = entry.item.merge_result
            if mr is not None and mr.merge_commit:
                return mr.merge_commit.strip()
        return None

    def frozen_prefix_tip(self, main_sha: str) -> str:
        """Return the base SHA that η should stack bounce/verify onto (ε=1890).

        Returns the newest frozen item's merge_commit (the tip of the frozen
        speculative stack) when the frozen prefix is non-empty, or main_sha
        when the frozen prefix is empty (tip == main when no item is verifying).

        Pure/synchronous (takes main_sha as a parameter to stay await-free,
        mirroring _pop_next_pickable's "Pure/synchronous" pattern so callers
        such as snapshot() and unit tests work without an event loop).
        """
        tip = self._newest_frozen_commit()
        return tip if tip is not None else main_sha

    def check_frozen_prefix_invariant(self, main_sha: str) -> list[str]:
        """Return §5.3 violations as human-readable strings (empty = healthy) (ε=1890).

        λ's integration-gate seam: callers assert violations == [] to gate a
        lane advance, a reorder, or a deploy.

        Checks performed:
          1. Base-chain integrity: for each frozen entry in submission order,
             expected_base = predecessor's merge_commit (or main_sha for the
             first entry).  If the entry has a merge_result AND its base_sha
             does not match expected_base, record a violation naming the rid,
             expected, and actual base.  Entries with no merge_result
             (passthrough / conflict) are skipped — they carry no commit.
          2. Frozen/suffix disjointness: any rid that appears in BOTH
             frozen_prefix() and unfrozen_suffix() is a structural bug; record
             one violation per duplicate.

        Pure/synchronous — reads stored in-memory state, no await.
        """
        violations: list[str] = []
        frozen_entries = self._frozen_inflight_entries()

        # ── 1. Base-chain integrity ───────────────────────────────────────────
        expected_base = main_sha.strip()
        for entry in frozen_entries:
            rid = entry.item.request.request_id
            mr = entry.item.merge_result
            if mr is None or not mr.merge_commit:
                # No merge_commit — nothing to chain; advance expected_base
                # only if there IS a commit to chain to (skip for passthroughs).
                continue
            actual_base = entry.item.base_sha.strip() if entry.item.base_sha else ''
            if actual_base != expected_base:
                violations.append(
                    f'frozen-prefix base-chain broken at {rid}: '
                    f'expected base {expected_base!r} but item has {actual_base!r}'
                )
            # Advance expected_base for the next entry regardless of violation,
            # so subsequent chain errors are also surfaced (not shadowed).
            expected_base = mr.merge_commit.strip()

        # ── 2. Frozen/suffix disjointness ────────────────────────────────────
        frozen_set = set(self.frozen_prefix())
        suffix_set = set(self.unfrozen_suffix())
        for rid in sorted(frozen_set & suffix_set):
            violations.append(
                f'frozen-prefix disjointness violated: {rid!r} appears in both '
                f'frozen prefix and unfrozen suffix'
            )

        return violations

    def two_layer_invariants(self, main_sha: str) -> list[str]:
        """Return ALL §5.3 two-layer violations as human-readable strings (λ=1895).

        Empty list → all §5.3 invariants hold.  Non-empty → each string names
        the offending ``request_id`` (or pair) and the violated invariant.

        Composes :meth:`check_frozen_prefix_invariant` (base-chain integrity +
        frozen∩suffix disjointness) with three additional checks:

        (i)  **textual⊆footprint contract**: every edge in
             ``_suffix_conflict_graph.textual_edges`` must also be in
             ``footprint_edges`` (β/γ contract: textual conflict ⇒ footprint
             overlap).  A pair in ``textual_edges`` but not in
             ``footprint_edges`` is a graph-consistency violation naming both
             request_ids.

        (ii) **conflicts_with_main ⊆ nodes**: every ``request_id`` in
             ``_suffix_conflict_graph.conflicts_with_main`` must be a node in
             ``_suffix_conflict_graph.nodes``.  A rid present in
             ``conflicts_with_main`` but absent from nodes is a δ=1889
             graph-consistency violation naming the stale rid.

        Pure / synchronous (no await).  Fail-safe: never raises; any
        unexpected exception inside a sub-check is caught and surfaced as a
        violation string so the caller always receives a ``list[str]``.

        Mirrors :meth:`check_frozen_prefix_invariant`'s ``-> list[str]``
        idiom and gives the operator dashboard a single §5.3 health surface
        (PRD §1).  Used by λ's integration test and the 'two_layer_invariants'
        snapshot key.
        """
        try:
            # ── inherited checks (base-chain + frozen∩suffix disjointness) ────
            # Skip when main_sha is unavailable ('unknown' sentinel from
            # snapshot()) to avoid spurious base-chain violations at startup
            # or after a get_main_sha() failure.  Graph-consistency checks
            # (i)+(ii) below still run — they don't need main_sha.
            if main_sha and main_sha != 'unknown':
                violations: list[str] = self.check_frozen_prefix_invariant(main_sha)
            else:
                violations = []
        except Exception as exc:  # pragma: no cover — defensive
            violations = [f'two_layer_invariants: check_frozen_prefix_invariant raised: {exc}']

        graph = self._suffix_conflict_graph

        # ── (i) textual_edges ⊆ footprint_edges ──────────────────────────────
        try:
            spurious = graph.textual_edges - graph.footprint_edges
            for edge in sorted(sorted(e) for e in spurious):
                rid_a, rid_b = edge[0], edge[1]
                violations.append(
                    f'textual⊈footprint contract violated: textual edge '
                    f'{rid_a!r}↔{rid_b!r} has no footprint-overlap counterpart'
                )
        except Exception as exc:  # pragma: no cover — defensive
            violations.append(f'two_layer_invariants: textual⊆footprint check raised: {exc}')

        # ── (ii) conflicts_with_main ⊆ nodes ─────────────────────────────────
        try:
            nodes_set = frozenset(graph.nodes)
            stale_rids = sorted(graph.conflicts_with_main - nodes_set)
            for rid in stale_rids:
                violations.append(
                    f'conflicts_with_main⊄nodes: {rid!r} is in conflicts_with_main '
                    f'but absent from suffix graph nodes'
                )
        except Exception as exc:  # pragma: no cover — defensive
            violations.append(
                f'two_layer_invariants: conflicts_with_main⊆nodes check raised: {exc}'
            )

        return violations

    def _warn_if_verify_base_not_frozen_tip(
        self,
        item: SpeculativeItem,
        main_sha: str,
    ) -> None:
        """Log a WARNING when a real-verify dispatch base is not the frozen tip (ε=1890).

        §5.3 invariant: a verify may only start against a base that is the tip
        of the frozen prefix.  This guard detects "verify against a
        speculative-only base" and surfaces it as a WARNING for observability
        (λ integration gate / production debugging) WITHOUT changing control
        flow, raising, or mutating any state.

        Called from _dispatch_item immediately before launching the real-verify
        task.  main_sha must be the caller's already-fetched get_main_sha()
        result (fail-open: the guard is skipped entirely on get_main_sha error,
        so this method is never reached with a stale/empty sha argument).

        Entries with no merge_result (passthrough/conflict) are excluded:
        they carry no merge_commit and are not part of the frozen prefix.

        Pure/synchronous (no await).  Returns None in all cases.
        """
        if item.merge_result is None or not item.base_sha:
            return  # passthrough / conflict — not a real-verify candidate
        expected_tip = self.frozen_prefix_tip(main_sha)
        if item.base_sha.strip() == expected_tip.strip():
            return  # invariant holds
        logger.warning(
            'ε=1890 §5.3 guard: task %s (rid=%s) dispatched for real-verify '
            'against a base (%r) that is NOT the frozen-prefix tip (%r); '
            'verify_depth=%d.  This may indicate a verify against a '
            'speculative-only base.  Control flow is unchanged (log-only guard).',
            item.request.task_id,
            item.request.request_id,
            item.base_sha,
            expected_tip,
            len(self.frozen_prefix()),
        )

    async def recompute_suffix_conflict_graph(self) -> None:
        """Thin delegator (task δ=1988) — see :meth:`SuffixConflictTracker.recompute`
        (orchestrator.suffix_graph) for the full debounce / fail-open /
        textual-pruning contract."""
        await self._suffix_tracker.recompute()

    async def _bounce_conflicting_suffix_items(self) -> None:
        """Thin delegator (task δ=1988) — see
        :meth:`SuffixConflictTracker.bounce_conflicting_suffix_items`
        (orchestrator.suffix_graph) for the full cap/escalation/TOCTOU
        contract.  Called by :meth:`_acquire_next_request` immediately after
        ``recompute_suffix_conflict_graph()`` and before ``_pop_next_pickable()``."""
        await self._suffix_tracker.bounce_conflicting_suffix_items()

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

            # δ/1889 — refresh the conflict-graph over the unfrozen suffix.
            # The debounce (signature check) makes repeated calls cheap when
            # neither the suffix composition nor main_sha has changed.
            # Note: the speculative-merge prefetch path may observe a slightly
            # stale graph for items dispatched before the next acquire; that
            # staleness window is bounded by one acquire cycle and is
            # intentional (ε/ζ/η consume the relation, not the merge outcome).
            await self.recompute_suffix_conflict_graph()

            # η/1892 — graph-time bounce: divert suffix items that conflict with
            # the frozen-prefix tip before they can consume a verify slot.
            # Bounced items are either re-queued (clean rebase) or escalated
            # (real conflict / cap exceeded); either way they are NOT returned
            # by _pop_next_pickable() this cycle.
            await self._bounce_conflicting_suffix_items()

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
                    self._buffer_owned_request(item)
            # Loop to try _pop_next_pickable again (maybe the resume unblocked a lane).

    # ── ι=1894 metric wiring helpers ─────────────────────────────────────────

    def _note_merge_started(self, request_id: str) -> None:
        """Stash the current main_position for a request entering the merger.

        Called when a MergeRequest is dequeued and about to be merged.
        Records the landing count at merge-start so that if the merge later
        conflicts we can compute drift = main_position_now − stashed_base.
        """
        self._drift_base[request_id] = self._merge_metrics.main_position

    def _note_merge_landing(self, request_id: str) -> None:
        """Record a clean landing and remove the stashed drift base.

        Called inside the ``outcome.status == 'done'`` clean-landing branch.
        Increments the landings counter (advancing main_position by 1) and
        pops any _drift_base entry for this request.
        """
        self._merge_metrics.record_landing()
        self._drift_base.pop(request_id, None)

    def _note_merge_retry(self) -> None:
        """Record a merge retry (CAS failure or gate reverify).

        Called at the ``result == 'rebased_pending_reverify'`` branch and at
        the ``result == 'cas_failed'`` retry path.

        **Retry definition**: ``retries_per_landing`` counts every incremental
        CAS or gate round: both ``cas_failed`` (transient atomic-swap loss) and
        ``rebased_pending_reverify`` (main advanced under the request; needs a
        fresh re-verify against the new tip).  Both represent additional merge
        work driven by contention and map naturally to the operator's question
        "how often does this queue re-attempt merges?"  If you need to
        distinguish rebase-rebounded reverifies from pure CAS races, use
        ``retries_total`` (the raw counter) alongside the two
        :attr:`MergeMetrics._gate_retries`/:attr:`_cas_retries` per-task dicts
        already maintained by the worker.
        """
        self._merge_metrics.record_retry()

    def _note_conflict_detected(self, request_id: str) -> None:
        """Record a drift sample when a merge conflict is detected.

        Computes drift = current main_position − stashed_base (the landing
        count at merge-start), records it as a drift sample, and pops the
        _drift_base entry.  Defensive: a missing entry (request not started
        via _note_merge_started) produces a drift of 0 rather than raising.
        """
        base = self._drift_base.pop(request_id, self._merge_metrics.main_position)
        drift = self._merge_metrics.main_position - base
        self._merge_metrics.record_drift(drift)

    def snapshot(self) -> dict:
        """Return a synchronous read-only snapshot of the merge worker pipeline state.

        Safe to call from any context (no await, no lock) because asyncio's
        single-loop model ensures in-memory reads are non-interleaved.

        Returns a dict with:
          entries: list of entry dicts, head-of-line first.
          depth: total number of entries.
          head_of_line: task_id of the first entry, or None.
          verify_in_progress: {task_id, phase, age_secs, verify_age_secs} when the
            deque head is actively verifying (phase in verifying/gate_reverify/finalizing),
            else None.  Passthrough entries (no verify task) produce None.
            verify_age_secs measures time since dispatch, which includes host-acquisition
            latency — it is NOT pure verify time.
          occupancy: {hosts_total, hosts_busy, by_host} — per-host in-flight count.
          is_wip_halted: bool.
          halt_owner_esc_id: str or None.

        Each entry dict contains:
          task_id, branch, state, enqueued_at, age_secs, position,
          waiter_alive, worktree, pre_rebased, request_id, lane.
          host, verify_started_at, verify_age_secs — non-None only on _inflight entries.
        State values: queued, merging, remerging, awaiting_verify, verifying,
          passthrough, gate_reverify, finalizing.
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
                # Uniform schema: present on all entries; non-None only on _inflight entries.
                'host': None,
                'verify_started_at': None,
                'verify_age_secs': None,
            }

        def _infl_entry(infl: InflightEntry, position: int) -> dict:
            """Build an entry dict for an in-flight (or finalizing) entry.

            Fills host/verify_started_at/verify_age_secs — fields shared by
            section 0 (_finalizing_head) and section 1 (_inflight loop).
            """
            e = _entry(
                infl.item.request,
                infl.phase or 'verifying',
                worktree_path=infl.merge_wt,
                position=position,
            )
            e['host'] = infl.lease.name if infl.lease is not None else None
            e['verify_started_at'] = infl.started_at
            e['verify_age_secs'] = (
                max(0.0, now - infl.started_at)
                if infl.started_at is not None else None
            )
            return e

        # 0. Finalize-head window: item popped from _inflight for finalization but
        # not yet complete.  Prepended at position 0 so it remains head-of-line —
        # it is the submission-order head.  Mirrors _remerging_item (section 1b)
        # and _inflight_req (section 3) — same transient-window side-field pattern.
        if self._finalizing_head is not None:
            entries.append(_infl_entry(self._finalizing_head, 0))

        # 1. In-flight verify entries: iterate self._inflight head-first.
        # self._inflight is the sole source of truth for concurrent-verify state.
        # The singular self._verify_item/_verify_phase are no longer read here
        # (they are set but never cleared after γ, causing a stale phantom entry).
        # Each entry carries host (from lease.name), started_at (dispatch time ≈
        # verify start), and phase (per-entry authoritative source under multi-host).
        for _infl in self._inflight:
            entries.append(_infl_entry(_infl, len(entries)))

        # 1b. Remerge-window entry: item popped from queue and being remerged
        # but not yet appended to _inflight.  Without this, the item is invisible
        # to all observability during the await self._remerge(...) call.
        if self._remerging_item is not None:
            entries.append(_entry(
                self._remerging_item, 'remerging',
                worktree_path=None,
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

        # verify_in_progress: non-None only when the deque head is actively
        # verifying or in a post-verify gate phase.  Passthrough entries
        # (phase='passthrough') produce None here — no verify task is running.
        # Includes 'phase' so consumers can distinguish verifying vs. gate_reverify
        # vs. finalizing without misreading the presence of this field.
        # Prefer _finalizing_head when set — it is the true submission-order head
        # (popped from _inflight); self._inflight[0] is the SECOND entry during
        # the finalize window and would misreport the verifying head.
        verify_in_progress = None
        _verify_phases = {'verifying', 'gate_reverify', 'finalizing'}
        _fh = self._finalizing_head
        # Prefer _finalizing_head only when its phase qualifies — a passthrough
        # finalize entry would otherwise mask a genuinely-verifying _inflight[0].
        _vip_head = (
            _fh if _fh is not None and _fh.phase in _verify_phases
            else (self._inflight[0] if self._inflight else None)
        )
        if _vip_head is not None and _vip_head.phase in _verify_phases:
            verify_in_progress = {
                'task_id': _vip_head.item.request.task_id,
                'phase': _vip_head.phase,
                # age_secs: total time since enqueued (queue wait + verify time).
                'age_secs': max(0.0, now - _vip_head.item.request.enqueued_at),
                # verify_age_secs: time since dispatch (includes host acquisition).
                # This is NOT pure verify time — see started_at on InflightEntry.
                'verify_age_secs': (
                    max(0.0, now - _vip_head.started_at)
                    if _vip_head.started_at is not None else None
                ),
            }

        # occupancy: per-host in-flight breakdown for heartbeat and dashboard consumers.
        _by_host = {
            _infl.lease.name: _infl.item.request.task_id
            for _infl in self._inflight
            if _infl.lease is not None
        }
        # Include the finalizing head (if any) — it is the submission-order head
        # and its host slot is still occupied while _finalize_inflight awaits.
        # Inserted head-first so 'local' leads the dict, matching position 0 in entries.
        if self._finalizing_head is not None and self._finalizing_head.lease is not None:
            _fh_name = self._finalizing_head.lease.name
            _fh_tid = self._finalizing_head.item.request.task_id
            _by_host = {_fh_name: _fh_tid, **_by_host}
        _hosts_total = (
            len(self._host_allocator.host_names)
            if self._host_allocator is not None else 1
        )
        occupancy = {
            'hosts_total': _hosts_total,
            'hosts_busy': len(_by_host),
            'by_host': _by_host,
        }

        return {
            'entries': entries,
            'depth': len(entries),
            'head_of_line': entries[0]['task_id'] if entries else None,
            'verify_in_progress': verify_in_progress,
            'is_wip_halted': self.is_wip_halted,
            'halt_owner_esc_id': self.halt_owner_esc_id,
            'occupancy': occupancy,
            # δ/1889 additive key: per-suffix conflict relation (backward-compatible).
            # Populated by recompute_suffix_conflict_graph() after each drain.
            # Read here synchronously — no await; the expensive async build is
            # decoupled from the read (snapshot() stays non-blocking).
            'suffix_conflict_graph': self._suffix_conflict_graph.to_snapshot_dict(),
            # ι=1894 additive key: live operator metrics.
            # Populated by the _note_merge_* helpers (wired at existing landing/
            # retry/conflict code points).  Pure synchronous read — no await,
            # no git calls.  No collision with existing keys (entries/depth/
            # head_of_line/verify_in_progress/occupancy/is_wip_halted/
            # halt_owner_esc_id/suffix_conflict_graph).
            'metrics': self._merge_metrics.as_snapshot(),
            # ε=1890 additive key: frozen-prefix / verify-frontier partition.
            # Populated by the pure frozen_prefix() / _newest_frozen_commit()
            # accessors — pure synchronous read, no await, no git calls.
            # No collision with existing keys (entries/depth/head_of_line/
            # verify_in_progress/occupancy/is_wip_halted/halt_owner_esc_id/
            # suffix_conflict_graph/metrics).
            'frozen_prefix': {
                'request_ids': list(self.frozen_prefix()),
                'tip_merge_commit': self._newest_frozen_commit(),
                'verify_depth': len(self.frozen_prefix()),
            },
            # λ=1895 additive key: consolidated §5.3 health surface.
            # Populated by two_layer_invariants() — pure synchronous read, no
            # await.  Empty list = healthy; non-empty = violation strings.
            # main_sha is best-effort: use the REAL main SHA cached at the last
            # recompute_suffix_conflict_graph() or finalize call; fall back to
            # 'unknown' when genuinely unavailable so the method is always
            # callable without an event loop.  Do NOT use _newest_frozen_commit()
            # here — that returns the frozen-stack tip (M1), not the real main
            # (M0), causing a spurious base-chain violation during normal
            # in-flight verify (λ=1895 fix).
            # No collision with existing keys.
            'two_layer_invariants': self.two_layer_invariants(
                self._last_known_main_sha or 'unknown'
            ),
        }

    def _check_request_liveness(self, now: float, *, threshold_s: float | None = None) -> None:
        """Sweep the request-liveness ledger and alarm on any entry stuck past threshold.

        Synchronous and clock-injectable (``now``/``threshold_s``) — mirrors
        :meth:`_maybe_log_queue_heartbeat`'s clock-injection convention so
        tests can drive the stuck boundary deterministically without real
        time.

        MQ-invariants eta (task 1992): OBSERVATION + ESCALATION ONLY (PRD
        design decision 4) — never resolves a Future, never mutates
        ``_queue``/``_inflight``, never halts the pipeline.  See
        ``merge_request_ledger.py``'s module docstring for the ledger
        lifecycle this sweeps, and
        :func:`orchestrator.merge_request_ledger._alarm_merge_request_stuck`
        for the dedup'd-escalation contract.

        ``threshold_s`` defaults to ``None``, resolved in-body to 1.5 ×
        :data:`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` (~4.5h — comfortably
        past the cold-verify ceiling, so anything still unresolved past it is
        definitively stuck).  Resolved via the same deferred reach-back
        import used throughout ``merge_liveness.py`` so a test's
        string-path monkeypatch of
        ``orchestrator.merge_queue.INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS``
        stays effective.
        """
        if threshold_s is None:
            from orchestrator.merge_queue import INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS

            threshold_s = 1.5 * INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS

        stuck_list = self._request_ledger.stuck_entries(now, threshold_s)
        if not stuck_list:
            return

        phase_by_request_id = {
            entry['request_id']: entry['state'] for entry in self.snapshot()['entries']
        }

        for stuck in stuck_list:
            phase = phase_by_request_id.get(stuck.request_id, 'unowned')
            stuck = dataclasses.replace(stuck, phase=phase)
            logger.warning(
                'merge request stuck: request_id=%s task=%s branch=%s age=%.0fs '
                'phase=%s — dequeued but never resolved (silent-hang failure mode)',
                stuck.request_id, stuck.task_id, stuck.branch, stuck.age_secs, stuck.phase,
            )
            _alarm_merge_request_stuck(
                self._escalation_queue, stuck, event_store=self._event_store,
            )

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

        occ = snap['occupancy']
        _occ_suffix = ''
        if occ['by_host']:
            _host_parts = ' '.join(
                f'{h}={tid}' for h, tid in occ['by_host'].items()
            )
            _occ_suffix = (
                f' | verifying {occ["hosts_busy"]}/{occ["hosts_total"]} hosts: '
                f'{_host_parts}'
            )

        logger.info(
            'merge queue heartbeat: %d in pipeline, oldest age=%.0fs, '
            'head=task %s (state=%s, age=%.0fs)%s',
            snap['depth'], oldest_age,
            head['task_id'], head['state'], head['age_secs'],
            _occ_suffix,
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
                    'occupancy': occ,
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

    async def _reprobe_loop(self) -> None:
        """Periodically probe quarantined remote hosts and clear on recovery.

        Runs independently of the merger/verifier loops so it fires even when
        those are idle.  Wakes every ``_reprobe_interval_s`` seconds and delegates
        to the clock-injectable :meth:`_reprobe_quarantined_hosts`.

        Any unexpected exception is logged and swallowed (same pattern as
        _heartbeat_loop) so a probe bug can never crash the worker.
        """
        while self._running:
            await asyncio.sleep(self._reprobe_interval_s)
            try:
                await self._reprobe_quarantined_hosts(time.time())
            except Exception:
                logger.exception('reprobe_quarantined_hosts: unexpected error in loop')

    # ── supervisor (task 1857) ───────────────────────────────────────────

    def _spawn_loop(self, name: str) -> asyncio.Task:  # type: ignore[type-arg]
        """Create (or re-create) the named loop task and register the supervisor callback.

        Stores the new task in ``self._merger_task`` or ``self._verifier_task`` so the
        current task reference is always up-to-date.  The done callback calls
        ``_on_loop_task_done`` on every completion so the supervisor can detect
        unexpected deaths and either restart or halt.

        **Verifier restart — in-flight state survival (task 1857 design decision)**

        On a verifier restart, this method does NOT clear ``self._inflight``,
        ``self._redispatch``, ``self._n_failed``, ``self._remerge_occurred``, or
        ``self._pending_verifier_get``.  These are *instance* attributes that outlive
        the dead loop task.  The restarted ``_verifier_loop`` naturally resumes
        draining them:

        * **DISPATCH-FILL** drains ``_redispatch`` first (front-priority re-dispatch
          queue), then ``_verifier_queue``, so surviving re-dispatch entries are
          re-verified in submission order ahead of any new arrivals.
        * **FINALIZE-HEAD** pops ``_inflight`` and calls ``_finalize_inflight``, whose
          ``if not req.result.done()`` guard makes re-finalization idempotent — an
          entry that was mid-finalize when the loop died cannot be double-resolved.

        Therefore surviving in-flight entries are neither lost nor double-finalized.
        The supervisor restart path **must not** clear those deques.  Finalize-or-requeue
        is owned by the resumed loop, not the supervisor.

        Args:
            name: ``'merger'`` or ``'verifier'``.

        Returns:
            The newly-created ``asyncio.Task``.
        """
        if name == 'merger':
            task: asyncio.Task = asyncio.create_task(self._merger_loop())  # type: ignore[type-arg]
            self._merger_task = task
        elif name == 'verifier':
            task = asyncio.create_task(self._verifier_loop())  # type: ignore[type-arg]
            self._verifier_task = task
        else:
            raise ValueError(f'Unknown loop name: {name!r}')
        task.add_done_callback(lambda t: self._on_loop_task_done(name, t))
        return task

    def _retire_loop(self, name: str) -> None:
        """Mark a loop as no longer live; set ``_loops_finished`` when all have retired.

        Called by the supervisor on any completion path (normal shutdown, cancellation,
        or terminal cap-exceeded halt).  Sets ``_loops_finished`` as soon as all loops
        have retired OR the worker has been supervisor-halted so ``run()`` can return.
        """
        self._live_loops.discard(name)
        if not self._live_loops or self._supervisor_halted:
            self._loops_finished.set()

    def _on_loop_task_done(self, name: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
        """Done callback for supervised loop tasks (merger and verifier).

        Classifies the completion and takes the appropriate action:

        * **Cancelled** → teardown; ``_retire_loop`` without escalating.
        * **Any completion while ``_running`` is False** → shutdown race or normal
          shutdown; ``_retire_loop`` without escalating.  stop() sets ``_running=False``
          BEFORE draining, so both the clean-sentinel exit and any late exception on the
          shutdown path hit this branch.
        * **Clean return while ``_running`` is True** → synthetic unexpected death; a
          supervised loop must not exit before stop() sets ``_running=False``.  Possible
          cause: stray shutdown sentinel (step-10 guards the known source).  Escalate
          and restart as if a real exception was thrown.
        * **Exception + ``_running`` is True** → unexpected death; emit a loud L1
          escalation then ``_spawn_loop(name)`` to restart (bounded by the cap).

        Calling ``task.exception()`` is also the **fix for the silent-death
        suppression**: without retrieval, asyncio would log "Task exception was never
        retrieved" at GC time — but because the run() task holds a strong reference to
        the gather result, GC never fires and the exception is silently lost.  This
        callback retrieves it unconditionally.
        """
        if task.cancelled():
            # Teardown cancellation — not an unexpected death.
            self._retire_loop(name)
            return
        # Retrieve the exception (kills the "never retrieved" suppression).
        exc = task.exception()
        # Shutdown guard FIRST: any completion (clean or late exception) while the
        # worker is shutting down is retired without escalation.  stop() sets
        # self._running=False BEFORE draining queues and itself pushes sentinels, so
        # the normal shutdown path (merger/verifier exit via the None sentinel) always
        # satisfies `not self._running` — step-5 part A/C assertions hold.
        if not self._running:
            self._retire_loop(name)
            return
        # Synthetic-death: a loop that returns cleanly (exc is None) while
        # self._running is True is anomalous — no supervised loop should exit before
        # stop() sets _running=False.  Possible cause: a stray shutdown sentinel (the
        # step-10 fix in _merger_loop's finally guards the known source; this is
        # belt-and-suspenders for any future regression).  Treat it as an unexpected
        # death: restart and escalate, honoring the "never fail silent" mandate.
        if exc is None:
            exc = RuntimeError(
                f'{name} loop returned cleanly while worker still running'
                ' — possible stray shutdown sentinel or unexpected clean exit'
            )
        # Unexpected death while worker is running: check restart cap then escalate.
        now = self._restart_clock()
        times = self._loop_restart_times[name]
        window = self._loop_restart_window_s
        # Prune restart timestamps outside the rolling window.
        while times and now - times[0] > window:
            times.popleft()
        if len(times) >= self._max_loop_restarts:
            # Bookkeeping first: guarantee run() can return even if escalation
            # delivery raises (asyncio would log it and the done-callback would exit
            # before the state mutations below, leaving _loops_finished un-set and
            # run() hanging forever).
            self._supervisor_halted = True
            self._supervisor_halt_reason = (
                f'{name} loop exceeded {self._max_loop_restarts} restarts '
                f'within {window}s'
            )
            self._retire_loop(name)  # sets _loops_finished → run() returns
            # Emit after bookkeeping: a submit() failure is logged by asyncio but
            # can no longer silently prevent the worker from halting.
            self._emit_loop_terminal_escalation(name, exc, len(times))
        else:
            # Within cap: record timestamp and respawn first so the worker is live
            # again even if escalation delivery fails.
            times.append(now)
            self._spawn_loop(name)
            self._emit_loop_death_escalation(name, exc)

    def _submit_loop_escalation(
        self,
        name: str,
        exc: BaseException,
        *,
        level: int,
        severity: str,
        summary: str,
        detail_prefix: str,
        tb_label: str = 'Full traceback',
        suggested_action: str,
        id_key: str,
    ) -> None:
        """None-safe helper: build and submit a loop-supervisor escalation.

        Centralises the None-guard, local ``escalation.models`` import, traceback
        rendering, ``Escalation`` construction, and ``submit`` call that were
        previously duplicated verbatim across
        ``_emit_loop_death_escalation`` and ``_emit_loop_terminal_escalation``.

        Args:
            name: Loop name (``'merger'`` or ``'verifier'``), embedded in the detail.
            exc: The exception from the dead loop task (used for traceback rendering).
            level: Escalation level (1 for restart, 2 for terminal born-at-L2).
            severity: ``'blocking'`` (L1) or ``'critical'`` (L2).
            summary: One-line summary string.
            detail_prefix: Multi-line block placed before the traceback section.
            tb_label: Label for the traceback section; defaults to ``'Full traceback'``.
            suggested_action: Operator instructions.
            id_key: Unique key used for both ``make_id`` and ``task_id``.
        """
        if self._escalation_queue is None:
            return
        from escalation.models import Escalation  # local import — escalation optional dep
        tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        detail = f'{detail_prefix}\n\n{tb_label}:\n{tb_str}'
        esc = Escalation(
            id=self._escalation_queue.make_id(id_key),
            task_id=id_key,
            agent_role='orchestrator-merge-worker-supervisor',
            severity=severity,
            level=level,
            category='infra_issue',
            summary=summary,
            detail=detail,
            suggested_action=suggested_action,
        )
        self._escalation_queue.submit(esc)

    def _emit_loop_terminal_escalation(
        self, name: str, exc: BaseException, restart_count: int
    ) -> None:
        """Emit a born-at-L2 terminal escalation when the restart cap is exceeded.

        ``severity='critical'`` + ``level=2`` routes straight to a human (born-at-L2).
        The ``orchestrator-`` agent_role prefix marks it as a harness sentinel so the
        escalation server never downgrades the severity.

        None-safe (delegated to ``_submit_loop_escalation``).
        """
        self._submit_loop_escalation(
            name, exc,
            level=2,
            severity='critical',
            summary=(
                f'merge_worker_loop_died: {name} loop exceeded restart cap '
                f'({self._max_loop_restarts} restarts/{self._loop_restart_window_s}s) '
                f'— merge worker HALTED'
            ),
            detail_prefix=(
                f'Loop: {name}\n'
                f'Restart attempts in window: {restart_count}\n'
                f'Max restarts allowed: {self._max_loop_restarts}\n'
                f'Window: {self._loop_restart_window_s}s\n'
                f'Last exception type: {type(exc).__name__}\n'
                f'Last exception: {exc}'
            ),
            tb_label='Full traceback (last death)',
            suggested_action=(
                'Restart the orchestrator; the merge worker is halted and no longer '
                'draining merges.'
            ),
            id_key=f'{_MERGE_WORKER_LOOP_DIED_SENTINEL}:{name}:terminal',
        )

    def _emit_loop_death_escalation(self, name: str, exc: BaseException) -> None:
        """Emit a loud L1 ``'merge_worker_loop_died'`` escalation.

        None-safe (delegated to ``_submit_loop_escalation``).

        The ``agent_role='orchestrator-merge-worker-supervisor'`` prefix marks these as
        harness sentinels so the escalation server never downgrades their severity.

        No ``has_open_l1`` dedup: the restart cap bounds volume (≤3 per window), and
        the project directive is "prefer loud escalation over silent degradation".
        """
        self._submit_loop_escalation(
            name, exc,
            level=1,
            severity='blocking',
            summary=f'merge_worker_loop_died: {name} loop died unexpectedly — restarting',
            detail_prefix=(
                f'Loop: {name}\n'
                f'Exception type: {type(exc).__name__}\n'
                f'Exception: {exc}'
            ),
            suggested_action='Worker auto-restarted the loop; investigate the traceback.',
            id_key=f'{_MERGE_WORKER_LOOP_DIED_SENTINEL}:{name}',
        )

    # ── run / stop ──────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start merger, verifier, heartbeat, and reprobe coroutines; wait for merge tasks.

        Uses ``await self._loops_finished.wait()`` instead of
        ``asyncio.gather(merger, verifier)`` so the supervisor can replace dead loop
        tasks in-place (gather binds to fixed task objects and propagates the first
        exception — incompatible with in-process restart).

        The outer-cancellation path (``except BaseException``) and heartbeat/reprobe
        cleanup (``finally``) preserve the original semantics.  The ``finally`` also
        cancels any still-live loop task on the terminal-halt path (so the healthy
        sibling of a cap-exceeded loop does not leak).
        """
        # Reset supervisor state for this invocation.
        self._live_loops = {'merger', 'verifier'}
        self._loops_finished.clear()
        # Spawn merger and verifier via _spawn_loop so the supervisor callbacks
        # are registered from the very first iteration.
        self._spawn_loop('merger')
        self._spawn_loop('verifier')
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._reprobe_task = asyncio.create_task(self._reprobe_loop())
        try:
            await self._loops_finished.wait()
        except BaseException:
            for t in (
                self._merger_task, self._verifier_task,
                self._heartbeat_task, self._reprobe_task,
            ):
                if t and not t.done():
                    t.cancel()
            await asyncio.gather(
                *[
                    t for t in (
                        self._merger_task, self._verifier_task,
                        self._heartbeat_task, self._reprobe_task,
                    )
                    if t is not None
                ],
                return_exceptions=True,
            )
            raise
        finally:
            # Cancel any still-live loop tasks (terminal-halt path has healthy sibling).
            # On the exception path the except block already cleaned them up.
            for t in (self._merger_task, self._verifier_task):
                if t and not t.done():
                    t.cancel()
                    await asyncio.gather(t, return_exceptions=True)
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            if self._reprobe_task and not self._reprobe_task.done():
                self._reprobe_task.cancel()
                await asyncio.gather(self._reprobe_task, return_exceptions=True)

    async def stop(self) -> None:
        """Graceful shutdown: drain queues and resolve all pending Futures."""
        self._running = False
        shutdown = MergeOutcome('blocked', reason=MERGE_WORKER_SHUTDOWN_REASON)
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

        # γ: resolve/cancel the persistent verifier getter.  If it already
        # harvested an item it is invisible to the queue-drain below, so resolve
        # that item's Future (and clean its worktree) here to avoid a hung
        # merge() caller; otherwise cancel the still-pending get().
        if self._pending_verifier_get is not None:
            _pvg = self._pending_verifier_get
            self._pending_verifier_get = None
            if _pvg.done() and not _pvg.cancelled():
                try:
                    _harvested = _pvg.result()
                except BaseException:
                    _harvested = None
                if _harvested is not None:
                    if _harvested.merge_wt is not None:
                        with contextlib.suppress(BaseException):
                            await self._cleanup_owned_merge_worktree(
                                _harvested.merge_wt
                            )
                    if not _harvested.request.result.done():
                        _harvested.request.result.set_result(shutdown)
            elif not _pvg.done():
                _pvg.cancel()

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

        # Drain _inflight: cancel every background verify task, clean owned
        # merge worktrees, release host leases, and resolve pending futures.
        # Done BEFORE sending sentinels so _verifier_loop sees an empty deque
        # when the None sentinel arrives — no CancelledError mid-finalize.
        # Futures are resolved first (before task cancel) so there's no window
        # where the task is cancelled but the future is still pending.
        for _ie in list(self._inflight):
            _ie_req = _ie.item.request
            if not _ie_req.result.done():
                _ie_req.result.set_result(shutdown)
            if _ie.verify_task is not None and not _ie.verify_task.done():
                # Fire remote cancel BEFORE task.cancel() so the remote
                # verify-merge process is signalled while _inflight_request_id
                # is still live (mirrors task-1757 _run_inflight_verify fix).
                # suppress(BaseException) matches the drain's shutdown-defensive
                # pattern (cf. cancel_and_release suppress below) so a
                # SIGTERM-driven CancelledError cannot abort the drain loop.
                if _ie.lease is not None:
                    with contextlib.suppress(BaseException):
                        await self._abort_remote_verify(_ie.lease, _ie_req.task_id)
                _ie.verify_task.cancel()
                with contextlib.suppress(BaseException):
                    await _ie.verify_task
            if _ie.merge_wt is not None:
                with contextlib.suppress(BaseException):
                    await self._cleanup_owned_merge_worktree(_ie.merge_wt)
            if _ie.lease is not None and self._host_allocator is not None:
                with contextlib.suppress(BaseException):
                    await self._host_allocator.cancel_and_release(_ie.lease)
            if _ie.was_speculative:
                self._speculation_slot.release()
        self._inflight.clear()

        # Drain _redispatch: items pending re-dispatch after a cascade.
        while self._redispatch:
            _rd = self._redispatch.popleft()
            if not _rd.request.result.done():
                _rd.request.result.set_result(shutdown)
            if _rd.merge_wt is not None:
                with contextlib.suppress(BaseException):
                    await self._cleanup_owned_merge_worktree(_rd.merge_wt)

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

        # Re-drain main queue: under operator halt, _dispatch_item returns
        # REQUEUED_PREDISPATCH and calls _queue.put_nowait(req) without resolving
        # the future.  If the verifier processed a SpeculativeItem during the
        # asyncio.wait() above (step G), the req landed back on _queue after the
        # initial drain (step A) was already done.  A second drain catches those.
        while not self._queue.empty():
            try:
                req_post = self._queue.get_nowait()
                if req_post is not None and not req_post.result.done():
                    req_post.result.set_result(shutdown)
            except asyncio.QueueEmpty:
                break

        # Check _inflight_req: if the merger was still blocked inside merge_to_main
        # when asyncio.wait() timed out, it still holds _inflight_req.  Resolve the
        # Future now so the caller doesn't hang forever.
        if self._inflight_req is not None and not self._inflight_req.result.done():
            self._inflight_req.result.set_result(shutdown)

        # Cancel background tasks that loop independently (no sentinel path).
        # _running is already False so the loops will not re-enter after
        # cancellation; we await each to ensure they are done before stop() returns.
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
        if self._reprobe_task is not None and not self._reprobe_task.done():
            self._reprobe_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reprobe_task

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

    async def _redrive_coalesce_members(
        self, req: GroupMergeRequest, main_sha: str,
    ) -> None:
        """Re-drive absorbed members still merge-deferred after a coalesce-train derail.

        For each member task still in 'merge-deferred' status (per req.status_check):
          - If the member's branch is already an ancestor of main (double-landing
            guard: a partner's merge brought it in), mark it done with
            found_on_main provenance via req.redrive_member(mid, True, sha).
          - Otherwise flip it to 'pending' via req.redrive_member(mid, False, None)
            so the scheduler re-dispatches a fresh solo-merge workflow that owns
            the merge-deferred→done transition.

        Members not in 'merge-deferred' (e.g. raced to 'in-progress' by a sibling)
        are left alone — the live workflow owns their done-transition.

        Per-member try/except ensures one member failure never aborts the rest or
        kills the merger loop.  req.redrive_member=None is a back-compat no-op.
        """
        if req.redrive_member is None:
            logger.warning(
                'Coalesce train %s: _redrive_coalesce_members called but '
                'req.redrive_member is None — cannot re-drive stranded members',
                req.train_id,
            )
            return

        statuses = await req.status_check(req.member_task_ids)
        deferred_mids = [
            mid for mid in req.member_task_ids
            if statuses.get(mid) == 'merge-deferred'
        ]

        for mid in deferred_mids:
            try:
                branch = f'{self._git_ops.config.branch_prefix}{mid}'
                on_main = await self._git_ops.is_ancestor(
                    branch, self._git_ops.config.main_branch,
                )
                if on_main:
                    sha = await self._git_ops.resolve_branch_sha(branch) or main_sha
                    await req.redrive_member(mid, True, sha)
                    logger.info(
                        'Coalesce train %s: member %s already on main — '
                        'marked done (found_on_main, sha=%s)',
                        req.train_id, mid, sha,
                    )
                else:
                    await req.redrive_member(mid, False, None)
                    logger.info(
                        'Coalesce train %s: member %s not on main — '
                        'flipped to pending for solo-merge re-dispatch',
                        req.train_id, mid,
                    )
            except Exception:
                logger.exception(
                    'Coalesce train %s: re-drive failed for member %s — '
                    'continuing with remaining members',
                    req.train_id, mid,
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
            redrive_member=callbacks.redrive_member,
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
        # Late-arrival attach state (task 1862): when the look-ahead peek finds
        # nothing pickable but the predecessor is still in-flight, RETAIN the
        # held speculation permit and record the predecessor's merge commit here.
        # A late-arriving successor will ATTACH to it at the next dequeue instead
        # of falling back to non-speculative plain-main.
        # Both are cleared after the ATTACH/FALLBACK decision following the next
        # _acquire_next_request() call, and on the shutdown/break paths.
        pending_spec_base: str | None = None   # predecessor's merge commit SHA
        pending_predecessor: MergeRequest | None = None  # predecessor request
        # Set True when the loop exits cleanly via the None shutdown sentinel from
        # _acquire_next_request().  Used in the finally to decide whether to forward
        # the sentinel to the verifier: on the crash path (exception while _running is
        # True) this is False, so the stray put(None) that would silently kill the
        # surviving verifier is suppressed.
        exited_via_sentinel: bool = False

        try:
            while self._running:
                # γ/1719 retroactive coalescing pass — design decisions summary:
                # • DD1: runs at the pre-dequeue point, gated on a clean pipeline
                #   (spec_base=None and prefetched=None and pending_spec_base=None)
                #   so a train is never enqueued behind an unverified speculative merge
                #   commit (:5239 warning).  The task-1862 retain path records the
                #   predecessor's commit in pending_spec_base while spec_base remains
                #   None, so pending_spec_base must also be tested here — otherwise
                #   coalescing could form a GroupMergeRequest behind an in-flight
                #   speculative predecessor and attach it to that predecessor's commit,
                #   violating DD1's invariant.  (merge_train_coalesce_enabled=False by
                #   default so this is latent; guard added for correctness when enabled.)
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
                if spec_base is None and prefetched is None and pending_spec_base is None:
                    await self._maybe_coalesce_waiting_singles()

                # Get next request: use pre-fetched (speculative) item if available,
                # otherwise acquire from the lane-priority pick system.
                if prefetched is not None:
                    req = prefetched
                    prefetched = None
                else:
                    req = await self._acquire_next_request()
                    if req is None:
                        exited_via_sentinel = True
                        # Clear pending late-arrival state before exiting.
                        pending_spec_base = None
                        pending_predecessor = None
                        break  # shutdown sentinel
                    # ATTACH or FALLBACK: decide whether this fresh dequeue is a
                    # late arrival that can attach to the in-flight predecessor's
                    # merge commit (task 1862).
                    #
                    # ATTACH — all four conditions must hold:
                    #   (a) pending_spec_base: predecessor's commit was recorded
                    #   (b) held_spec_permit: speculation permit is still held
                    #   (c) pending_predecessor: predecessor request is known
                    #   (d) predecessor still in-flight (result future not done)
                    #
                    # On ATTACH: spec_base = pending_spec_base so this late
                    # arrival merges against main+A, not plain main.  The
                    # retained permit transfers to the verifier item on drain.
                    #
                    # On FALLBACK (any condition fails): release the held permit
                    # if present and merge non-speculatively against actual main.
                    if (
                        pending_spec_base is not None
                        and held_spec_permit
                        and pending_predecessor is not None
                        and not pending_predecessor.result.done()
                    ):
                        spec_base = pending_spec_base  # ATTACH
                        logger.debug(
                            'Task %s: late arrival attaches to in-flight '
                            'predecessor %s (spec_base=%s)',
                            req.task_id,
                            pending_predecessor.task_id,
                            pending_spec_base[:8],
                        )
                    else:
                        # FALLBACK — release retained permit (if any) and merge
                        # against actual main (plain non-speculative).
                        if held_spec_permit:
                            self._speculation_slot.release()
                            held_spec_permit = False
                        spec_base = None
                    # Clear pending locals — consumed by ATTACH or dropped by FALLBACK.
                    pending_spec_base = None
                    pending_predecessor = None

                self._inflight_req = req  # track for stop() race resolution
                # ι=1894: stash main_position for this request so
                # _note_conflict_detected can compute drift later.
                self._note_merge_started(req.request_id)
                # Drop-on-detection: workflow soft-cancelled before worker
                # dequeued.  Skipping merge work avoids the orphan-halt
                # window where no escalation owner is registered.
                if self._request_abandoned(req):
                    if held_spec_permit:
                        self._speculation_slot.release()
                        held_spec_permit = False
                    spec_base = None
                    self._inflight_req = None
                    # ι=1894 amend: drop stashed drift base — request retired without
                    # landing or conflict detection, so it would otherwise leak forever.
                    self._drift_base.pop(req.request_id, None)
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
                        # 1867: coalesce-train derail recovery.
                        # Both hooks below are gated on the coalesce prefix so
                        # declared trains (kept alive by the tip workflow's re-fire)
                        # are never touched.
                        if req.train_id.startswith(_COALESCE_TRAIN_ID_PREFIX):
                            # δ/1720: one-strike exclusion on risky outcomes so
                            # derailed members are not immediately re-coalesced.
                            # Kept on the narrower _COALESCE_RISKY_TERMINAL_STATES
                            # set ('blocked','error') — conflicts and wip_halted are
                            # excluded intentionally (conflict may resolve, wip_halted
                            # is policy-driven).
                            if outcome.status in _COALESCE_RISKY_TERMINAL_STATES:
                                self._mark_coalesce_derailed(req.member_task_ids)
                            # 1867: durable re-drive — re-drive any absorbed members
                            # that are still merge-deferred to solo merge.  Gated on
                            # outcome != 'done' (success path owns all member flips).
                            # Broader than _COALESCE_RISKY_TERMINAL_STATES: the
                            # per-member is_ancestor guard makes all non-done outcomes
                            # safe (already-landed members flip done, genuinely-
                            # unlanded members flip pending for solo re-dispatch).
                            #
                            # Note — wip_halted IS included here (unlike one-strike
                            # exclusion).  A coalesce GroupMergeRequest with
                            # wip_halted has no awaiter to re-fire it, so absorbed
                            # members would be permanently stranded in merge-deferred
                            # if we skipped re-drive.  Re-pending sends them back to
                            # the scheduler as solo candidates; each fresh workflow
                            # will hit the WIP-halt barrier independently and wait
                            # correctly.  The one-strike exclusion rationale (wip_halted
                            # is "policy-driven") governs whether to EXCLUDE members
                            # from the NEXT coalescing pass, which is a separate gate.
                            #
                            # Known gap (out of scope): if _do_train_merge returns
                            # 'done' but mark_member_done suffers a transient
                            # get_statuses error, one or more members may be left
                            # un-flipped in merge-deferred.  The re-drive below is
                            # intentionally excluded for the 'done' path because the
                            # success flow already logs "manual cleanup required" for
                            # TRAIN_PARTIAL_FLIP.  Covering the done-path gap is a
                            # follow-up task.
                            if outcome.status != 'done':
                                try:
                                    await self._redrive_coalesce_members(req, actual_main)
                                except Exception:
                                    logger.exception(
                                        'Coalesce train %s: _redrive_coalesce_members '
                                        'raised unexpectedly — members may remain stranded',
                                        req.train_id,
                                    )
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
                    # Pass req.worktree for parity with _do_merge: the
                    # absent-ref worktree-HEAD fallback fires in both paths.
                    guard = await _classify_branch_presence(
                        self._git_ops, self._event_store, req.task_id,
                        req.branch, t0,
                        worktree=req.worktree,
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
                        # ι=1894: record drift = main_position − base at merge-start
                        self._note_conflict_detected(req.request_id)
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
                            self._buffer_owned_request(_item)
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
                        # Clear pending late-arrival state before exiting.
                        pending_spec_base = None
                        pending_predecessor = None
                        break  # shutdown sentinel and nothing left to speculate
                    else:
                        # Nothing pickable right now AND predecessor is in-flight.
                        # RETAIN the held speculation permit so a late-arriving
                        # successor can attach to the predecessor's merge commit
                        # (task 1862: close the disjoint-skip semantic-conflict hole).
                        #
                        # • DON'T release: the permit transfers to the late arrival
                        #   on ATTACH (or is released on FALLBACK at next dequeue).
                        # • DON'T set spec_base: the late arrival hasn't been dequeued
                        #   yet; spec_base is set at the ATTACH/FALLBACK decision.
                        # • Record predecessor's commit so the next dequeue can attach.
                        pending_spec_base = merge_commit  # predecessor's merge commit
                        pending_predecessor = req         # predecessor in-flight
                        # held_spec_permit stays True — permit retained for late arrival
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
                    MergeOutcome('blocked', reason=MERGE_WORKER_SHUTDOWN_REASON)
                )
            # Send shutdown sentinel ONLY when the merger is genuinely shutting down:
            # either (a) stop() set _running=False, or (b) the loop broke cleanly via
            # the None shutdown sentinel from _acquire_next_request (exited_via_sentinel).
            #
            # On the crash path (unhandled exception with _running still True and
            # exited_via_sentinel still False) the merger must NOT push a sentinel.  If
            # it did, the still-healthy verifier would consume it, drain _inflight, and
            # return cleanly — _on_loop_task_done would retire the verifier without
            # restarting it, silently killing the verifier on exactly the crash scenario
            # the supervisor exists to prevent: a restarted merger + a permanently-dead
            # verifier, with all newly merged items piling on _verifier_queue and result
            # Futures hanging forever.
            #
            # Guard semantics:
            #   • `not self._running` — covers the "stop() was called" case; stop() sets
            #     _running=False (:6246) BEFORE draining and itself pushes the verifier
            #     sentinel at :6364, so the re-drain loop there catches any late items.
            #   • `exited_via_sentinel` — covers the "got None from the input queue"
            #     case, which includes direct _merger_loop() test calls that put a None
            #     in the queue without going through stop().  In the real lifecycle both
            #     are True simultaneously (stop() sets _running=False first, THEN puts
            #     the None); here only one may be True, but either is sufficient.
            if not self._running or exited_via_sentinel:
                await self._verifier_queue.put(None)

    # ------------------------------------------------------------------
    # Verifier coroutine
    # ------------------------------------------------------------------

    async def _verifier_loop(self) -> None:
        """Verify and CAS-advance for each SpeculativeItem from the Merger.

        γ restructuring: dispatch-fill + finalize-head, backed by self._inflight.

        Each outer iteration:
          (a) DISPATCH-FILL: drain self._redispatch then _verifier_queue.get_nowait()
              while a host slot is free (or item is a passthrough).  Each item is
              dispatched via _dispatch_item → InflightEntry appended to self._inflight.
          (b) FINALIZE-HEAD: await self._inflight.popleft() via _finalize_inflight,
              advancing main in submission order.  If _inflight is empty, block on
              _verifier_queue.get() to avoid busy-looping.

        NONE SENTINEL: when the queue yields None, drain the remaining _inflight
        entries (all background verify tasks complete) and return.

        SINGLE-HOST degeneracy: with one slot, dispatch acquires the local slot →
        free_host_count() drops to 0 → the fill loop stops after one entry →
        finalize(head) releases the slot → next iteration dispatches the next item.
        This is byte-identical to the old serial loop.

        Exception handling: unexpected exceptions from _dispatch_item (e.g. a
        _remerge failure) are caught, logged, the request resolved with 'blocked',
        and the loop continues so a single bad item does not crash the queue.
        CancelledError is NOT caught; it propagates to stop() which cancels
        _verifier_task.
        """
        while True:
            # ── (a) DISPATCH-FILL ──────────────────────────────────────────────
            # Fill self._inflight as long as host slots are available.
            fill_done = False
            while not fill_done:
                # Get next item: front-priority _redispatch first, then queue nowait
                item: SpeculativeItem | None = None
                if self._redispatch:
                    item = self._redispatch.popleft()
                    is_from_verifier_queue = False
                elif (
                    self._pending_verifier_get is not None
                    and self._pending_verifier_get.done()
                ):
                    # A persistent getter launched by a prior race already harvested
                    # the next queue item — consume it before anything else so it is
                    # not lost (None is handled by the shutdown check below).
                    item = self._pending_verifier_get.result()
                    self._pending_verifier_get = None
                    is_from_verifier_queue = True
                else:
                    try:
                        if self._pending_verifier_get is not None:
                            # A getter is still pending → the queue is empty from our
                            # view (a pending getter consumes arrivals before
                            # get_nowait would see them).
                            raise asyncio.QueueEmpty
                        item = self._verifier_queue.get_nowait()
                        is_from_verifier_queue = True
                    except asyncio.QueueEmpty:
                        # Multi-host fill-ahead: when real verify tasks are
                        # STILL RUNNING (not done) AND a host slot is free, race the
                        # next queue item against the running verifies so dispatch-
                        # fill launches N+1 to the free slot while N verifies — but a
                        # verify *completing* must ALSO wake us so FINALIZE-HEAD runs.
                        # Otherwise the last item of a merge burst would hang: its
                        # verify finishes in the background but FINALIZE-HEAD is never
                        # reached, so _finalize_inflight() never runs and the merge()
                        # caller's result Future is never resolved.
                        # Single-host: free_host_count()==0 when the local slot is
                        # held → else branch, break immediately → byte-identical to
                        # prior serial behaviour.
                        # Guard: only block while in-flight tasks are running; once
                        # they are done the head needs to be finalized, not more
                        # items fetched.
                        _has_running_inflight = any(
                            e.verify_task is not None and not e.verify_task.done()
                            for e in self._inflight
                        )
                        if (
                            _has_running_inflight
                            and self._host_allocator is not None
                            and self._host_allocator.free_host_count() > 0
                        ):
                            # Persistent getter (never cancelled mid-race → no lost
                            # item) raced against the running verify tasks.
                            if self._pending_verifier_get is None:
                                self._pending_verifier_get = asyncio.ensure_future(
                                    self._verifier_queue.get()
                                )
                            _running = {
                                e.verify_task for e in self._inflight
                                if e.verify_task is not None
                                and not e.verify_task.done()
                            }
                            await asyncio.wait(
                                {self._pending_verifier_get, *_running},
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if self._pending_verifier_get.done():
                                # A new item arrived first → dispatch it.
                                # Guard against cancelled getter (stop() race):
                                # treat cancelled as nothing-arrived → fall
                                # through to FINALIZE-HEAD instead.
                                if self._pending_verifier_get.cancelled():
                                    self._pending_verifier_get = None
                                    fill_done = True
                                    break
                                item = self._pending_verifier_get.result()
                                self._pending_verifier_get = None
                                is_from_verifier_queue = True
                                # Fall through with item (None handled below).
                            else:
                                # A verify finished first → stop filling and proceed
                                # to FINALIZE-HEAD.  The getter persists to the next
                                # DISPATCH-FILL iteration so no queue item is lost.
                                fill_done = True
                                break
                        else:
                            fill_done = True
                            break

                if item is None:
                    # Shutdown sentinel: drain remaining in-flight entries, then exit.
                    while self._inflight:
                        head = self._inflight.popleft()
                        await self._finalize_inflight(head)
                    return

                # Dispatch the item (applies Mechanism 1, abandon/halt/passthrough/
                # chain-remerge logic, host acquire, verify task launch).
                try:
                    entry = await self._dispatch_item(item)
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except BaseException as exc:
                    # Unexpected dispatch error (e.g. _remerge raised; _git_ops
                    # unavailable).  Resolve the request and continue the loop.
                    logger.exception(
                        'Task %s: unexpected dispatch error', item.request.task_id
                    )
                    await self._resolve_and_release(
                        item, MergeOutcome('blocked', reason=f'Verifier error: {exc}'),
                        chain_failed=True,
                    )
                    continue

                if entry is None:
                    # No host available: put item back on _redispatch.
                    # counts_against_cap was already released in _dispatch_item;
                    # clear the flag to prevent a double-release on re-dispatch.
                    item_back = dataclasses.replace(item, counts_against_cap=False)
                    self._redispatch.appendleft(item_back)
                    fill_done = True
                    break

                # Passthrough entries (verify_task=None, lease=None) are already
                # decided: finalize them inline so _n_failed is updated before the
                # next dispatch.  This preserves the old serial loop's ordering where
                # n_failed was visible to the very next pickup.
                #
                # Ordering note: passthroughs finalized inline here can resolve
                # their result-Future BEFORE an earlier real-verify entry in
                # _inflight resolves its Future.  The InflightEntry submission-order
                # invariant covers main-ADVANCEMENT ordering (CAS is sequential),
                # not cross-item Future-resolution ordering.  Passthroughs never
                # advance main (they are conflict/already_merged/skip_verify), so
                # inline finalization does not violate the advancement order contract.
                # No consumer depends on strict cross-item Future-delivery ordering.
                if entry.verify_task is None:
                    try:
                        await self._finalize_inflight(entry)
                    except (asyncio.CancelledError, KeyboardInterrupt):
                        raise
                    except BaseException as exc:
                        logger.exception(
                            'Task %s: unexpected passthrough finalize error',
                            entry.item.request.task_id,
                        )
                        await self._resolve_and_release(
                            entry, MergeOutcome('blocked', reason=f'Verifier error: {exc}'),
                            chain_failed=True, release_resources=False,
                        )
                    continue  # don't append to _inflight; fetch next item

                self._inflight.append(entry)

                # Continue filling only if another slot is free (real verify entries
                # consume a host slot, so check free_host_count).
                # Also stop filling if we just dispatched from _redispatch and it is
                # now empty: cascade-recovery items should proceed to FINALIZE-HEAD
                # rather than blocking on _verifier_queue.get() waiting for new work
                # (which would deadlock when the queue is empty after a cascade).
                allocator = self._ensure_host_allocator(entry.item.request.config)
                if allocator.free_host_count() == 0 or (
                    not is_from_verifier_queue and not self._redispatch
                ):
                    fill_done = True
                    break

            # ── (b) FINALIZE-HEAD ──────────────────────────────────────────────
            if self._inflight:
                head = self._inflight.popleft()
                _head_advanced = False
                try:
                    _head_advanced = await self._finalize_inflight(head)
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except BaseException as exc:
                    logger.exception(
                        'Task %s: unexpected finalize error', head.item.request.task_id
                    )
                    # _finalize_inflight's finally already released the lease
                    # and speculation slot; the chokepoint just resolves the
                    # future and marks the chain as failed.
                    await self._resolve_and_release(
                        head, MergeOutcome('blocked', reason=f'Verifier error: {exc}'),
                        chain_failed=True, release_resources=False,
                    )

                # HEAD-FAILURE CASCADE (γ step-20): when the head fails, abort
                # all downstream in-flight verifies, re-merge each item onto
                # actual main, and front-queue the re-merged items on _redispatch
                # for strictly-ordered re-dispatch.
                #
                # This preserves chain-invalidation under overlap: a speculative
                # N+1 that launched against N's (not-yet-landed) merge commit is
                # now stale.  We cancel it (remote cancel_verify), re-merge it
                # against actual main, and re-verify it so the correct commit
                # lands.  Main still advances in submission order (N never landed
                # → N+1 re-merges and advances as the new head).
                if not _head_advanced and self._inflight:
                    _allocator = self._host_allocator
                    _downstream = list(self._inflight)
                    self._inflight.clear()

                    # Detect whether the head failure was due to operator halt
                    # (REQUEUED sentinel).  In that case downstream tasks will
                    # also detect halt via their abort-polls and self-requeue;
                    # we must NOT _remerge them.  If we cancel a downstream task
                    # before its abort-poll fires, we manually requeue its req.
                    _head_was_requeued = False
                    if head.verify_task is not None and head.verify_task.done():
                        try:
                            _hvt = head.verify_task.result()
                            _head_was_requeued = (
                                getattr(_hvt, 'status', None) == InflightStatus.REQUEUED
                            )
                        except BaseException:
                            pass

                    # Error containment (task-1856): each downstream entry is
                    # wrapped in try/except, mirroring the four sibling
                    # _verifier_loop branches (dispatch branch, passthrough-finalize
                    # branch, finalize-head branch, blocking-get branch).
                    # CancelledError/KeyboardInterrupt still re-raise to honour
                    # sentinel-based shutdown: stop() terminates the loop via a
                    # None sentinel rather than direct task cancellation, so the
                    # cascade runs to completion on normal shutdown; external
                    # task cancellation remains an accepted edge case.
                    # Any other BaseException is caught per entry: req.result
                    # resolves as MergeOutcome('blocked', reason='Verifier
                    # cascade error: ...') and the loop continues to the next
                    # downstream entry.
                    # Release discipline: lease+slot are released in-body at
                    # cancel_and_release / _speculation_slot.release() BEFORE
                    # _remerge is called; the _entry_released flag prevents a
                    # double-release in the except handler on the
                    # _remerge-raises path (which is the primary failure mode).
                    for _entry in _downstream:
                        _entry_status: str | None = None
                        _entry_released = False
                        try:
                            if _entry.verify_task is not None:
                                # Fire remote cancel BEFORE task.cancel() so the
                                # remote verify-merge process is signalled while
                                # _inflight_request_id is still live (mirrors
                                # task-1757 _run_inflight_verify fix).  Helper
                                # swallows Exception internally; CancelledError
                                # propagates to stop the loop (correct behaviour).
                                if _entry.lease is not None:
                                    await self._abort_remote_verify(
                                        _entry.lease, _entry.item.request.task_id,
                                    )
                                _entry.verify_task.cancel()
                                with contextlib.suppress(BaseException):
                                    await _entry.verify_task
                                # Peek at the completed result to detect REQUEUED
                                # (operator-halt): the request is already back on
                                # _queue via the abort-poll; _remerge must be
                                # skipped to avoid a duplicate re-dispatch.
                                if (
                                    _entry.verify_task.done()
                                    and not _entry.verify_task.cancelled()
                                ):
                                    try:
                                        _vt_res = _entry.verify_task.result()
                                        if hasattr(_vt_res, 'status'):
                                            _entry_status = _vt_res.status
                                    except BaseException:
                                        pass
                            if _entry.lease is not None and _allocator is not None:
                                await _allocator.cancel_and_release(_entry.lease)
                            if _entry.merge_wt is not None:
                                with contextlib.suppress(BaseException):
                                    await self._cleanup_owned_merge_worktree(
                                        _entry.merge_wt
                                    )
                            # LATE-ARRIVAL ATTACH SYMMETRY (task 1862 step-6):
                            # A late arrival B attached via pending_spec_base is
                            # dispatched with speculative=True and held_spec_permit
                            # retained (step-2), so B's InflightEntry carries
                            # was_speculative=True.  The release below fires here
                            # (predecessor failed → B is a downstream entry), and
                            # _remerge returns speculative=False → re-dispatched B
                            # has item_was_speculative=False → no duplicate release.
                            # Slot symmetry is maintained on the late-arrival path
                            # identically to the standard prefetch path.
                            if _entry.was_speculative:
                                self._speculation_slot.release()
                            # Past the in-body lease+slot release; the except
                            # handler must not re-release on any path below.
                            _entry_released = True
                            # REQUEUED: abort-poll already put req on _queue → skip.
                            if _entry_status == InflightStatus.REQUEUED:
                                continue
                            # Head was REQUEUED (operator halt) and we cancelled
                            # this downstream task before its abort-poll could
                            # requeue it: manually put the req on _queue so
                            # stop() can resolve it.
                            if (
                                _head_was_requeued
                                and _entry.verify_task is not None
                                and _entry.verify_task.cancelled()
                            ):
                                _entry_req = _entry.item.request
                                if not _entry_req.result.done():
                                    self._queue.put_nowait(_entry_req)
                                continue
                            _remerged = await self._remerge(
                                _entry.item.request,
                                _entry.item.started_monotonic,
                            )
                            self._redispatch.append(_remerged)
                        except (asyncio.CancelledError, KeyboardInterrupt):
                            raise
                        except BaseException as _cascade_exc:
                            logger.exception(
                                'Task %s: unexpected cascade error',
                                _entry.item.request.task_id,
                            )
                            # In-body release already ran (_entry_released=True)
                            # UNLESS it raised before completing (e.g.
                            # cancel_and_release itself raised) — in which case
                            # the chokepoint performs the best-effort release
                            # (lease/slot/merge-worktree) to avoid a leak.
                            await self._resolve_and_release(
                                _entry, MergeOutcome(
                                    'blocked',
                                    reason=f'Verifier cascade error: {_cascade_exc}',
                                ),
                                chain_failed=True,
                                release_resources=not _entry_released,
                                cancel_lease=True,
                            )
                            continue
                    # Signal dispatch that any not-yet-dispatched followers also
                    # need re-merge (chain_invalidated guard in _dispatch_item).
                    self._remerge_occurred = True
            else:
                # Nothing dispatched (no items in queue or no host yet free after
                # putting item back).  Block on the next item from the queue.
                # Reuse the persistent getter if a prior race left one outstanding,
                # so there is never a second concurrent getter on _verifier_queue
                # (two getters would race for one item → the loser blocks forever).
                if self._pending_verifier_get is not None:
                    _pvg = self._pending_verifier_get
                    self._pending_verifier_get = None
                    try:
                        item = await _pvg
                    except asyncio.CancelledError:
                        # Getter was cancelled (stop() ordering race); re-fetch
                        # via a fresh get so no queue item is lost.
                        item = await self._verifier_queue.get()
                else:
                    item = await self._verifier_queue.get()
                if item is None:
                    # Shutdown sentinel with an already-empty queue.
                    return

                try:
                    entry = await self._dispatch_item(item)
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except BaseException as exc:
                    logger.exception(
                        'Task %s: unexpected dispatch error (blocking get)',
                        item.request.task_id,
                    )
                    await self._resolve_and_release(
                        item, MergeOutcome('blocked', reason=f'Verifier error: {exc}'),
                        chain_failed=True,
                    )
                    continue

                if entry is None:
                    # No host (shouldn't happen with empty _inflight on a single-host
                    # system, but handle defensively: item goes to _redispatch).
                    item_back = dataclasses.replace(item, counts_against_cap=False)
                    self._redispatch.appendleft(item_back)
                    continue

                # Passthrough: finalize inline (no host slot held, never blocks
                # on a verify task) then restart the outer loop.
                if entry.verify_task is None:
                    try:
                        await self._finalize_inflight(entry)
                    except (asyncio.CancelledError, KeyboardInterrupt):
                        raise
                    except BaseException as exc:
                        logger.exception(
                            'Task %s: unexpected passthrough finalize error '
                            '(blocking-get path)', entry.item.request.task_id,
                        )
                        await self._resolve_and_release(
                            entry, MergeOutcome('blocked', reason=f'Verifier error: {exc}'),
                            chain_failed=True, release_resources=False,
                        )
                    continue  # restart outer loop → fill loop picks up next item

                # Real verify entry: append to _inflight and loop back to fill.
                # The fill loop will block for the next item if a host slot is
                # free (multi-host overlap) OR break immediately (single-host,
                # free_host_count()==0) → FINALIZE-HEAD processes the head.
                self._inflight.append(entry)
                continue  # restart outer loop

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
        full_branch = (
            await self._git_ops.resolve_queued_branch_ref(req.branch)
            or f'{self._git_ops.config.branch_prefix}{req.branch}'
        )
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

        # Capture the branch tip now, before merge_to_main creates a separate
        # merge worktree.  req.worktree HEAD does not change during _remerge —
        # all merge operations write into a fresh merge_worktree.  Mirrors the
        # merger-loop's rev-parse at Step 1; used as merged_branch_tip on the
        # success-path SpeculativeItem so _finalize_advanced_merge can compare
        # the equivalence gate against the actual tip (PRIMARY fix parity).
        _rc_bt, _branch_tip_raw, _err_bt = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=req.worktree,
        )
        _remerge_branch_tip: str | None = (
            _branch_tip_raw.strip() if _rc_bt == 0 else None
        )
        if _rc_bt != 0:
            logger.warning(
                'Task %s: _remerge: rev-parse HEAD failed (will not set '
                'merged_branch_tip): %s', req.task_id, _err_bt.strip(),
            )

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
                    merged_branch_tip=_remerge_branch_tip,  # γ2: parity with merger loop
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
            merged_branch_tip=_remerge_branch_tip,  # γ2: parity with merger loop
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

    async def _resolve_and_release(
        self,
        entry_or_item: InflightEntry | SpeculativeItem,
        outcome: MergeOutcome,
        *,
        chain_failed: bool,
        release_resources: bool = True,
        cancel_lease: bool = False,
    ) -> None:
        """Single resolve-and-release chokepoint for the _verifier_loop
        BaseException handlers (task 1991 / MQ-refactor zeta).

        Unifies the six near-identical except-handler bodies that used to
        inline: resolve the caller's Future, release the host lease (release
        vs cancel_and_release), clean up an owned merge worktree, release the
        speculation slot, and flag the chain as failed.  Each call site keeps
        its own ``logger.exception`` message and its own
        ``except (CancelledError, KeyboardInterrupt): raise`` clause; only the
        resolve-and-release tail is centralised here.

        *entry_or_item* is normalised to (req, lease, merge_wt, slot_held):
        an ``InflightEntry`` carries ``lease``/``merge_wt``/``was_speculative``
        (post-dispatch state); a bare ``SpeculativeItem`` has no lease yet
        (pre-dispatch state) and uses ``merge_wt``/``speculative`` directly.

        *release_resources* controls whether this call performs any release
        at all.  It must be ``False`` for the POST-finalize handlers
        (passthrough-finalize x2, finalize-head): ``_finalize_inflight``'s own
        ``finally`` clause already released the lease and speculation slot on
        every exit path including exceptions, so releasing again here would
        double-release.  It defaults to ``True`` for the PRE-finalize dispatch
        handlers, which own the item's resources outright.

        *cancel_lease* selects ``cancel_and_release`` over plain ``release``
        (mirrors ``_finalize_inflight``'s own ``_cancel_release`` switch) —
        used by the cascade handler, which aborts a still-verifying entry.

        *chain_failed* is guarded (``if chain_failed: self._n_failed = True``)
        rather than assigned unconditionally, so a caller can never reset
        ``_n_failed`` back to False.  All six current call sites pass True.

        Ordering (resolve-before-release): step 1 always resolves the
        caller's Future before step 2 releases resources.  This matches the
        pre-refactor cascade except-handler exactly, but reorders the
        PRE-finalize dispatch handlers, which used to clean the merge
        worktree BEFORE resolving the Future.  That reorder is intentional
        and safe: the merge worktree is a worker-internal implementation
        detail (tracked only in ``_owned_merge_worktrees``) that is never
        exposed to the ``req.result`` waiter, so a waiter woken by the
        just-resolved Future cannot observe the not-yet-cleaned worktree or
        not-yet-released lease/slot.  No consumer or test depends on
        intra-handler await ordering between resolve and release.

        Abandoned-drop semantics: step 1 resolves via
        ``_resolve_or_drop_abandoned`` — i.e. ``_request_abandoned``, which
        is ``True`` iff ``req.result.cancelled()`` — rather than each site's
        prior inline ``if not req.result.done(): req.result.set_result(...)``.
        The two are behaviourally IDENTICAL in every reachable state, not
        merely a compatible superset: ``cancelled()`` implies ``done()``, so
        when the Future is already cancelled both the old inline check and
        the new abandoned-check skip ``set_result``; when it is not
        cancelled, ``_resolve_or_drop_abandoned`` falls through to that same
        ``if not done(): set_result(...)``.  The sole producer of the
        cancelled state is ``PendingMergeRegistry.detach()``
        (merge_types.py), which implements "the waiter abandoned this
        request" as cancelling ``req.result`` directly — there is no
        separate detached-but-not-cancelled flag anywhere in this codebase,
        so that intermediate state is not reachable.  The only observable
        difference is a new INFO log line on the abandoned path.  This
        equivalence is intentional and holds uniformly across all six call
        sites now unified behind this chokepoint.

        This is also the intended hook surface for task eta's liveness-ledger
        'resolved' transition: every _verifier_loop error-resolution path now
        funnels through this one coroutine.
        """
        if isinstance(entry_or_item, InflightEntry):
            item = entry_or_item.item
            lease: Any | None = entry_or_item.lease
            merge_wt = entry_or_item.merge_wt
            slot_held = entry_or_item.was_speculative
        else:
            item = entry_or_item
            lease = None
            merge_wt = entry_or_item.merge_wt
            slot_held = entry_or_item.speculative
        req = item.request

        self._resolve_or_drop_abandoned(req, outcome)

        if release_resources:
            if lease is not None and self._host_allocator is not None:
                with contextlib.suppress(BaseException):
                    if cancel_lease:
                        await self._host_allocator.cancel_and_release(lease)
                    else:
                        await self._host_allocator.release(lease)
            if merge_wt is not None:
                with contextlib.suppress(BaseException):
                    await self._cleanup_owned_merge_worktree(merge_wt)
            if slot_held:
                self._speculation_slot.release()

        if chain_failed:
            self._n_failed = True

    async def _abort_remote_verify(self, lease: Any, task_id: str) -> None:
        """Fire remote cancel-verify while *_inflight_request_id* is still live.

        Must be called BEFORE ``verify_task.cancel()`` in each abort branch.
        The verify coroutine's finally clause (verify_runner.py:799) clears
        ``_inflight_request_id`` when the coroutine is cancelled, which makes a
        subsequent ``cancel_verify()`` a no-op (verify_runner.py:814).

        Guards:
        - No-op for local leases (LocalRunner has no ``cancel_verify`` method).
        - Logs a warning on non-zero return code or unexpected exception so
          orphaned remote verify-merge processes are visible in logs for
          diagnosis, while never blocking the load-bearing abort return.
        """
        if not lease.is_local:
            try:
                rc = await lease.runner.cancel_verify()
                if rc:
                    logger.warning(
                        'Task %s: remote cancel_verify returned rc=%d '
                        '(remote verify-merge process may be orphaned)',
                        task_id, rc,
                    )
            except Exception as exc:
                logger.warning(
                    'Task %s: remote cancel_verify raised (remote verify-merge '
                    'process may be orphaned): %s',
                    task_id, exc,
                )

    async def _run_inflight_verify(
        self,
        item: SpeculativeItem,
        lease: Any,  # HostLease
    ) -> InflightVerifyResult:
        """Run the verify portion for one in-flight item.

        This is the VERIFY HALF of _verify_and_advance (the CAS half is
        _finalize_inflight).  Warm-swap and _run_post_merge_verify are run
        here; CAS advance_main and lease release are deferred to _finalize_inflight.

        LOCAL lease (lease.is_local=True):
            Warm-swap runs: _verify_attempt_count incremented,
            _acquire_warm_verify_worktree called, runner=None so the internal
            LocalRunner sees the POST-swap merge_wt (byte-identical single-host
            path).

        REMOTE lease (lease.is_local=False):
            No warm-swap, no _verify_attempt_count increment.
            _run_post_merge_verify(runner=lease.runner) dispatches on the
            injected RemoteRunner.

        Returns an InflightVerifyResult; does NOT resolve req.result (that is
        _finalize_inflight's job) except on exception (error path resolves
        immediately so the item does not stall the queue).

        Abort-poll: wraps the inner verify in a VERIFY_ABANDON_POLL_SECS poll
        loop so sole-waiter abandon and operator-halt can abort mid-verify.
        Abandon-wins ordering matches _verify_and_advance: abandon (trigger 1)
        is checked before halt (trigger 2) when both land simultaneously.
        """
        req = item.request
        merge_wt = item.merge_wt
        assert merge_wt is not None
        assert item.merge_result is not None
        merge_commit = item.merge_result.merge_commit
        assert merge_commit is not None
        merge_commit = merge_commit.strip()

        _warm_results: dict[str, str] = {}
        _is_warm_path = False
        _warm_capture: list[VerifyResult] = []
        _spec_warm: bool = False  # set when acquire_spec_lane returns warm=True

        try:
            if lease.is_local:
                # ── LOCAL path: persistent warm-merge-verify worktree swap ──
                # Mirrors _verify_and_advance (PRD §10 κ): increment the attempt
                # counter, check the safety valve, swap to the warm persistent
                # path if eligible, deregister the ephemeral worktree if swapped.
                #
                # Moved inside try: if _acquire_warm_verify_worktree raises (it is
                # a git I/O operation that can fail), the except Exception handler
                # below cleans merge_wt and returns a 'blocked' InflightVerifyResult.
                # Before this fix the exception escaped uncaught, was re-raised in
                # _finalize_inflight via `await entry.verify_task`, and reached the
                # _verifier_loop finalize-head BaseException handler — which resolved
                # the future as 'blocked' but did NOT clean the ephemeral merge
                # worktree (a regression vs. the old _verifier_loop except clause).
                self._verify_attempt_count += 1
                _due = _safety_valve_due(
                    self._verify_attempt_count,
                    req.config.git.persistent_merge_worktree_safety_valve_every_n,
                )
                merge_wt, _spec_warm = await _acquire_warm_verify_worktree(
                    self._git_ops, req, merge_wt, merge_commit,
                    safety_valve_due=_due,
                    speculative=item.speculative,
                )
                assert merge_wt is not None
                if merge_wt is not item.merge_wt:
                    self._deregister_owned_merge_worktree(item.merge_wt)
                _is_warm_path = (
                    (req.config.git.persistent_merge_worktree and not _due)
                    or (req.config.git.merge_spec_warm_lane_pool and _spec_warm)
                )

            # NOTE: _verify_item/_verify_phase/_verify_started_at are vestigial
            # fields retained for future single-host shim compatibility only.
            # snapshot() no longer reads them — all verify observability derives
            # from self._inflight (InflightEntry.phase) and self._remerging_item.
            # These assignments are write-only; nothing currently reads them.
            self._verify_item = item
            self._verify_phase = 'verifying'
            self._verify_started_at = time.time()
            logger.info(
                f'Task {req.task_id}: verify start (merge={merge_commit[:8]}, '
                f'worktree={merge_wt.name})'
            )

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
                runner=None if lease.is_local else lease.runner,
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
                # DROP the request.  Checked first so a gave-up waiter wins
                # over the operator-halt re-queue when both hold simultaneously.
                if self._request_abandoned(req):
                    await self._abort_remote_verify(lease, req.task_id)
                    verify_task.cancel()
                    with contextlib.suppress(BaseException):
                        await verify_task
                    await self._release_or_cleanup(merge_wt, spec_warm=_spec_warm)
                    return InflightVerifyResult(
                        outcome=None,
                        merge_wt=None,
                        status=InflightStatus.DROPPED,
                    )
                # Abort trigger 2 — operator halt: terminate the in-flight
                # verify and RE-QUEUE the merge for re-verify after un-halt.
                # req.result is left pending; per-task retry counters untouched.
                if self._operator_halt.is_set():
                    logger.warning(
                        'Task %s: operator halt — aborting in-flight verify '
                        'and re-queuing merge for re-verify after un-halt',
                        req.task_id,
                    )
                    await self._abort_remote_verify(lease, req.task_id)
                    verify_task.cancel()
                    with contextlib.suppress(BaseException):
                        await verify_task
                    await self._release_or_cleanup(merge_wt, spec_warm=_spec_warm)
                    self._queue.put_nowait(req)
                    return InflightVerifyResult(
                        outcome=None,
                        merge_wt=None,
                        status=InflightStatus.REQUEUED,
                    )
        except RunnerUnavailable as exc:
            # Remote transport failure: do NOT clean merge_wt — the item will
            # be re-dispatched on a free host (local fallback) with its worktree
            # intact.  _finalize_inflight calls quarantine_and_release so the
            # dead remote is quarantined before the re-dispatch.
            logger.warning(
                'Task %s: remote runner unavailable (merge=%s) — '
                'will re-dispatch on another host',
                req.task_id, merge_commit[:8],
            )
            return InflightVerifyResult(
                outcome=None,
                merge_wt=merge_wt,
                status=InflightStatus.RUNNER_UNAVAILABLE,
                reason=str(exc),
            )
        except Exception as exc:
            logger.info(
                f'Task {req.task_id}: verify end '
                f'(merge={merge_commit[:8]}, error)'
            )
            await self._release_or_cleanup(merge_wt, spec_warm=_spec_warm)
            err_outcome = MergeOutcome('blocked', reason=f'Verification error: {exc}')
            if not req.result.done():
                req.result.set_result(err_outcome)
            return InflightVerifyResult(outcome=err_outcome, merge_wt=None)

        if out is None:
            logger.info(
                f'Task {req.task_id}: verify end (merge={merge_commit[:8]}, '
                f'passed=True)'
            )
            if _warm_capture:
                _warm_results = parse_per_test_results(_warm_capture[0].test_output or '')
                if not _warm_results and req.config.git.warm_verify_shadow_compare:
                    _alarm_warm_shadow_unparseable(
                        self._escalation_queue,
                        merge_commit,
                        _warm_capture[0].test_output or '',
                    )
            return InflightVerifyResult(outcome=None, merge_wt=merge_wt, warm_results=_warm_results, spec_warm=_spec_warm)

        if out.verify_skipped:
            logger.info(
                f'Task {req.task_id}: verify skipped: low disk '
                f'(merge={merge_commit[:8]})'
            )
        else:
            logger.info(
                f'Task {req.task_id}: verify end (merge={merge_commit[:8]}, '
                f'passed=False)'
            )
        return InflightVerifyResult(outcome=out, merge_wt=merge_wt, spec_warm=_spec_warm)

    async def _finalize_inflight(self, entry: InflightEntry) -> bool:
        """Run the CAS advance_main + post-advance work for one in-flight item.

        This is the FINALIZE HALF of _verify_and_advance (the VERIFY HALF is
        _run_inflight_verify).  Handles all entry kinds in one flat try/finally:

          PASSTHROUGH   — immediate_outcome (conflict/already_merged/blocked):
                          deliver in submission order (respecting already_delivered),
                          set _n_failed from outcome.status.
          FAIL/skip     — vr.outcome is not None: clean merge_wt, resolve req,
                          _n_failed=True, return False.
          DROPPED       — sole-waiter abandoned: cancel_and_release, _n_failed=True.
          REQUEUED      — operator halt (item already back on _queue):
                          cancel_and_release, _n_failed=True.
          PASS          — vr.outcome is None (or verify_task=None for compat shim):
                          CAS advance_main loop.

        Lease release and speculation-slot release happen in the single finally:
          · _cancel_release=True  → cancel_and_release  (DROPPED/REQUEUED)
          · _cancel_release=False → release              (FAIL/PASS)
          · _skip_release=True    → no allocator call    (PASSTHROUGH: lease is None)
        _n_failed is carried in _n_failed_val (None → defer to 'not advanced').

        Returns True iff main was advanced successfully, False otherwise.

        warm_results are taken from vr.warm_results (the per-test map returned
        by _run_inflight_verify) and threaded into _maybe_schedule_shadow_compare,
        restoring the same-candidate warm-vs-cold shadow compare (PRD §10 invariant
        6b).  When verify_task is None (compat shim / pre-established pass) there is
        no vr, so warm_results defaults to {}, matching the pre-refactor behaviour.
        """
        item = entry.item
        req = item.request

        advanced = False
        _skip_release = False     # True → no allocator call (passthrough / no lease)
        _cancel_release = False   # True → cancel_and_release; False → release
        _n_failed_val = None      # None → defer to 'not advanced' in finally (PASS path)

        try:
            # Finalize-head observability: set _finalizing_head so snapshot() surfaces
            # this entry while we await entry.verify_task.  There is no await between
            # _inflight.popleft() (caller) and this assignment (asyncio single-loop),
            # so the field's set window exactly covers the invisible finalize gap.
            # Set/clear are structurally symmetric (both inside this try/finally), so a
            # future edit adding a throwing statement before the try cannot leave
            # _finalizing_head permanently stale.
            self._finalizing_head = entry

            # ── Pre-dispatch sentinels (abandon / operator-halt) ─────────────────────
            # Handled inline in _dispatch_item: merge_wt already cleaned, req already
            # re-queued (REQUEUED_PREDISPATCH) or result already done (ABANDONED_PREDISPATCH).
            # Nothing to deliver; chain is stale → n_failed=True.
            if entry.status in (InflightStatus.ABANDONED_PREDISPATCH, InflightStatus.REQUEUED_PREDISPATCH):
                _n_failed_val = True
                _skip_release = True  # no lease
                return False

            # ── (b) PASSTHROUGH ─────────────────────────────────────────────
            # immediate_outcome entries (conflict/already_merged/blocked) with no
            # real verify task; deliver in submission order.
            if entry.passthrough_outcome is not None:
                if not item.already_delivered and not req.result.done():
                    req.result.set_result(entry.passthrough_outcome)
                # Mirrors original verifier-loop line :6473:
                #   n_failed = item.immediate_outcome.status not in ('done', 'already_merged')
                _n_failed_val = (
                    entry.passthrough_outcome.status not in ('done', 'already_merged')
                )
                _skip_release = True  # passthrough entries have no lease
                return entry.passthrough_outcome.status in ('done', 'already_merged')

            # ── Await verify task (if any) ───────────────────────────────────
            # verify_task=None means PASS was pre-established (compat shim /
            # step-12 tests where entry is constructed with a known-pass worktree).
            vr: InflightVerifyResult | None = None
            if entry.verify_task is not None:
                vr = await entry.verify_task

            # ── (c) DROPPED / REQUEUED sentinels ────────────────────────────
            if vr is not None and vr.status in (InflightStatus.DROPPED, InflightStatus.REQUEUED):
                _cancel_release = True
                _n_failed_val = True  # abandon / operator-halt → chain stale
                return False

            # ── (d) RUNNER_UNAVAILABLE ───────────────────────────────────────
            # Remote runner died.  Quarantine the host (so acquire() skips it)
            # and re-dispatch the item on any free host — degrading gracefully to
            # serial-local rather than stalling.  Not a chain failure: _n_failed
            # stays False (the merge is still a valid candidate; it just needs
            # re-verify on a healthy host).
            #
            # Downstream speculative entries in _inflight correctness:
            # This function returns False, so _verifier_loop's head-failure
            # cascade (`if not _head_advanced and self._inflight:`) fires for
            # any downstream speculative entries.  The cascade cancels each
            # downstream verify task, cleans its worktree, re-merges it against
            # actual main (via _remerge), and front-queues it on _redispatch in
            # submission order.  After the cascade, _inflight is empty and all
            # re-merged items are on _redispatch in the correct order.
            # Downstream entries do NOT remain in _inflight with stale commits;
            # the cascade is the correctness mechanism, not the CAS backstop alone.
            if vr is not None and vr.status == InflightStatus.RUNNER_UNAVAILABLE:
                _skip_release = True   # quarantine_and_release handles the lease
                _n_failed_val = False  # not a chain failure
                if entry.lease is not None and self._host_allocator is not None:
                    await self._host_allocator.quarantine_and_release(entry.lease)
                # ── Unavailability tracker + alarm (task 1795) ──────────────
                # Record the failure in the per-host streak tracker.  If the
                # streak reaches the configured threshold (or the time-based
                # threshold is exceeded via the reprobe loop) fire a dedup'd
                # L1 'verify_host_unreachable' escalation so an operator is
                # notified.  The dedup guard (has_open_l1) ensures exactly one
                # open alarm per host per downtime episode regardless of how
                # many RU events accumulate.
                if entry.lease is not None:
                    _ru_host = entry.lease.name
                    _ru_reason = vr.reason or '<unknown>'
                    _ru_now = time.time()  # capture once for consistent timestamps
                    _should_escalate = self._record_runner_unavailable(
                        _ru_host, _ru_reason, _ru_now
                    )
                    if _should_escalate:
                        _ru_entry = self._runner_unavailable.get(_ru_host)
                        _alarm_verify_host_unreachable(
                            self._escalation_queue,
                            _ru_host,
                            _ru_reason,
                            streak=_ru_entry.streak if _ru_entry is not None else 1,
                            duration_s=(
                                _ru_now - _ru_entry.first_unavailable_at
                                if _ru_entry is not None else 0.0
                            ),
                            event_store=self._event_store,
                        )
                # Re-merge against actual main and front-insert into _redispatch
                # so the item is retried before any newer queue arrivals.
                # The head-failure cascade (fired because this returns False) will
                # handle any downstream entries still in _inflight.
                _remerged_ru = await self._remerge(
                    entry.item.request, entry.item.started_monotonic,
                )
                self._redispatch.appendleft(_remerged_ru)
                return False

            # ── (a) FAIL / skip ──────────────────────────────────────────────
            if vr is not None and vr.outcome is not None:
                fail_merge_wt = vr.merge_wt
                await self._release_or_cleanup(fail_merge_wt, spec_warm=vr.spec_warm)
                self._resolve_or_drop_abandoned(req, vr.outcome)
                _n_failed_val = True
                return False

            # ── PASS: CAS advance_main ───────────────────────────────────────
            # Reached when verify passed (vr.outcome is None) or verify_task=None.
            merge_wt = entry.merge_wt if vr is None else vr.merge_wt
            assert merge_wt is not None
            assert item.merge_result is not None
            merge_commit = item.merge_result.merge_commit
            assert merge_commit is not None
            merge_commit = merge_commit.strip()

            # Thread warm_results from the warm-verify path through to
            # _maybe_schedule_shadow_compare so the same-candidate shadow compare
            # fires.  When vr is None (verify_task=None compat path) there is no
            # warm run, so default to {} — matching pre-refactor behaviour.
            _warm_results: dict[str, str] = vr.warm_results if vr is not None else {}

            # Precompute spec_warm for cleanup routing — True when the verify ran
            # in a warm _spec- lane (not an ephemeral throwaway worktree).
            _vr_spec_warm = (vr is not None and vr.spec_warm)

            # Short-circuit: if abandonment landed while verify completed,
            # skip the expensive CAS loop (mirrors _verify_and_advance :6934).
            if self._request_abandoned(req):
                await self._release_or_cleanup(merge_wt, spec_warm=_vr_spec_warm)
                return False

            # ── Step 5: CAS advance_main ──────────────────────────────────
            self._verify_phase = 'finalizing'
            entry.phase = 'finalizing'   # per-entry source of truth for snapshot()
            current_sha = merge_commit
            while True:
                result = await self._git_ops.advance_main(
                    current_sha, merge_wt,
                    branch=req.branch,
                    max_attempts=req.config.max_advance_attempts,
                    expected_main=item.base_sha,
                    reverify_on_rebase=True,
                )

                if result == 'advanced':
                    self._gate_retries.pop(req.task_id, None)
                    await self._release_or_cleanup(merge_wt, spec_warm=_vr_spec_warm)
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
                    if outcome.status == 'done':
                        # ι=1894: record clean landing, pop drift base for this req
                        self._note_merge_landing(req.request_id)
                        if (
                            outcome.merge_sha is not None
                            and self._on_merge_landed is not None
                        ):
                            try:
                                # PRIMARY in-process trigger for the offline
                                # deep-test lane: harness._note_merge_all fans
                                # out to _note_offline_lane (task 1951, β1).
                                # See the constructor comment above for the
                                # FALLBACK (orchestrator-down landings).
                                await self._on_merge_landed(
                                    req.task_id, item.base_sha, outcome.merge_sha
                                )
                            except Exception:
                                logger.warning(
                                    'on_merge_landed hook raised for task %s; ignoring (fail-open)',
                                    req.task_id,
                                    exc_info=True,
                                )
                        await _maybe_schedule_shadow_compare(
                            self, self._git_ops, req, merge_commit,
                            warm_results=_warm_results,
                            escalation_queue=self._escalation_queue,
                            event_store=self._event_store,
                        )
                        await _maybe_run_drift_check(
                            self, self._git_ops, req, merge_commit,
                        )
                        # D10 promote-provenance: advance the rolling warm base
                        # from the _merge-verify lane's target (the just-landed
                        # commit's warm build).  Gated on the warm worktree name
                        # so the refresh only fires when the verify ran in the
                        # persistent _merge-verify lane (knob on, safety-valve
                        # not due) — i.e. the source is the post-CAS confirmed
                        # head.  On the cold/ephemeral round the base stays at
                        # H-1 (benign-degrading, never blocks dispatch).
                        # refresh_warm_base is best-effort and never raises.
                        if merge_wt.name == PERSISTENT_MERGE_WORKTREE_NAME:
                            await self._git_ops.refresh_warm_base()
                    advanced = True
                    return True

                if result == 'rebased_pending_reverify':
                    # ι=1894: count this as a merge retry — main advanced under the
                    # request, so a fresh re-verify is required.  Both cas_failed and
                    # rebased_pending_reverify are included in retries_per_landing (see
                    # _note_merge_retry docstring for the rationale).
                    self._note_merge_retry()
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
                    entry.phase = 'gate_reverify'
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
                        if not req.result.done():
                            req.result.set_result(gate)
                        return False

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
                        await self._release_or_cleanup(merge_wt, spec_warm=_vr_spec_warm)
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

                    # I3 (task 1990): replace-only rebuild — dataclasses.replace
                    # copies EVERY field except base_sha, so this can no longer
                    # silently drop one (the task-1928 bug class this guards
                    # against; merged_branch_tip was the field 1928 had to add
                    # back by hand).  Re-invokes __post_init__, re-validating
                    # the rebuilt REAL item.
                    item = dataclasses.replace(item, base_sha=rebased_onto)
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
                    self._verify_phase = 'finalizing'
                    entry.phase = 'finalizing'
                    continue

                if result in _HALT_ADVANCE_RESULTS and self._request_abandoned(req):
                    await self._release_or_cleanup(merge_wt, spec_warm=_vr_spec_warm)
                    if result in ('unmerged_state', 'pop_conflict_no_advance'):
                        self._cas_retries.pop(req.task_id, None)
                        self._gate_retries.pop(req.task_id, None)
                    return False
                if result != 'cas_failed':
                    self._gate_retries.pop(req.task_id, None)
                    await self._release_or_cleanup(merge_wt, spec_warm=_vr_spec_warm)
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
                # ι=1894: count CAS retry
                self._note_merge_retry()
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
                    await self._release_or_cleanup(merge_wt, spec_warm=_vr_spec_warm)
                    if not req.result.done():
                        req.result.set_result(MergeOutcome(
                            'blocked',
                            reason=(
                                f'CAS retry limit exhausted after '
                                f'{self.MAX_CAS_RETRIES} attempts for task {req.task_id}'
                            ),
                        ))
                    return False

                # Update base_sha to current main for retry.
                # I3 (task 1990): replace-only rebuild — see the
                # rebased_pending_reverify rebuild above for rationale.
                item = dataclasses.replace(item, base_sha=await self._git_ops.get_main_sha())
                logger.info(
                    f'Task {req.task_id}: CAS failed (attempt {total}/'
                    f'{self.MAX_CAS_RETRIES}), retrying'
                )
                _emit_merge_attempt(self._event_store, req.task_id, 'cas_retry', attempt=total, duration_ms=_elapsed_ms(item.started_monotonic))

        finally:
            # Always: release the host lease (unless passthrough / already skipped),
            # release the speculation slot iff speculative, and update _n_failed.
            if not _skip_release and entry.lease is not None and self._host_allocator is not None:
                if _cancel_release:
                    await self._host_allocator.cancel_and_release(entry.lease)
                else:
                    await self._host_allocator.release(entry.lease)
            if entry.was_speculative:
                self._speculation_slot.release()
            # _n_failed_val is set by non-PASS branches; None means use 'not advanced'
            # (the PASS-path semantics: True when CAS failed, False when advanced).
            self._n_failed = _n_failed_val if _n_failed_val is not None else not advanced
            # Clear finalize-head observability window — covers all return/exception paths.
            self._finalizing_head = None
            # ι=1894 amend: defensive _drift_base cleanup — idempotent pop that covers
            # all terminal exit paths (blocked/error/halt/wip_halted/cas-exhausted/
            # rebased-gate-failure).  _note_merge_landing and _note_conflict_detected
            # both use pop(key, None), so re-popping an already-cleared entry is a no-op.
            self._drift_base.pop(req.request_id, None)

    async def _dispatch_item(
        self,
        item: SpeculativeItem,
    ) -> InflightEntry | None:
        """Apply pickup logic and dispatch one item to a host verify slot.

        Returns an InflightEntry on success (to be appended to self._inflight),
        or None when no host slot is currently available (caller should put the
        item back on self._redispatch unchanged — counts_against_cap already
        released, so caller must clear it first via dataclasses.replace).

        Handles in order:
          1. Mechanism 1 cap release (counts_against_cap).
          2. Pre-dispatch abandon: cleanup merge_wt; return ABANDONED_PREDISPATCH
             passthrough entry (req.result already done; finalize no-ops delivery).
          3. Pre-dispatch operator-halt: cleanup merge_wt, re-queue req on _queue;
             return REQUEUED_PREDISPATCH passthrough entry (req.result still pending
             — finalize MUST NOT deliver an outcome; the pre-dispatch branch in
             _finalize_inflight guards this).
          4. Immediate outcome: return passthrough InflightEntry (no lease / task).
          5. Real item: fast-path None if all hosts busy; chain re-merge (Mechanism 2
             + chain-invalidation) ONLY when self._inflight is empty; acquire host;
             launch asyncio.ensure_future(_run_inflight_verify); return entry.

        Chain-invalidation re-merge is gated on `not self._inflight` so it only
        fires when the predecessor has already been finalized — byte-identical for
        single-host (inflight always empty at dispatch) and correct for multi-host
        (speculative-on-in-flight items launch as-is; finalize's head-failure handler
        aborts + re-merges downstream entries).

        Mechanism 2 `main_advanced` guard added: `not item.speculative` prevents it
        from firing for speculative-on-in-flight items whose base_sha intentionally
        != current_main (γ design decision 3).
        """
        req = item.request

        # ── Mechanism 1: release merger-ahead cap ON-DRAIN ─────────────────
        # Mirrors original _verifier_loop :6327: release BEFORE any branching so
        # every drain path (normal verify, passthrough, abandon, halt) is covered.
        # cap release happens exactly once: here for items from _verifier_queue;
        # items put back onto _redispatch have counts_against_cap cleared.
        if item.counts_against_cap:
            self._merge_ahead_cap.release()

        # ── Pre-dispatch abandon ────────────────────────────────────────────
        if self._request_abandoned(req):
            if item.merge_wt is not None:
                with contextlib.suppress(BaseException):
                    await self._cleanup_owned_merge_worktree(item.merge_wt)
            self._remerge_occurred = False  # abandon → reset chain flag
            return InflightEntry(
                item=item,
                lease=None,
                verify_task=None,
                merge_wt=None,
                was_speculative=item.speculative,
                phase='abandoned',
                status=InflightStatus.ABANDONED_PREDISPATCH,
            )

        # ── Pre-dispatch operator-halt ──────────────────────────────────────
        # immediate_outcome items (trains / already-decided) are NOT halted here;
        # they fall through to the passthrough branch so they resolve in order.
        if self._operator_halt.is_set() and item.immediate_outcome is None:
            if item.merge_wt is not None:
                with contextlib.suppress(BaseException):
                    await self._cleanup_owned_merge_worktree(item.merge_wt)
            self._queue.put_nowait(req)
            self._remerge_occurred = False  # halt → reset chain flag
            return InflightEntry(
                item=item,
                lease=None,
                verify_task=None,
                merge_wt=None,
                was_speculative=item.speculative,
                phase='halted',
                status=InflightStatus.REQUEUED_PREDISPATCH,
            )

        # ── Immediate outcome (conflict / already_merged / blocked) ────────
        if item.immediate_outcome is not None:
            self._remerge_occurred = False  # passthrough → reset chain flag
            return InflightEntry(
                item=item,
                lease=None,
                verify_task=None,
                merge_wt=None,
                was_speculative=item.speculative,
                phase='passthrough',
                passthrough_outcome=item.immediate_outcome,
            )

        # ── Real item: host acquire + verify dispatch ───────────────────────
        # Fast-path: if no host is free RIGHT NOW, return None so the caller
        # puts the item back on _redispatch (counts_against_cap already cleared
        # above; caller must clear it on the item before putting back).
        # Checked BEFORE the potentially-expensive _remerge call so no work is
        # done for an item that will be re-tried on a free host.
        allocator = self._ensure_host_allocator(req.config)
        if allocator.free_host_count() == 0:
            return None

        # Capture the speculative flag BEFORE any _remerge reassignment so
        # the InflightEntry carries the ORIGINAL speculative state for slot
        # release (same pattern as old loop's item_was_speculative).
        item_was_speculative = item.speculative
        iteration_did_remerge = False

        # ── Chain re-merge (Mechanism 2 + chain-invalidation) ──────────────
        # Only when no REAL verify task is running (predecessor already finalized).
        # Passthrough entries (verify_task=None, lease=None) are already decided;
        # they don't represent an unknown predecessor outcome, so they don't block
        # the chain re-merge.
        #   Single-host: always true at dispatch → byte-identical.
        #   Multi-host: speculative-on-in-flight → skip; handled by finalize
        #               head-failure cascade (step-20).
        #
        # LATE-ARRIVAL ATTACH INTERACTION (task 1862 step-4 locking guard):
        # A late-arriving B attached to in-flight predecessor A via
        # pending_spec_base (_merger_loop step-2) composes correctly with this
        # gate in every dispatch race — no spurious remerge on the attach path:
        #
        # (a) Predecessor A still verifying at B's dispatch:
        #     → _has_inflight_verify True → entire block below skipped.
        #     → B dispatches as-is against A's merge commit (spec_base=A.commit).
        #
        # (b) Predecessor A landed CLEAN before B dispatched:
        #     → _has_inflight_verify False → block entered.
        #     → B.speculative=True, _n_failed=False (A landed), _remerge_occurred=False.
        #     → First branch (`item.speculative and _n_failed|_remerge_occurred`): False.
        #     → Second branch: `not item.speculative` γ guard False (B is speculative).
        #     → remerge_reason remains None → NO remerge → B dispatches against
        #       A's commit (== main-after-A) → advance_main clean CAS 'advanced'
        #       → _reverify_rebased_tree unreachable (DONE-WHEN 3, task 1862).
        #
        # (c) Predecessor A FAILED before B dispatched:
        #     → _has_inflight_verify False, _n_failed=True.
        #     → First branch fires with reason=previous_failed → _remerge vs
        #       actual main → existing head-failure cascade invalidates B
        #       (DONE-WHEN 4, task 1862; tested by step-5/step-6).
        #
        # The `not item.speculative` γ guard on Mechanism 2 is the speculative
        # carve-out that makes case (b) provable: even if B's base_sha happened
        # to differ from current_main, Mechanism 2 would not fire for speculative
        # items — preventing any main_advanced remerge on the clean-landed path.
        _has_inflight_verify = any(e.verify_task is not None for e in self._inflight)
        if not _has_inflight_verify:
            remerge_reason: str | None = None
            if item.speculative and (self._n_failed or self._remerge_occurred):
                remerge_reason = (
                    'previous_failed' if self._n_failed else 'chain_invalidated'
                )
            elif (
                # γ guard: Mechanism 2 staleness check for REAL (non-speculative) items
                # only.  Speculative items (including late-arrival attached via
                # pending_spec_base) must never be remerged here on base_sha mismatch —
                # their base_sha is intentionally the predecessor's commit, not current
                # main; the predecessor's finalize advances main to equal it (case b
                # above) so no staleness exists.  This guard is the explicit speculative
                # carve-out required by task 1862 step-4.
                not item.speculative
                and item.immediate_outcome is None
                and item.merge_result is not None
                and not isinstance(req, GroupMergeRequest)
            ):
                # Mechanism 2: check staleness at pickup for non-speculative items.
                current_main = await self._git_ops.get_main_sha()
                if item.base_sha != current_main:
                    remerge_reason = 'main_advanced'

            if remerge_reason is not None:
                iteration_did_remerge = True
                # Set _remerging_item so snapshot() surfaces this request during
                # the remerge window (item is popped from queue but not yet in
                # _inflight, so without this it is invisible to all observability).
                # Cleared to None immediately after _remerge() returns.
                self._remerging_item = req
                self._verify_item = item
                self._verify_phase = 'remerging'
                if item.merge_wt:
                    await self._cleanup_owned_merge_worktree(item.merge_wt)
                self._emit_speculative(
                    EventType.speculative_discard, req.task_id,
                    reason=remerge_reason,
                )
                logger.info(
                    'Task %s: discarding stale merge (%s), re-merging against actual main',
                    req.task_id, remerge_reason,
                )
                item = await self._remerge(req, item.started_monotonic)
                self._remerging_item = None
                self._verify_item = item

                # After remerge the new item may itself carry an immediate_outcome
                # (e.g. conflict during remerge, skip_verify=True, or a train slot).
                # Return it as a passthrough so _run_inflight_verify is never called
                # with merge_wt=None.
                if item.immediate_outcome is not None:
                    self._remerge_occurred = iteration_did_remerge
                    return InflightEntry(
                        item=item,
                        lease=None,
                        verify_task=None,
                        merge_wt=None,
                        was_speculative=item_was_speculative,
                        phase='passthrough',
                        passthrough_outcome=item.immediate_outcome,
                        started_at=time.time(),
                    )

        # Propagate chain-invalidation flag for the next dispatch call.
        self._remerge_occurred = iteration_did_remerge

        # ── Acquire host slot ───────────────────────────────────────────────
        # The local_factory is a closure over `item` (possibly the re-merged
        # item) and `req`.  It builds a LocalRunner; _run_inflight_verify will
        # override merge_wt via warm-swap on the local path, so the factory's
        # merge_wt is a reasonable initial value.
        # NOTE: the factory is called ONLY when the local slot is free (prefer-local
        # policy in HostAllocator.acquire); the remote path uses the remote runner
        # directly without calling the factory.
        _item_for_factory = item
        _req_for_factory = req

        def _local_factory() -> LocalRunner:
            assert _item_for_factory.merge_wt is not None, \
                'dispatch path: merge_wt must be non-None for a real item'
            return LocalRunner(
                _item_for_factory.merge_wt,
                _req_for_factory.config,
                _req_for_factory.module_configs,
                None,   # task_files — derived inside _run_post_merge_verify
                run_scoped=run_scoped_verification,
                run_unscoped=_run_unscoped_typechecks,
                task_id=_req_for_factory.task_id,
            )

        lease = await allocator.acquire(_local_factory)
        if lease is None:
            # Should not happen (free_host_count > 0 was checked above with no
            # intervening await that could yield to a concurrent dispatch — asyncio
            # is single-threaded and _dispatch_item is the only acquirer).
            # Return None defensively so the caller puts the item back.
            return None

        # ── ε=1890 log-only §5.3 guard: verify base must be frozen-prefix tip ──
        # Fail-open: fetch main_sha in a try/except so a transient git error
        # never blocks verify dispatch.  The guard is purely observational —
        # it never changes control flow.  NOT wired into _verify_and_advance
        # (the compat shim used by direct-call tests) to keep shim tests green.
        try:
            _guard_main_sha = await self._git_ops.get_main_sha()
            self._warn_if_verify_base_not_frozen_tip(item, _guard_main_sha)
        except Exception:
            pass  # fail-open: skip the check on any git error

        # ── Launch background verify task ────────────────────────────────────
        verify_task: asyncio.Task = asyncio.ensure_future(  # type: ignore[type-arg]
            self._run_inflight_verify(item, lease)
        )

        return InflightEntry(
            item=item,
            lease=lease,
            verify_task=verify_task,
            merge_wt=item.merge_wt,
            was_speculative=item_was_speculative,
            phase='verifying',
            started_at=time.time(),
        )

    async def _verify_and_advance(self, item: SpeculativeItem) -> bool:
        """Thin compat shim: acquire LOCAL lease → _run_inflight_verify → _finalize_inflight.

        Retained so the ~18 tests calling _verify_and_advance(item) directly stay
        green via the single-item trust-anchor path (β decision 5, γ design decision 5).

        Acquires a local lease (prefer-local policy), wraps the result in an
        InflightEntry, and awaits _finalize_inflight.  Returns True iff main was
        advanced (matches the original _verify_and_advance return contract).

        was_speculative=False: the shim path never manages the speculation semaphore
        (the old _verifier_loop managed it in its finally; the new _dispatch_item/
        _finalize_inflight manage it via InflightEntry.was_speculative in the loop).
        Direct-call tests that care about speculation test the full loop.
        """
        req = item.request
        allocator = self._ensure_host_allocator(req.config)

        _item_for_factory = item
        _req_for_factory = req

        def _local_factory() -> LocalRunner:
            assert _item_for_factory.merge_wt is not None, \
                'shim path: merge_wt must be non-None for a real item'
            return LocalRunner(
                _item_for_factory.merge_wt,
                _req_for_factory.config,
                _req_for_factory.module_configs,
                None,
                run_scoped=run_scoped_verification,
                run_unscoped=_run_unscoped_typechecks,
                task_id=_req_for_factory.task_id,
            )

        lease = await allocator.acquire(_local_factory)
        if lease is None:
            # Fallback: force-acquire the local slot (shim path; no competing acquirers
            # in direct-call tests).
            lease = allocator.acquire_local(_local_factory)
        if lease is None:
            # Still None: all slots parked.  Resolve as blocked and return.
            if not req.result.done():
                req.result.set_result(MergeOutcome(
                    'blocked', reason='No verify host available (shim path)',
                ))
            return False

        verify_task: asyncio.Task = asyncio.ensure_future(  # type: ignore[type-arg]
            self._run_inflight_verify(item, lease)
        )

        entry = InflightEntry(
            item=item,
            lease=lease,
            verify_task=verify_task,
            merge_wt=item.merge_wt,
            was_speculative=False,  # shim does not manage the speculation slot
            phase='verifying',
            started_at=time.time(),
        )

        return await self._finalize_inflight(entry)
