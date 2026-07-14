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
from collections.abc import Awaitable, Callable, Collection, Iterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from orchestrator.dry_run_unblock import run_dry_run_unblock
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import (
    PERSISTENT_MERGE_WORKTREE_NAME,
    AdvanceOutcome,
    GitOps,
    MergeResult,
    WorktreeMissing,
    _run,
)
from orchestrator.landed_outbox import LandedOutbox, LandedRow, MergeProvenance
from orchestrator.merge_disposition import MergeFailureDisposition
from orchestrator.merge_drift import (  # noqa: F401  re-export shim
    _maybe_run_drift_check,
    _run_drift_check,
)
from orchestrator.merge_gates import (  # noqa: F401  re-export shim
    _OVERLAP_GIT_ERROR_SENTINEL,
    DROPPED_PLAN_TARGETS_REASON_PREFIX,
    PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
    POST_ADVANCE_GATES,
    POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
    POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
    DropGuardResult,
    Gate,
    GateVerdict,
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
    newest_content_mtime,
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
from orchestrator.merge_speculation_controller import (  # noqa: F401  re-export shim
    PermitLedger,
    SpeculationController,
)
from orchestrator.merge_types import (  # noqa: F401  re-export shim
    _INFLIGHT_MERGE_ETA_ESTIMATE_SECS,
    CapPermit,
    Decided,
    DecidedItem,
    GroupMergeRequest,
    InflightEntry,
    InFlightMergeRegistry,
    InflightStatus,
    InflightVerifyResult,
    ItemLifecycleState,
    MainHealthAutoHealRegistry,
    MergeBounceRegistry,
    MergeDispatchResult,
    MergedOk,
    MergeOutcome,
    MergeReadyPredicate,
    MergeRequest,
    OutcomeKind,
    RealMergeItem,
    SoloVerifyResult,
    SpeculativeItem,
    TerminalOutcomeRecord,
    TerminalOutcomeRetention,
    TrainCallbackFactory,
    TrainCallbacks,
    WaiterRecord,
    _HostUnavailability,
    _InFlightEntry,
    item_merge_wt,
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
from orchestrator.task_status import WORKFLOW_PRESERVE_STATUSES
from orchestrator.unblock_types import BlockClass
from orchestrator.verify import (
    VerifyResult,
    _derive_task_files_from_git,
    run_scoped_verification,
    run_verification,
    verify_failure_is_preexisting_on_main,
)
from orchestrator.verify_categories import PREEXISTING_BREAK_SKIP_CATEGORIES
from orchestrator.verify_runner import (
    UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY,
    HostAllocator,
    LocalRunner,
    RemoteRunner,
    RunnerUnavailable,
    VerifyRunner,
    VerifyRunnerPool,
    build_merge_verify_spec,
    is_flock_contention_failure,
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
``self._merge_ahead_ledger.acquire()`` (task 2161/θ: ledger-mediated, wraps
``_merge_ahead_cap``) until the Verifier drains that item, at which point it
re-reads a fresh main HEAD for the next merge.  Values in [1, 2] are safe;
higher values allow more build-ahead but increase staleness risk.

Cap invariants (all verified by integration tests):
- Acquired at the single success-enqueue site in _merger_loop for non-speculative
  blocking-path items (trains continue before this site; speculative items are
  governed by _speculation_slot instead), stamped onto the item's
  :attr:`~orchestrator.merge_types.RealMergeItem.cap_permit`.
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

_DEBUG_ASSERTS: bool = os.environ.get('ORCH_DEBUG_ASSERTS', '') == '1'
"""Module-level single-writer debug-assert flag (task 1999 / MQ-invariants ξ, I7).

Seeded from the ``ORCH_DEBUG_ASSERTS`` env var at import time; gates
:meth:`SpeculativeMergeWorker._assert_single_writer`.  Read as a module
global (not re-read from ``os.environ`` per call) so tests/conftest can flip
it via ``monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)`` without
import-order fragility.  Off by default: zero production overhead beyond a
single module-bool branch at each call site."""


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
# ``verify_classify.classify_failure``; compile_error covers the tsc/type/lint class.
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

Shared between :class:`SpeculativeMergeWorker` and the retired serial
worker's test-local reference (see :class:`_TrainMergeHost`) to avoid
silent divergence: if the set of halt-triggering results ever changes,
updating this single constant propagates to both automatically."""


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


def _build_main_health_outcome(verify: VerifyResult, probe_sha: str) -> MergeOutcome:
    """Build the main-health-red ``MergeOutcome`` for a confirmed pre-existing break.

    Pure function — no I/O, no side effects, no event emission.  Extracted
    (task 2564) from :func:`_classify_main_health_red` so the reason /
    fingerprint / ``failure_category`` / ``failure_cause_hint`` composition is
    shared verbatim between the synchronous probe path
    (``_classify_main_health_red``) and the deferred off-critical-path probe
    (``_run_deferred_main_health_probe``) — the two can never diverge.

    *probe_sha* is the bare-main SHA the probe actually tested against (fed
    into the dedupe fingerprint so concurrent failing merges against the same
    main HEAD fold to one escalation parent).
    """
    detail = verify.failure_report()
    suffix = (verify.cause_hint or verify.summary or '')[:160]
    reason = (
        f'{MAIN_HEALTH_RED_REASON_PREFIX} '
        f'(category={verify.category!r}): {suffix}'
    )
    if detail:
        reason = f'{reason}\n\n{detail}'
    return MergeOutcome(
        'blocked',
        reason=reason,
        failure_category=verify.category,
        failure_cause_hint=verify.cause_hint,
        dedupe_fingerprint=_main_health_fingerprint(
            verify.category or '', verify.cause_hint, probe_sha,
        ),
    )


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
    outcome = _build_main_health_outcome(verify, probe_sha)
    _emit_merge_attempt(event_store, req.task_id, OutcomeKind.main_health_red)
    return outcome


# Sentinel task_id prefix for a laptop-side flock-worktree-contention alarm
# (task 2307 β, PRD plans/laptop-warm-verify-flock-orphan-prd.md §8.2).
# Mirrors _VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX's per-host sentinel shape.
_VERIFY_WORKTREE_CONTENTION_SENTINEL_PREFIX = '__verify_worktree_contention__'


def _verify_worktree_contention_sentinel(host: str) -> str:
    """Return the per-host sentinel task_id for a flock-worktree-contention alarm."""
    return f'{_VERIFY_WORKTREE_CONTENTION_SENTINEL_PREFIX}{host}'


def _alarm_verify_worktree_contention(
    escalation_queue: Any,
    *,
    host: str,
    holder_pgid: int | None,
    waiter_pgid: int | None,
) -> None:
    """Submit a born-at-L2 escalation for a laptop-side flock-worktree-contention outcome.

    Fired when a remote verify (task 2306 α's ``make_flock_contention_result``)
    reports that ``.merge_verify.lock`` could not be acquired within the
    bounded wait — another verify invocation already holds the persistent
    warm merge-verify worktree.  The laptop CLI holds no escalation client
    (the escalation MCP server binds 127.0.0.1:8100), so the workstation
    files this on the laptop's behalf.

    The escalation is:

    * ``level=2`` / ``severity='critical'`` — born-at-L2 (PRD §8.2): routes
      straight to a human, bypassing the auto-watcher.
    * ``agent_role='orchestrator-verify-host-monitor'`` — the ``orchestrator-``
      prefix marks this as a harness sentinel so the escalation server never
      downgrades the severity (mirrors ``_emit_loop_terminal_escalation``).
    * ``category='verify_worktree_contention'``
    * ``task_id=_verify_worktree_contention_sentinel(host)``

    None-safe: returns immediately when *escalation_queue* is None.  Dedup:
    returns immediately when an open L2 already exists for this host's
    sentinel task_id.  An orphaned lock-holder (the PRD scenario) persists
    until manually killed, so without this guard every subsequent distinct
    merge routed to the host would file another L2 — a burst of duplicate
    criticals at the human while the orphan lives.  The check goes through
    ``get_by_task(sentinel, status='pending', level=2)`` rather than
    ``_alarm_verify_host_unreachable``'s ``has_open_l1``: that helper is
    hardcoded to ``level=1`` (see ``escalation/queue.py``) and would never
    match this level=2 escalation.  No event emission — the escalation is
    the sole user-observable signal (PRD §8.2).

    Args:
        escalation_queue: Live escalation queue or ``None``.
        host: Laptop host name that reported the contention.
        holder_pgid: pgid of the process holding the lock, or ``None`` when
            the holder's pgid file is absent or corrupt (α's fail-safe).
        waiter_pgid: pgid of the process that lost the race and reported.
    """
    if escalation_queue is None:
        return

    from escalation.models import Escalation  # local import — escalation optional dep

    sentinel = _verify_worktree_contention_sentinel(host)

    # Dedup: don't re-alarm while an open L2 already exists for this host.
    # has_open_l1 is hardcoded to level=1 (escalation/queue.py), so it would
    # never match this level=2 escalation; get_by_task is used directly.
    if escalation_queue.get_by_task(sentinel, status='pending', level=2):
        return

    # Rendered once and reused below so the detail and suggested_action
    # fields agree on how a fail-safe None pgid reads to the operator.
    holder_display = holder_pgid if holder_pgid is not None else '<unknown>'
    waiter_display = waiter_pgid if waiter_pgid is not None else '<unknown>'

    summary = (
        f'Flock contention on {host!r}: another verify holds the persistent '
        f'merge-verify worktree lock'
    )
    detail = (
        f'Host: {host}\n'
        f'Holder pgid: {holder_display}\n'
        f'Waiter pgid: {waiter_display}\n'
        '\n'
        f'The laptop-side verify on {host!r} could not acquire '
        '.merge_verify.lock within the bounded wait, so it reported '
        'contention instead of falling back to an ephemeral worktree. '
        'The merge is blocked pending investigation.'
    )

    esc = Escalation(
        id=escalation_queue.make_id(sentinel),
        task_id=sentinel,
        agent_role='orchestrator-verify-host-monitor',
        severity='critical',
        level=2,
        category='verify_worktree_contention',
        summary=summary,
        detail=detail,
        suggested_action=(
            f'Inspect the holder process (pgid {holder_display}) on {host!r} — '
            'confirm it is a legitimate in-progress verify and not an orphan. '
            'Kill an orphaned holder to release .merge_verify.lock, or wait '
            'for it to finish, then re-submit the blocked merge.'
        ),
    )
    escalation_queue.submit(esc)


@dataclasses.dataclass(frozen=True)
class _DryRunInvestigationHandles:
    """Opaque bundle of harness-owned handles for the merge-verify dry-run spawn.

    SpeculativeMergeWorker holds none of scheduler/mcp/usage_gate/cost_store
    itself (the harness owns all of them); this bundles them into one
    optional, default-``None`` param so ``_run_post_merge_verify`` stays a
    pure git engine (mirrors the ``train_callback_factory`` opaque-injection
    pattern — the worker never imports the scheduler).  Only the production
    ``SpeculativeMergeWorker._run_inflight_verify`` call site passes a live
    instance; the solo-reverify and train module-level callers pass nothing,
    so their spawns automatically no-op.

    ``background_tasks`` is the SAME ``set`` instance the worker stores at
    ``self._background_tasks`` — a spawned investigation task's strong ref
    lives exactly as long as the worker, mirroring
    ``workflow.TaskWorkflow._background_tasks``.
    """

    scheduler: Any
    mcp: Any = None
    usage_gate: Any = None
    cost_store: Any = None
    background_tasks: set[asyncio.Task] = dataclasses.field(default_factory=set)


def _spawn_merge_verify_dry_run(
    handles: _DryRunInvestigationHandles | None,
    req: MergeRequest,
    reason: str,
    detail: str,
    *,
    event_store: EventStore | None = None,
) -> None:
    """Fire-and-forget: spawn an autonomous dry-run investigation for a
    MERGE_VERIFY_RED post-merge-verify block.

    Mirrors ``workflow._spawn_dry_run_unblock``'s fire-and-forget shape
    (try/except-wrapped ``asyncio.create_task``, strong-ref registration into
    a ``background_tasks`` set with a discard done-callback) so the
    merge-verify and agent-block investigation spawns behave identically.
    ``block_class=BlockClass.MERGE_VERIFY_RED`` is passed explicitly — the
    generic post-merge reason is deliberately not mapped by
    ``classify_block_reason`` (see ``unblock_types`` module docstring).

    None-safe: no-ops when *handles* is ``None`` or ``handles.scheduler`` is
    ``None`` — the solo-reverify and train module-level
    ``_run_post_merge_verify`` callers pass no handles.  Also no-ops when
    ``req.config.unblock_auto.enabled`` is falsy, when a not-done
    investigation task is already registered in ``handles.background_tasks``
    under this task's name (mirrors ``workflow._spawn_dry_run_unblock``'s
    enablement and in-flight-dedup guards), or when ``req.worktree`` is
    missing/nonexistent.

    The investigation always reads ``req.worktree`` — the task's OWN retained
    worktree — rather than the ephemeral merge worktree: by the time this
    fire-and-forget task actually runs, ``_run_post_merge_verify`` has already
    handed the merge worktree to ``cleanup_merge_worktree`` (task 2141
    step-17/18 — the no-op cleanup mock in the test suite hid this).
    ``req.worktree`` survives while the task stays blocked, and is what
    ``b3_gate`` re-checks at gate time. (The ephemeral merge worktree is no
    longer accepted as a parameter here — task 2141 amendment pass, review
    finding `dead_parameter`.)

    *event_store* is forwarded to ``run_dry_run_unblock`` so the
    investigation emits the same ``invocation_end``/``'blocked'`` telemetry
    event the agent-block path emits (observability parity).
    """
    if handles is None or handles.scheduler is None:
        return
    ua = getattr(req.config, 'unblock_auto', None)
    if not ua or not ua.enabled:
        return
    task_name = f'unblock-auto-{req.task_id}'
    # Skip if an investigation for this task is already running (e.g. rapid
    # re-blocks across successive merge-verify retries).  Duplicate proposals
    # are unhelpful and would multiply budget spend up to the investigation's
    # own timeout ceiling per re-block (mirrors
    # workflow._spawn_dry_run_unblock's dedup guard).
    if any(
        t.get_name() == task_name and not t.done()
        for t in handles.background_tasks
    ):
        logger.debug(
            'Task %s: merge-verify dry-run investigation already in '
            'progress, skipping duplicate spawn',
            req.task_id,
        )
        return
    if req.worktree is None or not req.worktree.exists():
        logger.debug(
            'Task %s: skipping merge-verify dry-run — task worktree missing',
            req.task_id,
        )
        return
    try:
        task = asyncio.create_task(
            run_dry_run_unblock(
                task_id=req.task_id,
                worktree=str(req.worktree),
                reason=reason,
                detail=detail,
                scheduler=handles.scheduler,
                mcp=handles.mcp,
                config=req.config,
                event_store=event_store,
                usage_gate=handles.usage_gate,
                cost_store=handles.cost_store,
                block_class=BlockClass.MERGE_VERIFY_RED,
            ),
            name=task_name,
        )
        handles.background_tasks.add(task)
        task.add_done_callback(handles.background_tasks.discard)
    except Exception as exc:
        logger.warning(
            'Task %s: failed to spawn merge-verify dry-run investigation: %s',
            req.task_id, exc,
        )


@dataclasses.dataclass(frozen=True)
class _MainHealthProbeHandles:
    """Opaque bundle carrying the worker's ``background_tasks`` set so the
    deferred main-health probe survives ``_run_post_merge_verify`` returning
    (task 2564, reify 5067 merge-slot-stall fix).

    A DISTINCT bundle from :class:`_DryRunInvestigationHandles` (not reused)
    — the main-health probe's ownership is independent of the dry-run
    investigation's, which keeps the shape minimal and future-friendly for
    the sibling host-affinity / warm-probe work.  ``background_tasks`` is the
    SAME ``set`` instance the worker stores at ``self._background_tasks`` —
    a spawned probe task's strong ref lives exactly as long as the worker
    and is drained by the worker's existing shutdown drain
    (``SpeculativeMergeWorker.stop``).

    Only the production ``SpeculativeMergeWorker._run_inflight_verify`` call
    site passes a live instance (task 2564 step-16).  The solo-reverify,
    train, and merge_gates module-level callers — and any bare test-local
    caller — pass ``None`` (the default), so ``_run_post_merge_verify`` keeps
    running the main-health probe SYNCHRONOUSLY exactly as it did before
    this task.

    ``auto_heal`` (task 2564 step-22) is an optional callback threaded
    through to :func:`_run_deferred_main_health_probe`: when a confirmed,
    still-fresh pre-existing break is found, the probe routes to it
    (``await auto_heal(outcome, req)``) INSTEAD of filing the
    escalation-only fallback.  ``None`` (the default) preserves the
    escalation-only behaviour for every bare/test caller.  Only the
    production call site passes
    ``SpeculativeMergeWorker._auto_heal_main_health_deferred`` (task 2564
    step-24).
    """

    background_tasks: set[asyncio.Task] = dataclasses.field(default_factory=set)
    auto_heal: (
        Callable[[MergeOutcome, MergeRequest], Awaitable[None]] | None
    ) = None


async def _run_deferred_main_health_probe(
    git_ops: GitOps,
    req: MergeRequest,
    verify: VerifyResult,
    *,
    escalation_queue: Any = None,
    event_store: EventStore | None = None,
    auto_heal: Callable[[MergeOutcome, MergeRequest], Awaitable[None]] | None = None,
) -> None:
    """Off-critical-path main-health classification (task 2564).

    Spawned by :func:`_spawn_main_health_probe` as a DETACHED task with no
    awaiter, so every externally-visible effect must come from what this
    coroutine does internally (emitting the ``main_health_red`` signal,
    filing a dedup'd escalation) — its return value is always discarded.

    Re-applies the SAME three cheap guards :func:`_spawn_main_health_probe`
    already applied before spawning (defense in depth for any future direct
    caller — mirrors :func:`_classify_main_health_red`'s guard block):

    - ``req.config.escalate_preexisting_main_break`` is ``False``
    - ``verify.timed_out`` is ``True``
    - ``verify.category`` is in :data:`PREEXISTING_BREAK_SKIP_CATEGORIES`

    Calls :func:`verify_failure_is_preexisting_on_main` directly (rather than
    :func:`_classify_main_health_red`) so it can capture ``probe_sha`` for
    the staleness check below.  A raising probe is caught and logged — this
    is a detached background task with no awaiter, so a swallowed exception
    here is the ONLY way it would ever be observed.

    Staleness re-check: the probe can run for minutes off the critical
    path, during which main may have advanced (e.g. a hotfix that already
    fixed the break).  ``probe_sha`` is the exact bare-main HEAD the probe
    actually tested against, so equality against a freshly-resolved
    ``git_ops.get_main_sha()`` is the precise freshness predicate.  Fails
    safe (skips the escalation, logging at ``info``) when the re-resolve
    raises, comes back empty, or disagrees with ``probe_sha`` — a
    genuinely-still-broken main will simply be re-probed and re-surfaced by
    the next failing merge.

    On a confirmed, still-fresh pre-existing break: builds the outcome via
    :func:`_build_main_health_outcome` and emits the ``main_health_red``
    merge-attempt signal (task 2564 step-22: BEFORE the branch below, so the
    signal fires whichever branch is taken).  Then, if *auto_heal* is
    supplied, routes the outcome to it (``await auto_heal(outcome, req)``)
    INSTEAD of filing an escalation directly — the callback owns filing
    (see :func:`SpeculativeMergeWorker._auto_heal_main_health_deferred`,
    which files the halt-owner escalation itself).  Otherwise (the default,
    every bare/test caller) falls back to the pre-existing escalation-only
    behaviour via :func:`_file_main_health_escalation`.
    """
    if not req.config.escalate_preexisting_main_break:
        return
    if verify.timed_out:
        return
    if (verify.category or '') in PREEXISTING_BREAK_SKIP_CATEGORIES:
        return
    try:
        is_preexisting, probe_sha = await verify_failure_is_preexisting_on_main(
            req.worktree, req.config, req.module_configs, req.task_files,
            verify, git_ops,
        )
    except Exception:
        logger.warning(
            'Task %s: deferred main-health probe failed', req.task_id,
            exc_info=True,
        )
        return
    if not is_preexisting:
        return

    try:
        current_main_sha = await git_ops.get_main_sha()
    except Exception:
        logger.info(
            'Task %s: deferred main-health probe: get_main_sha() re-resolve '
            'failed; skipping escalation (stale-check fail safe)',
            req.task_id,
            exc_info=True,
        )
        return
    if not current_main_sha or current_main_sha != probe_sha:
        logger.info(
            'Task %s: deferred main-health probe: main advanced since the '
            'probe ran (probe_sha=%s, current=%s); skipping escalation '
            '(stale-check fail safe)',
            req.task_id, probe_sha, current_main_sha,
        )
        return

    outcome = _build_main_health_outcome(verify, probe_sha)
    _emit_merge_attempt(event_store, req.task_id, OutcomeKind.main_health_red)
    if auto_heal is not None:
        await auto_heal(outcome, req)
    else:
        _file_main_health_escalation(escalation_queue, req, outcome)


def _file_main_health_escalation(
    escalation_queue: Any,
    req: MergeRequest,
    outcome: MergeOutcome,
    *,
    suggested_action: str = 'await_preexisting_main_hotfix',
) -> str | None:
    """File (or fold) a dedup'd ``preexisting_main_break`` escalation for a
    confirmed pre-existing main-red *outcome* (task 2564).

    Returns the id of the SURVIVING (parent) escalation, or ``None`` when
    *escalation_queue* is None.  A folded submission returns the PARENT's id
    (the child is absorbed via ``attach_dedupe_child`` and never becomes
    independently addressable) — callers that need to register lane-halt
    ownership (:func:`SpeculativeMergeWorker._auto_heal_main_health_deferred`)
    must reference the parent, not the folded child.  Existing callers that
    ignore the return value are unaffected (backward-compatible addition).

    *suggested_action* defaults to ``'await_preexisting_main_hotfix'`` —
    byte-identical to every call site predating this parameter — so an
    escalation-only caller (the bare/test fallback in
    :func:`_run_deferred_main_health_probe`) files the same operator
    instruction as before.  A caller performing the full auto-heal
    (:func:`SpeculativeMergeWorker._auto_heal_main_health_deferred`) passes
    ``'main_health_auto_heal_in_flight'`` instead, mirroring
    ``TaskWorkflow._auto_heal_main_health``'s halt-owner escalation.

    Folds via ``submit_or_dedupe`` using the SAME inf-window
    content-fingerprint :class:`~escalation.dedupe.DedupeConfig`
    ``workflow.py``'s ``_auto_heal_main_health`` uses (categories=
    ``('preexisting_main_break',)``), so a worker-filed and a (legacy)
    workflow-filed main-red escalation for the same
    ``(category, cause_hint, probe_sha)`` signature collapse into one
    parent — race-free against sibling probes/tasks. Falls back to a plain
    ``queue.submit`` when *outcome* carries no fingerprint (the fail-safe
    ``''`` path of :func:`_main_health_fingerprint`).
    """
    if escalation_queue is None:
        return None

    from escalation.dedupe import DedupeConfig, content_fingerprint_key, submit_or_dedupe
    from escalation.models import Escalation  # local import — escalation optional dep

    fp = outcome.dedupe_fingerprint or None
    esc = Escalation(
        id=escalation_queue.make_id(req.task_id),
        task_id=req.task_id,
        agent_role='orchestrator',
        severity='blocking',
        category='preexisting_main_break',
        summary=outcome.reason[:200],
        detail=outcome.reason,
        suggested_action=suggested_action,
        dedupe_fingerprint=fp,
    )
    if fp:
        submit_result = submit_or_dedupe(
            escalation_queue,
            esc,
            DedupeConfig(
                infra_dedupe_enabled=True,
                infra_dedupe_window_secs=float('inf'),
                infra_dedupe_categories=('preexisting_main_break',),
                key_fn=content_fingerprint_key,
            ),
        )
        return submit_result.get('parent_id') or submit_result.get('id') or esc.id
    escalation_queue.submit(esc)
    return esc.id


def _spawn_main_health_probe(
    handles: _MainHealthProbeHandles | None,
    git_ops: GitOps,
    req: MergeRequest,
    verify: VerifyResult,
    *,
    escalation_queue: Any = None,
    event_store: EventStore | None = None,
) -> None:
    """Fire-and-forget: spawn the deferred (off-critical-path) main-health
    classification for a post-merge-verify failure.

    None-safe: no-ops when *handles* is ``None`` (mirrors
    :func:`_spawn_merge_verify_dry_run`'s ``handles is None`` guard) — every
    call site invokes this unconditionally and relies on this internal check,
    so the solo-reverify, train, and merge_gates module-level
    ``_run_post_merge_verify`` callers (which pass no handles) are guaranteed
    no-ops.

    Applies the SAME three cheap guards :func:`_classify_main_health_red`
    applies before probing (task 2564 step-6 — kept byte-identical to that
    function's guard block so the deferred path never probes a case the
    synchronous path would have skipped):

    - ``req.config.escalate_preexisting_main_break`` is ``False``
    - ``verify.timed_out`` is ``True`` (non-deterministic; re-probing is wasteful)
    - ``verify.category`` is in :data:`PREEXISTING_BREAK_SKIP_CATEGORIES`
      (infra_timeout / flock_error — inherently flaky)

    Also dedupes in-flight: skips the spawn when a not-done task named
    ``f'main-health-probe-{req.task_id}'`` is already registered in
    ``handles.background_tasks`` (mirrors
    :func:`_spawn_merge_verify_dry_run`'s in-flight dedup guard) — a rapid
    sequence of failing merges for the same task must not pile up duplicate
    probes.

    Registers the spawned task into ``handles.background_tasks`` with a
    discard done-callback, named ``f'main-health-probe-{req.task_id}'`` so
    :class:`SpeculativeMergeWorker`'s existing ``self._background_tasks``
    shutdown drain cancels it deterministically on stop — no new drain code
    is needed.
    """
    if handles is None:
        return
    if not req.config.escalate_preexisting_main_break:
        return
    if verify.timed_out:
        return
    if (verify.category or '') in PREEXISTING_BREAK_SKIP_CATEGORIES:
        return
    task_name = f'main-health-probe-{req.task_id}'
    if any(
        t.get_name() == task_name and not t.done()
        for t in handles.background_tasks
    ):
        logger.debug(
            'Task %s: main-health probe already in progress, skipping '
            'duplicate spawn',
            req.task_id,
        )
        return
    try:
        task = asyncio.create_task(
            _run_deferred_main_health_probe(
                git_ops, req, verify,
                escalation_queue=escalation_queue, event_store=event_store,
                auto_heal=handles.auto_heal,
            ),
            name=task_name,
        )
        handles.background_tasks.add(task)
        task.add_done_callback(handles.background_tasks.discard)
    except Exception as exc:
        logger.warning(
            'Task %s: failed to spawn main-health probe: %s',
            req.task_id, exc,
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
    event_store: EventStore | None = None,
    merge_sha: str = '',
    on_result: Callable[[VerifyResult], None] | None = None,
    quarantine: set[str] | None = None,
    keep_worktrees: Collection[Path] | None = None,
    runner: VerifyRunner | None = None,
    escalation_queue: Any = None,
    dry_run_handles: _DryRunInvestigationHandles | None = None,
    main_health_probe_handles: _MainHealthProbeHandles | None = None,
    depth: int | None = None,
    speculative: bool | None = None,
) -> MergeOutcome | None:
    """Run post-merge verification for a single task.

    Shared by :class:`SpeculativeMergeWorker` and the retired serial worker's
    test-local reference (see :class:`_TrainMergeHost`).

    Returns ``None`` when verification passes; returns a ``MergeOutcome``
    (and cleans up *merge_wt*) when it fails via a controlled path (disk
    guard, verify-not-passed).  Does **not** contain a ``try/except`` — any
    exception from ``run_scoped_verification`` propagates to the caller.
    The test-local ``MergeWorker`` reference calls this bare (exceptions
    reach ``_process``); ``SpeculativeMergeWorker`` wraps the call in its
    existing ``try/except`` that maps a raised verify to a
    ``'Verification error: ...'`` outcome.

    Args:
        on_result: Optional callback invoked with the final :class:`~orchestrator.verify.VerifyResult`
            BEFORE the pass/fail branch — additive, default ``None`` keeps
            the test-local ``MergeWorker`` reference's call sites
            byte-identical.  Used by
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
        escalation_queue: Live escalation queue or ``None`` (default).  Used
            solely to file a born-at-L2 escalation (task 2307 β) when
            *verify* is the distinguished flock-worktree-contention outcome
            (``is_flock_contention_failure``); has no effect on any other
            failure path.  None-safe — omitting it keeps every existing
            call site byte-identical.
        dry_run_handles: Opaque bundle of scheduler/mcp/usage_gate/cost_store
            (task η, AFK coverage gap).  ``None`` (default) keeps the
            solo-reverify and train module-level callers byte-identical (no
            spawn).  Only the production ``SpeculativeMergeWorker`` call site
            passes a live bundle, at which point a MERGE_VERIFY_RED outcome
            (generic task-fault or unscoped-typecheck-FAILED) fires a
            fire-and-forget dry-run investigation via
            :func:`_spawn_merge_verify_dry_run`.
        main_health_probe_handles: Opaque bundle carrying the worker's
            ``background_tasks`` set (task 2564, reify 5067 merge-slot-stall
            fix).  ``None`` (default) keeps the solo-reverify, train, and
            merge_gates module-level callers — and any bare test-local
            caller — running the main-health probe SYNCHRONOUSLY exactly as
            before this task (byte-identical).  Only the production
            ``SpeculativeMergeWorker._run_inflight_verify`` call site passes
            a live bundle, at which point a main-health-eligible failure
            SKIPS the synchronous ``_classify_main_health_red`` probe,
            returns the provisional task-fault outcome immediately, and
            spawns the classification as a detached background task via
            :func:`_spawn_main_health_probe` — so the caller's
            ``verify_task`` (and the merge slot / host lease it holds) is
            freed within seconds of the verdict instead of blocking on a
            full local verify build bounded only by the cold timeout.
        depth: Verify-frontier stack height (task 2340, ε=1890) at dispatch
            time; threaded straight into ``pool.dispatch`` for the
            ``merge_verify`` event.  ``None`` (default) keeps every existing
            caller byte-identical.
        speculative: Mirrors ``item.speculative``; threaded straight into
            ``pool.dispatch`` alongside *depth*.  ``None`` (default) keeps
            every existing caller byte-identical.
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
    # callers: the test-local MergeWorker reference's _do_merge,
    # _reverify_rebased_tree, reverify_member_solo, _do_train_merge,
    # _run_cold_shadow_verify — recovery/train paths stay on the trust anchor
    # and out of slot accounting).
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
    #
    # Merge-verify lease (task 2315, BUG 1): record the SAME
    # ``.merge_verify.lock`` + holder-pgid lease the host verify-merge CLI
    # already records (cli.py:444-512) around this dispatch span — but ONLY
    # for a LOCAL in-process verify (runner is None) on the persistent warm
    # lane (``git.persistent_merge_worktree`` on AND *merge_wt* resolves to
    # ``persistent_merge_worktree_path``). Every other combination (remote
    # runner, ephemeral worktree, knob off) leaves the AsyncExitStack empty
    # — no lease recorded, byte-identical to before. Holding this lease lets
    # :meth:`GitOps.reset_persistent_merge_worktree` and
    # :meth:`GitOps._run_warm_lane_gc_reclaim` detect the in-flight verify
    # and avoid clobbering/reclaiming the worktree out from under it. Known
    # limitation: there is a small residual window between
    # ``reset_persistent_merge_worktree`` returning *merge_wt* (above the
    # caller of this function) and the lease being acquired here, during
    # which the worktree is unprotected.
    async with contextlib.AsyncExitStack() as stack:
        if (
            runner is None
            and req.config.git.persistent_merge_worktree
            and merge_wt.resolve() == git_ops.persistent_merge_worktree_path.resolve()
        ):
            await stack.enter_async_context(git_ops.merge_verify_lease())

        verify = await pool.dispatch(merge_sha, spec, depth=depth, speculative=speculative)

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
                verify = await pool.dispatch(
                    merge_sha, spec, attempt=1, depth=depth, speculative=speculative,
                )

    # Invoke the optional result-capture callback (PRD §10 invariant 6(b)):
    # called with the FINAL VerifyResult (after any ENOSPC retry) so the
    # warm per-test results are always the last-observed verify for this commit.
    # Default None keeps the test-local MergeWorker reference's call sites
    # byte-identical.
    if on_result is not None:
        on_result(verify)

    if not verify.passed:
        await git_ops.cleanup_merge_worktree(merge_wt)

        # Flock-worktree-contention sentinel (task 2307 β): a laptop-side verify
        # reported that .merge_verify.lock was already held (task 2306 α).  Checked
        # FIRST — before the unscoped-gate sentinel and the main-health probe — since
        # contention is not a pre-existing main-HEAD break; probing would be wasteful
        # and misleading.  Files a born-at-L2 escalation and keeps the merge blocked.
        if is_flock_contention_failure(verify):
            payload = verify.contention or {}
            _alarm_verify_worktree_contention(
                escalation_queue,
                host=payload.get('host', '<unknown>'),
                holder_pgid=payload.get('holder_pgid'),
                waiter_pgid=payload.get('waiter_pgid'),
            )
            return MergeOutcome(
                'blocked',
                reason=f'Post-merge verification blocked: {verify.summary} [category: {verify.category}]',
                failure_category=verify.category,
            )

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
            if verify.category != UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY:
                _spawn_merge_verify_dry_run(
                    dry_run_handles, req, reason, detail,
                    event_store=event_store,
                )
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
        #
        # DEFERRED mode (main_health_probe_handles is not None — only the
        # production SpeculativeMergeWorker._run_inflight_verify call site):
        # SKIP the synchronous probe entirely.  It can run for minutes
        # (bounded only by the cold verify timeout) and would otherwise hold
        # the caller's verify_task — hence the merge slot / host lease — for
        # its full duration (task 2564, the reify 5067 stall).  Fall through
        # to the provisional task-fault outcome below and spawn the
        # classification as a detached background task; a confirmed
        # pre-existing break is escalated separately, off the critical path,
        # by _run_deferred_main_health_probe once it completes.
        #
        # SYNCHRONOUS mode (handles is None — solo-reverify, train,
        # merge_gates, and test-local callers): unchanged, byte-identical
        # behaviour to before this task.
        if main_health_probe_handles is None:
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
        if not verify.timed_out:
            _spawn_merge_verify_dry_run(
                dry_run_handles, req, reason, detail,
                event_store=event_store,
            )
        # DEFERRED mode: spawn the off-critical-path main-health
        # classification now that the provisional outcome is fully built.
        # None-safe no-op in SYNCHRONOUS mode (handles is None).
        _spawn_main_health_probe(
            main_health_probe_handles, git_ops, req, verify,
            escalation_queue=escalation_queue, event_store=event_store,
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
    outcome: OutcomeKind,
    *,
    attempt: int | None = None,
    duration_ms: int | None = None,
    train_id: str | None = None,
    member_task_ids: list[str] | None = None,
    disposition: MergeFailureDisposition | None = None,
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

    *disposition* is the optional merge-skew attribution verdict (task 2381 α
    ``MergeFailureDisposition``, e.g. ``INTEGRATION_SKEW``/``BRANCH_BUG``).
    When supplied, its ``.value`` is stored under the ``'disposition'`` payload
    key so ``digest.merge_disposition_counts`` can separate integration-skew
    failures from branch bugs/indeterminate in runs.db stats (task 2384 γ,
    mechanism M2). When omitted (the default), no ``'disposition'`` key is
    added — existing callers' payloads stay byte-identical. Production
    call sites are threaded by task 2383 β; this parameter's mechanism is
    proven independently by a direct-emit unit test.
    """
    if event_store is not None:
        data: dict = {'outcome': outcome}
        if attempt is not None:
            data['attempt'] = attempt
        if train_id is not None:
            data['train_id'] = train_id
        if member_task_ids is not None:
            data['member_task_ids'] = member_task_ids
        if disposition is not None:
            data['disposition'] = disposition.value
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
                event_store, task_id, OutcomeKind.already_merged, duration_ms=_elapsed_ms(t0),
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
        event_store, task_id, OutcomeKind.unknown_branch, duration_ms=_elapsed_ms(t0),
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
    the test-local ``MergeWorker`` reference's CAS-retry path use an
    identical record shape.  If
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
    # worker (the test-local MergeWorker reference or SpeculativeMergeWorker)
    # finalizes gen_next.
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

    The sole PRODUCTION implementer is :class:`SpeculativeMergeWorker`, which
    inherits :class:`_WipHaltMixin` and defines every attribute / constant
    listed here.  The legacy serial worker once named ``MergeWorker`` is
    retired from production (MQ-refactor task nu, R7b); its readable-reference
    role now lives as a frozen test-local fixture
    (``orchestrator/tests/_serial_merge_worker.py``), which also satisfies this
    Protocol — kept structural (rather than inlined to ``SpeculativeMergeWorker``)
    so ``_do_train_merge`` stays reusable by that fixture without a
    production-side dependency on test code.

    This is the CANONICAL note on the retired serial worker's test-local
    reference; other docstrings/comments in this module that mention it
    point back here instead of repeating the file path, so there is a single
    place to update if the fixture is ever moved or renamed.

    The surface is intentionally narrow — only the state that the shared
    train-merge pipeline actually touches.  Adding new attributes here does
    NOT require touching ``_WipHaltMixin``; both implementers already
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


async def _already_merged_is_genuine(
    git_ops: GitOps,
    req: MergeRequest,
    branch_head: str,
    actual_main: str,
) -> bool:
    """Corroborate an ``is_ancestor(effective_tip, main)`` hit before skipping.

    ``effective_tip`` (``req.snapshot_tip or branch_head``) being an ancestor
    of main is NECESSARY but NOT SUFFICIENT to conclude the task's work is
    merged (task 5026).  A branch cut from main has a *base* commit that is
    trivially an ancestor of current main; if ``effective_tip`` resolves to
    that base — via a stale ``snapshot_tip`` captured while the branch was
    still zero-commit, or a ``None`` snapshot falling back to a worktree HEAD
    parked on a recycled lane at/under main — ``is_ancestor`` returns True
    even though the branch's real unique commit was never landed.  The
    short-circuit then returns a terminal ``already_merged`` that permanently
    ejects the task from the queue with its work still unmerged.

    This is the SAME false positive the reconciler already rejects for the
    degenerate zero-commit-branch shape (harness Guard-2, #1823): require
    POSITIVE evidence the work is on main before declaring already-merged.

    Returns ``True`` (genuinely merged — safe to skip) when ANY of:

      * the LIVE branch ref (or, when the ref is absent, the worktree HEAD) is
        itself an ancestor of main — the branch tip actually landed; OR
      * a merge-subject marker for this branch exists on main
        (``find_merge_marker``, ungated) — the work landed via a no-ff merge
        even though the branch tip diverged afterwards (the task-1917
        ``honors_snapshot_tip`` shape: ``snapshot_tip`` is the merged tip, the
        worktree/branch was rewritten to a divergent commit post-merge, and the
        on-main merge subject is what proves the work is safely on main).  The
        marker is keyed off the BRANCH, so it stays correct when
        ``task_id != branch``; OR
      * a commit on main cites this task id (``find_task_citation_commit``) —
        catches squash/cite merges whose subject is not the no-ff form.

    Returns ``False`` (false positive — caller must NOT skip) only when the
    branch still carries unique commits absent from main AND nothing on main
    cites the task.  Proceeding to merge in that case is always safe: if the
    commits truly were merged under a different sha the downstream rebase/merge
    no-ops, and a genuine merge cannot ghost-loop because a real merge leaves a
    citation that flips this guard back to ``True`` on the next pass.

    Note: on a project that has DISABLED citations (empty
    ``commit_citation_pattern``) ``find_task_citation_commit`` always returns
    None, so the rare post-merge-divergence shape degrades to a redundant
    (harmless) re-merge rather than a skip — the same limitation the reconciler
    carries.
    """
    branch_ref = await git_ops.resolve_queued_branch_ref(req.branch)
    branch_sha = (
        await git_ops.resolve_branch_sha(branch_ref)
        if branch_ref is not None else None
    )
    # Prefer the LIVE branch ref as the branch's real tip; fall back to the
    # worktree HEAD when the ref is absent (detached / lost-ref work).  This
    # deliberately does NOT consult the possibly-stale ``effective_tip``.
    candidate_tip = branch_sha or branch_head
    if await git_ops.is_ancestor(candidate_tip, actual_main):
        return True
    # candidate tip carries commits beyond main — only genuinely merged if
    # main carries positive evidence the work landed.  Prefer the branch-keyed
    # merge-subject marker (ungated: the branch ref legitimately still exists
    # here) so this stays correct when task_id != branch; fall back to the
    # task-id citation (parity with harness Guard-2, #1823) for squash/cite
    # merges whose subject is not the no-ff form.
    if branch_ref is not None and await git_ops.find_merge_marker(
        branch_ref, gate_on_existing_ref=False,
    ) is not None:
        return True
    citation = await git_ops.find_task_citation_commit(
        req.task_id,
        pattern_template=git_ops.config.commit_citation_pattern,
    )
    return citation is not None


async def classify_and_merge(
    worker: _TrainMergeHost,
    req: MergeRequest,
    base_sha: str,
    *,
    speculative: bool,
    started_monotonic: float | None,
) -> MergedOk | Decided:
    """Shared pre-merge guard + merge + drop-guard pipeline (MQ-refactor kappa).

    Covers the EQUIVALENCE-MATRIX CORE common to the retired serial worker's
    test-local reference (``_do_merge``; see :class:`_TrainMergeHost`),
    ``SpeculativeMergeWorker._merger_loop``, and
    ``SpeculativeMergeWorker._remerge``: branch-presence guard →
    already-merged detection → merge → conflict / non-conflict-failure →
    drop-guard.  Returns :class:`MergedOk` on success or :class:`Decided`
    wrapping the terminal :class:`MergeOutcome` otherwise.

    Divergences between the three call sites are preserved via parameters
    rather than per-caller branches inside this function:

    * ``speculative`` selects the merge base (``base_sha`` vs current main)
      and whether a ``speculative_merge`` event is emitted.
    * ``base_sha`` doubles as the merge base (when ``speculative``) and the
      drop-guard base (always).
    * Worker CAPABILITY — ``isinstance(worker, SpeculativeMergeWorker)`` —
      gates the drift-bookkeeping (``_note_conflict_detected``) and the rich
      failure diagnostic (``_build_merge_failure_diagnostic`` +
      ``_render_failure_diagnostic``).  The test-local ``MergeWorker``
      reference has neither, so routing ``_do_merge`` through this function
      reproduces its plain ``MergeOutcome('blocked', reason=details)``
      byte-identically.

    ``req.snapshot_tip`` (an orthogonal per-request field, not a parameter of
    this function) is honored uniformly for already-merged detection
    regardless of caller or ``speculative`` — this is an intentional
    CONVERGENCE, not a preserved divergence.  Pre-kappa, ``_do_merge``
    preferred ``snapshot_tip`` over worktree HEAD while ``_merger_loop``'s
    inline copy used worktree HEAD unconditionally.  Non-speculative
    ``_merger_loop`` requests CAN carry a ``snapshot_tip`` — e.g. the
    auto-chain-generation successor built by ``_maybe_auto_chain_generation``
    (:1669, ``snapshot_tip=current_head``) is enqueued generically and may be
    dispatched to either worker — so unifying the basis here is a deliberate
    fix, not an oversight.  See
    ``test_non_speculative_request_honors_snapshot_tip`` in
    test_merge_guard_pipeline.py.

    The verify-timeout loop-breaker and the ``_request_abandoned`` silent
    dequeue-drop are worker-lifecycle concerns and stay inline at each call
    site — they are not part of this pipeline.
    """
    git_ops = worker._git_ops
    event_store = worker._event_store

    # 1. Branch-presence guard.  Terminal outcomes (unknown_branch /
    # already_merged-via-marker) emit their own merge_attempt internally —
    # do not re-emit here.
    guard = await _classify_branch_presence(
        git_ops, event_store, req.task_id, req.branch, started_monotonic,
        worktree=req.worktree,
    )
    if guard is not None:
        return Decided(guard)

    # 2. Already-merged detection (ghost-loop fix).  effective_tip prefers
    # snapshot_tip when set, drift-proof vs a worktree HEAD that may have
    # been rebased to an orphaned lineage after snapshotting.  This basis is
    # shared across ALL callers/speculative values — see the docstring's
    # snapshot_tip paragraph for why that is an intentional convergence.
    rc, branch_head_raw, err = await _run(
        ['git', 'rev-parse', 'HEAD'], cwd=req.worktree,
    )
    if rc != 0:
        logger.warning(
            'Task %s: rev-parse HEAD failed: %s', req.task_id, err.strip(),
        )
        return Decided(MergeOutcome(
            'blocked', reason=f'rev-parse HEAD failed: {err.strip()}',
        ))
    branch_head = branch_head_raw.strip()
    actual_main = await git_ops.get_main_sha()
    effective_tip = req.snapshot_tip or branch_head
    if await git_ops.is_ancestor(effective_tip, actual_main):
        if await git_ops.has_uncommitted_work(req.worktree):
            # Guard: an agent may have started work since snapshotting —
            # don't skip the merge just because the tip already landed.
            logger.warning(
                'Task %s: branch is ancestor of main but worktree has '
                'uncommitted changes — not skipping merge', req.task_id,
            )
        elif not await _already_merged_is_genuine(
            git_ops, req, branch_head, actual_main,
        ):
            # False-positive guard (task 5026): effective_tip is an ancestor of
            # main, but the LIVE branch still carries unmerged unique commits
            # and no commit on main cites the task — effective_tip was a stale
            # base (zero-commit branch base / recycled-lane worktree HEAD /
            # stale snapshot_tip), NOT proof the work landed.  Do NOT emit a
            # terminal already_merged (which would eject the task from the
            # queue with its work unmerged); fall through and merge.
            logger.warning(
                'Task %s: effective_tip %s is an ancestor of main but the '
                'branch still has unmerged commits with no on-main citation '
                '— treating already-merged as a FALSE POSITIVE and merging',
                req.task_id, effective_tip[:12],
            )
        else:
            logger.info(
                'Task %s: branch already on main — skipping merge',
                req.task_id,
            )
            _emit_merge_attempt(
                event_store, req.task_id, OutcomeKind.already_merged,
                duration_ms=_elapsed_ms(started_monotonic),
            )
            return Decided(MergeOutcome('already_merged'))

    # 3. Merge (speculative or normal).  speculative=True is only ever passed
    # for a SpeculativeMergeWorker caller; the isinstance check is a static-
    # typing narrowing (the test-local MergeWorker reference has no
    # _emit_speculative), not a behavioural gate.
    if speculative and isinstance(worker, SpeculativeMergeWorker):
        # NOTE: _emit_speculative str-converts every data value (see its
        # `{k: str(v) ...}` coercion below), so depth lands here as a str
        # (e.g. "2") — unlike the merge_verify event's depth (dispatch, in
        # verify_runner.py), which is a native int. analyze_speculation_depth's
        # compute_per_depth() coerces both defensively, but a future consumer
        # that aggregates depth across both event types should not assume a
        # single type.
        worker._emit_speculative(
            EventType.speculative_merge, req.task_id, base_sha=base_sha,
            depth=worker._verify_frontier_depth(),
        )
    merge_result = await git_ops.merge_to_main(
        req.worktree, req.branch, base_sha=base_sha if speculative else None,
    )

    # Steps 4-7 run inside a try/except: an unexpected exception anywhere in
    # the post-merge pipeline (e.g. merge_commit is None despite a reported
    # success, or a drop-guard/diagnostic helper raises) must still clean up
    # the merge worktree before propagating.  Mirrors the pre-extraction
    # callers' merge_result_local-based safety net, which this function now
    # owns in full since it owns the merge_to_main call.
    try:
        # 4. Conflict → reject immediately (caller resolves outside queue).
        if merge_result.conflicts:
            logger.info('Task %s: merge conflicts detected', req.task_id)
            _emit_merge_attempt(
                event_store, req.task_id, OutcomeKind.conflict,
                duration_ms=_elapsed_ms(started_monotonic),
            )
            if isinstance(worker, SpeculativeMergeWorker):
                # ι=1894: record drift = main_position − base at merge-start.
                # Safe under kappa's unification (now reachable from both
                # _merger_loop's initial attempt and _remerge's re-attempt
                # for the same request_id): _note_conflict_detected pops
                # _drift_base defensively, and conflict is a terminal
                # outcome, so the two call sites can never both fire for one
                # request_id — see _note_conflict_detected's docstring.
                worker._note_conflict_detected(req.request_id)
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
            return Decided(
                MergeOutcome('conflict', conflict_details=merge_result.details),
                merge_result=merge_result,
            )

        # 5. Non-conflict failure.
        if not merge_result.success:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
            if isinstance(worker, SpeculativeMergeWorker):
                diag = await worker._build_merge_failure_diagnostic(
                    req,
                    base_sha=merge_result.pre_merge_sha or base_sha,
                    base_label='speculative' if speculative else 'main_head',
                    git_stderr=merge_result.details,
                )
                rendered = worker._render_failure_diagnostic(diag)
                return Decided(
                    MergeOutcome(
                        'blocked',
                        reason=f'{merge_result.details}\n{rendered}',
                        failure_diagnostic=diag,
                    ),
                    merge_result=merge_result,
                )
            return Decided(
                MergeOutcome('blocked', reason=merge_result.details),
                merge_result=merge_result,
            )

        # 6. Drop-guard: every file the task planned must survive the merge.
        assert merge_result.merge_commit is not None
        merge_commit = merge_result.merge_commit.strip()
        drop_result = await _check_plan_targets_in_tree(
            merge_commit, req.worktree, git_ops, base_sha, task_id=req.task_id,
        )
        if drop_result.dropped:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
            logger.warning(
                'Task %s: merge dropped plan targets: %s',
                req.task_id, drop_result.dropped,
            )
            _emit_merge_attempt(
                event_store, req.task_id, OutcomeKind.dropped_plan_targets,
                duration_ms=_elapsed_ms(started_monotonic),
            )
            reason = (
                f'{DROPPED_PLAN_TARGETS_REASON_PREFIX}: '
                f'{", ".join(drop_result.dropped)}. '
                f'Conflict resolution likely dropped planned work. '
                f'Review the merge commit and restore missing files.'
            )
            return Decided(MergeOutcome('blocked', reason=reason))

        # 7. Success.
        return MergedOk(
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            branch_tip=branch_head,
        )
    except Exception:
        if merge_result.merge_worktree:
            logger.debug(
                'Task %s: classify_and_merge: cleaning up merge worktree '
                'after post-merge error', req.task_id,
            )
            with contextlib.suppress(Exception):
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        raise


async def _journal_landed_then_advance(
    outbox: LandedOutbox | None,
    git_ops: Any,
    *,
    task_id: str,
    branch_tip_sha: str | None,
    advanced_sha: str,
    merge_wt: Path,
    **advance_kwargs: Any,
) -> AdvanceOutcome:
    """Record a LandedRow, THEN advance main — single-sourced write-ahead
    ordering (PRD WA-1) shared by BOTH CAS advance sites (single-branch
    ``_finalize_inflight`` and train ``_do_train_merge``).

    *advanced_sha* is the pre-advance SHA about to become main — the value
    knowable write-ahead, not ``advance_main``'s return value — and doubles
    as ``advance_main``'s first positional argument; *merge_wt* is the
    second. Remaining per-site ``**advance_kwargs`` pass through unchanged,
    keeping the advance call byte-identical to before this helper existed.
    A ``None`` *outbox* (no ``project_root``, e.g. bare-worker tests) no-ops
    the record — the advance still proceeds.

    CAUTION — the row is recorded UNCONDITIONALLY before the outcome of
    ``advance_main`` is known. If the returned :class:`AdvanceOutcome`'s
    ``result`` is anything other than ``'advanced'`` (e.g. ``'cas_failed'``,
    ``'not_descendant'``, ``'unmerged_state'``, ``'pop_conflict_no_advance'``),
    the task never actually landed on main, yet its row still persists in
    the outbox. A recorded row therefore means "write-ahead intent", NOT
    "confirmed landed" — this helper (and both its call sites) never
    ``.consume()`` on failure. Any consumer of :class:`LandedOutbox` /
    :class:`MergeProvenance` (e.g. a startup reconciler or a scheduler
    consult-before-dispatch gate) MUST treat row presence as provisional:
    verify the task actually landed (or cross-check ``advanced_sha`` against
    real main history) before relying on it, and ``.consume()`` a row only
    once the task is independently confirmed done.
    """
    if outbox is not None:
        outbox.record(LandedRow(
            task_id=task_id,
            branch_tip_sha=branch_tip_sha or '',
            advanced_sha=advanced_sha,
            landed_at=time.time(),
        ))
    return await git_ops.advance_main(advanced_sha, merge_wt, **advance_kwargs)


async def reconcile_landed_row(
    row: LandedRow,
    *,
    git_ops: Any,
    scheduler: Any,
    outbox: LandedOutbox,
    main_sha: str,
) -> str:
    """Reconcile a single :class:`LandedRow` against RC-1/RC-2/RC-3 (task 2155, W1 γ).

    A landed row is write-ahead INTENT, not confirmed landed (see
    ``_journal_landed_then_advance``'s docstring) — this is the shared
    per-row routine that closes the crash window between a merge advancing
    main and the task being marked done. Returns a disposition string used
    by :func:`reconcile_landed_outbox` to tally its report:

    * ``'pruned_not_landed'`` (RC-1) — ``row.advanced_sha`` is NOT an
      ancestor of ``main_sha``: the process crashed between the fsync'd
      record and the CAS advance, so the task never actually landed. Do NOT
      mark done; prune the row so the task re-dispatches through normal
      channels (no phantom done).
    * ``'already_done_pruned'`` (RC-3) — ``advanced_sha`` IS an ancestor of
      ``main_sha`` and the task's status is already a steward-owned resting
      state (:data:`~orchestrator.task_status.WORKFLOW_PRESERVE_STATUSES`:
      ``done``, ``cancelled``, ``deferred``, ``blocked``, or
      ``merge-deferred``): prune only, no done-write. A landed merge never
      overrides a status the workflow itself is forbidden from overwriting —
      e.g. a task the steward cancelled/blocked/deferred AFTER its merge
      landed but BEFORE the done-write must not be resurrected to ``'done'``
      by this reconciler, and a still-``'merge-deferred'`` train member's
      flip-to-done stays owned by the train-merge worker, not this scan.
      Checked before the RC-2 branch below so neither an already-done task
      nor any other steward-owned status is ever re-marked.
    * ``'marked_done'`` (RC-2) — ``advanced_sha`` IS an ancestor of
      ``main_sha`` and the task's status is NOT in
      ``WORKFLOW_PRESERVE_STATUSES``: the process crashed between the CAS
      advance and the done-write. Drive the task done via the existing
      ``merged`` done-write path, THEN consume the row — that ordering
      means a crash between the two re-drives cleanly on the next startup
      (the row survives to retry): if ``mark_done`` raises, the row is left
      unconsumed and the exception propagates to :func:`reconcile_landed_outbox`'s
      per-row ``try/except`` (tallied under ``'errors'``) rather than being
      silently pruned. A ``consume()`` failure AFTER ``mark_done`` already
      SUCCEEDED is different: it is swallowed (logged at WARNING) rather
      than propagated, and the disposition still reports ``'marked_done'``
      (reviewer_comprehensive amendment, task 2156) — see the inline
      try/except below for why.
    * ``'skipped'`` — ``scheduler.get_status`` returned ``None`` (a
      transient MCP failure): fail-safe, leave the row unconsumed for the
      next startup to retry rather than guessing done-or-not (no
      phantom-done, no premature prune).
    """
    if not await git_ops.is_ancestor(row.advanced_sha, main_sha):
        outbox.consume(row.task_id)
        return 'pruned_not_landed'

    status = await scheduler.get_status(row.task_id)
    if status is None:
        return 'skipped'
    if status in WORKFLOW_PRESERVE_STATUSES:
        outbox.consume(row.task_id)
        return 'already_done_pruned'
    await scheduler.mark_done(row.task_id, kind='merged', sha=row.advanced_sha)
    try:
        outbox.consume(row.task_id)
    except Exception:
        # mark_done ABOVE already succeeded, so the task IS genuinely done —
        # this guard exists so that fact can never be un-done by a failure
        # in the prune step. LandedOutbox.consume()/_save_raw already fails
        # open on OSError internally (logs at ERROR, counts save_failures,
        # never raises), so this is defense-in-depth for any OTHER
        # exception. Without this guard, the exception would propagate to
        # `reconcile_landed_task` and then to the scheduler's
        # `_consult_landed_outbox` (task 2156), whose per-candidate
        # try/except treats a raise as "not gated" (fails open) — that
        # would dispatch an agent for a task that was just marked done,
        # the exact ghost-dispatch this feature exists to prevent
        # (reviewer_comprehensive amendment #1). The stale row is harmless:
        # it self-heals on the next reconcile pass, which will find the
        # task's status already terminal and re-prune it via the RC-3
        # 'already_done_pruned' branch above.
        logger.warning(
            'reconcile_landed_row: mark_done succeeded for task_id=%s but '
            'outbox.consume() raised — task is done; leaving the row for '
            'the next pass to prune rather than risk a ghost re-dispatch',
            row.task_id, exc_info=True,
        )
    return 'marked_done'


async def reconcile_landed_task(
    task_id: str,
    *,
    git_ops: Any,
    scheduler: Any,
    outbox: LandedOutbox,
) -> bool:
    """Single-task landed-outbox consult for the scheduler dispatch gate (task 2156, W1 δ / SD-1).

    Guards the git ancestry round-trip behind a cheap in-memory outbox hit:
    landed rows exist only for the brief window between a merge advancing
    main and the task being marked done, so the overwhelmingly common case
    (no row) returns ``False`` with zero git I/O and zero scheduler calls —
    keeping the per-candidate cost of the scheduler's dispatch-gate consult
    to one dict lookup on the hot path.

    When a row IS present, resolves ``main_sha`` and delegates to the shared
    :func:`reconcile_landed_row` (task 2155, W1 γ) — the SAME RC-1/RC-2/RC-3
    routine the startup reconciler uses, so the scheduler consult and the
    startup reconciler converge through one done-write path and can never
    diverge.

    Returns ``True`` ⟺ ``row.advanced_sha`` is an ancestor of ``main`` ⟺ the
    task must NOT be dispatched — the disposition (drive to done via
    ``'marked_done'``, preserve via ``'already_done_pruned'``, or fail-safe
    wait via ``'skipped'``) has already happened inline via
    ``reconcile_landed_row``. Returns ``False`` when there is no row, or
    when the row's disposition is ``'pruned_not_landed'`` — the task never
    actually landed (crash before the CAS advance), so it stays normally
    dispatchable and its stale row has already been pruned.
    """
    row = outbox.lookup(task_id)
    if row is None:
        return False
    main_sha = await git_ops.get_main_sha()
    disposition = await reconcile_landed_row(
        row, git_ops=git_ops, scheduler=scheduler, outbox=outbox, main_sha=main_sha,
    )
    return disposition != 'pruned_not_landed'


async def reconcile_landed_outbox(
    outbox: LandedOutbox,
    git_ops: Any,
    scheduler: Any,
) -> dict[str, int]:
    """Scan *outbox* at startup and reconcile every unconsumed row (task 2155, W1 γ).

    Resolves ``main_sha`` once for the whole scan, then delegates each row to
    :func:`reconcile_landed_row`, tallying dispositions into the returned
    report (mirrors ``recover_pending_merges``'s count-report shape). A
    per-row try/except (fail-open, mirroring ``recover_pending_merges``)
    ensures one bad row never aborts the scan — its exception is logged and
    tallied under ``'errors'`` while the remaining rows still get reconciled.

    KNOWN LIMITATION (reviewer_comprehensive amendment #3, task 2155): the
    happy path (``advance_main`` → ``'advanced'`` → task marked done) never
    calls ``outbox.consume()`` — those ``Scheduler.mark_done(kind='merged',
    ...)`` call sites live in ``harness.py``'s ``build_train_callback_factory``
    (train members) and ``harness.py``'s ``found_on_main`` double-landing
    guard, plus several mirrored sites in ``workflow.py``, none of which
    currently hold a ``LandedOutbox`` reference. A long-running orchestrator
    therefore accumulates one row per successfully-landed task until the
    next startup prunes them here via RC-3 (``'already_done_pruned'``) —
    self-cleaning at boot, never a phantom-done risk. Wiring happy-path
    consume would require changing ``workflow.py``, which is outside this
    task's locked-module scope; left as a follow-up rather than expanded
    into this task (see escalation log).
    """
    report = {
        'pruned_not_landed': 0,
        'marked_done': 0,
        'already_done_pruned': 0,
        'skipped': 0,
        'errors': 0,
    }
    main_sha = await git_ops.get_main_sha()
    for row in outbox.all():
        try:
            disposition = await reconcile_landed_row(
                row, git_ops=git_ops, scheduler=scheduler, outbox=outbox, main_sha=main_sha,
            )
            report[disposition] += 1
        except Exception:
            logger.warning(
                'reconcile_landed_outbox: failed to reconcile row task_id=%s',
                row.task_id, exc_info=True,
            )
            report['errors'] += 1
    return report


async def _do_train_merge(
    worker: _TrainMergeHost,
    req: GroupMergeRequest,
) -> MergeOutcome:
    """Atomic train-merge pipeline shared by SpeculativeMergeWorker and the
    retired serial worker's test-local reference (see :class:`_TrainMergeHost`,
    the ``worker`` parameter's type above).

    BEHAVIOUR-ADDING (task 1596): trains now inherit the full shared post-merge
    core at PARITY with the test-local MergeWorker reference's ``_do_merge`` —
    specifically:
      • disk-guard pre-verify short-circuit (_run_post_merge_verify)
      • verify-timeout loop-breaker (worker._post_merge_verify_timeouts)
      • post-merge content-equivalence gate (_finalize_advanced_merge)
      • unscoped pyright gate (_finalize_advanced_merge)
      • push_main (outcome.push_status propagated from _finalize_advanced_merge)
      • wip-halt + escalation routing for non-'advanced' advance results
        (_map_advance_failure via worker.halt_for_wip)

    DEFENSIBLE DELTAS — behaviours intentionally absent from the train path
    despite being present in the test-local MergeWorker reference or
    SpeculativeMergeWorker:

    1. No ``reverify_on_rebase``: the 1595 disjoint-delta gate lives ONLY in
       SpeculativeMergeWorker._verify_and_advance's CAS loop.  The test-local
       MergeWorker reference (the readable serial reference) also does NOT
       pass it.  "Parity" means parity with that reference, not the
       spec-worker CAS loop.  Adding the gate here would duplicate
       speculative-worker logic that has no business in the train.

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
    # work — mirrors the test-local MergeWorker reference's ``_do_merge``.  A
    # stuck verify can otherwise burn merge-queue capacity for 30+ minutes
    # per attempt.
    prior_timeouts = worker._post_merge_verify_timeouts.get(req.task_id, 0)
    if prior_timeouts >= worker.MAX_POST_MERGE_VERIFY_TIMEOUTS:
        logger.warning(
            'Train %s: abandoning merge — %d consecutive post-merge '
            'verify timeouts (threshold=%d)',
            req.train_id, prior_timeouts,
            worker.MAX_POST_MERGE_VERIFY_TIMEOUTS,
        )
        _emit_merge_attempt(
            event_store, req.task_id, OutcomeKind.abandoned_verify_timeouts,
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
        _emit_merge_attempt(event_store, req.task_id, OutcomeKind.train_incomplete, duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
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
        _emit_merge_attempt(event_store, req.task_id, OutcomeKind.train_rebase_conflict, duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
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
        _emit_merge_attempt(event_store, req.task_id, OutcomeKind.conflict if merge_result.conflicts else OutcomeKind.merge_failed, duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
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
        _emit_merge_attempt(event_store, req.task_id, OutcomeKind.verify_failed, duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
        return verify_outcome

    # (f) CAS-advance main.
    # Write-ahead (PRD WA-1): record a LandedRow into the durable outbox
    # BEFORE advancing main — single-sourced via the shared helper (task β).
    # branch_tip_sha is best-effort (resolve_branch_sha returns str | None;
    # _do_train_merge has no MergedOk.branch_tip like the single-branch path).
    # Performance (amendment, task 2154): this costs one extra git subprocess
    # per train CAS-advance. Nothing upstream in this function already holds
    # a resolved tip SHA to reuse — rebase_onto_main returns bool only, and
    # the raw MergeResult from merge_to_main has no branch_tip field (that
    # field lives only on MergedOk, single-branch's classify_and_merge
    # result; see git_ops.py:368). The resolve is therefore an accepted,
    # intentional best-effort cost rather than a deferred TODO.
    _train_branch_tip = await git_ops.resolve_branch_sha(req.branch)
    adv_outcome = await _journal_landed_then_advance(
        getattr(worker, '_landed_outbox', None), git_ops,
        task_id=req.task_id,
        branch_tip_sha=_train_branch_tip,
        advanced_sha=merge_commit,
        merge_wt=merge_wt,
        branch=req.branch,
        max_attempts=req.config.max_advance_attempts,
        expected_main=main_sha,
    )
    adv = adv_outcome.result

    # Correctness (amendment, task 2154): the train call does not pass
    # reverify_on_rebase=True (unlike the single-branch site), so
    # advance_main's OWN internal CAS-retry loop can transparently rebase
    # merge_commit onto a moved main and land the REBASED sha directly,
    # returning 'advanced' with advanced_sha != the pre-advance merge_commit
    # journaled above. There is no caller-level rebased_pending_reverify
    # branch here to re-record from (that only exists in _finalize_inflight's
    # while-loop), so re-record now whenever the landed sha differs from what
    # was journaled write-ahead — the row's advanced_sha must always match
    # what actually landed on main. LandedOutbox is keyed by task_id ALONE
    # (last-write-wins — see landed_outbox.py's class docstring), so this
    # re-record OVERWRITES the write-ahead row in place rather than adding a
    # second (task_id, advanced_sha) entry; any consumer reading the row
    # after this point sees the landed sha.
    if (
        adv == 'advanced'
        and adv_outcome.advanced_sha is not None
        and adv_outcome.advanced_sha != merge_commit
    ):
        _train_outbox = getattr(worker, '_landed_outbox', None)
        if _train_outbox is not None:
            _train_outbox.record(LandedRow(
                task_id=req.task_id,
                branch_tip_sha=_train_branch_tip or '',
                advanced_sha=adv_outcome.advanced_sha,
                landed_at=time.time(),
            ))

    await git_ops.cleanup_merge_worktree(merge_wt)

    if adv != 'advanced':
        logger.info('Train %s: advance_main returned %r', req.train_id, adv)
        _emit_merge_attempt(event_store, req.task_id, OutcomeKind.advance_failed, duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
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
            advanced_sha=adv_outcome.advanced_sha,
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
    # applies ONLY to single-branch MergeRequest paths (the test-local
    # MergeWorker reference / SpeculativeMergeWorker).  chain_ctx=None is
    # passed explicitly here so the invariant is visible at the call site and
    # not left implicit.
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
        advanced_sha=adv_outcome.advanced_sha,
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
        _emit_merge_attempt(event_store, req.task_id, OutcomeKind.train_partial_flip, duration_ms=_elapsed_ms(t0), **_train_emit_kwargs)
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

    Provides the halt-owner methods that :class:`SpeculativeMergeWorker` (the
    sole production implementer) exposes as public API to ``workflow.py`` and
    ``harness.py``.  The retired serial worker's test-local reference (see
    :class:`_TrainMergeHost`) also subclasses this mixin.

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

        The mixin is also shared with the retired serial worker's test-local
        reference, which has no _resume_signal; the hasattr guard makes the
        call a no-op there.
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


class IllegalLifecycleTransition(RuntimeError):
    """Raised by :meth:`ItemLifecycle.transition` when a requested move is
    not a legal edge in the module-level ``_LEGAL_TRANSITIONS`` table, the
    caller's belief about the item's current state disagrees with the
    registry, or *request_id* is unregistered (merge-queue-reliability PRD
    scope-4 iota / task 2164, L-1).

    A dedicated exception type — raised explicitly, never via a bare
    ``assert`` (stripped under ``python -O``, the same hazard
    :meth:`PermitLedger.release` documents) — so the L-1 single-source guard
    survives optimized runs and callers/tests can catch precisely this
    failure mode.
    """


class ItemLifecycle:
    """Single source of truth for every in-flight item's lifecycle state,
    keyed by :attr:`MergeRequest.request_id` (merge-queue-reliability PRD
    scope-4 iota / task 2164, L-1).

    Replaces today's four redundant state encodings — container membership
    across the five queues, the free-form ``InflightEntry.phase: str`` field
    (now deleted by task lambda / task 2173 — phase is derived via
    :meth:`SpeculativeMergeWorker._entry_phase`), the :class:`InflightStatus`
    sentinel enum, and the four worker transient side-fields
    (``_inflight_req``/``_remerging_item``/``_finalizing_head``/
    ``_dispatching_item``) — with one :class:`ItemLifecycleState` per
    request_id, guarded on mutation by :meth:`transition` against the
    module-level ``_LEGAL_TRANSITIONS`` table.

    Task iota (this task) delivers only this substrate — ``register``/
    ``current``/``transition``. The sibling task kappa wires ``transition()``
    at every put/pop call site across the five queues and repoints
    ``snapshot()`` / the permit audit / liveness checks onto this registry.

    Concurrency model: storage is a plain ``dict`` with no locking. That is
    safe only because ``register``/``current``/``transition`` contain no
    ``await`` points, so each call runs atomically to completion under the
    single merge-queue asyncio event loop before another coroutine can
    interleave. This class is NOT thread-safe — the ``from_state``
    cross-check in :meth:`transition` defends against mis-wiring/logic
    races between coroutines sharing that one loop, not against concurrent
    access from a thread pool or a second event loop.
    """

    def __init__(self) -> None:
        self._states: dict[str, ItemLifecycleState] = {}

    def register(
        self,
        request_id: str,
        initial: ItemLifecycleState = ItemLifecycleState.QUEUED,
    ) -> None:
        """Seed *request_id* at *initial* (default QUEUED — every item enters
        the pipeline via the external queue).

        Raises ``ValueError`` if *request_id* is already registered — a
        duplicate ``register()`` call is a programming error. The
        request_id namespace is per-attempt: conflict auto-chain
        regeneration mints a NEW request_id for a fresh attempt
        (``_maybe_auto_chain_generation``) rather than re-registering an
        existing one, so a genuine duplicate here means two callers raced
        to register the same attempt.

        Deliberately ``ValueError``, not :class:`IllegalLifecycleTransition`:
        the latter models an illegal EDGE in ``_LEGAL_TRANSITIONS`` (or a
        from_state/registry disagreement) for an item already tracked by the
        registry, so kappa can catch it precisely at each wired transition
        call site. A duplicate ``register()`` is not an edge violation at
        all — there is no *from_state* and no row in the table to violate —
        it is a registry-identity precondition failure, the same category
        ``dict``/``set`` APIs signal with ``ValueError``/``KeyError`` rather
        than a domain-specific exception. Keeping the two distinct lets a
        caller that wants only L-1 edge violations ``except
        IllegalLifecycleTransition`` without also swallowing an unrelated
        double-registration bug.
        """
        if request_id in self._states:
            raise ValueError(
                f'ItemLifecycle.register: request_id {request_id!r} is already '
                f'registered (current state: {self._states[request_id]!r})'
            )
        self._states[request_id] = initial

    def current(self, request_id: str) -> ItemLifecycleState | None:
        """The current state for *request_id*, or ``None`` if unregistered."""
        return self._states.get(request_id)

    def transition(
        self,
        request_id: str,
        from_state: ItemLifecycleState,
        to_state: ItemLifecycleState,
    ) -> None:
        """Move *request_id* from *from_state* to *to_state*.

        Raises :class:`IllegalLifecycleTransition` — an explicit ``raise``,
        never a bare ``assert`` (stripped under ``python -O``, the same
        hazard :meth:`PermitLedger.release` documents) — and leaves the
        registry's stored state UNCHANGED when:

          * *request_id* is not registered;
          * the registry's actual current state disagrees with the caller's
            *from_state* (defense-in-depth cross-check — a mis-wiring or a
            race must never silently advance from the wrong base); or
          * ``(from_state, to_state)`` is not a legal edge in the
            module-level ``_LEGAL_TRANSITIONS`` table (defined below this
            class) — covers both skip-stage moves and moves out of the
            absorbing TERMINAL state.

        Updates the registry to *to_state* only once all three checks pass.
        """
        if request_id not in self._states:
            raise IllegalLifecycleTransition(
                f'ItemLifecycle.transition: request_id {request_id!r} is not '
                f'registered (cannot transition {from_state!r} -> {to_state!r})'
            )
        current = self._states[request_id]
        if current != from_state:
            raise IllegalLifecycleTransition(
                f'ItemLifecycle.transition: request_id {request_id!r} caller '
                f'believes from_state={from_state!r} but registry has '
                f'current={current!r} (single-source disagreement)'
            )
        if to_state not in _LEGAL_TRANSITIONS[from_state]:
            raise IllegalLifecycleTransition(
                f'ItemLifecycle.transition: illegal edge {from_state!r} -> {to_state!r} '
                f'for request_id {request_id!r}'
            )
        self._states[request_id] = to_state


_LEGAL_TRANSITIONS: dict[ItemLifecycleState, frozenset[ItemLifecycleState]] = {
    ItemLifecycleState.QUEUED: frozenset({
        ItemLifecycleState.LANE_BUFFERED,
        ItemLifecycleState.MERGING,
        ItemLifecycleState.TERMINAL,
    }),
    ItemLifecycleState.LANE_BUFFERED: frozenset({
        ItemLifecycleState.MERGING,
        ItemLifecycleState.TERMINAL,
    }),
    ItemLifecycleState.MERGING: frozenset({
        ItemLifecycleState.AWAITING_VERIFY,
        ItemLifecycleState.REDISPATCH_PARKED,
        ItemLifecycleState.DISPATCHING,
        ItemLifecycleState.TERMINAL,
    }),
    ItemLifecycleState.AWAITING_VERIFY: frozenset({
        ItemLifecycleState.DISPATCHING,
        ItemLifecycleState.REDISPATCH_PARKED,
        ItemLifecycleState.TERMINAL,
    }),
    ItemLifecycleState.REDISPATCH_PARKED: frozenset({
        ItemLifecycleState.DISPATCHING,
        ItemLifecycleState.TERMINAL,
    }),
    ItemLifecycleState.DISPATCHING: frozenset({
        ItemLifecycleState.VERIFYING,
        ItemLifecycleState.REDISPATCH_PARKED,
        ItemLifecycleState.QUEUED,
        ItemLifecycleState.MERGING,
        ItemLifecycleState.TERMINAL,
    }),
    ItemLifecycleState.VERIFYING: frozenset({
        ItemLifecycleState.GATE_REVERIFY,
        ItemLifecycleState.FINALIZING,
        ItemLifecycleState.MERGING,
        ItemLifecycleState.QUEUED,
        ItemLifecycleState.TERMINAL,
    }),
    ItemLifecycleState.GATE_REVERIFY: frozenset({
        ItemLifecycleState.FINALIZING,
        ItemLifecycleState.VERIFYING,
        ItemLifecycleState.MERGING,
        ItemLifecycleState.TERMINAL,
    }),
    ItemLifecycleState.FINALIZING: frozenset({
        ItemLifecycleState.TERMINAL,
        ItemLifecycleState.MERGING,
        ItemLifecycleState.GATE_REVERIFY,
    }),
    ItemLifecycleState.TERMINAL: frozenset(),
}
"""Legal-edge table for :meth:`ItemLifecycle.transition` (merge-queue-
reliability PRD scope-4 iota / task 2164, L-1). Defined after
:class:`ItemLifecycle` and looked up at call time (mirrors
:data:`_NON_TERMINAL_OUTCOMES` above, defined after :class:`OutcomeKind` for
the same reason), so the class body never references it eagerly.

Encodes the pipeline flow traced from snapshot construction and the
(now-deleted) phase-mutation call sites that used to set
``entry.phase = 'finalizing'``/``'gate_reverify'``, with an initial
``InflightEntry.phase = 'verifying'`` at dispatch (task lambda / task 2173
deleted the field itself; the states below are unchanged): queued ->
(lane_buffered ->) merging -> awaiting_verify -> (redispatch_parked <->)
dispatching -> verifying -> gate_reverify -> finalizing -> terminal, plus two
same-request_id in-place retry loops evidenced by ``_redispatch`` (the
redispatch bounce, DISPATCHING<->REDISPATCH_PARKED) and ``_remerging_item``
(the cascade remerge, VERIFYING/GATE_REVERIFY/FINALIZING -> MERGING ->
REDISPATCH_PARKED).

MQ-reliability kappa (task 2169) adds two further backward edges,
DISPATCHING -> QUEUED and VERIFYING -> QUEUED, for the three
``_queue.put_nowait(req)`` re-arm sites that put an in-flight request BACK
on the external input queue with its request_id and result Future intact
(operator-halt pre-dispatch requeue in ``_dispatch_item``, operator-halt
mid-verify requeue in ``_run_inflight_verify``, and the head-failure
cascade's downstream self-requeue in the verifier loop) — see
:meth:`SpeculativeMergeWorker._note_requeue`. Unlike the forward edges
above, these three sites are wired via the dynamic-current-state helper
rather than a hardcoded *from_state*, so a registered item legally lands
back at QUEUED regardless of which of the two states it was requeued from.

kappa also adds the DISPATCHING <-> MERGING pair for the dispatch-time
staleness/chain-invalidation remerge inside ``_dispatch_item`` (Mechanism 2):
an item already transitioned to DISPATCHING at the verifier's dispatch call
site discovers its base is stale, is recorded at MERGING for the duration of
the (pre-existing, ``_remerging_item``-tracked) ``_remerge()`` call, then
moves back to DISPATCHING once ``_remerge()`` returns — either to fall
through to a passthrough outcome (-> TERMINAL, wired by a later step) or to
proceed to a normal host-acquire + verify dispatch (-> VERIFYING via
``_inflight_append``). This is a DIFFERENT remerge window than the
VERIFYING/GATE_REVERIFY/FINALIZING -> MERGING head-failure cascade documented
above — that one never revisits DISPATCHING, it re-enters via
``_redispatch`` (-> REDISPATCH_PARKED) for a fresh dispatch attempt instead.

kappa also adds FINALIZING -> GATE_REVERIFY. ``_finalize_inflight`` sets the
registry to FINALIZING unconditionally at entry (previously mirroring its
own ``entry.phase = 'finalizing'`` set BEFORE the CAS loop's first
``advance_main`` attempt — that write was deleted by task lambda / task 2173;
the registry set at entry is now the sole source of truth), so a same-call
``rebased_pending_reverify`` result
moves the registry FINALIZING -> GATE_REVERIFY, not VERIFYING ->
GATE_REVERIFY (the latter edge, above, models a *fresh* dispatch that lands
directly on a gate re-verify with no intervening finalize attempt — not
exercised by current wiring, but kept for symmetry / future call sites).
Both edges coexist without conflict since ``ItemLifecycle.transition``
validates the caller's *from_state* against the registry's actual current
state independently of which edge licenses the move.

TERMINAL is absorbing (empty out-set): conflict auto-chain regeneration
mints a NEW request_id for the regenerated attempt
(``_maybe_auto_chain_generation``), so a request's lifecycle is always
forward-to-TERMINAL under its OWN id — a "restart" is a different registry
key entirely, never a backward edge on this one.
"""


def _request_id_of(obj: MergeRequest | SpeculativeItem | InflightEntry) -> str:
    """Return the stable request_id for *obj* (merge-queue-reliability PRD
    scope-4 kappa / task 2169).

    ``SpeculativeMergeWorker._live_items`` holds three different shapes
    across a request's pipeline lifetime — a :class:`MergeRequest`
    pre-merge, a :class:`SpeculativeItem` (``RealMergeItem | DecidedItem``)
    in-flight to the verifier, an :class:`InflightEntry` once dispatched —
    all keyed by the SAME request_id, so ``_register_item``/``_note_transition``
    callers can pass whichever shape they are currently holding without a
    per-call-site type check.
    """
    if isinstance(obj, InflightEntry):
        return obj.item.request.request_id
    if isinstance(obj, MergeRequest):
        return obj.request_id
    return obj.request.request_id


_LIFECYCLE_TRANSITION_REJECTED_SENTINEL_PREFIX = '__merge_lifecycle_transition_rejected__'


def _lifecycle_transition_rejected_sentinel(request_id: str) -> str:
    """Return the per-request dedup sentinel task_id for rejected-transition alarms."""
    return f'{_LIFECYCLE_TRANSITION_REJECTED_SENTINEL_PREFIX}{request_id}'


def _alarm_illegal_lifecycle_transition(
    escalation_queue: Any,
    request_id: str,
    from_state: ItemLifecycleState,
    to_state: ItemLifecycleState,
    exc: IllegalLifecycleTransition,
    *,
    event_store: Any = None,
) -> None:
    """Submit a dedup'd L1 escalation for a rejected :class:`ItemLifecycle`
    transition (merge-queue-reliability PRD scope-4 kappa / task 2169).

    Modeled verbatim on
    :func:`orchestrator.merge_request_ledger._alarm_merge_request_stuck`.
    Fires at most ONCE per open episode per request_id (dedup'd via
    ``has_open_l1``). Called by :meth:`SpeculativeMergeWorker._note_transition`
    when a wired put/pop call site's belief about an item's lifecycle state
    disagrees with the registry — a wiring-bug signal, not itself a merge
    failure. OBSERVATION + ESCALATION only: never mutates queue/inflight
    state or halts the pipeline (PRD design decision 4: invariants escalate
    loudly, degrade never).

    * ``level=1`` (L1 blocking).
    * ``category='merge_lifecycle_transition_rejected'``
    * ``task_id=_lifecycle_transition_rejected_sentinel(request_id)``

    None-safe: returns immediately when *escalation_queue* is None.
    Dedup: returns immediately when an open L1 already exists for the sentinel.
    """
    if escalation_queue is None:
        return

    sentinel = _lifecycle_transition_rejected_sentinel(request_id)

    # Dedup: don't re-alarm while an open L1 already exists for this request.
    if escalation_queue.has_open_l1(sentinel):
        return

    from escalation.models import Escalation  # local import — escalation optional dep

    summary = (
        f'ItemLifecycle rejected a transition for request_id {request_id!r}: '
        f'{from_state!r} -> {to_state!r}'
    )
    detail = (
        f'request_id: {request_id}\n'
        f'attempted transition: {from_state!r} -> {to_state!r}\n'
        f'ItemLifecycle.transition() error: {exc}\n'
        '\n'
        'A wired put/pop call site disagreed with the ItemLifecycle registry '
        '(unregistered request_id, an edge outside _LEGAL_TRANSITIONS, or a '
        'stale from_state belief). The registry state was left UNCHANGED and '
        'no pipeline state was mutated (PRD design decision 4: invariants '
        'escalate loudly, degrade never) — this is a wiring-bug signal, not '
        'itself a merge failure.'
    )

    esc = Escalation(
        id=escalation_queue.make_id(sentinel),
        task_id=sentinel,
        agent_role='orchestrator-merge-lifecycle-monitor',
        severity='blocking',
        level=1,
        category='merge_lifecycle_transition_rejected',
        summary=summary,
        detail=detail,
        suggested_action=(
            f'Inspect the ItemLifecycle wiring for request_id {request_id!r} '
            f'around the {from_state!r} -> {to_state!r} call site; the '
            'registry and the actual pipeline state have diverged.'
        ),
    )
    escalation_queue.submit(esc)

    if event_store is not None:
        from orchestrator.event_store import EventType

        event_store.emit(
            EventType.escalation_created,
            data={
                'request_id': request_id,
                'from_state': str(from_state),
                'to_state': str(to_state),
            },
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
    # After this many consecutive post-merge verify TIMEOUTS for the same
    # task, the merge queue stops trying and returns an 'abandoned' blocked
    # outcome.  Kept as a class attribute so tests can monkeypatch it.
    MAX_POST_MERGE_VERIFY_TIMEOUTS: int = 2
    # After a post-merge verify fails with an ENOSPC signature, prune stale
    # _merge-* worktrees and retry the verify at most this many times before
    # escalating as transient infra.  Kept as a class attribute so tests can
    # monkeypatch it.
    MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES: int = 1
    # Poll interval (seconds) used in the _verify_and_advance abort-loop that
    # checks whether a sole-waiter detach() cancelled req.result mid-verify.
    # Default ~10 s is negligible over a 10-40 min verify; kept as a class
    # attribute so tests can monkeypatch (e.g. worker.VERIFY_ABANDON_POLL_SECS
    # = 0.01) for fast, deterministic abort-path coverage.  Mirrors the MAX_*
    # monkeypatch convention above.
    VERIFY_ABANDON_POLL_SECS: float = 10.0
    # task 2420 (DEFECT 1, split from 2357; extends #1728): no-progress
    # budget for the in-flight verify abort-poll loop.  The #1728 alpha
    # owner-heartbeat keeps a LIVE worker's merge worktree ROOT mtime fresh
    # every _HEARTBEAT_POLL_S seconds even while its verify subprocess is
    # dead/hung, so the 3h root-mtime reaper
    # (INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS) never fires for that failure
    # mode.  Kept small relative to the reaper window so a genuinely dead
    # verify is caught promptly; kept as a class attribute so tests can
    # monkeypatch it tiny (e.g. worker.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS
    # = 0.2) for fast, deterministic abort-path coverage.  Mirrors the
    # VERIFY_ABANDON_POLL_SECS monkeypatch convention above.
    # INVARIANT (reviewer finding, task 2420 amend): this heuristic's
    # liveness signal is filesystem content mtime under merge_wt, so it
    # assumes a HEALTHY verify writes some file incrementally within every
    # budget window (test/lint/type-check output, build artifacts, etc.). A
    # purely CPU-bound stretch that writes nothing to disk for longer than
    # this budget (e.g. fully-buffered test output, a compiler that emits
    # artifacts only at the very end) would be misclassified as dead and
    # re-dispatched — and after MAX_INFLIGHT_DEAD_VERIFY_ABORTS repeats,
    # escalated to a terminal 'blocked' even though the verify was healthy.
    # Tune this budget to the longest expected no-write stretch of your
    # verify workloads, not just the dead-verify symptom's timescale.
    # BLIND SPOT (reviewer finding, task 2420 amend #2): the same risk
    # applies to any verify toolchain that writes primarily OUTSIDE
    # merge_wt — e.g. pytest/mypy/ruff caches under ~/.cache, $TMPDIR
    # scratch, or an out-of-tree build directory are all invisible to
    # newest_content_mtime(merge_wt). This assumption ("a healthy verify
    # writes some file incrementally under merge_wt within every budget
    # window") is NOT enforced or validated anywhere — it is a per-repo
    # tuning concern. Operators whose verify command writes its working
    # output outside the worktree should raise this budget accordingly (or,
    # if feasible, route that tool's cache/scratch dir back under merge_wt).
    INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS: float = 1800.0
    # Throttle for the newest_content_mtime() worktree-subtree walk that
    # drives the budget above — distinct from VERIFY_ABANDON_POLL_SECS so
    # the walk cost is bounded independently of the 10s abort-poll cadence.
    # Kept as a class attribute so tests can monkeypatch it tiny for fast,
    # deterministic coverage.  Mirrors the VERIFY_ABANDON_POLL_SECS
    # monkeypatch convention above.
    # COST (reviewer finding, task 2420 amend): each probe is a full
    # recursive os.walk + per-entry stat() over merge_wt — O(number of
    # filesystem entries below root); see newest_content_mtime's docstring.
    # The 60s default assumes the worktree's content tree (build/test
    # output, excluding .git) stays in the thousands-of-entries range, not
    # millions.  A repo whose verify writes a much larger tree should raise
    # this interval so the per-probe I/O stays negligible relative to the
    # verify itself — this is NOT currently self-tuning/adaptive.
    INFLIGHT_VERIFY_PROGRESS_PROBE_SECS: float = 60.0
    # task 2420 (DEFECT 1): busy-loop guard for the no-progress abort trigger.
    # After this many CONSECUTIVE no-progress aborts for the same task_id,
    # the request resolves terminally as 'blocked' instead of being
    # re-queued again — a deterministically-hanging verify would otherwise
    # re-queue forever, churning a slot without ever making progress.
    # Cleared on a successful verify for that task.  Kept as a class
    # attribute so tests can monkeypatch it small (e.g.
    # worker.MAX_INFLIGHT_DEAD_VERIFY_ABORTS = 2) for fast, deterministic
    # coverage.  Mirrors the VERIFY_ABANDON_POLL_SECS monkeypatch convention.
    MAX_INFLIGHT_DEAD_VERIFY_ABORTS: int = 3
    # MQ-invariants iota (task 1994): grace window (seconds, wall-clock mtime)
    # before an unregistered on-disk `_merge-*` worktree is flagged by
    # worktree_ledger_violations().  Tactical default sits strictly between
    # the shipped cold merge-verify budget
    # (merge_verify_cold_command_timeout_secs = 7200 s, defaults.yaml) — so a
    # legitimately slow cold-shadow/drift-check worktree never trips a false
    # alarm — and the reaper's INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS
    # (10800 s) — so a genuine leak is always caught well before the reaper
    # would otherwise silently absorb it.  Kept as a class attribute so tests
    # can monkeypatch (e.g. worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = 100.0)
    # for fast, deterministic coverage.  Mirrors the MAX_*/
    # VERIFY_ABANDON_POLL_SECS monkeypatch convention above.
    RESOURCE_AUDIT_WORKTREE_GRACE_SECS: float = 9000.0
    # MQ-invariants iota (task 1994): number of CONSECUTIVE
    # _check_resource_audit heartbeats a resource-conservation violation
    # (speculation_accounting_violations / worktree_ledger_violations) must
    # persist before the dedup'd L1 escalation (_alarm_resource_audit) fires.
    # Every violating call still logs a WARNING immediately — this only
    # gates the louder escalation, so a transient/racy single-poll blip
    # (e.g. a register/deregister race) never pages, while a genuine leak
    # trips well within a handful of heartbeat intervals. Kept as a class
    # attribute so tests can monkeypatch (e.g.
    # worker.RESOURCE_AUDIT_ESCALATION_STREAK = 1) for fast, deterministic
    # coverage. Mirrors the RESOURCE_AUDIT_WORKTREE_GRACE_SECS/MAX_*
    # monkeypatch convention above.
    RESOURCE_AUDIT_ESCALATION_STREAK: int = 3

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
        scheduler: Any = None,
        mcp: Any = None,
        usage_gate: Any = None,
        cost_store: Any = None,
    ):
        self._git_ops = git_ops
        self._queue = queue
        self._event_store = event_store
        # Task η (AFK coverage gap): harness-owned handles for the
        # merge-verify dry-run investigation spawn.  The worker holds none of
        # these itself in any other capacity — they exist solely to be
        # bundled into self._dry_run_handles below and threaded opaquely into
        # _run_post_merge_verify at the production _run_inflight_verify call
        # site (mirrors the train_callback_factory opaque-injection pattern
        # above: the worker never imports the scheduler).  All four default
        # to None so every existing bare git_ops+queue test constructor stays
        # byte-identical.
        self._scheduler = scheduler
        self._mcp = mcp
        self._usage_gate = usage_gate
        self._cost_store = cost_store
        self._background_tasks: set[asyncio.Task] = set()
        self._dry_run_handles = _DryRunInvestigationHandles(
            scheduler=scheduler, mcp=mcp, usage_gate=usage_gate,
            cost_store=cost_store, background_tasks=self._background_tasks,
        )
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
        # Durable landed-row journal (task 2153 / W1 α): survives until the
        # scheduler's consult-before-dispatch gate (δ, separate task) confirms
        # the task is done and calls consume(). None-safe (mirrors
        # _shadow_state_path above) so bare-worker/bare-harness tests without
        # project_root stay green. record() is not yet called at any
        # CAS-advance site — that wiring is task β. Bound onto MergeProvenance
        # so non-worker callers (e.g. the scheduler gate) can look up landed
        # rows without holding a reference to this worker.
        self._landed_outbox: LandedOutbox | None = (
            LandedOutbox(_root / 'data' / 'orchestrator' / 'landed_outbox.json')
        ) if _root is not None else None
        if self._landed_outbox is not None:
            MergeProvenance.bind(self._landed_outbox)
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
        # Per-task ENOSPC prune-and-retry counter (mirrors the test-local
        # MergeWorker reference's; see MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES).
        # Persists across submissions, reset on a successful CAS advance.
        self._post_merge_verify_enospc_retries: dict[str, int] = {}
        # γ2 per-branch generation auto-chain counter (mirrors the test-local
        # MergeWorker reference's).  Incremented on each consecutive
        # tip-advance equivalence failure; popped on a clean 'done' landing
        # or bound-exceeded escalation.
        self._generation_chain_counts: dict[str, int] = {}
        # task 2420 (DEFECT 1): per-task consecutive in-flight-verify
        # no-progress-abort counter (abort trigger 3).  Incremented each time
        # the LOCAL no-progress budget fires for a task; once it reaches
        # MAX_INFLIGHT_DEAD_VERIFY_ABORTS the request resolves terminally as
        # 'blocked' instead of being re-queued again, converting a
        # deterministically-hanging verify into a loud escalation rather than
        # an unbounded busy-loop.  Cleared on a successful verify for that
        # task so a later transient hang starts counting fresh.
        self._inflight_dead_verify_aborts: dict[str, int] = {}
        # Speculation-depth cap: one permit consumed by the Merger when it
        # prefetches a speculative item; released by the Verifier when it drains
        # that speculative item.  Symmetric accounting: acquire=prefetch,
        # release=drain so in-flight speculations are bounded at
        # _speculation_depth (K) at all times.  Plain Semaphore (not Bounded)
        # so stop() may over-release without raising.
        self._speculation_slot = asyncio.Semaphore(self._speculation_depth)
        # MQ-refactor zeta (task 2159): single owner of the semaphore above —
        # mediates every acquire/release through a SpecPermit token so
        # conservation (slot_available + len(live) == depth) holds by
        # construction. As of task eta (2160), the verifier-side releases
        # (_resolve_and_release/_finalize_inflight/cascade/stop-drain) all
        # route through self._speculation_ledger.release(item.permit) too,
        # so conservation is structural pipeline-wide and len(live) is the
        # authoritative outstanding-permit count (speculation_accounting_
        # violations reads it directly).
        self._speculation_ledger = PermitLedger(
            self._speculation_slot, self._speculation_depth,
        )
        # MQ-refactor theta (task 1993): owns the merger-side speculation
        # state machine (spec_base/prefetched/pending_spec_base/
        # pending_predecessor + the permit lifecycle) with EXPLICIT
        # ownership-transfer semantics. Holds a REFERENCE to the shared
        # _speculation_ledger above (never its own). The verifier-side
        # releases (_resolve_and_release/_finalize_inflight/cascade) now
        # release THROUGH that shared ledger via item.permit (task eta,
        # 2160) rather than raw-releasing the semaphore directly. See
        # merge_speculation_controller.py's module docstring for the full
        # lifecycle contract.
        self._speculation_controller = SpeculationController(
            self._speculation_ledger, self._speculation_depth,
        )
        # Merger-ahead cap (Mechanism 1, task 1646): limits non-speculative
        # build-ahead to speculation_depth items in the verifier queue.
        # Plain Semaphore (not BoundedSemaphore) so stop() may over-release
        # without raising.  Released ON-DRAIN (right after verifier_queue.get()
        # for a counted item) so the slot is free while verify runs.
        self._merge_ahead_cap = asyncio.Semaphore(self._speculation_depth)
        # MQ-refactor theta (task 2161): single owner of the cap semaphore
        # above — mediates every acquire/release through a CapPermit token
        # (PermitLedger generalized over token_factory, mirroring zeta's
        # _speculation_ledger) so conservation (slot_available + len(live) ==
        # depth) holds by construction. len(live) is the authoritative
        # in-flight-cap count (speculation_accounting_violations identity
        # (b) and _inflight_cap_count read it directly).
        self._merge_ahead_ledger: PermitLedger[CapPermit] = PermitLedger(
            self._merge_ahead_cap, self._speculation_depth,
            token_factory=CapPermit,
        )
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
        # Cross-workflow auto-heal attempt counter (mirrors the test-local
        # MergeWorker reference's; shared via self.merge_worker on
        # TaskWorkflow instances).
        self.auto_heal_registry: MainHealthAutoHealRegistry = MainHealthAutoHealRegistry()
        # Internal tasks created by run()
        self._merger_task: asyncio.Task | None = None
        self._verifier_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._reprobe_task: asyncio.Task | None = None
        # MQ-reliability kappa (task 2169): single source of truth for every
        # in-flight item's lifecycle state, keyed by request_id (see
        # ItemLifecycle's docstring above). Wired at every put/pop chokepoint
        # by _register_item/_note_transition/_retire_item below. Lockstep
        # `_live_items` holds the actual rich object (MergeRequest |
        # SpeculativeItem | InflightEntry) for each non-terminal request_id —
        # the registry alone only stores request_id -> ItemLifecycleState, but
        # snapshot()/occupancy/verify_in_progress need the rich object, so the
        # two are mutated together at the same chokepoints and can never
        # disagree on membership.
        self._lifecycle = ItemLifecycle()
        self._live_items: dict[str, MergeRequest | SpeculativeItem | InflightEntry] = {}
        # In-flight request being processed by the merger loop. Set after
        # dequeue, cleared after the SpeculativeItem is pushed to the verifier
        # queue. Used by stop() to resolve Futures for requests that were
        # mid-processing when shutdown was initiated.
        self._inflight_req: MergeRequest | None = None
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
        # Dispatch-gap census (task 2096): set to the item just popped from
        # _redispatch/_verifier_queue for the duration of `await
        # self._dispatch_item(item)` in _verifier_loop's DISPATCH-FILL — the
        # item is off the queue but not yet appended to _inflight during
        # that await (host-acquisition git calls, in-dispatch speculative-
        # remerge). Closes the I4 speculation-permit census gap the same way
        # _finalizing_head closes the finalize-head gap above. Set
        # immediately before the await in a nested try; cleared in its
        # finally so an exception inside _dispatch_item can never leave it
        # stale. Census-only: unlike _finalizing_head / _remerging_item, this
        # field is NOT read by snapshot()'s 'entries' section (task 2068).
        self._dispatching_item: SpeculativeItem | None = None
        # Persistent warm merge-verify worktree: counts verifying attempts so
        # _safety_valve_due can fire the periodic cold-verify (PRD §10 invariant 6).
        # Incremented on every LOCAL-lease verify attempt — task-1724 made verify
        # unconditional, and the skip_verify field it used to gate against was
        # retired by task ο (RealMergeItem carries no such escape hatch). Never
        # reset so the counter covers the full worker lifetime (cross-submission).
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
        #   (b) The retired serial worker's test-local reference (see
        #       _TrainMergeHost) holds ≤1 worktree whose build activity
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
        # MQ-invariants iota (task 1994): count of CONSECUTIVE
        # _check_resource_audit heartbeats that found a resource-
        # conservation violation. Reset to 0 the moment a heartbeat is
        # clean; once it reaches RESOURCE_AUDIT_ESCALATION_STREAK the
        # dedup'd L1 escalation (_alarm_resource_audit) fires. See
        # _check_resource_audit for the full contract.
        self._resource_audit_violation_streak: int = 0

    # ── MQ-reliability kappa (task 2169): ItemLifecycle chokepoint helpers ──
    # register/transition/retire an item's lifecycle state + the lockstep
    # `_live_items` object index in ONE place each, so every put/pop wiring
    # site below calls one of these three rather than touching
    # self._lifecycle/self._live_items directly.

    def _register_item(
        self,
        obj: MergeRequest | SpeculativeItem | InflightEntry,
        initial: ItemLifecycleState = ItemLifecycleState.QUEUED,
    ) -> str:
        """Register *obj* in the lifecycle registry at *initial* and index it
        in ``_live_items`` by request_id.

        Returns the request_id for convenience at call sites that need it
        right after registering. Raises ``ValueError`` if *obj*'s request_id
        is already registered — a genuine duplicate-register programming
        error (see :meth:`ItemLifecycle.register`'s docstring), NOT wrapped
        in the best-effort-loud handling :meth:`_note_transition` uses for
        edge violations: there is no legal way to recover from registering
        the same in-flight attempt twice.
        """
        rid = _request_id_of(obj)
        self._lifecycle.register(rid, initial=initial)
        self._live_items[rid] = obj
        return rid

    def _note_transition(
        self,
        request_id: str,
        from_state: ItemLifecycleState,
        to_state: ItemLifecycleState,
        *,
        live_obj: MergeRequest | SpeculativeItem | InflightEntry | None = None,
    ) -> None:
        """Best-effort-loud wrapper around :meth:`ItemLifecycle.transition`.

        NEVER raises — merge_queue.py is the #1-reliability file, and a
        mis-wired call site must never wedge the merger/verifier hot path
        (PRD design decision 4: invariants escalate loudly, degrade never).
        ``IllegalLifecycleTransition`` (unregistered request_id, an edge
        outside ``_LEGAL_TRANSITIONS``, or a stale *from_state* belief) is
        caught, logged at WARNING, and reported via a dedup'd L1 escalation
        (:func:`_alarm_illegal_lifecycle_transition`) instead of propagating.
        The registry state is left UNCHANGED on rejection (same contract as
        the wrapped ``transition()`` call).

        *live_obj*, when given, replaces ``_live_items[request_id]`` — used
        by call sites where the transition also changes the item's shape
        (e.g. a ``MergeRequest`` becoming a ``SpeculativeItem`` at merge
        time). Only applied on a SUCCESSFUL transition; omitted (``None``,
        the default) when the caller's existing ``_live_items`` entry is
        still the correct object (e.g. an in-place ``InflightEntry`` field
        mutation, such as ``verify_result``) or the object isn't tracked yet.
        """
        try:
            self._lifecycle.transition(request_id, from_state, to_state)
        except IllegalLifecycleTransition as exc:
            logger.warning(
                'ItemLifecycle: rejected transition for request_id=%s (%s -> %s): %s',
                request_id, from_state, to_state, exc,
            )
            _alarm_illegal_lifecycle_transition(
                self._escalation_queue, request_id, from_state, to_state, exc,
                event_store=self._event_store,
            )
            return
        if live_obj is not None:
            self._live_items[request_id] = live_obj

    def _retire_item(self, request_id: str) -> None:
        """Best-effort transition *request_id* to TERMINAL then drop it from
        ``_live_items``.

        Safe to call on an already-TERMINAL or unregistered request_id —
        both are treated as nothing-left-to-retire (no :meth:`_note_transition`
        call) and the ``_live_items`` pop always runs so a stale reference is
        never leaked either way.
        """
        current = self._lifecycle.current(request_id)
        if current is not None and current != ItemLifecycleState.TERMINAL:
            self._note_transition(request_id, current, ItemLifecycleState.TERMINAL)
        self._live_items.pop(request_id, None)

    def _note_requeue(
        self,
        request_id: str,
        live_obj: MergeRequest | SpeculativeItem | InflightEntry | None = None,
    ) -> None:
        """Best-effort transition *request_id* back to QUEUED at a requeue site
        (operator-halt abort-poll, pre-dispatch operator-halt, cascade
        downstream self-requeue — MQ-reliability kappa / task 2169).

        Reads the CURRENT registry state dynamically rather than a
        hardcoded *from_state*: these three sites sit downstream of the
        VERIFYING/DISPATCHING wiring landed by kappa's later steps, so
        during the additive rollout window the true current state may
        still be an earlier stage (e.g. AWAITING_VERIFY). No-ops silently
        (no :meth:`_note_transition` call, hence no WARNING/escalation)
        when *request_id* is unregistered — unlike the drain/pop
        chokepoints, an unregistered rid here is NOT itself a wiring-bug
        signal: several existing narrow unit tests exercise these requeue
        branches by constructing an item directly (bypassing the normal
        drain chokepoint), e.g.
        test_merge_queue_request_liveness.py::TestOperatorHaltRequeueNoFalseAlarm,
        which asserts zero escalations for exactly this call pattern.
        """
        current = self._lifecycle.current(request_id)
        if current is None:
            return
        self._note_transition(request_id, current, ItemLifecycleState.QUEUED, live_obj=live_obj)

    def _entry_phase(self, entry: InflightEntry) -> str:
        """Return *entry*'s current phase, derived from the ItemLifecycle
        registry rather than a stored field (merge-queue-reliability PRD
        scope-4 lambda / task 2173 step-4).

        Falls back to ``'verifying'`` when the entry's request is not
        registered in ``self._lifecycle``. ``InflightEntry`` no longer
        carries a ``phase`` field at all (deleted by task lambda / task 2173
        step-6) — this constant fallback simply preserves the pre-deletion
        ``infl.phase or 'verifying'`` default: every production
        ``_inflight`` entry is registered via ``_inflight_append``, so the
        fallback is defensive-only and never the hot path — it only matters
        for narrow unit tests that construct an ``InflightEntry`` without
        registering it.

        The fallback deliberately still returns a *qualifying* phase
        (rather than a fail-safe sentinel outside
        ``{'verifying', 'gate_reverify', 'finalizing'}``) so it keeps
        matching the pre-deletion default for those narrow unit tests
        exactly (task 2173 amendment review, esc-2173-reviewer_comprehensive:
        a non-qualifying sentinel would flip several existing
        registration-skipping ``_finalizing_head`` tests from included to
        excluded). Since this path is defensive-only in production, it logs
        at DEBUG on every hit so a future wiring regression that reaches it
        is at least observable rather than silently masquerading as a
        genuinely-verifying/frozen head.
        """
        state = self._lifecycle.current(entry.item.request.request_id)
        if state is not None:
            return state.value
        logger.debug(
            'Entry for request %s has no ItemLifecycle registry entry; '
            'falling back to phase=%r. Defensive-only in production — every '
            '_inflight entry is registered via _inflight_append, so this '
            'normally only fires from a narrow unit test that constructs an '
            'InflightEntry without registering it. If seen outside tests, '
            'it signals a lifecycle-registration wiring regression.',
            entry.item.request.request_id,
            'verifying',
        )
        return 'verifying'

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

    def _assert_single_writer(self, expected_task: asyncio.Task | None, structure: str) -> None:  # type: ignore[type-arg]
        """Raise if *structure* is being mutated from a coroutine other than *expected_task*.

        Debug-only single-writer discipline check (task 1999 / MQ-invariants
        ξ, I7) for ``_lane_buffers`` (owner: ``self._merger_task``) and
        ``_inflight`` (owner: ``self._verifier_task``) — see each attribute's
        own "Accessed only from the ... coroutine" comment.  A no-op in all
        of these cases (checked in order, cheapest first):

          * :data:`_DEBUG_ASSERTS` is off (production default) — single
            module-bool branch, effectively zero overhead.
          * *expected_task* is ``None`` — no loop has started yet (e.g. the
            hundreds of direct-call unit tests that construct a bare worker
            and call a mutation method without ever running ``_spawn_loop``).
          * ``not self._running`` — :meth:`stop` sets this False BEFORE its
            own shutdown drains of both structures from the stop coroutine
            (a legitimate non-owner mutation), so this gate exempts them.

        Otherwise raises :class:`AssertionError` when
        ``asyncio.current_task()`` is not *expected_task*.
        """
        if not _DEBUG_ASSERTS or expected_task is None or not self._running:
            return
        cur = asyncio.current_task()
        if cur is not expected_task:
            raise AssertionError(
                f'{structure} mutated from non-owner coroutine {cur!r}; '
                f'single-writer discipline requires owner {expected_task!r}'
            )

    def _buffer_owned_request(self, item: MergeRequest) -> None:
        """Append *item* to its lane buffer and record it in the durable journal.

        Registers a once-only done-callback on ``item.result`` so the journal
        entry is removed on ANY terminal outcome (done/error/abandoned/
        superseded).  The ``_journaled_request_ids`` set prevents duplicate
        callbacks when the same request_id is re-dispatched (e.g. CAS retry).

        Fail-open: store errors are logged and never propagate so a broken
        journal never stalls the merge pipeline.

        MQ-reliability kappa (task 2169): this is the worker's first sighting
        of *item* for a FRESH request — its three callers
        (``_drain_queue_into_lanes``, ``_acquire_next_request``'s harvest
        branch, and the merger loop's look-ahead harvest) all pull an item
        straight off the external ``_queue``/``_pending_get``. That item is
        USUALLY a brand-new request_id, but it may also be a request
        previously put back on ``_queue`` by a requeue site
        (:meth:`_note_requeue` — operator-halt / cascade self-requeue),
        which leaves it registered at QUEUED rather than removing it
        (:class:`ItemLifecycle` has no deregister op). Registers only when
        the request_id is not already tracked; an already-tracked rid skips
        straight to the QUEUED -> LANE_BUFFERED transition below instead of
        re-``register()``-ing, which would raise ``ValueError`` on the live
        duplicate (:meth:`ItemLifecycle.register`'s precondition). Either
        way, the item ends up seeded/confirmed at QUEUED then immediately
        transitioned to LANE_BUFFERED, matching this method's own
        postcondition (the item is now sitting in a lane buffer).
        """
        lane = _normalize_lane(item.lane)
        self._assert_single_writer(self._merger_task, '_lane_buffers')
        rid = item.request_id
        if self._lifecycle.current(rid) is None:
            self._register_item(item, initial=ItemLifecycleState.QUEUED)
        self._note_transition(
            rid, ItemLifecycleState.QUEUED, ItemLifecycleState.LANE_BUFFERED, live_obj=item,
        )
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

        MQ-reliability kappa (task 2169): every item in ``_lane_buffers`` got
        there via ``_buffer_owned_request`` (fresh drain) or the coalesce
        QUEUE SURGERY (``_maybe_coalesce_waiting_singles``'s GroupMergeRequest,
        registered directly at LANE_BUFFERED) — both leave the registry at
        LANE_BUFFERED, so a pop here is always a LANE_BUFFERED->MERGING
        transition. This is the ONLY place that pops ``_lane_buffers``
        (``_acquire_next_request``'s regular pick and the merger loop's
        speculative look-ahead both call this method), so wiring the
        transition here — rather than at each caller — covers both paths in
        one chokepoint; by the time either caller's ``_inflight_req = req``
        assignment runs, the registry already reads MERGING for *req*.
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
                    self._assert_single_writer(self._merger_task, '_lane_buffers')
                    del buf[i]
                    self._note_transition(
                        x.request_id, ItemLifecycleState.LANE_BUFFERED, ItemLifecycleState.MERGING,
                        live_obj=x,
                    )
                    return x
            # Defensive fallback (unreachable: the aging-minimal item is always
            # minimal by the above criterion, so the loop always returns).
            self._assert_single_writer(self._merger_task, '_lane_buffers')
            popped = buf.popleft()
            self._note_transition(
                popped.request_id, ItemLifecycleState.LANE_BUFFERED, ItemLifecycleState.MERGING,
                live_obj=popped,
            )
            return popped
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
            and self._entry_phase(self._finalizing_head) in {'verifying', 'gate_reverify', 'finalizing'}
        ):
            entries.append(self._finalizing_head)
        for e in self._inflight:
            if e.verify_task is not None:
                entries.append(e)
        return entries

    def _verify_frontier_depth(self) -> int:
        """Return the verify-frontier stack height (task 2340, reuses ε=1890).

        depth 0 = a head verify against real main; depth d = d speculated
        items already frozen/verifying ahead of the item joining the
        frontier.  Pure/synchronous (no await) — mirrors frozen_prefix().
        """
        return len(self._frozen_inflight_entries())

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
            if isinstance(entry.item, RealMergeItem) and entry.item.merge_result.merge_commit:
                return entry.item.merge_result.merge_commit.strip()
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

    def _frozen_base_chain(self, main_sha: str) -> Iterator[tuple[str, str, str]]:
        """Walk the frozen prefix's expected-base chain, yielding (rid, expected, actual).

        Shared chain-walk consumed by BOTH :meth:`check_frozen_prefix_invariant`
        and :meth:`_verify_base_frozen_tip_violations` — the two checks compare
        the exact same per-entry chained expected base (head expected =
        main_sha; each subsequent entry's expected = predecessor's
        merge_commit) and differ only in how they word a mismatch as a
        violation string. Factored here so that chain-walk is defined exactly
        once: if base-chain semantics ever change (e.g. how passthrough /
        conflict entries advance the chain), both surfaces pick up the fix
        together instead of silently drifting out of sync.

        Entries with no merge_result/merge_commit (passthrough / conflict)
        are skipped and do NOT advance the chain — mirrors the dispatch
        guard's own ``item.merge_result is None or not item.base_sha``
        exclusion.

        Pure/synchronous (no await); never raises on well-formed
        InflightEntry data.
        """
        expected_base = main_sha.strip()
        for entry in self._frozen_inflight_entries():
            rid = entry.item.request.request_id
            if not isinstance(entry.item, RealMergeItem):
                # Not a real merge — nothing to chain; do not advance.
                continue
            mr = entry.item.merge_result
            if not mr.merge_commit:
                # No merge_commit — nothing to chain; do not advance.
                continue
            actual_base = entry.item.base_sha.strip() if entry.item.base_sha else ''
            yield rid, expected_base, actual_base
            # Advance expected_base for the next entry regardless of whether
            # this one matched, so subsequent chain errors are also surfaced
            # (not shadowed).
            expected_base = mr.merge_commit.strip()

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

        The chain walk itself lives in :meth:`_frozen_base_chain` (shared with
        :meth:`_verify_base_frozen_tip_violations`); this method only compares
        expected vs. actual and formats its own violation wording.

        Pure/synchronous — reads stored in-memory state, no await.
        """
        violations: list[str] = []

        # ── 1. Base-chain integrity ───────────────────────────────────────────
        for rid, expected_base, actual_base in self._frozen_base_chain(main_sha):
            if actual_base != expected_base:
                violations.append(
                    f'frozen-prefix base-chain broken at {rid}: '
                    f'expected base {expected_base!r} but item has {actual_base!r}'
                )

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

        (iii) **verify-base ⊆ frozen-tip** (task 1999 / MQ-invariants ξ, I5):
             snapshot-granularity promotion of the dispatch-time
             :meth:`_warn_if_verify_base_not_frozen_tip` log-only guard — see
             :meth:`_verify_base_frozen_tip_violations` for the per-entry
             chained-base check.  Distinctly worded from
             :meth:`check_frozen_prefix_invariant`'s base-chain violation so
             it stands as its own named verify-frontier assertion target,
             even though it mirrors the same chain math.

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

        # ── (iii) verify-base ⊆ frozen-tip ───────────────────────────────────
        # Same main_sha-availability gate as the inherited base-chain check
        # above; own try/except so a failure here never shadows (or is
        # shadowed by) check_frozen_prefix_invariant's result.
        if main_sha and main_sha != 'unknown':
            try:
                violations.extend(self._verify_base_frozen_tip_violations(main_sha))
            except Exception as exc:  # pragma: no cover — defensive
                violations.append(f'two_layer_invariants: verify-base check raised: {exc}')

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

    def _verify_base_frozen_tip_violations(self, main_sha: str) -> list[str]:
        """Return verify-base/frozen-tip violations as human-readable strings (task 1999 I5).

        Snapshot-granularity promotion of :meth:`_warn_if_verify_base_not_frozen_tip`
        (the ε=1890 log-only dispatch guard): every currently-frozen entry's
        recorded ``base_sha`` must equal the frozen-tip base it was dispatched
        against — the head entry's expected base is *main_sha*, and each
        subsequent entry's expected base is its predecessor's ``merge_commit``.

        Walks the exact same chain as :meth:`check_frozen_prefix_invariant`,
        via the shared :meth:`_frozen_base_chain` generator (per-entry
        chained expected base, NOT a naive "every entry == newest tip"
        comparison, which would falsely flag a healthy multi-entry frozen
        prefix whose head is based on main rather than the stack's tip) —
        the two checks overlap mathematically but are kept as
        distinctly-worded, separately-named surfaces: this one parallels the
        retained ε=1890 dispatch-time WARN and gives it a dedicated
        verify-frontier assertion target in the snapshot health surface.
        Because both surfaces now build from the same generator, a future
        change to the chain-walk itself (e.g. how passthrough entries
        advance the chain) cannot make the two surfaces silently disagree.

        Pure/synchronous (no await). Fail-safe: never raises — callers
        (:meth:`two_layer_invariants`) wrap this in their own try/except, but
        the loop body itself cannot raise on well-formed InflightEntry data.
        """
        violations: list[str] = []
        for rid, expected_base, actual_base in self._frozen_base_chain(main_sha):
            if actual_base != expected_base:
                violations.append(
                    f'verify-base⊄frozen-tip at {rid!r}: dispatched for real-verify '
                    f'with base {actual_base!r} but expected frozen-tip base '
                    f'{expected_base!r} (ε=1890 §5.3 verify-base/frozen-tip rule)'
                )
        return violations

    # ── MQ-invariants iota (task 1994): resource-conservation audits ────────
    #
    # Two pure, fail-safe audit methods mirroring the two_layer_invariants
    # idiom immediately above: each returns list[str] (empty = healthy),
    # never raises, and is OBSERVATION ONLY (PRD design decision 4) — see
    # _check_resource_audit / _alarm_resource_audit below for the
    # heartbeat-driven WARNING + dedup'd L1 escalation this feeds.

    def _inflight_speculative_count(self) -> int:
        """Return the count of speculative items now owned by the verifier.

        Single source of truth for the ``inflight_speculative`` term shared
        by :meth:`snapshot`'s ``'speculation'`` key and
        :meth:`speculation_accounting_violations`.

        Task 2160 (η): conservation is now structural rather than location-
        audited. Every permit transfer (``SpeculationController.on_transfer``/
        ``on_transfer_terminal``) stamps the detached :class:`SpecPermit`
        token onto the pipeline item/entry's ``.permit`` field, and every
        verifier-side release routes through
        ``self._speculation_ledger.release(permit)``. So
        ``self._speculation_ledger.live`` already contains exactly the set of
        not-yet-released permits — both the one the merger currently holds
        (``held_by_merger``) and the ones transferred to the verifier (this
        method's return value). Subtracting the merger's own held permit from
        ``len(live)`` yields the verifier-owned count directly, independent
        of which of ``self._inflight``/``self._verifier_queue``/
        ``self._redispatch``/``self._finalizing_head``/
        ``self._dispatching_item`` the item currently sits in — the
        five-location scan those fields used to require (tasks 2063/2096) is
        no longer needed; the fields themselves remain for other consumers
        (snapshot's ``'entries'`` section, dispatch/remerge logic).
        """
        return (
            len(self._speculation_ledger.live)
            - self._speculation_controller.held_by_merger
        )

    def _inflight_cap_count(self) -> int:
        """Return the count of in-flight merge-ahead-cap permits.

        Task 2161 (θ): conservation is now structural rather than a
        ``_verifier_queue`` scan. Every acquired :class:`CapPermit` is
        registered in ``self._merge_ahead_ledger.live`` and discarded only
        when released ON-DRAIN (or by the caller's redispatch-clear, which
        follows an ON-DRAIN release that already happened — see
        ``_dispatch_item``), so ``len(live)`` is already exactly the
        in-flight count. Mirrors :meth:`_inflight_speculative_count`'s
        ledger-derived collapse (task 2160/η).
        """
        return len(self._merge_ahead_ledger.live)

    def speculation_accounting_violations(self) -> list[str]:
        """Return I4 permit/cap conservation violations as human-readable strings.

        Empty list → both conservation identities hold. Checks:

          (a) speculation-slot identity: ``slot_available + len(ledger.live)
              == depth`` (task 2160/η, PRD DD6) — the ledger-derived
              collapse of the permit-conservation identity task theta/1993
              built ``SpeculationController`` to make computable. Equivalent
              to the decomposed ``slot_available + held_by_merger +
              inflight_speculative == depth`` form (still exposed via
              :meth:`snapshot`'s ``'speculation'`` key for callers that want
              the merger/verifier split) since ``len(ledger.live) ==
              held_by_merger + inflight_speculative`` once every permit
              release routes through the ledger.
          (b) merge-ahead-cap identity: ``merge_ahead_ledger.slot_available +
              len(merge_ahead_ledger.live) == depth`` (task 2161/θ) — the
              ledger-derived collapse of the Mechanism-1 cap analogue,
              mirroring identity (a)'s structural form. Equivalent to the
              prior ``merge_ahead_cap._value + inflight_cap_count == depth``
              form since ``_inflight_cap_count`` now reads ``len(live)``
              directly.
          (c) merge-ahead-cap handoff cross-check: every :class:`RealMergeItem`
              currently sitting in ``_verifier_queue`` with a non-``None``
              ``cap_permit`` has that exact token present in
              ``merge_ahead_ledger.live``. (b) derives both of its operands
              from the ledger itself (``slot_available`` and ``len(live)``),
              so it is tautological against a handoff/threading regression
              (e.g. a stale or prematurely-released token stamped onto an
              item); this check instead walks the actual queue contents —
              independent of the ledger's own bookkeeping — and would catch
              exactly that class of bug even though (b) stays green.

        Returns ``[]`` immediately when ``not self._running``: ``stop()``
        deliberately over-releases both semaphores by ``depth + 1`` as a
        shutdown safety valve, which would otherwise read as a spurious
        violation (extra permits, not a leak).

        Pure/synchronous (no await, no git calls). Fail-safe: each identity
        check has its own try/except so an unexpected exception in one never
        suppresses the other, and is surfaced as a violation string rather
        than raised — mirrors :meth:`two_layer_invariants`'s idiom. Reads
        ``self._speculation_ledger``/``self._speculation_controller.snapshot()``
        and the cap helper below; deliberately does NOT call ``self.snapshot()``
        (that would recurse via the ``resource_audit`` key).
        """
        if not self._running:
            return []

        violations: list[str] = []

        # ── (a) speculation-slot identity ────────────────────────────────────
        try:
            spec = self._speculation_controller.snapshot()
            depth = spec['depth']
            slot_available = spec['slot_available']
            live_permits = len(self._speculation_ledger.live)
            total = slot_available + live_permits
            if total != depth:
                violations.append(
                    f'speculation-slot conservation violated: slot_available'
                    f'({slot_available}) + live_permits({live_permits}) == '
                    f'{total}, expected depth={depth}'
                )
        except Exception as exc:  # pragma: no cover — defensive
            violations.append(
                f'speculation_accounting_violations: speculation-slot check raised: {exc}'
            )

        # ── (b) merge-ahead-cap identity ─────────────────────────────────────
        try:
            depth = self._speculation_depth
            cap_available = self._merge_ahead_ledger.slot_available
            inflight_cap = self._inflight_cap_count()
            total = cap_available + inflight_cap
            if total != depth:
                violations.append(
                    f'merge-ahead-cap conservation violated: '
                    f'merge_ahead_cap_available({cap_available}) + '
                    f'inflight_cap_count({inflight_cap}) == {total}, '
                    f'expected depth={depth}'
                )
        except Exception as exc:  # pragma: no cover — defensive
            violations.append(
                f'speculation_accounting_violations: merge-ahead-cap check raised: {exc}'
            )

        # ── (c) merge-ahead-cap handoff cross-check ──────────────────────────
        # Independent of (b): (b)'s two operands both come from the ledger
        # itself, so it cannot see a handoff bug where an item's stamped
        # cap_permit has drifted out of sync with the ledger (a stale/foreign
        # token, or a premature release while the item is still queued). This
        # walks the actual verifier-queue contents instead — the same
        # internal-deque access `snapshot()` uses for its 'awaiting_verify'
        # section — and cross-validates each RealMergeItem's cap_permit
        # against merge_ahead_ledger.live by identity.
        #
        # This reintroduces an O(n) _verifier_queue scan that
        # _inflight_cap_count()'s ledger-derived len(live) collapse (above)
        # was written to avoid — deliberately: this method is a diagnostic/
        # audit path (heartbeat-driven, not per-item hot path), and (c) checks
        # a property (c) alone can see, so the O(1) win on the common
        # conservation check (b) still stands. Not gated behind a flag unless
        # this call site becomes hot.
        try:
            for _vq_item in list(self._verifier_queue._queue):  # type: ignore[attr-defined]
                if (
                    isinstance(_vq_item, RealMergeItem)
                    and _vq_item.cap_permit is not None
                    and _vq_item.cap_permit not in self._merge_ahead_ledger.live
                ):
                    violations.append(
                        'merge-ahead-cap handoff violated: RealMergeItem in '
                        '_verifier_queue carries a cap_permit not present in '
                        'merge_ahead_ledger.live (stale or prematurely-released '
                        'token)'
                    )
        except Exception as exc:  # pragma: no cover — defensive
            violations.append(
                f'speculation_accounting_violations: merge-ahead-cap handoff check raised: {exc}'
            )

        return violations

    def worktree_ledger_violations(self, *, now: float | None = None) -> list[str]:
        """Return I6 worktree-ledger violations as human-readable strings.

        Empty list → every on-disk ``_merge-*`` worktree is accounted for.
        Synchronously scans ``git_ops.worktree_base`` via ``os.scandir`` —
        NOT the async git-based ``GitOps._iter_merge_worktrees`` — because
        :meth:`snapshot` (this audit's primary caller) is strictly
        synchronous (no await, no lock).  A direct filesystem scan is also
        strictly-more-correct for leak detection: it catches worktrees git
        itself no longer tracks.

        Collects direct-child directories of ``worktree_base`` whose name
        starts with ``'_merge-'``, excluding the persistent warm worktree
        (:data:`PERSISTENT_MERGE_WORKTREE_NAME`, ``'_merge-verify'`` — reset
        in place every verify, never a leak).  Any such directory absent from
        :attr:`_owned_merge_worktrees` (this worker's liveness ledger; see
        its declaration for the full scope note) is a candidate leak, but is
        only flagged once its mtime age exceeds
        :attr:`RESOURCE_AUDIT_WORKTREE_GRACE_SECS`.  The grace window is the
        exemption mechanism for short-lived UNREGISTERED ``_merge-<uuid>``
        worktrees created by cold-shadow verify and drift-check (cleaned up
        in their caller's ``finally``, never registered) and for ordinary
        register/deregister races; a persistent leak eventually trips once
        it outlives the window.

        Returns ``[]`` immediately when ``not self._running``: mirrors
        :meth:`speculation_accounting_violations` — ``stop()`` drains and
        cleans up owned worktrees, so auditing during/after shutdown would
        report spurious violations.

        *now* is injectable for deterministic tests; defaults to
        ``time.time()``.

        Pure/synchronous (no await, no git subprocess). Fail-safe: never
        raises; any unexpected exception is caught and surfaced as a
        violation string, mirroring :meth:`two_layer_invariants`'s idiom.
        """
        if not self._running:
            return []

        violations: list[str] = []
        try:
            base = getattr(self._git_ops, 'worktree_base', None)
            if base is None or not base.is_dir():
                return []
            effective_now = now if now is not None else time.time()
            grace = self.RESOURCE_AUDIT_WORKTREE_GRACE_SECS
            owned = {p.resolve() for p in self._owned_merge_worktrees}
            with os.scandir(base) as it:
                candidates = list(it)
            for entry in candidates:
                name = entry.name
                if not name.startswith('_merge-') or name == PERSISTENT_MERGE_WORKTREE_NAME:
                    continue
                try:
                    if not entry.is_dir():
                        continue
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                path = Path(entry.path).resolve()
                if path in owned:
                    continue
                age = effective_now - mtime
                if age > grace:
                    violations.append(
                        f'unregistered on-disk merge worktree {path} '
                        f'(age {age:.0f}s > grace {grace:.0f}s) absent from '
                        f'owned ledger — possible leak'
                    )
        except Exception as exc:  # pragma: no cover — defensive
            violations.append(f'worktree_ledger_violations: check raised: {exc}')

        return violations

    async def reap_orphaned_merge_worktrees(
        self,
        *,
        recovered_branches: Collection[str] = (),
        now: float | None = None,
    ) -> dict[str, list[str]]:
        """Reap aged unregistered on-disk ``_merge-*`` worktrees (I6 leak
        closure, task 2060).

        The remediation counterpart to :meth:`worktree_ledger_violations`:
        that method only DETECTS leaked worktrees; this method REMOVES them.
        Intended to run once at worker construction/recovery time (see
        ``Harness._reap_orphaned_merge_worktrees``), backstopping two orphan
        sources — a caller's ``finally`` cleanup skipped by a mid-run SIGTERM,
        and an orchestrator restart that wipes :attr:`_owned_merge_worktrees`
        while the on-disk worktree (and its ``.git/worktrees`` admin entry)
        survives.

        Reuses :meth:`worktree_ledger_violations`'s exact os.scandir discovery
        so remediation stays symmetric with detection: direct children of
        ``git_ops.worktree_base`` whose name starts with ``'_merge-'``,
        excluding the persistent warm worktree
        (:data:`PERSISTENT_MERGE_WORKTREE_NAME`) and any path already present
        in :attr:`_owned_merge_worktrees`.  A candidate is only removed (via
        :meth:`GitOps.cleanup_merge_worktree`) once its mtime age exceeds
        :attr:`RESOURCE_AUDIT_WORKTREE_GRACE_SECS` — the same grace window
        :meth:`worktree_ledger_violations` uses — so a worktree from a
        just-started concurrent merge (register-after-create race) is never
        touched.

        Before the reap scan, *recovered_branches* (the branches of merge
        requests recovered from the durable journal — see
        ``Harness._recover_pending_merges``) are each resolved via
        :meth:`GitOps.find_inflight_merge_worktree`, the documented
        cross-restart branch→on-disk-worktree oracle.  Any match is
        RE-ADOPTED — registered into :attr:`_owned_merge_worktrees` via
        :meth:`_register_owned_merge_worktree` — rather than left to the reap
        scan, since it backs a legitimate in-flight merge regardless of age
        (re-adoption bypasses the grace gate entirely).

        *now* is injectable for deterministic tests; defaults to
        ``time.time()``.

        Returns ``{'readopted': [...], 'reaped': [...]}`` — string paths of
        every worktree re-adopted / removed this sweep.  Returns
        ``{'readopted': [], 'reaped': []}`` immediately when ``worktree_base``
        is missing or not a directory.
        """
        readopted: list[str] = []
        reaped: list[str] = []

        base = getattr(self._git_ops, 'worktree_base', None)
        if base is None or not base.is_dir():
            return {'readopted': readopted, 'reaped': reaped}

        for branch in recovered_branches:
            try:
                wt = await self._git_ops.find_inflight_merge_worktree(branch)
            except Exception:  # noqa: BLE001
                # Fail-open: one branch's oracle miss must never abort the
                # startup sweep (find_inflight_merge_worktree runs git).
                logger.warning(
                    'reap_orphaned_merge_worktrees: find_inflight_merge_worktree '
                    'failed for branch %s — skipping re-adoption',
                    branch, exc_info=True,
                )
                continue
            if wt is not None:
                self._register_owned_merge_worktree(wt)
                readopted.append(str(wt.resolve()))

        effective_now = now if now is not None else time.time()
        grace = self.RESOURCE_AUDIT_WORKTREE_GRACE_SECS
        # Computed AFTER re-adoption so the reap scan below skips paths just
        # re-adopted above (re-adoption is exempt from the grace gate).
        owned = {p.resolve() for p in self._owned_merge_worktrees}

        try:
            with os.scandir(base) as it:
                candidates = list(it)
        except OSError:
            # Fail-open: worktree_base vanished / became unreadable in the
            # TOCTOU window after the is_dir() guard above.  Honour the
            # never-raise contract — return the report-so-far (any re-adoptions
            # already applied are preserved) rather than aborting startup.
            # Mirrors worktree_ledger_violations' defensive scan wrap.
            logger.warning(
                'reap_orphaned_merge_worktrees: scandir of %s failed — '
                'skipping reap scan this sweep', base, exc_info=True,
            )
            candidates = []
        for entry in candidates:
            name = entry.name
            if not name.startswith('_merge-') or name == PERSISTENT_MERGE_WORKTREE_NAME:
                continue
            try:
                if not entry.is_dir():
                    continue
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            path = Path(entry.path).resolve()
            if path in owned:
                continue
            age = effective_now - mtime
            if age > grace:
                try:
                    await self._git_ops.cleanup_merge_worktree(path)
                except Exception:  # noqa: BLE001
                    # Fail-open: one worktree's removal failure must never
                    # abort the sweep of the remaining orphans.
                    logger.warning(
                        'reap_orphaned_merge_worktrees: cleanup_merge_worktree '
                        'failed for %s — leaving for a later sweep',
                        path, exc_info=True,
                    )
                    continue
                reaped.append(str(path))

        if reaped or readopted:
            logger.info(
                'reap_orphaned_merge_worktrees: reaped=%d readopted=%d',
                len(reaped),
                len(readopted),
            )

        return {'readopted': readopted, 'reaped': reaped}

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
        if not isinstance(item, RealMergeItem) or not item.base_sha:
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
        State values: queued, merging, remerging, awaiting_host, awaiting_verify,
          verifying, passthrough, gate_reverify, finalizing.
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
                self._entry_phase(infl),
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
        # The vestigial single-host _verify_item/_verify_phase fields (set but
        # never cleared after γ, causing a stale phantom entry) were deleted by
        # task λ (2173) — that phantom-entry class is now structurally
        # impossible.
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

        # 1c. Redispatch window: an item popped from the verifier queue —
        # either a speculative item bounced back because no host was free
        # (free_host_count() == 0, verify hosts < speculation_depth) or a
        # cascade-remerged item awaiting re-dispatch — parked on
        # self._redispatch, not yet re-appended to self._inflight.  Without
        # this section the item is invisible to dashboard/heartbeat entries
        # and depth for the whole multi-heartbeat window it spends parked
        # (task 2068, follow-up from esc-2063-2 / task 2063's I4
        # speculation-permit conservation characterization — 2063 fixes only
        # the conservation COUNT via _inflight_speculative_count() and
        # deliberately does not touch entries; this closes the matching
        # observability gap). This restores entries/depth visibility ONLY —
        # occupancy is deliberately left untouched: it derives solely from
        # self._inflight (hosts with an actual lease), and a redispatch-parked
        # item holds no host lease while it waits, so it must NOT be counted
        # toward occupancy. Mirrors 1b (_remerging_item)'s transient-window
        # side-field pattern. self._redispatch is front-priority (drained by
        # the verifier loop's DISPATCH-FILL ahead of _verifier_queue), so its
        # entries are listed here, ahead of the awaiting_verify section below.
        for _rd_item in self._redispatch:
            entries.append(_entry(
                _rd_item.request, 'awaiting_host',
                worktree_path=item_merge_wt(_rd_item),
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
                worktree_path=item_merge_wt(item),
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
        # (immediate-outcome delivery, no real verify task) produce None
        # here — no verify task is running.
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
            _fh if _fh is not None and self._entry_phase(_fh) in _verify_phases
            else (self._inflight[0] if self._inflight else None)
        )
        if _vip_head is not None and self._entry_phase(_vip_head) in _verify_phases:
            verify_in_progress = {
                'task_id': _vip_head.item.request.task_id,
                'phase': self._entry_phase(_vip_head),
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
            # θ=1993 additive key: merger-side speculation state (from
            # self._speculation_controller.snapshot()) plus
            # inflight_speculative — the count of speculative items now owned
            # by the verifier. Task 2160 (η): derived structurally as
            # len(self._speculation_ledger.live) - held_by_merger rather than
            # scanning self._inflight/self._verifier_queue/self._redispatch/
            # self._finalizing_head/self._dispatching_item (tasks 2063/2096's
            # historical location-scanning fixes — see
            # _inflight_speculative_count's docstring). Together these make
            # the permit-conservation identity slot_available + held_by_merger
            # + inflight_speculative == depth fully computable from
            # snapshot() (task iota's conservation audit). Pure synchronous
            # read — no await, no git calls. No collision with existing keys.
            'speculation': {
                **self._speculation_controller.snapshot(),
                'inflight_speculative': self._inflight_speculative_count(),
            },
            # ι=1994 additive key: resource-conservation audits (I4 permits/
            # caps + I6 worktree ledger). Each sub-key is the direct list[str]
            # result of the correspondingly-named audit method — empty list =
            # healthy. Pure synchronous read — no await, no git calls (see
            # speculation_accounting_violations / worktree_ledger_violations
            # docstrings for what each identity checks). No collision with
            # existing keys.
            'resource_audit': {
                'speculation_accounting': self.speculation_accounting_violations(),
                'worktree_ledger': self.worktree_ledger_violations(),
            },
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

        The WARNING log line is dedup'd exactly like the escalation: at most
        once per open episode per request_id (``stuck.already_warned``,
        cleared via ``self._request_ledger.mark_warned``) — otherwise a
        genuinely leaked/wedged request would log an identical WARNING on
        every ``_HEARTBEAT_POLL_S`` poll indefinitely. The escalation call
        below still runs on every sweep regardless of ``already_warned``; it
        has its own independent ``has_open_l1`` dedup so a resubmission after
        an operator resolves the L1 is unaffected by the log-dedup state.
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
            if not stuck.already_warned:
                logger.warning(
                    'merge request stuck: request_id=%s task=%s branch=%s age=%.0fs '
                    'phase=%s — dequeued but never resolved (silent-hang failure mode)',
                    stuck.request_id, stuck.task_id, stuck.branch, stuck.age_secs, stuck.phase,
                )
                self._request_ledger.mark_warned(stuck.request_id)
            _alarm_merge_request_stuck(
                self._escalation_queue, stuck, event_store=self._event_store,
            )

    def _check_resource_audit(self, now: float) -> None:
        """Run both resource-conservation audits and alarm on a persisting violation.

        Synchronous and clock-injectable (``now``) — mirrors
        :meth:`_check_request_liveness`'s clock-injection convention so tests
        can drive the escalation-streak boundary deterministically without
        real time.

        MQ-invariants iota (task 1994): OBSERVATION + ESCALATION ONLY (PRD
        design decision 4) — never mutates queue/inflight/worktree state and
        never halts the pipeline. Combines
        :meth:`speculation_accounting_violations` (I4 permits/caps) and
        :meth:`worktree_ledger_violations` (I6 worktree ledger, given *now*).

        A clean call (no violations) resets
        :attr:`_resource_audit_violation_streak` to 0 and returns. A
        violating call ALWAYS logs a WARNING naming every violation — unlike
        :meth:`_check_request_liveness`'s log-dedup, this is not
        per-episode-deduped, since the streak counter below already bounds
        how often the louder escalation fires — and increments the streak.
        Once the streak reaches :attr:`RESOURCE_AUDIT_ESCALATION_STREAK`
        consecutive violating calls, :func:`_alarm_resource_audit` is
        invoked on every further violating call; its own ``has_open_l1``
        dedup (see that function) ensures at most one open L1 regardless of
        how many times this method calls it.
        """
        violations = (
            self.speculation_accounting_violations()
            + self.worktree_ledger_violations(now=now)
        )

        if not violations:
            self._resource_audit_violation_streak = 0
            return

        logger.warning(
            'merge queue resource-conservation audit: %d violation(s) found '
            '(consecutive streak=%d): %s',
            len(violations), self._resource_audit_violation_streak + 1,
            '; '.join(violations),
        )
        self._resource_audit_violation_streak += 1

        if self._resource_audit_violation_streak >= self.RESOURCE_AUDIT_ESCALATION_STREAK:
            _alarm_resource_audit(
                self._escalation_queue, violations, event_store=self._event_store,
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

        MQ-invariants eta (task 1992): before either of the above checks,
        UNCONDITIONALLY runs the clock-injectable
        :meth:`_check_request_liveness` sweep — a request that has fallen out
        of every pipeline structure is invisible to ``snapshot()`` (depth==0
        would otherwise short-circuit this method before ever looking at it),
        so the liveness side-check must run first, on every poll, regardless
        of queue depth or heartbeat rate-limiting. Wrapped in try/except so a
        liveness-check bug can never suppress the depth heartbeat below
        (mirrors ``_heartbeat_loop``'s swallow-and-log convention). This
        side-check never affects this method's own return value, which still
        means exactly "a depth heartbeat was emitted".

        MQ-invariants iota (task 1994): immediately after, UNCONDITIONALLY
        and for the same reason, runs the clock-injectable
        :meth:`_check_resource_audit` sweep — a leaked speculation permit/
        merge-ahead-cap slot or an abandoned on-disk merge worktree can exist
        while the pipeline is otherwise idle (depth==0), which would
        otherwise short-circuit this method before the audit ever ran. Also
        wrapped in its own try/except so a resource-audit bug can never
        suppress the depth heartbeat below, and likewise never affects this
        method's own return value.
        """
        try:
            self._check_request_liveness(now)
        except Exception:
            logger.exception('merge queue heartbeat: request-liveness check failed')

        try:
            self._check_resource_audit(now)
        except Exception:
            logger.exception('merge queue heartbeat: resource-audit check failed')

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
        # θ (task 2161): routed through each ledger's release_for_shutdown(),
        # which over-releases the wrapped semaphore directly without touching
        # `live` -- the same over-release-is-safe shutdown valve as before,
        # now the sole acquire/release surface for both semaphores (P-2).
        self._speculation_ledger.release_for_shutdown(self._speculation_depth + 1)
        for ln in MERGE_LANES:
            self._lane_halt[ln].set()
        self._merge_ahead_ledger.release_for_shutdown(self._speculation_depth + 1)

        # Drain per-lane buffers (items already removed from _queue by the merger)
        # Intentionally mutates _lane_buffers directly (not via
        # _buffer_owned_request/_pop_next_pickable) from the stop coroutine —
        # a legitimate non-owner drain covered by _assert_single_writer's
        # not-self._running gate (task 1999 I7), not a wiring omission.
        for lane in MERGE_LANES:
            while self._lane_buffers[lane]:
                req = self._lane_buffers[lane].popleft()
                if not req.result.done():
                    req.result.set_result(shutdown)
                # MQ-reliability kappa (task 2169): stop() is a genuine last-
                # container exit for this request_id — retire unconditionally
                # (even if the Future was already done) or it leaks forever.
                self._retire_item(req.request_id)

        # Drain main queue (items not yet drained into lane buffers)
        while not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                # MQ-reliability kappa (task 2169): req may be the shutdown
                # sentinel itself (None) — guard before touching request_id.
                # A raw-queue item may also be genuinely pre-registry (never
                # drained); _retire_item is a documented no-op for an
                # unregistered request_id either way.
                if req is not None:
                    if not req.result.done():
                        req.result.set_result(shutdown)
                    self._retire_item(req.request_id)
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
                    _harvested_wt = item_merge_wt(_harvested)
                    if _harvested_wt is not None:
                        with contextlib.suppress(BaseException):
                            await self._cleanup_owned_merge_worktree(
                                _harvested_wt
                            )
                    if not _harvested.request.result.done():
                        _harvested.request.result.set_result(shutdown)
                    # MQ-reliability kappa (task 2169): last sighting of this
                    # request_id — retire it.
                    self._retire_item(_harvested.request.request_id)
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
                    _item_wt = item_merge_wt(item)
                    if _item_wt is not None:
                        with contextlib.suppress(BaseException):
                            await self._cleanup_owned_merge_worktree(_item_wt)
                    if not item.request.result.done():
                        item.request.result.set_result(shutdown)
                    # MQ-reliability kappa (task 2169): retire — this item
                    # never proceeds past AWAITING_VERIFY once drained here.
                    self._retire_item(item.request.request_id)
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
            # MQ-reliability kappa (task 2169): retire — this entry never
            # reaches its own FINALIZING/TERMINAL transition once stop()
            # short-circuits it here.
            self._retire_item(_ie_req.request_id)
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
            # η: release THROUGH the ledger (idempotent + discards the token
            # from ledger.live), guarded by the threaded token rather than
            # was_speculative — ledger.release(None) would AttributeError.
            if _ie.permit is not None:
                self._speculation_ledger.release(_ie.permit)
        # Intentionally mutates _inflight directly (not via _inflight_clear())
        # from the stop coroutine — a legitimate non-owner drain covered by
        # _assert_single_writer's not-self._running gate (task 1999 I7), not
        # a wiring omission.
        self._inflight.clear()

        # Drain _redispatch: items pending re-dispatch after a cascade.
        while self._redispatch:
            _rd = self._redispatch.popleft()
            if not _rd.request.result.done():
                _rd.request.result.set_result(shutdown)
            # MQ-reliability kappa (task 2169): retire — parked awaiting
            # re-dispatch, never reaches DISPATCHING again once dropped here.
            self._retire_item(_rd.request.request_id)
            _rd_wt = item_merge_wt(_rd)
            if _rd_wt is not None:
                with contextlib.suppress(BaseException):
                    await self._cleanup_owned_merge_worktree(_rd_wt)

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
                    _item_wt = item_merge_wt(item)
                    if _item_wt is not None:
                        with contextlib.suppress(BaseException):
                            await self._cleanup_owned_merge_worktree(_item_wt)
                    if not item.request.result.done():
                        item.request.result.set_result(shutdown)
                    # MQ-reliability kappa (task 2169): retire.
                    self._retire_item(item.request.request_id)
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
                # MQ-reliability kappa (task 2169): guard against the None
                # shutdown sentinel (already put on _queue above) before
                # touching request_id — mirrors the first-pass drain's guard.
                if req_post is not None:
                    if not req_post.result.done():
                        req_post.result.set_result(shutdown)
                    self._retire_item(req_post.request_id)
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

        # η: cancel any still-in-flight merge-verify dry-run investigations
        # (spawned by _spawn_merge_verify_dry_run and tracked in
        # self._background_tasks) so a detached run_dry_run_unblock — which
        # does git subprocess work and an LLM agent invocation — is not left
        # running after shutdown has been requested (task 2141 amendment
        # pass, review finding `resource_cleanup`). Mirrors the
        # _drift_check_tasks drain immediately above and
        # workflow.TaskWorkflow's own background_tasks cleanup. Take a
        # snapshot before iterating: the done-callback mutates the set.
        for _bt in list(self._background_tasks):
            if not _bt.done():
                _bt.cancel()
                with contextlib.suppress(BaseException):
                    await _bt

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
        #    carried by an InflightEntry in self._inflight — structurally
        #    absent from self._lane_buffers, so it is excluded without any
        #    explicit filter.  No buffer scan of _inflight_req/self._inflight is
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
        # MQ-reliability kappa (task 2169): group_req is a brand-new
        # request_id (GroupMergeRequest.request_id auto-generated at
        # construction, merge_types.py) that never passed through the
        # `_queue` drain, so it has no QUEUED registry entry to transition
        # from — register it directly at LANE_BUFFERED, matching the QUEUE
        # SURGERY below that places it straight into a lane buffer.
        self._register_item(group_req, initial=ItemLifecycleState.LANE_BUFFERED)

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
            # MQ-reliability kappa (task 2169): each absorbed single's OWN
            # request_id is a genuine terminal exit here — it never proceeds
            # past coalescing under its own id (the new GroupMergeRequest is
            # a separate, already-registered request_id) — so retire it
            # unconditionally, mirroring _resolve_or_drop_abandoned's
            # always-retire semantics.
            self._retire_item(s_req.request_id)

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
        # MQ-refactor theta (task 1993): all speculation state — spec_base,
        # prefetched, the held-permit flag, and the late-arrival
        # pending_spec_base/pending_predecessor pair — is now owned by
        # self._speculation_controller (constructed alongside
        # _speculation_slot in __init__).  See
        # merge_speculation_controller.py's module docstring for the full
        # ownership-transfer lifecycle.  Only exited_via_sentinel remains a
        # loop-local: it governs finally-block sentinel forwarding, which is
        # orthogonal to permit/speculation state.
        #
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
                #   (controller.is_idle(): spec_base=None and prefetched=None and
                #   pending_spec_base=None) so a train is never enqueued behind an
                #   unverified speculative merge commit (:5239 warning).  The
                #   task-1862 retain path records the predecessor's commit in
                #   pending_spec_base while spec_base remains None, so
                #   pending_spec_base must also be tested here — otherwise
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
                if self._speculation_controller.is_idle():
                    await self._maybe_coalesce_waiting_singles()

                # Get next request: use pre-fetched (speculative) item if available,
                # otherwise acquire from the lane-priority pick system.
                req = self._speculation_controller.take_prefetched()
                if req is None:
                    req = await self._acquire_next_request()
                    if req is None:
                        exited_via_sentinel = True
                        break  # shutdown sentinel
                    # ATTACH or FALLBACK: decide whether this fresh dequeue is a
                    # late arrival that can attach to the in-flight predecessor's
                    # merge commit (task 1862), delegated to the controller.
                    #
                    # ATTACH — all four conditions must hold:
                    #   (a) pending_spec_base: predecessor's commit was recorded
                    #   (b) held_by_merger: speculation permit is still held
                    #   (c) pending_predecessor: predecessor request is known
                    #   (d) predecessor still in-flight (result future not done)
                    #
                    # On ATTACH: spec_base = pending_spec_base so this late
                    # arrival merges against main+A, not plain main.  The
                    # retained permit transfers to the verifier item on drain.
                    #
                    # On FALLBACK (any condition fails): release the held permit
                    # if present and merge non-speculatively against actual main.
                    _pending_predecessor = self._speculation_controller.pending_predecessor
                    attached_base = self._speculation_controller.on_dequeue(req)
                    if attached_base is not None:
                        assert _pending_predecessor is not None
                        logger.debug(
                            'Task %s: late arrival attaches to in-flight '
                            'predecessor %s (spec_base=%s)',
                            req.task_id,
                            _pending_predecessor.task_id,
                            attached_base[:8],
                        )

                # MQ-invariants eta (task 1992): arm the request-liveness ledger
                # for this freshly-dequeued request.  Single hook site — covers
                # BOTH the prefetched-consumption branch (req = prefetched) and
                # the _acquire_next_request() branch above (whose `if req is
                # None` shutdown sentinel already `break`s the loop, so req is
                # guaranteed bound and non-None here).  Idempotent: a request
                # that is somehow already armed (should not happen at a single
                # hook site, but RequestLedger.on_dequeue is idempotent by
                # design) keeps its earliest dequeued_at rather than resetting
                # the age clock.  Observation-only — never resolves/mutates/halts;
                # see merge_request_ledger.py's module docstring.
                #
                # Known narrow blind spot: a speculative look-ahead item
                # harvested by _pop_next_pickable() into the `prefetched` local
                # (below, near the speculation-permit acquire) has already been
                # removed from the lane buffers/`_queue` but is NOT armed here
                # until it is consumed as `req` on the NEXT loop iteration — it
                # is invisible to both snapshot() and the liveness ledger while
                # merely sitting in `prefetched`. This is intentionally left
                # uncovered: the window is a single loop iteration (the very
                # next thing the loop does with a non-None `prefetched` is
                # consume it here), never spans an await, and prefetching only
                # happens when the current `req` is itself already armed and
                # being actively processed — so a hang cannot silently vanish,
                # it will simply attribute to the request that owns the
                # in-flight iteration until `prefetched` is consumed.
                self._request_ledger.on_dequeue(req, now=time.time())

                self._inflight_req = req  # track for stop() race resolution
                # MQ-reliability kappa (task 2169): no separate registry
                # transition needed here — `req` always came from a
                # `_pop_next_pickable()` pop (either just now via
                # `_acquire_next_request()`, or on a prior iteration via the
                # speculative look-ahead), which already transitioned it to
                # MERGING and indexed it in `_live_items`. By construction
                # the registry already agrees with this transient field.
                # ι=1894: stash main_position for this request so
                # _note_conflict_detected can compute drift later.
                self._note_merge_started(req.request_id)
                # Drop-on-detection: workflow soft-cancelled before worker
                # dequeued.  Skipping merge work avoids the orphan-halt
                # window where no escalation owner is registered.
                if self._request_abandoned(req):
                    self._speculation_controller.on_abort()
                    self._inflight_req = None
                    # ι=1894 amend: drop stashed drift base — request retired without
                    # landing or conflict detection, so it would otherwise leak forever.
                    self._drift_base.pop(req.request_id, None)
                    # MQ-reliability kappa (task 2169): this request never
                    # proceeds past MERGING — retire it here or it leaks in
                    # the registry forever (no Future is ever resolved on
                    # this path; the waiter already cancelled it).
                    self._retire_item(req.request_id)
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
                    spec = self._speculation_controller.spec_base
                    speculative = spec is not None
                    actual_main = await self._git_ops.get_main_sha()
                    base_for_merge = self._speculation_controller.base_for(actual_main)

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
                        if spec is not None:
                            logger.warning(
                                'Train %s: dequeued while speculative merge is '
                                'in-flight (spec_base=%s); advance_main retries '
                                'will absorb the CAS race — enqueuer should wait '
                                'for an idle pipeline before submitting a train',
                                req.train_id, spec[:12],
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
                        _train_decided_item = DecidedItem(
                            request=req,
                            base_sha=actual_main, speculative=False,
                            immediate_outcome=outcome,
                            started_monotonic=t0,
                        )
                        await self._verifier_queue.put(_train_decided_item)
                        # Train is put with speculative=False so the verifier
                        # will NOT release the slot on drain.  Release explicitly
                        # if the train was prefetched as a speculative item.
                        self._speculation_controller.on_abort()
                        self._note_transition(
                            req.request_id, ItemLifecycleState.MERGING,
                            ItemLifecycleState.AWAITING_VERIFY, live_obj=_train_decided_item,
                        )
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
                            OutcomeKind.abandoned_verify_timeouts,
                            attempt=prior_timeouts, duration_ms=_elapsed_ms(t0),
                        )
                        _abandon = self._abandon_outcome(req.task_id, prior_timeouts)
                        _already = self._oob_deliver(req, _abandon, speculative=speculative)
                        _decided_item = DecidedItem(
                            request=req,
                            base_sha=actual_main, speculative=speculative,
                            immediate_outcome=_abandon,
                            already_delivered=_already,
                            started_monotonic=t0,
                        )
                        await self._verifier_queue.put(_decided_item)
                        # η: stamp the detached token onto the enqueued item so
                        # the verifier can release this SAME token on drain.
                        # INVARIANT — no `await` between put() and this line:
                        # unbounded asyncio.Queue.put() never suspends, so the
                        # verifier cannot drain _decided_item while .permit is
                        # still None. An intervening await (or a future
                        # bounded queue that can suspend here) would let the
                        # verifier see permit=None and silently skip the
                        # release, leaking the token in ledger.live forever.
                        _decided_item.permit = self._speculation_controller.on_transfer_terminal()
                        self._note_transition(
                            req.request_id, ItemLifecycleState.MERGING,
                            ItemLifecycleState.AWAITING_VERIFY, live_obj=_decided_item,
                        )
                        self._inflight_req = None
                        continue

                    # ── Guard + merge + drop-guard pipeline (MQ-refactor kappa) ──
                    # classify_and_merge (module-level) covers branch-presence →
                    # already-merged → merge → conflict / non-conflict-failure →
                    # drop-guard.  It emits all merge_attempt/speculative_merge
                    # events internally, so the event stream is identical to the
                    # pre-extraction inline copy.  See its docstring for the full
                    # equivalence-matrix contract shared with _do_merge/_remerge.
                    result = await classify_and_merge(
                        self, req, base_for_merge, speculative=speculative,
                        started_monotonic=t0,
                    )
                    if isinstance(result, Decided):
                        # Uniform _oob_deliver call: its own predicate already
                        # excludes status in {'done','already_merged'}, so this
                        # is a no-op for the already-merged arm — identical to
                        # the old bare-put that skipped the call entirely.
                        _already = self._oob_deliver(req, result.outcome, speculative=speculative)
                        # base_sha=base_for_merge is used uniformly for EVERY
                        # Decided arm here (branch-presence / rev-parse-failure
                        # / already-merged included) — pre-kappa, those three
                        # arms used actual_main while only conflict/failure/drop
                        # used base_for_merge.  Confirmed inert for this
                        # immediate_outcome (terminal) item: _finalize_inflight's
                        # PASSTHROUGH branch resolves purely from
                        # entry.passthrough_outcome and never reads
                        # item.base_sha (:8381), _dispatch_item's Mechanism-2
                        # staleness check explicitly excludes
                        # item.immediate_outcome is not None (:8901), and
                        # _warn_if_verify_base_not_frozen_tip returns
                        # immediately when item.merge_result is None (:5228).
                        # So this divergence has no observable effect.
                        _decided_item = DecidedItem(
                            request=req,
                            base_sha=base_for_merge, speculative=speculative,
                            immediate_outcome=result.outcome,
                            already_delivered=_already,
                            failure_diagnostic=result.outcome.failure_diagnostic,
                            started_monotonic=t0,
                        )
                        await self._verifier_queue.put(_decided_item)
                        # η: stamp the detached token onto the enqueued item so
                        # the verifier can release this SAME token on drain.
                        # INVARIANT — no `await` between put() and this line:
                        # unbounded asyncio.Queue.put() never suspends, so the
                        # verifier cannot drain _decided_item while .permit is
                        # still None. An intervening await (or a future
                        # bounded queue that can suspend here) would let the
                        # verifier see permit=None and silently skip the
                        # release, leaking the token in ledger.live forever.
                        _decided_item.permit = self._speculation_controller.on_transfer_terminal()
                        self._note_transition(
                            req.request_id, ItemLifecycleState.MERGING,
                            ItemLifecycleState.AWAITING_VERIFY, live_obj=_decided_item,
                        )
                        self._inflight_req = None
                        continue

                    # ── Merge succeeded ────────────────────────────────────────
                    merge_result = result.merge_result
                    merge_result_local = merge_result  # track for cleanup on post-merge exception
                    branch_head = result.branch_tip
                    assert merge_result.merge_commit is not None
                    merge_commit = merge_result.merge_commit.strip()
                    # A successful merge always has a worktree; RealMergeItem.merge_wt
                    # is required non-None (task ο), so narrow explicitly.
                    assert merge_result.merge_worktree is not None

                    # Mechanism 1: cap non-speculative build-ahead.
                    # Trains (continue before this) and immediate-outcome guards
                    # (all return above) never reach this site, so `not speculative`
                    # is the exact predicate for blocking-path items.
                    counts_against_cap = not speculative
                    cap_permit: CapPermit | None = None
                    if counts_against_cap:
                        cap_permit = await self._merge_ahead_ledger.acquire()
                    _real_item: RealMergeItem | None = None
                    try:
                        self._register_owned_merge_worktree(merge_result.merge_worktree)
                        _real_item = RealMergeItem(
                            request=req, merge_result=merge_result,
                            merge_wt=merge_result.merge_worktree,
                            base_sha=base_for_merge, speculative=speculative,
                            started_monotonic=t0,
                            merged_branch_tip=branch_head,  # γ2: branch tip at merge time
                            cap_permit=cap_permit,
                        )
                        await self._verifier_queue.put(_real_item)
                    except BaseException:
                        # put() failed — the verifier will never drain this item
                        # and release the cap permit.  Release it here to prevent
                        # the merger from deadlocking at the next acquire.
                        #
                        # Double-release edge case: if CancelledError is raised
                        # at the `await` boundary AFTER put() already enqueued
                        # the item (a narrow asyncio race), this release fires
                        # AND the verifier releases again on drain — two release
                        # calls for the SAME token.  Unlike the speculation-slot
                        # permit (stamped onto the item AFTER put() succeeds —
                        # see below), cap_permit is necessarily stamped into the
                        # RealMergeItem constructor above, BEFORE put(); there is
                        # no deferred-stamping trick available here, so this
                        # genuinely is a double-release ATTEMPT on the same
                        # CapPermit token.  It is safe because
                        # PermitLedger.release() is idempotent (checks
                        # `.released` FIRST): the second call is a silent no-op,
                        # never an over-release or an assertion failure.
                        #
                        # The _speculation_slot / permit picture avoids the
                        # double-release attempt altogether via a DIFFERENT
                        # mechanism: execution never reaches the on_transfer()
                        # stamp below (it lives after this except block, past
                        # the `raise`), so `_real_item.permit` stays None.  The
                        # verifier later drains this item and, per its `if
                        # entry.permit is not None` release guard, skips
                        # releasing the ledger for it.  Meanwhile on_transfer()
                        # was never called, so the controller's held_by_merger
                        # stays True and the outer finally's on_shutdown() (or
                        # an enclosing except's on_abort()) performs the single
                        # authoritative ledger release of the still-held merger
                        # permit.  Net: released exactly once — via deferred
                        # stamping rather than the cap permit's idempotent
                        # double-release tolerance above.
                        if cap_permit is not None:
                            self._merge_ahead_ledger.release(cap_permit)
                            # Same race, a second symptom: if put() already
                            # enqueued _real_item onto _verifier_queue before
                            # the CancelledError propagated (the race above),
                            # that item is still sitting in the queue holding
                            # this now-released token —
                            # speculation_accounting_violations' identity (c)
                            # cross-checks every _verifier_queue item's
                            # cap_permit against merge_ahead_ledger.live and
                            # would report a transient false-positive "stale
                            # or prematurely-released token" violation until
                            # the verifier drains it. Clear it here so the
                            # item's visible state matches "no live token",
                            # exactly as an ordinary on-drain release is
                            # immediately followed by
                            # dataclasses.replace(item, cap_permit=None).
                            # Harmless when _real_item was never enqueued (or
                            # never constructed — the None-guard covers a
                            # failure before its assignment above): it is
                            # about to be discarded by the `raise` below.
                            if _real_item is not None:
                                _real_item.cap_permit = None
                        raise
                    # The put succeeded — verifier now owns the speculation
                    # permit for this item (released on drain if speculative).
                    # Plain on_transfer() (not on_transfer_terminal()): unlike
                    # the seven early-continue sites above, this is NOT a
                    # `continue` — the look-ahead immediately below re-derives
                    # spec_base for the NEXT request (on_lookahead_found) or
                    # clears all state (on_shutdown); the retain branch
                    # (on_lookahead_pending) leaves spec_base as-is (a
                    # pre-existing, harmless staleness window — never read
                    # again before the next on_dequeue overwrites it).
                    # η: stamp the detached token onto the enqueued item so
                    # the verifier can release this SAME token on drain.
                    # INVARIANT — no `await` between put() (above, in the
                    # try block) and this line: everything in between is
                    # synchronous comments plus an `except BaseException`
                    # handler that always re-raises before reaching here, so
                    # this line only runs when put() returned without
                    # suspending.  An intervening await would let the
                    # verifier drain _real_item while .permit is still None,
                    # silently skipping the release and leaking the token in
                    # ledger.live forever — same hazard as the
                    # on_transfer_terminal() sites above.
                    _real_item.permit = self._speculation_controller.on_transfer()
                    self._note_transition(
                        req.request_id, ItemLifecycleState.MERGING,
                        ItemLifecycleState.AWAITING_VERIFY, live_obj=_real_item,
                    )
                    self._inflight_req = None  # item is now owned by verifier

                    # ── Speculative look-ahead (depth-K cap) ──────────────────
                    # Acquire one speculation permit before the look-ahead peek.
                    # If an item is found and prefetched, the permit stays held;
                    # the Verifier releases it when draining this speculative item.
                    # If nothing is pickable (or shutdown), the permit is released
                    # immediately — symmetric accounting keeps in-flight speculations
                    # bounded at self._speculation_depth (K).
                    await self._speculation_controller.acquire_for_lookahead()  # depth-K cap
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
                        self._speculation_controller.on_lookahead_found(next_req, merge_commit)
                        logger.debug(
                            f'Task {req.task_id}: speculative look-ahead for '
                            f'{next_req.task_id} (base={merge_commit[:8]})'
                        )
                    elif self._shutdown_signaled:
                        self._speculation_controller.on_shutdown()  # return unused permit
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
                        self._speculation_controller.on_lookahead_pending(merge_commit, req)
                        # held_by_merger stays 1 — permit retained for late arrival
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
                    self._speculation_controller.on_abort()
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
                    self._speculation_controller.on_abort()
                    self._inflight_req = None
        finally:
            # Release any speculation permit still held (and clear all
            # speculation state) — covers BaseException paths (e.g.
            # CancelledError) that bypass the inner except clauses, as well
            # as the ordinary shutdown-sentinel break above (task 1993).
            self._speculation_controller.on_shutdown()
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

    def _inflight_append(self, entry: InflightEntry) -> None:
        """Append *entry* to ``self._inflight`` (verifier-owned single-writer choke point).

        Thin wrapper (task 1999 / MQ-invariants ξ, I7) asserting ownership
        via ``self._verifier_task`` before mutating — the single choke point
        every ``_verifier_loop`` append site routes through, mirroring
        ``_buffer_owned_request`` on the merger side.

        MQ-reliability kappa (task 2169): both append call sites only reach
        here for a REAL dispatch (``entry.verify_task`` is not None —
        passthrough entries finalize inline instead, never appending), and
        both call sites already transitioned the item to DISPATCHING right
        before calling ``_dispatch_item``, so this is always a
        DISPATCHING -> VERIFYING move. Single chokepoint covers both the
        DISPATCH-FILL and blocking-get call sites, mirroring
        ``_buffer_owned_request``'s single-chokepoint precedent.
        """
        self._assert_single_writer(self._verifier_task, '_inflight')
        self._note_transition(
            entry.item.request.request_id, ItemLifecycleState.DISPATCHING,
            ItemLifecycleState.VERIFYING, live_obj=entry,
        )
        self._inflight.append(entry)

    def _inflight_popleft(self) -> InflightEntry:
        """Pop and return the head of ``self._inflight`` (verifier-owned single-writer choke point).

        See :meth:`_inflight_append`.
        """
        self._assert_single_writer(self._verifier_task, '_inflight')
        return self._inflight.popleft()

    def _inflight_clear(self) -> None:
        """Clear ``self._inflight`` (verifier-owned single-writer choke point).

        See :meth:`_inflight_append`.  NOT used by :meth:`stop`'s shutdown
        drain, which intentionally mutates ``self._inflight`` directly from
        the stop coroutine (a non-owner task) — see the comment at that call
        site.
        """
        self._assert_single_writer(self._verifier_task, '_inflight')
        self._inflight.clear()

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
                        head = self._inflight_popleft()
                        await self._finalize_inflight(head)
                    return

                # Dispatch the item (applies Mechanism 1, abandon/halt/passthrough/
                # chain-remerge logic, host acquire, verify task launch).
                #
                # Dispatch-gap census (task 2096): self._dispatching_item is set
                # immediately before the await and cleared in a nested finally —
                # independent of the outer exception handling below — so a raise
                # inside _dispatch_item can never leave it stale.  This closes the
                # I4 speculation-permit census for the whole dispatch-gap window
                # (the item is off the queue but not yet in self._inflight).
                #
                # MQ-reliability kappa (task 2169): the registry mirrors this same
                # window — AWAITING_VERIFY -> DISPATCHING when `item` came from
                # `_verifier_queue` (is_from_verifier_queue), REDISPATCH_PARKED ->
                # DISPATCHING when it came from `_redispatch` — but, unlike the
                # census-only `_dispatching_item` field, covers BOTH
                # `_dispatch_item` call sites (this one and the blocking-get one
                # below), not just this one.
                try:
                    self._note_transition(
                        item.request.request_id,
                        ItemLifecycleState.AWAITING_VERIFY if is_from_verifier_queue
                        else ItemLifecycleState.REDISPATCH_PARKED,
                        ItemLifecycleState.DISPATCHING, live_obj=item,
                    )
                    self._dispatching_item = item
                    try:
                        entry = await self._dispatch_item(item)
                    finally:
                        self._dispatching_item = None
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
                    # No host available: put item back on _redispatch.  Only a
                    # RealMergeItem ever needs a host lease (passthroughs never
                    # fail to dispatch), so item is always a RealMergeItem here.
                    # cap_permit was already released in _dispatch_item; clear
                    # it to prevent a double-release on re-dispatch.
                    assert isinstance(item, RealMergeItem)
                    item_back = dataclasses.replace(item, cap_permit=None)
                    self._note_transition(
                        item_back.request.request_id, ItemLifecycleState.DISPATCHING,
                        ItemLifecycleState.REDISPATCH_PARKED, live_obj=item_back,
                    )
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
                            chain_failed=True,
                        )
                    continue  # don't append to _inflight; fetch next item

                self._inflight_append(entry)

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
                head = self._inflight_popleft()
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
                    # and speculation permit; the chokepoint's own release is
                    # a no-op here (idempotent lease FREE-check + permit
                    # .released guard) — it just resolves the future and
                    # marks the chain as failed.
                    await self._resolve_and_release(
                        head, MergeOutcome('blocked', reason=f'Verifier error: {exc}'),
                        chain_failed=True,
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
                    self._inflight_clear()

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
                    # Release discipline: lease+permit are released in-body
                    # at cancel_and_release / speculation_ledger.release()
                    # BEFORE _remerge is called. The permit release is
                    # genuinely idempotent (PermitLedger.release checks
                    # permit.released first), so the except handler below can
                    # re-run it unconditionally: no guard needed. The LEASE
                    # release is NOT idempotent for a REMOTE lease —
                    # HostAllocator.cancel_and_release unconditionally
                    # re-issues cancel_verify() before any FREE-check
                    # (verify_runner.py:2306), and a redundant cancel can
                    # return rc != 0 and PARK a healthy slot
                    # (verify_runner.py:2312). So _entry.lease is set to
                    # None immediately after a successful in-body
                    # cancel_and_release, below, which makes the chokepoint's
                    # own `lease is not None` guard skip the redundant cancel
                    # (task 2160/η).
                    for _entry in _downstream:
                        _entry_status: str | None = None
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
                                # Successful in-body cancel: clear the lease
                                # so the except handler's chokepoint call
                                # below (cancel_lease=True) does not re-issue
                                # a non-idempotent cancel_and_release on the
                                # same remote lease (task 2160/η step-10). If
                                # cancel_and_release itself raises, this line
                                # is never reached and the lease stays set,
                                # so the chokepoint still retries the cancel.
                                _entry.lease = None
                            if _entry.merge_wt is not None:
                                with contextlib.suppress(BaseException):
                                    await self._cleanup_owned_merge_worktree(
                                        _entry.merge_wt
                                    )
                            # LATE-ARRIVAL ATTACH SYMMETRY (task 1862 step-6):
                            # A late arrival B attached via pending_spec_base is
                            # dispatched with speculative=True and held_spec_permit
                            # retained (step-2), so B's InflightEntry carries a
                            # permit. The release below fires here (predecessor
                            # failed → B is a downstream entry), and _remerge
                            # returns speculative=False → re-dispatched B carries
                            # permit=None → no duplicate release.
                            # Slot symmetry is maintained on the late-arrival path
                            # identically to the standard prefetch path.
                            # η: release THROUGH the ledger, guarded by the
                            # threaded token rather than was_speculative.
                            if _entry.permit is not None:
                                self._speculation_ledger.release(_entry.permit)
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
                                    # MQ-invariants eta (task 1992): Future left
                                    # deliberately pending — remove the ledger
                                    # entry so this parked request never ages
                                    # out; the next dequeue re-arms it fresh.
                                    self._request_ledger.on_requeued(_entry_req.request_id)
                                    # MQ-reliability kappa (task 2169): mirror
                                    # the re-arm onto the lifecycle registry.
                                    self._note_requeue(_entry_req.request_id, live_obj=_entry_req)
                                continue
                            # MQ-reliability kappa follow-up (task 2441,
                            # amended per review): this is the ONE
                            # `_redispatch` append site task 2169 left
                            # unwired. Reconcile the registry to pipeline
                            # state around the remerge — current -> MERGING
                            # for the duration of _remerge(), then MERGING ->
                            # REDISPATCH_PARKED once landed on _redispatch —
                            # mirroring the RUNNER_UNAVAILABLE cascade template
                            # in _finalize_inflight. from_state is read
                            # dynamically off the registry (the same
                            # dynamic-current-state idiom as _note_requeue)
                            # rather than hardcoded VERIFYING: every downstream
                            # _inflight entry is guaranteed at VERIFYING by
                            # _inflight_append in the common case, but the
                            # documented cascade contract (_LEGAL_TRANSITIONS
                            # comment: "VERIFYING/GATE_REVERIFY/FINALIZING ->
                            # MERGING") allows a downstream entry to still be
                            # resident at GATE_REVERIFY or FINALIZING too — a
                            # hardcoded VERIFYING would misfire a spurious
                            # illegal-transition alarm for those. An
                            # unregistered rid (None) is a silent no-op for
                            # both hops, same as _note_requeue. Without this,
                            # the registry stays stale while the item is
                            # physically parked on _redispatch, so the
                            # DISPATCH-FILL drain's REDISPATCH_PARKED ->
                            # DISPATCHING _note_transition fails its
                            # from_state cross-check and fires a spurious
                            # dedup'd L1 escalation on every cascade-remerge
                            # redispatch.
                            _rid = _entry.item.request.request_id
                            _from_state = self._lifecycle.current(_rid)
                            if _from_state is not None:
                                self._note_transition(
                                    _rid, _from_state,
                                    ItemLifecycleState.MERGING, live_obj=_entry,
                                )
                            _remerged = await self._remerge(
                                _entry.item.request,
                                _entry.item.started_monotonic,
                            )
                            if _from_state is not None:
                                self._note_transition(
                                    _rid, ItemLifecycleState.MERGING,
                                    ItemLifecycleState.REDISPATCH_PARKED, live_obj=_remerged,
                                )
                            self._redispatch.append(_remerged)
                        except (asyncio.CancelledError, KeyboardInterrupt):
                            raise
                        except BaseException as _cascade_exc:
                            logger.exception(
                                'Task %s: unexpected cascade error',
                                _entry.item.request.task_id,
                            )
                            # The in-body release above may have already run
                            # (lease/permit/merge-worktree), or may not have
                            # completed if it raised (e.g. cancel_and_release
                            # itself raised). The permit/worktree releases
                            # are genuinely idempotent, so the chokepoint's
                            # release of those is safe either way. The lease
                            # is handled differently: a successful in-body
                            # cancel_and_release set _entry.lease to None
                            # above, so the chokepoint's own `lease is not
                            # None` guard skips a redundant (non-idempotent
                            # for a REMOTE lease) cancel_and_release; if the
                            # in-body cancel itself raised, _entry.lease is
                            # still set and this is the first-and-only
                            # release attempt (task 2160/η).
                            await self._resolve_and_release(
                                _entry, MergeOutcome(
                                    'blocked',
                                    reason=f'Verifier cascade error: {_cascade_exc}',
                                ),
                                chain_failed=True,
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

                # MQ-reliability kappa (task 2169): this branch's `item` is
                # ALWAYS sourced from `_verifier_queue` (the persistent-getter
                # harvest above or the direct get() here) — never from
                # `_redispatch` (checked first, and only reached, in
                # DISPATCH-FILL above) — so the registry's from_state is
                # unconditionally AWAITING_VERIFY, unlike the DISPATCH-FILL
                # call site. This call site previously had no `_dispatching_item`
                # (census-only field) coverage at all; kappa extends the field
                # here too (mirroring DISPATCH-FILL) so the legacy census and
                # the registry/`_live_items` agree on EVERY dispatch, not just
                # the DISPATCH-FILL one — a single fresh item on an otherwise
                # idle worker is dispatched from exactly this call site (the
                # verifier loop is already blocked on the queue get before the
                # merger's item arrives), so this is the common case, not an
                # edge case.
                try:
                    self._note_transition(
                        item.request.request_id, ItemLifecycleState.AWAITING_VERIFY,
                        ItemLifecycleState.DISPATCHING, live_obj=item,
                    )
                    self._dispatching_item = item
                    try:
                        entry = await self._dispatch_item(item)
                    finally:
                        self._dispatching_item = None
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
                    # system, but handle defensively: item goes to _redispatch).  Only
                    # a RealMergeItem ever needs a host lease, so item is always a
                    # RealMergeItem here (see the DISPATCH-FILL branch above).
                    assert isinstance(item, RealMergeItem)
                    item_back = dataclasses.replace(item, cap_permit=None)
                    self._note_transition(
                        item_back.request.request_id, ItemLifecycleState.DISPATCHING,
                        ItemLifecycleState.REDISPATCH_PARKED, live_obj=item_back,
                    )
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
                            chain_failed=True,
                        )
                    continue  # restart outer loop → fill loop picks up next item

                # Real verify entry: append to _inflight and loop back to fill.
                # The fill loop will block for the next item if a host slot is
                # free (multi-host overlap) OR break immediately (single-host,
                # free_host_count()==0) → FINALIZE-HEAD processes the head.
                self._inflight_append(entry)
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

        MQ-refactor kappa (task 1995): routed through the shared
        ``classify_and_merge`` guard pipeline — branch-presence, already-merged,
        drop-guard, and conflict/non-conflict-failure handling are now IDENTICAL
        to ``_do_merge``/``_merger_loop``.  This is an intentional behaviour
        GAIN documented in classify_and_merge's docstring and task 1995's design
        decisions: _remerge previously ran none of those guards.  The
        speculation-race retry below — this method's one truly unique behaviour
        — stays here, re-driving classify_and_merge a second time against a
        freshly-read main SHA (which also means the retry now gets its own
        full guard pass, including drop-guard).
        """
        actual_main = await self._git_ops.get_main_sha()
        result = await classify_and_merge(
            self, req, actual_main, speculative=False,
            started_monotonic=started_monotonic,
        )

        if isinstance(result, MergedOk):
            # A successful merge always has a worktree; RealMergeItem.merge_wt is
            # required non-None (task ο), so narrow explicitly.
            assert result.merge_wt is not None
            self._register_owned_merge_worktree(result.merge_wt)
            return RealMergeItem(
                request=req, merge_result=result.merge_result,
                merge_wt=result.merge_wt,
                base_sha=actual_main, speculative=False,
                merged_branch_tip=result.branch_tip,  # γ2: parity with merger loop
                started_monotonic=started_monotonic,
            )

        # ── Speculation-race retry ─────────────────────────────────────────────
        # When the first attempt fails with the load-bearing git porcelain phrase
        # ``not something we can merge`` (detected by _is_speculation_race) AND
        # the merge ran against a stale base (pre_merge_sha != actual_main), main
        # advanced between our get_main_sha() read and merge_to_main's own read,
        # so the worktree was built against a commit no longer on main.  Retry
        # exactly once against a freshly-read main HEAD to clear the stale-base
        # environment, by re-invoking classify_and_merge — which re-runs the
        # FULL guard pipeline against the new base.
        #
        # Design note: git emits this phrase when the merge argument (the branch
        # ref) cannot be resolved to a commit — e.g. a stale ref cache after
        # rapid concurrent pushes.  The pre_merge_sha != actual_main gate pins
        # the retry to cases where the base genuinely drifted; if the branch ref
        # was deleted or force-pushed between the two calls, the retry will fail
        # identically.  The full stderr is attached to the warning log below for
        # post-hoc diagnosis.
        mr = result.merge_result
        if (
            mr is not None
            and not mr.success
            and not mr.conflicts
            and _is_speculation_race(mr.details)
            and mr.pre_merge_sha is not None
            and mr.pre_merge_sha != actual_main
        ):
            retry_main = await self._git_ops.get_main_sha()
            logger.warning(
                'Task %s: speculation-race detected (first_base=%s, stderr=%r) '
                '— retrying against main %s',
                req.task_id, mr.pre_merge_sha[:8],
                mr.details[:120], retry_main[:8],
            )
            retry = await classify_and_merge(
                self, req, retry_main, speculative=False,
                started_monotonic=started_monotonic,
            )
            if isinstance(retry, MergedOk):
                logger.info(
                    'Task %s: merge_retry_after_speculation_race succeeded '
                    '(retry_base=%s)',
                    req.task_id, retry_main[:8],
                )
                # Verification always runs on the race-retry path (task-1724;
                # the skip_verify field itself was retired by task ο).
                #
                # This branch is reached ONLY after the gate confirmed main
                # advanced (merge_result.pre_merge_sha != actual_main): the branch
                # was pre-rebased onto the OLD main while the retry merges it
                # against the newer retry_main, integrating main commits the
                # branch never incorporated.  Skipping verification here would let
                # semantically-unverified main commits land on the protected
                # branch, so this always returns a RealMergeItem (verified by the
                # caller), never a passthrough.
                #
                # A successful merge always has a worktree; RealMergeItem.merge_wt
                # is required non-None (task ο), so narrow explicitly.
                assert retry.merge_wt is not None
                self._register_owned_merge_worktree(retry.merge_wt)
                return RealMergeItem(
                    request=req, merge_result=retry.merge_result,
                    merge_wt=retry.merge_wt,
                    base_sha=retry_main, speculative=False,
                    merged_branch_tip=retry.branch_tip,  # γ2: parity with merger loop
                    started_monotonic=started_monotonic,
                )

            # Retry Decided.  classify_and_merge already emitted whichever
            # merge_attempt event(s) apply (conflict / dropped_plan_targets /
            # already_merged / unknown_branch) and ran _note_conflict_detected
            # on a real conflict — nothing left to do for those but wrap the
            # outcome as-is.  Only a genuine non-conflict merge failure on
            # BOTH attempts gets the combined μ diagnostic treatment below
            # (failure_diagnostic is populated only on that path).
            first_diag = result.outcome.failure_diagnostic
            retry_diag = retry.outcome.failure_diagnostic
            if first_diag is None or retry_diag is None:
                return DecidedItem(
                    request=req,
                    base_sha=retry_main, speculative=False,
                    immediate_outcome=retry.outcome,
                    failure_diagnostic=retry_diag,
                    started_monotonic=started_monotonic,
                )

            # Retry non-conflict failure — combine μ diagnostics for BOTH attempts
            # and surface them together in reason and failure_diagnostic.
            combined_diag: dict[str, str] = {
                **retry_diag,
                'first_attempt_base_sha': first_diag['base_sha'],
                'first_attempt_git_stderr': first_diag['git_stderr'],
            }
            combined_reason = (
                f'Attempt 1: {result.outcome.reason}\n'
                f'Attempt 2 (retry against main {retry_main[:8]}): '
                f'{retry.outcome.reason}'
            )
            return DecidedItem(
                request=req,
                base_sha=retry_main, speculative=False,
                immediate_outcome=MergeOutcome(
                    'blocked',
                    reason=combined_reason,
                    failure_diagnostic=combined_diag,
                ),
                failure_diagnostic=combined_diag,
                started_monotonic=started_monotonic,
            )
        # ── END speculation-race retry ─────────────────────────────────────────

        # Non-retry Decided: branch-presence / already-merged / conflict /
        # non-conflict-failure / drop-guard — classify_and_merge already
        # emitted the matching event(s); just wrap its outcome.
        return DecidedItem(
            request=req,
            base_sha=actual_main, speculative=False,
            immediate_outcome=result.outcome,
            failure_diagnostic=result.outcome.failure_diagnostic,
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

        MQ-reliability kappa (task 2169): this is one of the two unified
        terminal chokepoints (the other, :meth:`_resolve_and_release`, itself
        calls this one) — every call site here is delivering a FINAL outcome
        or recognizing the request was already abandoned, so *req*'s
        request_id is retired (best-effort TERMINAL + drop from
        ``_live_items``) unconditionally, covering both branches below.
        """
        if self._request_abandoned(req):
            self._retire_item(req.request_id)
            return
        if not req.result.done():
            req.result.set_result(outcome)
        self._retire_item(req.request_id)

    async def _resolve_and_release(
        self,
        entry_or_item: InflightEntry | SpeculativeItem,
        outcome: MergeOutcome,
        *,
        chain_failed: bool,
        cancel_lease: bool = False,
    ) -> None:
        """Single resolve-and-release chokepoint for the _verifier_loop
        BaseException handlers (task 1991 / MQ-refactor zeta).

        Unifies the six near-identical except-handler bodies that used to
        inline: resolve the caller's Future, release the host lease (release
        vs cancel_and_release), clean up an owned merge worktree, release the
        speculation permit, and flag the chain as failed.  Each call site
        keeps its own ``logger.exception`` message and its own
        ``except (CancelledError, KeyboardInterrupt): raise`` clause; only the
        resolve-and-release tail is centralised here.

        *entry_or_item* is normalised to (req, lease, merge_wt, permit): an
        ``InflightEntry`` carries ``lease``/``merge_wt``/``permit``
        (post-dispatch state); a bare ``SpeculativeItem`` has no lease yet
        (pre-dispatch state) and uses ``merge_wt``/``permit`` directly.

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
        not-yet-released lease/permit.  No consumer or test depends on
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

        Releases are unconditional and idempotent for every resource except a
        ``cancel_lease=True`` release of a REMOTE lease (task 2160/η).
        ``_cleanup_owned_merge_worktree`` pops from ``_owned_merge_worktrees``
        and ``PermitLedger.release`` checks ``permit.released`` first, so
        both silently no-op on a repeat call; plain ``HostAllocator.release``
        also FREE-checks the lease slot before doing anything, so it too is
        idempotent. ``HostAllocator.cancel_and_release`` is NOT: for a remote
        lease it unconditionally re-issues ``cancel_verify()`` BEFORE any
        FREE-check (verify_runner.py:2306), and a redundant cancel on an
        already-cancelled remote verify can return a non-zero rc, which
        PARKs the slot rather than raising (verify_runner.py:2312) — a state
        mutation the surrounding ``contextlib.suppress`` does not undo. So a
        caller whose own code path already released these resources
        (``_finalize_inflight``'s ``finally``, which always uses plain
        ``release``; or the cascade handler's in-body release, which uses
        ``cancel_and_release`` and therefore also clears ``entry.lease`` to
        ``None`` immediately after a successful release) can invoke this
        chokepoint unconditionally, with no ``release_resources`` flag
        needed — removed along with the cascade loop's now-redundant
        ``_entry_released`` guard. The ordinary ``if lease is not None``
        guard below does double duty: it skips the lease step for a
        lease-less ``SpeculativeItem``/already-released ``InflightEntry``
        input, AND it is what the cascade's lease-clearing relies on to skip
        a redundant cancel.

        This is also the intended hook surface for task eta's liveness-ledger
        'resolved' transition: every _verifier_loop error-resolution path now
        funnels through this one coroutine.

        MQ-reliability kappa (task 2169): retirement (best-effort TERMINAL +
        drop from ``_live_items``) is NOT duplicated here — it is inherited
        from the internal :meth:`_resolve_or_drop_abandoned` call below,
        which every call site of THIS coroutine routes through.
        """
        if isinstance(entry_or_item, InflightEntry):
            item = entry_or_item.item
            lease: Any | None = entry_or_item.lease
            merge_wt = entry_or_item.merge_wt
            permit = entry_or_item.permit
        else:
            item = entry_or_item
            lease = None
            merge_wt = item_merge_wt(entry_or_item)
            permit = entry_or_item.permit
        req = item.request

        self._resolve_or_drop_abandoned(req, outcome)

        if lease is not None and self._host_allocator is not None:
            with contextlib.suppress(BaseException):
                if cancel_lease:
                    await self._host_allocator.cancel_and_release(lease)
                else:
                    await self._host_allocator.release(lease)
        if merge_wt is not None:
            with contextlib.suppress(BaseException):
                await self._cleanup_owned_merge_worktree(merge_wt)
        if permit is not None:
            self._speculation_ledger.release(permit)

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

    async def _post_submit_tasks(self, arguments_list: list[dict]) -> None:
        """Fire-and-forget: POST all submit_task calls to the fused-memory MCP.

        Worker-side mirror of ``TaskWorkflow._post_submit_tasks``
        (workflow.py:9092, task 2564) — a single shared ``httpx.AsyncClient``
        for the whole batch so only one TCP connection pool is opened
        regardless of how many tasks are being submitted.  Per-POST
        exceptions are caught and logged as warnings so a failure on one
        submission does not abort the rest.

        None-safe: no-ops when ``self._mcp`` is ``None`` (every bare-worker
        test constructor and any harness that hasn't wired an MCP client).
        """
        if self._mcp is None:
            return
        try:
            import httpx as httpx_mod
            async with httpx_mod.AsyncClient() as client:
                for arguments in arguments_list:
                    try:
                        await client.post(
                            f'{self._mcp.url}/mcp/',
                            json={
                                'jsonrpc': '2.0',
                                'id': 1,
                                'method': 'tools/call',
                                'params': {
                                    'name': 'submit_task',
                                    'arguments': arguments,
                                },
                            },
                            timeout=10,
                        )
                    except Exception as exc:
                        logger.warning(
                            'Failed to submit main-health fix task '
                            '(fire-and-forget): %s',
                            exc,
                        )
        except Exception as exc:
            logger.warning(
                'Failed to open HTTP client for main-health fix task '
                'submits: %s',
                exc,
            )

    async def _spawn_main_health_fix_task(
        self,
        req: MergeRequest,
        sig: str,
        esc_id: str,
        category: str,
        cause_hint: str,
        detail: str,
    ) -> None:
        """Schedule a HIGH-lane fix task for a confirmed main-health break.

        Worker-side mirror of ``TaskWorkflow._spawn_main_health_fix_task``
        (workflow.py:6106, task 2564).  Builds a submit_task argument block
        with:

        - title/description from :func:`compose_fix_main_brief`
        - ``priority='high'`` and ``metadata.merge_lane='high'`` so the fix
          merges via the HIGH lane
        - Correlation keys so the auto-watcher / ``unhalt_lanes_owned_by``
          can link the fix task back to this escalation: ``spawn_context``,
          ``main_health_signature``, ``main_health_escalation_id``

        Delegates the actual POST to :meth:`_post_submit_tasks` via
        ``asyncio.create_task`` (registered in ``self._background_tasks``,
        drained on shutdown — mirrors :func:`_spawn_merge_verify_dry_run`)
        so the caller is not blocked.
        """
        title, description = compose_fix_main_brief(category, cause_hint, detail)
        arguments = {
            'title': title,
            'description': description,
            'priority': 'high',
            'project_root': str(req.config.project_root),
            'metadata': {
                'merge_lane': 'high',
                'spawn_context': 'main_health_auto_heal',
                'main_health_signature': sig,
                'main_health_escalation_id': esc_id,
            },
        }
        try:
            task = asyncio.create_task(
                self._post_submit_tasks([arguments]),
                name=f'spawn_fix_main_{req.task_id}',
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except Exception as exc:
            logger.warning(
                'Task %s: failed to spawn main-health fix task: %s',
                req.task_id, exc,
            )

    async def _auto_heal_main_health_deferred(
        self, outcome: MergeOutcome, req: MergeRequest,
    ) -> None:
        """Worker-side auto-heal for a confirmed deferred main-health break (task 2564).

        Mirrors ``TaskWorkflow._auto_heal_main_health`` in full: branch (d)
        non-mechanical / no escalation_queue, branch (e) attempt-cap, branch
        (idempotency) lane-already-halted, and the (a) happy path — record
        the attempt, halt the 'normal' lane, submit/fold the dedup'd
        halt-owner escalation, register lane-halt ownership, and spawn a
        HIGH-lane fix task.  Called from the deferred main-health probe
        (:func:`_run_deferred_main_health_probe`, via the ``auto_heal``
        callback on :class:`_MainHealthProbeHandles`) once it confirms a
        still-fresh pre-existing break — this is the ONLY production path
        that can reach a main-health-red outcome, since the provisional
        outcome returned to the caller is always task-fault (see
        :func:`_run_post_merge_verify`'s DEFERRED-mode docstring), so
        ``TaskWorkflow._auto_heal_main_health`` (which keys off
        ``MAIN_HEALTH_RED_REASON_PREFIX``) never fires for it.

        Unlike ``TaskWorkflow._auto_heal_main_health`` there is no
        ``merge_worker is None`` guard — ``self`` (the
        :class:`SpeculativeMergeWorker`) already owns every auto-heal
        primitive directly (``auto_heal_registry``, ``halt_lane``, etc.), so
        that branch collapses into the non-mechanical / no-queue check.
        """
        # Local import — the SAME signature-keying authority
        # workflow._compute_merge_outcome_signature delegates to.  merge_queue
        # must NOT import from orchestrator.workflow (that would be a cycle);
        # shared.task_metadata has no orchestrator dependency, so this is safe.
        from shared.task_metadata import RetryLedger

        category = outcome.failure_category or ''
        cause_hint = outcome.failure_cause_hint or ''

        # Branch (d): non-mechanical, or no escalation_queue (cannot register
        # a halt-owner → unhalt_lanes_owned_by can never match → the 'normal'
        # lane would stay halted permanently → livelock).  Escalate-only, no
        # halt/spawn.
        if not is_auto_heal_eligible(category, cause_hint) or self._escalation_queue is None:
            _file_main_health_escalation(self._escalation_queue, req, outcome)
            return

        sig = RetryLedger.compute_merge_outcome_signature(
            category, cause_hint, outcome.reason,
        )

        # Branch (e): attempt cap reached — genuine re-break after a prior
        # heal.  Only fires when the lane is NOT currently halted; an
        # in-flight auto-heal (lane already halted) is the idempotency
        # branch below, not a re-break loop.
        if (
            self.auto_heal_registry.attempts(sig) >= MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS
            and not self.is_lane_halted('normal')
        ):
            _file_main_health_escalation(self._escalation_queue, req, outcome)
            return

        # Build and submit/fold the dedup'd halt-owner escalation — shared by
        # both the idempotency branch and the happy path below.
        esc_id = _file_main_health_escalation(
            self._escalation_queue, req, outcome,
            suggested_action='main_health_auto_heal_in_flight',
        )

        # Branch (idempotency): lane already halted → an auto-heal is already
        # in flight; the escalation above folds into it and no second
        # attempt/halt/owner/spawn is recorded.
        if self.is_lane_halted('normal'):
            return

        # Branch (a): happy path — record attempt, halt lane, register
        # owner, spawn fix.
        self.auto_heal_registry.record_attempt(sig)
        self.halt_lane(
            'normal',
            f'main-health auto-heal in flight (task {req.task_id})',
        )
        if esc_id:
            self.set_lane_halt_owner('normal', esc_id)

        await self._spawn_main_health_fix_task(
            req, sig, esc_id or '', category, cause_hint, outcome.reason,
        )

    async def _run_inflight_verify(
        self,
        item: RealMergeItem,
        lease: Any,  # HostLease
        depth: int | None = None,
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

        depth: Verify-frontier stack height (task 2340, ε=1890), computed
            synchronously by the caller (_dispatch_item) before this task is
            launched.  Forwarded into _run_post_merge_verify alongside
            item.speculative.  ``None`` (default) keeps the _verify_and_advance
            shim and any other non-production caller byte-identical.
        """
        req = item.request
        merge_wt = item.merge_wt
        assert merge_wt is not None
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
                escalation_queue=self._escalation_queue,
                dry_run_handles=self._dry_run_handles,
                main_health_probe_handles=_MainHealthProbeHandles(
                    background_tasks=self._background_tasks,
                ),
                depth=depth,
                speculative=item.speculative,
            ))
            # task 2420 (DEFECT 1, split from 2357; extends #1728): no-progress
            # budget seed.  LOCAL-only — a REMOTE lease's verify runs on the
            # remote host and writes nothing to this local merge_wt, so a
            # content-mtime budget would false-abort a healthy remote verify
            # (remote verify-hang is task 2362's ssh-keepalive territory, not
            # this trigger's).  newest_content_mtime never stats merge_wt's
            # own root inode, so the #1728 alpha owner-heartbeat's
            # os.utime(merge_wt) can never mask a dead verify here.
            _last_content_mtime = newest_content_mtime(merge_wt) if lease.is_local else None
            # task 2420 amend (reviewer finding, robustness): time.monotonic(),
            # not time.time(), for _last_progress_at/_last_probe_at/_now below
            # — all three are pure duration references (never persisted or
            # compared against a stored wall-clock value), so they must be
            # immune to NTP/manual clock steps. A backward step would delay a
            # real dead-verify abort; a forward step could prematurely trip
            # the budget and, after MAX_INFLIGHT_DEAD_VERIFY_ABORTS repeats,
            # false-'block' a healthy task. Matches the time.monotonic()
            # convention used elsewhere in this file (e.g. _elapsed_ms,
            # per-attempt t0 at the top of the dequeue loop).
            _last_progress_at = time.monotonic()
            _last_probe_at = _last_progress_at
            # task 2420 amend (reviewer finding, correctness): guard against
            # INFLIGHT_VERIFY_PROGRESS_PROBE_SECS not being comfortably
            # smaller than INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS. If the probe
            # interval were >= the budget, the very first eligible content
            # probe would land at/after the budget deadline, so the abort
            # would trip before a single progress probe ever ran — false-
            # aborting every LOCAL verify regardless of health. Clamp the
            # EFFECTIVE probe interval (local var; never mutates the class/
            # instance attribute) to a safe fraction of the budget and warn,
            # so a misconfigured or hot-reloaded pair cannot silently wedge
            # every local verify.
            _progress_probe_secs = self.INFLIGHT_VERIFY_PROGRESS_PROBE_SECS
            if lease.is_local and _progress_probe_secs >= self.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS:
                logger.warning(
                    'Task %s: INFLIGHT_VERIFY_PROGRESS_PROBE_SECS (%.3fs) >= '
                    'INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS (%.3fs) -- would '
                    'false-abort every local verify before a single progress '
                    'probe could run; clamping the effective probe interval '
                    'to budget/2 for this verify',
                    req.task_id, _progress_probe_secs,
                    self.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS,
                )
                _progress_probe_secs = self.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS / 2
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
                    # task 2420 amend (reviewer finding, resource_cleanup): this
                    # request is DROPPED (sole waiter gave up) — the per-task
                    # dead-verify-abort counter has no further purpose for it,
                    # so pop it here too (not just on the busy-loop-capped and
                    # normal-completion exit paths) to keep the dict scoped to
                    # genuinely live/in-flight task_ids on every exit path.
                    self._inflight_dead_verify_aborts.pop(req.task_id, None)
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
                    # MQ-invariants eta (task 1992): Future left deliberately
                    # pending — remove the ledger entry so this parked request
                    # never ages out; the next dequeue re-arms it fresh.
                    self._request_ledger.on_requeued(req.request_id)
                    # MQ-reliability kappa (task 2169): mirror the re-arm onto
                    # the lifecycle registry.
                    self._note_requeue(req.request_id, live_obj=req)
                    return InflightVerifyResult(
                        outcome=None,
                        merge_wt=None,
                        status=InflightStatus.REQUEUED,
                    )
                # Abort trigger 3 — LOCAL-only no in-flight verify progress
                # budget (task 2420 DEFECT 1, split from 2357; extends
                # #1728): terminate a deterministically dead/hung LOCAL
                # verify and RE-QUEUE, mirroring the operator-halt branch
                # above.  Checked last so abandon/halt precedence (triggers
                # 1/2) is preserved when they land on the same poll.  Gated
                # on lease.is_local — a REMOTE lease is never
                # progress-aborted (scope fence: task 2362 owns remote
                # verify-hang detection via ssh keepalive).  Content
                # progress under merge_wt resets the clock, so a genuinely
                # long-running healthy cold verify is never false-killed.
                if lease.is_local:
                    _now = time.monotonic()
                    if _now - _last_probe_at >= _progress_probe_secs:
                        _last_probe_at = _now
                        _cur_content_mtime = newest_content_mtime(merge_wt)
                        if _cur_content_mtime is not None and (
                            _last_content_mtime is None
                            or _cur_content_mtime > _last_content_mtime
                        ):
                            _last_content_mtime = _cur_content_mtime
                            _last_progress_at = _now
                    _no_progress_secs = _now - _last_progress_at
                    if _no_progress_secs >= self.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS:
                        # Busy-loop guard (task 2420): count CONSECUTIVE
                        # no-progress aborts per task_id.  Once the count
                        # reaches MAX_INFLIGHT_DEAD_VERIFY_ABORTS, stop
                        # re-queuing and resolve terminally instead —
                        # converting a deterministically-hanging verify into
                        # a loud 'blocked' escalation rather than an
                        # unbounded churn of the same dead slot.  Cleared on
                        # a successful verify for this task (see the `out is
                        # None` pass path below).
                        _dead_abort_n = self._inflight_dead_verify_aborts.get(req.task_id, 0) + 1
                        self._inflight_dead_verify_aborts[req.task_id] = _dead_abort_n
                        _busy_loop_capped = _dead_abort_n >= self.MAX_INFLIGHT_DEAD_VERIFY_ABORTS
                        logger.warning(
                            'Task %s: no in-flight verify progress for %.0fs '
                            '(budget=%.0fs) — %s (%d/%d consecutive dead aborts)',
                            req.task_id,
                            _no_progress_secs,
                            self.INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS,
                            'abandoning without re-queue' if _busy_loop_capped
                            else 'aborting and re-queuing merge for re-verify',
                            _dead_abort_n,
                            self.MAX_INFLIGHT_DEAD_VERIFY_ABORTS,
                        )
                        await self._abort_remote_verify(lease, req.task_id)
                        verify_task.cancel()
                        with contextlib.suppress(BaseException):
                            await verify_task
                        await self._release_or_cleanup(merge_wt, spec_warm=_spec_warm)
                        if _busy_loop_capped:
                            # task 2420 amend (reviewer finding #2): the
                            # counter has served its purpose once the
                            # request resolves terminally — pop it so a
                            # later re-submission/re-dispatch of this SAME
                            # task_id (e.g. after an operator resolves the
                            # 'blocked' outcome) gets a fresh dead-verify
                            # budget instead of immediately re-tripping the
                            # cap on its very first abort. Also bounds the
                            # dict to live/in-flight task_ids instead of
                            # growing unboundedly for permanently-blocked
                            # tasks.
                            self._inflight_dead_verify_aborts.pop(req.task_id, None)
                            err_outcome = MergeOutcome(
                                'blocked',
                                reason=(
                                    'repeated dead/hung in-flight verify (no '
                                    f'progress for budget) x{_dead_abort_n}'
                                ),
                            )
                            if not req.result.done():
                                req.result.set_result(err_outcome)
                            return InflightVerifyResult(outcome=err_outcome, merge_wt=None)
                        self._queue.put_nowait(req)
                        self._request_ledger.on_requeued(req.request_id)
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
            # task 2420 amend (reviewer finding, resource_cleanup): this is a
            # terminal 'blocked' resolution via a generic exception — pop the
            # per-task dead-verify-abort counter here too so a stale count
            # cannot linger for a task_id that exits via this path (mirrors
            # the abandoned/DROPPED and busy-loop-capped exit paths).
            self._inflight_dead_verify_aborts.pop(req.task_id, None)
            err_outcome = MergeOutcome('blocked', reason=f'Verification error: {exc}')
            if not req.result.done():
                req.result.set_result(err_outcome)
            return InflightVerifyResult(outcome=err_outcome, merge_wt=None)

        # task 2420 amend (reviewer finding #1): verify_task returned a
        # result HERE at all — pass, fail, or skipped — which proves this
        # task's verify subprocess was not hung.  Clear the per-task
        # no-progress busy-loop counter unconditionally (not only on the
        # `out is None` pass path below) so a hang -> real verify failure ->
        # hang sequence cannot silently accumulate two dead-abort counts
        # toward MAX_INFLIGHT_DEAD_VERIFY_ABORTS even though a verify
        # genuinely ran to completion in between.
        self._inflight_dead_verify_aborts.pop(req.task_id, None)

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
            #
            # MQ-reliability kappa (task 2169) / lambda (task 2173): the registry
            # VERIFYING -> FINALIZING hop is deferred to AFTER the verify_task
            # await below, NOT set here alongside _finalizing_head. _finalizing_head
            # is a pure snapshot() observability surface for the invisible-finalize
            # gap; the registry state is the item's true lifecycle position. While
            # entry.verify_task is still awaiting (a wedged/gated verify), the item
            # is genuinely VERIFYING — main's per-entry phase stayed 'verifying'
            # across that await and only became 'finalizing' after it returned. Firing
            # the FINALIZING hop here (before the await) would mislabel a wedged verify
            # as finalizing (regression caught by esc-2173-6:
            # test_wedged_verify_is_armed_and_alarmed_then_resolves_cleanly). The hop
            # now fires just past the await, before any sentinel/RU/fail/pass handling
            # (all of which already assume FINALIZING).
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
                # InflightEntry's shadow invariant (merge_types.py __post_init__)
                # guarantees a passthrough entry always wraps a DecidedItem (task ο).
                assert isinstance(item, DecidedItem)
                if not item.already_delivered and not req.result.done():
                    req.result.set_result(entry.passthrough_outcome)
                # Mirrors original verifier-loop line :6473:
                #   n_failed = item.immediate_outcome.status not in ('done', 'already_merged')
                _n_failed_val = (
                    entry.passthrough_outcome.status not in ('done', 'already_merged')
                )
                _skip_release = True  # passthrough entries have no lease
                # MQ-reliability kappa (task 2169): passthrough never reaches
                # FINALIZING (see the guard above) — retire straight from
                # whatever state it currently holds (DISPATCHING).
                self._retire_item(req.request_id)
                return entry.passthrough_outcome.status in ('done', 'already_merged')

            # ── Await verify task (if any) ───────────────────────────────────
            # verify_task=None means PASS was pre-established (compat shim /
            # step-12 tests where entry is constructed with a known-pass worktree).
            vr: InflightVerifyResult | None = None
            if entry.verify_task is not None:
                vr = await entry.verify_task

            # MQ-reliability lambda (task 2173): the verify await has returned, so
            # the item is now genuinely finalizing — fire the VERIFYING -> FINALIZING
            # hop here (deferred from the _finalizing_head assignment above so a
            # wedged verify reads VERIFYING, not FINALIZING). Guarded exactly as
            # before: passthrough / pre-dispatch-sentinel entries already returned
            # above and never reach here, but keep the guard explicit so the legal
            # edge (VERIFYING -> FINALIZING) is never mis-applied to an entry sitting
            # at DISPATCHING/QUEUED. Every path below (DROPPED/REQUEUED, RUNNER_
            # UNAVAILABLE's FINALIZING -> MERGING, FAIL/skip, PASS) already assumes
            # FINALIZING as its from-state.
            if (
                entry.passthrough_outcome is None
                and entry.status not in (
                    InflightStatus.ABANDONED_PREDISPATCH,
                    InflightStatus.REQUEUED_PREDISPATCH,
                )
            ):
                self._note_transition(
                    req.request_id, ItemLifecycleState.VERIFYING,
                    ItemLifecycleState.FINALIZING, live_obj=entry,
                )

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
                #
                # MQ-reliability kappa (task 2169): the registry observes
                # MERGING for the duration of the _remerge() call, then the
                # re-merged item lands at REDISPATCH_PARKED — it re-enters the
                # pipeline via _redispatch for a fresh dispatch attempt rather
                # than returning to FINALIZING/DISPATCHING, mirroring the
                # downstream head-failure cascade's own remerge-then-redispatch
                # shape.
                self._note_transition(
                    req.request_id, ItemLifecycleState.FINALIZING,
                    ItemLifecycleState.MERGING, live_obj=entry,
                )
                _remerged_ru = await self._remerge(
                    entry.item.request, entry.item.started_monotonic,
                )
                self._note_transition(
                    req.request_id, ItemLifecycleState.MERGING,
                    ItemLifecycleState.REDISPATCH_PARKED, live_obj=_remerged_ru,
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
            # Reached only by falling through the PASSTHROUGH/pre-dispatch returns
            # above, both of which return unconditionally for a DecidedItem-backed
            # entry — item is always a RealMergeItem here (task ο).
            assert isinstance(item, RealMergeItem)
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
            current_sha = merge_commit
            while True:
                # Write-ahead (PRD WA-1): record a LandedRow into the durable
                # outbox BEFORE advancing main — single-sourced via the shared
                # helper (task β).  Re-recorded each loop iteration so a
                # rebased retry's current_sha stays in sync (idempotent
                # last-write-wins, WA-2).
                adv_outcome = await _journal_landed_then_advance(
                    self._landed_outbox, self._git_ops,
                    task_id=req.task_id,
                    branch_tip_sha=item.merged_branch_tip,
                    advanced_sha=current_sha,
                    merge_wt=merge_wt,
                    branch=req.branch,
                    max_attempts=req.config.max_advance_attempts,
                    expected_main=item.base_sha,
                    reverify_on_rebase=True,
                )
                result = adv_outcome.result

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
                        advanced_sha=adv_outcome.advanced_sha,
                    )
                    self._resolve_or_drop_abandoned(req, outcome)
                    if outcome.status == 'done':
                        # ι=1894: record clean landing, pop drift base for this req
                        self._note_merge_landing(req.request_id)
                        # Task 2357 DEFECT 2: refresh the observation-only
                        # _last_known_main_sha cache here too — this land IS the
                        # new main tip.  Without this, an idle merger (blocked in
                        # _acquire_next_request, the cache's only other write
                        # site) leaves snapshot()'s two_layer_invariants() fed a
                        # stale SHA while the verifier keeps landing, producing
                        # false §5.3 verify-base⊄frozen-tip positives.
                        if outcome.merge_sha is not None:
                            self._last_known_main_sha = outcome.merge_sha
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
                    # advance_main always populates all three fields when it
                    # constructs AdvanceOutcome('rebased_pending_reverify', ...),
                    # but they're typed str | None on the dataclass; narrow
                    # explicitly for pyright (task 1996 explicit-guard style)
                    # rather than re-adding a verbose per-field diagnostic.
                    rebased_sha = adv_outcome.advanced_sha
                    rebased_from = adv_outcome.rebased_from
                    rebased_onto = adv_outcome.rebased_onto
                    if rebased_sha is None or rebased_from is None or rebased_onto is None:
                        raise AssertionError(
                            f'advance_main returned rebased_pending_reverify '
                            f'without SHA fields (task {req.task_id})'
                        )

                    self._note_transition(
                        req.request_id, ItemLifecycleState.FINALIZING,
                        ItemLifecycleState.GATE_REVERIFY, live_obj=entry,
                    )
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
                            self._event_store, req.task_id, OutcomeKind.cas_exhausted,
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
                        self._event_store, req.task_id, OutcomeKind.gate_retry,
                        attempt=gate_total,
                        duration_ms=_elapsed_ms(item.started_monotonic),
                    )
                    self._note_transition(
                        req.request_id, ItemLifecycleState.GATE_REVERIFY,
                        ItemLifecycleState.FINALIZING, live_obj=entry,
                    )
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
                        advanced_sha=adv_outcome.advanced_sha,
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
                    _emit_merge_attempt(self._event_store, req.task_id, OutcomeKind.cas_exhausted, attempt=total, duration_ms=_elapsed_ms(item.started_monotonic))
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
                _emit_merge_attempt(self._event_store, req.task_id, OutcomeKind.cas_retry, attempt=total, duration_ms=_elapsed_ms(item.started_monotonic))

        finally:
            # Always: release the host lease (unless passthrough / already skipped),
            # release the speculation permit iff one was threaded, and update
            # _n_failed.
            if not _skip_release and entry.lease is not None and self._host_allocator is not None:
                if _cancel_release:
                    await self._host_allocator.cancel_and_release(entry.lease)
                else:
                    await self._host_allocator.release(entry.lease)
            # η: release THROUGH the ledger, guarded by the threaded token
            # rather than was_speculative — ledger.release(None) would
            # AttributeError on `.released`.
            if entry.permit is not None:
                self._speculation_ledger.release(entry.permit)
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
        item back on self._redispatch unchanged — cap_permit already released,
        so caller must clear it first via dataclasses.replace).

        Handles in order:
          1. Mechanism 1 cap release (cap_permit).
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
        # items put back onto _redispatch have cap_permit cleared.
        # θ: release THROUGH the ledger, guarded by the threaded token rather
        # than a bool — ledger.release(None) would AttributeError on `.released`.
        if isinstance(item, RealMergeItem) and item.cap_permit is not None:
            self._merge_ahead_ledger.release(item.cap_permit)

        # ── Pre-dispatch abandon ────────────────────────────────────────────
        if self._request_abandoned(req):
            _abandon_wt = item_merge_wt(item)
            if _abandon_wt is not None:
                with contextlib.suppress(BaseException):
                    await self._cleanup_owned_merge_worktree(_abandon_wt)
            self._remerge_occurred = False  # abandon → reset chain flag
            # MQ-reliability kappa (task 2169): retire at this, the item's
            # own return site (mirrors the REQUEUED_PREDISPATCH branch's
            # own _note_requeue call below) — no Future is ever resolved on
            # this path (the waiter already cancelled it), so retirement is
            # the only observable exit for this request_id.
            self._retire_item(req.request_id)
            return InflightEntry(
                item=item,
                lease=None,
                verify_task=None,
                merge_wt=None,
                was_speculative=item.speculative,
                status=InflightStatus.ABANDONED_PREDISPATCH,
                permit=item.permit,
            )

        # ── Pre-dispatch operator-halt ──────────────────────────────────────
        # immediate_outcome items (trains / already-decided) are NOT halted here;
        # they fall through to the passthrough branch so they resolve in order.
        if self._operator_halt.is_set() and isinstance(item, RealMergeItem):
            # merge_wt is a required non-Optional Path on RealMergeItem — no
            # None/truthy guard needed here (task ο).
            with contextlib.suppress(BaseException):
                await self._cleanup_owned_merge_worktree(item.merge_wt)
            self._queue.put_nowait(req)
            # MQ-invariants eta (task 1992): Future left deliberately pending —
            # remove the ledger entry so this parked request never ages out;
            # the next dequeue re-arms it fresh.
            self._request_ledger.on_requeued(req.request_id)
            # MQ-reliability kappa (task 2169): mirror the re-arm onto the
            # lifecycle registry.
            self._note_requeue(req.request_id, live_obj=req)
            self._remerge_occurred = False  # halt → reset chain flag
            return InflightEntry(
                item=item,
                lease=None,
                verify_task=None,
                merge_wt=None,
                was_speculative=item.speculative,
                status=InflightStatus.REQUEUED_PREDISPATCH,
                permit=item.permit,
            )

        # ── Immediate outcome (conflict / already_merged / blocked) ────────
        if isinstance(item, DecidedItem):
            self._remerge_occurred = False  # passthrough → reset chain flag
            return InflightEntry(
                item=item,
                lease=None,
                verify_task=None,
                merge_wt=None,
                was_speculative=item.speculative,
                passthrough_outcome=item.immediate_outcome,
                permit=item.permit,
            )
        # item: RealMergeItem from here on (DecidedItem always returns above; task ο).

        # ── Real item: host acquire + verify dispatch ───────────────────────
        # Fast-path: if no host is free RIGHT NOW, return None so the caller
        # puts the item back on _redispatch (cap_permit already released
        # above; caller must clear it on the item before putting back).
        # Checked BEFORE the potentially-expensive _remerge call so no work is
        # done for an item that will be re-tried on a free host.
        allocator = self._ensure_host_allocator(req.config)
        if allocator.free_host_count() == 0:
            return None

        # Capture the speculative flag BEFORE any _remerge reassignment so
        # the InflightEntry carries the ORIGINAL speculative state for slot
        # release (same pattern as old loop's item_was_speculative).
        # item_permit travels alongside it (η) — _remerge always returns a
        # non-speculative item (see the LATE-ARRIVAL ATTACH SYMMETRY note in
        # the cascade handler above), so the ORIGINAL item's permit is the
        # one this dispatch must release.
        item_was_speculative = item.speculative
        item_permit = item.permit
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
                #
                # item is always a RealMergeItem here (a DecidedItem always returns
                # via the isinstance(item, DecidedItem) passthrough above), so the
                # old `item.immediate_outcome is None and item.merge_result is not
                # None` guard is now type-guaranteed rather than a runtime check
                # (task ο).
                not item.speculative
                and not isinstance(req, GroupMergeRequest)
            ):
                # Mechanism 2: check staleness at pickup for non-speculative items.
                current_main = await self._git_ops.get_main_sha()
                if item.base_sha != current_main:
                    remerge_reason = 'main_advanced'

            if remerge_reason is not None:
                iteration_did_remerge = True
                # MQ-reliability kappa (task 2169): DISPATCHING -> MERGING for
                # the duration of this dispatch-time remerge (Mechanism 2) —
                # see the kappa addendum to _LEGAL_TRANSITIONS' docstring above
                # this class for why this is a DIFFERENT remerge window than
                # the VERIFYING/GATE_REVERIFY/FINALIZING head-failure cascade.
                self._note_transition(
                    req.request_id, ItemLifecycleState.DISPATCHING,
                    ItemLifecycleState.MERGING, live_obj=req,
                )
                # Set _remerging_item so snapshot() surfaces this request during
                # the remerge window (item is popped from queue but not yet in
                # _inflight, so without this it is invisible to all observability).
                # Cleared to None immediately after _remerge() returns.
                self._remerging_item = req
                # merge_wt is a required non-Optional Path on RealMergeItem — no
                # truthy guard needed here (task ο).
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
                # MQ-reliability kappa (task 2169): "then back" — regardless of
                # whether the re-merged item now falls through to a passthrough
                # outcome (-> TERMINAL, wired by a later step) or proceeds to a
                # normal host-acquire + verify dispatch (-> VERIFYING via
                # _inflight_append below), the remerge itself is over, so the
                # registry moves back to DISPATCHING immediately.
                self._note_transition(
                    req.request_id, ItemLifecycleState.MERGING,
                    ItemLifecycleState.DISPATCHING, live_obj=item,
                )

                # After remerge the new item may itself carry an immediate_outcome
                # (e.g. conflict during remerge, or a train slot).  Return it as a
                # passthrough so _run_inflight_verify is never called with a
                # DecidedItem (item is the RealMergeItem | DecidedItem union again
                # here, since _remerge can return either variant; task ο).
                if isinstance(item, DecidedItem):
                    self._remerge_occurred = iteration_did_remerge
                    return InflightEntry(
                        item=item,
                        lease=None,
                        verify_task=None,
                        merge_wt=None,
                        was_speculative=item_was_speculative,
                        passthrough_outcome=item.immediate_outcome,
                        started_at=time.time(),
                        permit=item_permit,
                    )
                # item: RealMergeItem again (post-remerge fallthrough; task ο).

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
            # DEFECT 2 (task 2357): refresh the §5.3 snapshot cache from the
            # guard's own fresh SHA so snapshot()['two_layer_invariants'] never
            # lags behind this dispatch's view of main (piggybacks the fetch
            # above; no extra git round-trip).
            self._last_known_main_sha = _guard_main_sha
            self._warn_if_verify_base_not_frozen_tip(item, _guard_main_sha)
        except Exception:
            pass  # fail-open: skip the check (and the refresh) on any git error

        # ── Launch background verify task ────────────────────────────────────
        # depth (task 2340) is computed synchronously HERE — before
        # ensure_future launches the verify task — rather than read back out
        # of the deque later, so it reflects exactly the items already
        # frozen/verifying AHEAD of this item joining the frontier (no
        # async-timing fragility from a concurrent dispatch mutating
        # self._inflight between now and when the task actually runs).
        depth = self._verify_frontier_depth()
        verify_task: asyncio.Task = asyncio.ensure_future(  # type: ignore[type-arg]
            self._run_inflight_verify(item, lease, depth=depth)
        )

        return InflightEntry(
            item=item,
            lease=lease,
            verify_task=verify_task,
            merge_wt=item.merge_wt,
            was_speculative=item_was_speculative,
            started_at=time.time(),
            permit=item_permit,
        )

    async def _verify_and_advance(self, item: RealMergeItem) -> bool:
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
            started_at=time.time(),
        )

        return await self._finalize_inflight(entry)


# ── MQ-invariants iota (task 1994): resource-audit escalation ───────────────
#
# Module-level function (mirrors merge_request_ledger._alarm_merge_request_stuck,
# which is likewise a bare function rather than a worker method) fired by
# SpeculativeMergeWorker._check_resource_audit once a resource-conservation
# violation (I4 speculation permits/merge-ahead caps, I6 worktree ledger)
# persists across RESOURCE_AUDIT_ESCALATION_STREAK consecutive heartbeats.
# See speculation_accounting_violations / worktree_ledger_violations above
# for what each identity checks.

_RESOURCE_AUDIT_SENTINEL = '__merge_resource_leak__'
"""Fixed dedup sentinel task_id for resource-conservation-audit alarms.

Unlike ``_merge_request_stuck_sentinel`` (parameterized per request_id), this
audit is worker-level, not per-request — there is exactly one open/resolved
L1 at a time for the whole worker's resource-conservation health, so the
sentinel is a single fixed string rather than parameterized.
"""


def _alarm_resource_audit(
    escalation_queue: Any,
    violations: list[str],
    *,
    event_store: Any = None,
) -> None:
    """Submit a dedup'd L1 escalation for persisting resource-conservation violations.

    Modeled verbatim on
    :func:`orchestrator.merge_request_ledger._alarm_merge_request_stuck`.
    Fires at most ONCE while an L1 escalation is open for
    :data:`_RESOURCE_AUDIT_SENTINEL`. Callers (see
    ``SpeculativeMergeWorker._check_resource_audit``) only invoke this after
    *violations* has persisted across
    ``SpeculativeMergeWorker.RESOURCE_AUDIT_ESCALATION_STREAK`` consecutive
    heartbeats, so a transient/racy leak (see
    ``worktree_ledger_violations``'s grace-window docstring) that
    self-resolves between heartbeats never trips an alarm.

    This is OBSERVATION + ESCALATION only — it never mutates queue/inflight/
    worktree state or halts the pipeline (PRD design decision 4: invariants
    escalate loudly, degrade never).

    * ``level=1`` (L1 blocking).
    * ``category='merge_resource_leak'``
    * ``task_id=_RESOURCE_AUDIT_SENTINEL`` (fixed — one audit per worker, not
      per-request).

    None-safe: returns immediately when *escalation_queue* is None.
    Dedup: returns immediately when an open L1 already exists for the sentinel.

    Args:
        escalation_queue: Live escalation queue or ``None``.
        violations: Non-empty list[str] — the combined output of
            ``SpeculativeMergeWorker.speculation_accounting_violations()``
            and ``SpeculativeMergeWorker.worktree_ledger_violations()``.
        event_store: Optional event store; when provided an
            ``EventType.escalation_created`` event is emitted.
    """
    if escalation_queue is None:
        return

    sentinel = _RESOURCE_AUDIT_SENTINEL

    # Dedup: don't re-alarm while an open L1 already exists.
    if escalation_queue.has_open_l1(sentinel):
        return

    from escalation.models import Escalation  # local import — escalation optional dep

    count = len(violations)
    headline = violations[0] if violations else 'unknown violation'
    summary = (
        f'Merge queue resource-conservation audit found {count} '
        f'violation(s) persisting across consecutive heartbeats: {headline!r}'
        + (' (+ more — see detail)' if count > 1 else '')
    )
    detail = (
        'The following resource-conservation invariants (I4 speculation '
        'permits / merge-ahead caps, I6 worktree ledger) have been violated '
        'for multiple consecutive heartbeats:\n\n'
        + '\n'.join(f'- {v}' for v in violations)
        + '\n\n'
        'The orchestrator has NOT halted or mutated any pipeline state — '
        'this is observation-only (PRD design decision 4).'
    )

    esc = Escalation(
        id=escalation_queue.make_id(sentinel),
        task_id=sentinel,
        agent_role='orchestrator-merge-resource-monitor',
        severity='blocking',
        level=1,
        category='merge_resource_leak',
        summary=summary,
        detail=detail,
        suggested_action=(
            "Inspect the merge worker's snapshot()['resource_audit'] key "
            '(speculation_accounting / worktree_ledger sub-lists) to '
            'identify the leaked permit, cap, or worktree; fix the code '
            'path that failed to release it.'
        ),
    )
    escalation_queue.submit(esc)

    if event_store is not None:
        from orchestrator.event_store import EventType

        event_store.emit(
            EventType.escalation_created,
            data={
                'violations': list(violations),
                'count': count,
            },
        )
