"""Git worktree and merge operations.

.task/ contamination — now structural, not guard-defended (W11 θ/ι)
===================================================================
The .task/ directory is an ephemeral scratch space for orchestrator agents.
It used to leak onto main via worktree inheritance, `git add -A`, and merge
commits, so this module carried a "belts and braces" scrub layer on top of
it: scrub_task_dir_from_tree(), called from create_worktree(), acquire(),
merge_to_main(), and advance_main()'s retry loop, plus a post-staging
unstage net in commit().

That scrub layer has been removed.  .task/ execution metadata now lives
OUTSIDE the git worktree entirely, at <worktree_base>/.task-meta/<name>
(see TaskArtifacts.meta_root_for()) — so .task/ contamination of the git
tree is structurally impossible rather than merely guarded against:
nothing ever writes task metadata into a path git tracks.

Both migration-window safeguards have since been dropped too (W11 ι):

- _assert_no_task_dir() — the cheap tripwire that hard-asserted a given
  commit SHA carried no .task/ entries before advance_main() — is gone.
  Contamination is structural, so the tripwire had nothing left to catch.
  This removes the last commit-time checkpoint before main; that is an
  intentional defense-in-depth trade-off (not an oversight) now that
  relocation makes contamination structurally impossible rather than
  merely guarded against — see task 2262 (W11 ι) design decisions.
- _ensure_task_gitignore() — which used to write .task/.gitignore so any
  leftover <worktree>/.task/ scratch directory self-ignored under `git add
  -A` / `git status` — is gone too: the orchestrator hot path no longer
  writes task metadata under <worktree>/.task (it lives under .task-meta
  instead, see TaskArtifacts.meta_root_for()), so there was nothing left
  for a nested .gitignore to defend. A few non-hot-path callers still
  construct TaskArtifacts with meta_root=None (e.g. cli.py's eval flow)
  and do write under <worktree>/.task; that residual directory remains
  untracked via this repo's root .gitignore .task/ entry, independently
  of the removed nested one.

No .task-specific guards remain in this module.
"""

import asyncio
import contextlib
import fcntl
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Iterable,
    Mapping,
)
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypedDict

from shared.proc_group import (
    reap_process_groups,
    scan_process_groups_under_path,
    snapshot_process_group,
)
from shared.transcript_archive import archive_task_transcripts

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import TASK_META_DIRNAME, GitConfig, TranscriptArchiveConfig
from orchestrator.lane_lifecycle import (
    ACQUIRE_ROUTE_TRANSITIONS,
    LANE_STATE_DIRNAME,
    POOL_ROOT_SENTINEL,  # noqa: F401  re-export shim (test_pool_storage_guard.py)
    AcquireRoute,
    LaneLifecycle,
    LaneState,
)
from orchestrator.verify_cancel import (
    acquire_merge_verify_flock,
    lane_lock_holder_pids,
    lane_lock_path,
    read_lock_holder_pgid,
    release_merge_verify_flock,
    remove_lock_holder_pgid,
    write_lock_holder_pgid,
)
from orchestrator.warm_lane_pool import WarmLanePoolCensus
from orchestrator.worktree_identity import identities_match, read_worktree_title

logger = logging.getLogger(__name__)

# Return type for advance_main — lets callers distinguish transient
# (CAS) failures from permanent ones (not-a-descendant, contamination).
AdvanceResult = Literal[
    'advanced', 'cas_failed', 'not_descendant', 'contaminated',
    'stash_failed', 'wip_overlap', 'pop_conflict',
    'unmerged_state', 'pop_conflict_no_advance',
    'rebased_pending_reverify', 'conflict_markers',
]


@dataclass(frozen=True)
class AdvanceOutcome:
    """Value object returned by :meth:`GitOps.advance_main`.

    result: the :data:`AdvanceResult` code — same literal values and retry
        semantics as before; this is BEHAVIOR-PRESERVING, only the carrier
        changed from a bare literal to a field.
    advanced_sha: the SHA actually placed on main (or parked, for
        ``'rebased_pending_reverify'``).  Populated on ``'advanced'``,
        ``'pop_conflict'``, and ``'rebased_pending_reverify'``; ``None`` on
        every other (failure) result.
    rebased_from: the original base SHA the caller expected main to be at
        (``expected_main``), populated only on ``'rebased_pending_reverify'``.
    rebased_onto: the current main SHA the merge commit was rebased onto,
        populated only on ``'rebased_pending_reverify'``.
    """
    result: AdvanceResult
    advanced_sha: str | None = None
    rebased_from: str | None = None
    rebased_onto: str | None = None


PushResult = Literal['pushed', 'noop', 'rejected', 'error']


# Return type for recover_red_main — distinguishes a successful CAS move
# from an atomic abort (another writer beat us to the ref).
RecoverResult = Literal['rewound', 'cas_failed', 'error']


# Return type for remove_merge_worktree_guarded — the C1 lease-enforced
# removal primitive (PRD merge-worktree-lifecycle-integrity.md, task alpha).
# 'removed'            — uncontended: the flock was free, the path existed,
#                         and `git worktree remove --force` succeeded.
# 'skipped_lease_held' — a LIVE merge-verify holds the tree's flock; removal
#                         is skipped (never deferred/retried — true leaks are
#                         collected later by the merge reaper's age-grace
#                         sweep). A dead/stale holder never produces this
#                         outcome: the kernel auto-releases a crashed
#                         holder's flock, so the non-blocking acquire simply
#                         succeeds and removal proceeds (fail-open).
# 'skipped_persistent' — path resolves to persistent_merge_worktree_path or
#                         persistent_offline_deep_worktree_path; persistent
#                         lanes are never removed regardless of lease.
# 'not_present'        — the flock was acquired but the path no longer
#                         exists.
# 'failed'              — the flock was acquired and the path existed, but
#                         `git worktree remove --force` itself returned
#                         non-zero.
RemovalOutcome = Literal[
    'removed', 'skipped_lease_held', 'skipped_persistent', 'not_present', 'failed',
]


# Single source of truth for the WIP safety-commit subject prefix produced by
# _inter_iteration_rebase (workflow.py), and the requeue-rebase /
# warm-lane-reclaim paths below (commit() call sites in this module). Any
# code that mints a new "save WIP before X" safety-commit must share this
# prefix so is_wip_safety_commit() (and therefore TaskWorkflow.
# _detect_tip_wip_commits) keeps recognizing it.
WIP_SAFETY_COMMIT_PREFIXES = ('chore: save WIP before ',)


def is_wip_safety_commit(subject: str) -> bool:
    """Return True if ``subject`` is one of the harness's WIP safety-commits.

    These are auto-commits the harness makes to snapshot uncommitted work
    before a rebase/requeue/reclaim operation. They can land a still-pending
    plan step's complete implementation at branch HEAD before mark_step_done
    is called for that step — see TaskWorkflow._detect_tip_wip_commits.
    """
    return subject.strip().startswith(WIP_SAFETY_COMMIT_PREFIXES)


# Single source of truth for the private ref advance_main uses to park
# pre-advance WIP (task 2556).  Exclusively owned by the merge worker —
# never the shared refs/stash stack, which a human or other session in
# project_root can race (incident 13674d3c68: the worker popped/dropped a
# human's stash).  See GitOps._park_wip_on_private_ref and
# _safe_restore_park_with_recovery.
MERGE_PARK_REF = 'refs/dark-factory/merge-park'


class MergeParkError(Exception):
    """Base class for failures parking pre-advance WIP on MERGE_PARK_REF.

    Raised by :meth:`GitOps._park_wip_on_private_ref` when the ``git stash
    create`` / ``git update-ref`` infra sequence itself fails (not a
    contention condition — see :class:`MergeParkContentionError` for that).
    ``advance_main`` catches this and returns the existing
    ``AdvanceResult 'stash_failed'`` code (loud CRITICAL log + permanent
    halt to prevent code loss).
    """


class MergeParkContentionError(MergeParkError):
    """Raised when MERGE_PARK_REF already exists at park time.

    The merge worker is serialized, so a stale ref here is either an
    invariant violation or a crash-leftover holding real, unrecovered WIP.
    Never overwritten — :meth:`GitOps._park_wip_on_private_ref` raises this
    instead so the caller can halt loudly rather than silently destroying
    the preserved work.
    """


class TrainMembership(TypedDict, total=False):
    """Train metadata passed from task.metadata.train.

    All keys are optional at the type level; _train_predecessor validates
    presence of required keys at runtime with diagnostic error messages.
    """
    id: str
    order: int
    members: list[str] | None


@dataclass(frozen=True)
class TrainPredecessor:
    """Resolved predecessor for a train member with order > 0."""
    task_id: str
    branch: str


@dataclass(frozen=True)
class TrainStackResult:
    """Result of stack_train_branches: which members survived and which were ejected.

    survivors: member ids that were successfully rebased into the linear stack
               (or are the anchor, which is always the base).
    ejected:   member ids that conflicted during stacking and were dropped;
               their branches are left clean (rebase aborted) so they can
               merge solo.
    """
    survivors: list[str]
    ejected: list[str]


# Default commit-citation pattern for ``find_task_citation_commit``.
#
# Matches dark-factory / reify conventions on main:
#   - Conventional-commit subjects that cite the task id in parens or
#     after a colon: ``impl(50): xyz`` / ``fix(50): xyz`` / ``test(50: ...)``.
#   - Subjects that mention the task branch directly: ``... task/50 ...``
#     anywhere in the subject line.
#   - The canonical no-ff merge subject ``Merge task/50 into <main>`` produced
#     by ``merge_to_main``.
#   - Prefix-independent, unanchored paren citations anywhere in the
#     subject (task 2870): ``(50)`` and ``(#50)`` (via ``\(#?{tid}\)``) and
#     ``(task 50)`` (via ``\(task {tid}\)``). These are NOT tied to the
#     ``^(prefix)`` conventional-commit head, so they also rescue
#     otherwise-unmatched subjects such as a ``resolve:``-prefixed commit
#     that cites the task in parens — closing a latent false ``no_citation``
#     that stranded genuinely-landed tasks using one of these forms.
#
# The ``{tid}`` placeholder is interpolated via ``str.format`` with the
# escaped task id; a ``\b`` (word boundary) at each side of the
# conventional-commit / ``task/{tid}`` alternatives blocks substring overlap
# so ``task/3399`` doesn't match a row that cites ``task/339``. The paren
# alternatives instead use the literal ``\(``...``\)`` as an exact numeric
# boundary (no ``\b`` needed): ``(1175)`` never matches tid ``117`` or
# ``11750``. ``#`` and the space in ``(task 50)`` are literals in both
# engines, so the widened pattern stays valid as ERE and as Python ``re``.
#
# Collision assumption (task 2870, esc-5252-9): the unanchored ``(#{tid})``
# and bare ``({tid})`` alternatives assume a parenthesized number in a
# subject on main is ALWAYS a task citation in THIS repo — never a GitHub
# PR number, squash-merge suffix, or trailing issue number. This holds
# because dark-factory's merge worker writes the canonical
# ``Merge task/{tid} into <main>`` subject (see ``merge_to_main``), not a
# GitHub-style ``(#PR)`` squash suffix, and the repo does not take PR
# merges. The assumption matters because relaxing the FIX-2 bidirectional
# lineage guard (task 2870) left the FIX-1' effect-present check as the SOLE
# attribution gate: a landed but unrelated commit whose subject happened to
# carry ``(1175)`` for a non-task reason WOULD be attributed to task 1175
# (its effect genuinely is on main). That collision is accepted as a
# deliberate tradeoff given the convention above. If this repo ever begins
# emitting ``(#N)``/``(N)`` where N is a non-task number (e.g. a PR id),
# narrow these two alternatives to a citation-prefix-qualified form (or drop
# the bare-number arm) to restore disambiguation.
#
# Subject-only (task 2675 FIX 2): this pattern is written with ``^``
# anchors as if applied to a single SUBJECT line, but git's ``--grep``
# applies ``^``/``$`` per LINE across the whole commit message — a BODY
# line that happens to start with a conventional-commit token, or with
# ``Merge task/{tid} into ``, or that carries a bare paren citation like
# ``(#50)``, would otherwise false-cite.
# ``find_task_citation_commit`` uses ``--grep`` only as a coarse
# full-message PRE-filter (a sound superset) and then re-applies this
# same pattern string, compiled as a Python ``re`` (no ``re.MULTILINE``,
# so ``^`` anchors to the start of the string), to each candidate's
# SUBJECT ONLY — body-only matches (including via the unanchored paren
# alternatives) are therefore never treated as citations. A caller-supplied
# ``pattern_template`` override must therefore be valid as both a git
# ``--extended-regexp`` (ERE) pattern *and* a Python ``re`` pattern.
DEFAULT_COMMIT_CITATION_PATTERN: str = (
    r'^(merge|impl|amend|fix|test|feat|chore|docs|refactor|style|build)'
    r'(\(\b{tid}\b[):]|.*\btask/{tid}\b)'
    r'|^Merge task/{tid} into '
    r'|\(#?{tid}\)'
    r'|\(task {tid}\)'
)

# Fixed name for the persistent warm merge-verify worktree (task 1692).
# Lives at <worktree_base>/_merge-verify.  Excluded from prune and
# find_inflight enumeration (see _iter_merge_worktrees).
PERSISTENT_MERGE_WORKTREE_NAME: str = '_merge-verify'

# Bounded-wait timeout (seconds) for GitOps.merge_verify_lease()'s
# acquire_merge_verify_flock() call (task 2315, BUG 1; raised 10s -> 300s in
# task 2828). The lease now RAISES MergeVerifyLeaseContended (deferring the
# dispatch) instead of yielding unprotected on timeout, so this is the window
# a starting verify patiently blocks for the lane lock before it gives up and
# requeues. 300s (5 min) is <0.5% overhead on a 1--2h verify, yet comfortably
# outlasts every SHORT legitimate holder (a reseed/thin/gc/reset — seconds to
# low minutes) so they never force a needless requeue, while a genuinely long
# contender (another 1--2h verify) times out and correctly defers. The acquire
# runs OFF the event loop via asyncio.to_thread (see merge_verify_lease), so a
# minutes-long poll never freezes the orchestrator.
#
# A LOCAL constant rather than importing cli.MERGE_VERIFY_FLOCK_WAIT_SECS:
# cli.py imports GitOps, so a git_ops -> cli import would be architecturally
# backwards. Distinct from cli.py's own env-overridable
# MERGE_VERIFY_FLOCK_WAIT_SECS (the laptop host lane — task 2828 out of scope).
# A plain module constant (no config knob, no env override), monkeypatchable
# in tests via the module global, mirroring the sibling
# _SEED_WARM_LANE_LOCK_WAIT_SECS below.
_MERGE_VERIFY_LEASE_WAIT_SECS: float = 300.0

# Bounded-wait timeout (seconds) for GitOps.task_verify_lease()'s
# acquire_merge_verify_flock() call — the warm-lane consumer-hold (task 3027).
# task_verify_lease holds the SHARED <lane_dir>.lock across a task-lane verify
# so a concurrent reify warm-lane-gc.sh reclaim's per-lane `flock -n` refuses
# (reify task 5354, the paired acquire-time guard), preventing an in-flight
# nextest's test binaries from being reclaimed out from under it (esc-5236-7 /
# esc-5275-10). A SEPARATE constant from _MERGE_VERIFY_LEASE_WAIT_SECS keeps the
# task-lane wait independently tunable. Unlike the merge lease this one fails
# OPEN on timeout (WARNING + proceed unprotected, see task_verify_lease), so
# this window is only how long a task verify patiently waits for a racing
# reseed to finish before proceeding anyway — a modest 300s (5 min) comfortably
# outlasts every SHORT legitimate holder (a reseed/thin/gc — seconds to low
# minutes) so the common case takes the hold, while never blocking the verify.
# The acquire runs OFF the event loop via asyncio.to_thread (see
# task_verify_lease). A plain module constant (no config knob, no env override),
# monkeypatchable in tests via the module global, mirroring the sibling
# _MERGE_VERIFY_LEASE_WAIT_SECS above.
_TASK_VERIFY_LEASE_WAIT_SECS: float = 300.0

# Bounded-wait timeout (seconds) for GitOps._seed_warm_lane()'s outer
# <lane_dir>.lock flock -x (task 2599 amendment). Seeding runs on the
# latency-sensitive warm-lane acquisition hot path; a plain unbounded
# blocking flock -x would stall it indefinitely against a live-but-wedged
# holder (thin's rm -rf, GC reclaim, or another seed's cp --reflink) since
# flock only auto-releases a lock on holder *death*, never on a
# stuck-but-live process. All legitimate holders are bounded operations, so
# 30s gives generous headroom over normal (even slow-disk) durations while
# still turning a wedged holder into a diagnosable, bounded failure instead
# of a silent, unbounded one. A plain module constant, no config knob and no
# env override — mirrors _MERGE_VERIFY_LEASE_WAIT_SECS's own
# independently-defaulted, non-overridable copy above, and keeps this fix
# inside git_ops.py rather than reaching into config.py's green/red
# reload-tier surface for what is a narrow, self-contained safety margin.
#
# Declared `int` (task 2599 amendment), not `float`: the value is passed to
# the `flock(1)` CLI as `str(_SEED_WARM_LANE_LOCK_WAIT_SECS)` for `-w`, and
# an int stringifies to an unambiguous whole-second literal ("30") rather
# than a fractional one ("30.0"). Fractional `-w` parses fine on current
# util-linux, but this is the only place in git_ops.py that hands a
# CLI-parsed wait value to `flock` itself (`_MERGE_VERIFY_LEASE_WAIT_SECS`
# is only ever passed to an in-process helper, never a CLI arg) — no reason
# to lean on fractional-timeout CLI parsing for what is a whole-second value.
_SEED_WARM_LANE_LOCK_WAIT_SECS: int = 30

# Bounded-wait timeout (seconds) for GitOps.reset_persistent_merge_worktree()'s
# own <lane_dir>.lock acquire (task 3003).  SPLIT OUT of the shared
# _SEED_WARM_LANE_LOCK_WAIT_SECS above at the SAME value (30) — the split is
# the point, not a retune: the reset previously borrowed the seed's constant,
# which coupled two unrelated call sites with opposite tuning pressures.
#
# Why 30 here and NOT task 2828's 300s (_MERGE_VERIFY_LEASE_WAIT_SECS above):
#   1. 2828 chose 300s so a SHORT legitimate holder (a reseed/thin/gc —
#      seconds to low minutes) never forces a needless requeue.  That argument
#      barely applies on this path: the observed holder class is a 1--2h
#      verify or a speculative merge-ahead train, which outlasts 30s and 300s
#      identically.  No bounded wait short of HOURS changes the outcome for
#      the incident this constant was split for — the CLASSIFICATION of the
#      timeout (MergeVerifyLeaseContended -> DEFER, below) is the fix, not the
#      length of the wait.
#   2. Raising the seed constant in place would have been actively harmful:
#      _seed_warm_lane stringifies it straight into `flock(1) -w`
#      (str(_SEED_WARM_LANE_LOCK_WAIT_SECS), hence its `int` declaration) on
#      the latency-sensitive warm-lane ACQUISITION hot path.
#   3. This same acquire also backs the laptop `verify-merge` CLI via
#      acquire_host_verify_worktree, where cli.py treats the timeout as a
#      TERMINAL bail ("verify exits without ever building") rather than a
#      requeue — 300s there would buy up to 5 min of dead laptop wall-clock
#      per contended run for no benefit.
# Declared `int` to mirror the seed constant's whole-second convention; unlike
# it, this value is never handed to a CLI, only to the in-process
# acquire_merge_verify_flock helper.  A plain module constant (no config knob,
# no env override), monkeypatchable in tests via the module global.
_RESET_WARM_LANE_LOCK_WAIT_SECS: int = 30

# flock's --conflict-exit-code (-E) for _SEED_WARM_LANE_LOCK_WAIT_SECS above.
# Deliberately mirrors timeout(1)'s well-known 124 "command timed out"
# convention so the sentinel is self-documenting in logs, and is chosen
# distinct from every other rc _seed_warm_lane's docstring documents (0
# success, 75 disk-pressure, 127 absent-script/exception sentinel; any other
# value is a generic script fault). A genuine seed-warm-lane.sh exit code of
# 124 would be misattributed to a lock-wait timeout, but no script
# convention in this codebase uses 124 for anything else.
_SEED_WARM_LANE_LOCK_TIMEOUT_RC: int = 124

# The reify seed-warm-lane.sh opt-out flag a caller passes to assert it ALREADY
# holds ${LANE_DIR}.lock, so seed skips its own acquire instead of self-refusing
# against that lock (flock is not re-entrant across a process tree).  See
# :meth:`GitOps._seed_warm_lane` for why this is load-bearing (reify 5556).
_SEED_ASSUME_LANE_LOCK_HELD_FLAG = '--assume-lane-lock-held'


# ── warm-lane script resolution (task 3072, PRD leaf α) ───────────────────────
#
# dark-factory ships its own copies of the project-agnostic warm-lane scripts
# under orchestrator/scripts/warm-lane/ so a project that carries no warm-lane
# tooling still gets GC, disk guarding, thinning and auditing of its lane pool.
#
# Resolved repo-relative from this file (PRD open question 1 / design decision
# D3): orchestrator/pyproject.toml packages only ``src/orchestrator`` in the
# wheel and the deployed orchestrator runs ``uv run --project orchestrator``
# from a checkout, so a repo-relative walk is both sufficient and the smallest
# change — making the scripts package data would need a build-backend change
# that buys nothing for the only deployment mode in use.  Same idiom and same
# depth as workflow.py's ``_ORCH_PROJECT_DIR``, so it survives worktrees and
# CWD changes identically.  If dark-factory is ever installed as a wheel this
# path simply will not exist, and resolution then fails LOUDLY through
# :meth:`GitOps._resolve_warm_lane_script`'s both-paths WARNING at the call
# sites rather than silently — exactly the migration landmine leaf α exists to
# remove.
_DF_WARM_LANE_SCRIPT_DIR: Path = (
    Path(__file__).resolve().parents[2] / 'scripts' / 'warm-lane'
)

#: Test-only override for :data:`_DF_WARM_LANE_SCRIPT_DIR`.  Production NEVER
#: sets this: resolution is repo-relative.  It exists because ~200 existing
#: tests build a synthetic tmp_path repo and assert the "script absent →
#: fail-soft sentinel" path; without a seam an unconditional repo-relative
#: fallback would make every one of them execute the REAL warm-lane scripts
#: (rm -rf on lane dirs, flock acquisition, df probes) against tmp_path.  The
#: autouse ``_isolate_warm_lane_script_dir`` fixture in tests/conftest.py pins
#: it at a guaranteed-absent directory suite-wide.
_DF_WARM_LANE_SCRIPT_DIR_ENV = 'ORCH_WARM_LANE_SCRIPT_DIR'


def _df_warm_lane_script_dir() -> Path:
    """Return the directory holding dark-factory's own warm-lane scripts.

    Reads :data:`_DF_WARM_LANE_SCRIPT_DIR_ENV` at CALL time (not import time)
    so ``monkeypatch.setenv`` is effective without a module reload; falls back
    to the repo-relative :data:`_DF_WARM_LANE_SCRIPT_DIR`, which is what
    production always uses.
    """
    override = os.environ.get(_DF_WARM_LANE_SCRIPT_DIR_ENV)
    if override:
        return Path(override)
    return _DF_WARM_LANE_SCRIPT_DIR


@functools.lru_cache(maxsize=256)
def _seed_script_supports_assume_lane_lock_held(script: Path) -> bool:
    """Does this lane's ``seed-warm-lane.sh`` accept ``--assume-lane-lock-held``?

    The seed script is read from the LANE's own checkout, so its vintage varies
    per lane: a lane sitting on a pre-reify-5354 base predates the flag and
    would reject it as a usage error (exit 2), converting a working seed into a
    hard fault.  Probing the script text is the cheapest reliable capability
    check — the flag string appears in the arg parser of every version that
    supports it, and in none that don't.

    Fails CLOSED (``False``) on any read error: omitting the flag restores the
    pre-5354 behaviour, in which the script never takes the lane lock itself,
    so a false negative is never worse than not having this fix at all.

    Cached per resolved path — lane scripts change only on reseed, and a wrong
    cached answer degrades to the same safe fallback.
    """
    try:
        return _SEED_ASSUME_LANE_LOCK_HELD_FLAG in script.read_text(
            encoding='utf-8', errors='replace',
        )
    except OSError:
        logger.debug(
            '_seed_script_supports_assume_lane_lock_held: unreadable %s — '
            'assuming unsupported', script, exc_info=True,
        )
        return False


# Short window (seconds) over which the θ soft-floor defer path memoizes the
# α warm-lane-audit (:meth:`GitOps._warm_lane_audit_cached`).  α is
# observability-only (inv.12), so a slightly-stale HEADROOM in the defer
# WARNING is acceptable; the memo exists so a SUSTAINED soft-pressure
# condition — which requeues the same fresh allocation across many dispatch
# cycles — does not re-fork the audit subprocess on every cycle (amendment,
# reviewer_comprehensive performance-efficiency).  Read as a module global
# (not a default arg) so tests can monkeypatch it (0.0 disables the memo).
_WARM_LANE_AUDIT_CACHE_TTL_SECS: float = 30.0

# Re-fire cadence for the structural-exhaustion loudness callback once the
# pool-GLOBAL consecutive-EXHAUSTED counter is at-or-above threshold (task 2988,
# review amendment — efficiency).  The callback fires on the EXACT threshold
# crossing and then only every _STRUCTURAL_EXHAUSTION_L2_REFIRE_EVERY-th
# subsequent consecutive EXHAUSTED — NOT on every acquire.  Rationale: the
# harness filer's dedup scan (find_pending_l2_by_root_cause) is
# O(pending-escalations) and runs on the acquire chokepoint; unlike the
# WarmLanePool drift-counter (self-correcting: resets on every successful round)
# the consecutive-EXHAUSTED counter NEVER resets while the pool stays stuck (a
# structurally exhausted pool serves no fresh lane), so firing on every trip
# would run that scan on every acquire until a lane frees.  The periodic re-fire
# keeps the born-at-L2 recoverable — it re-files if an operator resolves the L2
# while the pool remains structurally exhausted — without a per-acquire scan.
# A fixed interval (not `count % threshold`) so a sensitive threshold=1 config
# still rate-limits instead of firing every trip.
_STRUCTURAL_EXHAUSTION_L2_REFIRE_EVERY: int = 50

# Fixed name for the SECOND persistent warm worktree (task 1952, PRD δ /
# §5 C5), dedicated to the offline-deep lane worker (β2).  Lives at
# <worktree_base>/_offline-deep.  Deliberately NOT prefixed `_merge-`, so it
# is structurally exempt from _iter_merge_worktrees / prune_stale_merge_worktrees
# / find_inflight_merge_worktree — the same mechanism that already exempts
# _spec-*/_lane-*/_solo-* worktrees — with no explicit skip required.
PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME: str = '_offline-deep'

# Sentinel filename marking worktree_base as backed by live pool storage
# (task 2099).  Lives ON the pool storage itself (plain dir or real mount
# alike — substrate-independent, no config knob), so it disappears along
# with an unmounted mountpoint even though the mountpoint DIR still exists.
# See GitOps.pool_storage_present() / mark_pool_storage_present().
#
# Folded into orchestrator.lane_lifecycle (W11 gamma sentinel fold): the
# sentinel FS read/write now live ONLY there (LaneLifecycle.
# pool_storage_present() / mark_pool_storage_present()), and this name is a
# re-export so `from orchestrator.git_ops import POOL_ROOT_SENTINEL` (and
# the literal '.pool-root') keep working for existing callers/tests.

# The _iact-* band (worktree_base/<iact_prefix><slug>, config.iact_prefix
# default '_iact-') minted by GitOps.create_interactive_worktree is
# invariantly disjoint from the _lane-* warm_lane_pool band and the _spec-*
# spec_warm_lane_pool band: create_interactive_worktree and its cap
# enumeration never read or mutate either pool, and the pools never
# enumerate, acquire, or release an _iact-* directory (isolation invariant
# I1). See InteractiveWorktreeInfo / InteractiveWorktreeLimitError /
# GitOps.create_interactive_worktree below for the full contract.

class WorktreeKind(Enum):
    """Ephemeral main-tip probe/sweep worktree kinds minted by
    :meth:`GitOps.ephemeral_worktree` (task θ, verify-plan PRD).

    Each member's value IS both the directory-name prefix
    ``ephemeral_worktree`` mints under ``worktree_base`` (e.g.
    ``_mainprobe-<hex>``) AND the :data:`PROTECTED_PREFIXES` registry key
    that keeps the reaper from ever reclaiming it mid-run — making the enum
    the single source of truth for both so naming and protection cannot
    drift apart.
    """

    MAIN_PROBE = '_mainprobe-'
    MAIN_SWEEP = '_mainsweep-'


# Band-ownership registry for worktree_base's ephemeral-worktree namespace
# (gitops-chokepoints PRD, Mechanism 3 / task ε).  Maps a band TOKEN to an
# owner tag identifying the subsystem that mints/reaps it.  A key ending in
# '-' is a PREFIX (matched via str.startswith); a key not ending in '-' is
# an EXACT worktree name (matched via ==).  Consulted by
# GitOps._refuse_foreign_band (via GitOps.protected_prefixes(), which also
# merges in the config-driven _iact-* band) so a destructive cleanup sweep
# can never remove a band it does not own — see that method's docstring for
# the full contract.  Exact-name keys use the persistent-name constants
# above (not independent literals) so this registry cannot drift from them.
PROTECTED_PREFIXES: dict[str, str] = {
    '_lane-': 'warm-lane-pool',
    '_spec-': 'merge-speculation-pool',
    '_merge-': 'merge-queue',
    '_solo-': 'attribution-solo',
    '_substrate-gate-': 'harness-substrate-gate',
    PERSISTENT_MERGE_WORKTREE_NAME: 'persistent-merge-verify',
    PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME: 'persistent-offline-deep',
    LANE_STATE_DIRNAME: 'warm-lane-lifecycle',
    TASK_META_DIRNAME: 'task-artifacts',
    # Ephemeral verify-probe bands minted by GitOps.ephemeral_worktree() —
    # keyed by WorktreeKind.*.value so this registry cannot drift from the
    # CM's own naming (task θ).
    WorktreeKind.MAIN_PROBE.value: 'verify-main-probe',
    WorktreeKind.MAIN_SWEEP.value: 'verify-main-sweep',
}


#: Bands a warm-lane POOL SWEEP owns, and must therefore never be handed as
#: protected.  Excluding them from a rendered protect glob is LOAD-BEARING, not
#: cosmetic: ``warm-lane-gc.sh``'s whole job is reclaiming ``_lane-*``/
#: ``_spec-*`` entries, so a naive render of :data:`PROTECTED_PREFIXES` would
#: make it skip every pool lane in both passes — reclaim would stop entirely and
#: the pool would accrete straight back to the 2026-07-10 ENOSPC outage the
#: sweep exists to prevent.  Named explicitly, in the same band-ownership
#: vocabulary :meth:`GitOps._refuse_foreign_band` already takes an ``owned``
#: argument in, so the Python guard and the bash consumer agree on what
#: "owned" means.
PROTECT_GLOB_OWNED_POOL_BANDS: frozenset[str] = frozenset({'_lane-', '_spec-'})


def default_protected_prefixes(iact_prefix: str | None = None) -> dict[str, str]:
    """The static band registry merged with ONE interactive band.

    :data:`PROTECTED_PREFIXES` alone is not the authoritative band map: the
    ``_iact-*`` band is config-shaped (:attr:`GitConfig.iact_prefix`), so it
    lives outside the constant.  Called with no argument this is the
    process-wide DEFAULT view, for callers with no :class:`GitOps` instance to
    ask; :meth:`GitOps.protected_prefixes` passes its own instance's prefix, so
    the registry + iact-band merge exists in exactly one place.

    Args:
        iact_prefix: The interactive band token to merge in.  ``None`` means
            :attr:`GitConfig.iact_prefix`'s field default.  Note this REPLACES
            the band rather than adding to it: a deployment that renamed its
            interactive band must not also get the default ``_iact-`` treated
            as protected, or :meth:`GitOps._refuse_foreign_band` would guard a
            band that deployment never mints.

    Returns a fresh dict; mutating it cannot affect the module registry.

    The bash bridge (``lane_protect_glob`` in
    ``orchestrator/scripts/warm-lane/lib_lane_state.sh``) has no
    :class:`GitOps` instance to ask, so it passes the deployment's band through
    the ``REIFY_WARM_LANE_IACT_PREFIX`` environment variable.  Leaving it unset
    there means the same thing as ``None`` here — the field default — which is
    correct only for a deployment that did not rename its band.
    """
    # Declared str, not a rebind of the str|None parameter: FieldInfo.default is
    # typed Any, so assigning it back into `iact_prefix` widens the rendered key
    # type to `str | None` and the return type stops matching dict[str, str].
    # GitConfig.iact_prefix is a required-typed `str` field, so its field default
    # is always a str.
    band: str = (
        iact_prefix
        if iact_prefix is not None
        else GitConfig.model_fields['iact_prefix'].default
    )
    return {**PROTECTED_PREFIXES, band: 'interactive'}


def render_protect_glob(
    prefixes: Mapping[str, str] | None = None,
    *,
    owned: Iterable[str] = (),
) -> str:
    """Render a band map as the comma-separated protect glob a sweep consumes.

    Pure — no I/O, no config read beyond the default band map.  Applies the two
    key semantics :data:`PROTECTED_PREFIXES` already documents, rather than
    inventing a third: a key ending in ``-`` is a PREFIX and renders
    ``<key>*``; a key not ending in ``-`` is an EXACT worktree name and renders
    verbatim.  Both directions matter — rendering an exact name as a glob
    silently WIDENS the protected set, and rendering a prefix verbatim silently
    NARROWS it, which is the direction that gets a live managed worktree
    reclaimed.

    Args:
        prefixes: The band map to render.  ``None`` means
            :func:`default_protected_prefixes`.
        owned: Band tokens the calling sweep OWNS, excluded from the output.
            Pass :data:`PROTECT_GLOB_OWNED_POOL_BANDS` for a warm-lane pool
            sweep; see that constant for why omitting them is load-bearing.

    Returns:
        The bands, comma-joined in the map's own iteration order (deterministic,
        and what an operator diffs against a previous default).

    The bash consumer is ``lane_protect_glob`` in
    ``orchestrator/scripts/warm-lane/lib_lane_state.sh``.  Its static fallback,
    ``LANE_PROTECT_GLOB_FALLBACK``, is the one artifact that CAN drift from this
    registry, and the INV-5 gate for that drift is
    ``orchestrator/tests/test_lane_state_lib.py::TestProtectGlobFallbackDrift``.
    """
    mapping = default_protected_prefixes() if prefixes is None else prefixes
    owned_set = frozenset(owned)
    return ','.join(
        f'{key}*' if key.endswith('-') else key
        for key in mapping
        if key not in owned_set
    )


# Positive-match namespace classifier for a worktree_base entry name (C2).
WorktreeClass = Literal['task', 'merge', 'infra']


def classify_worktree_entry(name: str) -> WorktreeClass:
    """Classify a ``worktree_base`` directory-entry name by namespace (C2).

    PRD: docs/prds/merge-worktree-lifecycle-integrity.md, task beta (§4
    Contract C2 — namespace invariant).

    This is the POSITIVE-match complement of :data:`PROTECTED_PREFIXES`
    (and reify's ``warm-lane-gc`` PROTECT_GLOB): rather than a NEGATIVE,
    hand-maintained per-name exclusion list — the whack-a-mole the
    2026-07-22 task/5326 incident proved unmaintainable, when the
    crash-recovery sweep force-removed the persistent ``_merge-verify`` 21s
    after the same process dispatched a verify into it — it encodes ONE
    invariant keyed on the ``_``/``.`` name prefix that every current AND
    future infra band already obeys (see PROTECTED_PREFIXES: ``_lane-``,
    ``_spec-``, ``_merge-``, ``_solo-``, ``_substrate-gate-``,
    ``_mainprobe-``, ``_mainsweep-``, ``_offline-deep``, ``.lane-state``,
    ``.task-meta``):

        * ``_merge-``-prefixed          => ``'merge'`` — the merge-queue band.
          The crash-recovery sweep / orphan reaper only SKIP + REPORT these
          to the merge reaper (``_reap_orphaned_merge_worktrees``, which
          owns their guarded readopt/age-grace disposition via
          :meth:`GitOps.remove_merge_worktree_guarded`); they NEVER remove a
          ``_merge-*`` directly.
        * any other ``_``/``.``-prefixed => ``'infra'`` — an infra-owned band
          left to its owner (the sweep/reaper skip it explicitly).
        * everything else                => ``'task'`` — the task-id-shaped
          namespace the sweep/reaper may act on.

    WARNING — adoptable warm/spec lanes (``_lane-`` / ``_spec-``) are
    ``_``-prefixed, so this classifier labels them ``'infra'``.  They are
    NOT infra: they carry recoverable task work and have their own
    adopt/release/quarantine handling.  Every caller MUST therefore check
    ``pool.is_lane()`` (warm AND spec pools) FIRST and only consult this
    classifier for non-lane entries.

    Fail-safe direction: a mis-shaped task name is left alone / leaked
    (recoverable by an operator), never destroyed.
    """
    if name.startswith('_merge-'):
        return 'merge'
    if name.startswith(('_', '.')):
        return 'infra'
    return 'task'


# ---------------------------------------------------------------------------
# Final-defense gate helpers (advance_main)
# ---------------------------------------------------------------------------

async def _assert_no_conflict_markers(sha: str, cwd: Path, context: str) -> None:
    """Raise RuntimeError if the given commit SHA's tree has tracked files
    carrying unresolved git conflict markers at column 0 (esc-2128-8).

    Layer-2 defense-in-depth backstop: the primary guard lives in
    :meth:`GitOps.commit` (unmerged-index detection, BEFORE the tree is
    even staged); this is the last checkpoint before a marker-carrying
    tree could reach main via :meth:`GitOps.advance_main`.

    Matches ONLY the unambiguous opening/closing brackets — ``^<<<<<<< ``
    and ``^>>>>>>> `` (exactly 7 chars + a trailing space, git's exact
    marker format) — anchored at column 0.  Deliberately does NOT match a
    bare ``^=======`` line: that would false-positive on reStructuredText /
    Markdown heading underlines, which are common and legitimate.  Marker-
    like text that isn't anchored at column 0 (e.g. inside a string
    literal) is not matched either.  The trailing space is intentionally
    required: it is git's canonical marker format (``<<<<<<< <label>`` /
    ``>>>>>>> <label>``), so a label-less bare ``<<<<<<<``/``>>>>>>>`` with
    no following text will not match — real git conflicts always emit the
    trailing ref label, so this does not narrow real-world coverage.

    Fail-open (no raise) when git reports no match (``git grep`` exits 1)
    OR on a git error: a broken git invocation must not itself become a
    false block.  Unlike the clean no-match case, a git-error fail-open is
    logged at WARNING (with stderr) so operators can tell a genuinely clean
    tree apart from one this gate failed to evaluate.
    """
    rc, out, err = await _run(
        ['git', 'grep', '-lE', r'^(<{7}|>{7}) ', sha, '--', '.'],
        cwd=cwd,
    )
    if rc == 0 and out.strip():
        # `git grep <tree> -- <pathspec>` prefixes each hit with
        # `<resolved-sha>:` — strip it down to the bare path.
        files = [line.partition(':')[2] for line in out.strip().splitlines()]
        raise RuntimeError(
            f'CONFLICT MARKER GATE FAILED ({context}): commit {sha[:8]} '
            f'contains {len(files)} file(s) with unresolved conflict '
            f'marker(s): {", ".join(files[:5])}. Refusing to advance main. '
            f'Resolve the conflict marker(s) (or abort the operation that '
            f'left them) and re-run.'
        )
    elif rc not in (0, 1):
        # A broken git invocation (rc >= 2) fails open just like a clean
        # no-match (rc == 1) — but silently doing so here would let a
        # genuinely-conflicted tree through undetected while looking
        # identical to "confirmed clean". Log it so operators can tell the
        # difference.
        logger.warning(
            'CONFLICT MARKER GATE (%s): git grep errored (rc=%d) scanning '
            'commit %s — gate could NOT be evaluated; treating as '
            'unevaluated (fail-open), NOT confirmed-clean: %s',
            context, rc, sha[:8], err.strip()[:300],
        )


@dataclass
class MergeResult:
    success: bool
    conflicts: bool = False
    details: str = ''
    merge_commit: str | None = None
    pre_merge_sha: str | None = None
    merge_worktree: Path | None = None


class WarmBaseHealth(Enum):
    """Tri-state health of the warm-lane CoW seed base (task 2061).

    Returned by :meth:`GitOps._warm_lane_base_resolvable`, which resolves the
    base with the SAME D8 rule as :meth:`GitOps._seed_warm_lane`
    (``base.parent / base.readlink()``, NOT ``Path.resolve()``) so its verdict
    matches what a real seed invocation would experience.

    * ``OK`` — the concrete base dir exists and is non-empty; safe to acquire.
    * ``ABSENT`` — the concrete base dir is provably missing or empty; a
      real seed would fail its ``cp``.  This is a HOST-SCOPED pool condition
      (one base serves every lane), so a definite ``ABSENT`` reading drives a
      single fail-open signal (:class:`WarmLanePoolHardDown` / the scheduler's
      warm-base hard-down latch) rather than a per-task fault.
    * ``INDETERMINATE`` — a stat/readlink error occurred while resolving the
      base (e.g. a torn read racing a concurrent rewrite, an EINTR, a
      non-directory base).  Treated as fail-safe "hold" by every consumer:
      never engages, clears, or promotes a hard-down latch, and never blocks
      an acquire — a transient hiccup must never masquerade as a genuine
      outage.
    """
    OK = 'ok'
    ABSENT = 'absent'
    INDETERMINATE = 'indeterminate'


class WarmLaneUnavailable(Enum):
    """Discriminated failure result from :meth:`acquire_warm_lane`.

    Returned instead of bare ``None`` so callers can distinguish:

    * ``EXHAUSTED`` — all pool lanes are ASSIGNED; signal backpressure / requeue.
    * ``FAULT`` — seed/worktree-add failure or absent seed script; signal blocked + L1.
    * ``DISK_PRESSURE`` — seed exited 75 (EX_TEMPFAIL); transient infra; requeue.
    * ``SOFT_PRESSURE`` — θ proactive soft-floor throttle (task 2443, §9.5):
      the reify ε script's ``check --soft`` reported soft pressure (rc=3,
      above the hard floor but below the soft one) for a FRESH allocation
      (no lane already mapped to the branch).  Distinct from
      ``DISK_PRESSURE``'s exit-75 — this is pure backpressure/defer
      (inv.11), never an escalation or a fault, and the pool lane is never
      touched (stays FREE).  A REUSE of an already-mapped branch is never
      throttled this way.
    * ``BASE_ABSENT`` — the warm-lane CoW seed base is provably absent/empty
      (:meth:`GitOps._warm_lane_base_resolvable` returned
      :attr:`WarmBaseHealth.ABSENT`), detected either by the pre-acquire gate
      (no lane touched) or via a seed exit-76 (reify contract, DORMANT until
      a shipped seed-warm-lane.sh emits it).  HOST-SCOPED pool condition —
      requeue (:class:`WarmLanePoolHardDown`), never a per-task BLOCKED+L1.
    * ``RESEED_CONTAMINATED`` — a FRESH-reseed acquire
      (:attr:`AcquireRoute.RECYCLE` / :attr:`AcquireRoute.CREATE_ONCE_FRESH`)
      failed its post-reseed verification
      (:meth:`GitOps._reseed_verified_clean`): the lane's checked-out branch
      is not at the base, still carrying a PRIOR occupant's retained commits
      (reify incident 2026-07-20: ``_lane-12`` acquired for task 5279 while
      ``task/5279`` sat at task 5264's commits).  A data-integrity /
      reseed-consistency defect — :meth:`create_worktree` maps it to
      :class:`WarmLaneReseedContaminated` so the task requeues to re-acquire a
      DIFFERENT lane rather than dispatch onto the stale tree (task 2854).
    * ``DISABLED`` — pool knob is off (``warm_lane_pool is None``); programming-error
      sentinel returned when :meth:`acquire_warm_lane` is called without first
      checking ``self.warm_lane_pool is not None``.  A disabled pool is NOT
      equivalent to backpressure — callers that receive ``DISABLED`` should NOT
      requeue; they should fall back to the cold path or raise.  The canonical
      caller (:meth:`create_worktree`) is already guarded and will never observe
      this value; it is provided solely to surface caller bugs clearly rather
      than silently requeuing forever.

    A lane is always released back to FREE before this value is returned, so no
    ASSIGNED lanes are ever leaked on failure.
    """
    EXHAUSTED = 'exhausted'
    FAULT = 'fault'
    DISK_PRESSURE = 'disk_pressure'
    SOFT_PRESSURE = 'soft_pressure'
    BASE_ABSENT = 'base_absent'
    RESEED_CONTAMINATED = 'reseed_contaminated'
    DISABLED = 'disabled'


def _seed_rc_to_unavailable(rc: int) -> WarmLaneUnavailable:
    """Discriminate a seed-warm-lane.sh exit code into a WarmLaneUnavailable.

    Shared by every seed-rc call site in :meth:`GitOps.acquire_warm_lane` so
    the 75/76/other mapping lives in exactly one place.

    * ``75`` (EX_TEMPFAIL) → ``DISK_PRESSURE`` — transient disk pressure.
    * ``76`` → ``BASE_ABSENT`` — reify contract for "CoW base missing".
      **DORMANT**: no shipped seed-warm-lane.sh emits 76 today: this branch
      is inert until a future reify version adopts the exit-76 convention. It
      is harmless meanwhile (no script exits 76, so it is simply never hit).
    * anything else (including ``127``, the absent-script / unexpected-
      exception sentinel) → ``FAULT`` — generic infra fault.
    """
    if rc == 75:
        return WarmLaneUnavailable.DISK_PRESSURE
    if rc == 76:
        return WarmLaneUnavailable.BASE_ABSENT
    return WarmLaneUnavailable.FAULT


@dataclass
class WorktreeInfo:
    """Return value from create_worktree - captures worktree path and base commit.

    The base_commit is the SHA of main at worktree creation time, pinned to
    ensure stable diffs even if main advances during task execution.

    stale_commits: how far local main was behind the remote at worktree creation
    time.  None means either (a) the fetch was unavailable (no remote configured),
    or (b) the worktree is train-stacked — branched from a sibling's tip rather
    than from main, so the "behind remote" concept does not apply.  0 means
    already current.  A positive stale_commits value means the remote was ahead by
    N commits.  When local main has diverged (has unpushed commits), the worktree
    is based on local main despite the positive count — check this field together
    with base_commit to determine actual freshness.

    reify_debug_port: per-worktree reify-debug port allocated during provisioning
    by running scripts/setup-worktree-debug-port.sh in the worktree.  None when
    no such script is present (non-reify projects) or provisioning failed (fail-open).
    """
    path: Path
    base_commit: str
    stale_commits: int | None = None
    reify_debug_port: int | None = None


@dataclass
class PoolPrewarmResult:
    """Structured outcome of :meth:`GitOps.prewarm_pool` (task 2879).

    Makes an under-provisioned warm-lane pool observable at startup — the
    VISIBLE SIGNAL the task requires so a disk/floor shortfall can never
    silently cap the pool below ``effective_N``:

    * ``target`` — the number of pool lanes prewarm attempted to materialize
      (``len(warm_lane_pool.lane_paths())`` == ``effective_N``).
    * ``already_resident`` — lanes already registered on disk, left untouched
      (no worktree add, no reseed) — the idempotent resident-skip count.
    * ``materialized`` — lanes freshly created + seeded this pass.
    * ``failed`` — lanes that could not be materialized (worktree-add or seed
      failure); each half-created worktree is torn down and NOT left resident.
    * ``failures`` — per-lane ``(lane, rc)`` diagnostics for every failed lane
      (``rc`` is the ``git worktree add`` exit code or the ``_seed_warm_lane``
      rc — e.g. 75 disk-pressure, 127 seed-script-absent).

    Invariant: ``already_resident + materialized + failed == target`` after a
    full pass.  Two early returns are the exceptions: the disabled-pool no-op
    (``warm_lane_pool is None``) returns ``PoolPrewarmResult(target=0)``, and
    the ABSENT-base short-circuit returns ``target == len(lanes)`` with all
    three counters zeroed — no lane is touched because the CoW seed base is
    provably missing, so that early return does NOT satisfy the invariant (it
    logs its own dedicated base-absent WARNING instead of the shortfall one).
    When ``already_resident + materialized < target`` after a full pass the
    pool could not reach ``effective_N`` and prewarm logs a shortfall WARNING.
    """
    target: int
    already_resident: int = 0
    materialized: int = 0
    failed: int = 0
    failures: list[tuple[Path, int]] = field(default_factory=list)


# Validation for create_interactive_worktree's slug argument.  slug is
# interpolated directly into a filesystem path segment
# (worktree_base / f'{iact_prefix}{slug}') and a git branch name
# (f'{branch_prefix}{slug}'), so it is restricted to a conservative safe
# charset — no '/' (would turn the f-string into a multi-component path via
# the Path '/' operator — traversal), no whitespace, no leading '.'/'-'.
# '..' is rejected separately below since it passes this charset but is
# invalid in a git ref component.
_INTERACTIVE_SLUG_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


@dataclass(frozen=True)
class InteractiveWorktreeInfo:
    """Return value from create_interactive_worktree — an isolated interactive worktree.

    Deliberately NOT WorktreeInfo: an interactive worktree is not a
    WarmLanePool/dispatch artifact (isolation invariant I1 — see
    GitOps.create_interactive_worktree).  The shape mirrors the documented
    escalation claim/release contract ``{path, branch, warm, base_ref}``
    (task 2010) consumed by the β claim/release verbs.

    path: the interactive worktree's location,
        ``worktree_base / f'{iact_prefix}{slug}'``.
    branch: the full branch name, ``f'{branch_prefix}{slug}'``.
    warm: True iff the CoW seed (_seed_warm_lane) ran and exited 0; False on
        any seed fault (fail-soft — the worktree is still usable, just cold).
    base_ref: the resolved SHA the worktree was created from (``start_ref``
        or the local ``main_branch`` tip, rev-parsed at creation time —
        deterministic, no remote fetch).
    """
    path: Path
    branch: str
    warm: bool
    base_ref: str


@dataclass(frozen=True)
class ReapedInteractiveWorktree:
    """Record of one ``_iact-*`` worktree removed by ``reap_interactive_worktrees``.

    Returned by :meth:`GitOps.reap_interactive_worktrees` (task δ/2012) — one
    entry per worktree the reaper force-removed during a sweep.

    path: the on-disk location that was removed (``worktree_base/_iact-<slug>``).
    branch: the full branch name that was checked out there (``task/<slug>``).
    slug: the interactive session's claim identity (the ``_iact-`` /
        ``branch_prefix`` suffix).
    reason: why it was reaped — one of ``'landed'`` (a merge marker for this
        branch exists on main), ``'ttl_idle'`` (no activity for longer than
        ``config.interactive_worktree_ttl``), ``'disk_pressure'`` (evicted
        under disk pressure despite being within TTL — idle-only, never a
        worktree carrying unmerged work), or ``'stale_no_stamp'`` (the
        ``.task/interactive.json`` stamp was missing/corrupt and the
        worktree carried no unmerged work).
    """
    path: Path
    branch: str
    slug: str
    reason: str


class ConflictProbe(NamedTuple):
    """Result of merge_tree_conflicts — a lightweight, tuple-destructurable probe.

    Fields
    ------
    clean:
        True  → branch_head would merge onto base_tip without conflicts.
        False → at least one file conflict was detected.
    conflicted_paths:
        Paths of conflicting files (relative to repo root).  Empty when clean.

    As a NamedTuple, supports both named-field access and tuple destructuring::

        probe = await git_ops.merge_tree_conflicts(base_tip, branch_head)
        # named-field access:
        if probe.clean: ...
        # tuple-destructuring (PRD §5.2 contract):
        clean, paths = probe
    """

    clean: bool
    conflicted_paths: list[str]


class WorktreeMissing(FileNotFoundError):
    """Raised when a subprocess cannot start because its ``cwd`` does not exist.

    The orchestrator races against humans who may delete a task's worktree
    out-of-band.  When that happens, ``asyncio.create_subprocess_exec`` raises
    a generic ``FileNotFoundError`` whose ``.filename`` is the missing
    directory.  We re-raise as this typed exception so callers can distinguish
    a missing worktree (recoverable: task may already be done) from a missing
    binary (real bug).
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        super().__init__(f'Worktree missing: {self.path}')


class InteractiveWorktreeLimitError(Exception):
    """Raised by create_interactive_worktree when the _iact-* cap is reached.

    REJECT policy (task 2010 design decision): rather than evicting an idle
    interactive worktree, creation is simply refused once the on-disk
    ``_iact-*`` count under ``worktree_base`` reaches
    ``config.max_interactive_worktrees``.  Evict-oldest-idle would require
    TTL/idle discrimination that belongs to the δ reaper, not this primitive.
    Raised BEFORE any git operation — callers must free a slot (the β release
    verb, a direct ``git worktree remove``, or the δ reaper) and retry.

    The count is raw on-disk ``_iact-*`` directory names under
    ``worktree_base`` — it is NOT cross-checked against ``git worktree
    list``, so a stale/unregistered ``_iact-*`` directory (e.g. left behind
    by a crashed create, before self-heal removes it on its own next
    attempt) occupies a cap slot too.  Reclaiming those is the δ reaper's
    job, not this primitive's.
    """


class EphemeralWorktreeError(Exception):
    """Raised by :meth:`GitOps.ephemeral_worktree` when ``git worktree add``
    fails on every retry attempt (task θ, verify-plan PRD).

    Raised BEFORE the context manager yields, so the caller's ``async with``
    body never runs and no cleanup ``git worktree remove`` is issued (the
    add never succeeded, so there is nothing registered to remove) — mirrors
    :class:`InteractiveWorktreeLimitError`'s raise-before-any-git-op
    contract.  Callers should catch this and fall back to their existing
    fail-safe behavior (e.g. the main-tip probes in verify.py return their
    established sentinel and log a warning).
    """


class WarmLaneRequeue(Exception):
    """Base class for warm-lane failures that should requeue (not block + L1).

    Raised by :meth:`GitOps.create_worktree` when warm_lane_pool is enabled
    and the pool cannot allocate a seeded lane for a transient reason.
    Callers (workflow.run()) should return :attr:`WorkflowOutcome.REQUEUED`
    rather than letting this propagate to the broad ``except Exception`` handler.

    Subclasses:
        WarmLanePoolExhausted — all lanes ASSIGNED (backpressure).
        WarmLaneDiskPressure  — seed exited 75 (EX_TEMPFAIL, transient infra).
        WarmLanePoolHardDown  — warm base absent (host-scoped pool condition).
        WarmLaneSoftPressure  — θ proactive soft-floor throttle (task 2443,
            §9.5): pure backpressure/defer for a FRESH allocation, distinct
            from WarmLaneDiskPressure's exit-75 hard floor.
        WarmLaneReseedContaminated — fresh reseed failed verification: the
            lane still carries a prior occupant's commits (task 2854,
            data-integrity); requeue to re-acquire a DIFFERENT lane.
    """


class WarmLanePoolExhausted(WarmLaneRequeue):
    """All pool lanes are ASSIGNED; task must be requeued (backpressure).

    Scheduler should release the task and re-dispatch it when a lane frees up.
    """


class WarmLaneSoftPressure(WarmLaneRequeue):
    """θ proactive soft-floor throttle detected soft pressure for a FRESH
    allocation (task 2443, §9.5 inv.11) — pure backpressure/defer.

    Raised by :meth:`GitOps.create_worktree` when ``config.warm_lane_soft_floor``
    is enabled and :meth:`GitOps._warm_lane_soft_pressure_defer` reports soft
    disk pressure (rc=3), OR hard disk pressure (rc=75) while
    ``warm_lane_disk_guard`` is disabled (amendment, reviewer_comprehensive
    robustness — a gap-closing belt-and-suspenders for the soft-only
    configuration; see that method's docstring), for a branch with no lane
    already mapped to it (a reuse/live-requeue is never throttled this way).
    Distinct from :class:`WarmLaneDiskPressure` (ε's exit-75 hard floor):
    this is NEVER an escalation or a fault — it is deliberately weaker
    backpressure than the hard floor's requeue, and its disposition-table
    row sets ``counts_against_requeue_cap=False`` so it never contributes to
    a requeue-cap escalation.

    Note (amendment, reviewer_comprehensive robustness — confirmed intended
    per inv.11): a FRESH allocation under *sustained* soft pressure requeues
    indefinitely with no escalation — by design, this is pure backpressure,
    not a fault, so it must never itself trip an escalation path. The only
    operator-facing signal is the per-defer WARNING journal line (grep for
    ``warm_lane_soft_pressure`` / the θ soft-floor throttle message) and the
    ``warm_lane_soft_pressure (backpressure)`` disposition reason_prefix. A
    bounded consecutive-defer counter promoting to an info-level escalation
    would need to track state across dispatch/requeue cycles — that lives in
    the scheduler/harness layer, outside this task's locked module scope —
    so it is intentionally left as a possible future follow-up rather than
    implemented here.
    """


class WarmLaneDiskPressure(WarmLaneRequeue):
    """Seed exited 75 (EX_TEMPFAIL) — transient disk pressure / infra issue.

    Task should be requeued with a ``warm_lane_disk_pressure (transient infra)``
    block-reason annotation so the requeue is distinguishable from backpressure.
    """


class WarmLanePoolHardDown(WarmLaneRequeue):
    """Warm-lane CoW seed base is absent — a HOST-SCOPED pool condition
    (task 2061), not a per-task fault.

    Raised instead of a generic FAULT/RuntimeError so the workflow requeues
    (fail-open, inv.6) rather than escalating a per-task blocked+L1 — one
    dispatched task hitting this is symptomatic of a host-wide condition that
    would otherwise produce N identical escalations for N dispatched tasks.
    The scheduler's warm-base hard-down watchdog (see
    ``Scheduler._apply_warm_base_hard_down_watchdog``) is the PRIMARY
    defense (halts dispatch before any task attempts an acquire); this
    exception is defense-in-depth for any task already in flight when the
    base vanishes.  Task should be requeued with a
    ``warm_lane_pool_hard_down`` block-reason annotation.  Run
    reify/scripts/ensure-warm-base.sh to rebuild the base.
    """


class WarmLaneReseedContaminated(WarmLaneRequeue):
    """A fresh-reseed acquire (RECYCLE / CREATE_ONCE_FRESH) failed its
    post-reseed verification — the lane's branch is not at the base, still
    carrying a PRIOR occupant's retained commits (task 2854).

    A data-integrity / reseed-consistency FAULT, not transient backpressure:
    raised by :meth:`GitOps.create_worktree` instead of a generic
    FAULT/RuntimeError so the workflow REQUEUES (via the shared
    :class:`WarmLaneRequeue` base handler) to re-acquire a DIFFERENT lane
    rather than dispatch a task onto the stale tree.
    :meth:`GitOps.acquire_warm_lane` returns
    :attr:`WarmLaneUnavailable.RESEED_CONTAMINATED` for this condition and has
    already released the contaminated lane back to FREE — retaining the
    commit-bearing branch, so a re-grab of the SAME single-pool lane converges
    to the existing reattach→:class:`BranchResetError` block instead of a
    silent livelock.

    Unlike the transient WarmLaneDiskPressure / WarmLanePoolHardDown /
    WarmLaneSoftPressure rows, its disposition-table row sets
    ``counts_against_requeue_cap=True`` so a persistent/pathological
    contamination eventually trips the requeue-cap escalation — a loud human
    signal — instead of requeuing forever silently.
    """


class WorktreeConflictError(RuntimeError):
    """Raised by :meth:`GitOps.commit` when the worktree has unresolved
    (unmerged-index) conflicts at commit time (esc-2128-8).

    All harness WIP-save auto-commits ('chore: save WIP before …') funnel
    through :meth:`GitOps.commit`.  Without this guard, a prior unresolved
    stash-pop (or any other operation that leaves ``UU``/``AA``/``DD``
    index entries) would be silently snapshotted VERBATIM — including any
    ``<<<<<<<``/``=======``/``>>>>>>>`` conflict markers left in the tree —
    by the unconditional ``git add -A`` + ``git commit``.

    Subclasses ``RuntimeError`` so existing ``RuntimeError``→blocked+L1
    routing (e.g. the requeue reuse path in ``create_worktree``) handles it
    for free without any change on that side; callers that need to
    distinguish this specific condition (e.g. the inter-iteration rebase
    path in ``workflow.run()``) can catch it explicitly.

    ``conflicted_paths`` carries the sorted list of paths reported by
    :meth:`GitOps._detect_unmerged_paths` at raise time.
    """

    def __init__(self, worktree: Path, conflicted_paths: list[str]):
        self.worktree = worktree
        self.conflicted_paths = conflicted_paths
        super().__init__(
            f'Refusing to commit in {worktree}: {len(conflicted_paths)} '
            f'unresolved conflict(s) in the index: '
            f'{", ".join(conflicted_paths[:10])}. Resolve the conflict(s) '
            f'(or abort the operation that caused them) before committing.'
        )


class BranchResetError(RuntimeError):
    """Raised by :meth:`GitOps.rebase_preserving_task_commits` when a
    requeue/inter-iteration rebase would collapse a task branch to zero
    commits ahead of main, silently destroying committed work (RCA: task
    2403 — during merge-train churn around a frozen-prefix merge tip, this
    exact path reset task/2261 and task/2223 to that tip with zero commits
    ahead of main, wiping each task's WIP).

    ``git rebase <ref>`` (the ``rebase_onto_main`` primitive) reports
    success even when its result collapses the branch onto main (or an
    unrelated tip) — nothing about the primitive's return value
    distinguishes "clean rebase, work retained" from "branch silently
    zeroed". :meth:`GitOps.rebase_preserving_task_commits` adds the missing
    POST-CONDITION: if the branch carried commits beyond its rebase
    baseline before the rebase (*n_before* > 0) and carries none after —
    and that isn't just a patch-id dedup of work already applied there,
    see the guard method's docstring — the wipe is detected and a
    ``git reset --hard`` back to the pre-rebase HEAD is attempted before
    this is raised.

    Subclasses ``RuntimeError`` — like :class:`WorktreeConflictError` — so
    it flows to ``TaskWorkflow.run()``'s shared ``except Exception``
    handler, where a dedicated ``isinstance(e, BranchResetError)`` branch
    routes it to a targeted human escalation (``category='branch_reset'``,
    ``escalate_to_human=True``) rather than the generic
    ``'Workflow error:'`` steward-routed path.

    ``worktree``/``onto``/``pre_rebase_head``/``n_before`` carry the
    diagnostic context captured by the guard at raise time. ``restore_ok``
    (default ``True``, kept for back-compat with existing direct
    constructions e.g. in tests) records whether the recovery
    ``git reset --hard`` itself actually succeeded — when it did NOT, the
    work is not confirmed safe, and the message says so explicitly instead
    of asserting the pre-rebase HEAD was restored.
    """

    def __init__(
        self,
        worktree: Path,
        onto: str | None,
        pre_rebase_head: str,
        n_before: int,
        restore_ok: bool = True,
    ):
        self.worktree = worktree
        self.onto = onto
        self.pre_rebase_head = pre_rebase_head
        self.n_before = n_before
        self.restore_ok = restore_ok
        if restore_ok:
            restore_note = f'restored pre-rebase HEAD {pre_rebase_head}'
        else:
            restore_note = (
                f'FAILED TO RESTORE pre-rebase HEAD {pre_rebase_head} — the '
                f"recovery 'git reset --hard' itself failed, so the task's "
                f'work may still be lost from the worktree; recover '
                f'manually from git reflog before proceeding'
            )
        super().__init__(
            f'Refusing requeue/inter-iteration rebase in {worktree}: it '
            f'would collapse the branch from {n_before} commit(s) over main '
            f'to 0 (onto={onto}); {restore_note}'
        )


class MergeVerifyLeaseHeld(RuntimeError):
    """Raised by :meth:`GitOps.reset_persistent_merge_worktree` when a
    DIFFERENT live process holds the merge-verify lease (task 2315, BUG 1).

    Incident: the persistent ``_merge-verify`` worktree was clobbered (a
    reset-in-place ``git reset --hard`` or a create-once stale-dir
    ``shutil.rmtree``) while a verify was still running in it, racing the
    in-flight build out from under itself. The merge-verify flock +
    holder-pgid lease (task 2306) already records who is running a verify
    there — this guard refuses to reset when that lease is held by a pgid
    other than our own (fail-CLOSED). Lease *detection*
    (:meth:`GitOps._merge_verify_lease_active`) is fail-OPEN — a stale,
    dead, or unreadable holder is never treated as held — so this guard can
    never permanently wedge a legitimate reset.

    A caller hitting this should back off and retry once the in-flight
    verify completes and releases the lease.
    """

    def __init__(self, warm_path: Path, holder_pgid: int | None):
        self.warm_path = warm_path
        self.holder_pgid = holder_pgid
        super().__init__(
            f'Refusing to reset persistent merge worktree {warm_path}: '
            f'merge-verify lease is held by a different live process '
            f'(holder pgid={holder_pgid}, self pgid={os.getpgrp()})'
        )


class MergeVerifyLeaseContended(RuntimeError):
    """Raised when the shared merge-verify ``<lane_dir>.lock`` stays contended
    past a bounded wait, so the caller must DEFER rather than proceed
    unprotected.

    TWO raise sites, both on the SAME lock inode with the SAME correct
    response:

    * :meth:`GitOps.merge_verify_lease` — the verify-span lease (task 2828,
      limb 2).  Before that task a contended flock
      (``acquire_merge_verify_flock``'s bounded wait timing out) made the
      lease yield WITHOUT recording a lease — the local verify then ran 1--2h
      fully UNPROTECTED, so a concurrent reseed/thin/gc could clobber its
      working tree mid-run.
    * :meth:`GitOps.reset_persistent_merge_worktree` — the warm-swap RESET's
      own acquire, reached via ``_acquire_warm_verify_worktree`` (task 3003).
      Here the raise means the warm worktree was NEVER touched (fail-CLOSED,
      the tree is left exactly as the holder found it) — NOT that a verify ran
      unprotected.  It fires BEFORE any verify is dispatched at all.

    Both propagate to the merge worker's requeue seam
    (``_run_inflight_verify``), which DEFERS the dispatch — re-queued to try
    again later — instead of resolving a ``MergeOutcome('blocked')``.  That
    placement matters: a blocked resolution here would carry a DETERMINISTIC
    reason string, producing an identical ``merge_outcome_signature`` on every
    attempt and tripping workflow.py's ``consecutive_merge_thrash`` ladder into
    a false-positive human escalation.

    Modeled on :class:`MergeVerifyLeaseHeld` (a RuntimeError carrying lock
    context). Its workflow_types disposition row (REQUEUE,
    ``counts_against_requeue_cap=False``) mirrors MergeVerifyLeaseHeld's — a
    contended lane is a transient "come back later," not a task failure.

    The message states only what was OBSERVED — the lock, the wait, the
    acquire, and the refusal to proceed — never what the caller will do next
    (task 3003 amend, reviewer_comprehensive error_message_accuracy).  The
    disposition is NOT a property of this exception: the merge worker defers and
    re-queues (and logs so itself), while ``cli.py``'s ``verify-merge`` lets the
    same raise propagate as a TERMINAL bail — "deferring this dispatch" would
    tell that operator the exact opposite of what just happened.

    Args:
        lock_path: The contended ``<lane_dir>.lock``.
        wait_secs: The bounded wait that elapsed before giving up.
        operation: Name of the acquire that contended, so an operator log says
            WHICH one lost the race.  Defaults to the task-2828 lease acquire
            (the original raiser), so the ``(lock_path, wait_secs)`` positional
            signature keeps working unchanged for every existing caller and
            test.  ONE message template, not one per raiser: nothing matches on
            the wording (only on ``lock_path``/``wait_secs``), so a second
            f-string would be duplication whose only real effect is letting the
            two variants drift (task 3003 amend, reviewer_comprehensive
            simplification).
        protected_path: Optional tree the refusal is protecting, named in the
            message when given.  The reset's replaced bare ``RuntimeError`` said
            "refusing to mutate {warm_path} unprotected"; this keeps that one
            piece of context — WHICH tree was being guarded — which the lock
            path alone does not convey.
        holder_facts: Optional already-rendered kernel-holder attribution
            (task 3081), appended verbatim.  Attributing reify ``esc-5548-5``
            took manual ``/proc/locks`` + ``stat -c %i`` forensics across a
            roughly three-hour fleet stall; naming the holder costs nothing once
            :func:`~orchestrator.verify_cancel.lane_lock_holder_pids` exists and
            honours the structured-facts-at-failure invariant.  Rendered by the
            raise site rather than here so this class stays a pure carrier and
            performs no I/O in a constructor.  Still ONE template: the facts are
            an appended clause, not a second message.
    """

    def __init__(
        self,
        lock_path: Path,
        wait_secs: float,
        *,
        operation: str = 'the merge-verify lease acquire',
        protected_path: Path | None = None,
        holder_facts: str | None = None,
    ):
        self.lock_path = lock_path
        self.wait_secs = wait_secs
        self.operation = operation
        self.protected_path = protected_path
        self.holder_facts = holder_facts
        _protecting = (
            f' (protecting {protected_path})' if protected_path is not None else ''
        )
        _facts = f'; {holder_facts}' if holder_facts else ''
        super().__init__(
            f'merge-verify lane lock {lock_path} still contended after a '
            f'{wait_secs}s bounded wait during {operation}{_protecting} — '
            f'refusing to proceed unprotected{_facts}'
        )


class LaneLockSelfOwnedLeak(MergeVerifyLeaseContended):
    """Raised when the contended ``<lane_dir>.lock`` is held by an fd in THIS
    process that nothing will ever release — a SELF-OWNED LEAK, not contention
    (task 3081; PRD ``plans/warm-lane-infra-repatriation-prd.md`` §D8/B13).

    Incident anchor: reify ``esc-5548-5``.  A cancelled
    :func:`asyncio.to_thread` acquire won the flock after its awaiting coroutine
    had already gone away, so the fd was discarded unreleased and the lane lock
    stayed held until process exit.  Three tasks then blocked behind one
    identical ``merge_outcome_signature``, and because the symptom was
    indistinguishable from ordinary contention nothing surfaced until an
    unattended restart roughly three hours later.  Kernel forensics — by hand,
    at the time — read ``/proc/locks`` and inode-matched
    ``FLOCK ADVISORY WRITE 588232 07:1d:4300647613`` against
    ``_merge-verify.lock``.  This class is that diagnosis, made automatic.

    Explicitly NOT either neighbouring case:

    * NOT foreign contention.  A reify ``flock(1)``, a ``verify-merge`` CLI
      subprocess, or another orchestrator holding the lane is healthy, and stays
      on the plain :class:`MergeVerifyLeaseContended` fail-closed path.
    * NOT DF 3003's long-holder case.  A genuine live verify holding the lane
      past the bounded wait is contention too; it defers quietly rather than
      raising this.

    IS-A :class:`MergeVerifyLeaseContended`, and payload-compatible with it.
    That is load-bearing, not taxonomy: both merge-worker consumers
    (``merge_queue.py``'s cross-check fail-safe and its bounded contended-defer
    arm) are isinstance-based on the parent, so a standalone type would fall
    through to the generic ``except Exception`` → ``MergeOutcome('blocked')``
    with a DETERMINISTIC reason string — an identical
    ``merge_outcome_signature`` on every attempt, tripping ``workflow.py``'s
    ``consecutive_merge_thrash`` ladder into precisely the false-positive human
    escalation DF 3003 was chartered to stop.  ``operation`` and
    ``protected_path`` are forwarded, and the message keeps both of the parent's
    contractual properties (it names the protected tree; it never says
    "deferring", since this raise also reaches ``cli.py``'s ``verify-merge``
    where it is a TERMINAL bail).

    Detection is REPORT-ONLY — the leaked fd is deliberately not released.  B12
    (the shielded acquire) is the mechanism that guarantees the hold ends; this
    is the backstop diagnosis for any residual path.  Blind-releasing an fd
    whose true owner we could not identify would risk yanking the lock out from
    under a live in-process span whose registration we failed to observe,
    converting a diagnosable stall into a silent tree clobber — the exact
    failure class the lane lock exists to prevent.

    Args:
        lock_path: The leaked ``<lane_dir>.lock``.
        wait_secs: The bounded wait that elapsed before giving up.
        holder_pids: Kernel-reported FLOCK holders of that lock's inode; our own
            pid is among them, which is what makes this a leak.
        self_pid: This process's pid.
        self_pgid: This process's pgid — the corroborating fact the holder-pgid
            rendezvous would have carried had the orphaned acquire ever
            completed (it writes the rendezvous only AFTER the acquire returns,
            which is exactly why a pgid-only check cannot see this fault).
        operation: Forwarded to the parent — WHICH acquire hit the leak.
        protected_path: Forwarded to the parent — the tree being guarded.
        holder_pgid: The recorded rendezvous pgid at detection time, or ``None``
            when unset.  ``None`` is the expected reading for a genuine leak and
            is stated in the message so an operator can tell "nothing recorded"
            from "recorded but dead".
    """

    def __init__(
        self,
        lock_path: Path,
        wait_secs: float,
        *,
        holder_pids: Iterable[int],
        self_pid: int,
        self_pgid: int,
        operation: str = 'the merge-verify lease acquire',
        protected_path: Path | None = None,
        holder_pgid: int | None = None,
    ):
        self.holder_pids = list(holder_pids)
        self.self_pid = self_pid
        self.self_pgid = self_pgid
        self.holder_pgid = holder_pgid
        super().__init__(
            lock_path,
            wait_secs,
            operation=operation,
            protected_path=protected_path,
            holder_facts=(
                f'SELF-OWNED LEAK — this process (pid={self_pid}, pgid={self_pgid}) '
                f'is itself among the kernel FLOCK holders {self.holder_pids} of '
                f'that lock, with no in-process hold registered and no live '
                f'merge-verify lease (recorded holder pgid={holder_pgid}): an '
                f'earlier acquire in this process leaked its fd, and nothing will '
                f'release it before process exit'
            ),
        )


# ---------------------------------------------------------------------------
# In-process held-lane-lock registry (task 3081, D8/B13 layer 2)
#
# Layer (1) of the leak predicate — "is our pid among the kernel's FLOCK
# holders" — cannot tell a LEAKED fd from a LIVE one, because both are held by
# this process.  This registry is the discriminator: every in-process lane-lock
# acquire records its fd here and every release forgets it, so a lock the kernel
# attributes to us with NO entry here is held by an fd no code path owns.
#
# Completeness is what makes layer (2) sound, so all three in-process acquire
# sites register: both leases (via _acquire_lane_flock_off_thread) and
# remove_merge_worktree_guarded's sub-millisecond sync acquire.  That last one
# is ephemeral-lane-only and would almost never overlap — but
# merge_verify_lease(lane_dir=...) can be handed an ephemeral lane (the DF 2822
# per-land cross-check), so the inodes CAN coincide, and a false leak report is
# a loud human escalation that must not be reachable from a legitimate hold.
#
# Asymmetric by design: a MISSED forget can only mask a real leak (a false
# negative, degrading to today's behaviour), whereas a missed register would
# libel a healthy hold.  When in doubt, register.
# ---------------------------------------------------------------------------

#: fd -> resolved lock path, for every lane lock currently held by this process.
_HELD_LANE_LOCK_FDS: dict[int, str] = {}

#: Guards :data:`_HELD_LANE_LOCK_FDS`.  A plain ``threading.Lock`` and not an
#: asyncio primitive: registration happens on ``asyncio.to_thread`` WORKER
#: THREADS (that is the whole point of the off-thread acquire), so the mutation
#: is genuinely cross-thread, not merely cross-coroutine.
_HELD_LANE_LOCK_FDS_LOCK = threading.Lock()


def _register_held_lane_lock(fd: int, lock_path: Path) -> None:
    """Record *fd* as a LIVE in-process hold of *lock_path*."""
    with _HELD_LANE_LOCK_FDS_LOCK:
        _HELD_LANE_LOCK_FDS[fd] = str(Path(lock_path).resolve())


def _forget_held_lane_lock(fd: int) -> None:
    """Drop *fd*'s registration; idempotent, so a double release is harmless.

    Must run on EVERY release, including the orphan-callback release, or a
    reused fd number could later be mistaken for a live hold.
    """
    with _HELD_LANE_LOCK_FDS_LOCK:
        _HELD_LANE_LOCK_FDS.pop(fd, None)


def _lane_lock_held_in_process(lock_path: Path) -> bool:
    """True iff some live in-process hold is registered for *lock_path*.

    Compares RESOLVED paths so the persistent lane reached through a symlinked
    ``worktree_base`` and the same lane reached directly agree.
    """
    target = str(Path(lock_path).resolve())
    with _HELD_LANE_LOCK_FDS_LOCK:
        return target in _HELD_LANE_LOCK_FDS.values()


def _lane_lock_holder_facts(lock_path: Path) -> str:
    """Render the kernel's view of who holds *lock_path*, for a failure message.

    Names each holder pid and its pgid, flagging any that shares ours — the two
    facts the incident's manual ``/proc/locks`` + ``stat -c %i`` forensics had to
    reconstruct by hand.  Fail-safe throughout: an unreadable procfs or a holder
    that exits mid-render degrades the clause, never the raise it decorates.
    """
    pids = lane_lock_holder_pids(lock_path)
    if not pids:
        return (
            'the kernel reports no FLOCK holder of that lock (it was likely '
            'released between the timeout and this probe)'
        )
    ours = os.getpgrp()
    rendered = []
    for pid in pids:
        try:
            pgid = os.getpgid(pid)
        except OSError:
            # Exited between the /proc/locks read and here, or not ours to see.
            rendered.append(f'pid {pid} (pgid unknown)')
            continue
        rendered.append(f'pid {pid} (pgid {pgid}{", ours" if pgid == ours else ""})')
    return 'kernel FLOCK holders: ' + ', '.join(rendered)


async def _run(
    cmd: list[str], cwd: Path | None = None, *, input_text: str | None = None,
) -> tuple[int, str, str]:
    """Run an arbitrary subprocess command and return (returncode, stdout, stderr).

    Used throughout for git invocations and for any other subprocess call
    (e.g. project setup scripts).  Raises :class:`WorktreeMissing` if ``cwd``
    is provided but does not exist, so the caller can distinguish a deleted
    worktree (recoverable race) from other ``FileNotFoundError``\\ s (e.g.
    missing binary on ``PATH``).

    Stdin feeding (``input_text``): when provided, the child is spawned with
    ``stdin=PIPE`` and ``input_text.encode()`` is written to it via
    ``communicate(input=...)``.  This is what lets callers pipe a diff into a
    stdin-only filter such as ``git patch-id`` (see
    :meth:`GitOps.find_equivalent_commit`).  When ``None`` (the default) the
    behaviour is exactly as before — stdin is not piped and the child inherits
    the parent's — so no existing caller is affected.  The capability is inert
    unless ``input_text`` is passed.

    Locale: ``LC_ALL=C`` and ``LANG=C`` are forced in the child environment so
    that git (and other tools) always emit English-locale diagnostics.  This is
    required for :func:`_git_clean_failure_is_benign`, which substring-matches
    English warning text; a non-C locale would produce translated output that
    the matcher cannot recognise, silently defeating the R3 ENOENT-tolerance
    fix for the 4892-class warm-lane FAULT.

    Cancellation safety (task 2608): if the ``await proc.communicate()`` below
    is cancelled — e.g. by a caller wrapping ``_run`` in
    ``asyncio.wait_for(..., timeout=...)``, as delivered_checks.py's
    ``_run_script_check`` does for script-kind delivered checks — the spawned
    child would otherwise keep running as an orphan with its stdout/stderr
    pipes open. For a persistently-hung script this recurred every scheduler
    sweep, leaking a process and file descriptors. The child is now
    best-effort killed and reaped before the triggering exception (including
    ``asyncio.CancelledError``) is re-raised.
    """
    # Pre-flight: a missing cwd surfaces as a generic FileNotFoundError from
    # posix_spawn whose .filename is not reliably set.  Check explicitly so we
    # can raise a typed exception consumers can pattern-match on.
    if cwd is not None and not Path(cwd).is_dir():
        raise WorktreeMissing(cwd)
    # Force a stable C locale so git output is always in English and amenable
    # to substring matching (see docstring above).
    _env = {**os.environ, 'LC_ALL': 'C', 'LANG': 'C'}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd else None,
            stdin=asyncio.subprocess.PIPE if input_text is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_env,
        )
    except FileNotFoundError as e:
        # Race: cwd existed at the pre-flight check but vanished before spawn.
        # Re-classify as WorktreeMissing if cwd is now gone; otherwise the
        # error is about the binary itself.
        if cwd is not None and not Path(cwd).is_dir():
            raise WorktreeMissing(cwd) from e
        raise
    try:
        stdout, stderr = await proc.communicate(
            input=input_text.encode() if input_text is not None else None,
        )
    except BaseException:
        # The await was interrupted (most commonly asyncio.CancelledError from
        # a caller-side asyncio.wait_for(..., timeout=...)) before the child
        # exited. Best-effort kill + reap it so it doesn't leak as an orphan
        # process with dangling stdout/stderr pipes, then propagate the
        # original exception unchanged.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()  # already exited
        with contextlib.suppress(BaseException):
            await proc.wait()  # reap is best-effort; never let it mask the original error
        raise
    return proc.returncode if proc.returncode is not None else 1, stdout.decode().strip(), stderr.decode().strip()


def _git_clean_failure_is_benign(stderr: str) -> bool:
    """Return True iff every non-empty stderr line is a benign ENOENT ``failed to remove`` warning.

    git 2.43+ emits ``warning: failed to remove '<path>': No such file or
    directory`` (via ``warning_errno("failed to remove %s")``) when a path it
    planned to remove was already deleted by a concurrent process (e.g. the
    detached ``rm -rf`` from reify's warm-lane reseed).  This is the R3 race
    from the 4892-class warm-lane FAULT: we can safely ignore it because the
    desired outcome (path gone) is already achieved.

    Decision criteria:
    - Empty stderr (no lines) → **not benign** — an unknown non-zero exit with
      no diagnostic context must still raise so we do not silently swallow
      genuine failures.
    - Any line that does NOT contain both ``'failed to remove'`` AND
      ``'No such file or directory'`` → **not benign** — could be a real error
      (permission denied, I/O error, …) co-occurring with an ENOENT line.
    - All non-empty lines are ENOENT ``failed to remove`` warnings → benign.

    Substring matching is robust to path quoting and multi-line output ordering.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return False
    return all(
        'failed to remove' in line and 'No such file or directory' in line
        for line in lines
    )


def _git_stderr_is_unresolved_ref(stderr: str) -> bool:
    """Return True iff *stderr* is git's "this ref does not resolve" diagnostic.

    A ``git diff <base>...<ref>`` for a ``<ref>`` that does not exist (e.g. a
    not-yet-dispatched ``pending`` task whose ``task/<id>`` branch has never
    been created) fails non-zero with a *fatal* rev-parse diagnostic rather
    than a genuine git fault:

    * ``fatal: bad revision 'main...task/999'``
    * ``fatal: ambiguous argument 'main...nope': unknown revision or path not
      in the working tree.``
    * ``fatal: Not a valid object name ...`` (older git)

    Callers that legitimately expect some refs to be absent — the merge-skew
    pipeline-landing tripwire scans every in-flight task, many of which have
    no branch yet — use this to classify the absent-ref case as an expected
    quiet skip (DEBUG) while a genuine git failure (corrupt repo, I/O error,
    permission denied — none of which carry these markers) still surfaces
    loudly (WARNING), so the absent-ref noise never buries a real diff error.

    Empty stderr → False: with no diagnostic there is no evidence it is a mere
    absent ref, so it is treated as a genuine failure rather than silently
    downgraded.  Substring matching is case-insensitive and robust to git's
    ref/path quoting and version-to-version wording.
    """
    haystack = stderr.lower()
    if not haystack.strip():
        return False
    return any(
        marker in haystack
        for marker in (
            'unknown revision',
            'bad revision',
            'ambiguous argument',
            'not a valid object name',
        )
    )


def _merge_subject(branch: str, main_branch: str) -> str:
    """Return the canonical subject line for a no-ff merge of *branch* into *main_branch*.

    Single source of truth for the merge commit subject format consumed by
    ``find_merge_marker``, ``merge_to_main``, and the retry path in
    ``advance_main``.  Changing this function is the one place where the
    format needs to be updated — all three consumers will automatically
    use the new format.
    """
    return f'Merge {branch} into {main_branch}'


# Sentinel range used to represent files that are fully deleted or renamed.
# The range (0, 2**30) spans every plausible line number, so an intersection
# check against any real hunk range always returns True (not stackable).
_WHOLE_FILE_SENTINEL: tuple[int, int] = (0, 2**30)


def parse_diff_line_ranges(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse a unified diff and return old-side (BASE) line ranges per file.

    Given the output of ``git diff <main>...<ref> --unified=0 --no-color``,
    returns a mapping of file path → list of (start, end) tuples representing
    old-side (BASE/main-relative) changed line ranges.  Using old-side ranges
    from both branches diffed against the same main makes ranges directly
    comparable for stackability checks.

    Pure insertion hunks (old_count == 0, e.g. ``@@ -7,0 +8,3 @@``) are
    mapped to a point range ``(old_start, old_start)`` so they are still
    comparable; ``@@ -N,0 ... @@`` anchors at line N (the line *before* the
    insertion in the old file).

    Deleted files (``+++ /dev/null``), pure renames (``rename from``), and
    renames with content changes (``--- a/old`` → ``+++ b/new``) are
    represented via ``_WHOLE_FILE_SENTINEL`` on the old-side path.  This
    ensures that a modify/delete or rename/modify pair between two tasks is
    always flagged non-stackable by the stackability gate.

    Returns an empty dict for an empty or header-only diff.
    """
    import re

    result: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    old_path: str | None = None  # from '--- a/<path>'; reset per diff block

    hunk_re = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@')

    for line in diff_text.splitlines():
        if line.startswith('diff --git '):
            # Start of a new file block — reset per-file state.
            current_file = None
            old_path = None
        elif line.startswith('--- a/'):
            # Record old-side path for deletion / rename detection below.
            old_path = line[6:]
        elif line.startswith('+++ b/'):
            new_path = line[6:]
            # Rename with content changes: old_path ≠ new_path.  The old file is
            # gone; represent it with a sentinel so tasks touching the old name
            # are flagged non-stackable with this rename.
            if old_path and old_path != new_path and old_path not in result:
                result[old_path] = [_WHOLE_FILE_SENTINEL]
            current_file = new_path
            if current_file not in result:
                result[current_file] = []
        elif line.startswith('+++ /dev/null'):
            # File deletion: old file is completely gone.  Represent old_path
            # with the whole-file sentinel so any task modifying this file is
            # flagged non-stackable with the deletion.
            if old_path and old_path not in result:
                result[old_path] = [_WHOLE_FILE_SENTINEL]
            current_file = None  # no new file; skip hunk parsing
        elif line.startswith('rename from '):
            # Pure rename (R100) header — no --- / +++ lines follow for the old
            # path.  Add it with the sentinel so tasks touching the old name are
            # flagged.  For renames with content changes the --- a/ handler above
            # also runs, but the 'not in result' guard prevents a double-insert.
            renamed_from = line[len('rename from '):]
            if renamed_from not in result:
                result[renamed_from] = [_WHOLE_FILE_SENTINEL]
        elif current_file is not None:
            m = hunk_re.match(line)
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                # Pure insertion: old_count == 0 → point range at old_start.
                end = old_start + max(old_count, 1) - 1
                result[current_file].append((old_start, end))

    return result


def parse_diff_added_line_ranges(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse a unified diff and return new-side (HEAD) line ranges per file.

    NEW-side counterpart of :func:`parse_diff_line_ranges`.  Given the output
    of ``git diff <from>..HEAD --unified=0 --no-color``, returns a mapping of
    new file path (from ``+++ b/<path>``) → list of (start, end) tuples
    representing the new-side (HEAD-relative) changed line ranges.  Reviewer
    ``location`` line numbers are new-side, so these ranges — not the old-side
    ranges the sibling parser produces — are what a suggestion's line number
    is matched against when scoping a post-amendment review to the amendment
    delta.

    A hunk header ``@@ -x,y +a,b @@`` describes new-side lines
    ``[a, a + max(b, 1) - 1]`` (``,b`` defaults to 1 when absent).  Hunks whose
    new-side count is ``0`` (pure deletions, e.g. ``@@ -5,3 +4,0 @@``)
    contribute NO new-side range.  A fully deleted file (``+++ /dev/null``) has
    no new-side path and produces no entry.  Pure renames (no ``+++ b/`` line,
    no content change) likewise produce no new-side entry.

    A ``+++ b/…`` / ``+++ /dev/null`` line is treated as a file header only when
    it immediately follows the ``--- `` old-side header.  An added *content*
    line whose text begins with ``++ b/`` renders in a unified diff as
    ``+++ b/…`` — indistinguishable at a string-prefix level from a real header;
    the preceding-line guard keeps such a hunk-body line from being misread as a
    new-file header and silently resetting the current file (which would drop
    that file's later hunks onto a bogus path).

    Returns an empty dict for an empty or header-only diff.
    """
    import re

    result: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    # True iff the previous line was the ``--- `` old-side diff header; gates
    # the ``+++`` header branches below (see docstring).
    prev_is_old_header = False

    hunk_re = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')

    for line in diff_text.splitlines():
        if line.startswith('diff --git '):
            # Start of a new file block — reset per-file state.
            current_file = None
        elif prev_is_old_header and line.startswith('+++ b/'):
            current_file = line[6:]
            if current_file not in result:
                result[current_file] = []
        elif prev_is_old_header and line.startswith('+++ /dev/null'):
            # File deletion: no new-side path — skip hunk parsing for this block.
            current_file = None
        elif current_file is not None:
            m = hunk_re.match(line)
            if m:
                new_start = int(m.group(1))
                new_count = int(m.group(2)) if m.group(2) is not None else 1
                # Pure-deletion hunk (new_count == 0): no new-side lines.
                if new_count == 0:
                    continue
                end = new_start + new_count - 1
                result[current_file].append((new_start, end))
        prev_is_old_header = line.startswith('--- ')

    return result


class GitOps:
    """Git worktree and merge operations."""

    def __init__(
        self,
        config: GitConfig,
        project_root: Path,
        *,
        warm_lane_pool_size: int = 0,
        merge_spec_warm_lane_pool_size: int = 0,
        transcript_archive: TranscriptArchiveConfig | None = None,
    ):
        self.config = config
        self.project_root = project_root
        # Teardown-archival backstop config (task 2786,
        # agent-transcript-archival-prd β).  None at the 3 non-dispatch
        # construction sites (cli/recover/evals) so the cleanup_worktree
        # backstop is byte-identical (inert) there; the live submodel is
        # installed only by Harness (harness.py), which holds the full
        # OrchestratorConfig — passing the reference (not a copy) preserves
        # in-place green-tier hot-reload of enabled/root.
        self.transcript_archive = transcript_archive
        self.worktree_base = (project_root / config.worktree_dir).resolve()
        # Durable per-lane lifecycle record writer (W11 gamma).  Shared by
        # acquire_warm_lane/release_warm_lane (durable ASSIGNED/RELEASED
        # writes below) and the .pool-root sentinel delegators. escalation_
        # queue=None: GitOps has no escalation queue wired (mirrors the other
        # unwired-callback attributes in this constructor) — delta/harness can
        # inject a real one later. quarantine_worktree is wired now (harmless
        # in gamma; consumed by delta).
        self._lane_lifecycle = LaneLifecycle(
            self.worktree_base, quarantine_worktree=self.quarantine_worktree,
        )
        # Warm-lane pool — None when knob off or size=0 (default-off, trivially
        # revertible, mirrors persistent_merge_worktree).  Size is passed from
        # OrchestratorConfig.max_concurrent_tasks by Harness at startup (D9).
        self._warm_lane_pool_size = warm_lane_pool_size
        if warm_lane_pool_size > 0 and config.warm_lane_pool:
            from orchestrator.warm_lane_pool import WarmLanePool
            self.warm_lane_pool: WarmLanePool | None = WarmLanePool(
                worktree_base=self.worktree_base,
                size=warm_lane_pool_size,
                drift_l2_threshold=config.warm_lane_drift_l2_threshold,
            )
            # Wire the shared durable-record writer so the pool is the SOLE
            # writer of ASSIGNED/RELEASED .lane-state records at the moment the
            # in-memory state flips (task 2986, W2b I1: record ≡ map after every
            # mutation).  Previously only harness crash-recovery wired this;
            # doing it here makes write-through + fail-open hold UNCONDITIONALLY,
            # including GitOps-level unit tests.  The harness crash-recovery
            # set_lane_lifecycle call is now a redundant idempotent no-op (same
            # LaneLifecycle instance).
            self.warm_lane_pool.set_lane_lifecycle(self._lane_lifecycle)
        else:
            self.warm_lane_pool = None
        # Merge-speculation warm-lane pool — None when knob off or size=0.
        # Second WarmLanePool instance with name_prefix='_spec-' for K>1 LOCAL
        # speculative verify slots.  Size K = speculation_depth, sized from the
        # SAME shared K source as the worker (steps 5-6, harness.py).
        # Default-off, byte-identical at default, mirrors warm_lane_pool above.
        self._merge_spec_warm_lane_pool_size = merge_spec_warm_lane_pool_size
        if merge_spec_warm_lane_pool_size > 0 and config.merge_spec_warm_lane_pool:
            from orchestrator.warm_lane_pool import WarmLanePool as _WLP
            self.spec_warm_lane_pool: WarmLanePool | None = _WLP(
                worktree_base=self.worktree_base,
                size=merge_spec_warm_lane_pool_size,
                name_prefix='_spec-',
            )
        else:
            self.spec_warm_lane_pool = None
        # Serialize first-time `git worktree add` for _spec- lanes.
        # Git serializes worktree administration via a repo-level lock; concurrent
        # adds from the same project_root can transiently fail during the initial
        # K>1 warm-up burst.  Reset-in-place (already-registered) acquires are
        # per-lane and don't contend, so this only guards the one-time create path.
        self._spec_wt_create_lock: asyncio.Lock = asyncio.Lock()
        # Serialize create_interactive_worktree's cap-count + create-once span
        # on THIS instance (task 2010 amendment) — mirrors _spec_wt_create_lock
        # above.  Without it, two concurrent calls can both observe the
        # on-disk _iact-* count under max_interactive_worktrees and both
        # proceed, overrunning the REJECT cap (TOCTOU).  In-process only —
        # matches the pool's own in-process concurrency scope, not
        # cross-process safe.
        self._interactive_wt_lock: asyncio.Lock = asyncio.Lock()
        # Reclaim-on-exhaustion safety valve callbacks (task 1933).
        # Declared here (default None), installed by Harness.__init__ when
        # config.git.warm_lane_reclaim_on_exhaustion is True — mirrors the
        # _on_park_stop_trip / _on_external_dep_block declare-in-callee /
        # install-in-harness pattern.  Both default None so an unwired GitOps
        # (cli/recover/evals, knob-off) is byte-identical.
        self.warm_lane_reclaim_candidate_provider: (
            Callable[[list[str]], Awaitable[set[str]]] | None
        ) = None
        self.warm_lane_dispatched_predicate: Callable[[str], bool] | None = None
        # Pool-storage-absent safety-valve callback (task 2099).  Declared
        # here (default None), installed by Harness.__init__ — mirrors the
        # declare-on-callee / install-in-harness pattern used by
        # warm_lane_reclaim_candidate_provider above.  Fired best-effort by
        # _note_pool_storage_absent() from every destructive-sweep guard site
        # (prune_worktrees, _run_warm_lane_gc_reclaim, acquire create-once)
        # when pool_storage_present() is False.  None (unwired) is
        # byte-identical to today — e.g. cli/recover/evals call sites.
        self._on_pool_storage_absent: Callable[..., Any] | None = None
        # Structural-exhaustion (PRD ε pole-2) loudness callback (task 2988).
        # A pool-GLOBAL consecutive-EXHAUSTED counter: incremented at the single
        # EXHAUSTED return in _acquire_warm_lane_impl (via
        # _note_structural_exhaustion), reset to 0 on any successful FRESH lane
        # allocation (acquire_for reused=False) or safety-valve reclaim.  Once it
        # reaches config.warm_lane_structural_exhaustion_l2_threshold and the
        # callback is installed, fires _on_structural_exhaustion(count, census)
        # best-effort so the Harness files ONE deduped born-at-L2 — the sole loud
        # signal for a pool stuck emitting EXHAUSTED forever (silent-infinite-
        # requeue).  Declared here (default None), installed by Harness.__init__
        # when a pool exists — same declare-on-callee / install-in-harness
        # pattern as _on_pool_storage_absent above (None = byte-identical when
        # unwired: cli/recover/evals, knob-off, pool-less).
        self._consecutive_exhausted: int = 0
        self._on_structural_exhaustion: (
            Callable[[int, WarmLanePoolCensus], None] | None
        ) = None
        # θ soft-floor defer α-audit memo (task 2443, amendment): a
        # (monotonic_deadline, headroom) pair, or None before the first defer.
        # Consulted only by _warm_lane_audit_cached() on the (non-hot) defer
        # path; see _WARM_LANE_AUDIT_CACHE_TTL_SECS.  This GitOps is long-lived
        # (Harness holds one for the process lifetime), so the memo genuinely
        # survives across the requeue cycles a sustained soft-pressure
        # condition produces.
        self._warm_lane_audit_cache: tuple[float, str | None] | None = None
        # Merge serialization is handled by MergeWorker in merge_queue.py.
        # See task 292 for design rationale (ghost loops, lock starvation,
        # branch drift at 64 max concurrency with external actors).

    def protected_prefixes(self) -> dict[str, str]:
        """Authoritative band-ownership registry for this instance.

        Returns the module-level :data:`PROTECTED_PREFIXES` (the static
        bands) merged with this instance's config-driven interactive band
        (``self.config.iact_prefix`` -> ``'interactive'``).  The iact band
        is config-shaped (:attr:`GitConfig.iact_prefix` may be overridden
        per deployment), so a single module constant cannot capture the
        authoritative band map — the per-instance view is the correct one
        for callers to consult, including :meth:`_refuse_foreign_band`.

        Built on :func:`default_protected_prefixes` so the registry +
        iact-band merge exists in exactly one place.  This instance's
        ``iact_prefix`` is passed in rather than layered on top, so an
        override REPLACES the default band instead of widening the map with
        an ``_iact-`` this deployment never mints.
        """
        return default_protected_prefixes(self.config.iact_prefix)

    def _refuse_foreign_band(
        self, path: Path, owned: frozenset[str], context: str,
    ) -> bool:
        """True if *path* belongs to a protected band this sweep does not own.

        Band-ownership guard for destructive worktree cleanup — defense in
        depth against a filter bug that steers a foreign band's directory
        into a sweep's removal step (gitops-chokepoints PRD, Mechanism 3).
        This is a band-ownership check, not a general ACL: it fails OPEN
        (returns False, i.e. "proceed") for anything outside its narrow
        scope —

        - *path* is not a direct child of :attr:`worktree_base` (task
          worktrees, quarantine dirs, and other nested paths are outside
          this check).
        - *path*'s name matches no band token in :meth:`protected_prefixes`.
        - The matched band token is already in *owned*.

        Otherwise this logs a WARNING naming the matched band token, its
        owner, and *context*, and returns True.  Callers must skip the
        destructive call for this one candidate (never raise) — every call
        site sits inside a best-effort / never-raise sweep.
        """
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved.parent != self.worktree_base:
            return False

        registry = self.protected_prefixes()
        name = resolved.name
        exact_match = next(
            (key for key in registry if not key.endswith('-') and name == key),
            None,
        )
        prefix_match = next(
            (key for key in registry if key.endswith('-') and name.startswith(key)),
            None,
        )
        # Exact-name keys take precedence over prefix keys: e.g. the
        # persistent `_merge-verify` worktree matches both its own exact
        # name and the `_merge-` prefix, and must resolve to the exact
        # persistent-merge-verify token so a plain `_merge-` owner still
        # refuses it (see the design_decisions entry on exact-first
        # precedence).
        token = exact_match if exact_match is not None else prefix_match
        if token is None:
            return False
        if token in owned:
            return False

        logger.warning(
            '%s: refusing to remove %s — belongs to protected band %r '
            '(owner=%r), not owned by this sweep (owned=%r)',
            context, path, token, registry[token], sorted(owned),
        )
        return True

    def refuse_foreign_band(
        self, path: Path, owned: frozenset[str], context: str,
    ) -> bool:
        """Public entry point for the :meth:`_refuse_foreign_band` guard.

        Cross-module callers (e.g. harness.py's substrate-gate pre-clean)
        should consult the band-ownership guard through this supported
        public method rather than reaching across the module boundary into
        the leading-underscore internal name — mirrors
        :meth:`protected_prefixes`, which is public for the same reason.
        Delegates to :meth:`_refuse_foreign_band`; see that method's
        docstring for the full contract (fail-open rules, WARNING
        semantics, never-raise guarantee). Kept as a thin wrapper (rather
        than renaming the internal method) so existing intra-class callers
        and unit tests targeting the primitive directly are undisturbed.
        """
        return self._refuse_foreign_band(path, owned, context)

    @contextlib.asynccontextmanager
    async def ephemeral_worktree(
        self, kind: WorktreeKind, sha: str, *, warm_seed: bool = False,
    ) -> AsyncIterator[Path]:
        """Mint a throwaway detached worktree pinned at *sha*, with
        GUARANTEED scoped cleanup on exit (task θ, verify-plan PRD).

        Single extraction point for the main-tip probes' (
        ``verify_failure_is_preexisting_on_main``, ``run_main_tip_sweep`` —
        both in verify.py) previously copy-pasted worktree lifecycle: mint
        ``worktree_base/<kind.value><hex>`` (*kind*'s value IS both the
        directory-name prefix and its :data:`PROTECTED_PREFIXES` registry
        key — see :class:`WorktreeKind`), retry ``git worktree add
        --detach`` up to 3 times with ``0.5 * (attempt + 1)``\\ s linear
        backoff on transient lock contention (concurrent sibling probes
        serialise on git's repo-level metadata lock), then yield the path.

        On exit — normal return OR an exception raised in the ``async
        with`` body — cleanup ALWAYS runs: scoped ``git worktree remove
        --force <path>`` + an unconditional ``shutil.rmtree(path,
        ignore_errors=True)`` belt-and-suspenders in case ``remove`` left an
        empty skeleton.  This method NEVER issues ``git worktree prune``
        (DD5 — the comment-only invariant that failed in the 2026-07-04
        warm-lane broad-prune registration-wipe incident, df 2097-2100): a
        broad prune would deregister ANY concurrently-active sibling
        probe/merge worktree, not just this one.  Scoped ``remove --force``
        deregisters ONLY the path minted here.

        Args:
            kind: Selects the minted directory's prefix band (also its
                :data:`PROTECTED_PREFIXES` key, so the reaper never
                reclaims it mid-run).
            sha: Commit-ish to pin the detached worktree at.
            warm_seed: When True AND the warm-lane CoW seed base is
                resolvable (:meth:`_warm_lane_base_resolvable` returns
                :attr:`WarmBaseHealth.OK`), CoW-seeds the minted
                worktree's ``target/`` from the shared warm base via
                :meth:`_seed_warm_lane` (mode ``'--fresh-checkout'``,
                ``take_lane_lock=False`` since this CM already holds
                ``<lane_dir>.lock`` for its own lifetime — see the Note
                below) after a successful add and BEFORE the body runs,
                turning a cold from-scratch build into a warm incremental
                one. Any non-zero seed rc (absent script, disk pressure,
                generic fault) is logged and the CM proceeds COLD — a
                seed fault never breaks the probe (fail-soft, mirrors
                :meth:`create_interactive_worktree`'s
                seed-then-retain-cold-on-fault contract). The whole gate
                (base-health check + seed call) is also wrapped in a
                broad ``except Exception`` so this holds even if an
                unexpected exception escapes those helpers — the gate
                runs before the ``git worktree remove --force`` cleanup
                below, so an unsuppressed raise here would leak the
                registered worktree instead of degrading to cold. A
                non-resolvable base (ABSENT/INDETERMINATE) skips the seed
                subprocess entirely. Default ``False`` keeps
                ``run_main_tip_sweep`` (``MAIN_SWEEP``) byte-identical to
                before this parameter existed.

        Yields:
            The minted worktree path, already checked out at *sha*.

        Raises:
            EphemeralWorktreeError: either (a) the sibling ``<name>.lock``
                flock (task 2507 — see below) was already held by another
                consumer (``fcntl.flock(LOCK_EX|LOCK_NB)`` denied) — raised
                BEFORE ``git worktree add`` is even attempted, so no
                worktree is minted and no add argv is issued; or (b)
                ``git worktree add`` itself failed on all 3 attempts.  In
                both cases the caller's ``async with`` body never runs.
                For (b), because the add never succeeded, no cleanup ``git
                worktree remove`` is issued (there is nothing registered to
                remove) — but a belt-and-suspenders ``shutil.rmtree`` of
                *tmp_path* still runs before the exception propagates, in
                case a failed add left a partial/empty directory behind
                (this prefix is :data:`PROTECTED_PREFIXES`-registered, so
                the reaper would never reclaim a leaked directory itself).

        Note:
            Acquires an exclusive, non-blocking ``fcntl.flock`` on a
            sibling ``<worktree_base>/<kind.value><hex>.lock`` file BEFORE
            minting the worktree, and holds it for the CM's entire
            lifetime (task 2507). This is the exact lock path reify
            warm-lane-gc.sh derives (``${WORKTREES_DIR}/${name}.lock``,
            gc.sh:488/564) for its own ``flock -n`` orphan-removal
            contender, so a live probe/sweep here is correctly seen as a
            "live consumer" by gc.sh and preserved rather than force-
            removed mid-verify. The lock file is unlinked on exit ONLY
            when this call was the one that acquired it (never a foreign
            holder's lock) — see the ``EphemeralWorktreeError`` case above.
        """
        base = self.worktree_base
        base.mkdir(parents=True, exist_ok=True)
        tmp_path = base / f'{kind.value}{uuid.uuid4().hex[:8]}'
        # Sibling flock path — EXACTLY the shape reify warm-lane-gc.sh
        # derives (`${WORKTREES_DIR}/${name}.lock`, gc.sh:488/564). Held
        # for the CM's entire lifetime so gc.sh's `flock -n` orphan-removal
        # contender (gc.sh:564-574) sees a live consumer and preserves this
        # worktree instead of force-removing it out from under a still-
        # running probe/sweep (task 2507).
        lock_path = base / f'{tmp_path.name}.lock'

        _MAX_ADD_RETRIES = 3
        worktree_added = False
        rc, _, err = 1, '', 'not attempted'

        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        acquired = False
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except (BlockingIOError, OSError) as e:
                # Someone else already holds this lock — should never
                # happen in practice (tmp_path's hex is a fresh uuid4 per
                # call), but fail safe rather than proceed unprotected.
                # `acquired` stays False, so the outer finally below closes
                # our fd WITHOUT unlinking the foreign holder's lock file.
                # Deliberately catches bare OSError too, not just
                # BlockingIOError/EWOULDBLOCK: ANY flock failure here means
                # we cannot safely assert liveness, so we fail safe either
                # way. But a bare OSError (e.g. EINTR/EIO/EBADF) is not
                # necessarily lock contention, so the message carries the
                # underlying exception class + text rather than asserting
                # "live consumer" unconditionally — lets an operator tell
                # genuine EWOULDBLOCK contention apart from an OS-level fault.
                raise EphemeralWorktreeError(
                    f'ephemeral_worktree({kind.name}): flock LOCK_NB denied on '
                    f'{lock_path} ({e.__class__.__name__}: {e}) — likely a live '
                    f'consumer holds it (could also be a lock-fault); skipping'
                ) from e

            try:
                for attempt in range(_MAX_ADD_RETRIES):
                    rc, _, err = await _run(
                        ['git', 'worktree', 'add', '--detach', str(tmp_path), sha],
                        cwd=self.project_root,
                    )
                    if rc == 0:
                        worktree_added = True
                        break
                    if attempt < _MAX_ADD_RETRIES - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                if not worktree_added:
                    raise EphemeralWorktreeError(
                        f'ephemeral_worktree({kind.name}): git worktree add failed '
                        f'after {_MAX_ADD_RETRIES} retries (rc={rc}): {err}'
                    )
            except EphemeralWorktreeError:
                # Belt-and-suspenders: a failed `git worktree add` may still have
                # left a partial/empty directory at tmp_path (git creates the
                # target dir early during add) — and since this prefix is
                # PROTECTED_PREFIXES-registered, the reaper will NEVER reclaim it.
                # Clean up here or it leaks under worktree_base permanently.
                # Restores the pre-extraction probes' unconditional
                # "rmtree in case ... the worktree add never ran" guarantee.
                with contextlib.suppress(Exception):
                    shutil.rmtree(tmp_path, ignore_errors=True)
                raise

            # task 2567: optionally CoW-seed the freshly-added worktree's
            # target/ from the shared warm-lane base before the body runs,
            # so a probe/sweep opted into warm_seed starts from a pre-built
            # main instead of a cold from-scratch recompile. Fail-soft: any
            # non-zero seed rc just logs and proceeds COLD — never raises,
            # never removes the worktree. take_lane_lock=False because this
            # CM already holds <lane_dir>.lock (above) for its entire
            # lifetime; re-taking it inside _seed_warm_lane would
            # self-deadlock against the identical path (see that method's
            # take_lane_lock docstring note).
            #
            # task 2567 amendment: the whole gate is wrapped in a broad
            # except so the never-raise contract is structural rather than
            # relying on _warm_lane_base_resolvable() (catches only
            # OSError) and _seed_warm_lane() (catches Exception) each
            # behaving today. This runs BEFORE the `try: yield ... finally:
            # git worktree remove --force` block below, so an unsuppressed
            # raise here would skip that scoped cleanup entirely — leaking
            # a registered git worktree that DD5 forbids reclaiming via a
            # broad `git worktree prune`.
            if worktree_added and warm_seed:
                try:
                    if self._warm_lane_base_resolvable() is WarmBaseHealth.OK:
                        seed_rc = await self._seed_warm_lane(
                            tmp_path, '--fresh-checkout', take_lane_lock=False,
                        )
                        if seed_rc != 0:
                            logger.info(
                                'ephemeral_worktree(%s): warm seed failed (rc=%d) '
                                'for %s — proceeding COLD (fail-soft)',
                                kind.name, seed_rc, tmp_path,
                            )
                    else:
                        logger.debug(
                            'ephemeral_worktree(%s): warm base not resolvable — '
                            'skipping seed, proceeding COLD for %s',
                            kind.name, tmp_path,
                        )
                except Exception:
                    logger.warning(
                        'ephemeral_worktree(%s): warm-seed gate raised '
                        'unexpectedly for %s — proceeding COLD (fail-soft)',
                        kind.name, tmp_path, exc_info=True,
                    )

            try:
                yield tmp_path
            finally:
                # Scoped cleanup: remove ONLY this specific ephemeral worktree.
                # INTENTIONALLY NO 'git worktree prune' here (DD5 guarantee) —
                # see the docstring above / the 2097-2100 incident.
                if worktree_added:
                    try:
                        await _run(
                            ['git', 'worktree', 'remove', '--force', str(tmp_path)],
                            cwd=self.project_root,
                        )
                    except Exception:
                        logger.debug(
                            'ephemeral_worktree(%s): worktree remove failed',
                            kind.name, exc_info=True,
                        )
                with contextlib.suppress(Exception):
                    # Belt-and-suspenders: rmtree in case git worktree remove
                    # left an empty skeleton.
                    shutil.rmtree(tmp_path, ignore_errors=True)
        finally:
            # Release the advisory lock and — ONLY when this call was the
            # one that acquired it — best-effort unlink the lock file,
            # mirroring gc.sh's own `rm -f "$orphan_lock"` (gc.sh:606) on
            # removal. Never unlink a lock we did not acquire: that would
            # delete a foreign holder's lock file out from under it.
            #
            # Unlink BEFORE closing the fd — i.e. while we still hold the
            # flock — rather than after. This mirrors gc.sh, which performs
            # its own analogous `rm -f` while still holding ITS flock, and
            # it closes a window that a close-then-unlink ordering would
            # leave open: between releasing the flock and removing the
            # directory entry, a new contender could open+flock the same
            # (about-to-be-deleted) path, only to have our unlink yank the
            # file out from under it. Unlinking first means any contender
            # that opens the path afterward necessarily creates a fresh
            # inode, so it can never observe a lock we are about to drop.
            if acquired:
                with contextlib.suppress(Exception):
                    os.unlink(lock_path)
            with contextlib.suppress(Exception):
                os.close(lock_fd)

    def pool_in_use(self) -> bool:
        """True iff a warm or spec lane pool is configured on this host (task 2099).

        ``pool_storage_present()``'s only writer (:meth:`_seed_warm_lane` on
        ``rc == 0``) never runs unless a pool is configured, so on a
        pool-less host (``warm_lane_pool=False`` and
        ``merge_spec_warm_lane_pool=False``, the default) ``.pool-root`` is
        NEVER written and ``pool_storage_present()`` is permanently False by
        design — not because a mount went down. The destructive-sweep guards
        gate on ``pool_in_use() and not pool_storage_present()`` rather than
        ``not pool_storage_present()`` alone so that a pool-less host is
        never mistaken for a mount-down incident.
        """
        return self.warm_lane_pool is not None or self.spec_warm_lane_pool is not None

    def pool_storage_present(self) -> bool:
        """True iff worktree_base is backed by live pool storage (task 2099).

        Thin delegator (W11 gamma sentinel fold) to
        ``self._lane_lifecycle.pool_storage_present()`` — the ``.pool-root``
        sentinel FS read now lives only in ``lane_lifecycle.py``.  Public
        contract (fail-safe on ``OSError``, never raises) is unchanged; see
        :meth:`LaneLifecycle.pool_storage_present` for the full rationale.
        """
        return self._lane_lifecycle.pool_storage_present()

    def mark_pool_storage_present(self) -> None:
        """Write the ``.pool-root`` sentinel marking storage as present.

        Thin delegator (W11 gamma sentinel fold) to
        ``self._lane_lifecycle.mark_pool_storage_present()`` — the
        ``.pool-root`` sentinel FS write now lives only in
        ``lane_lifecycle.py``.  Public contract (idempotent, best-effort,
        never raises) is unchanged; see
        :meth:`LaneLifecycle.mark_pool_storage_present` for the full
        rationale.

        Called from exactly ONE chokepoint — :meth:`_seed_warm_lane` on
        ``rc == 0`` — because a successful seed proves the mount is present
        and writable (see that method's docstring for the full rationale).
        """
        self._lane_lifecycle.mark_pool_storage_present()

    def _note_pool_storage_absent(self) -> None:
        """Best-effort dispatch to the injected ``_on_pool_storage_absent`` hook.

        No-op when unwired (default None).  Swallows any exception raised by
        the callback so a guard site can never be broken by escalation-filing
        failing — mirrors the other declare-on-callee callback dispatchers in
        this class.

        **No dedup/debounce here (review-fix, gitops-chokepoints α)**: this
        dispatcher fires on EVERY refusal with no rate-limiting of its own.
        Since :meth:`_prune_registrations` is now the chokepoint for all six
        callers (``prune_worktrees`` plus five converted sweep sites,
        including hot paths like ``create_worktree``'s leftover-branch
        cleanup and ``reap_interactive_worktrees``), a sustained mount-down
        can drive many calls here per tick.  Collapsing those repeats into a
        single operator-visible signal is entirely the installed handler's
        responsibility, not this method's — in production that's
        ``Harness._file_pool_storage_absent_escalation``, which dedupes via
        ``has_open_l1`` so only one pool-storage-absent escalation is ever
        open at a time.  A handler wired without that dedup would see one
        escalation attempt per refused prune.
        """
        if self._on_pool_storage_absent is None:
            return
        try:
            self._on_pool_storage_absent()
        except Exception:
            logger.warning(
                '_note_pool_storage_absent: callback raised', exc_info=True,
            )

    def _pool_storage_bootstrap_ok(self) -> bool:
        """True iff a missing ``.pool-root`` is a first-seed BOOTSTRAP rather
        than an unmounted mountpoint (task 2099 review-fix).

        The ``.pool-root`` sentinel has exactly one writer — :meth:`_seed_warm_lane`
        on ``rc == 0`` — which is reached only AFTER the create-once
        discriminators (:meth:`acquire_warm_lane` / :meth:`acquire_spec_lane`).
        Those discriminators refuse when ``worktree_base.exists() and not
        pool_storage_present()``.  On a genuinely fresh host the sentinel has
        never been written, so absent a bootstrap escape the very first lane
        acquisition is refused forever, the seed never runs, the sentinel is
        never written, and the pool is permanently disabled — a
        chicken-and-egg deadlock.

        This predicate breaks the cycle by recognising the ONE case where a
        missing sentinel is provably safe to seed through: the warm-lane CoW
        seed base resolves ``OK`` (present AND non-empty) AND lives INSIDE
        :attr:`worktree_base`.  A populated base target under worktree_base
        cannot exist on an empty, unmounted mountpoint — so its presence is
        substrate-independent proof that worktree_base's own storage is
        mounted and writable.  When that holds, the create-once site marks the
        sentinel and proceeds (a real seed then re-marks it idempotently).

        Fail-safe (returns False → refuse, never bootstrap) when:
        - the base is ABSENT/INDETERMINATE (empty unmounted mountpoint, or a
          transient stat/readlink hiccup), or
        - the base is configured OFF worktree_base
          (``config.warm_lane_base_target_dir`` points elsewhere) — its
          presence on a different mount says nothing about worktree_base's
          own mount, so a bootstrap there could still create a shadow lane on
          the underlying root fs of an unmounted mountpoint.

        **Spec-only-host caveat (review-fix)**: this same predicate is the
        bootstrap discriminator for :meth:`acquire_spec_lane`, but it always
        proxies through the WARM-lane base (``_warm_lane_base_resolvable`` /
        ``warm_lane_base_target_path``) — there is no spec-pool-specific
        substrate signal.  On a host that configures
        ``spec_warm_lane_pool`` without a ``warm_lane_pool``, this means the
        spec pool's own bootstrap depends on the merge-verify warm base
        being populated: if that base is empty/absent (no warm-lane seed or
        ``refresh_warm_base`` has run yet), this returns False and the spec
        pool's create-once discriminator cold-falls-back indefinitely
        instead of bootstrapping ``.pool-root`` itself, even though
        ``worktree_base`` may in fact be mounted.  This is the conservative
        (never-shadow-an-unmounted-mount) direction, not a correctness bug,
        but it means a spec-only host stays cold-fallback-only until some
        other path (e.g. a warm-lane seed) populates the warm base and
        writes the sentinel.
        """
        if self._warm_lane_base_resolvable() is not WarmBaseHealth.OK:
            return False
        try:
            return self.warm_lane_base_target_path.is_relative_to(self.worktree_base)
        except (OSError, ValueError):
            return False

    def _reconcile_pool_storage_before_sweep(self, context: str) -> bool:
        """Shared pre-sweep gate for the two destructive-sweep sites (task 2315, BUG 2).

        Both :meth:`_run_warm_lane_gc_reclaim` and :meth:`_prune_registrations`
        must refuse to run against an unmounted mountpoint (the Jul-3 task
        2099 incident), but a HEALTHY mount that merely lost its
        ``.pool-root`` sentinel must self-heal rather than refuse forever.
        Pre-2315, one sweep site had no bootstrap escape at all and the
        other only SKIPPED without recreating the sentinel — a
        chicken-and-egg deadlock (sweeps refused -> stale lanes never
        reseeded -> the only sentinel writer, :meth:`_seed_warm_lane` on
        ``rc == 0``, never runs -> sentinel stays missing forever). This
        helper lifts the acquire-side create-once "bootstrap-ok => mark
        sentinel + proceed" pattern (see :meth:`acquire_warm_lane` /
        :meth:`acquire_spec_lane`) into both sweep sites uniformly.

        Args:
            context: Short identifier for the calling sweep, threaded into
                the refusal WARNING so operators can attribute which caller
                asked (mirrors the ``context`` argument already threaded
                through :meth:`_prune_registrations`).

        Returns:
            True  — safe to proceed with the sweep. Either pool storage
                    was never in play (:meth:`pool_in_use` False), the
                    sentinel was already present, or the sentinel was
                    absent but provably a first-seed bootstrap
                    (:meth:`_pool_storage_bootstrap_ok` True) — in that
                    last case the sentinel is recreated
                    (:meth:`mark_pool_storage_present`) before returning.
            False — refuse. The sentinel is absent and NOT provably a
                    bootstrap (a suspected unmount) — :meth:`_note_pool_storage_absent`
                    is invoked to notify the installed callback.

        Pure predicate plus best-effort sentinel recreation; never raises.
        """
        if not (self.pool_in_use() and not self.pool_storage_present()):
            return True
        if self._pool_storage_bootstrap_ok():
            logger.info(
                '%s: .pool-root absent but mount confirmed present at %s '
                '(CoW seed base already resolves underneath it) — '
                'recreating sentinel and proceeding',
                context, self.worktree_base,
            )
            self.mark_pool_storage_present()
            return True
        logger.warning(
            '%s: pool storage absent/unmounted at %s — refusing sweep',
            context, self.worktree_base,
        )
        self._note_pool_storage_absent()
        return False

    def _merge_verify_lease_active(self) -> bool:
        """True iff a merge-verify lease is currently held by a LIVE holder
        (task 2315, BUG 1).

        Reads the holder-pgid rendezvous key recorded by either
        :meth:`merge_verify_lease` (the local in-process span) or the host
        verify-merge CLI (``cli.py:444-512`` — the SAME
        ``write_lock_holder_pgid`` key), then liveness-checks it via
        ``os.killpg(pgid, 0)``.

        Fail-OPEN: a missing, stale (dead), or unreadable holder-pgid is
        treated as NOT held, so a lease can never permanently wedge a
        legitimate caller merely because a prior holder died without
        cleanup. Contrast with the fail-CLOSED *use* of this predicate in
        :meth:`reset_persistent_merge_worktree` (refuses when a DIFFERENT
        live holder holds the lease).
        """
        pgid = read_lock_holder_pgid(self.worktree_base)
        if pgid is None:
            return False
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False  # holder is dead — stale lease, ignore it
        except PermissionError:
            return True  # process exists but we can't signal it — still alive
        except OSError:
            return False  # any other signal failure — fail-open, not held
        return True

    def _lane_lock_self_owned_leak(
        self, lock_path: Path, wait_secs: float, **ctx,
    ) -> LaneLockSelfOwnedLeak | None:
        """Return a :class:`LaneLockSelfOwnedLeak` iff *lock_path* is leaked BY
        US, else ``None`` (task 3081, D8/B13).

        Called only after a bounded-wait acquire has already timed out, to ask
        the one question that timeout cannot answer on its own: is somebody else
        legitimately busy, or did WE leak this lock?

        Three layers, ALL required — each alone yields a false positive, and a
        leak report is a LOUD, human-escalating event:

        1. **Kernel** — our pid is among the FLOCK holders of the lock's inode
           (:func:`~orchestrator.verify_cancel.lane_lock_holder_pids`).  Any
           other holder — reify's ``flock(1)``, a ``verify-merge`` CLI
           subprocess, another orchestrator — is foreign contention and stays on
           the fail-closed path where it belongs.
        2. **Registry** — no fd is registered for that path
           (:func:`_lane_lock_held_in_process`).  Without this, every legitimate
           concurrent in-process holder would be libelled, most sharply
           :meth:`task_verify_lease`, which by design never writes the
           rendezvous layer 3 reads.  This layer gets strictly more important
           as in-process holds widen.
        3. **Liveness** — no live recorded verify
           (:meth:`_merge_verify_lease_active`, reused unchanged with its
           fail-OPEN semantics).  A genuine long verify holding the lane past
           the bounded wait is DF 3003's case: contention, to be deferred
           quietly, not a leak.

        The ``logger.error`` here is the LOUD first-occurrence signal, and is
        deliberately independent of whatever the caller does with the returned
        fault.  On the merge path this exception IS-A
        :class:`MergeVerifyLeaseContended` and so is caught by DF 3003's bounded
        contended-defer arm BEFORE the block-disposition table is ever
        consulted — relying on that row's ``escalate_to_human`` alone would let
        a permanent-until-process-exit leak defer quietly for up to four hours,
        the same outage shape as the incident (~3h to an unattended restart).

        *ctx* forwards ``operation``/``protected_path`` to the fault so it keeps
        the parent's full payload contract.
        """
        holder_pids = lane_lock_holder_pids(lock_path)
        self_pid = os.getpid()
        if self_pid not in holder_pids:
            return None  # layer 1: somebody else holds it — foreign contention
        if _lane_lock_held_in_process(lock_path):
            return None  # layer 2: a registered in-process hold is LIVE
        if self._merge_verify_lease_active():
            return None  # layer 3: a live recorded verify — DF 3003's case
        leak = LaneLockSelfOwnedLeak(
            lock_path,
            wait_secs,
            holder_pids=holder_pids,
            self_pid=self_pid,
            self_pgid=os.getpgrp(),
            holder_pgid=read_lock_holder_pgid(self.worktree_base),
            **ctx,
        )
        logger.error('%s', leak)
        return leak

    @staticmethod
    async def _acquire_lane_flock_off_thread(
        lock_path: Path, wait_secs: float,
    ) -> int | None:
        """Bounded-wait acquire of a lane's ``<lane_dir>.lock`` flock, OFF the
        event loop.

        Shared acquire skeleton of :meth:`merge_verify_lease` and
        :meth:`task_verify_lease` (task 3027): the acquire itself is identical
        in both leases — they diverge only in their timeout constant, their
        fail-mode on a ``None`` return, and (merge only) the holder-pgid write.
        Factored out so the two leases cannot drift on the off-thread acquire.

        Runs via :func:`asyncio.to_thread` because
        :func:`acquire_merge_verify_flock` is a synchronous ``time.sleep`` poll
        whose bounded wait is now minutes, so an inline call would freeze the
        whole orchestrator (mirrors :meth:`reset_persistent_merge_worktree`'s
        off-thread acquire).

        Returns the held fd, or ``None`` if the bounded wait timed out
        (contended). Each lease encodes its OWN policy on that ``None``:
        :meth:`merge_verify_lease` RAISES :class:`MergeVerifyLeaseContended`;
        :meth:`task_verify_lease` fails OPEN (WARNING + proceed).

        A won fd is registered as a LIVE in-process hold (task 3081) before it
        is returned, so :meth:`GitOps._lane_lock_self_owned_leak` can tell this
        legitimate hold from a leaked one — at kernel level the two are
        identical, both being flocks attributed to our pid.
        """
        fd = await asyncio.to_thread(
            acquire_merge_verify_flock, lock_path, wait_secs,
        )
        if fd is not None:
            _register_held_lane_lock(fd, lock_path)
        return fd

    @staticmethod
    def _release_lane_flock(fd: int | None) -> None:
        """Release a lane flock from :meth:`_acquire_lane_flock_off_thread`,
        guarding the fail-open ``None`` case (task 3027).

        A ``None`` fd means the acquire timed out and the lease proceeded
        WITHOUT the hold (:meth:`task_verify_lease`'s fail-open path); calling
        :func:`release_merge_verify_flock` on it would raise an unsuppressed
        ``TypeError`` (``fcntl.flock(None, ...)``). :meth:`merge_verify_lease`
        never reaches its finally with a ``None`` fd (it raises on contention
        first), so the guard is a harmless no-op there — a shared release that
        is safe for both leases.

        Also drops *fd*'s in-process hold registration (task 3081), keeping the
        registry symmetric with :meth:`_acquire_lane_flock_off_thread`.  It is
        forgotten BEFORE the kernel-level release so the registry never claims a
        hold the kernel has already dropped — the ordering that keeps a stale
        entry from masking a genuine leak.
        """
        if fd is not None:
            _forget_held_lane_lock(fd)
            release_merge_verify_flock(fd)

    @contextlib.asynccontextmanager
    async def merge_verify_lease(self, lane_dir: Path | None = None):
        """Async context manager recording the merge-verify lease for the
        duration of a span (task 2315, BUG 1; lock path converged onto the
        shared ``<lane_dir>.lock`` in task 2685).

        Mirrors the host verify-merge CLI's acquire -> write-holder-pgid ->
        finally-release-and-clear span (``cli.py:444-512``) so that
        :meth:`reset_persistent_merge_worktree` and
        :meth:`_run_warm_lane_gc_reclaim` can consult the SAME lease
        (:meth:`_merge_verify_lease_active`) regardless of whether the
        in-flight verify is dispatched locally (in-process, via this ctx
        mgr) or remotely (via the CLI, which already records it).

        Holds the SHARED ``<lane_dir>.lock``
        (:func:`~orchestrator.verify_cancel.lane_lock_path` of
        :attr:`persistent_merge_worktree_path`, i.e.
        ``<worktree_base>/_merge-verify.lock`` for the singleton persistent
        merge lane) — the SAME lock reify's ``seed-warm-lane.sh`` /
        ``thin-warm-lane.sh`` / ``warm-lane-gc.sh`` and DF's own
        :meth:`_seed_warm_lane` take (task 2685; reify PRD
        ``warm-lane-pool-cow-seeding.md`` §9.3/§9.5 inv.11), NOT the
        divergent ``.merge_verify.lock`` (task 2306's original lock). As of
        task 2830 the laptop host verify-merge CLI span ALSO holds this shared
        lane lock as its PRIMARY; it retains ``.merge_verify.lock`` only as a
        transitional rollout co-lock (so an in-flight OLD verify-merge still
        mutually excludes during checkout-sync rollout — see
        ``verify_cancel.merge_verify_lock_path``), and a post-rollout follow-up
        drops that co-lock, leaving the CLI on the lane lock alone (matching
        this lease). Converging onto one lock is what makes a reify/DF reseed,
        thin, or gc of the lane mutually exclude with a live local (or laptop)
        verify — the flock holder-pgid rendezvous below is unchanged.

        *lane_dir* selects WHICH lane's ``<lane_dir>.lock`` inode is flocked.
        Defaults to ``None`` → :attr:`persistent_merge_worktree_path` (the
        singleton persistent merge lane), which keeps the sole no-arg caller
        (``merge_queue.py``'s LOCAL-dispatch guard) and every existing lease
        test byte-identical. A non-``None`` *lane_dir* (e.g. the EPHEMERAL
        ``_merge-<hash>`` speculation worktree the DF 2822 per-land REMOTE-green
        cross-check actually verifies in) flocks THAT lane instead, so the
        cross-check mutually excludes a concurrent reseed/reclaim of its OWN
        lane (task 2873). Only the flocked inode is parametrized — the
        holder-pgid rendezvous below stays keyed to the GLOBAL
        :attr:`worktree_base` (a fail-open liveness hint consumed only by
        persistent-lane actors; an ephemeral-lane lease writing it is a safe
        over-approximation that at worst makes a concurrent persistent
        reseed/GC defer during the cross-check, never a clobber).

        On a contended flock (the bounded wait in
        :func:`acquire_merge_verify_flock` times out after
        ``_MERGE_VERIFY_LEASE_WAIT_SECS``), RAISES
        :class:`MergeVerifyLeaseContended` so the caller DEFERS/requeues the
        dispatch rather than running the verify unprotected (task 2828, limb
        2) — the old behaviour yielded without a lease, letting a 1--2h verify
        race a concurrent reseed/thin/gc clobber. When the kernel attributes
        that contended lock to THIS process with no registered in-process hold
        and no live verify, the raise is instead the
        :class:`LaneLockSelfOwnedLeak` subclass naming our pid/pgid (task 3081,
        D8/B13) — a leaked fd is not contention, and the two were previously
        indistinguishable. The acquire runs OFF the
        event loop via :func:`asyncio.to_thread`, because the now-minutes-long
        synchronous poll would otherwise freeze the orchestrator (mirroring
        :meth:`reset_persistent_merge_worktree`'s off-thread acquire). The
        bounded-wait flock remains the primary cross-process serialization;
        this lease is defense-in-depth on top of it for the DF-side
        teardown/GC actors.
        """
        lock_path = lane_lock_path(
            lane_dir if lane_dir is not None else self.persistent_merge_worktree_path
        )
        # Off-thread bounded-wait acquire (shared skeleton, task 3027):
        # _acquire_lane_flock_off_thread wraps the asyncio.to_thread(
        # acquire_merge_verify_flock, ...) that both leases use identically.
        fd = await self._acquire_lane_flock_off_thread(
            lock_path, _MERGE_VERIFY_LEASE_WAIT_SECS,
        )
        if fd is None:
            # Is this OUR OWN leaked lock rather than somebody else's live
            # hold?  Asked first, because the answer changes the diagnosis
            # entirely (task 3081) — and only ever REPORTS: the refusal below
            # is unchanged either way.
            leak = self._lane_lock_self_owned_leak(
                lock_path, _MERGE_VERIFY_LEASE_WAIT_SECS,
            )
            if leak is not None:
                raise leak
            # Contended past the bounded wait: RAISE so the dispatch is
            # DEFERRED/requeued rather than run unprotected (task 2828, limb
            # 2). The old path yielded without a lease here, letting a 1--2h
            # verify race a concurrent reseed/thin/gc clobber.
            raise MergeVerifyLeaseContended(
                lock_path,
                _MERGE_VERIFY_LEASE_WAIT_SECS,
                holder_facts=_lane_lock_holder_facts(lock_path),
            )
        write_lock_holder_pgid(self.worktree_base, os.getpgrp())
        try:
            yield
        finally:
            remove_lock_holder_pgid(self.worktree_base)
            self._release_lane_flock(fd)

    @contextlib.asynccontextmanager
    async def task_verify_lease(self, lane_dir: Path):
        """Async context manager holding the warm-lane consumer-hold across a
        task-lane verify window (task 3027).

        Holds the SHARED ``<lane_dir>.lock``
        (:func:`~orchestrator.verify_cancel.lane_lock_path` of *lane_dir*, the
        task's warm lane) for the duration of the block — the SAME per-lane
        inode reify's ``seed-warm-lane.sh`` / ``thin-warm-lane.sh`` /
        ``warm-lane-gc.sh`` and DF's own :meth:`_seed_warm_lane` flock. Holding
        it across the task-lane verify makes a concurrent reify
        ``warm-lane-gc.sh reclaim``'s per-lane ``flock -n`` REFUSE/queue (reify
        task 5354, the paired reify-side acquire-time guard) rather than reseed
        a lane whose LIVE consumer is mid-nextest and delete its in-flight test
        binaries out from under it (esc-5236-7 / esc-5275-10, the exit-127
        vanished-artifact storm this closes).

        Deliberately DIFFERENT from :meth:`merge_verify_lease` in two ways:

        * **flock-ONLY** — it does NOT write the merge-verify holder-pgid
          rendezvous (:func:`write_lock_holder_pgid`). That single GLOBAL key
          (keyed to :attr:`worktree_base`) is read fail-CLOSED by
          :meth:`reset_persistent_merge_worktree` and by
          :meth:`_run_warm_lane_gc_reclaim`; many concurrent task-lane verifies
          would stomp it (one lease's finally clearing it while others still
          hold) and would spuriously gate merge-lane resets/GC. The per-lane
          flock alone is the cross-process mechanism reify consults, so the
          rendezvous stays single-purpose to the merge lane.
        * **fail-OPEN on contention** — on acquire timeout (a racing reseed
          held the lane lock past ``_TASK_VERIFY_LEASE_WAIT_SECS``) it logs a
          WARNING and yields WITHOUT the hold, rather than raising. The hold is
          defense-in-depth over reify's per-lane guard; blocking or aborting the
          task's OWN verify would be more disruptive than proceeding, and
          proceeding-without-the-hold is exactly today's baseline (nothing held
          it), so fail-open is strictly non-regressive. Contrast
          :meth:`merge_verify_lease`, which RAISES
          :class:`MergeVerifyLeaseContended` (safe there because the merge
          worker cleanly requeues the dispatch; a task verify has no such clean
          requeue and must never be blocked by its own lane lease).

        No change is made to :meth:`_run_warm_lane_gc_reclaim`: it honors the
        hold transitively by invoking reify's ``warm-lane-gc.sh``, which does
        the per-lane ``flock -n`` check itself.
        """
        lock_path = lane_lock_path(lane_dir)
        # Off-thread bounded-wait acquire (shared skeleton with
        # merge_verify_lease, task 3027): _acquire_lane_flock_off_thread wraps
        # the asyncio.to_thread(acquire_merge_verify_flock, ...) both leases use.
        fd = await self._acquire_lane_flock_off_thread(
            lock_path, _TASK_VERIFY_LEASE_WAIT_SECS,
        )
        if fd is None:
            # Contended past the bounded wait: FAIL OPEN. Proceed WITHOUT the
            # hold rather than raise/block — the task verify must never be
            # aborted by its own lane lease, and running unprotected is exactly
            # today's baseline (nothing held it). Only the rare racing reseed
            # holds this lane lock, so this WARNING is the diagnosable signal
            # that the defense-in-depth hold was skipped for this run.
            logger.warning(
                'task_verify_lease: contended lane lock %s (no acquire within '
                '%.1fs) — proceeding without the hold (fail-open)',
                lock_path, _TASK_VERIFY_LEASE_WAIT_SECS,
            )
        try:
            yield
        finally:
            # Shared None-guarded release (task 3027): on the fail-open path
            # fd is None and _release_lane_flock skips the release, avoiding the
            # unsuppressed TypeError from fcntl.flock(None, ...).
            self._release_lane_flock(fd)

    async def _is_registered_worktree(self, path: Path) -> bool:
        """Check if *path* is a registered git worktree.

        Uses ``git worktree list --porcelain`` and matches by **canonical
        (resolved) path on both sides** — each listed ``worktree <path>``
        is ``Path.resolve()``-d and compared against ``path.resolve()``.
        This recognizes a registration recorded under a *symlink* path
        (e.g. reify's ``.worktrees`` symlinked to a mount after migration,
        esc-4146-268) that an exact-string compare would miss, while still
        rejecting stale directories (containing only .task/ state files)
        that were never registered.
        """
        resolved = path.resolve()
        rc, output, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=self.project_root,
        )
        if rc != 0:
            return False  # fail-safe is provided by the destroy gate in create_worktree
        for line in output.splitlines():
            if line.startswith('worktree ') and Path(line[9:]).resolve() == resolved:
                return True
        return False

    async def _repair_orphaned_reuse_lane(self, lane: Path, branch_name: str) -> bool:
        """Attempt in-place recovery of a REUSE lane whose worktree registration
        was lost (task 2097).

        Runs ``git worktree repair <lane>`` from ``project_root`` (cheapest
        recovery — restores a stale/broken registration WITHOUT touching
        ``.task/plan.json`` or uncommitted WIP) and re-checks registration
        with ``_is_registered_worktree`` REGARDLESS of the repair
        subprocess's exit code (review, task 2097): exit code alone is not
        treated as authoritative, since what the caller ultimately cares
        about — and bases its routing decision on — is whether the lane is
        ACTUALLY registered afterwards, not whether the repair subprocess
        happened to report success. Checking unconditionally also means a
        False return here is a fresh, just-taken confirmation, not a stale
        snapshot — safe for the caller to treat as decisive without a
        further re-probe of its own.

        Returns:
            True when the lane is a registered worktree afterwards (safe to
            proceed with the normal reuse path); False when the caller must
            drop the assignment and fall through to the create-once
            self-heal/reattach path.

        NOTE: cannot reconstruct a fully-deleted ``.git/worktrees/<name>``
        admin dir (``git worktree prune`` wipe) — returns False there, and
        reattach recovers the committed work instead.
        """
        if not lane.exists():
            return False
        rc, _, err = await _run(
            ['git', 'worktree', 'repair', str(lane)], cwd=self.project_root,
        )
        if await self._is_registered_worktree(lane):
            logger.info(
                'acquire_warm_lane: lane %s is a registered worktree after '
                '`git worktree repair` (rc=%d) — registration is the '
                'authoritative signal, not exit code (review, task 2097); '
                '.task/ + WIP preserved',
                lane, rc,
            )
            return True
        logger.warning(
            'acquire_warm_lane: git worktree repair could not restore lane %s '
            'for branch %s (rc=%d: %s) — dropping assignment, routing to '
            'create-once reattach',
            lane, branch_name, rc, err.strip(),
        )
        return False

    async def _freshen_main(self) -> tuple[str, int | None]:
        """Fetch from remote and return the freshest ref to use as worktree base.

        Returns:
            (ref, stale_commits) where:
            - ref: the git ref to pass to ``git worktree add`` / ``git rev-parse``
            - stale_commits: None  → fetch failed (no remote configured)
                             0     → local main is already current with remote
                             N > 0 → local main was N commits behind remote

        Design decisions:
        - Best-effort fetch: if fetch fails (no remote in tests), return
          (main_branch, None) silently — matches the pattern in
          _create_merge_worktree (line 578).
        - No mutation of local main ref: advance_main() uses CAS on the local
          main ref; updating it here could cause spurious CAS failures.  We
          return the remote-tracking ref (origin/main) as the start-point
          instead.
        - Divergence guard: if local main has commits not in origin/main (e.g.
          from advance_main calls not yet pushed), using origin/main would lose
          those commits.  In the diverged case we fall back to local main and
          log a warning.
        """
        remote_ref = f'{self.config.remote}/{self.config.main_branch}'

        # Best-effort fetch — silently ignore failure (no remote in tests)
        rc, _, _ = await _run(
            ['git', 'fetch', self.config.remote, self.config.main_branch],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.debug(
                '_freshen_main: fetch from %s failed — using local %s',
                self.config.remote, self.config.main_branch,
            )
            return self.config.main_branch, None

        # Count commits local main is BEHIND origin/main
        rc, behind_out, _ = await _run(
            ['git', 'rev-list', '--count',
             f'{self.config.main_branch}..{remote_ref}'],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(
                '_freshen_main: rev-list (behind) failed (rc=%d) — using local %s',
                rc, self.config.main_branch,
            )
            return self.config.main_branch, None
        try:
            behind = int(behind_out.strip())
        except ValueError:
            logger.warning(
                '_freshen_main: unexpected behind-count output: %r', behind_out,
            )
            return self.config.main_branch, None

        if behind == 0:
            return self.config.main_branch, 0

        # Check for divergence: count commits local main is AHEAD of origin/main
        rc, ahead_out, _ = await _run(
            ['git', 'rev-list', '--count',
             f'{remote_ref}..{self.config.main_branch}'],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(
                '_freshen_main: rev-list (ahead) failed (rc=%d) — using local %s',
                rc, self.config.main_branch,
            )
            return self.config.main_branch, behind
        try:
            ahead = int(ahead_out.strip())
        except ValueError:
            logger.warning(
                '_freshen_main: unexpected ahead-count output: %r', ahead_out,
            )
            # Fall back to local main; report behind count as-is (ref is local, not remote)
            return self.config.main_branch, behind

        if ahead > 0:
            logger.warning(
                '_freshen_main: local %s diverged from %s (%d ahead, %d behind) '
                '— using local ref to avoid losing advance_main commits',
                self.config.main_branch, remote_ref, ahead, behind,
            )
            return self.config.main_branch, behind

        # Strictly behind: use remote-tracking ref as worktree start-point
        logger.info(
            '_freshen_main: local %s is %d commits behind %s — using %s',
            self.config.main_branch, behind, remote_ref, remote_ref,
        )
        return remote_ref, behind

    async def _train_predecessor(self, train: TrainMembership) -> TrainPredecessor:
        """Resolve the predecessor for a train member with order > 0.

        Reads train['members'][order - 1] and derives its branch name using
        the configured branch_prefix.  Raises ValueError when invariants are
        violated (order <= 0, members absent/None, members too short).
        """
        order = train.get('order', 0)
        if order <= 0:
            raise ValueError(
                f'_train_predecessor called with order={order!r}; '
                'must only be called when order > 0'
            )
        members = train.get('members')
        if not members or not isinstance(members, list):
            raise ValueError(
                f'_train_predecessor: members is {members!r}; '
                'expected a non-empty list of task ids'
            )
        if len(members) < order:
            raise ValueError(
                f'_train_predecessor: members has {len(members)} entries but '
                f'order={order} requires at least {order} entries; members={members!r}'
            )
        predecessor_id = str(members[order - 1])
        return TrainPredecessor(
            task_id=predecessor_id,
            branch=f'{self.config.branch_prefix}{predecessor_id}',
        )

    async def _resolve_predecessor_tip(
        self, train: TrainMembership, branch_name: str
    ) -> tuple[str, str]:
        """Resolve predecessor branch tip SHA and branch name for a train member.

        Returns ``(predecessor_sha, predecessor_branch)`` where
        *predecessor_sha* is the 40-char tip SHA of the predecessor's branch
        and *predecessor_branch* is its full branch name.

        Raises ``RuntimeError`` when the predecessor branch does not exist.
        Both the initial-create path and the reuse path share this invariant:
        the predecessor must be present before any successor can be created or
        requeued.  Centralising the guard here prevents the two callers from
        drifting apart — the pre-refactor duplication would have allowed the
        two error messages (and their None-guard logic) to diverge, silently
        reintroducing the main-rebase corruption this task fixes.
        """
        predecessor = await self._train_predecessor(train)
        predecessor_sha = await self.resolve_branch_sha(predecessor.branch)
        if predecessor_sha is None:
            raise RuntimeError(
                f'create_worktree: predecessor branch {predecessor.branch!r} '
                f'does not exist (train_id={train.get("id")!r}, '
                f'order={train.get("order")}, branch_name={branch_name!r}). '
                'The predecessor branch must exist before any successor can be '
                'created or requeued.'
            )
        return predecessor_sha, predecessor.branch

    async def disable_shared_repo_auto_maintenance(self) -> None:
        """Disable git auto-gc/maintenance on this orchestrator-managed shared repo.

        PRD plans/os-sandbox-worktree-containment-prd.md task α5 (D2 corollary):
        under the OS-sandbox narrow shared-.git write-set (α2: the .git root and
        packed-refs are read-only), background auto-gc/maintenance would fail
        benignly-but-noisily.  Set ``gc.auto=0`` and ``maintenance.auto=false``
        repo-locally (``.git/config``) so it never fires; gc ownership moves
        out-of-band to the orchestrator/operator (see
        docs/shared-repo-git-maintenance.md).

        Idempotent: ``git config`` overwrites the value in place, so calling this
        repeatedly (harness startup + every create_worktree) leaves the same
        result.  Best-effort/loud: a non-zero git rc is logged at WARNING but
        never raised — failing to set the key merely leaves auto-gc enabled
        (itself only a benign-but-noisy failure), which must not block
        orchestrator startup or a task dispatch (loud-over-silent-degradation).
        """
        for key, value in (('gc.auto', '0'), ('maintenance.auto', 'false')):
            rc, _, stderr = await _run(
                ['git', 'config', key, value], cwd=self.project_root,
            )
            if rc != 0:
                logger.warning(
                    'disable_shared_repo_auto_maintenance: failed to set '
                    '%s=%s on %s (rc=%s): %s',
                    key, value, self.project_root, rc, stderr.strip(),
                )

    async def create_worktree(
        self,
        branch_name: str,
        *,
        expected_title: str | None = None,
        train: TrainMembership | None = None,
    ) -> WorktreeInfo:
        """Create a git worktree for a task branch, based off main.

        Returns a WorktreeInfo with the worktree path and the base commit SHA
        (main's SHA at creation time) so diffs remain stable even if main
        advances during task execution.

        ``train`` — when supplied and ``train['order'] > 0``, the worktree is
        branched from the prior train member's branch tip instead of
        ``origin/main``.  See PRD § 9.4 for the full train-branching spec.
        ``train=None`` (default) and ``train['order'] == 0`` both fall through
        to the existing ``_freshen_main()`` path unchanged.

        If the worktree/branch already exist (e.g., from a requeued task),
        reuses them instead of failing — UNLESS ``expected_title`` is supplied
        and the existing worktree's stored title fails to match it (a recycled
        task id whose orphaned worktree holds unrelated WIP).  On mismatch the
        stale worktree is quarantined and a fresh one is created instead.
        ``expected_title=None`` (the default) skips this guard entirely, so all
        existing callers/tests are unaffected.
        """
        worktree_path = self.worktree_base / branch_name
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        full_branch = f'{self.config.branch_prefix}{branch_name}'

        # ── Ensure core.hooksPath is set ──────────────────────────────
        # The pre-commit hook in hooks/pre-commit strips .task/ from the
        # staging area on ALL branches.  core.hooksPath must point to
        # hooks/ (relative) so worktrees find the hook via their own
        # working tree.  This is idempotent — safe to run every time.
        await _run(
            ['git', 'config', 'core.hooksPath', 'hooks'],
            cwd=self.project_root,
        )

        # ── Disable auto-gc/maintenance on the shared repo ─────────────
        # PRD os-sandbox α5 (D2 corollary): reassert gc.auto=0 /
        # maintenance.auto=false on every worktree-create so background
        # auto-gc never fires under the narrow shared-.git write-set (and
        # any config drift/re-clone is re-covered).  Idempotent & best-effort
        # (never raises) — same "safe to run every time" shape as the
        # core.hooksPath block above.
        await self.disable_shared_repo_auto_maintenance()

        # ── Resolve start-ref: train-predecessor tip or freshened main ──
        # PRD § 9.4: when a train member has order > 0, branch from the prior
        # member's branch tip so the chain is contiguous.  order=0 (degenerate
        # train) and train=None both fall through to _freshen_main().
        if train is not None and train.get('order', 0) > 0:
            # ── Train path: branch from predecessor's tip ─────────────────
            # PRD § 9.4: resolve the predecessor's branch and use its tip SHA
            # as start_ref so the new worktree stacks directly on top.
            # Guard (raise RuntimeError on None) is in _resolve_predecessor_tip.
            start_ref, _ = await self._resolve_predecessor_tip(train, branch_name)
            stale_commits = None  # "behind remote" does not apply to sibling branches
        else:
            # ── Freshen main from remote (best-effort) ────────────────────
            # If origin/main has advanced since session start, use the remote-
            # tracking ref as the worktree base so agents start from the freshest
            # code.  Falls back to local main silently when no remote is configured
            # (e.g. in test repos).  Never mutates the local main ref — that would
            # interfere with advance_main's CAS logic.
            start_ref, stale_commits = await self._freshen_main()
        logger.info(
            'create_worktree: freshening result: ref=%s, stale_commits=%s',
            start_ref, stale_commits,
        )

        # Capture the freshened ref's SHA (used as stable base for diffs)
        rc, base_sha, _ = await _run(
            ['git', 'rev-parse', start_ref],
            cwd=self.project_root,
        )
        if rc != 0:
            if train is not None and train.get('order', 0) > 0:
                # start_ref was a SHA just verified by resolve_branch_sha; if
                # rev-parse fails here it indicates git state corruption, not a
                # missing remote ref.  Falling back to main would silently violate
                # the train-stacking invariant, so raise instead.
                raise RuntimeError(
                    f'create_worktree: rev-parse of confirmed predecessor SHA '
                    f'{start_ref!r} failed (rc={rc}); this is unexpected — '
                    f'the SHA was just resolved by resolve_branch_sha and should '
                    f'be stable'
                )
            logger.warning(
                'create_worktree: rev-parse %s failed (rc=%d) — falling back to local %s',
                start_ref, rc, self.config.main_branch,
            )
            start_ref = self.config.main_branch
            rc, base_sha, _ = await _run(
                ['git', 'rev-parse', start_ref],
                cwd=self.project_root,
            )
            if rc != 0:
                raise RuntimeError(
                    f'create_worktree: rev-parse of local {start_ref} also failed (rc={rc})'
                )

        # ── Warm-lane pool (ζ): try to allocate a pre-seeded lane ──────────
        # Only for non-train, non-reuse-by-name fresh dispatches.  Train-
        # stacked members (predecessor-tip branching) and identity-guard
        # reuse-by-name have bespoke logic below; pooling them is out of ζ's
        # scope and the cold path stays correct.
        # Pool exhaustion / absent seed script / seed failure all fall through
        # to the unchanged cold path below (inv.6: never block/deadlock).
        if (
            self.warm_lane_pool is not None
            and (train is None or train.get('order', 0) == 0)
            and not await self._is_registered_worktree(worktree_path)
        ):
            pool_info = await self.acquire_warm_lane(
                branch_name, start_ref, expected_title=expected_title,
            )
            if isinstance(pool_info, WorktreeInfo):
                # Carry stale_commits from the freshen result onto the pool info
                return WorktreeInfo(
                    path=pool_info.path,
                    base_commit=pool_info.base_commit,
                    stale_commits=stale_commits,
                    reify_debug_port=pool_info.reify_debug_port,
                )
            # β: pool is enabled — no cold-path fall-through.  Route discriminant
            # to the appropriate exception so callers get typed failure signals.
            if pool_info is WarmLaneUnavailable.EXHAUSTED:
                # Task 2984 (PRD α): APPEND the typed census to the message
                # (prefix preserved so bare pytest.raises sites with no match=
                # stay green) via the single render() shared with the WARNING
                # at the EXHAUSTED return.
                census = self._assemble_warm_lane_census()
                raise WarmLanePoolExhausted(
                    f'warm-lane pool exhausted for branch {branch_name!r}; '
                    f'requeue — {census.render()}'
                )
            if pool_info is WarmLaneUnavailable.DISK_PRESSURE:
                raise WarmLaneDiskPressure(
                    f'warm-lane seed disk pressure for branch {branch_name!r}; requeue'
                )
            if pool_info is WarmLaneUnavailable.SOFT_PRESSURE:
                raise WarmLaneSoftPressure(
                    f'warm-lane soft-floor backpressure for branch {branch_name!r}; '
                    f'defer/requeue'
                )
            if pool_info is WarmLaneUnavailable.BASE_ABSENT:
                raise WarmLanePoolHardDown(
                    f'warm-lane base absent (host-scoped pool hard-down) for '
                    f'branch {branch_name!r}; requeue (fail-open) — run '
                    f'reify/scripts/ensure-warm-base.sh'
                )
            if pool_info is WarmLaneUnavailable.RESEED_CONTAMINATED:
                # Data-integrity / reseed-consistency fault (task 2854): the
                # fresh reseed left the lane carrying a prior occupant's
                # commits. Requeue (WarmLaneRequeue) to re-acquire a DIFFERENT
                # lane rather than dispatch onto the stale tree — never a
                # cold-path fall-through onto the contaminated content.
                raise WarmLaneReseedContaminated(
                    f'warm-lane reseed contamination for branch {branch_name!r} '
                    f"(lane retained a prior occupant's commits beyond base); "
                    f'requeue to re-acquire a different lane (task 2854)'
                )
            # FAULT or DISABLED → RuntimeError reuses existing blocked+L1 plumbing.
            # DISABLED is a programming error (caller bypassed the pool-enabled
            # guard); it is treated as a fault here so blocked+L1 surfaces the
            # bug rather than silently requeueing forever.
            raise RuntimeError(
                f'warm-lane acquire fault for branch {branch_name!r} '
                f'(seed/worktree-add failure, absent seed script, or pool disabled)'
            )

        # If worktree already exists, reuse it (common after requeue) —
        # but ONLY if it is a real registered git worktree.  A stale
        # directory (e.g. containing only .task/ state files from a previous
        # run) must be removed so a fresh worktree can be created.
        if worktree_path.exists():
            reuse_ok = await self._is_registered_worktree(worktree_path)
            # ── Identity guard (Fix C, defense-in-depth) ──────────────
            # A registered worktree whose stored title does not match the
            # live task's title is a recycled-id collision: the dir name
            # equals the new task's numeric id but the contents belong to a
            # deleted task.  Quarantine it (preserving its WIP) and fall
            # through to a fresh create.  identities_match fails open, so a
            # title-less legacy worktree is reused as before.
            if reuse_ok and expected_title is not None:
                stored_title = read_worktree_title(worktree_path)
                if not identities_match(stored_title, expected_title):
                    logger.warning(
                        'create_worktree: reuse identity MISMATCH for %s — '
                        'stored title %r != expected %r; quarantining and '
                        'creating fresh',
                        worktree_path, stored_title, expected_title,
                    )
                    await self.quarantine_worktree(
                        worktree_path, branch_name, 'reuse-identity-mismatch',
                    )
                    reuse_ok = False
            if reuse_ok:
                logger.info(f'Reusing existing worktree at {worktree_path} on branch {full_branch}')

                # Save any uncommitted tracked work before rebasing
                # (.task/ is gitignored so plan.json is unaffected)
                await self.commit(
                    worktree_path,
                    'chore: save WIP before requeue rebase',
                )

                # ── Resolve rebase target: train-predecessor tip or main ──────
                # Mirror the create-path start-ref logic (lines 777-791): a
                # stacked train member (order > 0) must rebase onto its
                # predecessor's tip so the stacking invariant is preserved
                # across requeues.  Rebasing onto main (the old unconditional
                # behaviour) corrupts the train by re-parenting the delta off
                # main instead of the predecessor's commits.
                if train is not None and train.get('order', 0) > 0:
                    # Guard (raise RuntimeError on None) is in _resolve_predecessor_tip.
                    rebase_target, base_ref = await self._resolve_predecessor_tip(
                        train, branch_name
                    )
                else:
                    rebase_target: str | None = None  # rebase_onto_main defaults to main
                    base_ref: str = self.config.main_branch

                # Rebase onto the resolved target (predecessor tip for stacked
                # trains, main for non-train / order==0).  Critical for plan
                # revalidation: the architect needs to see current file
                # contents from the right base.
                #
                # rebase_preserving_task_commits (not the bare primitive):
                # guards against a silent branch-reset wipe (task 2403) — see
                # its docstring.  A BranchResetError raised here propagates
                # out of create_worktree to _drive()'s exception handler.
                if await self.rebase_preserving_task_commits(worktree_path, onto=rebase_target):
                    # Re-capture base from worktree's own merge-base after the
                    # rebase completes.  merge-base from inside the worktree
                    # is race-immune to concurrent base-ref advances during
                    # rev-parse / rebase.
                    _, mb_out, _ = await _run(
                        ['git', 'merge-base', base_ref, 'HEAD'],
                        cwd=worktree_path,
                    )
                    actual_base = mb_out.strip() or base_sha.strip()
                else:
                    # Rebase failed (conflict) — continue on old base.
                    # Compute the actual merge-base so WorktreeInfo is truthful.
                    _, mb_out, _ = await _run(
                        ['git', 'merge-base', base_ref, 'HEAD'],
                        cwd=worktree_path,
                    )
                    actual_base = mb_out.strip() or base_sha.strip()
                    if train is not None and train.get('order', 0) > 0:
                        # For stacked train members a rebase conflict means the
                        # stacking invariant is broken — log at ERROR so the
                        # degradation is visible rather than silently shipped on
                        # a stale base.
                        logger.error(
                            'Rebase conflict for stacked train member %s '
                            '(train_id=%r, order=%s) — continuing on stale '
                            'base %s; stack integrity is degraded.',
                            worktree_path, train.get('id'),
                            train.get('order'), actual_base[:8],
                        )
                    else:
                        logger.warning(
                            'Rebase failed for reused worktree %s — continuing '
                            'on old base %s',
                            worktree_path, actual_base[:8],
                        )

                # Re-run on reuse so the requeued agent re-acquires a free
                # port and re-patches its .mcp.json.  The script must be
                # idempotent (return the same port for the same worktree dir)
                # to avoid leaking ports across requeues — see the docstring
                # of _provision_reify_debug_port for the full contract.
                port = await self._provision_reify_debug_port(worktree_path)
                return WorktreeInfo(
                    path=worktree_path,
                    base_commit=actual_base,
                    stale_commits=stale_commits,
                    reify_debug_port=port,
                )
            elif worktree_path.exists():
                # The directory exists but git does not recognize it as a
                # registered worktree.  Two very different cases share this
                # branch, and conflating them is what destroyed live work in
                # esc-4146-268 (the silent rmtree below):
                #   (a) a genuinely stale leftover — only .task/ residue (and/
                #       or empty), no .git link, no branch — safe to remove;
                #   (b) a de-registered LIVE worktree whose admin entry was
                #       lost (reify's symlink migration, or a stray prune) —
                #       its .git link or source files are still on disk, or
                #       its task branch still carries commits.  Deleting it
                #       destroys work, including the gitignored .task/plan.json
                #       that git cannot restore.
                # Discriminate git-independently (so the gate still holds under
                # ENOSPC / total git failure): a .git link present, or content
                # beyond .task/, or a branch with commits beyond main => live.
                # Mirror _cleanup_leftover_branch: raise rather than delete
                # anything live (RuntimeError from create_worktree routes to
                # blocked + L1, non-stranding via Harness Fix #1a — see below).
                entries = {p.name for p in worktree_path.iterdir()}
                has_git_link = '.git' in entries
                has_substantive_content = bool(entries - {'.task', '.git'})
                rc_v, _, _ = await _run(
                    ['git', 'rev-parse', '--verify', full_branch],
                    cwd=self.project_root,
                )
                branch_has_work = (
                    rc_v == 0
                    and await self._branch_has_commits_beyond_main(full_branch)
                )
                if has_git_link or has_substantive_content or branch_has_work:
                    raise RuntimeError(
                        f'create_worktree: refusing to delete directory '
                        f'{worktree_path} — it looks like a live worktree whose '
                        f'git registration was lost (a .git link is present, '
                        f'source files exist, or branch {full_branch!r} carries '
                        f'commits beyond {self.config.main_branch}). Deleting '
                        f'would destroy work, including the gitignored .task/ '
                        f'plan state that git cannot restore. Recover by '
                        f're-registering it (`git worktree repair '
                        f'{worktree_path}`) and re-dispatching, or quarantine it '
                        f'manually once any wanted work is preserved. (fail-safe; '
                        f'was the silent rmtree at git_ops.py:702)'
                    )
                if self._refuse_foreign_band(
                    worktree_path, frozenset(), 'create_worktree-self-heal',
                ):
                    # Refused: the WARNING has already been emitted by the
                    # helper. This site's legitimate target is always a
                    # non-band task worktree, so this branch is unreachable
                    # in real usage — pure defense-in-depth. Never delete a
                    # protected band here; leave it in place (the `git
                    # worktree add` below will then fail loudly on the
                    # still-non-empty directory rather than silently
                    # destroying foreign-band content).
                    pass
                else:
                    logger.warning(
                        f'Directory {worktree_path} exists but is NOT a registered '
                        f'git worktree, and holds no live work (no .git link, only '
                        f'.task/ residue, branch has no commits beyond '
                        f'{self.config.main_branch}) — removing stale directory and '
                        f'creating fresh worktree'
                    )
                    shutil.rmtree(worktree_path)

        # If the branch ref already exists (stale from a previous run, or — the
        # 3576 trigger — still checked out in a leftover worktree), clean it up
        # ONLY when deterministically non-destructive.  The old code ran a blind
        # `git branch -D` and ignored its rc: when the branch was checked out in
        # a leftover worktree the delete silently failed, then `git worktree add`
        # raised the opaque "a branch named ... already exists" (2026-05-29).
        # Worse, a blind delete of a branch carrying commits beyond main would
        # have destroyed orphan work.  Hard rule: never delete uncommitted WIP
        # or orphan commits — prove the cleanup is non-destructive, else raise
        # (→ blocked + L1, now non-stranding via Harness Fix #1a).
        rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', full_branch],
            cwd=self.project_root,
        )
        if rc == 0:
            # ── γ reattach guard (cold path) ──────────────────────────────
            # Mirrors acquire_warm_lane's create-once γ reattach site: if the
            # leftover branch's worktree dir is gone (e.g. the stranded-in-
            # progress reconciler's stale-lock path reaped it via
            # cleanup_worktree, which retains a branch carrying commits
            # beyond main) AND the branch still carries commits beyond main,
            # re-attach and resume it instead of raising via
            # _cleanup_leftover_branch. _orphan_has_commits is existence-
            # gated (False for a nonexistent/zero-commit branch), so brand-
            # new task ids fall through to _cleanup_leftover_branch /
            # fresh-create unchanged.
            if not worktree_path.exists() and await self._orphan_has_commits(full_branch):
                return await self._reattach_cold_worktree(
                    worktree_path, full_branch, stale_commits,
                )
            await self._cleanup_leftover_branch(full_branch, branch_name)

        # Create worktree with new branch from the freshened ref
        rc, out, err = await _run(
            ['git', 'worktree', 'add', '-b', full_branch, str(worktree_path), start_ref],
            cwd=self.project_root,
        )
        if rc != 0:
            raise RuntimeError(f'Failed to create worktree: {err}')

        logger.info(
            'Created worktree at %s on branch %s (base=%s, stale_commits=%s)',
            worktree_path, full_branch, base_sha[:8], stale_commits,
        )

        # Re-capture base from the worktree's own merge-base after
        # positioning.  merge-base from inside the freshly-created worktree
        # is race-immune to concurrent main advances between rev-parse and
        # `git worktree add`: it is the fork point of HEAD with the
        # freshened start_ref regardless of when main advanced.  We use
        # start_ref (the ref the worktree was actually based on — may be
        # origin/main when local main lags) rather than
        # self.config.main_branch, so the freshen-from-remote semantic is
        # preserved (see test_create_worktree_freshens_from_remote).
        _, mb_out, _ = await _run(
            ['git', 'merge-base', start_ref, 'HEAD'],
            cwd=worktree_path,
        )
        post_create_base = mb_out.strip() or base_sha.strip()
        port = await self._provision_reify_debug_port(worktree_path)
        return WorktreeInfo(
            path=worktree_path,
            base_commit=post_create_base,
            stale_commits=stale_commits,
            reify_debug_port=port,
        )

    async def _reattach_cold_worktree(
        self, worktree_path: Path, full_branch: str, stale_commits: int | None,
    ) -> 'WorktreeInfo':
        """Re-attach ``worktree_path`` to a surviving orphan ``full_branch``.

        Called by :meth:`create_worktree`'s cold path (the γ reattach guard)
        when the worktree directory is gone but the branch still carries
        commits beyond main — the reaped-but-retained shape left behind by
        ``cleanup_worktree`` -> ``_delete_branch_if_on_main``. Mirrors
        :meth:`acquire_warm_lane`'s create-once γ reattach site
        (``git worktree add`` with no ``-b``, raise-not-destroy on failure,
        delegate the resume tail to :meth:`_reuse_warm_lane`) minus the
        warm-lane-pool-only seeding/route-recording calls, which do not apply
        to the cold (unpooled) path.

        Base-ref divergence: unlike :meth:`create_worktree`'s fresh-create
        tail — which computes its merge-base off the freshened ``start_ref``
        (may be ``origin/main`` when local main lags, preserving the
        freshen-from-remote semantic; see
        ``test_create_worktree_freshens_from_remote``) — the resume tail
        here (:meth:`_reuse_warm_lane`) rebases onto ``self.config.
        main_branch`` (LOCAL main only). A resumed cold worktree may
        therefore start from a staler base than a worktree freshly created
        in the same dispatch tick. This matches :meth:`acquire_warm_lane`'s
        γ reattach tail exactly (same tradeoff, same delegate), so cold and
        warm re-attach stay consistent with each other — it just means a
        cold re-attach is not guaranteed to reflect a same-tick remote
        fetch the way a fresh create would.

        Raises:
            RuntimeError: ``git worktree add`` failed (e.g. *full_branch* is
                still checked out in another live worktree). The branch is
                left intact; the caller's RuntimeError routes to blocked+L1
                (non-stranding via Harness Fix #1a) rather than destroying
                anything.

        Returns:
            WorktreeInfo for the resumed worktree, with *stale_commits*
            carried over from the freshen result (mirrors create_worktree's
            reuse-existing path).
        """
        logger.info(
            'create_worktree: cold-path reattach — orphan %s has commits; '
            'attaching %s without -b',
            full_branch, worktree_path,
        )
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        rc, _, err = await _run(
            ['git', 'worktree', 'add', str(worktree_path), full_branch],
            cwd=self.project_root,
        )
        if rc != 0:
            raise RuntimeError(
                f'create_worktree: refusing to reset {full_branch!r} — it '
                f'carries commits beyond {self.config.main_branch} and '
                f'cannot be safely re-attached to {worktree_path} (git '
                f'worktree add failed: {err.strip()!r}). This would destroy '
                f'work. Inspect the branch and, once any wanted work is '
                f'preserved, remove the other worktree and retry: '
                f'`git branch -D {full_branch}` only after preserving work.'
            )
        info = await self._reuse_warm_lane(worktree_path, full_branch)
        return WorktreeInfo(
            path=info.path,
            base_commit=info.base_commit,
            stale_commits=stale_commits,
            reify_debug_port=info.reify_debug_port,
        )

    async def create_interactive_worktree(
        self, slug: str, *, start_ref: str | None = None,
    ) -> 'InteractiveWorktreeInfo':
        """Mint an isolated interactive warm-worktree in the ``_iact-*`` band.

        Creates a fresh worktree at ``worktree_base/<iact_prefix><slug>`` on
        branch ``<branch_prefix><slug>``, based on *start_ref* (or the local
        ``main_branch`` tip, rev-parsed deterministically — no remote fetch),
        then CoW-seeds its ``target/`` by reusing :meth:`_seed_warm_lane`.

        **Isolation invariant I1**: this method NEVER references
        ``self.warm_lane_pool`` / ``self.spec_warm_lane_pool`` — the
        ``_iact-*`` band this method creates is structurally disjoint from
        the ``_lane-*`` dispatch pool and the ``_spec-*`` merge-speculation
        pool.

        **Branch namespace** (NOT disjoint): the branch is
        ``f'{branch_prefix}{slug}'`` — the SAME namespace shared with
        dispatch task branches and every other interactive slug. Only the
        on-disk ``_iact-*`` directory band is dedicated; the branch
        namespace is not. A leftover ``branch_prefix``-namespaced branch for
        this slug (a live dispatch task reusing the id, or a prior
        interactive session whose worktree directory was removed without
        deleting its branch) is handled exactly like :meth:`create_worktree`
        handles it: deleted when provably safe (no commits beyond main, no
        dirty tree — see :meth:`_cleanup_leftover_branch`), otherwise the
        create is refused with a ``RuntimeError`` rather than risk
        destroying work.

        **Concurrency**: cap-count + create-once are serialized by a
        per-instance lock (``_interactive_wt_lock``), so two concurrent
        calls on the SAME GitOps instance cannot both slip past the cap
        (TOCTOU-safe in-process; not cross-process).

        Unlike :meth:`acquire_warm_lane`, a seed failure is FAIL-SOFT: the
        worktree is retained (never removed) and ``warm=False`` is returned
        rather than raising — an interactive session should get a usable
        cold worktree rather than no worktree at all.

        Args:
            slug: caller-chosen claim identity; becomes both the branch
                suffix and the ``.task/interactive.json`` owner. Must match
                ``^[A-Za-z0-9][A-Za-z0-9._-]*$`` and must not contain
                ``'..'`` — it is interpolated directly into a filesystem
                path segment and a git branch name, so anything else raises
                ``ValueError`` immediately rather than risking an escaped
                path or an opaque git failure.
            start_ref: optional ref/SHA to pin as the base; defaults to the
                current local ``main_branch`` tip.

        Returns:
            InteractiveWorktreeInfo(path, branch, warm, base_ref).

        Raises:
            ValueError: slug fails the safe-charset validation above.
                Raised FIRST, before any state is inspected.
            InteractiveWorktreeLimitError: the on-disk ``_iact-*`` count under
                ``worktree_base`` is already at ``config.max_interactive_worktrees``.
                Raised before any git operation (REJECT policy).
            RuntimeError: start_ref/main_branch fails to resolve; a leftover
                branch for this slug provably carries work that would be
                destroyed by reuse (see :meth:`_cleanup_leftover_branch`); or
                ``git worktree add`` fails for another reason.
        """
        # ── Slug validation, FIRST — a caller bug independent of any state ──
        # slug is interpolated directly into a filesystem path segment and a
        # git branch name (see _INTERACTIVE_SLUG_RE).  '..' passes the
        # charset but is invalid in a git ref component, so it is rejected
        # separately here instead.
        if not _INTERACTIVE_SLUG_RE.match(slug) or '..' in slug:
            raise ValueError(
                f'create_interactive_worktree: invalid slug {slug!r} — must '
                f"match {_INTERACTIVE_SLUG_RE.pattern!r} and must not contain "
                f"'..' (it becomes both a path segment under "
                f'{self.worktree_base} and a git branch suffix).'
            )

        # ── Cap enforcement (REJECT) + create-once, serialized ────────────
        # _interactive_wt_lock guards count-through-`git worktree add` so two
        # concurrent callers on this instance can't both observe count < cap
        # and both create, overrunning the REJECT cap (TOCTOU) — mirrors
        # _spec_wt_create_lock's create-once serialization. In-process only.
        async with self._interactive_wt_lock:
            # Strictly filtered by iact_prefix so this never counts _lane-*/
            # _spec-*/_merge-verify/_offline-deep (isolation invariant I1).
            # Raw on-disk directory names, NOT cross-checked against `git
            # worktree list` — see InteractiveWorktreeLimitError's docstring.
            if self.worktree_base.exists():
                existing = sum(
                    1 for child in self.worktree_base.iterdir()
                    if child.is_dir() and child.name.startswith(self.config.iact_prefix)
                )
            else:
                existing = 0
            if existing >= self.config.max_interactive_worktrees:
                raise InteractiveWorktreeLimitError(
                    f'create_interactive_worktree({slug!r}): at cap — '
                    f'{existing}/{self.config.max_interactive_worktrees} '
                    f'{self.config.iact_prefix}* worktrees already exist under '
                    f'{self.worktree_base}. Free a slot (release one, or wait for '
                    f'the δ reaper) and retry.'
                )

            path = self.worktree_base / f'{self.config.iact_prefix}{slug}'
            full_branch = f'{self.config.branch_prefix}{slug}'

            ref_rc, ref_out, ref_err = await _run(
                ['git', 'rev-parse', start_ref or self.config.main_branch],
                cwd=self.project_root,
            )
            if ref_rc != 0:
                raise RuntimeError(
                    f'create_interactive_worktree: failed to resolve start ref '
                    f'{start_ref or self.config.main_branch!r}: {ref_err.strip()!r}'
                )
            base_ref = ref_out.strip()

            path.parent.mkdir(parents=True, exist_ok=True)

            # Self-heal a stale unregistered directory (mirrors acquire_warm_lane's
            # create-once branch) so `git worktree add` doesn't refuse a non-empty dir.
            if path.exists() and not await self._is_registered_worktree(path):
                if self._refuse_foreign_band(
                    path, frozenset({self.config.iact_prefix}),
                    'create_interactive_worktree-self-heal',
                ):
                    # Refused: the WARNING has already been emitted by the
                    # helper. Real candidates here are always this site's
                    # own iact band, so this branch is unreachable in real
                    # usage — pure defense-in-depth. Leave the directory in
                    # place rather than delete a foreign band.
                    pass
                else:
                    logger.warning(
                        'create_interactive_worktree: %s exists but is not registered; '
                        'removing stale directory (self-heal)', path,
                    )
                    shutil.rmtree(path)

            # Branch-namespace self-heal: full_branch shares config.branch_prefix
            # with dispatch task branches and other interactive slugs (see the
            # docstring's "Branch namespace" note), so a leftover ref for this
            # slug would otherwise collide with `git worktree add -b` below.
            # Mirror create_worktree's leftover-branch handling: clean it up
            # ONLY when _cleanup_leftover_branch proves it non-destructive (no
            # commits beyond main, no dirty tree); it raises RuntimeError
            # rather than delete anything it cannot prove safe.
            branch_exists_rc, _, _ = await _run(
                ['git', 'rev-parse', '--verify', full_branch], cwd=self.project_root,
            )
            if branch_exists_rc == 0:
                await self._cleanup_leftover_branch(full_branch, slug)

            add_rc, _, add_err = await _run(
                ['git', 'worktree', 'add', '-b', full_branch, str(path), base_ref],
                cwd=self.project_root,
            )
            if add_rc != 0:
                raise RuntimeError(
                    f'create_interactive_worktree: git worktree add failed for '
                    f'{path} (branch {full_branch!r}, base {base_ref!r}): '
                    f'{add_err.strip()!r}'
                )

        # .task-meta/<name>/interactive.json stamp for the δ reaper
        # (owner/created_at let the reaper age out worktrees with no activity
        # past the TTL). Written to the NEW .task-meta location ONLY (W11
        # gamma relocation; PRD `.task-meta` path-derivation contract: writes
        # new-path-only) — a worktree_base sibling of the worktree, so
        # `git add -A` in the lane can never stage it.
        stamp = {
            'owner': slug,
            'created_at': datetime.now(UTC).isoformat(),
            'branch': full_branch,
            'slug': slug,
        }
        meta_root = TaskArtifacts.meta_root_for(self.worktree_base, path.name)
        meta_root.mkdir(parents=True, exist_ok=True)
        (meta_root / 'interactive.json').write_text(json.dumps(stamp))

        # FAIL-SOFT (the key deviation from acquire_warm_lane): a seed fault
        # never removes the worktree and never raises — the worktree + stamp
        # above are retained either way, so the caller always gets a usable
        # worktree (warm or cold).  This is deliberately the opposite of
        # acquire_warm_lane, which removes the lane and returns FAULT/
        # DISK_PRESSURE on seed failure — that policy fits a pooled dispatch
        # lane (release the slot, let a future acquire retry); an interactive
        # session has no such retry path, so the worktree must survive.
        seed_rc = await self._seed_warm_lane(path, '--fresh-checkout')
        warm = seed_rc == 0
        if not warm:
            if seed_rc == 127:
                logger.warning(
                    'create_interactive_worktree: seed script absent for %s '
                    '(rc=127) — worktree retained but cold (warm=False)',
                    path,
                )
            else:
                logger.warning(
                    'create_interactive_worktree: seed failed (rc=%d) for %s '
                    '— worktree retained but cold (warm=False); fail-soft, '
                    'never removed on seed fault',
                    seed_rc, path,
                )

        return InteractiveWorktreeInfo(
            path=path, branch=full_branch, warm=warm, base_ref=base_ref,
        )

    def _scrub_seeded_lane_target(self, lane_dir: Path) -> None:
        """Delete configured glob-matched subtrees under a seeded lane's
        build-artifact dir so they regenerate fresh, per-lane (task 2315, BUG 3).

        The warm-lane CoW seed base's build-artifact dir (``target/`` by
        default) can contain generated artifacts that embed an ABSOLUTE
        path pointing back at the shared ``_merge-verify/target`` OUT_DIR
        they were originally generated under (e.g. tauri permission autogen
        files). ``cp -a --reflink``-seeding that base into a lane
        (:meth:`_seed_warm_lane`) copies those baked-in paths verbatim into
        every lane. Rather than rewriting the baked paths in place (fragile
        across TOML/binary formats), this deletes the configured subtrees
        so they regenerate fresh on first use.

        Args:
            lane_dir: The seeded lane's worktree root (build-artifact dir
                is resolved as ``lane_dir / reap_build_artifact_dirs[0]``,
                defaulting to ``lane_dir / 'target'``).

        No-op when :attr:`GitConfig.warm_lane_seed_scrub_globs` is empty
        (the default — opt-in, byte-identical). Best-effort: a glob
        matching nothing is a silent no-op, and any ``OSError`` during
        deletion is logged at WARNING and swallowed — never raises.
        """
        globs = self.config.warm_lane_seed_scrub_globs
        if not globs:
            return
        artifact_dir_name = (
            self.config.reap_build_artifact_dirs[0]
            if self.config.reap_build_artifact_dirs
            else 'target'
        )
        artifact_dir = lane_dir / artifact_dir_name
        try:
            for pattern in globs:
                for match in artifact_dir.glob(pattern):
                    if match.is_dir() and not match.is_symlink():
                        shutil.rmtree(match, ignore_errors=True)
                    else:
                        match.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                '_scrub_seeded_lane_target: failed to scrub %s under %s',
                globs, artifact_dir, exc_info=True,
            )

    async def _seed_warm_lane(
        self, lane_dir: Path, mode: str, *, take_lane_lock: bool = True,
    ) -> int:
        """Run seed-warm-lane.sh to CoW-seed the lane's target/ from the warm base.

        Invokes ``<lane_dir>/scripts/seed-warm-lane.sh <base_target> <lane_dir> <mode>``
        where *base_target* is the resolved base path and *mode* is either
        ``'--fresh-checkout'`` or ``'--reset-in-place'``.

        **D8 gen-dir symlink resolution (reify activation-seam R1)**:
        When :attr:`warm_lane_base_target_path` is a symlink (e.g. ``target`` →
        ``.gen.N`` as created by ``ln -sfn .gen.N target`` in reify), the CONCRETE
        gen dir is resolved via ``base.parent / base.readlink()`` (relative-sibling
        join, NOT ``Path.resolve()``, to avoid tmp-prefix canonicalization drift).
        The resolved path is what gets passed to the script so ``cp -a --reflink``
        copies the gen dir contents rather than the symlink itself.

        For a plain directory or nonexistent path the raw path is passed unchanged
        (back-compat: default config where base == _merge-verify/target plain dir).

        The script lives in the LANE's own scripts dir (the lane's checked-out
        tree provides it, consistent with the debug-port script pattern).

        **Takes ``<lane_dir>.lock`` exclusively (task 2599)**: the seed
        subprocess is wrapped in an OUTER blocking exclusive
        ``flock -x <lane_dir>.lock`` spanning its full duration — nesting
        outside the INNER *shared* per-gen-dir lock above in the symlink
        branch (outer exclusive lane lock, inner shared gen lock; both held
        for the script's whole lifetime and auto-released on exit, including
        on holder crash). ``flock`` consumes its own leading args, so the
        script itself still only ever sees ``base_target lane_dir mode`` as
        its argv. This gives :meth:`_run_thin_warm_lane`'s non-blocking T3
        ``flock -n`` a real counterparty: a concurrent release-thin and a
        re-acquire's seed can no longer both proceed against the same
        ``target/`` at once — see that method's "Lane-lock coupling gap"
        docstring note for the full race analysis (now closed).

        **``take_lane_lock`` (task 2567)**: when ``False``, the OUTER
        ``flock -x <lane_dir>.lock`` wrapper described above is omitted
        entirely — only the INNER per-gen-dir ``flock -s <gen>.lock``
        (symlink branch only; a different path) is still taken. Callers
        that already hold ``<lane_dir>.lock`` themselves for the whole
        call (e.g. :meth:`GitOps.ephemeral_worktree`'s CM-lifetime flock,
        task 2507) MUST pass ``take_lane_lock=False`` — re-acquiring the
        IDENTICAL path from the same process would self-deadlock against
        the bounded wait below, timing out at
        ``_SEED_WARM_LANE_LOCK_TIMEOUT_RC`` after
        ``_SEED_WARM_LANE_LOCK_WAIT_SECS`` on every call. Default ``True``
        keeps every other existing caller (``acquire_warm_lane``,
        ``create_interactive_worktree``, recycle) byte-identical.

        **Bounded wait, not unbounded (task 2599 amendment)**: seeding runs
        on the latency-sensitive warm-lane acquisition hot path, so the
        outer lock is acquired with ``-w _SEED_WARM_LANE_LOCK_WAIT_SECS``
        and ``-E _SEED_WARM_LANE_LOCK_TIMEOUT_RC`` rather than a plain
        unbounded ``flock -x`` — ``flock`` only auto-releases a lock on
        holder *death*, never on a stuck-but-live process, so an unbounded
        wait could stall lane acquisition (and therefore task dispatch)
        indefinitely against a live-but-wedged holder (thin's ``rm -rf``, GC
        reclaim, or another seed). ``-E`` gives the timeout its own exit
        code, mirroring ``timeout(1)``'s well-known 124 "command timed out"
        convention, so it is logged as a distinct, diagnosable condition
        rather than folded into the generic non-zero-fault warning below.
        This is deliberately fail-CLOSED, not fail-open: unlike
        :meth:`merge_verify_lease`'s contended-flock fallback (that flock is
        a secondary/observational lease over a primary enforced elsewhere,
        so proceeding without it is safe), ``<lane_dir>.lock`` IS the
        primary mutual-exclusion mechanism this task establishes —
        proceeding without it on timeout would silently reopen the exact
        race this task closes. No retry is added either: every existing
        caller already treats any non-zero rc as a fault (cold fallback /
        release-and-retry / escalate — see the callsites below), so a
        timeout is handled identically to any other seed fault, and
        stacking an in-method retry on top of the bounded wait would only
        extend the hot-path stall this timeout exists to bound.

        Returns:
            0   — script ran and exited 0 (seed succeeded, lane is warm).
            75  — script exited 75 (EX_TEMPFAIL, disk-pressure discriminant).
            124 — outer <lane_dir>.lock wait timed out after
                  _SEED_WARM_LANE_LOCK_WAIT_SECS — a live-but-wedged lock
                  holder; the script itself never ran (task 2599 amendment).
            1..N — script exited with any other non-zero code (generic fault).
            127 — script absent (command-not-found sentinel).
            127 — any unexpected exception (non-zero sentinel, never raises).

        Callers must use ``rc == 0`` for success and may inspect the exact
        code to discriminate disk-pressure (75) or a lock-wait timeout (124,
        see ``_SEED_WARM_LANE_LOCK_TIMEOUT_RC``) from a generic fault (any
        other non-zero).
        """
        try:
            script = lane_dir / 'scripts' / 'seed-warm-lane.sh'
            if not script.exists():
                logger.debug('_seed_warm_lane: seed script absent at %s', script)
                return 127  # command-not-found sentinel
            # task 2599: outer exclusive lane lock spans the ENTIRE seed
            # subprocess below (both branches), giving _run_thin_warm_lane's
            # T3 flock -n a real counterparty — see the "Takes <lane_dir>.lock
            # exclusively" docstring note above. Bounded via -w/-E (task 2599
            # amendment) — see the "Bounded wait, not unbounded" docstring
            # note — so a live-but-wedged holder fails closed with a
            # distinct, diagnosable rc instead of stalling this hot path
            # forever.
            lane_lock = lane_lock_path(lane_dir)
            lane_lock_flock = (
                [
                    'flock', '-x',
                    '-w', str(_SEED_WARM_LANE_LOCK_WAIT_SECS),
                    '-E', str(_SEED_WARM_LANE_LOCK_TIMEOUT_RC),
                    str(lane_lock),
                ]
                if take_lane_lock
                else []
            )
            # reify 5556: when WE hold the outer lane lock above, seed must NOT
            # re-open+flock the same file. reify's seed-warm-lane.sh acquires
            # ${LANE_DIR}.lock by DEFAULT under --fresh-checkout as of reify
            # 7b20d010c6 (task 5354) — previously opt-in via --lane-lock — and
            # flock is not re-entrant across a process tree, so the script's
            # own flock -n self-refuses against this method's lock and exits
            # 75. That 75 is indistinguishable from genuine disk pressure at
            # _classify_seed_rc, so every dispatch requeued as
            # WarmLaneDiskPressure with agent_invocations=0, released the lane,
            # and re-picked the same lowest-index free lane: a fleet-wide
            # dispatch livelock (349 requeues / 4 completions per day).
            # --assume-lane-lock-held (reify db9ea9387b, same task) is the
            # sanctioned opt-out for exactly this caller shape.
            #
            # Capability-probed rather than passed blind: `script` is the LANE's
            # own checked-out copy, so a lane sitting on a pre-5354 base would
            # reject the unknown flag as a usage error (exit 2) and turn a
            # working seed into a hard fault. Probe absent → omit the flag and
            # keep the pre-5354 behaviour, where the script never self-locks.
            seed_flags: list[str] = []
            if take_lane_lock and _seed_script_supports_assume_lane_lock_held(script):
                seed_flags.append(_SEED_ASSUME_LANE_LOCK_HELD_FLAG)
            base_path = self.warm_lane_base_target_path
            if base_path.is_symlink():
                # D8: resolve relative-sibling symlink (target -> .gen.N) to the
                # concrete gen dir so the cp receives the directory, not the symlink.
                gen_dir = base_path.parent / base_path.readlink()
                base_target = str(gen_dir)
                # D8 reader-refcount GC: hold a shared lock (flock -s) on the per-gen
                # lock file for exactly the cp lifetime.  A concurrent reify GC rewrite
                # (flock -n -x) is deferred while the shared lock is held, preventing a
                # torn read (ENOENT mid-walk).  Lock auto-released on script exit.
                lock_path = gen_dir.with_name(gen_dir.name + '.lock')
                if not lock_path.exists():
                    # The lock file should be pre-created by reify alongside the gen dir.
                    # flock(1) creates it if absent, but the resulting inode is unshared
                    # with reify's exclusive locker — the GC protocol is desynced.
                    # Log at debug so a missing lock file surfaces rather than degrading
                    # silently to an ineffective lock.
                    logger.debug(
                        '_seed_warm_lane: lock file %s does not pre-exist — '
                        'flock will create a new inode unshared with reify GC; '
                        'this indicates a reify/DF gen-dir protocol mismatch',
                        lock_path,
                    )
                cmd = [
                    *lane_lock_flock,
                    'flock', '-s', str(lock_path),
                    str(script), base_target, str(lane_dir), mode, *seed_flags,
                ]
            else:
                base_target = str(base_path)
                cmd = [
                    *lane_lock_flock,
                    str(script), base_target, str(lane_dir), mode, *seed_flags,
                ]
            rc, _, err = await _run(cmd, cwd=lane_dir)
            if rc == _SEED_WARM_LANE_LOCK_TIMEOUT_RC:
                logger.warning(
                    '_seed_warm_lane: timed out after %.0fs waiting for '
                    '%s — a concurrent holder (thin rm -rf / GC reclaim / '
                    'another seed) is still live; failing closed rather '
                    'than risk a torn target/ (rc=%d)',
                    _SEED_WARM_LANE_LOCK_WAIT_SECS, lane_lock, rc,
                )
            elif rc != 0:
                logger.warning(
                    '_seed_warm_lane: script exited %d for %s (stderr=%r)',
                    rc, lane_dir, err,
                )
            else:
                # task 2099: rc == 0 proves the cp --reflink from the on-mount
                # base target succeeded, i.e. the pool storage is present and
                # writable — the ONE chokepoint where the .pool-root sentinel
                # is safe to (re-)write. See mark_pool_storage_present().
                self.mark_pool_storage_present()
                # task 2315 (BUG 3): rc == 0 also means the lane's target/ was
                # just CoW-seeded from the shared base, which may carry
                # forward stale baked-in absolute OUT_DIR paths. Scrub the
                # configured subtrees so they regenerate per-lane. Order vs.
                # mark_pool_storage_present() is irrelevant — scrub is
                # idempotent/best-effort and never raises. Covers every seed
                # site (create-once, reset-in-place, recycle) uniformly.
                self._scrub_seeded_lane_target(lane_dir)
            return rc
        except Exception:
            logger.warning(
                '_seed_warm_lane: unexpected error for %s', lane_dir, exc_info=True,
            )
            return 127  # exception sentinel (non-zero, non-75 → generic fault)

    async def refresh_warm_base(self, landed_commit: str | None = None) -> bool:
        """Advance the rolling warm base by running the refresh script.

        Invokes ``<_merge-verify>/scripts/refresh-warm-base.sh <advancing> <base>
        [--landed-commit <sha>]`` where:
        - *advancing* = ``persistent_merge_worktree_path / reap_build_artifact_dirs[0]``
          (the _merge-verify lane's target — the warm build of the just-landed commit)
        - *base* = ``warm_lane_base_target_path`` (the configured rolling base)
        - ``--landed-commit <sha>`` (D10) — appended when *landed_commit* is truthy.

        **D10 --landed-commit derivation**: when *landed_commit* is ``None`` (the
        default), it is derived from ``git rev-parse HEAD`` in the _merge-verify
        worktree (fail-soft: if rev-parse fails the flag is omitted).  The refresh
        only fires for the _merge-verify lane (merge_queue.py gate), whose HEAD IS
        the just-landed commit — the derived sha is exactly what reify's inv.9
        HEAD-match guard expects.  The optional *landed_commit* parameter allows
        deterministic test injection.

        **inv.9 promote-provenance** is enforced STRUCTURALLY: the advancing dir
        is hardcoded to the _merge-verify target and no caller-supplied advancing
        parameter is exposed, so DF can never source a task lane (un-landed WIP).
        The reify B12 guard inside refresh-warm-base.sh is defense-in-depth.

        Self-guards (all return False, never raise):
        - ``warm_lane_pool is None`` — pool knob off; refresh has no meaning.
        - ``advancing == base`` — no separate rolling base configured; degenerate
          self-copy skipped (default config has ``warm_lane_base_target_dir=None``
          so base == ``_merge-verify/target`` == advancing).
        - Script absent or non-zero exit — fail-soft.

        Returns:
            True  — script ran and exited 0 (base refreshed).
            False — any guard triggered or the script absent/failed (fail-soft;
                    never raises — a base-refresh hiccup cannot block merges).
        """
        if self.warm_lane_pool is None:
            return False

        try:
            advancing = self._merge_verify_artifact_path
            base = self.warm_lane_base_target_path

            if advancing.resolve() == base.resolve():
                logger.debug(
                    'refresh_warm_base: advancing == base (%s) — no separate rolling '
                    'base configured, skipping degenerate self-copy', advancing,
                )
                return False

            script = self.persistent_merge_worktree_path / 'scripts' / 'refresh-warm-base.sh'
            if not script.exists():
                logger.debug(
                    'refresh_warm_base: script absent at %s — no-op', script,
                )
                return False

            # D10: derive --landed-commit from HEAD of the _merge-verify worktree
            # (fail-soft: rev-parse failure omits the flag rather than blocking).
            if landed_commit is None:
                rc_h, head_out, _ = await _run(
                    ['git', 'rev-parse', 'HEAD'],
                    cwd=self.persistent_merge_worktree_path,
                )
                if rc_h == 0 and head_out.strip():
                    landed_commit = head_out.strip()
                else:
                    logger.debug(
                        'refresh_warm_base: could not derive HEAD from %s (rc=%d) — '
                        'omitting --landed-commit flag',
                        self.persistent_merge_worktree_path, rc_h,
                    )

            argv = [str(script), str(advancing), str(base)]
            if landed_commit:
                argv += ['--landed-commit', landed_commit]

            rc, _, err = await _run(argv, cwd=self.persistent_merge_worktree_path,
            )
            if rc != 0:
                logger.warning(
                    'refresh_warm_base: script exited %d (stderr=%r)', rc, err,
                )
                return False
            return True
        except Exception:
            logger.warning('refresh_warm_base: unexpected error', exc_info=True)
            return False

    # warm-lane script resolution (task 3072, PRD leaf α) ----------------------

    def _warm_lane_script_candidates(
        self, name: str,
    ) -> tuple[tuple[Path, str], ...]:
        """Every candidate location for ``name``, in search order.

        The SINGLE source of truth for both halves of resolution:
        :meth:`_resolve_warm_lane_script` takes the first entry that exists,
        and the not-found WARNING in
        :meth:`_resolve_warm_lane_script_logged` names every entry.  Built once
        rather than re-derived per consumer so the order an operator reads can
        never disagree with the order actually searched, and so adding a third
        location (or a subdirectory layout) is one edit instead of two that
        must be kept in lockstep.

        Args:
            name: Bare script filename, e.g. ``'warm-lane-gc.sh'``.

        Returns:
            ``((path, origin), ...)`` in preference order.
        """
        return (
            (self.project_root / 'scripts' / name, 'project'),
            (_df_warm_lane_script_dir() / name, 'dark-factory'),
        )

    def _resolve_warm_lane_script(self, name: str) -> tuple[Path, str] | None:
        """Locate warm-lane script ``name``, project override first.

        Walks :meth:`_warm_lane_script_candidates` and returns the first entry
        that exists.  Resolution order (PRD
        ``warm-lane-infra-repatriation-prd.md`` design decision D3):

        1. ``<project_root>/scripts/<name>`` — the PROJECT OVERRIDE. A project
           that has invested in its own warm-lane tooling keeps it;
           dark-factory's copy is the floor, not the ceiling. This is also
           what makes leaf α a behavioural no-op for reify, whose own copies
           still win at every call site.
        2. :func:`_df_warm_lane_script_dir` ``/ <name>`` — dark-factory's own
           relocated copy under ``orchestrator/scripts/warm-lane/``.
        3. Neither → ``None``, so the caller emits a WARNING naming BOTH tried
           paths and returns its existing fail-soft sentinel.

        Keys on **existence, not the execute bit** — byte-identical to the
        ``script.exists()`` predicate every call site used pre-relocation. A
        present-but-broken project override therefore still reaches the
        subprocess spawn and fails there, keeping the failure attributable to
        the project rather than silently masked by substituting dark-factory's
        copy.

        Args:
            name: Bare script filename, e.g. ``'warm-lane-gc.sh'``.

        Returns:
            ``(resolved_path, origin)`` where origin is ``'project'`` or
            ``'dark-factory'``, or ``None`` when neither location has it.
        """
        for path, origin in self._warm_lane_script_candidates(name):
            if path.exists():
                return (path, origin)
        return None

    def _resolve_warm_lane_script_logged(
        self, name: str, method: str,
    ) -> tuple[Path, str] | None:
        """:meth:`_resolve_warm_lane_script`, plus the operator-facing log line.

        Shared by all six warm-lane wrappers so the message shape is identical
        across them and one ``grep`` matches every site. Both outcomes are
        logged from here rather than open-coded per wrapper precisely so the
        two messages cannot drift apart into six dialects.

        On SUCCESS emits an INFO naming the resolved path and its origin,
        immediately before the caller spawns it. **Unconditional per
        invocation** — no memo, no once-per-process guard: PRD leaf ζ's
        go/no-go reads this line off a live reclaim pass, which may be the
        hundredth of the process, so suppressing repeats would make that read
        silently empty. Pinned by
        ``test_warm_lane_script_resolution.py::TestResolvedPathIsLoggedAtInfo::
        test_info_line_is_emitted_on_every_invocation``.

        On FAILURE emits a WARNING naming BOTH tried paths. WARNING, not the
        pre-relocation DEBUG: after leaf α dark-factory always ships its own
        copy, so "neither location" can only mean a genuinely broken
        deployment — rare in production, never routine noise — and a DEBUG
        line naming only one of two searched locations is actively misleading
        to an operator asking why GC stopped.

        Callers keep their own ``return <sentinel>`` on ``None`` because the
        six sentinels differ (127 / None / False), and keep their
        pre-resolution guards (merge-verify lease, pool storage) ahead of this
        call so a skip stays attributable to the real cause.

        Args:
            name: Bare script filename, e.g. ``'warm-lane-gc.sh'``.
            method: Calling method name, prefixed onto both log lines.

        Returns:
            ``(resolved_path, origin)``, or ``None`` when neither location
            has the script (already logged).
        """
        resolved = self._resolve_warm_lane_script(name)
        if resolved is None:
            tried = ' and '.join(
                str(path)
                for path, _origin in self._warm_lane_script_candidates(name)
            )
            logger.warning(
                '%s: no warm-lane script implementation found at either '
                'location — tried %s',
                method, tried,
            )
            return None
        path, origin = resolved
        logger.info('%s: resolved %s -> %s (%s)', method, name, path, origin)
        return resolved

    # ε: warm-lane disk-guard admission helpers --------------------------------

    async def _run_warm_lane_disk_guard(self) -> int:
        """Invoke ``warm-lane-disk-guard.sh check``.

        Located via :meth:`_resolve_warm_lane_script`:
        ``<project_root>/scripts/warm-lane-disk-guard.sh`` first, then
        dark-factory's own copy under ``orchestrator/scripts/warm-lane/``. A
        project that carries its own warm-lane tooling keeps it (PRD
        ``warm-lane-infra-repatriation-prd.md`` D3); dark-factory's copy is
        the floor, not the ceiling.

        Mirrors the ``_seed_warm_lane``/``refresh_warm_base`` fail-soft helper
        pattern: no implementation at either location → 127 sentinel; any
        unexpected exception → 127; never raises.

        Returns:
            0   — healthy (disk pressure below threshold).
            75  — disk pressure (EX_TEMPFAIL, admission should block).
            127 — no implementation at either location, or exception
                  (fail-open sentinel).
            other non-zero — script error (treated as fail-open by caller).
        """
        try:
            resolved = self._resolve_warm_lane_script_logged(
                'warm-lane-disk-guard.sh', '_run_warm_lane_disk_guard',
            )
            if resolved is None:
                return 127
            script, _origin = resolved
            # NOTE: worktree_base may not exist yet on a fresh host where no lane
            # has been created.  In that case the real γ script will likely stat a
            # non-existent path and return a non-(0,75) exit code, which the caller
            # treats as fail-open (guard is an inert no-op until worktree_base first
            # appears).  This is deliberate: the guard cannot measure a volume that
            # does not yet exist, and the script-level fail-closed convention (df
            # failure → 75) only applies to volume access errors, not missing mount
            # points.  Seed the first lane to create worktree_base, then the guard
            # becomes active.
            cmd = [
                str(script), 'check',
                '--mount', str(self.worktree_base),
                '--min-free-gib', str(self.config.warm_lane_min_free_gib),
                '--min-free-inodes', str(self.config.warm_lane_min_free_inodes),
            ]
            rc, _, err = await _run(cmd, cwd=self.project_root)
            if rc not in (0, 75):
                logger.warning(
                    '_run_warm_lane_disk_guard: script exited %d (stderr=%r)', rc, err,
                )
            return rc
        except Exception:
            logger.warning(
                '_run_warm_lane_disk_guard: unexpected error', exc_info=True,
            )
            return 127

    # θ: warm-lane PROACTIVE soft-floor throttle helpers (task 2443) --------

    async def _run_warm_lane_soft_guard(self) -> int:
        """Invoke ``warm-lane-disk-guard.sh check --soft``.

        Located via :meth:`_resolve_warm_lane_script` — project override
        first, then dark-factory's own copy (PRD D3); see
        :meth:`_run_warm_lane_disk_guard`.

        θ (task 2443, §9.5): the proactive soft-floor counterpart to
        :meth:`_run_warm_lane_disk_guard`, run BEFORE it's too late — a soft
        floor ABOVE the hard floor lets the caller throttle new allocations
        before the hard floor's exit-75 backstop is ever reached. Mirrors
        the ``_run_warm_lane_disk_guard`` fail-soft helper pattern exactly:
        absent script → 127 sentinel; any unexpected exception → 127; never
        raises.

        Returns:
            0   — healthy (both hard and soft floors clear).
            3   — soft pressure (above hard floor, below soft floor;
                  stdout carries the ``@@REIFY_WARM_LANE_SOFT_PRESSURE@@``
                  sentinel per the reify contract — not parsed here).
            75  — hard disk pressure (EX_TEMPFAIL); takes precedence over
                  soft per the reify script's own contract.
                  :meth:`_warm_lane_soft_pressure_defer` treats this rc as a
                  defer signal unconditionally (rc==75 is never healthy), so
                  a below-hard-floor condition that slips past ε's upstream
                  check (the narrow TOCTOU window) — or a soft-only
                  configuration with ε disabled — still backpressures rather
                  than failing open. See that method's docstring (amendment,
                  reviewer_comprehensive robustness).
            127 — no implementation at either location, or exception
                  (fail-open sentinel).
            other non-zero — script error (treated as fail-open by caller).
        """
        try:
            resolved = self._resolve_warm_lane_script_logged(
                'warm-lane-disk-guard.sh', '_run_warm_lane_soft_guard',
            )
            if resolved is None:
                return 127
            script, _origin = resolved
            cmd = [
                str(script), 'check',
                '--mount', str(self.worktree_base),
                '--min-free-gib', str(self.config.warm_lane_min_free_gib),
                '--min-free-inodes', str(self.config.warm_lane_min_free_inodes),
                '--soft',
                '--soft-free-gib', str(self.config.warm_lane_soft_free_gib),
                '--soft-free-inodes', str(self.config.warm_lane_soft_free_inodes),
            ]
            rc, _, err = await _run(cmd, cwd=self.project_root)
            if rc not in (0, 3, 75):
                logger.warning(
                    '_run_warm_lane_soft_guard: script exited %d (stderr=%r)', rc, err,
                )
            return rc
        except Exception:
            logger.warning(
                '_run_warm_lane_soft_guard: unexpected error', exc_info=True,
            )
            return 127

    async def _run_warm_lane_audit(self) -> str | None:
        """Invoke ``warm-lane-audit.sh --mount <worktree_base>``.

        Located via :meth:`_resolve_warm_lane_script` — project override
        first, then dark-factory's own copy (PRD D3); see
        :meth:`_run_warm_lane_disk_guard`.

        α (task 2443, §9.5 inv.12): OBSERVABILITY-ONLY. This wrapper never
        gates an admission decision — it exists solely to enrich the θ
        soft-floor defer journal line
        (:meth:`_warm_lane_soft_pressure_defer`) with pool headroom context.
        Mirrors the :meth:`warm_lane_ref_is_degenerate`/
        :meth:`_run_thin_warm_lane` fail-soft wrapper pattern: no
        implementation at either location, non-zero exit, or any exception
        all degrade to ``None``; never raises.

        **Read-only (A1)**: invoked with ONLY ``--mount`` — no reset/reclaim
        subcommand or flag. reify's ``warm-lane-audit.sh`` is read-only by
        its own contract (never mutates a lane); this wrapper does not add
        any mutating flag on top of that.

        Returns:
            The trailing ``HEADROOM ...`` summary line from the script's
            default table-format stdout, or ``None`` if no implementation
            exists at either location, the script exits non-zero, its stdout
            carries no line beginning ``HEADROOM``, or an unexpected
            exception occurred.
        """
        try:
            resolved = self._resolve_warm_lane_script_logged(
                'warm-lane-audit.sh', '_run_warm_lane_audit',
            )
            if resolved is None:
                return None
            script, _origin = resolved
            cmd = [str(script), '--mount', str(self.worktree_base)]
            rc, out, err = await _run(cmd, cwd=self.project_root)
            if rc != 0:
                logger.debug(
                    '_run_warm_lane_audit: script exited %d (stderr=%r) — no headroom',
                    rc, err,
                )
                return None
            for line in out.splitlines():
                if line.startswith('HEADROOM'):
                    return line
            logger.debug(
                '_run_warm_lane_audit: no HEADROOM line in stdout (stdout=%r)', out,
            )
            return None
        except Exception:
            logger.warning(
                '_run_warm_lane_audit: unexpected error', exc_info=True,
            )
            return None

    async def _warm_lane_audit_cached(self) -> str | None:
        """Short-window memo over :meth:`_run_warm_lane_audit` (α, inv.12).

        The θ soft-floor defer path (:meth:`_warm_lane_soft_pressure_defer`)
        enriches its WARNING with the α HEADROOM summary. Under SUSTAINED
        soft pressure a fresh allocation requeues indefinitely (documented,
        inv.11), so without a memo each dispatch/requeue cycle re-forks the
        ``warm-lane-audit.sh`` subprocess even though nothing has changed
        (amendment, reviewer_comprehensive performance-efficiency). This
        memoizes the last HEADROOM for :data:`_WARM_LANE_AUDIT_CACHE_TTL_SECS`
        so repeated defers within that window reuse it instead of re-forking.

        α is OBSERVABILITY-ONLY (inv.12): a slightly-stale cached HEADROOM in
        a log line is acceptable, and this NEVER gates the defer decision —
        the per-defer WARNING itself is retained (it is the intended B10
        backpressure signal); only the audit *subprocess* is rate-limited.
        Fail-soft like the underlying wrapper: never raises (any error from
        :meth:`_run_warm_lane_audit` already degrades to ``None``).

        A TTL of ``0.0`` (e.g. monkeypatched in tests) disables the memo —
        the deadline never lies strictly in the future — so every call
        re-forks.

        **Interaction with the resolved-path INFO line (task 3072).** That
        line is emitted per *invocation of the wrapper*, and this memo
        suppresses the wrapper itself on a cache hit — so a hit produces no
        resolved-path line either. That is unchanged pre-existing behaviour,
        not a rate-limit on the log: the memo has always suppressed the whole
        subprocess, and α is observability-only. The line's
        "every invocation" guarantee is about
        :meth:`_run_warm_lane_audit` and the other five wrappers never
        memoising it themselves; an operator reading resolution off a live
        pass should use the reclaim path (:meth:`_run_warm_lane_gc_reclaim`),
        which has no memo.
        """
        now = time.monotonic()
        cache = self._warm_lane_audit_cache
        if cache is not None and now < cache[0]:
            return cache[1]
        headroom = await self._run_warm_lane_audit()
        self._warm_lane_audit_cache = (
            now + _WARM_LANE_AUDIT_CACHE_TTL_SECS, headroom,
        )
        return headroom

    async def _run_warm_lane_gc_reclaim(self) -> int:
        """Invoke ``warm-lane-gc.sh reclaim``.

        Located via :meth:`_resolve_warm_lane_script` — project override
        first, then dark-factory's own copy (PRD D3); see
        :meth:`_run_warm_lane_disk_guard`.

        **Seed-primitive passthrough (task 3072).** Passes ``--seed-script
        <project_root>/scripts/seed-warm-lane.sh`` when that file exists.
        ``warm-lane-gc.sh`` otherwise defaults ``SEED_SCRIPT`` to its own
        sibling ``$SCRIPT_DIR/seed-warm-lane.sh`` and invokes it
        UNCONDITIONALLY on the Pass-1 lane-reset path — but PRD §5 keeps
        ``seed-warm-lane.sh`` with the project as one of the two genuinely
        toolchain-bound primitives, so it does not travel with the relocated
        policy script.  Once dark-factory's copy is the one running (leaf
        ζ/κ), that sibling default would point at a file that is not there and
        fail EVERY lane reset — a silently-stopped GC accreting the pool to
        ENOSPC.  Naming the project's primitive explicitly is resolution
        wiring, not a policy change; ``gc.sh`` already exposes the flag.

        Strictly no-op today: for a reify-shaped ``project_root`` the passed
        path is byte-identical to the one gc.sh's own default computes, and a
        project with no seed script gets no flag, so argv is unchanged.
        Patching the relocated script's default instead would make
        dark-factory guess at a project-owned path, violating PRD invariant
        C-1.

        **The degraded mode is announced, not inferred.**  A project that
        carries no warm-lane tooling at all — the very case dark-factory ships
        these copies for — has no ``seed-warm-lane.sh`` to name, so the flag is
        omitted and the relocated script falls back to a sibling default that
        PRD §5 guarantees will never exist beside it.  Every non-disk-pressure
        Pass-1 lane reset then fails inside gc.sh, is warned about there, and
        counts toward ``preserved`` while reclaiming nothing: the same
        accrete-to-ENOSPC class this wrapper guards against, reached from the
        opposite branch.  So when the resolved origin is ``'dark-factory'`` and
        the project has no seed script, this logs a WARNING naming the missing
        path and what stops working, making the degradation attributable from
        dark-factory's own logs instead of only from gc.sh's stderr.  Not
        raised and not a sentinel: orphan removal and disk-pressure target
        removal still work, so reclaim is degraded rather than dead.

        Fail-soft: absent script → 127 sentinel; any unexpected exception → 127;
        never raises.  A non-zero exit is logged at WARNING and treated as
        'nothing reclaimed' by the caller (``_warm_lane_disk_admission_blocked``).

        **Merge-verify lease guard (task 2315, BUG 1)**: defers (127) while
        ANY merge-verify lease is held — INCLUDING our own.  The reclaim
        script operates over the whole pool mount (which CONTAINS
        ``_merge-verify``), so an in-process local verify must be deferred
        to just as much as a foreign one; unlike
        :meth:`reset_persistent_merge_worktree`'s lease guard, self is NOT
        excluded here.  Checked BEFORE the pool-storage guard below so the
        skip is attributable to the lease even when the sentinel is fine.

        **Pool-storage guard (task 2099, self-heal task 2315)**: routes
        through :meth:`_reconcile_pool_storage_before_sweep`, which refuses
        to spawn the script when a pool is configured
        (:meth:`pool_in_use`) but :meth:`pool_storage_present` is False AND
        that absence is not provably a first-seed bootstrap — an unmounted
        mountpoint must never let the GC script reclaim/reset lanes it can
        only see as missing.  A HEALTHY mount that merely lost its sentinel
        (:meth:`_pool_storage_bootstrap_ok` True) self-heals: the sentinel
        is recreated and the reclaim proceeds.  Skipped entirely when no
        pool is in use: ``pool_storage_present()`` is permanently False on
        a pool-less host (its only writer never runs without a pool), so
        that alone must never be treated as a mount-down incident.
        Returns the same 127 fail-soft sentinel used for an absent script,
        so callers treat it identically to 'nothing reclaimed'.

        Returns:
            0   — reclaim succeeded.
            127 — no implementation at either location, pool storage absent,
                merge-verify lease held, or exception (fail-soft sentinel).
            other non-zero — reclaim script error (caller still re-checks).
        """
        if self._merge_verify_lease_active():
            logger.info(
                '_run_warm_lane_gc_reclaim: merge-verify in flight (lease '
                'held) — deferring reclaim',
            )
            return 127
        if not self._reconcile_pool_storage_before_sweep('_run_warm_lane_gc_reclaim'):
            return 127
        try:
            resolved = self._resolve_warm_lane_script_logged(
                'warm-lane-gc.sh', '_run_warm_lane_gc_reclaim',
            )
            if resolved is None:
                return 127
            script, origin = resolved
            cmd = [
                str(script), 'reclaim',
                '--mount', str(self.worktree_base),
            ]
            seed = self.project_root / 'scripts' / 'seed-warm-lane.sh'
            if seed.exists():
                cmd += ['--seed-script', str(seed)]
            elif origin == 'dark-factory':
                logger.warning(
                    '_run_warm_lane_gc_reclaim: no project seed script at %s, '
                    'and none ships beside %s (PRD §5 keeps seed-warm-lane.sh '
                    'project-owned) — lane RESETS will fail inside the script '
                    '(each logs "reset failed … (seed-script error)" and '
                    'counts as preserved); reclaim degrades to orphan removal '
                    'plus disk-pressure target rm',
                    seed, script,
                )
            rc, _, err = await _run(cmd, cwd=self.project_root)
            if rc != 0:
                logger.warning(
                    '_run_warm_lane_gc_reclaim: script exited %d (stderr=%r)', rc, err,
                )
            return rc
        except Exception:
            logger.warning(
                '_run_warm_lane_gc_reclaim: unexpected error', exc_info=True,
            )
            return 127

    async def warm_lane_ref_is_degenerate(self, task_id: str) -> bool:
        """Invoke ``warm-lane-degenerate-ref-check.sh`` for one task's ref.

        Located via :meth:`_resolve_warm_lane_script` — project override
        first, then dark-factory's own copy (PRD D3); see
        :meth:`_run_warm_lane_disk_guard`.

        Task 2112: wraps reify's read-only classifier primitive (contract in
        reify ``docs/design/warm-lane-degenerate-ref-seam.md``, reify task
        5006) so callers can ask "is task <task_id>'s branch ref degenerate
        (zero commits over main AND the tip does not cite this task)?"
        without duplicating the git plumbing. Mirrors the
        ``_run_warm_lane_disk_guard`` / ``_run_warm_lane_gc_reclaim``
        invocation pattern.

        Single-ref exit-code taxonomy (per the reify contract):
            0 — degenerate (count==0 over main AND tip does NOT cite
                task_id) — skip/prune-safe.
            1 — live.
            2 — usage error.
            3 — structural error.
            4 — landed (count==0 over main AND tip DOES cite task_id).
            5 — absent (ref does not exist).

        **FAIL-SOFT contract**: returns True ONLY on exit 0. No
        implementation at either location, any other exit code
        (1/2/3/4/5/other), or any exception all return False — so on any
        doubt this is a no-op and existing behaviour is preserved.
        Read-only; never mutates refs. Never raises.
        """
        try:
            resolved = self._resolve_warm_lane_script_logged(
                'warm-lane-degenerate-ref-check.sh', 'warm_lane_ref_is_degenerate',
            )
            if resolved is None:
                return False
            script, _origin = resolved
            cmd = [
                str(script),
                '--task', str(task_id),
                '--main-ref', self.config.main_branch,
                '--branch-prefix', self.config.branch_prefix,
                '--repo', str(self.project_root),
            ]
            rc, _, err = await _run(cmd, cwd=self.project_root)
            if rc != 0:
                logger.debug(
                    'warm_lane_ref_is_degenerate: script exited %d for task %s '
                    '(stderr=%r) — treating as not-degenerate',
                    rc, task_id, err,
                )
            return rc == 0
        except Exception:
            logger.warning(
                'warm_lane_ref_is_degenerate: unexpected error for task %s',
                task_id, exc_info=True,
            )
            return False

    async def _run_thin_warm_lane(self, lane_dir: Path) -> int:
        """Invoke ``thin-warm-lane.sh <lane_dir>``.

        Located via :meth:`_resolve_warm_lane_script` — project override
        first, then dark-factory's own copy (PRD D3); see
        :meth:`_run_warm_lane_disk_guard`.

        Task 2442 (§9.5 η): fail-soft, never-raise wrapper around reify δ's
        free-first target-reclaim primitive, modeled on
        :meth:`_run_warm_lane_gc_reclaim`. Invoked WITHOUT ``--reseed`` (D3) —
        the next :meth:`acquire_warm_lane` always re-seeds ``target/`` from
        the current base regardless (D10), so leaving ``target/`` empty here
        does not change net warmth; only the idle-hold between release and a
        re-acquire that may never come is eliminated.

        **Pool-storage guard (task 2099)**: refuses to spawn the script when
        a pool is configured (:meth:`pool_in_use`) but
        :meth:`pool_storage_present` is False — an unmounted mountpoint must
        never let this script operate against a lane it can only see as
        missing. This is the RAW refuse-only check (``pool_in_use() and not
        pool_storage_present()``) — it does NOT route through
        :meth:`_reconcile_pool_storage_before_sweep`, the self-healing
        variant :meth:`_run_warm_lane_gc_reclaim` uses, which recreates a
        merely-lost ``.pool-root`` sentinel on a provably-healthy mount
        instead of refusing forever. The raw check is sufficient here:
        release-thin is a purely optional, fail-open reclaim that nothing
        else depends on for forward progress, and this method never writes
        the sentinel itself (:meth:`_seed_warm_lane`'s ``rc == 0`` is the
        only writer) — so there is nothing for it to self-heal. A spurious
        refusal here just idle-holds ``target/`` for one more cycle until
        the sentinel is recreated elsewhere, never the chicken-and-egg
        deadlock :meth:`_reconcile_pool_storage_before_sweep` exists to
        break (see that method's docstring).

        **Flock contract (pinned — this wrapper holds no lock of its own)**:
        safety against a concurrent re-acquire racing this call rests
        entirely on the script side. reify's ``scripts/thin-warm-lane.sh``
        acquires ``<lane_dir>.lock`` (a sibling lock file, non-blocking
        ``flock -n``) BEFORE touching ``target/`` and exits 75 if it cannot
        — see that script's "T3" block and PRD
        ``docs/prds/warm-lane-pool-sizing-lifecycle.md`` §9.3 invariant T3 /
        §9.5 inv.10 (reify repo). This wrapper only spawns the script and
        interprets its exit code; if the script's flock behavior ever
        regresses, this wrapper provides no independent defense. See
        ``TestRunThinWarmLaneFlockContention`` in ``test_git_ops.py`` for a
        unit test that exercises this against a real held flock (not just a
        scripted exit code).

        **Lane-lock coupling gap (task 2442 review — CLOSED by task 2599)**:
        the mutual exclusion above additionally requires the OTHER party in
        a concurrent re-acquire — reify's
        ``<lane_dir>/scripts/seed-warm-lane.sh``, invoked by
        :meth:`_seed_warm_lane` — to ALSO take ``<lane_dir>.lock`` before
        writing into ``target/``. It now does: :meth:`_seed_warm_lane` wraps
        that script in an outer blocking exclusive ``flock -x
        <lane_dir>.lock`` spanning its full subprocess duration (task 2599),
        bounded rather than unbounded (task 2599 amendment — see that
        method's "Bounded wait, not unbounded" docstring note), nested
        outside its own gen-dir flock (mirrored on the DF side,
        above). So a genuine concurrent re-acquire's seed DOES contend with
        this method's ``rm -rf`` on the same lock file — the rc=75
        benign-skip path below is now reachable on a real re-acquire race,
        not only when something ELSE (another thin or GC invocation) holds
        the lock. DF's own :class:`WarmLanePool` still does not itself
        provide this — its ASSIGNED/FREE bookkeeping is purely in-memory,
        guarded by a single ``asyncio.Lock`` (see its class docstring) that
        serializes the fast state-dict flip only, NOT the slow subprocess
        calls either side makes — but that no longer matters: the
        ``<lane_dir>.lock`` file itself is now the shared mutual-exclusion
        primitive between this method's ``rm -rf`` and
        :meth:`_seed_warm_lane`'s ``cp --reflink``, both real OS subprocesses
        that would otherwise be free to genuinely overlap in wall-clock time
        across an ``await``. ``TestSeedWarmLaneTakesLaneLock`` in
        ``test_git_ops.py`` pins :meth:`_seed_warm_lane`'s new blocking
        behavior against an externally-held lock, and
        ``TestSeedAndThinMutualExclusion`` pins the end-to-end mutual
        exclusion between this method and :meth:`_seed_warm_lane` racing
        each other on the SAME lock file.

        Exit-code taxonomy (per the reify δ contract):
            0   — thinned (``target/`` removed).
            1   — guard refusal (logged at WARNING).
            2   — usage error (logged at WARNING).
            75  — EX_TEMPFAIL: the lane's own flock is held (already
                  re-acquired concurrently) — a BENIGN skip, never logged at
                  WARNING (§9.5 inv.11: release-thin is not an
                  escalation/fault).
            127 — no implementation at either location, pool storage absent,
                  or unexpected exception (fail-soft sentinel).

        Never raises.
        """
        if self.pool_in_use() and not self.pool_storage_present():
            logger.warning(
                '_run_thin_warm_lane: pool storage absent/unmounted at %s — '
                'refusing to spawn thin-warm-lane.sh for %s',
                self.worktree_base, lane_dir,
            )
            self._note_pool_storage_absent()
            return 127
        try:
            resolved = self._resolve_warm_lane_script_logged(
                'thin-warm-lane.sh', '_run_thin_warm_lane',
            )
            if resolved is None:
                return 127
            script, _origin = resolved
            cmd = [str(script), str(lane_dir)]
            rc, _, err = await _run(cmd, cwd=self.project_root)
            if rc == 0:
                logger.info('_run_thin_warm_lane: thinned %s', lane_dir)
            elif rc == 75:
                logger.debug(
                    '_run_thin_warm_lane: lane %s already re-acquired '
                    '(rc=75, flock held) — benign skip',
                    lane_dir,
                )
            else:
                logger.warning(
                    '_run_thin_warm_lane: script exited %d for %s (stderr=%r)',
                    rc, lane_dir, err,
                )
            return rc
        except Exception:
            logger.warning(
                '_run_thin_warm_lane: unexpected error for %s', lane_dir, exc_info=True,
            )
            return 127

    async def _warm_lane_disk_admission_blocked(self) -> bool:
        """Run the ε disk-pressure admission check: check → reclaim → recheck.

        Mirrors ``merge_queue._ensure_verify_disk_space`` (check → prune → recheck →
        block) for the warm-lane acquire side.

        Logic:
            1. Run γ ``warm-lane-disk-guard.sh check``.
            2. Exit 0 or 127 or any non-(0,75) → admit (return False, fail-open).
            3. Exit 75 (disk pressure) → log WARNING, run δ ``warm-lane-gc.sh reclaim``
               (fail-soft; reclaim failure is logged but never blocks admission),
               re-run γ check.
            4. Re-check exit 75 → still pressured → block (return True).
               Re-check exit 0/other → recovered → admit (return False).

        Never raises.
        """
        rc = await self._run_warm_lane_disk_guard()
        if rc != 75:
            # 0 = healthy; 127 = absent (fail-open); other = script error (fail-open)
            return False
        # Disk pressure detected — attempt GC reclaim
        logger.warning(
            '_warm_lane_disk_admission_blocked: disk pressure detected (rc=75); '
            'invoking warm-lane-gc.sh reclaim before re-check',
        )
        await self._run_warm_lane_gc_reclaim()
        rc2 = await self._run_warm_lane_disk_guard()
        if rc2 == 75:
            logger.warning(
                '_warm_lane_disk_admission_blocked: disk pressure persists after '
                'reclaim (rc=75); blocking warm-lane admission',
            )
            return True
        return False

    async def _warm_lane_soft_pressure_defer(self, branch_name: str) -> bool:
        """θ proactive soft-floor throttle decision (task 2443, §9.5 inv.11).

        Mirrors :meth:`_warm_lane_disk_admission_blocked`'s shape, one floor
        earlier: runs :meth:`_run_warm_lane_soft_guard` and defers (True) on
        rc==3 (soft pressure — above the hard floor, below the soft one).

        rc==75 (hard pressure) ALSO defers, unconditionally — independent of
        the ε (``warm_lane_disk_guard``) knob (amendment,
        reviewer_comprehensive robustness). The soft guard runs the same
        reify script with both the hard AND soft flags, so it reports rc==75
        whenever free space/inodes are actually below the hard floor ("75
        takes precedence over soft" per the reify contract); rc==75 is never
        healthy, so deferring is strictly safer than failing open. In the
        common ε-enabled case a genuine below-hard-floor condition is caught
        UPSTREAM by :meth:`_warm_lane_disk_admission_blocked` in
        ``_acquire_warm_lane_impl`` (ε short-circuits to DISK_PRESSURE before
        this method runs), so this rc==75 arm only fires in the narrow TOCTOU
        window where free space fell below the hard floor BETWEEN ε's check
        and this one — deferring there (rather than failing open into a fresh
        below-hard-floor allocation) closes that window. It also covers the
        soft-only configuration (``warm_lane_soft_floor=True``,
        ``warm_lane_disk_guard=False``), where nothing else observes the
        hard-floor signal for a FRESH allocation. Either way the defer is
        pure backpressure (inv.11: routes through the same
        WarmLaneSoftPressure REQUEUE, never an escalation or ε's
        exit-75/WarmLaneDiskPressure fault path) and never touches ε's
        byte-identical hard path.

        Every other outcome fails open (False): 0 (healthy), 127 (script
        absent), 2 (usage error), or any other unrecognized code.

        On a defer, emits a structured WARNING journal line naming
        *branch_name* — the user-observable B10 signal that a fresh
        allocation was throttled as backpressure (inv.11: never an
        escalation or fault). The journal line is enriched with α's HEADROOM
        summary (:meth:`_warm_lane_audit_cached`, a short-window memo over
        :meth:`_run_warm_lane_audit` so a sustained soft-pressure condition
        does not re-fork the audit subprocess on every requeue cycle) for
        operator context — OBSERVABILITY ONLY (inv.12): a ``None`` headroom
        (α absent/errored) degrades the log line gracefully and never affects
        the ``True`` return value below; α is never consulted in the decision
        itself, only after it has already been made. The WARNING itself is
        emitted on every defer (it is the intended backpressure signal); only
        the audit subprocess is rate-limited.

        Never raises.
        """
        rc = await self._run_warm_lane_soft_guard()
        # Defer on soft pressure (rc==3) OR hard pressure (rc==75). rc==75 is
        # never healthy — the soft guard runs the same reify script with both
        # the hard and soft thresholds, so it reports 75 whenever free space
        # is actually below the hard floor ("75 takes precedence over soft").
        # Deferring is strictly safer than failing open, so we do it
        # unconditionally, independent of the ε (warm_lane_disk_guard) knob
        # (amendment, reviewer_comprehensive robustness). See docstring.
        if rc not in (3, 75):
            return False
        # α (inv.12, observability-only) — memoized for a short window so a
        # sustained soft-pressure condition that requeues the same fresh
        # allocation across many dispatch cycles does not re-fork the audit
        # subprocess on every cycle (amendment, reviewer_comprehensive
        # performance-efficiency).  Never affects the return value below.
        headroom = await self._warm_lane_audit_cached()
        reason = (
            'soft disk pressure'
            if rc == 3
            else 'hard disk pressure (rc=75 — free fell below the hard floor)'
        )
        logger.warning(
            'θ soft-floor throttle: %s (rc=%d) for branch %r — deferring '
            'dispatch (backpressure, inv.11); audit_headroom=%s',
            reason, rc, branch_name, headroom,
        )
        return True

    async def acquire_spec_lane(
        self,
        merge_commit: str,
    ) -> tuple[Path, bool]:
        """Acquire a warm ``_spec-`` lane for a speculative merge verify (inv.8).

        **Create-once path** (lane not yet a registered worktree):
            ``git worktree add --detach <lane> <merge_commit>``, then seed via
            :meth:`_seed_warm_lane(lane, '--reset-in-place')`.

        **Reset-in-place path** (lane already registered):
            ``git reset --hard <merge_commit>`` + ONE ``git clean -xfd -e <dir>``
            invocation (one -e per :attr:`~GitConfig.reap_build_artifact_dirs`
            entry) mirroring :meth:`reset_persistent_merge_worktree`'s pattern,
            then always re-seeded via :meth:`_seed_warm_lane(lane, '--reset-in-place')`.

        **inv.8 always-re-seed-at-acquire**: target/ is re-CoW-seeded from the
        current warm base on every acquire, so each speculative verify starts
        from the latest base regardless of which lane it lands on.

        **Cold-fallback path** (pool exhausted or seed failure — inv.6):
            On any failure (pool exhausted, worktree creation failure, seed
            failure) the partially-acquired lane (if any) is released back to
            FREE and an ephemeral cold worktree is returned with warm=False.
            Logs at DEBUG; never raises (a fallback cannot block the scheduler).

        Args:
            merge_commit: The merge commit SHA to check out in the spec lane.

        Returns:
            ``(lane_path, True)`` when the warm spec lane is ready;
            ``(cold_path, False)`` on pool exhaustion or any failure (inv.6).
        """
        if self.spec_warm_lane_pool is None:
            # Knob off or size=0 — always cold (byte-identical to default).
            logger.debug(
                'acquire_spec_lane: spec pool absent (knob off) — cold fallback '
                'for %s', merge_commit[:8],
            )
            wt = await self.create_throwaway_verify_worktree(merge_commit)
            return wt, False

        lane = await self.spec_warm_lane_pool.try_acquire()
        if lane is None:
            # Pool exhausted — inv.6: cold ephemeral fallback, never block.
            logger.debug(
                'acquire_spec_lane: pool exhausted — cold fallback for %s',
                merge_commit[:8],
            )
            wt = await self.create_throwaway_verify_worktree(merge_commit)
            return wt, False

        # ── Create-once or reset-in-place ────────────────────────────────────
        if not await self._is_registered_worktree(lane):
            # Create-once: serialized via _spec_wt_create_lock so that the K>1
            # initial warm-up burst does not issue concurrent `git worktree add`
            # calls against the same repo-level git lock (which would cause
            # transient failures and a burst of cold fallbacks at warm-up time).
            # Reset-in-place acquires (already-registered path below) are
            # per-lane and don't contend, so no lock is needed there.
            async with self._spec_wt_create_lock:
                # Pool ownership is per-lane exclusive (try_acquire assigns a
                # unique lane to exactly one caller), so no double-check is needed
                # here — we're the only caller that can be creating this lane.
                # The lock purely serializes the repo-level git worktree add
                # against other lanes' concurrent first-time creates.
                # Pool-storage-absent discriminator (task 2099). This is the
                # PRIMARY guard for the spec pool — unlike acquire_warm_lane,
                # acquire_spec_lane has no task-2061 base-health pre-gate, and
                # without this check `git worktree add --detach` below would
                # create a shadow lane on the rootfs skeleton during a
                # mount-down window. Same missing-vs-empty distinction as
                # acquire_warm_lane's create-once discriminator: base MISSING
                # => genuinely fresh host => allow create; base EXISTS but
                # `.pool-root` absent => suspected unmounted mountpoint =>
                # refuse (release lane, cold fallback).
                if self.worktree_base.exists() and not self.pool_storage_present():
                    if self._pool_storage_bootstrap_ok():
                        # First-seed bootstrap (task 2099 review-fix): the CoW
                        # seed base is present & non-empty INSIDE worktree_base,
                        # proving the mount is up — a missing sentinel here is a
                        # fresh host, not an unmounted mountpoint. Mark it now so
                        # subsequent acquisitions see storage present, then fall
                        # through to the normal create (the seed re-marks it).
                        logger.info(
                            'acquire_spec_lane: create-once — .pool-root absent '
                            'but warm base present under %s; first-seed bootstrap '
                            '(marking sentinel, proceeding)', self.worktree_base,
                        )
                        self.mark_pool_storage_present()
                    else:
                        logger.warning(
                            'acquire_spec_lane: create-once — pool storage '
                            'absent/unmounted at %s — refusing to create spec '
                            'lane %s on the underlying filesystem; cold fallback',
                            self.worktree_base, lane,
                        )
                        self._note_pool_storage_absent()
                        await self.spec_warm_lane_pool.release(lane)
                        wt = await self.create_throwaway_verify_worktree(merge_commit)
                        return wt, False
                # Self-heal a stale unregistered directory first
                # (mirrors reset_persistent_merge_worktree's self-heal pattern).
                if lane.exists():
                    import shutil as _shutil
                    _shutil.rmtree(lane)
                lane.parent.mkdir(parents=True, exist_ok=True)
                rc, _, err = await _run(
                    ['git', 'worktree', 'add', '--detach', str(lane), merge_commit],
                    cwd=self.project_root,
                )
                if rc != 0:
                    logger.warning(
                        'acquire_spec_lane: worktree add failed for %s (rc=%d,'
                        ' err=%r) — releasing lane, cold fallback', lane, rc, err,
                    )
                    await self.spec_warm_lane_pool.release(lane)
                    wt = await self.create_throwaway_verify_worktree(merge_commit)
                    return wt, False
                logger.info(
                    'acquire_spec_lane: created %s (HEAD=%s)', lane, merge_commit[:8],
                )
        else:
            # Reset-in-place: mirror reset_persistent_merge_worktree's exact sequence
            # (git reset --hard + ONE git clean with all -e flags in a single pass).
            rc, _, err = await _run(
                ['git', 'reset', '--hard', merge_commit],
                cwd=lane,
            )
            if rc != 0:
                logger.debug(
                    'acquire_spec_lane: reset --hard failed for %s (rc=%d, err=%r)'
                    ' — releasing lane, cold fallback', lane, rc, err,
                )
                await self.spec_warm_lane_pool.release(lane)
                wt = await self.create_throwaway_verify_worktree(merge_commit)
                return wt, False
            ok, err = await self._clean_lane_retaining_artifacts(
                lane, caller='acquire_spec_lane',
            )
            if not ok:
                logger.debug(
                    'acquire_spec_lane: git clean failed for %s (err=%r)'
                    ' — releasing lane, cold fallback', lane, err,
                )
                await self.spec_warm_lane_pool.release(lane)
                wt = await self.create_throwaway_verify_worktree(merge_commit)
                return wt, False
            logger.info(
                'acquire_spec_lane: reset %s to HEAD=%s', lane, merge_commit[:8],
            )

        # ── inv.8: ALWAYS re-seed target/ from the current warm base ─────────
        rc = await self._seed_warm_lane(lane, '--reset-in-place')
        if rc != 0:
            # Seed failed — release the partially-acquired lane back to FREE
            # (inv.6: cold fallback on seed failure, never block scheduler).
            logger.debug(
                'acquire_spec_lane: seed failed for %s (rc=%d) — releasing lane, cold fallback',
                lane, rc,
            )
            await self.spec_warm_lane_pool.release(lane)
            wt = await self.create_throwaway_verify_worktree(merge_commit)
            return wt, False

        return lane, True

    async def release_spec_lane(self, lane: Path, *, warm: bool) -> None:
        """Release a ``_spec-`` lane after a speculative merge verify.

        **Warm path** (``warm=True`` — pool lane):
            ``await self.spec_warm_lane_pool.release(lane)`` — flips
            ASSIGNED→FREE retaining the worktree and ``target/`` on disk so
            the next :meth:`acquire_spec_lane` can re-seed from a warm base
            (CoW-cheap, harmless to retain).  The worktree is never removed.

        **Cold path** (``warm=False`` — ephemeral fallback worktree):
            ``await self.cleanup_merge_worktree(lane)`` — removes the
            throwaway ``_merge-<uuid>`` worktree created by
            :meth:`create_throwaway_verify_worktree`.

        **Idempotent**: releasing a FREE lane, an already-cleaned path, or an
        unknown path is a no-op (never raises).  Mirrors the
        ``release_warm_lane`` best-effort / never-raise contract so a hiccup
        cannot strand the scheduler.

        Args:
            lane: Path returned by :meth:`acquire_spec_lane`.
            warm: True when *lane* is a warm ``_spec-`` pool lane; False when
                it is a cold ephemeral fallback worktree.
        """
        try:
            if warm and self.spec_warm_lane_pool is not None:
                await self.spec_warm_lane_pool.release(lane)
                logger.debug('release_spec_lane: released warm lane %s', lane)
            else:
                await self.cleanup_merge_worktree(lane)
                logger.debug('release_spec_lane: cleaned up cold lane %s', lane)
        except Exception:
            logger.warning(
                'release_spec_lane: error releasing %s (warm=%s)',
                lane, warm, exc_info=True,
            )

    async def _try_reclaim_lane_for(
        self,
        branch_name: str,
        *,
        title: str | None = None,
        branch: str | None = None,
    ) -> Path | None:
        """Attempt to steal a non-dispatched non-terminal lane for *branch_name*.

        Called from :meth:`acquire_warm_lane` when :meth:`acquire_for` returns
        None (pool exhausted).  Returns the reclaimed lane Path on success, or
        None if either callback is not wired, the candidate set is empty, or
        no eligible victim exists.

        *title*/*branch* are threaded into :meth:`WarmLanePool.reclaim_victim`'s
        durable write-through (task 2986, single writer) so the re-keyed
        ASSIGNED record keeps the thief's task_id/title/branch.

        Victim eligibility (checked by :meth:`WarmLanePool.reclaim_victim`):
        - ``victim != branch_name`` — never steal from self.
        - ``victim in candidates`` — non-terminal, per the async provider.
        - ``not is_dispatched(victim)`` — re-checked atomically under the pool
          lock (TOCTOU guard; see design note in task 1933).
        - ``lane state == ASSIGNED`` — only steal a live assignment.

        Before routing the stolen lane into the reset, commits any uncommitted
        *tracked* WIP onto the victim's still-checked-out branch so 1912
        branch-retention preserves it via the retained branch ref for future
        ``reattach`` recovery.  ``.task/plan.json`` is intentionally excluded
        (task metadata lives outside the worktree for the orchestrator hot
        path; any leftover ``.task/`` is covered by this repo's root
        ``.gitignore`` ``.task/`` entry) and is **not** preserved across the
        reclaim; the resumed victim takes the orphan-commits reattach path,
        not disk-backstop reuse.

        Emits a WARNING on every steal as an ops signal that pool pressure
        required the safety valve.
        """
        if (
            self.warm_lane_reclaim_candidate_provider is None
            or self.warm_lane_dispatched_predicate is None
            or self.warm_lane_pool is None
        ):
            return None

        pool = self.warm_lane_pool
        candidates = list(pool.assignments_snapshot().keys())
        if not candidates:
            return None

        eligible = await self.warm_lane_reclaim_candidate_provider(candidates)
        if not eligible:
            return None

        victim_result = await pool.reclaim_victim(
            branch_name, eligible, self.warm_lane_dispatched_predicate,
            title=title, branch=branch,
        )
        if victim_result is None:
            return None

        victim_branch, lane = victim_result
        # Best-effort: commit any uncommitted WIP onto the victim's branch
        # (which is still checked out at this lane) before resetting it.
        # 1912 branch-retention preserves the committed WIP so the resumed
        # victim recovers it via the reattach path.
        try:
            await self.commit(
                lane,
                'chore: save WIP before warm-lane reclaim (task 1933)',
            )
        except Exception:
            logger.warning(
                'acquire_warm_lane: reclaim WIP commit failed for lane %s '
                '(victim=%r) — proceeding with reset; uncommitted WIP may be lost',
                lane, victim_branch,
            )

        logger.warning(
            'acquire_warm_lane: reclaim-on-exhaustion — stole lane %s from '
            'non-dispatched task %r for %r (pool pressure)',
            lane, victim_branch, branch_name,
        )
        return lane

    def _assemble_warm_lane_census(self) -> WarmLanePoolCensus:
        """Assemble the typed warm-lane pool census (PRD α / W2a).

        The SINGLE census assembler (INV-5 / PRD dec.7): reads
        ``self.warm_lane_pool`` + ``self.warm_lane_dispatched_predicate`` and
        counts durable QUARANTINED records via
        ``self._lane_lifecycle.all_records()``, then delegates the pure
        classification to :meth:`WarmLanePool.census`.  Both α consumers — the
        WARNING at the EXHAUSTED return and the ``WarmLanePoolExhausted``
        message at the raise site — call this; PRD β (MCP tool) and ε
        (structural-exhaustion L2) will too, so the four consumers cannot drift
        in how the counts are derived.

        Never raises: this is a diagnostic on the error path and must not mask
        the EXHAUSTED signal.
        - Pool disabled (``warm_lane_pool is None``): returns an all-zero
          census (defensive; the α call sites only reach here with the pool
          enabled, but the assembler stays total for β/ε reuse).
        - Durable-record scan I/O error (``OSError``): logs a WARNING and
          counts ``n_quarantined`` as 0 rather than letting a disk hiccup
          crash the census (loud-over-silent, but never mis-loud).
        """
        if self.warm_lane_pool is None:
            return WarmLanePoolCensus(
                size=0,
                n_free=0,
                n_assigned_dispatched=0,
                n_pinned_non_dispatched=0,
                n_unknown_dispatch=0,
                n_quarantined=0,
            )
        try:
            n_quarantined = sum(
                1
                for record in self._lane_lifecycle.all_records().values()
                if record.state is LaneState.QUARANTINED
            )
        except OSError:
            logger.warning(
                'acquire_warm_lane: census could not scan durable lane '
                'records for QUARANTINED count — reporting n_quarantined=0',
                exc_info=True,
            )
            n_quarantined = 0
        return self.warm_lane_pool.census(
            is_dispatched=self.warm_lane_dispatched_predicate,
            n_quarantined=n_quarantined,
        )

    def _note_structural_exhaustion(self, census: 'WarmLanePoolCensus') -> None:
        """Count consecutive pool EXHAUSTED; fire the loudness callback at
        threshold (task 2988, PRD ε pole-2 — the silent-infinite-requeue pole).

        Increments the pool-GLOBAL ``_consecutive_exhausted`` counter (reset to 0
        on any successful FRESH allocation or safety-valve reclaim in
        :meth:`_acquire_warm_lane_impl`).  Once the counter reaches
        ``config.warm_lane_structural_exhaustion_l2_threshold`` and a
        ``_on_structural_exhaustion`` callback is installed (by the Harness when
        a pool exists), fires it with ``(count, census)`` so the Harness files
        ONE deduped born-at-L2 — the sole loud signal that the pool is stuck
        handing out EXHAUSTED forever.

        RATE-LIMITED (review amendment — efficiency): fires on the EXACT
        threshold crossing and then only every
        ``_STRUCTURAL_EXHAUSTION_L2_REFIRE_EVERY``-th subsequent consecutive
        EXHAUSTED — NOT on every trip.  Unlike the :class:`WarmLanePool`
        drift-counter (which resets on every successful round, so its per-trip
        fire is naturally bounded), this counter never resets while the pool
        stays stuck, so firing on every trip would run the harness filer's
        O(pending-escalations) dedup scan (``find_pending_l2_by_root_cause``) on
        the acquire chokepoint for every attempt.  The periodic re-fire still
        re-files the born-at-L2 if an operator resolves it while the pool remains
        structurally exhausted; dedup keeps at most one pending L2 regardless of
        trip count.  Callback exceptions are swallowed+logged (I3 fail-open) —
        escalation-filing must never break the acquire path.  Increment + fire
        run synchronously (no ``await``), so they are atomic under the single
        asyncio loop despite concurrent acquires — the same property the pool
        drift-counter relies on.
        """
        self._consecutive_exhausted += 1
        count = self._consecutive_exhausted
        threshold = self.config.warm_lane_structural_exhaustion_l2_threshold
        # Fire on the crossing (count == threshold → (count - threshold) == 0)
        # and every _STRUCTURAL_EXHAUSTION_L2_REFIRE_EVERY-th trip after it —
        # never on every acquire (see the constant's rationale).
        if (
            count >= threshold
            and (count - threshold) % _STRUCTURAL_EXHAUSTION_L2_REFIRE_EVERY == 0
            and self._on_structural_exhaustion is not None
        ):
            try:
                self._on_structural_exhaustion(count, census)
            except Exception:
                logger.warning(
                    'acquire_warm_lane: _on_structural_exhaustion callback raised '
                    '(consecutive_exhausted=%d) — swallowed (fail-open, I3)',
                    count, exc_info=True,
                )

    async def acquire_warm_lane(
        self,
        branch_name: str,
        start_ref: str,
        *,
        expected_title: str | None = None,
    ) -> 'WorktreeInfo | WarmLaneUnavailable':
        """Bare passthrough to :meth:`_acquire_warm_lane_impl`.

        Delegates to :meth:`_acquire_warm_lane_impl` for the full acquire
        logic (see that method's docstring for the complete contract). The
        durable ASSIGNED lifecycle edge is recorded INSIDE the impl, at each
        named route's success return, via :meth:`_note_assigned_via_route`
        (PRD W11 eta Mechanism 3) — so this wrapper no longer needs a
        post-hoc chokepoint. Fault paths return WarmLaneUnavailable (never
        WorktreeInfo), so they never write ASSIGNED — consistent with
        :meth:`_abort_lane_acquisition` teardown.
        """
        return await self._acquire_warm_lane_impl(
            branch_name, start_ref, expected_title=expected_title,
        )

    async def prewarm_pool(self, start_ref: str) -> PoolPrewarmResult:
        """Eagerly materialize every pool lane to its at-rest idle state (task 2879).

        ROOT CAUSE this addresses: the warm-lane pool is provisioned LAZILY —
        ``WarmLanePool`` builds ``effective_N`` FREE lanes in memory, but the
        on-disk ``git worktree add`` for each ``_lane-k`` happens only inside
        the acquire create-once branch, the first time that specific lane is
        acquired.  Since acquire hands out the lowest-numbered FREE lane first,
        a high-numbered lane materializes on disk only when that many lanes are
        simultaneously ASSIGNED — so on a host peaking below ``effective_N``
        the spare lanes are never demanded, never created, and the intended
        headroom is a phantom (present in the in-memory state machine, absent
        on disk).

        ``prewarm_pool`` closes that gap: for each lane not already registered
        it runs ``git worktree add --detach <lane> <start_ref>`` then
        :meth:`_seed_warm_lane` ``--fresh-checkout``, leaving the pool slot
        FREE.  A prewarmed lane is byte-identical to a RELEASED idle lane (a
        registered worktree on a DETACHED HEAD — no ``task/...`` branch — with
        a seeded ``target/``), so the EXISTING reset-in-place acquire path
        adopts it unchanged: a brand-new task id has no orphan commits, so
        ``_reset_warm_lane`` does ``checkout -f -B task/<id> start_ref`` from
        the detached HEAD.

        Contract:
        - **No-op when the pool is disabled** (``warm_lane_pool is None``) —
          returns ``PoolPrewarmResult(target=0)``.  Callers still gate on
          ``warm_lane_pool is not None``; this keeps the method safe to call
          unconditionally.
        - **ABSENT-base short-circuit** — mirrors acquire's pre-acquire gate:
          when :meth:`_warm_lane_base_resolvable` is
          :attr:`WarmBaseHealth.ABSENT` (a provably missing/empty CoW seed
          base — a HOST-SCOPED condition, one base serves every lane), logs a
          WARNING and returns without touching any lane
          (``materialized == 0``).
        - **SEQUENTIAL** — ``git worktree add`` serializes on a repo-level
          administrative lock and concurrent adds from the same
          ``project_root`` transiently fail during a warm-up burst (the
          codebase already guards this for ``_spec-`` lanes via
          ``_spec_wt_create_lock``).  Startup is not latency-critical, and
          sequential creation also bounds the transient disk pressure of a
          create+seed burst.
        - **Idempotent** — a lane already registered on disk is counted
          ``already_resident`` and left completely untouched (no worktree add,
          no reseed), so prewarm is safely re-entrant across restarts and
          after the startup reconcile sweeps have restored existing lanes.
        - **Leaves every slot FREE** — prewarm materializes, it does NOT
          assign; it never calls ``acquire_for``/``note_assignment``.

        Returns a :class:`PoolPrewarmResult` (target / already_resident /
        materialized / failed / failures).

        Fail-open by contract — this method never raises.  A per-lane
        worktree-add or seed failure is logged, the half-created worktree
        torn down (:meth:`_teardown_prewarm_lane`), recorded in ``failures``,
        counted, and the loop continues to the next lane.  After the loop, a
        shortfall (``already_resident + materialized < target``) emits ONE
        summary WARNING — the VISIBLE SIGNAL that the pool could not reach
        ``effective_N``.
        """
        pool = self.warm_lane_pool
        if pool is None:
            return PoolPrewarmResult(target=0)

        lanes = pool.lane_paths()
        result = PoolPrewarmResult(target=len(lanes))

        # ABSENT-base gate — mirror acquire's pre-acquire short-circuit; touch
        # no lane when the CoW seed base is provably absent/empty.
        if self._warm_lane_base_resolvable() is WarmBaseHealth.ABSENT:
            logger.warning(
                'prewarm_pool: warm-lane CoW seed base %s is absent/empty — '
                'skipping prewarm of %d lane(s) (host-scoped pool condition); '
                'run reify/scripts/ensure-warm-base.sh to rebuild it',
                self.warm_lane_base_target_path, result.target,
            )
            return result

        for lane in lanes:
            # Idempotent resident-skip: leave an already-registered lane
            # completely untouched (no `git worktree add`, no `_seed_warm_lane`)
            # and count it `already_resident`. This is the sole re-entrancy
            # guard — it makes prewarm safe to call on every boot after the
            # startup reconcile sweeps have already restored existing lanes,
            # and makes a second prewarm on a fully-resident pool a pure no-op.
            if await self._is_registered_worktree(lane):
                result.already_resident += 1
                logger.debug(
                    'prewarm_pool: lane %s already resident — skipping '
                    '(no worktree add, no reseed)', lane,
                )
                continue
            # Wrap each lane's materialization so one bad lane never aborts the
            # loop or wedges startup — a per-lane failure is logged, the
            # half-created worktree torn down, counted, and the loop continues.
            try:
                # Create-once NEUTRAL: --detach (NOT -b task/...) so the lane
                # carries no task branch — byte-identical to a released idle
                # lane. Then CoW-seed target/ so the resident spare is a genuine
                # warm lane. Leave the pool slot FREE (never acquire_for/
                # note_assignment).
                add_rc, _, add_err = await _run(
                    ['git', 'worktree', 'add', '--detach', str(lane), start_ref],
                    cwd=self.project_root,
                )
                if add_rc != 0:
                    # The add itself failed — no worktree was created, so there
                    # is nothing to tear down.
                    logger.warning(
                        'prewarm_pool: git worktree add --detach failed (rc=%d) '
                        'for lane %s: %s', add_rc, lane, add_err.strip(),
                    )
                    result.failures.append((lane, add_rc))
                    result.failed += 1
                    continue

                seed_rc = await self._seed_warm_lane(lane, '--fresh-checkout')
                if seed_rc != 0:
                    # The add succeeded but the seed failed — tear the
                    # half-created worktree back down so a cold shell is never
                    # left registered-but-unseeded (a failed lane must not
                    # masquerade as a materialized warm lane, and would
                    # otherwise be adopted by acquire as if it were warm).
                    # rc sentinels mirror acquire's seed messages: 127 =
                    # seed-warm-lane.sh absent (deploy error); 75 = EX_TEMPFAIL
                    # disk pressure; any other non-zero = generic fault.
                    if seed_rc == 127:
                        logger.warning(
                            'prewarm_pool: seed script absent (rc=127) for lane '
                            '%s — check seed-warm-lane.sh deployment; tearing '
                            'down the half-created worktree', lane,
                        )
                    elif seed_rc == 75:
                        logger.warning(
                            'prewarm_pool: seed reported disk pressure (rc=75) '
                            'for lane %s — tearing down the half-created '
                            'worktree', lane,
                        )
                    else:
                        logger.warning(
                            'prewarm_pool: seed failed (rc=%d) for lane %s — '
                            'tearing down the half-created worktree',
                            seed_rc, lane,
                        )
                    await self._teardown_prewarm_lane(lane)
                    result.failures.append((lane, seed_rc))
                    result.failed += 1
                    continue

                result.materialized += 1
            except Exception as e:
                # Unexpected fault (e.g. a git subprocess raising) — never
                # propagate out of prewarm. Best-effort teardown of any
                # half-created worktree, count it failed, and continue so the
                # remaining lanes still get their chance. rc=-1 marks an
                # unexpected exception (distinct from any real git/seed rc).
                logger.warning(
                    'prewarm_pool: unexpected error materializing lane %s: %s '
                    '— counting as failed and continuing', lane, e, exc_info=True,
                )
                with contextlib.suppress(Exception):
                    await self._teardown_prewarm_lane(lane)
                result.failures.append((lane, -1))
                result.failed += 1
                continue

        # VISIBLE SIGNAL (task requirement): a disk/floor shortfall must never
        # silently cap the pool below effective_N. When fewer lanes are
        # resident than targeted, emit ONE loud structured-facts WARNING naming
        # the shortfall and the failing lanes, so an under-provisioned pool is
        # observable at startup rather than only surfacing later as downstream
        # acquire exhaustion.
        reached = result.already_resident + result.materialized
        if reached < result.target:
            logger.warning(
                'prewarm_pool: SHORTFALL — only %d/%d warm lane(s) resident '
                'after prewarm (already_resident=%d, materialized=%d, '
                'failed=%d); pool is below effective_N. Failing lanes: %s',
                reached, result.target, result.already_resident,
                result.materialized, result.failed,
                ', '.join(
                    f'{lane.name}(rc={rc})' for lane, rc in result.failures
                ) or '(none)',
            )
        else:
            logger.info(
                'prewarm_pool: pool fully materialized — %d/%d warm lane(s) '
                'resident (already_resident=%d, materialized=%d)',
                reached, result.target, result.already_resident,
                result.materialized,
            )

        return result

    async def _teardown_prewarm_lane(self, lane: Path) -> None:
        """Best-effort teardown of a half-created prewarm lane (task 2879).

        Used when ``git worktree add --detach`` succeeded but the subsequent
        seed failed: ``git worktree remove --force <lane>`` drops the worktree
        and its checkout, then a registration-global
        :meth:`_prune_registrations` clears any stale ``.git/worktrees`` admin
        entry left behind, so the failed lane is NOT left registered-but-cold
        (which acquire would otherwise adopt as if it were a warm lane).

        Never raises — teardown is best-effort by contract; a removal failure
        is logged and the prewarm loop still continues to the next lane. A
        prewarmed lane carries a DETACHED HEAD (no ``task/...`` branch), so
        there is never a branch ref to preserve/delete here — unlike acquire's
        ``_abort_lane_acquisition``, which must gate branch deletion.
        """
        try:
            rc, _, err = await _run(
                ['git', 'worktree', 'remove', '--force', str(lane)],
                cwd=self.project_root,
            )
            if rc != 0:
                logger.warning(
                    'prewarm_pool: worktree remove --force failed (rc=%d) for '
                    'lane %s: %s', rc, lane, err.strip(),
                )
        except Exception as e:
            logger.warning(
                'prewarm_pool: worktree remove --force raised for lane %s: %s',
                lane, e,
            )
        # Registration-global prune clears any stale admin entry (routed
        # through the guarded chokepoint so the pool-storage guard applies).
        await self._prune_registrations(context='prewarm_pool-teardown')

    async def _acquire_warm_lane_impl(
        self,
        branch_name: str,
        start_ref: str,
        *,
        expected_title: str | None = None,
    ) -> 'WorktreeInfo | WarmLaneUnavailable':
        """Allocate a FREE warm lane, seed/reset it, and return a WorktreeInfo.

        **Create-once path** (lane not yet a registered worktree):
            ``git worktree add -b task/<branch_name> <lane> <start_ref>``,
            then seed via :meth:`_seed_warm_lane(lane, '--fresh-checkout')`.

        **Reset-in-place path** (lane already registered — added in step-10):
            Handled by :meth:`_reset_warm_lane` then always re-seeded from the
            current base via ``_seed_warm_lane(lane, '--fresh-checkout')`` (D10
            always-re-seed-at-acquire).  The lane is the ``§9.5`` within-assignment
            reset_lane primitive; the re-seed layers on top to deliver at-head
            warmth for the NEW task.

        **Identity guard** (``expected_title``, step-26):
            When *expected_title* is supplied, any reuse candidate (in-memory
            map hit *or* on-disk plan.json match) has its stored title checked
            via :func:`identities_match`.  On mismatch the stale assignment is
            dropped from the pool and the lane is reset in-place (fresh path) —
            so a recycled-id task never inherits the prior task's ``.task/``
            state.  ``expected_title=None`` (default) disables the guard and
            all existing callers/tests are unaffected.

        **Pre-acquire base-health gate** (task 2061, runs FIRST — before the ε
        disk-guard and before :func:`acquire_for`):
            Calls :meth:`_warm_lane_base_resolvable`.  A definite
            ``WarmBaseHealth.ABSENT`` short-circuits with
            ``WarmLaneUnavailable.BASE_ABSENT`` — no lane is touched (no
            worktree-add, no seed invocation, disk-guard scripts never run).
            This is a HOST-SCOPED pool condition (one base serves every
            lane), so :meth:`create_worktree` raises
            :class:`WarmLanePoolHardDown` for it, and the workflow requeues
            (fail-open, inv.6) instead of escalating a per-task blocked+L1.
            ``WarmBaseHealth.INDETERMINATE`` (a transient stat/readlink
            hiccup) falls through to the normal acquire path — it must never
            masquerade as a genuine outage.

        **ε pre-acquire disk-guard** (``config.warm_lane_disk_guard``, task-1860):
            When the knob is True, runs γ ``warm-lane-disk-guard.sh check`` →
            on exit 75 (EX_TEMPFAIL/disk pressure) invokes δ ``warm-lane-gc.sh
            reclaim`` to free stale capacity → re-checks.  If STILL pressured,
            returns ``WarmLaneUnavailable.DISK_PRESSURE`` so
            :meth:`create_worktree` raises :class:`WarmLaneDiskPressure` →
            workflow requeues (exit-75) instead of proceeding into an ENOSPC
            build.  Runs BEFORE :func:`acquire_for` so all idle lanes remain
            FREE for δ's reclaim.  Fail-open on absent scripts (rc 127): no
            guard / nothing reclaimed — byte-identical to today until reify
            γ/δ are deployed.

        **θ proactive soft-floor throttle** (``config.warm_lane_soft_floor``,
        task 2443, §9.5): runs immediately AFTER the ε hard-floor check above
        and BEFORE :func:`acquire_for`, and ONLY for a FRESH allocation (no
        lane already mapped to *branch_name* — a reuse/live-requeue is never
        throttled).  When the knob is True, runs ε's ``warm-lane-disk-guard.sh
        check --soft`` (a soft floor ABOVE the hard floor).  On soft pressure
        (rc=3), returns ``WarmLaneUnavailable.SOFT_PRESSURE`` so
        :meth:`create_worktree` raises :class:`WarmLaneSoftPressure` →
        workflow requeues as backpressure (inv.11: never an escalation/fault;
        the hard-floor exit-75 path above is unchanged).  Independent of
        ``warm_lane_disk_guard``; fail-open on absent script (rc 127) —
        byte-identical to today until reify ships ``check --soft``.

        Returns:
            WorktreeInfo  — success; lane is ASSIGNED and seeded.
            WarmLaneUnavailable.EXHAUSTED — all pool lanes are ASSIGNED
                (backpressure; caller should requeue).
            WarmLaneUnavailable.FAULT — seed/worktree-add failure or absent
                seed script (infra fault; caller should escalate blocked+L1).
            WarmLaneUnavailable.DISK_PRESSURE — pre-acquire ε disk-guard
                detected persistent pressure (rc=75 after reclaim), OR seed
                exited 75 (EX_TEMPFAIL); transient disk pressure (caller should
                requeue with annotation).
            WarmLaneUnavailable.SOFT_PRESSURE — θ proactive soft-floor
                throttle detected soft pressure (rc=3) for a FRESH allocation
                (caller should requeue as backpressure; never an escalation —
                distinct from DISK_PRESSURE's exit-75 hard floor).
            WarmLaneUnavailable.BASE_ABSENT — pre-acquire base-health gate
                found the warm-lane CoW seed base provably absent/empty
                (:meth:`_warm_lane_base_resolvable` returned
                ``WarmBaseHealth.ABSENT``).  HOST-SCOPED pool condition;
                caller should requeue (fail-open), never escalate blocked+L1.
            WarmLaneUnavailable.DISABLED — pool knob is off; this is a
                programming error (callers MUST guard with
                ``if self.warm_lane_pool is not None`` before calling).
                Returned rather than raising so the function contract
                "never raises internally" holds; callers that receive
                DISABLED must NOT requeue — they should fall back to the
                cold path or raise immediately.

            Never raises internally; the lane is always released before
            returning any WarmLaneUnavailable value.
        """
        if self.warm_lane_pool is None:
            # Programming error: callers must guard with
            # `if self.warm_lane_pool is not None` before invoking.
            # A disabled pool is NOT 'backpressure' — returning EXHAUSTED
            # here would silently requeue a task forever if a future caller
            # skips the guard.  DISABLED is a distinct sentinel so callers
            # that don't handle it hit an explicit error path rather than
            # infinite requeue.
            return WarmLaneUnavailable.DISABLED

        # Task 2061: pre-acquire base-health gate.  Runs FIRST (before the ε
        # disk-guard and before acquire_for) so a definite ABSENT short-circuits
        # without running the disk-guard scripts or touching/assigning any lane.
        # A base-absent CoW seed base is a HOST-SCOPED pool condition (one base
        # serves every lane) — create_worktree maps BASE_ABSENT to
        # WarmLanePoolHardDown so the workflow requeues (fail-open, inv.6)
        # instead of every dispatched task independently faulting into a
        # per-task blocked+L1 escalation.  INDETERMINATE (a transient
        # stat/readlink hiccup) deliberately falls through to the normal
        # acquire path below — it must never masquerade as a genuine outage.
        if self._warm_lane_base_resolvable() is WarmBaseHealth.ABSENT:
            logger.warning(
                'acquire_warm_lane: warm-lane CoW seed base %s is absent/empty '
                '— refusing to allocate a lane (host-scoped pool condition); '
                'run reify/scripts/ensure-warm-base.sh to rebuild it',
                self.warm_lane_base_target_path,
            )
            return WarmLaneUnavailable.BASE_ABSENT

        # ε: pre-acquire disk-guard (check → reclaim → recheck → DISK_PRESSURE/exit-75).
        # Running BEFORE acquire_for keeps all idle lanes FREE so δ's reclaim can reset
        # them.  On still-pressured (rc=75 after reclaim), return DISK_PRESSURE so
        # create_worktree raises WarmLaneDiskPressure → workflow requeues as transient
        # infra (exit-75) instead of proceeding into an ENOSPC build that SIGBUSes the
        # linker.  Fail-open (absent script rc=127) → byte-identical to today.
        #
        # Concurrency note: the guard + δ reclaim run WITHOUT holding warm_lane_pool
        # _lock.  This is intentional — serializing on the pool lock would prevent
        # concurrent acquires from even attempting during a reclaim.  The safety
        # invariant is delegated to the γ/δ scripts: warm-lane-gc.sh MUST only reset
        # lanes that have been idle beyond a configurable safety threshold (e.g. no
        # active acquire_for in flight for those lanes).  Because acquire_for itself
        # holds the pool lock for the registration step, a lane that is mid-acquire
        # will be ASSIGNED in pool state and δ must skip ASSIGNED lanes.  Until reify
        # δ (task-4717) ships with that guarantee documented, the guard is fail-open
        # (rc 127) and the window is a no-op; document it here so the δ author knows
        # the expected contract.
        if self.config.warm_lane_disk_guard and await self._warm_lane_disk_admission_blocked():
            return WarmLaneUnavailable.DISK_PRESSURE

        # θ: proactive soft-floor throttle (task 2443, §9.5 inv.11/inv.12) —
        # runs AFTER the ε hard-floor check above (so a hard-pressure result
        # always wins/short-circuits first, byte-identical ε precedence) and
        # BEFORE acquire_for (so a defer touches no lane; it stays FREE, same
        # as ε).  Gated on BOTH the independent master knob AND a FRESH
        # allocation: an already-mapped branch (assignment_for is not None)
        # is a reuse/live-requeue, not new resident-divergent growth, so it
        # is never throttled — only a fresh lane allocation defers.
        #
        # Amendment (reviewer_comprehensive robustness): _warm_lane_soft_pressure_defer
        # treats the soft-guard's own rc==75 (hard pressure) as a defer signal
        # unconditionally — closing the narrow TOCTOU window where free space
        # falls below the hard floor between ε's check above and the soft
        # guard, and covering the soft-only config (ε off, θ on alone). Either
        # way a fresh lane is never allocated straight past the hard floor.
        # See that method's docstring for detail.
        if (
            self.config.warm_lane_soft_floor
            and self.warm_lane_pool.assignment_for(branch_name) is None
            and await self._warm_lane_soft_pressure_defer(branch_name)
        ):
            return WarmLaneUnavailable.SOFT_PRESSURE

        # Computed BEFORE acquire_for so it can be threaded into the pool's
        # durable write-through (task 2986, single writer): the pool writes the
        # ASSIGNED record with task_id/title/branch at the moment it flips the
        # in-memory slot, so GitOps must supply the branch (and title) up front.
        full_branch = f'{self.config.branch_prefix}{branch_name}'

        acq = await self.warm_lane_pool.acquire_for(
            branch_name, title=expected_title, branch=full_branch,
        )
        if acq is None:
            # Pool exhausted — try to reclaim a non-dispatched non-terminal lane
            # before falling back to EXHAUSTED (task 1933 safety valve).
            reclaimed = await self._try_reclaim_lane_for(
                branch_name, title=expected_title, branch=full_branch,
            )
            if reclaimed is None:
                # Task 2984 (PRD α): carry the typed census on the exhaustion
                # path so an operator sees WHY the pool is full (free / held by
                # a dispatched task / pinned by a non-dispatched task / unknown
                # / quarantined).  Assembled fresh here (cheap on the rare
                # exhaustion path, race-free) rather than threaded through the
                # shared WarmLaneUnavailable sentinel.
                census = self._assemble_warm_lane_census()
                logger.warning(
                    'acquire_warm_lane: warm-lane pool EXHAUSTED for branch %r — %s',
                    branch_name, census.render(),
                )
                # Pole-2 (task 2988, PRD ε): count consecutive structural
                # exhaustion and, at threshold, fire the harness-installed
                # born-at-L2 loudness callback — reusing the SAME census just
                # assembled for the WARNING so the L2 carries identical counts
                # (INV-5, single assembler).  Best-effort / fail-open: never
                # breaks the acquire path.
                self._note_structural_exhaustion(census)
                return WarmLaneUnavailable.EXHAUSTED  # Pool exhausted → backpressure
            # Stolen lane: registered worktree, reused=False.  Falls through past
            # `if reused:` (False) and `elif not _is_registered_worktree(lane)`
            # (False — it IS registered) into the `else:` already-registered fresh-
            # reset path (_reset_and_seed_recycled_lane + shared tail), reusing all
            # existing reset/reseed/provision logic with zero new git plumbing.
            lane, reused = reclaimed, False
            # Pole-2 (task 2988): a successful safety-valve reclaim proves the
            # pool served a NEW lane — reset the consecutive-EXHAUSTED counter.
            self._consecutive_exhausted = 0
        else:
            lane, reused = acq
            if not reused:
                # A genuinely FRESH allocation (not a live-requeue reuse) proves
                # free capacity — reset the consecutive-EXHAUSTED counter.  A
                # reuse does NOT reset: it is not evidence of a free lane, only
                # of the same branch being re-handed its existing mapping.
                self._consecutive_exhausted = 0

        # Call-LOCAL route classifier (W11 eta, PRD Mechanism 3) — NOT
        # instance state: acquire_warm_lane runs concurrently for different
        # tasks/lanes, so shared state would race. Set at each branch's
        # routing decision below and consumed by _note_assigned_via_route at
        # every success return (never on a path that faults).
        route: AcquireRoute | None = None

        try:
            # ── Orphaned-lane reuse guard (task 2097) ───────────────────────
            # The reuse path (_reuse_warm_lane -> commit()) and the identity-
            # MISMATCH _reset_warm_lane call both assume a mapped lane is a
            # valid registered worktree. A lane whose dir + .git pointer
            # survive but whose .git/worktrees/<name> admin dir was pruned
            # (mount-down startup window) or corrupted (stale gitdir pointer)
            # is NOT registered and hard-faults ('not a git repository').
            # Try the cheapest recovery first — an in-place `git worktree
            # repair` — and only demote to the create-once self-heal/reattach
            # path (which recovers the committed work via alpha-retention,
            # task 1912) when repair cannot restore the registration. Never
            # FAULT a healable orphan.
            #
            # Perf note (review, task 2097): this puts one `git worktree list
            # --porcelain` fork on every warm-lane REUSE, including the
            # healthy common case. A cheap local stat-only pre-check was
            # considered and rejected: the step-3 repairable-registration
            # fixture leaves the admin dir itself present with only its
            # internal back-pointer corrupted, so a mere existence check on
            # the admin dir cannot distinguish "healthy" from "repairable"
            # without re-deriving `git worktree repair`'s own validation
            # rules — fragile to hand-roll without dedicated tests. Caching
            # the result is also out of scope for this task: it would live
            # in warm_lane_pool.py, which is not a locked module here.
            # Accepting the extra fork; revisit only if profiling shows it
            # matters against real reuse volume.
            #
            # `_orphan_confirmed_unregistered` is set True only when we reach
            # the demote below, at which point `_repair_orphaned_reuse_lane`
            # has ALREADY re-checked registration itself, freshly, as the
            # very last step of its own attempt — regardless of the repair
            # subprocess's exit code (see its docstring: exit code alone is
            # not authoritative). Basing the flag on that post-attempt check,
            # not on the pre-repair check a few lines up, matters: it means
            # the `elif` below can skip a third, redundant
            # `_is_registered_worktree` probe using the FRESHEST available
            # answer rather than a snapshot that predates the repair attempt
            # — if repair's side effects registered the lane despite a
            # nonzero exit, `_repair_orphaned_reuse_lane` already returned
            # True above and this demote never runs at all.
            _orphan_confirmed_unregistered = False
            # route=REUSE_REPAIR (W11 eta): distinguishes the repaired-in-
            # place route from plain REUSE below — set only when the lane
            # WAS unregistered and `_repair_orphaned_reuse_lane` restored its
            # registration in place (as opposed to already being registered,
            # which needs no repair and stays plain REUSE).
            if reused and not await self._is_registered_worktree(lane):
                if await self._repair_orphaned_reuse_lane(lane, branch_name):
                    route = AcquireRoute.REUSE_REPAIR
                else:
                    self.warm_lane_pool.drop_assignment(branch_name)
                    reused = False
                    _orphan_confirmed_unregistered = True
            # else: not reused, already registered, or repair restored the
            # registration in place — proceed (possibly still reused) below.

            if reused:
                # ── Reuse path: live requeue of same task on same lane ────
                # Identity guard: if expected_title is set, verify the stored
                # title matches before reusing .task/plan.json + WIP.
                # Fail-open: identities_match returns True when either side is
                # empty, so a title-less lane always reuses as before.
                _ident_ok = (
                    expected_title is None
                    or identities_match(read_worktree_title(lane), expected_title)
                )
                if _ident_ok:
                    # .task/plan.json is preserved; WIP is committed + rebased.
                    if route is None:
                        route = AcquireRoute.REUSE
                    info = await self._reuse_warm_lane(lane, full_branch)
                    self._note_assigned_via_route(
                        info.path, route, branch_name, expected_title, full_branch,
                    )
                    return info
                # Mismatch: stale assignment from a recycled id — drop it and
                # reset in-place so the new task starts clean.
                logger.warning(
                    'acquire_warm_lane: in-memory reuse identity MISMATCH for '
                    '%s — expected %r; running fresh reset',
                    lane, expected_title,
                )
                route = AcquireRoute.RECYCLE
                self.warm_lane_pool.drop_assignment(branch_name)
                await self._reset_warm_lane(lane, full_branch, start_ref)
                # β: THIN re-seed — rm target/ before seeding so the re-seed is a
                # fresh CoW copy rather than an in-place overlay on stale blobs.
                # Defensive rmtree is robust even if reify α's replace-capable seed
                # is not yet landed (an emptied target/ seeds cleanly either way).
                shutil.rmtree(lane / 'target', ignore_errors=True)
                rc = await self._seed_warm_lane(lane, '--fresh-checkout')
                if rc != 0:
                    if rc == 127:
                        logger.warning(
                            'acquire_warm_lane: recycle re-seed — seed script absent '
                            'for lane %s (rc=127) — check seed-warm-lane.sh '
                            'deployment; EVERY task on this host will fault while '
                            'pool is enabled and the script is missing',
                            lane,
                        )
                    else:
                        logger.warning(
                            'acquire_warm_lane: recycle re-seed failed (rc=%d) for '
                            '%s; releasing lane',
                            rc, lane,
                        )
                    # _abort_lane_acquisition (task 2199) — detaches HEAD
                    # first so the branch _reset_warm_lane just checked out
                    # here doesn't leak a "already used by worktree"
                    # collision for the next acquire (task 2062 mid-run leak).
                    await self._abort_lane_acquisition(
                        lane, branch_name, remove_worktree=False,
                    )
                    return _seed_rc_to_unavailable(rc)
                # Falls through to shared tail

            # `_orphan_confirmed_unregistered` (set above) short-circuits this
            # probe when the guard already confirmed the lane unregistered
            # and repair failed — avoids a third `git worktree list
            # --porcelain` fork for the same lane (review, task 2097) and
            # makes the routing decisive: once the guard commits to
            # demote-and-reattach we do not re-probe and risk flipping into
            # the already-registered/reset branch below on a (theoretical,
            # unlikely per review) registration flip-flop between the
            # guard's check and here.
            elif _orphan_confirmed_unregistered or not await self._is_registered_worktree(lane):
                # ── Create-once branch ────────────────────────────────────
                # Pool-storage-absent discriminator (task 2099). This is a
                # BACKSTOP behind the task-2061 base-health gate above
                # (_warm_lane_base_resolvable), which already short-circuits
                # the common BASE_ABSENT case before any lane is touched —
                # this catches its INDETERMINATE fall-through and off-mount-
                # base configs. worktree_base MISSING => an fstab mountpoint
                # dir must exist for the mount to attach, so a missing dir
                # cannot be an unmounted configured mount => genuinely fresh
                # host => safe to build the skeleton below. worktree_base
                # EXISTS but `.pool-root` is absent => suspected unmounted
                # mountpoint => refuse to create a NEW lane on the underlying
                # root fs.
                if self.worktree_base.exists() and not self.pool_storage_present():
                    if self._pool_storage_bootstrap_ok():
                        # First-seed bootstrap (task 2099 review-fix): the CoW
                        # seed base is present & non-empty INSIDE worktree_base,
                        # proving the mount is up — a missing sentinel here is a
                        # fresh host, not an unmounted mountpoint. Without this
                        # escape the very first warm lane on a fresh host would
                        # be refused forever (the sentinel's only writer,
                        # _seed_warm_lane, lives past this gate). Mark it now,
                        # then fall through to the normal create (the seed
                        # re-marks it idempotently).
                        logger.info(
                            'acquire_warm_lane: create-once — .pool-root absent '
                            'but warm base present under %s; first-seed bootstrap '
                            '(marking sentinel, proceeding)', self.worktree_base,
                        )
                        self.mark_pool_storage_present()
                    else:
                        logger.warning(
                            'acquire_warm_lane: create-once — pool storage '
                            'absent/unmounted at %s — refusing to create lane '
                            '%s on the underlying filesystem',
                            self.worktree_base, lane,
                        )
                        self._note_pool_storage_absent()
                        await self._abort_lane_acquisition(
                            lane, branch_name, remove_worktree=False,
                        )
                        return WarmLaneUnavailable.BASE_ABSENT
                # Self-heal: remove a stale unregistered directory so
                # git worktree add doesn't refuse a non-empty dir.
                if lane.exists():
                    logger.warning(
                        'acquire_warm_lane: lane %s exists but is not registered; '
                        'removing stale directory (self-heal)', lane,
                    )
                    shutil.rmtree(lane)
                lane.parent.mkdir(parents=True, exist_ok=True)

                # Clear the sibling .task-meta/<name> before (re)building the
                # worktree (W11 ε1). This create-once branch is reached only
                # for an UNREGISTERED lane, whose worktree is always rebuilt
                # fresh below (self-heal rmtree above, or `git worktree add`
                # into an absent dir). In the legacy `.task/` world the
                # metadata lived INSIDE the worktree, so that rebuild always
                # destroyed the prior occupant's plan.json/already_done.json/
                # false_premise.json. The relocated .task-meta/<name> lives
                # OUTSIDE the worktree and survives the rebuild unscathed, so
                # a DIFFERENT-task acquisition through this branch (orphan
                # window: registration lost but sibling meta survived, guard
                # at ~2846) would otherwise hand the incoming task the prior
                # occupant's metadata — a data-integrity regression relative
                # to the legacy cleanup (reviewer_comprehensive blocker,
                # create-once route). Cleared unconditionally so it covers
                # BOTH the FRESH (`add -b`) and REATTACH (`_reuse_warm_lane`)
                # sub-routes below AND the lane-dir-already-gone case where
                # the self-heal rmtree never ran. init()/_reuse_warm_lane
                # re-provision this task's own metadata afterward.
                self._clear_foreign_meta_root(lane)

                # ── γ reattach guard (create-once site) ──────────────────────
                # If the leftover task/<id> branch still carries commits beyond
                # main, attach the worktree to it (no -b) rather than letting
                # the -b create collide ('branch already exists').  Route through
                # _reuse_warm_lane after seeding — the REUSE tail (commit-WIP →
                # rebase_onto_main → re-provision), NOT the shared create tail
                # (gitignore/scrub/merge-base below).  Matches the disk-backstop
                # reuse path so orphan commits are rebased onto current main.
                #
                # _orphan_has_commits wraps the rev-parse --verify existence gate
                # + _branch_has_commits_beyond_main (fail-safe True on git error).
                if await self._orphan_has_commits(full_branch):
                    logger.info(
                        'acquire_warm_lane: reattach (create-once site) — '
                        'orphan %s has commits; attaching lane %s without -b',
                        full_branch, lane,
                    )
                    route = AcquireRoute.CREATE_ONCE_REATTACH
                    _co_add_rc, _, _co_err = await _run(
                        ['git', 'worktree', 'add', str(lane), full_branch],
                        cwd=self.project_root,
                    )
                    if _co_add_rc != 0:
                        # Cannot re-attach (e.g. branch is checked out in another
                        # live worktree).  Raise rather than falling through to any
                        # destructive op — mirrors _cleanup_leftover_branch's
                        # raise-not-destroy contract (inv.10 fail-safe-retain).
                        # acquire_warm_lane's top-level except Exception converts
                        # this to WarmLaneUnavailable.FAULT (lane released, caller
                        # escalates blocked+L1) while leaving full_branch intact.
                        raise RuntimeError(
                            f'acquire_warm_lane: refusing to reset {full_branch!r} '
                            f'— it carries commits beyond {self.config.main_branch} '
                            f'and cannot be safely re-attached to lane {lane} '
                            f'(git worktree add failed: {_co_err.strip()!r}). '
                            f'This would destroy work. Inspect the branch and, '
                            f'once any wanted work is preserved, remove the other '
                            f'worktree and retry: '
                            f'`git branch -D {full_branch}` only after preserving work.'
                        )
                    # Seed as in the normal create-once path; the post-seed tail
                    # is _reuse_warm_lane (commit-WIP → rebase → re-provision),
                    # NOT the shared create tail at the bottom of this method.
                    _co_seed_rc = await self._seed_warm_lane(lane, '--fresh-checkout')
                    if _co_seed_rc != 0:
                        if _co_seed_rc == 127:
                            logger.warning(
                                'acquire_warm_lane: create-once reattach seed script '
                                'absent for lane %s (rc=127)', lane,
                            )
                        else:
                            logger.warning(
                                'acquire_warm_lane: create-once reattach seed failed '
                                '(rc=%d) for %s; removing worktree',
                                _co_seed_rc, lane,
                            )
                        await self._abort_lane_acquisition(
                            lane, branch_name, remove_worktree=True,
                        )
                        return _seed_rc_to_unavailable(_co_seed_rc)
                    info = await self._reuse_warm_lane(lane, full_branch)
                    self._note_assigned_via_route(
                        info.path, route, branch_name, expected_title, full_branch,
                    )
                    return info

                git_add_rc, _, err = await _run(
                    ['git', 'worktree', 'add', '-b', full_branch, str(lane), start_ref],
                    cwd=self.project_root,
                )
                if git_add_rc != 0:
                    logger.warning(
                        'acquire_warm_lane: git worktree add failed for %s: %s', lane, err,
                    )
                    await self._abort_lane_acquisition(
                        lane, branch_name, remove_worktree=True,
                    )
                    return WarmLaneUnavailable.FAULT

                seed_rc = await self._seed_warm_lane(lane, '--fresh-checkout')
                if seed_rc != 0:
                    # Seed failed — remove the just-created worktree and release.
                    # rc=127 specifically means seed-warm-lane.sh is absent from
                    # the lane's scripts/ dir — a deploy/config error that will
                    # affect EVERY task on this host while the pool is enabled,
                    # producing one BLOCKED+L1 escalation per dispatched task.
                    # Operators should check that seed-warm-lane.sh is present
                    # and executable in the lane's checked-out scripts/ directory.
                    if seed_rc == 127:
                        logger.warning(
                            'acquire_warm_lane: seed script absent for lane %s '
                            '(rc=127) — check seed-warm-lane.sh deployment; '
                            'EVERY task on this host will fault while pool is '
                            'enabled and the script is missing',
                            lane,
                        )
                    else:
                        logger.warning(
                            'acquire_warm_lane: seed failed (rc=%d) for %s; '
                            'removing worktree',
                            seed_rc, lane,
                        )
                    # _abort_lane_acquisition (task 2199) now absorbs the
                    # former Task-2112/angle-A-2 degenerate-branch-delete
                    # logic that used to live here inline: `git worktree add
                    # -b full_branch <lane> start_ref` (above) already
                    # created full_branch — often at a foreign 'Merge
                    # task/<other> into main' commit — and merely removing
                    # the worktree would leave that zero-commit branch ref
                    # parked. The primitive gates deletion on
                    # warm_lane_ref_is_degenerate (passed `branch_name`, the
                    # BARE task id — e.g. '321', NOT the prefixed
                    # 'task/321' — exactly as before) then
                    # _delete_branch_if_on_main, so a commit-bearing branch
                    # is never destroyed.
                    await self._abort_lane_acquisition(
                        lane, branch_name, remove_worktree=True,
                    )
                    return _seed_rc_to_unavailable(seed_rc)
                route = AcquireRoute.CREATE_ONCE_FRESH
            else:
                # ── Already-registered lane — check on-disk backstop first ─
                # If the lane still carries THIS task's plan.json (e.g. after a
                # process restart that cleared the in-memory _assignments map),
                # treat it as a REUSE: restore the assignment and route to
                # _reuse_warm_lane so .task/plan.json + WIP are preserved.
                # Identity guard: if expected_title is set and the stored title
                # does not match, treat as fresh (recycled-id guard).
                # Any read/parse error falls safe toward the fresh reset path.
                #
                # Read new-then-old (W11 ε1): plan.json now lives at the sibling
                # .task-meta/<name> location first (workflow.py's self.artifacts
                # writes new-path-only), falling back to the legacy
                # <lane>/.task/plan.json so lanes seeded before this relocation
                # still resolve — mirrors _find_lane_by_plan_task_id's idiom.
                disk_reuse = False
                try:
                    plan_path = TaskArtifacts.meta_root_for(self.worktree_base, lane.name) / 'plan.json'
                    if not plan_path.exists():
                        plan_path = lane / '.task' / 'plan.json'
                    if plan_path.exists():
                        import json as _json
                        data = _json.loads(plan_path.read_text())
                        if data.get('task_id') == branch_name:
                            # Check identity guard before declaring reuse
                            _disk_ident_ok = (
                                expected_title is None
                                or identities_match(
                                    read_worktree_title(lane), expected_title,
                                )
                            )
                            if _disk_ident_ok:
                                # Record the mapping and route to reuse. Thread
                                # title/branch so the pool's durable write-through
                                # (task 2986, single writer) keeps the record's
                                # task_id/title/branch on the disk-backstop path.
                                self.warm_lane_pool.note_assignment(
                                    branch_name, lane,
                                    title=expected_title, branch=full_branch,
                                )
                                disk_reuse = True
                            else:
                                logger.warning(
                                    'acquire_warm_lane: disk backstop identity '
                                    'MISMATCH for %s — expected %r; fresh reset',
                                    lane, expected_title,
                                )
                except Exception:
                    # Fail safe: unreadable or non-JSON plan → treat fresh
                    pass

                if disk_reuse:
                    route = AcquireRoute.DISK_BACKSTOP_REUSE
                    info = await self._reuse_warm_lane(lane, full_branch)
                    self._note_assigned_via_route(
                        info.path, route, branch_name, expected_title, full_branch,
                    )
                    return info

                # ── γ reattach guard (reset-in-place site) ───────────────────
                # If the orphaned task/<id> branch still carries commits beyond
                # main, reattach to it (checkout -f <branch>, no -B) rather than
                # destroying those commits with a reset.  Route through
                # _reuse_warm_lane (commit-WIP → rebase_onto_main → re-provision)
                # — the same tail as the disk-backstop reuse path above.
                #
                # _orphan_has_commits wraps the rev-parse --verify existence gate
                # + _branch_has_commits_beyond_main (fail-safe True on git error),
                # ensuring brand-new task ids (no branch yet) reach the byte-
                # identical reset-in-place path below, not an erroneous reattach.
                if await self._orphan_has_commits(full_branch):
                    logger.info(
                        'acquire_warm_lane: reattach (reset-in-place site) — '
                        'lane %s has orphan %s with commits; reattaching',
                        lane, full_branch,
                    )
                    route = AcquireRoute.RESET_IN_PLACE_REATTACH
                    _co_rc, _, _co_err = await _run(
                        ['git', 'checkout', '-f', full_branch], cwd=lane,
                    )
                    if _co_rc != 0:
                        # Cannot re-attach (e.g. branch is checked out in another
                        # live worktree after a process restart).  Raise rather than
                        # proceeding: _reuse_warm_lane would commit WIP onto the
                        # wrong (previous-occupant) branch and corrupt state.
                        # Mirrors the create-once site (~1809-1826) and
                        # _cleanup_leftover_branch's raise-not-destroy contract
                        # (inv.10 fail-safe-retain).
                        # acquire_warm_lane's top-level except Exception converts
                        # this to WarmLaneUnavailable.FAULT (lane released, caller
                        # escalates blocked+L1) while leaving full_branch intact.
                        raise RuntimeError(
                            f'acquire_warm_lane: refusing to reuse lane {lane} '
                            f'— orphan {full_branch!r} carries commits beyond '
                            f'{self.config.main_branch} but lane checkout failed '
                            f'(git checkout -f rc={_co_rc}: {_co_err.strip()!r}). '
                            f'Proceeding would commit WIP onto the wrong branch '
                            f'and corrupt state. The branch is left intact. '
                            f'Inspect the branch and, once wanted work is preserved, '
                            f'remove the other worktree and retry.'
                        )
                    # Reaching this guard means .task/ (if present) belongs to
                    # the lane's PREVIOUS occupant — the same-task case already
                    # returned via the disk-backstop reuse above.  checkout -f
                    # replaces tracked files but leaves untracked .task/ intact,
                    # and _reuse_warm_lane deliberately preserves .task/ (its
                    # same-task contract) — clear it here or the incoming task
                    # inherits a foreign plan.json/iterations.jsonl/reviews/
                    # (reify esc-4920-163: _lane-26 4949→4920 contamination).
                    shutil.rmtree(lane / '.task', ignore_errors=True)
                    # The sibling .task-meta/<name> (W11 ε1) belongs to the same
                    # previous occupant but, unlike .task/, lives OUTSIDE the
                    # worktree — checkout -f cannot touch it, so it must be
                    # cleared explicitly too (same rationale as immediately
                    # above; reviewer_comprehensive robustness/data-integrity
                    # blocker at workflow.py:1736).
                    self._clear_foreign_meta_root(lane)
                    info = await self._reuse_warm_lane(lane, full_branch)
                    self._note_assigned_via_route(
                        info.path, route, branch_name, expected_title, full_branch,
                    )
                    return info

                # ── Fresh reset-in-place (new task on a recycled FREE lane) ─
                # R4: one-shot in-process retry on a transient _reset_warm_lane
                # exception (task 1932; relates-to 1859 / 1931).
                recycle_result = await self._reset_and_seed_recycled_lane(
                    lane, full_branch, start_ref,
                )
                if recycle_result is not None:
                    return recycle_result
                route = AcquireRoute.RECYCLE

            # ── Reseed-consistency post-condition (task 2854) ──────────────
            # The two FRESH-reseed routes (RECYCLE / CREATE_ONCE_FRESH) reset
            # or create full_branch at start_ref and are the ONLY routes that
            # reach this shared tail; unlike every REUSE/REATTACH/DISK_BACKSTOP
            # route (which returns early through _reuse_warm_lane and is already
            # protected by the rebase-collapse BranchResetError guard that
            # caught the incident downstream), they have NO post-condition
            # check today. Verify the reseed actually landed clean — HEAD on
            # full_branch, zero commits beyond start_ref — BEFORE handing the
            # lane out for dispatch. A lane still serving a PRIOR occupant's
            # tree (reify incident 2026-07-20: _lane-12 acquired for task 5279
            # while task/5279 sat at task 5264's commits) is faulted here and
            # requeued onto a DIFFERENT lane rather than dispatched onto stale
            # content — closing the gap "before it hits a case the [collapse]
            # guard misses". _reseed_verified_clean is fail-closed, so an
            # unprovable-clean lane is treated as contaminated
            # (loud-over-silent-degradation); never returning a WorktreeInfo
            # here guarantees the lane is left FREE, never ASSIGNED.
            if route in (
                AcquireRoute.RECYCLE, AcquireRoute.CREATE_ONCE_FRESH,
            ) and not await self._reseed_verified_clean(lane, full_branch, start_ref):
                _, offending_head, _ = await _run(
                    ['git', 'rev-parse', 'HEAD'], cwd=lane,
                )
                logger.warning(
                    'acquire_warm_lane: reseed contamination detected — lane %s '
                    'branch %s HEAD %s carries retained prior-occupant commits '
                    'beyond base %s (data-integrity / reseed-consistency defect, '
                    'task 2854); faulting to re-acquire a different lane',
                    lane, full_branch, offending_head.strip() or '?',
                    start_ref[:12] if start_ref else '?',
                )
                await self._abort_lane_acquisition(
                    lane, branch_name, remove_worktree=False,
                )
                return WarmLaneUnavailable.RESEED_CONTAMINATED

            # ── Shared tail: base, debug-port ──────────────────────────────
            _, mb_out, _ = await _run(
                ['git', 'merge-base', start_ref, 'HEAD'],
                cwd=lane,
            )
            _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=lane)
            base_commit = mb_out.strip() or head_sha.strip()

            port = await self._provision_reify_debug_port(lane)
            logger.info(
                'acquire_warm_lane: acquired %s on branch %s (base=%s, port=%s)',
                lane, full_branch, base_commit[:8] if base_commit else '?', port,
            )
            self._note_assigned_via_route(
                lane, route, branch_name, expected_title, full_branch,
            )
            return WorktreeInfo(path=lane, base_commit=base_commit, reify_debug_port=port)

        except BranchResetError:
            # task 2403 (reviewer_comprehensive error_handling finding): a
            # BranchResetError raised by _reuse_warm_lane's guarded rebase
            # (rebase_preserving_task_commits, reached via the disk-backstop
            # reuse route @3975 or the orphan-reattach reuse route @4039)
            # must escape to _drive's targeted branch_reset escalation
            # (workflow.py's isinstance(e, BranchResetError) branch) — NOT
            # be flattened by the broad `except Exception` below into
            # WarmLaneUnavailable.FAULT, which create_worktree maps to a
            # generic RuntimeError that branch_reset routing can never
            # match, mislabeling a wipe as an ordinary warm-lane fault.
            # The guard already attempted (best-effort; see
            # BranchResetError.restore_ok — a failed restore is called out
            # explicitly in str(e), not silently assumed safe) to restore
            # the pre-rebase HEAD before raising, so this is a lost-
            # escalation-SIGNAL fix, not a data-loss fix. Deliberately do
            # NOT call _abort_lane_acquisition here — that would
            # release/reset the lane and discard whatever the guard's
            # restore left in place; fail-safe-retain the (possibly
            # restored) work on the ASSIGNED lane for human recovery,
            # mirroring the raise-not-destroy contract of the
            # orphan-checkout RuntimeError guards above (~4013). Scoped to
            # BranchResetError only — every other exception (seed/
            # worktree-add failure, absent seed script, a WorktreeConflict-
            # Error from commit(), etc.) still flows through the broad
            # handler below unchanged.
            raise
        except Exception:
            logger.warning(
                'acquire_warm_lane: unexpected error for %s; releasing', lane, exc_info=True,
            )
            await self._abort_lane_acquisition(lane, branch_name, remove_worktree=False)
            return WarmLaneUnavailable.FAULT

    def _note_assigned_via_route(
        self, lane: Path, route: AcquireRoute, task_id: str, title: str | None, branch: str,
    ) -> None:
        """Route-classification INFO log for :meth:`_acquire_warm_lane_impl`.

        The durable ASSIGNED record is now written by the :class:`WarmLanePool`
        SINGLE writer (task 2986, PRD warm-lane-exhaustion-hardening W2b I2) at
        the moment the in-memory state flips — ``acquire_for`` (fresh alloc),
        ``reclaim_victim`` (steal) and ``note_assignment`` (disk-backstop) each
        thread ``task_id``/``title``/``branch`` into the pool's durable
        write-through. This method NO LONGER writes or normalizes the durable
        record; it retains ONLY the route-classification INFO log line (PRD W11
        eta Mechanism 3 observability), naming which acquire route the lane took
        and the route's canonical durable edge (read from
        ``ACQUIRE_ROUTE_TRANSITIONS[route]``, keeping the route table
        load-bearing for observability).

        ``task_id``/``title``/``branch`` are retained on the signature (the 5
        call sites pass them) to keep the log line and future observability
        self-describing; they no longer drive any durable write.

        Best-effort / never-raise (mirrors :meth:`mark_pool_storage_present`):
        any exception is logged and swallowed so a logging hiccup never
        regresses ``acquire_warm_lane`` itself.
        """
        try:
            src, dst = ACQUIRE_ROUTE_TRANSITIONS[route]
            logger.info(
                'acquire_warm_lane: route=%s edge=%s->%s lane=%s task=%s',
                route.value,
                src.value if src is not None else 'none',
                dst.value, lane, task_id,
            )
        except Exception:
            logger.warning(
                '_note_assigned_via_route: route-classification log failed for '
                'lane %s (task %s, route %s)',
                lane, task_id, getattr(route, 'value', route), exc_info=True,
            )

    async def _reuse_warm_lane(
        self, lane_dir: Path, full_branch: str,
    ) -> 'WorktreeInfo':
        """Handle a live-requeue: same task on the same lane (in-memory map hit).

        Mirrors the cold-requeue reuse block (git_ops.py ~806-858):
        1. Commit any uncommitted WIP so it is preserved across the rebase.
        2. Rebase onto main (best-effort; log failure and continue on old base).
        4. Recompute ``base_commit`` as ``merge-base main HEAD``.
        5. Re-provision the debug port (inv.7).

        ``.task/plan.json`` is NOT committed by ``commit()`` (task metadata
        lives outside the worktree for the orchestrator hot path; any
        leftover ``.task/`` is covered by this repo's root ``.gitignore``
        ``.task/`` entry) and survives the rebase intact because git rebase
        only touches tracked files.

        Returns:
            WorktreeInfo for the reused lane.  Never raises — exceptions
            propagate to the caller's try/except-release wrapper.
        """
        # 1. Commit WIP (returns None if nothing to commit — that's fine)
        await self.commit(lane_dir, 'chore: save WIP before requeue rebase')

        # 2. Rebase onto main (best-effort; failure leaves the branch on old
        #    base).  rebase_preserving_task_commits (not the bare primitive):
        #    guards against a silent branch-reset wipe (task 2403) — a
        #    BranchResetError raised here propagates to the caller's
        #    try/except-release wrapper (see docstring above) and on to
        #    _drive()'s exception handler.
        rebased = await self.rebase_preserving_task_commits(lane_dir)
        if not rebased:
            logger.info(
                '_reuse_warm_lane: rebase failed for %s; continuing on old base', lane_dir,
            )

        # 2b. Rebind refs/heads/<full_branch> to the lane's current HEAD and
        #     re-attach the lane onto the branch.
        #
        #     Task-1923 residual: release_warm_lane detaches the lane via
        #     `git checkout --detach`.  If the same task is re-dispatched via
        #     the disk-backstop reuse route (route 2) or the in-memory reuse
        #     route (route 1), _reuse_warm_lane runs commit()+rebase_onto_main
        #     on a DETACHED HEAD and never moves refs/heads/<full_branch> to
        #     match.  The stale ref then causes α's retention guard to evaluate
        #     0-commits-beyond-main (incorrect) and delete it, and the merge
        #     worker (resolve_queued_branch_ref) to hit unknown_branch.
        #
        #     best-effort: if the rebind fails (e.g. invalid branch name or
        #     some other git error), log and continue — WIP is still on the
        #     lane's HEAD, which is no worse than today's detached-HEAD state.
        #     NOTE: checkout -B bypasses git's single-checkout guard; a
        #     concurrent worktree on full_branch is NOT a fail-safe — it is a
        #     duplicate-dispatch hazard (both worktrees end up on the branch).
        #     Route 3 (γ reattach) already re-attaches before this call, so
        #     the rebind is a harmless reset-to-self there.
        await self.rebind_branch_to_head(lane_dir, full_branch)

        # 4. Recompute base: merge-base between main_branch and HEAD
        _, mb_out, _ = await _run(
            ['git', 'merge-base', self.config.main_branch, 'HEAD'],
            cwd=lane_dir,
        )
        _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=lane_dir)
        base_commit = mb_out.strip() or head_sha.strip()

        # 5. Re-provision debug port (inv.7)
        port = await self._provision_reify_debug_port(lane_dir)

        logger.info(
            '_reuse_warm_lane: reused %s on branch %s (base=%s, port=%s)',
            lane_dir, full_branch, base_commit[:8] if base_commit else '?', port,
        )
        return WorktreeInfo(path=lane_dir, base_commit=base_commit, reify_debug_port=port)

    async def rebind_branch_to_head(self, worktree: Path, full_branch: str) -> bool:
        """Rebind *full_branch* to the current HEAD of *worktree* and attach the lane.

        Runs ``git checkout -B <full_branch>`` (no explicit start point, so the
        implicit start point is HEAD).  This both:
        * resets ``refs/heads/<full_branch>`` to the worktree's current HEAD, and
        * attaches the worktree onto the branch (no longer detached).

        This mirrors the ``checkout -B`` idiom used by :meth:`_reset_warm_lane`
        (without ``-f``, since _reuse_warm_lane's tree is already
        committed/clean after :meth:`commit`).

        **Best-effort / never-raise** (task-1923 live-requeue residual fix):
        ``git checkout -B`` bypasses git's linked-worktree single-checkout
        guard: if *full_branch* is currently checked out in another live
        worktree, the command still succeeds (rc=0), force-resets the ref to
        the current HEAD, and leaves both worktrees tracking the same branch —
        a duplicate-dispatch hazard, NOT a safe fail-safe.  This method
        returns ``False`` only on a genuine non-zero rc (e.g. an invalid
        branch name); in that case it logs a WARNING and never raises.
        It never converts a reuse into a FAULT.

        Returns:
            True on success (rc=0); False on any non-zero git rc.
        """
        rc, _, err = await _run(
            ['git', 'checkout', '-B', full_branch],
            cwd=worktree,
        )
        if rc != 0:
            logger.warning(
                'rebind_branch_to_head: checkout -B %s failed for %s (rc=%d): %s',
                full_branch, worktree, rc, err.strip(),
            )
            return False
        logger.debug(
            'rebind_branch_to_head: rebound %s to HEAD of %s', full_branch, worktree,
        )
        return True

    async def _clean_lane_retaining_artifacts(
        self, cwd: Path, *, caller: str = 'git_ops',
    ) -> tuple[bool, str]:
        """Run a single ``git clean -xfd`` preserving artifact dirs and reseed-trash siblings.

        Builds one invocation excluding ALL configured artifact dirs (via
        ``self.config.reap_build_artifact_dirs``) and their
        ``<dir>.reseed-trash.*`` siblings — single-pass so >1 artifact dir all
        survive (per-dir-loop step-19 regression).  A benign ENOENT race (R3:
        concurrent detached ``rm -rf`` already removed the path before git clean
        reached it) is treated as success and logged at WARNING level.

        This is the single-source implementation shared by :meth:`_reset_warm_lane`,
        :meth:`acquire_spec_lane`, and :meth:`reset_persistent_merge_worktree`.
        Each call site only differs in how it handles the ``(False, err)`` case
        (raise vs. cold fallback).

        Returns:
            ``(True, '')`` if git clean exited 0.
            ``(True, err)`` if the failure was a benign ENOENT race (logged at WARNING).
            ``(False, err)`` on a genuine clean failure.
        """
        clean_cmd = ['git', 'clean', '-xfd']
        for artifact_dir in self.config.reap_build_artifact_dirs:
            clean_cmd += ['-e', artifact_dir, '-e', f'{artifact_dir}.reseed-trash.*']
        rc, _, err = await _run(clean_cmd, cwd=cwd)
        if rc == 0:
            return True, ''
        if _git_clean_failure_is_benign(err):
            logger.warning(
                '%s: git clean exited %d for %s but only benign ENOENT '
                '"failed to remove" warnings detected (concurrent deleter raced '
                'the clean walk; working tree left clean); treating as success: %s',
                caller, rc, cwd, err,
            )
            return True, err
        return False, err

    def _clear_foreign_meta_root(self, lane: Path) -> None:
        """Remove the sibling ``.task-meta/<name>`` dir for *lane* (W11 ε1).

        ``.task-meta/<name>`` lives OUTSIDE the worktree, so none of
        ``checkout -f -B``, ``git clean``, ``checkout -f``, or the create-once
        worktree rebuild (self-heal ``rmtree`` + ``git worktree add``) touch
        it — a DIFFERENT-task acquisition (RECYCLE / RESET_IN_PLACE_REATTACH /
        CREATE_ONCE_FRESH / CREATE_ONCE_REATTACH) would otherwise hand the
        incoming task the PRIOR occupant's
        plan.json/metadata.json/blocking_dependency.json. Mirrors the
        already-landed interactive-reap cleanup (``reap_interactive_worktrees``).
        Best-effort (``ignore_errors=True``); never raises. Must NOT be called
        on same-task reuse routes (REUSE / DISK_BACKSTOP_REUSE) — those
        legitimately preserve the lane's own artifacts.
        """
        shutil.rmtree(
            TaskArtifacts.meta_root_for(self.worktree_base, lane.name),
            ignore_errors=True,
        )
        logger.debug(
            '_clear_foreign_meta_root: cleared .task-meta/%s for a different-task '
            'acquisition',
            lane.name,
        )

    async def _reset_warm_lane(
        self, lane_dir: Path, full_branch: str, target_commit: str,
    ) -> None:
        """Reset an already-registered lane to *target_commit* on *full_branch*.

        Implements reset-determinism (inv.1) + warmth retention:
        ``git checkout -B <full_branch> <target_commit>`` establishes the new
        task branch and updates the tracked tree; then a SINGLE
        ``git clean -xfd -e <dir>`` (one -e per reap_build_artifact_dirs)
        removes stray untracked files while retaining all artifact dirs —
        mirroring reset_persistent_merge_worktree's single-pass clean so
        >1 artifact dir all survive (per-dir-loop bug from κ, step-19).

        Only ever reached for a DIFFERENT-task acquisition (RECYCLE) — same-
        task reuse routes through ``_reuse_warm_lane`` instead — so once the
        in-worktree reset succeeds, the sibling ``.task-meta/<name>`` (which
        the reset above cannot touch) is cleared too (W11 ε1).

        Added as a stub here (step-8); fully exercised by step-10 tests.
        """
        # -f (force) discards local modifications to tracked files before the
        # branch switch — required when the lane is being reused from a prior
        # task that left uncommitted work.  -B creates or resets the branch.
        rc, _, err = await _run(
            ['git', 'checkout', '-f', '-B', full_branch, target_commit],
            cwd=lane_dir,
        )
        if rc != 0:
            raise RuntimeError(
                f'_reset_warm_lane: checkout -f -B {full_branch} {target_commit} '
                f'failed for {lane_dir}: {err}'
            )
        ok, err = await self._clean_lane_retaining_artifacts(
            lane_dir, caller='_reset_warm_lane',
        )
        if not ok:
            raise RuntimeError(
                f'_reset_warm_lane: git clean failed for {lane_dir}: {err}'
            )
        self._clear_foreign_meta_root(lane_dir)

    async def _reset_and_seed_recycled_lane(
        self,
        lane: Path,
        full_branch: str,
        start_ref: str,
    ) -> 'WarmLaneUnavailable | None':
        """Reset and re-seed a recycled FREE lane with one-shot retry on transient faults.

        R4 defense-in-depth (task 1932; relates-to 1859, 1931).

        R3/1931 already absorbs the one KNOWN transient (benign-ENOENT git-clean
        race vs reify's reseed-trash rm) INSIDE ``_reset_warm_lane``, returning
        success so it never raises.  R4 is the DURABLE SAFETY NET for any OTHER
        failure that still makes ``_reset_warm_lane`` RAISE — e.g. the
        ``checkout -f -B`` failure in ``_reset_warm_lane``, or a
        genuine/empty-stderr git-clean failure in ``_reset_warm_lane``.
        A single immediate retry self-heals transient
        lane-disk-state races; a genuine fault recurs on retry and still
        surfaces as FAULT (preserving 1859's no-silent-degrade contract).

        Only exceptions from ``_reset_warm_lane`` are retried.  Seed rc-failures
        (rc=127/75/other) are genuine/disk discriminants and are never retried
        (``_seed_warm_lane`` never raises; its rc sentinels are already
        discriminated below).

        Returns:
            ``None``
                Reset+seed succeeded; caller falls through to the shared tail.
            ``WarmLaneUnavailable.FAULT`` / ``.DISK_PRESSURE``
                Seed rc-failure or persistent reset fault; lane already released;
                caller must return the sentinel without touching the lane further.
        """
        # Called only after the pool handed us a lane, so it cannot be None.
        assert self.warm_lane_pool is not None
        # One-shot retry on _reset_warm_lane exception.
        # attempt == 1: first try; attempt == 2: retry (scrub target/ first).
        for attempt in (1, 2):
            try:
                await self._reset_warm_lane(lane, full_branch, start_ref)
                break  # reset succeeded — proceed to seed below
            except Exception:
                if attempt == 1:
                    logger.warning(
                        '_reset_and_seed_recycled_lane: transient reset fault '
                        'for %s; scrubbing target/ and retrying reset once (R4)',
                        lane, exc_info=True,
                    )
                    # Pre-retry scrub: defensively remove any partial target/
                    # state that may have contributed to the transient fault.
                    # _reset_warm_lane's checkout -f -B + git clean do not
                    # require target/ to be absent, but this best-effort removal
                    # eliminates stale disk state before the retry, maximising
                    # the chance the retry self-heals.  The unconditional scrub
                    # at the seed site below is redundant on this path but
                    # intentional — it normalises lane state regardless of which
                    # attempt succeeded.
                    shutil.rmtree(lane / 'target', ignore_errors=True)
                    continue  # retry
                # attempt == 2: fault persisted — genuine fault, escalate per 1859
                logger.warning(
                    '_reset_and_seed_recycled_lane: reset fault persisted after '
                    'retry for %s -> FAULT (escalate per 1859)',
                    lane, exc_info=True,
                )
                # _abort_lane_acquisition (task 2199) — detaches HEAD first
                # so a branch left checked out by a partially-applied
                # _reset_warm_lane doesn't leak a collision for the next
                # acquire (task 2062 mid-run leak).
                await self._abort_lane_acquisition(
                    lane, full_branch[len(self.config.branch_prefix):],
                    remove_worktree=False,
                )
                return WarmLaneUnavailable.FAULT

        # Reset succeeded (attempt 1 or 2).  Run the thin re-seed unchanged.
        # β: rm target/ before seeding (no retained bloat).
        shutil.rmtree(lane / 'target', ignore_errors=True)
        rc = await self._seed_warm_lane(lane, '--fresh-checkout')
        if rc != 0:
            if rc == 127:
                logger.warning(
                    'acquire_warm_lane: reset-in-place re-seed — seed script '
                    'absent for lane %s (rc=127) — check seed-warm-lane.sh '
                    'deployment; EVERY task on this host will fault while '
                    'pool is enabled and the script is missing',
                    lane,
                )
            else:
                logger.warning(
                    'acquire_warm_lane: recycle re-seed failed (rc=%d) for '
                    '%s; releasing lane',
                    rc, lane,
                )
            # _abort_lane_acquisition (task 2199) — detaches HEAD first so
            # the branch _reset_warm_lane just checked out here doesn't leak
            # a collision for the next acquire (task 2062 mid-run leak).
            await self._abort_lane_acquisition(
                lane, full_branch[len(self.config.branch_prefix):],
                remove_worktree=False,
            )
            return _seed_rc_to_unavailable(rc)

        return None  # success — caller falls through to shared tail

    async def release_warm_lane(self, lane_dir: Path, branch_name: str) -> None:
        """Release a warm lane back to the FREE pool.

        Steps:
        1. ``git -C <lane> checkout --detach`` — detach HEAD so the just-used
           branch is deletable while the lane directory stays in place.
        2. ``git branch -D task/<branch_name>`` — **on-main only**: deleted
           only when the branch carries no commits beyond main (i.e. is at the
           main tip).  When it carries commits, the branch is RETAINED and a
           log line is emitted; the pool release still proceeds.
        3. ``await self.warm_lane_pool.release(lane_dir)`` — flip ASSIGNED→FREE.
        4. **§9.5 η (task 2442)**: when ``self.config.warm_lane_release_thin``
           is True, invoke :meth:`_run_thin_warm_lane` — an eager free-first
           reclaim of the lane's ``target/`` via reify δ's
           ``scripts/thin-warm-lane.sh``, run strictly AFTER the ASSIGNED→FREE
           flip.  The script holds the lane's own flock (T3), and (task 2599)
           :meth:`_seed_warm_lane` now holds that SAME ``<lane_dir>.lock``
           exclusively for its own subprocess duration, so a concurrent
           re-acquire of this just-freed lane genuinely contends and makes
           this call exit 75 (benign skip) — never thinned while ASSIGNED,
           by construction (inv.10) and now actually enforced DF-side rather
           than delegated to an un-honored cross-repo contract.  Only
           ``target/`` is ever removed (T1); invoked WITHOUT ``--reseed`` — the
           next :meth:`acquire_warm_lane` always re-seeds from the current
           base regardless (D10), so net warmth is unchanged and only the
           idle-hold of a divergent ``target/`` is eliminated.  Best-effort /
           never-raise (:meth:`_run_thin_warm_lane` never raises), so a thin
           hiccup can never strand the pool release or the scheduler
           (inv.11: release-thin is not an escalation/fault).

        Absent the knob (default False), ``target/`` is left in place
        incidentally (CoW-cheap, harmless) — the *next*
        :meth:`acquire_warm_lane` always re-seeds from the current base (D10),
        so a released lane's target/ drift is irrelevant either way.

        Fully best-effort / never-raise (mirrors ``cleanup_merge_worktree``
        contract) so a hiccup cannot strand the scheduler.
        """
        if self.warm_lane_pool is None:
            return

        full_branch = f'{self.config.branch_prefix}{branch_name}'
        try:
            # Detach HEAD so the task branch can be deleted while the lane
            # remains checked out warm (target/ survives).
            rc, _, err = await _run(
                ['git', 'checkout', '--detach'],
                cwd=lane_dir,
            )
            if rc != 0:
                logger.warning(
                    'release_warm_lane: checkout --detach failed for %s: %s', lane_dir, err,
                )
        except Exception:
            logger.warning(
                'release_warm_lane: checkout --detach error for %s', lane_dir, exc_info=True,
            )

        await self._delete_branch_if_on_main(full_branch, context='release_warm_lane')

        # The pool's release() writes the durable RELEASED record (task 2986,
        # single writer): _note_released_durable transitions an ASSIGNED/IN_USE
        # lane -> RELEASED at the moment the in-memory slot flips to FREE, so
        # GitOps no longer issues a separate durable RELEASED write here.
        await self.warm_lane_pool.release(lane_dir)
        logger.info('release_warm_lane: released %s (branch %s)', lane_dir, full_branch)

        # §9.5 η (task 2442): eager free-first release-thin — LAST step, so a
        # thin hiccup can never strand the pool release above (inv.11).
        if self.config.warm_lane_release_thin:
            await self._run_thin_warm_lane(lane_dir)

    async def detach_lane_checkout(self, lane: Path, bare_id: str) -> bool:
        """Commit WIP then detach *lane*'s HEAD, preserving branch and worktree.

        Used by :meth:`Harness._reconcile_lane_checkouts` to free git's
        single-checkout lock on a stale ``task/<bare_id>`` branch without
        losing uncommitted work: :meth:`commit` snapshots WIP onto the
        currently-checked-out branch first (mirrors the reclaim/
        ``_reuse_warm_lane`` commit-before-mutate contract), THEN ``git
        checkout --detach`` frees the branch for a future ``git worktree
        add`` elsewhere.

        Does NOT remove the worktree (would dangle the pool's registered
        lane path) and does NOT delete the branch — the branch is exactly
        the WIP-recovery handle the next acquire's ``_orphan_has_commits`` →
        reattach → ``_reuse_warm_lane`` path depends on.

        Returns:
            True on success.  False (and logs at ERROR) if ``git checkout
            --detach`` fails — the caller must NOT treat the branch as freed.
        """
        # commit() returns None when the tree is already clean — fine.
        await self.commit(lane, 'chore: save WIP before lane-checkout reconcile detach')

        rc, _, err = await _run(['git', 'checkout', '--detach'], cwd=lane)
        if rc != 0:
            logger.error(
                'detach_lane_checkout: checkout --detach failed for %s (task %s): %s',
                lane, bare_id, err,
            )
            return False
        logger.info(
            'detach_lane_checkout: detached %s (task %s) — branch and worktree retained',
            lane, bare_id,
        )
        return True

    async def _abort_lane_acquisition(
        self, lane: Path, bare_id: str, *, remove_worktree: bool,
    ) -> None:
        """Never-raise teardown for any :meth:`acquire_warm_lane` fault exit.

        Task 2199 (task-2062 residual): every fault-exit path in
        ``acquire_warm_lane`` and its recycle/reset helpers routes through
        this single primitive so a partially-acquired lane is always left in
        a consistent state before its pool slot is freed — in particular,
        HEAD must be detached before the lane is released, or a requeue can
        collide with "already used by worktree".

        Never raises. ``self.warm_lane_pool.release(lane)`` is always the
        LAST action, so the slot is never marked FREE while lane git state
        is still being mutated.
        """
        if self.warm_lane_pool is None:
            return

        full_branch = f'{self.config.branch_prefix}{bare_id}'

        # (a) Best-effort WIP snapshot BEFORE any ref movement — mirrors
        # detach_lane_checkout's commit-before-detach contract (never
        # discard uncommitted WIP). Guarded on HEAD actually being on
        # full_branch: a reattach-guard raise (γ, create-once and
        # reset-in-place sites) can leave the lane still checked out on a
        # STALE PREVIOUS OCCUPANT's branch when the reattach itself failed —
        # committing unconditionally would contaminate that foreign branch
        # with WIP that was never its own. Skip (fail toward not touching
        # foreign state) unless HEAD affirmatively resolves to full_branch.
        # Best-effort: the very fault that triggered this abort may itself
        # be a commit() failure (e.g. the _reuse_warm_lane commit()
        # RuntimeError) — never re-raise.
        try:
            _rc_head, _cur_branch, _ = await _run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=lane,
            )
            if _rc_head == 0 and _cur_branch.strip() == full_branch:
                # Gate the commit on real changes existing OUTSIDE target/
                # (review-fix, task 2199). target/ is the reify build/seed
                # output dir and is NOT gitignored inside a lane checkout
                # (the root .gitignore's `.worktrees/` entry does not reach
                # into a lane's own working tree). The create-once and
                # recycle re-seed fault routes reach this step right after a
                # FAILED seed script, which can leave a partially-written
                # target/ behind. There is by definition no genuine user WIP
                # under target/ on those routes, so unconditionally
                # committing here would turn a would-be-degenerate 0-commit
                # ref into a RETAINED, commit-bearing branch full of seed
                # garbage — defeating the degenerate-gated retention below.
                # Fail toward committing (never-discard-WIP) if the status
                # probe itself errors.
                _rc_status, _status_out, _ = await _run(
                    ['git', 'status', '--porcelain', '--', '.', ':!target'],
                    cwd=lane,
                )
                if _rc_status != 0 or _status_out.strip():
                    await self.commit(lane, 'chore: save WIP before lane-acquire abort')
                else:
                    logger.debug(
                        '_abort_lane_acquisition: lane %s has no changes '
                        'outside target/ — skipping WIP commit (avoids '
                        'committing seed residue as a retained garbage '
                        'branch)', lane,
                    )
            else:
                logger.debug(
                    '_abort_lane_acquisition: lane %s HEAD is not on %s '
                    '(got %r) — skipping WIP commit to avoid contaminating '
                    'a foreign branch', lane, full_branch,
                    _cur_branch.strip() if _rc_head == 0 else None,
                )
        except Exception:
            logger.warning(
                '_abort_lane_acquisition: commit error for %s (task %s)',
                lane, bare_id, exc_info=True,
            )

        # (b) Detach HEAD — frees git's single-checkout lock (the task-2062
        # fix: a lane must never be released FREE while task/<bare_id> is
        # still checked out).
        try:
            rc, _, err = await _run(['git', 'checkout', '--detach'], cwd=lane)
            if rc != 0:
                logger.warning(
                    '_abort_lane_acquisition: checkout --detach failed for '
                    '%s (task %s): %s', lane, bare_id, err,
                )
        except Exception:
            logger.warning(
                '_abort_lane_acquisition: checkout --detach error for %s '
                '(task %s)', lane, bare_id, exc_info=True,
            )

        # (c) Degenerate-gated retention — upgrades the task-2112 classifier
        # from create-once-only to every fault route. Both guards are
        # retention-biased (fail-soft False → retain; commit-bearing →
        # retain), so this can never destroy an orphan commit.
        try:
            if await self.warm_lane_ref_is_degenerate(bare_id):
                await self._delete_branch_if_on_main(
                    full_branch, context='acquire_warm_lane abort',
                )
        except Exception:
            logger.warning(
                '_abort_lane_acquisition: degenerate-check/delete error for '
                '%s (task %s)', full_branch, bare_id, exc_info=True,
            )

        if remove_worktree:
            try:
                rc, _, err = await _run(
                    ['git', 'worktree', 'remove', '--force', str(lane)],
                    cwd=self.project_root,
                )
                if rc != 0:
                    logger.warning(
                        '_abort_lane_acquisition: worktree remove --force '
                        'failed for %s (task %s): %s', lane, bare_id, err,
                    )
            except Exception:
                logger.warning(
                    '_abort_lane_acquisition: worktree remove error for %s '
                    '(task %s)', lane, bare_id, exc_info=True,
                )

        await self.warm_lane_pool.release(lane)

    async def release_lane_for_terminal_task(
        self,
        task_id_or_branch: str,
        *,
        allow_disk_backstop: bool = False,
    ) -> bool:
        """Idempotent, never-raise primitive — release the warm lane assigned to a terminal task.

        Shared by all terminal-exit paths (B1/B2/B3 event wiring + A periodic
        reconciler) so that B+A double-fire is a harmless no-op.

        Resolution order:
        1. Strip ``config.branch_prefix`` to get the bare task id.
        2. In-memory: ``pool.assignment_for(bare_id)`` — O(1), covers the
           common live-process path.  This is the ONLY resolution used when
           ``allow_disk_backstop=False`` (the default).  When the in-memory
           lookup returns None the primitive returns ``False`` immediately —
           a true no-op (no disk scan, no redundant ``cleanup_worktree`` /
           ``git branch -D`` retry).
        3. On-disk backstop: ``_find_lane_by_plan_task_id(bare_id)`` — scans
           ``worktree_base/_lane-*/.task/plan.json``; used ONLY when
           ``allow_disk_backstop=True`` (reserved for the lost-map /
           post-restart path: :meth:`_mark_in_progress_done`).  A theft guard
           checks whether the resolved lane has since been re-acquired by a
           different live task; if so, the release is refused (returns False)
           to prevent stealing a live task's lane via a stale plan.json.

        If neither resolves (or the theft guard refuses), returns ``False``
        (no lane to free — already free or not assigned to this task).
        Otherwise routes through :meth:`cleanup_worktree` (pool-aware: calls
        :meth:`release_warm_lane` for warm lanes) and returns ``True``.

        All exceptions are caught and logged; never raises.

        Args:
            task_id_or_branch: Task id or full branch name (prefix stripped).
            allow_disk_backstop: When True, fall back to scanning ``plan.json``
                files on disk if the in-memory assignment map has no entry.
                Defaults to False — callers on B1/B2/B3/A paths must NOT
                pass this; it is reserved for ``_mark_in_progress_done``
                (T9 lost-map / post-restart path).

        Returns:
            ``True`` if a lane was found and freed, ``False`` otherwise.
        """
        if self.warm_lane_pool is None:
            return False

        # Strip the branch prefix so we work with the bare task id throughout
        prefix = self.config.branch_prefix  # e.g. 'task/'
        bare_id = (
            task_id_or_branch[len(prefix):]
            if task_id_or_branch.startswith(prefix)
            else task_id_or_branch
        )

        # Resolve lane via in-memory assignment first
        lane = self.warm_lane_pool.assignment_for(bare_id)

        # Optionally fall back to the on-disk plan.json backstop (opt-in only)
        if lane is None and allow_disk_backstop:
            disk_lane = self._find_lane_by_plan_task_id(bare_id)
            if disk_lane is not None:
                # Theft guard: check if the disk-resolved lane has since been
                # re-acquired by a DIFFERENT live task (stale plan.json window).
                # Use assignments_snapshot() — no lock needed (single event loop).
                snap = self.warm_lane_pool.assignments_snapshot()
                holder = next(
                    (br for br, ln in snap.items() if ln == disk_lane), None
                )
                if holder is not None and holder != bare_id:
                    logger.warning(
                        'release_lane_for_terminal_task: theft guard refused — '
                        'disk-resolved lane %s for task %r is now held by %r',
                        disk_lane, bare_id, holder,
                    )
                    disk_lane = None
            lane = disk_lane

        if lane is None:
            logger.debug(
                'release_lane_for_terminal_task: no lane found for %r — already free or unknown',
                bare_id,
            )
            return False

        try:
            await self.cleanup_worktree(lane, bare_id)
            logger.info(
                'release_lane_for_terminal_task: released lane %s for task %r',
                lane, bare_id,
            )
            return True
        except Exception:
            logger.warning(
                'release_lane_for_terminal_task: cleanup error for %s task %r',
                lane, bare_id, exc_info=True,
            )
            return False

    def _find_lane_by_plan_task_id(self, task_id: str) -> Path | None:
        """Scan each pool lane's plan.json (new-then-old) for *task_id*.

        On-disk backstop for :meth:`release_lane_for_terminal_task`: used when
        ``pool.assignment_for`` returns None (e.g. post-restart where the
        in-memory assignment map was not rebuilt for a terminal task).

        Reads plan.json new-then-old (W11 gamma ``.task`` -> ``.task-meta``
        relocation, PRD decision 7 — side-effect-free reads, no migration
        write-back here): first ``TaskArtifacts.meta_root_for(worktree_base,
        entry.name) / 'plan.json'`` (the new location, a sibling of the lane
        dir), falling back to the legacy ``entry / '.task' / 'plan.json'`` so
        lanes seeded before this relocation still resolve.

        Uses a local ``import json as _json`` (mirrors git_ops.py:1816 idiom —
        no module-level ``import json``).  Hoisted once above the loop rather
        than repeated per entry (import is cached after first call, but
        re-executing the statement inside the loop is needlessly noisy).
        Catches ``(ValueError, OSError)`` per corrupt/missing plan; never raises.

        Returns the lane ``Path`` whose ``plan.json`` carries
        ``str(plan.get('task_id')) == task_id``, or ``None`` if not found.
        """
        pool = self.warm_lane_pool
        if pool is None:
            return None
        base = self.worktree_base
        try:
            if not base.exists():
                return None
            entries = list(base.iterdir())
        except OSError:
            return None
        import json as _json  # local-import idiom; hoisted above loop
        for entry in entries:
            if not entry.is_dir():
                continue
            if not pool.is_lane(entry):
                continue
            try:
                plan_path = TaskArtifacts.meta_root_for(base, entry.name) / 'plan.json'
                if not plan_path.exists():
                    plan_path = entry / '.task' / 'plan.json'
                    if not plan_path.exists():
                        continue
                data = _json.loads(plan_path.read_text())
                if str(data.get('task_id')) == task_id:
                    return entry
            except (ValueError, OSError):
                continue
        return None

    async def _provision_reify_debug_port(self, worktree_path: Path) -> int | None:
        """Run setup-worktree-debug-port.sh in the worktree and return the allocated port.

        Best-effort and fail-open: returns None on any miss or failure so
        worktree creation is never blocked by a debug-port hiccup.

        **Idempotency contract**: This helper is invoked on *both* the
        fresh-create and reuse/requeue return paths of ``create_worktree``.
        On reuse the script is re-run to re-acquire a free port and re-patch
        ``<worktree>/.mcp.json``.  The script (``scripts/setup-worktree-
        debug-port.sh`` in the provisioned worktree) is therefore expected to
        be idempotent with respect to the worktree directory — successive calls
        for the same worktree must return the same port rather than allocating
        a new one each time.  If the script is not idempotent it may churn
        (leak) ports across requeues; the existence guard and ``try/except``
        wrapper below ensure this function itself is always safe to call, but
        port stability is the script's responsibility.
        """
        try:
            script = worktree_path / 'scripts' / 'setup-worktree-debug-port.sh'
            if not script.exists():
                return None
            rc, out, err = await _run([str(script), str(worktree_path)], cwd=worktree_path)
            if rc != 0:
                logger.warning(
                    '_provision_reify_debug_port: script exited %d for %s (stderr=%r)',
                    rc, worktree_path, err,
                )
                return None
            lines = [line for line in out.splitlines() if line.strip()]
            return int(lines[-1])
        except (ValueError, IndexError):
            logger.warning(
                '_provision_reify_debug_port: could not parse port from stdout for %s',
                worktree_path,
            )
            return None
        except Exception:
            logger.warning(
                '_provision_reify_debug_port: unexpected error for %s',
                worktree_path, exc_info=True,
            )
            return None

    async def _worktree_holding_branch(self, full_branch: str) -> Path | None:
        """Path of the registered worktree that has *full_branch* checked out.

        Returns ``None`` when no worktree holds it (a dangling ref) or when
        ``git worktree list`` errors.  Callers treat ``None`` conservatively —
        a dangling ref has no working tree to be dirty, so only commits-beyond-
        main can carry work, and that is checked separately and fail-safe.
        """
        rc, out, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=self.project_root,
        )
        if rc != 0:
            return None
        target = f'refs/heads/{full_branch}'
        current: Path | None = None
        for line in out.splitlines():
            if line.startswith('worktree '):
                current = Path(line[len('worktree '):].strip())
            elif line.startswith('branch ') and line[len('branch '):].strip() == target:
                return current
        return None

    async def lane_branch_checkouts(self) -> dict[str, Path] | None:
        """Map every pool-lane branch checkout to its bare task id.

        Generalises :meth:`_worktree_holding_branch`'s porcelain parse:
        instead of resolving one target branch, it walks every ``worktree``/
        ``branch`` pair in ``git worktree list --porcelain`` and keeps those
        whose branch is namespaced under ``config.branch_prefix`` AND whose
        worktree path resolves to a registered :class:`WarmLanePool` lane
        (via :meth:`WarmLanePool._match_lane`).  Non-task branches, detached
        entries, and non-pool worktrees are filtered out.

        Returns:
            ``{bare_id: canonical_lane}`` (bare_id has ``branch_prefix``
            stripped; canonical_lane is the pool's registered lane Path, not
            necessarily identical to the porcelain path e.g. under symlinks).
            ``None`` when the warm-lane pool is disabled, or when ``git
            worktree list`` errors — callers must treat both as a no-op and
            never mass-mutate on an unreliable read.
        """
        pool = self.warm_lane_pool
        if pool is None:
            return None

        rc, out, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=self.project_root,
        )
        if rc != 0:
            return None

        head_prefix = f'refs/heads/{self.config.branch_prefix}'
        result: dict[str, Path] = {}
        current: Path | None = None
        for line in out.splitlines():
            if line.startswith('worktree '):
                current = Path(line[len('worktree '):].strip())
                continue
            if not line.startswith('branch ') or current is None:
                continue
            name = line[len('branch '):].strip()
            if not name.startswith(head_prefix):
                continue
            matched = pool._match_lane(current)
            if matched is not None:
                result[name[len(head_prefix):]] = matched
        return result

    async def _branch_has_commits_beyond_main(self, full_branch: str) -> bool:
        """Whether *full_branch* carries commits beyond main.

        **Fail-safe ``True``** on any git error or unparseable output — never
        report a branch as empty (safe to delete) when we cannot prove it.
        """
        rc, out, _ = await _run(
            ['git', 'rev-list', '--count',
             f'{self.config.main_branch}..{full_branch}'],
            cwd=self.project_root,
        )
        if rc != 0:
            return True
        try:
            return int(out.strip()) > 0
        except ValueError:
            return True

    async def _orphan_has_commits(self, full_branch: str) -> bool:
        """Whether *full_branch* exists AND carries commits beyond main.

        Combines an explicit ``git rev-parse --verify`` existence gate with
        :meth:`_branch_has_commits_beyond_main`.  The existence gate is required
        because ``_branch_has_commits_beyond_main`` fail-safe-returns ``True``
        for a nonexistent branch (its ``rev-list`` errors, rc != 0 → True);
        using the probe alone would wrongly fire the reattach guard for brand-new
        task ids.

        Returns ``False`` when the branch does not exist — the fresh-create or
        reset-in-place path is correct.  Returns ``True`` when the branch exists
        AND the commits probe confirms work beyond main (including fail-safe
        ``True`` on git error — retain direction).

        Used by both γ reattach sites in :meth:`acquire_warm_lane` to avoid
        duplicating the two-step existence-then-probe gate.
        """
        rp_rc, _, _ = await _run(
            ['git', 'rev-parse', '--verify', '--quiet', full_branch],
            cwd=self.project_root,
        )
        if rp_rc != 0:
            return False  # branch does not exist — take the fresh path
        return await self._branch_has_commits_beyond_main(full_branch)

    async def _reseed_verified_clean(
        self, lane: Path, full_branch: str, base_ref: str,
    ) -> bool:
        """Whether a fresh reseed of *lane* landed clean (task 2854).

        The reseed-consistency contract for the fresh-reseed acquire routes
        (:attr:`AcquireRoute.RECYCLE` / :attr:`AcquireRoute.CREATE_ONCE_FRESH`)
        is: the lane's checked-out branch is *full_branch*, reset to
        *base_ref*, carrying NO retained prior-occupant commits. This
        predicate asserts exactly that post-condition — the symmetric
        counterpart to the rebase-collapse :class:`BranchResetError` guard
        that already protects the reuse/reattach routes — so a lane still
        serving a PRIOR task's tree (reify incident 2026-07-20: ``_lane-12``
        acquired for task 5279 while its ``task/5279`` branch sat at task
        5264's commits) is faulted BEFORE dispatch, rather than relied on the
        downstream collapse guard to catch it late.

        Returns ``True`` iff BOTH hold, measured in the lane against its live
        HEAD (what actually gets dispatched):

        1. ``git rev-parse --abbrev-ref HEAD`` == *full_branch* — the reseed
           actually switched the checkout to the incoming task's branch (a
           detached or foreign-branch HEAD is not a verified clean reseed).
        2. ``git rev-list --count <base_ref>..HEAD`` parses to ``0`` — HEAD
           carries zero commits beyond the base the lane was reset to.

        **Fail-closed** (the opposite direction to
        :meth:`_branch_has_commits_beyond_main`'s fail-safe ``True``): any
        non-zero git rc, a detached/other-branch HEAD, or unparseable output
        returns ``False`` — if we cannot PROVE the lane is a clean reseed we
        must not dispatch a task onto it (a false "contaminated" costs one
        cheap requeue onto a different lane; a false "clean" reopens the
        data-integrity defect this closes). Checks against *base_ref* (the
        base the lane was reset to), not a hardcoded main, so it stays
        contract-accurate even if *base_ref* ever differs from the main ref.
        """
        # (1) HEAD must be on full_branch — the reseed must have switched the
        # checkout, not left a detached or foreign (prior-occupant) branch.
        rc, cur_branch, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=lane,
        )
        if rc != 0 or cur_branch.strip() != full_branch:
            return False
        # (2) HEAD must carry zero commits beyond base_ref (reusing the
        # rev-list --count idiom from _branch_has_commits_beyond_main, but
        # base-parameterized on start_ref and run in the lane against HEAD,
        # fail-closed instead of fail-safe-True).
        rc, out, _ = await _run(
            ['git', 'rev-list', '--count', f'{base_ref}..HEAD'], cwd=lane,
        )
        if rc != 0:
            return False
        try:
            return int(out.strip()) == 0
        except ValueError:
            return False

    async def _delete_branch_if_on_main(
        self, full_branch: str, *, context: str,
    ) -> None:
        """Delete *full_branch* only when it carries no commits beyond main.

        Guard: calls :meth:`_branch_has_commits_beyond_main`; if it returns
        ``True`` (branch has WIP) or raises (fail-safe ``True``), the branch
        is *retained* and a retain message is logged at INFO.  The delete is
        attempted only when the branch is provably at the main tip (0 commits
        beyond main).

        Best-effort / never-raise: all git errors are logged at WARNING and
        the method returns normally regardless.

        Args:
            full_branch: Fully-qualified branch name (e.g. ``task/123``).
            context: Short identifier used in log messages (e.g.
                ``"release_warm_lane"`` or ``"cleanup_worktree"``).
        """
        try:
            if await self._branch_has_commits_beyond_main(full_branch):
                logger.info(
                    '%s: retaining branch %s — carries commits beyond %s',
                    context, full_branch, self.config.main_branch,
                )
            else:
                rc, _, err = await _run(
                    ['git', 'branch', '-D', full_branch],
                    cwd=self.project_root,
                )
                if rc != 0:
                    logger.warning(
                        '%s: branch -D %s failed: %s', context, full_branch, err,
                    )
        except Exception:
            logger.warning(
                '%s: branch -D error for %s', context, full_branch, exc_info=True,
            )

    async def _cleanup_leftover_branch(
        self, full_branch: str, branch_name: str,
    ) -> None:
        """Remove a leftover branch ref ONLY when provably non-destructive.

        Called by ``create_worktree`` when ``full_branch`` already exists.
        Raises :class:`RuntimeError` (→ the task blocks with an L1, now
        non-stranding via Harness Fix #1a) rather than deleting anything when
        the leftover carries commits beyond main, has a dirty working tree, or
        its state cannot be verified.  Hard rule: never destroy WIP / orphan
        commits — escalate when not deterministically certain.
        """
        holding = await self._worktree_holding_branch(full_branch)
        if holding is not None and holding.exists():
            # Branch is checked out in a live tree — full check (commits beyond
            # main OR dirty working tree, fail-safe True on any error).
            unsafe = await self.worktree_has_unsaved_work(holding, branch_name)
        else:
            # Dangling ref, or a worktree admin entry whose directory is gone
            # (e.g. rmtree'd above but still tracked by git — the 3576 shape):
            # no live working tree to be dirty, so only commits-beyond-main
            # can carry work.
            unsafe = await self._branch_has_commits_beyond_main(full_branch)

        if unsafe:
            raise RuntimeError(
                f'create_worktree: refusing to delete leftover branch '
                f'{full_branch!r} — it carries commits beyond '
                f'{self.config.main_branch}, has uncommitted changes, or its '
                f'state could not be verified (fail-safe). This would destroy '
                f'work. Inspect it and, once any wanted work is preserved, '
                f'remove it manually: '
                f'`git worktree remove --force <path>` (if checked out) then '
                f'`git branch -D {full_branch}`.'
            )

        # Provably empty AND clean → safe to remove.  Clear any worktree (and
        # its admin entry) holding the branch first, else `git branch -D` fails
        # with "branch is checked out" (the silent-failure that caused 3576).
        if holding is not None:
            rc_rm, _, err_rm = await _run(
                ['git', 'worktree', 'remove', '--force', str(holding)],
                cwd=self.project_root,
            )
            if rc_rm != 0:
                logger.warning(
                    'create_worktree: `git worktree remove` for leftover %s '
                    'failed (rc=%d): %s — pruning admin entries and retrying '
                    'branch delete', holding, rc_rm, err_rm.strip(),
                )
            await self._prune_registrations(context='create_worktree-leftover')

        rc_del, _, err_del = await _run(
            ['git', 'branch', '-D', full_branch], cwd=self.project_root,
        )
        if rc_del != 0:
            raise RuntimeError(
                f'create_worktree: failed to delete provably-empty leftover '
                f'branch {full_branch!r} (rc={rc_del}): {err_del.strip()}. It '
                f'may still be checked out in a worktree; remove that worktree '
                f'first (`git worktree list` to find it).'
            )

        # Re-verify the ref is actually gone before `git worktree add` collides.
        rc_chk, _, _ = await _run(
            ['git', 'rev-parse', '--verify', full_branch], cwd=self.project_root,
        )
        if rc_chk == 0:
            raise RuntimeError(
                f'create_worktree: leftover branch {full_branch!r} still '
                f'present after `git branch -D`; aborting rather than colliding '
                f'on `git worktree add`.'
            )
        logger.info(
            'create_worktree: removed provably-empty leftover branch %s '
            '(no commits beyond main, clean/no working tree)', full_branch,
        )

    async def commit(self, worktree: Path, message: str) -> str | None:
        """Stage all changes and commit. Returns sha or None if nothing to commit.

        .task/ execution metadata now lives outside the worktree entirely
        for the orchestrator hot path (see module docstring /
        TaskArtifacts.meta_root_for), so no pathspec exclusion or
        post-staging unstage is needed here for that path: `git add -A`
        has nothing under .task/ to stage. A few callers still construct
        TaskArtifacts with meta_root=None (e.g. cli.py's eval flow) and do
        write under <worktree>/.task; that residual directory is covered
        by this repo's root .gitignore `.task/` entry, so `git add -A`
        never stages anything under it either way.

        Pre-staging conflict guard (esc-2128-8): if *worktree* has any
        unresolved (unmerged-index) paths — e.g. a stash-pop that conflicted
        just before this call — raise :class:`WorktreeConflictError` instead
        of staging/committing.  This is checked BEFORE `git add -A` so a
        conflicted tree (which may contain literal conflict markers) is
        never snapshotted.  All WIP-save call sites funnel through this
        method, so this single guard covers every one of them.
        """
        conflicted = await self._detect_unmerged_paths(worktree)
        if conflicted:
            raise WorktreeConflictError(worktree, conflicted)

        await _run(['git', 'add', '-A', '--', '.', ':!.claude'], cwd=worktree)

        # Check for changes
        rc, _, _ = await _run(['git', 'diff', '--cached', '--quiet'], cwd=worktree)
        if rc == 0:
            return None  # nothing staged

        rc, out, err = await _run(['git', 'commit', '-m', message], cwd=worktree)
        if rc != 0:
            raise RuntimeError(f'Commit failed: {err}')

        # Get sha
        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        return sha

    async def get_diff_from_main(self, worktree: Path) -> str:
        """Get diff of worktree branch vs main (dynamic — may be empty if main moved)."""
        _, diff, _ = await _run(
            ['git', 'diff', f'{self.config.main_branch}...HEAD'],
            cwd=worktree,
        )
        return diff

    async def get_diff_from_base(self, worktree: Path, base_commit: str) -> str:
        """Get diff of worktree HEAD vs a fixed base commit.

        Use this instead of get_diff_from_main when main may have advanced
        since the worktree was created (e.g. during review stage).
        """
        _, diff, _ = await _run(
            ['git', 'diff', f'{base_commit}...HEAD'],
            cwd=worktree,
        )
        return diff

    async def get_changed_line_ranges(
        self, ref: str,
    ) -> dict[str, list[tuple[int, int]]]:
        """Return old-side (BASE/main) changed line ranges for *ref* vs main.

        Runs ``git diff {main}...{ref} --unified=0 --no-color`` in
        ``self.project_root`` and delegates parsing to
        :func:`parse_diff_line_ranges`.  Using ``--unified=0`` gives exact
        hunk boundaries with no context padding, so the old-side ranges are
        the minimal set of lines actually modified.  The ``main...{ref}``
        three-dot syntax diffs *ref* against the merge-base of main and ref,
        so both tasks diffed against the same main share BASE coordinates that
        are directly comparable for stackability.

        Returns an empty dict when the diff is empty (no changes vs main).
        """
        _, diff, _ = await _run(
            ['git', 'diff', f'{self.config.main_branch}...{ref}',
             '--unified=0', '--no-color'],
            cwd=self.project_root,
        )
        return parse_diff_line_ranges(diff)

    async def get_new_side_changed_line_ranges(
        self, worktree: Path, from_sha: str, to_sha: str = 'HEAD',
    ) -> dict[str, list[tuple[int, int]]]:
        """Return new-side (HEAD) changed line ranges for ``{from_sha}..{to_sha}``.

        New-side counterpart of :meth:`get_changed_line_ranges`.  Runs
        ``git diff {from_sha}..{to_sha} --unified=0 --no-color`` in *worktree*
        (NOT ``self.project_root`` — the amendment SHA and HEAD live in the
        task worktree) and delegates parsing to
        :func:`parse_diff_added_line_ranges`.  ``--unified=0`` gives exact hunk
        boundaries with no context padding, so the new-side ranges are the
        minimal set of lines the amendment actually touched.

        Used to scope a post-amendment review to the amendment delta: reviewer
        ``location`` line numbers are new-side (HEAD-relative), so they are
        directly comparable to these ranges.

        Returns an empty dict when the diff is empty (no changes in the range).
        """
        _, diff, _ = await _run(
            ['git', 'diff', f'{from_sha}..{to_sha}',
             '--unified=0', '--no-color'],
            cwd=worktree,
        )
        return parse_diff_added_line_ranges(diff)

    async def get_current_branch(self, worktree: Path) -> str:
        """Get the current branch name in a worktree."""
        _, branch, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=worktree,
        )
        return branch

    async def get_main_sha(self) -> str:
        """Return current main branch SHA."""
        _, sha, _ = await _run(
            ['git', 'rev-parse', self.config.main_branch],
            cwd=self.project_root,
        )
        return sha.strip()

    async def get_head_tree_hash(self, worktree: Path) -> str | None:
        """Return HEAD's committed tree hash in *worktree*, or ``None``.

        Fail-safe (mirrors ``get_main_sha``'s shape but NEVER raises): the
        committed tree hash keys the review verdict cache, an optimization/
        churn guard that must degrade to always-review rather than crash the
        workflow loop.  Returns the stripped ``git rev-parse HEAD^{tree}``
        output on success; on any non-zero exit code, empty output, or
        subprocess error (e.g. a vanished/non-git worktree), logs a warning
        and returns ``None``.
        """
        try:
            rc, stdout, stderr = await _run(
                ['git', 'rev-parse', 'HEAD^{tree}'],
                cwd=worktree,
            )
        except Exception as exc:
            logger.warning(
                'get_head_tree_hash: git failed in %s: %s', worktree, exc
            )
            return None
        tree_hash = stdout.strip()
        if rc == 0 and tree_hash:
            return tree_hash
        logger.warning(
            'get_head_tree_hash: unexpected rc=%s in %s: %s',
            rc, worktree, stderr.strip(),
        )
        return None

    async def resolve_branch_sha(self, branch_name: str) -> str | None:
        """Resolve a branch name to its 40-char commit SHA via ``git rev-parse --verify``.

        Uses ``refs/heads/{branch_name}`` to constrain resolution to local
        branches, preventing ambiguous resolution against tags or remote refs
        that happen to share the same name.

        Returns the SHA on success, or None when the ref does not exist or
        cannot be resolved (e.g. branch deleted post-merge, malformed name).
        """
        rc, sha, _ = await _run(
            ['git', 'rev-parse', '--verify', f'refs/heads/{branch_name}'],
            cwd=self.project_root,
        )
        return sha if rc == 0 else None

    async def resolve_queued_branch_ref(self, branch: str) -> str | None:
        """Resolve a caller-submitted *branch* to its canonical full ref, or None.

        Accepts three input shapes and resolves them deterministically:

        1. **Bare task id** (e.g. ``'4778'``) — the common case.  The prefixed
           form ``{branch_prefix}{branch}`` (``'task/4778'``) is tried first;
           if it resolves, it is returned.  This preserves the bare-id contract
           and wins the tie-break when both forms happen to exist.

        2. **Already-prefixed** (e.g. ``'task/4778'``) — submitted by automated
           callers (auto-unblock, steward) that supply the full branch name.
           Rule 1 tries ``'task/task/4778'`` (absent), then rule 2 tries
           ``'task/4778'`` directly and returns it.

        3. **Full non-task name** (e.g. ``'cost-min-prd'``, ``'feature/x'``) —
           rule 1 tries ``'task/cost-min-prd'`` (absent), rule 2 tries
           ``'cost-min-prd'`` and returns it.

        4. **Neither form resolves** → returns ``None`` (genuine misroute;
           caller emits ``unknown_branch``).

        **Tie-break**: when both ``{branch_prefix}{branch}`` and ``branch`` exist
        as live refs, rule 1 wins and the prefixed form is returned.  The
        orchestrator-owned ``task/*`` namespace is authoritative.

        Built on :meth:`resolve_branch_sha` which constrains resolution to
        ``refs/heads/*``, preventing ambiguity against tags or remote refs.
        """
        prefixed = f'{self.config.branch_prefix}{branch}'
        if await self.resolve_branch_sha(prefixed) is not None:
            return prefixed
        if await self.resolve_branch_sha(branch) is not None:
            return branch
        return None

    async def find_merge_marker(
        self, branch: str, *, gate_on_existing_ref: bool = True,
    ) -> str | None:
        """Search main's history for a merge commit whose subject matches
        ``Merge {branch} into {main_branch}``.

        This is the companion check to ``is_ancestor`` for the case where the
        branch ref has already been deleted (e.g., ``cleanup_worktree`` ran
        after ``advance_main`` but before ``set_task_status('done')``).

        **Branch-existence gate**: by default calls ``resolve_branch_sha(branch)``
        first.  If it returns non-None the branch still exists, so this method
        returns None immediately — the caller should rely on ``is_ancestor``
        instead.  This prevents finding a stale merge marker from a *previous*
        run of a re-opened task that shared the same branch name.

        **gate_on_existing_ref=False**: skip that gate and grep main
        unconditionally.  Used by ``classify_and_merge``'s already-merged
        false-positive guard (task 5026), where the branch ref legitimately
        still exists (its tip may have diverged post-merge, the task-1917
        ``honors_snapshot_tip`` shape) yet a positive merge-subject marker on
        main is exactly the evidence that the work landed.  This branch-keyed
        marker is robust when ``task_id != branch`` (the merge subject is keyed
        off the branch, not the task id).

        **Subject pattern**: the exact output of ``_merge_subject(branch,
        self.config.main_branch)`` matched with ``--fixed-strings`` (literal
        match — no BRE metacharacter interpretation, so branch names like
        ``task/v1.0`` are safe).  Because ``_merge_subject`` is also called
        by ``merge_to_main`` and the retry path in ``advance_main``, writer
        and reader share the same derivation and can never silently drift
        apart.  Substring-safety is preserved: ``'Merge task/1 into main'``
        cannot appear inside ``'Merge task/10 into main'`` because the ``0``
        after ``task/1`` falls where the pattern has a space.

        Args:
            branch: Full prefixed branch name, e.g. ``'task/123'``.
                    Same convention as ``is_ancestor`` and ``resolve_branch_sha``.

        Returns:
            The 40-char merge commit SHA on success, or None when the branch
            still exists, the branch never existed, or no matching marker was
            found on main.
        """
        # Gate: if the branch ref still exists, caller should use is_ancestor
        # (unless the caller opted out — see gate_on_existing_ref).
        if gate_on_existing_ref and await self.resolve_branch_sha(branch) is not None:
            return None

        # Branch is gone — search main for a merge commit with the expected subject.
        # Pattern derivation shared with merge_to_main — see docstring for substring-safety argument.
        grep_pattern = _merge_subject(branch, self.config.main_branch)
        rc, out, _ = await _run(
            [
                'git', 'log', self.config.main_branch,
                '--fixed-strings',
                f'--grep={grep_pattern}',
                '--max-count=1',
                '--format=%H',
            ],
            cwd=self.project_root,
        )
        if rc != 0 or not out:
            return None
        return out

    async def find_task_citation_commit(
        self, tid: str, *, pattern_template: str | None = None,
    ) -> str | None:
        """Search main's history for a commit whose SUBJECT cites *tid*.

        Used by the reconciler to gate the ``is_ancestor==True`` fast-path:
        ``is_ancestor`` returns True trivially for zero-commit branches
        whose tip equals the main HEAD at branch-create time, which
        false-positives blocked/escalated tasks.  Requiring a positive
        citation on main rejects that degenerate case.

        Matching is constrained to each candidate commit's SUBJECT line
        only (task 2675 FIX 2): git's ``--grep`` applies ``^``/``$`` per
        LINE across the whole commit message, so a BODY line that merely
        happens to start with a conventional-commit token, or with
        ``Merge task/{tid} into ``, would otherwise false-cite.  ``--grep``
        is used only as a coarse, uncapped full-message PRE-filter (a
        sound superset — any subject match is necessarily a message
        match); each candidate's subject is then re-tested against the
        same pattern, compiled as a Python ``re`` (see
        ``DEFAULT_COMMIT_CITATION_PATTERN``'s doc comment). Candidates are
        walked in git-log order (most-recent-first) and the first whose
        SUBJECT matches wins, so a body-only false match on a newer
        commit can never shadow an older genuine subject citation.

        **Accepted tradeoff — uncapped walk**: dropping ``--max-count=1``
        means git enumerates every commit the coarse ``--grep``
        pre-filter matches (subject OR body), not just the most recent
        one — a task referenced in many commit bodies makes git emit and
        this method re-test all of them.  This is bounded by how many
        commits actually mention the pattern (never the full history) and
        is required for correctness (see above), so it is accepted as-is;
        a caller on a hot path with a very chatty history could additionally
        cap the walk (e.g. an extra ``--max-count=N`` under the assumption
        a genuine subject citation appears within the N most recent
        message-matches), but no such cap is applied here.

        Args:
            tid: Bare task id (no ``task/`` prefix); the prefix is added
                by the default pattern where appropriate.
            pattern_template: Optional override for the citation pattern.
                Defaults to ``DEFAULT_COMMIT_CITATION_PATTERN``.  Empty
                string disables the check by returning None immediately
                (caller opt-out for projects without citation
                conventions).  Must be valid as both a git
                ``--extended-regexp`` pattern and a Python ``re`` pattern
                — an uncompilable override is treated as fail-safe
                no-citation (logs a warning and returns None), mirroring
                the prior git-error-means-None behavior.

        Returns:
            The 40-char commit SHA of the most recent commit on main
            whose SUBJECT cites *tid*, or None when no commit's subject
            cites the task, the pattern is disabled, or an override
            pattern fails to compile as a Python ``re``.
        """
        template = (
            pattern_template
            if pattern_template is not None
            else DEFAULT_COMMIT_CITATION_PATTERN
        )
        if template == '':
            return None
        pattern_str = template.format(tid=re.escape(tid))
        try:
            compiled = re.compile(pattern_str)
        except re.error:
            logger.warning(
                'find_task_citation_commit: pattern_template is not a '
                'valid Python re pattern (tid=%s); treating as '
                'no-citation (fail-safe): %r',
                tid, pattern_str,
            )
            return None
        rc, out, _ = await _run(
            [
                'git', 'log', self.config.main_branch,
                '--extended-regexp',
                f'--grep={pattern_str}',
                '-z',
                '--format=%H%x1f%s',
            ],
            cwd=self.project_root,
        )
        if rc != 0 or not out:
            return None
        for record in out.split('\0'):
            if not record:
                continue
            sha, _, subject = record.partition('\x1f')
            if compiled.search(subject):
                return sha.strip()
        return None

    async def rebase_onto_main(self, worktree: Path, onto: str | None = None) -> bool:
        """Rebase the task branch in *worktree* onto *onto* (default: main).

        When *onto* is None (the default), rebases onto the configured
        ``main_branch`` — identical to the original behaviour, keeping all
        existing callers byte-compatible.

        When *onto* is provided (e.g. a sibling branch like ``task/123``),
        rebases the branch in *worktree* onto that ref instead.  This is used
        by ``stack_train_branches`` to chain members into a linear stack.

        Returns True on success.  On failure, aborts the rebase so the
        worktree is left in a clean state, and returns False.

        Caller must NOT hold ``_merge_lock`` — this is designed to run
        outside the lock so multiple tasks can rebase concurrently in
        their own worktrees.
        """
        target = onto if onto is not None else self.config.main_branch
        rc, _, err = await _run(
            ['git', 'rebase', target],
            cwd=worktree,
        )
        if rc != 0:
            await _run(['git', 'rebase', '--abort'], cwd=worktree)
            logger.info(f'Pre-merge rebase failed in {worktree}: {err}')
            return False
        return True

    async def rebase_preserving_task_commits(
        self, worktree: Path, onto: str | None = None,
    ) -> bool:
        """Rebase *worktree* via :meth:`rebase_onto_main`, guarding against a
        silent branch-reset (RCA: task 2403).

        Wraps the plain ``rebase_onto_main`` primitive with a
        mechanism-independent POST-CONDITION check: capture how many commits
        the branch carries beyond a *baseline* ref BEFORE the rebase
        (*n_before*, via ``rev-list --count <baseline>..HEAD``) and again
        AFTER (*n_after*). If the rebase reports success but the branch had
        committed work that vanished (``n_before > 0`` and ``n_after == 0``)
        and that isn't just a patch-id dedup of work already applied at the
        baseline (see below), something collapsed the branch onto the
        baseline's (or an unrelated) tip and silently destroyed that work —
        restore the pre-rebase HEAD and raise :class:`BranchResetError`
        instead of returning success.

        The baseline is ``onto`` when the caller passes one — resolved to a
        concrete sha *before* the rebase runs, so a moving ref can't skew
        the pre/post comparison — else ``main``. This matters for stacked
        train members (``onto=<predecessor sha>``): measuring against main
        unconditionally would under-count there, since the predecessor's own
        commits are already ahead of main *before* this branch's commits are
        added — a wipe of just this branch's own delta would still leave
        ``n_after`` (over main) > 0 and the guard would never fire.

        This is the difference from :meth:`rebase_onto_main`:
        * A real conflict (``rebase_onto_main`` returns ``False``) is
          returned unchanged — the caller's existing conflict handling is
          untouched.
        * ``n_before == 0`` (an empty task branch, or a legitimate
          fast-forward) is a no-op — the guard only ever fires on a TOTAL
          wipe of existing work, never on an empty branch.
        * A partial drop (``n_after > 0`` but less than *n_before*) does NOT
          fire — only the total wipe does.
        * A total wipe (``n_after == 0``) is still not automatically treated
          as a reset: if every one of the branch's pre-rebase commits is
          patch-id-equivalent to a commit already applied at the baseline
          (per ``git cherry``) — e.g. a re-dispatched task whose work
          previously landed, or cherry-picked content — ``git rebase``
          legitimately skipped replaying them; that's a dedup, not a wipe,
          and the guard returns ``True`` without touching HEAD. Any
          uncertainty in that determination (the ``git cherry`` call itself
          failing, or producing no output) fails toward treating it as a
          wipe rather than silently trusting it.
        * Both commit counts fail-safe to ``0`` on a git error or unparseable
          output (mirroring the ``rev-list --count`` + int-parse idiom used
          by ``get_rebase_distance``/``_branch_has_commits_beyond_main``): an
          unmeasurable *n_before* means there's no basis to claim a wipe
          happened (guard no-ops, same as a genuinely empty branch), while an
          unmeasurable *n_after* is treated the same as a confirmed wipe
          (fail toward restoring + escalating rather than silently trusting
          an unreadable post-state).
        * If ``git rev-parse HEAD`` itself fails (an unreadable pre-rebase
          HEAD, ``pre_rebase_head == ''``), there is no usable restore
          target for a recovery ``git reset --hard`` — *n_before* is forced
          to ``0`` so the guard no-ops the same way a genuinely unmeasurable
          *n_before* does, rather than proceed toward a wipe check it could
          not actually recover from.
        * The recovery ``git reset --hard`` back to the pre-rebase HEAD is
          itself best-effort: if it fails too, :class:`BranchResetError` is
          still raised, but with ``restore_ok=False`` so the resulting
          escalation says so explicitly rather than asserting the work is
          safe.

        Callers: the three WIP-save requeue/inter-iteration rebase sites
        (``TaskWorkflow._inter_iteration_rebase``, ``GitOps.create_worktree``'s
        cold-requeue reuse block, and ``GitOps._reuse_warm_lane`` — see
        ``WIP_SAFETY_COMMIT_PREFIXES``). Train/merge callers of
        ``rebase_onto_main`` (``stack_train_branches``, the train tip-rebase,
        the suffix_graph frozen-tip rebase) intentionally keep calling the
        unguarded primitive directly — see this task's design decisions for
        why an unconditional guard on the primitive itself would be wrong.
        """
        baseline_ref = onto if onto is not None else self.config.main_branch
        rc, baseline_out, _ = await _run(
            ['git', 'rev-parse', baseline_ref], cwd=worktree,
        )
        baseline_sha = baseline_out.strip() if rc == 0 else baseline_ref

        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        pre_rebase_head = head_out.strip() if rc == 0 else ''

        rc, before_out, _ = await _run(
            ['git', 'rev-list', '--count', f'{baseline_sha}..HEAD'],
            cwd=worktree,
        )
        try:
            n_before = int(before_out.strip()) if rc == 0 else 0
        except ValueError:
            n_before = 0

        if not pre_rebase_head:
            # HEAD itself was unreadable — there is no usable restore target
            # if a wipe were later detected (`git reset --hard ''` would
            # fail with "ambiguous argument"). Treat the pre-state as
            # unmeasurable and no-op the guard, mirroring the existing
            # fail-safe treatment of an unreadable n_before.
            n_before = 0

        ok = await self.rebase_onto_main(worktree, onto=onto)
        if not ok:
            return False

        if n_before > 0:
            rc, after_out, _ = await _run(
                ['git', 'rev-list', '--count', f'{baseline_sha}..HEAD'],
                cwd=worktree,
            )
            try:
                n_after = int(after_out.strip()) if rc == 0 else 0
            except ValueError:
                n_after = 0
            if n_after == 0:
                # Candidate wipe — but a rebase that legitimately dedups
                # against patch-equivalent commits already applied at the
                # baseline also collapses commits-over-baseline to zero.
                # `git cherry` marks each pre-rebase commit '-' (already
                # applied at baseline_sha) or '+' (genuinely unique); only
                # a confirmed all-'-' result exempts this from being
                # treated as a wipe.
                rc, cherry_out, _ = await _run(
                    ['git', 'cherry', baseline_sha, pre_rebase_head], cwd=worktree,
                )
                cherry_lines = cherry_out.strip().splitlines() if rc == 0 else []
                already_landed = bool(cherry_lines) and all(
                    line.startswith('-') for line in cherry_lines
                )
                if already_landed:
                    logger.info(
                        'rebase_preserving_task_commits: %s collapsed to 0 '
                        'commits over baseline %s but all %d pre-rebase '
                        "commit(s) are patch-id dedups already applied "
                        'there — treating as a legitimate rebase, not a '
                        'wipe',
                        worktree, baseline_sha, n_before,
                    )
                    return True

                reset_rc, _, reset_err = await _run(
                    ['git', 'reset', '--hard', pre_rebase_head], cwd=worktree,
                )
                if reset_rc != 0:
                    logger.error(
                        'rebase_preserving_task_commits: failed to restore '
                        'pre-rebase HEAD %s in %s after detecting a wipe '
                        '(git reset --hard rc=%s: %s) — the task\'s work may '
                        'be unrecoverable from the worktree; check git reflog',
                        pre_rebase_head, worktree, reset_rc, reset_err,
                    )
                raise BranchResetError(
                    worktree, onto, pre_rebase_head, n_before,
                    restore_ok=reset_rc == 0,
                )

        return True

    async def merge_tree_conflicts(
        self,
        base_tip: str,
        branch_head: str,
    ) -> ConflictProbe:
        """Probe whether *branch_head* would merge cleanly onto *base_tip*.

        Uses ``git merge-tree --write-tree --name-only -z`` to perform an
        object-store-only 3-way merge (git auto-computes the merge-base).

        **Object-store-only contract** — MUST NOT touch worktree_base or any
        warm-lane path.  This primitive runs at ``self.project_root`` (the
        in-repo object store) and writes only loose tree/blob objects.
        Consumers δ (conflict-graph), η (bounce), and ι (drift metric) all
        depend on this ref-stable, disk-free, idempotent contract.

        **Performance note** — each call forks a git subprocess (fork/exec).
        Consumer δ (conflict-graph) probes O(n²) branch pairs by design;
        it MUST cache results or batch probes rather than calling this method
        naively for every pair on every scheduler tick.

        Returns
        -------
        ConflictProbe(clean=True, conflicted_paths=[])
            when branch_head merges cleanly onto base_tip.
        ConflictProbe(clean=False, conflicted_paths=[...])
            when at least one conflict is detected.  Paths are relative to
            the repo root; ``-z`` (NUL-delimited) + ``-c core.quotePath=false``
            together make paths byte-faithful for any filename, including those
            containing newlines or non-ASCII characters.

        Raises
        ------
        RuntimeError
            when git exits with a code other than 0 or 1 (e.g. 128 for an
            unknown / bad object name), OR when git exits 1 with no stdout
            (error rather than a genuine conflict).  Silently returning
            clean=True would admit a broken branch into the verify frontier;
            returning clean=False would falsely bounce a mergeable branch.
            Loud failure for a caller bug is the correct contract for a
            foundation primitive.
        """
        # Run at self.project_root (in-repo object store) — NEVER at a
        # worktree/lane path.  git merge-tree --write-tree performs a real
        # 3-way merge writing ONLY loose tree/blob objects to the object store;
        # it does not create worktrees, mutate the index, perform a checkout,
        # or write any refs.  Consumers δ/η/ι depend on this invariant.
        rc, out, err = await _run(
            [
                'git', '-c', 'core.quotePath=false',
                'merge-tree', '--write-tree', '--name-only', '-z',
                base_tip, branch_head,
            ],
            cwd=self.project_root,
        )
        if rc == 0:
            return ConflictProbe(clean=True, conflicted_paths=[])
        if rc == 1:
            # Distinguish a genuine conflict from a git error (e.g. bad/unknown
            # ref).  A genuine conflict always has a merged-tree OID as the
            # first NUL-terminated field; a git error (e.g. "not something we
            # can merge") also exits 1 but writes nothing to stdout.
            #
            # stdout layout for a genuine conflict (git 2.43.0, --name-only -z):
            #   field 0: merged tree OID (NUL-terminated)
            #   fields 1..k: conflicted file paths (one per NUL-terminated field)
            #   empty field (\0\0 boundary): section separator
            #   remaining fields: stage/informational data (suppressed by -z
            #                     in some versions; may still appear — stop at
            #                     the first empty field regardless)
            #
            # NUL is not Python whitespace, so _run's .strip() preserves it;
            # splitting on '\0' handles filenames that contain newlines.
            fields = out.split('\0')
            oid_field = fields[0] if fields else ''
            if not oid_field:
                # stdout is empty → git reported an error (e.g. bad ref).
                raise RuntimeError(
                    f'git merge-tree failed (rc={rc}) '
                    f'merging {branch_head} onto {base_tip}: {err}'
                )
            paths: list[str] = []
            for field in fields[1:]:  # skip the tree OID at field 0
                if field == '':  # empty field = \0\0 section boundary
                    break
                paths.append(field)
            return ConflictProbe(clean=False, conflicted_paths=paths)
        raise RuntimeError(
            f'git merge-tree failed (rc={rc}) '
            f'merging {branch_head} onto {base_tip}: {err}'
        )

    async def stack_train_branches(self, member_ids: list[str]) -> TrainStackResult:
        """Materialize a linear branch stack for a merge-train formation.

        The anchor (``member_ids[0]``) is always the stack base and always
        survives — it is NOT rebased (the _do_train_merge tip-rebase at
        merge time handles the anchor→main rebase).

        Each successor member's worktree (``self.worktree_base / member_id``)
        is rebased onto the last-surviving member's branch
        (``self.config.branch_prefix + last_good_id``) via
        ``rebase_onto_main(wt, onto=...)``.

        On a clean rebase the member is appended to *survivors* and becomes
        the new last-good predecessor for the next member.

        On a rebase conflict the member is added to *ejected*; the last-good
        predecessor is NOT advanced, so the next member re-links onto the last
        survivor (re-link invariant).  The conflicting branch is left clean by
        rebase_onto_main's ``git rebase --abort``.

        A missing worktree directory is treated as an eject (defensive;
        logged at WARNING level).

        Args:
            member_ids: Ordered list of member task ids, anchor first.

        Returns:
            TrainStackResult(survivors, ejected).
        """
        if not member_ids:
            return TrainStackResult(survivors=[], ejected=[])

        anchor_id = member_ids[0]
        survivors: list[str] = [anchor_id]
        ejected: list[str] = []
        last_good_id = anchor_id

        for member_id in member_ids[1:]:
            wt_path = self.worktree_base / member_id
            if not wt_path.is_dir():
                logger.warning(
                    'stack_train_branches: worktree %s not found for member %s — ejecting',
                    wt_path, member_id,
                )
                ejected.append(member_id)
                # Do not advance last_good_id — next member re-links onto last survivor.
                continue

            onto_branch = f'{self.config.branch_prefix}{last_good_id}'
            success = await self.rebase_onto_main(wt_path, onto=onto_branch)
            if success:
                survivors.append(member_id)
                last_good_id = member_id
            else:
                ejected.append(member_id)
                # Do not advance last_good_id.

        return TrainStackResult(survivors=survivors, ejected=ejected)

    async def materialize_member_solo(
        self,
        member_id: str,
        predecessor_ref: str,
        *,
        solo_prefix: str = '_solo-',
    ) -> WorktreeInfo | None:
        """Un-stack a train member's own delta onto current main.

        Creates an isolated ``<solo_prefix><member_id>`` branch starting at the
        member's current branch tip (``branch_prefix + member_id``) and checks
        it out in a new worktree at ``worktree_base / <solo_prefix><member_id>``.
        Then runs::

            git rebase --onto <main_branch> <predecessor_ref>

        inside that worktree, replaying only the commits between
        ``predecessor_ref`` and the member branch tip onto the current main
        HEAD.  This is the opposite of the cumulative stacking performed by
        ``stack_train_branches``: it extracts the member's *own* delta.

        On success returns a :class:`WorktreeInfo` whose *path* is the solo
        worktree directory and *base_commit* is the rebased tip SHA.  The
        caller is responsible for cleaning up the worktree via
        :meth:`cleanup_merge_worktree` when done.

        On rebase conflict (non-zero rc): aborts the rebase, removes the
        worktree and its temporary branch, and returns ``None``.  No dangling
        ``_solo-*`` worktrees or branches are left behind.

        Does NOT hold ``_merge_lock`` — runs outside the lock like
        ``rebase_onto_main`` and ``stack_train_branches``.

        Args:
            member_id: The member's task id (used to locate its branch/worktree).
            predecessor_ref: The ref that forms the base of the member's own
                commits — ``task/<predecessor_id>`` for non-anchors, or
                ``self.config.main_branch`` for the anchor.
            solo_prefix: Prefix for the temporary branch/worktree name.
                Defaults to ``'_solo-'``.

        Returns:
            WorktreeInfo on success, None on conflict.
        """
        member_branch = f'{self.config.branch_prefix}{member_id}'
        solo_name = f'{solo_prefix}{member_id}'
        solo_wt = self.worktree_base / solo_name
        solo_wt.parent.mkdir(parents=True, exist_ok=True)

        # Defensively pre-clean any stale solo artifacts for this member from a
        # prior crashed or un-torn-down attribution run.  A leaked _solo-<id>
        # branch/worktree from the previous attempt would make `git worktree add
        # -b _solo-<id>` fail (rc!=0) → materialize returns None → member is
        # mis-classified as 'unstackable'.  All three commands are best-effort
        # (we ignore their rc) — if there is nothing stale they are no-ops.
        await _run(['git', 'worktree', 'remove', str(solo_wt), '--force'],
                   cwd=self.project_root)
        await self._prune_registrations(context='materialize_member_solo')
        await _run(['git', 'branch', '-D', solo_name], cwd=self.project_root)

        # Create a temporary branch _solo-<member_id> starting at the member's
        # current tip and check it out in an isolated worktree.  Using -b with
        # the member branch as start-point avoids modifying the original branch.
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '-b', solo_name, str(solo_wt), member_branch],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(
                'materialize_member_solo: failed to create worktree %s for member %s: %s',
                solo_wt, member_id, err,
            )
            return None

        # Rebase the temporary branch's commits (predecessor_ref..HEAD) onto main.
        rc, _, err = await _run(
            ['git', 'rebase', '--onto', self.config.main_branch, predecessor_ref],
            cwd=solo_wt,
        )
        if rc != 0:
            # Abort the rebase and clean up both the worktree and temp branch.
            await _run(['git', 'rebase', '--abort'], cwd=solo_wt)
            logger.info(
                'materialize_member_solo: rebase conflict for member %s '
                '(predecessor=%s): %s — cleaning up',
                member_id, predecessor_ref, err,
            )
            # Remove the worktree first, then delete the branch.
            rm_rc, _, rm_err = await _run(
                ['git', 'worktree', 'remove', str(solo_wt), '--force'],
                cwd=self.project_root,
            )
            if rm_rc != 0:
                logger.warning(
                    'materialize_member_solo: failed to remove worktree %s: %s',
                    solo_wt, rm_err,
                )
            del_rc, _, del_err = await _run(
                ['git', 'branch', '-D', solo_name],
                cwd=self.project_root,
            )
            if del_rc != 0:
                logger.warning(
                    'materialize_member_solo: failed to delete branch %s: %s',
                    solo_name, del_err,
                )
            return None

        # Resolve the rebased tip SHA.
        _, tip_sha_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=solo_wt)
        tip_sha = tip_sha_raw.strip()

        logger.info(
            'materialize_member_solo: member %s un-stacked onto main, '
            'solo branch %s tip=%s worktree=%s',
            member_id, solo_name, tip_sha, solo_wt,
        )
        return WorktreeInfo(path=solo_wt, base_commit=tip_sha)

    async def delete_solo_branch(self, solo_branch: str) -> None:
        """Delete a bare ``_solo-<id>`` branch and prune any stale worktree entry.

        Companion to :meth:`materialize_member_solo`.  Called by
        ``_attribute_train_failure`` (and its tests) to tear down the temporary
        solo branch after a passer has been landed (or when the all-pass
        interaction case land-nothing branch is taken).

        Unlike :meth:`cleanup_worktree`, this method operates on the BARE
        branch name (e.g. ``_solo-b2``) — **NOT** prefixed with
        ``config.branch_prefix`` — because the solo branch is created without
        the prefix by :meth:`materialize_member_solo`.

        Best-effort / never-raises: non-zero rc from either git command is
        logged as a warning and ignored, mirroring the
        :meth:`cleanup_merge_worktree` never-raise contract.

        Args:
            solo_branch: Bare solo branch name (e.g. ``'_solo-b2'``).
        """
        # Prune first so that a removed (but not yet pruned) worktree entry
        # does not keep git from deleting the branch. Failure/refusal is
        # logged by the chokepoint itself (context-tagged 'delete_solo_branch').
        await self._prune_registrations(context='delete_solo_branch')

        del_rc, _, del_err = await _run(
            ['git', 'branch', '-D', solo_branch], cwd=self.project_root,
        )
        if del_rc != 0:
            logger.warning(
                'delete_solo_branch: failed to delete branch %s: %s',
                solo_branch, del_err,
            )
        else:
            logger.debug('delete_solo_branch: deleted %s', solo_branch)

    async def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Return True if *ancestor* is an ancestor of *descendant*."""
        rc, _, _ = await _run(
            ['git', 'merge-base', '--is-ancestor', ancestor, descendant],
            cwd=self.project_root,
        )
        return rc == 0

    async def branch_content_in_main(self, branch: str) -> bool:
        """Return True iff every file *branch* touched is byte-identical on main.

        Companion check to :meth:`is_ancestor` for landings that are NOT
        ancestors of main — squashed, rebased, or manually-applied commits
        whose content nonetheless matches what *branch* set out to change.
        Computes ``changed = git diff --name-only <merge-base> <branch>``
        (the branch's own changed files vs its base) and returns whether
        ``git diff --quiet <main> <branch> -- <changed...>`` reports no
        difference — i.e. main already carries identical content for every
        one of those paths.

        Returns False (never raises) when:
        - the merge-base cannot be resolved (git error);
        - *branch* has zero commits beyond its base (``changed`` is empty —
          the degenerate/no-work guard; an empty pathspec diff would
          trivially report "no difference" and false-positive); or
        - any changed file differs between main and *branch*.

        Fail-safe by construction: only an rc==0 ``git diff --quiet`` counts
        as "content already landed" — any other git error also falls through
        to False, so this primitive never claims a landing on doubt.

        **Path-quoting caveat**: like :meth:`commit_effect_present_in_main`,
        the ``--name-only`` diff above is read without ``-z``/``-c
        core.quotePath=false``, so a changed path containing non-ASCII (or
        otherwise "unusual") bytes comes back quoted and then fails to
        match itself as a ``--`` pathspec on the follow-up ``diff --quiet``
        call — an empty/mismatched pathspec makes that call report "no
        difference" (rc == 0) rather than erroring, i.e. a FALSE POSITIVE
        for "content already landed" rather than the intended fail-safe
        False. Not hardened here; :meth:`commit_effect_present_in_main`
        carries the ``-z``/``core.quotePath=false`` fix (task 2500
        amendment) and empirically confirms this exact failure mode.

        **Accepted risk — coincidental match on incomplete work**: this
        primitive only compares the files *branch* has touched so far
        against its own merge-base, not the task's full intended scope.  A
        branch that is genuinely mid-task (e.g. it has only gotten around to
        one of several files it will eventually touch) can still return True
        here if that one file happens to already match main's independent
        content — main receiving the same change for unrelated reasons, or
        the branch itself having reverted the file back to match main.  This
        is a deliberate tradeoff so this primitive can catch real
        squash/rebase/manually-applied landings that are NOT ancestors of
        main; callers that need stronger evidence before treating a landing
        as authoritative should additionally require a task-citing commit on
        main (see ``Harness._already_landed_dispatch_gate``, which anchors on
        such a citation when one is present).
        """
        rc, merge_base, _ = await _run(
            ['git', 'merge-base', self.config.main_branch, branch],
            cwd=self.project_root,
        )
        if rc != 0 or not merge_base:
            return False
        rc, changed_out, _ = await _run(
            ['git', 'diff', '--name-only', merge_base, branch],
            cwd=self.project_root,
        )
        if rc != 0:
            return False
        changed = [f for f in changed_out.strip().splitlines() if f.strip()]
        if not changed:
            return False
        rc, _, _ = await _run(
            ['git', 'diff', '--quiet', self.config.main_branch, branch, '--', *changed],
            cwd=self.project_root,
        )
        return rc == 0

    async def commit_effect_present_in_main(self, commit_sha: str) -> bool:
        """Return True iff *commit_sha*'s own effect is still present at main HEAD.

        Companion check to :meth:`is_ancestor` for the found_on_main
        post-hoc-revert blind spot (task 2500): a cited commit can remain
        an ancestor of main forever — ancestry is immutable history — even
        after a LATER commit on main changes exactly the paths it
        touched.  ``is_ancestor`` alone cannot see that the commit's own
        effect is gone from current HEAD.

        Resolves *commit_sha*'s parents via ``git rev-list --parents -n 1
        <commit_sha>`` and branches on parent count:

        - **Merge commit** (2+ parents; task 2675 FIX 1′) — the old plain
          ``diff-tree`` touched-set is empty by git's own default
          behavior for merge commits, which used to make this primitive
          return True *unconditionally* for every merge (the task-1175
          "reverted merge" blind spot: a ``Merge task/1175 into main``
          marker exists and the merge commit is an ancestor of main
          forever, but a later commit on main removed the deliverable —
          effect NOT present, yet the old code said True).  Instead this
          diffs EVERY non-first parent's (each merged branch's) content
          against current main, requiring ALL of them to still be present
          (task 2675 amendment — octopus-merge safety, so a later revert
          of a third-or-later parent's deliverable cannot silently read
          as effect-present): for each ``other_parent`` in
          ``parents[1:]``, ``merge_base = git merge-base <parents[0]>
          <other_parent>`` (that parent's FORK POINT — stable regardless
          of later main history; **CRITICAL**: this must be
          ``merge-base(first_parent, other_parent)``, NOT
          ``merge-base(main, other_parent)`` — because the merge commit
          is itself an ancestor of main in the found_on_main scenario,
          ``merge-base(main, other_parent)`` collapses to
          ``other_parent`` and yields an empty, useless diff), then
          ``touched = git -c core.quotePath=false diff --name-only -z
          <merge_base> <other_parent>`` (the paths that parent introduced
          since its fork point), and finally whether ``git diff --quiet
          <other_parent> <main> -- <touched...>`` reports no difference —
          i.e. main HEAD still carries that parent's content
          byte-identical for every path it touched.  For an ordinary
          two-parent merge this is exactly one iteration, byte-identical
          to checking the second parent alone.

        - **Non-merge commit** (root or single-parent) — UNCHANGED from
          prior behavior (task 2500): ``touched = git -c
          core.quotePath=false diff-tree --no-commit-id --name-only -r
          -z <commit_sha>`` (the commit's own diff against its sole
          parent) and, when non-empty, whether ``git diff --quiet
          <commit_sha> <main> -- <touched...>`` reports no difference.

        ``-z`` + ``core.quotePath=false`` together make every path list
        byte-faithful for any filename, including non-ASCII or
        newline-containing ones — see the path-quoting caveat on
        :meth:`branch_content_in_main`, which shares this primitive's
        underlying merge-base/diff/diff-quiet pattern but not yet this
        hardening.

        Returns True (path-based revert detection inapplicable) when:
        - the commit is non-merge and its own touched-set is empty — a
          genuinely empty ordinary commit.  This deliberately preserves
          prior mark-done behavior for that case (task 2500).

        Returns False (fail-safe — never claim an effect is present on
        doubt) when:
        - ``rev-list --parents`` errors or returns nothing (rc != 0, or
          *commit_sha* is unresolvable);
        - for a merge commit, the ``merge-base`` call errors/is empty for
          ANY non-first parent, that parent's ``diff --name-only`` call
          errors, or its touched-set is empty — an empty branch merge has
          no deliverable to confirm on main, so this is fail-safe False
          (unlike the non-merge empty-touched case above, which stays
          True); the per-parent check short-circuits on the first
          failing parent, so an octopus merge (3+ parents) requires
          EVERY parent to pass;
        - for a non-merge commit, the ``diff-tree`` call errors (rc !=
          0);
        - the final ``diff --quiet`` call errors for a reason other than
          "paths differ" (rc not in {0, 1}); or
        - any touched path differs (rc == 1) between the relevant commit
          (*commit_sha* for non-merge, any non-first parent for merge) and
          main HEAD — produced by a post-hoc revert of those paths, but
          equally by any OTHER later change to the same paths (e.g.
          another already-landed task's follow-up edit, or this task's
          own later commit on the same branch overlapping the same
          files).  This primitive cannot distinguish the two; see the
          accepted-risk note below.

        **Accepted risk — later evolution reads the same as a revert**:
        because this primitive only compares the relevant commit's own
        touched paths against current main HEAD, ordinary subsequent
        evolution of those paths (not just a genuine revert) also
        returns False here.  This is a deliberate fail-safe trade-off,
        not a bug: the caller's own recovery path on False is idempotent
        (re-open to pending / withhold the flip — never a wrong terminal
        state), so the cost of a false negative here is a re-check,
        whereas a false True would wrongly cement a completion that
        never happened. Callers with a same-branch multi-commit shape
        should anchor this check on the branch's own tip rather than a
        possibly-stale intermediate commit — see
        ``Harness._already_landed_dispatch_gate``'s citation-lineage
        handling (task 2500).

        **Accepted risk — conflict-resolved merges**: the merge-commit
        branch above compares each non-first parent's OWN pre-merge
        content against main, not the merge commit's own
        conflict-resolution snapshot.  A merge that needed manual
        conflict resolution can therefore read as effect-absent (False)
        here even though the merge landed cleanly on main, because the
        resolved content on main no longer matches that parent's
        unresolved pre-merge blob for the conflicting paths.  Same
        fail-safe trade-off as above: a false False costs the caller an
        idempotent re-check, never a wrongly-cemented completion.
        """
        rc, parents_out, _ = await _run(
            ['git', 'rev-list', '--parents', '-n', '1', commit_sha],
            cwd=self.project_root,
        )
        if rc != 0 or not parents_out:
            return False
        parents = parents_out.split()[1:]

        if len(parents) >= 2:
            # Merge commit (task 2675 FIX 1′): check EVERY non-first
            # parent's (each merged branch's) content — the paths it
            # touched since its fork point — against current main HEAD.
            # For an ordinary two-parent merge this is exactly one
            # iteration (byte-identical to the original second-parent-only
            # check); for an octopus merge (3+ parents) ALL parents must
            # pass, else a later revert of a third-or-later parent's
            # deliverable would silently read as effect-present (task 2675
            # amendment — the octopus blind spot).  Touched paths MUST
            # derive from merge-base(first_parent, other_parent), NOT
            # merge-base(main, other_parent) — see the docstring above.
            first_parent = parents[0]
            for other_parent in parents[1:]:
                rc, merge_base, _ = await _run(
                    ['git', 'merge-base', first_parent, other_parent],
                    cwd=self.project_root,
                )
                if rc != 0 or not merge_base:
                    return False
                rc, touched_out, _ = await _run(
                    [
                        'git', '-c', 'core.quotePath=false',
                        'diff', '--name-only', '-z', merge_base, other_parent,
                    ],
                    cwd=self.project_root,
                )
                if rc != 0:
                    return False
                touched = [f for f in touched_out.split('\0') if f]
                if not touched:
                    # Empty branch merge — no deliverable to confirm; fail-safe.
                    return False
                rc, _, _ = await _run(
                    [
                        'git', 'diff', '--quiet', other_parent,
                        self.config.main_branch, '--', *touched,
                    ],
                    cwd=self.project_root,
                )
                if rc != 0:
                    return False
            return True

        # Non-merge (root or single-parent) commit: unchanged existing logic.
        rc, touched_out, _ = await _run(
            [
                'git', '-c', 'core.quotePath=false',
                'diff-tree', '--no-commit-id', '--name-only', '-r', '-z', commit_sha,
            ],
            cwd=self.project_root,
        )
        if rc != 0:
            return False
        touched = [f for f in touched_out.split('\0') if f]
        if not touched:
            return True
        rc, _, _ = await _run(
            ['git', 'diff', '--quiet', commit_sha, self.config.main_branch, '--', *touched],
            cwd=self.project_root,
        )
        return rc == 0

    async def worktree_head_beyond_main(self, worktree: Path) -> str | None:
        """Return the HEAD SHA when *worktree* carries commits beyond main, else None.

        Resolves ``HEAD`` in *worktree* via ``git rev-parse HEAD`` and returns
        the stripped SHA only when it is **not** an ancestor of main (i.e. the
        worktree holds unmerged commits).  Returns ``None`` when:

        * the worktree directory cannot be read (``git rev-parse`` rc≠0), or
        * the HEAD commit is already an ancestor of (or equal to) main.

        This is the single source of truth for the absent-ref fallback shared
        by :meth:`merge_to_main` (merge-source selection) and
        :func:`_classify_branch_presence` (proceed-vs-misroute decision).
        Extracting it here keeps both callers in lockstep: a future change to
        the "beyond main" definition only needs to be made in one place.

        .. note::
            This method does **not** check which branch the worktree is
            attached to.  The ``_classify_branch_presence`` function runs the
            symbolic-ref misroute guard before calling this helper, so the
            branch-identity vetting is handled there.  Callers invoking
            :meth:`merge_to_main` directly are responsible for ensuring the
            worktree belongs to the intended task before relying on this
            fallback.
        """
        rc_head, head_sha_raw, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree,
        )
        if rc_head != 0:
            return None
        head_sha = head_sha_raw.strip()
        main_sha = await self.get_main_sha()
        if not await self.is_ancestor(head_sha, main_sha):
            return head_sha
        return None

    async def has_uncommitted_work(self, worktree: Path) -> bool:
        """Return True if worktree has staged or unstaged changes.

        A leftover ``.task/`` (if any) is covered by this repo's root
        ``.gitignore`` (a tracked ``.task/`` entry every worktree inherits),
        so it never surfaces in ``git status`` output — no pathspec
        exclusion is needed here.
        """
        rc, output, _ = await _run(
            ['git', 'status', '--porcelain', '--', '.'],
            cwd=worktree,
        )
        return rc == 0 and bool(output.strip())

    async def get_commit_subjects(
        self, worktree: Path, base_sha: str,
    ) -> list[tuple[str, str]]:
        """Return HEAD-first (sha, subject) pairs for ``base_sha..HEAD``.

        Uses the ``\\x1f`` unit separator (rather than a space or colon) to
        split each log line, since commit subjects may themselves contain
        colons or arbitrary punctuation.  Returns ``[]`` on any git error or
        an empty range (``base_sha == HEAD``) — never raises.

        Used by ``TaskWorkflow._detect_tip_wip_commits`` to scan for
        WIP safety-commits sitting at branch HEAD.
        """
        rc, out, _ = await _run(
            ['git', 'log', '--format=%H\x1f%s', f'{base_sha}..HEAD'],
            cwd=worktree,
        )
        if rc != 0 or not out.strip():
            return []
        pairs = []
        for line in out.splitlines():
            if not line.strip():
                continue
            sha, _, subject = line.partition('\x1f')
            pairs.append((sha, subject))
        return pairs

    async def find_equivalent_commit(
        self, worktree: Path, base_sha: str, target_sha: str,
    ) -> str | None:
        """Recover the sha a rebase replayed *target_sha* onto, via ``git patch-id``.

        When a requeue / inter-iteration rebase rewrites a task branch, an
        already-recorded done-step commit (*target_sha*) can be orphaned — no
        longer reachable from HEAD — even though its content was replayed
        byte-for-byte onto the new base as a fresh commit. This locates that
        replayed commit so :meth:`TaskWorkflow._reconcile_done_step_commits`
        can re-point the plan step to it instead of escalating.

        Two tiers, tried in order:

        1. **patch-id** (exact): compute *target_sha*'s patch-id
           (``git log -p -1 --no-color`` piped to ``git patch-id --stable``)
           and look it up in a ``{patch-id: sha}`` map built the same way over
           ``base_sha..HEAD``. ``git patch-id --stable`` hashes the normalized
           diff hunks, so a clean replay reproduces byte-identical hunks and an
           identical patch-id — the canonical, whitespace/line-number-robust
           equivalence the ``git cherry`` dedup in
           :meth:`rebase_preserving_task_commits` also relies on. Using the
           identical diff-emitting command (``git log -p``) for both the single
           target and the range guarantees both sides are normalized by the
           same code path. The ``<patch-id> <sha>`` line the range form emits
           maps the shared patch-id to the REPLAYED (HEAD-side) sha, which is
           exactly the remap target. If TWO commits in ``base_sha..HEAD`` share
           a patch-id (a diff reverted then re-applied, or genuinely identical
           hunks), that patch-id is ambiguous — we cannot tell which sha the
           orphan replayed onto — so it is skipped rather than resolved to an
           arbitrary one of them, the same 'never guess' posture tier 2 applies
           to an ambiguous subject.
        2. **unique exact-subject** (fallback): when the patch-id lookup
           misses — e.g. a rebase that resolved a conflict altered the diff
           but preserved the commit message — recover the replayed sha only if
           EXACTLY ONE commit in ``base_sha..HEAD`` shares *target_sha*'s
           subject (the ``%H\\x1f%s`` idiom from :meth:`get_commit_subjects`,
           ``\\x1f``-split because subjects may contain colons/punctuation). An
           ambiguous subject (two or more matches) or no match returns
           ``None`` — the method fails toward the caller's escalation, never
           toward a wrong re-point.

        Fully fail-safe: any git error, an unresolvable/GC'd *target_sha*, or
        empty patch-id output yields ``None``, letting the caller fall through
        to its existing escalation path. No persisted state — the mapping is
        re-derived from live git on every call, so it behaves identically
        across orchestrator restarts, and a false negative simply reverts to
        the pre-fix baseline rather than sinking the caller. Same best-effort
        posture as :meth:`get_commit_subjects` / :meth:`get_commit_changed_files`.
        """
        try:
            # Tier 1: patch-id equivalence. Compute the orphaned target's
            # patch-id from its own diff.
            rc, target_diff, _ = await _run(
                ['git', 'log', '-p', '-1', '--no-color', target_sha], cwd=worktree,
            )
            if rc != 0 or not target_diff.strip():
                return None
            _, target_pid_out, _ = await _run(
                ['git', 'patch-id', '--stable'], cwd=worktree, input_text=target_diff,
            )
            target_pid = target_pid_out.split()[0] if target_pid_out.strip() else None

            # Build the {patch-id: sha} map over base..HEAD (the live branch).
            rc, range_diff, _ = await _run(
                ['git', 'log', '-p', '--no-color', f'{base_sha}..HEAD'], cwd=worktree,
            )
            if rc == 0 and range_diff.strip():
                _, range_pid_out, _ = await _run(
                    ['git', 'patch-id', '--stable'], cwd=worktree, input_text=range_diff,
                )
                pid_to_sha: dict[str, str] = {}
                ambiguous_pids: set[str] = set()
                for line in range_pid_out.splitlines():
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[0]
                        if pid in pid_to_sha:
                            # Two commits in base..HEAD share a patch-id (a
                            # diff reverted then re-applied, or genuinely
                            # identical hunks). We cannot know which one the
                            # orphaned step replayed onto, so mark the patch-id
                            # ambiguous and fall through — the same 'never
                            # guess' posture tier 2 applies to an ambiguous
                            # subject, rather than silently keeping whichever
                            # colliding sha the dict happens to retain.
                            ambiguous_pids.add(pid)
                        pid_to_sha[pid] = parts[1]
                if (
                    target_pid
                    and target_pid in pid_to_sha
                    and target_pid not in ambiguous_pids
                ):
                    return pid_to_sha[target_pid]

            # Tier 2: unique exact-subject fallback. Only fires when the
            # patch-id lookup missed; recovers the replayed sha only when
            # exactly ONE commit in base..HEAD shares the target's subject, so
            # we never mis-point a step whose subject two replayed commits
            # happen to share (e.g. a generic "feat: GREEN — implementation").
            rc, target_subject, _ = await _run(
                ['git', 'log', '-1', '--format=%s', target_sha], cwd=worktree,
            )
            if rc != 0 or not target_subject.strip():
                return None
            rc, subj_out, _ = await _run(
                ['git', 'log', '--format=%H\x1f%s', f'{base_sha}..HEAD'], cwd=worktree,
            )
            if rc != 0 or not subj_out.strip():
                return None
            subject_matches: list[str] = []
            for line in subj_out.splitlines():
                if not line.strip():
                    continue
                sha, _, subject = line.partition('\x1f')
                if subject == target_subject:
                    subject_matches.append(sha)
            if len(subject_matches) == 1:
                return subject_matches[0]
            return None
        except Exception:
            # Fully fail-safe: any git error / unresolvable target yields None so
            # the caller falls through to its existing escalation path. Log at
            # WARN so a persistent failure is diagnosable rather than silent.
            logger.warning(
                'find_equivalent_commit failed for target_sha=%s base_sha=%s '
                'in %s; returning None (caller will fall through to escalation)',
                target_sha, base_sha, worktree, exc_info=True,
            )
            return None

    async def get_changed_files(self, from_sha: str, to_sha: str) -> list[str]:
        """Return list of files changed between two commits."""
        _, output, _ = await _run(
            ['git', 'diff', '--name-only', from_sha, to_sha],
            cwd=self.project_root,
        )
        return [f for f in output.strip().splitlines() if f.strip()]

    async def get_commit_changed_files(self, sha: str) -> list[str]:
        """Return the files *sha* itself changed relative to its parent.

        Uses ``git diff-tree --no-commit-id --name-only -r --root <sha>`` —
        ``--root`` makes a ROOT commit (no parent) diff against the empty
        tree instead of showing nothing, which plain ``diff-tree``/the
        ``{sha}^..{sha}`` idiom used by :meth:`get_changed_files` cannot
        express for a parentless commit.

        Returns ``[]`` (never raises) on any git error — e.g. *sha* is a
        garbage/nonexistent or GC'd object — so an unresolvable commit
        yields an empty changed-file set rather than propagating a failure.

        Used by ``TaskWorkflow._reconcile_done_step_commits`` (task 2386) to
        match, by filename set only (not byte content), an orphaned
        done-step commit against the tip WIP safety-commit run's
        changed-file set.
        """
        rc, output, _ = await _run(
            ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', '--root', sha],
            cwd=self.project_root,
        )
        if rc != 0:
            return []
        return [f for f in output.strip().splitlines() if f.strip()]

    async def get_rebase_distance(self, old_base: str, new_base: str) -> int:
        """Count commits in ``old_base..new_base`` (i.e. how far main advanced).

        Returns the exact git rev-list count, or ``-1`` on any git error or
        unparseable output.  -1 is a distinct sentinel so an unmeasurable
        distance is never mistaken for a 0-commit (no-op) rebase.

        Modelled on the ``rev-list --count`` + int-parse + fail-safe pattern
        used in ``_freshen_main`` (line ~561) and
        ``_branch_has_commits_beyond_main`` (line ~1508).
        """
        rc, out, _ = await _run(
            ['git', 'rev-list', '--count', f'{old_base}..{new_base}'],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(
                'get_rebase_distance: rev-list failed (rc=%d) for %s..%s',
                rc, old_base, new_base,
            )
            return -1
        try:
            return int(out.strip())
        except ValueError:
            logger.warning(
                'get_rebase_distance: unexpected output %r for %s..%s',
                out, old_base, new_base,
            )
            return -1

    async def get_merge_diff_files(
        self, base_sha: str, head_sha: str,
    ) -> tuple[list[str], Exception | None]:
        """Files changed by the merge ``base_sha..head_sha``, excluding ``.task/``.

        Returns a ``(files, error)`` tuple — **total, never raises**:

        * ``(files, None)`` on success.  An empty ``files`` list is a
          legitimate outcome (revert merges, ``.task/``-only merges).
        * ``([], exception)`` on any error: ``rc != 0`` from ``git diff``
          (returns :class:`subprocess.CalledProcessError`) or an unexpected
          raise from the subprocess helper (e.g. ``WorktreeMissing``,
          ``FileNotFoundError``).

        Callers should branch on ``err is not None`` (not ``resolver_failed``),
        because an empty diff is a valid non-error outcome for this function.

        Used by ``TaskWorkflow._reconcile_metadata_files_for_done`` to write
        the actually-changed paths into ``metadata.files`` instead of the
        architect's ``plan.files`` (which the merge may have squashed or
        refactored away).  Uses ``--no-renames`` so a rename surfaces as
        both add+delete; downstream consumers can decide whether to keep
        or drop the deleted path.
        """
        cmd = [
            'git', 'diff', '--name-only', '--no-renames',
            base_sha, head_sha, '--', ':!.task/',
        ]
        try:
            rc, output, stderr = await _run(cmd, cwd=self.project_root)
        except Exception as exc:
            return [], exc
        if rc != 0:
            logger.warning(
                'get_merge_diff_files: git diff %s..%s failed (rc=%s): %s',
                base_sha, head_sha, rc, (stderr or '').strip()[:200],
            )
            return [], subprocess.CalledProcessError(rc, cmd, output=output, stderr=stderr)
        return [f for f in output.strip().splitlines() if f.strip()], None

    async def get_merge_commit_diff_files(
        self, merge_sha: str,
    ) -> tuple[list[str], Exception | None]:
        """Files this merge introduced relative to its OWN first parent.

        Returns ``get_merge_diff_files(f'{merge_sha}^1', merge_sha)`` —
        inheriting ``--no-renames``, the ``:!.task/`` exclusion, and the
        total never-raises ``(files, err)`` contract unchanged.

        This is the contamination-free anchor for done-time
        ``metadata.files`` capture.  ``merge_to_main`` always merges with
        ``git merge --no-ff``, so *merge_sha* is a two-parent merge commit:
        ``merge_sha^1`` is main's tip immediately BEFORE this merge, and
        ``merge_sha^2`` is the task branch tip that was merged in — the
        same invariant ``advance_main`` already relies on when it derives
        the verified branch tip from ``merge_sha^2`` ("--no-ff guarantees
        M^2 is the branch commit", see ``advance_main`` docstring).

        By symmetry, ``merge_sha^1`` is captured atomically at merge time
        and diffing against it yields EXACTLY this task's own branch
        changes — sibling files present in both trees cancel out.

        Contrast the stale-base two-dot diff a caller might otherwise use
        (a task's own ``_base_commit`` captured once at worktree creation,
        never refreshed after a rebase, against the final ``merge_sha``):
        that diff unions in every sibling task that merged into main during
        the window between this task's branch point and its own merge —
        the reported cross-task ``metadata.files`` contamination.
        """
        return await self.get_merge_diff_files(f'{merge_sha}^1', merge_sha)

    async def get_files_touched_in_branch(
        self, base_sha: str, branch_head: str,
    ) -> list[str]:
        """Files touched by any commit in ``base_sha..branch_head``.

        Union of file paths that appeared in any commit on the branch
        (history-based, not just the diff).  Used by the pre-merge
        Decision-1 check: an architect-declared plan target is "touched"
        if it appears in this set.

        Excludes ``.task/`` and uses ``--no-renames`` so a rename
        surfaces both old and new paths (the old path is "touched" too).

        Returns ``[]`` on git error so the helper fails open — its
        consumer logs and proceeds rather than blocking the merge on
        a transient diff error.
        """
        rc, output, stderr = await _run(
            [
                'git', 'log', '--name-only', '--no-renames',
                '--pretty=format:', f'{base_sha}..{branch_head}',
                '--', ':!.task/',
            ],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(
                'get_files_touched_in_branch: git log %s..%s failed (rc=%s): %s',
                base_sha, branch_head, rc, (stderr or '').strip()[:200],
            )
            return []
        seen: set[str] = set()
        for ln in output.splitlines():
            ln = ln.strip()
            if ln:
                seen.add(ln)
        return sorted(seen)

    async def get_branch_changed_files(
        self, ref: str,
    ) -> tuple[list[str], Exception | None]:
        """Files *ref* changed relative to ``main_branch``, excluding ``.task/``.

        Three-dot diff (``main_branch...ref``) — the files touched by *ref*
        since it diverged from ``main_branch``, not a naive two-dot diff
        against main's current (possibly since-advanced) tip.  Returns a
        ``(files, error)`` tuple — **total, never raises** — mirroring
        :meth:`get_merge_diff_files`'s contract and ``.task/`` exclusion:

        * ``(files, None)`` on success.  An empty ``files`` list is a
          legitimate outcome (a branch with no net changes vs main).
        * ``([], exception)`` on any error: ``rc != 0`` from ``git diff``
          (e.g. *ref* does not resolve — returns
          :class:`subprocess.CalledProcessError`) or an unexpected raise
          from the subprocess helper.

        Used by the merge-skew pipeline-landing tripwire
        (:mod:`orchestrator.merge_skew_tripwire`) to compute each in-flight
        task's own changed-file set for overlap against a landing's
        changed files.
        """
        cmd = [
            'git', 'diff', '--name-only',
            f'{self.config.main_branch}...{ref}', '--', ':!.task/',
        ]
        try:
            rc, output, stderr = await _run(cmd, cwd=self.project_root)
        except Exception as exc:
            return [], exc
        if rc != 0:
            stderr_text = (stderr or '').strip()
            if _git_stderr_is_unresolved_ref(stderr or ''):
                # Expected for the merge-skew tripwire's active-task scan: a
                # not-yet-dispatched (pending) task has no task/<id> branch
                # yet, so git cannot resolve the ref.  A quiet DEBUG skip, not
                # a WARNING — a project with many pending tasks would otherwise
                # burst one WARNING per branchless task on every load-bearing
                # landing, burying genuine diff failures (which still WARN
                # below).  Return contract is unchanged: still ([], error).
                logger.debug(
                    'get_branch_changed_files: ref %s does not resolve vs %s '
                    '(rc=%s): %s',
                    ref, self.config.main_branch, rc, stderr_text[:200],
                )
            else:
                logger.warning(
                    'get_branch_changed_files: git diff %s...%s failed (rc=%s): %s',
                    self.config.main_branch, ref, rc, stderr_text[:200],
                )
            return [], subprocess.CalledProcessError(rc, cmd, output=output, stderr=stderr)
        return [f for f in output.strip().splitlines() if f.strip()], None

    async def merge_to_main(
        self,
        worktree: Path,
        branch: str,
        base_sha: str | None = None,
    ) -> MergeResult:
        """Merge a task branch into main using a temporary merge worktree.

        Creates a disposable worktree, performs the merge there, and returns
        the result.  The caller is responsible for calling :meth:`advance_main`
        after verification and :meth:`cleanup_merge_worktree` when done.

        When *base_sha* is provided the merge worktree is created at that
        commit rather than current main HEAD.  This supports speculative
        merges where N+1 is merged against N's merge commit SHA.

        Never touches ``project_root``'s working tree or index.
        Called by the MergeWorker (serialized via the merge queue).

        **Absent-ref fallback**: when ``resolve_queued_branch_ref(branch)``
        returns None (the named ``task/<id>`` ref was deleted while the work
        still sat in the worktree), the merge SOURCE is derived from the
        worktree HEAD via :meth:`worktree_head_beyond_main`.  The merge-commit
        subject remains the canonical ``Merge task/<id> into main`` form so
        that ``find_merge_marker`` keeps already_merged idempotency on
        re-dispatch.  If the worktree HEAD cannot be resolved or is already an
        ancestor of main, the prior ``full_branch`` fallback is preserved so a
        genuine misroute still fails loudly.

        **Misroute note**: this method does not replicate the symbolic-ref
        branch-name guard that :func:`_classify_branch_presence` performs.  In
        the normal MergeWorker flow ``_classify_branch_presence`` runs first
        and rejects attached-HEAD misroutes before this method is reached.
        Callers invoking ``merge_to_main`` directly are responsible for
        ensuring the worktree belongs to the intended task.
        """
        resolved = await self.resolve_queued_branch_ref(branch)
        full_branch = resolved or f'{self.config.branch_prefix}{branch}'

        # Derive the merge source: prefer the named ref; when absent, fall back
        # to the worktree HEAD SHA via the shared helper so the proceed-decision
        # in _classify_branch_presence and the source-selection here stay in
        # lockstep (both call worktree_head_beyond_main).
        # Derive the merge source: prefer the named ref; when absent, fall back
        # to the worktree HEAD SHA via the shared helper so the proceed-decision
        # in _classify_branch_presence and the source-selection here stay in
        # lockstep (both call worktree_head_beyond_main).
        # If the helper returns None (HEAD on/behind main, or unreadable worktree)
        # we fall back to full_branch so git fails loudly with "not something we
        # can merge" — preserving the genuine-misroute signal.
        _head_sha = None if resolved is not None else await self.worktree_head_beyond_main(worktree)
        merge_source: str = resolved or _head_sha or full_branch

        merge_wt: Path | None = None

        try:
            merge_wt, pre_merge_sha = await self._create_merge_worktree(base_sha)

            # Pre-merge cleanup: remove .task/ from filesystem if inherited
            # from a contaminated main.  Belt-and-braces only — .task/
            # execution metadata now lives outside the worktree entirely
            # (see module docstring), so nothing repopulates this directory
            # from the branch being merged.
            task_dir = merge_wt / '.task'
            if task_dir.exists():
                shutil.rmtree(task_dir)

            # Merge with no-ff.
            # merge_source is the named branch ref when present, or the worktree
            # HEAD SHA as an absent-ref fallback.  full_branch is always the
            # canonical prefixed name for _merge_subject so the commit message
            # stays 'Merge task/<id> into main' for find_merge_marker.
            rc, out, err = await _run(
                ['git', 'merge', '--no-ff', merge_source,
                 '-m', _merge_subject(full_branch, self.config.main_branch)],
                cwd=merge_wt,
            )

            if rc != 0:
                if 'CONFLICT' in out or 'CONFLICT' in err:
                    conflict_details = await self.get_conflict_details(merge_wt)
                    return MergeResult(
                        success=False, conflicts=True,
                        details=conflict_details,
                        pre_merge_sha=pre_merge_sha,
                        merge_worktree=merge_wt,
                    )
                # Non-conflict failure — clean up immediately
                await self.cleanup_merge_worktree(merge_wt)
                return MergeResult(
                    success=False, details=f'{out}\n{err}',
                    pre_merge_sha=pre_merge_sha,
                )

            _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=merge_wt)
            return MergeResult(
                success=True, merge_commit=sha,
                pre_merge_sha=pre_merge_sha,
                merge_worktree=merge_wt,
            )

        except BaseException:
            if merge_wt:
                await self.cleanup_merge_worktree(merge_wt)
            raise

    async def _create_merge_worktree(
        self, base_sha: str | None = None,
    ) -> tuple[Path, str]:
        """Create a temporary detached worktree at *base_sha* (or main HEAD).

        When *base_sha* is None the worktree is created at current main HEAD
        (normal case).  When *base_sha* is provided the worktree is created
        at that exact commit, supporting speculative merges where N+1 is
        merged against N's merge commit.
        """
        import uuid
        merge_id = uuid.uuid4().hex[:8]
        merge_wt = self.worktree_base / f'_merge-{merge_id}'
        merge_wt.parent.mkdir(parents=True, exist_ok=True)

        if base_sha is None:
            # Fetch latest (best-effort — no remote in tests)
            await _run(
                ['git', 'fetch', self.config.remote, self.config.main_branch],
                cwd=self.project_root,
            )
            # Capture current main SHA
            _, pre_merge_sha, _ = await _run(
                ['git', 'rev-parse', self.config.main_branch],
                cwd=self.project_root,
            )
            checkout_ref = self.config.main_branch
        else:
            pre_merge_sha = base_sha
            checkout_ref = base_sha.strip()

        # Detached worktree avoids "branch already checked out" error
        rc, _, err = await _run(
            ['git', 'worktree', 'add', '--detach', str(merge_wt), checkout_ref],
            cwd=self.project_root,
        )
        if rc != 0:
            raise RuntimeError(f'Failed to create merge worktree: {err}')

        logger.info(f'Created merge worktree at {merge_wt} (HEAD={pre_merge_sha[:8]})')
        return merge_wt, pre_merge_sha.strip()

    async def remove_merge_worktree_guarded(
        self, path: Path, *, reason: str,
    ) -> RemovalOutcome:
        """Remove a merge worktree, gated by its merge-verify flock (C1).

        PRD: docs/prds/merge-worktree-lifecycle-integrity.md, task alpha.

        Acquire-then-remove, never check-then-remove: non-blocking try-
        acquires *path*'s per-lane merge-verify flock
        (:func:`~orchestrator.verify_cancel.lane_lock_path`, the SAME inode
        a verify holds via :meth:`merge_verify_lease`) and HOLDS it across
        the ``git worktree remove`` — so no verify can start in the window
        between a liveness check and the delete (the incident's 23s
        TOCTOU). A live holder makes the non-blocking acquire fail
        immediately, in which case removal is skipped
        (``'skipped_lease_held'``, logged as a single WARNING naming the
        holder pgid) rather than deferred or retried; a dead or stale
        holder's flock is auto-released by the kernel, so the acquire
        simply succeeds and removal proceeds (fail-open, intrinsic to
        flock — this method never consults holder liveness directly). The
        holder-pgid rendezvous file is read ONLY to name the holder in
        that WARNING — a best-effort, fail-open diagnostic hint, never a
        removal gate.

        **Persistent-worktree exemption**: if *path* resolves to
        :attr:`persistent_merge_worktree_path` OR
        :attr:`persistent_offline_deep_worktree_path`, this method returns
        ``'skipped_persistent'`` WITHOUT touching any flock — both
        persistent worktrees survive across attempts and across verify
        failures regardless of lease state, so there is nothing to
        serialize against.

        On the tree-gone outcomes (``'removed'`` / ``'not_present'``) the
        sibling ``<path>.lock`` flock file is unlinked while the flock is
        still held (mirroring :meth:`ephemeral_worktree`), so an ephemeral
        merge-worktree removal leaves no orphan ``_merge-<uuid>.lock`` behind.
        The lock is NEVER unlinked on ``'skipped_lease_held'`` (this call did
        not acquire it — a live holder's lock is left untouched) or on
        ``'failed'`` (the tree, and thus its lane, survives). This differs
        from :meth:`merge_verify_lease`, whose persistent-lane lock is
        deliberately retained across attempts.

        *reason* is a short caller-supplied label (e.g. the calling
        function's name) recorded in logs for diagnostics.
        """
        if path.resolve() == self.persistent_merge_worktree_path.resolve():
            logger.debug('remove_merge_worktree_guarded: persistent merge worktree retained: %s', path)
            return 'skipped_persistent'
        if path.resolve() == self.persistent_offline_deep_worktree_path.resolve():
            logger.debug('remove_merge_worktree_guarded: persistent offline-deep worktree retained: %s', path)
            return 'skipped_persistent'

        lock_path = lane_lock_path(path)
        fd = acquire_merge_verify_flock(lock_path, 0.0)
        if fd is not None:
            # Registered like every other in-process lane-lock hold (task 3081):
            # this acquire is sub-millisecond and ephemeral-lane-only, but
            # merge_verify_lease(lane_dir=...) can be handed an ephemeral lane
            # (the DF 2822 per-land cross-check), so the inodes CAN coincide.
            # Layer (2) of the leak predicate is only sound if the registry is
            # COMPLETE, and a leak report is a loud human escalation that must
            # never be reachable from a legitimate hold.
            _register_held_lane_lock(fd, lock_path)
        if fd is None:
            holder = read_lock_holder_pgid(self.worktree_base)
            logger.warning(
                'remove_merge_worktree_guarded: merge-verify lease held by live '
                'holder (pgid=%s); skipping removal of %s (reason=%s) -- leaving '
                'for the merge reaper',
                holder, path, reason,
            )
            return 'skipped_lease_held'
        # Only unlink the sibling ``.lock`` file when THIS call both acquired
        # the flock and drove the tree gone (removed/not_present) — never on
        # 'failed' (the tree, and thus its lane, survives) and never on
        # 'skipped_lease_held' (we did not acquire the lock, so we must never
        # yank a live holder's lock file). See the unlink_lock finally below.
        unlink_lock = False
        try:
            if not path.exists():
                unlink_lock = True
                return 'not_present'
            rc, _, err = await _run(
                ['git', 'worktree', 'remove', str(path), '--force'],
                cwd=self.project_root,
            )
            if rc == 0:
                logger.info(
                    'remove_merge_worktree_guarded: removed %s (reason=%s)', path, reason,
                )
                unlink_lock = True
                return 'removed'
            logger.warning(
                'remove_merge_worktree_guarded: failed to remove %s (reason=%s): %s',
                path, reason, err,
            )
            return 'failed'
        finally:
            # Unlink the lock file BEFORE releasing/closing the flock, mirroring
            # ephemeral_worktree (git_ops.py ~1797): a contender that opens the
            # path after our unlink necessarily creates a fresh inode, so it can
            # never observe the lock we are about to drop. Unlinking on the
            # tree-gone outcomes keeps the merge-worktree lane from leaving an
            # orphan ``_merge-<uuid>.lock`` behind — the ephemeral lane no longer
            # exists to serialize against, unlike merge_verify_lease's persistent
            # lane whose lock is deliberately retained across attempts.
            if unlink_lock:
                with contextlib.suppress(Exception):
                    os.unlink(lock_path)
            _forget_held_lane_lock(fd)
            release_merge_verify_flock(fd)

    async def cleanup_merge_worktree(self, merge_wt: Path) -> None:
        """Remove a temporary merge worktree, crash-safely (task 2922).

        Routes through the C1 guarded primitive,
        :meth:`remove_merge_worktree_guarded` — lease-held trees are
        *skipped* (leaving them for the merge reaper) rather than
        force-removed out from under a live verify, and the persistent
        worktrees are exempted. This method inspects the primitive's
        returned outcome and, ONLY on ``'failed'``, applies a crash-safe
        filesystem fallback for the task-2922 shape-1 leak.

        Shape-1 is the canonical trigger: an interrupted teardown
        (SIGTERM/restart mid-merge) leaves a full ``_merge-<uuid>`` checkout
        on disk while its ``.git/worktrees/<name>`` admin dir is already
        gone, so ``git worktree remove --force`` errors ('not a working
        tree') and the primitive returns ``'failed'`` and LEAVES the tree
        (its pinned contract — see task 2924's
        ``test_non_worktree_directory_returns_failed``). But ``'failed'`` is
        NOT proof of shape-1 specifically: it is simply *any* non-zero git
        worktree removal — a ``git worktree lock``-ed tree or a transient
        filesystem/I/O error yield it too. The fallback deliberately does
        NOT try to distinguish the cause; it force-removes any unleased
        ``_merge-`` tree git could not remove, whatever the reason. That is
        safe here because merge worktrees are throwaway/ephemeral by
        construction AND ``'failed'`` already means the primitive's lease
        acquire confirmed NO live holder (a live holder yields
        ``'skipped_lease_held'``), so the tree is unleased and safe to
        remove from the filesystem.

        On ``'failed'`` the fallback: (1) band-guards via
        :meth:`_refuse_foreign_band` (defense-in-depth — the outcome check
        is the primary safety gate, but this can never let a
        non-``_merge-`` path be rmtree'd); (2)
        ``shutil.rmtree(..., ignore_errors=True)`` the on-disk tree git can
        no longer remove; (3) unlinks the sibling :func:`lane_lock_path`
        ``.lock`` (the primitive RETAINS it on ``'failed'``, so we clear the
        now-orphaned lane lock, honouring the no-leak convention); (4)
        :meth:`_prune_registrations` clears any dangling admin entry. The
        order — remove-tree-then-prune — leaves at most the benign inverse
        shape (admin entry without a tree) that git prune / name-reuse
        already handle, so an interrupted teardown is always completable by
        a later sweep.

        Every other outcome (``'removed'`` / ``'not_present'`` / the two
        ``'skipped_*'``) returns immediately, so a live-leased tree and the
        warm persistent lanes are NEVER force-removed. Never raises;
        idempotent — a re-call sees the tree already gone → primitive
        ``'not_present'`` → early return.
        """
        outcome = await self.remove_merge_worktree_guarded(
            merge_wt, reason='cleanup_merge_worktree',
        )
        if outcome != 'failed':
            return

        # Crash-safe fallback (task 2922): the guarded git removal returned
        # 'failed' — i.e. ANY non-zero git worktree removal. Most commonly the
        # shape-1 case (the .git/worktrees/<name> admin dir was already removed
        # by an interrupted teardown), but a locked tree or a transient FS/I/O
        # error too. We intentionally do NOT distinguish the cause: the
        # primitive's lease acquire already confirmed no live holder, so we
        # force-remove this unleased throwaway _merge- tree git could no longer
        # remove, whatever the reason. The band guard is defense-in-depth over
        # the outcome check; it can never fire for a genuine _merge- path.
        if self._refuse_foreign_band(
            merge_wt, frozenset({'_merge-'}), 'cleanup_merge_worktree',
        ):
            return
        logger.warning(
            'cleanup_merge_worktree: git worktree remove failed for %s — '
            'commonly the admin dir was already removed by an interrupted '
            'teardown; applying crash-safe rmtree fallback (task 2922 shape-1)',
            merge_wt,
        )
        shutil.rmtree(merge_wt, ignore_errors=True)
        # The primitive retains the sibling .lock on 'failed'; unlink it now
        # that the lane is gone (mirrors the primitive's 'removed'-path unlink).
        with contextlib.suppress(Exception):
            os.unlink(lane_lock_path(merge_wt))
        await self._prune_registrations(context='cleanup_merge_worktree')

    async def create_throwaway_verify_worktree(self, merge_commit: str) -> Path:
        """Create an ephemeral ``_merge-<uuid>`` worktree at *merge_commit*.

        Thin public wrapper over :meth:`_create_merge_worktree` for use by
        the warm-vs-cold shadow compare (PRD §10 invariant 6(b)).  The
        returned worktree is:

        * Checked out at *merge_commit* (detached HEAD).
        * Named ``_merge-<uuid>`` — NEVER the fixed ``_merge-verify`` path.
        * Intended for a single cold verify run; callers must remove it via
          :meth:`cleanup_merge_worktree` (in a ``finally`` block) after use.

        Unlike the warm :meth:`reset_persistent_merge_worktree` path, this
        worktree has no retained ``target/`` warmth — it is a true from-scratch
        cold verify worktree (PRD §10 invariant 6(b): "cold throwaway").

        Args:
            merge_commit: The merge commit SHA to check out in the new worktree.

        Returns:
            Path to the freshly created ephemeral worktree directory.
        """
        wt, _ = await self._create_merge_worktree(base_sha=merge_commit)
        return wt

    @property
    def persistent_merge_worktree_path(self) -> Path:
        """Fixed path for the persistent warm merge-verify worktree.

        Always ``<worktree_base>/_merge-verify``.  The path is independent of
        the ``git.persistent_merge_worktree`` knob — the property always
        returns the canonical location so callers can compare against it even
        when the feature is off.
        """
        return self.worktree_base / PERSISTENT_MERGE_WORKTREE_NAME

    @property
    def persistent_offline_deep_worktree_path(self) -> Path:
        """Fixed path for the second persistent warm worktree (offline-deep lane).

        Always ``<worktree_base>/_offline-deep``.  Dedicated to the β2
        offline-deep lane worker (task 1952, PRD δ / §5 C5) — reset in place,
        retaining its own ``target/``, and NEVER sharing or overlaying the
        merge lane's ``target/`` (see :attr:`persistent_merge_worktree_path`).
        The path is independent of the ``git.persistent_offline_deep_worktree``
        knob — the property always returns the canonical location so callers
        can compare against it even when the feature is off.
        """
        return self.worktree_base / PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME

    @property
    def _merge_verify_artifact_path(self) -> Path:
        """Path of the build-artifact directory inside the persistent _merge-verify worktree.

        ``persistent_merge_worktree_path / reap_build_artifact_dirs[0]``
        (``<worktree_base>/_merge-verify/target`` by default when
        ``reap_build_artifact_dirs`` is empty).

        This is the single canonical spelling used as the *advancing* side in
        :meth:`refresh_warm_base` and as the derived default in
        :attr:`warm_lane_base_target_path`, so both stay in sync automatically.
        """
        artifact_dir = (
            self.config.reap_build_artifact_dirs[0]
            if self.config.reap_build_artifact_dirs
            else 'target'
        )
        return self.persistent_merge_worktree_path / artifact_dir

    @property
    def warm_lane_base_target_path(self) -> Path:
        """Absolute path of the warm BASE target/ to CoW-seed lane target/ from.

        Resolution order:
        1. ``config.warm_lane_base_target_dir`` when explicitly set (override).
        2. :attr:`_merge_verify_artifact_path`
           (derived default: ``<worktree_base>/_merge-verify/target``).
        """
        if self.config.warm_lane_base_target_dir is not None:
            return Path(self.config.warm_lane_base_target_dir)
        return self._merge_verify_artifact_path

    def _warm_lane_base_resolvable(self) -> WarmBaseHealth:
        """Tri-state health check of the warm-lane CoW seed base (task 2061).

        Resolves :attr:`warm_lane_base_target_path` with the SAME D8 rule as
        :meth:`_seed_warm_lane` (``base.parent / base.readlink()`` for a
        symlink base, NOT ``Path.resolve()`` — avoids tmp-prefix
        canonicalization drift) so this check's verdict matches what a real
        seed invocation would experience, then checks the concrete gen dir
        exists and is non-empty.

        Pure filesystem check (is_symlink/readlink/exists/one iterdir-next) —
        synchronous, no subprocess, no await.  Called directly from the
        pre-acquire gate in :meth:`acquire_warm_lane` and, via the Harness's
        injected async probe, once per scheduler tick (see
        ``Scheduler._apply_warm_base_hard_down_watchdog``).

        The natural clear for a host-scoped hard-down latch built on this
        probe is :meth:`refresh_warm_base` (git_ops.py ~1659) successfully
        rebuilding/advancing the base — once it does, this method observes
        the newly-populated concrete gen dir and returns ``OK`` again.

        Returns:
            WarmBaseHealth.OK — concrete gen dir exists and is non-empty.
            WarmBaseHealth.ABSENT — concrete gen dir is missing or empty.
            WarmBaseHealth.INDETERMINATE — a stat/readlink error occurred
                (e.g. non-directory base, torn read racing a concurrent
                rewrite).  Fail-safe: never treated as ABSENT by callers.

        Deliberately NOT memoized/cached (reviewer_comprehensive performance,
        task 2061 amendment pass): every call is a small, bounded number of
        syscalls (a stat/readlink, then at most one ``next(iterdir())`` that
        stops after its first entry — never a full directory scan or
        recursive walk), regardless of how large the base directory is. This
        keeps the check cheap enough to call unconditionally from the
        per-tick probe (``Scheduler._apply_warm_base_hard_down_watchdog``,
        once per scheduler tick) and from the pre-acquire gate in
        :meth:`acquire_warm_lane` (once per lane-acquire attempt) without
        throttling. Caching would also delay observing a genuine recovery by
        up to one cache TTL, undermining the "auto-clears within seconds"
        behaviour documented on the scheduler watchdog above.
        """
        try:
            base_path = self.warm_lane_base_target_path
            if base_path.is_symlink():
                # D8: resolve relative-sibling symlink (target -> .gen.N) to
                # the concrete gen dir — same join _seed_warm_lane uses.
                gen_dir = base_path.parent / base_path.readlink()
            else:
                gen_dir = base_path
            if not gen_dir.exists():
                return WarmBaseHealth.ABSENT
            try:
                next(gen_dir.iterdir())
            except StopIteration:
                return WarmBaseHealth.ABSENT
            return WarmBaseHealth.OK
        except OSError:
            logger.debug(
                '_warm_lane_base_resolvable: stat/readlink error resolving '
                'warm base — treating as INDETERMINATE (fail-safe)',
                exc_info=True,
            )
            return WarmBaseHealth.INDETERMINATE

    #: Name of the counter file used to persist the verify attempt count across
    #: stateless CLI invocations.  Scope is **per-project-worktree** — the file
    #: lives under ``worktree_base`` (``project_root / config.worktree_dir``), so
    #: a single laptop host running ``verify-merge`` for multiple projects keeps
    #: independent counters per project.  The file is never inside a registered
    #: worktree, so it is never pruned or git-cleaned.
    _VERIFY_ATTEMPT_COUNTER_FILENAME: str = '.merge_verify_host_attempts'

    def _bump_host_verify_attempt_count(self) -> int:
        """Read, increment, and persist the per-project-worktree verify attempt counter.

        The counter is stored as a plain integer in
        ``<worktree_base>/.merge_verify_host_attempts`` so that it survives
        across the stateless ``orchestrator verify-merge`` CLI invocations
        (each invocation is a fresh process; an in-memory counter cannot
        persist on the laptop host).  The counter is **per-project-worktree**:
        a single host running verify-merge for multiple projects has one
        independent counter file per project under that project's
        ``worktree_base``.

        A missing or corrupt counter file is treated as count 0 so the next
        call returns 1 — fail-safe, no exception raised.

        The non-atomic read / modify / write is safe because the per-host
        serial invariant enforced by
        :func:`~orchestrator.merge_queue.enforce_persistent_worktree_serial_lane`
        guarantees that at most one ``verify-merge`` process runs at a time on
        this host for this project.

        Returns:
            The new 1-based attempt count after the increment.
        """
        counter_file = self.worktree_base / self._VERIFY_ATTEMPT_COUNTER_FILENAME
        # Read existing count; treat missing/corrupt file as 0 (fail-safe)
        try:
            current = int(counter_file.read_text().strip())
        except (FileNotFoundError, ValueError, OSError):
            current = 0
        new_count = current + 1
        # Ensure worktree_base exists before writing
        self.worktree_base.mkdir(parents=True, exist_ok=True)
        counter_file.write_text(str(new_count))
        return new_count

    async def acquire_host_verify_worktree(self, merge_sha: str) -> Path:
        """Acquire a verify worktree for the laptop (host-side) verify-merge CLI.

        Mirrors :func:`~orchestrator.merge_queue._acquire_warm_verify_worktree`
        for the off-host CLI path (PRD §8 η / §A invariant 4).  Picks between
        the warm fixed-path worktree and a fresh ephemeral worktree based on
        the ``git.persistent_merge_worktree`` knob and the per-host safety
        valve (PRD §10 invariant 6).

        **Warm path** (knob ON, safety valve not due):
            Calls :meth:`reset_persistent_merge_worktree` which creates or
            resets-in-place the fixed ``_merge-verify`` worktree retaining
            build-artifact dirs (invariants 1+4).  Returns the fixed path.

        **Ephemeral path** (knob OFF or safety valve due):
            Calls :meth:`_create_merge_worktree` for a fresh ``_merge-<uuid>``
            worktree.  The valve fires on ``attempt % every_n == 0`` (1-based,
            every_n > 0), mirroring :func:`~orchestrator.merge_queue._safety_valve_due`
            but inlined to avoid a git_ops→merge_queue import cycle.  Returns
            the ephemeral path; ``cleanup_merge_worktree`` will remove it in
            the caller's finally block (invariant 6: cold verify, target NOT
            retained).

        Args:
            merge_sha: The merge commit SHA to check out (passed to
                :meth:`reset_persistent_merge_worktree` or
                :meth:`_create_merge_worktree` as appropriate).

        Returns:
            The worktree path to use for verification.
        """
        if not self.config.persistent_merge_worktree:
            # Knob off — ephemeral path (byte-identical to today's behavior)
            wt, _ = await self._create_merge_worktree(base_sha=merge_sha)
            return wt

        # Bump the disk-persistent counter; check the valve predicate inline
        attempt = self._bump_host_verify_attempt_count()
        every_n = self.config.persistent_merge_worktree_safety_valve_every_n
        # Inlined from merge_queue._safety_valve_due to avoid an import cycle
        # (merge_queue already imports git_ops).
        due = every_n > 0 and attempt > 0 and attempt % every_n == 0

        if due:
            # Safety-valve fired: use a fresh ephemeral worktree so that a
            # true cold verify runs without a retained target/ (invariant 6).
            wt, _ = await self._create_merge_worktree(base_sha=merge_sha)
            return wt

        # Warm path: reset the fixed worktree in place (invariants 1+4).
        return await self.reset_persistent_merge_worktree(merge_sha)

    async def reap_merge_verify_survivors(self, *, grace_secs: float = 5.0) -> bool:
        """One-time STARTUP SURVIVOR BARRIER for the warm ``_merge-verify`` lane
        (task 2828, limb 1 — the restart-collateral clobber).

        On orchestrator restart the merge_verify flock dies with the process
        and the holder-pgid lease goes stale (its pgid is dead →
        :meth:`_merge_verify_lease_active` is fail-OPEN → False).  But the
        PREVIOUS run's verify subtree (bash → cargo → rustc, ``setsid``'d /
        reparented to init) can still be ALIVE under
        :attr:`persistent_merge_worktree_path`.  The merge worker's first
        :meth:`reset_persistent_merge_worktree` then ``git reset --hard`` +
        ``git clean -xfd``s that tree, clobbering the live build out from
        under itself — BOTH existing guards are blind to such a survivor (the
        flock holder and the lease pgid both died with the parent).

        This barrier scans /proc for process groups whose cwd / an open fd /
        an mmap'd path falls at-or-under the ``_merge-verify`` subtree and
        reaps them (SIGTERM → bounded wait → SIGKILL) BEFORE the worker loop
        starts, hence before that first reset can run.  The Harness invokes it
        once from :meth:`Harness._start_merge_worker`, immediately before
        creating the merge-worker task.

        Scope + fail-open: confined to the LOCAL in-process lane
        (``git.persistent_merge_worktree`` on); best-effort reap of same-user
        orphans (SIGKILL essentially always succeeds).  Our OWN process group
        is excluded (``exclude_pgids``) so the barrier never signals itself.
        On the rare non-clearable residual it returns ``False`` and logs a
        loud ERROR (surfacing, not silencing, the hazard) rather than crashing
        startup — the caller treats the barrier as best-effort.  The blocking
        /proc scan + reap runs off the event loop via :func:`asyncio.to_thread`
        so it never stalls other coroutines.

        Returns:
            ``True`` when the lane is clear (knob off, nothing found, or every
            survivor reaped); ``False`` when a residual group is still alive
            after the reap.
        """
        if not self.config.persistent_merge_worktree:
            # Knob off — no fixed warm lane to protect (nothing to scan/reap).
            return True

        root = str(self.persistent_merge_worktree_path)
        own_pgid = os.getpgrp()

        def _scan_reap_rescan() -> tuple[
            set[int], dict[int, str], dict[int, str], set[int]
        ]:
            found = scan_process_groups_under_path(
                root, exclude_pgids=frozenset({own_pgid}),
            )
            if not found:
                return set(), {}, {}, set()
            # task 2828 amend (reviewer_comprehensive, robustness): snapshot each
            # group's member processes (pid / comm / cmdline) WHILE THEY ARE
            # STILL ALIVE — before the reap below — so the WARNING identifies
            # exactly what was killed.  The reap targets any same-user group by
            # PATH signal (cwd / fd / mmap under the lane), NOT by toolchain
            # comm, so it can also catch an operator's interactive shell or
            # debugger cwd'd under the lane mid-incident; logging the cmdline
            # lets them tell after the fact that their own session was the
            # casualty (snapshot_process_group never raises).
            snapshots = {pgid: snapshot_process_group(pgid) for pgid in found}
            outcomes = reap_process_groups(found, grace_secs=grace_secs)
            remaining = scan_process_groups_under_path(
                root, exclude_pgids=frozenset({own_pgid}),
            )
            return found, snapshots, outcomes, remaining

        found, snapshots, outcomes, remaining = await asyncio.to_thread(
            _scan_reap_rescan
        )

        if not found:
            return True

        logger.warning(
            'reap_merge_verify_survivors: found %d orphaned process group(s) '
            '%s under the warm merge-verify lane %s (a previous-run verify '
            'subtree survived restart); reap outcomes=%s. Reaped process '
            'group details (pid/comm/cmdline captured pre-reap, so an operator '
            'can identify a same-user session caught by the path-based reap):'
            '\n%s',
            len(found), sorted(found), root, outcomes,
            '\n'.join(snapshots[pgid] for pgid in sorted(snapshots)),
        )

        if remaining:
            logger.error(
                'reap_merge_verify_survivors: %d process group(s) %s STILL '
                'alive under %s after reap — the first '
                'reset_persistent_merge_worktree may clobber a live tree '
                '(fail-open: this barrier is best-effort)',
                len(remaining), sorted(remaining), root,
            )
            return False

        return True

    async def reset_persistent_merge_worktree(self, merge_commit: str) -> Path:
        """Create or reset-in-place the persistent warm merge-verify worktree.

        **Create-once path** (worktree not yet registered):
            ``git worktree add --detach <fixed_path> <merge_commit>``

        **Reset-in-place path** (worktree already registered):
            ``git reset --hard <merge_commit>`` followed by
            ``git clean -xfd -e <dir>`` for each dir in
            ``config.reap_build_artifact_dirs`` — so the source tree is
            bit-identical to a fresh checkout of *merge_commit* while
            build-artifact dirs (e.g. ``target/``) are retained (PRD §10
            invariant 1: source bit-identical to fresh checkout; build-cache
            dirs retained for warmth).

        Returns the fixed path (:attr:`persistent_merge_worktree_path`).
        Raises :exc:`RuntimeError` on git failure (mirrors
        :meth:`_create_merge_worktree`).
        Raises :exc:`MergeVerifyLeaseContended` (task 3003) on a bounded-wait
        timeout acquiring the lane lock below (fail-CLOSED — a live reify/DF
        holder must block the mutation, never be silently ignored).  The
        typed class is load-bearing, not decoration: the tree is untouched,
        so the caller is expected to DEFER the whole dispatch (requeue and
        retry later), NOT to classify it as a merge/verify failure.  A bare
        ``RuntimeError`` here used to be mapped to a deterministic-reason
        ``MergeOutcome('blocked')`` by the merge worker's generic handler,
        whose identical signature on every attempt tripped
        ``consecutive_merge_thrash`` into a false-positive human escalation.
        The raise is deliberately SITE-LOCAL to the acquire — the git
        failures inside the body below stay plain ``RuntimeError`` so a
        genuine git fault still classifies as blocked.
        Raises :exc:`LaneLockSelfOwnedLeak` (task 3081, D8/B13) — a SUBCLASS
        of the above, so every caller keeps working unchanged — when that
        same timeout is caused by a lane lock the kernel attributes to THIS
        process with no registered in-process hold and no live verify.  That
        is a leaked fd, not contention: nothing will release it before process
        exit, so deferring will never succeed.  The two were previously
        indistinguishable, which is why reify ``esc-5548-5`` took roughly
        three hours and manual ``/proc/locks`` forensics to attribute.  The
        tree is left untouched either way — this changes the DIAGNOSIS, not
        the refusal.
        Raises :exc:`MergeVerifyLeaseHeld` (task 2315, BUG 1) BEFORE
        touching the tree at all when a DIFFERENT live process holds the
        merge-verify lease — self pgid is excluded so the normal
        reset-then-verify flow (this orchestrator resetting the worktree
        it is about to verify) is unaffected.

        Holds the shared ``<lane_dir>.lock``
        (:func:`~orchestrator.verify_cancel.lane_lock_path` of *warm_path*)
        across the tree-mutating body below (task 2685, step-4) — the SAME
        lock reify's ``seed-warm-lane.sh`` / ``thin-warm-lane.sh`` /
        ``warm-lane-gc.sh``, DF's own :meth:`_seed_warm_lane`, and
        :meth:`merge_verify_lease` all take. This closes the residual
        window between a reset returning and the caller's later
        :meth:`merge_verify_lease` acquiring its own (separate,
        sequential) hold on the same lock — reset and verify now each
        serialize against reify/DF actors on the one lock, eliminating the
        gap a reseed could previously race through. The bounded-wait
        acquire runs off the event loop (:func:`asyncio.to_thread`) so a
        contended wait never stalls other in-process coroutines.

        Cancellation caveat: :func:`asyncio.to_thread` cannot stop the
        worker thread mid-wait — if this coroutine is cancelled while the
        thread is still polling for the flock, the ``await`` raises
        ``CancelledError`` immediately, but the thread keeps running to
        completion. If it goes on to acquire the flock, the returned fd is
        never seen by this (already-cancelled) coroutine, so
        :func:`release_merge_verify_flock` never runs for it and the lane
        lock stays held until process exit. This window requires
        cancellation to race the acquire and is bounded by
        ``_RESET_WARM_LANE_LOCK_WAIT_SECS``, so it is treated as an accepted,
        documented edge case here rather than guarded — a shielded-cleanup
        fix would add async-ownership complexity out of proportion to this
        task's scope.
        """
        warm_path = self.persistent_merge_worktree_path

        holder_pgid = read_lock_holder_pgid(self.worktree_base)
        if self._merge_verify_lease_active() and holder_pgid != os.getpgrp():
            raise MergeVerifyLeaseHeld(warm_path, holder_pgid)

        lock_path = lane_lock_path(warm_path)
        # See the cancellation caveat in this method's docstring: cancelling
        # this await cannot stop the to_thread worker, so a fd acquired
        # after cancellation has already propagated is discarded unreleased.
        fd = await asyncio.to_thread(
            acquire_merge_verify_flock, lock_path, _RESET_WARM_LANE_LOCK_WAIT_SECS,
        )
        if fd is None:
            # Is this OUR OWN leaked lock rather than a live foreign hold?
            # Asked FIRST (task 3081): the incident's symptom was exactly this
            # timeout, and the leak is invisible unless something asks.  It
            # only ever REPORTS — the fail-CLOSED refusal below is unchanged
            # either way, so the tree is untouched in both cases.
            leak = self._lane_lock_self_owned_leak(
                lock_path,
                _RESET_WARM_LANE_LOCK_WAIT_SECS,
                operation='the warm merge-worktree reset',
                protected_path=warm_path,
            )
            if leak is not None:
                raise leak
            # SITE-LOCAL raise (task 3003): scoped to the lock ACQUIRE alone.
            # A live reify/DF actor still holds the lane lock, so the tree is
            # left completely untouched (fail-CLOSED) and the caller must
            # DEFER — this is a transient "come back later," not a merge
            # failure.  The git faults inside the try body below deliberately
            # keep raising plain RuntimeError so a genuine git fault still
            # classifies as 'blocked' rather than looping on a defer.
            raise MergeVerifyLeaseContended(
                lock_path,
                _RESET_WARM_LANE_LOCK_WAIT_SECS,
                operation='the warm merge-worktree reset',
                # Restores the one piece of context the replaced bare
                # RuntimeError carried ('refusing to mutate {warm_path}
                # unprotected'): WHICH tree this refusal is protecting.
                protected_path=warm_path,
                holder_facts=_lane_lock_holder_facts(lock_path),
            )
        try:
            if not await self._is_registered_worktree(warm_path):
                # Create-once branch — self-heal a stale unregistered directory first.
                # A previous run may have left the directory on disk without a git
                # worktree registration (e.g. worktree metadata pruned after a crash).
                # `git worktree add` refuses a non-empty directory, permanently
                # wedging the warm path until manual cleanup.  Removing the orphaned
                # directory here mirrors the stale-directory removal in create_worktree
                # and makes the create-once path self-healing.
                if warm_path.exists():
                    logger.warning(
                        'Persistent merge worktree path %s exists on disk but is not '
                        'a registered git worktree; removing stale directory to allow '
                        'fresh creation (self-heal)',
                        warm_path,
                    )
                    shutil.rmtree(warm_path)
                warm_path.parent.mkdir(parents=True, exist_ok=True)
                rc, _, err = await _run(
                    ['git', 'worktree', 'add', '--detach', str(warm_path), merge_commit],
                    cwd=self.project_root,
                )
                if rc != 0:
                    raise RuntimeError(
                        f'Failed to create persistent merge worktree at {warm_path}: {err}'
                    )
                logger.info(
                    'Created persistent merge worktree at %s (HEAD=%s)',
                    warm_path, merge_commit[:8],
                )
            else:
                # Reset-in-place branch (added in step-6)
                rc, _, err = await _run(
                    ['git', 'reset', '--hard', merge_commit],
                    cwd=warm_path,
                )
                if rc != 0:
                    raise RuntimeError(
                        f'Failed to reset persistent merge worktree {warm_path} '
                        f'to {merge_commit}: {err}'
                    )
                ok, err = await self._clean_lane_retaining_artifacts(
                    warm_path, caller='reset_persistent_merge_worktree',
                )
                if not ok:
                    raise RuntimeError(
                        f'Failed to clean persistent merge worktree {warm_path}: {err}'
                    )
                logger.info(
                    'Reset persistent merge worktree %s to HEAD=%s',
                    warm_path, merge_commit[:8],
                )
        finally:
            release_merge_verify_flock(fd)

        return warm_path

    async def reset_persistent_offline_deep_worktree(self, merge_commit: str) -> Path:
        """Create or reset-in-place the second persistent warm worktree (offline-deep lane).

        Dedicated to the offline-deep lane worker (β2, task 1952, PRD δ /
        §5 C5) — modeled on :meth:`reset_persistent_merge_worktree` but at
        its own fixed path (:attr:`persistent_offline_deep_worktree_path`)
        with its own retained ``target/``, NEVER sharing or overlaying the
        merge lane's ``target/`` (C5).

        **Create-once path** (worktree not yet registered):
            ``git worktree add --detach <fixed_path> <merge_commit>``

        **Reset-in-place path** (worktree already registered):
            ``git reset --hard <merge_commit>`` followed by
            ``git clean -xfd -e <dir>`` for each dir in
            ``config.reap_build_artifact_dirs`` — so the source tree is
            bit-identical to a fresh checkout of *merge_commit* while
            build-artifact dirs (e.g. ``target/``) are retained — this
            worktree's OWN warmth, self-warming across resets.

        Returns the fixed path (:attr:`persistent_offline_deep_worktree_path`).
        Raises :exc:`RuntimeError` on git failure (mirrors
        :meth:`reset_persistent_merge_worktree`).
        """
        warm_path = self.persistent_offline_deep_worktree_path

        if not await self._is_registered_worktree(warm_path):
            # Create-once branch — self-heal a stale unregistered directory first.
            # See reset_persistent_merge_worktree for rationale (a previous run
            # may have left the directory on disk without a git worktree
            # registration, e.g. worktree metadata pruned after a crash).
            if warm_path.exists():
                logger.warning(
                    'Persistent offline-deep worktree path %s exists on disk but '
                    'is not a registered git worktree; removing stale directory '
                    'to allow fresh creation (self-heal)',
                    warm_path,
                )
                shutil.rmtree(warm_path)
            warm_path.parent.mkdir(parents=True, exist_ok=True)
            rc, _, err = await _run(
                ['git', 'worktree', 'add', '--detach', str(warm_path), merge_commit],
                cwd=self.project_root,
            )
            if rc != 0:
                raise RuntimeError(
                    f'Failed to create persistent offline-deep worktree at {warm_path}: {err}'
                )
            logger.info(
                'Created persistent offline-deep worktree at %s (HEAD=%s)',
                warm_path, merge_commit[:8],
            )
        else:
            # Reset-in-place branch — retains this worktree's OWN target/,
            # never touching or seeding from the merge lane's target/ (C5).
            rc, _, err = await _run(
                ['git', 'reset', '--hard', merge_commit],
                cwd=warm_path,
            )
            if rc != 0:
                raise RuntimeError(
                    f'Failed to reset persistent offline-deep worktree {warm_path} '
                    f'to {merge_commit}: {err}'
                )
            ok, err = await self._clean_lane_retaining_artifacts(
                warm_path, caller='reset_persistent_offline_deep_worktree',
            )
            if not ok:
                raise RuntimeError(
                    f'Failed to clean persistent offline-deep worktree {warm_path}: {err}'
                )
            logger.info(
                'Reset persistent offline-deep worktree %s to HEAD=%s',
                warm_path, merge_commit[:8],
            )

        return warm_path

    async def _iter_merge_worktrees(self):
        """Yield ``(wt_path, wt_resolved)`` pairs for registered ``_merge-*`` worktrees.

        Private async-generator helper shared by :meth:`prune_stale_merge_worktrees`
        and :meth:`find_inflight_merge_worktree`.  Enumerates via
        ``git worktree list --porcelain``, filtering to direct children of
        ``worktree_base`` whose name starts with ``_merge-``.  Yields nothing
        on git error (fail-closed).

        *wt_path* is the raw path from porcelain output (used for git commands).
        *wt_resolved* is the resolved path (used for identity comparisons such
        as the ``keep`` exclusion in :meth:`prune_stale_merge_worktrees`).

        **Persistent-worktree exemption**: the fixed
        :data:`PERSISTENT_MERGE_WORKTREE_NAME` (``_merge-verify``) is always
        skipped so that both :meth:`prune_stale_merge_worktrees` (PRD §10
        invariant 4) and :meth:`find_inflight_merge_worktree` never touch or
        return the warm worktree.
        """
        rc, out, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=self.project_root,
        )
        if rc != 0:
            return

        for line in out.splitlines():
            if not line.startswith('worktree '):
                continue
            wt_path = Path(line[len('worktree '):].strip())
            try:
                wt_resolved = wt_path.resolve()
            except OSError:
                wt_resolved = wt_path
            if wt_resolved.parent != self.worktree_base:
                continue
            if not wt_resolved.name.startswith('_merge-'):
                continue
            # Exempt the persistent warm merge-verify worktree — prune and
            # find_inflight must never touch it (invariant 4).
            if wt_resolved.name == PERSISTENT_MERGE_WORKTREE_NAME:
                continue
            yield wt_path, wt_resolved

    async def prune_stale_merge_worktrees(
        self, keep: Path | Collection[Path] | None = None,
    ) -> list[str]:
        """Force-remove leftover ``_merge-*`` worktrees; return paths removed.

        Disk-pressure recovery helper.  A crashed or abandoned merge can leave
        ``_merge-<id>`` worktrees behind under ``worktree_base``, each holding
        a full checkout — dead weight that contributes to ENOSPC.  This
        force-removes every such *registered* worktree EXCEPT *keep* (the merge
        worktree currently in use), then runs ``git worktree prune`` to clear
        stale admin entries.

        *keep* may be a single :class:`~pathlib.Path`, a collection (set, list,
        …) of paths, or ``None`` (remove all).  Every path in the keep-set is
        resolved before comparison so symlinks and relative paths are handled
        correctly.

        NEVER touches task worktrees (``worktree_base/<task_id>``) — those hold
        live builds.  Only paths that are direct children of ``worktree_base``
        AND whose name starts with ``_merge-`` are eligible, so a task whose id
        happens to start with ``_merge`` cannot be caught (task ids are not
        prefixed that way).  Enumerates via :meth:`_iter_merge_worktrees`
        (``git worktree list --porcelain``), so a half-created directory git
        doesn't track is never removed.

        The persistent warm merge-verify worktree
        (:data:`PERSISTENT_MERGE_WORKTREE_NAME`) is always exempted via
        :meth:`_iter_merge_worktrees` — it is never removed by prune
        (PRD §10 invariant 4).
        """
        removed: list[str] = []
        if isinstance(keep, (str, bytes)):
            raise TypeError(
                f'prune_stale_merge_worktrees: keep must be a Path, '
                f'Collection[Path], or None — got {type(keep).__name__!r}; '
                f'wrap in Path(...) if a string path was intended'
            )
        if keep is None:
            keep_resolved: set[Path] = set()
        elif isinstance(keep, Path):
            keep_resolved = {keep.resolve()}
        else:
            keep_resolved = {p.resolve() for p in keep}

        async for wt_path, wt_resolved in self._iter_merge_worktrees():
            if wt_resolved in keep_resolved:
                continue
            if self._refuse_foreign_band(
                wt_path, frozenset({'_merge-'}), 'prune_stale_merge_worktrees',
            ):
                continue
            rc_rm, _, err = await _run(
                ['git', 'worktree', 'remove', '--force', str(wt_path)],
                cwd=self.project_root,
            )
            if rc_rm == 0:
                removed.append(str(wt_path))
            else:
                logger.warning(
                    'prune_stale_merge_worktrees: failed to remove %s: %s',
                    wt_path, err.strip(),
                )

        if removed:
            await self._prune_registrations(context='prune_stale_merge_worktrees')
            logger.info(
                'prune_stale_merge_worktrees: removed %d stale merge '
                'worktree(s)', len(removed),
            )
        return removed

    async def find_inflight_merge_worktree(self, branch: str) -> Path | None:
        """Find an on-disk ``_merge-*`` worktree whose HEAD matches *branch*.

        Uses :meth:`_iter_merge_worktrees` to enumerate candidates (direct
        children of ``worktree_base`` whose name starts with ``_merge-``).
        For each candidate, reads its HEAD commit subject with
        ``git log -1 --format=%s`` and compares it by **literal equality** to
        ``_merge_subject(f'{branch_prefix}{branch}', main_branch)``.

        Returns the first matching :class:`~pathlib.Path`, or ``None`` if no
        match is found.

        Fail-closed on git errors: a candidate whose ``git log`` fails is
        skipped (logged at WARNING level) rather than raising — avoids
        crashing the coalesce dispatch on a partially-written worktree.

        Crash-safety / cross-restart source of truth: even if the in-memory
        ``InFlightMergeRegistry`` was cleared by a process restart, an
        in-progress merger's ``_merge-*`` worktree persists on disk and is
        correctly detected here.
        """
        full_branch = (
            await self.resolve_queued_branch_ref(branch)
            or f'{self.config.branch_prefix}{branch}'
        )
        target_subject = _merge_subject(
            full_branch,
            self.config.main_branch,
        )

        async for wt_path, _ in self._iter_merge_worktrees():
            # Read HEAD commit subject of this candidate
            rc_log, subject, err_log = await _run(
                ['git', 'log', '-1', '--format=%s'],
                cwd=wt_path,
            )
            if rc_log != 0:
                logger.warning(
                    'find_inflight_merge_worktree: git log failed for %s: %s',
                    wt_path, err_log.strip(),
                )
                continue
            if subject.strip() == target_subject:
                return wt_path

        return None

    # ── δ: interactive-worktree (_iact-*) crash-safety reaper — task 2012 ──
    #
    # WarmLanePool._recover_crashed_tasks covers _lane-* only; the _iact-*
    # band task α (2010) mints via create_interactive_worktree is
    # structurally disjoint from that pool (isolation invariant I1) and had
    # no crash-cleanup path.  reap_interactive_worktrees supplies it: a
    # crashed/idle/landed interactive worktree is force-removed the same way
    # prune_stale_merge_worktrees reclaims _merge-* worktrees.

    async def _iter_interactive_worktrees(self):
        """Yield ``(wt_path, wt_resolved)`` pairs for registered ``_iact-*`` worktrees.

        Mirrors :meth:`_iter_merge_worktrees` (same porcelain-enumeration and
        direct-child-of-``worktree_base`` filter) but matches
        ``config.iact_prefix`` instead of the hardcoded ``'_merge-'`` prefix,
        and has no persistent-worktree exemption — there is no ``_iact-*``
        equivalent of the always-on ``_merge-verify`` worktree.  Yields
        nothing on git error (fail-closed, matching ``_iter_merge_worktrees``).
        """
        rc, out, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=self.project_root,
        )
        if rc != 0:
            return

        for line in out.splitlines():
            if not line.startswith('worktree '):
                continue
            wt_path = Path(line[len('worktree '):].strip())
            try:
                wt_resolved = wt_path.resolve()
            except OSError:
                wt_resolved = wt_path
            if wt_resolved.parent != self.worktree_base:
                continue
            if not wt_resolved.name.startswith(self.config.iact_prefix):
                continue
            yield wt_path, wt_resolved

    async def _interactive_worktree_landed(self, full_branch: str) -> bool:
        """True if a ``Merge {full_branch} into {main_branch}`` marker exists on main.

        Reproduces :func:`find_merge_marker`'s grep core (``git log
        <main_branch> --fixed-strings --grep=<subject> --max-count=1
        --format=%H``) but deliberately WITHOUT its branch-existence gate:
        :meth:`find_merge_marker` returns ``None`` immediately whenever the
        branch ref still resolves, on the assumption that a live branch means
        ``is_ancestor`` is the right check — but an ``_iact-*`` branch is
        *always* still checked out in its own worktree at reap time, so that
        gate would short-circuit to ``False`` here every single time.

        Deliberately NOT ``is_ancestor(HEAD, main)``: a freshly-created
        ``_iact-*`` worktree has zero commits of its own, so its HEAD trivially
        *is* an ancestor of (equal to) main at creation time — using
        ``is_ancestor`` here would misclassify every brand-new interactive
        session as "landed" and reap it immediately. The merge-marker grep
        only matches a REAL merge commit, so it cannot false-positive on that
        shape (see :meth:`worktree_head_beyond_main`'s docstring and
        ``find_task_citation_commit`` for the same is_ancestor pitfall
        elsewhere in this module).
        """
        grep_pattern = _merge_subject(full_branch, self.config.main_branch)
        rc, out, _ = await _run(
            [
                'git', 'log', self.config.main_branch,
                '--fixed-strings',
                f'--grep={grep_pattern}',
                '--max-count=1',
                '--format=%H',
            ],
            cwd=self.project_root,
        )
        return rc == 0 and bool(out.strip())

    async def _worktree_dirty(self, worktree: Path) -> bool:
        """True if *worktree* has uncommitted changes (``git status --porcelain``).

        Used by :meth:`reap_interactive_worktrees` to avoid reaping a worktree
        out from under an actively-editing user in two situations: a landed
        (merged-to-main) worktree the user resumed working in without
        committing yet, and a live (unmerged-commits) worktree whose newest
        commit is old but whose working tree shows fresh, uncommitted edits.

        **Fail-safe True** on any git error (unreadable worktree, etc.) —
        callers treat "dirty" as "leave it alone this sweep", so an I/O
        hiccup defers a reap decision rather than risking one on a false
        "clean" reading. This mirrors :meth:`_branch_has_commits_beyond_main`'s
        fail-safe-toward-retention convention.
        """
        rc, out, _ = await _run(['git', 'status', '--porcelain'], cwd=worktree)
        if rc != 0:
            return True
        return bool(out.strip())

    async def _worktree_head_readable(self, worktree: Path) -> bool:
        """True if ``git rev-parse HEAD`` succeeds in *worktree*.

        :meth:`worktree_head_beyond_main` returns ``None`` both when a
        worktree genuinely has no commits beyond main AND when its internal
        ``git rev-parse HEAD`` call fails (rc != 0) — a transient git error
        collapses to the exact same signal as "safe, no unmerged work".
        :meth:`reap_interactive_worktrees` uses this helper to tell those two
        cases apart *before* trusting a ``None`` result: a worktree that
        actually carries unmerged commits must never be force-removed just
        because HEAD happened to be unreadable for one sweep.
        """
        rc, _, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        return rc == 0

    async def reap_interactive_worktrees(
        self, *, now: datetime | None = None,
    ) -> list[ReapedInteractiveWorktree]:
        """Crash-safety sweep over the ``_iact-*`` interactive-worktree band (task δ/2012).

        Enumerates registered ``_iact-*`` worktrees exactly like
        :meth:`_iter_interactive_worktrees` (direct children of
        ``worktree_base`` matching ``config.iact_prefix``, per ``git worktree
        list --porcelain``).  For each candidate, ``slug``/``branch`` are
        derived from the directory name by the same convention
        :meth:`create_interactive_worktree` used to create it
        (``worktree_base/{iact_prefix}{slug}`` on branch
        ``{branch_prefix}{slug}``) — both are always derivable this way, so
        this does not depend on the ``interactive.json`` stamp being intact.

        Per-worktree reap predicate:

        * :meth:`worktree_head_beyond_main` returns ``None`` both when a
          worktree genuinely has no commits beyond main AND when its
          internal ``git rev-parse HEAD`` call fails (transient git error) —
          the two cases are indistinguishable from that return value alone.
          Before trusting a ``None`` result as "no unmerged work",
          :meth:`_worktree_head_readable` independently re-confirms HEAD is
          actually readable; if it is not, this candidate is skipped for the
          sweep (retention) instead of risking a false "safe to reap"
          classification that could force-remove genuinely unmerged commits.
        * :meth:`worktree_head_beyond_main` is ``None`` (no commits beyond
          main) AND :meth:`_interactive_worktree_landed` finds a ``Merge
          <branch> into <main_branch>`` marker on main AND the working tree
          is clean (:meth:`_worktree_dirty` is ``False``) → reaped
          immediately as ``'landed'``, regardless of age (the work is
          already safe on main and nothing remains uncommitted). A landed
          worktree that still carries commits beyond main, or uncommitted
          edits, is NOT reaped immediately — it falls through to the
          idle/TTL handling below instead, so a session the user resumed
          *after* the merge (committed or not) isn't reclaimed out from
          under them.
        * Otherwise, when :meth:`worktree_head_beyond_main` is ``None`` (no
          commits beyond main) → age is measured from the
          ``interactive.json`` stamp's ``created_at``, resolved new-then-old
          (the ``.task-meta/<name>/interactive.json`` path first, falling
          back to the legacy ``<wt>/.task/interactive.json`` path — W11
          gamma relocation).  Reaped (``'ttl_idle'``) when that age exceeds
          ``config.interactive_worktree_ttl``; otherwise, when
          ``_run_warm_lane_disk_guard()`` reports disk pressure (rc==75),
          reaped anyway (``'disk_pressure'``) regardless of age — safe
          because there is no unmerged work to lose.  The disk-guard check
          is deferred until the first candidate that actually reaches this
          branch (idle, within TTL, not landed) and its result is reused for
          the rest of the sweep — so it still runs at most once per sweep,
          but a sweep with no such candidate never invokes it at all.
        * Otherwise (the worktree carries unmerged commits) → a dirty
          working tree (:meth:`_worktree_dirty`) is treated as recent
          activity and preserves the worktree outright, since an in-progress
          session can be actively editing for hours between commits.
          Otherwise, age is measured from the newest commit's time (``git
          show -s --format=%ct HEAD``), NOT the stamp — an in-progress
          session that keeps committing must never be reaped merely because
          it is old.  Reaped (``'ttl_idle'``) when that age exceeds the same
          TTL. Note this means uncommitted edits made *after* the TTL window
          has already lapsed on a stale worktree are not specially
          protected — only edits present at sweep time are. A worktree with
          unmerged commits is NEVER reaped for disk pressure alone — only
          the TTL (or landed) rule can reclaim it.

        Removal uses ``git worktree remove --force`` per reaped candidate,
        followed by a single ``git worktree prune`` if at least one was
        removed — mirrors :meth:`prune_stale_merge_worktrees`.

        Args:
            now: reference time for TTL comparisons; defaults to
                ``datetime.now(UTC)``. Injectable so callers/tests get
                deterministic TTL boundaries.

        Returns:
            One :class:`ReapedInteractiveWorktree` per worktree actually
            removed. Never raises.
        """
        if now is None:
            now = datetime.now(UTC)
        ttl = self.config.interactive_worktree_ttl

        reaped: list[ReapedInteractiveWorktree] = []
        # `None` = not checked yet this sweep. Computed lazily on first
        # actual need (see the disk-pressure branch below) and reused for
        # every remaining candidate, so it still runs at most once per
        # sweep — but a sweep with zero _iact-* worktrees, or none that are
        # idle-and-within-TTL, never pays for the subprocess call at all.
        disk_pressure: bool | None = None
        try:
            async for wt_path, wt_resolved in self._iter_interactive_worktrees():
                slug = wt_resolved.name[len(self.config.iact_prefix):]
                full_branch = f'{self.config.branch_prefix}{slug}'

                reason: str | None = None
                beyond = await self.worktree_head_beyond_main(wt_path)
                # worktree_head_beyond_main returns None both when there are
                # genuinely no commits beyond main AND when its internal
                # `git rev-parse HEAD` failed (transient git error) — those
                # two cases are indistinguishable from the return value
                # alone. Independently confirm HEAD is actually readable
                # before trusting None as "no unmerged work"; a rev-parse
                # failure defers this candidate (retention) rather than
                # risking a false "safe to reap" classification that could
                # force-remove genuinely unmerged commits.
                if beyond is None and not await self._worktree_head_readable(wt_path):
                    logger.warning(
                        'reap_interactive_worktrees: could not read HEAD '
                        'for %s this sweep — deferring reap decision',
                        wt_path,
                    )
                    continue
                # Only a worktree with no commits beyond main can possibly be
                # "landed" (a real merge marker); short-circuiting here also
                # skips the merge-marker grep entirely once beyond is known
                # not-None, since that case always falls to the live-commit
                # branch below regardless of landed status.
                landed = beyond is None and await self._interactive_worktree_landed(
                    full_branch,
                )
                if landed and not await self._worktree_dirty(wt_path):
                    # Landed on main and the working tree is clean — safe to
                    # reap regardless of age (the PROMPT-safe case: the work
                    # already lives on main and nothing remains uncommitted).
                    reason = 'landed'
                elif beyond is None:
                    # Either not landed, or landed-but-dirty (resumed
                    # post-merge editing with no new commit yet) — either way
                    # there are no commits beyond main, so age on the stamp
                    # exactly like any other idle candidate.
                    stamp_path = TaskArtifacts.meta_root_for(
                        self.worktree_base, wt_resolved.name,
                    ) / 'interactive.json'
                    if not stamp_path.exists():
                        stamp_path = wt_path / '.task' / 'interactive.json'
                    try:
                        stamp = json.loads(stamp_path.read_text())
                        created_at = datetime.fromisoformat(stamp['created_at'])
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        # Missing/corrupt stamp and no unmerged work to
                        # lose — fail-soft towards reaping (I2: never an
                        # indefinite leak) rather than preserving forever.
                        # A worktree WITH unmerged commits never reaches
                        # this branch at all (beyond is not None below),
                        # so a corrupt stamp can never silently drop work.
                        logger.warning(
                            'reap_interactive_worktrees: unreadable stamp '
                            'for %s (%s: %s) — reaping as stale (no '
                            'unmerged work to lose)',
                            wt_path, type(exc).__name__, exc,
                        )
                        reason = 'stale_no_stamp'
                    else:
                        if (now - created_at).total_seconds() > ttl:
                            reason = 'ttl_idle'
                        else:
                            if disk_pressure is None:
                                disk_pressure = (
                                    await self._run_warm_lane_disk_guard()
                                ) == 75
                            if disk_pressure:
                                # No unmerged work to lose — safe to evict
                                # under pressure even though still within
                                # TTL.
                                reason = 'disk_pressure'
                elif not await self._worktree_dirty(wt_path):
                    # Unmerged commits and a clean tree — age on the
                    # newest commit. A dirty tree here (the `if` this
                    # `elif` pairs with) is treated as recent activity
                    # and preserves the worktree outright, since a
                    # session can be actively editing for hours between
                    # commits.
                    rc_ct, ct_out, _ = await _run(
                        ['git', 'show', '-s', '--format=%ct', 'HEAD'],
                        cwd=wt_path,
                    )
                    if rc_ct == 0 and ct_out.strip():
                        commit_time = datetime.fromtimestamp(
                            int(ct_out.strip()), tz=UTC,
                        )
                        if (now - commit_time).total_seconds() > ttl:
                            reason = 'ttl_idle'

                if reason is None:
                    continue

                if self._refuse_foreign_band(
                    wt_path, frozenset({self.config.iact_prefix}),
                    'reap_interactive_worktrees',
                ):
                    continue

                rc_rm, _, err = await _run(
                    ['git', 'worktree', 'remove', '--force', str(wt_path)],
                    cwd=self.project_root,
                )
                if rc_rm == 0:
                    # The interactive.json stamp now lives in the .task-meta
                    # sibling dir (S10), OUTSIDE the worktree, so `git
                    # worktree remove` above no longer cleans it up
                    # incidentally the way it did when the stamp lived inside
                    # the worktree. Remove it here, best-effort, so a reaped
                    # worktree never leaves an orphaned .task-meta/<name> dir
                    # behind (I2: never an indefinite leak).
                    shutil.rmtree(
                        TaskArtifacts.meta_root_for(self.worktree_base, wt_resolved.name),
                        ignore_errors=True,
                    )
                    reaped.append(ReapedInteractiveWorktree(
                        path=wt_path, branch=full_branch, slug=slug, reason=reason,
                    ))
                else:
                    logger.warning(
                        'reap_interactive_worktrees: failed to remove %s: %s',
                        wt_path, err.strip(),
                    )

            if reaped:
                await self._prune_registrations(context='reap_interactive_worktrees')
        except Exception:
            logger.warning(
                'reap_interactive_worktrees: unexpected error during sweep '
                '(returning %d worktree(s) reaped before the fault)',
                len(reaped), exc_info=True,
            )

        return reaped

    # ── PHASE 4: Speculative merge-verify pipeline ────────────────────
    #
    # Once the merge queue (task 292) is stable and we have metrics on
    # queue depth and cycle time, consider a 2-step speculative pipeline:
    #
    #   Worker A (merger):   dequeue → merge_wt → git merge → scrub
    #   Worker B (verifier): verify → CAS update-ref → notify
    #
    # While B verifies merge N, A speculatively merges N+1 against N's
    # merge SHA (not current main).  If N succeeds, N+1 is already a
    # descendant — CAS works immediately.  If N fails, discard N+1 and
    # re-merge against actual main.  Cap speculation depth at 1.
    #
    # Expected throughput gain: ~2-3x when queue depth >3, because
    # verification (~15-25s) dominates cycle time and is fully overlapped.
    #
    # Key risk: verification validity.  N+1 is verified against a tree
    # that includes N's changes.  If N is later rejected, N+1 passed
    # verification against a state that never existed on main.  Mitigated
    # by scoped verification (task_files only) and depth-1 cap.
    #
    # Unblock condition: merge queue metrics showing sustained queue
    # depth >3 and merge cycle time dominating task throughput.
    # See blocked task that depends on task 292.
    # ─────────────────────────────────────────────────────────────────

    async def advance_main(
        self,
        merge_sha: str,
        merge_worktree: Path | None = None,
        branch: str | None = None,
        max_attempts: int = 3,
        expected_main: str | None = None,
        reverify_on_rebase: bool = False,
    ) -> AdvanceOutcome:
        """Advance main branch ref to *merge_sha* atomically.

        Uses ``update-ref`` to advance the ref, then syncs the working tree
        via ``read-tree`` when project_root has main checked out.  Uncommitted
        changes are stashed before the advance and popped after, so user work
        survives and merge conflicts become visible markers rather than silent
        reverts (see incident ``0ea23cb5c``).

        Returns an :class:`AdvanceOutcome` value object whose ``result`` field
        is an :data:`AdvanceResult` literal:

        * ``'advanced'`` — success.
        * ``'cas_failed'`` — CAS ``update-ref`` failed (transient; caller
          can re-enqueue).
        * ``'not_descendant'`` — merge commit couldn't become a descendant
          of main after *max_attempts* (permanent; stop retrying).
        * ``'contaminated'`` — retained in :data:`AdvanceResult` /
          :class:`AdvanceOutcome` for typing and merge_queue.py
          compatibility only; this method no longer produces it. The
          ``.task/`` contamination gate (``_assert_no_task_dir``) that used
          to return this outcome was retired in W11 ι: contamination
          prevention now rests entirely on structural relocation of task
          metadata to ``.task-meta`` (for the orchestrator hot path) plus
          this repo's root ``.gitignore`` ``.task/`` entry — an intentional
          defense-in-depth trade-off, not an oversight (see module
          docstring).
        * ``'conflict_markers'`` — the merge tree contains tracked file(s)
          with unresolved (column-0) conflict markers (permanent; stop
          retrying).  esc-2128-8 Layer-2 backstop — see
          :func:`_assert_no_conflict_markers`.
        * ``'stash_failed'`` — parking pre-advance WIP onto the private
          ``MERGE_PARK_REF`` failed before the advance, either because the
          ``git stash create`` / ``update-ref`` infra sequence itself failed
          (:class:`MergeParkError`) or because ``MERGE_PARK_REF`` already
          existed — a stale or contended ref that is never overwritten
          (:class:`MergeParkContentionError`).  Permanent; halt merge to
          prevent code loss.  See :meth:`GitOps._park_wip_on_private_ref`.
        * ``'pop_conflict_no_advance'`` — CAS ``update-ref`` failed AND the
          subsequent stash pop conflicted.  The merge did NOT land.  WIP is
          preserved on a ``wip/recovery-*`` branch; routes to a human-level
          escalation.
        * ``'unmerged_state'`` — ``project_root`` already has unresolved merge
          conflicts in its index (UU/AA/DD paths detected via
          ``git status --porcelain``).  Halts immediately; manual cleanup of
          the conflict markers is required before retrying.  Routes to a
          human-level escalation, not the steward corrective loop.

        When *branch* is provided and a rebase fails, the method will abort
        the rebase, reset to current main, and re-merge *branch* before
        retrying.  Up to *max_attempts* rounds are attempted.

        When *expected_main* is provided, the final ``update-ref`` uses a
        compare-and-swap: ``git update-ref refs/heads/main <new> <old>``.
        If main has moved (external actor), update-ref fails atomically
        and this method returns an outcome with ``result == 'cas_failed'``.

        IMPORTANT: This method is the LAST checkpoint before code reaches
        main.  update-ref bypasses most git hooks (including pre-commit),
        so the conflict-marker gate here is the final defense.
        Exception: git's ``reference-transaction`` hook (git>=2.28) DOES
        fire on update-ref — advance_main's main_gate mark (task 1678)
        sanctions that hook by writing a sentinel immediately before the
        CAS so reify-style projects record the move as SANCTIONED rather
        than UNSANCTIONED.  See also task 7 for the same stale assumption.

        On a successful ``'advanced'`` outcome, ``outcome.advanced_sha`` holds
        the SHA actually placed on main.  When CAS retry rebases the merge
        commit, the post-rebase SHA is captured here — callers must read
        this field for done_provenance instead of the pre-rebase
        ``MergeResult.merge_commit`` (which is stale after a rebase).  On a
        ``'rebased_pending_reverify'`` outcome, ``outcome.rebased_from`` and
        ``outcome.rebased_onto`` are also populated (original base / moved
        main respectively).
        """
        full_branch = (
            (await self.resolve_queued_branch_ref(branch) or f'{self.config.branch_prefix}{branch}')
            if branch else None
        )
        rebased = False  # Track whether any rebase/re-merge occurred this call

        # Derive the verified branch tip from M^2 — the exact branch commit
        # that merge_to_main incorporated (--no-ff guarantees M^2 is the
        # branch commit verify ran against).  Captured ONCE here, before the
        # CAS loop, so the re-merge fallback can pin to this SHA rather than
        # re-resolving the moving full_branch ref.
        verified_branch_tip: str | None = None
        _vbt_rc, _vbt_sha, _ = await _run(
            ['git', 'rev-parse', f'{merge_sha}^2'],
            cwd=self.project_root,
        )
        if _vbt_rc == 0 and _vbt_sha.strip():
            verified_branch_tip = _vbt_sha.strip()

        for attempt in range(max_attempts):
            # ── conflict-marker gate (FINAL DEFENSE, esc-2128-8 Layer-2) ──
            try:
                await _assert_no_conflict_markers(
                    merge_sha, self.project_root,
                    f'advance_main(attempt={attempt + 1})',
                )
            except RuntimeError as e:
                logger.error(str(e))
                return AdvanceOutcome('conflict_markers')

            rc, _, _ = await _run(
                ['git', 'merge-base', '--is-ancestor',
                 self.config.main_branch, merge_sha],
                cwd=self.project_root,
            )
            if rc == 0:
                break  # merge_sha is a descendant of main — safe to advance

            if merge_worktree is None:
                logger.warning(
                    f'Cannot fast-forward: {merge_sha[:8]} is not a descendant '
                    f'of {self.config.main_branch} (no merge worktree for retry)'
                )
                return AdvanceOutcome('not_descendant')

            logger.info(
                f'advance_main attempt {attempt + 1}/{max_attempts}: '
                f'main advanced past {merge_sha[:8]}'
            )

            # Try rebasing the merge commit onto current main
            rebase_rc, _, rebase_err = await _run(
                ['git', 'rebase', self.config.main_branch],
                cwd=merge_worktree,
            )
            if rebase_rc == 0:
                _, new_sha, _ = await _run(
                    ['git', 'rev-parse', 'HEAD'], cwd=merge_worktree,
                )
                merge_sha = new_sha.strip()
                rebased = True
                continue  # re-check is_ancestor at top of loop

            # Rebase failed — abort and try a fresh re-merge if we have
            # the branch name
            logger.warning(
                f'Rebase failed (attempt {attempt + 1}): {rebase_err}'
            )
            await _run(['git', 'rebase', '--abort'], cwd=merge_worktree)

            if full_branch is None:
                # No branch to re-merge from — cannot recover
                continue

            # Reset merge worktree to current main and re-merge.
            # Pin to the verified branch tip (M^2) so that post-verify
            # commits pushed to the task branch cannot silently land on main.
            # Fall back to full_branch only if M^2 was unresolvable (defensive).
            _remerge_target = verified_branch_tip if verified_branch_tip else full_branch

            # Divergence canary: if the live branch ref has advanced past
            # verified M^2, emit a structured WARNING so any future stale-tip
            # mismatch is self-evident in logs.  Fail-open: a rev-parse error
            # must not block the advance.
            if verified_branch_tip and full_branch:
                _live_rc, _live_sha, _ = await _run(
                    ['git', 'rev-parse', full_branch],
                    cwd=self.project_root,
                )
                if _live_rc == 0:
                    _live_tip = _live_sha.strip()
                    if _live_tip != verified_branch_tip:
                        logger.warning(
                            'advance_main: branch ref diverged from verified M^2 '
                            'during re-merge fallback — pinning to verified tip. '
                            'branch=%s verified_tip=%s live_ref_tip=%s',
                            branch or full_branch,
                            verified_branch_tip[:8],
                            _live_tip[:8],
                        )

            await _run(
                ['git', 'reset', '--hard', self.config.main_branch],
                cwd=merge_worktree,
            )
            merge_rc, merge_out, merge_err = await _run(
                ['git', 'merge', '--no-ff', _remerge_target,
                 '-m', _merge_subject(full_branch, self.config.main_branch)],
                cwd=merge_worktree,
            )
            if merge_rc != 0:
                # True conflict with current main — stop retrying
                logger.warning(
                    f'Re-merge failed (true conflict): {merge_out}\n{merge_err}'
                )
                return AdvanceOutcome('not_descendant')

            _, new_sha, _ = await _run(
                ['git', 'rev-parse', 'HEAD'], cwd=merge_worktree,
            )
            merge_sha = new_sha.strip()
            rebased = True
            continue  # re-check is_ancestor at top of loop
        else:
            # Exhausted all attempts
            logger.warning(
                f'Cannot fast-forward after {max_attempts} attempts: '
                f'{merge_sha[:8]} is not a descendant of '
                f'{self.config.main_branch}'
            )
            return AdvanceOutcome('not_descendant')

        # ── Reverify-on-rebase gate ──────────────────────────────────
        # When reverify_on_rebase is set and a rebase (or re-merge) occurred,
        # park merge_worktree at the rebased SHA and hand control back to the
        # caller WITHOUT advancing main.  The caller must intersect the
        # intervening delta with the branch-touched file set; if overlapping it
        # must re-verify the rebased tree before calling advance_main again.
        if reverify_on_rebase and rebased:
            _, onto_sha, _ = await _run(
                ['git', 'rev-parse', self.config.main_branch],
                cwd=self.project_root,
            )
            rebased_onto = onto_sha.strip()
            logger.info(
                'advance_main: reverify_on_rebase — rebased tree parked at '
                '%s; returning rebased_pending_reverify (no update-ref)',
                merge_sha[:8],
            )
            return AdvanceOutcome(
                'rebased_pending_reverify',
                advanced_sha=merge_sha,
                rebased_from=expected_main,
                rebased_onto=rebased_onto,
            )

        # ── Pre-advance unmerged state guard ────────────────────────
        # Belt-and-braces: reject immediately if project_root already has
        # unresolved merge conflicts in the index.  Any git-stash-create-based
        # park (see _park_wip_on_private_ref) in this state would fail too
        # ("needs merge" / "Cannot save the current index state"), producing
        # 'stash_failed' and hiding the real problem.  Detecting here
        # produces a distinct 'unmerged_state' code that routes to a
        # human-escalation path instead of the steward corrective loop.
        _unmerged_entry_paths = await self._detect_unmerged_paths(self.project_root)
        if _unmerged_entry_paths:
            logger.critical(
                'CRITICAL: project_root has %d pre-existing unresolved merge '
                'conflict(s) (%s) — halting advance_main to prevent data loss. '
                'Manual cleanup required before retrying.',
                len(_unmerged_entry_paths),
                ', '.join(_unmerged_entry_paths[:10]),
            )
            return AdvanceOutcome('unmerged_state')

        # ── Working-tree protection ──────────────────────────────────
        # When project_root has main checked out, update-ref will desync
        # the working tree from HEAD.  Park any uncommitted work first, sync
        # after, then restore.  Parking uses `git stash create` (writes a
        # stash commit WITHOUT touching refs/stash) plus `git update-ref` to
        # record it on the private MERGE_PARK_REF the merge worker
        # exclusively owns — never the shared refs/stash stack, which a
        # human or other session in project_root can race (incident
        # 13674d3c68).  This prevents silent reverts (see 0ea23cb5c).  See
        # MERGE_PARK_REF's module-level docstring and
        # GitOps._park_wip_on_private_ref.
        is_on_main = False
        did_park = False

        rc, current_branch, _ = await _run(
            ['git', 'symbolic-ref', '--short', 'HEAD'],
            cwd=self.project_root,
        )
        if rc == 0 and current_branch.strip() == self.config.main_branch:
            is_on_main = True

            # Check for uncommitted changes (staged or unstaged)
            _, porcelain, _ = await _run(
                ['git', 'status', '--porcelain'],
                cwd=self.project_root,
            )
            if porcelain.strip():
                # ── WIP overlap check ────────────────────────────────
                # Before stashing, check if dirty tracked files overlap
                # with the merge diff.  If they do, abort the advance
                # to prevent stash-pop conflicts that destroy WIP.
                #
                # Use git diff to get tracked dirty filenames reliably.
                # Porcelain parsing is fragile because _run strips stdout,
                # which eats the leading space from " M filename" status.
                # Exclude the worktree dir (managed by git); any leftover
                # .task/ is covered by this repo's root .gitignore .task/
                # entry.
                wt_dir = self.config.worktree_dir
                _, unstaged_files, _ = await _run(
                    ['git', 'diff', '--name-only', '--',
                     '.', f':!{wt_dir}'],
                    cwd=self.project_root,
                )
                _, staged_files, _ = await _run(
                    ['git', 'diff', '--name-only', '--cached', '--',
                     '.', f':!{wt_dir}'],
                    cwd=self.project_root,
                )
                dirty_tracked = {
                    f.strip() for f in
                    (unstaged_files + '\n' + staged_files).splitlines()
                    if f.strip()
                }
                if dirty_tracked:
                    _, merge_diff_files, _ = await _run(
                        ['git', 'diff', '--name-only',
                         await self.get_main_sha(), merge_sha],
                        cwd=self.project_root,
                    )
                    merge_files = {
                        f.strip() for f in merge_diff_files.splitlines() if f.strip()
                    }
                    overlap = dirty_tracked & merge_files
                    if overlap:
                        self._last_overlap_files = sorted(overlap)
                        logger.warning(
                            'WIP overlap detected: %d file(s) overlap merge diff '
                            'for %s — aborting advance to prevent stash-pop '
                            'conflict. Overlapping: %s',
                            len(overlap), branch or merge_sha[:8],
                            ', '.join(sorted(overlap)[:10]),
                        )
                        return AdvanceOutcome('wip_overlap')

                # Only park if there are tracked dirty files.  Untracked-only
                # (??) entries survive read-tree without conflict — parking
                # them risks spurious apply failures (e.g. .worktrees/).
                if dirty_tracked:
                    try:
                        await self._park_wip_on_private_ref(branch or merge_sha[:8])
                    except MergeParkContentionError as e:
                        # The merge worker is serialized, so a resolvable
                        # MERGE_PARK_REF here is either an invariant
                        # violation or a crash-leftover holding real,
                        # unrecovered WIP — never overwritten (see
                        # GitOps._park_wip_on_private_ref).  Halt loudly so
                        # a human can recover the ref rather than silently
                        # destroying the preserved work.
                        logger.critical(
                            'CRITICAL: stale %s present — refusing to '
                            'overwrite; halting to preserve WIP. error=%s',
                            MERGE_PARK_REF, e,
                        )
                        # Surface the dirty tracked files that could not be
                        # parked so _map_advance_failure can name them in the
                        # halt escalation (task 2758) — mirrors the
                        # _last_overlap_files side channel set above.
                        self._last_stash_dirty_files = sorted(dirty_tracked)
                        return AdvanceOutcome('stash_failed')
                    except MergeParkError as e:
                        # git stash create / update-ref infra failure (not a
                        # contention condition — see above).
                        logger.error(
                            'CRITICAL: park failed before advance_main '
                            '— halting merge to prevent code loss. error=%s',
                            e,
                        )
                        # Surface the dirty tracked files that could not be
                        # parked so _map_advance_failure can name them in the
                        # halt escalation (task 2758) — mirrors the
                        # _last_overlap_files side channel set above.
                        self._last_stash_dirty_files = sorted(dirty_tracked)
                        return AdvanceOutcome('stash_failed')
                    did_park = True
                    logger.info('Parked uncommitted changes before advance_main')

        # ── Main-gate mark (best-effort) ─────────────────────────────────
        # Run the project-configurable sentinel command immediately before
        # the update-ref so that reify's reference-transaction hook
        # (git>=2.28, which DOES fire on update-ref) sees the one-shot
        # marker and records this advance as SANCTIONED.  Skipped when the
        # field is unset (feature off — other projects unaffected).
        # Non-zero return is logged as WARNING but never aborts the advance:
        # the task's whole purpose is to prevent queue bricking; under
        # reify ENFORCE a failed mark simply lets update-ref abort →
        # existing 'cas_failed' handling.  Re-runs on every invocation
        # that reaches this point so the one-shot sentinel is refreshed on
        # caller-level CAS retries.
        #
        # SUCCESS PATH: the project's reference-transaction hook is
        # responsible for consuming the sentinel after the successful
        # update-ref.  A missing or non-consuming hook (absent hook, or
        # git < 2.28) leaves the mark stale; the exposure is bounded to
        # at most ONE intervening non-orchestrator move before the next
        # advance_main invocation re-marks + consumes it.
        if self.config.main_gate_mark_command:
            mark_rc, _, mark_err = await _run(
                ['sh', '-c', self.config.main_gate_mark_command],
                cwd=self.project_root,
            )
            if mark_rc != 0:
                logger.warning(
                    'main_gate_mark_command returned non-zero rc=%d: %s',
                    mark_rc, mark_err,
                )

        # All checks passed — advance the ref (CAS when expected_main provided)
        update_cmd = [
            'git', 'update-ref',
            f'refs/heads/{self.config.main_branch}', merge_sha,
        ]
        if expected_main is not None:
            update_cmd.append(expected_main)
        rc, _, err = await _run(update_cmd, cwd=self.project_root)
        if rc != 0:
            # ── Main-gate unmark (best-effort cleanup) ────────────────────
            # A mark written immediately before this failed/aborted update-ref
            # may not have been consumed by the aborted reference-transaction;
            # clear it now so it cannot falsely sanction a later non-
            # orchestrator move.  Runs at the TOP of rc!=0 so it covers both
            # the 'cas_failed' and 'pop_conflict_no_advance' return paths.
            #
            # When main_gate_unmark_command is unset the residual exposure is
            # bounded: a lingering mark sanctions at most ONE intervening move
            # before the next advance_main invocation re-marks+consumes it.
            # This is documented and accepted ("prefer explicit cleanup").
            if self.config.main_gate_unmark_command:
                unmark_rc, _, unmark_err = await _run(
                    ['sh', '-c', self.config.main_gate_unmark_command],
                    cwd=self.project_root,
                )
                if unmark_rc != 0:
                    logger.warning(
                        'main_gate_unmark_command returned non-zero rc=%d: %s',
                        unmark_rc, unmark_err,
                    )

            # Restore parked WIP before returning — ref didn't move.
            # Use _safe_restore_park_with_recovery so that an apply conflict
            # here does NOT leave UU markers in project_root and is
            # escalated to humans rather than silently cascading to
            # 'stash_failed' on the next cycle.
            if did_park:
                restore_ok, recovery = await self._safe_restore_park_with_recovery(
                    branch or merge_sha[:8],
                )
                if not restore_ok:
                    self._last_recovery_branch = recovery
                    logger.critical(
                        'CRITICAL: stash pop conflicted during CAS-failure recovery '
                        '(task %s). WIP preserved on recovery branch: %s. '
                        'Halting — manual intervention required.',
                        branch or merge_sha[:8], recovery,
                    )
                    return AdvanceOutcome('pop_conflict_no_advance')
            if expected_main is not None:
                logger.warning(
                    f'CAS update-ref failed (expected {expected_main[:8]}): {err}'
                )
            else:
                logger.error(f'update-ref failed: {err}')
            return AdvanceOutcome('cas_failed')

        logger.info(f'Advanced {self.config.main_branch} to {merge_sha[:8]}')

        # ── Sync working tree to new HEAD ────────────────────────────
        # update-ref moved the ref but left the working tree stale.
        # read-tree syncs the index and working tree to the new HEAD.
        # Then pop the stash to restore any in-progress user work.
        if is_on_main:
            sync_rc, _, sync_err = await _run(
                ['git', 'read-tree', '-u', '--reset', 'HEAD'],
                cwd=self.project_root,
            )
            if sync_rc != 0:
                logger.error(
                    'read-tree failed after advancing main — working tree '
                    'is stale. error=%s', sync_err,
                )

            if did_park:
                restore_ok, recovery = await self._safe_restore_park_with_recovery(
                    branch or merge_sha[:8],
                )
                if not restore_ok:
                    self._last_recovery_branch = recovery
                    logger.warning(
                        'Stash pop conflicted after merge advance (task %s). '
                        'WIP preserved on recovery branch: %s',
                        branch or merge_sha[:8], recovery,
                    )
                    # Main was advanced before the stash pop ran — the
                    # returned AdvanceOutcome.advanced_sha lets callers
                    # handling done_wip_recovery record correct
                    # done_provenance.
                    return AdvanceOutcome('pop_conflict', advanced_sha=merge_sha)

        # Main was advanced — the returned AdvanceOutcome.advanced_sha lets
        # callers record done_provenance against the SHA actually on main,
        # not the stale pre-rebase SHA from MergeResult.merge_commit.
        return AdvanceOutcome('advanced', advanced_sha=merge_sha)

    async def recover_red_main(
        self,
        target_sha: str,
        expected_main: str,
    ) -> RecoverResult:
        """Move refs/heads/main BACKWARD to *target_sha* atomically, break-glass past reify's gate.

        This is the enforce-safe break-glass recovery operation: a SINGLE CAS
        update-ref that drops a bad merge in one move (no rewind-then-readvance).
        Because the move is BACKWARD (history-rewriting), a project with an
        always-on non-fast-forward main-gate guard (reify) rejects it even when
        sanctioned, so — when configured — this engages a DURABLE bypass
        (main_gate_bypass_command) immediately before the update-ref and clears
        it (main_gate_bypass_clear_command) on every path afterward.  The bypass
        SUPERSEDES the sanction mark (they are mutually exclusive; see the
        engage block below).  Projects with only a sanction gate leave the
        bypass unset and fall back to advance_main's mark → update-ref →
        unmark-on-failure sequence unchanged.

        Args:
            target_sha:    The good SHA to restore main to (the pre-bad-merge state).
            expected_main: The current (bad) value of refs/heads/main; used as the
                           CAS old-value so a concurrent ref-move aborts cleanly.

        Returns:
            ``'rewound'``    — update-ref succeeded; main now points at target_sha.
            ``'cas_failed'`` — another writer moved the ref first; no change made.
            ``'error'``      — a SHA failed pre-validation; fix the SHA and retry.

        Note:
            The caller (skill) must ensure project_root is clean before invoking
            this method.  recover_red_main does NOT stash/pop uncommitted WIP —
            that dance is out of scope for a break-glass operation.
        """
        # ── Check working tree ────────────────────────────────────────────
        is_on_main = False
        rc, branch_out, _ = await _run(
            ['git', 'symbolic-ref', '--short', 'HEAD'],
            cwd=self.project_root,
        )
        if rc == 0 and branch_out.strip() == self.config.main_branch:
            is_on_main = True

        # ── Pre-validate SHAs (distinguish bad-input from CAS mismatch) ─────
        # A non-existent or non-commit SHA would make update-ref fail with a
        # repo-level error indistinguishable from a CAS mismatch.  Fail early
        # with 'error' so the runbook routes to "fix the SHA" rather than the
        # retry loop intended for genuine CAS races.
        for _sha_label, _sha_val in (
            ('target_sha', target_sha),
            ('expected_main', expected_main),
        ):
            _v_rc, _, _v_err = await _run(
                ['git', 'rev-parse', '--verify', f'{_sha_val}^{{commit}}'],
                cwd=self.project_root,
            )
            if _v_rc != 0:
                logger.error(
                    'recover_red_main: %s %r does not resolve to a commit; '
                    'fix the SHA before retrying. detail=%s',
                    _sha_label, (_sha_val or '')[:8], _v_err.strip(),
                )
                return 'error'

        # ── Main-gate engage + CAS move (bypass SUPERSEDES mark) ──────────
        # reify's reference-transaction hook (git>=2.28) runs an ALWAYS-ON
        # non-fast-forward guard whose ordering is: (1) if the durable bypass
        # is engaged -> `continue` (skip the ref, never reach the non-ff
        # check); (2) else reject any backward/history-rewriting ref move
        # UNCONDITIONALLY (not gated by ENFORCE, not opted out by the
        # sanction); (3) only if the non-ff guard passes is the SANCTION
        # sentinel consulted.  recover_red_main moves main BACKWARD
        # (expected_main is NOT an ancestor of target_sha), so the non-ff
        # guard (step 2) rejects the move BEFORE the sanction (step 3) is
        # reached — the mark alone is useless here.  So engage the DURABLE
        # bypass immediately before the CAS update-ref.
        #
        # Bypass is MUTUALLY EXCLUSIVE with the mark: reify's real config sets
        # BOTH (advance_main's forward ff moves need the sanction; recover's
        # backward move needs the bypass).  If we ran the mark alongside an
        # engaged bypass, the hook `continue`s on the bypass BEFORE it would
        # consume the one-shot sanction sentinel -> the sentinel lingers and
        # falsely sanctions the NEXT unrelated ref move (a real leak).  So when
        # the bypass command is configured we engage it and SKIP the mark.
        # Projects with a sanction-only gate (no non-ff guard) leave the bypass
        # unset and keep the existing mark path byte-for-byte unchanged.
        #
        # The engage step AND the CAS update-ref share ONE try/finally so the
        # DURABLE bypass is cleared on EVERY exit path — success, CAS failure,
        # an exception from the update-ref subprocess, AND an exception raised
        # mid-engage (e.g. a transport/output-capture error rather than a
        # non-zero rc).  bypass_engaged is therefore set True IMMEDIATELY
        # BEFORE the engage await, not after it: a bypass that partially
        # applies its durable state and then raises must still be cleared by
        # the finally.  (A bypassed txn consumes nothing — unlike the one-shot
        # mark sentinel reify's hook consumes on a successful SANCTIONED txn —
        # so recover_red_main is solely responsible for clearing the bypass;
        # leaving it engaged would disable the project's non-ff guard for all
        # subsequent ref moves.)
        bypass_engaged = False
        try:
            if self.config.main_gate_bypass_command:
                # Set engaged BEFORE the await (see the try/finally note above):
                # a partially-applied bypass whose _run then raises is still
                # cleared by the finally.
                bypass_engaged = True
                bypass_rc, _, bypass_err = await _run(
                    ['sh', '-c', self.config.main_gate_bypass_command],
                    cwd=self.project_root,
                )
                if bypass_rc != 0:
                    logger.warning(
                        'main_gate_bypass_command returned non-zero rc=%d: %s',
                        bypass_rc, bypass_err,
                    )
            elif self.config.main_gate_mark_command:
                mark_rc, _, mark_err = await _run(
                    ['sh', '-c', self.config.main_gate_mark_command],
                    cwd=self.project_root,
                )
                if mark_rc != 0:
                    logger.warning(
                        'main_gate_mark_command returned non-zero rc=%d: %s',
                        mark_rc, mark_err,
                    )

            # ── CAS move of refs/heads/main ───────────────────────────────
            rc, _, err = await _run(
                ['git', 'update-ref',
                 f'refs/heads/{self.config.main_branch}',
                 target_sha, expected_main],
                cwd=self.project_root,
            )
        finally:
            if bypass_engaged:
                if self.config.main_gate_bypass_clear_command:
                    clear_rc, _, clear_err = await _run(
                        ['sh', '-c', self.config.main_gate_bypass_clear_command],
                        cwd=self.project_root,
                    )
                    if clear_rc != 0:
                        logger.warning(
                            'main_gate_bypass_clear_command returned non-zero '
                            'rc=%d: %s',
                            clear_rc, clear_err,
                        )
                else:
                    # Defense-in-depth: GitConfig._reject_bypass_command_without_clear
                    # rejects bypass-without-clear at config load, so this branch
                    # should be UNREACHABLE.  If it is ever reached (a
                    # post-construction config mutation, or a future removal of
                    # that validator), fail LOUD rather than silently leave the
                    # durable bypass — and thus the project's non-fast-forward
                    # main-gate guard — DISABLED for every later ref move.
                    logger.error(
                        'recover_red_main: durable main-gate bypass was engaged '
                        'but main_gate_bypass_clear_command is unset — the '
                        'non-fast-forward main-gate guard has been left DISABLED. '
                        'This should be unreachable (the GitConfig validator '
                        'rejects it at load); investigate the config.'
                    )
        if rc != 0:
            # ── Main-gate unmark (best-effort cleanup) ────────────────────
            # Only on the mark path: when the bypass was engaged the mark was
            # never written (bypass supersedes mark), so there is nothing to
            # unmark — and the durable bypass has already been cleared in the
            # finally above.
            if not bypass_engaged and self.config.main_gate_unmark_command:
                unmark_rc, _, unmark_err = await _run(
                    ['sh', '-c', self.config.main_gate_unmark_command],
                    cwd=self.project_root,
                )
                if unmark_rc != 0:
                    logger.warning(
                        'main_gate_unmark_command returned non-zero rc=%d: %s',
                        unmark_rc, unmark_err,
                    )
            logger.warning(
                'recover_red_main: CAS update-ref failed (expected %s): %s',
                expected_main[:8], err,
            )
            return 'cas_failed'

        logger.info(
            'recover_red_main: rewound %s to %s',
            self.config.main_branch, target_sha[:8],
        )

        # ── Sync working tree to new HEAD ─────────────────────────────────
        if is_on_main:
            # Warn if the tree is dirty — read-tree will silently discard
            # uncommitted tracked changes.  The caller/runbook is expected to
            # clean up first; this is a last-resort advisory so an operator
            # under stress isn't silently burned.
            _st_rc, _st_out, _ = await _run(
                ['git', 'status', '--porcelain'],
                cwd=self.project_root,
            )
            if _st_rc == 0 and _st_out.strip():
                logger.warning(
                    'recover_red_main: working tree has uncommitted changes '
                    '(git status --porcelain non-empty); '
                    'git read-tree will silently discard tracked WIP. '
                    'Ensure project_root is clean before break-glass recovery.',
                )
            sync_rc, _, sync_err = await _run(
                ['git', 'read-tree', '-u', '--reset', 'HEAD'],
                cwd=self.project_root,
            )
            if sync_rc != 0:
                logger.error(
                    'read-tree failed after recover_red_main — working tree '
                    'is stale. error=%s', sync_err,
                )

        return 'rewound'

    async def push_main(self) -> PushResult:
        """Push local main to ``<remote>/<main_branch>`` as a fast-forward.

        Best-effort mirror step for ``advance_main``: keeps origin in sync
        without ever blocking the merge worker. Local main is the source of
        truth; this is a one-way replication. Never raises and never uses
        ``--force``.

        Returns:
            ``'pushed'``   — push succeeded.
            ``'noop'``     — disabled via ``config.push_after_advance``.
            ``'rejected'`` — non-fast-forward (origin diverged); logged at ERROR.
            ``'error'``    — network / auth / other transient failure;
                             logged at WARNING.
        """
        if not self.config.push_after_advance:
            return 'noop'

        refspec = f'{self.config.main_branch}:{self.config.main_branch}'
        rc, _, err = await _run(
            ['git', 'push', self.config.remote, refspec],
            cwd=self.project_root,
        )
        if rc == 0:
            logger.info(
                'Pushed %s to %s', self.config.main_branch, self.config.remote,
            )
            return 'pushed'

        # Classify the failure. git push surfaces non-ff in stderr with one of
        # several phrasings depending on version/locale.
        err_lower = err.lower()
        if any(s in err_lower for s in ('non-fast-forward', 'fetch first', '! [rejected]')):
            logger.error(
                'Push of %s to %s rejected (non-fast-forward) — origin has '
                'diverged. NOT force-pushing. stderr=%s',
                self.config.main_branch, self.config.remote, err,
            )
            return 'rejected'

        logger.warning(
            'Push of %s to %s failed (rc=%d) — leaving origin behind; '
            'next successful push will catch up. stderr=%s',
            self.config.main_branch, self.config.remote, rc, err,
        )
        return 'error'

    async def _park_wip_on_private_ref(self, label: str) -> None:
        """Park uncommitted WIP in project_root onto MERGE_PARK_REF.

        Uses ``git stash create`` (writes a stash commit WITHOUT touching
        the shared ``refs/stash`` stack) plus ``git update-ref`` to record
        it on a private ref the merge worker exclusively owns, then
        ``git read-tree -u --reset HEAD`` to clean the working tree —
        mirroring what ``git stash push`` used to do.  See MERGE_PARK_REF's
        module-level docstring for why the shared stash stack is unsafe here
        (incident 13674d3c68).

        Raises :class:`MergeParkError` if ``git stash create`` fails or
        produces no commit, or if the final ``git read-tree -u --reset HEAD``
        (which cleans the working tree after the WIP is captured on the ref)
        fails — the WIP is already safe on MERGE_PARK_REF at that point, but
        the caller must still halt rather than proceed with a dirty tree.
        Raises :class:`MergeParkContentionError` if MERGE_PARK_REF already
        exists — the merge worker is serialized, so a resolvable ref here is
        either an invariant violation or a crash-leftover holding real,
        unrecovered WIP; it is never overwritten.
        """
        # Single-flight guard: explicit pre-check so a stale/contended ref
        # fails loudly with a clear message rather than via the terser
        # update-ref rc=128 below (which is a belt-and-braces backstop for
        # the TOCTOU window, not the primary signal).
        guard_rc, guard_sha, _ = await _run(
            ['git', 'rev-parse', '--verify', '--quiet', MERGE_PARK_REF],
            cwd=self.project_root,
        )
        if guard_rc == 0:
            raise MergeParkContentionError(
                f'{MERGE_PARK_REF} already exists at {guard_sha.strip()!r} — '
                'refusing to overwrite (stale or contended park).'
            )

        stash_rc, stash_sha, stash_err = await _run(
            ['git', 'stash', 'create', f'merge-queue: pre-advance for {label}'],
            cwd=self.project_root,
        )
        stash_sha = stash_sha.strip()
        if stash_rc != 0 or not stash_sha:
            raise MergeParkError(
                f'git stash create failed or produced no commit (rc={stash_rc}, '
                f'stdout={stash_sha!r}, stderr={stash_err!r})'
            )

        # Atomic create-only update-ref: the all-zeros old-value makes this
        # fail (rc=128) if the ref already exists — a belt-and-braces
        # backstop closing the TOCTOU window between the guard check above
        # and this update-ref (the merge worker's serialization precludes
        # this in practice, but the atomic form costs nothing).
        update_rc, _, update_err = await _run(
            ['git', 'update-ref', MERGE_PARK_REF, stash_sha, '0' * 40],
            cwd=self.project_root,
        )
        if update_rc != 0:
            raise MergeParkContentionError(
                f'update-ref {MERGE_PARK_REF} refused (rc={update_rc}) — ref '
                f'appeared concurrently: {update_err!r}'
            )

        reset_rc, _, reset_err = await _run(
            ['git', 'read-tree', '-u', '--reset', 'HEAD'],
            cwd=self.project_root,
        )
        if reset_rc != 0:
            raise MergeParkError(
                f'read-tree -u --reset HEAD failed after parking WIP on '
                f'{MERGE_PARK_REF} (rc={reset_rc}, stderr={reset_err!r}) — '
                'WIP is safe on the ref, but the working tree may still be '
                'dirty; halting rather than proceeding out of sync.'
            )

    async def _create_recovery_branch_from_park_ref(self, label: str) -> str:
        """Create a branch from MERGE_PARK_REF to preserve WIP, then clean up.

        1. Create a deterministic branch name.
        2. ``git branch <name> MERGE_PARK_REF`` — makes the parked commit
           reachable from a plain branch.
        3. ``git update-ref -d MERGE_PARK_REF`` — safe now (WIP reachable
           via the branch); frees the private ref for the next park.
        4. ``git read-tree -u --reset HEAD`` — clean working tree (removes
           conflict markers and UU state).

        Returns the recovery branch name.
        """
        from datetime import UTC, datetime

        iso = datetime.now(UTC).strftime('%Y%m%dT%H%M%S')
        name = f'wip/recovery-{label}-{iso}'

        # Create branch pointing at the parked commit
        await _run(
            ['git', 'branch', name, MERGE_PARK_REF],
            cwd=self.project_root,
        )
        # Delete the private ref (WIP is now reachable via the branch).  A
        # failed delete leaks the ref rather than losing WIP (the branch
        # already anchors it), but the leak would surface as a confusing
        # MergeParkContentionError halt on the *next* park — log loudly now
        # so the real cause is diagnosable.
        del_rc, _, del_err = await _run(
            ['git', 'update-ref', '-d', MERGE_PARK_REF], cwd=self.project_root,
        )
        if del_rc != 0:
            logger.critical(
                'CRITICAL: failed to delete %s after branching WIP to %s '
                '(rc=%d, err=%s) — the ref may leak and cause a spurious '
                'contention halt on the next park.',
                MERGE_PARK_REF, name, del_rc, del_err,
            )
        # Reset working tree to HEAD (removes conflict markers / UU state)
        await _run(
            ['git', 'read-tree', '-u', '--reset', 'HEAD'],
            cwd=self.project_root,
        )
        return name

    async def _safe_restore_park_with_recovery(
        self, label: str,
    ) -> tuple[bool, str | None]:
        """Apply MERGE_PARK_REF and preserve WIP on a recovery branch if it conflicts.

        1. Run ``git stash apply MERGE_PARK_REF`` — restores WIP from the
           private ref the merge worker exclusively owns, never the shared
           stash stack.
        2. Check return code AND ``_detect_unmerged_paths`` — either signal
           is sufficient to declare failure (belt-and-braces).
        3. On failure: call ``_create_recovery_branch_from_park_ref(label)``
           which branches off MERGE_PARK_REF, deletes the ref, and resets
           the working tree to HEAD.
        4. On success: delete MERGE_PARK_REF (WIP is now applied to the
           working tree).
        5. Return ``(True, None)`` on clean apply, or
           ``(False, recovery_branch_name)`` on conflict.
        """
        apply_rc, _, apply_err = await _run(
            ['git', 'stash', 'apply', MERGE_PARK_REF], cwd=self.project_root,
        )
        unmerged = await self._detect_unmerged_paths(self.project_root)

        if apply_rc != 0 or unmerged:
            logger.warning(
                'Park ref apply failed (rc=%d, unmerged=%s, err=%s) for label %r — '
                'creating recovery branch to preserve WIP.',
                apply_rc, unmerged or [], apply_err, label,
            )
            recovery = await self._create_recovery_branch_from_park_ref(label)
            return (False, recovery)

        # Delete the private ref (WIP is now applied to the working tree).
        # A failed delete leaks the ref — it doesn't lose WIP (already
        # applied), but the leak surfaces as a confusing spurious
        # MergeParkContentionError halt on the *next* park; log loudly now
        # so the real cause is diagnosable.
        del_rc, _, del_err = await _run(
            ['git', 'update-ref', '-d', MERGE_PARK_REF], cwd=self.project_root,
        )
        if del_rc != 0:
            logger.critical(
                'CRITICAL: failed to delete %s after a clean apply for label '
                '%r (rc=%d, err=%s) — the ref may leak and cause a spurious '
                'contention halt on the next park.',
                MERGE_PARK_REF, label, del_rc, del_err,
            )
        return (True, None)

    async def has_dirty_working_tree(self) -> str:
        """Return names of tracked dirty files, or empty string if clean.

        Excludes untracked files.  The leftover ``.task/`` (if any) is
        covered by this repo's root ``.gitignore`` ``.task/`` entry, so it
        never appears here without a pathspec exclusion.
        """
        _, unstaged, _ = await _run(
            ['git', 'diff', '--name-only', '--', '.'],
            cwd=self.project_root,
        )
        _, staged, _ = await _run(
            ['git', 'diff', '--name-only', '--cached', '--', '.'],
            cwd=self.project_root,
        )
        files = {f.strip() for f in (unstaged + '\n' + staged).splitlines() if f.strip()}
        return '\n'.join(sorted(files))

    async def _detect_unmerged_paths(self, cwd: Path) -> list[str]:
        """Return sorted list of file paths that are in an unmerged state.

        Uses ``git status --porcelain`` XY parsing — a path is unmerged if
        either the index (X) or working-tree (Y) column is ``U``, OR if both
        columns are the same add/delete marker (``AA`` or ``DD``).

        Returns an empty list when the tree is clean or fully merged.
        """
        _, porcelain, _ = await _run(
            ['git', 'status', '--porcelain'],
            cwd=cwd,
        )
        unmerged: list[str] = []
        for line in porcelain.splitlines():
            if len(line) < 4:
                continue
            xy = line[:2]
            path = line[3:]
            if 'U' in xy or xy in ('AA', 'DD'):
                unmerged.append(path.strip())
        return sorted(unmerged)

    async def get_conflict_details(self, cwd: Path) -> str:
        """Parse conflict markers and return structured description."""
        _, status, _ = await _run(['git', 'diff', '--name-only', '--diff-filter=U'], cwd=cwd)
        if not status:
            return 'No conflicting files detected'

        details = [f'Conflicting files:\n{status}\n']
        for filepath in status.splitlines():
            filepath = filepath.strip()
            if filepath:
                _, diff, _ = await _run(['git', 'diff', '--', filepath], cwd=cwd)
                details.append(f'--- {filepath} ---\n{diff[:2000]}')

        return '\n'.join(details)

    async def abort_merge(self, cwd: Path) -> None:
        """Abort an in-progress merge."""
        await _run(['git', 'merge', '--abort'], cwd=cwd)
        logger.info('Merge aborted')

    async def rename_worktree(
        self,
        old_path: Path,
        new_path: Path,
        old_branch: str,
        new_branch: str,
    ) -> None:
        """Rename a registered worktree and its branch atomically.

        Used by the auto-eval hook to preserve the original
        attempt's branch + worktree (suffixed ``-skip-attempt``) so the
        full-architect redo can use the original branch name without
        clobbering the artefacts of the optimistic-path attempt.

        Args:
            old_path: Current worktree path (registered with git).
            new_path: Destination worktree path (must not exist).
            old_branch: Branch name without the ``branch_prefix``.
            new_branch: Destination branch name without the ``branch_prefix``.

        Raises:
            RuntimeError: if ``git worktree move`` or ``git branch -m``
                returns a non-zero exit code. The caller is expected to
                surface this as an auto-eval failure and fall back to the
                normal block path.
        """
        full_old = f'{self.config.branch_prefix}{old_branch}'
        full_new = f'{self.config.branch_prefix}{new_branch}'

        new_path.parent.mkdir(parents=True, exist_ok=True)

        rc, _, err = await _run(
            ['git', 'worktree', 'move', str(old_path), str(new_path)],
            cwd=self.project_root,
        )
        if rc != 0:
            raise RuntimeError(
                f'rename_worktree: git worktree move {old_path} -> '
                f'{new_path} failed (rc={rc}): {err}'
            )

        rc, _, err = await _run(
            ['git', 'branch', '-m', full_old, full_new],
            cwd=self.project_root,
        )
        if rc != 0:
            # Best-effort rollback of the worktree move so the caller can
            # retry. The directory rename is the half that actually
            # surfaces conflicts; the branch rename rarely fails alone.
            await _run(
                ['git', 'worktree', 'move', str(new_path), str(old_path)],
                cwd=self.project_root,
            )
            raise RuntimeError(
                f'rename_worktree: git branch -m {full_old} -> {full_new} '
                f'failed (rc={rc}): {err}'
            )

        logger.info(
            'Renamed worktree %s -> %s and branch %s -> %s',
            old_path, new_path, full_old, full_new,
        )

    async def cleanup_worktree(self, worktree: Path, branch: str) -> None:
        """Remove worktree and delete branch.

        **Pool-aware**: if *worktree* is a warm lane, routes to
        :meth:`release_warm_lane` (retain worktree + ``target/``, flip FREE)
        instead of removing.  Mirrors :meth:`cleanup_merge_worktree`'s
        persistent-path no-op pattern.  Covers the workflow done-gate and all
        harness reconcile call sites without touching each individually.
        """
        if self.warm_lane_pool is not None and self.warm_lane_pool.is_lane(worktree):
            await self.release_warm_lane(worktree, branch)
            return

        # Symmetric for the merge-speculation pool: a '_spec-' lane must be
        # RELEASED back to its pool (retain worktree + target/, flip FREE),
        # never removed.  Without this, a spec lane routed here (e.g. by the
        # crash-recovery sweep) would be git-worktree-removed and lose its
        # pool slot.  warm=True selects the pool-retain path in
        # release_spec_lane.
        if (
            self.spec_warm_lane_pool is not None
            and self.spec_warm_lane_pool.is_lane(worktree)
        ):
            await self.release_spec_lane(worktree, warm=True)
            return

        # ── Teardown-archival backstop (task 2786, agent-transcript-archival-prd
        # β) ────────────────────────────────────────────────────────────────
        # Before the worktree (and the per-task Claude config dir INSIDE it) is
        # destroyed, archive any still-un-archived agent transcript to the
        # durable root OUTSIDE the worktree. This closes the abandoned-in-flight
        # tail the producer hook (α/workflow.py _invoke) cannot: a role in-flight
        # when the orchestrator died, whose task is reaped without a completed
        # resume. Idempotent with the producer — same archive_root + task_id, so
        # the helper's size/mtime skip fires (a no-op in the normal case).
        # Reached only on COLD removals: warm/spec lanes returned above (they are
        # retained, not removed), and branch == task_id at every cold call site,
        # so it is the task_id the config-dir path and archive layout key on.
        # Offloaded to a worker thread to keep the shared event loop free for the
        # rare real-gzip path (mirrors the producer's loop-stall avoidance).
        if self.transcript_archive is not None and self.transcript_archive.enabled:
            config_dir = worktree / '.task' / f'claude-config-{branch}'
            # Fast-skip when the config dir is already gone (external worktrees,
            # already-cleaned dirs): a cheap no-op that never spins up a worker
            # thread just to glob an absent projects/ tree. The producer already
            # archived the normal case; the size/mtime skip inside the helper
            # (matching archive_root + task_id) makes any overlap idempotent.
            if config_dir.exists():
                archive_root = self.project_root / self.transcript_archive.root
                try:
                    await asyncio.to_thread(
                        archive_task_transcripts,
                        config_dir,
                        branch,
                        None,
                        archive_root=archive_root,
                    )
                except asyncio.CancelledError:
                    # Cooperative cancellation (loop teardown / hard-kill)
                    # surfaces here from the await, NOT an archival error — it
                    # must propagate, never be swallowed (mirrors the producer,
                    # workflow.py _invoke). CancelledError is a BaseException, so
                    # the `except Exception` below deliberately does not catch it.
                    raise
                except Exception:
                    # Best-effort: teardown must never be blocked by a broken or
                    # contract-regressed archiver. archive_task_transcripts is
                    # total by contract (per-file OSErrors are swallowed +
                    # counted), but its top-level glob / Path / archive_root
                    # construction is not individually guarded — swallow any
                    # escaped non-cancellation error here so `git worktree remove`
                    # still runs. Loud, not silent: logged as a structured fact.
                    logger.warning(
                        'Transcript archival backstop failed for task %s '
                        '(worktree %s)',
                        branch,
                        worktree,
                        exc_info=True,
                        extra={'task_id': branch, 'worktree': str(worktree)},
                    )

        full_branch = f'{self.config.branch_prefix}{branch}'

        # Remove worktree
        rc, _, err = await _run(
            ['git', 'worktree', 'remove', str(worktree), '--force'],
            cwd=self.project_root,
        )
        if rc != 0:
            logger.warning(f'Failed to remove worktree {worktree}: {err}')

        # Delete branch — on-main only: skip if the branch carries commits
        # beyond main (i.e. still holds unmerged WIP).
        await self._delete_branch_if_on_main(full_branch, context='cleanup_worktree')

        logger.info(f'Cleaned up worktree {worktree} and branch {full_branch}')

    async def reclaim_worktree_build_artifacts(
        self,
        worktree: Path,
        dir_names: list[str] | None = None,
    ) -> list[Path]:
        """Remove regenerable build-artifact directories from a done worktree.

        Drops only the named build-output subdirectories (e.g. ``target/``)
        and never touches git refs, the worktree admin entry, or any other
        content.  This is appropriate when the task's merge commit is
        confirmed on main but the branch tip is a pre-rebase duplicate —
        the forensic history is preserved while the large regenerable cache
        is reclaimed.

        *dir_names* overrides which subdirectory names to reap.  When
        ``None``, falls back to ``self.config.reap_build_artifact_dirs``
        (default ``['target']``).

        Best-effort: each removal is wrapped in try/except; failures are
        logged as warnings but never propagated.  Mirrors the
        never-raise contract of ``cleanup_merge_worktree`` and
        ``prune_worktrees``.

        Returns the list of directory paths that were successfully removed.
        Returns ``[]`` when nothing was reaped (dirs absent or worktree
        path does not exist).
        """
        names = dir_names if dir_names is not None else self.config.reap_build_artifact_dirs
        removed: list[Path] = []

        for name in names:
            candidate = worktree / name
            if not candidate.is_dir():
                continue
            try:
                shutil.rmtree(candidate)
                removed.append(candidate)
            except Exception:
                logger.warning(
                    'reclaim_worktree_build_artifacts: failed to remove %s',
                    candidate, exc_info=True,
                )

        if removed:
            logger.info(
                'reclaim_worktree_build_artifacts: removed %d dir(s) from %s: %s',
                len(removed), worktree, [str(p) for p in removed],
            )
        return removed

    # ── Orphan-worktree hygiene (Fix B/C) ─────────────────────────────

    @property
    def quarantine_base(self) -> Path:
        """Sibling base for quarantined worktrees — OUTSIDE ``worktree_base``.

        A direct sibling (``<worktree_dir>-orphaned``) rather than a child, so
        a quarantined worktree is never re-scanned by crash-recovery or the
        orphan reaper (both iterate ``worktree_base`` only).
        """
        return self.worktree_base.parent / f'{self.worktree_base.name}-orphaned'

    async def worktree_has_unsaved_work(self, worktree: Path, branch: str) -> bool:
        """Whether a worktree holds work that must be preserved before removal.

        ``True`` if EITHER the branch carries commits beyond main
        (``rev-list --count main..task/<branch> > 0``) OR the working tree is
        dirty (``git status --porcelain`` non-empty).  **Fail-safe ``True``**
        on any git error (including a missing branch) — never report a worktree
        as safe-to-reap when we cannot prove it is empty and clean.
        """
        full_branch = f'{self.config.branch_prefix}{branch}'
        try:
            # Commits beyond main.  A missing branch makes rev-list fail → True.
            rc, out, _ = await _run(
                ['git', 'rev-list', '--count',
                 f'{self.config.main_branch}..{full_branch}'],
                cwd=self.project_root,
            )
            if rc != 0:
                return True
            if int(out.strip()) > 0:
                return True
            # No commits beyond main — check for uncommitted WIP in the tree.
            rc, status_out, _ = await _run(
                ['git', 'status', '--porcelain'],
                cwd=worktree,
            )
            if rc != 0:
                return True
            return bool(status_out.strip())
        except (WorktreeMissing, ValueError, OSError) as e:
            logger.warning(
                'worktree_has_unsaved_work: error inspecting %s (%s) — '
                'treating as unsaved (fail-safe)', worktree, e,
            )
            return True

    async def quarantine_worktree(
        self, worktree: Path, branch: str, reason: str,
    ) -> Path | None:
        """Relocate a worktree (and its branch) into the quarantine base.

        Best-effort: commits any uncommitted WIP first (so it is preserved on
        the renamed branch), then moves the worktree to
        ``quarantine_base/<branch>-<UTC-ts>`` and renames the branch to
        ``task/<branch>-<ts>``.  Logs a WARNING and returns the destination
        path, or ``None`` if the relocation could not complete.  **Never
        raises** — callers treat a ``None`` return as "left in place".
        """
        ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
        dest_name = f'{branch}-{ts}'
        dest_path = self.quarantine_base / dest_name
        try:
            # Preserve uncommitted WIP on the branch before relocating.
            try:
                await self.commit(worktree, f'chore: quarantine WIP ({reason})')
            except Exception as e:
                logger.warning(
                    'quarantine_worktree: WIP commit failed for %s (%s) — '
                    'continuing with relocation: %s', worktree, reason, e,
                )
            await self.rename_worktree(worktree, dest_path, branch, dest_name)
            logger.warning(
                'QUARANTINED worktree %s -> %s (reason=%s)',
                worktree, dest_path, reason,
            )
            return dest_path
        except Exception as e:
            logger.warning(
                'quarantine_worktree: failed to relocate %s (reason=%s): %s',
                worktree, reason, e,
            )
            return None

    async def _prune_registrations(self, context: str) -> None:
        """Best-effort ``git worktree prune`` — clears stale admin entries.

        Single chokepoint for every raw ``['git', 'worktree', 'prune']``
        call site in this module (gitops-chokepoints PRD, task α). Clears
        the ``.git/worktrees`` administrative records left behind by
        worktrees removed off-band (manual ``rm -rf``, quarantine, reap).
        Never raises.

        Args:
            context: Short identifier for the calling sweep (e.g.
                ``'prune_worktrees'``, ``'create_worktree-leftover'``),
                threaded into every log line so operators can attribute
                which caller asked for the prune.

        **Pool-storage guard (task 2099, self-heal task 2315)**: routes
        through :meth:`_reconcile_pool_storage_before_sweep`, which refuses
        to run when a pool is configured (:meth:`pool_in_use`) but
        :meth:`pool_storage_present` is False AND that absence is not
        provably a first-seed bootstrap.  An unmounted mountpoint dir makes
        every mount-resident worktree APPEAR removed off-band, so an
        unguarded prune would wipe every registered lane + ``_merge-verify``
        admin entry the instant the mount comes back — exactly the Jul-3
        incident this guards against.  Skipped entirely when no pool is in
        use: ``pool_storage_present()`` is permanently False on a pool-less
        host (its only writer never runs without a pool), so that alone
        must never disable ``git worktree prune`` on every default host.

        **Pre-first-seed bootstrap / self-heal (task 2099 + 2315)**: a
        freshly-provisioned pool-configured host that has created
        ``worktree_base`` (e.g. pool warmup ``mkdir``) but has not yet run a
        successful seed — or a previously-healthy host that simply lost its
        ``.pool-root`` sentinel — also has no sentinel, indistinguishable
        from an unmounted mount by the sentinel alone.  When
        :meth:`_pool_storage_bootstrap_ok` confirms this is the benign case
        (the CoW seed base already resolves under ``worktree_base``), the
        shared helper RECREATES the sentinel and the prune PROCEEDS
        normally (task 2315: previously this only skipped without
        recreating the sentinel, a chicken-and-egg that suppressed
        self-heal forever); the escalation callback is suppressed either
        way so a legitimate cold start/self-heal does not file operator
        noise.

        **Escalation debounce**: the refusal branch (routed through
        :meth:`_reconcile_pool_storage_before_sweep`) calls
        :meth:`_note_pool_storage_absent` unconditionally on every TRUE
        refusal — see that method's docstring for why repeated calls from
        hot sweep sites (e.g. ``create_worktree``, ``reap_interactive_worktrees``)
        do not multiply operator-visible escalations.
        """
        if not self._reconcile_pool_storage_before_sweep(context):
            return
        try:
            rc, _, err = await _run(
                ['git', 'worktree', 'prune'], cwd=self.project_root,
            )
            if rc != 0:
                logger.warning('%s: git worktree prune failed: %s', context, err)
        except Exception as e:
            logger.warning('%s: git worktree prune raised: %s', context, e)

    async def prune_worktrees(self, context: str = 'prune_worktrees') -> None:
        """Best-effort ``git worktree prune`` — thin public delegate.

        See :meth:`_prune_registrations` for the full guard/skip/escalate
        semantics. Kept as a separate public method so existing harness
        callers (:meth:`orchestrator.harness.Harness`, the orphan reaper)
        are unaffected.
        """
        await self._prune_registrations(context=context)
