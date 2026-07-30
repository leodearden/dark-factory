"""Merge-worktree lifecycle integrity zeta done-gate: restart-simulation
boundary suite (PRD Sec.9 rows 1-9).

PRD: docs/prds/merge-worktree-lifecycle-integrity.md, task zeta (the B+H
done-gate).  All five prerequisite legs are LANDED and BEHAVIOUR-FROZEN for
this batch:

  alpha (2924) -- GitOps.remove_merge_worktree_guarded (git_ops.py):
      lease-enforced removal primitive; outcome vocabulary 'removed' /
      'skipped_lease_held' / 'skipped_persistent' / 'not_present' / 'failed'.
  beta  (2925) -- classify_worktree_entry (git_ops.py) + the C2 namespace
      guard in Harness._recover_crashed_tasks (harness.py): the
      crash-recovery sweep SKIPS+REPORTS `_merge-*`/infra bands instead of
      force-removing them (the 2026-07-22 task/5326 incident).
  gamma (2926) -- recover_pending_merges' registry-gated per-branch collapse
      (merge_queue_store.py): a branch with N surviving journal
      entries enqueues exactly ONE winner (descendant-most snapshot tip);
      every loser attaches as a peer waiter whose future mirrors the
      winner's terminal outcome.
  delta (2927) -- coalesce_or_enqueue_merge_request's duplicate_in_verify
      reject (merge_queue.py, reject code _C3_DUPLICATE_IN_VERIFY_CODE): a
      newer SHA submitted while the earlier SHA is IN VERIFY is structurally
      REJECTed, not coalesced or replaced.
  epsilon (2928) -- retire_cancelled_merge_request (see
      test_merge_cancel_retire.py): a merge_cancel FULLY retires the
      cancelled entry (registry slot + worktree + sticky retention) before
      returning, so an immediate resubmit gets a fresh, uncorrupted slot.

This file is the ONE NEW test file the done-gate adds -- a TEST-ONLY
COMPOSITION gate exercising all five legs, alone and together, across
Sec.9 boundary rows 1-9:

  1. A live-leased persistent `_merge-verify` survives a concurrent
     crash-recovery sweep (C2 skip-by-name; C1 futureproofs the lease).
  2. A live-leased ephemeral `_merge-<uuid>` survives the same sweep.
  3. A DEAD-holder ephemeral fails OPEN (guarded removal succeeds); a
     LIVE-held ephemeral is skipped with exactly one WARNING naming the
     holder pgid + reason.
  4. Non-merge infra bands (`.reseed-trash`, `_mainprobe-x`, ...) are left
     to their owner by the SAME sweep that cleans a task-shaped planless
     dir (the positive control proving the sweep is not inert).
  5. (capstone) the live verify observes its own worktree intact across the
     concurrent sweep -- see the capstone class docstring for the exact
     zero-ENOENT causal-proxy chain; this row has no standalone test.
  6. Two journal entries for one branch with the SAME snapshot tip
     collapse to ONE enqueued winner; the loser's future mirrors the
     winner's terminal outcome (OBSERVED, not inferred).
  7. Two journal entries for one branch with ancestor/descendant tips
     collapse to the DESCENDANT, order-independently.
  8. A newer SHA submitted while the branch's earlier SHA is IN VERIFY is
     structurally REJECTed (`duplicate_in_verify`) -- the live entry is
     left undisturbed.
  9. A cancelled merge is FULLY retired (slot + worktree + sticky) before
     an immediate resubmit, which gets a genuinely fresh entry rather than
     coalescing onto the retired corpse.

Row 10 (the C4 concurrent-local-verify serial-lane telemetry tripwire) is
OUT OF SCOPE for this gate -- it belongs to task eta (a separate rider
leaf, PRD Sec.8/Sec.9), so it is not exercised here.

Concurrency model -- READ THIS BEFORE editing test bodies
-----------------------------------------------------------
Harness.run()'s two startup recovery entry points, `_recover_pending_merges`
(run() step 1c0a) and `_recover_crashed_tasks` (run() step 2c), are
SEQUENTIAL awaits -- NOT gathered/parallelized. The
2026-07-22 task/5326 incident's concurrency was the pre-launched merge-worker
BACKGROUND TASK (step 1b, `_start_merge_worker` -> create_task) draining the
re-enqueued `_merge_queue` WHILE the crash-recovery sweep scanned worktrees.
The capstone below reproduces this exactly: it starts a REAL merge worker
with a gated `run_scoped_verification` (holding a verify live in its own
`_merge-<hash>` tree) BEFORE awaiting the sweep. Do NOT add an assertion
that `_recover_pending_merges`/`_recover_crashed_tasks` run concurrently
with EACH OTHER -- that would fail against current code and misrepresent
the design (PRD D6: no startup reordering; C1+C2 make the ordering
irrelevant for this class of bug).

'Zero ENOENT' is a failure MODE, not a matchable token
----------------------------------------------------------
The incident's `Error: ENOENT ... uv_cwd` signature appears only in PRD
prose -- it is not a FailureCategory, not an EventType, and is not asserted
anywhere in the tree via string match. It is proved via CAUSAL PROXIES
instead:
  (a) every `_merge-*`/infra tree survives the concurrent sweep, with each
      survival ATTRIBUTED to the specific mechanism that protects it (see
      the attribution matrix below) and each leg carrying its own positive
      control -- survival alone is not evidence, since an inert sweep
      produces the same `.exists()`;
  (b) the gated verify runner asserts its OWN `_merge-<hash>` cwd worktree
      exists at entry AND at completion -- a tree yanked mid-verify would
      fail this assertion, directly modelling the incident;
  (c) the recovered merge reaches `outcome.status == 'done'` and its
      branch lands on main ('merge finalizes');
  (d) zero spurious `verify_cross_check_mismatch` L1 escalations are filed
      (the incident's clobbered-worktree false-FAIL signature).

Attribution matrix for proxy (a)
------------------------------------
The capstone drives TWO sweeps, and a tree that survives one of them is
NOT thereby protected by the other's mechanism. Each is asserted through
the evidence that is actually load-bearing for that leg:

  MERGE-REAPER leg (`_reap_orphaned_merge_worktrees`), evidence = the
  `(path, reason, outcome)` records of a delegating spy over the C1
  primitive, plus the filesystem:
    * `_merge-verify`  -- never reaches C1 at all: excluded by name
      (PERSISTENT_MERGE_WORKTREE_NAME) inside the reap scan.
    * the live `_merge-<hash>` verify tree -- protected by the worker's
      owned-worktree ledger and/or the untouched grace window; only its
      `.exists()` is asserted (its re-adoption path is NOT asserted --
      `find_inflight_merge_worktree` on a detached-HEAD merge tree is
      unverified here).
    * `_merge-cafe5326` (aged, LEASED) -- the ONLY C1-attributable
      survival on this leg: OFFERED to C1, which answered
      `skipped_lease_held`.
    * the aged UNLEASED REAL ephemeral `_merge-<uuid>` worktree --
      POSITIVE CONTROL: offered to C1, which answered `removed`, and
      gone from disk. Both are pinned, and the control is a genuinely
      REGISTERED git worktree on purpose: C1 answers `failed` for a
      merely `mkdir()`'d directory (its pinned contract) and the tree
      then vanishes via cleanup_merge_worktree's rmtree FALLBACK, so a
      fake-worktree control would stay green even if C1's uncontended
      removal branch regressed -- the exact branch the leased decoy's
      refusal is contrasted against.
    Both merge-band decoys are BACK-DATED past
    RESOURCE_AUDIT_WORKTREE_GRACE_SECS first; without that the reap loop
    `continue`s past them and the whole leg is vacuous.

  CRASH-RECOVERY-SWEEP leg (`_recover_crashed_tasks`), evidence = the SET
  of paths offered to `cleanup_worktree`, NOT `.exists()`:
    * every `_merge-*` / infra entry survives by the C2
      `classify_worktree_entry` skip, proven by
      `cleaned_paths.isdisjoint(protected)` plus an UPPER-BOUND pin
      (`cleaned_paths <= {the two task-id-shaped entries}`, so any extra
      offered path fails; the bound is deliberately not an equality --
      see the pin's own comment for why freezing the live merge's source
      worktree as REQUIRED would be pinning a mock artifact).
    * the task-shaped planless '999' dir is the POSITIVE CONTROL the same
      sweep cleans.
    DELIBERATE NON-FIX: `cleanup_worktree` stays an AsyncMock spy on this
    leg -- a real delegate would race the in-flight merge by deleting the
    live task/5326 worktree mid-verify. That makes `.exists()` INERT here
    (a full C2 regression would leave every tree on disk), so the
    cleaned-SET assertions are this leg's only regression detector. They
    were OBSERVED to fire: neutering the C2 `merge` arm in a scratch,
    uncommitted edit turned them red, naming `_merge-verify`,
    `_merge-cafe5326` and the live `_merge-<hash>` tree.

SCOPE -- TEST-ONLY / BEHAVIOUR-FROZEN
----------------------------------------
Every production surface exercised below (alpha-epsilon, tasks 2924-2928)
already SHIPPED and is frozen for this batch. This is a COMPOSITION gate:
it wires already-landed callables together and asserts their combined
behaviour. If a scenario surfaces a GENUINE production defect, ESCALATE
(category='design_concern' or 'scope_violation') rather than editing
production here -- editing frozen production from this task would widen
the concurrency lock on the hottest files in the repo (harness.py,
merge_queue.py) and conflict with the frozen seam the prerequisite tasks
already landed.

NO LINE-OFFSET CITATIONS
----------------------------
Nothing in this file cites a `module.py:NNNN` offset, deliberately: an
offset into harness.py / merge_queue.py / git_ops.py is stale within days
of being written, and a citation a reader has to re-verify costs more than
it saves. Every reference above and below names the SYMBOL (function,
method, class, constant, or pytest node id) instead -- locate it with
grep/search. Keep it that way when editing: cite by name, never by line.

PROVENANCE -- these row classes are PORTS, not originals
------------------------------------------------------------
Only :class:`TestFiveThreeTwoSixReplayGate` is new behaviour coverage (it
is the only test that composes the five legs under the incident's actual
concurrency). Every other class here is a deliberate PORT of an existing
unit test, re-homed so the PRD Sec.9 row matrix is legible and executable
as ONE gate rather than scattered across five modules:

  row(s) | test in this file                          | ported from
  -------+--------------------------------------------+----------------------
  1,2,4  | TestDeleterFace::                          | test_crash_recovery.py::
         |   test_merge_and_infra_trees_survive_      |   TestRecoverCrashedTasksC2Namespace::
         |   sweep_task_shaped_cleaned                |   test_infra_and_merge_survive_sweep_only_task_shaped_cleaned
  3      | TestDeleterFace::                          | test_remove_merge_worktree_guarded.py::
         |   test_dead_holder_fails_open_live_        |   the live-held-skip / dead-holder
         |   holder_skips                             |   fail-open pair
  6,7    | TestIdentityFaceRecoveryDedupe (x3)        | test_merge_queue_store.py::
         |                                            |   TestRecoverPendingMergesRegistryDedup
  8      | TestIdentityFaceInVerifyReject             | test_merge_queue_c3_submit_identity.py::
         |                                            |   TestC3SubmitGateInVerify::test_in_verify_newer_sha_rejects
  9      | TestIdentityFaceCancelResubmit             | test_merge_cancel_retire.py::
         |                                            |   TestRetireCancelledMergeRequest (basic-retire
         |                                            |   + identity-guard)
  5      | (capstone only -- no standalone test)      | (new)

MAINTENANCE CONSEQUENCE, stated so it is never a surprise: a production
behaviour change on any of these legs turns BOTH this file and the module
in the right-hand column red, and both must be updated. That cost is
accepted on purpose -- the PRD's done gate is defined as a single readable
Sec.9 row matrix -- but if you are here because a change made two files
red, the right-hand column is where the ORIGINAL unit-level contract
lives; treat this file as the composition/traceability layer over it, and
do not deepen a port here that would be better expressed upstream. The
MEASURED split below tells you which assertions in each port are safe to
re-sync mechanically from that column versus which are original
contracts you must re-derive by hand. This acceptance is a recorded
DECISION, not an assumption: see D9 and the `G7 waiver:
no-lockstep-duplication` in docs/prds/merge-worktree-lifecycle-integrity.md
Sec.5.

MEASURED duplicate-vs-unique split (per row) -- vs upstream, vs capstone
----------------------------------------------------------------------------
The PROVENANCE table above says WHERE each row was ported from; it does
not say WHICH assertions duplicate that origin (safe to drop) versus the
gate's own capstone (:class:`TestFiveThreeTwoSixReplayGate`, load-bearing
HERE). Conflating the two is how row 3's port silently DROPPED an
upstream assertion (repaired 2026-07-30, task 3153) while row 9's port
silently GAINED unique coverage -- neither fact was written down until
now. Measured by a full assertion-level diff, not estimated:

  rows 1,2,4 | vs upstream: FULLY SUBSUMED -- same assertions, same
             |   failure-message strings, same mocking depth
             |   (cleanup_worktree / quarantine_worktree /
             |   _is_registered_worktree are AsyncMocks in BOTH; the
             |   gate's real-git repo is inert on the C2 arm, which is
             |   pure in-process classification).
             | vs capstone: NOT subsumed -- the capstone asserts survival,
             |   cleaned_paths.isdisjoint(protected) and the upper-bound
             |   pin, but NEVER the INFO skip-REPORTING (no info_messages
             |   name loop). LOAD-BEARING HERE.
  3          | vs upstream: PARTIAL -- the gate ADDS a stale-.lock file
             |   plus a guaranteed-dead pgid setup (upstream's dead tree
             |   has no lock file at all) and a logger-scoped WARNING
             |   count; it had DROPPED upstream's no-skip-WARNING-on-
             |   fail-open assertion, restored 2026-07-30 (task 3153).
             |   Upstream's 8 sibling outcome tests (skipped_persistent
             |   x2, not_present, failed, lock-unlink/preserve, the
             |   CleanupMergeWorktreeRouting pair) are deliberately NOT
             |   ported.
             | vs capstone: PARTIAL -- the capstone re-covers only the
             |   skipped_lease_held / removed OUTCOMES via the reaper
             |   spy, never the dead-holder fail-open and never the
             |   WARNING count/pgid/reason. LOAD-BEARING HERE.
  6,7        | vs upstream: FULLY SUBSUMED -- verbatim, down to the
             |   'order-independent: the descendant wins regardless of
             |   journal order' message; git_ops is a MagicMock in BOTH,
             |   so the gate is not "more real" than its origin.
             | vs capstone: NOT subsumed -- the capstone asserts
             |   recovered/coalesced/len(requests)/winner-is-descendant/
             |   qsize but NEVER the peer-future MIRROR (no waiters
             |   assertion, no peer .result()) and drives only ONE
             |   journal order. LOAD-BEARING HERE.
  8          | vs upstream: FULLY SUBSUMED -- 11 of 11 assertions
             |   identical, in order. NOTE: the gate passes
             |   event_store=None where upstream passes a real
             |   EventStore; this is DELIBERATE and INERT --
             |   coalesce_or_enqueue_merge_request's duplicate_in_verify
             |   reject returns BEFORE the function's only
             |   event_store.emit call, on a different branch, so a real
             |   EventStore is never touched on this path. Do not "fix" it.
             | vs capstone: not covered at all. SOLE COVERAGE IN THIS FILE.
  9          | vs upstream: PARTIAL, and the ONE port with genuinely NEW
             |   detection -- the retirement -> IMMEDIATE-resubmit
             |   composition driven through the production entry point
             |   coalesce_or_enqueue_merge_request WITH the retention ring
             |   is absent upstream, which never composes the two and
             |   never passes retention= to a submit call. Segments A
             |   (full retirement) and C (late stale retirement /
             |   identity guard) ARE verbatim ports.
             | vs capstone: not covered at all. SOLE COVERAGE IN THIS FILE.

Practical upshot for the next production change that turns two files red:
rows 1,2,4,6,7,8's "FULLY SUBSUMED" halves can be re-synced MECHANICALLY
from the upstream node id named in the PROVENANCE table above. Row 3's
capstone-delta and row 9's retention-composition are ORIGINAL contracts
with no upstream analogue -- re-derive them by hand; do not assume
re-copying the upstream diff covers them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.lane_lifecycle import LaneLifecycle
from orchestrator.merge_queue import (
    SpeculativeMergeWorker,
    coalesce_or_enqueue_merge_request,
    retire_cancelled_merge_request,
)
from orchestrator.merge_queue_store import MergeQueueStore, recover_pending_merges
from orchestrator.merge_types import (
    InFlightMergeRegistry,
    MergeRequest,
    QueuedBranch,
    TerminalOutcomeRecord,
    TerminalOutcomeRetention,
)
from orchestrator.verify_cancel import (
    acquire_merge_verify_flock,
    lane_lock_path,
    release_merge_verify_flock,
    remove_lock_holder_pgid,
    write_lock_holder_pgid,
)

#: A pgid guaranteed to be dead: os.killpg on this must raise
#: ProcessLookupError (Linux pid_max is nowhere near 2**31-1). Mirrors
#: test_merge_verify_lease_guard.py's _DEAD_PGID (per-file duplication
#: convention).
_DEAD_PGID = 2**31 - 1


def _git_ops_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """WARNING messages emitted by git_ops' OWN logger during the current capture.

    Both legs of row 3 need the same filter, and `caplog.at_level(logger=...)`
    does NOT filter `caplog.records` -- it only lowers the level for that logger,
    so records from every other propagated logger still land in the list. Scoping
    by `r.name` is what makes a count assertion a statement about the contract
    under test rather than about global session log noise.
    """
    return [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING and r.name == 'orchestrator.git_ops'
    ]


# ---------------------------------------------------------------------------
# Real-git fixtures (adapted from test_remove_merge_worktree_guarded.py /
# test_crash_recovery.py's harness fixture, per-file duplication convention)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a git repo with a single commit (README.md) on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _head_sha(repo: Path) -> str:
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0
    return out.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


async def _make_ephemeral_worktree(git_ops: GitOps) -> Path:
    """Build a real ephemeral ``_merge-<uuid>`` worktree at the repo's HEAD.

    Ported from test_remove_merge_worktree_guarded.py -- remove_merge_worktree_
    guarded's 'removed'/'failed' outcome split is only meaningful against a
    REAL registered git worktree (a plain ``mkdir()``'d directory always
    yields 'failed' -- see that module's test_non_worktree_directory_returns_failed).
    """
    return await git_ops.create_throwaway_verify_worktree(await _head_sha(git_ops.project_root))


# ---------------------------------------------------------------------------
# Recovery-harness factory (ported from test_crash_recovery.py's ``harness``
# fixture) -- a plain factory function (not a fixture) so the
# capstone can attach additional real components (merge store/registry/
# worker) after construction without a second fixture indirection layer.
# ---------------------------------------------------------------------------


def _build_recovery_harness(mock_orch_config: MagicMock, git_repo: Path) -> Harness:
    """Build a Harness wired for crash-recovery / merge-reap composition tests.

    McpLifecycle/Scheduler/BriefingAssembler are patched at construction so
    no fused-memory/live-scheduler machinery starts. ``harness.git_ops`` is
    then REBOUND to a REAL GitOps over *git_repo* (a real git-initialized
    repo, decoupled from ``mock_orch_config.project_root``) so real git
    worktree/lease operations succeed; the scheduler is replaced with a bare
    MagicMock exposing exactly the async surface ``_recover_crashed_tasks``
    consults.
    """
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.get_task = AsyncMock(return_value={})
    h.scheduler.get_status = AsyncMock(return_value=None)
    h.scheduler._dispatched = set()
    h.scheduler.is_deterministic = MagicMock(return_value=False)

    recovery_git_config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    h.git_ops = GitOps(recovery_git_config, git_repo)
    h.git_ops.worktree_base = (git_repo / '.worktrees').resolve()
    h.git_ops.mark_pool_storage_present()
    h.git_ops.cleanup_worktree = AsyncMock()
    h.git_ops.quarantine_worktree = AsyncMock(return_value=None)
    # GitOps.__init__ built _lane_lifecycle against the ORIGINAL
    # worktree_base (before the reassignment above) -- rebind it so the
    # record-driven recovery path reads/writes the same .lane-state dir the
    # rest of this harness targets (mirrors test_crash_recovery.py's W11 fix).
    h.git_ops._lane_lifecycle = LaneLifecycle(
        h.git_ops.worktree_base, quarantine_worktree=h.git_ops.quarantine_worktree,
    )
    h.git_ops._is_registered_worktree = AsyncMock(return_value=True)
    h.event_store = MagicMock()

    return h


# ---------------------------------------------------------------------------
# Planting helpers -- build worktree_base entries in the various dispositions
# the deleter face must classify (leased / dead-holder / infra / task-shaped).
# ---------------------------------------------------------------------------


def _plant_leased_tree(base: Path, path: Path) -> int:
    """Ensure *path* exists and hold a LIVE merge-verify lease on it.

    Records the live holder pgid at the fixed rendezvous key so a
    remove_merge_worktree_guarded skip WARNING can name it. Returns the
    held fd -- release via ``release_merge_verify_flock(fd)`` (and
    ``remove_lock_holder_pgid(base)`` once no other lease is live) when done.
    """
    path.mkdir(parents=True, exist_ok=True)
    fd = acquire_merge_verify_flock(lane_lock_path(path), 5.0)
    assert fd is not None, f'test setup: must be able to acquire the {path.name} lease'
    write_lock_holder_pgid(base, os.getpgrp())
    return fd


def _plant_dead_holder_tree(base: Path, path: Path) -> None:
    """Ensure *path* exists with a STALE lease: acquire then immediately
    release its own flock (leaving a stale ``<path>.lock`` file with no
    live holder -- the kernel already auto-released the advisory lock) and
    record a guaranteed-dead pgid (``_DEAD_PGID``) at the fixed rendezvous
    key -- the fail-open positive control proving removal gates on the
    flock itself, never on the best-effort pgid rendezvous file.
    """
    path.mkdir(parents=True, exist_ok=True)
    fd = acquire_merge_verify_flock(lane_lock_path(path), 5.0)
    assert fd is not None, f'test setup: must be able to acquire the {path.name} lease'
    release_merge_verify_flock(fd)
    write_lock_holder_pgid(base, _DEAD_PGID)


def _plant_infra_dir(base: Path, name: str) -> Path:
    """Create a plain (unleased) infra-band directory under *base*."""
    path = base / name
    path.mkdir(parents=True)
    return path


def _plant_task_dir(base: Path, task_id: str) -> Path:
    """Create a plain task-id-shaped, planless directory under *base*."""
    path = base / task_id
    path.mkdir(parents=True)
    return path


def _age_tree(path: Path, age_secs: float) -> None:
    """Back-date *path*'s mtime by *age_secs* seconds.

    ``SpeculativeMergeWorker.reap_orphaned_merge_worktrees`` only offers a
    ``_merge-*`` candidate to ``GitOps.cleanup_merge_worktree`` (and thus
    to the C1 guarded-removal primitive) once
    ``now - entry.stat().st_mtime`` exceeds
    :attr:`SpeculativeMergeWorker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS`
    (9000.0) -- the register-after-create race guard. A freshly-planted
    decoy is therefore ``continue``d past before any removal machinery
    runs, which would make a "survives the reaper" assertion VACUOUS.
    Pushing the mtime past that window makes the reap loop actually reach
    the primitive, so the outcome it returns is what the assertion pins.

    Only the directory's own mtime matters: the lane ``.lock`` flock file
    is a SIBLING of the tree (``lane_lock_path``), so planting a lease
    neither creates nor touches anything inside *path*.
    """
    stamp = time.time() - age_secs
    os.utime(path, (stamp, stamp))


class _GuardedRemovalSpy:
    """Recording DELEGATE over ``GitOps.remove_merge_worktree_guarded``.

    Mirrors the :class:`_RetireSpy` idiom (see
    test_merge_cancel_retire.py's git_ops spy) but with one deliberate
    difference: it does NOT fake an outcome -- it awaits the REAL bound
    method and records what the production primitive actually answered.
    The point of the merge-reaper leg is attribution ("this tree survived
    BECAUSE C1 refused it"), which a faked outcome would destroy.

    Installed as an INSTANCE attribute on the live GitOps object, so it
    shadows the class method and therefore also captures
    ``cleanup_merge_worktree``'s internal
    ``self.remove_merge_worktree_guarded(...)`` call -- the only path by
    which the reaper reaches C1.

    ``records`` accumulates ``(path, reason, outcome)`` per call, with the
    path resolved so it compares equal to a ``worktree_base / name`` the
    test built (``worktree_base`` is itself resolved).
    """

    def __init__(self, inner: Callable[..., Awaitable[str]]) -> None:
        self._inner = inner
        self.records: list[tuple[Path, str, str]] = []

    async def __call__(self, path: Path, *, reason: str) -> str:
        outcome = await self._inner(path, reason=reason)
        self.records.append((Path(path).resolve(), reason, outcome))
        return outcome

    def outcomes_for(
        self, path: Path, *, records: list[tuple[Path, str, str]] | None = None,
    ) -> list[str]:
        """Every outcome C1 returned for *path*, in call order.

        *records* accepts a caller-taken SNAPSHOT of :attr:`records` so an
        assertion about one sweep is not contaminated by removals a
        concurrently-running merge worker drove afterwards.
        """
        source = self.records if records is None else records
        target = Path(path).resolve()
        return [outcome for p, _reason, outcome in source if p == target]


# ---------------------------------------------------------------------------
# TestDeleterFace -- PRD Sec.9 rows 1-4
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDeleterFace:
    """PRD Sec.9 rows 1-4: the crash-recovery sweep's deleter face.

    (a) rows 1, 2, 4 + positive control -- Harness._recover_crashed_tasks()
    must SKIP+REPORT every `_merge-*`/infra tree (never remove one
    directly; that is the merge reaper's job) while still cleaning a
    task-shaped planless dir in the SAME pass (the positive control
    proving the sweep is not inert). Ported from
    test_crash_recovery.py::TestRecoverCrashedTasksC2Namespace.

    (b) row 3 -- GitOps.remove_merge_worktree_guarded's dead-holder
    fail-open contrasted with row 1/2's live-held skip, on REAL ephemeral
    git worktrees (remove_merge_worktree_guarded's 'removed'/'failed'
    outcomes are only meaningful against a real registered worktree; a
    plain ``mkdir()``'d directory always returns 'failed'). Ported from
    test_remove_merge_worktree_guarded.py's live-held-skip / dead-holder
    fail-open pair.
    """

    async def test_merge_and_infra_trees_survive_sweep_task_shaped_cleaned(
        self,
        mock_orch_config: MagicMock,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Rows 1, 2, 4 + positive control: protected trees survive the
        sweep with an explicit INFO skip/report line each; the task-shaped
        planless dir is the ONLY cleanup_worktree call."""
        harness = _build_recovery_harness(mock_orch_config, git_repo)
        base = harness.git_ops.worktree_base

        merge_verify = base / '_merge-verify'
        fd_verify = _plant_leased_tree(base, merge_verify)
        merge_uuid = base / '_merge-ba97f10a'
        fd_uuid = _plant_leased_tree(base, merge_uuid)

        infra_dirs = {
            name: _plant_infra_dir(base, name)
            for name in (
                '.reseed-trash', '_mainprobe-x', '.lane-state',
                '.task-meta', '_offline-deep',
            )
        }

        wt_task = _plant_task_dir(base, '999')

        try:
            with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
                await harness._recover_crashed_tasks()
        finally:
            release_merge_verify_flock(fd_verify)
            release_merge_verify_flock(fd_uuid)
            remove_lock_holder_pgid(base)

        # Positive control: the ONLY cleanup_worktree call is the
        # task-shaped planless dir -- any merge/infra cleanup call would
        # push the count past one (the 5326 "Cleaned up worktree
        # _merge-verify" regression).
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt_task, '999')  # type: ignore[attr-defined]

        cleaned_paths = {
            c.args[0] for c in harness.git_ops.cleanup_worktree.call_args_list  # type: ignore[attr-defined]
        }
        protected = {merge_verify, merge_uuid, *infra_dirs.values()}
        assert cleaned_paths.isdisjoint(protected), (
            f'C2 violated -- sweep cleaned protected entries: '
            f'{cleaned_paths & protected}'
        )
        for d in protected:
            assert d.exists(), f'{d.name} must survive the recovery sweep'

        # Skip disposition OBSERVED (not silence): every protected entry is
        # named in an explicit INFO record.
        info_messages = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.INFO
        ]
        for name in ('_merge-verify', '_merge-ba97f10a', '_mainprobe-x',
                     '_offline-deep', '.reseed-trash', '.lane-state',
                     '.task-meta'):
            assert any(name in m for m in info_messages), (
                f'missing explicit skip/report line naming {name}'
            )

    async def test_dead_holder_fails_open_live_holder_skips(
        self,
        git_ops: GitOps,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Row 3: a dead/stale lease holder never wedges removal (fail
        open), contrasted with a genuinely live holder (skip, single
        WARNING naming pgid + reason), on real ephemeral merge worktrees."""
        dead_wt = await _make_ephemeral_worktree(git_ops)
        live_wt = await _make_ephemeral_worktree(git_ops)
        base = git_ops.worktree_base

        _plant_dead_holder_tree(base, dead_wt)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            outcome_dead = await git_ops.remove_merge_worktree_guarded(dead_wt, reason='reaper')
        assert outcome_dead == 'removed', (
            'a stale holder-pgid record with no live flock must fail OPEN'
        )
        assert not dead_wt.exists()
        assert _git_ops_warnings(caplog) == [], (
            'fail-open must not emit a skip WARNING: a stale holder-pgid record is not '
            'contention. Restored from test_remove_merge_worktree_guarded.py::'
            'test_dead_holder_pgid_fails_open_and_removes, which the row-3 port dropped.'
        )

        fd_live = _plant_leased_tree(base, live_wt)
        try:
            # Clear first: `caplog.records` accumulates across the whole capture
            # phase, so without this the dead-holder removal above would leak
            # into this call's WARNING count. `_git_ops_warnings` (module-level,
            # above) does the per-logger scoping.
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
                outcome_live = await git_ops.remove_merge_worktree_guarded(live_wt, reason='reaper')

            assert outcome_live == 'skipped_lease_held', (
                'a LIVE lease holder must skip removal, never force through'
            )
            assert live_wt.exists(), 'a live lease holder must leave the tree intact'

            warnings = _git_ops_warnings(caplog)
            assert len(warnings) == 1, (
                f'expected exactly one orchestrator.git_ops WARNING, got '
                f'{len(warnings)}: {warnings}'
            )
            assert str(os.getpgrp()) in warnings[0], warnings[0]
            assert 'reaper' in warnings[0], warnings[0]
        finally:
            release_merge_verify_flock(fd_live)
            remove_lock_holder_pgid(base)


# ---------------------------------------------------------------------------
# Recovery-dedupe seed builders (rows 6-7) -- ported from
# test_merge_queue_store.py::TestRecoverPendingMergesRegistryDedup
# (_make_req / _make_git_ops).
# ---------------------------------------------------------------------------


def _seed_dup_journal(
    store: MergeQueueStore,
    branch: str,
    tips: list[str],
    config: OrchestratorConfig,
    worktree: Path,
) -> list[MergeRequest]:
    """Record ``len(tips)`` PersistedMergeRequest rows for *branch* on
    *store*, one per tip in *tips* (distinct auto-generated request_ids;
    journal insertion order == *tips* order).

    Returns the seed MergeRequest objects built along the way -- their
    ``make_placeholder_future()`` ``.result`` futures are throwaway:
    recover_pending_merges' Phase 2 reconstructs fresh live MergeRequests
    via ``reconstruct_merge_request`` bound to the REAL running loop, so
    only each seed's ``.request_id`` is meaningful after seeding (safe
    despite these async test bodies -- see make_placeholder_future's
    docstring caveat, which applies to a future that is itself awaited/
    resolved, not to one that is merely a throwaway identity carrier).
    """
    reqs = []
    for tip in tips:
        req = MergeRequest(
            task_id=branch,
            branch=QueuedBranch.parse(branch, config.git.branch_prefix),
            worktree=worktree,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=make_placeholder_future(),
            snapshot_tip=tip,
        )
        store.record(req)
        reqs.append(req)
    return reqs


def _make_git_ops(
    *,
    full_branch: str,
    branch_sha: str = 'sha-live',
    ancestor_pairs: set[tuple[str, str]] | None = None,
) -> MagicMock:
    """Fake git_ops for the recovery-dedupe tests (rows 6-7).

    * ``resolve_branch_sha(full_branch)`` -> *branch_sha* (survives Phase 1
      of recover_pending_merges), None for any other ref.
    * ``is_ancestor(a, b)`` -> True iff ``(a, b)`` in *ancestor_pairs*. The
      survival check ``is_ancestor(full_branch, 'main')`` is therefore
      False (branch not yet landed) unless that exact pair is supplied,
      and the Phase-2 tip classification is driven entirely by the
      snapshot-tip pairs the caller supplies.
    """
    pairs = ancestor_pairs if ancestor_pairs is not None else set()

    async def fake_resolve(branch: str) -> str | None:
        return branch_sha if branch == full_branch else None

    async def fake_is_ancestor(ancestor: str, descendant: str) -> bool:
        return (ancestor, descendant) in pairs

    git_ops = MagicMock()
    git_ops.resolve_branch_sha = fake_resolve
    git_ops.is_ancestor = fake_is_ancestor
    return git_ops


# ---------------------------------------------------------------------------
# TestIdentityFaceRecoveryDedupe -- PRD Sec.9 rows 6-7
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestIdentityFaceRecoveryDedupe:
    """PRD Sec.9 rows 6-7: recover_pending_merges' registry-gated per-branch
    collapse (gamma, task 2926). Ported from
    test_merge_queue_store.py::TestRecoverPendingMergesRegistryDedup.

    (row 6) Two journal entries for one branch with the SAME snapshot tip
    collapse to ONE enqueued winner; the loser attaches as a peer whose
    future MIRRORS the winner's terminal outcome (OBSERVED, not inferred).

    (row 7) Two journal entries with ancestor/descendant tips collapse to
    the DESCENDANT, order-independently (both journal insertion orders are
    asserted).
    """

    async def test_same_sha_coalesces_to_one_with_peer_mirror(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """Row 6: same-SHA duplicate journal entries -> ONE winner, peer
        future mirrors the winner's terminal outcome."""
        wt = tmp_path / 'wt'
        wt.mkdir()
        store = MergeQueueStore(tmp_path / 'merge_queue.json')
        reqs = _seed_dup_journal(store, '5326', ['sha-same', 'sha-same'], config, wt)

        registry = InFlightMergeRegistry()
        git_ops = _make_git_ops(full_branch='task/5326')
        queue: asyncio.Queue = asyncio.Queue()

        report = await recover_pending_merges(
            store, queue, git_ops, config, event_store=None,
            main_branch='main', branch_prefix='task/', registry=registry,
        )

        assert queue.qsize() == 1
        assert report['recovered'] == 1
        assert report['coalesced'] == 1

        entry = registry.entry('5326')
        assert entry is not None
        assert len(entry.waiters) == 2, (
            f'Expected primary+peer waiters; got {len(entry.waiters)}'
        )

        # First-seen wins the SAME tie -> reqs[0] is the enqueued winner.
        assert len(report['requests']) == 1
        winner_req = report['requests'][0]
        assert winner_req.request_id == reqs[0].request_id

        # Grab the PEER future BEFORE resolving the winner.
        peer_futures = [
            w.future for w in entry.waiters if w.future is not winner_req.result
        ]
        assert len(peer_futures) == 1
        peer = peer_futures[0]
        assert not peer.done()

        # Resolving the winner mirrors the terminal outcome onto the peer:
        # both requesters resolve -- the coalesce attach is OBSERVED.
        sentinel = object()
        winner_req.result.set_result(sentinel)
        await asyncio.sleep(0)
        assert peer.done()
        assert peer.result() is sentinel

        # The loser's journal entry is removed; the winner stays journaled.
        remaining_ids = {r.request_id for r in store.load()}
        assert reqs[1].request_id not in remaining_ids, 'loser must be store.remove()d'
        assert reqs[0].request_id in remaining_ids, 'winner stays journaled'

    async def test_descendant_wins_ancestor_first(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """Row 7: journal order [ancestor, descendant] -> the DESCENDANT is
        the single enqueued winner (REPLACE)."""
        wt = tmp_path / 'wt'
        wt.mkdir()
        store = MergeQueueStore(tmp_path / 'merge_queue.json')
        reqs = _seed_dup_journal(store, '5326', ['anc', 'desc'], config, wt)

        registry = InFlightMergeRegistry()
        git_ops = _make_git_ops(full_branch='task/5326', ancestor_pairs={('anc', 'desc')})
        queue: asyncio.Queue = asyncio.Queue()

        report = await recover_pending_merges(
            store, queue, git_ops, config, event_store=None,
            main_branch='main', branch_prefix='task/', registry=registry,
        )

        assert queue.qsize() == 1
        assert report['coalesced'] == 1
        enqueued = queue.get_nowait()
        assert enqueued.request_id == reqs[1].request_id, (
            'the DESCENDANT record must be the single enqueued winner'
        )

    async def test_descendant_wins_descendant_first(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """Row 7: journal order [descendant, ancestor] -> still the
        DESCENDANT is enqueued (order-independence: the pre-grouping picks
        the descendant-most tip regardless of journal insertion order)."""
        wt = tmp_path / 'wt'
        wt.mkdir()
        store = MergeQueueStore(tmp_path / 'merge_queue.json')
        reqs = _seed_dup_journal(store, '5326', ['desc', 'anc'], config, wt)

        registry = InFlightMergeRegistry()
        git_ops = _make_git_ops(full_branch='task/5326', ancestor_pairs={('anc', 'desc')})
        queue: asyncio.Queue = asyncio.Queue()

        report = await recover_pending_merges(
            store, queue, git_ops, config, event_store=None,
            main_branch='main', branch_prefix='task/', registry=registry,
        )

        assert queue.qsize() == 1
        assert report['coalesced'] == 1
        enqueued = queue.get_nowait()
        assert enqueued.request_id == reqs[0].request_id, (
            'order-independent: the descendant wins regardless of journal order'
        )


# ---------------------------------------------------------------------------
# Gated tree-liveness verify runner (capstone concurrency driver) -- holds a
# verify LIVE in its own worktree while the test drives the concurrent
# crash-recovery sweep, mirroring the 2026-07-22 task/5326 incident's actual
# concurrency (a pre-launched merge-worker background task draining the
# re-enqueued queue WHILE the sweep scans worktrees -- see the module
# docstring's "Concurrency model" section).
# ---------------------------------------------------------------------------


def _gated_tree_liveness_verify(
    entered: asyncio.Event,
    release: asyncio.Event,
    observations: list[bool],
):
    """Build a ``run_scoped_verification`` stand-in that holds ONE verify
    live in its own ``_merge-<hash>`` worktree across a concurrent sweep.

    Returns an async callable matching ``run_scoped_verification``'s call
    shape (``worktree`` first positional; everything else absorbed via
    ``*args``/``**kwargs`` so it tolerates the production call site's exact
    kwarg set drifting -- mirrors test_merge_queue_restart_hook.py's
    ``_mock_verify_pass`` duck-typed-VerifyResult idiom). Each call:

      1. Appends ``worktree.exists()`` to *observations* -- entry liveness.
      2. Sets *entered* so the test can synchronize past this point.
      3. Awaits *release* -- holds the verify live while the test drives the
         concurrent crash-recovery sweep.
      4. Appends ``worktree.exists()`` again -- exit liveness.  A tree
         yanked mid-verify by a concurrent sweep would record a ``False``
         here, directly modelling the 2026-07-22 ENOENT incident.

    Returns a duck-typed passing ``VerifyResult`` (``passed=True``).
    """

    async def _verify(worktree: Path, *args: object, **kwargs: object) -> object:
        observations.append(worktree.exists())
        entered.set()
        await release.wait()
        observations.append(worktree.exists())
        return type(
            'VR', (), {'passed': True, 'summary': '', 'failing_test_ids': None},
        )()

    return _verify


# ---------------------------------------------------------------------------
# TestFiveThreeTwoSixReplayGate -- capstone: PRD Sec.9 rows 1, 2, 4, 5, 6, 7
# end-to-end + 'merge finalizes' + the zero-ENOENT causal-proxy chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(180)  # heavy class: real git + real merge worker end-to-end
class TestFiveThreeTwoSixReplayGate:
    """The headline done-gate: replays the 2026-07-22 task/5326 restart
    incident end-to-end, driving the ACTUAL startup substrate (real
    GitOps, real MergeQueueStore, real InFlightMergeRegistry, real
    SpeculativeMergeWorker) rather than any single leg in isolation.

    Exercises PRD Sec.9 rows 1, 2, 4 (protected trees survive BOTH
    concurrent sweeps -- each survival attributed to the mechanism that
    actually protects it, per the matrix below), 6 (dup-journal
    same-branch collapse; here a descendant-tip variant), 7 (descendant
    wins), and row 5 (the live verify observes its own worktree intact
    across the concurrent sweep -- this row has NO standalone test; it is
    only exercised HERE), plus 'merge finalizes' (the recovered request
    reaches a terminal 'done' and its branch lands on main).

    Zero-ENOENT causal-proxy chain (see the module docstring's
    "Attribution matrix" for the full per-leg rationale) is asserted via
    all four legs in ONE test:
      (a) protected-tree survival, per sweep and per mechanism --
          MERGE-REAPER leg: `_merge-verify` survives by the
          PERSISTENT_MERGE_WORKTREE_NAME exclusion, the live
          `_merge-<hash>` verify tree by the owned-ledger/grace window,
          and the aged LEASED `_merge-cafe5326` by C1 answering
          `skipped_lease_held` when the reaper OFFERED it (spy-recorded)
          -- with an aged UNLEASED REAL ephemeral `_merge-<uuid>`
          worktree, which C1 answered `removed` for, as that leg's
          positive control;
          CRASH-RECOVERY-SWEEP leg: every `_merge-*`/infra entry survives
          by the C2 classify_worktree_entry skip, proven by the SET of
          paths offered to `cleanup_worktree` (disjoint-from-protected +
          upper-bound pin), NOT by `.exists()` -- `cleanup_worktree` is
          deliberately an AsyncMock spy here, so `.exists()` is inert on
          this leg -- with the '999' planless dir as its positive control;
      (b) the gated verify runner's own worktree-existence observations
          (entry AND exit) are all True;
      (c) the recovered merge reaches `outcome.status == 'done'` and its
          branch is an ancestor of main;
      (d) zero `verify_cross_check_mismatch` L1 escalations were filed.
    """

    async def test_concurrent_startup_sweep_survives_and_merge_finalizes(
        self,
        mock_orch_config: MagicMock,
        git_repo: Path,
        git_config: GitConfig,
        tmp_path: Path,
    ) -> None:
        harness = _build_recovery_harness(mock_orch_config, git_repo)
        # Rebind harness.config to a REAL OrchestratorConfig (mirrors the
        # git_ops rebind above): the merge-verify dispatch path this capstone
        # actually drives reads several config fields directly off
        # MergeRequest.config (project_root for git-cwd/archive-root
        # resolution, merge_verify_min_free_disk_bytes for the pre-verify
        # disk guard, ...) that mock_orch_config deliberately leaves
        # unconfigured (it is tuned for the lighter harness-lifecycle-loop
        # surface TestDeleterFace exercises, not a real verify dispatch) --
        # an un-set MagicMock field reaching a real `int >= ...` comparison
        # raises TypeError, not a graceful skip. Harness.__init__ already
        # consumed mock_orch_config's neutralizing fields (usage_cap.enabled=
        # False, review.enabled=False, background-loop toggles, ...) by this
        # point and this test never calls harness.run() (only targeted
        # internal methods), so those neutralizations are moot post-init --
        # safe to swap in a fully-real, self-consistent config here.
        harness.config = OrchestratorConfig(project_root=git_repo, git=git_config)
        base = harness.git_ops.worktree_base

        # --- Pre-state: a REAL branch task/5326 with a real worktree and a
        # descendant tip (sha2 descends sha1). -----------------------------
        wt = (await harness.git_ops.create_worktree('5326')).path
        (wt / 'capstone_a.py').write_text('a = 1\n')
        await harness.git_ops.commit(wt, 'Add capstone_a.py')
        _, sha1_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        sha1 = sha1_raw.strip()

        (wt / 'capstone_b.py').write_text('b = 2\n')
        await harness.git_ops.commit(wt, 'Add capstone_b.py')
        _, sha2_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        sha2 = sha2_raw.strip()

        # Durable journal seeded with TWO entries for task/5326 (descendant
        # variant), both pointing at the same real worktree.
        reqs = _seed_dup_journal(
            harness._merge_store, '5326', [sha1, sha2], harness.config, wt,
        )

        # --- Decoys: leased persistent + ephemeral merge trees, infra bands,
        # and the task-shaped planless '999' dir (positive control). ------
        merge_verify = base / '_merge-verify'
        fd_verify = _plant_leased_tree(base, merge_verify)
        merge_uuid = base / '_merge-cafe5326'
        fd_uuid = _plant_leased_tree(base, merge_uuid)
        infra_dirs = {
            name: _plant_infra_dir(base, name)
            for name in ('.reseed-trash', '_mainprobe-x')
        }
        wt_task = _plant_task_dir(base, '999')

        # --- Merge-reaper leg fixture -------------------------------------
        # SpeculativeMergeWorker.reap_orphaned_merge_worktrees only reaches
        # GitOps.cleanup_merge_worktree (and thus the C1 guarded primitive)
        # for entries whose mtime age EXCEEDS
        # RESOURCE_AUDIT_WORKTREE_GRACE_SECS (9000.0); a freshly-planted
        # decoy is `continue`d past, which would make the reaper leg below
        # VACUOUS (survival by never-being-looked-at, not by the lease
        # guard). Back-date both merge-band decoys well past the window --
        # the lane `.lock` is a SIBLING of the tree, so planting the lease
        # above did not re-touch this directory's mtime.
        #
        # The reaper-specific POSITIVE CONTROL is a REAL registered ephemeral
        # `_merge-<uuid>` worktree, aged and UNLEASED: the same sweep that must
        # SKIP the leased decoy must REMOVE this one, and it must do so through
        # C1's `'removed'` branch. A plain `mkdir()`'d directory would NOT
        # prove that -- per GitOps.cleanup_merge_worktree's own docstring (and
        # the contract pinned by test_remove_merge_worktree_guarded.py::
        # test_non_worktree_directory_returns_failed), C1 answers `'failed'`
        # for a non-worktree dir and the tree then disappears via
        # cleanup_merge_worktree's shutil.rmtree FALLBACK. That control would
        # stay green even if C1's uncontended removal branch regressed to
        # always returning `'failed'` -- exactly the branch the leased decoy's
        # `skipped_lease_held` is being contrasted against. Using a real
        # worktree here makes the outcome pinnable, matching row 3's use of
        # `_make_ephemeral_worktree` for the same reason.
        merge_unleased = await _make_ephemeral_worktree(harness.git_ops)
        _age_tree(merge_uuid, 100_000)
        _age_tree(merge_unleased, 100_000)

        entered = asyncio.Event()
        release = asyncio.Event()
        observations: list[bool] = []
        gated = AsyncMock(
            side_effect=_gated_tree_liveness_verify(entered, release, observations),
        )

        harness._escalation_queue = EscalationQueue(tmp_path / 'escalations')
        worker_task: asyncio.Task | None = None

        try:
            # (1) Recover the durable journal via the ACTUAL harness entry
            # point -- exactly ONE winner enqueued for task/5326.
            report = await harness._recover_pending_merges()
            assert report['recovered'] == 1, report
            assert report['coalesced'] == 1, report
            assert len(report['requests']) == 1, report
            winner = report['requests'][0]
            assert winner.branch.bare_id == '5326'
            assert winner.request_id == reqs[1].request_id, (
                'the DESCENDANT record must be the recovered winner'
            )
            assert harness._merge_queue.qsize() == 1

            # (2) Start the merge-worker task with run_scoped_verification
            # patched to the gated tree-liveness runner; await entry.
            harness._merge_worker = SpeculativeMergeWorker(
                harness.git_ops,
                harness._merge_queue,
                merge_store=harness._merge_store,
                escalation_queue=harness._escalation_queue,
            )
            with patch('orchestrator.merge_queue.run_scoped_verification', gated):
                worker_task = asyncio.create_task(
                    harness._merge_worker.run(), name='capstone-merge-worker',
                )
                await asyncio.wait_for(entered.wait(), timeout=60)

                # (3) WHILE the verify is live, run the concurrent sweep: the
                # merge reaper THEN the crash-recovery sweep (mirrors run()'s
                # step 1b/1c0a -> 2c ordering -- see the module docstring's
                # "Concurrency model" section).
                #
                # The guarded-removal spy DELEGATES to the real bound method
                # (this instance attribute shadows the class method, so
                # cleanup_merge_worktree's `self.remove_merge_worktree_
                # guarded(...)` call is what gets captured) -- it records
                # what the reaper OFFERED to C1 and what C1 answered, so
                # tree survival is attributable to the lease guard rather
                # than to an inert sweep.
                spy = _GuardedRemovalSpy(harness.git_ops.remove_merge_worktree_guarded)
                harness.git_ops.remove_merge_worktree_guarded = spy  # type: ignore[method-assign]
                await harness._reap_orphaned_merge_worktrees(report['requests'])
                # Snapshot IMMEDIATELY: worker-driven removals later in this
                # test (the throwaway verify worktree's own teardown) must
                # not mask what THIS sweep did.
                reap_records = list(spy.records)
                await harness._recover_crashed_tasks()

                # --- MERGE-REAPER leg: attribution, not mere survival -----
                # The aged LEASED decoy was OFFERED to C1 and survived
                # BECAUSE C1 refused it ('skipped_lease_held').
                offered = {p for p, _reason, _outcome in reap_records}
                assert merge_uuid in offered, (
                    f'the aged leased decoy was never offered to the C1 '
                    f'guarded primitive -- the reaper leg is vacuous; '
                    f'offered={offered}'
                )
                uuid_outcomes = spy.outcomes_for(merge_uuid, records=reap_records)
                assert 'skipped_lease_held' in uuid_outcomes, (
                    f'C1 must REFUSE removal of a live-leased merge tree; '
                    f'got outcomes={uuid_outcomes}'
                )
                assert merge_uuid.exists(), (
                    '_merge-cafe5326 must survive the reaper: C1 answered '
                    'skipped_lease_held'
                )
                # REAPER POSITIVE CONTROL: the aged UNLEASED real worktree was
                # offered too, and C1 itself REMOVED it -- so the leg exercises
                # the same guarded-removal branch the leased decoy refused, and
                # the leased decoy's survival above is a real refusal rather
                # than an inert (or fallback-only) sweep. Both the C1 outcome
                # and the filesystem are pinned: `'removed'` alone would not
                # prove the tree is gone, and `not .exists()` alone would also
                # be satisfied by cleanup_merge_worktree's `'failed'`->rmtree
                # fallback (which would leave a regressed C1 undetected).
                assert merge_unleased in offered, (
                    f'the aged unleased worktree was never offered to C1 -- the '
                    f'reaper never reached the removal primitive; '
                    f'offered={offered}'
                )
                unleased_outcomes = spy.outcomes_for(
                    merge_unleased, records=reap_records,
                )
                assert 'removed' in unleased_outcomes, (
                    f'C1 must REMOVE an aged, unleased, unowned merge worktree '
                    f'through its own guarded-removal branch (not via '
                    f'cleanup_merge_worktree\'s rmtree fallback); got '
                    f'outcomes={unleased_outcomes}'
                )
                assert not merge_unleased.exists(), (
                    f'{merge_unleased.name} (aged, unleased, unowned) must be '
                    f'reaped -- otherwise this leg proves nothing about the '
                    f'leased decoy that survived it'
                )

                # Survivors protected by OTHER mechanisms on this leg (see
                # the module docstring's attribution matrix): `_merge-verify`
                # never reaches C1 at all (PERSISTENT_MERGE_WORKTREE_NAME
                # exclusion in reap_orphaned_merge_worktrees), the live
                # verify's own `_merge-<hash>` tree is protected by the
                # owned-ledger/grace window, and the infra bands are outside
                # the reaper's `_merge-` band entirely.
                call = gated.call_args
                live_merge_tree = Path(
                    call.args[0] if call.args else call.kwargs['worktree'],  # type: ignore[index]
                )
                assert live_merge_tree.exists(), (
                    'the live verify\'s own worktree must survive the sweep '
                    '(zero-ENOENT proxy (a))'
                )
                for d in (merge_verify, *infra_dirs.values()):
                    assert d.exists(), f'{d.name} must survive the concurrent sweep'

                # --- CRASH-RECOVERY-SWEEP leg: the cleaned SET is the C2
                # regression detector, NOT `.exists()` ---------------------
                # `_build_recovery_harness` spies `cleanup_worktree` as an
                # AsyncMock (deliberately -- a real delegate would race the
                # in-flight merge by deleting the live task/5326 worktree
                # mid-verify), so `.exists()` above is INERT on this leg: a
                # full C2 regression (the 2026-07-22 "Cleaned up worktree
                # _merge-verify" force-removal) would leave every tree on
                # disk and every `.exists()` green. What C2 actually
                # promises is that the sweep never OFFERS a protected entry
                # to cleanup_worktree at all -- so pin the offered set.
                cleaned_paths = {
                    c.args[0] for c in harness.git_ops.cleanup_worktree.call_args_list  # type: ignore[attr-defined]
                }
                protected = {
                    merge_verify, merge_uuid, live_merge_tree, *infra_dirs.values(),
                }
                assert cleaned_paths.isdisjoint(protected), (
                    f'C2 violated -- the crash-recovery sweep cleaned protected '
                    f'entries: {cleaned_paths & protected}'
                )
                # Positive control: the SAME sweep cleaned the planless dir --
                # proves the sweep is not inert.
                assert wt_task in cleaned_paths, (
                    f'positive control: the task-shaped planless dir must be '
                    f'cleaned by this sweep; cleaned={cleaned_paths}'
                )
                # UPPER-BOUND pin (OBSERVED, not inferred): the sweep offers
                # at most the two task-id-shaped entries -- the planless '999'
                # decoy and the real task/5326 branch worktree, which is
                # itself task-id-shaped and planless under this test's
                # MagicMock scheduler. Pinning the whole set (rather than
                # `assert_any_call`) makes ANY extra cleanup call a failure,
                # which is what turns a C2 regression into a red test.
                # Nothing is actually deleted here -- cleanup_worktree is an
                # AsyncMock spy -- so the live merge below is unaffected.
                #
                # Deliberately `<=`, not `==`: `wt` is in the observed set only
                # because the MagicMock scheduler makes a real branch worktree
                # look planless. Pinning its PRESENCE would freeze a mock
                # artifact into this gate, so a future production change that
                # (correctly) taught the sweep to skip a worktree backing an
                # in-flight merge would turn the leg's own regression detector
                # red. The subset bound keeps every tooth that matters -- any
                # EXTRA offered path, protected or not, still fails -- while
                # `wt_task in cleaned_paths` above keeps the positive control.
                assert cleaned_paths <= {wt_task, wt}, (
                    f'unexpected cleanup_worktree calls beyond the two '
                    f'task-id-shaped entries: {cleaned_paths - {wt_task, wt}}'
                )

                # (4) Release the gated verify; await the recovered merge.
                release.set()
                outcome = await asyncio.wait_for(winner.result, timeout=60)

            assert outcome.status == 'done', f'Expected done, got: {outcome}'
            full_branch = f'{harness.config.git.branch_prefix}5326'
            assert await harness.git_ops.is_ancestor(
                full_branch, harness.config.git.main_branch,
            ), 'merge finalizes: the branch must land on main'

            # (5) gamma collapse held under the live path (one verify total);
            # the gated runner never observed a missing worktree (zero-ENOENT
            # proxy); zero spurious cross-check L1 escalations were filed.
            assert gated.call_count == 1, (
                f'expected exactly one verify for task/5326; got {gated.call_count}'
            )
            assert observations and all(observations), (
                f'gated verify observed a missing worktree: {observations}'
            )
            cross_check_l1 = [
                e for e in harness._escalation_queue.get_pending()
                if e.category == 'verify_cross_check_mismatch' and e.level == 1
            ]
            assert cross_check_l1 == [], cross_check_l1
        finally:
            release.set()
            if harness._merge_worker is not None:
                await harness._merge_worker.stop()
            if worker_task is not None:
                worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker_task
            release_merge_verify_flock(fd_verify)
            release_merge_verify_flock(fd_uuid)
            remove_lock_holder_pgid(base)


# ---------------------------------------------------------------------------
# In-verify submit-identity reject helpers (row 8) -- ported from
# test_merge_queue_c3_submit_identity.py (_commit, _make_request,
# _verify_snapshot).
# ---------------------------------------------------------------------------


async def _commit(repo: Path, name: str) -> str:
    """Add a commit on the current branch/HEAD and return its SHA.

    Successive calls build a linear history, so an earlier SHA is a strict
    ancestor of a later one (SUPERSET when the earlier SHA is the in-flight
    tip and the later SHA is submitted).
    """
    (repo / f'{name}.txt').write_text(f'{name}\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', name], cwd=repo)
    return await _head_sha(repo)


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    *,
    request_id: str | None = None,
    snapshot_tip: str | None = None,
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future bound to the running loop
    (ported from test_merge_queue_c3_submit_identity.py's _make_request)."""
    kwargs: dict[str, Any] = {}
    if request_id is not None:
        kwargs['request_id'] = request_id
    if snapshot_tip is not None:
        kwargs['snapshot_tip'] = snapshot_tip
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
        **kwargs,
    )


def _verify_snapshot(request_id: str, branch: str, *, verify_age_secs: float = 42.0):
    """Fake ``live_snapshot`` reporting *request_id* as IN VERIFY.

    Matches the SpeculativeMergeWorker.snapshot() entry schema --
    ``verify_started_at`` non-None means the verify has started (ported from
    test_merge_queue_c3_submit_identity.py's _verify_snapshot).
    """
    def _snap() -> dict:
        return {'entries': [{
            'request_id': request_id,
            'branch': branch,
            'state': 'verifying',
            'verify_started_at': 1000.0,
            'verify_age_secs': verify_age_secs,
        }]}
    return _snap


# ---------------------------------------------------------------------------
# TestIdentityFaceInVerifyReject -- PRD Sec.9 row 8
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestIdentityFaceInVerifyReject:
    """PRD Sec.9 row 8: coalesce_or_enqueue_merge_request's duplicate_in_verify
    reject (delta, task 2927). Ported from
    test_merge_queue_c3_submit_identity.py::TestC3SubmitGateInVerify::
    test_in_verify_newer_sha_rejects.

    A newer SHA submitted while the branch's earlier SHA is already IN
    VERIFY is structurally REJECTed -- never coalesced, never replaced --
    leaving the live in-flight entry UNDISTURBED (D3).
    """

    async def test_newer_sha_in_verify_rejects_leaves_entry_undisturbed(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path, tmp_path: Path,
    ) -> None:
        """OBSERVED-TO-FIRE negative assertion (the reject must actually
        fire) paired with an UNDISTURBED positive control (the live
        in-flight entry survives untouched -- not dropped, not cancelled,
        nothing enqueued)."""
        sha1 = await _commit(git_repo, 'c1')
        sha2 = await _commit(git_repo, 'c2')  # strict descendant of sha1 -> SUPERSET

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = None

        old_fut: asyncio.Future = asyncio.get_running_loop().create_future()
        assert registry.acquire(
            '5326', 'task-old', old_fut, request_id='mr-old', snapshot_tip=sha1,
        )

        req_new = _make_request(
            'task-new', '5326', tmp_path, config, request_id='mr-new', snapshot_tip=sha2,
        )

        result = await coalesce_or_enqueue_merge_request(
            queue, req_new, event_store, registry,
            git_ops=None,
            live_snapshot=_verify_snapshot('mr-old', '5326', verify_age_secs=42.0),
            classifier_git_ops=git_ops,
        )

        assert result.rejected is True, (
            f'a newer SHA submitted while the earlier SHA is in verify must '
            f'REJECT; got {result}'
        )
        assert result.reject_code == 'duplicate_in_verify'
        assert result.dispatched is False
        assert result.in_flight is False
        assert result.existing_sha == sha1
        assert result.inflight_request_id == 'mr-old'
        assert result.verify_age_secs is not None

        # UNDISTURBED positive control: the live entry survives untouched.
        assert registry.is_inflight('5326')
        entry = registry.entry('5326')
        assert entry is not None and entry.request_id == 'mr-old'
        assert not old_fut.cancelled()
        assert queue.qsize() == 0


# ---------------------------------------------------------------------------
# Retire+resubmit fixture (row 9) -- ported from
# test_merge_cancel_retire.py's ``_RetireSpy``.
# ---------------------------------------------------------------------------


class _RetireSpy:
    """git_ops spy for retirement tests.

    ``find_inflight_merge_worktree`` returns the planted scratch path only
    WHILE it still exists on disk, so a post-removal re-scan finds nothing.
    ``remove_merge_worktree_guarded`` records ``(path, reason)`` and deletes
    the dir, returning the ``'removed'`` outcome -- mirrors the disk-scan/
    removal surface of test_merge_queue_c3_submit_identity.py's _ScratchSpy.
    """

    def __init__(self, scratch: Path | None) -> None:
        self._scratch = scratch
        self.removed: list[tuple[Path, str]] = []
        self.find_calls = 0

    async def find_inflight_merge_worktree(self, branch: str) -> Path | None:  # noqa: ARG002
        self.find_calls += 1
        if self._scratch is not None and self._scratch.exists():
            return self._scratch
        return None

    async def remove_merge_worktree_guarded(self, path: Path, *, reason: str) -> str:
        self.removed.append((path, reason))
        if path.exists():
            shutil.rmtree(path)
        return 'removed'


# ---------------------------------------------------------------------------
# TestIdentityFaceCancelResubmit -- PRD Sec.9 row 9
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestIdentityFaceCancelResubmit:
    """PRD Sec.9 row 9: retire_cancelled_merge_request's full retirement
    (epsilon, task 2928) followed by an immediate resubmit. Ported from
    test_merge_cancel_retire.py::TestRetireCancelledMergeRequest::
    test_basic_release_worktree_and_forget +
    test_identity_guard_does_not_drop_fresh_slot.

    A cancelled merge is FULLY retired (registry slot + worktree via the C1
    guarded primitive + sticky retention) before an immediate resubmit,
    which gets a genuinely FRESH entry rather than coalescing onto /
    mirroring the retired corpse.
    """

    async def test_full_retirement_then_immediate_resubmit_gets_fresh_slot(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        registry = InFlightMergeRegistry()
        retention = TerminalOutcomeRetention()
        fut_old: asyncio.Future = asyncio.get_running_loop().create_future()
        registry.acquire(
            '5326', 'T', fut_old, request_id='mr-old', snapshot_tip='sha-old',
        )
        retention.record(TerminalOutcomeRecord(
            request_id='mr-old', branch='5326', task_id='T', state='abandoned',
        ))
        wt = tmp_path / 'merge-5326'
        wt.mkdir()
        spy = _RetireSpy(wt)

        outcome = await retire_cancelled_merge_request(
            request_id='mr-old', branch='5326', task_id='T',
            registry=registry, retention=retention, git_ops=spy, event_store=None,
        )

        # FULL retirement: slot released, worktree routed through the C1
        # guarded primitive, sticky cleared across every retention index.
        assert registry.is_inflight('5326') is False
        assert spy.removed == [(wt, 'merge_cancel_retire')]
        assert not wt.exists()
        assert outcome == 'removed'
        assert retention.get_by_branch('5326') is None
        assert retention.get_by_task('T') is None
        assert retention.get('mr-old') is None

        # IMMEDIATE resubmit through the PRODUCTION entry point -- a genuinely
        # FRESH DISPATCH, not a coalesce onto / mirror of the retired corpse.
        #
        # Driven via coalesce_or_enqueue_merge_request (NOT a bare
        # registry.acquire) precisely because that is the only call that
        # consults BOTH sources the failure mode lives in: the registry slot
        # AND the TerminalOutcomeRetention ring the retirement just cleared. A
        # direct acquire on a just-released registry cannot observe a resubmit
        # coalescing onto a surviving sticky record, so it could not detect the
        # regression this row exists to catch.
        queue: asyncio.Queue = asyncio.Queue()
        req_new = _make_request(
            'T2', '5326', tmp_path, config,
            request_id='mr-new', snapshot_tip='sha-new',
        )
        result = await coalesce_or_enqueue_merge_request(
            queue, req_new, None, registry, git_ops=None, retention=retention,
        )

        assert result.dispatched is True, (
            f'the resubmit must DISPATCH a fresh merge; got {result}'
        )
        assert result.in_flight is False, (
            f'the resubmit must NOT coalesce onto the retired entry; got {result}'
        )
        assert result.rejected is False, result
        assert queue.qsize() == 1
        assert queue.get_nowait() is req_new
        entry = registry.entry('5326')
        assert entry is not None and entry.request_id == 'mr-new'
        assert not req_new.result.cancelled()
        # The retired sticky record stayed forgotten -- a resurrected 'mr-old'
        # record is exactly what would mirror a terminal outcome onto the
        # fresh submitter.
        assert retention.get('mr-old') is None
        assert retention.get_by_branch('5326') is None

        # IDENTITY GUARD: a LATE duplicate retirement for the now-stale
        # 'mr-old' (the shape a retried/duplicated merge_cancel produces) must
        # NOT drop the fresh slot the resubmit above just claimed.
        await retire_cancelled_merge_request(
            request_id='mr-old', branch='5326', task_id='T',
            registry=registry, retention=retention, git_ops=None, event_store=None,
        )
        entry_after = registry.entry('5326')
        assert entry_after is not None, (
            'the late stale retirement dropped the fresh slot entirely'
        )
        assert entry_after.request_id == 'mr-new', (
            f'the identity guard must only release a slot still owned by the '
            f'retiring request_id; got {entry_after.request_id}'
        )
        assert not req_new.result.done(), (
            'the fresh submitter\'s future was resolved/cancelled by a stale '
            'retirement'
        )
        req_new.result.cancel()  # cleanup: nothing will ever resolve it here
