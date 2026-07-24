"""Merge-worktree lifecycle integrity zeta done-gate: restart-simulation
boundary suite (PRD Sec.9 rows 1-9).

PRD: docs/prds/merge-worktree-lifecycle-integrity.md, task zeta (the B+H
done-gate).  All five prerequisite legs are LANDED and BEHAVIOUR-FROZEN for
this batch:

  alpha (2924) -- GitOps.remove_merge_worktree_guarded (git_ops.py:8239):
      lease-enforced removal primitive; outcome vocabulary 'removed' /
      'skipped_lease_held' / 'skipped_persistent' / 'not_present' / 'failed'.
  beta  (2925) -- classify_worktree_entry (git_ops.py:467) + the C2 namespace
      guard in Harness._recover_crashed_tasks (harness.py:2842-2859): the
      crash-recovery sweep SKIPS+REPORTS `_merge-*`/infra bands instead of
      force-removing them (the 2026-07-22 task/5326 incident).
  gamma (2926) -- recover_pending_merges' registry-gated per-branch collapse
      (merge_queue_store.py:433-537): a branch with N surviving journal
      entries enqueues exactly ONE winner (descendant-most snapshot tip);
      every loser attaches as a peer waiter whose future mirrors the
      winner's terminal outcome.
  delta (2927) -- coalesce_or_enqueue_merge_request's duplicate_in_verify
      reject (merge_queue.py:2991/4293): a newer SHA submitted while the
      earlier SHA is IN VERIFY is structurally REJECTed, not coalesced or
      replaced.
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
(step 1c0a, harness.py:1881) and `_recover_crashed_tasks` (step 2c,
harness.py:2010), are SEQUENTIAL awaits -- NOT gathered/parallelized. The
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
  (a) every `_merge-*`/infra tree survives the concurrent sweep
      (`.exists()`), paired with a POSITIVE CONTROL (the task-shaped '999'
      dir the SAME sweep cleans) so a survives-because-the-sweep-is-inert
      false pass is impossible;
  (b) the gated verify runner asserts its OWN `_merge-<hash>` cwd worktree
      exists at entry AND at completion -- a tree yanked mid-verify would
      fail this assertion, directly modelling the incident;
  (c) the recovered merge reaches `outcome.status == 'done'` and its
      branch lands on main ('merge finalizes');
  (d) zero spurious `verify_cross_check_mismatch` L1 escalations are filed
      (the incident's clobbered-worktree false-FAIL signature).

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

STALE-OFFSET WARNING
------------------------
Every `:NNNN` line citation above (and in inline comments below) can drift
as the modules it cites are edited by unrelated work. Always locate
symbols BY NAME (grep/search), never trust a line offset.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.lane_lifecycle import LaneLifecycle
from orchestrator.merge_queue import SpeculativeMergeWorker
from orchestrator.merge_queue_store import MergeQueueStore, recover_pending_merges
from orchestrator.merge_types import InFlightMergeRegistry, MergeRequest, QueuedBranch
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
# fixture, lines 29-88) -- a plain factory function (not a fixture) so the
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

        outcome_dead = await git_ops.remove_merge_worktree_guarded(dead_wt, reason='reaper')
        assert outcome_dead == 'removed', (
            'a stale holder-pgid record with no live flock must fail OPEN'
        )
        assert not dead_wt.exists()

        fd_live = _plant_leased_tree(base, live_wt)
        try:
            with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
                outcome_live = await git_ops.remove_merge_worktree_guarded(live_wt, reason='reaper')

            assert outcome_live == 'skipped_lease_held', (
                'a LIVE lease holder must skip removal, never force through'
            )
            assert live_wt.exists(), 'a live lease holder must leave the tree intact'

            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1, (
                f'expected exactly one WARNING, got {len(warnings)}: '
                f'{[r.getMessage() for r in warnings]}'
            )
            message = warnings[0].getMessage()
            assert str(os.getpgrp()) in message, message
            assert 'reaper' in message, message
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

    Exercises PRD Sec.9 rows 1, 2, 4 (protected trees survive the
    concurrent sweep), 6 (dup-journal same-branch collapse; here a
    descendant-tip variant), 7 (descendant wins), and row 5 (the live
    verify observes its own worktree intact across the concurrent sweep --
    this row has NO standalone test; it is only exercised HERE), plus
    'merge finalizes' (the recovered request reaches a terminal 'done' and
    its branch lands on main).

    Zero-ENOENT causal-proxy chain (see the module docstring for the full
    rationale) is asserted via all four legs in ONE test:
      (a) every `_merge-*`/infra tree still `.exists()` during the live
          verify, paired with the '999' positive control the SAME sweep
          cleans;
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
                await harness._reap_orphaned_merge_worktrees(report['requests'])
                await harness._recover_crashed_tasks()

                for d in (merge_verify, merge_uuid, *infra_dirs.values()):
                    assert d.exists(), f'{d.name} must survive the concurrent sweep'
                # Positive control: the SAME sweep cleaned the planless dir --
                # proves the sweep is not inert.
                harness.git_ops.cleanup_worktree.assert_any_call(wt_task, '999')  # type: ignore[attr-defined]

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
