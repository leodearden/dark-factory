"""Tests for SpeculativeMergeWorker.reap_orphaned_merge_worktrees (I6 leak
closure — reap/re-adopt orphaned unregistered ``_merge-<uuid>`` worktrees on
worker startup/recovery, task 2060).

Steps covered:
  step-1  RED   — core reap + headline signal (bare worker, real worktrees)
  step-2  GREEN — implement reap_orphaned_merge_worktrees() reap core
  step-3  RED   — preservation guards (persistent / owned / fresh-within-grace)
  step-4  GREEN — confirm/refine skip ordering + INFO summary log
  step-5  RED   — re-adoption via recovered_branches
  step-6  GREEN — implement re-adoption
  step-7  RED   — fail-open robustness
  step-8  GREEN — implement fail-open guards

This module reuses the bare-worker git_repo/git_config/git_ops fixtures and
_setup_repo from test_merge_queue_resource_audit.py (per-file duplication
convention — see that module's docstring), along with its
_mkdir_worktree/_GRACE/_NOW age-test helper pattern. It imports
orchestrator.merge_queue LOCALLY inside each test so a not-yet-implemented
symbol never breaks collection of the rest of the file during the RED steps.

Unlike test_merge_queue_resource_audit.py's pure-sync worktree_ledger_violations
tests, reap_orphaned_merge_worktrees is async (it awaits
git_ops.cleanup_merge_worktree / git_ops.find_inflight_merge_worktree), so
every test in this module is ``async def`` under ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import PERSISTENT_MERGE_WORKTREE_NAME, GitOps, _run

# ---------------------------------------------------------------------------
# Fixtures + helpers (per-file duplication convention — see
# test_merge_queue_resource_audit.py)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
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


def _mkdir_worktree(git_ops: GitOps, name: str, *, mtime: float | None = None) -> Path:
    """Create worktree_base/name (creating worktree_base itself if absent).

    Copied from test_merge_queue_resource_audit.py (per-file duplication
    convention). Used only for cases that don't need a real git admin entry
    (e.g. the persistent ``_merge-verify`` preservation check) — real
    reapable/re-adoptable ephemeral worktrees are made via
    ``git_ops._create_merge_worktree()`` below so the reap primitive
    (``git worktree remove --force``) has a genuine git admin entry to act on.
    """
    git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
    wt = git_ops.worktree_base / name
    wt.mkdir()
    if mtime is not None:
        os.utime(wt, (mtime, mtime))
    return wt


_GRACE = 100.0
_NOW = 1_000_000.0


async def _create_backdated_merge_worktree(git_ops: GitOps, *, age: float) -> Path:
    """Create a real ephemeral ``_merge-<uuid>`` worktree, backdated to age
    seconds before ``_NOW`` (so it reads as aged-past-grace at ``now=_NOW``).
    """
    wt, _ = await git_ops._create_merge_worktree()
    backdated = _NOW - age
    os.utime(wt, (backdated, backdated))
    return wt


async def _registered_worktree_paths(git_ops: GitOps) -> list[str]:
    """Return the ``git worktree list --porcelain`` registered paths."""
    rc, out, _ = await _run(
        ['git', 'worktree', 'list', '--porcelain'], cwd=git_ops.project_root,
    )
    assert rc == 0
    return [
        line[len('worktree '):].strip()
        for line in out.splitlines()
        if line.startswith('worktree ')
    ]


def _lane_admin_dir(lane: Path) -> Path:
    """Parse the ``.git/worktrees/<name>`` admin dir out of a lane's ``.git``
    pointer file (``gitdir: <repo>/.git/worktrees/<name>``).

    Local per-file duplication (see module docstring / the same helper in
    test_git_ops.py:_lane_admin_dir) — used to simulate the task-2922 shape-1
    interrupted-teardown leak by rmtree-ing the admin dir while the on-disk
    tree survives, so ``git worktree remove --force`` errors.
    """
    content = (lane / '.git').read_text().strip()
    prefix = 'gitdir:'
    assert content.startswith(prefix), f'unexpected worktree .git pointer: {content!r}'
    return Path(content[len(prefix):].strip())


# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: core reap + headline signal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapOrphanedMergeWorktreesCore:
    """Unit tests for the reap core.

    RED until step-2 GREEN adds reap_orphaned_merge_worktrees to
    merge_queue.py.
    """

    async def test_aged_orphan_reaped_and_ledger_clears(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        wt = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)

        # Sanity: the detector flags this orphan BEFORE the sweep runs.
        before = worker.worktree_ledger_violations(now=_NOW)
        assert len(before) == 1, f'expected exactly one violation, got: {before!r}'

        report = await worker.reap_orphaned_merge_worktrees(now=_NOW)

        assert not wt.exists(), 'aged orphan must be removed from disk'
        registered = await _registered_worktree_paths(git_ops)
        assert str(wt) not in registered, (
            f'aged orphan must be gone from git worktree list: {registered}'
        )
        assert str(wt.resolve()) in report['reaped']
        assert report['readopted'] == []
        assert worker.worktree_ledger_violations(now=_NOW) == []
        assert worker.snapshot()['resource_audit']['worktree_ledger'] == []

    async def test_fresh_orphan_within_grace_not_reaped(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        # Real ephemeral worktree with a FRESH mtime (not backdated) — within
        # grace at now=_NOW.
        wt, _ = await git_ops._create_merge_worktree()

        report = await worker.reap_orphaned_merge_worktrees(now=_NOW)

        assert wt.exists(), 'fresh-within-grace orphan must be preserved'
        assert str(wt.resolve()) not in report['reaped']


# ---------------------------------------------------------------------------
# step-3 RED / step-4 GREEN: preservation guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapOrphanedMergeWorktreesPreservation:
    """Unit tests asserting the sweep preserves everything it must not touch
    — the persistent ``_merge-verify`` worktree, an already-owned
    (registered) worktree, and a fresh (within-grace) unregistered worktree
    — while still reaping the one genuine aged orphan alongside them in the
    SAME sweep (task 2060 step-3).

    The ``_merge-verify`` and grace-skip preservations already hold
    structurally from step-2 (same discovery/skip logic as
    ``worktree_ledger_violations``); this class is the first to exercise the
    owned-skip explicitly, and the first to combine all three preservation
    guards with a genuine reap in a single sweep.
    """

    async def test_sweep_reaps_only_the_genuine_orphan(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE

        # (1) genuine aged orphan — expected reaped.
        orphan = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)

        # (2) persistent _merge-verify, aged past grace — expected preserved.
        verify_wt = _mkdir_worktree(
            git_ops, PERSISTENT_MERGE_WORKTREE_NAME, mtime=_NOW - _GRACE - 10,
        )

        # (3) aged ephemeral, but pre-registered (owned) — expected preserved.
        owned_wt = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)
        worker._register_owned_merge_worktree(owned_wt)

        # (4) fresh (within-grace) unregistered — expected preserved.
        fresh_wt, _ = await git_ops._create_merge_worktree()

        report = await worker.reap_orphaned_merge_worktrees(now=_NOW)

        assert str(orphan.resolve()) in report['reaped']
        assert not orphan.exists()

        assert verify_wt.exists(), '_merge-verify must survive the sweep'
        assert str(verify_wt.resolve()) not in report['reaped']

        assert owned_wt.exists(), 'owned/registered worktree must survive the sweep'
        assert str(owned_wt.resolve()) not in report['reaped']

        assert fresh_wt.exists(), 'fresh-within-grace worktree must survive the sweep'
        assert str(fresh_wt.resolve()) not in report['reaped']

        assert report['reaped'] == [str(orphan.resolve())], (
            f'expected only the genuine orphan reaped, got: {report["reaped"]!r}'
        )


# ---------------------------------------------------------------------------
# step-5 RED / step-6 GREEN: re-adoption via recovered_branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapOrphanedMergeWorktreesReadoption:
    """Unit tests for re-adoption of worktrees backing recovered in-flight
    merges via ``recovered_branches`` (task 2060 step-5).

    RED until step-6 GREEN adds re-adoption to reap_orphaned_merge_worktrees.
    """

    async def test_recovered_branch_readopts_its_aged_worktree(
        self, git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        # Aged past grace — if this were an ordinary orphan it would be
        # reaped, but it backs a recovered in-flight merge so it must be
        # re-adopted instead.
        wt = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)

        async def _fake_find_inflight(branch: str) -> Path | None:
            return wt if branch == 'recovered-x' else None

        monkeypatch.setattr(
            git_ops, 'find_inflight_merge_worktree',
            AsyncMock(side_effect=_fake_find_inflight),
        )

        report = await worker.reap_orphaned_merge_worktrees(
            recovered_branches=['recovered-x', 'no-match-branch'], now=_NOW,
        )

        assert wt.resolve() in worker._owned_merge_worktrees, (
            're-adoption must register the worktree into the liveness ledger'
        )
        assert report['readopted'] == [str(wt.resolve())], (
            f'the no-match branch must contribute nothing to readopted, got: {report!r}'
        )
        assert str(wt.resolve()) not in report['reaped']
        assert wt.exists(), 're-adoption must bypass the grace/reap gate despite its age'
        assert worker.worktree_ledger_violations(now=_NOW) == [], (
            're-adoption (registration) must cure the false violation'
        )


# ---------------------------------------------------------------------------
# step-7 RED / step-8 GREEN: fail-open robustness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapOrphanedMergeWorktreesFailOpen:
    """Unit tests asserting the sweep never raises and never lets one bad
    record abort the rest — this is a startup maintenance sweep and must not
    block orchestrator startup (task 2060 step-7).

    (a) already holds from step-2's initial guard. (b)/(c) are RED until
    step-8 GREEN adds the per-path/per-branch try/except guards.
    """

    async def test_missing_worktree_base_returns_empty_report_without_raising(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        assert not git_ops.worktree_base.exists()

        report = await worker.reap_orphaned_merge_worktrees(now=_NOW)

        assert report == {'readopted': [], 'reaped': []}

    async def test_cleanup_failure_for_one_path_does_not_abort_the_sweep(
        self, git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        wt_a = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)
        wt_b = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)

        real_cleanup = git_ops.cleanup_merge_worktree

        async def _flaky_cleanup(path: Path) -> None:
            if path == wt_a.resolve():
                raise RuntimeError('boom')
            await real_cleanup(path)

        monkeypatch.setattr(
            git_ops, 'cleanup_merge_worktree', AsyncMock(side_effect=_flaky_cleanup),
        )

        report = await worker.reap_orphaned_merge_worktrees(now=_NOW)

        assert str(wt_b.resolve()) in report['reaped'], (
            f'the non-raising path must still be reaped: {report!r}'
        )
        assert not wt_b.exists()
        assert str(wt_a.resolve()) not in report['reaped'], (
            'the raising path must not be recorded as reaped'
        )

    async def test_find_inflight_failure_for_one_branch_does_not_abort_the_sweep(
        self, git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        orphan = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)

        async def _boom(branch: str) -> Path | None:
            raise RuntimeError('boom')

        monkeypatch.setattr(
            git_ops, 'find_inflight_merge_worktree', AsyncMock(side_effect=_boom),
        )

        report = await worker.reap_orphaned_merge_worktrees(
            recovered_branches=['some-branch'], now=_NOW,
        )

        assert report['readopted'] == []
        assert str(orphan.resolve()) in report['reaped'], (
            f'an unrelated aged orphan must still be reaped: {report!r}'
        )

    async def test_scandir_failure_does_not_raise_and_preserves_readoptions(
        self, git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A TOCTOU disappearance of ``worktree_base`` between the ``is_dir()``
        guard and the ``os.scandir`` must NOT raise out of the method — it
        returns the report-so-far, preserving re-adoptions applied before the
        scan.  Honours the never-raise contract self-containedly (not just via
        the harness caller's try/except).
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE

        # A recovered in-flight worktree re-adopted BEFORE the reap scan.
        inflight = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)
        monkeypatch.setattr(
            git_ops, 'find_inflight_merge_worktree',
            AsyncMock(return_value=inflight),
        )

        # worktree_base is a real dir (is_dir guard passes) but the scandir
        # itself blows up mid-sweep (base vanished / unreadable).
        assert git_ops.worktree_base.is_dir()

        def _boom_scandir(path: object) -> object:
            raise OSError('base vanished')

        monkeypatch.setattr('orchestrator.merge_queue.os.scandir', _boom_scandir)

        report = await worker.reap_orphaned_merge_worktrees(
            recovered_branches=['recovered-x'], now=_NOW,
        )

        # No raise; reap scan skipped, but the pre-scan re-adoption survives.
        assert report['reaped'] == []
        assert str(inflight.resolve()) in report['readopted']
        assert inflight.resolve() in {
            p.resolve() for p in worker._owned_merge_worktrees
        }


# ---------------------------------------------------------------------------
# task 2922: crash-safe shape-1 interrupted-teardown leak — end-to-end
# (reaper reaps it via the fixed cleanup + the ledger audit reads clean)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapShape1InterruptedTeardown:
    """End-to-end acceptance for the task-2922 shape-1 leak.

    Shape-1: an interrupted teardown (SIGTERM/restart mid-merge) leaves a
    full ``_merge-<uuid>`` checkout on disk while its
    ``.git/worktrees/<name>`` admin dir is already gone, so
    ``git worktree remove --force`` errors ('not a working tree') and
    ``cleanup_merge_worktree``'s guarded primitive returns 'failed'. The
    existing startup reaper (which calls cleanup_merge_worktree per aged
    candidate) must still remove the tree — via the crash-safe cleanup
    fallback added in step-2 — and the observation-only ledger audit
    (``worktree_ledger_violations``) must then read clean, satisfying the
    acceptance criterion "zero unregistered _merge-* violations after a
    simulated mid-teardown crash".

    RED on base: cleanup_merge_worktree fires-and-forgets the primitive's
    'failed' outcome, so the reaper leaves the on-disk tree (and its orphan
    ``.lock``) and the audit keeps flagging it forever.
    """

    async def test_aged_shape1_orphan_reaped_and_ledger_clears(
        self, git_ops: GitOps,
    ) -> None:
        import shutil

        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        wt = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)

        # Simulate shape-1: remove the .git/worktrees/<name> admin dir (located
        # via the lane's <wt>/.git gitdir pointer) so the on-disk tree survives
        # but `git worktree remove` errors — the interrupted-teardown shape.
        shutil.rmtree(_lane_admin_dir(wt))

        # Sanity: the audit flags this shape-1 orphan BEFORE the sweep.
        before = worker.worktree_ledger_violations(now=_NOW)
        assert len(before) == 1, f'expected exactly one violation, got: {before!r}'

        report = await worker.reap_orphaned_merge_worktrees(now=_NOW)

        assert not wt.exists(), 'shape-1 orphan must be removed from disk'
        assert str(wt.resolve()) in report['reaped']
        assert report['readopted'] == []
        # No _merge-* directory OR sibling .lock orphan survives the sweep
        # (task 2924's strict no-leak glob).
        leaks = list(git_ops.worktree_base.glob('_merge-*'))
        assert not leaks, f'no _merge-* leak may survive the sweep: {leaks}'
        # The audit reads clean AFTER the sweep — the acceptance criterion.
        assert worker.worktree_ledger_violations(now=_NOW) == []


# ---------------------------------------------------------------------------
# task 3018: periodic (steady-state) reap wired into the heartbeat loop
#
# Closes the observe-but-never-reclaim gap: reap_orphaned_merge_worktrees
# above is wired ONLY at startup/recovery (Harness._reap_orphaned_merge_worktrees),
# so a leak that first crosses grace mid-run sat unreaped until the next
# restart. _maybe_reap_orphaned_merge_worktrees is the periodic, rate-limited
# steady-state counterpart, invoked from _heartbeat_loop.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPeriodicReapHelper:
    """Unit tests for SpeculativeMergeWorker._maybe_reap_orphaned_merge_worktrees
    (task 3018 step-1..step-4).

    test_first_call_reaps_orphan_preserves_owned: RED until step-2 GREEN adds
    _maybe_reap_orphaned_merge_worktrees and the _last_reap_at/_reap_interval_s
    fields to merge_queue.py.

    test_reap_is_rate_limited_by_interval: RED until step-4 GREEN adds the
    interval gate — step-2 always reaps unconditionally, so a call within
    _reap_interval_s of the previous one would still reap.
    """

    async def test_first_call_reaps_orphan_preserves_owned(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        # The periodic sweep destroys at PERIODIC_REAP_MIN_AGE_SECS, not at
        # the detection grace (task 3018 step-14 — see
        # TestPeriodicReapDestructiveFloor for that split's own coverage).
        # Pinned to _GRACE here so this test keeps testing what it is ABOUT —
        # first-call/owned-preservation semantics — rather than incidentally
        # re-testing the age floor.
        worker.PERIODIC_REAP_MIN_AGE_SECS = _GRACE
        # _last_reap_at is seeded to real construction time, not 0.0 (task
        # 3018 amendment — avoids racing the harness startup recovery
        # sequence; see test_last_reap_at_seeded_to_construction_time).
        # Reset it to 0.0 here to simulate "a sweep is due" deterministically
        # against this test's fake _NOW clock, independent of wall-clock
        # time at worker construction.
        worker._last_reap_at = 0.0

        orphan = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)
        owned_wt, _ = await git_ops._create_merge_worktree()
        worker._register_owned_merge_worktree(owned_wt)

        await worker._maybe_reap_orphaned_merge_worktrees(now=_NOW)

        assert not orphan.exists(), 'aged orphan must be removed from disk'
        registered = await _registered_worktree_paths(git_ops)
        assert str(orphan) not in registered, (
            f'aged orphan must be gone from git worktree list: {registered}'
        )
        assert owned_wt.exists(), 'owned/registered worktree must survive the sweep'
        assert worker.worktree_ledger_violations(now=_NOW) == [], (
            'the audit must read clean after the periodic reap reclaims the orphan'
        )
        assert worker._last_reap_at == _NOW, 'the reap ran and advanced the clock'

    async def test_reap_is_rate_limited_by_interval(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        # See test_first_call_reaps_orphan_preserves_owned: pin the task-3018
        # destruction floor to _GRACE so this test stays about the INTERVAL
        # rate-limit, not the age floor.
        worker.PERIODIC_REAP_MIN_AGE_SECS = _GRACE
        worker._reap_interval_s = 50.0
        # See test_first_call_reaps_orphan_preserves_owned: reset the
        # construction-time seed to 0.0 so "no prior call" is deterministic
        # against this test's fake _NOW clock (task 3018 amendment).
        worker._last_reap_at = 0.0

        # First call: nothing to reap yet, but it still runs (no prior call)
        # and advances the clock.
        await worker._maybe_reap_orphaned_merge_worktrees(now=_NOW)
        assert worker._last_reap_at == _NOW

        # A new aged orphan appears after the first call.
        orphan = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)

        # WITHIN the interval (10s < 50s) — must be rate-limited: no reap,
        # no clock advance.
        await worker._maybe_reap_orphaned_merge_worktrees(now=_NOW + 10)
        assert orphan.exists(), 'a call within _reap_interval_s must be rate-limited'
        assert worker._last_reap_at == _NOW, (
            'a rate-limited call must not advance _last_reap_at'
        )

        # PAST the interval (60s > 50s) — must fire and reap.
        await worker._maybe_reap_orphaned_merge_worktrees(now=_NOW + 60)
        assert not orphan.exists(), 'a call past _reap_interval_s must reap'
        assert worker._last_reap_at == _NOW + 60

    async def test_last_reap_at_seeded_to_construction_time(self, git_ops: GitOps) -> None:
        """_last_reap_at must NOT default to 0.0 (unlike _last_heartbeat_at).

        Harness._start_merge_worker spawns this worker's loops (including
        _heartbeat_loop) via lifecycle.start_all() BEFORE the startup
        recovery sequence runs (_recover_pending_merges, then the startup
        Harness._reap_orphaned_merge_worktrees(recovered_branches) call —
        see harness.py ~1844-1893). An init-0.0 first poll (~_HEARTBEAT_POLL_S
        later) could fire the periodic sweep — which passes no
        recovered_branches — before that startup sweep re-adopts a worktree
        backing a recovered in-flight merge, reaping it out from under
        recovery. Seeding to real construction time instead defers the
        first periodic sweep by a full _reap_interval_s (task 3018
        amendment).
        """
        import time as time_mod

        from orchestrator.merge_queue import SpeculativeMergeWorker

        before = time_mod.time()
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        after = time_mod.time()

        assert before <= worker._last_reap_at <= after, (
            f'_last_reap_at must be seeded to real construction time, got '
            f'{worker._last_reap_at!r} (expected within [{before}, {after}])'
        )


# ---------------------------------------------------------------------------
# task 3018 step-5..step-8: _heartbeat_loop wiring for the periodic reap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHeartbeatLoopReapWiring:
    """Unit tests asserting ``_heartbeat_loop`` invokes the periodic reap
    helper each poll, and that the loop survives a raising reap (fail-open
    — a reap bug must never take down the heartbeat loop, mirroring the
    existing touch/heartbeat swallow-and-log convention).

    test_heartbeat_loop_invokes_periodic_reap: RED until step-6 GREEN wires
    ``await self._maybe_reap_orphaned_merge_worktrees(time.time())`` into
    ``_heartbeat_loop``.

    test_heartbeat_loop_survives_raising_reap: RED until step-8 GREEN wraps
    that call in its own try/except — step-6 wired it unguarded (outside
    the existing touch/heartbeat try-block), so a raise from the reap
    helper currently propagates out of ``_heartbeat_loop`` and kills the
    task instead of being logged and swallowed.
    """

    async def test_heartbeat_loop_invokes_periodic_reap(
        self, git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import orchestrator.merge_queue as mq_mod
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        monkeypatch.setattr(mq_mod, '_HEARTBEAT_POLL_S', 0.0)
        # Isolate the reap: no-op the touch + log-heartbeat calls the loop
        # also makes each poll.
        monkeypatch.setattr(worker, '_touch_owned_merge_worktrees', lambda: 0)
        monkeypatch.setattr(worker, '_maybe_log_queue_heartbeat', lambda now: False)

        calls: list[float] = []

        async def _fake_reap(now: float) -> None:
            calls.append(now)
            # Stop the loop deterministically after the first reap so
            # awaiting the coroutine below returns instead of spinning.
            worker._running = False

        monkeypatch.setattr(worker, '_maybe_reap_orphaned_merge_worktrees', _fake_reap)

        worker._running = True
        await asyncio.wait_for(worker._heartbeat_loop(), timeout=2.0)

        assert len(calls) >= 1, '_heartbeat_loop must invoke the periodic reap helper'
        assert isinstance(calls[0], float), (
            f'_heartbeat_loop must pass a float now= to the reap helper, got {calls[0]!r}'
        )

    async def test_heartbeat_loop_survives_raising_reap(
        self, git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import orchestrator.merge_queue as mq_mod
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        monkeypatch.setattr(mq_mod, '_HEARTBEAT_POLL_S', 0.01)
        # Isolate the reap: no-op the touch + log-heartbeat calls the loop
        # also makes each poll.
        monkeypatch.setattr(worker, '_touch_owned_merge_worktrees', lambda: 0)
        monkeypatch.setattr(worker, '_maybe_log_queue_heartbeat', lambda now: False)

        async def _raising_reap(now: float) -> None:
            raise RuntimeError('boom: periodic reap exploded')

        monkeypatch.setattr(worker, '_maybe_reap_orphaned_merge_worktrees', _raising_reap)

        worker._running = True
        task = asyncio.create_task(worker._heartbeat_loop())
        try:
            # Let several polls occur — each must swallow the raise
            # (fail-open). If it doesn't, the loop dies on the very first
            # poll and `task` completes with the RuntimeError long before
            # we get here.
            await asyncio.sleep(0.05)
            worker._running = False

            await asyncio.wait_for(task, timeout=2.0)
        finally:
            if not task.done():
                task.cancel()

        assert task.exception() is None, (
            'a raising periodic reap must not kill/propagate out of '
            '_heartbeat_loop (fail-open, mirrors the touch/heartbeat guard)'
        )

    async def test_heartbeat_loop_bounds_hanging_reap_with_timeout(
        self, git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hung periodic reap (e.g. a wedged git subprocess — GitOps._run
        has no internal timeout) must not block _heartbeat_loop forever: the
        call is bounded by _reap_sweep_timeout_s via asyncio.wait_for, so the
        loop keeps polling — and keeps touching owned worktrees — instead of
        stalling on the first hung sweep indefinitely (task 3018 amendment;
        reviewer_comprehensive robustness finding at merge_queue.py:10440).
        """
        import orchestrator.merge_queue as mq_mod
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        monkeypatch.setattr(mq_mod, '_HEARTBEAT_POLL_S', 0.01)
        worker._reap_sweep_timeout_s = 0.05
        # Isolate the reap: no-op the touch + log-heartbeat calls the loop
        # also makes each poll.
        monkeypatch.setattr(worker, '_touch_owned_merge_worktrees', lambda: 0)
        monkeypatch.setattr(worker, '_maybe_log_queue_heartbeat', lambda now: False)

        calls = 0

        async def _hanging_reap(now: float) -> None:
            nonlocal calls
            calls += 1
            await asyncio.sleep(10.0)  # never completes on its own

        monkeypatch.setattr(worker, '_maybe_reap_orphaned_merge_worktrees', _hanging_reap)

        worker._running = True
        task = asyncio.create_task(worker._heartbeat_loop())
        try:
            # Poll for the second timeout-bounded reap instead of a
            # hard-coded wall-clock window: under full-suite xdist
            # contention the event loop can be starved badly enough that a
            # fixed 0.3s window only ever observes one poll (task 3684).
            # Without the asyncio.wait_for bound, the first hung reap would
            # block for the full 10s sleep and `calls` would never reach 2,
            # so the deadline below is exhausted and the assertion still
            # fails — the guard keeps its teeth.
            deadline = asyncio.get_running_loop().time() + 5.0
            while calls < 2 and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.02)
            worker._running = False
            timed_out = False
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except TimeoutError:
                # Guard-regression case: the first hung reap is still
                # sleeping unbounded. Let the assertions below report
                # it cleanly instead of surfacing a raw TimeoutError.
                timed_out = True
        finally:
            if not task.done():
                task.cancel()

        assert calls >= 2, (
            f'a hung reap must be cut off by _reap_sweep_timeout_s so the loop '
            f'keeps polling instead of blocking on the first hung sweep '
            f'forever; got {calls} call(s) in the observation window'
        )
        assert not timed_out, '_heartbeat_loop must exit promptly once _running is False'

    async def test_heartbeat_loop_reap_survives_raising_touch(
        self, git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The converse of test_heartbeat_loop_survives_raising_reap: a
        raising _touch_owned_merge_worktrees (caught by the FIRST,
        touch/heartbeat try/except) must never suppress the periodic reap
        (run by the SECOND, separate try/except) on that same poll.

        test_heartbeat_loop_survives_raising_reap alone couldn't pin the
        substantive claim in _heartbeat_loop's docstring — "a fault in the
        periodic reap can never suppress — or be suppressed by — the
        touch/heartbeat step" — because the reap runs AFTER the
        touch/heartbeat block, so a raising reap could never have suppressed
        it anyway. This test exercises the direction that actually depends
        on the two try/excepts being separate: before that separation, a
        raising touch would have short-circuited anything appended to its
        try-block (task 3018 amendment; reviewer_comprehensive test-coverage
        finding at test_merge_queue_orphan_reaper.py:592).
        """
        import orchestrator.merge_queue as mq_mod
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        monkeypatch.setattr(mq_mod, '_HEARTBEAT_POLL_S', 0.0)

        def _raising_touch() -> None:
            raise RuntimeError('boom: touch exploded')

        monkeypatch.setattr(worker, '_touch_owned_merge_worktrees', _raising_touch)
        monkeypatch.setattr(worker, '_maybe_log_queue_heartbeat', lambda now: False)

        calls: list[float] = []

        async def _fake_reap(now: float) -> None:
            calls.append(now)
            # Stop the loop deterministically after the first reap so
            # awaiting the coroutine below returns instead of spinning.
            worker._running = False

        monkeypatch.setattr(worker, '_maybe_reap_orphaned_merge_worktrees', _fake_reap)

        worker._running = True
        await asyncio.wait_for(worker._heartbeat_loop(), timeout=2.0)

        assert len(calls) >= 1, (
            'the periodic reap must still run on a poll where the touch/'
            'heartbeat step raised — it is a SEPARATE try/except, not '
            'short-circuited by the touch/heartbeat guard'
        )


# ---------------------------------------------------------------------------
# task 3018 (steps 13-14): the periodic sweep must not reuse the audit's
# DETECTION grace as its DESTRUCTION deadline.
#
# Promoting reap_orphaned_merge_worktrees from a startup-only sweep to a
# steady-state one (steps 1-8 above) silently re-purposed
# RESOURCE_AUDIT_WORKTREE_GRACE_SECS from a *detection* threshold into a
# *destruction* deadline.  Those are different questions with different right
# answers: "old enough to be worth REPORTING" is deliberately eager, while
# "old enough that destroying it cannot interrupt live work" must be
# conservative.  Splitting them keeps detection eager (a tree in the band is
# still reported every heartbeat) while giving destruction a strictly larger,
# derived floor.
#
# This is defense-in-depth BEHIND the primary protection (steps 10/12: live
# throwaway verify worktrees now hold merge_verify_lease(lane_dir=wt), so the
# reap skips them via 'skipped_lease_held' regardless of age).  It exists to
# cover any live-but-unleased tree on a path not yet enumerated.
# ---------------------------------------------------------------------------


def _shipped_cold_command_timeout_secs() -> float:
    """The shipped per-command cold merge-verify budget from defaults.yaml.

    Read from the packaged defaults rather than hardcoded so a retune of the
    shipped budget is caught by the derivation assertion below instead of
    silently invalidating it.
    """
    from orchestrator.config import _load_defaults

    value = _load_defaults().get('merge_verify_cold_command_timeout_secs')
    assert value is not None, (
        'defaults.yaml no longer ships merge_verify_cold_command_timeout_secs — '
        'the destructive floor below is DERIVED from it, so this test must be '
        'updated alongside that removal rather than silently losing its basis'
    )
    return float(value)


@pytest.mark.asyncio
class TestPeriodicReapDestructiveFloor:
    """The periodic sweep destroys only past PERIODIC_REAP_MIN_AGE_SECS (3018).

    RED until step-14 GREEN adds PERIODIC_REAP_MIN_AGE_SECS and the
    ``min_age_secs`` parameter: today the periodic helper destroys at the
    detection grace, so a tree in the detect-but-do-not-destroy band is
    deleted.
    """

    async def test_periodic_floor_exceeds_detection_grace_and_cold_ceiling(
        self,
    ) -> None:
        """The floor is DERIVED, not guessed — assert the relations it must hold.

        Asserts relations rather than the literal 21600.0 so a future retune of
        either constant stays free as long as the derivation still holds.

        (a) strictly above the DETECTION grace — otherwise the two thresholds
            are the same number wearing two hats, which is exactly the
            conflation this split exists to undo; and
        (b) at or above 2x the shipped per-command cold budget.  That budget is
            PER-COMMAND and a cold merge verify runs >=2 command phases (scoped
            verify + unscoped typechecks) before counting worktree creation and
            venv/dep seeding, so 2x it is the floor under the total-lifetime
            ceiling of a legitimately slow cold verify.

        This is also why the reviewer-suggested
        INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS (10800) was NOT reused: it is
        below that 2x ceiling, so it would not actually clear a slow cold
        verify.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        floor = SpeculativeMergeWorker.PERIODIC_REAP_MIN_AGE_SECS
        detection_grace = SpeculativeMergeWorker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS
        cold_ceiling = 2 * _shipped_cold_command_timeout_secs()

        assert floor > detection_grace, (
            f'the periodic DESTRUCTION floor ({floor}) must be strictly larger '
            f'than the audit DETECTION grace ({detection_grace}) — reusing one '
            f'number for both is the conflation this split exists to undo'
        )
        assert floor >= cold_ceiling, (
            f'the periodic DESTRUCTION floor ({floor}) must clear the derived '
            f'cold-verify total-lifetime ceiling ({cold_ceiling} = 2 x the '
            f'shipped per-command cold budget), else a legitimately slow cold '
            f'verify can still have its checkout deleted mid-run'
        )
        assert floor < 8 * 3600, (
            f'the periodic DESTRUCTION floor ({floor}) must stay below the 8h '
            f'fleet-redeploy window, else the periodic sweep reclaims nothing '
            f'the next restart would not have reclaimed anyway — defeating the '
            f'whole point of task 3018'
        )

    async def test_tree_past_detection_grace_but_below_floor_survives_periodic_sweep(
        self, git_ops: GitOps,
    ) -> None:
        """A tree in the detect-but-do-not-destroy band must NOT be reaped."""
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        worker.PERIODIC_REAP_MIN_AGE_SECS = _GRACE * 3
        worker._last_reap_at = 0.0

        # Past DETECTION (age > _GRACE) but below the DESTRUCTION floor.
        in_band = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)

        await worker._maybe_reap_orphaned_merge_worktrees(now=_NOW)

        assert in_band.exists(), (
            f'a worktree aged past the DETECTION grace but below the periodic '
            f'DESTRUCTION floor must survive the sweep — it may still be a live '
            f'verify: {in_band}'
        )
        registered = await _registered_worktree_paths(git_ops)
        assert str(in_band) in registered, (
            f'the surviving worktree must also still be registered with git '
            f'(not half-torn-down): {registered}'
        )

    async def test_detection_still_flags_tree_below_destructive_floor(
        self, git_ops: GitOps,
    ) -> None:
        """Detection is deliberately UNCHANGED — it keeps observing in the band.

        The audit stays observation-only (PRD design decision 4) and keeps
        flagging at RESOURCE_AUDIT_WORKTREE_GRACE_SECS, so a tree in the
        detect-but-do-not-destroy band is still reported on every heartbeat.
        It is simply not destroyed yet.  Raising the destruction floor must
        NOT have silently raised the detection threshold with it.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        worker.PERIODIC_REAP_MIN_AGE_SECS = _GRACE * 3
        worker._last_reap_at = 0.0

        in_band = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)

        await worker._maybe_reap_orphaned_merge_worktrees(now=_NOW)

        violations = worker.worktree_ledger_violations(now=_NOW)
        assert any(str(in_band) in v for v in violations), (
            f'the audit must STILL flag the un-destroyed in-band worktree '
            f'{in_band} — detect-early/destroy-late means the operator keeps '
            f'seeing it every heartbeat, got violations={violations!r}'
        )

    async def test_tree_past_periodic_floor_is_reaped(self, git_ops: GitOps) -> None:
        """Past the destruction floor, the periodic sweep still reclaims."""
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        worker.PERIODIC_REAP_MIN_AGE_SECS = _GRACE * 3
        worker._last_reap_at = 0.0

        aged = await _create_backdated_merge_worktree(git_ops, age=_GRACE * 3 + 10)

        await worker._maybe_reap_orphaned_merge_worktrees(now=_NOW)

        assert not aged.exists(), (
            f'a worktree aged past the periodic DESTRUCTION floor must still be '
            f'reclaimed in-run — raising the floor must not disable the sweep: '
            f'{aged}'
        )
        registered = await _registered_worktree_paths(git_ops)
        assert str(aged) not in registered, (
            f'the reaped worktree must be gone from git worktree list: {registered}'
        )

    async def test_startup_reap_still_uses_detection_grace(
        self, git_ops: GitOps,
    ) -> None:
        """The startup/harness caller is byte-identical — no floor applied.

        ``Harness._reap_orphaned_merge_worktrees`` calls
        ``reap_orphaned_merge_worktrees`` with no ``min_age_secs``, and at
        startup there is by construction no in-process live verify to protect
        (the worker's own loops have not run a verify yet).  Pins that the new
        parameter defaults back to RESOURCE_AUDIT_WORKTREE_GRACE_SECS so that
        caller's behaviour is unchanged.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = _GRACE
        worker.PERIODIC_REAP_MIN_AGE_SECS = _GRACE * 3

        orphan = await _create_backdated_merge_worktree(git_ops, age=_GRACE + 10)

        report = await worker.reap_orphaned_merge_worktrees(now=_NOW)

        assert not orphan.exists(), (
            f'the direct (startup/harness) caller must keep reaping at the '
            f'DETECTION grace, not the larger periodic floor: {orphan}'
        )
        assert report['reaped'] == [str(orphan)], (
            f"the startup sweep's report must be unchanged, got {report!r}"
        )
