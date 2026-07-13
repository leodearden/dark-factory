"""Tests for the ``.pool-root`` sentinel guarding destructive worktree_base sweeps.

Incident (task 2099): after a Jul-3 host crash, reify's orchestrator started
BEFORE the warm-lanes XFS mount came up.  ``worktree_base`` (the mountpoint
dir) existed but was EMPTY, so the existing ``worktree_base.exists()`` guards
were satisfied and the orphan-reaper's unconditional ``prune_worktrees()``
tail ran ``git worktree prune`` against a repo whose mount-resident worktree
dirs all APPEARED missing — wiping every registered ``_lane-*`` +
``_merge-verify`` admin entry.

``GitOps.pool_storage_present()`` answers "is the pool storage actually
mounted?" via a ``.pool-root`` sentinel FILE that lives ON the pool storage
itself — substrate-independent (works for plain dirs and real mounts alike,
no config knob).  ``GitOps.mark_pool_storage_present()`` writes it.  It is
written from exactly ONE chokepoint, ``_seed_warm_lane`` on ``rc == 0``
(step-8), because a successful seed proves the mount is present and writable.

This module builds up the guard from the bottom:
  step-1/2  — the predicate + writer themselves
  step-3/4  — prune_worktrees() guard
  step-5/6  — _run_warm_lane_gc_reclaim() guard
  step-7/8  — acquire_warm_lane / acquire_spec_lane create-once discriminator
              + the _seed_warm_lane mark-on-success chokepoint
  step-9/10 — harness escalation helper + resolver (test_pool_storage_guard.py
              continues to grow via later iterations; harness-side tests may
              also live here)
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import (
    POOL_ROOT_SENTINEL,
    GitOps,
    WarmLaneUnavailable,
    WorktreeInfo,
    _run,
)
from orchestrator.harness import Harness
from orchestrator.warm_lane_pool import WarmLanePool


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
def git_ops(git_config: GitConfig, tmp_path: Path) -> GitOps:
    """A GitOps whose worktree_base (tmp_path/project/.worktrees) is NOT
    pre-created — lets tests control base-exists/sentinel-exists independently.

    project_root need not be a real git repo: these predicate/writer tests
    are pure filesystem checks that never shell out to git.
    """
    project_root = tmp_path / 'project'
    project_root.mkdir()
    return GitOps(git_config, project_root)


class TestPoolStoragePresentPredicate:
    """GitOps.pool_storage_present() / mark_pool_storage_present() (step-1/2)."""

    def test_sentinel_constant(self):
        assert POOL_ROOT_SENTINEL == '.pool-root'

    def test_base_missing_is_absent(self, git_ops: GitOps):
        assert not git_ops.worktree_base.exists()
        assert git_ops.pool_storage_present() is False

    def test_base_exists_no_sentinel_is_absent(self, git_ops: GitOps):
        """Simulates an unmounted mountpoint: the dir exists (empty) but the
        sentinel that lives ON the mounted storage was never written."""
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        assert git_ops.worktree_base.exists()
        assert git_ops.pool_storage_present() is False

    def test_mark_then_present_is_true(self, git_ops: GitOps):
        git_ops.mark_pool_storage_present()
        sentinel = git_ops.worktree_base / POOL_ROOT_SENTINEL
        assert sentinel.is_file()
        assert git_ops.pool_storage_present() is True

    def test_mark_creates_missing_base(self, git_ops: GitOps):
        """mark_pool_storage_present() mkdir(parents=True)s worktree_base
        when it does not yet exist (fresh-host / pool-warmup bootstrap)."""
        assert not git_ops.worktree_base.exists()
        git_ops.mark_pool_storage_present()
        assert git_ops.worktree_base.exists()
        assert git_ops.pool_storage_present() is True

    def test_mark_is_idempotent(self, git_ops: GitOps):
        git_ops.mark_pool_storage_present()
        git_ops.mark_pool_storage_present()  # second call: no-op, no raise
        sentinel = git_ops.worktree_base / POOL_ROOT_SENTINEL
        assert sentinel.is_file()
        assert git_ops.pool_storage_present() is True

    def test_present_fail_safe_on_oserror(self, git_ops: GitOps, monkeypatch):
        """A stat that raises OSError must yield False (fail-safe-absent),
        never propagate — an unreadable mount must skip destructive sweeps,
        not crash them."""
        git_ops.mark_pool_storage_present()
        assert git_ops.pool_storage_present() is True

        def _raise(*_args, **_kwargs):
            raise OSError('simulated stat failure')

        monkeypatch.setattr(Path, 'is_file', _raise)
        assert git_ops.pool_storage_present() is False


class TestPoolInUsePredicate:
    """GitOps.pool_in_use() (step-15/16 review-fix).

    ``pool_storage_present()``'s only writer (``_seed_warm_lane`` on
    ``rc == 0``) NEVER runs unless a warm or spec pool is configured, so on
    the DEFAULT config (``warm_lane_pool=False``,
    ``merge_spec_warm_lane_pool=False``) ``.pool-root`` is never written and
    ``pool_storage_present()`` is PERMANENTLY False by design — not because a
    real mount went down. ``pool_in_use()`` lets the destructive-sweep
    guards distinguish "no pool configured at all" (skip the guard, restore
    pre-2099 behaviour) from "a pool IS configured but its mount is down"
    (fire the guard — the real Jul-3 incident).
    """

    def test_no_pools_configured_is_false(self, git_ops: GitOps):
        assert git_ops.warm_lane_pool is None
        assert git_ops.spec_warm_lane_pool is None
        assert git_ops.pool_in_use() is False

    def test_warm_lane_pool_configured_is_true(self, git_ops: GitOps):
        git_ops.warm_lane_pool = Mock()
        assert git_ops.pool_in_use() is True

    def test_spec_warm_lane_pool_configured_is_true(self, git_ops: GitOps):
        git_ops.spec_warm_lane_pool = Mock()
        assert git_ops.pool_in_use() is True


class TestPoolStorageBootstrapOk:
    """GitOps._pool_storage_bootstrap_ok() (task 2099 review-fix).

    Distinguishes a fresh-host first-seed BOOTSTRAP (warm base present &
    non-empty INSIDE worktree_base — mount provably up, sentinel just not
    written yet) from a genuine unmounted mountpoint (base absent) or an
    off-mount base config (base present elsewhere, says nothing about
    worktree_base's own mount).  Fail-safe: anything but the proven case
    returns False.
    """

    def test_base_present_inside_worktree_base_is_true(self, git_ops: GitOps):
        base = git_ops.worktree_base / '_merge-verify' / 'target'
        base.mkdir(parents=True, exist_ok=True)
        (base / '.keep').write_text('warm base\n')
        assert git_ops._pool_storage_bootstrap_ok() is True

    def test_base_absent_is_false(self, git_ops: GitOps):
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)  # empty mountpoint
        assert git_ops._pool_storage_bootstrap_ok() is False

    def test_base_empty_dir_is_false(self, git_ops: GitOps):
        base = git_ops.worktree_base / '_merge-verify' / 'target'
        base.mkdir(parents=True, exist_ok=True)  # exists but EMPTY -> ABSENT
        assert git_ops._pool_storage_bootstrap_ok() is False

    def test_off_mount_base_is_false(self, git_config: GitConfig, tmp_path: Path):
        """Base configured OFF worktree_base: even present & non-empty, its
        presence on another mount cannot prove worktree_base's own mount, so
        bootstrap is refused (a shadow lane could still land on the root fs)."""
        off_mount = tmp_path / 'elsewhere' / 'base'
        off_mount.mkdir(parents=True, exist_ok=True)
        (off_mount / '.keep').write_text('off-mount base\n')
        cfg = GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False,
            warm_lane_base_target_dir=str(off_mount),
        )
        project_root = tmp_path / 'project2'
        project_root.mkdir()
        git_ops = GitOps(cfg, project_root)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)  # empty mountpoint
        assert git_ops._warm_lane_base_resolvable().name == 'OK'
        assert git_ops._pool_storage_bootstrap_ok() is False


class TestReconcilePoolStorageBeforeSweep:
    """GitOps._reconcile_pool_storage_before_sweep(context) (task 2315, BUG 2).

    Shared predicate + best-effort sentinel-recreation helper that both
    destructive-sweep sites (``_run_warm_lane_gc_reclaim``,
    ``_prune_registrations``) route through, so a HEALTHY mount that merely
    lost its ``.pool-root`` sentinel (active lanes, populated
    ``_merge-verify/target``) self-heals — recreates the sentinel and
    proceeds — instead of refusing the sweep forever (the chicken-and-egg
    described in the task 2315 analysis: sweeps refused -> stale lanes never
    reseeded -> the only sentinel writer never runs -> sentinel stays
    missing).
    """

    def test_healthy_mount_missing_sentinel_recreates_and_proceeds(
        self, git_ops: GitOps,
    ):
        """Case A: bootstrap-ok True (mount provably healthy) but the
        sentinel is absent -- helper must recreate it and return True
        without escalating."""
        git_ops.warm_lane_pool = WarmLanePool(worktree_base=git_ops.worktree_base, size=1)
        base = git_ops.worktree_base / '_merge-verify' / 'target'
        base.mkdir(parents=True, exist_ok=True)
        (base / '.keep').write_text('warm base\n')
        assert not git_ops.pool_storage_present()
        assert git_ops._pool_storage_bootstrap_ok()

        callback = Mock()
        git_ops._on_pool_storage_absent = callback

        result = git_ops._reconcile_pool_storage_before_sweep('unit-test-ctx')

        assert result is True
        assert git_ops.pool_storage_present() is True, (
            'expected the sentinel to be recreated on the healthy-mount path'
        )
        callback.assert_not_called()

    def test_suspected_unmount_refuses_and_escalates(self, git_ops: GitOps):
        """Case B: bootstrap-ok False (base absent/empty -- the genuine
        unmounted-mountpoint case) -- helper must refuse (False), leave the
        sentinel absent, and escalate exactly once."""
        git_ops.warm_lane_pool = WarmLanePool(worktree_base=git_ops.worktree_base, size=1)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        assert not git_ops.pool_storage_present()
        assert not git_ops._pool_storage_bootstrap_ok()

        callback = Mock()
        git_ops._on_pool_storage_absent = callback

        result = git_ops._reconcile_pool_storage_before_sweep('unit-test-ctx')

        assert result is False
        assert git_ops.pool_storage_present() is False
        callback.assert_called_once()

    def test_sentinel_present_returns_true_without_touching_callback(
        self, git_ops: GitOps,
    ):
        """Case C: sentinel already present -- normal proceed, callback untouched."""
        git_ops.warm_lane_pool = WarmLanePool(worktree_base=git_ops.worktree_base, size=1)
        git_ops.mark_pool_storage_present()
        assert git_ops.pool_storage_present()

        callback = Mock()
        git_ops._on_pool_storage_absent = callback

        result = git_ops._reconcile_pool_storage_before_sweep('unit-test-ctx')

        assert result is True
        callback.assert_not_called()

    def test_pool_not_in_use_returns_true_without_touching_callback(
        self, git_ops: GitOps,
    ):
        """Case D: no pool configured at all (pool_in_use() False) --
        pool_storage_present() is permanently False by design on a
        pool-less host, so it must never be mistaken for a mount-down
        incident."""
        assert git_ops.warm_lane_pool is None
        assert git_ops.spec_warm_lane_pool is None
        assert not git_ops.pool_in_use()

        callback = Mock()
        git_ops._on_pool_storage_absent = callback

        result = git_ops._reconcile_pool_storage_before_sweep('unit-test-ctx')

        assert result is True
        callback.assert_not_called()


@pytest.mark.asyncio
class TestPruneWorktreesGuard:
    """prune_worktrees() refuses ``git worktree prune`` when pool storage is
    absent (step-3/4) — this is the direct fix for the Jul-3 incident: an
    unmounted mountpoint dir must never let git "clean up" every registered
    lane + _merge-verify admin entry."""

    async def test_prune_skipped_when_storage_absent(self, git_ops: GitOps, caplog):
        # base exists (dir present) but the sentinel was never written —
        # simulates an unmounted mountpoint with a live, empty mount dir.
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        # A pool must be configured for this guard to fire (step-15
        # review-fix): pool_storage_present() is permanently False on a
        # pool-less host by design (see TestPoolInUsePredicate and
        # test_prune_runs_when_no_pool_configured below), so pool_in_use()
        # is what distinguishes a real mount-down incident from that.
        git_ops.warm_lane_pool = WarmLanePool(worktree_base=git_ops.worktree_base, size=1)
        assert not git_ops.pool_storage_present()

        callback = Mock()
        git_ops._on_pool_storage_absent = callback
        mock_run = AsyncMock(return_value=(0, '', ''))

        with (
            caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', mock_run),
        ):
            await git_ops.prune_worktrees()

        mock_run.assert_not_called()
        callback.assert_called_once()
        assert any('pool storage' in r.message.lower() for r in caplog.records), (
            f'Expected a loud WARNING naming pool storage; got '
            f'{[r.message for r in caplog.records]}'
        )

    async def test_prune_runs_when_storage_present(self, git_ops: GitOps):
        git_ops.mark_pool_storage_present()
        assert git_ops.pool_storage_present()
        mock_run = AsyncMock(return_value=(0, '', ''))

        with patch('orchestrator.git_ops._run', mock_run):
            await git_ops.prune_worktrees()

        mock_run.assert_awaited_once_with(
            ['git', 'worktree', 'prune'], cwd=git_ops.project_root,
        )

    async def test_prune_runs_when_no_pool_configured(self, git_ops: GitOps):
        """Step-15 review-fix: on a pool-less host (both pools None — the
        default git_ops fixture) `.pool-root` is never written, so
        pool_storage_present() is permanently False by design. The guard
        must NOT treat that as a mount-down incident — prune must proceed
        normally (restoring pre-2099 behaviour) and the callback must not
        fire."""
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        assert git_ops.warm_lane_pool is None
        assert git_ops.spec_warm_lane_pool is None
        assert not git_ops.pool_in_use()
        assert not git_ops.pool_storage_present()

        callback = Mock()
        git_ops._on_pool_storage_absent = callback
        mock_run = AsyncMock(return_value=(0, '', ''))

        with patch('orchestrator.git_ops._run', mock_run):
            await git_ops.prune_worktrees()

        mock_run.assert_awaited_once_with(
            ['git', 'worktree', 'prune'], cwd=git_ops.project_root,
        )
        callback.assert_not_called()


def _write_warm_lane_gc_stub(project_root: Path) -> Path:
    """Write a minimal ``warm-lane-gc.sh`` stub at ``<project_root>/scripts/``.

    Mirrors test_warm_lane_disk_guard.py's ``_write_disk_guard_stubs`` gc
    stub — existence is all ``_run_warm_lane_gc_reclaim``'s ``script.exists()``
    check cares about; ``_run`` itself is mocked in these tests so the stub
    body never actually runs.
    """
    scripts_dir = project_root / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'warm-lane-gc.sh'
    script.write_text('#!/usr/bin/env bash\nexit 0\n')
    script.chmod(0o755)
    return script


@pytest.mark.asyncio
class TestWarmLaneGcReclaimGuard:
    """_run_warm_lane_gc_reclaim() refuses to spawn warm-lane-gc.sh when pool
    storage is absent (step-5/6) — same rationale as the prune guard: an
    unmounted mountpoint must never let the GC script reclaim/reset lanes
    against it."""

    async def test_reclaim_skipped_when_storage_absent(self, git_ops: GitOps, caplog):
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        _write_warm_lane_gc_stub(git_ops.project_root)
        # A pool must be configured for this guard to fire (step-15
        # review-fix) — see test_prune_skipped_when_storage_absent above for
        # the full rationale.
        git_ops.warm_lane_pool = WarmLanePool(worktree_base=git_ops.worktree_base, size=1)
        assert not git_ops.pool_storage_present()

        callback = Mock()
        git_ops._on_pool_storage_absent = callback
        mock_run = AsyncMock(return_value=(0, '', ''))

        with (
            caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', mock_run),
        ):
            rc = await git_ops._run_warm_lane_gc_reclaim()

        assert rc == 127, f'Expected fail-soft 127 sentinel, got {rc}'
        mock_run.assert_not_called()
        callback.assert_called_once()
        assert any('pool storage' in r.message.lower() for r in caplog.records), (
            f'Expected a loud WARNING naming pool storage; got '
            f'{[r.message for r in caplog.records]}'
        )

    async def test_reclaim_runs_when_storage_present(self, git_ops: GitOps):
        git_ops.mark_pool_storage_present()
        script = _write_warm_lane_gc_stub(git_ops.project_root)
        assert git_ops.pool_storage_present()
        mock_run = AsyncMock(return_value=(0, '', ''))

        with patch('orchestrator.git_ops._run', mock_run):
            rc = await git_ops._run_warm_lane_gc_reclaim()

        assert rc == 0
        mock_run.assert_awaited_once_with(
            [str(script), 'reclaim', '--mount', str(git_ops.worktree_base)],
            cwd=git_ops.project_root,
        )

    async def test_reclaim_runs_when_no_pool_configured(self, git_ops: GitOps):
        """Step-15 review-fix: same rationale as
        test_prune_runs_when_no_pool_configured — no pool configured means
        pool_storage_present() is permanently False by design, not a real
        mount-down incident; the guard must not fire and reclaim must
        proceed normally."""
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        script = _write_warm_lane_gc_stub(git_ops.project_root)
        assert git_ops.warm_lane_pool is None
        assert git_ops.spec_warm_lane_pool is None
        assert not git_ops.pool_in_use()
        assert not git_ops.pool_storage_present()

        callback = Mock()
        git_ops._on_pool_storage_absent = callback
        mock_run = AsyncMock(return_value=(0, '', ''))

        with patch('orchestrator.git_ops._run', mock_run):
            rc = await git_ops._run_warm_lane_gc_reclaim()

        assert rc == 0
        mock_run.assert_awaited_once_with(
            [str(script), 'reclaim', '--mount', str(git_ops.worktree_base)],
            cwd=git_ops.project_root,
        )
        callback.assert_not_called()


# ---------------------------------------------------------------------------
# Real-repo fixtures for the acquire create-once discriminator (step-7/8).
# Mirrors test_git_ops.py's git_repo / _seed_default_warm_base / _add_warm_lane_scripts.
# ---------------------------------------------------------------------------


async def _init_acquire_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


def _seed_default_warm_base(repo: Path) -> None:
    """Pre-create the DEFAULT derived warm-lane base (task 2061) so the
    acquire_warm_lane pre-acquire base-health gate sees WarmBaseHealth.OK.

    These tests exercise the task-2099 create-once DISCRIMINATOR — a
    DIFFERENT (later) gate — not the 2061 base-health gate.  Mirrors
    test_git_ops.py's ``_seed_default_warm_base``.  Deliberately does NOT
    mark ``.pool-root`` — that is exactly the condition under test.
    """
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')


async def _add_warm_lane_scripts(repo: Path) -> None:
    """Commit a stub seed-warm-lane.sh into repo/scripts/ (mirrors test_git_ops.py)."""
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed_script = scripts_dir / 'seed-warm-lane.sh'
    seed_script.write_text(
        '#!/usr/bin/env bash\nmkdir -p "$2/target"\necho "seeded" > "$2/target/seeded.bin"\n'
    )
    seed_script.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add warm-lane scripts'], cwd=repo)


@pytest.fixture
def acquire_git_repo(tmp_path: Path) -> Path:
    """Real git repo with the default warm base pre-seeded (2061 gate: OK),
    but WITHOUT the .pool-root sentinel.

    This is the FRESH-HOST BOOTSTRAP state (task 2099 review-fix): the CoW
    seed base (``.worktrees/_merge-verify/target``) is present & non-empty
    INSIDE worktree_base — proof the mount is up — yet ``.pool-root`` was
    never written because its only writer (``_seed_warm_lane`` on ``rc == 0``)
    lives PAST the create-once discriminator.  The discriminator must
    recognise this as a first-seed bootstrap and proceed (marking the
    sentinel), NOT mistake it for an unmounted mountpoint and refuse forever.

    Contrast ``empty_mountpoint_git_repo`` below, where the base is ABSENT —
    the genuine unmounted-mountpoint case that must still refuse.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_acquire_repo(repo))
    _seed_default_warm_base(repo)
    return repo


@pytest.fixture
def empty_mountpoint_git_repo(tmp_path: Path) -> Path:
    """Real git repo whose worktree_base exists but is EMPTY — the genuine
    unmounted-mountpoint case (task 2099).  No warm base target, no
    ``.pool-root``: ``_pool_storage_bootstrap_ok()`` is False, so the
    create-once discriminator must refuse (cold fallback), never bootstrap a
    shadow lane on the underlying root fs.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_acquire_repo(repo))
    (repo / '.worktrees').mkdir()  # empty mountpoint dir, no base target
    return repo


def _acquire_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        merge_spec_warm_lane_pool=True,
    )


@pytest.mark.asyncio
class TestAcquireCreateOnceDiscriminatorAndMarkChokepoint:
    """acquire_warm_lane / acquire_spec_lane create-once discriminator +
    the _seed_warm_lane mark-on-success chokepoint (step-7/8)."""

    async def test_acquire_warm_lane_create_once_bootstraps_when_base_present(
        self, acquire_git_repo: Path,
    ):
        """Fresh-host bootstrap (task 2099 review-fix): worktree_base exists
        with a present warm base but no ``.pool-root`` (its writer lives past
        this gate).  The create-once discriminator must recognise this as a
        first-seed bootstrap — mark the sentinel and PROCEED — not refuse
        forever.  Without this the very first warm lane on a fresh host
        deadlocks the pool permanently."""
        await _add_warm_lane_scripts(acquire_git_repo)
        git_ops = GitOps(
            _acquire_config(), acquire_git_repo,
            warm_lane_pool_size=1, merge_spec_warm_lane_pool_size=1,
        )
        # worktree_base (.worktrees) already exists via _seed_default_warm_base
        # (it created .worktrees/_merge-verify/target) — but .pool-root was
        # never written: the fresh-host bootstrap state.
        assert git_ops.worktree_base.exists()
        assert not git_ops.pool_storage_present()

        callback = Mock()
        git_ops._on_pool_storage_absent = callback
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=acquire_git_repo)
        start_ref = start_ref_raw.strip()

        result = await git_ops.acquire_warm_lane('A', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'Expected first-seed bootstrap to succeed, got {result!r}'
        )
        assert result.path == git_ops.worktree_base / '_lane-0'
        assert git_ops.pool_storage_present() is True, (
            'bootstrap must write .pool-root so subsequent acquires see storage present'
        )
        callback.assert_not_called()

    async def test_acquire_warm_lane_create_once_refuses_when_base_absent(
        self, empty_mountpoint_git_repo: Path,
    ):
        """Genuine unmounted-mountpoint (task 2099): worktree_base exists but
        is EMPTY (no base target).  The 2061 base-health gate short-circuits
        with BASE_ABSENT before the discriminator is even reached — no lane
        dir is created on the underlying root fs."""
        await _add_warm_lane_scripts(empty_mountpoint_git_repo)
        git_ops = GitOps(
            _acquire_config(), empty_mountpoint_git_repo,
            warm_lane_pool_size=1, merge_spec_warm_lane_pool_size=1,
        )
        assert git_ops.worktree_base.exists()
        assert not git_ops.pool_storage_present()
        assert not git_ops._pool_storage_bootstrap_ok()

        _, start_ref_raw, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=empty_mountpoint_git_repo,
        )
        start_ref = start_ref_raw.strip()

        result = await git_ops.acquire_warm_lane('A', start_ref)

        assert result is WarmLaneUnavailable.BASE_ABSENT, (
            f'Expected BASE_ABSENT requeue signal, got {result!r}'
        )
        assert not (git_ops.worktree_base / '_lane-0').exists()
        assert git_ops.pool_storage_present() is False, (
            'an empty unmounted mountpoint must never get a bootstrap sentinel'
        )

    async def test_acquire_spec_lane_create_once_bootstraps_when_base_present(
        self, acquire_git_repo: Path,
    ):
        """Fresh-host bootstrap for the spec pool (task 2099 review-fix).
        acquire_spec_lane has no 2061 base-health pre-gate, so its create-once
        discriminator is the primary gate — it must equally recognise the
        base-present/sentinel-absent state as a first-seed bootstrap."""
        await _add_warm_lane_scripts(acquire_git_repo)
        git_ops = GitOps(
            _acquire_config(), acquire_git_repo,
            warm_lane_pool_size=1, merge_spec_warm_lane_pool_size=1,
        )
        assert git_ops.worktree_base.exists()
        assert not git_ops.pool_storage_present()

        callback = Mock()
        git_ops._on_pool_storage_absent = callback
        _, head_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=acquire_git_repo)
        merge_commit = head_raw.strip()

        wt, warm = await git_ops.acquire_spec_lane(merge_commit)

        assert warm is True, f'Expected first-seed bootstrap warm success, got warm={warm}'
        assert wt == git_ops.worktree_base / '_spec-0'
        assert git_ops.pool_storage_present() is True
        callback.assert_not_called()

    async def test_acquire_spec_lane_create_once_refuses_when_base_absent(
        self, empty_mountpoint_git_repo: Path,
    ):
        """Genuine unmounted-mountpoint for the spec pool (task 2099): an empty
        worktree_base with no base target => _pool_storage_bootstrap_ok() False
        => refuse, cold fallback, callback fired, no spec lane on root fs."""
        await _add_warm_lane_scripts(empty_mountpoint_git_repo)
        git_ops = GitOps(
            _acquire_config(), empty_mountpoint_git_repo,
            warm_lane_pool_size=1, merge_spec_warm_lane_pool_size=1,
        )
        assert git_ops.worktree_base.exists()
        assert not git_ops.pool_storage_present()
        assert not git_ops._pool_storage_bootstrap_ok()

        callback = Mock()
        git_ops._on_pool_storage_absent = callback
        _, head_raw, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=empty_mountpoint_git_repo,
        )
        merge_commit = head_raw.strip()

        wt, warm = await git_ops.acquire_spec_lane(merge_commit)

        assert warm is False, f'Expected cold fallback (warm=False), got warm={warm}'
        assert not (git_ops.worktree_base / '_spec-0').exists(), (
            'create-once must NOT create the spec lane dir when storage is absent'
        )
        assert git_ops.pool_storage_present() is False
        callback.assert_called_once()

    async def test_seed_warm_lane_marks_sentinel_on_success(
        self, acquire_git_repo: Path,
    ):
        await _add_warm_lane_scripts(acquire_git_repo)
        git_ops = GitOps(_acquire_config(), acquire_git_repo)
        assert not git_ops.pool_storage_present()

        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=acquire_git_repo)
        start_ref = start_ref_raw.strip()
        lane = git_ops.worktree_base / 'manual-lane'
        rc_add, _, err = await _run(
            ['git', 'worktree', 'add', '--detach', str(lane), start_ref],
            cwd=acquire_git_repo,
        )
        assert rc_add == 0, f'setup: worktree add failed: {err}'

        rc = await git_ops._seed_warm_lane(lane, '--fresh-checkout')

        assert rc == 0, f'seed should succeed via the stub script, got rc={rc}'
        assert git_ops.pool_storage_present() is True, (
            '_seed_warm_lane must mark the .pool-root sentinel on rc==0'
        )

    async def test_seed_warm_lane_does_not_mark_sentinel_on_failure(
        self, acquire_git_repo: Path,
    ):
        # NO _add_warm_lane_scripts -> seed script absent -> rc == 127.
        git_ops = GitOps(_acquire_config(), acquire_git_repo)
        assert not git_ops.pool_storage_present()

        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=acquire_git_repo)
        start_ref = start_ref_raw.strip()
        lane = git_ops.worktree_base / 'manual-lane'
        rc_add, _, err = await _run(
            ['git', 'worktree', 'add', '--detach', str(lane), start_ref],
            cwd=acquire_git_repo,
        )
        assert rc_add == 0, f'setup: worktree add failed: {err}'

        rc = await git_ops._seed_warm_lane(lane, '--fresh-checkout')

        assert rc == 127, f'expected seed-script-absent sentinel, got rc={rc}'
        assert git_ops.pool_storage_present() is False, (
            '_seed_warm_lane must NOT mark the sentinel on non-zero rc'
        )

    async def test_acquire_warm_lane_create_once_proceeds_when_storage_present(
        self, acquire_git_repo: Path,
    ):
        """Control (regression guard): sentinel present -> normal create-once."""
        await _add_warm_lane_scripts(acquire_git_repo)
        git_ops = GitOps(
            _acquire_config(), acquire_git_repo,
            warm_lane_pool_size=1, merge_spec_warm_lane_pool_size=1,
        )
        git_ops.mark_pool_storage_present()
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=acquire_git_repo)
        start_ref = start_ref_raw.strip()

        result = await git_ops.acquire_warm_lane('A', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'Expected normal create-once success, got {result!r}'
        )
        assert result.path == git_ops.worktree_base / '_lane-0'

    async def test_acquire_spec_lane_create_once_proceeds_when_storage_present(
        self, acquire_git_repo: Path,
    ):
        """Control (regression guard): sentinel present -> normal create-once."""
        await _add_warm_lane_scripts(acquire_git_repo)
        git_ops = GitOps(
            _acquire_config(), acquire_git_repo,
            warm_lane_pool_size=1, merge_spec_warm_lane_pool_size=1,
        )
        git_ops.mark_pool_storage_present()
        _, head_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=acquire_git_repo)
        merge_commit = head_raw.strip()

        wt, warm = await git_ops.acquire_spec_lane(merge_commit)

        assert warm is True, f'Expected normal warm create-once success, got warm={warm}'
        assert wt == git_ops.worktree_base / '_spec-0'


# ---------------------------------------------------------------------------
# Harness escalation helper + resolver (step-9/10).
# Mirrors test_harness_warm_lane_wiring.py's _build_harness / _make_config /
# TestWarmBaseHardDownFilingDedupResolve patterns.
# ---------------------------------------------------------------------------


def _build_harness(config: OrchestratorConfig) -> Harness:
    """Construct a Harness with heavy constructors patched out."""
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        return Harness(config)


def _make_harness_with_queue(tmp_path: Path) -> tuple[Harness, EscalationQueue]:
    """Bare Harness with a real EscalationQueue wired on."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / '.git').mkdir()  # bare minimum to satisfy path checks
    config = OrchestratorConfig(
        project_root=repo, max_concurrent_tasks=2, git=GitConfig(),
    )
    harness = _build_harness(config)
    queue = EscalationQueue(tmp_path / 'esc')
    harness._escalation_queue = queue
    return harness, queue


@pytest.mark.asyncio
class TestPoolStorageAbsentEscalationHelperAndResolver:
    """_file_pool_storage_absent_escalation / _resolve_pool_storage_absent_escalation
    (step-9/10) — mirrors Harness._file_scheduler_pause_escalation's has_open_l1
    dedup pattern and _resolve_warm_base_hard_down's resolve-all-pending pattern."""

    async def test_files_one_pending_blocking_l1(self, tmp_path: Path):
        harness, queue = _make_harness_with_queue(tmp_path)

        harness._file_pool_storage_absent_escalation()

        pending = [
            e
            for e in queue.get_by_task(
                Harness._POOL_STORAGE_ABSENT_SENTINEL, status='pending',
            )
            if e.agent_role == Harness._POOL_STORAGE_ABSENT_ROLE
        ]
        assert len(pending) == 1, f'expected exactly one pending L1; got {pending!r}'
        esc = pending[0]
        assert esc.category == 'infra_issue', f'expected category="infra_issue"; got {esc.category!r}'
        assert esc.level == 1, f'expected level=1; got {esc.level}'
        assert esc.severity == 'blocking', f'expected severity="blocking"; got {esc.severity!r}'

    async def test_second_call_deduped_still_one_open(self, tmp_path: Path):
        harness, queue = _make_harness_with_queue(tmp_path)

        harness._file_pool_storage_absent_escalation()
        harness._file_pool_storage_absent_escalation()

        pending = [
            e
            for e in queue.get_by_task(
                Harness._POOL_STORAGE_ABSENT_SENTINEL, status='pending',
            )
            if e.agent_role == Harness._POOL_STORAGE_ABSENT_ROLE
        ]
        assert len(pending) == 1, (
            f'expected exactly one pending L1 after two calls (dedup); got {pending!r}'
        )

    async def test_no_escalation_queue_is_noop(self, tmp_path: Path):
        harness, _queue = _make_harness_with_queue(tmp_path)
        harness._escalation_queue = None

        harness._file_pool_storage_absent_escalation()  # must not raise

    async def test_resolve_clears_pending_l1(self, tmp_path: Path):
        harness, queue = _make_harness_with_queue(tmp_path)
        harness._file_pool_storage_absent_escalation()
        assert queue.has_open_l1(Harness._POOL_STORAGE_ABSENT_SENTINEL)

        await harness._resolve_pool_storage_absent_escalation()

        assert not queue.has_open_l1(Harness._POOL_STORAGE_ABSENT_SENTINEL), (
            'expected no open L1 after resolve'
        )


@pytest.mark.asyncio
class TestWarmLaneGcPassNoEscalationWhenPoolNotInUse:
    """Harness._run_warm_lane_gc_pass() must NOT file a pool-storage-absent
    escalation on a pool-less default host (step-15 review-fix).

    On the DEFAULT config (``warm_lane_pool=False``,
    ``merge_spec_warm_lane_pool=False``) BOTH ``git_ops.warm_lane_pool`` and
    ``spec_warm_lane_pool`` are None, so ``.pool-root`` is NEVER written
    (its only writer, ``_seed_warm_lane`` on ``rc == 0``, requires a pool) —
    ``pool_storage_present()`` is PERMANENTLY False by design, not because a
    real mount went down. ``warm_lane_gc_enabled`` defaults True, so this
    cadence tick fires automatically on every pool-less host; pre-fix it
    filed a spurious, unresolvable blocking L1 every 600s (the cadence gate
    at harness.py:4455 is independent of pool presence).
    """

    async def test_no_escalation_filed(self, tmp_path: Path):
        harness, queue = _make_harness_with_queue(tmp_path)
        assert harness.git_ops.warm_lane_pool is None
        assert harness.git_ops.spec_warm_lane_pool is None
        assert not harness.git_ops.pool_storage_present()

        await harness._run_warm_lane_gc_pass()

        assert not queue.has_open_l1(Harness._POOL_STORAGE_ABSENT_SENTINEL), (
            'a pool-less host must never file a pool-storage-absent escalation'
        )
