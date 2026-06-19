"""Integration tests for warm-lane base coherence contract (D8/D10).

DF consumer side of the reify activation-seam R1:
- D8: _seed_warm_lane resolves a gen-dir symlink base to the concrete .gen.N dir
      and holds flock -s on .gen.N.lock during the cp.
- D10: refresh_warm_base passes --landed-commit <sha> to satisfy reify's inv.9 guard.

This file is self-contained (no shared conftest fixtures beyond tmp_path).
Scaffolding patterns mirror test_warm_lane_pool.py.
"""

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run


# ---------------------------------------------------------------------------
# Scaffolding helpers (module-local, mirrors test_warm_lane_pool.py patterns)
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    """Initialise a bare git repo suitable for worktree tests."""
    await _run(['git', 'init', '-b', 'main', str(repo)])
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _make_lane_dir(repo: Path, lane_name: str = '_lane-0') -> Path:
    """Create a real worktree at <repo>/.worktrees/<lane_name>."""
    worktree_base = repo / '.worktrees'
    worktree_base.mkdir(parents=True, exist_ok=True)
    lane = worktree_base / lane_name
    await _run(
        ['git', 'worktree', 'add', '-b', f'task/{lane_name}', str(lane), 'HEAD'],
        cwd=repo,
    )
    return lane


def _make_gendir_base(base_dir: Path) -> tuple[Path, Path, Path]:
    """Create a reify-style gen-dir base layout.

    Creates:
        <base_dir>/.gen.0/          — the concrete gen dir
        <base_dir>/.gen.0/sentinel  — sentinel content file
        <base_dir>/target           — symlink -> .gen.0 (relative sibling)
        <base_dir>/.gen.0.lock      — reader-refcount lock file (reify GC protocol)

    Returns:
        (base_dir, gen_dir, target_symlink)
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = base_dir / '.gen.0'
    gen_dir.mkdir()
    (gen_dir / 'sentinel').write_text('gen-dir-content\n')
    target_symlink = base_dir / 'target'
    # Relative sibling, matches reify's: ln -sfn .gen.N target
    target_symlink.symlink_to('.gen.0')
    (base_dir / '.gen.0.lock').touch()
    return base_dir, gen_dir, target_symlink


def _wl_config_with_base(base_target_dir: str) -> GitConfig:
    """GitConfig with warm_lane_pool=True and warm_lane_base_target_dir set."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        warm_lane_base_target_dir=base_target_dir,
    )


def _install_recording_seed_stub(lane: Path, marker: Path) -> None:
    """Install a seed-warm-lane.sh that records its argv to *marker* and exits 0."""
    scripts_dir = lane / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'seed-warm-lane.sh'
    script.write_text(f'#!/usr/bin/env bash\necho "$@" >> {marker}\n')
    script.chmod(0o755)


# ===========================================================================
# Step 1 (RED): seed resolves gen-dir symlink / back-compat plain dir
# ===========================================================================


@pytest.mark.asyncio
class TestSeedSymlinkResolution:
    """D8 — _seed_warm_lane resolves gen-dir symlink base to concrete .gen.N dir."""

    async def test_seed_resolves_symlink_to_gen_dir(self, tmp_path: Path) -> None:
        """When warm_lane_base_target_path is a ``target``→``.gen.N`` symlink,
        _seed_warm_lane must pass the RESOLVED .gen.N directory path to the
        script, NOT the symlink path.

        RED against today's impl which passes str(warm_lane_base_target_path) raw.
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_repo(repo)

        # Build a reify-style gen-dir base: .gen.0/ + sentinel + target->.gen.0 + .gen.0.lock
        base_dir = tmp_path / 'bases'
        _, gen_dir, target_symlink = _make_gendir_base(base_dir)

        # Point warm_lane_base_target_dir at the symlink path (as reify sets it)
        config = _wl_config_with_base(str(target_symlink))
        git_ops = GitOps(config, repo, warm_lane_pool_size=1)

        lane = await _make_lane_dir(repo, '_lane-0')
        marker = lane / 'seed_args.txt'
        _install_recording_seed_stub(lane, marker)

        result = await git_ops._seed_warm_lane(lane, '--fresh-checkout')

        assert result is True
        assert marker.exists(), 'seed-warm-lane.sh was not called'
        # First arg recorded is base_target
        recorded_base_target = marker.read_text().strip().split()[0]

        # Must be the RESOLVED .gen.0 path, NOT the target symlink
        assert recorded_base_target == str(gen_dir), (
            f'Expected resolved gen dir {gen_dir!r}, '
            f'got {recorded_base_target!r}'
        )
        assert recorded_base_target != str(target_symlink), (
            'base_target must NOT be the symlink path'
        )

    async def test_seed_plaindir_base_passes_raw_path(self, tmp_path: Path) -> None:
        """Back-compat: plain directory (not a symlink) is passed raw, unchanged.

        Existing default config (base == _merge-verify/target plain dir) must keep
        working unchanged. GREEN against today's impl and must stay green after step-2.
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_repo(repo)

        # Plain directory — no symlink
        plain_dir = tmp_path / 'plain_base'
        plain_dir.mkdir()

        config = _wl_config_with_base(str(plain_dir))
        git_ops = GitOps(config, repo, warm_lane_pool_size=1)

        lane = await _make_lane_dir(repo, '_lane-0')
        marker = lane / 'seed_args.txt'
        _install_recording_seed_stub(lane, marker)

        result = await git_ops._seed_warm_lane(lane, '--fresh-checkout')

        assert result is True
        assert marker.exists()
        recorded_base_target = marker.read_text().strip().split()[0]

        # Must pass the RAW plain dir path (no resolution)
        assert recorded_base_target == str(plain_dir), (
            f'Expected raw path {plain_dir!r}, got {recorded_base_target!r}'
        )
