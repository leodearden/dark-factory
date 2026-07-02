"""Tests for GitOps.create_interactive_worktree — task α (2010).

Isolated interactive warm-worktree primitive: mints a fresh worktree in a new
_iact-* band and CoW-seeds its build cache by reusing _seed_warm_lane, WITHOUT
ever touching WarmLanePool (isolation invariant I1 — strictly disjoint from
the _lane-* dispatch pool and the _spec-* merge-speculation pool).

Mirrors test_warm_lane_integration_gate.py conventions: a committed seed stub
that creates <lane>/target/seeded.bin (orchestration-observable seededness —
literal filefrag/CoW extent-sharing is a reify/filesystem guarantee, NOT
asserted on a CI tmpfs), an ig-style git-repo fixture, and WarmLanePool
FREE-count / assignments_snapshot / is_lane assertions for the I1 signal.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from orchestrator import config as orchestrator_config
from orchestrator.config import GitConfig
from orchestrator.git_ops import (
    GitOps,
    InteractiveWorktreeInfo,
    InteractiveWorktreeLimitError,
    _run,
)


# ---------------------------------------------------------------------------
# Repo fixture + seed-stub helper (mirrors test_warm_lane_integration_gate.py)
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def iw_git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit for interactive-worktree tests."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _add_seed_script(repo: Path, *, exit_code: int = 0) -> None:
    """Commit a seed-warm-lane.sh stub into repo/scripts/.

    On exit_code == 0: creates <lane>/target/seeded.bin (orchestration-observable
    seededness — mirrors _add_all_warm_lane_scripts' seed stub in
    test_warm_lane_integration_gate.py).  On non-zero: exits with that code
    immediately (no target/ created).
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed = scripts_dir / 'seed-warm-lane.sh'
    if exit_code == 0:
        seed.write_text(
            '#!/usr/bin/env bash\n'
            'mkdir -p "$2/target"\n'
            'echo "seeded" > "$2/target/seeded.bin"\n'
        )
    else:
        seed.write_text(
            f'#!/usr/bin/env bash\necho "seed failure" >&2\nexit {exit_code}\n'
        )
    seed.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add seed-warm-lane.sh stub'], cwd=repo)


async def _commit_file(repo: Path, name: str, content: str, message: str) -> str:
    """Write+commit a file on the repo's current branch; return the new commit SHA."""
    (repo / name).write_text(content)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', message], cwd=repo)
    rc, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'rev-parse HEAD failed after committing {name!r}'
    return sha.strip()


# ---------------------------------------------------------------------------
# Step-01: config knobs
# ---------------------------------------------------------------------------


class TestInteractiveWorktreeConfigDefaults:
    """RED — config knobs: GitConfig + packaged defaults.yaml carry the three new knobs."""

    def test_gitconfig_defaults(self) -> None:
        """GitConfig() exposes max_interactive_worktrees/interactive_worktree_ttl/iact_prefix."""
        config = GitConfig()
        assert config.max_interactive_worktrees == 2, (
            f'expected default max_interactive_worktrees == 2, '
            f'got {config.max_interactive_worktrees!r}'
        )
        assert config.interactive_worktree_ttl == 86400.0, (
            f'expected default interactive_worktree_ttl == 86400.0, '
            f'got {config.interactive_worktree_ttl!r}'
        )
        assert config.iact_prefix == '_iact-', (
            f"expected default iact_prefix == '_iact-', got {config.iact_prefix!r}"
        )

    def test_packaged_defaults_carry_knobs(self) -> None:
        """orchestrator.config._load_defaults()['git'] carries the same three knobs."""
        defaults = orchestrator_config._load_defaults()
        git_defaults = defaults['git']
        assert git_defaults['iact_prefix'] == '_iact-', (
            f"expected packaged defaults.yaml git.iact_prefix == '_iact-', "
            f"got {git_defaults.get('iact_prefix')!r}"
        )
        assert git_defaults['max_interactive_worktrees'] == 2, (
            f"expected packaged defaults.yaml git.max_interactive_worktrees == 2, "
            f"got {git_defaults.get('max_interactive_worktrees')!r}"
        )
        assert git_defaults['interactive_worktree_ttl'] == 86400.0, (
            f"expected packaged defaults.yaml git.interactive_worktree_ttl == 86400.0, "
            f"got {git_defaults.get('interactive_worktree_ttl')!r}"
        )


# ---------------------------------------------------------------------------
# Step-03: primitive core happy path (pool DISABLED)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateInteractiveWorktreeHappyPath:
    """RED — create_interactive_worktree core happy path, no WarmLanePool involved."""

    async def test_creates_worktree_on_iact_branch_warm_and_base_ref(
        self, iw_git_repo: Path,
    ) -> None:
        """Fresh create: path/branch/warm/base_ref all match the documented contract."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        rc, main_sha, _ = await _run(['git', 'rev-parse', 'main'], cwd=iw_git_repo)
        assert rc == 0, 'setup: rev-parse main failed'
        main_sha = main_sha.strip()

        info = await git_ops.create_interactive_worktree('demo-slug')

        assert isinstance(info, InteractiveWorktreeInfo), (
            f'create_interactive_worktree must return InteractiveWorktreeInfo, '
            f'got {info!r}'
        )
        assert info.path == git_ops.worktree_base / '_iact-demo-slug', (
            f'expected path worktree_base/_iact-demo-slug, got {info.path}'
        )
        assert info.branch == 'task/demo-slug', (
            f"expected branch 'task/demo-slug', got {info.branch!r}"
        )
        assert info.warm is True, f'expected warm is True (seed ran), got {info.warm!r}'
        assert info.base_ref == main_sha, (
            f'expected base_ref == main tip {main_sha!r}, got {info.base_ref!r}'
        )

        # Registered worktree, HEAD on task/demo-slug
        rc_branch, branch_out, _ = await _run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=info.path,
        )
        assert rc_branch == 0 and branch_out.strip() == 'task/demo-slug', (
            f'expected HEAD on task/demo-slug, got {branch_out.strip()!r} (rc={rc_branch})'
        )
        rc_list, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=iw_git_repo,
        )
        assert rc_list == 0 and str(info.path.resolve()) in wt_list, (
            f'expected {info.path} to be a registered git worktree; '
            f'worktree list: {wt_list!r}'
        )

        # Seed ran: target/seeded.bin present (CoW-populated, orchestration-observable)
        assert (info.path / 'target' / 'seeded.bin').exists(), (
            'expected target/seeded.bin to exist after seed run (CoW-populated)'
        )

    async def test_explicit_start_ref_pins_base_ref(self, iw_git_repo: Path) -> None:
        """A caller-supplied start_ref (a prior commit SHA) pins base_ref exactly."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        # Create a distinct prior commit to pin, then advance main further so
        # main's tip != prior_sha (proves the pin — not the fresh tip — was used).
        prior_sha = await _commit_file(
            iw_git_repo, 'second.txt', 'second\n', 'second commit',
        )
        await _commit_file(iw_git_repo, 'third.txt', 'third\n', 'third commit')

        info = await git_ops.create_interactive_worktree(
            'pinned-slug', start_ref=prior_sha,
        )
        assert isinstance(info, InteractiveWorktreeInfo), (
            f'create_interactive_worktree must return InteractiveWorktreeInfo, '
            f'got {info!r}'
        )
        assert info.base_ref == prior_sha, (
            f'expected base_ref == pinned start_ref {prior_sha!r}, got {info.base_ref!r}'
        )


# ---------------------------------------------------------------------------
# Step-05: .task/interactive.json stamp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInteractiveWorktreeStamp:
    """RED — .task/interactive.json stamp written for the δ reaper to consume."""

    async def test_stamp_written_and_gitignored(self, iw_git_repo: Path) -> None:
        """The stamp records owner/created_at and is never git-tracked."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info = await git_ops.create_interactive_worktree('stamp-slug')

        stamp_path = info.path / '.task' / 'interactive.json'
        assert stamp_path.exists(), f'expected stamp to exist at {stamp_path}'

        stamp = json.loads(stamp_path.read_text())
        assert stamp['owner'] == 'stamp-slug', (
            f"expected stamp['owner'] == 'stamp-slug', got {stamp.get('owner')!r}"
        )
        # Machine-parseable ISO8601 — raises ValueError on failure.
        datetime.fromisoformat(stamp['created_at'])

        # Not git-tracked: ls-files --error-unmatch must fail (non-zero rc),
        # and `git status --porcelain` must not list it (gitignored via
        # .task/.gitignore, written by _ensure_task_gitignore).
        rc_ls, _, _ = await _run(
            ['git', 'ls-files', '--error-unmatch', '.task/interactive.json'],
            cwd=info.path,
        )
        assert rc_ls != 0, (
            'expected .task/interactive.json to NOT be git-tracked '
            '(git ls-files --error-unmatch unexpectedly succeeded)'
        )
        rc_status, status_out, _ = await _run(
            ['git', 'status', '--porcelain'], cwd=info.path,
        )
        assert rc_status == 0 and 'interactive.json' not in status_out, (
            f'expected .task/interactive.json to be absent from git status '
            f'(gitignored); got status: {status_out!r}'
        )


# ---------------------------------------------------------------------------
# Step-07: fail-soft warmth (the key deviation from acquire_warm_lane)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateInteractiveWorktreeFailSoftWarmth:
    """RED — a seed fault must be fail-soft: worktree retained, warm=False, no raise."""

    async def test_absent_seed_script_returns_cold_but_usable(
        self, iw_git_repo: Path,
    ) -> None:
        """No scripts/seed-warm-lane.sh committed (rc=127) → cold, usable, stamped."""
        # Deliberately do NOT commit a seed script.
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info = await git_ops.create_interactive_worktree('cold-a')

        assert info.warm is False, (
            f'expected warm is False (no seed script present), got {info.warm!r}'
        )
        assert info.path.exists(), f'expected {info.path} to exist'

        rc_list, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=iw_git_repo,
        )
        assert rc_list == 0 and str(info.path.resolve()) in wt_list, (
            f'expected {info.path} to be a registered git worktree even though '
            f'cold; worktree list: {wt_list!r}'
        )
        assert (info.path / '.task' / 'interactive.json').exists(), (
            'expected the stamp to be present even though the worktree is cold '
            '(worktree is usable)'
        )

    async def test_seed_script_failure_retains_worktree(self, iw_git_repo: Path) -> None:
        """Seed script exiting 1 → worktree RETAINED (not removed), warm=False, no raise."""
        await _add_seed_script(iw_git_repo, exit_code=1)
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info = await git_ops.create_interactive_worktree('cold-b')

        assert info.warm is False, (
            f'expected warm is False (seed script exited 1), got {info.warm!r}'
        )
        assert info.path.exists(), (
            f'expected {info.path} to exist — fail-soft retention, NOT removed'
        )

        rc_list, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=iw_git_repo,
        )
        assert rc_list == 0 and str(info.path.resolve()) in wt_list, (
            f'expected {info.path} to remain a REGISTERED git worktree after a '
            f'seed failure (fail-soft retention, not the acquire_warm_lane '
            f'remove-on-failure behaviour); worktree list: {wt_list!r}'
        )


# ---------------------------------------------------------------------------
# Step-09: max_interactive_worktrees cap (REJECT)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateInteractiveWorktreeCap:
    """RED — max_interactive_worktrees enforces a REJECT cap before any git op."""

    async def test_cap_rejects_third_worktree_then_frees_on_removal(
        self, iw_git_repo: Path,
    ) -> None:
        """Cap of 2: 'a'/'b' succeed, 'c' is rejected, 'd' succeeds once freed."""
        await _add_seed_script(iw_git_repo)
        config = GitConfig(max_interactive_worktrees=2)
        git_ops = GitOps(config, iw_git_repo)

        info_a = await git_ops.create_interactive_worktree('a')
        info_b = await git_ops.create_interactive_worktree('b')
        assert info_a.path.exists() and info_b.path.exists(), (
            'setup: both a and b should be created within the cap'
        )

        with pytest.raises(InteractiveWorktreeLimitError):
            await git_ops.create_interactive_worktree('c')

        c_path = git_ops.worktree_base / '_iact-c'
        assert not c_path.exists(), (
            f'expected {c_path} to NOT exist after a cap-rejected create'
        )
        rc_branch, _, _ = await _run(
            ['git', 'rev-parse', '--verify', 'task/c'], cwd=iw_git_repo,
        )
        assert rc_branch != 0, (
            'expected no task/c branch to have been created by the rejected call'
        )

        # Free a slot by removing one of the existing interactive worktrees.
        rc_remove, _, remove_err = await _run(
            ['git', 'worktree', 'remove', '--force', str(info_a.path)],
            cwd=iw_git_repo,
        )
        assert rc_remove == 0, f'setup: failed to remove {info_a.path}: {remove_err}'

        info_d = await git_ops.create_interactive_worktree('d')
        assert info_d.path.exists(), (
            'expected create_interactive_worktree("d") to succeed once a slot freed'
        )
