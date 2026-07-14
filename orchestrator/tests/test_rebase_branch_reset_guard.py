"""Regression tests for the requeue/inter-iteration rebase branch-reset guard
(RCA + fix, task 2403).

Incident: during merge-train churn around a frozen-prefix merge tip
(d48743f90a), the orchestrator's requeue/inter-iteration rebase path
silently RESET two unrelated task branches (task/2261, task/2223) to that
tip with ZERO commits ahead of main, wiping each task's committed work.
Reflog signature: ``branch: Created from main`` then ``rebase (finish):
... onto d48743f90a``. ``git rebase <ref>`` records the RESOLVED sha, so
"onto d48743f90a" only means main's tip WAS d48743 at rebase time — the
branch's own commits were dropped DURING the rebase, not that an unrelated
ref was deliberately chosen as --onto.

Fix: a mechanism-independent POST-CONDITION guard —
:meth:`GitOps.rebase_preserving_task_commits` — wraps the existing
``rebase_onto_main`` primitive with a pre/post "commits ahead of main"
check. If the wrapped rebase reports success but the branch's own commits
vanished (n_before > 0 and n_after == 0), the guard restores the
pre-rebase HEAD and raises :class:`BranchResetError` instead of silently
returning success.

This file starts RED (step-1): neither ``GitOps.rebase_preserving_task_commits``
nor ``BranchResetError`` exist yet on ``orchestrator.git_ops`` — the import
below fails collection until step-2 implements them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import BranchResetError, GitOps, WorktreeInfo, _run

# ---------------------------------------------------------------------------
# Fixtures — copied from test_git_ops.py's git_repo/git_config/git_ops trio
# (test_git_ops.py:35/114/126) so this file builds real temp git repos
# without importing test-internal helpers from another test module.
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


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


async def _assert_no_rebase_in_progress(wt_path: Path) -> None:
    """Assert the worktree has no rebase in progress (copied from
    test_git_ops.py's helper of the same name)."""
    _, gitdir_str, _ = await _run(['git', 'rev-parse', '--git-dir'], cwd=wt_path)
    gitdir = Path(gitdir_str.strip())
    if not gitdir.is_absolute():
        gitdir = wt_path / gitdir
    rebase_merge = gitdir / 'rebase-merge'
    assert not rebase_merge.exists(), (
        f'rebase-merge directory {rebase_merge} should not exist — '
        'rebase was not properly aborted'
    )


async def _commits_over_main(worktree: Path) -> int:
    _, out, _ = await _run(
        ['git', 'rev-list', '--count', 'main..HEAD'], cwd=worktree,
    )
    return int(out.strip())


async def _head_sha(worktree: Path) -> str:
    _, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
    return out.strip()


async def _commit_unique_work(worktree: Path, filename: str = 'feature.txt') -> None:
    (worktree / filename).write_text('unique task work\n')
    await _run(['git', 'add', '-A'], cwd=worktree)
    await _run(['git', 'commit', '-m', 'task: add feature'], cwd=worktree)


async def _wipe_via_reset(worktree: Path, onto: str | None = None) -> bool:
    """Fake ``rebase_onto_main`` replacement: collapses the branch straight
    onto *onto* (default 'main') via ``git reset --hard`` and reports
    success — used to engineer a deterministic zero-commit wipe without
    depending on git-version-specific patch-id-drop heuristics during a real
    rebase. The guard's contract is mechanism-independent (see module
    docstring), so simulating the "rebase reported success but the branch's
    commits are gone" observable is sufficient to exercise it.
    """
    target = onto if onto is not None else 'main'
    await _run(['git', 'reset', '--hard', target], cwd=worktree)
    return True


# ---------------------------------------------------------------------------
# step-1: GitOps.rebase_preserving_task_commits + BranchResetError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRebasePreservingTaskCommitsGuard:
    async def test_guard_raises_and_restores_on_zero_commit_wipe(
        self, git_ops: GitOps, git_repo: Path, monkeypatch,
    ):
        """A rebase that collapses the branch to 0 commits over main raises
        BranchResetError and restores the pre-rebase HEAD (work preserved)."""
        wt_info = await git_ops.create_worktree('wipe-me')
        wt = wt_info.path
        await _commit_unique_work(wt)

        pre_head = await _head_sha(wt)
        assert await _commits_over_main(wt) == 1

        # Engineer the wipe: rebase_onto_main "succeeds" but actually
        # collapses HEAD onto main via reset (git_ops.py primitive is
        # patched, not `git rebase` behavior itself — see _wipe_via_reset).
        monkeypatch.setattr(git_ops, 'rebase_onto_main', _wipe_via_reset)

        with pytest.raises(BranchResetError):
            await git_ops.rebase_preserving_task_commits(wt)

        assert await _head_sha(wt) == pre_head, 'pre-rebase HEAD must be restored'
        assert await _commits_over_main(wt) == 1, 'task commit must survive'

    async def test_guard_happy_path_retains_commits(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """A clean rebase onto an advanced main retains the branch's own
        commit and returns True without raising."""
        wt_info = await git_ops.create_worktree('happy')
        wt = wt_info.path
        await _commit_unique_work(wt)

        # Advance main with an UNRELATED sibling commit.
        (git_repo / 'sibling.txt').write_text('sibling change\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'sibling change'], cwd=git_repo)

        result = await git_ops.rebase_preserving_task_commits(wt)

        assert result is True
        assert await _commits_over_main(wt) == 1

    async def test_guard_passthrough_on_real_conflict_returns_false(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """A genuine rebase conflict makes the guard return False (not
        raise); the branch is left clean on its old base."""
        wt_info = await git_ops.create_worktree('conflict')
        wt = wt_info.path
        # Branch and main both add fileX.txt with different content ->
        # add/add conflict on rebase (mirrors test_git_ops.py's
        # TestRebaseOntoArbitraryRef conflict pattern).
        (wt / 'fileX.txt').write_text('branch version\n')
        await _run(['git', 'add', '-A'], cwd=wt)
        await _run(['git', 'commit', '-m', 'task: add fileX'], cwd=wt)

        (git_repo / 'fileX.txt').write_text('main version\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'main: add fileX'], cwd=git_repo)

        pre_head = await _head_sha(wt)

        result = await git_ops.rebase_preserving_task_commits(wt)

        assert result is False
        assert await _head_sha(wt) == pre_head, 'branch must be left on its old base'
        await _assert_no_rebase_in_progress(wt)

    async def test_guard_noop_when_branch_had_zero_commits_over_main(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """A branch with no commits of its own (pure fast-forward) returns
        True and never raises — no false positive on an empty task."""
        wt_info = await git_ops.create_worktree('empty')
        wt = wt_info.path
        assert await _commits_over_main(wt) == 0

        (git_repo / 'sibling.txt').write_text('sibling change\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'sibling change'], cwd=git_repo)

        result = await git_ops.rebase_preserving_task_commits(wt)

        assert result is True
        assert await _commits_over_main(wt) == 0


# ---------------------------------------------------------------------------
# step-3: the two git_ops requeue reuse sites route THROUGH the guard.
#
# Warm-lane helpers copied from test_git_ops.py (`_seed_default_warm_base`
# at :43, `_add_warm_lane_scripts` at :6636) so this file can drive
# `_reuse_warm_lane` without importing test-internal helpers from another
# test module.
# ---------------------------------------------------------------------------


def _seed_default_warm_base(repo: Path) -> None:
    """Pre-create the DEFAULT derived warm-lane base (task 2061) so the
    acquire_warm_lane pre-acquire base-health gate sees WarmBaseHealth.OK."""
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()


async def _add_warm_lane_scripts(repo: Path, port: int = 39411) -> None:
    """Commit stub seed-warm-lane.sh + setup-worktree-debug-port.sh into repo."""
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed_script = scripts_dir / 'seed-warm-lane.sh'
    seed_script.write_text(
        '#!/usr/bin/env bash\nmkdir -p "$2/target"\necho "seeded" > "$2/target/seeded.bin"\n'
    )
    seed_script.chmod(0o755)
    debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
    debug_script.write_text(f'#!/usr/bin/env bash\necho {port}\n')
    debug_script.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add warm-lane scripts'], cwd=repo)


@pytest.mark.asyncio
class TestRequeueReuseSitesRouteThroughGuard:
    """The cold-requeue (``create_worktree``) and live-requeue
    (``_reuse_warm_lane``) reuse paths must route through
    ``rebase_preserving_task_commits`` so a wipe raises BranchResetError
    and preserves the branch's commit instead of silently losing it.

    RED today (step-3): both sites still call the unguarded
    ``rebase_onto_main`` directly (rewired to the guard in step-4).
    """

    async def test_create_worktree_reuse_raises_on_wipe(
        self, git_ops: GitOps, git_repo: Path, monkeypatch,
    ):
        """A second create_worktree() call (the cold-requeue reuse block)
        that would zero the branch must raise BranchResetError and leave
        the branch's commit intact."""
        wt_info = await git_ops.create_worktree('requeue-wipe')
        wt = wt_info.path
        await _commit_unique_work(wt)
        assert await _commits_over_main(wt) == 1

        monkeypatch.setattr(git_ops, 'rebase_onto_main', _wipe_via_reset)

        with pytest.raises(BranchResetError):
            await git_ops.create_worktree('requeue-wipe')

        assert await _commits_over_main(wt) == 1, (
            'branch commit must survive the cold-requeue reuse rebase'
        )

    async def test_reuse_warm_lane_raises_on_wipe(
        self, git_repo: Path, monkeypatch,
    ):
        """_reuse_warm_lane (the live-requeue reuse path) that would zero
        the branch must raise BranchResetError and preserve the commit."""
        _seed_default_warm_base(git_repo)
        await _add_warm_lane_scripts(git_repo)
        _, start_ref_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        start_ref = start_ref_raw.strip()

        warm_config = GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        )
        warm_git_ops = GitOps(warm_config, git_repo, warm_lane_pool_size=1)

        info = await warm_git_ops.acquire_warm_lane('WR', start_ref)
        assert isinstance(info, WorktreeInfo), f'Acquire failed: {info!r}'
        lane = info.path

        await _commit_unique_work(lane, filename='wr_feature.txt')
        assert await _commits_over_main(lane) == 1

        monkeypatch.setattr(warm_git_ops, 'rebase_onto_main', _wipe_via_reset)

        with pytest.raises(BranchResetError):
            await warm_git_ops._reuse_warm_lane(lane, 'task/WR')

        assert await _commits_over_main(lane) == 1, (
            'branch commit must survive the live-requeue (warm-lane) reuse rebase'
        )
