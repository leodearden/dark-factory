"""Real-git tests for a LOCKED worktree admin entry whose tree is gone (task 4828).

git keeps listing such an entry in ``git worktree list --porcelain``, and
``git worktree prune`` never reclaims it while its ``locked`` marker exists.
In production one ``_merge-*`` entry of this shape made every
``merge_request`` enqueue fail with ``Worktree missing: <path>``, while the
already-queued entries kept landing.

TestLockedDanglingEntryDoesNotPoisonMergeEnumeration pins that merge-worktree
enumeration, and through it the enqueue gate, skips such an entry rather than
raising. TestPruneReclaimsLockedDanglingEntries pins that the prune chokepoint
reclaims such an entry under ``worktree_base``, and nowhere else, and never
past the task-2099 pool-storage refusal.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path

import pytest
from _git_fixtures import seed_repo

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.types import (
    InFlightMergeRegistry,
    MergeRequest,
    QueuedBranch,
)
from orchestrator.merge_lane.worker import coalesce_or_enqueue_merge_request
from orchestrator.warm_lane_pool import WarmLanePool


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout


def _lane_admin_dir(lane: Path) -> Path:
    """Parse the ``.git/worktrees/<name>`` admin dir path out of a lane's
    ``.git`` pointer file (``gitdir: <repo>/.git/worktrees/<name>``)."""
    content = (lane / '.git').read_text().strip()
    prefix = 'gitdir:'
    assert content.startswith(prefix), f'unexpected worktree .git pointer: {content!r}'
    return Path(content[len(prefix):].strip())


def _registered_paths(repo: Path) -> set[str]:
    porcelain = _git(repo, 'worktree', 'list', '--porcelain')
    return {
        str(Path(line[len('worktree '):]).resolve())
        for line in porcelain.splitlines()
        if line.startswith('worktree ')
    }


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


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


async def _plant_locked_dangling_entry(git_ops: GitOps) -> tuple[Path, Path]:
    """Reproduce the incident: a real ``_merge-*`` registration whose admin
    entry carries git's interrupted-``worktree add`` lock and whose tree is
    gone. Returns ``(worktree_path, admin_dir)``."""
    head = _git(git_ops.project_root, 'rev-parse', 'HEAD').strip()
    wt = await git_ops.create_throwaway_verify_worktree(head)
    admin = _lane_admin_dir(wt)
    (admin / 'locked').write_text('initializing')
    shutil.rmtree(wt)
    return wt, admin


@pytest.mark.asyncio
class TestLockedDanglingEntryDoesNotPoisonMergeEnumeration:
    async def test_precondition_git_keeps_the_entry_and_prune_skips_it(
        self, git_ops: GitOps, git_repo: Path,
    ):
        wt, _ = await _plant_locked_dangling_entry(git_ops)

        assert str(wt.resolve()) in _registered_paths(git_repo), (
            'git must keep listing a locked registration whose tree is gone'
        )
        dry_run = _git(git_repo, 'worktree', 'prune', '--dry-run', '-v')
        assert wt.name not in dry_run, (
            f'a bare prune must skip a locked entry; dry run said: {dry_run!r}'
        )

    async def test_find_inflight_skips_the_dangling_entry_with_a_warning(
        self, git_ops: GitOps, caplog,
    ):
        wt, _ = await _plant_locked_dangling_entry(git_ops)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            found = await git_ops.find_inflight_merge_worktree('some-branch')

        assert found is None
        assert any(
            record.levelno >= logging.WARNING and str(wt) in record.getMessage()
            for record in caplog.records
        ), f'expected a WARNING naming {wt}; got {[r.getMessage() for r in caplog.records]}'

    async def test_merge_request_enqueue_succeeds_with_the_entry_present(
        self, git_ops: GitOps, git_repo: Path, git_config: GitConfig,
    ):
        wt, _ = await _plant_locked_dangling_entry(git_ops)
        config = OrchestratorConfig(project_root=git_repo, git=git_config)
        req = MergeRequest(
            task_id='7001',
            branch=QueuedBranch.parse('7001', config.git.branch_prefix),
            worktree=git_repo,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=asyncio.get_running_loop().create_future(),
            lane='normal',
        )
        queue: asyncio.Queue = asyncio.Queue()

        result = await coalesce_or_enqueue_merge_request(
            queue, req, None, InFlightMergeRegistry(), git_ops=git_ops,
        )

        symptom = (
            f'a locked registration of the vanished {wt} must not block an '
            f'unrelated merge_request (task 4828: "Worktree missing: <path>" '
            f'failed every enqueue)'
        )
        assert result.dispatched is True, symptom
        assert queue.qsize() == 1, symptom


@pytest.mark.asyncio
class TestPruneReclaimsLockedDanglingEntries:
    async def test_locked_dangling_entry_under_worktree_base_is_reclaimed(
        self, git_ops: GitOps, git_repo: Path, caplog,
    ):
        wt, admin = await _plant_locked_dangling_entry(git_ops)

        await git_ops.prune_worktrees()

        assert not admin.exists(), 'the locked, tree-gone admin entry must be reclaimed'
        assert str(wt.resolve()) not in _registered_paths(git_repo)

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            assert await git_ops.find_inflight_merge_worktree('x') is None
        assert not any(str(wt) in record.getMessage() for record in caplog.records), (
            'a reclaimed entry is gone, so enumeration must no longer even skip it'
        )

    async def test_lock_on_a_live_tree_is_left_alone(
        self, git_ops: GitOps, git_repo: Path,
    ):
        head = _git(git_repo, 'rev-parse', 'HEAD').strip()
        wt = await git_ops.create_throwaway_verify_worktree(head)
        admin = _lane_admin_dir(wt)
        (admin / 'locked').write_text('initializing')

        await git_ops.prune_worktrees()

        assert wt.is_dir()
        assert (admin / 'locked').read_text() == 'initializing'
        assert str(wt.resolve()) in _registered_paths(git_repo)

    async def test_unlocked_dangling_entry_is_still_pruned(
        self, git_ops: GitOps, git_repo: Path,
    ):
        head = _git(git_repo, 'rev-parse', 'HEAD').strip()
        wt = await git_ops.create_throwaway_verify_worktree(head)
        shutil.rmtree(wt)

        await git_ops.prune_worktrees()

        assert str(wt.resolve()) not in _registered_paths(git_repo)

    async def test_locked_dangling_entry_outside_worktree_base_keeps_its_lock(
        self, git_ops: GitOps, git_repo: Path, tmp_path: Path,
    ):
        """git documents ``worktree lock`` for a worktree on a portable
        device that is not always mounted: locked and absent is its intended
        shape outside the orchestrator's namespace."""
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        portable = tmp_path / 'portable'
        _git(git_repo, 'worktree', 'add', '--detach', str(portable), 'HEAD')
        _git(git_repo, 'worktree', 'lock', '--reason', 'usb drive', str(portable))
        admin = _lane_admin_dir(portable)
        shutil.rmtree(portable)

        await git_ops.prune_worktrees()

        assert str(portable.resolve()) in _registered_paths(git_repo)
        assert (admin / 'locked').exists()

    async def test_pool_storage_refusal_leaves_the_lock_in_place(
        self, git_ops: GitOps, git_repo: Path,
    ):
        """An unmounted ``worktree_base`` makes every lane look dangling, so
        an unlock outside the task-2099 refusal gate would hand the next
        prune a wholesale wipe of every registration."""
        wt, admin = await _plant_locked_dangling_entry(git_ops)
        git_ops.warm_lane_pool = WarmLanePool(worktree_base=git_ops.worktree_base, size=1)
        assert git_ops.pool_in_use()
        assert not git_ops.pool_storage_present()

        await git_ops.prune_worktrees()

        assert (admin / 'locked').exists()
        assert str(wt.resolve()) in _registered_paths(git_repo)
