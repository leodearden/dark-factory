"""Real-git tests for a LOCKED worktree admin entry whose tree is gone (task 4828).

git keeps listing such an entry in ``git worktree list --porcelain``, and
``git worktree prune`` never reclaims it while its ``locked`` marker exists.
In production one ``_merge-*`` entry of this shape made every
``merge_request`` enqueue fail with ``Worktree missing: <path>``, while the
already-queued entries kept landing.

TestLockedDanglingEntryDoesNotPoisonMergeEnumeration pins that merge-worktree
enumeration, and through it the enqueue gate, skips such an entry rather than
raising.
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
