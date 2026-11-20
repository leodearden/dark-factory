"""A warm-lane resume must never hand its task branch a FOREIGN commit (task 10991).

On 2026-11-18 reify task/10089 lost its commit on a lane resume.  The lane had
been left on another task's branch with that task's abandoned rebase still on
disk; a defensive abort reset the other branch and checked it out, and the
resume's ``checkout -B task/10089`` then adopted that foreign HEAD as
task/10089's tip.  These cases drive the real ``GitOps.acquire_warm_lane``
over a real repository, through public seams only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from _lane_test_helpers import (
    acquire_lane,
    add_warm_lane_scripts,
    get_head,
    make_warm_lane_repo,
)

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, WarmLaneUnavailable, WorktreeInfo, _run


def _warm_git_ops(repo: Path) -> GitOps:
    config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
    )
    return GitOps(config, repo, warm_lane_pool_size=1)


async def _git(cwd: Path, *args: str) -> str:
    rc, out, err = await _run(['git', *args], cwd=cwd)
    assert rc == 0, f'git {args!r} failed in {cwd}: {err}'
    return out.strip()


async def _commit_file(lane: Path, name: str) -> str:
    (lane / name).write_text(f'{name}\n')
    await _git(lane, 'add', name)
    await _git(lane, 'commit', '-m', f'add {name}')
    return await _git(lane, 'rev-parse', 'HEAD')


class _Scene:
    def __init__(self, repo: Path, git_ops: GitOps, main: str) -> None:
        self.repo = repo
        self.git_ops = git_ops
        self.main = main

    async def tip(self, branch: str) -> str:
        return await _git(self.repo, 'rev-parse', f'refs/heads/{branch}')


@pytest_asyncio.fixture
async def scene(tmp_path: Path) -> _Scene:
    repo = make_warm_lane_repo(tmp_path)
    await add_warm_lane_scripts(repo)
    return _Scene(repo, _warm_git_ops(repo), await get_head(repo))


@pytest.mark.asyncio
class TestLiveRequeueOntoAForeignHead:
    """REUSE route: 'V' is acquired twice with no release in between."""

    async def _lane_with_c1_on_v(self, scene: _Scene) -> tuple[Path, str]:
        lane = await acquire_lane(scene.git_ops, 'V', scene.main)
        c1 = await _commit_file(lane, 'c1.txt')
        await _git(scene.repo, 'branch', 'task/F', 'main')
        return lane, c1

    async def _assert_resumed_on_v(self, scene: _Scene, lane: Path) -> None:
        assert await _git(lane, 'rev-parse', '--abbrev-ref', 'HEAD') == 'task/V'
        await _git(scene.repo, 'show', 'task/V:c1.txt')
        assert int(await _git(scene.repo, 'rev-list', '--count', 'main..task/V')) >= 1
        assert await scene.tip('task/F') == scene.main

    async def test_a_lane_left_on_a_foreign_branch_resumes_on_its_own(
        self, scene: _Scene,
    ) -> None:
        lane, _ = await self._lane_with_c1_on_v(scene)
        await _git(lane, 'checkout', 'task/F')

        result = await scene.git_ops.acquire_warm_lane('V', scene.main)

        assert isinstance(result, WorktreeInfo), result
        await self._assert_resumed_on_v(scene, lane)

    async def test_a_lane_detached_at_a_foreign_commit_resumes_on_its_own(
        self, scene: _Scene,
    ) -> None:
        lane, _ = await self._lane_with_c1_on_v(scene)
        await _git(lane, 'checkout', '--detach', 'main')

        result = await scene.git_ops.acquire_warm_lane('V', scene.main)

        assert isinstance(result, WorktreeInfo), result
        await self._assert_resumed_on_v(scene, lane)

    async def test_tracked_changes_on_a_foreign_branch_fault_instead_of_landing(
        self, scene: _Scene,
    ) -> None:
        lane, c1 = await self._lane_with_c1_on_v(scene)
        await _git(lane, 'checkout', 'task/F')
        (lane / 'README.md').write_text('# edited on a foreign branch\n')

        result = await scene.git_ops.acquire_warm_lane('V', scene.main)

        assert result is WarmLaneUnavailable.FAULT
        assert await scene.tip('task/V') == c1
        assert await scene.tip('task/F') == scene.main
