"""Real-git tests for the per-tick already-landed view and its abstain predicate."""

import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _git_fixtures import seed_repo

from orchestrator.already_landed_tick_view import (
    AlreadyLandedTickView,
    branch_absent_and_unmarked,
    read_already_landed_tick_view,
)
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps


@pytest.fixture
def git_ops(tmp_path: Path) -> GitOps:
    config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(config, seed_repo(tmp_path / 'repo'))


async def _land_and_delete_branch(git_ops: GitOps, tid: str) -> str:
    """Merge ``task/<tid>`` into main, then delete it; return the marker sha."""
    wt_info = await git_ops.create_worktree(tid)
    assert wt_info is not None
    (wt_info.path / f'{tid}.py').write_text(f'{tid} = True\n')
    await git_ops.commit(wt_info.path, f'Add {tid}')
    result = await git_ops.merge_to_main(wt_info.path, tid)
    assert result.success
    assert result.merge_commit is not None
    assert result.merge_worktree is not None
    adv = await git_ops.advance_main(result.merge_commit)
    assert adv.result == 'advanced'
    await git_ops.cleanup_merge_worktree(result.merge_worktree)
    await git_ops.cleanup_worktree(wt_info.path, tid)
    return result.merge_commit


@pytest.mark.asyncio
class TestAlreadyLandedTickView:
    async def test_reader_snapshots_the_local_branches_into_a_frozen_view(
        self, git_ops: GitOps,
    ):
        view = await read_already_landed_tick_view(git_ops)

        assert isinstance(view, AlreadyLandedTickView)
        assert view.local_branches == await git_ops.list_local_branches()
        with pytest.raises(dataclasses.FrozenInstanceError):
            view.local_branches = frozenset()  # type: ignore[misc]

    async def test_a_never_created_unmarked_branch_abstains(self, git_ops: GitOps):
        view = await read_already_landed_tick_view(git_ops)

        assert await branch_absent_and_unmarked(view, git_ops, 'task/never') is True

    async def test_a_listed_branch_does_not_abstain(self, git_ops: GitOps):
        await git_ops.create_worktree('listed')
        view = await read_already_landed_tick_view(git_ops)

        assert await branch_absent_and_unmarked(view, git_ops, 'task/listed') is False

    async def test_an_absent_but_marked_branch_does_not_abstain(
        self, git_ops: GitOps,
    ):
        await _land_and_delete_branch(git_ops, 'landed')
        view = await read_already_landed_tick_view(git_ops)
        assert view.local_branches is not None
        assert 'task/landed' not in view.local_branches

        assert await branch_absent_and_unmarked(view, git_ops, 'task/landed') is False

    async def test_an_unknown_enumeration_does_not_abstain_or_consult_markers(self):
        git_ops_double = MagicMock(spec=GitOps)
        git_ops_double.find_merge_marker = AsyncMock(return_value=None)
        view = AlreadyLandedTickView(local_branches=None)

        assert await branch_absent_and_unmarked(view, git_ops_double, 'task/x') is False
        git_ops_double.find_merge_marker.assert_not_awaited()

    async def test_a_stale_absence_over_an_old_marker_does_not_abstain(
        self, git_ops: GitOps,
    ):
        """The hazard task 4900 names: the branch is re-created after the view
        was read, over a marker an earlier incarnation left on main."""
        await _land_and_delete_branch(git_ops, 'x')
        stale_view = await read_already_landed_tick_view(git_ops)
        await git_ops.create_worktree('x')
        assert await git_ops.resolve_branch_sha('task/x') is not None

        assert await branch_absent_and_unmarked(stale_view, git_ops, 'task/x') is False
