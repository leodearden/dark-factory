"""Real-git boundary suite for the merge content relation (task 10215).

PRD ``plans/one-merge-request-per-task-prd.md`` leaf A: the D10 containment
classifier, the range patch-id reader, and the D6 content relation.  Every case
runs against a genuine temporary repository, because the answers are defined by
what ``git cherry``, ``git patch-id`` and ``git merge-base`` do to real
histories; a mock would only restate this file's assumptions about them.

The file imports merge-lane modules, so the merge-lane ratchet measures it: it
reads no private attribute and patches no lane name.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.landing_evidence import (
    Containment,
    GitReadFailed,
    RangePatchIds,
    classify_containment,
    is_ancestor_or_raise,
    range_patch_ids,
)

SEED = RepoSeed(
    files=(('README.md', '# content relation fixtures\n'),),
    message='chore: initial commit',
)

#: A well-formed sha that names no object in any fixture repository.
MISSING_SHA = 'deadbeef' * 5


def numbered_lines(prefix: str, count: int, *, start: int = 0) -> str:
    """*count* unique lines, so no two fixture commits share a patch id by coincidence."""
    return ''.join(f'{prefix}_{i:04d} = {i * 7919}\n' for i in range(start, start + count))


class GuardedRepo:
    """A fixture repository whose every git call carries both isolation guards."""

    def __init__(self, root: Path) -> None:
        assert_isolated_git_repo(root)
        self.root = root
        self.env = git_env_with_ceiling(root)

    def git(self, *args: str, input_text: str | None = None) -> str:
        proc = subprocess.run(
            ['git', *args], cwd=str(self.root), capture_output=True,
            env=self.env, text=True, check=False, input=input_text,
        )
        assert proc.returncode == 0, (
            f'git {" ".join(args)} failed (rc={proc.returncode}): {proc.stderr.strip()}'
        )
        return proc.stdout.strip()

    def commit(self, message: str, files: dict[str, str]) -> str:
        """Write *files*, stage everything and commit; return the new sha."""
        for rel, content in files.items():
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        self.git('add', '-A')
        self.git('commit', '-m', message)
        return self.sha('HEAD')

    def sha(self, ref: str) -> str:
        return self.git('rev-parse', ref)

    def patch_id(self, ref: str) -> str:
        """*ref*'s own ``git patch-id --stable``, read independently of the code under test."""
        diff = self.git('show', '--no-color', ref) + '\n'
        return self.git('patch-id', '--stable', input_text=diff).split()[0]


@pytest.fixture
def repo(tmp_path: Path) -> GuardedRepo:
    return GuardedRepo(seed_repo(tmp_path / 'repo', SEED))


@pytest.fixture
def git_ops(repo: GuardedRepo) -> GitOps:
    return GitOps(
        GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
        repo.root,
    )


def advance_main(repo: GuardedRepo, name: str) -> str:
    """Commit an unrelated file on main; return main's new tip."""
    repo.git('checkout', 'main')
    return repo.commit(f'chore: main advances ({name})', {f'main/{name}.txt': numbered_lines(name, 5)})


def branch_with_commit(repo: GuardedRepo, branch: str, *, start: str = 'main') -> str:
    """Create *branch* at *start* with one content-bearing commit; return its sha."""
    repo.git('checkout', '-b', branch, start)
    sha = repo.commit(f'feat: {branch} work', {f'work/{branch}.py': numbered_lines(branch, 12)})
    repo.git('checkout', 'main')
    return sha


@pytest.mark.asyncio
class TestClassifyContainment:
    async def test_tip_on_the_target_is_contained(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        tip = branch_with_commit(repo, 'task/merged')
        advance_main(repo, 'before-merge')
        repo.git('merge', '--no-ff', 'task/merged', '-m', 'Merge task/merged into main')

        assert await classify_containment(git_ops, tip=tip, target='main') is Containment.CONTAINED

    async def test_work_landed_by_rebase_is_contained(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        advance_main(repo, 'moved')
        tip = branch_with_commit(repo, 'task/rebased', start='main~1')
        repo.git('cherry-pick', tip)
        assert repo.git('cherry', 'main', tip).startswith('-')

        assert await classify_containment(git_ops, tip=tip, target='main') is Containment.CONTAINED

    async def test_merge_only_tip_beyond_the_target_is_not_contained(
        self, repo: GuardedRepo, git_ops: GitOps,
    ) -> None:
        advance_main(repo, 'merge-parent')
        tip = repo.git('commit-tree', 'main^{tree}', '-p', 'main~1', '-p', 'main', '-m', 'merge')
        assert repo.git('cherry', 'main', tip) == ''

        assert await classify_containment(git_ops, tip=tip, target='main') is Containment.NOT_CONTAINED

    async def test_unlanded_branch_commit_is_not_contained(
        self, repo: GuardedRepo, git_ops: GitOps,
    ) -> None:
        tip = branch_with_commit(repo, 'task/unlanded')
        advance_main(repo, 'unrelated')

        assert await classify_containment(git_ops, tip=tip, target='main') is Containment.NOT_CONTAINED

    async def test_unreadable_tip_is_unknown(self, git_ops: GitOps) -> None:
        assert await classify_containment(git_ops, tip=MISSING_SHA, target='main') is Containment.UNKNOWN


@pytest.mark.asyncio
class TestRangePatchIds:
    async def test_tip_on_the_target_has_an_empty_range(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        advance_main(repo, 'second')
        empty = RangePatchIds(patch_ids=(), has_merge_commit=False)

        assert await range_patch_ids(git_ops, target='main', tip=repo.sha('main')) == empty
        assert await range_patch_ids(git_ops, target='main', tip=repo.sha('main~1')) == empty

    async def test_two_commit_branch_gives_two_sorted_ids(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        repo.git('checkout', '-b', 'task/two')
        first = repo.commit('feat: first', {'work/one.py': numbered_lines('one', 8)})
        second = repo.commit('feat: second', {'work/two.py': numbered_lines('two', 8)})

        result = await range_patch_ids(git_ops, target='main', tip=second)

        assert len(result.patch_ids) == 2
        assert result.patch_ids == tuple(sorted(result.patch_ids))
        assert set(result.patch_ids) == {repo.patch_id(first), repo.patch_id(second)}
        assert result.has_merge_commit is False

    async def test_merge_in_range_is_flagged_and_contributes_no_id(
        self, repo: GuardedRepo, git_ops: GitOps,
    ) -> None:
        own = branch_with_commit(repo, 'task/synced')
        advance_main(repo, 'sync-source')
        repo.git('checkout', 'task/synced')
        repo.git('merge', '--no-ff', 'main', '-m', 'Merge main into task/synced')

        result = await range_patch_ids(git_ops, target='main', tip=repo.sha('HEAD'))

        assert result.has_merge_commit is True
        assert result.patch_ids == (repo.patch_id(own),)

    async def test_a_reapplied_diff_counts_twice(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        repo.git('checkout', '-b', 'task/readd')
        added = repo.commit('feat: add f', {'work/f.py': numbered_lines('f', 10)})
        repo.git('rm', '-q', 'work/f.py')
        repo.git('commit', '-m', 'feat: drop f')
        readded = repo.commit('feat: add f again', {'work/f.py': numbered_lines('f', 10)})

        result = await range_patch_ids(git_ops, target='main', tip=readded)

        assert len(result.patch_ids) == 3
        assert result.patch_ids.count(repo.patch_id(added)) == 2

    async def test_unreadable_tip_raises_with_git_facts(self, git_ops: GitOps) -> None:
        with pytest.raises(GitReadFailed) as caught:
            await range_patch_ids(git_ops, target='main', tip=MISSING_SHA)

        assert isinstance(caught.value.command, tuple)
        assert caught.value.command[0] == 'git'
        assert 'fatal' in caught.value.stderr

    async def test_result_is_frozen(self, git_ops: GitOps) -> None:
        result = await range_patch_ids(git_ops, target='main', tip='main')

        pytest.raises(dataclasses.FrozenInstanceError, setattr, result, 'patch_ids', ('x',))


@pytest.mark.asyncio
class TestIsAncestorOrRaise:
    async def test_parent_is_an_ancestor(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        advance_main(repo, 'child')

        assert await is_ancestor_or_raise(git_ops, ancestor='main~1', descendant='main') is True

    async def test_branch_commit_is_not_an_ancestor(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        tip = branch_with_commit(repo, 'task/aside')

        assert await is_ancestor_or_raise(git_ops, ancestor=tip, descendant='main') is False

    async def test_unreadable_sha_raises_with_stderr(self, git_ops: GitOps) -> None:
        with pytest.raises(GitReadFailed) as caught:
            await is_ancestor_or_raise(git_ops, ancestor=MISSING_SHA, descendant='main')

        assert 'fatal' in caught.value.stderr
