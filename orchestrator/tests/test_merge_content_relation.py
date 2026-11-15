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
from typing import Any

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.content_relation import (
    ContentRelation,
    ContentUnclassifiable,
    classify_content,
)
from orchestrator.merge_lane.landing_evidence import (
    Containment,
    GitReadFailed,
    RangePatchIds,
    classify_containment,
    is_ancestor_or_raise,
    range_patch_ids,
)
from orchestrator.merge_lane.types import MergeContent, QueuedBranch

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


def two_commit_branch(repo: GuardedRepo, branch: str) -> tuple[str, str]:
    """Create *branch* off main with two content-bearing commits; return both shas."""
    repo.git('checkout', '-b', branch, 'main')
    first = repo.commit(f'feat: {branch} first', {f'work/{branch}-1.py': numbered_lines(f'{branch}1', 8)})
    second = repo.commit(f'feat: {branch} second', {f'work/{branch}-2.py': numbered_lines(f'{branch}2', 8)})
    repo.git('checkout', 'main')
    return first, second


def with_line(lines: list[str], index: int, text: str) -> str:
    """*lines* joined as a file, with line *index* replaced by *text*."""
    edited = list(lines)
    edited[index] = text
    return ''.join(f'{line}\n' for line in edited)


def rebase_onto_main(repo: GuardedRepo, branch: str) -> str:
    """Rebase *branch* onto main, asserting no conflict; return the rewritten tip."""
    repo.git('checkout', branch)
    repo.git('rebase', 'main')
    tip = repo.sha('HEAD')
    repo.git('checkout', 'main')
    return tip


@pytest.mark.asyncio
class TestClassifyContent:
    async def test_equal_tips_are_same(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        tip = branch_with_commit(repo, 'task/same')

        relation = await classify_content(git_ops, target='main', old_tip=tip, new_tip=tip)

        assert relation is ContentRelation.SAME

    async def test_clean_rebase_is_rebased(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        _, old_tip = two_commit_branch(repo, 'task/clean')
        advance_main(repo, 'unrelated')
        new_tip = rebase_onto_main(repo, 'task/clean')

        relation = await classify_content(git_ops, target='main', old_tip=old_tip, new_tip=new_tip)

        assert relation is ContentRelation.REBASED

    async def test_conflict_resolution_is_different(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        lines = numbered_lines('shared', 12).splitlines()
        repo.commit('chore: shared file', {'shared.txt': numbered_lines('shared', 12)})
        repo.git('checkout', '-b', 'task/conflict')
        old_tip = repo.commit('feat: branch side', {'shared.txt': with_line(lines, 5, 'branch')})
        repo.git('checkout', 'main')
        repo.commit('chore: main side', {'shared.txt': with_line(lines, 5, 'main')})
        repo.git('checkout', '-b', 'task/conflict-resolved')
        new_tip = repo.commit('feat: branch side, resolved', {'shared.txt': with_line(lines, 5, 'branch')})

        relation = await classify_content(git_ops, target='main', old_tip=old_tip, new_tip=new_tip)

        assert relation is ContentRelation.DIFFERENT

    async def test_shifted_context_is_different(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        lines = numbered_lines('context', 12).splitlines()
        repo.commit('chore: context file', {'context.txt': numbered_lines('context', 12)})
        repo.git('checkout', '-b', 'task/shifted')
        old_tip = repo.commit('feat: edit line 6', {'context.txt': with_line(lines, 5, 'branch line 6')})
        repo.git('checkout', 'main')
        repo.commit('chore: edit line 3', {'context.txt': with_line(lines, 2, 'main line 3')})
        new_tip = rebase_onto_main(repo, 'task/shifted')

        relation = await classify_content(git_ops, target='main', old_tip=old_tip, new_tip=new_tip)

        assert relation is ContentRelation.DIFFERENT

    async def test_added_commit_is_different(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        old_tip = branch_with_commit(repo, 'task/added')
        repo.git('checkout', 'task/added')
        new_tip = repo.commit('feat: one more', {'work/more.py': numbered_lines('more', 6)})

        relation = await classify_content(git_ops, target='main', old_tip=old_tip, new_tip=new_tip)

        assert relation is ContentRelation.DIFFERENT

    async def test_dropped_commit_is_different(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        kept, old_tip = two_commit_branch(repo, 'task/dropped')
        advance_main(repo, 'moved')
        repo.git('checkout', '-b', 'task/dropped-new', 'main')
        repo.git('cherry-pick', kept)
        new_tip = repo.sha('HEAD')

        relation = await classify_content(git_ops, target='main', old_tip=old_tip, new_tip=new_tip)

        assert relation is ContentRelation.DIFFERENT

    async def test_reset_to_an_ancestor_is_older(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        first, old_tip = two_commit_branch(repo, 'task/reset')

        relation = await classify_content(git_ops, target='main', old_tip=old_tip, new_tip=first)

        assert relation is ContentRelation.OLDER

    async def test_merge_commit_in_range_is_different(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        old_tip = branch_with_commit(repo, 'task/merging')
        advance_main(repo, 'merged-in')
        repo.git('checkout', 'task/merging')
        repo.git('merge', '--no-ff', 'main', '-m', 'Merge main into task/merging')
        new_tip = repo.sha('HEAD')

        relation = await classify_content(git_ops, target='main', old_tip=old_tip, new_tip=new_tip)

        assert relation is ContentRelation.DIFFERENT

    async def test_merge_commit_only_in_the_old_range_is_different(
        self, repo: GuardedRepo, git_ops: GitOps,
    ) -> None:
        own = branch_with_commit(repo, 'task/synced-then-linear')
        advance_main(repo, 'synced-in')
        repo.git('checkout', 'task/synced-then-linear')
        repo.git('merge', '--no-ff', 'main', '-m', 'Merge main into task/synced-then-linear')
        old_tip = repo.sha('HEAD')
        repo.git('checkout', '-b', 'task/linear', 'main')
        repo.git('cherry-pick', own)
        new_tip = repo.sha('HEAD')
        old_range = await range_patch_ids(git_ops, target='main', tip=old_tip)
        new_range = await range_patch_ids(git_ops, target='main', tip=new_tip)
        assert old_range.patch_ids == new_range.patch_ids
        assert (old_range.has_merge_commit, new_range.has_merge_commit) == (True, False)

        relation = await classify_content(git_ops, target='main', old_tip=old_tip, new_tip=new_tip)

        assert relation is ContentRelation.DIFFERENT

    async def test_an_added_empty_commit_reads_rebased_an_accepted_limit(
        self, repo: GuardedRepo, git_ops: GitOps,
    ) -> None:
        """An empty commit has no patch id, so adding one changes no range's content."""
        old_tip = branch_with_commit(repo, 'task/empty-added')
        advance_main(repo, 'moved')
        repo.git('checkout', '-b', 'task/empty-added-new', 'main')
        repo.git('cherry-pick', old_tip)
        repo.git('commit', '--allow-empty', '-m', 'chore: empty')
        new_tip = repo.sha('HEAD')

        relation = await classify_content(git_ops, target='main', old_tip=old_tip, new_tip=new_tip)

        assert relation is ContentRelation.REBASED

    async def test_reapplied_diff_is_different(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        repo.git('checkout', '-b', 'task/multiset')
        repo.commit('feat: add f', {'work/f.py': numbered_lines('f', 10)})
        repo.git('rm', '-q', 'work/f.py')
        repo.git('commit', '-m', 'feat: drop f')
        old_tip = repo.sha('HEAD')
        new_tip = repo.commit('feat: add f again', {'work/f.py': numbered_lines('f', 10)})

        relation = await classify_content(git_ops, target='main', old_tip=old_tip, new_tip=new_tip)

        assert relation is ContentRelation.DIFFERENT

    async def test_both_ranges_empty_is_rebased(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        old_tip = branch_with_commit(repo, 'task/landed')
        advance_main(repo, 'before-merge')
        repo.git('merge', '--no-ff', 'task/landed', '-m', 'Merge task/landed into main')
        new_tip = repo.sha('main')

        relation = await classify_content(git_ops, target='main', old_tip=old_tip, new_tip=new_tip)

        assert relation is ContentRelation.REBASED

    async def test_unreadable_new_tip_is_unclassifiable(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        old_tip = branch_with_commit(repo, 'task/bad-new')

        with pytest.raises(ContentUnclassifiable) as caught:
            await classify_content(git_ops, target='main', old_tip=old_tip, new_tip=MISSING_SHA)

        assert (caught.value.old_tip, caught.value.new_tip, caught.value.target) == (
            old_tip, MISSING_SHA, 'main',
        )
        assert caught.value.stderr.strip()

    async def test_unreadable_target_is_unclassifiable(self, repo: GuardedRepo, git_ops: GitOps) -> None:
        old_tip = branch_with_commit(repo, 'task/old-side')
        new_tip = branch_with_commit(repo, 'task/new-side')

        with pytest.raises(ContentUnclassifiable) as caught:
            await classify_content(git_ops, target=MISSING_SHA, old_tip=old_tip, new_tip=new_tip)

        assert (caught.value.old_tip, caught.value.new_tip, caught.value.target) == (
            old_tip, new_tip, MISSING_SHA,
        )
        assert caught.value.stderr.strip()


def test_unclassifiable_is_a_runtime_error() -> None:
    assert issubclass(ContentUnclassifiable, RuntimeError)


def merge_content_fields(worktree: Path) -> dict[str, Any]:
    return {
        'branch': QueuedBranch.parse('10215', 'task/'),
        'tip': 'a' * 40,
        'worktree': worktree,
        'task_files': ('a.py',),
        'module_configs': (),
        'pre_rebased': False,
    }


class TestMergeContent:
    def test_content_seq_starts_at_one(self, tmp_path: Path) -> None:
        assert MergeContent(**merge_content_fields(tmp_path)).content_seq == 1

    def test_value_is_frozen(self, tmp_path: Path) -> None:
        value = MergeContent(**merge_content_fields(tmp_path))

        pytest.raises(dataclasses.FrozenInstanceError, setattr, value, 'tip', 'b' * 40)

    def test_a_transition_makes_a_new_value(self, tmp_path: Path) -> None:
        original = MergeContent(**merge_content_fields(tmp_path))

        swapped = dataclasses.replace(original, tip='b' * 40, content_seq=2)

        assert (swapped.tip, swapped.content_seq) == ('b' * 40, 2)
        assert (original.tip, original.content_seq) == ('a' * 40, 1)

    def test_empty_tip_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='tip'):
            MergeContent(**{**merge_content_fields(tmp_path), 'tip': ''})

    def test_content_seq_below_one_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r'content_seq.*\b0\b'):
            MergeContent(**merge_content_fields(tmp_path), content_seq=0)

    def test_tip_is_required(self, tmp_path: Path) -> None:
        fields = merge_content_fields(tmp_path)
        del fields['tip']

        with pytest.raises(TypeError, match='tip'):
            MergeContent(**fields)
