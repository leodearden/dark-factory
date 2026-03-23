"""Tests for git operations — worktree lifecycle."""

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path):
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
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.mark.asyncio
class TestWorktreeLifecycle:
    async def test_create_worktree(self, git_ops: GitOps):
        worktree, base_sha = await git_ops.create_worktree('feature-1')
        assert worktree.exists()
        assert (worktree / 'README.md').exists()
        assert len(base_sha) == 40

    async def test_commit_in_worktree(self, git_ops: GitOps):
        worktree, _ = await git_ops.create_worktree('feature-2')
        (worktree / 'new_file.py').write_text('print("hello")\n')
        sha = await git_ops.commit(worktree, 'Add new file')
        assert sha is not None
        assert len(sha) == 40

    async def test_commit_nothing(self, git_ops: GitOps):
        worktree, _ = await git_ops.create_worktree('feature-3')
        sha = await git_ops.commit(worktree, 'Nothing')
        assert sha is None

    async def test_diff_from_main(self, git_ops: GitOps):
        worktree, _ = await git_ops.create_worktree('feature-4')
        (worktree / 'change.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add change')
        diff = await git_ops.get_diff_from_main(worktree)
        assert 'change.py' in diff
        assert 'x = 1' in diff

    async def test_diff_from_base(self, git_ops: GitOps):
        worktree, base_sha = await git_ops.create_worktree('feature-4b')
        (worktree / 'base_change.py').write_text('y = 2\n')
        await git_ops.commit(worktree, 'Add base change')
        diff = await git_ops.get_diff_from_base(worktree, base_sha)
        assert 'base_change.py' in diff
        assert 'y = 2' in diff

    async def test_commit_excludes_taskmaster_tasks(self, git_ops: GitOps):
        """Files in .taskmaster/tasks/ must not be staged by commit()."""
        worktree, _ = await git_ops.create_worktree('feature-exclude')

        tasks_dir = worktree / '.taskmaster' / 'tasks'
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / 'tasks.json').write_text('{"tasks": []}')
        (worktree / 'real_change.py').write_text('x = 1\n')

        sha = await git_ops.commit(worktree, 'Test exclusion')
        assert sha is not None

        rc, files, _ = await _run(
            ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', sha],
            cwd=worktree,
        )
        assert 'real_change.py' in files
        assert '.taskmaster/tasks/tasks.json' not in files

    async def test_cleanup_worktree(self, git_ops: GitOps):
        worktree, _ = await git_ops.create_worktree('feature-5')
        assert worktree.exists()
        await git_ops.cleanup_worktree(worktree, 'feature-5')
        assert not worktree.exists()

    async def test_merge_to_main(self, git_ops: GitOps):
        worktree, _ = await git_ops.create_worktree('feature-6')
        (worktree / 'merged.py').write_text('merged = True\n')
        await git_ops.commit(worktree, 'Add merged file')

        result = await git_ops.merge_to_main(worktree, 'feature-6')
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None
        assert result.merge_worktree != git_ops.project_root

        # Merge worktree has the file
        assert (result.merge_worktree / 'merged.py').exists()

        # Main ref not advanced yet — project_root working tree untouched
        assert not (git_ops.project_root / 'merged.py').exists()

        # Advance main and verify
        assert await git_ops.advance_main(result.merge_commit)
        _, content, _ = await _run(
            ['git', 'show', 'main:merged.py'], cwd=git_ops.project_root,
        )
        assert 'merged = True' in content

        await git_ops.cleanup_merge_worktree(result.merge_worktree)
        assert not result.merge_worktree.exists()

    async def test_advance_main_rejects_non_ancestor(self, git_ops: GitOps):
        """advance_main rejects a SHA that isn't a descendant of main."""
        # Use a commit from a branch that hasn't been merged
        worktree, _ = await git_ops.create_worktree('orphan')
        (worktree / 'orphan.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Orphan commit')

        # Advance main to a different commit first
        worktree2, _ = await git_ops.create_worktree('advance-first')
        (worktree2 / 'first.py').write_text('y = 1\n')
        await git_ops.commit(worktree2, 'First commit')
        result = await git_ops.merge_to_main(worktree2, 'advance-first')
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None
        await git_ops.advance_main(result.merge_commit)
        await git_ops.cleanup_merge_worktree(result.merge_worktree)

        # Now the orphan branch's commit is NOT a descendant of new main
        _, orphan_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree,
        )
        assert not await git_ops.advance_main(orphan_sha)

    async def test_get_current_branch(self, git_ops: GitOps):
        worktree, _ = await git_ops.create_worktree('feature-7')
        branch = await git_ops.get_current_branch(worktree)
        assert branch == 'task/feature-7'


@pytest.mark.asyncio
class TestMergeConflicts:
    async def test_conflict_detection(self, git_ops: GitOps):
        # Create BOTH branches before merging either (both fork from same main)
        wt_a, _ = await git_ops.create_worktree('branch-a')
        wt_b, _ = await git_ops.create_worktree('branch-b')

        # Both modify same file differently
        (wt_a / 'shared.py').write_text('value = "A"\n')
        await git_ops.commit(wt_a, 'Branch A change')

        (wt_b / 'shared.py').write_text('value = "B"\n')
        await git_ops.commit(wt_b, 'Branch B change')

        # Merge A first — should succeed
        result_a = await git_ops.merge_to_main(wt_a, 'branch-a')
        assert result_a.success
        assert result_a.merge_commit is not None
        assert result_a.merge_worktree is not None
        await git_ops.advance_main(result_a.merge_commit)
        await git_ops.cleanup_merge_worktree(result_a.merge_worktree)

        # Merge B — should conflict (main now has "A", branch has "B")
        result_b = await git_ops.merge_to_main(wt_b, 'branch-b')
        assert not result_b.success
        assert result_b.conflicts
        assert result_b.merge_worktree is not None

        # Clean up the conflict worktree
        await git_ops.cleanup_merge_worktree(result_b.merge_worktree)
