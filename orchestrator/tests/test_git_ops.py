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
        assert await git_ops.advance_main(result.merge_commit) == 'advanced'
        _, content, _ = await _run(
            ['git', 'show', 'main:merged.py'], cwd=git_ops.project_root,
        )
        assert 'merged = True' in content

        # File should also be in the working tree (working tree synced)
        assert (git_ops.project_root / 'merged.py').exists()

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
        assert await git_ops.advance_main(orphan_sha) == 'not_descendant'

    async def test_get_current_branch(self, git_ops: GitOps):
        worktree, _ = await git_ops.create_worktree('feature-7')
        branch = await git_ops.get_current_branch(worktree)
        assert branch == 'task/feature-7'


@pytest.mark.asyncio
class TestCommitTaskStatuses:
    async def test_commits_changed_tasks_json(self, git_ops: GitOps):
        """commit_task_statuses commits only .taskmaster/tasks/tasks.json."""
        tasks_dir = git_ops.project_root / '.taskmaster' / 'tasks'
        tasks_dir.mkdir(parents=True)
        tasks_file = tasks_dir / 'tasks.json'
        tasks_file.write_text('{"tasks": []}')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Add tasks.json'], cwd=git_ops.project_root)

        # Modify tasks.json in working tree (simulates set_task_status)
        tasks_file.write_text('{"tasks": [{"id": 1, "status": "done"}]}')

        sha = await git_ops.commit_task_statuses()
        assert sha is not None

        # Verify the commit contains only tasks.json
        rc, files, _ = await _run(
            ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', sha],
            cwd=git_ops.project_root,
        )
        assert '.taskmaster/tasks/tasks.json' in files
        assert files.strip() == '.taskmaster/tasks/tasks.json'

    async def test_noop_when_unchanged(self, git_ops: GitOps):
        """Returns None when tasks.json has no changes."""
        tasks_dir = git_ops.project_root / '.taskmaster' / 'tasks'
        tasks_dir.mkdir(parents=True)
        (tasks_dir / 'tasks.json').write_text('{"tasks": []}')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Add tasks.json'], cwd=git_ops.project_root)

        sha = await git_ops.commit_task_statuses()
        assert sha is None

    async def test_noop_when_no_tasks_file(self, git_ops: GitOps):
        """Returns None when tasks.json doesn't exist."""
        sha = await git_ops.commit_task_statuses()
        assert sha is None

    async def test_does_not_stage_other_files(self, git_ops: GitOps):
        """Other dirty files in the working tree are not committed."""
        tasks_dir = git_ops.project_root / '.taskmaster' / 'tasks'
        tasks_dir.mkdir(parents=True)
        (tasks_dir / 'tasks.json').write_text('{"tasks": []}')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Add tasks.json'], cwd=git_ops.project_root)

        # Dirty both tasks.json and another file
        (tasks_dir / 'tasks.json').write_text('{"tasks": [{"id": 1}]}')
        (git_ops.project_root / 'unrelated.py').write_text('x = 1\n')

        sha = await git_ops.commit_task_statuses()
        assert sha is not None

        # Only tasks.json in the commit
        _, files, _ = await _run(
            ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', sha],
            cwd=git_ops.project_root,
        )
        assert 'unrelated.py' not in files

        # unrelated.py is still untracked
        _, status, _ = await _run(
            ['git', 'status', '--porcelain', '--', 'unrelated.py'],
            cwd=git_ops.project_root,
        )
        assert 'unrelated.py' in status


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


@pytest.mark.asyncio
class TestHasUncommittedWork:
    async def test_clean_worktree_returns_false(self, git_ops: GitOps):
        wt, _ = await git_ops.create_worktree('clean-wt')
        assert not await git_ops.has_uncommitted_work(wt)

    async def test_untracked_file_returns_true(self, git_ops: GitOps):
        wt, _ = await git_ops.create_worktree('untracked-wt')
        (wt / 'new_file.py').write_text('x = 1\n')
        assert await git_ops.has_uncommitted_work(wt)

    async def test_modified_tracked_file_returns_true(self, git_ops: GitOps):
        wt, _ = await git_ops.create_worktree('modified-wt')
        (wt / 'README.md').write_text('# Changed\n')
        assert await git_ops.has_uncommitted_work(wt)

    async def test_file_only_in_task_dir_returns_false(self, git_ops: GitOps):
        wt, _ = await git_ops.create_worktree('taskdir-wt')
        task_dir = wt / '.task'
        task_dir.mkdir(exist_ok=True)
        (task_dir / 'plan.json').write_text('{}')
        assert not await git_ops.has_uncommitted_work(wt)


@pytest.mark.asyncio
class TestWorkingTreeSync:
    """Tests for the stash/read-tree/pop working-tree protection in advance_main."""

    async def _merge_and_advance(self, git_ops: GitOps, branch: str, filename: str, content: str):
        """Helper: create a file on a branch, merge it, advance main."""
        worktree, _ = await git_ops.create_worktree(branch)
        (worktree / filename).write_text(content)
        await git_ops.commit(worktree, f'Add {filename}')
        result = await git_ops.merge_to_main(worktree, branch)
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None
        advance = await git_ops.advance_main(result.merge_commit)
        await git_ops.cleanup_merge_worktree(result.merge_worktree)
        return advance

    async def test_advance_syncs_working_tree(self, git_ops: GitOps):
        """Merged file appears in the working tree after advance_main."""
        result = await self._merge_and_advance(git_ops, 'sync-basic', 'synced.py', 'synced = True\n')
        assert result == 'advanced'
        assert (git_ops.project_root / 'synced.py').exists()
        assert 'synced = True' in (git_ops.project_root / 'synced.py').read_text()

    async def test_advance_stashes_and_restores_dirty_work(self, git_ops: GitOps):
        """Uncommitted user work survives the merge advance."""
        # Create dirty file in project_root (uncommitted)
        (git_ops.project_root / 'wip.py').write_text('work_in_progress = True\n')

        result = await self._merge_and_advance(git_ops, 'stash-restore', 'merged.py', 'merged = True\n')
        assert result == 'advanced'

        # Merged file should be in working tree
        assert (git_ops.project_root / 'merged.py').exists()
        # User's dirty file should survive
        assert (git_ops.project_root / 'wip.py').exists()
        assert 'work_in_progress' in (git_ops.project_root / 'wip.py').read_text()

    async def test_advance_stash_pop_conflict_markers(self, git_ops: GitOps):
        """Conflicting dirty changes produce git conflict markers."""
        # Create dirty change to README.md in project_root (don't commit)
        (git_ops.project_root / 'README.md').write_text('# Local WIP edit\n')

        # Merge a conflicting change to README.md
        worktree, _ = await git_ops.create_worktree('conflict-readme')
        (worktree / 'README.md').write_text('# Merged from branch\n')
        await git_ops.commit(worktree, 'Change README on branch')
        merge_result = await git_ops.merge_to_main(worktree, 'conflict-readme')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'advanced'

        # README.md should have conflict markers
        readme_content = (git_ops.project_root / 'README.md').read_text()
        assert '<<<<<<<' in readme_content

    async def test_advance_no_stash_when_clean(self, git_ops: GitOps):
        """Clean working tree: no stash, but read-tree still syncs files."""
        result = await self._merge_and_advance(git_ops, 'clean-sync', 'clean.py', 'clean = True\n')
        assert result == 'advanced'
        assert (git_ops.project_root / 'clean.py').exists()

        # No stash entries should exist
        _, stash_list, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_list.strip() == ''

    async def test_advance_skips_sync_when_not_on_main(self, git_ops: GitOps):
        """When project_root is on another branch, working tree is untouched."""
        # Switch project_root to a different branch
        await _run(['git', 'checkout', '-b', 'other-branch'], cwd=git_ops.project_root)

        # Create a marker file to detect working tree changes
        marker_content = '# Should not change\n'
        (git_ops.project_root / 'README.md').write_text(marker_content)
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Mark README'], cwd=git_ops.project_root)

        # Merge a file to main (via worktree from main)
        # Need to create worktree from main for the merge to work
        worktree, _ = await git_ops.create_worktree('not-on-main')
        (worktree / 'should_not_appear.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add file')
        merge_result = await git_ops.merge_to_main(worktree, 'not-on-main')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'advanced'

        # File should NOT appear in working tree (we're on other-branch)
        assert not (git_ops.project_root / 'should_not_appear.py').exists()
        # README should be unchanged
        assert (git_ops.project_root / 'README.md').read_text() == marker_content

    async def test_stash_failure_returns_stash_failed(self, git_ops: GitOps):
        """If git stash push fails, advance_main returns 'stash_failed' without moving the ref."""
        # Get current main SHA before the attempt
        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )

        # Create a merge commit that could be advanced
        worktree, _ = await git_ops.create_worktree('stash-fail')
        (worktree / 'stash_fail.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add file')
        merge_result = await git_ops.merge_to_main(worktree, 'stash-fail')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Make working tree dirty
        (git_ops.project_root / 'dirty.py').write_text('dirty = True\n')

        # Sabotage git stash by making .git/refs/stash unwritable
        # Instead, use a simpler approach: lock the index
        from unittest.mock import patch

        original_run = _run

        async def mock_run(cmd, cwd=None):
            if cmd[:3] == ['git', 'stash', 'push']:
                return (1, '', 'fatal: cannot stash changes')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await git_ops.advance_main(merge_result.merge_commit)

        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'stash_failed'

        # Main ref should NOT have moved
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_before.strip() == main_after.strip()

    async def test_stash_restored_on_cas_failure(self, git_ops: GitOps):
        """On CAS failure, stash is popped to restore the original working tree."""
        # Create and merge a file
        worktree, _ = await git_ops.create_worktree('cas-stash')
        (worktree / 'cas_file.py').write_text('cas = True\n')
        await git_ops.commit(worktree, 'Add file')
        merge_result = await git_ops.merge_to_main(worktree, 'cas-stash')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Make working tree dirty
        (git_ops.project_root / 'user_wip.py').write_text('wip = True\n')

        # Force CAS failure by passing a wrong expected_main
        result = await git_ops.advance_main(
            merge_result.merge_commit,
            merge_result.merge_worktree,
            branch='cas-stash',
            expected_main='0000000000000000000000000000000000000000',
        )
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'cas_failed'

        # Dirty file should be restored (stash popped)
        assert (git_ops.project_root / 'user_wip.py').exists()
        assert 'wip = True' in (git_ops.project_root / 'user_wip.py').read_text()

        # No leftover stash entries
        _, stash_list, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_list.strip() == ''
