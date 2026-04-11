"""Tests for git operations — worktree lifecycle."""

import asyncio
import logging
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import (
    GitOps,
    WorktreeInfo,
    _run,
    _scrub_task_dir_from_tree,
    ScrubResult,
)


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


async def _inject_uu_state(cwd: Path, path: str, tag: str = '') -> None:
    """Inject unmerged (stage 1/2/3) index entries for *path* via index surgery.

    Uses ``git hash-object -w --stdin`` to write three blob objects and
    ``git update-index --index-info`` to register them at stages 1, 2, 3.
    The resulting UU entry is detectable by ``_detect_unmerged_paths`` without
    creating an actual conflicting merge commit or setting MERGE_HEAD.

    *tag* is interpolated into the blob content so that multiple calls in the
    same repository produce distinct shas even for different paths.
    """
    def _run_sync(cmd, **kwargs):
        return subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, check=True, **kwargs,
        )

    h1 = _run_sync(
        ['git', 'hash-object', '-w', '--stdin'],
        input=f'version base{tag}\n'.encode(),
    ).stdout.decode().strip()
    h2 = _run_sync(
        ['git', 'hash-object', '-w', '--stdin'],
        input=f'version ours{tag}\n'.encode(),
    ).stdout.decode().strip()
    h3 = _run_sync(
        ['git', 'hash-object', '-w', '--stdin'],
        input=f'version theirs{tag}\n'.encode(),
    ).stdout.decode().strip()

    index_info = (
        f'100644 {h1} 1\t{path}\n'
        f'100644 {h2} 2\t{path}\n'
        f'100644 {h3} 3\t{path}\n'
    )
    _run_sync(['git', 'update-index', '--index-info'], input=index_info.encode())


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
        worktree_info = await git_ops.create_worktree('feature-1')
        assert worktree_info.path.exists()
        assert (worktree_info.path / 'README.md').exists()
        assert len(worktree_info.base_commit) == 40

    async def test_create_worktree_returns_worktree_info(self, git_ops: GitOps):
        """create_worktree returns WorktreeInfo with path and base_commit."""
        result = await git_ops.create_worktree('feature-wi')
        assert isinstance(result, WorktreeInfo)
        assert isinstance(result.path, Path)
        assert result.path.exists()
        assert (result.path / 'README.md').exists()
        assert len(result.base_commit) == 40
        # Assert base_commit matches main's HEAD at creation time
        _, main_sha, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root
        )
        assert result.base_commit == main_sha.strip()

    async def test_create_worktree_replaces_stale_directory(self, git_ops: GitOps):
        """A directory that exists but is NOT a registered git worktree must be
        replaced with a real worktree — not silently reused (regression test for
        esc-158-11: stale .task/ dirs mistaken for valid worktrees)."""
        worktree_path = git_ops.worktree_base / 'stale-1'
        worktree_path.mkdir(parents=True, exist_ok=True)
        # Simulate a stale .task/ directory left from a previous run
        task_dir = worktree_path / '.task'
        task_dir.mkdir()
        (task_dir / 'state.json').write_text('{}')

        # The directory exists but is NOT a registered git worktree
        assert worktree_path.exists()

        worktree_info = await git_ops.create_worktree('stale-1')
        # Should have created a real worktree with repo content
        assert worktree_info.path.exists()
        assert (worktree_info.path / 'README.md').exists()
        assert len(worktree_info.base_commit) == 40

    async def test_create_worktree_reuses_registered_worktree(self, git_ops: GitOps):
        """A directory that IS a registered git worktree should be reused."""
        worktree_info = await git_ops.create_worktree('reuse-1')
        assert worktree_info.path.exists()
        assert (worktree_info.path / 'README.md').exists()

        # Call again — should reuse (not fail or recreate)
        worktree_info2 = await git_ops.create_worktree('reuse-1')
        assert worktree_info2.path == worktree_info.path
        assert (worktree_info2.path / 'README.md').exists()

    async def test_commit_in_worktree(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-2')
        (worktree_info.path / 'new_file.py').write_text('print("hello")\n')
        sha = await git_ops.commit(worktree_info.path, 'Add new file')
        assert sha is not None
        assert len(sha) == 40

    async def test_commit_nothing(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-3')
        sha = await git_ops.commit(worktree_info.path, 'Nothing')
        assert sha is None

    async def test_diff_from_main(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-4')
        (worktree_info.path / 'change.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add change')
        diff = await git_ops.get_diff_from_main(worktree_info.path)
        assert 'change.py' in diff
        assert 'x = 1' in diff

    async def test_diff_from_base(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-4b')
        (worktree_info.path / 'base_change.py').write_text('y = 2\n')
        await git_ops.commit(worktree_info.path, 'Add base change')
        diff = await git_ops.get_diff_from_base(worktree_info.path, worktree_info.base_commit)
        assert 'base_change.py' in diff
        assert 'y = 2' in diff

    async def test_commit_excludes_taskmaster_tasks(self, git_ops: GitOps):
        """Files in .taskmaster/tasks/ must not be staged by commit()."""
        worktree_info = await git_ops.create_worktree('feature-exclude')

        tasks_dir = worktree_info.path / '.taskmaster' / 'tasks'
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / 'tasks.json').write_text('{"tasks": []}')
        (worktree_info.path / 'real_change.py').write_text('x = 1\n')

        sha = await git_ops.commit(worktree_info.path, 'Test exclusion')
        assert sha is not None

        rc, files, _ = await _run(
            ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', sha],
            cwd=worktree_info.path,
        )
        assert 'real_change.py' in files
        assert '.taskmaster/tasks/tasks.json' not in files

    async def test_cleanup_worktree(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-5')
        assert worktree_info.path.exists()
        await git_ops.cleanup_worktree(worktree_info.path, 'feature-5')
        assert not worktree_info.path.exists()

    async def test_merge_to_main(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-6')
        (worktree_info.path / 'merged.py').write_text('merged = True\n')
        await git_ops.commit(worktree_info.path, 'Add merged file')

        result = await git_ops.merge_to_main(worktree_info.path, 'feature-6')
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
        worktree_info = await git_ops.create_worktree('orphan')
        (worktree_info.path / 'orphan.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Orphan commit')

        # Advance main to a different commit first
        worktree_info2 = await git_ops.create_worktree('advance-first')
        (worktree_info2.path / 'first.py').write_text('y = 1\n')
        await git_ops.commit(worktree_info2.path, 'First commit')
        result = await git_ops.merge_to_main(worktree_info2.path, 'advance-first')
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None
        await git_ops.advance_main(result.merge_commit)
        await git_ops.cleanup_merge_worktree(result.merge_worktree)

        # Now the orphan branch's commit is NOT a descendant of new main
        _, orphan_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree_info.path,
        )
        assert await git_ops.advance_main(orphan_sha) == 'not_descendant'

    async def test_get_current_branch(self, git_ops: GitOps):
        worktree_info = await git_ops.create_worktree('feature-7')
        branch = await git_ops.get_current_branch(worktree_info.path)
        assert branch == 'task/feature-7'

    async def test_merge_to_main_cleans_worktree_on_cancellation(
        self, git_ops: GitOps,
    ):
        """merge_to_main must clean up the merge worktree on CancelledError.

        Covers review issue [resource_leak_on_cancellation] at git_ops.py:495.
        The cleanup guard uses ``except Exception:`` which does NOT catch
        ``asyncio.CancelledError`` (a BaseException subclass).  This test
        fails with the old guard and passes with ``except BaseException:``.
        """
        # Set up a feature branch with a committed file.
        worktree_info = await git_ops.create_worktree('feature-cancel')
        (worktree_info.path / 'cancel_test.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add cancel test file')

        # Patch _scrub_task_dir_from_tree to raise CancelledError, simulating
        # task cancellation at the point where the merge commit already exists
        # but cleanup has not yet been called.
        with patch(
            'orchestrator.git_ops._scrub_task_dir_from_tree',
            side_effect=asyncio.CancelledError,
        ), pytest.raises(asyncio.CancelledError):
            await git_ops.merge_to_main(worktree_info.path, 'feature-cancel')

        # After CancelledError, no _merge-* worktrees should be registered.
        _, worktree_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=git_ops.project_root,
        )
        leak_lines = [
            line for line in worktree_list.splitlines()
            if '_merge-' in line
        ]
        assert not leak_lines, (
            f'Leaked merge worktrees still registered: {leak_lines}'
        )

        # Also confirm no _merge-* directories exist on disk.
        worktree_base = git_ops.worktree_base
        if worktree_base.exists():
            leak_dirs = list(worktree_base.glob('_merge-*'))
            assert not leak_dirs, (
                f'Leaked merge worktree directories on disk: {leak_dirs}'
            )


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
        wt_a_info = await git_ops.create_worktree('branch-a')
        wt_b_info = await git_ops.create_worktree('branch-b')

        # Both modify same file differently
        (wt_a_info.path / 'shared.py').write_text('value = "A"\n')
        await git_ops.commit(wt_a_info.path, 'Branch A change')

        (wt_b_info.path / 'shared.py').write_text('value = "B"\n')
        await git_ops.commit(wt_b_info.path, 'Branch B change')

        # Merge A first — should succeed
        result_a = await git_ops.merge_to_main(wt_a_info.path, 'branch-a')
        assert result_a.success
        assert result_a.merge_commit is not None
        assert result_a.merge_worktree is not None
        await git_ops.advance_main(result_a.merge_commit)
        await git_ops.cleanup_merge_worktree(result_a.merge_worktree)

        # Merge B — should conflict (main now has "A", branch has "B")
        result_b = await git_ops.merge_to_main(wt_b_info.path, 'branch-b')
        assert not result_b.success
        assert result_b.conflicts
        assert result_b.merge_worktree is not None


@pytest.mark.asyncio
class TestHasUncommittedWork:
    async def test_clean_worktree_returns_false(self, git_ops: GitOps):
        wt_info = await git_ops.create_worktree('clean-wt')
        assert not await git_ops.has_uncommitted_work(wt_info.path)

    async def test_untracked_file_returns_true(self, git_ops: GitOps):
        wt_info = await git_ops.create_worktree('untracked-wt')
        (wt_info.path / 'new_file.py').write_text('x = 1\n')
        assert await git_ops.has_uncommitted_work(wt_info.path)

    async def test_modified_tracked_file_returns_true(self, git_ops: GitOps):
        wt_info = await git_ops.create_worktree('modified-wt')
        (wt_info.path / 'README.md').write_text('# Changed\n')
        assert await git_ops.has_uncommitted_work(wt_info.path)

    async def test_file_only_in_task_dir_returns_false(self, git_ops: GitOps):
        wt_info = await git_ops.create_worktree('taskdir-wt')
        task_dir = wt_info.path / '.task'
        task_dir.mkdir(exist_ok=True)
        (task_dir / 'plan.json').write_text('{}')
        assert not await git_ops.has_uncommitted_work(wt_info.path)


@pytest.mark.asyncio
class TestWorkingTreeSync:
    """Tests for the stash/read-tree/pop working-tree protection in advance_main."""

    async def _merge_and_advance(self, git_ops: GitOps, branch: str, filename: str, content: str):
        """Helper: create a file on a branch, merge it, advance main."""
        worktree_info = await git_ops.create_worktree(branch)
        (worktree_info.path / filename).write_text(content)
        await git_ops.commit(worktree_info.path, f'Add {filename}')
        result = await git_ops.merge_to_main(worktree_info.path, branch)
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
        """Uncommitted tracked changes survive the merge advance."""
        # Modify a TRACKED file in project_root (uncommitted) — triggers stash/pop
        (git_ops.project_root / 'README.md').write_text('# work in progress\n')

        result = await self._merge_and_advance(git_ops, 'stash-restore', 'merged.py', 'merged = True\n')
        assert result == 'advanced'

        # Merged file should be in working tree
        assert (git_ops.project_root / 'merged.py').exists()
        # User's dirty tracked change should survive (stash/pop restored it)
        assert '# work in progress' in (git_ops.project_root / 'README.md').read_text()

    async def test_wip_overlap_blocks_advance(self, git_ops: GitOps):
        """Dirty file overlapping merge diff returns 'wip_overlap' without moving ref."""
        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )

        # Create dirty change to README.md in project_root (don't commit)
        (git_ops.project_root / 'README.md').write_text('# Local WIP edit\n')

        # Merge a conflicting change to README.md
        worktree_info = await git_ops.create_worktree('overlap-readme')
        (worktree_info.path / 'README.md').write_text('# Merged from branch\n')
        await git_ops.commit(worktree_info.path, 'Change README on branch')
        merge_result = await git_ops.merge_to_main(worktree_info.path, 'overlap-readme')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'wip_overlap'

        # Main ref should NOT have moved
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_before.strip() == main_after.strip()

        # Working tree should be unchanged (no stash happened)
        assert (git_ops.project_root / 'README.md').read_text() == '# Local WIP edit\n'

        # Overlap files should be recorded
        assert hasattr(git_ops, '_last_overlap_files')
        assert 'README.md' in git_ops._last_overlap_files

    async def test_wip_overlap_with_staged_file(self, git_ops: GitOps):
        """Staged change overlapping merge diff returns 'wip_overlap'."""
        # Stage a change to README.md
        (git_ops.project_root / 'README.md').write_text('# Local WIP edit\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)

        # Merge a branch that also modifies README.md
        worktree_info = await git_ops.create_worktree('pop-recovery')
        (worktree_info.path / 'new_file.py').write_text('x = 1\n')
        (worktree_info.path / 'README.md').write_text('# Merged from branch\n')
        await git_ops.commit(worktree_info.path, 'Change files on branch')
        merge_result = await git_ops.merge_to_main(worktree_info.path, 'pop-recovery')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        # Overlap detected before stash — staged README.md overlaps merge diff
        assert result == 'wip_overlap'

    async def test_pop_conflict_recovery_via_mock(self, git_ops: GitOps):
        """When stash pop fails, advance_main creates recovery branch and returns 'pop_conflict'."""
        # Create a merge commit
        worktree_info = await git_ops.create_worktree('pop-mock')
        (worktree_info.path / 'pop_file.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add pop file')
        merge_result = await git_ops.merge_to_main(worktree_info.path, 'pop-mock')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Modify a tracked file (not overlapping merge diff) to trigger stash
        (git_ops.project_root / 'README.md').write_text('# WIP edit\n')

        # Mock _run: stash push succeeds, stash pop fails (simulating conflict)
        original_run = _run

        async def mock_run(cmd, cwd=None):
            if cmd[:3] == ['git', 'stash', 'pop']:
                return (1, '', 'CONFLICT: merge conflict in README.md')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await git_ops.advance_main(merge_result.merge_commit)

        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'pop_conflict'

        # Recovery branch should be recorded
        assert hasattr(git_ops, '_last_recovery_branch')
        recovery = git_ops._last_recovery_branch
        assert recovery is not None
        assert recovery.startswith('wip/recovery-')

    async def test_consecutive_advance_after_pop_conflict(self, git_ops: GitOps):
        """After pop_conflict recovery, a subsequent advance_main succeeds normally."""
        # Merge first file
        wt1_info = await git_ops.create_worktree('consec-1')
        (wt1_info.path / 'first.py').write_text('first = True\n')
        await git_ops.commit(wt1_info.path, 'Add first')
        merge1 = await git_ops.merge_to_main(wt1_info.path, 'consec-1')
        assert merge1.success
        assert merge1.merge_commit is not None
        assert merge1.merge_worktree is not None

        # Modify a tracked file so stash is triggered, mock stash pop failure
        (git_ops.project_root / 'README.md').write_text('# WIP edit\n')
        original_run = _run

        async def mock_run(cmd, cwd=None):
            if cmd[:3] == ['git', 'stash', 'pop']:
                return (1, '', 'CONFLICT: merge conflict')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result1 = await git_ops.advance_main(merge1.merge_commit)
        await git_ops.cleanup_merge_worktree(merge1.merge_worktree)
        assert result1 == 'pop_conflict'

        # Working tree should be clean after recovery (read-tree reset)
        _, unstaged, _ = await _run(
            ['git', 'diff', '--name-only'], cwd=git_ops.project_root,
        )
        _, staged, _ = await _run(
            ['git', 'diff', '--name-only', '--cached'], cwd=git_ops.project_root,
        )
        assert not unstaged.strip(), f'Unstaged changes after recovery: {unstaged}'
        assert not staged.strip(), f'Staged changes after recovery: {staged}'

        # Second merge should succeed normally (no stash needed, tree is clean)
        wt2_info = await git_ops.create_worktree('consec-2')
        (wt2_info.path / 'second.py').write_text('second = True\n')
        await git_ops.commit(wt2_info.path, 'Add second')
        merge2 = await git_ops.merge_to_main(wt2_info.path, 'consec-2')
        assert merge2.success
        assert merge2.merge_commit is not None
        assert merge2.merge_worktree is not None

        result2 = await git_ops.advance_main(merge2.merge_commit)
        await git_ops.cleanup_merge_worktree(merge2.merge_worktree)
        assert result2 == 'advanced'

        # Both files should be on main
        _, content, _ = await _run(
            ['git', 'show', 'main:first.py'], cwd=git_ops.project_root,
        )
        assert 'first = True' in content
        _, content2, _ = await _run(
            ['git', 'show', 'main:second.py'], cwd=git_ops.project_root,
        )
        assert 'second = True' in content2

    async def test_wip_overlap_disjoint_files_proceeds(self, git_ops: GitOps):
        """Dirty file NOT overlapping merge diff proceeds normally."""
        # Create dirty file in a different path than what will be merged
        (git_ops.project_root / 'wip_unrelated.py').write_text('wip = True\n')

        # Merge a different file
        worktree_info = await git_ops.create_worktree('disjoint')
        (worktree_info.path / 'merged_file.py').write_text('merged = True\n')
        await git_ops.commit(worktree_info.path, 'Add merged file')
        merge_result = await git_ops.merge_to_main(worktree_info.path, 'disjoint')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'advanced'

        # Merged file appears on main
        _, content, _ = await _run(
            ['git', 'show', 'main:merged_file.py'], cwd=git_ops.project_root,
        )
        assert 'merged = True' in content

        # User's dirty file should survive
        assert (git_ops.project_root / 'wip_unrelated.py').exists()
        assert 'wip = True' in (git_ops.project_root / 'wip_unrelated.py').read_text()

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
        worktree_info = await git_ops.create_worktree('not-on-main')
        (worktree_info.path / 'should_not_appear.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add file')
        merge_result = await git_ops.merge_to_main(worktree_info.path, 'not-on-main')
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
        worktree_info = await git_ops.create_worktree('stash-fail')
        (worktree_info.path / 'stash_fail.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add file')
        merge_result = await git_ops.merge_to_main(worktree_info.path, 'stash-fail')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Make working tree dirty with a TRACKED file (untracked files no longer
        # trigger stash — only tracked modifications do)
        (git_ops.project_root / 'README.md').write_text('# dirty tracked edit\n')

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
        worktree_info = await git_ops.create_worktree('cas-stash')
        worktree = worktree_info.path
        (worktree / 'cas_file.py').write_text('cas = True\n')
        await git_ops.commit(worktree, 'Add file')
        merge_result = await git_ops.merge_to_main(worktree, 'cas-stash')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Make working tree dirty with a TRACKED file modification
        (git_ops.project_root / 'README.md').write_text('# WIP edit\n')

        # Force CAS failure by passing a wrong expected_main
        result = await git_ops.advance_main(
            merge_result.merge_commit,
            merge_result.merge_worktree,
            branch='cas-stash',
            expected_main='0000000000000000000000000000000000000000',
        )
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'cas_failed'

        # Dirty tracked file should be restored (stash popped)
        assert '# WIP edit' in (git_ops.project_root / 'README.md').read_text()

        # No leftover stash entries
        _, stash_list, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_list.strip() == ''

    async def test_sync_path_pop_conflict_still_uses_safe_helper(
        self, git_ops: GitOps,
    ):
        """After refactor, sync-path pop still returns 'pop_conflict' and leaves tree clean.

        This is a regression guard: the result code must remain 'pop_conflict'
        (merge DID advance), and _detect_unmerged_paths must return [] after
        recovery (proving the helper cleaned the tree).
        """
        # Create a merge commit
        wt = await git_ops.create_worktree('sync-safe-helper')
        (wt.path / 'sync_file.py').write_text('x = 1\n')
        await git_ops.commit(wt.path, 'Add sync_file')
        merge_result = await git_ops.merge_to_main(wt.path, 'sync-safe-helper')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Dirty tracked file so stash is created
        (git_ops.project_root / 'README.md').write_text('# WIP sync guard\n')

        # Mock: stash pop returns failure (simulates sync-path pop conflict)
        original_run = _run

        async def mock_run(cmd, cwd=None):
            if cmd[:3] == ['git', 'stash', 'pop']:
                return (1, '', 'CONFLICT: merge conflict in README.md')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        # Sync path: merge DID advance, WIP conflicted — must still be 'pop_conflict'
        assert result == 'pop_conflict'

        # Tree must be fully clean (recovery helper ran read-tree reset)
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert unmerged == [], f'Expected clean tree after recovery, got: {unmerged}'

    async def test_cas_failure_pop_conflict_returns_pop_conflict_no_advance(
        self, git_ops: GitOps,
    ):
        """When CAS fails AND stash pop conflicts, advance_main returns 'pop_conflict_no_advance'.

        Main ref must not move, _last_recovery_branch must be set, and no
        unmerged paths may remain in project_root after the call.
        """
        # Create a merge commit (adds cas_pop.py — no overlap with README.md)
        wt = await git_ops.create_worktree('cas-pop-conflict')
        (wt.path / 'cas_pop.py').write_text('x = 1\n')
        await git_ops.commit(wt.path, 'Add cas_pop')
        merge_result = await git_ops.merge_to_main(wt.path, 'cas-pop-conflict')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Dirty tracked file (no overlap with merge diff) so stash is created
        (git_ops.project_root / 'README.md').write_text('# WIP for CAS pop conflict\n')

        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )

        # Mock: stash pop returns failure (simulates pop conflict after CAS failure)
        original_run = _run

        async def mock_run(cmd, cwd=None):
            if cmd[:3] == ['git', 'stash', 'pop']:
                return (1, '', 'CONFLICT: merge conflict in README.md')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await git_ops.advance_main(
                merge_result.merge_commit,
                merge_result.merge_worktree,
                branch='cas-pop-conflict',
                expected_main='0' * 40,  # force CAS failure
            )
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        assert result == 'pop_conflict_no_advance'

        # Recovery branch must have been created and recorded
        assert hasattr(git_ops, '_last_recovery_branch')
        recovery = git_ops._last_recovery_branch
        assert recovery is not None and recovery.startswith('wip/recovery-')

        # Main ref must NOT have moved
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_before.strip() == main_after.strip()

        # No leftover unmerged paths
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert unmerged == []

    async def test_advance_main_halts_on_preexisting_unmerged_state(
        self, git_ops: GitOps,
    ):
        """advance_main returns 'unmerged_state' immediately when project_root has UU markers.

        No stash must be created and main ref must not advance.
        """
        # Step 1: prepare a valid merge commit via a clean worktree
        wt = await git_ops.create_worktree('uu-guard-advance')
        (wt.path / 'new_feature.py').write_text('feature = True\n')
        await git_ops.commit(wt.path, 'Add new_feature')
        merge_result = await git_ops.merge_to_main(wt.path, 'uu-guard-advance')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Record state before injecting UU markers
        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        _, stash_before, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )

        # Step 2: inject unmerged (stage 1/2/3) entries into the index without
        # doing an actual merge commit or setting MERGE_HEAD.
        await _inject_uu_state(git_ops.project_root, 'uu_conflict_test.py')

        # Verify the UU state is detectable
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert len(unmerged) >= 1, f'Expected unmerged paths after index surgery, got: {unmerged}'

        # Step 3: advance_main must detect UU state and halt without touching main
        result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        assert result == 'unmerged_state'

        # Main ref must NOT have moved
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_before.strip() == main_after.strip()

        # No stash was created during the halted advance attempt
        _, stash_after, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_before.strip() == stash_after.strip()

    async def test_advance_main_halts_on_preexisting_unmerged_state_speculative_shape(
        self, git_ops: GitOps,
    ):
        """advance_main with full speculative-worker call shape returns 'unmerged_state' without stash.

        Uses the full merge_queue.py call shape (branch, expected_main) and dirties
        README.md so the working-tree protection block is armed.  Without the
        unmerged-state guard, advance_main would reach the stash block and return
        'stash_failed' (git stash push fails on a UU index with 'you have unmerged
        paths') -- the guard must fire first so we see 'unmerged_state' instead,
        and no stash entry is ever attempted.

        Mirrors the real caller in merge_queue.py:265-270:
            result = await self._git_ops.advance_main(
                merge_result.merge_commit, merge_wt,
                branch=req.branch, max_attempts=..., expected_main=main_sha,
            )
        """
        # Step 1: prepare a valid merge commit via a clean worktree
        wt = await git_ops.create_worktree('uu-guard-spec')
        (wt.path / 'new_spec_feature.py').write_text('spec_feature = True\n')
        await git_ops.commit(wt.path, 'Add new_spec_feature')
        merge_result = await git_ops.merge_to_main(wt.path, 'uu-guard-spec')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Record state before injecting UU markers and dirtying the tree
        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        _, stash_before, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )

        # Step 2: dirty a tracked file so the working-tree protection block
        # would attempt a stash if the unmerged guard were not present.
        # Uses a DIFFERENT path (README.md) from the UU injection below so
        # 'dirty file' and 'unmerged index entry' are independent states.
        (git_ops.project_root / 'README.md').write_text('# WIP speculative guard test\n')

        # Step 3: inject unmerged (stage 1/2/3) entries on 'uu_conflict_spec.py'
        # via index surgery -- no real conflicting merge needed.
        await _inject_uu_state(git_ops.project_root, 'uu_conflict_spec.py', tag=' spec')

        # Precondition: confirm injected UU state is detectable
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert len(unmerged) >= 1, f'Expected unmerged paths after index surgery, got: {unmerged}'

        # Step 4: call advance_main with the full speculative-worker call shape.
        # expected_main is the REAL current main SHA -- without the guard the CAS
        # would succeed and the ref would advance.  A passing test therefore certifies
        # the guard fires BEFORE the entire happy path, not just before CAS.
        #
        # Orthogonal probe: record every _run invocation during advance_main to
        # assert that git stash push was never attempted (decisive narrowing that
        # the unmerged-state guard short-circuited the working-tree protection block).
        original_run = _run
        recorded: list[list[str]] = []

        async def recording_run(cmd, cwd=None, **kwargs):
            recorded.append(list(cmd))
            return await original_run(cmd, cwd=cwd, **kwargs)

        with patch('orchestrator.git_ops._run', side_effect=recording_run):
            result = await git_ops.advance_main(
                merge_result.merge_commit,
                merge_result.merge_worktree,
                branch='uu-guard-spec',
                expected_main=main_before.strip(),
            )
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        assert result == 'unmerged_state'

        # Decisive narrowing: the guard returned before the working-tree protection
        # block had any chance to invoke git stash push.
        assert not any(c[:3] == ['git', 'stash', 'push'] for c in recorded), (
            f'guard should fire before stash push; recorded stash cmds: '
            f'{[c for c in recorded if c[:2] == ["git", "stash"]]}'
        )

        # Positive-path: confirm the unmerged-state guard was actually entered.
        # _detect_unmerged_paths calls ['git', 'status', '--porcelain']; this
        # command does not appear in the pre-guard path (ls-tree / merge-base),
        # so its presence uniquely certifies that the guard ran rather than
        # short-circuiting for an unrelated reason.
        assert any(c[:2] == ['git', 'status'] and '--porcelain' in c for c in recorded), (
            f'expected _detect_unmerged_paths to invoke git status --porcelain '
            f'(guard path marker); recorded commands: {recorded}'
        )

        # Main ref must NOT have moved
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_before.strip() == main_after.strip()

        # Corroborating: stash list is unchanged (the guard returned before the stash
        # block, and even if the stash block had been entered it would have failed to
        # create an entry -- the decisive narrowing is the recording-_run probe above).
        _, stash_after, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_before.strip() == stash_after.strip()

    async def test_cas_failure_pop_conflict_does_not_cascade_to_stash_failed(
        self, git_ops: GitOps,
    ):
        """Full cascade regression guard: after pop_conflict_no_advance the tree is clean.

        Simulates the exact cascade the bug report describes:
        1. CAS failure → stash pop conflicts → pop_conflict_no_advance returned.
        2. Second advance_main call (no mocks, no dirty WIP) must NOT return
           'stash_failed' or 'unmerged_state' — it must succeed normally.
        This proves _safe_stash_pop_with_recovery fully cleans the tree.
        """
        # Setup: create a merge commit
        wt = await git_ops.create_worktree('cascade-regr')
        (wt.path / 'cascade.py').write_text('x = 1\n')
        await git_ops.commit(wt.path, 'Add cascade file')
        merge_result = await git_ops.merge_to_main(wt.path, 'cascade-regr')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        # Dirty a tracked file so advance_main creates a stash
        (git_ops.project_root / 'README.md').write_text('# WIP cascade regression\n')

        # First call: force CAS failure AND mock stash pop to conflict
        original_run = _run

        async def mock_run_conflict(cmd, cwd=None):
            if cmd[:3] == ['git', 'stash', 'pop']:
                return (1, '', 'CONFLICT: merge conflict in README.md')
            return await original_run(cmd, cwd=cwd)

        with patch('orchestrator.git_ops._run', side_effect=mock_run_conflict):
            result1 = await git_ops.advance_main(
                merge_result.merge_commit,
                merge_result.merge_worktree,
                branch='cascade-regr',
                expected_main='0' * 40,  # force CAS failure
            )
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        assert result1 == 'pop_conflict_no_advance', f'Expected pop_conflict_no_advance, got {result1}'

        # Tree must be fully clean now (recovery helper ran read-tree reset)
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert unmerged == [], f'Tree not clean after first advance: {unmerged}'

        # Second call: create a fresh merge commit, no patching, no dirty WIP
        wt2 = await git_ops.create_worktree('cascade-regr-2')
        (wt2.path / 'cascade2.py').write_text('y = 2\n')
        await git_ops.commit(wt2.path, 'Add cascade2')
        merge_result2 = await git_ops.merge_to_main(wt2.path, 'cascade-regr-2')
        assert merge_result2.success

        assert merge_result2.merge_commit is not None
        result2 = await git_ops.advance_main(merge_result2.merge_commit)
        if merge_result2.merge_worktree:
            await git_ops.cleanup_merge_worktree(merge_result2.merge_worktree)

        # Must NOT cascade to stash_failed or unmerged_state
        assert result2 not in ('stash_failed', 'unmerged_state'), (
            f'Cascade failure: second advance returned {result2!r} '
            f'(tree was not cleaned by recovery helper)'
        )
        assert result2 == 'advanced', (
            f'Unexpected result: {result2!r} (fully controlled path: no CAS injection, '
            f'no mocks, no dirty WIP \u2014 cas_failed indicates an environmental regression)'
        )


@pytest.mark.asyncio
class TestUnmergedDetection:
    """Tests for the _detect_unmerged_paths helper."""

    async def test_detect_unmerged_paths_empty_on_clean_tree(self, git_ops: GitOps):
        """On a freshly-initialized repo with no conflicts, helper returns []."""
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert unmerged == []

    async def test_detect_unmerged_paths_returns_uu_files(self, git_ops: GitOps):
        """After a conflicting merge, helper returns paths containing the conflicted file."""
        # Create a divergent branch with a conflicting change to README.md
        await _run(
            ['git', 'checkout', '-b', 'conflict-b'],
            cwd=git_ops.project_root,
        )
        (git_ops.project_root / 'README.md').write_text('# From B\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'change README on B'],
            cwd=git_ops.project_root,
        )

        # Go back to main and make a divergent change
        await _run(['git', 'checkout', 'main'], cwd=git_ops.project_root)
        (git_ops.project_root / 'README.md').write_text('# From Main\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'change README on main'],
            cwd=git_ops.project_root,
        )

        # Trigger a conflicting merge — leaves UU markers in index/worktree
        rc, _, _ = await _run(
            ['git', 'merge', 'conflict-b'],
            cwd=git_ops.project_root,
        )
        assert rc != 0  # Must have conflicted

        # Now test the helper
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert 'README.md' in unmerged
        assert len(unmerged) >= 1

    async def test_inject_uu_state_helper_creates_unmerged_entries(
        self, git_ops: GitOps,
    ):
        """_inject_uu_state creates detectable UU index entries for the given path."""
        await _inject_uu_state(git_ops.project_root, 'helper_probe.py')
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert 'helper_probe.py' in unmerged, (
            f'Expected helper_probe.py in unmerged paths, got: {unmerged}'
        )

    async def test_inject_uu_state_raises_on_non_git_cwd(
        self, tmp_path: Path,
    ):
        """_inject_uu_state raises CalledProcessError when cwd is not a git repo.

        git hash-object exits with rc != 0 in a non-git directory; without
        check=True the helper silently builds an invalid payload.  With
        check=True it raises immediately, turning silent corruption into an
        actionable CalledProcessError that includes stderr.
        """
        with pytest.raises(subprocess.CalledProcessError):
            await _inject_uu_state(tmp_path, 'foo.py')


@pytest.mark.asyncio
class TestSafeStashPopWithRecovery:
    """Tests for the _safe_stash_pop_with_recovery helper."""

    async def test_safe_stash_pop_success_returns_ok(self, git_ops: GitOps):
        """_safe_stash_pop_with_recovery returns (True, None) on a clean pop.

        Dirty file content is restored, stash list is empty, no recovery
        branch is created.
        """
        # Stash a dirty tracked file
        (git_ops.project_root / 'README.md').write_text('# WIP content\n')
        await _run(
            ['git', 'stash', 'push', '-m', 'test stash'], cwd=git_ops.project_root,
        )

        # Verify stash was created
        _, stash_list, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_list.strip(), 'Stash should have an entry before pop'

        # Call the helper — no conflict exists, should succeed
        ok, recovery = await git_ops._safe_stash_pop_with_recovery('label-1')

        assert ok is True
        assert recovery is None

        # Dirty file content must be restored
        assert '# WIP content' in (git_ops.project_root / 'README.md').read_text()

        # No recovery branch was created
        _, branches, _ = await _run(['git', 'branch'], cwd=git_ops.project_root)
        assert 'wip/recovery' not in branches

        # Stash list is now empty
        _, stash_after, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_after.strip() == ''

    async def test_safe_stash_pop_conflict_creates_recovery_branch(
        self, git_ops: GitOps,
    ):
        """_safe_stash_pop_with_recovery returns (False, branch) and cleans up on conflict.

        The recovery branch points at the original stash tree, stash list is
        empty, and project_root has no leftover unmerged paths afterward.
        """
        # Write WIP content to README.md and stash it
        (git_ops.project_root / 'README.md').write_text('# WIP content for conflict\n')
        await _run(
            ['git', 'stash', 'push', '-m', 'wip-for-conflict'],
            cwd=git_ops.project_root,
        )

        # Capture stash tree before pop attempt (to verify recovery branch later)
        _, stash_tree, _ = await _run(
            ['git', 'rev-parse', 'stash@{0}^{tree}'], cwd=git_ops.project_root,
        )

        # Commit a DIFFERENT version of README.md on main so stash pop will conflict.
        # Three-way merge scenario:
        #   base (stash parent) : '# Test\n'
        #   ours (HEAD)         : '# Main version…\n'
        #   theirs (stash)      : '# WIP content for conflict\n'
        (git_ops.project_root / 'README.md').write_text('# Main version — conflicts with WIP\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Commit conflicting README'],
            cwd=git_ops.project_root,
        )

        # Call the helper — git stash pop will conflict (real git conflict)
        ok, recovery = await git_ops._safe_stash_pop_with_recovery('label-2')

        assert ok is False
        assert recovery is not None
        assert recovery.startswith('wip/recovery-label-2-'), (
            f'Recovery branch name should start with wip/recovery-label-2-, got {recovery!r}'
        )

        # Recovery branch tree must match the original stash tree
        _, branch_tree, _ = await _run(
            ['git', 'rev-parse', f'{recovery}^{{tree}}'], cwd=git_ops.project_root,
        )
        assert stash_tree.strip() == branch_tree.strip(), (
            'Recovery branch tree must equal original stash tree'
        )

        # Stash list must be empty (stash was dropped after branch creation)
        _, stash_after, _ = await _run(
            ['git', 'stash', 'list'], cwd=git_ops.project_root,
        )
        assert stash_after.strip() == ''

        # No unmerged paths remain in project_root
        unmerged = await git_ops._detect_unmerged_paths(git_ops.project_root)
        assert unmerged == [], f'Expected no unmerged paths after recovery, got: {unmerged}'


@pytest.mark.asyncio
class TestScrubTaskDirFromTree:
    async def test_scrub_returns_failed_when_git_rm_fails(
        self, git_ops: GitOps, caplog,
    ):
        """_scrub_task_dir_from_tree returns FAILED and skips rmtree/commit when git rm fails."""
        # Create a real worktree for a realistic working directory (no mock yet)
        worktree_info = await git_ops.create_worktree('scrub-rm-fail')

        # Create a .task/ directory with sentinel content on disk
        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        sentinel = task_dir / 'sentinel.txt'
        sentinel.write_text('keep-me\n')

        commit_calls: list = []

        async def mock_run(cmd, cwd=None):
            # (a) Fake .task/ contamination detected via ls-tree
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '.task/tracked.txt', '')
            # (b) Fail git rm --cached to simulate index corruption / permission error
            if cmd[:5] == ['git', 'rm', '-r', '--cached', '--']:
                return (1, '', 'fatal: simulated git rm failure')
            # Strict — no other git commands should be reached on the failure path
            pytest.fail(f'unexpected _run call on git-rm failure path: {cmd}')

        with (
            caplog.at_level(logging.ERROR, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', side_effect=mock_run),
        ):
            result = await _scrub_task_dir_from_tree(worktree_info.path, 'test-rm-fail')

        # Return value must be FAILED — git rm failed, scrub did not complete
        assert result == ScrubResult.FAILED, (
            f'Expected ScrubResult.FAILED on git rm failure, got {result!r}'
        )

        # An ERROR must have been logged containing the context label and the stderr
        error_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(
            'test-rm-fail' in m and 'simulated git rm failure' in m
            for m in error_msgs
        ), f'Expected ERROR log with context and stderr, got: {error_msgs}'

        # Filesystem .task/ must still exist — rmtree must have been skipped
        assert sentinel.exists(), (
            'sentinel.txt was deleted — rmtree must be skipped on git rm failure'
        )

        # No git commit should have been issued (early return before commit step)
        assert not commit_calls, (
            f'Expected no commit calls on git rm failure, got: {commit_calls}'
        )

    async def test_scrub_returns_scrubbed_on_happy_path(
        self, git_ops: GitOps, caplog,
    ):
        """_scrub_task_dir_from_tree returns SCRUBBED, runs rmtree, and commits on success."""
        worktree_info = await git_ops.create_worktree('scrub-happy')

        # Sentinel inside .task/ — must be removed by rmtree after a successful scrub
        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        sentinel = task_dir / 'sentinel.txt'
        sentinel.write_text('remove-me\n')

        commit_calls: list = []

        async def mock_run(cmd, cwd=None):
            # (a) Fake tracked .task/ file via ls-tree
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '.task/tracked.txt', '')
            # (b) git rm --cached succeeds
            if cmd[:5] == ['git', 'rm', '-r', '--cached', '--']:
                return (0, '', '')
            # (c) git commit (amend) succeeds — record and ack
            if len(cmd) >= 2 and cmd[1] == 'commit':
                commit_calls.append(list(cmd))
                return (0, '', '')
            # Strict — any other command is unexpected on the success path
            pytest.fail(f'unexpected _run call on scrub happy path: {cmd}')

        with (
            caplog.at_level(logging.INFO, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', side_effect=mock_run),
        ):
            result = await _scrub_task_dir_from_tree(worktree_info.path, 'test-happy')

        # Return value must be SCRUBBED
        assert result == ScrubResult.SCRUBBED, (
            f'Expected ScrubResult.SCRUBBED on success, got {result!r}'
        )

        # No ERROR should have been logged
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, f'Unexpected ERROR log entries: {[r.getMessage() for r in errors]}'

        # Filesystem .task/ must have been cleaned up (rmtree ran)
        assert not sentinel.exists(), (
            'sentinel.txt still exists — rmtree must run on a successful scrub'
        )

        # git commit must have been called exactly once
        assert len(commit_calls) == 1, (
            f'Expected exactly one commit call, got: {commit_calls}'
        )

    async def test_scrub_returns_failed_when_git_commit_fails(
        self, git_ops: GitOps, caplog,
    ):
        """_scrub_task_dir_from_tree returns FAILED and logs error when commit fails post-rm."""
        worktree_info = await git_ops.create_worktree('scrub-commit-fail')

        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        sentinel = task_dir / 'sentinel.txt'
        sentinel.write_text('already-removed-by-rmtree\n')

        async def mock_run(cmd, cwd=None):
            # (a) Fake tracked .task/ file via ls-tree
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '.task/tracked.txt', '')
            # (b) git rm --cached succeeds (contamination removed from index)
            if cmd[:5] == ['git', 'rm', '-r', '--cached', '--']:
                return (0, '', '')
            # (c) git commit fails (e.g. locked index, hook failure)
            if len(cmd) >= 2 and cmd[1] == 'commit':
                return (1, '', 'fatal: simulated commit failure')
            # Strict — unexpected command
            pytest.fail(f'unexpected _run call on commit-failure path: {cmd}')

        with (
            caplog.at_level(logging.ERROR, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', side_effect=mock_run),
        ):
            result = await _scrub_task_dir_from_tree(worktree_info.path, 'test-commit-fail')

        # Return value must be FAILED — commit did not succeed
        assert result == ScrubResult.FAILED, (
            f'Expected ScrubResult.FAILED on commit failure, got {result!r}'
        )

        # An ERROR must have been logged with context and the commit stderr
        error_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(
            'test-commit-fail' in m and 'simulated commit failure' in m
            for m in error_msgs
        ), f'Expected ERROR log with context and stderr, got: {error_msgs}'

        # Filesystem .task/ must be GONE — rmtree runs before the commit step
        assert not sentinel.exists(), (
            'sentinel.txt still exists — rmtree must run before git commit attempt'
        )
