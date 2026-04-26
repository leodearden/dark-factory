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
    ScrubOutcome,
    ScrubResult,
    WorktreeInfo,
    _run,
    scrub_task_dir_from_tree,
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
        # Default push off in tests; TestPushMain enables it explicitly per-case.
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


async def _setup_repo_with_remote(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare origin repo and a local clone for remote-fetch tests."""
    origin = tmp_path / 'origin.git'
    origin.mkdir()
    await _run(['git', 'init', '--bare', '-b', 'main'], cwd=origin)

    # Seed origin via a temp non-bare repo
    seed = tmp_path / 'seed'
    seed.mkdir()
    await _run(['git', 'init', '-b', 'main'], cwd=seed)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=seed)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=seed)
    (seed / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=seed)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=seed)
    await _run(['git', 'remote', 'add', 'origin', str(origin)], cwd=seed)
    await _run(['git', 'push', 'origin', 'main'], cwd=seed)

    # Clone origin to local
    local = tmp_path / 'local'
    await _run(['git', 'clone', str(origin), str(local)])
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=local)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=local)

    return origin, local


@pytest.fixture
def git_repo_with_remote(tmp_path: Path) -> tuple[Path, Path]:
    """Bare origin repo and a local clone with configured user (origin_path, local_path)."""
    return asyncio.run(_setup_repo_with_remote(tmp_path))


@pytest.fixture
def git_ops_with_remote(
    git_config: GitConfig,
    git_repo_with_remote: tuple[Path, Path],
) -> tuple[GitOps, Path]:
    """GitOps against a local clone that has a configured remote (origin)."""
    origin, local = git_repo_with_remote
    return GitOps(git_config, local), origin


async def _push_n_commits_to_origin(
    origin: Path,
    n: int,
    prefix: str = 'remote',
) -> None:
    """Push n new commits to the bare origin repo via a temporary clone."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        temp = Path(td) / 'temp_push'
        await _run(['git', 'clone', str(origin), str(temp)])
        await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=temp)
        await _run(['git', 'config', 'user.name', 'Test'], cwd=temp)
        for i in range(n):
            (temp / f'{prefix}_{i}.txt').write_text(f'{prefix} content {i}\n')
            await _run(['git', 'add', '-A'], cwd=temp)
            await _run(['git', 'commit', '-m', f'{prefix} commit {i}'], cwd=temp)
        rc, _, err = await _run(['git', 'push', 'origin', 'main'], cwd=temp)
        assert rc == 0, f'push to bare origin failed: {err}'



@pytest.mark.asyncio
class TestWorktreeLifecycle:
    async def test_worktree_info_stale_commits_field(self, git_ops: GitOps):
        """WorktreeInfo.stale_commits defaults to None and can be set explicitly."""
        info_default = WorktreeInfo(path=git_ops.project_root, base_commit='a' * 40)
        assert info_default.stale_commits is None

        info_explicit = WorktreeInfo(
            path=git_ops.project_root, base_commit='a' * 40, stale_commits=5,
        )
        assert info_explicit.stale_commits == 5

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

    async def test_diff_from_base_stable_when_main_advances(self, git_ops: GitOps):
        """get_diff_from_base returns branch changes even after main advances.

        This is the key test for the fix: when main advances during task execution,
        get_diff_from_base must still return the branch's changes by using the
        pinned base_commit instead of the moving main ref.
        """
        # Create worktree and capture base_commit
        worktree_info = await git_ops.create_worktree('feature-adv')
        base_commit = worktree_info.base_commit

        # Make a commit in the branch
        (worktree_info.path / 'branch_change.py').write_text('z = 3\n')
        await git_ops.commit(worktree_info.path, 'Add branch change')

        # Advance main with a separate commit (simulating another task merging)
        (git_ops.project_root / 'main_change.py').write_text('x = 1\n')
        await _run(['git', 'add', 'main_change.py'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Advance main'], cwd=git_ops.project_root)

        # get_diff_from_base should still return branch changes
        diff = await git_ops.get_diff_from_base(worktree_info.path, base_commit)
        assert 'branch_change.py' in diff
        assert 'z = 3' in diff

        # Contrast: get_diff_from_main might return empty/different (main absorbed branch)
        # This demonstrates that base_commit is needed for stable diffs

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

    async def test_tasks_json_staged_directly_is_unstaged_by_commit(
        self, git_ops: GitOps, caplog,
    ):
        """Direct ``git add .taskmaster/tasks/tasks.json`` is unstaged by commit().

        The bulk-add pathspec excludes tasks.json, but an agent could bypass
        that by staging the file directly.  The post-staging safety net inside
        commit() must catch that and unstage it (mirror of the .task/ guard).
        """
        worktree_info = await git_ops.create_worktree('feature-direct-tasks')

        tasks_dir = worktree_info.path / '.taskmaster' / 'tasks'
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / 'tasks.json').write_text('{"agent": "directly staged"}\n')
        await _run(
            ['git', 'add', '.taskmaster/tasks/tasks.json'],
            cwd=worktree_info.path,
        )

        # Need an unrelated change so commit() still produces a sha
        (worktree_info.path / 'real_change.py').write_text('x = 1\n')

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            sha = await git_ops.commit(worktree_info.path, 'Add real change')

        assert sha is not None

        rc, files, _ = await _run(
            ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', sha],
            cwd=worktree_info.path,
        )
        assert 'real_change.py' in files
        assert '.taskmaster/tasks/tasks.json' not in files

        assert any(
            '.taskmaster/tasks/' in rec.getMessage()
            and 'CONTAMINATION' in rec.getMessage()
            for rec in caplog.records
        )

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

        # Patch scrub_task_dir_from_tree to raise CancelledError, simulating
        # task cancellation at the point where the merge commit already exists
        # but cleanup has not yet been called.
        with patch(
            'orchestrator.git_ops.scrub_task_dir_from_tree',
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
class TestFreshenMain:
    async def test_freshen_main_no_remote(self, git_ops: GitOps):
        """Without a remote, _freshen_main returns (main_branch, None)."""
        ref, stale = await git_ops._freshen_main()
        assert ref == git_ops.config.main_branch
        assert stale is None

    async def test_freshen_main_remote_ahead(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """When origin/main is 3 commits ahead, returns ('origin/main', 3)."""
        git_ops, origin = git_ops_with_remote
        await _push_n_commits_to_origin(origin, 3)
        ref, stale = await git_ops._freshen_main()
        assert ref == f'{git_ops.config.remote}/{git_ops.config.main_branch}'
        assert stale == 3

    async def test_freshen_main_already_current(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """When local main == origin/main (no new commits), returns (main_branch, 0)."""
        git_ops, _origin = git_ops_with_remote
        ref, stale = await git_ops._freshen_main()
        assert ref == git_ops.config.main_branch
        assert stale == 0

    async def test_freshen_main_diverged(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """When local and remote have diverged, returns (main_branch, N) with N behind count."""
        git_ops, origin = git_ops_with_remote
        local = git_ops.project_root
        # Add a local-only commit (not pushed to origin)
        (local / 'local_only.txt').write_text('local only\n')
        await _run(['git', 'add', '-A'], cwd=local)
        await _run(['git', 'commit', '-m', 'Local only commit'], cwd=local)
        # Add a different commit to origin (creates divergence)
        await _push_n_commits_to_origin(origin, 1, prefix='remote_div')
        ref, stale = await git_ops._freshen_main()
        # Diverged: use local ref to avoid losing advance_main commits
        assert ref == git_ops.config.main_branch
        assert stale == 1

    async def test_freshen_main_behind_rev_list_fails(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """When behind rev-list exits non-zero, _freshen_main returns (main_branch, None)."""
        git_ops, _origin = git_ops_with_remote

        async def fake_run(cmd, cwd=None):
            if 'fetch' in cmd:
                return (0, '', '')          # fetch succeeds
            return (128, '', 'fatal: bad revision')   # rev-list fails

        with patch('orchestrator.git_ops._run', side_effect=fake_run):
            ref, stale = await git_ops._freshen_main()

        assert ref == git_ops.config.main_branch
        assert stale is None

    async def test_freshen_main_behind_count_value_error(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """When behind rev-list returns non-numeric stdout, _freshen_main returns (main_branch, None)."""
        git_ops, _origin = git_ops_with_remote

        async def fake_run(cmd, cwd=None):
            if 'fetch' in cmd:
                return (0, '', '')           # fetch succeeds
            return (0, 'not-a-number', '')   # rev-list returns garbage

        with patch('orchestrator.git_ops._run', side_effect=fake_run):
            ref, stale = await git_ops._freshen_main()

        assert ref == git_ops.config.main_branch
        assert stale is None

    async def test_freshen_main_ahead_count_value_error(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """When ahead rev-list returns non-numeric stdout, falls back to (main_branch, behind)."""
        git_ops, _origin = git_ops_with_remote

        remote_ref = f'{git_ops.config.remote}/'

        async def fake_run(cmd, cwd=None):
            if 'fetch' in cmd:
                return (0, '', '')           # fetch succeeds
            if 'rev-list' in cmd:
                # Distinguish behind vs ahead by which side of '..' the remote ref is on:
                #   behind range: <local>..<remote>  (e.g. main..origin/main)
                #   ahead  range: <remote>..<local>  (e.g. origin/main..main)
                range_arg = next((arg for arg in cmd if '..' in arg), '')
                if range_arg.startswith(remote_ref):
                    return (0, 'not-a-number', '')  # ahead rev-list: garbage
                return (0, '3', '')                 # behind rev-list: 3 behind
            return (0, '', '')

        with patch('orchestrator.git_ops._run', side_effect=fake_run):
            ref, stale = await git_ops._freshen_main()

        # Falls back to local main; reports behind count as-is
        assert ref == git_ops.config.main_branch
        assert stale == 3

    async def test_freshen_main_ahead_rev_list_fails(
        self, git_ops_with_remote: tuple[GitOps, Path], caplog,
    ):
        """When ahead rev-list exits non-zero, _freshen_main returns (main_branch, behind) and logs a warning."""
        git_ops, _origin = git_ops_with_remote

        call_count = 0

        async def fake_run(cmd, cwd=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (0, '', '')                      # fetch succeeds
            elif call_count == 2:
                return (0, '3', '')                     # behind rev-list: 3 commits behind
            return (128, '', 'fatal: bad revision')     # ahead rev-list fails

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'), \
             patch('orchestrator.git_ops._run', side_effect=fake_run):
            ref, stale = await git_ops._freshen_main()

        assert ref == git_ops.config.main_branch
        assert stale == 3
        assert any('rev-list (ahead) failed' in r.message for r in caplog.records)


@pytest.mark.asyncio
class TestCreateWorktreeFreshening:
    async def test_create_worktree_freshens_from_remote(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """Worktree based on remote ref when origin is ahead — new file is present."""
        git_ops, origin = git_ops_with_remote
        await _push_n_commits_to_origin(origin, 1, prefix='fresh')
        worktree_info = await git_ops.create_worktree('freshen-test')
        assert (worktree_info.path / 'fresh_0.txt').exists()
        # Exactly 1 commit was pushed, so stale_commits must reflect that.
        assert worktree_info.stale_commits == 1
        # base_commit must match origin/main SHA captured after create_worktree
        # (which internally fetched, so origin/main is now up-to-date in the local repo).
        _, expected_sha, _ = await _run(
            ['git', 'rev-parse', 'origin/main'], cwd=git_ops.project_root,
        )
        assert worktree_info.base_commit == expected_sha

    async def test_create_worktree_stale_commits_populated(
        self, git_ops_with_remote: tuple[GitOps, Path],
    ):
        """stale_commits == 2 when origin is 2 commits ahead at create_worktree time."""
        git_ops, origin = git_ops_with_remote
        await _push_n_commits_to_origin(origin, 2)
        worktree_info = await git_ops.create_worktree('stale-commits-test')
        assert worktree_info.stale_commits == 2

    async def test_create_worktree_stale_commits_none_without_remote(
        self, git_ops: GitOps,
    ):
        """stale_commits is None when no remote is configured (graceful degradation)."""
        worktree_info = await git_ops.create_worktree('no-remote-test')
        assert worktree_info.stale_commits is None

    async def test_create_worktree_revparse_fallback(self, git_ops: GitOps):
        """When rev-parse of start_ref fails, create_worktree falls back to local main.

        _freshen_main returns 'origin/nonexistent-ref' (a ref that doesn't exist
        in this no-remote repo). The rev-parse should fail, triggering a fallback
        to local main. The worktree should still be created successfully with a
        valid base_commit SHA.
        """
        _, local_main_sha, _ = await _run(
            ['git', 'rev-parse', git_ops.config.main_branch],
            cwd=git_ops.project_root,
        )
        local_main_sha = local_main_sha.strip()

        with patch.object(
            git_ops, '_freshen_main', return_value=('origin/nonexistent-ref', 3),
        ):
            worktree_info = await git_ops.create_worktree('revparse-fallback-test')

        assert (worktree_info.path / 'README.md').exists()
        assert len(worktree_info.base_commit) == 40
        assert worktree_info.base_commit == local_main_sha
        assert worktree_info.stale_commits == 3  # persists through fallback


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

    async def test_tasks_json_dirty_does_not_block_advance(self, git_ops: GitOps):
        """Dirty .taskmaster/tasks/tasks.json overlapping merge diff does NOT block advance.

        The fused-memory MCP commits tasks.json out-of-band on a fire-and-forget
        schedule, racing with the overlap check.  Branches never legitimately
        introduce tasks.json deltas (commit() excludes it from staging), so a
        tasks.json-only overlap is always the MCP race — must not halt the queue.
        """
        # Seed tasks.json on main so it's a tracked file
        tasks_dir = git_ops.project_root / '.taskmaster' / 'tasks'
        tasks_dir.mkdir(parents=True)
        (tasks_dir / 'tasks.json').write_text('{"v": 1}\n')
        await _run(
            ['git', 'add', '.taskmaster/tasks/tasks.json'],
            cwd=git_ops.project_root,
        )
        await _run(
            ['git', 'commit', '-m', 'seed tasks.json'],
            cwd=git_ops.project_root,
        )

        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )

        # Simulate the MCP race: tasks.json dirty in project_root before overlap check
        (tasks_dir / 'tasks.json').write_text('{"v": 2}\n')

        # Build a branch whose merge diff includes tasks.json (bypass commit()'s
        # pathspec by staging directly — this is what an out-of-band agent would do)
        worktree_info = await git_ops.create_worktree('overlap-tasks-json')
        branch_tasks = worktree_info.path / '.taskmaster' / 'tasks' / 'tasks.json'
        branch_tasks.parent.mkdir(parents=True, exist_ok=True)
        branch_tasks.write_text('{"v": 3}\n')
        await _run(
            ['git', 'add', '.taskmaster/tasks/tasks.json'],
            cwd=worktree_info.path,
        )
        await _run(
            ['git', 'commit', '-m', 'change tasks.json on branch'],
            cwd=worktree_info.path,
        )
        merge_result = await git_ops.merge_to_main(
            worktree_info.path, 'overlap-tasks-json',
        )
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        result = await git_ops.advance_main(merge_result.merge_commit)
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        # tasks.json overlap is filtered by pathspec → advance proceeds
        assert result == 'advanced'

        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_before.strip() != main_after.strip()

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
    async def test_scrub_returns_clean_when_no_contamination(
        self, git_ops: GitOps, caplog,
    ):
        """scrub_task_dir_from_tree returns CLEAN when ls-tree shows no tracked .task/ files."""
        # Create a real worktree for a realistic working directory (no mock yet)
        worktree_info = await git_ops.create_worktree('scrub-clean')

        # Create a .task/ directory with sentinel content on disk — canary for rmtree
        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        sentinel = task_dir / 'sentinel.txt'
        sentinel.write_text('canary-no-rmtree\n')

        async def mock_run(cmd, cwd=None):
            # ls-tree returns empty stdout — no tracked .task/ files
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '', '')
            # Strict — no other git commands should be reached on the CLEAN path
            pytest.fail(f'unexpected _run call on CLEAN path: {cmd}')

        with (
            caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', side_effect=mock_run),
        ):
            result = await scrub_task_dir_from_tree(worktree_info.path, 'test-clean')

        # Return value must be CLEAN — no tracked .task/ files in tree
        assert result.outcome == ScrubOutcome.CLEAN, (
            f'Expected ScrubOutcome.CLEAN when ls-tree is empty, got {result!r}'
        )

        # Filesystem .task/ must still exist — rmtree must NOT have run
        assert sentinel.exists(), (
            'sentinel.txt was deleted — rmtree must not run on CLEAN path'
        )

        # No WARNING or ERROR should have been logged on the CLEAN path
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, (
            f'Unexpected log entries on CLEAN path: {[r.getMessage() for r in warnings]}'
        )

    async def test_scrub_returns_failed_when_git_rm_fails(
        self, git_ops: GitOps, caplog,
    ):
        """scrub_task_dir_from_tree returns FAILED and skips rmtree/commit when git rm fails."""
        # Create a real worktree for a realistic working directory (no mock yet)
        worktree_info = await git_ops.create_worktree('scrub-rm-fail')

        # Create a .task/ directory with sentinel content on disk
        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        sentinel = task_dir / 'sentinel.txt'
        sentinel.write_text('keep-me\n')

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
            result = await scrub_task_dir_from_tree(worktree_info.path, 'test-rm-fail')

        # Return value must be FAILED — git rm failed, scrub did not complete
        assert result.outcome == ScrubOutcome.FAILED, (
            f'Expected outcome=FAILED on git rm failure, got {result!r}'
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

    async def test_scrub_returns_scrubbed_on_happy_path(
        self, git_ops: GitOps, caplog,
    ):
        """scrub_task_dir_from_tree returns SCRUBBED, runs rmtree, and commits on success."""
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
            result = await scrub_task_dir_from_tree(worktree_info.path, 'test-happy')

        # Return value must be SCRUBBED
        assert result.outcome == ScrubOutcome.SCRUBBED, (
            f'Expected outcome=SCRUBBED on success, got {result!r}'
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
        """scrub_task_dir_from_tree returns FAILED and logs error when commit fails post-rm."""
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
            result = await scrub_task_dir_from_tree(worktree_info.path, 'test-commit-fail')

        # Return value must be FAILED — commit did not succeed
        assert result.outcome == ScrubOutcome.FAILED, (
            f'Expected outcome=FAILED on commit failure, got {result!r}'
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

    async def test_scrub_failed_result_carries_error(
        self, tmp_path: Path,
    ):
        """When git rm fails, the returned ScrubResult must carry the git stderr.

        After the ScrubResult → dataclass conversion, the failure path sets
        outcome=ScrubOutcome.FAILED and error=<stderr>.strip().  This test drives
        that conversion by asserting .outcome and .error on the return value.

        Uses tmp_path directly (no real worktree) since _run is fully mocked.
        """
        async def mock_run(cmd, cwd=None):
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '.task/tracked.txt', '')
            if cmd[:5] == ['git', 'rm', '-r', '--cached', '--']:
                return (1, '', 'fatal: pathspec error from git rm')
            pytest.fail(f'unexpected _run call: {cmd}')

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await scrub_task_dir_from_tree(tmp_path, 'carries-err')

        assert result.outcome == ScrubOutcome.FAILED, (
            f'Expected outcome=FAILED on git rm failure, got {result!r}'
        )
        assert result.error is not None, 'Expected error to be set on git rm failure'
        assert 'pathspec' in result.error, (
            f'Expected git rm stderr in .error, got: {result.error!r}'
        )

    async def test_scrub_failed_whitespace_stderr_collapses_to_none(
        self, tmp_path: Path,
    ):
        """Whitespace-only git rm stderr must collapse to error=None.

        The production code uses ``err.strip() or None`` (git_ops.py:126) so that
        whitespace-only stderr (e.g. '   \\n') is normalised to None rather than
        stored as a meaningless whitespace string.  This companion test to
        test_scrub_failed_result_carries_error drives that normalisation branch.

        Uses tmp_path directly (no real worktree) since _run is fully mocked.
        """
        async def mock_run(cmd, cwd=None):
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '.task/tracked.txt', '')
            if cmd[:5] == ['git', 'rm', '-r', '--cached', '--']:
                return (1, '', '   \n')
            pytest.fail(f'unexpected _run call: {cmd}')

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await scrub_task_dir_from_tree(tmp_path, 'whitespace-err')

        assert result.outcome == ScrubOutcome.FAILED, (
            f'Expected outcome=FAILED on git rm failure, got {result!r}'
        )
        assert result.error is None, (
            f'Expected error=None for whitespace-only stderr, got: {result.error!r}'
        )

    async def test_scrub_scrubbed_result_has_no_error(
        self, tmp_path: Path,
    ):
        """When scrub succeeds, ScrubResult must have outcome=SCRUBBED and error=None.

        Uses tmp_path directly (no real worktree) since _run is fully mocked.
        """
        async def mock_run(cmd, cwd=None):
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '.task/tracked.txt', '')
            if cmd[:5] == ['git', 'rm', '-r', '--cached', '--']:
                return (0, '', '')
            if len(cmd) >= 2 and cmd[1] == 'commit':
                return (0, '', '')
            pytest.fail(f'unexpected _run call: {cmd}')

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await scrub_task_dir_from_tree(tmp_path, 'no-err-ok')

        assert result.outcome == ScrubOutcome.SCRUBBED, (
            f'Expected outcome=SCRUBBED on success, got {result!r}'
        )
        assert result.error is None, f'Expected error=None on success, got {result.error!r}'

    async def test_scrub_clean_result_has_no_error(
        self, tmp_path: Path,
    ):
        """When no .task/ files are present, ScrubResult must have outcome=CLEAN and error=None.

        Uses tmp_path directly (no real worktree) since _run is fully mocked.
        """
        async def mock_run(cmd, cwd=None):
            if cmd[:4] == ['git', 'ls-tree', '-r', '--name-only'] and '.task/' in cmd:
                return (0, '', '')  # empty — no .task/ tracked
            pytest.fail(f'unexpected _run call on clean path: {cmd}')

        with patch('orchestrator.git_ops._run', side_effect=mock_run):
            result = await scrub_task_dir_from_tree(tmp_path, 'clean-no-err')

        assert result.outcome == ScrubOutcome.CLEAN, (
            f'Expected outcome=CLEAN on empty tree, got {result!r}'
        )
        assert result.error is None, f'Expected error=None on clean, got {result.error!r}'

    async def test_scrub_amend_false_creates_new_commit(
        self, git_ops: GitOps,
    ):
        """scrub_task_dir_from_tree(amend=False) extends the commit chain.

        The amend=False path (used by create_worktree, line 342 of git_ops.py)
        must create a NEW child commit rather than rewriting the existing one.
        This integration test verifies against a real git repository:
        (a) outcome == SCRUBBED and error is None,
        (b) HEAD moved to a new SHA after the scrub,
        (c) the old HEAD is the first parent of the new HEAD (new commit, not amend),
        (d) .task/ is absent from the new HEAD commit tree,
        (e) the new commit message contains 'chore: remove .task/ contamination'.

        Uses git_ops fixture for a real git repo — no mocks.
        """
        # Create a worktree on a fresh branch with a regular commit.
        worktree_info = await git_ops.create_worktree('amend-false-branch')
        (worktree_info.path / 'work.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add work file')

        # Inject .task/ contamination, bypassing the .task/.gitignore defence.
        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text('{"contamination": true}\n')
        rc, _, _ = await _run(['git', 'add', '-f', '.task/plan.json'], cwd=worktree_info.path)
        assert rc == 0, 'setup: git add -f .task/plan.json failed'
        rc, _, _ = await _run(
            ['git', 'commit', '-m', 'Simulated .task/ contamination'],
            cwd=worktree_info.path,
        )
        assert rc == 0, 'setup: git commit of contamination failed'

        # Record HEAD before scrub.
        _, old_head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree_info.path)
        old_head = old_head.strip()
        assert len(old_head) == 40, f'Pre-condition: expected 40-char SHA, got {old_head!r}'

        # Verify contamination is present before scrub.
        _, ls_before, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--', '.task/'],
            cwd=worktree_info.path,
        )
        assert '.task/plan.json' in ls_before, (
            f'Pre-condition: expected .task/plan.json in tree, got: {ls_before!r}'
        )

        # Call scrub with amend=False — must create a new child commit.
        result = await scrub_task_dir_from_tree(
            worktree_info.path, 'amend-false-test', amend=False,
        )

        # (a) Outcome must be SCRUBBED with no error.
        assert result.outcome == ScrubOutcome.SCRUBBED, (
            f'Expected outcome=SCRUBBED, got {result!r}'
        )
        assert result.error is None, (
            f'Expected error=None on successful scrub, got: {result.error!r}'
        )

        # (b) HEAD must have moved to a new SHA.
        _, new_head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree_info.path)
        new_head = new_head.strip()
        assert new_head != old_head, (
            f'Expected HEAD to move after amend=False scrub, but still {old_head!r}'
        )

        # (c) Old HEAD must be the first parent of new HEAD — proves a new child
        #     commit was created rather than an amendment of the contamination commit.
        _, parent, _ = await _run(
            ['git', 'rev-parse', 'HEAD^'],
            cwd=worktree_info.path,
        )
        assert parent.strip() == old_head, (
            f'Expected old HEAD ({old_head}) to be parent of new HEAD, '
            f'but HEAD^ is {parent.strip()!r}'
        )

        # (d) .task/ must be absent from the new HEAD commit tree.
        _, task_in_tree, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--', '.task/'],
            cwd=worktree_info.path,
        )
        assert not task_in_tree.strip(), (
            f'.task/ must be absent from new commit tree, but found: {task_in_tree!r}'
        )

        # (e) New commit message must contain the expected scrub marker text.
        _, commit_msg, _ = await _run(
            ['git', 'log', '-1', '--format=%B'],
            cwd=worktree_info.path,
        )
        assert 'chore: remove .task/ contamination' in commit_msg, (
            f'Expected commit message to contain scrub marker, got: {commit_msg!r}'
        )


@pytest.mark.asyncio
class TestMergeToMainScrubFailure:
    """Tests for merge_to_main returning success=False when scrub fails."""

    async def test_merge_to_main_fails_when_scrub_fails(
        self, git_ops: GitOps,
    ):
        """merge_to_main must return MergeResult(success=False) when scrub fails.

        When scrub_task_dir_from_tree returns ScrubResult.FAILED, merge_to_main
        should fail fast: clean up the merge worktree and return
        MergeResult(success=False, conflicts=False, ...) rather than returning
        MergeResult(success=True) with a contaminated commit.
        """
        # Set up a feature branch with a committed file.
        worktree_info = await git_ops.create_worktree('scrub-fail-branch')
        (worktree_info.path / 'scrub_test.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add scrub test file')

        # Patch _scrub_task_dir_from_tree to return FAILED, simulating a scrub
        # failure after the merge commit has been created.
        async def fake_scrub(*args, **kwargs):
            return ScrubResult(outcome=ScrubOutcome.FAILED)

        with patch(
            'orchestrator.git_ops.scrub_task_dir_from_tree',
            new=fake_scrub,
        ):
            result = await git_ops.merge_to_main(worktree_info.path, 'scrub-fail-branch')

        # (1) success must be False — scrub failure is a hard stop
        assert result.success is False, (
            f'Expected success=False on scrub failure, got success={result.success!r}'
        )

        # (2) conflicts must be False — this is NOT a merge conflict
        assert result.conflicts is False, (
            f'Expected conflicts=False on scrub failure, got conflicts={result.conflicts!r}'
        )

        # (3) details must mention 'scrub' and the branch name
        assert 'scrub' in result.details.lower(), (
            f'Expected "scrub" in details, got: {result.details!r}'
        )
        assert 'task/scrub-fail-branch' in result.details, (
            f'Expected full prefixed branch name in details, got: {result.details!r}'
        )

        # (4) pre_merge_sha must be a valid 40-char SHA
        assert result.pre_merge_sha is not None, 'Expected pre_merge_sha to be set'
        assert len(result.pre_merge_sha.strip()) == 40, (
            f'Expected 40-char SHA, got: {result.pre_merge_sha!r}'
        )

        # (5) merge_commit must be None — no committed merge SHA on failure
        assert result.merge_commit is None, (
            f'Expected merge_commit=None on scrub failure, got: {result.merge_commit!r}'
        )

        # (6) merge_worktree must be None — mirrors the non-conflict failure path
        assert result.merge_worktree is None, (
            f'Expected merge_worktree=None on scrub failure, got: {result.merge_worktree!r}'
        )

        # (7) No _merge-* worktrees should remain registered.
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

        # (8) No _merge-* directories should remain on disk.
        worktree_base = git_ops.worktree_base
        if worktree_base.exists():
            leak_dirs = list(worktree_base.glob('_merge-*'))
            assert not leak_dirs, (
                f'Leaked merge worktree directories on disk: {leak_dirs}'
            )

    async def test_merge_to_main_succeeds_when_scrub_cleans_task_dir(
        self, git_ops: GitOps,
    ):
        """merge_to_main must return success=True when scrub_task_dir_from_tree
        returns ScrubResult.SCRUBBED (i.e. .task/ was found and removed cleanly).

        Guards against regressions where a future change accidentally treats
        SCRUBBED the same as FAILED.
        """
        worktree_info = await git_ops.create_worktree('scrub-ok-branch')
        (worktree_info.path / 'scrub_ok.py').write_text('y = 2\n')
        await git_ops.commit(worktree_info.path, 'Add scrub-ok file')

        async def fake_scrub_ok(*args, **kwargs):
            return ScrubResult(outcome=ScrubOutcome.SCRUBBED)

        with patch(
            'orchestrator.git_ops.scrub_task_dir_from_tree',
            new=fake_scrub_ok,
        ):
            result = await git_ops.merge_to_main(worktree_info.path, 'scrub-ok-branch')

        # SCRUBBED must not trigger the failure path — merge should succeed.
        assert result.success is True, (
            f'Expected success=True when scrub returns SCRUBBED, got {result.success!r}'
        )
        assert result.merge_commit is not None, (
            'Expected a valid merge_commit SHA when scrub returns SCRUBBED'
        )
        assert len(result.merge_commit.strip()) == 40, (
            f'Expected 40-char merge_commit SHA, got: {result.merge_commit!r}'
        )
        assert result.conflicts is False, (
            f'Expected conflicts=False on SCRUBBED result, got {result.conflicts!r}'
        )

        # Clean up the merge worktree to avoid polluting other tests.
        if result.merge_worktree is not None:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)

    async def test_merge_to_main_scrubs_real_task_dir(
        self, git_ops: GitOps,
    ):
        """merge_to_main strips .task/ from the merge commit via the real scrub.

        Unlike test_merge_to_main_succeeds_when_scrub_cleans_task_dir, this test
        uses NO mock — it commits a real .task/plan.json file on the feature branch
        and verifies that merge_to_main produces a clean merge commit with no .task/
        entries in the tree.  This exercises the real scrub_task_dir_from_tree with
        amend=True on an actual contaminated merge commit.
        """
        # Create a worktree and commit a regular file so the branch has content.
        worktree_info = await git_ops.create_worktree('scrub-real-branch')
        (worktree_info.path / 'feature.py').write_text('def feature(): pass\n')
        await git_ops.commit(worktree_info.path, 'Add feature file')

        # Inject .task/ contamination directly via git commands, bypassing the
        # safety guards in git_ops.commit (which would normally unstage .task/).
        # Use -f to force-add past the .task/.gitignore ('*') that create_worktree
        # places there as a defence-in-depth measure.
        task_dir = worktree_info.path / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text('{"contamination": true}\n')
        await _run(['git', 'add', '-f', '.task/plan.json'], cwd=worktree_info.path)
        await _run(
            ['git', 'commit', '-m', 'Simulated .task/ contamination'],
            cwd=worktree_info.path,
        )

        # Verify contamination is present on the branch before merge.
        _, ls_before, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'HEAD', '--', '.task/'],
            cwd=worktree_info.path,
        )
        assert '.task/plan.json' in ls_before, (
            f'Pre-condition: expected .task/plan.json on branch, got: {ls_before!r}'
        )

        # Call merge_to_main with NO mock — uses real scrub_task_dir_from_tree.
        result = await git_ops.merge_to_main(worktree_info.path, 'scrub-real-branch')

        try:
            # (a) Merge must succeed.
            assert result.success is True, (
                f'Expected success=True when real scrub cleans .task/, got {result.success!r}'
            )

            # (b) A merge commit must have been created.
            assert result.merge_commit is not None, (
                'Expected a valid merge_commit SHA when scrub succeeds'
            )

            # (c) Verify .task/ is absent from the merge commit tree.
            _, task_in_tree, _ = await _run(
                ['git', 'ls-tree', '-r', '--name-only', result.merge_commit.strip(), '--', '.task/'],
                cwd=git_ops.project_root,
            )
            assert not task_in_tree.strip(), (
                f'.task/ must be absent from merge commit tree, but found: {task_in_tree!r}'
            )
        finally:
            # Ensure merge worktree is cleaned up even when assertions fail.
            if result.merge_worktree is not None:
                await git_ops.cleanup_merge_worktree(result.merge_worktree)

    async def test_merge_to_main_scrub_failure_details_include_root_cause(
        self, git_ops: GitOps,
    ):
        """merge_to_main must surface ScrubResult.error in MergeResult.details.

        When scrub_task_dir_from_tree returns a ScrubResult with error set,
        the failure reason (raw git stderr) must appear in MergeResult.details
        so MergeQueue propagates it to MergeOutcome.reason without log scraping.

        This test is the failing test for step-5.  It will pass once step-6
        wires scrub_result.error into the details f-string.
        """
        worktree_info = await git_ops.create_worktree('scrub-root-cause-branch')
        (worktree_info.path / 'rc_test.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add rc_test file')

        root_cause = 'fatal: cannot amend merge commit'

        async def fake_scrub_with_error(*args, **kwargs):
            return ScrubResult(outcome=ScrubOutcome.FAILED, error=root_cause)

        with patch(
            'orchestrator.git_ops.scrub_task_dir_from_tree',
            new=fake_scrub_with_error,
        ):
            result = await git_ops.merge_to_main(
                worktree_info.path, 'scrub-root-cause-branch',
            )

        assert result.success is False, (
            f'Expected success=False on scrub failure, got {result.success!r}'
        )
        assert 'cannot amend merge commit' in result.details, (
            f'Expected git stderr in details for operator visibility, got: {result.details!r}'
        )

    async def test_merge_to_main_scrub_failure_no_stderr_fallback(
        self, git_ops: GitOps,
    ):
        """merge_to_main must NOT include '(no stderr)' when scrub error is None.

        When scrub_task_dir_from_tree returns FAILED with error=None (e.g. stderr
        was empty/whitespace), MergeResult.details must not contain the old
        '(no stderr)' fallback string. The standardised format_error() helper
        returns '' in that case, so nothing extra is appended.
        """
        worktree_info = await git_ops.create_worktree('scrub-no-stderr-branch')
        (worktree_info.path / 'ns_test.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Add ns_test file')

        async def fake_scrub_no_error(*args, **kwargs):
            return ScrubResult(outcome=ScrubOutcome.FAILED)  # error=None

        with patch(
            'orchestrator.git_ops.scrub_task_dir_from_tree',
            new=fake_scrub_no_error,
        ):
            result = await git_ops.merge_to_main(
                worktree_info.path, 'scrub-no-stderr-branch',
            )

        assert result.success is False, (
            f'Expected success=False on scrub failure, got {result.success!r}'
        )
        assert '(no stderr)' not in result.details, (
            f'Expected no "(no stderr)" fallback in details, got: {result.details!r}'
        )


class TestScrubResultInvariant:
    """Unit tests for ScrubResult.__post_init__ guard.

    The invariant: error may only be non-None when outcome is ScrubOutcome.FAILED.
    All other (outcome, error) combinations are semantically invalid and should
    raise ValueError at construction time.
    """

    def test_clean_with_error_raises(self):
        """ScrubResult(CLEAN, error=...) must raise ValueError."""
        with pytest.raises(ValueError):
            ScrubResult(outcome=ScrubOutcome.CLEAN, error='some error')

    def test_scrubbed_with_error_raises(self):
        """ScrubResult(SCRUBBED, error=...) must raise ValueError."""
        with pytest.raises(ValueError):
            ScrubResult(outcome=ScrubOutcome.SCRUBBED, error='some error')

    def test_failed_with_error_succeeds(self):
        """ScrubResult(FAILED, error=...) is valid and must not raise."""
        result = ScrubResult(outcome=ScrubOutcome.FAILED, error='fatal: git error')
        assert result.outcome == ScrubOutcome.FAILED
        assert result.error == 'fatal: git error'

    def test_failed_without_error_succeeds(self):
        """ScrubResult(FAILED) with error=None is valid (no error captured)."""
        result = ScrubResult(outcome=ScrubOutcome.FAILED)
        assert result.outcome == ScrubOutcome.FAILED
        assert result.error is None

    def test_clean_without_error_succeeds(self):
        """ScrubResult(CLEAN) with error=None is valid."""
        result = ScrubResult(outcome=ScrubOutcome.CLEAN)
        assert result.outcome == ScrubOutcome.CLEAN
        assert result.error is None

    def test_scrubbed_without_error_succeeds(self):
        """ScrubResult(SCRUBBED) with error=None is valid."""
        result = ScrubResult(outcome=ScrubOutcome.SCRUBBED)
        assert result.outcome == ScrubOutcome.SCRUBBED
        assert result.error is None


class TestScrubResultFormatError:
    """Unit tests for ScrubResult.format_error() helper method.

    format_error(prefix='') returns prefix+error when error is set,
    or empty string when error is None.
    """

    def test_failed_with_error_default_prefix(self):
        """FAILED with error and no prefix returns the bare error string."""
        result = ScrubResult(outcome=ScrubOutcome.FAILED, error='fatal: git rm failed')
        assert result.format_error() == 'fatal: git rm failed', (
            f'Expected bare error string, got {result.format_error()!r}'
        )

    def test_failed_with_error_custom_prefix(self):
        """FAILED with error and custom prefix returns prefix+error."""
        result = ScrubResult(outcome=ScrubOutcome.FAILED, error='fatal: git rm failed')
        assert result.format_error(prefix=' Error: ') == ' Error: fatal: git rm failed', (
            f'Expected prefixed error, got {result.format_error(prefix=" Error: ")!r}'
        )

    def test_failed_with_no_error_returns_empty(self):
        """FAILED with error=None returns empty string regardless of prefix."""
        result = ScrubResult(outcome=ScrubOutcome.FAILED)
        assert result.format_error() == '', (
            f'Expected empty string when error is None, got {result.format_error()!r}'
        )
        assert result.format_error(prefix=' Error: ') == '', (
            'Expected empty string even with prefix when error is None'
        )

    def test_clean_with_no_error_returns_empty(self):
        """CLEAN with error=None returns empty string."""
        result = ScrubResult(outcome=ScrubOutcome.CLEAN)
        assert result.format_error() == '', (
            f'Expected empty string for CLEAN outcome, got {result.format_error()!r}'
        )

    def test_scrubbed_with_no_error_returns_empty(self):
        """SCRUBBED with error=None returns empty string."""
        result = ScrubResult(outcome=ScrubOutcome.SCRUBBED)
        assert result.format_error() == '', (
            f'Expected empty string for SCRUBBED outcome, got {result.format_error()!r}'
        )

    def test_failed_with_empty_string_error_raises_value_error(self):
        """FAILED with error='' is rejected at construction time.

        All production call-sites normalise empty/whitespace stderr to None via
        ``err.strip() or None`` before constructing ScrubResult.  Permitting an
        empty-string error would create an ambiguous state (``error is not None``
        but ``not error``).  The __post_init__ guard makes the invariant explicit:
        ``error`` is either None or a non-empty, non-whitespace-only string.
        """
        with pytest.raises(ValueError, match='empty or whitespace-only'):
            ScrubResult(outcome=ScrubOutcome.FAILED, error='')

    def test_failed_with_whitespace_only_error_raises_value_error(self):
        """FAILED with error='   ' (whitespace only) is also rejected."""
        with pytest.raises(ValueError, match='empty or whitespace-only'):
            ScrubResult(outcome=ScrubOutcome.FAILED, error='   ')


@pytest.mark.asyncio
class TestPushMain:
    """Best-effort push of local main to <remote>/<main_branch>.

    Lives next to advance_main: each successful CAS advance is mirrored to
    origin so an external clone (humans, CI, mirrors) sees the same history
    the merge worker just produced.
    """

    async def test_push_main_pushes_local_advance(
        self, git_repo_with_remote: tuple[Path, Path],
    ):
        """Happy path: a commit added locally lands on the bare origin."""
        origin, local = git_repo_with_remote
        git_ops = GitOps(GitConfig(push_after_advance=True), local)

        (local / 'local.txt').write_text('local\n')
        await _run(['git', 'add', '-A'], cwd=local)
        await _run(['git', 'commit', '-m', 'local commit'], cwd=local)
        _, local_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=local)

        result = await git_ops.push_main()

        assert result == 'pushed'
        _, origin_sha, _ = await _run(['git', 'rev-parse', 'main'], cwd=origin)
        assert origin_sha == local_sha

    async def test_push_main_noop_when_disabled(
        self, git_repo_with_remote: tuple[Path, Path],
    ):
        """push_after_advance=False short-circuits to 'noop' without contacting origin."""
        origin, local = git_repo_with_remote
        cfg = GitConfig(push_after_advance=False)
        git_ops = GitOps(cfg, local)

        (local / 'local.txt').write_text('local\n')
        await _run(['git', 'add', '-A'], cwd=local)
        await _run(['git', 'commit', '-m', 'local commit'], cwd=local)
        _, origin_sha_before, _ = await _run(['git', 'rev-parse', 'main'], cwd=origin)

        result = await git_ops.push_main()

        assert result == 'noop'
        _, origin_sha_after, _ = await _run(['git', 'rev-parse', 'main'], cwd=origin)
        assert origin_sha_after == origin_sha_before  # origin unchanged

    async def test_push_main_rejected_on_diverged_origin(
        self, git_repo_with_remote: tuple[Path, Path], caplog,
    ):
        """When origin has commits we lack, push must be rejected and NOT forced."""
        origin, local = git_repo_with_remote
        git_ops = GitOps(GitConfig(push_after_advance=True), local)

        # Origin gets a commit we don't have
        await _push_n_commits_to_origin(origin, 1, prefix='diverge')

        # Local diverges with its own commit (without fetching/merging)
        (local / 'local.txt').write_text('local\n')
        await _run(['git', 'add', '-A'], cwd=local)
        await _run(['git', 'commit', '-m', 'local divergent commit'], cwd=local)

        _, origin_sha_before, _ = await _run(['git', 'rev-parse', 'main'], cwd=origin)

        with caplog.at_level(logging.ERROR, logger='orchestrator.git_ops'):
            result = await git_ops.push_main()

        assert result == 'rejected'
        # Origin must be unchanged — no force-push
        _, origin_sha_after, _ = await _run(['git', 'rev-parse', 'main'], cwd=origin)
        assert origin_sha_after == origin_sha_before
        assert any('rejected (non-fast-forward)' in r.message for r in caplog.records)

    async def test_push_main_error_on_unreachable_remote(
        self, tmp_path: Path, caplog,
    ):
        """Unreachable remote returns 'error' — best-effort, never raises."""
        # Local repo with origin pointing at a path that does not exist
        local = tmp_path / 'local'
        local.mkdir()
        await _run(['git', 'init', '-b', 'main'], cwd=local)
        await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=local)
        await _run(['git', 'config', 'user.name', 'Test'], cwd=local)
        (local / 'README.md').write_text('# Test\n')
        await _run(['git', 'add', '-A'], cwd=local)
        await _run(['git', 'commit', '-m', 'Initial commit'], cwd=local)
        await _run(
            ['git', 'remote', 'add', 'origin', str(tmp_path / 'does-not-exist')],
            cwd=local,
        )

        git_ops = GitOps(GitConfig(), local)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = await git_ops.push_main()

        assert result == 'error'
        assert any('Push of main to origin failed' in r.message for r in caplog.records)


@pytest.mark.asyncio
class TestResolveBranchSha:
    async def test_returns_sha_for_existing_ref(self, git_ops: GitOps):
        """resolve_branch_sha returns the 40-char SHA for a branch that exists.

        Uses create_worktree to materialise a task/resolve-1 branch, then
        asserts the returned SHA matches a direct rev-parse call.
        """
        wt_info = await git_ops.create_worktree('resolve-1')
        # Confirm the branch was created
        assert wt_info is not None

        resolved = await git_ops.resolve_branch_sha('task/resolve-1')

        # Get the expected SHA via _run
        _, expected_sha, _ = await _run(
            ['git', 'rev-parse', 'task/resolve-1'],
            cwd=git_ops.project_root,
        )
        expected_sha = expected_sha.strip()

        assert resolved is not None
        assert resolved == expected_sha
        assert len(resolved) == 40

    async def test_returns_none_for_missing_ref(self, git_ops: GitOps):
        """resolve_branch_sha returns None (not empty string, not an exception)
        when the branch ref does not exist.

        Regression lock: a future refactor must not silently switch to raising
        or returning '' — both would break the harness fallback path.
        """
        result = await git_ops.resolve_branch_sha('task/does-not-exist')
        assert result is None

    @pytest.mark.parametrize(
        'bad_ref',
        [
            'task/does-not-exist',   # simply absent branch
            'not a valid ref',       # contains spaces — syntactically malformed
            '..bad..',               # double-dot traversal form — rejected by git
        ],
    )
    async def test_returns_none_for_bad_refs(self, git_ops: GitOps, bad_ref: str):
        """resolve_branch_sha returns None for any ref git cannot resolve.

        Covers both 'missing' (rc=128 from rev-parse not finding the ref) and
        'malformed' (rc=128 from git rejecting the name) error modes, locking
        in the rc-based fallback contract for the harness fallback path.
        """
        result = await git_ops.resolve_branch_sha(bad_ref)
        assert result is None


@pytest.mark.asyncio
class TestFindMergeMarker:
    """Real-git tests for GitOps.find_merge_marker.

    Tests cover the four cases described in the plan:
    (a) branch deleted with a merge marker on main → returns SHA
    (b) branch still exists → returns None (resolve_branch_sha gate)
    (c) branch never existed, no marker → returns None
    (d) substring safety: task/1 query must not match 'Merge task/10 into main'
    """

    async def test_returns_merge_sha_when_branch_deleted_with_marker(
        self, git_ops: GitOps
    ):
        """find_merge_marker returns the merge commit SHA when the branch was
        merged to main and then deleted via cleanup_worktree.

        Real git fixture: create_worktree → commit → merge_to_main → advance_main
        → cleanup_merge_worktree → cleanup_worktree (branch deleted), then assert
        the returned SHA matches the merge commit SHA.
        """
        tid = 'mm-1'
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        (wt_info.path / f'{tid}.py').write_text(f'{tid} = True\n')
        await git_ops.commit(wt_info.path, f'Add {tid}')

        result = await git_ops.merge_to_main(wt_info.path, tid)
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None

        adv = await git_ops.advance_main(result.merge_commit)
        assert adv == 'advanced'

        await git_ops.cleanup_merge_worktree(result.merge_worktree)
        await git_ops.cleanup_worktree(wt_info.path, tid)

        # Branch is now deleted — find_merge_marker should find the merge commit
        marker_sha = await git_ops.find_merge_marker(f'task/{tid}')

        assert marker_sha is not None
        assert marker_sha == result.merge_commit
        assert len(marker_sha) == 40

    async def test_returns_none_when_branch_still_exists(self, git_ops: GitOps):
        """find_merge_marker returns None when the branch ref still exists,
        even if there happens to be a merge commit matching the pattern.

        resolve_branch_sha gates the git-log search: if the branch is still
        present, is_ancestor is the authoritative check.
        """
        tid = 'still-here'
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        # Branch created but NOT merged — still exists

        result = await git_ops.find_merge_marker(f'task/{tid}')

        assert result is None

    async def test_returns_none_when_branch_never_existed_no_marker(
        self, git_ops: GitOps
    ):
        """find_merge_marker returns None when no such branch was ever created
        and no merge commit matching the pattern exists on main.
        """
        result = await git_ops.find_merge_marker('task/never-existed')
        assert result is None

    async def test_substring_safety_task_1_does_not_match_task_10(
        self, git_ops: GitOps
    ):
        """Substring safety: merging task/10 writes 'Merge task/10 into main'.
        find_merge_marker('task/1') must NOT match this commit.

        The '^' anchor + trailing ' into ' in the --grep pattern prevent
        task/1 from being found inside 'Merge task/10 into main'.
        """
        # Merge task/10 and delete branch
        tid = '10'
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        (wt_info.path / f'task_{tid}.py').write_text(f'task_{tid} = True\n')
        await git_ops.commit(wt_info.path, f'Add task {tid}')

        result = await git_ops.merge_to_main(wt_info.path, tid)
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None

        adv = await git_ops.advance_main(result.merge_commit)
        assert adv == 'advanced'

        await git_ops.cleanup_merge_worktree(result.merge_worktree)
        await git_ops.cleanup_worktree(wt_info.path, tid)

        # task/10 branch is deleted; merge marker 'Merge task/10 into main' exists
        # find_merge_marker('task/1') must NOT find it
        marker_sha = await git_ops.find_merge_marker('task/1')
        assert marker_sha is None

    async def test_returns_none_when_branch_deleted_without_merging(
        self, git_ops: GitOps
    ):
        """Branch was created and abandoned: deleted without ever being merged.

        This is subtly different from 'branch never existed' (case c): the
        branch ref existed at some point but was cleaned up without writing a
        merge commit on main.  find_merge_marker must return None because there
        is no matching marker subject to find.
        """
        tid = 'abandoned-1'
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        (wt_info.path / f'{tid}.py').write_text(f'{tid} = True\n')
        await git_ops.commit(wt_info.path, f'Add {tid}')

        # Delete the worktree and branch WITHOUT merging to main
        await git_ops.cleanup_worktree(wt_info.path, tid)

        # Branch is gone but no merge marker was ever written on main
        result = await git_ops.find_merge_marker(f'task/{tid}')
        assert result is None

    async def test_returns_single_sha_when_branch_reopened_with_two_merges(
        self, git_ops: GitOps
    ):
        """find_merge_marker returns exactly one 40-char SHA in the re-opened-task
        scenario described in the function's own docstring: a task branch is merged,
        deleted, then re-created under the same name, merged again, and deleted again.
        Both merge commits share the same subject ('Merge task/reopened-1 into main'),
        so a git-log invocation with conflicting --max-count=1 and -n 5000 flags would
        return both SHAs newline-joined (last-wins: -n 5000 overrides --max-count=1),
        corrupting done_provenance={'commit': marker_sha} in harness reconcile.
        After dropping -n 5000, --max-count=1 alone ensures a single SHA is returned.
        """
        tid = 'reopened-1'

        # --- Iteration 1 ---
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        (wt_info.path / f'iter1_{tid}.py').write_text(f'iter1_{tid} = True\n')
        await git_ops.commit(wt_info.path, f'Add iter1')

        result1 = await git_ops.merge_to_main(wt_info.path, tid)
        assert result1.success
        assert result1.merge_commit is not None
        assert result1.merge_worktree is not None

        adv1 = await git_ops.advance_main(result1.merge_commit)
        assert adv1 == 'advanced'

        await git_ops.cleanup_merge_worktree(result1.merge_worktree)
        await git_ops.cleanup_worktree(wt_info.path, tid)
        first_sha = result1.merge_commit

        # --- Iteration 2 (same tid — branch was deleted, so this is a fresh branch) ---
        wt_info2 = await git_ops.create_worktree(tid)
        assert wt_info2 is not None
        (wt_info2.path / f'iter2_{tid}.py').write_text(f'iter2_{tid} = True\n')
        await git_ops.commit(wt_info2.path, f'Add iter2')

        result2 = await git_ops.merge_to_main(wt_info2.path, tid)
        assert result2.success
        assert result2.merge_commit is not None
        assert result2.merge_worktree is not None

        adv2 = await git_ops.advance_main(result2.merge_commit)
        assert adv2 == 'advanced'

        await git_ops.cleanup_merge_worktree(result2.merge_worktree)
        await git_ops.cleanup_worktree(wt_info2.path, tid)
        second_sha = result2.merge_commit

        # Both merge commits exist on main with the same subject.
        assert first_sha != second_sha  # sanity: two distinct commits

        # --- Assertion ---
        marker_sha = await git_ops.find_merge_marker(f'task/{tid}')

        assert marker_sha is not None
        assert '\n' not in marker_sha   # anti-multiline regression
        assert len(marker_sha) == 40    # single-SHA shape
        assert marker_sha == second_sha  # most-recent first (reverse chrono + --max-count=1)
