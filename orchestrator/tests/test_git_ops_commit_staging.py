"""Real-git tests for how ``GitOps.commit`` stages embedded git repositories (task 3389).

Pins that a WIP-save commit never introduces a gitlink for an embedded repo,
whatever its directory name, while a registered submodule's pointer bump still
lands.  Git is real because the behaviour under test is git's own staging.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps


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


def _git(cwd: Path, *args: str) -> str:
    assert_isolated_git_repo(cwd)
    proc = subprocess.run(
        ['git', *args], cwd=cwd, env=git_env_with_ceiling(cwd),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f'git {" ".join(args)} failed (rc={proc.returncode}): {proc.stderr.strip()}'
    )
    return proc.stdout.strip()


def _plant_embedded_repo(worktree: Path, rel: str, *, with_commit: bool) -> Path:
    assert_isolated_git_repo(worktree)
    target = worktree / rel
    target.mkdir(parents=True)
    proc = subprocess.run(
        ['git', 'init', '-b', 'main'], cwd=target, env=git_env_with_ceiling(target),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f'git init failed: {proc.stderr.strip()}'
    if with_commit:
        _git(target, 'config', 'user.email', 'test@test.com')
        _git(target, 'config', 'user.name', 'Test')
        (target / 'f.txt').write_text('scratch\n')
        _git(target, 'add', 'f.txt')
        _git(target, 'commit', '-m', 'scratch commit')
    return target


def _committed_entries(cwd: Path, sha: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for record in _git(cwd, 'ls-tree', '-r', '-z', sha).split('\0'):
        if not record:
            continue
        meta, path = record.split('\t', 1)
        entries[path] = meta.split()[0]
    return entries


def _gitlink_paths(entries: dict[str, str]) -> list[str]:
    return [path for path, mode in entries.items() if mode == '160000']


@pytest.mark.asyncio
class TestCommitEmbeddedRepoGitlinks:
    async def test_embedded_repo_is_not_committed_as_gitlink(
        self, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ) -> None:
        wt = (await git_ops.create_worktree('gitlink-a')).path
        (wt / 'work.py').write_text('x = 1\n')
        _plant_embedded_repo(wt, 'scratch/t0', with_commit=True)

        with caplog.at_level(logging.WARNING, logger='orchestrator.staged_gitlinks'):
            sha = await git_ops.commit(wt, 'wip')

        assert isinstance(sha, str) and len(sha) == 40
        entries = _committed_entries(wt, sha)
        assert 'work.py' in entries
        assert _gitlink_paths(entries) == []
        assert (wt / 'scratch/t0' / '.git').is_dir()
        assert any(
            record.levelno >= logging.WARNING and 'scratch/t0' in record.getMessage()
            for record in caplog.records
        )

    async def test_gitlink_staged_before_commit_is_also_dropped(
        self, git_ops: GitOps,
    ) -> None:
        wt = (await git_ops.create_worktree('gitlink-b')).path
        _plant_embedded_repo(wt, 'scratch/t0', with_commit=True)
        _git(wt, 'add', 'scratch/t0')
        (wt / 'work.py').write_text('x = 1\n')

        sha = await git_ops.commit(wt, 'wip')

        assert sha is not None
        entries = _committed_entries(wt, sha)
        assert 'work.py' in entries
        assert _gitlink_paths(entries) == []

    async def test_unstaging_a_glob_named_gitlink_keeps_its_lookalike_sibling(
        self, git_ops: GitOps,
    ) -> None:
        wt = (await git_ops.create_worktree('gitlink-glob')).path
        _plant_embedded_repo(wt, 'scratch/[t]0', with_commit=True)
        (wt / 'scratch/t0').write_text('real work\n')

        sha = await git_ops.commit(wt, 'wip')

        assert sha is not None
        entries = _committed_entries(wt, sha)
        assert 'scratch/t0' in entries
        assert _gitlink_paths(entries) == []

    async def test_non_utf8_staged_path_fails_closed(self, git_ops: GitOps) -> None:
        wt = (await git_ops.create_worktree('non-utf8')).path
        (wt / os.fsdecode(b'\xe9.txt')).write_text('x\n')
        head_before = _git(wt, 'rev-parse', 'HEAD')

        with pytest.raises(RuntimeError, match='not valid UTF-8') as excinfo:
            await git_ops.commit(wt, 'wip')

        assert str(wt) in str(excinfo.value)
        assert _git(wt, 'rev-parse', 'HEAD') == head_before

    async def test_embedded_repo_as_only_change_commits_nothing(
        self, git_ops: GitOps,
    ) -> None:
        wt = (await git_ops.create_worktree('gitlink-only')).path
        _plant_embedded_repo(wt, 'scratch/t0', with_commit=True)
        head_before = _git(wt, 'rev-parse', 'HEAD')

        sha = await git_ops.commit(wt, 'wip')

        assert sha is None
        assert _git(wt, 'rev-parse', 'HEAD') == head_before
        assert _git(wt, 'diff', '--cached', '--name-only') == ''

    async def test_commitless_embedded_repo_does_not_abort_staging(
        self, git_ops: GitOps,
    ) -> None:
        wt = (await git_ops.create_worktree('commitless')).path
        (wt / 'work.py').write_text('x = 1\n')
        _plant_embedded_repo(wt, 'scratch/t1', with_commit=False)
        head_before = _git(wt, 'rev-parse', 'HEAD')

        sha = await git_ops.commit(wt, 'wip')

        assert sha is not None
        assert sha != head_before
        entries = _committed_entries(wt, sha)
        assert 'work.py' in entries
        assert [path for path in entries if path.startswith('scratch/')] == []

    async def test_registered_submodule_pointer_bump_is_still_committed(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        old = _git(git_repo, 'rev-parse', 'HEAD')
        _git(git_repo, 'update-index', '--add', '--cacheinfo', f'160000,{old},vendor/sub')
        (git_repo / '.gitmodules').write_text(
            '[submodule "vendor/sub"]\n\tpath = vendor/sub\n\turl = ./vendor/sub\n'
        )
        _git(git_repo, 'add', '.gitmodules')
        _git(git_repo, 'commit', '-m', 'register vendor/sub')
        new = _git(git_repo, 'rev-parse', 'HEAD')

        wt = (await git_ops.create_worktree('bump')).path
        _git(wt, 'update-index', '--cacheinfo', f'160000,{new},vendor/sub')
        (wt / 'work.py').write_text('x = 1\n')

        sha = await git_ops.commit(wt, 'wip')

        assert sha is not None
        assert _committed_entries(wt, sha)['vendor/sub'] == '160000'
        assert _git(wt, 'rev-parse', f'{sha}:vendor/sub') == new
