"""Tests for orchestrator.review_working_tree: the uncommitted state of a review's checkout."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import git_env_with_ceiling
from df_pytest_isolation import git_redirect_env

from orchestrator.review_working_tree import WorkingTreeState, read_working_tree_state


def _git(cwd: Path, *args: str) -> None:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    subprocess.run(['git', *args], cwd=cwd, env=env, check=True, capture_output=True, text=True)


def _write(root: Path, relpath: str, text: str) -> None:
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


_MIXED_SEED = RepoSeed(
    files=(
        ('scripts/ok.py', 'x = 1\n'),
        ('keep.py', 'k = 1\n'),
        ('gone.py', 'g = 1\n'),
        ('.gitignore', 'build/\n'),
    ),
    message='init',
)


@pytest.mark.asyncio
class TestReadWorkingTreeState:
    async def test_clean_repo_is_clean(self, tmp_path: Path) -> None:
        repo = seed_repo(tmp_path / 'repo')

        state = await read_working_tree_state(repo)

        assert state is not None
        assert state.is_clean
        assert state.untracked == ()
        assert state.uncommitted == ()

    async def test_mixed_dirty_repo_lists_untracked_and_uncommitted_sorted(
        self, tmp_path: Path,
    ) -> None:
        repo = seed_repo(tmp_path / 'repo', _MIXED_SEED)
        _write(repo, 'scripts/bad.py', 'import os\n')
        _write(repo, 'build/out.py', 'ignored = True\n')
        _write(repo, 'notes dir/a b.py', 'spaced = True\n')
        _write(repo, 'keep.py', 'k = 2\n')
        _write(repo, 'staged.py', 's = 1\n')
        _git(repo, 'add', 'staged.py')
        (repo / 'gone.py').unlink()

        state = await read_working_tree_state(repo)

        assert state is not None
        assert not state.is_clean
        assert state.untracked == ('notes dir/a b.py', 'scripts/bad.py')
        assert state.uncommitted == ('gone.py', 'keep.py', 'staged.py')

    async def test_non_repo_directory_returns_none(self, tmp_path: Path) -> None:
        plain = tmp_path / 'plain'
        plain.mkdir()

        assert await read_working_tree_state(plain) is None

    async def test_missing_directory_returns_none(self, tmp_path: Path) -> None:
        assert await read_working_tree_state(tmp_path / 'does-not-exist') is None


class TestWorkingTreeStateInvariant:
    def test_path_in_both_tuples_is_rejected_by_name(self) -> None:
        with pytest.raises(ValueError, match='a.py'):
            WorkingTreeState(untracked=('a.py',), uncommitted=('a.py',))

    @pytest.mark.parametrize('field', ['untracked', 'uncommitted'])
    def test_absolute_path_is_rejected_by_name(self, field: str) -> None:
        with pytest.raises(ValueError, match='/x.py'):
            WorkingTreeState(**{'untracked': (), 'uncommitted': (), field: ('/x.py',)})
