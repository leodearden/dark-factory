"""Tests for resolve_main_checkout() — maps any worktree/subdir path to the main checkout."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from fused_memory.models import scope
from fused_memory.models.scope import resolve_main_checkout


@pytest.fixture(autouse=True)
def clear_resolver_cache():
    """Prevent cache bleed between tests."""
    scope._MAIN_CHECKOUT_CACHE.clear()
    yield
    scope._MAIN_CHECKOUT_CACHE.clear()


def _init_repo(path: Path) -> Path:
    """Create a git repo with one commit; return resolved repo path."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-q'], cwd=path, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=path, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=path, check=True)
    subprocess.run(['git', 'config', 'commit.gpgsign', 'false'], cwd=path, check=True)
    (path / 'README.md').write_text('seed\n')
    subprocess.run(['git', 'add', 'README.md'], cwd=path, check=True)
    subprocess.run(
        ['git', 'commit', '-q', '--no-verify', '-m', 'init'],
        cwd=path, check=True,
    )
    return path.resolve()


def _add_worktree(main: Path, wt_dir: Path, branch: str) -> Path:
    """Add a worktree at wt_dir on a new branch; return resolved worktree path."""
    wt_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['git', 'worktree', 'add', '-q', '-b', branch, str(wt_dir)],
        cwd=main, check=True,
    )
    return wt_dir.resolve()


class TestResolveMainCheckout:
    def test_resolves_main_path_to_itself(self, tmp_path):
        main = _init_repo(tmp_path / 'repo')
        assert resolve_main_checkout(str(main)) == str(main)

    def test_resolves_subdir_inside_main(self, tmp_path):
        main = _init_repo(tmp_path / 'repo')
        subdir = main / 'a' / 'b'
        subdir.mkdir(parents=True)
        assert resolve_main_checkout(str(subdir)) == str(main)

    def test_resolves_worktree_path_to_main(self, tmp_path):
        main = _init_repo(tmp_path / 'repo')
        wt = _add_worktree(main, tmp_path / 'wt1', 'feature')
        assert resolve_main_checkout(str(wt)) == str(main)

    def test_resolves_subdir_inside_worktree(self, tmp_path):
        main = _init_repo(tmp_path / 'repo')
        wt = _add_worktree(main, tmp_path / 'wt2', 'feature2')
        subdir = wt / 'nested' / 'deep'
        subdir.mkdir(parents=True)
        assert resolve_main_checkout(str(subdir)) == str(main)

    def test_raises_on_non_git_path(self, tmp_path):
        bare = tmp_path / 'not-a-repo'
        bare.mkdir()
        with pytest.raises(ValueError):
            resolve_main_checkout(str(bare))

    def test_caches_results(self, tmp_path):
        main = _init_repo(tmp_path / 'repo')
        real_run = subprocess.run
        call_count = 0

        def counting_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return real_run(*args, **kwargs)

        with patch('fused_memory.models.scope.subprocess.run', side_effect=counting_run):
            first = resolve_main_checkout(str(main))
            second = resolve_main_checkout(str(main))
        assert first == second == str(main)
        assert call_count == 1
