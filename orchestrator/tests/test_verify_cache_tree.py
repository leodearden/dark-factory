"""Verify-cache reads of a dispatched git tree and its uv workspace graph (task 5652)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import git_env_with_ceiling

from orchestrator.verify_cache import tree_objects

_ROOT_PYPROJECT = """\
[project]
name = "fixture-root"
version = "0"

[tool.uv.workspace]
members = ["alpha", "beta"]
"""

_ALPHA_PYPROJECT = """\
[project]
name = "alpha"
version = "0"
dependencies = ["click"]
"""

_BETA_PYPROJECT = """\
[project]
name = "beta"
version = "0"
dependencies = ["alpha"]
"""

_WORKSPACE_SEED = RepoSeed(
    files=(
        ('pyproject.toml', _ROOT_PYPROJECT),
        ('uv.lock', 'version = 1\n'),
        ('conftest.py', '# root conftest\n'),
        ('alpha/pyproject.toml', _ALPHA_PYPROJECT),
        ('alpha/src/alpha/__init__.py', 'VALUE = 1\n'),
        ('beta/pyproject.toml', _BETA_PYPROJECT),
        ('tools/check.py', 'print("check")\n'),
        ('docs/notes.md', '# notes\n'),
    ),
    message='workspace fixture',
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=repo, env=git_env_with_ceiling(repo),
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _write_tree(repo: Path) -> str:
    _git(repo, 'add', '-A')
    return _git(repo, 'write-tree')


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo', _WORKSPACE_SEED)


@pytest.fixture
def tree(repo: Path) -> str:
    return _git(repo, 'rev-parse', 'HEAD^{tree}')


class TestTreeObjects:
    @pytest.mark.asyncio
    async def test_tree_of_resolves_a_rev_and_is_idempotent_on_a_tree(self, repo, tree):
        assert await tree_objects.tree_of(repo, 'HEAD') == tree
        assert await tree_objects.tree_of(repo, tree) == tree

    @pytest.mark.asyncio
    async def test_root_blobs_are_the_sorted_root_level_files_only(self, repo, tree):
        blobs = await tree_objects.root_blobs(repo, tree)

        assert [name for name, _oid in blobs] == ['conftest.py', 'pyproject.toml', 'uv.lock']
        assert dict(blobs)['uv.lock'] == _git(repo, 'rev-parse', f'{tree}:uv.lock')

    @pytest.mark.asyncio
    async def test_object_ids_answers_trees_blobs_and_absences(self, repo, tree):
        ids = await tree_objects.object_ids(repo, tree, ['alpha', 'tools/check.py', 'nope/x.py'])

        assert ids == {
            'alpha': _git(repo, 'rev-parse', f'{tree}:alpha'),
            'tools/check.py': _git(repo, 'rev-parse', f'{tree}:tools/check.py'),
            'nope/x.py': None,
        }

    @pytest.mark.asyncio
    async def test_blob_text_reads_a_file_and_is_none_for_an_absent_path(self, repo, tree):
        assert await tree_objects.blob_text(repo, tree, 'tools/check.py') == 'print("check")\n'
        assert await tree_objects.blob_text(repo, tree, 'tools/absent.py') is None

    @pytest.mark.asyncio
    async def test_an_unknown_tree_is_an_error_naming_the_git_command(self, repo):
        with pytest.raises(tree_objects.TreeReadError, match='git'):
            await tree_objects.root_blobs(repo, '0' * 40)

    @pytest.mark.asyncio
    async def test_an_edit_moves_its_directory_and_the_root_but_not_a_sibling(self, repo, tree):
        before = await tree_objects.object_ids(repo, tree, ['alpha', 'beta'])
        (repo / 'alpha' / 'src' / 'alpha' / '__init__.py').write_text('VALUE = 2\n')
        edited = _write_tree(repo)

        after = await tree_objects.object_ids(repo, edited, ['alpha', 'beta'])

        assert edited != tree
        assert after['alpha'] != before['alpha']
        assert after['beta'] == before['beta']
