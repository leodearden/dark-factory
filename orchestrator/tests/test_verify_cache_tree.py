"""Verify-cache reads of a dispatched git tree and its uv workspace graph (task 5652)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import git_env_with_ceiling

from orchestrator.verify_cache import tree_objects, workspace

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
    @pytest.mark.parametrize('ch', ['\x0b', '\x0c', '\x1c', '\x85', '\u2028'])
    async def test_an_absent_path_holding_a_non_newline_line_break_is_one_answer(self, repo, tree, ch):
        absent = f'nope/a{ch}b.py'

        ids = await tree_objects.object_ids(repo, tree, ['tools/check.py', absent])

        assert ids == {'tools/check.py': _git(repo, 'rev-parse', f'{tree}:tools/check.py'), absent: None}

    @pytest.mark.asyncio
    async def test_blob_text_reads_a_file_and_is_none_for_an_absent_path(self, repo, tree):
        assert await tree_objects.blob_text(repo, tree, 'tools/check.py') == 'print("check")'
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

    @pytest.mark.asyncio
    async def test_directories_lists_every_directory_recursively(self, repo, tree):
        assert set(await tree_objects.directories(repo, tree)) == {
            'alpha', 'alpha/src', 'alpha/src/alpha', 'beta', 'docs', 'tools',
        }


def _member(name: str, dependencies: tuple[str, ...] = (), groups: str = '') -> str:
    deps = ', '.join(f'"{dep}"' for dep in dependencies)
    return f'[project]\nname = "{name}"\nversion = "0"\ndependencies = [{deps}]\n{groups}'


def _root(*members: str, exclude: tuple[str, ...] = ()) -> str:
    listed = ', '.join(f'"{member}"' for member in members)
    excluded = ', '.join(f'"{member}"' for member in exclude)
    return f'[tool.uv.workspace]\nmembers = [{listed}]\nexclude = [{excluded}]\n'


_GRAPH_MEMBERS = {
    'alpha': _member('Alpha', ('click>=8',)),
    'beta': _member('beta', ("alpha[x]>=1; python_version>'3'",)),
    'gamma': _member(
        'gamma', groups='[dependency-groups]\ndev = [{include-group = "lint"}]\nlint = ["beta"]\n',
    ),
    'fused': _member('Fused_Memory'),
    'epsilon': _member('epsilon', groups='[dependency-groups]\ndev = ["fused.memory"]\n'),
}


def _graph() -> workspace.WorkspaceGraph:
    return workspace.WorkspaceGraph.from_manifests(_root(*_GRAPH_MEMBERS), _GRAPH_MEMBERS)


class TestWorkspaceGraph:
    def test_members_are_the_listed_workspace_members(self):
        assert _graph().members == frozenset(_GRAPH_MEMBERS)

    def test_a_member_name_is_pep_503_normalized(self):
        assert _graph().names['fused'] == 'fused-memory'

    def test_a_dependency_on_a_member_is_an_edge_whatever_its_specifier(self):
        assert _graph().closure({'beta'}) == {'beta', 'alpha'}

    def test_a_dependency_outside_the_workspace_is_no_edge(self):
        assert _graph().closure({'alpha'}) == {'alpha'}

    def test_a_dev_group_include_group_is_followed_transitively(self):
        assert _graph().closure({'gamma'}) == {'gamma', 'beta', 'alpha'}

    def test_a_dev_group_dependency_matches_by_normalized_name(self):
        assert _graph().closure({'epsilon'}) == {'epsilon', 'fused'}

    def test_an_include_group_cycle_terminates(self):
        cyclic = {
            'alpha': _member('alpha'),
            'beta': _member(
                'beta',
                groups=(
                    '[dependency-groups]\n'
                    'dev = [{include-group = "lint"}, "alpha"]\n'
                    'lint = [{include-group = "dev"}]\n'
                ),
            ),
        }
        graph = workspace.WorkspaceGraph.from_manifests(_root('alpha', 'beta'), cyclic)

        assert graph.closure({'beta'}) == {'beta', 'alpha'}

    def test_members_containing_matches_whole_path_segments(self):
        assert _graph().members_containing(
            {'beta/src/x.py', 'tests/scripts', 'alpha', 'alphabet/x.py'},
        ) == {'beta', 'alpha'}

    def test_a_root_without_a_workspace_is_an_empty_graph(self):
        graph = workspace.WorkspaceGraph.from_manifests('[project]\nname = "solo"\n', {})

        assert graph.members == frozenset()
        assert graph.closure({'alpha'}) == frozenset()

    def test_a_listed_member_without_a_pyproject_is_an_error_naming_it(self):
        with pytest.raises(ValueError, match='beta'):
            workspace.WorkspaceGraph.from_manifests(_root('alpha', 'beta'), {'alpha': _member('alpha')})

    @pytest.mark.asyncio
    async def test_read_workspace_reads_the_dispatched_tree(self, repo, tree):
        graph = await workspace.read_workspace(repo, tree)

        assert graph.closure({'beta'}) == {'alpha', 'beta'}

    @pytest.mark.asyncio
    async def test_an_unparseable_member_pyproject_is_a_tree_read_error_naming_the_member(self, repo):
        (repo / 'beta' / 'pyproject.toml').write_text('[project\nname = "beta"\n')

        with pytest.raises(tree_objects.TreeReadError, match="member 'beta'"):
            await workspace.read_workspace(repo, _write_tree(repo))

    @pytest.mark.asyncio
    async def test_a_glob_member_expands_against_the_trees_directories(self, tmp_path):
        globbed = seed_repo(tmp_path / 'globbed', RepoSeed(
            files=(
                ('pyproject.toml', _root('pkgs/*', exclude=('pkgs/skip',))),
                ('pkgs/one/pyproject.toml', _member('one')),
                ('pkgs/one/sub/deeper/x.py', 'X = 1\n'),
                ('pkgs/two/pyproject.toml', _member('two', ('one',))),
                ('pkgs/skip/README.md', '# not a member\n'),
            ),
            message='globbed workspace',
        ))
        graph = await workspace.read_workspace(globbed, _git(globbed, 'rev-parse', 'HEAD^{tree}'))

        assert graph.members == {'pkgs/one', 'pkgs/two'}
        assert graph.closure({'pkgs/two'}) == {'pkgs/one', 'pkgs/two'}

    @pytest.mark.asyncio
    async def test_a_tree_without_a_root_pyproject_is_an_empty_graph(self, tmp_path):
        bare = seed_repo(tmp_path / 'bare', RepoSeed(files=(('README.md', '# x\n'),), message='bare'))

        graph = await workspace.read_workspace(bare, _git(bare, 'rev-parse', 'HEAD^{tree}'))

        assert graph.members == frozenset()
