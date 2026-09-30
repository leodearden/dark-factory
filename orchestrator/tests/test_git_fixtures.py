"""Contract tests for ``_git_fixtures``: a copied template repo is indistinguishable
from a freshly built one, to porcelain AND plumbing, and costs one git spawn.
"""
from __future__ import annotations

import ast
import asyncio
import os
import shlex
import shutil
import subprocess
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import README_SEED, RepoSeed, RepoTemplates, build_repo, seed_repo
from _orch_helpers import git_env_with_ceiling
from _workflow_helpers import _init_git_repo, _init_repo, _init_transcript_repo


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', *args], cwd=repo, env=git_env_with_ceiling(repo),
        check=False, capture_output=True, text=True,
    )


def _out(repo: Path, *args: str) -> str:
    result = _git(repo, *args)
    assert result.returncode == 0, f'git {args} failed: {result.stderr}'
    return result.stdout.strip()


def _worktree_files(repo: Path) -> dict[str, str]:
    return {
        path.relative_to(repo).as_posix(): path.read_text()
        for path in repo.rglob('*')
        if path.is_file() and '.git' not in path.relative_to(repo).parts
    }


def _tracked_paths(repo: Path) -> set[str]:
    return set(_out(repo, 'ls-tree', '-r', '--name-only', 'HEAD').splitlines())


def _repo_facts(repo: Path) -> dict[str, object]:
    return {
        'head': _out(repo, 'symbolic-ref', 'HEAD'),
        'local_config': sorted(_out(repo, 'config', '--local', '--list').splitlines()),
        'commit_count': _out(repo, 'rev-list', '--count', 'HEAD'),
        'last_commit': _out(repo, 'log', '-1', '--format=%s%n%an%n%ae%n%cn%n%ce'),
        'tree': _out(repo, 'rev-parse', 'HEAD^{tree}'),
        'remotes': _out(repo, 'remote'),
        'files': _worktree_files(repo),
    }


def _install_git_spawn_counter(monkeypatch: pytest.MonkeyPatch, where: Path) -> Path:
    """Put a counting ``git`` shim first on PATH; return the log it appends to."""
    real_git = shutil.which('git')
    assert real_git is not None
    bin_dir = where / 'bin'
    bin_dir.mkdir(parents=True)
    log = where / 'git-spawns.log'
    log.touch()
    shim = bin_dir / 'git'
    shim.write_text(
        '#!/bin/sh\n'
        f'printf "x\\n" >> {shlex.quote(str(log))}\n'
        f'exec {shlex.quote(real_git)} "$@"\n'
    )
    shim.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    return log


def _spawns(log: Path) -> int:
    return len(log.read_text().splitlines())


@pytest.fixture
def templates(tmp_path: Path) -> RepoTemplates:
    return RepoTemplates(tmp_path / 'templates')


def test_a_copy_matches_a_fresh_build(templates: RepoTemplates, tmp_path: Path) -> None:
    copied = templates.seed(tmp_path / 'a', README_SEED)
    built = build_repo(tmp_path / 'b', README_SEED)

    assert copied == tmp_path / 'a'
    assert _repo_facts(copied) == _repo_facts(built)
    facts = _repo_facts(copied)
    assert facts['head'] == 'refs/heads/main'
    assert facts['commit_count'] == '1'
    assert facts['last_commit'] == 'Initial commit\nTest\ntest@test.com\nTest\ntest@test.com'
    assert facts['remotes'] == ''
    assert facts['files'] == {'README.md': '# Test\n'}


def test_a_copy_is_stat_clean_to_plumbing(templates: RepoTemplates, tmp_path: Path) -> None:
    repo = templates.seed(tmp_path / 'repo', README_SEED)

    assert _git(repo, 'diff-files', '--quiet').returncode == 0
    assert _git(repo, 'diff-index', '--quiet', 'HEAD').returncode == 0
    assert _out(repo, 'status', '--porcelain') == ''


def test_a_copy_carries_no_template_path(templates: RepoTemplates, tmp_path: Path) -> None:
    repo = templates.seed(tmp_path / 'repo', README_SEED)
    needles = {str(templates.root).encode(), str(templates.root.resolve()).encode()}

    leaks = [
        path.relative_to(repo).as_posix()
        for path in (repo / '.git').rglob('*')
        if path.is_file() and any(needle in path.read_bytes() for needle in needles)
    ]
    assert leaks == []


def test_mutating_a_copy_does_not_leak_into_the_next(
    templates: RepoTemplates, tmp_path: Path,
) -> None:
    first = templates.seed(tmp_path / 'first', README_SEED)
    initial_sha = _out(first, 'rev-parse', 'HEAD')
    (first / 'new.txt').write_text('new\n')
    _out(first, 'add', 'new.txt')
    _out(first, 'commit', '-m', 'mutation')
    _out(first, 'branch', 'side')
    _out(first, 'worktree', 'add', '../wt', '-b', 'wtb')
    hook = first / '.git' / 'hooks' / 'pre-commit'
    hook.write_text('#!/bin/sh\nexit 1\n')
    hook.chmod(0o755)
    _out(first, 'config', 'foo.bar', 'x')

    second = templates.seed(tmp_path / 'second', README_SEED)

    assert _out(second, 'rev-parse', 'HEAD') == initial_sha
    assert _out(second, 'branch', '--format=%(refname:short)').splitlines() == ['main']
    worktrees = _out(second, 'worktree', 'list', '--porcelain').splitlines()
    assert len([line for line in worktrees if line.startswith('worktree ')]) == 1
    assert not (second / '.git' / 'hooks' / 'pre-commit').exists()
    assert _git(second, 'config', 'foo.bar').returncode != 0


def test_distinct_seeds_get_distinct_contents(
    templates: RepoTemplates, tmp_path: Path,
) -> None:
    other_seed = RepoSeed(files=(('src/a.py', 'A = 1\n'),), message='other')

    other = templates.seed(tmp_path / 'other', other_seed)
    readme = templates.seed(tmp_path / 'readme', README_SEED)

    assert _tracked_paths(other) == {'src/a.py'}
    assert _worktree_files(other) == {'src/a.py': 'A = 1\n'}
    assert _out(other, 'log', '-1', '--format=%s') == 'other'
    assert _tracked_paths(readme) == {'README.md'}
    assert _worktree_files(readme) == {'README.md': '# Test\n'}
    assert _out(readme, 'log', '-1', '--format=%s') == 'Initial commit'


def test_a_non_empty_destination_commits_its_existing_files(
    templates: RepoTemplates, tmp_path: Path,
) -> None:
    dest = tmp_path / 'populated'
    dest.mkdir()
    (dest / 'extra.txt').write_text('already here\n')

    repo = templates.seed(dest, README_SEED)

    assert _tracked_paths(repo) == {'README.md', 'extra.txt'}
    assert _out(repo, 'rev-list', '--count', 'HEAD') == '1'
    assert _out(repo, 'log', '-1', '--format=%s') == 'Initial commit'


def test_a_copy_costs_exactly_one_git_spawn(
    templates: RepoTemplates, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    templates.seed(tmp_path / 'warm', README_SEED)
    log = _install_git_spawn_counter(monkeypatch, tmp_path / 'shim')

    templates.seed(tmp_path / 'measured', README_SEED)

    assert _spawns(log) == 1


def test_a_non_empty_destination_costs_a_full_build(
    templates: RepoTemplates, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    populated = tmp_path / 'populated'
    populated.mkdir()
    (populated / 'extra.txt').write_text('already here\n')
    log = _install_git_spawn_counter(monkeypatch, tmp_path / 'shim')

    build_repo(tmp_path / 'reference', README_SEED)
    full_build_cost = _spawns(log)
    templates.seed(populated, README_SEED)
    in_place_cost = _spawns(log) - full_build_cost

    assert full_build_cost > 1
    assert in_place_cost == full_build_cost


def test_seed_repo_uses_the_session_templates(tmp_path: Path) -> None:
    repo = seed_repo(tmp_path / 'repo')

    assert repo == tmp_path / 'repo'
    assert (repo / 'README.md').read_text() == '# Test\n'
    assert _out(repo, 'symbolic-ref', 'HEAD') == 'refs/heads/main'
    assert _out(repo, 'log', '-1', '--format=%s') == 'Initial commit'
    assert _out(repo, 'config', 'user.email') == 'test@test.com'


def test_session_templates_live_under_this_runs_basetemp(
    pristine_repo_templates: RepoTemplates, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    basetemp = tmp_path_factory.getbasetemp().resolve()
    assert pristine_repo_templates.root.resolve().is_relative_to(basetemp)


Seeder = Callable[[Path], Coroutine[Any, Any, None]]


@pytest.mark.parametrize(('seeder', 'files', 'subject'), [
    pytest.param(_init_git_repo, {'README.md': '# Test\n'}, 'init', id='_init_git_repo'),
    pytest.param(
        _init_repo,
        {
            'lib.py': 'def greet(name: str) -> str:\n    return f"Hello, {name}"\n',
            'test_lib.py': (
                'from lib import greet\n\ndef test_greet():\n'
                '    assert greet("world") == "Hello, world"\n'
            ),
        },
        'Initial commit',
        id='_init_repo',
    ),
    pytest.param(
        _init_transcript_repo,
        {'lib.py': 'def greet(name): return name\n'},
        'Initial commit',
        id='_init_transcript_repo',
    ),
])
class TestSharedWorkflowSeeders:
    def test_seeds_the_legacy_contract(
        self, seeder: Seeder, files: dict[str, str], subject: str, tmp_path: Path,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()

        asyncio.run(seeder(repo))

        assert _tracked_paths(repo) == set(files)
        assert _worktree_files(repo) == files
        assert _out(repo, 'log', '-1', '--format=%s') == subject
        assert _out(repo, 'rev-list', '--count', 'HEAD') == '1'
        assert _out(repo, 'symbolic-ref', 'HEAD') == 'refs/heads/main'
        assert _out(repo, 'config', 'user.email') == 'test@test.com'
        assert _out(repo, 'config', 'user.name') == 'Test'
        assert _git(repo, 'diff-files', '--quiet').returncode == 0

    def test_a_warm_call_costs_exactly_one_git_spawn(
        self, seeder: Seeder, files: dict[str, str], subject: str,
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / 'warm').mkdir()
        asyncio.run(seeder(tmp_path / 'warm'))
        log = _install_git_spawn_counter(monkeypatch, tmp_path / 'shim')
        (tmp_path / 'measured').mkdir()

        asyncio.run(seeder(tmp_path / 'measured'))

        assert _spawns(log) == 1


_MIGRATED_MODULES = tuple(
    Path(__file__).parent / name
    for name in (
        'test_git_ops.py', 'test_merge_queue.py', 'test_warm_lane_pool.py', '_workflow_helpers.py',
    )
)


def _git_argvs(func: ast.AST) -> list[list[object]]:
    """Every ``['git', <verb>, ...]`` list literal under *func*; non-constant elements are None."""
    argvs: list[list[object]] = []
    for node in ast.walk(func):
        if isinstance(node, ast.List):
            argv: list[object] = [
                elt.value if isinstance(elt, ast.Constant) else None for elt in node.elts
            ]
            if len(argv) >= 2 and argv[0] == 'git' and isinstance(argv[1], str):
                argvs.append(argv)
    return argvs


def _local_repo_seeders(tree: ast.Module) -> list[str]:
    """Module-level functions that both ``git init`` a non-bare repo and ``git commit``."""
    seeders: list[str] = []
    for func in tree.body:
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            argvs = _git_argvs(func)
            if (
                any(argv[1] == 'init' and '--bare' not in argv for argv in argvs)
                and any(argv[1] == 'commit' for argv in argvs)
            ):
                seeders.append(func.name)
    return seeders


class TestNoLocalRepoSeederInMigratedModules:
    @pytest.mark.parametrize('module', _MIGRATED_MODULES, ids=[m.name for m in _MIGRATED_MODULES])
    def test_module_has_no_local_repo_seeder(self, module: Path) -> None:
        seeders = _local_repo_seeders(ast.parse(module.read_text(encoding='utf-8')))
        assert seeders == [], (
            f'{module.name} defines its own repo seeder(s) {seeders}: delegate to '
            '_git_fixtures.seed_repo instead. When migrating another module, add it '
            'to _MIGRATED_MODULES in test_git_fixtures.py.'
        )

    def test_the_legacy_seeder_is_flagged(self) -> None:
        tree = ast.parse(
            'async def _setup_repo(repo):\n'
            "    await _run(['git', 'init', '-b', 'main'], cwd=repo)\n"
            "    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)\n"
            "    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)\n"
            "    (repo / 'README.md').write_text('# Test\\n')\n"
            "    await _run(['git', 'add', '-A'], cwd=repo)\n"
            "    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)\n"
        )

        assert _local_repo_seeders(tree) == ['_setup_repo']

    def test_a_bare_origin_helper_is_not_flagged(self) -> None:
        tree = ast.parse(
            'async def _make_origin(origin, seed):\n'
            "    await _run(['git', 'init', '--bare', '-b', 'main'], cwd=origin)\n"
            "    await _run(['git', 'push', str(origin), 'main'], cwd=seed)\n"
        )

        assert _local_repo_seeders(tree) == []

    def test_a_clone_then_commit_helper_is_not_flagged(self) -> None:
        tree = ast.parse(
            'async def _clone_and_commit(origin, local):\n'
            "    await _run(['git', 'clone', str(origin), str(local)])\n"
            "    await _run(['git', 'commit', '--allow-empty', '-m', 'x'], cwd=local)\n"
        )

        assert _local_repo_seeders(tree) == []
