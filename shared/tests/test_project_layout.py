"""Tests for shared.project_layout: where a project's stores live under its root.

The literal path values are the contract (plans/review-all-s14-shared-leaves-prd.md):
every locator must return exactly the spelling its current call sites build.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import shared.project_layout
from shared.project_layout import (
    ESCALATION_QUEUE_RELPATH,
    KNOWN_PROJECT_ROOTS_ENV,
    CheckoutRootError,
    agent_transcripts_dir,
    default_project_root,
    escalation_queue_dir,
    known_project_roots,
    main_checkout_of,
    orchestrator_data_dir,
    park_eviction_requests_db,
    runs_db,
    scheduler_overrides_db,
    scheduler_state_json,
    tasks_db,
)

_ORCHESTRATOR_STORES = (
    (runs_db, 'runs.db'),
    (scheduler_state_json, 'scheduler_state.json'),
    (scheduler_overrides_db, 'scheduler_overrides.db'),
    (park_eviction_requests_db, 'park_eviction_requests.db'),
)


@pytest.fixture(params=['str', 'path'])
def root(request: pytest.FixtureRequest, tmp_path: Path) -> str | Path:
    project = tmp_path / 'project'
    return str(project) if request.param == 'str' else project


class TestStoreLocators:
    def test_escalation_queue_dir(self, root: str | Path) -> None:
        assert escalation_queue_dir(root) == Path(root) / 'data' / 'escalations'
        assert escalation_queue_dir(root) == Path(root) / ESCALATION_QUEUE_RELPATH
        assert ESCALATION_QUEUE_RELPATH == 'data/escalations'

    def test_orchestrator_data_dir(self, root: str | Path) -> None:
        assert orchestrator_data_dir(root) == Path(root) / 'data' / 'orchestrator'

    def test_agent_transcripts_dir(self, root: str | Path) -> None:
        assert agent_transcripts_dir(root) == Path(root) / 'data' / 'orchestrator' / 'agent-transcripts'

    @pytest.mark.parametrize(('locator', 'name'), _ORCHESTRATOR_STORES)
    def test_orchestrator_store(self, root: str | Path, locator, name: str) -> None:
        assert locator(root) == Path(root) / 'data' / 'orchestrator' / name

    def test_tasks_db_is_the_real_store_not_the_decoy(self, root: str | Path) -> None:
        assert tasks_db(root) == Path(root) / '.taskmaster' / 'tasks' / 'tasks.db'

    @pytest.mark.parametrize(
        'locator',
        [
            escalation_queue_dir,
            orchestrator_data_dir,
            *(locator for locator, _ in _ORCHESTRATOR_STORES),
            tasks_db,
        ],
    )
    def test_returns_a_path(self, root: str | Path, locator) -> None:
        assert isinstance(locator(root), Path)

    @pytest.mark.parametrize(('locator', 'name'), _ORCHESTRATOR_STORES)
    def test_orchestrator_stores_share_one_parent(
        self, root: str | Path, locator, name: str
    ) -> None:
        assert locator(root).parent == orchestrator_data_dir(root)


class TestLocatorsArePure:
    @pytest.mark.parametrize(
        'locator',
        [
            escalation_queue_dir,
            orchestrator_data_dir,
            *(locator for locator, _ in _ORCHESTRATOR_STORES),
            tasks_db,
        ],
    )
    def test_no_io_on_a_missing_root(self, tmp_path: Path, locator) -> None:
        missing = tmp_path / 'absent'
        locator(missing)
        assert not missing.exists()
        assert list(tmp_path.iterdir()) == []

    def test_relative_root_stays_relative(self) -> None:
        assert runs_db('proj') == Path('proj/data/orchestrator/runs.db')
        assert tasks_db('proj') == Path('proj/.taskmaster/tasks/tasks.db')
        assert escalation_queue_dir('proj') == Path('proj/data/escalations')


class TestKnownProjectRoots:
    def test_env_var_name(self) -> None:
        assert KNOWN_PROJECT_ROOTS_ENV == 'DASHBOARD_KNOWN_PROJECT_ROOTS'

    def test_strips_entries_and_drops_empties(self) -> None:
        environ = {KNOWN_PROJECT_ROOTS_ENV: ' /a , ,/b,, '}
        assert known_project_roots(environ) == (Path('/a'), Path('/b'))

    @pytest.mark.parametrize('environ', [{}, {KNOWN_PROJECT_ROOTS_ENV: ''}, {KNOWN_PROJECT_ROOTS_ENV: ' , '}])
    def test_unset_or_blank_is_empty(self, environ: dict[str, str]) -> None:
        assert known_project_roots(environ) == ()

    def test_order_and_duplicates_are_preserved(self) -> None:
        environ = {KNOWN_PROJECT_ROOTS_ENV: '/c,/a,/c,/b'}
        assert known_project_roots(environ) == (Path('/c'), Path('/a'), Path('/c'), Path('/b'))

    def test_returns_a_tuple(self) -> None:
        assert isinstance(known_project_roots({KNOWN_PROJECT_ROOTS_ENV: '/a'}), tuple)

    def test_default_reads_the_live_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, '/x,/y')
        assert known_project_roots() == (Path('/x'), Path('/y'))


# default_project_root() answers from where its module FILE lives, so the honest
# fixture moves the file: the local package is copied into a temp checkout and
# asked from a child interpreter. `-I` keeps an inherited PYTHONPATH or
# VIRTUAL_ENV from supplying another tree's copy; the __file__ assert proves
# which copy answered.
_CHILD = """
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import shared.project_layout as layout

assert Path(layout.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()), layout.__file__
print(layout.default_project_root())
"""


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout


def _git_main_checkout(checkout: Path) -> Path:
    common_dir = _git(checkout, 'rev-parse', '--path-format=absolute', '--git-common-dir')
    return Path(common_dir.strip()).parent.resolve()


def _answer_from(dest_src: Path) -> tuple[int, str, str]:
    package = Path(shared.project_layout.__file__).resolve().parent
    shutil.copytree(package, dest_src / 'shared', ignore=shutil.ignore_patterns('__pycache__'))
    proc = subprocess.run(
        [sys.executable, '-I', '-c', _CHILD, str(dest_src)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _resolved_answer(dest_src: Path) -> Path:
    returncode, stdout, stderr = _answer_from(dest_src)
    assert returncode == 0, f'child exited {returncode}\n--- stderr ---\n{stderr}'
    return Path(stdout.strip())


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return tmp_path.resolve()


@pytest.fixture
def main_repo(base: Path) -> Path:
    main = base / 'main'
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(main)], check=True, capture_output=True)
    _git(main, 'config', 'user.email', 'project-layout-test@example.com')
    _git(main, 'config', 'user.name', 'Project Layout Test')
    _git(main, 'config', 'commit.gpgsign', 'false')
    _git(main, 'commit', '-q', '--allow-empty', '--no-verify', '-m', 'seed')
    return main


def _add_worktree(main: Path, worktree: Path, branch: str) -> Path:
    _git(main, 'worktree', 'add', '-q', str(worktree), '-b', branch)
    return worktree


def _gitdir_of_worktree(worktree: Path) -> Path:
    text = (worktree / '.git').read_text()
    assert text.startswith('gitdir: '), text
    return Path(text.removeprefix('gitdir: ').strip())


def _src_of(checkout: Path) -> Path:
    src = checkout / 'src'
    src.mkdir(parents=True, exist_ok=True)
    return src


class TestMainCheckoutOf:
    def test_main_checkout_answers_itself(self, main_repo: Path) -> None:
        src = _src_of(main_repo)
        assert main_checkout_of(src) == _git_main_checkout(src) == main_repo

    def test_nested_linked_worktree_answers_the_main_checkout(self, main_repo: Path) -> None:
        src = _src_of(_add_worktree(main_repo, main_repo / '.worktrees' / 'wt', 'wt'))
        assert main_checkout_of(src) == _git_main_checkout(src) == main_repo

    def test_linked_worktree_outside_the_main_tree(self, base: Path, main_repo: Path) -> None:
        src = _src_of(_add_worktree(main_repo, base / 'elsewhere' / 'wt2', 'wt2'))
        assert main_checkout_of(str(src)) == _git_main_checkout(src) == main_repo

    def test_relative_gitdir_in_the_worktree_file(self, main_repo: Path) -> None:
        worktree = _add_worktree(main_repo, main_repo / '.worktrees' / 'wt', 'wt')
        relative = os.path.relpath(_gitdir_of_worktree(worktree).resolve(), worktree)
        (worktree / '.git').write_text(f'gitdir: {relative}\n')
        src = _src_of(worktree)
        assert main_checkout_of(src) == _git_main_checkout(src) == main_repo

    def test_no_enclosing_checkout_is_refused(self, base: Path) -> None:
        src = base / 'plain' / 'src'
        assert not any((ancestor / '.git').exists() for ancestor in src.parents), (
            f'fixture premise: no ancestor of {src} may hold .git'
        )
        with pytest.raises(CheckoutRootError, match=re.escape(str(src))):
            main_checkout_of(src)

    def test_missing_gitdir_is_refused_rather_than_answering_the_worktree(
        self, base: Path, main_repo: Path
    ) -> None:
        worktree = _add_worktree(main_repo, main_repo / '.worktrees' / 'wt', 'wt')
        nowhere = base / 'nowhere'
        (worktree / '.git').write_text(f'gitdir: {nowhere}\n')
        with pytest.raises(CheckoutRootError, match=re.escape(str(nowhere))):
            main_checkout_of(_src_of(worktree))

    def test_bare_backed_worktree_is_refused(self, base: Path, main_repo: Path) -> None:
        bare = base / 'bare.git'
        subprocess.run(
            ['git', 'clone', '-q', '--bare', str(main_repo), str(bare)], check=True, capture_output=True
        )
        src = _src_of(_add_worktree(bare, base / 'bare-wt', 'bwt'))
        common_dir = _git(src, 'rev-parse', '--path-format=absolute', '--git-common-dir').strip()
        assert Path(common_dir) == bare, 'fixture premise: the bare repo is the common git dir'
        with pytest.raises(CheckoutRootError, match=re.escape(str(bare))):
            main_checkout_of(src)

    @pytest.mark.parametrize('pointer', ['gitfile', 'commondir'])
    def test_undecodable_pointer_file_is_refused_naming_it(
        self, main_repo: Path, pointer: str
    ) -> None:
        worktree = _add_worktree(main_repo, main_repo / '.worktrees' / 'wt', 'wt')
        dot_git = worktree / '.git'
        target = dot_git if pointer == 'gitfile' else _gitdir_of_worktree(worktree) / 'commondir'
        target.write_bytes(b'gitdir: \xff\xfe\n' if pointer == 'gitfile' else b'\xff\xfe\n')
        with pytest.raises(CheckoutRootError) as refused:
            main_checkout_of(_src_of(worktree))
        assert str(target) in str(refused.value)
        assert isinstance(refused.value.__cause__, UnicodeDecodeError)

    @pytest.mark.skipif(os.geteuid() == 0, reason='root reads a mode-000 file')
    def test_unreadable_gitfile_is_refused_naming_it(self, main_repo: Path) -> None:
        worktree = _add_worktree(main_repo, main_repo / '.worktrees' / 'wt', 'wt')
        src = _src_of(worktree)
        dot_git = worktree / '.git'
        dot_git.chmod(0)
        try:
            with pytest.raises(CheckoutRootError) as refused:
                main_checkout_of(src)
        finally:
            dot_git.chmod(0o644)
        assert str(dot_git) in str(refused.value)
        assert isinstance(refused.value.__cause__, PermissionError)


class TestDefaultProjectRoot:
    def test_main_checkout_answers_itself(self, main_repo: Path) -> None:
        src = main_repo / 'src'
        answer = _resolved_answer(src)
        assert answer == _git_main_checkout(src) == main_repo

    def test_nested_linked_worktree_answers_the_main_checkout(self, main_repo: Path) -> None:
        worktree = _add_worktree(main_repo, main_repo / '.worktrees' / 'wt', 'wt')
        src = worktree / 'src'
        answer = _resolved_answer(src)
        assert answer == _git_main_checkout(src) == main_repo

    def test_no_enclosing_checkout_fails_loudly(self, base: Path) -> None:
        src = base / 'plain' / 'src'
        assert not any((ancestor / '.git').exists() for ancestor in src.parents), (
            f'fixture premise: no ancestor of {src} may hold .git'
        )
        returncode, _, stderr = _answer_from(src)
        assert returncode != 0
        assert 'CheckoutRootError' in stderr
        assert str(src / 'shared' / 'project_layout.py') in stderr

    def test_real_tree_answer_matches_git(self) -> None:
        module_dir = Path(shared.project_layout.__file__).resolve().parent
        answer = default_project_root()
        assert answer == _git_main_checkout(module_dir)
        assert (answer / '.git').is_dir()

    def test_error_is_a_runtime_error_not_an_os_error(self) -> None:
        assert issubclass(CheckoutRootError, RuntimeError)
        assert not issubclass(CheckoutRootError, OSError)
