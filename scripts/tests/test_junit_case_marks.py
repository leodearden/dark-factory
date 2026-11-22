"""Contract of scripts/junit_case_marks.py, on a real two-commit git repo."""

from __future__ import annotations

import subprocess
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from junit_case_marks import CaseMarks, Unresolved
from shared.testing_timeout_markers import TimeoutSite
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from verify_log_junits import CaseOutcome, JunitCase

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_UNMARKED_MODULE = '''
    import pytest

    def test_alpha() -> None:
        pass

    class TestScan:
        def test_sweep(self) -> None:
            pass
    '''

_MARKED_MODULE = '''
    import pytest
    from shared.testing_ceiling_anchors import MAGICMOCK_REPO_SCAN_TEST_TIMEOUT
    from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

    pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

    def test_alpha() -> None:
        pass

    @pytest.mark.timeout(MAGICMOCK_REPO_SCAN_TEST_TIMEOUT)
    class TestScan:
        def test_sweep(self) -> None:
            pass
    '''

_ROOT_MODULE = '''
    import pytest

    @pytest.mark.timeout(300)
    def test_root() -> None:
        pass
    '''


@dataclass(frozen=True)
class _Repo:
    root: Path
    unmarked: str
    marked: str


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(root), '-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(root: Path, files: dict[str, str], message: str) -> str:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding='utf-8')
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '--no-verify', '-m', message)
    return _git(root, 'rev-parse', 'HEAD')


@pytest.fixture
def repo(tmp_path: Path) -> _Repo:
    root = tmp_path / 'repo'
    root.mkdir()
    _git(root, 'init', '-q')
    unmarked = _commit(root, {'pkg/tests/test_m.py': _UNMARKED_MODULE}, 'A')
    marked = _commit(
        root, {'pkg/tests/test_m.py': _MARKED_MODULE, 'tests/scripts/test_r.py': _ROOT_MODULE}, 'B'
    )
    return _Repo(root, unmarked, marked)


@pytest.fixture
def marks(repo: _Repo) -> Iterator[CaseMarks]:
    with CaseMarks(repo.root) as case_marks:
        yield case_marks


def _case(classname: str, name: str) -> JunitCase:
    return JunitCase(classname, name, 99.0, CaseOutcome.PASSED)


def _constant(answer: TimeoutSite | Unresolved | None) -> str | None:
    assert isinstance(answer, TimeoutSite), answer
    return answer.constant


def test_the_same_row_resolves_per_tree(repo: _Repo, marks: CaseMarks) -> None:
    row = _case('tests.test_m', 'test_alpha')

    assert marks.closest(repo.unmarked, 'pkg', row) is None
    assert _constant(marks.closest(repo.marked, 'pkg', row)) == 'WHOLE_TREE_SCAN_TEST_TIMEOUT'


def test_a_class_test_resolves_to_its_class_mark(repo: _Repo, marks: CaseMarks) -> None:
    row = _case('tests.test_m.TestScan', 'test_sweep')

    assert _constant(marks.closest(repo.marked, 'pkg', row)) == 'MAGICMOCK_REPO_SCAN_TEST_TIMEOUT'
    assert marks.closest(repo.unmarked, 'pkg', row) is None


@pytest.mark.parametrize('name', ['test_sweep@bare_magicmock_repo_scan', 'test_sweep[case-1]', 'test_sweep[a.b]@group'])
def test_group_and_param_suffixes_resolve_to_the_same_function(
    repo: _Repo, marks: CaseMarks, name: str
) -> None:
    answer = marks.closest(repo.marked, 'pkg', _case('tests.test_m.TestScan', name))

    assert _constant(answer) == 'MAGICMOCK_REPO_SCAN_TEST_TIMEOUT'


def test_a_root_relative_classname_resolves_when_the_leg_dir_has_no_such_module(
    repo: _Repo, marks: CaseMarks
) -> None:
    answer = marks.closest(repo.marked, 'pkg', _case('tests.scripts.test_r', 'test_root'))

    assert isinstance(answer, TimeoutSite)
    assert answer.spelling == '300'


def test_a_classname_naming_no_file_at_the_tree_is_unresolved(repo: _Repo, marks: CaseMarks) -> None:
    answer = marks.closest(repo.unmarked, 'pkg', _case('tests.scripts.test_r', 'test_root'))

    assert isinstance(answer, Unresolved)
    assert answer.reason


def test_an_unknown_sha_is_unresolved(marks: CaseMarks) -> None:
    answer = marks.closest('0' * 40, 'pkg', _case('tests.test_m', 'test_alpha'))

    assert isinstance(answer, Unresolved)
    assert answer.reason


def test_commit_exists_answers_over_the_same_process(repo: _Repo, marks: CaseMarks) -> None:
    assert marks.commit_exists(repo.marked)
    assert marks.commit_exists(repo.unmarked)
    assert not marks.commit_exists('0' * 40)
    assert not marks.commit_exists('not-a-sha')
