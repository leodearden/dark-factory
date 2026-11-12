"""Contract of shared/src/shared/testing_repo_wide_tests.py: the repo-wide detector and its census."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from shared.testing_repo_wide_tests import (
    RepoWideFloors,
    RepoWideSignal,
    RepoWideVerdict,
    judge_repo_wide_family,
    repo_wide_signals,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_NO_FLOORS = RepoWideFloors(test_files=0, repo_wide_modules=0)

_IMPORT_LINE = 'from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT'
_MARK_LINE = 'pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)'


def _signals(source: str) -> frozenset[RepoWideSignal]:
    return repo_wide_signals(ast.parse(textwrap.dedent(source)))


# ---------------------------------------------------------------------------
# repo_wide_signals -- one positive and one negative per signal.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('source', [
    pytest.param("root.rglob('*.json')\n", id='rglob-any-argument'),
    pytest.param('import os\nos.walk(root)\n', id='os-walk'),
    pytest.param("root.glob('**/fixtures')\n", id='glob-double-star'),
    pytest.param("root.glob('*.py')\n", id='glob-star-py'),
    pytest.param("root.glob('src/*_test.py')\n", id='glob-nested-star-py'),
    pytest.param("command = 'ls-files'\n", id='ls-files-literal'),
])
def test_tree_walk_is_detected(source: str) -> None:
    assert RepoWideSignal.TREE_WALK in _signals(source)


@pytest.mark.parametrize('source', [
    pytest.param("root.glob('conftest.py')\n", id='literal-filename'),
    pytest.param("root.glob('*/src/*')\n", id='non-py-wildcard'),
    pytest.param('root.glob(pattern)\n', id='non-literal-pattern'),
    pytest.param("walk(root)\n", id='bare-walk-is-not-os-walk'),
    pytest.param("command = 'ls-files --others'\n", id='ls-files-inside-a-longer-string'),
])
def test_tree_walk_is_not_detected(source: str) -> None:
    assert RepoWideSignal.TREE_WALK not in _signals(source)


@pytest.mark.parametrize('source', [
    pytest.param("command = ['git', 'status']\n", id='list-literal'),
    pytest.param("command = ('git', 'log')\n", id='tuple-literal'),
    pytest.param("run('git', 'status')\n", id='first-positional-argument'),
])
def test_git_subprocess_is_detected(source: str) -> None:
    assert RepoWideSignal.GIT_SUBPROCESS in _signals(source)


@pytest.mark.parametrize('source', [
    pytest.param("command = ['ls', 'git']\n", id='git-not-first-element'),
    pytest.param("command = ['gitk', 'status']\n", id='other-program'),
    pytest.param("run(program, 'git')\n", id='git-not-first-argument'),
])
def test_git_subprocess_is_not_detected(source: str) -> None:
    assert RepoWideSignal.GIT_SUBPROCESS not in _signals(source)


@pytest.mark.parametrize('source', [
    pytest.param('root = Path(__file__).resolve().parents[2]\n', id='parents-subscript'),
    pytest.param('print(PROJECT_ROOT)\n', id='project-root-name'),
    pytest.param('_REPO_ROOT = here\n', id='underscored-repo-root'),
    pytest.param('print(layout.REPO_ROOT)\n', id='repo-root-attribute'),
])
def test_repo_anchor_is_detected(source: str) -> None:
    assert RepoWideSignal.REPO_ANCHOR in _signals(source)


@pytest.mark.parametrize('source', [
    pytest.param('here = Path(__file__).resolve().parent\n', id='parent-not-parents'),
    pytest.param('every = path.parents\n', id='parents-not-subscripted'),
    pytest.param('print(MY_REPO_ROOT)\n', id='prefixed-name'),
    pytest.param('print(REPO_ROOTS)\n', id='suffixed-name'),
])
def test_repo_anchor_is_not_detected(source: str) -> None:
    assert RepoWideSignal.REPO_ANCHOR not in _signals(source)


@pytest.mark.parametrize('word', [
    'CONTRIBUTING.md', 'CLAUDE.md', 'ARCHITECTURE.md', 'OPERATIONS.md', 'README',
    'codebook', 'design-invariants', 'code-quality',
])
def test_committed_doc_name_is_detected(word: str) -> None:
    assert _signals(f'name = {f"docs/{word}"!r}\n') == {RepoWideSignal.COMMITTED_DOC_NAME}


def test_committed_doc_name_is_not_detected() -> None:
    assert RepoWideSignal.COMMITTED_DOC_NAME not in _signals("name = 'notes/readme-later.md'\n")


@pytest.mark.parametrize('word', [
    'dark-factory-orchestrator.yaml', 'defaults.yaml', 'orchestrator.yaml',
    'capability-manifest', '.service', 'systemd',
])
def test_committed_config_name_is_detected(word: str) -> None:
    assert _signals(f'name = {f"cfg/{word}"!r}\n') == {RepoWideSignal.COMMITTED_CONFIG_NAME}


@pytest.mark.parametrize('name', [
    pytest.param('pyproject.toml', id='other-config'),
    pytest.param('fused_memory.services.write_journal', id='extension-inside-a-dotted-path'),
])
def test_committed_config_name_is_not_detected(name: str) -> None:
    assert RepoWideSignal.COMMITTED_CONFIG_NAME not in _signals(f'name = {name!r}\n')


_EVERY_WORD = (
    "rglob('*.py') os.walk glob('**') ls-files git parents[2] PROJECT_ROOT REPO_ROOT "
    'CONTRIBUTING.md CLAUDE.md ARCHITECTURE.md OPERATIONS.md README codebook '
    'design-invariants code-quality dark-factory-orchestrator.yaml defaults.yaml '
    'orchestrator.yaml capability-manifest x.service systemd'
)


def test_docstrings_mentioning_every_vocabulary_word_yield_no_signals() -> None:
    source = (
        f'"""{_EVERY_WORD}"""\n'
        'class TestThing:\n'
        f'    """{_EVERY_WORD}"""\n'
        '    def test_method(self) -> None:\n'
        f'        """{_EVERY_WORD}"""\n'
        'def test_function() -> None:\n'
        f'    """{_EVERY_WORD}"""\n'
        'async def test_coroutine() -> None:\n'
        f'    """{_EVERY_WORD}"""\n'
    )
    assert _signals(source) == frozenset()


def test_a_comment_only_mention_yields_no_signals() -> None:
    assert _signals(f'# {_EVERY_WORD}\nvalue = 1\n') == frozenset()


@pytest.mark.parametrize('source', [
    pytest.param(f'assert ok, {_EVERY_WORD!r}\n', id='assert-message'),
    pytest.param("assert ok, f'see CLAUDE.md about {why}'\n", id='assert-f-string-message'),
    pytest.param("assert ok, 'see ' + 'OPERATIONS.md'\n", id='assert-concatenated-message'),
    pytest.param(
        "with pytest.raises(ValueError, match='orchestrator.yaml'):\n    pass\n", id='match-keyword',
    ),
    pytest.param("check(ok, msg='the README says so')\n", id='msg-keyword'),
    pytest.param("pytest.skip(reason='needs x.service')\n", id='reason-keyword'),
])
def test_prose_naming_a_committed_file_yields_no_signals(source: str) -> None:
    assert _signals(source) == frozenset()


@pytest.mark.parametrize(('source', 'signal'), [
    pytest.param(
        "assert (root / 'CLAUDE.md').exists(), 'missing'\n",
        RepoWideSignal.COMMITTED_DOC_NAME,
        id='asserted-expression',
    ),
    pytest.param(
        "load(path='orchestrator.yaml')\n", RepoWideSignal.COMMITTED_CONFIG_NAME, id='other-keyword',
    ),
])
def test_a_committed_name_outside_prose_still_counts(source: str, signal: RepoWideSignal) -> None:
    assert _signals(source) == {signal}


def test_an_f_string_piece_counts() -> None:
    assert _signals("path = f'{root}/CLAUDE.md'\n") == {RepoWideSignal.COMMITTED_DOC_NAME}


def test_a_plain_unit_test_module_yields_no_signals() -> None:
    assert _signals(
        """
        import pytest

        from shared.streak import Streak


        def test_streak_counts() -> None:
            assert Streak().bump().count == 1
        """
    ) == frozenset()


# ---------------------------------------------------------------------------
# judge_repo_wide_family -- the census over a tests directory.
# ---------------------------------------------------------------------------

_REPO_WIDE_BODY = (
    '\n\n'
    'def test_reads_the_tree() -> None:\n'
    "    assert list(Path('.').rglob('*.py'))\n"
)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def _module(*mark_lines: str) -> str:
    return 'from pathlib import Path\n\nimport pytest\n' + ''.join(
        f'{line}\n' for line in mark_lines
    ) + _REPO_WIDE_BODY


def _judge(root: Path, floors: RepoWideFloors = _NO_FLOORS) -> RepoWideVerdict:
    return judge_repo_wide_family(root, floors)


def test_an_unmarked_repo_wide_test_module_is_an_offender(tmp_path: Path) -> None:
    _write(tmp_path, 'sub/test_x.py', _module())

    verdict = _judge(tmp_path)

    assert [offender.module for offender in verdict.offenders] == ['sub/test_x.py']
    failure = verdict.offender_failure
    assert failure is not None
    assert 'sub/test_x.py' in failure
    assert str(RepoWideSignal.TREE_WALK) in failure
    assert _IMPORT_LINE in failure
    assert _MARK_LINE in failure


@pytest.mark.parametrize('mark_lines', [
    pytest.param((_IMPORT_LINE, _MARK_LINE), id='alone'),
    pytest.param(
        (_IMPORT_LINE, 'pytestmark = [pytest.mark.slow, pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)]'),
        id='inside-a-list',
    ),
])
def test_a_house_marked_repo_wide_module_is_not_an_offender(
    tmp_path: Path, mark_lines: tuple[str, ...],
) -> None:
    _write(tmp_path, 'test_x.py', _module(*mark_lines))

    verdict = _judge(tmp_path)

    assert [module.module for module in verdict.repo_wide] == ['test_x.py']
    assert verdict.offenders == ()
    assert verdict.offender_failure is None


def test_a_literal_timeout_at_the_ceiling_is_still_an_offender(tmp_path: Path) -> None:
    _write(tmp_path, 'test_x.py', _module('pytestmark = pytest.mark.timeout(1200)'))

    assert [offender.module for offender in _judge(tmp_path).offenders] == ['test_x.py']


@pytest.mark.parametrize('relative', ['_helpers.py', 'conftest.py', 'sub/repo_scan.py'])
def test_a_non_collectable_repo_wide_module_is_ignored(tmp_path: Path, relative: str) -> None:
    _write(tmp_path, relative, _module())

    verdict = _judge(tmp_path)

    assert verdict.repo_wide == ()
    assert verdict.offender_failure is None


def test_the_suffix_spelling_of_a_test_module_is_collectable(tmp_path: Path) -> None:
    _write(tmp_path, 'scan_test.py', _module())

    assert [offender.module for offender in _judge(tmp_path).offenders] == ['scan_test.py']


def test_a_non_utf8_file_is_reported_unreadable_not_raised(tmp_path: Path) -> None:
    (tmp_path / 'test_bad.py').write_bytes(b'\xff\xfe root.rglob("*.py")\n')

    verdict = _judge(tmp_path)

    assert verdict.unreadable == ('test_bad.py',)
    assert verdict.examined == 0


def test_a_syntax_error_file_is_examined_but_never_flagged(tmp_path: Path) -> None:
    _write(tmp_path, 'test_broken.py', "root.rglob('*.py')\ndef f(:\n")

    verdict = _judge(tmp_path)

    assert verdict.examined == 1
    assert verdict.repo_wide == ()
    assert verdict.offender_failure is None


@pytest.fixture
def two_file_tree(tmp_path: Path) -> Path:
    _write(tmp_path, 'test_wide.py', _module(_IMPORT_LINE, _MARK_LINE))
    _write(tmp_path, 'test_unit.py', 'def test_one() -> None:\n    assert 1\n')
    return tmp_path


@pytest.mark.parametrize('floors', [
    pytest.param(RepoWideFloors(test_files=3, repo_wide_modules=1), id='file-floor-above'),
    pytest.param(RepoWideFloors(test_files=2, repo_wide_modules=2), id='module-floor-above'),
])
def test_floors_above_the_counts_fail_the_census_naming_both_counts(
    two_file_tree: Path, floors: RepoWideFloors,
) -> None:
    failure = _judge(two_file_tree, floors).census_failure

    assert failure is not None
    assert '2 .py files' in failure
    assert '1 repo-wide test module' in failure


@pytest.mark.parametrize('floors', [
    pytest.param(RepoWideFloors(test_files=2, repo_wide_modules=1), id='at'),
    pytest.param(RepoWideFloors(test_files=1, repo_wide_modules=0), id='below'),
])
def test_floors_at_or_below_the_counts_pass_the_census(
    two_file_tree: Path, floors: RepoWideFloors,
) -> None:
    assert _judge(two_file_tree, floors).census_failure is None


def test_a_sweep_below_its_file_floor_cannot_report_a_clean_tree(two_file_tree: Path) -> None:
    verdict = _judge(two_file_tree, RepoWideFloors(test_files=3, repo_wide_modules=0))

    assert verdict.offenders == ()
    assert verdict.offender_failure == verdict.census_failure
    assert verdict.offender_failure is not None
