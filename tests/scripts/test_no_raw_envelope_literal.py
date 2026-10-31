"""Tests of the raw envelope-literal guard, and the guard's whole-repo live check.

The rule, its population and its needles are defined in
``tests/scripts/raw_envelope_literal_guard.py``; this file tests them and holds
``test_no_markup_handling_file_spells_a_raw_envelope_literal``, the live check
over every file git lists.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
import raw_envelope_literal_guard
from git_listing import git, listed_files
from raw_envelope_literal_guard import (
    ESCAPED_BRACKET,
    LITERAL_OWNER,
    RAW_PREFIXES,
    guarded_sources,
    handles_envelope_markup,
    main,
    raw_literal_violations,
    raw_prefix_lines,
    report,
)
from shared.python_env import without_python_env_selection
from shared.toolcall_markup import ENVELOPE_LITERALS

REPO_ROOT = Path(__file__).parents[2]

_LT = chr(60)


@pytest.fixture(scope='module')
def repo_population() -> dict[str, str]:
    return guarded_sources(REPO_ROOT)


def test_no_markup_handling_file_spells_a_raw_envelope_literal(
    repo_population: dict[str, str],
) -> None:
    violations = raw_literal_violations(repo_population)
    assert not violations, report(violations)


def test_this_guard_is_inside_its_own_population(repo_population: dict[str, str]) -> None:
    guard_files = {
        Path(raw_envelope_literal_guard.__file__).resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        Path(__file__).resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
    }
    missing = sorted(guard_files - repo_population.keys())
    assert not missing, (
        f'{missing} spell chr(60), so git discovery plus the population predicate must '
        'select them; if they do not, the repo-wide check above is passing vacuously.'
    )


def test_every_envelope_literal_starts_with_a_scanned_prefix() -> None:
    unscanned = [
        literal.replace(_LT, ESCAPED_BRACKET)
        for literal in ENVELOPE_LITERALS
        if not literal.startswith(RAW_PREFIXES)
    ]
    assert not unscanned, (
        f'{LITERAL_OWNER}.ENVELOPE_LITERALS gained literal(s) {unscanned} that start '
        f'with none of the scanned prefixes '
        f'{[prefix.replace(_LT, ESCAPED_BRACKET) for prefix in RAW_PREFIXES]}; '
        'widen RAW_PREFIXES so the repo-wide guard still sees them.'
    )


def test_raw_prefix_lines_reports_each_prefix_under_its_escaped_spelling() -> None:
    source = (
        'a = 1\n'
        'b = "' + _LT + '/content>"\n'
        'c = "' + _LT + 'parameter name=\'x\'>"\n'
        'd = "' + _LT + '/invoke>"\n'
    )
    hits = raw_prefix_lines(source)
    assert hits == {'\\x3c/': (2, 4), '\\x3cparameter': (3,)}


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('b = "\\x3c/content>"\nc = "\\x3cparameter name=\'x\'>"\n', id='escaped'),
        pytest.param('ok = a ' + _LT + ' b\n', id='bare-comparison'),
        pytest.param('page = "' + _LT + 'div>"\n', id='html-opener'),
    ],
)
def test_raw_prefix_lines_ignores_brackets_that_open_no_envelope_literal(source: str) -> None:
    hits = raw_prefix_lines(source)
    assert hits == {}


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('LT = chr(60)\n', id='chr-60-text'),
        pytest.param('ESC = "\\x3c"\n', id='escape-text'),
        pytest.param('from shared.toolcall_markup import closer_for\n', id='from-owner-import'),
        pytest.param('from shared import toolcall_markup\n', id='from-package-import-owner'),
        pytest.param('import shared.toolcall_markup as markup\n', id='import-owner'),
        pytest.param('def broken(:\n    pass  # toolcall_markup\n', id='unparsable-fails-closed'),
    ],
)
def test_handles_envelope_markup_selects_markup_handling_sources(source: str) -> None:
    assert handles_envelope_markup(source)


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('x = 1\n', id='plain-code'),
        pytest.param(
            '"""Built on shared.toolcall_markup."""\n# see shared.toolcall_markup\nx = 1\n',
            id='owner-named-only-in-prose',
        ),
        pytest.param('import toolcall_markup_corpus_extract\n', id='lookalike-module'),
    ],
)
def test_handles_envelope_markup_rejects_other_sources(source: str) -> None:
    assert not handles_envelope_markup(source)


def _plant(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    """A git repo planting one file per discovery case; staged, never committed."""
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    _plant(root, '.gitignore', 'ignored/\n')
    _plant(root, 'tracked_violator.py', 'LT = chr(60)\nX = "' + _LT + '/content>"\n')
    _plant(root, 'html_fixture.py', 'PAGE = "' + _LT + 'p>hi' + _LT + '/p>"\n')
    _plant(root, 'clean_handler.py', 'from shared.toolcall_markup import closer_for\n')
    _plant(root, 'deleted.py', 'LT = chr(60)\n')
    _plant(root, 'notes.txt', 'chr(60) ' + _LT + '/content>\n')
    git(
        root, 'add', '.gitignore', 'tracked_violator.py', 'html_fixture.py',
        'clean_handler.py', 'deleted.py', 'notes.txt',
    )
    (root / 'deleted.py').unlink()
    _plant(
        root, 'pkg/untracked_violator.py',
        'ESC = "\\x3c"\n\n\nY = "' + _LT + 'parameter name=\'a\'>"\n',
    )
    _plant(root, 'ignored/handler.py', 'LT = chr(60)\nX = "' + _LT + '/content>"\n')
    return root


def test_discovery_lists_existing_tracked_and_unignored_python_only(
    synthetic_repo: Path,
) -> None:
    expected = [
        synthetic_repo / 'clean_handler.py',
        synthetic_repo / 'html_fixture.py',
        synthetic_repo / 'pkg' / 'untracked_violator.py',
        synthetic_repo / 'tracked_violator.py',
    ]
    assert sorted(listed_files(synthetic_repo, '*.py')) == expected


def test_raw_literal_violations_reports_tracked_and_untracked_violators(
    synthetic_repo: Path,
) -> None:
    violations = raw_literal_violations(guarded_sources(synthetic_repo))
    assert violations == {
        'tracked_violator.py': {'\\x3c/': (2,)},
        'pkg/untracked_violator.py': {'\\x3cparameter': (4,)},
    }


@pytest.fixture
def scoped_repo(synthetic_repo: Path) -> Path:
    """``synthetic_repo`` plus the files the CLI's directory scoping turns on."""
    _plant(synthetic_repo, 'pkg/sub/nested_violator.py', 'LT = chr(60)\nX = "' + _LT + '/invoke>"\n')
    _plant(synthetic_repo, 'pkgextra/violator.py', 'LT = chr(60)\n\nX = "' + _LT + '/content>"\n')
    _plant(synthetic_repo, 'clean/clean_handler.py', 'from shared.toolcall_markup import closer_for\n')
    _plant(synthetic_repo, 'clean/html_fixture.py', 'PAGE = "' + _LT + 'p>hi' + _LT + '/p>"\n')
    return synthetic_repo


PKG_REPORT_LINES = (
    "  pkg/sub/nested_violator.py: \\x3c/ on line(s) [2]",
    "  pkg/untracked_violator.py: \\x3cparameter on line(s) [4]",
)
PKGEXTRA_REPORT_LINE = "  pkgextra/violator.py: \\x3c/ on line(s) [3]"


def _assert_no_raw_prefix(text: str) -> None:
    assert not any(prefix in text for prefix in RAW_PREFIXES), (
        'the report must name each prefix by its escaped spelling only'
    )


def test_cli_scans_only_the_named_directory_by_path_component(
    scoped_repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(['pkg'], root=scoped_repo)
    out = capsys.readouterr().out
    assert code == 1
    reported = [line for line in out.splitlines() if line.startswith('  ')]
    assert reported == list(PKG_REPORT_LINES), (
        'only pkg/ files may be reported: not pkgextra/violator.py, the top-level '
        f'tracked_violator.py or the gitignored ignored/handler.py; got\n{out}'
    )
    assert 'Sentinel-literal hazard' in out
    _assert_no_raw_prefix(out)


def test_cli_passes_silently_on_a_directory_without_violations(
    scoped_repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(['clean'], root=scoped_repo)
    captured = capsys.readouterr()
    assert (code, captured.out, captured.err) == (0, '', '')


def test_cli_scans_the_union_of_several_directories(
    scoped_repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(['pkg', 'clean', 'pkgextra'], root=scoped_repo)
    out = capsys.readouterr().out
    assert code == 1
    reported = [line for line in out.splitlines() if line.startswith('  ')]
    assert reported == [*PKG_REPORT_LINES, PKGEXTRA_REPORT_LINE]
    _assert_no_raw_prefix(out)


@pytest.mark.parametrize(
    'directory',
    [
        pytest.param('no_such_dir', id='missing'),
        pytest.param('tracked_violator.py', id='a-file'),
        pytest.param('..', id='outside-root'),
    ],
)
def test_cli_refuses_a_directory_argument_that_is_not_a_directory_under_root(
    scoped_repo: Path, capsys: pytest.CaptureFixture[str], directory: str,
) -> None:
    code = main(['clean', directory], root=scoped_repo)
    captured = capsys.readouterr()
    assert code == 2
    assert repr(directory) in captured.err
    assert captured.out == ''


def test_cli_reports_an_unreadable_source_as_a_usage_error_not_a_violation(
    scoped_repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (scoped_repo / 'pkg' / 'not_utf8.py').write_bytes(b'LT = chr(60)\n# caf\xe9\n')
    code = main(['pkg'], root=scoped_repo)
    captured = capsys.readouterr()
    assert code == 2
    assert 'not_utf8.py' in captured.err
    assert captured.out == ''


@pytest.fixture
def broken_checkout(tmp_path: Path) -> Path:
    """A checkout whose ``.git`` points nowhere, so every git call in it fails."""
    root = tmp_path / 'broken'
    _plant(root, '.git', f'gitdir: {tmp_path / "nowhere"}\n')
    _plant(root, 'pkg/handler.py', 'LT = chr(60)\n')
    return root


def test_cli_reports_a_git_failure_as_a_usage_error_not_a_violation(
    broken_checkout: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(['pkg'], root=broken_checkout)
    captured = capsys.readouterr()
    assert code == 2
    assert 'git' in captured.err
    assert captured.out == ''


_RUN_MAIN_ON_ROOT = (
    'import pathlib, sys, raw_envelope_literal_guard as guard; '
    'sys.exit(guard.main(sys.argv[2:], root=pathlib.Path(sys.argv[1])))'
)


def test_cli_reports_a_git_failure_with_assertions_stripped(broken_checkout: Path) -> None:
    proc = subprocess.run(
        [sys.executable, '-O', '-S', '-E', '-c', _RUN_MAIN_ON_ROOT, str(broken_checkout), 'pkg'],
        cwd=Path(raw_envelope_literal_guard.__file__).parent,
        capture_output=True, text=True, check=False,
    )
    assert (proc.returncode, proc.stdout) == (2, ''), proc.stderr
    assert 'git discovery failed' in proc.stderr


def _lint_leg_env(environ: Mapping[str, str]) -> dict[str, str]:
    """*environ* with no Python environment selected and no venv's ``bin`` on PATH."""
    env = without_python_env_selection(environ)
    env['PATH'] = os.pathsep.join(
        entry for entry in env.get('PATH', '').split(os.pathsep)
        if entry and not (Path(entry).parent / 'pyvenv.cfg').is_file()
    )
    return env


LINT_LEG_ENV = _lint_leg_env(os.environ)
LINT_LEG_PYTHON3 = shutil.which('python3', path=LINT_LEG_ENV['PATH'])


@pytest.mark.parametrize(
    'interpreter',
    [
        pytest.param(sys.executable, id='this-venv'),
        pytest.param(LINT_LEG_PYTHON3, id='lint-leg-python3'),
    ],
)
def test_cli_runs_as_a_script_under_bare_python_from_the_checkout_root(
    interpreter: str | None,
) -> None:
    """The lint leg runs PATH's ``python3`` outside any venv, older than this venv's on this host."""
    if interpreter is None:
        pytest.skip('no python3 on PATH outside a venv')
    script = Path(raw_envelope_literal_guard.__file__).resolve()
    proc = subprocess.run(
        [interpreter, '-S', '-E', str(script), 'tests/scripts'],
        cwd=REPO_ROOT, env=LINT_LEG_ENV, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f'the lint-leg invocation under {interpreter} exited {proc.returncode}; '
        f'stdout: {proc.stdout!r}; stderr: {proc.stderr!r}'
    )
