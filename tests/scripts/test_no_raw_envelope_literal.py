"""Repo-wide enforcement of the 'Sentinel-literal hazard' rule.

The rule is owned by ``shared/src/shared/toolcall_markup.py``: a source file
that handles MCP envelope markup must never spell an envelope literal with a
raw opening bracket, because an agent editing that file would have to emit the
literal inside its own tool-call envelope and so truncate its own edit.

POPULATION. Every ``.py`` file git knows about (tracked, or untracked and not
ignored) whose source handles envelope markup: it spells the bracket the
sanctioned way (the text ``chr(60)`` or the ``\\x3c`` escape), or it imports
``shared.toolcall_markup``. Files are found by path scan and never imported.

NEEDLES. The two raw prefixes every envelope literal starts with: the bracket
followed by a slash, and the bracket followed by ``parameter``. Reports name a
prefix by its escaped spelling and give line numbers only, so a red run never
prints a raw literal back into an agent's context.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from shared.toolcall_markup import ENVELOPE_LITERALS

REPO_ROOT = Path(__file__).parents[2]

_LT = chr(60)
ESCAPED_BRACKET = '\\x3c'
RAW_PREFIXES: tuple[str, ...] = (_LT + '/', _LT + 'parameter')
SANCTIONED_BRACKET_SPELLINGS: tuple[str, ...] = ('chr(60)', ESCAPED_BRACKET)
LITERAL_OWNER = 'shared.toolcall_markup'


def imports_literal_owner(source: str) -> bool:
    """Whether *source* imports :data:`LITERAL_OWNER`; an unparsable source counts."""
    raise NotImplementedError


def handles_envelope_markup(source: str) -> bool:
    """Whether *source* belongs to the guarded population."""
    raise NotImplementedError


def raw_prefix_lines(source: str) -> dict[str, tuple[int, ...]]:
    """1-based lines holding each raw prefix, keyed by the prefix's escaped spelling."""
    raise NotImplementedError


def python_files(root: Path) -> list[Path]:
    """Existing ``.py`` files git tracks, or sees untracked and not ignored, under *root*."""
    raise NotImplementedError


def guarded_population(root: Path) -> list[Path]:
    """The :func:`python_files` under *root* whose source handles envelope markup."""
    raise NotImplementedError


def raw_literal_violations(root: Path) -> dict[str, dict[str, tuple[int, ...]]]:
    """Repo-relative posix path -> :func:`raw_prefix_lines`, for each guarded file with hits."""
    raise NotImplementedError


def _git(cwd: Path, *args: str) -> str:
    """Run git in *cwd* with ``GIT_*`` scrubbed, failing loudly and never skipping.

    The idiom, and its reasons, are tests/scripts/test_nonmember_ruff_config.py::_git.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    command = ['git', *args]
    try:
        proc = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, env=env, check=False,
        )
    except OSError as exc:
        raise AssertionError(f'could not run {command} in {cwd}: {exc!r}') from exc
    assert proc.returncode == 0, (
        f'{command} failed in {cwd} with rc={proc.returncode}; stderr: {proc.stderr!r}'
    )
    return proc.stdout


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
    _git(root, 'init', '-q')
    _plant(root, '.gitignore', 'ignored/\n')
    _plant(root, 'tracked_violator.py', 'LT = chr(60)\nX = "' + _LT + '/content>"\n')
    _plant(root, 'html_fixture.py', 'PAGE = "' + _LT + 'p>hi' + _LT + '/p>"\n')
    _plant(root, 'clean_handler.py', 'from shared.toolcall_markup import closer_for\n')
    _plant(root, 'deleted.py', 'LT = chr(60)\n')
    _plant(root, 'notes.txt', 'chr(60) ' + _LT + '/content>\n')
    _git(
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


def test_python_files_lists_existing_tracked_and_unignored_python_only(
    synthetic_repo: Path,
) -> None:
    expected = [
        synthetic_repo / 'clean_handler.py',
        synthetic_repo / 'html_fixture.py',
        synthetic_repo / 'pkg' / 'untracked_violator.py',
        synthetic_repo / 'tracked_violator.py',
    ]
    assert sorted(python_files(synthetic_repo)) == expected


def test_raw_literal_violations_reports_tracked_and_untracked_violators(
    synthetic_repo: Path,
) -> None:
    violations = raw_literal_violations(synthetic_repo)
    assert violations == {
        'tracked_violator.py': {'\\x3c/': (2,)},
        'pkg/untracked_violator.py': {'\\x3cparameter': (4,)},
    }
