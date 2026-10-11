"""Repo-wide scan enforcing the 'Sentinel-literal hazard' rule on markup-handling files.

The rule is owned by ``shared/src/shared/toolcall_markup.py``: a source file
that handles MCP envelope markup must never spell an envelope literal with a
raw opening bracket, because an agent editing that file would have to emit the
literal inside its own tool-call envelope and so truncate its own edit.

POPULATION. Every ``.py`` file git knows about (tracked, or untracked and not
ignored) whose source already handles envelope markup the sanctioned way: it
spells the bracket as the text ``chr(60)`` or the ``\\x3c`` escape, or it
imports ``shared.toolcall_markup``. Files are found by path scan, read once and
never imported. The population selects itself, so this guard stops a
markup-handling file from regressing to a raw literal; a file that spells
envelope literals ONLY raw never enters it and is not caught here.

NEEDLES. The two raw prefixes every envelope literal starts with: the bracket
followed by a slash, and the bracket followed by ``parameter``. Reports name a
prefix by its escaped spelling and give line numbers only, so a red run never
prints a raw literal back into an agent's context.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import pytest
from git_listing import git, listed_files
from shared.toolcall_markup import ENVELOPE_LITERALS

REPO_ROOT = Path(__file__).parents[2]

_LT = chr(60)
ESCAPED_BRACKET = '\\x3c'
RAW_PREFIXES: tuple[str, ...] = (_LT + '/', _LT + 'parameter')
SANCTIONED_BRACKET_SPELLINGS: tuple[str, ...] = ('chr(60)', ESCAPED_BRACKET)
LITERAL_OWNER = 'shared.toolcall_markup'
_OWNER_PACKAGE, _, _OWNER_MODULE = LITERAL_OWNER.rpartition('.')


def _imports_owner(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.module == LITERAL_OWNER or (
            node.module == _OWNER_PACKAGE
            and any(alias.name == _OWNER_MODULE for alias in node.names)
        )
    if isinstance(node, ast.Import):
        return any(alias.name == LITERAL_OWNER for alias in node.names)
    return False


def imports_literal_owner(source: str) -> bool:
    """Whether *source* imports :data:`LITERAL_OWNER`; an unparsable source counts."""
    if _OWNER_MODULE not in source:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True
    return any(_imports_owner(node) for node in ast.walk(tree))


def handles_envelope_markup(source: str) -> bool:
    """Whether *source* belongs to the guarded population."""
    spells_bracket = any(spelling in source for spelling in SANCTIONED_BRACKET_SPELLINGS)
    return spells_bracket or imports_literal_owner(source)


def raw_prefix_lines(source: str) -> dict[str, tuple[int, ...]]:
    """1-based lines holding each raw prefix, keyed by the prefix's escaped spelling."""
    lines = source.split('\n')
    hits = {
        prefix.replace(_LT, ESCAPED_BRACKET): tuple(
            number for number, line in enumerate(lines, start=1) if prefix in line
        )
        for prefix in RAW_PREFIXES
    }
    return {key: numbers for key, numbers in hits.items() if numbers}


def guarded_sources(root: Path) -> dict[str, str]:
    """Repo-relative posix path -> source, for each ``.py`` file under *root* that handles envelope markup."""
    read = ((path, path.read_text(encoding='utf-8')) for path in listed_files(root, '*.py'))
    return {
        path.relative_to(root).as_posix(): source
        for path, source in read
        if handles_envelope_markup(source)
    }


def raw_literal_violations(population: Mapping[str, str]) -> dict[str, dict[str, tuple[int, ...]]]:
    """Path -> :func:`raw_prefix_lines`, for each file in *population* with hits."""
    hits_by_path = {path: raw_prefix_lines(source) for path, source in population.items()}
    return {path: hits for path, hits in hits_by_path.items() if hits}


def _describe(violations: dict[str, dict[str, tuple[int, ...]]]) -> str:
    return '\n'.join(
        f'  {path}: '
        + '; '.join(f'{prefix} on line(s) {list(lines)}' for prefix, lines in hits.items())
        for path, hits in sorted(violations.items())
    )


@pytest.fixture(scope='module')
def repo_population() -> dict[str, str]:
    return guarded_sources(REPO_ROOT)


def test_no_markup_handling_file_spells_a_raw_envelope_literal(
    repo_population: dict[str, str],
) -> None:
    violations = raw_literal_violations(repo_population)
    assert not violations, (
        'These files handle MCP envelope markup yet spell an envelope literal with a '
        f'raw opening bracket (each prefix is shown escaped):\n{_describe(violations)}\n'
        f'Build the literal in code from {LITERAL_OWNER} constants or chr(60), or spell '
        f'the bracket as {ESCAPED_BRACKET} in prose and strings. Why: the '
        "'Sentinel-literal hazard' section of shared/src/shared/toolcall_markup.py."
    )


def test_this_guard_is_inside_its_own_population(repo_population: dict[str, str]) -> None:
    this_guard = Path(__file__).relative_to(REPO_ROOT).as_posix()
    assert this_guard in repo_population, (
        'This guard spells chr(60) and imports the literal owner, so git discovery plus '
        'the population predicate must select it; if they do not, the repo-wide check above '
        'is passing vacuously.'
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
