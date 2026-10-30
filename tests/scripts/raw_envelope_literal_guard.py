"""The 'Sentinel-literal hazard' rule, enforced over markup-handling source files.

The rule is owned by ``shared/src/shared/toolcall_markup.py``: a source file
that handles MCP envelope markup must never spell an envelope literal with a
raw opening bracket. That module's 'Sentinel-literal hazard' section says why.

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

RUN AS A SCRIPT, ``python3 tests/scripts/raw_envelope_literal_guard.py DIR...``
scans the files under the named directories of the checkout holding this file:
exit 0 silently when clean, 1 with a report on violations, and 2 when a
directory is not a directory under the checkout or git discovery fails. Each
module's ``lint_command`` runs it over that module's own tree. Standard library
only, and ``git_listing`` beside it, because that leg runs under plain
``python3`` with no virtualenv.
"""
from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from git_listing import listed_files

CHECKOUT_ROOT = Path(__file__).resolve().parents[2]

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


def guarded_sources(root: Path, *directories: str) -> dict[str, str]:
    """Repo-relative posix path -> source, for each ``.py`` file that handles envelope markup.

    Scans the files under *directories*, each relative to *root* and matched by
    path component, or the whole of *root* when none is named.
    """
    pathspecs = [f'{Path(directory).as_posix()}/*.py' for directory in directories] or ['*.py']
    read = ((path, path.read_text(encoding='utf-8')) for path in listed_files(root, *pathspecs))
    return {
        path.relative_to(root).as_posix(): source
        for path, source in read
        if handles_envelope_markup(source)
    }


def raw_literal_violations(population: Mapping[str, str]) -> dict[str, dict[str, tuple[int, ...]]]:
    """Path -> :func:`raw_prefix_lines`, for each file in *population* with hits."""
    hits_by_path = {path: raw_prefix_lines(source) for path, source in population.items()}
    return {path: hits for path, hits in hits_by_path.items() if hits}


def describe(violations: Mapping[str, Mapping[str, tuple[int, ...]]]) -> str:
    """One indented line per violating path, each prefix escaped, sorted by path."""
    return '\n'.join(
        f'  {path}: '
        + '; '.join(f'{prefix} on line(s) {list(lines)}' for prefix, lines in hits.items())
        for path, hits in sorted(violations.items())
    )


def report(violations: Mapping[str, Mapping[str, tuple[int, ...]]]) -> str:
    """The violation report: what is wrong, :func:`describe`'s lines, and the remedy."""
    return (
        'These files handle MCP envelope markup yet spell an envelope literal with a '
        f'raw opening bracket (each prefix is shown escaped):\n{describe(violations)}\n'
        f'Build the literal in code from {LITERAL_OWNER} constants or chr(60), or spell '
        f'the bracket as {ESCAPED_BRACKET} in prose and strings. Why: the '
        "'Sentinel-literal hazard' section of shared/src/shared/toolcall_markup.py."
    )


def _is_directory_under(root: Path, directory: str) -> bool:
    candidate = (root / directory).resolve()
    return candidate.is_dir() and candidate.is_relative_to(root.resolve())


def main(argv: Sequence[str], *, root: Path = CHECKOUT_ROOT) -> int:
    """Scan the markup-handling files under *argv*'s directories of *root*; return the exit code."""
    parser = argparse.ArgumentParser(
        description='Report markup-handling .py files that spell a raw envelope literal.',
    )
    parser.add_argument(
        'directories', nargs='+', metavar='DIRECTORY', help='relative to the checkout root',
    )
    directories: list[str] = parser.parse_args(argv).directories
    not_directories = [repr(d) for d in directories if not _is_directory_under(root, d)]
    if not_directories:
        print(f'not a directory under {root}: {", ".join(not_directories)}', file=sys.stderr)
        return 2
    try:
        population = guarded_sources(root, *directories)
    except AssertionError as exc:
        print(f'git discovery failed, so nothing was scanned: {exc}', file=sys.stderr)
        return 2
    violations = raw_literal_violations(population)
    if not violations:
        return 0
    print(report(violations))
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
