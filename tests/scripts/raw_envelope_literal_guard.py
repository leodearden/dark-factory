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

Standard library only, and ``git_listing`` beside it: a module's lint leg runs
this file under plain ``python3``, with no virtualenv.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

from git_listing import listed_files

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


def describe(violations: Mapping[str, Mapping[str, tuple[int, ...]]]) -> str:
    """One indented line per violating path, each prefix escaped, sorted by path."""
    return '\n'.join(
        f'  {path}: '
        + '; '.join(f'{prefix} on line(s) {list(lines)}' for prefix, lines in hits.items())
        for path, hits in sorted(violations.items())
    )
