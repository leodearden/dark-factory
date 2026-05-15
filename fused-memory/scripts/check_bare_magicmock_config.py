#!/usr/bin/env python3
"""Lint check: flag bare MagicMock() assigned to config-named variables in test files.

Rule: Any assignment of the form ``<config_name> = MagicMock()`` where the call has
no ``spec``, no ``spec_set`` keyword argument, and no positional argument (MagicMock's
first positional IS spec) is a violation — unless the immediately-preceding non-blank
source line is a structured exemption comment:

    # noqa: bare-magicmock — <reason>   (em-dash or ASCII hyphen; reason must be non-empty)

Config-name set: exact ``config`` and ``cfg``, plus any name ending with ``_config`` or
``_cfg`` (e.g. ``orch_config``, ``mock_cfg``).  Generic names like ``mcp``, ``mock``, ``m``
are intentionally excluded — this rule targets config objects only (non-goal: no scope creep).

Attribute target exclusion: only ``ast.Name`` assignment targets are detected (i.e.,
module-level and local ``config = MagicMock()``).  Attribute assignments such as
``self.config = MagicMock()`` or ``obj.cfg = MagicMock()`` are intentionally excluded —
resolving attribute targets requires class/instance context that is not available during
AST inspection.  This exclusion is an intentional non-goal; it is documented here so the
scope limit is discoverable.

Preferred alternatives named in the rejection message:
  • ``mock_orch_config`` fixture (orchestrator/tests/conftest.py:91)
  • ``MagicMock(spec_set=pydantic_spec(...))`` (orchestrator/tests/_orch_helpers.py:19)

Origin: Task 1339 (migrate existing bare configs), task 1313/1064 (spec discipline).
This guard is implemented in task 1372 to prevent regressions after the migration.

This script is intentionally stdlib-only (ast, argparse, pathlib, re, sys, typing) so
hooks/project-checks can invoke it via plain python3 without uv env-resolution overhead.
Adding a third-party dependency here would break that fast path.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import NamedTuple


class Violation(NamedTuple):
    """A lint violation found by the checker."""

    filename: str
    lineno: int
    col_offset: int
    message: str


# Names that the checker considers "config objects".
# Exact matches plus suffix rules — generic names are excluded by design.
_CONFIG_EXACT: frozenset[str] = frozenset({'config', 'cfg'})
_CONFIG_SUFFIXES: tuple[str, ...] = ('_config', '_cfg')


def _is_config_name(name: str) -> bool:
    """Return True if *name* identifies a config-named variable."""
    if name in _CONFIG_EXACT:
        return True
    return any(name.endswith(suffix) for suffix in _CONFIG_SUFFIXES)


def _is_magicmock_call(node: ast.expr) -> bool:
    """Return True if *node* is a call whose func resolves to ``MagicMock``.

    Accepts:
      - ``MagicMock(...)``                  → ast.Name(id='MagicMock')
      - ``mock.MagicMock(...)``             → ast.Attribute(attr='MagicMock')
      - ``unittest.mock.MagicMock(...)``    → ast.Attribute(attr='MagicMock')
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == 'MagicMock'
    if isinstance(func, ast.Attribute):
        return func.attr == 'MagicMock'
    return False


def _is_specced(call: ast.Call) -> bool:
    """Return True if *call* provides a real spec via positional arg or spec/spec_set kwarg.

    Edge cases handled explicitly:
    - ``MagicMock(*args)`` (ast.Starred positional): treated as NOT specced.
      The spread is opaque at AST-inspection time, so we cannot guarantee a spec
      is present; flagging is safer than a false negative.
    - ``MagicMock(spec=None)`` / ``MagicMock(spec_set=None)``: treated as NOT specced.
      Passing None is semantically equivalent to omitting spec altogether and defeats
      the rule's intent.
    """
    # Concrete (non-Starred) positional args only — *args spread cannot be inspected.
    concrete_args = [a for a in call.args if not isinstance(a, ast.Starred)]
    if concrete_args:
        # First positional parameter of MagicMock IS spec.
        return True
    # spec/spec_set keyword args — but only when the value is not the literal None.
    for kw in call.keywords:
        if kw.arg in ('spec', 'spec_set') and not (
            isinstance(kw.value, ast.Constant) and kw.value.value is None
        ):
            return True
    return False


# Exemption comment regex.
# Matches: # noqa: bare-magicmock — <non-empty-reason>
# Accepts em-dash (—) or ASCII hyphen (-) as separator.
# Requires at least one non-space character after the separator.
_EXEMPT_RE = re.compile(r'#\s*noqa:\s*bare-magicmock\s*[—\-]+\s*\S.*')

_VIOLATION_MSG = (
    'bare MagicMock() assigned to a config variable with no spec/spec_set.'
    ' Use the mock_orch_config fixture or MagicMock(spec_set=pydantic_spec(...))'
    ' instead (see task 1339 regression guard; tasks 1313/1064).'
    ' To suppress: add # noqa: bare-magicmock — <reason> on the preceding non-blank line.'
)


def _is_exempted(lines: list[str], lineno: int) -> bool:
    """Return True if the assignment at *lineno* (1-based) is preceded by a valid exemption comment.

    Walks upward from ``lineno - 1`` over blank lines to the nearest non-blank line.
    If that line matches _EXEMPT_RE the assignment is exempt.
    Any intervening non-blank, non-matching line breaks the exemption.
    """
    # lineno is 1-based; convert to 0-based index of the line ABOVE the assignment.
    idx = lineno - 2  # the line immediately above
    while idx >= 0:
        line = lines[idx]
        stripped = line.strip()
        if stripped == '':
            idx -= 1
            continue
        # Nearest non-blank line found — must match the exemption regex.
        return bool(_EXEMPT_RE.match(stripped))
    return False


def find_violations(source: str, filename: str) -> list[Violation]:
    """Parse *source* and return violations for bare MagicMock() config assignments.

    A violation is emitted for each ``<config_name> = MagicMock()`` call that:
      - targets a single ast.Name whose id matches the config-name set,
      - calls MagicMock (by name or attribute) with no spec/spec_set,
      - is NOT preceded (on the nearest non-blank source line) by a valid exemption comment.

    SyntaxError in *source* → returns an empty list.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    lines = source.splitlines()
    violations: list[Violation] = []

    for node in ast.walk(tree):
        # Determine the assignment target name and value.
        if isinstance(node, ast.Assign):
            # Only single-name targets: skip tuple/list unpacking.
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            value = node.value
            assignment_lineno = node.lineno
            col_offset = target.col_offset
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if not isinstance(target, ast.Name):
                continue
            if node.value is None:
                continue
            name = target.id
            value = node.value
            assignment_lineno = node.lineno
            col_offset = target.col_offset
        else:
            continue

        if not _is_config_name(name):
            continue
        if not _is_magicmock_call(value):
            continue
        if _is_specced(value):  # type: ignore[arg-type]
            continue
        if _is_exempted(lines, assignment_lineno):
            continue

        violations.append(
            Violation(
                filename=filename,
                lineno=assignment_lineno,
                col_offset=col_offset,
                message=_VIOLATION_MSG,
            )
        )

    return violations


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Accepts file paths and/or directories.

    For directories, recursively scans for test_*.py and conftest.py files only.
    Prints violations to stdout in 'path:lineno:col: message' format (ruff-style).

    Explicit file paths are validated up front; a missing explicit path fails
    fast with exit code 2 before any scan work. Mid-scan OSErrors (e.g. a file
    yanked between rglob discovery and read) are accumulated and reported on
    stderr without discarding violations already collected.

    Returns 0 if clean, 1 if only violations were found, 2 on any fatal error
    (missing explicit path or transient read failure).
    """
    parser = argparse.ArgumentParser(
        description='Check for bare MagicMock() assignments to config-named variables in test files.'
    )
    parser.add_argument('paths', nargs='+', help='Files or directories to check')
    args = parser.parse_args(argv)

    # Phase 1: discovery + upfront validation of explicit paths.
    # rglob results are guaranteed to exist at discovery time, so only
    # non-directory (explicit) paths need the existence check.
    files_to_scan: list[Path] = []
    for path_str in args.paths:
        p = Path(path_str)
        if p.is_dir():
            files_to_scan.extend(
                sorted(set(p.rglob('test_*.py')) | set(p.rglob('conftest.py')))
            )
        else:
            if not p.exists():
                print(f'error: {p}: No such file or directory', file=sys.stderr)
                return 2
            files_to_scan.append(p)

    # Phase 2: scan. Accumulate per-file read errors without returning early,
    # so a transient OSError on one file never discards violations already
    # collected from earlier files.
    all_violations: list[Violation] = []
    read_errors: list[tuple[Path, Exception]] = []
    for file_path in files_to_scan:
        try:
            source = file_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            # UnicodeDecodeError is a ValueError subclass, not an OSError,
            # but a malformed file must be reported via the read_errors channel
            # rather than crashing with an unhandled traceback.
            read_errors.append((file_path, exc))
            continue

        violations = find_violations(source, str(file_path))
        all_violations.extend(violations)

    # Phase 3: reporting.
    for v in all_violations:
        print(f'{v.filename}:{v.lineno}:{v.col_offset}: {v.message}')
    for file_path, exc in read_errors:
        print(f'error reading {file_path}: {exc}', file=sys.stderr)

    if read_errors:
        return 2
    return 1 if all_violations else 0


if __name__ == '__main__':
    sys.exit(main())
