#!/usr/bin/env python3
"""Lint check: flag assert_not_called() when assert_not_awaited() is also in the same function.

Rule: If a function body contains BOTH assert_not_called() and assert_not_awaited() attribute
calls, emit a violation for each assert_not_called() call, suggesting assert_not_awaited()
instead.

Origin: Task 525 standardised AsyncMock assertions to assert_not_awaited() in the edge-fetch
guard test. Task 571 removed the introspection-based meta-test that guarded that convention.
This script replaces that guard with a durable AST-based lint check integrated into
hooks/project-checks.

The rule is narrow by design: it only flags the exact regression pattern (mixing both styles
in the same function body) and produces zero false positives against the current
fused-memory/tests/ codebase, where no single function mixes the two styles.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import NamedTuple


class Violation(NamedTuple):
    """A lint violation found by the checker."""

    filename: str
    lineno: int
    col_offset: int
    message: str


class _AssertionCallCollector(ast.NodeVisitor):
    """Collect attribute-call nodes by assertion name within a single function scope.

    Stops at nested function boundaries so inner functions are counted separately.
    """

    def __init__(self) -> None:
        self.not_called: list[ast.Call] = []
        self.not_awaited: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Record assert_not_called and assert_not_awaited attribute calls."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'assert_not_called':
                self.not_called.append(node)
            elif node.func.attr == 'assert_not_awaited':
                self.not_awaited.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        """Stop at nested function boundary — do not descend into inner functions."""

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        """Stop at nested async function boundary — do not descend into inner functions."""


def _collect_function_scopes(
    tree: ast.AST,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return all FunctionDef/AsyncFunctionDef nodes found anywhere in the AST.

    Uses ast.walk so nested functions are discovered as separate scopes.
    """
    scopes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(node)
    return scopes


_VIOLATION_MSG = (
    'assert_not_called() on attribute appears alongside assert_not_awaited() in the same'
    ' function \u2014 use assert_not_awaited() for consistency with the AsyncMock'
    ' assertion-style convention (task 525)'
)


def find_violations(source: str, filename: str) -> list[Violation]:
    """Parse *source* and return violations for mixed assert_not_called/assert_not_awaited usage.

    A violation is emitted for each assert_not_called() attribute call in any function that
    ALSO contains at least one assert_not_awaited() attribute call in the same body.
    Nested functions are scoped independently.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    violations: list[Violation] = []

    for func in _collect_function_scopes(tree):
        collector = _AssertionCallCollector()
        for child in func.body:
            collector.visit(child)

        if collector.not_called and collector.not_awaited:
            for call in collector.not_called:
                violations.append(
                    Violation(
                        filename=filename,
                        lineno=call.lineno,
                        col_offset=call.col_offset,
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
        description='Check for assert_not_called/assert_not_awaited style mixing in test files.'
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
            # Only scan test files and conftest; runtime code does not use AsyncMock.
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
    read_errors: list[tuple[Path, OSError]] = []
    for file_path in files_to_scan:
        try:
            source = file_path.read_text(encoding='utf-8')
        except OSError as exc:
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
