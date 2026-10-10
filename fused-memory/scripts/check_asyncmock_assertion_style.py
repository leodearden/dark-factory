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

This script is intentionally stdlib-only (ast, pathlib, sys, plus its stdlib-only sibling
_lint_cli.py) so hooks/project-checks can invoke it via plain python3 without uv
env-resolution overhead. Adding a third-party dependency here would break that fast path.
"""
from __future__ import annotations

import ast
import functools
import sys
from pathlib import Path

# Sibling import that survives `python3 -I`: see _lint_cli.py's module docstring.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from _lint_cli import Violation, discover_files, run_cli
finally:
    del sys.path[0]

# The files a DIRECTORY argument expands to; runtime code does not use AsyncMock.
_DISCOVERY_GLOBS: tuple[str, ...] = ('test_*.py', 'conftest.py')


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

    For directories, recursively scans test_*.py and conftest.py files only.
    Output and the 0/1/2 exit ladder are ``_lint_cli.run_cli``'s.
    """
    return run_cli(
        argv,
        description='Check for assert_not_called/assert_not_awaited style mixing in test files.',
        discover=functools.partial(discover_files, globs=_DISCOVERY_GLOBS),
        find_violations=find_violations,
    )


if __name__ == '__main__':
    sys.exit(main())
