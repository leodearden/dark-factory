#!/usr/bin/env python3
"""Lint check: flag a pytest FIXTURE that constructs its own ``TestClient``.

Rule: inside a ``test_*.py`` module, a function marked as a pytest fixture must
not construct a ``TestClient``.  ``dashboard/tests/conftest.py`` already provides
a shared module-scoped ``_client``; a second one means a second app lifespan per
module — the exact cost the shared fixture exists to pay once.

Origin: task 3571 landed an AST guard class (``TestNoModuleLocalClientFixtures``
in ``dashboard/tests/test_jsx_source_helpers.py``) that lint-ed its SIBLING test
modules from inside the pytest suite.  Review rejected that shape and it was
deleted in commit 9096654196.  Task 4485 re-expresses the same invariant here,
as a lint OUTSIDE the suite, where a lint belongs: a test module asserting facts
about other test modules' source text is not a test, and the deleted guard's
three concrete defects (a hardcoded filename whitelist living in a third module,
a bidirectional equality assertion that went red when a violation was REMOVED,
and a trailing-callee-name match blind to ``import TestClient as TC``) are each
answered by construction below.

Not a ruff rule because ruff (0.15.9 here) has no user-defined-rule plugin
mechanism, and no built-in rule can express "a pytest fixture constructs its own
TestClient".  flake8 is not in this repo's toolchain.

THIS SCRIPT CONTAINS NO FILENAME EXEMPTION LIST, deliberately.  A module that
legitimately needs its own client fixture says so AT THE SITE, with
``# noqa: module-local-testclient — <reason>`` on the preceding non-blank line.
That pragma is the repo's stated norm (pyproject.toml): local, visible and
greppable, and it travels with the code under both rename and split — unlike the
whitelist it replaces, which broke on a rename of a file it did not even name.

This script is intentionally stdlib-only (ast, argparse, pathlib, re, sys) so
hooks/project-checks can invoke it via plain python3 without uv env-resolution
overhead.  Adding a third-party dependency here would break that fast path.
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


_VIOLATION_MSG = (
    'pytest fixture constructs its own TestClient. Request conftest.py\'s shared'
    ' module-scoped `_client` fixture instead — a second TestClient means a second'
    ' app lifespan per module, the cost the shared fixture exists to pay once'
    ' (task 4485; guard originally task 3571).'
    ' To suppress: add # noqa: module-local-testclient — <reason> on the'
    ' preceding non-blank line.'
)


def _trailing_name(node: ast.expr) -> str | None:
    """Return the trailing identifier of *node*: ``X`` for ``X``, ``a.b.X`` or ``X()``."""
    if isinstance(node, ast.Call):
        return _trailing_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_fixture(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if *func* carries a pytest fixture decorator.

    Accepts both the bare ``@pytest.fixture`` and the called
    ``@pytest.fixture(scope='module')`` forms.
    """
    return any(_trailing_name(dec) == 'fixture' for dec in func.decorator_list)


def _is_testclient_construction(call: ast.Call) -> bool:
    """Return True if *call* constructs a TestClient."""
    return _trailing_name(call.func) == 'TestClient'


def find_violations(source: str, filename: str) -> list[Violation]:
    """Parse *source* and return violations for TestClient constructions in fixture bodies.

    A violation is emitted for each ``ast.Call`` that constructs a TestClient and
    lies inside the body of a function marked as a pytest fixture.

    Matching a real ``ast.Call`` node rather than source text is load-bearing:
    several dashboard test modules (and conftest.py itself) carry the literal
    ``TestClient(app)`` inside DOCSTRINGS, so a text search would report them as
    offenders.

    SyntaxError in *source* → returns an empty list.

    Returned violations are sorted ascending by (lineno, col_offset) for
    deterministic source-order output (``ast.walk`` yields BFS order, not source
    order).
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    violations: list[Violation] = []
    seen: set[int] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _is_fixture(node):
            continue
        for stmt in node.body:
            for child in ast.walk(stmt):
                if not isinstance(child, ast.Call):
                    continue
                # A fixture nested inside a fixture would otherwise be walked
                # twice; key on node identity so each construction is reported once.
                if id(child) in seen:
                    continue
                if not _is_testclient_construction(child):
                    continue
                seen.add(id(child))
                violations.append(
                    Violation(
                        filename=filename,
                        lineno=child.lineno,
                        col_offset=child.col_offset,
                        message=_VIOLATION_MSG,
                    )
                )

    violations.sort(key=lambda v: (v.lineno, v.col_offset))
    return violations


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Accepts file paths and/or directories."""
    parser = argparse.ArgumentParser(
        description='Check for pytest fixtures constructing their own TestClient.'
    )
    parser.add_argument('paths', nargs='+', help='Files or directories to check')
    args = parser.parse_args(argv)

    files_to_scan: list[Path] = []
    for path_str in args.paths:
        p = Path(path_str)
        if p.is_dir():
            files_to_scan.extend(sorted(p.rglob('test_*.py')))
        else:
            files_to_scan.append(p)

    all_violations: list[Violation] = []
    for file_path in files_to_scan:
        source = file_path.read_text(encoding='utf-8')
        all_violations.extend(find_violations(source, str(file_path)))

    for v in all_violations:
        print(f'{v.filename}:{v.lineno}:{v.col_offset}: {v.message}')

    return 1 if all_violations else 0


if __name__ == '__main__':
    sys.exit(main())
