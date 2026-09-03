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


# Exemption comment regex.
# Matches: ``# noqa: module-local-testclient — <non-empty-reason>``
# Accepts em-dash (—) or ASCII hyphen (-) as separator.
# Requires at least one non-space character after the separator.
#
# Template and contract are inherited VERBATIM from
# ``check_bare_magicmock_config.py::_EXEMPT_TEMPLATE`` so this repo keeps ONE
# suppression grammar rather than two.  Keying on the rule's own code also
# guarantees a ``bare-magicmock`` pragma can never silently exempt this rule:
# the remedies are unrelated, so a pragma for one is not informed consent for
# the other.
_EXEMPT_TEMPLATE = r'#\s*noqa:\s*{code}\s*[—\-]+\s*\S.*'

_RULE_CODE = 'module-local-testclient'

_EXEMPT_RE = re.compile(_EXEMPT_TEMPLATE.format(code=re.escape(_RULE_CODE)))


def _is_exempted(lines: list[str], lineno: int, code: str) -> bool:
    """Return True if the node at *lineno* (1-based) carries a valid ``code`` exemption.

    Walks upward from the line ABOVE *lineno* over blank lines to the nearest
    non-blank line.  If that line matches the exemption regex the node is exempt.
    Any intervening non-blank, non-matching line breaks the exemption.

    Inline trailing exemption NOT honored: only the nearest *preceding* non-blank
    line is inspected.  A ``# noqa: ...`` comment on the same line as the node is
    intentionally ignored — same contract as ``check_bare_magicmock_config.py``.
    """
    if code != _RULE_CODE:
        return False
    # lineno is 1-based; convert to 0-based index of the line ABOVE the node.
    idx = lineno - 2  # the line immediately above
    while idx >= 0:
        stripped = lines[idx].strip()
        if stripped == '':
            idx -= 1
            continue
        # Nearest non-blank line found — must match the exemption regex.
        return bool(_EXEMPT_RE.match(stripped))
    return False


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


def _bound_name(node: ast.expr) -> str | None:
    """Return the LOCAL BINDING *node* resolves to, or None if it is not a plain name.

    ``X`` and ``X(...)`` both resolve to ``X``; ``a.b.X`` resolves to None,
    because the binding introduced by the import statement is ``a``, not ``X``.
    """
    if isinstance(node, ast.Call):
        return _bound_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    return None


class _AliasMap(NamedTuple):
    """Local bindings a module's imports attach to the two symbols this rule cares about."""

    testclient: frozenset[str]
    fixture: frozenset[str]


def _build_alias_map(tree: ast.AST) -> _AliasMap:
    """Record which local names this module binds to ``*.testclient.TestClient`` / ``pytest.fixture``.

    Only ``ast.ImportFrom`` introduces a binding this map can use.  ``import
    a.b`` binds the ROOT package name ``a`` alone, so ``a.b.TestClient(app)``
    reaches the matcher through the trailing-name arm instead — which is exactly
    why the matcher is a union rather than an alias lookup alone.
    """
    testclient: set[str] = set()
    fixture: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        # Compare on the module's LAST segment so starlette.testclient,
        # fastapi.testclient and a bare testclient are all recognised.
        tail = (node.module or '').rpartition('.')[2]
        for alias in node.names:
            local = alias.asname or alias.name
            if alias.name == 'TestClient' and tail == 'testclient':
                testclient.add(local)
            elif alias.name == 'fixture' and tail == 'pytest':
                fixture.add(local)

    return _AliasMap(frozenset(testclient), frozenset(fixture))


# WHY BOTH ARMS, IN BOTH MATCHERS BELOW.
#
# The deleted task-3571 guard matched on the trailing callee NAME alone, so
# ``from starlette.testclient import TestClient as TC`` then ``TC(app)`` walked
# straight past it — the blind spot its own docstring claimed to close.  An
# alias map ALONE would be narrower than that guard in the other direction: it
# cannot see ``testclient.TestClient(app)`` (module-attribute access), nor any
# import shape it does not model.
#
# The UNION is deliberately BROADER than the deleted guard and never narrower.
# That is the correct direction of error for a guard: a false positive is
# locally suppressible with a one-line
# ``# noqa: module-local-testclient — <reason>`` pragma, while a false negative
# is SILENT — and a silent false negative is exactly how the deleted guard
# shipped green over a live duplicate for eleven days.


def _is_fixture(func: ast.FunctionDef | ast.AsyncFunctionDef, aliases: _AliasMap) -> bool:
    """Return True if *func* carries a pytest fixture decorator.

    Union of (decorator resolves through an alias bound to ``pytest.fixture``)
    OR (decorator's trailing name is ``fixture``).  Accepts both the bare
    ``@pytest.fixture`` and the called ``@pytest.fixture(scope='module')`` forms.

    The decorator is alias-resolved for the same reason the construction is: a
    rule that keys on the literal name ``fixture`` reopens one level up, where
    ``from pytest import fixture as fx`` stops the function looking like a
    fixture at all.
    """
    for dec in func.decorator_list:
        if _bound_name(dec) in aliases.fixture:
            return True
        if _trailing_name(dec) == 'fixture':
            return True
    return False


def _is_testclient_construction(call: ast.Call, aliases: _AliasMap) -> bool:
    """Return True if *call* constructs a TestClient.

    Union of (callee is a local name bound to ``*.testclient.TestClient``) OR
    (callee's trailing ``Name.id``/``Attribute.attr`` is ``TestClient``).
    A decoy alias (``from foo import Bar as TC`` then ``TC(app)``) matches
    neither arm.
    """
    if _bound_name(call.func) in aliases.testclient:
        return True
    return _trailing_name(call.func) == 'TestClient'


def is_scannable(filename: str) -> bool:
    """Return True if *filename* is a file this rule applies to.

    Two exclusions, applied UNIFORMLY to explicitly-passed paths and
    directory-scan results alike:

    ``conftest.py`` — LOAD-BEARING, not incidental.  ``hooks/project-checks``
    passes explicit staged file paths, so ``dashboard/tests/conftest.py`` is
    handed to this checker whenever it is edited.  conftest is the intended HOME
    of the shared ``_client`` fixture; scanning it would flag the very thing this
    rule exists to promote, and the rejection message would tell its author to
    request the fixture they are looking at.  The skip must therefore key on the
    BASENAME rather than only on a directory-scan glob, which an explicit path
    bypasses entirely.

    Anything not matching ``test_*.py`` — the rule is about a per-module app
    lifespan being stood up twice, which only happens in a test module.
    """
    name = Path(filename).name
    if name == 'conftest.py':
        return False
    return name.startswith('test_') and name.endswith('.py')


def find_violations(source: str, filename: str) -> list[Violation]:
    """Parse *source* and return violations for TestClient constructions in fixture bodies.

    A violation is emitted for each ``ast.Call`` that constructs a TestClient and
    lies inside the body of a function marked as a pytest fixture.

    Scoping to fixture BODIES is what lets this rule ship with no whitelist at
    all: a client built inline in a plain test function, or in a
    ``@contextmanager`` helper, is excluded STRUCTURALLY rather than by name.
    Files outside ``is_scannable`` return [] regardless of content.

    Matching a real ``ast.Call`` node rather than source text is load-bearing:
    several dashboard test modules (and conftest.py itself) carry the literal
    ``TestClient(app)`` inside DOCSTRINGS, so a text search would report them as
    offenders.

    SyntaxError in *source* → returns an empty list.

    Returned violations are sorted ascending by (lineno, col_offset) for
    deterministic source-order output (``ast.walk`` yields BFS order, not source
    order).
    """
    if not is_scannable(filename):
        return []

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    aliases = _build_alias_map(tree)
    lines = source.splitlines()

    violations: list[Violation] = []
    seen: set[int] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _is_fixture(node, aliases):
            continue
        for stmt in node.body:
            for child in ast.walk(stmt):
                if not isinstance(child, ast.Call):
                    continue
                # A fixture nested inside a fixture would otherwise be walked
                # twice; key on node identity so each construction is reported once.
                if id(child) in seen:
                    continue
                if not _is_testclient_construction(child, aliases):
                    continue
                seen.add(id(child))
                # Computed LAZILY — only after a construction has matched — so
                # the upward line walk never runs on every call node in the body.
                if _is_exempted(lines, child.lineno, _RULE_CODE):
                    continue
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

    # The same gate as find_violations, applied to discovery so an explicitly
    # passed conftest.py is never even read.
    files_to_scan = [f for f in files_to_scan if is_scannable(str(f))]

    all_violations: list[Violation] = []
    for file_path in files_to_scan:
        source = file_path.read_text(encoding='utf-8')
        all_violations.extend(find_violations(source, str(file_path)))

    for v in all_violations:
        print(f'{v.filename}:{v.lineno}:{v.col_offset}: {v.message}')

    return 1 if all_violations else 0


if __name__ == '__main__':
    sys.exit(main())
