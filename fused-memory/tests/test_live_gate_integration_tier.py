"""Every test a live-service reachability gate applies to is in the integration tier.

The default `-m 'not integration'` suite is the merge gate, which cannot tell a backing-store
outage from a test failure; the offline `-m integration` lane can (task 11484). Per-module
behavioral guards for the same rule: tests/test_integration_marker_real_service.py.
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from _ast_guard import parse_python_module
from shared.pytest_mark_grammar import marker_name, pytestmark_elements
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

TESTS_DIR = Path(__file__).parent

LIVE_GATE_FACTORIES = frozenset({'falkor_skipif', 'qdrant_skipif'})


def _is_live_gate(mark: ast.expr) -> bool:
    if not isinstance(mark, ast.Call):
        return False
    callee = mark.func
    return (isinstance(callee, ast.Name) and callee.id in LIVE_GATE_FACTORIES) or (
        isinstance(callee, ast.Attribute) and callee.attr in LIVE_GATE_FACTORIES
    )


def _test_items(
    body: Sequence[ast.stmt], applied: list[ast.expr], prefix: str,
) -> Iterator[tuple[str, list[ast.expr]]]:
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name.startswith('test'):
            yield f'{prefix}{stmt.name}', [*applied, *stmt.decorator_list]
        elif isinstance(stmt, ast.ClassDef) and stmt.name.startswith('Test'):
            class_marks = [*applied, *stmt.decorator_list, *pytestmark_elements(stmt.body)]
            yield from _test_items(stmt.body, class_marks, f'{prefix}{stmt.name}::')


def scan_live_gated_tests(tree: ast.Module) -> tuple[list[str], list[str]]:
    """Return (live-gated test suffixes, those not in the integration tier), as `Class::test`."""
    gated: list[str] = []
    not_integration: list[str] = []
    for suffix, marks in _test_items(tree.body, pytestmark_elements(tree.body), ''):
        if any(_is_live_gate(mark) for mark in marks):
            gated.append(suffix)
            if not any(marker_name(mark) == 'integration' for mark in marks):
                not_integration.append(suffix)
    return gated, not_integration


def test_every_live_gated_test_is_in_the_integration_tier() -> None:
    gated: list[str] = []
    not_integration: list[str] = []
    for path in sorted(TESTS_DIR.rglob('test_*.py')):
        relpath = path.relative_to(TESTS_DIR).as_posix()
        module_gated, module_not_integration = scan_live_gated_tests(parse_python_module(path))
        gated.extend(f'{relpath}::{suffix}' for suffix in module_gated)
        not_integration.extend(f'{relpath}::{suffix}' for suffix in module_not_integration)

    assert gated, f'no test under {TESTS_DIR} applies a live gate; the scan has gone blind'
    assert not not_integration, (
        'these tests apply a live-service reachability gate but run in the default '
        "-m 'not integration' suite, where a backing-store outage reds the merge gate. "
        'Add @pytest.mark.integration on the test, its class, or the module pytestmark:\n'
        + '\n'.join(not_integration)
    )


@pytest.mark.parametrize(('source', 'expected_gated', 'expected_not_integration'), [
    pytest.param(
        """
        pytestmark = [falkor_skipif(), pytest.mark.timeout(300)]
        def test_live(): ...
        """,
        ['test_live'], ['test_live'],
        id='module-pytestmark-gate-alone',
    ),
    pytest.param(
        """
        @qdrant_skipif()
        class TestLive:
            def test_live(self): ...
        """,
        ['TestLive::test_live'], ['TestLive::test_live'],
        id='class-decorator-gate-alone',
    ),
    pytest.param(
        """
        class TestLive:
            @_fm_falkor_live.falkor_skipif()
            async def test_live(self): ...
            def test_unit(self): ...
        """,
        ['TestLive::test_live'], ['TestLive::test_live'],
        id='method-decorator-gate-alone',
    ),
    pytest.param(
        """
        pytestmark: list = [falkor_skipif(), pytest.mark.integration]
        def test_live(): ...
        """,
        ['test_live'], [],
        id='integration-in-module-pytestmark',
    ),
    pytest.param(
        """
        @pytest.mark.integration
        class TestOuter:
            class TestInner:
                @falkor_skipif()
                def test_live(self): ...
        """,
        ['TestOuter::TestInner::test_live'], [],
        id='integration-on-enclosing-class-decorator',
    ),
    pytest.param(
        """
        pytestmark = qdrant_skipif()
        class TestLive:
            pytestmark = pytest.mark.integration
            def test_live(self): ...
        """,
        ['TestLive::test_live'], [],
        id='integration-in-class-pytestmark',
    ),
    pytest.param(
        """
        pytestmark = qdrant_skipif()
        @pytest.mark.integration
        def test_live(): ...
        def test_other(): ...
        """,
        ['test_live', 'test_other'], ['test_other'],
        id='integration-on-function-decorator',
    ),
    pytest.param(
        """
        @falkor_skipif()
        @pytest.mark.parametrize('n', [pytest.param(1, marks=pytest.mark.integration)])
        def test_live(n): ...
        """,
        ['test_live'], ['test_live'],
        id='integration-only-on-a-param-variant',
    ),
    pytest.param(
        """
        pytestmark = [pytest.mark.integration]
        pytestmark = [falkor_skipif()]
        def test_live(): ...
        """,
        ['test_live'], ['test_live'],
        id='rebound-module-pytestmark-drops-integration',
    ),
    pytest.param(
        """
        @pytest.mark.skipif(True, reason='qdrant_skipif')
        def test_unit(): ...
        @falkor_skipif()
        def helper(): ...
        class Helper:
            @falkor_skipif()
            def test_live(self): ...
        """,
        [], [],
        id='ungated-and-non-test-items',
    ),
])
def test_the_scan_places_marks_as_pytest_applies_them(
    source: str, expected_gated: list[str], expected_not_integration: list[str],
) -> None:
    tree = ast.parse(textwrap.dedent(source))

    assert scan_live_gated_tests(tree) == (expected_gated, expected_not_integration)
