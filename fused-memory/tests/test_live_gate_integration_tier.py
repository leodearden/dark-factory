"""Every test a live-service reachability gate applies to is in the integration tier.

The default `-m 'not integration'` suite is the merge gate, which cannot tell a backing-store
outage from a test failure; the offline `-m integration` lane can (task 11484). Per-module
behavioral guards for the same rule: tests/test_integration_marker_real_service.py.

The scan is static. It sees a gate only where a `LIVE_GATE_FACTORIES` call is spelled in a
test's own marks, so an aliased gate factory and live access through a fixture are invisible.
It does not follow class inheritance: a test method is read only in the `Test*` class that
defines it, with that class's marks, never in a subclass that inherits it.
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import pytest
from shared.pytest_mark_grammar import marker_name, pytestmark_elements
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

TESTS_DIR = Path(__file__).parent

LIVE_GATE_FACTORIES = frozenset({'falkor_skipif', 'qdrant_skipif'})


def _live_gate_factory(mark: ast.expr) -> str | None:
    match mark:
        case ast.Call(func=ast.Name(id=name)) | ast.Call(func=ast.Attribute(attr=name)):
            return name if name in LIVE_GATE_FACTORIES else None
    return None


def _test_items(
    body: Sequence[ast.stmt], applied: list[ast.expr], prefix: str,
) -> Iterator[tuple[str, list[ast.expr]]]:
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name.startswith('test'):
            yield f'{prefix}{stmt.name}', [*applied, *stmt.decorator_list]
        elif isinstance(stmt, ast.ClassDef) and stmt.name.startswith('Test'):
            class_marks = [*applied, *stmt.decorator_list, *pytestmark_elements(stmt.body)]
            yield from _test_items(stmt.body, class_marks, f'{prefix}{stmt.name}::')


class LiveGates(NamedTuple):
    gated: tuple[str, ...]
    not_integration: tuple[str, ...]
    factories: frozenset[str]


def scan_live_gated_module(tree: ast.Module) -> LiveGates:
    """Name each live-gated test `Class::test`, those not in the integration tier, and the gates used."""
    gated: list[str] = []
    not_integration: list[str] = []
    factories: set[str] = set()
    for suffix, marks in _test_items(tree.body, pytestmark_elements(tree.body), ''):
        item_factories = {factory for mark in marks if (factory := _live_gate_factory(mark))}
        if item_factories:
            factories |= item_factories
            gated.append(suffix)
            if not any(marker_name(mark) == 'integration' for mark in marks):
                not_integration.append(suffix)
    return LiveGates(tuple(gated), tuple(not_integration), frozenset(factories))


class LiveGateScan(NamedTuple):
    examined: int
    gates: LiveGates


def scan_live_gated_tree(root: Path) -> LiveGateScan:
    """Scan every `test_*.py` under *root*, naming each item `<relpath>::Class::test`."""
    paths = sorted(root.rglob('test_*.py'))
    gated: list[str] = []
    not_integration: list[str] = []
    factories: set[str] = set()
    for path in paths:
        source = path.read_text(encoding='utf-8')
        if not any(name in source for name in LIVE_GATE_FACTORIES):  # a superset of _live_gate_factory
            continue
        relpath = path.relative_to(root).as_posix()
        module = scan_live_gated_module(ast.parse(source, filename=str(path)))
        gated.extend(f'{relpath}::{suffix}' for suffix in module.gated)
        not_integration.extend(f'{relpath}::{suffix}' for suffix in module.not_integration)
        factories |= module.factories
    return LiveGateScan(
        len(paths), LiveGates(tuple(gated), tuple(not_integration), frozenset(factories)),
    )


def test_every_live_gated_test_is_in_the_integration_tier() -> None:
    scan = scan_live_gated_tree(TESTS_DIR)

    unused_factories = sorted(LIVE_GATE_FACTORIES - scan.gates.factories)
    assert not unused_factories, (
        f'{unused_factories} gate no test in the {scan.examined} test modules under {TESTS_DIR}; '
        'a renamed or aliased gate factory leaves the scan blind to every test it gates'
    )
    assert not scan.gates.not_integration, (
        'these tests apply a live-service reachability gate but run in the default '
        "-m 'not integration' suite, where a backing-store outage reds the merge gate. "
        'Add @pytest.mark.integration to each listed test itself. Mark its class or the module '
        'pytestmark only when every test in that scope is live: otherwise the mark also drops '
        "the scope's ungated unit tests from the merge gate.\n"
        + '\n'.join(scan.gates.not_integration)
    )


def test_the_tree_scan_reads_every_test_module_and_names_items_by_relpath(tmp_path: Path) -> None:
    (tmp_path / 'nested').mkdir()
    (tmp_path / 'test_flat.py').write_text(
        '@falkor_skipif()\ndef test_live(): ...\n', encoding='utf-8',
    )
    (tmp_path / 'nested' / 'test_deep.py').write_text(
        'pytestmark = [qdrant_skipif(), pytest.mark.integration]\ndef test_live(): ...\n',
        encoding='utf-8',
    )
    (tmp_path / 'test_ungated.py').write_text('def test_unit(): ...\n', encoding='utf-8')
    (tmp_path / 'helpers.py').write_text(
        '@falkor_skipif()\ndef test_live(): ...\n', encoding='utf-8',
    )

    assert scan_live_gated_tree(tmp_path) == LiveGateScan(
        examined=3,
        gates=LiveGates(
            gated=('nested/test_deep.py::test_live', 'test_flat.py::test_live'),
            not_integration=('test_flat.py::test_live',),
            factories=frozenset({'falkor_skipif', 'qdrant_skipif'}),
        ),
    )


@pytest.mark.parametrize(('source', 'expected'), [
    pytest.param(
        """
        pytestmark = [falkor_skipif(), pytest.mark.timeout(300)]
        def test_live(): ...
        """,
        LiveGates(('test_live',), ('test_live',), frozenset({'falkor_skipif'})),
        id='module-pytestmark-gate-alone',
    ),
    pytest.param(
        """
        @qdrant_skipif()
        class TestLive:
            def test_live(self): ...
        """,
        LiveGates(('TestLive::test_live',), ('TestLive::test_live',), frozenset({'qdrant_skipif'})),
        id='class-decorator-gate-alone',
    ),
    pytest.param(
        """
        class TestLive:
            @_fm_falkor_live.falkor_skipif()
            async def test_live(self): ...
            def test_unit(self): ...
        """,
        LiveGates(('TestLive::test_live',), ('TestLive::test_live',), frozenset({'falkor_skipif'})),
        id='method-decorator-gate-alone',
    ),
    pytest.param(
        """
        pytestmark: list = [falkor_skipif(), pytest.mark.integration]
        def test_live(): ...
        """,
        LiveGates(('test_live',), (), frozenset({'falkor_skipif'})),
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
        LiveGates(('TestOuter::TestInner::test_live',), (), frozenset({'falkor_skipif'})),
        id='integration-on-enclosing-class-decorator',
    ),
    pytest.param(
        """
        pytestmark = qdrant_skipif()
        class TestLive:
            pytestmark = pytest.mark.integration
            def test_live(self): ...
        """,
        LiveGates(('TestLive::test_live',), (), frozenset({'qdrant_skipif'})),
        id='integration-in-class-pytestmark',
    ),
    pytest.param(
        """
        pytestmark = qdrant_skipif()
        @pytest.mark.integration
        def test_live(): ...
        def test_other(): ...
        """,
        LiveGates(('test_live', 'test_other'), ('test_other',), frozenset({'qdrant_skipif'})),
        id='integration-on-function-decorator',
    ),
    pytest.param(
        """
        @falkor_skipif()
        @pytest.mark.parametrize('n', [pytest.param(1, marks=pytest.mark.integration)])
        def test_live(n): ...
        """,
        LiveGates(('test_live',), ('test_live',), frozenset({'falkor_skipif'})),
        id='integration-only-on-a-param-variant',
    ),
    pytest.param(
        """
        pytestmark = [pytest.mark.integration]
        pytestmark = [falkor_skipif()]
        def test_live(): ...
        """,
        LiveGates(('test_live',), ('test_live',), frozenset({'falkor_skipif'})),
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
        LiveGates((), (), frozenset()),
        id='ungated-and-non-test-items',
    ),
])
def test_the_scan_places_marks_as_pytest_applies_them(source: str, expected: LiveGates) -> None:
    tree = ast.parse(textwrap.dedent(source))

    assert scan_live_gated_module(tree) == expected
