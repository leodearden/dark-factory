"""Structural guard: the merge-phase scaffolding has ONE home (task 6780).

Every test module that imports orchestrator/tests/_merge_phase_fixtures.py is
found by that import, never listed by hand.  Each must bind that module's own
objects under the names it shares with it (SPOT, heuristic 11), and none may
use a sibling test module as an import source, so each test file makes sense
on its own (heuristic 13).
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

# A module import, not a name import: importing a fixture by name would
# register it in this test module.
import _merge_phase_fixtures as fixtures
import pytest

_TESTS_DIR = Path(__file__).parent


def _imported_modules(source: str) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    return imported


def _consumers() -> tuple[str, ...]:
    found = []
    for path in sorted(_TESTS_DIR.glob('*.py')):
        if path.name == Path(__file__).name:
            continue
        source = path.read_text()
        if fixtures.__name__ in source and fixtures.__name__ in _imported_modules(source):
            found.append(path.stem)
    return tuple(found)


CONSUMERS = _consumers()


def _names_shared_with_fixtures(consumer: ModuleType) -> list[str]:
    shared = vars(consumer).keys() & vars(fixtures).keys()
    return sorted(n for n in shared if not (n.startswith('__') and n.endswith('__')))


def test_the_scan_finds_the_support_modules_consumers() -> None:
    assert CONSUMERS, (
        f'No module in {_TESTS_DIR} imports {fixtures.__name__}, so every '
        'consumer check below is an empty parametrization. Either the scan '
        'is broken or the support module is dead and should be deleted.'
    )


@pytest.mark.parametrize('name', CONSUMERS)
def test_each_consumer_binds_the_support_modules_objects(name: str) -> None:
    consumer = importlib.import_module(name)
    # Identity, never equality: an equal re-paste passes a value comparison.
    separate = [
        n for n in _names_shared_with_fixtures(consumer)
        if getattr(consumer, n) is not getattr(fixtures, n)
    ]
    assert not separate, (
        f'{name} binds its own objects under {separate}, names that '
        'orchestrator/tests/_merge_phase_fixtures.py also binds. That module '
        'is their one home; import them from there, or rename the local.'
    )


@pytest.mark.parametrize('name', CONSUMERS)
def test_no_consumer_imports_a_sibling_test_module(name: str) -> None:
    imported = _imported_modules((_TESTS_DIR / f'{name}.py').read_text())
    sibling_tests = sorted(m for m in imported if m.startswith('test_'))
    assert not sibling_tests, (
        f'{name} imports from the test module(s) {sibling_tests}. Import '
        'shared scaffolding from a _-prefixed support module instead.'
    )
