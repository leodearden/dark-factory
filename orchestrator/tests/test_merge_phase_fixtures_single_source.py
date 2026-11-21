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
from _tree_scan import (
    TESTS_DIR,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
    python_files,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]


def _imported_modules(tree: ast.Module) -> tuple[str, ...]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    return tuple(imported)


def _top_level_modules() -> tuple[Path, ...]:
    return tuple(
        path for path in python_files(TESTS_DIR)
        if path.parent == TESTS_DIR and path.name != Path(__file__).name
    )


def _imports_if_a_consumer(module: ParsedModule) -> tuple[str, ...]:
    """The modules *module* imports when one is the support module, else ``()``."""
    if not module.may_name(fixtures.__name__):
        return ()
    imported = _imported_modules(module.tree)
    return imported if fixtures.__name__ in imported else ()


ANALYSIS = TreeAnalysis(name=Path(__file__).stem, files=_top_level_modules, extract=_imports_if_a_consumer)
TREE_ANALYSES = (ANALYSIS,)


@pytest.fixture(scope='module')
def imports_by_consumer(tree_pass: TreePass) -> dict[str, tuple[str, ...]]:
    return {
        path.stem: imported
        for path, imported in tree_pass.facts(ANALYSIS).items()
        if imported
    }


def _names_shared_with_fixtures(consumer: ModuleType) -> list[str]:
    shared = vars(consumer).keys() & vars(fixtures).keys()
    return sorted(n for n in shared if not (n.startswith('__') and n.endswith('__')))


def test_the_scan_finds_the_support_modules_consumers(
    imports_by_consumer: dict[str, tuple[str, ...]],
) -> None:
    assert imports_by_consumer, (
        f'No module in {TESTS_DIR} imports {fixtures.__name__}, so every '
        'consumer check below asserts over nothing. Either the scan is broken '
        'or the support module is dead and should be deleted.'
    )


def test_each_consumer_binds_the_support_modules_objects(
    imports_by_consumer: dict[str, tuple[str, ...]],
) -> None:
    separate_by_consumer = {}
    for name in imports_by_consumer:
        consumer = importlib.import_module(name)
        # Identity, never equality: an equal re-paste passes a value comparison.
        separate = [
            n for n in _names_shared_with_fixtures(consumer)
            if getattr(consumer, n) is not getattr(fixtures, n)
        ]
        if separate:
            separate_by_consumer[name] = separate
    assert not separate_by_consumer, (
        f'{separate_by_consumer}: each consumer binds its own objects under '
        'names that orchestrator/tests/_merge_phase_fixtures.py also binds. '
        'That module is their one home; import them from there, or rename the local.'
    )


def test_no_consumer_imports_a_sibling_test_module(
    imports_by_consumer: dict[str, tuple[str, ...]],
) -> None:
    sibling_tests = {
        name: sorted(m for m in imported if m.startswith('test_'))
        for name, imported in imports_by_consumer.items()
    }
    offenders = {name: modules for name, modules in sibling_tests.items() if modules}
    assert not offenders, (
        f'{offenders}: each consumer imports from the listed test module(s). '
        'Import shared scaffolding from a _-prefixed support module instead.'
    )
