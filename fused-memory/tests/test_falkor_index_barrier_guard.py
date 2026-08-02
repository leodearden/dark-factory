"""Migration guard: every live-index test module must route through the shared
``_fm_helpers.await_index_operational`` barrier — and may not re-fork it.

Task 3377. FalkorDB builds indices **asynchronously**, so a test that queries
an index immediately after creating it can silently succeed for a query the
engine would otherwise reject. Measured while characterising task 3334: a
known-unparseable fulltext query issued right after
``CALL db.idx.fulltext.createNodeIndex(...)`` across six fresh graphs gave
FAIL, OK, OK, FAIL, FAIL, FAIL — **2 of 6 runs falsely reported success**.
This is not fulltext-specific: RANGE indices build asynchronously too
(measured on a 200k-node graph, ``CALL db.indexes()`` read
``'[Indexing] N/200000: UNDER CONSTRUCTION'`` for 40 consecutive polls without
ever reaching ``OPERATIONAL``).

Task 3334 fixed the one call site by hand-rolling the barrier inside
test_falkor_fulltext_integration.py. The harm this guard addresses is
PROPAGATION: those fixtures are the repo's copy-paste template for
live-FalkorDB tests, so a per-file fix cures today's instance but not the
mechanism. The guard converts "remember to add the barrier" into a checked
invariant.

AST (not string grep) so a docstring or comment that merely *describes* the
barrier — e.g. "see _fm_helpers.await_index_operational" — cannot satisfy the
check; only real ImportFrom / Call / FunctionDef nodes count. This mirrors
tests/test_gather_idiom_helper_routing.py.

Scoped to EXACTLY the two modules that create indices on a live async graph,
verified to be the complete set by intersecting index-creating files
(``CREATE INDEX`` / ``createNodeIndex`` / ``db.indexes()``) with live-graph
fixture files. The other ``select_graph`` users (test_startup_identity_scan,
test_reassign_edge, test_merge_entities, test_refresh_entity_summary,
test_falkor_fulltext_query) build no index and are deliberately NOT forced to
adopt a barrier they do not need. When a third live-index module appears,
widening ``LIVE_INDEX_MODULES`` is the intended action.

NOT integration-marked: this file only parses source, so it must run in the
default ``-m 'not integration'`` lane with no FalkorDB — the configuration
least able to notice the regression the barrier prevents.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

TESTS_ROOT = pathlib.Path(__file__).parent

HELPERS_MODULE = '_fm_helpers'
BARRIER = 'await_index_operational'
# The pre-extraction private spelling (task 3334) must not come back either.
LOCAL_FORKS = {BARRIER, f'_{BARRIER}'}

LIVE_INDEX_MODULES = [
    TESTS_ROOT / 'test_falkor_fulltext_integration.py',
    TESTS_ROOT / 'test_list_indices_integration.py',
]


def _parse(path: pathlib.Path) -> ast.Module:
    assert path.exists(), f'{path} not found'
    return ast.parse(path.read_text(), filename=str(path))


def _names_imported_from(tree: ast.Module, module: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def _calls_named(tree: ast.Module, name: str) -> list[ast.Call]:
    """Every ast.Call whose callee is a bare ``name(...)`` or ends in ``.name(...)``."""
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == name) or (
            isinstance(func, ast.Attribute) and func.attr == name
        ):
            found.append(node)
    return found


def _local_defs(tree: ast.Module, names: set[str]) -> list[ast.stmt]:
    """Every (async) function definition in *tree* whose name is in *names*."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]


@pytest.mark.parametrize('path', LIVE_INDEX_MODULES, ids=lambda p: p.name)
class TestLiveIndexModulesUseSharedBarrier:
    """Both live-index modules import, call, and do not re-fork the shared barrier."""

    def test_imports_the_shared_barrier(self, path):
        tree = _parse(path)
        imported = _names_imported_from(tree, HELPERS_MODULE)
        assert BARRIER in imported, (
            f'{path.name}: imports {sorted(imported) or "nothing"} from {HELPERS_MODULE}, '
            f'but this module creates an index on a live FalkorDB graph and so must '
            f'import the shared {BARRIER} barrier. FalkorDB builds indices '
            f'asynchronously; without the barrier this module is a false-green '
            f'generator (measured: 2 of 6 unbarriered runs falsely reported success).'
        )

    def test_actually_calls_the_barrier(self, path):
        tree = _parse(path)
        calls = _calls_named(tree, BARRIER)
        assert calls, (
            f'{path.name}: imports the barrier but never calls {BARRIER}(...). '
            f'The call belongs in the live-graph fixture, AFTER index creation and '
            f'BEFORE `yield graph` — importing it alone gates nothing.'
        )

    def test_does_not_redefine_the_barrier_locally(self, path):
        tree = _parse(path)
        defs = _local_defs(tree, LOCAL_FORKS)
        assert not defs, (
            f'{path.name}: defines {[d.name for d in defs]} locally at line(s) '
            f'{[d.lineno for d in defs]}. The barrier lives in '
            f'{HELPERS_MODULE}.{BARRIER} — re-forking it into a test module is how '
            f'the duplication this guard exists to prevent arose in the first place '
            f'(task 3334 hand-rolled it here; task 3377 extracted it). Import the '
            f'shared helper instead.'
        )
