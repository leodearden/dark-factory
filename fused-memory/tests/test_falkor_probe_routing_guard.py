"""Migration guard: the six live-FalkorDB test modules migrated by task 3502
must route through the shared ``_fm_helpers`` reachability scaffolding.

Before this task, the ``_falkor_available()`` probe plus its ``FALKOR_HOST`` /
``FALKOR_PORT`` env reads had been copy-pasted, byte-identical modulo docstring,
into six test modules. Task 3502 consolidated them into ``_fm_helpers``
alongside the pre-existing ``_qdrant_available`` / ``qdrant_skipif`` pair, and
exposed ``falkor_skipif()`` mirroring ``qdrant_skipif()``.

This guard encodes the task's acceptance criterion for the six migrated
modules, in the shape of the accepted sibling precedent
tests/test_gather_idiom_helper_routing.py: each module must apply the shared
skip marker, must not re-declare the probe locally, and must source its
scaffolding from ``_fm_helpers``.

The load-bearing clause is :meth:`test_applies_the_shared_skip_marker`, which
guards a real runtime property: a module that loses its ``falkor_skipif()``
call still imports FALKOR_HOST / FALKOR_PORT for its fixture, so it goes on
looking migrated while every live test ERRORS instead of skipping on a
FalkorDB-less machine.

Scoped to EXACTLY the six modules this task migrated — deliberately a fixed
list, not a discovered set. A future live-FalkorDB module may legitimately gate
reachability some other way (a fixture-level check, an autouse skip), and this
guard must not pre-empt that choice.

AST (not string grep) so prose that merely *mentions* the scaffolding — e.g.
test_integration_marker_real_service.py's docstring, which discusses
``_falkor_available`` at length — cannot satisfy or trip a clause.

NOT integration-marked: this file only parses source, so it must run in the
default ``-m 'not integration'`` lane with no FalkorDB — the configuration
least able to notice the regression it prevents. Mirrors
tests/test_falkor_index_barrier_guard.py and
tests/test_gather_idiom_helper_routing.py.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from _fm_helpers import calls_named, parse_test_module

TESTS_ROOT = pathlib.Path(__file__).parent

HELPERS_MODULE = '_fm_helpers'

# The marker factory a migrated module applies — as `pytestmark`, or as a class
# or function decorator. Calling it is what actually gates the live tests.
SKIP_MARKER = 'falkor_skipif'

# The names a migrated module may import to satisfy the import clause.
SHARED_SCAFFOLDING = {SKIP_MARKER, 'FALKOR_HOST', 'FALKOR_PORT'}

# The probe's spelling in the six pre-migration copies. A copier is free to
# rename it; the load-bearing skip-marker clause below is what actually holds.
LOCAL_PROBE_FORK = '_falkor_available'

MIGRATED_MODULES = [
    TESTS_ROOT / name
    for name in (
        'test_falkor_fulltext_integration.py',
        'test_list_indices_integration.py',
        'test_merge_entities.py',
        'test_reassign_edge.py',
        'test_refresh_entity_summary.py',
        'test_startup_identity_scan.py',
    )
]


@pytest.mark.parametrize('path', MIGRATED_MODULES, ids=lambda p: p.name)
class TestMigratedModulesRouteThroughHelpers:
    """The six migrated modules gate on, and source, the shared scaffolding."""

    def test_applies_the_shared_skip_marker(self, path):
        """The module must CALL falkor_skipif() — importing it gates nothing.

        This is the clause that catches a lost skip guard. Deleting the
        ``@falkor_skipif()`` decorator (or the ``falkor_skipif()`` entry from a
        module's ``pytestmark``) leaves the module still importing FALKOR_HOST /
        FALKOR_PORT for its fixture, so the import clause below stays green
        while every live test errors instead of skipping on a FalkorDB-less
        machine. Only the call proves the gate is applied.
        """
        calls = calls_named(parse_test_module(path), SKIP_MARKER)
        assert calls, (
            f'{path.name}: never calls {SKIP_MARKER}(). This module drives a live '
            f'FalkorDB graph, so it must be gated on reachability — as '
            f'`pytestmark = [..., {SKIP_MARKER}(), ...]` or as a class/function '
            f'decorator. Without the call its live assertions FAIL rather than '
            f'skip wherever FalkorDB is absent.'
        )

    def test_does_not_define_the_probe_locally(self, path):
        tree = parse_test_module(path)
        defs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == LOCAL_PROBE_FORK
        ]
        assert not defs, (
            f'{path.name}: re-defines {LOCAL_PROBE_FORK} locally at line(s) '
            f'{[d.lineno for d in defs]}. The FalkorDB reachability probe lives in '
            f'{HELPERS_MODULE}.{LOCAL_PROBE_FORK} — six byte-identical copies of it '
            f'is the duplication task 3502 consolidated. Import '
            f'{HELPERS_MODULE}.{SKIP_MARKER}() and use it as the skip marker '
            f'instead of hand-rolling the probe.'
        )

    def test_imports_shared_falkor_scaffolding(self, path):
        tree = parse_test_module(path)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == HELPERS_MODULE:
                imported.update(alias.name for alias in node.names)
        assert imported & SHARED_SCAFFOLDING, (
            f'{path.name}: imports {sorted(imported) or "nothing"} from '
            f'{HELPERS_MODULE}, but this module drives a live FalkorDB graph and so '
            f'must source its scaffolding there — at least one of '
            f'{sorted(SHARED_SCAFFOLDING)}. Importing none of them means the '
            f'connection settings or the skip marker came from somewhere else, '
            f'which is how six byte-identical forks accumulated (task 3502).'
        )
