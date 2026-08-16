"""Migration guard: the six live-FalkorDB test modules migrated by task 3502
must apply the shared ``_fm_helpers.falkor_skipif()`` reachability gate.

Task 3502 consolidated a ``_falkor_available()`` probe that had been
copy-pasted, byte-identical modulo docstring, into six test modules, and
exposed ``falkor_skipif()`` mirroring the pre-existing ``qdrant_skipif()``.

This guard asserts ONE property, and it is a runtime one: each migrated module
still gates its live tests on FalkorDB reachability. A module that loses its
gate goes on importing FALKOR_HOST / FALKOR_PORT for its fixture, so it keeps
looking migrated while every live test ERRORS instead of skipping wherever
FalkorDB is absent — a failure that reads as a broken test suite rather than as
a missing service.

Only a gate in a **gating position** counts: an entry in the module's
``pytestmark``, or a decorator on a class or function. A bare
``falkor_skipif()`` call elsewhere in the file — inside a helper body, bound to
an unused local — collects the marker and applies it to nothing, so it does not
satisfy this guard.

Deliberately NOT asserted: that the probe is absent under some other name, or
which identifiers the module imports from ``_fm_helpers``. Both are name-pinning
over sibling test source — a re-fork called ``_falkordb_up`` evades them, so
they buy no coverage while giving false assurance that duplication cannot
return (review esc-3502-2).

AST (not string grep) so prose that merely *mentions* the scaffolding — e.g.
test_integration_marker_real_service.py's docstring, which discusses
``_falkor_available`` at length — cannot satisfy or trip the check.

Scoped to EXACTLY the six modules this task migrated — a fixed list, not a
discovered set. A future live-FalkorDB module may legitimately gate reachability
some other way (a fixture-level check, an autouse skip), and this guard must not
pre-empt that choice.

NOT integration-marked: this file only parses source, so it must run in the
default ``-m 'not integration'`` lane with no FalkorDB — the configuration least
able to notice the regression it prevents. Mirrors
tests/test_falkor_index_barrier_guard.py and
tests/test_gather_idiom_helper_routing.py.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from _fm_helpers import calls_named, parse_python_module

TESTS_ROOT = pathlib.Path(__file__).parent

# The marker factory a migrated module applies. Calling it in a gating position
# is what actually skips the live tests when FalkorDB is unreachable.
SKIP_MARKER = 'falkor_skipif'

# The module-level name pytest collects markers from.
PYTESTMARK = 'pytestmark'

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


def _gating_skip_marker_calls(tree: ast.Module) -> list[ast.Call]:
    """Every ``falkor_skipif()`` call in a position pytest actually applies.

    Two such positions, which is what the six modules between them use: an
    entry in a ``pytestmark`` assignment, and a decorator on a class or
    function. Anywhere else the call merely constructs a marker object and
    drops it.
    """
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                found.extend(calls_named(decorator, SKIP_MARKER))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(
                isinstance(target, ast.Name) and target.id == PYTESTMARK
                for target in targets
            ):
                continue
            if node.value is not None:
                found.extend(calls_named(node.value, SKIP_MARKER))
    return found


@pytest.mark.parametrize('path', MIGRATED_MODULES, ids=lambda p: p.name)
def test_applies_the_shared_skip_marker(path):
    """The module must apply ``falkor_skipif()`` where pytest will honour it."""
    calls = _gating_skip_marker_calls(parse_python_module(path))
    assert calls, (
        f'{path.name}: never applies {SKIP_MARKER}() in a gating position. This '
        f'module drives a live FalkorDB graph, so it must be gated on '
        f'reachability — as `{PYTESTMARK} = [..., {SKIP_MARKER}(), ...]` or as a '
        f'class/function decorator. Without that its live assertions ERROR rather '
        f'than skip wherever FalkorDB is absent. Note that calling {SKIP_MARKER}() '
        f'anywhere else builds a marker and discards it: only pytestmark and '
        f'decorator positions count.'
    )
