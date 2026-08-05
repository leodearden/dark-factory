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

The module set is **DISCOVERED, not hand-listed**. A hand-maintained literal
would only pin regression in the two already-fixed modules; the third module
that copy-pastes the fixture — precisely the failure this guard exists to
catch — would be invisible to it. So :func:`_discover_live_index_modules`
walks every ``test_*.py`` under this directory and selects the ones that both
create an index and drive a live async graph. Add such a module without the
barrier and this guard fails on the next run, with no edit here required.

Selection criterion (the intersection the task verified by hand):
  1. contains a **string constant** matching ``CREATE INDEX`` /
     ``createNodeIndex`` — i.e. it creates an index; and
  2. contains a real **call** to ``select_graph(...)`` — i.e. it drives a live
     FalkorDB graph rather than a mock.
Both are AST facts, so a comment or docstring that merely *mentions* index
creation cannot drag an unrelated module into scope. The other
``select_graph`` users (test_startup_identity_scan, test_reassign_edge,
test_merge_entities, test_refresh_entity_summary, test_falkor_fulltext_query)
build no index and are correctly excluded — they are not forced to adopt a
barrier they do not need.

A cheap ``'select_graph' in source`` text prefilter runs before any parsing so
discovery costs ~130ms rather than ~15s: an ``ast.Call`` to ``select_graph``
cannot exist without that identifier appearing literally in the source, so the
prefilter is a strict superset of criterion (2) and cannot hide a module. Only
the ~7 survivors are parsed. The prefilter narrows the candidate set; every
actual assertion below is still AST-based.

Discovery that silently found nothing would be a vacuous guard, so
:data:`VERIFIED_LIVE_INDEX_MODULES` is a floor: ``test_discovery_*`` asserts
the discovered set still contains the two modules this task verified by hand.
A discriminator broken by a FalkorDB API rename therefore fails loudly instead
of quietly passing zero modules.

AST (not string grep) for the assertions themselves so a docstring or comment
that merely *describes* the barrier — e.g. "see
_fm_helpers.await_index_operational" — cannot satisfy the check; only real
ImportFrom / Call / FunctionDef nodes count. The parse and node-search
machinery is shared with the sibling guards via ``_fm_helpers``
(``parse_python_module`` / ``calls_named`` / ``imported_names_from``), so the
three agree on those semantics by construction rather than by copy. The
selection criteria below stay local: they are this guard's own policy, and it
is their only consumer.

NOT integration-marked: this file only parses source, so it must run in the
default ``-m 'not integration'`` lane with no FalkorDB — the configuration
least able to notice the regression the barrier prevents.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest
from _fm_helpers import calls_named, imported_names_from, parse_python_module

TESTS_ROOT = pathlib.Path(__file__).parent
SELF = pathlib.Path(__file__).resolve()

HELPERS_MODULE = '_fm_helpers'
BARRIER = 'await_index_operational'
# The pre-extraction private spelling (task 3334) must not come back either.
LOCAL_FORKS = {BARRIER, f'_{BARRIER}'}

# Criterion (1): the module issues an index-creating statement. Matches both
# the range form (`CREATE INDEX FOR (n:Entity) ON (n.name)`) and the fulltext
# procedure (`CALL db.idx.fulltext.createNodeIndex(...)`).
_INDEX_CREATION = re.compile(r'CREATE\s+INDEX|createNodeIndex', re.IGNORECASE)
# Criterion (2): the module drives a live FalkorDB graph. Also the text
# prefilter — see the module docstring for why that is safe.
_LIVE_GRAPH_CALL = 'select_graph'

# The floor: modules verified by hand (task 3377) to create an index on a live
# async graph. Discovery must keep finding at least these; it is free to find
# more, and finding more is the point.
VERIFIED_LIVE_INDEX_MODULES = {
    'test_falkor_fulltext_integration.py',
    'test_list_indices_integration.py',
}


def _has_index_creating_literal(tree: ast.Module) -> bool:
    """True when some string CONSTANT in *tree* issues an index-creating statement.

    A constant, not raw text: a comment saying "we should CREATE INDEX here"
    must not pull a module into scope.
    """
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _INDEX_CREATION.search(node.value)
        for node in ast.walk(tree)
    )


def _discover_live_index_modules() -> list[pathlib.Path]:
    """Every test module that creates an index on a live async FalkorDB graph.

    Derived rather than hand-listed so a newly-added live-index module is
    covered automatically — see the module docstring.
    """
    found: list[pathlib.Path] = []
    for path in sorted(TESTS_ROOT.glob('**/test_*.py')):
        if path.resolve() == SELF:
            # This guard names the tokens in prose; it drives no graph.
            continue
        if _LIVE_GRAPH_CALL not in path.read_text():
            # Cheap prefilter — strict superset of criterion (2). Deliberately a
            # raw read rather than the memoised one behind parse_python_module:
            # this reads every test module but only ~7 survive to be parsed, so
            # caching here would retain all of them for the session to save a
            # handful of re-reads.
            continue
        tree = parse_python_module(path)
        if _has_index_creating_literal(tree) and calls_named(tree, _LIVE_GRAPH_CALL):
            found.append(path)
    return found


LIVE_INDEX_MODULES = _discover_live_index_modules()


class TestDiscoveryItself:
    """The discriminator must keep finding the modules we verified by hand.

    Without this floor a guard whose criteria silently stopped matching
    (FalkorDB renames ``select_graph``; the fixtures move to a helper) would
    parametrize over an empty set and report green having checked nothing.
    """

    def test_discovery_finds_the_verified_live_index_modules(self):
        discovered = {p.name for p in LIVE_INDEX_MODULES}
        missing = VERIFIED_LIVE_INDEX_MODULES - discovered
        assert not missing, (
            f'live-index module discovery no longer finds {sorted(missing)} '
            f'(found {sorted(discovered) or "nothing"}). The selection criteria in '
            f'this file have gone stale — most likely FalkorDB or the fixtures '
            f'renamed {_LIVE_GRAPH_CALL!r} or the index-creation syntax. Fix the '
            f'criteria; do NOT shrink the floor, or this guard silently checks '
            f'nothing.'
        )


@pytest.mark.parametrize('path', LIVE_INDEX_MODULES, ids=lambda p: p.name)
class TestLiveIndexModulesUseSharedBarrier:
    """Every discovered live-index module imports, calls, and does not re-fork the barrier."""

    def test_imports_the_shared_barrier(self, path):
        imported = imported_names_from(parse_python_module(path), HELPERS_MODULE)
        assert BARRIER in imported, (
            f'{path.name}: imports {sorted(imported) or "nothing"} from {HELPERS_MODULE}, '
            f'but this module creates an index on a live FalkorDB graph and so must '
            f'import the shared {BARRIER} barrier. FalkorDB builds indices '
            f'asynchronously; without the barrier this module is a false-green '
            f'generator (measured: 2 of 6 unbarriered runs falsely reported success).'
        )

    def test_actually_calls_the_barrier(self, path):
        calls = calls_named(parse_python_module(path), BARRIER)
        assert calls, (
            f'{path.name}: never calls {BARRIER}(...). The call belongs in the '
            f'live-graph fixture, AFTER index creation and BEFORE `yield graph` — '
            f'importing the barrier alone gates nothing.'
        )

    def test_does_not_redefine_the_barrier_locally(self, path):
        """Forward-looking: no module may re-fork the helper back into itself.

        This is a regression clause, not a claim that every module once had a
        fork — only test_falkor_fulltext_integration.py did (task 3334, removed
        by 3377). It applies uniformly because re-forking is exactly how the
        duplication arose the first time, and a copy-pasted fixture carries the
        fork along with it.
        """
        tree = parse_python_module(path)
        defs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in LOCAL_FORKS
        ]
        assert not defs, (
            f'{path.name}: defines {[d.name for d in defs]} locally at line(s) '
            f'{[d.lineno for d in defs]}. The barrier lives in '
            f'{HELPERS_MODULE}.{BARRIER} — re-forking it into a test module is how '
            f'the duplication this guard exists to prevent arose in the first place '
            f'(task 3334 hand-rolled it here; task 3377 extracted it). Import the '
            f'shared helper instead.'
        )
