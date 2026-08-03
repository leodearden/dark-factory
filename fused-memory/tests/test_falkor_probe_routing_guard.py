"""Migration guard: every live-FalkorDB test module must route through the shared
``_fm_helpers`` reachability scaffolding — and may not re-fork it.

Task 3502. The ``_falkor_available()`` probe plus its ``FALKOR_HOST`` /
``FALKOR_PORT`` env reads had been copy-pasted, **byte-identical modulo
docstring**, into six test modules. Every copy used the same
``socket_connect_timeout=2``, the same ``select_graph('_probe').query('RETURN 1')``,
the same suppressed ``close()`` in a ``finally``, and the same bare
``except Exception: return False``. Three of the six docstrings literally read
"mirrors test_merge_entities.py" / "mirrors test_list_indices_integration.py" —
the copy-paste was self-documented.

The hazard here is **maintenance drift**, not false green (that is the distinct
concern of tests/test_falkor_index_barrier_guard.py, task 3377, which guards a
narrower two-module set: those that create an index AND drive a live graph).
Six independent copies of a connection probe means a fix to the timeout, the
probe query, or the connection-leak handling lands in one file and silently
misses five. This guard converts "remember not to re-fork the probe" into a
checked invariant.

The module set is **DISCOVERED, not hand-listed**. A hand-maintained literal
would only pin the six modules already migrated; the seventh module that
copy-pastes a live-FalkorDB fixture — precisely the failure this guard exists
to catch — would be invisible to it. So :func:`_discover_live_falkor_modules`
walks every ``test_*.py`` under this directory and selects the ones that drive
a live FalkorDB graph.

Selection criterion (broader than 3377's, which additionally requires index
creation): the module contains a real **call** to ``select_graph(...)``. That
is an AST fact, so prose that merely *mentions* a live graph — e.g.
test_integration_marker_real_service.py's docstring, which discusses
``_falkor_available`` at length — cannot drag an unrelated module into scope,
nor satisfy or trip any clause below.

A cheap ``'select_graph' in source`` text prefilter runs before any parsing so
discovery stays fast: an ``ast.Call`` to ``select_graph`` cannot exist without
that identifier appearing literally in the source, so the prefilter is a strict
superset of the criterion and cannot hide a module. Only the survivors are
parsed. The prefilter narrows the candidate set; every assertion is still
AST-based.

Discovery that silently found nothing would be a vacuous guard, so
:data:`VERIFIED_LIVE_FALKOR_MODULES` is a floor: :class:`TestDiscoveryItself`
asserts the discovered set still contains all six modules verified by hand. A
discriminator broken by a FalkorDB API rename therefore fails loudly instead of
quietly passing zero modules.

Note the deliberate asymmetry between the two kinds of clause:

* The **no-re-fork** clauses apply to *every* discovered module, including ones
  added later. Re-forking the probe is exactly how this duplication arose, and
  a copy-pasted fixture carries the fork along with it.
* The **routing** clause applies only to :data:`VERIFIED_LIVE_FALKOR_MODULES`.
  A future live-FalkorDB module may legitimately gate reachability some other
  way (a fixture-level check, an autouse skip); forcing it to import a specific
  name would over-constrain it. What it may *not* do is fork the probe.

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

TESTS_ROOT = pathlib.Path(__file__).parent
SELF = pathlib.Path(__file__).resolve()

HELPERS_MODULE = '_fm_helpers'

# The names a migrated module may import to satisfy the routing clause. Any one
# of them proves the module reaches into the shared scaffolding rather than
# rolling its own.
SHARED_SCAFFOLDING = {'falkor_skipif', 'FALKOR_HOST', 'FALKOR_PORT'}

# The probe's private spelling, plus the public spelling a well-meaning
# "promote it to public" edit would reach for. Neither may live in a test module.
LOCAL_PROBE_FORKS = {'_falkor_available', 'falkor_available'}

# The env-read half of the fork: module-level connection constants.
LOCAL_CONSTANT_FORKS = {'FALKOR_HOST', 'FALKOR_PORT'}

# The criterion: the module drives a live FalkorDB graph. Also the text
# prefilter — see the module docstring for why that is safe.
_LIVE_GRAPH_CALL = 'select_graph'

# The floor: modules verified by hand (task 3502) to carry the forked probe.
# Discovery must keep finding at least these; it is free to find more, and
# finding more is the point.
VERIFIED_LIVE_FALKOR_MODULES = {
    'test_falkor_fulltext_integration.py',
    'test_list_indices_integration.py',
    'test_merge_entities.py',
    'test_reassign_edge.py',
    'test_refresh_entity_summary.py',
    'test_startup_identity_scan.py',
}


def _parse(path: pathlib.Path) -> ast.Module:
    assert path.exists(), f'{path} not found'
    return ast.parse(path.read_text(), filename=str(path))


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


def _discover_live_falkor_modules() -> list[pathlib.Path]:
    """Every test module that drives a live FalkorDB graph.

    Derived rather than hand-listed so a newly-added live-FalkorDB module is
    covered automatically — see the module docstring.
    """
    found: list[pathlib.Path] = []
    for path in sorted(TESTS_ROOT.glob('**/test_*.py')):
        if path.resolve() == SELF:
            # This guard names the tokens in prose; it drives no graph.
            continue
        if _LIVE_GRAPH_CALL not in path.read_text():
            continue  # cheap prefilter — strict superset of the AST criterion
        if _calls_named(_parse(path), _LIVE_GRAPH_CALL):
            found.append(path)
    return found


LIVE_FALKOR_MODULES = _discover_live_falkor_modules()

# The routing clause runs against the hand-verified six only (see the module
# docstring's "deliberate asymmetry" note), resolved to paths so failures point
# at a file.
VERIFIED_MODULE_PATHS = [
    TESTS_ROOT / name for name in sorted(VERIFIED_LIVE_FALKOR_MODULES)
]


class TestDiscoveryItself:
    """The discriminator must keep finding the modules we verified by hand.

    Without this floor a guard whose criteria silently stopped matching
    (FalkorDB renames ``select_graph``; the fixtures move to a helper) would
    parametrize over an empty set and report green having checked nothing.
    """

    def test_discovery_finds_the_verified_live_falkor_modules(self):
        discovered = {p.name for p in LIVE_FALKOR_MODULES}
        missing = VERIFIED_LIVE_FALKOR_MODULES - discovered
        assert not missing, (
            f'live-FalkorDB module discovery no longer finds {sorted(missing)} '
            f'(found {sorted(discovered) or "nothing"}). The selection criterion in '
            f'this file has gone stale — most likely FalkorDB or the fixtures '
            f'renamed {_LIVE_GRAPH_CALL!r}. Fix the criterion; do NOT shrink the '
            f'floor, or this guard silently checks nothing.'
        )


@pytest.mark.parametrize('path', LIVE_FALKOR_MODULES, ids=lambda p: p.name)
class TestNoModuleReForksTheProbe:
    """No live-FalkorDB module may re-fork the shared reachability scaffolding.

    Applies uniformly to every discovered module, including ones added after
    this task: re-forking is exactly how the duplication arose, and a
    copy-pasted fixture carries the fork along with it.
    """

    def test_does_not_define_the_probe_locally(self, path):
        tree = _parse(path)
        defs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in LOCAL_PROBE_FORKS
        ]
        assert not defs, (
            f'{path.name}: defines {[d.name for d in defs]} locally at line(s) '
            f'{[d.lineno for d in defs]}. The FalkorDB reachability probe lives in '
            f'{HELPERS_MODULE}._falkor_available — six byte-identical copies of it '
            f'is how the duplication this guard exists to prevent arose in the '
            f'first place (task 3502 consolidated them). Import '
            f'{HELPERS_MODULE}.falkor_skipif() and use it as the skip marker '
            f'instead of hand-rolling the probe.'
        )

    def test_does_not_read_falkor_env_at_module_scope(self, path):
        """The env-read half of the fork must not come back either.

        A module that keeps its own ``FALKOR_HOST = os.environ.get(...)`` has
        re-forked half the scaffolding even if it imports the skip marker: the
        probe and the fixture would then be able to disagree about which
        FalkorDB they are talking to.
        """
        tree = _parse(path)
        offenders: list[tuple[str, int]] = []
        for node in tree.body:  # module scope only, not nested assignments
            if isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.Assign):
                targets = node.targets
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id in LOCAL_CONSTANT_FORKS:
                    offenders.append((target.id, node.lineno))
        assert not offenders, (
            f'{path.name}: assigns {offenders} at module scope. '
            f'{sorted(LOCAL_CONSTANT_FORKS)} are shared constants in '
            f'{HELPERS_MODULE} (task 3502) — import them instead of re-reading the '
            f'environment. A local copy lets this module\'s fixture and the shared '
            f'probe disagree about which FalkorDB they are talking to.'
        )


@pytest.mark.parametrize('path', VERIFIED_MODULE_PATHS, ids=lambda p: p.name)
class TestVerifiedModulesRouteThroughHelpers:
    """The six migrated modules must actually import the shared scaffolding.

    Scoped to the hand-verified six rather than all discovered modules: a
    future live-FalkorDB module may legitimately gate reachability some other
    way, and forcing it to import a specific name would over-constrain it.
    Dropping the fork without picking up the shared helper, however, would mean
    a module lost its skip guard entirely — which this clause catches.
    """

    def test_imports_shared_falkor_scaffolding(self, path):
        tree = _parse(path)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == HELPERS_MODULE:
                imported.update(alias.name for alias in node.names)
        assert imported & SHARED_SCAFFOLDING, (
            f'{path.name}: imports {sorted(imported) or "nothing"} from '
            f'{HELPERS_MODULE}, but this module drives a live FalkorDB graph and so '
            f'must route through the shared scaffolding — at least one of '
            f'{sorted(SHARED_SCAFFOLDING)}. Without it the module either re-forked '
            f'the probe or lost its reachability skip guard, which would make its '
            f'live assertions fail (rather than skip) wherever FalkorDB is absent.'
        )
