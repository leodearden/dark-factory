"""Migration guard: every live-FalkorDB test module must route through the shared
``_fm_helpers`` reachability scaffolding — and may not re-fork it.

Task 3502. The ``_falkor_available()`` probe plus its ``FALKOR_HOST`` /
``FALKOR_PORT`` env reads had been copy-pasted, byte-identical modulo docstring,
into six test modules. Six independent copies of a connection probe means a fix
to the timeout, the probe query, or the connection-leak handling lands in one
file and silently misses five. This guard converts "remember not to re-fork the
probe" into a checked invariant.

The hazard is **maintenance drift**, not false green — that is the distinct
concern of tests/test_falkor_index_barrier_guard.py (task 3377), which guards
the narrower set of modules that create an index AND drive a live graph.

The module set is **DISCOVERED, not hand-listed**. A hand-maintained literal
would only pin the modules already migrated; the next module that copy-pastes a
live-FalkorDB fixture — precisely the failure this guard exists to catch — would
be invisible to it. Selection criterion (broader than 3377's, which additionally
requires index creation): the module contains a real **call** to
``select_graph(...)``. That is an AST fact, so prose that merely *mentions* a
live graph — e.g. test_integration_marker_real_service.py's docstring, which
discusses ``_falkor_available`` at length — cannot drag an unrelated module into
scope, nor satisfy or trip any clause below.

Discovery, parsing and the cheap text prefilter come from
``_fm_helpers.discover_test_modules`` / ``parse_test_module`` / ``calls_named``,
shared with the sibling guard: a de-duplication guard that forked its own
discovery machinery would reproduce, one level up, exactly the drift it exists
to prevent.

Discovery that silently found nothing would be a vacuous guard, so
:data:`VERIFIED_LIVE_FALKOR_MODULES` is a floor: :class:`TestDiscoveryItself`
asserts the discovered set still contains all six modules verified by hand. A
discriminator broken by a FalkorDB API rename therefore fails loudly instead of
quietly passing zero modules.

Note the deliberate asymmetry between the two kinds of clause:

* The **no-re-fork** clauses apply to *every* discovered module, including ones
  added later. Re-forking the probe is exactly how this duplication arose, and
  a copy-pasted fixture carries the fork along with it. They key on the fork's
  structural fingerprints — a FALKOR_* environment read, the ``'_probe'`` graph,
  the bounded-connect keyword — not only on the names the six copies happened to
  use, because a copier is free to rename ``_falkor_available`` to
  ``_falkordb_reachable`` and to move the env read inside a fixture body.
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
from _fm_helpers import calls_named, discover_test_modules, parse_test_module

TESTS_ROOT = pathlib.Path(__file__).parent
SELF = pathlib.Path(__file__).resolve()

HELPERS_MODULE = '_fm_helpers'

# The marker factory a migrated module applies — as `pytestmark`, or as a class
# or function decorator. Calling it is what actually gates the live tests.
SKIP_MARKER = 'falkor_skipif'

# The names a migrated module may import to satisfy the import clause.
SHARED_SCAFFOLDING = {SKIP_MARKER, 'FALKOR_HOST', 'FALKOR_PORT'}

# The probe's private spelling, plus the public spelling a well-meaning
# "promote it to public" edit would reach for. Neither may live in a test
# module. Names are the weakest of the no-re-fork clauses (a copier can rename);
# the structural clauses below cover what a rename would hide.
LOCAL_PROBE_FORKS = {'_falkor_available', 'falkor_available'}

# The env-read half of the fork: local connection constants.
LOCAL_CONSTANT_FORKS = {'FALKOR_HOST', 'FALKOR_PORT'}

# Structural fingerprints of the probe body, independent of what it is named.
FALKOR_ENV_PREFIX = 'FALKOR_'
ENV_READ_ATTRS = {'get', 'getenv'}  # os.environ.get / os.getenv / getenv
PROBE_GRAPH = '_probe'
CONNECT_TIMEOUT_KWARG = 'socket_connect_timeout'

# The criterion: the module drives a live FalkorDB graph. Also the text
# prefilter — an ast.Call to select_graph cannot exist without the identifier
# appearing literally, so the prefilter is a strict superset of the criterion
# and cannot hide a module.
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


def _drives_a_live_graph(tree: ast.Module) -> bool:
    return bool(calls_named(tree, _LIVE_GRAPH_CALL))


LIVE_FALKOR_MODULES = discover_test_modules(
    _drives_a_live_graph,
    # This guard names the guarded tokens in prose; it drives no graph.
    exclude=[SELF],
    text_prefilter=_LIVE_GRAPH_CALL,
)

# The routing clause runs against the hand-verified six only (see the module
# docstring's "deliberate asymmetry" note), resolved to paths so failures point
# at a file.
VERIFIED_MODULE_PATHS = [
    TESTS_ROOT / name for name in sorted(VERIFIED_LIVE_FALKOR_MODULES)
]


def _falkor_env_reads(tree: ast.Module) -> list[tuple[str, int]]:
    """Every read of a ``FALKOR_*`` environment variable, at any depth.

    Covers the call forms (``os.environ.get('FALKOR_PORT', …)``,
    ``os.getenv(…)``, a bare ``getenv(…)`` after ``from os import getenv``) and
    the subscript form (``os.environ['FALKOR_HOST']``). ``monkeypatch.setenv``
    is deliberately not matched — patching the variable is not reading it.
    """
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name)
                else None
            )
            if name not in ENV_READ_ATTRS or not node.args:
                continue
            key = node.args[0]
        elif isinstance(node, ast.Subscript):
            container = node.value
            is_environ = (
                isinstance(container, ast.Attribute) and container.attr == 'environ'
            ) or (isinstance(container, ast.Name) and container.id == 'environ')
            if not is_environ:
                continue
            key = node.slice
        else:
            continue
        if (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and key.value.startswith(FALKOR_ENV_PREFIX)
        ):
            found.add((key.value, node.lineno))
    return sorted(found)


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
        tree = parse_test_module(path)
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
            f'{HELPERS_MODULE}.{SKIP_MARKER}() and use it as the skip marker '
            f'instead of hand-rolling the probe.'
        )

    def test_does_not_bind_falkor_connection_constants(self, path):
        """No local ``FALKOR_HOST`` / ``FALKOR_PORT`` binding, at any depth.

        A module that keeps its own copy has re-forked half the scaffolding
        even if it imports the skip marker: the probe and the fixture would
        then be able to disagree about which FalkorDB they are talking to.
        Walks the whole tree, not just module scope — an assignment moved
        inside a fixture body is the same fork.
        """
        tree = parse_test_module(path)
        offenders: list[tuple[str, int]] = []
        for node in ast.walk(tree):
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
            f'{path.name}: binds {offenders}. '
            f'{sorted(LOCAL_CONSTANT_FORKS)} are shared constants in '
            f'{HELPERS_MODULE} (task 3502) — import them instead of re-declaring '
            f'them. A local copy lets this module\'s fixture and the shared probe '
            f'disagree about which FalkorDB they are talking to.'
        )

    def test_does_not_read_falkor_env(self, path):
        """No ``FALKOR_*`` environment read, under any variable name or scope.

        The naming clauses above are evadable by construction: a copier is free
        to call the probe ``_falkordb_reachable`` and to read the environment
        inside a fixture body rather than at module scope. The env read itself
        is the fingerprint that survives both, so it is asserted structurally —
        any ``os.environ.get('FALKOR_…')`` / ``os.getenv(…)`` /
        ``os.environ['FALKOR_…']`` anywhere in the tree.
        """
        reads = _falkor_env_reads(parse_test_module(path))
        assert not reads, (
            f'{path.name}: reads {reads} from the environment. The FalkorDB '
            f'connection settings are derived once in {HELPERS_MODULE} '
            f'(FALKOR_HOST / FALKOR_PORT) — import them. Re-reading the '
            f'environment here re-forks the half of the scaffolding that decides '
            f'*which* FalkorDB this module talks to, and it does so under a name '
            f'the other clauses in this guard cannot see.'
        )

    def test_does_not_reimplement_the_probe_body(self, path):
        """No hand-rolled reachability probe, whatever it is called.

        Two structural fingerprints of the probe body, both independent of the
        function's name: the ``'_probe'`` throwaway graph, and the bounded
        ``socket_connect_timeout`` connect that makes a probe a probe rather
        than a real client. A fixture that genuinely needs a bounded connect
        against a *live* client is a different thing — extend this clause
        deliberately rather than growing a second probe.
        """
        tree = parse_test_module(path)
        offenders: list[tuple[str, int]] = []
        for call in calls_named(tree, _LIVE_GRAPH_CALL):
            first = call.args[0] if call.args else None
            if isinstance(first, ast.Constant) and first.value == PROBE_GRAPH:
                offenders.append((f'{_LIVE_GRAPH_CALL}({PROBE_GRAPH!r})', call.lineno))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(
                kw.arg == CONNECT_TIMEOUT_KWARG for kw in node.keywords
            ):
                offenders.append((f'{CONNECT_TIMEOUT_KWARG}=', node.lineno))
        assert not offenders, (
            f'{path.name}: re-implements the reachability probe — found '
            f'{sorted(set(offenders))}. The probe lives in '
            f'{HELPERS_MODULE}._falkor_available; apply it via '
            f'{HELPERS_MODULE}.{SKIP_MARKER}(). Renaming a copied probe does not '
            f'make it a different probe, which is why this clause keys on the '
            f'body rather than on the name.'
        )


@pytest.mark.parametrize('path', VERIFIED_MODULE_PATHS, ids=lambda p: p.name)
class TestVerifiedModulesRouteThroughHelpers:
    """The six migrated modules must actually route through the shared scaffolding.

    Scoped to the hand-verified six rather than all discovered modules: a
    future live-FalkorDB module may legitimately gate reachability some other
    way, and forcing it to import a specific name would over-constrain it.
    """

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

    def test_imports_shared_falkor_scaffolding(self, path):
        """The names this module uses must come from _fm_helpers, not a local copy.

        Narrower than it looks, deliberately: this asserts sourcing, not
        gating. ``test_applies_the_shared_skip_marker`` above is what pins that
        the skip guard is actually applied.
        """
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
