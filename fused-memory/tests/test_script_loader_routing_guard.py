"""Migration guard: a test module must not construct its own
``importlib.util.spec_from_file_location`` loader — loading a non-package
script goes through ``_fm_helpers.load_script_module``.

Task 3738 hoisted that loader into ``_fm_helpers`` after finding two byte-identical
copies of it. Task 3895 found 42 more: the helper existed, was documented, and the
forks kept accumulating anyway — one of them
(``test_census_memory_metadata.py``) had even hand-rolled the helper's own reuse
check. This guard is what makes that population stay at zero.

The property is one property, and the hazard it prevents is real rather than
stylistic. A local fork re-executes a script a sibling module may already own,
so one ``sys.modules`` key ends up naming two live module objects whose identity
depends on collection order. Four keys were contended that way before this task
— worst among them ``cleanup_test_collections``, which a session-scoped conftest
fixture loads and then holds a lease on while four test modules re-executed and
replaced it. ``load_script_module`` reuses by resolved file identity, tracks
which keys it installed, and refuses to shadow an entry it did not install.

The scope is DISCOVERED — every ``*.py`` under ``tests/`` — where the sibling
``test_falkor_probe_routing_guard.py`` deliberately pins a fixed list of six.
That difference is a property of the two properties, not an inconsistency.
Reachability-gating has several legitimate shapes (a ``pytestmark`` entry, a
fixture-level check, an autouse skip), so a discovered set would pre-empt a
future module's valid choice. Loading a non-package script by file path has
exactly ONE correct shape, and the 42 independent forks this task removed are
the evidence that a fixed list would not have held the line. The cost is paid
honestly, in three explicitly-documented exemptions, rather than by weakening
the property.

``_fm_helpers.py`` is excluded from discovery: the one call there DEFINES the
shared loader, so it is the implementation rather than a fork of it.

Deliberately NOT asserted: which identifiers a module imports from
``_fm_helpers``. That is name-pinning over sibling test source — a re-fork
under another name evades it, so it buys no coverage while giving false
assurance that duplication cannot return (review esc-3502-2).

AST (not string grep) so prose that merely *mentions* the idiom cannot trip the
check: ``conftest.py`` and ``test_fm_helpers.py`` both discuss
``spec_from_file_location`` at length and neither calls it.

NOT integration-marked: this file only parses source, so it must run in the
default ``-m 'not integration'`` lane. Mirrors
tests/test_falkor_probe_routing_guard.py, tests/test_falkor_index_barrier_guard.py
and tests/test_gather_idiom_helper_routing.py.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from _ast_guard import calls_named, parse_python_module

TESTS_ROOT = pathlib.Path(__file__).parent

LOADER_FACTORY = 'spec_from_file_location'

HELPER_IMPORT = 'from _fm_helpers import load_script_module'

# The AST nodes that contribute a segment to a qualified name.
SCOPE_NODES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

# Call sites that may keep a local loader, as (tests-root-relative module,
# QUALIFIED function name) pairs. Qualified — ``Class.method``, not the bare
# method name — because a bare name is not unique within a file: 148 of the 365
# modules under tests/ define at least one name twice, and
# test_bake_off_storage_shape.py, which carries two of the three exemptions,
# is one of them. A bare key would silently extend an exemption to a
# same-named method in another class.
#
# Granularity is per-FUNCTION, never per-module: a module-level exemption would
# silently excuse an ordinary fork living in the same file as a legitimate one.
# An exemption DOES cover scopes nested inside the named function, so a loader
# moved into a local helper of an exempt test stays exempt.
#
# Every entry is checked against the source by
# ``test_every_exemption_names_one_live_loader_site``, so a stale or misspelled
# exemption fails loudly instead of silently excusing nothing.
EXEMPT_CALL_SITES = frozenset(
    {
        # Both copy the script into tmp_path and load THAT relocated copy under
        # a throwaway name, popping it in `finally`, to prove the script's
        # default paths are __file__-derived rather than baked in.
        # load_script_module reuses by file identity and has no unload, so
        # routing these through it would leave a throwaway module installed for
        # the rest of the session. The same file's module-level loader call is
        # an ordinary fork and stays covered.
        (
            'test_bake_off_storage_shape.py',
            'TestDefaultFixturePaths.test_paths_are_derived_from___file___not_baked_in',
        ),
        (
            'test_bake_off_storage_shape.py',
            'TestLoadRegrowthInjections.test_the_default_path_follows_a_relocated_script',
        ),
        # Loads a PACKAGE module (fused_memory.reconciliation.stage1_stall_detector),
        # not a scripts/ file, under a synthetic name with sys.modules['escalation']
        # and ['escalation.models'] stubbed to None and restored in `finally`. It
        # needs a guaranteed-FRESH exec and no residue; load_script_module reuses by
        # file identity and never unloads. Outside the helper's stated purpose —
        # "Load a non-package script".
        (
            'reconciliation/test_stage1_stall_detector.py',
            'TestEscalationBinding.test_except_branch_binds_escalation_to_none',
        ),
    }
)

# The loader DEFINITION, not a fork of it.
HELPER_MODULE = TESTS_ROOT / '_fm_helpers.py'


def _discovered_modules():
    """Every Python module under ``tests/``, the helper itself excepted.

    Discovered rather than listed: a new test module that forks the loader must
    fail this guard on arrival, without anyone remembering to enrol it.
    """
    return sorted(path for path in TESTS_ROOT.rglob('*.py') if path != HELPER_MODULE)


def _module_key(path):
    """*path* as the guard names it: relative to the tests root, POSIX-spelled.

    Precondition: *path* is under ``TESTS_ROOT`` — the shape
    ``_discovered_modules()`` yields. Anything outside it is refused rather
    than coerced to fit, for the reason the refusal itself gives.
    """
    path = pathlib.Path(path)
    if not path.is_relative_to(TESTS_ROOT):
        raise ValueError(
            f'{str(path)!r} is not under the tests root {TESTS_ROOT}. _module_key names '
            f'a module by its path relative to that root, so it takes the absolute path '
            f"_discovered_modules() yields — build an ad-hoc one as TESTS_ROOT / 'sub/test_x.py'. "
            f'A bare relative path is rejected rather than resolved because its root is '
            f'ambiguous: the cwd, the fused-memory package root and the tests root each name '
            f'a different file.'
        )
    return path.relative_to(TESTS_ROOT).as_posix()


def _nodes_with_enclosing_scope(tree):
    """Every node under *tree*, paired with the scope chain enclosing it.

    A chain is the names of the ClassDef/FunctionDef ancestors between the node
    and the module, outermost first; a module-scope node's chain is empty. The
    chain paired with a scope node is its ENCLOSING one, so that node's own
    qualified name is ``(*chain, node.name)``.
    """

    def walk(node, chain):
        for child in ast.iter_child_nodes(node):
            yield child, chain
            yield from walk(
                child, (*chain, child.name) if isinstance(child, SCOPE_NODES) else chain
            )

    yield from walk(tree, ())


def _is_exempt(module_key, chain):
    """Does an exemption for *module_key* cover a call in scope *chain*?

    Every ancestor scope counts, not just the innermost one, so a loader call a
    legitimately-exempt test delegates to a local helper is exempt too.
    """
    return any(
        (module_key, '.'.join(chain[:depth])) in EXEMPT_CALL_SITES
        for depth in range(1, len(chain) + 1)
    )


def _unrouted_loader_lines(path):
    """The line of every non-exempt ``spec_from_file_location(...)`` call in *path*.

    Naming the module comes first: ``_module_key`` is the only step that checks
    the path's shape, and it does no I/O. Without that ordering a cwd-relative
    path the text prefilter misses gets a clean ``[]`` — a false "forks no
    loader" verdict about a file this guard never located under the tests root.

    A cheap text prefilter runs before any parse, the idiom
    ``test_falkor_index_barrier_guard.py`` established for a discovered scope:
    ``_ast_guard.parse_python_module`` memoises for the whole session, so
    parsing all 365 modules under tests/ would pin every tree for the rest of
    the run — measured here at 24 s and 475 MB of retained trees, paid per
    xdist worker. Prefiltered, eight modules are parsed: 0.4 s and 25 MB.

    The prefilter cannot hide a fork: ``calls_named`` matches a callee spelled
    ``spec_from_file_location`` or ``….spec_from_file_location``, and neither
    can exist without that identifier appearing literally in the source, so the
    prefilter is a strict SUPERSET of the AST criterion. Every assertion stays
    AST-based, which is what keeps the prose in ``conftest.py`` and
    ``test_fm_helpers.py`` from tripping the guard.

    It narrows the PARSE, deliberately not the discovered set the parametrize
    walks — one visible test case per module, so a criterion that stopped
    matching could not quietly leave the guard checking nothing. That is the
    hazard the barrier guard answers with a hand-verified floor set; here the
    shape of the parametrize answers it instead.
    """
    module_key = _module_key(path)
    if LOADER_FACTORY not in path.read_text():
        return []
    tree = parse_python_module(path)
    chain_of = {id(node): chain for node, chain in _nodes_with_enclosing_scope(tree)}
    return sorted(
        call.lineno
        for call in calls_named(tree, LOADER_FACTORY)
        if not _is_exempt(module_key, chain_of[id(call)])
    )


@pytest.mark.parametrize('path', _discovered_modules(), ids=_module_key)
def test_loads_scripts_through_the_shared_helper(path):
    """The module must not fork its own non-package script loader."""
    lines = _unrouted_loader_lines(path)
    assert not lines, (
        f'{_module_key(path)}: constructs its own {LOADER_FACTORY}() loader at '
        f'line(s) {lines}. Loading a non-package script by '
        f'path goes through the shared reuse-aware loader instead — `{HELPER_IMPORT}`, '
        f"then `load_script_module(SCRIPT_PATH, mod_name='<existing sys.modules key>')` "
        f'(pass the existing key explicitly so it stays greppable and collection-order '
        f'behaviour is unchanged). A local fork re-executes a script a sibling module '
        f'may already own, leaving one sys.modules key naming two live module objects '
        f'whose identity depends on collection order (task 3895).'
    )


@pytest.mark.parametrize(('module_key', 'qualname'), sorted(EXEMPT_CALL_SITES))
def test_every_exemption_names_one_live_loader_site(module_key, qualname):
    """An exemption must resolve to exactly one function that really forks the loader.

    Two failure modes this closes, both of which otherwise pass silently. An
    AMBIGUOUS key — one qualified name matching two definitions in the file —
    would excuse a fork nobody read. A STALE key — a function since renamed,
    moved or routed through the helper — would sit in the set looking like a
    standing licence to fork, which is how an exemption list rots into a
    shrinking-allowlist ratchet.
    """
    path = TESTS_ROOT / module_key
    assert path.exists(), f'{module_key}: exempted module does not exist'
    tree = parse_python_module(path)
    matches = [
        node
        for node, chain in _nodes_with_enclosing_scope(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and '.'.join((*chain, node.name)) == qualname
    ]
    assert len(matches) == 1, (
        f'{module_key}: the exemption for {qualname!r} matches {len(matches)} function '
        f'definitions, not 1. An exemption names one function — spell it qualified '
        f'(a module-level function bare, a method as `Class.method`) so it cannot reach '
        f'a same-named sibling.'
    )
    assert calls_named(matches[0], LOADER_FACTORY), (
        f'{module_key}: {qualname} no longer calls {LOADER_FACTORY}(), so its exemption '
        f'is dead. Delete the entry — a leftover exemption reads as a standing licence '
        f'to fork the loader there again.'
    )


class TestModuleKeyNamesPathsUnderTheTestsRoot:
    """``_module_key`` takes a path under ``TESTS_ROOT``, and says so when it does not.

    Sighting: ``plans/confusion-census-2026-09-20.md`` §1.4 — an ad-hoc probe
    of this guard passed ``pathlib.Path('tests/_scratch_guard_probe.py')`` and
    got the stdlib's ``is not in the subpath of`` refusal, which states the
    offence and leaves the caller to rediscover the spelling that works.
    """

    @pytest.mark.parametrize(
        'probe',
        ['tests/_scratch_guard_probe.py', pathlib.Path('tests/_scratch_guard_probe.py')],
        ids=['str', 'Path'],
    )
    def test_a_path_outside_the_tests_root_names_the_spelling_that_works(self, probe):
        """Both spellings an ad-hoc caller types reach the same informative refusal.

        A bare ``str`` used to reach ``AttributeError: 'str' object has no
        attribute 'is_relative_to'`` — the same remedy-less stdlib refusal,
        through the other door.
        """
        with pytest.raises(ValueError) as excinfo:
            _module_key(probe)
        message = str(excinfo.value)
        assert str(probe) in message, (
            'the refusal must quote the path it rejected, so a caller who passed '
            'several knows which one was wrong'
        )
        assert str(TESTS_ROOT) in message, (
            'the refusal must name the boundary the path was measured against'
        )
        assert 'TESTS_ROOT' in message, (
            'the refusal must name the module constant an ad-hoc caller types to build '
            "a path that works (TESTS_ROOT / 'sub/test_x.py')"
        )

    def test_a_path_under_the_tests_root_keys_on_its_posix_relative_spelling(self):
        """The shape ``_discovered_modules()`` yields keeps its key, nesting included.

        The regression pin on the precondition check: the guard's own callers pass
        exactly this shape, and ``EXEMPT_CALL_SITES`` keys on the POSIX spelling.
        The path is synthetic rather than a real sibling module: ``_module_key``
        does pure path arithmetic, so naming one would couple this case to that
        module's name, and ``sub/test_x.py`` is the spelling the refusal
        advertises — the remedy and its pin stay in lockstep.
        """
        assert _module_key(TESTS_ROOT / 'sub' / 'test_x.py') == 'sub/test_x.py'

    def test_a_cwd_relative_path_is_refused_rather_than_read_against_the_cwd(
        self, tmp_path, monkeypatch
    ):
        """Naming the module comes before reading it, so a false clean is impossible.

        The probe is deliberately innocuous so that it MISSES the text
        prefilter: that is what makes the case about the ordering rather than
        about the refusal's wording.
        """
        (tmp_path / 'probe.py').write_text('value = 1\n')
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError) as excinfo:
            _unrouted_loader_lines(pathlib.Path('probe.py'))
        assert 'TESTS_ROOT' in str(excinfo.value), (
            'the refusal must come from _module_key — the one place this precondition '
            'is stated — rather than from a second check that could drift from it'
        )
