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

MODULE_SCOPE = '<module>'

# Call sites that may keep a local loader, as (tests-root-relative module,
# enclosing function) pairs. Granularity is per-FUNCTION, never per-module: a
# module-level exemption would silently excuse an ordinary fork living in the
# same file as a legitimate one.
EXEMPT_CALL_SITES = frozenset(
    {
        # Both copy the script into tmp_path and load THAT relocated copy under
        # a throwaway name, popping it in `finally`, to prove the script's
        # default paths are __file__-derived rather than baked in.
        # load_script_module reuses by file identity and has no unload, so
        # routing these through it would leave a throwaway module installed for
        # the rest of the session. The same file's module-level `_load_module`
        # is an ordinary fork and stays covered.
        (
            'test_bake_off_storage_shape.py',
            'test_paths_are_derived_from___file___not_baked_in',
        ),
        ('test_bake_off_storage_shape.py', 'test_the_default_path_follows_a_relocated_script'),
        # Loads a PACKAGE module (fused_memory.reconciliation.stage1_stall_detector),
        # not a scripts/ file, under a synthetic name with sys.modules['escalation']
        # and ['escalation.models'] stubbed to None and restored in `finally`. It
        # needs a guaranteed-FRESH exec and no residue; load_script_module reuses by
        # file identity and never unloads. Outside the helper's stated purpose —
        # "Load a non-package script".
        (
            'reconciliation/test_stage1_stall_detector.py',
            'test_except_branch_binds_escalation_to_none',
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
    """*path* as the guard names it: relative to the tests root, POSIX-spelled."""
    return path.relative_to(TESTS_ROOT).as_posix()


def _unrouted_loader_calls(path):
    """Every non-exempt ``spec_from_file_location(...)`` call in *path*.

    Each call is attributed to its INNERMOST enclosing function, so an
    exemption granted to one test never reaches a loader in a sibling function
    or at module scope.
    """
    tree = parse_python_module(path)
    enclosing_of = {}

    def attribute(node, enclosing):
        for child in ast.iter_child_nodes(node):
            enclosing_of[id(child)] = enclosing
            attribute(
                child,
                child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else enclosing,
            )

    attribute(tree, MODULE_SCOPE)
    return [
        call
        for call in calls_named(tree, LOADER_FACTORY)
        if (_module_key(path), enclosing_of[id(call)]) not in EXEMPT_CALL_SITES
    ]


@pytest.mark.parametrize('path', _discovered_modules(), ids=_module_key)
def test_loads_scripts_through_the_shared_helper(path):
    """The module must not fork its own non-package script loader."""
    calls = _unrouted_loader_calls(path)
    assert not calls, (
        f'{_module_key(path)}: constructs its own {LOADER_FACTORY}() loader at '
        f'line(s) {[call.lineno for call in calls]}. Loading a non-package script by '
        f'path goes through the shared reuse-aware loader instead — `{HELPER_IMPORT}`, '
        f"then `load_script_module(SCRIPT_PATH, mod_name='<existing sys.modules key>')` "
        f'(pass the existing key explicitly so it stays greppable and collection-order '
        f'behaviour is unchanged). A local fork re-executes a script a sibling module '
        f'may already own, leaving one sys.modules key naming two live module objects '
        f'whose identity depends on collection order (task 3895).'
    )
