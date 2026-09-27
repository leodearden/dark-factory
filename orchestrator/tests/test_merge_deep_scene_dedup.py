"""Structural guard: every shared deep-merge scene helper has exactly ONE home.

The deep-merge suites used to carry code-identical copies of the same repo
fixtures, builders, git helpers, spies, permit census and durable-tier readers.
They now live in _merge_deep_scene.py, and this file keeps them from rotting
back into copies.

Single-sourcing is asserted by OBJECT IDENTITY, ``consumer.name is
scene.name``, never by equality: an equal re-paste passes any value comparison
while restoring exactly the drift the extraction removed.  Identity cannot see
a local ``def`` that shadows the import before it has diverged, so a static AST
sweep also forbids a consumer from re-DEFINING any name the scene defines.
Both checks carry a positive control, because a satisfied check and an inert
one look the same.

Scope: the deep integration gate and deep landing modules, plus the invariant
gate's re-definition sweep.  test_merge_queue_deep_dispatch.py and test_merge_queue_build_chain.py
are NOT yet covered; task 5172 owns extending this file to them.
"""
from __future__ import annotations

import ast
import importlib
from collections.abc import Iterable
from pathlib import Path
from types import FunctionType, ModuleType

# Imported by BARE module name, as _merge_queue_harness is: conftest.py puts
# the tests dir on sys.path.
import _merge_deep_scene as scene
import pytest

DEEP_GATE = 'test_merge_queue_deep_integration_gate'
DEEP_LANDING = 'test_merge_queue_deep_landing'
INVARIANT_GATE = 'test_merge_queue_invariant_integration_gate'

_TESTS_DIR = Path(__file__).parent
_SCENE_PATH = _TESTS_DIR / '_merge_deep_scene.py'

_Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef

SHARED_SCENE_LAYER: tuple[str, ...] = (
    # repo fixtures and builders
    '_setup_repo',
    '_add_recording_seed_to_repo',
    'git_repo',
    '_make_spec_git_config',
    '_make_git_ops',
    '_make_config',
    '_make_req',
    '_make_item',
    '_ephemeral_merge_wt',
    '_CapturingEventStore',

    # git helpers
    '_create_branch_editing',
    '_rev_parse',
    '_shared_txt_with',
    '_merge_commit_off_main',
    '_merge_commit_onto',
    '_external_main_bump',
    '_abort_hook_at',

    # spies, census and readers
    '_local_lease',
    '_fake_pass_runner',
    '_spy_chain_lane_release',
    '_spy_advance_main',
    '_PermitCensus',
    '_permit_census',
    '_drain_residue',
    '_events_for_task',
    '_canary_predicate_items_per',
)

# These share a NAME with scene helpers, not a body: a README-only repo with no
# recording seed, and a positional lane plus merge_first_enqueued_at/request_id.
INVARIANT_GATE_LOCAL_BY_DESIGN: frozenset[str] = frozenset({
    'git_repo',
    '_setup_repo',
    '_make_req',
})


def _module_path(module_name: str) -> Path:
    return _TESTS_DIR / f'{module_name}.py'


def _top_level_definition_nodes(path: Path) -> list[_Definition]:
    tree = ast.parse(path.read_text())
    return [node for node in tree.body if isinstance(node, _Definition)]


def _top_level_definitions(path: Path) -> frozenset[str]:
    """Names *path* DEFINES at module level — not what it merely imports."""
    return frozenset(node.name for node in _top_level_definition_nodes(path))


def _single_sourcing_violations(
    consumer: ModuleType, names: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Return ``(violations, checked)`` for *consumer* over *names*.

    A name the consumer does not bind is skipped: the re-definition sweep is
    what stops a module growing a local copy of a name it does not import.
    *checked* lets the caller reject a vacuous pass.
    """
    violations: list[str] = []
    checked: list[str] = []
    for name in names:
        if not hasattr(scene, name):
            violations.append(
                f'{consumer.__name__}: _merge_deep_scene has no {name!r}, so '
                'the shared layer is missing its single home'
            )
            continue
        if not hasattr(consumer, name):
            continue
        checked.append(name)
        if getattr(consumer, name) is not getattr(scene, name):
            violations.append(
                f'{consumer.__name__}: {name!r} is a separate local object, '
                'not the one _merge_deep_scene defines'
            )
    return violations, checked


def _assert_single_sourced(consumer: ModuleType, names: Iterable[str]) -> None:
    names = tuple(names)
    violations, checked = _single_sourcing_violations(consumer, names)
    if violations:
        listed = '\n  '.join(violations)
        raise AssertionError(
            'Each shared deep-merge scene helper must have exactly ONE home, '
            'orchestrator/tests/_merge_deep_scene.py. Import it from there '
            'instead of keeping a local copy.\n'
            f'\nViolations:\n  {listed}'
        )
    assert checked, (
        f'{consumer.__name__} binds none of the shared scene layer, so this '
        'check asserted nothing: the module stopped importing the scene.'
    )


def _redefinitions(
    consumer_path: Path,
    shared: frozenset[str],
    exempt: frozenset[str] = frozenset(),
) -> list[str]:
    """Report every *shared* name *consumer_path* re-defines at module level."""
    forbidden = shared - exempt
    return [
        f'{consumer_path.name}: re-defines {node.name!r} at line {node.lineno}'
        for node in _top_level_definition_nodes(consumer_path)
        if node.name in forbidden
    ]


def _assert_no_redefinitions(
    consumer_path: Path,
    shared: frozenset[str],
    exempt: frozenset[str] = frozenset(),
) -> None:
    violations = _redefinitions(consumer_path, shared, exempt)
    if violations:
        listed = '\n  '.join(violations)
        raise AssertionError(
            'A consumer may IMPORT a shared deep-merge scene helper but must '
            'never RE-DEFINE one: a local definition shadows the import and '
            'restores the copy drift. Delete it and import the name from '
            'orchestrator/tests/_merge_deep_scene.py.\n'
            f'\nViolations:\n  {listed}'
        )


def _scene_definitions() -> frozenset[str]:
    return _top_level_definitions(_SCENE_PATH)


def test_the_scene_module_imports_by_bare_module_name() -> None:
    imported = importlib.import_module('_merge_deep_scene')
    assert imported is scene
    assert imported.__name__ == '_merge_deep_scene'


def test_the_scene_defines_every_name_in_the_shared_layer() -> None:
    undefined = sorted(set(SHARED_SCENE_LAYER) - _scene_definitions())
    assert not undefined, (
        f'_merge_deep_scene.py must DEFINE, not re-export, {undefined}'
    )


_EACH_DEEP_CONSUMER = pytest.mark.parametrize(
    'consumer_name', [DEEP_GATE, DEEP_LANDING], ids=['deep_gate', 'deep_landing'],
)


@_EACH_DEEP_CONSUMER
def test_the_deep_module_single_sources_the_shared_layer(consumer_name: str) -> None:
    _assert_single_sourced(
        importlib.import_module(consumer_name), SHARED_SCENE_LAYER
    )


@_EACH_DEEP_CONSUMER
def test_the_deep_module_re_defines_no_shared_name(consumer_name: str) -> None:
    _assert_no_redefinitions(_module_path(consumer_name), _scene_definitions())


def test_the_invariant_gate_re_defines_no_shared_name() -> None:
    _assert_no_redefinitions(
        _module_path(INVARIANT_GATE),
        _scene_definitions(),
        exempt=INVARIANT_GATE_LOCAL_BY_DESIGN,
    )


def test_the_invariant_gates_exemptions_are_all_still_load_bearing() -> None:
    """An exemption that no longer names a real collision is a hole in the sweep."""
    invariant_defines = _top_level_definitions(_module_path(INVARIANT_GATE))
    scene_defines = _scene_definitions()
    stale = sorted(
        name
        for name in INVARIANT_GATE_LOCAL_BY_DESIGN
        if name not in invariant_defines or name not in scene_defines
    )
    assert not stale, (
        f'{stale} no longer collide between {INVARIANT_GATE}.py and '
        '_merge_deep_scene.py; drop them from INVARIANT_GATE_LOCAL_BY_DESIGN '
        'so the sweep covers them again'
    )


def test_the_identity_check_fires_on_a_local_copy() -> None:
    name = '_rev_parse'
    original = getattr(scene, name)

    importer = ModuleType('imports_from_the_scene')
    setattr(importer, name, original)
    assert _single_sourcing_violations(importer, (name,)) == ([], [name])

    equal_copy = FunctionType(
        original.__code__,
        original.__globals__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    assert equal_copy.__code__ is original.__code__
    recloner = ModuleType('keeps_a_local_copy')
    setattr(recloner, name, equal_copy)
    violations, checked = _single_sourcing_violations(recloner, (name,))
    assert checked == [name]
    assert len(violations) == 1 and name in violations[0], violations
    with pytest.raises(AssertionError, match='exactly ONE home'):
        _assert_single_sourced(recloner, (name,))

    bystander = ModuleType('binds_nothing')
    assert _single_sourcing_violations(bystander, (name,)) == ([], [])
    with pytest.raises(AssertionError, match='asserted nothing'):
        _assert_single_sourced(bystander, (name,))


def test_the_sweep_fires_on_a_re_cloned_definition(tmp_path: Path) -> None:
    name = '_permit_census'
    shared = _scene_definitions()
    assert name in shared

    importer = tmp_path / 'imports_from_the_scene.py'
    importer.write_text(f'from _merge_deep_scene import {name}\n')
    assert _redefinitions(importer, shared) == []

    recloner = tmp_path / 'keeps_a_local_copy.py'
    recloner.write_text(
        f'from _merge_deep_scene import {name}\n\n\n'
        f'def {name}(worker):\n    return {{}}\n'
    )
    violations = _redefinitions(recloner, shared)
    assert len(violations) == 1 and name in violations[0], violations
    with pytest.raises(AssertionError, match='must never RE-DEFINE'):
        _assert_no_redefinitions(recloner, shared)
