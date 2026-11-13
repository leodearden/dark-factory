"""In a module that constructs a merge worker, or a sibling module it imports,
an ``OrchestratorConfig`` built with both ``project_root=`` and ``git=`` is a
lane scene's config: it goes through ``_merge_lane_fakes.py::lane_scene_config``,
which switches the main-health probe off, unless it names
``escalate_preexisting_main_break`` itself, which is how a scene that exercises
the probe opts in. ``_merge_lane_fakes.py::main_health_probe_spawned`` says why
a scene keeps the probe off. Not seen: a config built in a module that
constructs no worker and handed to a helper that does, as callers of
``_workflow_helpers.py::_build_workflow`` do.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# Whole-tree scanner: see shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT and
# test_whole_tree_scan_timeout_guard.py (task 4215).
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_TESTS_DIR = Path(__file__).parent

_WORKER_CONSTRUCTORS = frozenset({'SpeculativeMergeWorker', 'MergeLane', 'make_lane'})

_PROBE_FLAG = 'escalate_preexisting_main_break'


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _constructs_a_worker(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Call) and _called_name(node) in _WORKER_CONSTRUCTORS
        for node in ast.walk(tree)
    )


def _imported_modules(tree: ast.Module) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def _implicit_probe_lines(tree: ast.Module) -> list[int]:
    return sorted(
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _called_name(node) == 'OrchestratorConfig'
        and {'project_root', 'git'} <= {keyword.arg for keyword in node.keywords}
        and _PROBE_FLAG not in {keyword.arg for keyword in node.keywords}
    )


def _implicit_probe_scene_configs(trees: dict[str, ast.Module]) -> dict[str, list[int]]:
    """Lines of scene configs that leave the probe flag implicit, by module name.

    Scans the modules among *trees* that construct a worker, and the sibling
    modules among *trees* they import.
    """
    worker_modules = {name for name, tree in trees.items() if _constructs_a_worker(tree)}
    scanned = worker_modules.union(*(_imported_modules(trees[name]) for name in worker_modules))
    found = {name: _implicit_probe_lines(trees[name]) for name in sorted(scanned & trees.keys())}
    return {name: lines for name, lines in found.items() if lines}


@pytest.fixture(scope='module')
def implicit_scene_configs_by_module() -> dict[str, list[int]]:
    return _implicit_probe_scene_configs({
        module.stem: ast.parse(module.read_text(encoding='utf-8'))
        for module in sorted(_TESTS_DIR.glob('*.py'))
    })


def test_no_worker_module_builds_a_probe_on_scene_config_implicitly(
    implicit_scene_configs_by_module: dict[str, list[int]],
) -> None:
    assert implicit_scene_configs_by_module == {}, (
        f'scene config(s) with the main-health probe implicitly on, by module and line: '
        f'{implicit_scene_configs_by_module}. Build them with '
        '_merge_lane_fakes.lane_scene_config, or name escalate_preexisting_main_break '
        'explicitly if the scene exercises the main-health probe.'
    )


def _flagged(**modules: str) -> dict[str, list[int]]:
    return _implicit_probe_scene_configs({name: ast.parse(source) for name, source in modules.items()})


_BARE_SCENE_HELPER = (
    'def scene_config(repo, git):\n'
    '    return OrchestratorConfig(project_root=repo, git=git)\n'
)


@pytest.mark.parametrize('worker', [
    'MergeLane(git_ops, queue)',
    'SpeculativeMergeWorker(git_ops, queue)',
    'make_lane(git_ops)',
    'merge_lane.SpeculativeMergeWorker(git_ops, queue)',
])
def test_a_bare_scene_config_beside_a_worker_is_flagged(worker: str) -> None:
    assert _flagged(test_scene=(
        'def scene(repo, git, git_ops, queue):\n'
        '    config = OrchestratorConfig(project_root=repo, git=git)\n'
        f'    worker = {worker}\n'
    )) == {'test_scene': [2]}


@pytest.mark.parametrize('import_line', [
    'from _scene_helper import scene_config',
    'import _scene_helper',
])
def test_a_bare_scene_config_in_a_module_a_worker_module_imports_is_flagged(
    import_line: str,
) -> None:
    assert _flagged(
        _scene_helper=_BARE_SCENE_HELPER,
        test_scene=(
            f'{import_line}\n'
            'def scene(git_ops):\n'
            '    lane = make_lane(git_ops)\n'
        ),
    ) == {'_scene_helper': [2]}


def test_a_bare_scene_config_in_a_module_only_workerless_modules_import_is_not_flagged() -> None:
    assert _flagged(
        _scene_helper=_BARE_SCENE_HELPER,
        test_unit=(
            'from _scene_helper import scene_config\n'
            'def unit(repo, git):\n'
            '    config = scene_config(repo, git)\n'
        ),
    ) == {}


def test_a_lane_scene_config_is_not_flagged() -> None:
    assert _flagged(test_scene=(
        'def scene(repo, git, git_ops):\n'
        '    config = lane_scene_config(repo, git)\n'
        '    lane = make_lane(git_ops)\n'
    )) == {}


@pytest.mark.parametrize('value', ['True', 'flag'])
def test_an_explicitly_named_probe_flag_is_not_flagged(value: str) -> None:
    assert _flagged(test_scene=(
        'def scene(repo, git, git_ops, flag):\n'
        '    config = OrchestratorConfig(\n'
        f'        project_root=repo, git=git, escalate_preexisting_main_break={value},\n'
        '    )\n'
        '    lane = make_lane(git_ops)\n'
    )) == {}


def test_a_bare_scene_config_in_a_module_without_a_worker_is_not_flagged() -> None:
    assert _flagged(test_unit=_BARE_SCENE_HELPER) == {}


def test_a_config_without_git_is_not_a_scene_config() -> None:
    assert _flagged(test_unit=(
        'def unit(repo, git_ops):\n'
        '    config = OrchestratorConfig(project_root=repo)\n'
        '    lane = make_lane(git_ops)\n'
    )) == {}


def test_a_splat_does_not_name_the_probe_flag() -> None:
    assert _flagged(test_scene=(
        'def scene(repo, git, git_ops, extra):\n'
        '    config = OrchestratorConfig(project_root=repo, git=git, **extra)\n'
        '    lane = make_lane(git_ops)\n'
    )) == {'test_scene': [2]}
