"""In a module that constructs a merge worker, an ``OrchestratorConfig`` built
with both ``project_root=`` and ``git=`` is a lane scene's config: it goes
through ``_merge_lane_fakes.py::lane_scene_config``, which switches the
main-health probe off, unless it names ``escalate_preexisting_main_break``
itself, which is how a scene that exercises the probe opts in.
``_merge_lane_fakes.py::main_health_probe_spawned`` says why a scene keeps the
probe off.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT

# Whole-tree scanner: see WHOLE_TREE_SCAN_TEST_TIMEOUT in _orch_helpers.py and
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


def _implicit_probe_scene_configs(tree: ast.Module) -> list[int]:
    """Lines of scene configs that leave the probe flag implicit, in a worker-constructing *tree*."""
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    if not any(_called_name(call) in _WORKER_CONSTRUCTORS for call in calls):
        return []
    return sorted(
        call.lineno for call in calls
        if _called_name(call) == 'OrchestratorConfig'
        and {'project_root', 'git'} <= {keyword.arg for keyword in call.keywords}
        and _PROBE_FLAG not in {keyword.arg for keyword in call.keywords}
    )


@pytest.fixture(scope='module')
def implicit_scene_configs_by_module() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for module in sorted(_TESTS_DIR.glob('*.py')):
        lines = _implicit_probe_scene_configs(ast.parse(module.read_text(encoding='utf-8')))
        if lines:
            found[module.name] = lines
    return found


def test_no_worker_module_builds_a_probe_on_scene_config_implicitly(
    implicit_scene_configs_by_module: dict[str, list[int]],
) -> None:
    assert implicit_scene_configs_by_module == {}, (
        f'scene config(s) with the main-health probe implicitly on, by module and line: '
        f'{implicit_scene_configs_by_module}. Build them with '
        '_merge_lane_fakes.lane_scene_config, or name escalate_preexisting_main_break '
        'explicitly if the scene exercises the main-health probe.'
    )


@pytest.mark.parametrize('worker', [
    'MergeLane(git_ops, queue)',
    'SpeculativeMergeWorker(git_ops, queue)',
    'make_lane(git_ops)',
    'merge_lane.SpeculativeMergeWorker(git_ops, queue)',
])
def test_a_bare_scene_config_beside_a_worker_is_flagged(worker: str) -> None:
    tree = ast.parse(
        'def scene(repo, git, git_ops, queue):\n'
        '    config = OrchestratorConfig(project_root=repo, git=git)\n'
        f'    worker = {worker}\n'
    )

    assert _implicit_probe_scene_configs(tree) == [2]


def test_a_lane_scene_config_is_not_flagged() -> None:
    tree = ast.parse(
        'def scene(repo, git, git_ops):\n'
        '    config = lane_scene_config(repo, git)\n'
        '    lane = make_lane(git_ops)\n'
    )

    assert _implicit_probe_scene_configs(tree) == []


@pytest.mark.parametrize('value', ['True', 'flag'])
def test_an_explicitly_named_probe_flag_is_not_flagged(value: str) -> None:
    tree = ast.parse(
        'def scene(repo, git, git_ops, flag):\n'
        '    config = OrchestratorConfig(\n'
        f'        project_root=repo, git=git, escalate_preexisting_main_break={value},\n'
        '    )\n'
        '    lane = make_lane(git_ops)\n'
    )

    assert _implicit_probe_scene_configs(tree) == []


def test_a_bare_scene_config_in_a_module_without_a_worker_is_not_flagged() -> None:
    tree = ast.parse(
        'def scene(repo, git):\n'
        '    return OrchestratorConfig(project_root=repo, git=git)\n'
    )

    assert _implicit_probe_scene_configs(tree) == []


def test_a_config_without_git_is_not_a_scene_config() -> None:
    tree = ast.parse(
        'def unit(repo, git_ops):\n'
        '    config = OrchestratorConfig(project_root=repo)\n'
        '    lane = make_lane(git_ops)\n'
    )

    assert _implicit_probe_scene_configs(tree) == []


def test_a_splat_does_not_name_the_probe_flag() -> None:
    tree = ast.parse(
        'def scene(repo, git, git_ops, extra):\n'
        '    config = OrchestratorConfig(project_root=repo, git=git, **extra)\n'
        '    lane = make_lane(git_ops)\n'
    )

    assert _implicit_probe_scene_configs(tree) == [2]
