"""Agents reach their task's state only through the lane `.task/` links
(task 3160): no prompt computes a `.task-meta` path, and every `.task/<name>`
a prompt names resolves into this task's durable storage. The linked names
are `orchestrator/src/orchestrator/artifacts.py::LANE_SYMLINKED_ARTIFACTS`.
"""

import re
from pathlib import Path

import pytest

from orchestrator.agents.roles import ROLES
from orchestrator.artifacts import TaskArtifacts

_LANE_PATH = re.compile(r'\.task/[A-Za-z0-9_.-]*[A-Za-z0-9_]')
_HISTORY_PATHS = ('.task/plan.json', '.task/iterations.jsonl')


def build_dual_root_lane(tmp_path: Path) -> tuple[Path, TaskArtifacts]:
    """A dispatched lane for task 3160 holding a plan and one logged iteration."""
    wt = tmp_path / 'worktrees' / '_lane-3'
    wt.mkdir(parents=True)
    arts = TaskArtifacts.for_task(wt, '3160')
    arts.init('3160', 'T', 'D')
    arts.ensure_lane_symlinks()
    arts.write_plan({'steps': []})
    arts.append_iteration_log({'iteration': 1, 'agent': 'implementer', 'summary': 'S'})
    return wt, arts


def assert_lane_paths_resolve(prompt: str, wt: Path, arts: TaskArtifacts) -> None:
    task_root = arts.task_root.resolve()
    for rel in sorted(set(_LANE_PATH.findall(prompt))):
        lane_path = wt / rel
        assert lane_path.resolve().is_relative_to(task_root), (
            f'{rel} does not resolve into the task root {task_root}'
        )
        assert lane_path.read_text() == arts.artifact_path(Path(rel).name).read_text(), (
            f'{rel} does not read as this task\'s {Path(rel).name}'
        )


@pytest.mark.parametrize('name', sorted(ROLES))
def test_no_role_prompt_computes_a_meta_root_path(name):
    prompt = ROLES[name].system_prompt
    assert '<worktree_base>' not in prompt, f'{name} names <worktree_base>'
    assert '.task-meta/<' not in prompt, f'{name} builds a .task-meta path'


@pytest.mark.parametrize('name', ['implementer', 'debugger'])
def test_history_reading_roles_name_both_lane_paths(name):
    prompt = ROLES[name].system_prompt
    for path in _HISTORY_PATHS:
        assert path in prompt, f'{name} does not name {path}'


def test_every_lane_path_a_role_names_resolves_to_this_tasks_state(tmp_path):
    wt, arts = build_dual_root_lane(tmp_path)
    for role in ROLES.values():
        assert_lane_paths_resolve(role.system_prompt, wt, arts)
