"""plan-tools flags a plan entry naming another task's ``.task-meta`` (task 4825).

add_plan_step, add_prerequisite, replace_plan_step and confirm_plan attach
``foreign_task_meta`` to their OK response when an entry's description names
an absolute path no lane of this task can write. The flag is advisory: the
entry is still stored and the plan is still confirmed. A sibling of
``test_plan_tools_step_type.py``, run on a REAL dual-root linked worktree.
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp.foreign_task_meta import FOREIGN_TASK_META_KEY, FOREIGN_TASK_META_NOTE
from orchestrator.mcp.plan_tools import (
    _add_plan_step,
    _add_prerequisite,
    _confirm_plan,
    _create_plan,
    _replace_plan_step,
)


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def wt(tmp_path: Path) -> Path:
    main = tmp_path / 'main-repo'
    main.mkdir()
    _git(['init', '-b', 'main'], main)
    _git(['config', 'user.email', 'test@test.com'], main)
    _git(['config', 'user.name', 'Test'], main)
    (main / 'README.md').write_text('# test\n')
    _git(['add', '-A'], main)
    _git(['commit', '-m', 'initial commit'], main)
    base = tmp_path / 'worktrees'
    base.mkdir()
    worktree = base / '4825'
    _git(['worktree', 'add', '-b', 'task/4825', str(worktree), 'main'], main)
    return worktree


@pytest.fixture
def artifacts(wt: Path) -> TaskArtifacts:
    a = TaskArtifacts.for_task(wt, '4825')
    a.init('4825', 'T', 'D')
    _create_plan(a, '4825', 'T', 'A', ['mod/a.py'])
    return a


@pytest.fixture
def foreign(wt: Path) -> str:
    return str(wt.parent / '.task-meta' / '4762' / 'plan.json')


def _dirty(foreign: str) -> str:
    return f'cp {foreign} aside'


def _via_add_plan_step(a: TaskArtifacts, description: str) -> tuple[dict[str, Any], str, str]:
    return _add_plan_step(a, 'step-1', 'impl', description), 'step-1', 'steps'


def _via_add_prerequisite(a: TaskArtifacts, description: str) -> tuple[dict[str, Any], str, str]:
    return _add_prerequisite(a, 'pre-1', description), 'pre-1', 'prerequisites'


def _via_replace_plan_step(a: TaskArtifacts, description: str) -> tuple[dict[str, Any], str, str]:
    _add_plan_step(a, 'step-1', 'impl', 'clean')
    return _replace_plan_step(a, 'step-1', 'impl', description), 'step-1', 'steps'


_Writer = Callable[[TaskArtifacts, str], tuple[dict[str, Any], str, str]]
_WRITERS = pytest.mark.parametrize(
    'write', [_via_add_plan_step, _via_add_prerequisite, _via_replace_plan_step],
    ids=['add_plan_step', 'add_prerequisite', 'replace_plan_step'],
)


@_WRITERS
def test_a_dirty_entry_is_stored_and_flagged(artifacts: TaskArtifacts, foreign: str, write: _Writer):
    response, entry_id, collection = write(artifacts, _dirty(foreign))

    assert response['status'] == 'ok'
    stored = next(e for e in artifacts.read_plan()[collection] if e['id'] == entry_id)
    assert stored['description'] == _dirty(foreign)
    assert response[FOREIGN_TASK_META_KEY] == {
        'paths': {entry_id: [foreign]},
        'note': FOREIGN_TASK_META_NOTE,
    }


@_WRITERS
def test_a_clean_entry_carries_no_flag(artifacts: TaskArtifacts, write: _Writer):
    response, _, _ = write(artifacts, 'edit mod/a.py')

    assert response['status'] == 'ok'
    assert FOREIGN_TASK_META_KEY not in response


def test_confirm_plan_flags_every_dirty_entry_and_still_confirms(
    artifacts: TaskArtifacts, foreign: str,
):
    _add_prerequisite(artifacts, 'pre-1', 'clean')
    _add_plan_step(artifacts, 'step-1', 'impl', _dirty(foreign))
    _add_plan_step(artifacts, 'step-2', 'test', 'clean')

    response = _confirm_plan(artifacts)

    assert response['status'] == 'ok'
    assert response['finalized'] is True
    assert response[FOREIGN_TASK_META_KEY]['paths'] == {'step-1': [foreign]}
    assert '_finalized_at' in artifacts.read_plan()


def test_confirm_plan_on_a_clean_plan_carries_no_flag(artifacts: TaskArtifacts):
    _add_plan_step(artifacts, 'step-1', 'impl', 'clean')

    response = _confirm_plan(artifacts)

    assert response['status'] == 'ok'
    assert FOREIGN_TASK_META_KEY not in response


def test_a_legacy_worktree_with_no_linked_git_fails_open(tmp_path: Path):
    legacy = TaskArtifacts(tmp_path)
    legacy.init('4825', 'T', 'D')
    _create_plan(legacy, '4825', 'T', 'A', ['mod/a.py'])
    foreign = str(tmp_path / 'worktrees' / '.task-meta' / '4762' / 'plan.json')

    response = _add_plan_step(legacy, 'step-1', 'impl', _dirty(foreign))

    assert response['status'] == 'ok'
    assert FOREIGN_TASK_META_KEY not in response
