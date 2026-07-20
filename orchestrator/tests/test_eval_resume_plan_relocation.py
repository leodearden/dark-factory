"""Focused tests: --worktree eval resume reads the RELOCATED plan.json (D1, task 2812).

`run_eval`'s ``--worktree`` resume path used to read the frozen-plan-with-
step-statuses from the LEGACY in-worktree ``<worktree>/.task/plan.json``.
Production ``TaskWorkflow`` relocated plan.json to the sibling
``<worktree_base>/.task-meta/<worktree_name>/`` root (W11 / task 2258), so the
legacy read silently missed and the resume path fell back to the frozen
``task['plan']`` — discarding all completed-step progress. This is the same
D1 relocation-drift class already fixed on the eval diff-read path
(``snapshots.get_diff`` / ``test_eval_diff_threading.py``, task 2469).

Follows ``test_eval_diff_threading.py``'s convention: focused, direct-
invocation tests against ``tmp_path`` fixtures — no full-workflow drive.
Fixture placement follows ``test_worktree_identity.py``'s convention: derive
the relocated root via ``TaskArtifacts.meta_root_for`` rather than hand-
joining ``.task-meta``.
"""

from __future__ import annotations

import json
from pathlib import Path

from orchestrator.artifacts import TaskArtifacts
from orchestrator.evals.runner import _resume_plan_from_worktree


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_relocated_plan_preserves_progress(tmp_path: Path) -> None:
    """(a) RELOCATED .task-meta/<name>/plan.json wins, preserving progress.

    Core RED for the bug: reading the legacy ``.task/`` path would instead
    return the frozen ``task['plan']`` sentinel (0 done steps).
    """
    worktree = tmp_path / 'base' / 'run-abc'
    meta_root = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
    relocated_plan = {
        'steps': [
            {'id': 'step-1', 'status': 'done'},
            {'id': 'step-2', 'status': 'done'},
            {'id': 'step-3', 'status': 'pending'},
        ],
    }
    _write_json(meta_root / 'plan.json', relocated_plan)

    # A DISTINCT frozen sentinel with 0 done steps — must NOT be returned.
    frozen_sentinel = {'steps': [{'id': 'frozen', 'status': 'pending'}]}
    task = {'plan': frozen_sentinel}

    result = _resume_plan_from_worktree(worktree, task)

    assert result is not None
    assert result == relocated_plan
    assert result != frozen_sentinel
    done = sum(1 for s in result['steps'] if s.get('status') == 'done')
    assert done == 2


def test_legacy_task_dir_plan_is_ignored(tmp_path: Path) -> None:
    """(b) STALE legacy `<worktree>/.task/plan.json` must NOT be consulted.

    Locks in that the fix no longer reads the pre-relocation in-worktree
    path — only the relocated root and the frozen task-JSON fallback.
    """
    worktree = tmp_path / 'base' / 'run-abc'
    legacy_plan = {'steps': [{'id': 'legacy', 'status': 'done'}]}
    _write_json(worktree / '.task' / 'plan.json', legacy_plan)

    frozen_sentinel = {'steps': [{'id': 'frozen', 'status': 'pending'}]}
    task = {'plan': frozen_sentinel}

    result = _resume_plan_from_worktree(worktree, task)

    assert result == frozen_sentinel
    assert result != legacy_plan


def test_neither_present_falls_back_to_task_plan(tmp_path: Path) -> None:
    """(c) NEITHER path present → fallback to `task.get('plan')` preserved.

    Includes the ``None`` case when *task* carries no 'plan' key at all.
    """
    worktree = tmp_path / 'base' / 'run-abc'

    frozen_sentinel = {'steps': []}
    task_with_plan = {'plan': frozen_sentinel}
    assert _resume_plan_from_worktree(worktree, task_with_plan) is frozen_sentinel

    task_without_plan: dict = {}
    assert _resume_plan_from_worktree(worktree, task_without_plan) is None
