"""A hermetic eval workflow cell: runner's executor seams stubbed so run_eval and run_end_to_end drive a given workflow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.evals import runner
from orchestrator.evals.metrics import EvalMetrics


def executor_task(tmp_path: Path, *, task_id: str = 'df_task_cell', **extra) -> dict:
    """A minimal task record an executor cell can run, with `extra` merged over it."""
    return {
        'id': task_id,
        'project_root': str(tmp_path),
        'pre_task_commit': 'basecommit',
        'task_definition': {'title': 'W', 'description': 'd'},
        'modules': ['pkg/mod.py'],
        'plan': {'steps': [{'id': 's1', 'status': 'pending'}]},
        **extra,
    }


async def fake_create_worktree(*_a, **_k) -> tuple[Path, str]:
    return Path('/fake/wt'), 'run-cell'


async def fake_collect(
    _wf, _wt, _task, *, role, replay_frame, sandbox_backend, endpoint_kind,
) -> EvalMetrics:
    return EvalMetrics(
        role_under_test=role, replay_frame=replay_frame,
        sandbox_backend=sandbox_backend, endpoint_kind=endpoint_kind,
    )


def stub_workflow_cell(
    monkeypatch: pytest.MonkeyPatch, task: dict, workflow, *, collect=fake_collect,
) -> None:
    """Make the next executor cell load `task`, run `workflow` in a fake worktree and score it with `collect`."""
    monkeypatch.setattr(runner, 'create_eval_worktree', fake_create_worktree)
    monkeypatch.setattr(runner, 'build_workflow', lambda **_kw: workflow)
    monkeypatch.setattr(runner, 'collect_metrics', AsyncMock(side_effect=collect))
    monkeypatch.setattr(runner, 'load_task', lambda _p: task)
