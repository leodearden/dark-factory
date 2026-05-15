"""Tests for ``TaskWorkflow._reconcile_metadata_files_for_done``.

Refactored 2026-05-10 (Stage 2 of stuck-done recovery): truth source is
the merge-diff (``git diff base..merge_sha``), not the architect's
``plan.files``.  The architect's plan can include files that get squashed
or refactored away before merge; the merge-diff is the actual record of
what landed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from _orch_helpers import pydantic_spec
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow


def _make_workflow(
    *,
    project_root: Path,
    task_id: str = '101',
) -> tuple[TaskWorkflow, AsyncMock, AsyncMock]:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    _spec.for_module = None  # custom method not in model_fields; expose to spec_set
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.project_root = project_root

    update_task = AsyncMock(return_value=True)
    scheduler = MagicMock()
    scheduler.update_task = update_task

    git_ops = MagicMock()
    get_merge_diff_files = AsyncMock(return_value=[])
    git_ops.get_merge_diff_files = get_merge_diff_files

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    return wf, update_task, get_merge_diff_files


@pytest.mark.asyncio
async def test_writes_merge_diff_files_when_merge_sha_and_base_commit_set(
    tmp_path: Path,
):
    """Happy path: writes git-diff base..merge_sha (not plan.files)."""
    wf, update_task, get_merge_diff_files = _make_workflow(project_root=tmp_path)
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    # plan.files names what the architect said it would touch — but the
    # merge actually landed a different set of paths.
    wf.plan = {'files': ['old/path.py'], 'steps': []}
    get_merge_diff_files.return_value = [
        'src/landed_a.py', 'src/landed_b.py',
    ]

    await wf._reconcile_metadata_files_for_done()

    get_merge_diff_files.assert_awaited_once_with('a' * 40, 'b' * 40)
    update_task.assert_awaited_once_with(
        '101', {'files': ['src/landed_a.py', 'src/landed_b.py']},
    )


@pytest.mark.asyncio
async def test_clears_when_merge_sha_missing(tmp_path: Path):
    """No merge_sha (already-on-main shortcuts) → write empty list.

    The gate-skip-when-verified-provenance branch in fused-memory's
    task_interceptor.py handles the missing-files case.
    """
    wf, update_task, get_merge_diff_files = _make_workflow(project_root=tmp_path)
    wf._base_commit = 'a' * 40
    wf._merge_sha = None
    wf.plan = {'files': ['anything.py']}

    await wf._reconcile_metadata_files_for_done()

    get_merge_diff_files.assert_not_awaited()
    update_task.assert_awaited_once_with('101', {'files': []})


@pytest.mark.asyncio
async def test_clears_when_base_commit_missing(tmp_path: Path):
    """No base_commit (eval mode without create_worktree) → empty list."""
    wf, update_task, get_merge_diff_files = _make_workflow(project_root=tmp_path)
    wf._base_commit = None
    wf._merge_sha = 'b' * 40

    await wf._reconcile_metadata_files_for_done()

    get_merge_diff_files.assert_not_awaited()
    update_task.assert_awaited_once_with('101', {'files': []})


@pytest.mark.asyncio
async def test_writes_empty_list_when_diff_returns_empty(tmp_path: Path):
    """Empty diff (e.g. revert merge) → empty list, no error."""
    wf, update_task, get_merge_diff_files = _make_workflow(project_root=tmp_path)
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    get_merge_diff_files.return_value = []

    await wf._reconcile_metadata_files_for_done()

    update_task.assert_awaited_once_with('101', {'files': []})
