"""Tests for ``TaskWorkflow._reconcile_metadata_files_for_done`` (Fix 1).

The helper updates ``metadata.files`` to the architect-declared file set
(or clears it on no-plan paths) immediately before each
``set_task_status('done')`` call.  Without this step the phantom-done gate
in fused-memory rejects the transition with ``done_gate_missing_files``
when the architect has rewritten scope, leaving the task stranded
in-progress despite a successful merge.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.workflow import TaskWorkflow


def _make_workflow(
    *,
    project_root: Path,
    task_id: str = '101',
) -> tuple[TaskWorkflow, AsyncMock]:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    config = MagicMock()
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.project_root = project_root

    update_task = AsyncMock(return_value=True)
    scheduler = MagicMock()
    scheduler.update_task = update_task

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    return wf, update_task


@pytest.mark.asyncio
async def test_updates_with_plan_files_when_present(tmp_path: Path):
    """A populated plan refreshes ``metadata.files`` to the architect's set."""
    wf, update_task = _make_workflow(project_root=tmp_path)
    wf.plan = {'files': ['src/a.py', 'src/b.py'], 'steps': []}

    await wf._reconcile_metadata_files_for_done()

    update_task.assert_awaited_once_with(
        '101', {'files': ['src/a.py', 'src/b.py']},
    )


@pytest.mark.asyncio
async def test_clears_when_plan_empty(tmp_path: Path):
    """No plan ran (found-on-main paths) → clear ``metadata.files``.

    An empty list disarms the phantom-done gate (``if declared:`` is False
    in fused-memory's ``_extract_metadata_files``).
    """
    wf, update_task = _make_workflow(project_root=tmp_path)
    # self.plan defaults to empty dict — exercise the no-plan path.

    await wf._reconcile_metadata_files_for_done()

    update_task.assert_awaited_once_with('101', {'files': []})


@pytest.mark.asyncio
async def test_clears_when_plan_has_no_files_key(tmp_path: Path):
    """Plan present but missing the ``files`` key → still clear safely."""
    wf, update_task = _make_workflow(project_root=tmp_path)
    wf.plan = {'steps': [{'description': 's'}]}

    await wf._reconcile_metadata_files_for_done()

    update_task.assert_awaited_once_with('101', {'files': []})
