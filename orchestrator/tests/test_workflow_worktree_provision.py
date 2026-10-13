"""Dispatch-setup venv sync must never uninstall from a shared workspace venv (task 3443)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow

_MEMBERS = ('alpha', 'beta')


def _workflow_touching_members(worktree: Path) -> TaskWorkflow:
    for member in _MEMBERS:
        (worktree / member).mkdir()
        (worktree / member / 'pyproject.toml').write_text('[project]\nname = "m"\n')

    assignment = MagicMock()
    assignment.task_id = '3443'
    assignment.task = {'id': '3443', 'title': 'T', 'description': 'd'}
    assignment.modules = [f'{member}/src/mod.py' for member in _MEMBERS]

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.project_root = worktree

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    wf.worktree = worktree
    return wf


async def _recorded_uv_argvs(wf: TaskWorkflow) -> list[tuple[str, ...]]:
    argvs: list[tuple[str, ...]] = []

    async def fake_exec(*argv: str, **_kwargs: object) -> MagicMock:
        argvs.append(argv)
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b'', None))
        return proc

    with patch('orchestrator.workflow.asyncio.create_subprocess_exec', new=fake_exec):
        await wf._sync_worktree_venvs()
    return argvs


@pytest.mark.asyncio
async def test_dispatch_sync_never_prunes_other_workspace_members(tmp_path: Path):
    argvs = await _recorded_uv_argvs(_workflow_touching_members(tmp_path))

    assert argvs
    for argv in argvs:
        assert '--inexact' in argv
        assert '--all-packages' in argv
