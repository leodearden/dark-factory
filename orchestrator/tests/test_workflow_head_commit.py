"""Pins ``TaskWorkflow._get_head_commit``'s ``''`` failure sentinel against a real unreadable HEAD."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _git_fixtures import seed_repo
from _unreadable_head import make_head_unreadable

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture
def workflow(repo: Path) -> TaskWorkflow:
    config = OrchestratorConfig(project_root=repo)
    assignment = TaskAssignment(
        task_id='42',
        task={
            'id': '42', 'title': 'X', 'description': '',
            'status': 'pending', 'metadata': {}, 'dependencies': [],
        },
        modules=[],
    )
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=GitOps(config.git, repo),
        scheduler=MagicMock(),  # type: ignore[arg-type]
        briefing=MagicMock(),  # type: ignore[arg-type]
        mcp=MagicMock(),  # type: ignore[arg-type]
    )
    workflow.worktree = repo
    return workflow


@pytest.mark.asyncio
class TestGetHeadCommit:
    async def test_readable_head_returns_the_sha(self, repo, workflow):
        _, expected, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)

        assert await workflow._get_head_commit() == expected.strip()

    async def test_unreadable_head_returns_empty_string(self, repo, workflow):
        await make_head_unreadable(repo)

        assert await workflow._get_head_commit() == ''

    async def test_unreadable_head_logs_a_warning_with_the_exit_code(
        self, repo, workflow, caplog,
    ):
        await make_head_unreadable(repo)

        with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
            await workflow._get_head_commit()

        messages = [
            record.getMessage() for record in caplog.records
            if record.name == 'orchestrator.workflow' and record.levelno == logging.WARNING
        ]
        assert any('rev-parse' in m and '128' in m for m in messages), messages
