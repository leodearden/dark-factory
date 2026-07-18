"""Producer-hook coverage: TaskWorkflow._invoke's finally archives transcripts
(task 2742, plans/agent-transcript-archival-prd.md α).

Reuses the _invoke-probe pattern (build TaskWorkflow + patch
invoke_with_cap_retry) from test_invoke_role_config_resolution.py, with the
git_repo/git_ops/task_assignment fixture trio duplicated module-local per the
established convention. These probes drive _invoke directly (skipping run()'s
setup), so workflow._config_dir is set MANUALLY.

Fixtures are kept module-local (no conftest.py) — see
test_config_verify_admission_reload.py's rationale.
"""

from __future__ import annotations

import asyncio
import gzip
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from shared.config_dir import TaskConfigDir

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import SIMPLE_TASK
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow

# The encoded-project dir the fake transcript is laid down under.
ENC = '-home-leo-projX'


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'lib.py').write_text('def greet(name): return name\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_ops(git_repo: Path) -> GitOps:
    return GitOps(
        GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
        git_repo,
    )


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'X',
            'description': 'Y',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


def _config(git_repo: Path, **overrides) -> OrchestratorConfig:
    kwargs: dict[str, Any] = dict(
        project_root=git_repo,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )
    kwargs.update(overrides)
    return OrchestratorConfig(**kwargs)


async def _make_workflow(config, git_ops, task_assignment):
    """Build a probe TaskWorkflow with _config_dir set manually (run() skipped)."""
    wt_info = await git_ops.create_worktree(task_assignment.task_id)
    cwd = wt_info.path
    workflow = TaskWorkflow(
        assignment=task_assignment,
        config=config,
        git_ops=git_ops,
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
    )
    workflow.artifacts = None
    # Direct _invoke skips run() setup where _config_dir is created.
    workflow._config_dir = TaskConfigDir(task_assignment.task_id, base_dir=cwd / '.task')
    return workflow, cwd


@pytest.mark.asyncio
class TestProducerHook:

    async def test_end_to_end_transcript_gz_appears(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)  # transcript_archive enabled by default
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)
        config_dir = workflow._config_dir
        fake_bytes = b'{"transcript":"hello"}\n'

        def _side_effect(**kwargs):
            # The session id _invoke generated is forwarded as session_id=.
            assert config_dir is not None
            sid = kwargs['session_id']
            p = config_dir.path / 'projects' / ENC / f'{sid}.jsonl'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(fake_bytes)
            return AgentResult(success=True, output='')

        with patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            side_effect=_side_effect,
        ):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        sid = workflow._last_invoke_session_id
        gz = (
            git_repo / 'data' / 'orchestrator' / 'agent-transcripts'
            / task_assignment.task_id / ENC / f'{sid}.jsonl.gz'
        )
        assert gz.exists()
        with gzip.open(gz, 'rb') as fh:
            assert fh.read() == fake_bytes

    async def test_flag_guard_disabled_skips_helper(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo, transcript_archive={'enabled': False})
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)

        with patch('orchestrator.workflow.archive_task_transcripts') as mock_helper, patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        mock_helper.assert_not_called()

    async def test_arg_wiring(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)  # enabled by default
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)

        with patch('orchestrator.workflow.archive_task_transcripts') as mock_helper, patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        assert workflow._config_dir is not None
        mock_helper.assert_called_once_with(
            workflow._config_dir.path,
            workflow.task_id,
            workflow._last_invoke_session_id,
            archive_root=config.project_root / 'data/orchestrator/agent-transcripts',
        )
