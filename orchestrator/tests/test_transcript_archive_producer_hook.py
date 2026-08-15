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

    async def test_end_to_end_transcript_appears(
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
        archived = (
            git_repo / 'data' / 'orchestrator' / 'agent-transcripts'
            / task_assignment.task_id / ENC / f'{sid}.jsonl'
        )
        assert archived.exists()
        assert archived.read_bytes() == fake_bytes

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

    async def test_helper_error_does_not_mask_in_flight(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """A non-cancellation error escaping the helper is swallowed, not raised.

        archive_task_transcripts guards per-file work with ``except OSError`` but
        NOT its top-level glob/Path/archive_root construction. Awaiting it inside
        _invoke's finally means any such escaped error would REPLACE the in-flight
        exception the finally is unwinding (finally-masks-original). The hook's own
        ``try/except Exception`` is the structural guarantee that it cannot — even
        if the helper's contract regresses (review #1).
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)  # enabled by default
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)

        with patch(
            'orchestrator.workflow.archive_task_transcripts',
            side_effect=RuntimeError('unguarded glob boom'),
        ) as mock_helper, patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ):
            # Must NOT raise RuntimeError out of the finally — _invoke returns
            # its result normally and the hook logs+swallows the archival error.
            result = await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        assert result.success is True
        mock_helper.assert_called_once()

    async def test_cancellation_propagates_not_swallowed(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """CancelledError from the archive await must re-raise, never be swallowed.

        Loop teardown / hard-kill surfaces CancelledError from the ``await``; the
        hook re-raises it (cooperative cancellation must propagate), so a killed
        invocation is deliberately not archived here — that abandoned tail is
        β/task 2729's teardown-backstop's job (review #2). The masking guard's
        ``except Exception`` deliberately does NOT catch CancelledError (a
        BaseException), so this asserts the two clauses are ordered correctly.
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)  # enabled by default
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)

        with patch(
            'orchestrator.workflow.archive_task_transcripts',
            side_effect=asyncio.CancelledError,
        ), patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ), pytest.raises(asyncio.CancelledError):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)
