"""Sidecar v2 wiring + preserve-on-cancellation coverage for TaskWorkflow._invoke
(task 2771, plans/warm-lane-session-resume-prd.md α).

Reuses the _invoke-probe pattern (build TaskWorkflow + patch
invoke_with_cap_retry) from test_transcript_archive_producer_hook.py, but
attaches a REAL TaskArtifacts (rooted at the worktree's .task dir) instead of
None so the sidecar read/write/clear actually hit disk. These probes drive
_invoke directly (skipping run()'s setup), so workflow._config_dir is set
MANUALLY.

Fixtures are kept module-local (no conftest.py) per the established convention
(see test_config_verify_admission_reload.py's rationale).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from shared.config_dir import TaskConfigDir

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import SIMPLE_TASK
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow


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
        # Keep these probes hermetic: the sidecar is the unit under test, not
        # the transcript producer hook (its own finally arm is covered by
        # test_transcript_archive_producer_hook.py).
        transcript_archive={'enabled': False},
    )
    kwargs.update(overrides)
    return OrchestratorConfig(**kwargs)


async def _make_workflow(config, git_ops, task_assignment, resume_session_id=None):
    """Build a probe TaskWorkflow with a REAL on-disk TaskArtifacts attached.

    Unlike test_transcript_archive_producer_hook.py (which sets artifacts=None),
    the sidecar read/write/clear here must hit disk, so we init a real
    TaskArtifacts rooted at the worktree's .task dir.
    """
    wt_info = await git_ops.create_worktree(task_assignment.task_id)
    cwd = wt_info.path
    workflow = TaskWorkflow(
        assignment=task_assignment,
        config=config,
        git_ops=git_ops,
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
        resume_session_id=resume_session_id,
    )
    artifacts = TaskArtifacts(cwd)
    artifacts.init(task_assignment.task_id, 'X', 'Y')
    workflow.artifacts = artifacts
    # Direct _invoke skips run() setup where _config_dir is created.
    workflow._config_dir = TaskConfigDir(task_assignment.task_id, base_dir=cwd / '.task')
    return workflow, cwd


@pytest.mark.asyncio
class TestSidecarV2Wiring:
    """MECHANISM 1 wiring: _invoke writes the v2 sidecar binding (task_id +
    resume_count) through write_agent_session."""

    async def test_invoke_writes_v2_sidecar_binding(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)

        seen: dict[str, Any] = {}

        def _side_effect(**kwargs):
            # Mid-invocation: the sidecar was written before the try/await, so
            # it is on disk now. Capture what _invoke bound.
            assert workflow.artifacts is not None
            session = workflow.artifacts.read_agent_session()
            assert session is not None
            seen['task_id'] = session.task_id
            seen['schema_version'] = session.schema_version
            seen['resume_count'] = session.resume_count
            return AgentResult(success=True, output='')

        with patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            side_effect=_side_effect,
        ):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        assert seen['task_id'] == str(workflow.task_id)
        assert seen['schema_version'] == 2
        assert seen['resume_count'] == 0

    async def test_invoke_increments_resume_count_on_adopted_resume(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)
        resume = {
            'session_id': 'prior-sess-xyz',
            'role': SIMPLE_TASK.name,
            'resume_count': 2,
        }
        workflow, cwd = await _make_workflow(
            config, git_ops, task_assignment, resume_session_id=resume
        )

        seen: dict[str, Any] = {}

        def _side_effect(**kwargs):
            assert workflow.artifacts is not None
            session = workflow.artifacts.read_agent_session()
            assert session is not None
            seen['resume_count'] = session.resume_count
            seen['session_id'] = session.session_id
            return AgentResult(success=True, output='')

        with patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            side_effect=_side_effect,
        ):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        # base count 2 + 1 for this adopted resume = 3
        assert seen['resume_count'] == 3
        # resumed against the prior session id, not a fresh uuid4
        assert seen['session_id'] == 'prior-sess-xyz'


def _preserve_records(caplog) -> list[logging.LogRecord]:
    """LogRecords carrying the structured agent_session_preserved fact.

    The fact lives entirely in ``extra`` (INV-2 — no log-scrape), so it is
    identified by the ``event`` attribute, not by matching prose.
    """
    return [
        r for r in caplog.records
        if getattr(r, 'event', None) == 'agent_session_preserved'
    ]


@pytest.mark.asyncio
class TestPreserveOnCancellation:
    """MECHANISM 2 (I1): _invoke clears the sidecar on completion but PRESERVES
    it on CancelledError (cancellation is not completion), emitting a structured
    agent_session_preserved fact at the preserve point."""

    async def test_completion_clears_sidecar(
        self, monkeypatch, caplog, git_repo, git_ops, task_assignment
    ):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        caplog.set_level(logging.INFO)
        config = _config(git_repo)
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)

        with patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        # Completion → sidecar cleared, and NO preserve fact was emitted.
        assert workflow.artifacts is not None
        assert workflow.artifacts.read_agent_session() is None
        assert _preserve_records(caplog) == []

    async def test_cancellation_preserves_sidecar_and_emits_fact(
        self, monkeypatch, caplog, git_repo, git_ops, task_assignment
    ):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        caplog.set_level(logging.INFO)
        config = _config(git_repo)
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)

        with patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError,
        ), pytest.raises(asyncio.CancelledError):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        # CancelledError from the invoke → the sidecar SURVIVES (still in-flight),
        # carrying the durable v2 binding.
        assert workflow.artifacts is not None
        session = workflow.artifacts.read_agent_session()
        assert session is not None
        assert session.task_id == str(workflow.task_id)
        assert session.schema_version == 2

        # Exactly one structured preserve fact, with its fields at the decision
        # point (task_id / session_id / role in extra).
        records = _preserve_records(caplog)
        assert len(records) == 1
        rec = records[0]
        # Fields live in ``extra`` (INV-2), so LogRecord doesn't statically
        # declare them — read via getattr with a default, matching
        # ``_preserve_records`` above (the default also keeps ruff B009 happy).
        assert getattr(rec, 'task_id', None) == str(workflow.task_id)
        assert getattr(rec, 'session_id', None) == workflow._last_invoke_session_id
        assert getattr(rec, 'role', None) == SIMPLE_TASK.name
