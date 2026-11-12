"""Task 5579 across the workflow seam: a resumed run that did work is kept.

Drives ``TaskWorkflow._execute_iterations`` through the REAL ``_invoke`` and
the REAL gated ``shared.cli_invoke.invoke_with_cap_retry``; only
``orchestrator.workflow.invoke_agent`` (the subprocess boundary) is stubbed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_gate_yielding
from shared.testing_resume_rejection import measured_resume_rejection
from test_invoke import _make_slot
from test_session_resume_integration_gate import (
    _config,
    _init_resume_repo,
    _make_resumed_workflow,
    _RecordingEventStore,
    _RecordingSink,
    _resume_assignment,
    _seed_live_transcript,
)

from orchestrator.agents.invoke import AgentResult
from orchestrator.config import GitConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

SID = '4aed993b-20c0-4b91-a8dd-60180e7db2e0'
STEP_IDS = ['step-1', 'step-2']


def _downgraded_end_turn() -> AgentResult:
    return AgentResult(
        success=False, output='', subtype='success', stop_reason='end_turn',
        ended_awaiting_background=True, turns=40, cost_usd=2.195,
        session_id=SID, transcript_turns=120,
    )


async def _resumed_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[TaskWorkflow, _RecordingSink, _RecordingEventStore]:
    repo = tmp_path / 'repo'
    repo.mkdir()
    await _init_resume_repo(repo)
    git_ops = GitOps(
        GitConfig(main_branch='main', branch_prefix='task/', remote='origin', worktree_dir='.worktrees'),
        repo,
    )
    sink = _RecordingSink()
    workflow, cwd = await _make_resumed_workflow(
        _config(repo), git_ops, _resume_assignment('5579'),
        {'session_id': SID, 'role': 'implementer'},
        resume_outcome_sink=sink,
    )
    assert workflow._config_dir is not None and workflow.artifacts is not None
    _seed_live_transcript(workflow._config_dir.path, SID)
    workflow.worktree = cwd
    store = _RecordingEventStore()
    monkeypatch.setattr(workflow, 'event_store', store)
    workflow.usage_gate = make_gate_yielding([_make_slot(cap_hit=False), _make_slot(cap_hit=False)])
    workflow.artifacts.write_plan({
        'prerequisites': [],
        'steps': [{'id': s, 'description': s, 'status': 'pending'} for s in STEP_IDS],
    })
    workflow.artifacts.stamp_plan_provenance(workflow.session_id)
    monkeypatch.setattr(workflow, '_check_escalations', MagicMock(return_value=[]))
    monkeypatch.setattr(workflow, '_get_head_commit', AsyncMock(return_value='abc123'))
    monkeypatch.setattr(workflow, '_reconcile_done_step_commits', AsyncMock())
    monkeypatch.setattr(workflow, '_iteration_commit_provenance', AsyncMock(return_value={}))
    return workflow, sink, store


def _completing_invoke_agent(workflow: TaskWorkflow, first: AgentResult) -> AsyncMock:
    """Stub ``invoke_agent``: the call that returns a non-rejection completes every step."""
    calls = 0

    async def _invoke_agent(**kwargs) -> AgentResult:
        nonlocal calls
        calls += 1
        result = first if calls == 1 else AgentResult(success=True, output='ok', cost_usd=0.5)
        if not getattr(result, 'stderr', ''):
            assert workflow.artifacts is not None
            for step_id in STEP_IDS:
                workflow.artifacts.update_step_status(step_id, 'done')
        return result

    return AsyncMock(side_effect=_invoke_agent)


async def _run(workflow: TaskWorkflow, invoke_agent: AsyncMock) -> WorkflowOutcome:
    with (
        patch('orchestrator.workflow.invoke_agent', invoke_agent),
        patch.object(TaskWorkflow, '_build_agent_env', return_value=None),
    ):
        return await workflow._execute_iterations()


@pytest.mark.usefixtures('code_default_config')
@pytest.mark.asyncio
async def test_a_downgraded_resumed_run_that_completed_the_plan_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    workflow, sink, store = await _resumed_workflow(tmp_path, monkeypatch)
    invoke_agent = _completing_invoke_agent(workflow, _downgraded_end_turn())

    outcome = await _run(workflow, invoke_agent)

    invoke_agent.assert_awaited_once()
    assert invoke_agent.await_args_list[0].kwargs['resume_session_id'] == SID
    assert outcome == WorkflowOutcome.DONE
    assert workflow.artifacts is not None
    entries, _ = workflow.artifacts.read_iteration_log()
    assert sorted(entries[-1]['steps_completed']) == STEP_IDS
    assert store.of_type(EventType.session_resume_failed) == []
    assert sink.failures == []
    assert sink.successes == 1


@pytest.mark.usefixtures('code_default_config')
@pytest.mark.asyncio
async def test_a_rejected_resume_is_retried_fresh_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    workflow, sink, store = await _resumed_workflow(tmp_path, monkeypatch)
    invoke_agent = _completing_invoke_agent(workflow, measured_resume_rejection(SID))

    outcome = await _run(workflow, invoke_agent)

    assert invoke_agent.await_count == 2
    assert invoke_agent.await_args_list[0].kwargs['resume_session_id'] == SID
    assert 'resume_session_id' not in invoke_agent.await_args_list[1].kwargs
    assert outcome == WorkflowOutcome.DONE
    [emitted] = store.of_type(EventType.session_resume_failed)
    assert emitted['data']['stage'] == 'cli'
    assert emitted['data']['session_id'] == SID
    assert sink.only_failure().stage == 'cli'
