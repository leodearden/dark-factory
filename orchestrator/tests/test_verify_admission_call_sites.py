"""Each verify call site hands verify its admission role and its event store (task 5628).

A gated verify records its admission wait as a ``verify_admission_plan`` row only
when its caller passes an event store, and the periodic review checkpoint queues
behind task verifies only when it says it is a background verify
(plans/verify-admission-fifo-prd.md, resolved design decisions 4 and 5).
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import (
    MOCK_WORKFLOW_PROJECT_ROOT,
    _init_harness_state_for_test,
    pydantic_spec,
    stamp_stock_routing_config,
)
from _recording_event_store import _RecordingEventStore
from shared.cli_invoke import AllAccountsCappedException

from orchestrator import verify as verify_module
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.harness import Harness
from orchestrator.review_checkpoint import ReviewCheckpoint
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

_PASSED = VerifyResult(
    passed=True, summary='OK', test_output='', lint_output='', type_output='',
)


def _awaited_kwargs(mock: AsyncMock) -> Mapping[str, Any]:
    mock.assert_awaited_once()
    assert mock.await_args is not None
    return mock.await_args.kwargs


def _review_checkpoint(event_store: _RecordingEventStore) -> ReviewCheckpoint:
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = Path('/tmp')
    config.models.deep_reviewer = 'opus'
    config.max_turns.deep_reviewer = 50
    config.budgets.deep_reviewer = 10.0
    config.effort.deep_reviewer = 'max'
    config.backends.deep_reviewer = 'claude'
    config.fused_memory.project_id = 'test-project'
    config.escalation.host = 'localhost'
    config.escalation.port = 9999
    stamp_stock_routing_config(config)
    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}
    checkpoint = ReviewCheckpoint(config, mcp=mcp, usage_gate=None)
    checkpoint.event_store = event_store  # type: ignore[assignment]
    return checkpoint


def _workflow_on_mock_config(event_store: _RecordingEventStore) -> TaskWorkflow:
    task_id = '5628'
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': '', 'metadata': {}}
    assignment.modules = ['mod_a']

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = MOCK_WORKFLOW_PROJECT_ROOT
    config.max_consecutive_infra_resumes = 3
    config.verify_infra_retry_max_attempts = 3
    config.verify_infra_retry_backoff_secs = 0.01
    config.verify_infra_retry_max_backoff_secs = 0.1
    config.max_verify_attempts = 5
    config.max_failure_signature_repeat = 3
    config.max_opaque_timeout_attempts = 3
    config.rebase_before_verify = False
    config.escalate_preexisting_main_break = False

    scheduler = MagicMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')
    scheduler.get_task = AsyncMock(return_value={'id': task_id, 'metadata': {}})
    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='SHA-A')
    queue = MagicMock()
    queue.get_by_task = MagicMock(return_value=[])

    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=queue,
        event_store=event_store,  # type: ignore[arg-type]
    )
    workflow.worktree = Path('/tmp/fake-worktree-5628')
    workflow.artifacts = MagicMock()
    workflow.artifacts.read_iteration_log = MagicMock(return_value=([], []))
    return workflow


def _workflow_on_real_lane(tmp_path: Path, event_store: _RecordingEventStore) -> TaskWorkflow:
    project_root = tmp_path / 'project'
    project_root.mkdir()
    config = OrchestratorConfig(
        project_root=project_root,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False,
        ),
    )
    git_ops = GitOps(config.git, config.project_root)
    worktree = git_ops.worktree_base / '42'
    worktree.mkdir(parents=True)
    assignment = TaskAssignment(
        task_id='42',
        task={
            'id': '42', 'title': 'X', 'description': '', 'status': 'pending',
            'metadata': {'files': ['lib']}, 'dependencies': [],
        },
        modules=['lib'],
    )
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=MagicMock(),  # type: ignore[arg-type]
        briefing=MagicMock(),  # type: ignore[arg-type]
        mcp=MagicMock(),  # type: ignore[arg-type]
        event_store=event_store,  # type: ignore[arg-type]
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': []}
    return workflow


class _StopMergePhase(Exception):
    """Raised by the merge-phase re-verify spy to unwind before the merge submit."""


def _sweep_harness(event_store: _RecordingEventStore) -> Harness:
    harness = Harness.__new__(Harness)
    _init_harness_state_for_test(harness)
    harness.config = OrchestratorConfig()
    harness.git_ops = MagicMock()
    harness.git_ops.get_main_sha = AsyncMock(return_value='b' * 40)
    harness._last_swept_main_sha = None
    harness._run_store = None
    harness._main_sweep_cold_control = None
    harness._close_superseded_main_sweep_escalations = AsyncMock()  # type: ignore[method-assign]
    harness.event_store = event_store  # type: ignore[assignment]
    return harness


@pytest.mark.asyncio
async def test_review_checkpoint_verifies_as_background_on_its_event_store(monkeypatch):
    store = _RecordingEventStore()
    verify = AsyncMock(return_value=_PASSED)
    monkeypatch.setattr('orchestrator.review_checkpoint.run_full_verification', verify)
    monkeypatch.setattr(
        'orchestrator.review_checkpoint.invoke_with_cap_retry',
        AsyncMock(side_effect=AllAccountsCappedException(
            retries=1, elapsed_secs=0.0, label='Review checkpoint [x]',
        )),
    )

    await _review_checkpoint(store).run_focused()

    kwargs = _awaited_kwargs(verify)
    assert kwargs['role'] == 'background'
    assert kwargs['event_store'] is store


@pytest.mark.asyncio
async def test_implement_phase_verify_carries_task_id_and_event_store():
    store = _RecordingEventStore()
    workflow = _workflow_on_mock_config(store)
    verify = AsyncMock(return_value=_PASSED)

    with patch('orchestrator.workflow.run_scoped_verification', verify):
        outcome = await workflow._verify_debugfix_loop()

    assert outcome == WorkflowOutcome.DONE
    kwargs = _awaited_kwargs(verify)
    assert kwargs['event_store'] is store
    assert kwargs['task_id'] == workflow.task_id


@pytest.mark.asyncio
async def test_post_rebase_reverify_carries_task_id_and_event_store(tmp_path, monkeypatch):
    store = _RecordingEventStore()
    workflow = _workflow_on_real_lane(tmp_path, store)
    workflow._maybe_defer_as_train_member = AsyncMock(return_value=None)  # type: ignore[method-assign]
    workflow._enter_phase = MagicMock()  # type: ignore[method-assign]
    workflow._check_scope_invariant = AsyncMock(return_value=None)  # type: ignore[method-assign]
    workflow._recover_before_merge = AsyncMock(return_value=None)  # type: ignore[method-assign]
    workflow.git_ops.get_main_sha = AsyncMock(return_value='mainsha')  # type: ignore[method-assign]
    workflow.git_ops.rebase_onto_main = AsyncMock(return_value=True)  # type: ignore[method-assign]
    verify = AsyncMock(side_effect=_StopMergePhase)
    monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', verify)
    monkeypatch.setattr(
        'orchestrator.workflow._run', AsyncMock(return_value=(0, 'headsha\n', '')),
    )

    with pytest.raises(_StopMergePhase):
        await workflow._run_merge_phase('task/42')

    kwargs = _awaited_kwargs(verify)
    assert kwargs['event_store'] is store
    assert kwargs['task_id'] == workflow.task_id


@pytest.mark.asyncio
async def test_main_tip_sweep_records_on_the_harness_event_store():
    store = _RecordingEventStore()
    harness = _sweep_harness(store)

    async def _passing_sweep(config, git_ops, *, main_sha, **kwargs):
        return (main_sha, _PASSED)

    sweep = AsyncMock(side_effect=_passing_sweep)
    with patch.object(verify_module, 'run_main_tip_sweep', new=sweep):
        await harness._run_main_tip_sweep()

    assert _awaited_kwargs(sweep)['event_store'] is store
