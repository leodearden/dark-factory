"""Each verify call site hands verify its admission role and its event store (task 5628).

A gated verify records its admission wait as a ``verify_admission_plan`` row only
when its caller passes an event store, and the periodic review checkpoint queues
behind task verifies only when it says it is a background verify
(plans/verify-admission-fifo-prd.md, resolved design decisions 4 and 5). The
implement-phase verify runs end to end, through the real verify and flock, onto
a real event store (PRD scenario 11).
"""
from __future__ import annotations

import asyncio
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from _orch_helpers import (
    pydantic_spec,
    stamp_stock_routing_config,
)
from _recording_event_store import _RecordingEventStore
from shared.cli_invoke import AllAccountsCappedException

from orchestrator import verify as verify_module
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig, load_config
from orchestrator.event_store import EventStore, EventType
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
    checkpoint.event_store = cast(EventStore, event_store)
    return checkpoint


_LANE_GIT = {
    'main_branch': 'main', 'branch_prefix': 'task/', 'remote': 'origin',
    'worktree_dir': '.worktrees', 'push_after_advance': False,
}


def _lane_config(tmp_path: Path, **overrides: Any) -> OrchestratorConfig:
    project_root = tmp_path / 'project'
    project_root.mkdir()
    return OrchestratorConfig(
        project_root=project_root, git=GitConfig(**_LANE_GIT), **overrides,
    )


def _gated_lane_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, exit_code: int = 0,
) -> OrchestratorConfig:
    """A project whose one module, ``lib``, has a test command exiting *exit_code*
    (passing by default), loaded the way the orchestrator loads it, with verify
    admission on in a tmp slots dir. The project root is *tmp_path*, where
    conftest's ORCH_PROJECT_ROOT pins it."""
    project_root = tmp_path
    (project_root / 'lib').mkdir()
    sleep = (
        f'{shlex.quote(sys.executable)} -c '
        f'"import sys, time; time.sleep(0.05); sys.exit({exit_code})"'
    )
    (project_root / 'lib' / 'orchestrator.yaml').write_text(yaml.safe_dump({
        'test_command': sleep, 'lint_command': None, 'type_check_command': None,
    }))
    config_path = project_root / 'dark-factory-orchestrator.yaml'
    config_path.write_text(yaml.safe_dump({
        'project_root': str(project_root),
        'git': _LANE_GIT,
        'rebase_before_verify': False,
        'verify_admission_slots_dir': str(tmp_path / 'slots'),
        'verify_admission_nice_task': 'nice -n 10',
        'verify_admission_nice_background': 'nice -n 10',
        'verify_clock_stop_enabled': False,
    }))
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_path))
    return load_config(config_path)


def _workflow_on_real_lane(
    config: OrchestratorConfig, event_store: EventStore | _RecordingEventStore,
) -> TaskWorkflow:
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
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
        event_store=cast(EventStore, event_store),
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': []}
    return workflow


class _StopMergePhase(Exception):
    """Raised by the merge-phase re-verify spy to unwind before the merge submit."""


def _sweep_harness(tmp_path: Path, event_store: _RecordingEventStore) -> Harness:
    harness = Harness(OrchestratorConfig(project_root=tmp_path))
    harness.git_ops = MagicMock()
    harness.git_ops.get_main_sha = AsyncMock(return_value='b' * 40)
    harness.event_store = cast(EventStore, event_store)
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


@pytest.mark.real_verify_admission
@pytest.mark.usefixtures('code_default_config')
@pytest.mark.asyncio
async def test_implement_phase_verify_records_one_admission_plan_for_its_task(
    tmp_path, monkeypatch,
):
    store = EventStore(tmp_path / 'runs.db', run_id='t')
    workflow = _workflow_on_real_lane(_gated_lane_config(tmp_path, monkeypatch), store)

    async with asyncio.timeout(60):
        outcome = await workflow._verify_debugfix_loop()

    assert outcome == WorkflowOutcome.DONE
    [row] = store.fetch_events_by_type(EventType.verify_admission_plan)
    assert row['task_id'] == workflow.task_id
    data = row['data']
    assert (data['outcome'], data['role'], data['task_id']) == ('returned', 'task', workflow.task_id)
    [leg] = data['legs']
    assert (leg['module'], leg['slot_held'], leg['rc']) == ('lib', True, 0)


@pytest.mark.asyncio
async def test_post_rebase_reverify_carries_task_id_and_event_store(tmp_path, monkeypatch):
    """Rollback arm only: the re-verify runs when ``pre_merge_reverify`` is on (task 6580)."""
    store = _RecordingEventStore()
    workflow = _workflow_on_real_lane(_lane_config(tmp_path, pre_merge_reverify=True), store)
    monkeypatch.setattr(workflow, '_maybe_defer_as_train_member', AsyncMock(return_value=None))
    monkeypatch.setattr(workflow, '_check_scope_invariant', AsyncMock(return_value=None))
    workflow.git_ops.get_main_sha = AsyncMock(return_value='mainsha')
    workflow.git_ops.rebase_onto_main = AsyncMock(return_value=True)
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


@pytest.mark.real_verify_admission
@pytest.mark.usefixtures('code_default_config')
@pytest.mark.asyncio
async def test_default_merge_phase_records_no_admission_plan(tmp_path, monkeypatch):
    """Task 6580: by default a rebased tip whose task-scope verify is RED reaches
    the merge-queue submit without taking a task admission slot."""
    store = EventStore(tmp_path / 'runs.db', run_id='t')
    workflow = _workflow_on_real_lane(
        _gated_lane_config(tmp_path, monkeypatch, exit_code=1), store,
    )
    monkeypatch.setattr(workflow, '_maybe_defer_as_train_member', AsyncMock(return_value=None))
    monkeypatch.setattr(workflow, '_check_scope_invariant', AsyncMock(return_value=None))
    monkeypatch.setattr(workflow.git_ops, 'get_main_sha', AsyncMock(return_value='mainsha'))
    monkeypatch.setattr(workflow.git_ops, 'rebase_onto_main', AsyncMock(return_value=True))
    monkeypatch.setattr(
        'orchestrator.workflow._run', AsyncMock(return_value=(0, 'headsha\n', '')),
    )
    submit = AsyncMock(side_effect=_StopMergePhase)
    monkeypatch.setattr(workflow, '_submit_to_merge_queue', submit)

    async with asyncio.timeout(60):
        with pytest.raises(_StopMergePhase):
            await workflow._run_merge_phase('task/42')

    assert store.fetch_events_by_type(EventType.verify_admission_plan) == []
    assert _awaited_kwargs(submit)['pre_rebased'] is True


@pytest.mark.asyncio
async def test_main_tip_sweep_records_on_the_harness_event_store(tmp_path):
    store = _RecordingEventStore()
    harness = _sweep_harness(tmp_path, store)

    async def _passing_sweep(config, git_ops, *, main_sha, **kwargs):
        return (main_sha, _PASSED)

    sweep = AsyncMock(side_effect=_passing_sweep)
    with patch.object(verify_module, 'run_main_tip_sweep', new=sweep):
        await harness._run_main_tip_sweep()

    assert _awaited_kwargs(sweep)['event_store'] is store
