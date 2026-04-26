"""Tests for ``TaskWorkflow._handle_no_plan_failure`` (Fix C).

The helper increments ``consecutive_no_plan_failures`` keyed by
``last_no_plan_main_sha`` in the task metadata.  When the counter hits
``>= 2`` with the same main SHA, the workflow short-circuits to
BLOCKED + L1 instead of letting the steward attempt resolution again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow
    update_task: AsyncMock
    get_main_sha: AsyncMock
    mark_blocked: AsyncMock


def _make(
    *,
    project_root: Path,
    task_id: str = '99',
    main_sha: str = 'mainSHA-1',
    metadata: dict | None = None,
    update_task_raises: bool = False,
    get_main_sha_raises: bool = False,
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd',
        'metadata': metadata or {},
    }
    assignment.modules = ['mod_a']

    config = MagicMock()
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = project_root

    if update_task_raises:
        update_task = AsyncMock(side_effect=RuntimeError('mcp down'))
    else:
        update_task = AsyncMock(return_value=True)
    if get_main_sha_raises:
        get_main_sha = AsyncMock(side_effect=RuntimeError('git down'))
    else:
        get_main_sha = AsyncMock(return_value=main_sha)

    scheduler = MagicMock()
    scheduler.update_task = update_task
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')

    git_ops = MagicMock()
    git_ops.get_main_sha = get_main_sha

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )

    # Stub _mark_blocked — we only assert how _handle_no_plan_failure routes.
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(
        wf=wf,
        update_task=update_task,
        get_main_sha=get_main_sha,
        mark_blocked=mark_blocked,
    )


def _persisted_metadata(update_task: AsyncMock) -> dict:
    """Return the metadata dict passed to the most recent update_task call."""
    assert update_task.await_args is not None
    args, kwargs = update_task.await_args
    return kwargs.get('metadata') or args[1]


@pytest.mark.asyncio
async def test_first_failure_increments_counter_to_one_and_routes_normally(
    tmp_path: Path,
):
    f = _make(project_root=tmp_path, main_sha='SHA-A')

    outcome = await f.wf._handle_no_plan_failure(
        'no plan.json produced', detail='boom',
    )

    assert outcome == WorkflowOutcome.BLOCKED
    # _mark_blocked called WITHOUT escalate_to_human (normal steward path).
    f.mark_blocked.assert_awaited_once()
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is not True
    # Counter persisted to 1 with current main SHA.
    f.update_task.assert_awaited_once()
    metadata = _persisted_metadata(f.update_task)
    assert metadata['consecutive_no_plan_failures'] == 1
    assert metadata['last_no_plan_main_sha'] == 'SHA-A'


@pytest.mark.asyncio
async def test_second_failure_same_main_sha_escalates_to_human(tmp_path: Path):
    f = _make(
        project_root=tmp_path,
        main_sha='SHA-A',
        metadata={
            'last_no_plan_main_sha': 'SHA-A',
            'consecutive_no_plan_failures': 1,
        },
    )

    outcome = await f.wf._handle_no_plan_failure(
        'plan missing "steps"', detail='dump',
    )

    assert outcome == WorkflowOutcome.BLOCKED
    # _mark_blocked called with escalate_to_human=True (Fix C path).
    f.mark_blocked.assert_awaited_once()
    args, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True
    assert 'counter=2' in args[0] or 'Repeated no-plan' in args[0]
    metadata = _persisted_metadata(f.update_task)
    assert metadata['consecutive_no_plan_failures'] == 2


@pytest.mark.asyncio
async def test_main_sha_changed_resets_counter_to_one(tmp_path: Path):
    """If main advanced between failures, the counter resets — the missing
    premise may now exist."""
    f = _make(
        project_root=tmp_path,
        main_sha='SHA-NEW',
        metadata={
            'last_no_plan_main_sha': 'SHA-OLD',
            'consecutive_no_plan_failures': 5,
        },
    )

    outcome = await f.wf._handle_no_plan_failure(
        'no plan.json produced', detail='boom',
    )

    assert outcome == WorkflowOutcome.BLOCKED
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is not True
    metadata = _persisted_metadata(f.update_task)
    assert metadata['consecutive_no_plan_failures'] == 1
    assert metadata['last_no_plan_main_sha'] == 'SHA-NEW'


@pytest.mark.asyncio
async def test_third_consecutive_failure_still_escalates(tmp_path: Path):
    """Counter ≥ 2 always routes to escalate_to_human, not just exactly 2."""
    f = _make(
        project_root=tmp_path,
        main_sha='SHA-A',
        metadata={
            'last_no_plan_main_sha': 'SHA-A',
            'consecutive_no_plan_failures': 2,
        },
    )

    await f.wf._handle_no_plan_failure('still no plan', detail='')

    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True


@pytest.mark.asyncio
async def test_metadata_persistence_failure_is_non_fatal(tmp_path: Path):
    """If scheduler.update_task raises, we still route to _mark_blocked."""
    f = _make(
        project_root=tmp_path, main_sha='SHA-A',
        update_task_raises=True,
    )

    outcome = await f.wf._handle_no_plan_failure('no plan', detail='')

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_main_sha_failure_treats_as_unknown_main(tmp_path: Path):
    """If we can't read main SHA, counter resets to 1 (normal path)."""
    f = _make(
        project_root=tmp_path,
        metadata={
            'last_no_plan_main_sha': 'SHA-A',
            'consecutive_no_plan_failures': 5,
        },
        get_main_sha_raises=True,
    )

    outcome = await f.wf._handle_no_plan_failure('no plan', detail='')

    assert outcome == WorkflowOutcome.BLOCKED
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is not True


@pytest.mark.asyncio
async def test_corrupt_counter_metadata_treated_as_zero(tmp_path: Path):
    """If metadata has a non-int counter (e.g. legacy task), we don't crash."""
    f = _make(
        project_root=tmp_path,
        main_sha='SHA-A',
        metadata={
            'last_no_plan_main_sha': 'SHA-A',
            'consecutive_no_plan_failures': 'one',  # corrupt
        },
    )

    outcome = await f.wf._handle_no_plan_failure('no plan', detail='')

    assert outcome == WorkflowOutcome.BLOCKED
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is not True
    metadata = _persisted_metadata(f.update_task)
    assert metadata['consecutive_no_plan_failures'] == 1
