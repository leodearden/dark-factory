"""Tests for Fix 2 — anti-thrash guard on repeated infra-issue resumes.

Mirrors the style of ``test_workflow_no_plan_cycle.py``: builds a minimal
``TaskWorkflow`` with mocks and drives ``_check_infra_resume_thrash``
directly to assert state transitions.

The counter is keyed by iteration-log entry count (canonical "agent ran
real work" signal).  Steward fix-commits will reset the counter via
iteration-log growth — that is intentional and the counter resets to 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.models import Escalation

from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow
    update_task: AsyncMock
    mark_blocked: AsyncMock
    iteration_log: list[dict]
    queue: MagicMock


def _esc(
    *,
    task_id: str = '99',
    category: str = 'infra_issue',
    status: str = 'resolved',
    level: int = 0,
    resolved_at: str | None = '2026-04-27T12:00:00Z',
) -> Escalation:
    return Escalation(
        id=f'esc-{task_id}-1',
        task_id=task_id,
        agent_role='implementer',
        severity='blocking',
        category=category,
        summary='infra blocker',
        detail='infra detail',
        status=status,
        level=level,
        resolved_at=resolved_at,
    )


def _make(
    *,
    task_id: str = '99',
    metadata: dict | None = None,
    iteration_log: list[dict] | None = None,
    resolved_l0s: list[Escalation] | None = None,
    update_task_raises: bool = False,
    max_consecutive_infra_resumes: int = 3,
    no_queue: bool = False,
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
    config.project_root = Path('/tmp/non-existent-for-test')
    config.max_consecutive_infra_resumes = max_consecutive_infra_resumes

    if update_task_raises:
        update_task = AsyncMock(side_effect=RuntimeError('mcp down'))
    else:
        update_task = AsyncMock(return_value=True)

    scheduler = MagicMock()
    scheduler.update_task = update_task
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='SHA-A')

    if no_queue:
        queue = None
    else:
        queue = MagicMock()
        queue.get_by_task = MagicMock(return_value=resolved_l0s or [])

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=queue,  # type: ignore[arg-type]
    )

    # Stub artifacts.read_iteration_log() — no real .task/ directory needed.
    iter_log = list(iteration_log or [])
    wf.artifacts = MagicMock()
    wf.artifacts.read_iteration_log = MagicMock(return_value=(iter_log, []))

    # Stub _mark_blocked — we only care about how _check_infra_resume_thrash
    # routes.
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(
        wf=wf,
        update_task=update_task,
        mark_blocked=mark_blocked,
        iteration_log=iter_log,
        queue=queue,  # type: ignore[arg-type]
    )


def _persisted_metadata(update_task: AsyncMock) -> dict:
    assert update_task.await_args is not None
    args, kwargs = update_task.await_args
    return kwargs.get('metadata') or args[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_counter_increments_on_consecutive_infra_resumes_no_iter_growth():
    """Same iteration-log size + infra_issue category → counter increments."""
    f = _make(
        metadata={
            'consecutive_infra_resume_failures': 1,
            'last_infra_resume_iteration_count': 5,
        },
        iteration_log=[{'iteration': i} for i in range(5)],  # still 5
        resolved_l0s=[_esc(category='infra_issue')],
    )

    outcome = await f.wf._check_infra_resume_thrash()

    # Below threshold → fall through (None).
    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['consecutive_infra_resume_failures'] == 2
    assert md['last_infra_resume_iteration_count'] == 5
    f.mark_blocked.assert_not_awaited()


@pytest.mark.asyncio
async def test_counter_resets_to_one_when_iteration_log_grows():
    """Steward fix-commit advanced the iteration log → counter resets."""
    f = _make(
        metadata={
            'consecutive_infra_resume_failures': 2,
            'last_infra_resume_iteration_count': 5,
        },
        iteration_log=[{'iteration': i} for i in range(8)],  # grew 5 → 8
        resolved_l0s=[_esc(category='infra_issue')],
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['consecutive_infra_resume_failures'] == 1, (
        f'Counter must reset to 1 on iteration-log growth: {md}'
    )
    assert md['last_infra_resume_iteration_count'] == 8


@pytest.mark.asyncio
async def test_counter_resets_to_zero_on_non_infra_category():
    """task_failure / design_concern / etc. → reset to zero."""
    f = _make(
        metadata={
            'consecutive_infra_resume_failures': 2,
            'last_infra_resume_iteration_count': 5,
        },
        iteration_log=[{'iteration': i} for i in range(5)],  # unchanged
        resolved_l0s=[_esc(category='task_failure')],
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['consecutive_infra_resume_failures'] == 0
    assert md['last_infra_resume_iteration_count'] == 5


@pytest.mark.asyncio
async def test_counter_promotes_to_l1_at_threshold():
    """Counter reaches max_consecutive_infra_resumes → escalate_to_human=True."""
    f = _make(
        metadata={
            'consecutive_infra_resume_failures': 2,  # one below default 3
            'last_infra_resume_iteration_count': 5,
        },
        iteration_log=[{'iteration': i} for i in range(5)],  # unchanged
        resolved_l0s=[_esc(category='infra_issue')],
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    args, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True
    assert 'counter=3' in args[0] or 'thrash' in args[0].lower()


@pytest.mark.asyncio
async def test_threshold_is_configurable_below_default():
    """Lowering max_consecutive_infra_resumes promotes earlier."""
    f = _make(
        metadata={
            'consecutive_infra_resume_failures': 1,  # one below threshold=2
            'last_infra_resume_iteration_count': 0,
        },
        iteration_log=[],
        resolved_l0s=[_esc(category='infra_issue')],
        max_consecutive_infra_resumes=2,
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome == WorkflowOutcome.BLOCKED
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True


@pytest.mark.asyncio
async def test_no_queue_skips_classification_and_resets_counter():
    """Eval mode (no escalation queue) cannot classify the L0 → reset."""
    f = _make(
        metadata={
            'consecutive_infra_resume_failures': 2,
            'last_infra_resume_iteration_count': 5,
        },
        iteration_log=[{'iteration': i} for i in range(5)],
        no_queue=True,
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['consecutive_infra_resume_failures'] == 0


@pytest.mark.asyncio
async def test_metadata_persistence_failure_is_non_fatal():
    """If scheduler.update_task raises, we still route correctly."""
    f = _make(
        metadata={
            'consecutive_infra_resume_failures': 0,
            'last_infra_resume_iteration_count': 0,
        },
        iteration_log=[{'iteration': 1}],
        resolved_l0s=[_esc(category='infra_issue')],
        update_task_raises=True,
    )

    outcome = await f.wf._check_infra_resume_thrash()

    # Below threshold → None despite the persistence failure.
    assert outcome is None
    f.mark_blocked.assert_not_awaited()


@pytest.mark.asyncio
async def test_corrupt_counter_metadata_treated_as_zero():
    """Non-int counter (e.g. legacy task) must not crash the helper."""
    f = _make(
        metadata={
            'consecutive_infra_resume_failures': 'three',  # corrupt
            'last_infra_resume_iteration_count': 5,
        },
        iteration_log=[{'iteration': i} for i in range(5)],
        resolved_l0s=[_esc(category='infra_issue')],
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['consecutive_infra_resume_failures'] == 1


@pytest.mark.asyncio
async def test_picks_most_recent_resolved_l0_by_resolved_at():
    """Multiple resolved L0s: use the most recent one for category."""
    older = _esc(category='task_failure', resolved_at='2026-04-27T10:00:00Z')
    newer = _esc(category='infra_issue', resolved_at='2026-04-27T12:00:00Z')
    f = _make(
        metadata={
            'consecutive_infra_resume_failures': 2,  # threshold=3 default
            'last_infra_resume_iteration_count': 5,
        },
        iteration_log=[{'iteration': i} for i in range(5)],
        resolved_l0s=[older, newer],  # in any order
    )

    outcome = await f.wf._check_infra_resume_thrash()

    # Newer == infra_issue → counter increments to 3 → threshold hit.
    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True


@pytest.mark.asyncio
async def test_metadata_round_trips_via_scheduler_update():
    """Persisted metadata is the same dict layout the helper expects on
    the next invocation — defends against typo regressions on the metadata
    keys."""
    f = _make(
        metadata={},
        iteration_log=[{'iteration': i} for i in range(3)],
        resolved_l0s=[_esc(category='infra_issue')],
    )

    await f.wf._check_infra_resume_thrash()

    md = _persisted_metadata(f.update_task)
    # Both keys must be present and machine-readable on the next call.
    assert isinstance(md['consecutive_infra_resume_failures'], int)
    assert isinstance(md['last_infra_resume_iteration_count'], int)
    assert md['consecutive_infra_resume_failures'] == 1
    assert md['last_infra_resume_iteration_count'] == 3
