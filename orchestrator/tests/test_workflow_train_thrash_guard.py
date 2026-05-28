"""Tests for steward-guard: train members bypass infra-resume + merge-outcome
thrash counters.

PRD § 9.8 — train members have their own loop-guard (γ₂/task 1523) for
verify-phase thrash and the train-merge worker owns the merge phase.  The
existing consecutive_infra_resume_failures and consecutive_merge_thrash
counters add no value for train members and risk false-trips on legitimate
merge-deferred → merge-deferred re-stamps.

Guard shape: `if isinstance(metadata.get('train'), dict): return None`
inserted immediately after `metadata = self.task.get('metadata') or {}`
in BOTH _check_infra_resume_thrash and _check_merge_outcome_thrash.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from escalation.models import Escalation

from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


# ---------------------------------------------------------------------------
# Shared fixture helpers (mirrors test_workflow_infra_thrash.py)
# ---------------------------------------------------------------------------

@dataclass
class _Fixture:
    wf: TaskWorkflow
    update_task: AsyncMock
    mark_blocked: AsyncMock
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
    max_consecutive_infra_resumes: int = 3,
    max_consecutive_merge_thrash: int = 3,
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd',
        'metadata': metadata or {},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = Path('/tmp/non-existent-for-test')
    config.max_consecutive_infra_resumes = max_consecutive_infra_resumes
    config.max_consecutive_merge_thrash = max_consecutive_merge_thrash

    update_task = AsyncMock(return_value=True)

    scheduler = MagicMock()
    scheduler.update_task = update_task
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')
    scheduler.get_task = AsyncMock(return_value={'id': task_id, 'metadata': metadata or {}})

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='SHA-A')

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

    # Stub artifacts.read_iteration_log()
    iter_log = list(iteration_log or [])
    wf.artifacts = MagicMock()
    wf.artifacts.read_iteration_log = MagicMock(return_value=(iter_log, []))

    # Stub _mark_blocked
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(
        wf=wf,
        update_task=update_task,
        mark_blocked=mark_blocked,
        queue=queue,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# (a) _check_infra_resume_thrash — train member short-circuit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_infra_resume_thrash_skips_train_member():
    """Train member → _check_infra_resume_thrash returns None immediately.

    No counter persisted, no _mark_blocked fired — the loop-guard (γ₂/1523)
    owns train verify-phase thrash.
    """
    f = _make(
        metadata={'train': {'id': 'T1', 'order': 1}},
        # resolved L0 with category='infra_issue' — would normally trigger counter
        resolved_l0s=[_esc(category='infra_issue')],
        iteration_log=[{'iteration': i} for i in range(5)],
    )

    outcome = await f.wf._check_infra_resume_thrash()

    assert outcome is None, f'Expected None for train member, got {outcome!r}'
    # No metadata update must have been persisted
    f.update_task.assert_not_awaited()
    # No escalation-to-human
    f.mark_blocked.assert_not_awaited()


@pytest.mark.asyncio
async def test_infra_resume_thrash_fires_for_non_train():
    """Regression: non-train task still increments counter + escalates at threshold.

    This guards against the train short-circuit accidentally swallowing
    non-train thrash detection.
    """
    f = _make(
        metadata={
            # No 'train' key — non-train task
            'consecutive_infra_resume_failures': 2,
            'last_infra_resume_iteration_count': 5,
        },
        resolved_l0s=[_esc(category='infra_issue')],
        iteration_log=[{'iteration': i} for i in range(5)],  # unchanged → counter increments
        max_consecutive_infra_resumes=3,
    )

    outcome = await f.wf._check_infra_resume_thrash()

    # At threshold (2+1=3) → escalates → WorkflowOutcome.BLOCKED
    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()


# ---------------------------------------------------------------------------
# (c) _check_merge_outcome_thrash — train member short-circuit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_merge_outcome_thrash_skips_train_member():
    """Train member → _check_merge_outcome_thrash returns None immediately.

    No counter persisted, no _mark_blocked fired — the train-merge worker
    owns the merge phase for train members.
    """
    f = _make(metadata={'train': {'id': 'T1', 'order': 1}})

    outcome = await f.wf._check_merge_outcome_thrash(
        prev_signature='sig-abc',
        current_signature='sig-abc',  # same signature → would increment counter
    )

    assert outcome is None, f'Expected None for train member, got {outcome!r}'
    f.update_task.assert_not_awaited()
    f.mark_blocked.assert_not_awaited()


@pytest.mark.asyncio
async def test_merge_outcome_thrash_fires_for_non_train():
    """Regression: non-train task still increments counter + escalates at threshold.

    counter=2, same signature → counter becomes 3 = threshold → BLOCKED.
    """
    f = _make(
        metadata={
            # No 'train' key — non-train task
            'consecutive_merge_thrash': 2,
            'last_merge_outcome_signature': 'sig-xyz',
        },
        max_consecutive_merge_thrash=3,
    )

    outcome = await f.wf._check_merge_outcome_thrash(
        prev_signature='sig-xyz',
        current_signature='sig-xyz',  # same signature → counter increments to 3
    )

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
