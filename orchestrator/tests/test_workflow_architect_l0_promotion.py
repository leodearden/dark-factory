"""Tests for the architect-filed-L0-without-plan early-promote path (RC2a).

When the architect calls ``escalate_blocker`` from plan-tools MCP and exits
without writing ``.task/plan.json`` or any rejection artifact, the workflow
must detect the new L0 (compared to the pre-architect snapshot) and route
directly to ``_mark_blocked(escalate_to_human=True, skip_escalation=True)``,
bypassing both ``_handle_no_plan_failure`` and the steward.  This stops the
no-plan loop after the FIRST occurrence rather than letting the cycle
counter reset on every main-SHA bump (the bug behind 16 successive Opus
calls on task 917).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.models import Escalation
from shared.cli_invoke import AgentResult

from _orch_helpers import pydantic_spec
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    escalation_queue: MagicMock
    mark_blocked: AsyncMock
    handle_no_plan: AsyncMock
    invoke: AsyncMock


def _make(
    *,
    worktree: Path,
    project_root: Path,
    task_id: str = '917',
    new_l0: Escalation | None = None,
) -> _Fixture:
    """Build a TaskWorkflow ready for ``_plan()`` invocation.

    The escalation_queue stub returns ``[]`` for the pre-architect snapshot
    and ``[new_l0]`` (when provided) for the post-architect check.  A wrapper
    list lets us flip the second call's result without rebuilding the mock.
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd', 'metadata': {},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    _spec.for_module = None  # custom method not in model_fields; expose to spec_set
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = project_root
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_status = AsyncMock(return_value='in-progress')

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='currentmain')

    escalation_queue = MagicMock()
    # First call (pre-architect snapshot) → empty.
    # Second call (post-architect) → [new_l0] when provided, [] otherwise.
    post_value = [new_l0] if new_l0 is not None else []
    escalation_queue.get_by_task = MagicMock(side_effect=[[], post_value])

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=escalation_queue,
    )

    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'T', 'd', base_commit='oldbase')
    wf.artifacts = artifacts
    wf.worktree = worktree

    # Stub the architect invoke — succeeds but writes nothing (no plan,
    # no rejection artifact).  The L0 it would have filed via plan-tools
    # MCP is simulated by the escalation_queue side_effect.
    invoke = AsyncMock(
        return_value=AgentResult(
            success=True,
            output='filed L0',
            cost_usd=0.50,
            duration_ms=10_000,
            turns=8,
        ),
    )
    wf._invoke = invoke  # type: ignore[method-assign]

    # Stub _mark_blocked + _handle_no_plan_failure so we can observe the
    # routing decision without exercising their full machinery.
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]
    handle_no_plan = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._handle_no_plan_failure = handle_no_plan  # type: ignore[method-assign]

    # The architect prompt builder is unused in this path but needs to exist.
    wf.briefing.build_architect_prompt = AsyncMock(return_value='prompt')
    wf.briefing.build_revalidation_prompt = AsyncMock(return_value='prompt')

    return _Fixture(
        wf=wf,
        artifacts=artifacts,
        escalation_queue=escalation_queue,
        mark_blocked=mark_blocked,
        handle_no_plan=handle_no_plan,
        invoke=invoke,
    )


def _make_l0(
    task_id: str = '917',
    summary: str = 'plan-tools missing_premise: helpers.foo not found',
    detail: str = 'Architect could not locate helpers.foo on main',
) -> Escalation:
    return Escalation(
        id=f'esc-{task_id}-001',
        task_id=task_id,
        agent_role='architect',
        severity='blocking',
        category='missing_premise',
        summary=summary,
        detail=detail,
        level=0,
        status='pending',
    )


@pytest.mark.asyncio
async def test_new_l0_promotes_to_l1_directly(tmp_path: Path):
    """Architect fires escalate_blocker without writing plan.json →
    workflow auto-promotes to _mark_blocked(escalate_to_human=True,
    skip_escalation=True) with the architect's summary/detail surfaced."""
    new_l0 = _make_l0(
        summary='plan-tools missing_premise: helpers.foo not found',
        detail='Architect could not locate helpers.foo on main',
    )
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        new_l0=new_l0,
    )

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.BLOCKED
    # _mark_blocked was called with the right flags and the architect's
    # summary surfaced into the L1 reason/detail.
    f.mark_blocked.assert_awaited_once()
    args, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True
    assert kwargs.get('skip_escalation') is True, (
        'skip_escalation must be True — the architect already filed an L0; '
        'creating a duplicate would clutter the queue'
    )
    assert 'helpers.foo not found' in args[0]
    assert 'Architect could not locate helpers.foo on main' in kwargs['detail']
    # _handle_no_plan_failure must NOT fire — the deterministic catch
    # short-circuits before the cycle counter is touched.
    f.handle_no_plan.assert_not_called()


@pytest.mark.asyncio
async def test_pre_existing_l0_not_treated_as_new(tmp_path: Path):
    """If the same L0 is present BOTH before and after the architect runs
    (e.g. left over from a prior run), it must NOT trigger the early-promote.
    The check filters by id-not-in-pre-set.
    """
    pre_existing = _make_l0()
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
    )
    # Override side_effect — both calls return the same L0 (no new ones).
    f.escalation_queue.get_by_task = MagicMock(
        side_effect=[[pre_existing], [pre_existing]],
    )

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.BLOCKED
    # Must fall through to _handle_no_plan_failure (the existing path).
    f.handle_no_plan.assert_awaited_once()
    f.mark_blocked.assert_not_called()


@pytest.mark.asyncio
async def test_no_l0_falls_through_to_no_plan_handler(tmp_path: Path):
    """When the architect produces no plan AND files no L0, the workflow
    must fall through to the existing ``_handle_no_plan_failure`` path
    (the RC2b total-counter backstop applies there)."""
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        new_l0=None,
    )

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.BLOCKED
    f.handle_no_plan.assert_awaited_once()
    f.mark_blocked.assert_not_called()
