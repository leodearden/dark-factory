"""The workflow's plan-schema-repair and replan architect dispatches.

Both are briefed by the BriefingAssembler for this task, and a repaired plan
advances only when the architect itself attested it with ``confirm_plan`` —
the ``_finalized_at`` gate in ``TaskWorkflow._plan`` is the sole judge, so
the orchestrator never stamps the marker on a repair.

Driven through ``_plan()`` / ``_replan()`` with the agent-dispatch seam
(``_invoke``) scripted, as test_workflow_architect_transient_retry.py does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from shared.cli_invoke import AgentResult

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import BlastRadiusResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

_ARCHITECT_FINALIZED_AT = '2026-10-05T00:00:00+00:00'


@dataclass
class _Fixture:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    briefing: MagicMock
    invoke: AsyncMock
    handle_no_plan: AsyncMock


def _succeeded() -> AgentResult:
    return AgentResult(
        success=True, output='done', cost_usd=1.2, duration_ms=90_000, turns=14,
    )


def _make(tmp_path: Path) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = '5356'
    assignment.task = {'id': '5356', 'title': 'T', 'description': 'd', 'metadata': {}}
    assignment.modules = ['mod_a']

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.project_root = tmp_path / 'proj'
    config.max_review_cycles = 2

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_status = AsyncMock(return_value='in-progress')

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='currentmain')

    escalation_queue = MagicMock()
    escalation_queue.get_by_task = MagicMock(return_value=[])

    briefing = MagicMock()
    briefing.build_architect_prompt = AsyncMock(return_value='ARCHITECT PROMPT')
    briefing.build_plan_schema_repair_prompt = AsyncMock(return_value='REPAIR PROMPT')
    briefing.build_replan_prompt = AsyncMock(return_value='REPLAN PROMPT')

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=briefing,
        mcp=MagicMock(),
        escalation_queue=escalation_queue,
    )

    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init('5356', 'T', 'd', base_commit='oldbase')
    wf.artifacts = artifacts
    wf.worktree = worktree

    invoke = AsyncMock(return_value=_succeeded())
    wf._invoke = invoke  # type: ignore[method-assign]
    wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]
    handle_no_plan = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._handle_no_plan_failure = handle_no_plan  # type: ignore[method-assign]
    wf._reconcile_scope_locks = AsyncMock(  # type: ignore[method-assign]
        return_value=BlastRadiusResult(applied=True),
    )
    wf._write_decisions_to_memory = AsyncMock()  # type: ignore[method-assign]

    return _Fixture(
        wf=wf, artifacts=artifacts, briefing=briefing,
        invoke=invoke, handle_no_plan=handle_no_plan,
    )


def _write_stepless_plan(artifacts: TaskArtifacts) -> None:
    """A plan whose step content sits under a key no normalization rule knows."""
    artifacts.write_plan({
        'task_id': '5356',
        'title': 'T',
        'analysis': 'A' * 30_000,
        'files': ['mod_a/x.py'],
        'work_items': ['do a', 'do b'],
        'prerequisites': [],
        'design_decisions': [],
        'reuse': [],
    })


def _record_steps(artifacts: TaskArtifacts, *, finalized_at: str | None) -> None:
    """What a repairing architect leaves: steps recorded, attested or not."""
    plan = artifacts.read_plan()
    plan['steps'] = [
        {'id': 'step-1', 'type': 'test', 'description': 'do a',
         'status': 'pending', 'commit': None},
        {'id': 'step-2', 'type': 'impl', 'description': 'do b',
         'status': 'pending', 'commit': None},
    ]
    if finalized_at is not None:
        plan['_finalized_at'] = finalized_at
    artifacts.write_plan(plan)


async def _drive_plan(
    tmp_path: Path, repair: Callable[[TaskArtifacts], None],
) -> tuple[_Fixture, WorkflowOutcome]:
    """Architect call 1 leaves a stepless plan; call 2 (the repair) runs *repair*."""
    f = _make(tmp_path)
    calls = {'n': 0}

    async def scripted(*_args, **_kwargs) -> AgentResult:
        calls['n'] += 1
        if calls['n'] == 1:
            _write_stepless_plan(f.artifacts)
        else:
            repair(f.artifacts)
        return _succeeded()

    f.invoke.side_effect = scripted
    return f, await f.wf._plan()


class TestSchemaRepairDispatch:
    @pytest.mark.asyncio
    async def test_repair_dispatch_is_the_assembler_prompt_for_this_task(
        self, tmp_path: Path,
    ):
        f, _ = await _drive_plan(
            tmp_path, lambda a: _record_steps(a, finalized_at=_ARCHITECT_FINALIZED_AT),
        )

        assert f.invoke.await_count == 2
        assert f.invoke.await_args_list[1].args[1] == 'REPAIR PROMPT'
        f.briefing.build_plan_schema_repair_prompt.assert_awaited_once()
        assert f.briefing.build_plan_schema_repair_prompt.await_args.args[0] is f.wf.task

    @pytest.mark.asyncio
    async def test_unconfirmed_repair_is_routed_as_unfinalized(self, tmp_path: Path):
        f, _ = await _drive_plan(tmp_path, lambda a: _record_steps(a, finalized_at=None))

        f.handle_no_plan.assert_awaited_once()
        assert f.handle_no_plan.await_args.args[0].startswith(
            'Planning failed: plan not finalized',
        )
        assert '_finalized_at' not in f.artifacts.read_plan()

    @pytest.mark.asyncio
    async def test_confirmed_repair_advances_to_planned(self, tmp_path: Path):
        f, outcome = await _drive_plan(
            tmp_path, lambda a: _record_steps(a, finalized_at=_ARCHITECT_FINALIZED_AT),
        )

        assert outcome is WorkflowOutcome.PLANNED
        f.handle_no_plan.assert_not_awaited()
        assert f.artifacts.read_plan()['_finalized_at'] == _ARCHITECT_FINALIZED_AT

    @pytest.mark.asyncio
    async def test_repair_that_records_no_steps_fails_as_missing_steps(
        self, tmp_path: Path,
    ):
        f, _ = await _drive_plan(tmp_path, lambda _a: None)

        f.handle_no_plan.assert_awaited_once()
        assert f.handle_no_plan.await_args.args[0] == 'Planning failed: plan missing "steps"'


class TestReplanDispatch:
    @pytest.mark.asyncio
    async def test_replan_dispatch_is_the_assembler_prompt_for_this_task_and_feedback(
        self, tmp_path: Path,
    ):
        f = _make(tmp_path)
        f.artifacts.write_plan({
            'task_id': '5356',
            'title': 'T',
            'files': ['mod_a/x.py'],
            'prerequisites': [],
            'steps': [
                {'id': 'step-1', 'type': 'test', 'description': 'do a',
                 'status': 'done', 'commit': 'abc'},
                {'id': 'step-2', 'type': 'impl', 'description': 'do b',
                 'status': 'done', 'commit': 'def'},
            ],
            '_finalized_at': _ARCHITECT_FINALIZED_AT,
        })
        reviews = MagicMock()
        reviews.format_for_replan.return_value = 'FEEDBACK'

        await f.wf._replan(reviews)

        f.briefing.build_replan_prompt.assert_awaited_once()
        assert f.briefing.build_replan_prompt.await_args.args == (f.wf.task, 'FEEDBACK')
        f.invoke.assert_awaited_once()
        assert f.invoke.await_args.args[1] == 'REPLAN PROMPT'
