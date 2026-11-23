"""An architect phase that already confirmed its plan is neither re-dispatched unbriefed nor overridden by an exit report (task 6814).

``TestReDispatchAfterConfirmIsDeclined`` drives the REAL ``_invoke`` and the
REAL ``invoke_with_cap_retry`` against a real ``plan.json``; only the agent
subprocess (``invoke_agent``) and the usage gate are doubles.
``TestConfirmedPlanAndExitReportConflict`` scripts ``_invoke`` itself, so what
the architect left on disk is the only input.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from _recording_event_store import _RecordingEventStore
from shared.testing import make_gate_mock

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome
from orchestrator.workflow_types import DryRunSpawn

_TASK_ID = '6814'
T0 = '2026-11-23T10:00:00+00:00'
T1 = '2026-11-23T11:00:00+00:00'


def _make_workflow(tmp_path: Path, *, event_store: _RecordingEventStore | None = None) -> TaskWorkflow:
    assignment = MagicMock()
    assignment.task_id = _TASK_ID
    assignment.task = {'id': _TASK_ID, 'title': 'T', 'description': 'd'}
    assignment.modules = []

    cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    cfg.fused_memory.project_id = 'dark_factory'
    cfg.fused_memory.url = 'http://localhost:8002'
    cfg.project_root = tmp_path / 'proj'
    cfg.timeouts.working_idle_secs = 1800.0
    cfg.invocation_timeout = 7200.0
    cfg.budgets.architect = 5.0
    cfg.max_turns.architect = 50
    cfg.backends.architect = 'claude'
    cfg.sandbox.enabled = False
    cfg.git.main_branch = 'main'

    escalation_queue = MagicMock()
    escalation_queue.get_by_task.return_value = []

    wf = TaskWorkflow(
        assignment=assignment,
        config=cfg,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
        event_store=event_store,  # type: ignore[arg-type]
        escalation_queue=escalation_queue,
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    wf.worktree = worktree
    wf.artifacts = TaskArtifacts(worktree)
    wf.artifacts.init(_TASK_ID, 'T', 'd')
    wf.merge_queue = MagicMock()
    wf._module_configs = []
    wf._build_agent_env = MagicMock(return_value=None)  # type: ignore[method-assign]
    wf.usage_gate = make_gate_mock(
        account_count=2,
        detect_cap_hit=MagicMock(side_effect=[True, False]),
        scope_capacity_snapshot=MagicMock(return_value={}),
    )
    return wf


def _write_plan(artifacts: TaskArtifacts, *, finalized_at: str | None) -> None:
    plan: dict = {
        'task_id': _TASK_ID,
        'title': 'T',
        'analysis': 'A',
        'files': ['orchestrator/src/orchestrator/x.py'],
        'prerequisites': [],
        'steps': [
            {'id': 'step-1', 'type': 'test', 'description': 'one', 'status': 'pending'},
            {'id': 'step-2', 'type': 'impl', 'description': 'two', 'status': 'pending'},
        ],
    }
    if finalized_at is not None:
        plan['_finalized_at'] = finalized_at
    artifacts.write_plan(plan)


def _cap_hit_fresh() -> AgentResult:
    return AgentResult(
        success=False, output="You've hit your usage limit", session_id='',
        cost_usd=0.3, turns=1, duration_ms=8_000,
    )


def _success() -> AgentResult:
    return AgentResult(success=True, output='planned', cost_usd=0.5, turns=3, duration_ms=5_000)


async def _run_architect(wf: TaskWorkflow, agent: Callable, builder: AsyncMock) -> tuple[object, AsyncMock]:
    with (
        patch('orchestrator.workflow.invoke_agent', new_callable=AsyncMock, side_effect=agent) as inv,
        patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
    ):
        outcome = await wf._run_architect_for_plan(builder)
    return outcome, inv


def _events(rec: _RecordingEventStore, event_type: EventType) -> list[dict]:
    return [entry for (etype, entry) in rec.events if etype == event_type]


@pytest.mark.asyncio
class TestReDispatchAfterConfirmIsDeclined:
    async def test_retry_after_the_run_confirmed_its_plan_is_declined_and_salvaged(self, tmp_path):
        rec = _RecordingEventStore()
        wf = _make_workflow(tmp_path, event_store=rec)
        artifacts = wf.artifacts
        assert artifacts is not None
        builder = AsyncMock(return_value='BRIEF')

        async def agent(**_kwargs):
            _write_plan(artifacts, finalized_at=T1)
            return _cap_hit_fresh()

        outcome, inv = await _run_architect(wf, agent, builder)

        assert inv.await_count == 1
        assert outcome is None
        assert wf.plan['_finalized_at'] == T1
        assert _events(rec, EventType.invocation_end)[-1]['data']['retry_aborted'] is True
        assert _events(rec, EventType.plan_salvaged)

    async def test_retry_of_an_unconfirmed_plan_still_dispatches(self, tmp_path):
        wf = _make_workflow(tmp_path)
        artifacts = wf.artifacts
        assert artifacts is not None
        builder = AsyncMock(return_value='BRIEF')
        prompts: list[str] = []

        async def agent(**kwargs):
            prompts.append(kwargs['prompt'])
            if len(prompts) == 1:
                _write_plan(artifacts, finalized_at=None)
                return _cap_hit_fresh()
            _write_plan(artifacts, finalized_at=T1)
            return _success()

        outcome, inv = await _run_architect(wf, agent, builder)

        assert inv.await_count == 2
        assert prompts[1] == 'BRIEF'
        assert outcome is None

    async def test_revalidation_reconfirmed_with_a_new_stamp_is_declined(self, tmp_path):
        wf = _make_workflow(tmp_path)
        artifacts = wf.artifacts
        assert artifacts is not None
        _write_plan(artifacts, finalized_at=T0)
        builder = AsyncMock(return_value='BRIEF')

        async def agent(**_kwargs):
            _write_plan(artifacts, finalized_at=T1)
            return _cap_hit_fresh()

        outcome, inv = await _run_architect(wf, agent, builder)

        assert inv.await_count == 1
        assert outcome is None
        assert wf.plan['_finalized_at'] == T1

    async def test_revalidation_leaving_the_prior_stamp_still_retries(self, tmp_path):
        wf = _make_workflow(tmp_path)
        artifacts = wf.artifacts
        assert artifacts is not None
        _write_plan(artifacts, finalized_at=T0)
        builder = AsyncMock(return_value='BRIEF')
        calls: list[str] = []

        async def agent(**kwargs):
            calls.append(kwargs['prompt'])
            if len(calls) == 1:
                return _cap_hit_fresh()
            return _success()

        _outcome, inv = await _run_architect(wf, agent, builder)

        assert inv.await_count == 2
        assert calls[1] == 'BRIEF'


def _make_scripted(
    tmp_path: Path, architect_run: Callable[[TaskArtifacts], None],
) -> tuple[TaskWorkflow, AsyncMock]:
    """A workflow whose ``_invoke`` runs *architect_run* against the real artifacts and succeeds, and its ``_mark_blocked``."""
    wf = _make_workflow(tmp_path)
    artifacts = wf.artifacts
    assert artifacts is not None

    async def invoke(role, build_prompt, cwd, output_schema=None):
        architect_run(artifacts)
        return AgentResult(success=True, output='done', cost_usd=1.2, turns=14, duration_ms=60_000)

    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._invoke = AsyncMock(side_effect=invoke)  # type: ignore[method-assign]
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]
    return wf, mark_blocked


def _only_block(mark_blocked: AsyncMock) -> tuple[str, dict]:
    """The reason and keyword arguments of the single ``_mark_blocked`` call."""
    mark_blocked.assert_awaited_once()
    call = mark_blocked.await_args
    assert call is not None
    (reason,), kwargs = call
    return reason, kwargs


def _confirm_and(report: Callable[[TaskArtifacts], None]) -> Callable[[TaskArtifacts], None]:
    def architect_run(artifacts: TaskArtifacts) -> None:
        _write_plan(artifacts, finalized_at=T1)
        report(artifacts)
    return architect_run


def _unactionable(artifacts: TaskArtifacts) -> None:
    artifacts.write_unactionable_task('spec unworkable', 'ev')


def _blocking_dependency(artifacts: TaskArtifacts) -> None:
    artifacts.write_blocking_dependency('5763', 'needs the router first', 'main0000')


@pytest.mark.asyncio
class TestConfirmedPlanAndExitReportConflict:
    async def test_plan_confirmed_alongside_an_unactionable_report_blocks_on_the_conflict(self, tmp_path):
        wf, mark_blocked = _make_scripted(tmp_path, _confirm_and(_unactionable))
        artifacts = wf.artifacts
        assert artifacts is not None

        outcome = await wf._run_architect_for_plan(AsyncMock(return_value='BRIEF'))

        assert outcome == WorkflowOutcome.BLOCKED
        reason, kwargs = _only_block(mark_blocked)
        assert kwargs['escalate_to_human'] is True
        assert kwargs['category'] == 'design_concern'
        assert kwargs['dry_run_spawn'] is DryRunSpawn.NEVER
        assert 'unactionable_task' in kwargs['detail']
        assert T1 in kwargs['detail']
        assert not reason.startswith('Architect reported task unactionable')
        assert artifacts.read_unactionable_task() is None
        plan = artifacts.read_plan()
        assert plan['_finalized_at'] == T1
        assert len(plan['steps']) == 2

    async def test_plan_confirmed_alongside_a_blocking_dependency_blocks_on_the_conflict(self, tmp_path):
        wf, mark_blocked = _make_scripted(tmp_path, _confirm_and(_blocking_dependency))
        handle_dep = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        wf._handle_blocking_dep_report = handle_dep  # type: ignore[method-assign]

        outcome = await wf._run_architect_for_plan(AsyncMock(return_value='BRIEF'))

        assert outcome == WorkflowOutcome.BLOCKED
        _reason, kwargs = _only_block(mark_blocked)
        assert kwargs['category'] == 'design_concern'
        assert 'blocking_dependency' in kwargs['detail']
        handle_dep.assert_not_awaited()
        assert wf.artifacts is not None
        assert wf.artifacts.read_blocking_dependency() is None

    async def test_rejection_over_a_plan_confirmed_before_the_phase_still_wins(self, tmp_path):
        wf, mark_blocked = _make_scripted(tmp_path, _unactionable)
        assert wf.artifacts is not None
        _write_plan(wf.artifacts, finalized_at=T0)

        outcome = await wf._run_architect_for_plan(AsyncMock(return_value='BRIEF'))

        assert outcome == WorkflowOutcome.BLOCKED
        reason, _kwargs = _only_block(mark_blocked)
        assert reason.startswith('Architect reported task unactionable')

    async def test_a_confirmed_plan_without_a_report_proceeds(self, tmp_path):
        wf, mark_blocked = _make_scripted(tmp_path, lambda artifacts: _write_plan(artifacts, finalized_at=T1))

        outcome = await wf._run_architect_for_plan(AsyncMock(return_value='BRIEF'))

        assert outcome is None
        mark_blocked.assert_not_awaited()
