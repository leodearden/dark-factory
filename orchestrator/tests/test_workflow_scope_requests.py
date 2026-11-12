"""A writer's scope request is applied at the session boundary (task 3090).

Drives the REAL ``TaskWorkflow._execute_iterations`` and
``_verify_debugfix_loop`` against a real ``plan.json``.  The agent double files
a request through ``TaskArtifacts.write_scope_request``; the workflow locks
what is free, widens the plan, metadata.files and its modules together, and
briefs the next writer with the outcome.  A contested path changes nothing,
and no outcome re-pends or escalates.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from _recording_event_store import _RecordingEventStore
from _workflow_helpers import FakeScheduler
from shared.testing import make_gate_mock

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import PlanState, TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.scope_drift import (
    IMPLEMENT_BLAST_RADIUS_LOCK_CONFLICT,
    IMPLEMENT_SCOPE_EXPANDED,
    ScopeDriftReport,
    ScopeRequest,
)
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

_TASK_ID = '3090'
_DECLARED = 'pkg/a.py'
_REQUESTED = 'pkg/b.py'
_HOLDER = '777'


def _make_workflow(tmp_path: Path) -> tuple[TaskWorkflow, FakeScheduler, _RecordingEventStore]:
    assignment = MagicMock()
    assignment.task_id = _TASK_ID
    assignment.task = {
        'id': _TASK_ID, 'title': 'T', 'description': 'd', 'metadata': {'files': [_DECLARED]},
    }
    assignment.modules = [_DECLARED]

    cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    cfg.fused_memory.project_id = 'dark_factory'
    cfg.fused_memory.url = 'http://localhost:8002'
    cfg.lock_depth = 12
    cfg.for_module = MagicMock(return_value=None)
    cfg.project_root = tmp_path / 'proj'
    cfg.max_execute_iterations = 10
    cfg.max_consecutive_zero_output_timeouts = 2
    cfg.max_progress_resume_iterations = 20
    cfg.recycle_config_dir_on_zero_output = False
    cfg.judge_after_each_iteration = False
    cfg.inter_iteration_rebase = False
    cfg.timeouts.working_idle_secs = 1800.0
    cfg.invocation_timeout = 7200.0
    for role in ('implementer', 'debugger'):
        setattr(cfg.budgets, role, 5.0)
        setattr(cfg.max_turns, role, 50)
        setattr(cfg.backends, role, 'claude')
    cfg.sandbox.enabled = False
    cfg.git.main_branch = 'main'

    git_ops = MagicMock()
    git_ops.has_uncommitted_work = AsyncMock(return_value=False)
    git_ops.get_commit_subjects = AsyncMock(return_value=[])

    briefing = MagicMock()
    briefing.build_implementer_prompt = AsyncMock(return_value='Implement the plan')
    briefing.build_debugger_prompt = AsyncMock(return_value='Fix the failure')

    scheduler = FakeScheduler()
    events = _RecordingEventStore()
    wf = TaskWorkflow(
        assignment=assignment,
        config=cfg,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=briefing,
        mcp=MagicMock(),
        event_store=events,  # type: ignore[arg-type]
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    wf.worktree = worktree
    wf.artifacts = TaskArtifacts(worktree)
    wf.merge_queue = MagicMock()
    wf._module_configs = []
    return wf, scheduler, events


def _write_dispatched_plan(wf: TaskWorkflow, *, steps: int) -> str | None:
    """Write a confirmed plan and stamp it for this session; return its confirmed_at."""
    assert wf.artifacts is not None
    wf.artifacts.init(_TASK_ID, 'T', 'd')
    wf.artifacts.write_plan({
        'task_id': _TASK_ID,
        'title': 'T',
        'analysis': 'A',
        'files': [_DECLARED],
        'prerequisites': [],
        'steps': [
            {'id': f'step-{n}', 'type': 'impl', 'description': f'step {n}', 'status': 'pending'}
            for n in range(1, steps + 1)
        ],
        '_finalized_at': '2026-11-12T00:00:00+00:00',
        '_confirmed_by': 'architect',
    })
    wf.artifacts.stamp_plan_provenance(wf.session_id)
    wf.plan = wf.artifacts.read_plan()
    stamped = wf.artifacts.peek_plan_model()
    assert stamped is not None and stamped.state is PlanState.STAMPED
    return stamped.lifecycle.confirmed_at


def _stub_loop_collaborators(wf: TaskWorkflow, *, escalations: Callable[[], list] | None = None) -> None:
    assert wf.artifacts is not None
    wf.artifacts.validate_plan_owner = MagicMock(return_value=True)  # type: ignore[method-assign]
    wf.artifacts.stamp_plan_provenance = MagicMock()  # type: ignore[method-assign]
    wf._check_escalations = (  # type: ignore[method-assign]
        MagicMock(side_effect=escalations) if escalations else MagicMock(return_value=[])
    )
    wf._get_head_commit = AsyncMock(return_value='head0000')  # type: ignore[method-assign]
    wf._build_agent_env = MagicMock(return_value=None)  # type: ignore[method-assign]
    wf._reconcile_done_step_commits = AsyncMock()  # type: ignore[method-assign]
    wf.usage_gate = make_gate_mock(
        account_count=2,
        detect_cap_hit=MagicMock(return_value=False),
        scope_capacity_snapshot=MagicMock(return_value={}),
    )


def _success() -> AgentResult:
    return AgentResult(success=True, output='done', cost_usd=0.5, turns=3, duration_ms=5_000)


def _implementer_reporting_on_first_iteration(wf: TaskWorkflow):
    """Iteration N marks step-N done; iteration 1 also files the scope request."""
    assert wf.artifacts is not None
    artifacts = wf.artifacts
    calls = 0

    async def agent(**_kwargs) -> AgentResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            artifacts.write_scope_request(ScopeRequest(_REQUESTED, 'plan omitted it'))
        artifacts.update_step_status(f'step-{calls}', 'done', None)
        return _success()

    return agent


async def _run_execute_loop(wf: TaskWorkflow) -> WorkflowOutcome:
    with (
        patch(
            'orchestrator.workflow.invoke_agent', new_callable=AsyncMock,
            side_effect=_implementer_reporting_on_first_iteration(wf),
        ),
        patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
    ):
        return await wf._execute_iterations()


def _scope_drift_events(events: _RecordingEventStore) -> list[dict]:
    return [entry['data'] for (etype, entry) in events.events if etype == EventType.scope_drift]


def _scope_drift_kwargs(builder: AsyncMock) -> list[ScopeDriftReport | None]:
    return [call.kwargs.get('scope_drift') for call in builder.await_args_list]


def _files_written(scheduler: FakeScheduler) -> list[list[str]]:
    return [
        list(metadata.get('files') or [])
        for (_task_id, metadata) in scheduler.update_task_calls
        if isinstance(metadata, dict)
    ]


def _plan_files(wf: TaskWorkflow) -> list[str]:
    assert wf.artifacts is not None
    return list(wf.artifacts.read_plan()['files'])


def _assert_never_repended(scheduler: FakeScheduler) -> None:
    written = scheduler.statuses.get(_TASK_ID, [])
    assert 'pending' not in written and 'blocked' not in written


@pytest.mark.asyncio
class TestImplementerScopeRequest:
    async def test_a_free_path_is_locked_and_declared_without_unconfirming_the_plan(self, tmp_path):
        wf, scheduler, events = _make_workflow(tmp_path)
        confirmed_at = _write_dispatched_plan(wf, steps=2)
        _stub_loop_collaborators(wf)

        outcome = await _run_execute_loop(wf)

        assert outcome == WorkflowOutcome.DONE
        assert _plan_files(wf) == [_DECLARED, _REQUESTED]
        assert _REQUESTED in _files_written(scheduler)[-1]
        assert _REQUESTED in wf.modules
        assert scheduler.acquire_uncontested_calls == [
            (_TASK_ID, [_REQUESTED], IMPLEMENT_SCOPE_EXPANDED),
        ]
        _assert_never_repended(scheduler)
        assert _scope_drift_events(events) == [{
            'reason': IMPLEMENT_SCOPE_EXPANDED,
            'role': 'implementer',
            'iteration': 1,
            'requested': [_REQUESTED],
            'acquired': [_REQUESTED],
            'contested': {},
        }]
        second_brief = _scope_drift_kwargs(wf.briefing.build_implementer_prompt)[1]  # type: ignore[attr-defined]
        assert second_brief is not None
        assert second_brief.acquired == (_REQUESTED,)
        assert wf.artifacts is not None
        assert wf.artifacts.read_scope_requests() == ()
        widened = wf.artifacts.peek_plan_model()
        assert widened is not None
        assert widened.lifecycle.confirmed_at == confirmed_at
        assert widened.state is PlanState.STAMPED

    async def test_a_contested_path_advances_nothing_and_is_briefed_with_its_holder(self, tmp_path):
        wf, scheduler, events = _make_workflow(tmp_path)
        scheduler.contested_modules = {_REQUESTED: _HOLDER}
        _write_dispatched_plan(wf, steps=2)
        _stub_loop_collaborators(wf)

        outcome = await _run_execute_loop(wf)

        assert outcome == WorkflowOutcome.DONE
        assert _plan_files(wf) == [_DECLARED]
        assert not any(_REQUESTED in files for files in _files_written(scheduler))
        assert _REQUESTED not in wf.modules
        _assert_never_repended(scheduler)
        [drift] = _scope_drift_events(events)
        assert drift['reason'] == IMPLEMENT_BLAST_RADIUS_LOCK_CONFLICT
        assert drift['acquired'] == []
        assert drift['contested'] == {_REQUESTED: _HOLDER}
        second_brief = _scope_drift_kwargs(wf.briefing.build_implementer_prompt)[1]  # type: ignore[attr-defined]
        assert second_brief is not None
        assert second_brief.contested == ((_REQUESTED, _HOLDER),)

    async def test_an_iteration_without_a_request_briefs_no_drift(self, tmp_path):
        wf, scheduler, events = _make_workflow(tmp_path)
        _write_dispatched_plan(wf, steps=3)
        _stub_loop_collaborators(wf)

        outcome = await _run_execute_loop(wf)

        assert outcome == WorkflowOutcome.DONE
        briefs = _scope_drift_kwargs(wf.briefing.build_implementer_prompt)  # type: ignore[attr-defined]
        assert len(briefs) == 3
        assert briefs[1] is not None
        assert briefs[2] is None
        assert len(scheduler.acquire_uncontested_calls) == 1
        assert len(_scope_drift_events(events)) == 1

    async def test_requests_are_applied_before_the_escalation_gate(self, tmp_path):
        wf, _scheduler, _events = _make_workflow(tmp_path)
        _write_dispatched_plan(wf, steps=2)
        blocker = SimpleNamespace(severity='blocking', level=0)
        _stub_loop_collaborators(wf, escalations=lambda: [blocker])

        outcome = await _run_execute_loop(wf)

        assert outcome == WorkflowOutcome.ESCALATED
        assert _plan_files(wf) == [_DECLARED, _REQUESTED]


def _failing_verify() -> VerifyResult:
    return VerifyResult(
        passed=False, test_output='FAILED test_x', lint_output='', type_output='',
        summary='1 test failed', category='test_failure', cause_hint='boom',
    )


def _passing_verify() -> VerifyResult:
    return VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')


def _debugger_reporting_on_first_dispatch(wf: TaskWorkflow):
    assert wf.artifacts is not None
    artifacts = wf.artifacts
    calls = 0

    async def agent(**_kwargs) -> AgentResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            artifacts.write_scope_request(ScopeRequest(_REQUESTED, 'the fix lives here'))
        return _success()

    return agent


@pytest.mark.asyncio
class TestDebuggerScopeRequest:
    async def test_the_next_debugger_is_briefed_with_the_widened_scope(self, tmp_path):
        wf, _scheduler, events = _make_workflow(tmp_path)
        _write_dispatched_plan(wf, steps=1)
        _stub_loop_collaborators(wf)
        wf.config.rebase_before_verify = False
        wf.config.escalate_preexisting_main_break = False
        wf.config.max_opaque_timeout_attempts = 3
        wf.config.max_failure_signature_repeat = 5
        wf.config.max_verify_attempts = 5
        wf._run_scoped_verification_with_infra_retry = AsyncMock(  # type: ignore[method-assign]
            side_effect=[_failing_verify(), _failing_verify(), _passing_verify()],
        )
        wf._maybe_file_chronic_flakes = AsyncMock()  # type: ignore[method-assign]

        with (
            patch(
                'orchestrator.workflow.invoke_agent', new_callable=AsyncMock,
                side_effect=_debugger_reporting_on_first_dispatch(wf),
            ),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            outcome = await wf._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.DONE
        [drift] = _scope_drift_events(events)
        assert drift['role'] == 'debugger'
        assert drift['iteration'] == 1
        assert _plan_files(wf) == [_DECLARED, _REQUESTED]
        briefs = _scope_drift_kwargs(wf.briefing.build_debugger_prompt)  # type: ignore[attr-defined]
        assert len(briefs) == 2
        assert briefs[0] is None
        assert briefs[1] is not None
        assert briefs[1].acquired == (_REQUESTED,)
