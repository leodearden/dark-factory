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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from _workflow_helpers import FakeScheduler
from shared.testing import make_gate_mock

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import PlanState, TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.scope_drift import ScopeDriftReason, ScopeDriftReport, ScopeRequest
from orchestrator.verify import VerifyResult
from orchestrator.workflow import MAX_SCOPE_DRIFT_FOLLOWUPS, TaskWorkflow, WorkflowOutcome
from orchestrator.workflow_types import WorkflowState

_TASK_ID = '3090'
_DECLARED = 'pkg/a.py'
_REQUESTED = 'pkg/b.py'
_HOLDER = '777'


@dataclass(frozen=True)
class _Rig:
    wf: TaskWorkflow
    scheduler: FakeScheduler
    events: EventStore
    cfg: MagicMock
    implementer_prompt: AsyncMock
    debugger_prompt: AsyncMock
    amender_prompt: AsyncMock
    followup_prompt: AsyncMock


def _make_workflow(tmp_path: Path) -> _Rig:
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

    implementer_prompt = AsyncMock(return_value='Implement the plan')
    debugger_prompt = AsyncMock(return_value='Fix the failure')
    amender_prompt = AsyncMock(return_value='Amend')
    followup_prompt = AsyncMock(return_value='Settle the contested file')
    briefing = MagicMock()
    briefing.build_scope_drift_followup_prompt = followup_prompt
    briefing.build_implementer_prompt = implementer_prompt
    briefing.build_debugger_prompt = debugger_prompt
    briefing.build_amender_prompt = amender_prompt

    scheduler = FakeScheduler()
    events = EventStore(tmp_path / 'runs.db', 'run-1')
    wf = TaskWorkflow(
        assignment=assignment,
        config=cfg,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=briefing,
        mcp=MagicMock(),
        event_store=events,
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    wf.worktree = worktree
    wf.artifacts = TaskArtifacts(worktree)
    wf.merge_queue = MagicMock()
    wf._module_configs = []
    return _Rig(
        wf=wf,
        scheduler=scheduler,
        events=events,
        cfg=cfg,
        implementer_prompt=implementer_prompt,
        debugger_prompt=debugger_prompt,
        amender_prompt=amender_prompt,
        followup_prompt=followup_prompt,
    )


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


def _stub_loop_collaborators(
    wf: TaskWorkflow,
    monkeypatch: pytest.MonkeyPatch,
    *,
    escalations: Callable[[], list] | None = None,
    stub_owner_check: bool = True,
    stub_restamp: bool = True,
) -> None:
    assert wf.artifacts is not None
    if stub_owner_check:
        monkeypatch.setattr(wf.artifacts, 'validate_plan_owner', MagicMock(return_value=True))
    if stub_restamp:
        monkeypatch.setattr(wf.artifacts, 'stamp_plan_provenance', MagicMock())
    monkeypatch.setattr(
        wf,
        '_check_escalations',
        MagicMock(side_effect=escalations) if escalations else MagicMock(return_value=[]),
    )
    monkeypatch.setattr(wf, '_get_head_commit', AsyncMock(return_value='head0000'))
    monkeypatch.setattr(wf, '_build_agent_env', MagicMock(return_value=None))
    monkeypatch.setattr(wf, '_reconcile_done_step_commits', AsyncMock())
    wf.usage_gate = make_gate_mock(
        account_count=2,
        detect_cap_hit=MagicMock(return_value=False),
        scope_capacity_snapshot=MagicMock(return_value={}),
    )


def _success() -> AgentResult:
    return AgentResult(success=True, output='done', cost_usd=0.5, turns=3, duration_ms=5_000)


def _failed_before_its_first_turn() -> AgentResult:
    return AgentResult(success=False, output='', turns=0, duration_ms=1_000)


def _timed_out(*, transcript_turns: int) -> AgentResult:
    return AgentResult(
        success=False, output='', cost_usd=0.5, turns=transcript_turns, duration_ms=5_000,
        timed_out=True, transcript_turns=transcript_turns,
    )


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


def _scope_drift_events(store: EventStore) -> list[dict]:
    return [row['data'] for row in store.fetch_events_by_type(EventType.scope_drift)]


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
    async def test_a_free_path_is_locked_and_declared_without_unconfirming_the_plan(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf, scheduler, events = rig.wf, rig.scheduler, rig.events
        confirmed_at = _write_dispatched_plan(wf, steps=2)
        _stub_loop_collaborators(wf, monkeypatch)

        outcome = await _run_execute_loop(wf)

        assert outcome == WorkflowOutcome.DONE
        assert _plan_files(wf) == [_DECLARED, _REQUESTED]
        assert _REQUESTED in _files_written(scheduler)[-1]
        assert _REQUESTED in wf.modules
        assert scheduler.acquire_uncontested_calls == [
            (_TASK_ID, [_REQUESTED], ScopeDriftReason.IMPLEMENT_SCOPE_EXPANDED),
        ]
        _assert_never_repended(scheduler)
        assert _scope_drift_events(events) == [{
            'reason': ScopeDriftReason.IMPLEMENT_SCOPE_EXPANDED,
            'boundary': 'execute',
            'role': 'implementer',
            'iteration': 1,
            'requested': [_REQUESTED],
            'acquired': [_REQUESTED],
            'contested': {},
        }]
        second_brief = _scope_drift_kwargs(rig.implementer_prompt)[1]
        assert second_brief is not None
        assert second_brief.acquired == (_REQUESTED,)
        assert wf.artifacts is not None
        assert wf.artifacts.read_scope_requests() == ()
        widened = wf.artifacts.peek_plan_model()
        assert widened is not None
        assert widened.lifecycle.confirmed_at == confirmed_at
        assert widened.state is PlanState.STAMPED

    async def test_a_contested_path_advances_nothing_and_is_briefed_with_its_holder(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf, scheduler, events = rig.wf, rig.scheduler, rig.events
        scheduler.contested_modules = {_REQUESTED: _HOLDER}
        _write_dispatched_plan(wf, steps=2)
        _stub_loop_collaborators(wf, monkeypatch)

        outcome = await _run_execute_loop(wf)

        assert outcome == WorkflowOutcome.DONE
        assert _plan_files(wf) == [_DECLARED]
        assert not any(_REQUESTED in files for files in _files_written(scheduler))
        assert _REQUESTED not in wf.modules
        _assert_never_repended(scheduler)
        [drift] = _scope_drift_events(events)
        assert drift['reason'] == ScopeDriftReason.IMPLEMENT_BLAST_RADIUS_LOCK_CONFLICT
        assert drift['acquired'] == []
        assert drift['contested'] == {_REQUESTED: _HOLDER}
        second_brief = _scope_drift_kwargs(rig.implementer_prompt)[1]
        assert second_brief is not None
        assert second_brief.contested == ((_REQUESTED, _HOLDER),)

    async def test_an_iteration_without_a_request_briefs_no_drift(self, tmp_path, monkeypatch):
        rig = _make_workflow(tmp_path)
        wf, scheduler, events = rig.wf, rig.scheduler, rig.events
        _write_dispatched_plan(wf, steps=3)
        _stub_loop_collaborators(wf, monkeypatch)

        outcome = await _run_execute_loop(wf)

        assert outcome == WorkflowOutcome.DONE
        briefs = _scope_drift_kwargs(rig.implementer_prompt)
        assert len(briefs) == 3
        assert briefs[1] is not None
        assert briefs[2] is None
        assert len(scheduler.acquire_uncontested_calls) == 1
        assert len(_scope_drift_events(events)) == 1

    async def test_a_contested_request_in_the_last_session_gets_a_follow_up_session(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf, scheduler = rig.wf, rig.scheduler
        scheduler.contested_modules = {_REQUESTED: _HOLDER}
        _write_dispatched_plan(wf, steps=1)
        _stub_loop_collaborators(wf, monkeypatch)
        agent = AsyncMock(side_effect=_implementer_reporting_on_first_iteration(wf))

        with (
            patch('orchestrator.workflow.invoke_agent', agent),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            outcome = await wf._execute_iterations()
            again = await wf._execute_iterations()

        assert outcome == again == WorkflowOutcome.DONE
        [followup_call] = rig.followup_prompt.await_args_list
        drift = followup_call.args[1]
        assert drift.contested == ((_REQUESTED, _HOLDER),)
        assert agent.await_count == 2
        _assert_never_repended(scheduler)

    async def test_a_contested_request_before_a_progress_timeout_resume_gets_a_follow_up(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf, scheduler = rig.wf, rig.scheduler
        scheduler.contested_modules = {_REQUESTED: _HOLDER}
        _write_dispatched_plan(wf, steps=1)
        _stub_loop_collaborators(wf, monkeypatch)
        assert wf.artifacts is not None
        artifacts = wf.artifacts
        calls = 0

        async def agent(**_kwargs) -> AgentResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                artifacts.write_scope_request(ScopeRequest(_REQUESTED, 'plan omitted it'))
                return _timed_out(transcript_turns=5)
            if calls == 2:
                artifacts.update_step_status('step-1', 'done', None)
            return _success()

        with (
            patch('orchestrator.workflow.invoke_agent', AsyncMock(side_effect=agent)),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        assert wf.metrics.progress_resume_total == 1
        [followup_call] = rig.followup_prompt.await_args_list
        assert followup_call.args[1].contested == ((_REQUESTED, _HOLDER),)

    async def test_a_zero_output_session_does_not_consume_the_brief(self, tmp_path, monkeypatch):
        rig = _make_workflow(tmp_path)
        wf, scheduler = rig.wf, rig.scheduler
        scheduler.contested_modules = {_REQUESTED: _HOLDER}
        rig.cfg.max_consecutive_zero_output_timeouts = 3
        _write_dispatched_plan(wf, steps=2)
        _stub_loop_collaborators(wf, monkeypatch)
        assert wf.artifacts is not None
        artifacts = wf.artifacts
        calls = 0

        async def agent(**_kwargs) -> AgentResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                artifacts.write_scope_request(ScopeRequest(_REQUESTED, 'plan omitted it'))
                artifacts.update_step_status('step-1', 'done', None)
            elif calls == 2:
                return _timed_out(transcript_turns=0)
            else:
                artifacts.update_step_status('step-2', 'done', None)
            return _success()

        with (
            patch('orchestrator.workflow.invoke_agent', AsyncMock(side_effect=agent)),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        briefs = _scope_drift_kwargs(rig.implementer_prompt)
        assert len(briefs) == 3
        after_the_wedge = briefs[2]
        assert after_the_wedge is not None
        assert after_the_wedge.contested == ((_REQUESTED, _HOLDER),)

    async def test_a_granted_request_in_the_last_session_needs_no_follow_up(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf = rig.wf
        _write_dispatched_plan(wf, steps=1)
        _stub_loop_collaborators(wf, monkeypatch)

        outcome = await _run_execute_loop(wf)

        assert outcome == WorkflowOutcome.DONE
        assert _plan_files(wf) == [_DECLARED, _REQUESTED]
        rig.followup_prompt.assert_not_awaited()

    async def test_a_contested_report_survives_the_escalation_gate_to_the_next_brief(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf, scheduler = rig.wf, rig.scheduler
        scheduler.contested_modules = {_REQUESTED: _HOLDER}
        _write_dispatched_plan(wf, steps=2)
        blocker = SimpleNamespace(severity='blocking', level=0)
        gates = iter([[blocker]])
        _stub_loop_collaborators(wf, monkeypatch, escalations=lambda: next(gates, []))

        with (
            patch(
                'orchestrator.workflow.invoke_agent', new_callable=AsyncMock,
                side_effect=_implementer_reporting_on_first_iteration(wf),
            ),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            escalated = await wf._execute_iterations()
            resumed = await wf._execute_iterations()

        assert escalated == WorkflowOutcome.ESCALATED
        assert resumed == WorkflowOutcome.DONE
        resumed_brief = _scope_drift_kwargs(rig.implementer_prompt)[1]
        assert resumed_brief is not None
        assert resumed_brief.contested == ((_REQUESTED, _HOLDER),)
        rig.followup_prompt.assert_not_awaited()

    async def test_a_session_that_failed_before_its_first_turn_does_not_consume_the_brief(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf, scheduler = rig.wf, rig.scheduler
        scheduler.contested_modules = {_REQUESTED: _HOLDER}
        _write_dispatched_plan(wf, steps=2)
        _stub_loop_collaborators(wf, monkeypatch)
        assert wf.artifacts is not None
        artifacts = wf.artifacts
        calls = 0

        async def agent(**_kwargs) -> AgentResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                artifacts.write_scope_request(ScopeRequest(_REQUESTED, 'plan omitted it'))
                artifacts.update_step_status('step-1', 'done', None)
            elif calls == 2:
                return _failed_before_its_first_turn()
            else:
                artifacts.update_step_status('step-2', 'done', None)
            return _success()

        with (
            patch('orchestrator.workflow.invoke_agent', AsyncMock(side_effect=agent)),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        after_the_failure = _scope_drift_kwargs(rig.implementer_prompt)[-1]
        assert after_the_failure is not None
        assert after_the_failure.contested == ((_REQUESTED, _HOLDER),)

    async def test_a_request_kept_off_an_overwritten_plan_is_applied_after_the_restamp(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf, scheduler, events = rig.wf, rig.scheduler, rig.events
        _write_dispatched_plan(wf, steps=1)
        _stub_loop_collaborators(wf, monkeypatch, stub_owner_check=False, stub_restamp=False)
        assert wf.artifacts is not None and wf.worktree is not None
        artifacts, worktree = wf.artifacts, wf.worktree

        async def agent(**_kwargs) -> AgentResult:
            artifacts.write_scope_request(ScopeRequest(_REQUESTED, 'plan omitted it'))
            artifacts.update_step_status('step-1', 'done', None)
            TaskArtifacts(worktree).stamp_plan_provenance('foreign-session')
            return _success()

        with (
            patch('orchestrator.workflow.invoke_agent', AsyncMock(side_effect=agent)),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        assert _plan_files(wf) == [_DECLARED, _REQUESTED]
        assert len(scheduler.acquire_uncontested_calls) == 1
        assert artifacts.read_scope_requests() == ()
        [drift] = _scope_drift_events(events)
        assert (drift['boundary'], drift['iteration']) == ('execute', 1)

    async def test_a_retry_after_a_failed_declare_still_widens_the_module_set(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf, scheduler = rig.wf, rig.scheduler
        _write_dispatched_plan(wf, steps=2)
        _stub_loop_collaborators(wf, monkeypatch)
        assert wf.artifacts is not None
        set_plan_files = wf.artifacts.set_plan_files
        writes = 0

        def set_plan_files_failing_once(files, session_id):
            nonlocal writes
            writes += 1
            if writes == 1:
                raise OSError('plan.json write failed')
            return set_plan_files(files, session_id)

        monkeypatch.setattr(wf.artifacts, 'set_plan_files', set_plan_files_failing_once)

        outcome = await _run_execute_loop(wf)

        assert outcome == WorkflowOutcome.DONE
        assert len(scheduler.acquire_uncontested_calls) == 2
        assert _plan_files(wf) == [_DECLARED, _REQUESTED]
        assert _REQUESTED in wf.modules

    async def test_requests_are_applied_before_the_escalation_gate(self, tmp_path, monkeypatch):
        wf = _make_workflow(tmp_path).wf
        _write_dispatched_plan(wf, steps=2)
        blocker = SimpleNamespace(severity='blocking', level=0)
        _stub_loop_collaborators(wf, monkeypatch, escalations=lambda: [blocker])

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
    async def test_the_next_debugger_is_briefed_with_the_widened_scope(self, tmp_path, monkeypatch):
        rig = _make_workflow(tmp_path)
        wf, events = rig.wf, rig.events
        _write_dispatched_plan(wf, steps=1)
        _stub_loop_collaborators(wf, monkeypatch)
        rig.cfg.rebase_before_verify = False
        rig.cfg.escalate_preexisting_main_break = False
        rig.cfg.max_opaque_timeout_attempts = 3
        rig.cfg.max_failure_signature_repeat = 5
        rig.cfg.max_verify_attempts = 5
        monkeypatch.setattr(
            wf,
            '_run_scoped_verification_with_infra_retry',
            AsyncMock(side_effect=[_failing_verify(), _failing_verify(), _passing_verify()]),
        )
        monkeypatch.setattr(wf, '_maybe_file_chronic_flakes', AsyncMock())

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
        assert drift['boundary'] == 'debug_fix'
        assert drift['iteration'] == 1
        assert _plan_files(wf) == [_DECLARED, _REQUESTED]
        briefs = _scope_drift_kwargs(rig.debugger_prompt)
        assert len(briefs) == 2
        assert briefs[0] is None
        assert briefs[1] is not None
        assert briefs[1].acquired == (_REQUESTED,)


_SUGGESTIONS = [{'reviewer': 'r', 'category': 'c', 'location': _DECLARED, 'description': 'd'}]


def _amender_reporting_on_first_dispatch(wf: TaskWorkflow, *, overwrite_plan: bool = False):
    """The first pass files a request; with *overwrite_plan* it also restamps plan.json.

    The restamp goes through a fresh TaskArtifacts over the same root, the way
    an agent-side write would, so the workflow's own patched stamp is bypassed.
    """
    assert wf.artifacts is not None and wf.worktree is not None
    artifacts = wf.artifacts
    worktree = wf.worktree
    calls = 0

    async def agent(**_kwargs) -> AgentResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            artifacts.write_scope_request(ScopeRequest(_REQUESTED, 'the fix lives here'))
            if overwrite_plan:
                TaskArtifacts(worktree).stamp_plan_provenance('foreign-session')
        return _success()

    return agent


async def _run_amendment_pass(wf: TaskWorkflow, *, overwrite_plan: bool = False) -> bool:
    with (
        patch(
            'orchestrator.workflow.invoke_agent', new_callable=AsyncMock,
            side_effect=_amender_reporting_on_first_dispatch(wf, overwrite_plan=overwrite_plan),
        ),
        patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
    ):
        return await wf._amend(_SUGGESTIONS, 1)


@pytest.mark.asyncio
class TestAmendmentPassScopeRequest:
    async def test_a_request_filed_during_an_amendment_pass_is_applied_not_orphaned(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf, scheduler, events = rig.wf, rig.scheduler, rig.events
        _write_dispatched_plan(wf, steps=1)
        _stub_loop_collaborators(wf, monkeypatch)

        ok = await _run_amendment_pass(wf)

        assert ok is True
        assert _plan_files(wf) == [_DECLARED, _REQUESTED]
        assert scheduler.acquire_uncontested_calls == [
            (_TASK_ID, [_REQUESTED], ScopeDriftReason.IMPLEMENT_SCOPE_EXPANDED),
        ]
        [drift] = _scope_drift_events(events)
        assert drift['role'] == 'implementer'
        assert drift['boundary'] == 'amendment'
        assert drift['acquired'] == [_REQUESTED]
        assert wf.artifacts is not None
        assert wf.artifacts.read_scope_requests() == ()

    async def test_the_event_phase_is_the_workflow_phase_the_boundary_ran_in(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf, events = rig.wf, rig.events
        _write_dispatched_plan(wf, steps=1)
        _stub_loop_collaborators(wf, monkeypatch)
        wf._enter_phase(WorkflowState.REVIEW)

        assert await _run_amendment_pass(wf) is True

        [row] = events.fetch_events_by_type(EventType.scope_drift)
        assert row['phase'] == 'review'
        assert row['data']['boundary'] == 'amendment'

    async def test_a_scope_request_never_restamps_a_plan_the_agent_overwrote(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf, scheduler, events = rig.wf, rig.scheduler, rig.events
        _write_dispatched_plan(wf, steps=1)
        _stub_loop_collaborators(wf, monkeypatch, stub_owner_check=False)
        overwrite = MagicMock()
        monkeypatch.setattr(wf, '_escalate_plan_overwrite', overwrite)

        ok = await _run_amendment_pass(wf, overwrite_plan=True)

        assert ok is False
        overwrite.assert_called_once()
        assert scheduler.acquire_uncontested_calls == []
        assert _plan_files(wf) == [_DECLARED]
        assert _scope_drift_events(events) == []
        assert wf.artifacts is not None
        assert wf.artifacts.read_scope_requests() == (
            ScopeRequest(_REQUESTED, 'the fix lives here'),
        )

    async def test_a_zero_output_amendment_pass_does_not_consume_the_brief(
        self, tmp_path, monkeypatch
    ):
        rig = _make_workflow(tmp_path)
        wf, scheduler = rig.wf, rig.scheduler
        scheduler.contested_modules = {_REQUESTED: _HOLDER}
        _write_dispatched_plan(wf, steps=1)
        _stub_loop_collaborators(wf, monkeypatch)
        assert wf.artifacts is not None
        artifacts = wf.artifacts
        passes = iter([
            lambda: artifacts.write_scope_request(ScopeRequest(_REQUESTED, 'the fix lives here')),
            None,
            lambda: None,
        ])

        async def agent(**_kwargs) -> AgentResult:
            files_request = next(passes)
            if files_request is None:
                return _timed_out(transcript_turns=0)
            files_request()
            return _success()

        with (
            patch('orchestrator.workflow.invoke_agent', AsyncMock(side_effect=agent)),
            patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
        ):
            for amendment_round in (1, 2, 3):
                assert await wf._amend(_SUGGESTIONS, amendment_round) is True

        first, wedged, after_the_wedge = _scope_drift_kwargs(rig.amender_prompt)
        assert first is None
        assert wedged is not None
        assert after_the_wedge is not None
        assert after_the_wedge.contested == ((_REQUESTED, _HOLDER),)


@pytest.mark.asyncio
async def test_a_contested_debugger_request_before_a_green_verify_is_followed_up_and_reverified(
    tmp_path, monkeypatch
):
    rig = _make_workflow(tmp_path)
    wf, scheduler = rig.wf, rig.scheduler
    scheduler.contested_modules = {_REQUESTED: _HOLDER}
    _write_dispatched_plan(wf, steps=1)
    _stub_loop_collaborators(wf, monkeypatch)
    rig.cfg.rebase_before_verify = False
    rig.cfg.escalate_preexisting_main_break = False
    rig.cfg.max_opaque_timeout_attempts = 3
    rig.cfg.max_failure_signature_repeat = 5
    rig.cfg.max_verify_attempts = 5
    verify = AsyncMock(side_effect=[_failing_verify(), _passing_verify(), _passing_verify()])
    monkeypatch.setattr(wf, '_run_scoped_verification_with_infra_retry', verify)
    monkeypatch.setattr(wf, '_maybe_file_chronic_flakes', AsyncMock())

    with (
        patch(
            'orchestrator.workflow.invoke_agent', new_callable=AsyncMock,
            side_effect=_debugger_reporting_on_first_dispatch(wf),
        ),
        patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
    ):
        outcome = await wf._verify_debugfix_loop()

    assert outcome == WorkflowOutcome.DONE
    [followup_call] = rig.followup_prompt.await_args_list
    assert followup_call.args[1].contested == ((_REQUESTED, _HOLDER),)
    assert verify.await_count == 3


def _verify_loop_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verify_results: list):
    rig = _make_workflow(tmp_path)
    rig.scheduler.contested_modules = {_REQUESTED: _HOLDER}
    _write_dispatched_plan(rig.wf, steps=1)
    _stub_loop_collaborators(rig.wf, monkeypatch)
    rig.cfg.rebase_before_verify = False
    rig.cfg.escalate_preexisting_main_break = False
    rig.cfg.max_opaque_timeout_attempts = 3
    rig.cfg.max_failure_signature_repeat = 5
    rig.cfg.max_verify_attempts = 5
    rig.cfg.max_consecutive_zero_output_timeouts = 3
    verify = AsyncMock(side_effect=verify_results)
    monkeypatch.setattr(rig.wf, '_run_scoped_verification_with_infra_retry', verify)
    monkeypatch.setattr(rig.wf, '_maybe_file_chronic_flakes', AsyncMock())
    return rig, verify


@pytest.mark.asyncio
async def test_a_wedged_follow_up_does_not_consume_the_brief(tmp_path, monkeypatch):
    rig, verify = _verify_loop_rig(
        tmp_path, monkeypatch, [_passing_verify(), _passing_verify(), _passing_verify()],
    )
    wf = rig.wf
    assert wf.artifacts is not None
    artifacts = wf.artifacts
    calls = 0

    async def agent(**_kwargs) -> AgentResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            artifacts.write_scope_request(ScopeRequest(_REQUESTED, 'plan omitted it'))
            artifacts.update_step_status('step-1', 'done', None)
        elif calls == 2:
            return _timed_out(transcript_turns=0)
        return _success()

    with (
        patch('orchestrator.workflow.invoke_agent', AsyncMock(side_effect=agent)),
        patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
    ):
        executed = await wf._execute_iterations()
        verified = await wf._verify_debugfix_loop()

    assert executed == verified == WorkflowOutcome.DONE
    assert calls == 3
    assert [c.args[1].contested for c in rig.followup_prompt.await_args_list] == [
        ((_REQUESTED, _HOLDER),),
    ] * 2
    assert verify.await_count == 2


@pytest.mark.asyncio
async def test_the_follow_up_is_a_fresh_session_not_a_dangling_progress_resume(
    tmp_path, monkeypatch
):
    rig, _verify = _verify_loop_rig(tmp_path, monkeypatch, [])
    wf = rig.wf
    assert wf.artifacts is not None
    artifacts = wf.artifacts
    resumed_ids: list[object] = []

    async def agent(**kwargs) -> AgentResult:
        resumed_ids.append(kwargs.get('resume_session_id'))
        if len(resumed_ids) == 1:
            artifacts.write_scope_request(ScopeRequest(_REQUESTED, 'plan omitted it'))
            artifacts.update_step_status('step-1', 'done', None)
            return _timed_out(transcript_turns=5)
        return _success()

    with (
        patch('orchestrator.workflow.invoke_agent', AsyncMock(side_effect=agent)),
        patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
    ):
        outcome = await wf._execute_iterations()

    assert outcome == WorkflowOutcome.DONE
    assert len(resumed_ids) == 2
    assert resumed_ids[1] is None
    [followup_call] = rig.followup_prompt.await_args_list
    assert followup_call.args[1].contested == ((_REQUESTED, _HOLDER),)


@pytest.mark.asyncio
async def test_a_cap_hit_rebuild_of_a_resumed_session_does_not_consume_the_brief(
    tmp_path, monkeypatch
):
    rig = _make_workflow(tmp_path)
    wf, scheduler = rig.wf, rig.scheduler
    scheduler.contested_modules = {_REQUESTED: _HOLDER}
    _write_dispatched_plan(wf, steps=1)
    _stub_loop_collaborators(wf, monkeypatch)
    wf.usage_gate = make_gate_mock(
        account_count=2,
        detect_cap_hit=MagicMock(side_effect=lambda *args, **_kw: 'CAPPED' in args),
        scope_capacity_snapshot=MagicMock(return_value={}),
    )
    assert wf.artifacts is not None
    artifacts = wf.artifacts
    resumed_ids: list[object] = []

    async def agent(**kwargs) -> AgentResult:
        resumed_ids.append(kwargs.get('resume_session_id'))
        call = len(resumed_ids)
        if call == 1:
            artifacts.write_scope_request(ScopeRequest(_REQUESTED, 'plan omitted it'))
            return _timed_out(transcript_turns=5)
        if call == 2:
            return AgentResult(
                success=False, output='CAPPED', cost_usd=0.1, turns=2, duration_ms=1_000,
                session_id='sess-capped',
            )
        if call == 3:
            artifacts.update_step_status('step-1', 'done', None)
        return _success()

    with (
        patch('orchestrator.workflow.invoke_agent', AsyncMock(side_effect=agent)),
        patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
    ):
        outcome = await wf._execute_iterations()

    assert outcome == WorkflowOutcome.DONE
    assert resumed_ids[2] == 'sess-capped'
    [followup_call] = rig.followup_prompt.await_args_list
    assert followup_call.args[1].contested == ((_REQUESTED, _HOLDER),)


@pytest.mark.asyncio
async def test_follow_ups_are_bounded_per_run_and_the_brief_stays_waiting(tmp_path, monkeypatch):
    rig, verify = _verify_loop_rig(tmp_path, monkeypatch, [_passing_verify()] * 5)
    wf = rig.wf
    assert wf.artifacts is not None
    artifacts = wf.artifacts
    calls = 0

    async def agent(**_kwargs) -> AgentResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            artifacts.write_scope_request(ScopeRequest(_REQUESTED, 'plan omitted it'))
            artifacts.update_step_status('step-1', 'done', None)
            return _success()
        return _timed_out(transcript_turns=0)

    with (
        patch('orchestrator.workflow.invoke_agent', AsyncMock(side_effect=agent)),
        patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
    ):
        executed = await wf._execute_iterations()
        verified = await wf._verify_debugfix_loop()

    assert executed == verified == WorkflowOutcome.DONE
    assert len(rig.followup_prompt.await_args_list) == MAX_SCOPE_DRIFT_FOLLOWUPS
    assert verify.await_count == MAX_SCOPE_DRIFT_FOLLOWUPS
    waiting = artifacts.read_scope_drift_brief()
    assert waiting is not None
    assert waiting.contested == ((_REQUESTED, _HOLDER),)

