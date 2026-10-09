"""TaskWorkflow hands every agent a prompt BUILDER, re-gathered before each retry (task 5730).

``TaskWorkflow._invoke`` awaits its builder for the first dispatch and passes
the same object down as ``invoke_with_cap_retry``'s ``rebuild_prompt``, so a
retry after a cap wait is briefed from the plan as it is THEN, not as it was
when the first attempt started.  The execute-loop implementer builder declines
(``None``) once no step is pending, which cancels a retry of finished work.

The end-to-end tests drive the REAL ``_invoke`` and the REAL
``invoke_with_cap_retry`` against a real ``plan.json``; only the agent
subprocess (``invoke_agent``) and the usage gate are doubles.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from _recording_event_store import _RecordingEventStore
from shared.testing import make_gate_mock

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import IMPLEMENTER
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

_TASK_ID = '5730'
_WIP_SHA = 'abc123def4567890abc123def4567890abc12345'


def _make_workflow(tmp_path: Path, *, event_store: _RecordingEventStore | None = None) -> TaskWorkflow:
    assignment = MagicMock()
    assignment.task_id = _TASK_ID
    assignment.task = {'id': _TASK_ID, 'title': 'T', 'description': 'd'}
    assignment.modules = []

    cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    cfg.fused_memory.project_id = 'dark_factory'
    cfg.fused_memory.url = 'http://localhost:8002'
    cfg.max_review_cycles = 2
    cfg.max_amendment_rounds = 1
    cfg.lock_depth = 2
    cfg.steward_completion_timeout = 300.0
    cfg.project_root = tmp_path / 'proj'
    cfg.max_execute_iterations = 10
    cfg.max_consecutive_zero_output_timeouts = 2
    cfg.max_progress_resume_iterations = 20
    cfg.recycle_config_dir_on_zero_output = False
    cfg.judge_after_each_iteration = False
    cfg.inter_iteration_rebase = False
    cfg.timeouts.working_idle_secs = 1800.0
    cfg.invocation_timeout = 7200.0
    cfg.budgets.implementer = 5.0
    cfg.max_turns.implementer = 50
    cfg.backends.implementer = 'claude'
    cfg.sandbox.enabled = False

    git_ops = MagicMock()
    git_ops.has_uncommitted_work = AsyncMock(return_value=False)
    git_ops.get_commit_subjects = AsyncMock(return_value=[])

    briefing = MagicMock()

    async def render_implementer(plan, *, rebase_notice=None, task_id=None, wip_notice=None):
        return f"analysis={plan['analysis']} wip={[c['sha'] for c in wip_notice or []]}"

    briefing.build_implementer_prompt = AsyncMock(side_effect=render_implementer)

    wf = TaskWorkflow(
        assignment=assignment,
        config=cfg,
        git_ops=git_ops,
        scheduler=MagicMock(),
        briefing=briefing,
        mcp=MagicMock(),
        event_store=event_store,  # type: ignore[arg-type]
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    wf.worktree = worktree
    wf.artifacts = TaskArtifacts(worktree)
    wf.merge_queue = MagicMock()
    wf._module_configs = []
    return wf


def _write_two_step_plan(wf: TaskWorkflow, *, base_commit: str | None = None) -> None:
    assert wf.artifacts is not None
    wf.artifacts.init(_TASK_ID, 'T', 'd', base_commit=base_commit)
    plan = {
        'task_id': _TASK_ID,
        'title': 'T',
        'analysis': 'ORIGINAL-ANALYSIS',
        'files': [],
        'prerequisites': [],
        'steps': [
            {'id': 'step-1', 'type': 'impl', 'description': 'one', 'status': 'pending'},
            {'id': 'step-2', 'type': 'impl', 'description': 'two', 'status': 'pending'},
        ],
    }
    wf.artifacts.write_plan(plan)
    wf.plan = wf.artifacts.read_plan()


def _stub_loop_collaborators(wf: TaskWorkflow) -> None:
    """Stub what lies outside the prompt seam; plan reads and _invoke stay real."""
    assert wf.artifacts is not None
    wf.artifacts.validate_plan_owner = MagicMock(return_value=True)  # type: ignore[method-assign]
    wf.artifacts.stamp_plan_provenance = MagicMock()  # type: ignore[method-assign]
    wf._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    wf._get_head_commit = AsyncMock(return_value='head0000')  # type: ignore[method-assign]
    wf._build_agent_env = MagicMock(return_value=None)  # type: ignore[method-assign]
    wf._reconcile_done_step_commits = AsyncMock()  # type: ignore[method-assign]
    wf.usage_gate = make_gate_mock(
        account_count=2,
        detect_cap_hit=MagicMock(side_effect=[True, False]),
        scope_capacity_snapshot=MagicMock(return_value={}),
    )


def _mark_done(wf: TaskWorkflow, *step_ids: str, commit: str | None = None) -> None:
    assert wf.artifacts is not None
    for step_id in step_ids:
        wf.artifacts.update_step_status(step_id, 'done', commit)


def _cap_hit_fresh() -> AgentResult:
    return AgentResult(
        success=False, output="You've hit your usage limit", session_id='',
        cost_usd=0.3, turns=1, duration_ms=8_000,
    )


def _success() -> AgentResult:
    return AgentResult(success=True, output='done', cost_usd=0.5, turns=3, duration_ms=5_000)


async def _run_execute_loop(wf: TaskWorkflow, agent) -> tuple[WorkflowOutcome, AsyncMock]:
    with (
        patch('orchestrator.workflow.invoke_agent', new_callable=AsyncMock, side_effect=agent) as inv,
        patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock),
    ):
        outcome = await wf._execute_iterations()
    return outcome, inv


@pytest.mark.asyncio
class TestRetryIsBriefedFromTheLivePlan:
    async def test_fresh_retry_renders_the_plan_as_it_is_at_retry_time(self, tmp_path):
        wf = _make_workflow(tmp_path)
        _write_two_step_plan(wf)
        _stub_loop_collaborators(wf)
        artifacts = wf.artifacts
        assert artifacts is not None
        prompts: list[str] = []

        async def agent(**kwargs):
            prompts.append(kwargs['prompt'])
            if len(prompts) == 1:
                plan = artifacts.read_plan()
                plan['analysis'] = 'MUTATED-ANALYSIS'
                artifacts.write_plan(plan)
                return _cap_hit_fresh()
            _mark_done(wf, 'step-1', 'step-2')
            return _success()

        outcome, _inv = await _run_execute_loop(wf, agent)

        assert outcome == WorkflowOutcome.DONE
        assert len(prompts) == 2
        assert 'ORIGINAL-ANALYSIS' in prompts[0]
        assert 'MUTATED-ANALYSIS' in prompts[1]
        assert 'ORIGINAL-ANALYSIS' not in prompts[1]

    async def test_wip_notice_is_deduped_against_the_on_disk_plan(self, tmp_path):
        wf = _make_workflow(tmp_path)
        _write_two_step_plan(wf, base_commit='base0000')
        _stub_loop_collaborators(wf)
        wf.git_ops.get_commit_subjects = AsyncMock(  # type: ignore[method-assign]
            return_value=[(_WIP_SHA, 'chore: save WIP before rebase')],
        )
        prompts: list[str] = []

        async def agent(**kwargs):
            prompts.append(kwargs['prompt'])
            if len(prompts) == 1:
                _mark_done(wf, 'step-1', commit=_WIP_SHA[:12])
                return _cap_hit_fresh()
            _mark_done(wf, 'step-2')
            return _success()

        await _run_execute_loop(wf, agent)

        assert len(prompts) == 2
        assert _WIP_SHA in prompts[0]
        assert _WIP_SHA not in prompts[1]


@pytest.mark.asyncio
class TestRetryOfFinishedWorkIsCancelled:
    async def test_no_pending_steps_cancels_the_retry(self, tmp_path):
        rec = _RecordingEventStore()
        wf = _make_workflow(tmp_path, event_store=rec)
        _write_two_step_plan(wf)
        _stub_loop_collaborators(wf)

        async def agent(**_kwargs):
            _mark_done(wf, 'step-1', 'step-2')
            return _cap_hit_fresh()

        outcome, inv = await _run_execute_loop(wf, agent)

        assert inv.await_count == 1
        assert outcome == WorkflowOutcome.DONE
        assert wf.artifacts is not None
        entries, _corrupted = wf.artifacts.read_iteration_log()
        implementer = [e for e in entries if e.get('agent') == 'implementer']
        assert implementer[-1]['steps_completed'] == ['step-1', 'step-2']
        ends = [entry for (etype, entry) in rec.events if etype == EventType.invocation_end]
        assert ends[-1]['data']['retry_aborted'] is True


@pytest.mark.asyncio
class TestInvokeTakesABuilder:
    async def test_builder_supplies_the_prompt_and_is_handed_down_for_retries(self, tmp_path):
        wf = _make_workflow(tmp_path)
        builder = AsyncMock(return_value='BUILT')
        with (
            patch(
                'orchestrator.workflow.invoke_with_cap_retry',
                new=AsyncMock(return_value=_success()),
            ) as iwcr,
            patch.object(wf, '_build_agent_env', return_value=None),
        ):
            await wf._invoke(IMPLEMENTER, builder, tmp_path)

        builder.assert_awaited_once_with()
        kwargs = iwcr.call_args.kwargs
        assert kwargs['prompt'] == 'BUILT'
        assert kwargs['rebuild_prompt'] is builder

    async def test_builder_declining_the_first_dispatch_is_a_loud_error(self, tmp_path):
        wf = _make_workflow(tmp_path)
        wf.artifacts = MagicMock()
        builder = AsyncMock(return_value=None)
        with (
            patch(
                'orchestrator.workflow.invoke_with_cap_retry', new=AsyncMock(),
            ) as iwcr,
            patch.object(wf, '_build_agent_env', return_value=None),
            pytest.raises(ValueError, match=IMPLEMENTER.name),
        ):
            await wf._invoke(IMPLEMENTER, builder, tmp_path)

        iwcr.assert_not_awaited()
        wf.artifacts.write_agent_session.assert_not_called()


@pytest.mark.asyncio
class TestCallSiteBuildersReGatherThePlan:
    async def test_amender_builder_rereads_the_plan_on_every_build(self, tmp_path):
        wf = _make_workflow(tmp_path)
        _write_two_step_plan(wf)
        wf._get_head_commit = AsyncMock(return_value='head0000')  # type: ignore[method-assign]
        wf.artifacts.validate_plan_owner = MagicMock(return_value=True)  # type: ignore[method-assign]
        wf.briefing.build_amender_prompt = AsyncMock(return_value='amend please')

        artifacts = wf.artifacts
        assert artifacts is not None

        async def fake_invoke(role, build_prompt, cwd, output_schema=None):
            await build_prompt()
            plan = artifacts.read_plan()
            plan['analysis'] = 'REWRITTEN-ANALYSIS'
            artifacts.write_plan(plan)
            await build_prompt()
            return _success()

        wf._invoke = fake_invoke  # type: ignore[method-assign]

        assert await wf._amend([{'id': 's-1', 'text': 'tidy'}], amendment_round=1)

        calls = wf.briefing.build_amender_prompt.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs['plan']['analysis'] == 'ORIGINAL-ANALYSIS'
        assert calls[1].kwargs['plan']['analysis'] == 'REWRITTEN-ANALYSIS'
