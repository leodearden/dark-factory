"""PRD boundary row 15 and the whole-plan parity tripwire, executed against a
real TaskWorkflow in eval mode (task 5385, plans/live-shadow-eval-prd.md
decision 14, INV-10).

The cell config is built exactly as ``evals/runner.py::run_eval`` builds it:
a candidate resolved by name, ``build_eval_orch_config``, a REAL
``BriefingAssembler`` and ``build_workflow(..., initial_plan=...)``. Only the
agent subprocess is faked, at ``orchestrator.workflow.invoke_agent`` (the
``test_routing_integration_gate.py`` seam), so the real ``_invoke`` resolves
and sends every route. Memory recall's transport refuses every call, as the
eval profile's ``127.0.0.1:1`` null sentinel does, so the brief renders
without memory and no test reaches a live fused-memory server.

Fixtures are module-local: a conftest.py or _workflow_helpers.py edit widens
the merge-time scoped-test selection to the whole package.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _eval_config_lookup import named_eval_config
from _implementer_action_oracle import ACTION_MARKER, WHOLE_PLAN_ACTION_PIN
from _merge_lane_fakes import lane_scene_config
from _workflow_helpers import (
    AgentStub,
    FakeMcp,
    FakeScheduler,
    _derive_meta_root_like_production,
    _init_repo,
    agent_side_artifacts,
)

from orchestrator.agents import memory_recall
from orchestrator.agents.briefing import IMPLEMENTER_ACTIONS, BriefingAssembler
from orchestrator.agents.invoke import AgentResult
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.evals.runner import build_eval_orch_config
from orchestrator.git_ops import GitOps, _run
from orchestrator.implementer_brief_variant import ImplementerBriefVariant
from orchestrator.merge_lane import MergeLane
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, build_workflow

pytestmark = pytest.mark.usefixtures(
    'code_default_config', _derive_meta_root_like_production.__name__,
)

TASK_ID = 'df_nu'
TASK_MAX_EXECUTE_ITERATIONS = 4
ONE_STEP_ACTION = IMPLEMENTER_ACTIONS[ImplementerBriefVariant.ONE_STEP]
WHOLE_PLAN_ACTION = IMPLEMENTER_ACTIONS[ImplementerBriefVariant.WHOLE_PLAN]


def _item(item_id: str) -> dict:
    return {
        'id': item_id,
        'type': 'impl',
        'description': f'write {item_id}.txt',
        'status': 'pending',
        'commit': None,
    }


def _plan(*, prerequisites: int = 0, steps: int = 3) -> dict:
    return {
        'task_id': TASK_ID,
        'title': 'Write one file per step',
        'files': [f'step-{n}.txt' for n in range(1, steps + 1)],
        'analysis': 'Each step writes and commits one file.',
        'prerequisites': [_item(f'pre-{n}') for n in range(1, prerequisites + 1)],
        'steps': [_item(f'step-{n}') for n in range(1, steps + 1)],
        'design_decisions': [],
        'reuse': [],
        '_finalized_at': '2026-01-01T00:00:00+00:00',
    }


class _ObedientImplementer(AgentStub):
    """Records every implementer call at the seam and obeys its brief.

    Under the one-step action it completes only the first pending item,
    otherwise every pending item. ``stalled`` completes nothing, and the first
    ``progress_timeouts`` calls are ceiling-kills of a productive session.
    Every other role is delegated to ``AgentStub``.
    """

    def __init__(self, *, stalled: bool = False, progress_timeouts: int = 0) -> None:
        super().__init__()
        self.stalled = stalled
        self.progress_timeouts_left = progress_timeouts
        self.implementer_calls: list[dict] = []

    async def invoke_agent(self, **kwargs) -> AgentResult:
        if self._detect_role(kwargs['system_prompt']) != 'implementer':
            return await super().invoke_agent(**kwargs)
        self.implementer_calls.append({
            'prompt': kwargs['prompt'],
            'max_turns': kwargs['max_turns'],
            'max_budget_usd': kwargs['max_budget_usd'],
        })
        if self.progress_timeouts_left:
            self.progress_timeouts_left -= 1
            return AgentResult(
                success=False, output='', timed_out=True, turns=0,
                duration_ms=1_200_000, transcript_turns=5,
            )
        if self.stalled:
            return AgentResult(success=True, output='No step completed', cost_usd=0.1)
        return await self._complete(kwargs['cwd'], one_step=ONE_STEP_ACTION in kwargs['prompt'])

    async def _complete(self, cwd: Path, *, one_step: bool) -> AgentResult:
        artifacts = agent_side_artifacts(cwd)
        pending = artifacts.get_pending_steps()
        for item in pending[:1] if one_step else pending:
            (cwd / f'{item["id"]}.txt').write_text(f'{item["id"]}\n')
            await _run(['git', 'add', '-A'], cwd=cwd)
            await _run(['git', 'commit', '-m', f'Complete {item["id"]}'], cwd=cwd)
            _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=cwd)
            artifacts.update_step_status(item['id'], 'done', commit=sha.strip())
        return AgentResult(success=True, output='Completed', cost_usd=1.0)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo(repo)
    return repo


def _cell_config(repo: Path, candidate_name: str) -> OrchestratorConfig:
    base = lane_scene_config(
        repo,
        GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
        max_concurrent_tasks=1,
    )
    task = {
        'id': TASK_ID,
        'project_root': str(repo),
        'max_execute_iterations': TASK_MAX_EXECUTE_ITERATIONS,
    }
    return build_eval_orch_config(named_eval_config(candidate_name), task, base_config=base)


def _workflow(orch: OrchestratorConfig, plan: dict) -> TaskWorkflow:
    git_ops = GitOps(orch.git, orch.project_root)
    merge_queue: asyncio.Queue = asyncio.Queue()
    asyncio.create_task(MergeLane(git_ops, merge_queue).run(), name='test-merge-worker')
    return build_workflow(
        assignment=TaskAssignment(
            task_id=TASK_ID,
            task={
                'id': TASK_ID,
                'title': plan['title'],
                'description': 'One-step variant boundary task',
                'status': 'pending',
                'metadata': {'files': plan['files']},
                'dependencies': [],
            },
            modules=['lib'],
        ),
        config=orch,
        git_ops=git_ops,
        scheduler=FakeScheduler(),
        briefing=BriefingAssembler(orch),
        mcp=FakeMcp(),
        initial_plan=plan,
        merge_queue=merge_queue,
    )


async def _run_cell(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_name: str,
    stub: _ObedientImplementer,
    plan: dict,
) -> tuple[OrchestratorConfig, TaskWorkflow, WorkflowOutcome]:
    orch = _cell_config(repo, candidate_name)
    workflow = _workflow(orch, plan)
    monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
    monkeypatch.setattr(
        memory_recall, 'mcp_call',
        AsyncMock(side_effect=ConnectionRefusedError('null sentinel')),
    )
    monkeypatch.setattr(
        'orchestrator.workflow.run_scoped_verification',
        AsyncMock(return_value=VerifyResult(
            passed=True, test_output='OK', lint_output='', type_output='',
            summary='All checks passed',
        )),
    )
    outcome = (await workflow.run()).outcome
    return orch, workflow, outcome


def _implementer_log(workflow: TaskWorkflow) -> list[dict]:
    assert workflow.artifacts is not None
    entries, corrupted = workflow.artifacts.read_iteration_log()
    assert corrupted == []
    return [entry for entry in entries if entry.get('agent') == 'implementer']


def _action_of(prompt: str) -> str:
    assert prompt.count(ACTION_MARKER) == 1, prompt
    return prompt.split(ACTION_MARKER)[1]


@pytest.mark.asyncio
class TestOneStepRow15:
    async def test_a_clean_run_iterates_once_per_step_at_a_third_of_the_bounds(
        self, repo, monkeypatch,
    ):
        stub = _ObedientImplementer()
        orch, workflow, outcome = await _run_cell(
            repo, monkeypatch, 'claude-sonnet-max@one-step', stub, _plan(steps=3),
        )

        assert orch.implementer_brief_variant is ImplementerBriefVariant.ONE_STEP
        assert outcome == WorkflowOutcome.DONE
        assert len(stub.implementer_calls) == 3
        assert workflow.metrics.execute_iterations == 3
        for call in stub.implementer_calls:
            assert ONE_STEP_ACTION in call['prompt']
            assert WHOLE_PLAN_ACTION not in call['prompt']
            assert call['max_turns'] == math.ceil(orch.max_turns.implementer / 3)
            assert call['max_budget_usd'] == orch.budgets.implementer / 3
        log = _implementer_log(workflow)
        assert [entry['steps_completed'] for entry in log] == [
            ['step-1'], ['step-2'], ['step-3'],
        ]

    async def test_the_cap_counts_prerequisites_and_ignores_the_task_bound(
        self, repo, monkeypatch,
    ):
        stub = _ObedientImplementer(stalled=True)
        _orch, _workflow, outcome = await _run_cell(
            repo, monkeypatch, 'claude-sonnet-max@one-step', stub,
            _plan(prerequisites=1, steps=3),
        )

        assert len(stub.implementer_calls) == 1 + 3 + 3
        assert outcome == WorkflowOutcome.BLOCKED

    async def test_a_progress_resume_is_excluded_from_the_scaled_cap(
        self, repo, monkeypatch,
    ):
        stub = _ObedientImplementer(stalled=True, progress_timeouts=1)
        _orch, workflow, outcome = await _run_cell(
            repo, monkeypatch, 'claude-sonnet-max@one-step', stub, _plan(steps=3),
        )

        assert workflow.metrics.progress_resume_total == 1
        assert len(stub.implementer_calls) == (3 + 3) + 1
        assert outcome == WorkflowOutcome.BLOCKED


@pytest.mark.asyncio
class TestWholePlanParityTripwire:
    async def test_a_clean_run_is_one_whole_plan_iteration_with_the_pinned_action(self, repo, monkeypatch):
        stub = _ObedientImplementer()
        orch, workflow, outcome = await _run_cell(
            repo, monkeypatch, 'claude-sonnet-max', stub, _plan(steps=3),
        )

        assert orch.implementer_brief_variant is ImplementerBriefVariant.WHOLE_PLAN
        assert outcome == WorkflowOutcome.DONE
        assert len(stub.implementer_calls) == 1
        [call] = stub.implementer_calls
        assert _action_of(call['prompt']) == WHOLE_PLAN_ACTION_PIN
        assert call['max_turns'] == orch.max_turns.implementer
        assert call['max_budget_usd'] == orch.budgets.implementer
        log = _implementer_log(workflow)
        assert [entry['steps_completed'] for entry in log] == [['step-1', 'step-2', 'step-3']]

    async def test_a_stalled_run_stops_at_the_tasks_own_bound(self, repo, monkeypatch):
        stub = _ObedientImplementer(stalled=True)
        orch, _workflow, outcome = await _run_cell(
            repo, monkeypatch, 'claude-sonnet-max', stub, _plan(steps=3),
        )

        assert orch.max_execute_iterations == TASK_MAX_EXECUTE_ITERATIONS
        assert len(stub.implementer_calls) == TASK_MAX_EXECUTE_ITERATIONS
        assert outcome == WorkflowOutcome.BLOCKED
