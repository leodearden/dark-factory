"""Deterministic e2e tests for the workflow state machine.

Uses real git operations and file I/O. Agent invocations and verification
are stubbed with deterministic side-effect functions that write actual files
to the worktree, simulating what real agents would do.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, WorkflowState
from tests.conftest import _spy_set_cached_status

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A bare-minimum git repo with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path):
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    # Seed with a simple Python file so the repo isn't empty
    (repo / 'lib.py').write_text('def greet(name: str) -> str:\n    return f"Hello, {name}"\n')
    (repo / 'test_lib.py').write_text(
        'from lib import greet\n\ndef test_greet():\n    assert greet("world") == "Hello, world"\n'
    )
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        max_execute_iterations=5,
        max_verify_attempts=3,
        max_review_cycles=2,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'Add farewell function',
            'description': 'Add a farewell(name) function to lib.py with tests',
            'status': 'pending',
            'metadata': {'modules': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------

PLAN = {
    'task_id': '42',
    'title': 'Add farewell function',
    'modules': ['lib'],
    'files': ['lib.py', 'test_lib.py'],
    'analysis': 'Simple function addition with TDD',
    'prerequisites': [],
    'steps': [
        {
            'id': 'step-1',
            'type': 'test',
            'description': 'Write failing test for farewell()',
            'status': 'pending',
            'commit': None,
        },
        {
            'id': 'step-2',
            'type': 'impl',
            'description': 'Implement farewell() to pass test',
            'status': 'pending',
            'commit': None,
        },
    ],
    'design_decisions': [
        {'decision': 'Mirror greet() signature', 'rationale': 'Consistency'},
    ],
    'reuse': [],
}


def _make_review(reviewer: str, verdict: str = 'PASS', issues: list | None = None):
    return {
        'reviewer': reviewer,
        'verdict': verdict,
        'issues': issues or [],
        'summary': f'{reviewer} review: {verdict}',
    }


class AgentStub:
    """Deterministic agent stub that performs real file operations.

    Tracks which roles have been invoked and their order.
    """

    def __init__(
        self,
        verify_results: list[VerifyResult] | None = None,
        judge_verdicts: list | None = None,
    ):
        self.calls: list[str] = []
        # Last env_overrides seen per role — set by invoke_agent before dispatch.
        self.env_overrides_by_role: dict[str, dict[str, str] | None] = {}
        self._impl_iteration = 0
        # Sequence of verify results to return (pops from front)
        self._verify_results = list(verify_results or [VerifyResult(
            passed=True, test_output='', lint_output='', type_output='',
            summary='All checks passed',
        )])
        # Sequence of judge verdicts to return (pops from front; last persists).
        # Each entry may be a dict (structured verdict), an Exception to raise,
        # None (to simulate structured_output=None), or the string 'SUCCESS_FALSE'
        # to simulate result.success=False.
        self._judge_verdicts = list(judge_verdicts or [])

    async def invoke_agent(
        self,
        prompt: str,
        system_prompt: str,
        cwd: Path,
        model: str = 'opus',
        max_turns: int = 50,
        max_budget_usd: float = 5.0,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        mcp_config: dict | None = None,
        output_schema: dict | None = None,
        permission_mode: str = 'bypassPermissions',
        sandbox_modules: list[str] | None = None,
        effort: str | None = None,
        backend: str = 'claude',
        oauth_token: str | None = None,
        resume_session_id: str | None = None,
        timeout_seconds: float | None = None,
        config_dir: Path | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> AgentResult:
        """Determine role from system_prompt content, perform side effects."""
        role = self._detect_role(system_prompt)
        self.calls.append(role)
        self.env_overrides_by_role[role] = env_overrides

        if role == 'architect':
            return await self._architect(cwd)
        elif role == 'implementer':
            return await self._implementer(cwd)
        elif role == 'debugger':
            return await self._debugger(cwd)
        elif role.startswith('reviewer'):
            return self._reviewer(role, output_schema)
        elif role == 'merger':
            return self._merger()
        elif role == 'judge':
            return self._judge()
        else:
            return AgentResult(success=True, output='OK')

    def _detect_role(self, system_prompt: str) -> str:
        # Judge check goes first — its system prompt does not contain
        # architect/implementer/reviewer substrings so ordering is safe.
        if 'completion judge' in system_prompt.lower():
            return 'judge'
        if 'TDD architect' in system_prompt:
            return 'architect'
        if 'TDD implementer' in system_prompt:
            return 'implementer'
        if 'debugger' in system_prompt.lower() and 'You are a debugger' in system_prompt:
            return 'debugger'
        if 'code reviewer' in system_prompt.lower():
            return 'reviewer_comprehensive'
        if 'merge conflict resolver' in system_prompt.lower():
            return 'merger'
        # If re-planning (architect called with review feedback)
        if 'blocking issues were found' in system_prompt or 'Update the plan' in system_prompt:
            return 'architect'
        return 'unknown'

    async def _architect(self, cwd: Path) -> AgentResult:
        """Write plan.json to .task/ directory."""
        task_dir = cwd / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        plan = dict(PLAN)
        plan['_schema_version'] = 1
        (task_dir / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        return AgentResult(success=True, output='Plan created', cost_usd=0.50)

    async def _implementer(self, cwd: Path) -> AgentResult:
        """Create code files and update plan.json step statuses."""
        self._impl_iteration += 1
        artifacts = TaskArtifacts(cwd)
        pending = artifacts.get_pending_steps()

        if not pending:
            return AgentResult(success=True, output='Nothing to do')

        completed_ids = []
        for step in pending:
            step_id = step['id']
            if step['type'] == 'test':
                # Write failing test
                (cwd / 'test_lib.py').write_text(
                    'from lib import greet, farewell\n\n'
                    'def test_greet():\n'
                    '    assert greet("world") == "Hello, world"\n\n'
                    'def test_farewell():\n'
                    '    assert farewell("world") == "Goodbye, world"\n'
                )
            elif step['type'] == 'impl':
                # Implement
                (cwd / 'lib.py').write_text(
                    'def greet(name: str) -> str:\n'
                    '    return f"Hello, {name}"\n\n'
                    'def farewell(name: str) -> str:\n'
                    '    return f"Goodbye, {name}"\n'
                )

            # Git commit
            await _run(['git', 'add', '-A'], cwd=cwd)
            await _run(['git', 'commit', '-m', f'Complete {step_id}'], cwd=cwd)
            _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=cwd)

            artifacts.update_step_status(step_id, 'done', commit=sha)
            completed_ids.append(step_id)

        return AgentResult(
            success=True, output=f'Completed {completed_ids}', cost_usd=1.00
        )

    async def _debugger(self, cwd: Path) -> AgentResult:
        """Fix whatever is broken. For our stub, ensure files are correct."""
        (cwd / 'lib.py').write_text(
            'def greet(name: str) -> str:\n'
            '    return f"Hello, {name}"\n\n'
            'def farewell(name: str) -> str:\n'
            '    return f"Goodbye, {name}"\n'
        )
        await _run(['git', 'add', '-A'], cwd=cwd)
        await _run(['git', 'commit', '-m', 'fix: correct implementation'], cwd=cwd)
        return AgentResult(success=True, output='Fixed', cost_usd=0.30)

    def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
        """Return PASS review."""
        review = _make_review(role)
        return AgentResult(
            success=True,
            output=json.dumps(review),
            structured_output=review,
            cost_usd=0.10,
        )

    def _merger(self) -> AgentResult:
        return AgentResult(success=True, output='Merged', cost_usd=0.20)

    def _judge(self) -> AgentResult:
        """Return the next queued judge verdict.

        If the entry is an Exception, raise it (simulates invoke failure).
        If the entry is 'SUCCESS_FALSE', return success=False.
        If the entry is None, return success=True with structured_output=None.
        Otherwise treat the entry as a verdict dict.
        """
        if not self._judge_verdicts:
            # No verdicts queued — default: incomplete
            verdict = {
                'complete': False,
                'reasoning': 'default stub verdict',
                'uncovered_plan_steps': [],
                'substantive_work': False,
            }
        elif len(self._judge_verdicts) > 1:
            verdict = self._judge_verdicts.pop(0)
        else:
            verdict = self._judge_verdicts[0]

        if isinstance(verdict, Exception):
            raise verdict
        if verdict == 'SUCCESS_FALSE':
            return AgentResult(success=False, output='', cost_usd=0.05)
        if verdict is None:
            return AgentResult(success=True, output='', structured_output=None, cost_usd=0.05)
        return AgentResult(
            success=True,
            output=json.dumps(verdict),
            structured_output=verdict,
            cost_usd=0.05,
        )

    def next_verify_result(self) -> VerifyResult:
        """Pop the next verify result, or return the last one forever."""
        if len(self._verify_results) > 1:
            return self._verify_results.pop(0)
        return self._verify_results[0]


class FakeMcp:
    """Minimal McpLifecycle stand-in."""

    @property
    def url(self) -> str:
        return 'http://localhost:9999'

    def mcp_config_json(self, escalation_url: str | None = None) -> dict:
        return {'mcpServers': {}}


class FakeScheduler:
    """Scheduler that tracks status changes without HTTP calls."""

    def __init__(self):
        self.statuses: dict[str, list[str]] = {}

    async def set_task_status(self, task_id: str, status: str) -> None:
        self.statuses.setdefault(task_id, []).append(status)

    async def handle_blast_radius_expansion(
        self, task_id: str, current: list[str], needed: list[str]
    ) -> bool:
        return True

    def get_cached_status(self, task_id: str) -> str | None:
        history = self.statuses.get(task_id)
        return history[-1] if history else None

    def release(self, task_id: str) -> None:
        pass


class FakeBriefing:
    """BriefingAssembler that returns canned prompts."""

    async def build_architect_prompt(self, task: dict, worktree=None, context: str | None = None) -> str:
        return f'Plan task: {task.get("title", "")}'

    async def build_implementer_prompt(
        self, plan: dict, iteration_log: list, context: str | None = None,
        rebase_notice: dict | None = None, task_id: str | None = None,
    ) -> str:
        return 'Implement the plan'

    async def build_debugger_prompt(
        self, failures: str, plan: dict, context: str | None = None,
        task_id: str | None = None,
    ) -> str:
        return f'Fix: {failures[:100]}'

    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = None
    ) -> str:
        return f'Review ({reviewer_type}): {diff[:100]}'

    async def build_completion_judge_prompt(
        self,
        plan: dict,
        iteration_log: list,
        diff: str,
        task_id: str | None = None,
        context: str | None = None,
    ) -> str:
        return f'Judge task {task_id}: plan has {len(plan.get("steps", []))} steps'

    async def build_merger_prompt(
        self, conflicts: str, task_intent: str, context: str | None = None
    ) -> str:
        return f'Merge: {conflicts[:100]}'

    async def build_resume_prompt(
        self,
        task: dict,
        plan: dict,
        escalation_summary: str,
        resolution: str,
        worktree=None,
    ) -> str:
        return f'Resume: {resolution[:100]}'


def _build_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    agent_stub: AgentStub,
) -> tuple[TaskWorkflow, FakeScheduler]:
    """Wire up a TaskWorkflow with all fakes injected."""
    from orchestrator.merge_queue import MergeWorker

    scheduler = FakeScheduler()
    merge_queue: asyncio.Queue = asyncio.Queue()
    worker = MergeWorker(git_ops, merge_queue)
    # Start merge worker — cleaned up when event loop tears down after test
    asyncio.create_task(worker.run(), name='test-merge-worker')
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
        merge_queue=merge_queue,
    )
    return workflow, scheduler


# ---------------------------------------------------------------------------
# Tests: Happy Path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHappyPath:
    """Full PLAN → EXECUTE → VERIFY → REVIEW → MERGE → DONE."""

    async def test_single_task_completes(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        # Patch invoke_agent and run_verification
        monkeypatch.setattr(
            'orchestrator.agents.invoke.invoke_agent', stub.invoke_agent
        )
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE
        assert workflow.state == WorkflowState.DONE

    async def test_status_transitions(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        await workflow.run()

        # Scheduler saw: in-progress → done
        assert scheduler.statuses['42'] == ['in-progress', 'done']

    async def test_agent_invocation_order(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        await workflow.run()

        # Expected: architect, implementer, 1 comprehensive reviewer
        assert stub.calls[0] == 'architect'
        assert stub.calls[1] == 'implementer'
        assert stub.calls[2] == 'reviewer_comprehensive'

    async def test_code_appears_on_main_after_merge(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()
        assert outcome == WorkflowOutcome.DONE

        # Verify the farewell function is in main's git tree
        # (update-ref doesn't update the working tree, so check via git show)
        _, lib_content, _ = await _run(
            ['git', 'show', 'main:lib.py'], cwd=config.project_root,
        )
        assert 'def farewell' in lib_content
        assert 'Goodbye' in lib_content

    async def test_worktree_cleaned_up_on_success(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        await workflow.run()

        # Worktree should be cleaned up
        worktree_dir = git_ops.worktree_base / '42'
        assert not worktree_dir.exists()

    async def test_metrics_tracked(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        await workflow.run()

        # 1 architect + 1 implementer + 1 reviewer = 3
        assert workflow.metrics.agent_invocations == 3
        assert workflow.metrics.total_cost_usd > 0
        assert workflow.metrics.execute_iterations == 1


# ---------------------------------------------------------------------------
# Tests: Completion Judge (ζ)
# ---------------------------------------------------------------------------


def _config_with_judge(config: OrchestratorConfig, enabled: bool) -> OrchestratorConfig:
    """Return a new config identical to *config* with judge_after_each_iteration toggled."""
    return OrchestratorConfig(
        project_root=config.project_root,
        max_concurrent_tasks=config.max_concurrent_tasks,
        max_execute_iterations=config.max_execute_iterations,
        max_verify_attempts=config.max_verify_attempts,
        max_review_cycles=config.max_review_cycles,
        judge_after_each_iteration=enabled,
        git=config.git,
    )


@pytest.mark.asyncio
class TestCompletionJudge:
    """Judge-LLM early-exit hook (ζ). Exercises _execute_iterations + _run_completion_judge."""

    async def test_execute_iterations_exits_early_when_judge_says_complete(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        judge_cfg = _config_with_judge(config, enabled=True)
        stub = AgentStub(judge_verdicts=[{
            'complete': True,
            'reasoning': 'diff implements both plan steps end-to-end.',
            'uncovered_plan_steps': [],
            'substantive_work': True,
        }])
        workflow, _ = _build_workflow(judge_cfg, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE
        assert workflow.metrics.execute_iterations == 1
        assert workflow.metrics.judge_invocations == 1
        assert workflow.metrics.judge_early_exits == 1
        assert 'judge' in stub.calls

    async def test_execute_iterations_rejects_judge_complete_when_substantive_work_false(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """complete=True + substantive_work=False must NOT trigger an early exit."""
        judge_cfg = _config_with_judge(config, enabled=True)
        # Queue: first verdict is the dangerous one; fallback to incomplete so loop continues.
        stub = AgentStub(judge_verdicts=[
            {
                'complete': True,
                'reasoning': 'bogus — diff is empty',
                'uncovered_plan_steps': ['step-1', 'step-2'],
                'substantive_work': False,
            },
            {
                'complete': False,
                'reasoning': 'still nothing',
                'uncovered_plan_steps': ['step-1', 'step-2'],
                'substantive_work': False,
            },
        ])
        workflow, _ = _build_workflow(judge_cfg, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        # The implementer still completes all plan steps in iteration 1, so the
        # loop exits via the normal `while pending_steps` gate — NOT via judge
        # early-exit. judge_early_exits must remain 0.
        assert outcome == WorkflowOutcome.DONE
        assert workflow.metrics.judge_early_exits == 0
        assert workflow.metrics.judge_invocations >= 1

    async def test_execute_iterations_continues_on_judge_exception(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """A judge invocation exception must not blow up the workflow."""
        judge_cfg = _config_with_judge(config, enabled=True)
        stub = AgentStub(judge_verdicts=[ConnectionError('judge backend down')])
        workflow, _ = _build_workflow(judge_cfg, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        # Workflow still completes via normal plan-step completion path
        assert outcome == WorkflowOutcome.DONE
        assert workflow.metrics.judge_early_exits == 0

    async def test_execute_iterations_continues_on_judge_malformed_output(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Malformed judge output (missing required keys) must fall through."""
        judge_cfg = _config_with_judge(config, enabled=True)
        stub = AgentStub(judge_verdicts=[
            # Missing 'substantive_work' and 'uncovered_plan_steps'
            {'complete': True, 'reasoning': 'incomplete verdict'},
        ])
        workflow, _ = _build_workflow(judge_cfg, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE
        assert workflow.metrics.judge_early_exits == 0
        assert workflow.metrics.judge_invocations >= 1

    async def test_execute_iterations_disabled_by_default(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """With judge_after_each_iteration=False (the default), judge is never called."""
        judge_cfg = _config_with_judge(config, enabled=False)
        stub = AgentStub()  # No verdicts queued; judge should never run
        workflow, _ = _build_workflow(judge_cfg, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE
        assert workflow.metrics.judge_invocations == 0
        assert workflow.metrics.judge_early_exits == 0
        assert 'judge' not in stub.calls

    async def test_judge_does_not_inherit_env_overrides(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Judge must NOT receive env_overrides (always hits Claude API).

        Propagating ANTHROPIC_BASE_URL routes the judge through the vLLM
        bridge where max_model_len causes ServerDisconnectedError after
        2 tool-use rounds (~48 KB each exceed 80k context).  See 3cd380a079.
        """
        judge_cfg = _config_with_judge(config, enabled=True)
        judge_cfg.env_overrides = {'ANTHROPIC_BASE_URL': 'http://127.0.0.1:9999'}

        stub = AgentStub(judge_verdicts=[{
            'complete': True,
            'reasoning': 'diff implements plan.',
            'uncovered_plan_steps': [],
            'substantive_work': True,
        }])
        workflow, _ = _build_workflow(judge_cfg, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE
        # Implementer and debugger receive env_overrides.
        assert stub.env_overrides_by_role.get('implementer') == {
            'ANTHROPIC_BASE_URL': 'http://127.0.0.1:9999',
        }
        # Judge must NOT receive env_overrides — it always uses Claude API.
        assert stub.env_overrides_by_role.get('judge') is None


# ---------------------------------------------------------------------------
# Tests: Verify-Debugfix Loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVerifyDebugfixLoop:
    """Verification fails, debugger fixes, re-verify passes."""

    async def test_first_verify_fails_debugger_fixes(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)

        # First verify fails, second passes
        call_count = 0

        async def verify_sequence(worktree, cfg, module_configs=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Post-execute verify: fail
                return VerifyResult(
                    passed=False, test_output='FAILED test_farewell',
                    lint_output='', type_output='',
                    summary='Failures: tests failed',
                )
            else:
                return VerifyResult(
                    passed=True, test_output='OK', lint_output='',
                    type_output='', summary='All checks passed',
                )

        monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', verify_sequence)

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE
        # Should see debugger in the call sequence
        assert 'debugger' in stub.calls
        assert workflow.metrics.verify_attempts == 1

    async def test_verify_exhaustion_blocks(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """All verify attempts fail → BLOCKED."""
        stub = AgentStub()
        config_strict = OrchestratorConfig(
            project_root=config.project_root,
            max_verify_attempts=2,
            git=config.git,
        )
        workflow, scheduler = _build_workflow(config_strict, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=False, test_output='FAILED', lint_output='',
                type_output='', summary='tests failed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.BLOCKED
        assert scheduler.statuses['42'][-1] == 'blocked'
        # Worktree kept for debugging
        assert workflow.worktree is not None


# ---------------------------------------------------------------------------
# Tests: Review Loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReviewLoop:
    """Review finds blocking issues → re-plan → re-execute → pass."""

    async def test_blocking_review_triggers_replan(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        review_round = 0

        class ReviewAgentStub(AgentStub):
            def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
                nonlocal review_round
                # First round: reviewer finds blocking issue
                if review_round == 0 and role == 'reviewer_comprehensive':
                    review = _make_review(role, 'ISSUES_FOUND', [{
                        'severity': 'blocking',
                        'location': 'lib.py:5',
                        'category': 'missing_edge_case',
                        'description': 'No test for empty name',
                        'suggested_fix': 'Add test_farewell_empty',
                    }])
                    return AgentResult(
                        success=True, output=json.dumps(review),
                        structured_output=review, cost_usd=0.10,
                    )
                # All other reviews pass
                review = _make_review(role)
                return AgentResult(
                    success=True, output=json.dumps(review),
                    structured_output=review, cost_usd=0.10,
                )

            async def _architect(self, cwd: Path) -> AgentResult:
                """On re-plan, add a new step to address the review feedback."""
                nonlocal review_round
                task_dir = cwd / '.task'
                plan_path = task_dir / 'plan.json'

                if plan_path.exists():
                    # This is a re-plan — read existing, add fix step
                    plan = json.loads(plan_path.read_text())
                    plan['steps'].append({
                        'id': 'step-3',
                        'type': 'test',
                        'description': 'Add test for empty name',
                        'status': 'pending',
                        'commit': None,
                    })
                    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
                    review_round += 1
                    return AgentResult(success=True, output='Plan updated', cost_usd=0.40)

                # First plan
                result = await super()._architect(cwd)
                return result

            async def _implementer(self, cwd: Path) -> AgentResult:
                """Handle the extra step-3 on second pass."""
                artifacts = TaskArtifacts(cwd)
                pending = artifacts.get_pending_steps()

                if not pending:
                    return AgentResult(success=True, output='Nothing to do')

                for step in pending:
                    if step['id'] == 'step-3':
                        # Add edge case test
                        test_content = (cwd / 'test_lib.py').read_text()
                        test_content += '\ndef test_farewell_empty():\n    assert farewell("") == "Goodbye, "\n'
                        (cwd / 'test_lib.py').write_text(test_content)

                    await _run(['git', 'add', '-A'], cwd=cwd)
                    rc, _, _ = await _run(['git', 'diff', '--cached', '--quiet'], cwd=cwd)
                    if rc != 0:
                        await _run(['git', 'commit', '-m', f'Complete {step["id"]}'], cwd=cwd)
                        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=cwd)
                        artifacts.update_step_status(step['id'], 'done', commit=sha)
                    else:
                        artifacts.update_step_status(step['id'], 'done')

                return await super()._implementer(cwd)

        stub = ReviewAgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE
        # Should have gone through review cycle
        assert workflow.metrics.review_cycles >= 1
        # Architect called twice (initial + replan)
        assert stub.calls.count('architect') == 2


@pytest.mark.asyncio
class TestBaseCommitDiff:
    """Verify review uses base_commit diff, not moving main ref."""

    async def test_worktree_base_commit_written_to_artifacts(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Verify that create_worktree's base_commit flows to artifacts.init().

        This test verifies the fix for: "Reviewers receive empty diff, rubber-stamp PASS"

        The bug occurred when:
        1. Worktree A is created based on main at SHA M1
        2. While A runs, another task B merges to main (advancing main to M2)
        3. In review, workflow called get_diff_from_main() using `git diff main...HEAD`
        4. Since main now points to M2, the diff would be empty
        5. Reviewers would see no changes and rubber-stamp PASS

        The fix: workflow captures base_commit=main's SHA at worktree creation,
        passes it to artifacts.init() to persist, then reads it during review
        and uses get_diff_from_base(worktree, base_commit) instead.
        """

        class MinimalStub(AgentStub):
            async def _architect(self, cwd: Path) -> AgentResult:
                # Write minimal plan.json
                task_dir = cwd / '.task'
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / 'plan.json').write_text(json.dumps(PLAN, indent=2) + '\n')
                return AgentResult(success=True, output='{}', cost_usd=0.10)

            async def _implementer(self, cwd: Path) -> AgentResult:
                return AgentResult(success=True, output='{}', cost_usd=0.10)

            def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
                return AgentResult(success=True, output=json.dumps(_make_review(role)),
                                   structured_output=_make_review(role), cost_usd=0.10)

        stub = MinimalStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        # Mock verify to pass immediately
        monkeypatch.setattr(
            'orchestrator.agents.invoke.invoke_agent', stub.invoke_agent
        )
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All passed',
            )),
        )

        # Verify pre-condition: main advancing would normally cause empty diff
        # But we verify base_commit is captured and stored
        await workflow.run()

        # Either way the workflow completes, we can verify base_commit was stored
        if workflow.artifacts:
            stored = workflow.artifacts.read_base_commit()
            assert stored is not None, "base_commit should have been stored"
            assert len(stored) == 40, f"base_commit should be 40-char SHA, got {stored!r}"

    async def test_review_uses_base_commit_diff(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Verify get_diff_from_base returns branch changes when main advances.

        This is the critical test for the fix - simulates the scenario where:
        1. Worktree is created based on main at SHA M1
        2. Branch makes changes
        3. Another task advances main to M2
        4. get_diff_from_base must still return branch changes (not empty)
        """
        # Create worktree and capture the base_commit
        worktree_info = await git_ops.create_worktree('df-review-test')
        base_commit = worktree_info.base_commit

        # Make changes in the branch
        (worktree_info.path / 'branch_change.py').write_text('x = 1\n')
        await git_ops.commit(worktree_info.path, 'Branch change')

        # Advance main with a separate commit
        (git_ops.project_root / 'main_advance.py').write_text('y = 2\n')
        await _run(['git', 'add', 'main_advance.py'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Advance main'], cwd=git_ops.project_root)

        # KEY TEST: get_diff_from_base must return branch changes even after main advanced
        diff_from_base = await git_ops.get_diff_from_base(worktree_info.path, base_commit)
        assert 'branch_change.py' in diff_from_base, \
            f"get_diff_from_base should return branch changes after main advances, got: {diff_from_base[:200]}"

        # Verify the old method might return empty/different (proves base_commit needed)
        # Note: we don't assert on this as behavior varies by git merge topology


# ---------------------------------------------------------------------------
# Tests: Post-Merge Verification Failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPostRebaseVerifyFailure:
    """Post-rebase verification failure is non-blocking — merge queue runs its own verify."""

    async def test_post_rebase_verify_fails_proceeds_to_merge(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)

        verify_call = 0

        async def verify_fn(cwd, cfg, module_configs=None, **kwargs):
            nonlocal verify_call
            verify_call += 1
            # First verify: in-worktree (pass).
            # Second verify: post-rebase (fail) — workflow should still proceed.
            # Third+ verify: merge queue's own verification (pass).
            if verify_call <= 1 or verify_call >= 3:
                return VerifyResult(
                    passed=True, test_output='OK', lint_output='',
                    type_output='', summary='All checks passed',
                )
            return VerifyResult(
                passed=False, test_output='FAILED post-rebase',
                lint_output='', type_output='',
                summary='Post-rebase failure',
            )

        monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', verify_fn)
        monkeypatch.setattr('orchestrator.merge_queue.run_scoped_verification', verify_fn)

        outcome = await workflow.run()

        # Post-rebase verify failure is non-blocking; merge queue handles it
        assert outcome == WorkflowOutcome.DONE


# ---------------------------------------------------------------------------
# Tests: Blast Radius Expansion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBlastRadiusExpansion:
    """Plan discovers wider module set than originally assigned."""

    async def test_expansion_succeeds(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Scheduler approves expansion → workflow continues."""

        class ExpandingArchitectStub(AgentStub):
            async def _architect(self, cwd: Path) -> AgentResult:
                plan = dict(PLAN)
                plan['files'] = ['lib.py', 'test_lib.py']  # wider than assigned ['lib']
                plan['_schema_version'] = 1
                task_dir = cwd / '.task'
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
                return AgentResult(success=True, output='Plan created', cost_usd=0.50)

        stub = ExpandingArchitectStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE
        assert set(workflow.modules) == {'lib.py', 'test_lib.py'}

    async def test_expansion_denied_requeues(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Scheduler denies expansion → task requeued."""

        class ExpandingArchitectStub(AgentStub):
            async def _architect(self, cwd: Path) -> AgentResult:
                plan = dict(PLAN)
                plan['files'] = ['lib.py', 'locked_module.py']
                plan['_schema_version'] = 1
                task_dir = cwd / '.task'
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
                return AgentResult(success=True, output='Plan created', cost_usd=0.50)

        class DenyingScheduler(FakeScheduler):
            async def handle_blast_radius_expansion(self, task_id, current, needed):
                return False  # Can't acquire locks

        stub = ExpandingArchitectStub()
        deny_scheduler = DenyingScheduler()
        workflow = TaskWorkflow(
            assignment=task_assignment,
            config=config,
            git_ops=git_ops,
            scheduler=deny_scheduler,  # type: ignore[arg-type]
            briefing=FakeBriefing(),  # type: ignore[arg-type]
            mcp=FakeMcp(),  # type: ignore[arg-type]
        )

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.REQUEUED


# ---------------------------------------------------------------------------
# Tests: Artifacts Integrity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestArtifactsIntegrity:
    """Verify plan.json and iteration log are correct after workflow."""

    async def test_plan_steps_all_done(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        # We need to capture the worktree path before cleanup
        original_cleanup = git_ops.cleanup_worktree
        plan_after: dict = {}

        async def capture_then_cleanup(worktree, branch):
            nonlocal plan_after
            artifacts = TaskArtifacts(worktree)
            plan_after = artifacts.read_plan()
            await original_cleanup(worktree, branch)

        git_ops.cleanup_worktree = capture_then_cleanup

        await workflow.run()

        # All steps should be done
        for step in plan_after.get('steps', []):
            assert step['status'] == 'done', f"Step {step['id']} not done"
            assert step['commit'] is not None, f"Step {step['id']} has no commit"

    async def test_iteration_log_populated(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        iteration_log: list = []
        original_cleanup = git_ops.cleanup_worktree

        async def capture_then_cleanup(worktree, branch):
            nonlocal iteration_log
            artifacts = TaskArtifacts(worktree)
            iteration_log, _ = artifacts.read_iteration_log()
            await original_cleanup(worktree, branch)

        git_ops.cleanup_worktree = capture_then_cleanup

        await workflow.run()

        assert len(iteration_log) >= 1
        assert iteration_log[0]['agent'] == 'implementer'
        assert iteration_log[0]['source'] == 'orchestrator'
        assert len(iteration_log[0]['steps_completed']) > 0


# ---------------------------------------------------------------------------
# Tests: Plan Lock and Provenance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPlanLockAndProvenance:
    """Workflow acquires plan.lock and stamps provenance after planning."""

    async def test_workflow_has_unique_session_id(
        self, config, git_ops, task_assignment
    ):
        """TaskWorkflow generates a unique session_id in __init__."""
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)
        assert hasattr(workflow, 'session_id')
        assert workflow.session_id.startswith('42-')
        assert len(workflow.session_id) > 4  # more than just task_id

    async def test_plan_phase_creates_lock_and_stamps_provenance(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """After _plan() completes, plan.lock exists and plan.json has provenance."""
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()
        assert outcome == WorkflowOutcome.DONE

        # Worktree is cleaned up on success — capture artifacts before run
        # We can check via the workflow's plan (stamped in-memory)
        assert '_session_id' in workflow.plan
        assert '_created_at' in workflow.plan
        assert workflow.plan['_session_id'] == workflow.session_id

    async def test_execute_restamps_after_plan_overwrite(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path
    ):
        """Simulate plan overwrite: implementer stub overwrites plan.json with
        a different _session_id mid-execution. Workflow should re-stamp
        _session_id (defense-in-depth) and continue rather than blocking."""

        class OverwritingImplementerStub(AgentStub):
            async def _implementer(self, cwd: Path) -> AgentResult:
                # First call: complete real work
                result = await super()._implementer(cwd)
                # Then overwrite plan.json with a different session_id
                from orchestrator.artifacts import TaskArtifacts
                arts = TaskArtifacts(cwd)
                plan = arts.read_plan()
                plan['_session_id'] = 'hijacked-session-id'
                (cwd / '.task' / 'plan.json').write_text(
                    __import__('json').dumps(plan, indent=2) + '\n'
                )
                return result

        stub = OverwritingImplementerStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        # Workflow continues past the overwrite (re-stamps instead of blocking)
        assert outcome == WorkflowOutcome.DONE

    async def test_plan_phase_skips_architect_when_plan_locked(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """When plan.lock already exists, architect is never invoked and workflow requeues."""
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        # Pre-create the worktree, plan.json, and plan.lock
        worktree_info = await git_ops.create_worktree(task_assignment.task_id)
        workflow.worktree = worktree_info.path

        # Write .task/ artifacts directly
        from orchestrator.artifacts import TaskArtifacts
        arts = TaskArtifacts(worktree_info.path)
        arts.init(task_assignment.task_id, 'Add farewell function', 'desc', base_commit=worktree_info.base_commit)
        arts.write_plan(PLAN)
        arts.stamp_plan_provenance('pre-existing-session')
        arts.lock_plan('pre-existing-session')

        outcome = await workflow.run()
        # A duplicate workflow finding an existing lock should REQUEUE (not continue executing)
        assert outcome == WorkflowOutcome.REQUEUED

        # Architect should never have been called (plan was already locked)
        assert 'architect' not in stub.calls

    async def test_duplicate_workflow_returns_requeued_when_plan_locked(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Second workflow that finds plan.lock (owned by different session) returns REQUEUED.

        This prevents the duplicate workflow from hijacking ownership and entering
        the execute loop. The original lock-holder retains ownership.
        """
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        # Pre-create the worktree with plan.json (valid steps) stamped with a DIFFERENT session
        worktree_info = await git_ops.create_worktree(task_assignment.task_id)
        workflow.worktree = worktree_info.path

        from orchestrator.artifacts import TaskArtifacts
        arts = TaskArtifacts(worktree_info.path)
        arts.init(task_assignment.task_id, 'Add farewell function', 'desc', base_commit=worktree_info.base_commit)
        arts.write_plan(PLAN)
        original_session = 'original-owner-session'
        arts.stamp_plan_provenance(original_session)
        arts.lock_plan(original_session)

        # This is the second (duplicate) workflow — it has a different session_id
        assert workflow.session_id != original_session

        outcome = await workflow.run()

        # Duplicate workflow must REQUEUE, not continue executing
        assert outcome == WorkflowOutcome.REQUEUED
        # Architect must not have been invoked
        assert 'architect' not in stub.calls

    async def test_duplicate_workflow_does_not_restamp_provenance(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Second workflow that finds plan.lock must NOT overwrite _session_id.

        If the duplicate re-stamps provenance with its own session_id, the original
        workflow's subsequent validate_plan_owner() checks will fail with a spurious
        BLOCKED outcome. The original session_id must be preserved.
        """
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        # Pre-create the worktree with plan.json owned by a different session
        worktree_info = await git_ops.create_worktree(task_assignment.task_id)
        workflow.worktree = worktree_info.path

        from orchestrator.artifacts import TaskArtifacts
        arts = TaskArtifacts(worktree_info.path)
        arts.init(task_assignment.task_id, 'Add farewell function', 'desc', base_commit=worktree_info.base_commit)
        arts.write_plan(PLAN)
        original_session = 'original-owner-session'
        arts.stamp_plan_provenance(original_session)
        arts.lock_plan(original_session)

        # Run the duplicate workflow
        outcome = await workflow.run()
        assert outcome == WorkflowOutcome.REQUEUED

        # plan.json's _session_id must still be the original owner's session
        plan_after = arts.read_plan()
        assert plan_after.get('_session_id') == original_session, (
            f"Duplicate workflow must not overwrite _session_id. "
            f"Expected {original_session!r}, got {plan_after.get('_session_id')!r}"
        )

    async def test_plan_phase_creates_lock_and_stamps_provenance_captured(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Capture plan before cleanup to verify lock file and provenance."""
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        captured_plan: dict = {}
        lock_existed: list[bool] = []

        original_cleanup = git_ops.cleanup_worktree

        async def capture_then_cleanup(worktree, branch):
            nonlocal captured_plan
            from orchestrator.artifacts import TaskArtifacts
            arts = TaskArtifacts(worktree)
            captured_plan = arts.read_plan()
            lock_existed.append(arts.is_plan_locked())
            await original_cleanup(worktree, branch)

        git_ops.cleanup_worktree = capture_then_cleanup

        outcome = await workflow.run()
        assert outcome == WorkflowOutcome.DONE

        # plan.lock must have been created
        assert lock_existed == [True]

        # plan.json must contain provenance matching workflow session_id
        assert captured_plan.get('_session_id') == workflow.session_id
        assert '_created_at' in captured_plan


# ---------------------------------------------------------------------------
# Tests: Task Failure Escalation
# ---------------------------------------------------------------------------


def _build_workflow_with_escalation(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    agent_stub: AgentStub,
    tmp_path: Path,
) -> tuple[TaskWorkflow, FakeScheduler, EscalationQueue]:
    """Wire up a TaskWorkflow with an EscalationQueue attached."""
    from orchestrator.merge_queue import MergeWorker

    scheduler = FakeScheduler()
    queue_dir = tmp_path / 'escalation_queue'
    queue = EscalationQueue(queue_dir)
    merge_queue: asyncio.Queue = asyncio.Queue()
    worker = MergeWorker(git_ops, merge_queue)
    asyncio.create_task(worker.run(), name='test-merge-worker')
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
        escalation_queue=queue,
        merge_queue=merge_queue,
        merge_worker=worker,
    )
    return workflow, scheduler, queue


def _make_resolving_steward(queue: EscalationQueue, task_id: str) -> type:
    """Return a steward class that resolves all pending L0 escalations in start().

    Used by TestMarkBlockedFalseDoneGuard tests to drive _mark_blocked through
    the post-steward `if not remaining:` branch without a full workflow.run() cycle.
    """

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            pass

        async def start(self) -> None:
            pending = queue.get_by_task(task_id, status='pending', level=0)
            assert pending, 'expected at least one pending L0 escalation to resolve'
            for esc in pending:
                queue.resolve(
                    esc.id, 'Resolved by FakeSteward', resolved_by='fake-steward',
                )

        async def stop(self) -> None:
            pass

    return _FakeSteward


@pytest.mark.asyncio
class TestTaskFailureEscalation:
    """Blocked tasks create escalation entries in the queue."""

    async def test_blocked_task_creates_escalation(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path
    ):
        """When a task is blocked, an escalation with category='task_failure' is created."""
        stub = AgentStub()
        workflow, scheduler, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=False, test_output='FAILED', lint_output='',
                type_output='', summary='tests failed',
            )),
        )

        # Use strict verify limits so it blocks quickly
        config_strict = OrchestratorConfig(
            project_root=config.project_root,
            max_verify_attempts=1,
            git=config.git,
        )
        workflow.config = config_strict

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.BLOCKED

        # Check escalation was created
        escalations = queue.get_by_task('42')
        assert len(escalations) == 1
        esc = escalations[0]
        assert esc.category == 'task_failure'
        assert esc.severity == 'blocking'
        assert esc.agent_role == 'orchestrator'
        assert esc.task_id == '42'
        assert esc.status == 'pending'

    async def test_escalation_has_correct_fields(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path
    ):
        """Escalation carries workflow_state, worktree, and reason in summary/detail."""

        class FailingArchitectStub(AgentStub):
            async def _architect(self, cwd: Path) -> AgentResult:
                return AgentResult(success=False, output='Cannot plan', cost_usd=0.10)

        stub = FailingArchitectStub()
        workflow, scheduler, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='ok',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.BLOCKED

        escalations = queue.get_by_task('42')
        assert len(escalations) == 1
        esc = escalations[0]
        assert esc.workflow_state == 'blocked'
        assert esc.suggested_action == 'investigate_and_retry'
        assert 'Planning failed' in esc.summary

    async def test_no_escalation_without_queue(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Without an escalation_queue, _mark_blocked still works (no crash)."""
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=False, test_output='FAILED', lint_output='',
                type_output='', summary='tests failed',
            )),
        )

        config_strict = OrchestratorConfig(
            project_root=config.project_root,
            max_verify_attempts=1,
            git=config.git,
        )
        workflow.config = config_strict

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.BLOCKED
        assert scheduler.statuses['42'][-1] == 'blocked'


# ---------------------------------------------------------------------------
# Tests: Corrupted Iteration Log Escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCorruptedIterationLogEscalation:
    """Corrupted iteration log lines trigger info-severity escalation."""

    async def test_corrupted_iteration_log_escalates(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path
    ):
        """Workflow completes DONE despite corruption; info escalation created."""

        class CorruptingAgentStub(AgentStub):
            async def _architect(self, cwd: Path) -> AgentResult:
                result = await super()._architect(cwd)
                # Inject a corrupted iteration log line so the orchestrator
                # finds it when it reads the log before the first implementer call
                log_path = cwd / '.task' / 'iterations.jsonl'
                log_path.write_text(r'{"bad\!escape": true}' + '\n')
                return result

        stub = CorruptingAgentStub()
        workflow, scheduler, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE

        # Check that an info-severity escalation was created for the corruption
        escalations = queue.get_by_task('42')
        info_escs = [e for e in escalations if e.severity == 'info']
        assert len(info_escs) >= 1
        esc = info_escs[0]
        assert esc.category == 'infra_issue'
        assert 'corrupted' in esc.summary.lower()


# ---------------------------------------------------------------------------
# Tests: Done-is-Terminal (_mark_blocked guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDoneIsTerminal:
    """_mark_blocked must be a no-op when workflow.state is already DONE."""

    async def test_mark_blocked_skipped_when_done(
        self, config, git_ops, task_assignment
    ):
        """When state=DONE, _mark_blocked returns DONE without calling set_task_status."""
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        # Manually put the workflow into the DONE state (simulates completed task)
        workflow.state = WorkflowState.DONE

        outcome = await workflow._mark_blocked('late error from duplicate workflow')

        assert outcome == WorkflowOutcome.DONE
        # 'blocked' must NOT appear in scheduler status history
        assert 'blocked' not in scheduler.statuses.get(task_assignment.task_id, [])

    async def test_mark_blocked_skips_escalation_when_done(
        self, config, git_ops, task_assignment, tmp_path
    ):
        """When state=DONE, _mark_blocked must not create an escalation entry."""
        stub = AgentStub()
        workflow, scheduler, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path
        )

        workflow.state = WorkflowState.DONE

        outcome = await workflow._mark_blocked('late error after done')

        assert outcome == WorkflowOutcome.DONE
        # No escalation should be created for this task
        assert queue.get_by_task(task_assignment.task_id) == []

    async def test_mark_blocked_works_normally_when_not_done(
        self, config, git_ops, task_assignment, tmp_path
    ):
        """Without DONE state, _mark_blocked sets blocked and creates an escalation."""
        stub = AgentStub()
        workflow, scheduler, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path
        )

        # Default state is PLAN — not DONE
        assert workflow.state == WorkflowState.PLAN

        outcome = await workflow._mark_blocked('genuine failure')

        assert outcome == WorkflowOutcome.BLOCKED
        assert 'blocked' in scheduler.statuses.get(task_assignment.task_id, [])
        escalations = queue.get_by_task(task_assignment.task_id)
        assert len(escalations) == 1
        assert escalations[0].category == 'task_failure'


# ---------------------------------------------------------------------------
# Tests: Reviewer Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReviewerErrors:
    """Reviewer failures are detected, retried, and escalated."""

    async def test_reviewer_error_blocks_task(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path
    ):
        """A reviewer that persistently fails blocks the task."""
        config.max_reviewer_retries = 1
        config.reviewer_stagger_secs = 0.0  # no delay in tests

        class ErrorReviewerStub(AgentStub):
            def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
                if role == 'reviewer_comprehensive':
                    # Simulate 401 — unparseable output, no structured_output
                    return AgentResult(
                        success=False,
                        output='Failed to authenticate. API Error: 401',
                        structured_output=None,
                        cost_usd=0.0,
                    )
                review = _make_review(role)
                return AgentResult(
                    success=True, output=json.dumps(review),
                    structured_output=review, cost_usd=0.10,
                )

        stub = ErrorReviewerStub()
        workflow, scheduler, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.BLOCKED
        assert 'blocked' in scheduler.statuses.get(task_assignment.task_id, [])

    async def test_all_reviewers_error_blocks_task(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path
    ):
        """When every reviewer errors, the task is blocked."""
        config.max_reviewer_retries = 0
        config.reviewer_stagger_secs = 0.0

        class AllErrorStub(AgentStub):
            def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
                return AgentResult(
                    success=False,
                    output='Failed to authenticate. API Error: 401',
                    structured_output=None,
                    cost_usd=0.0,
                )

        stub = AllErrorStub()
        workflow, scheduler, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.BLOCKED
        # Verify the review artifacts have ERROR verdicts
        assert workflow.artifacts is not None
        reviews = workflow.artifacts.read_reviews()
        for review in reviews.values():
            assert review['verdict'] == 'ERROR'

    async def test_reviewer_retry_heals(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """A reviewer that fails then succeeds on retry lets the task pass."""
        config.max_reviewer_retries = 2
        config.reviewer_stagger_secs = 0.0

        call_counts: dict[str, int] = {}

        class RetryHealStub(AgentStub):
            def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
                call_counts[role] = call_counts.get(role, 0) + 1
                if role == 'reviewer_comprehensive' and call_counts[role] <= 1:
                    # First call fails
                    return AgentResult(
                        success=False,
                        output='Connection refused',
                        structured_output=None,
                        cost_usd=0.0,
                    )
                # All others (and retry) pass
                review = _make_review(role)
                return AgentResult(
                    success=True, output=json.dumps(review),
                    structured_output=review, cost_usd=0.10,
                )

        stub = RetryHealStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE
        # comprehensive reviewer was called twice (initial fail + retry success)
        assert call_counts['reviewer_comprehensive'] == 2


# ---------------------------------------------------------------------------
# Tests: Ghost-Loop Guard (stale worktree false positive)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGhostLoopGuard:
    """Ghost-loop check must not skip when no implementation was done."""

    async def test_stale_branch_point_proceeds_to_execute(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Reused worktree with no implementation entries → normal execution."""
        # 1. Pre-create the worktree (simulates a prior run that planned but requeued)
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        task_dir = wt / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text(json.dumps(PLAN, indent=2) + '\n')

        # 2. Advance main so the worktree HEAD becomes a stale ancestor
        (config.project_root / 'other.py').write_text('x = 1\n')
        from orchestrator.git_ops import _run
        await _run(['git', 'add', '-A'], cwd=config.project_root)
        await _run(['git', 'commit', '-m', 'Advance main'], cwd=config.project_root)

        # 3. Build and run workflow — it should reuse the worktree and NOT skip
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE
        # The implementer MUST have been called (ghost-loop did not skip)
        assert 'implementer' in stub.calls

    async def test_legitimate_ghost_loop_skips_to_done(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Worktree with implementer iteration entry + HEAD on main → skip."""
        # 1. Create worktree and simulate prior implementation
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        task_dir = wt / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'plan.json').write_text(json.dumps(PLAN, indent=2) + '\n')

        # Write an implementer iteration log entry
        import json as _json
        (task_dir / 'iterations.jsonl').write_text(
            _json.dumps({'agent': 'implementer', 'iteration': 1, 'steps_attempted': []}) + '\n'
        )

        # Make an implementation commit on the branch
        (wt / 'farewell.py').write_text('def farewell(name):\n    return f"Bye, {name}"\n')
        await git_ops.commit(wt, 'Implement farewell')

        # 2. Merge the branch into main so its HEAD becomes an ancestor
        result = await git_ops.merge_to_main(wt, task_assignment.task_id)
        assert result.success
        await git_ops.advance_main(result.merge_commit)
        if result.merge_worktree:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)

        # 3. Build and run workflow — should detect prior merge and skip
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE
        # The implementer must NOT have been called (ghost-loop skipped)
        assert 'implementer' not in stub.calls


# ---------------------------------------------------------------------------
# Tests: WIP Recovery No Advance (pop_conflict on CAS-failure path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWipRecoveryNoAdvance:
    """wip_recovery_no_advance outcome creates L1 wip_conflict escalation and returns BLOCKED."""

    async def test_workflow_handles_wip_recovery_no_advance_creates_level1_escalation(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        """When advance_main returns pop_conflict_no_advance, the workflow creates a
        level-1 wip_conflict escalation and returns BLOCKED (not DONE — the merge
        did NOT land on main).
        """
        recovery_branch = 'wip/recovery-no-advance-test'
        stub = AgentStub()
        workflow, scheduler, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )
        monkeypatch.setattr(
            'orchestrator.merge_queue.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        async def _fake_advance(*args, **kwargs):
            git_ops._last_recovery_branch = recovery_branch
            return 'pop_conflict_no_advance'

        monkeypatch.setattr(git_ops, 'advance_main', _fake_advance)

        # Run the workflow as a separate task; it will block waiting on _escalation_event
        workflow_task = asyncio.create_task(workflow.run())

        # Poll until the wip_conflict escalation appears (meaning handler reached its wait)
        for _ in range(200):
            escs = queue.get_by_task('42')
            if any(e.category == 'wip_conflict' for e in escs):
                break
            await asyncio.sleep(0.05)
        else:
            workflow_task.cancel()
            pytest.fail('Timeout: wip_conflict escalation never created within 10s')

        # Fire the event to unblock _handle_wip_recovery_no_advance
        assert workflow._escalation_event is not None, 'Handler must have set _escalation_event'
        workflow._escalation_event.set()

        outcome = await workflow_task

        # Merge did NOT land — must return BLOCKED, not DONE
        assert outcome == WorkflowOutcome.BLOCKED

        wip_escs = [e for e in queue.get_by_task('42') if e.category == 'wip_conflict']
        assert len(wip_escs) == 1
        esc = wip_escs[0]
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert recovery_branch in (esc.summary + esc.detail)
        # Detail must explain merge did not advance (not the same as done_wip_recovery)
        assert 'did not advance' in esc.detail.lower() or 'merge did not' in esc.detail.lower()


# ---------------------------------------------------------------------------
# Protocol conformance — see TYPE_CHECKING block at the bottom of this file.
# Static assertions (pyright-verified) replaced the old hasattr/isinstance
# runtime checks, which only tested attribute presence, not method signatures.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: FakeScheduler.get_cached_status
# ---------------------------------------------------------------------------


class TestFakeSchedulerCachedStatus:
    """FakeScheduler.get_cached_status returns the last status set for a task."""

    def test_get_cached_status_returns_none_before_any_set(self):
        """Before any set_task_status call, get_cached_status returns None."""
        fake = FakeScheduler()
        assert fake.get_cached_status('x') is None

    @pytest.mark.asyncio
    async def test_get_cached_status_returns_last_set_status(self):
        """get_cached_status returns the most recent status after multiple sets."""
        fake = FakeScheduler()
        await fake.set_task_status('x', 'in-progress')
        await fake.set_task_status('x', 'done')
        assert fake.get_cached_status('x') == 'done'


# ---------------------------------------------------------------------------
# Tests: _EvalScheduler.get_cached_status
# ---------------------------------------------------------------------------


class TestEvalSchedulerCachedStatus:
    """_EvalScheduler.get_cached_status tracks status set via set_task_status."""

    def test_eval_scheduler_get_cached_status_returns_none_initially(self):
        """Before any set_task_status call, get_cached_status returns None."""
        from orchestrator.config import OrchestratorConfig
        from orchestrator.evals.runner import _EvalScheduler

        sched = _EvalScheduler(OrchestratorConfig())
        assert sched.get_cached_status('99') is None

    @pytest.mark.asyncio
    async def test_eval_scheduler_get_cached_status_tracks_set_task_status(self):
        """get_cached_status returns the status written by set_task_status."""
        from orchestrator.config import OrchestratorConfig
        from orchestrator.evals.runner import _EvalScheduler

        sched = _EvalScheduler(OrchestratorConfig())
        await sched.set_task_status('99', 'done')
        assert sched.get_cached_status('99') == 'done'

    @pytest.mark.asyncio
    async def test_eval_scheduler_uses_status_cache_attribute(self):
        """_EvalScheduler uses _status_cache (not _cache) to match Scheduler naming."""
        from orchestrator.config import OrchestratorConfig
        from orchestrator.evals.runner import _EvalScheduler

        sched = _EvalScheduler(OrchestratorConfig())
        await sched.set_task_status('99', 'done')
        assert sched._status_cache == {'99': 'done'}
        assert not hasattr(sched, '_cache')

    @pytest.mark.asyncio
    async def test_eval_scheduler_set_task_status_writes_via_set_cached_status(self):
        """_EvalScheduler.set_task_status() must write via _set_cached_status()."""
        from orchestrator.config import OrchestratorConfig
        from orchestrator.evals.runner import _EvalScheduler

        sched = _EvalScheduler(OrchestratorConfig())

        recorded = _spy_set_cached_status(sched)

        await sched.set_task_status('99', 'done')
        assert ('99', 'done') in recorded


# ---------------------------------------------------------------------------
# Tests: Prerequisites format validation
# ---------------------------------------------------------------------------


class StringPrereqsArchitectStub(AgentStub):
    """AgentStub subclass that writes a plan with plain-string prerequisites."""

    async def _architect(self, cwd: Path) -> AgentResult:
        task_dir = cwd / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        bad_plan = {
            'task_id': '42',
            'title': 'Add farewell function',
            'modules': ['lib'],
            'analysis': 'Simple function addition with TDD',
            'prerequisites': ['install deps', 'set up config'],  # plain strings — invalid
            'steps': [
                {
                    'id': 'step-1',
                    'type': 'test',
                    'description': 'Write failing test for farewell()',
                    'status': 'pending',
                    'commit': None,
                },
            ],
            '_schema_version': 1,
        }
        (task_dir / 'plan.json').write_text(json.dumps(bad_plan, indent=2) + '\n')
        return AgentResult(success=True, output='Plan created', cost_usd=0.50)


@pytest.mark.asyncio
class TestPrerequisitesValidation:
    """Workflow blocks with descriptive error when architect writes string prerequisites."""

    async def test_string_prerequisites_block_workflow(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path
    ):
        """Architect plan with ['string prereq'] prerequisites causes BLOCKED outcome.

        The escalation message must mention 'prerequisites' (descriptive error),
        not a generic AttributeError crash.  This verifies that validate_plan_prerequisites()
        is called at the plan checkpoint rather than failing silently later.
        """
        stub = StringPrereqsArchitectStub()
        workflow, scheduler, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )

        monkeypatch.setattr('orchestrator.agents.invoke.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.BLOCKED
        assert scheduler.statuses['42'][-1] == 'blocked'

        # The escalation must describe the prerequisites problem, not a generic crash
        escalations = queue.get_by_task('42')
        assert len(escalations) == 1
        esc = escalations[0]
        # summary or detail must mention 'prerequisites' to confirm the validation ran
        assert 'prerequisites' in esc.summary.lower() or 'prerequisites' in (esc.detail or '').lower()


# ---------------------------------------------------------------------------
# Tests: _mark_blocked false-done guard (_has_prior_implementation check)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMarkBlockedFalseDoneGuard:
    """Empty-branch is_ancestor path must not false-done — must requeue instead.

    Regression for task 839: an empty task branch (worktree HEAD == base commit)
    trivially satisfies is_ancestor(HEAD, main), causing _mark_blocked to mark
    the task 'done' with no code landed on main.
    """

    async def test_empty_branch_with_resolved_l0_requeues_not_done(
        self, config, git_ops, task_assignment, tmp_path
    ):
        """Empty task branch with steward-resolved L0 must requeue, not false-done.

        The base commit is always an ancestor of main, so is_ancestor returns True
        even when no implementation was committed.  Without the
        _has_prior_implementation() guard this path marks the task 'done'.
        With the guard it detects the stale branch point and falls through to the
        normal requeue path.  This test FAILS on unmodified code (step-1 TDD).
        """
        # 1. Create a fresh worktree with no implementation commits (base commit only)
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path

        # 2. Build workflow with escalation queue, wire up worktree and artifacts
        #    (no iterations.jsonl → _has_prior_implementation() returns False)
        stub = AgentStub()
        workflow, scheduler, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )
        workflow.worktree = wt
        workflow.artifacts = TaskArtifacts(wt)

        # 3. FakeSteward: resolves all pending L0 escalations immediately in start()
        #    so _await_steward_completion() exits without timeout, and remaining == []
        workflow._steward_factory = _make_resolving_steward(
            queue, task_assignment.task_id,
        )

        # 4. Call _mark_blocked directly (no full run() cycle needed)
        outcome = await workflow._mark_blocked('synthetic failure')

        # 5. Assertions: must REQUEUE, not false-done
        assert outcome == WorkflowOutcome.REQUEUED, (
            f'Expected REQUEUED but got {outcome!r} — '
            'empty-branch is_ancestor guard is missing or broken'
        )
        statuses = scheduler.statuses.get(task_assignment.task_id, [])
        assert 'done' not in statuses, (
            f"'done' must not appear in scheduler statuses for an empty branch: {statuses}"
        )
        assert statuses[-1] == 'pending', (
            f"Last scheduler status must be 'pending' after requeue, got: {statuses}"
        )
        assert workflow.state != WorkflowState.DONE, (
            f'workflow.state must not be DONE after empty-branch guard: {workflow.state!r}'
        )

    async def test_genuine_prior_merge_still_fast_paths_to_done(
        self, config, git_ops, task_assignment, tmp_path
    ):
        """Positive control: branch with impl entry + merged to main must still fast-path to DONE.

        Verifies the guard is not over-conservative: when a real implementer
        iteration entry exists and the branch HEAD is a genuine ancestor of main
        (because it was merged), _mark_blocked must still return DONE.
        """
        import json as _json

        # 1. Create worktree and write an implementer iteration entry
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        task_dir = wt / '.task'
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'iterations.jsonl').write_text(
            _json.dumps({'agent': 'implementer', 'iteration': 1, 'steps_attempted': []}) + '\n'
        )

        # 2. Make a real implementation commit on the branch and merge it to main
        (wt / 'farewell.py').write_text('def farewell(name):\n    return f"Bye, {name}"\n')
        await git_ops.commit(wt, 'Implement farewell')
        result = await git_ops.merge_to_main(wt, task_assignment.task_id)
        assert result.success
        await git_ops.advance_main(result.merge_commit)
        if result.merge_worktree:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)

        # 3. Build workflow with escalation queue, wire up worktree and artifacts
        #    (has implementer iteration entry → _has_prior_implementation() returns True)
        stub = AgentStub()
        workflow, scheduler, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )
        workflow.worktree = wt
        workflow.artifacts = TaskArtifacts(wt)

        # 4. FakeSteward resolves all pending L0 escalations immediately in start()
        workflow._steward_factory = _make_resolving_steward(
            queue, task_assignment.task_id,
        )

        # 5. Call _mark_blocked directly
        outcome = await workflow._mark_blocked('synthetic failure')

        # 6. Assertions: genuine prior merge must still fast-path to DONE
        assert outcome == WorkflowOutcome.DONE, (
            f'Expected DONE for genuine prior merge but got {outcome!r} — '
            'the _has_prior_implementation guard is too conservative'
        )
        statuses = scheduler.statuses.get(task_assignment.task_id, [])
        assert statuses[-1] == 'done', (
            f"Last scheduler status must be 'done' for genuine prior merge, got: {statuses}"
        )
        assert workflow.state == WorkflowState.DONE

    async def test_git_exception_in_ancestor_check_requeues(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path
    ):
        """git exception during is_ancestor must fall through to requeue, not crash.

        The try/except at the is_ancestor call site swallows all exceptions and
        falls through to the normal requeue path.  This test ensures a future
        refactor can't accidentally turn a RuntimeError into an unguarded raise
        or a false-done transition.
        """
        # 1. Create a fresh worktree (base commit only — no implementation commits)
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path

        # 2. Build workflow, wire up worktree and artifacts
        stub = AgentStub()
        workflow, scheduler, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )
        workflow.worktree = wt
        workflow.artifacts = TaskArtifacts(wt)

        # 3. FakeSteward resolves the L0 so we reach the is_ancestor check
        workflow._steward_factory = _make_resolving_steward(
            queue, task_assignment.task_id,
        )

        # 4. Force is_ancestor to raise RuntimeError — simulates a git subprocess
        #    failure (e.g. corrupted index, network mount offline, etc.)
        async def _boom(*args, **kwargs):  # noqa: ARG001
            raise RuntimeError('simulated git failure')

        monkeypatch.setattr(git_ops, 'is_ancestor', _boom)

        # 5. Call _mark_blocked — must survive the exception and requeue
        outcome = await workflow._mark_blocked('synthetic failure')

        # 6. Assertions: exception path must fall through to requeue
        assert outcome == WorkflowOutcome.REQUEUED, (
            f'Expected REQUEUED after git exception but got {outcome!r} — '
            'the except-block in _mark_blocked is broken'
        )
        statuses = scheduler.statuses.get(task_assignment.task_id, [])
        assert 'done' not in statuses, (
            f"'done' must not appear after a git exception: {statuses}"
        )
        assert statuses[-1] == 'pending', (
            f"Last status must be 'pending' after exception-path requeue: {statuses}"
        )


# ---------------------------------------------------------------------------
# File-structure invariants
# ---------------------------------------------------------------------------


class TestFileStructureInvariants:
    """Structural invariants about test_workflow_e2e.py itself.

    These meta-tests catch accidental duplication or deletion of critical
    comment blocks and code sections that must appear exactly once.
    """

    def test_conformance_comment_header_appears_exactly_once(self) -> None:
        """The 'Static Protocol conformance checks (pyright-verified)' header must appear exactly once.

        This guards against the orphaned-duplicate scenario: if an edit accidentally
        leaves a stale copy of the comment block mid-file (while the real block with
        the ``if TYPE_CHECKING:`` code lives near the bottom), this test fails loudly
        rather than silently misleading future readers. See task 735.

        Counts only lines whose stripped content IS the header comment (excludes
        string literals inside test code that happen to contain the same text).
        """
        content = Path(__file__).read_text()
        # Count only lines whose stripped content is exactly the header comment,
        # not string literals inside test code that happen to reference it.
        header_line = "# Static Protocol conformance checks (pyright-verified)"
        count = sum(1 for line in content.splitlines() if line.strip() == header_line)
        assert count == 1, (
            f"Expected exactly 1 comment line matching {header_line!r} in {__file__}, "
            f"found {count}. An orphaned duplicate of the conformance comment "
            "block must be removed — keep only the real block near the end of "
            "the file that precedes the 'if TYPE_CHECKING:' conformance code. "
            "See task 735."
        )


# ---------------------------------------------------------------------------
# Static Protocol conformance checks (pyright-verified)
#
# This block MUST live at the bottom of the file, after the FakeScheduler class
# definition (line ~361). Placing it at the top would require forward-referencing
# FakeScheduler() before its definition, which forces a # type: ignore[name-defined]
# suppression. pyright then infers the expression type as Unknown, which trivially
# satisfies any Protocol — making the check catch nothing. Keeping the block here
# allows pyright to resolve FakeScheduler to its concrete class and verify full
# structural conformance (parameter names, types, return types, positional-only
# markers), not just attribute presence.
#
# CI gate — how enforcement actually reaches this file:
#   • hooks/project-checks (invoked by hooks/pre-commit on main-branch commits)
#     iterates over PYRIGHT_PACKAGES=(fused-memory orchestrator dashboard) and
#     runs `uv run pyright` from each package directory, failing the commit on
#     any error.
#   • [tool.pyright] include = ["src", "tests"] in orchestrator/pyproject.toml
#     ensures every file under orchestrator/tests/ — including this one — is
#     type-checked when pyright runs from orchestrator/.
#   Both legs of the gate are pinned by tests/test_pyright_gate_for_workflow_e2e.py,
#   so accidental removal of either fails at normal pytest time.
#
# Experimental verification: enforcement was confirmed in commit 357fa4d6a5 and
# re-verified during task 699 by temporarily mutating FakeScheduler.get_cached_status
# to return `int | None`; pyright flagged line 2098 with reportAssignmentType as
# expected, then the file was reverted.
#
# Runtime belt-and-braces: TestFakeSchedulerCachedStatus and
# TestEvalSchedulerCachedStatus (above) provide runtime coverage of
# get_cached_status behaviour — the static conformance block only catches
# signature drift, not behavioural regressions.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from orchestrator.evals.runner import _EvalScheduler as _EvalSchedulerStatic
    from orchestrator.workflow import _SchedulerLike

    _fake_scheduler_conforms: _SchedulerLike = FakeScheduler()
    _eval_scheduler_conforms: _SchedulerLike = _EvalSchedulerStatic(OrchestratorConfig())
