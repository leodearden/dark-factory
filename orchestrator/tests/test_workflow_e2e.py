"""Deterministic e2e tests for the workflow state machine.

Uses real git operations and file I/O. Agent invocations and verification
are stubbed with deterministic side-effect functions that write actual files
to the worktree, simulating what real agents would do.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
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

    def __init__(self, verify_results: list[VerifyResult] | None = None):
        self.calls: list[str] = []
        self._impl_iteration = 0
        # Sequence of verify results to return (pops from front)
        self._verify_results = list(verify_results or [VerifyResult(
            passed=True, test_output='', lint_output='', type_output='',
            summary='All checks passed',
        )])

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
    ) -> AgentResult:
        """Determine role from system_prompt content, perform side effects."""
        role = self._detect_role(system_prompt)
        self.calls.append(role)

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
        else:
            return AgentResult(success=True, output='OK')

    def _detect_role(self, system_prompt: str) -> str:
        if 'TDD architect' in system_prompt:
            return 'architect'
        if 'TDD implementer' in system_prompt:
            return 'implementer'
        if 'debugger' in system_prompt.lower() and 'You are a debugger' in system_prompt:
            return 'debugger'
        if 'code reviewer' in system_prompt.lower():
            # Extract reviewer name from prompt
            if 'test_analyst' in system_prompt:
                return 'reviewer_test_analyst'
            if 'reuse_auditor' in system_prompt:
                return 'reviewer_reuse_auditor'
            if 'architect_reviewer' in system_prompt:
                return 'reviewer_architect_reviewer'
            if 'performance' in system_prompt:
                return 'reviewer_performance'
            if 'robustness' in system_prompt:
                return 'reviewer_robustness'
            return 'reviewer_unknown'
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

    def mcp_config_json(self, **kwargs) -> dict:
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

    def release(self, task_id: str) -> None:
        pass


class FakeBriefing:
    """BriefingAssembler that returns canned prompts."""

    async def build_architect_prompt(self, task: dict, worktree=None, context: str | None = None) -> str:
        return f'Plan task: {task.get("title", "")}'

    async def build_implementer_prompt(
        self, plan: dict, iteration_log: list, context: str | None = None
    ) -> str:
        return 'Implement the plan'

    async def build_debugger_prompt(
        self, failures: str, plan: dict, context: str | None = None
    ) -> str:
        return f'Fix: {failures[:100]}'

    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = None
    ) -> str:
        return f'Review ({reviewer_type}): {diff[:100]}'

    async def build_merger_prompt(
        self, conflicts: str, task_intent: str, context: str | None = None
    ) -> str:
        return f'Merge: {conflicts[:100]}'


def _build_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    agent_stub: AgentStub,
) -> tuple[TaskWorkflow, FakeScheduler]:
    """Wire up a TaskWorkflow with all fakes injected."""
    scheduler = FakeScheduler()
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
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
            'orchestrator.workflow.invoke_agent', stub.invoke_agent
        )
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        await workflow.run()

        # Expected: architect, implementer, 5 reviewers (parallel, order varies)
        assert stub.calls[0] == 'architect'
        assert stub.calls[1] == 'implementer'
        reviewer_calls = sorted(stub.calls[2:7])
        assert reviewer_calls == sorted([
            'reviewer_architect_reviewer',
            'reviewer_performance',
            'reviewer_reuse_auditor',
            'reviewer_robustness',
            'reviewer_test_analyst',
        ])

    async def test_code_appears_on_main_after_merge(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()
        assert outcome == WorkflowOutcome.DONE

        # Verify the farewell function is on main
        lib_content = (config.project_root / 'lib.py').read_text()
        assert 'def farewell' in lib_content
        assert 'Goodbye' in lib_content

    async def test_worktree_cleaned_up_on_success(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, _ = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        await workflow.run()

        # 1 architect + 1 implementer + 5 reviewers = 7
        assert workflow.metrics.agent_invocations == 7
        assert workflow.metrics.total_cost_usd > 0
        assert workflow.metrics.execute_iterations == 1


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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)

        # First verify fails, second passes
        call_count = 0

        async def verify_sequence(worktree, cfg, module_config=None):
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

        monkeypatch.setattr('orchestrator.workflow.run_verification', verify_sequence)

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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
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
                # First round: one reviewer finds blocking issue
                if review_round == 0 and role == 'reviewer_test_analyst':
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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
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


# ---------------------------------------------------------------------------
# Tests: Post-Merge Verification Failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPostMergeFailure:
    """Merge succeeds textually but post-merge tests fail → revert → BLOCKED."""

    async def test_post_merge_verify_fails_reverts(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)

        verify_call = 0

        async def verify_fn(cwd, cfg, module_config=None):
            nonlocal verify_call
            verify_call += 1
            if cwd == config.project_root:
                # Post-merge verify on main — fail
                return VerifyResult(
                    passed=False, test_output='FAILED on main',
                    lint_output='', type_output='',
                    summary='Post-merge failure',
                )
            # In-worktree verify — pass
            return VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )

        monkeypatch.setattr('orchestrator.workflow.run_verification', verify_fn)

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.BLOCKED
        assert scheduler.statuses['42'][-1] == 'blocked'

        # Main should be reverted — farewell should NOT be on main
        lib_content = (config.project_root / 'lib.py').read_text()
        assert 'farewell' not in lib_content


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
                plan['modules'] = ['lib', 'utils']  # wider than assigned ['lib']
                plan['_schema_version'] = 1
                task_dir = cwd / '.task'
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
                return AgentResult(success=True, output='Plan created', cost_usd=0.50)

        stub = ExpandingArchitectStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.DONE
        assert set(workflow.modules) == {'lib', 'utils'}

    async def test_expansion_denied_requeues(
        self, config, git_ops, task_assignment, monkeypatch
    ):
        """Scheduler denies expansion → task requeued."""

        class ExpandingArchitectStub(AgentStub):
            async def _architect(self, cwd: Path) -> AgentResult:
                plan = dict(PLAN)
                plan['modules'] = ['lib', 'locked_module']
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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
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
    scheduler = FakeScheduler()
    queue_dir = tmp_path / 'escalation_queue'
    queue = EscalationQueue(queue_dir)
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
        escalation_queue=queue,
    )
    return workflow, scheduler, queue


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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
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

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_verification',
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
