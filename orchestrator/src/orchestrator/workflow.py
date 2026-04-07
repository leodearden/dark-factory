"""Per-task workflow state machine: PLAN → EXECUTE → VERIFY → REVIEW → MERGE → DONE."""

from __future__ import annotations

import asyncio
import enum
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.invoke import AgentResult, invoke_agent
from orchestrator.agents.roles import (
    ALL_REVIEWERS,
    ARCHITECT,
    DEBUGGER,
    IMPLEMENTER,
    MERGER,
    AgentRole,
)
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.verify import run_verification

if TYPE_CHECKING:
    from orchestrator.mcp_lifecycle import McpLifecycle

logger = logging.getLogger(__name__)


class WorkflowState(enum.Enum):
    PLAN = 'plan'
    EXECUTE = 'execute'
    VERIFY = 'verify'
    REVIEW = 'review'
    MERGE = 'merge'
    DONE = 'done'
    BLOCKED = 'blocked'


class WorkflowOutcome(enum.Enum):
    DONE = 'done'
    BLOCKED = 'blocked'
    REQUEUED = 'requeued'


@dataclass
class WorkflowMetrics:
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    agent_invocations: int = 0
    execute_iterations: int = 0
    verify_attempts: int = 0
    review_cycles: int = 0


class TaskWorkflow:
    """Per-task state machine."""

    def __init__(
        self,
        assignment: TaskAssignment,
        config: OrchestratorConfig,
        git_ops: GitOps,
        scheduler: Scheduler,
        briefing: BriefingAssembler,
        mcp: McpLifecycle,
    ):
        self.assignment = assignment
        self.config = config
        self.git_ops = git_ops
        self.scheduler = scheduler
        self.briefing = briefing
        self.mcp = mcp

        self.state = WorkflowState.PLAN
        self.task = assignment.task
        self.task_id = assignment.task_id
        self.modules = list(assignment.modules)
        self.worktree: Path | None = None
        self.artifacts: TaskArtifacts | None = None
        self.plan: dict = {}
        self.metrics = WorkflowMetrics()

    async def run(self) -> WorkflowOutcome:
        """Execute the full state machine."""
        branch_name = self.task_id
        try:
            # Set task in-progress
            await self.scheduler.set_task_status(self.task_id, 'in-progress')

            # Create worktree
            self.worktree = await self.git_ops.create_worktree(branch_name)
            self.artifacts = TaskArtifacts(self.worktree)
            self.artifacts.init(
                self.task_id,
                self.task.get('title', ''),
                self.task.get('description', ''),
            )

            # PLAN
            self.state = WorkflowState.PLAN
            plan_outcome = await self._plan()
            if plan_outcome == WorkflowOutcome.REQUEUED:
                return WorkflowOutcome.REQUEUED
            if plan_outcome == WorkflowOutcome.BLOCKED:
                return await self._mark_blocked('Planning failed')

            # EXECUTE + VERIFY + REVIEW loop
            outcome = await self._execute_verify_review_loop()
            if outcome != WorkflowOutcome.DONE:
                return outcome

            # MERGE
            self.state = WorkflowState.MERGE
            merge_outcome = await self._merge(branch_name)
            if merge_outcome != WorkflowOutcome.DONE:
                return merge_outcome

            # POST-MERGE VERIFY
            post_merge = await run_verification(self.config.project_root, self.config)
            if not post_merge.passed:
                logger.error(f'Task {self.task_id}: post-merge verification failed')
                await self.git_ops.revert_last_merge(self.config.project_root)
                return await self._mark_blocked(
                    f'Post-merge verification failed: {post_merge.summary}'
                )

            # SUCCESS
            self.state = WorkflowState.DONE
            await self.scheduler.set_task_status(self.task_id, 'done')
            logger.info(
                f'Task {self.task_id} DONE — '
                f'cost=${self.metrics.total_cost_usd:.2f} '
                f'invocations={self.metrics.agent_invocations}'
            )
            return WorkflowOutcome.DONE

        except Exception as e:
            logger.exception(f'Task {self.task_id} workflow error: {e}')
            return await self._mark_blocked(f'Workflow error: {e}')

        finally:
            # Cleanup worktree (only if done — keep for debugging if blocked)
            if self.state == WorkflowState.DONE and self.worktree:
                await self.git_ops.cleanup_worktree(self.worktree, branch_name)

    async def _plan(self) -> WorkflowOutcome:
        """Invoke the architect to produce a plan."""
        assert self.worktree is not None and self.artifacts is not None
        prompt = await self.briefing.build_architect_prompt(self.task)
        result = await self._invoke(ARCHITECT, prompt, self.worktree)

        if not result.success:
            logger.error(f'Task {self.task_id}: architect failed: {result.output[:200]}')
            return WorkflowOutcome.BLOCKED

        # Read the plan the architect wrote
        self.plan = self.artifacts.read_plan()
        if not self.plan:
            logger.error(f'Task {self.task_id}: architect produced no plan.json')
            return WorkflowOutcome.BLOCKED

        # Check if plan needs modules beyond what we hold
        plan_modules = self.plan.get('modules', [])
        if set(plan_modules) != set(self.modules):
            expanded = await self.scheduler.handle_blast_radius_expansion(
                self.task_id, self.modules, plan_modules
            )
            if not expanded:
                return WorkflowOutcome.REQUEUED
            self.modules = plan_modules

        # Write plan decisions to memory
        await self._write_decisions_to_memory()

        logger.info(
            f'Task {self.task_id}: plan created with '
            f'{len(self.plan.get("prerequisites", []))} prerequisites, '
            f'{len(self.plan.get("steps", []))} steps'
        )
        return WorkflowOutcome.DONE

    async def _execute_verify_review_loop(self) -> WorkflowOutcome:
        """Execute → Verify → Review loop with retry limits."""
        review_cycle = 0

        while True:
            # EXECUTE
            self.state = WorkflowState.EXECUTE
            exec_outcome = await self._execute_iterations()
            if exec_outcome == WorkflowOutcome.BLOCKED:
                return await self._mark_blocked('Execution iterations exhausted')

            # VERIFY + DEBUGFIX loop
            self.state = WorkflowState.VERIFY
            verify_outcome = await self._verify_debugfix_loop()
            if verify_outcome == WorkflowOutcome.BLOCKED:
                return await self._mark_blocked('Verification attempts exhausted')

            # REVIEW
            self.state = WorkflowState.REVIEW
            reviews = await self._review()
            if not reviews.has_blocking_issues:
                # Write suggestions to memory
                await self._write_suggestions_to_memory(reviews)
                return WorkflowOutcome.DONE

            review_cycle += 1
            if review_cycle >= self.config.max_review_cycles:
                return await self._mark_blocked(
                    f'Review cycles exhausted ({review_cycle}). '
                    f'Blocking issues: {len(reviews.blocking_issues)}'
                )

            # Re-plan based on review feedback
            logger.info(
                f'Task {self.task_id}: review cycle {review_cycle}, '
                f'{len(reviews.blocking_issues)} blocking issues'
            )
            await self._replan(reviews)
            self.metrics.review_cycles += 1

    async def _execute_iterations(self) -> WorkflowOutcome:
        """Run implementer iterations until plan is complete."""
        assert self.worktree is not None and self.artifacts is not None
        while self.artifacts.get_pending_steps():
            if self.metrics.execute_iterations >= self.config.max_execute_iterations:
                return WorkflowOutcome.BLOCKED

            self.plan = self.artifacts.read_plan()
            iteration_log = self.artifacts.read_iteration_log()

            prompt = await self.briefing.build_implementer_prompt(self.plan, iteration_log)
            result = await self._invoke(IMPLEMENTER, prompt, self.worktree)

            self.metrics.execute_iterations += 1

            # Re-read plan to see progress
            self.plan = self.artifacts.read_plan()

            if not result.success:
                logger.warning(
                    f'Task {self.task_id}: implementer iteration '
                    f'{self.metrics.execute_iterations} failed'
                )

        return WorkflowOutcome.DONE

    async def _verify_debugfix_loop(self) -> WorkflowOutcome:
        """Run verification, invoke debugger on failures."""
        assert self.worktree is not None and self.artifacts is not None
        verify_attempt = 0

        while True:
            result = await run_verification(self.worktree, self.config)
            if result.passed:
                return WorkflowOutcome.DONE

            verify_attempt += 1
            if verify_attempt >= self.config.max_verify_attempts:
                return WorkflowOutcome.BLOCKED

            self.metrics.verify_attempts += 1
            logger.info(
                f'Task {self.task_id}: verify attempt {verify_attempt} failed: {result.summary}'
            )

            # Invoke debugger
            self.plan = self.artifacts.read_plan()
            prompt = await self.briefing.build_debugger_prompt(
                result.failure_report(), self.plan
            )
            debug_result = await self._invoke(DEBUGGER, prompt, self.worktree)

            if not debug_result.success:
                logger.warning(f'Task {self.task_id}: debugger failed')

    async def _review(self):
        """Run all 5 reviewers in parallel, aggregate results."""
        assert self.worktree is not None and self.artifacts is not None
        diff = await self.git_ops.get_diff_from_main(self.worktree)

        # Launch all reviewers concurrently
        review_tasks = []
        for reviewer_role in ALL_REVIEWERS:
            review_tasks.append(self._run_reviewer(reviewer_role, diff))

        results = await asyncio.gather(*review_tasks, return_exceptions=True)

        # Collect successful reviews
        for role, result in zip(ALL_REVIEWERS, results, strict=True):
            if isinstance(result, Exception):
                logger.error(f'Reviewer {role.name} failed: {result}')
                continue
            if isinstance(result, dict):
                self.artifacts.write_review(role.name, result)

        return self.artifacts.aggregate_reviews()

    async def _run_reviewer(self, role: AgentRole, diff: str) -> dict:
        """Run a single reviewer and parse its JSON output."""
        assert self.worktree is not None
        prompt = await self.briefing.build_reviewer_prompt(role.name, diff)

        # Use structured output for reviewers
        review_schema = {
            'type': 'object',
            'properties': {
                'reviewer': {'type': 'string'},
                'verdict': {'type': 'string', 'enum': ['PASS', 'ISSUES_FOUND']},
                'issues': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'severity': {'type': 'string', 'enum': ['blocking', 'suggestion']},
                            'location': {'type': 'string'},
                            'category': {'type': 'string'},
                            'description': {'type': 'string'},
                            'suggested_fix': {'type': 'string'},
                        },
                        'required': ['severity', 'location', 'category', 'description'],
                    },
                },
                'summary': {'type': 'string'},
            },
            'required': ['reviewer', 'verdict', 'issues', 'summary'],
        }

        result = await self._invoke(
            role, prompt, self.worktree, output_schema=review_schema
        )

        if result.structured_output:
            return result.structured_output

        # Try parsing output as JSON
        try:
            return json.loads(result.output)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f'Reviewer {role.name} did not produce valid JSON')
            return {
                'reviewer': role.name,
                'verdict': 'PASS',
                'issues': [],
                'summary': f'Reviewer output could not be parsed: {result.output[:200]}',
            }

    async def _replan(self, reviews) -> None:
        """Feed review feedback back to architect for re-planning."""
        assert self.worktree is not None and self.artifacts is not None
        feedback = reviews.format_for_replan()
        self.plan = self.artifacts.read_plan()

        prompt = f"""\
The implementation was reviewed and blocking issues were found.

{feedback}

# Current Plan

```json
{json.dumps(self.plan, indent=2)}
```

# Action

Update the plan to address the blocking issues. You may add new steps to the `steps` array, but do NOT remove or reorder existing steps. Set new steps to status "pending". Write the updated plan to `.task/plan.json`.
"""
        await self._invoke(ARCHITECT, prompt, self.worktree)
        self.plan = self.artifacts.read_plan()

    async def _merge(self, branch_name: str) -> WorkflowOutcome:
        """Merge task branch into main."""
        assert self.worktree is not None
        merge_result = await self.git_ops.merge_to_main(self.worktree, branch_name)

        if merge_result.success:
            return WorkflowOutcome.DONE

        if not merge_result.conflicts:
            return await self._mark_blocked(f'Merge failed: {merge_result.details}')

        # Try to resolve conflicts with merger agent
        logger.info(f'Task {self.task_id}: merge conflicts detected, invoking merger')
        task_intent = f"Task: {self.task.get('title', '')}\n{self.task.get('description', '')}"
        prompt = await self.briefing.build_merger_prompt(
            merge_result.details, task_intent
        )
        merger_result = await self._invoke(MERGER, prompt, self.config.project_root)

        if merger_result.success and 'BLOCKED' not in merger_result.output.upper():
            # Verify the merge resolution
            verify = await run_verification(self.config.project_root, self.config)
            if verify.passed:
                return WorkflowOutcome.DONE
            else:
                await self.git_ops.abort_merge(self.config.project_root)
                return await self._mark_blocked(
                    f'Merge conflict resolution failed verification: {verify.summary}'
                )

        await self.git_ops.abort_merge(self.config.project_root)
        return await self._mark_blocked(
            f'Merger could not resolve conflicts: {merger_result.output[:200]}'
        )

    async def _invoke(
        self,
        role: AgentRole,
        prompt: str,
        cwd: Path,
        output_schema: dict | None = None,
    ) -> AgentResult:
        """Invoke an agent with role-specific configuration."""
        # Get role-specific config overrides
        models = self.config.models
        budgets = self.config.budgets
        turns = self.config.max_turns

        model = getattr(models, role.name.split('_')[0], role.default_model)
        budget = getattr(budgets, role.name.split('_')[0], role.default_budget)
        max_turns_val = getattr(turns, role.name.split('_')[0], role.default_max_turns)

        # Use reviewer config for all reviewer variants
        if role.name.startswith('reviewer'):
            model = models.reviewer
            budget = budgets.reviewer
            max_turns_val = turns.reviewer

        result = await invoke_agent(
            prompt=prompt,
            system_prompt=role.system_prompt,
            cwd=cwd,
            model=model,
            max_turns=max_turns_val,
            max_budget_usd=budget,
            allowed_tools=role.allowed_tools or None,
            disallowed_tools=role.disallowed_tools or None,
            mcp_config=self.mcp.mcp_config_json(),
            output_schema=output_schema,
        )

        # Track metrics
        self.metrics.total_cost_usd += result.cost_usd
        self.metrics.total_duration_ms += result.duration_ms
        self.metrics.agent_invocations += 1

        logger.info(
            f'Task {self.task_id} [{role.name}]: '
            f'success={result.success} cost=${result.cost_usd:.2f} '
            f'turns={result.turns}'
        )

        return result

    async def _mark_blocked(self, reason: str) -> WorkflowOutcome:
        """Mark task as blocked."""
        self.state = WorkflowState.BLOCKED
        await self.scheduler.set_task_status(self.task_id, 'blocked')
        logger.warning(f'Task {self.task_id} BLOCKED: {reason}')
        return WorkflowOutcome.BLOCKED

    async def _write_decisions_to_memory(self) -> None:
        """Write plan design decisions to fused-memory."""
        decisions = self.plan.get('design_decisions', [])
        if not decisions:
            return
        try:
            async with __import__('httpx').AsyncClient() as client:
                for decision in decisions:
                    await client.post(
                        f'{self.mcp.url}/mcp/',
                        json={
                            'jsonrpc': '2.0',
                            'id': 1,
                            'method': 'tools/call',
                            'params': {
                                'name': 'add_memory',
                                'arguments': {
                                    'content': f"Decision: {decision['decision']}\nRationale: {decision['rationale']}",
                                    'category': 'decisions_and_rationale',
                                    'project_id': self.config.fused_memory.project_id,
                                    'agent_id': f'orchestrator-task-{self.task_id}',
                                },
                            },
                        },
                        timeout=10,
                    )
        except Exception as e:
            logger.warning(f'Failed to write decisions to memory: {e}')

    async def _write_suggestions_to_memory(self, reviews) -> None:
        """Write review suggestions (non-blocking) to memory as conventions."""
        suggestions = reviews.suggestions
        if not suggestions:
            return
        try:
            import httpx as httpx_mod
            async with httpx_mod.AsyncClient() as client:
                for suggestion in suggestions[:5]:  # cap at 5 to avoid noise
                    await client.post(
                        f'{self.mcp.url}/mcp/',
                        json={
                            'jsonrpc': '2.0',
                            'id': 1,
                            'method': 'tools/call',
                            'params': {
                                'name': 'add_memory',
                                'arguments': {
                                    'content': f"[{suggestion.get('category', '')}] {suggestion.get('description', '')}",
                                    'category': 'preferences_and_norms',
                                    'project_id': self.config.fused_memory.project_id,
                                    'agent_id': f'orchestrator-task-{self.task_id}',
                                },
                            },
                        },
                        timeout=10,
                    )
        except Exception as e:
            logger.warning(f'Failed to write suggestions to memory: {e}')
