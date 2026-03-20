"""Per-task workflow state machine: PLAN → EXECUTE → VERIFY → REVIEW → MERGE → DONE."""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from orchestrator.agents.invoke import AgentResult, invoke_with_cap_retry
from orchestrator.agents.roles import (
    ALL_REVIEWERS,
    ARCHITECT,
    DEBUGGER,
    IMPLEMENTER,
    MERGER,
    AgentRole,
)
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment, files_to_modules
from orchestrator.usage_gate import SessionBudgetExhausted as _SessionBudgetExhausted
from orchestrator.verify import run_scoped_verification

if TYPE_CHECKING:
    from orchestrator.usage_gate import UsageGate


# ---------------------------------------------------------------------------
# Structural protocols — allow test doubles without inheriting concrete classes
# ---------------------------------------------------------------------------


class _SchedulerLike(Protocol):
    async def set_task_status(self, task_id: str, status: str, /) -> None: ...
    async def handle_blast_radius_expansion(
        self, task_id: str, current: list[str], needed: list[str], /
    ) -> bool: ...


class _McpLike(Protocol):
    @property
    def url(self) -> str: ...
    def mcp_config_json(self, escalation_url: str | None = None) -> dict: ...


class _BriefingLike(Protocol):
    async def build_architect_prompt(
        self, task: dict, worktree: Path | None = ..., context: str | None = ...
    ) -> str: ...
    async def build_resume_prompt(
        self,
        task: dict,
        plan: dict,
        escalation_summary: str,
        resolution: str,
        worktree: Path | None = ...,
    ) -> str: ...
    async def build_implementer_prompt(
        self, plan: dict, iteration_log: list, context: str | None = ...
    ) -> str: ...
    async def build_debugger_prompt(
        self, failures: str, plan: dict, context: str | None = ...
    ) -> str: ...
    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = ...
    ) -> str: ...
    async def build_merger_prompt(
        self, conflicts: str, task_intent: str, context: str | None = ...
    ) -> str: ...

logger = logging.getLogger(__name__)


class WorkflowState(enum.Enum):
    PLAN = 'plan'
    EXECUTE = 'execute'
    VERIFY = 'verify'
    REVIEW = 'review'
    MERGE = 'merge'
    DONE = 'done'
    BLOCKED = 'blocked'
    ESCALATED = 'escalated'


class WorkflowOutcome(enum.Enum):
    DONE = 'done'
    BLOCKED = 'blocked'
    REQUEUED = 'requeued'
    ESCALATED = 'escalated'


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
        scheduler: _SchedulerLike,
        briefing: _BriefingLike,
        mcp: _McpLike,
        escalation_queue=None,
        escalation_event: asyncio.Event | None = None,
        usage_gate: UsageGate | None = None,
        initial_plan: dict | None = None,
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
        self._worktree_external = False  # True when worktree was pre-created (eval mode)
        self.artifacts: TaskArtifacts | None = None
        self.plan: dict = {}
        self.initial_plan = initial_plan
        self.metrics = WorkflowMetrics()

        # Per-module configs for scoped verification
        self._module_configs = self._resolve_module_configs()

        # Escalation support
        self.escalation_queue = escalation_queue
        self._escalation_event = escalation_event

        # Usage cap gate
        self.usage_gate = usage_gate

        # Unique session identifier for plan ownership (format: {task_id}-{uuid_hex[:8]})
        self.session_id = f'{self.task_id}-{uuid.uuid4().hex[:8]}'

    @property
    def _task_files(self) -> list[str] | None:
        """Return the file list from the current plan, or None if unavailable/empty."""
        files = self.plan.get('files', [])
        return files if files else None

    async def run(self) -> WorkflowOutcome:
        """Execute the full state machine."""
        branch_name = self.task_id
        try:
            # Set task in-progress
            await self.scheduler.set_task_status(self.task_id, 'in-progress')

            # Create worktree (captures base commit for stable diffs)
            # If worktree is already set (e.g. eval mode), skip creation
            if self.worktree is None:
                self.worktree, base_commit = await self.git_ops.create_worktree(branch_name)
            else:
                # Eval mode: worktree was pre-created, skip creation and cleanup
                self._worktree_external = True
                proc = await asyncio.create_subprocess_exec(
                    'git', 'rev-parse', 'HEAD',
                    cwd=str(self.worktree),
                    stdout=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                base_commit = stdout.decode().strip()
            # Sync per-worktree venvs so imports resolve from worktree source
            if not self._worktree_external:
                await self._sync_worktree_venvs()

            self.artifacts = TaskArtifacts(self.worktree)
            self.artifacts.init(
                self.task_id,
                self.task.get('title', ''),
                self.task.get('description', ''),
                base_commit=base_commit,
            )

            # PLAN (skip if initial_plan was provided — eval mode)
            if self.initial_plan:
                self.artifacts.write_plan(self.initial_plan)
                self.artifacts.stamp_plan_provenance(self.session_id)
                self.artifacts.lock_plan(self.session_id)
                self.plan = self.artifacts.read_plan()
                logger.info(
                    f'Task {self.task_id}: using provided plan '
                    f'({len(self.plan.get("steps", []))} steps)'
                )
            else:
                self.state = WorkflowState.PLAN
                plan_outcome = await self._plan()
                if plan_outcome == WorkflowOutcome.REQUEUED:
                    return WorkflowOutcome.REQUEUED
                if plan_outcome == WorkflowOutcome.BLOCKED:
                    return await self._mark_blocked('Planning failed')

            # EXECUTE + VERIFY + REVIEW loop (with escalation retry)
            while True:
                outcome = await self._execute_verify_review_loop()
                if outcome == WorkflowOutcome.ESCALATED:
                    self.state = WorkflowState.ESCALATED
                    logger.info(f'Task {self.task_id}: waiting for escalation resolution')
                    resolution = await self._wait_for_resolution()
                    # Check if any escalation was dismissed (terminate)
                    assert self.escalation_queue is not None
                    dismissed = [
                        e for e in self.escalation_queue.get_by_task(self.task_id)
                        if e.status == 'dismissed'
                    ]
                    if dismissed:
                        return await self._mark_blocked('Task terminated via escalation')
                    # Resume with resolution context
                    logger.info(f'Task {self.task_id}: resuming after escalation resolution')
                    resume_prompt = await self.briefing.build_resume_prompt(
                        self.task, self.plan,
                        '\n'.join(e.summary for e in self._check_escalations()),
                        resolution, self.worktree,
                    )
                    await self._invoke(IMPLEMENTER, resume_prompt, self.worktree)
                    continue
                if outcome != WorkflowOutcome.DONE:
                    return outcome
                break

            # MERGE (skip for eval mode — no merge into main)
            if not self._worktree_external:
                self.state = WorkflowState.MERGE
                merge_outcome = await self._merge(branch_name)
                if merge_outcome != WorkflowOutcome.DONE:
                    return merge_outcome

                # POST-MERGE VERIFY
                post_merge = await run_scoped_verification(self.config.project_root, self.config, self._module_configs, task_files=self._task_files)
                if not post_merge.passed:
                    logger.error(f'Task {self.task_id}: post-merge verification failed')
                    await self.git_ops.revert_last_merge(self.config.project_root)
                    return await self._mark_blocked(
                        f'Post-merge verification failed: {post_merge.summary}'
                    )

            # SUCCESS — write completion knowledge before status change (ordering guarantee)
            await self._write_completion_to_memory()
            self.state = WorkflowState.DONE
            await self.scheduler.set_task_status(self.task_id, 'done')
            logger.info(
                f'Task {self.task_id} DONE — '
                f'cost=${self.metrics.total_cost_usd:.2f} '
                f'invocations={self.metrics.agent_invocations}'
            )
            return WorkflowOutcome.DONE

        except _SessionBudgetExhausted:
            logger.warning(f'Task {self.task_id}: session budget exhausted')
            return await self._mark_blocked('Session budget exhausted')

        except Exception as e:
            logger.exception(f'Task {self.task_id} workflow error: {e}')
            return await self._mark_blocked(f'Workflow error: {e}')

        finally:
            # Cleanup worktree (only if done — keep for debugging if blocked)
            # Skip cleanup for externally-managed worktrees (eval mode)
            if self.state == WorkflowState.DONE and self.worktree and not self._worktree_external:
                await self.git_ops.cleanup_worktree(self.worktree, branch_name)

    def _resolve_module_configs(self) -> list[ModuleConfig]:
        """Collect ModuleConfigs for this task's modules.

        Groups modules by subproject prefix and returns one ModuleConfig per
        subproject that has an ``orchestrator.yaml``.  Warns for subprojects
        without configs.  Returns an empty list when no modules are assigned
        (triggers global fallback in ``run_scoped_verification``).
        """
        if not self.modules:
            return []
        seen: dict[str, ModuleConfig] = {}
        missing: set[str] = set()
        for m in self.modules:
            mc = self.config.for_module(m)
            if mc:
                seen[mc.prefix] = mc
            else:
                prefix = m.strip('/').split('/')[0]
                missing.add(prefix)
        if missing:
            logger.warning(
                'Task %s: subprojects without orchestrator.yaml: %s — '
                'these will use global verification config',
                self.task_id, missing,
            )
        return list(seen.values())

    async def _sync_worktree_venvs(self) -> None:
        """Run ``uv sync`` for task subprojects in the worktree.

        Creates per-worktree venvs so Python imports resolve from the
        worktree's source code rather than the main tree's editable installs.
        Local ``[tool.uv.sources]`` dependencies (e.g. ``shared``) are
        pulled in automatically via relative editable paths.
        """
        assert self.worktree is not None

        # Derive unique subproject prefixes from task modules
        prefixes: set[str] = set()
        for m in self.modules:
            prefix = m.strip('/').split('/')[0]
            if (self.worktree / prefix / 'pyproject.toml').exists():
                prefixes.add(prefix)

        if not prefixes:
            return

        worktree = self.worktree  # bind for closure (narrowed to Path)

        # Sync subprojects in parallel
        async def _sync(prefix: str) -> None:
            project_dir = str(worktree / prefix)
            proc = await asyncio.create_subprocess_exec(
                'uv', 'sync', '--project', project_dir,
                cwd=str(self.worktree),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(
                    'Task %s: uv sync failed for %s: %s',
                    self.task_id, prefix, stdout.decode()[:500],
                )
            else:
                logger.info('Task %s: synced venv for %s', self.task_id, prefix)

        await asyncio.gather(*(_sync(p) for p in sorted(prefixes)))

    async def _plan(self) -> WorkflowOutcome:
        """Invoke the architect to produce a plan."""
        assert self.worktree is not None and self.artifacts is not None

        # Defense-in-depth: if plan.lock already exists, this is a duplicate workflow.
        # Return REQUEUED so run() exits immediately — the original lock-holder retains
        # ownership. Do NOT re-stamp provenance (that would hijack the first workflow's
        # session_id and cause its validate_plan_owner() checks to fail).
        if self.artifacts.is_plan_locked() and self.artifacts.read_plan():
            lock_data = self.artifacts.read_plan_lock()
            lock_owner = lock_data.get('session_id', 'unknown') if lock_data else 'unknown'
            logger.info(
                f'Task {self.task_id}: plan.lock is held by session {lock_owner!r}, '
                f'skipping architect — requeuing to avoid duplicate execution'
            )
            return WorkflowOutcome.REQUEUED

        prompt = await self.briefing.build_architect_prompt(self.task, worktree=self.worktree)
        result = await self._invoke(ARCHITECT, prompt, self.worktree)

        if not result.success:
            logger.error(f'Task {self.task_id}: architect failed: {result.output[:200]}')
            return WorkflowOutcome.BLOCKED

        # Read the plan the architect wrote
        self.plan = self.artifacts.read_plan()

        if not self.plan:
            logger.error(f'Task {self.task_id}: architect produced no plan.json')
            return WorkflowOutcome.BLOCKED

        # Stamp provenance and acquire lock
        self.artifacts.stamp_plan_provenance(self.session_id)
        self.artifacts.lock_plan(self.session_id)
        self.plan = self.artifacts.read_plan()

        # Derive modules from plan's file list (deterministic) or fall back to
        # the plan's module list (heuristic).
        plan_files = self.plan.get('files', [])
        if plan_files:
            plan_modules = files_to_modules(plan_files, self.config.lock_depth)
            logger.info(
                f'Task {self.task_id}: derived {len(plan_modules)} modules '
                f'from {len(plan_files)} files: {plan_modules}'
            )
        else:
            plan_modules = self.plan.get('modules', [])

        if set(plan_modules) != set(self.modules):
            expanded = await self.scheduler.handle_blast_radius_expansion(
                self.task_id, self.modules, plan_modules
            )
            if not expanded:
                return WorkflowOutcome.REQUEUED
            self.modules = plan_modules
            self._module_configs = self._resolve_module_configs()

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
            if exec_outcome == WorkflowOutcome.ESCALATED:
                return WorkflowOutcome.ESCALATED
            if exec_outcome == WorkflowOutcome.BLOCKED:
                return await self._mark_blocked('Execution iterations exhausted')

            # VERIFY + DEBUGFIX loop
            self.state = WorkflowState.VERIFY
            verify_outcome = await self._verify_debugfix_loop()
            if verify_outcome == WorkflowOutcome.ESCALATED:
                return WorkflowOutcome.ESCALATED
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
            # Re-stamp provenance — architect may have overwritten plan.json
            assert self.artifacts is not None
            self.artifacts.stamp_plan_provenance(self.session_id)
            self.plan = self.artifacts.read_plan()
            self.metrics.review_cycles += 1

    async def _execute_iterations(self) -> WorkflowOutcome:
        """Run implementer iterations until plan is complete."""
        assert self.worktree is not None and self.artifacts is not None
        while self.artifacts.get_pending_steps():
            if self.metrics.execute_iterations >= self.config.max_execute_iterations:
                return WorkflowOutcome.BLOCKED

            # Validate plan ownership before each implementer invocation
            if not self.artifacts.validate_plan_owner(self.session_id):
                logger.error(
                    f'Task {self.task_id}: plan.json ownership mismatch — '
                    f'expected session {self.session_id}, plan has different _session_id'
                )
                self._escalate_plan_overwrite()
                return WorkflowOutcome.BLOCKED

            self.plan = self.artifacts.read_plan()
            iteration_log, corrupted = self.artifacts.read_iteration_log()
            if corrupted:
                self._escalate_corruption(corrupted)

            # Snapshot completed steps before invocation
            completed_before = {
                s['id']
                for col in ('prerequisites', 'steps')
                for s in self.plan.get(col, [])
                if s.get('status') == 'done'
            }

            prompt = await self.briefing.build_implementer_prompt(self.plan, iteration_log)
            result = await self._invoke(IMPLEMENTER, prompt, self.worktree)

            self.metrics.execute_iterations += 1

            # Check for escalations
            blocking = [e for e in self._check_escalations() if e.severity == 'blocking']
            if blocking:
                return WorkflowOutcome.ESCALATED

            # Re-read plan to see progress
            self.plan = self.artifacts.read_plan()

            # Compute newly completed steps and write iteration log
            completed_after = {
                s['id']
                for col in ('prerequisites', 'steps')
                for s in self.plan.get(col, [])
                if s.get('status') == 'done'
            }
            newly_completed = sorted(completed_after - completed_before)
            head_commit = await self._get_head_commit()

            if newly_completed:
                step_descs = [
                    s.get('description', s['id'])
                    for col in ('prerequisites', 'steps')
                    for s in self.plan.get(col, [])
                    if s['id'] in newly_completed
                ]
                summary = '; '.join(step_descs)
            else:
                summary = 'No new steps completed'

            self.artifacts.append_iteration_log({
                'iteration': self.metrics.execute_iterations,
                'agent': 'implementer',
                'steps_attempted': newly_completed,
                'steps_completed': newly_completed,
                'commit': head_commit,
                'summary': summary,
                'source': 'orchestrator',
            })

            # Validate plan ownership after implementer (detect post-write tamper)
            if not self.artifacts.validate_plan_owner(self.session_id):
                logger.error(
                    f'Task {self.task_id}: plan.json ownership mismatch after implementer — '
                    f'expected session {self.session_id}'
                )
                self._escalate_plan_overwrite()
                return WorkflowOutcome.BLOCKED

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
            result = await run_scoped_verification(self.worktree, self.config, self._module_configs, task_files=self._task_files)
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

            # Write debugger iteration log entry
            head_commit = await self._get_head_commit()
            self.artifacts.append_iteration_log({
                'iteration': verify_attempt,
                'agent': 'debugger',
                'steps_attempted': [],
                'steps_completed': [],
                'commit': head_commit,
                'summary': f'Debug fix for: {result.summary[:100]}',
                'source': 'orchestrator',
            })

            # Check for escalations from debugger
            blocking = [e for e in self._check_escalations() if e.severity == 'blocking']
            if blocking:
                return WorkflowOutcome.ESCALATED

            if not debug_result.success:
                logger.warning(f'Task {self.task_id}: debugger failed')

    async def _review(self):
        """Run all 5 reviewers in parallel, aggregate results."""
        assert self.worktree is not None and self.artifacts is not None
        base_commit = self.artifacts.read_base_commit()
        if base_commit:
            diff = await self.git_ops.get_diff_from_base(self.worktree, base_commit)
        else:
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
            verify = await run_scoped_verification(self.config.project_root, self.config, self._module_configs, task_files=self._task_files)
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

    def _select_model_for_role(self, role: AgentRole, base_model: str) -> str:
        """Override model for implementer/debugger based on task complexity."""
        if role.name not in ('implementer', 'debugger'):
            return base_model

        # Check for Rust modules (crates/ prefix is the convention)
        rust_modules = [m for m in self.modules if m.startswith('crates/')]
        if len(rust_modules) < 3:
            return base_model

        # Check step count if plan is available (always true for implementer/debugger)
        if self.plan:
            step_count = len(self.plan.get('steps', []))
            if step_count >= 15:
                logger.info(
                    'Task %s: upgrading %s to opus (%d Rust modules, %d steps)',
                    self.task_id, role.name, len(rust_modules), step_count,
                )
                return 'opus'

        return base_model

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
        effort_cfg = self.config.effort
        backends_cfg = self.config.backends

        role_key = role.name.split('_')[0]

        model = getattr(models, role_key, role.default_model)
        model = self._select_model_for_role(role, model)
        budget = getattr(budgets, role_key, role.default_budget)
        max_turns_val = getattr(turns, role_key, role.default_max_turns)
        effort_val = getattr(effort_cfg, role_key, 'high')
        backend_val = getattr(backends_cfg, role_key, 'claude')

        # Use reviewer config for all reviewer variants
        if role.name.startswith('reviewer'):
            model = models.reviewer
            budget = budgets.reviewer
            max_turns_val = turns.reviewer
            effort_val = effort_cfg.reviewer
            backend_val = backends_cfg.reviewer

        # Determine sandbox modules based on role
        sandbox_modules = None
        if self.config.sandbox.enabled and role.name in ('implementer', 'debugger'):
            sandbox_modules = self.modules

        # Build MCP config with escalation server if available
        mcp_config = None
        if self.escalation_queue and role.name in ('architect', 'implementer', 'debugger', 'merger'):
            esc = self.config.escalation
            mcp_config = self.mcp.mcp_config_json(
                escalation_url=f'http://{esc.host}:{esc.port}/mcp'
            )

        result = await invoke_with_cap_retry(
            usage_gate=self.usage_gate,
            label=f'Task {self.task_id} [{role.name}]',
            prompt=prompt,
            system_prompt=role.system_prompt,
            cwd=cwd,
            model=model,
            max_turns=max_turns_val,
            max_budget_usd=budget,
            allowed_tools=role.allowed_tools or None,
            disallowed_tools=role.disallowed_tools or None,
            mcp_config=mcp_config,
            output_schema=output_schema,
            sandbox_modules=sandbox_modules,
            effort=effort_val,
            backend=backend_val,
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

    def _check_escalations(self):
        """Check for pending escalations for this task."""
        if not self.escalation_queue:
            return []
        return self.escalation_queue.get_by_task(self.task_id, status='pending')

    async def _wait_for_resolution(self) -> str:
        """Wait for all pending escalations to be resolved."""
        if self._escalation_event is None:
            self._escalation_event = asyncio.Event()

        # Poll until all pending escalations for this task are resolved
        while True:
            pending = self._check_escalations()
            if not pending:
                break
            self._escalation_event.clear()
            await self._escalation_event.wait()

        # Collect resolutions
        assert self.escalation_queue is not None
        resolved = [
            e for e in self.escalation_queue.get_by_task(self.task_id)
            if e.status == 'resolved' and e.resolution
        ]
        return '\n'.join(e.resolution for e in resolved)

    async def _get_head_commit(self) -> str:
        """Return the HEAD commit SHA for the current worktree."""
        proc = await asyncio.create_subprocess_exec(
            'git', 'rev-parse', 'HEAD',
            cwd=str(self.worktree),
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip()

    def _escalate_plan_overwrite(self) -> None:
        """Submit a blocking escalation when plan.json is overwritten by a foreign session."""
        summary = f'plan.json overwrite detected for task {self.task_id}'
        detail = (
            f'Expected _session_id={self.session_id} but plan.json contains a different value. '
            f'A duplicate workflow may have overwritten plan.json.'
        )
        logger.error(f'Task {self.task_id}: {summary}')

        if not self.escalation_queue:
            return

        from escalation.models import Escalation

        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=summary,
            detail=detail,
            suggested_action='investigate_and_retry',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)

    def _escalate_corruption(self, corrupted: list[str]) -> None:
        """Submit an info-severity escalation for corrupted iteration log lines."""
        if not self.escalation_queue:
            logger.warning(
                'Task %s: %d corrupted iteration log lines (no escalation queue)',
                self.task_id, len(corrupted),
            )
            return

        from escalation.models import Escalation

        detail = f'{len(corrupted)} corrupted line(s):\n' + '\n'.join(
            line[:200] for line in corrupted[:10]
        )
        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='info',
            category='infra_issue',
            summary=f'{len(corrupted)} corrupted iteration log line(s)',
            detail=detail,
            suggested_action='investigate_log_corruption',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)

    async def _mark_blocked(self, reason: str) -> WorkflowOutcome:
        """Mark task as blocked and create an escalation entry."""
        if self.state == WorkflowState.DONE:
            logger.warning(
                'Task %s: already DONE, ignoring late blocked transition: %s',
                self.task_id, reason,
            )
            return WorkflowOutcome.DONE
        self.state = WorkflowState.BLOCKED
        await self.scheduler.set_task_status(self.task_id, 'blocked')
        logger.warning(f'Task {self.task_id} BLOCKED: {reason}')

        if self.escalation_queue:
            from escalation.models import Escalation

            esc = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category='task_failure',
                summary=reason[:200],
                detail=reason,
                suggested_action='investigate_and_retry',
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            self.escalation_queue.submit(esc)

        return WorkflowOutcome.BLOCKED

    async def _write_completion_to_memory(self) -> None:
        """Write task completion summary so dependent tasks find it in briefings."""
        parts = [f"Completed: {self.task.get('title', '')}"]

        desc = self.task.get('description', '')
        if desc:
            parts.append(f"Description: {desc}")

        analysis = self.plan.get('analysis', '')
        if analysis:
            parts.append(f"Analysis: {analysis}")

        decisions = self.plan.get('design_decisions', [])
        if decisions:
            decision_text = '; '.join(
                d.get('decision', '') for d in decisions[:3]
            )
            parts.append(f"Key decisions: {decision_text}")

        steps = self.plan.get('steps', [])
        done_count = sum(1 for s in steps if s.get('status') == 'done')
        parts.append(f"Steps completed: {done_count}/{len(steps)}")

        if self.modules:
            parts.append(f"Modules: {', '.join(self.modules)}")

        content = '\n'.join(parts)

        try:
            import httpx as httpx_mod
            async with httpx_mod.AsyncClient() as client:
                await client.post(
                    f'{self.mcp.url}/mcp/',
                    json={
                        'jsonrpc': '2.0',
                        'id': 1,
                        'method': 'tools/call',
                        'params': {
                            'name': 'add_memory',
                            'arguments': {
                                'content': content,
                                'category': 'observations_and_summaries',
                                'project_id': self.config.fused_memory.project_id,
                                'agent_id': f'orchestrator-task-{self.task_id}',
                            },
                        },
                    },
                    timeout=10,
                )
        except Exception as e:
            logger.warning(f'Failed to write completion to memory: {e}')

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
