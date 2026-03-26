"""Top-level orchestration loop."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.invoke import invoke_with_cap_retry
from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.mcp_lifecycle import McpLifecycle, mcp_call
from orchestrator.review_checkpoint import ReviewCheckpoint
from orchestrator.scheduler import Scheduler, files_to_modules
from orchestrator.usage_gate import UsageGate
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

try:
    from escalation.queue import EscalationQueue
    from escalation.server import create_server
    HAS_ESCALATION = True
except ImportError:
    HAS_ESCALATION = False

try:
    from orchestrator.steward import TaskSteward
    HAS_STEWARD = True
except ImportError:
    HAS_STEWARD = False

logger = logging.getLogger(__name__)


@dataclass
class TaskReport:
    task_id: str
    title: str
    outcome: WorkflowOutcome
    cost_usd: float = 0.0
    duration_ms: int = 0
    agent_invocations: int = 0
    execute_iterations: int = 0
    verify_attempts: int = 0
    review_cycles: int = 0
    steward_cost_usd: float = 0.0
    steward_invocations: int = 0
    completed_at: str = ''


@dataclass
class HarnessReport:
    started_at: str = ''
    completed_at: str = ''
    total_tasks: int = 0
    completed: int = 0
    blocked: int = 0
    escalated: int = 0
    total_cost_usd: float = 0.0
    task_reports: list[TaskReport] = field(default_factory=list)
    paused_for_cap: bool = False
    cap_pause_duration_secs: float = 0.0
    review_checkpoints: int = 0
    review_findings: int = 0
    review_tasks_created: int = 0
    review_cost_usd: float = 0.0

    def summary(self) -> str:
        lines = [
            f'Orchestrator run complete: {self.completed}/{self.total_tasks} tasks done',
            f'  Blocked: {self.blocked}',
            f'  Escalated: {self.escalated}',
            f'  Total cost: ${self.total_cost_usd:.2f}',
            f'  Duration: {self.started_at} → {self.completed_at}',
        ]
        if self.paused_for_cap:
            lines.append(f'  Cap pause: {self.cap_pause_duration_secs:.0f}s total')
        if self.review_checkpoints > 0:
            lines.append(
                f'  Review checkpoints: {self.review_checkpoints} '
                f'({self.review_findings} findings, '
                f'{self.review_tasks_created} tasks, '
                f'${self.review_cost_usd:.2f})'
            )
        lines.extend([
            '',
            'Per-task results:',
        ])
        for r in self.task_reports:
            lines.append(
                f'  {r.task_id}: {r.outcome.value} '
                f'(${r.cost_usd:.2f}, {r.agent_invocations} invocations)'
            )
        return '\n'.join(lines)


class Harness:
    """Top-level orchestration loop."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.mcp = McpLifecycle(config)
        self.git_ops = GitOps(config.git, config.project_root)
        self.scheduler = Scheduler(config)
        self.briefing = BriefingAssembler(config)
        self.report = HarnessReport()
        self._recovered_plans: dict[str, dict] = {}

        # Usage cap gate
        self.usage_gate: UsageGate | None = (
            UsageGate(config.usage_cap) if config.usage_cap.enabled else None
        )

        # Review checkpoints
        self.review_checkpoint: ReviewCheckpoint | None = (
            ReviewCheckpoint(config, self.mcp, self.usage_gate)
            if config.review.enabled else None
        )
        self._review_running = False
        self._pending_review_task: asyncio.Task | None = None
        self._task_modules: dict[str, list[str]] = {}  # task_id -> modules

        # Escalation support
        self._escalation_queue: EscalationQueue | None = None
        self._escalation_events: dict[str, asyncio.Event] = {}
        self._escalation_task: asyncio.Task | None = None

    async def run(self, prd_path: Path | None = None, dry_run: bool = False) -> HarnessReport:
        """Execute the full orchestration pipeline.

        If *prd_path* is ``None``, skip PRD parsing and run existing tasks.
        """
        self.report.started_at = datetime.now(UTC).isoformat()

        # 1. Start fused-memory HTTP server
        logger.info('Starting fused-memory HTTP server...')
        await self.mcp.start()

        # 1b. Start escalation server
        await self._start_escalation_server()

        try:
            # 1c. Dismiss stale escalations from prior runs (non-fatal)
            try:
                await self._dismiss_stale_escalations()
            except Exception as e:
                logger.warning(f'Failed to dismiss stale escalations: {e}')

            # 1d. Usage cap startup check
            if self.usage_gate:
                logger.info('Checking usage cap status...')
                await self.usage_gate.check_at_startup()
                if self.usage_gate.is_paused:
                    logger.warning(f'Usage cap already hit: {self.usage_gate.paused_reason}')
                    if not self.config.usage_cap.wait_for_reset:
                        raise RuntimeError(
                            f'Usage cap hit at startup: {self.usage_gate.paused_reason}'
                        )
                    # wait_for_reset=True: probe loop is already running,
                    # workflows will block in before_invoke until gate reopens

            # 2. Parse PRD into tasks (skipped when no PRD given)
            if prd_path is not None:
                logger.info(f'Parsing PRD: {prd_path}')
                pre_ids = {str(t.get('id', '')) for t in await self.scheduler.get_tasks()}
                await self._populate_tasks(prd_path)

                # 2a. Tag newly-created tasks with PRD source
                await self._tag_prd_metadata(prd_path, pre_ids)
            else:
                logger.info('No PRD given — running existing tasks')
                existing = await self.scheduler.get_tasks()
                if not any(t.get('status') == 'pending' for t in existing):
                    raise RuntimeError(
                        'No PRD given and no pending tasks found. '
                        'Pass --prd to decompose a PRD, or create tasks first.'
                    )

            # 2b. Tag tasks with code modules for concurrency locking
            logger.info('Tagging tasks with code modules...')
            await self._tag_task_modules()

            # 2c. Recover crashed tasks from surviving worktrees
            await self._recover_crashed_tasks()

            tasks = await self.scheduler.get_tasks()
            self.report.total_tasks = len([t for t in tasks if t.get('status') == 'pending'])
            logger.info(f'Task tree populated: {self.report.total_tasks} pending tasks')

            if dry_run:
                logger.info('Dry run — stopping after task population')
                return self.report

            # 3. Run workflow slots
            sem = asyncio.Semaphore(self.config.max_concurrent_tasks)
            active: set[asyncio.Task] = set()
            task_reports: list[TaskReport] = []

            while True:
                # Pick up any pending review task spawned by _collect_done_reports
                if self._pending_review_task is not None:
                    active.add(self._pending_review_task)
                    self._pending_review_task = None

                # If a review checkpoint is running, don't acquire new tasks —
                # wait for in-flight tasks (and the review) to complete.
                if self._review_running:
                    if not active:
                        break  # shouldn't happen — review task is in active
                    done, active = await asyncio.wait(
                        active, return_when=asyncio.FIRST_COMPLETED
                    )
                    self._collect_done_reports(done, task_reports)
                    continue

                assignment = await self.scheduler.acquire_next()

                if assignment is None:
                    if not active:
                        break  # all done or all blocked
                    # Wait for any active task to complete, then retry
                    done, active = await asyncio.wait(
                        active, return_when=asyncio.FIRST_COMPLETED
                    )
                    self._collect_done_reports(done, task_reports)
                    continue

                await sem.acquire()
                self._task_modules[assignment.task_id] = list(assignment.modules)
                task = asyncio.create_task(
                    self._run_slot(assignment, sem),
                    name=f'workflow-{assignment.task_id}',
                )
                active.add(task)
                task.add_done_callback(active.discard)

            # Drain remaining
            if active:
                done, _ = await asyncio.wait(active)
                self._collect_done_reports(done, task_reports)

            self.report.task_reports = task_reports
            self.report.completed = sum(
                1 for r in task_reports if r.outcome == WorkflowOutcome.DONE
            )
            self.report.blocked = sum(
                1 for r in task_reports if r.outcome == WorkflowOutcome.BLOCKED
            )
            self.report.escalated = sum(
                1 for r in task_reports if r.outcome == WorkflowOutcome.ESCALATED
            )
            self.report.total_cost_usd = sum(r.cost_usd for r in task_reports)

            # 3b. Optional full review after all tasks complete
            if (self.review_checkpoint
                    and self.config.review.full_review_on_complete
                    and self.report.completed > 0):
                logger.info('Running full post-completion review...')
                try:
                    review_report = await self.review_checkpoint.run_full()
                    self.report.review_checkpoints += 1
                    self.report.review_findings += review_report.findings_count
                    self.report.review_tasks_created += len(review_report.tasks_created)
                    self.report.review_cost_usd += review_report.cost_usd
                    logger.info(
                        'Full review complete: %d findings, %d tasks created',
                        review_report.findings_count,
                        len(review_report.tasks_created),
                    )
                except Exception as e:
                    logger.error(f'Full review failed: {e}')

            # Commit accumulated task status changes so they survive
            # working-tree resets and are visible to future merge worktrees.
            if self.report.completed > 0:
                sha = await self.git_ops.commit_task_statuses()
                if sha:
                    logger.info(f'Committed task statuses: {sha[:8]}')

        finally:
            # 4. Shutdown
            self.report.completed_at = datetime.now(UTC).isoformat()

            # Persist run metrics to SQLite
            try:
                from orchestrator.run_store import RunStore

                db_path = self.config.project_root / 'data' / 'orchestrator' / 'runs.db'
                store = RunStore(db_path)
                store.save_run(
                    self.report,
                    self.config.fused_memory.project_id,
                    str(prd_path) if prd_path else '',
                )
            except Exception as e:
                logger.warning(f'Failed to persist run metrics: {e}')

            # Save HarnessReport alongside review checkpoint reports
            if self.report.review_checkpoints > 0:
                self._save_harness_report()

            if self.usage_gate:
                if self.usage_gate.total_pause_secs > 0:
                    self.report.paused_for_cap = True
                    self.report.cap_pause_duration_secs = self.usage_gate.total_pause_secs
                await self.usage_gate.shutdown()
            await self._stop_escalation_server()
            await self.mcp.stop()

        logger.info(self.report.summary())
        return self.report

    async def _populate_tasks(self, prd_path: Path) -> None:
        """Use taskmaster parse_prd to decompose PRD into tasks."""
        try:
            result = await mcp_call(
                f'{self.mcp.url}/mcp',
                'tools/call',
                {
                    'name': 'parse_prd',
                    'arguments': {
                        'input': str(prd_path.resolve()),
                        'project_root': str(self.config.project_root),
                    },
                },
                timeout=120,  # PRD parsing can be slow
            )
            content = result.get('result', {}).get('content', [{}])
            text = content[0].get('text', '') if content else ''
            if content and content[0].get('isError'):
                raise RuntimeError(f'parse_prd error: {text}')
            logger.info(f'PRD parsed: {text[:200]}')
        except RuntimeError as e:
            if 'already contains' in str(e):
                logger.info('Task tree already populated — skipping parse_prd')
            else:
                raise
        except Exception as e:
            raise RuntimeError(f'Failed to parse PRD: {e}') from e

    async def _tag_prd_metadata(self, prd_path: Path, pre_parse_ids: set[str]) -> None:
        """Tag tasks with the PRD they were created from."""
        resolved_prd = str(prd_path.resolve())
        tasks = await self.scheduler.get_tasks()
        new_ids = {str(t.get('id', '')) for t in tasks} - pre_parse_ids

        tagged = 0
        for t in tasks:
            tid = str(t.get('id', ''))
            if not tid:
                continue
            metadata = t.get('metadata') or {}
            if metadata.get('prd'):
                continue  # already tagged
            if new_ids and tid not in new_ids:
                continue  # existed before parse_prd
            await self.scheduler.update_task(tid, {'prd': resolved_prd})
            tagged += 1

        if tagged:
            logger.info(f'Tagged {tagged} tasks with PRD: {resolved_prd}')

    async def _tag_task_modules(self) -> None:
        """Invoke a Claude agent to tag each task with the code modules it touches.

        Uses structured output to get a JSON mapping of task_id → [modules],
        then persists via scheduler.update_task().
        """
        tasks = await self.scheduler.get_tasks()

        # Filter to pending tasks that don't already have modules in metadata
        # (done/cancelled tasks don't need module tags — they won't be executed)
        untagged = []
        for t in tasks:
            if t.get('status') not in ('pending', 'in-progress'):
                continue
            metadata = t.get('metadata') or {}
            modules = metadata.get('modules', [])
            if not modules:
                untagged.append(t)

        if not untagged:
            logger.info('All tasks already have module tags — skipping')
            return

        # Get top-level directory listing for context
        try:
            entries = sorted(p.name for p in self.config.project_root.iterdir() if p.is_dir() and not p.name.startswith('.'))
        except OSError:
            entries = []

        task_summaries = []
        for t in untagged:
            task_summaries.append({
                'id': str(t.get('id', '')),
                'title': t.get('title', ''),
                'description': t.get('description', ''),
            })

        schema = {
            'type': 'object',
            'properties': {
                'tasks': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'string'},
                            'files': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'Predicted file paths this task will create or modify',
                            },
                            'modules': {'type': 'array', 'items': {'type': 'string'}},
                        },
                        'required': ['id', 'files', 'modules'],
                    },
                },
            },
            'required': ['tasks'],
        }

        prompt = f"""\
Given these tasks and this codebase structure, predict which files each task
will create or modify, and which code modules (directories) it will touch.

Be specific and exhaustive with file predictions — include source files AND
test files. Use paths relative to the project root. The `files` field is used
to derive concurrency locks, so accuracy prevents unnecessary serialization.

The `modules` field should list directory-level groupings (e.g. "src/backends",
"src/server", "tests") as a human-readable summary.

# Codebase top-level directories
{json.dumps(entries)}

# Tasks to tag
{json.dumps(task_summaries, indent=2)}

Output JSON matching the schema. Every task must appear in the output.
"""

        result = await invoke_with_cap_retry(
            usage_gate=self.usage_gate,
            label='Module tagging',
            prompt=prompt,
            system_prompt='You are a code module classifier. Given task descriptions and a codebase structure, determine which code modules each task will modify. Be precise and conservative.',
            cwd=self.config.project_root,
            model=self.config.models.module_tagger,
            max_turns=self.config.max_turns.module_tagger,
            max_budget_usd=self.config.budgets.module_tagger,
            output_schema=schema,
        )

        if not result.success:
            logger.warning(f'Module tagger agent failed: {result.output[:200]}')
            return

        # Parse the structured output
        mapping = result.structured_output
        if not mapping:
            try:
                mapping = json.loads(result.output)
            except (json.JSONDecodeError, TypeError):
                logger.warning('Module tagger produced no parseable output')
                return

        tagged_count = 0
        for entry in mapping.get('tasks', []):
            task_id = str(entry.get('id', ''))
            modules = entry.get('modules', [])
            files = entry.get('files', [])
            if task_id and (modules or files):
                metadata: dict = {}
                if files:
                    metadata['files'] = files
                if modules:
                    metadata['modules'] = modules
                await self.scheduler.update_task(task_id, json.dumps(metadata))
                # Also populate in-memory cache (update_task metadata persistence
                # is broken — taskmaster lacks an update_task tool)
                depth = self.config.lock_depth
                if files:
                    derived = files_to_modules(files, depth)
                    if derived:
                        self.scheduler._module_cache[task_id] = derived
                elif modules:
                    from orchestrator.scheduler import normalize_lock
                    self.scheduler._module_cache[task_id] = [
                        normalize_lock(m, depth) for m in modules
                    ]
                tagged_count += 1

        logger.info(f'Tagged {tagged_count}/{len(untagged)} tasks with module metadata')
        logger.info(f'Module cache has {len(self.scheduler._module_cache)} entries')
        if self.scheduler._module_cache:
            sample = dict(list(self.scheduler._module_cache.items())[:3])
            logger.info(f'Module cache sample: {sample}')

    async def _recover_crashed_tasks(self) -> None:
        """Scan surviving worktrees and recover plans with completed work.

        For each worktree in the worktree base directory:
        - If it has a plan.json with completed steps, store the plan for
          injection into the resumed workflow.
        - Otherwise, clean up the worktree (no useful work to recover).

        Also resets any in-progress tasks to pending so acquire_next() picks
        them up.
        """
        worktree_base = self.git_ops.worktree_base
        if not worktree_base.exists():
            return

        recovered = 0
        cleaned = 0

        for entry in worktree_base.iterdir():
            if not entry.is_dir():
                continue
            task_id = entry.name
            plan_path = entry / '.task' / 'plan.json'

            if not plan_path.exists():
                logger.info(
                    f'Recovery: worktree {task_id} has no plan — cleaning up'
                )
                await self.git_ops.cleanup_worktree(entry, task_id)
                cleaned += 1
                continue

            try:
                plan = json.loads(plan_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f'Recovery: worktree {task_id} has corrupt plan — '
                    f'cleaning up ({e})'
                )
                await self.git_ops.cleanup_worktree(entry, task_id)
                cleaned += 1
                continue

            # Check if plan has any completed steps
            completed = [
                s for col in ('prerequisites', 'steps')
                for s in plan.get(col, [])
                if s.get('status') == 'done'
            ]

            if not completed:
                logger.info(
                    f'Recovery: worktree {task_id} has plan but no '
                    f'completed steps — cleaning up'
                )
                await self.git_ops.cleanup_worktree(entry, task_id)
                cleaned += 1
                continue

            total = sum(len(plan.get(col, [])) for col in ('prerequisites', 'steps'))
            logger.info(
                f'Recovery: worktree {task_id} has plan with '
                f'{len(completed)}/{total} steps done — storing for resumption'
            )
            self._recovered_plans[task_id] = plan
            recovered += 1

        # Reset in-progress tasks to pending
        tasks = await self.scheduler.get_tasks()
        reset_count = 0
        for t in tasks:
            if t.get('status') == 'in-progress':
                tid = str(t.get('id', ''))
                await self.scheduler.set_task_status(tid, 'pending')
                logger.info(f'Recovery: reset task {tid} from in-progress to pending')
                reset_count += 1

        if recovered or cleaned or reset_count:
            logger.info(
                f'Crash recovery: {recovered} plans recovered, '
                f'{cleaned} worktrees cleaned, {reset_count} tasks reset'
            )

    async def _run_slot(
        self, assignment, sem: asyncio.Semaphore
    ) -> TaskReport | None:
        """Run a single workflow slot."""
        try:
            logger.info(
                f'Starting workflow for task {assignment.task_id}: '
                f'{assignment.task.get("title", "")}'
            )
            # Create escalation event for this task
            esc_event = None
            if self._escalation_queue:
                esc_event = asyncio.Event()
                self._escalation_events[assignment.task_id] = esc_event

            recovered_plan = self._recovered_plans.pop(assignment.task_id, None)

            # Build steward factory — steward starts when the workflow
            # creates its worktree (it needs the path).
            steward_factory = None
            if HAS_STEWARD and self._escalation_queue:
                esc_q = self._escalation_queue  # capture for closure (narrows type)

                def _make_steward(worktree: Path, *, _assign=assignment) -> TaskSteward:  # type: ignore[name-defined]
                    return TaskSteward(
                        task_id=_assign.task_id,
                        task=_assign.task,
                        worktree=worktree,
                        config=self.config,
                        mcp=self.mcp,
                        escalation_queue=esc_q,
                        briefing=self.briefing,
                        usage_gate=self.usage_gate,
                    )
                steward_factory = _make_steward

            workflow = TaskWorkflow(
                assignment=assignment,
                config=self.config,
                git_ops=self.git_ops,
                scheduler=self.scheduler,
                briefing=self.briefing,
                mcp=self.mcp,
                escalation_queue=self._escalation_queue,
                escalation_event=esc_event,
                usage_gate=self.usage_gate,
                initial_plan=recovered_plan,
                steward_factory=steward_factory,
            )
            outcome = await workflow.run()

            steward_cost = 0.0
            steward_invocations = 0
            steward = workflow._steward
            if steward and hasattr(steward, 'metrics'):
                steward_cost = steward.metrics.total_cost_usd
                steward_invocations = steward.metrics.invocations

            return TaskReport(
                task_id=assignment.task_id,
                title=assignment.task.get('title', ''),
                outcome=outcome,
                cost_usd=workflow.metrics.total_cost_usd,
                duration_ms=workflow.metrics.total_duration_ms,
                agent_invocations=workflow.metrics.agent_invocations,
                execute_iterations=workflow.metrics.execute_iterations,
                verify_attempts=workflow.metrics.verify_attempts,
                review_cycles=workflow.metrics.review_cycles,
                steward_cost_usd=steward_cost,
                steward_invocations=steward_invocations,
                completed_at=datetime.now(UTC).isoformat(),
            )
        except Exception as e:
            logger.exception(f'Workflow slot error for task {assignment.task_id}: {e}')
            return TaskReport(
                task_id=assignment.task_id,
                title=assignment.task.get('title', ''),
                outcome=WorkflowOutcome.BLOCKED,
            )
        finally:
            self._escalation_events.pop(assignment.task_id, None)
            self.scheduler.release(assignment.task_id)
            sem.release()

    def _collect_done_reports(
        self, done: set[asyncio.Task], task_reports: list[TaskReport]
    ) -> None:
        """Extract TaskReports from completed asyncio.Tasks and track merges for review."""
        for t in done:
            # Handle review checkpoint completion
            if t.get_name() == 'review-checkpoint':
                self._review_running = False
                try:
                    t.result()  # propagate exceptions
                except Exception as e:
                    logger.error(f'Review checkpoint error: {e}')
                continue

            try:
                report = t.result()
                if report:
                    task_reports.append(report)
                    # Track module merges for review checkpoints
                    if (report.outcome == WorkflowOutcome.DONE
                            and self.review_checkpoint):
                        modules = self._task_modules.pop(report.task_id, [])
                        self.review_checkpoint.record_merge(modules)
                        # Check if review should trigger
                        if (self.review_checkpoint.should_trigger()
                                and not self._review_running):
                            self._trigger_review_checkpoint()
            except Exception as e:
                logger.error(f'Workflow slot error: {e}')

    def _trigger_review_checkpoint(self) -> None:
        """Spawn a review checkpoint as a concurrent task.

        The main loop will pause task acquisition while the review runs
        (``_review_running`` flag) but in-flight tasks continue in their
        worktrees.
        """
        self._review_running = True
        # We can't add to active here — the caller's done set is immutable.
        # Instead, the review task is spawned and tracked; the main loop checks
        # _review_running and waits on active (which includes this task after
        # the next iteration adds it).

        # Actually, we need the task in `active` for the main loop's asyncio.wait.
        # The caller iterates `done`, so we store the task on self for the loop
        # to pick up.
        self._pending_review_task = asyncio.create_task(
            self._run_review_checkpoint(), name='review-checkpoint',
        )

    async def _run_review_checkpoint(self) -> None:
        """Execute a focused review checkpoint."""
        assert self.review_checkpoint is not None
        logger.info('Starting review checkpoint...')
        try:
            review_report = await self.review_checkpoint.run_focused()
            self.report.review_checkpoints += 1
            self.report.review_findings += review_report.findings_count
            self.report.review_tasks_created += len(review_report.tasks_created)
            self.report.review_cost_usd += review_report.cost_usd
            logger.info(
                'Review checkpoint complete: %d findings, %d tasks created, '
                'cost=$%.2f',
                review_report.findings_count,
                len(review_report.tasks_created),
                review_report.cost_usd,
            )
        except Exception as e:
            logger.error(f'Review checkpoint failed: {e}')

    def _save_harness_report(self) -> None:
        """Persist HarnessReport as JSON alongside review checkpoint reports."""
        reports_dir = self.config.project_root / self.config.review.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        ts = self.report.started_at.replace(':', '').replace('-', '')[:15]
        path = reports_dir / f'harness-{ts}.json'

        data = {
            'started_at': self.report.started_at,
            'completed_at': self.report.completed_at,
            'total_tasks': self.report.total_tasks,
            'completed': self.report.completed,
            'blocked': self.report.blocked,
            'escalated': self.report.escalated,
            'total_cost_usd': self.report.total_cost_usd,
            'paused_for_cap': self.report.paused_for_cap,
            'cap_pause_duration_secs': self.report.cap_pause_duration_secs,
            'review_checkpoints': self.report.review_checkpoints,
            'review_findings': self.report.review_findings,
            'review_tasks_created': self.report.review_tasks_created,
            'review_cost_usd': self.report.review_cost_usd,
            'task_reports': [
                {
                    'task_id': r.task_id,
                    'title': r.title,
                    'outcome': r.outcome.value,
                    'cost_usd': r.cost_usd,
                    'duration_ms': r.duration_ms,
                    'agent_invocations': r.agent_invocations,
                    'completed_at': r.completed_at,
                }
                for r in self.report.task_reports
            ],
        }

        try:
            path.write_text(json.dumps(data, indent=2))
            logger.info('HarnessReport saved: %s', path)
        except OSError as e:
            logger.warning('Failed to save HarnessReport: %s', e)

    async def _start_escalation_server(self) -> None:
        """Start the escalation MCP server as a background asyncio task."""
        if not HAS_ESCALATION:
            logger.info('Escalation package not installed — skipping escalation server')
            return

        queue_dir = Path(self.config.escalation.queue_dir)
        if not queue_dir.is_absolute():
            queue_dir = self.config.project_root / queue_dir
        self._escalation_queue = EscalationQueue(queue_dir)  # type: ignore[possibly-undefined]
        self._escalation_queue.set_notify_callback(self._on_escalation)
        self._escalation_queue.set_resolve_callback(self._on_escalation_resolved)

        mcp_server = create_server(self._escalation_queue)  # type: ignore[possibly-undefined]
        host = self.config.escalation.host
        port = self.config.escalation.port

        async def _serve():
            try:
                import uvicorn
                app = mcp_server.http_app()
                uv_config = uvicorn.Config(
                    app, host=host, port=port, log_level='warning',
                )
                server = uvicorn.Server(uv_config)
                await server.serve()
            except Exception as e:
                logger.error(f'Escalation server error: {e}')

        self._escalation_task = asyncio.create_task(_serve(), name='escalation-server')
        logger.info(f'Escalation MCP server starting on {host}:{port}')
        # Give the server a moment to bind
        await asyncio.sleep(0.5)

    async def _dismiss_stale_escalations(self) -> None:
        """Dismiss all pending escalations left over from prior orchestrator runs.

        Called right after _start_escalation_server() so that any escalations
        persisted in the queue directory from a previous (crashed or completed)
        run are cleared before the new run begins.  All pending escalations at
        startup are by definition stale — a new run should never inherit
        unresolved escalations from a prior one.
        """
        if self._escalation_queue is None:
            return

        resolution = (
            'Auto-dismissed: orchestrator restarted — stale from prior run'
        )
        count = self._escalation_queue.dismiss_all_pending(resolution)
        if count:
            logger.info(f'Dismissed {count} stale escalation(s) from prior run')

    async def _stop_escalation_server(self) -> None:
        """Stop the escalation server."""
        if self._escalation_task is not None:
            self._escalation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._escalation_task
            self._escalation_task = None
            logger.info('Escalation server stopped')

    def _on_escalation(self, escalation) -> None:
        """Callback when any escalation is submitted — wake the waiting workflow/steward."""
        event = self._escalation_events.get(escalation.task_id)
        if event:
            event.set()

    def _on_escalation_resolved(self, escalation) -> None:
        """Callback when an escalation is resolved — wake the waiting workflow."""
        event = self._escalation_events.get(escalation.task_id)
        if event:
            event.set()
