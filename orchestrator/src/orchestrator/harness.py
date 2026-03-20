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
from orchestrator.scheduler import Scheduler
from orchestrator.usage_gate import UsageGate
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

try:
    from escalation.queue import EscalationQueue
    from escalation.server import create_server
    HAS_ESCALATION = True
except ImportError:
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)


@dataclass
class TaskReport:
    task_id: str
    title: str
    outcome: WorkflowOutcome
    cost_usd: float = 0.0
    duration_ms: int = 0
    agent_invocations: int = 0


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

        # Escalation support
        self._escalation_queue: EscalationQueue | None = None
        self._escalation_events: dict[str, asyncio.Event] = {}
        self._escalation_task: asyncio.Task | None = None

    async def run(self, prd_path: Path, dry_run: bool = False) -> HarnessReport:
        """Execute the full orchestration pipeline."""
        self.report.started_at = datetime.now(UTC).isoformat()

        # 1. Start fused-memory HTTP server
        logger.info('Starting fused-memory HTTP server...')
        await self.mcp.start()

        # 1b. Start escalation server
        await self._start_escalation_server()

        try:
            # 1c. Usage cap startup check
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

            # 2. Parse PRD into tasks
            logger.info(f'Parsing PRD: {prd_path}')
            pre_ids = {str(t.get('id', '')) for t in await self.scheduler.get_tasks()}
            await self._populate_tasks(prd_path)

            # 2a. Tag newly-created tasks with PRD source
            await self._tag_prd_metadata(prd_path, pre_ids)

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
                assignment = await self.scheduler.acquire_next()

                if assignment is None:
                    if not active:
                        break  # all done or all blocked
                    # Wait for any active task to complete, then retry
                    done, active = await asyncio.wait(
                        active, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done:
                        try:
                            report = t.result()
                            if report:
                                task_reports.append(report)
                        except Exception as e:
                            logger.error(f'Workflow slot error: {e}')
                    continue

                await sem.acquire()
                task = asyncio.create_task(
                    self._run_slot(assignment, sem),
                    name=f'workflow-{assignment.task_id}',
                )
                active.add(task)
                task.add_done_callback(active.discard)

            # Drain remaining
            if active:
                done, _ = await asyncio.wait(active)
                for t in done:
                    try:
                        report = t.result()
                        if report:
                            task_reports.append(report)
                    except Exception as e:
                        logger.error(f'Workflow slot error: {e}')

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

        finally:
            # 4. Shutdown
            self.report.completed_at = datetime.now(UTC).isoformat()
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
            logger.info(f'PRD parsed: {text[:200]}')
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
                tagged_count += 1

        logger.info(f'Tagged {tagged_count}/{len(untagged)} tasks with module metadata')

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
            )
            outcome = await workflow.run()

            return TaskReport(
                task_id=assignment.task_id,
                title=assignment.task.get('title', ''),
                outcome=outcome,
                cost_usd=workflow.metrics.total_cost_usd,
                duration_ms=workflow.metrics.total_duration_ms,
                agent_invocations=workflow.metrics.agent_invocations,
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

        mcp_server = create_server(self._escalation_queue)  # type: ignore[possibly-undefined]
        host = self.config.escalation.host
        port = self.config.escalation.port

        async def _serve():
            try:
                await mcp_server.run_http_async(host=host, port=port)
            except Exception as e:
                logger.error(f'Escalation server error: {e}')

        self._escalation_task = asyncio.create_task(_serve(), name='escalation-server')
        logger.info(f'Escalation MCP server starting on {host}:{port}')
        # Give the server a moment to bind
        await asyncio.sleep(0.5)

    async def _stop_escalation_server(self) -> None:
        """Stop the escalation server."""
        if self._escalation_task is not None:
            self._escalation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._escalation_task
            self._escalation_task = None
            logger.info('Escalation server stopped')

    def _on_escalation(self, escalation) -> None:
        """Callback when a blocking escalation is submitted — wake the waiting workflow."""
        if escalation.severity == 'blocking':
            event = self._escalation_events.get(escalation.task_id)
            if event:
                event.set()
