"""Top-level orchestration loop."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.mcp_lifecycle import McpLifecycle
from orchestrator.scheduler import Scheduler
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

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
    total_cost_usd: float = 0.0
    task_reports: list[TaskReport] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f'Orchestrator run complete: {self.completed}/{self.total_tasks} tasks done',
            f'  Blocked: {self.blocked}',
            f'  Total cost: ${self.total_cost_usd:.2f}',
            f'  Duration: {self.started_at} → {self.completed_at}',
            '',
            'Per-task results:',
        ]
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

    async def run(self, prd_path: Path, dry_run: bool = False) -> HarnessReport:
        """Execute the full orchestration pipeline."""
        self.report.started_at = datetime.now(UTC).isoformat()

        # 1. Start fused-memory HTTP server
        logger.info('Starting fused-memory HTTP server...')
        await self.mcp.start()

        try:
            # 2. Parse PRD into tasks
            logger.info(f'Parsing PRD: {prd_path}')
            await self._populate_tasks(prd_path)

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
            self.report.total_cost_usd = sum(r.cost_usd for r in task_reports)

        finally:
            # 4. Shutdown
            self.report.completed_at = datetime.now(UTC).isoformat()
            await self.mcp.stop()

        logger.info(self.report.summary())
        return self.report

    async def _populate_tasks(self, prd_path: Path) -> None:
        """Use taskmaster parse_prd to decompose PRD into tasks."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f'{self.mcp.url}/mcp/',
                    json={
                        'jsonrpc': '2.0',
                        'id': 1,
                        'method': 'tools/call',
                        'params': {
                            'name': 'parse_prd',
                            'arguments': {
                                'input': str(prd_path.resolve()),
                                'project_root': str(self.config.project_root),
                            },
                        },
                    },
                    timeout=120,  # PRD parsing can be slow
                )
                result = resp.json()
                logger.info(f'PRD parsed: {result.get("result", {}).get("content", [{}])[0].get("text", "")[:200]}')
        except Exception as e:
            raise RuntimeError(f'Failed to parse PRD: {e}') from e

    async def _run_slot(
        self, assignment, sem: asyncio.Semaphore
    ) -> TaskReport | None:
        """Run a single workflow slot."""
        try:
            logger.info(
                f'Starting workflow for task {assignment.task_id}: '
                f'{assignment.task.get("title", "")}'
            )
            workflow = TaskWorkflow(
                assignment=assignment,
                config=self.config,
                git_ops=self.git_ops,
                scheduler=self.scheduler,
                briefing=self.briefing,
                mcp=self.mcp,
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
            self.scheduler.release(assignment.task_id)
            sem.release()
