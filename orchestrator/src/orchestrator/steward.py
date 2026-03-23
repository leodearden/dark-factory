"""Per-task process steward — autonomous first-line escalation handler.

The steward watches for escalations on a specific task and handles them
by invoking a Claude Code session with the STEWARD role. It runs as an
asyncio.Task alongside the workflow, started by the harness on first
blocking escalation.

Event loop: check pending → run watcher → handle → repeat until done/stopped.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from orchestrator.agents.invoke import invoke_with_cap_retry
from orchestrator.agents.roles import STEWARD

if TYPE_CHECKING:
    from escalation.models import Escalation
    from escalation.queue import EscalationQueue

    from orchestrator.config import OrchestratorConfig
    from orchestrator.mcp_lifecycle import McpLifecycle
    from orchestrator.usage_gate import UsageGate

logger = logging.getLogger(__name__)


@dataclass
class StewardMetrics:
    invocations: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    escalations_handled: int = 0


class TaskSteward:
    """Per-task process steward — handles escalations autonomously."""

    def __init__(
        self,
        task_id: str,
        task: dict,
        worktree: Path,
        config: OrchestratorConfig,
        mcp: McpLifecycle,
        escalation_queue: EscalationQueue,
        briefing,
        usage_gate: UsageGate | None = None,
    ):
        self.task_id = task_id
        self.task = task
        self.worktree = worktree
        self.config = config
        self.mcp = mcp
        self.escalation_queue = escalation_queue
        self.briefing = briefing
        self.usage_gate = usage_gate

        self._task: asyncio.Task | None = None
        self._stopped = False
        self.metrics = StewardMetrics()

    async def start(self) -> None:
        """Start the steward event loop as a background asyncio.Task."""
        self._task = asyncio.create_task(
            self._run_loop(), name=f'steward-{self.task_id}',
        )
        logger.info(f'Steward started for task {self.task_id}')

    async def stop(self) -> None:
        """Cancel the steward loop and cleanup."""
        self._stopped = True
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info(
            f'Steward stopped for task {self.task_id} '
            f'(invocations={self.metrics.invocations}, '
            f'cost=${self.metrics.total_cost_usd:.2f})'
        )

    async def _run_loop(self) -> None:
        """Watch for escalations, handle them, repeat until done or stopped."""
        while not self._stopped:
            try:
                escalation = await self._next_escalation()
                if escalation is None or self._stopped:
                    break

                await self._handle_escalation(escalation)

                if await self._is_task_complete():
                    logger.info(
                        f'Steward for task {self.task_id}: task complete, exiting'
                    )
                    break
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    f'Steward for task {self.task_id}: unhandled error in loop'
                )
                # Don't crash the loop — wait a moment and retry
                await asyncio.sleep(2)

    async def _next_escalation(self) -> Escalation | None:
        """Get the next level-0 pending escalation (agent→steward scope).

        Checks for already-pending escalations before starting the watcher
        to avoid a race between escalation submission and watcher startup.
        Level-1 escalations (steward→human) are ignored — those are for
        the interactive session / /unblock skill.
        """
        # Check for existing pending level-0 escalations first
        pending = self.escalation_queue.get_by_task(
            self.task_id, status='pending', level=0,
        )
        if pending:
            return pending[0]

        # No pending level-0 — block on watcher subprocess
        # The watcher doesn't filter by level, so we re-check after it fires
        esc = await self._watch_for_escalation()
        if esc is not None and esc.level != 0:
            # Watcher picked up a level-1 (steward→human) escalation — ignore it
            return None
        return esc

    async def _watch_for_escalation(self) -> Escalation | None:
        """Run the escalation watcher as a blocking subprocess.

        The watcher uses inotify to detect new escalation files and exits
        after the first matching event, printing the escalation JSON to stdout.
        """
        from escalation.models import Escalation as EscModel

        queue_dir = self.escalation_queue.queue_dir
        cmd = [
            'python', '-m', 'escalation.watcher',
            '--queue-dir', str(queue_dir),
            '--task-id', self.task_id,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.config.project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f'Steward watcher failed for task {self.task_id}: {e}')
            return None

        if proc.returncode != 0 or not stdout.strip():
            if stderr:
                logger.debug(
                    f'Steward watcher stderr: {stderr.decode()[:200]}'
                )
            return None

        try:
            data = json.loads(stdout.decode())
            return EscModel.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f'Steward watcher parse error: {e}')
            return None

    async def _handle_escalation(self, escalation: Escalation) -> None:
        """Invoke the steward agent to handle a single escalation."""
        logger.info(
            f'Steward for task {self.task_id}: handling escalation '
            f'{escalation.id} [{escalation.category}] — {escalation.summary}'
        )

        # Collect all pending escalations for context
        pending = self.escalation_queue.get_by_task(
            self.task_id, status='pending',
        )
        pending_dicts = [e.to_dict() for e in pending]

        # Build the prompt
        prompt = await self.briefing.build_steward_prompt(
            task=self.task,
            escalation=escalation.to_dict(),
            pending_escalations=pending_dicts,
            worktree=self.worktree,
        )

        # Build MCP config — fused-memory + escalation
        esc_cfg = self.config.escalation
        escalation_url = f'http://{esc_cfg.host}:{esc_cfg.port}/mcp'
        mcp_config = self.mcp.mcp_config_json(escalation_url=escalation_url)

        role = STEWARD

        result = await invoke_with_cap_retry(
            usage_gate=self.usage_gate,
            label=f'Steward for task {self.task_id}',
            prompt=prompt,
            system_prompt=role.system_prompt,
            cwd=self.worktree,
            model=self.config.models.steward,
            max_turns=self.config.max_turns.steward,
            max_budget_usd=self.config.budgets.steward,
            allowed_tools=role.allowed_tools or None,
            mcp_config=mcp_config,
            effort=self.config.effort.steward,
            backend=self.config.backends.steward,
        )

        self.metrics.invocations += 1
        self.metrics.total_cost_usd += result.cost_usd
        self.metrics.total_duration_ms += result.duration_ms
        self.metrics.escalations_handled += 1

        logger.info(
            f'Steward for task {self.task_id}: invocation complete '
            f'(success={result.success}, cost=${result.cost_usd:.2f}, '
            f'turns={result.turns})'
        )

        # Check if the escalation was resolved
        still_pending = self.escalation_queue.get_by_task(
            self.task_id, status='pending',
        )
        if still_pending:
            pending_ids = [e.id for e in still_pending]
            logger.warning(
                f'Steward for task {self.task_id}: escalation(s) still pending '
                f'after invocation: {pending_ids}'
            )

    async def _is_task_complete(self) -> bool:
        """Check if the task has reached a terminal state."""
        try:
            from orchestrator.mcp_lifecycle import mcp_call

            result = await mcp_call(
                f'{self.mcp.url}/mcp',
                'tools/call',
                {
                    'name': 'get_task',
                    'arguments': {
                        'id': self.task_id,
                        'project_root': str(self.config.project_root),
                    },
                },
                timeout=10,
            )
            content = result.get('result', {}).get('content', [{}])
            text = content[0].get('text', '') if content else ''
            try:
                task_data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return False

            status = task_data.get('status', '')
            return status in ('done', 'cancelled')
        except Exception as e:
            logger.debug(f'Steward task status check failed: {e}')
            return False
