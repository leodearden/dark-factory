"""Per-task persistent steward — autonomous escalation handler.

The steward maintains a single Claude session across all escalations for a task,
accumulating context via ``--resume``.  It handles both blocking escalations
(code fixes) and review suggestion triage (create tasks / write conventions /
dismiss).

Lifecycle:
- Started lazily on first escalation (not at workflow start).
- Each escalation either resumes the existing session or creates a fresh one.
- Budget-capped at $12 lifetime; auto-re-escalates to level-1 on exhaustion.
- Retries each escalation up to 3 times before re-escalating to level-1.
- Stopped by the workflow after task completion + grace period.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from orchestrator.agents.invoke import invoke_agent
from orchestrator.agents.roles import STEWARD

if TYPE_CHECKING:
    from escalation.models import Escalation
    from escalation.queue import EscalationQueue
    from shared.cost_store import CostStore

    from orchestrator.agents.briefing import BriefingAssembler
    from orchestrator.config import OrchestratorConfig
    from orchestrator.mcp_lifecycle import McpLifecycle
    from orchestrator.usage_gate import UsageGate

logger = logging.getLogger(__name__)

_CAP_HIT_COOLDOWN_SECS = 5.0


@dataclass
class StewardMetrics:
    invocations: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    escalations_handled: int = 0
    escalations_reescalated: int = 0


class TaskSteward:
    """Per-task persistent steward — handles escalations via a resumable session."""

    def __init__(
        self,
        task_id: str,
        task: dict,
        worktree: Path,
        config: OrchestratorConfig,
        mcp: McpLifecycle,
        escalation_queue: EscalationQueue,
        briefing: BriefingAssembler,
        usage_gate: UsageGate | None = None,
        cost_store: CostStore | None = None,
        run_id: str = '',
        project_id: str = '',
    ):
        self.task_id = task_id
        self.task = task
        self.worktree = worktree
        self.config = config
        self.mcp = mcp
        self.escalation_queue = escalation_queue
        self.briefing = briefing
        self.usage_gate = usage_gate

        # Cost tracking — optional; when None, cost recording is silently skipped
        self._cost_store = cost_store
        self._run_id = run_id
        self._project_id = project_id

        self._session_id: str | None = None
        self._stopped = False
        self._task: asyncio.Task | None = None
        self._retry_counts: dict[str, int] = {}
        self.metrics = StewardMetrics()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Watch for escalations, handle them, repeat until stopped."""
        while not self._stopped:
            try:
                escalation = await self._next_escalation()
                if self._stopped:
                    break
                if escalation is None:
                    # Transient failure (watcher crash, non-level-0 event,
                    # parse error).  Brief backoff before retrying.
                    await asyncio.sleep(1)
                    continue
                await self._handle_escalation(escalation)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    f'Steward for task {self.task_id}: unhandled error in loop'
                )
                await asyncio.sleep(2)

    # ------------------------------------------------------------------
    # Escalation retrieval
    # ------------------------------------------------------------------

    async def _next_escalation(self) -> Escalation | None:
        """Get the next level-0 pending escalation.

        Checks for already-pending escalations before starting the watcher
        to avoid a race between escalation submission and watcher startup.
        """
        pending = self.escalation_queue.get_by_task(
            self.task_id, status='pending', level=0,
        )
        if pending:
            return pending[0]

        esc = await self._watch_for_escalation()
        if esc is not None and esc.level != 0:
            return None
        return esc

    async def _watch_for_escalation(self) -> Escalation | None:
        """Block on the inotify watcher subprocess until an escalation arrives."""
        from escalation.models import Escalation as EscModel

        queue_dir = self.escalation_queue.queue_dir
        cmd = [
            'python', '-m', 'escalation.watcher',
            '--queue-dir', str(queue_dir),
            '--task-id', self.task_id,
            '--level', '0',
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
                logger.warning(
                    f'Steward watcher stderr for task {self.task_id}: '
                    f'{stderr.decode()[:200]}'
                )
            return None

        try:
            data = json.loads(stdout.decode())
            return EscModel.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f'Steward watcher parse error: {e}')
            return None

    # ------------------------------------------------------------------
    # Escalation handling
    # ------------------------------------------------------------------

    async def _handle_escalation(self, escalation: Escalation) -> None:
        """Handle a single escalation via the persistent session."""
        logger.info(
            f'Steward for task {self.task_id}: handling escalation '
            f'{escalation.id} [{escalation.category}] — {escalation.summary}'
        )

        # Guard: lifetime budget exhausted
        if self.metrics.total_cost_usd >= self.config.steward_lifetime_budget:
            self._auto_escalate_to_human(
                escalation, 'Steward lifetime budget exhausted '
                f'(${self.metrics.total_cost_usd:.2f} / '
                f'${self.config.steward_lifetime_budget:.2f})',
            )
            return

        # Guard: per-escalation retry limit
        retry_count = self._retry_counts.get(escalation.id, 0)
        if retry_count >= self.config.steward_max_retries:
            self._auto_escalate_to_human(
                escalation,
                f'Failed after {retry_count} attempts: {escalation.summary}',
            )
            return

        # CWD: suggestions read from main (post-merge), others from worktree
        if escalation.category == 'review_suggestions':
            cwd = self.config.project_root
        else:
            cwd = self.worktree

        # Pre-triage large suggestion sets before invoking the steward session
        if escalation.category == 'review_suggestions' and escalation.detail:
            try:
                suggestions = json.loads(escalation.detail)
            except (json.JSONDecodeError, TypeError):
                suggestions = []
            if len(suggestions) >= self.config.suggestion_triage_threshold:
                logger.info(
                    f'Steward for task {self.task_id}: pre-triaging '
                    f'{len(suggestions)} suggestions'
                )
                escalation = await self._pre_triage_suggestions(escalation)

        # Prompt: initial (full briefing) vs continuation (just the escalation)
        is_initial = self._session_id is None
        if is_initial:
            pending_dicts = [
                e.to_dict()
                for e in self.escalation_queue.get_by_task(
                    self.task_id, status='pending',
                )
            ]
            prompt = await self.briefing.build_steward_initial_prompt(
                task=self.task,
                escalation=escalation.to_dict(),
                pending_escalations=pending_dicts,
                worktree=self.worktree,
            )
        else:
            prompt = await self.briefing.build_steward_continuation_prompt(
                task=self.task,
                escalation=escalation.to_dict(),
            )

        # MCP config
        esc_cfg = self.config.escalation
        escalation_url = f'http://{esc_cfg.host}:{esc_cfg.port}/mcp'
        mcp_config = self.mcp.mcp_config_json(escalation_url=escalation_url)

        # Per-invocation budget: capped by lifetime remaining
        remaining = self.config.steward_lifetime_budget - self.metrics.total_cost_usd
        per_invocation = min(self.config.budgets.steward, remaining)

        # Invoke with session persistence and cap-hit recovery
        result = await self._invoke_with_session(
            prompt=prompt,
            cwd=cwd,
            mcp_config=mcp_config,
            per_invocation_budget=per_invocation,
            escalation=escalation,
        )

        # Track metrics
        self.metrics.invocations += 1
        self.metrics.total_cost_usd += result.cost_usd
        self.metrics.total_duration_ms += result.duration_ms

        logger.info(
            f'Steward for task {self.task_id}: invocation complete '
            f'(success={result.success}, cost=${result.cost_usd:.2f}, '
            f'turns={result.turns})'
        )

        # Patch resolution metadata
        self._patch_resolution_metadata(escalation.id, result)

        # Check if escalation was actually resolved
        updated = self.escalation_queue.get(escalation.id)
        if updated and updated.status == 'pending':
            self._retry_counts[escalation.id] = retry_count + 1
            logger.warning(
                f'Steward for task {self.task_id}: escalation {escalation.id} '
                f'still pending (attempt {retry_count + 1}/'
                f'{self.config.steward_max_retries})'
            )
        else:
            self.metrics.escalations_handled += 1

    # ------------------------------------------------------------------
    # Session-aware invocation with cap-hit recovery
    # ------------------------------------------------------------------

    async def _invoke_with_session(
        self,
        prompt: str,
        cwd: Path,
        mcp_config: dict,
        per_invocation_budget: float,
        escalation: Escalation,
    ):
        """Invoke the steward agent, resuming the session if one exists.

        On usage-cap hit: resets the session and rebuilds the full initial
        prompt (context is lost across account switches).
        """
        from shared.cli_invoke import AgentResult

        while True:
            oauth_token = None
            account_name = ''
            if self.usage_gate:
                oauth_token = await self.usage_gate.before_invoke()
                account_name = self.usage_gate.active_account_name or ''

            kwargs: dict = dict(
                prompt=prompt,
                system_prompt=STEWARD.system_prompt,
                cwd=cwd,
                model=self.config.models.steward,
                max_turns=self.config.max_turns.steward,
                max_budget_usd=per_invocation_budget,
                allowed_tools=STEWARD.allowed_tools or None,
                mcp_config=mcp_config,
                effort=self.config.effort.steward,
                backend=self.config.backends.steward,
                oauth_token=oauth_token,
            )

            if self._session_id is not None:
                kwargs['resume_session_id'] = self._session_id

            started_at = datetime.now(UTC).isoformat()
            result: AgentResult = await invoke_agent(**kwargs)
            completed_at = datetime.now(UTC).isoformat()

            # Cap-hit: sleep, then reset session (can't resume across account switch)
            if self.usage_gate and self.usage_gate.detect_cap_hit(
                result.stderr, result.output, 'claude', oauth_token=oauth_token,
            ):
                logger.warning(
                    f'Steward for task {self.task_id}: cap hit, sleeping '
                    f'{_CAP_HIT_COOLDOWN_SECS}s before retry',
                )
                await asyncio.sleep(_CAP_HIT_COOLDOWN_SECS)
                self._session_id = None
                # Rebuild full initial prompt (context lost)
                pending_dicts = [
                    e.to_dict()
                    for e in self.escalation_queue.get_by_task(
                        self.task_id, status='pending',
                    )
                ]
                prompt = await self.briefing.build_steward_initial_prompt(
                    task=self.task,
                    escalation=escalation.to_dict(),
                    pending_escalations=pending_dicts,
                    worktree=self.worktree,
                )
                continue

            if self.usage_gate:
                self.usage_gate.on_agent_complete(result.cost_usd)

            # Capture session ID for subsequent --resume calls
            if result.session_id:
                self._session_id = result.session_id

            result.account_name = account_name

            # Record cost to CostStore if available
            if self._cost_store is not None:
                try:
                    await self._cost_store.save_invocation(
                        run_id=self._run_id,
                        task_id=self.task_id,
                        project_id=self._project_id,
                        account_name=account_name,
                        model=self.config.models.steward,
                        role='steward',
                        cost_usd=result.cost_usd,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        cache_read_tokens=result.cache_read_tokens,
                        cache_create_tokens=result.cache_create_tokens,
                        duration_ms=result.duration_ms,
                        capped=False,
                        started_at=started_at,
                        completed_at=completed_at,
                    )
                except Exception:
                    logger.warning('Failed to save steward invocation cost', exc_info=True)

            return result

    # ------------------------------------------------------------------
    # Pre-triage for large suggestion sets
    # ------------------------------------------------------------------

    async def _pre_triage_suggestions(
        self, escalation: Escalation,
    ) -> Escalation:
        """Run a lightweight triage agent on large suggestion sets.

        Returns the escalation with its detail field replaced by pre-triaged
        markdown.  If triage fails, returns the original escalation unchanged
        so the steward falls back to inline triage.
        """
        from orchestrator.agents.triage import (
            TRIAGE_OUTPUT_SCHEMA,
            TRIAGE_SYSTEM_PROMPT,
            build_triage_prompt,
            format_pretriaged_detail,
            parse_triage_result,
        )

        suggestions = json.loads(escalation.detail)
        prompt = build_triage_prompt(suggestions, self.task)

        oauth_token = None
        if self.usage_gate:
            oauth_token = await self.usage_gate.before_invoke()

        result = await invoke_agent(
            prompt=prompt,
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            cwd=self.config.project_root,
            model=self.config.models.triage,
            max_turns=self.config.max_turns.triage,
            max_budget_usd=self.config.budgets.triage,
            allowed_tools=['Read', 'Glob', 'Grep'],
            output_schema=TRIAGE_OUTPUT_SCHEMA,
            effort=self.config.effort.triage,
            backend=self.config.backends.triage,
            oauth_token=oauth_token,
        )

        # Track cost against steward metrics
        self.metrics.invocations += 1
        self.metrics.total_cost_usd += result.cost_usd
        self.metrics.total_duration_ms += result.duration_ms

        if self.usage_gate:
            self.usage_gate.on_agent_complete(result.cost_usd)

        triage_result = parse_triage_result(result)
        if triage_result is None:
            logger.warning(
                f'Steward for task {self.task_id}: pre-triage failed, '
                f'falling back to inline triage'
            )
            return escalation

        new_detail = format_pretriaged_detail(triage_result, suggestions)

        # Return a modified copy — don't mutate the queue's version
        from escalation.models import Escalation as EscModel

        return EscModel(
            **{
                **escalation.to_dict(),
                'detail': new_detail,
                'summary': (
                    f'{len(suggestions)} suggestions pre-triaged: '
                    f'{len(triage_result["accepted"])} accepted, '
                    f'{len(triage_result["skipped"])} skipped'
                ),
            },
        )

    # ------------------------------------------------------------------
    # Auto-escalation to level-1 (human)
    # ------------------------------------------------------------------

    def _auto_escalate_to_human(
        self, escalation: Escalation, reason: str,
    ) -> None:
        """Re-escalate to level-1 and dismiss the original level-0."""
        from escalation.models import Escalation as EscModel

        reesc = EscModel(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='steward',
            severity=escalation.severity,
            category=escalation.category,
            summary=f'Steward re-escalation: {reason}',
            detail=(
                f'Original escalation: {escalation.summary}\n\n'
                f'{escalation.detail}\n\n'
                f'Steward reason for re-escalation: {reason}'
            ),
            suggested_action='manual_intervention',
            worktree=str(self.worktree),
            level=1,
        )
        self.escalation_queue.submit(reesc)

        self.escalation_queue.resolve(
            escalation.id,
            f'Auto-dismissed: re-escalated to level 1 — {reason}',
            dismiss=True,
            resolved_by='steward',
        )

        self.metrics.escalations_reescalated += 1
        logger.warning(
            f'Steward for task {self.task_id}: re-escalated {escalation.id} '
            f'to level 1: {reason}'
        )

    # ------------------------------------------------------------------
    # Metadata patching
    # ------------------------------------------------------------------

    def _patch_resolution_metadata(self, escalation_id: str, result) -> None:
        """Post-fill resolved_by and resolution_turns if the agent resolved it."""
        updated = self.escalation_queue.get(escalation_id)
        if (
            updated
            and updated.status in ('resolved', 'dismissed')
            and not updated.resolved_by
        ):
            updated.resolved_by = 'steward'
            updated.resolution_turns = result.turns
            try:
                self.escalation_queue._rewrite(escalation_id, updated)  # noqa: SLF001
            except Exception as e:
                logger.warning(
                    f'Failed to patch steward metadata on {escalation_id}: {e}'
                )
