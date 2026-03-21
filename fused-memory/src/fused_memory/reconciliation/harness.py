"""Pipeline orchestrator — runs the three-stage reconciliation pipeline."""

from __future__ import annotations

import asyncio
import logging
import traceback
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from shared.usage_gate import UsageGate

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.reconciliation import (
    ReconciliationRun,
    StageId,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.judge import Judge
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)
from fused_memory.services.memory_service import MemoryService

if TYPE_CHECKING:
    from escalation.models import Escalation
    from escalation.queue import EscalationQueue

try:
    from escalation.models import Escalation  # noqa: F811
    from escalation.queue import EscalationQueue  # noqa: F811
    from escalation.server import create_server as create_escalation_server
    HAS_ESCALATION = True
except ImportError:
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)


@dataclass
class TierConfig:
    """Model tier configuration for a reconciliation cycle."""

    model: str = 'sonnet'
    episode_limit: int = 125
    memory_limit: int = 250


class ReconciliationHarness:
    """Orchestrates the three-stage reconciliation pipeline."""

    def __init__(
        self,
        memory_service: MemoryService,
        taskmaster: TaskmasterBackend | None,
        journal: ReconciliationJournal,
        event_buffer: EventBuffer,
        config: FusedMemoryConfig,
    ):
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.journal = journal
        self.buffer = event_buffer
        self.config = config.reconciliation

        # Usage gate (multi-account cap failover)
        self.usage_gate: UsageGate | None = None
        if hasattr(self.config, 'usage_cap') and self.config.usage_cap.enabled:
            self.usage_gate = UsageGate(self.config.usage_cap)

        # Build stages
        stage1 = MemoryConsolidator(
            StageId.memory_consolidator, memory_service, taskmaster, journal, self.config,
            usage_gate=self.usage_gate,
        )

        stage2 = TaskKnowledgeSync(
            StageId.task_knowledge_sync, memory_service, taskmaster, journal, self.config,
            usage_gate=self.usage_gate,
        )

        stage3 = IntegrityCheck(
            StageId.integrity_check, memory_service, taskmaster, journal, self.config,
            usage_gate=self.usage_gate,
        )

        self.stages = [stage1, stage2, stage3]

        # Judge
        self.judge = (
            Judge(self.config, journal, usage_gate=self.usage_gate)
            if self.config.judge_enabled else None
        )

        # Escalation support
        self._escalation_queue: EscalationQueue | None = None
        self._escalation_task: asyncio.Task | None = None

    # ── Stale-run recovery ─────────────────────────────────────────────

    async def _recover_stale_runs(self) -> None:
        """Find runs stuck in 'running' state beyond 2x cycle_timeout and mark them failed."""
        cutoff = self.config.cycle_timeout_seconds * 2
        stale_runs = await self.journal.get_stale_runs(cutoff)
        for run in stale_runs:
            logger.warning(
                f'Recovering stale run {run.id} for {run.project_id} '
                f'(started {run.started_at.isoformat()})'
            )
            run.stage_reports['_error'] = {
                'error_type': 'StaleRunRecovery',
                'error_message': f'Run stuck for >{cutoff}s, recovered by harness',
                'failed_stage': None,
            }
            await self.journal.update_run_stage_reports(run.id, run.stage_reports)
            await self.journal.complete_run(run.id, 'failed')
            restored = await self.buffer.restore_drained(run.project_id)
            if restored:
                logger.info(f'Restored {restored} drained events for stale run {run.id}')
            await self.buffer.mark_run_complete(run.project_id)
            self._escalate('recon_stale_run', run.id, f'Run stuck for >{cutoff}s, recovered')

    # ── Escalation support ─────────────────────────────────────────────

    async def _start_escalation_server(self) -> None:
        """Start the escalation MCP server as a background asyncio task."""
        if not HAS_ESCALATION:
            logger.info('Escalation package not installed — skipping escalation server')
            return

        queue_dir = Path(self.config.escalation_queue_dir)
        if not queue_dir.is_absolute():
            queue_dir = Path(self.config.explore_codebase_root) / queue_dir
        self._escalation_queue = EscalationQueue(queue_dir)  # type: ignore[possibly-undefined]

        mcp_server = create_escalation_server(self._escalation_queue)  # type: ignore[possibly-undefined]
        host = self.config.escalation_host
        port = self.config.escalation_port

        async def _serve():
            try:
                await mcp_server.run_http_async(host=host, port=port)
            except Exception as e:
                logger.error(f'Escalation server error: {e}')

        self._escalation_task = asyncio.create_task(_serve(), name='recon-escalation-server')
        logger.info(f'Reconciliation escalation server starting on {host}:{port}')
        await asyncio.sleep(0.5)

        # Set escalation URL on all stages
        escalation_url = f'http://{host}:{port}/mcp'
        for stage in self.stages:
            stage._escalation_url = escalation_url

    async def _stop_escalation_server(self) -> None:
        """Stop the escalation server."""
        if self._escalation_task is not None:
            self._escalation_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._escalation_task
            self._escalation_task = None
            logger.info('Reconciliation escalation server stopped')

    def _escalate(
        self,
        category: str,
        run_id: str,
        summary: str,
        detail: str = '',
    ) -> None:
        """Submit an escalation to the queue (fire-and-forget)."""
        if not HAS_ESCALATION or self._escalation_queue is None:
            return
        try:
            queue = self._escalation_queue
            esc = Escalation(  # type: ignore[possibly-undefined]
                id=queue.make_id(f'recon-{run_id[:8]}'),
                task_id=f'recon-{run_id[:8]}',
                agent_role='reconciliation-harness',
                severity='info' if category == 'recon_stale_run' else 'blocking',
                category=category,
                summary=summary,
                detail=detail,
            )
            queue.submit(esc)
        except Exception as e:
            logger.warning(f'Failed to submit escalation: {e}')

    # ── Tier selection ─────────────────────────────────────────────────

    async def _select_tier(self, project_id: str) -> TierConfig:
        """Choose model tier based on buffer size."""
        buffer_count = (await self.buffer.get_buffer_stats(project_id)).get('size', 0)
        use_opus = buffer_count > (self.config.buffer_size_threshold * self.config.opus_threshold_ratio)

        if use_opus:
            return TierConfig(
                model=self.config.opus_model,
                episode_limit=self.config.opus_episode_limit,
                memory_limit=self.config.opus_memory_limit,
            )
        return TierConfig(
            model=self.config.sonnet_model,
            episode_limit=self.config.sonnet_episode_limit,
            memory_limit=self.config.sonnet_memory_limit,
        )

    # ── Main loop ──────────────────────────────────────────────────────

    async def run_loop(self) -> None:
        """Background loop — check trigger conditions, run pipeline when needed."""
        logger.info('Reconciliation harness background loop started')
        if self.usage_gate:
            await self.usage_gate.check_at_startup()

        await self._start_escalation_server()

        loop_count = 0
        try:
            while True:
                try:
                    # Recover stale runs each iteration
                    await self._recover_stale_runs()

                    for project_id in await self.buffer.get_active_projects():
                        should, reason = await self.buffer.should_trigger(project_id)
                        if should:
                            acquired = await self.buffer.mark_run_active(project_id)
                            if acquired:
                                # Select tier before draining
                                tier = await self._select_tier(project_id)

                                # Check if backlog iteration is needed
                                iterator = BacklogIterator(
                                    self.config, self.journal, self.buffer, self,
                                )
                                if await iterator.should_iterate(project_id):
                                    heartbeat_task = asyncio.create_task(
                                        self._heartbeat_loop(project_id)
                                    )
                                    try:
                                        await iterator.run(project_id)
                                    finally:
                                        heartbeat_task.cancel()
                                        with suppress(asyncio.CancelledError):
                                            await heartbeat_task
                                        await self.buffer.mark_run_complete(project_id)
                                else:
                                    heartbeat_task = asyncio.create_task(
                                        self._heartbeat_loop(project_id)
                                    )
                                    try:
                                        await asyncio.wait_for(
                                            self.run_full_cycle(
                                                project_id, reason, tier=tier,
                                            ),
                                            timeout=self.config.cycle_timeout_seconds,
                                        )
                                    except TimeoutError:
                                        logger.error(
                                            f'Full cycle timed out after '
                                            f'{self.config.cycle_timeout_seconds}s for {project_id}'
                                        )
                                        await self.buffer.restore_drained(project_id)
                                    finally:
                                        heartbeat_task.cancel()
                                        with suppress(asyncio.CancelledError):
                                            await heartbeat_task
                                        await self.buffer.mark_run_complete(project_id)

                    # Periodic cleanup of drained events (~every 50s / 10 iterations)
                    loop_count += 1
                    if loop_count % 10 == 0:
                        try:
                            deleted = await self.buffer.cleanup_drained()
                            if deleted:
                                logger.debug(f'Cleaned up {deleted} drained events')
                        except Exception as e:
                            logger.warning(f'Drained event cleanup failed: {e}')

                except Exception as e:
                    logger.error(f'Reconciliation loop error: {e}')
                await asyncio.sleep(5)
        finally:
            await self._stop_escalation_server()

    async def _heartbeat_loop(self, project_id: str) -> None:
        """Keep the reconciliation lock alive while a run is in progress."""
        while True:
            await asyncio.sleep(60)
            try:
                await self.buffer.heartbeat(project_id)
            except Exception as e:
                logger.warning(f'Heartbeat failed for {project_id}: {e}')

    async def _run_judge(self, run_id: str) -> None:
        """Fire-and-forget judge wrapper with error logging."""
        try:
            assert self.judge is not None
            verdict = await self.judge.review_run(run_id)
            if verdict:
                logger.info(f'Judge verdict for {run_id}: severity={verdict.severity}')
            else:
                logger.warning(f'Judge returned no verdict for run {run_id}')
        except Exception:
            logger.error(f'Judge task failed for run {run_id}', exc_info=True)

    async def run_full_cycle(
        self,
        project_id: str,
        trigger_reason: str,
        tier: TierConfig | None = None,
    ) -> ReconciliationRun:
        """Execute the three-stage pipeline for a project."""
        tier = tier or TierConfig()
        run_id = str(uuid4())
        watermark = await self.journal.get_watermark(project_id)
        events = await self.buffer.drain(project_id)

        run = ReconciliationRun(
            id=run_id,
            project_id=project_id,
            run_type='full',
            trigger_reason=trigger_reason,
            started_at=datetime.now(UTC),
            events_processed=len(events),
            status='running',
        )
        await self.journal.start_run(run)

        logger.info(
            'reconciliation.run_started',
            extra={
                'run_id': run_id,
                'project_id': project_id,
                'run_type': 'full',
                'trigger_reason': trigger_reason,
                'events_to_process': len(events),
                'model': tier.model,
            },
        )

        # Extract project_root from event payloads (first event with _project_root key)
        project_root = project_id  # fallback for backwards compatibility
        for ev in events:
            pr = ev.payload.get('_project_root')
            if pr:
                project_root = pr
                break

        current_stage_name: str | None = None
        try:
            reports = []
            for stage in self.stages:
                current_stage_name = stage.stage_id.value
                stage.project_id = project_id
                stage.project_root = project_root

                # Apply tier limits to Stage 1
                if isinstance(stage, MemoryConsolidator):
                    stage.episode_limit = tier.episode_limit
                    stage.memory_limit = tier.memory_limit

                report = await stage.run(
                    events, watermark, reports, run_id, model=tier.model,
                )
                reports.append(report)
                run.stage_reports[stage.stage_id.value] = report

            # Update watermark
            watermark.last_full_run_id = run_id
            watermark.last_full_run_completed = datetime.now(UTC)
            watermark.last_episode_timestamp = datetime.now(UTC)
            watermark.last_memory_timestamp = datetime.now(UTC)
            watermark.last_task_change_timestamp = datetime.now(UTC)
            await self.journal.update_watermark(watermark)

            run.completed_at = datetime.now(UTC)
            run.status = 'completed'
            await self.journal.complete_run(run_id, 'completed')

            # Async judge review
            if self.judge:
                asyncio.create_task(self._run_judge(run_id))

            logger.info(
                'reconciliation.run_completed',
                extra={
                    'run_id': run_id,
                    'project_id': project_id,
                    'status': 'completed',
                },
            )
            return run

        except Exception as e:
            run.status = 'failed'
            run.stage_reports['_error'] = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'failed_stage': current_stage_name,
                'traceback': traceback.format_exc(),
            }
            await self.journal.complete_run(run_id, 'failed')
            await self.buffer.restore_drained(project_id)
            logger.error(f'Reconciliation failed: {e}')
            self._escalate(
                'recon_failure', run_id,
                f'Stage {current_stage_name} failed: {e}',
            )
            raise
        finally:
            await self.journal.update_run_stage_reports(run_id, run.stage_reports)


# ── Backlog iteration ──────────────────────────────────────────────────


class BacklogIterator:
    """Processes large backlogs in chunks, oldest-first, then consolidates."""

    def __init__(
        self,
        config,
        journal: ReconciliationJournal,
        buffer: EventBuffer,
        harness: ReconciliationHarness,
    ):
        self.config = config
        self.journal = journal
        self.buffer = buffer
        self.harness = harness

    async def should_iterate(self, project_id: str) -> bool:
        """Buffer count > 150% of trigger threshold."""
        stats = await self.buffer.get_buffer_stats(project_id)
        count = stats.get('size', 0)
        threshold = self.config.buffer_size_threshold * self.config.opus_threshold_ratio
        return count > threshold

    async def run(self, project_id: str) -> None:
        """Process backlog in chunks, oldest-first, then consolidate."""
        chunk_size = self.config.buffer_size_threshold
        opus_tier = TierConfig(
            model=self.config.opus_model,
            episode_limit=self.config.opus_episode_limit,
            memory_limit=self.config.opus_memory_limit,
        )

        chunk_num = 0
        while True:
            chunk = await self.buffer.drain_oldest_chunk(project_id, chunk_size)
            if not chunk:
                break

            chunk_num += 1
            chunk_id = str(uuid4())
            await self.journal.record_chunk_boundary(
                project_id, chunk_id, len(chunk),
            )

            logger.info(
                f'Backlog chunk {chunk_num}: processing {len(chunk)} events '
                f'for {project_id}'
            )

            try:
                await asyncio.wait_for(
                    self.harness.run_full_cycle(
                        project_id, f'backlog_chunk:{chunk_num}:{len(chunk)}',
                        tier=opus_tier,
                    ),
                    timeout=self.config.cycle_timeout_seconds,
                )
            except Exception as e:
                logger.error(f'Backlog chunk {chunk_num} failed: {e}')
                self.harness._escalate(
                    'recon_backlog_overflow', chunk_id,
                    f'Backlog chunk {chunk_num} failed, stopping iteration: {e}',
                )
                await self.buffer.restore_drained(project_id)
                return  # Stop iteration on failure

        # Final consolidation pass
        if chunk_num > 0:
            logger.info(f'Backlog final consolidation for {project_id}')
            try:
                await asyncio.wait_for(
                    self.harness.run_full_cycle(
                        project_id, 'backlog_final_consolidation',
                        tier=opus_tier,
                    ),
                    timeout=self.config.cycle_timeout_seconds,
                )
            except Exception as e:
                logger.error(f'Backlog final consolidation failed: {e}')
                await self.buffer.restore_drained(project_id)
