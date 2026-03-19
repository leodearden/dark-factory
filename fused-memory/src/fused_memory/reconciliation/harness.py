"""Pipeline orchestrator — runs the three-stage reconciliation pipeline."""

import asyncio
import logging
import traceback
from contextlib import suppress
from datetime import UTC, datetime
from uuid import uuid4

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.reconciliation import (
    ReconciliationRun,
    StageId,
)
from fused_memory.reconciliation.agent_loop import CircuitBreakerError
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.judge import Judge
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)
from fused_memory.reconciliation.verify import CodebaseVerifier
from fused_memory.services.memory_service import MemoryService
from shared.usage_gate import UsageGate

logger = logging.getLogger(__name__)


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

        # Verifier
        verifier = CodebaseVerifier(self.config) if self.config.explore_codebase_root else None

        # Build stages
        stage1 = MemoryConsolidator(
            StageId.memory_consolidator, memory_service, taskmaster, journal, self.config,
            usage_gate=self.usage_gate,
        )
        stage1.verifier = verifier

        stage2 = TaskKnowledgeSync(
            StageId.task_knowledge_sync, memory_service, taskmaster, journal, self.config,
            usage_gate=self.usage_gate,
        )
        stage2.verifier = verifier

        stage3 = IntegrityCheck(
            StageId.integrity_check, memory_service, taskmaster, journal, self.config,
            usage_gate=self.usage_gate,
        )
        stage3.verifier = verifier

        self.stages = [stage1, stage2, stage3]

        # Judge
        self.judge = (
            Judge(self.config, journal, usage_gate=self.usage_gate)
            if self.config.judge_enabled else None
        )

    async def run_loop(self) -> None:
        """Background loop — check trigger conditions, run pipeline when needed."""
        logger.info('Reconciliation harness background loop started')
        if self.usage_gate:
            await self.usage_gate.check_at_startup()
        loop_count = 0
        while True:
            try:
                for project_id in await self.buffer.get_active_projects():
                    should, reason = await self.buffer.should_trigger(project_id)
                    if should:
                        acquired = await self.buffer.mark_run_active(project_id)
                        if acquired:
                            heartbeat_task = asyncio.create_task(
                                self._heartbeat_loop(project_id)
                            )
                            try:
                                await asyncio.wait_for(
                                    self.run_full_cycle(project_id, reason),
                                    timeout=self.config.cycle_timeout_seconds,
                                )
                            except TimeoutError:
                                logger.error(
                                    f'Full cycle timed out after {self.config.cycle_timeout_seconds}s '
                                    f'for {project_id}'
                                )
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

    async def _heartbeat_loop(self, project_id: str) -> None:
        """Keep the reconciliation lock alive while a run is in progress."""
        while True:
            await asyncio.sleep(60)
            try:
                await self.buffer.heartbeat(project_id)
            except Exception as e:
                logger.warning(f'Heartbeat failed for {project_id}: {e}')

    async def run_full_cycle(self, project_id: str, trigger_reason: str) -> ReconciliationRun:
        """Execute the three-stage pipeline for a project."""
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
                report = await stage.run(events, watermark, reports, run_id)
                reports.append(report)
                run.stage_reports[stage.stage_id.value] = report

            # Update watermark
            watermark.last_full_run_id = run_id
            watermark.last_full_run_completed = datetime.now(UTC)
            await self.journal.update_watermark(watermark)

            run.completed_at = datetime.now(UTC)
            run.status = 'completed'
            await self.journal.complete_run(run_id, 'completed')

            # Async judge review
            if self.judge:
                asyncio.create_task(self.judge.review_run(run_id))

            logger.info(
                'reconciliation.run_completed',
                extra={
                    'run_id': run_id,
                    'project_id': project_id,
                    'status': 'completed',
                },
            )
            return run

        except CircuitBreakerError as e:
            run.status = 'circuit_breaker'
            run.stage_reports['_error'] = {
                'error_type': 'CircuitBreakerError',
                'error_message': str(e),
                'failed_stage': current_stage_name,
                'traceback': traceback.format_exc(),
            }
            await self.journal.complete_run(run_id, 'circuit_breaker')
            logger.error(f'Reconciliation circuit breaker: {e}')
            raise
        except Exception as e:
            run.status = 'failed'
            run.stage_reports['_error'] = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'failed_stage': current_stage_name,
                'traceback': traceback.format_exc(),
            }
            await self.journal.complete_run(run_id, 'failed')
            logger.error(f'Reconciliation failed: {e}')
            raise
        finally:
            await self.journal.update_run_stage_reports(run_id, run.stage_reports)
