"""Pipeline orchestrator — runs the three-stage reconciliation pipeline."""

from __future__ import annotations

import asyncio
import json
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
    ReconciliationEvent,
    ReconciliationRun,
    RunStatus,
    RunType,
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
            await self._replay_deferred_writes(run.project_id)
            self._escalate('recon_stale_run', run.id, f'Run stuck for >{cutoff}s, recovered')

    # ── Deferred write replay ─────────────────────────────────────────

    async def _replay_deferred_writes(self, project_id: str) -> None:
        """Replay targeted-recon writes that were deferred during a full cycle."""
        deferred = await self.buffer.pop_deferred_writes(project_id)
        if not deferred:
            return
        logger.info(f'Replaying {len(deferred)} deferred writes for {project_id}')
        for write in deferred:
            try:
                await self.memory.add_memory(
                    content=write['content'],
                    category=write['category'],
                    project_id=project_id,
                    metadata=write['metadata'],
                    _source='targeted_recon',
                )
            except Exception as e:
                logger.warning(f'Failed to replay deferred write: {e}')

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
                severity='info' if category in ('recon_stale_run', 'recon_integrity_issue') else 'blocking',
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
                                # Skip cycle for halted projects
                                if self.judge and self.judge.is_halted(project_id):
                                    logger.warning(
                                        f'Skipping cycle for halted project {project_id}'
                                    )
                                    await self.buffer.mark_run_complete(project_id)
                                    await self._replay_deferred_writes(project_id)
                                    continue

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
                                        await self._replay_deferred_writes(project_id)
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
                                        await self._replay_deferred_writes(project_id)

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
        events: list[ReconciliationEvent] | None = None,
    ) -> ReconciliationRun:
        """Execute the three-stage pipeline for a project.

        Args:
            events: Optional pre-drained event list.  When provided, buffer.drain()
                    is skipped and these events are used directly.  This allows
                    BacklogIterator to pass already-drained chunk events without
                    a double-drain.
        """
        tier = tier or TierConfig()
        run_id = str(uuid4())
        watermark = await self.journal.get_watermark(project_id)
        if events is None:
            events = await self.buffer.drain(project_id)

        run = ReconciliationRun(
            id=run_id,
            project_id=project_id,
            run_type=RunType.full,
            trigger_reason=trigger_reason,
            started_at=datetime.now(UTC),
            events_processed=len(events),
            status=RunStatus.running,
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

        # Load prior S3 findings from last completed run (backstop for normal pass)
        prior_s3_findings = await self._get_prior_s3_findings(project_id)

        current_stage_name: str | None = None
        cycle_start_time = datetime.now(UTC)
        try:
            reports = []
            for stage in self.stages:
                current_stage_name = stage.stage_id.value
                stage.project_id = project_id
                stage.project_root = project_root

                # Apply tier limits, prior S3 findings, and cycle fence to Stage 1
                if isinstance(stage, MemoryConsolidator):
                    stage.episode_limit = tier.episode_limit
                    stage.memory_limit = tier.memory_limit
                    stage.prior_s3_findings = prior_s3_findings
                    stage.cycle_fence_time = cycle_start_time

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
            run.status = RunStatus.completed
            await self.journal.complete_run(run_id, 'completed')

            # Persist stage reports before judge — the judge reads from the DB,
            # so reports must be committed before firing the async task.
            await self.journal.update_run_stage_reports(run_id, run.stage_reports)

            # Async judge review
            if self.judge:
                asyncio.create_task(self._run_judge(run_id))

            # Remediation pass: extract S3 findings and trigger second pass
            await self._maybe_remediate(project_id, project_root, run_id, run, tier)

            logger.info(
                'reconciliation.run_completed',
                extra={
                    'run_id': run_id,
                    'project_id': project_id,
                    'status': 'completed',
                },
            )
            return run

        except asyncio.CancelledError:
            # asyncio.wait_for cancels via CancelledError, which is NOT a subclass of
            # Exception in Python 3.8+.  Without this handler the journal run is left
            # stuck in 'running'.  Mark it failed, restore events, then re-raise so
            # asyncio cancellation semantics are preserved.
            #
            # Two defences against cleanup being interrupted:
            # 1. asyncio.shield() — runs the cleanup coroutine in its own Task so a
            #    second cancellation (e.g. server shutdown) cannot abort the DB write.
            # 2. Independent try/except BaseException per cleanup step — each step
            #    runs regardless of the other's outcome, and CancelledError is still
            #    re-raised to the caller.
            run.status = RunStatus.failed
            run.stage_reports['_error'] = {
                'error_type': 'CancelledError',
                'error_message': 'Run cancelled (timeout or external cancellation)',
                'failed_stage': current_stage_name,
                'traceback': '',
            }
            try:
                await asyncio.shield(self.journal.complete_run(run_id, 'failed'))
            except BaseException as cleanup_err:
                logger.error(f'complete_run failed after cancellation: {cleanup_err}')
            try:
                await asyncio.shield(self.buffer.restore_drained(project_id))
            except BaseException as cleanup_err:
                logger.error(f'restore_drained failed after cancellation: {cleanup_err}')
            logger.error(
                f'Reconciliation run {run_id} cancelled for {project_id} '
                f'(stage: {current_stage_name})'
            )
            raise
        except Exception as e:
            run.status = RunStatus.failed
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
            # Reset stage state to prevent leaking into next cycle
            for stage in self.stages:
                if isinstance(stage, MemoryConsolidator):
                    stage.prior_s3_findings = None
                    stage.cycle_fence_time = None

    # ── Remediation support ───────────────────────────────────────────

    async def _get_prior_s3_findings(self, project_id: str) -> list[dict] | None:
        """Extract S3 findings from the last completed run's stage reports."""
        try:
            recent = await self.journal.get_recent_runs(project_id, limit=3)
            for r in recent:
                if r.status != 'completed':
                    continue
                s3_report = r.stage_reports.get('integrity_check')
                if s3_report is None:
                    continue
                if isinstance(s3_report, dict):
                    items = s3_report.get('items_flagged', [])
                else:
                    items = s3_report.items_flagged
                if items:
                    return items
        except Exception as e:
            logger.warning(f'Failed to load prior S3 findings: {e}')
        return None

    async def _maybe_remediate(
        self,
        project_id: str,
        project_root: str,
        parent_run_id: str,
        parent_run: ReconciliationRun,
        tier: TierConfig,
    ) -> None:
        """Extract Stage 3 findings from the parent run and trigger remediation if needed."""
        try:
            s3_report = parent_run.stage_reports.get('integrity_check')
            if s3_report is None:
                return

            if isinstance(s3_report, dict):
                all_findings = s3_report.get('items_flagged', [])
            else:
                all_findings = s3_report.items_flagged

            if not all_findings:
                return

            # Partition into actionable vs escalation
            actionable = [f for f in all_findings if f.get('actionable', False)]
            non_actionable = [f for f in all_findings if not f.get('actionable', False)]

            # Escalate non-actionable findings immediately
            for finding in non_actionable:
                self._escalate(
                    'recon_integrity_issue',
                    parent_run_id,
                    f'Non-actionable integrity finding: {finding.get("description", "?")}',
                    detail=json.dumps(finding, default=str),
                )

            if not actionable:
                return

            logger.info(
                f'Remediation: {len(actionable)} actionable findings from run {parent_run_id}, '
                f'triggering second pass'
            )
            await self._run_remediation_pass(
                project_id, project_root, parent_run_id, actionable, tier,
            )
        except Exception as e:
            logger.error(f'Remediation check failed for run {parent_run_id}: {e}')
            self._escalate(
                'recon_integrity_issue',
                parent_run_id,
                f'Remediation orchestration failed: {e}',
            )

    async def _run_remediation_pass(
        self,
        project_id: str,
        project_root: str,
        parent_run_id: str,
        findings: list[dict],
        tier: TierConfig,
    ) -> None:
        """Run a focused S1→S2→S3 pass to remediate actionable findings."""
        run_id = str(uuid4())
        run = ReconciliationRun(
            id=run_id,
            project_id=project_id,
            run_type=RunType.remediation,
            trigger_reason=f'integrity_findings:{len(findings)}',
            started_at=datetime.now(UTC),
            events_processed=0,
            status=RunStatus.running,
            triggered_by=parent_run_id,
        )
        await self.journal.start_run(run)

        logger.info(
            'reconciliation.remediation_started',
            extra={
                'run_id': run_id,
                'parent_run_id': parent_run_id,
                'project_id': project_id,
                'findings_count': len(findings),
            },
        )

        current_stage_name: str | None = None
        try:
            # Configure stages for remediation mode
            stage1 = self.stages[0]
            stage2 = self.stages[1]
            assert isinstance(stage1, MemoryConsolidator)
            assert isinstance(stage2, TaskKnowledgeSync)
            stage1.remediation_findings = findings
            stage2.remediation_mode = True

            watermark = await self.journal.get_watermark(project_id)
            reports = []
            for stage in self.stages:
                current_stage_name = stage.stage_id.value
                stage.project_id = project_id
                stage.project_root = project_root

                report = await stage.run(
                    [], watermark, reports, run_id, model=tier.model,
                )
                reports.append(report)
                run.stage_reports[stage.stage_id.value] = report

            # Do NOT update watermark — remediation processed no new episodes/events

            run.completed_at = datetime.now(UTC)
            run.status = RunStatus.completed
            await self.journal.complete_run(run_id, 'completed')

            # Persist stage reports before judge (same fix as run_full_cycle)
            await self.journal.update_run_stage_reports(run_id, run.stage_reports)

            # Judge review for remediation run
            if self.judge:
                asyncio.create_task(self._run_judge(run_id))

            # After second-pass S3: escalate ALL remaining findings (never a third pass)
            s3_report = run.stage_reports.get('integrity_check')
            if s3_report is not None:
                if isinstance(s3_report, dict):
                    remaining = s3_report.get('items_flagged', [])
                else:
                    remaining = s3_report.items_flagged
                for finding in remaining:
                    self._escalate(
                        'recon_integrity_issue',
                        run_id,
                        f'Unresolved after remediation: {finding.get("description", "?")}',
                        detail=json.dumps(finding, default=str),
                    )

            logger.info(
                'reconciliation.remediation_completed',
                extra={'run_id': run_id, 'parent_run_id': parent_run_id},
            )

        except Exception as e:
            run.status = RunStatus.failed
            run.stage_reports['_error'] = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'failed_stage': current_stage_name,
            }
            await self.journal.complete_run(run_id, 'failed')
            # Do NOT re-raise — parent run already completed
            # Do NOT restore events — there are none
            logger.error(f'Remediation pass failed: {e}')
            self._escalate(
                'recon_integrity_issue',
                run_id,
                f'Remediation pass failed at {current_stage_name}: {e}',
            )

        finally:
            await self.journal.update_run_stage_reports(run_id, run.stage_reports)
            # Reset stage state to prevent leaking into next cycle
            for stage in self.stages:
                if isinstance(stage, MemoryConsolidator):
                    stage.remediation_findings = None
                    stage.prior_s3_findings = None
                if isinstance(stage, TaskKnowledgeSync):
                    stage.remediation_mode = False


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
        """Process backlog in chunks, oldest-first, then consolidate.

        Only drains events buffered before this method was called.  Events
        that arrive *during* processing are left for the next trigger cycle,
        preventing the loop from running indefinitely when events arrive
        faster than chunks are processed.
        """
        chunk_size = self.config.buffer_size_threshold
        opus_tier = TierConfig(
            model=self.config.opus_model,
            episode_limit=self.config.opus_episode_limit,
            memory_limit=self.config.opus_memory_limit,
        )

        # Snapshot: only process events that existed when we started.
        cutoff = datetime.now(UTC)

        chunk_num = 0
        while True:
            chunk = await self.buffer.drain_oldest_chunk(
                project_id, chunk_size, before=cutoff,
            )
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
                        events=chunk,
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
