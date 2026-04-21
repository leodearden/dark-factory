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
from uuid import uuid4

from shared.usage_gate import UsageGate

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.reconciliation import (
    AssembledPayload,
    ReconciliationEvent,
    ReconciliationRun,
    RunStatus,
    RunType,
    StageId,
)
from fused_memory.reconciliation.backlog_policy import BacklogPolicy
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.judge import Judge
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)
from fused_memory.reconciliation.stats_verifier import verify_and_rewrite_stats
from fused_memory.reconciliation.task_filter import FilteredTaskTree, filter_task_tree
from fused_memory.services.memory_service import MemoryService

try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]
    from escalation.server import (  # type: ignore[import-untyped]
        create_server as create_escalation_server,
    )
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
        backlog_policy: BacklogPolicy | None = None,
    ):
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.journal = journal
        self.buffer = event_buffer
        self.config = config.reconciliation
        self._backlog_policy = backlog_policy
        # WP-D: track which halted projects we've already escalated so we
        # don't re-fire every harness tick.
        self._halt_escalated: set[str] = set()

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

        # Judge — receives a callback that clears _halt_escalated so a
        # subsequent halt in the same process re-fires the escalation.
        self.judge = (
            Judge(
                self.config, journal,
                usage_gate=self.usage_gate,
                on_unhalt_cb=self._on_judge_unhalt,
            )
            if self.config.judge_enabled else None
        )

        # Escalation support
        self._escalation_queue: EscalationQueue | None = None
        self._escalation_task: asyncio.Task | None = None
        self._escalation_url: str | None = None

        # Per-project concurrent loops
        self._project_tasks: dict[str, asyncio.Task] = {}

        # Drain mode: stop starting new cycles, let current ones finish
        self._draining: bool = False

    async def _notify_judge_halt(self, project_id: str, reason: str) -> None:
        """WP-D: forward judge halts to the backlog policy exactly once.

        Routes to escalation when an orchestrator is live for this project;
        otherwise the next mutating MCP call will surface the halt as a
        structured rejection via :class:`BacklogPolicy`. Best-effort: a
        failure here must not break the harness loop.
        """
        if self._backlog_policy is None or project_id in self._halt_escalated:
            return
        self._halt_escalated.add(project_id)
        try:
            await self._backlog_policy.on_judge_halt(project_id, reason)
        except Exception:
            logger.exception(
                'harness: backlog_policy.on_judge_halt raised for %s', project_id,
            )

    def _on_judge_unhalt(self, project_id: str) -> None:
        """Callback invoked by Judge.unhalt so a subsequent halt re-escalates.

        Without clearing the escalation sentinel, a manual unhalt followed by
        the halt re-firing (for whatever reason) would silently skip the
        escalation path because _notify_judge_halt dedupes per-process.
        """
        self._halt_escalated.discard(project_id)

    def drain(self) -> None:
        """Signal the harness to stop starting new reconciliation cycles.

        Currently-running project loops complete their current cycle.
        The server continues serving reads/writes — only new cycles are suppressed.
        """
        if self._draining:
            logger.info('Harness already draining')
            return
        self._draining = True
        logger.info('Harness drain requested — will finish current cycles, no new ones')

    @property
    def is_drained(self) -> bool:
        """True when draining and all project loops have completed."""
        return self._draining and not any(
            not t.done() for t in self._project_tasks.values()
        )

    def _make_stages(self) -> list:
        """Create a fresh set of stage instances for one reconciliation cycle."""
        stage1 = MemoryConsolidator(
            StageId.memory_consolidator, self.memory, self.taskmaster, self.journal,
            self.config, usage_gate=self.usage_gate,
        )
        stage2 = TaskKnowledgeSync(
            StageId.task_knowledge_sync, self.memory, self.taskmaster, self.journal,
            self.config, usage_gate=self.usage_gate,
        )
        stage3 = IntegrityCheck(
            StageId.integrity_check, self.memory, self.taskmaster, self.journal,
            self.config, usage_gate=self.usage_gate,
        )
        stages = [stage1, stage2, stage3]
        if self._escalation_url:
            for s in stages:
                s._escalation_url = self._escalation_url
        return stages

    @staticmethod
    def _configure_consolidator(
        stage: MemoryConsolidator,
        tier: TierConfig,
        *,
        prior_s3_findings: list[dict] | None = None,
        cycle_fence_time: datetime | None = None,
        assembled_payload: AssembledPayload | None = None,
        remediation_findings: list[dict] | None = None,
        filtered_task_tree: FilteredTaskTree | None = None,
    ) -> None:
        """Apply tier limits and mode-specific attributes to MemoryConsolidator.

        Shared between run_full_cycle and _run_remediation_pass to prevent
        attribute-configuration divergence.

        Note: filtered_task_tree here applies only to Stage 1 (MemoryConsolidator).
        Stage 2 (TaskKnowledgeSync) wiring is handled by the symmetric
        _configure_task_sync helper.
        """
        stage.episode_limit = tier.episode_limit
        stage.memory_limit = tier.memory_limit
        stage.prior_s3_findings = prior_s3_findings
        stage.cycle_fence_time = cycle_fence_time
        stage.assembled_payload = assembled_payload
        stage.remediation_findings = remediation_findings
        stage.filtered_task_tree = filtered_task_tree

    @staticmethod
    def _configure_task_sync(
        stage: TaskKnowledgeSync,
        *,
        filtered_task_tree: FilteredTaskTree | None = None,
        remediation_mode: bool = False,
    ) -> None:
        """Apply Stage-2 attributes to TaskKnowledgeSync.

        Mirrors _configure_consolidator for Stage 1 — both full-cycle and
        remediation paths use this helper to keep attribute wiring symmetric
        and discoverable. Stage-2-specific attributes (filtered_task_tree,
        remediation_mode) are set here; Stage-1 wiring is handled by
        _configure_consolidator.
        """
        stage.filtered_task_tree = filtered_task_tree
        stage.remediation_mode = remediation_mode

    async def _fetch_filtered_task_tree(self, project_root: str) -> FilteredTaskTree:
        """Fetch the task tree once and return a filtered subset of active tasks.

        Degrades gracefully on failure — returns an empty FilteredTaskTree so
        stages can still do useful memory work without task data. (ref: task 455)

        Args:
            project_root: Absolute path to the project root for taskmaster.

        Returns:
            FilteredTaskTree with active tasks sorted by priority and aggregate
            counts. Returns empty FilteredTaskTree if taskmaster is unavailable,
            project_root is empty, or the fetch fails.
        """
        if not self.taskmaster or not project_root:
            return FilteredTaskTree()
        try:
            tasks_data = await self.taskmaster.get_tasks(project_root=project_root)
            return filter_task_tree(tasks_data)
        except Exception as exc:
            logger.warning(
                f'_fetch_filtered_task_tree failed for {project_root!r}: {exc}'
            )
            return FilteredTaskTree()

    # ── Stale-run recovery ─────────────────────────────────────────────

    async def _recover_stale_runs(self) -> None:
        """Find runs stuck in 'running' state and mark them failed.

        Uses stale_run_recovery_seconds as the age cutoff (default 600s),
        then double-checks that the project's reconciliation lock is actually
        stale before recovering — protecting legitimately long-running cycles.
        """
        cutoff = self.config.stale_run_recovery_seconds
        stale_runs = await self.journal.get_stale_runs(cutoff)
        for run in stale_runs:
            # Skip if lock is still actively held (legitimate long-running cycle)
            if await self.buffer._is_run_locked(run.project_id):
                continue

            logger.warning(
                f'Recovering stale run {run.id} for {run.project_id} '
                f'(started {run.started_at.isoformat()}, lock expired)'
            )
            run.stage_reports['_error'] = {
                'error_type': 'StaleRunRecovery',
                'error_message': f'Run stale (>{cutoff}s, lock expired), recovered by harness',
                'failed_stage': None,
            }
            await self.journal.update_run_stage_reports(run.id, run.stage_reports)
            await self.journal.complete_run(run.id, 'failed')
            restored = await self.buffer.restore_drained(run.project_id)
            if restored:
                logger.info(f'Restored {restored} drained events for stale run {run.id}')
            await self.buffer.mark_run_complete(run.project_id)
            await self._replay_deferred_writes(run.project_id)
            self._escalate('recon_stale_run', run.id, f'Run stale (>{cutoff}s, lock expired), recovered')

    # ── Deferred write replay ─────────────────────────────────────────

    async def _replay_deferred_writes(self, project_id: str) -> None:
        """Replay targeted-recon writes that were deferred during a full cycle.

        Uses a claim → replay-one → delete-on-success pattern so that
        cancellation or process crash mid-loop does not lose any writes:
        claimed-but-not-deleted rows remain in SQLite and are recovered
        on next startup by `release_stale_claims`.
        """
        deferred = await self.buffer.claim_deferred_writes(project_id)
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
                # Leave the row claimed so it isn't retried in this process;
                # release_stale_claims at next startup will re-queue it.
                logger.warning(f'Failed to replay deferred write {write["id"]}: {e}')
                continue
            await self.buffer.delete_deferred_write(write['id'])

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

        # Store escalation URL for _make_stages() and set on existing stages
        escalation_url = f'http://{host}:{port}/mcp'
        self._escalation_url = escalation_url
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
        """Management loop — discover active projects, spawn per-project loops."""
        logger.info('Reconciliation harness background loop started')
        if self.usage_gate:
            await self.usage_gate.check_at_startup()

        # Rehydrate persistent halt state from the journal so a restart cannot
        # silently clear a halt that hasn't been explicitly cleared by an
        # operator. Called once at startup.
        if self.judge is not None:
            await self.judge.initialize()

        await self._start_escalation_server()

        # Re-queue any deferred writes left in-progress by a crashed prior process.
        # Cutoff is 0 (release *every* currently-claimed row) rather than
        # stale_claim_recovery_seconds.  This closes the fast-restart edge case:
        # if a supervisor restarts the harness within stale_claim_recovery_seconds
        # of the previous crash, a time-based cutoff would miss rows whose
        # claimed_at is younger than the horizon, silently stalling those writes.
        # Safety: the per-project reconciliation lock (EventBuffer._is_run_locked /
        # mark_run_active) serialises replay — at startup no project loop has
        # spawned yet, so there is nothing to race with.
        try:
            released = await self.buffer.release_stale_claims(0)
            if released:
                logger.info(
                    f'Recovered {released} stale deferred write claim(s) on startup'
                )
        except Exception as e:
            logger.warning(f'release_stale_claims at startup failed: {e}')

        loop_count = 0
        try:
            while True:
                try:
                    await self._recover_stale_runs()

                    # Discover active projects, spawn loops for new ones
                    if not self._draining:
                        for project_id in await self.buffer.get_active_projects():
                            existing = self._project_tasks.get(project_id)
                            if existing is None or existing.done():
                                task = asyncio.create_task(
                                    self._project_loop(project_id),
                                    name=f'recon-{project_id}',
                                )
                                self._project_tasks[project_id] = task

                    # Reap completed tasks, log unexpected failures
                    for pid in list(self._project_tasks):
                        task = self._project_tasks[pid]
                        if task.done():
                            del self._project_tasks[pid]
                            if not task.cancelled() and task.exception():
                                logger.error(
                                    f'Project loop for {pid} crashed: {task.exception()}'
                                )

                    # Drain status logging
                    if self._draining:
                        active = sum(1 for t in self._project_tasks.values() if not t.done())
                        if active == 0:
                            logger.info('Harness fully drained — safe to restart')
                        else:
                            logger.info(f'Harness draining: {active} project loop(s) still running')

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
            # Graceful shutdown: cancel all project loops
            for task in self._project_tasks.values():
                task.cancel()
            if self._project_tasks:
                await asyncio.gather(
                    *self._project_tasks.values(), return_exceptions=True,
                )
            self._project_tasks.clear()
            await self._stop_escalation_server()

    async def _project_loop(self, project_id: str) -> None:
        """Independent reconciliation loop for a single project."""
        logger.info(f'Project reconciliation loop started for {project_id}')
        idle_ticks = 0
        while True:
            try:
                should, reason = await self.buffer.should_trigger(project_id)
                if not should:
                    idle_ticks += 1
                    if idle_ticks > 12:  # ~60s idle → exit, respawn on demand
                        logger.debug(f'Project loop idle exit for {project_id}')
                        return
                    await asyncio.sleep(5)
                    continue

                idle_ticks = 0
                acquired = await self.buffer.mark_run_active(project_id)
                if not acquired:
                    await asyncio.sleep(5)
                    continue

                # Halt check
                if self.judge and self.judge.is_halted(project_id):
                    logger.warning(f'Skipping cycle for halted project {project_id}')
                    await self._notify_judge_halt(
                        project_id, reason='judge halted reconciliation',
                    )
                    await self.buffer.mark_run_complete(project_id)
                    await self._replay_deferred_writes(project_id)
                    return  # Don't keep spinning on a halted project

                # Decrement post-unhalt grace counter. A just-unhalted project
                # runs `halt_grace_cycles` cycles with trend detection skipped,
                # so stale moderates in the DB can age out before the detector
                # re-engages.
                if self.judge is not None:
                    remaining = await self.judge.consume_grace_cycle(project_id)
                    if remaining > 0:
                        logger.info(
                            f'Running {project_id} within post-unhalt grace '
                            f'({remaining} cycles remaining)'
                        )

                tier = await self._select_tier(project_id)
                iterator = BacklogIterator(self.config, self.journal, self.buffer, self)
                heartbeat_task = asyncio.create_task(self._heartbeat_loop(project_id))
                try:
                    if await iterator.should_iterate(project_id):
                        await iterator.run(project_id)
                    else:
                        await asyncio.wait_for(
                            self.run_full_cycle(project_id, reason, tier=tier),
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

            except asyncio.CancelledError:
                raise  # Propagate shutdown
            except Exception as e:
                logger.error(f'Project loop error for {project_id}: {e}')

            await asyncio.sleep(5)  # Cooldown between cycles

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
        assembled_payload: AssembledPayload | None = None,
    ) -> ReconciliationRun:
        """Execute the three-stage pipeline for a project.

        Args:
            events: Optional pre-drained event list.  When provided, buffer.drain()
                    is skipped and these events are used directly.  This allows
                    BacklogIterator to pass already-drained chunk events without
                    a double-drain.
            assembled_payload: Optional token-budgeted payload from ContextAssembler.
                    When provided, Stage 1 uses this instead of generic
                    time-windowed episode/memory fetches.
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

        # Fetch filtered task tree once for the whole cycle (ref: task 455)
        filtered_task_tree = await self._fetch_filtered_task_tree(project_root)

        current_stage_name: str | None = None
        cycle_start_time = datetime.now(UTC)
        stages = self._make_stages()
        try:
            reports = []
            for stage in stages:
                current_stage_name = stage.stage_id.value
                stage.project_id = project_id
                stage.project_root = project_root

                # Apply tier limits, prior S3 findings, cycle fence, and task tree to Stage 1
                if isinstance(stage, MemoryConsolidator):
                    self._configure_consolidator(
                        stage, tier,
                        prior_s3_findings=prior_s3_findings,
                        cycle_fence_time=cycle_start_time,
                        assembled_payload=assembled_payload,
                        filtered_task_tree=filtered_task_tree,
                    )

                # Wire harness-fetched task tree into Stage 2 via symmetric helper (ref: task 455)
                if isinstance(stage, TaskKnowledgeSync):
                    self._configure_task_sync(stage, filtered_task_tree=filtered_task_tree)

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

            # Cross-check self-reported stats against write-journal ops BEFORE the
            # judge reads them. Stage agents sometimes over-report successful
            # writes when Mem0 silently dedups; the verifier overwrites those
            # counts with observed truth and keeps the originals under _reported.
            await verify_and_rewrite_stats(
                run_id, run.stage_reports, self.journal.write_journal,
            )

            # Persist stage reports before judge — the judge reads from the DB,
            # so reports must be committed before firing the async task.
            await self.journal.update_run_stage_reports(run_id, run.stage_reports)

            # Async judge review
            if self.judge:
                asyncio.create_task(self._run_judge(run_id))

            # Remediation pass: pass pre-fetched tree to avoid a redundant fetch (ref: task 478)
            await self._maybe_remediate(project_id, project_root, run_id, run, tier,
                                        filtered_task_tree=filtered_task_tree)

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
        *,
        filtered_task_tree: FilteredTaskTree | None = None,
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
                filtered_task_tree=filtered_task_tree,
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
        *,
        filtered_task_tree: FilteredTaskTree | None = None,
    ) -> None:
        """Run a focused S1→S2→S3 pass to remediate actionable findings.

        If filtered_task_tree is provided it is used directly; otherwise a fresh
        tree is fetched via _fetch_filtered_task_tree.  Callers that already hold
        a fetched tree (e.g. run_full_cycle) should pass it through to avoid a
        redundant taskmaster round-trip.
        """
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

        # Use caller-supplied tree if available; otherwise fetch (ref: task 455, task 478)
        remediation_tree = (
            filtered_task_tree
            if filtered_task_tree is not None
            else await self._fetch_filtered_task_tree(project_root)
        )

        current_stage_name: str | None = None
        stages = self._make_stages()
        try:
            # Configure stages for remediation mode
            stage1 = stages[0]
            stage2 = stages[1]
            assert isinstance(stage1, MemoryConsolidator)
            assert isinstance(stage2, TaskKnowledgeSync)
            self._configure_consolidator(
                stage1, tier,
                remediation_findings=findings,
                filtered_task_tree=remediation_tree,
            )
            self._configure_task_sync(stage2, filtered_task_tree=remediation_tree, remediation_mode=True)

            watermark = await self.journal.get_watermark(project_id)
            reports = []
            for stage in stages:
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

            # Cross-check self-reported stats against write-journal ops before
            # the judge reads them (same as run_full_cycle).
            await verify_and_rewrite_stats(
                run_id, run.stage_reports, self.journal.write_journal,
            )

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


# ── Backlog iteration ──────────────────────────────────────────────────


class BacklogIterator:
    """Processes large backlogs in token-budgeted chunks, oldest-first."""

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
        """Process backlog in token-budgeted chunks, oldest-first.

        Uses peek → assemble → drain: peeks at buffered events, builds a
        token-budgeted payload via ContextAssembler, then drains only the
        events that fit.  Stops when no events remain before the cutoff.
        """
        from fused_memory.reconciliation.context_assembler import ContextAssembler

        opus_tier = TierConfig(
            model=self.config.opus_model,
            episode_limit=self.config.opus_episode_limit,
            memory_limit=self.config.opus_memory_limit,
        )

        # Snapshot: only process events that existed when we started.
        cutoff = datetime.now(UTC)

        # Extract project_root from first event (for taskmaster calls)
        peeked_first = await self.buffer.peek_buffered(project_id, limit=1, before=cutoff)
        project_root = project_id
        if peeked_first:
            project_root = peeked_first[0].payload.get('_project_root', project_id)

        assembler = ContextAssembler(
            memory_service=self.harness.memory,
            taskmaster=self.harness.taskmaster,
            config=self.config,
            project_root=project_root,
        )

        watermark = await self.journal.get_watermark(project_id)

        chunk_num = 0
        while True:
            # Peek at up to 1000 events (far more than a single budget can hold)
            peeked = await self.buffer.peek_buffered(
                project_id, limit=1000, before=cutoff,
            )
            if not peeked:
                break

            # Assemble token-budgeted payload
            assembled = await assembler.assemble(peeked, watermark, project_id)
            if not assembled.events:
                break

            # Drain exactly the events that fit the budget
            event_ids = [e.id for e in assembled.events]
            await self.buffer.drain_by_ids(project_id, event_ids)

            chunk_num += 1
            chunk_id = str(uuid4())
            await self.journal.record_chunk_boundary(
                project_id, chunk_id, len(assembled.events),
            )

            logger.info(
                f'Backlog chunk {chunk_num}: processing {len(assembled.events)} events '
                f'({assembled.total_tokens} tokens, {len(assembled.context_items)} context items, '
                f'{assembled.events_remaining} remaining) for {project_id}'
            )

            try:
                await asyncio.wait_for(
                    self.harness.run_full_cycle(
                        project_id, f'backlog_chunk:{chunk_num}:{len(assembled.events)}',
                        tier=opus_tier,
                        events=assembled.events,
                        assembled_payload=assembled,
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
