"""Pipeline orchestrator — runs the three-stage reconciliation pipeline."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import traceback
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from shared.cli_invoke import AllAccountsCappedException
from shared.usage_gate import UsageGate

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.reconciliation import (
    AssembledPayload,
    ReconciliationEvent,
    ReconciliationRun,
    RunStatus,
    RunType,
    StageId,
)
from fused_memory.models.scope import build_known_projects_map
from fused_memory.reconciliation.backlog_policy import BacklogPolicy
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.judge import Judge
from fused_memory.reconciliation.mem0_dedup import find_prior_memory
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)
from fused_memory.reconciliation.stats_verifier import verify_and_rewrite_stats
from fused_memory.reconciliation.task_filter import FilteredTaskTree, filter_task_tree
from fused_memory.services.live_workflow_detector import is_workflow_live_for_task
from fused_memory.services.memory_service import MemoryService

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol

try:
    from escalation.dedupe import (  # type: ignore[import-untyped]
        DedupeConfig,
        compute_content_fingerprint,
        submit_or_dedupe,
    )
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]
    from escalation.server import (  # type: ignore[import-untyped]
        create_server as create_escalation_server,
    )
    HAS_ESCALATION = True
except ImportError:
    HAS_ESCALATION = False

# Recon-wide dedup config: covers all four recon escalation categories.
# Wider than DedupeConfig.for_recon() (which only covers recon_integrity_issue)
# because A7b also folds non-finding categories so each DISTINCT recurring message
# files once — see design_decisions in plan.json for rationale.
# Set to None when escalation package is not installed; _escalate checks
# HAS_ESCALATION before using it.
#
# Escalation-closure contract (A7b / plans/afk-A7-recon-closure.md):
#   - The reconciliation harness NEVER calls queue.resolve().
#   - The watcher session (port 8103) is the sole closer of recon escalations.
#   - Dedup folds on the way IN only, via submit_or_dedupe + _RECON_DEDUP_CONFIG.
#   See ReconciliationHarness._escalate() docstring for per-call-site details.
_RECON_DEDUP_CONFIG = (
    dataclasses.replace(
        DedupeConfig.for_recon(),  # type: ignore[possibly-undefined]
        infra_dedupe_categories=(
            'recon_integrity_issue',
            'recon_failure',
            'recon_stale_run',
            'recon_backlog_overflow',
        ),
    )
    if HAS_ESCALATION else None
)

logger = logging.getLogger(__name__)

# Task 1512: minimum number of completed runs that must contain a finding
# before it is escalated from _run_remediation_pass.  Below this count the
# finding is suppressed (emitting a structured log instead) on the grounds
# that remediation is likely to fix it on the next attempt.
#
# Counting note: each reconciliation cycle produces TWO completed runs — the
# parent full run (persisted at run_full_cycle ~line 1065) and the remediation
# run (persisted at _run_remediation_pass ~line 1405).  Both are counted by
# _finding_persistence_count because both are 'completed' journal rows.  A
# threshold of 4 therefore fires after 2 complete reconciliation cycles that
# both flagged the same finding — short enough that a genuinely broken finding
# escalates within a watchable window (≈ 10–180 minutes depending on cycle
# duration), long enough to filter transient findings.
_INTEGRITY_FINDING_RECURRENCE_THRESHOLD = 4


def _derive_affected_ids(finding: dict) -> list[str]:
    """Derive an affected-ids-equivalent list from a finding's typed citations.

    The recon_report cutover (task γ) retired the free-form ``affected_ids``
    field in favour of typed citation lists — ``cited_entities`` ({entity_uuid,
    canonical_name}), ``cited_edges`` ({edge_uuid, ...}), ``cited_tasks``
    ({project_id, task_id, ...}), ``cited_memories`` ({memory_id, ...}).  This
    helper flattens those typed citations back into a flat list of identity
    strings so the escalation dedup/recurrence fingerprint
    (``compute_content_fingerprint``) and the log/detail payloads keep a stable,
    structured identity component instead of degrading to description-only.

    A legacy ``affected_ids`` field takes precedence when present, so cross-run
    recurrence counting still works against pre-cutover journal rows that carry
    the old shape.
    """
    legacy = finding.get('affected_ids')
    if legacy:
        return [str(a) for a in legacy]
    ids: list[str] = []
    for c in finding.get('cited_tasks') or []:
        tid = c.get('task_id') if isinstance(c, dict) else None
        if tid:
            ids.append(str(tid))
    for c in finding.get('cited_entities') or []:
        if isinstance(c, dict):
            val = c.get('canonical_name') or c.get('entity_uuid')
            if val:
                ids.append(str(val))
    for c in finding.get('cited_edges') or []:
        eid = c.get('edge_uuid') if isinstance(c, dict) else None
        if eid:
            ids.append(str(eid))
    for c in finding.get('cited_memories') or []:
        mid = c.get('memory_id') if isinstance(c, dict) else None
        if mid:
            ids.append(str(mid))
    return ids


# Module-local sleep binding — allows tests to patch sleep without touching
# the global asyncio namespace.
_sleep = asyncio.sleep


class UnknownProjectError(ValueError):
    """Raised when a project_id has no entry in the KNOWN_PROJECT_ROOTS registry.

    Introduced by task 1143 as the pre-flight cross-contamination guard.

    Design notes:
    (a) Signals a KNOWN_PROJECT_ROOTS misconfiguration — the project_id is not
        registered via ``taskmaster.project_root`` or ``DASHBOARD_KNOWN_PROJECT_ROOTS``.
    (b) Subclasses ``ValueError`` for backward-compat: any existing or future
        ``except ValueError`` callsite (test code, callers of
        ``_known_project_root_for``) continues to match.
    (c) Exists as a distinct type so ``_project_loop`` can narrowly catch ONLY
        misconfiguration and let unrelated ``ValueError``s fall through to the
        existing ``except Exception`` retry path:
        - ``stages/base.py:108`` watermark↔stage project_id mismatch — transient
          during instance handover; naturally recoverable on the next cycle.
        - ``stages/memory_consolidator.py:96`` unset episode_limit/memory_limit —
          programming bug; should surface and retry, not abort the project loop.
    """


@dataclass
class TierConfig:
    """Model tier configuration for a reconciliation cycle."""

    model: str = 'sonnet'
    episode_limit: int = 125
    memory_limit: int = 250


def build_stale_run_diagnostics(
    run: ReconciliationRun,
    lock_holder: str | None,
    lock_age: float | None,
    cutoff: float,
) -> dict:
    """Return a diagnostic dict for a stale run's lock disposition.

    Classifies the disposition into one of five cases:

    - ``pre_migration``       — run.instance_id is None (legacy row)
    - ``no_lock``             — no lock row exists for the project
    - ``handed_off``          — lock held by a different instance
    - ``dead_owner_shielded`` — same iid as run, heartbeat older than cutoff
    - ``live``                — same iid as run, heartbeat fresh (within cutoff)

    Also includes: project_id, run_type (str), instance_id, age_seconds
    (seconds since run.started_at), lock_holder, lock_heartbeat_age.
    """
    now = datetime.now(UTC)
    # Normalize naive datetimes before subtracting — mirrors the guard in get_lock_status.
    started = run.started_at if run.started_at.tzinfo else run.started_at.replace(tzinfo=UTC)
    age_seconds = (now - started).total_seconds()

    if run.instance_id is None:
        disposition = 'pre_migration'
    elif lock_holder is None:
        disposition = 'no_lock'
    elif lock_holder != run.instance_id:
        disposition = 'handed_off'
    elif lock_age is not None and lock_age > cutoff:
        # Same iid, but heartbeat is older than the liveness threshold.
        disposition = 'dead_owner_shielded'
    else:
        # Same iid, heartbeat is fresh — owner is still alive.
        disposition = 'live'

    return {
        'disposition': disposition,
        'project_id': run.project_id,
        'run_type': str(run.run_type.value) if isinstance(run.run_type, RunType) else str(run.run_type),
        'instance_id': run.instance_id,
        'age_seconds': age_seconds,
        'lock_holder': lock_holder,
        'lock_heartbeat_age': lock_age,
    }


class ReconciliationHarness:
    """Orchestrates the three-stage reconciliation pipeline."""

    def __init__(
        self,
        memory_service: MemoryService,
        taskmaster: TaskBackendProtocol | None,
        journal: ReconciliationJournal,
        event_buffer: EventBuffer,
        config: FusedMemoryConfig,
        backlog_policy: BacklogPolicy | None = None,
        known_projects: dict[str, str] | None = None,
        recon_report_state=None,
        server_ready_event: asyncio.Event | None = None,
    ):
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.journal = journal
        self.buffer = event_buffer
        # PRD γ recon_report threading: capture port and state before config is
        # narrowed to config.reconciliation so stages can reach the full config values.
        self._recon_report_state = recon_report_state
        # Reviewer finding race_condition: run_loop() awaits this event before the
        # first stage subprocess fires, ensuring the recon-report MCP server is
        # accepting connections.  None when no recon_report server is started.
        self._server_ready_event = server_ready_event
        self._recon_report_port: int = config.server.recon_report_port
        _raw_root = config.taskmaster.project_root if config.taskmaster else ''
        if _raw_root:
            _raw_root = os.path.expanduser(_raw_root)
        self._project_root = str(Path(_raw_root).resolve()) if _raw_root else ''
        self.config = config.reconciliation
        self._backlog_policy = backlog_policy
        # Multi-project routing: build a {project_id → project_root} map from
        # the configured project + DASHBOARD_KNOWN_PROJECT_ROOTS env var so
        # Stage 2 can surface other projects to its LLM and authorise
        # cross-project task filing.  Reuses the dashboard's existing env var
        # to avoid two sources of truth — see plans/deep-squishing-lagoon.md
        # for the trade-off (rename to a neutral name is a tracked followup).
        # When a known_projects kwarg is supplied (server startup), the harness
        # uses that snapshot instead of building its own — single source of
        # truth across harness, ticket janitor, and path-scope guard (task 1164).
        self._known_projects: dict[str, str] = (
            dict(known_projects) if known_projects is not None
            else build_known_projects_map(self._project_root)
        )
        # WP-D: track which halted projects we've already escalated so we
        # don't re-fire every harness tick.
        self._halt_escalated: set[str] = set()

        # Usage gate (multi-account cap failover)
        self.usage_gate: UsageGate | None = None
        if hasattr(self.config, 'usage_cap') and self.config.usage_cap.enabled:
            self.usage_gate = UsageGate(self.config.usage_cap)

        # Build stages — thread recon_report_port and recon_report_state so BaseStage.run
        # can call start_report / get_assembled_report directly (PRD γ, task 1546).
        stage1 = MemoryConsolidator(
            StageId.memory_consolidator, memory_service, taskmaster, journal, self.config,
            usage_gate=self.usage_gate,
            recon_report_port=self._recon_report_port,
            recon_report_state=self._recon_report_state,
        )

        stage2 = TaskKnowledgeSync(
            StageId.task_knowledge_sync, memory_service, taskmaster, journal, self.config,
            usage_gate=self.usage_gate,
            recon_report_port=self._recon_report_port,
            recon_report_state=self._recon_report_state,
        )

        stage3 = IntegrityCheck(
            StageId.integrity_check, memory_service, taskmaster, journal, self.config,
            usage_gate=self.usage_gate,
            recon_report_port=self._recon_report_port,
            recon_report_state=self._recon_report_state,
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
        # One-shot gate shared by drain() and run_loop()'s drain-status block.
        # Whichever site fires the 'Harness fully drained' marker first sets this
        # to True so subsequent drain() calls and main-loop iterations are silent.
        # Lifecycle is tied to process lifetime: there is no undrain path today,
        # but if one is ever added it MUST reset _drain_complete_logged together
        # with _draining, otherwise the next drain will silently fail to re-emit
        # the 'Harness fully drained' marker.
        self._drain_complete_logged: bool = False

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

    @property
    def project_root(self) -> str:
        """Configured project root (from taskmaster config).

        Returns ``''`` when no taskmaster config is present.  Callers that need
        the canonical root for a *project_id* should use
        ``_known_project_root_for(project_id)`` instead (task 1143).
        """
        return self._project_root

    def _known_project_root_for(self, project_id: str) -> str:
        """Return the canonical project_root for *project_id* from the registry.

        This is the pre-flight cross-contamination guard introduced by task 1143.
        It looks up ``self._known_projects`` (populated at init from the configured
        taskmaster root + ``DASHBOARD_KNOWN_PROJECT_ROOTS`` env var) and raises
        ``ValueError`` if no entry exists.  The error message includes both the
        unrecognised project_id *and* the sorted list of registered project_ids so
        the operator can immediately attribute and fix a misconfiguration.

        Raising here — before any journal or buffer side-effects — ensures that a
        missing registry entry never causes a partial cycle (events drained, journal
        row created, then a stage failure that leaves those events unrecoverable).

        Args:
            project_id: Canonical project identifier (e.g. ``'reify'``,
                ``'autopilot_video'``).  Must match a key in
                ``self._known_projects`` (derived from path basename,
                lowercase, dashes to underscores).

        Returns:
            The absolute project_root path for *project_id*.

        Raises:
            UnknownProjectError: (a ``ValueError`` subclass) If *project_id* is not in ``self._known_projects``.
        """
        try:
            return self._known_projects[project_id]
        except KeyError:
            known_sorted = sorted(self._known_projects)
            raise UnknownProjectError(
                f'reconciliation: project_id {project_id!r} has no entry in '
                f'KNOWN_PROJECT_ROOTS (known: {known_sorted}). '
                f'Set DASHBOARD_KNOWN_PROJECT_ROOTS env var or add a '
                f'TaskmasterConfig.project_root that resolves to a known project.'
            ) from None

    def drain(self) -> None:
        """Signal the harness to stop starting new reconciliation cycles.

        Currently-running project loops complete their current cycle.
        The server continues serving reads/writes — only new cycles are suppressed.

        If no project loops are active at the moment of the call, emits
        ``'Harness fully drained — safe to restart'`` synchronously so that
        ``scripts/restart-fused-memory.sh --drain`` detects completion in <5 s
        rather than waiting up to 120 s for the main loop to observe idle state.
        When at least one loop is still in flight the marker is suppressed here;
        the main reconciliation loop emits it after all loops finish (existing
        behaviour, task 1053).
        """
        if self._draining:
            logger.info('Harness already draining')
            return
        self._draining = True
        logger.info('Harness drain requested — will finish current cycles, no new ones')
        # Task 1053: short-circuit when no project loops are active at drain time.
        # Delegates to _no_active_loops() so drain(), is_drained, and the main-loop
        # emission site (harness.py ~line 591) stay semantically coupled.
        # When at least one loop is still running, the main loop emits the marker
        # after all loops finish (existing behaviour, up to the 120-s timeout).
        if self._no_active_loops() and not self._drain_complete_logged:
            self._drain_complete_logged = True
            logger.info('Harness fully drained — safe to restart')

    def _no_active_loops(self) -> bool:
        """Return True when no project loops are currently running.

        A task is considered active when ``.done()`` returns False.
        Shared by :meth:`drain`, :attr:`is_drained`, and the main reconciliation
        loop so the three sites stay semantically coupled (task 1053).
        """
        return all(t.done() for t in self._project_tasks.values())

    @property
    def is_drained(self) -> bool:
        """True when draining and all project loops have completed."""
        return self._draining and self._no_active_loops()

    def _make_stages(self) -> list:
        """Create a fresh set of stage instances for one reconciliation cycle."""
        stage1 = MemoryConsolidator(
            StageId.memory_consolidator, self.memory, self.taskmaster, self.journal,
            self.config, usage_gate=self.usage_gate,
            recon_report_port=self._recon_report_port,
            recon_report_state=self._recon_report_state,
        )
        stage2 = TaskKnowledgeSync(
            StageId.task_knowledge_sync, self.memory, self.taskmaster, self.journal,
            self.config, usage_gate=self.usage_gate,
            recon_report_port=self._recon_report_port,
            recon_report_state=self._recon_report_state,
        )
        stage3 = IntegrityCheck(
            StageId.integrity_check, self.memory, self.taskmaster, self.journal,
            self.config, usage_gate=self.usage_gate,
            recon_report_port=self._recon_report_port,
            recon_report_state=self._recon_report_state,
        )
        stages = [stage1, stage2, stage3]
        self._propagate_escalation_queue(stages)
        return stages

    def _propagate_escalation_queue(self, stages: Iterable[Any]) -> None:
        """Apply harness's escalation URL and queue to each stage in *stages*.

        Shared by ``_make_stages`` (cold path: fresh stages each cycle) and
        ``_start_escalation_server`` (hot path: stages pre-built in __init__).
        Defensive: only assigns when the harness has a value, so cold-path
        callers before escalation startup leave stages untouched.

        Single-pass over *stages* so single-pass iterables (generators, ``iter(...)``)
        work correctly — the prior two-pass form would silently skip the queue
        assignment once the URL pass exhausted the iterator.
        """
        for s in stages:
            if self._escalation_url:
                s._escalation_url = self._escalation_url
            if self._escalation_queue is not None:
                s._escalation_queue = self._escalation_queue

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
            project_root is empty, non-absolute, or the fetch fails.
            Non-absolute paths are rejected before calling taskmaster to avoid
            silent failures from the backend's absolute-path validator.

        Log-level policy:
            Anomaly branches (taskmaster disabled, empty/non-absolute project_root)
            always log at INFO.  Successful fetches log at DEBUG unless
            ``raw_count > 0 and total_count == 0``, which indicates the filter
            silently dropped every task (e.g. all task entries are non-dict and
            skipped by the defensive isinstance guard in filter_task_tree —
            bare ints, malformed entries) and warrants an INFO-level alert.
            Healthy-but-empty projects (``raw_count == 0``) are classified as
            DEBUG (non-anomalous).
        """
        if not self.taskmaster:
            logger.info(
                'reconciliation.task_tree_taskmaster_disabled',
                extra={'project_root': project_root},
            )
            return FilteredTaskTree()
        if not project_root:
            logger.info(
                'reconciliation.task_tree_empty_project_root',
                extra={'project_root_repr': repr(project_root)},
            )
            return FilteredTaskTree()
        if not os.path.isabs(project_root):
            logger.warning(
                '_fetch_filtered_task_tree rejected non-absolute project_root %r'
                ' — cannot fetch tasks; Stage 2 will see empty tree',
                project_root,
            )
            return FilteredTaskTree()
        try:
            tasks_data = await self.taskmaster.get_tasks(project_root=project_root)
            raw_count = len(tasks_data.get('tasks', []))
            filtered = filter_task_tree(tasks_data)
            # Anomaly predicate is intentionally narrow (task-985 policy): only the
            # full-drop case (every top-level entry skipped by filter_task_tree's
            # defensive isinstance guard) is escalated to INFO.  Partial drops
            # (raw_count >> total_count when only some entries are skipped) remain at
            # DEBUG by design — the structured fields raw_count/total_count carry the
            # differentiating signal, and operators can grep DEBUG when investigating.
            # Broadening to a `raw_count > N * total_count` heuristic was considered and
            # rejected because total_count is post-flatten (includes subtasks) while
            # raw_count is the top-level pre-flatten count, making any threshold
            # semantically muddy without exposing a new skipped_count field from
            # filter_task_tree.
            is_anomaly = raw_count > 0 and filtered.total_count == 0
            logger.log(
                logging.INFO if is_anomaly else logging.DEBUG,
                'reconciliation.task_tree_fetched',
                extra={
                    'project_root': project_root,
                    'raw_count': raw_count,
                    'total_count': filtered.total_count,
                },
            )
            return filtered
        except Exception as exc:
            logger.warning(
                f'_fetch_filtered_task_tree failed for {project_root!r}: {exc}'
            )
            return FilteredTaskTree()

    # ── Stale-run recovery ─────────────────────────────────────────────

    async def _recover_stale_runs(self) -> None:
        """Find runs stuck in 'running' state and mark them failed.

        Uses stale_run_recovery_seconds as the age cutoff (default 1800s),
        then double-checks that the *same* instance still holds the project's
        reconciliation lock before skipping — protecting legitimately long-running
        cycles owned by this process while still recovering orphans whose owning
        instance is dead even when a fresh instance has since acquired the lock.

        Pre-migration runs (instance_id IS NULL) are recovered unconditionally
        — the bug this guards against was exactly stale rows being shielded
        forever by a new instance's lock acquisition.
        """
        cutoff = self.config.stale_run_recovery_seconds
        stale_runs = await self.journal.get_stale_runs(cutoff)
        for run in stale_runs:
            # Skip only when the *same* instance that started the run still
            # holds the lock AND the lock is freshly heartbeated (within cutoff).
            # A dead owner's lock row satisfies the identity check but its
            # heartbeat_at will be > cutoff seconds old — treat that as an orphan.
            # A lock held by a different instance, or no lock at all, also means
            # the original owner is gone and the run is an orphan.
            lock_holder, lock_age = await self.buffer.get_lock_status(run.project_id)
            if (
                lock_holder is not None
                and run.instance_id is not None
                and lock_holder == run.instance_id
                and lock_age is not None
                and lock_age <= cutoff
            ):
                continue

            diag = build_stale_run_diagnostics(run, lock_holder, lock_age, cutoff)
            logger.warning(
                f'Recovering stale run {run.id} for {run.project_id} '
                f'(started {run.started_at.isoformat()}, lock expired, '
                f'instance={run.instance_id}, disposition={diag["disposition"]})'
            )
            run.stage_reports['_error'] = {
                'error_type': 'StaleRunRecovery',
                'error_message': f'Run stale (>{cutoff}s, lock expired), recovered by harness',
                'failed_stage': None,
                **diag,
            }
            await self.journal.update_run_stage_reports(run.id, run.stage_reports)
            await self.journal.complete_run(run.id, 'failed')
            restored = await self.buffer.restore_drained(run.project_id)
            if restored:
                logger.info(f'Restored {restored} drained events for stale run {run.id}')
            # Ownership-scoped release: see plans/recon-stale-recovery-rca.md.
            # Releasing without an instance_id filter would strip a live cycle's
            # lock on the same project, causing the next reaper tick to
            # misclassify the live run as stale (cross-instance lock theft, the
            # 2026-05-28 false-positive cascade).  Only release when the
            # orphan's instance_id is known AND owns the current lock — the
            # `continue` above guarantees we're not on the "same live instance"
            # branch, so a match here means a defunct previous incarnation of
            # this very instance left the row behind.  Genuinely-abandoned
            # locks held by other dead instances are cleaned up by the
            # heartbeat-staleness sweep (stale_lock_seconds = 7200s) inside
            # mark_run_active / get_lock_holder_instance_id.
            if (
                run.instance_id is not None
                and lock_holder == run.instance_id
            ):
                await self.buffer.mark_run_complete(
                    run.project_id, instance_id=run.instance_id,
                )
            await self._replay_deferred_writes(run.project_id)
            detail = (
                f"project={diag['project_id']} run_type={diag['run_type']} "
                f"instance={diag['instance_id']} age={diag['age_seconds']:.0f}s "
                f"disposition={diag['disposition']}"
            )
            self._escalate(
                'recon_stale_run',
                run.id,
                f'Run stale (>{cutoff}s, lock expired), recovered',
                detail,
            )

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
            meta = write['metadata'] or {}
            tid = meta.get('task_id')
            transition = meta.get('transition')

            # Dedup check: skip completion-summary writes that already exist in Mem0.
            # Only for transition='done' writes — other transitions are left as-is.
            # find_prior_memory degrades gracefully: search failures log a WARNING
            # under logger and return None so the write proceeds normally.
            if transition == 'done' and tid:
                prior = await find_prior_memory(
                    self.memory,
                    project_id=project_id,
                    task_id=tid,
                    kind={'transition': 'done'},
                    query=f'task {tid} targeted_reconciliation completion done',
                    # categories pinned to match the writer in targeted.py:_on_task_done
                    # (TargetedReconciliation writes completion summaries with
                    # transition='done' under 'observations_and_summaries'). Keep these
                    # in sync — a future writer using a different category would silently
                    # bypass this dedup.
                    categories=['observations_and_summaries'],
                    limit=20,
                    log=logger,
                )
                if prior is not None:
                    logger.info(
                        'Skipping deferred completion-summary for task %s — already written',
                        tid,
                    )
                    await self.buffer.delete_deferred_write(write['id'])
                    continue

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
        await _sleep(0.5)

        # Store escalation URL and queue for _make_stages() and set on existing stages
        escalation_url = f'http://{host}:{port}/mcp'
        self._escalation_url = escalation_url
        # Both _escalation_url and _escalation_queue are set above, so helper will propagate.
        self._propagate_escalation_queue(self.stages)

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
        *,
        finding: dict | None = None,
    ) -> None:
        """Submit an escalation to the queue (fire-and-forget).

        Routing contract (A7b):
        - The harness NEVER calls queue.resolve() — the escalation-watcher
          session (port 8103) is the sole closer per plans/afk-A7-recon-closure.md.
        - Dedup folds only on the way IN, via submit_or_dedupe + _RECON_DEDUP_CONFIG.
        - When finding is not None (a finding dict with category / affected_ids /
          description), the fingerprint is keyed on finding identity so the same
          target across N cycles produces exactly one pending escalation.
        - When finding is None, the fingerprint falls back to a description-only
          hash of the summary, so identical recurring messages fold while distinct
          ones stay individually visible.
        """
        if not HAS_ESCALATION or self._escalation_queue is None:
            return
        try:
            queue = self._escalation_queue
            if finding is not None:
                fingerprint = compute_content_fingerprint(  # type: ignore[possibly-undefined]
                    category,
                    finding.get('category') or '',
                    _derive_affected_ids(finding),
                    finding.get('description') or '',
                )
            else:
                # No finding in scope: use '' for finding_category (sentinel for
                # "summary-only, no finding identity") so the description-hash branch
                # of compute_content_fingerprint is used.  Using '' instead of
                # re-using escalation_category keeps the identity composition
                # semantically accurate — the second arg is the finding's own
                # category, which is absent here.
                fingerprint = compute_content_fingerprint(  # type: ignore[possibly-undefined]
                    category,
                    '',
                    [],
                    summary,
                )
            esc = Escalation(  # type: ignore[possibly-undefined]
                id=queue.make_id(f'recon-{run_id[:8]}'),
                task_id=f'recon-{run_id[:8]}',
                agent_role='reconciliation-harness',
                severity='info' if category in ('recon_stale_run', 'recon_integrity_issue') else 'blocking',
                category=category,
                summary=summary,
                detail=detail,
                dedupe_fingerprint=fingerprint,
            )
            submit_or_dedupe(queue, esc, _RECON_DEDUP_CONFIG)  # type: ignore[possibly-undefined]
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
        # Reviewer finding race_condition: wait for the recon-report server to be
        # accepting connections before the first stage subprocess is launched.  This
        # encodes the ordering invariant in code rather than relying on comment-only
        # guarantees.  server_ready_event is None when recon_report is not active.
        if self._server_ready_event is not None:
            logger.debug('Harness waiting for recon-report server readiness signal...')
            await self._server_ready_event.wait()
            logger.debug('Recon-report server ready — starting reconciliation loop')
        if self.usage_gate:
            await self.usage_gate.check_at_startup()

        # Rehydrate persistent halt state from the journal so a restart cannot
        # silently clear a halt that hasn't been explicitly cleared by an
        # operator. Called once at startup.
        if self.judge is not None:
            await self.judge.initialize()

        await self._start_escalation_server()

        # Re-queue any deferred writes left in-progress by a crashed prior process.
        # Cutoff is 0 (release every currently-claimed row) rather than a
        # time-based horizon.  A time-based horizon would miss rows whose
        # claimed_at falls within that horizon on a fast restart, silently
        # stalling those writes.
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
                        if self._no_active_loops():
                            if not self._drain_complete_logged:
                                self._drain_complete_logged = True
                                logger.info('Harness fully drained — safe to restart')
                            # else: silent — drained marker already emitted
                        else:
                            active = sum(
                                1 for t in self._project_tasks.values() if not t.done()
                            )
                            logger.info(
                                f'Harness draining: {active} project loop(s) still running'
                            )

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
                await _sleep(5)
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
                    await _sleep(5)
                    continue

                idle_ticks = 0
                acquired = await self.buffer.mark_run_active(project_id)
                if not acquired:
                    await _sleep(5)
                    continue

                # Halt check
                if self.judge and self.judge.is_halted(project_id):
                    logger.warning(f'Skipping cycle for halted project {project_id}')
                    await self._notify_judge_halt(
                        project_id, reason='judge halted reconciliation',
                    )
                    try:
                        await self._replay_deferred_writes(project_id)
                    finally:
                        # Scope the release to this instance — see
                        # plans/recon-stale-recovery-rca.md.
                        await self.buffer.mark_run_complete(
                            project_id, instance_id=self.buffer.instance_id,
                        )
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
                    try:
                        await self._replay_deferred_writes(project_id)
                    finally:
                        # Cancel heartbeat only after replay so the lock heartbeat
                        # keeps the per-project lock alive for the full replay
                        # duration, preventing a concurrent instance from claiming
                        # the lock as stale mid-replay.
                        heartbeat_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await heartbeat_task
                        # Scope the release to this instance — see
                        # plans/recon-stale-recovery-rca.md.
                        await self.buffer.mark_run_complete(
                            project_id, instance_id=self.buffer.instance_id,
                        )

            except asyncio.CancelledError:
                raise  # Propagate shutdown
            except UnknownProjectError as e:
                # task 1143: KNOWN_PROJECT_ROOTS misconfiguration — fail fast, no retry.
                # Catching the narrow UnknownProjectError (not bare ValueError) ensures
                # generic ValueErrors from stages fall through to the except Exception
                # retry path below:
                #   - stages/base.py: watermark↔stage project_id mismatch (transient
                #     during instance handover; naturally recoverable next cycle)
                #   - stages/memory_consolidator.py: unset episode_limit/memory_limit
                #     (programming bug; should surface and retry, not abort the loop)
                #
                # task 1549: quarantine buffered events so get_active_projects()
                # (WHERE status='buffered') stops returning this project_id, ending
                # the management-loop respawn storm (2026-05-28 'know-live' incident).
                # The dead_letter flip + WARNING below make the incident observable to
                # the recon-watcher; no escalation is emitted (deferred per design).
                # See task 1143 (read-side strictness) and task 1549 (this complement).
                quarantined = await self.buffer.mark_project_dead_letter(project_id)
                logger.warning(
                    f'Project loop quarantined {quarantined} buffered dead_letter event(s) '
                    f'for unknown project_id={project_id!r} and is aborting. '
                    f'Misconfiguration: {e}. '
                    f'Set DASHBOARD_KNOWN_PROJECT_ROOTS to register this project '
                    f'(task 1143 / task 1549).'
                )
                return
            except Exception as e:
                logger.error(f'Project loop error for {project_id}: {e}')

            await _sleep(5)  # Cooldown between cycles

    async def _heartbeat_loop(self, project_id: str) -> None:
        """Keep the reconciliation lock alive while a run is in progress."""
        while True:
            await _sleep(60)
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
        # task 1143: pre-flight guard — raises before any side effects (no journal row,
        # no buffer drain) if project_id has no KNOWN_PROJECT_ROOTS entry.
        project_root = self._known_project_root_for(project_id)

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
            instance_id=self.buffer.instance_id,
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

        # project_root is already hard-bound via _known_project_root_for at the top
        # of this function (task 1143); no _resolve_project_root call needed here.

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
                stage.known_projects = self._known_projects

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

                # Wire harness-fetched task tree into Stage 3 for task-dump spot-check (task 1661)
                if isinstance(stage, IntegrityCheck):
                    stage.filtered_task_tree = filtered_task_tree

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

            # Remediation pass: thread project_root resolved above (task 1163) and pass
            # pre-fetched tree to avoid a redundant fetch (ref: task 478).
            await self._maybe_remediate(project_id, run_id, run, tier,
                                        project_root=project_root,
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
        except AllAccountsCappedException as e:
            run.status = RunStatus.failed
            run.stage_reports['_error'] = {
                'error_type': 'AllAccountsCappedException',
                'error_message': str(e),
                'failed_stage': current_stage_name,
                'deferred': True,
            }
            await self.journal.complete_run(run_id, 'failed')
            await self.buffer.restore_drained(project_id)
            logger.warning(
                f'Reconciliation deferred: all accounts capped during stage '
                f'{current_stage_name} ({e.retries} retries in {e.elapsed_secs:.1f}s)'
            )
            return run
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

    async def _finding_persistence_count(
        self,
        project_id: str,
        finding: dict,
        lookback: int = max(5, _INTEGRITY_FINDING_RECURRENCE_THRESHOLD),
    ) -> int:
        """Count how many of the last `lookback` completed runs contain a matching finding.

        Identity is determined by compute_content_fingerprint (same key as _escalate uses
        for dedup), so findings that differ only in non-identity fields do not inflate the
        count.  Counted per-run — duplicates within one run's items_flagged do not count
        twice.

        The default lookback is ``max(5, _INTEGRITY_FINDING_RECURRENCE_THRESHOLD)`` so
        that raising the threshold never silently caps the count below the gate value.

        Returns 0 if HAS_ESCALATION is False, if the fingerprint call raises, or if any
        other exception occurs (graceful degradation).  When the journal call itself
        fails, a warning is logged so debugging a 'why didn't this escalate?' incident
        is possible — the caller proceeds as if persistence is 0 (fail-closed).

        Task 1512 / plans/afk-A7-recon-closure.md.
        """
        if not HAS_ESCALATION:
            return 0
        try:
            target_fp = compute_content_fingerprint(  # type: ignore[possibly-undefined]
                'recon_integrity_issue',
                finding.get('category') or '',
                _derive_affected_ids(finding),
                finding.get('description') or '',
            )
        except Exception:
            return 0
        try:
            recent = await self.journal.get_recent_runs(project_id, limit=lookback)
            count = 0
            for run in recent:
                if run.status != 'completed':
                    continue
                s3_report = run.stage_reports.get('integrity_check')
                if s3_report is None:
                    continue
                if isinstance(s3_report, dict):
                    items = s3_report.get('items_flagged', []) or []
                else:
                    items = s3_report.items_flagged or []
                # Count this run once if any item matches the target fingerprint
                for item in items:
                    try:
                        fp = compute_content_fingerprint(  # type: ignore[possibly-undefined]
                            'recon_integrity_issue',
                            item.get('category') or '',
                            _derive_affected_ids(item),
                            item.get('description') or '',
                        )
                    except Exception:
                        continue
                    if fp == target_fp:
                        count += 1
                        break  # one match per run is enough
            return count
        except Exception as _e:
            logger.warning(
                'reconciliation.persistence_count_failed',
                extra={
                    'project_id': project_id,
                    'finding_category': finding.get('category', ''),
                    'error': str(_e),
                },
            )
            return 0

    def _log_non_actionable_finding(
        self,
        project_id: str,
        run_id: str,
        finding: dict,
    ) -> None:
        """Emit a structured INFO log for a non-actionable Stage-3 integrity finding.

        Non-actionable findings are suppressed from the escalation queue (task 1512 /
        plans/afk-A7-recon-closure.md): the only possible human action — "accept as
        known" — is achieved by *not* filing.  This helper keeps the finding observable
        in the recon log/journal without touching the queue.

        Called from both _maybe_remediate (parent pass) and _run_remediation_pass (after
        the second-pass actionable partition) so both sites stay in sync as fields evolve.
        """
        logger.info(
            'reconciliation.non_actionable_integrity_finding',
            extra={
                'project_id': project_id,
                'run_id': run_id,
                'finding_category': finding.get('category', ''),
                'affected_ids': _derive_affected_ids(finding),
                'description': finding.get('description', ''),
                'severity': finding.get('severity', ''),
            },
        )

    def _finding_has_open_escalation(
        self,
        finding: dict,
        pending_fps: set[str] | None = None,
    ) -> bool:
        """Return True iff an OPEN pending escalation already covers this finding.

        Uses the same compute_content_fingerprint key as _escalate and
        _finding_persistence_count (category='recon_integrity_issue', finding
        category, _derive_affected_ids, description) to guarantee suppression
        is consistent with what _escalate would fold via submit_or_dedupe.

        pending_fps: pre-fetched set of dedupe_fingerprints for
        'recon_integrity_issue' pending escalations.  When supplied (built once
        by _maybe_remediate), the check is an O(1) set membership test — no
        extra disk scan.  Pass None to fall back to a full get_pending() scan
        per-finding (direct / unit-test use).

        Fail-open: any exception (or HAS_ESCALATION False / queue None) returns
        False so that a transient queue-read glitch costs at most one extra
        remediation pass rather than silently suppressing a needed pass.

        Task 1570 / FIX A.
        """
        if not HAS_ESCALATION or self._escalation_queue is None:
            return False
        try:
            target_fp = compute_content_fingerprint(  # type: ignore[possibly-undefined]
                'recon_integrity_issue',
                finding.get('category') or '',
                _derive_affected_ids(finding),
                finding.get('description') or '',
            )
            if pending_fps is not None:
                return target_fp in pending_fps
            # Fallback: per-finding scan (no pre-fetched set supplied).
            for e in self._escalation_queue.get_pending():
                if (
                    e.category == 'recon_integrity_issue'
                    and e.dedupe_fingerprint == target_fp
                ):
                    return True
            return False
        except Exception as _err:
            logger.warning(
                'reconciliation.open_escalation_check_failed',
                extra={
                    'finding_category': finding.get('category', ''),
                    'error': str(_err),
                },
            )
            return False  # fail-open: proceed with remediation

    async def _maybe_remediate(
        self,
        project_id: str,
        parent_run_id: str,
        parent_run: ReconciliationRun,
        tier: TierConfig,
        *,
        project_root: str,
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

            # Task 1512 / plans/afk-A7-recon-closure.md:
            # Non-actionable findings are NOT escalated.  Per the Stage-3 contract
            # they are already (a) persisted in stage_reports.integrity_check.items_flagged
            # and (b) forward-fed into the next cycle's S1/S2 via _get_prior_s3_findings.
            # The only possible human action — "accept as known" — is achieved by *not*
            # filing, so escalating them is a category error.  Instead, emit a structured
            # log record so the finding stays observable in the recon log/journal.
            for finding in non_actionable:
                self._log_non_actionable_finding(project_id, parent_run_id, finding)

            if not actionable:
                return

            # Task 1570 / FIX A: suppress remediation for any actionable finding
            # that already has an OPEN pending recon_integrity_issue escalation.
            # Performance: call get_pending() ONCE and build a fingerprint set
            # (single directory scan per cycle) rather than O(N findings) scans.
            # Fail-open: on any queue-read error emit a warning, treat pending_fps
            # as empty (no suppressions), and proceed with full remediation.
            pending_fps: set[str] = set()
            if HAS_ESCALATION and self._escalation_queue is not None:
                try:
                    pending_fps = {
                        e.dedupe_fingerprint
                        for e in self._escalation_queue.get_pending()
                        if e.category == 'recon_integrity_issue'
                        and e.dedupe_fingerprint is not None
                    }
                except Exception as _fps_err:
                    logger.warning(
                        'reconciliation.open_escalation_check_failed',
                        extra={
                            'finding_category': '',
                            'error': str(_fps_err),
                        },
                    )
                    # pending_fps stays empty → no suppressions → remediation
                    # proceeds normally (fail-open).

            suppressed: list[dict] = []
            to_remediate: list[dict] = []
            for finding in actionable:
                if self._finding_has_open_escalation(finding, pending_fps=pending_fps):
                    suppressed.append(finding)
                else:
                    to_remediate.append(finding)

            for finding in suppressed:
                logger.info(
                    'reconciliation.remediation_suppressed_open_escalation',
                    extra={
                        'project_id': project_id,
                        'parent_run_id': parent_run_id,
                        'finding_category': finding.get('category', ''),
                        'affected_ids': _derive_affected_ids(finding),
                        'description': finding.get('description', ''),
                    },
                )

            if not to_remediate:
                return

            logger.info(
                f'Remediation: {len(to_remediate)} actionable findings from run {parent_run_id}, '
                f'triggering second pass'
                + (
                    f' ({len(suppressed)} suppressed — open escalation exists)'
                    if suppressed else ''
                )
            )
            await self._run_remediation_pass(
                project_id, parent_run_id, to_remediate, tier,
                project_root=project_root,
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
        parent_run_id: str,
        findings: list[dict],
        tier: TierConfig,
        *,
        project_root: str,
        filtered_task_tree: FilteredTaskTree | None = None,
    ) -> None:
        """Run a focused S1→S2→S3 pass to remediate actionable findings.

        project_root is threaded from the parent caller: run_full_cycle resolves it
        once at entry via _known_project_root_for, before any side-effects, and
        threads it through _maybe_remediate so remediation always uses the pre-cycle
        snapshot, immune to any mid-cycle registry mutations (task 1163).

        If filtered_task_tree is provided it is used directly; otherwise a fresh
        tree is fetched via _fetch_filtered_task_tree.  Callers that already hold
        a fetched tree (e.g. run_full_cycle) should pass it through to avoid a
        redundant taskmaster round-trip.
        """
        # Defense-in-depth assert deliberately omitted.  A registry-bound check such
        # as `assert project_root in self._known_projects.values()` would fail during
        # the mid-cycle mutation window that task 1163 was specifically designed to
        # tolerate: both test_remediation_pass_uses_threaded_project_root_over_registry
        # and test_remediation_uses_threaded_project_root_not_mutated_registry pass a
        # project_root that intentionally differs from the current registry value.

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
            instance_id=self.buffer.instance_id,
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
                stage.known_projects = self._known_projects

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

            # After second-pass S3: gate escalation on persistence (task 1512 /
            # plans/afk-A7-recon-closure.md).  Never a third pass.
            # A finding is only escalated when it has appeared in at least
            # _INTEGRITY_FINDING_RECURRENCE_THRESHOLD completed runs, which signals
            # that remediation cannot fix it on its own.  Below the threshold we
            # suppress the escalation and emit a structured log so the finding
            # stays observable without polluting the queue.
            # The except handlers further below (AllAccountsCappedException and
            # bare Exception) are untouched — they fire on stage *exceptions*, not
            # on Stage-3 findings, and are the genuine "needs human" signals.
            #
            # task 1512 review_feedback (design_inconsistency):
            # Partition remaining findings by actionability — identical to
            # _maybe_remediate's partition at lines 1260-1261.  Non-actionable
            # findings are logged exactly as in the parent pass and must NEVER reach
            # the persistence-gated escalation branch regardless of recurrence count.
            s3_report = run.stage_reports.get('integrity_check')
            if s3_report is not None:
                if isinstance(s3_report, dict):
                    all_remaining = s3_report.get('items_flagged', [])
                else:
                    all_remaining = s3_report.items_flagged
                actionable_remaining = [f for f in all_remaining if f.get('actionable', False)]
                non_actionable_remaining = [f for f in all_remaining if not f.get('actionable', False)]
                # Non-actionable findings are logged but never escalated — same
                # contract as the parent pass in _maybe_remediate.
                for finding in non_actionable_remaining:
                    self._log_non_actionable_finding(project_id, run_id, finding)
                for finding in actionable_remaining:
                    persistence = await self._finding_persistence_count(project_id, finding)
                    if persistence >= _INTEGRITY_FINDING_RECURRENCE_THRESHOLD:
                        # Live-workflow gate (task 1655): if any cited task has an active
                        # workflow (registered worktree, recent branch commits, or live
                        # orchestrator), suppress this cycle's escalation with an INFO log.
                        # The finding keeps recurring and will escalate after the workflow
                        # finishes — so genuine stranded cases (all signals False) still
                        # escalate.  Guard detector errors as not-live (fail toward escalating
                        # rather than toward silencing a genuine stranded-work escalation).
                        affected_ids = _derive_affected_ids(finding)
                        # For liveness, iterate only cited task ids.
                        # _derive_affected_ids mixes in entity canonical_names,
                        # edge_uuids, and memory_ids; passing those to
                        # is_workflow_live_for_task would build nonsensical
                        # branch names (e.g. 'task/<canonical_name>') and waste
                        # git subprocess calls that can never match.
                        cited_task_ids = [
                            str(c['task_id'])
                            for c in finding.get('cited_tasks') or []
                            if isinstance(c, dict) and 'task_id' in c
                        ]
                        any_live = False
                        for tid in cited_task_ids:
                            try:
                                if is_workflow_live_for_task(tid, project_root):
                                    any_live = True
                                    break
                            except Exception as _det_exc:
                                logger.debug(
                                    'live_workflow_detector error for task %s; treating as not-live: %s',
                                    tid, _det_exc,
                                )
                        if any_live:
                            logger.info(
                                'reconciliation.integrity_escalation_suppressed_live_workflow',
                                extra={
                                    'project_id': project_id,
                                    'run_id': run_id,
                                    'affected_ids': affected_ids,
                                    'description': finding.get('description', ''),
                                    'finding_category': finding.get('category', ''),
                                },
                            )
                        else:
                            self._escalate(
                                'recon_integrity_issue',
                                run_id,
                                f'Persistently unresolved after remediation '
                                f'({persistence} cycles): {finding.get("description", "?")}',
                                detail=json.dumps(
                                    {**finding, 'persistence': persistence},
                                    default=str,
                                ),
                                finding=finding,
                            )
                    else:
                        logger.info(
                            'reconciliation.unresolved_after_remediation_suppressed',
                            extra={
                                'project_id': project_id,
                                'run_id': run_id,
                                'parent_run_id': parent_run_id,
                                'finding_category': finding.get('category', ''),
                                'affected_ids': _derive_affected_ids(finding),
                                'description': finding.get('description', ''),
                                'persistence': persistence,
                                'threshold': _INTEGRITY_FINDING_RECURRENCE_THRESHOLD,
                            },
                        )

            logger.info(
                'reconciliation.remediation_completed',
                extra={'run_id': run_id, 'parent_run_id': parent_run_id},
            )

        except AllAccountsCappedException as e:
            run.status = RunStatus.failed
            run.stage_reports['_error'] = {
                'error_type': 'AllAccountsCappedException',
                'error_message': str(e),
                'failed_stage': current_stage_name,
                'deferred': True,
            }
            await self.journal.complete_run(run_id, 'failed')
            # Do NOT re-raise — parent run already completed
            # Do NOT restore events — there are none (remediation has no drained events)
            logger.warning(
                f'Remediation deferred: all accounts capped during stage '
                f'{current_stage_name} ({e.retries} retries in {e.elapsed_secs:.1f}s)'
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

        # task 1143: hard-bind project_root from registry — event payloads are informational only.
        project_root = self.harness._known_project_root_for(project_id)

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
