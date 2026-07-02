"""Stage 1: Memory Consolidator — consolidates memories within and across stores."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from fused_memory.models.reconciliation import (
    AssembledPayload,
    ContextItem,
    ReconciliationEvent,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.cli_stage_runner import (
    STAGE1_DISALLOWED,
    build_summary_nonce_section,
)
from fused_memory.reconciliation.flag_dedup import (
    dedup_flags,
    filter_false_absence_flags,
    filter_stale_count_snapshot_corrections,
    filter_terminal_metadata_flags,
)
from fused_memory.reconciliation.prompts import _STAGE1_PROJECT_ID_GUIDELINE
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stage1_stall_detector import (
    compute_stalled_task_ids,
    extract_human_operator_task_ids,
    maybe_escalate_stalled_tasks,
    track_human_operator_stalls,
)
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    _render_live_workflow_section,
)
from fused_memory.reconciliation.summary_pool import pretrim_summary_pool
from fused_memory.reconciliation.task_filter import (
    FilteredTaskTree,
    detect_census_inconsistency,
    format_filtered_task_tree,
    strip_snapshot_lines,
)

logger = logging.getLogger(__name__)

# Stage 1 per-cycle summary pool cap and related constants (task 1942).
# Mirrors Stage 2's cycle-summary pool cap (task 1657 + trim-then-write, task
# 1831): every per-cycle summary add_memory call is tagged
# recon_pool='stage1_cycle_summary' (producer contract: Stage 1 prompt).
# MemoryConsolidator.run() pre-trims the pool to cap-1 BEFORE the agent write
# via the shared reconciliation.summary_pool core.
_STAGE1_CYCLE_SUMMARY_RECON_POOL = 'stage1_cycle_summary'
STAGE1_CYCLE_SUMMARY_POOL_CAP: int = 2
_STAGE1_CYCLE_SUMMARY_TRIM_SOURCE = 'stage1_cycle_summary_trim'


class MemoryConsolidator(BaseStage):
    """Stage 1: Review and consolidate memories across Graphiti and Mem0."""

    # Tier limits — set by harness before run(); None until explicitly assigned
    episode_limit: int | None = None
    memory_limit: int | None = None

    # Remediation support — set by harness for second pass
    remediation_findings: list[dict] | None = None
    prior_s3_findings: list[dict] | None = None

    # Cycle fence — set by harness to protect targeted-recon writes
    cycle_fence_time: datetime | None = None

    # Token-budget assembled payload — set by harness when using ContextAssembler.
    # When set, assemble_payload() uses this instead of the generic time-windowed fetch.
    assembled_payload: AssembledPayload | None = None

    # Active task tree — set by harness before run() (task 455)
    filtered_task_tree: FilteredTaskTree | None = None

    # Authoritative task-count cross-verification record — set by harness (task 1785).
    # None when the status map was unavailable (e.g. taskmaster down) or in remediation passes.
    task_count_verification: dict | None = None

    # Graphiti async-queue health record — set by harness (task 1785).
    # None when durable_queue was unavailable or in remediation passes.
    graphiti_queue_health: dict | None = None

    # Cached project_status_correction memory vs. live get_statuses census
    # diff/supersede record — set by harness (task 1938).
    # None when the status map was unavailable or in remediation passes.
    status_correction_reconciliation: dict | None = None

    # Count of snapshot lines stripped from the payload in the current cycle (task 1547)
    _entity_summary_snapshot_lines_stripped: int = 0

    # Fetch degraded sources for the current run — reset at the top of assemble_payload
    # and _format_assembled_payload; copied to report.stats['stage1_fetch_degraded'] in run()
    # (task 1812). Empty list = genuine empty corpus; non-empty = fetch failure on that source.
    # Initialized per-instance in __init__ to avoid the shared-mutable-default hazard.
    _fetch_degraded_sources: list

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-instance initialisation: avoids shared-mutable-default hazard.
        # A class-level [] would be mutated in-place by any append() that ran before
        # the run-local reset in assemble_payload / _format_assembled_payload, leaking
        # state across all instances (and across reused-instance runs).
        self._fetch_degraded_sources: list = []

    async def run(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
        run_id: str,
        model: str | None = None,
    ) -> StageReport:
        """Execute Stage 1 and post-process items_flagged through the flag deduplicator.

        Remediation runs (``remediation_findings`` set) skip dedup — the whole
        point of a remediation pass is to re-emit a curated list, and running
        dedup on those flags would defeat the remediation contract.
        """
        # --- trim-then-write: pre-trim pool to cap-1 BEFORE agent writes (task 1942) ---
        # Mirrors Stage 2's task-1831 wiring. Runs unconditionally (full +
        # remediation) so the pool is bounded every cycle. Best-effort.
        pretrimmed = await pretrim_summary_pool(
            self.memory,
            self.project_id,
            run_id,
            recon_pool=_STAGE1_CYCLE_SUMMARY_RECON_POOL,
            trim_source=_STAGE1_CYCLE_SUMMARY_TRIM_SOURCE,
            cap=STAGE1_CYCLE_SUMMARY_POOL_CAP,
        )
        report = await super().run(events, watermark, prior_reports, run_id, model=model)
        report.stats['entity_summary_snapshot_lines_stripped'] = (
            self._entity_summary_snapshot_lines_stripped
        )
        report.stats['stage1_fetch_degraded'] = self._fetch_degraded_sources or []
        report.stats['stage1_cycle_summary_pool_trimmed'] = pretrimmed

        # Skip dedup for remediation passes
        if self.remediation_findings is not None:
            return report

        if report.items_flagged:
            # ── Stale count-snapshot correction filter (task-1786): drop false ────
            # 'off-by-N correction' findings on stale-by-design task-count snapshot
            # edges BEFORE dedup_flags so no stage1_flag_marker churn occurs.
            # Pure, sync — no I/O needed (reads only the flag's own text fields).
            # Surfaces dropped count as report.stats['stale_count_snapshot_corrections_dropped'].
            _before_snapshot_filter = len(report.items_flagged)
            report.items_flagged = filter_stale_count_snapshot_corrections(
                report.items_flagged,
            )
            report.stats['stale_count_snapshot_corrections_dropped'] = (
                _before_snapshot_filter - len(report.items_flagged)
            )
            # ── Terminal-metadata guard (task-1725): drop stale_metadata flags for ──
            # cancelled/done tasks BEFORE dedup_flags so no marker write/delete churn
            # occurs.  Fail-safe: only drops on positively-confirmed terminal status.
            report.items_flagged = await filter_terminal_metadata_flags(
                taskmaster=self.taskmaster,
                project_root=self.project_root,
                flags=report.items_flagged,
            )
            report.items_flagged = await dedup_flags(
                memory_service=self.memory,
                project_id=self.project_id,
                run_id=run_id,
                flags=report.items_flagged,
            )
            # ── Deletion guard: drop absence-type flags that cannot be confirmed absent ──
            # filter_false_absence_flags is fail-closed: keeps an absence-asserting flag
            # ONLY when get_task POSITIVELY confirms the task does not exist.  Present or
            # inconclusive → drop to prevent irreversible delete_memory ops on real tasks.
            report.items_flagged = await filter_false_absence_flags(
                taskmaster=self.taskmaster,
                project_root=self.project_root,
                flags=report.items_flagged,
            )

        # ── Census inconsistency detection ────────────────────────────────────
        # Compare task IDs referenced in this cycle's events against the census
        # max from the filtered task tree.  IDs exceeding the max indicate a
        # partial/wrong-source bulk task read (the "highest-ID below known task IDs"
        # signature) and should be surfaced as a structured observation so operators
        # can investigate the root cause.
        #
        # Guard: skip when max_task_id == 0 (empty tree or all IDs unparseable).
        # If every referenced task ID is "above" a zero ceiling the result is a
        # false-positive flood rather than a genuine truncation signal.
        if self.filtered_task_tree is not None and self.filtered_task_tree.max_task_id > 0:
            event_task_ids = [
                e.payload.get('task_id')
                for e in events
                if e.payload.get('task_id') is not None
            ]
            inconsistent = detect_census_inconsistency(
                self.filtered_task_tree.max_task_id, event_task_ids
            )
            if inconsistent:
                report.stats['task_tree_census_inconsistent'] = inconsistent
                logger.warning(
                    'reconciliation.task_tree_census_inconsistent',
                    extra={
                        'project_id': self.project_id,
                        'run_id': run_id,
                        'census_max': self.filtered_task_tree.max_task_id,
                        'offending_ids': inconsistent,
                    },
                )

        # ── Task-count cross-verification (task 1785) ─────────────────────────
        # Surface the authoritative get_statuses census vs. get_tasks-derived
        # FilteredTaskTree comparison so the async judge / recon report can act
        # on count divergence without relying on a possibly-dropped Graphiti edge.
        if self.task_count_verification is not None:
            report.stats['task_count_verification'] = self.task_count_verification
            if not self.task_count_verification.get('consistent', True):
                logger.warning(
                    'reconciliation.task_count_divergence',
                    extra={
                        'project_id': self.project_id,
                        'run_id': run_id,
                        'authoritative': self.task_count_verification.get('authoritative'),
                        'tree': self.task_count_verification.get('tree'),
                    },
                )

        # ── Graphiti async-queue health (task 1785) ────────────────────────────
        # Surface the dead-letter count from DurableWriteQueue so the silent
        # add_memory drop (success=True at enqueue, dead-lettered asynchronously)
        # is visible in the Stage 1 report.
        if self.graphiti_queue_health is not None:
            report.stats['graphiti_queue_health'] = self.graphiti_queue_health
            if not self.graphiti_queue_health.get('healthy', True):
                logger.warning(
                    'reconciliation.graphiti_queue_unhealthy_stage1',
                    extra={
                        'project_id': self.project_id,
                        'run_id': run_id,
                        'dead_count': self.graphiti_queue_health.get('dead_count'),
                    },
                )

        # ── Status-correction reconciliation (task 1938) ───────────────────────
        # Surface the cached project_status_correction memory vs. live get_statuses
        # diff/supersede record so a stale Mem0 correction never silently outlives
        # the authoritative census.
        if self.status_correction_reconciliation is not None:
            report.stats['status_correction_reconciliation'] = (
                self.status_correction_reconciliation
            )

        # ── Stale human-operator-required detector (task 1201) ────────────────
        # Short-circuit when the escalation queue is unavailable — writing Mem0
        # markers that nothing will ever consume only wastes Qdrant capacity.
        # If a queue becomes available in a future cycle the counter starts
        # from zero for that cycle; that is an acceptable trade-off.
        if self._escalation_queue is not None:
            hor_task_ids = extract_human_operator_task_ids(report.items_flagged or [])
            if hor_task_ids:
                stall_counts = await track_human_operator_stalls(
                    memory_service=self.memory,
                    project_id=self.project_id,
                    run_id=run_id,
                    task_ids=hor_task_ids,
                )
                stalled = compute_stalled_task_ids(stall_counts)
                report.stats['stage1_human_operator_stalled'] = len(stalled)
                if stalled:
                    escalated = await maybe_escalate_stalled_tasks(
                        escalation_queue=self._escalation_queue,
                        project_id=self.project_id,
                        run_id=run_id,
                        stalled_task_ids=stalled,
                        stall_counts=stall_counts,
                        flags=report.items_flagged or [],
                    )
                    report.stats['stage1_human_operator_escalated'] = len(escalated)
                else:
                    report.stats['stage1_human_operator_escalated'] = 0
        return report

    def get_system_prompt(self) -> str:
        return STAGE1_SYSTEM_PROMPT

    def get_disallowed_tools(self) -> list[str]:
        return STAGE1_DISALLOWED

    async def assemble_payload(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
    ) -> str:
        # Token-budget assembled payload — skip generic fetch
        if self.assembled_payload is not None:
            return await self._format_assembled_payload(watermark)

        # Validate that limits were explicitly set by the harness
        if self.episode_limit is None or self.memory_limit is None:
            raise ValueError(
                f'episode_limit and memory_limit must be explicitly set by the harness before run(); '
                f'got episode_limit={self.episode_limit}, memory_limit={self.memory_limit}'
            )

        # Reset run-local degraded list before fetches (must precede every early-return so
        # reused instances never leak a stale list into a subsequent remediation run's stats)
        self._fetch_degraded_sources = []

        # Remediation mode: return focused payload with findings only
        if self.remediation_findings is not None:
            return self._assemble_remediation_payload()

        # 1. Episodes since last reconciliation
        try:
            episodes = await self.memory.get_episodes(
                project_id=self.project_id, last_n=self.episode_limit
            )
        except Exception as e:
            logger.warning(
                'reconciliation.stage1_episodes_fetch_failed',
                extra={'project_id': self.project_id, 'error': str(e)},
            )
            episodes = []
            self._fetch_degraded_sources.append('episodes')
        new_episodes = episodes
        if watermark.last_episode_timestamp:
            wm_str = str(watermark.last_episode_timestamp)
            new_episodes = [
                e for e in episodes
                if (e.get('created_at') or '') > wm_str
            ]

        # 2. Mem0 memories (recent)
        from fused_memory.models.scope import Scope
        scope = Scope(project_id=self.project_id)
        try:
            all_memories = await self.memory.mem0.get_all(scope, limit=self.memory_limit)
            results = all_memories.get('results') if isinstance(all_memories, dict) else None
            if not isinstance(results, list):
                logger.warning(
                    "mem0.get_all returned malformed/absent 'results' key; "
                    "treating as empty. got: %s",
                    repr(all_memories)[:200],
                )
                results = []
            mem0_memories = results
        except Exception:
            logger.warning(
                "mem0.get_all raised an exception; treating as empty memories.",
                exc_info=True,
            )
            mem0_memories = []
            self._fetch_degraded_sources.append('mem0')

        new_memories = mem0_memories
        if watermark.last_memory_timestamp:
            wm_str = str(watermark.last_memory_timestamp)
            new_memories = [
                m for m in mem0_memories
                if (m.get('created_at') or m.get('updated_at') or '') > wm_str
            ]

        # 3. Store stats
        status = await self._fetch_status()

        # 4. Events summary
        event_summary = _format_events(events)

        # 5. Prior S3 findings (backstop from last completed run)
        prior_s3_section = ''
        if self.prior_s3_findings:
            prior_s3_section = (
                f'\n### Prior Stage 3 Findings ({len(self.prior_s3_findings)})\n'
                f'These issues were found in the last integrity check and should be addressed '
                f'during this consolidation pass if possible.\n'
                f'{_format_findings(self.prior_s3_findings)}\n'
            )

        # 6. Cycle fence section
        cycle_fence_section = ''
        if self.cycle_fence_time:
            cycle_fence_section = (
                f'\n### Cycle Fence\n'
                f'This cycle started at {self.cycle_fence_time.isoformat()}.\n'
                f'Do NOT delete, merge, or modify any memory whose metadata includes '
                f'`source=targeted_reconciliation` and was created after this timestamp. '
                f'These are recent targeted reconciliation writes that must be preserved.\n'
            )

        # 7. Active task tree (task 455)
        task_tree_section = self._build_task_tree_section()

        # 7b. Task Count Census (task 1785)
        task_count_census_section = self._build_task_count_census_section()

        # 7c. Live-Workflow Signals (task 1977 — mirrors Stage 2's task 1655)
        live_workflow_section = self._build_live_workflow_section()

        # 8. Format
        episodes_str, ep_n = _format_episodes(new_episodes)
        memories_str, mem_n = _format_memories(new_memories)
        self._entity_summary_snapshot_lines_stripped = ep_n + mem_n

        # 9. Per-cycle summary nonce (task 1574)
        summary_nonce_section = self._build_summary_nonce_section()

        return f"""## Reconciliation Run — Stage 1: Memory Consolidation
## Project: {self.project_id}

### Buffered Events ({len(events)})
{event_summary}

### New Episodes Since Last Reconciliation ({len(new_episodes)})
{episodes_str}

### New Mem0 Memories Since Last Reconciliation ({len(new_memories)})
{memories_str}

### Store Status
{json.dumps(status, indent=2, default=str)}

### Previous Reconciliation
{_format_watermark(watermark)}
{prior_s3_section}{cycle_fence_section}{task_tree_section}{task_count_census_section}{live_workflow_section}{summary_nonce_section}
## Your Task
Review the above data and perform memory consolidation:
1. Within Mem0: identify duplicates, contradictions, stale entries. Merge/delete as needed.
2. Within Graphiti: review entity consistency, superseded temporal facts.
3. Cross-store: check for contradictions between stores. Promote solidified patterns.
4. Flag any items that are relevant to task planning for Stage 2.
5. When you have completed your work, produce your final structured report as your response.

{_STAGE1_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}{self._build_project_root_directive()}"""

    async def _fetch_status(self) -> dict:
        """Fetch store status, logging a WARNING and tracking degradation on failure.

        Single source of truth for the status-fetch try/except that both
        assemble_payload (legacy time-windowed path) and _format_assembled_payload
        (assembled/token-budget path) share.  The caller is responsible for resetting
        self._fetch_degraded_sources before calling this method.
        """
        try:
            return await self.memory.get_status(project_id=self.project_id)
        except Exception as e:
            logger.warning(
                'reconciliation.stage1_status_fetch_failed',
                extra={'project_id': self.project_id, 'error': str(e)},
            )
            self._fetch_degraded_sources.append('status')
            return {}

    async def _format_assembled_payload(self, watermark: Watermark) -> str:
        """Format a payload from ContextAssembler output — event-driven context."""
        ap = self.assembled_payload
        assert ap is not None

        # assemble_payload's line-262 reset is bypassed by the early return at lines 250-251.
        # Reset here so each assembled-path call starts clean (no leak from a prior run).
        self._fetch_degraded_sources = []

        event_summary = _format_events(ap.events)

        # Store status (cheap fetch, always useful)
        status = await self._fetch_status()

        # Prior S3 findings
        prior_s3_section = ''
        if self.prior_s3_findings:
            prior_s3_section = (
                f'\n### Prior Stage 3 Findings ({len(self.prior_s3_findings)})\n'
                f'These issues were found in the last integrity check and should be addressed '
                f'during this consolidation pass if possible.\n'
                f'{_format_findings(self.prior_s3_findings)}\n'
            )

        # Cycle fence
        cycle_fence_section = ''
        if self.cycle_fence_time:
            cycle_fence_section = (
                f'\n### Cycle Fence\n'
                f'This cycle started at {self.cycle_fence_time.isoformat()}.\n'
                f'Do NOT delete, merge, or modify any memory whose metadata includes '
                f'`source=targeted_reconciliation` and was created after this timestamp. '
                f'These are recent targeted reconciliation writes that must be preserved.\n'
            )

        # Active task tree (task 455)
        task_tree_section = self._build_task_tree_section()

        # Task Count Census (task 1785)
        task_count_census_section = self._build_task_count_census_section()

        # Live-Workflow Signals (task 1977 — mirrors Stage 2's task 1655)
        live_workflow_section = self._build_live_workflow_section()

        # Per-cycle summary nonce (task 1574)
        summary_nonce_section = self._build_summary_nonce_section()

        ctx_str, ctx_n = _format_context_items(ap.context_items)
        self._entity_summary_snapshot_lines_stripped = ctx_n

        return f"""## Reconciliation Run — Stage 1: Memory Consolidation
## Project: {self.project_id}

### Buffered Events ({len(ap.events)})
{event_summary}

### Related Context ({len(ap.context_items)} items)
{ctx_str}

### Store Status
{json.dumps(status, indent=2, default=str)}

### Previous Reconciliation
{_format_watermark(watermark)}
{prior_s3_section}{cycle_fence_section}{task_tree_section}{task_count_census_section}{live_workflow_section}{summary_nonce_section}
## Your Task
Review the above data and perform memory consolidation:
1. Within Mem0: identify duplicates, contradictions, stale entries. Merge/delete as needed.
2. Within Graphiti: review entity consistency, superseded temporal facts.
3. Cross-store: check for contradictions between stores. Promote solidified patterns.
4. Flag any items that are relevant to task planning for Stage 2.
5. When you have completed your work, produce your final structured report as your response.

{_STAGE1_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}{self._build_project_root_directive()}"""

    def _build_project_root_directive(self) -> str:
        """Return the project_root directive line for payload footers, or empty string.

        When ``self.project_root`` is falsy (the BaseStage default ``''``), returns
        ``''`` so the payload ends immediately after the project_id guideline with no
        trailing newline.  When set, returns ``'\\nUse project_root=...'`` (leading
        newline keeps it on its own line after the preceding guideline).  Both payload
        methods call this helper so the f-string fragment stays in a single place.
        """
        if not self.project_root:
            return ''
        return f'\nUse project_root="{self.project_root}" for tasks scoped to this project.'

    def _build_task_tree_section(self) -> str:
        """Return the Active Task Tree prompt section, or empty string if no tree set.

        Eliminates the duplicate block across assemble_payload and
        _format_assembled_payload. (task 455)
        """
        if self.filtered_task_tree is None:
            return ''
        return '\n' + format_filtered_task_tree(self.filtered_task_tree) + '\n'

    def _build_live_workflow_section(self) -> str:
        """Return the Live-Workflow Signals prompt section, or empty string if inapplicable.

        Mirrors Stage 2's guard/source exactly (task 1655): renders over
        ``self.filtered_task_tree.active_tasks`` when ``self.project_root`` and
        ``self.filtered_task_tree`` are set and ``active_tasks`` is non-empty.
        Reuses Stage 2's ``_render_live_workflow_section`` renderer (imported from
        task_knowledge_sync.py) so both stages emit byte-identical section
        formatting for the same underlying live-workflow signals (task 1977).

        Returns '' when the guard fails or no active task is currently live —
        keeps the payload tight, matching _build_task_tree_section's pattern.
        """
        if not (
            self.project_root
            and self.filtered_task_tree
            and self.filtered_task_tree.active_tasks
        ):
            return ''
        section = _render_live_workflow_section(
            self.filtered_task_tree.active_tasks,
            self.project_root,
        )
        if not section:
            return ''
        return '\n' + section

    def _build_task_count_census_section(self) -> str:
        """Return the Task Count Census payload section, or empty string if unavailable.

        Mirrors _build_task_tree_section — inserted into both assemble_payload and
        _format_assembled_payload alongside task_tree_section.  The authoritative
        counts come from get_statuses (a deterministic compact read) rather than the
        async-queued Graphiti snapshot edge that triggered the run-929b4135 incident.

        Returns '' when task_count_verification is None so the payload is unchanged
        for remediation passes and for cycles where get_statuses was unavailable.
        """
        if self.task_count_verification is None:
            return ''
        v = self.task_count_verification
        if not v.get('available'):
            return ''
        auth = v.get('authoritative') or {}
        auth_total = auth.get('total', 'unknown')
        auth_done = auth.get('done', 'unknown')
        divergence_note = ''
        if not v.get('consistent', True):
            tree = v.get('tree') or {}
            divergence_note = (
                f'\n⚠ Divergence detected: tree reports done={tree.get("done", "?")} '
                f'vs authoritative done={auth_done}.'
            )
        return (
            f'\n### Task Count Census (authoritative — from get_statuses)\n'
            f'Total: {auth_total}, Done: {auth_done}{divergence_note}\n'
        )

    def _build_summary_nonce_section(self) -> str:
        """Return the Per-Cycle Summary Nonce payload section with a fresh STAGE1-prefixed nonce.

        Delegates to the shared cli_stage_runner.build_summary_nonce_section() helper
        so Stage 1 and Stage 2 produce identical section wording and cannot drift
        (task 1574 single-source-of-truth design).

        Task 1590: passes 'STAGE1' prefix so Stage 1 nonces always lead with 'STAGE1_',
        making stage origin explicit and ensuring Stage 1 and Stage 2 summaries can never
        share a leading nonce token, further separating them in Mem0's embedding space.

        This single method feeds BOTH the legacy _format_payload (line ~263) and the
        event-driven _format_assembled_payload (line ~331) paths.
        """
        return build_summary_nonce_section('STAGE1')

    def _assemble_remediation_payload(self) -> str:
        """Focused payload for remediation runs — findings only, no full data."""
        self._entity_summary_snapshot_lines_stripped = 0
        findings = self.remediation_findings or []
        return f"""## Remediation Run — Stage 1: Targeted Memory Fixes
## Project: {self.project_id}

### Actionable Findings to Remediate ({len(findings)})
{_format_findings(findings)}

## Your Task
This is a focused remediation run. Address ONLY the specific findings listed above:
1. For each finding: investigate the affected IDs, apply the suggested action, verify the fix.
2. If a finding cannot be resolved, flag it for Stage 2 with an explanation.
3. Do NOT perform general consolidation — only fix the listed findings.
4. Report each finding's resolution status in your structured report.

{_STAGE1_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}
"""


def _format_events(events: list[ReconciliationEvent]) -> str:
    if not events:
        return 'No events.'
    lines = []
    for e in events:
        lines.append(f'- [{e.type.value}] {e.timestamp.isoformat()}: {json.dumps(e.payload)}')
    return '\n'.join(lines)


def _format_episodes(episodes: list[dict]) -> tuple[str, int]:
    if not episodes:
        return 'No new episodes.', 0
    lines = []
    total_stripped = 0
    for ep in episodes:
        raw = ep.get('content') or ''
        cleaned, n = strip_snapshot_lines(raw)
        total_stripped += n
        content = cleaned[:500]
        lines.append(f'- [{ep.get("uuid", "?")}] {ep.get("created_at", "?")}: {content}')
    return '\n'.join(lines), total_stripped


def _format_memories(memories: list[dict]) -> tuple[str, int]:
    if not memories:
        return 'No memories.', 0
    lines = []
    total_stripped = 0
    for m in memories:
        raw = m.get('memory') or ''
        cleaned, n = strip_snapshot_lines(raw)
        total_stripped += n
        content = cleaned[:500]
        meta = m.get('metadata', {}) or {}
        cat = meta.get('category', '?')
        lines.append(f'- [{m.get("id", "?")}] ({cat}): {content}')
    return '\n'.join(lines), total_stripped


def _format_context_items(items: dict[str, ContextItem]) -> tuple[str, int]:
    """Format context items grouped by source, stripping count-snapshot lines."""
    if not items:
        return 'No related context.', 0
    by_source: dict[str, list[ContextItem]] = {}
    for item in items.values():
        by_source.setdefault(item.source, []).append(item)
    sections = []
    total_stripped = 0
    for source, source_items in sorted(by_source.items()):
        sections.append(f'**{source}** ({len(source_items)})')
        for item in source_items:
            cleaned, n = strip_snapshot_lines(item.formatted)
            total_stripped += n
            sections.append(cleaned)
    return '\n'.join(sections), total_stripped


def _format_findings(findings: list[dict]) -> str:
    """Render a list of finding dicts as a human-readable string.

    Each finding is formatted with its header fields (description, severity,
    category, suggested_action) followed by the four typed citation lists
    (cited_entities, cited_edges, cited_tasks, cited_memories) from PRD §9.3.
    Empty citation lists are omitted from output.

    Rendering limits (reviewer finding incomplete_rendering):
    - Each citation list is capped at 5 entries (``... +N more`` appended).
    - ``fact_text_snapshot`` is truncated at 120 characters to avoid inflating
      payload size when feeding findings into the next reconciliation context.
    - cited_tasks includes the task title so downstream consumers can read
      findings without re-fetching tasks.
    """
    _CITE_CAP = 5
    _SNAPSHOT_CAP = 120
    # Caps for LLM-controlled free-text fields (reviewer finding incomplete_rendering):
    # without a cap, verbose descriptions from a handful of findings can dominate
    # the downstream stage payload and swamp per-citation limits.
    _DESC_CAP = 500
    _ACTION_CAP = 500

    if not findings:
        return 'No findings.'
    lines = []
    for i, f in enumerate(findings, 1):
        desc = f.get('description', '?')
        severity = f.get('severity', '?')
        category = f.get('category', '?')
        action = f.get('suggested_action', '?')

        if len(desc) > _DESC_CAP:
            desc = desc[:_DESC_CAP] + '...'
        if len(action) > _ACTION_CAP:
            action = action[:_ACTION_CAP] + '...'

        parts = [
            f'{i}. [{severity}/{category}] {desc}',
            f'   Suggested action: {action}',
        ]

        cited_entities = f.get('cited_entities') or []
        if cited_entities:
            shown = cited_entities[:_CITE_CAP]
            entity_strs = ', '.join(
                f"{e.get('canonical_name', '?')} ({e.get('entity_uuid', '?')})"
                for e in shown
            )
            extra = len(cited_entities) - len(shown)
            if extra:
                entity_strs += f', ... +{extra} more'
                logger.debug('_format_findings: finding %d cited_entities truncated (%d dropped)', i, extra)
            parts.append(f'   Cited entities: {entity_strs}')

        cited_edges = f.get('cited_edges') or []
        if cited_edges:
            shown = cited_edges[:_CITE_CAP]
            edge_parts = []
            for e in shown:
                snapshot = e.get('fact_text_snapshot', '?')
                if len(snapshot) > _SNAPSHOT_CAP:
                    snapshot = snapshot[:_SNAPSHOT_CAP] + '...'
                edge_parts.append(f"{e.get('edge_uuid', '?')}: {snapshot}")
            edge_strs = ', '.join(edge_parts)
            extra = len(cited_edges) - len(shown)
            if extra:
                edge_strs += f', ... +{extra} more'
                logger.debug('_format_findings: finding %d cited_edges truncated (%d dropped)', i, extra)
            parts.append(f'   Cited edges: {edge_strs}')

        cited_tasks = f.get('cited_tasks') or []
        if cited_tasks:
            shown = cited_tasks[:_CITE_CAP]
            task_strs = ', '.join(
                f"{t.get('project_id', '?')}/{t.get('task_id', '?')} ({t.get('title', '?')})"
                for t in shown
            )
            extra = len(cited_tasks) - len(shown)
            if extra:
                task_strs += f', ... +{extra} more'
                logger.debug('_format_findings: finding %d cited_tasks truncated (%d dropped)', i, extra)
            parts.append(f'   Cited tasks: {task_strs}')

        cited_memories = f.get('cited_memories') or []
        if cited_memories:
            shown = cited_memories[:_CITE_CAP]
            memory_strs = ', '.join(
                f"{m.get('memory_id', '?')} ({m.get('store', '?')})"
                for m in shown
            )
            extra = len(cited_memories) - len(shown)
            if extra:
                memory_strs += f', ... +{extra} more'
                logger.debug('_format_findings: finding %d cited_memories truncated (%d dropped)', i, extra)
            parts.append(f'   Cited memories: {memory_strs}')

        lines.append('\n'.join(parts))
    return '\n'.join(lines)


def _format_watermark(watermark: Watermark) -> str:
    if watermark.last_full_run_completed is None:
        return 'First run — no previous reconciliation.'
    return (
        f'Last full run: {watermark.last_full_run_id} '
        f'at {watermark.last_full_run_completed.isoformat()}'
    )
