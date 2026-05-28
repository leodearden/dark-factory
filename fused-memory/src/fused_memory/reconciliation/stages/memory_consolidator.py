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
from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
from fused_memory.reconciliation.flag_dedup import dedup_flags, filter_false_absence_flags
from fused_memory.reconciliation.prompts import _STAGE1_PROJECT_ID_GUIDELINE
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stage1_stall_detector import (
    compute_stalled_task_ids,
    extract_human_operator_task_ids,
    maybe_escalate_stalled_tasks,
    track_human_operator_stalls,
)
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.task_filter import (
    FilteredTaskTree,
    detect_census_inconsistency,
    format_filtered_task_tree,
    strip_snapshot_lines,
)

logger = logging.getLogger(__name__)


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

    # Count of snapshot lines stripped from the payload in the current cycle (task 1547)
    _entity_summary_snapshot_lines_stripped: int = 0

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
        report = await super().run(events, watermark, prior_reports, run_id, model=model)
        report.stats['entity_summary_snapshot_lines_stripped'] = (
            self._entity_summary_snapshot_lines_stripped
        )

        # Skip dedup for remediation passes
        if self.remediation_findings is not None:
            return report

        if report.items_flagged:
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

        # Remediation mode: return focused payload with findings only
        if self.remediation_findings is not None:
            return self._assemble_remediation_payload()

        # 1. Episodes since last reconciliation
        try:
            episodes = await self.memory.get_episodes(
                project_id=self.project_id, last_n=self.episode_limit
            )
        except Exception:
            episodes = []
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
            mem0_memories = all_memories.get('results', [])
        except Exception:
            mem0_memories = []

        new_memories = mem0_memories
        if watermark.last_memory_timestamp:
            wm_str = str(watermark.last_memory_timestamp)
            new_memories = [
                m for m in mem0_memories
                if (m.get('created_at') or m.get('updated_at') or '') > wm_str
            ]

        # 3. Store stats
        try:
            status = await self.memory.get_status(project_id=self.project_id)
        except Exception:
            status = {}

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

        # 8. Format
        episodes_str, ep_n = _format_episodes(new_episodes)
        memories_str, mem_n = _format_memories(new_memories)
        self._entity_summary_snapshot_lines_stripped = ep_n + mem_n

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
{prior_s3_section}{cycle_fence_section}{task_tree_section}
## Your Task
Review the above data and perform memory consolidation:
1. Within Mem0: identify duplicates, contradictions, stale entries. Merge/delete as needed.
2. Within Graphiti: review entity consistency, superseded temporal facts.
3. Cross-store: check for contradictions between stores. Promote solidified patterns.
4. Flag any items that are relevant to task planning for Stage 2.
5. When you have completed your work, produce your final structured report as your response.

{_STAGE1_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}{self._build_project_root_directive()}"""

    async def _format_assembled_payload(self, watermark: Watermark) -> str:
        """Format a payload from ContextAssembler output — event-driven context."""
        ap = self.assembled_payload
        assert ap is not None

        event_summary = _format_events(ap.events)

        # Store status (cheap fetch, always useful)
        try:
            status = await self.memory.get_status(project_id=self.project_id)
        except Exception:
            status = {}

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
{prior_s3_section}{cycle_fence_section}{task_tree_section}
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
    if not findings:
        return 'No findings.'
    lines = []
    for i, f in enumerate(findings, 1):
        desc = f.get('description', '?')
        severity = f.get('severity', '?')
        category = f.get('category', '?')
        action = f.get('suggested_action', '?')
        affected = f.get('affected_ids', [])
        lines.append(
            f'{i}. [{severity}/{category}] {desc}\n'
            f'   Affected: {affected}\n'
            f'   Suggested action: {action}'
        )
    return '\n'.join(lines)


def _format_watermark(watermark: Watermark) -> str:
    if watermark.last_full_run_completed is None:
        return 'First run — no previous reconciliation.'
    return (
        f'Last full run: {watermark.last_full_run_id} '
        f'at {watermark.last_full_run_completed.isoformat()}'
    )
