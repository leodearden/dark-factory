"""Stage 1: Memory Consolidator — consolidates memories within and across stores."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from fused_memory.models.reconciliation import (
    AssembledPayload,
    ContextItem,
    ReconciliationEvent,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
from fused_memory.reconciliation.prompts import _STAGE1_PROJECT_ID_GUIDELINE
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stages.base import BaseStage

if TYPE_CHECKING:
    pass


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

        # 7. Format
        return f"""## Reconciliation Run — Stage 1: Memory Consolidation
## Project: {self.project_id}

### Buffered Events ({len(events)})
{event_summary}

### New Episodes Since Last Reconciliation ({len(new_episodes)})
{_format_episodes(new_episodes)}

### New Mem0 Memories Since Last Reconciliation ({len(new_memories)})
{_format_memories(new_memories)}

### Store Status
{json.dumps(status, indent=2, default=str)}

### Previous Reconciliation
{_format_watermark(watermark)}
{prior_s3_section}{cycle_fence_section}
## Your Task
Review the above data and perform memory consolidation:
1. Within Mem0: identify duplicates, contradictions, stale entries. Merge/delete as needed.
2. Within Graphiti: review entity consistency, superseded temporal facts.
3. Cross-store: check for contradictions between stores. Promote solidified patterns.
4. Flag any items that are relevant to task planning for Stage 2.
5. When you have completed your work, produce your final structured report as your response.

{_STAGE1_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}
"""

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

        return f"""## Reconciliation Run — Stage 1: Memory Consolidation
## Project: {self.project_id}

### Buffered Events ({len(ap.events)})
{event_summary}

### Related Context ({len(ap.context_items)} items)
{_format_context_items(ap.context_items)}

### Store Status
{json.dumps(status, indent=2, default=str)}

### Previous Reconciliation
{_format_watermark(watermark)}
{prior_s3_section}{cycle_fence_section}
## Your Task
Review the above data and perform memory consolidation:
1. Within Mem0: identify duplicates, contradictions, stale entries. Merge/delete as needed.
2. Within Graphiti: review entity consistency, superseded temporal facts.
3. Cross-store: check for contradictions between stores. Promote solidified patterns.
4. Flag any items that are relevant to task planning for Stage 2.
5. When you have completed your work, produce your final structured report as your response.

{_STAGE1_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}
"""

    def _assemble_remediation_payload(self) -> str:
        """Focused payload for remediation runs — findings only, no full data."""
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


def _format_episodes(episodes: list[dict]) -> str:
    if not episodes:
        return 'No new episodes.'
    lines = []
    for ep in episodes:
        content = (ep.get('content') or '')[:500]
        lines.append(f'- [{ep.get("uuid", "?")}] {ep.get("created_at", "?")}: {content}')
    return '\n'.join(lines)


def _format_memories(memories: list[dict]) -> str:
    if not memories:
        return 'No memories.'
    lines = []
    for m in memories:
        content = (m.get('memory') or '')[:500]
        meta = m.get('metadata', {}) or {}
        cat = meta.get('category', '?')
        lines.append(f'- [{m.get("id", "?")}] ({cat}): {content}')
    return '\n'.join(lines)


def _format_context_items(items: dict[str, ContextItem]) -> str:
    """Format context items grouped by source."""
    if not items:
        return 'No related context.'
    by_source: dict[str, list[ContextItem]] = {}
    for item in items.values():
        by_source.setdefault(item.source, []).append(item)
    sections = []
    for source, source_items in sorted(by_source.items()):
        sections.append(f'**{source}** ({len(source_items)})')
        for item in source_items:
            sections.append(item.formatted)
    return '\n'.join(sections)


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
