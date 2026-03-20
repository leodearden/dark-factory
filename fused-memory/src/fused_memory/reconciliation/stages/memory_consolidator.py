"""Stage 1: Memory Consolidator — consolidates memories within and across stores."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fused_memory.models.reconciliation import (
    ReconciliationEvent,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stages.base import BaseStage

if TYPE_CHECKING:
    pass


class MemoryConsolidator(BaseStage):
    """Stage 1: Review and consolidate memories across Graphiti and Mem0."""

    # Tier limits — set by harness before run()
    episode_limit: int = 500
    memory_limit: int = 1000

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

        # 3. Store stats
        try:
            status = await self.memory.get_status(project_id=self.project_id)
        except Exception:
            status = {}

        # 4. Events summary
        event_summary = _format_events(events)

        # 5. Format
        return f"""## Reconciliation Run — Stage 1: Memory Consolidation
## Project: {self.project_id}

### Buffered Events ({len(events)})
{event_summary}

### New Episodes Since Last Reconciliation ({len(new_episodes)})
{_format_episodes(new_episodes[:200])}

### Recent Mem0 Memories ({len(mem0_memories)})
{_format_memories(mem0_memories[:200])}

### Store Status
{json.dumps(status, indent=2, default=str)}

### Previous Reconciliation
{_format_watermark(watermark)}

## Your Task
Review the above data and perform memory consolidation:
1. Within Mem0: identify duplicates, contradictions, stale entries. Merge/delete as needed.
2. Within Graphiti: review entity consistency, superseded temporal facts.
3. Cross-store: check for contradictions between stores. Promote solidified patterns.
4. Flag any items that are relevant to task planning for Stage 2.
5. When you have completed your work, produce your final structured report as your response.

Always pass project_id="{self.project_id}" when calling fused-memory MCP tools.
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


def _format_watermark(watermark: Watermark) -> str:
    if watermark.last_full_run_completed is None:
        return 'First run — no previous reconciliation.'
    return (
        f'Last full run: {watermark.last_full_run_id} '
        f'at {watermark.last_full_run_completed.isoformat()}'
    )
