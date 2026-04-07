"""Stage 1: Memory Consolidator — consolidates memories within and across stores."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fused_memory.models.reconciliation import (
    ReconciliationEvent,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.agent_loop import ToolDefinition
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stages.base import BaseStage

if TYPE_CHECKING:
    from fused_memory.reconciliation.verify import CodebaseVerifier


class MemoryConsolidator(BaseStage):
    """Stage 1: Review and consolidate memories across Graphiti and Mem0."""

    verifier: CodebaseVerifier | None = None

    def get_system_prompt(self) -> str:
        return STAGE1_SYSTEM_PROMPT

    def get_tools(self) -> dict[str, ToolDefinition]:
        tools = {}
        tools.update(self._memory_read_tools())
        tools.update(self._memory_write_tools())

        if self.verifier:
            tools['verify_against_codebase'] = self._verify_tool()

        return tools

    def _verify_tool(self) -> ToolDefinition:
        verifier = self.verifier
        assert verifier is not None  # only called when verifier is set

        async def verify(claim: str, context: str = '', scope_hints: list[str] | None = None):
            result = await verifier.verify(
                claim=claim, context=context, scope_hints=scope_hints, project_id=self.project_id
            )
            return result.model_dump()

        return ToolDefinition(
            name='verify_against_codebase',
            description=(
                'Verify a factual claim against the codebase. Spawns a read-only explore agent '
                'that checks files and git history. Use when you encounter conflicting factual '
                'claims about the codebase.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'claim': {'type': 'string', 'description': 'The factual claim to verify'},
                    'context': {'type': 'string', 'description': 'Additional context'},
                    'scope_hints': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'File paths or directories to focus on',
                    },
                },
                'required': ['claim'],
            },
            function=verify,
        )

    async def assemble_payload(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
    ) -> str:
        # 1. Episodes since last reconciliation
        episodes = await self.memory.get_episodes(project_id=self.project_id, last_n=100)
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
            all_memories = await self.memory.mem0.get_all(scope, limit=500)
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
{_format_episodes(new_episodes[:50])}

### Recent Mem0 Memories ({len(mem0_memories)})
{_format_memories(mem0_memories[:50])}

### Store Status
{json.dumps(status, indent=2, default=str)}

### Previous Reconciliation
{_format_watermark(watermark)}

## Your Task
Review the above data and perform memory consolidation:
1. Within Mem0: identify duplicates, contradictions, stale entries. Merge/delete as needed.
2. Within Graphiti: review entity consistency, superseded temporal facts.
3. Cross-store: check for contradictions between stores. Promote solidified patterns.
4. If you encounter conflicting factual claims about the codebase, use verify_against_codebase.
5. Flag any items that are relevant to task planning for Stage 2.
6. Call stage_complete with your report when done.
"""


def _format_events(events: list[ReconciliationEvent]) -> str:
    if not events:
        return 'No events.'
    lines = []
    for e in events[:30]:
        lines.append(f'- [{e.type.value}] {e.timestamp.isoformat()}: {json.dumps(e.payload)}')
    if len(events) > 30:
        lines.append(f'... and {len(events) - 30} more events')
    return '\n'.join(lines)


def _format_episodes(episodes: list[dict]) -> str:
    if not episodes:
        return 'No new episodes.'
    lines = []
    for ep in episodes:
        content = (ep.get('content') or '')[:200]
        lines.append(f'- [{ep.get("uuid", "?")}] {ep.get("created_at", "?")}: {content}')
    return '\n'.join(lines)


def _format_memories(memories: list[dict]) -> str:
    if not memories:
        return 'No memories.'
    lines = []
    for m in memories:
        content = (m.get('memory') or '')[:200]
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
