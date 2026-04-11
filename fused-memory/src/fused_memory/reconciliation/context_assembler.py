"""Token-budget context assembler for reconciliation payloads.

Replaces the fixed count-based chunking with dynamic, event-driven payload
assembly.  For each event in the backlog, fetches related context (memories,
entities, task details) and accumulates until the token budget is reached.

Context is deduplicated across events — if two events reference the same
entity, its context is counted only once.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fused_memory.models.reconciliation import (
    AssembledPayload,
    ContextItem,
    EventType,
    ReconciliationEvent,
    Watermark,
)
from fused_memory.utils.async_utils import propagate_cancellations

if TYPE_CHECKING:
    from fused_memory.backends.taskmaster_client import TaskmasterBackend
    from fused_memory.config.schema import ReconciliationConfig
    from fused_memory.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for mixed English/JSON."""
    return len(text) // 4


def format_event(event: ReconciliationEvent) -> str:
    """Format an event the same way _format_events does in memory_consolidator."""
    return f'- [{event.type.value}] {event.timestamp.isoformat()}: {json.dumps(event.payload)}'


def _format_memory_result(result) -> str:
    """Format a MemoryResult into a context line."""
    content = (result.content or '')[:500]
    source = result.source_store.value if result.source_store else '?'
    cat = result.category.value if result.category else '?'
    return f'- [{result.id}] ({source}/{cat}): {content}'


def _format_task(task: dict) -> str:
    """Format a task dict into a context line."""
    tid = task.get('id', '?')
    title = task.get('title', '?')
    status = task.get('status', '?')
    deps = task.get('dependencies', [])
    desc = (task.get('description') or '')[:300]
    hints = task.get('metadata', {}).get('memory_hints', '')
    parts = [f'- [task:{tid}] ({status}) {title} deps={deps}']
    if desc:
        parts.append(f'  desc: {desc}')
    if hints:
        parts.append(f'  memory_hints: {json.dumps(hints)}')
    return '\n'.join(parts)


class ContextAssembler:
    """Builds a token-budgeted payload by iterating through events and
    fetching related context for each one."""

    def __init__(
        self,
        memory_service: MemoryService,
        taskmaster: TaskmasterBackend | None,
        config: ReconciliationConfig,
        project_root: str = '',
    ):
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.config = config
        self.project_root = project_root
        self._search_limit = config.context_search_limit
        self._batch_size = config.context_fetch_batch_size

    async def assemble(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        project_id: str,
    ) -> AssembledPayload:
        """Build a token-budgeted payload from events with per-event context.

        Iterates through events oldest-first, fetching related context for
        each one.  Stops when the token budget is reached.  Returns an
        AssembledPayload with the events that fit and their deduplicated
        context.
        """
        budget = self.config.token_budget

        # Fixed overhead: system prompt (~950 tokens) + template chrome (~450)
        used = 1_400
        chunk_events: list[ReconciliationEvent] = []
        context_items: dict[str, ContextItem] = {}

        # Compute effective watermark: align context window with event batch
        effective_wm = self._compute_effective_watermark(events, watermark)

        # Process events in batches for parallelized context fetching
        event_idx = 0
        budget_reached = False
        while event_idx < len(events) and not budget_reached:
            batch = events[event_idx:event_idx + self._batch_size]
            event_idx += len(batch)

            # Fetch context for all events in batch concurrently
            fetch_tasks = [
                self._fetch_context(event, project_id)
                for event in batch
            ]
            batch_contexts = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Two-tier check for asyncio.gather(return_exceptions=True) results.
            # Pass 1: delegates to propagate_cancellations (fused_memory.utils.async_utils)
            # — the shared Pass 1 guard used by all gather(return_exceptions=True) callsites.
            # Re-raising here preserves the structured-cancellation contract and prevents
            # the assembler from silently converting a shutdown signal into an empty context list.
            # See fused_memory.utils.async_utils.propagate_cancellations for the shared
            # Pass 1 guard contract.
            propagate_cancellations(batch_contexts)

            for event, ctx_result in zip(batch, batch_contexts, strict=True):
                if isinstance(ctx_result, BaseException):
                    logger.warning(
                        f'Context fetch failed for event {event.id}: {ctx_result}'
                    )
                    ctx_result = []

                # Deduplicate context items
                new_items = [
                    item for item in ctx_result
                    if item.id not in context_items
                ]

                # Estimate cost of this event + its new context
                event_text = format_event(event)
                event_cost = estimate_tokens(event_text)
                context_cost = sum(item.token_estimate for item in new_items)
                total_cost = event_cost + context_cost

                if used + total_cost > budget and chunk_events:
                    # Budget exceeded and we have at least one event — stop
                    budget_reached = True
                    break

                # Add event and its context
                chunk_events.append(event)
                for item in new_items:
                    context_items[item.id] = item
                used += total_cost

        events_remaining = len(events) - len(chunk_events)

        logger.info(
            f'Context assembly complete: {len(chunk_events)} events, '
            f'{len(context_items)} context items, '
            f'{used} estimated tokens, '
            f'{events_remaining} events remaining'
        )

        return AssembledPayload(
            events=chunk_events,
            context_items=context_items,
            total_tokens=used,
            events_remaining=events_remaining,
            effective_watermark=effective_wm,
        )

    def _compute_effective_watermark(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
    ) -> datetime:
        """Effective watermark = min(last_run_completed, earliest_event.timestamp).

        Ensures the context window covers the period the events are about.
        """
        candidates: list[datetime] = []
        if watermark.last_full_run_completed:
            candidates.append(watermark.last_full_run_completed)
        if events:
            candidates.append(min(e.timestamp for e in events))
        if not candidates:
            return datetime.now(UTC)
        return min(candidates)

    async def _fetch_context(
        self,
        event: ReconciliationEvent,
        project_id: str,
    ) -> list[ContextItem]:
        """Dispatch context fetching by event type."""
        dispatch = {
            EventType.memory_added: self._ctx_memory_added,
            EventType.memory_deleted: self._ctx_memory_deleted,
            EventType.task_status_changed: self._ctx_task_event,
            EventType.task_created: self._ctx_task_event,
            EventType.task_modified: self._ctx_task_event,
            EventType.task_deleted: self._ctx_task_deleted,
            EventType.episode_added: self._ctx_episode_added,
            EventType.tasks_bulk_created: self._ctx_tasks_bulk_created,
        }
        handler = dispatch.get(event.type)
        if handler is None:
            return []
        try:
            return await handler(event, project_id)
        except Exception as e:
            logger.warning(f'Context handler {event.type.value} failed: {e}')
            return []

    async def _ctx_memory_added(
        self, event: ReconciliationEvent, project_id: str,
    ) -> list[ContextItem]:
        """Search for related/duplicate memories based on content preview."""
        preview = event.payload.get('content_preview', '')
        if not preview:
            return []
        category = event.payload.get('category')
        categories = [category] if category else None
        results = await self.memory.search(
            query=preview,
            project_id=project_id,
            categories=categories,
            limit=self._search_limit,
        )
        return [
            ContextItem(
                id=r.id,
                source=r.source_store.value,
                formatted=_format_memory_result(r),
            )
            for r in results
        ]

    async def _ctx_memory_deleted(
        self, event: ReconciliationEvent, project_id: str,
    ) -> list[ContextItem]:
        """Search for references to the deleted memory."""
        memory_id = event.payload.get('memory_id', '')
        if not memory_id:
            return []
        results = await self.memory.search(
            query=f'memory {memory_id}',
            project_id=project_id,
            limit=3,
        )
        return [
            ContextItem(
                id=r.id,
                source=r.source_store.value,
                formatted=_format_memory_result(r),
            )
            for r in results
        ]

    async def _ctx_task_event(
        self, event: ReconciliationEvent, project_id: str,
    ) -> list[ContextItem]:
        """Fetch task details and its memory hints."""
        task_id = event.payload.get('task_id', '')
        if not task_id or not self.taskmaster:
            return []
        task = await self.taskmaster.get_task(
            task_id=str(task_id),
            project_root=self.project_root,
        )
        if not task:
            return []
        items = [
            ContextItem(
                id=f'task:{task_id}',
                source='task',
                formatted=_format_task(task),
            ),
        ]
        # If task has memory hints with queries, search for each
        hints = (task.get('metadata') or {}).get('memory_hints', {})
        queries = hints.get('queries', []) if isinstance(hints, dict) else []
        for query in queries[:3]:  # cap hint queries
            try:
                results = await self.memory.search(
                    query=query,
                    project_id=project_id,
                    limit=3,
                )
                for r in results:
                    items.append(ContextItem(
                        id=r.id,
                        source=r.source_store.value,
                        formatted=_format_memory_result(r),
                    ))
            except Exception:
                pass
        return items

    async def _ctx_task_deleted(
        self, event: ReconciliationEvent, project_id: str,
    ) -> list[ContextItem]:
        """Search for orphaned references to the deleted task."""
        task_id = event.payload.get('task_id', '')
        if not task_id:
            return []
        results = await self.memory.search(
            query=f'task {task_id}',
            project_id=project_id,
            limit=3,
        )
        return [
            ContextItem(
                id=r.id,
                source=r.source_store.value,
                formatted=_format_memory_result(r),
            )
            for r in results
        ]

    async def _ctx_episode_added(
        self, event: ReconciliationEvent, project_id: str,
    ) -> list[ContextItem]:
        """Search for entities related to the new episode."""
        preview = event.payload.get('content_preview', '')
        if not preview:
            return []
        results = await self.memory.search(
            query=preview,
            project_id=project_id,
            limit=self._search_limit,
        )
        return [
            ContextItem(
                id=r.id,
                source=r.source_store.value,
                formatted=_format_memory_result(r),
            )
            for r in results
        ]

    async def _ctx_tasks_bulk_created(
        self, event: ReconciliationEvent, project_id: str,
    ) -> list[ContextItem]:
        """Fetch parent task for bulk-created tasks."""
        parent_id = event.payload.get('parent_task_id', '')
        if not parent_id or not self.taskmaster:
            return []
        task = await self.taskmaster.get_task(
            task_id=str(parent_id),
            project_root=self.project_root,
        )
        if not task:
            return []
        return [
            ContextItem(
                id=f'task:{parent_id}',
                source='task',
                formatted=_format_task(task),
            ),
        ]
