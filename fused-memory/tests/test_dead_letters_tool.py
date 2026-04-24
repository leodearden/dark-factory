"""MCP-level tests for the get_dead_letters tool."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.event_queue import EventQueue
from fused_memory.server.tools import create_mcp_server


# ── helpers ────────────────────────────────────────────────────────────────


def _make_mock_service(get_dead_items_return=None):
    """Build a minimal mock MemoryService with a durable_queue."""
    svc = AsyncMock()
    svc.durable_queue = MagicMock()
    svc.durable_queue.get_dead_items = AsyncMock(
        return_value=get_dead_items_return or []
    )
    return svc


_FAKE_DEAD_ITEMS = [
    {
        'id': 11,
        'group_id': 'proj1',
        'operation': 'add_episode',
        'payload': {'content': 'newer', 'group_id': 'proj1'},
        'attempts': 3,
        'error': 'RuntimeError: timeout',
        'created_at': 1_700_000_011.0,
    },
    {
        'id': 10,
        'group_id': 'proj1',
        'operation': 'add_episode',
        'payload': {'content': 'older', 'group_id': 'proj1'},
        'attempts': 3,
        'error': 'RuntimeError: timeout',
        'created_at': 1_700_000_010.0,
    },
]


# ── step-9 tests ───────────────────────────────────────────────────────────


class TestGetDeadLettersDurableQueue:
    """get_dead_letters with only durable_queue (no event_queue)."""

    @pytest.mark.asyncio
    async def test_returns_items_from_durable_queue(self):
        """Tool merges durable-queue dead items into the 'items' list."""
        svc = _make_mock_service(get_dead_items_return=_FAKE_DEAD_ITEMS)
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'get_dead_letters',
            {'project_id': 'proj1', 'limit': 50},
        )

        # Top-level structure
        assert 'items' in result, f'Expected items key; got: {result}'
        assert 'counts' in result, f'Expected counts key; got: {result}'

        items = result['items']
        assert len(items) == 2

        for item in items:
            assert item['source'] == 'durable_queue'
            assert 'operation' in item
            assert 'payload' in item
            assert 'error' in item
            assert 'attempts' in item
            assert 'timestamp' in item or 'created_at' in item

    @pytest.mark.asyncio
    async def test_passes_group_id_and_limit_to_get_dead_items(self):
        """get_dead_items is called with group_id=project_id, limit=limit."""
        svc = _make_mock_service(get_dead_items_return=_FAKE_DEAD_ITEMS)
        server = create_mcp_server(svc)

        await server._tool_manager.call_tool(
            'get_dead_letters',
            {'project_id': 'proj1', 'limit': 50},
        )

        svc.durable_queue.get_dead_items.assert_called_once_with(
            group_id='proj1', limit=50,
        )

    @pytest.mark.asyncio
    async def test_no_durable_queue_returns_empty(self):
        """When durable_queue is None, tool returns empty items without raising."""
        svc = AsyncMock()
        svc.durable_queue = None
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'get_dead_letters',
            {},
        )

        assert 'items' in result
        assert result['items'] == []
        assert 'error' not in result


# ── helpers for EventQueue tests ───────────────────────────────────────────


def _make_recon_event(project_id: str) -> ReconciliationEvent:
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=EventType.task_created,
        source=EventSource.agent,
        project_id=project_id,
        timestamp=datetime.now(UTC),
        payload={'key': 'value'},
    )


def _make_failing_buf() -> AsyncMock:
    """A mock EventBuffer whose push raises non-retriable ValueError."""
    buf = AsyncMock()
    buf.push = AsyncMock(side_effect=ValueError('schema mismatch'))
    return buf


# ── step-11 tests ──────────────────────────────────────────────────────────


class TestGetDeadLettersEventQueue:
    """get_dead_letters with event_queue wired in."""

    @pytest.mark.asyncio
    async def test_event_queue_items_filtered_by_project_id(self, tmp_path):
        """Items from event_queue filtered by project_id; each has required fields."""
        dl = tmp_path / 'dl.jsonl'
        buf = _make_failing_buf()
        eq = EventQueue(
            buf,
            dead_letter_path=dl,
            maxsize=100,
            retry_initial_seconds=0.01,
            retry_max_seconds=0.05,
            shutdown_flush_seconds=1.0,
        )
        await eq.start()
        try:
            # Enqueue 3 events for proj-a, 2 for proj-b
            for _ in range(3):
                eq.enqueue(_make_recon_event('proj-a'))
            for _ in range(2):
                eq.enqueue(_make_recon_event('proj-b'))
            # Wait for drainer to dead-letter all of them
            await asyncio.wait_for(eq._queue.join(), timeout=2.0)
            assert eq.stats()['dead_letters'] == 5
        finally:
            await eq.close()

        svc = AsyncMock()
        svc.durable_queue = None
        server = create_mcp_server(svc, event_queue=eq)

        result = await server._tool_manager.call_tool(
            'get_dead_letters',
            {'project_id': 'proj-a', 'limit': 10},
        )

        assert 'items' in result, f'Expected items; got: {result}'
        items = result['items']
        assert len(items) == 3, f'Expected 3 proj-a items; got {len(items)}'

        for item in items:
            assert item['source'] == 'event_queue'
            assert 'type' in item
            assert 'payload' in item
            assert 'reason' in item
            assert 'timestamp' in item
            assert 'attempts' in item

    @pytest.mark.asyncio
    async def test_event_queue_merges_with_durable_queue_without_project_id(self, tmp_path):
        """Without project_id, items from both sources appear in the result."""
        dl = tmp_path / 'dl2.jsonl'
        buf = _make_failing_buf()
        eq = EventQueue(
            buf,
            dead_letter_path=dl,
            maxsize=100,
            retry_initial_seconds=0.01,
            retry_max_seconds=0.05,
            shutdown_flush_seconds=1.0,
        )
        await eq.start()
        try:
            for _ in range(3):
                eq.enqueue(_make_recon_event('proj-a'))
            for _ in range(2):
                eq.enqueue(_make_recon_event('proj-b'))
            await asyncio.wait_for(eq._queue.join(), timeout=2.0)
        finally:
            await eq.close()

        svc = _make_mock_service(get_dead_items_return=_FAKE_DEAD_ITEMS)
        server = create_mcp_server(svc, event_queue=eq)

        result = await server._tool_manager.call_tool(
            'get_dead_letters',
            {'limit': 100},
        )

        assert 'items' in result
        items = result['items']
        sources = {item['source'] for item in items}
        assert 'durable_queue' in sources
        assert 'event_queue' in sources
        # 2 from durable_queue + 5 from event_queue = 7 total
        assert len(items) == 7, f'Expected 7 merged items; got {len(items)}'
