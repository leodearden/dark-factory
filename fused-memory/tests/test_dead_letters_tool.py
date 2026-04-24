"""MCP-level tests for the get_dead_letters tool."""

from __future__ import annotations

import asyncio
import json
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
from fused_memory.server.tools import (
    _DEAD_LETTER_PAYLOAD_MAX_BYTES,
    _truncate_payload,
    create_mcp_server,
)

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


# ── step-13 tests ──────────────────────────────────────────────────────────


class TestGetDeadLettersPayloadTruncation:
    """Payload truncation: items whose serialised payload exceeds 2048 bytes."""

    @pytest.mark.asyncio
    async def test_large_durable_queue_payload_is_truncated(self):
        """Durable-queue items with large payloads are truncated and flagged."""
        big_payload = {'blob': 'x' * 5000}
        dead_item = {
            'id': 99,
            'group_id': 'proj1',
            'operation': 'add_episode',
            'payload': big_payload,
            'attempts': 3,
            'error': 'timeout',
            'created_at': 1_700_000_099.0,
        }
        svc = _make_mock_service(get_dead_items_return=[dead_item])
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool('get_dead_letters', {})

        assert 'items' in result
        items = result['items']
        assert len(items) == 1
        item = items[0]
        assert item.get('payload_truncated') is True
        # Truncated payload is a stable-typed envelope dict, never a bare string.
        payload = item['payload']
        assert isinstance(payload, dict), f'Expected dict envelope, got {type(payload)}'
        assert payload.get('_truncated') is True
        assert 'text' in payload
        assert 'original_type' in payload
        # The 'text' field must fit within the byte budget.
        assert len(payload['text'].encode()) <= 2048

    @pytest.mark.asyncio
    async def test_small_durable_queue_payload_not_truncated(self):
        """Small payloads pass through unchanged without a truncation flag."""
        small_payload = {'content': 'hello'}
        dead_item = {
            'id': 1,
            'group_id': 'proj1',
            'operation': 'add_episode',
            'payload': small_payload,
            'attempts': 1,
            'error': 'timeout',
            'created_at': 1_700_000_001.0,
        }
        svc = _make_mock_service(get_dead_items_return=[dead_item])
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool('get_dead_letters', {})

        items = result['items']
        assert len(items) == 1
        item = items[0]
        # no truncation flag on small payloads
        assert not item.get('payload_truncated', False)
        assert item['payload'] == small_payload

    @pytest.mark.asyncio
    async def test_large_event_queue_payload_is_truncated(self, tmp_path):
        """Event-queue items with large payloads are also truncated and flagged."""
        import json as _json

        dl = tmp_path / 'dl.jsonl'
        big_payload = {'blob': 'x' * 5000}
        event = _make_recon_event('proj-trunc')
        # Manually write a record with an oversized payload into the JSONL file.
        record = {
            'event': {
                **event.model_dump(mode='json'),
                'payload': big_payload,
            },
            'reason': 'non_retriable',
            'attempts': 1,
            'failed_at': '2026-01-01T00:00:00+00:00',
        }
        dl.write_text(_json.dumps(record) + '\n')

        # Use start/close symmetry so the test is resilient to future init
        # changes in EventQueue (drainer task lifecycle etc.).  No events are
        # enqueued — the queue drains immediately on close.
        buf = AsyncMock()
        eq = EventQueue(buf, dead_letter_path=dl)
        await eq.start()
        try:
            svc = AsyncMock()
            svc.durable_queue = None
            server = create_mcp_server(svc, event_queue=eq)

            result = await server._tool_manager.call_tool(
                'get_dead_letters',
                {'project_id': 'proj-trunc'},
            )
        finally:
            await eq.close()

        items = result['items']
        assert len(items) == 1
        item = items[0]
        assert item.get('payload_truncated') is True
        # Truncated payload is a stable-typed envelope dict, never a bare string.
        payload = item['payload']
        assert isinstance(payload, dict), f'Expected dict envelope, got {type(payload)}'
        assert payload.get('_truncated') is True
        assert 'text' in payload
        assert 'original_type' in payload
        # The 'text' field must fit within the byte budget.
        assert len(payload['text'].encode()) <= 2048


# ── _truncate_payload unit tests ───────────────────────────────────────────


class TestTruncatePayloadHardening:
    """Direct unit tests for _truncate_payload edge cases."""

    def test_non_serialisable_small_payload_returned_as_string_with_truncated_true(self):
        """Non-JSON-serialisable payload that fits the budget must NOT be returned raw."""
        # Build a self-referencing dict — reliably raises ValueError from json.dumps.
        payload: dict = {}
        payload['self'] = payload

        # Sanity-guard 1: confirm the payload genuinely triggers the except branch.
        with pytest.raises((TypeError, ValueError)):
            json.dumps(payload, default=str)

        # Sanity-guard 2: confirm str() form fits under the budget.
        assert len(str(payload).encode('utf-8')) < _DEAD_LETTER_PAYLOAD_MAX_BYTES

        result, truncated = _truncate_payload(payload)

        # The buggy behaviour returns (payload, False); the fix must return (str, True).
        assert truncated is True
        assert isinstance(result, str)
        assert result == str(payload)
        assert result is not payload
