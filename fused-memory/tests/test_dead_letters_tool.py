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

    def test_non_serialisable_large_payload_returns_capped_envelope(self):
        """Non-serialisable payload whose str() also exceeds the budget returns capped envelope."""
        # Self-referencing dict with extra bulk so str() representation exceeds 2 KB.
        payload: dict = {}
        payload['self'] = payload
        payload['filler'] = 'x' * (_DEAD_LETTER_PAYLOAD_MAX_BYTES * 2)

        # Sanity-guard 1: circular reference still triggers the except branch.
        with pytest.raises((TypeError, ValueError)):
            json.dumps(payload, default=str)

        # Sanity-guard 2: confirm str() representation exceeds the budget.
        assert len(str(payload).encode('utf-8')) > _DEAD_LETTER_PAYLOAD_MAX_BYTES

        result, truncated = _truncate_payload(payload)

        # Expects the capped envelope dict, not the raw str or the original object.
        assert truncated is True
        assert isinstance(result, dict)
        assert result.get('_truncated') is True
        assert 'text' in result
        assert 'original_type' in result
        # The 'text' field must fit within the byte budget — the whole point of the cap.
        assert len(result['text'].encode('utf-8')) <= _DEAD_LETTER_PAYLOAD_MAX_BYTES

    def test_unicode_heavy_payload_fits_under_utf8_budget_not_truncated(self):
        """A payload that fits in UTF-8 but exceeds ASCII-escape size must not be truncated."""
        # '日' is 3 bytes UTF-8 but 6 bytes as \uXXXX ASCII escape.
        # 500×3=1500 bytes UTF-8 (fits 2048); 500×6=3000 bytes ASCII-escaped (exceeds 2048).
        payload = {'blob': '日' * 500}

        # Sanity-guard 1: UTF-8 form fits the budget.
        assert len(json.dumps(payload, default=str, ensure_ascii=False).encode('utf-8')) <= _DEAD_LETTER_PAYLOAD_MAX_BYTES
        # Sanity-guard 2: ASCII-escaped form exceeds the budget.
        assert len(json.dumps(payload, default=str).encode('utf-8')) > _DEAD_LETTER_PAYLOAD_MAX_BYTES

        result, truncated = _truncate_payload(payload)

        # With ensure_ascii=True (the bug), the inflated form triggers truncation.
        # With ensure_ascii=False (the fix), the UTF-8 form fits and passes through.
        assert truncated is False
        assert result == payload


# ── step-3 tests (delete_dead_letters MCP tool) ────────────────────────────


def _make_delete_mock_service(delete_dead_return=None):
    """Build a minimal mock MemoryService with delete_dead wired on durable_queue."""
    svc = AsyncMock()
    svc.durable_queue = MagicMock()
    svc.durable_queue.delete_dead = AsyncMock(
        return_value=delete_dead_return or {'deleted': [], 'not_found': []}
    )
    return svc


class TestDeleteDeadLetters:
    """MCP-level tests for the delete_dead_letters tool."""

    @pytest.mark.asyncio
    async def test_routes_to_durable_queue_delete_dead(self):
        """Tool calls durable_queue.delete_dead with correct group_id and ids."""
        svc = _make_delete_mock_service(delete_dead_return={'deleted': [1, 2, 3], 'not_found': []})
        server = create_mcp_server(svc)

        await server._tool_manager.call_tool(
            'delete_dead_letters',
            {'project_id': 'proj1', 'ids': [1, 2, 3]},
        )

        svc.durable_queue.delete_dead.assert_called_once_with(
            group_id='proj1', ids=[1, 2, 3],
        )

    @pytest.mark.asyncio
    async def test_envelope_passthrough(self):
        """Tool returns the exact envelope from durable_queue.delete_dead."""
        envelope = {'deleted': [1, 2], 'not_found': [3]}
        svc = _make_delete_mock_service(delete_dead_return=envelope)
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'delete_dead_letters',
            {'project_id': 'proj1', 'ids': [1, 2, 3]},
        )

        assert result == envelope

    @pytest.mark.asyncio
    async def test_no_durable_queue_returns_configuration_error(self):
        """When durable_queue is None, tool returns ConfigurationError dict without raising."""
        svc = AsyncMock()
        svc.durable_queue = None
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'delete_dead_letters',
            {'project_id': 'proj1', 'ids': [1, 2]},
        )

        assert result == {'error': 'Queue not initialized', 'error_type': 'ConfigurationError'}

    @pytest.mark.asyncio
    async def test_invalid_project_id_returns_validation_error(self):
        """Invalid project_id (empty, injection chars) returns validation error dict."""
        svc = _make_delete_mock_service()
        server = create_mcp_server(svc)

        # Empty project_id
        result = await server._tool_manager.call_tool(
            'delete_dead_letters',
            {'project_id': '', 'ids': [1]},
        )

        assert result.get('error_type') == 'ValidationError'
        svc.durable_queue.delete_dead.assert_not_called()

    @pytest.mark.asyncio
    async def test_project_id_with_injection_chars_rejected(self):
        """project_id containing prompt-injection characters is rejected."""
        svc = _make_delete_mock_service()
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'delete_dead_letters',
            {'project_id': 'proj1; DROP TABLE write_queue', 'ids': [1]},
        )

        assert result.get('error_type') == 'ValidationError'
        svc.durable_queue.delete_dead.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_ids_returns_empty_envelope(self):
        """empty ids=[] returns {'deleted': [], 'not_found': []}."""
        svc = _make_delete_mock_service(delete_dead_return={'deleted': [], 'not_found': []})
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'delete_dead_letters',
            {'project_id': 'proj1', 'ids': []},
        )

        assert result == {'deleted': [], 'not_found': []}


# ── TestDeleteDeadLettersIdGuard (step-1 / step-3 tests) ──────────────────────


class TestDeleteDeadLettersIdGuard:
    """Guard tests that call tool.fn directly, bypassing pydantic schema validation.

    Using server._tool_manager.get_tool('delete_dead_letters').fn(...) simulates
    internal callers and transports that bypass pydantic — the exact surface the
    guard must protect.
    """

    @pytest.mark.asyncio
    async def test_uuid_string_in_ids_rejected_with_validation_error_envelope(self):
        """UUID string in ids is rejected by the guard before reaching delete_dead."""
        svc = _make_delete_mock_service()
        server = create_mcp_server(svc)
        _tool = server._tool_manager.get_tool('delete_dead_letters')
        assert _tool is not None, 'delete_dead_letters tool not registered'
        tool_fn = _tool.fn

        result = await tool_fn(
            project_id='proj1',
            ids=['550e8400-e29b-41d4-a716-446655440000'],
        )

        assert isinstance(result, dict), 'expected a dict envelope'
        assert result.get('error_type') == 'ValidationError', (
            f"expected 'ValidationError', got: {result}"
        )
        svc.durable_queue.delete_dead.assert_not_called()

    @pytest.mark.asyncio
    async def test_bool_in_ids_rejected_because_bool_is_int_subclass(self):
        """Bool values in ids are rejected even though isinstance(True, int) is True.

        Python's bool is a subclass of int, so a guard using only isinstance(i, int)
        would silently accept True/False as 1/0 and pass them to delete_dead.
        The guard must also check not isinstance(i, bool) to close this gap.
        """
        svc = _make_delete_mock_service()
        server = create_mcp_server(svc)
        _tool = server._tool_manager.get_tool('delete_dead_letters')
        assert _tool is not None, 'delete_dead_letters tool not registered'
        tool_fn = _tool.fn

        result = await tool_fn(
            project_id='proj1',
            ids=[True, False],
        )

        assert isinstance(result, dict), 'expected a dict envelope'
        assert result.get('error_type') == 'ValidationError', (
            f"expected 'ValidationError', got: {result}"
        )
        svc.durable_queue.delete_dead.assert_not_called()
