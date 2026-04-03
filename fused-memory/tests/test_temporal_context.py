"""Tests for temporal_context and reference_time threading through the add_episode pipeline.

Organised by pipeline layer (bottom-up):
  TestGraphitiBackendTemporalContext    — step-1 / step-2
  TestExecuteGraphitiWriteTemporalContext — step-3 / step-4
  TestAddEpisodeTemporalContext         — step-5 / step-6
  TestMCPToolAddEpisodeTemporalContext  — step-7 / step-8

  TestGraphitiBackendReferenceTime      — step-1 (reference_time)
  TestExecuteGraphitiWriteReferenceTime — step-3 (reference_time)
  TestAddEpisodeReferenceTime           — step-5 (reference_time)
  TestMCPToolAddEpisodeReferenceTime    — step-7 (reference_time)
  TestReferenceTimeTemporalContextInteraction — step-9

  TestExecuteGraphitiWriteReferenceTimeErrorHandling — step-12
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import MemoryService

# ---------------------------------------------------------------------------
# Fixtures shared across test classes
# ---------------------------------------------------------------------------


@pytest.fixture
def backend(mock_config):
    """GraphitiBackend with a mocked graphiti_core client."""
    b = GraphitiBackend(mock_config)
    mock_client = MagicMock()
    mock_client.add_episode = AsyncMock(return_value=None)
    b.client = mock_client
    return b


@pytest.fixture
def service(mock_config):
    """MemoryService with fully-mocked backends and durable queue."""
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti.search = AsyncMock(return_value=[])
    svc.graphiti.search_nodes = AsyncMock(return_value=[])
    svc.graphiti.retrieve_episodes = AsyncMock(return_value=[])
    svc.graphiti.remove_episode = AsyncMock()
    svc.graphiti.remove_edge = AsyncMock()
    svc.graphiti._require_client = MagicMock()

    svc.mem0 = MagicMock()
    svc.mem0.search = AsyncMock(return_value={'results': []})
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})

    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.durable_queue.enqueue_batch = AsyncMock(return_value=[])
    svc.durable_queue.close = AsyncMock()
    return svc


# ---------------------------------------------------------------------------
# Step 1: GraphitiBackend.add_episode — temporal_context tagging
# ---------------------------------------------------------------------------


class TestGraphitiBackendTemporalContext:
    """GraphitiBackend.add_episode prepends '[temporal:X] ' to source_description."""

    @pytest.mark.asyncio
    async def test_retrospective_prepends_source_description(self, backend):
        """temporal_context='retrospective' → source_description prefixed."""
        await backend.add_episode(
            name='ep',
            content='some content',
            source_description='session notes',
            temporal_context='retrospective',
        )
        call_kwargs = backend.client.add_episode.call_args[1]
        assert call_kwargs['source_description'] == '[temporal:retrospective] session notes'

    @pytest.mark.asyncio
    async def test_planning_prepends_source_description(self, backend):
        """temporal_context='planning' → source_description prefixed correctly."""
        await backend.add_episode(
            name='ep',
            content='content',
            source_description='planning doc',
            temporal_context='planning',
        )
        call_kwargs = backend.client.add_episode.call_args[1]
        assert call_kwargs['source_description'] == '[temporal:planning] planning doc'

    @pytest.mark.asyncio
    async def test_current_prepends_source_description(self, backend):
        """temporal_context='current' → source_description prefixed correctly."""
        await backend.add_episode(
            name='ep',
            content='content',
            source_description='live feed',
            temporal_context='current',
        )
        call_kwargs = backend.client.add_episode.call_args[1]
        assert call_kwargs['source_description'] == '[temporal:current] live feed'

    @pytest.mark.asyncio
    async def test_none_passes_source_description_unchanged(self, backend):
        """temporal_context=None → source_description passed through unchanged."""
        await backend.add_episode(
            name='ep',
            content='content',
            source_description='session notes',
            temporal_context=None,
        )
        call_kwargs = backend.client.add_episode.call_args[1]
        assert call_kwargs['source_description'] == 'session notes'

    @pytest.mark.asyncio
    async def test_default_temporal_context_is_none(self, backend):
        """Omitting temporal_context defaults to None — no prefix added."""
        await backend.add_episode(
            name='ep',
            content='content',
            source_description='notes',
        )
        call_kwargs = backend.client.add_episode.call_args[1]
        assert call_kwargs['source_description'] == 'notes'

    @pytest.mark.asyncio
    async def test_empty_source_description_with_temporal_context(self, backend):
        """Empty source_description with temporal_context → '[temporal:X] '."""
        await backend.add_episode(
            name='ep',
            content='content',
            source_description='',
            temporal_context='retrospective',
        )
        call_kwargs = backend.client.add_episode.call_args[1]
        assert call_kwargs['source_description'] == '[temporal:retrospective] '


# ---------------------------------------------------------------------------
# Step 3: MemoryService._execute_graphiti_write — temporal_context extraction
# ---------------------------------------------------------------------------


class TestExecuteGraphitiWriteTemporalContext:
    """_execute_graphiti_write pops temporal_context from payload and forwards it."""

    @pytest.mark.asyncio
    async def test_temporal_context_in_payload_forwarded_to_graphiti(self, service):
        """Payload with temporal_context='retrospective' → graphiti.add_episode called with it."""
        payload = {
            'uuid': 'test-uuid',
            'name': 'episode_test',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': 'notes',
            'temporal_context': 'retrospective',
        }
        await service._execute_graphiti_write('add_episode', payload)
        call_kwargs = service.graphiti.add_episode.call_args[1]
        assert call_kwargs.get('temporal_context') == 'retrospective'

    @pytest.mark.asyncio
    async def test_missing_temporal_context_passes_none(self, service):
        """Payload without temporal_context → graphiti.add_episode called with temporal_context=None."""
        payload = {
            'uuid': 'test-uuid',
            'name': 'episode_test',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': 'notes',
        }
        await service._execute_graphiti_write('add_episode', payload)
        call_kwargs = service.graphiti.add_episode.call_args[1]
        assert call_kwargs.get('temporal_context') is None

    @pytest.mark.asyncio
    async def test_temporal_context_popped_not_leaked_into_graphiti_call(self, service):
        """temporal_context must not appear in leftover payload passed to graphiti."""
        payload = {
            'uuid': 'test-uuid',
            'name': 'episode_test',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': 'notes',
            'temporal_context': 'planning',
        }
        await service._execute_graphiti_write('add_episode', payload)
        # temporal_context is an explicit kwarg, not part of remaining payload
        # The payload dict itself should have had temporal_context popped
        assert 'temporal_context' not in payload


# ---------------------------------------------------------------------------
# Step 5: MemoryService.add_episode — temporal_context in enqueue payload
# ---------------------------------------------------------------------------


class TestAddEpisodeTemporalContext:
    """MemoryService.add_episode includes temporal_context in the enqueue payload."""

    @pytest.mark.asyncio
    async def test_temporal_context_planning_in_enqueue_payload(self, service):
        """add_episode(temporal_context='planning') → enqueue payload has 'planning'."""
        await service.add_episode(
            content='planning content',
            project_id='test',
            temporal_context='planning',
        )
        call_kwargs = service.durable_queue.enqueue.call_args[1]
        payload = call_kwargs['payload']
        assert payload.get('temporal_context') == 'planning'

    @pytest.mark.asyncio
    async def test_temporal_context_none_in_enqueue_payload(self, service):
        """add_episode(temporal_context=None) → enqueue payload has None."""
        await service.add_episode(
            content='test content',
            project_id='test',
            temporal_context=None,
        )
        call_kwargs = service.durable_queue.enqueue.call_args[1]
        payload = call_kwargs['payload']
        assert payload.get('temporal_context') is None

    @pytest.mark.asyncio
    async def test_temporal_context_retrospective_in_enqueue_payload(self, service):
        """add_episode(temporal_context='retrospective') → enqueue payload has 'retrospective'."""
        await service.add_episode(
            content='past event',
            project_id='test',
            temporal_context='retrospective',
        )
        call_kwargs = service.durable_queue.enqueue.call_args[1]
        payload = call_kwargs['payload']
        assert payload.get('temporal_context') == 'retrospective'

    @pytest.mark.asyncio
    async def test_default_temporal_context_omitted_is_none(self, service):
        """Calling add_episode without temporal_context → payload has None (default)."""
        await service.add_episode(
            content='default content',
            project_id='test',
        )
        call_kwargs = service.durable_queue.enqueue.call_args[1]
        payload = call_kwargs['payload']
        assert payload.get('temporal_context') is None


# ---------------------------------------------------------------------------
# Step 7: MCP tool add_episode — validation + passthrough
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory_service():
    """Mocked MemoryService for MCP tool tests."""
    svc = MagicMock()
    svc.add_episode = AsyncMock(
        return_value=MagicMock(
            model_dump=MagicMock(
                return_value={
                    'episode_id': 'ep-test-id',
                    'status': 'queued',
                    'message': 'Episode queued for processing in project test',
                }
            )
        )
    )
    return svc


@pytest.fixture
def mcp_server(mock_memory_service):
    return create_mcp_server(mock_memory_service)


class TestMCPToolAddEpisodeTemporalContext:
    """MCP tool add_episode validates and forwards temporal_context."""

    @pytest.mark.asyncio
    async def test_valid_retrospective_passed_to_memory_service(
        self, mcp_server, mock_memory_service
    ):
        """temporal_context='retrospective' → memory_service.add_episode called with it."""
        await mcp_server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'some content',
                'project_id': 'test',
                'temporal_context': 'retrospective',
            },
        )
        mock_memory_service.add_episode.assert_called_once()
        _, kwargs = mock_memory_service.add_episode.call_args
        assert kwargs.get('temporal_context') == 'retrospective'

    @pytest.mark.asyncio
    async def test_valid_planning_passed_to_memory_service(
        self, mcp_server, mock_memory_service
    ):
        """temporal_context='planning' → memory_service.add_episode called with it."""
        await mcp_server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'some content',
                'project_id': 'test',
                'temporal_context': 'planning',
            },
        )
        _, kwargs = mock_memory_service.add_episode.call_args
        assert kwargs.get('temporal_context') == 'planning'

    @pytest.mark.asyncio
    async def test_invalid_temporal_context_returns_error(
        self, mcp_server, mock_memory_service
    ):
        """temporal_context='future' (invalid) → error dict returned, service not called."""
        result = await mcp_server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'some content',
                'project_id': 'test',
                'temporal_context': 'future',
            },
        )
        assert 'error' in result
        mock_memory_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_temporal_context_accepted(
        self, mcp_server, mock_memory_service
    ):
        """temporal_context=None (omitted) → accepted, memory_service called with None."""
        await mcp_server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'some content',
                'project_id': 'test',
            },
        )
        mock_memory_service.add_episode.assert_called_once()
        _, kwargs = mock_memory_service.add_episode.call_args
        assert kwargs.get('temporal_context') is None

    @pytest.mark.asyncio
    async def test_current_temporal_context_accepted(
        self, mcp_server, mock_memory_service
    ):
        """temporal_context='current' → accepted, memory_service called with 'current'."""
        await mcp_server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'live event',
                'project_id': 'test',
                'temporal_context': 'current',
            },
        )
        _, kwargs = mock_memory_service.add_episode.call_args
        assert kwargs.get('temporal_context') == 'current'


# ---------------------------------------------------------------------------
# Step 1 (reference_time): GraphitiBackend.add_episode — reference_time forwarding
# ---------------------------------------------------------------------------


class TestGraphitiBackendReferenceTime:
    """GraphitiBackend.add_episode forwards reference_time to the Graphiti client.

    This verifies existing backend behavior that becomes load-bearing once we
    thread reference_time from the MCP layer through the full pipeline.
    """

    @pytest.mark.asyncio
    async def test_explicit_datetime_passed_to_client(self, backend):
        """Explicit reference_time datetime → client receives exactly that datetime."""
        ref_dt = datetime(2026, 3, 22, 12, 0, 0, tzinfo=UTC)
        await backend.add_episode(
            name='ep',
            content='historical content',
            source_description='session notes',
            reference_time=ref_dt,
        )
        call_kwargs = backend.client.add_episode.call_args[1]
        assert call_kwargs['reference_time'] == ref_dt

    @pytest.mark.asyncio
    async def test_none_reference_time_defaults_to_now(self, backend):
        """reference_time=None → client receives a datetime close to now (UTC)."""
        before = datetime.now(UTC)
        await backend.add_episode(
            name='ep',
            content='content',
            source_description='notes',
            reference_time=None,
        )
        after = datetime.now(UTC)
        call_kwargs = backend.client.add_episode.call_args[1]
        ref = call_kwargs['reference_time']
        # Should be a UTC-aware datetime between before and after
        assert ref.tzinfo is not None
        assert before <= ref <= after

    @pytest.mark.asyncio
    async def test_default_omission_same_as_none(self, backend):
        """Omitting reference_time defaults to None → client gets datetime.now(UTC)."""
        before = datetime.now(UTC)
        await backend.add_episode(
            name='ep',
            content='content',
            source_description='notes',
            # reference_time intentionally omitted
        )
        after = datetime.now(UTC)
        call_kwargs = backend.client.add_episode.call_args[1]
        ref = call_kwargs['reference_time']
        assert ref.tzinfo is not None
        assert before <= ref <= after


# ---------------------------------------------------------------------------
# Step 3 (reference_time): MemoryService._execute_graphiti_write — extraction
# ---------------------------------------------------------------------------


class TestExecuteGraphitiWriteReferenceTime:
    """_execute_graphiti_write pops reference_time ISO string from payload and
    forwards it as a parsed datetime to graphiti.add_episode.

    These tests will FAIL until step-4 implementation.
    """

    @pytest.mark.asyncio
    async def test_reference_time_iso_string_forwarded_as_datetime(self, service):
        """Payload with 'reference_time' ISO string → graphiti.add_episode called with parsed datetime."""
        ref_iso = '2026-03-22T12:00:00+00:00'
        ref_expected = datetime(2026, 3, 22, 12, 0, 0, tzinfo=UTC)
        payload = {
            'uuid': 'test-uuid',
            'name': 'episode_test',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': 'notes',
            'reference_time': ref_iso,
        }
        await service._execute_graphiti_write('add_episode', payload)
        call_kwargs = service.graphiti.add_episode.call_args[1]
        assert call_kwargs.get('reference_time') == ref_expected

    @pytest.mark.asyncio
    async def test_missing_reference_time_passes_none(self, service):
        """Payload without reference_time → graphiti.add_episode called with reference_time=None."""
        payload = {
            'uuid': 'test-uuid',
            'name': 'episode_test',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': 'notes',
        }
        await service._execute_graphiti_write('add_episode', payload)
        call_kwargs = service.graphiti.add_episode.call_args[1]
        assert call_kwargs.get('reference_time') is None

    @pytest.mark.asyncio
    async def test_reference_time_popped_not_leaked(self, service):
        """reference_time must be popped from payload dict (not passed via remaining kwargs)."""
        ref_iso = '2026-03-22T00:00:00+00:00'
        payload = {
            'uuid': 'test-uuid',
            'name': 'episode_test',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': 'notes',
            'reference_time': ref_iso,
        }
        await service._execute_graphiti_write('add_episode', payload)
        # The key should have been popped from the payload dict
        assert 'reference_time' not in payload


# ---------------------------------------------------------------------------
# Step 5 (reference_time): MemoryService.add_episode — serialization into payload
# ---------------------------------------------------------------------------


class TestAddEpisodeReferenceTime:
    """MemoryService.add_episode serializes reference_time datetime to ISO string
    in the durable queue enqueue payload.

    These tests will FAIL until step-6 implementation.
    """

    @pytest.mark.asyncio
    async def test_explicit_datetime_serialized_to_iso_in_payload(self, service):
        """add_episode(reference_time=datetime(2026,3,22,...)) → payload has ISO string."""
        ref_dt = datetime(2026, 3, 22, 0, 0, 0, tzinfo=UTC)
        await service.add_episode(
            content='historical content',
            project_id='test',
            reference_time=ref_dt,
        )
        call_kwargs = service.durable_queue.enqueue.call_args[1]
        payload = call_kwargs['payload']
        assert payload.get('reference_time') == ref_dt.isoformat()

    @pytest.mark.asyncio
    async def test_none_reference_time_in_payload(self, service):
        """add_episode(reference_time=None) → payload has reference_time=None or key absent."""
        await service.add_episode(
            content='test content',
            project_id='test',
            reference_time=None,
        )
        call_kwargs = service.durable_queue.enqueue.call_args[1]
        payload = call_kwargs['payload']
        # Either None or absent — both acceptable
        assert payload.get('reference_time') is None

    @pytest.mark.asyncio
    async def test_default_omission_reference_time_is_none(self, service):
        """Calling add_episode without reference_time → payload has None (default)."""
        await service.add_episode(
            content='default content',
            project_id='test',
        )
        call_kwargs = service.durable_queue.enqueue.call_args[1]
        payload = call_kwargs['payload']
        assert payload.get('reference_time') is None


# ---------------------------------------------------------------------------
# Step 7 (reference_time): MCP tool add_episode — validation + passthrough
# ---------------------------------------------------------------------------


class TestMCPToolAddEpisodeReferenceTime:
    """MCP add_episode tool accepts reference_time as ISO 8601 string, validates,
    parses to datetime, and passes to memory_service.add_episode.

    These tests will FAIL until step-8 implementation.
    """

    @pytest.mark.asyncio
    async def test_valid_iso_string_parsed_and_forwarded(
        self, mcp_server, mock_memory_service
    ):
        """Valid ISO string '2026-03-22T00:00:00+00:00' → service called with parsed datetime."""
        iso_str = '2026-03-22T00:00:00+00:00'
        expected_dt = datetime(2026, 3, 22, 0, 0, 0, tzinfo=UTC)
        await mcp_server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'historical content',
                'project_id': 'test',
                'reference_time': iso_str,
            },
        )
        mock_memory_service.add_episode.assert_called_once()
        _, kwargs = mock_memory_service.add_episode.call_args
        assert kwargs.get('reference_time') == expected_dt

    @pytest.mark.asyncio
    async def test_invalid_iso_string_returns_error(
        self, mcp_server, mock_memory_service
    ):
        """Invalid string 'not-a-date' → error dict returned, service not called."""
        result = await mcp_server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'some content',
                'project_id': 'test',
                'reference_time': 'not-a-date',
            },
        )
        assert 'error' in result
        mock_memory_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_reference_time_passes_none_to_service(
        self, mcp_server, mock_memory_service
    ):
        """reference_time omitted → service called with reference_time=None."""
        await mcp_server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'some content',
                'project_id': 'test',
            },
        )
        mock_memory_service.add_episode.assert_called_once()
        _, kwargs = mock_memory_service.add_episode.call_args
        assert kwargs.get('reference_time') is None

    @pytest.mark.asyncio
    async def test_iso_string_with_timezone_offset_accepted(
        self, mcp_server, mock_memory_service
    ):
        """ISO 8601 string with timezone offset is accepted and parsed correctly."""
        iso_str = '2026-03-22T14:30:00+05:30'
        expected_dt = datetime.fromisoformat(iso_str)
        await mcp_server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'some content',
                'project_id': 'test',
                'reference_time': iso_str,
            },
        )
        mock_memory_service.add_episode.assert_called_once()
        _, kwargs = mock_memory_service.add_episode.call_args
        assert kwargs.get('reference_time') == expected_dt


# ---------------------------------------------------------------------------
# Step 9: Integration — reference_time + temporal_context work together
# ---------------------------------------------------------------------------


class TestReferenceTimeTemporalContextInteraction:
    """Verify reference_time and temporal_context work together correctly
    across the full mocked pipeline (MCP tool → service → _execute_graphiti_write → backend).
    """

    @pytest.mark.asyncio
    async def test_both_reference_time_and_temporal_context_forwarded(
        self, mcp_server, mock_memory_service
    ):
        """add_episode with both temporal_context='retrospective' and reference_time →
        service receives both parameters.
        """
        iso_str = '2026-03-22T00:00:00+00:00'
        expected_dt = datetime(2026, 3, 22, 0, 0, 0, tzinfo=UTC)
        await mcp_server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'historical state summary',
                'project_id': 'test',
                'temporal_context': 'retrospective',
                'reference_time': iso_str,
            },
        )
        mock_memory_service.add_episode.assert_called_once()
        _, kwargs = mock_memory_service.add_episode.call_args
        assert kwargs.get('temporal_context') == 'retrospective'
        assert kwargs.get('reference_time') == expected_dt

    @pytest.mark.asyncio
    async def test_service_serializes_both_to_queue_payload(self, service):
        """add_episode(temporal_context='retrospective', reference_time=dt) →
        enqueue payload contains both 'temporal_context' and 'reference_time' (ISO string).
        """
        ref_dt = datetime(2026, 3, 22, 0, 0, 0, tzinfo=UTC)
        await service.add_episode(
            content='historical state summary',
            project_id='test',
            temporal_context='retrospective',
            reference_time=ref_dt,
        )
        call_kwargs = service.durable_queue.enqueue.call_args[1]
        payload = call_kwargs['payload']
        assert payload.get('temporal_context') == 'retrospective'
        assert payload.get('reference_time') == ref_dt.isoformat()

    @pytest.mark.asyncio
    async def test_execute_graphiti_write_forwards_both(self, service):
        """_execute_graphiti_write with both temporal_context and reference_time →
        graphiti.add_episode called with both.
        """
        ref_iso = '2026-03-22T00:00:00+00:00'
        ref_expected = datetime(2026, 3, 22, 0, 0, 0, tzinfo=UTC)
        payload = {
            'uuid': 'test-uuid',
            'name': 'episode_test',
            'content': 'historical content',
            'source': 'text',
            'group_id': 'test',
            'source_description': 'session notes',
            'temporal_context': 'retrospective',
            'reference_time': ref_iso,
        }
        await service._execute_graphiti_write('add_episode', payload)
        call_kwargs = service.graphiti.add_episode.call_args[1]
        assert call_kwargs.get('temporal_context') == 'retrospective'
        assert call_kwargs.get('reference_time') == ref_expected


# ---------------------------------------------------------------------------
# Step 12: MemoryService._execute_graphiti_write — invalid reference_time handling
# ---------------------------------------------------------------------------


class TestExecuteGraphitiWriteReferenceTimeErrorHandling:
    """_execute_graphiti_write gracefully handles invalid reference_time values
    in the queue payload by falling back to None and logging a warning.

    These tests FAIL until step-13 wraps the fromisoformat() call in try/except.
    """

    @pytest.mark.asyncio
    async def test_invalid_reference_time_in_payload_falls_back_to_none(
        self, service, caplog
    ):
        """Payload with reference_time='not-a-date' → does NOT raise, graphiti.add_episode
        called with reference_time=None, and a warning is logged.
        """
        import logging

        payload = {
            'uuid': 'test-uuid',
            'name': 'episode_test',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': 'notes',
            'reference_time': 'not-a-date',
        }
        with caplog.at_level(logging.WARNING):
            # Must NOT raise — invalid value should be silently discarded with a warning
            await service._execute_graphiti_write('add_episode', payload)

        # graphiti.add_episode must still be called with reference_time=None
        call_kwargs = service.graphiti.add_episode.call_args[1]
        assert call_kwargs.get('reference_time') is None

        # A warning must be emitted containing the invalid value
        warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any('not-a-date' in str(w) for w in warning_texts), (
            f'Expected warning containing invalid value, got: {warning_texts}'
        )

    @pytest.mark.asyncio
    async def test_empty_string_reference_time_falls_back_to_none(
        self, service, caplog
    ):
        """Payload with reference_time='' (empty string) → does NOT raise,
        graphiti.add_episode called with reference_time=None, and a warning is logged.
        """
        import logging

        payload = {
            'uuid': 'test-uuid',
            'name': 'episode_test',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': 'notes',
            'reference_time': '',
        }
        with caplog.at_level(logging.WARNING):
            # Must NOT raise
            await service._execute_graphiti_write('add_episode', payload)

        call_kwargs = service.graphiti.add_episode.call_args[1]
        assert call_kwargs.get('reference_time') is None

        warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(w for w in warning_texts), (
            f'Expected at least one warning log, got: {warning_texts}'
        )
