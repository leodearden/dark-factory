"""Tests for temporal_context threading through the add_episode pipeline.

Organised by pipeline layer (bottom-up):
  TestGraphitiBackendTemporalContext    — step-1 / step-2
  TestExecuteGraphitiWriteTemporalContext — step-3 / step-4
  TestAddEpisodeTemporalContext         — step-5 / step-6
  TestMCPToolAddEpisodeTemporalContext  — step-7 / step-8
"""

from __future__ import annotations

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
