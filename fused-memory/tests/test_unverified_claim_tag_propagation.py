"""The `unverified_claim` tag must reach the DERIVED artefacts, not just the
MCP response (task 3142, PRD leaf pi).

The harm in reify esc-5603-1 was five false GRAPHITI EDGES extracted from one
episode. A tag that lived only on the tool's return dict would have labelled
none of them. So the tag rides the identical channel `temporal_context` does:
add_episode parameter -> durable-queue payload key -> '[unverified_claim] '
source_description prefix on the Graphiti episodic node, and (see the Mem0 half
below) metadata['unverified_claim']=True on every derived fact.

add_episode deliberately never persists its `metadata` argument, so a metadata
key could not have carried it — this channel is the only one that reaches
persistence.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.services.memory_service import MemoryService


@pytest.fixture
def backend(mock_config):
    """GraphitiBackend with a mocked graphiti_core client."""
    b = GraphitiBackend(mock_config)
    mock_client = MagicMock()
    mock_client.add_episode = AsyncMock(return_value=None)
    b.client = mock_client
    b._client_for = MagicMock(return_value=mock_client)
    return b


@pytest.fixture
def service(mock_config):
    """MemoryService with fully-mocked backends and durable queue."""
    from _fm_helpers import install_identity_mocks

    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti._require_client = MagicMock()
    install_identity_mocks(svc.graphiti)

    svc.mem0 = MagicMock()
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})

    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.durable_queue.enqueue_batch = AsyncMock(return_value=[])
    svc.durable_queue.close = AsyncMock()
    return svc


def _graphiti_payload(**overrides):
    payload = {
        'uuid': 'test-uuid',
        'name': 'episode_test',
        'content': 'test content',
        'source': 'text',
        'group_id': 'test',
        'source_description': 'notes',
    }
    payload.update(overrides)
    return payload


class TestGraphitiBackendUnverifiedClaimTag:
    """The tag lands on the episodic node's source_description, where every
    downstream reader can see it without parsing content."""

    @pytest.mark.asyncio
    async def test_tag_prefixes_source_description(self, backend):
        await backend.add_episode(
            name='e', content='c', group_id='g',
            source_description='notes', unverified_claim=True,
        )

        call_kwargs = backend.client.add_episode.call_args[1]
        assert call_kwargs['source_description'] == '[unverified_claim] notes'

    @pytest.mark.asyncio
    async def test_tag_composes_with_the_temporal_prefix(self, backend):
        await backend.add_episode(
            name='e', content='c', group_id='g',
            source_description='notes',
            temporal_context='planning', unverified_claim=True,
        )

        call_kwargs = backend.client.add_episode.call_args[1]
        assert call_kwargs['source_description'] == (
            '[unverified_claim] [temporal:planning] notes'
        )

    @pytest.mark.asyncio
    async def test_untagged_source_description_is_byte_identical(self, backend):
        await backend.add_episode(
            name='e', content='c', group_id='g', source_description='notes',
        )

        call_kwargs = backend.client.add_episode.call_args[1]
        assert call_kwargs['source_description'] == 'notes'


class TestExecuteGraphitiWriteUnverifiedClaim:
    """The flag is popped off the queue payload alongside temporal_context and
    reference_time, and forwarded to the backend."""

    @pytest.mark.asyncio
    async def test_flag_in_payload_is_forwarded(self, service):
        await service._execute_graphiti_write(
            'add_episode', _graphiti_payload(unverified_claim=True),
        )

        assert service.graphiti.add_episode.call_args[1]['unverified_claim'] is True

    @pytest.mark.asyncio
    async def test_absent_flag_forwards_false(self, service):
        await service._execute_graphiti_write('add_episode', _graphiti_payload())

        assert service.graphiti.add_episode.call_args[1]['unverified_claim'] is False

    @pytest.mark.asyncio
    async def test_flag_is_popped_not_left_on_the_payload(self, service):
        payload = _graphiti_payload(unverified_claim=True)

        await service._execute_graphiti_write('add_episode', payload)

        assert 'unverified_claim' not in payload


class TestAddEpisodeEnqueuesTheFlag:
    @pytest.mark.asyncio
    async def test_tagged_episode_carries_the_flag_on_the_queue_payload(self, service):
        await service.add_episode(
            content='task 5422 has been applied',
            project_id='dark_factory',
            unverified_claim=True,
        )

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        assert payload['unverified_claim'] is True

    @pytest.mark.asyncio
    async def test_default_is_false_on_the_payload(self, service):
        await service.add_episode(content='ordinary note', project_id='dark_factory')

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        assert payload['unverified_claim'] is False
