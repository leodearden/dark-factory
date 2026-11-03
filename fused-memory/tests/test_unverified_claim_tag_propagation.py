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


class TestMem0HalfCarriesTheTag:
    """The five false artefacts in the incident were EDGES derived from the
    episode. The Mem0 half rides its own payload channel, so the tag has to be
    copied onto each per-edge item explicitly — mirroring `planned`."""

    @pytest.mark.asyncio
    async def test_every_derived_fact_payload_carries_the_flag(self, service):
        result = MagicMock()
        result.edges = [MagicMock(fact='the fix has been applied')]
        service._refresh_entity_summaries_from_result = AsyncMock()

        await service._dual_write_callback(
            'dual_write_episode', result,
            {'project_id': 'dark_factory', 'unverified_claim': True},
        )

        batch = service.durable_queue.enqueue_batch.call_args[0][0]
        assert batch
        assert all(item['payload']['unverified_claim'] is True for item in batch)

    @pytest.mark.asyncio
    async def test_tagged_fact_is_stamped_on_the_mem0_record(self, service):
        service.classifier = MagicMock()
        classification = MagicMock()
        classification.primary = MagicMock(value='observations_and_summaries')
        classification.secondary = None
        classification.confidence = 0.9
        service.classifier.classify = AsyncMock(return_value=classification)

        from fused_memory.services.memory_service import MEM0_PRIMARY

        classification.primary = next(iter(MEM0_PRIMARY))

        await service._execute_mem0_classify_and_add(
            {'fact_text': 'the fix has been applied', 'project_id': 'dark_factory',
             'unverified_claim': True},
        )

        metadata = service.mem0.add.call_args[1]['metadata']
        assert metadata['unverified_claim'] is True

    @pytest.mark.asyncio
    async def test_untagged_fact_carries_no_key_at_all(self, service):
        service.classifier = MagicMock()
        classification = MagicMock()
        from fused_memory.services.memory_service import MEM0_PRIMARY

        classification.primary = next(iter(MEM0_PRIMARY))
        classification.secondary = None
        classification.confidence = 0.9
        service.classifier.classify = AsyncMock(return_value=classification)

        await service._execute_mem0_classify_and_add(
            {'fact_text': 'an ordinary note', 'project_id': 'dark_factory'},
        )

        metadata = service.mem0.add.call_args[1]['metadata']
        # Absent, not present-and-False: no existing record shape changes.
        assert 'unverified_claim' not in metadata


class TestAddMemoryCarriesTheFlag:
    """`MemoryService.add_memory`'s `unverified_claim` parameter (task 4715):
    the tag lands on whichever store(s) the write reaches."""

    @pytest.mark.asyncio
    async def test_graph_bound_write_carries_the_flag_on_the_queue_payload(self, service):
        await service.add_memory(
            content='task 5422 has been applied',
            category='decisions_and_rationale',
            project_id='dark_factory',
            unverified_claim=True,
        )

        call = service.durable_queue.enqueue.call_args[1]
        assert call['operation'] == 'add_memory_graphiti'
        assert call['payload']['unverified_claim'] is True

    @pytest.mark.asyncio
    async def test_graph_bound_default_is_false_on_the_payload(self, service):
        await service.add_memory(
            content='task 5422 has been applied',
            category='decisions_and_rationale',
            project_id='dark_factory',
        )

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        assert payload['unverified_claim'] is False

    @pytest.mark.asyncio
    async def test_the_flag_survives_the_queue_round_trip(self, service):
        await service.add_memory(
            content='task 5422 has been applied',
            category='decisions_and_rationale',
            project_id='dark_factory',
            unverified_claim=True,
        )
        payload = service.durable_queue.enqueue.call_args[1]['payload']

        await service._execute_graphiti_write('add_memory_graphiti', payload)

        assert service.graphiti.add_episode.call_args.kwargs['unverified_claim'] is True

    @pytest.mark.asyncio
    async def test_mem0_bound_write_stamps_the_record(self, service):
        await service.add_memory(
            content='task 5422 has been applied',
            category='procedural_knowledge',
            project_id='dark_factory',
            unverified_claim=True,
        )

        metadata = service.mem0.add.call_args.kwargs['metadata']
        assert metadata['unverified_claim'] is True

    @pytest.mark.asyncio
    async def test_untagged_mem0_record_carries_no_key(self, service):
        await service.add_memory(
            content='task 5422 has been applied',
            category='procedural_knowledge',
            project_id='dark_factory',
        )

        metadata = service.mem0.add.call_args.kwargs['metadata']
        # Absent, not present-and-False: no existing record shape changes.
        assert 'unverified_claim' not in metadata

    @pytest.mark.asyncio
    async def test_dual_write_tags_both_legs(self, service):
        await service.add_memory(
            content='task 5422 has been applied',
            category='decisions_and_rationale',
            project_id='dark_factory',
            dual_write=True,
            unverified_claim=True,
        )

        payload = service.durable_queue.enqueue.call_args[1]['payload']
        assert payload['unverified_claim'] is True
        metadata = service.mem0.add.call_args.kwargs['metadata']
        assert metadata['unverified_claim'] is True


class TestOnlyTheGateWritesTheTag:
    """`unverified_claim` is SERVER-stamped, like `category`: registering it in
    `SERVER_STAMPED_KEYS` silences the unknown-key census for it, so the write
    seam must be what stops a caller forging it, or persisting it as False on
    an untagged record (task 4715)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('supplied', [True, False])
    async def test_a_caller_supplied_value_is_discarded(self, service, supplied):
        await service.add_memory(
            content='renamed the helper for clarity',
            category='procedural_knowledge',
            project_id='dark_factory',
            metadata={'unverified_claim': supplied},
        )

        metadata = service.mem0.add.call_args.kwargs['metadata']
        assert 'unverified_claim' not in metadata, (
            f'caller-supplied {supplied!r} reached the record: {metadata!r}'
        )

    @pytest.mark.asyncio
    async def test_a_caller_supplied_false_cannot_untag_a_flagged_write(self, service):
        await service.add_memory(
            content='task 5422 has been applied',
            category='procedural_knowledge',
            project_id='dark_factory',
            metadata={'unverified_claim': False},
            unverified_claim=True,
        )

        metadata = service.mem0.add.call_args.kwargs['metadata']
        assert metadata['unverified_claim'] is True

    @pytest.mark.asyncio
    async def test_the_system_record_seam_discards_it_too(self, service):
        # add_system_record shares add_memory's metadata validator, so the
        # server-stamped registration silenced the census on this seam as well.
        service.mem0.add_system_record = AsyncMock(
            return_value={'results': [{'id': 'mem0-sys-1'}]},
        )

        await service.add_system_record(
            content='cycle summary',
            project_id='dark_factory',
            agent_id='recon-stage-task_knowledge_sync',
            category='observations_and_summaries',
            metadata={'unverified_claim': True},
        )

        metadata = service.mem0.add_system_record.call_args.kwargs['metadata']
        assert 'unverified_claim' not in metadata, f'{metadata!r}'
