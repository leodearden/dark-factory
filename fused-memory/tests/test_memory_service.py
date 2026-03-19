"""Tests for the memory service — unit tests with mocked backends."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.scope import Scope
from fused_memory.services.memory_service import MemoryService


@pytest.fixture
def service(mock_config):
    """MemoryService with mocked backends (no real DB needed)."""
    svc = MemoryService(mock_config)
    # Mock backends
    svc.graphiti = MagicMock()
    svc.graphiti.search = AsyncMock(return_value=[])
    svc.graphiti.search_nodes = AsyncMock(return_value=[])
    svc.graphiti.retrieve_episodes = AsyncMock(return_value=[])
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti.remove_episode = AsyncMock()
    svc.graphiti.remove_edge = AsyncMock()
    svc.graphiti._require_client = MagicMock()

    svc.mem0 = MagicMock()
    svc.mem0.search = AsyncMock(return_value={'results': []})
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})
    svc.mem0.get_all = AsyncMock(return_value={'results': []})
    svc.mem0.delete = AsyncMock(return_value={'message': 'deleted'})

    # Mock durable queue
    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.durable_queue.enqueue_batch = AsyncMock(return_value=[1, 2, 3])
    svc.durable_queue.get_stats = AsyncMock(return_value={'counts': {}, 'oldest_pending_age_seconds': None})
    svc.durable_queue.replay_dead = AsyncMock(return_value=0)
    svc.durable_queue.close = AsyncMock()
    return svc


class TestScope:
    def test_graphiti_group_id(self):
        scope = Scope(project_id='myproject')
        assert scope.graphiti_group_id == 'myproject'

    def test_mem0_collection_name(self):
        scope = Scope(project_id='myproject')
        assert scope.mem0_collection_name('fused') == 'fused_myproject'

    def test_mem0_user_id(self):
        scope = Scope(project_id='myproject')
        assert scope.mem0_user_id == 'myproject'


class TestAddMemory:
    @pytest.mark.asyncio
    async def test_graphiti_primary_enqueued(self, service):
        result = await service.add_memory(
            content='The auth service depends on Redis',
            category='entities_and_relations',
            project_id='test',
        )
        assert SourceStore.graphiti in result.stores_written
        assert result.category == MemoryCategory.entities_and_relations
        # Graphiti write goes through durable queue, not directly
        service.durable_queue.enqueue.assert_called_once()
        service.graphiti.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_mem0_primary_category(self, service):
        result = await service.add_memory(
            content='Always use type hints in Python code',
            category='preferences_and_norms',
            project_id='test',
        )
        assert SourceStore.mem0 in result.stores_written
        assert result.category == MemoryCategory.preferences_and_norms
        # Mem0 written directly, no queue
        service.mem0.add.assert_called_once()
        service.durable_queue.enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_dual_write(self, service):
        result = await service.add_memory(
            content='We decided to use PostgreSQL for its JSON support',
            category='decisions_and_rationale',
            project_id='test',
            dual_write=True,
        )
        assert SourceStore.graphiti in result.stores_written
        assert SourceStore.mem0 in result.stores_written
        service.durable_queue.enqueue.assert_called_once()
        service.mem0.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_classification(self, service):
        result = await service.add_memory(
            content='The payment gateway depends on the billing API',
            project_id='test',
        )
        # Should auto-classify — with heuristic-only config, entities_and_relations
        assert result.category is not None


class TestAddEpisode:
    @pytest.mark.asyncio
    async def test_episode_enqueued(self, service):
        result = await service.add_episode(
            content='User discussed auth changes',
            project_id='test',
        )
        assert result.status == 'queued'
        assert result.episode_id is not None
        service.durable_queue.enqueue.assert_called_once()
        call_kwargs = service.durable_queue.enqueue.call_args[1]
        assert call_kwargs['callback_type'] == 'dual_write_episode'
        assert call_kwargs['payload']['project_id'] == 'test'

    @pytest.mark.asyncio
    async def test_enqueue_payload_contains_uuid(self, service):
        """The enqueue payload must include 'uuid' matching the returned episode_id."""
        result = await service.add_episode(
            content='User discussed auth changes',
            project_id='test',
        )
        call_kwargs = service.durable_queue.enqueue.call_args[1]
        payload = call_kwargs['payload']
        assert 'uuid' in payload, "Payload must include 'uuid' field"
        assert payload['uuid'] == result.episode_id


class TestExecuteGraphitiWrite:
    @pytest.mark.asyncio
    async def test_uuid_passed_to_graphiti_backend(self, service):
        """_execute_graphiti_write must forward uuid from payload to graphiti.add_episode."""
        test_uuid = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
        payload = {
            'uuid': test_uuid,
            'name': 'episode_aaaaaaaa',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': '',
        }
        await service._execute_graphiti_write('add_episode', payload)
        service.graphiti.add_episode.assert_called_once()
        call_kwargs = service.graphiti.add_episode.call_args[1]
        assert call_kwargs.get('uuid') == test_uuid

    @pytest.mark.asyncio
    async def test_missing_uuid_passes_none(self, service):
        """Legacy payloads without uuid should pass uuid=None without error."""
        payload = {
            'name': 'episode_legacy',
            'content': 'legacy content',
            'source': 'text',
            'group_id': 'test',
            'source_description': '',
        }
        await service._execute_graphiti_write('add_episode', payload)
        service.graphiti.add_episode.assert_called_once()
        call_kwargs = service.graphiti.add_episode.call_args[1]
        assert call_kwargs.get('uuid') is None


class TestReplayFromStore:
    @pytest.mark.asyncio
    async def test_replay_enqueues_mem0_memories(self, service):
        service.mem0.get_all = AsyncMock(return_value={
            'results': [
                {'memory': 'fact one', 'metadata': {'category': 'temporal_facts'}},
                {'memory': 'fact two', 'metadata': {'category': 'entities_and_relations'}},
                {'memory': '', 'metadata': {}},  # empty — should be skipped
            ]
        })
        count = await service.replay_from_store(source_project_id='reify')
        assert count == 2  # empty one skipped
        service.durable_queue.enqueue_batch.assert_called_once()
        batch = service.durable_queue.enqueue_batch.call_args[0][0]
        assert len(batch) == 2
        assert batch[0]['group_id'] == 'reify'


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_list(self, service):
        results = await service.search(query='test query', project_id='test')
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_store_override(self, service):
        results = await service.search(
            query='test', project_id='test', stores=['mem0']
        )
        assert isinstance(results, list)
        # Only Mem0 should have been queried
        service.graphiti.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_category_filter(self, service):
        # Mock Mem0 returning a result with a category
        service.mem0.search = AsyncMock(return_value={
            'results': [
                {
                    'id': 'm1',
                    'memory': 'Always use black for formatting',
                    'score': 0.9,
                    'metadata': {'category': 'preferences_and_norms'},
                },
                {
                    'id': 'm2',
                    'memory': 'The build system changed last week',
                    'score': 0.8,
                    'metadata': {'category': 'temporal_facts'},
                },
            ]
        })
        results = await service.search(
            query='formatting', project_id='test',
            stores=['mem0'],
            categories=['preferences_and_norms'],
        )
        assert len(results) == 1
        assert results[0].category == MemoryCategory.preferences_and_norms

    @pytest.mark.asyncio
    async def test_search_category_filter_includes_graphiti_when_graphiti_primary_requested(
        self, service
    ):
        """When filtering by a GRAPHITI_PRIMARY category, Graphiti results
        (which have category=None) must NOT be silently dropped."""
        from tests.conftest import MockEdge, MockNode

        # Mock Graphiti returning edges (category=None in MemoryResult)
        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Auth service depends on Redis',
                uuid='edge-uuid-1',
                source_node=MockNode(name='Auth Service'),
                target_node=MockNode(name='Redis'),
            ),
        ])
        # Mock Mem0 returning a result with matching category
        service.mem0.search = AsyncMock(return_value={
            'results': [
                {
                    'id': 'm1',
                    'memory': 'Redis is the caching layer',
                    'score': 0.8,
                    'metadata': {'category': 'entities_and_relations'},
                },
            ]
        })
        results = await service.search(
            query='Redis dependencies',
            project_id='test',
            categories=['entities_and_relations'],
        )
        # Both the Graphiti edge and the Mem0 result should be present
        source_stores = {r.source_store for r in results}
        assert SourceStore.graphiti in source_stores, (
            'Graphiti results were silently dropped by category filter'
        )
        assert SourceStore.mem0 in source_stores

    @pytest.mark.asyncio
    async def test_search_category_filter_excludes_graphiti_when_only_mem0_primary_requested(
        self, service
    ):
        """When filtering by only MEM0_PRIMARY categories (e.g. preferences_and_norms),
        Graphiti results (category=None) should be correctly excluded."""
        from tests.conftest import MockEdge, MockNode

        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Auth service depends on Redis',
                uuid='edge-uuid-1',
                source_node=MockNode(name='Auth Service'),
                target_node=MockNode(name='Redis'),
            ),
        ])
        service.mem0.search = AsyncMock(return_value={
            'results': [
                {
                    'id': 'm1',
                    'memory': 'Always use black for formatting',
                    'score': 0.9,
                    'metadata': {'category': 'preferences_and_norms'},
                },
            ]
        })
        results = await service.search(
            query='formatting preferences',
            project_id='test',
            categories=['preferences_and_norms'],
        )
        source_stores = {r.source_store for r in results}
        # Only Mem0 results should remain; Graphiti results must be excluded
        assert SourceStore.graphiti not in source_stores
        assert SourceStore.mem0 in source_stores
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_graphiti_results_get_inferred_category_when_single_primary(
        self, service
    ):
        """When exactly one GRAPHITI_PRIMARY category is in the filter,
        Graphiti results should have that category assigned (not left as None)."""
        from tests.conftest import MockEdge, MockNode

        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Auth service depends on Redis',
                uuid='edge-uuid-1',
                source_node=MockNode(name='Auth Service'),
                target_node=MockNode(name='Redis'),
            ),
        ])
        service.mem0.search = AsyncMock(return_value={'results': []})

        results = await service.search(
            query='Redis',
            project_id='test',
            categories=['entities_and_relations'],
        )
        assert len(results) == 1
        assert results[0].source_store == SourceStore.graphiti
        # Should have the inferred category, not None
        assert results[0].category == MemoryCategory.entities_and_relations


class TestDeleteMemory:
    @pytest.mark.asyncio
    async def test_delete_graphiti(self, service):
        result = await service.delete_memory(
            memory_id='abc-123', store='graphiti', project_id='test'
        )
        assert result['status'] == 'deleted'
        assert result['store'] == 'graphiti'

    @pytest.mark.asyncio
    async def test_delete_mem0(self, service):
        result = await service.delete_memory(
            memory_id='xyz-456', store='mem0', project_id='test'
        )
        assert result['status'] == 'deleted'
        assert result['store'] == 'mem0'

    @pytest.mark.asyncio
    async def test_delete_memory_graphiti_calls_remove_edge(self, service):
        """delete_memory(store='graphiti') should call remove_edge (not remove_episode)
        because search returns edge UUIDs."""
        service.graphiti.remove_edge = AsyncMock()
        await service.delete_memory(
            memory_id='edge-uuid-123', store='graphiti', project_id='test'
        )
        service.graphiti.remove_edge.assert_called_once_with('edge-uuid-123')
        service.graphiti.remove_episode.assert_not_called()


class TestGraphitiBackendRemoveEdge:
    @pytest.mark.asyncio
    async def test_graphiti_backend_remove_edge(self, mock_config):
        """Unit test for GraphitiBackend.remove_edge — should call
        EntityEdge.get_by_uuid then edge.delete."""
        from unittest.mock import patch

        from fused_memory.backends.graphiti_client import GraphitiBackend

        backend = GraphitiBackend(mock_config)
        # Provide a mock client so _require_client succeeds
        mock_client = MagicMock()
        backend.client = mock_client

        mock_edge = AsyncMock()
        mock_edge.delete = AsyncMock()

        with patch(
            'fused_memory.backends.graphiti_client.EntityEdge'
        ) as MockEntityEdge:
            MockEntityEdge.get_by_uuid = AsyncMock(return_value=mock_edge)
            await backend.remove_edge('test-edge-uuid')

            MockEntityEdge.get_by_uuid.assert_called_once_with(
                mock_client.driver, 'test-edge-uuid'
            )
            mock_edge.delete.assert_called_once_with(mock_client.driver)


class TestGetEpisodes:
    @pytest.mark.asyncio
    async def test_returns_list(self, service):
        episodes = await service.get_episodes(project_id='test')
        assert isinstance(episodes, list)
