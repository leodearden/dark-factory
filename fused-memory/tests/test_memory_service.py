"""Tests for the memory service — unit tests with mocked backends."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.scope import Scope
from fused_memory.services.memory_service import MemoryService, _serialize_temporal


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
        # Mem0 now routed through durable queue
        service.durable_queue.enqueue.assert_called_once()
        call_kwargs = service.durable_queue.enqueue.call_args[1]
        assert call_kwargs['operation'] == 'mem0_add'
        assert call_kwargs['group_id'].startswith('mem0_')

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
        # Both stores go through the queue now
        assert service.durable_queue.enqueue.call_count == 2
        ops = [c[1]['operation'] for c in service.durable_queue.enqueue.call_args_list]
        assert 'add_memory_graphiti' in ops
        assert 'mem0_add' in ops

    @pytest.mark.asyncio
    async def test_auto_classification(self, service):
        result = await service.add_memory(
            content='The payment gateway depends on the billing API',
            project_id='test',
        )
        # Should auto-classify — with heuristic-only config, entities_and_relations
        assert result.category is not None


    @pytest.mark.asyncio
    async def test_mem0_enqueue_error_surfaced_in_response(self, service):
        """Mem0 enqueue errors must appear in the response message."""
        service.durable_queue.enqueue = AsyncMock(
            side_effect=ValueError('sqlite disk full')
        )

        result = await service.add_memory(
            content='Always use type hints',
            category='preferences_and_norms',
            project_id='test',
        )

        assert 'mem0_error' in result.message, (
            f'Expected mem0_error in response message, got: {result.message!r}'
        )

    @pytest.mark.asyncio
    async def test_success_false_when_only_targeted_store_fails(self, service):
        """success must be False when the only targeted store's enqueue fails.

        For a Mem0-only write (preferences_and_norms), if enqueue raises,
        _graphiti_error is None and _mem0_error is set.
        """
        service.durable_queue.enqueue = AsyncMock(
            side_effect=ValueError('sqlite disk full')
        )
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service._write_journal = mock_journal

        await service.add_memory(
            content='Always use type hints',
            category='preferences_and_norms',
            project_id='test',
        )

        mock_journal.log_write_op.assert_called_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs['success'] is False, (
            'Expected success=False when the only targeted store (Mem0) enqueue fails, '
            f'but got success={call_kwargs["success"]}'
        )


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


class TestDurableWriteDispatcher:
    @pytest.mark.asyncio
    async def test_routes_graphiti_operations(self, service):
        """add_episode and add_memory_graphiti route to _execute_graphiti_write."""
        payload = {
            'name': 'test', 'content': 'test', 'source': 'text',
            'group_id': 'test', 'source_description': '',
        }
        await service._execute_durable_write('add_episode', dict(payload))
        service.graphiti.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_mem0_add(self, service):
        """mem0_add operation routes to _execute_mem0_write."""
        payload = {
            'content': 'Always use type hints',
            'metadata': {'category': 'preferences_and_norms'},
            'project_id': 'test',
        }
        await service._execute_durable_write('mem0_add', payload)
        service.mem0.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_mem0_classify_and_add(self, service):
        """mem0_classify_and_add routes to _execute_mem0_classify_and_add."""
        payload = {
            'fact_text': 'Always format with black',
            'project_id': 'test',
        }
        # classifier will route this — may or may not call mem0.add depending on
        # heuristic classification. Just verify no error.
        await service._execute_durable_write('mem0_classify_and_add', payload)


class TestDualWriteCallback:
    @pytest.mark.asyncio
    async def test_callback_enqueues_facts_as_batch(self, service):
        """_dual_write_callback should batch-enqueue extracted facts."""
        from tests.conftest import MockAddEpisodeResult, MockEdge

        result = MockAddEpisodeResult(entity_edges=[
            MockEdge(fact='Redis uses port 6379'),
            MockEdge(fact='Auth service depends on Redis'),
        ])
        payload = {
            'project_id': 'test',
            'agent_id': 'test-agent',
            '_causation_id': 'caus-1',
        }
        await service._dual_write_callback('dual_write_episode', result, payload)

        service.durable_queue.enqueue_batch.assert_called_once()
        batch = service.durable_queue.enqueue_batch.call_args[0][0]
        assert len(batch) == 2
        assert all(item['operation'] == 'mem0_classify_and_add' for item in batch)
        assert all(item['group_id'] == 'mem0_test' for item in batch)
        assert batch[0]['payload']['fact_text'] == 'Redis uses port 6379'

    @pytest.mark.asyncio
    async def test_callback_no_edges_is_noop(self, service):
        """No entity_edges → no enqueue."""
        from tests.conftest import MockAddEpisodeResult

        result = MockAddEpisodeResult(entity_edges=[])
        await service._dual_write_callback('dual_write_episode', result, {'project_id': 'test'})
        service.durable_queue.enqueue_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_none_result_is_noop(self, service):
        """None result → no enqueue."""
        await service._dual_write_callback('dual_write_episode', None, {'project_id': 'test'})
        service.durable_queue.enqueue_batch.assert_not_called()


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

    @pytest.mark.asyncio
    async def test_search_graphiti_temporal_valid_at_only(self, service):
        """Graphiti search results with only valid_at set get temporal dict with null invalid_at."""
        from tests.conftest import MockEdge, MockNode

        dt_valid = datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)
        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Service A started',
                uuid='edge-temporal-1',
                source_node=MockNode(name='Service A'),
                valid_at=dt_valid,
                invalid_at=None,
            ),
        ])
        service.mem0.search = AsyncMock(return_value={'results': []})

        results = await service.search(
            query='Service A',
            project_id='test',
            categories=['entities_and_relations'],
        )
        assert len(results) == 1
        assert results[0].temporal is not None
        assert results[0].temporal['valid_at'] == '2024-03-01T10:00:00+00:00'
        assert results[0].temporal['invalid_at'] is None

    @pytest.mark.asyncio
    async def test_search_graphiti_only_invalid_at_returns_temporal(self, service):
        """Graphiti search results with only invalid_at set get temporal dict with null valid_at."""
        from tests.conftest import MockEdge, MockNode

        dt_invalid = datetime(2024, 9, 1, 10, 0, 0, tzinfo=UTC)
        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Service B deprecated',
                uuid='edge-temporal-2',
                source_node=MockNode(name='Service B'),
                valid_at=None,
                invalid_at=dt_invalid,
            ),
        ])
        service.mem0.search = AsyncMock(return_value={'results': []})

        results = await service.search(
            query='Service B',
            project_id='test',
            categories=['entities_and_relations'],
        )
        assert len(results) == 1
        assert results[0].temporal is not None
        assert results[0].temporal['valid_at'] is None
        assert results[0].temporal['invalid_at'] == '2024-09-01T10:00:00+00:00'


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


class TestSearchDeleteRoundtrip:
    @pytest.mark.asyncio
    async def test_search_then_delete_graphiti_roundtrip(self, service):
        """End-to-end contract test: search returns edge UUIDs that work with delete_memory."""
        from tests.conftest import MockEdge, MockNode

        edge_uuid = 'edge-roundtrip-uuid-42'
        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Payment gateway depends on billing API',
                uuid=edge_uuid,
                source_node=MockNode(name='Payment Gateway'),
                target_node=MockNode(name='Billing API'),
            ),
        ])
        service.mem0.search = AsyncMock(return_value={'results': []})

        # Step 1: Search
        results = await service.search(query='payment', project_id='test')
        assert len(results) >= 1
        graphiti_result = next(
            r for r in results if r.source_store == SourceStore.graphiti
        )
        assert graphiti_result.id == edge_uuid

        # Step 2: Delete using search result's id and source_store
        result = await service.delete_memory(
            memory_id=graphiti_result.id,
            store=graphiti_result.source_store.value,
            project_id='test',
        )
        assert result['status'] == 'deleted'
        assert result['store'] == 'graphiti'

        # Verify remove_edge was called with the correct edge UUID
        service.graphiti.remove_edge.assert_called_once_with(edge_uuid)
        # Verify remove_episode was NOT called (edge != episode)
        service.graphiti.remove_episode.assert_not_called()


class TestDeleteEpisode:
    @pytest.mark.asyncio
    async def test_delete_episode_still_uses_remove_episode(self, service):
        """Regression guard: delete_episode must continue to call remove_episode
        (not the new remove_edge), since episodes are EpisodicNodes."""
        await service.delete_episode(episode_id='ep-uuid-123', project_id='test')
        service.graphiti.remove_episode.assert_called_once_with('ep-uuid-123')
        service.graphiti.remove_edge.assert_not_called()


class TestClose:
    @pytest.mark.asyncio
    async def test_close_calls_all_sub_resource_close_methods(self, service):
        """Bug 3: close() must close durable_queue, graphiti, mem0,
        _write_journal, and _event_buffer — not just the first two.
        """
        # Wire up mock close() on all sub-resources
        service.graphiti.close = AsyncMock()
        service.mem0.close = AsyncMock()

        mock_journal = MagicMock()
        mock_journal.close = AsyncMock()
        service._write_journal = mock_journal

        mock_buffer = MagicMock()
        mock_buffer.close = AsyncMock()
        service._event_buffer = mock_buffer

        await service.close()

        service.durable_queue.close.assert_called_once()
        service.graphiti.close.assert_called_once()
        service.mem0.close.assert_called_once()
        mock_journal.close.assert_called_once()
        mock_buffer.close.assert_called_once()


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


class TestMem0BackendClose:
    @pytest.mark.asyncio
    async def test_close_awaits_client_close(self, mock_config):
        """Mem0Backend.close() must await client.close() for each cached instance."""
        from fused_memory.backends.mem0_client import Mem0Backend

        backend = Mem0Backend(mock_config)

        # Build a mock instance with a vector_store.client.close AsyncMock
        mock_client = MagicMock()
        mock_client.close = AsyncMock()

        mock_vector_store = MagicMock()
        mock_vector_store.client = mock_client

        mock_instance = MagicMock()
        mock_instance.vector_store = mock_vector_store

        backend._instances = {'test_project': mock_instance}

        await backend.close()

        # The close coroutine must have been awaited, not just created
        mock_client.close.assert_awaited_once()
        # All instances must be cleared
        assert backend._instances == {}


class TestGetEpisodes:
    @pytest.mark.asyncio
    async def test_returns_list(self, service):
        episodes = await service.get_episodes(project_id='test')
        assert isinstance(episodes, list)

    @pytest.mark.asyncio
    async def test_none_created_at_returns_none(self, service):
        """When an episode has created_at=None, the dict value should be None (not 'None')."""
        mock_ep = MagicMock()
        mock_ep.uuid = 'ep-uuid-1'
        mock_ep.name = 'test episode'
        mock_ep.content = 'some content'
        mock_ep.created_at = None
        mock_ep.source = None
        mock_ep.group_id = 'test'
        service.graphiti.retrieve_episodes = AsyncMock(return_value=[mock_ep])

        episodes = await service.get_episodes(project_id='test')

        assert len(episodes) == 1
        assert episodes[0]['created_at'] is None, (
            f"Expected None but got {episodes[0]['created_at']!r} — str(None) bug"
        )


class TestGetEntity:
    # ------------------------------------------------------------------
    # baseline success regression test
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_success_returns_nodes_and_edges(self, service):
        """Baseline regression: get_entity returns correct node and edge data."""
        from tests.conftest import MockEdge, MockNode

        mock_node = MockNode(name='Auth Service', uuid='node-uuid-1')
        mock_edge = MockEdge(fact='Auth service depends on Redis', uuid='edge-uuid-1')

        service.graphiti.search_nodes = AsyncMock(return_value=[mock_node])
        service.graphiti.search = AsyncMock(return_value=[mock_edge])

        result = await service.get_entity('Auth Service', project_id='test')

        assert isinstance(result, dict)
        assert 'nodes' in result
        assert 'edges' in result
        assert len(result['nodes']) == 1
        assert result['nodes'][0]['name'] == 'Auth Service'
        assert result['nodes'][0]['uuid'] == 'node-uuid-1'
        assert len(result['edges']) == 1
        assert result['edges'][0]['fact'] == 'Auth service depends on Redis'
        assert result['edges'][0]['uuid'] == 'edge-uuid-1'

    # ------------------------------------------------------------------
    # concurrent execution — both coroutines run in parallel
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, service):
        """Both Graphiti calls run concurrently — verified by asyncio.Event gates.

        Each side_effect sets its own started-event then waits on the other's.
        With sequential execution the first waiter times out (TimeoutError).
        With concurrent execution both events fire and the call completes.
        """
        import asyncio as _asyncio

        search_nodes_started = _asyncio.Event()
        search_started = _asyncio.Event()

        async def search_nodes_gate(*args, **kwargs):
            search_nodes_started.set()
            # Wait for search() to start — only possible if both run concurrently
            await _asyncio.wait_for(search_started.wait(), timeout=1.0)
            return []

        async def search_gate(*args, **kwargs):
            search_started.set()
            # Wait for search_nodes() to start — only possible if both run concurrently
            await _asyncio.wait_for(search_nodes_started.wait(), timeout=1.0)
            return []

        service.graphiti.search_nodes = AsyncMock(side_effect=search_nodes_gate)
        service.graphiti.search = AsyncMock(side_effect=search_gate)

        # Concurrent execution allows both events to fire; sequential causes TimeoutError
        result = await service.get_entity('entity', project_id='test')
        assert result['nodes'] == []
        assert result['edges'] == []

    # ------------------------------------------------------------------
    # search_nodes failure — error propagates (concurrent gather)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_search_nodes_failure_raises_and_search_settles(self, service):
        """When search_nodes raises, exception propagates.

        With concurrent gather, search() is still called (both coroutines
        settle before the exception is inspected).
        """
        service.graphiti.search_nodes = AsyncMock(
            side_effect=RuntimeError('search_nodes failed')
        )
        service.graphiti.search = AsyncMock(return_value=[])

        with pytest.raises(RuntimeError, match='search_nodes failed'):
            await service.get_entity('entity', project_id='test')

        # search() still settles in the concurrent gather even though search_nodes failed
        service.graphiti.search.assert_called_once()

    # ------------------------------------------------------------------
    # search failure — search_nodes succeeds, search raises
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_search_failure_raises_and_search_nodes_settles(self, service):
        """When search() raises, exception propagates AND search_nodes() was called.

        With concurrent gather, both coroutines settle before any exception is
        inspected — search_nodes() completes even though search() raises.
        """
        service.graphiti.search_nodes = AsyncMock(return_value=[])
        service.graphiti.search = AsyncMock(
            side_effect=RuntimeError('search failed')
        )

        with pytest.raises(RuntimeError, match='search failed'):
            await service.get_entity('entity', project_id='test')

        # search_nodes() settled in the concurrent gather even though search() failed
        service.graphiti.search_nodes.assert_called_once()

    # ------------------------------------------------------------------
    # both calls fail — first exception is re-raised
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_both_failures_raises_first_exception(self, service):
        """When both calls fail, the first exception from gather results is re-raised.

        gather() returns results in positional order: search_nodes is index 0,
        search is index 1. The first exception found in that order is re-raised.
        """
        service.graphiti.search_nodes = AsyncMock(
            side_effect=RuntimeError('search_nodes failed')
        )
        service.graphiti.search = AsyncMock(
            side_effect=RuntimeError('search failed')
        )

        with pytest.raises(RuntimeError, match='search_nodes failed'):
            await service.get_entity('entity', project_id='test')

    # ------------------------------------------------------------------
    # both calls fail — both warnings emitted before re-raise
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_both_failures_emits_two_warnings(self, service):
        """When both calls fail, logger.warning is called twice — once per exception.

        The exception filter iterates all gather results and logs each Exception
        before raising the first one. Both failures must be visible in logs.
        """
        from unittest.mock import patch

        service.graphiti.search_nodes = AsyncMock(
            side_effect=RuntimeError('search_nodes failed')
        )
        service.graphiti.search = AsyncMock(
            side_effect=RuntimeError('search failed')
        )

        with patch('fused_memory.services.memory_service.logger') as mock_logger:
            with pytest.raises(RuntimeError, match='search_nodes failed'):
                await service.get_entity('entity', project_id='test')

        assert mock_logger.warning.call_count == 2

    # ------------------------------------------------------------------
    # temporal serialization in edge_data
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_edges_include_temporal(self, service):
        """Edge dicts in get_entity result include a 'temporal' key with ISO 8601 strings."""
        from tests.conftest import MockEdge

        dt_valid = datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)
        dt_invalid = datetime(2024, 9, 1, 10, 0, 0, tzinfo=UTC)
        mock_edge = MockEdge(
            fact='Service A calls Service B',
            uuid='edge-temporal-1',
            valid_at=dt_valid,
            invalid_at=dt_invalid,
        )
        service.graphiti.search_nodes = AsyncMock(return_value=[])
        service.graphiti.search = AsyncMock(return_value=[mock_edge])

        result = await service.get_entity('Service A', project_id='test')

        assert len(result['edges']) == 1
        edge = result['edges'][0]
        assert 'temporal' in edge
        temporal = edge['temporal']
        assert temporal is not None
        assert temporal['valid_at'] == '2024-03-01T10:00:00+00:00'
        assert temporal['invalid_at'] == '2024-09-01T10:00:00+00:00'

    @pytest.mark.asyncio
    async def test_edges_only_valid_at_has_null_invalid_at(self, service):
        """When edge has only valid_at, temporal['invalid_at'] is None (not 'None')."""
        from tests.conftest import MockEdge

        dt_valid = datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)
        mock_edge = MockEdge(
            fact='Service A calls Service B',
            uuid='edge-temporal-2',
            valid_at=dt_valid,
            invalid_at=None,
        )
        service.graphiti.search_nodes = AsyncMock(return_value=[])
        service.graphiti.search = AsyncMock(return_value=[mock_edge])

        result = await service.get_entity('Service A', project_id='test')

        assert len(result['edges']) == 1
        edge = result['edges'][0]
        assert edge['temporal'] is not None
        assert edge['temporal']['valid_at'] == '2024-03-01T10:00:00+00:00'
        assert edge['temporal']['invalid_at'] is None


class TestSerializeTemporal:
    """Unit tests for the _serialize_temporal module-level helper."""

    def test_both_none_returns_none(self):
        """When both valid_at and invalid_at are None, returns None."""
        result = _serialize_temporal(None, None)
        assert result is None

    def test_both_set_returns_isoformat_dict(self):
        """When both are set, returns dict with ISO 8601 strings."""
        dt_valid = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        dt_invalid = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        result = _serialize_temporal(dt_valid, dt_invalid)
        assert result is not None
        assert result['valid_at'] == '2024-01-15T12:00:00+00:00'
        assert result['invalid_at'] == '2024-06-15T12:00:00+00:00'

    def test_only_valid_at(self):
        """When only valid_at is set, returns dict with valid_at string and invalid_at=None."""
        dt = datetime(2024, 3, 1, 0, 0, 0, tzinfo=UTC)
        result = _serialize_temporal(dt, None)
        assert result is not None
        assert result['valid_at'] == '2024-03-01T00:00:00+00:00'
        assert result['invalid_at'] is None

    def test_only_invalid_at(self):
        """When only invalid_at is set, returns dict with valid_at=None and invalid_at string.

        This covers the old truthiness bug: the original code used `if not valid_at and
        not invalid_at` as the outer guard — which would return None for falsy-but-not-None
        values. The fix uses identity checks (`is None`) for correctness.
        """
        dt = datetime(2024, 9, 1, 0, 0, 0, tzinfo=UTC)
        result = _serialize_temporal(None, dt)
        assert result is not None
        assert result['valid_at'] is None
        assert result['invalid_at'] == '2024-09-01T00:00:00+00:00'

    def test_string_input_uses_str_fallback(self):
        """Raw ISO 8601 strings pass through str() instead of raising AttributeError.

        Graphiti may return temporal values as pre-serialized strings rather than
        datetime objects. _serialize_temporal must not call .isoformat() on a string.
        """
        iso_valid = '2024-01-15T12:00:00+00:00'
        iso_invalid = '2024-06-15T12:00:00+00:00'
        result = _serialize_temporal(iso_valid, iso_invalid)
        assert result is not None
        assert result['valid_at'] == iso_valid
        assert result['invalid_at'] == iso_invalid

    def test_integer_timestamp_uses_str_fallback(self):
        """Integer timestamps pass through str() instead of raising AttributeError.

        If a caller passes an integer Unix timestamp, _serialize_temporal must not
        call .isoformat() on it. str(int) is returned as the fallback.
        """
        ts = 1705320000
        result = _serialize_temporal(ts, None)
        assert result is not None
        assert result['valid_at'] == str(ts)
        assert result['invalid_at'] is None
