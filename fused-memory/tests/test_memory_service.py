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
    async def test_empty_string_uuid_preserved_in_search_graphiti(self, service):
        """An edge with an empty-string uuid must not be replaced by its index.

        MockEdge defaults uuid to '' (empty string).  The buggy code uses
        `d['uuid'] or str(i)` which coerces '' to '0'.  The correct code uses
        `d['uuid'] if d['uuid'] is not None else str(i)` which preserves ''.
        """
        from tests.conftest import MockEdge

        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(fact='some fact'),  # uuid defaults to ''
        ])
        service.mem0.search = AsyncMock(return_value={'results': []})

        results = await service.search(query='test', project_id='test')
        assert len(results) == 1
        assert results[0].id == '', (
            f"Expected empty-string uuid to be preserved, got {results[0].id!r}"
        )

    @pytest.mark.asyncio
    async def test_none_uuid_edge_falls_back_to_index_in_search(self, service):
        """An edge with no uuid attribute at all uses its index position as the result id.

        When _build_edge_dict returns uuid=None (via getattr default), _search_graphiti
        should use str(i) as the result id.  For the first (index 0) edge this is '0'.
        This covers the end-to-end uuid=None fallback path, complementing the unit test
        test_edge_with_no_uuid_attr_returns_none which only tests the helper in isolation.
        """

        from tests.conftest import EdgeWithoutUuid

        service.graphiti.search = AsyncMock(return_value=[EdgeWithoutUuid()])
        service.mem0.search = AsyncMock(return_value={'results': []})

        results = await service.search(query='test', project_id='test')
        assert len(results) == 1
        assert results[0].id == '0', (
            f"Expected str(0) fallback id when uuid is None, got {results[0].id!r}"
        )


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


class TestGetEntityTemporalMetadata:
    """Tests that get_entity() returns temporal metadata on each edge."""

    @pytest.mark.asyncio
    async def test_get_entity_edges_include_temporal_metadata(self, service):
        """Each edge dict must include a 'temporal' key with valid_at/invalid_at."""
        from tests.conftest import MockEdge

        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Auth service uses Redis',
                uuid='edge-1',
                valid_at='2026-01-01T00:00:00+00:00',
                invalid_at=None,
            ),
            MockEdge(
                fact='Auth service was standalone',
                uuid='edge-2',
                valid_at='2025-01-01T00:00:00+00:00',
                invalid_at='2026-01-01T00:00:00+00:00',
            ),
        ])
        service.graphiti.search_nodes = AsyncMock(return_value=[])

        result = await service.get_entity(name='Auth', project_id='test')
        edges = result['edges']

        assert len(edges) == 2

        # First edge: valid_at set, invalid_at None
        e0 = edges[0]
        assert 'temporal' in e0, "Edge must have 'temporal' key"
        assert e0['temporal'] is not None
        assert e0['temporal']['valid_at'] == '2026-01-01T00:00:00+00:00'
        assert e0['temporal']['invalid_at'] is None

        # Second edge: both set
        e1 = edges[1]
        assert 'temporal' in e1, "Edge must have 'temporal' key"
        assert e1['temporal'] is not None
        assert e1['temporal']['valid_at'] == '2025-01-01T00:00:00+00:00'
        assert e1['temporal']['invalid_at'] == '2026-01-01T00:00:00+00:00'

    @pytest.mark.asyncio
    async def test_get_entity_edge_without_temporal_has_none(self, service):
        """An edge with neither valid_at nor invalid_at should return temporal=None."""
        from tests.conftest import MockEdge

        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(fact='plain fact', uuid='edge-3'),
        ])
        service.graphiti.search_nodes = AsyncMock(return_value=[])

        result = await service.get_entity(name='Something', project_id='test')
        edges = result['edges']

        assert len(edges) == 1
        assert 'temporal' in edges[0]
        assert edges[0]['temporal'] is None


class TestGetEntityNameEntitiesProvenance:
    """Tests that get_entity() returns name, entities, and provenance on each edge."""

    @pytest.mark.asyncio
    async def test_get_entity_edges_include_name_entities_provenance(self, service):
        """Each edge dict must include 'name', 'entities', and 'provenance'."""
        from tests.conftest import MockEdge, MockNode

        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Auth service uses Redis',
                uuid='edge-1',
                name='USES',
                source_node=MockNode(name='Auth Service'),
                target_node=MockNode(name='Redis'),
                episodes=['ep-uuid-1', 'ep-uuid-2'],
            ),
        ])
        service.graphiti.search_nodes = AsyncMock(return_value=[])

        result = await service.get_entity(name='Auth', project_id='test')
        edges = result['edges']

        assert len(edges) == 1
        e = edges[0]

        assert 'name' in e, "Edge must have 'name' key"
        assert e['name'] == 'USES'

        assert 'entities' in e, "Edge must have 'entities' key"
        assert 'Auth Service' in e['entities']
        assert 'Redis' in e['entities']

        assert 'provenance' in e, "Edge must have 'provenance' key"
        assert e['provenance'] == ['ep-uuid-1', 'ep-uuid-2']

    @pytest.mark.asyncio
    async def test_get_entity_edge_without_nodes_has_empty_entities(self, service):
        """An edge with no source/target nodes returns entities=[]."""
        from tests.conftest import MockEdge

        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(fact='bare fact', uuid='edge-2', name=None),
        ])
        service.graphiti.search_nodes = AsyncMock(return_value=[])

        result = await service.get_entity(name='Bare', project_id='test')
        edges = result['edges']

        assert len(edges) == 1
        e = edges[0]
        assert e['name'] is None
        assert e['entities'] == []
        assert e['provenance'] == []


class TestBuildEdgeDict:
    """Tests for the _build_edge_dict helper method."""

    def test_full_edge_returns_all_six_keys(self, service):
        """A fully populated edge returns all six expected keys with correct values."""
        from tests.conftest import MockEdge, MockNode

        edge = MockEdge(
            fact='Auth service depends on Redis',
            uuid='edge-uuid-1',
            name='DEPENDS_ON',
            source_node=MockNode(name='Auth Service'),
            target_node=MockNode(name='Redis'),
            episodes=['ep-1', 'ep-2'],
            valid_at='2026-01-01T00:00:00+00:00',
            invalid_at='2026-06-01T00:00:00+00:00',
        )
        d = service._build_edge_dict(edge)

        assert set(d.keys()) == {'uuid', 'fact', 'name', 'temporal', 'entities', 'provenance'}
        assert d['uuid'] == 'edge-uuid-1'
        assert d['fact'] == 'Auth service depends on Redis'
        assert d['name'] == 'DEPENDS_ON'
        assert d['temporal'] == {
            'valid_at': '2026-01-01T00:00:00+00:00',
            'invalid_at': '2026-06-01T00:00:00+00:00',
        }
        assert d['entities'] == ['Auth Service', 'Redis']
        assert d['provenance'] == ['ep-1', 'ep-2']

    def test_minimal_edge_returns_safe_defaults(self, service):
        """An edge with no temporal/nodes/episodes returns temporal=None, entities=[], provenance=[]."""
        from tests.conftest import MockEdge

        edge = MockEdge(fact='bare fact')  # uuid defaults to ''
        d = service._build_edge_dict(edge)

        assert d['temporal'] is None
        assert d['entities'] == []
        assert d['provenance'] == []
        assert d['uuid'] == ''
        assert d['fact'] == 'bare fact'
        assert d['name'] is None

    def test_edge_with_only_valid_at_returns_temporal_dict(self, service):
        """Edge with only valid_at set returns temporal dict with valid_at set and invalid_at=None."""
        from tests.conftest import MockEdge

        edge = MockEdge(fact='fact', uuid='u1', valid_at='2026-01-01T00:00:00+00:00')
        d = service._build_edge_dict(edge)

        assert d['temporal'] is not None
        assert d['temporal']['valid_at'] == '2026-01-01T00:00:00+00:00'
        assert d['temporal']['invalid_at'] is None

    def test_edge_with_only_invalid_at_returns_temporal_dict(self, service):
        """Edge with only invalid_at set returns temporal dict with invalid_at set and valid_at=None."""
        from tests.conftest import MockEdge

        edge = MockEdge(fact='fact', uuid='u-inv', invalid_at='2026-06-01T00:00:00+00:00')
        d = service._build_edge_dict(edge)

        assert d['temporal'] is not None
        assert d['temporal']['valid_at'] is None
        assert d['temporal']['invalid_at'] == '2026-06-01T00:00:00+00:00'

    def test_edge_with_only_source_node_returns_single_entity(self, service):
        """Edge with only source_node and no target_node returns single-element entities list."""
        from tests.conftest import MockEdge, MockNode

        edge = MockEdge(
            fact='fact',
            uuid='u2',
            source_node=MockNode(name='SourceEntity'),
        )
        d = service._build_edge_dict(edge)

        assert d['entities'] == ['SourceEntity']

    def test_edge_with_no_uuid_attr_returns_none(self, service):
        """Edge object without a uuid attribute returns uuid=None."""
        from tests.conftest import EdgeWithoutUuid

        d = service._build_edge_dict(EdgeWithoutUuid())
        assert d['uuid'] is None

    def test_edge_without_fact_attr_falls_back_to_str(self, service):
        """Edge object without fact attr uses str(edge) as the fact value."""

        class EdgeWithoutFact:
            uuid = 'u3'
            name = None
            source_node = None
            target_node = None
            episodes = []
            valid_at = None
            invalid_at = None

            def __str__(self):
                return 'EdgeWithoutFact-str-representation'

        edge = EdgeWithoutFact()
        d = service._build_edge_dict(edge)
        assert d['fact'] == 'EdgeWithoutFact-str-representation'

    def test_non_string_episode_objects_converted_to_str_in_provenance(self, service):
        """Non-string episode objects in an edge's episodes list are str()-converted."""

        class CustomEpisode:
            def __str__(self):
                return 'custom-ep-repr'

        class EdgeWithCustomEpisodes:
            uuid = 'u-ep'
            fact = 'episode fact'
            name = None
            source_node = None
            target_node = None
            episodes = [CustomEpisode(), CustomEpisode()]
            valid_at = None
            invalid_at = None

        d = service._build_edge_dict(EdgeWithCustomEpisodes())
        assert d['provenance'] == ['custom-ep-repr', 'custom-ep-repr']

    def test_empty_string_valid_at_builds_temporal_dict(self, service):
        """An edge with valid_at='' (empty string, not None) must still produce a temporal dict.

        The buggy code uses `if valid_at or invalid_at:` which treats '' as falsy,
        causing the temporal dict to be skipped entirely.  The correct code uses
        `is not None` checks so that empty-string values are preserved.
        """
        from tests.conftest import MockEdge

        edge = MockEdge(fact='fact', uuid='u-va', valid_at='')
        d = service._build_edge_dict(edge)

        assert d['temporal'] is not None, (
            "temporal dict should be built even when valid_at is '' (not None)"
        )
        assert d['temporal']['valid_at'] == '', (
            f"valid_at '' should be preserved, got {d['temporal']['valid_at']!r}"
        )
        assert d['temporal']['invalid_at'] is None, (
            f"invalid_at should be None when not set, got {d['temporal']['invalid_at']!r}"
        )

    def test_empty_string_invalid_at_builds_temporal_dict(self, service):
        """An edge with invalid_at='' (empty string, not None) must still produce a temporal dict.

        The buggy code uses `if valid_at or invalid_at:` which treats '' as falsy,
        causing the temporal dict to be skipped entirely.  The correct code uses
        `is not None` checks so that empty-string values are preserved.
        """
        from tests.conftest import MockEdge

        edge = MockEdge(fact='fact', uuid='u-ia', invalid_at='')
        d = service._build_edge_dict(edge)

        assert d['temporal'] is not None, (
            "temporal dict should be built even when invalid_at is '' (not None)"
        )
        assert d['temporal']['invalid_at'] == '', (
            f"invalid_at '' should be preserved, got {d['temporal']['invalid_at']!r}"
        )
        assert d['temporal']['valid_at'] is None, (
            f"valid_at should be None when not set, got {d['temporal']['valid_at']!r}"
        )

    def test_none_episodes_attr_produces_empty_provenance(self, service):
        """Edge with episodes=None (attribute present but None) returns provenance=[].

        Documents the is-not-None semantic: None episodes must not raise and must
        produce an empty list, same as a missing episodes attribute.
        """

        class EdgeWithNoneEpisodes:
            uuid = 'u-ep-none'
            fact = 'fact with none episodes'
            name = None
            source_node = None
            target_node = None
            episodes = None  # attribute present, value is None
            valid_at = None
            invalid_at = None

        d = service._build_edge_dict(EdgeWithNoneEpisodes())
        assert d['provenance'] == [], (
            f"provenance should be [] when episodes is None, got {d['provenance']!r}"
        )


class TestGetEntityValidOnly:
    """Tests that get_entity() valid_only parameter filters invalidated edges."""

    @pytest.mark.asyncio
    async def test_valid_only_filters_invalidated_edges(self, service):
        """valid_only=True should exclude edges where invalid_at is not None."""
        from tests.conftest import MockEdge

        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(fact='current fact', uuid='e1', invalid_at=None),           # valid
            MockEdge(fact='old fact', uuid='e2', invalid_at='2026-01-01T00:00:00+00:00'),  # invalidated
            MockEdge(fact='no dates fact', uuid='e3'),                            # no dates → valid
        ])
        service.graphiti.search_nodes = AsyncMock(return_value=[])

        result = await service.get_entity(name='Thing', project_id='test', valid_only=True)
        edges = result['edges']

        uuids = [e['uuid'] for e in edges]
        assert 'e1' in uuids, "valid edge (invalid_at=None) must be included"
        assert 'e3' in uuids, "edge with no dates must be included"
        assert 'e2' not in uuids, "invalidated edge must be excluded"

    @pytest.mark.asyncio
    async def test_valid_only_false_returns_all_edges(self, service):
        """valid_only=False (default) must return all edges including invalidated ones."""
        from tests.conftest import MockEdge

        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(fact='current fact', uuid='e1', invalid_at=None),
            MockEdge(fact='old fact', uuid='e2', invalid_at='2026-01-01T00:00:00+00:00'),
        ])
        service.graphiti.search_nodes = AsyncMock(return_value=[])

        result = await service.get_entity(name='Thing', project_id='test', valid_only=False)
        edges = result['edges']
        uuids = [e['uuid'] for e in edges]
        assert 'e1' in uuids
        assert 'e2' in uuids

    @pytest.mark.asyncio
    async def test_valid_only_defaults_to_false(self, service):
        """Calling get_entity() without valid_only should return all edges."""
        from tests.conftest import MockEdge

        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(fact='current fact', uuid='e1'),
            MockEdge(fact='old fact', uuid='e2', invalid_at='2026-01-01T00:00:00+00:00'),
        ])
        service.graphiti.search_nodes = AsyncMock(return_value=[])

        result = await service.get_entity(name='Thing', project_id='test')
        edges = result['edges']
        uuids = [e['uuid'] for e in edges]
        # Default (valid_only=False) returns all
        assert 'e1' in uuids
        assert 'e2' in uuids
