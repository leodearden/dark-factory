"""Tests for the memory service — unit tests with mocked backends."""

import asyncio
import types
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestDualWriteCallbackTemporalContext:
    """step-7: _dual_write_callback propagates temporal_context to Mem0 batch items."""

    @pytest.mark.asyncio
    async def test_temporal_context_planning_propagated(self, service):
        """When payload has temporal_context='planning', each batch item includes it."""
        from tests.conftest import MockAddEpisodeResult, MockEdge

        result = MockAddEpisodeResult(entity_edges=[
            MockEdge(fact='CostStore extends AgentResult'),
        ])
        payload = {
            'project_id': 'test',
            'temporal_context': 'planning',
        }
        await service._dual_write_callback('dual_write_episode', result, payload)

        service.durable_queue.enqueue_batch.assert_called_once()
        batch = service.durable_queue.enqueue_batch.call_args[0][0]
        assert len(batch) == 1
        assert batch[0]['payload']['temporal_context'] == 'planning'

    @pytest.mark.asyncio
    async def test_no_temporal_context_absent_from_batch(self, service):
        """When temporal_context is absent in payload, it should NOT appear in batch items."""
        from tests.conftest import MockAddEpisodeResult, MockEdge

        result = MockAddEpisodeResult(entity_edges=[
            MockEdge(fact='Auth depends on Redis'),
        ])
        payload = {
            'project_id': 'test',
            # no temporal_context
        }
        await service._dual_write_callback('dual_write_episode', result, payload)

        service.durable_queue.enqueue_batch.assert_called_once()
        batch = service.durable_queue.enqueue_batch.call_args[0][0]
        assert len(batch) == 1
        # temporal_context is absent or None — not 'planning'
        assert batch[0]['payload'].get('temporal_context') is None


class TestExecuteMem0ClassifyAndAddPlanningMetadata:
    """step-9: _execute_mem0_classify_and_add adds planned=True to metadata when planning."""

    @pytest.mark.asyncio
    async def test_planning_temporal_context_adds_planned_metadata(self, service):
        """When payload has temporal_context='planning', metadata must include planned=True.

        Forces Mem0 routing via classifier mock to remove the vacuous-assertion risk.
        """
        from fused_memory.models.enums import MemoryCategory
        mock_classification = MagicMock()
        mock_classification.primary = MemoryCategory.preferences_and_norms
        mock_classification.secondary = None
        mock_classification.confidence = 0.95
        service.classifier.classify = AsyncMock(return_value=mock_classification)

        payload = {
            'fact_text': 'Always use type hints',
            'project_id': 'test',
            'temporal_context': 'planning',
        }
        await service._execute_mem0_classify_and_add(payload)

        # Unconditional assertion — classifier is forced to Mem0 so this must be called
        service.mem0.add.assert_called_once()
        call_kwargs = service.mem0.add.call_args[1]
        metadata = call_kwargs.get('metadata', {})
        assert metadata.get('planned') is True, (
            f'Expected planned=True in metadata, got: {metadata}'
        )

    @pytest.mark.asyncio
    async def test_no_temporal_context_no_planned_metadata(self, service):
        """Without temporal_context, planned key must not be in metadata.

        Forces Mem0 routing via classifier mock to remove the vacuous-assertion risk.
        """
        from fused_memory.models.enums import MemoryCategory
        mock_classification = MagicMock()
        mock_classification.primary = MemoryCategory.preferences_and_norms
        mock_classification.secondary = None
        mock_classification.confidence = 0.95
        service.classifier.classify = AsyncMock(return_value=mock_classification)

        payload = {
            'fact_text': 'Always use type hints',
            'project_id': 'test',
            # no temporal_context
        }
        await service._execute_mem0_classify_and_add(payload)

        # Unconditional assertion — classifier is forced to Mem0 so this must be called
        service.mem0.add.assert_called_once()
        call_kwargs = service.mem0.add.call_args[1]
        metadata = call_kwargs.get('metadata', {})
        assert 'planned' not in metadata, (
            f'Unexpected planned key in metadata: {metadata}'
        )

    @pytest.mark.asyncio
    async def test_planning_routed_to_mem0_is_tagged(self, service):
        """Specifically test a fact that should route to Mem0 gets planned=True."""
        # Preferences/norms always route to Mem0; patch classifier to force it
        from unittest.mock import AsyncMock, MagicMock

        from fused_memory.models.enums import MemoryCategory
        mock_classification = MagicMock()
        mock_classification.primary = MemoryCategory.preferences_and_norms
        mock_classification.secondary = None
        mock_classification.confidence = 0.95
        service.classifier.classify = AsyncMock(return_value=mock_classification)

        payload = {
            'fact_text': 'Always use type hints in Python',
            'project_id': 'test',
            'temporal_context': 'planning',
        }
        await service._execute_mem0_classify_and_add(payload)

        service.mem0.add.assert_called_once()
        call_kwargs = service.mem0.add.call_args[1]
        metadata = call_kwargs.get('metadata', {})
        assert metadata.get('planned') is True


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
        assert result['nodes'][0]['labels'] == []

    # ------------------------------------------------------------------
    # getattr fallback — missing attributes return None / [] defaults
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_getattr_fallback_missing_attributes(self, service):
        """Bare objects without optional attrs return None/[] defaults."""
        bare_node = types.SimpleNamespace(name='BareNode')
        bare_edge = types.SimpleNamespace()  # no attributes at all

        service.graphiti.search_nodes = AsyncMock(return_value=[bare_node])
        service.graphiti.search = AsyncMock(return_value=[bare_edge])

        result = await service.get_entity('BareNode', project_id='test')

        node = result['nodes'][0]
        assert node['uuid'] is None
        assert node['summary'] is None
        assert node['labels'] == []

        edge = result['edges'][0]
        assert edge['uuid'] is None
        assert edge['fact'] == str(bare_edge)

    @pytest.mark.asyncio
    async def test_getattr_labels_none_returns_empty_list(self, service):
        """labels=None (attribute present but explicitly None) must return []."""
        bare_node = types.SimpleNamespace(name='BareNode')
        bare_node.labels = None  # attribute IS present, value is None

        service.graphiti.search_nodes = AsyncMock(return_value=[bare_node])
        service.graphiti.search = AsyncMock(return_value=[])

        result = await service.get_entity('BareNode', project_id='test')

        assert result['nodes'][0]['labels'] == []

    @pytest.mark.asyncio
    async def test_getattr_labels_nonempty_passthrough(self, service):
        """Non-empty labels list passes through unchanged (or [] short-circuit does not fire)."""
        node = types.SimpleNamespace(name='AuthService')
        node.labels = ['Service', 'Auth']

        service.graphiti.search_nodes = AsyncMock(return_value=[node])
        service.graphiti.search = AsyncMock(return_value=[])

        result = await service.get_entity('AuthService', project_id='test')

        assert result['nodes'][0]['labels'] == ['Service', 'Auth']

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
        search_nodes_started = asyncio.Event()
        search_started = asyncio.Event()

        async def search_nodes_gate(*args, **kwargs):
            search_nodes_started.set()
            # Wait for search() to start — only possible if both run concurrently
            await asyncio.wait_for(search_started.wait(), timeout=1.0)
            return []

        async def search_gate(*args, **kwargs):
            search_started.set()
            # Wait for search_nodes() to start — only possible if both run concurrently
            await asyncio.wait_for(search_nodes_started.wait(), timeout=1.0)
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

        With concurrent gather, search() settles (both coroutines run to
        completion before the exception is inspected).
        """
        service.graphiti.search_nodes = AsyncMock(
            side_effect=RuntimeError('search_nodes failed')
        )
        service.graphiti.search = AsyncMock(return_value=[])

        with pytest.raises(RuntimeError, match='search_nodes failed'):
            await service.get_entity('entity', project_id='test')

        # search() settles in the concurrent gather even though search_nodes failed
        service.graphiti.search.assert_called_once()

    # ------------------------------------------------------------------
    # search failure — search_nodes succeeds, search raises
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_search_failure_raises_and_search_nodes_settles(self, service):
        """When search() raises, exception propagates AND search_nodes() settles.

        With concurrent gather, both coroutines run to completion before any exception
        is inspected — search_nodes() settles even though search() raises.
        """
        service.graphiti.search_nodes = AsyncMock(return_value=[])
        service.graphiti.search = AsyncMock(
            side_effect=RuntimeError('search failed')
        )

        with pytest.raises(RuntimeError, match='search failed'):
            await service.get_entity('entity', project_id='test')

        # search_nodes() settles in the concurrent gather even though search() failed
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

        # gather(return_exceptions=True) settles both coroutines — search() was called
        service.graphiti.search.assert_called_once()

    # ------------------------------------------------------------------
    # both calls fail — both warnings emitted before re-raise
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_both_failures_emits_two_warnings(self, service):
        """When both calls fail, logger.warning is called twice — once per exception.

        The exception filter iterates all gather results and logs each Exception
        before raising the first one. Both failures must be visible in logs.
        """
        service.graphiti.search_nodes = AsyncMock(
            side_effect=RuntimeError('search_nodes failed')
        )
        service.graphiti.search = AsyncMock(
            side_effect=RuntimeError('search failed')
        )

        with patch('fused_memory.services.memory_service.logger') as mock_logger, \
             pytest.raises(RuntimeError, match='search_nodes failed'):
            await service.get_entity('entity', project_id='test')

        assert mock_logger.warning.call_count == 2

    # ------------------------------------------------------------------
    # CancelledError propagation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self, service):
        """asyncio.CancelledError raised by search_nodes must propagate unchanged.

        asyncio.gather(return_exceptions=True) captures ALL BaseException subclasses
        as values in the results list — including CancelledError, which is a BaseException
        but NOT an Exception.  The detection guard must use isinstance(r, BaseException)
        so CancelledError is recognised as a captured error value and re-raised.

        Without the fix (guard uses Exception instead of BaseException) the CancelledError
        value slips past the guard, code falls through to cast(list, results[0]), and
        'for n in nodes:' raises TypeError — silently converting cancellation into a
        confusing TypeError.
        """
        service.graphiti.search_nodes = AsyncMock(
            side_effect=asyncio.CancelledError()
        )
        service.graphiti.search = AsyncMock(return_value=[])

        with patch('fused_memory.services.memory_service.logger') as mock_logger, \
             pytest.raises(asyncio.CancelledError):
            await service.get_entity('entity', project_id='test')

        # CancelledError is BaseException, NOT Exception — the isinstance(r, Exception)
        # logging guard must NOT fire, so warning.call_count must be zero.
        assert mock_logger.warning.call_count == 0

    @pytest.mark.asyncio
    async def test_cancelled_error_from_search_propagates(self, service):
        """asyncio.CancelledError raised by search() must propagate unchanged.

        Symmetric to test_cancelled_error_propagates, but with search() raising
        CancelledError (gather index 1) and search_nodes() returning normally
        (gather index 0).  The detection guard must iterate ALL gather results
        regardless of which position holds the CancelledError.

        Also asserts logger.warning.call_count == 0 — CancelledError must NOT
        be logged via the isinstance(r, Exception) guard.
        """
        service.graphiti.search_nodes = AsyncMock(return_value=[])
        service.graphiti.search = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        with patch('fused_memory.services.memory_service.logger') as mock_logger, \
             pytest.raises(asyncio.CancelledError):
            await service.get_entity('entity', project_id='test')

        # CancelledError is BaseException, NOT Exception — logging guard must not fire.
        assert mock_logger.warning.call_count == 0

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


# ---------------------------------------------------------------------------
# Tests for _dedup_episode_edges  (steps 1, 3, 5, 6, 7)
# ---------------------------------------------------------------------------

class TestDedupEpisodeEdges:
    """Unit tests for MemoryService._dedup_episode_edges.

    step-1: 3 edges sharing same (source_node_uuid, target_node_uuid, fact)
            → 2 duplicates removed via bulk_remove_edges, returns count=2

    step-3: all edges distinct → returns 0, bulk_remove_edges NOT called

    step-5: None result / empty edges list → returns 0, no backend calls

    step-6: normalization (case + whitespace) → 'Auth depends on Redis' and
            '  auth  depends  on  redis  ' treated as duplicates

    step-7: same fact but different source/target pairs → NOT duplicates
    """

    @pytest.mark.asyncio
    async def test_three_duplicate_edges_removes_two(self, service):
        """step-1: 3 edges with same (source, target, fact) → 2 removed."""
        from unittest.mock import AsyncMock

        from tests.conftest import MockAddEpisodeResult, MockEdge

        service.graphiti.bulk_remove_edges = AsyncMock(return_value=2)

        edges = [
            MockEdge(
                fact='Auth depends on Redis',
                uuid='uuid-1',
                source_node_uuid='src-A',
                target_node_uuid='tgt-B',
            ),
            MockEdge(
                fact='Auth depends on Redis',
                uuid='uuid-2',
                source_node_uuid='src-A',
                target_node_uuid='tgt-B',
            ),
            MockEdge(
                fact='Auth depends on Redis',
                uuid='uuid-3',
                source_node_uuid='src-A',
                target_node_uuid='tgt-B',
            ),
        ]
        result = MockAddEpisodeResult(edges=edges)
        # Clear the entity_edges mirror so we test the 'edges' path directly
        result.entity_edges = []

        removed = await service._dedup_episode_edges(result)

        assert removed == 2
        service.graphiti.bulk_remove_edges.assert_called_once()
        deleted_uuids = service.graphiti.bulk_remove_edges.call_args[0][0]
        assert sorted(deleted_uuids) == ['uuid-2', 'uuid-3']

    @pytest.mark.asyncio
    async def test_no_duplicates_returns_zero_no_backend_call(self, service):
        """step-3: all edges distinct → 0 removed, bulk_remove_edges NOT called."""
        from unittest.mock import AsyncMock

        from tests.conftest import MockAddEpisodeResult, MockEdge

        service.graphiti.bulk_remove_edges = AsyncMock(return_value=0)

        edges = [
            MockEdge(
                fact='Auth depends on Redis',
                uuid='uuid-1',
                source_node_uuid='src-A',
                target_node_uuid='tgt-B',
            ),
            MockEdge(
                fact='Redis stores sessions',
                uuid='uuid-2',
                source_node_uuid='src-A',
                target_node_uuid='tgt-B',
            ),
            MockEdge(
                fact='Auth depends on Redis',
                uuid='uuid-3',
                source_node_uuid='src-C',   # different source — different relationship
                target_node_uuid='tgt-B',
            ),
        ]
        result = MockAddEpisodeResult(edges=edges)
        result.entity_edges = []

        removed = await service._dedup_episode_edges(result)

        assert removed == 0
        service.graphiti.bulk_remove_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_result_returns_zero(self, service):
        """step-5a: None result → 0, no backend calls."""
        from unittest.mock import AsyncMock

        service.graphiti.bulk_remove_edges = AsyncMock(return_value=0)

        removed = await service._dedup_episode_edges(None)

        assert removed == 0
        service.graphiti.bulk_remove_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_edges_list_returns_zero(self, service):
        """step-5b: result with empty edges list → 0, no backend calls."""
        from unittest.mock import AsyncMock

        from tests.conftest import MockAddEpisodeResult

        service.graphiti.bulk_remove_edges = AsyncMock(return_value=0)

        result = MockAddEpisodeResult(edges=[])
        result.entity_edges = []

        removed = await service._dedup_episode_edges(result)

        assert removed == 0
        service.graphiti.bulk_remove_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_normalization_treats_case_whitespace_variants_as_duplicate(self, service):
        """step-6: case + whitespace normalization catches near-duplicate facts."""
        from unittest.mock import AsyncMock

        from tests.conftest import MockAddEpisodeResult, MockEdge

        service.graphiti.bulk_remove_edges = AsyncMock(return_value=1)

        edges = [
            MockEdge(
                fact='Auth depends on Redis',
                uuid='uuid-1',
                source_node_uuid='src-A',
                target_node_uuid='tgt-B',
            ),
            MockEdge(
                fact='  auth  depends  on  redis  ',
                uuid='uuid-2',
                source_node_uuid='src-A',
                target_node_uuid='tgt-B',
            ),
        ]
        result = MockAddEpisodeResult(edges=edges)
        result.entity_edges = []

        removed = await service._dedup_episode_edges(result)

        assert removed == 1
        service.graphiti.bulk_remove_edges.assert_called_once()
        deleted_uuids = service.graphiti.bulk_remove_edges.call_args[0][0]
        assert deleted_uuids == ['uuid-2']

    @pytest.mark.asyncio
    async def test_same_fact_different_node_pairs_not_duplicates(self, service):
        """step-7: same fact text but different source/target pairs → distinct edges."""
        from unittest.mock import AsyncMock

        from tests.conftest import MockAddEpisodeResult, MockEdge

        service.graphiti.bulk_remove_edges = AsyncMock(return_value=0)

        edges = [
            MockEdge(
                fact='depends on Redis',
                uuid='uuid-1',
                source_node_uuid='src-A',
                target_node_uuid='tgt-B',
            ),
            MockEdge(
                fact='depends on Redis',
                uuid='uuid-2',
                source_node_uuid='src-C',   # different source
                target_node_uuid='tgt-B',
            ),
            MockEdge(
                fact='depends on Redis',
                uuid='uuid-3',
                source_node_uuid='src-A',
                target_node_uuid='tgt-D',   # different target
            ),
        ]
        result = MockAddEpisodeResult(edges=edges)
        result.entity_edges = []

        removed = await service._dedup_episode_edges(result)

        assert removed == 0
        service.graphiti.bulk_remove_edges.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for _execute_graphiti_write integration with dedup  (steps 8, 10)
# ---------------------------------------------------------------------------

class TestExecuteGraphitiWriteWithDedup:
    """Integration tests for dedup wiring in _execute_graphiti_write.

    step-8: _execute_graphiti_write calls _dedup_episode_edges with result,
            duplicate edges are removed before returning

    step-10: _execute_graphiti_write returns normally when add_episode
             returns None (no edges to dedup, no crash)
    """

    @pytest.mark.asyncio
    async def test_execute_graphiti_write_calls_dedup_and_removes_duplicates(self, service):
        """step-8: dedup is called after add_episode; duplicates are removed."""
        from unittest.mock import AsyncMock

        from tests.conftest import MockAddEpisodeResult, MockEdge

        dup_edges = [
            MockEdge(fact='X depends on Y', uuid='u1', source_node_uuid='s1', target_node_uuid='t1'),
            MockEdge(fact='X depends on Y', uuid='u2', source_node_uuid='s1', target_node_uuid='t1'),
        ]
        mock_result = MockAddEpisodeResult(edges=dup_edges)
        mock_result.entity_edges = []

        service.graphiti.add_episode = AsyncMock(return_value=mock_result)
        service.graphiti.bulk_remove_edges = AsyncMock(return_value=1)

        payload = {
            'name': 'ep_test',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': '',
        }
        await service._execute_graphiti_write('add_episode', payload)

        # dedup must have removed the second edge
        service.graphiti.bulk_remove_edges.assert_called_once()
        deleted_uuids = service.graphiti.bulk_remove_edges.call_args[0][0]
        assert deleted_uuids == ['u2']

    @pytest.mark.asyncio
    async def test_execute_graphiti_write_none_result_no_crash(self, service):
        """step-10: add_episode returns None → _execute_graphiti_write does not crash."""
        from unittest.mock import AsyncMock

        service.graphiti.add_episode = AsyncMock(return_value=None)
        service.graphiti.bulk_remove_edges = AsyncMock(return_value=0)

        payload = {
            'name': 'ep_test',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': '',
        }
        result = await service._execute_graphiti_write('add_episode', payload)

        # Should return None without crashing
        assert result is None
        service.graphiti.bulk_remove_edges.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for _dual_write_callback reading result.edges  (step 11)
# ---------------------------------------------------------------------------

class TestDualWriteCallbackEdgesField:
    """step-11: _dual_write_callback reads edges from result.edges (the real
    field name) not result.entity_edges.
    """

    @pytest.mark.asyncio
    async def test_callback_reads_real_edges_field(self, service):
        """result.edges (not result.entity_edges) drives dual-write enqueue."""
        from tests.conftest import MockAddEpisodeResult, MockEdge

        # Build a mock where entity_edges is empty but edges has content.
        # After our pre-1 fix, MockAddEpisodeResult mirrors entity_edges→edges,
        # but here we explicitly set them to different values to verify the
        # callback reads from 'edges'.
        result = MockAddEpisodeResult.__new__(MockAddEpisodeResult)
        result.entity_edges = []
        result.edges = [
            MockEdge(fact='Auth depends on Redis', uuid='e1'),
            MockEdge(fact='Redis stores sessions', uuid='e2'),
        ]

        payload = {
            'project_id': 'test',
            'agent_id': 'test-agent',
            '_causation_id': 'caus-1',
        }
        await service._dual_write_callback('dual_write_episode', result, payload)

        service.durable_queue.enqueue_batch.assert_called_once()
        batch = service.durable_queue.enqueue_batch.call_args[0][0]
        assert len(batch) == 2, (
            '_dual_write_callback must read from result.edges (got 2 edges) '
            f'but found {len(batch)} items — it may still be reading entity_edges'
        )


class TestExecuteGraphitiWritePlanningRegistration:
    """step-5: _execute_graphiti_write registers episodes when temporal_context='planning'."""

    @pytest.mark.asyncio
    async def test_planning_episode_registered_in_registry(self, service):
        """After successful graphiti.add_episode with temporal_context='planning',
        the episode UUID should be registered in the planned_episode_registry."""
        mock_registry = MagicMock()
        mock_registry.register = AsyncMock()
        service.planned_episode_registry = mock_registry

        payload = {
            'uuid': 'episode-plan-uuid',
            'name': 'episode_plan',
            'content': 'PRD content',
            'source': 'text',
            'group_id': 'myproject',
            'source_description': '[temporal:planning] plan',
            'temporal_context': 'planning',
        }
        await service._execute_graphiti_write('add_episode', payload)

        mock_registry.register.assert_called_once_with('episode-plan-uuid', 'myproject')

    @pytest.mark.asyncio
    async def test_no_temporal_context_skips_registration(self, service):
        """Without temporal_context, no registration should occur."""
        mock_registry = MagicMock()
        mock_registry.register = AsyncMock()
        service.planned_episode_registry = mock_registry

        payload = {
            'uuid': 'episode-normal-uuid',
            'name': 'episode_normal',
            'content': 'Regular content',
            'source': 'text',
            'group_id': 'myproject',
            'source_description': 'normal',
        }
        await service._execute_graphiti_write('add_episode', payload)

        mock_registry.register.assert_not_called()

    @pytest.mark.asyncio
    async def test_current_temporal_context_skips_registration(self, service):
        """temporal_context='current' should NOT trigger registration."""
        mock_registry = MagicMock()
        mock_registry.register = AsyncMock()
        service.planned_episode_registry = mock_registry

        payload = {
            'uuid': 'episode-current-uuid',
            'name': 'episode_current',
            'content': 'Current content',
            'source': 'text',
            'group_id': 'myproject',
            'source_description': 'current',
            'temporal_context': 'current',
        }
        await service._execute_graphiti_write('add_episode', payload)

        mock_registry.register.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_registry_no_error_on_planning(self, service):
        """If planned_episode_registry is None (not wired), no error on planning episode."""
        # Ensure attribute is None (default)
        service.planned_episode_registry = None

        payload = {
            'uuid': 'episode-orphan-uuid',
            'name': 'episode_orphan',
            'content': 'PRD content',
            'source': 'text',
            'group_id': 'myproject',
            'source_description': 'plan',
            'temporal_context': 'planning',
        }
        # Should not raise
        await service._execute_graphiti_write('add_episode', payload)

    @pytest.mark.asyncio
    async def test_empty_group_id_skips_registration(self, service):
        """When group_id is empty string in payload, register() must NOT be called.

        Currently FAILS because the code calls register(uuid, '') with an empty
        group_id, silently making the episode invisible to search filtering which
        queries by actual project_id (never matches '').
        """
        mock_registry = MagicMock()
        mock_registry.register = AsyncMock()
        service.planned_episode_registry = mock_registry

        # Payload with temporal_context='planning' but group_id is empty string
        payload = {
            'uuid': 'episode-empty-group',
            'name': 'episode_emptygroup',
            'content': 'PRD content with empty group_id',
            'source': 'text',
            'group_id': '',  # Empty — falsy group_id
            'source_description': '[temporal:planning] plan',
            'temporal_context': 'planning',
        }
        await service._execute_graphiti_write('add_episode', payload)

        # Registration must be skipped — not called with empty group_id
        mock_registry.register.assert_not_called()


class TestSearchGraphitiFiltering:
    """step-11: _search_graphiti excludes edges whose ALL episodes are in planned registry."""

    @pytest.fixture
    def service_with_registry(self, service):
        """Service with a mocked planned_episode_registry."""
        from unittest.mock import AsyncMock, MagicMock
        mock_registry = MagicMock()
        mock_registry.get_planned_uuids = AsyncMock(return_value=set())
        service.planned_episode_registry = mock_registry
        return service

    @pytest.mark.asyncio
    async def test_edge_with_all_planned_episodes_excluded_by_default(
        self, service_with_registry
    ):
        """Edge whose ALL episode UUIDs are in the planned registry is excluded by default."""
        from fused_memory.models.scope import Scope
        from tests.conftest import MockEdge

        ep1, ep2 = 'plan-ep-1', 'plan-ep-2'
        service_with_registry.planned_episode_registry.get_planned_uuids = AsyncMock(
            return_value={ep1, ep2}
        )
        service_with_registry.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='PRD: CostStore extends AgentResult',
                uuid='edge-planned-1',
                episodes=[ep1, ep2],
            ),
        ])

        scope = Scope(project_id='test')
        results = await service_with_registry._search_graphiti('CostStore', scope, 10)

        assert len(results) == 0, (
            'Edge with all-planned episodes must be excluded from default search results'
        )

    @pytest.mark.asyncio
    async def test_edge_with_mixed_episodes_not_excluded(
        self, service_with_registry
    ):
        """Edge with mixed episodes (some planned, some not) is NOT excluded."""
        from fused_memory.models.scope import Scope
        from tests.conftest import MockEdge

        ep_planned = 'plan-ep-1'
        ep_real = 'real-ep-2'
        service_with_registry.planned_episode_registry.get_planned_uuids = AsyncMock(
            return_value={ep_planned}
        )
        service_with_registry.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Fact confirmed in both plan and reality',
                uuid='edge-mixed-1',
                episodes=[ep_planned, ep_real],
            ),
        ])

        scope = Scope(project_id='test')
        results = await service_with_registry._search_graphiti('fact', scope, 10)

        assert len(results) == 1, (
            'Edge with mixed episodes must NOT be excluded (only exclude all-planned)'
        )

    @pytest.mark.asyncio
    async def test_edge_with_no_episodes_not_excluded(
        self, service_with_registry
    ):
        """Edge with no episode provenance is NOT excluded (not a planned edge)."""
        from fused_memory.models.scope import Scope
        from tests.conftest import MockEdge

        service_with_registry.planned_episode_registry.get_planned_uuids = AsyncMock(
            return_value={'plan-ep-1'}
        )
        service_with_registry.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Fact with no provenance',
                uuid='edge-no-ep-1',
                episodes=[],
            ),
        ])

        scope = Scope(project_id='test')
        results = await service_with_registry._search_graphiti('fact', scope, 10)

        assert len(results) == 1, 'Edge with no episodes must NOT be excluded'

    @pytest.mark.asyncio
    async def test_edge_with_non_planned_episodes_not_excluded(
        self, service_with_registry
    ):
        """Edge with all non-planned episodes is NOT excluded."""
        from fused_memory.models.scope import Scope
        from tests.conftest import MockEdge

        service_with_registry.planned_episode_registry.get_planned_uuids = AsyncMock(
            return_value={'plan-ep-1'}
        )
        service_with_registry.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Implemented fact',
                uuid='edge-real-1',
                episodes=['real-ep-a', 'real-ep-b'],
            ),
        ])

        scope = Scope(project_id='test')
        results = await service_with_registry._search_graphiti('fact', scope, 10)

        assert len(results) == 1, 'Edge with non-planned episodes must NOT be excluded'

    @pytest.mark.asyncio
    async def test_no_registry_does_not_filter(self, service):
        """When planned_episode_registry is None, no filtering occurs."""
        from fused_memory.models.scope import Scope
        from tests.conftest import MockEdge

        service.planned_episode_registry = None
        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(fact='Some fact', uuid='edge-1', episodes=['ep-1']),
        ])

        scope = Scope(project_id='test')
        results = await service._search_graphiti('fact', scope, 10)

        assert len(results) == 1, 'Without registry, all edges must be returned'


class TestSearchGraphitiIncludePlanned:
    """step-13: _search_graphiti with include_planned=True includes planned edges and marks them."""

    @pytest.fixture
    def service_with_registry(self, service):
        """Service with a mocked planned_episode_registry."""
        from unittest.mock import AsyncMock, MagicMock
        mock_registry = MagicMock()
        mock_registry.get_planned_uuids = AsyncMock(return_value=set())
        service.planned_episode_registry = mock_registry
        return service

    @pytest.mark.asyncio
    async def test_planned_edges_included_when_include_planned_true(
        self, service_with_registry
    ):
        """With include_planned=True, edges that would normally be filtered are included."""
        from fused_memory.models.scope import Scope
        from tests.conftest import MockEdge

        ep1, ep2 = 'plan-ep-1', 'plan-ep-2'
        service_with_registry.planned_episode_registry.get_planned_uuids = AsyncMock(
            return_value={ep1, ep2}
        )
        service_with_registry.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='PRD: CostStore extends AgentResult',
                uuid='edge-planned-1',
                episodes=[ep1, ep2],
            ),
        ])

        scope = Scope(project_id='test')
        results = await service_with_registry._search_graphiti(
            'CostStore', scope, 10, include_planned=True
        )

        assert len(results) == 1, (
            'With include_planned=True, planned edges must be returned'
        )

    @pytest.mark.asyncio
    async def test_planned_edges_marked_in_metadata(
        self, service_with_registry
    ):
        """With include_planned=True, planned edges have metadata['planned'] = True."""
        from fused_memory.models.scope import Scope
        from tests.conftest import MockEdge

        ep1 = 'plan-ep-1'
        service_with_registry.planned_episode_registry.get_planned_uuids = AsyncMock(
            return_value={ep1}
        )
        service_with_registry.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='PRD: planned fact',
                uuid='edge-planned-2',
                episodes=[ep1],
            ),
        ])

        scope = Scope(project_id='test')
        results = await service_with_registry._search_graphiti(
            'planned', scope, 10, include_planned=True
        )

        assert len(results) == 1
        assert results[0].metadata.get('planned') is True, (
            "Planned edges must have metadata['planned'] = True when include_planned=True"
        )


class TestSearchMem0Filtering:
    """step-15: _search_mem0 filtering — exclude planned=True by default, include with flag."""

    @pytest.mark.asyncio
    async def test_planned_result_excluded_by_default(self, service):
        """Result with metadata.planned=True is excluded from default search."""
        from fused_memory.models.scope import Scope

        service.mem0.search = AsyncMock(return_value={
            'results': [
                {
                    'id': 'm-planned-1',
                    'memory': 'PRD: system will use GraphQL',
                    'score': 0.9,
                    'metadata': {'category': 'decisions_and_rationale', 'planned': True},
                },
            ]
        })

        scope = Scope(project_id='test')
        results = await service._search_mem0('GraphQL', scope, 10)

        assert len(results) == 0, (
            'Result with planned=True must be excluded from default Mem0 search'
        )

    @pytest.mark.asyncio
    async def test_planned_result_included_when_include_planned_true(self, service):
        """Result with metadata.planned=True is included when include_planned=True."""
        from fused_memory.models.scope import Scope

        service.mem0.search = AsyncMock(return_value={
            'results': [
                {
                    'id': 'm-planned-2',
                    'memory': 'PRD: system will use GraphQL',
                    'score': 0.9,
                    'metadata': {'category': 'decisions_and_rationale', 'planned': True},
                },
            ]
        })

        scope = Scope(project_id='test')
        results = await service._search_mem0('GraphQL', scope, 10, include_planned=True)

        assert len(results) == 1, (
            'Result with planned=True must be included when include_planned=True'
        )

    @pytest.mark.asyncio
    async def test_non_planned_result_not_excluded(self, service):
        """Result without planned metadata is NOT excluded."""
        from fused_memory.models.scope import Scope

        service.mem0.search = AsyncMock(return_value={
            'results': [
                {
                    'id': 'm-real-1',
                    'memory': 'We use PostgreSQL for persistence',
                    'score': 0.85,
                    'metadata': {'category': 'decisions_and_rationale'},
                },
            ]
        })

        scope = Scope(project_id='test')
        results = await service._search_mem0('PostgreSQL', scope, 10)

        assert len(results) == 1, 'Non-planned result must NOT be excluded'


class TestSearchIncludePlannedPassthrough:
    """step-17: MemoryService.search passes include_planned to _search_graphiti and _search_mem0."""

    @pytest.mark.asyncio
    async def test_include_planned_true_passes_through_to_graphiti(self, service):
        """search(include_planned=True) passes the flag to _search_graphiti."""
        from unittest.mock import patch

        captured_kwargs = {}

        async def mock_search_graphiti(query, scope, limit, include_planned=False):
            captured_kwargs['include_planned'] = include_planned
            return []

        with patch.object(service, '_search_graphiti', side_effect=mock_search_graphiti):
            await service.search(
                query='test', project_id='test', stores=['graphiti'],
                include_planned=True
            )

        assert captured_kwargs.get('include_planned') is True, (
            'include_planned=True must be forwarded to _search_graphiti'
        )

    @pytest.mark.asyncio
    async def test_include_planned_false_passes_through_to_mem0(self, service):
        """search(include_planned=False) [default] passes False to _search_mem0."""
        from unittest.mock import patch

        captured_kwargs = {}

        async def mock_search_mem0(query, scope, limit, include_planned=False):
            captured_kwargs['include_planned'] = include_planned
            return []

        with patch.object(service, '_search_mem0', side_effect=mock_search_mem0):
            await service.search(
                query='test', project_id='test', stores=['mem0'],
            )

        assert captured_kwargs.get('include_planned') is False, (
            'include_planned=False (default) must be forwarded to _search_mem0'
        )

    @pytest.mark.asyncio
    async def test_include_planned_true_passes_through_to_mem0(self, service):
        """search(include_planned=True) passes True to _search_mem0."""
        from unittest.mock import patch

        captured_kwargs = {}

        async def mock_search_mem0(query, scope, limit, include_planned=False):
            captured_kwargs['include_planned'] = include_planned
            return []

        with patch.object(service, '_search_mem0', side_effect=mock_search_mem0):
            await service.search(
                query='test', project_id='test', stores=['mem0'],
                include_planned=True
            )

        assert captured_kwargs.get('include_planned') is True, (
            'include_planned=True must be forwarded to _search_mem0'
        )


class TestInitializeLifecycleConflict:
    """step-29/30: initialize() must not overwrite an externally-set registry."""

    @pytest.mark.asyncio
    async def test_external_registry_preserved_after_initialize(self, service, mock_config):
        """If set_planned_registry() was called before initialize(), the external
        registry must be preserved. Currently FAILS because initialize() unconditionally
        creates a new registry, replacing the external one.
        """
        # Simulate external wiring via set_planned_registry()
        external_registry = MagicMock()
        external_registry.initialize = AsyncMock()
        service.set_planned_registry(external_registry)

        # Patch DurableWriteQueue and PlannedEpisodeRegistry so we can call initialize()
        # without real backends. The key contract: when planned_episode_registry is already
        # set externally, initialize() must NOT create a new one.
        mock_registry_inst = MagicMock()
        mock_registry_inst.initialize = AsyncMock()
        MockRegistryCls = MagicMock(return_value=mock_registry_inst)

        mock_dq_inst = MagicMock()
        mock_dq_inst.initialize = AsyncMock()
        mock_dq_inst.register_callback = MagicMock()

        with (
            patch.object(service.graphiti, 'initialize', new_callable=AsyncMock),
            patch('fused_memory.services.memory_service.DurableWriteQueue', return_value=mock_dq_inst),
            patch(
                'fused_memory.services.planned_episode_registry.PlannedEpisodeRegistry',
                MockRegistryCls,
            ),
        ):
            await service.initialize()

        # The externally-set registry must be the one that's still in place
        assert service.planned_episode_registry is external_registry, (
            'initialize() must not replace an externally-set registry'
        )
        # PlannedEpisodeRegistry constructor must NOT have been called
        MockRegistryCls.assert_not_called()
