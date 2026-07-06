"""Tests for the memory service — unit tests with mocked backends."""

import asyncio
import logging
import types
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.scope import Scope
from fused_memory.services import memory_service
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
    svc.graphiti.update_edge = AsyncMock(
        return_value={'uuid': 'test-uuid', 'fact': 'updated', 'refreshed_nodes': []}
    )
    svc.graphiti.get_edge_text = AsyncMock(return_value=('edge-name', 'updated'))
    svc.graphiti.get_edge_invalid_at = AsyncMock(return_value=None)
    svc.graphiti._require_client = MagicMock()
    svc.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[])
    svc.graphiti.get_valid_edges_for_node = AsyncMock(return_value=[])

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
        # Mem0 is now a direct synchronous call — NOT enqueued
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
        # Graphiti still goes through the durable queue; Mem0 is now a direct call
        assert service.durable_queue.enqueue.call_count == 1
        ops = [c[1]['operation'] for c in service.durable_queue.enqueue.call_args_list]
        assert ops == ['add_memory_graphiti']
        service.mem0.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_dual_write_returns_mem0_memory_ids(self, service):
        """dual_write=True must still return Mem0 memory_ids synchronously."""
        result = await service.add_memory(
            content='We decided to use PostgreSQL for its JSON support',
            category='decisions_and_rationale',
            project_id='test',
            dual_write=True,
        )
        assert result.memory_ids == ['mem0-1'], (
            f'Expected memory_ids=[\'mem0-1\'] for dual_write, got {result.memory_ids!r}'
        )
        assert SourceStore.graphiti in result.stores_written
        assert SourceStore.mem0 in result.stores_written

    @pytest.mark.asyncio
    async def test_auto_classification(self, service):
        result = await service.add_memory(
            content='The payment gateway depends on the billing API',
            project_id='test',
        )
        # Should auto-classify — with heuristic-only config, entities_and_relations
        assert result.category is not None

    @pytest.mark.asyncio
    async def test_mem0_primary_returns_memory_ids(self, service):
        """add_memory must return the server-assigned Mem0 IDs synchronously.

        The fixture has svc.mem0.add returning {'results': [{'id': 'mem0-1'}]}.
        After the fix (direct synchronous call instead of durable-queue enqueue),
        result.memory_ids must be ['mem0-1'].
        """
        result = await service.add_memory(
            content='Always use type hints',
            category='preferences_and_norms',
            project_id='test',
        )
        assert result.memory_ids == ['mem0-1'], (
            f'Expected memory_ids=[\'mem0-1\'], got {result.memory_ids!r}. '
            'The Mem0 write path must be synchronous so IDs are available to the caller.'
        )

    @pytest.mark.asyncio
    async def test_mem0_add_called_with_scope(self, service):
        """mem0.add must be called with correct scope and metadata kwargs."""
        from fused_memory.models.scope import Scope

        await service.add_memory(
            content='Always use type hints',
            category='preferences_and_norms',
            project_id='test',
            agent_id='a1',
            session_id='s1',
        )

        service.mem0.add.assert_called_once()
        call_kwargs = service.mem0.add.call_args[1]
        assert call_kwargs['content'] == 'Always use type hints'
        scope: Scope = call_kwargs['scope']
        assert scope.project_id == 'test'
        assert scope.agent_id == 'a1'
        assert scope.session_id == 's1'
        metadata = call_kwargs['metadata']
        assert metadata.get('category') == 'preferences_and_norms'

    @pytest.mark.asyncio
    async def test_mem0_direct_error_surfaced_in_response(self, service):
        """Mem0 direct-call errors must appear in the response message."""
        service.mem0.add = AsyncMock(
            side_effect=RuntimeError('qdrant write failed')
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
        """success must be False when the only targeted store's direct call fails.

        For a Mem0-only write (preferences_and_norms), if mem0.add raises,
        _graphiti_error is None and _mem0_error is set.
        """
        service.mem0.add = AsyncMock(
            side_effect=ValueError('qdrant unreachable')
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

    @pytest.mark.asyncio
    async def test_memory_ids_logged_to_write_journal(self, service):
        """write_journal.log_write_op result_summary must contain the real memory_ids."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        mock_journal.log_backend_op = AsyncMock()
        service._write_journal = mock_journal

        await service.add_memory(
            content='Always use type hints',
            category='preferences_and_norms',
            project_id='test',
        )

        mock_journal.log_write_op.assert_called_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs['result_summary']['memory_ids'] == ['mem0-1'], (
            f'Expected memory_ids=[\'mem0-1\'] in write journal, '
            f'got {call_kwargs["result_summary"]["memory_ids"]!r}'
        )

    @pytest.mark.asyncio
    async def test_memory_ids_emitted_in_reconciliation_event(self, service):
        """ReconciliationEvent payload must include the real memory_ids."""
        pushed_events: list = []

        class FakeBuffer:
            async def push(self, event):
                pushed_events.append(event)

        service.set_event_buffer(FakeBuffer())

        await service.add_memory(
            content='Always use type hints',
            category='preferences_and_norms',
            project_id='test',
        )

        assert len(pushed_events) == 1, f'Expected 1 event, got {len(pushed_events)}'
        event = pushed_events[0]
        assert event.payload['memory_ids'] == ['mem0-1'], (
            f'Expected memory_ids=[\'mem0-1\'] in reconciliation event, '
            f'got {event.payload["memory_ids"]!r}'
        )

    @pytest.mark.asyncio
    async def test_mem0_add_empty_result_logs_warning(self, service, caplog):
        """task 1974: a mem0.add call that returns {'results': []} WITHOUT
        raising is a silent dedup/no-op drop — the exact failure mode that
        made Stage 2's cycle_summary write go fully absent with no trace.
        It must be logged as a WARNING even though add_memory() itself does
        not raise and behavior (response/stores_written/journal) is unchanged.
        """
        service.mem0.add = AsyncMock(return_value={'results': []})

        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            result = await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='dark_factory',
                metadata={
                    'kind': 'cycle_summary',
                    'stage': 'task_knowledge_sync',
                    'run_id': 'run-x',
                },
                causation_id='run-x',
            )

        # Behavior must be UNCHANGED — this is observability-only.
        assert result.memory_ids == [], (
            f'Expected memory_ids=[] for an empty mem0 result, got {result.memory_ids!r}'
        )
        assert SourceStore.mem0 in result.stores_written, (
            'stores_written must still include mem0 — the write was attempted '
            'and did not raise; only the returned ids are empty.'
        )

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1, (
            f'Expected exactly 1 WARNING for the silent empty-result drop, '
            f'got {len(warning_records)}: {[r.message for r in warning_records]}'
        )
        record = warning_records[0]
        message = record.message.lower()
        assert 'mem0' in message and ('empty' in message or 'zero' in message), (
            f'Expected the WARNING to identify a silent mem0 empty-result drop, '
            f'got: {record.message!r}'
        )
        assert record.kind == 'cycle_summary'
        assert record.category == 'observations_and_summaries'
        assert record.causation_id == 'run-x'
        assert record.project_id == 'dark_factory'

    @pytest.mark.asyncio
    async def test_mem0_add_nonempty_result_no_empty_result_warning(self, service, caplog):
        """The happy path (fixture default, non-empty mem0 result) must NOT
        trigger the empty-result WARNING — no happy-path false positives."""
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            result = await service.add_memory(
                content='Always use type hints in Python code',
                category='observations_and_summaries',
                project_id='dark_factory',
                metadata={
                    'kind': 'cycle_summary',
                    'stage': 'task_knowledge_sync',
                    'run_id': 'run-y',
                },
                causation_id='run-y',
            )

        assert result.memory_ids == ['mem0-1']
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            f'Expected no WARNING on the happy path, got: '
            f'{[r.message for r in warning_records]}'
        )

    @pytest.mark.asyncio
    async def test_mem0_add_empty_result_warning_gated_on_infer_pin(
        self, service, caplog, monkeypatch
    ):
        """The empty-result WARNING is only meaningful while Mem0Backend.add()
        pins infer=False (a successful write always returns exactly one
        result). It is gated on `_MEM0_ADD_INFER_PINNED_FALSE` so that if the
        pin is ever lifted or made configurable, flipping that one constant
        silences the warning instead of it becoming a recurring,
        non-actionable log line for a legitimate infer-driven no-op."""
        monkeypatch.setattr(memory_service, '_MEM0_ADD_INFER_PINNED_FALSE', False)
        service.mem0.add = AsyncMock(return_value={'results': []})

        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            result = await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='dark_factory',
                metadata={
                    'kind': 'cycle_summary',
                    'stage': 'task_knowledge_sync',
                    'run_id': 'run-x',
                },
                causation_id='run-x',
            )

        # Behavior (response/stores_written) is unaffected by the gate.
        assert result.memory_ids == []
        assert SourceStore.mem0 in result.stores_written

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            f'Expected no WARNING once the infer=False pin no longer holds, got: '
            f'{[r.message for r in warning_records]}'
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
    async def test_mem0_add_queue_operation_still_dispatches(self, service):
        """Backward-compat: in-flight 'mem0_add' queue items must still drain.

        The add_memory() path no longer enqueues 'mem0_add', but items
        written to the queue before this fix must still execute correctly
        so no data is lost on restart.
        """
        payload = {
            'content': 'Legacy queued content',
            'metadata': {'category': 'preferences_and_norms'},
            'project_id': 'test',
            'agent_id': 'legacy-agent',
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
        from _fm_helpers import MockAddEpisodeResult, MockEdge

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
        from _fm_helpers import MockAddEpisodeResult

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
        from _fm_helpers import MockAddEpisodeResult, MockEdge

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
        from _fm_helpers import MockAddEpisodeResult, MockEdge

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
        from _fm_helpers import MockEdge, MockNode

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
        from _fm_helpers import MockEdge, MockNode

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
        from _fm_helpers import MockEdge, MockNode

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
        from _fm_helpers import MockEdge, MockNode

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
        """Invalidated edges (non-null invalid_at) are excluded from search results.

        Task 312: changed behavior — previously this test asserted that an edge
        with only invalid_at set was returned (len==1). After the fix, superseded
        edges are filtered out, so the result must be empty (len==0).
        """
        from _fm_helpers import MockEdge, MockNode

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
        # Task 312: invalidated edges are now filtered out — expect empty results
        assert len(results) == 0, (
            'Edge with non-null invalid_at must be excluded from search results (task 312)'
        )

    @pytest.mark.asyncio
    async def test_search_category_false_negative_regression_task_1083(self, service):
        """E2E regression: category-scoped search must return the matching low-similarity
        memory even when higher-similarity non-matching memories exist.

        Without the pushdown fix (task 1083), the matching procedural_knowledge memory
        would never reach the Python-side post-filter because Mem0 returns top-N
        irrespective of category, saturating the result set with observations_and_summaries.
        With the fix, categories is forwarded to Mem0 server-side so only
        procedural_knowledge memories are returned, and the matching one is found.
        """
        high_sim_obs = [
            {
                'id': f'obs-{i}',
                'memory': f'observation {i}',
                'score': 0.9 - i * 0.01,
                'metadata': {'category': 'observations_and_summaries'},
            }
            for i in range(10)
        ]
        proc_memory = {
            'id': 'proc-1',
            'memory': 'Deploy via: uv run python -m app.deploy',
            'score': 0.4,
            'metadata': {'category': 'procedural_knowledge'},
        }

        def mem0_side_effect(**kwargs):
            cats = kwargs.get('categories') or []
            if cats == ['procedural_knowledge']:
                return {'results': [proc_memory]}
            return {'results': high_sim_obs}

        service.mem0.search = AsyncMock(side_effect=mem0_side_effect)

        results = await service.search(
            query='how do I deploy',
            project_id='test',
            categories=['procedural_knowledge'],
            stores=['mem0'],
        )
        assert len(results) == 1, (
            f'Expected 1 result but got {len(results)} — category pushdown may be broken'
        )
        assert results[0].id == 'proc-1'

    @pytest.mark.asyncio
    async def test_search_pushes_categories_to_mem0_backend(self, service):
        """categories kwarg must be forwarded from MemoryService.search → _search_mem0
        → Mem0Backend.search so the filter is applied server-side (task 1083)."""
        service.mem0.search = AsyncMock(return_value={'results': []})
        await service.search(
            query='x',
            project_id='test',
            categories=['preferences_and_norms'],
            stores=['mem0'],
        )
        call_kwargs = service.mem0.search.call_args.kwargs
        assert 'categories' in call_kwargs, (
            '_search_mem0 did not forward categories kwarg to mem0.search'
        )
        assert call_kwargs['categories'] == ['preferences_and_norms']


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
        service.graphiti.remove_edge.assert_called_once_with('edge-uuid-123', group_id='test')
        service.graphiti.remove_episode.assert_not_called()


class TestUpdateEdge:
    @pytest.mark.asyncio
    async def test_update_edge_success(self, service):
        result = await service.update_edge(
            edge_uuid='edge-1', fact='new fact', project_id='test'
        )
        assert result['status'] == 'updated'
        assert result['store'] == 'graphiti'

    @pytest.mark.asyncio
    async def test_update_edge_calls_backend(self, service):
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'new fact', 'refreshed_nodes': []}
        )
        await service.update_edge(edge_uuid='e-1', fact='new fact', project_id='proj')
        service.graphiti.update_edge.assert_called_once_with(
            'e-1', 'new fact', group_id='proj', invalid_at=None,
            clear_invalid_at=False,
        )

    @pytest.mark.asyncio
    async def test_update_edge_invalid_at_only(self, service):
        from datetime import UTC, datetime
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'unchanged', 'refreshed_nodes': []}
        )
        ts = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)
        await service.update_edge(
            edge_uuid='e-1', project_id='proj', invalid_at=ts,
        )
        service.graphiti.update_edge.assert_called_once_with(
            'e-1', None, group_id='proj', invalid_at=ts,
            clear_invalid_at=False,
        )

    @pytest.mark.asyncio
    async def test_update_edge_requires_fact_or_invalid_at(self, service):
        with pytest.raises(ValueError, match='fact, invalid_at, or clear_invalid_at'):
            await service.update_edge(edge_uuid='e-1', project_id='proj')

    @pytest.mark.asyncio
    async def test_update_edge_not_found_propagates(self, service):
        from graphiti_core.errors import EdgeNotFoundError

        service.graphiti.update_edge = AsyncMock(
            side_effect=EdgeNotFoundError('e-missing')
        )
        with pytest.raises(EdgeNotFoundError):
            await service.update_edge(
                edge_uuid='e-missing', fact='x', project_id='test'
            )

    @pytest.mark.asyncio
    async def test_update_edge_returns_refreshed_nodes(self, service):
        service.graphiti.update_edge = AsyncMock(return_value={
            'uuid': 'e-1', 'fact': 'new', 'refreshed_nodes': ['n-src', 'n-tgt'],
        })
        result = await service.update_edge(
            edge_uuid='e-1', fact='new', project_id='test'
        )
        assert result['refreshed_nodes'] == ['n-src', 'n-tgt']


class TestUpdateEdgeVerification:
    """Tests for the post-write persistence verification in update_edge (Guard 2)."""

    @pytest.mark.asyncio
    async def test_update_edge_returns_verified_true_when_readback_matches(self, service):
        """When get_edge_text returns the same fact text, verified must be True."""
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'new fact', 'refreshed_nodes': []}
        )
        service.graphiti.get_edge_text = AsyncMock(return_value=('Edge', 'new fact'))

        result = await service.update_edge(
            edge_uuid='e-1', fact='new fact', project_id='test'
        )

        assert result['verified'] is True
        service.graphiti.get_edge_text.assert_called_once_with('e-1', group_id='test')

    @pytest.mark.asyncio
    async def test_update_edge_returns_verified_false_when_readback_differs(self, service):
        """When get_edge_text returns a different fact text, verified must be False."""
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'new fact', 'refreshed_nodes': []}
        )
        service.graphiti.get_edge_text = AsyncMock(
            return_value=('Edge', 'old stale fact')
        )

        result = await service.update_edge(
            edge_uuid='e-1', fact='new fact', project_id='test'
        )

        assert result['verified'] is False

    @pytest.mark.asyncio
    async def test_update_edge_returns_verified_false_when_readback_raises(self, service):
        """When get_edge_text raises EdgeNotFoundError, verified must be False
        and verification_error must be a non-empty string mentioning EdgeNotFoundError.
        The save itself succeeded — do NOT re-raise the readback exception.
        """
        from graphiti_core.errors import EdgeNotFoundError

        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'new fact', 'refreshed_nodes': []}
        )
        service.graphiti.get_edge_text = AsyncMock(
            side_effect=EdgeNotFoundError('e-1')
        )

        result = await service.update_edge(
            edge_uuid='e-1', fact='new fact', project_id='test'
        )

        assert result['verified'] is False
        assert 'verification_error' in result
        assert 'EdgeNotFoundError' in result['verification_error']
        assert result['verification_error']  # non-empty string

    @pytest.mark.asyncio
    async def test_update_edge_invalid_at_only_returns_verified_true_without_readback(
        self, service
    ):
        """When only invalid_at is supplied (no fact), verified must be True
        and get_edge_text must NOT be called.
        """
        from datetime import UTC, datetime

        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'unchanged', 'refreshed_nodes': []}
        )
        service.graphiti.get_edge_text = AsyncMock()

        ts = datetime(2026, 5, 4, tzinfo=UTC)
        result = await service.update_edge(
            edge_uuid='e-1', project_id='test', invalid_at=ts
        )

        assert result['verified'] is True
        service.graphiti.get_edge_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_edge_with_empty_string_fact_runs_verification(self, service):
        """fact='' is not None so get_edge_text must still be called and verified computed.

        Guards against an accidental future change from ``if fact is not None:``
        to ``if fact:`` which would silently skip verification for empty-string
        facts and regress the invalid_at-only bypass.
        """
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': '', 'refreshed_nodes': []}
        )
        service.graphiti.get_edge_text = AsyncMock(return_value=('Edge', ''))

        result = await service.update_edge(
            edge_uuid='e-1', fact='', project_id='test'
        )

        service.graphiti.get_edge_text.assert_called_once_with('e-1', group_id='test')
        assert result['verified'] is True

    @pytest.mark.asyncio
    async def test_update_edge_logs_verified_in_journal(self, service):
        """The verified field must appear in the result_summary logged to the write journal."""
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'new fact', 'refreshed_nodes': []}
        )
        service.graphiti.get_edge_text = AsyncMock(return_value=('Edge', 'new fact'))
        journal_mock = MagicMock()
        journal_mock.log_write_op = AsyncMock()
        journal_mock.log_backend_op = AsyncMock()
        service._write_journal = journal_mock

        # --- matching readback: verified=True ---
        await service.update_edge(edge_uuid='e-1', fact='new fact', project_id='test')

        call_args = journal_mock.log_write_op.call_args
        result_summary = call_args.kwargs.get('result_summary', {})
        assert result_summary.get('verified') is True
        assert result_summary.get('status') == 'updated'

        # --- mismatched readback: verified=False ---
        journal_mock.log_write_op.reset_mock()
        service.graphiti.get_edge_text = AsyncMock(return_value=('Edge', 'stale'))

        await service.update_edge(edge_uuid='e-1', fact='new fact', project_id='test')

        call_args = journal_mock.log_write_op.call_args
        result_summary = call_args.kwargs.get('result_summary', {})
        assert result_summary.get('verified') is False


class TestSearchDeleteRoundtrip:
    @pytest.mark.asyncio
    async def test_search_then_delete_graphiti_roundtrip(self, service):
        """End-to-end contract test: search returns edge UUIDs that work with delete_memory."""
        from _fm_helpers import MockEdge, MockNode

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
        service.graphiti.remove_edge.assert_called_once_with(edge_uuid, group_id='test')
        # Verify remove_episode was NOT called (edge != episode)
        service.graphiti.remove_episode.assert_not_called()


class TestDeleteEpisode:
    @pytest.mark.asyncio
    async def test_delete_episode_still_uses_remove_episode(self, service):
        """Regression guard: delete_episode must continue to call remove_episode
        (not the new remove_edge), since episodes are EpisodicNodes."""
        await service.delete_episode(episode_id='ep-uuid-123', project_id='test')
        service.graphiti.remove_episode.assert_called_once_with('ep-uuid-123', group_id='test')
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


class TestCloseLogsExceptions:
    """Tests that MemoryService.close() logs exceptions via logger.exception
    and continues closing remaining resources (failure isolation)."""

    def _make_resource_mocks(self, service):
        """Wire all six resources with AsyncMock close() methods and return a
        name-to-mock mapping for assertion convenience."""
        service.graphiti.close = AsyncMock()
        service.mem0.close = AsyncMock()

        mock_journal = MagicMock()
        mock_journal.close = AsyncMock()
        service._write_journal = mock_journal

        mock_buffer = MagicMock()
        mock_buffer.close = AsyncMock()
        service._event_buffer = mock_buffer

        mock_registry = MagicMock()
        mock_registry.close = AsyncMock()
        service.planned_episode_registry = mock_registry

        return {
            'durable_queue': service.durable_queue,
            'graphiti': service.graphiti,
            'mem0': service.mem0,
            '_write_journal': service._write_journal,
            '_event_buffer': service._event_buffer,
            'planned_episode_registry': service.planned_episode_registry,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'failing_resource,log_fragment',
        [
            ('durable_queue', 'durable_queue'),
            ('graphiti', 'graphiti'),
            ('mem0', 'mem0'),
            ('_write_journal', 'write_journal'),
            ('_event_buffer', 'event_buffer'),
            ('planned_episode_registry', 'planned_episode_registry'),
        ],
    )
    async def test_close_logs_exception_per_resource(
        self, service, caplog, failing_resource, log_fragment
    ):
        """Each resource failure is logged at ERROR (logger.exception) and
        does NOT prevent the remaining resources from being closed."""
        resources = self._make_resource_mocks(service)

        # Make ONLY the failing resource raise
        resources[failing_resource].close.side_effect = RuntimeError('boom')

        with caplog.at_level(logging.ERROR, logger='fused_memory.services.memory_service'):
            # Must NOT raise
            await service.close()

        # All six resources must have been awaited (failure isolation)
        for _name, resource in resources.items():
            resource.close.assert_awaited_once()

        # Exactly one ERROR record
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1, (
            f'Expected 1 ERROR record, got {len(error_records)}: '
            f'{[r.message for r in error_records]}'
        )
        record = error_records[0]
        assert 'MemoryService.close' in record.message
        assert f'{log_fragment}.close failed' in record.message
        # logger.exception sets exc_info — verify traceback is captured
        assert record.exc_info is not None, (
            'Expected exc_info to be populated (use logger.exception, not logger.error)'
        )

    @pytest.mark.asyncio
    async def test_close_does_not_raise_when_every_resource_fails(self, service, caplog):
        """When all six resources raise, close() logs six ERROR records and never re-raises."""
        resources = self._make_resource_mocks(service)

        # Wire ALL resources to fail with distinct exceptions
        resources['durable_queue'].close.side_effect = RuntimeError('dq-fail')
        resources['graphiti'].close.side_effect = ValueError('graphiti-fail')
        resources['mem0'].close.side_effect = OSError('mem0-fail')
        resources['_write_journal'].close.side_effect = RuntimeError('journal-fail')
        resources['_event_buffer'].close.side_effect = TimeoutError('buffer-fail')
        resources['planned_episode_registry'].close.side_effect = RuntimeError('registry-fail')

        with caplog.at_level(logging.ERROR, logger='fused_memory.services.memory_service'):
            # Must NOT raise even when every resource fails
            await service.close()

        # All six must have been attempted
        for _name, resource in resources.items():
            resource.close.assert_awaited_once()

        # Exactly six ERROR records
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 6, (
            f'Expected 6 ERROR records, got {len(error_records)}'
        )
        # Every record has exc_info (logger.exception, not logger.error)
        for record in error_records:
            assert record.exc_info is not None, (
                f'Expected exc_info on record: {record.message}'
            )


class TestGraphitiBackendRemoveEdge:
    @pytest.mark.asyncio
    async def test_graphiti_backend_remove_edge(self, mock_config):
        """Unit test for GraphitiBackend.remove_edge — should call
        EntityEdge.get_by_uuid then edge.delete."""
        from fused_memory.backends.graphiti_client import GraphitiBackend

        backend = GraphitiBackend(mock_config)
        # Provide a mock driver so _require_driver succeeds
        mock_driver = MagicMock()
        mock_cloned_driver = MagicMock()
        mock_driver.clone = MagicMock(return_value=mock_cloned_driver)
        backend._driver = mock_driver
        backend.client = MagicMock()

        mock_edge = AsyncMock()
        mock_edge.delete = AsyncMock()

        with patch(
            'fused_memory.backends.graphiti_client.EntityEdge'
        ) as MockEntityEdge:
            MockEntityEdge.get_by_uuid = AsyncMock(return_value=mock_edge)
            await backend.remove_edge('test-edge-uuid', group_id='test')

            MockEntityEdge.get_by_uuid.assert_called_once_with(
                mock_cloned_driver, 'test-edge-uuid'
            )
            mock_edge.delete.assert_called_once_with(mock_cloned_driver)


class TestMem0BackendAddInferContract:
    """Mem0Backend.add must pin infer=False so Mem0 stores distilled
    content verbatim instead of silently dropping it when its LLM
    fact-extractor finds no declarative facts (the Task-360/1040 bug).
    """

    @pytest.mark.asyncio
    async def test_add_passes_infer_false_to_async_memory(self, mock_config):
        from fused_memory.backends.mem0_client import Mem0Backend

        backend = Mem0Backend(mock_config)

        mock_instance = MagicMock()
        mock_instance.add = AsyncMock(
            return_value={'results': [{'id': 'mem0-new', 'event': 'ADD'}]}
        )
        backend._instances = {'autopilot_video': mock_instance}

        scope = Scope(project_id='autopilot_video', agent_id='t')
        result = await backend.add(
            content='ATTRIBUTION_STUB_GUARDRAIL: do not fabricate...',
            scope=scope,
            metadata={'category': 'preferences_and_norms'},
        )

        mock_instance.add.assert_awaited_once()
        call_kwargs = mock_instance.add.call_args.kwargs
        assert call_kwargs.get('infer') is False, (
            'Mem0Backend.add must call AsyncMemory.add with infer=False; '
            'otherwise the LLM fact-extractor silently drops '
            'normative/procedural content and returns empty results.'
        )
        assert result == {'results': [{'id': 'mem0-new', 'event': 'ADD'}]}


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


class TestGraphitiBackendClose:
    @pytest.mark.asyncio
    async def test_close_awaits_close_on_all_cloned_drivers(self, mock_config):
        """GraphitiBackend.close() must await close() on every cached cloned driver."""
        from fused_memory.backends.graphiti_client import GraphitiBackend

        backend = GraphitiBackend(mock_config)

        # Primary driver mock
        backend._driver = MagicMock()
        backend._driver.close = AsyncMock()

        # Two cloned-driver mocks
        clone_a = MagicMock()
        clone_a.close = AsyncMock()
        clone_b = MagicMock()
        clone_b.close = AsyncMock()
        backend._cloned_drivers = {'group-a': clone_a, 'group-b': clone_b}

        await backend.close()

        clone_a.close.assert_awaited_once()
        clone_b.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_clears_cloned_drivers_dict(self, mock_config):
        """GraphitiBackend.close() must clear _cloned_drivers after closing them."""
        from fused_memory.backends.graphiti_client import GraphitiBackend

        backend = GraphitiBackend(mock_config)

        # Primary driver mock
        backend._driver = MagicMock()
        backend._driver.close = AsyncMock()

        # Two cloned-driver mocks
        clone_a = MagicMock()
        clone_a.close = AsyncMock()
        clone_b = MagicMock()
        clone_b.close = AsyncMock()
        backend._cloned_drivers = {'group-a': clone_a, 'group-b': clone_b}

        await backend.close()

        assert backend._cloned_drivers == {}

    @pytest.mark.asyncio
    async def test_close_resilient_to_cloned_driver_close_error(self, mock_config):
        """A failing clone.close() must not prevent other clones or the primary from closing."""
        from fused_memory.backends.graphiti_client import GraphitiBackend

        backend = GraphitiBackend(mock_config)

        # Primary driver mock — save close reference before close() nulls _driver
        primary_driver = MagicMock()
        primary_close = AsyncMock()
        primary_driver.close = primary_close
        backend._driver = primary_driver

        # clone_a raises, clone_b should still be closed
        clone_a = MagicMock()
        clone_a.close = AsyncMock(side_effect=RuntimeError('boom'))
        clone_b = MagicMock()
        clone_b.close = AsyncMock()
        backend._cloned_drivers = {'group-a': clone_a, 'group-b': clone_b}

        # Must not raise
        await backend.close()

        # clone_b still closed despite clone_a raising
        clone_b.close.assert_awaited_once()
        # Primary driver still closed
        primary_close.assert_awaited_once()
        # Dict cleared
        assert backend._cloned_drivers == {}
        # Primary driver nulled out
        assert backend._driver is None


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

    @pytest.mark.asyncio
    async def test_non_utc_offset_normalized_to_canonical_utc_iso8601(self, service):
        """created_at with a non-UTC offset must be normalized, not str()-passthrough."""
        non_utc = datetime(2026, 5, 1, 9, 30, tzinfo=timezone(timedelta(hours=5)))
        mock_ep = MagicMock()
        mock_ep.uuid = 'ep-uuid-1'
        mock_ep.name = 'test episode'
        mock_ep.content = 'some content'
        mock_ep.created_at = non_utc
        mock_ep.source = None
        mock_ep.group_id = 'test'
        service.graphiti.retrieve_episodes = AsyncMock(return_value=[mock_ep])

        episodes = await service.get_episodes(project_id='test')

        created_at = episodes[0]['created_at']
        assert created_at == non_utc.astimezone(UTC).isoformat()
        assert 'T' in created_at
        assert created_at.endswith('+00:00')

    @pytest.mark.asyncio
    async def test_mixed_offsets_emit_lexically_non_increasing_created_at(self, service):
        """Emitted created_at strings must sort lexically iff the instants do.

        str(created_at) preserves each episode's stored UTC offset, so two
        episodes with the same *instant* ordering but different offsets can
        emit strings that compare out of order (an apparent non-monotonic
        result on the observed/serialized path). Constructing from explicit
        UTC instants and then converting to differing display offsets
        isolates that: the instants are non-increasing by construction, but
        the raw str() reprs (mismatched offsets) are not lexically
        non-increasing, while the canonical UTC ISO-8601 reprs are.
        """
        later_instant = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)
        earlier_instant = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
        ep1_created = later_instant.astimezone(timezone(timedelta(hours=-3)))
        ep2_created = earlier_instant.astimezone(timezone(timedelta(hours=5)))

        mock_ep1 = MagicMock()
        mock_ep1.uuid = 'ep-1'
        mock_ep1.name = 'e1'
        mock_ep1.content = 'c1'
        mock_ep1.created_at = ep1_created
        mock_ep1.source = None
        mock_ep1.group_id = 'test'

        mock_ep2 = MagicMock()
        mock_ep2.uuid = 'ep-2'
        mock_ep2.name = 'e2'
        mock_ep2.content = 'c2'
        mock_ep2.created_at = ep2_created
        mock_ep2.source = None
        mock_ep2.group_id = 'test'

        service.graphiti.retrieve_episodes = AsyncMock(return_value=[mock_ep1, mock_ep2])

        episodes = await service.get_episodes(project_id='test')

        created_ats = [ep['created_at'] for ep in episodes]
        assert created_ats == [
            later_instant.astimezone(UTC).isoformat(),
            earlier_instant.astimezone(UTC).isoformat(),
        ]
        assert created_ats == sorted(created_ats, reverse=True)


class TestGetEpisodeContent:
    """Tests for MemoryService.get_episode_content() (task 2033).

    Thin passthrough to graphiti.get_episode_by_uuid that extracts just the
    episode's content string. Backs the reconciliation promotion-time
    batch-plan gate, which needs the original episode text (not just edge
    fact text, which is all MemoryResult carries) to run
    is_batch_plan_framing on.
    """

    @pytest.mark.asyncio
    async def test_returns_episode_content_string(self, service):
        service.graphiti.get_episode_by_uuid = AsyncMock(
            return_value=types.SimpleNamespace(
                content='Merge-queue batch queued as df 1985-2002'
            )
        )

        result = await service.get_episode_content('ep-x', 'proj')

        assert result == 'Merge-queue batch queued as df 1985-2002'
        service.graphiti.get_episode_by_uuid.assert_awaited_once_with(
            'ep-x', group_id='proj'
        )

    @pytest.mark.asyncio
    async def test_missing_episode_returns_none(self, service):
        """When graphiti returns None (episode not found), return None rather than raising."""
        service.graphiti.get_episode_by_uuid = AsyncMock(return_value=None)

        result = await service.get_episode_content('ep-missing', 'proj')

        assert result is None


class TestGetEntity:
    # ------------------------------------------------------------------
    # baseline success regression test
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_success_returns_nodes_and_edges(self, service):
        """Baseline regression: get_entity returns correct node and edge data."""
        from _fm_helpers import MockEdge, MockNode

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

    @pytest.mark.asyncio
    async def test_cancelled_error_takes_precedence_over_exception(self, service):
        """CancelledError takes precedence over RuntimeError even when RuntimeError comes first.

        Scenario: search_nodes() raises RuntimeError('boom') — it is at results[0].
                  search() raises CancelledError — it is at results[1].

        Under the OLD code, `next((r for r in results if isinstance(r, BaseException)), None)`
        picks the first match by position: RuntimeError IS a BaseException subclass, so it
        wins and RuntimeError is raised.

        Under the NEW code (propagate_cancellations called first), the helper scans the
        full sequence for bare-BaseException (BaseException but NOT Exception). RuntimeError
        IS an Exception so it is skipped; CancelledError is not an Exception, so it is raised.

        This aligns get_entity with the convention in graphiti_client.rebuild_entity_summaries
        and context_assembler.assemble where cancellation signals always take precedence over
        per-call application errors — structured concurrency semantics.
        """
        service.graphiti.search_nodes = AsyncMock(
            side_effect=RuntimeError('boom')
        )
        service.graphiti.search = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        with patch('fused_memory.services.memory_service.logger') as mock_logger, \
             pytest.raises(asyncio.CancelledError):
            await service.get_entity('entity', project_id='test')

        # Cancellation propagates before the per-call warning loop executes.
        assert mock_logger.warning.call_count == 0

    # ------------------------------------------------------------------
    # temporal serialization in edge_data
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_edges_include_temporal(self, service):
        """Edge dicts in get_entity result include a 'temporal' key with ISO 8601 strings."""
        from _fm_helpers import MockEdge

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
        from _fm_helpers import MockEdge

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

    # ------------------------------------------------------------------
    # exact-first lookup (task-1975): get_nodes_by_exact_name precedes fuzzy search_nodes
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_exact_match_preferred_over_fuzzy(self, service):
        """An exact-name hit is returned as-is; fuzzy search_nodes is skipped entirely."""
        from _fm_helpers import MockNode

        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=[{'uuid': 'u-115', 'name': 'Task 115', 'summary': 's', 'labels': []}]
        )
        service.graphiti.search_nodes = AsyncMock(
            return_value=[MockNode(name='Task 95', uuid='u-95')]
        )
        service.graphiti.search = AsyncMock(return_value=[])

        result = await service.get_entity('Task 115', project_id='test')

        assert len(result['nodes']) == 1
        assert result['nodes'][0]['name'] == 'Task 115'
        assert result['nodes'][0]['uuid'] == 'u-115'
        service.graphiti.search_nodes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_exact_match_falls_back_to_fuzzy(self, service):
        """An empty exact-name result falls through to the existing fuzzy search_nodes path."""
        from _fm_helpers import MockNode

        service.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[])
        service.graphiti.search_nodes = AsyncMock(
            return_value=[MockNode(name='Approx', uuid='u-1')]
        )
        service.graphiti.search = AsyncMock(return_value=[])

        result = await service.get_entity('Approx', project_id='test')

        assert result['nodes'][0]['name'] == 'Approx'
        service.graphiti.search_nodes.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exact_match_multiple_returns_all(self, service):
        """Multiple exact-name matches are all returned (no ambiguity error)."""
        service.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[
            {'uuid': 'u-1', 'name': 'Alice', 'summary': 's1', 'labels': []},
            {'uuid': 'u-2', 'name': 'Alice', 'summary': 's2', 'labels': []},
        ])
        service.graphiti.search = AsyncMock(return_value=[])

        result = await service.get_entity('Alice', project_id='test')

        assert len(result['nodes']) == 2
        assert result['nodes'][0]['name'] == 'Alice'
        assert result['nodes'][1]['name'] == 'Alice'
        assert result['nodes'][0]['uuid'] == 'u-1'
        assert result['nodes'][1]['uuid'] == 'u-2'

    @pytest.mark.asyncio
    async def test_exact_match_edges_come_from_resolved_node_uuid(self, service):
        """On an exact node hit, edges come from the resolved node's uuid via
        get_valid_edges_for_node — NOT from a semantic graphiti.search() fact
        search, which can return edges belonging to unrelated nodes.
        """
        from _fm_helpers import MockEdge

        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=[{'uuid': 'u-138', 'name': 'Task 138', 'summary': 's', 'labels': []}]
        )
        service.graphiti.get_valid_edges_for_node = AsyncMock(
            return_value=[
                {'uuid': 'e-own', 'fact': 'Task 138 depends on Task 5', 'name': 'DEPENDS_ON'}
            ]
        )
        # Decoy: if the (buggy) semantic search were still consulted, this
        # unrelated edge would leak into the result.
        service.graphiti.search = AsyncMock(
            return_value=[MockEdge(fact='Task 68 depends on Task 12', uuid='e-wrong-68')]
        )

        result = await service.get_entity('Task 138', project_id='test')

        assert len(result['edges']) == 1
        assert result['edges'][0]['uuid'] == 'e-own'
        assert result['edges'][0]['fact'] == 'Task 138 depends on Task 5'
        assert all(e['uuid'] != 'e-wrong-68' for e in result['edges'])
        service.graphiti.get_valid_edges_for_node.assert_awaited_once_with('u-138', group_id='test')
        service.graphiti.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exact_match_edges_union_across_duplicate_nodes(self, service):
        """When multiple nodes share an exact name (duplicate-name pathology),
        edges are fetched for EVERY resolved uuid and unioned, deduped by edge
        uuid, so `edges` stays consistent with the full `nodes` array.
        """
        service.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[
            {'uuid': 'u-1', 'name': 'Alice', 'summary': 's1', 'labels': []},
            {'uuid': 'u-2', 'name': 'Alice', 'summary': 's2', 'labels': []},
        ])

        edges_by_uuid = {
            'u-1': [
                {'uuid': 'e-a', 'fact': 'A', 'name': 'x'},
                {'uuid': 'e-shared', 'fact': 'S', 'name': 'y'},
            ],
            'u-2': [
                {'uuid': 'e-shared', 'fact': 'S', 'name': 'y'},
                {'uuid': 'e-b', 'fact': 'B', 'name': 'z'},
            ],
        }

        def fake_edges(node_uuid, *, group_id):
            return edges_by_uuid[node_uuid]

        service.graphiti.get_valid_edges_for_node = AsyncMock(side_effect=fake_edges)

        result = await service.get_entity('Alice', project_id='test')

        assert [e['uuid'] for e in result['edges']] == ['e-a', 'e-shared', 'e-b']
        service.graphiti.get_valid_edges_for_node.assert_any_await('u-1', group_id='test')
        service.graphiti.get_valid_edges_for_node.assert_any_await('u-2', group_id='test')
        assert service.graphiti.get_valid_edges_for_node.await_count == 2

    # ------------------------------------------------------------------
    # exact-match edge fetch failure — mirrors the fuzzy fallback's two-tier
    # gather(return_exceptions=True) discipline (task 2102 amendment)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_exact_match_edge_fetch_failure_raises_and_settles_siblings(self, service):
        """When one concurrent get_valid_edges_for_node call fails, the exception
        propagates AND the sibling call still settles (return_exceptions=True)
        rather than being left running as an orphaned background task — the same
        discipline the fuzzy fallback already has (see
        test_both_failures_raises_first_exception / test_both_failures_emits_two_warnings).
        """
        service.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[
            {'uuid': 'u-1', 'name': 'Alice', 'summary': 's1', 'labels': []},
            {'uuid': 'u-2', 'name': 'Alice', 'summary': 's2', 'labels': []},
        ])

        async def fake_edges(node_uuid, *, group_id):
            if node_uuid == 'u-1':
                raise RuntimeError('u-1 failed')
            return [{'uuid': 'e-b', 'fact': 'B', 'name': 'z'}]

        service.graphiti.get_valid_edges_for_node = AsyncMock(side_effect=fake_edges)

        with patch('fused_memory.services.memory_service.logger') as mock_logger, \
             pytest.raises(RuntimeError, match='u-1 failed'):
            await service.get_entity('Alice', project_id='test')

        # Both coroutines settled under gather(return_exceptions=True) even
        # though one raised — no orphaned background task for 'u-2'.
        assert service.graphiti.get_valid_edges_for_node.await_count == 2
        assert mock_logger.warning.call_count == 1

    @pytest.mark.asyncio
    async def test_exact_match_edge_fetch_cancelled_error_propagates(self, service):
        """asyncio.CancelledError from get_valid_edges_for_node must propagate
        unchanged and must NOT be logged via the isinstance(r, Exception) guard —
        same structured-concurrency precedence as the fuzzy fallback's
        test_cancelled_error_propagates.
        """
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=[{'uuid': 'u-138', 'name': 'Task 138', 'summary': 's', 'labels': []}]
        )
        service.graphiti.get_valid_edges_for_node = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        with patch('fused_memory.services.memory_service.logger') as mock_logger, \
             pytest.raises(asyncio.CancelledError):
            await service.get_entity('Task 138', project_id='test')

        assert mock_logger.warning.call_count == 0

    @pytest.mark.asyncio
    async def test_exact_match_labels_default_empty(self, service):
        """An exact node dict with no/None labels yields labels == [] in the result."""
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=[{'uuid': 'u-115', 'name': 'Task 115', 'summary': 's', 'labels': None}]
        )
        service.graphiti.search = AsyncMock(return_value=[])

        result = await service.get_entity('Task 115', project_id='test')

        assert result['nodes'][0]['labels'] == []

    # ------------------------------------------------------------------
    # edge_limit parameter (task 2089): threads to graphiti.search num_results
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_edge_limit_threads_to_num_results_fuzzy_path(self, service):
        """edge_limit is forwarded as num_results to graphiti.search on the fuzzy-fallback path."""
        service.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[])
        service.graphiti.search_nodes = AsyncMock(return_value=[])
        service.graphiti.search = AsyncMock(return_value=[])

        await service.get_entity('entity', project_id='test', edge_limit=25)

        service.graphiti.search.assert_awaited_once()
        assert service.graphiti.search.call_args.kwargs['num_results'] == 25

    @pytest.mark.asyncio
    async def test_edge_limit_does_not_reach_search_on_exact_path(self, service):
        """edge_limit is a fuzzy-path-only cap. On the exact-match path (task 2102)
        edges come from get_valid_edges_for_node(uuid) — an uncapped uuid traversal —
        so graphiti.search is never consulted and edge_limit does not thread anywhere.
        Regression guard: an exact hit must not fall back to the semantic edge search
        that could return edges belonging to unrelated nodes.
        """
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=[{'uuid': 'u-115', 'name': 'Task 115', 'summary': 's', 'labels': []}]
        )
        service.graphiti.get_valid_edges_for_node = AsyncMock(return_value=[])
        service.graphiti.search = AsyncMock(return_value=[])

        await service.get_entity('Task 115', project_id='test', edge_limit=25)

        service.graphiti.search.assert_not_awaited()
        service.graphiti.get_valid_edges_for_node.assert_awaited_once_with('u-115', group_id='test')

    @pytest.mark.asyncio
    async def test_edge_limit_defaults_to_ten(self, service):
        """Without an explicit edge_limit, num_results stays at 10 (regression guard)."""
        service.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[])
        service.graphiti.search_nodes = AsyncMock(return_value=[])
        service.graphiti.search = AsyncMock(return_value=[])

        await service.get_entity('entity', project_id='test')

        service.graphiti.search.assert_awaited_once()
        assert service.graphiti.search.call_args.kwargs['num_results'] == 10


class TestEdgeToDict:
    """Unit tests for the module-level _edge_to_dict helper."""

    def test_edge_to_dict_normalizes_dict_shape(self):
        """A plain EdgeDict (from get_valid_edges_for_node) normalizes uuid/fact,
        drops 'name', and yields temporal=None (EdgeDict carries no valid_at/invalid_at).
        """
        result = memory_service._edge_to_dict({'uuid': 'e-1', 'fact': 'F', 'name': 'N'})
        assert result == {'uuid': 'e-1', 'fact': 'F', 'temporal': None}

    def test_edge_to_dict_dict_shape_with_temporal_serializes(self):
        """A dict input that DOES carry valid_at/invalid_at still serializes them via
        _serialize_temporal, locking in the docstring's forward-compatibility claim
        that the dict branch stays correct if EdgeDict is later enriched with
        temporal fields.
        """
        dt_valid = datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)
        result = memory_service._edge_to_dict(
            {'uuid': 'e-1', 'fact': 'F', 'valid_at': dt_valid, 'invalid_at': None}
        )
        assert result == {
            'uuid': 'e-1',
            'fact': 'F',
            'temporal': {'valid_at': '2024-03-01T10:00:00+00:00', 'invalid_at': None},
        }


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

        from _fm_helpers import MockAddEpisodeResult, MockEdge

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

        removed = await service._dedup_episode_edges(result, group_id='test')

        assert removed == 2
        service.graphiti.bulk_remove_edges.assert_called_once()
        deleted_uuids = service.graphiti.bulk_remove_edges.call_args[0][0]
        assert sorted(deleted_uuids) == ['uuid-2', 'uuid-3']

    @pytest.mark.asyncio
    async def test_no_duplicates_returns_zero_no_backend_call(self, service):
        """step-3: all edges distinct → 0 removed, bulk_remove_edges NOT called."""
        from unittest.mock import AsyncMock

        from _fm_helpers import MockAddEpisodeResult, MockEdge

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

        removed = await service._dedup_episode_edges(result, group_id='test')

        assert removed == 0
        service.graphiti.bulk_remove_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_result_returns_zero(self, service):
        """step-5a: None result → 0, no backend calls."""
        from unittest.mock import AsyncMock

        service.graphiti.bulk_remove_edges = AsyncMock(return_value=0)

        removed = await service._dedup_episode_edges(None, group_id='test')

        assert removed == 0
        service.graphiti.bulk_remove_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_edges_list_returns_zero(self, service):
        """step-5b: result with empty edges list → 0, no backend calls."""
        from unittest.mock import AsyncMock

        from _fm_helpers import MockAddEpisodeResult

        service.graphiti.bulk_remove_edges = AsyncMock(return_value=0)

        result = MockAddEpisodeResult(edges=[])
        result.entity_edges = []

        removed = await service._dedup_episode_edges(result, group_id='test')

        assert removed == 0
        service.graphiti.bulk_remove_edges.assert_not_called()

    @pytest.mark.asyncio
    async def test_normalization_treats_case_whitespace_variants_as_duplicate(self, service):
        """step-6: case + whitespace normalization catches near-duplicate facts."""
        from unittest.mock import AsyncMock

        from _fm_helpers import MockAddEpisodeResult, MockEdge

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

        removed = await service._dedup_episode_edges(result, group_id='test')

        assert removed == 1
        service.graphiti.bulk_remove_edges.assert_called_once()
        deleted_uuids = service.graphiti.bulk_remove_edges.call_args[0][0]
        assert deleted_uuids == ['uuid-2']

    @pytest.mark.asyncio
    async def test_same_fact_different_node_pairs_not_duplicates(self, service):
        """step-7: same fact text but different source/target pairs → distinct edges."""
        from unittest.mock import AsyncMock

        from _fm_helpers import MockAddEpisodeResult, MockEdge

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

        removed = await service._dedup_episode_edges(result, group_id='test')

        assert removed == 0
        service.graphiti.bulk_remove_edges.assert_not_called()


# ---------------------------------------------------------------------------
# task 2073 step-3: MemoryService._dedup_episode_nodes
# ---------------------------------------------------------------------------

class TestDedupEpisodeNodes:
    """Unit tests for MemoryService._dedup_episode_nodes — the post-write
    exact-name node-dedup sweep that merges duplicate entity nodes
    graphiti_core's ingestion-time resolution failed to catch (task 2073).

    step-3: core behavior — merges deprecated dups into the canonical
            survivor, de-dupes repeated names to a single lookup, no-ops on
            None/empty/<2 matches, skips nodes with empty/missing name.

    step-5: best-effort resilience — a merge_entities failure for one
            duplicate/name does not propagate and does not stop subsequent
            duplicates/names from being processed.
    """

    @pytest.mark.asyncio
    async def test_merges_deprecated_into_canonical_survivor(self, service):
        """3 canonical-ordered dups -> 2 merge_entities calls (dep, canon), returns 2."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'canon', 'created_at': 100, 'edge_count': 5},
            {'uuid': 'd1', 'created_at': 200, 'edge_count': 2},
            {'uuid': 'd2', 'created_at': 300, 'edge_count': 1},
        ])
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[
            MockNode(name='orchestrator-reify.service'),
            MockNode(name='orchestrator-reify.service'),
        ])

        merged = await service._dedup_episode_nodes(result, group_id='test')

        assert merged == 2
        assert service.graphiti.merge_entities.await_count == 2
        calls = service.graphiti.merge_entities.call_args_list
        assert calls[0][0] == ('d1', 'canon')
        assert calls[0][1] == {'group_id': 'test'}
        assert calls[1][0] == ('d2', 'canon')
        assert calls[1][1] == {'group_id': 'test'}

    @pytest.mark.asyncio
    async def test_repeated_name_dedupes_lookup_to_one_call(self, service):
        """Distinct names are collected; find_duplicate_entity_nodes awaited once per name."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'canon', 'created_at': 100, 'edge_count': 5},
            {'uuid': 'd1', 'created_at': 200, 'edge_count': 2},
        ])
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[
            MockNode(name='orchestrator-reify.service'),
            MockNode(name='orchestrator-reify.service'),
            MockNode(name='orchestrator-reify.service'),
        ])

        await service._dedup_episode_nodes(result, group_id='test')

        service.graphiti.find_duplicate_entity_nodes.assert_awaited_once_with(
            'orchestrator-reify.service', group_id='test',
        )

    @pytest.mark.asyncio
    async def test_none_result_returns_zero(self, service):
        """None result -> 0, no backend calls."""
        service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        service.graphiti.merge_entities = AsyncMock(return_value={})

        merged = await service._dedup_episode_nodes(None, group_id='test')

        assert merged == 0
        service.graphiti.find_duplicate_entity_nodes.assert_not_awaited()
        service.graphiti.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_nodes_returns_zero(self, service):
        """Empty result.nodes -> 0, no backend calls."""
        from _fm_helpers import MockAddEpisodeResult

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[])
        merged = await service._dedup_episode_nodes(result, group_id='test')

        assert merged == 0
        service.graphiti.find_duplicate_entity_nodes.assert_not_awaited()
        service.graphiti.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_zero_matches_is_noop(self, service):
        """find_duplicate_entity_nodes returns [] -> no merge, returns 0."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[MockNode(name='Solo')])
        merged = await service._dedup_episode_nodes(result, group_id='test')

        assert merged == 0
        service.graphiti.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_match_is_noop(self, service):
        """find_duplicate_entity_nodes returns exactly 1 match -> no merge, returns 0."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'only', 'created_at': 100, 'edge_count': 0},
        ])
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[MockNode(name='Solo')])
        merged = await service._dedup_episode_nodes(result, group_id='test')

        assert merged == 0
        service.graphiti.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_nodes_with_empty_or_missing_name(self, service):
        """Nodes with falsy or entirely absent 'name' are excluded from lookup."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[
            MockNode(name=''),
            MockNode(name=None),  # type: ignore[arg-type]  # deliberately invalid: exercises falsy-name skip
            types.SimpleNamespace(),  # type: ignore[arg-type]  # no .name attribute at all
        ])
        merged = await service._dedup_episode_nodes(result, group_id='test')

        assert merged == 0
        service.graphiti.find_duplicate_entity_nodes.assert_not_awaited()

    # step-5: best-effort resilience

    @pytest.mark.asyncio
    async def test_merge_failure_is_swallowed_and_second_merge_still_attempted(
        self, service, caplog,
    ):
        """A merge_entities failure for the first duplicate is swallowed; the
        second duplicate is still attempted; only successful merges count."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'canon', 'created_at': 100, 'edge_count': 5},
            {'uuid': 'd1', 'created_at': 200, 'edge_count': 2},
            {'uuid': 'd2', 'created_at': 300, 'edge_count': 1},
        ])
        service.graphiti.merge_entities = AsyncMock(
            side_effect=[RuntimeError('lock contention'), {}],
        )

        result = MockAddEpisodeResult(nodes=[MockNode(name='orchestrator-reify.service')])

        with caplog.at_level(logging.ERROR, logger='fused_memory.services.memory_service'):
            merged = await service._dedup_episode_nodes(result, group_id='test')

        assert merged == 1, 'Only the second (successful) merge should count'
        assert service.graphiti.merge_entities.await_count == 2, (
            'The second duplicate must still be attempted after the first fails'
        )
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records, 'Expected an exception/error log for the failed merge'

    @pytest.mark.asyncio
    async def test_merge_failure_for_one_name_does_not_block_next_name(
        self, service, caplog,
    ):
        """When the first name's merge raises, the second name is still processed."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        async def fake_find_duplicates(name, *, group_id):
            if name == 'ServiceA':
                return [
                    {'uuid': 'canon-a', 'created_at': 100, 'edge_count': 5},
                    {'uuid': 'dup-a', 'created_at': 200, 'edge_count': 1},
                ]
            return [
                {'uuid': 'canon-b', 'created_at': 100, 'edge_count': 5},
                {'uuid': 'dup-b', 'created_at': 200, 'edge_count': 1},
            ]

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(side_effect=fake_find_duplicates)

        async def fake_merge(dep_uuid, survivor, *, group_id):
            if dep_uuid == 'dup-a':
                raise RuntimeError('transient write timeout')
            return {}

        service.graphiti.merge_entities = AsyncMock(side_effect=fake_merge)

        result = MockAddEpisodeResult(nodes=[
            MockNode(name='ServiceA'),
            MockNode(name='ServiceB'),
        ])

        with caplog.at_level(logging.ERROR, logger='fused_memory.services.memory_service'):
            merged = await service._dedup_episode_nodes(result, group_id='test')

        assert merged == 1, 'Only the ServiceB duplicate merge should succeed'
        assert service.graphiti.find_duplicate_entity_nodes.await_count == 2, (
            'Both names must be looked up even though the first name\'s merge failed'
        )
        calls = [c[0][0] for c in service.graphiti.merge_entities.call_args_list]
        assert calls == ['dup-a', 'dup-b'], 'Both duplicate merges must be attempted'


# ---------------------------------------------------------------------------
# task 2110 step-5: MemoryService._normalize_task_node_names
# ---------------------------------------------------------------------------

class TestNormalizeTaskNodeNames:
    """Unit tests for MemoryService._normalize_task_node_names — the post-write
    hook that canonicalizes non-canonical task-entity node names (e.g.
    'task 132', 'tasks 153') minted by graphiti_core's LLM extraction to the
    canonical 'Task N' form (task 2110).

    Collision policy: when a canonical 'Task N' node already exists, the
    bad-named node is MERGED into it; only when no canonical node exists is
    the bad-named survivor RENAMED (and any remaining bad-named duplicates
    merged into it).
    """

    @pytest.mark.asyncio
    async def test_renames_when_no_canonical_exists(self, service):
        """(a) A single 'task 132' node with no existing 'Task 132' canonical
        is renamed in place — merge_entities is never called."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        async def fake_find_duplicates(name, *, group_id):
            if name == 'task 132':
                return [{'uuid': 'survivor', 'created_at': 100, 'edge_count': 3}]
            return []  # 'Task 132' has no canonical match

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(side_effect=fake_find_duplicates)
        service.graphiti.rename_entity_node = AsyncMock(return_value={})
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[MockNode(name='task 132')])
        count = await service._normalize_task_node_names(result, group_id='test')

        assert count == 1
        service.graphiti.rename_entity_node.assert_awaited_once_with(
            'survivor', 'Task 132', group_id='test',
        )
        service.graphiti.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_merges_into_existing_canonical(self, service):
        """(b) A 'tasks 153' node with an existing 'Task 153' canonical is
        merged into the canonical survivor — rename_entity_node is never called."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        async def fake_find_duplicates(name, *, group_id):
            if name == 'tasks 153':
                return [{'uuid': 'bad-uuid', 'created_at': 100, 'edge_count': 1}]
            if name == 'Task 153':
                return [{'uuid': 'canon-uuid', 'created_at': 50, 'edge_count': 5}]
            return []

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(side_effect=fake_find_duplicates)
        service.graphiti.rename_entity_node = AsyncMock(return_value={})
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[MockNode(name='tasks 153')])
        count = await service._normalize_task_node_names(result, group_id='test')

        assert count == 1
        service.graphiti.merge_entities.assert_awaited_once_with(
            'bad-uuid', 'canon-uuid', group_id='test',
        )
        service.graphiti.rename_entity_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_renames_survivor_and_merges_remaining_dups_when_no_canonical(self, service):
        """(c) Two 'task 132' duplicates with no canonical -> the survivor is
        renamed and the remaining duplicate is merged into it."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        async def fake_find_duplicates(name, *, group_id):
            if name == 'task 132':
                return [
                    {'uuid': 'survivor', 'created_at': 100, 'edge_count': 5},
                    {'uuid': 'dup-1', 'created_at': 200, 'edge_count': 1},
                ]
            return []  # 'Task 132' has no canonical match

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(side_effect=fake_find_duplicates)
        service.graphiti.rename_entity_node = AsyncMock(return_value={})
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[MockNode(name='task 132')])
        count = await service._normalize_task_node_names(result, group_id='test')

        assert count == 2
        service.graphiti.rename_entity_node.assert_awaited_once_with(
            'survivor', 'Task 132', group_id='test',
        )
        service.graphiti.merge_entities.assert_awaited_once_with(
            'dup-1', 'survivor', group_id='test',
        )

    @pytest.mark.asyncio
    async def test_non_task_node_name_is_untouched(self, service):
        """(d) A non-task entity name ('Alice') is never looked up or mutated."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        service.graphiti.rename_entity_node = AsyncMock(return_value={})
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[MockNode(name='Alice')])
        count = await service._normalize_task_node_names(result, group_id='test')

        assert count == 0
        service.graphiti.find_duplicate_entity_nodes.assert_not_awaited()
        service.graphiti.rename_entity_node.assert_not_awaited()
        service.graphiti.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_already_canonical_name_is_noop(self, service):
        """(e) An already-canonical name ('Task 42') is a no-op — canonicalize
        maps it to itself, so the canonical!=name guard skips it entirely."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        service.graphiti.rename_entity_node = AsyncMock(return_value={})
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[MockNode(name='Task 42')])
        count = await service._normalize_task_node_names(result, group_id='test')

        assert count == 0
        service.graphiti.find_duplicate_entity_nodes.assert_not_awaited()
        service.graphiti.rename_entity_node.assert_not_awaited()
        service.graphiti.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_none_result_returns_zero(self, service):
        """(f) None result -> 0, no backend calls."""
        service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        service.graphiti.rename_entity_node = AsyncMock(return_value={})
        service.graphiti.merge_entities = AsyncMock(return_value={})

        count = await service._normalize_task_node_names(None, group_id='test')

        assert count == 0
        service.graphiti.find_duplicate_entity_nodes.assert_not_awaited()
        service.graphiti.rename_entity_node.assert_not_awaited()
        service.graphiti.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_nodes_returns_zero(self, service):
        """(f) Empty result.nodes -> 0, no backend calls."""
        from _fm_helpers import MockAddEpisodeResult

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        service.graphiti.rename_entity_node = AsyncMock(return_value={})
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[])
        count = await service._normalize_task_node_names(result, group_id='test')

        assert count == 0
        service.graphiti.find_duplicate_entity_nodes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rename_failure_is_swallowed_and_other_names_still_processed(
        self, service, caplog,
    ):
        """(g) A rename_entity_node failure for one bad name does not propagate
        and does not stop a second bad name from being processed."""
        from _fm_helpers import MockAddEpisodeResult, MockNode

        async def fake_find_duplicates(name, *, group_id):
            if name in ('task 132', 'Task 132'):
                return [{'uuid': 'survivor-a', 'created_at': 100, 'edge_count': 1}] \
                    if name == 'task 132' else []
            if name in ('tasks 200', 'Task 200'):
                return [{'uuid': 'survivor-b', 'created_at': 100, 'edge_count': 1}] \
                    if name == 'tasks 200' else []
            return []

        service.graphiti.find_duplicate_entity_nodes = AsyncMock(side_effect=fake_find_duplicates)

        async def fake_rename(node_uuid, new_name, *, group_id):
            if node_uuid == 'survivor-a':
                raise RuntimeError('transient write timeout')
            return {}

        service.graphiti.rename_entity_node = AsyncMock(side_effect=fake_rename)
        service.graphiti.merge_entities = AsyncMock(return_value={})

        result = MockAddEpisodeResult(nodes=[
            MockNode(name='task 132'),
            MockNode(name='tasks 200'),
        ])

        with caplog.at_level(logging.ERROR, logger='fused_memory.services.memory_service'):
            count = await service._normalize_task_node_names(result, group_id='test')

        assert count == 1, 'Only the second (successful) rename should count'
        assert service.graphiti.rename_entity_node.await_count == 2, (
            'The second name must still be attempted after the first fails'
        )
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records, 'Expected an exception/error log for the failed rename'


# ---------------------------------------------------------------------------
# Tests for _execute_graphiti_write integration with dedup  (steps 8, 10)
# ---------------------------------------------------------------------------

class TestExecuteGraphitiWriteWithDedup:
    """Integration tests for dedup wiring in _execute_graphiti_write.

    step-8: _execute_graphiti_write calls _dedup_episode_edges with result,
            duplicate edges are removed before returning

    step-10: _execute_graphiti_write returns normally when add_episode
             returns None (no edges to dedup, no crash)

    task 2073 step-7: _execute_graphiti_write also invokes the node-dedup
             sweep (_dedup_episode_nodes) with the add_episode result and
             group_id, alongside the existing edge-dedup/restore sweeps.
    """

    @pytest.mark.asyncio
    async def test_execute_graphiti_write_calls_dedup_and_removes_duplicates(self, service):
        """step-8: dedup is called after add_episode; duplicates are removed."""
        from unittest.mock import AsyncMock

        from _fm_helpers import MockAddEpisodeResult, MockEdge

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

    @pytest.mark.asyncio
    async def test_execute_graphiti_write_calls_restore_hook_for_invalidated_dep_edges(self, service):
        """_execute_graphiti_write invokes the restore hook when add_episode returns
        an invalidated dependency edge — guards against the hook being accidentally removed.
        """
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock

        from _fm_helpers import MockAddEpisodeResult, MockEdge

        ts = datetime(2026, 1, 1, tzinfo=UTC)
        dep_edge = MockEdge(
            fact='Task 562 depends on Task 557',
            uuid='dep-edge-1',
            source_node_uuid='s1',
            target_node_uuid='t1',
        )
        dep_edge.invalid_at = ts  # falsely superseded

        mock_result = MockAddEpisodeResult(edges=[dep_edge])
        mock_result.entity_edges = []

        service.graphiti.add_episode = AsyncMock(return_value=mock_result)
        service.graphiti.bulk_remove_edges = AsyncMock(return_value=0)
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'dep-edge-1', 'fact': 'Task 562 depends on Task 557',
                          'refreshed_nodes': []}
        )

        payload = {
            'name': 'ep_restore',
            'content': 'Task 562 depends on Task 557',
            'source': 'text',
            'group_id': 'test',
            'source_description': '',
        }
        await service._execute_graphiti_write('add_episode', payload)

        # The restore hook must have cleared the false invalidation
        service.graphiti.update_edge.assert_called_once_with(
            'dep-edge-1', group_id='test', clear_invalid_at=True
        )

    @pytest.mark.asyncio
    async def test_execute_graphiti_write_calls_node_dedup_sweep(self, service):
        """task 2073 step-7: the node-dedup sweep is invoked with the
        add_episode result and group_id, alongside the edge-dedup/restore
        sweeps already wired into _execute_graphiti_write."""
        from unittest.mock import AsyncMock

        from _fm_helpers import MockAddEpisodeResult, MockNode

        mock_result = MockAddEpisodeResult(nodes=[MockNode(name='Reify')])

        service.graphiti.add_episode = AsyncMock(return_value=mock_result)
        service.graphiti.bulk_remove_edges = AsyncMock(return_value=0)
        service._dedup_episode_nodes = AsyncMock(return_value=0)

        payload = {
            'name': 'ep_test',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': '',
        }
        await service._execute_graphiti_write('add_episode', payload)

        service._dedup_episode_nodes.assert_awaited_once()
        assert service._dedup_episode_nodes.call_args == ((mock_result,), {'group_id': 'test'})

    @pytest.mark.asyncio
    async def test_execute_graphiti_write_calls_normalize_task_node_names(self, service):
        """task 2110 step-7: the task-node-name normalization hook is invoked
        with the add_episode result and group_id, alongside the edge-dedup/
        restore/node-dedup sweeps already wired into _execute_graphiti_write."""
        from unittest.mock import AsyncMock

        from _fm_helpers import MockAddEpisodeResult, MockNode

        mock_result = MockAddEpisodeResult(nodes=[MockNode(name='task 132')])

        service.graphiti.add_episode = AsyncMock(return_value=mock_result)
        service.graphiti.bulk_remove_edges = AsyncMock(return_value=0)
        service._dedup_episode_nodes = AsyncMock(return_value=0)
        service._normalize_task_node_names = AsyncMock(return_value=0)

        payload = {
            'name': 'ep_test',
            'content': 'test content',
            'source': 'text',
            'group_id': 'test',
            'source_description': '',
        }
        await service._execute_graphiti_write('add_episode', payload)

        service._normalize_task_node_names.assert_awaited_once()
        assert service._normalize_task_node_names.call_args == ((mock_result,), {'group_id': 'test'})


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
        from _fm_helpers import MockAddEpisodeResult, MockEdge

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


# ---------------------------------------------------------------------------
# Tests for _refresh_entity_summaries_from_result (task 1949, fix a)
# ---------------------------------------------------------------------------


class TestRefreshEntitySummariesFromResult:
    """step-01/02/03/04: helper that refreshes summaries for edge endpoints
    touched by an add_episode/add_memory_graphiti result."""

    @pytest.mark.asyncio
    async def test_dedups_and_skips_empty_uuids(self, service):
        """Refreshes each unique non-empty edge endpoint uuid exactly once."""
        from _fm_helpers import MockAddEpisodeResult, MockEdge

        service.graphiti.refresh_entity_summary = AsyncMock(return_value={})

        result = MockAddEpisodeResult(edges=[
            MockEdge(fact='a', source_node_uuid='n1', target_node_uuid='n2'),
            MockEdge(fact='b', source_node_uuid='n2', target_node_uuid='n3'),
            MockEdge(fact='c', source_node_uuid='', target_node_uuid=''),
        ])

        await service._refresh_entity_summaries_from_result(result, group_id='proj')

        assert service.graphiti.refresh_entity_summary.await_count == 3, (
            'Expected exactly 3 refresh calls (n1, n2, n3 — n2 deduped, empty uuids skipped), '
            f'got {service.graphiti.refresh_entity_summary.await_count}'
        )
        called_uuids = {
            call.args[0] for call in service.graphiti.refresh_entity_summary.call_args_list
        }
        assert called_uuids == {'n1', 'n2', 'n3'}
        for call in service.graphiti.refresh_entity_summary.call_args_list:
            assert call.kwargs['group_id'] == 'proj'

    @pytest.mark.asyncio
    async def test_best_effort_continues_past_failure(self, service, caplog):
        """A refresh failure for one uuid must not stop refreshes for the rest."""
        from _fm_helpers import MockAddEpisodeResult, MockEdge

        async def _raise_for_n1(uuid, *, group_id):
            if uuid == 'n1':
                raise RuntimeError('boom')
            return {}

        service.graphiti.refresh_entity_summary = AsyncMock(side_effect=_raise_for_n1)

        result = MockAddEpisodeResult(edges=[
            MockEdge(fact='a', source_node_uuid='n1', target_node_uuid='n2'),
            MockEdge(fact='b', source_node_uuid='n2', target_node_uuid='n3'),
        ])

        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            # Must NOT raise despite n1's refresh failing
            await service._refresh_entity_summaries_from_result(result, group_id='proj')

        assert service.graphiti.refresh_entity_summary.await_count >= 3, (
            'Expected refresh to be attempted for n1 (fails), n2, and n3 (both succeed), '
            f'got {service.graphiti.refresh_entity_summary.await_count} attempts'
        )
        called_uuids = {
            call.args[0] for call in service.graphiti.refresh_entity_summary.call_args_list
        }
        assert {'n1', 'n2', 'n3'} <= called_uuids

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1, (
            f'Expected 1 WARNING record for the n1 failure, got {len(warning_records)}: '
            f'{[r.message for r in warning_records]}'
        )
        assert 'n1' in warning_records[0].message


class TestDualWriteCallbackRefreshesSummaries:
    """step-05/06: _dual_write_callback also triggers a post-ingestion summary
    refresh for edge endpoints touched by add_episode, in addition to its
    existing Mem0 batch-enqueue side effect."""

    @pytest.mark.asyncio
    async def test_callback_refreshes_endpoints_and_still_enqueues_mem0(self, service):
        """refresh_entity_summary is awaited for each edge endpoint, and the
        existing Mem0 batch-enqueue side effect is preserved."""
        from _fm_helpers import MockAddEpisodeResult, MockEdge

        service.graphiti.refresh_entity_summary = AsyncMock(return_value={})

        result = MockAddEpisodeResult(edges=[
            MockEdge(
                fact='Auth depends on Redis',
                source_node_uuid='auth',
                target_node_uuid='redis',
            ),
        ])
        payload = {
            'project_id': 'test',
            'group_id': 'test',
            'agent_id': 'a',
            '_causation_id': 'c',
        }
        await service._dual_write_callback('dual_write_episode', result, payload)

        service.durable_queue.enqueue_batch.assert_called_once()

        assert service.graphiti.refresh_entity_summary.await_count == 2, (
            'Expected refresh for both edge endpoints (auth, redis), got '
            f'{service.graphiti.refresh_entity_summary.await_count}'
        )
        called_uuids = {
            call.args[0] for call in service.graphiti.refresh_entity_summary.call_args_list
        }
        assert called_uuids == {'auth', 'redis'}
        for call in service.graphiti.refresh_entity_summary.call_args_list:
            assert call.kwargs['group_id'] == 'test'


class TestRefreshSummariesCallback:
    """step-07/08: dedicated refresh-only callback for the add_memory_graphiti /
    replay_from_store ingestion paths. Unlike _dual_write_callback, this must
    NOT enqueue a Mem0 batch — those paths already write Mem0 directly, so a
    second enqueue here would double-write."""

    @pytest.mark.asyncio
    async def test_refreshes_without_mem0_enqueue(self, service):
        """Refreshes both edge endpoints and never touches the Mem0 queue."""
        from _fm_helpers import MockAddEpisodeResult, MockEdge

        service.graphiti.refresh_entity_summary = AsyncMock(return_value={})

        result = MockAddEpisodeResult(edges=[
            MockEdge(fact='x', source_node_uuid='e1', target_node_uuid='e2'),
        ])
        payload = {'group_id': 'proj'}

        await service._refresh_summaries_callback(
            'refresh_entity_summaries', result, payload
        )

        assert service.graphiti.refresh_entity_summary.await_count == 2, (
            'Expected refresh for both edge endpoints (e1, e2), got '
            f'{service.graphiti.refresh_entity_summary.await_count}'
        )
        called_uuids = {
            call.args[0] for call in service.graphiti.refresh_entity_summary.call_args_list
        }
        assert called_uuids == {'e1', 'e2'}
        for call in service.graphiti.refresh_entity_summary.call_args_list:
            assert call.kwargs['group_id'] == 'proj'
        service.durable_queue.enqueue_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_result_is_noop(self, service):
        """None result → no refresh attempts, no raise."""
        service.graphiti.refresh_entity_summary = AsyncMock(return_value={})

        await service._refresh_summaries_callback(
            'refresh_entity_summaries', None, {'group_id': 'proj'}
        )

        service.graphiti.refresh_entity_summary.assert_not_called()
        service.durable_queue.enqueue_batch.assert_not_called()


class TestAddMemoryGraphitiRefreshWiring:
    """step-09/10: the add_memory_graphiti enqueue must carry
    callback_type='refresh_entity_summaries' so ingestion-time edge
    resolution on this path also triggers a post-write summary refresh."""

    @pytest.mark.asyncio
    async def test_add_memory_graphiti_enqueue_carries_refresh_callback_type(self, service):
        await service.add_memory(
            content='Auth depends on Redis',
            category='entities_and_relations',
            project_id='test',
        )

        graphiti_calls = [
            c for c in service.durable_queue.enqueue.call_args_list
            if c.kwargs.get('operation') == 'add_memory_graphiti'
        ]
        assert len(graphiti_calls) == 1, (
            f'Expected exactly 1 add_memory_graphiti enqueue, got {len(graphiti_calls)}'
        )
        assert graphiti_calls[0].kwargs.get('callback_type') == 'refresh_entity_summaries', (
            'add_memory_graphiti enqueue must carry callback_type=refresh_entity_summaries '
            f'so its ingestion-time edge resolution triggers a refresh; got '
            f'{graphiti_calls[0].kwargs.get("callback_type")!r}'
        )


class TestReplayFromStoreRefreshWiring:
    """step-11/12: replay_from_store's add_memory_graphiti batch items must
    carry callback_type='refresh_entity_summaries' so bulk re-ingestion also
    triggers a post-write summary refresh."""

    @pytest.mark.asyncio
    async def test_replay_batch_items_carry_refresh_callback_type(self, service):
        service.mem0.get_all = AsyncMock(return_value={
            'results': [
                {'memory': 'Auth depends on Redis', 'metadata': {'category': 'entities_and_relations'}},
            ]
        })

        await service.replay_from_store('test')

        service.durable_queue.enqueue_batch.assert_called_once()
        batch = service.durable_queue.enqueue_batch.call_args[0][0]
        graphiti_items = [item for item in batch if item['operation'] == 'add_memory_graphiti']
        assert graphiti_items, 'Expected at least one add_memory_graphiti batch item'
        for item in graphiti_items:
            assert item.get('callback_type') == 'refresh_entity_summaries', (
                'replay_from_store add_memory_graphiti batch items must carry '
                f'callback_type=refresh_entity_summaries; got {item.get("callback_type")!r}'
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
        from _fm_helpers import MockEdge

        from fused_memory.models.scope import Scope

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
        from _fm_helpers import MockEdge

        from fused_memory.models.scope import Scope

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
        from _fm_helpers import MockEdge

        from fused_memory.models.scope import Scope

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
        from _fm_helpers import MockEdge

        from fused_memory.models.scope import Scope

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
        from _fm_helpers import MockEdge

        from fused_memory.models.scope import Scope

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
        from _fm_helpers import MockEdge

        from fused_memory.models.scope import Scope

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
        from _fm_helpers import MockEdge

        from fused_memory.models.scope import Scope

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


class TestSearchMem0CreatedAt:
    """step-1 (task-1369): _search_mem0 propagates Mem0 server-stamped created_at to MemoryResult."""

    @pytest.mark.asyncio
    async def test_created_at_populated_from_top_level_field(self, service):
        """When mem0.search returns a result with top-level created_at, MemoryResult.created_at is set."""
        from fused_memory.models.scope import Scope

        service.mem0.search = AsyncMock(return_value={
            'results': [
                {
                    'id': 'm1',
                    'memory': 'x',
                    'score': 0.9,
                    'created_at': '2026-05-15T10:00:00+00:00',
                    'metadata': {'category': 'observations_and_summaries'},
                },
            ]
        })

        scope = Scope(project_id='test')
        results = await service._search_mem0('x', scope, 10)

        assert len(results) == 1
        assert results[0].created_at == '2026-05-15T10:00:00+00:00', (
            'MemoryResult.created_at must be populated from the Mem0 top-level created_at field'
        )

    @pytest.mark.asyncio
    async def test_created_at_is_none_when_key_absent(self, service):
        """When mem0.search returns a result without created_at, MemoryResult.created_at is None."""
        from fused_memory.models.scope import Scope

        service.mem0.search = AsyncMock(return_value={
            'results': [
                {
                    'id': 'm2',
                    'memory': 'y',
                    'score': 0.8,
                    'metadata': {'category': 'observations_and_summaries'},
                },
            ]
        })

        scope = Scope(project_id='test')
        results = await service._search_mem0('y', scope, 10)

        assert len(results) == 1
        assert results[0].created_at is None, (
            'MemoryResult.created_at must be None when created_at is absent in the Mem0 result'
        )


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

        async def mock_search_mem0(query, scope, limit, include_planned=False, categories=None):
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

        async def mock_search_mem0(query, scope, limit, include_planned=False, categories=None):
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


class TestInitializeQueueTransientConfigWiring:
    """Task 1936: initialize() must forward QueueConfig's error-aware retry
    fields (transient_max_attempts, transient_error_names) into the
    DurableWriteQueue construction.
    """

    @pytest.mark.asyncio
    async def test_transient_retry_config_forwarded(self, service, mock_config):
        mock_config.queue.transient_max_attempts = 999
        mock_config.queue.transient_error_names = ['SentinelTransientError']

        mock_registry_inst = MagicMock()
        mock_registry_inst.initialize = AsyncMock()
        MockRegistryCls = MagicMock(return_value=mock_registry_inst)

        mock_dq_inst = MagicMock()
        mock_dq_inst.initialize = AsyncMock()
        mock_dq_inst.register_callback = MagicMock()

        with (
            patch.object(service.graphiti, 'initialize', new_callable=AsyncMock),
            patch(
                'fused_memory.services.memory_service.DurableWriteQueue',
                return_value=mock_dq_inst,
            ) as MockDurableWriteQueueCls,
            patch(
                'fused_memory.services.planned_episode_registry.PlannedEpisodeRegistry',
                MockRegistryCls,
            ),
        ):
            await service.initialize()

        MockDurableWriteQueueCls.assert_called_once()
        kwargs = MockDurableWriteQueueCls.call_args.kwargs
        assert kwargs['transient_max_attempts'] == 999
        assert kwargs['transient_error_names'] == ['SentinelTransientError']


class TestSearchGraphitiInvalidatedFiltering:
    """Task 312: _search_graphiti filters out edges where invalid_at is not None."""

    # step-1: edge with invalid_at set is excluded
    @pytest.mark.asyncio
    async def test_edge_with_invalid_at_excluded(self, service):
        """Edge with non-null invalid_at is excluded from _search_graphiti results."""
        from _fm_helpers import MockEdge

        from fused_memory.models.scope import Scope

        dt_invalid = datetime(2024, 9, 1, 10, 0, 0, tzinfo=UTC)
        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Service B deprecated',
                uuid='edge-invalidated-1',
                valid_at=None,
                invalid_at=dt_invalid,
            ),
        ])

        scope = Scope(project_id='test')
        results = await service._search_graphiti('Service B', scope, 10)

        assert len(results) == 0, (
            'Edge with non-null invalid_at must be excluded from search results'
        )

    # step-3: edge with invalid_at=None is NOT excluded
    @pytest.mark.asyncio
    async def test_edge_without_invalid_at_not_excluded(self, service):
        """Edge with invalid_at=None (valid fact) is included in _search_graphiti results."""
        from _fm_helpers import MockEdge

        from fused_memory.models.scope import Scope

        dt_valid = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(
                fact='Service A is healthy',
                uuid='edge-valid-1',
                valid_at=dt_valid,
                invalid_at=None,
            ),
        ])

        scope = Scope(project_id='test')
        results = await service._search_graphiti('Service A', scope, 10)

        assert len(results) == 1, (
            'Edge with invalid_at=None must be included in search results'
        )
        assert results[0].id == 'edge-valid-1'

    # step-5: mixed valid and invalidated edges — only valid ones survive
    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalidated_edges_filtered(self, service):
        """3 edges (2 valid, 1 invalidated) → exactly 2 results, invalidated excluded."""
        from _fm_helpers import MockEdge

        from fused_memory.models.scope import Scope

        dt_valid = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        dt_invalid = datetime(2024, 9, 1, 0, 0, 0, tzinfo=UTC)
        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(fact='Current fact A', uuid='edge-valid-a', valid_at=dt_valid, invalid_at=None),
            MockEdge(fact='Superseded fact B', uuid='edge-invalid-b', valid_at=dt_valid, invalid_at=dt_invalid),
            MockEdge(fact='Current fact C', uuid='edge-valid-c', valid_at=dt_valid, invalid_at=None),
        ])

        scope = Scope(project_id='test')
        results = await service._search_graphiti('fact', scope, 10)

        assert len(results) == 2, (
            'Only 2 of 3 edges are valid; 1 invalidated edge must be excluded'
        )
        result_ids = {r.id for r in results}
        assert 'edge-invalid-b' not in result_ids, (
            'Invalidated edge must not appear in results'
        )
        assert 'edge-valid-a' in result_ids
        assert 'edge-valid-c' in result_ids

    # step-7: _search_graphiti over-fetches from Graphiti to compensate for filtered edges
    @pytest.mark.asyncio
    async def test_overfetch_compensates_for_filtered_edges(self, service):
        """_search_graphiti calls graphiti.search with num_results=int(limit*1.5)+1."""
        from fused_memory.models.scope import Scope

        service.graphiti.search = AsyncMock(return_value=[])

        scope = Scope(project_id='test')
        await service._search_graphiti('query', scope, limit=10)

        service.graphiti.search.assert_called_once()
        call_kwargs = service.graphiti.search.call_args
        actual_num_results = call_kwargs.kwargs.get('num_results', call_kwargs.args[2] if len(call_kwargs.args) > 2 else None)
        expected_num_results = int(10 * 1.5) + 1  # = 16
        assert actual_num_results == expected_num_results, (
            f'graphiti.search must be called with num_results={expected_num_results} '
            f'(int(limit * 1.5) + 1 for limit=10), got {actual_num_results}'
        )

    # step-9: results are truncated to limit after filtering
    @pytest.mark.asyncio
    async def test_results_truncated_to_limit(self, service):
        """When Graphiti returns more valid edges than limit, results are capped at limit."""
        from _fm_helpers import MockEdge

        from fused_memory.models.scope import Scope

        dt_valid = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        # 15 valid edges returned by Graphiti
        edges = [
            MockEdge(fact=f'Fact {n}', uuid=f'edge-valid-{n}', valid_at=dt_valid, invalid_at=None)
            for n in range(15)
        ]
        service.graphiti.search = AsyncMock(return_value=edges)

        scope = Scope(project_id='test')
        results = await service._search_graphiti('fact', scope, limit=10)

        assert len(results) == 10, (
            'Results must be truncated to limit=10 when Graphiti returns more valid edges'
        )

    # step-11: scores reflect original rank position from Graphiti, not re-ranked positions
    @pytest.mark.asyncio
    async def test_scores_reflect_original_rank_position(self, service):
        """Surviving edges keep scores from their original Graphiti rank positions.

        Graphiti returns 5 edges at positions 0-4. Edges at positions 1 and 3
        are invalidated (invalid_at set). The surviving edges at positions 0, 2,
        and 4 must score 1.0, 0.9, 0.8 respectively (score = max(0, 1 - i*0.05)),
        NOT re-ranked to 1.0, 0.95, 0.9 based on their post-filter positions.
        """
        from _fm_helpers import MockEdge

        from fused_memory.models.scope import Scope

        dt_valid = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        dt_invalid = datetime(2024, 9, 1, 0, 0, 0, tzinfo=UTC)

        service.graphiti.search = AsyncMock(return_value=[
            MockEdge(fact='Fact at pos 0', uuid='pos-0', valid_at=dt_valid, invalid_at=None),        # i=0, score=1.00
            MockEdge(fact='Superseded at pos 1', uuid='pos-1', valid_at=dt_valid, invalid_at=dt_invalid),  # filtered
            MockEdge(fact='Fact at pos 2', uuid='pos-2', valid_at=dt_valid, invalid_at=None),        # i=2, score=0.90
            MockEdge(fact='Superseded at pos 3', uuid='pos-3', valid_at=dt_valid, invalid_at=dt_invalid),  # filtered
            MockEdge(fact='Fact at pos 4', uuid='pos-4', valid_at=dt_valid, invalid_at=None),        # i=4, score=0.80
        ])

        scope = Scope(project_id='test')
        results = await service._search_graphiti('fact', scope, limit=10)

        assert len(results) == 3, 'Three edges should survive filtering'

        scores_by_id = {r.id: r.relevance_score for r in results}
        assert scores_by_id['pos-0'] == pytest.approx(1.00), (
            'Edge at original rank 0 must score 1.0 (1.0 - 0*0.05)'
        )
        assert scores_by_id['pos-2'] == pytest.approx(0.90), (
            'Edge at original rank 2 must score 0.9 (1.0 - 2*0.05)'
        )
        assert scores_by_id['pos-4'] == pytest.approx(0.80), (
            'Edge at original rank 4 must score 0.8 (1.0 - 4*0.05)'
        )


class TestGetStatusScoping:
    """get_status forwards project_id as group_id to durable_queue.get_stats."""

    @pytest.mark.asyncio
    async def test_get_status_with_project_id_passes_group_id_to_queue(self, service):
        """get_status(project_id='dark_factory') passes group_id='dark_factory'."""
        service.graphiti.list_graphs = AsyncMock(return_value=[])
        service.mem0.list_projects = AsyncMock(return_value=[])

        await service.get_status(project_id='dark_factory')

        service.durable_queue.get_stats.assert_called_once()
        call_kwargs = service.durable_queue.get_stats.call_args.kwargs
        assert call_kwargs.get('group_id') == 'dark_factory', (
            f'Expected group_id="dark_factory", got call_kwargs={call_kwargs}'
        )

    @pytest.mark.asyncio
    async def test_get_status_without_project_id_passes_no_group_filter(self, service):
        """get_status() with no project_id passes group_id=None (unscoped)."""
        service.graphiti.list_graphs = AsyncMock(return_value=[])
        service.mem0.list_projects = AsyncMock(return_value=[])

        await service.get_status()

        service.durable_queue.get_stats.assert_called_once()
        call_kwargs = service.durable_queue.get_stats.call_args.kwargs
        assert call_kwargs.get('group_id') is None, (
            f'Expected group_id=None, got call_kwargs={call_kwargs}'
        )

    @pytest.mark.asyncio
    async def test_get_status_returns_queue_section_unchanged_shape(self, service):
        """The queue section returned by get_status equals get_stats mock output."""
        service.graphiti.list_graphs = AsyncMock(return_value=[])
        service.mem0.list_projects = AsyncMock(return_value=[])

        fixed_stats = {'counts': {'pending': 2, 'dead': 1}, 'oldest_pending_age_seconds': 5.0}
        service.durable_queue.get_stats = AsyncMock(return_value=fixed_stats)

        result = await service.get_status(project_id='proj1')

        assert 'queue' in result, f'Expected "queue" key in result; got {list(result)}'
        assert result['queue'] == fixed_stats, (
            f'queue section should equal get_stats output; got {result["queue"]}'
        )


# ---------------------------------------------------------------------------
# Step 9: TRACK B.2 restore hook RED tests
# ---------------------------------------------------------------------------

class TestRestoreSupersededDependencyEdges:
    """_restore_superseded_dependency_edges must clear false dependency invalidations."""

    def _make_mock_edge(self, *, uuid, fact, invalid_at=None):
        edge = MagicMock()
        edge.uuid = uuid
        edge.fact = fact
        edge.invalid_at = invalid_at
        return edge

    @pytest.mark.asyncio
    async def test_restores_only_invalidated_dependency_edges(self, service):
        """Only edges that are (a) invalidated AND (b) dependency facts are restored."""
        from datetime import UTC, datetime
        ts = datetime(2026, 1, 1, tzinfo=UTC)

        edge_a = self._make_mock_edge(
            uuid='a', fact='Task 562 depends on Task 557', invalid_at=ts,
        )
        edge_b = self._make_mock_edge(
            uuid='b', fact='Task 562 reached status done', invalid_at=ts,
        )
        edge_c = self._make_mock_edge(
            uuid='c', fact='Task 563 depends on Task 558', invalid_at=None,
        )

        result = MagicMock()
        result.edges = [edge_a, edge_b, edge_c]

        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'a', 'fact': 'Task 562 depends on Task 557', 'refreshed_nodes': []}
        )

        count = await service._restore_superseded_dependency_edges(result, group_id='proj')
        assert count == 1, f'Expected 1 restored edge, got {count}'
        service.graphiti.update_edge.assert_called_once_with(
            'a', group_id='proj', clear_invalid_at=True
        )

    @pytest.mark.asyncio
    async def test_returns_zero_for_none_result(self, service):
        """Returns 0 and makes no calls when result is None."""
        service.graphiti.update_edge = AsyncMock()
        count = await service._restore_superseded_dependency_edges(None, group_id='proj')
        assert count == 0
        service.graphiti.update_edge.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_edges(self, service):
        """Returns 0 and makes no calls when result.edges is empty."""
        result = MagicMock()
        result.edges = []
        service.graphiti.update_edge = AsyncMock()
        count = await service._restore_superseded_dependency_edges(result, group_id='proj')
        assert count == 0
        service.graphiti.update_edge.assert_not_called()


# ---------------------------------------------------------------------------
# Step 7: TRACK B.2 _is_dependency_fact helper RED tests
# ---------------------------------------------------------------------------

class TestIsDependencyFact:
    """Module-level _is_dependency_fact helper matches 'depends on' regardless of case/spacing."""

    def test_canonical_dependency_fact(self):
        from fused_memory.services.memory_service import _is_dependency_fact
        assert _is_dependency_fact('Task 562 depends on Task 557') is True

    def test_depends_on_case_insensitive(self):
        from fused_memory.services.memory_service import _is_dependency_fact
        assert _is_dependency_fact('X  Depends On  Y') is True

    def test_lowercase_depends_on(self):
        from fused_memory.services.memory_service import _is_dependency_fact
        assert _is_dependency_fact('module a depends on module b') is True

    def test_non_dependency_status_fact(self):
        from fused_memory.services.memory_service import _is_dependency_fact
        assert _is_dependency_fact('Task 562 reached status done') is False

    def test_non_dependency_owns_fact(self):
        from fused_memory.services.memory_service import _is_dependency_fact
        assert _is_dependency_fact('Task 562 owns module X') is False

    def test_empty_string(self):
        from fused_memory.services.memory_service import _is_dependency_fact
        assert _is_dependency_fact('') is False

    def test_none_value(self):
        from fused_memory.services.memory_service import _is_dependency_fact
        assert _is_dependency_fact(None) is False  # type: ignore[arg-type]


class TestReconPoolAutoTag:
    """Module-level _infer_recon_pool helper + _CYCLE_SUMMARY_STAGE_TO_RECON_POOL map.

    recon_pool is the only key the pool-cap trim and prune script filter on
    (reconciliation/summary_pool.py, scripts/prune_recon_cycle_summaries.py).
    _infer_recon_pool derives it server-side from metadata.stage for
    kind == 'cycle_summary' writes only, so tagging no longer depends on LLM
    prompt compliance (task 2077).
    """

    def test_memory_consolidator_stage_maps_to_stage1_pool(self):
        from fused_memory.services.memory_service import _infer_recon_pool
        meta = {'kind': 'cycle_summary', 'stage': 'memory_consolidator'}
        assert _infer_recon_pool(meta) == 'stage1_cycle_summary'

    def test_task_knowledge_sync_stage_maps_to_stage2_pool(self):
        from fused_memory.services.memory_service import _infer_recon_pool
        meta = {'kind': 'cycle_summary', 'stage': 'task_knowledge_sync'}
        assert _infer_recon_pool(meta) == 'stage2_cycle_summary'

    def test_missing_stage_returns_none(self):
        """stage absent -> cannot infer; add_memory must leave any caller recon_pool alone."""
        from fused_memory.services.memory_service import _infer_recon_pool
        meta = {'kind': 'cycle_summary'}
        assert _infer_recon_pool(meta) is None

    def test_unknown_stage_returns_none(self):
        from fused_memory.services.memory_service import _infer_recon_pool
        meta = {'kind': 'cycle_summary', 'stage': 'unknown_stage'}
        assert _infer_recon_pool(meta) is None

    def test_non_cycle_summary_kind_returns_none(self):
        """Only kind == 'cycle_summary' writes are touched; other kinds pass through untouched."""
        from fused_memory.services.memory_service import _infer_recon_pool
        meta = {'kind': 'note', 'stage': 'memory_consolidator'}
        assert _infer_recon_pool(meta) is None

    def test_empty_metadata_returns_none(self):
        from fused_memory.services.memory_service import _infer_recon_pool
        assert _infer_recon_pool({}) is None

    def test_map_matches_canonical_per_stage_constants(self):
        """Drift guard: the map's values must equal the per-stage canonical constants
        that stage1/stage2 themselves emit, so the two copies can't silently diverge."""
        from fused_memory.reconciliation.stages.memory_consolidator import (
            _STAGE1_CYCLE_SUMMARY_RECON_POOL,
        )
        from fused_memory.reconciliation.stages.task_knowledge_sync import (
            _STAGE2_CYCLE_SUMMARY_RECON_POOL,
        )
        from fused_memory.services.memory_service import (
            _CYCLE_SUMMARY_STAGE_TO_RECON_POOL,
        )
        assert (
            _CYCLE_SUMMARY_STAGE_TO_RECON_POOL['memory_consolidator']
            == _STAGE1_CYCLE_SUMMARY_RECON_POOL
        )
        assert (
            _CYCLE_SUMMARY_STAGE_TO_RECON_POOL['task_knowledge_sync']
            == _STAGE2_CYCLE_SUMMARY_RECON_POOL
        )


class TestMissingCycleSummaryKeys:
    """Module-level _missing_cycle_summary_keys helper (task 2094).

    Returns which required keys among ('stage', 'run_id') are missing/invalid
    on a cycle_summary write, else []. stage is invalid when absent, non-str,
    or not a known key in _CYCLE_SUMMARY_STAGE_TO_RECON_POOL. run_id is
    invalid when absent, non-str, or empty/whitespace-only. Non-cycle_summary
    kinds (and empty metadata) are never flagged.
    """

    def test_known_stage1_and_run_id_present_returns_empty(self):
        from fused_memory.services.memory_service import _missing_cycle_summary_keys
        meta = {'kind': 'cycle_summary', 'stage': 'memory_consolidator', 'run_id': 'r1'}
        assert _missing_cycle_summary_keys(meta) == []

    def test_known_stage2_and_run_id_present_returns_empty(self):
        from fused_memory.services.memory_service import _missing_cycle_summary_keys
        meta = {'kind': 'cycle_summary', 'stage': 'task_knowledge_sync', 'run_id': 'r1'}
        assert _missing_cycle_summary_keys(meta) == []

    def test_valid_stage_missing_run_id_flags_run_id(self):
        meta = {'kind': 'cycle_summary', 'stage': 'memory_consolidator'}
        from fused_memory.services.memory_service import _missing_cycle_summary_keys
        assert _missing_cycle_summary_keys(meta) == ['run_id']

    def test_valid_stage_empty_run_id_flags_run_id(self):
        from fused_memory.services.memory_service import _missing_cycle_summary_keys
        meta = {
            'kind': 'cycle_summary',
            'stage': 'memory_consolidator',
            'run_id': '',
        }
        assert _missing_cycle_summary_keys(meta) == ['run_id']

    def test_valid_stage_whitespace_run_id_flags_run_id(self):
        from fused_memory.services.memory_service import _missing_cycle_summary_keys
        meta = {
            'kind': 'cycle_summary',
            'stage': 'memory_consolidator',
            'run_id': '   ',
        }
        assert _missing_cycle_summary_keys(meta) == ['run_id']

    def test_missing_stage_present_run_id_flags_stage(self):
        from fused_memory.services.memory_service import _missing_cycle_summary_keys
        meta = {'kind': 'cycle_summary', 'run_id': 'r1'}
        assert _missing_cycle_summary_keys(meta) == ['stage']

    def test_unknown_stage_present_run_id_flags_stage(self):
        from fused_memory.services.memory_service import _missing_cycle_summary_keys
        meta = {'kind': 'cycle_summary', 'stage': 'unknown_stage', 'run_id': 'r1'}
        assert _missing_cycle_summary_keys(meta) == ['stage']

    def test_both_missing_returns_stable_order(self):
        """Both absent -> ['stage', 'run_id'], deterministic order."""
        from fused_memory.services.memory_service import _missing_cycle_summary_keys
        meta = {'kind': 'cycle_summary'}
        assert _missing_cycle_summary_keys(meta) == ['stage', 'run_id']

    def test_non_cycle_summary_kind_returns_empty(self):
        from fused_memory.services.memory_service import _missing_cycle_summary_keys
        meta = {'kind': 'note'}
        assert _missing_cycle_summary_keys(meta) == []

    def test_empty_metadata_returns_empty(self):
        from fused_memory.services.memory_service import _missing_cycle_summary_keys
        assert _missing_cycle_summary_keys({}) == []


class TestCycleSummaryRunIdBackfillHelper:
    """Module-level _cycle_summary_run_id_backfill pure helper (task 2109).

    Auto-backfills a missing/invalid cycle_summary run_id from an
    authoritative causation id, server-side, independent of LLM prompt
    compliance — mirrors the _infer_recon_pool precedent (task 2077).

    Two candidate sources are checked, in order:
      1. meta['_causation_id'] — the task's literal wording; also what a
         direct in-process caller would set.
      2. the causation_id parameter — the production MCP-boundary path:
         server/tools.py::_extract_causation pops '_causation_id' out of the
         metadata dict into this parameter before MemoryService.add_memory
         runs, so on the real reconciliation stage-agent path
         meta['_causation_id'] is always absent and the run_id value lives
         only in this parameter.

    RED: the helper does not exist yet (ImportError).
    """

    def test_missing_run_id_backfills_from_meta_causation_id(self):
        from fused_memory.services.memory_service import _cycle_summary_run_id_backfill
        meta = {
            'kind': 'cycle_summary',
            'stage': 'memory_consolidator',
            '_causation_id': 'run-a',
        }
        assert _cycle_summary_run_id_backfill(meta, None) == 'run-a'

    def test_missing_run_id_backfills_from_causation_id_param(self):
        """The production MCP-boundary path: _extract_causation already
        popped _causation_id out of metadata into the causation_id
        parameter, so meta has no '_causation_id' key at all."""
        from fused_memory.services.memory_service import _cycle_summary_run_id_backfill
        meta = {'kind': 'cycle_summary', 'stage': 'memory_consolidator'}
        assert _cycle_summary_run_id_backfill(meta, 'run-b') == 'run-b'

    def test_both_sources_present_prefers_meta_causation_id(self):
        from fused_memory.services.memory_service import _cycle_summary_run_id_backfill
        meta = {
            'kind': 'cycle_summary',
            'stage': 'memory_consolidator',
            '_causation_id': 'run-a',
        }
        assert _cycle_summary_run_id_backfill(meta, 'run-b') == 'run-a'

    def test_run_id_already_valid_returns_none(self):
        """Nothing to repair — must not clobber a valid caller-supplied run_id."""
        from fused_memory.services.memory_service import _cycle_summary_run_id_backfill
        meta = {
            'kind': 'cycle_summary',
            'stage': 'memory_consolidator',
            'run_id': 'r1',
            '_causation_id': 'run-a',
        }
        assert _cycle_summary_run_id_backfill(meta, 'run-b') is None

    def test_empty_or_whitespace_run_id_backfills_stripped_value(self):
        """run_id present but empty/whitespace-only is treated as missing,
        and the backfilled value is stripped."""
        from fused_memory.services.memory_service import _cycle_summary_run_id_backfill
        meta = {
            'kind': 'cycle_summary',
            'stage': 'memory_consolidator',
            'run_id': '   ',
        }
        assert _cycle_summary_run_id_backfill(meta, '  run-c  ') == 'run-c'

    def test_no_source_available_returns_none(self):
        """run_id absent AND no causation source anywhere -> unrepairable, None."""
        from fused_memory.services.memory_service import _cycle_summary_run_id_backfill
        meta = {'kind': 'cycle_summary', 'stage': 'memory_consolidator'}
        assert _cycle_summary_run_id_backfill(meta, None) is None

    def test_non_cycle_summary_kind_returns_none(self):
        """Only kind == 'cycle_summary' writes are ever touched, even when a
        causation source is available."""
        from fused_memory.services.memory_service import _cycle_summary_run_id_backfill
        meta = {'kind': 'note'}
        assert _cycle_summary_run_id_backfill(meta, 'run-b') is None


class TestReconPoolAutoTagInjection:
    """Integration: add_memory must inject recon_pool into the metadata dict
    handed to mem0.add for cycle_summary writes, server-side, independent of
    whether the caller passed recon_pool (task 2077).

    RED until _infer_recon_pool is wired into add_memory (step-4).
    """

    @pytest.mark.asyncio
    async def test_stage1_recon_pool_injected(self, service):
        await service.add_memory(
            content='Cycle 3 summary: completed steps 1-4',
            category='observations_and_summaries',
            project_id='test',
            metadata={
                'kind': 'cycle_summary',
                'stage': 'memory_consolidator',
                'run_id': 'r1',
            },
        )
        call_kwargs = service.mem0.add.call_args[1]
        assert call_kwargs['metadata']['recon_pool'] == 'stage1_cycle_summary'

    @pytest.mark.asyncio
    async def test_stage2_recon_pool_injected(self, service):
        await service.add_memory(
            content='Cycle 3 summary: completed steps 1-4',
            category='observations_and_summaries',
            project_id='test',
            metadata={
                'kind': 'cycle_summary',
                'stage': 'task_knowledge_sync',
                'run_id': 'r1',
            },
        )
        call_kwargs = service.mem0.add.call_args[1]
        assert call_kwargs['metadata']['recon_pool'] == 'stage2_cycle_summary'

    @pytest.mark.asyncio
    async def test_non_cycle_summary_kind_not_tagged(self, service):
        """metadata.kind != 'cycle_summary' -> recon_pool must not appear at all."""
        await service.add_memory(
            content='Always use type hints',
            category='observations_and_summaries',
            project_id='test',
            metadata={'kind': 'note'},
        )
        call_kwargs = service.mem0.add.call_args[1]
        assert 'recon_pool' not in call_kwargs['metadata']

    @pytest.mark.asyncio
    async def test_no_metadata_not_tagged(self, service):
        """No metadata at all -> recon_pool must not appear."""
        await service.add_memory(
            content='Always use type hints',
            category='observations_and_summaries',
            project_id='test',
        )
        call_kwargs = service.mem0.add.call_args[1]
        assert 'recon_pool' not in call_kwargs['metadata']

    @pytest.mark.asyncio
    async def test_authoritative_override_corrects_wrong_caller_value(self, service):
        """A known stage's derived recon_pool wins over a caller-supplied value."""
        await service.add_memory(
            content='Cycle 3 summary: completed steps 1-4',
            category='observations_and_summaries',
            project_id='test',
            metadata={
                'kind': 'cycle_summary',
                'stage': 'memory_consolidator',
                'recon_pool': 'WRONG',
            },
        )
        call_kwargs = service.mem0.add.call_args[1]
        assert call_kwargs['metadata']['recon_pool'] == 'stage1_cycle_summary'

    @pytest.mark.asyncio
    async def test_unknown_stage_preserves_caller_recon_pool(self, service):
        """Stage unknown -> cannot infer -> caller-supplied recon_pool must survive untouched."""
        await service.add_memory(
            content='Cycle 3 summary: completed steps 1-4',
            category='observations_and_summaries',
            project_id='test',
            metadata={
                'kind': 'cycle_summary',
                'stage': 'unknown',
                'recon_pool': 'caller_val',
            },
        )
        call_kwargs = service.mem0.add.call_args[1]
        assert call_kwargs['metadata']['recon_pool'] == 'caller_val'


class TestReconPoolAutoTagMissingStageWarning:
    """metadata.stage is itself LLM-supplied (see the reconciliation
    stage1/stage2 prompts), so a missing/unknown stage means recon_pool
    could not be derived server-side either — the same prompt-compliance
    failure task 2077 targets, just shifted from the recon_pool field to the
    stage field. A WARNING makes these writes observable instead of silently
    relying on whatever (if anything) the caller passed for recon_pool
    (amendment review, task 2077).
    """

    @pytest.mark.asyncio
    async def test_missing_stage_logs_warning(self, service, caplog):
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='dark_factory',
                metadata={'kind': 'cycle_summary', 'run_id': 'run-x'},
                causation_id='run-x',
            )

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1, (
            f'Expected exactly 1 WARNING for a cycle_summary with missing stage, '
            f'got {len(warning_records)}: {[r.message for r in warning_records]}'
        )
        record = warning_records[0]
        message = record.message.lower()
        assert 'cycle_summary' in message and 'stage' in message, (
            f'Expected the WARNING to name the missing/unknown-stage condition, '
            f'got: {record.message!r}'
        )
        assert record.stage is None
        assert record.caller_recon_pool is None
        assert record.causation_id == 'run-x'
        assert record.project_id == 'dark_factory'

    @pytest.mark.asyncio
    async def test_unknown_stage_logs_warning(self, service, caplog):
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='dark_factory',
                metadata={
                    'kind': 'cycle_summary',
                    'stage': 'unknown_stage',
                    'recon_pool': 'caller_val',
                    'run_id': 'run-x',
                },
                causation_id='run-x',
            )

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1, (
            f'Expected exactly 1 WARNING for a cycle_summary with an unknown stage, '
            f'got {len(warning_records)}: {[r.message for r in warning_records]}'
        )
        record = warning_records[0]
        assert record.stage == 'unknown_stage'
        assert record.caller_recon_pool == 'caller_val'

    @pytest.mark.asyncio
    async def test_known_stage_no_warning(self, service, caplog):
        """Happy path (known stage) must not trigger the missing-stage WARNING."""
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='dark_factory',
                metadata={
                    'kind': 'cycle_summary',
                    'stage': 'memory_consolidator',
                    'run_id': 'run-x',
                },
                causation_id='run-x',
            )
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            f'Expected no WARNING when stage is known, got: '
            f'{[r.message for r in warning_records]}'
        )

    @pytest.mark.asyncio
    async def test_non_cycle_summary_kind_no_warning_even_without_stage(self, service, caplog):
        """Only kind == 'cycle_summary' writes are subject to this warning."""
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Always use type hints',
                category='observations_and_summaries',
                project_id='dark_factory',
                metadata={'kind': 'note'},
            )
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            f'Expected no WARNING for a non-cycle_summary kind, got: '
            f'{[r.message for r in warning_records]}'
        )

    @pytest.mark.asyncio
    async def test_no_metadata_no_warning(self, service, caplog):
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Always use type hints',
                category='observations_and_summaries',
                project_id='dark_factory',
            )
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            f'Expected no WARNING when no metadata is supplied, got: '
            f'{[r.message for r in warning_records]}'
        )


class TestCycleSummaryRunIdGuard:
    """add_memory's cycle_summary run_id guard (task 2094), updated for the
    task-2109 auto-backfill.

    Task 2077/1653 only warned when recon_pool could not be inferred (i.e.
    stage missing/unknown) via an `elif` gated on `inferred_recon_pool is
    None`. A write with a VALID stage short-circuits into the `if` branch and
    never reaches a run_id check, so a dropped/empty run_id sailed through
    with zero warnings — invisible to the Path-2 triple-filter
    count_memories_by_metadata({kind, run_id, stage}) pre-check. Task 2094
    decoupled the check so it always ran, but only WARNED about the drop.

    Task 2109 replaces that warn-only behavior with server-side auto-backfill
    (see TestCycleSummaryRunIdBackfillInjection for the full backfill
    contract): a dropped/empty run_id with a usable causation_id ('run-x'
    here) is now REPAIRED rather than merely logged, so the two tests below
    assert the repair and the absence of a warning. The remaining control
    test documents the still-silent happy path (stage and run_id both
    present and valid).
    """

    @pytest.mark.asyncio
    async def test_run_id_dropped_with_valid_stage_is_backfilled(self, service, caplog):
        """Superseded by task 2109: a dropped run_id is now backfilled from
        causation_id rather than merely warned about."""
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='dark_factory',
                metadata={'kind': 'cycle_summary', 'stage': 'memory_consolidator'},
                causation_id='run-x',
            )

        call_kwargs = service.mem0.add.call_args[1]
        assert call_kwargs['metadata']['run_id'] == 'run-x'
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            f'Expected no WARNING once the dropped run_id is backfilled from '
            f'causation_id, got: {[r.message for r in warning_records]}'
        )

    @pytest.mark.asyncio
    async def test_empty_run_id_with_valid_stage_is_backfilled(self, service, caplog):
        """Superseded by task 2109: an empty run_id is now backfilled from
        causation_id rather than merely warned about."""
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='dark_factory',
                metadata={
                    'kind': 'cycle_summary',
                    'stage': 'memory_consolidator',
                    'run_id': '',
                },
                causation_id='run-x',
            )

        call_kwargs = service.mem0.add.call_args[1]
        assert call_kwargs['metadata']['run_id'] == 'run-x'
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            f'Expected no WARNING once the empty run_id is backfilled from '
            f'causation_id, got: {[r.message for r in warning_records]}'
        )

    @pytest.mark.asyncio
    async def test_stage_and_run_id_both_present_no_warning(self, service, caplog):
        """Control: known stage AND present run_id -> silent (happy path)."""
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='dark_factory',
                metadata={
                    'kind': 'cycle_summary',
                    'stage': 'memory_consolidator',
                    'run_id': 'run-x',
                },
                causation_id='run-x',
            )
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            f'Expected no WARNING when both stage and run_id are present and valid, '
            f'got: {[r.message for r in warning_records]}'
        )


class TestCycleSummaryRunIdBackfillInjection:
    """Integration: add_memory must auto-backfill metadata.run_id for
    cycle_summary writes from an authoritative causation id, server-side,
    independent of LLM prompt compliance (task 2109 — replaces the
    warn-only guard from task 2094; see TestCycleSummaryRunIdGuard).

    RED until _cycle_summary_run_id_backfill is wired into add_memory
    (step-4).
    """

    @pytest.mark.asyncio
    async def test_backfills_from_meta_causation_id_no_warning(self, service, caplog):
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='test',
                metadata={
                    'kind': 'cycle_summary',
                    'stage': 'memory_consolidator',
                    '_causation_id': 'run-a',
                },
            )
        call_kwargs = service.mem0.add.call_args[1]
        assert call_kwargs['metadata']['run_id'] == 'run-a'
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            f'Expected no WARNING once run_id is backfilled from '
            f"meta['_causation_id'], got: {[r.message for r in warning_records]}"
        )

    @pytest.mark.asyncio
    async def test_backfills_from_causation_id_param_production_path(self, service, caplog):
        """The production MCP path: _extract_causation already popped
        _causation_id out of metadata into the causation_id parameter, so
        meta itself carries no '_causation_id' key."""
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='test',
                metadata={'kind': 'cycle_summary', 'stage': 'memory_consolidator'},
                causation_id='run-b',
            )
        call_kwargs = service.mem0.add.call_args[1]
        assert call_kwargs['metadata']['run_id'] == 'run-b'
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            f'Expected no WARNING once run_id is backfilled from the '
            f'causation_id parameter, got: {[r.message for r in warning_records]}'
        )

    @pytest.mark.asyncio
    async def test_empty_run_id_backfilled_from_causation_id_param(self, service, caplog):
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='test',
                metadata={
                    'kind': 'cycle_summary',
                    'stage': 'memory_consolidator',
                    'run_id': '',
                },
                causation_id='run-c',
            )
        call_kwargs = service.mem0.add.call_args[1]
        assert call_kwargs['metadata']['run_id'] == 'run-c'
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            f'Expected no WARNING once the empty run_id is backfilled, got: '
            f'{[r.message for r in warning_records]}'
        )

    @pytest.mark.asyncio
    async def test_no_causation_source_falls_back_to_warning(self, service, caplog):
        """Unrepairable: no meta['_causation_id'] and no causation_id param ->
        the pre-task-2109 warn-only fallback still fires."""
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='test',
                metadata={'kind': 'cycle_summary', 'stage': 'memory_consolidator'},
            )
        call_kwargs = service.mem0.add.call_args[1]
        run_id = call_kwargs['metadata'].get('run_id')
        assert not (isinstance(run_id, str) and run_id.strip()), (
            f'Expected no valid run_id when no causation source is available, '
            f'got {run_id!r}'
        )
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1, (
            f'Expected exactly 1 fallback WARNING when run_id is unrepairable, '
            f'got {len(warning_records)}: {[r.message for r in warning_records]}'
        )
        assert 'run_id' in warning_records[0].missing_cycle_summary_keys

    @pytest.mark.asyncio
    async def test_backfill_does_not_suppress_missing_stage_warning(self, service, caplog):
        """run_id is backfillable, but stage is NOT — the fallback WARNING
        must still fire, scoped to ['stage'] only (run_id must no longer be
        flagged once repaired)."""
        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='test',
                metadata={'kind': 'cycle_summary', '_causation_id': 'run-z'},
            )
        call_kwargs = service.mem0.add.call_args[1]
        assert call_kwargs['metadata']['run_id'] == 'run-z'
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1, (
            f'Expected exactly 1 WARNING (stage still missing/unrepairable), '
            f'got {len(warning_records)}: {[r.message for r in warning_records]}'
        )
        assert warning_records[0].missing_cycle_summary_keys == ['stage']


class TestCycleSummaryRunIdBackfillToolsBoundary:
    """Boundary-level coverage between server/tools.py::_extract_causation's
    UUID-generation fallback and the task-2109 run_id auto-backfill
    (reviewer finding from the task 2109 amendment pass).

    _extract_causation never actually passes a None causation_id through:
    when metadata carries no '_causation_id' (e.g. the reconciliation stage
    agent dropped it under the same kind of prompt-compliance failure this
    task's run_id backfill exists to paper over), it synthesizes a fresh
    str(uuid4()) on the spot (tools.py ~398-399). MemoryService.add_memory
    has no signal to tell that call-scoped synthetic value apart from a
    genuine run-scoped causation id — reconciliation/harness.py generates
    run_id the same way, via str(uuid4()) — so
    `_cycle_summary_run_id_backfill` backfills run_id from it regardless.

    This is a KNOWN, ACCEPTED-FOR-NOW LIMITATION, not intended long-term
    behavior: the write stops warning (looks healthy), but the backfilled
    run_id is a meaningless per-call value that will never match any
    sibling cycle_summary write from the same actual reconciliation run in
    the Path-2 triple-filter count_memories_by_metadata({kind, run_id,
    stage}) pre-check. Closing the gap requires _extract_causation to flag
    whether it synthesized (vs. received) the id and, for a synthesized id,
    skip the run_id backfill in favor of the fallback warning — a change to
    fused_memory/server/tools.py, outside this task's locked module scope
    (fused_memory/services + this test file). Tracked as a follow-up; this
    test pins today's (masking) behavior so a future fix must consciously
    come back and update it rather than silently regress.
    """

    @pytest.mark.asyncio
    async def test_synthesized_causation_id_is_still_backfilled_as_run_id(self, service, caplog):
        """Neither run_id nor metadata['_causation_id'] survive to the MCP
        boundary -> _extract_causation synthesizes a fresh UUID -> today
        that synthetic value is (mis)backfilled into run_id with no
        warning. See class docstring for why this is a tracked gap rather
        than intended behavior."""
        from fused_memory.server.tools import _extract_causation

        causation_id, _source, cleaned_meta = _extract_causation(
            {'kind': 'cycle_summary', 'stage': 'memory_consolidator'},
            'recon-stage-memory_consolidator',
        )
        assert cleaned_meta is not None
        assert '_causation_id' not in cleaned_meta, (
            "_extract_causation is expected to pop '_causation_id' out of "
            'metadata before returning it — this test only exercises the '
            'no-_causation_id-supplied fallback path.'
        )

        with caplog.at_level(logging.WARNING, logger='fused_memory.services.memory_service'):
            await service.add_memory(
                content='Cycle 3 summary: completed steps 1-4',
                category='observations_and_summaries',
                project_id='test',
                metadata=cleaned_meta,
                causation_id=causation_id,
            )

        call_kwargs = service.mem0.add.call_args[1]
        assert call_kwargs['metadata']['run_id'] == causation_id, (
            'Documents the known gap: a causation_id synthesized by '
            '_extract_causation (no real relationship to any reconciliation '
            'run) is currently backfilled into run_id verbatim.'
        )
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == [], (
            'The write looks healthy (no warning) despite run_id being a '
            f'meaningless synthetic id, got: {[r.message for r in warning_records]}'
        )


# ---------------------------------------------------------------------------
# Step 3: TRACK B.1 service clear_invalid_at RED tests
# ---------------------------------------------------------------------------

class TestUpdateEdgeClearInvalidAt:
    """Service update_edge must forward clear_invalid_at through to the backend."""

    @pytest.mark.asyncio
    async def test_clear_invalid_at_forwards_to_backend(self, service):
        """service.update_edge(clear_invalid_at=True) calls backend with clear_invalid_at=True."""
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'unchanged', 'refreshed_nodes': []}
        )
        await service.update_edge(edge_uuid='e-1', project_id='proj', clear_invalid_at=True)
        service.graphiti.update_edge.assert_called_once_with(
            'e-1', None, group_id='proj', invalid_at=None, clear_invalid_at=True,
        )

    @pytest.mark.asyncio
    async def test_clear_invalid_at_result_is_verified_true(self, service):
        """Clear-only update is verified=True when the invalid_at readback confirms the clear.

        Revised (task 1940) from the old 'verified=True without a readback'
        contract: the no-fact branch used to hardcode verified=True even
        though the clear itself was never confirmed to have persisted.
        """
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'unchanged', 'refreshed_nodes': []}
        )
        service.graphiti.get_edge_text = AsyncMock(return_value=('edge', 'fact'))
        service.graphiti.get_edge_invalid_at = AsyncMock(return_value=None)
        result = await service.update_edge(
            edge_uuid='e-1', project_id='proj', clear_invalid_at=True
        )
        assert result.get('verified') is True, (
            'clear-only update must be verified=True when invalid_at readback is None'
        )
        service.graphiti.get_edge_invalid_at.assert_awaited_once_with(
            'e-1', group_id='proj'
        )
        service.graphiti.get_edge_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clear_only_verified_false_when_still_invalid(self, service):
        """get_edge_invalid_at returning a non-None value means the clear did not persist."""
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'unchanged', 'refreshed_nodes': []}
        )
        service.graphiti.get_edge_invalid_at = AsyncMock(
            return_value='2026-01-01T00:00:00+00:00'
        )
        result = await service.update_edge(
            edge_uuid='e-1', project_id='proj', clear_invalid_at=True
        )
        assert result.get('verified') is False
        assert 'verification_error' in result
        assert result['verification_error']  # non-empty string

    @pytest.mark.asyncio
    async def test_fact_plus_clear_verified_false_when_not_cleared(self, service):
        """fact readback matches but invalid_at readback is non-None → verified False (AND)."""
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'new fact', 'refreshed_nodes': []}
        )
        service.graphiti.get_edge_text = AsyncMock(return_value=('Edge', 'new fact'))
        service.graphiti.get_edge_invalid_at = AsyncMock(
            return_value='2026-01-01T00:00:00+00:00'
        )
        result = await service.update_edge(
            edge_uuid='e-1', fact='new fact', project_id='proj', clear_invalid_at=True
        )
        assert result.get('verified') is False, (
            'verified must be the AND of the fact and invalid_at readbacks'
        )
        assert 'invalid_at' in result.get('verification_error', '')

    @pytest.mark.asyncio
    async def test_fact_plus_clear_verified_true_when_both_ok(self, service):
        """fact readback matches AND invalid_at readback is None → verified True."""
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'new fact', 'refreshed_nodes': []}
        )
        service.graphiti.get_edge_text = AsyncMock(return_value=('Edge', 'new fact'))
        service.graphiti.get_edge_invalid_at = AsyncMock(return_value=None)
        result = await service.update_edge(
            edge_uuid='e-1', fact='new fact', project_id='proj', clear_invalid_at=True
        )
        assert result.get('verified') is True

    @pytest.mark.asyncio
    async def test_fact_fail_plus_clear_ok_verified_false(self, service):
        """fact readback mismatches even though invalid_at readback clears → verified False (AND).

        Exercises the AND logic from the 'fact fails' side: a successful
        clear must not paper over a failed fact-text verification.
        """
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'new fact', 'refreshed_nodes': []}
        )
        service.graphiti.get_edge_text = AsyncMock(return_value=('Edge', 'stale fact'))
        service.graphiti.get_edge_invalid_at = AsyncMock(return_value=None)
        result = await service.update_edge(
            edge_uuid='e-1', fact='new fact', project_id='proj', clear_invalid_at=True
        )
        assert result.get('verified') is False, (
            'verified must stay False when the fact readback fails, even if invalid_at clears'
        )
        assert 'Readback fact mismatch' in result.get('verification_error', '')

    @pytest.mark.asyncio
    async def test_fact_fail_plus_clear_exception_preserves_fact_error(self, service):
        """An exception in the clear_invalid_at readback must not discard a prior fact-mismatch error.

        Regression test: the Guard-2b exception handler used to overwrite
        result['verification_error'] outright, discarding the fact-mismatch
        diagnostic set by the block above it.
        """
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'new fact', 'refreshed_nodes': []}
        )
        service.graphiti.get_edge_text = AsyncMock(return_value=('Edge', 'stale fact'))
        service.graphiti.get_edge_invalid_at = AsyncMock(side_effect=RuntimeError('boom'))
        result = await service.update_edge(
            edge_uuid='e-1', fact='new fact', project_id='proj', clear_invalid_at=True
        )
        assert result.get('verified') is False
        error = result.get('verification_error', '')
        assert 'Readback fact mismatch' in error, (
            'the earlier fact-verification error must be preserved, not overwritten'
        )
        assert 'RuntimeError: boom' in error

    @pytest.mark.asyncio
    async def test_clear_invalid_at_alone_does_not_raise(self, service):
        """Guard must allow clear_invalid_at=True even when fact and invalid_at are None."""
        service.graphiti.update_edge = AsyncMock(
            return_value={'uuid': 'e-1', 'fact': 'f', 'refreshed_nodes': []}
        )
        # Should NOT raise ValueError
        await service.update_edge(edge_uuid='e-1', project_id='proj', clear_invalid_at=True)

    @pytest.mark.asyncio
    async def test_all_none_still_raises(self, service):
        """ValueError must still be raised when all three (fact/invalid_at/clear_invalid_at) are falsy."""
        with pytest.raises(ValueError, match='fact, invalid_at, or clear_invalid_at'):
            await service.update_edge(
                edge_uuid='e-1', project_id='proj',
                fact=None, invalid_at=None, clear_invalid_at=False,
            )

    @pytest.mark.asyncio
    async def test_journal_params_include_clear_invalid_at(self, service):
        """Journal params dict must include clear_invalid_at when True."""
        logged_params = {}

        async def fake_journaled_call(**kwargs):
            logged_params.update(kwargs.get('payload', {}))
            return {'uuid': 'e-1', 'fact': 'f', 'refreshed_nodes': []}

        service._journaled_backend_call = fake_journaled_call
        await service.update_edge(
            edge_uuid='e-1', project_id='proj', clear_invalid_at=True
        )
        assert logged_params.get('clear_invalid_at') is True, (
            f'clear_invalid_at must appear in journal params; got {logged_params}'
        )


# ---------------------------------------------------------------------------
# Step 1 (task-1940): GraphitiBackend.get_edge_invalid_at RED tests
# ---------------------------------------------------------------------------

class TestGraphitiBackendGetEdgeInvalidAt:
    """GraphitiBackend.get_edge_invalid_at(uuid, group_id) reads back the raw invalid_at property.

    Used by MemoryService.update_edge's clear_invalid_at verification (task 1940) to
    confirm an edge was actually restored to active status, independent of the
    fact-text readback used for the ``fact`` verification path.
    """

    @pytest.mark.asyncio
    async def test_returns_none_when_invalid_at_is_null(
        self, mock_config, make_backend, make_graph_mock
    ):
        """A [[None]] row (invalid_at IS NULL) must return None."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([[None]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_edge_invalid_at('edge-uuid-1', group_id='test')
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_raw_invalid_at_value(
        self, mock_config, make_backend, make_graph_mock
    ):
        """A row carrying a stored invalid_at must be returned verbatim (no parsing)."""
        backend = make_backend(mock_config)
        ts = '2026-06-01T00:00:00+00:00'
        graph = make_graph_mock([[ts]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_edge_invalid_at('edge-uuid-1', group_id='test')
        assert result == ts

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(
        self, mock_config, make_backend, make_graph_mock
    ):
        """get_edge_invalid_at is a read — it must use ro_query and never call graph.query."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([[None]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_edge_invalid_at('edge-uuid-1', group_id='test')
        graph.ro_query.assert_awaited_once()
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_passes_uuid_param_and_matches_relates_to(
        self, mock_config, make_backend, make_graph_mock
    ):
        """The Cypher matches RELATES_TO and returns e.invalid_at; uuid is passed as a param."""
        from _fm_helpers import extract_cypher, extract_params

        backend = make_backend(mock_config)
        graph = make_graph_mock([[None]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_edge_invalid_at('specific-edge-uuid', group_id='test')
        cypher = extract_cypher(graph.ro_query.call_args)
        params = extract_params(graph.ro_query.call_args)
        assert 'RELATES_TO' in cypher
        assert 'invalid_at' in cypher
        assert params == {'uuid': 'specific-edge-uuid'}

    @pytest.mark.asyncio
    async def test_raises_edge_not_found_when_missing(
        self, mock_config, make_backend, make_graph_mock
    ):
        """An empty result_set (no matching edge) must raise EdgeNotFoundError."""
        from graphiti_core.errors import EdgeNotFoundError

        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(EdgeNotFoundError):
            await backend.get_edge_invalid_at('missing-edge-uuid', group_id='test')


# ---------------------------------------------------------------------------
# Step 1: TRACK B.1 backend clear_invalid_at RED tests
# ---------------------------------------------------------------------------

class TestGraphitiBackendUpdateEdgeClearInvalidAt:
    """Backend update_edge must accept clear_invalid_at=True and set edge.invalid_at = None."""

    def _make_backend_and_edge(self, mock_config, make_graph_mock):
        """Set up a GraphitiBackend with mock driver/client, an awaitable graph, and a mock edge.

        Returns (backend, mock_edge, graph) — ``graph`` is the same object
        ``backend._graph_for(...)`` resolves to, so tests can assert on the
        explicit direct-Cypher clear write (task 1940) issued via
        ``graph.query(...)``.
        """
        from fused_memory.backends.graphiti_client import GraphitiBackend

        backend = GraphitiBackend(mock_config)
        mock_driver = MagicMock()
        mock_cloned_driver = MagicMock()
        mock_driver.clone = MagicMock(return_value=mock_cloned_driver)
        backend._driver = mock_driver
        backend.client = MagicMock()

        graph = make_graph_mock([])
        mock_driver._get_graph = MagicMock(return_value=graph)

        mock_edge = AsyncMock()
        mock_edge.source_node_uuid = 'src-uuid'
        mock_edge.target_node_uuid = 'tgt-uuid'
        mock_edge.uuid = 'e-1'
        mock_edge.fact = 'original fact'
        mock_edge.invalid_at = datetime(2026, 1, 1, tzinfo=UTC)
        mock_edge.save = AsyncMock()
        mock_edge.generate_embedding = AsyncMock()
        backend.refresh_entity_summary = AsyncMock(return_value={})
        return backend, mock_edge, graph

    @pytest.mark.asyncio
    async def test_clear_invalid_at_alone_sets_none(self, mock_config, make_graph_mock):
        """clear_invalid_at=True with no fact/invalid_at clears edge.invalid_at and calls save."""
        backend, mock_edge, _graph = self._make_backend_and_edge(mock_config, make_graph_mock)
        with patch(
            'fused_memory.backends.graphiti_client.EntityEdge'
        ) as MockEntityEdge:
            MockEntityEdge.get_by_uuid = AsyncMock(return_value=mock_edge)
            await backend.update_edge('e-1', group_id='proj', clear_invalid_at=True)
            assert mock_edge.invalid_at is None, (
                'clear_invalid_at=True must set edge.invalid_at to None'
            )
            mock_edge.save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_clear_invalid_at_with_fact_updates_both(self, mock_config, make_graph_mock):
        """clear_invalid_at=True combined with fact updates fact AND clears invalid_at."""
        backend, mock_edge, _graph = self._make_backend_and_edge(mock_config, make_graph_mock)
        with patch(
            'fused_memory.backends.graphiti_client.EntityEdge'
        ) as MockEntityEdge:
            MockEntityEdge.get_by_uuid = AsyncMock(return_value=mock_edge)
            await backend.update_edge('e-1', 'new fact', group_id='proj', clear_invalid_at=True)
            assert mock_edge.fact == 'new fact', (
                'fact must be updated when fact is provided'
            )
            assert mock_edge.invalid_at is None, (
                'invalid_at must be cleared when clear_invalid_at=True'
            )
            mock_edge.save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_clear_invalid_at_takes_precedence_over_invalid_at(self, mock_config, make_graph_mock):
        """When both invalid_at and clear_invalid_at=True are given, clear wins."""
        backend, mock_edge, _graph = self._make_backend_and_edge(mock_config, make_graph_mock)
        ts = datetime(2026, 6, 1, tzinfo=UTC)
        with patch(
            'fused_memory.backends.graphiti_client.EntityEdge'
        ) as MockEntityEdge:
            MockEntityEdge.get_by_uuid = AsyncMock(return_value=mock_edge)
            await backend.update_edge(
                'e-1', group_id='proj', invalid_at=ts, clear_invalid_at=True
            )
            assert mock_edge.invalid_at is None, (
                'clear_invalid_at=True must take precedence: invalid_at ends up None'
            )
            mock_edge.save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_clear_invalid_at_issues_explicit_null_cypher(self, mock_config, make_graph_mock):
        """clear_invalid_at=True must issue an explicit direct-Cypher NULL write.

        graphiti-core's FalkorDB map-based ``edge.save()`` does not reliably
        clear a null-valued property, so update_edge must force the clear via
        an explicit ``SET e.invalid_at = NULL`` Cypher write (task 1940).
        """
        from _fm_helpers import extract_cypher, extract_params

        backend, mock_edge, graph = self._make_backend_and_edge(mock_config, make_graph_mock)
        with patch(
            'fused_memory.backends.graphiti_client.EntityEdge'
        ) as MockEntityEdge:
            MockEntityEdge.get_by_uuid = AsyncMock(return_value=mock_edge)
            await backend.update_edge('e-1', group_id='proj', clear_invalid_at=True)

        graph.query.assert_awaited_once()
        cypher = extract_cypher(graph.query.call_args)
        params = extract_params(graph.query.call_args)
        assert 'RELATES_TO' in cypher
        assert 'SET' in cypher
        assert 'invalid_at' in cypher
        assert 'NULL' in cypher
        assert params == {'uuid': 'e-1'}

    @pytest.mark.asyncio
    async def test_fact_plus_clear_still_issues_explicit_clear(self, mock_config, make_graph_mock):
        """The explicit clear Cypher must fire even when fact is also supplied."""
        backend, mock_edge, graph = self._make_backend_and_edge(mock_config, make_graph_mock)
        with patch(
            'fused_memory.backends.graphiti_client.EntityEdge'
        ) as MockEntityEdge:
            MockEntityEdge.get_by_uuid = AsyncMock(return_value=mock_edge)
            await backend.update_edge('e-1', 'new fact', group_id='proj', clear_invalid_at=True)

        graph.query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_explicit_clear_precedes_summary_refresh(self, mock_config, make_graph_mock):
        """The explicit clear Cypher must be awaited BEFORE the endpoint summary refresh runs.

        Otherwise refresh_entity_summary's get_valid_edges_for_node query (which
        filters ``WHERE e.invalid_at IS NULL``) would run before the clear lands
        and exclude the just-restored edge from the rebuilt summary.
        """
        backend, mock_edge, graph = self._make_backend_and_edge(mock_config, make_graph_mock)
        observed_await_counts = []

        async def _fake_refresh(node_uuid, *, group_id):
            observed_await_counts.append(graph.query.await_count)
            return {}

        backend.refresh_entity_summary = AsyncMock(side_effect=_fake_refresh)

        with patch(
            'fused_memory.backends.graphiti_client.EntityEdge'
        ) as MockEntityEdge:
            MockEntityEdge.get_by_uuid = AsyncMock(return_value=mock_edge)
            await backend.update_edge('e-1', group_id='proj', clear_invalid_at=True)

        assert observed_await_counts, 'refresh_entity_summary was never called'
        assert all(count > 0 for count in observed_await_counts), (
            'explicit clear Cypher must be awaited before summary refresh; '
            f'observed graph.query.await_count at refresh time: {observed_await_counts}'
        )


class TestGetMemoriesByMetadata:
    """MemoryService.get_memories_by_metadata delegates to mem0.scroll_by_metadata."""

    @pytest.mark.asyncio
    async def test_returns_scroll_result_unchanged(self, mock_config):
        """get_memories_by_metadata returns the list from mem0.scroll_by_metadata verbatim."""
        from fused_memory.services.memory_service import MemoryService

        svc = MemoryService(mock_config)
        expected = [
            {
                'id': 'a',
                'created_at': '2026-01-01T00:00:00+00:00',
                'metadata': {'recon_pool': 'stage2_cycle_summary'},
            }
        ]
        svc.mem0 = MagicMock()
        svc.mem0.scroll_by_metadata = AsyncMock(return_value=expected)

        result = await svc.get_memories_by_metadata(
            project_id='dark_factory',
            filters={'recon_pool': 'stage2_cycle_summary'},
        )

        assert result == expected

    @pytest.mark.asyncio
    async def test_calls_scroll_with_correct_scope_and_filters(self, mock_config):
        """get_memories_by_metadata builds Scope(project_id=...) and passes filters+limit."""
        from fused_memory.services.memory_service import MemoryService

        svc = MemoryService(mock_config)
        svc.mem0 = MagicMock()
        svc.mem0.scroll_by_metadata = AsyncMock(return_value=[])

        filters = {'recon_pool': 'stage2_cycle_summary'}
        await svc.get_memories_by_metadata(
            project_id='dark_factory',
            filters=filters,
            limit=42,
        )

        svc.mem0.scroll_by_metadata.assert_awaited_once()
        call_args = svc.mem0.scroll_by_metadata.call_args

        # scope must match project_id
        scope = call_args.args[0] if call_args.args else call_args.kwargs.get('scope')
        assert scope.project_id == 'dark_factory'

        # filters forwarded
        passed_filters = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get('filters')
        assert passed_filters == filters

        # limit forwarded
        passed_limit = call_args.args[2] if len(call_args.args) > 2 else call_args.kwargs.get('limit')
        assert passed_limit == 42

    @pytest.mark.asyncio
    async def test_does_not_use_semantic_search(self, mock_config):
        """get_memories_by_metadata must NOT call mem0.search (semantic search path)."""
        from fused_memory.services.memory_service import MemoryService

        svc = MemoryService(mock_config)
        svc.mem0 = MagicMock()
        svc.mem0.search = AsyncMock(return_value={'results': []})
        svc.mem0.scroll_by_metadata = AsyncMock(return_value=[])

        await svc.get_memories_by_metadata(
            project_id='dark_factory',
            filters={'recon_pool': 'stage2_cycle_summary'},
        )

        svc.mem0.search.assert_not_awaited()


# ---------------------------------------------------------------------------
# Task 1752: get_status uptime fields
# ---------------------------------------------------------------------------

class TestGetStatusUptime:
    """get_status() reports server started_at and uptime_seconds."""

    @pytest.mark.asyncio
    async def test_started_at_present_and_utc(self, service):
        """get_status() returns a top-level 'started_at' UTC ISO-format string."""
        from datetime import timedelta
        service.graphiti.list_graphs = AsyncMock(return_value=[])
        service.mem0.list_projects = AsyncMock(return_value=[])

        result = await service.get_status()

        assert 'started_at' in result, (
            f'Expected "started_at" in result; got keys={list(result)}'
        )
        dt = datetime.fromisoformat(result['started_at'])
        assert dt.utcoffset() == timedelta(0), (
            f'started_at must be UTC; utcoffset={dt.utcoffset()}'
        )
        assert result['started_at'] == service._started_at.isoformat(), (
            f'started_at should match construction-time stamp; '
            f'got {result["started_at"]!r}, expected {service._started_at.isoformat()!r}'
        )

    @pytest.mark.asyncio
    async def test_uptime_seconds_monotonic_int(self, service):
        """get_status() returns 'uptime_seconds' as a non-decreasing int."""
        # Override the monotonic baseline so we control the delta precisely
        service._start_monotonic = 1000.0
        service.graphiti.list_graphs = AsyncMock(return_value=[])
        service.mem0.list_projects = AsyncMock(return_value=[])

        with patch('fused_memory.services.memory_service.time.monotonic',
                   side_effect=[1005.4, 1010.9]):
            result1 = await service.get_status()
            result2 = await service.get_status()

        assert 'uptime_seconds' in result1, (
            f'Expected "uptime_seconds" in result; got keys={list(result1)}'
        )
        assert isinstance(result1['uptime_seconds'], int), (
            f'uptime_seconds must be int; got {type(result1["uptime_seconds"])}'
        )
        assert result1['uptime_seconds'] == 5, (
            f'Expected uptime_seconds=5 (int(1005.4-1000.0)); got {result1["uptime_seconds"]}'
        )
        assert result2['uptime_seconds'] == 10, (
            f'Expected uptime_seconds=10 (int(1010.9-1000.0)); got {result2["uptime_seconds"]}'
        )
        assert result2['uptime_seconds'] >= result1['uptime_seconds'], (
            f'uptime_seconds must be non-decreasing; '
            f'first={result1["uptime_seconds"]}, second={result2["uptime_seconds"]}'
        )


# ---------------------------------------------------------------------------
# step-11 (RED) / step-12 (GREEN): SearchResults in-band degrade marker
# step-13 (RED) / step-14 (GREEN): journal logs success=False when degraded
# ---------------------------------------------------------------------------


class TestSearchResultsDegradeMarker:
    """search() must return a SearchResults list-subclass carrying .degraded/.failed_stores.

    When a selected store raises, results are still returned (best-effort) but
    .degraded is True and the failing store name appears in .failed_stores.

    RED until step-12 defines SearchResults and wires it into search().
    """

    @pytest.mark.asyncio
    async def test_store_raise_sets_degraded_true(self, service):
        """When mem0.search raises, results.degraded is True and 'mem0' in results.failed_stores."""
        service.mem0.search = AsyncMock(side_effect=RuntimeError('mem0 boom'))

        results = await service.search(
            query='x', project_id='test', stores=['mem0']
        )

        # Must still be list-like
        assert hasattr(results, '__len__'), 'results must be list-like'

        degraded = getattr(results, 'degraded', None)
        assert degraded is True, (
            f'Expected results.degraded is True when store raises, got {degraded!r}. '
            'RED: search() returns a plain list with no .degraded attribute.'
        )

        failed = getattr(results, 'failed_stores', None)
        assert failed is not None, (
            'Expected results.failed_stores to exist, got None. '
            'RED: search() returns a plain list with no .failed_stores attribute.'
        )
        assert 'mem0' in failed, (
            f"Expected 'mem0' in results.failed_stores, got {failed!r}."
        )

    @pytest.mark.asyncio
    async def test_clean_search_sets_degraded_false(self, service):
        """When all stores succeed, results.degraded is False and failed_stores == []."""
        results = await service.search(
            query='x', project_id='test', stores=['mem0']
        )

        degraded = getattr(results, 'degraded', None)
        assert degraded is False, (
            f'Expected results.degraded is False on clean search, got {degraded!r}. '
            'RED: search() returns a plain list with no .degraded attribute.'
        )

        failed = getattr(results, 'failed_stores', None)
        assert failed == [], (
            f'Expected results.failed_stores == [] on clean search, got {failed!r}.'
        )


class TestSearchJournalSuccessFlagOnDegrade:
    """search() must log success=False to the journal when a store fails.

    RED until step-14 changes the hardcoded success=True to success=not degraded.
    """

    @pytest.mark.asyncio
    async def test_journal_logs_failure_when_store_raises(self, service):
        """When mem0 raises, journal.log_write_op is called with success=False."""
        service.mem0.search = AsyncMock(side_effect=RuntimeError('mem0 boom'))

        mock_journal = AsyncMock()
        mock_journal.log_write_op = AsyncMock()
        service._write_journal = mock_journal

        await service.search(
            query='x', project_id='test', stores=['mem0'],
            causation_id='run-123',
        )

        mock_journal.log_write_op.assert_called_once()
        call_kwargs = mock_journal.log_write_op.call_args.kwargs
        assert call_kwargs.get('success') is False, (
            f"Expected journal.log_write_op called with success=False when store raises, "
            f"got success={call_kwargs.get('success')!r}. "
            "RED: success is hardcoded to True."
        )

    @pytest.mark.asyncio
    async def test_journal_logs_success_true_on_clean_search(self, service):
        """When all stores succeed, journal.log_write_op is called with success=True."""
        mock_journal = AsyncMock()
        mock_journal.log_write_op = AsyncMock()
        service._write_journal = mock_journal

        await service.search(
            query='x', project_id='test', stores=['mem0'],
            causation_id='run-456',
        )

        mock_journal.log_write_op.assert_called_once()
        call_kwargs = mock_journal.log_write_op.call_args.kwargs
        assert call_kwargs.get('success') is True, (
            f"Expected success=True on clean search, got {call_kwargs.get('success')!r}."
        )
