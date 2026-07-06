"""Tests for rename_entity across backend, service, and MCP tool layers.

rename_entity_node deterministically corrects a mis-cased/mis-pluralized
task-entity node name (e.g. 'task 132' -> 'Task 132') minted by graphiti-core's
LLM entity extraction. It is the shared primitive behind both the write-path
post-add_episode normalization hook (MemoryService._normalize_task_node_names,
see test_memory_service.py) and the operator-facing rename_entity MCP tool
(task 2110) that corrects pre-existing bad nodes an episode may never re-touch.

Covers:
- GraphitiBackend.rename_entity_node()
- MemoryService.rename_entity()
- MCP tool rename_entity in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend, NodeNotFoundError

# ---------------------------------------------------------------------------
# step-3: GraphitiBackend.rename_entity_node
# ---------------------------------------------------------------------------


class TestGraphitiBackendRenameEntityNode:
    """GraphitiBackend.rename_entity_node(uuid, new_name, *, group_id) overwrites
    an Entity node's name property, then best-effort regenerates its name_embedding."""

    @pytest.mark.asyncio
    async def test_returns_result_dict(self, mock_config, make_backend):
        """(a) Returns dict with uuid, old_name, new_name; old_name comes from get_node_text."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('task 132', 'some summary'))
        backend.update_node_name = AsyncMock()
        backend.update_node_embedding = AsyncMock()
        backend.client.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])

        result = await backend.rename_entity_node('node-uuid-1', 'Task 132', group_id='test')

        assert result == {
            'uuid': 'node-uuid-1',
            'old_name': 'task 132',
            'new_name': 'Task 132',
        }

    @pytest.mark.asyncio
    async def test_calls_update_node_name_exactly_once(self, mock_config, make_backend):
        """(b) Awaits update_node_name(uuid, new_name, group_id=...) exactly once."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('task 132', ''))
        backend.update_node_name = AsyncMock()
        backend.update_node_embedding = AsyncMock()
        backend.client.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])

        await backend.rename_entity_node('node-uuid-1', 'Task 132', group_id='test')

        backend.update_node_name.assert_awaited_once_with(
            'node-uuid-1', 'Task 132', group_id='test'
        )

    @pytest.mark.asyncio
    async def test_regenerates_name_embedding(self, mock_config, make_backend):
        """(c) Awaits _require_client().embedder.create(new_name), then
        update_node_embedding(uuid, <embedding>, group_id=...)."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('task 132', ''))
        backend.update_node_name = AsyncMock()
        backend.update_node_embedding = AsyncMock()
        embedding = [0.1, 0.2, 0.3]
        backend.client.embedder.create = AsyncMock(return_value=embedding)

        await backend.rename_entity_node('node-uuid-1', 'Task 132', group_id='test')

        backend.client.embedder.create.assert_awaited_once_with('Task 132')
        backend.update_node_embedding.assert_awaited_once_with(
            'node-uuid-1', embedding, group_id='test'
        )

    @pytest.mark.asyncio
    async def test_node_not_found_propagates(self, mock_config, make_backend):
        """(d) NodeNotFoundError from get_node_text propagates — a missing node
        is not a silent no-op."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(
            side_effect=NodeNotFoundError('Entity node not found: missing-uuid')
        )
        backend.update_node_name = AsyncMock()
        backend.update_node_embedding = AsyncMock()

        with pytest.raises(NodeNotFoundError):
            await backend.rename_entity_node('missing-uuid', 'Task 132', group_id='test')

        backend.update_node_name.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embedder_failure_is_swallowed_best_effort(self, mock_config, make_backend):
        """(e) An embedder.create failure does not propagate — the rename still
        returns its result dict, and update_node_name still ran."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('task 132', ''))
        backend.update_node_name = AsyncMock()
        backend.update_node_embedding = AsyncMock()
        backend.client.embedder.create = AsyncMock(
            side_effect=RuntimeError('embedder unavailable')
        )

        result = await backend.rename_entity_node('node-uuid-1', 'Task 132', group_id='test')

        assert result == {
            'uuid': 'node-uuid-1',
            'old_name': 'task 132',
            'new_name': 'Task 132',
        }
        backend.update_node_name.assert_awaited_once_with(
            'node-uuid-1', 'Task 132', group_id='test'
        )
        backend.update_node_embedding.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when backend has no driver (not initialized)."""
        backend = GraphitiBackend(mock_config)  # _driver is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.rename_entity_node('node-uuid-1', 'Task 132', group_id='test')


# ---------------------------------------------------------------------------
# step-9: MemoryService.rename_entity
# ---------------------------------------------------------------------------


class TestMemoryServiceRenameEntity:
    """MemoryService.rename_entity() delegates to the graphiti backend.

    Mirrors TestMemoryServiceSetEntitySummary (test_set_entity_summary.py) —
    same identifier-resolution, validation, and try/finally journal-logging
    contracts, applied to the rename-node operational escape hatch.
    """

    @pytest.fixture
    def service(self, mock_config):
        """MemoryService with mocked backends (no real DB needed)."""
        from fused_memory.services.memory_service import MemoryService
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        svc.graphiti.rename_entity_node = AsyncMock(
            return_value={
                'uuid': 'node-1',
                'old_name': 'task 132',
                'new_name': 'Task 132',
            }
        )
        svc.mem0 = MagicMock()
        svc.durable_queue = MagicMock()
        svc.durable_queue.enqueue = AsyncMock(return_value=1)
        return svc

    @pytest.mark.asyncio
    async def test_delegates_to_graphiti_backend(self, service):
        """Delegates to graphiti.rename_entity_node(uuid, new_name, group_id=...) and
        returns the backend dict unchanged."""
        result = await service.rename_entity(
            entity_uuid='node-1',
            new_name='Task 132',
            project_id='dark_factory',
        )
        service.graphiti.rename_entity_node.assert_awaited_once_with(
            'node-1', 'Task 132', group_id='dark_factory'
        )
        assert result == {
            'uuid': 'node-1',
            'old_name': 'task 132',
            'new_name': 'Task 132',
        }

    @pytest.mark.asyncio
    async def test_resolves_entity_name_to_uuid(self, service):
        """(a) entity_name resolves via resolve_entity_by_name, then delegates with the
        resolved uuid."""
        service.graphiti.resolve_entity_by_name = AsyncMock(return_value='node-1')
        result = await service.rename_entity(
            entity_name='task 132',
            new_name='Task 132',
            project_id='dark_factory',
        )
        service.graphiti.resolve_entity_by_name.assert_awaited_once_with(
            'task 132', group_id='dark_factory'
        )
        service.graphiti.rename_entity_node.assert_awaited_once_with(
            'node-1', 'Task 132', group_id='dark_factory'
        )
        assert result['uuid'] == 'node-1'

    @pytest.mark.asyncio
    async def test_prefers_uuid_when_both_provided(self, service):
        """(b) entity_uuid takes precedence when both are supplied — resolve is never awaited."""
        service.graphiti.resolve_entity_by_name = AsyncMock(return_value='other-node')
        await service.rename_entity(
            entity_uuid='node-1',
            entity_name='task 132',
            new_name='Task 132',
            project_id='dark_factory',
        )
        service.graphiti.resolve_entity_by_name.assert_not_awaited()
        service.graphiti.rename_entity_node.assert_awaited_once_with(
            'node-1', 'Task 132', group_id='dark_factory'
        )

    @pytest.mark.asyncio
    async def test_raises_value_error_when_neither_identifier_provided(self, service):
        """(c) Raises ValueError when neither entity_uuid nor entity_name is provided."""
        with pytest.raises(ValueError, match='entity_uuid.*entity_name|entity_name.*entity_uuid'):
            await service.rename_entity(new_name='Task 132', project_id='dark_factory')

    @pytest.mark.asyncio
    async def test_raises_value_error_when_new_name_is_none(self, service):
        """(d) Raises ValueError when new_name is None (missing/omitted)."""
        with pytest.raises(ValueError, match='new_name'):
            await service.rename_entity(entity_uuid='node-1', project_id='dark_factory')

    @pytest.mark.asyncio
    async def test_raises_value_error_when_new_name_is_empty(self, service):
        """(d) Raises ValueError when new_name is an empty string."""
        with pytest.raises(ValueError, match='new_name'):
            await service.rename_entity(entity_uuid='node-1', new_name='', project_id='dark_factory')

    @pytest.mark.asyncio
    async def test_journal_logs_operation_and_project_id(self, service):
        """(e) log_write_op is awaited with operation='rename_entity' and project_id set."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.rename_entity(
            entity_uuid='node-1',
            new_name='Task 132',
            project_id='dark_factory',
            agent_id='test-agent',
        )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('operation') == 'rename_entity'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_journal_logs_entity_name_param_when_resolved(self, service):
        """(e) params include entity_name when the name-resolution path is used."""
        service.graphiti.resolve_entity_by_name = AsyncMock(return_value='node-1')
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.rename_entity(
            entity_name='task 132',
            new_name='Task 132',
            project_id='dark_factory',
        )
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs['params'].get('entity_name') == 'task 132'

    @pytest.mark.asyncio
    async def test_journal_failure_does_not_mask_successful_result(self, service):
        """(f) A journal.log_write_op failure must not mask an otherwise-successful result."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock(side_effect=RuntimeError('journal db is full'))
        service.set_write_journal(mock_journal)
        expected = {'uuid': 'node-1', 'old_name': 'task 132', 'new_name': 'Task 132'}
        service.graphiti.rename_entity_node = AsyncMock(return_value=expected)

        result = await service.rename_entity(
            entity_uuid='node-1', new_name='Task 132', project_id='dark_factory'
        )
        assert result == expected

    @pytest.mark.asyncio
    async def test_backend_failure_still_journals_success_false(self, service):
        """(f) A backend exception still journals success=False with the error text, and
        the exception still propagates to the caller."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        service.graphiti.rename_entity_node = AsyncMock(side_effect=ValueError('FalkorDB timeout'))

        with pytest.raises(ValueError, match='FalkorDB timeout'):
            await service.rename_entity(
                entity_uuid='node-1', new_name='Task 132', project_id='dark_factory'
            )

        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('success') is False
        assert call_kwargs.get('error') is not None
        assert 'FalkorDB timeout' in call_kwargs['error']
