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
