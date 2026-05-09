"""Regression tests pinning the silent no-op behavior of delete_memory / remove_edge
on truncated 8-char hex UUID prefixes.

Claim under test (stage1.py:74-78):
    Calling delete_memory with a truncated 8-char hex prefix (e.g. '2531b4d8') instead
    of a full 36-char UUID returns {status: deleted} and silently no-ops — providing
    no error signal.  This claim anchors stage1.py's "canonical enforcement point for
    UUID resolution."

If EITHER test in this file fails, the claim in stage1.py:74-78 has drifted.  Update
stage1.py:74-78 to describe the new (hardened) behavior rather than patching the tests.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.errors import EdgeNotFoundError

from fused_memory.services.memory_service import MemoryService


# ---------------------------------------------------------------------------
# Local service fixture — mirrors test_memory_service.py:16-47
# ---------------------------------------------------------------------------


@pytest.fixture
def service(mock_config):
    """MemoryService with mocked backends (no real DB needed)."""
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.search = AsyncMock(return_value=[])
    svc.graphiti.search_nodes = AsyncMock(return_value=[])
    svc.graphiti.retrieve_episodes = AsyncMock(return_value=[])
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti.remove_episode = AsyncMock()
    svc.graphiti.remove_edge = AsyncMock(return_value=None)
    svc.graphiti.update_edge = AsyncMock(
        return_value={'uuid': 'test-uuid', 'fact': 'updated', 'refreshed_nodes': []}
    )
    svc.graphiti.get_edge_text = AsyncMock(return_value=('edge-name', 'updated'))
    svc.graphiti._require_client = MagicMock()

    svc.mem0 = MagicMock()
    svc.mem0.search = AsyncMock(return_value={'results': []})
    svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-1'}]})
    svc.mem0.get_all = AsyncMock(return_value={'results': []})
    svc.mem0.delete = AsyncMock(return_value={'message': 'deleted'})

    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    svc.durable_queue.enqueue_batch = AsyncMock(return_value=[1, 2, 3])
    svc.durable_queue.get_stats = AsyncMock(
        return_value={'counts': {}, 'oldest_pending_age_seconds': None}
    )
    svc.durable_queue.replay_dead = AsyncMock(return_value=0)
    svc.durable_queue.close = AsyncMock()
    return svc


# ---------------------------------------------------------------------------
# Service-layer pin
# ---------------------------------------------------------------------------


class TestDeleteMemoryTruncatedUUIDServiceLayer:
    @pytest.mark.asyncio
    async def test_delete_memory_truncated_8char_prefix_returns_deleted_status_silently(
        self, service
    ):
        """Regression pin for stage1.py:74-78.

        MemoryService.delete_memory called with a truncated 8-char hex prefix
        ('2531b4d8' — the exact example from stage1.py:74) must return exactly
        {'status': 'deleted', 'store': 'graphiti', 'id': '2531b4d8'} with no
        exception, even though no real edge was removed.

        The service never validates UUID format or inspects the backend's return
        value; it constructs the success dict unconditionally.  This is the
        observable contract stage1.py warns about.

        If this test fails, MemoryService.delete_memory has been hardened (UUID
        format validation, existence pre-check, or backend-return inspection).
        Update stage1.py:74-78 to describe the new behavior rather than patching
        this test.
        """
        result = await service.delete_memory(
            memory_id='2531b4d8', store='graphiti', project_id='test'
        )

        assert result == {'status': 'deleted', 'store': 'graphiti', 'id': '2531b4d8'}
        service.graphiti.remove_edge.assert_called_once_with('2531b4d8', group_id='test')


# ---------------------------------------------------------------------------
# Backend-layer pin
# ---------------------------------------------------------------------------


class TestGraphitiBackendRemoveEdgeTruncatedUUID:
    @pytest.mark.asyncio
    async def test_remove_edge_silently_noops_on_edge_not_found_for_truncated_uuid(
        self, mock_config
    ):
        """Regression pin for stage1.py:74-78.

        GraphitiBackend.remove_edge called with a truncated 8-char hex prefix
        ('2531b4d8' — the exact example from stage1.py:74) must swallow the
        EdgeNotFoundError raised by EntityEdge.get_by_uuid, return None, and
        never call edge.delete().

        This is the underlying mechanism that produces the service-layer silent
        success described in stage1.py:74-78: the try/except EdgeNotFoundError
        in remove_edge logs at INFO and returns early without re-raising.

        If this test fails, the try/except has been removed or changed to
        re-raise.  Update stage1.py:74-78 to reflect the new behavior rather
        than patching this test.
        """
        from fused_memory.backends.graphiti_client import GraphitiBackend

        backend = GraphitiBackend(mock_config)
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
            MockEntityEdge.get_by_uuid = AsyncMock(
                side_effect=EdgeNotFoundError('2531b4d8')
            )
            result = await backend.remove_edge('2531b4d8', group_id='test')

        assert result is None
        MockEntityEdge.get_by_uuid.assert_called_once_with(
            mock_cloned_driver, '2531b4d8'
        )
        mock_edge.delete.assert_not_called()
