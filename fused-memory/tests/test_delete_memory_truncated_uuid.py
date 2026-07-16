"""Regression test pinning delete_memory's silent no-op on truncated UUIDs.

stage1.py's "UUID Resolution Discipline" section (reconciliation/prompts/stage1.py
~line 101-103) tells agents to *never* construct `delete_memory` ids from
truncated 8-char hex prefixes (e.g. search-result snippet prefixes like
'2531b4d8') because Graphiti returns `{status: deleted}` and silently no-ops
instead of erroring — providing no signal that nothing was actually deleted.

That claim was previously unpinned by any test (task 1175's deliverable never
landed on main — see task 2673, the clean refile). This file pins the
documented behavior at two levels:

1. `GraphitiBackend.remove_edge` — when the driver has no edge matching the
   given uuid (as is always the case for a truncated 8-char prefix, since it
   can never equal a full 36-char UUID primary key), `EntityEdge.get_by_uuid`
   raises `EdgeNotFoundError`, which `remove_edge` swallows and treats as
   "already deleted" — it returns `None` without raising and without calling
   `edge.delete`.
2. `MemoryService.delete_memory(store='graphiti')` — built on top of
   `remove_edge`, it returns `{'status': 'deleted', ...}` for a truncated
   prefix exactly as it would for a real, successfully-deleted edge UUID.

If `delete_memory`/`remove_edge` is ever hardened to raise on a truncated or
otherwise-nonexistent UUID, these tests will fail and stage1.py's UUID
Resolution Discipline section must be updated to match the new behavior.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.services.memory_service import MemoryService

# A search-result snippet prefix, per stage1.py's own example — 8 hex chars,
# nowhere near a valid 36-char UUID.
TRUNCATED_PREFIX = '2531b4d8'


class TestGraphitiBackendRemoveEdgeTruncatedUuid:
    @pytest.mark.asyncio
    async def test_remove_edge_silently_no_ops_on_truncated_prefix(self, mock_config):
        """remove_edge(truncated_prefix) must not raise, and must not call
        edge.delete — EdgeNotFoundError from the failed lookup is swallowed."""
        from graphiti_core.errors import EdgeNotFoundError

        from fused_memory.backends.graphiti_client import GraphitiBackend

        backend = GraphitiBackend(mock_config)
        mock_driver = MagicMock()
        mock_cloned_driver = MagicMock()
        mock_driver.clone = MagicMock(return_value=mock_cloned_driver)
        backend._driver = mock_driver
        backend.client = MagicMock()

        with patch(
            'fused_memory.backends.graphiti_client.EntityEdge'
        ) as MockEntityEdge:
            MockEntityEdge.get_by_uuid = AsyncMock(
                side_effect=EdgeNotFoundError(TRUNCATED_PREFIX)
            )

            # Must not raise.
            result = await backend.remove_edge(TRUNCATED_PREFIX, group_id='test')

            assert result is None
            MockEntityEdge.get_by_uuid.assert_called_once_with(
                mock_cloned_driver, TRUNCATED_PREFIX
            )


class TestDeleteMemoryTruncatedUuid:
    @pytest.fixture
    def service(self, mock_config):
        """MemoryService with a graphiti mock that reproduces the real
        backend's no-op behavior: remove_edge on an unmatched uuid returns
        None without raising (see GraphitiBackend.remove_edge's
        EdgeNotFoundError → return None path)."""
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        svc.graphiti.remove_edge = AsyncMock(return_value=None)
        return svc

    @pytest.mark.asyncio
    async def test_delete_memory_truncated_prefix_returns_deleted_without_raising(
        self, service
    ):
        """delete_memory(store='graphiti') with a truncated 8-char hex prefix
        must return the same {'status': 'deleted', ...} shape it would for a
        real edge uuid — no exception, no error field — pinning the silent
        no-op stage1.py warns agents about."""
        result = await service.delete_memory(
            memory_id=TRUNCATED_PREFIX, store='graphiti', project_id='test'
        )

        assert result == {
            'status': 'deleted',
            'store': 'graphiti',
            'id': TRUNCATED_PREFIX,
        }
        service.graphiti.remove_edge.assert_called_once_with(
            TRUNCATED_PREFIX, group_id='test'
        )
