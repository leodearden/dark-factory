"""Regression test pinning delete_memory's silent no-op on truncated UUIDs.

stage1.py's "UUID Resolution Discipline" section (reconciliation/prompts/stage1.py
~line 101-103) tells agents to *never* construct `delete_memory` ids from
truncated 8-char hex prefixes (e.g. search-result snippet prefixes like
'2531b4d8') because Graphiti returns `{status: deleted}` and silently no-ops
instead of erroring — providing no signal that nothing was actually deleted.

That claim was previously unpinned by any test (task 1175's deliverable never
landed on main — see task 2673, the clean refile). This file pins the
lower-level half of the documented behavior:

`GraphitiBackend.remove_edge` — when the driver has no edge matching the
given uuid (as is always the case for a truncated 8-char prefix, since it can
never equal a full 36-char UUID primary key — see the length invariant
below), `EntityEdge.get_by_uuid` raises `EdgeNotFoundError`, which
`remove_edge` swallows and treats as "already deleted" — it returns `None`
without raising and without calling `edge.delete`.

The upper-level half — `MemoryService.delete_memory(store='graphiti')`
returning `{'status': 'deleted', ...}` regardless of whether `remove_edge`
actually found anything — is already pinned in test_memory_service.py's
TestDeleteMemory (test_delete_graphiti's status/store assertions,
test_delete_memory_graphiti_calls_remove_edge's remove_edge(memory_id,
group_id=...) call assertion). delete_memory is a pure passthrough over
remove_edge's return value and has no awareness of whether the id it was
given was truncated, so those existing tests plus the not-found-branch test
below jointly cover the full silent-no-op path without duplicating
assertions across files.

If `delete_memory`/`remove_edge` is ever hardened to raise on a truncated or
otherwise-nonexistent UUID, this test will fail and stage1.py's UUID
Resolution Discipline section must be updated to match the new behavior.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# A search-result snippet prefix, per stage1.py's own example — 8 hex chars,
# nowhere near a valid 36-char UUID.
TRUNCATED_PREFIX = '2531b4d8'

# Invariant that makes TRUNCATED_PREFIX a reliable not-found trigger: a
# truncated hex prefix can never equal a full 36-char UUID primary key, so a
# real driver lookup keyed on it always misses and get_by_uuid always raises
# EdgeNotFoundError. The test below injects that error directly rather than
# exercising a real driver lookup — this assertion documents *why* the
# injected error is representative of what a truncated prefix actually
# causes in production, rather than pinning behavior for an arbitrary string.
assert len(TRUNCATED_PREFIX) != 36


class TestGraphitiBackendRemoveEdgeTruncatedUuid:
    """Companion to TestGraphitiBackendRemoveEdge in test_memory_service.py
    (currently :1659), which covers remove_edge's happy path (get_by_uuid
    finds an edge, edge.delete is called). This class covers the not-found
    branch. These would ideally be co-located as methods on one class —
    left separate here because test_memory_service.py is outside this
    task's locked file scope (task 2673 amendment pass); re-surface as a
    follow-up if this file is touched again.
    """

    @pytest.mark.asyncio
    async def test_remove_edge_silently_no_ops_on_truncated_prefix(self, mock_config):
        """remove_edge(truncated_prefix) must not raise, and must not call
        edge.delete — EdgeNotFoundError from the failed lookup is swallowed.

        This pins the generic EdgeNotFoundError -> return-None swallow
        branch by injecting the error directly into get_by_uuid, rather
        than exercising a real driver lookup keyed on TRUNCATED_PREFIX. The
        module-level length invariant above documents why a truncated
        prefix reliably drives this exact branch in production.
        """
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


# NOTE (task 2673 amendment pass): a second class here previously duplicated
# MemoryService.delete_memory(store='graphiti') coverage that already exists
# in test_memory_service.py's TestDeleteMemory (test_delete_graphiti's
# status/store assertions, test_delete_memory_graphiti_calls_remove_edge's
# remove_edge(memory_id, group_id=...) call assertion). delete_memory is a
# pure passthrough over remove_edge's return value, so it has no awareness
# of whether the id it was given was truncated — swapping in
# TRUNCATED_PREFIX exercised the same wrapper contract as those existing
# tests with a different string literal, not new behavior. Dropped rather
# than folded into test_memory_service.py, which is outside this task's
# locked file scope; see the module docstring above for how the remaining
# coverage jointly pins the full path.
