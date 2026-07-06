"""Tests for write-time identity primitives on GraphitiBackend (task 2198, W6-α).

Covers:
- GraphitiBackend._identity_lock_for (S2): per-group_id lazy asyncio.Lock registry,
  separate from DurableWriteQueue._group_locks.
- GraphitiBackend._resolve_or_create_entity (S1): exact-name resolve-or-collapse
  chokepoint — 0 matches is a no-op (None), 1 match resolves, >=2 collapses via
  find_duplicate_entity_nodes + merge_entities. group_id-scoped per the
  2026-07-06 amendment guarding task-2115's active cross-graph leak.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend

# ---------------------------------------------------------------------------
# step-1/2: GraphitiBackend._identity_lock_for
# ---------------------------------------------------------------------------

class TestIdentityLockFor:
    """GraphitiBackend._identity_lock_for(group_id) returns a per-group_id asyncio.Lock.

    Synchronous accessor (returns the Lock, not a coroutine); lazily creates and
    caches one Lock per group_id, mirroring DurableWriteQueue._group_locks
    (durable_queue.py:136, 259-260). Needs no initialized driver.
    """

    def test_returns_asyncio_lock(self, mock_config):
        """Returns an asyncio.Lock instance."""
        backend = GraphitiBackend(mock_config)
        lock = backend._identity_lock_for('g1')
        assert isinstance(lock, asyncio.Lock)

    def test_same_group_id_returns_same_lock(self, mock_config):
        """Two calls with the SAME group_id return the exact same object."""
        backend = GraphitiBackend(mock_config)
        a = backend._identity_lock_for('g1')
        b = backend._identity_lock_for('g1')
        assert a is b

    def test_different_group_ids_return_distinct_locks(self, mock_config):
        """Two calls with DIFFERENT group_ids return distinct objects."""
        backend = GraphitiBackend(mock_config)
        a = backend._identity_lock_for('g1')
        b = backend._identity_lock_for('g2')
        assert a is not b


# ---------------------------------------------------------------------------
# step-3/4: GraphitiBackend._resolve_or_create_entity — 0/1-match resolve path
# ---------------------------------------------------------------------------

class TestResolveOrCreateEntityResolve:
    """GraphitiBackend._resolve_or_create_entity(name, *, group_id) — 0/1-match
    resolve/no-op fast path (no collapse machinery), mirroring the
    TestMergeEntities.backend_with_mocks orchestration-mock pattern
    (test_merge_entities.py:154)."""

    @pytest.fixture
    def backend_with_mocks(self, mock_config, make_backend):
        """GraphitiBackend with get_nodes_by_exact_name/find_duplicate_entity_nodes/
        merge_entities mocked as AsyncMocks for orchestration-only testing."""
        backend = make_backend(mock_config)
        backend.get_nodes_by_exact_name = AsyncMock(return_value=[])
        backend.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        backend.merge_entities = AsyncMock()
        return backend

    @pytest.mark.asyncio
    async def test_single_match_resolves_without_collapse(self, backend_with_mocks):
        """Exactly one match: returns its uuid; neither collapse method runs."""
        backend = backend_with_mocks
        backend.get_nodes_by_exact_name.return_value = [
            {'uuid': 'u-1', 'name': 'Foo', 'summary': '', 'labels': []}
        ]
        result = await backend._resolve_or_create_entity('Foo', group_id='test')
        assert result == 'u-1'
        backend.find_duplicate_entity_nodes.assert_not_awaited()
        backend.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_zero_matches_returns_none_without_minting(self, backend_with_mocks):
        """Zero matches: returns None — documented no-op; minting stays
        graphiti_core's job, this primitive only resolves/collapses."""
        backend = backend_with_mocks
        backend.get_nodes_by_exact_name.return_value = []
        result = await backend._resolve_or_create_entity('Ghost', group_id='test')
        assert result is None
        backend.find_duplicate_entity_nodes.assert_not_awaited()
        backend.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_calls_get_nodes_by_exact_name_with_name_and_group_id(self, backend_with_mocks):
        """get_nodes_by_exact_name is awaited with the name and group_id."""
        backend = backend_with_mocks
        backend.get_nodes_by_exact_name.return_value = []
        await backend._resolve_or_create_entity('Foo', group_id='proj-x')
        backend.get_nodes_by_exact_name.assert_awaited_once_with('Foo', group_id='proj-x')
