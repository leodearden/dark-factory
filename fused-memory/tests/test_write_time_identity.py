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


# ---------------------------------------------------------------------------
# step-5/6: GraphitiBackend._resolve_or_create_entity — >=2-match collapse path
# ---------------------------------------------------------------------------

class TestResolveOrCreateEntityCollapse:
    """GraphitiBackend._resolve_or_create_entity(name, *, group_id) — >=2-match
    collapse path: folds every non-survivor duplicate into the survivor-first
    match from find_duplicate_entity_nodes via merge_entities, mirroring the
    TestMergeEntities.backend_with_mocks orchestration-mock pattern
    (test_merge_entities.py:154)."""

    @pytest.fixture
    def backend_with_mocks(self, mock_config, make_backend):
        """GraphitiBackend with get_nodes_by_exact_name/find_duplicate_entity_nodes/
        merge_entities mocked as AsyncMocks for orchestration-only testing.

        get_nodes_by_exact_name defaults to 3 same-name matches (>=2 branch).
        find_duplicate_entity_nodes returns them survivor-first (edge_count
        DESC, created_at ASC, uuid ASC per its own contract): 'surv' has the
        most edges, 'dup1'/'dup2' are the duplicates to fold in.
        """
        backend = make_backend(mock_config)
        backend.get_nodes_by_exact_name = AsyncMock(return_value=[
            {'uuid': 'surv', 'name': 'Foo', 'summary': '', 'labels': []},
            {'uuid': 'dup1', 'name': 'Foo', 'summary': '', 'labels': []},
            {'uuid': 'dup2', 'name': 'Foo', 'summary': '', 'labels': []},
        ])
        backend.find_duplicate_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'surv', 'created_at': 1, 'edge_count': 5},
            {'uuid': 'dup1', 'created_at': 2, 'edge_count': 1},
            {'uuid': 'dup2', 'created_at': 3, 'edge_count': 0},
        ])
        backend.merge_entities = AsyncMock()
        return backend

    @pytest.mark.asyncio
    async def test_merges_each_non_survivor_into_survivor(self, backend_with_mocks):
        """merge_entities is awaited once per non-survivor duplicate, with
        deprecated_uuid=that dup's uuid, surviving_uuid='surv' (the
        find_duplicate_entity_nodes[0] survivor), group_id=the caller's
        group_id. Foreign/other uuids are never passed."""
        backend = backend_with_mocks
        await backend._resolve_or_create_entity('Foo', group_id='test')
        assert backend.merge_entities.await_count == 2  # len(dups) - 1
        merged_deprecated_uuids = set()
        for call in backend.merge_entities.await_args_list:
            args, kwargs = call
            deprecated_uuid, surviving_uuid = args[0], args[1]
            merged_deprecated_uuids.add(deprecated_uuid)
            assert surviving_uuid == 'surv'
            assert kwargs.get('group_id') == 'test'
        assert merged_deprecated_uuids == {'dup1', 'dup2'}

    @pytest.mark.asyncio
    async def test_returns_survivor_uuid(self, backend_with_mocks):
        """Returns the survivor's uuid (find_duplicate_entity_nodes[0])."""
        backend = backend_with_mocks
        result = await backend._resolve_or_create_entity('Foo', group_id='test')
        assert result == 'surv'

    @pytest.mark.asyncio
    async def test_idempotent_second_call_does_not_remerge(self, backend_with_mocks):
        """Idempotency: once collapsed to a single survivor, a subsequent call
        (reflecting post-merge reality — get_nodes_by_exact_name now returns
        only the survivor) resolves via the 1-match fast path and does NOT
        invoke merge_entities again."""
        backend = backend_with_mocks
        first = await backend._resolve_or_create_entity('Foo', group_id='test')
        assert first == 'surv'
        assert backend.merge_entities.await_count == 2

        # Simulate the post-collapse graph state: only the survivor remains.
        backend.get_nodes_by_exact_name.return_value = [
            {'uuid': 'surv', 'name': 'Foo', 'summary': '', 'labels': []},
        ]
        second = await backend._resolve_or_create_entity('Foo', group_id='test')
        assert second == 'surv'
        assert backend.merge_entities.await_count == 2  # unchanged — no re-merge
