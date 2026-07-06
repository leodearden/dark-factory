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
