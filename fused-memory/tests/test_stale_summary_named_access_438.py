"""Regression tests for StaleSummaryResult.all_edges field rename and named-access refactor.

Task 438: Fix StaleSummaryResult positional unpacking at both call sites.

Locks in:
  (a) The renamed field ``all_edges`` exists on StaleSummaryResult.
  (b) ``detect_stale_summaries`` uses named attribute access (``result.stale``).
  (c) ``rebuild_entity_summaries`` uses named attribute access and routes the
      per-entity edge list correctly from the renamed ``all_edges`` field.

Any accidental reversion of the ``edges`` → ``all_edges`` rename would break tests
(a) and (b) at NamedTuple construction time (unexpected keyword argument).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, call

import pytest

from fused_memory.backends.graphiti_client import (
    GraphitiBackend,
    StaleSummaryResult,
)


class TestStaleSummaryResultAllEdgesField:
    """StaleSummaryResult exposes the renamed ``all_edges`` field."""

    def test_stale_summary_result_has_all_edges_field(self):
        """StaleSummaryResult accepts and exposes the ``all_edges`` keyword argument."""
        result = StaleSummaryResult(stale=[], all_edges={}, total_count=0)
        assert result.all_edges == {}

    def test_all_edges_holds_per_entity_map(self):
        """all_edges stores the full graph-wide edge dict keyed by entity UUID."""
        edge_map = {'u1': [{'fact': 'Alice knows Bob'}]}
        result = StaleSummaryResult(stale=[], all_edges=edge_map, total_count=1)
        assert result.all_edges is edge_map

    def test_tuple_unpacking_still_works_after_rename(self):
        """NamedTuple backward-compat: positional unpacking still works after rename."""
        stale = [{'uuid': 'u1'}]
        edges = {'u1': []}
        result = StaleSummaryResult(stale=stale, all_edges=edges, total_count=4)
        a, b, c = result
        assert a is stale
        assert b is edges
        assert c == 4


class TestDetectStaleSummariesNamedAccess:
    """detect_stale_summaries uses result.stale (named access) not positional unpacking."""

    @pytest.mark.asyncio
    async def test_detect_stale_summaries_named_access(self, mock_config, make_backend):
        """detect_stale_summaries returns the stale list from a StaleSummaryResult."""
        backend = make_backend(mock_config)
        stale_list = [{'uuid': 'u1', 'name': 'Alice', 'summary': 'old'}]
        detect_result = StaleSummaryResult(
            stale=stale_list,
            all_edges={},
            total_count=3,
        )
        backend._detect_stale_summaries_with_edges = AsyncMock(return_value=detect_result)

        returned = await backend.detect_stale_summaries(group_id='t')

        assert returned == stale_list
        backend._detect_stale_summaries_with_edges.assert_awaited_once_with(group_id='t')


class TestRebuildEntitySummariesNamedAccess:
    """rebuild_entity_summaries uses named attribute access on StaleSummaryResult."""

    @pytest.mark.asyncio
    async def test_rebuild_entity_summaries_named_access(self, mock_config, make_backend):
        """rebuild_entity_summaries routes per-entity edges from result.all_edges correctly."""
        backend = make_backend(mock_config)
        stale_list = [{'uuid': 'u1', 'name': 'A', 'summary': 'old'}]
        per_entity_edges = [{'fact': 'new'}]
        all_edges = {'u1': per_entity_edges}

        detect_result = StaleSummaryResult(
            stale=stale_list,
            all_edges=all_edges,
            total_count=5,
        )
        backend._detect_stale_summaries_with_edges = AsyncMock(return_value=detect_result)
        backend._rebuild_entity_from_edges = AsyncMock(return_value={
            'uuid': 'u1',
            'name': 'A',
            'old_summary': 'old',
            'new_summary': 'A: new',
            'edge_count': 1,
        })

        result = await backend.rebuild_entity_summaries(group_id='test', force=False)

        # total_entities flows from total_count (named access on result)
        assert result['total_entities'] == 5
        assert result['stale_entities'] == 1
        assert result['rebuilt'] == 1
        assert result['errors'] == 0

        # per-entity edge list comes from all_edges['u1']
        backend._rebuild_entity_from_edges.assert_awaited_once_with(
            'u1', 'A', per_entity_edges,
            group_id='test',
            old_summary='old',
        )
