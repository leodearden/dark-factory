"""Regression tests for detect_stale_summaries and rebuild_entity_summaries named-attribute access.

Task 438: Fix StaleSummaryResult positional unpacking at both call sites.

Locks in:
  (a) ``detect_stale_summaries`` uses named attribute access (``result.stale``).
  (b) ``rebuild_entity_summaries`` uses named attribute access and routes the
      per-entity edge list correctly from the ``all_edges`` field.

Structural field contract (all_edges, tuple unpacking) is covered in
test_graphiti_rebuild_pipeline.py::TestStaleSummaryResult. This file focuses on
integration behaviour of detect_stale_summaries and rebuild_entity_summaries.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.backends.graphiti_client import (
    EdgeDict,
    StaleSummaryResult,
)


class TestDetectStaleSummariesNamedAccess:
    """detect_stale_summaries uses result.stale (named access) not positional unpacking."""

    @pytest.mark.asyncio
    async def test_detect_stale_summaries_named_access(self, mock_config, make_backend):
        """detect_stale_summaries returns the stale list from a StaleSummaryResult."""
        backend = make_backend(mock_config)
        stale_list = [{'uuid': 'u1', 'name': 'Alice', 'summary': 'old', 'duplicate_count': 0, 'stale_line_count': 1, 'valid_fact_count': 1, 'summary_line_count': 1}]
        detect_result = StaleSummaryResult(
            stale=stale_list,
            all_edges={},
            total_count=3,
        )
        backend._detect_stale_summaries_with_edges = AsyncMock(return_value=detect_result)

        returned = await backend.detect_stale_summaries(group_id='t')

        assert returned is stale_list
        backend._detect_stale_summaries_with_edges.assert_awaited_once_with(group_id='t')

    @pytest.mark.asyncio
    async def test_detect_stale_summaries_empty_stale_list(self, mock_config, make_backend):
        """detect_stale_summaries returns [] when no stale entities exist.

        Documents that the function makes no non-empty assumption: it simply
        returns result.stale regardless of length, so an empty list is a
        valid and expected return value.
        """
        backend = make_backend(mock_config)
        detect_result = StaleSummaryResult(stale=[], all_edges={}, total_count=0)
        backend._detect_stale_summaries_with_edges = AsyncMock(return_value=detect_result)

        returned = await backend.detect_stale_summaries(group_id='t')

        assert returned is detect_result.stale
        backend._detect_stale_summaries_with_edges.assert_awaited_once_with(group_id='t')

    @pytest.mark.asyncio
    async def test_detect_stale_summaries_includes_summary_field_named_access(
        self, mock_config, make_backend
    ):
        """Confirms named attribute access on StaleSummaryResult fields works correctly
        for the summary-field scenario. Verifies that result.stale contains summary data,
        result.all_edges maps entity UUIDs to edge lists, and result.total_count reflects
        the scanned entity count.
        """
        backend = make_backend(mock_config)
        original_summary = 'old stale fact'
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': original_summary},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={
            'uuid-1': [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}],
        })

        result = await backend._detect_stale_summaries_with_edges(group_id='test')

        assert len(result.stale) == 1
        assert 'summary' in result.stale[0]
        assert result.stale[0]['summary'] == original_summary
        assert result.all_edges == {'uuid-1': [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}]}
        assert result.total_count == 1


class TestRebuildEntitySummariesNamedAccess:
    """rebuild_entity_summaries uses named attribute access on StaleSummaryResult."""

    @pytest.mark.asyncio
    async def test_rebuild_entity_summaries_named_access(self, mock_config, make_backend):
        """rebuild_entity_summaries routes per-entity edges from result.all_edges correctly."""
        backend = make_backend(mock_config)
        stale_list = [{'uuid': 'u1', 'name': 'A', 'summary': 'old', 'duplicate_count': 0, 'stale_line_count': 1, 'valid_fact_count': 1, 'summary_line_count': 1}]
        per_entity_edges: list[EdgeDict] = [{'uuid': 'e-1', 'fact': 'new', 'name': 'knows'}]
        all_edges: dict[str, list[EdgeDict]] = {'u1': per_entity_edges}

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
