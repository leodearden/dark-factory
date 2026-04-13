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

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import (
    EdgeDict,
    StaleSummaryResult,
)


def _make_svc(mock_config):
    """Service with mocked graphiti backend — mirrors helper in test_rebuild_entity_summaries.py."""
    from fused_memory.services.memory_service import MemoryService
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.mem0 = MagicMock()
    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    return svc


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
        backend.detect_stale_with_edges = AsyncMock(return_value=detect_result)

        returned = await backend.detect_stale_summaries(group_id='t')

        assert returned is stale_list
        backend.detect_stale_with_edges.assert_awaited_once_with(group_id='t')

    @pytest.mark.asyncio
    async def test_detect_stale_summaries_empty_stale_list(self, mock_config, make_backend):
        """detect_stale_summaries returns [] when no stale entities exist.

        Documents that the function makes no non-empty assumption: it simply
        returns result.stale regardless of length, so an empty list is a
        valid and expected return value.
        """
        backend = make_backend(mock_config)
        detect_result = StaleSummaryResult(stale=[], all_edges={}, total_count=0)
        backend.detect_stale_with_edges = AsyncMock(return_value=detect_result)

        returned = await backend.detect_stale_summaries(group_id='t')

        assert returned is detect_result.stale
        backend.detect_stale_with_edges.assert_awaited_once_with(group_id='t')



class TestDetectStaleSummariesWithEdgesNamedAccess:
    """detect_stale_with_edges returns a fully-populated StaleSummaryResult with named access."""

    @pytest.mark.asyncio
    async def test_detect_stale_summaries_includes_summary_field_named_access(
        self, mock_config, make_backend, make_edge_backend
    ):
        """Confirms named attribute access on StaleSummaryResult fields works correctly
        for the summary-field scenario. Verifies that result.stale contains summary data,
        result.all_edges maps entity UUIDs to edge lists, and result.total_count reflects
        the scanned entity count.
        """
        original_summary = 'old stale fact'
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': original_summary},
        ], edges={
            'uuid-1': [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}],
        })

        result = await backend.detect_stale_with_edges(group_id='test')

        assert len(result.stale) == 1
        assert 'summary' in result.stale[0]
        assert result.stale[0]['summary'] == original_summary
        assert result.all_edges == {'uuid-1': [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}]}
        assert result.total_count == 1


class TestRebuildEntitySummariesNamedAccess:
    """MemoryService.rebuild_entity_summaries uses named attribute access on StaleSummaryResult."""

    @pytest.mark.asyncio
    async def test_rebuild_entity_summaries_named_access(self, mock_config):
        """rebuild_entity_summaries routes per-entity edges from result.all_edges correctly."""
        svc = _make_svc(mock_config)
        stale_list = [{'uuid': 'u1', 'name': 'A', 'summary': 'old', 'duplicate_count': 0, 'stale_line_count': 1, 'valid_fact_count': 1, 'summary_line_count': 1}]
        per_entity_edges: list[EdgeDict] = [{'uuid': 'e-1', 'fact': 'new', 'name': 'knows'}]
        all_edges: dict[str, list[EdgeDict]] = {'u1': per_entity_edges}

        detect_result = StaleSummaryResult(
            stale=stale_list,
            all_edges=all_edges,
            total_count=5,
        )
        svc.graphiti.detect_stale_with_edges = AsyncMock(return_value=detect_result)
        svc.graphiti.rebuild_entity_from_edges = AsyncMock(return_value={
            'uuid': 'u1',
            'name': 'A',
            'old_summary': 'old',
            'new_summary': 'A: new',
            'edge_count': 1,
        })

        result = await svc.rebuild_entity_summaries(project_id='test')

        # total_entities flows from total_count (named access on result)
        assert result['total_entities'] == 5
        assert result['stale_entities'] == 1
        assert result['rebuilt'] == 1
        assert result['errors'] == 0

        # per-entity edge list comes from all_edges['u1']
        svc.graphiti.rebuild_entity_from_edges.assert_awaited_once_with(
            'u1', 'A', per_entity_edges,
            group_id='test',
            old_summary='old',
        )
