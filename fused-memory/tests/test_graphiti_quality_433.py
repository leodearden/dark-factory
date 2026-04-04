"""Tests for code-quality improvements to graphiti_client.py rebuild/refresh pipeline.

Task 433: 8 code-quality improvements deferred from task-419 review.
- StaleSummaryResult NamedTuple (steps 1-2)
- _canonical_facts() @staticmethod (steps 3-4)
- refresh_entity_summary optional name/old_summary params (steps 5-6)
- rebuild_entity_summaries force+dry_run edge-fetch skip (steps 7-8)
- rebuild_entity_summaries variable scoping / data-flow clarity (steps 9-10)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import (
    GraphitiBackend,
    StaleSummaryResult,
)


# ---------------------------------------------------------------------------
# step-1: StaleSummaryResult named tuple with backward-compat tuple unpacking
# ---------------------------------------------------------------------------

class TestStaleSummaryResult:
    """StaleSummaryResult has named attrs and supports 3-tuple unpacking."""

    def test_named_attribute_stale(self):
        """StaleSummaryResult.stale holds the stale list."""
        stale_list = [{'uuid': 'u1', 'name': 'Alice'}]
        edges = {'u1': [{'fact': 'fact1'}]}
        result = StaleSummaryResult(stale=stale_list, edges=edges, total_count=5)
        assert result.stale is stale_list

    def test_named_attribute_edges(self):
        """StaleSummaryResult.edges holds the edges dict."""
        stale_list: list[dict] = []
        edges = {'u1': [{'fact': 'fact1'}]}
        result = StaleSummaryResult(stale=stale_list, edges=edges, total_count=3)
        assert result.edges is edges

    def test_named_attribute_total_count(self):
        """StaleSummaryResult.total_count holds the total entity count."""
        result = StaleSummaryResult(stale=[], edges={}, total_count=42)
        assert result.total_count == 42

    def test_tuple_unpacking_backward_compat(self):
        """StaleSummaryResult supports 3-tuple unpacking (backward compat)."""
        stale_list = [{'uuid': 'u1'}]
        edges = {'u1': []}
        result = StaleSummaryResult(stale=stale_list, edges=edges, total_count=7)
        a, b, c = result
        assert a is stale_list
        assert b is edges
        assert c == 7

    def test_is_tuple_subclass(self):
        """StaleSummaryResult IS a tuple — plain tuple equality works."""
        stale_list = [{'uuid': 'u1'}]
        edges: dict = {}
        result = StaleSummaryResult(stale=stale_list, edges=edges, total_count=1)
        # Plain tuple comparison (existing mock code uses this implicitly)
        assert isinstance(result, tuple)

    @pytest.mark.asyncio
    async def test_detect_stale_summaries_returns_named_result(self, mock_config, make_backend):
        """_detect_stale_summaries_with_edges returns StaleSummaryResult with named access."""
        backend = make_backend(mock_config)
        backend.list_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'u1', 'name': 'Alice', 'summary': 'stale summary'},
        ])
        backend.get_all_valid_edges = AsyncMock(return_value={
            'u1': [{'fact': 'fresh fact'}],
        })
        result = await backend._detect_stale_summaries_with_edges(group_id='test')

        # Named access
        assert isinstance(result, StaleSummaryResult)
        assert result.total_count == 1
        assert isinstance(result.stale, list)
        assert isinstance(result.edges, dict)

        # Positional (backward compat)
        stale, edges, total = result
        assert total == 1
        assert isinstance(stale, list)
        assert isinstance(edges, dict)
