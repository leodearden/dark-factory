"""Tests for uuid-keyed dedup in GraphitiBackend read methods.

Covers:
- GraphitiBackend.get_valid_edges_for_node() — dedup keyed on e.uuid
- GraphitiBackend.get_all_valid_edges() — dedup keyed on (n.uuid, e.uuid)

Task 2213 (W6-zeta): replaces the WITH DISTINCT e / WITH DISTINCT n, e
graph-element-identity dedup idiom with plain uuid-keyed Python dedup, now
that RELATES_TO edge uuids are unique graph-wide (post tasks 2207/2210).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from _fm_helpers import extract_cypher, extract_params

# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.get_valid_edges_for_node
# ---------------------------------------------------------------------------


class TestGetValidEdgesForNodeUuidDedup:
    """GraphitiBackend.get_valid_edges_for_node(uuid, *, group_id) dedups by e.uuid."""

    @pytest.mark.asyncio
    async def test_cypher_has_no_with_distinct(self, mock_config, make_backend, make_graph_mock):
        """Emitted Cypher no longer relies on WITH DISTINCT (B7); read path unchanged."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_valid_edges_for_node('u', group_id='test')
        cypher = extract_cypher(graph.ro_query.call_args)
        assert 'WITH DISTINCT' not in cypher
        assert 'RELATES_TO' in cypher
        assert 'invalid_at IS NULL' in cypher
        params = extract_params(graph.ro_query.call_args)
        assert params.get('uuid') == 'u'
        graph.ro_query.assert_awaited_once()
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_self_loop_collapsed_by_uuid(self, mock_config, make_backend, make_graph_mock):
        """A self-loop's duplicate rows (same e.uuid) collapse to exactly one edge."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['e1', 'f', 'n'], ['e1', 'f', 'n']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_valid_edges_for_node('u', group_id='test')
        assert len(result) == 1
        assert result[0]['uuid'] == 'e1'

    @pytest.mark.asyncio
    async def test_distinct_uuid_edges_all_returned(self, mock_config, make_backend, make_graph_mock):
        """Rows with distinct e.uuid values are not over-collapsed."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['e1', 'f1', 'n1'], ['e2', 'f2', 'n2']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_valid_edges_for_node('u', group_id='test')
        assert len(result) == 2
        assert {e['uuid'] for e in result} == {'e1', 'e2'}

    @pytest.mark.asyncio
    async def test_null_fact_name_coerced(self, mock_config, make_backend, make_graph_mock):
        """NULL fact/name properties coerce to empty string via _edge_dict."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['e1', None, None]])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_valid_edges_for_node('u', group_id='test')
        assert result[0]['fact'] == ''
        assert result[0]['name'] == ''
