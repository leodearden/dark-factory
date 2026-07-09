"""Tests for uuid-keyed dedup in GraphitiBackend read methods.

Covers:
- GraphitiBackend.get_valid_edges_for_node() — dedup keyed on e.uuid
- GraphitiBackend.get_all_valid_edges() — dedup keyed on (n.uuid, e.uuid)

Task 2213 (W6-zeta): replaces the WITH DISTINCT e / WITH DISTINCT n, e
graph-element-identity dedup idiom with plain uuid-keyed Python dedup, now
that RELATES_TO edge uuids are unique graph-wide (post tasks 2207/2210).
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from _fm_helpers import extract_cypher, extract_params

_LOGGER_NAME = 'fused_memory.backends.graphiti_client'

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

    @pytest.mark.asyncio
    async def test_shared_uuid_with_differing_content_logs_diagnostic(
        self, mock_config, make_backend, make_graph_mock, caplog
    ):
        """A shared e.uuid with differing fact/name keeps the first row and logs a debug diagnostic.

        The uuid-keyed dedup premise (task 2213) is that RELATES_TO edge uuids
        are unique graph-wide post tasks 2207/2210. This can't happen through
        the normal self-loop path (which repeats the identical row), so this
        directly guards the "invariant silently violated" failure mode called
        out in review: dedup must not silently undercount without a trace.
        """
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['e1', 'first-fact', 'first-name'], ['e1', 'second-fact', 'second-name']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            result = await backend.get_valid_edges_for_node('u', group_id='test')
        assert result == [{'uuid': 'e1', 'fact': 'first-fact', 'name': 'first-name'}]
        assert any('e1' in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_self_loop_identical_rows_do_not_log_diagnostic(
        self, mock_config, make_backend, make_graph_mock, caplog
    ):
        """The ordinary self-loop double-match (identical rows) stays silent — no false-positive diagnostic."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['e1', 'f', 'n'], ['e1', 'f', 'n']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await backend.get_valid_edges_for_node('u', group_id='test')
        assert caplog.records == []


# ---------------------------------------------------------------------------
# step-3: GraphitiBackend.get_all_valid_edges
# ---------------------------------------------------------------------------


class TestGetAllValidEdgesUuidDedup:
    """GraphitiBackend.get_all_valid_edges(*, group_id) dedups by (n.uuid, e.uuid)."""

    @pytest.mark.asyncio
    async def test_cypher_has_no_with_distinct(self, mock_config, make_backend, make_graph_mock):
        """Emitted Cypher no longer relies on WITH DISTINCT (B7); read path unchanged."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_all_valid_edges(group_id='test')
        cypher = extract_cypher(graph.ro_query.call_args)
        assert 'WITH DISTINCT' not in cypher
        assert 'RELATES_TO' in cypher
        assert 'invalid_at IS NULL' in cypher
        graph.ro_query.assert_awaited_once()
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_double_attribution_preserved(self, mock_config, make_backend, make_graph_mock):
        """A directed edge appears under BOTH endpoints: distinct n.uuid, same e.uuid → both kept."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['A', 'e1', 'f', 'n'], ['B', 'e1', 'f', 'n']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_all_valid_edges(group_id='test')
        assert set(result.keys()) == {'A', 'B'}
        assert result['A'] == [{'uuid': 'e1', 'fact': 'f', 'name': 'n'}]
        assert result['B'] == [{'uuid': 'e1', 'fact': 'f', 'name': 'n'}]

    @pytest.mark.asyncio
    async def test_self_loop_collapsed_per_entity(self, mock_config, make_backend, make_graph_mock):
        """Two identical (n.uuid, e.uuid) rows (self-loop double-match) collapse to one entry."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['A', 'e1', 'f', 'n'], ['A', 'e1', 'f', 'n']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_all_valid_edges(group_id='test')
        assert result['A'] == [{'uuid': 'e1', 'fact': 'f', 'name': 'n'}]

    @pytest.mark.asyncio
    async def test_distinct_edges_same_entity_preserved(self, mock_config, make_backend, make_graph_mock):
        """Two distinct edges under the same entity are both returned (no over-collapse)."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['A', 'e1', 'f1', 'n1'], ['A', 'e2', 'f2', 'n2']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_all_valid_edges(group_id='test')
        assert result['A'] == [
            {'uuid': 'e1', 'fact': 'f1', 'name': 'n1'},
            {'uuid': 'e2', 'fact': 'f2', 'name': 'n2'},
        ]

    @pytest.mark.asyncio
    async def test_shared_pair_with_differing_content_logs_diagnostic(
        self, mock_config, make_backend, make_graph_mock, caplog
    ):
        """A shared (n.uuid, e.uuid) pair with differing fact/name keeps the first row and logs a diagnostic.

        Guards the same "invariant silently violated" failure mode as
        get_valid_edges_for_node, keyed on the (entity, edge) pair instead.
        """
        backend = make_backend(mock_config)
        graph = make_graph_mock(
            ro_rows=[['A', 'e1', 'first-fact', 'first-name'], ['A', 'e1', 'second-fact', 'second-name']]
        )
        backend._driver._get_graph = MagicMock(return_value=graph)
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            result = await backend.get_all_valid_edges(group_id='test')
        assert result['A'] == [{'uuid': 'e1', 'fact': 'first-fact', 'name': 'first-name'}]
        assert any('e1' in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_self_loop_identical_rows_do_not_log_diagnostic(
        self, mock_config, make_backend, make_graph_mock, caplog
    ):
        """The ordinary self-loop double-match (identical rows) stays silent — no false-positive diagnostic."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['A', 'e1', 'f', 'n'], ['A', 'e1', 'f', 'n']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            await backend.get_all_valid_edges(group_id='test')
        assert caplog.records == []
