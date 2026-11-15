"""Graph replies carry what the graph holds: endpoint names on search hits, dates on entity edges.

Two defects made a graph reply say less than the graph stores:

- ``MemoryService.get_entity``'s exact-name branch returned every edge with
  ``temporal=None``, because ``GraphitiBackend.get_valid_edges_for_node`` read
  no dates.
- ``MemoryService.search`` returned every graph hit with ``entities=[]``,
  because it read ``source_node`` / ``target_node`` attributes that
  graphiti_core's ``EntityEdge`` does not have.

The service-level tests drive the PUBLIC ``MemoryService`` reads over a REAL
``GraphitiBackend`` whose raw graph is a Cypher-routing double, so the service
reads exactly what the real backend read produced.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from _fm_helpers import extract_cypher, extract_params
from _graph_access_fake import FakeGraphAccess

GROUP = 'test'

Rows = list[list[Any]]


def _return_clause(cypher: str) -> str:
    return cypher.rsplit('RETURN', 1)[1]


class _RoutingGraph:
    """A raw falkordb-shaped graph answering each read by a substring of its Cypher.

    Every read is recorded as ``(cypher, params)``. A read no route claims is
    an AssertionError, so a test cannot pass on a query it did not expect.
    """

    def __init__(self, routes: Sequence[tuple[str, Callable[[dict[str, Any]], Rows]]]):
        self._routes = list(routes)
        self.reads: list[tuple[str, dict[str, Any]]] = []

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None):
        bound = params or {}
        self.reads.append((cypher, bound))
        for needle, respond in self._routes:
            if needle in cypher:
                result = MagicMock()
                result.header = []
                result.result_set = respond(bound)
                return result
        raise AssertionError(f'no route for read: {cypher}')

    async def query(self, cypher: str, params: dict[str, Any] | None = None):
        raise AssertionError(f'a read path issued a write-budget query: {cypher}')

    def reads_matching(self, needle: str) -> list[tuple[str, dict[str, Any]]]:
        return [read for read in self.reads if needle in read[0]]


def _service_over(mock_config, make_backend, graph: _RoutingGraph):
    from fused_memory.services.memory_service import MemoryService

    backend = make_backend(mock_config, graph_access=FakeGraphAccess({GROUP: graph}))
    svc = MemoryService(mock_config)
    svc.graphiti = backend
    return svc, backend


class TestValidEdgesCarryTheirDates:
    """get_valid_edges_for_node returns each edge's stored validity window verbatim."""

    @pytest.mark.asyncio
    async def test_rows_carry_their_stored_dates(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[
            ['e1', 'Task 42 blocks Task 43', 'BLOCKS', '2026-11-01T09:00:00+00:00', None],
            ['e2', 'Task 42 uses Redis', 'USES', None, None],
        ])
        backend._driver._get_graph = MagicMock(return_value=graph)

        edges = await backend.get_valid_edges_for_node('n', group_id=GROUP)

        assert edges == [
            {
                'uuid': 'e1', 'fact': 'Task 42 blocks Task 43', 'name': 'BLOCKS',
                'valid_at': '2026-11-01T09:00:00+00:00', 'invalid_at': None,
            },
            {
                'uuid': 'e2', 'fact': 'Task 42 uses Redis', 'name': 'USES',
                'valid_at': None, 'invalid_at': None,
            },
        ]

    @pytest.mark.asyncio
    async def test_cypher_returns_the_validity_window(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)

        await backend.get_valid_edges_for_node('n', group_id=GROUP)

        returned = _return_clause(extract_cypher(graph.ro_query.call_args))
        assert 'e.valid_at' in returned
        assert 'e.invalid_at' in returned


_STORED_DATES = {
    'e1': '2026-10-01T08:00:00+00:00',
    'e2': '2026-10-15T12:30:00+00:00',
    'e3': '2026-11-01T09:00:00.123456+00:00',
}


class TestExactNameEntityEdgesCarryDates:
    """B9: an exact-name get_entity dates each edge with the window the graph stores."""

    @pytest.mark.asyncio
    async def test_each_edge_carries_its_stored_valid_at(self, mock_config, make_backend):
        graph = _RoutingGraph([
            ('{name: $name}', lambda _p: [['n-42', 'Task 42', 'summary', ['Entity']]]),
            ('RELATES_TO', lambda _p: [
                [uuid, f'fact {uuid}', 'RELATES', valid_at, None]
                for uuid, valid_at in _STORED_DATES.items()
            ]),
        ])
        svc, _backend = _service_over(mock_config, make_backend, graph)

        reply = await svc.get_entity('Task 42', project_id=GROUP)

        assert [node['uuid'] for node in reply['nodes']] == ['n-42']
        assert len(reply['edges']) == 3
        assert {edge['uuid']: edge['temporal'] for edge in reply['edges']} == {
            uuid: {'valid_at': valid_at, 'invalid_at': None}
            for uuid, valid_at in _STORED_DATES.items()
        }


class TestNodeNamesByUuids:
    """get_node_names_by_uuids names a batch of Entity nodes in one read."""

    @pytest.mark.asyncio
    async def test_names_each_distinct_node_in_one_read(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['n-42', 'Task 42'], ['n-r', 'Redis']])
        backend._driver._get_graph = MagicMock(return_value=graph)

        names = await backend.get_node_names_by_uuids(['n-42', 'n-r', 'n-42'], group_id=GROUP)

        assert names == {'n-42': 'Task 42', 'n-r': 'Redis'}
        graph.ro_query.assert_awaited_once()
        graph.query.assert_not_awaited()
        cypher = extract_cypher(graph.ro_query.call_args)
        assert 'Entity' in cypher
        assert 'n.uuid IN $uuids' in cypher
        assert extract_params(graph.ro_query.call_args)['uuids'] == ['n-42', 'n-r']

    @pytest.mark.asyncio
    async def test_unnamed_and_absent_nodes_are_left_out(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['n-42', 'Task 42'], ['n-null', None], ['n-empty', '']])
        backend._driver._get_graph = MagicMock(return_value=graph)

        names = await backend.get_node_names_by_uuids(
            ['n-42', 'n-null', 'n-empty', 'n-absent'], group_id=GROUP,
        )

        assert names == {'n-42': 'Task 42'}

    @pytest.mark.asyncio
    async def test_empty_input_issues_no_read(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['n-42', 'Task 42']])
        backend._driver._get_graph = MagicMock(return_value=graph)

        assert await backend.get_node_names_by_uuids([], group_id=GROUP) == {}
        graph.ro_query.assert_not_awaited()
        graph.query.assert_not_awaited()
