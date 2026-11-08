"""The two Graphiti STARTS WITH reads behind the uuid-prefix resolver (task 5323).

``GraphitiBackend.entity_uuids_with_prefix`` / ``edge_uuids_with_prefix`` are
the graph half of ``services/id_resolver.py::IdResolver``: every id in one
project graph that starts with a prefix.  The resolver classifies a prefix
from their answer, so an outage must surface as an exception (INV-11), never
as an empty list that reads as "no such id".
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import FALKOR_HOST, FALKOR_PORT, falkor_skipif, unique_graph_name
from falkordb.asyncio import FalkorDB

from fused_memory.backends.falkor_indices import range_create_statement

_NODE_UUID = 'bff81530-70e5-4ab6-939e-000000000001'
_NEIGHBOUR_UUID = 'bff81531-0000-4000-8000-000000000002'
_EDGE_UUID = 'abcd1234-0000-4000-8000-000000000003'


def _params_of(call) -> object:
    if len(call.args) > 1:
        return call.args[1]
    return call.kwargs.get('params')


def _cypher_of(call) -> str:
    return call.args[0] if call.args else call.kwargs['cypher']


class TestEntityUuidsWithPrefix:
    @pytest.mark.asyncio
    async def test_returns_uuid_name_pairs_deduped_with_none_name_as_empty(
        self, mock_config, make_backend, make_graph_driver, make_graph_mock,
    ):
        backend = make_backend(mock_config, driver=make_graph_driver({'reify': make_graph_mock([
            [_NODE_UUID, 'Prefix resolver'],
            ['bff81530-70e5-4ab6-939e-000000000009', None],
            [_NODE_UUID, 'Prefix resolver'],
        ])}))

        got = await backend.entity_uuids_with_prefix('bff81530', group_id='reify')

        assert got == [
            (_NODE_UUID, 'Prefix resolver'),
            ('bff81530-70e5-4ab6-939e-000000000009', ''),
        ]

    @pytest.mark.asyncio
    async def test_reads_the_graph_of_the_given_group(
        self, mock_config, make_backend, make_graph_driver, make_graph_mock,
    ):
        graph = make_graph_mock([])
        backend = make_backend(mock_config, driver=make_graph_driver({'reify': graph}))

        await backend.entity_uuids_with_prefix('bff81530', group_id='reify')

        graph.ro_query.assert_awaited()


class TestEdgeUuidsWithPrefix:
    @pytest.mark.asyncio
    async def test_returns_uuid_fact_pairs_with_none_fact_as_empty(
        self, mock_config, make_backend, make_graph_driver, make_graph_mock,
    ):
        backend = make_backend(mock_config, driver=make_graph_driver({'reify': make_graph_mock([
            [_EDGE_UUID, 'resolver reads the graph'],
            ['abcd1234-0000-4000-8000-000000000004', None],
        ])}))

        got = await backend.edge_uuids_with_prefix('abcd1234', group_id='reify')

        assert got == [
            (_EDGE_UUID, 'resolver reads the graph'),
            ('abcd1234-0000-4000-8000-000000000004', ''),
        ]


_READS = ('entity_uuids_with_prefix', 'edge_uuids_with_prefix')


class TestPrefixIsBoundNeverInterpolated:
    @pytest.mark.parametrize('method', _READS)
    @pytest.mark.asyncio
    async def test_every_query_binds_the_prefix_as_a_parameter(
        self, method, mock_config, make_backend, make_graph_driver, make_graph_mock,
    ):
        graph = make_graph_mock([['deadbeef-0000-4000-8000-000000000001', 'x']])
        backend = make_backend(mock_config, driver=make_graph_driver({'reify': graph}))

        await getattr(backend, method)('deadbeef', group_id='reify')

        calls = graph.ro_query.await_args_list
        assert len(calls) >= 2, 'a census and at least one page are expected'
        for call in calls:
            params = _params_of(call)
            assert isinstance(params, dict) and params['prefix'] == 'deadbeef'
            cypher = _cypher_of(call)
            assert all(str(value) not in cypher for value in params.values())


class TestOutagePropagates:
    @pytest.mark.parametrize('method', _READS)
    @pytest.mark.parametrize('error', [ConnectionError('falkordb down'), TimeoutError('slow')])
    @pytest.mark.asyncio
    async def test_a_failing_read_raises_rather_than_returning_empty(
        self, method, error, mock_config, make_backend, make_graph_driver,
    ):
        graph = MagicMock()
        graph.ro_query = AsyncMock(side_effect=error)
        backend = make_backend(mock_config, driver=make_graph_driver({'reify': graph}))

        with pytest.raises(type(error)):
            await getattr(backend, method)('bff81530', group_id='reify')


# ---------------------------------------------------------------------------
# Live FalkorDB: the queries run against the indexed uuid property
# ---------------------------------------------------------------------------


class _RecordingGraph:
    """Delegates ro_query to a live graph and records each (cypher, params) it was sent."""

    def __init__(self, graph) -> None:
        self._graph = graph
        self.queries: list[tuple[str, object]] = []

    async def ro_query(self, cypher, params=None, *args, **kwargs):
        self.queries.append((cypher, params))
        return await self._graph.ro_query(cypher, params, *args, **kwargs)


async def _seed(graph) -> None:
    await graph.query(range_create_statement(('Entity', 'NODE', 'uuid', 'RANGE')))
    await graph.query(range_create_statement(('RELATES_TO', 'RELATIONSHIP', 'uuid', 'RANGE')))
    await graph.query(
        'CREATE (a:Entity {uuid: $a, name: $a_name}), '
        '(b:Entity {uuid: $b, name: $b_name}), '
        '(a)-[:RELATES_TO {uuid: $e, fact: $fact}]->(b)',
        {
            'a': _NODE_UUID, 'a_name': 'first node',
            'b': _NEIGHBOUR_UUID, 'b_name': 'neighbour',
            'e': _EDGE_UUID, 'fact': 'first node relates to neighbour',
        },
    )


@pytest_asyncio.fixture
async def prefix_live_graph():
    """Provision a throwaway, uniquely-named FalkorDB graph, yield it, clean up."""
    graph_name = unique_graph_name('5323_uuid_prefix')
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    with contextlib.suppress(Exception):
        stale = client.select_graph(graph_name)
        await stale.delete()
    graph = client.select_graph(graph_name)
    try:
        yield graph
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


@falkor_skipif()
@pytest.mark.timeout(60)
@pytest.mark.integration
class TestPrefixReadsLiveFalkorDB:
    @pytest.mark.asyncio
    async def test_starts_with_matches_exactly_the_prefixed_ids(
        self, mock_config, make_backend, make_graph_driver, prefix_live_graph,
    ):
        graph = prefix_live_graph
        await _seed(graph)
        backend = make_backend(mock_config, driver=make_graph_driver({'test': graph}))

        assert await backend.entity_uuids_with_prefix('bff81530', group_id='test') == [
            (_NODE_UUID, 'first node'),
        ]
        assert await backend.edge_uuids_with_prefix('abcd1234', group_id='test') == [
            (_EDGE_UUID, 'first node relates to neighbour'),
        ]
        assert await backend.entity_uuids_with_prefix(_NODE_UUID, group_id='test') == [
            (_NODE_UUID, 'first node'),
        ]

    @pytest.mark.parametrize(('method', 'prefix'), [
        ('entity_uuids_with_prefix', 'bff81530'),
        ('edge_uuids_with_prefix', 'abcd1234'),
    ])
    @pytest.mark.asyncio
    async def test_every_query_is_served_by_the_uuid_index(
        self, method, prefix, mock_config, make_backend, make_graph_driver, prefix_live_graph,
    ):
        """A bare STARTS WITH is planned as a full label/node scan (measured on
        live reify: ~13ms nodes, ~110ms edges); the reads must reach the
        uuid RANGE index instead.
        """
        graph = prefix_live_graph
        await _seed(graph)
        recorder = _RecordingGraph(graph)
        backend = make_backend(mock_config, driver=make_graph_driver({'test': recorder}))

        await getattr(backend, method)(prefix, group_id='test')

        assert len(recorder.queries) >= 2
        for cypher, params in recorder.queries:
            plan = str(await graph.explain(cypher, params))
            assert 'Index Scan' in plan, f'{cypher!r} is not index-served:\n{plan}'
