"""GraphitiBackend.update_edge's write contract (task 7723).

graphiti's entity-edge projection omits ``fact_embedding``, so an edge loaded
by ``EntityEdge.get_by_uuid`` carries ``fact_embedding=None``.  A full-map
``edge.save`` of that edge writes ``vecf32(null)`` and wipes the stored
embedding.  So only a ``fact`` change, which regenerates the embedding, may use
the save; an ``invalid_at`` / ``clear_invalid_at`` change is one targeted
``SET e.invalid_at`` write that touches no other property.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT, falkor_skipif, unique_graph_name
from _fm_helpers import (
    extract_cypher,
    extract_params,
)
from falkordb.asyncio import FalkorDB
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import parse_db_date

_GROUP = 'proj'
_TS = datetime(2026, 6, 1, tzinfo=UTC)
_NEW_EMBEDDING = [0.5, 0.25]

# Keyed by graphiti's get_entity_edge_return_query projection, which has no
# fact_embedding column: that omission is the defect's premise.
_EDGE_ROW = {
    'uuid': 'e-1',
    'source_node_uuid': 'src',
    'target_node_uuid': 'tgt',
    'group_id': _GROUP,
    'created_at': '2026-01-01T00:00:00+00:00',
    'name': 'rel',
    'fact': 'original fact',
    'episodes': ['ep-1'],
    'expired_at': None,
    'valid_at': None,
    'invalid_at': '2026-01-01T00:00:00+00:00',
    'attributes': {},
}

_INVALIDITY_ONLY = pytest.mark.parametrize(
    ('invalidity', 'stored_invalid_at'),
    [
        ({'invalid_at': _TS}, _TS.isoformat()),
        ({'clear_invalid_at': True}, None),
        ({'invalid_at': _TS, 'clear_invalid_at': True}, None),
    ],
    ids=['set', 'clear', 'clear-wins-over-set'],
)


@pytest.fixture
def backend_over_edge_rows(mock_config, make_backend, make_graph_driver, make_graph_mock):
    """Return ``_backend(rows) -> (backend, graph)`` over a real driver and one graph double.

    Every Cypher, graphiti's and ``_graph_for``'s alike, lands in issue order
    on ``graph.query``.
    """
    def _backend(rows: list[list]):
        graph = make_graph_mock(rows, header=[[1, k] for k in _EDGE_ROW])
        backend = make_backend(mock_config, driver=make_graph_driver({_GROUP: graph}))
        backend.refresh_entity_summary = AsyncMock(return_value={})
        backend.client.embedder.create = AsyncMock(return_value=_NEW_EMBEDDING)
        return backend, graph

    return _backend


_PRESENT = [list(_EDGE_ROW.values())]


def _writes(graph) -> list[tuple[str, dict]]:
    """The ``(cypher, params)`` of every non-read ``graph.query`` call, in issue order."""
    calls = [(extract_cypher(c), extract_params(c)) for c in graph.query.call_args_list]
    return [(cypher, params) for cypher, params in calls if params.get('routing_') != 'r']


def _is_targeted_invalid_at_write(cypher: str) -> bool:
    return 'RELATES_TO {uuid: $uuid}' in cypher and 'SET e.invalid_at = $invalid_at' in cypher


class TestUpdateEdgeWrites:
    @_INVALIDITY_ONLY
    @pytest.mark.asyncio
    async def test_an_invalidity_only_change_is_one_targeted_write(
        self, backend_over_edge_rows, invalidity, stored_invalid_at,
    ):
        backend, graph = backend_over_edge_rows(_PRESENT)

        await backend.update_edge('e-1', group_id=_GROUP, **invalidity)

        writes = _writes(graph)
        assert not [cypher for cypher, _ in writes if 'fact_embedding' in cypher], writes
        assert len(writes) == 1, writes
        cypher, params = writes[0]
        assert _is_targeted_invalid_at_write(cypher), cypher
        assert params == {'uuid': 'e-1', 'invalid_at': stored_invalid_at}

    @pytest.mark.asyncio
    async def test_a_fact_change_saves_the_fact_with_a_regenerated_embedding(
        self, backend_over_edge_rows,
    ):
        backend, graph = backend_over_edge_rows(_PRESENT)

        result = await backend.update_edge('e-1', 'new fact', group_id=_GROUP)

        writes = _writes(graph)
        assert len(writes) == 1, writes
        _, params = writes[0]
        assert params['edge_data']['fact'] == 'new fact'
        assert params['edge_data']['fact_embedding'] == _NEW_EMBEDDING
        assert not [cypher for cypher, _ in writes if _is_targeted_invalid_at_write(cypher)]
        assert result['fact'] == 'new fact'

    @pytest.mark.parametrize(
        ('invalidity', 'stored_invalid_at'),
        [({'invalid_at': _TS}, _TS.isoformat()), ({'clear_invalid_at': True}, None)],
        ids=['set', 'clear'],
    )
    @pytest.mark.asyncio
    async def test_a_fact_and_invalidity_change_saves_then_writes_invalid_at(
        self, backend_over_edge_rows, invalidity, stored_invalid_at,
    ):
        backend, graph = backend_over_edge_rows(_PRESENT)

        await backend.update_edge('e-1', 'new fact', group_id=_GROUP, **invalidity)

        writes = _writes(graph)
        assert len(writes) == 2, writes
        (_, save_params), (targeted_cypher, targeted_params) = writes
        assert save_params['edge_data']['fact'] == 'new fact'
        assert save_params['edge_data']['fact_embedding'] == _NEW_EMBEDDING
        assert _is_targeted_invalid_at_write(targeted_cypher), targeted_cypher
        assert targeted_params == {'uuid': 'e-1', 'invalid_at': stored_invalid_at}

    @pytest.mark.asyncio
    async def test_a_missing_edge_raises_before_any_write(self, backend_over_edge_rows):
        backend, graph = backend_over_edge_rows([])

        with pytest.raises(EdgeNotFoundError):
            await backend.update_edge('e-1', group_id=_GROUP, clear_invalid_at=True)

        assert _writes(graph) == []

    @pytest.mark.parametrize(
        'invalidity', [{'invalid_at': _TS}, {'clear_invalid_at': True}], ids=['set', 'clear'],
    )
    @pytest.mark.asyncio
    async def test_the_invalidity_write_lands_before_the_summary_refresh(
        self, backend_over_edge_rows, invalidity,
    ):
        """Task 1940: the refresh counts only ``invalid_at IS NULL`` edges, so a
        restored edge must already be restored when it runs."""
        backend, graph = backend_over_edge_rows(_PRESENT)
        invalidity_writes_at_refresh: list[int] = []

        async def _record(node_uuid, *, group_id):
            invalidity_writes_at_refresh.append(len([
                cypher for cypher, _ in _writes(graph) if _is_targeted_invalid_at_write(cypher)
            ]))
            return {}

        backend.refresh_entity_summary = AsyncMock(side_effect=_record)

        await backend.update_edge('e-1', group_id=_GROUP, **invalidity)

        assert invalidity_writes_at_refresh, 'refresh_entity_summary was never called'
        assert all(count > 0 for count in invalidity_writes_at_refresh), (
            invalidity_writes_at_refresh
        )


_EMBEDDING = [0.5, 0.25, 0.125]


@pytest_asyncio.fixture
async def update_edge_live_graph():
    """Provision a throwaway, uniquely-named FalkorDB graph, yield it, clean up."""
    graph_name = unique_graph_name('7723_update_edge')
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


async def _seed(graph) -> None:
    await graph.query(
        "CREATE (a:Entity {uuid: 'A', name: 'a', summary: '', group_id: $group}),"
        " (b:Entity {uuid: 'B', name: 'b', summary: '', group_id: $group})"
        " CREATE (a)-[e:RELATES_TO {uuid: 'e1', name: 'rel', fact: 'a relates to b',"
        " group_id: $group, episodes: ['ep-1'], created_at: '2026-01-01T00:00:00+00:00',"
        " invalid_at: '2026-01-01T00:00:00+00:00'}]->(b)"
        ' SET e.fact_embedding = vecf32($embedding)',
        {'group': _GROUP, 'embedding': _EMBEDDING},
    )


async def _stored_embedding(graph) -> object:
    result = await graph.ro_query(
        "MATCH ()-[e:RELATES_TO {uuid: 'e1'}]->() RETURN e.fact_embedding"
    )
    return result.result_set[0][0]


@falkor_skipif()
@pytest.mark.timeout(60)
@pytest.mark.integration
class TestUpdateEdgeLiveFalkorDB:
    @_INVALIDITY_ONLY
    @pytest.mark.asyncio
    async def test_an_invalidity_only_change_keeps_the_stored_embedding(
        self, mock_config, make_backend, make_graph_driver, update_edge_live_graph,
        invalidity, stored_invalid_at,
    ):
        graph = update_edge_live_graph
        await _seed(graph)
        backend = make_backend(mock_config, driver=make_graph_driver({_GROUP: graph}))

        await backend.update_edge('e1', group_id=_GROUP, **invalidity)

        assert await _stored_embedding(graph) == _EMBEDDING
        raw = await backend.get_edge_invalid_at('e1', group_id=_GROUP)
        assert raw == stored_invalid_at
        if stored_invalid_at is not None:
            assert parse_db_date(raw) == _TS

    @pytest.mark.asyncio
    async def test_invalidating_then_restoring_keeps_the_stored_embedding(
        self, mock_config, make_backend, make_graph_driver, update_edge_live_graph,
    ):
        graph = update_edge_live_graph
        await _seed(graph)
        backend = make_backend(mock_config, driver=make_graph_driver({_GROUP: graph}))

        await backend.update_edge('e1', group_id=_GROUP, invalid_at=_TS)
        assert await _stored_embedding(graph) == _EMBEDDING
        await backend.update_edge('e1', group_id=_GROUP, clear_invalid_at=True)

        assert await _stored_embedding(graph) == _EMBEDDING
        assert await backend.get_edge_invalid_at('e1', group_id=_GROUP) is None
