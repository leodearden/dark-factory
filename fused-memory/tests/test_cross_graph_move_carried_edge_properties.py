"""Every cross-graph RELATES_TO recreate in ``fused_memory.maintenance.cross_graph_move``
must carry ``expired_at`` (with ``invalid_at`` NULL) and the merge audit stamps
``superseded_edge_uuid`` / ``reassigned_from_node_uuid`` to the target copy.
That pair is the restore hooks' deliberately-restored signature; why losing it
matters is in ``fused_memory/backends/graphiti_client.py::GraphitiBackend.redirect_node_edges``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import extract_cypher, extract_params
from _relates_to_doubles import EdgeFixture, answer_edge_read, written_edge_properties

from fused_memory.maintenance import cross_graph_move
from fused_memory.maintenance.cross_graph_move import (
    merge_foreign_duplicate,
    move_entity_across_graphs,
    recreate_subgraph_relationships,
)

SOURCE_GRAPH = 'carry_source'
TARGET_GRAPH = 'carry_target'
NODE_UUID = 'node-carry-0001'
OTHER_NODE_UUID = 'node-carry-0002'
THIRD_NODE_UUID = 'node-carry-0003'
RESTORED_EDGE_UUID = 'edge-restored-0001'
PLAIN_EDGE_UUID = 'edge-plain-0002'

RESTORED_EXPIRED_AT = '2026-01-02T03:04:05Z'
SUPERSEDED_EDGE_UUID = 'edge-superseded-0001'
REASSIGNED_FROM_NODE_UUID = 'node-reassigned-from-0001'

NODE_ROW = [NODE_UUID, 'Alice', SOURCE_GRAPH, 'Alice is a person.', '2026-01-01T00:00:00+00:00']

RESTORED_EDGE = EdgeFixture(
    src_uuid=NODE_UUID,
    dst_uuid=OTHER_NODE_UUID,
    properties={
        'uuid': RESTORED_EDGE_UUID,
        'name': 'is_related_to',
        'fact': 'Alice is related to Bob.',
        'valid_at': '2026-01-01T00:00:00+00:00',
        'invalid_at': None,
        'expired_at': RESTORED_EXPIRED_AT,
        'created_at': '2026-01-01T00:00:00+00:00',
        'group_id': SOURCE_GRAPH,
        'episodes': ['episode-carry-1'],
        'superseded_edge_uuid': SUPERSEDED_EDGE_UUID,
        'reassigned_from_node_uuid': REASSIGNED_FROM_NODE_UUID,
    },
)

PLAIN_EDGE = EdgeFixture(
    src_uuid=NODE_UUID,
    dst_uuid=THIRD_NODE_UUID,
    properties={
        'uuid': PLAIN_EDGE_UUID,
        'name': 'is_related_to',
        'fact': 'Alice is related to Carol.',
        'valid_at': '2026-01-01T00:00:00+00:00',
        'invalid_at': None,
        'created_at': '2026-01-01T00:00:00+00:00',
        'group_id': SOURCE_GRAPH,
        'episodes': ['episode-carry-2'],
    },
)


@pytest.fixture
def fixed_fact_embedding(monkeypatch):
    monkeypatch.setattr(
        cross_graph_move, '_read_compact_vector', AsyncMock(return_value='[0.5, 0.25]'),
    )


def _route(graphs: dict) -> MagicMock:
    return MagicMock(side_effect=lambda name: graphs[name])


def _spec(uuid: str, disposition: str) -> dict:
    return {
        'uuid': uuid, 'disposition': disposition,
        'source_graph': SOURCE_GRAPH, 'target_graph': TARGET_GRAPH,
    }


def _relates_to_writes(graph: MagicMock) -> dict[str, dict]:
    """The properties each RELATES_TO CREATE on *graph* wrote, keyed by edge uuid."""
    writes = {}
    for call in graph.query.call_args_list:
        cypher = extract_cypher(call)
        if 'RELATES_TO' in cypher and 'CREATE' in cypher:
            written = written_edge_properties(cypher, extract_params(call))
            writes[written['uuid']] = written
    return writes


def _assert_restored_signature_and_stamps(written: dict) -> None:
    assert written['uuid'] == RESTORED_EDGE_UUID
    assert written.get('expired_at') == RESTORED_EXPIRED_AT
    assert 'invalid_at' in written
    assert written['invalid_at'] is None
    assert written.get('superseded_edge_uuid') == SUPERSEDED_EDGE_UUID
    assert written.get('reassigned_from_node_uuid') == REASSIGNED_FROM_NODE_UUID


@pytest.mark.asyncio
async def test_move_entity_across_graphs_carries_restored_signature_and_audit_stamps(
    mock_config, make_backend, make_graph_mock, fixed_fact_embedding,
):
    source = make_graph_mock()

    async def _source_reads(cypher, params=None):
        if 'RELATES_TO' in cypher:
            return answer_edge_read(cypher, RESTORED_EDGE, PLAIN_EDGE)
        if 'MENTIONS' in cypher:
            return MagicMock(result_set=[])
        return MagicMock(result_set=[NODE_ROW])

    source.ro_query = AsyncMock(side_effect=_source_reads)
    target = make_graph_mock()
    backend = make_backend(mock_config)
    backend._driver._get_graph = _route({SOURCE_GRAPH: source, TARGET_GRAPH: target})

    await move_entity_across_graphs(backend, NODE_UUID, SOURCE_GRAPH, TARGET_GRAPH)

    writes = _relates_to_writes(target)
    _assert_restored_signature_and_stamps(writes[RESTORED_EDGE_UUID])
    plain = writes[PLAIN_EDGE_UUID]
    assert plain.get('expired_at') is None
    assert plain.get('superseded_edge_uuid') is None
    assert plain.get('reassigned_from_node_uuid') is None


@pytest.mark.asyncio
async def test_phase_b_move_pass_carries_restored_signature_and_audit_stamps(
    mock_config, make_backend, make_graph_mock, fixed_fact_embedding,
):
    source = make_graph_mock()

    async def _source_reads(cypher, params=None):
        if 'RELATES_TO' in cypher:
            return answer_edge_read(cypher, RESTORED_EDGE)
        return MagicMock(result_set=[])

    source.ro_query = AsyncMock(side_effect=_source_reads)
    target = make_graph_mock()
    target.ro_query = AsyncMock(return_value=MagicMock(result_set=[]))
    target.query = AsyncMock(return_value=MagicMock(relationships_created=1))
    backend = make_backend(mock_config)
    backend._driver._get_graph = _route({SOURCE_GRAPH: source, TARGET_GRAPH: target})

    result = await recreate_subgraph_relationships(
        backend, [_spec(NODE_UUID, 'MOVE'), _spec(OTHER_NODE_UUID, 'MOVE')],
    )

    assert result.edges_recreated == 1
    _assert_restored_signature_and_stamps(_relates_to_writes(target)[RESTORED_EDGE_UUID])


@pytest.mark.asyncio
async def test_phase_b_merge_fold_carries_restored_signature_and_audit_stamps(
    mock_config, make_backend, make_graph_mock, fixed_fact_embedding,
):
    wrong = make_graph_mock()

    async def _wrong_reads(cypher, params=None):
        if 'RELATES_TO' in cypher:
            return answer_edge_read(cypher, RESTORED_EDGE)
        return MagicMock(result_set=[])

    wrong.ro_query = AsyncMock(side_effect=_wrong_reads)
    home = make_graph_mock()
    home.ro_query = AsyncMock(return_value=MagicMock(result_set=[]))
    home.query = AsyncMock(return_value=MagicMock(relationships_created=1))
    backend = make_backend(mock_config)
    backend._driver._get_graph = _route({SOURCE_GRAPH: wrong, TARGET_GRAPH: home})

    await recreate_subgraph_relationships(backend, [_spec(NODE_UUID, 'MERGE')])

    written = _relates_to_writes(home)[RESTORED_EDGE_UUID]
    _assert_restored_signature_and_stamps(written)
    assert written['group_id'] == RESTORED_EDGE.properties['group_id']


@pytest.mark.asyncio
async def test_merge_foreign_duplicate_carries_restored_signature_and_audit_stamps(
    mock_config, make_backend, make_graph_mock, fixed_fact_embedding,
):
    wrong = make_graph_mock()
    wrong.ro_query = AsyncMock(
        side_effect=lambda cypher, params=None: answer_edge_read(cypher, RESTORED_EDGE),
    )
    home = make_graph_mock()
    home.ro_query = AsyncMock(side_effect=[
        MagicMock(result_set=[]),
        MagicMock(result_set=[[RESTORED_EDGE_UUID]]),
    ])
    backend = make_backend(mock_config)
    backend._driver._get_graph = _route({SOURCE_GRAPH: wrong, TARGET_GRAPH: home})

    await merge_foreign_duplicate(backend, NODE_UUID, SOURCE_GRAPH, TARGET_GRAPH)

    _assert_restored_signature_and_stamps(_relates_to_writes(home)[RESTORED_EDGE_UUID])


@pytest.mark.asyncio
async def test_edge_read_row_width_mismatch_fails_loudly_before_any_mutation(
    mock_config, make_backend, make_graph_mock, fixed_fact_embedding,
):
    wrong = make_graph_mock()
    wrong.ro_query = AsyncMock(return_value=MagicMock(result_set=[['edge-x', 'name', 'fact']]))
    home = make_graph_mock()
    backend = make_backend(mock_config)
    backend._driver._get_graph = _route({SOURCE_GRAPH: wrong, TARGET_GRAPH: home})

    with pytest.raises(ValueError) as excinfo:
        await merge_foreign_duplicate(backend, NODE_UUID, SOURCE_GRAPH, TARGET_GRAPH)

    message = str(excinfo.value)
    assert 'expired_at' in message
    assert 'edge-x' in message
    home.query.assert_not_awaited()
    wrong.query.assert_not_awaited()
