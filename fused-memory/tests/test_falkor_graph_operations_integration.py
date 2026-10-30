"""Live FalkorDB: graphiti's edge get-by-uuid(s) through FalkorGraphOperations (task 7714).

With the production index set present (task 3708), FalkorDB plans graphiti-core's
stock ``EntityEdge.get_by_uuid(s)`` as a full Entity label scan beneath the edge
index scan.  These tests seed a scratch graph through graphiti's real bulk write,
which also drives the forwarded bulk saves, provision it with the production
sweep, and pin the plan of the Cypher the hardened driver issues (against a stock
negative control), its rows against graphiti's built-in path, and
``GraphitiBackend.remove_edge`` / ``update_edge`` on top of it.

Requires a running FalkorDB; skipped when one is not reachable, and deselected by
the default ``-m 'not integration'`` addopts.  Graphs and backends come only from
``_falkor_index_live``, which enforces the live-FalkorDB HAZARD rules.
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from _falkor_index_live import injected_driver, live_backends, scratch_graphs
from _falkor_plans import explain, scan_operators, spy_on_queries
from _fm_helpers import await_index_operational, falkor_skipif
from _query_recorder import IssuedQuery, recording_driver
from _unused_embedder import UnusedEmbedder
from falkordb.asyncio.graph import AsyncGraph
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.nodes import EntityNode
from graphiti_core.utils import bulk_utils

from fused_memory.backends.graphiti_client import GraphitiBackend

pytestmark = [
    falkor_skipif(),
    pytest.mark.integration,
]

# The planner's choice is structural, not cost-based: 50 and 300 nodes gave the
# production plan shapes.  The ~1 s cost itself appears only near 45k Entity nodes.
EDGE_COUNT = 50

SEEDED_AT = datetime(2026, 10, 29, tzinfo=UTC)

OFF_LABEL_UUID = 'edge-off-label'
OFF_LABEL_FACT = 'off-label fact'
MISSING_UUID = 'edge-missing'

SEED_OFF_LABEL_CYPHER = (
    "CREATE (:CanaryEndpoint {uuid: 'canary-source', name: 'canary-source', group_id: $group_id})"
    "-[:RELATES_TO {uuid: $uuid, group_id: $group_id, name: 'RELATES_TO', fact: $fact, "
    "episodes: ['ep'], created_at: $created_at}]->"
    "(:CanaryEndpoint {uuid: 'canary-target', name: 'canary-target', group_id: $group_id})"
)


def ring_nodes(group_id: str, count: int) -> list[EntityNode]:
    return [
        EntityNode(
            uuid=f'node-{i}',
            name=f'node-{i}',
            group_id=group_id,
            created_at=SEEDED_AT,
            name_embedding=[1.0, 0.0, 0.0],
        )
        for i in range(1, count + 1)
    ]


def ring_edges(group_id: str, count: int) -> list[EntityEdge]:
    """Edge ``i`` runs node ``i`` -> node ``i % count + 1``."""
    return [
        EntityEdge(
            uuid=f'edge-{i}',
            group_id=group_id,
            source_node_uuid=f'node-{i}',
            target_node_uuid=f'node-{i % count + 1}',
            created_at=SEEDED_AT,
            name='RELATES_TO',
            fact=f'fact of edge-{i}',
            fact_embedding=[0.0, 1.0, 0.0],
            episodes=['ep'],
        )
        for i in range(1, count + 1)
    ]


@dataclass(frozen=True)
class SeededGraph:
    name: str
    graph: AsyncGraph
    backend: GraphitiBackend
    driver: GraphDriver
    edges: dict[str, EntityEdge]


@pytest_asyncio.fixture
async def seeded(mock_config) -> AsyncIterator[SeededGraph]:
    """A ring of Entity edges written by graphiti's bulk write, plus one off-label edge,
    production-provisioned."""
    async with scratch_graphs('7714') as scratch, live_backends(mock_config) as backends:
        name, graph = scratch('edges')
        backend = backends({name})
        driver = injected_driver(backend).clone(name)
        edges = ring_edges(name, EDGE_COUNT)
        # Also creates the graph KEY, which the provisioning sweep requires.
        await bulk_utils.add_nodes_and_edges_bulk(
            driver, [], [], ring_nodes(name, EDGE_COUNT), edges, UnusedEmbedder()
        )
        await graph.query(
            SEED_OFF_LABEL_CYPHER,
            {
                'uuid': OFF_LABEL_UUID,
                'fact': OFF_LABEL_FACT,
                'group_id': name,
                'created_at': SEEDED_AT.isoformat(),
            },
        )
        await backend.provision_registered_graphs()
        await await_index_operational(graph)
        yield SeededGraph(
            name=name,
            graph=graph,
            backend=backend,
            driver=driver,
            edges={edge.uuid: edge for edge in edges},
        )


def seeded_endpoints(edge: EntityEdge) -> tuple[str, str]:
    return (edge.source_node_uuid, edge.target_node_uuid)


def builtin_driver(driver: GraphDriver) -> GraphDriver:
    """*driver* on the same connection, taking graphiti's built-in get-by-uuid Cypher."""
    builtin = copy.copy(driver)
    builtin.graph_operations_interface = None
    return builtin


class TestBulkWrite:
    @pytest.mark.asyncio
    async def test_seeded_edges_landed_between_their_seeded_endpoints(self, seeded):
        """The forwarded bulk saves wrote every edge, bound to its endpoints by topology."""
        result = await seeded.graph.ro_query(
            'MATCH (s)-[e:RELATES_TO]->(t) WHERE e.uuid IN $uuids RETURN e.uuid, s.uuid, t.uuid',
            {'uuids': list(seeded.edges)},
        )

        landed = {edge_uuid: (source, target) for edge_uuid, source, target in result.result_set}
        assert landed == {uuid: seeded_endpoints(edge) for uuid, edge in seeded.edges.items()}


# --- Plan shape ---------------------------------------------------------------

GetLeg = Callable[[GraphDriver], Awaitable[object]]

GET_LEGS: dict[str, GetLeg] = {
    'get_by_uuid': lambda driver: EntityEdge.get_by_uuid(driver, 'edge-1'),
    'get_by_uuids': lambda driver: EntityEdge.get_by_uuids(driver, ['edge-1', 'edge-2']),
}


async def issued_by_stock_graphiti(leg: GetLeg) -> IssuedQuery:
    """The Cypher stock graphiti emits: a bare FalkorDriver has no graph_operations_interface."""
    stock, issued = recording_driver(FalkorDriver)
    # The recorder returns no rows, which get_by_uuid reports as not found.
    with contextlib.suppress(EdgeNotFoundError):
        await leg(stock)
    (query,) = issued
    return query


class TestPlanShape:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', sorted(GET_LEGS))
    async def test_issued_cypher_plans_a_single_edge_index_scan(self, seeded, leg):
        issued = spy_on_queries(seeded.driver)

        await GET_LEGS[leg](seeded.driver)

        (query,) = issued
        plan = await explain(seeded.graph, query)
        assert scan_operators(plan) == ['Edge By Index Scan'], str(plan)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', sorted(GET_LEGS))
    async def test_stock_cypher_plans_a_node_label_scan(self, seeded, leg):
        """Negative control: without it the plan assertion above could pass vacuously."""
        query = await issued_by_stock_graphiti(GET_LEGS[leg])

        plan = await explain(seeded.graph, query)

        assert 'Node By Label Scan' in scan_operators(plan), str(plan)


# --- Rows ---------------------------------------------------------------------


class TestRowsMatchGraphitiBuiltin:
    @pytest.mark.asyncio
    async def test_seeded_uuid_returns_the_builtin_edge(self, seeded):
        builtin = builtin_driver(seeded.driver)
        issued_by_builtin = spy_on_queries(builtin)

        edge = await EntityEdge.get_by_uuid(seeded.driver, 'edge-1')
        expected = await EntityEdge.get_by_uuid(builtin, 'edge-1')

        (oracle_query,) = issued_by_builtin
        assert '(n:Entity)' in oracle_query.cypher, 'the oracle must run the built-in Cypher'
        assert edge.model_dump() == expected.model_dump()
        assert seeded_endpoints(edge) == seeded_endpoints(seeded.edges['edge-1'])

    @pytest.mark.asyncio
    @pytest.mark.parametrize('uuid', [OFF_LABEL_UUID, MISSING_UUID], ids=['off-label', 'missing'])
    async def test_unreturnable_uuid_is_not_found_on_both_paths(self, seeded, uuid):
        for driver in (seeded.driver, builtin_driver(seeded.driver)):
            with pytest.raises(EdgeNotFoundError):
                await EntityEdge.get_by_uuid(driver, uuid)

    @pytest.mark.asyncio
    async def test_get_by_uuids_returns_the_builtin_edge_set(self, seeded):
        uuids = ['edge-1', 'edge-2', OFF_LABEL_UUID, MISSING_UUID]

        rewritten = {edge.uuid for edge in await EntityEdge.get_by_uuids(seeded.driver, uuids)}
        builtin = {
            edge.uuid
            for edge in await EntityEdge.get_by_uuids(builtin_driver(seeded.driver), uuids)
        }

        assert rewritten == builtin == {'edge-1', 'edge-2'}


# --- GraphitiBackend ----------------------------------------------------------


class TestBackendEdgeOperations:
    @pytest.mark.asyncio
    async def test_remove_edge_deletes_the_edge_and_is_idempotent(self, seeded):
        await seeded.backend.remove_edge('edge-1', group_id=seeded.name)

        with pytest.raises(EdgeNotFoundError):
            await seeded.backend.get_edge_text('edge-1', group_id=seeded.name)
        assert await seeded.backend.remove_edge('edge-1', group_id=seeded.name) is None

    @pytest.mark.asyncio
    async def test_remove_edge_leaves_an_off_label_edge_in_place(self, seeded):
        await seeded.backend.remove_edge(OFF_LABEL_UUID, group_id=seeded.name)

        assert await seeded.backend.get_edge_text(OFF_LABEL_UUID, group_id=seeded.name) == (
            'RELATES_TO',
            OFF_LABEL_FACT,
        )

    @pytest.mark.asyncio
    async def test_update_edge_invalidates_the_edge(self, seeded):
        result = await seeded.backend.update_edge(
            'edge-2', group_id=seeded.name, invalid_at=datetime(2026, 10, 30, tzinfo=UTC)
        )

        assert (result['uuid'], result['fact']) == ('edge-2', seeded.edges['edge-2'].fact)
        assert await seeded.backend.get_edge_invalid_at('edge-2', group_id=seeded.name) is not None

    @pytest.mark.asyncio
    async def test_update_edge_on_a_missing_edge_raises(self, seeded):
        with pytest.raises(EdgeNotFoundError):
            await seeded.backend.update_edge(
                MISSING_UUID, group_id=seeded.name, invalid_at=datetime(2026, 10, 30, tzinfo=UTC)
            )
