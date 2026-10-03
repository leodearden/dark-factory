"""Live FalkorDB: graphiti's edge search through FalkorEdgeSearch on a provisioned graph (task 6238).

With the production index set present (task 3708), FalkorDB plans graphiti-core's
stock edge legs as a scan re-executed once per input row.  These tests drive
graphiti's real ``search_utils`` legs through the hardened driver against a
seeded scratch graph provisioned by the production sweep, and pin the plan of the
Cypher actually issued (against a stock negative control), the rows, and for
BM25 a wall-clock budget.

Requires a running FalkorDB; skipped when one is not reachable, and deselected by
the default ``-m 'not integration'`` addopts.  Graphs and backends come only from
``_falkor_index_live``, which enforces the live-FalkorDB HAZARD rules.
"""

from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio
from _falkor_index_live import injected_driver, live_backends, scratch_graphs
from _fm_helpers import await_index_operational, falkor_skipif
from falkordb.asyncio.graph import AsyncGraph
from falkordb.execution_plan import ExecutionPlan
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import SearchFilters

pytestmark = [
    falkor_skipif(),
    pytest.mark.timeout(120),
    pytest.mark.integration,
]

# Large enough that stock BM25 (one index scan per hit x Entity node) overruns
# the server TIMEOUT: measured past 1000 ms at 800 x 800.
EDGE_COUNT = 800

# Not a graphiti stopword, alphanumeric, and a single RediSearch token.
CANARY_TOKEN = 'edgesearchcanary'

# Half of production's GRAPH.CONFIG TIMEOUT of 1000 ms.  The direct bind
# measured ~30 ms for all 800 hits on the seeded graph.
BM25_BUDGET_SECONDS = 0.5

SEEDED_AT = '2026-10-03T00:00:00+00:00'


@dataclass(frozen=True)
class SeededEdge:
    uuid: str
    source_node_uuid: str
    target_node_uuid: str
    fact_embedding: tuple[float, float, float]


def ring_embedding(index: int, count: int) -> tuple[float, float, float]:
    """A unit vector whose angle grows with *index* across the first octant."""
    angle = (math.pi / 4) * index / count
    return (math.cos(angle), math.sin(angle), 0.0)


def ring_edges(count: int) -> list[SeededEdge]:
    """Edge ``i`` runs node ``i`` -> node ``i % count + 1``: every node has one in- and one out-edge."""
    return [
        SeededEdge(
            uuid=f'edge-{i}',
            source_node_uuid=f'node-{i}',
            target_node_uuid=f'node-{i % count + 1}',
            fact_embedding=ring_embedding(i, count),
        )
        for i in range(1, count + 1)
    ]


def off_label_twin(edge: SeededEdge) -> SeededEdge:
    """*edge*'s embedding and canary fact, between nodes graphiti's edge search must not return."""
    return SeededEdge(
        uuid=f'{edge.uuid}-off-label',
        source_node_uuid='canary-source',
        target_node_uuid='canary-target',
        fact_embedding=edge.fact_embedding,
    )


def seed_edges_cypher(endpoint_label: str) -> str:
    return (
        'UNWIND $edges AS edge '
        f'MERGE (s:{endpoint_label} {{uuid: edge.source_node_uuid}}) '
        'ON CREATE SET s.name = edge.source_node_uuid, s.group_id = $group_id '
        f'MERGE (t:{endpoint_label} {{uuid: edge.target_node_uuid}}) '
        'ON CREATE SET t.name = edge.target_node_uuid, t.group_id = $group_id '
        'CREATE (s)-[:RELATES_TO {uuid: edge.uuid, group_id: $group_id, '
        "name: 'RELATES_TO', fact: edge.fact, episodes: ['ep'], created_at: $created_at, "
        'fact_embedding: vecf32(edge.fact_embedding)}]->(t)'
    )


async def seed_edges(
    graph: AsyncGraph, group_id: str, endpoint_label: str, edges: list[SeededEdge]
) -> None:
    rows = [
        {
            'uuid': edge.uuid,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'fact': f'{CANARY_TOKEN} fact {edge.uuid}',
            'fact_embedding': list(edge.fact_embedding),
        }
        for edge in edges
    ]
    await graph.query(
        seed_edges_cypher(endpoint_label),
        {'edges': rows, 'group_id': group_id, 'created_at': SEEDED_AT},
    )


@dataclass(frozen=True)
class SeededGraph:
    name: str
    graph: AsyncGraph
    driver: GraphDriver
    entity_edges: dict[str, SeededEdge]
    off_label_edge: SeededEdge

    def seeded_endpoints(self, edges: list[EntityEdge]) -> list[tuple[str, str]]:
        return [
            (
                self.entity_edges[edge.uuid].source_node_uuid,
                self.entity_edges[edge.uuid].target_node_uuid,
            )
            for edge in edges
        ]


def returned_endpoints(edges: list[EntityEdge]) -> list[tuple[str, str]]:
    return [(edge.source_node_uuid, edge.target_node_uuid) for edge in edges]


@pytest_asyncio.fixture
async def scratch():
    async with scratch_graphs('6238') as make:
        yield make


@pytest_asyncio.fixture
async def live_backend_factory(mock_config):
    async with live_backends(mock_config) as make:
        yield make


@pytest_asyncio.fixture
async def seed_ring(scratch, live_backend_factory) -> Callable[[int], Awaitable[SeededGraph]]:
    """``seed(count)``: a ring of *count* Entity edges plus an off-label twin, production-provisioned."""

    async def seed(count: int) -> SeededGraph:
        name, graph = scratch('ring')
        ring = ring_edges(count)
        off_label_edge = off_label_twin(ring[0])
        # Seeding also creates the graph KEY, which the provisioning sweep requires.
        await seed_edges(graph, name, 'Entity', ring)
        await seed_edges(graph, name, 'CanaryEndpoint', [off_label_edge])
        backend = live_backend_factory({name})
        await backend.provision_registered_graphs()
        await await_index_operational(graph)
        return SeededGraph(
            name=name,
            graph=graph,
            driver=injected_driver(backend).clone(name),
            entity_edges={edge.uuid: edge for edge in ring},
            off_label_edge=off_label_edge,
        )

    return seed


# --- Plan shape ---------------------------------------------------------------


@dataclass(frozen=True)
class IssuedQuery:
    cypher: str
    params: dict[str, Any]


SearchLeg = Callable[[GraphDriver], Awaitable[list[EntityEdge]]]


def spy_on_queries(driver: GraphDriver) -> list[IssuedQuery]:
    """Record every query *driver* issues while still executing it."""
    issued: list[IssuedQuery] = []
    execute = driver.execute_query

    async def spy(cypher: str, **params: Any):
        issued.append(IssuedQuery(cypher, params))
        return await execute(cypher, **params)

    driver.execute_query = spy  # pyright: ignore[reportAttributeAccessIssue]
    return issued


async def issued_by_hardened_driver(seeded: SeededGraph, leg: SearchLeg) -> IssuedQuery:
    issued = spy_on_queries(seeded.driver)
    await leg(seeded.driver)
    (query,) = issued
    return query


async def issued_by_stock_graphiti(leg: SearchLeg) -> IssuedQuery:
    """The Cypher stock graphiti emits: a bare FalkorDriver has no search_interface."""
    stock = object.__new__(FalkorDriver)
    issued: list[IssuedQuery] = []

    async def record(cypher: str, **params: Any):
        issued.append(IssuedQuery(cypher, params))
        return [], [], None

    stock.execute_query = record  # pyright: ignore[reportAttributeAccessIssue]
    await leg(stock)
    (query,) = issued
    return query


async def explain(graph: AsyncGraph, query: IssuedQuery) -> ExecutionPlan:
    params = {key: value for key, value in query.params.items() if key != 'routing_'}
    return await graph.explain(query.cypher, params)


def non_leaf_scans(plan: ExecutionPlan) -> list[str]:
    """Scan operators that have children; each is re-executed once per input row."""
    found: list[str] = []
    pending = [plan.structured_plan]
    while pending:
        operation = pending.pop()
        if operation.name.endswith('Scan') and operation.children:
            found.append(operation.name)
        pending.extend(operation.children)
    return found


# --- BM25 leg -----------------------------------------------------------------


def bm25_leg(group_id: str, limit: int) -> SearchLeg:
    return lambda driver: search_utils.edge_fulltext_search(
        driver, CANARY_TOKEN, SearchFilters(), [group_id], limit
    )


class TestEdgeFulltextLeg:
    @pytest.mark.asyncio
    async def test_issued_cypher_plans_without_a_per_row_scan(self, seed_ring):
        seeded = await seed_ring(EDGE_COUNT)

        query = await issued_by_hardened_driver(seeded, bm25_leg(seeded.name, 20))
        plan = await explain(seeded.graph, query)

        assert non_leaf_scans(plan) == [], str(plan)

    @pytest.mark.asyncio
    async def test_stock_cypher_plans_a_per_row_scan(self, seed_ring):
        """Negative control: without it the plan assertion above could pass vacuously."""
        seeded = await seed_ring(EDGE_COUNT)

        query = await issued_by_stock_graphiti(bm25_leg(seeded.name, 20))
        plan = await explain(seeded.graph, query)

        assert non_leaf_scans(plan) != [], str(plan)

    @pytest.mark.asyncio
    async def test_returns_topology_endpoints_within_budget(self, seed_ring):
        seeded = await seed_ring(EDGE_COUNT)

        started = time.perf_counter()
        edges = await bm25_leg(seeded.name, 20)(seeded.driver)
        elapsed = time.perf_counter() - started

        assert len(edges) == 20
        assert returned_endpoints(edges) == seeded.seeded_endpoints(edges)
        assert elapsed < BM25_BUDGET_SECONDS

    @pytest.mark.asyncio
    async def test_excludes_edges_between_non_entity_nodes(self, seed_ring):
        """Parity with stock's ``(n:Entity)-[e]->(m:Entity)``.

        A small ring, so stock graphiti finishes too and this test pins parity
        with it rather than only the rewrite's own behaviour.
        """
        seeded = await seed_ring(50)

        edges = await bm25_leg(seeded.name, 1000)(seeded.driver)

        returned = {edge.uuid for edge in edges}
        assert seeded.off_label_edge.uuid not in returned
        assert returned == set(seeded.entity_edges)
