"""Live FalkorDB: graphiti's edge search and node BFS through FalkorEdgeSearch on a provisioned graph (tasks 6238, 6487).

With the production index set present (task 3708), FalkorDB plans graphiti-core's
stock edge legs as a scan re-executed once per input row.  These tests drive
graphiti's real ``search_utils`` legs through the hardened driver against a
seeded scratch graph provisioned by the production sweep, and pin the plan of the
Cypher actually issued (against a stock negative control), the rows, and for
BM25 a work budget: the records FalkorDB's profile counts per operator.

Requires a running FalkorDB; skipped when one is not reachable, and deselected by
the default ``-m 'not integration'`` addopts.  Graphs and backends come only from
``_falkor_index_live``, which enforces the live-FalkorDB HAZARD rules.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from typing import TypeVar

import pytest
import pytest_asyncio
from _falkor_index_live import live_backends, scratch_graphs
from _falkor_plans import explain, non_leaf_scans, peak_records_produced, profile, spy_on_queries
from _fm_falkor_live import await_index_operational, falkor_skipif
from _query_recorder import IssuedQuery, recording_driver
from falkordb.asyncio.graph import AsyncGraph
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import calculate_cosine_similarity

pytestmark = [
    falkor_skipif(),
    pytest.mark.integration,
]

# Large enough that SIMILARITY_MIN_SCORE cuts the cosine ranking between the two
# limits of test_returns_the_top_scoring_entity_edges: roughly 440 to 2000 edges.
EDGE_COUNT = 800

# Not a graphiti stopword, alphanumeric, and a single RediSearch token.
CANARY_TOKEN = 'edgesearchcanary'

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
        'ON CREATE SET s.name = edge.source_node_uuid, s.group_id = $group_id, '
        "s.created_at = $created_at, s.summary = '' "
        f'MERGE (t:{endpoint_label} {{uuid: edge.target_node_uuid}}) '
        'ON CREATE SET t.name = edge.target_node_uuid, t.group_id = $group_id, '
        "t.created_at = $created_at, t.summary = '' "
        'CREATE (s)-[:RELATES_TO {uuid: edge.uuid, group_id: $group_id, '
        "name: 'RELATES_TO', fact: edge.fact, episodes: ['ep'], created_at: $created_at, "
        'fact_embedding: vecf32(edge.fact_embedding)}]->(t)'
    )


def canary_fact(edge: SeededEdge) -> str:
    return f'{CANARY_TOKEN} fact {edge.uuid}'


async def seed_edges(
    graph: AsyncGraph,
    group_id: str,
    endpoint_label: str,
    edges: list[SeededEdge],
    fact_of: Callable[[SeededEdge], str] = canary_fact,
) -> None:
    rows = [
        {
            'uuid': edge.uuid,
            'source_node_uuid': edge.source_node_uuid,
            'target_node_uuid': edge.target_node_uuid,
            'fact': fact_of(edge),
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

    @property
    def canary_fact_count(self) -> int:
        """Edges whose fact carries CANARY_TOKEN: every entity edge and the off-label twin."""
        return len(self.entity_edges) + 1

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
async def provision(live_backend_factory) -> Callable[[str, AsyncGraph], Awaitable[GraphDriver]]:
    """``provision(name, graph)``: the production index sweep over a seeded graph, and its hardened driver.

    Seeding creates the graph KEY, which the provisioning sweep requires.
    """

    async def provisioned(name: str, graph: AsyncGraph) -> GraphDriver:
        backend = live_backend_factory({name})
        await backend.provision_registered_graphs()
        await await_index_operational(graph)
        return live_backend_factory.driver_of(backend).clone(name)

    return provisioned


@pytest_asyncio.fixture
async def seed_ring(scratch, provision) -> Callable[[int], Awaitable[SeededGraph]]:
    """``seed(count)``: a ring of *count* Entity edges plus an off-label twin, production-provisioned."""

    async def seed(count: int) -> SeededGraph:
        name, graph = scratch('ring')
        ring = ring_edges(count)
        off_label_edge = off_label_twin(ring[0])
        await seed_edges(graph, name, 'Entity', ring)
        await seed_edges(graph, name, 'CanaryEndpoint', [off_label_edge])
        return SeededGraph(
            name=name,
            graph=graph,
            driver=await provision(name, graph),
            entity_edges={edge.uuid: edge for edge in ring},
            off_label_edge=off_label_edge,
        )

    return seed


@dataclass(frozen=True)
class FactGraph:
    name: str
    graph: AsyncGraph
    driver: GraphDriver

    async def bm25_ranking(self, query: str, limit: int) -> list[str]:
        """The uuids graphiti's edge BM25 leg returns for *query*, best first."""
        edges = await search_utils.edge_fulltext_search(
            self.driver, query, SearchFilters(), [self.name], limit
        )
        return [edge.uuid for edge in edges]


@pytest_asyncio.fixture
async def seed_facts(scratch, provision) -> Callable[[dict[str, str]], Awaitable[FactGraph]]:
    """``seed(facts)``: one ring Entity edge per ``{uuid: fact}`` item, production-provisioned."""

    async def seed(facts: dict[str, str]) -> FactGraph:
        name, graph = scratch('facts')
        edges = [replace(edge, uuid=uuid) for edge, uuid in zip(ring_edges(len(facts)), facts, strict=True)]
        await seed_edges(graph, name, 'Entity', edges, lambda edge: facts[edge.uuid])
        return FactGraph(name=name, graph=graph, driver=await provision(name, graph))

    return seed


# --- Plan shape ---------------------------------------------------------------


SearchLeg = Callable[[GraphDriver], Awaitable[list[EntityEdge]]]
NodeSearchLeg = Callable[[GraphDriver], Awaitable[list[EntityNode]]]
AnyLeg = Callable[[GraphDriver], Awaitable[object]]


LegResult = TypeVar('LegResult')


async def run_on_hardened_driver(
    seeded: SeededGraph, leg: Callable[[GraphDriver], Awaitable[LegResult]]
) -> tuple[LegResult, IssuedQuery]:
    """*leg*'s rows through the hardened driver, and the one query it issued for them."""
    issued = spy_on_queries(seeded.driver)
    rows = await leg(seeded.driver)
    (query,) = issued
    return rows, query


async def issued_by_hardened_driver(seeded: SeededGraph, leg: AnyLeg) -> IssuedQuery:
    _, query = await run_on_hardened_driver(seeded, leg)
    return query


async def issued_by_stock_graphiti(leg: AnyLeg) -> IssuedQuery:
    """The Cypher stock graphiti emits: a bare FalkorDriver has no search_interface."""
    stock, issued = recording_driver(FalkorDriver)
    await leg(stock)
    (query,) = issued
    return query


def without_search_interface(driver: GraphDriver) -> GraphDriver:
    """*driver* on the same connection, taking graphiti's built-in search Cypher."""
    builtin = copy.copy(driver)
    builtin.search_interface = None
    return builtin


Row = TypeVar('Row')


async def hardened_and_builtin_rows(
    seeded: SeededGraph, leg: Callable[[GraphDriver], Awaitable[list[Row]]]
) -> tuple[list[Row], list[Row]]:
    """*leg*'s rows through the hardened driver, then through graphiti's built-in Cypher."""
    # Copy before spying, or the copy would carry the hardened driver's spy.
    builtin_driver = without_search_interface(seeded.driver)
    issued_by_builtin = spy_on_queries(builtin_driver)
    issued_by_hardened = spy_on_queries(seeded.driver)

    hardened = await leg(seeded.driver)
    builtin = await leg(builtin_driver)

    (hardened_query,) = issued_by_hardened
    (oracle_query,) = issued_by_builtin
    assert hardened_query.cypher != oracle_query.cypher, 'the oracle must run the built-in Cypher'
    return hardened, builtin


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
        """No operator of the issued Cypher's profile yields more records than the fulltext hits."""
        seeded = await seed_ring(EDGE_COUNT)

        edges, query = await run_on_hardened_driver(seeded, bm25_leg(seeded.name, 20))
        plan = await profile(seeded.graph, query)

        assert len(edges) == 20
        assert returned_endpoints(edges) == seeded.seeded_endpoints(edges)
        assert peak_records_produced(plan) == seeded.canary_fact_count, str(plan)

    @pytest.mark.asyncio
    async def test_stock_cypher_exceeds_the_budget(self, seed_ring):
        """Negative control: stock's per-row scan multiplies the hits, so the budget above discriminates.

        A small ring, so profiling stock's per-row scan stays cheap on the shared server.
        """
        seeded = await seed_ring(50)

        plan = await profile(seeded.graph, await issued_by_stock_graphiti(bm25_leg(seeded.name, 20)))

        assert peak_records_produced(plan) > seeded.canary_fact_count, str(plan)

    @pytest.mark.asyncio
    async def test_returns_the_edge_set_of_graphiti_builtin_cypher(self, seed_ring):
        """Parity with stock's ``(n:Entity)-[e]->(m:Entity)``, which excludes the off-label twin.

        A small ring, so the built-in per-row scan finishes inside the server TIMEOUT.
        """
        seeded = await seed_ring(50)

        edges, builtin_edges = await hardened_and_builtin_rows(seeded, bm25_leg(seeded.name, 1000))

        rewritten = {edge.uuid for edge in edges}
        assert rewritten == {edge.uuid for edge in builtin_edges} == set(seeded.entity_edges)
        assert seeded.off_label_edge.uuid not in rewritten


# --- BM25 ranking ------------------------------------------------------------

LONG_FACT = (
    'AllAccountsCappedException catch sites were added in the production orchestrator'
    ' via the task {} commit'
)

# Measured on scratch FalkorDB 41800 while planning task 9542: under a
# (@group_id:...) prefix long-rare is absent from the top 5; without it, rank 1.
RARE_TERM_FACTS = {
    **{f'alpha-{i}': f'task alpha note {i}' for i in range(20)},
    **{f'prose-{i}': 'context decisions related' for i in range(20)},
    'long-rare': LONG_FACT.format('zyzzogeton'),
    'long-id': LONG_FACT.format('877'),
}


# Measured on scratch FalkorDB 41800 while planning task 9542: the union alone
# ranks long-id 21st for IDENTIFIER_QUERY.
IDENTIFIER_FACTS = {
    **{f'prose-{i}': 'task context decisions related' for i in range(20)},
    'long-id': (
        'AllAccountsCappedException catch sites were added in the production orchestrator'
        ' via ticket 877 commit 371d4763 after the usage cap work landed'
    ),
    'filler-1': 'unrelated filler sentence one',
    'filler-2': 'unrelated filler sentence two',
}
IDENTIFIER_QUERY = 'task 877 context and related decisions'


class TestEdgeFulltextRanking:
    @pytest.mark.asyncio
    async def test_a_rare_term_outranks_common_terms_on_shorter_facts(self, seed_facts):
        seeded = await seed_facts(RARE_TERM_FACTS)

        ranking = await seeded.bm25_ranking('task zyzzogeton alpha', 5)

        assert ranking[0] == 'long-rare', ranking

    @pytest.mark.asyncio
    async def test_an_edge_matching_every_identifier_outranks_prose_edges(self, seed_facts):
        seeded = await seed_facts(IDENTIFIER_FACTS)

        ranking = await seeded.bm25_ranking(IDENTIFIER_QUERY, 5)

        assert ranking[0] == 'long-id', ranking

    @pytest.mark.asyncio
    async def test_identifiers_are_conjunctive(self, seed_facts):
        seeded = await seed_facts(IDENTIFIER_FACTS)

        ranking = await seeded.bm25_ranking('task 877 999 context and related decisions', 5)

        assert 'long-id' not in ranking, ranking

    @pytest.mark.asyncio
    async def test_tier_and_union_queries_plan_without_a_per_row_scan(self, seed_facts):
        seeded = await seed_facts(IDENTIFIER_FACTS)
        issued = spy_on_queries(seeded.driver)

        await seeded.bm25_ranking(IDENTIFIER_QUERY, 5)
        plans = [await explain(seeded.graph, query) for query in issued]

        assert len(plans) == 2, 'the tier query, then the union'
        assert [non_leaf_scans(plan) for plan in plans] == [[], []], [str(plan) for plan in plans]


# --- Cosine leg ---------------------------------------------------------------

# Outside the ring's octant, so neighbouring ring edges score >= 3e-4 apart, well
# clear of float32 rounding in vec.cosineDistance.
SIMILARITY_QUERY_VECTOR = (math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0)
SIMILARITY_MIN_SCORE = 0.84


def similarity_score(embedding: tuple[float, float, float]) -> float:
    """``vec.cosineDistance`` is ``1 - cos``; graphiti scores ``(2 - distance) / 2``."""
    cosine = calculate_cosine_similarity(list(SIMILARITY_QUERY_VECTOR), list(embedding))
    return (1 + cosine) / 2


def similarity_ranking(edges: Iterable[SeededEdge]) -> list[str]:
    """Uuids scoring above SIMILARITY_MIN_SCORE, best first."""
    scored = sorted(((similarity_score(edge.fact_embedding), edge.uuid) for edge in edges), reverse=True)
    return [uuid for score, uuid in scored if score > SIMILARITY_MIN_SCORE]


def similarity_leg(group_id: str, limit: int) -> SearchLeg:
    return lambda driver: search_utils.edge_similarity_search(
        driver,
        list(SIMILARITY_QUERY_VECTOR),
        None,
        None,
        SearchFilters(),
        [group_id],
        limit,
        SIMILARITY_MIN_SCORE,
    )


class TestEdgeSimilarityLeg:
    @pytest.mark.asyncio
    async def test_issued_cypher_plans_without_a_per_row_scan(self, seed_ring):
        """The discriminating gate for this leg; no wall-clock budget is asserted.

        Stock's per-row scan costs one index scan per Entity node, which stays
        fast at test scale (4.7 ms on 300 edges) and overruns the server TIMEOUT
        only at production size.  The plan shape differs at any size.
        """
        seeded = await seed_ring(EDGE_COUNT)

        query = await issued_by_hardened_driver(seeded, similarity_leg(seeded.name, 20))
        plan = await explain(seeded.graph, query)

        assert non_leaf_scans(plan) == [], str(plan)

    @pytest.mark.asyncio
    async def test_stock_cypher_plans_a_per_row_scan(self, seed_ring):
        """Negative control: without it the plan assertion above could pass vacuously."""
        seeded = await seed_ring(EDGE_COUNT)

        query = await issued_by_stock_graphiti(similarity_leg(seeded.name, 20))
        plan = await explain(seeded.graph, query)

        assert non_leaf_scans(plan) != [], str(plan)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('limit', [20, 100], ids=['cut-by-limit', 'cut-by-min-score'])
    async def test_returns_the_top_scoring_entity_edges(self, seed_ring, limit):
        seeded = await seed_ring(EDGE_COUNT)
        ranking = similarity_ranking(seeded.entity_edges.values())
        assert 20 < len(ranking) < 100, 'each case id must name the cut that applies'

        edges = await similarity_leg(seeded.name, limit)(seeded.driver)

        assert [edge.uuid for edge in edges] == ranking[:limit]
        assert returned_endpoints(edges) == seeded.seeded_endpoints(edges)
        assert seeded.off_label_edge.uuid not in {edge.uuid for edge in edges}


# --- BFS leg ------------------------------------------------------------------

BFS_DEPTH = 3
BFS_ENTITY_ORIGIN = 1
BFS_MENTIONED = 10
BFS_EPISODE_UUID = 'episode-bfs-origin'
BFS_ORIGINS = (f'node-{BFS_ENTITY_ORIGIN}', BFS_EPISODE_UUID)

SEED_EPISODE_CYPHER = (
    'MATCH (n:Entity {uuid: $mentioned}) '
    'CREATE (:Episodic {uuid: $episode, name: $episode, group_id: $group_id, created_at: $created_at})'
    '-[:MENTIONS {uuid: $mention, group_id: $group_id, created_at: $created_at}]->(n)'
)

# Fact edges into and out of a non-Entity node, on a path from the Entity
# origin to its ring successor: reached by the BFS, but not Entity-to-Entity.
BFS_DETOUR_EDGES = ('detour-in', 'detour-out')
BFS_DETOUR_NODE = 'detour-node'

SEED_DETOUR_CYPHER = (
    'MATCH (a:Entity {uuid: $origin}), (b:Entity {uuid: $successor}) '
    'CREATE (a)-[:RELATES_TO {uuid: $into, group_id: $group_id, name: $into, fact: $into, '
    "episodes: ['ep'], created_at: $created_at}]->"
    '(:CanaryEndpoint {uuid: $detour, name: $detour, group_id: $group_id})'
    '-[:RELATES_TO {uuid: $out_of, group_id: $group_id, name: $out_of, fact: $out_of, '
    "episodes: ['ep'], created_at: $created_at}]->(b)"
)


def ring_reachable(start: int, hops: int, count: int) -> set[str]:
    """The ring edge uuids on directed paths of 1..*hops* edges from node *start*."""
    return {f'edge-{(start - 1 + h) % count + 1}' for h in range(hops)}


def bfs_expected(count: int) -> set[str]:
    """Fact edges reached from both BFS origins; the MENTIONS hop spends one of the episode's hops."""
    return ring_reachable(BFS_ENTITY_ORIGIN, BFS_DEPTH, count) | ring_reachable(
        BFS_MENTIONED, BFS_DEPTH - 1, count
    )


def bfs_leg(group_id: str, limit: int) -> SearchLeg:
    return lambda driver: search_utils.edge_bfs_search(
        driver,
        list(BFS_ORIGINS),
        BFS_DEPTH,
        SearchFilters(),
        [group_id],
        limit,
    )


@pytest_asyncio.fixture
async def seed_bfs(seed_ring) -> Callable[[int], Awaitable[SeededGraph]]:
    """``seed(count)``: ``seed_ring`` plus an Episodic BFS origin that MENTIONS ring node
    BFS_MENTIONED, and the BFS_DETOUR_EDGES."""

    async def seed(count: int) -> SeededGraph:
        seeded = await seed_ring(count)
        await seeded.graph.query(
            SEED_EPISODE_CYPHER,
            {
                'mentioned': f'node-{BFS_MENTIONED}',
                'episode': BFS_EPISODE_UUID,
                'mention': 'mention-bfs-origin',
                'group_id': seeded.name,
                'created_at': SEEDED_AT,
            },
        )
        into, out_of = BFS_DETOUR_EDGES
        await seeded.graph.query(
            SEED_DETOUR_CYPHER,
            {
                'origin': f'node-{BFS_ENTITY_ORIGIN}',
                'successor': f'node-{BFS_ENTITY_ORIGIN + 1}',
                'into': into,
                'out_of': out_of,
                'detour': BFS_DETOUR_NODE,
                'group_id': seeded.name,
                'created_at': SEEDED_AT,
            },
        )
        return seeded

    return seed


class TestEdgeBfsLeg:
    @pytest.mark.asyncio
    async def test_issued_cypher_plans_without_a_per_row_scan(self, seed_bfs):
        """The discriminating gate for this leg; no wall-clock budget is asserted.

        Stock measured 134 ms on the 800-edge ring, inside the server TIMEOUT,
        so a budget cannot tell it from the rewrite at test scale.  The plan
        shape differs at any size.
        """
        seeded = await seed_bfs(EDGE_COUNT)

        query = await issued_by_hardened_driver(seeded, bfs_leg(seeded.name, 20))
        plan = await explain(seeded.graph, query)

        assert non_leaf_scans(plan) == [], str(plan)

    @pytest.mark.asyncio
    async def test_stock_cypher_plans_a_per_row_scan(self, seed_bfs):
        """Negative control: without it the plan assertion above could pass vacuously."""
        seeded = await seed_bfs(EDGE_COUNT)

        query = await issued_by_stock_graphiti(bfs_leg(seeded.name, 20))
        plan = await explain(seeded.graph, query)

        assert non_leaf_scans(plan) != [], str(plan)

    @pytest.mark.asyncio
    async def test_returns_the_edge_set_of_graphiti_builtin_cypher(self, seed_bfs):
        """Each reached Entity-to-Entity fact edge once, oriented as stored.

        The MENTIONS hop and the detour through a non-Entity node are traversed, not returned.
        """
        seeded = await seed_bfs(50)

        edges, builtin_edges = await hardened_and_builtin_rows(seeded, bfs_leg(seeded.name, 1000))

        rewritten = {edge.uuid for edge in edges}
        assert rewritten.isdisjoint(BFS_DETOUR_EDGES)
        assert rewritten == {edge.uuid for edge in builtin_edges} == bfs_expected(50)
        assert len(edges) == len(rewritten)
        assert returned_endpoints(edges) == seeded.seeded_endpoints(edges)


# --- Node BFS leg -------------------------------------------------------------

# An Entity outside the seeded group, one fact edge past the Entity origin.
BFS_FOREIGN_NODE = 'foreign-group-node'
BFS_FOREIGN_GROUP = 'foreign-group'

SEED_FOREIGN_CYPHER = (
    'MATCH (a:Entity {uuid: $origin}) '
    'CREATE (a)-[:RELATES_TO {uuid: $edge, group_id: $group_id, name: $edge, fact: $edge, '
    "episodes: ['ep'], created_at: $created_at}]->"
    "(:Entity {uuid: $foreign, name: $foreign, group_id: $foreign_group, created_at: $created_at, summary: ''})"
)


def ring_nodes_reached(start: int, hops: int, count: int) -> set[str]:
    """The ring node uuids 1..*hops* edges past node *start*."""
    return {f'node-{(start - 1 + h) % count + 1}' for h in range(1, hops + 1)}


def node_bfs_expected(count: int) -> set[str]:
    """Entity nodes reached from both BFS origins; the MENTIONS hop's target is itself reached."""
    return (
        ring_nodes_reached(BFS_ENTITY_ORIGIN, BFS_DEPTH, count)
        | {f'node-{BFS_MENTIONED}'}
        | ring_nodes_reached(BFS_MENTIONED, BFS_DEPTH - 1, count)
    )


def node_bfs_leg(
    group_ids: list[str] | None, limit: int, search_filter: SearchFilters | None = None
) -> NodeSearchLeg:
    return lambda driver: search_utils.node_bfs_search(
        driver,
        list(BFS_ORIGINS),
        search_filter or SearchFilters(),
        BFS_DEPTH,
        group_ids,
        limit,
    )


# One ring node reached only from the Entity origin, one only from the Episodic origin.
BFS_NODE_LABEL = 'BfsLabelled'
BFS_LABELLED_NODES = (f'node-{BFS_ENTITY_ORIGIN + 2}', f'node-{BFS_MENTIONED + 1}')

SEED_LABEL_CYPHER = f'MATCH (n:Entity) WHERE n.uuid IN $uuids SET n:{BFS_NODE_LABEL}'


@pytest_asyncio.fixture
async def seed_node_bfs(seed_bfs) -> Callable[[int], Awaitable[SeededGraph]]:
    """``seed(count)``: ``seed_bfs`` plus BFS_FOREIGN_NODE, one hop past the Entity origin.

    Kept out of ``seed_bfs``, where edge BFS would reach the foreign node's edge.
    """

    async def seed(count: int) -> SeededGraph:
        seeded = await seed_bfs(count)
        await seeded.graph.query(
            SEED_FOREIGN_CYPHER,
            {
                'origin': f'node-{BFS_ENTITY_ORIGIN}',
                'edge': 'foreign-edge',
                'foreign': BFS_FOREIGN_NODE,
                'foreign_group': BFS_FOREIGN_GROUP,
                'group_id': seeded.name,
                'created_at': SEEDED_AT,
            },
        )
        return seeded

    return seed


SEVERAL_PATHS_CYPHER = (
    f'MATCH path = (origin)-[:RELATES_TO|MENTIONS*1..{BFS_DEPTH}]->(n:Entity)'
    ' WHERE origin.uuid IN $origins'
    ' WITH n, count(path) AS paths WHERE paths > 1 RETURN n.uuid'
)


async def nodes_on_several_bfs_paths(seeded: SeededGraph) -> set[str]:
    """The Entity nodes that more than one path of the BFS links to BFS_ORIGINS."""
    result = await seeded.graph.query(SEVERAL_PATHS_CYPHER, {'origins': list(BFS_ORIGINS)})
    return {uuid for (uuid,) in result.result_set}


GroupIdsOf = Callable[[SeededGraph], list[str] | None]


@pytest.mark.parametrize(
    'group_ids_of',
    [lambda seeded: [seeded.name], lambda seeded: None],
    ids=['group-scoped', 'unscoped'],
)
class TestNodeBfsLeg:
    @pytest.mark.asyncio
    async def test_issued_cypher_plans_without_a_per_row_scan(
        self, seed_node_bfs, group_ids_of: GroupIdsOf
    ):
        """The discriminating gate for this leg; no wall-clock budget is asserted.

        Stock measured 10.9 ms on the 800-node ring, inside the server TIMEOUT,
        so a budget cannot tell it from the rewrite at test scale.  The plan
        shape differs at any size.
        """
        seeded = await seed_node_bfs(EDGE_COUNT)

        query = await issued_by_hardened_driver(seeded, node_bfs_leg(group_ids_of(seeded), 20))
        plan = await explain(seeded.graph, query)

        assert non_leaf_scans(plan) == [], str(plan)

    @pytest.mark.asyncio
    async def test_stock_cypher_plans_a_per_row_scan(self, seed_node_bfs, group_ids_of: GroupIdsOf):
        """Negative control: without it the plan assertion above could pass vacuously.

        Measured: a Node By Index Scan (group-scoped) or Node By Label Scan
        (unscoped) over the target ``n:Entity``, beneath the origin Unwind.
        """
        seeded = await seed_node_bfs(EDGE_COUNT)

        query = await issued_by_stock_graphiti(node_bfs_leg(group_ids_of(seeded), 20))
        plan = await explain(seeded.graph, query)

        assert non_leaf_scans(plan) != [], str(plan)

    @pytest.mark.asyncio
    async def test_returns_the_node_set_of_graphiti_builtin_cypher(
        self, seed_node_bfs, group_ids_of: GroupIdsOf
    ):
        """Each reached Entity node of the origin's group once.

        The detour's non-Entity node is traversed, not returned.  Unscoped, only
        ``n.group_id = origin.group_id`` excludes BFS_FOREIGN_NODE.
        """
        seeded = await seed_node_bfs(50)

        assert await nodes_on_several_bfs_paths(seeded), 'else one row per node holds vacuously'

        nodes, builtin_nodes = await hardened_and_builtin_rows(
            seeded, node_bfs_leg(group_ids_of(seeded), 1000)
        )

        rewritten = {node.uuid for node in nodes}
        assert rewritten.isdisjoint({BFS_DETOUR_NODE, BFS_FOREIGN_NODE})
        assert rewritten == {node.uuid for node in builtin_nodes} == node_bfs_expected(50)
        assert len(nodes) == len(rewritten)
        assert {node.group_id for node in nodes} == {seeded.name}

    @pytest.mark.asyncio
    async def test_label_filter_applies_to_nodes_reached_from_every_origin_label(
        self, seed_node_bfs, group_ids_of: GroupIdsOf
    ):
        """A filter one origin-label leg skipped would let that leg's unlabelled nodes through."""
        seeded = await seed_node_bfs(50)
        await seeded.graph.query(SEED_LABEL_CYPHER, {'uuids': list(BFS_LABELLED_NODES)})
        leg = node_bfs_leg(
            group_ids_of(seeded), 1000, SearchFilters(node_labels=[BFS_NODE_LABEL])
        )

        nodes, builtin_nodes = await hardened_and_builtin_rows(seeded, leg)

        rewritten = {node.uuid for node in nodes}
        assert rewritten == {node.uuid for node in builtin_nodes} == set(BFS_LABELLED_NODES)
