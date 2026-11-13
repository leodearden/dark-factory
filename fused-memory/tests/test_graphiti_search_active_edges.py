"""GraphitiBackend.search ranks only active edges, so a retired edge takes no leg or fused slot (task 6523).

graphiti's edge search truncates each leg and the fused RRF list before any
caller sees a row.  A retired edge dropped only after that truncation still
displaced a live candidate.  The unit test pins that every edge leg Cypher the
public search issues carries the active-edge predicate.  The live test pins the
semantics: retired edges that lead both legs fill no slot.

The live class needs a running FalkorDB; it is skipped when none is reachable
and deselected by the default ``-m 'not integration'`` addopts.  Graphs and
backends come only from ``_falkor_index_live``, which enforces the
live-FalkorDB HAZARD rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _falkor_index_live import live_backends, scratch_graphs
from _fm_falkor_live import await_index_operational, falkor_skipif
from _fm_helpers import extract_cypher
from falkordb.asyncio.graph import AsyncGraph
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.llm_client.client import LLMClient

from fused_memory.backends.graphiti_client import _ACTIVE_EDGE_PREDICATE

QUERY_VECTOR = (1.0, 0.0, 0.0)


def edge_leg_cyphers(graph: MagicMock) -> tuple[list[str], list[str]]:
    """The (BM25, cosine) edge-leg Cypher strings issued on *graph*."""
    issued = [
        extract_cypher(call)
        for call in [*graph.ro_query.call_args_list, *graph.query.call_args_list]
    ]
    bm25 = [cypher for cypher in issued if 'queryRelationships' in cypher]
    cosine = [cypher for cypher in issued if 'fact_embedding' in cypher]
    return bm25, cosine


class TestSearchRanksOnlyActiveEdges:
    @pytest.mark.asyncio
    async def test_every_edge_leg_cypher_restricts_to_active_edges(
        self, mock_config, make_graph_mock, make_graph_driver, make_wired_backend,
    ):
        graph = make_graph_mock([])
        driver = make_graph_driver({'g': graph})
        backend, subs = make_wired_backend(mock_config, driver)
        subs.embedder.create = AsyncMock(return_value=list(QUERY_VECTOR))
        backend.client = Graphiti(graph_driver=driver, **subs._asdict())

        await backend.search('task 877 context', group_ids=['g'], num_results=8)

        bm25, cosine = edge_leg_cyphers(graph)
        assert bm25, 'no BM25 edge leg was issued'
        assert cosine, 'no cosine edge leg was issued'
        unfiltered = [
            cypher for cypher in [*bm25, *cosine] if _ACTIVE_EDGE_PREDICATE not in cypher
        ]
        assert unfiltered == []


# --- Live FalkorDB ------------------------------------------------------------


# Not a graphiti stopword, alphanumeric, and a single RediSearch token.
TOKEN = 'activeedgecanary'

SEEDED_AT = '2026-11-12T00:00:00+00:00'
RETIRED_AT = '2026-11-12T01:00:00+00:00'


@dataclass(frozen=True)
class SeededEdge:
    uuid: str
    fact: str
    angle_degrees: float
    retired_at: str | None

    def as_row(self) -> dict[str, object]:
        angle = math.radians(self.angle_degrees)
        return {
            'uuid': self.uuid,
            'source_node_uuid': f'{self.uuid}-source',
            'target_node_uuid': f'{self.uuid}-target',
            'fact': self.fact,
            'fact_embedding': [math.cos(angle), math.sin(angle), 0.0],
            'retired_at': self.retired_at,
        }


# Short facts outrank long ones for one shared token, and these embeddings sit
# closest to QUERY_VECTOR: the retired edges lead both legs.
RETIRED_EDGES = tuple(
    SeededEdge(f'retired-{i}', f'{TOKEN} retired {i}', angle_degrees=i, retired_at=RETIRED_AT)
    for i in range(4)
)

# Cosine 0.34-0.5 is still above graphiti's 0.6 floor on (1 + cos) / 2.
LIVE_EDGES = tuple(
    SeededEdge(
        f'live-{i}',
        f'{TOKEN} live fact {i} padded with several extra ordinary words',
        angle_degrees=60 + 5 * i,
        retired_at=None,
    )
    for i in range(3)
)

SEED_EDGES_CYPHER = (
    'UNWIND $edges AS edge '
    'MERGE (s:Entity {uuid: edge.source_node_uuid}) '
    'ON CREATE SET s.name = edge.source_node_uuid, s.group_id = $group_id, '
    "s.created_at = $created_at, s.summary = '' "
    'MERGE (t:Entity {uuid: edge.target_node_uuid}) '
    'ON CREATE SET t.name = edge.target_node_uuid, t.group_id = $group_id, '
    "t.created_at = $created_at, t.summary = '' "
    'CREATE (s)-[r:RELATES_TO {uuid: edge.uuid, group_id: $group_id, '
    "name: 'RELATES_TO', fact: edge.fact, episodes: ['ep'], created_at: $created_at, "
    'fact_embedding: vecf32(edge.fact_embedding)}]->(t) '
    'SET r.invalid_at = edge.retired_at, r.expired_at = edge.retired_at'
)


async def seed_edges(graph: AsyncGraph, group_id: str, edges: tuple[SeededEdge, ...]) -> None:
    await graph.query(
        SEED_EDGES_CYPHER,
        {
            'edges': [edge.as_row() for edge in edges],
            'group_id': group_id,
            'created_at': SEEDED_AT,
        },
    )


class QueryVectorEmbedder(EmbedderClient):
    async def create(self, input_data) -> list[float]:
        return list(QUERY_VECTOR)


@pytest_asyncio.fixture
async def scratch():
    async with scratch_graphs('6523') as make:
        yield make


@pytest_asyncio.fixture
async def live_backend_factory(mock_config):
    async with live_backends(mock_config) as make:
        yield make


@falkor_skipif()
@pytest.mark.integration
class TestSearchRanksOnlyActiveEdgesLive:
    @pytest.mark.asyncio
    async def test_retired_edges_leading_both_legs_take_no_slot(
        self, scratch, live_backend_factory,
    ):
        name, graph = scratch('active_edges')
        # Seeding also creates the graph KEY, which the provisioning sweep requires.
        await seed_edges(graph, name, (*RETIRED_EDGES, *LIVE_EDGES))
        backend = live_backend_factory({name})
        await backend.provision_registered_graphs()
        await await_index_operational(graph)
        backend.client = Graphiti(
            graph_driver=live_backend_factory.driver_of(backend),
            llm_client=MagicMock(spec=LLMClient),
            embedder=QueryVectorEmbedder(),
            cross_encoder=MagicMock(spec=CrossEncoderClient),
        )

        edges = await backend.search(TOKEN, group_ids=[name], num_results=len(LIVE_EDGES))

        assert {edge.uuid for edge in edges} == {edge.uuid for edge in LIVE_EDGES}
        assert [edge.invalid_at for edge in edges] == [None] * len(LIVE_EDGES)
