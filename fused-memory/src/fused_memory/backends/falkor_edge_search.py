"""graphiti ``search_interface`` for FalkorDB: search legs FalkorDB can plan (tasks 6238, 6487).

graphiti-core's stock edge legs reach each edge through a labeled-endpoint
pattern, e.g. ``(n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)``.  Once
the provisioned range indices exist (task 3708,
``docs/prds/falkordb-index-provisioning.md``), FalkorDB plans that as a full
Entity label scan driving one index scan per row, which overruns the server's
query TIMEOUT on production graphs.  Every edge leg here instead reads its edges
through :func:`fused_memory.backends.falkor_entity_edges.read_entity_edges`,
which FalkorDB plans with no per-row scan.

The edge BM25 leg ranks first the live edges matching every identifier term of
the query, when they all fit in the leg
(:func:`fused_memory.backends.falkor_fulltext.build_edge_queries`,
``plans/edge-bm25-leg-ranking-9542/README.md``).

Dispatch contract: ``graphiti_core.search.search_utils`` hands five legs to
``driver.search_interface`` UNCONDITIONALLY, with no NotImplementedError
fallback: edge fulltext, edge similarity, node fulltext, node similarity and
episode fulltext.  All five must therefore be overridden here; the three that
are not edge legs run graphiti's built-in Cypher on a copy of the driver
without the seam.  The remaining legs catch NotImplementedError and fall back
to the built-in Cypher on their own, so they are left alone, except the two BFS
legs, edge BFS (task 6252) and node BFS (task 6487), whose stock FalkorDB Cypher
plans per-row scans.
"""

import copy
from collections.abc import Callable
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.graph_queries import get_relationships_query, get_vector_cosine_func_query
from graphiti_core.helpers import validate_group_ids
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
from graphiti_core.nodes import get_entity_node_from_record
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import (
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

from fused_memory.backends.falkor_entity_edges import bind_entity_endpoints, read_entity_edges
from fused_memory.backends.falkor_fulltext import build_edge_queries
from fused_memory.backends.relates_to_schema import ACTIVE_EDGE_PREDICATE

# The origin labels graphiti's SearchInterface.edge_bfs_search documents; edge
# BFS and node BFS both start from them.
_BFS_ORIGIN_LABELS = ('Entity', 'Episodic')


def _bfs_hops(depth: int) -> str:
    return f'-[:RELATES_TO|MENTIONS*1..{depth}]->'


def _union_over_origin_labels(leg: Callable[[str], str]) -> str:
    """``CALL { leg('Entity') UNION leg('Episodic') }``: the BFS origin lookup FalkorDB can plan.

    One leg per origin label, so each looks its origins up through that label's
    uuid index as a plan leaf; UNION dedupes rows reached along several paths.
    """
    return f'CALL {{ {" UNION ".join(leg(label) for label in _BFS_ORIGIN_LABELS)} }}'


def _bfs_reached_edges(depth: int) -> str:
    """Every fact edge on a path of 1..*depth* hops from a BFS origin, once, bound as ``e``.

    Paths also cross MENTIONS edges, which are traversed but not returned.
    """

    def leg(label: str) -> str:
        return (
            f'MATCH path = (origin:{label}){_bfs_hops(depth)}(:Entity)'
            ' WHERE origin.uuid IN $bfs_origin_node_uuids'
            " UNWIND relationships(path) AS e WITH e WHERE type(e) = 'RELATES_TO' RETURN e"
        )

    return _union_over_origin_labels(leg)


def _bfs_reached_nodes(depth: int, filter_queries: list[str]) -> str:
    """Every Entity node of its origin's group 1..*depth* hops from a BFS origin, once, bound as ``n``.

    *filter_queries* may name ``origin``, which the CALL does not export, so
    every leg applies them.
    """
    predicates = [
        'origin.uuid IN $bfs_origin_node_uuids',
        'n.group_id = origin.group_id',
        *filter_queries,
    ]

    def leg(label: str) -> str:
        return (
            f'MATCH (origin:{label}){_bfs_hops(depth)}(n:Entity)'
            f' WHERE {" AND ".join(predicates)} RETURN n'
        )

    return _union_over_origin_labels(leg)


async def _promoted_uuids(
    driver: Any,
    identifiers: str,
    filter_queries: list[str],
    limit: int,
    params: dict[str, Any],
) -> list[str]:
    """The live edges matching every identifier term, promoted only when all of them fit in the leg.

    Under the leg's own filters, plus ``relates_to_schema.py::ACTIVE_EDGE_PREDICATE``:
    the read path discards retired edges after fusion, so promoting one would
    spend a slot.
    ``limit + 1`` rows are read so an overflowing tier is told apart from one
    that fits.
    """
    if identifiers == '':
        return []
    fulltext_hits = get_relationships_query(
        'edge_name_and_fact', limit=limit, provider=driver.provider
    )
    records, _, _ = await driver.execute_query(
        bind_entity_endpoints(
            f'{fulltext_hits} YIELD relationship AS e',
            [*filter_queries, ACTIVE_EDGE_PREDICATE],
        )
        + ' RETURN e.uuid AS uuid LIMIT $tier_cap',
        query=identifiers,
        tier_cap=limit + 1,
        routing_='r',
        **params,
    )
    uuids = [record['uuid'] for record in records]
    return uuids if len(uuids) <= limit else []


def _builtin_search_driver(driver: GraphDriver) -> GraphDriver:
    """A shallow copy of *driver* on which graphiti takes its built-in branch.

    Same idiom as upstream ``GraphDriver.with_database``: the copy shares the
    connection and every instance attribute, and the instance-level ``None``
    shadows the class-level search_interface.
    """
    view = copy.copy(driver)
    view.search_interface = None
    return view


class FalkorEdgeSearch(SearchInterface):
    async def edge_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        validate_group_ids(group_ids)
        queries = build_edge_queries(
            driver.sanitize(query), group_ids, search_utils.MAX_QUERY_LENGTH
        )
        if queries.union == '':
            return []

        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, driver.provider
        )
        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        promoted_uuids = await _promoted_uuids(
            driver, queries.identifiers, filter_queries, limit, filter_params
        )
        fulltext_hits = get_relationships_query(
            'edge_name_and_fact', limit=limit, provider=driver.provider
        )
        return await read_entity_edges(
            driver,
            f'{fulltext_hits} YIELD relationship AS e, score',
            filter_queries,
            order_by='e.uuid IN $promoted_uuids DESC, score DESC',
            query=queries.union,
            limit=limit,
            promoted_uuids=promoted_uuids,
            **filter_params,
        )

    async def edge_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, driver.provider
        )
        group_filter = ''
        if group_ids is not None:
            group_filter = ' WHERE e.group_id IN $group_ids'
            filter_params['group_ids'] = group_ids
            if source_node_uuid is not None:
                filter_params['source_uuid'] = source_node_uuid
                filter_queries.append('n.uuid = $source_uuid')
            if target_node_uuid is not None:
                filter_params['target_uuid'] = target_node_uuid
                filter_queries.append('m.uuid = $target_uuid')

        score = get_vector_cosine_func_query('e.fact_embedding', '$search_vector', driver.provider)
        return await read_entity_edges(
            driver,
            f'MATCH ()-[e:RELATES_TO]->(){group_filter}'
            f' WITH e, {score} AS score WHERE score > $min_score',
            filter_queries,
            order_by='score DESC',
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            **filter_params,
        )

    async def edge_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        bfs_max_depth: int,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not bfs_origin_node_uuids or bfs_max_depth < 1:
            return []

        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, driver.provider
        )
        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        return await read_entity_edges(
            driver,
            _bfs_reached_edges(bfs_max_depth),
            filter_queries,
            bfs_origin_node_uuids=bfs_origin_node_uuids,
            limit=limit,
            **filter_params,
        )

    async def node_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        search_filter: Any,
        bfs_max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        if not bfs_origin_node_uuids or bfs_max_depth < 1:
            return []

        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, driver.provider
        )
        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_queries.append('origin.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        records, _, _ = await driver.execute_query(
            f'{_bfs_reached_nodes(bfs_max_depth, filter_queries)}'
            f' RETURN {get_entity_node_return_query(driver.provider)} LIMIT $limit',
            bfs_origin_node_uuids=bfs_origin_node_uuids,
            limit=limit,
            routing_='r',
            **filter_params,
        )
        return [get_entity_node_from_record(record, driver.provider) for record in records]

    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await search_utils.node_fulltext_search(
            _builtin_search_driver(driver), query, search_filter, group_ids, limit
        )

    async def node_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        return await search_utils.node_similarity_search(
            _builtin_search_driver(driver), search_vector, search_filter, group_ids, limit, min_score
        )

    async def episode_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        return await search_utils.episode_fulltext_search(
            _builtin_search_driver(driver), query, search_filter, group_ids, limit
        )
