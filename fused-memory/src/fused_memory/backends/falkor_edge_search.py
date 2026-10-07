"""graphiti ``search_interface`` for FalkorDB: edge search FalkorDB can plan (task 6238).

graphiti-core's stock edge legs reach each edge through a labeled-endpoint
pattern, e.g. ``(n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)``.  Once
the provisioned range indices exist (task 3708,
``docs/prds/falkordb-index-provisioning.md``), FalkorDB plans that as a full
Entity label scan driving one index scan per row, which overruns the server's
query TIMEOUT on production graphs.  Every edge leg here instead binds each
edge's endpoints from its own topology, which FalkorDB plans with no per-row
scan, and returns each edge once in its stored start -> end orientation.

Dispatch contract: ``graphiti_core.search.search_utils`` hands five legs to
``driver.search_interface`` UNCONDITIONALLY, with no NotImplementedError
fallback: edge fulltext, edge similarity, node fulltext, node similarity and
episode fulltext.  All five must therefore be overridden here; the three that
are not edge legs run graphiti's built-in Cypher on a copy of the driver
without the seam.  The remaining legs catch NotImplementedError and fall back
to the built-in Cypher on their own, so they are left alone, except edge BFS,
whose stock FalkorDB Cypher also plans per-row label scans (task 6252).
"""

import copy
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.graph_queries import get_relationships_query, get_vector_cosine_func_query
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import edge_search_filter_query_constructor


async def _entity_edges(
    driver: GraphDriver,
    edges: str,
    filter_queries: list[str],
    order_by: str | None = None,
    **params: Any,
) -> list[EntityEdge]:
    """Run *edges* (Cypher leaving ``e`` bound, one row per edge) and return up to ``$limit`` of them.

    Endpoints are bound from the edge's own topology rather than re-matched, and
    held to ``:Entity`` as graphiti's stock pattern does.  The projection and
    record parsing are graphiti's own.
    """
    ranking = f' ORDER BY {order_by}' if order_by else ''
    cypher = (
        f'{edges} WITH *, startNode(e) AS n, endNode(e) AS m'
        f' WHERE {" AND ".join(["n:Entity", "m:Entity", *filter_queries])}'
        f' RETURN {get_entity_edge_return_query(driver.provider)}{ranking} LIMIT $limit'
    )
    records, _, _ = await driver.execute_query(cypher, routing_='r', **params)
    return [get_entity_edge_from_record(record, driver.provider) for record in records]


# The origin labels graphiti's SearchInterface.edge_bfs_search documents.
_BFS_ORIGIN_LABELS = ('Entity', 'Episodic')


def _bfs_reached_edges(depth: int) -> str:
    """Every edge on a path of 1..*depth* hops from a BFS origin, once, bound as ``e``.

    One leg per origin label, so each looks its origins up through that label's
    uuid index; UNION dedupes edges reached along several paths.
    """
    legs = (
        f'MATCH path = (origin:{label})-[:RELATES_TO|MENTIONS*1..{depth}]->(:Entity)'
        ' WHERE origin.uuid IN $bfs_origin_node_uuids'
        ' UNWIND relationships(path) AS e RETURN e'
        for label in _BFS_ORIGIN_LABELS
    )
    return f'CALL {{ {" UNION ".join(legs)} }}'


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
        fuzzy_query = search_utils.fulltext_query(query, group_ids, driver)
        if fuzzy_query == '':
            return []

        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, driver.provider
        )
        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        fulltext_hits = get_relationships_query(
            'edge_name_and_fact', limit=limit, provider=driver.provider
        )
        return await _entity_edges(
            driver,
            f'{fulltext_hits} YIELD relationship AS e, score',
            filter_queries,
            order_by='score DESC',
            query=fuzzy_query,
            limit=limit,
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
        return await _entity_edges(
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
        # Paths also cross MENTIONS edges; stock returns only the RELATES_TO ones.
        filter_queries.insert(0, "type(e) = 'RELATES_TO'")
        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        return await _entity_edges(
            driver,
            _bfs_reached_edges(bfs_max_depth),
            filter_queries,
            bfs_origin_node_uuids=bfs_origin_node_uuids,
            limit=limit,
            **filter_params,
        )

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
