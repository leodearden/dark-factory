"""graphiti ``search_interface`` for FalkorDB: edge search FalkorDB can plan (task 6238).

graphiti-core's stock edge legs reach each edge through a labeled-endpoint
pattern, e.g. ``(n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)``.  Once
the provisioned range indices exist (task 3708,
``docs/prds/falkordb-index-provisioning.md``), FalkorDB plans that as a full
Entity label scan driving one index scan per row, which overruns the server's
query TIMEOUT on production graphs.  The edge legs are what this module exists
to replace.

Dispatch contract: ``graphiti_core.search.search_utils`` hands five legs to
``driver.search_interface`` UNCONDITIONALLY, with no NotImplementedError
fallback: edge fulltext, edge similarity, node fulltext, node similarity and
episode fulltext.  All five must therefore be overridden here, and every leg
not replaced runs graphiti's built-in Cypher on a copy of the driver without
the seam.  The remaining legs catch NotImplementedError and fall back to the
built-in Cypher on their own, so they are left alone.
"""

import copy
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.search import search_utils


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
        return await search_utils.edge_fulltext_search(
            _builtin_search_driver(driver), query, search_filter, group_ids, limit
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
        return await search_utils.edge_similarity_search(
            _builtin_search_driver(driver),
            search_vector,
            source_node_uuid,
            target_node_uuid,
            search_filter,
            group_ids,
            limit,
            min_score,
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
