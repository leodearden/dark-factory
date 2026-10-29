"""graphiti ``graph_operations_interface`` for FalkorDB (task 7714).

Dispatch contract: graphiti-core hands six graph operations to
``driver.graph_operations_interface`` UNCONDITIONALLY, with no
NotImplementedError fallback: the four bulk saves in
``graphiti_core.utils.bulk_utils.add_nodes_and_edges_bulk_tx`` (the
``add_episode`` write path) and the two bulk embedding loads in
``graphiti_core.search.search_utils``.  Each is forwarded here to graphiti's
built-in behaviour.  Every other operation is inherited, raises
NotImplementedError, and graphiti falls back to its built-in Cypher.  The
sibling seam for search legs is :mod:`fused_memory.backends.falkor_edge_search`.
"""

import copy
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_episodic_edge_save_bulk_query,
)
from graphiti_core.models.nodes.node_db_queries import (
    get_entity_node_save_bulk_query,
    get_episode_node_save_bulk_query,
)
from graphiti_core.search import search_utils


def _builtin_operations_driver(driver: GraphDriver) -> GraphDriver:
    """A shallow copy of *driver* on which graphiti takes its built-in branch.

    Same idiom as upstream ``GraphDriver.with_database``: the copy shares the
    connection and every instance attribute, and the instance-level ``None``
    shadows the class-level graph_operations_interface.
    """
    view = copy.copy(driver)
    view.graph_operations_interface = None
    return view


class FalkorGraphOperations(GraphOperationsInterface):
    async def node_load_embeddings_bulk(
        self,
        driver: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        return await search_utils.get_embeddings_for_nodes(
            _builtin_operations_driver(driver), nodes
        )

    async def edge_load_embeddings_bulk(
        self,
        driver: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        return await search_utils.get_embeddings_for_edges(
            _builtin_operations_driver(driver), edges
        )

    async def episodic_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        await transaction.run(get_episode_node_save_bulk_query(driver.provider), episodes=nodes)

    async def node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        await transaction.run(get_entity_node_save_bulk_query(driver.provider, nodes), nodes=nodes)

    async def episodic_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        episodic_edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        await transaction.run(
            get_episodic_edge_save_bulk_query(driver.provider), episodic_edges=episodic_edges
        )

    async def edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        await transaction.run(get_entity_edge_save_bulk_query(driver.provider), entity_edges=edges)
