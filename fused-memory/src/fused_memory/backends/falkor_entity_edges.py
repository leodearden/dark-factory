"""Read graphiti entity edges with Cypher FalkorDB can plan (tasks 6238, 7714).

graphiti-core reaches an edge through a labelled-endpoint pattern,
``(n:Entity)-[e:RELATES_TO ...]->(m:Entity)``, which FalkorDB plans as a Node By
Label Scan driving one edge-index scan per Entity node.  Binding the endpoints
from the edge's own topology (``startNode(e)``, ``endNode(e)``) plans no such
scan, while keeping graphiti's ``:Entity`` restriction on both endpoints and the
edge's stored start -> end orientation.
"""

from collections.abc import Sequence
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query


async def read_entity_edges(
    driver: GraphDriver,
    edges: str,
    filter_queries: Sequence[str] = (),
    *,
    order_by: str | None = None,
    limit: int | None = None,
    **params: Any,
) -> list[EntityEdge]:
    """Run *edges* (Cypher leaving ``e`` bound, one row per edge) and return the entity edges.

    *filter_queries* are ANDed onto the ``:Entity`` endpoint checks.  Without
    *limit* every row is returned.  The projection and record parsing are
    graphiti's own.
    """
    ranking = f' ORDER BY {order_by}' if order_by else ''
    cap = ''
    if limit is not None:
        cap = ' LIMIT $limit'
        params['limit'] = limit
    cypher = (
        f'{edges} WITH *, startNode(e) AS n, endNode(e) AS m'
        f' WHERE {" AND ".join(["n:Entity", "m:Entity", *filter_queries])}'
        f' RETURN {get_entity_edge_return_query(driver.provider)}{ranking}{cap}'
    )
    records, _, _ = await driver.execute_query(cypher, routing_='r', **params)
    return [get_entity_edge_from_record(record, driver.provider) for record in records]
