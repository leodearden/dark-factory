"""Read a group's newest episodes in one bounded FalkorDB query with no range predicate.

Why it is neither keyset-paged nor a full scan: plans/falkordb-resultset-cap-audit.md,
the ``retrieve_episodes`` entry.
"""

from typing import Any

from graphiti_core.models.nodes.node_db_queries import EPISODIC_NODE_RETURN
from graphiti_core.nodes import EpisodicNode, get_episodic_node_from_record

_RECENT_GROUP_EPISODES_CYPHER = (
    'MATCH (e:Episodic) WHERE e.group_id IN $group_ids RETURN DISTINCT'
    + EPISODIC_NODE_RETURN
    + 'ORDER BY created_at DESC, uuid DESC LIMIT $limit'
)


async def read_recent_group_episodes(
    graph: Any, group_ids: list[str], last_n: int,
) -> list[EpisodicNode]:
    """Return at most *last_n* of *group_ids*' episodes, newest created_at first."""
    result = await graph.ro_query(
        _RECENT_GROUP_EPISODES_CYPHER, {'group_ids': group_ids, 'limit': last_n},
    )
    names = [column[1] for column in result.header]
    return [
        get_episodic_node_from_record(dict(zip(names, row, strict=True)))
        for row in result.result_set
    ]
