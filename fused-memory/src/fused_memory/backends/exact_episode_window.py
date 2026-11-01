"""``graphiti_core.Graphiti.add_episode`` reads the episodes up to reference_time as extraction context through ``self.retrieve_episodes``.

Upstream's ``e.valid_at <= $reference_time`` is over-served by FalkorDB's RANGE
index, which handed backfills later episodes. This subclass makes that read
exact. See plans/falkordb-resultset-cap-audit.md, section 'String < / <= on
RANGE-indexed properties (task 7543)'.
"""

from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.decorators import handle_multiple_group_ids
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.models.nodes.node_db_queries import EPISODIC_NODE_RETURN
from graphiti_core.nodes import EpisodeType, EpisodicNode, get_episodic_node_from_record
from graphiti_core.utils.maintenance.graph_data_operations import EPISODE_WINDOW_LEN

from fused_memory.backends.falkor_string_bounds import exact_string_upper_bound

_VALID_AT_NOT_AFTER_REFERENCE = exact_string_upper_bound('e.valid_at', 'reference_time', inclusive=True)


class ExactEpisodeWindowGraphiti(Graphiti):
    """Graphiti whose previous-episode read returns only episodes valid at or before reference_time."""

    @handle_multiple_group_ids
    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = EPISODE_WINDOW_LEN,
        group_ids: list[str] | None = None,
        source: EpisodeType | None = None,
        driver: GraphDriver | None = None,
        saga: str | None = None,
    ) -> list[EpisodicNode]:
        if saga is not None:
            raise NotImplementedError(
                f'ExactEpisodeWindowGraphiti.retrieve_episodes does not support saga={saga!r}; '
                'fused-memory never passes saga'
            )
        driver = driver or self.clients.driver
        query = (
            'MATCH (e:Episodic) WHERE ' + _VALID_AT_NOT_AFTER_REFERENCE
            + (' AND e.group_id IN $group_ids' if group_ids else '')
            + (' AND e.source = $source' if source is not None else '')
            + ' RETURN' + EPISODIC_NODE_RETURN
            + 'ORDER BY e.valid_at DESC LIMIT $num_episodes'
        )
        records, _, _ = await driver.execute_query(
            query,
            reference_time=reference_time,
            num_episodes=last_n,
            group_ids=group_ids,
            source=source.name if source is not None else None,
        )
        return [get_episodic_node_from_record(record) for record in reversed(records)]
