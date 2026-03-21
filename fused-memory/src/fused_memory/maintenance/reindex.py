"""Maintenance utility for re-indexing stale FalkorDB vector embeddings."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ReindexResult:
    """Result of a reindex operation."""

    nodes_updated: int = 0
    edges_updated: int = 0
    errors: int = 0


class ReindexManager:
    """Finds stale embeddings in FalkorDB and re-embeds them with the current model.

    After all stale embeddings are updated, dead-lettered queue items can be
    replayed successfully (deduplication cosine distance queries no longer
    hit dimension mismatches).

    Args:
        backend: A GraphitiBackend instance (must be initialized before use).
        embedder: An OpenAIEmbedder (or compatible) instance with a .create(text) method.
        expected_dim: The target embedding dimension (from config, typically 1536).
    """

    def __init__(self, backend, embedder, expected_dim: int) -> None:
        self.backend = backend
        self.embedder = embedder
        self.expected_dim = expected_dim

    async def reindex(self) -> ReindexResult:
        """Find all stale-dimension embeddings, re-embed them, and update FalkorDB.

        Returns:
            ReindexResult with counts of nodes_updated, edges_updated, and errors.
        """
        result = ReindexResult()

        # --- Re-index stale Entity nodes ---
        stale_nodes = await self.backend.query_stale_node_embeddings(
            expected_dim=self.expected_dim
        )
        logger.info(f'Found {len(stale_nodes)} stale node embeddings to reindex')
        for uuid, name, current_dim in stale_nodes:
            try:
                node_name, summary = await self.backend.get_node_text(uuid)
                text = f'{node_name} {summary}'
                embedding = await self.embedder.create(text)
                await self.backend.update_node_embedding(uuid, embedding)
                result.nodes_updated += 1
                logger.debug(f'Reindexed node {uuid!r} ({name!r}): {current_dim}→{self.expected_dim}')
            except Exception as exc:
                result.errors += 1
                logger.error(f'Failed to reindex node {uuid!r}: {exc}')

        # --- Re-index stale RELATES_TO edges ---
        stale_edges = await self.backend.query_stale_edge_embeddings(
            expected_dim=self.expected_dim
        )
        logger.info(f'Found {len(stale_edges)} stale edge embeddings to reindex')
        for uuid, name, current_dim in stale_edges:
            try:
                edge_name, fact = await self.backend.get_edge_text(uuid)
                text = f'{edge_name} {fact}'
                embedding = await self.embedder.create(text)
                await self.backend.update_edge_embedding(uuid, embedding)
                result.edges_updated += 1
                logger.debug(f'Reindexed edge {uuid!r} ({name!r}): {current_dim}→{self.expected_dim}')
            except Exception as exc:
                result.errors += 1
                logger.error(f'Failed to reindex edge {uuid!r}: {exc}')

        logger.info(
            f'Reindex complete: {result.nodes_updated} nodes, '
            f'{result.edges_updated} edges, {result.errors} errors'
        )
        return result

    async def reindex_and_replay(
        self,
        durable_queue,
        group_id: str | None = None,
    ) -> dict:
        """Run reindex() then replay dead-lettered queue items.

        Args:
            durable_queue: A DurableWriteQueue instance.
            group_id: Optional group to scope the replay (None = all groups).

        Returns:
            Dict with reindex_result (ReindexResult) and replay_count (int).
        """
        reindex_result = await self.reindex()
        replay_count = await durable_queue.replay_dead(group_id=group_id)
        logger.info(f'Replayed {replay_count} dead-letter items after reindex')
        return {
            'reindex_result': reindex_result,
            'replay_count': replay_count,
        }
