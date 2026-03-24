"""Maintenance utility for re-indexing stale FalkorDB vector embeddings."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.embedder.openai import OpenAIEmbedderConfig

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.services.memory_service import MemoryService

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
        drop_indices: bool = False,
    ) -> dict:
        """Run reindex() then replay dead-lettered queue items.

        Args:
            durable_queue: A DurableWriteQueue instance.
            group_id: Optional group to scope the replay (None = all groups).
            drop_indices: When True, drop all VECTOR indices before re-embedding.
                This is required when the embedding model has changed and the old
                fixed-dimension indices must be rebuilt.  Defaults to False for
                backward compatibility.

        Returns:
            Dict with reindex_result (ReindexResult), replay_count (int), and
            indices_dropped (list of {label, field} dicts).
        """
        indices_dropped: list[dict] = []
        if drop_indices:
            logger.info('Dropping stale VECTOR indices before reindex')
            indices_dropped = await self.backend.drop_vector_indices()
            logger.info(f'Dropped {len(indices_dropped)} VECTOR index(es)')

        reindex_result = await self.reindex()
        replay_count = await durable_queue.replay_dead(group_id=group_id)
        logger.info(f'Replayed {replay_count} dead-letter items after reindex')
        return {
            'reindex_result': reindex_result,
            'replay_count': replay_count,
            'indices_dropped': indices_dropped,
        }


async def run_reindex(
    config_path: str | None = None,
    drop_indices: bool = False,
) -> dict:
    """Load config, initialize service, run reindex+replay, close resources.

    This is the CLI-callable entrypoint.  It:
    1. Loads FusedMemoryConfig (honouring CONFIG_PATH env var or config_path arg).
    2. Creates and initialises a MemoryService (which owns GraphitiBackend +
       DurableWriteQueue).
    3. Builds an OpenAIEmbedder with the dimension from config.
    4. Creates a ReindexManager and runs reindex_and_replay().
    5. Closes all resources in a finally block.

    Args:
        config_path: Optional path to the YAML config file.  When given it is
                     set as CONFIG_PATH before constructing FusedMemoryConfig.
        drop_indices: When True, drop all VECTOR indices before re-embedding.
                      Use this to fix a dimension mismatch (e.g. after changing
                      the embedding model).  Defaults to False.

    Returns:
        Dict with 'reindex_result' (ReindexResult), 'replay_count' (int), and
        'indices_dropped' (list of {label, field} dicts).
    """
    import os

    old_config_path = os.environ.get('CONFIG_PATH')
    service = None
    try:
        if config_path is not None:
            os.environ['CONFIG_PATH'] = config_path

        config = FusedMemoryConfig()
        service = MemoryService(config)

        # Build a dedicated embedder for re-embedding stale content.
        emb_cfg = config.embedder
        openai_provider = emb_cfg.providers.openai
        openai_api_key: str | None = None
        if openai_provider is not None:
            openai_api_key = openai_provider.api_key
        embedder_config = OpenAIEmbedderConfig(
            api_key=openai_api_key,
            embedding_model=emb_cfg.model,
            embedding_dim=emb_cfg.dimensions,
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        manager = ReindexManager(
            backend=service.graphiti,
            embedder=embedder,
            expected_dim=config.embedder.dimensions,
        )

        await service.initialize()
        result = await manager.reindex_and_replay(
            service.durable_queue,
            drop_indices=drop_indices,
        )
        ri = result['reindex_result']
        logger.info(
            f'run_reindex complete: nodes={ri.nodes_updated}, '
            f'edges={ri.edges_updated}, errors={ri.errors}, '
            f'replayed={result["replay_count"]}, '
            f'indices_dropped={len(result.get("indices_dropped", []))}'
        )
        return result
    finally:
        if service is not None:
            # Catch close() errors so CONFIG_PATH restoration below always runs.
            try:
                await service.close()
            except Exception:
                logger.warning('Error closing service during run_reindex cleanup', exc_info=True)
        if config_path is not None:
            if old_config_path is None:
                os.environ.pop('CONFIG_PATH', None)
            else:
                os.environ['CONFIG_PATH'] = old_config_path


if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Re-embed stale FalkorDB vectors and replay dead-letter queue'
    )
    parser.add_argument(
        '--drop-indices',
        action='store_true',
        default=False,
        help=(
            'Drop all VECTOR indices before re-embedding. '
            'Use this when the embedding model dimension has changed.'
        ),
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Path to the YAML config file (overrides CONFIG_PATH env var).',
    )
    args = parser.parse_args()
    asyncio.run(run_reindex(config_path=args.config, drop_indices=args.drop_indices))
