"""Thin async wrapper around the Graphiti client."""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import urlparse

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.embedder.openai import OpenAIEmbedderConfig
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.nodes import EpisodeType, EpisodicNode

from fused_memory.config.schema import FusedMemoryConfig

logger = logging.getLogger(__name__)


class GraphitiBackend:
    """Owns the Graphiti client lifecycle."""

    def __init__(self, config: FusedMemoryConfig):
        self.config = config
        self.client: Graphiti | None = None
        self._driver: FalkorDriver | None = None
        self._read_timeout: float = config.queue.backend_read_timeout_seconds
        self._write_timeout: float = config.queue.backend_write_timeout_seconds

    async def initialize(self) -> None:
        """Create FalkorDriver + Graphiti client from unified config."""
        cfg = self.config

        # --- LLM client ---
        llm_client = None
        if cfg.llm.provider == 'openai' and cfg.llm.providers.openai:
            api_key = cfg.llm.providers.openai.api_key
            if api_key:
                llm_config = GraphitiLLMConfig(
                    api_key=api_key,
                    model=cfg.llm.model,
                    small_model=cfg.llm.model,
                    temperature=cfg.llm.temperature or 0.0,
                    max_tokens=cfg.llm.max_tokens,
                )
                llm_client = OpenAIClient(config=llm_config)
                logger.info(f'Graphiti LLM: {cfg.llm.provider}/{cfg.llm.model}')
        elif cfg.llm.provider == 'anthropic' and cfg.llm.providers.anthropic:
            api_key = cfg.llm.providers.anthropic.api_key
            if api_key:
                try:
                    from graphiti_core.llm_client.anthropic_client import AnthropicClient

                    llm_config = GraphitiLLMConfig(
                        api_key=api_key,
                        model=cfg.llm.model,
                        temperature=cfg.llm.temperature or 0.0,
                        max_tokens=cfg.llm.max_tokens,
                    )
                    llm_client = AnthropicClient(config=llm_config)
                    logger.info(f'Graphiti LLM: {cfg.llm.provider}/{cfg.llm.model}')
                except ImportError:
                    logger.warning('Anthropic client not available for Graphiti')

        # --- Embedder ---
        embedder_client = None
        if cfg.embedder.provider == 'openai' and cfg.embedder.providers.openai:
            api_key = cfg.embedder.providers.openai.api_key
            if api_key:
                embedder_config = OpenAIEmbedderConfig(
                    api_key=api_key,
                    embedding_model=cfg.embedder.model,
                    base_url=cfg.embedder.providers.openai.api_url,
                    embedding_dim=cfg.embedder.dimensions,
                )
                embedder_client = OpenAIEmbedder(config=embedder_config)
                logger.info(f'Graphiti embedder: {cfg.embedder.provider}/{cfg.embedder.model}')

        # --- FalkorDB driver ---
        falkor_cfg = cfg.graphiti.falkordb
        parsed = urlparse(falkor_cfg.uri)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 6379

        self._driver = FalkorDriver(
            host=host,
            port=port,
            password=falkor_cfg.password,
            database=falkor_cfg.database,
        )

        self.client = Graphiti(
            graph_driver=self._driver,
            llm_client=llm_client,
            embedder=embedder_client,
            max_coroutines=cfg.queue.graphiti_max_coroutines,
        )

        await self.client.build_indices_and_constraints()
        logger.info(f'GraphitiBackend initialized (FalkorDB {host}:{port})')

    def _require_client(self) -> Graphiti:
        if self.client is None:
            raise RuntimeError('GraphitiBackend not initialized — call initialize() first')
        return self.client

    async def add_episode(
        self,
        name: str,
        content: str,
        source: EpisodeType = EpisodeType.text,
        group_id: str = 'main',
        source_description: str = '',
        reference_time: datetime | None = None,
        entity_types: dict | None = None,
        uuid: str | None = None,
        temporal_context: str | None = None,
    ) -> Any:
        """Add an episode to Graphiti and return the result."""
        client = self._require_client()
        ref_time = reference_time or datetime.now(UTC)
        if temporal_context is not None:
            source_description = f'[temporal:{temporal_context}] {source_description}'
        return await asyncio.wait_for(
            client.add_episode(
                name=name,
                episode_body=content,
                source=source,
                group_id=group_id,
                source_description=source_description,
                reference_time=ref_time,
                entity_types=entity_types,
                uuid=uuid,
            ),
            timeout=self._write_timeout,
        )

    async def search(
        self,
        query: str,
        group_ids: list[str] | None = None,
        num_results: int = 10,
        center_node_uuid: str | None = None,
    ) -> list[Any]:
        """Search for entity edges (facts)."""
        client = self._require_client()
        try:
            return await asyncio.wait_for(
                client.search(
                    query=query,
                    group_ids=group_ids or [],
                    num_results=num_results,
                    center_node_uuid=center_node_uuid,
                ),
                timeout=self._read_timeout,
            )
        except TimeoutError:
            logger.warning(f'Graphiti search timed out after {self._read_timeout}s')
            return []

    async def search_nodes(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_nodes: int = 10,
    ) -> list[Any]:
        """Search for entity nodes."""
        client = self._require_client()
        from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

        try:
            results = await asyncio.wait_for(
                client.search_(
                    query=query,
                    config=NODE_HYBRID_SEARCH_RRF,
                    group_ids=group_ids or [],
                ),
                timeout=self._read_timeout,
            )
            return (results.nodes or [])[:max_nodes]
        except TimeoutError:
            logger.warning(f'Graphiti search_nodes timed out after {self._read_timeout}s')
            return []

    async def retrieve_episodes(
        self,
        group_ids: list[str],
        last_n: int = 10,
        reference_time: datetime | None = None,
    ) -> list[Any]:
        """Retrieve recent episodes by group."""
        client = self._require_client()
        try:
            episodes = await asyncio.wait_for(
                EpisodicNode.get_by_group_ids(
                    client.driver, group_ids, limit=last_n
                ),
                timeout=self._read_timeout,
            )
            return episodes or []
        except TimeoutError:
            logger.warning(f'Graphiti retrieve_episodes timed out after {self._read_timeout}s')
            return []

    async def remove_episode(self, episode_uuid: str) -> None:
        """Delete an episode by UUID."""
        client = self._require_client()
        node = await EpisodicNode.get_by_uuid(client.driver, episode_uuid)
        await asyncio.wait_for(
            node.delete(client.driver),
            timeout=self._write_timeout,
        )

    async def remove_edge(self, edge_uuid: str) -> None:
        """Delete an entity edge (fact) by UUID."""
        client = self._require_client()
        edge = await EntityEdge.get_by_uuid(client.driver, edge_uuid)
        await asyncio.wait_for(
            edge.delete(client.driver),
            timeout=self._write_timeout,
        )

    async def build_communities(self, group_ids: list[str] | None = None) -> None:
        """Build community summaries."""
        client = self._require_client()
        await asyncio.wait_for(
            client.build_communities(group_ids=group_ids),
            timeout=self._write_timeout,
        )

    async def query_stale_node_embeddings(
        self, expected_dim: int
    ) -> list[tuple[str, str, int]]:
        """Return (uuid, name, dim) for Entity nodes whose embedding dim != expected_dim."""
        client = self._require_client()
        driver = cast(Any, client.driver)
        graph = driver._get_graph(None)
        cypher = (
            'MATCH (n:Entity) '
            'WHERE n.name_embedding IS NOT NULL '
            'AND size(n.name_embedding) <> $dim '
            'RETURN n.uuid, n.name, size(n.name_embedding) AS dim'
        )
        result = await graph.query(cypher, {'dim': expected_dim})
        return [
            (row[0], row[1], row[2])
            for row in (result.result_set or [])
        ]

    async def query_stale_edge_embeddings(
        self, expected_dim: int
    ) -> list[tuple[str, str, int]]:
        """Return (uuid, name, dim) for RELATES_TO edges whose embedding dim != expected_dim."""
        client = self._require_client()
        driver = cast(Any, client.driver)
        graph = driver._get_graph(None)
        cypher = (
            'MATCH (n)-[e:RELATES_TO]->(m) '
            'WHERE e.fact_embedding IS NOT NULL '
            'AND size(e.fact_embedding) <> $dim '
            'RETURN e.uuid, e.name, size(e.fact_embedding) AS dim'
        )
        result = await graph.query(cypher, {'dim': expected_dim})
        return [
            (row[0], row[1], row[2])
            for row in (result.result_set or [])
        ]

    async def list_graphs(self) -> list[str]:
        """Enumerate non-empty FalkorDB graphs (excluding default_db)."""
        client = self._require_client()
        all_graphs = await cast(Any, client.driver).client.list_graphs()
        return [g for g in all_graphs if g != 'default_db' and not g.endswith('_db')]

    async def node_count(self, graph_name: str) -> int:
        """Count nodes in a specific FalkorDB graph."""
        client = self._require_client()
        graph = cast(Any, client.driver)._get_graph(graph_name)
        result = await graph.query('MATCH (n) RETURN count(n) as count')
        return result.result_set[0][0] if result.result_set else 0

    async def close(self) -> None:
        """Shut down the driver."""
        if self._driver is not None:
            with contextlib.suppress(Exception):
                await self._driver.close()
        self.client = None
        self._driver = None
