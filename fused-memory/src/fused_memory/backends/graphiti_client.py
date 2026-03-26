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
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.nodes import EpisodeType, EpisodicNode

from fused_memory.config.schema import FusedMemoryConfig

logger = logging.getLogger(__name__)


class NodeNotFoundError(Exception):
    """Raised when a node UUID is not found in FalkorDB."""


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

        assert self.client is not None
        await self.client.build_indices_and_constraints()
        await self.ensure_graph_timeout(falkor_cfg.graph_timeout_ms)
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
        """Delete an entity edge (fact) by UUID. Idempotent — missing edges are ignored."""
        client = self._require_client()
        try:
            edge = await EntityEdge.get_by_uuid(client.driver, edge_uuid)
        except EdgeNotFoundError:
            logger.info(f'Edge {edge_uuid} not found (already deleted or episode-cascaded)')
            return
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
        """Return (uuid, name, dim) for Entity nodes whose embedding dim != expected_dim.

        FalkorDB's ``size()`` does not work on Vectorf32 properties, so we
        return all nodes with embeddings and filter client-side by parsing the
        raw vector text representation (``<v1, v2, ...>``).
        """
        client = self._require_client()
        driver = cast(Any, client.driver)
        graph = driver._get_graph(None)
        cypher = (
            'MATCH (n:Entity) '
            'WHERE n.name_embedding IS NOT NULL '
            'RETURN n.uuid, n.name, n.name_embedding'
        )
        result = await graph.ro_query(cypher)
        stale: list[tuple[str, str, int]] = []
        for row in result.result_set or []:
            raw = row[2]
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='replace')
            dim = len(str(raw).strip('<>').split(', '))
            if dim != expected_dim:
                stale.append((row[0], row[1], dim))
        return stale

    async def query_stale_edge_embeddings(
        self, expected_dim: int
    ) -> list[tuple[str, str, int]]:
        """Return (uuid, name, dim) for RELATES_TO edges whose embedding dim != expected_dim.

        See ``query_stale_node_embeddings`` for why client-side filtering is needed.
        """
        client = self._require_client()
        driver = cast(Any, client.driver)
        graph = driver._get_graph(None)
        cypher = (
            'MATCH (n)-[e:RELATES_TO]->(m) '
            'WHERE e.fact_embedding IS NOT NULL '
            'RETURN e.uuid, e.name, e.fact_embedding'
        )
        result = await graph.ro_query(cypher)
        stale: list[tuple[str, str, int]] = []
        for row in result.result_set or []:
            raw = row[2]
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='replace')
            dim = len(str(raw).strip('<>').split(', '))
            if dim != expected_dim:
                stale.append((row[0], row[1], dim))
        return stale

    async def query_edges_by_time_range(
        self, start: str, end: str
    ) -> list[dict]:
        """Return edges whose valid_at falls within [start, end] (ISO 8601 strings).

        Args:
            start: ISO 8601 string for the lower bound (inclusive).
            end: ISO 8601 string for the upper bound (inclusive).

        Returns:
            List of dicts with keys: uuid, fact, name, valid_at, invalid_at.
        """
        client = self._require_client()
        driver = cast(Any, client.driver)
        graph = driver._get_graph(None)
        cypher = (
            'MATCH ()-[e:RELATES_TO]->() '
            'WHERE e.valid_at >= $start AND e.valid_at <= $end '
            'RETURN e.uuid, e.fact, e.name, e.valid_at, e.invalid_at'
        )
        result = await graph.query(cypher, {'start': start, 'end': end})
        return [
            {
                'uuid': row[0],
                'fact': row[1],
                'name': row[2],
                'valid_at': row[3],
                'invalid_at': row[4],
            }
            for row in (result.result_set or [])
        ]

    async def bulk_remove_edges(self, uuids: list[str]) -> int:
        """Delete RELATES_TO edges by UUID list. Returns count of actually matched edges.

        Uses a pre-count MATCH query before deletion to return the true number of
        edges that exist (and will be deleted), rather than the input list length.
        This is critical for irreversible operations where accuracy matters.

        Args:
            uuids: List of edge UUIDs to delete.

        Returns:
            Number of edges that matched (and were deleted). 0 for empty list.
        """
        if not uuids:
            return 0
        client = self._require_client()
        logger.info('Deleting %d edge(s)', len(uuids))
        logger.debug('Edge UUIDs to delete: %s', uuids)
        driver = cast(Any, client.driver)
        graph = driver._get_graph(None)
        # Pre-count: how many of the requested UUIDs actually exist as edges
        count_cypher = (
            'MATCH ()-[e:RELATES_TO]->() '
            'WHERE e.uuid IN $uuids '
            'RETURN count(e) AS found'
        )
        count_result = await graph.query(count_cypher, {'uuids': uuids})
        found = int(count_result.result_set[0][0]) if count_result.result_set else 0
        # Delete the edges
        delete_cypher = (
            'MATCH ()-[e:RELATES_TO]->() '
            'WHERE e.uuid IN $uuids '
            'DELETE e'
        )
        await graph.query(delete_cypher, {'uuids': uuids})
        return found

    async def get_node_text(self, uuid: str) -> tuple[str, str]:
        """Return (name, summary) for the Entity node with the given UUID.

        Raises:
            NodeNotFoundError: if no node with that UUID exists.
        """
        client = self._require_client()
        driver = cast(Any, client.driver)
        graph = driver._get_graph(None)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid}) '
            'RETURN n.name, n.summary'
        )
        result = await graph.query(cypher, {'uuid': uuid})
        if not result.result_set:
            raise NodeNotFoundError(f'Entity node not found: {uuid}')
        row = result.result_set[0]
        return (row[0], row[1] or '')

    async def get_edge_text(self, uuid: str) -> tuple[str, str]:
        """Return (name, fact) for the RELATES_TO edge with the given UUID.

        Raises:
            EdgeNotFoundError: if no edge with that UUID exists.
        """
        client = self._require_client()
        driver = cast(Any, client.driver)
        graph = driver._get_graph(None)
        cypher = (
            'MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() '
            'RETURN e.name, e.fact'
        )
        result = await graph.query(cypher, {'uuid': uuid})
        if not result.result_set:
            raise EdgeNotFoundError(f'RELATES_TO edge not found: {uuid}')
        row = result.result_set[0]
        return (row[0] or '', row[1] or '')

    async def update_node_embedding(self, uuid: str, embedding: list[float]) -> None:
        """Update the name_embedding vector for an Entity node using vecf32()."""
        client = self._require_client()
        driver = cast(Any, client.driver)
        graph = driver._get_graph(None)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid}) '
            'SET n.name_embedding = vecf32($embedding)'
        )
        await graph.query(cypher, {'uuid': uuid, 'embedding': embedding})

    async def update_edge_embedding(self, uuid: str, embedding: list[float]) -> None:
        """Update the fact_embedding vector for a RELATES_TO edge using vecf32()."""
        client = self._require_client()
        driver = cast(Any, client.driver)
        graph = driver._get_graph(None)
        cypher = (
            'MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() '
            'SET e.fact_embedding = vecf32($embedding)'
        )
        await graph.query(cypher, {'uuid': uuid, 'embedding': embedding})

    async def list_indices(self) -> list[dict]:
        """Return parsed index records from the graph.

        Each record is a dict with keys: label, field, type, entity_type.
        """
        client = self._require_client()
        driver = cast(Any, client.driver)
        graph = driver._get_graph(None)
        result = await graph.query('CALL db.indexes()')
        indices = []
        for row in (result.result_set or []):
            indices.append({
                'label': row[0],
                'field': row[1],
                'type': row[2],
                'entity_type': row[3],
            })
        return indices

    async def drop_index(self, label: str, field: str) -> None:
        """Drop an index on the given label and field (FalkorDB syntax)."""
        client = self._require_client()
        driver = cast(Any, client.driver)
        graph = driver._get_graph(None)
        cypher = f'DROP INDEX ON :{label}({field})'
        await graph.query(cypher)

    async def drop_vector_indices(self) -> list[dict]:
        """Drop all VECTOR-type indices in the graph.

        Calls list_indices() to find indices with type == 'VECTOR', then calls
        drop_index() for each.  Returns a list of {'label': ..., 'field': ...}
        dicts for each dropped index.
        """
        indices = await self.list_indices()
        dropped: list[dict] = []
        for entry in indices:
            if entry.get('type') == 'VECTOR':
                await self.drop_index(entry['label'], entry['field'])
                dropped.append({'label': entry['label'], 'field': entry['field']})
        logger.info(f'Dropped {len(dropped)} VECTOR index(es)')
        return dropped

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

    async def ensure_graph_timeout(self, timeout_ms: int) -> None:
        """Set the FalkorDB GRAPH.CONFIG TIMEOUT value.

        Sends ``GRAPH.CONFIG SET TIMEOUT <timeout_ms>`` via the underlying
        Redis client, then verifies with ``GRAPH.CONFIG GET TIMEOUT`` and logs
        the confirmed value.

        Args:
            timeout_ms: Timeout in milliseconds. 0 (or negative) disables the
                override entirely — no GRAPH.CONFIG command is sent.

        Raises:
            RuntimeError: if self._driver is None (backend not initialized).
        """
        if self._driver is None:
            raise RuntimeError('GraphitiBackend not initialized — call initialize() first')

        if timeout_ms <= 0:
            logger.debug('FalkorDB graph timeout override disabled (timeout_ms=%d)', timeout_ms)
            return

        try:
            await self._driver.client.execute_command('GRAPH.CONFIG', 'SET', 'TIMEOUT', timeout_ms)
            result = await self._driver.client.execute_command('GRAPH.CONFIG', 'GET', 'TIMEOUT')
            # result is typically [b'TIMEOUT', b'<value>']
            confirmed = result[1] if result and len(result) > 1 else timeout_ms
            logger.info('FalkorDB GRAPH.CONFIG TIMEOUT set to %s ms', confirmed)
        except Exception as exc:
            logger.warning(
                'Failed to set FalkorDB GRAPH.CONFIG TIMEOUT=%d: %s',
                timeout_ms,
                exc,
            )

    async def close(self) -> None:
        """Shut down the driver."""
        if self._driver is not None:
            with contextlib.suppress(Exception):
                await self._driver.close()
        self.client = None
        self._driver = None
