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
    """Owns the Graphiti client lifecycle.

    FalkorDB is multi-tenant: each project's data lives in its own graph
    (named after the project_id / group_id).  The driver is cloned per-request
    so every operation targets the correct graph.
    """

    def __init__(self, config: FusedMemoryConfig):
        self.config = config
        self.client: Graphiti | None = None
        self._driver: FalkorDriver | None = None
        self._read_timeout: float = config.queue.backend_read_timeout_seconds
        self._write_timeout: float = config.queue.backend_write_timeout_seconds
        self._indexed_graphs: set[str] = set()

    # --- Per-request driver routing ---

    def _driver_for(self, group_id: str) -> FalkorDriver:
        """Return a driver clone targeting the FalkorDB graph for *group_id*.

        Lazily ensures indices exist on the target graph (tracked in
        ``_indexed_graphs`` to avoid redundant CREATE INDEX calls).
        """
        driver = self._require_driver()
        return driver.clone(database=group_id)

    def _graph_for(self, group_id: str) -> Any:
        """Return the FalkorGraph object for *group_id* (for direct Cypher)."""
        driver = self._require_driver()
        return driver._get_graph(group_id)

    def _require_driver(self) -> FalkorDriver:
        if self._driver is None:
            raise RuntimeError('GraphitiBackend not initialized — call initialize() first')
        return self._driver

    async def _ensure_indices(self, group_id: str) -> None:
        """Build indices on *group_id*'s graph if not already done this session."""
        if group_id in self._indexed_graphs:
            return
        driver = self._driver_for(group_id)
        await driver.build_indices_and_constraints()
        self._indexed_graphs.add(group_id)
        logger.debug('Ensured indices on graph %r', group_id)

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
        # The driver is created with a placeholder database.  Actual graph
        # selection happens per-request via _driver_for() / _graph_for().
        falkor_cfg = cfg.graphiti.falkordb
        if falkor_cfg.database is not None:
            logger.warning(
                'graphiti.falkordb.database=%r is ignored — graph name is '
                'derived from group_id at request time',
                falkor_cfg.database,
            )
        parsed = urlparse(falkor_cfg.uri)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 6379

        self._driver = FalkorDriver(
            host=host,
            port=port,
            password=falkor_cfg.password,
            database='__placeholder__',
        )

        self.client = Graphiti(
            graph_driver=self._driver,
            llm_client=llm_client,
            embedder=embedder_client,
            max_coroutines=cfg.queue.graphiti_max_coroutines,
        )

        # Build indices on all existing project graphs (lazy set avoids repeats).
        try:
            existing = await cast(Any, self._driver).client.list_graphs()
            for graph_name in existing:
                if graph_name != 'default_db' and not graph_name.endswith('_db'):
                    await self._ensure_indices(graph_name)
        except Exception:
            logger.warning('Could not enumerate existing graphs for index setup', exc_info=True)

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
        gids = group_ids or []
        driver = self._driver_for(gids[0]) if gids else None
        try:
            return await asyncio.wait_for(
                client.search(
                    query=query,
                    group_ids=gids,
                    num_results=num_results,
                    center_node_uuid=center_node_uuid,
                    driver=driver,
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

        gids = group_ids or []
        driver = self._driver_for(gids[0]) if gids else None
        try:
            results = await asyncio.wait_for(
                client.search_(
                    query=query,
                    config=NODE_HYBRID_SEARCH_RRF,
                    group_ids=gids,
                    driver=driver,
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
        driver = self._driver_for(group_ids[0]) if group_ids else self._require_driver()
        try:
            episodes = await asyncio.wait_for(
                EpisodicNode.get_by_group_ids(
                    driver, group_ids, limit=last_n
                ),
                timeout=self._read_timeout,
            )
            return episodes or []
        except TimeoutError:
            logger.warning(f'Graphiti retrieve_episodes timed out after {self._read_timeout}s')
            return []

    async def remove_episode(self, episode_uuid: str, *, group_id: str) -> None:
        """Delete an episode by UUID."""
        driver = self._driver_for(group_id)
        node = await EpisodicNode.get_by_uuid(driver, episode_uuid)
        await asyncio.wait_for(
            node.delete(driver),
            timeout=self._write_timeout,
        )

    async def remove_edge(self, edge_uuid: str, *, group_id: str) -> None:
        """Delete an entity edge (fact) by UUID. Idempotent — missing edges are ignored."""
        driver = self._driver_for(group_id)
        try:
            edge = await EntityEdge.get_by_uuid(driver, edge_uuid)
        except EdgeNotFoundError:
            logger.info(f'Edge {edge_uuid} not found (already deleted or episode-cascaded)')
            return
        await asyncio.wait_for(
            edge.delete(driver),
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
        self, expected_dim: int, *, group_id: str
    ) -> list[tuple[str, str, int]]:
        """Return (uuid, name, dim) for Entity nodes whose embedding dim != expected_dim.

        FalkorDB's ``size()`` does not work on Vectorf32 properties, so we
        return all nodes with embeddings and filter client-side by parsing the
        raw vector text representation (``<v1, v2, ...>``).
        """
        graph = self._graph_for(group_id)
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
        self, expected_dim: int, *, group_id: str
    ) -> list[tuple[str, str, int]]:
        """Return (uuid, name, dim) for RELATES_TO edges whose embedding dim != expected_dim.

        See ``query_stale_node_embeddings`` for why client-side filtering is needed.
        """
        graph = self._graph_for(group_id)
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
        self, start: str, end: str, *, group_id: str
    ) -> list[dict]:
        """Return edges whose valid_at falls within [start, end] (ISO 8601 strings).

        Args:
            start: ISO 8601 string for the lower bound (inclusive).
            end: ISO 8601 string for the upper bound (inclusive).
            group_id: Project graph to query.

        Returns:
            List of dicts with keys: uuid, fact, name, valid_at, invalid_at.
        """
        graph = self._graph_for(group_id)
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

    async def get_valid_edges_for_node(self, node_uuid: str, *, group_id: str) -> list[dict]:
        """Return all currently-valid RELATES_TO edges for an Entity node.

        Matches the node as either source or target (undirected) and filters
        edges where invalid_at IS NULL (i.e. not yet invalidated).

        Args:
            node_uuid: UUID of the Entity node.
            group_id: Project graph to query.

        Returns:
            List of dicts with keys: uuid, fact, name.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid})-[e:RELATES_TO]-() '
            'WHERE e.invalid_at IS NULL '
            'RETURN DISTINCT e.uuid, e.fact, e.name'
        )
        result = await graph.query(cypher, {'uuid': node_uuid})
        return [
            {
                'uuid': row[0],
                'fact': row[1] or '',
                'name': row[2] or '',
            }
            for row in (result.result_set or [])
        ]

    async def bulk_remove_edges(self, uuids: list[str], *, group_id: str) -> int:
        """Delete RELATES_TO edges by UUID list. Returns count of actually matched edges.

        Uses a pre-count MATCH query before deletion to return the true number of
        edges that exist (and will be deleted), rather than the input list length.
        This is critical for irreversible operations where accuracy matters.

        Args:
            uuids: List of edge UUIDs to delete.
            group_id: Project graph to query.

        Returns:
            Number of edges that matched (and were deleted). 0 for empty list.
        """
        if not uuids:
            return 0
        logger.info('Deleting %d edge(s)', len(uuids))
        logger.debug('Edge UUIDs to delete: %s', uuids)
        graph = self._graph_for(group_id)
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

    async def redirect_node_edges(
        self, deprecated_uuid: str, surviving_uuid: str, *, group_id: str
    ) -> dict:
        """Redirect all RELATES_TO edges from deprecated node to surviving node.

        Three Cypher phases:
        (1) Count and delete inter-node edges between the two nodes (they become
            meaningless self-loops after merge).
        (2) Count then redirect outgoing edges: deprecated→target becomes
            surviving→target (all properties copied individually to preserve
            vecf32 embedding type).
        (3) Count then redirect incoming edges: source→deprecated becomes
            source→surviving.

        Args:
            deprecated_uuid: UUID of the entity node to be deleted.
            surviving_uuid: UUID of the entity node that will absorb the edges.

        Returns:
            Dict with keys: outgoing_redirected, incoming_redirected,
            inter_node_deleted.
        """
        graph = self._graph_for(group_id)

        # Phase 1: Delete inter-node edges (edges between the two merging nodes)
        count_inter = await graph.query(
            'MATCH (dep:Entity {uuid: $dep_uuid})-[e:RELATES_TO]-(sur:Entity {uuid: $sur_uuid}) '
            'RETURN count(e) AS cnt',
            {'dep_uuid': deprecated_uuid, 'sur_uuid': surviving_uuid},
        )
        inter_node_deleted = (
            int(count_inter.result_set[0][0]) if count_inter.result_set else 0
        )
        await graph.query(
            'MATCH (dep:Entity {uuid: $dep_uuid})-[e:RELATES_TO]-(sur:Entity {uuid: $sur_uuid}) '
            'DELETE e',
            {'dep_uuid': deprecated_uuid, 'sur_uuid': surviving_uuid},
        )

        # Phase 2: Redirect outgoing edges (deprecated → target)
        count_out = await graph.query(
            'MATCH (dep:Entity {uuid: $dep_uuid})-[e:RELATES_TO]->() '
            'RETURN count(e) AS cnt',
            {'dep_uuid': deprecated_uuid},
        )
        outgoing_redirected = (
            int(count_out.result_set[0][0]) if count_out.result_set else 0
        )
        await graph.query(
            'MATCH (dep:Entity {uuid: $dep_uuid})-[old:RELATES_TO]->(target) '
            'WITH old, target '
            'MATCH (sur:Entity {uuid: $sur_uuid}) '
            'CREATE (sur)-[new:RELATES_TO]->(target) '
            'SET new.uuid = old.uuid, '
            '    new.name = old.name, '
            '    new.fact = old.fact, '
            '    new.fact_embedding = old.fact_embedding, '
            '    new.valid_at = old.valid_at, '
            '    new.invalid_at = old.invalid_at, '
            '    new.created_at = old.created_at, '
            '    new.group_id = old.group_id, '
            '    new.episodes = old.episodes, '
            '    new.source_node_uuid = $sur_uuid '
            'DELETE old',
            {'dep_uuid': deprecated_uuid, 'sur_uuid': surviving_uuid},
        )

        # Phase 3: Redirect incoming edges (source → deprecated)
        count_in = await graph.query(
            'MATCH ()-[e:RELATES_TO]->(dep:Entity {uuid: $dep_uuid}) '
            'RETURN count(e) AS cnt',
            {'dep_uuid': deprecated_uuid},
        )
        incoming_redirected = (
            int(count_in.result_set[0][0]) if count_in.result_set else 0
        )
        await graph.query(
            'MATCH (source)-[old:RELATES_TO]->(dep:Entity {uuid: $dep_uuid}) '
            'WITH old, source '
            'MATCH (sur:Entity {uuid: $sur_uuid}) '
            'CREATE (source)-[new:RELATES_TO]->(sur) '
            'SET new.uuid = old.uuid, '
            '    new.name = old.name, '
            '    new.fact = old.fact, '
            '    new.fact_embedding = old.fact_embedding, '
            '    new.valid_at = old.valid_at, '
            '    new.invalid_at = old.invalid_at, '
            '    new.created_at = old.created_at, '
            '    new.group_id = old.group_id, '
            '    new.episodes = old.episodes, '
            '    new.target_node_uuid = $sur_uuid '
            'DELETE old',
            {'dep_uuid': deprecated_uuid, 'sur_uuid': surviving_uuid},
        )

        logger.info(
            'redirect_node_edges: dep=%s sur=%s inter_deleted=%d out=%d in=%d',
            deprecated_uuid, surviving_uuid, inter_node_deleted,
            outgoing_redirected, incoming_redirected,
        )
        return {
            'outgoing_redirected': outgoing_redirected,
            'incoming_redirected': incoming_redirected,
            'inter_node_deleted': inter_node_deleted,
        }

    async def merge_entities(
        self, deprecated_uuid: str, surviving_uuid: str, *, group_id: str
    ) -> dict:
        """Merge two entity nodes by redirecting edges and deleting the deprecated node.

        Orchestrates the full merge workflow:
        1. Validate both nodes exist via get_node_text (raises NodeNotFoundError if
           either is missing).
        2. Redirect all RELATES_TO edges from deprecated to surviving via
           redirect_node_edges.
        3. Delete the deprecated node via delete_entity_node.
        4. Rebuild the surviving node's summary via refresh_entity_summary.

        Args:
            deprecated_uuid: UUID of the entity node to be deleted.
            surviving_uuid: UUID of the entity node that absorbs the edges.

        Returns:
            Audit dict with keys: surviving_uuid, surviving_name, deprecated_uuid,
            deprecated_name, edges_redirected (sub-dict with redirect counts),
            surviving_summary (dict with old/new summary and edge_count).

        Raises:
            NodeNotFoundError: if either UUID does not exist.
            RuntimeError: if the backend is not initialized.
        """
        # Validate both nodes exist and capture their names
        dep_name, _ = await self.get_node_text(deprecated_uuid, group_id=group_id)
        sur_name, _ = await self.get_node_text(surviving_uuid, group_id=group_id)

        # Redirect edges
        edges_redirected = await self.redirect_node_edges(
            deprecated_uuid, surviving_uuid, group_id=group_id,
        )

        # Delete the deprecated node
        await self.delete_entity_node(deprecated_uuid, group_id=group_id)

        # Rebuild the surviving node's summary
        refresh_result = await self.refresh_entity_summary(surviving_uuid, group_id=group_id)

        logger.info(
            'merge_entities: dep=%s (%r) sur=%s (%r) redirected=%s',
            deprecated_uuid, dep_name, surviving_uuid, sur_name, edges_redirected,
        )
        return {
            'surviving_uuid': surviving_uuid,
            'surviving_name': sur_name,
            'deprecated_uuid': deprecated_uuid,
            'deprecated_name': dep_name,
            'edges_redirected': edges_redirected,
            'surviving_summary': {
                'before': refresh_result.get('old_summary', ''),
                'after': refresh_result.get('new_summary', ''),
                'edge_count': refresh_result.get('edge_count', 0),
            },
        }

    async def delete_entity_node(self, uuid: str, *, group_id: str) -> None:
        """Delete an Entity node and all remaining relationships.

        Validates that the node exists first, then issues DETACH DELETE.

        Args:
            uuid: UUID of the Entity node to delete.
            group_id: Project graph to query.

        Raises:
            NodeNotFoundError: if no node with that UUID exists.
            RuntimeError: if the backend is not initialized.
        """
        graph = self._graph_for(group_id)
        # Pre-check: verify node exists before deleting
        check_result = await graph.query(
            'MATCH (n:Entity {uuid: $uuid}) RETURN n.name, n.summary',
            {'uuid': uuid},
        )
        if not check_result.result_set:
            raise NodeNotFoundError(f'Entity node not found: {uuid}')
        await graph.query(
            'MATCH (n:Entity {uuid: $uuid}) DETACH DELETE n',
            {'uuid': uuid},
        )
        logger.info('delete_entity_node: deleted node=%s', uuid)

    async def get_node_text(self, uuid: str, *, group_id: str) -> tuple[str, str]:
        """Return (name, summary) for the Entity node with the given UUID.

        Raises:
            NodeNotFoundError: if no node with that UUID exists.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid}) '
            'RETURN n.name, n.summary'
        )
        result = await graph.query(cypher, {'uuid': uuid})
        if not result.result_set:
            raise NodeNotFoundError(f'Entity node not found: {uuid}')
        row = result.result_set[0]
        return (row[0], row[1] or '')

    async def refresh_entity_summary(self, node_uuid: str, *, group_id: str) -> dict:
        """Regenerate an Entity node's summary from its currently-valid edges.

        Fetches the node's current name and summary, queries all valid
        (non-invalidated) RELATES_TO edges, deduplicates their facts
        (preserving order), joins them with newlines, and writes the result
        back to the node's summary property.

        Summary regeneration uses simple fact concatenation (deduped), consistent
        with Graphiti's own _extract_entity_summaries_batch pattern — no LLM call.

        Args:
            node_uuid: UUID of the Entity node to refresh.

        Returns:
            Dict with keys: uuid, name, old_summary, new_summary, edge_count.
        """
        name, old_summary = await self.get_node_text(node_uuid, group_id=group_id)
        edges = await self.get_valid_edges_for_node(node_uuid, group_id=group_id)
        # Deduplicate facts while preserving insertion order
        facts = list(dict.fromkeys(e['fact'] for e in edges if e.get('fact')))
        new_summary = '\n'.join(facts)
        await self.update_node_summary(node_uuid, new_summary, group_id=group_id)
        logger.info(
            'refresh_entity_summary: node=%s name=%r edges=%d old_len=%d new_len=%d',
            node_uuid, name, len(edges), len(old_summary), len(new_summary),
        )
        return {
            'uuid': node_uuid,
            'name': name,
            'old_summary': old_summary,
            'new_summary': new_summary,
            'edge_count': len(edges),
        }

    async def update_node_summary(self, uuid: str, summary: str, *, group_id: str) -> None:
        """Update the summary text property on an Entity node.

        Args:
            uuid: UUID of the Entity node to update.
            summary: New summary text (may be empty string to clear).
            group_id: Project graph to query.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid}) '
            'SET n.summary = $summary'
        )
        await graph.query(cypher, {'uuid': uuid, 'summary': summary})

    async def get_edge_text(self, uuid: str, *, group_id: str) -> tuple[str, str]:
        """Return (name, fact) for the RELATES_TO edge with the given UUID.

        Raises:
            EdgeNotFoundError: if no edge with that UUID exists.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() '
            'RETURN e.name, e.fact'
        )
        result = await graph.query(cypher, {'uuid': uuid})
        if not result.result_set:
            raise EdgeNotFoundError(f'RELATES_TO edge not found: {uuid}')
        row = result.result_set[0]
        return (row[0] or '', row[1] or '')

    async def update_node_embedding(self, uuid: str, embedding: list[float], *, group_id: str) -> None:
        """Update the name_embedding vector for an Entity node using vecf32()."""
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid}) '
            'SET n.name_embedding = vecf32($embedding)'
        )
        await graph.query(cypher, {'uuid': uuid, 'embedding': embedding})

    async def update_edge_embedding(self, uuid: str, embedding: list[float], *, group_id: str) -> None:
        """Update the fact_embedding vector for a RELATES_TO edge using vecf32()."""
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() '
            'SET e.fact_embedding = vecf32($embedding)'
        )
        await graph.query(cypher, {'uuid': uuid, 'embedding': embedding})

    async def list_indices(self, *, group_id: str) -> list[dict]:
        """Return parsed index records from the graph.

        Each record is a dict with keys: label, field, type, entity_type.
        """
        graph = self._graph_for(group_id)
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

    async def drop_index(self, label: str, field: str, *, group_id: str) -> None:
        """Drop an index on the given label and field (FalkorDB syntax)."""
        graph = self._graph_for(group_id)
        cypher = f'DROP INDEX ON :{label}({field})'
        await graph.query(cypher)

    async def drop_vector_indices(self, *, group_id: str) -> list[dict]:
        """Drop all VECTOR-type indices in the graph.

        Calls list_indices() to find indices with type == 'VECTOR', then calls
        drop_index() for each.  Returns a list of {'label': ..., 'field': ...}
        dicts for each dropped index.
        """
        indices = await self.list_indices(group_id=group_id)
        dropped: list[dict] = []
        for entry in indices:
            if entry.get('type') == 'VECTOR':
                await self.drop_index(entry['label'], entry['field'], group_id=group_id)
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

    async def close(self) -> None:
        """Shut down the driver."""
        if self._driver is not None:
            with contextlib.suppress(Exception):
                await self._driver.close()
        self.client = None
        self._driver = None
