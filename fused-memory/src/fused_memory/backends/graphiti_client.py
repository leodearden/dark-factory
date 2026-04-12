"""Thin async wrapper around the Graphiti client."""

import asyncio
import contextlib
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, NamedTuple, TypedDict, cast
from urllib.parse import urlparse

from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.embedder.openai import OpenAIEmbedderConfig
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.nodes import EpisodeType, EpisodicNode

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.utils.async_utils import propagate_cancellations

logger = logging.getLogger(__name__)


class EdgeDict(TypedDict):
    """Normalised edge dict returned by GraphitiBackend._edge_dict.

    Consumed by get_valid_edges_for_node, get_all_valid_edges,
    _canonical_facts, and _rebuild_entity_from_edges.
    """

    uuid: str
    fact: str
    name: str


class StaleSummaryResult(NamedTuple):
    """Structured return type for _detect_stale_summaries_with_edges.

    Use named attribute access — the canonical idiom after Task 438/465:

    - ``result.stale`` — list of stale entity dicts (each has uuid, name, summary, etc.)
    - ``result.all_edges`` — dict[uuid, list[EdgeDict]] of valid edges for every scanned entity
    - ``result.total_count`` — total number of entity nodes scanned

    Because StaleSummaryResult is a NamedTuple (a tuple subclass), positional
    unpacking still works at runtime, but named access is the preferred idiom
    across the codebase.
    """

    stale: list[dict]
    all_edges: dict[str, list[EdgeDict]]
    total_count: int


class NodeNotFoundError(Exception):
    """Raised when a node UUID is not found in FalkorDB."""


class AmbiguousEntityError(Exception):
    """Raised when multiple entity nodes share the same name.

    The error message includes all matching UUIDs so the caller can
    disambiguate and call refresh_entity_summary with a specific UUID.
    """


class _MultiTenantFalkorDriver(FalkorDriver):
    """FalkorDriver that suppresses auto-indexing.

    Upstream ``__init__`` schedules ``build_indices_and_constraints()``
    against ``_database`` as a fire-and-forget task.  In multi-tenant
    mode indices are built explicitly via ``_ensure_indices()`` — the
    fire-and-forget path is suppressed here to prevent redundant
    CREATE INDEX commands from saturating FalkorDB's single-threaded
    execution.

    ``clone()`` is overridden to return another ``_MultiTenantFalkorDriver``
    so cloned per-graph drivers also suppress auto-indexing.
    """

    async def build_indices_and_constraints(self, delete_existing=False):
        pass

    def clone(self, database: str) -> 'GraphDriver':
        if database == self._database:
            return self
        cloned = _MultiTenantFalkorDriver(falkor_db=self.client, database=database)
        return cloned


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
        self._cloned_drivers: dict[str, GraphDriver] = {}

    # --- Per-request driver routing ---

    def _driver_for(self, group_id: str) -> GraphDriver:
        """Return a cached driver clone targeting the FalkorDB graph for *group_id*.

        Caches cloned drivers to avoid creating new connections per request.
        """
        cached = self._cloned_drivers.get(group_id)
        if cached is not None:
            return cached
        driver = self._require_driver()
        cloned = driver.clone(database=group_id)
        self._cloned_drivers[group_id] = cloned
        return cloned

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

        self._driver = _MultiTenantFalkorDriver(
            host=host,
            port=port,
            password=falkor_cfg.password,
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

        Uses ro_query since no writes are performed.

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
        result = await graph.ro_query(cypher, {'start': start, 'end': end})
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

    async def get_valid_edges_for_node(self, node_uuid: str, *, group_id: str) -> list[EdgeDict]:
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
        result = await graph.ro_query(cypher, {'uuid': node_uuid})
        return [
            self._edge_dict(row[0], row[1], row[2])
            for row in (result.result_set or [])
        ]

    async def get_all_valid_edges(self, *, group_id: str) -> dict[str, list[EdgeDict]]:
        """Return all currently-valid RELATES_TO edges grouped by entity UUID.

        Bulk variant of get_valid_edges_for_node that issues a single Cypher query
        instead of O(N) per-entity round-trips.  The undirected MATCH pattern causes
        each directed edge to appear under both its source and target entity: for a
        directed A→B edge, traversal matches it from A's side (row: A.uuid, e.uuid)
        and from B's side (row: B.uuid, e.uuid) — two genuinely distinct rows because
        n.uuid differs.  RETURN DISTINCT guards only against self-loop duplicates
        (A→A edges, where both traversal directions yield identical rows).

        Uses ro_query since no writes are performed.

        Args:
            group_id: Project graph to query.

        Returns:
            Dict mapping entity UUID → list of edge dicts with keys: uuid, fact, name.
            fact and name default to empty string when the property is NULL.
            Each directed edge appears under both its source and target entity UUID
            (double-attribution from the undirected MATCH pattern).

        Note:
            Using a directed pattern (n:Entity)-[e:RELATES_TO]->() would give
            single-appearance semantics per edge if ever needed.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity)-[e:RELATES_TO]-() '
            'WHERE e.invalid_at IS NULL '
            'RETURN DISTINCT n.uuid, e.uuid, e.fact, e.name'
        )
        result = await graph.ro_query(cypher)
        grouped: dict[str, list[EdgeDict]] = {}
        for row in (result.result_set or []):
            entity_uuid = row[0]
            grouped.setdefault(entity_uuid, []).append(self._edge_dict(row[1], row[2], row[3]))
        return grouped

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
        Pre-check uses ro_query since it performs no writes; the DETACH DELETE
        itself uses graph.query.

        Args:
            uuid: UUID of the Entity node to delete.
            group_id: Project graph to query.

        Raises:
            NodeNotFoundError: if no node with that UUID exists.
            RuntimeError: if the backend is not initialized.
        """
        graph = self._graph_for(group_id)
        # Pre-check: verify node exists before deleting (read-only)
        check_result = await graph.ro_query(
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

        Uses ro_query since no writes are performed.

        Raises:
            NodeNotFoundError: if no node with that UUID exists.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH (n:Entity {uuid: $uuid}) '
            'RETURN n.name, n.summary'
        )
        result = await graph.ro_query(cypher, {'uuid': uuid})
        if not result.result_set:
            raise NodeNotFoundError(f'Entity node not found: {uuid}')
        row = result.result_set[0]
        return (row[0], row[1] or '')

    async def resolve_entity_by_name(self, name: str, *, group_id: str) -> str:
        """Resolve an entity name to its UUID via an exact Cypher lookup.

        Uses ro_query since no writes are performed.

        Args:
            name: Exact name of the Entity node to resolve.
            group_id: Project graph to query.

        Returns:
            The UUID string of the matching entity.

        Raises:
            NodeNotFoundError: if no entity with that name exists.
            AmbiguousEntityError: if multiple entities share the same name,
                with all matching UUIDs listed in the error message.
            RuntimeError: if the backend is not initialized.
        """
        graph = self._graph_for(group_id)
        cypher = 'MATCH (n:Entity {name: $name}) RETURN n.uuid, n.name'
        result = await graph.ro_query(cypher, {'name': name})
        rows = result.result_set
        if not rows:
            raise NodeNotFoundError(f'No entity found with name: {name!r}')
        if len(rows) > 1:
            uuids = [row[0] for row in rows]
            raise AmbiguousEntityError(
                f'Multiple entities found with name {name!r}: {uuids}'
            )
        return rows[0][0]

    @staticmethod
    def _edge_dict(uuid: str, fact: str | None, name: str | None) -> EdgeDict:
        """Build a normalised edge dict, coercing NULL fact/name to empty string.

        Args:
            uuid: Edge UUID. Must not be None — a NULL uuid from the graph would
                propagate silently through downstream callers and is treated as
                a hard error.
            fact: Edge fact text, or None when the property is NULL in the graph.
            name: Edge name, or None when the property is NULL in the graph.

        Returns:
            EdgeDict with keys: uuid, fact, name. fact and name default to '' when None.

        Raises:
            ValueError: If uuid is None.
        """
        if uuid is None:
            raise ValueError('edge uuid must not be None')
        return {
            'uuid': uuid,
            'fact': fact if fact is not None else '',
            'name': name if name is not None else '',
        }

    @staticmethod
    def _canonical_facts(edges: Sequence[Mapping[str, Any]]) -> list[str]:
        """Deduplicate edge facts preserving insertion order, skipping missing/falsy values.

        Args:
            edges: List of edge dicts, each optionally containing a 'fact' key.

        Returns:
            List of unique non-empty fact strings in their first-seen order.
        """
        return list(dict.fromkeys(e['fact'] for e in edges if (e.get('fact') or '').strip()))

    async def refresh_entity_summary(
        self,
        node_uuid: str,
        *,
        group_id: str,
        name: str | None = None,
        old_summary: str | None = None,
    ) -> dict[str, Any]:
        """Regenerate an Entity node's summary from its currently-valid edges.

        Fetches the node's current name and summary, queries all valid
        (non-invalidated) RELATES_TO edges, deduplicates their facts
        (preserving order), joins them with newlines, and writes the result
        back to the node's summary property.

        Summary regeneration uses simple fact concatenation (deduped), consistent
        with Graphiti's own _extract_entity_summaries_batch pattern — no LLM call.

        For bulk use see ``_rebuild_entity_from_edges``, which accepts
        caller-supplied edges, name, and old_summary to avoid per-entity
        ``get_node_text`` and ``get_valid_edges_for_node`` round-trips when
        rebuilding many entities at once.  The two methods are an intentional
        fork: ``refresh_entity_summary`` is self-contained for single-entity
        callers; ``_rebuild_entity_from_edges`` is batch-internal and consumes
        pre-fetched data from the ``rebuild_entity_summaries`` pipeline.

        Args:
            node_uuid: UUID of the Entity node to refresh.
            group_id: Project graph to target.
            name: Optional entity name (must be paired with old_summary). When
                both are supplied, get_node_text is skipped — useful when the
                caller already has this data (e.g. _rebuild_entity_from_edges).
            old_summary: Optional current summary text (must be paired with name).

        Returns:
            Dict with keys: uuid, name, old_summary, new_summary, edge_count.

        Raises:
            ValueError: if exactly one of name/old_summary is provided.
        """
        if (name is None) != (old_summary is None):
            raise ValueError('name and old_summary must both be provided or both omitted')
        if name is None:
            name, old_summary = await self.get_node_text(node_uuid, group_id=group_id)
        edges = await self.get_valid_edges_for_node(node_uuid, group_id=group_id)
        facts = self._canonical_facts(edges)
        new_summary = '\n'.join(facts)
        await self.update_node_summary(node_uuid, new_summary, group_id=group_id)
        logger.info(
            'refresh_entity_summary: node=%s name=%r edges=%d old_len=%d new_len=%d',
            node_uuid, name, len(edges), len(old_summary or ''), len(new_summary),
        )
        return {
            'uuid': node_uuid,
            'name': name,
            'old_summary': old_summary,
            'new_summary': new_summary,
            'edge_count': len(edges),
        }

    async def list_entity_nodes(self, *, group_id: str) -> list[dict]:
        """Return all Entity nodes (uuid, name, summary) for a given group_id.

        FalkorDB is multi-tenant — each project lives in its own graph, so no
        group_id filter is needed in the Cypher itself.  Uses ro_query since
        no writes are performed.

        Args:
            group_id: Project graph to query.

        Returns:
            List of dicts with keys: uuid, name, summary (summary defaults to
            empty string when the node property is NULL).
        """
        graph = self._graph_for(group_id)
        cypher = 'MATCH (n:Entity) RETURN n.uuid, n.name, n.summary'
        result = await graph.ro_query(cypher)
        return [
            {
                'uuid': row[0],
                'name': row[1] or '',
                'summary': row[2] or '',
            }
            for row in (result.result_set or [])
        ]

    async def _detect_stale_summaries_with_edges(
        self, *, group_id: str
    ) -> StaleSummaryResult:
        """Internal: detect stale summaries and return a StaleSummaryResult.

        Shared by detect_stale_summaries (public API) and rebuild_entity_summaries
        to avoid a duplicate bulk edge fetch when both are needed.

        Args:
            group_id: Project graph to query.

        Returns:
            StaleSummaryResult with fields:
              .stale       - list of stale entity dicts
              .all_edges   - dict[uuid, list[EdgeDict]] of valid edges for all entities
              .total_count - total number of entity nodes scanned
        """
        entities = await self.list_entity_nodes(group_id=group_id)
        all_edges = await self.get_all_valid_edges(group_id=group_id)
        stale: list[dict] = []
        for entity in entities:
            summary = entity['summary']
            if not summary:
                # Empty summary — not stale by definition
                continue
            edges = all_edges.get(entity['uuid'], [])
            valid_facts = self._canonical_facts(edges)
            canonical = '\n'.join(valid_facts)
            if summary == canonical:
                continue  # Already up-to-date
            # Compute diagnostic counts
            summary_lines = summary.split('\n')
            valid_fact_set = set(valid_facts)
            # duplicate_count: sum of extra occurrences for each unique line that
            # appears more than once in the current summary.
            line_counts = Counter(summary_lines)
            duplicate_count = sum(c - 1 for c in line_counts.values() if c > 1)
            # stale_line_count: lines in summary not in the valid fact set
            stale_line_count = sum(1 for line in summary_lines if line not in valid_fact_set)
            stale.append({
                'uuid': entity['uuid'],
                'name': entity['name'],
                'summary': summary,
                'duplicate_count': duplicate_count,
                'stale_line_count': stale_line_count,
                'valid_fact_count': len(valid_facts),
                'summary_line_count': len(summary_lines),
            })
        return StaleSummaryResult(stale=stale, all_edges=all_edges, total_count=len(entities))

    async def _detect_stale_summaries_dry_run(
        self, *, group_id: str
    ) -> tuple[list[dict], int]:
        """Internal: detect stale summaries using per-entity edge fetching (dry_run variant).

        Memory-cheaper alternative to ``_detect_stale_summaries_with_edges`` for use
        in the ``force=False, dry_run=True`` code path.  Unlike the bulk variant, this
        method never materialises the O(E) all-edges dict because:

        - The dry_run path short-circuits before ``_rebuild_entity_from_edges``, so
          the edges dict is only needed for staleness comparison, not for writing.
        - Fetching edges per-entity (only for non-empty-summary entities) avoids
          holding the full graph's edge data in Python memory when none of it will
          be used to write.

        Trade-off vs ``_detect_stale_summaries_with_edges``:
        - Issues up-to-N targeted ``get_valid_edges_for_node`` queries rather than a
          single bulk ``get_all_valid_edges`` query.
        - Entities with empty summaries are skipped without any edge query (matching
          the existing empty-summary semantics, adding a Pareto improvement for graphs
          with many empty-summary entities).

        Args:
            group_id: Project graph to query.

        Returns:
            Tuple of (stale_list, total_count) where stale_list contains the same
            per-entity dict schema as ``_detect_stale_summaries_with_edges``
            (uuid, name, summary, duplicate_count, stale_line_count, valid_fact_count,
            summary_line_count) and total_count is len(all entities).
        """
        entities = await self.list_entity_nodes(group_id=group_id)
        stale: list[dict] = []
        for entity in entities:
            summary = entity['summary']
            if not summary:
                # Empty summary — not stale by definition; skip without an edge query.
                continue
            edges = await self.get_valid_edges_for_node(entity['uuid'], group_id=group_id)
            valid_facts = self._canonical_facts(edges)
            canonical = '\n'.join(valid_facts)
            if summary == canonical:
                continue  # Already up-to-date
            # Compute diagnostic counts (same schema as _detect_stale_summaries_with_edges)
            summary_lines = summary.split('\n')
            valid_fact_set = set(valid_facts)
            line_counts = Counter(summary_lines)
            duplicate_count = sum(c - 1 for c in line_counts.values() if c > 1)
            stale_line_count = sum(1 for line in summary_lines if line not in valid_fact_set)
            stale.append({
                'uuid': entity['uuid'],
                'name': entity['name'],
                'summary': summary,
                'duplicate_count': duplicate_count,
                'stale_line_count': stale_line_count,
                'valid_fact_count': len(valid_facts),
                'summary_line_count': len(summary_lines),
            })
        return (stale, len(entities))

    async def detect_stale_summaries(self, *, group_id: str) -> list[dict]:
        """Identify Entity nodes whose summary is out of sync with valid edge facts.

        For each entity node, fetches its valid RELATES_TO edges and computes
        the canonical summary (deduped facts joined with newlines).  An entity
        is considered *stale* when:

        - Its current summary is non-empty (empty summaries are skipped), AND
        - Its current summary differs from the canonical summary.

        Diagnostic fields help callers understand *why* an entity is stale:
        - ``duplicate_count``: extra occurrences of duplicated lines (lines that
          appear more than once in the current summary).
        - ``stale_line_count``: lines in the current summary that are not backed
          by any valid edge fact.
        - ``valid_fact_count``: number of unique valid edge facts.
        - ``summary_line_count``: number of lines in the current summary.

        Args:
            group_id: Project graph to query.

        Returns:
            List of dicts (one per stale entity) with keys: uuid, name, summary,
            duplicate_count, stale_line_count, valid_fact_count,
            summary_line_count. The ``summary`` key holds the current
            (pre-rebuild) entity summary text so callers can diff it against
            the canonical fact set without a second DB query.
        """
        result = await self._detect_stale_summaries_with_edges(group_id=group_id)
        return result.stale

    async def _rebuild_entity_from_edges(
        self, uuid: str, name: str, edges: list[EdgeDict], *, group_id: str,
        old_summary: str,
    ) -> dict[str, Any]:
        """Rebuild one Entity node's summary from pre-fetched edges.

        Accepts the edges already fetched by the bulk call, avoiding a
        per-entity get_valid_edges_for_node round-trip.

        For single-entity use (not bulk) see ``refresh_entity_summary``, which
        fetches its own name/old_summary via ``get_node_text`` and its own valid
        edges via ``get_valid_edges_for_node``.  This method exists as the
        bulk-optimised counterpart: it accepts caller-supplied edges and
        old_summary to eliminate per-entity DB round-trips when rebuilding many
        entities at once.

        .. note:: TOCTOU / eventual-consistency risk:
            The ``edges`` argument is pre-fetched by the caller in a single
            bulk query.  By the time this method runs (potentially after a
            concurrency gap), the graph may have changed — new edges added,
            existing edges invalidated.  Summaries written here therefore
            reflect a snapshot, not necessarily the current DB state.  This
            is an accepted trade-off: callers that require stronger consistency
            should re-fetch edges per entity via ``refresh_entity_summary``.

        Args:
            uuid: Entity UUID.
            name: Entity name (for logging / result dict).
            edges: Pre-fetched valid edge dicts (uuid, fact, name).
            group_id: Graph to update.
            old_summary: Current summary text (caller must supply — avoids
                per-entity ``get_node_text`` DB round-trip).

        Returns:
            Dict with keys: uuid, name, old_summary, new_summary, edge_count.
        """
        facts = self._canonical_facts(edges)
        new_summary = '\n'.join(facts)
        await self.update_node_summary(uuid, new_summary, group_id=group_id)
        logger.info(
            '_rebuild_entity_from_edges: node=%s name=%r edges=%d new_len=%d',
            uuid, name, len(edges), len(new_summary),
        )
        return {
            'uuid': uuid,
            'name': name,
            'old_summary': old_summary,
            'new_summary': new_summary,
            'edge_count': len(edges),
        }

    async def rebuild_entity_summaries(
        self,
        *,
        group_id: str,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Batch-rebuild Entity node summaries from their current valid edges.

        When ``force=False`` (default), only entities identified by
        ``detect_stale_summaries`` are rebuilt.  When ``force=True``, all
        entities are rebuilt regardless of staleness.

        When ``dry_run=True``, stale detection is performed but
        ``refresh_entity_summary`` is never called — useful for inspecting
        what *would* be rebuilt without making any changes.

        Args:
            group_id: Project graph to target.
            force: Rebuild every entity even if its summary appears clean.
            dry_run: Detect but do not actually rebuild.

        Returns:
            Dict with keys:
            - ``total_entities``: total Entity nodes in the graph.
            - ``stale_entities``: number of entities identified as stale (or
              total when force=True).
            - ``rebuilt``: number of entities whose summary was written.
            - ``skipped``: number of entities skipped (dry_run or up-to-date).
            - ``errors``: number of per-entity refresh failures.
            - ``details``: list of per-entity result dicts.
        """
        # Declare before if/else for explicit scoping — all branches assign these.
        targets: list[dict] = []
        all_edges: dict[str, list[EdgeDict]] = {}
        total_entities: int = 0

        if force:
            all_entities = await self.list_entity_nodes(group_id=group_id)
            targets = [{'uuid': e['uuid'], 'name': e['name'], 'old_summary': e['summary']} for e in all_entities]
            total_entities = len(all_entities)
            # Only fetch edges when we will actually use them (not in dry_run)
            if not dry_run:
                all_edges = await self.get_all_valid_edges(group_id=group_id)
        else:
            if dry_run:
                # dry_run=True: use the memory-cheaper per-entity probe.
                # The bulk all_edges dict is never needed because the dry_run block
                # below short-circuits before _rebuild_entity_from_edges consumes it.
                stale, total_entities = await self._detect_stale_summaries_dry_run(group_id=group_id)
                # all_edges stays as the empty dict declared above (never materialised)
            else:
                result = await self._detect_stale_summaries_with_edges(group_id=group_id)
                stale = result.stale
                all_edges = result.all_edges
                total_entities = result.total_count
            targets = [{'uuid': s['uuid'], 'name': s['name'], 'old_summary': s['summary']} for s in stale]

        stale_entities = len(targets)
        rebuilt = 0
        skipped = 0
        errors = 0
        details: list[dict] = []

        if dry_run:
            skipped = stale_entities
            for t in targets:
                details.append({'uuid': t['uuid'], 'name': t['name'], 'status': 'skipped_dry_run'})
        else:
            sem = asyncio.Semaphore(20)

            async def _rebuild_one(t: dict) -> dict:
                async with sem:
                    edges = all_edges.get(t['uuid'], [])
                    return await self._rebuild_entity_from_edges(
                        t['uuid'], t['name'], edges, group_id=group_id,
                        old_summary=t['old_summary'],
                    )

            results = await asyncio.gather(
                *(_rebuild_one(t) for t in targets), return_exceptions=True
            )

            # Two-tier check for asyncio.gather(return_exceptions=True) results.
            # Pass 1: delegates to propagate_cancellations (fused_memory.utils.async_utils)
            # — the shared Pass 1 guard used by all gather(return_exceptions=True) callsites.
            # The try/except wrapper preserves the per-batch warning log (which needs
            # group_id and per-batch counters that are local to this callsite).
            # See fused_memory.utils.async_utils.propagate_cancellations for the shared
            # Pass 1 guard contract.
            try:
                propagate_cancellations(results)
            except BaseException as e:
                # Gate the warning on the helper's documented raise contract:
                # propagate_cancellations only raises bare BaseExceptions (not
                # Exception subclasses).  If that contract ever widens (e.g. a
                # validation error on the input sequence), we re-raise silently
                # rather than mislabelling a regular Exception as a
                # "cancellation signal".
                if not isinstance(e, Exception):
                    logger.warning(
                        'rebuild_entity_summaries: cancellation signal received '
                        'group=%s rebuilt_so_far=%d errors_so_far=%d; propagating',
                        group_id, rebuilt, errors,
                    )
                raise

            # Pass 2: per-entity accumulation.  Using isinstance(r, Exception) instead
            # of BaseException ensures that only application-level failures (RuntimeError,
            # etc.) are recorded as error detail entries.  CancelledError is already
            # handled above.
            for t, r in zip(targets, results, strict=True):
                if isinstance(r, Exception):
                    errors += 1
                    logger.error(
                        'rebuild_entity_summaries: failed to rebuild node=%s name=%r: %s',
                        t['uuid'], t['name'], r,
                    )
                    details.append({
                        'uuid': t['uuid'],
                        'name': t['name'],
                        'status': 'error',
                        'error': str(r),
                    })
                else:
                    if not isinstance(r, dict):
                        raise TypeError(
                            f'rebuild_entity_summaries: _rebuild_entity_from_edges returned '
                            f'unexpected type {type(r).__name__!r} for node={t["uuid"]} '
                            f'name={t["name"]!r}'
                        )
                    rebuilt += 1
                    details.append({
                        'uuid': t['uuid'],
                        'name': t['name'],
                        'status': 'rebuilt',
                        'old_summary': r.get('old_summary', ''),
                        'new_summary': r.get('new_summary', ''),
                        'edge_count': r.get('edge_count', 0),
                    })

        logger.info(
            'rebuild_entity_summaries: group=%s total=%d stale=%d rebuilt=%d '
            'skipped=%d errors=%d dry_run=%s force=%s',
            group_id, total_entities, stale_entities, rebuilt, skipped, errors,
            dry_run, force,
        )
        return {
            'total_entities': total_entities,
            'stale_entities': stale_entities,
            'rebuilt': rebuilt,
            'skipped': skipped,
            'errors': errors,
            'details': details,
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

        Uses ro_query since no writes are performed.

        Raises:
            EdgeNotFoundError: if no edge with that UUID exists.
        """
        graph = self._graph_for(group_id)
        cypher = (
            'MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() '
            'RETURN e.name, e.fact'
        )
        result = await graph.ro_query(cypher, {'uuid': uuid})
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

        Uses ro_query since no writes are performed.

        Each record is a dict with keys: label, field, type, entity_type.

        Note on the CALL db.indexes() procedure and the read-only path:
        ``CALL db.indexes()`` is the *only* stored-procedure call sent on the
        read-only path in this file — all other ``ro_query`` callers use plain
        MATCH queries.  Stored procedures are sometimes classified as
        write-capable by graph databases, so this usage was validated
        empirically against FalkorDB module v41800 (4.18.0): the call is
        accepted via ``GRAPH.RO_QUERY`` without error.

        The live verification is pinned in
        ``fused-memory/tests/test_list_indices_integration.py``
        (Task 530 / esc-486-49).  If a future FalkorDB upgrade rejects
        ``CALL`` on the RO path, revert this call to ``graph.query(...)``
        (the write-capable command) and update the integration test to pin
        the new behavior.
        """
        graph = self._graph_for(group_id)
        # CALL db.indexes() is a read-only procedure; FalkorDB accepts it via
        # GRAPH.RO_QUERY (verified via test_list_indices_integration.py).
        result = await graph.ro_query('CALL db.indexes()')
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
        driver = self._require_driver()
        all_graphs = await cast(Any, driver).client.list_graphs()
        return [g for g in all_graphs if g != 'default_db' and not g.endswith('_db')]

    async def node_count(self, graph_name: str) -> int:
        """Count nodes in a specific FalkorDB graph.

        Uses ro_query since no writes are performed.
        """
        driver = self._require_driver()
        graph: Any = driver._get_graph(graph_name)
        result = await graph.ro_query('MATCH (n) RETURN count(n) as count')
        return result.result_set[0][0] if result.result_set else 0

    async def close(self) -> None:
        """Shut down the driver."""
        if self._driver is not None:
            with contextlib.suppress(Exception):
                await self._driver.close()
        self.client = None
        self._driver = None
