"""FastMCP tool definitions for the Fused Memory server."""

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from fused_memory.services.memory_service import MemoryService

logger = logging.getLogger(__name__)

FUSED_MEMORY_INSTRUCTIONS = """\
Fused Memory is a unified memory system that combines Graphiti (temporal knowledge graph)
and Mem0 (vector memory store) behind a single interface.

It organizes memories into six categories:
1. Entities & Relations — facts about things and how they connect (Graphiti)
2. Temporal Facts — state that changes over time (Graphiti)
3. Decisions & Rationale — choices made and why (Graphiti)
4. Preferences & Norms — conventions, style rules (Mem0)
5. Procedural Knowledge — workflows, how-to steps (Mem0)
6. Observations & Summaries — high-level takeaways (Mem0)

Write operations:
- add_episode: Full ingestion pipeline (raw content → extraction → dual-store routing)
- add_memory: Lightweight classified write (skip extraction, direct store)

Read operations:
- search: Unified search across both stores with automatic routing
- get_entity: Direct entity lookup in the knowledge graph
- get_episodes: Retrieve raw episode history

Management:
- delete_memory / delete_episode: Remove specific memories
- get_status: Health check for both backends
"""


def create_mcp_server(memory_service: MemoryService) -> FastMCP:
    """Create and configure the FastMCP server with all tools."""

    mcp = FastMCP('Fused Memory', instructions=FUSED_MEMORY_INSTRUCTIONS)

    # ------------------------------------------------------------------
    # Write tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def add_episode(
        content: str,
        project_id: str,
        source: str = 'text',
        agent_id: str | None = None,
        session_id: str | None = None,
        source_description: str = '',
    ) -> dict[str, Any]:
        """Add an episode to memory. Full ingestion pipeline: raw content is processed
        through Graphiti's extraction pipeline, then classified facts are dual-written
        to Mem0 as appropriate. Returns immediately; processing happens in background.

        Args:
            content: Raw text, conversation, or JSON to ingest
            project_id: Project scope (required)
            source: Source type — "text", "json", or "message"
            agent_id: Which agent is writing (optional)
            session_id: Session context (optional)
            source_description: E.g. "pair programming session"
        """
        try:
            result = await memory_service.add_episode(
                content=content,
                source=source,
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                source_description=source_description,
            )
            return result.model_dump()
        except Exception as e:
            logger.error(f'add_episode error: {e}')
            return {'error': str(e)}

    @mcp.tool()
    async def add_memory(
        content: str,
        project_id: str,
        category: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        dual_write: bool = False,
    ) -> dict[str, Any]:
        """Add a classified memory directly. Skips the extraction pipeline.
        Use when the agent has already identified a specific, discrete memory.

        Args:
            content: The memory itself (a fact, preference, procedure, etc.)
            project_id: Project scope (required)
            category: One of: entities_and_relations, temporal_facts, decisions_and_rationale,
                      preferences_and_norms, procedural_knowledge, observations_and_summaries.
                      If omitted, the system classifies automatically.
            agent_id: Which agent is writing (optional)
            session_id: Session context (optional)
            metadata: Arbitrary key-value pairs (optional)
            dual_write: Force write to both stores (default: false)
        """
        try:
            result = await memory_service.add_memory(
                content=content,
                category=category,
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                metadata=metadata,
                dual_write=dual_write,
            )
            return result.model_dump()
        except Exception as e:
            logger.error(f'add_memory error: {e}')
            return {'error': str(e)}

    # ------------------------------------------------------------------
    # Read tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def search(
        query: str,
        project_id: str,
        categories: list[str] | None = None,
        stores: list[str] | None = None,
        limit: int = 10,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Search across both memory stores with automatic routing.

        The system classifies the query to determine which store(s) to search:
        - Entity/relational/temporal queries → Graphiti primary
        - Preference/procedural queries → Mem0 primary
        - Broad queries → both stores

        Args:
            query: Natural language query
            project_id: Project scope (required)
            categories: Filter to specific taxonomy categories (optional)
            stores: Force "graphiti" and/or "mem0" (optional, default: auto)
            limit: Max results (default: 10)
            agent_id: Filter by authoring agent (optional)
            session_id: Filter by session (optional)
        """
        try:
            results = await memory_service.search(
                query=query,
                project_id=project_id,
                categories=categories,
                stores=stores,
                limit=limit,
                agent_id=agent_id,
                session_id=session_id,
            )
            return {'results': [r.model_dump() for r in results]}
        except Exception as e:
            logger.error(f'search error: {e}')
            return {'error': str(e)}

    @mcp.tool()
    async def get_entity(
        name: str,
        project_id: str,
    ) -> dict[str, Any]:
        """Look up an entity in the knowledge graph. Returns the entity node(s),
        their edges, and connected entities.

        Args:
            name: Entity name (fuzzy matched)
            project_id: Project scope (required)
        """
        try:
            return await memory_service.get_entity(name=name, project_id=project_id)
        except Exception as e:
            logger.error(f'get_entity error: {e}')
            return {'error': str(e)}

    @mcp.tool()
    async def get_episodes(
        project_id: str,
        last_n: int = 10,
    ) -> dict[str, Any]:
        """Retrieve raw episodes from the knowledge graph. Useful for reviewing
        interaction history or tracing provenance.

        Args:
            project_id: Project scope (required)
            last_n: Number of most recent episodes to return (default: 10)
        """
        try:
            episodes = await memory_service.get_episodes(
                project_id=project_id, last_n=last_n
            )
            return {'episodes': episodes}
        except Exception as e:
            logger.error(f'get_episodes error: {e}')
            return {'error': str(e)}

    # ------------------------------------------------------------------
    # Delete tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def delete_memory(
        memory_id: str,
        store: str,
        project_id: str,
    ) -> dict[str, Any]:
        """Delete a specific memory from a store.

        Args:
            memory_id: The memory ID (from search results)
            store: "graphiti" or "mem0" (from search results)
            project_id: Project scope (required)
        """
        try:
            return await memory_service.delete_memory(
                memory_id=memory_id, store=store, project_id=project_id
            )
        except Exception as e:
            logger.error(f'delete_memory error: {e}')
            return {'error': str(e)}

    @mcp.tool()
    async def delete_episode(
        episode_id: str,
        project_id: str,
        cascade: bool = True,
    ) -> dict[str, Any]:
        """Delete a Graphiti episode.

        Args:
            episode_id: Graphiti episode UUID
            project_id: Project scope (required)
            cascade: Also delete exclusive facts (default: true)
        """
        try:
            return await memory_service.delete_episode(
                episode_id=episode_id, project_id=project_id, cascade=cascade
            )
        except Exception as e:
            logger.error(f'delete_episode error: {e}')
            return {'error': str(e)}

    # ------------------------------------------------------------------
    # Management tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def get_status(
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Health check and statistics for both backends.

        Args:
            project_id: Get stats for a specific project (optional)
        """
        try:
            return await memory_service.get_status(project_id=project_id)
        except Exception as e:
            logger.error(f'get_status error: {e}')
            return {'error': str(e)}

    return mcp
