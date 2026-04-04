"""FastMCP tool definitions for the Fused Memory server."""

from __future__ import annotations

import json
import logging
import uuid as uuid_mod
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context, FastMCP

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.services.memory_service import MemoryService
from fused_memory.utils.validation import validate_project_id, validate_project_root

if TYPE_CHECKING:
    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.reconciliation.harness import ReconciliationHarness
    from fused_memory.services.write_journal import WriteJournal

logger = logging.getLogger(__name__)

FUSED_MEMORY_INSTRUCTIONS = """\
Fused Memory is a unified memory system that combines Graphiti (temporal knowledge graph)
and Mem0 (vector memory store) behind a single interface. It also provides proxied access
to Taskmaster AI for task management, with automatic reconciliation between memory and tasks.

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

Task operations (when Taskmaster is connected):
- get_tasks / get_task: Read task tree
- set_task_status: Update status (triggers reconciliation for done/blocked/cancelled)
- add_task / update_task / add_subtask / remove_task: Task CRUD
- add_dependency / remove_dependency: Dependency management
- expand_task / parse_prd: Bulk task generation

Management:
- delete_memory: Remove a specific memory (edges for Graphiti, vector entries for Mem0)
- delete_episode: Remove a Graphiti episode (with optional cascade)
- refresh_entity_summary: Rebuild an entity node's summary from its valid edges (accepts entity_uuid or entity_name)
- merge_entities: Consolidate two duplicate entity nodes (redirects edges, deletes deprecated)
- get_status: Health check for all backends

Reconciliation:
- Task status transitions (done/blocked/cancelled/deferred) trigger targeted reconciliation
  automatically — memory_hints may be attached, completion knowledge written, dependent tasks flagged.
- Bulk operations (expand_task, parse_prd) trigger cross-referencing between new tasks and
  existing knowledge.
- A background pipeline runs periodically for full-cycle reconciliation (consolidation,
  cross-store integrity, task-knowledge sync).

Conventions:
- Always include project_id on every call (scopes data isolation).
- Include agent_id for attribution (e.g. "claude-interactive", "claude-task-7").
- Prefer add_memory over add_episode for discrete, pre-distilled facts (lower cost: 0-3 vs 5-15 LLM calls).
- Tasks may carry memory_hints in metadata — structured pointers (search queries + entity names)
  that help future agents prefetch relevant context. Execute hint queries via search, look up
  hint entities via get_entity.
"""


def _extract_causation(
    metadata: dict | None, agent_id: str | None
) -> tuple[str, str, dict | None]:
    """Extract or generate causation_id, determine source, clean metadata.

    Returns (causation_id, source, cleaned_metadata).
    """
    causation_id: str | None = None
    cleaned = dict(metadata) if metadata else None

    if cleaned and '_causation_id' in cleaned:
        causation_id = cleaned.pop('_causation_id')

    if causation_id is None:
        causation_id = str(uuid_mod.uuid4())

    source = 'mcp_tool'
    if agent_id and agent_id.startswith('recon-stage-'):
        source = 'full_recon'

    return causation_id, source, cleaned


def _resolve_identity(
    agent_id: str | None,
    session_id: str | None,
    ctx: Context[Any, Any, Any] | None,
) -> tuple[str | None, str | None]:
    """Derive agent_id/session_id from MCP Context when not explicitly set.

    - agent_id ← ctx.session.client_params.clientInfo.name
    - session_id ← mcp-session-id HTTP request header

    Explicit caller values always take precedence. Gracefully returns
    originals on stdio transport, stateless HTTP, or missing context.
    """
    if ctx is None:
        return agent_id, session_id

    if agent_id is None:
        client_params = getattr(ctx.session, 'client_params', None)
        client_info = getattr(client_params, 'clientInfo', None)
        name = getattr(client_info, 'name', None)
        if isinstance(name, str):
            agent_id = name

    if session_id is None:
        req_ctx = getattr(ctx, 'request_context', None)
        request = getattr(req_ctx, 'request', None)
        headers = getattr(request, 'headers', None)
        if headers is not None:
            val = headers.get('mcp-session-id')
            if isinstance(val, str):
                session_id = val

    return agent_id, session_id


def create_mcp_server(
    memory_service: MemoryService,
    task_interceptor: TaskInterceptor | None = None,
    write_journal: WriteJournal | None = None,
    *,
    reconciliation_harness: ReconciliationHarness | None = None,
) -> FastMCP:
    """Create and configure the FastMCP server with all tools."""

    mcp = FastMCP('Fused Memory', instructions=FUSED_MEMORY_INSTRUCTIONS)
    _taskmaster_configured = task_interceptor is not None

    async def _log_read(
        operation: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        params: dict | None = None,
        result_summary: dict | str | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Fire-and-forget read logging — mirrors write pattern."""
        if write_journal is None:
            return
        try:
            await write_journal.log_write_op(
                write_op_id=str(uuid_mod.uuid4()),
                source='mcp_tool',
                operation=operation,
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                kind='read',
                params=params,
                result_summary=result_summary,
                success=success,
                error=error,
            )
        except Exception as e:
            logger.warning(f'Failed to log read op: {e}')

    # ------------------------------------------------------------------
    # Health endpoint (used by orchestrator's McpLifecycle._wait_for_health)
    # ------------------------------------------------------------------

    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        import asyncio

        graphiti_ok = False
        mem0_ok = False
        try:
            async with asyncio.timeout(5):
                await memory_service.graphiti.list_graphs()
                graphiti_ok = True
        except Exception:
            pass
        try:
            async with asyncio.timeout(5):
                await memory_service.mem0.list_projects()
                mem0_ok = True
        except Exception:
            pass

        ok = graphiti_ok and mem0_ok
        body = {"status": "ok" if ok else "degraded",
                "graphiti": graphiti_ok, "mem0": mem0_ok}
        return JSONResponse(body, status_code=200 if ok else 503)

    # ------------------------------------------------------------------
    # Write tools
    # ------------------------------------------------------------------

    _VALID_TEMPORAL_CONTEXTS = frozenset({'retrospective', 'planning', 'current'})
    _VALID_TASK_STATUSES = frozenset({
        'pending', 'done', 'in-progress', 'review', 'deferred', 'cancelled', 'blocked',
    })
    _VALID_STORES = frozenset(v.value for v in SourceStore)
    _VALID_CATEGORIES = frozenset(v.value for v in MemoryCategory)

    @mcp.tool()
    async def add_episode(
        content: str,
        project_id: str,
        source: str = 'text',
        agent_id: str | None = None,
        session_id: str | None = None,
        source_description: str = '',
        metadata: dict | None = None,
        temporal_context: str | None = None,
        reference_time: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Add an episode to memory. Full ingestion pipeline: raw content is processed
        through Graphiti's extraction pipeline, then classified facts are dual-written
        to Mem0 as appropriate. Returns immediately; processing happens in background.

        Args:
            content: Raw text, conversation, or JSON to ingest
            project_id: Project scope (required)
            source: Source type — "text", "json", or "message"
            agent_id: Which agent is writing (optional, auto-derived from MCP context)
            session_id: Session context (optional, auto-derived from MCP context)
            source_description: E.g. "pair programming session"
            metadata: Optional key-value pairs (may contain _causation_id for recon)
            temporal_context: Optional temporal framing — one of "retrospective",
                "planning", or "current". When set, the value is prepended to
                source_description as '[temporal:X] ' so downstream readers can
                infer the time-frame of the episode without parsing content.
            reference_time: Optional ISO 8601 datetime string (e.g.
                "2026-03-22T00:00:00+00:00") that sets the historical valid_at
                anchor for Graphiti edge extraction. Use when ingesting retrospective
                episodes to prevent temporal contamination (valid_at = ingestion
                time instead of the date the described state was current).
                Complements temporal_context='retrospective': temporal_context marks
                the *kind* of episode; reference_time sets the *timestamp*.
        """
        agent_id, session_id = _resolve_identity(agent_id, session_id, ctx)
        if err := validate_project_id(project_id):
            return err
        if temporal_context is not None and temporal_context not in _VALID_TEMPORAL_CONTEXTS:
            return {
                'error': (
                    f'Invalid temporal_context {temporal_context!r}. '
                    f'Must be one of {sorted(_VALID_TEMPORAL_CONTEXTS)} or None.'
                ),
                'error_type': 'ValidationError',
            }
        parsed_reference_time = None
        if reference_time is not None:
            try:
                parsed_reference_time = datetime.fromisoformat(reference_time)
            except ValueError:
                return {
                    'error': (
                        f'Invalid reference_time {reference_time!r}. '
                        'Must be an ISO 8601 datetime string, e.g. "2026-03-22T00:00:00+00:00".'
                    ),
                    'error_type': 'ValidationError',
                }
        try:
            causation_id, op_source, _ = _extract_causation(metadata, agent_id)
            result = await memory_service.add_episode(
                content=content,
                source=source,
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                source_description=source_description,
                causation_id=causation_id,
                temporal_context=temporal_context,
                reference_time=parsed_reference_time,
                _source=op_source,
            )
            return result.model_dump()
        except Exception as e:
            logger.error(f'add_episode error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def add_memory(
        content: str,
        project_id: str,
        category: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        dual_write: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Add a classified memory directly. Skips the extraction pipeline.
        Use when the agent has already identified a specific, discrete memory.

        Args:
            content: The memory itself (a fact, preference, procedure, etc.)
            project_id: Project scope (required)
            category: One of: entities_and_relations, temporal_facts, decisions_and_rationale,
                      preferences_and_norms, procedural_knowledge, observations_and_summaries.
                      If omitted, the system classifies automatically.
            agent_id: Which agent is writing (optional, auto-derived from MCP context)
            session_id: Session context (optional, auto-derived from MCP context)
            metadata: Arbitrary key-value pairs (optional)
            dual_write: Force write to both stores (default: false)
        """
        agent_id, session_id = _resolve_identity(agent_id, session_id, ctx)
        if err := validate_project_id(project_id):
            return err
        if category is not None and category not in _VALID_CATEGORIES:
            return {
                'error': (
                    f'Invalid category {category!r}. '
                    f'Must be one of {sorted(_VALID_CATEGORIES)} or None.'
                ),
                'error_type': 'ValidationError',
            }
        try:
            causation_id, source, cleaned_meta = _extract_causation(metadata, agent_id)
            result = await memory_service.add_memory(
                content=content,
                category=category,
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                metadata=cleaned_meta,
                dual_write=dual_write,
                causation_id=causation_id,
                _source=source,
            )
            return result.model_dump()
        except Exception as e:
            logger.error(f'add_memory error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

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
        include_planned: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Search across both memory stores with automatic routing.

        The system classifies the query to determine which store(s) to search:
        - Entity/relational/temporal queries → Graphiti primary
        - Preference/procedural queries → Mem0 primary
        - Broad queries → both stores

        Graphiti results represent entity edges (facts) with edge UUIDs as IDs.
        When category filtering is active and targets exactly one Graphiti-primary
        category, that category is inferred on Graphiti results (which otherwise
        lack category metadata).

        By default, results from planning episodes (temporal_context='planning')
        are excluded to prevent aspirational/PRD content from contaminating
        factual search results.  Set include_planned=True to include them — useful
        for reconciliation, auditing, or explicitly querying planned work.

        Args:
            query: Natural language query
            project_id: Project scope (required)
            categories: Filter to specific taxonomy categories (optional)
            stores: Force "graphiti" and/or "mem0" (optional, default: auto)
            limit: Max results (default: 10)
            agent_id: Filter by authoring agent (optional, auto-derived from MCP context)
            session_id: Filter by session (optional, auto-derived from MCP context)
            include_planned: Include planning-episode edges (default: False)
        """
        agent_id, session_id = _resolve_identity(agent_id, session_id, ctx)
        if err := validate_project_id(project_id):
            return err
        if limit <= 0:
            return {
                'error': f'Invalid limit {limit!r}. Must be a positive integer.',
                'error_type': 'ValidationError',
            }
        if limit > 1000:
            limit = 1000
        try:
            results = await memory_service.search(
                query=query,
                project_id=project_id,
                categories=categories,
                stores=stores,
                limit=limit,
                agent_id=agent_id,
                session_id=session_id,
                include_planned=include_planned,
            )
            response = {'results': [r.model_dump() for r in results]}
            await _log_read(
                operation='search',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'query': query[:200], 'limit': limit},
                result_summary={'count': len(results)},
            )
            return response
        except Exception as e:
            logger.error(f'search error: {e}')
            await _log_read(
                operation='search',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'query': query[:200], 'limit': limit},
                success=False,
                error=str(e),
            )
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_entity(
        name: str,
        project_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Look up an entity in the knowledge graph by name (fuzzy matched).

        Returns entity nodes (with names, summaries, labels), their edges as
        relationship facts, and connected entities. Use this for direct entity
        lookup when you know the name; use search() for broader semantic queries.

        Args:
            name: Entity name (fuzzy matched — partial or approximate names work)
            project_id: Project scope (required)
            agent_id: Which agent is reading (optional, auto-derived from MCP context)
            session_id: Session context (optional, auto-derived from MCP context)
        """
        agent_id, session_id = _resolve_identity(agent_id, session_id, ctx)
        if err := validate_project_id(project_id):
            return err
        try:
            result = await memory_service.get_entity(name=name, project_id=project_id)
            await _log_read(
                operation='get_entity',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'name': name},
                result_summary={
                    'nodes': len(result.get('nodes', [])),
                    'edges': len(result.get('edges', [])),
                },
            )
            return result
        except Exception as e:
            logger.error(f'get_entity error: {e}')
            await _log_read(
                operation='get_entity',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'name': name},
                success=False,
                error=str(e),
            )
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_episodes(
        project_id: str,
        last_n: int = 10,
        agent_id: str | None = None,
        session_id: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Retrieve raw episodes from the knowledge graph. Episodes are the original
        ingested content chunks — each represents one add_episode call with its
        timestamp, source type, and content. Useful for reviewing interaction history,
        tracing provenance of extracted facts, or auditing what was ingested.

        Args:
            project_id: Project scope (required)
            last_n: Number of most recent episodes to return (default: 10)
            agent_id: Which agent is reading (optional, auto-derived from MCP context)
            session_id: Session context (optional, auto-derived from MCP context)
        """
        agent_id, session_id = _resolve_identity(agent_id, session_id, ctx)
        if err := validate_project_id(project_id):
            return err
        if last_n <= 0:
            return {
                'error': f'Invalid last_n {last_n!r}. Must be a positive integer.',
                'error_type': 'ValidationError',
            }
        if last_n > 1000:
            last_n = 1000
        try:
            episodes = await memory_service.get_episodes(
                project_id=project_id, last_n=last_n
            )
            await _log_read(
                operation='get_episodes',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'last_n': last_n},
                result_summary={'count': len(episodes)},
            )
            return {'episodes': episodes}
        except Exception as e:
            logger.error(f'get_episodes error: {e}')
            await _log_read(
                operation='get_episodes',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'last_n': last_n},
                success=False,
                error=str(e),
            )
            return {'error': str(e), 'error_type': type(e).__name__}

    # ------------------------------------------------------------------
    # Delete tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def delete_memory(
        memory_id: str,
        store: str,
        project_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Delete a specific memory from a store. IRREVERSIBLE.

        For Mem0: removes the vector entry directly.
        For Graphiti: removes the edge (fact) found by search. Use delete_episode
        to remove an episode and its exclusively-sourced entities/edges.

        The memory_id and store values come from search results — each result
        includes its id (edge UUID for Graphiti, memory UUID for Mem0) and
        source_store.

        Args:
            memory_id: The memory ID (from search results)
            store: "graphiti" or "mem0" (from search results)
            project_id: Project scope (required)
            agent_id: Which agent is deleting (optional, auto-derived from MCP context)
            session_id: Session context (optional, auto-derived from MCP context)
            metadata: Optional key-value pairs (may contain _causation_id for recon)
        """
        agent_id, session_id = _resolve_identity(agent_id, session_id, ctx)
        if err := validate_project_id(project_id):
            return err
        if store not in _VALID_STORES:
            return {
                'error': (
                    f'Invalid store {store!r}. '
                    f'Must be one of {sorted(_VALID_STORES)}.'
                ),
                'error_type': 'ValidationError',
            }
        try:
            causation_id, source, _ = _extract_causation(metadata, agent_id)
            return await memory_service.delete_memory(
                memory_id=memory_id, store=store, project_id=project_id,
                agent_id=agent_id, session_id=session_id,
                causation_id=causation_id, _source=source,
            )
        except Exception as e:
            logger.error(f'delete_memory error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def delete_episode(
        episode_id: str,
        project_id: str,
        cascade: bool = True,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Delete a Graphiti episode. IRREVERSIBLE.

        When cascade=true (default): also removes entities and edges that were
        exclusively sourced from this episode. Entities/edges shared with other
        episodes are preserved.

        Args:
            episode_id: Graphiti episode UUID
            project_id: Project scope (required)
            cascade: Also delete exclusive entities/edges (default: true)
            agent_id: Which agent is deleting (optional, auto-derived from MCP context)
            session_id: Session context (optional, auto-derived from MCP context)
            metadata: Optional key-value pairs (may contain _causation_id for recon)
        """
        agent_id, session_id = _resolve_identity(agent_id, session_id, ctx)
        if err := validate_project_id(project_id):
            return err
        try:
            causation_id, source, _ = _extract_causation(metadata, agent_id)
            return await memory_service.delete_episode(
                episode_id=episode_id, project_id=project_id, cascade=cascade,
                agent_id=agent_id, session_id=session_id,
                causation_id=causation_id, _source=source,
            )
        except Exception as e:
            logger.error(f'delete_episode error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def refresh_entity_summary(
        project_id: str,
        entity_uuid: str | None = None,
        entity_name: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Regenerate a Graphiti entity node's summary from its valid edges.

        After deleting edges from an entity, call this tool to rebuild its
        summary from the remaining valid edges. This prevents stale duplicate
        text from persisting in entity summaries.

        Accepts either *entity_uuid* (the canonical FalkorDB node UUID) or
        *entity_name* (exact entity name — resolved to a UUID automatically).
        When both are supplied, entity_uuid takes precedence. At least one must
        be provided.

        The summary is rebuilt by deduplicating the facts of all currently-valid
        RELATES_TO edges — no LLM call is made.

        Args:
            project_id: Project scope (required)
            entity_uuid: UUID of the Graphiti Entity node to refresh (optional when
                entity_name is provided)
            entity_name: Exact name of the Entity node to resolve and refresh
                (optional when entity_uuid is provided)
            agent_id: Which agent is calling (optional, auto-derived from MCP context)
            session_id: Session context (optional, auto-derived from MCP context)
            metadata: Optional key-value pairs (may contain _causation_id for recon)
        """
        agent_id, session_id = _resolve_identity(agent_id, session_id, ctx)
        if err := validate_project_id(project_id):
            return err
        if not entity_uuid and not entity_name:
            return {
                'error': 'Either entity_uuid or entity_name must be provided',
                'error_type': 'ValidationError',
            }
        try:
            causation_id, source, _ = _extract_causation(metadata, agent_id)
            return await memory_service.refresh_entity_summary(
                entity_uuid=entity_uuid or None,
                entity_name=entity_name or None,
                project_id=project_id,
                agent_id=agent_id, session_id=session_id,
                causation_id=causation_id, _source=source,
            )
        except Exception as e:
            logger.error(f'refresh_entity_summary error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def merge_entities(
        deprecated_uuid: str,
        surviving_uuid: str,
        project_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Consolidate two duplicate entity nodes into one by redirecting all edges.

        When the same real-world concept exists as two separate Entity nodes (e.g.,
        'Anthropic' and 'Anthropic Inc'), use this tool to merge them. All RELATES_TO
        edges from the deprecated node are redirected to the surviving node. The
        deprecated node is then deleted and the surviving node's summary is rebuilt
        from its (now-combined) edges.

        This operation is irreversible. Always verify both UUIDs before calling.

        Args:
            deprecated_uuid: UUID of the entity node to delete (will be removed)
            surviving_uuid: UUID of the entity node to keep (absorbs all edges)
            project_id: Project scope (required)
            agent_id: Which agent is calling (optional, auto-derived from MCP context)
            session_id: Session context (optional, auto-derived from MCP context)
            metadata: Optional key-value pairs (may contain _causation_id for recon)
        """
        agent_id, session_id = _resolve_identity(agent_id, session_id, ctx)
        if err := validate_project_id(project_id):
            return err
        if not deprecated_uuid or not deprecated_uuid.strip():
            return {'error': 'deprecated_uuid must be a non-empty string', 'error_type': 'ValidationError'}
        if not surviving_uuid or not surviving_uuid.strip():
            return {'error': 'surviving_uuid must be a non-empty string', 'error_type': 'ValidationError'}
        if deprecated_uuid.strip() == surviving_uuid.strip():
            return {'error': 'deprecated_uuid and surviving_uuid must be different', 'error_type': 'ValidationError'}
        try:
            causation_id, source, _ = _extract_causation(metadata, agent_id)
            return await memory_service.merge_entities(
                deprecated_uuid=deprecated_uuid,
                surviving_uuid=surviving_uuid,
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                causation_id=causation_id,
                _source=source,
            )
        except Exception as e:
            logger.error(f'merge_entities error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def rebuild_entity_summaries(
        project_id: str,
        force: bool = False,
        dry_run: bool = False,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Batch-rebuild Graphiti entity summaries from their current valid edges.

        Scans all Entity nodes in the project graph and identifies those whose
        summary is out of sync with their currently-valid RELATES_TO edges
        (duplicated lines, or lines from invalidated edges).  For each stale
        entity, rebuilds the summary using the same deduplication logic as
        refresh_entity_summary — no LLM call is made.

        Use ``dry_run=True`` to inspect which entities are stale without making
        any changes.  Use ``force=True`` to rebuild every entity regardless of
        detected staleness.

        Args:
            project_id: Project scope (required)
            force: Rebuild every entity regardless of staleness (default: false)
            dry_run: Detect stale entities but do not write summaries (default: false)
            agent_id: Which agent is calling (optional, auto-derived from MCP context)
            session_id: Session context (optional, auto-derived from MCP context)
            metadata: Optional key-value pairs (may contain _causation_id for recon)
        """
        agent_id, session_id = _resolve_identity(agent_id, session_id, ctx)
        if err := validate_project_id(project_id):
            return err
        try:
            causation_id, source, _ = _extract_causation(metadata, agent_id)
            return await memory_service.rebuild_entity_summaries(
                project_id=project_id,
                force=force,
                dry_run=dry_run,
                agent_id=agent_id,
                session_id=session_id,
                causation_id=causation_id,
                _source=source,
            )
        except Exception as e:
            logger.error(f'rebuild_entity_summaries error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

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
            return {'error': str(e), 'error_type': type(e).__name__}

    # ------------------------------------------------------------------
    # Queue management tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def replay_to_graphiti(
        project_id: str,
        source_store: str = 'mem0',
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Replay memories from Mem0 into Graphiti via the durable write queue.

        Use this to backfill the knowledge graph from Mem0 after Graphiti write
        failures, or to migrate memories into the graph. Items are processed
        through the queue with retry and dead-lettering.

        Args:
            project_id: Project whose memories to replay
            source_store: Source store to replay from (currently only "mem0")
            limit: Max memories to replay (None = all)
        """
        if err := validate_project_id(project_id):
            return err
        try:
            count = await memory_service.replay_from_store(
                source_project_id=project_id,
                limit=limit,
            )
            return {'status': 'queued', 'items_queued': count, 'project_id': project_id}
        except Exception as e:
            logger.error(f'replay_to_graphiti error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_queue_stats() -> dict[str, Any]:
        """Get durable write queue statistics — pending, retry, dead, completed
        counts and oldest pending item age. Use to monitor queue health.
        """
        try:
            if memory_service.durable_queue is None:
                return {'error': 'Queue not initialized', 'error_type': 'ConfigurationError'}
            return await memory_service.durable_queue.get_stats()
        except Exception as e:
            logger.error(f'get_queue_stats error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def replay_dead_letters(
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Reset dead-lettered queue items back to pending for retry.

        Dead-lettered items are writes that exhausted all retry attempts.
        This resets them so workers can try again (e.g. after fixing the
        underlying issue).

        Args:
            project_id: Scope to a specific project (optional — all if omitted)
        """
        try:
            if memory_service.durable_queue is None:
                return {'error': 'Queue not initialized', 'error_type': 'ConfigurationError'}
            count = await memory_service.durable_queue.replay_dead(group_id=project_id)
            return {'status': 'replayed', 'items_reset': count}
        except Exception as e:
            logger.error(f'replay_dead_letters error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    # ------------------------------------------------------------------
    # Reconciliation tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def trigger_reconciliation(project_id: str) -> dict[str, Any]:
        """Manually trigger a full reconciliation cycle for a project.

        Bypasses normal threshold/staleness logic. The reconciliation harness
        will pick this up on its next loop iteration (~5 seconds).

        Args:
            project_id: Project to trigger reconciliation for
        """
        if err := validate_project_id(project_id):
            return err
        if not _taskmaster_configured:
            return {
                'error': 'Taskmaster is not configured. Cannot trigger reconciliation.',
                'error_type': 'ConfigurationError',
            }
        try:
            await task_interceptor.buffer.request_trigger(project_id)  # type: ignore[union-attr]
            return {
                'status': 'requested',
                'project_id': project_id,
                'message': 'Reconciliation will trigger within ~5 seconds',
            }
        except Exception as e:
            logger.error(f'trigger_reconciliation error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def unhalt_reconciliation(project_id: str) -> dict[str, Any]:
        """Clear a judge-imposed halt on reconciliation for a project.

        The judge halts a project when it detects serious issues or error
        trends. This tool clears the halt so reconciliation cycles can resume.

        Args:
            project_id: Project to unhalt
        """
        if err := validate_project_id(project_id):
            return err
        if reconciliation_harness is None or reconciliation_harness.judge is None:
            return {
                'error': 'Reconciliation harness or judge is not configured.',
                'error_type': 'ConfigurationError',
            }
        was_halted = reconciliation_harness.judge.is_halted(project_id)
        reconciliation_harness.judge.unhalt(project_id)
        return {
            'status': 'unhalted' if was_halted else 'already_running',
            'project_id': project_id,
            'message': (
                f'Reconciliation unhalted for {project_id}. Next cycle will run within ~5 seconds.'
                if was_halted
                else f'Project {project_id} was not halted.'
            ),
        }

    # ------------------------------------------------------------------
    # Task proxy tools (always registered; errors if Taskmaster unavailable)
    # ------------------------------------------------------------------

    # If no interceptor was provided, create a bare one so tools are always
    # callable and return structured errors when Taskmaster is unavailable.
    if task_interceptor is None:
        from fused_memory.middleware.task_interceptor import TaskInterceptor
        from fused_memory.reconciliation.event_buffer import EventBuffer

        _fallback_buffer = EventBuffer(db_path=None)
        task_interceptor = TaskInterceptor(None, None, _fallback_buffer)

    @mcp.tool()
    async def get_tasks(
        project_root: str,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """List all tasks in the project.

        Args:
            project_root: Absolute path to project root
            tag: Tag context (optional)
        """
        if err := validate_project_root(project_root):
            return err
        try:
            return await task_interceptor.get_tasks(project_root=project_root, tag=tag)
        except Exception as e:
            logger.error(f'get_tasks error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_task(
        id: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Get a single task by ID.

        Args:
            id: Task ID (e.g., "15", "15.2")
            project_root: Absolute path to project root
            tag: Tag context (optional)
        """
        if err := validate_project_root(project_root):
            return err
        try:
            return await task_interceptor.get_task(
                task_id=id, project_root=project_root, tag=tag
            )
        except Exception as e:
            logger.error(f'get_task error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def set_task_status(
        id: str,
        status: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Update task status. Triggers targeted reconciliation for
        done/blocked/cancelled/deferred transitions.

        Reconciliation may: attach memory_hints to the task, write completion
        knowledge to memory stores, or flag dependent tasks that need attention.

        Args:
            id: Task ID (comma-separated for multiple)
            status: pending, done, in-progress, blocked, review, deferred, or cancelled
            project_root: Absolute path to project root
            tag: Tag context (optional)
        """
        if err := validate_project_root(project_root):
            return err
        if status not in _VALID_TASK_STATUSES:
            return {
                'error': (
                    f'Invalid status {status!r}. '
                    f'Must be one of {sorted(_VALID_TASK_STATUSES)}.'
                ),
                'error_type': 'ValidationError',
            }
        try:
            return await task_interceptor.set_task_status(
                task_id=id, status=status, project_root=project_root, tag=tag
            )
        except Exception as e:
            logger.error(f'set_task_status error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def add_task(
        project_root: str,
        prompt: str | None = None,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        dependencies: str | None = None,
        priority: str | None = None,
        metadata: str | dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Add a new task to the project.

        Consider including memory_hints in the description or details so future
        agents can prefetch relevant context (e.g. entity names to look up,
        search queries to run).

        Args:
            project_root: Absolute path to project root
            prompt: Task description for AI generation
            title: Manual task title
            description: Manual task description
            details: Manual task details
            dependencies: Comma-separated dependency task IDs
            priority: high, medium, or low
            metadata: Task metadata (e.g. {"source": "review-cycle", "modules": ["path/to/module"]}).
                Persisted via a follow-up update_task call after creation.
            tag: Tag context (optional)
        """
        if err := validate_project_root(project_root):
            return err
        try:
            return await task_interceptor.add_task(
                project_root=project_root,
                prompt=prompt,
                title=title,
                description=description,
                details=details,
                dependencies=dependencies,
                priority=priority,
                metadata=metadata,
                tag=tag,
            )
        except Exception as e:
            logger.error(f'add_task error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def update_task(
        id: str,
        project_root: str,
        prompt: str | None = None,
        metadata: str | dict | None = None,
        append: bool = False,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing task.

        Args:
            id: Task ID to update
            project_root: Absolute path to project root
            prompt: New information to incorporate
            metadata: JSON metadata to merge (object or JSON string)
            append: Append instead of full update
            tag: Tag context (optional)
        """
        if err := validate_project_root(project_root):
            return err
        try:
            if isinstance(metadata, dict):
                metadata = json.dumps(metadata)
            return await task_interceptor.update_task(
                task_id=id,
                project_root=project_root,
                prompt=prompt,
                metadata=metadata,
                append=append,
                tag=tag,
            )
        except Exception as e:
            logger.error(f'update_task error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def add_subtask(
        id: str,
        project_root: str,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Add a subtask to an existing task.

        Args:
            id: Parent task ID
            project_root: Absolute path to project root
            title: Subtask title
            description: Subtask description
            details: Subtask details
            tag: Tag context (optional)
        """
        if err := validate_project_root(project_root):
            return err
        try:
            return await task_interceptor.add_subtask(
                parent_id=id,
                project_root=project_root,
                title=title,
                description=description,
                details=details,
                tag=tag,
            )
        except Exception as e:
            logger.error(f'add_subtask error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def remove_task(
        id: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Remove a task or subtask.

        Args:
            id: Task/subtask ID to remove (comma-separated for multiple)
            project_root: Absolute path to project root
            tag: Tag context (optional)
        """
        if err := validate_project_root(project_root):
            return err
        try:
            return await task_interceptor.remove_task(
                task_id=id, project_root=project_root, tag=tag
            )
        except Exception as e:
            logger.error(f'remove_task error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def add_dependency(
        id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Add a dependency between tasks.

        Args:
            id: Task ID that will depend on another
            depends_on: Task ID that becomes a dependency
            project_root: Absolute path to project root
            tag: Tag context (optional)
        """
        if err := validate_project_root(project_root):
            return err
        try:
            return await task_interceptor.add_dependency(
                task_id=id,
                depends_on=depends_on,
                project_root=project_root,
                tag=tag,
            )
        except Exception as e:
            logger.error(f'add_dependency error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def remove_dependency(
        id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Remove a dependency between tasks.

        Args:
            id: Task ID to remove dependency from
            depends_on: Dependency task ID to remove
            project_root: Absolute path to project root
            tag: Tag context (optional)
        """
        if err := validate_project_root(project_root):
            return err
        try:
            return await task_interceptor.remove_dependency(
                task_id=id,
                depends_on=depends_on,
                project_root=project_root,
                tag=tag,
            )
        except Exception as e:
            logger.error(f'remove_dependency error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def expand_task(
        id: str,
        project_root: str,
        num: str | None = None,
        prompt: str | None = None,
        force: bool = False,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Expand a task into subtasks. Triggers bulk reconciliation.

        Args:
            id: Task ID to expand
            project_root: Absolute path to project root
            num: Number of subtasks to generate
            prompt: Additional context for generation
            force: Force expansion even if subtasks exist
            tag: Tag context (optional)
        """
        if err := validate_project_root(project_root):
            return err
        try:
            return await task_interceptor.expand_task(
                task_id=id,
                project_root=project_root,
                num=num,
                prompt=prompt,
                force=force,
                tag=tag,
            )
        except Exception as e:
            logger.error(f'expand_task error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def parse_prd(
        input: str,
        project_root: str,
        num_tasks: str | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Parse a PRD document to generate tasks. Triggers bulk reconciliation.

        Args:
            input: Path to PRD file (.txt, .md, etc.)
            project_root: Absolute path to project root
            num_tasks: Approximate number of tasks to generate
            tag: Tag context (optional)
        """
        if err := validate_project_root(project_root):
            return err
        try:
            return await task_interceptor.parse_prd(
                input_path=input,
                project_root=project_root,
                num_tasks=num_tasks,
                tag=tag,
            )
        except Exception as e:
            logger.error(f'parse_prd error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    return mcp
