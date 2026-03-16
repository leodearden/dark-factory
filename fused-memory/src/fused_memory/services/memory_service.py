"""Core orchestration layer — owns backends, classifier, router, durable queue."""

from __future__ import annotations

import asyncio
import logging
import uuid as uuid_mod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from graphiti_core.nodes import EpisodeType

from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.enums import (
    GRAPHITI_PRIMARY,
    MEM0_PRIMARY,
    MemoryCategory,
    SourceStore,
)
from fused_memory.models.memory import (
    AddEpisodeResponse,
    AddMemoryResponse,
    ClassificationResult,
    MemoryResult,
    ReadRouteResult,
)
from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.models.scope import Scope
from fused_memory.routing.classifier import WriteClassifier
from fused_memory.routing.router import ReadRouter
from fused_memory.services.durable_queue import DurableWriteQueue

if TYPE_CHECKING:
    from fused_memory.reconciliation.event_buffer import EventBuffer

logger = logging.getLogger(__name__)


class MemoryService:
    """Central orchestration — fused read/write across Graphiti + Mem0."""

    def __init__(self, config: FusedMemoryConfig):
        self.config = config
        self.graphiti = GraphitiBackend(config)
        self.mem0 = Mem0Backend(config)
        self.classifier = WriteClassifier(config)
        self.router = ReadRouter(config)
        self.durable_queue: DurableWriteQueue | None = None
        self._event_buffer: EventBuffer | None = None

    def set_event_buffer(self, buffer: EventBuffer) -> None:
        """Wire the reconciliation event buffer into the service."""
        self._event_buffer = buffer

    async def _emit_event(self, event: ReconciliationEvent) -> None:
        if self._event_buffer:
            await self._event_buffer.push(event)

    async def initialize(self) -> None:
        """Initialize backends and the durable write queue."""
        await self.graphiti.initialize()

        qcfg = self.config.queue
        self.durable_queue = DurableWriteQueue(
            data_dir=qcfg.data_dir,
            execute_write=self._execute_graphiti_write,
            workers_per_group=qcfg.workers_per_group,
            semaphore_limit=qcfg.semaphore_limit,
            max_attempts=qcfg.max_attempts,
            retry_base_seconds=qcfg.retry_base_seconds,
            write_timeout_seconds=qcfg.write_timeout_seconds,
        )
        self.durable_queue.register_callback(
            'dual_write_episode', self._dual_write_callback
        )
        await self.durable_queue.initialize()

        logger.info('MemoryService initialized')

    async def close(self) -> None:
        if self.durable_queue:
            await self.durable_queue.close()
        await self.graphiti.close()

    # ------------------------------------------------------------------
    # Durable queue: execute write dispatcher
    # ------------------------------------------------------------------

    async def _execute_graphiti_write(
        self, operation: str, payload: dict[str, Any]
    ) -> Any:
        """Dispatch a queued write to the Graphiti backend."""
        source_str = payload.get('source', 'text')
        try:
            episode_type = EpisodeType[source_str]
        except (KeyError, AttributeError):
            episode_type = EpisodeType.text

        return await self.graphiti.add_episode(
            name=payload.get('name', ''),
            content=payload['content'],
            source=episode_type,
            group_id=payload['group_id'],
            source_description=payload.get('source_description', ''),
            uuid=payload.get('uuid'),
        )

    async def _dual_write_callback(
        self, callback_type: str, result: Any, payload: dict[str, Any]
    ) -> None:
        """Post-process callback: classify extracted edges and dual-write to Mem0."""
        scope = Scope(
            project_id=payload.get('project_id', 'main'),
            agent_id=payload.get('agent_id'),
            session_id=payload.get('session_id'),
        )
        await self._dual_write_from_episode(result, scope)

    # ------------------------------------------------------------------
    # Write: add_episode
    # ------------------------------------------------------------------

    async def add_episode(
        self,
        content: str,
        source: str = 'text',
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        reference_time: datetime | None = None,
        source_description: str = '',
    ) -> AddEpisodeResponse:
        """Full ingestion pipeline — durably enqueue episode, return immediately."""
        scope = Scope(project_id=project_id, agent_id=agent_id, session_id=session_id)
        episode_id = str(uuid_mod.uuid4())

        # Parse source type name for storage
        try:
            source_name = EpisodeType[source.lower()].name
        except (KeyError, AttributeError):
            source_name = 'text'

        assert self.durable_queue is not None

        await self.durable_queue.enqueue(
            group_id=scope.graphiti_group_id,
            operation='add_episode',
            payload={
                'name': f'episode_{episode_id[:8]}',
                'content': content,
                'source': source_name,
                'group_id': scope.graphiti_group_id,
                'source_description': source_description,
                'uuid': episode_id,
                # Scope fields for callback reconstruction
                'project_id': project_id,
                'agent_id': agent_id,
                'session_id': session_id,
            },
            callback_type='dual_write_episode',
        )

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.episode_added,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(timezone.utc),
            payload={'episode_id': episode_id, 'content_preview': content[:200]},
        ))

        return AddEpisodeResponse(
            episode_id=episode_id,
            status='queued',
            message=f'Episode queued for processing in project {project_id}',
        )

    async def _dual_write_from_episode(self, result: Any, scope: Scope) -> None:
        """Classify extracted facts and write Mem0-bound memories."""
        if result is None:
            return

        # Graphiti's add_episode returns AddEpisodeResults with .entity_edges
        edges = getattr(result, 'entity_edges', None) or []
        for edge in edges:
            fact_text = getattr(edge, 'fact', None) or str(edge)
            try:
                classification = await self.classifier.classify(fact_text)
                if classification.primary in MEM0_PRIMARY or classification.secondary is not None:
                    metadata = {
                        'category': classification.primary.value,
                        'source': 'episode_extraction',
                        'confidence': classification.confidence,
                    }
                    if classification.secondary:
                        metadata['secondary_category'] = classification.secondary.value
                    await self.mem0.add(content=fact_text, scope=scope, metadata=metadata)
                    logger.debug(f'Dual-wrote fact to Mem0: {fact_text[:80]}')
            except Exception as e:
                logger.error(f'Dual-write failed for fact: {e}')

    # ------------------------------------------------------------------
    # Write: add_memory
    # ------------------------------------------------------------------

    async def add_memory(
        self,
        content: str,
        category: str | MemoryCategory | None = None,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        dual_write: bool = False,
    ) -> AddMemoryResponse:
        """Lightweight classified write — skip extraction pipeline."""
        scope = Scope(project_id=project_id, agent_id=agent_id, session_id=session_id)

        # Resolve category
        if category is None:
            classification = await self.classifier.classify(content)
            resolved_category = classification.primary
        elif isinstance(category, str):
            resolved_category = MemoryCategory(category)
        else:
            resolved_category = category

        memory_ids: list[str] = []
        stores_written: list[SourceStore] = []
        meta = dict(metadata or {})
        meta['category'] = resolved_category.value

        write_graphiti = (
            resolved_category in GRAPHITI_PRIMARY or dual_write
        )
        write_mem0 = (
            resolved_category in MEM0_PRIMARY or dual_write
        )

        _graphiti_error = None

        # Graphiti: enqueue via durable queue (async, but durably persisted)
        if write_graphiti:
            try:
                assert self.durable_queue is not None
                await self.durable_queue.enqueue(
                    group_id=scope.graphiti_group_id,
                    operation='add_memory_graphiti',
                    payload={
                        'name': f'memory_{resolved_category.value}',
                        'content': content,
                        'source': 'text',
                        'group_id': scope.graphiti_group_id,
                        'source_description': f'add_memory:{resolved_category.value}',
                    },
                )
                # Durably persisted to SQLite — report as written
                stores_written.append(SourceStore.graphiti)
            except Exception as e:
                logger.error(f'Graphiti enqueue failed: {e}')
                _graphiti_error = f'{type(e).__name__}: {e}'

        # Mem0: direct write (fast, no queue needed)
        if write_mem0:
            try:
                result = await self.mem0.add(content=content, scope=scope, metadata=meta)
                results = result.get('results', [])
                for r in results:
                    if 'id' in r:
                        memory_ids.append(r['id'])
                stores_written.append(SourceStore.mem0)
            except Exception as e:
                logger.error(f'Mem0 write failed: {e}')

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.memory_added,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(timezone.utc),
            payload={
                'memory_ids': memory_ids,
                'category': resolved_category.value,
                'content_preview': content[:200],
            },
        ))

        msg = f'Memory stored in {[s.value for s in stores_written]}'
        if _graphiti_error:
            msg += f' [graphiti_error: {_graphiti_error}]'

        return AddMemoryResponse(
            memory_ids=memory_ids,
            stores_written=stores_written,
            category=resolved_category,
            message=msg,
        )

    # ------------------------------------------------------------------
    # Replay: re-ingest Mem0 memories into Graphiti
    # ------------------------------------------------------------------

    async def replay_from_store(
        self,
        source_project_id: str,
        target_project_id: str | None = None,
    ) -> int:
        """Fetch all memories from Mem0 and enqueue each for Graphiti write.

        Returns the count of items queued.
        """
        target = target_project_id or source_project_id
        scope = Scope(project_id=source_project_id)
        all_mems = await self.mem0.get_all(scope, limit=1000)
        memories = all_mems.get('results', [])
        if not memories:
            return 0

        assert self.durable_queue is not None
        batch = []
        for mem in memories:
            content = mem.get('memory', '')
            if not content:
                continue
            meta = mem.get('metadata', {}) or {}
            category = meta.get('category', 'observations_and_summaries')
            batch.append({
                'group_id': target,
                'operation': 'add_memory_graphiti',
                'payload': {
                    'name': f'replay_{category}',
                    'content': content,
                    'source': 'text',
                    'group_id': target,
                    'source_description': f'replay_from_mem0:{category}',
                },
            })

        if batch:
            await self.durable_queue.enqueue_batch(batch)
        return len(batch)

    # ------------------------------------------------------------------
    # Read: search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        project_id: str = 'main',
        categories: list[str] | None = None,
        stores: list[str] | None = None,
        limit: int = 10,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[MemoryResult]:
        """Unified search across both stores with automatic fan-out."""
        scope = Scope(project_id=project_id, agent_id=agent_id, session_id=session_id)

        # Determine routing
        stores_override = [SourceStore(s) for s in stores] if stores else None
        route: ReadRouteResult = await self.router.route(query, stores_override)

        # Fan out to stores in parallel
        tasks: dict[SourceStore, asyncio.Task] = {}
        if SourceStore.graphiti in route.stores:
            tasks[SourceStore.graphiti] = asyncio.create_task(
                self._search_graphiti(query, scope, limit)
            )
        if SourceStore.mem0 in route.stores:
            tasks[SourceStore.mem0] = asyncio.create_task(
                self._search_mem0(query, scope, limit)
            )

        results: list[MemoryResult] = []
        for store, task in tasks.items():
            try:
                store_results = await task
                results.extend(store_results)
            except Exception as e:
                logger.error(f'Search failed for {store.value}: {e}')

        # Sort: primary store results first, then by relevance score
        def sort_key(r: MemoryResult) -> tuple[int, float]:
            is_primary = 0 if r.source_store == route.primary_store else 1
            return (is_primary, -r.relevance_score)

        results.sort(key=sort_key)

        # Filter by categories if requested
        if categories:
            cat_set = {MemoryCategory(c) for c in categories}
            results = [r for r in results if r.category in cat_set]

        return results[:limit]

    async def _search_graphiti(
        self, query: str, scope: Scope, limit: int
    ) -> list[MemoryResult]:
        """Search Graphiti and convert results to MemoryResult."""
        edges = await self.graphiti.search(
            query=query,
            group_ids=[scope.graphiti_group_id],
            num_results=limit,
        )
        results = []
        for i, edge in enumerate(edges):
            fact = getattr(edge, 'fact', str(edge))
            valid_at = getattr(edge, 'valid_at', None)
            invalid_at = getattr(edge, 'invalid_at', None)
            temporal = None
            if valid_at or invalid_at:
                temporal = {
                    'valid_at': str(valid_at) if valid_at else None,
                    'invalid_at': str(invalid_at) if invalid_at else None,
                }

            # Extract entity names from source/target nodes
            entities = []
            source_node = getattr(edge, 'source_node', None)
            target_node = getattr(edge, 'target_node', None)
            if source_node and hasattr(source_node, 'name'):
                entities.append(source_node.name)
            if target_node and hasattr(target_node, 'name'):
                entities.append(target_node.name)

            # Episode provenance
            episodes = getattr(edge, 'episodes', []) or []
            provenance = [str(ep) for ep in episodes]

            # Score: rank-based (no explicit score from Graphiti search)
            score = max(0.0, 1.0 - (i * 0.05))

            results.append(MemoryResult(
                id=getattr(edge, 'uuid', str(i)),
                content=fact,
                category=None,
                source_store=SourceStore.graphiti,
                relevance_score=score,
                provenance=provenance,
                temporal=temporal,
                entities=entities,
            ))
        return results

    async def _search_mem0(
        self, query: str, scope: Scope, limit: int
    ) -> list[MemoryResult]:
        """Search Mem0 and convert results to MemoryResult."""
        response = await self.mem0.search(query=query, scope=scope, limit=limit)
        mem0_results = response.get('results', [])
        results = []
        for item in mem0_results:
            content = item.get('memory', '')
            score = float(item.get('score', 0.0))
            meta = item.get('metadata', {}) or {}

            category = None
            if 'category' in meta:
                try:
                    category = MemoryCategory(meta['category'])
                except ValueError:
                    pass

            results.append(MemoryResult(
                id=item.get('id', ''),
                content=content,
                category=category,
                source_store=SourceStore.mem0,
                relevance_score=min(score, 1.0),
                metadata=meta,
            ))
        return results

    # ------------------------------------------------------------------
    # Read: get_entity
    # ------------------------------------------------------------------

    async def get_entity(
        self,
        name: str,
        project_id: str = 'main',
    ) -> dict:
        """Entity lookup in Graphiti — returns nodes + edges."""
        nodes = await self.graphiti.search_nodes(
            query=name,
            group_ids=[project_id],
            max_nodes=5,
        )
        edges = await self.graphiti.search(
            query=name,
            group_ids=[project_id],
            num_results=10,
        )

        node_data = []
        for n in nodes:
            node_data.append({
                'uuid': getattr(n, 'uuid', None),
                'name': getattr(n, 'name', None),
                'summary': getattr(n, 'summary', None),
                'labels': getattr(n, 'labels', []),
            })

        edge_data = []
        for e in edges:
            edge_data.append({
                'uuid': getattr(e, 'uuid', None),
                'fact': getattr(e, 'fact', str(e)),
            })

        return {'nodes': node_data, 'edges': edge_data}

    # ------------------------------------------------------------------
    # Read: get_episodes
    # ------------------------------------------------------------------

    async def get_episodes(
        self,
        project_id: str = 'main',
        last_n: int = 10,
    ) -> list[dict]:
        """Retrieve raw episodes from Graphiti."""
        episodes = await self.graphiti.retrieve_episodes(
            group_ids=[project_id],
            last_n=last_n,
        )
        return [
            {
                'uuid': getattr(ep, 'uuid', None),
                'name': getattr(ep, 'name', None),
                'content': getattr(ep, 'content', None),
                'created_at': str(getattr(ep, 'created_at', '')) or None,
                'source': getattr(ep, 'source', None),
                'group_id': getattr(ep, 'group_id', None),
            }
            for ep in episodes
        ]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_memory(
        self,
        memory_id: str,
        store: str,
        project_id: str = 'main',
    ) -> dict:
        """Delete a memory from the specified store."""
        scope = Scope(project_id=project_id)
        source = SourceStore(store)

        if source == SourceStore.graphiti:
            await self.graphiti.remove_episode(memory_id)
            result = {'status': 'deleted', 'store': 'graphiti', 'id': memory_id}
        else:
            del_result = await self.mem0.delete(memory_id, scope)
            result = {'status': 'deleted', 'store': 'mem0', 'id': memory_id, **del_result}

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.memory_deleted,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(timezone.utc),
            payload={'memory_id': memory_id, 'store': store},
        ))

        return result

    async def delete_episode(
        self,
        episode_id: str,
        project_id: str = 'main',
        cascade: bool = True,
    ) -> dict:
        """Delete a Graphiti episode."""
        await self.graphiti.remove_episode(episode_id)
        return {'status': 'deleted', 'episode_id': episode_id, 'cascade': cascade}

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    async def get_status(self, project_id: str | None = None) -> dict:
        """Health check and statistics for both backends."""
        status: dict[str, Any] = {}

        # Graphiti health
        try:
            client = self.graphiti._require_client()
            async with client.driver.session() as session:
                result = await session.run('MATCH (n) RETURN count(n) as count')
                records = [record async for record in result]
                node_count = records[0]['count'] if records else 0
            status['graphiti'] = {'connected': True, 'node_count': node_count}
        except Exception as e:
            status['graphiti'] = {'connected': False, 'error': str(e)}

        # Mem0 health
        try:
            if project_id:
                scope = Scope(project_id=project_id)
                all_mems = await self.mem0.get_all(scope, limit=1)
                mem_count = len(all_mems.get('results', []))
                status['mem0'] = {'connected': True, 'memory_count': mem_count}
            else:
                status['mem0'] = {'connected': True, 'memory_count': 'unknown (no project_id)'}
        except Exception as e:
            status['mem0'] = {'connected': False, 'error': str(e)}

        # Queue stats
        if self.durable_queue:
            try:
                status['queue'] = await self.durable_queue.get_stats()
            except Exception as e:
                status['queue'] = {'error': str(e)}

        return status

    def get_consolidation_tools(self) -> dict:
        """Return the restricted tool set for the consolidation agent."""
        return {
            'search': self.search,
            'add_memory': self.add_memory,
            'delete_memory': self.delete_memory,
            'get_episodes': self.get_episodes,
            'get_status': self.get_status,
        }
