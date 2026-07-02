"""Core orchestration layer — owns backends, classifier, router, durable queue."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import uuid as uuid_mod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

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
    EpisodeStatus,
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
from fused_memory.utils.async_utils import propagate_cancellations

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
    from fused_memory.reconciliation.event_buffer import EventBuffer
    from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry
    from fused_memory.services.write_journal import WriteJournal

logger = logging.getLogger(__name__)

# Canonical relational verb for dependency facts (mirrors routing/classifier.py:19).
# Used by _restore_superseded_dependency_edges to identify edges that should
# never be superseded by LLM edge-resolution.
_DEPENDENCY_FACT_RE = re.compile(r'\bdepends on\b', re.I)

# Mem0Backend.add() (backends/mem0_client.py) currently pins infer=False
# unconditionally for every write, which guarantees a successful mem0.add()
# always returns exactly one result with an id — so an empty result is
# always anomalous (task 1974; see the empty-result WARNING below). Flip
# this to False in lockstep if that pin is ever lifted or made
# configurable, so the WARNING doesn't degenerate into a recurring,
# non-actionable per-write log line under a legitimate infer-driven
# dedup/no-op.
_MEM0_ADD_INFER_PINNED_FALSE = True


def _is_dependency_fact(fact: str | None) -> bool:
    """Return True when *fact* expresses a dependency relationship.

    Uses the canonical ``depends on`` phrasing (case-insensitive, word-boundary
    anchored) that matches the project's relational verb in classifier.py.
    """
    return bool(fact) and _DEPENDENCY_FACT_RE.search(fact) is not None  # type: ignore[arg-type]


class MemoryNotFoundError(Exception):
    """Raised when a mem0 memory id is not found."""


def _serialize_temporal(
    valid_at: Any,
    invalid_at: Any,
) -> dict[str, str | None] | None:
    """Serialize valid_at/invalid_at to an ISO 8601 dict or None.

    Returns None when both values are None (common case — no temporal context).
    Uses .isoformat() when available, falls back to str() for pre-serialized strings
    or other types.
    """
    if valid_at is None and invalid_at is None:
        return None

    def _to_iso(v: Any) -> str | None:
        if v is None:
            return None
        if hasattr(v, 'isoformat'):
            return v.isoformat()
        return str(v)

    return {
        'valid_at': _to_iso(valid_at),
        'invalid_at': _to_iso(invalid_at),
    }


class SearchResults(list):
    """list subclass returned by MemoryService.search carrying in-band degrade metadata.

    All ~11 internal list-consuming callers (context_assembler, targeted, flag_dedup,
    mem0_dedup, task_knowledge_sync) keep working unchanged — only tools.py reads the
    extra attributes (task 1812).

    Attributes:
        degraded: True when one or more selected stores raised or timed out.
        failed_stores: List of store name strings (SourceStore.value) that failed.

    .. warning::
        The `degraded` and `failed_stores` metadata do **not** survive list-returning
        operations (slicing, sorted(), concatenation, list comprehensions).  Those
        operations return a plain ``list``, silently dropping the degrade metadata.
        Callers that need the metadata after a transform should read the attributes
        *before* the transform, or pass the SearchResults object directly without
        intermediate list operations.
    """

    def __init__(self, iterable=(), *, degraded: bool = False, failed_stores=None):
        super().__init__(iterable)
        self.degraded = degraded
        self.failed_stores: list[str] = failed_stores if failed_stores is not None else []


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
        self._write_journal: WriteJournal | None = None
        self.taskmaster: TaskBackendProtocol | None = None
        self.planned_episode_registry: PlannedEpisodeRegistry | None = None
        # Process-start baselines for uptime reporting
        self._started_at: datetime = datetime.now(UTC)
        self._start_monotonic: float = time.monotonic()

    def set_event_buffer(self, buffer: EventBuffer) -> None:
        """Wire the reconciliation event buffer into the service."""
        self._event_buffer = buffer

    def set_write_journal(self, journal: WriteJournal) -> None:
        """Wire the write journal for durable auditing."""
        self._write_journal = journal

    def set_planned_registry(self, registry: PlannedEpisodeRegistry) -> None:
        """Wire the planned episode registry into the service."""
        self.planned_episode_registry = registry

    async def _emit_event(self, event: ReconciliationEvent) -> None:
        if self._event_buffer:
            await self._event_buffer.push(event)

    async def initialize(self) -> None:
        """Initialize backends and the durable write queue."""
        await self.graphiti.initialize()

        qcfg = self.config.queue

        # Initialize planned episode registry (co-located with durable queue data).
        # If set_planned_registry() was called before initialize(), honour the external
        # registry instead of creating a new one (preventing the lifecycle conflict).
        if self.planned_episode_registry is None:
            from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry
            self.planned_episode_registry = PlannedEpisodeRegistry(data_dir=qcfg.data_dir)
            await self.planned_episode_registry.initialize()

        self.durable_queue = DurableWriteQueue(
            data_dir=qcfg.data_dir,
            execute_write=self._execute_durable_write,
            workers_per_group=qcfg.workers_per_group,
            semaphore_limit=qcfg.semaphore_limit,
            max_attempts=qcfg.max_attempts,
            retry_base_seconds=qcfg.retry_base_seconds,
            retry_max_delay_seconds=qcfg.retry_max_delay_seconds,
            write_timeout_seconds=qcfg.write_timeout_seconds,
            transient_max_attempts=qcfg.transient_max_attempts,
            transient_error_names=qcfg.transient_error_names,
        )
        self.durable_queue.register_callback(
            'dual_write_episode', self._dual_write_callback
        )
        self.durable_queue.register_callback(
            'refresh_entity_summaries', self._refresh_summaries_callback
        )
        await self.durable_queue.initialize()

        logger.info('MemoryService initialized')

    async def _safe_close(self, label: str, resource: Any) -> None:
        """Close one resource, logging any failure without re-raising.

        Lets close() continue through every resource even when one fails.
        """
        try:
            await resource.close()
        except Exception:
            logger.exception('MemoryService.close: %s.close failed', label)

    async def close(self) -> None:
        if self.durable_queue:
            await self._safe_close('durable_queue', self.durable_queue)
        await self._safe_close('graphiti', self.graphiti)
        await self._safe_close('mem0', self.mem0)
        if self._write_journal:
            await self._safe_close('write_journal', self._write_journal)
        if self._event_buffer:
            await self._safe_close('event_buffer', self._event_buffer)
        if self.planned_episode_registry:
            await self._safe_close('planned_episode_registry', self.planned_episode_registry)

    # ------------------------------------------------------------------
    # Journal helper
    # ------------------------------------------------------------------

    async def _journaled_backend_call(
        self,
        write_op_id: str | None,
        causation_id: str | None,
        backend: str,
        operation: str,
        payload: dict[str, Any],
        coro: Any,
    ) -> Any:
        """Execute a backend call and log to write journal."""
        result = None
        try:
            result = await coro
            if self._write_journal:
                await self._write_journal.log_backend_op(
                    write_op_id=write_op_id,
                    causation_id=causation_id,
                    backend=backend,
                    operation=operation,
                    payload=payload,
                    result_summary=str(result)[:500] if result else None,
                    success=True,
                )
            return result
        except Exception as e:
            if self._write_journal:
                await self._write_journal.log_backend_op(
                    write_op_id=write_op_id,
                    causation_id=causation_id,
                    backend=backend,
                    operation=operation,
                    payload=payload,
                    success=False,
                    error=str(e),
                )
            raise

    # ------------------------------------------------------------------
    # Durable queue: execute write dispatcher
    # ------------------------------------------------------------------

    async def _execute_durable_write(
        self, operation: str, payload: dict[str, Any]
    ) -> Any:
        """Route a queued write to the appropriate backend handler."""
        if operation == 'mem0_add':
            return await self._execute_mem0_write(payload)
        if operation == 'mem0_classify_and_add':
            return await self._execute_mem0_classify_and_add(payload)
        return await self._execute_graphiti_write(operation, payload)

    # ------------------------------------------------------------------
    # Dedup: remove duplicate edges created by a single add_episode call
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_fact(text: str) -> str:
        """Normalize a fact string for dedup comparison.

        Replicates graphiti_core's internal normalization:
        lowercase + collapse whitespace.  Two facts that differ only in
        capitalisation or spacing are treated as the same edge.
        """
        return re.sub(r'\s+', ' ', text.lower()).strip()

    async def _dedup_episode_edges(self, result: Any, *, group_id: str) -> int:
        """Remove duplicate edges produced by a single add_episode call.

        Graphiti's LLM extraction pipeline can emit multiple edges that
        express the same fact (same source_node_uuid, target_node_uuid,
        and normalised fact text).  This method groups the edges returned
        in *result* by that triple and deletes all but the first edge in
        each group via ``bulk_remove_edges``.

        Args:
            result: The value returned by ``add_episode`` (typically an
                    AddEpisodeResults object with an ``edges`` attribute).
                    Handles ``None`` and objects with empty/missing edges
                    gracefully.

        Returns:
            Number of duplicate edges removed (0 when nothing to do).
        """
        if result is None:
            return 0

        edges = getattr(result, 'edges', None) or getattr(result, 'entity_edges', None) or []
        if not edges:
            return 0

        # Group edges by (source_node_uuid, target_node_uuid, normalized_fact)
        seen: dict[tuple[str, str, str], str] = {}   # key → first uuid
        duplicates: list[str] = []

        for edge in edges:
            src_uuid = getattr(edge, 'source_node_uuid', '') or ''
            tgt_uuid = getattr(edge, 'target_node_uuid', '') or ''
            fact_norm = self._normalize_fact(getattr(edge, 'fact', '') or '')
            edge_uuid = getattr(edge, 'uuid', '') or ''
            key = (src_uuid, tgt_uuid, fact_norm)

            if key in seen:
                duplicates.append(edge_uuid)
            else:
                seen[key] = edge_uuid

        if not duplicates:
            return 0

        logger.info('Deduplicating %d edge(s) after add_episode', len(duplicates))
        return await self.graphiti.bulk_remove_edges(duplicates, group_id=group_id)

    async def _restore_superseded_dependency_edges(
        self, result: Any, *, group_id: str
    ) -> int:
        """Undo false dependency-edge invalidations caused by LLM edge-resolution.

        Graphiti's upstream ``add_episode`` LLM pipeline can falsely supersede
        existing "X depends on Y" edges when a new dependency fact is added for
        a hub entity (dependencies are additive, so supersession is wrong for
        them). This method scans the edges returned in *result* — exactly the
        edges this episode touched — and clears ``invalid_at`` for any that are
        both (a) invalidated AND (b) express a dependency fact.

        Modelled on ``_dedup_episode_edges``; handles None / empty result the
        same way. Legitimate dependency removals flow through
        ``remove_dependency`` (a different code path) and are unaffected.

        Args:
            result: The value returned by ``add_episode`` (typically an
                    AddEpisodeResults object with an ``edges`` attribute).
                    Handles ``None`` and objects with empty/missing edges
                    gracefully.

        Returns:
            Number of dependency edges whose invalidation was reversed (0 when
            nothing to do).
        """
        if result is None:
            return 0

        edges = (
            getattr(result, 'edges', None)
            or getattr(result, 'entity_edges', None)
            or []
        )
        if not edges:
            return 0

        restored = 0
        failed = 0
        for edge in edges:
            if getattr(edge, 'invalid_at', None) is None:
                continue
            fact = getattr(edge, 'fact', '') or ''
            if not _is_dependency_fact(fact):
                continue
            edge_uuid = getattr(edge, 'uuid', '') or ''
            try:
                await self.graphiti.update_edge(
                    edge_uuid, group_id=group_id, clear_invalid_at=True
                )
                restored += 1
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                # Best-effort: a transient backend error (lock contention,
                # write timeout) must not fail an already-committed episode
                # write.  Log and continue so the episode reports success.
                logger.exception(
                    'Failed to restore dependency edge %s; will retry on next episode',
                    edge_uuid,
                )
                failed += 1

        if restored > 0:
            logger.info(
                'Restored %d falsely-superseded dependency edge(s) after add_episode',
                restored,
            )
        if failed > 0:
            logger.warning(
                'Failed to restore %d dependency edge(s) after add_episode',
                failed,
            )
        return restored

    async def _execute_graphiti_write(
        self, operation: str, payload: dict[str, Any]
    ) -> Any:
        """Dispatch a queued write to the Graphiti backend."""
        source_str = payload.get('source', 'text')
        try:
            episode_type = EpisodeType[source_str]
        except (KeyError, AttributeError):
            episode_type = EpisodeType.text

        # Extract journal metadata from payload (injected at enqueue time)
        causation_id = payload.pop('_causation_id', None)
        write_op_id = payload.pop('_write_op_id', None)
        temporal_context = payload.pop('temporal_context', None)
        reference_time_iso = payload.pop('reference_time', None)
        reference_time = None
        if reference_time_iso is not None:
            try:
                reference_time = datetime.fromisoformat(reference_time_iso)
            except (ValueError, TypeError):
                logger.warning(
                    'Invalid reference_time %r in queue payload; treating as None',
                    reference_time_iso,
                )

        result = await self._journaled_backend_call(
            write_op_id=write_op_id,
            causation_id=causation_id,
            backend='graphiti',
            operation='add_episode',
            payload={'content': payload['content'][:200], 'group_id': payload.get('group_id')},
            coro=self.graphiti.add_episode(
                name=payload.get('name', ''),
                content=payload['content'],
                source=episode_type,
                group_id=payload['group_id'],
                source_description=payload.get('source_description', ''),
                uuid=payload.get('uuid'),
                temporal_context=temporal_context,
                reference_time=reference_time,
            ),
        )
        # Post-write dedup: remove duplicate edges created within this episode
        await self._dedup_episode_edges(result, group_id=payload['group_id'])
        # Post-write restore: undo false dependency-edge supersessions
        await self._restore_superseded_dependency_edges(result, group_id=payload['group_id'])

        # Register planning episodes so they can be filtered from search results
        if temporal_context == 'planning' and self.planned_episode_registry is not None:
            episode_uuid = payload.get('uuid')
            group_id = payload.get('group_id')
            if episode_uuid and group_id:
                await self.planned_episode_registry.register(episode_uuid, group_id)
            elif episode_uuid and not group_id:
                logger.warning(
                    'Skipping planned episode registration: group_id missing from payload '
                    'for episode %s',
                    episode_uuid,
                )

        return result

    async def _execute_mem0_write(self, payload: dict[str, Any]) -> Any:
        """Execute a queued Mem0 add operation."""
        causation_id = payload.pop('_causation_id', None)
        write_op_id = payload.pop('_write_op_id', None)
        scope = Scope(
            project_id=payload['project_id'],
            agent_id=payload.get('agent_id'),
            session_id=payload.get('session_id'),
        )
        metadata = payload.get('metadata', {})

        result = await self._journaled_backend_call(
            write_op_id=write_op_id,
            causation_id=causation_id,
            backend='mem0',
            operation='add',
            payload={'content': payload['content'][:200]},
            coro=self.mem0.add(
                content=payload['content'], scope=scope, metadata=metadata
            ),
        )

        # Log Layer 1 for the queued write
        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id or str(uuid_mod.uuid4()),
                causation_id=causation_id,
                source='durable_queue',
                operation='add_memory',
                project_id=payload['project_id'],
                agent_id=payload.get('agent_id'),
                session_id=payload.get('session_id'),
                params={
                    'content': payload['content'][:200],
                    'category': metadata.get('category', ''),
                },
                result_summary=str(result)[:500] if result else None,
                success=True,
            )

        return result

    async def _execute_mem0_classify_and_add(
        self, payload: dict[str, Any]
    ) -> Any:
        """Classify a fact extracted from an episode and write to Mem0 if appropriate."""
        fact_text = payload['fact_text']
        causation_id = payload.get('_causation_id')
        temporal_context = payload.get('temporal_context')
        write_op_id = str(uuid_mod.uuid4())
        scope = Scope(
            project_id=payload.get('project_id', 'main'),
            agent_id=payload.get('agent_id'),
            session_id=payload.get('session_id'),
        )

        classification = await self.classifier.classify(fact_text)
        if classification.primary not in MEM0_PRIMARY and classification.secondary is None:
            return None  # Not Mem0-bound

        metadata = {
            'category': classification.primary.value,
            'source': 'episode_extraction',
            'confidence': classification.confidence,
        }
        if classification.secondary:
            metadata['secondary_category'] = classification.secondary.value
        if temporal_context == 'planning':
            metadata['planned'] = True

        result = await self._journaled_backend_call(
            write_op_id=write_op_id,
            causation_id=causation_id,
            backend='mem0',
            operation='add',
            payload={'content': fact_text[:200]},
            coro=self.mem0.add(content=fact_text, scope=scope, metadata=metadata),
        )

        # Log Layer 1 for the derived write
        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source='dual_write',
                provenance='derived',
                operation='add_memory',
                project_id=scope.project_id,
                agent_id=scope.agent_id,
                session_id=scope.session_id,
                params={
                    'content': fact_text[:200],
                    'category': classification.primary.value,
                },
                result_summary=str(result)[:500] if result else None,
                success=True,
            )

        logger.debug(f'Durable dual-wrote fact to Mem0: {fact_text[:80]}')
        return result

    async def _refresh_entity_summaries_from_result(
        self, result: Any, group_id: str
    ) -> None:
        """Best-effort post-ingestion summary refresh for edge endpoints in *result*.

        Graphiti's ``add_episode`` LLM extraction pipeline invalidates/supersedes/
        dedups edges internally as a side effect of ingestion, without any
        fused-memory code observing which edges changed. This helper closes that
        gap: it reads the edges returned by the ingestion call (mirrors the
        ``result.edges`` / ``result.entity_edges`` idiom used elsewhere in this
        file), collects the deduplicated set of non-empty source/target node
        uuids, and calls ``refresh_entity_summary`` for each — so the entity
        nodes touched by this write get a freshly rebuilt summary.

        Args:
            result: The value returned by ``add_episode`` (or equivalent),
                    typically carrying an ``edges`` (or ``entity_edges``)
                    attribute. ``None`` and empty/missing edges are a no-op.
            group_id: Graphiti graph id to refresh within.
        """
        if result is None:
            return

        edges = getattr(result, 'edges', None) or getattr(result, 'entity_edges', None) or []
        if not edges:
            return

        uuids: dict[str, None] = {}  # ordered dedup set
        for edge in edges:
            src_uuid = getattr(edge, 'source_node_uuid', '') or ''
            tgt_uuid = getattr(edge, 'target_node_uuid', '') or ''
            if src_uuid:
                uuids[src_uuid] = None
            if tgt_uuid:
                uuids[tgt_uuid] = None

        for node_uuid in uuids:
            try:
                await self.graphiti.refresh_entity_summary(node_uuid, group_id=group_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    'post-ingest refresh_entity_summary failed for uuid=%s group=%s: %s',
                    node_uuid, group_id, exc,
                )

    async def _dual_write_callback(
        self, callback_type: str, result: Any, payload: dict[str, Any]
    ) -> None:
        """Post-process callback: extract facts and enqueue each for durable Mem0
        write, then trigger a best-effort post-ingestion entity-summary refresh.

        Instead of writing directly to Mem0 (fire-and-forget), we batch-enqueue
        each extracted fact as a ``mem0_classify_and_add`` queue item so it gets
        independent retry / dead-letter handling. After the Mem0 enqueue, we also
        refresh the summary of every entity node touched by an edge endpoint in
        this result — closing the ingestion-time staleness gap where graphiti_core
        invalidates/supersedes/dedups edges internally without any fused-memory
        code observing which nodes changed.
        """
        if result is None:
            return

        edges = getattr(result, 'edges', None) or getattr(result, 'entity_edges', None) or []

        if edges:
            project_id = payload.get('project_id', 'main')
            group_id = f'mem0_{project_id}'

            batch = [
                {
                    'group_id': group_id,
                    'operation': 'mem0_classify_and_add',
                    'payload': {
                        'fact_text': getattr(edge, 'fact', None) or str(edge),
                        'project_id': project_id,
                        'agent_id': payload.get('agent_id'),
                        'session_id': payload.get('session_id'),
                        '_causation_id': payload.get('_causation_id'),
                        'temporal_context': payload.get('temporal_context'),
                    },
                }
                for edge in edges
            ]

            assert self.durable_queue is not None
            await self.durable_queue.enqueue_batch(batch)

        refresh_group_id = payload.get('group_id') or payload.get('project_id') or 'main'
        await self._refresh_entity_summaries_from_result(result, group_id=refresh_group_id)

    async def _refresh_summaries_callback(
        self, callback_type: str, result: Any, payload: dict[str, Any]
    ) -> None:
        """Refresh-only post-ingestion callback: no Mem0 enqueue.

        Used by ingestion paths that already handle Mem0 elsewhere —
        ``add_memory`` writes Mem0 directly and ``replay_from_store`` only
        re-ingests into Graphiti — so, unlike ``_dual_write_callback``, this
        callback must not also batch-enqueue a ``mem0_classify_and_add`` for
        each edge (that would double-write Mem0). It only triggers the
        best-effort post-ingestion entity-summary refresh.
        """
        group_id = payload.get('group_id') or payload.get('project_id') or 'main'
        await self._refresh_entity_summaries_from_result(result, group_id=group_id)

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
        causation_id: str | None = None,
        temporal_context: str | None = None,
        _source: str = 'mcp_tool',
    ) -> AddEpisodeResponse:
        """Full ingestion pipeline — durably enqueue episode, return immediately."""
        scope = Scope(project_id=project_id, agent_id=agent_id, session_id=session_id)
        episode_id = str(uuid_mod.uuid4())
        write_op_id = str(uuid_mod.uuid4())

        # Parse source type name for storage
        try:
            source_name = EpisodeType[source.lower()].name
        except (KeyError, AttributeError):
            source_name = 'text'

        assert self.durable_queue is not None

        success = True
        error_msg = None
        try:
            await self.durable_queue.enqueue(
                group_id=scope.graphiti_group_id,
                operation='add_episode',
                payload={
                    'uuid': episode_id,
                    'name': f'episode_{episode_id[:8]}',
                    'content': content,
                    'source': source_name,
                    'group_id': scope.graphiti_group_id,
                    'source_description': source_description,
                    # Scope fields for callback reconstruction
                    'project_id': project_id,
                    'agent_id': agent_id,
                    'session_id': session_id,
                    # Journal metadata (popped by _execute_graphiti_write)
                    '_causation_id': causation_id,
                    '_write_op_id': write_op_id,
                    'temporal_context': temporal_context,
                    'reference_time': reference_time.isoformat() if reference_time is not None else None,
                },
                callback_type='dual_write_episode',
            )
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                await self._write_journal.log_write_op(
                    write_op_id=write_op_id,
                    causation_id=causation_id,
                    source=_source,
                    operation='add_episode',
                    project_id=project_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    params={'content': content[:200], 'source': source},
                    result_summary={'episode_id': episode_id, 'status': 'queued'} if success else None,
                    success=success,
                    error=error_msg,
                )

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.episode_added,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload={'episode_id': episode_id, 'content_preview': content[:200]},
            agent_id=agent_id,
        ))

        return AddEpisodeResponse(
            episode_id=episode_id,
            status=EpisodeStatus.queued,
            message=f'Episode queued for processing in project {project_id}',
        )

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
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> AddMemoryResponse:
        """Lightweight classified write — skip extraction pipeline."""
        scope = Scope(project_id=project_id, agent_id=agent_id, session_id=session_id)
        write_op_id = str(uuid_mod.uuid4())

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
        _mem0_error = None

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
                        '_causation_id': causation_id,
                        '_write_op_id': write_op_id,
                    },
                    callback_type='refresh_entity_summaries',
                )
                # Durably persisted to SQLite — report as written
                stores_written.append(SourceStore.graphiti)
            except Exception as e:
                logger.error(f'Graphiti enqueue failed: {e}')
                _graphiti_error = f'{type(e).__name__}: {e}'

        # Mem0: direct synchronous call so memory_ids are returned to the caller.
        # The durable-queue path cannot return server-assigned IDs because Mem0
        # assigns IDs server-side and the queue worker has no path back to the caller.
        # Durability is retained via write_journal (log_backend_op captures every call).
        # The _execute_durable_write 'mem0_add' dispatcher is kept intact for backward
        # compat — any in-flight queue items from before this fix still drain correctly.
        if write_mem0:
            try:
                mem0_result = await self._journaled_backend_call(
                    write_op_id=write_op_id,
                    causation_id=causation_id,
                    backend='mem0',
                    operation='add',
                    payload={'content': content[:200]},
                    coro=self.mem0.add(content=content, scope=scope, metadata=meta),
                )
                mem0_ids = [
                    r['id']
                    for r in (mem0_result or {}).get('results', [])
                    if isinstance(r, dict) and 'id' in r
                ]
                memory_ids.extend(mem0_ids)
                stores_written.append(SourceStore.mem0)
                if not mem0_ids and _MEM0_ADD_INFER_PINNED_FALSE:
                    # mem0.add() did not raise but returned no results — a silent
                    # dedup/infer no-op drop (task 1974). Under the pinned
                    # infer=False, a successful write always returns exactly one
                    # result with an id, so this is always anomalous.
                    logger.warning(
                        'MemoryService.add_memory: mem0 add returned zero memory_ids '
                        '(silent empty-result drop; possible dedup/infer no-op)',
                        extra={
                            'project_id': project_id,
                            'category': resolved_category.value,
                            'causation_id': causation_id,
                            'kind': meta.get('kind'),
                        },
                    )
            except Exception as e:
                logger.error(f'Mem0 write failed: {e}')
                _mem0_error = str(e)

        # Layer 1 journal entry
        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source=_source,
                operation='add_memory',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'content': content[:200], 'category': resolved_category.value},
                result_summary={
                    'memory_ids': memory_ids,
                    'stores': [s.value for s in stores_written],
                },
                success=not (_graphiti_error or _mem0_error),
                error=_graphiti_error or _mem0_error,
            )

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.memory_added,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload={
                'memory_ids': memory_ids,
                'category': resolved_category.value,
                'content_preview': content[:200],
            },
            agent_id=agent_id,
        ))

        msg = f'Memory queued for {[s.value for s in stores_written]}'
        if _graphiti_error:
            msg += f' [graphiti_error: {_graphiti_error}]'
        if _mem0_error:
            msg += f' [mem0_error: {_mem0_error}]'

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
        limit: int | None = None,
    ) -> int:
        """Fetch memories from Mem0 and enqueue each for Graphiti write.

        Args:
            limit: Max memories to replay. None = all (up to 1000).

        Returns the count of items queued.
        """
        target = target_project_id or source_project_id
        scope = Scope(project_id=source_project_id)
        fetch_limit = limit if limit else 1000
        all_mems = await self.mem0.get_all(scope, limit=fetch_limit)
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
                'callback_type': 'refresh_entity_summaries',
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
        causation_id: str | None = None,
        include_planned: bool = False,
    ) -> list[MemoryResult]:
        """Unified search across both stores with automatic fan-out.

        When include_planned=False (default), edges and memories from planning
        episodes (temporal_context='planning') are excluded.  Set include_planned=True
        to include them — useful for reconciliation and auditing.
        """
        scope = Scope(project_id=project_id, agent_id=agent_id, session_id=session_id)

        # Determine routing
        stores_override = [SourceStore(s) for s in stores] if stores else None
        route: ReadRouteResult = await self.router.route(query, stores_override)

        # Fan out to stores in parallel with timeout
        search_timeout = self.config.queue.search_timeout_seconds
        store_list: list[SourceStore] = []
        task_list: list[asyncio.Task] = []

        if SourceStore.graphiti in route.stores:
            store_list.append(SourceStore.graphiti)
            task_list.append(asyncio.create_task(
                self._search_graphiti(query, scope, limit, include_planned=include_planned)
            ))
        if SourceStore.mem0 in route.stores:
            store_list.append(SourceStore.mem0)
            task_list.append(asyncio.create_task(
                self._search_mem0(query, scope, limit, include_planned=include_planned, categories=categories)
            ))

        results: list[MemoryResult] = []
        failed_stores: list[SourceStore] = []
        if task_list:
            done, pending = await asyncio.wait(
                task_list, timeout=search_timeout, return_when=asyncio.ALL_COMPLETED
            )
            for t in pending:
                t.cancel()

            timed_out_stores = [
                store_list[i] for i, t in enumerate(task_list) if t in pending
            ]
            if timed_out_stores:
                logger.warning(
                    f'Search timed out for stores: {[s.value for s in timed_out_stores]}'
                )
            failed_stores.extend(timed_out_stores)

            for i, t in enumerate(task_list):
                if t not in done:
                    continue
                try:
                    store_results = t.result()
                    results.extend(store_results)
                except Exception as e:
                    logger.warning(
                        'search.store_failed',
                        extra={'store': store_list[i].value, 'error': str(e)},
                    )
                    failed_stores.append(store_list[i])

        # Sort: primary store results first, then by relevance score
        def sort_key(r: MemoryResult) -> tuple[int, float]:
            is_primary = 0 if r.source_store == route.primary_store else 1
            return (is_primary, -r.relevance_score)

        results.sort(key=sort_key)

        # Graphiti results: filter by category and infer category for ambiguous
        # nodes.  Mem0 results are already category-scoped server-side via the
        # pushdown in Mem0Backend.search (see task 1083), so this block is
        # redundant-but-harmless for mem0 and acts as defence-in-depth.
        if categories:
            cat_set = {MemoryCategory(c) for c in categories}
            graphiti_overlap = cat_set & GRAPHITI_PRIMARY
            results = [
                r for r in results
                if r.category in cat_set
                or (
                    r.source_store == SourceStore.graphiti
                    and r.category is None
                    and graphiti_overlap
                )
            ]
            # Assign inferred category to Graphiti results when unambiguous
            if len(graphiti_overlap) == 1:
                inferred = next(iter(graphiti_overlap))
                for r in results:
                    if r.source_store == SourceStore.graphiti and r.category is None:
                        r.category = inferred

        degraded = bool(failed_stores)
        final = results[:limit]

        # Log search when causation_id is present (recon paths)
        if causation_id and self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=str(uuid_mod.uuid4()),
                causation_id=causation_id,
                source='mcp_tool',
                operation='search',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                kind='read',
                params={'query': query[:200], 'limit': limit},
                result_summary={
                    'count': len(final),
                    'failed_stores': [s.value for s in failed_stores],
                },
                success=not degraded,
            )

        return SearchResults(
            final,
            degraded=degraded,
            failed_stores=[s.value for s in failed_stores],
        )

    async def _search_graphiti(
        self, query: str, scope: Scope, limit: int, include_planned: bool = False
    ) -> list[MemoryResult]:
        """Search Graphiti and convert results to MemoryResult.

        When include_planned=False (default), edges whose entire provenance is
        composed of planned-only episodes are excluded.  When include_planned=True,
        those edges are returned and marked with metadata['planned'] = True.
        """
        edges = await self.graphiti.search(
            query=query,
            group_ids=[scope.graphiti_group_id],
            num_results=int(limit * 1.5) + 1,
        )

        # Fetch planned UUIDs once (avoid per-edge DB hits).
        planned_uuids: set[str] = set()
        if self.planned_episode_registry is not None:
            planned_uuids = await self.planned_episode_registry.get_planned_uuids(
                scope.graphiti_group_id
            )

        results = []
        for i, edge in enumerate(edges):
            fact = getattr(edge, 'fact', str(edge))
            valid_at = getattr(edge, 'valid_at', None)
            invalid_at = getattr(edge, 'invalid_at', None)

            # Skip superseded edges (invalid_at set means the fact has been
            # replaced by a newer edge).  Check this before anything else to
            # avoid unnecessary work on edges that will be discarded.
            if invalid_at is not None:
                continue

            temporal = _serialize_temporal(valid_at, invalid_at)

            # Extract entity names from source/target nodes
            entities = []
            source_node = getattr(edge, 'source_node', None)
            target_node = getattr(edge, 'target_node', None)
            if source_node and hasattr(source_node, 'name'):
                entities.append(source_node.name)
            if target_node and hasattr(target_node, 'name'):
                entities.append(target_node.name)

            # Episode provenance
            episodes = getattr(edge, 'episodes', None) or []
            provenance = [str(ep) for ep in episodes]

            # Determine whether this edge is purely aspirational (all episodes planned).
            is_planned_edge = bool(provenance) and all(
                ep in planned_uuids for ep in provenance
            )

            if is_planned_edge and not include_planned:
                # Skip planning-only edges in normal search results.
                continue

            # Score: rank-based (no explicit score from Graphiti search)
            score = max(0.0, 1.0 - (i * 0.05))

            metadata: dict[str, Any] = {}
            if is_planned_edge:
                metadata['planned'] = True

            results.append(MemoryResult(
                id=getattr(edge, 'uuid', str(i)),
                content=fact,
                category=None,
                source_store=SourceStore.graphiti,
                relevance_score=score,
                provenance=provenance,
                temporal=temporal,
                entities=entities,
                metadata=metadata,
            ))
        # Truncate to the original limit (over-fetch may have produced extras).
        return results[:limit]

    async def _search_mem0(
        self,
        query: str,
        scope: Scope,
        limit: int,
        include_planned: bool = False,
        categories: list[str] | None = None,
    ) -> list[MemoryResult]:
        """Search Mem0 and convert results to MemoryResult.

        When include_planned=False (default), results tagged with planned=True
        in their metadata are excluded.  When include_planned=True they are returned.
        """
        # Forward categories so Mem0Backend pushes the filter down to Qdrant
        # (task 1083: prevents false-negatives caused by post-filtering on
        # an already-truncated top-N that excludes low-ranked matching memories).
        response = await self.mem0.search(query=query, scope=scope, limit=limit, categories=categories)
        mem0_results = response.get('results', [])
        results = []
        for item in mem0_results:
            content = item.get('memory', '')
            score = float(item.get('score', 0.0))
            meta = item.get('metadata', {}) or {}

            # Filter out planning-tagged results unless explicitly requested.
            if meta.get('planned') is True and not include_planned:
                continue

            category = None
            if 'category' in meta:
                with contextlib.suppress(ValueError):
                    category = MemoryCategory(meta['category'])

            results.append(MemoryResult(
                id=item.get('id', ''),
                content=content,
                category=category,
                source_store=SourceStore.mem0,
                relevance_score=min(score, 1.0),
                metadata=meta,
                created_at=item.get('created_at'),
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
        """Entity lookup in Graphiti — returns nodes + edges.

        Both Graphiti calls run concurrently via asyncio.gather(return_exceptions=True).
        This ensures neither call becomes an orphaned background task in the error path:
        gather() awaits both coroutines to settlement before returning, even when one
        (or both) raise an exception.  If any call fails, all exceptions are logged
        (warning) then the first exception is re-raised.
        """
        results = await asyncio.gather(
            self.graphiti.search_nodes(
                query=name,
                group_ids=[project_id],
                max_nodes=5,
            ),
            self.graphiti.search(
                query=name,
                group_ids=[project_id],
                num_results=10,
            ),
            return_exceptions=True,
        )

        # Two-tier check for asyncio.gather(return_exceptions=True) results.
        # Both coroutines have already settled at this point (no orphans).
        #
        # Pass 1: propagate_cancellations handles structured-cancellation signals
        #   (CancelledError, KeyboardInterrupt, SystemExit) before any per-call logging.
        #   Cancellation takes precedence over application-level failures regardless
        #   of position in the results list.
        #   See fused_memory.utils.async_utils.propagate_cancellations for the shared
        #   Pass 1 guard contract.
        #
        # Pass 2: log each captured Exception and raise the first — these are
        #   application-level failures from the Graphiti backend.
        propagate_cancellations(results)
        first_exc = next((r for r in results if isinstance(r, Exception)), None)
        if first_exc is not None:
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(
                        'get_entity: Graphiti call failed: %s: %s',
                        type(r).__name__,
                        r,
                    )
            raise first_exc

        nodes = cast(list, results[0])
        edges = cast(list, results[1])

        node_data = []
        for n in nodes:
            node_data.append({
                'uuid': getattr(n, 'uuid', None),
                'name': getattr(n, 'name', None),
                'summary': getattr(n, 'summary', None),
                'labels': getattr(n, 'labels', None) or [],
            })

        edge_data = []
        for e in edges:
            edge_data.append({
                'uuid': getattr(e, 'uuid', None),
                'fact': getattr(e, 'fact', str(e)),
                'temporal': _serialize_temporal(
                    getattr(e, 'valid_at', None),
                    getattr(e, 'invalid_at', None),
                ),
            })

        return {'nodes': node_data, 'edges': edge_data}

    # ------------------------------------------------------------------
    # Read: get_edge (thin wrapper for cite_edge — task β)
    # ------------------------------------------------------------------

    async def get_edge(self, edge_uuid: str, project_id: str) -> dict:
        """Fetch a single edge by UUID from Graphiti.

        Returns {uuid, name, fact} on success.  Raises EdgeNotFoundError on miss.
        group_id is mapped from project_id to match the Graphiti storage convention.
        """
        name, fact = await self.graphiti.get_edge_text(edge_uuid, group_id=project_id)
        return {'uuid': edge_uuid, 'name': name, 'fact': fact}

    # ------------------------------------------------------------------
    # Read: get_memory (thin dispatcher for cite_memory — task β)
    # ------------------------------------------------------------------

    async def get_memory(self, memory_id: str, store: str, project_id: str) -> dict:
        """Fetch a memory by id from either Graphiti or Mem0.

        Returns a minimal metadata fingerprint dict:
          {category, agent_id, created_at} for mem0;
          {name, fact_snippet} for graphiti.
        Raises EdgeNotFoundError (graphiti) or ValueError (mem0 not found).
        """
        if store == 'graphiti':
            name, fact = await self.graphiti.get_edge_text(memory_id, group_id=project_id)
            return {'name': name, 'fact_snippet': fact[:120]}

        # mem0 path
        from fused_memory.models.scope import Scope

        scope = Scope(project_id=project_id)
        rec = await self.mem0.get(memory_id, scope)
        if rec is None:
            raise MemoryNotFoundError(memory_id)
        metadata = rec.get('metadata') or {}
        return {
            'category': rec.get('category'),
            'agent_id': metadata.get('agent_id'),
            'created_at': rec.get('created_at'),
        }

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
                'created_at': str(_ca) if (_ca := getattr(ep, 'created_at', None)) is not None else None,
                'source': getattr(ep, 'source', None),
                'group_id': getattr(ep, 'group_id', None),
            }
            for ep in episodes
        ]

    # ------------------------------------------------------------------
    # Read: deterministic metadata count
    # ------------------------------------------------------------------

    async def count_memories_by_metadata(
        self,
        project_id: str,
        filters: dict,
    ) -> int:
        """Deterministic Mem0 count for the given metadata equality filters.

        Use this when you need a reliable key-equality lookup instead of
        semantic search — e.g. counting persistence markers or escalation
        markers keyed by ``flag_id``.  Goes through ``Mem0Backend.count_by_metadata``
        which talks to Qdrant's count API directly with a payload filter, so
        the result is exact rather than top-N-bounded.
        """
        scope = Scope(project_id=project_id)
        return await self.mem0.count_by_metadata(scope, filters)

    async def get_memories_by_metadata(
        self,
        project_id: str,
        filters: dict,
        limit: int = 1000,
    ) -> list[dict]:
        """Deterministic (non-semantic) enumeration of Mem0 memories matching *filters*.

        Counterpart to ``count_memories_by_metadata``: where that returns only
        an int, this returns the full list of matching memory dicts so callers
        can inspect IDs, timestamps, and metadata for GC or pool-cap enforcement.

        Goes through ``Mem0Backend.scroll_by_metadata`` which talks to Qdrant's
        scroll API directly with a payload filter — NOT semantic search.  Using
        semantic search for pool GC is the failure mode that caused the
        stage2_cycle_summary pool to grow unboundedly (tasks 20e8c2f1, 45489c2b,
        db2ea69e).

        Returns a list of ``{'id', 'created_at', 'metadata'}`` dicts.
        """
        scope = Scope(project_id=project_id)
        return await self.mem0.scroll_by_metadata(scope, filters, limit)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_memory(
        self,
        memory_id: str,
        store: str,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Delete a memory from the specified store."""
        scope = Scope(project_id=project_id)
        source = SourceStore(store)
        write_op_id = str(uuid_mod.uuid4())

        if source == SourceStore.graphiti:
            await self._journaled_backend_call(
                write_op_id=write_op_id,
                causation_id=causation_id,
                backend='graphiti',
                operation='remove_edge',
                payload={'memory_id': memory_id},
                coro=self.graphiti.remove_edge(memory_id, group_id=project_id),
            )
            result = {'status': 'deleted', 'store': 'graphiti', 'id': memory_id}
        else:
            del_result = await self._journaled_backend_call(
                write_op_id=write_op_id,
                causation_id=causation_id,
                backend='mem0',
                operation='delete',
                payload={'memory_id': memory_id},
                coro=self.mem0.delete(memory_id, scope),
            )
            result = {'status': 'deleted', 'store': 'mem0', 'id': memory_id, **del_result}

        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source=_source,
                operation='delete_memory',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'memory_id': memory_id, 'store': store},
                result_summary=result,
                success=True,
            )

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.memory_deleted,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload={'memory_id': memory_id, 'store': store},
        ))

        return result

    async def update_edge(
        self,
        edge_uuid: str,
        fact: str | None = None,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
        invalid_at: datetime | None = None,
        clear_invalid_at: bool = False,
    ) -> dict:
        """Update an existing Graphiti edge's fact text and/or invalidate it.

        At least one of ``fact``, ``invalid_at``, or ``clear_invalid_at`` must
        be supplied. Setting ``invalid_at`` marks the edge as superseded as of
        that moment (used by reconciliation to retire contradicted facts without
        destroying their audit trail). Setting ``clear_invalid_at=True`` resets
        ``invalid_at`` to ``None``, restoring the edge to active status.

        **Guard-2 verification (TOCTOU note):** when *fact* is supplied, a
        ``get_edge_text`` readback is performed after the save to confirm
        persistence. When *clear_invalid_at* is supplied, a separate
        ``get_edge_invalid_at`` readback confirms the edge was actually
        restored to active status — this is independent of the fact
        verdict, so a combined ``fact`` + ``clear_invalid_at`` call is
        ``verified`` only when *both* readbacks confirm. A pure
        ``invalid_at=<timestamp>`` supersede (no fact, no clear) is not
        readback-verified — it persists reliably via the same write path
        used for ``fact``. There is a small TOCTOU window between each save
        and its readback — if a concurrent writer updates the same edge in
        that window (another reconciliation cycle, an interactive agent, or
        a retry) the readback may return a different value and ``verified``
        will be ``False`` even though *our* save did persist.  This is a
        known false-negative mode; reconciliation is mostly single-writer
        per edge so it is rare in practice.  Undercounting ``edges_updated``
        (the conservative outcome) is safer than overcounting, so the
        behaviour is intentional.  Both readbacks (``get_edge_text`` and
        ``get_edge_invalid_at``) resolve their Cypher through the same
        per-``group_id`` graph object (``GraphitiBackend._graph_for``) that
        the backend's explicit ``clear_invalid_at`` write uses, so
        read-after-write consistency for *our own* write is expected within
        that connection — the TOCTOU risk above is about a *different*
        writer racing us, not about our own write lagging its readback.
        """
        if fact is None and invalid_at is None and not clear_invalid_at:
            raise ValueError('update_edge requires fact, invalid_at, or clear_invalid_at to be set')
        write_op_id = str(uuid_mod.uuid4())

        params: dict[str, Any] = {'edge_uuid': edge_uuid}
        if fact is not None:
            # Truncated copy for journal/payload logging only.  The full fact
            # is passed to graphiti.update_edge and used for the Guard-2
            # readback comparison.  A verified=False journal entry will show
            # params['fact'] (truncated) as the 'expected' value — the actual
            # equality check was against the un-truncated string.
            params['fact'] = fact[:200]
        if invalid_at is not None:
            params['invalid_at'] = invalid_at.isoformat()
        if clear_invalid_at:
            params['clear_invalid_at'] = True

        result_data = await self._journaled_backend_call(
            write_op_id=write_op_id,
            causation_id=causation_id,
            backend='graphiti',
            operation='update_edge',
            payload=params,
            coro=self.graphiti.update_edge(
                edge_uuid, fact, group_id=project_id, invalid_at=invalid_at,
                clear_invalid_at=clear_invalid_at,
            ),
        )
        result = {'status': 'updated', 'store': 'graphiti', **result_data}

        # Guard 2: post-write persistence verification.
        # Only when fact text was supplied — invalid_at-only updates have nothing to compare.
        if fact is not None:
            try:
                _name, readback_fact = await self.graphiti.get_edge_text(
                    edge_uuid, group_id=project_id
                )
                result['verified'] = readback_fact == fact
                if not result['verified']:
                    result['verification_error'] = (
                        f'Readback fact mismatch: expected {fact!r}, got {readback_fact!r}'
                    )
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                # Log unexpected exceptions (e.g. AttributeError from a future
                # signature change) so they remain observable in logs even
                # though we don't re-raise — the save succeeded and must be
                # reported; only the verification signal is lost.
                logger.exception(
                    'update_edge verification readback failed for edge %s',
                    edge_uuid,
                    exc_info=e,
                )
                result['verified'] = False
                result['verification_error'] = f'{type(e).__name__}: {e}'
        else:
            # No fact supplied — nothing to compare yet. May be revised below
            # by the clear_invalid_at readback; a pure invalid_at=<timestamp>
            # supersede (no fact, no clear) stays trivially verified here.
            result['verified'] = True

        # Guard 2b: post-write clear_invalid_at verification, independent of
        # the fact verdict above. Combines via AND — a fact+clear call is
        # verified only when both readbacks confirm.
        if clear_invalid_at:
            try:
                readback_invalid_at = await self.graphiti.get_edge_invalid_at(
                    edge_uuid, group_id=project_id
                )
                cleared = readback_invalid_at is None
                result['verified'] = bool(result.get('verified', True)) and cleared
                if not cleared:
                    error_msg = f'invalid_at not cleared: readback={readback_invalid_at!r}'
                    existing_error = result.get('verification_error')
                    result['verification_error'] = (
                        f'{existing_error}; {error_msg}' if existing_error else error_msg
                    )
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.exception(
                    'update_edge clear_invalid_at verification readback failed for edge %s',
                    edge_uuid,
                    exc_info=e,
                )
                result['verified'] = False
                error_msg = f'{type(e).__name__}: {e}'
                existing_error = result.get('verification_error')
                result['verification_error'] = (
                    f'{existing_error}; {error_msg}' if existing_error else error_msg
                )

        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source=_source,
                operation='update_edge',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params=params,
                result_summary=result,
                success=True,
            )

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.memory_updated,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload={'edge_uuid': edge_uuid, 'store': 'graphiti'},
        ))

        return result

    async def delete_episode(
        self,
        episode_id: str,
        project_id: str = 'main',
        cascade: bool = True,
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Delete a Graphiti episode."""
        write_op_id = str(uuid_mod.uuid4())

        await self._journaled_backend_call(
            write_op_id=write_op_id,
            causation_id=causation_id,
            backend='graphiti',
            operation='remove_episode',
            payload={'episode_id': episode_id},
            coro=self.graphiti.remove_episode(episode_id, group_id=project_id),
        )

        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source=_source,
                operation='delete_episode',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'episode_id': episode_id, 'cascade': cascade},
                result_summary={'status': 'deleted'},
                success=True,
            )

        return {'status': 'deleted', 'episode_id': episode_id, 'cascade': cascade}

    async def refresh_entity_summary(
        self,
        entity_uuid: str | None = None,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
        entity_name: str | None = None,
    ) -> dict:
        """Regenerate a Graphiti entity node's summary from its valid edges.

        Accepts either *entity_uuid* (canonical identifier) or *entity_name*
        (resolved via an exact name lookup).  When both are supplied, entity_uuid
        takes precedence.  Raises ValueError if neither is provided.

        Delegates to GraphitiBackend.refresh_entity_summary(), which queries
        remaining valid edges, deduplicates their facts, and writes back a
        clean summary. Logs the operation via write journal if available.

        Args:
            entity_uuid: UUID of the Entity node to refresh (optional when entity_name is given).
            entity_name: Exact entity name to resolve to a UUID (optional when entity_uuid is given).
            project_id: Project scope (for journal logging).
            agent_id: Which agent is calling (optional).
            session_id: Session context (optional).
            causation_id: Reconciliation causation ID (optional).
            _source: Source label for journal entry.

        Returns:
            Dict from backend: {uuid, name, old_summary, new_summary, edge_count}.

        Raises:
            ValueError: if neither entity_uuid nor entity_name is provided.
        """
        if entity_uuid is None and entity_name is None:
            raise ValueError('Either entity_uuid or entity_name must be provided')

        # Resolve entity_name → UUID when UUID is not directly supplied
        if entity_uuid is None:
            assert entity_name is not None  # guaranteed by the ValueError check above
            entity_uuid = await self.graphiti.resolve_entity_by_name(
                entity_name, group_id=project_id
            )

        write_op_id = str(uuid_mod.uuid4())
        success = True
        error_msg = None
        result: dict = {}
        journal_params: dict = {'entity_uuid': entity_uuid}
        if entity_name is not None:
            journal_params['entity_name'] = entity_name
        try:
            result = await self.graphiti.refresh_entity_summary(entity_uuid, group_id=project_id)
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                try:
                    await self._write_journal.log_write_op(
                        write_op_id=write_op_id,
                        causation_id=causation_id,
                        source=_source,
                        operation='refresh_entity_summary',
                        project_id=project_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        params=journal_params,
                        result_summary=result if success else None,
                        success=success,
                        error=error_msg,
                    )
                except Exception as journal_exc:
                    logger.warning(
                        'refresh_entity_summary: journal log_write_op failed: %s',
                        journal_exc,
                    )

        return result

    async def rebuild_entity_summaries(
        self,
        project_id: str = 'main',
        force: bool = False,
        dry_run: bool = False,
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Batch-rebuild Entity node summaries from their current valid edges.

        Orchestrates the rebuild pipeline:
        1. Target selection — detect_stale_with_edges (force=False) or
           list_entity_nodes (force=True).
        2. Fan-out — asyncio.Semaphore(20) + asyncio.gather(return_exceptions=True)
           calling graphiti.rebuild_entity_from_edges for each target.
        3. Error accumulation — two-tier: propagate_cancellations (Pass 1) then
           per-entity dict accumulation (Pass 2).

        Logs the operation via write journal if available.

        Args:
            project_id: Project scope (determines FalkorDB graph).
            force: Rebuild every entity regardless of staleness.
            dry_run: Detect stale entities but do not write any summaries.
            agent_id: Which agent is calling (optional).
            session_id: Session context (optional).
            causation_id: Reconciliation causation ID (optional).
            _source: Source label for journal entry.

        Returns:
            Dict with keys: total_entities, stale_entities, rebuilt, skipped,
            errors, details.
        """
        write_op_id = str(uuid_mod.uuid4())
        success = True
        error_msg = None
        result: dict = {}
        try:
            # --- Target selection ---
            targets: list[dict] = []
            all_edges: dict[str, list] = {}
            total_entities: int = 0

            if force:
                all_entities = await self.graphiti.list_entity_nodes(group_id=project_id)
                targets = [
                    {'uuid': e['uuid'], 'name': e['name'], 'old_summary': e['summary']}
                    for e in all_entities
                ]
                total_entities = len(all_entities)
                if not dry_run:
                    all_edges = await self.graphiti.get_all_valid_edges(group_id=project_id)
            else:
                if dry_run:
                    stale, total_entities = await self.graphiti.detect_stale_dry_run(
                        group_id=project_id
                    )
                else:
                    detect_result = await self.graphiti.detect_stale_with_edges(
                        group_id=project_id
                    )
                    stale = detect_result.stale
                    all_edges = detect_result.all_edges
                    total_entities = detect_result.total_count
                targets = [
                    {'uuid': s['uuid'], 'name': s['name'], 'old_summary': s['summary']}
                    for s in stale
                ]

            stale_entities = len(targets)
            rebuilt = 0
            skipped = 0
            errors = 0
            details: list[dict] = []

            if dry_run:
                skipped = stale_entities
                for t in targets:
                    details.append({
                        'uuid': t['uuid'],
                        'name': t['name'],
                        'status': 'skipped_dry_run',
                    })
            else:
                sem = asyncio.Semaphore(20)

                async def _rebuild_one(t: dict) -> dict:
                    async with sem:
                        edges = all_edges.get(t['uuid'], [])
                        return await self.graphiti.rebuild_entity_from_edges(
                            t['uuid'], t['name'], edges,
                            group_id=project_id,
                            old_summary=t['old_summary'],
                        )

                gather_results = await asyncio.gather(
                    *(_rebuild_one(t) for t in targets), return_exceptions=True
                )

                # Pass 1: propagate CancelledError before per-entity accumulation.
                try:
                    propagate_cancellations(gather_results)
                except BaseException as e:
                    if not isinstance(e, Exception):
                        logger.warning(
                            'rebuild_entity_summaries: cancellation signal received '
                            'group=%s rebuilt_so_far=%d errors_so_far=%d; propagating',
                            project_id, rebuilt, errors,
                        )
                    raise

                # Pass 2: per-entity accumulation.
                for t, r in zip(targets, gather_results, strict=True):
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
                                f'rebuild_entity_summaries: rebuild_entity_from_edges returned '
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
                project_id, total_entities, stale_entities, rebuilt, skipped, errors,
                dry_run, force,
            )
            result = {
                'total_entities': total_entities,
                'stale_entities': stale_entities,
                'rebuilt': rebuilt,
                'skipped': skipped,
                'errors': errors,
                'details': details,
            }
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                try:
                    await self._write_journal.log_write_op(
                        write_op_id=write_op_id,
                        causation_id=causation_id,
                        source=_source,
                        operation='rebuild_entity_summaries',
                        project_id=project_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        params={'force': force, 'dry_run': dry_run},
                        result_summary={
                            'total_entities': result.get('total_entities', 0),
                            'stale_entities': result.get('stale_entities', 0),
                            'rebuilt': result.get('rebuilt', 0),
                            'skipped': result.get('skipped', 0),
                            'errors': result.get('errors', 0),
                        } if success else None,
                        success=success,
                        error=error_msg,
                    )
                except Exception as journal_exc:
                    logger.warning(
                        'rebuild_entity_summaries: journal log_write_op failed: %s',
                        journal_exc,
                    )

        return result

    async def merge_entities(
        self,
        deprecated_uuid: str,
        surviving_uuid: str,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Merge two Graphiti entity nodes by redirecting edges and deleting the deprecated.

        Delegates to GraphitiBackend.merge_entities(), which validates both nodes,
        redirects all edges from the deprecated node to the surviving node, deletes
        the deprecated node, and refreshes the surviving node's summary.
        Logs the operation via write journal if available.

        Args:
            deprecated_uuid: UUID of the entity node to be deleted.
            surviving_uuid: UUID of the entity node that absorbs the edges.
            project_id: Project scope (for journal logging).
            agent_id: Which agent is calling (optional).
            session_id: Session context (optional).
            causation_id: Reconciliation causation ID (optional).
            _source: Source label for journal entry.

        Returns:
            Audit dict from backend: {surviving_uuid, surviving_name, deprecated_uuid,
            deprecated_name, edges_redirected, surviving_summary}.
        """
        write_op_id = str(uuid_mod.uuid4())
        success = True
        error_msg = None
        result: dict = {}
        try:
            result = await self.graphiti.merge_entities(deprecated_uuid, surviving_uuid, group_id=project_id)
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                try:
                    await self._write_journal.log_write_op(
                        write_op_id=write_op_id,
                        causation_id=causation_id,
                        source=_source,
                        operation='merge_entities',
                        project_id=project_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        params={
                            'deprecated_uuid': deprecated_uuid,
                            'surviving_uuid': surviving_uuid,
                        },
                        result_summary=result if success else None,
                        success=success,
                        error=error_msg,
                    )
                except Exception as journal_exc:
                    logger.warning(
                        'merge_entities: journal log_write_op failed: %s',
                        journal_exc,
                    )

        return result

    async def delete_entity(
        self,
        entity_uuid: str,
        project_id: str = 'main',
        force: bool = False,
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Delete a Graphiti entity node by UUID, refreshing connected neighbours.

        Delegates to GraphitiBackend.delete_entity(), which validates the node
        exists, guards against accidental deletion of nodes with active edges
        (unless force=True), collects neighbours before deletion, performs
        DETACH DELETE, then refreshes each neighbour's summary.
        Logs the operation via write journal if available.

        Args:
            entity_uuid: UUID of the entity node to delete.
            project_id: Project scope (for Graphiti group_id and journal logging).
            force: When True, bypass the active-edges guard.
            agent_id: Which agent is calling (optional).
            session_id: Session context (optional).
            causation_id: Reconciliation causation ID (optional).
            _source: Source label for journal entry.

        Returns:
            Audit dict from backend: {deleted_uuid, deleted_name, active_edge_count,
            forced, connected_refreshed}.

        Raises:
            NodeNotFoundError: if the entity does not exist.
            ActiveEdgesError: if the node has valid active edges and force=False.
        """
        write_op_id = str(uuid_mod.uuid4())
        success = True
        error_msg = None
        result: dict = {}
        try:
            result = await self.graphiti.delete_entity(entity_uuid, group_id=project_id, force=force)
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                try:
                    await self._write_journal.log_write_op(
                        write_op_id=write_op_id,
                        causation_id=causation_id,
                        source=_source,
                        operation='delete_entity',
                        project_id=project_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        params={
                            'entity_uuid': entity_uuid,
                            'force': force,
                        },
                        result_summary=result if success else None,
                        success=success,
                        error=error_msg,
                    )
                except Exception as journal_exc:
                    logger.warning(
                        'delete_entity: journal log_write_op failed: %s',
                        journal_exc,
                    )

        return result

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    async def get_status(self, project_id: str | None = None) -> dict:
        """Health check and per-project statistics for both backends."""
        status: dict[str, Any] = {}

        # Graphiti connectivity + project discovery
        graphiti_counts: dict[str, int] = {}
        try:
            graphs = await self.graphiti.list_graphs()
            for graph_name in graphs:
                try:
                    graphiti_counts[graph_name] = await self.graphiti.node_count(graph_name)
                except Exception:
                    graphiti_counts[graph_name] = -1
            status['graphiti'] = {'connected': True}
        except Exception as e:
            status['graphiti'] = {'connected': False, 'error': str(e)}

        # Mem0 connectivity + project discovery
        mem0_counts: dict[str, int] = {}
        try:
            mem0_projects = await self.mem0.list_projects()
            for pid, _collection_name in mem0_projects:
                try:
                    scope = Scope(project_id=pid)
                    mem0_counts[pid] = await self.mem0.count(scope)
                except Exception:
                    mem0_counts[pid] = -1
            status['mem0'] = {'connected': True}
        except Exception as e:
            status['mem0'] = {'connected': False, 'error': str(e)}

        # Merge into per-project dict
        all_project_ids = sorted(set(graphiti_counts) | set(mem0_counts))
        if project_id:
            all_project_ids = [p for p in all_project_ids if p == project_id]
        projects: dict[str, dict] = {}
        for pid in all_project_ids:
            projects[pid] = {
                'graphiti_nodes': graphiti_counts.get(pid, 0),
                'mem0_memories': mem0_counts.get(pid, 0),
            }
        status['projects'] = projects

        # Queue stats — scoped to project_id when given so dead-letter counts
        # reflect only this project's rows, not the global write_queue table.
        if self.durable_queue:
            try:
                status['queue'] = await self.durable_queue.get_stats(group_id=project_id)
            except Exception as e:
                status['queue'] = {'error': str(e)}

        # Server uptime fields (informational; dashboard reads uptime_seconds as primary)
        status['started_at'] = self._started_at.isoformat()
        status['uptime_seconds'] = int(time.monotonic() - self._start_monotonic)

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
