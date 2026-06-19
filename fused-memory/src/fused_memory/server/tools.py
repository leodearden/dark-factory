"""FastMCP tool definitions for the Fused Memory server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid as uuid_mod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
from mcp.server.fastmcp import Context, FastMCP
from shared.async_sqlite_base import CheckpointResult, apply_full_durability_pragmas, connect_daemon

from fused_memory.mcp_tools.scheduler_state import (
    read_scheduler_events,
    read_scheduler_state,
)
from fused_memory.middleware.task_interceptor import (
    TERMINAL_STATUSES,
    _is_ticket_id,
    _looks_like_task_id,
)
from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.scope import resolve_main_checkout, resolve_project_id
from fused_memory.reconciliation.task_filter import ACTIVE_TASK_STATUSES, is_count_snapshot
from fused_memory.services.memory_service import MemoryService
from fused_memory.utils.validation import (
    validate_int_ids,
    validate_known_project_id,
    validate_project_id,
    validate_project_root,
)

if TYPE_CHECKING:
    from shared.usage_gate import UsageGate

    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.reconciliation.backlog_policy import BacklogPolicy
    from fused_memory.reconciliation.event_queue import EventQueue
    from fused_memory.reconciliation.harness import ReconciliationHarness
    from fused_memory.services.write_journal import WriteJournal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scheduler-override constants
#
# These are intentional duplicates of orchestrator.overrides._SCHEMA and
# orchestrator.config.PRIORITY_RANK.  fused-memory has no orchestrator
# dependency in pyproject.toml; adding one would invert the dependency graph.
# (TERMINAL_STATUSES and ACTIVE_TASK_STATUSES are imported directly from their
# canonical homes rather than duplicated here — see _VALID_TASK_STATUSES below.)
# ---------------------------------------------------------------------------

_OVERRIDE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS overrides (
    project_root  TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    boost_tier    TEXT,
    pinned        INTEGER NOT NULL DEFAULT 0,
    pin_order     INTEGER,
    reserve_now   INTEGER NOT NULL DEFAULT 0,
    ttl_until     TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (project_root, task_id)
);

CREATE INDEX IF NOT EXISTS idx_overrides_pinned
    ON overrides(project_root, pinned, pin_order);
"""  # Source of truth: orchestrator/src/orchestrator/overrides.py:33-49

# Mirrors orchestrator/src/orchestrator/config.py PRIORITY_RANK keys.
# Source of truth: orchestrator/src/orchestrator/config.py:27.
_PRIORITY_TIERS: tuple[str, ...] = ('critical', 'high', 'medium', 'low', 'polish')

# Valid field names for clear_task_priority_override.
# Mirrors orchestrator.overrides.OverrideStore.clear_override validation at
# orchestrator/src/orchestrator/overrides.py:258.
_VALID_CLEAR_FIELDS: frozenset[str] = frozenset({'boost_tier', 'pinned', 'reserve_now', 'ttl'})


def _overrides_db_path(project_root: str) -> Path:
    """Return the canonical path to ``scheduler_overrides.db`` for *project_root*.

    Single source of truth for ``<root>/data/orchestrator/scheduler_overrides.db``
    so that :func:`_open_overrides_db`, :func:`_connect_overrides_db`,
    :func:`_checkpoint_overrides_db`, and the existence guard
    :func:`_checkpoint_overrides_db_if_exists` can never drift from each other.
    """
    return Path(project_root) / 'data' / 'orchestrator' / 'scheduler_overrides.db'


async def _open_overrides_db(
    project_root: str,
    *,
    autocommit: bool = False,
) -> aiosqlite.Connection:
    """Open (and initialise) the scheduler_overrides.db for project_root.

    Creates parent directories on first call, runs idempotent DDL, and
    applies the full Phase-3 durability pragma triad (journal_mode=WAL,
    busy_timeout, synchronous=FULL, wal_autocheckpoint=100,
    journal_size_limit=64MiB) via ``apply_full_durability_pragmas``.
    See ``docs/task-recovery-2026-05-13/`` for the production incident that
    mandated this convention across all fused-memory SQLite stores.

    When ``autocommit=True`` the connection is opened with
    ``isolation_level=None`` and ``timeout=30`` so callers can use explicit
    ``BEGIN IMMEDIATE`` / ``COMMIT`` / ``ROLLBACK`` to serialize
    read-then-write sequences against concurrent writers.  Mirrors the
    source-of-truth concurrency contract at
    orchestrator/src/orchestrator/overrides.py:177-195 which documents
    why ``set_override`` MUST use this pattern.
    """
    db_path = _overrides_db_path(project_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if autocommit:
        db = await connect_daemon(str(db_path), timeout=30, isolation_level=None)
    else:
        db = await connect_daemon(str(db_path))
    # busy_timeout=30000ms when autocommit so BEGIN IMMEDIATE will wait up
    # to 30s for the write lock (matches source-of-truth timeout=30 above).
    busy_ms = 30000 if autocommit else 5000
    await apply_full_durability_pragmas(db, busy_timeout_ms=busy_ms)
    await db.executescript(_OVERRIDE_SCHEMA)
    return db


async def _connect_overrides_db(
    project_root: str,
    *,
    autocommit: bool = False,
) -> aiosqlite.Connection:
    """Open scheduler_overrides.db with the full durability pragma triad — no DDL.

    Thinner sibling to :func:`_open_overrides_db`.  Applies the same five
    Phase-3 pragmas (via ``apply_full_durability_pragmas``) but skips the
    ``_OVERRIDE_SCHEMA`` DDL step.  Use when the schema is guaranteed to
    exist already (e.g. :func:`_checkpoint_overrides_db`, which only needs a
    live connection with correct pragmas — re-running DDL on every checkpoint
    tick is wasted IO and can produce spurious write-lock contention).

    When ``autocommit=True`` the connection is opened with
    ``isolation_level=None`` and ``timeout=30``, matching the semantics of
    :func:`_open_overrides_db`.
    """
    db_path = _overrides_db_path(project_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if autocommit:
        db = await connect_daemon(str(db_path), timeout=30, isolation_level=None)
    else:
        db = await connect_daemon(str(db_path))
    busy_ms = 30000 if autocommit else 5000
    await apply_full_durability_pragmas(db, busy_timeout_ms=busy_ms)
    return db


async def _checkpoint_overrides_db(project_root: str) -> CheckpointResult:
    """Run ``PRAGMA wal_checkpoint(TRUNCATE)`` on the scheduler_overrides.db.

    Opens a fresh connection via :func:`_connect_overrides_db` (no DDL —
    schema must already exist), executes a TRUNCATE checkpoint, parses the
    result row into a ``CheckpointResult`` named-tuple, and closes the
    connection.

    Using :func:`_connect_overrides_db` rather than :func:`_open_overrides_db`
    avoids re-running ``_OVERRIDE_SCHEMA`` DDL on every checkpoint tick, which
    would be wasted IO and could produce spurious write-lock contention in a
    periodic-checkpoint loop.

    Callers drive this helper to bound WAL growth on the override DB.  It is
    invoked from the periodic checkpoint loop via the existence-guarded wrapper
    :func:`_checkpoint_overrides_db_if_exists`, which is registered per known
    project in ``server.main._collect_checkpoint_targets`` and driven by
    ``_periodic_checkpoint_loop`` on the standard ``_CHECKPOINT_INTERVAL`` cadence.

    Returns:
        ``CheckpointResult(busy, log, checkpointed)`` — same shape as
        ``AsyncSqliteBase.checkpoint()`` so future wiring slots in with no
        adapter.  ``busy == 0`` means all WAL frames were checkpointed.

    Raises:
        RuntimeError: if ``PRAGMA wal_checkpoint`` returns no rows (should not
            happen in normal operation; mirrors ``AsyncSqliteBase.checkpoint``
            at ``shared/src/shared/async_sqlite_base.py:199-200``).
    """
    db = await _connect_overrides_db(project_root)
    try:
        async with db.execute('PRAGMA wal_checkpoint(TRUNCATE)') as cur:
            row = await cur.fetchone()
        if row is None:
            raise RuntimeError('PRAGMA wal_checkpoint returned no rows')
        return CheckpointResult(int(row[0]), int(row[1]), int(row[2]))
    finally:
        await db.close()


async def _checkpoint_overrides_db_if_exists(project_root: str) -> CheckpointResult:
    """Checkpoint the override DB only if it already exists on disk.

    Returns ``CheckpointResult(0, 0, 0)`` immediately when
    ``scheduler_overrides.db`` is absent — preventing the periodic checkpoint
    loop from creating empty DB files (+ WAL + SHM side-cars) for projects that
    have never set an override.  When the file exists, delegates to
    :func:`_checkpoint_overrides_db`.

    The existence check is re-evaluated on every call (every
    ``_CHECKPOINT_INTERVAL`` cycle) so a DB created after startup is picked up
    on the next tick.  The TOCTOU window between ``exists()`` and
    ``_connect_overrides_db`` is benign: a racing create yields a harmless
    empty-DB checkpoint.

    Returns:
        ``CheckpointResult(busy, log, checkpointed)`` — ``(0, 0, 0)`` if absent;
        otherwise the result of :func:`_checkpoint_overrides_db`.
    """
    if not _overrides_db_path(project_root).exists():
        return CheckpointResult(0, 0, 0)
    return await _checkpoint_overrides_db(project_root)


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
- get_statuses: Compact {id: status} mapping (~95% smaller than get_tasks) for status-only callers
- search_tasks: Semantic search over already-filed tasks (ranked by similarity, enriched with current status) — use to check if a task like X was already filed
- set_task_status: Update status (triggers reconciliation for done/blocked/cancelled)
- update_task / remove_task: Task CRUD
- add_dependency / remove_dependency: Dependency management
Management:
- delete_memory: Remove a specific memory (edges for Graphiti, vector entries for Mem0)
- delete_episode: Remove a Graphiti episode (with optional cascade)
- update_edge: Update an existing Graphiti edge's fact text directly (no LLM pipeline)
- refresh_entity_summary: Rebuild an entity node's summary from its valid edges (accepts entity_uuid or entity_name)
- merge_entities: Consolidate two duplicate entity nodes (redirects edges, deletes deprecated)
- delete_entity: Delete an entity node by UUID (DETACH DELETE; guards on active edges unless force=True; refreshes neighbour summaries)
- get_status: Health check for all backends
- get_dead_letters: Inspect dead-lettered items from the durable write queue and event queue
- replay_dead_letters: Reset dead-lettered queue items to pending for retry (use for retriable transient failures)
- delete_dead_letters: Permanently delete dead-lettered items by id (use for non-retriable errors such as NodeNotFoundError after a graph wipe)

Reconciliation:
- Task status transitions (done/blocked/cancelled/deferred) trigger targeted reconciliation
  automatically — memory_hints may be attached, completion knowledge written, dependent tasks flagged.
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


def _summarise_ticket_row(row: dict) -> dict:
    """Project a ticket row into a triage-friendly summary.

    Pulls a candidate title out of the stored ``candidate_json`` blob
    (kwargs.title / kwargs.prompt fallback chain mirrors the path-guard
    rejection helper in :mod:`task_interceptor`). Drops ``result_json`` —
    the full payload is large and verbose; callers can use ``get_ticket``
    when they want the raw row.
    """
    title = '<unknown>'
    candidate_json = row.get('candidate_json')
    if candidate_json:
        try:
            blob = json.loads(candidate_json)
        except (TypeError, ValueError):
            blob = {}
        kwargs = blob.get('kwargs') if isinstance(blob, dict) else None
        if isinstance(kwargs, dict):
            raw = kwargs.get('title') or kwargs.get('prompt') or '<unknown>'
            title = str(raw)[:200]
    return {
        'ticket_id': row['ticket_id'],
        'status': row['status'],
        'reason': row.get('reason'),
        'task_id': row.get('task_id'),
        'candidate_title': title,
        'created_at': row['created_at'],
        'expires_at': row['expires_at'],
        'resolved_at': row.get('resolved_at'),
        'escalated_at': row.get('escalated_at'),
    }


def _extract_causation(metadata: dict | None, agent_id: str | None) -> tuple[str, str, dict | None]:
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


# ---------------------------------------------------------------------------
# Dead-letter payload truncation
# ---------------------------------------------------------------------------

_DEAD_LETTER_PAYLOAD_MAX_BYTES = 2048


def _truncate_payload(payload: Any) -> tuple[Any, bool]:
    """Truncate *payload* if its JSON serialisation exceeds the byte budget.

    Returns ``(payload, truncated)`` where *truncated* is ``True`` when the
    payload was cut.  Small payloads are returned unchanged with ``False``.

    When truncated, a typed envelope dict is returned instead of the original
    value, so callers never receive a surprising type change::

        {
            '_truncated': True,
            'text': '<first N bytes of the JSON text>',
            'original_type': '<type name>',
        }

    This lets downstream consumers key into ``payload['text']`` without having
    to special-case a str-vs-dict union on the ``payload`` field.

    The budget check is applied to ``json.dumps(payload, ensure_ascii=False)``
    so that the byte count reflects actual UTF-8 transport size rather than the
    inflated ASCII-escape form.  If the caller requires ASCII-safe output at the
    envelope layer, that conversion should happen there, not in the budget
    measurement.  The ``text`` field in the returned envelope is capped to
    ``_DEAD_LETTER_PAYLOAD_MAX_BYTES`` bytes when UTF-8 encoded.

    Non-JSON-serialisable payloads (e.g. circular references) cannot be safely
    passed through — the MCP transport would crash trying to JSON-encode them.
    In that case the payload is coerced to ``str(payload)``, which is itself
    subject to the same byte-budget check: if it fits, ``(str_value, True)`` is
    returned; if it also exceeds the budget, the capped-envelope form is returned
    (with ``original_type`` reflecting the real payload type, not ``str``).
    Either way ``truncated=True`` signals the lossy conversion to the caller.
    """
    try:
        serialised = json.dumps(payload, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        serialised = str(payload)
        if len(serialised.encode('utf-8')) <= _DEAD_LETTER_PAYLOAD_MAX_BYTES:
            return serialised, True
        text = serialised.encode('utf-8')[:_DEAD_LETTER_PAYLOAD_MAX_BYTES].decode(
            'utf-8', errors='replace'
        )
        return {
            '_truncated': True,
            'text': text,
            'original_type': type(payload).__name__,
        }, True
    if len(serialised.encode('utf-8')) <= _DEAD_LETTER_PAYLOAD_MAX_BYTES:
        return payload, False
    # Cap the raw JSON text to the byte budget, then return a stable-typed
    # envelope so the `payload` field stays a dict regardless of truncation.
    text = serialised.encode('utf-8')[:_DEAD_LETTER_PAYLOAD_MAX_BYTES].decode(
        'utf-8', errors='replace'
    )
    return {
        '_truncated': True,
        'text': text,
        'original_type': type(payload).__name__,
    }, True


def create_mcp_server(
    memory_service: MemoryService,
    task_interceptor: TaskInterceptor | None = None,
    write_journal: WriteJournal | None = None,
    *,
    reconciliation_harness: ReconciliationHarness | None = None,
    backlog_policy: BacklogPolicy | None = None,
    event_queue: EventQueue | None = None,
    curator_usage_gate: UsageGate | None = None,
    known_projects: dict[str, str] | None = None,
) -> FastMCP:
    """Create and configure the FastMCP server with all tools."""

    mcp = FastMCP('Fused Memory', instructions=FUSED_MEMORY_INSTRUCTIONS)
    _taskmaster_configured = task_interceptor is not None

    async def _backlog_gate(project_id: str) -> dict | None:
        """WP-D: reject memory writes when the per-project backlog is over the
        hard limit. For memory tools we don't have ``project_root``; the
        policy uses its internal cache (populated by task ops) to locate
        the escalation directory. Reads are never gated.
        """
        if backlog_policy is None:
            return None
        verdict = await backlog_policy.check(project_id)
        if verdict.is_rejection:
            return verdict.to_error_dict()
        return None

    # WP-E (task 1549): reject write-tool calls whose project_id is absent from
    # the known_projects registry.  Mirrors _backlog_gate but is synchronous and
    # fires BEFORE _backlog_gate so an unknown id never touches downstream state.
    # Read tools are left ungated — unknown project_id reads return empty today,
    # matching current behaviour.  See task 1143 (read-side strictness) and task
    # 1549 (this write-side complement).
    #
    # Task-mutation tools (set_task_status, submit_task, update_task, …) are
    # intentionally NOT gated here.  Those tools use project_root (not project_id)
    # as their primary scope key; applying the gate would require an inversion of
    # the known_projects map (project_id→root) and additional validation logic
    # outside this task's scope.  The harness quarantine (mark_project_dead_letter
    # called from _project_loop's UnknownProjectError handler) provides
    # defence-in-depth: if a task write buffers an event for an unknown project_id,
    # the loop quarantines those rows on first encounter and stops respawning.
    _kp = known_projects or {}

    def _known_project_gate(project_id: str) -> dict | None:
        """Return an error dict if project_id is absent from the known_projects registry."""
        return validate_known_project_id(project_id, _kp)

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

    @mcp.custom_route('/health', methods=['GET'])
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
        body = {'status': 'ok' if ok else 'degraded', 'graphiti': graphiti_ok, 'mem0': mem0_ok}
        return JSONResponse(body, status_code=200 if ok else 503)

    # ------------------------------------------------------------------
    # Write tools
    # ------------------------------------------------------------------

    _VALID_TEMPORAL_CONTEXTS = frozenset({'retrospective', 'planning', 'current'})
    # Derived from the authoritative sets so this validator stays in lockstep
    # automatically when a new status is added to either partition:
    #   ACTIVE_TASK_STATUSES  — non-terminal in-flight statuses
    #                           (fused_memory.reconciliation.task_filter)
    #   TERMINAL_STATUSES     — terminal statuses requiring reopen_reason to exit
    #                           (fused_memory.middleware.task_interceptor)
    # Do NOT hardcode this as a literal; use the union so a future status added
    # to ACTIVE_TASK_STATUSES is automatically accepted here without a separate
    # edit to this file.
    _VALID_TASK_STATUSES: frozenset[str] = ACTIVE_TASK_STATUSES | TERMINAL_STATUSES
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
        if err := _known_project_gate(project_id):
            return err
        if err := await _backlog_gate(project_id):
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
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'add_episode error: {e}')
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
        if err := _known_project_gate(project_id):
            return err
        if err := await _backlog_gate(project_id):
            return err
        if category is not None and category not in _VALID_CATEGORIES:
            return {
                'error': (
                    f'Invalid category {category!r}. '
                    f'Must be one of {sorted(_VALID_CATEGORIES)} or None.'
                ),
                'error_type': 'ValidationError',
            }
        if (
            category == 'temporal_facts'
            and isinstance(agent_id, str)
            and agent_id.startswith('recon-stage-')
            and is_count_snapshot(content)
        ):
            return {
                'error': 'count_snapshot_write_blocked',
                'error_type': 'ReconSnapshotWriteRejected',
                'agent_id': agent_id,
                'content_excerpt': content[:200],
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
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'add_memory error: {e}')
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
            response: dict[str, Any] = {'results': [r.model_dump() for r in results]}
            # Fault-only loudness: surface degraded/failed_stores only when the
            # search was degraded (a selected store timed out or raised).  Uses
            # getattr so a plain list return (back-compat callers) is harmless.
            if getattr(results, 'degraded', False):
                response['degraded'] = True
                response['failed_stores'] = getattr(results, 'failed_stores', [])
            await _log_read(
                operation='search',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'query': query[:200], 'limit': limit},
                result_summary={'count': len(results)},
            )
            return response
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'search error: {e}')
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
    async def count_memories_by_metadata(
        project_id: str,
        filters: dict,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Count memories matching exact metadata equality filters (deterministic, not semantic).

        Unlike ``search``, this tool does NOT rank by vector similarity — it queries
        Qdrant's count API with a direct payload filter, returning an exact integer count.
        Use it for deterministic existence checks where semantic ranking may miss a
        present-but-low-similarity result.

        **Mem0/Qdrant-only scope:** This tool counts only memories stored in the
        Mem0/Qdrant backend (categories: observations_and_summaries, preferences_and_norms,
        procedural_knowledge). It does NOT query Graphiti and will return 0 for facts
        stored in the graph store (entities_and_relations, temporal_facts,
        decisions_and_rationale). Do not use it to confirm the existence of
        Graphiti-stored facts — it will silently return 0 even when those facts exist.

        Primary use-case: confirming whether a Stage 2 per-cycle summary exists for a
        given run_id before concluding it is missing and triggering reconstruction.
        Example call:
            count_memories_by_metadata(
                project_id="dark_factory",
                filters={"kind": "cycle_summary", "run_id": "<run_id>"},
            )
        A return value > 0 means the summary is present; 0 means it was not found by
        metadata key (legacy summaries lacking metadata.run_id will return 0 — fall back
        to semantic search as Path 1).

        This tool is intentionally read-only and is NOT included in any DISALLOW_* list,
        so it is auto-allowed in Stage 3's read-only integrity-check mode.

        Args:
            project_id: Project scope (required)
            filters: Exact metadata key-value pairs to match (e.g. {'kind': 'cycle_summary', 'run_id': '...'})
        """
        if err := validate_project_id(project_id):
            return err
        try:
            count = await memory_service.count_memories_by_metadata(
                project_id=project_id,
                filters=filters,
            )
            return {
                'count': count,
                'project_id': project_id,
                'filters': filters,
            }
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'count_memories_by_metadata error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_memories_by_metadata(
        project_id: str,
        filters: dict,
        limit: int = 1000,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Enumerate memories matching exact metadata equality filters (deterministic scroll, not semantic).

        Deterministic-scroll counterpart to ``count_memories_by_metadata``: instead of
        returning a count, returns the full list of matching memory records so callers
        can inspect each record's metadata fields.  Like ``count_memories_by_metadata``,
        this tool queries Qdrant's payload-filter API directly — it does NOT rank by
        vector similarity, so it will not silently drop low-similarity matches.

        **Mem0/Qdrant-only scope:** This tool enumerates only memories stored in the
        Mem0/Qdrant backend (categories: observations_and_summaries, preferences_and_norms,
        procedural_knowledge). It does NOT query Graphiti.

        **Bounded enumeration:** Results are capped at *limit* records (default 1000).
        If the total matching record count (from ``count_memories_by_metadata``) exceeds
        *limit*, this tool silently returns only the first *limit* records.  Pass an
        explicit *limit* value or cross-check the returned list length against the count
        tool to detect truncation.

        Primary use-case: enumerating stage1_flag_markers to detect orphans that have
        ``source='stage1_flag_marker'`` but are missing ``kind='stage1_flag_marker'``.
        Semantic search is unsuitable for this because its top-N cutoff silently drops
        low-similarity records — the exact failure mode documented in ``_query_stage2_flags``.
        Example call:
            get_memories_by_metadata(
                project_id="dark_factory",
                filters={"source": "stage1_flag_marker"},
            )

        This tool is intentionally read-only and is NOT included in any DISALLOW_* list,
        so it is auto-allowed in Stage 3's read-only integrity-check mode (the same
        property that ``count_memories_by_metadata`` documents).

        Args:
            project_id: Project scope (required)
            filters: Exact metadata key-value pairs to match (e.g. {'source': 'stage1_flag_marker'})
            limit: Maximum records to return (default 1000; service-level cap).

        Returns:
            {'memories': [...], 'project_id': ..., 'filters': ..., 'limit': ...} on success,
            or {'error': ..., 'error_type': ...} on failure.
        """
        if err := validate_project_id(project_id):
            return err
        try:
            memories = await memory_service.get_memories_by_metadata(
                project_id=project_id,
                filters=filters,
                limit=limit,
            )
            return {
                'memories': memories,
                'project_id': project_id,
                'filters': filters,
                'limit': limit,
            }
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_memories_by_metadata error: {e}')
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
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_entity error: {e}')
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
            episodes = await memory_service.get_episodes(project_id=project_id, last_n=last_n)
            await _log_read(
                operation='get_episodes',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'last_n': last_n},
                result_summary={'count': len(episodes)},
            )
            return {'episodes': episodes}
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_episodes error: {e}')
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
        if err := _known_project_gate(project_id):
            return err
        if store not in _VALID_STORES:
            return {
                'error': (f'Invalid store {store!r}. Must be one of {sorted(_VALID_STORES)}.'),
                'error_type': 'ValidationError',
            }
        try:
            causation_id, source, _ = _extract_causation(metadata, agent_id)
            return await memory_service.delete_memory(
                memory_id=memory_id,
                store=store,
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                causation_id=causation_id,
                _source=source,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'delete_memory error: {e}')
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
        if err := _known_project_gate(project_id):
            return err
        try:
            causation_id, source, _ = _extract_causation(metadata, agent_id)
            return await memory_service.delete_episode(
                episode_id=episode_id,
                project_id=project_id,
                cascade=cascade,
                agent_id=agent_id,
                session_id=session_id,
                causation_id=causation_id,
                _source=source,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'delete_episode error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def update_edge(
        edge_uuid: str,
        project_id: str,
        fact: str | None = None,
        invalid_at: str | None = None,
        clear_invalid_at: bool = False,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Update an existing Graphiti edge's fact text and/or invalidate it.

        At least one of ``fact``, ``invalid_at``, or ``clear_invalid_at`` must
        be provided.

        - ``fact``: new fact text. Bypasses the LLM extraction and edge
          resolution pipeline — the embedding is regenerated and both endpoint
          entity summaries are refreshed. Use this when refining or restating
          an existing relationship found via search, instead of add_memory
          (which could false-invalidate active edges).
        - ``invalid_at``: ISO 8601 timestamp marking the edge as superseded
          as of that moment. Used by Stage-2 reconciliation to retire
          contradicted facts (e.g. a 'shipped via X' edge where X isn't in
          the task's recorded commit diff) without destroying the audit trail.
        - ``clear_invalid_at``: when True, resets the edge's ``invalid_at``
          field to ``None``, restoring it to active (non-superseded) status.
          Useful to undo a false invalidation. Compatible with ``fact``
          (update text and un-supersede in one call). Mutually exclusive
          with ``invalid_at`` (cannot set and clear simultaneously).

        All other edge properties (valid_at, endpoints, episodes) are preserved.

        Args:
            edge_uuid: UUID of the existing edge (from search results)
            project_id: Project scope (required)
            fact: New fact text for the edge (optional)
            invalid_at: ISO 8601 timestamp to mark the edge as superseded
                (optional; e.g. "2026-04-19T12:34:56+00:00")
            clear_invalid_at: When True, clear invalid_at (un-supersede the edge).
                Mutually exclusive with invalid_at. Compatible with fact.
            agent_id: Which agent is updating (optional, auto-derived from MCP context)
            session_id: Session context (optional, auto-derived from MCP context)
            metadata: Optional key-value pairs (may contain _causation_id for recon)
        """
        agent_id, session_id = _resolve_identity(agent_id, session_id, ctx)
        if err := validate_project_id(project_id):
            return err
        if err := _known_project_gate(project_id):
            return err
        if not edge_uuid or not edge_uuid.strip():
            return {'error': 'edge_uuid is required', 'error_type': 'ValidationError'}
        normalised_fact: str | None = None
        if fact is not None:
            if not fact.strip():
                return {
                    'error': 'fact text must be non-empty when provided',
                    'error_type': 'ValidationError',
                }
            normalised_fact = fact
        parsed_invalid_at: datetime | None = None
        if invalid_at is not None:
            try:
                parsed_invalid_at = datetime.fromisoformat(invalid_at)
            except ValueError as e:
                return {
                    'error': f'invalid_at must be ISO 8601: {e}',
                    'error_type': 'ValidationError',
                }
            if parsed_invalid_at.tzinfo is None:
                parsed_invalid_at = parsed_invalid_at.replace(tzinfo=UTC)
        if clear_invalid_at and parsed_invalid_at is not None:
            return {
                'error': 'clear_invalid_at and invalid_at are mutually exclusive: '
                'cannot set and clear invalid_at in the same call',
                'error_type': 'ValidationError',
            }
        if normalised_fact is None and parsed_invalid_at is None and not clear_invalid_at:
            return {
                'error': 'update_edge requires fact, invalid_at, or clear_invalid_at',
                'error_type': 'ValidationError',
            }
        try:
            causation_id, source, _ = _extract_causation(metadata, agent_id)
            return await memory_service.update_edge(
                edge_uuid=edge_uuid,
                fact=normalised_fact,
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                causation_id=causation_id,
                _source=source,
                invalid_at=parsed_invalid_at,
                clear_invalid_at=clear_invalid_at,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'update_edge error: {e}')
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
                agent_id=agent_id,
                session_id=session_id,
                causation_id=causation_id,
                _source=source,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'refresh_entity_summary error: {e}')
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
            return {
                'error': 'deprecated_uuid must be a non-empty string',
                'error_type': 'ValidationError',
            }
        if not surviving_uuid or not surviving_uuid.strip():
            return {
                'error': 'surviving_uuid must be a non-empty string',
                'error_type': 'ValidationError',
            }
        if deprecated_uuid.strip() == surviving_uuid.strip():
            return {
                'error': 'deprecated_uuid and surviving_uuid must be different',
                'error_type': 'ValidationError',
            }
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
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'merge_entities error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def delete_entity(
        entity_uuid: str,
        project_id: str,
        force: bool = False,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Delete a Graphiti Entity node by UUID, refreshing connected neighbours.

        Performs a DETACH DELETE of the specified entity node from the FalkorDB
        graph. Before deleting, collects all neighbours connected via valid
        (non-invalidated) edges and refreshes their summaries afterwards.

        By default, refuses to delete a node that still has valid active edges
        (raises ActiveEdgesError). Pass force=True to override this guard and
        delete unconditionally.

        The operation is scoped to the given project_id. The node must exist —
        a NodeNotFoundError is returned if it does not.

        Args:
            entity_uuid: UUID of the entity node to delete (required, non-empty)
            project_id: Project scope (required, must be non-empty)
            force: When True, bypass the active-edges guard and delete anyway
            agent_id: Which agent is calling (optional, auto-derived from MCP context)
            session_id: Session context (optional, auto-derived from MCP context)
            metadata: Optional key-value pairs (may contain _causation_id for recon)
        """
        agent_id, session_id = _resolve_identity(agent_id, session_id, ctx)
        if err := validate_project_id(project_id):
            return err
        if not entity_uuid or not entity_uuid.strip():
            return {
                'error': 'entity_uuid must be a non-empty string',
                'error_type': 'ValidationError',
            }
        try:
            causation_id, source, _ = _extract_causation(metadata, agent_id)
            return await memory_service.delete_entity(
                entity_uuid=entity_uuid,
                project_id=project_id,
                force=force,
                agent_id=agent_id,
                session_id=session_id,
                causation_id=causation_id,
                _source=source,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'delete_entity error: {e}')
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
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'rebuild_entity_summaries error: {e}')
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
            result = await memory_service.get_status(project_id=project_id)

            # Enrich the queue section with a dead_letters sub-dict that
            # mirrors the shape returned by get_dead_letters, so Stage 1
            # reconciliation can compare both values without false positives.
            #
            # durable_dead: already project-scoped by MemoryService (step-4).
            # event_dead: read from the JSONL dead-letter file via to_thread
            #   (same offload pattern as get_dead_letters) to avoid blocking
            #   the event loop on large files.
            queue_section = result.get('queue') if isinstance(result, dict) else None
            if queue_section is None:
                # No durable_queue configured — attach a zero dead_letters anyway
                if isinstance(result, dict):
                    result.setdefault('queue', {})
                    result['queue']['dead_letters'] = {
                        'durable_queue': 0,
                        'event_queue': 0,
                        'total': 0,
                    }
            elif 'error' not in queue_section:
                # Skip enrichment for error queue sections (e.g. {'error': '...'}).
                # Mutating an error dict with dead_letters stats would create a
                # confusing mixed error/stats shape for consumers.
                durable_dead: int = queue_section.get('counts', {}).get('dead', 0)

                event_dead: int = 0
                if event_queue is not None:
                    # Use count_dead_letters (streaming line count) instead of
                    # len(read_dead_letters(...)) — get_status is polled frequently by
                    # Stage 1 reconciliation and we do not need to materialise records.
                    event_dead = await asyncio.to_thread(
                        event_queue.count_dead_letters,
                        project_id=project_id,
                    )

                queue_section['dead_letters'] = {
                    'durable_queue': durable_dead,
                    'event_queue': event_dead,
                    'total': durable_dead + event_dead,
                }

            return result
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_status error: {e}')
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
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'replay_to_graphiti error: {e}')
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
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_queue_stats error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_curator_state() -> dict[str, Any]:
        """Read-only snapshot of the curator UsageGate's current state.

        Returns ``{'paused', 'paused_reason', 'soonest_open_at', 'account_count'}``.
        ``paused_reason`` is None when the gate is healthy or unconfigured.
        ``soonest_open_at`` is an ISO 8601 string or None.
        When no gate is wired (curator disabled), returns safe zero-defaults.

        The dashboard polls this to surface WHY the curator is paused alongside
        the existing ``capped_now`` indicator.

        Note: the four values are read from the live gate without a lock. During
        a state transition (e.g. ``_on_resume()`` flipping paused → unpaused) the
        snapshot may be momentarily inconsistent; refresh on the next poll.
        """
        try:
            if curator_usage_gate is None:
                return {
                    'paused': False,
                    'paused_reason': None,
                    'soonest_open_at': None,
                    'account_count': 0,
                }
            gate = curator_usage_gate
            paused = gate.is_paused
            paused_reason = gate.paused_reason or None
            resets_at = gate.soonest_resets_at
            soonest_open_at = resets_at.isoformat() if resets_at is not None else None
            return {
                'paused': paused,
                'paused_reason': paused_reason,
                'soonest_open_at': soonest_open_at,
                'account_count': gate.account_count,
            }
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_curator_state error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_wal_status() -> dict[str, Any]:
        """Latest per-store WAL checkpoint status.

        Returns ``{'stores': {store_name: {ts, busy, log, checkpointed, detail}}}``
        — one row per live SQLite store. ``ts`` is the iso8601 of the most
        recent checkpoint pass. ``busy>0`` means readers/writers blocked
        the TRUNCATE from fully truncating the WAL; ``log`` reports frames
        present pre-checkpoint; ``checkpointed`` reports frames copied to
        the main DB this pass. A missing store means the periodic loop has
        not yet checkpointed it (typically only true within the first
        five minutes of startup).

        The dashboard polls this to surface WAL health — drift in ``ts``
        means the periodic loop has stalled; ``busy>0`` consistently
        means a long-held reader/writer is preventing TRUNCATE; ``log``
        growing without bound suggests checkpoint backpressure.

        Added in response to the 2026-05-13 incident — see
        ``docs/task-recovery-2026-05-13/`` for forensic detail.
        """
        from fused_memory.server.wal_status import CHECKPOINT_STATUS

        # Defensive copy so concurrent updates from the periodic loop
        # don't tear the wire payload mid-serialise.
        return {'stores': {name: dict(row) for name, row in CHECKPOINT_STATUS.items()}}

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
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'replay_dead_letters error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_dead_letters(
        project_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Inspect dead-lettered items from both the durable write queue and
        the event dead-letter JSONL file.

        Returns a merged list of dead-letter records from all sources,
        newest-first, with a ``source`` discriminator so operators can triage
        in one place.

        Args:
            project_id: Filter to a specific project (optional — all if omitted)
            limit: Maximum total items to return (default 100)
        """
        try:
            items: list[dict[str, Any]] = []

            # --- durable write queue ---
            if memory_service.durable_queue is not None:
                dead = await memory_service.durable_queue.get_dead_items(
                    group_id=project_id,
                    limit=limit,
                )
                for row in dead:
                    payload, truncated = _truncate_payload(row.get('payload'))
                    item: dict[str, Any] = {
                        'source': 'durable_queue',
                        'id': row.get('id'),
                        'operation': row.get('operation'),
                        'payload': payload,
                        'error': row.get('error'),
                        'timestamp': row.get('created_at'),
                        'attempts': row.get('attempts'),
                    }
                    if truncated:
                        item['payload_truncated'] = True
                    items.append(item)

            # --- event queue dead-letter JSONL ---
            eq_items: list[dict[str, Any]] = []
            if event_queue is not None:
                remaining = limit - len(items)
                if remaining > 0:
                    # read_dead_letters does synchronous file I/O; offload to a
                    # thread so the event loop is not blocked on large files.
                    records = await asyncio.to_thread(
                        event_queue.read_dead_letters,
                        limit=remaining,
                        project_id=project_id,
                    )
                    for rec in records:
                        ev = rec.get('event') or {}
                        payload, truncated = _truncate_payload(ev.get('payload'))
                        eq_item: dict[str, Any] = {
                            'source': 'event_queue',
                            'id': ev.get('id'),
                            'type': ev.get('type'),
                            'payload': payload,
                            'reason': rec.get('reason'),
                            'timestamp': rec.get('failed_at'),
                            'attempts': rec.get('attempts'),
                            'project_id': ev.get('project_id'),
                        }
                        if truncated:
                            eq_item['payload_truncated'] = True
                        eq_items.append(eq_item)

            all_items = items + eq_items
            counts: dict[str, int] = {
                'durable_queue': sum(1 for i in all_items if i['source'] == 'durable_queue'),
                'event_queue': sum(1 for i in all_items if i['source'] == 'event_queue'),
            }

            return {'items': all_items[:limit], 'counts': counts}
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_dead_letters error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def delete_dead_letters(
        project_id: str,
        ids: list[int],
    ) -> dict[str, Any]:
        """Permanently delete dead-lettered durable-queue items by id.

        Use this tool for non-retriable errors (e.g. NodeNotFoundError after a
        graph wipe) where replaying would always fail.  For retriable transient
        failures use ``replay_dead_letters`` instead.

        Only rows with ``status='dead'`` that belong to ``project_id`` are
        eligible.  Cross-project ids, non-existent ids, and non-dead-status
        ids land in ``not_found`` without leaking information.

        .. note::
            This tool only covers entries in the *durable write queue* (SQLite).
            Dead letters in the **event_queue** (JSONL) use string UUIDs, not
            integers.  event_queue UUIDs are not valid integer ids and will be
            rejected at input validation before the request reaches the tool;
            those entries therefore never appear in ``not_found``.
            Filter ``get_dead_letters`` output on ``source == 'durable_queue'``
            before constructing the ``ids`` list for this tool.

            **Large id lists** are safe: the tool internally chunks requests into
            batches of 500 so SQLite's ``SQLITE_MAX_VARIABLE_NUMBER`` limit is
            never exceeded.

            **Transient SQLite errors** (database locked, disk full) are returned
            as a retriable envelope rather than raised::

                {
                    'error':      '<exception message>',
                    'error_type': 'TransientSqliteError',
                    'retriable':  True,
                    'deleted':    [...ids durably deleted in prior chunks...],
                    'not_found':  [...ineligible ids from prior chunks...],
                    'remaining':  [...ids in the failing chunk and later — never attempted...],
                }

            ``remaining`` excludes ineligible ids already classified in prior
            chunks, so re-calling with ``ids=remaining`` is safe and
            non-redundant.  Re-call after the underlying issue is resolved.

        Args:
            project_id: Project scope (required — prevents accidental cross-project deletes).
            ids: Integer row ids to delete (e.g. [1820, 2017] for the dark_factory entries).
                 Any number of ids is accepted; large lists are chunked automatically.

        Returns:
            On success: ``{'deleted': [...sorted ids removed...], 'not_found': [...sorted ids missed...]}``

            On transient SQLite error: ``{'error': ..., 'error_type': 'TransientSqliteError', 'retriable': True, 'deleted': [...], 'not_found': [...], 'remaining': [...]}``
        """
        if err := validate_project_id(project_id):
            return err
        if err := validate_int_ids(ids):
            return err
        try:
            if memory_service.durable_queue is None:
                return {'error': 'Queue not initialized', 'error_type': 'ConfigurationError'}
            result = await memory_service.durable_queue.delete_dead(
                group_id=project_id,
                ids=ids,
            )
            return result
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'delete_dead_letters error: {e}')
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
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'trigger_reconciliation error: {e}')
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
        await reconciliation_harness.judge.unhalt(project_id)
        grace = reconciliation_harness.judge.unhalt_grace_remaining(project_id)
        return {
            'status': 'unhalted' if was_halted else 'already_running',
            'project_id': project_id,
            'grace_cycles_remaining': grace,
            'message': (
                f'Reconciliation unhalted for {project_id}. Next cycle will run '
                f'within ~5 seconds; trend detector suppressed for {grace} cycles.'
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

    def _normalize_project_root(project_root: str) -> str | dict:
        """Validate then redirect project_root to the main git checkout.

        Worktrees must never hold their own tasks.json — every task tool
        funnels through this choke point so reads and writes see the same
        canonical copy regardless of which path the caller passed in.

        Returns the normalized path (str) on success, or an error payload
        (dict with 'error' and 'error_type' keys) on failure. Call sites
        should `isinstance(result, dict)` to narrow.
        """
        if err := validate_project_root(project_root):
            return err
        try:
            return resolve_main_checkout(project_root)
        except ValueError as e:
            return {'error': str(e), 'error_type': 'ValidationError'}

    def _reject_if_ticket_id(name: str, value: object) -> dict | None:
        """Return a ValidationError dict if ``value`` is a ticket-shaped id.

        Ticket ids (``tkt_`` prefix) are returned by ``submit_task`` and must
        be resolved via ``resolve_ticket`` before being passed to id-accepting
        task tools. Returning a clear error here prevents confusing downstream
        failures inside the taskmaster backend.

        Delegates to :func:`~fused_memory.middleware.task_interceptor._is_ticket_id`
        so there is a single source of truth for the ticket-id prefix.
        """
        if _is_ticket_id(value):
            return {
                'error': (
                    f'Ticket-shaped id {value!r} not allowed here; '
                    'call resolve_ticket first to obtain a numeric task_id.'
                ),
                'error_type': 'ValidationError',
            }
        return None

    def _reject_status_in_update_task(
        task_id: str,
        status: str | None,
    ) -> dict | None:
        """Return a rejection dict if ``update_task`` was called with ``status``.

        ``set_task_status`` is the only sanctioned writer for task status — it
        enforces the terminal-exit gate, the phantom-done gate, and the
        done-provenance gate. ``update_task(status=…)`` slipped through all
        three and was used to mark reify tasks done with the implementing
        commit only on the task branch (2026-05-08 forensics: 9 historical
        ``done`` writes via this path in 36 h). Lock the door.
        """
        if status is None:
            return None
        return {
            'success': False,
            'error': 'status_via_update_task',
            'task_id': task_id,
            'status': status,
            'hint': (
                'update_task is metadata-only. Use '
                'set_task_status(status=…, done_provenance={...} when '
                'status="done") to change status — it enforces the '
                'terminal-exit, phantom-done, and done-provenance gates.'
            ),
        }

    def _reject_done_provenance_in_metadata(
        metadata: str | dict | None,
    ) -> dict | None:
        """Return a ValidationError dict if ``metadata`` carries done_provenance.

        ``set_task_status`` is the only sanctioned writer for
        ``metadata.done_provenance`` — it validates the schema (kind, commit,
        note) and runs an ancestor backstop on merge SHAs. Allowing
        ``update_task`` to write the field bypasses that gate, which is how
        a workflow agent stamped a self-contradicting "done" record on
        2026-04-27. Reject the call before it reaches the interceptor.
        """
        parsed: dict | None = None
        if isinstance(metadata, dict):
            parsed = metadata
        elif isinstance(metadata, str):
            try:
                loaded = json.loads(metadata)
            except (ValueError, TypeError):
                return None
            parsed = loaded if isinstance(loaded, dict) else None
        if parsed is not None and 'done_provenance' in parsed:
            return {
                'error': (
                    'update_task cannot write metadata.done_provenance. Use '
                    'set_task_status(status="done", done_provenance={...}) '
                    'instead — it validates the schema and runs an ancestor '
                    'backstop on the merge sha.'
                ),
                'error_type': 'ValidationError',
            }
        return None

    @mcp.tool()
    async def get_tasks(
        project_root: str,
        tag: str | None = None,
        page_size: int | None = None,
        offset: int = 0,
        statuses: Any = None,
    ) -> dict[str, Any]:
        """List all tasks in the project.

        Args:
            project_root: Absolute path to project root
            tag: Tag context (optional)
            page_size: If provided, return at most this many tasks (must be a positive
                integer).  When omitted (default), the full task list is returned with no
                ``pagination`` key — backward-compatible behaviour for the scheduler and
                all existing callers.
            offset: Zero-based index of the first task to return (default 0).  Only
                meaningful when page_size is provided.
            statuses: Opt-in status filter.  When omitted or None, the full unfiltered
                task tree is returned (byte-identical to the current behaviour).  Pass a
                list of status strings (e.g. ``['pending', 'in-progress']``) to restrict
                results to matching tasks via a SQL ``status IN (...)`` predicate.  An
                empty list (``[]``) is valid and returns no tasks.  A bare string is
                rejected with a ValidationError.
        """
        # Input validation — early-exit before touching the interceptor.
        if page_size is not None and (not isinstance(page_size, int) or isinstance(page_size, bool) or page_size <= 0):
            return {
                'error': 'page_size must be a positive integer',
                'error_type': 'ValidationError',
            }
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            return {
                'error': 'offset must be a non-negative integer',
                'error_type': 'ValidationError',
            }
        if statuses is not None and not isinstance(statuses, list):
            return {
                'error': 'statuses must be a list of status strings',
                'error_type': 'ValidationError',
            }
        if statuses is not None and any(not isinstance(s, str) for s in statuses):
            return {
                'error': 'statuses must be a list of status strings',
                'error_type': 'ValidationError',
            }

        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        try:
            result = await task_interceptor.get_tasks(project_root=project_root, tag=tag, statuses=statuses)
            if isinstance(result, dict) and 'error' not in result:
                pid = resolve_project_id(project_root)
                # Shallow copy to avoid mutating a potentially shared/cached interceptor dict.
                result = {**result, 'project_id': pid, 'project_root': project_root}

                # Opt-in pagination — only applied when page_size is explicitly provided.
                # When page_size is None the response is the full untouched list (no
                # ``pagination`` key), preserving backward compatibility for the scheduler.
                if page_size is not None:
                    all_tasks = result.get('tasks')
                    # Only paginate when tasks is a proper list — a non-standard backend
                    # could return None or a dict; in that case skip pagination and leave
                    # the result untouched rather than masking the real failure with a
                    # generic slicing error.
                    if isinstance(all_tasks, list):
                        total = len(all_tasks)
                        page = all_tasks[offset:offset + page_size]
                        result['tasks'] = page
                        result['pagination'] = {
                            'total': total,
                            'offset': offset,
                            'page_size': page_size,
                            'returned': len(page),
                            'has_more': offset + len(page) < total,
                        }

                await _log_read(
                    'get_tasks',
                    project_id=pid,
                    result_summary={
                        'project_id': pid,
                        'count': len(result.get('tasks', [])),
                    },
                )
            return result
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_tasks error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_statuses(
        project_root: str,
        ids: list[str] | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Return a compact ``{id: status}`` mapping — status-only, ~95% smaller than get_tasks.

        Use this instead of get_tasks when callers only need task statuses (e.g.
        reconcile loops, startup checks). Full task data is still available via
        get_tasks or get_task.

        Args:
            project_root: Absolute path to project root
            ids: Optional list of task ids to filter to (unknown ids silently omitted).
                 Omit or pass null for all tasks.
            tag: Tag context (optional)
        """
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        try:
            result = await task_interceptor.get_statuses(
                project_root=project_root, ids=ids, tag=tag
            )
            await _log_read(
                'get_statuses',
                result_summary={'count': len(result)},
            )
            return {'statuses': result}
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_statuses error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_external_statuses(deps: list[str]) -> dict[str, str]:
        """Return cross-project task statuses keyed by the verbatim dep string.

        Each dep is a ``"<project_id>:<task_id>"`` string.  For each dep the
        tool normalises the project_id, looks it up in the known-projects
        registry, reads the foreign project's top-level task status via
        ``task_interceptor.get_statuses``, and returns the result keyed by the
        **original** verbatim dep string.

        Possible values per key:
        - A real task status (e.g. ``"done"``, ``"pending"``)
        - ``"unknown_project"`` — project_id not in the registry
        - ``"unknown_task"``    — project known, but no top-level task with that id
        - ``"malformed"``       — dep cannot be parsed as ``<project_id>:<int>``

        Read-only: no reconciliation, no event emission, no task mutation.
        Registry/DB unavailability raises (transient) — NOT mapped to a sentinel.
        """
        result: dict[str, str] = {}
        # Collect well-formed, known-project deps grouped by normalised project_id
        # so we can issue ONE get_statuses call per distinct foreign project.
        # Maps norm_project_id → list of (verbatim_dep, task_id) tuples.
        project_batches: dict[str, list[tuple[str, str]]] = {}

        for dep in deps:
            # Parse: split on first colon
            project_id, sep, task_id = dep.partition(':')
            # Validate: malformed if no colon, empty project_id, empty task_id,
            # or task_id not a plain non-negative integer (rejects dotted subtask
            # forms like "15.2", signs, non-numerics — mirrors add_dependency's
            # subtask rejection).
            if not sep or not project_id or not task_id or not task_id.isdigit():
                result[dep] = 'malformed'
                continue
            # Normalise project_id: lowercase + hyphen→underscore, mirroring
            # models/scope.py:resolve_project_id so 'dark-factory' == 'dark_factory'.
            # Registry lookup uses the normalised form; result is keyed by the
            # original verbatim dep string.
            norm_project_id = project_id.lower().replace('-', '_')
            # Look up project_root in registry
            if norm_project_id not in _kp:
                result[dep] = 'unknown_project'
                continue
            project_batches.setdefault(norm_project_id, []).append((dep, task_id))

        # Issue ONE get_statuses call per distinct foreign project (minimises reads).
        # Intentionally NOT wrapped in try/except — transient errors (DB/registry
        # unavailability) must propagate as exceptions, not be mapped to a sentinel.
        # Sentinels = semantic "unresolvable"; exceptions = transient "couldn't answer".
        for norm_project_id, dep_pairs in project_batches.items():
            project_root = _kp[norm_project_id]
            # Redirect worktree roots to the main checkout, mirroring all other
            # read tools (see _normalize_project_root docstring).  Registry roots
            # from build_known_projects_map are expected to be canonical
            # main-checkout paths, but this guard prevents a silent divergence if
            # a registered root were ever a worktree path.
            # Resolution failure raises (transient) — consistent with this tool's
            # "no sentinel for transient failures" contract.
            _norm = _normalize_project_root(project_root)
            if isinstance(_norm, dict):
                raise RuntimeError(
                    f'Cannot resolve registered root for {norm_project_id!r}: {_norm}'
                )
            project_root = _norm
            task_ids = [tid for _, tid in dep_pairs]
            statuses = await task_interceptor.get_statuses(project_root=project_root, ids=task_ids)
            for dep, task_id in dep_pairs:
                if task_id not in statuses:
                    result[dep] = 'unknown_task'
                else:
                    result[dep] = statuses[task_id]

        await _log_read('get_external_statuses', result_summary={'count': len(result)})
        return result

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
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        try:
            result = await task_interceptor.get_task(task_id=id, project_root=project_root, tag=tag)
            if isinstance(result, dict) and 'error' not in result:
                pid = resolve_project_id(project_root)
                # Shallow copy to avoid mutating a potentially shared/cached interceptor dict.
                result = {**result, 'project_id': pid, 'project_root': project_root}
                await _log_read(
                    'get_task',
                    project_id=pid,
                    result_summary={'project_id': pid, 'task_id': id},
                )
            return result
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_task error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def set_task_status(
        id: str,
        status: str,
        project_root: str,
        tag: str | None = None,
        done_provenance: dict | None = None,
        reopen_reason: str | None = None,
    ) -> dict[str, Any]:
        """Update task status. Triggers targeted reconciliation for
        done/blocked/cancelled/deferred transitions. Entering merge-deferred is
        a non-terminal hold and does NOT trigger reconciliation; the group merge
        that flips members to done provides the signal.

        Reconciliation may: attach memory_hints to the task, write completion
        knowledge to memory stores, or flag dependent tasks that need attention.

        Pre-done hook: when ``FUSED_MEMORY_PREDONE_HOOK_<PROJECT_ID>`` is
        configured (where ``<PROJECT_ID>`` is the upper-cased project id
        derived from ``project_root``, e.g. ``FUSED_MEMORY_PREDONE_HOOK_REIFY``
        for ``/home/leo/src/reify``), ``status='done'`` transitions invoke that
        subprocess as a pre-write validator before any state is mutated.
        Non-zero exit refuses the transition and returns
        ``{'success': False, 'error': 'pre_done_hook_rejected', ...}``.
        Unset or empty value = no-op (no subprocess, no latency overhead).

        Args:
            id: Task ID (comma-separated for multiple)
            status: pending, done, in-progress, blocked, review, deferred, cancelled, or
                merge-deferred (non-terminal holding state for atomic-train members
                awaiting group merge; see PRD orchestrator-atomic-train-merge §9.2,
                task 1519)
            project_root: Absolute path to project root
            tag: Tag context (optional)
            done_provenance: Verified evidence for a done transition; Stage-2
                reconciliation uses this instead of fabricating 'shipped via X'
                edges from metadata.modules. Schema::

                  {
                      'kind': 'merged' | 'found_on_main',  # required
                      'commit': '<sha-or-ref>',  # required for both kinds
                      'note': '<free text>',  # required if kind="found_on_main"
                  }

                ``kind="merged"``: the work landed on main via a merge commit.
                ``commit`` is resolved via ``git rev-parse`` to a full 40-char
                SHA and ancestor-checked against main via
                ``git merge-base --is-ancestor <sha> main``.
                ``kind="found_on_main"``: the implementation is already on main
                from a sibling task / prior orchestrator run. Both ``commit``
                and ``note`` are required; ``commit`` is resolved + ancestor-
                checked identically. ``note`` must cite the providing
                task/commit.
                A bare ``{"note": ...}`` (no ``kind``) is always rejected.
                When reconciliation.require_done_provenance is False (default
                during rollout), missing/empty provenance logs a warning and
                the transition proceeds; malformed provenance (wrong type,
                unknown kind, unresolvable commit, branch-only SHA) always
                errors regardless.
            reopen_reason: Required to exit a terminal status (done, cancelled).
                Short free-text explanation — e.g. 'un-defer script',
                'manual re-scope', 'reconciliation: re-implementation required'.
                Persisted on the task as metadata.reopen_reason for audit.
                Ignored for non-terminal transitions.
        """
        if err := _reject_if_ticket_id('id', id):
            return err
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        if status not in _VALID_TASK_STATUSES:
            return {
                'error': (
                    f'Invalid status {status!r}. Must be one of {sorted(_VALID_TASK_STATUSES)}.'
                ),
                'error_type': 'ValidationError',
            }
        try:
            return await task_interceptor.set_task_status(
                task_id=id,
                status=status,
                project_root=project_root,
                tag=tag,
                done_provenance=done_provenance,
                reopen_reason=reopen_reason,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'set_task_status error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def submit_task(
        project_root: str,
        prompt: str | None = None,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        dependencies: str | None = None,
        priority: str | None = None,
        metadata: str | dict[str, Any] | None = None,
        tag: str | None = None,
        planning_mode: bool = False,
        routing_override_reason: str = '',
    ) -> dict[str, Any]:
        """Phase-1 of two-phase task creation: persist a ticket and return its id immediately.

        Returns ``{"ticket": "tkt_<id>"}`` so the caller can either poll or
        block via ``resolve_ticket``.  Does NOT call the Taskmaster backend
        directly — that happens asynchronously in the curator worker.

        Callers should follow up with ``resolve_ticket`` to obtain the final
        task_id once the curator has decided (create / drop / combine).

        ``planning_mode=True`` switches to a synchronous, curator-bypassing
        path for batched human decomposition (e.g., breaking a PRD into ~50
        tasks).  In planning mode the task is created directly in ``deferred``
        status (a single committed write — no transient ``pending`` state) so
        the orchestrator scheduler cannot claim it before the planner has wired
        up sibling dependencies.  The planner commits the batch by calling
        ``commit_planning`` once all siblings and dependencies are in place.

        Planning mode returns ``{"task_id": "<id>", "status": "deferred",
        "planning_mode": True}`` synchronously — no ticket, no
        ``resolve_ticket`` follow-up needed.

        Args:
            project_root: Absolute path to project root
            prompt: Task description for AI generation (forwarded to Taskmaster)
            title: Task title
            description: Task description
            details: Task details / implementation notes
            dependencies: Comma-separated dependency task IDs
            priority: critical, high, medium, low, or polish (default medium)
            metadata: Task metadata (object or JSON string)
            tag: Tag context (optional)
            planning_mode: When True, bypass the curator and create the task
                directly in ``deferred`` status.  Use this during heavy
                decomposition sessions where you do not want curator
                deduplication to recombine sibling tasks.  Persists
                ``human_decomposed=True`` in task metadata.
            routing_override_reason: When set (non-empty), the path guards are
                skipped and the task is filed in the submitting project.  The
                reason is recorded on task metadata and emitted as a WARNING
                audit log so a deliberate override is greppable.  Use only
                when sure the task belongs to the submitting project.  If
                unsure, escalate rather than risking a mis-filed task.
        """
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        try:
            return await task_interceptor.submit_task(
                project_root=project_root,
                prompt=prompt,
                title=title,
                description=description,
                details=details,
                dependencies=dependencies,
                priority=priority,
                metadata=metadata,
                tag=tag,
                planning_mode=planning_mode,
                routing_override_reason=routing_override_reason,
            )
        except Exception as e:
            logger.error(f'submit_task error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def resolve_ticket(
        ticket: str,
        project_root: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Phase-2 of two-phase task creation: block until the curator worker decides.

        Returns ``{status, task_id?, reason?}`` once the ticket is terminal.
        If the ticket is already terminal, returns immediately.

        Status values:
        - ``created``  — a new task was created; ``task_id`` is the numeric id.
        - ``combined`` — candidate was folded into an existing task; ``task_id``
          is the target task's id.
        - ``failed``   — an error occurred; ``reason`` describes it. Common
          reasons: ``timeout``, ``server_restart``, ``expired``.

        Callers that receive ``status=failed, reason=timeout`` should either
        retry or report an error.

        Args:
            ticket: Ticket id returned by ``submit_task`` (must start with ``tkt_``)
            project_root: Absolute path to project root (same as supplied to submit_task)
            timeout_seconds: Maximum seconds to wait.  Defaults to 115 s (just
                under the MCP 120 s hard limit) so external callers that omit this
                parameter cannot hang indefinitely on an orphaned ticket.
        """
        if not _is_ticket_id(ticket):
            # Idempotent passthrough: if the caller passed a numeric task id,
            # short-circuit instead of erroring.  This catches agents that
            # blindly use the two-step submit/resolve dance after a
            # ``planning_mode=True`` submit_task that already returned a
            # synchronous ``task_id``.  No store lookup is performed — the
            # contract is that any numeric id is treated as already-resolved.
            if _looks_like_task_id(ticket):
                return {
                    'status': 'created',
                    'task_id': str(ticket).strip(),
                    'reason': 'idempotent_passthrough',
                }
            return {
                'error': f'ticket must start with tkt_ (got {ticket!r})',
                'error_type': 'ValidationError',
            }
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        # Apply a safe default timeout at the MCP layer so external callers
        # cannot block indefinitely.
        effective_timeout = 115.0 if timeout_seconds is None else timeout_seconds
        try:
            return await task_interceptor.resolve_ticket(
                ticket=ticket,
                project_root=project_root,
                timeout_seconds=effective_timeout,
            )
        except Exception as e:
            logger.error(f'resolve_ticket error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def list_tickets(
        project_root: str,
        status: str | None = None,
        since: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """List recent submit_task tickets for a project. Default window: last 7 days.

        Returns each ticket with extracted candidate_title for at-a-glance
        triage. ``result_json`` is omitted (verbose; not useful here).

        Note on ``expires_at``: this field is a non-load-bearing advisory
        far-future placeholder (now + 365 days at submit time). The retired
        wall-clock TTL janitor no longer runs. Stuck or abandoned pending
        tickets are reaped by the worker-liveness reaper (TicketJanitor.tick),
        which marks pending tickets failed with reason='worker_dead' when the
        project's curator worker has died — NOT by a wall-clock TTL sweep.

        Args:
            project_root: Absolute path to project root.
            status: Optional status filter ('pending', 'created', 'failed',
                'combined'). When None, all statuses are returned.
            since: Optional ISO-8601 timestamp; only tickets with
                ``created_at >= since`` are returned. Default: now − 7 days.
            limit: Max rows to return. Clamped to [1, 2000].
        """
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        since_dt: datetime | None = None
        if since is not None:
            try:
                parsed = datetime.fromisoformat(since)
            except ValueError:
                return {
                    'error': (f'list_tickets: invalid ISO-8601 timestamp for "since": {since!r}'),
                    'error_type': 'ValidationError',
                }
            since_dt = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        try:
            result = await task_interceptor.list_tickets(
                project_root=project_root,
                status=status,
                since=since_dt,
                limit=limit,
            )
        except Exception as e:
            logger.exception(f'list_tickets error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}
        if 'error' in result:
            return result
        rows = result.pop('rows', [])
        result['tickets'] = [_summarise_ticket_row(r) for r in rows]
        return result

    @mcp.tool()
    async def search_tasks(
        project_root: str,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.3,
    ) -> dict[str, Any]:
        """Semantic search over already-filed tasks (NOT memories — use `search` for those).

        Answers "was a task like X already filed?" by matching ``query`` against
        the curator's corpus of every filed task's title+description+files.
        Returns matches ranked by cosine similarity, each enriched with its
        current ``status`` (done/pending/cancelled/…) so you can tell whether a
        near-match is already complete. A ``status`` of ``null`` means the task
        no longer exists.

        Each result carries: ``task_id``, ``title``, ``description``,
        ``files_to_modify``, ``priority``, ``updated_at``, ``score`` (cosine
        similarity, higher = more similar), and ``status``.

        Args:
            project_root: Absolute path to project root.
            query: Free-text query (e.g. a paraphrase of the task you're about to file).
            limit: Max number of matches to return (clamped to [1, 100]).
            score_threshold: Minimum cosine similarity in 0–1 (default 0.3).
                Weak matches are dropped server-side; raise it (e.g. 0.7) for
                near-duplicates only, lower it for broader recall.
        """
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        if not query or not query.strip():
            return {'error': 'query must be non-empty', 'error_type': 'ValidationError'}
        if limit <= 0:
            return {'error': 'limit must be positive', 'error_type': 'ValidationError'}
        if limit > 100:
            limit = 100
        try:
            return await task_interceptor.search_tasks(
                project_root=project_root,
                query=query,
                limit=limit,
                score_threshold=score_threshold,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'search_tasks error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def cancel_ticket(ticket_id: str) -> dict[str, Any]:
        """Cancel a pending curator ticket by its ticket_id.

        Four outcome shapes (v1 contract):

        * **config_error** — ticket store not configured (server misconfiguration):
          ``{'error': 'ticket_store not configured', 'error_type': 'ConfigError',
          'ticket_id': ticket_id}``
        * **not_found** — ticket does not exist:
          ``{'error': 'not_found', 'ticket_id': ticket_id}``
        * **no_op** — ticket is already in a terminal/non-pending status:
          ``{'status': <current>, 'ticket_id': ticket_id, 'no_op': True}``
        * **cancelled** — ticket was pending and has been cancelled:
          ``{'status': 'cancelled', 'ticket_id': ticket_id}``

        v1 trade-off: in-flight curator/LLM calls are NOT interrupted.
        Queued-but-not-started tickets are dropped cleanly when the worker
        dequeues them and finds the row no longer pending.

        Args:
            ticket_id: The ``tkt_…`` ticket identifier returned by ``submit_task``.
        """
        try:
            return await task_interceptor.cancel_ticket(ticket_id=ticket_id)
        except Exception as e:
            logger.exception(f'cancel_ticket error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def commit_planning(
        project_root: str,
        task_ids: str,
        target_status: str = 'pending',
    ) -> dict[str, Any]:
        """Commit a batch of tasks created via ``submit_task(planning_mode=True)``.

        Flips a comma-separated batch of task ids from ``deferred`` to the
        target status (default ``pending``) atomically within the per-project
        write lock, so the orchestrator scheduler sees the batch as a coherent
        unit on its next ~15 s poll rather than picking up siblings one at a
        time as planning proceeds.

        Use this paired with ``submit_task(planning_mode=True)`` to safely
        decompose a large PRD into many tasks plus dependencies before the
        orchestrator can claim any of them.

        Args:
            project_root: Absolute path to project root (matches submit_task).
            task_ids: Comma-separated task ids to flip (e.g. ``"42,43,44"``).
                Each id is processed under the same per-project write lock as
                ``set_task_status`` and runs the same gates.
            target_status: ``pending`` to release for scheduling (default),
                ``deferred`` to leave them parked, or ``cancelled`` to discard
                the planned batch.  Other status values are rejected.

        Returns ``{success, results: [{task_id, result: ...}, ...]}`` matching
        the multi-id ``set_task_status`` response shape.
        """
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized

        valid_targets = {'pending', 'deferred', 'cancelled'}
        if target_status not in valid_targets:
            return {
                'error': (
                    f'commit_planning: target_status must be one of '
                    f'{sorted(valid_targets)} (got {target_status!r})'
                ),
                'error_type': 'ValidationError',
            }

        if not isinstance(task_ids, str) or not task_ids.strip():
            return {
                'error': 'commit_planning: task_ids must be a non-empty comma-separated string',
                'error_type': 'ValidationError',
            }
        ids = [t.strip() for t in task_ids.split(',') if t.strip()]
        if not ids:
            return {
                'error': 'commit_planning: task_ids parsed to an empty list',
                'error_type': 'ValidationError',
            }
        for tid in ids:
            if _is_ticket_id(tid):
                return {
                    'error': (
                        f'commit_planning: task_ids must contain numeric task ids, '
                        f'not tickets (got {tid!r}). If you got a ticket from '
                        f'submit_task without planning_mode=True, use resolve_ticket first.'
                    ),
                    'error_type': 'ValidationError',
                }

        try:
            return await task_interceptor.set_task_status(
                task_id=','.join(ids),
                status=target_status,
                project_root=project_root,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'commit_planning error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def update_task(
        id: str,
        project_root: str,
        prompt: str | None = None,
        metadata: str | dict | None = None,
        append: bool | None = None,
        tag: str | None = None,
        metadata_mode: str | None = None,
        title: str | None = None,
        description: str | None = None,
        details: str | None = None,
        priority: str | None = None,
        status: str | None = None,
        dependencies: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update an existing task.

        Prefer structured fields (``title``, ``description``, ``details``,
        ``priority``, ``status``, ``dependencies``) — agents already have
        the full context needed to set them directly. Each non-None field
        overwrites the corresponding column.

        ``prompt`` is legacy: it routes through the LLM-driven Taskmaster
        path which can drift on re-rewrite. It will be removed once the
        sqlite cutover is complete.

        Args:
            id: Task ID to update
            project_root: Absolute path to project root
            prompt: DEPRECATED — pass structured fields instead.
            metadata: JSON metadata (object or JSON string). Default behavior
                is a shallow last-write-wins merge: ``{**existing, **incoming}``.
                Omitted keys from ``metadata`` are preserved; every supplied key
                (scalar or list) overwrites wholesale. Use ``metadata_mode`` to
                change this behavior.
            metadata_mode: Controls how ``metadata`` is merged with the existing
                blob. One of:
                - ``'merge'`` (default when omitted) — shallow last-write-wins.
                  Omitted keys preserved; supplied keys overwrite wholesale.
                - ``'additive'`` — recursive list union+dedup, scalar/type-collision
                  OLD-wins. Use for list-append callers (dry_run_proposals, etc.).
                - ``'replace'`` — whole-blob overwrite. Bypasses the corrupt-blob
                  guard; the sanctioned repair path.
            append: DEPRECATED shim. ``True`` → ``'additive'``, ``False`` →
                ``'replace'``. **Also the only knob that governs ``details``/
                ``prompt`` append** — ``metadata_mode`` does NOT affect the
                details path, so callers that need details-append must still
                pass ``append=True`` even after migrating metadata writes to
                ``metadata_mode``. Resolution is single-sourced in the backend.
            tag: Tag context (optional)
            title: New title (overwrites)
            description: New description (overwrites)
            details: New details (overwrites, or appends when ``append=True``)
            priority: New priority (e.g. "high"/"medium"/"low")
            status: New status (e.g. "pending"/"in-progress"/"done")
            dependencies: Replacement list of dependency task ids (top-level only)
        """
        if err := _reject_if_ticket_id('id', id):
            return err
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        if err := _reject_status_in_update_task(id, status):
            return err
        if err := _reject_done_provenance_in_metadata(metadata):
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
                metadata_mode=metadata_mode,
                tag=tag,
                title=title,
                description=description,
                details=details,
                priority=priority,
                status=status,
                dependencies=dependencies,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'update_task error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def remove_task(
        id: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Remove a task or subtask.

        Supported ID grammar
        --------------------
        Each id must be one of:
        - Top-level integer id, e.g. ``"292"``
        - 2-level dotted subtask id, e.g. ``"292.1"``

        3+-level nested ids (e.g. ``"1.2.3"``) are **not** supported and
        raise ``TaskmasterError(code='INVALID_TASK_ID')``.

        Multiple ids may be supplied as a comma-separated string
        (e.g. ``"292, 293.1"``); the wire boundary splits and strips them
        before forwarding a structured ``list[str]`` to the backend.

        Atomicity
        ---------
        All ids in the batch are parsed upfront.  A single malformed id
        causes the **entire** batch to fail with
        ``TaskmasterError(code='INVALID_TASK_ID')`` and **no** removals
        occur.  The error surfaces at the wire boundary as
        ``{'error': 'INVALID_TASK_ID: <reason>', 'error_type': 'TaskmasterError'}``.

        Success response shape
        ----------------------
        ``{'successful': int, 'failed': int, 'removed_ids': list[str], 'message': str}``

        Args:
            id: Task/subtask ID to remove (comma-separated for multiple).
                Each id must be a top-level integer (e.g. "292") or a
                2-level dotted subtask id (e.g. "292.1").
            project_root: Absolute path to project root
            tag: Tag context (optional)
        """
        if err := _reject_if_ticket_id('id', id):
            return err
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        # Wire shape advertises CSV for multi-removal; normalise to list[str]
        # at the boundary so backends speak a single, structured shape.
        ids = [s.strip() for s in id.split(',') if s.strip()]
        if not ids:
            return {
                'successful': 0,
                'failed': 0,
                'removed_ids': [],
                'message': 'no ids supplied',
            }
        try:
            return await task_interceptor.remove_tasks(ids=ids, project_root=project_root, tag=tag)
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'remove_task error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def add_dependency(
        id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Add a dependency between tasks.

        ``depends_on`` accepts two forms:

        - **Bare integer** (e.g. ``"5"``): records the dependency in the integer
          ``dependencies`` table — the original behavior.
        - **Qualified** (e.g. ``"dark_factory:13"``): records the dependency in
          the dependent task's ``metadata.external_deps`` list via an append-safe
          merge.  ``'-'`` in the project_id is normalised to ``'_'``.  The
          foreign target is **not** verified at write time (lenient); it may be
          filed later or live in another project.

        The qualified form is **not** a ``tkt_``-prefixed ticket id and is
        therefore not rejected by the ticket-id guard.

        Self-loops (same project + same task id) and malformed qualified strings
        raise ``TaskmasterError`` and surface as
        ``{'error': 'TASKMASTER_TOOL_ERROR: …', 'error_type': 'TaskmasterError'}``.

        Args:
            id: Task ID that will depend on another
            depends_on: Task ID that becomes a dependency.  Either a bare integer
                string (``"5"``) or a qualified cross-project reference
                (``"project_id:task_id"``).
            project_root: Absolute path to project root
            tag: Tag context (optional)
        """
        if err := _reject_if_ticket_id('id', id):
            return err
        if err := _reject_if_ticket_id('depends_on', depends_on):
            return err
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        try:
            return await task_interceptor.add_dependency(
                task_id=id,
                depends_on=depends_on,
                project_root=project_root,
                tag=tag,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'add_dependency error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def remove_dependency(
        id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Remove a dependency from a task.

        ``depends_on`` accepts two forms:

        - **Bare integer** (e.g. ``"5"``): issues an unconditional ``DELETE``
          from the integer ``dependencies`` table — the original behavior.
        - **Qualified** (e.g. ``"dark_factory:13"``): removes the canonical
          entry from the dependent task's ``metadata.external_deps`` list.
          ``'-'`` in the project_id is normalised to ``'_'``.  Idempotent —
          no error if the entry is absent.

        Args:
            id: Task ID to remove dependency from
            depends_on: Dependency to remove.  Either a bare integer string
                (``"5"``) or a qualified cross-project reference
                (``"project_id:task_id"``).
            project_root: Absolute path to project root
            tag: Tag context (optional)
        """
        if err := _reject_if_ticket_id('id', id):
            return err
        if err := _reject_if_ticket_id('depends_on', depends_on):
            return err
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        try:
            return await task_interceptor.remove_dependency(
                task_id=id,
                depends_on=depends_on,
                project_root=project_root,
                tag=tag,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'remove_dependency error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    # ------------------------------------------------------------------
    # Scheduler-override tools
    #
    # These four tools operate on
    # ``<project_root>/data/orchestrator/scheduler_overrides.db`` directly
    # via aiosqlite.  They duplicate the SQL semantics from
    # orchestrator.overrides.OverrideStore (no import) — the same
    # ossified-contract pattern as TERMINAL_STATUSES at
    # fused_memory/middleware/task_interceptor.py:108-112.
    #
    # Invariant: overrides live in scheduler_overrides.db, physically separate
    # from the tasks store.  set_task_status writes to the tasks store via
    # task_interceptor and the MCP-tool body does not touch override rows.
    # test_set_task_status_done_does_not_clear_override_row pins both:
    #   (1) delegation wiring — task_interceptor.set_task_status.assert_called_once()
    #   (2) cross-store separation — override row survives the status transition.
    # ------------------------------------------------------------------

    async def _emit_override_audit(
        project_root: str,
        tool_name: str,
        task_id: str | None,
        content: str,
        metadata: dict,
    ) -> None:
        """Best-effort audit write — awaited inline; failures are logged but never propagated.

        Mirrors the _log_read pattern at tools.py:272 — audit failures log
        a warning but never fail the user-visible tool response.
        """
        try:
            pid = resolve_project_id(project_root)
            await memory_service.add_memory(
                content=content,
                category='decisions_and_rationale',
                project_id=pid,
                agent_id='scheduler-overrides',
                metadata=metadata,
            )
        except Exception as audit_exc:
            logger.warning(
                'override audit emit failed (tool=%s task_id=%s): %s',
                tool_name,
                task_id,
                audit_exc,
            )

    @mcp.tool()
    async def set_task_priority_override(
        project_root: str,
        task_id: str,
        boost_tier: str | None = None,
        pinned: bool | None = None,
        pin_order: int | None = None,
        reserve_now: bool | None = None,
        ttl_secs: int | None = None,
    ) -> dict[str, Any]:
        """Upsert a scheduler priority override for a task.

        Only supplied (non-None) fields are written; unsupplied fields preserve
        existing values via COALESCE in the UPSERT.

        Args:
            project_root: Absolute path to project root.
            task_id: Task ID to override.
            boost_tier: One of critical/high/medium/low/polish.
            pinned: Pin this task to the front of the queue.
            pin_order: Explicit position in the pin queue (only with pinned=True).
            reserve_now: Reserve the next worker slot for this task.
            ttl_secs: Seconds from now until this override expires.
        """
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized

        if boost_tier is not None and boost_tier not in _PRIORITY_TIERS:
            return {
                'error': f'boost_tier must be one of {list(_PRIORITY_TIERS)}; got {boost_tier!r}',
                'error_type': 'ValidationError',
            }

        # Guard: pin_order without pinned=True is an invariant violation.
        # Mirrors orchestrator/src/orchestrator/overrides.py:151-155.
        if pin_order is not None and pinned is not True:
            return {
                'error': (
                    'pin_order may only be supplied together with pinned=True; '
                    f'got pin_order={pin_order!r} with pinned={pinned!r}'
                ),
                'error_type': 'ValidationError',
            }

        try:
            now_iso = datetime.now(UTC).isoformat()
            ttl_until_iso: str | None = None
            if ttl_secs is not None:
                from datetime import timedelta

                ttl_until_iso = (datetime.now(UTC) + timedelta(seconds=ttl_secs)).isoformat()

            db = await _open_overrides_db(project_root, autocommit=True)
            collision_response: dict[str, Any] | None = None
            try:
                # Acquire a write lock up-front (IMMEDIATE) so the MAX(pin_order)
                # read and the subsequent UPSERT are serialized against concurrent
                # set_task_priority_override callers.  Without BEGIN IMMEDIATE
                # two concurrent callers can both read MAX=N and both attempt
                # to write N+1, producing a collision or silently duplicating
                # pin_order values.  Mirrors source-of-truth at
                # orchestrator/src/orchestrator/overrides.py:188-195.
                await db.execute('BEGIN IMMEDIATE')
                try:
                    # Auto-assign or preserve pin_order when pinning without an
                    # explicit order. Mirrors orchestrator/src/orchestrator/overrides.py:160-180.
                    if pinned is True and pin_order is None:
                        already = await (
                            await db.execute(
                                'SELECT pin_order FROM overrides '
                                'WHERE project_root=? AND task_id=? AND pinned=1',
                                (project_root, task_id),
                            )
                        ).fetchone()
                        if already is not None:
                            pin_order = already[0]
                        else:
                            row = await (
                                await db.execute(
                                    'SELECT COALESCE(MAX(pin_order), 0) + 1 '
                                    'FROM overrides WHERE project_root=? AND pinned=1',
                                    (project_root,),
                                )
                            ).fetchone()
                            # Aggregate query always returns a row; assert for pyright.
                            assert row is not None
                            pin_order = row[0]

                    # Collision check for explicit or auto-assigned pin_order.
                    if pin_order is not None:
                        existing = await (
                            await db.execute(
                                'SELECT task_id FROM overrides '
                                'WHERE project_root=? AND pinned=1 AND pin_order=? AND task_id != ?',
                                (project_root, pin_order, task_id),
                            )
                        ).fetchone()
                        if existing:
                            await db.execute('ROLLBACK')
                            collision_response = {
                                'error': 'pin_order_collision',
                                'conflicting_task_id': existing[0],
                                'pin_order': pin_order,
                            }

                    if collision_response is None:
                        pinned_int = int(pinned) if pinned is not None else None
                        reserve_now_int = int(reserve_now) if reserve_now is not None else None

                        # CASE WHEN ?=0 THEN NULL ELSE COALESCE(?, pin_order) END
                        # — passing pinned=False (pinned_int=0) zeroes pinned AND
                        # nulls pin_order in one atomic write, preserving the
                        # structural invariant `pinned=0 → pin_order IS NULL`.
                        # Mirrors orchestrator/src/orchestrator/overrides.py:252-253.
                        await db.execute(
                            """
                            INSERT INTO overrides
                                (project_root, task_id, boost_tier, pinned, pin_order,
                                 reserve_now, ttl_until, created_at, updated_at)
                            VALUES (?, ?, ?, COALESCE(?, 0), ?, COALESCE(?, 0), ?, ?, ?)
                            ON CONFLICT(project_root, task_id) DO UPDATE SET
                                boost_tier  = COALESCE(?, boost_tier),
                                pinned      = COALESCE(?, pinned),
                                pin_order   = CASE WHEN ?=0 THEN NULL
                                                   ELSE COALESCE(?, pin_order) END,
                                reserve_now = COALESCE(?, reserve_now),
                                ttl_until   = COALESCE(?, ttl_until),
                                updated_at  = ?
                            """,
                            (
                                # INSERT values
                                project_root,
                                task_id,
                                boost_tier,
                                pinned_int,
                                pin_order,
                                reserve_now_int,
                                ttl_until_iso,
                                now_iso,
                                now_iso,
                                # UPDATE SET values
                                boost_tier,
                                pinned_int,
                                pinned_int,
                                pin_order,
                                reserve_now_int,
                                ttl_until_iso,
                                now_iso,
                            ),
                        )
                        await db.execute('COMMIT')
                except Exception:
                    with contextlib.suppress(Exception):
                        await db.execute('ROLLBACK')
                    raise
            finally:
                await db.close()

            if collision_response is not None:
                return collision_response

            # Build changed_fields for audit (use original ttl_secs, not derived absolute).
            changed_fields: dict[str, Any] = {}
            if boost_tier is not None:
                changed_fields['boost_tier'] = boost_tier
            if pinned is not None:
                changed_fields['pinned'] = pinned
            if pin_order is not None:
                # Intentionally the post-auto-assignment value — if pinned=True was
                # supplied without an explicit pin_order, this records the integer
                # that was actually written to the DB (auto-MAX+1 logic above).
                # Contrast with ttl_secs below, which is intentionally the raw
                # caller-supplied input rather than the derived ttl_until ISO string.
                changed_fields['pin_order'] = pin_order
            if reserve_now is not None:
                changed_fields['reserve_now'] = reserve_now
            if ttl_secs is not None:
                changed_fields['ttl_secs'] = ttl_secs

            await _emit_override_audit(
                project_root,
                'set_task_priority_override',
                task_id,
                f'Set priority override for task {task_id}: {changed_fields}',
                {'task_id': task_id, 'fields': changed_fields},
            )
            return {'success': True, 'task_id': task_id}
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'set_task_priority_override error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def clear_task_priority_override(
        project_root: str,
        task_id: str,
        field: str | None = None,
    ) -> dict[str, Any]:
        """Clear a priority override row or a single field within it.

        When ``field`` is None the entire row is deleted. Otherwise one
        field is nulled/zeroed. Valid field names: boost_tier, pinned,
        reserve_now, ttl. Clearing pinned also sets pin_order=NULL.

        Args:
            project_root: Absolute path to project root.
            task_id: Task ID whose override to clear.
            field: Field to clear, or None to delete the entire row.
        """
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized

        if field is not None and field not in _VALID_CLEAR_FIELDS:
            return {
                'error': (
                    f'field must be one of {sorted(_VALID_CLEAR_FIELDS)} or None; got {field!r}'
                ),
                'error_type': 'ValidationError',
            }

        try:
            now_iso = datetime.now(UTC).isoformat()
            db = await _open_overrides_db(project_root)
            try:
                if field is None:
                    await db.execute(
                        'DELETE FROM overrides WHERE project_root=? AND task_id=?',
                        (project_root, task_id),
                    )
                elif field == 'boost_tier':
                    await db.execute(
                        'UPDATE overrides SET boost_tier=NULL, updated_at=? '
                        'WHERE project_root=? AND task_id=?',
                        (now_iso, project_root, task_id),
                    )
                elif field == 'pinned':
                    # Clearing pinned also clears pin_order.
                    # Mirrors orchestrator/src/orchestrator/overrides.py:267-271.
                    await db.execute(
                        'UPDATE overrides SET pinned=0, pin_order=NULL, updated_at=? '
                        'WHERE project_root=? AND task_id=?',
                        (now_iso, project_root, task_id),
                    )
                elif field == 'reserve_now':
                    await db.execute(
                        'UPDATE overrides SET reserve_now=0, updated_at=? '
                        'WHERE project_root=? AND task_id=?',
                        (now_iso, project_root, task_id),
                    )
                else:  # 'ttl'
                    await db.execute(
                        'UPDATE overrides SET ttl_until=NULL, updated_at=? '
                        'WHERE project_root=? AND task_id=?',
                        (now_iso, project_root, task_id),
                    )
                await db.commit()
            finally:
                await db.close()

            label = 'all' if field is None else field
            await _emit_override_audit(
                project_root,
                'clear_task_priority_override',
                task_id,
                f'Cleared {label} priority override(s) for task {task_id}',
                {'task_id': task_id, 'field': field},
            )
            return {'success': True, 'task_id': task_id, 'field': field}
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'clear_task_priority_override error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def reorder_pin_queue(
        project_root: str,
        ordered_task_ids: list[str] | str,
    ) -> dict[str, Any]:
        """Rewrite pin_order values to match the supplied ordering.

        ``ordered_task_ids`` may be a list or a comma-separated string. The
        supplied set must exactly match the current pinned-task set.

        Args:
            project_root: Absolute path to project root.
            ordered_task_ids: New ordering — first element gets pin_order=1.

        Mirrors orchestrator/src/orchestrator/overrides.py:289-329.
        """
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized

        # Normalise CSV string input. Mirrors overrides.py:300-301.
        if isinstance(ordered_task_ids, str):
            ordered_task_ids = [t.strip() for t in ordered_task_ids.split(',') if t.strip()]

        # Duplicate-id guard MUST run before the set-equality check below,
        # otherwise an input like ['A','A','B','C'] when the current pin set is
        # {'A','B','C'} would pass set-equality and the rewrite loop would
        # silently produce pin_orders that don't match the user's request
        # (UPDATE A→1 then A→2, B→3, C→4 — creating gaps/duplicates depending
        # on order).  Mirrors orchestrator/src/orchestrator/overrides.py:352-355.
        if len(ordered_task_ids) != len(set(ordered_task_ids)):
            seen: set[str] = set()
            dups: list[str] = []
            for tid in ordered_task_ids:
                if tid in seen and tid not in dups:
                    dups.append(tid)
                seen.add(tid)
            return {
                'error': (
                    f'reorder_pin_queue: duplicate task ids in input: '
                    f'{ordered_task_ids!r} (duplicates: {sorted(dups)!r})'
                ),
                'error_type': 'ValidationError',
            }

        try:
            now_iso = datetime.now(UTC).isoformat()
            # Use autocommit=True so we can issue BEGIN IMMEDIATE, making the
            # set-equality SELECT and the rewrite loop a single atomic
            # read-then-write under a write lock.  Without BEGIN IMMEDIATE a
            # concurrent set_task_priority_override(pinned=True) call could pin
            # a new task between the SELECT and the UPDATE loop, invalidating
            # the set-equality invariant that was checked before the writes.
            # Mirrors set_task_priority_override's concurrency pattern above.
            db = await _open_overrides_db(project_root, autocommit=True)
            try:
                await db.execute('BEGIN IMMEDIATE')
                try:
                    # Set-equality check before writing.
                    # Mirrors orchestrator/src/orchestrator/overrides.py:303-312.
                    cursor = await db.execute(
                        'SELECT task_id FROM overrides WHERE project_root=? AND pinned=1',
                        (project_root,),
                    )
                    current_rows = await cursor.fetchall()
                    current_set = {r[0] for r in current_rows}
                    supplied_set = set(ordered_task_ids)
                    if supplied_set != current_set:
                        await db.execute('ROLLBACK')
                        return {
                            'error': (
                                f'reorder_pin_queue: ids do not match current pin queue. '
                                f'supplied={sorted(supplied_set)!r}, '
                                f'expected={sorted(current_set)!r}'
                            ),
                            'error_type': 'ValidationError',
                        }

                    # Single-transaction rewrite. Mirrors overrides.py:318-326.
                    for idx, tid in enumerate(ordered_task_ids, start=1):
                        await db.execute(
                            'UPDATE overrides SET pin_order=?, updated_at=? '
                            'WHERE project_root=? AND task_id=?',
                            (idx, now_iso, project_root, tid),
                        )
                    await db.execute('COMMIT')
                except Exception:
                    with contextlib.suppress(Exception):
                        await db.execute('ROLLBACK')
                    raise
            finally:
                await db.close()

            await _emit_override_audit(
                project_root,
                'reorder_pin_queue',
                None,
                f'Reordered pin queue: {ordered_task_ids}',
                {'ordered_task_ids': list(ordered_task_ids)},
            )
            return {'success': True}
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'reorder_pin_queue error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_pin_queue(
        project_root: str,
    ) -> dict[str, Any]:
        """Return the current pinned-task queue in ascending pin_order.

        Read-only.  Does NOT emit an audit add_memory call.

        Args:
            project_root: Absolute path to project root.

        Returns:
            ``{'pin_queue': [{'task_id': ..., 'boost_tier': ..., 'pinned': ...,
            'pin_order': ..., 'reserve_now': ..., 'ttl_until': ...}, ...]}``
        """
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        try:
            db = await _open_overrides_db(project_root)
            try:
                cursor = await db.execute(
                    'SELECT task_id, boost_tier, pinned, pin_order, reserve_now, ttl_until '
                    'FROM overrides '
                    'WHERE project_root=? AND pinned=1 '
                    'ORDER BY pin_order ASC',
                    (project_root,),
                )
                rows = await cursor.fetchall()
                pin_queue = [
                    {
                        'task_id': r[0],
                        'boost_tier': r[1],
                        'pinned': bool(r[2]),
                        'pin_order': r[3],
                        'reserve_now': bool(r[4]),
                        'ttl_until': r[5],
                    }
                    for r in rows
                ]
                return {'pin_queue': pin_queue}
            finally:
                await db.close()
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_pin_queue error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_scheduler_state(
        project_root: str,
    ) -> dict[str, Any]:
        """Return the latest in-memory scheduler state snapshot.

        The snapshot is written atomically by the orchestrator after every
        ``acquire_next`` tick.  This tool reads the JSON file from disk;
        it never touches the live scheduler process.

        Read-only.  Does NOT emit an audit add_memory call.

        Args:
            project_root: Absolute path to project root.

        Returns:
            Snapshot dict with keys: skip_counts, parks, effective_priorities,
            pin_queue, overrides, current_holders, snapshot_at.
            Returns the empty skeleton when no snapshot file exists yet.
        """
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        try:
            return await asyncio.to_thread(read_scheduler_state, Path(project_root))
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_scheduler_state error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    @mcp.tool()
    async def get_scheduler_events(
        project_root: str,
        since: str | None = None,
        limit: int = 200,
        event_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a tail of scheduler events from runs.db, newest-first.

        Reads ``<project_root>/data/orchestrator/runs.db`` in read-only mode
        via aiosqlite.  Never mutates data.  Does NOT emit an audit add_memory
        call.

        Args:
            project_root: Absolute path to project root.
            since: Optional ISO8601 lower bound (inclusive) on event timestamp.
            limit: Maximum number of events to return (default 200).
            event_types: Optional list of EventType values to include; if
                omitted all event types are returned.

        Returns:
            ``{'events': [...], 'count': <int>}`` where each event is a dict
            with keys: id, timestamp, run_id, task_id, event_type, data.
            Returns ``{'events': [], 'count': 0}`` when runs.db is missing.
        """
        _normalized = _normalize_project_root(project_root)
        if isinstance(_normalized, dict):
            return _normalized
        project_root = _normalized
        try:
            return await read_scheduler_events(
                Path(project_root),
                since=since,
                limit=limit,
                event_types=event_types,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.exception(f'get_scheduler_events error: {e}')
            return {'error': str(e), 'error_type': type(e).__name__}

    return mcp
