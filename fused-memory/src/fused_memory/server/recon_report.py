"""recon_report MCP namespace — in-process state + tool scaffold (task α/β).

Provides nine tools: start_report / add_finding / set_stat / inc_stat / complete /
cite_entity / cite_edge / cite_task / cite_memory.
State is owned by :class:`ReconReportState`; tools are thin delegates registered
by :func:`create_recon_report_server`.  This split lets unit tests drive the state
directly without spinning up FastMCP.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from graphiti_core.errors import EdgeNotFoundError

from fused_memory.services.memory_service import MemoryNotFoundError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------


@dataclass
class _Finding:
    finding_id: str  # 36-char uuid4
    severity: str
    category: str
    description: str
    suggested_action: str
    actionable: bool
    task_id: str | None
    flag_type: str | None
    cited_entities: list[dict] = field(default_factory=list)  # populated by task β
    cited_edges: list[dict] = field(default_factory=list)
    cited_tasks: list[dict] = field(default_factory=list)
    cited_memories: list[dict] = field(default_factory=list)


@dataclass
class _ReportEntry:
    run_id: str
    stage: str
    project_id: str
    findings: list[_Finding] = field(default_factory=list)
    stats: dict[str, int | float | str] = field(default_factory=dict)
    summary: str = ''
    summary_warnings: list[str] = field(default_factory=list)
    completed_at: float | None = None  # clock() — None when in-progress
    created_at: float = 0.0  # clock() — for diagnostics
    # in-run dedup: (task_id, flag_type) → finding_id
    _signature_to_finding: dict[tuple[str | None, str | None], str] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

_ERR_RUN_UNKNOWN: dict[str, str] = {
    'error': 'run_id_unknown',
    'error_type': 'ReconReportRunUnknown',
}

# Returned when add_finding / set_stat / inc_stat are called after complete()
# has already stamped completed_at.  complete() is documented as the closing
# call; mutating a completed entry would silently corrupt the assembled report
# and the next complete()'s cached flagged_count/stats.
_ERR_ALREADY_COMPLETED: dict[str, str] = {
    'error': 'report_already_completed',
    'error_type': 'ReconReportAlreadyCompleted',
}


def _duplicate_finding_error(existing_id: str) -> dict[str, str]:
    return {
        'error': 'duplicate_finding',
        'error_type': 'ReconReportDuplicateFinding',
        'existing_finding_id': existing_id,
    }


def _stat_type_mismatch_error(key: str) -> dict[str, str]:
    return {
        'error': 'stat_type_mismatch',
        'error_type': 'ReconReportStatTypeMismatch',
        'key': key,
        'current_type': 'str',
    }


# ---------------------------------------------------------------------------
# cite_* error helpers (task β)
# ---------------------------------------------------------------------------

# Compiled UUID shape gate: ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$
# re.IGNORECASE: Graphiti/Neo4j and mem0 do not normalise UUID case on read-back,
# and Python's stdlib uuid.UUID accepts mixed case — rejecting uppercase here would
# mask real edges/memories as malformed before the service is even called.
_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)

_ERR_FINDING_UNKNOWN: dict[str, str] = {
    'error': 'finding_unknown',
    'error_type': 'ReconReportFindingUnknown',
}

_ERR_ENTITY_NOT_FOUND: dict[str, str] = {
    'error': 'entity_not_found',
    'error_type': 'ReconReportEntityNotFound',
}

_ERR_EDGE_NOT_FOUND: dict[str, str] = {
    'error': 'edge_not_found',
    'error_type': 'ReconReportEdgeNotFound',
}

_ERR_TASK_NOT_FOUND: dict[str, str] = {
    'error': 'task_not_found',
    'error_type': 'ReconReportTaskNotFound',
}

_ERR_UNKNOWN_PROJECT: dict[str, str] = {
    'error': 'unknown_project',
    'error_type': 'ReconReportUnknownProject',
}

_ERR_MEMORY_NOT_FOUND: dict[str, str] = {
    'error': 'memory_not_found',
    'error_type': 'ReconReportMemoryNotFound',
}

_ERR_INVALID_UUID_SHAPE: dict[str, str] = {
    'error': 'invalid_uuid_shape',
    'error_type': 'ReconReportInvalidUuid',
}

_ERR_SERVICE_UNAVAILABLE: dict[str, str] = {
    'error': 'service_not_configured',
    'error_type': 'ReconReportServiceUnavailable',
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ReconReportState:
    """In-process state store for the recon_report MCP namespace.

    Args:
        ttl_seconds:    Entries are evicted ``ttl_seconds`` after ``complete()``
                        stamps ``completed_at``.  In-progress entries are never
                        evicted (per PRD §9.4).
        clock:          Callable returning the current time as a float.
                        Defaults to ``asyncio.get_running_loop().time``.
                        Inject a fake clock in tests.
        reaper_interval: How many seconds the background reaper sleeps between
                        sweeps.  Default 60.
    """

    def __init__(
        self,
        ttl_seconds: float,
        clock: Callable[[], float] | None = None,
        reaper_interval: float = 60.0,
        memory_service: Any = None,
        task_interceptor: Any = None,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock_fn = clock
        self._reaper_interval = reaper_interval
        self._state: dict[tuple[str, str], _ReportEntry] = {}
        self._active: dict[str, str] = {}  # run_id → current stage
        self._reaper_task: asyncio.Task | None = None
        # cite_* service injection (task β)
        self._memory_service = memory_service
        self._task_interceptor = task_interceptor
        self.known_projects: dict[str, str] = {}  # project_id → project_root

    def _clock(self) -> float:
        if self._clock_fn is not None:
            return self._clock_fn()
        return asyncio.get_running_loop().time()

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def start_report(
        self,
        run_id: str,
        stage: str,
        project_id: str,
    ) -> dict[str, Any]:
        """Create a new in-progress report entry."""
        entry = _ReportEntry(
            run_id=run_id,
            stage=stage,
            project_id=project_id,
            created_at=self._clock(),
        )
        self._state[(run_id, stage)] = entry
        self._active[run_id] = stage
        return {'run_id': run_id, 'stage': stage}

    def add_finding(
        self,
        run_id: str,
        severity: str,
        category: str,
        description: str,
        suggested_action: str,
        actionable: bool = True,
        task_id: str | None = None,
        flag_type: str | None = None,
    ) -> dict[str, Any]:
        """Append a finding to the current report entry, with in-run dedup.

        In-run dedup (PRD §9.2) is scoped to the ``run_id`` across ALL stages
        of the same run.  If (task_id, flag_type) was already filed by any
        earlier stage of this run, ``duplicate_finding`` is returned with the
        original finding_id — Stage 2 can then attach citations to that finding
        rather than creating a redundant row.  Cross-run isolation is preserved:
        findings from a different run_id are never considered.
        """
        entry = self._resolve_entry(run_id)
        if entry is None:
            return _ERR_RUN_UNKNOWN.copy()

        # Guard: reject mutations after complete() to prevent silent corruption
        # of the assembled report.  cite_* tools (task β) operate on _Finding
        # objects directly and are not affected by this guard.
        if entry.completed_at is not None:
            logger.warning(
                'recon_report: add_finding called after complete() for run_id=%r stage=%r; rejected',
                run_id,
                entry.stage,
            )
            return _ERR_ALREADY_COMPLETED.copy()

        # In-run dedup: skip when both are None (informational findings).
        # Scan ALL stages of this run_id so Stage 2 cannot duplicate Stage 1's
        # findings.  Each stage's _signature_to_finding map remains the per-stage
        # source of truth; the scan reads across them while filtering strictly on
        # run_id to preserve cross-run isolation.
        sig = (task_id, flag_type)
        if sig != (None, None):
            for (rid, _stage), other in self._state.items():
                if rid == run_id:
                    existing_id = other._signature_to_finding.get(sig)
                    if existing_id is not None:
                        return _duplicate_finding_error(existing_id)

        finding_id = str(uuid.uuid4())
        finding = _Finding(
            finding_id=finding_id,
            severity=severity,
            category=category,
            description=description,
            suggested_action=suggested_action,
            actionable=actionable,
            task_id=task_id,
            flag_type=flag_type,
        )
        entry.findings.append(finding)
        if sig != (None, None):
            entry._signature_to_finding[sig] = finding_id

        return {'finding_id': finding_id}

    def set_stat(
        self,
        run_id: str,
        key: str,
        value: int | float | str,
    ) -> dict[str, Any]:
        """Set a named statistic on the current report entry."""
        entry = self._resolve_entry(run_id)
        if entry is None:
            return _ERR_RUN_UNKNOWN.copy()

        if entry.completed_at is not None:
            logger.warning(
                'recon_report: set_stat called after complete() for run_id=%r stage=%r key=%r; rejected',
                run_id,
                entry.stage,
                key,
            )
            return _ERR_ALREADY_COMPLETED.copy()

        entry.stats[key] = value
        return {'value': value}

    def inc_stat(
        self,
        run_id: str,
        key: str,
        delta: int | float = 1,
    ) -> dict[str, Any]:
        """Increment a named statistic (initialises to 0 if absent)."""
        entry = self._resolve_entry(run_id)
        if entry is None:
            return _ERR_RUN_UNKNOWN.copy()

        if entry.completed_at is not None:
            logger.warning(
                'recon_report: inc_stat called after complete() for run_id=%r stage=%r key=%r; rejected',
                run_id,
                entry.stage,
                key,
            )
            return _ERR_ALREADY_COMPLETED.copy()

        current_raw = entry.stats.get(key, 0)
        # Guard: if a caller has previously set this key to a string via
        # set_stat and now calls inc_stat, silently coercing to 0 would lose
        # the original value with no diagnostic.  Return a structured error
        # so the caller can diagnose the key-type conflict explicitly.
        if isinstance(current_raw, str):
            logger.warning(
                'recon_report: inc_stat called on string-valued stat key=%r '
                'for run_id=%r stage=%r; use set_stat to replace it',
                key,
                run_id,
                entry.stage,
            )
            return _stat_type_mismatch_error(key)
        new_value = current_raw + delta
        entry.stats[key] = new_value
        return {'value': new_value}

    def complete(
        self,
        run_id: str,
        summary: str,
    ) -> dict[str, Any]:
        """Stamp the entry as complete.  Idempotent per PRD §9.2."""
        entry = self._resolve_entry(run_id)
        if entry is None:
            return _ERR_RUN_UNKNOWN.copy()

        cached_response = {
            'flagged_count': len(entry.findings),
            'stats': dict(entry.stats),
        }

        if entry.completed_at is not None:
            # Already completed
            if summary == entry.summary:
                # Exact same summary — pure no-op
                return cached_response
            else:
                # Different summary — warn, do NOT overwrite
                warning = (
                    f'complete() called again with a different summary '
                    f'(original kept). Incoming: {summary!r}'
                )
                entry.summary_warnings.append(warning)
                logger.warning(
                    'recon_report: duplicate complete() with different summary '
                    'for run_id=%r stage=%r; original summary preserved',
                    run_id,
                    entry.stage,
                )
                return cached_response

        # First-time path
        entry.completed_at = self._clock()
        entry.summary = summary
        return cached_response

    def get_assembled_report(
        self,
        run_id: str,
        stage: str,
    ) -> dict[str, Any] | None:
        """Return the §9.3 assembled report dict, or None if unknown."""
        entry = self._state.get((run_id, stage))
        if entry is None:
            return None

        flagged_items = [
            {
                'finding_id': f.finding_id,
                'severity': f.severity,
                'category': f.category,
                'description': f.description,
                'suggested_action': f.suggested_action,
                'actionable': f.actionable,
                'task_id': f.task_id,
                'flag_type': f.flag_type,
                # Empty in α; task β populates via cite_* tools
                'cited_entities': list(f.cited_entities),
                'cited_edges': list(f.cited_edges),
                'cited_tasks': list(f.cited_tasks),
                'cited_memories': list(f.cited_memories),
            }
            for f in entry.findings
        ]
        return {
            'summary': entry.summary,
            'stats': dict(entry.stats),
            'flagged_items': flagged_items,
            'summary_warnings': list(entry.summary_warnings),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_entry(self, run_id: str) -> _ReportEntry | None:
        stage = self._active.get(run_id)
        if stage is None:
            return None
        return self._state.get((run_id, stage))

    def _resolve_finding(
        self, run_id: str, finding_id: str
    ) -> tuple[_ReportEntry, _Finding] | None:
        """Return (entry, finding) or None if either run_id or finding_id is unknown.

        The scan covers ALL stage entries that share this ``run_id``, so a
        finding_id returned via a cross-stage ``duplicate_finding`` response
        (where the original finding lives in an earlier stage's entry) remains
        citable from a later stage.  The lookup is still strictly scoped to
        ``run_id`` to preserve cross-run isolation.
        """
        for (rid, _stage), entry in self._state.items():
            if rid != run_id:
                continue
            for f in entry.findings:
                if f.finding_id == finding_id:
                    return entry, f
        return None  # finding_id not in this run

    # ------------------------------------------------------------------
    # cite_* tools (task β)
    # ------------------------------------------------------------------

    async def cite_entity(
        self,
        run_id: str,
        finding_id: str,
        name: str,
    ) -> dict[str, Any]:
        """Resolve *name* to a Graphiti entity and record the citation.

        Returns {entity_uuid, canonical_name} on success, or a structured
        error dict (run_id_unknown / finding_unknown / entity_not_found).
        Appends to finding.cited_entities only on success — never on error.
        """
        entry = self._resolve_entry(run_id)
        if entry is None:
            return _ERR_RUN_UNKNOWN.copy()

        resolved = self._resolve_finding(run_id, finding_id)
        if resolved is None:
            return _ERR_FINDING_UNKNOWN.copy()
        _, finding = resolved

        if self._memory_service is None:
            return _ERR_SERVICE_UNAVAILABLE.copy()

        result = await self._memory_service.get_entity(name, entry.project_id)
        nodes = result.get('nodes', [])
        if not nodes:
            return _ERR_ENTITY_NOT_FOUND.copy()

        node = nodes[0]
        citation = {'entity_uuid': node['uuid'], 'canonical_name': node['name']}
        finding.cited_entities.append(citation)
        return citation

    async def cite_edge(
        self,
        run_id: str,
        finding_id: str,
        edge_uuid: str,
    ) -> dict[str, Any]:
        """Validate *edge_uuid* shape, fetch the edge, and record the citation.

        Returns {edge_uuid, fact_text_snapshot} on success, or a structured
        error dict (run_id_unknown / finding_unknown / invalid_uuid_shape /
        edge_not_found).  UUID shape is checked before any service call.
        Appends to finding.cited_edges only on success.
        """
        entry = self._resolve_entry(run_id)
        if entry is None:
            return _ERR_RUN_UNKNOWN.copy()

        resolved = self._resolve_finding(run_id, finding_id)
        if resolved is None:
            return _ERR_FINDING_UNKNOWN.copy()
        _, finding = resolved

        if not _UUID_RE.match(edge_uuid):
            return _ERR_INVALID_UUID_SHAPE.copy()

        if self._memory_service is None:
            return _ERR_SERVICE_UNAVAILABLE.copy()

        try:
            result = await self._memory_service.get_edge(edge_uuid, entry.project_id)
        except EdgeNotFoundError:
            return _ERR_EDGE_NOT_FOUND.copy()

        citation = {'edge_uuid': edge_uuid, 'fact_text_snapshot': result['fact']}
        finding.cited_edges.append(citation)
        return citation

    async def cite_task(
        self,
        run_id: str,
        finding_id: str,
        project_id: str,
        task_id: str,
    ) -> dict[str, Any]:
        """Look up *task_id* in *project_id* and record the citation.

        Returns {project_id, task_id, title} on success, or a structured error
        dict (run_id_unknown / finding_unknown / unknown_project / task_not_found).
        project_id is validated against self.known_projects before any service call.
        Appends to finding.cited_tasks only on success.
        """
        entry = self._resolve_entry(run_id)
        if entry is None:
            return _ERR_RUN_UNKNOWN.copy()

        resolved = self._resolve_finding(run_id, finding_id)
        if resolved is None:
            return _ERR_FINDING_UNKNOWN.copy()
        _, finding = resolved

        if project_id not in self.known_projects:
            return _ERR_UNKNOWN_PROJECT.copy()

        if self._task_interceptor is None:
            return _ERR_SERVICE_UNAVAILABLE.copy()

        project_root = self.known_projects[project_id]
        result = await self._task_interceptor.get_task(task_id, project_root)

        if not result or 'error' in result:
            return _ERR_TASK_NOT_FOUND.copy()

        # Guard against data=None (some get_task paths return data: null explicitly)
        data = result.get('data') if isinstance(result.get('data'), dict) else {}
        title = result.get('title') or data.get('title', '')
        citation = {'project_id': project_id, 'task_id': task_id, 'title': title}
        finding.cited_tasks.append(citation)
        return citation

    async def cite_memory(
        self,
        run_id: str,
        finding_id: str,
        memory_id: str,
        store: str,
    ) -> dict[str, Any]:
        """Validate *memory_id* shape, fetch fingerprint, and record the citation.

        Returns {memory_id, store, metadata_fingerprint} on success, or a structured
        error dict (run_id_unknown / finding_unknown / invalid_uuid_shape /
        memory_not_found).  UUID shape is checked before any service call.
        Appends to finding.cited_memories only on success.
        """
        entry = self._resolve_entry(run_id)
        if entry is None:
            return _ERR_RUN_UNKNOWN.copy()

        resolved = self._resolve_finding(run_id, finding_id)
        if resolved is None:
            return _ERR_FINDING_UNKNOWN.copy()
        _, finding = resolved

        if not _UUID_RE.match(memory_id):
            return _ERR_INVALID_UUID_SHAPE.copy()

        if self._memory_service is None:
            return _ERR_SERVICE_UNAVAILABLE.copy()

        try:
            fingerprint = await self._memory_service.get_memory(
                memory_id, store, entry.project_id
            )
        except (EdgeNotFoundError, MemoryNotFoundError):
            return _ERR_MEMORY_NOT_FOUND.copy()

        citation = {'memory_id': memory_id, 'store': store, 'metadata_fingerprint': fingerprint}
        finding.cited_memories.append(citation)
        return citation

    # ------------------------------------------------------------------
    # Reaper
    # ------------------------------------------------------------------

    def tick(self) -> int:
        """Sweep completed entries past TTL.  Returns count evicted."""
        now = self._clock()
        to_evict = [
            (rid, stage)
            for (rid, stage), entry in self._state.items()
            if entry.completed_at is not None
            and now - entry.completed_at > self._ttl_seconds
        ]
        for key in to_evict:
            rid, stage = key
            del self._state[key]
            # Remove _active pointer only if it still points at this stage
            if self._active.get(rid) == stage:
                del self._active[rid]
        if to_evict:
            logger.debug('recon_report reaper evicted %d entries', len(to_evict))
        return len(to_evict)

    async def _reaper_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._reaper_interval)
                self.tick()
        except asyncio.CancelledError:
            pass

    async def start_reaper(self) -> None:
        """Start the background reaper task (called by run_server, not the factory)."""
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.get_running_loop().create_task(
                self._reaper_loop(),
                name='recon_report_reaper',
            )

    async def stop_reaper(self) -> None:
        """Cancel and await the reaper task."""
        if self._reaper_task is not None and not self._reaper_task.done():
            self._reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper_task
            self._reaper_task = None


# ---------------------------------------------------------------------------
# FastMCP server factory
# ---------------------------------------------------------------------------

RECON_REPORT_INSTRUCTIONS = """\
This server provides the recon_report MCP namespace for the Dark Factory
reconciliation pipeline.

Tools: start_report, add_finding, set_stat, inc_stat, complete,
       cite_entity, cite_edge, cite_task, cite_memory.

Usage pattern (per PRD §9.2):
1. start_report — open a new report at the start of a stage run.
2. add_finding — append a diagnostic finding (deduplicated by task_id + flag_type
                  across ALL stages of the same run_id).
3. set_stat / inc_stat — track numeric metrics during the run.
4. complete — stamp the summary and close the report; idempotent.

Citation tools (call after add_finding, before or after complete):
5. cite_entity(run_id, finding_id, name) — resolve entity by name and attach.
6. cite_edge(run_id, finding_id, edge_uuid) — validate UUID and attach edge.
7. cite_task(run_id, finding_id, project_id, task_id) — look up task and attach.
8. cite_memory(run_id, finding_id, memory_id, store) — look up memory and attach.
"""


def create_recon_report_server(state: ReconReportState):  # -> FastMCP
    """Build and return a FastMCP('Recon Report') instance wired to *state*.

    The returned server is NOT started here — call ``run_server()`` to bind
    it to the network.  This keeps the factory testable without sockets.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP('Recon Report', instructions=RECON_REPORT_INSTRUCTIONS)

    @mcp.tool()
    async def start_report(run_id: str, stage: str, project_id: str) -> dict:
        """Open a new in-progress report for this stage run.

        PRD §9.2 — start_report(run_id, stage, project_id).
        Returns {run_id, stage}.
        """
        return state.start_report(run_id=run_id, stage=stage, project_id=project_id)

    @mcp.tool()
    async def add_finding(
        run_id: str,
        severity: str,
        category: str,
        description: str,
        suggested_action: str,
        actionable: bool = True,
        task_id: str | None = None,
        flag_type: str | None = None,
    ) -> dict:
        """Append a diagnostic finding.

        PRD §9.2 — add_finding(run_id, severity, category, description,
                                suggested_action, actionable, task_id, flag_type).
        Returns {finding_id} or a structured duplicate/error dict.
        In-run dedup: if (task_id, flag_type) was already reported by ANY stage
        of this run_id (and neither is None), returns
        {error: duplicate_finding, existing_finding_id}.  Attach citations to
        existing_finding_id instead of creating a redundant row.
        """
        return state.add_finding(
            run_id=run_id,
            severity=severity,
            category=category,
            description=description,
            suggested_action=suggested_action,
            actionable=actionable,
            task_id=task_id,
            flag_type=flag_type,
        )

    @mcp.tool()
    async def set_stat(run_id: str, key: str, value: float) -> dict:
        """Set a named statistic on the current report entry.

        PRD §9.2 — set_stat(run_id, key, value).
        Returns {value: <new_value>}.
        """
        return state.set_stat(run_id=run_id, key=key, value=value)

    @mcp.tool()
    async def inc_stat(run_id: str, key: str, delta: float = 1) -> dict:
        """Increment a named statistic (initialised to 0 if absent).

        PRD §9.2 — inc_stat(run_id, key, delta=1).
        Returns {value: <new_value>}.
        """
        return state.inc_stat(run_id=run_id, key=key, delta=delta)

    @mcp.tool()
    async def complete(run_id: str, summary: str) -> dict:
        """Stamp the entry as complete and write the summary.

        PRD §9.2 — complete(run_id, summary).
        Returns {flagged_count, stats}.
        Idempotent: repeated call with same summary is a no-op; different
        summary appends a warning but does NOT overwrite the original.
        """
        return state.complete(run_id=run_id, summary=summary)

    @mcp.tool()
    async def cite_entity(run_id: str, finding_id: str, name: str) -> dict:
        """Resolve a Graphiti entity by name and attach it to a finding.

        PRD §9.2 (task β) — cite_entity(run_id, finding_id, name).
        Returns {entity_uuid, canonical_name} or a structured error dict.
        entity_not_found when the name resolves to no nodes.
        """
        return await state.cite_entity(run_id=run_id, finding_id=finding_id, name=name)

    @mcp.tool()
    async def cite_edge(run_id: str, finding_id: str, edge_uuid: str) -> dict:
        """Validate an edge UUID shape and attach the edge fact to a finding.

        PRD §9.2 (task β) — cite_edge(run_id, finding_id, edge_uuid).
        Returns {edge_uuid, fact_text_snapshot} or a structured error dict.
        invalid_uuid_shape when edge_uuid doesn't match the canonical UUID regex.
        edge_not_found when the UUID is valid but not in the graph.
        """
        return await state.cite_edge(
            run_id=run_id, finding_id=finding_id, edge_uuid=edge_uuid
        )

    @mcp.tool()
    async def cite_task(
        run_id: str, finding_id: str, project_id: str, task_id: str
    ) -> dict:
        """Look up a task and attach it to a finding.

        PRD §9.2 (task β) — cite_task(run_id, finding_id, project_id, task_id).
        Both project_id and task_id are required; omitting either raises a
        validation error (PRD D4 / P4 boundary guard).
        Returns {project_id, task_id, title} or a structured error dict.
        unknown_project when project_id is not in the known_projects registry.
        task_not_found when the task does not exist in the project.
        """
        return await state.cite_task(
            run_id=run_id,
            finding_id=finding_id,
            project_id=project_id,
            task_id=task_id,
        )

    @mcp.tool()
    async def cite_memory(
        run_id: str,
        finding_id: str,
        memory_id: str,
        store: Literal['graphiti', 'mem0'],
    ) -> dict:
        """Validate a memory UUID shape and attach the memory fingerprint to a finding.

        PRD §9.2 (task β) — cite_memory(run_id, finding_id, memory_id, store).
        Returns {memory_id, metadata_fingerprint} or a structured error dict.
        invalid_uuid_shape when memory_id doesn't match the canonical UUID regex.
        memory_not_found when the UUID is valid but the memory doesn't exist.
        """
        return await state.cite_memory(
            run_id=run_id, finding_id=finding_id, memory_id=memory_id, store=store
        )

    return mcp
