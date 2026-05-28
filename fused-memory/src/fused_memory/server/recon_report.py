"""recon_report MCP namespace — in-process state + tool scaffold (task α).

Provides five tools: start_report / add_finding / set_stat / inc_stat / complete.
State is owned by :class:`ReconReportState`; tools are thin delegates registered
by :func:`create_recon_report_server`.  This split lets unit tests drive the state
directly without spinning up FastMCP.

cite_entity / cite_edge / cite_task / cite_memory are explicitly out of scope
for task α — see task β.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock_fn = clock
        self._reaper_interval = reaper_interval
        self._state: dict[tuple[str, str], _ReportEntry] = {}
        self._active: dict[str, str] = {}  # run_id → current stage
        self._reaper_task: asyncio.Task | None = None

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
        """Append a finding to the current report entry, with in-run dedup."""
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

        # In-run dedup: skip when both are None (informational findings)
        sig = (task_id, flag_type)
        if sig != (None, None):
            existing_id = entry._signature_to_finding.get(sig)
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

Tools: start_report, add_finding, set_stat, inc_stat, complete.

Usage pattern (per PRD §9.2):
1. start_report — open a new report at the start of a stage run.
2. add_finding — append a diagnostic finding (deduplicated by task_id + flag_type).
3. set_stat / inc_stat — track numeric metrics during the run.
4. complete — stamp the summary and close the report; idempotent.
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
        In-run dedup: if (task_id, flag_type) was already reported (and
        neither is None), returns {error: duplicate_finding, existing_finding_id}.
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

    return mcp
