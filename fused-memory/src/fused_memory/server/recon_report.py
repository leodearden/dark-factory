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
import hashlib
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from graphiti_core.errors import EdgeNotFoundError

from fused_memory.services.memory_service import MemoryNotFoundError

logger = logging.getLogger(__name__)


def _normalize_description(description: str) -> str:
    """Collapse whitespace and casefold for description dedup."""
    return ' '.join(description.split()).casefold()


def _description_hash(description: str) -> str:
    """SHA-256 hex digest of the normalized description."""
    return hashlib.sha256(_normalize_description(description).encode('utf-8')).hexdigest()


def _canonical_sig_field(value: Any) -> str | None:
    """Coerce an in-run dedup signature field (task_id or flag_type) to ``str``.

    Mirrors ``reconciliation.flag_dedup.compute_flag_signature``'s str-coercion,
    which exists because "integer task_id is common in LLM output" — without
    it, ``(1976, ft) != ('1976', ft)`` and two calls describing the same
    logical (task_id, flag_type) produce distinct dedup keys (task-1979 /
    run 9542fa10).  Re-implemented locally rather than imported because
    server/ must not import from the reconciliation package — see
    ``_normalize_description`` above for the same local-copy convention.

    ``None`` is preserved as ``None`` (never coerced to the string ``'None'``)
    so the ``(None, None)`` null-null desc-hash sentinel and partial-None
    signature routing in :meth:`ReconReportState.add_finding` are unaffected.
    """
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# Fix 1 helpers — citation identity + same-run echo suppression (task-1654)
# ---------------------------------------------------------------------------


def _citation_identities(finding: dict) -> set[str]:
    """Return the set of identity strings for a finding's typed citations.

    Flattens all four citation lists into a single flat set of stable identity
    strings:
    - ``cited_tasks``    → ``'{project_id}:{task_id}'`` per entry
    - ``cited_entities`` → ``entity_uuid`` per entry
    - ``cited_edges``    → ``edge_uuid`` per entry
    - ``cited_memories`` → ``memory_id`` per entry

    Mirrors :func:`harness._derive_affected_ids` traversal but:
    - uses ``project_id:task_id`` for tasks (cross-project disambiguation)
    - returns a ``set`` (not a list) so subset tests are O(n) not O(n²)
    - lives in server/recon_report.py to avoid a server←reconciliation import

    Used by :func:`_traces_exclusively_to_stage1` and
    :func:`ReconReportState.get_assembled_report` for Fix-1 echo suppression.
    Pure, sync, no I/O; safe to call from any context.
    """
    ids: set[str] = set()
    for c in finding.get('cited_tasks') or []:
        if isinstance(c, dict):
            pid = c.get('project_id')
            tid = c.get('task_id')
            if pid is not None and tid is not None:
                ids.add(f'{pid}:{tid}')
    for c in finding.get('cited_entities') or []:
        if isinstance(c, dict):
            val = c.get('entity_uuid')
            if val is not None:
                ids.add(str(val))
    for c in finding.get('cited_edges') or []:
        if isinstance(c, dict):
            val = c.get('edge_uuid')
            if val is not None:
                ids.add(str(val))
    for c in finding.get('cited_memories') or []:
        if isinstance(c, dict):
            val = c.get('memory_id')
            if val is not None:
                ids.add(str(val))
    return ids


def _traces_exclusively_to_stage1(
    finding: dict,
    stage1_identities: set[str],
) -> bool:
    """Return True iff *finding* is a non-actionable echo of Stage-1 citations.

    Predicate: True iff ALL of the following hold:
    1. ``finding['actionable']`` is False
    2. The finding's citation identity set is non-empty (has >=1 typed citation)
    3. The identity set is a SUBSET of *stage1_identities*

    Returning False for condition 2 (empty set) prevents citation-less
    non-actionable findings from being silently collapsed — they carry no
    structural evidence that they duplicate Stage 1.  Returning False for
    partial overlap (condition 3) ensures only *complete* echoes are
    suppressed; a finding with even one uncovered citation may represent new
    structural evidence.

    Used by :func:`ReconReportState.get_assembled_report` for Fix-1 read-time
    suppression.  Pure, sync, no I/O.
    """
    if finding.get('actionable') is not False:
        return False
    identities = _citation_identities(finding)
    if not identities:
        return False
    return identities <= stage1_identities


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
    # in-run dedup: description_hash → finding_id (null-null findings only)
    _deschash_to_finding: dict[str, str] = field(default_factory=dict)


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
        # Run-level O(1) indices so add_finding / _resolve_finding avoid scanning
        # all entries across every live run_id.  Populated on the miss path of
        # add_finding; cleaned up by tick() when entries are evicted.
        self._run_sig_index: dict[str, dict[tuple, str]] = {}  # run_id → {sig → finding_id}
        self._run_finding_index: dict[str, dict[str, _ReportEntry]] = {}  # run_id → {finding_id → entry}
        self._run_desc_index: dict[str, dict[str, str]] = {}  # run_id → {desc_hash → finding_id}
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

        # In-run dedup: two separate namespaces.
        # (1) Signature path: (task_id, flag_type) != (None, None) — O(1) lookup in
        #     _run_sig_index, scoped to this run_id across ALL stages.  Each field
        #     is canonicalized via _canonical_sig_field so an int task_id (common
        #     in LLM output) and the equivalent str task_id collapse to one key.
        # (2) Null-null path: both are None — dedup by normalized description hash
        #     in _run_desc_index so identical informational observations collapse.
        # The two namespaces are kept separate so a null-null finding with description
        # 'd' never collides with a real-signature finding that shares description 'd'.
        c_task_id = _canonical_sig_field(task_id)
        c_flag_type = _canonical_sig_field(flag_type)
        sig = (c_task_id, c_flag_type)
        desc_hash = ""
        if sig != (None, None):
            existing_id = self._run_sig_index.get(run_id, {}).get(sig)
            if existing_id is not None:
                return _duplicate_finding_error(existing_id)
        else:
            # Blank/whitespace-only descriptions normalize to '' — skip dedup so
            # each blank informational finding allocates independently.  The empty
            # string is not a meaningful dedup key (two observations with no text
            # are not the same observation).
            if _normalize_description(description):
                desc_hash = _description_hash(description)
                existing_id = self._run_desc_index.get(run_id, {}).get(desc_hash)
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
            self._run_sig_index.setdefault(run_id, {})[sig] = finding_id
        else:
            if desc_hash:  # empty when description normalizes to blank — skip index
                entry._deschash_to_finding[desc_hash] = finding_id
                self._run_desc_index.setdefault(run_id, {})[desc_hash] = finding_id
        self._run_finding_index.setdefault(run_id, {})[finding_id] = entry

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
        """Return the §9.3 assembled report dict, or None if unknown.

        Fix 1 (task-1654): read-time suppression of non-actionable same-run
        echoes.  Before returning flagged_items, collect the union of
        _citation_identities over ALL same-run findings whose entry.stage ==
        'memory_consolidator', EXCLUDING the candidate finding by finding_id
        (self-exclusion so a Stage-1 finding is never suppressed by its own
        citations — closes the aba1ac28 null-null desc-hash bypass).  Any
        finding for which _traces_exclusively_to_stage1 returns True is dropped
        from flagged_items.  Finding rows remain in _state/_run_finding_index so
        cross-stage cite_* resolution is unaffected by the read-time filter.

        Suppression is skipped entirely when ``stage == 'memory_consolidator'``
        to prevent two sibling non-actionable Stage-1 findings that cite the
        same target from mutually suppressing each other.
        """
        entry = self._state.get((run_id, stage))
        if entry is None:
            return None

        # --- Fix 1: build stage-1 citation identity set for this run ---
        # Collect _citation_identities from all memory_consolidator findings
        # across this run (may be spread across multiple start_report calls for
        # the same stage, though in practice there is one entry per stage).
        #
        # Skipped entirely when stage IS memory_consolidator: applying the
        # filter there would let two sibling non-actionable Stage-1 findings
        # that cite the same target mutually suppress each other.  Fix 1 only
        # targets cross-stage echoes, not intra-Stage-1 duplicates.
        stage1_identities_by_finding: dict[str, set[str]] = {}
        if stage != 'memory_consolidator':
            for (r_id, s), other_entry in self._state.items():
                if r_id == run_id and s == 'memory_consolidator':
                    for f in other_entry.findings:
                        stage1_identities_by_finding[f.finding_id] = _citation_identities({
                            'cited_tasks': list(f.cited_tasks),
                            'cited_entities': list(f.cited_entities),
                            'cited_edges': list(f.cited_edges),
                            'cited_memories': list(f.cited_memories),
                        })

        # Pre-compute full union of all stage-1 identity sets once; per-candidate
        # we subtract only that finding's own identities — O(F+S) instead of O(F·S).
        _stage1_full_union: set[str] = (
            set().union(*stage1_identities_by_finding.values())
            if stage1_identities_by_finding
            else set()
        )

        flagged_items: list[dict[str, Any]] = []
        for f in entry.findings:
            finding_dict = {
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
            # Fix 1 suppression: stage1_ids = full union minus this candidate's
            # own identities (self-exclusion so Stage-1 findings are not
            # suppressed by their own citations).
            stage1_ids = (
                _stage1_full_union
                - stage1_identities_by_finding.get(f.finding_id, set())
            )
            if _traces_exclusively_to_stage1(finding_dict, stage1_ids):
                # Drop: non-actionable finding whose citations trace exclusively
                # to a same-run Stage-1 finding.  Finding row remains in _state
                # so cite_* resolution is unaffected.
                continue
            flagged_items.append(finding_dict)

        return {
            'summary': entry.summary,
            'stats': dict(entry.stats),
            'flagged_items': flagged_items,
            'summary_warnings': list(entry.summary_warnings),
        }

    def get_findings_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Return every finding filed under *run_id*, across ALL stages, raw.

        Aggregates ``entry.findings`` from every ``(r_id, stage)`` entry whose
        ``r_id == run_id`` — robust to however many stages have filed findings
        for this run.  Each ``_Finding`` is projected to the same dict shape
        used by :meth:`get_assembled_report` (finding_id, severity, category,
        description, suggested_action, actionable, task_id, flag_type, and
        copies of the four cited_* lists).

        Unlike :meth:`get_assembled_report`, this method does **NOT** apply
        Fix-1 read-time echo suppression (task-1654).  It is intentionally a
        raw, run-scoped read: task-1966 uses it as an independent "recon_report
        channel" for Stage 2 to poll for ``systemic_pattern`` findings, so that
        channel stays genuinely independent of the primary (Mem0 flagged-items)
        channel even when the latter is suppressed by a ``stage1_flag_suppression``
        record.

        Returns ``[]`` for an unknown ``run_id`` (never raises).
        """
        results: list[dict[str, Any]] = []
        for (r_id, _stage), entry in self._state.items():
            if r_id != run_id:
                continue
            for f in entry.findings:
                results.append({
                    'finding_id': f.finding_id,
                    'severity': f.severity,
                    'category': f.category,
                    'description': f.description,
                    'suggested_action': f.suggested_action,
                    'actionable': f.actionable,
                    'task_id': f.task_id,
                    'flag_type': f.flag_type,
                    'cited_entities': list(f.cited_entities),
                    'cited_edges': list(f.cited_edges),
                    'cited_tasks': list(f.cited_tasks),
                    'cited_memories': list(f.cited_memories),
                })
        return results

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

        The lookup covers ALL stage entries that share this ``run_id`` via
        _run_finding_index, so a finding_id returned via a cross-stage
        ``duplicate_finding`` response (where the original finding lives in an
        earlier stage's entry) remains citable from a later stage.  The index
        is keyed by run_id, preserving cross-run isolation.
        """
        entry = self._run_finding_index.get(run_id, {}).get(finding_id)
        if entry is None:
            return None
        for f in entry.findings:
            if f.finding_id == finding_id:
                return entry, f
        return None

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
        finding_entry, finding = resolved

        if self._memory_service is None:
            return _ERR_SERVICE_UNAVAILABLE.copy()

        result = await self._memory_service.get_entity(name, finding_entry.project_id)
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
        finding_entry, finding = resolved

        if not _UUID_RE.match(edge_uuid):
            return _ERR_INVALID_UUID_SHAPE.copy()

        if self._memory_service is None:
            return _ERR_SERVICE_UNAVAILABLE.copy()

        try:
            result = await self._memory_service.get_edge(edge_uuid, finding_entry.project_id)
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
        finding_entry, finding = resolved

        if not _UUID_RE.match(memory_id):
            return _ERR_INVALID_UUID_SHAPE.copy()

        if self._memory_service is None:
            return _ERR_SERVICE_UNAVAILABLE.copy()

        try:
            fingerprint = await self._memory_service.get_memory(
                memory_id, store, finding_entry.project_id
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
            evicted = self._state[key]
            del self._state[key]
            # Remove _active pointer only if it still points at this stage
            if self._active.get(rid) == stage:
                del self._active[rid]
            # Clean up run-level indices for evicted findings
            if rid in self._run_sig_index:
                for sig, fid in evicted._signature_to_finding.items():
                    if self._run_sig_index[rid].get(sig) == fid:
                        del self._run_sig_index[rid][sig]
                if not self._run_sig_index[rid]:
                    del self._run_sig_index[rid]
            if rid in self._run_finding_index:
                for f in evicted.findings:
                    self._run_finding_index[rid].pop(f.finding_id, None)
                if not self._run_finding_index[rid]:
                    del self._run_finding_index[rid]
            if rid in self._run_desc_index:
                for dhash, fid in evicted._deschash_to_finding.items():
                    if self._run_desc_index[rid].get(dhash) == fid:
                        del self._run_desc_index[rid][dhash]
                if not self._run_desc_index[rid]:
                    del self._run_desc_index[rid]
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
