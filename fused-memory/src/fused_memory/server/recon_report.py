"""recon_report MCP namespace — in-process state + tool scaffold (task α/β).

Provides ten tools: start_report / add_finding / set_stat / inc_stat / complete /
delete_finding / cite_entity / cite_edge / cite_task / cite_memory.
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
# Fix 2 helpers — overlength description/suggested_action/category truncation
# (task-2410)
# ---------------------------------------------------------------------------

# Generous cap: legitimate findings never come close to this; only
# pathological input (e.g. a runaway agent dumping a huge blob into
# description/suggested_action/category) is ever capped.  severity is
# intentionally NOT capped: it is expected to be a short enum-like label
# (e.g. 'low'/'moderate'/'high'), not open-ended free text, so it does not
# carry the same pathological-length risk as the three free-text fields.
_MAX_FINDING_TEXT_CHARS = 10_000
_TRUNCATION_MARKER = '…[truncated]'


def _truncate_field(value: str, cap: int) -> tuple[str, bool]:
    """Cap *value* to *cap* chars, appending ``_TRUNCATION_MARKER`` if truncated.

    Returns ``(possibly-truncated value, was_truncated)``.  A *value* whose
    length is ``<= cap`` is returned unchanged with ``False``.

    Note: the marker is appended AFTER slicing to *cap*, so a truncated
    result is ``cap + len(_TRUNCATION_MARKER)`` chars long, not exactly
    *cap* — harmless given today's generous ``_MAX_FINDING_TEXT_CHARS``
    bound, but relevant if *cap* is ever tightened to a hard byte/char
    budget.
    """
    if len(value) > cap:
        return value[:cap] + _TRUNCATION_MARKER, True
    return value, False


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
    # in-run dedup: (task_id, flag_type) → finding_id.  Mirrors the entries
    # this ONE stage contributed to the run-scoped ReconReportState._run_sig_index.
    # Not consulted for eviction teardown: since task-2088, run-level indices
    # are released wholesale at run quiescence (see ReconReportState.tick()),
    # not by walking this per-entry map — do not resurrect a per-entry loop.
    _signature_to_finding: dict[tuple[str | None, str | None], str] = field(
        default_factory=dict
    )
    # in-run dedup: description_hash → finding_id (null-null findings only).
    # Mirrors this stage's contribution to _run_desc_index; same
    # run-quiescence-release caveat as _signature_to_finding above.
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


def _duplicate_finding_error(
    existing_id: str, warnings: list[str] | None = None
) -> dict[str, Any]:
    """Build the duplicate_finding error dict, optionally carrying truncation
    warnings (task-2410).

    add_finding truncates description/suggested_action/category BEFORE the
    dedup check runs (see its docstring), so a caller whose overlength input turns
    out to be a duplicate must still learn its text was capped — the
    truncated text itself is discarded (a duplicate is never stored), but
    the warning is not.  Mirrors the success-path contract: ``'warnings'``
    is present only when *warnings* is non-empty.
    """
    error: dict[str, Any] = {
        'error': 'duplicate_finding',
        'error_type': 'ReconReportDuplicateFinding',
        'existing_finding_id': existing_id,
    }
    if warnings:
        error['warnings'] = warnings
    return error


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
        # add_finding.  RUN-QUIESCENCE-scoped lifetime (task-2088): released by
        # tick() only once the run's LAST (run_id, *) entry evicts — NOT torn
        # down per-entry — so cross-stage in-run dedup and duplicate_finding
        # citation pointers survive an individual stage's TTL eviction for as
        # long as the run itself is still live.  See tick()'s docstring.
        self._run_sig_index: dict[str, dict[tuple, str]] = {}  # run_id → {sig → finding_id}
        self._run_finding_index: dict[str, dict[str, _ReportEntry]] = {}  # run_id → {finding_id → entry}
        self._run_desc_index: dict[str, dict[str, str]] = {}  # run_id → {desc_hash → finding_id}
        # run_id → {"project_id:task_id" → finding_id} for a null-task_id
        # finding's PRIMARY (first-ever) cited external task (task-2425).
        # Same run-quiescence-scoped lifetime as the three indices above —
        # see tick()'s docstring — released together, never torn down
        # per-entry.
        self._run_cited_task_index: dict[str, dict[str, str]] = {}
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

        A null/missing ``flag_type`` on a re-raise of an already-flagged
        ``task_id`` inherits that task's single established flag_type before
        the signature lookup runs (task-2318), so an under-specified re-raise
        collapses onto the canonical finding instead of allocating a distinct
        ``(task_id, None)`` row.

        ``description``/``suggested_action``/``category`` are truncated to
        ``_MAX_FINDING_TEXT_CHARS`` BEFORE dedup hashing (task-2410;
        see :func:`_truncate_field`).  One consequence for the null-null
        (no task_id/flag_type) path: the description-hash dedup key is
        computed from the POST-truncation string, so two null-null findings
        that are identical for the first ``_MAX_FINDING_TEXT_CHARS``
        characters and differ only in the truncated-away tail collapse into
        a single finding. This is an accepted tradeoff at today's generous
        10_000-char cap — revisit only if the cap is ever tightened.
        ``category`` is capped alongside description/suggested_action
        because it is free text supplied by the calling agent and is
        surfaced verbatim in the assembled report; ``severity`` is left
        unbounded as it is expected to be a short enum-like label (see the
        module-level comment above ``_MAX_FINDING_TEXT_CHARS``).
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

        # Fix 2 (task-2410): gracefully cap pathologically long text fields
        # BEFORE dedup hashing, so the stored text, the assembled-report
        # value, and the null-null description-hash dedup key all derive
        # from the same (possibly-capped) string.  category is capped here
        # too (same free-text-from-a-runaway-agent risk as description/
        # suggested_action); it does not participate in dedup hashing, so
        # its ordering relative to the dedup check below is not load-bearing
        # the way description's is.
        warnings: list[str] = []
        description, description_truncated = _truncate_field(description, _MAX_FINDING_TEXT_CHARS)
        if description_truncated:
            warnings.append(f'description truncated to {_MAX_FINDING_TEXT_CHARS} chars')
        suggested_action, suggested_action_truncated = _truncate_field(
            suggested_action, _MAX_FINDING_TEXT_CHARS
        )
        if suggested_action_truncated:
            warnings.append(f'suggested_action truncated to {_MAX_FINDING_TEXT_CHARS} chars')
        category, category_truncated = _truncate_field(category, _MAX_FINDING_TEXT_CHARS)
        if category_truncated:
            warnings.append(f'category truncated to {_MAX_FINDING_TEXT_CHARS} chars')

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

        # Flag-type inheritance (task-2318): a re-raise that omits flag_type
        # for an already-flagged task_id must not fork into its own (task_id,
        # None) signature — it should collapse onto that task's established
        # finding so Stage 2 can attach citations to the canonical row.
        # Inherit ONLY when EXACTLY ONE non-null flag_type is established for
        # this task_id in the run: zero means this null re-raise is the first
        # signal (kept as (task_id, None), still dedups against later bare-null
        # re-raises); more than one is ambiguous — a bare re-raise gives no
        # basis to pick which condition it restates, so guessing risks
        # merging two genuinely distinct findings, and is deliberately not
        # done. Scanning _run_sig_index[run_id] (rather than a new index)
        # reuses its existing cross-stage, run-quiescence-scoped lifetime
        # (task-2088), so inheritance works across stages and survives an
        # earlier stage's own entry being TTL-evicted, with no tick() change.
        # Cost: this is an O(signatures-in-run) scan on every add_finding call
        # that has a non-null task_id and a null flag_type — this sub-path is
        # expected to be rare (an under-specified re-raise), so the scan is
        # acceptable and not worth a 4th index today. If profiling ever shows
        # bare-null re-raises are common on high-finding-count runs, add a
        # per-run {task_id -> set(flag_type)} auxiliary index maintained
        # alongside _run_sig_index for an O(1) lookup instead.
        if c_task_id is not None and c_flag_type is None:
            established = {
                ft for (tid, ft) in self._run_sig_index.get(run_id, {}) if tid == c_task_id and ft is not None
            }
            if len(established) == 1:
                c_flag_type = next(iter(established))

        sig = (c_task_id, c_flag_type)
        desc_hash = ""
        if sig != (None, None):
            existing_id = self._run_sig_index.get(run_id, {}).get(sig)
            if existing_id is not None:
                return _duplicate_finding_error(existing_id, warnings)
        else:
            # Blank/whitespace-only descriptions normalize to '' — skip dedup so
            # each blank informational finding allocates independently.  The empty
            # string is not a meaningful dedup key (two observations with no text
            # are not the same observation).
            if _normalize_description(description):
                desc_hash = _description_hash(description)
                existing_id = self._run_desc_index.get(run_id, {}).get(desc_hash)
                if existing_id is not None:
                    return _duplicate_finding_error(existing_id, warnings)

        finding_id = str(uuid.uuid4())
        finding = _Finding(
            finding_id=finding_id,
            severity=severity,
            category=category,
            description=description,
            suggested_action=suggested_action,
            actionable=actionable,
            task_id=c_task_id,
            flag_type=c_flag_type,
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

        result: dict[str, Any] = {'finding_id': finding_id}
        if warnings:
            result['warnings'] = warnings
        return result

    def delete_finding(self, run_id: str, finding_id: str) -> dict[str, Any]:
        """Permanently remove a finding.  IRREVERSIBLE.

        Mirrors ``delete_memory``'s semantics (server/tools.py): validate,
        then perform an irreversible removal and return a structured
        ``{'status': 'deleted', 'finding_id': ...}`` dict.  Scoped by
        ``run_id`` + ``finding_id``.

        The finding is resolved cross-stage via :meth:`_resolve_finding`
        (same as the ``cite_*`` tools), so a finding filed by an earlier
        stage of this run can still be deleted from a later stage.

        Rejected with ``report_already_completed`` when the finding's
        OWNING entry has already been completed — consistent with the
        add_finding/set_stat/inc_stat post-completion guard, and protects
        complete()'s cached ``flagged_count``/``stats`` from silent
        corruption.  Retraction is intended for in-progress stages.

        Removes the finding from ``entry.findings`` and
        ``_run_finding_index``, and also cleans whichever dedup index it
        was filed under (``_run_sig_index`` / ``_run_desc_index`` and
        their per-entry mirrors) — recomputed from the finding's own
        already-canonicalized ``task_id``/``flag_type``/``description`` —
        so a corrected finding can be re-filed under the same
        signature/description after retraction instead of bouncing off a
        stale ``duplicate_finding`` pointer.
        """
        entry = self._resolve_entry(run_id)
        if entry is None:
            return _ERR_RUN_UNKNOWN.copy()

        resolved = self._resolve_finding(run_id, finding_id)
        if resolved is None:
            return _ERR_FINDING_UNKNOWN.copy()
        owning_entry, finding = resolved

        if owning_entry.completed_at is not None:
            logger.warning(
                'recon_report: delete_finding called after complete() for '
                'run_id=%r stage=%r finding_id=%r; rejected',
                run_id,
                owning_entry.stage,
                finding_id,
            )
            return _ERR_ALREADY_COMPLETED.copy()

        # Identity-based removal (not list.remove(), which dispatches to
        # _Finding's dataclass-generated __eq__): _resolve_finding returned
        # this exact object by reference from owning_entry.findings, so
        # filtering by `is` makes the "remove precisely the resolved
        # object" invariant explicit rather than relying on field-by-field
        # equality (harmless today since finding_id is unique and is one of
        # the compared fields, but identity is the actual invariant).
        owning_entry.findings[:] = [f for f in owning_entry.findings if f is not finding]
        self._run_finding_index.get(run_id, {}).pop(finding_id, None)

        sig = (finding.task_id, finding.flag_type)
        if sig != (None, None):
            self._run_sig_index.get(run_id, {}).pop(sig, None)
            owning_entry._signature_to_finding.pop(sig, None)
        elif _normalize_description(finding.description):
            desc_hash = _description_hash(finding.description)
            self._run_desc_index.get(run_id, {}).pop(desc_hash, None)
            owning_entry._deschash_to_finding.pop(desc_hash, None)

        return {'status': 'deleted', 'finding_id': finding_id}

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

    def _purge_finding(
        self,
        run_id: str,
        owning_entry: _ReportEntry,
        finding: _Finding,
    ) -> None:
        """Remove *finding* from *owning_entry* and every dedup index it may
        be registered under: ``_run_finding_index``, whichever of
        ``_run_sig_index`` / ``_run_desc_index`` it was filed under, and
        ``_run_cited_task_index`` if it is a primary-cited-task fold anchor.

        Single-sourced (task-2425) by :meth:`delete_finding` and the in-run
        cited-task fold's retract path in :meth:`cite_task`, so the four
        run-level dedup indices can never drift out of sync between the two
        removal paths, and neither path can leave a stale pointer to a
        finding that no longer exists.

        Identity-based removal (``is``, not ``==``) — *finding* is the exact
        object reference returned by :meth:`_resolve_finding`; see
        ``delete_finding``'s docstring for why this matters.
        """
        owning_entry.findings[:] = [f for f in owning_entry.findings if f is not finding]
        self._run_finding_index.get(run_id, {}).pop(finding.finding_id, None)

        sig = (finding.task_id, finding.flag_type)
        if sig != (None, None):
            self._run_sig_index.get(run_id, {}).pop(sig, None)
            owning_entry._signature_to_finding.pop(sig, None)
        elif _normalize_description(finding.description):
            desc_hash = _description_hash(finding.description)
            self._run_desc_index.get(run_id, {}).pop(desc_hash, None)
            owning_entry._deschash_to_finding.pop(desc_hash, None)

        if finding.task_id is None and finding.cited_tasks:
            primary = finding.cited_tasks[0]
            primary_key = f"{primary['project_id']}:{primary['task_id']}"
            run_cited_tasks = self._run_cited_task_index.get(run_id, {})
            if run_cited_tasks.get(primary_key) == finding.finding_id:
                run_cited_tasks.pop(primary_key, None)

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

        In-run cited-task fold (task-2425): when *finding* has a null
        top-level task_id and this is its PRIMARY (first-ever) citation, the
        (project_id, task_id) pair doubles as an in-run dedup anchor. If an
        earlier finding in this run already registered the same primary
        cited task, the just-looked-up finding is purged and
        ``duplicate_finding`` is returned instead — two findings citing the
        same external blocker are semantically duplicate regardless of how
        differently they are worded. Findings with a real top-level task_id,
        and secondary (non-first) citations, are never fold anchors.

        The fold EXEMPTS ``memory_consolidator`` (Stage-1) findings, mirroring
        Fix-1's read-time ``stage != 'memory_consolidator'`` carve-out in
        :meth:`get_assembled_report`: two sibling Stage-1 findings that cite the
        same target stay distinct, and a Stage-2 echo of a Stage-1 citation is
        already suppressed at read time — so the write-time fold only collapses
        same-run duplicates in a non-Stage-1 stage that Fix-1 cannot reach.
        """
        entry = self._resolve_entry(run_id)
        if entry is None:
            return _ERR_RUN_UNKNOWN.copy()

        resolved = self._resolve_finding(run_id, finding_id)
        if resolved is None:
            return _ERR_FINDING_UNKNOWN.copy()
        finding_entry, finding = resolved

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

        # In-run cited-task fold (task-2425) — see docstring above. Only the
        # PRIMARY citation (finding.cited_tasks still empty) of a null-
        # top-level-task_id finding is a fold anchor.
        #
        # memory_consolidator (Stage 1) findings are EXEMPT from the fold —
        # mirroring Fix-1's own read-time carve-out in get_assembled_report
        # (`stage != 'memory_consolidator'`). Two sibling Stage-1 findings that
        # cite the same target are deliberately kept distinct (see
        # test_sibling_stage1_findings_not_mutually_suppressed); a Stage-2 echo
        # of a Stage-1 citation is already handled at read time by
        # _traces_exclusively_to_stage1. The write-time fold therefore only
        # targets the gap Fix-1 cannot reach: two same-run findings in a
        # non-Stage-1 stage (e.g. task_knowledge_sync) that cite the same
        # external blocker Stage 1 never cited — the autopilot_video repro.
        if (
            finding.task_id is None
            and not finding.cited_tasks
            and finding_entry.stage != 'memory_consolidator'
        ):
            cited_task_key = f'{project_id}:{task_id}'
            run_cited_tasks = self._run_cited_task_index.setdefault(run_id, {})
            existing_id = run_cited_tasks.get(cited_task_key)
            if existing_id is not None and existing_id != finding.finding_id:
                self._purge_finding(run_id, finding_entry, finding)
                return _duplicate_finding_error(existing_id)
            run_cited_tasks[cited_task_key] = finding.finding_id

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
        """Sweep completed entries past TTL.  Returns count evicted.

        Each evicted entry's own ``_state``/``_active`` lookup slots are
        removed immediately — that part of the contract is unchanged. This is
        NOT the same as the ``_ReportEntry`` object being garbage-collected
        immediately: while its run is still live, ``_run_finding_index``
        keeps a reference to the evicted entry (see below) so its findings
        stay citable, and the object only becomes unreachable once the run's
        indices are popped at quiescence.

        The three shared run-level dedup indices (``_run_sig_index`` /
        ``_run_desc_index`` / ``_run_finding_index``) are RUN-QUIESCENCE
        scoped, not per-entry (task-2088). They are keyed for the WHOLE run
        across all of its stages, and reconciliation runs are multi-stage and
        long-lived: Stage 1 (memory_consolidator) typically files a finding
        and completes early, while Stage 2/3 + remediation keep the run live
        for minutes. Releasing a run's indices the moment any ONE completed
        stage ages out — while a sibling stage of the SAME run is still live
        — would drop the in-run dedup key mid-run and let a later stage
        double-file a signature/description already reported by an evicted
        stage (run 33c324b0 regression), and would dangle any
        duplicate_finding pointer already handed to another stage by
        collapsing _run_finding_index out from under it.

        So: while at least one ``(run_id, *)`` entry remains in ``_state``
        (in-progress, or completed but not yet past its own TTL), all three
        indices for that run_id are preserved untouched, even for stages that
        have already been evicted. Only when a run's LAST entry evicts do we
        release its three indices, wholesale via ``pop(rid, None)`` rather
        than by walking the evicted entry's own signature/desc-hash/finding
        maps — this correctly reclaims contributions from every stage of the
        run in one shot and is robust to several same-run entries evicting
        within the same tick() call. Do not reintroduce per-entry teardown
        here; see test_eviction_partial_run_canonical_stage_cleared and the
        sibling tests in TestReconReportReaper for the regression this
        guards against.

        Quiescence is computed ONCE per tick() call — the set of run_ids
        still present in ``_state`` after all of this sweep's deletions —
        rather than via a per-evicted-entry ``any()`` scan over ``_state``.
        The latter is O(evicted × remaining _state size) and becomes
        quadratic when many entries across many runs age out in the same
        sweep; computing the surviving run_id set once and then doing an O(1)
        membership check per evicted run_id is O(evicted + remaining).

        Retention caveat: reaching quiescence requires every ``(run_id, *)``
        entry to individually hit completed_at + TTL. An in-progress entry
        never expires on its own — it is immortal by design (PRD §9.4; see
        test_inprogress_not_evicted_by_ttl) — so a stalled or crashed stage
        that never calls complete() pins that run_id's three indices, and
        every already-evicted sibling entry object kept reachable through
        ``_run_finding_index``, for as long as the process keeps running.
        There is currently no separate max-lifetime sweep for in-progress
        entries to bound this. This is an accepted tradeoff for task-2088
        (worst case is bounded by process restart, not memory exhaustion
        under normal operation), not a defect this task fixes.
        """
        now = self._clock()
        to_evict = [
            (rid, stage)
            for (rid, stage), entry in self._state.items()
            if entry.completed_at is not None
            and now - entry.completed_at > self._ttl_seconds
        ]
        evicted_run_ids: set[str] = set()
        for key in to_evict:
            rid, stage = key
            del self._state[key]
            evicted_run_ids.add(rid)
            # Remove _active pointer only if it still points at this stage
            if self._active.get(rid) == stage:
                del self._active[rid]
        if evicted_run_ids:
            # Run-quiescence gate (task-2088) — see the tick() docstring above.
            # Compute the set of run_ids with a surviving _state entry ONCE,
            # after all of this sweep's deletions, instead of rescanning
            # _state per evicted entry. A run_id that evicted this tick and
            # has no surviving entry is fully quiescent — release its three
            # shared indices wholesale.
            live_run_ids = {r for (r, _s) in self._state}
            for rid in evicted_run_ids:
                if rid not in live_run_ids:
                    self._run_sig_index.pop(rid, None)
                    self._run_finding_index.pop(rid, None)
                    self._run_desc_index.pop(rid, None)
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

Tools: start_report, add_finding, set_stat, inc_stat, complete, delete_finding,
       cite_entity, cite_edge, cite_task, cite_memory.

Usage pattern (per PRD §9.2):
1. start_report — open a new report at the start of a stage run.
2. add_finding — append a diagnostic finding (deduplicated by task_id + flag_type
                  across ALL stages of the same run_id).  Overlength
                  description/suggested_action are truncated with a
                  'warnings' entry on the response, never rejected.
3. set_stat / inc_stat — track numeric metrics during the run.
4. complete — stamp the summary and close the report; idempotent.
5. delete_finding(run_id, finding_id) — IRREVERSIBLE retraction of a
                  finding filed earlier in this run (any stage); rejected
                  once that finding's owning stage has been completed.

Citation tools (call after add_finding, before or after complete):
6. cite_entity(run_id, finding_id, name) — resolve entity by name and attach.
7. cite_edge(run_id, finding_id, edge_uuid) — validate UUID and attach edge.
8. cite_task(run_id, finding_id, project_id, task_id) — look up task and attach.
9. cite_memory(run_id, finding_id, memory_id, store) — look up memory and attach.
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
    async def delete_finding(run_id: str, finding_id: str) -> dict:
        """Permanently remove a finding.  IRREVERSIBLE.

        Scoped by run_id + finding_id; mirrors delete_memory's semantics.
        Returns {status: 'deleted', finding_id} on success, or a structured
        error dict (run_id_unknown / finding_unknown / report_already_completed).
        Rejected once the finding's owning stage entry has been completed —
        retraction is for in-progress stages only.
        """
        return state.delete_finding(run_id=run_id, finding_id=finding_id)

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
        duplicate_finding (task-2425) when this is a null-task_id finding's
        primary citation and an earlier finding in this run already cited
        the same (project_id, task_id) — the newly-cited finding is purged
        and existing_finding_id points at the original.
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
