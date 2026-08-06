"""recon_report MCP namespace — in-process state + tool scaffold (task α/β).

Provides eleven tools: start_report / add_finding / set_stat / inc_stat / complete /
delete_finding / cite_entity / cite_edge / cite_task / cite_memory / cite_run.
State is owned by :class:`ReconReportState`; tools are thin delegates registered
by :func:`create_recon_report_server`.  This split lets unit tests drive the state
directly without spinning up FastMCP.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from graphiti_core.errors import EdgeNotFoundError

from fused_memory.services.memory_service import MemoryNotFoundError
from fused_memory.utils.validation import is_full_uuid

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


def _split_task_id_parts(value: str) -> set[str]:
    """Return the set of individual task-id parts within *value*.

    ``value`` is a (possibly comma-joined) canonical task_id string, e.g.
    ``'5040,5149'`` -> ``{'5040', '5149'}`` and ``'5040'`` -> ``{'5040'}``.
    Each part is stripped of surrounding whitespace; empty parts (from
    leading/trailing/doubled commas) are dropped.

    Used both by :func:`_canonicalize_task_id_string` (dedup-signature
    normalization, task-2432 bullet 4) and by the entity-scoped cite_task
    fold's subset-membership eligibility gate (task-2432 bullets 1b/2/3).
    """
    return {p.strip() for p in value.split(',') if p.strip()}


def _task_id_part_sort_key(part: str) -> tuple[int, Any]:
    """Sort key for a single task-id part: numeric parts by int value (and
    ordered before non-numeric parts), non-numeric parts lexically.

    A stable, deterministic ordering is all that's required for dedup
    purposes — the exact numeric-before-lexical grouping is not itself load
    -bearing, only that two calls describing the same set of parts always
    sort to the same joined string.
    """
    try:
        return (0, int(part))
    except ValueError:
        return (1, part)


def _canonicalize_task_id_string(value: str) -> str:
    """Canonicalize a top-level ``task_id`` string for dedup-signature purposes.

    Splits *value* on ``','``, strips whitespace from each part, drops empty
    parts, sorts (numeric parts by int value, non-numeric parts lexically —
    see :func:`_task_id_part_sort_key`), dedupes, and rejoins with ``','``.
    A single-value input canonicalizes to itself (e.g. ``'5040'`` ->
    ``'5040'``); an input with no non-empty parts (e.g. ``''``) is returned
    unchanged.

    This ensures a comma-joined task_id whose components are reordered (or
    duplicated) between two ``add_finding`` calls collapses onto the same
    dedup signature (task-2432 bullet 4).

    Both this helper and ``reconciliation.flag_dedup.compute_flag_signature``'s
    ``cited_tasks`` union produce a sorted, comma-joined string, but the two
    sort orders are independent, not mirrored: this one is numeric-aware (see
    :func:`_task_id_part_sort_key`), while ``flag_dedup``'s is plain
    lexicographic. Each subsystem is internally consistent among its own
    callers and neither ever compares its signature against the other's — the
    in-run (server) and cross-cycle (reconciliation) dedup layers run at
    different boundaries and each recomputes its own signature fresh — so the
    differing order is intentional, not a bug. Do not assume the two strings
    are interchangeable if a future caller ever bridges server/ and
    reconciliation/.
    """
    parts = _split_task_id_parts(value)
    if not parts:
        return value
    return ','.join(sorted(parts, key=_task_id_part_sort_key))


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


def _cited_task_key(project_id: str, task_id: str) -> str:
    """Build the canonical ``'{project_id}:{task_id}'`` identity string that
    anchors the in-run primary-cited-task fold (task-2425).

    Single-sourced so :meth:`ReconReportState.cite_task` (fold registration)
    and :meth:`ReconReportState._purge_finding` (fold-key cleanup) can never
    drift apart — if the two built this string independently and the formats
    diverged, a purge would silently fail to clear the anchor it registered,
    leaking a stale ``_run_cited_task_index`` entry.  Mirrors the
    ``f'{pid}:{tid}'`` convention :func:`_citation_identities` already uses
    for ``cited_tasks`` entries.
    """
    return f'{project_id}:{task_id}'


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
# Cross-project routing taxonomy guard (task-2453)
# ---------------------------------------------------------------------------

# category='cross_project_routing' is claimed by Stage 2 (task_knowledge_sync)
# for findings asserting that work belongs to a different project. Without a
# cite_task-produced cited_tasks entry, that claim is unverified — see
# _apply_cross_project_routing_guard below.
_CROSS_PROJECT_ROUTING_CATEGORY = 'cross_project_routing'
_CROSS_PROJECT_INFO_DOWNGRADE_CATEGORY = 'other'
_CROSS_PROJECT_INFO_FLAG_TYPE = 'cross_project_info'


def _apply_cross_project_routing_guard(finding: dict) -> dict:
    """Downgrade an anchor-less ``cross_project_routing`` finding in place.

    A non-empty ``cited_tasks`` list is the sole machine-checkable proof that
    a ``cite_task`` -> ``task_interceptor.get_task`` routing check actually
    ran for the cited project/task (see :meth:`ReconReportState.cite_task`).
    ``cited_entities``/``cited_edges``/``cited_memories`` do NOT count — they
    resolve graph/memory objects, not task routing.

    When ``finding['category'] == 'cross_project_routing'`` and
    ``finding['cited_tasks']`` is empty, this rewrites the category to
    ``'other'`` and the flag_type to ``'cross_project_info'`` — the claim is
    downgraded to a plain informational note rather than surfaced as a
    routing finding. All other fields (finding_id/description/actionable/
    task_id/etc.) are left untouched, and a finding with a cited_tasks entry
    is returned unchanged.

    *finding* is mutated in place and also returned for call-site convenience.
    Callers must pass a freshly-built projection dict (as
    :meth:`ReconReportState.get_assembled_report` does) — never the stored
    ``_Finding`` — so re-reads stay idempotent and cite_* resolution by
    finding_id is unaffected.
    """
    if finding.get('category') == _CROSS_PROJECT_ROUTING_CATEGORY and not finding.get('cited_tasks'):
        finding['category'] = _CROSS_PROJECT_INFO_DOWNGRADE_CATEGORY
        finding['flag_type'] = _CROSS_PROJECT_INFO_FLAG_TYPE
    return finding


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
    cited_runs: list[dict] = field(default_factory=list)  # task-2595
    # Hook B (task 2897 δ): the composite id of the ACTIVE entity standing
    # decision that adjudicates this finding, or None. Set by cite_entity when a
    # cited entity carries an active decision; defaults None so old persisted
    # rows hydrate round-trip-safe via _Finding(**fd).
    standing_decision_id: str | None = None


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
# Entry (de)serialization — SQLite write-through persistence (task 2716)
# ---------------------------------------------------------------------------
#
# entry_json's shape is an internal implementation detail of the persistence
# write-through path (ReconReportState._persist_run / hydrate_from_store) and
# ReconReportStore rows — never surfaced to MCP tool callers.


def _encode_sig_map(
    sig_map: dict[tuple[str | None, str | None], str],
) -> list[list[str | None]]:
    """Encode a ``(task_id|None, flag_type|None) -> finding_id`` map as a JSON-safe
    list of ``[task_id, flag_type, finding_id]`` triples.

    A dict with a tuple key is not JSON-serializable directly, and coercing the
    tuple to a string key (e.g. ``str((task_id, flag_type))``) would collapse
    ``None`` into the literal string ``'None'`` — indistinguishable from a real
    ``'None'`` task_id/flag_type string on decode.  A flat list of triples sidesteps
    both problems: ``None`` round-trips through JSON ``null`` unchanged.
    """
    return [[sig[0], sig[1], finding_id] for sig, finding_id in sig_map.items()]


def _decode_sig_map(
    rows: list[list[str | None]],
) -> dict[tuple[str | None, str | None], str]:
    """Inverse of :func:`_encode_sig_map`."""
    return {(row[0], row[1]): row[2] for row in rows}  # type: ignore[misc]


def _serialize_entry(
    entry: _ReportEntry,
    *,
    sig_anchor_slice: dict[tuple[str | None, str | None], str],
    cited_task_slice: dict[str, str],
) -> str:
    """Serialize *entry* (plus its run-level fold-anchor slices) to a JSON string.

    *sig_anchor_slice* and *cited_task_slice* are NOT part of ``_ReportEntry`` —
    they are this entry's OWNED portion of the run-scoped
    ``ReconReportState._run_sig_index`` / ``_run_cited_task_index``, computed by
    the caller (``ReconReportState._persist_run``) from the live indices at
    persist time.  They ride along in the same JSON blob (rather than a second
    row/table) purely so one row still fully describes one ``(run_id, stage)``
    entry; :func:`_deserialize_entry` ignores them (they are not part of
    ``_ReportEntry``'s fields) — use :func:`_deserialize_fold_anchor_slices` to
    read them back for rebuilding the run-level indices on hydrate.

    ``_Finding`` and the entry's own scalar/collection fields are all built from
    JSON-safe primitives already (str / bool / float / None / list[dict]), so
    ``dataclasses.asdict`` needs no custom encoder.
    """
    payload = {
        'run_id': entry.run_id,
        'stage': entry.stage,
        'project_id': entry.project_id,
        'findings': [asdict(f) for f in entry.findings],
        'stats': dict(entry.stats),
        'summary': entry.summary,
        'summary_warnings': list(entry.summary_warnings),
        'completed_at': entry.completed_at,
        'created_at': entry.created_at,
        'signature_to_finding': _encode_sig_map(entry._signature_to_finding),
        'deschash_to_finding': dict(entry._deschash_to_finding),
        'sig_anchor_slice': _encode_sig_map(sig_anchor_slice),
        'cited_task_slice': dict(cited_task_slice),
    }
    return json.dumps(payload)


def _deserialize_entry(entry_json: str) -> _ReportEntry:
    """Inverse of :func:`_serialize_entry`'s ``_ReportEntry``-shaped fields.

    Does NOT restore the fold-anchor slices — call
    :func:`_deserialize_fold_anchor_slices` on the same *entry_json* for those;
    they are run-level, not part of ``_ReportEntry``.
    """
    data = json.loads(entry_json)
    findings = [_Finding(**fd) for fd in data['findings']]
    return _ReportEntry(
        run_id=data['run_id'],
        stage=data['stage'],
        project_id=data['project_id'],
        findings=findings,
        stats=dict(data['stats']),
        summary=data['summary'],
        summary_warnings=list(data['summary_warnings']),
        completed_at=data['completed_at'],
        created_at=data['created_at'],
        _signature_to_finding=_decode_sig_map(data['signature_to_finding']),
        _deschash_to_finding=dict(data['deschash_to_finding']),
    )


def _deserialize_fold_anchor_slices(
    entry_json: str,
) -> tuple[dict[tuple[str | None, str | None], str], dict[str, str]]:
    """Return ``(sig_anchor_slice, cited_task_slice)`` persisted by
    :func:`_serialize_entry` — this entry's owned slice of the run-scoped
    ``_run_sig_index`` / ``_run_cited_task_index``.  Used by
    :meth:`ReconReportState.hydrate_from_store` to rebuild those indices by
    unioning every entry's slice.
    """
    data = json.loads(entry_json)
    return (
        _decode_sig_map(data['sig_anchor_slice']),
        dict(data['cited_task_slice']),
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

# The UUID shape gate below is `is_full_uuid` from utils/validation.py (task
# 3132) — the same predicate delete_memory's guards and citation_verifier's
# forwarding-pointer guard answer through, per INV-5.  It replaced a local
# anchored regex that ACCEPTED a canonical id with a trailing newline, because
# Python's `$` matches immediately before one; such an id passed this gate and
# then resolved to nothing.  Only the shape predicate is shared — the
# `invalid_uuid_shape` / `ReconReportInvalidUuid` envelope below stays this
# module's own.

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

# task-2595: returned by cite_run when count_memories_by_metadata finds no
# mem0 records under {'run_id': cited_run_id} — the cited run does not exist.
_ERR_RUN_NOT_FOUND: dict[str, str] = {
    'error': 'run_not_found',
    'error_type': 'ReconReportRunNotFound',
}

_ERR_INVALID_UUID_SHAPE: dict[str, str] = {
    'error': 'invalid_uuid_shape',
    'error_type': 'ReconReportInvalidUuid',
}

_ERR_SERVICE_UNAVAILABLE: dict[str, str] = {
    'error': 'service_not_configured',
    'error_type': 'ReconReportServiceUnavailable',
}

# task 2895 β: returned by write_entity_standing_decision when the grounds value
# is outside GROUNDS_ENUM (the ledger's ValueError). A ``hint`` carrying the
# ledger message is added at return time.
_ERR_INVALID_GROUNDS: dict[str, str] = {
    'error': 'invalid_grounds',
    'error_type': 'StandingDecisionInvalidGrounds',
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
        store:          Optional :class:`~fused_memory.server.recon_report_store.ReconReportStore`
                        (task 2716). When provided, every mutator write-throughs
                        its owning run's entries to durable SQLite after the
                        mutation succeeds (see :meth:`_persist_run`), and
                        :meth:`hydrate_from_store` can rebuild in-memory state
                        from it at startup. ``None`` (the default) makes
                        persistence a complete no-op — fresh in-process runs are
                        byte-identical whether or not a store is attached.
    """

    def __init__(
        self,
        ttl_seconds: float,
        clock: Callable[[], float] | None = None,
        reaper_interval: float = 60.0,
        memory_service: Any = None,
        task_interceptor: Any = None,
        store: Any = None,
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
        # SQLite write-through persistence (task 2716); None = fully inert.
        self._store = store

    def _clock(self) -> float:
        if self._clock_fn is not None:
            return self._clock_fn()
        return asyncio.get_running_loop().time()

    # ------------------------------------------------------------------
    # Persistence (task 2716)
    # ------------------------------------------------------------------

    def _persist_run(self, run_id: str) -> None:
        """Write every ``(run_id, *)`` entry through to the store.  No-op if
        ``self._store is None``.

        Upserts ALL of the run's entries (bounded to the handful of recon
        stages), not just the one a caller just mutated: several mutators have
        cross-stage effects (``delete_finding`` purges from the finding's
        OWNING entry, which may be an earlier stage; ``cite_task``'s in-run
        folds purge the losing finding from ITS owning entry) — upserting the
        whole run is trivially correct where "persist only what changed" would
        need fragile per-method reasoning about which stage's row to write.

        For each entry, computes its OWNED slice of the two run-level fold
        anchors (``_run_cited_task_index`` / the derived-signature entries in
        ``_run_sig_index``) fresh from the live indices — see
        :func:`_serialize_entry`'s docstring for why these ride along instead
        of a companion table. ``is_active`` is set per row from
        ``self._active.get(run_id) == stage``, so an active-stage transition
        self-corrects on the very next persist.

        All of the run's serialized rows are written through in ONE
        ``store.upsert_many`` transaction — a single commit / fsync per
        mutation rather than one per entry (review: performance).

        Best-effort: a store failure is logged loudly (WARNING, structured)
        but never raised — a shadow-store hiccup must not abort a recon stage
        (mirrors ``start_report``'s existing degradation posture).  Per-entry
        serialization is likewise resilient: an entry that fails to serialize
        is skipped-and-logged, and the rest of the run still persists.
        """
        if self._store is None:
            return
        active_stage = self._active.get(run_id)
        updated_at = self._clock()
        rows: list[dict[str, Any]] = []
        for (rid, stage), entry in self._state.items():
            if rid != run_id:
                continue
            # Serialize each entry independently so a single un-serializable
            # entry is skipped-and-logged without dropping the rest of the run.
            # This loop touches ONLY in-memory state (never the store), so the
            # single store write is the batched upsert_many below — one
            # transaction / one fsync for the whole run (review: performance),
            # instead of a commit per entry.
            try:
                finding_ids = {f.finding_id for f in entry.findings}
                sig_anchor_slice = {
                    sig: finding_id
                    for sig, finding_id in self._run_sig_index.get(run_id, {}).items()
                    if finding_id in finding_ids and sig not in entry._signature_to_finding
                }
                cited_task_slice = {
                    key: finding_id
                    for key, finding_id in self._run_cited_task_index.get(run_id, {}).items()
                    if finding_id in finding_ids
                }
                entry_json = _serialize_entry(
                    entry,
                    sig_anchor_slice=sig_anchor_slice,
                    cited_task_slice=cited_task_slice,
                )
            except Exception:
                logger.warning(
                    'recon_report: failed to serialize run_id=%r stage=%r for '
                    'persistence; skipping this entry',
                    rid,
                    stage,
                    exc_info=True,
                )
                continue
            rows.append(
                {
                    'run_id': rid,
                    'stage': stage,
                    'project_id': entry.project_id,
                    'is_active': (active_stage == stage),
                    'entry_json': entry_json,
                    'updated_at': updated_at,
                }
            )
        if not rows:
            return
        try:
            self._store.upsert_many(rows)
        except Exception:
            logger.warning(
                'recon_report: failed to persist run_id=%r (%d entr%s) to store',
                run_id,
                len(rows),
                'y' if len(rows) == 1 else 'ies',
                exc_info=True,
            )

    def hydrate_from_store(self) -> None:
        """Rebuild in-memory state from the persisted store — run ONCE at boot.

        A full no-op when ``self._store is None`` (the byte-identical no-store
        path) or when the store is empty (a fresh boot).  Otherwise, for every
        persisted row it restores ``_state[(run_id, stage)]`` and ``_active``
        (the row flagged ``is_active`` is the run's live stage), then rebuilds
        all four run-level dedup indices by UNIONING each entry's persisted
        per-entry mirrors and fold-anchor slices:

        * ``_run_finding_index`` — ``{finding_id -> owning entry}`` for every
          finding in every entry.
        * ``_run_sig_index`` — each entry's ``_signature_to_finding`` PLUS its
          persisted derived-signature slice (the ``cite_task`` entity-scoped
          fold anchors owned by this entry's findings but absent from its own
          ``_signature_to_finding``).
        * ``_run_desc_index`` — each entry's ``_deschash_to_finding``.
        * ``_run_cited_task_index`` — each entry's persisted cited-task slice
          (the ``cite_task`` project-scoped fold anchors).

        The union is faithful by construction — each sig / desc-hash /
        cited-task key maps to exactly one finding_id owned by exactly one
        entry — so no re-derivation of ``cite_task`` fold-eligibility is
        needed (see :func:`_serialize_entry`).  This gives σ a correct dedup
        substrate for continued filing against a run whose earlier stages were
        filed before the restart.

        Best-effort, mirroring :meth:`_persist_run`: a store read failure is
        logged loudly (WARNING, structured) but never raised — a hydrate
        hiccup must not abort server boot.  On a total read failure the process
        simply starts with empty in-memory state, exactly as if the persisted
        rows were absent; a single undeserializable row is skipped (logged)
        without discarding the rest.

        Durable-leak tradeoff (in-progress rows).  Only entries whose
        ``completed_at`` is set are TTL-evictable; an IN-PROGRESS entry
        (``completed_at is None``) is immortal by design (PRD §9.4), so its
        run never quiesces and :meth:`ReconReportStore.delete_run` (the only GC
        path) never fires for it.  With persistence this immortality becomes
        DURABLE: an abandoned/crashed run that files a stage but never reaches
        :meth:`complete` leaves its rows in ``recon_report_state.db``
        permanently, and this method resurrects them into memory on EVERY
        subsequent boot — an unbounded on-disk growth path the pure in-memory
        version bounded at process lifetime.  This mirrors the existing
        in-memory immortal-in-progress semantics (not a new regression, just a
        longer-lived one), and is a bounded practical risk because a completed
        run's rows DO self-GC at quiescence and only genuinely abandoned runs
        accumulate.  A bounded durable backstop (drop hydrated in-progress rows
        older than N) is deliberately NOT added here: every persisted timestamp
        (``created_at`` / ``completed_at`` / the row's ``updated_at``) is a
        MONOTONIC event-loop clock value, not wall-clock, and is not comparable
        across a restart (a fresh event loop restarts the clock near 0), so a
        reliable age-based sweep would require adding a wall-clock column — a
        store/serialization FORMAT change out of scope for this task (task 2716;
        the σ resume work owns interrupted-run adoption).  Operators reclaim
        space by deleting the abandoned run's rows (or the whole DB, which a
        fresh boot recreates empty).
        """
        if self._store is None:
            return
        try:
            rows = self._store.load_all()
        except Exception:
            logger.warning(
                'recon_report: failed to load persisted state on hydrate; '
                'starting with empty in-memory state',
                exc_info=True,
            )
            return
        for row in rows:
            run_id = row['run_id']
            stage = row['stage']
            entry_json = row['entry_json']
            try:
                entry = _deserialize_entry(entry_json)
                sig_anchor_slice, cited_task_slice = _deserialize_fold_anchor_slices(
                    entry_json
                )
            except Exception:
                logger.warning(
                    'recon_report: failed to deserialize persisted row '
                    'run_id=%r stage=%r on hydrate; skipping',
                    run_id,
                    stage,
                    exc_info=True,
                )
                continue
            # Re-anchor the TTL baseline across the restart boundary.
            # ``completed_at`` was stamped by the PRIOR process's clock, which
            # defaults to ``asyncio.get_running_loop().time()`` (monotonic) —
            # a value that restarts near 0 on a fresh event loop.  Restoring
            # that large stale value verbatim would make ``tick()``'s
            # ``now - completed_at`` negative against THIS process's fresh
            # monotonic clock, so it never exceeds ``ttl_seconds``: the hydrated
            # completed entry would never evict, its run would never quiesce,
            # and ``store.delete_run`` (the run-quiescence GC, step-12) would
            # never fire — ``recon_report_state.db`` rows (and their in-memory
            # entries) would leak unboundedly across every restart.  Resetting
            # to ``self._clock()`` counts the TTL from restart, so the entry
            # evicts ``ttl_seconds`` after boot via that same GC path.
            # In-progress entries (``completed_at is None``) are left untouched
            # so they stay immortal by design (PRD §9.4).  ``created_at`` is
            # diagnostics-only (not used for eviction) and is NOT re-anchored.
            if entry.completed_at is not None:
                entry.completed_at = self._clock()
            self._state[(run_id, stage)] = entry
            if row['is_active']:
                self._active[run_id] = stage
            # Rebuild the four run-level dedup indices by unioning this entry's
            # persisted per-entry mirrors / fold-anchor slices.  Keys never
            # collide across a run's entries (each sig/desc/cited-task key is
            # owned by exactly one entry), so plain dict.update is a safe union.
            finding_index = self._run_finding_index.setdefault(run_id, {})
            for f in entry.findings:
                finding_index[f.finding_id] = entry
            sig_index = self._run_sig_index.setdefault(run_id, {})
            sig_index.update(entry._signature_to_finding)
            sig_index.update(sig_anchor_slice)
            self._run_desc_index.setdefault(run_id, {}).update(
                entry._deschash_to_finding
            )
            self._run_cited_task_index.setdefault(run_id, {}).update(cited_task_slice)

    def start_persistence(self) -> None:
        """Open the shadow store and hydrate in-memory state — call ONCE at boot.

        A full no-op when ``self._store is None`` (persistence disabled — fresh
        in-process runs stay byte-identical).  Otherwise opens the persistent
        SQLite connection and replays any persisted rows into memory via
        :meth:`hydrate_from_store`, so a mid-stage restart resumes with the
        findings earlier stages already filed.

        Deliberately NOT called by the socket-free
        ``_build_recon_report_components`` factory (which only constructs the
        store) — mirroring the reaper-not-started-at-build invariant, only
        ``run_server`` opens it, right before :meth:`start_reaper`.  Paired with
        :meth:`stop_persistence` at shutdown.

        Best-effort, matching the loud-but-non-fatal posture of every other
        persistence touchpoint (:meth:`_persist_run`, :meth:`hydrate_from_store`,
        :meth:`tick`'s GC hook): a failure to open/hydrate the shadow store is
        logged loudly (WARNING, structured) and then DEGRADES to no persistence
        (``self._store`` is dropped to ``None``) rather than crashing server
        boot or spamming a per-write warning from a half-open store.  The
        server keeps running with an in-memory-only report state — exactly the
        byte-identical no-store path — instead of dying because the shadow
        store hiccuped.
        """
        if self._store is None:
            return
        try:
            self._store.open()
            self.hydrate_from_store()
        except Exception:
            logger.warning(
                'recon_report: failed to open/hydrate persistence store %r; '
                'continuing WITHOUT write-through persistence',
                getattr(self._store, 'db_path', None),
                exc_info=True,
            )
            with contextlib.suppress(Exception):
                self._store.close()
            self._store = None

    def stop_persistence(self) -> None:
        """Close the shadow store — call at shutdown.

        A no-op when ``self._store is None`` (persistence disabled or already
        degraded).  :meth:`ReconReportStore.close` is itself idempotent, so
        repeated calls are safe.
        """
        if self._store is None:
            return
        self._store.close()

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
        self._persist_run(run_id)
        return {'run_id': run_id, 'stage': stage}

    def add_finding(
        self,
        run_id: str,
        severity: str,
        category: str,
        description: str,
        suggested_action: str,
        actionable: bool | None = None,
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

        ``actionable`` (task-2432 bullet 1a) is a COMPUTED default: when the
        caller omits it (leaves it ``None``), it resolves to
        ``not (task_id is None or category.startswith('cross_project'))`` —
        i.e. False for a null-task_id finding or a ``cross_project*``
        category (both are informational/routing findings, not directly
        actionable), True otherwise. An explicit ``True``/``False`` from the
        caller is always honored regardless of task_id/category. The None
        check uses the raw ``task_id`` parameter (equivalent to the
        canonicalized form, since only ``None`` coerces to ``None``); the
        prefix check uses the POST-truncation ``category``.

        A null/missing ``flag_type`` on a re-raise of an already-flagged
        ``task_id`` inherits that task's single established flag_type before
        the signature lookup runs (task-2318), so an under-specified re-raise
        collapses onto the canonical finding instead of allocating a distinct
        ``(task_id, None)`` row.

        A comma-joined ``task_id`` (e.g. ``'5040,5149'``) is canonicalized
        via :func:`_canonicalize_task_id_string` (task-2432 bullet 4) BEFORE
        the dedup signature is computed and BEFORE it is stored on the
        resulting ``_Finding`` — split/stripped/deduped/sorted and rejoined,
        so two calls describing the same set of task ids in a different
        order (or with duplicated parts) collapse onto the same signature.
        A single-value ``task_id`` canonicalizes to itself, so this is a
        strict generalization of the pre-existing single-value dedup.
        Because the stored ``finding.task_id`` is already canonical,
        :meth:`_purge_finding` recomputes an identical signature from it —
        delete/refile stays consistent with no separate change there.

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
        if c_task_id is not None:
            c_task_id = _canonicalize_task_id_string(c_task_id)
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
        if actionable is None:
            actionable = not (task_id is None or category.startswith('cross_project'))
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
        self._persist_run(run_id)
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

        Removes the finding from ``entry.findings`` and every dedup index
        it may be registered under — ``_run_finding_index``, whichever of
        ``_run_sig_index`` / ``_run_desc_index`` it was filed under (and
        their per-entry mirrors), and ``_run_cited_task_index`` if it is a
        primary-cited-task fold anchor (task-2425) — via the single-sourced
        :meth:`_purge_finding` helper shared with the in-run cited-task
        fold's retract path in :meth:`cite_task`. Indices are recomputed
        from the finding's own already-canonicalized
        ``task_id``/``flag_type``/``description``/``cited_tasks``, so a
        corrected finding can be re-filed under the same signature,
        description, or primary cited task after retraction instead of
        bouncing off a stale ``duplicate_finding`` pointer.
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

        self._purge_finding(run_id, owning_entry, finding)
        self._persist_run(run_id)

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
        self._persist_run(run_id)
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
        self._persist_run(run_id)
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
                self._persist_run(run_id)
                return cached_response

        # First-time path
        entry.completed_at = self._clock()
        entry.summary = summary
        self._persist_run(run_id)
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
                'cited_runs': list(f.cited_runs),  # task-2595
                'standing_decision_id': f.standing_decision_id,  # task 2897 δ
            }
            # Cross-project routing taxonomy guard (task-2453): downgrade an
            # anchor-less cross_project_routing claim before the Fix-1 check
            # below, which reads only actionable/cited_* — ordering is safe.
            _apply_cross_project_routing_guard(finding_dict)
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
        copies of the five cited_* lists).

        Unlike :meth:`get_assembled_report`, this method does **NOT** apply
        Fix-1 read-time echo suppression (task-1654), nor the task-2453
        cross_project_routing taxonomy guard
        (:func:`_apply_cross_project_routing_guard`).  It is intentionally a
        raw, run-scoped read: task-1966 uses it as an independent "recon_report
        channel" for Stage 2 to poll for ``systemic_pattern`` findings, so that
        channel stays genuinely independent of the primary (Mem0 flagged-items)
        channel even when the latter is suppressed by a ``stage1_flag_suppression``
        record.  Skipping the task-2453 guard here is deliberate, not an
        oversight: this channel is filtered to ``category == 'systemic_pattern'``
        at the poll site (task_knowledge_sync.py), so a ``cross_project_routing``
        finding never flows through it regardless of citation state — applying
        the guard here would be cosmetic and would muddy this method's raw,
        no-suppression contract.  A caller reading ``cross_project_routing``
        findings through this channel should use :meth:`get_assembled_report`
        instead if it needs the guarded/downgraded taxonomy.

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
                    'cited_runs': list(f.cited_runs),  # task-2595
                    'standing_decision_id': f.standing_decision_id,  # task 2897 δ
                })
        return results

    def active_run_for_stage(self, stage: str, project_id: str) -> str | None:
        """Resolve the run_id currently in-progress on *stage* for *project_id*.

        Scans ``_active`` (run_id -> its CURRENT stage) rather than
        ``_state`` directly: a run that has since moved on to a later stage
        still has its earlier ``(run_id, stage)`` entry sitting in
        ``_state`` (never retroactively deleted), so ``_active`` is what
        distinguishes "this stage is what the run is doing right now" from
        "this run passed through this stage at some point". Returns
        ``None`` when no run is currently active on that stage for that
        project, or when the ``(run_id, stage)`` entry has since been
        completed (``complete()`` stamps ``completed_at`` but does not
        clear ``_active``).

        Used by :mod:`server.recon_lifecycle_filer` (task 2624) to resolve
        the live Stage-2 (``task_knowledge_sync``) run a code-detected
        ``task_lifecycle_reset_detected`` finding should land against.
        """
        for run_id, active_stage in self._active.items():
            if active_stage != stage:
                continue
            entry = self._state.get((run_id, stage))
            if entry is None or entry.completed_at is not None:
                continue
            if entry.project_id != project_id:
                continue
            return run_id
        return None

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
        ``_run_sig_index`` / ``_run_desc_index`` it was filed under,
        ``_run_cited_task_index`` if it is a primary-cited-task fold anchor
        (task-2425), and the derived projectless ``(cited_task_id,
        flag_type)`` key in ``_run_sig_index`` if it is an entity-scoped
        fold anchor (task-2432) — see :meth:`cite_task`'s docstring for both
        folds.

        Single-sourced (task-2425) by :meth:`delete_finding` and the in-run
        cited-task fold's retract path in :meth:`cite_task`, so the
        run-level dedup indices can never drift out of sync between the two
        removal paths, and neither path can leave a stale pointer to a
        finding that no longer exists.

        Identity-based removal (``is``, not ``==``) — *finding* is the exact
        object reference returned by :meth:`_resolve_finding`; see
        ``delete_finding``'s docstring for why this matters.

        Removal is wholesale: *finding* is dropped from ``owning_entry.findings``
        entirely, so any ``cited_entities`` / ``cited_edges`` / ``cited_memories``
        already recorded on it (e.g. via ``cite_entity`` / ``cite_edge`` /
        ``cite_memory`` calls made before this purge) are discarded along with
        it — not just the four dedup indices. For the cite_task fold this is
        intentional (task-2425): a finding judged a same-run duplicate of an
        earlier one contributes no new information, so citations already
        attached to it are not worth preserving. A caller that records
        cite_entity/cite_edge/cite_memory citations on a null-task_id finding
        BEFORE its first cite_task call should know those citations vanish
        silently if that cite_task later folds the finding into an existing
        duplicate.
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
            primary_key = _cited_task_key(primary['project_id'], primary['task_id'])
            run_cited_tasks = self._run_cited_task_index.get(run_id, {})
            if run_cited_tasks.get(primary_key) == finding.finding_id:
                run_cited_tasks.pop(primary_key, None)

        # task-2432: clear the entity-scoped fold's derived (cited_task_id,
        # flag_type) anchor. Unlike the project-scoped anchor above, this
        # is NOT gated on finding.task_id is None — a finding whose top-level
        # task_id matched its primary citation (single or comma-joined) can
        # also be a derived-sig anchor (see cite_task's docstring). A finding
        # purged MID-FOLD in cite_task (the losing side of a collision) has
        # empty cited_tasks at purge time — its citation is only appended
        # after the fold-check succeeds — so this guard correctly skips it;
        # only an established anchor (whose citation is already recorded)
        # clears its derived sig here.
        if finding.flag_type is not None and finding.cited_tasks:
            derived_sig = (_canonical_sig_field(finding.cited_tasks[0]['task_id']), finding.flag_type)
            run_sig_index = self._run_sig_index.get(run_id, {})
            if run_sig_index.get(derived_sig) == finding.finding_id:
                run_sig_index.pop(derived_sig, None)

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

        # Hook B (task 2897 δ): annotate the finding with an ACTIVE entity
        # standing decision, if one exists. Best-effort / fail-open — a missing
        # recon_ledger or ANY lookup error skips annotation and NEVER drops the
        # citation or the finding (PRD decision 3 never-drops + fail-open
        # ledger-read posture). Only the RETURNED response (nested under a
        # `standing_decision` key, keeping the {entity_uuid, canonical_name}
        # contract intact) and finding.standing_decision_id carry it; the stored
        # cited_entities citation is left unannotated so citation-identity /
        # cross-channel dedup are unaffected. Reuses the shared Hook A/B
        # active-decision-by-uuid lookup (INV-5) — no duplicate lookup.
        annotation: dict[str, Any] | None = None
        ledger = getattr(self._memory_service, 'recon_ledger', None)
        if ledger is not None:
            try:
                record = await ledger.get_active_entity_standing_decision(
                    finding_entry.project_id, node['uuid']
                )
            except Exception:
                logger.warning(
                    'Hook B: standing-decision lookup failed for entity %s '
                    '(project %s); skipping annotation (finding not dropped)',
                    node['uuid'],
                    finding_entry.project_id,
                    exc_info=True,
                )
            else:
                if record is not None:
                    standing_decision_id = f'{record.entity_uuid}:{record.flag_type}'
                    annotation = {
                        'standing_decision_id': standing_decision_id,
                        'grounds': record.flag_type,
                        'decided_at': record.created_at,
                        'summary': (
                            'Entity already adjudicated by an active standing '
                            f'decision (grounds={record.flag_type}); no '
                            're-investigation needed absent a new concrete fact.'
                        ),
                    }
                    finding.standing_decision_id = standing_decision_id

        self._persist_run(run_id)
        if annotation is not None:
            return {**citation, 'standing_decision': annotation}
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

        if not is_full_uuid(edge_uuid):
            return _ERR_INVALID_UUID_SHAPE.copy()

        if self._memory_service is None:
            return _ERR_SERVICE_UNAVAILABLE.copy()

        try:
            result = await self._memory_service.get_edge(edge_uuid, finding_entry.project_id)
        except EdgeNotFoundError:
            return _ERR_EDGE_NOT_FOUND.copy()

        citation = {'edge_uuid': edge_uuid, 'fact_text_snapshot': result['fact']}
        finding.cited_edges.append(citation)
        self._persist_run(run_id)
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
        Appends to finding.cited_tasks only on success, skipping the append when
        an identical {project_id, task_id} citation is already present so a
        re-citation of the same task (e.g. re-citing a finding's own primary
        task) stays idempotent instead of accumulating a duplicate entry
        (task-2425 amend). This idempotency check keys ONLY on
        {project_id, task_id} — first-cited title wins: if the upstream
        task's title has since changed, a re-citation is still skipped and
        the stored citation keeps the original title rather than refreshing
        it. Titles are cosmetic display text, not part of the citation's
        identity, so this staleness is accepted rather than reconciled.

        Two in-run folds anchor on this call — BOTH are CHECKED before
        EITHER registers, so a call that folds under either one always
        purges/returns rather than leaving a half-registered anchor from the
        other:

        1. Project-scoped null+null fold (task-2425): when *finding* has a
           null top-level task_id and this is its PRIMARY (first-ever)
           citation, the (project_id, task_id) pair doubles as an in-run
           dedup anchor keyed in ``_run_cited_task_index``. Findings with a
           real top-level task_id, and secondary (non-first) citations, are
           never anchors here. EXEMPTS ``memory_consolidator`` (Stage-1)
           findings, mirroring Fix-1's read-time ``stage !=
           'memory_consolidator'`` carve-out in :meth:`get_assembled_report`:
           two sibling Stage-1 findings that cite the same target stay
           distinct, and a Stage-2 echo of a Stage-1 citation is already
           suppressed at read time — so this fold only collapses same-run
           duplicates in a non-Stage-1 stage that Fix-1 cannot reach.

        2. Entity-scoped derived-signature fold (task-2432 bullets 1b/2/3):
           reuses ``_run_sig_index`` — the SAME index add_finding's ordinary
           (task_id, flag_type) signature lookup consults — with a
           PROJECTLESS derived key ``(canonical(task_id), finding.flag_type)``
           built from *this citation's* task_id. Registering it there means a
           LATER add_finding whose top-level task_id equals this cited tid
           collapses via add_finding's own ordinary signature lookup, with no
           add_finding change needed. Eligible only for a finding's PRIMARY
           citation (``cited_tasks`` still empty) with a non-null
           ``flag_type`` (a null flag_type can't form a meaningful derived
           signature — every null-flag_type citation of the same task would
           collide), and only when ``finding.task_id is None`` or this
           citation's task_id (canonicalized) is a MEMBER of
           ``finding.task_id``'s comma-split parts (:func:`_split_task_id_parts`)
           — a single-value top-level task_id is a singleton part set, so
           this subsumes the equality case. A None, a foreign-single, a
           local-single, and a comma-joined top-level task_id therefore all
           fold through this ONE path when they share a cited task; a
           finding citing a task outside its own top-level part set is never
           folded (see test_non_null_task_id_finding_is_never_a_fold_anchor
           and test_local_task_id_citing_different_task_is_not_folded).
           Unlike fold 1, this fold has NO memory_consolidator carve-out:
           the derived signature lives in the same run-wide, cross-stage
           index add_finding already consults from every stage, so
           exempting one stage from registering it would just let that
           stage's findings silently evade the whole-run fold.
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

        # In-run cited-task folds (task-2425 project-scoped; task-2432
        # entity-scoped) — see docstring above. Both folds' EXISTENCE CHECKS
        # run before EITHER registers.
        project_fold_eligible = (
            finding.task_id is None
            and not finding.cited_tasks
            and finding_entry.stage != 'memory_consolidator'
        )
        # Always compute the concrete key (pure, side-effect-free) so its
        # static type stays non-Optional for the dict lookup/assignment below
        # — mirrors _purge_finding's primary_key/derived_sig convention.
        # Its *use* is still gated on project_fold_eligible.
        cited_task_key = _cited_task_key(project_id, task_id)
        project_existing_id = (
            self._run_cited_task_index.get(run_id, {}).get(cited_task_key)
            if project_fold_eligible
            else None
        )

        c_cited_task_id = _canonical_sig_field(task_id)
        entity_fold_eligible = (
            not finding.cited_tasks
            and finding.flag_type is not None
            and (
                finding.task_id is None
                or c_cited_task_id in _split_task_id_parts(finding.task_id)
            )
        )
        # Same rationale as cited_task_key above: compute unconditionally,
        # gate the use on entity_fold_eligible.
        derived_sig = (c_cited_task_id, finding.flag_type)
        entity_existing_id = (
            self._run_sig_index.get(run_id, {}).get(derived_sig)
            if entity_fold_eligible
            else None
        )

        # Sequential (not project_hit/entity_hit booleans + a re-derived
        # existing_id) so pyright narrows each *_existing_id to `str` from
        # its own `is not None` check at the call site — see cited_task_key
        # comment above for why the naive boolean-flag version doesn't
        # narrow. Semantics are unchanged: project fold takes priority when
        # both would hit, purge runs exactly once, either way.
        if project_existing_id is not None and project_existing_id != finding.finding_id:
            self._purge_finding(run_id, finding_entry, finding)
            self._persist_run(run_id)
            return _duplicate_finding_error(project_existing_id)
        if entity_existing_id is not None and entity_existing_id != finding.finding_id:
            self._purge_finding(run_id, finding_entry, finding)
            self._persist_run(run_id)
            return _duplicate_finding_error(entity_existing_id)

        if project_fold_eligible:
            self._run_cited_task_index.setdefault(run_id, {})[cited_task_key] = finding.finding_id
        if entity_fold_eligible:
            self._run_sig_index.setdefault(run_id, {})[derived_sig] = finding.finding_id

        # task-2425 amend: skip the append when an identical {project_id,
        # task_id} citation is already present. Without this, re-citing a
        # finding's own primary task (see
        # test_only_primary_citation_is_a_fold_anchor) appends a second,
        # redundant citation entry — harmless to the fold itself (which
        # always keys off cited_tasks[0]) but it lets cited_tasks accumulate
        # duplicate rows. Keyed on (project_id, task_id) only, NOT title —
        # first-cited title wins; a re-citation after the upstream title
        # changed is still skipped rather than refreshing the stored title
        # (see cite_task's docstring).
        already_cited = any(
            c['project_id'] == project_id and c['task_id'] == task_id
            for c in finding.cited_tasks
        )
        if not already_cited:
            finding.cited_tasks.append(citation)
        self._persist_run(run_id)
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

        if not is_full_uuid(memory_id):
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
        self._persist_run(run_id)
        return citation

    async def cite_run(
        self,
        run_id: str,
        finding_id: str,
        cited_run_id: str,
    ) -> dict[str, Any]:
        """Validate *cited_run_id* shape, confirm it exists, and record the citation.

        Closes the gap named by task 2595: a run_id quoted inline in a
        finding's free-text ``description``/``suggested_action`` was
        previously validated by NONE of cite_entity/cite_edge/cite_task/
        cite_memory, so an LLM re-typing a historical run_id from memory
        (instead of copying it verbatim off a fresh tool result) could
        silently drift by a hex group with nothing downstream catching it
        until a future agent manually re-fetched the source record. cite_run
        hands the caller a structured ``run_not_found`` error to self-correct
        on mid-cycle instead.

        Existence is confirmed via
        ``memory_service.count_memories_by_metadata(project_id, {'run_id':
        cited_run_id})`` rather than a live-run registry lookup:
        reconciliation runs always write cycle summaries to mem0 keyed by
        their run_id, so a >0 mem0 count is a sound existence proxy — the
        same signal the originating incident's remediation used to
        self-catch the bug by hand (a 0-count
        ``get_memories_by_metadata(run_id=...)`` call). No memory_service
        change was needed: ``count_memories_by_metadata`` already existed as
        the exact Qdrant metadata-equality count primitive.

        Caveat (reviewed task-2595 amendment): this existence check is
        Mem0/Qdrant-only, so it can false-negative for a run that genuinely
        existed — a run whose cycle summary predates the
        ``metadata.run_id`` convention, or whose provenance lives only in
        Graphiti, will also read as ``run_not_found``. Treat ``run_not_found``
        as "not confirmed via mem0", not as infallible proof the run_id
        never existed — it is a strong self-correction signal for the
        common case (a mistyped/re-typed id), not a guarantee for every
        historical run.

        Returns {run_id, match_count} on success, or a structured error dict
        (run_id_unknown / finding_unknown / invalid_uuid_shape / run_not_found
        / service_not_configured). UUID shape is checked before any service
        call. Appends to finding.cited_runs only on success, always (no
        dedup) — matching cite_edge/cite_memory rather than cite_task's fold,
        whose dedup-anchor machinery a run citation does not need.

        ``cited_runs`` is deliberately EXCLUDED from
        :func:`_citation_identities` (and therefore from Fix-1 same-run echo
        suppression): a cited run_id is provenance for a claim, not a
        task/entity/edge/memory identity that suppression reasons about.
        Folding it in could silently change which findings get suppressed;
        leaving it out keeps all existing suppression/dedup behavior
        byte-identical.
        """
        entry = self._resolve_entry(run_id)
        if entry is None:
            return _ERR_RUN_UNKNOWN.copy()

        resolved = self._resolve_finding(run_id, finding_id)
        if resolved is None:
            return _ERR_FINDING_UNKNOWN.copy()
        finding_entry, finding = resolved

        if not is_full_uuid(cited_run_id):
            return _ERR_INVALID_UUID_SHAPE.copy()

        if self._memory_service is None:
            return _ERR_SERVICE_UNAVAILABLE.copy()

        # Deliberately not narrowed to a specific exception type here, unlike
        # cite_edge's `except EdgeNotFoundError` or cite_memory's `except
        # (EdgeNotFoundError, MemoryNotFoundError)`: a transient backend/
        # connection failure inside count_memories_by_metadata propagates as
        # a raw ToolError to the caller instead of being swallowed into a
        # fail-safe result. This matches cite_entity's get_entity call and
        # cite_task's get_task call, which are equally unnarrowed — a
        # transient hiccup should surface loudly rather than risk masquerading
        # as a (possibly wrong) "confirmed absent" run_not_found verdict.
        count = await self._memory_service.count_memories_by_metadata(
            finding_entry.project_id, {'run_id': cited_run_id}
        )
        if count == 0:
            return _ERR_RUN_NOT_FOUND.copy()

        citation = {'run_id': cited_run_id, 'match_count': count}
        finding.cited_runs.append(citation)
        self._persist_run(run_id)
        return citation

    async def write_entity_standing_decision(
        self,
        *,
        project_id: str,
        entity_uuid: str,
        grounds: str,
        evidence: Any,
    ) -> dict[str, Any]:
        """Write an entity standing decision to α's ledger (Stage-2, always gated).

        Delegates to
        :func:`~fused_memory.reconciliation.standing_decision_writer.write_entity_standing_decision`
        WITHOUT ``authorized_by`` — so the two-armed evidence gate is enforced on
        this (the sole Stage-2) write path; the operator/backfill (η) bypass seam
        lives on the helper, never on this tool.

        Returns the helper's success dict on a write; the structured
        ``insufficient_evidence`` rejection (unmet_arms + hint) when the gate is
        not satisfied; ``_ERR_SERVICE_UNAVAILABLE`` when no memory_service is
        wired OR its ``recon_ledger`` is unset (an unwired ledger raises
        :class:`LedgerUnavailable`, mapped here); or ``_ERR_INVALID_GROUNDS``
        (with the ledger's message as a hint) when ``grounds`` is outside
        ``GROUNDS_ENUM``. A graphiti sampling failure propagates loudly (INV-1)
        rather than persisting a poisoned row.
        """
        if self._memory_service is None:
            return _ERR_SERVICE_UNAVAILABLE.copy()

        from fused_memory.reconciliation.standing_decision_writer import (  # noqa: PLC0415
            EvidenceGateRejected,
            LedgerUnavailable,
        )
        from fused_memory.reconciliation.standing_decision_writer import (  # noqa: PLC0415
            write_entity_standing_decision as _write_helper,
        )

        try:
            return await _write_helper(
                self._memory_service,
                project_id=project_id,
                entity_uuid=entity_uuid,
                grounds=grounds,
                evidence=evidence,
            )
        except EvidenceGateRejected as exc:
            return dict(exc.rejection)
        except LedgerUnavailable:
            # recon_ledger not wired on the memory service: a durable standing
            # decision cannot be persisted, so surface the same structured
            # service-unavailable error the docstring promises rather than
            # letting the helper's loud failure propagate as a raw exception.
            return _ERR_SERVICE_UNAVAILABLE.copy()
        except ValueError as exc:
            rejected = dict(_ERR_INVALID_GROUNDS)
            rejected['hint'] = str(exc)
            return rejected

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

        The four shared run-level dedup indices (``_run_sig_index`` /
        ``_run_desc_index`` / ``_run_finding_index`` / ``_run_cited_task_index``)
        are RUN-QUIESCENCE scoped, not per-entry (task-2088). They are keyed for the WHOLE run
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
        (in-progress, or completed but not yet past its own TTL), all four
        indices for that run_id are preserved untouched, even for stages that
        have already been evicted. Only when a run's LAST entry evicts do we
        release its four indices, wholesale via ``pop(rid, None)`` rather
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
        that never calls complete() pins that run_id's four indices, and
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
            # has no surviving entry is fully quiescent — release its four
            # shared indices wholesale.
            live_run_ids = {r for (r, _s) in self._state}
            for rid in evicted_run_ids:
                if rid not in live_run_ids:
                    self._run_sig_index.pop(rid, None)
                    self._run_finding_index.pop(rid, None)
                    self._run_desc_index.pop(rid, None)
                    self._run_cited_task_index.pop(rid, None)
                    # GC the run's persisted rows at quiescence (task 2716) —
                    # "rows are GC'd with their run".  Best-effort: a shadow-store
                    # hiccup must not abort the reaper sweep or the in-memory
                    # eviction just performed (mirrors _persist_run's
                    # loud-but-non-fatal posture).  No-op when store is None.
                    if self._store is not None:
                        try:
                            self._store.delete_run(rid)
                        except Exception:
                            logger.warning(
                                'recon_report: failed to GC persisted rows for '
                                'run_id=%r at quiescence',
                                rid,
                                exc_info=True,
                            )
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
       cite_entity, cite_edge, cite_task, cite_memory, cite_run.

Usage pattern (per PRD §9.2):
1. start_report — open a new report at the start of a stage run.
2. add_finding — append a diagnostic finding (deduplicated by task_id + flag_type
                  across ALL stages of the same run_id).  Overlength
                  description/suggested_action are truncated with a
                  'warnings' entry on the response, never rejected.
                  actionable defaults to False when task_id is None or
                  category starts with 'cross_project'; True otherwise. An
                  explicit actionable=True/False is always honored.
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
10. cite_run(run_id, finding_id, cited_run_id) — confirm a quoted run_id exists
                  (via mem0 count) and attach it.  Copy cited_run_id verbatim
                  from a fresh tool result's run_id/metadata.run_id field —
                  never re-type or paraphrase it from memory.
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
        actionable: bool | None = None,
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

        actionable (task-2432 bullet 1a): when omitted, defaults to False for
        a null task_id or a cross_project* category, True otherwise — see
        ReconReportState.add_finding's docstring. The `None` sentinel is
        passed through unchanged so the state method's computed default
        applies; an explicit True/False is always honored.
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

    @mcp.tool()
    async def cite_run(run_id: str, finding_id: str, cited_run_id: str) -> dict:
        """Confirm a cited run_id exists (via mem0 count) and attach it to a finding.

        PRD §9.2 (task-2595) — cite_run(run_id, finding_id, cited_run_id).
        Returns {run_id, match_count} or a structured error dict.
        invalid_uuid_shape when cited_run_id doesn't match the canonical UUID regex.
        run_not_found when the UUID is valid but no mem0 records carry it as
        their run_id (count_memories_by_metadata returns 0) — this is the
        structural fix for run_id transcription drift: copy cited_run_id
        verbatim from the run_id/metadata.run_id field of a fresh tool
        result, never re-type or paraphrase it from memory. Note: this check
        is mem0-only, so a legacy run predating the metadata.run_id
        convention (or one whose provenance lives only in Graphiti) can also
        surface as run_not_found even though it once existed.
        """
        return await state.cite_run(
            run_id=run_id, finding_id=finding_id, cited_run_id=cited_run_id
        )

    @mcp.tool()
    async def write_entity_standing_decision(
        project_id: str,
        entity_uuid: str,
        grounds: str,
        evidence: list[dict] | None = None,
    ) -> dict:
        """Write an entity standing decision to the reconciliation ledger (task 2895 β).

        Stage-2 ONLY (blocked in Stage 1 / Stage 3 — the first recon-report tool
        with durable SQLite-ledger writes). Records that a class of complaint
        about *entity_uuid* — identified by *grounds* (a closed-enum value) — has
        been investigated and dismissed, so γ/δ can filter/annotate future recon
        flags instead of re-raising them.

        This path is ALWAYS evidence-gated (there is deliberately no
        ``authorized_by`` parameter): the write succeeds only if EITHER arm holds
        — arm 1: ≥1 cited, locally-resolvable, human-authored mem0 evidence
        record; arm 2: ≥3 investigation_outcome mem0 records for this entity with
        actionable=false and distinct run_ids. *evidence* is a list of cited-ref
        dicts ({type, id, ...}); mem0 refs are resolved for provenance, foreign
        refs (escalation/task ids) are recorded but never count toward a gate arm.

        Returns {status:'written', entity_uuid, grounds, edge_count_at_decision,
        expires_at, decided_at} on success, or a structured error dict:
        insufficient_evidence (unmet_arms + hint) when neither arm is satisfied,
        invalid_grounds when grounds is outside the enum, or service_not_configured
        when the memory service is unavailable.
        """
        return await state.write_entity_standing_decision(
            project_id=project_id,
            entity_uuid=entity_uuid,
            grounds=grounds,
            evidence=evidence or [],
        )

    return mcp


def get_recon_report_tool_signatures() -> dict[str, inspect.Signature]:
    """Return ``{tool_name: signature}`` for every tool on a throwaway recon_report server.

    Builds a state-backed server via :func:`create_recon_report_server` and reads
    each registered tool's call signature (task-2559). This is the single place
    that reaches into FastMCP's tool-manager internals
    (``mcp._tool_manager._tools[name].fn``) on behalf of tool-guidance generation
    (see ``reconciliation/prompts/__init__.py``'s ``render_recon_report_tool_guidance``)
    — centralizing that coupling here means a FastMCP version bump that changes
    this shape needs a fix in only one place, and callers get an actionable
    RuntimeError instead of a bare AttributeError deep inside prompt generation.
    """
    state = ReconReportState(ttl_seconds=300, clock=lambda: 0.0)
    mcp = create_recon_report_server(state)
    try:
        tools = mcp._tool_manager._tools
        return {name: inspect.signature(tool.fn) for name, tool in tools.items()}
    except AttributeError as exc:
        raise RuntimeError(
            "FastMCP's tool-manager internals (_tool_manager._tools[name].fn) have "
            'changed shape; get_recon_report_tool_signatures() needs updating for '
            'this FastMCP version.'
        ) from exc
