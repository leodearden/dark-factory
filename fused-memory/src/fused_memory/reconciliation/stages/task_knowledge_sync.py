"""Stage 2: Task-Knowledge Sync — reconcile task state against memory state.
Stage 3: Cross-System Integrity Check — read-only verification."""

from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import logging
import sys
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol

from fused_memory.backends.task_backend_errors import DuplicateCandidateKeyError
from fused_memory.mcp_tools.scheduler_state import read_scheduler_state
from fused_memory.middleware.task_interceptor import TERMINAL_STATUSES
from fused_memory.models.reconciliation import (
    ReconciliationEvent,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.models.scope import ProjectRoot, ProjectScope, resolve_main_checkout
from fused_memory.reconciliation.cli_stage_runner import (
    STAGE2_DISALLOWED,
    STAGE3_DISALLOWED,
    STAGE3_REPORT_SCHEMA,
)
from fused_memory.reconciliation.flag_dedup import (
    _content_fingerprint,
    _normalize_content_description,
    acknowledge_resolved_flags,
    compute_flag_signature,
    filter_blocked_snapshot_findings,
    filter_contamination_ceiling_findings,
    filter_false_phantom_task_creation_flags,
)
from fused_memory.reconciliation.mem0_tombstone import (
    is_protected_mirror_record,
    record_mem0_deletion_tombstones,
)
from fused_memory.reconciliation.policies import is_snapshot_write_blocked
from fused_memory.reconciliation.prompts import (
    _STAGE2_PROJECT_ID_GUIDELINE,
    _STAGE3_PROJECT_ID_GUIDELINE,
)
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt
from fused_memory.reconciliation.recon_pool_map import (
    STAGE2_CYCLE_SUMMARY_RECON_POOL as _STAGE2_CYCLE_SUMMARY_RECON_POOL,
)
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.standing_decision_constants import (
    EXPIRY_REASON_GROWTH,
    STATE_ACTIVE,
)
from fused_memory.reconciliation.standing_decision_writer import (
    expire_entity_standing_decision,
)
from fused_memory.reconciliation.summary_pool import (
    write_cycle_summary,
)
from fused_memory.reconciliation.task_count_snapshot_cadence import (
    SNAPSHOT_PRUNE_ENUMERATED_STAT_KEY,
    SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY,
    SNAPSHOT_PRUNE_TRUNCATED_STAT_KEY,
    SNAPSHOT_PRUNED_STAT_KEY,
    SNAPSHOT_WRITTEN_STAT_KEY,
    TASK_COUNT_SNAPSHOT_CATEGORY,
    TASK_COUNT_SNAPSHOT_KIND,
    build_task_count_snapshot_content,
    build_task_count_snapshot_unavailable_content,
)
from fused_memory.reconciliation.task_filter import (
    FilteredTaskTree,
    detect_task_dump_contamination,
    filter_task_tree,
    format_task_list,
    id_key,
    render_active_section,
    select_done_since_boundary,
)
from fused_memory.services.live_workflow_detector import (
    corroboration_for_task,
    detect_live_workflow,
    is_pure_gate_metadata,
)
from fused_memory.services.orchestrator_detector import (
    is_orchestrator_live_for,
    orchestrator_started_at,
)
from fused_memory.utils.async_utils import gather_collect

try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - escalation package is optional
    Escalation = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# Projects allowed to use the briefing-refresh hook.  This is a reify-specific
# feature; gating on project_id prevents accidental triggering by other projects
# that happen to have the same file layout.  Extend when needed.
_BRIEFING_REFRESH_PROJECT_ALLOWLIST: frozenset[str] = frozenset({'reify'})

# Persistence-threshold constant for FIX D (stale-flag escalation guard).
# A flag that has survived this many Stage 2 cycles without being deleted is
# considered stale and is surfaced in the payload for operator escalation.
STAGE2_FLAG_PERSISTENCE_THRESHOLD: int = 3

# Clock-skew grace period for the run-window guard in _marker_is_within_run_window.
# run_window_start is sourced from the orchestrator clock (journal-persisted), while
# created_at is stamped by the Mem0 server clock.  A small negative inter-host skew
# would cause a legitimate same-cycle marker (created_at slightly before
# run_window_start) to be misclassified as stale.  We absorb that risk by lowering
# the effective threshold by this amount.  30 seconds covers typical NTP drift
# between containers; the worst-case outcome of over-rescuing is one extra LLM render
# of a marker that should wait — strictly safer than the underlying bug (deletion).
_CLOCK_SKEW_GRACE: timedelta = timedelta(seconds=30)


def _assume_utc(dt: datetime) -> datetime:
    """Return *dt* with UTC timezone attached if it is naive; return *dt* unchanged otherwise.

    Centralises the ``"naive datetimes from our journal/Mem0 are UTC"`` convention so
    that the assumed-timezone behaviour has a single source of truth.
    """
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _suppress_same_run_human_operator_dups(
    stage2_flagged: list[dict],
    stage1_flagged: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition *stage2_flagged* into (kept, suppressed) by comparing against Stage 1.

    A Stage 2 item is **suppressed** iff ALL of the following hold:

    1. The Stage 2 item has ``resolution_status == 'human_operator_required'``.
    2. Stage 1 has an item with the **same** ``(task_id, flag_type)`` pair whose
       ``resolution_status`` is also ``'human_operator_required'``.

    In other words, this filter enforces the 3-tuple
    ``(task_id, flag_type, resolution_status='human_operator_required')``
    deduplication described in task 1154, but expresses the ``resolution_status``
    constraint as a predicate on both sides rather than as a key dimension.
    This mirrors the established ``compute_flag_signature`` convention in
    ``flag_dedup.py``, which uses a 2-tuple ``(task_id, flag_type)`` key
    and treats ``None`` values as a "skip this entry" sentinel.

    **str() coercion**: ``task_id`` and ``flag_type`` are coerced via ``str()``
    when building or looking up keys.  LLM output frequently emits ``task_id``
    as an integer in some cycles and as a string in others; without coercion,
    ``42 != '42'`` would silently miss duplicates.

    **None-skip rule**: Stage 1 entries where ``task_id`` or ``flag_type`` is
    ``None`` (absent from the dict) form no key and therefore can never suppress
    any Stage 2 item.  Falsy-but-valid values like ``task_id=0`` or
    ``flag_type=''`` *do* form valid keys (matching ``compute_flag_signature``).

    Args:
        stage2_flagged: The ``items_flagged`` list from Stage 2's ``StageReport``.
        stage1_flagged: The ``items_flagged`` list from Stage 1's ``StageReport``.

    Returns:
        ``(kept, suppressed)`` — two lists that together partition *stage2_flagged*.
    """
    # Build the set of (task_id, flag_type) keys from Stage 1 entries that are
    # human_operator_required AND have both fields present.  Delegates key
    # construction to compute_flag_signature so the coercion logic stays in
    # one place and both modules stay in sync if it ever changes.
    stage1_hor_keys: set[tuple[str, str]] = {
        sig
        for item in stage1_flagged
        if item.get('resolution_status') == 'human_operator_required'
        and (sig := compute_flag_signature(item)) is not None
    }

    kept: list[dict] = []
    suppressed: list[dict] = []
    for item in stage2_flagged:
        if item.get('resolution_status') == 'human_operator_required':
            sig = compute_flag_signature(item)
            if sig is not None and sig in stage1_hor_keys:
                suppressed.append(item)
                continue
        kept.append(item)

    return kept, suppressed


# ── Stage 2 post-flight guard helpers (task 1137) ────────────────────────────


async def _acknowledge_resolved_stage1_markers(
    memory_service: Any,
    project_id: str,
    run_id: str,
    flag_deleted_records: object,
    rendered_flags: object,
) -> int:
    """Best-effort: tag Channel-1 ``stage1_flag_marker``(s) for flags Stage 2 resolved.

    Joins *flag_deleted_records* (Stage 2's own FIX C deletion action records,
    each carrying a ``flag_id``) against *rendered_flags* (the ``combined_flags``
    Stage 2 rendered into its prompt this run — see ``assemble_payload`` —
    each carrying ``flag_id``/``task_id``/``flag_type``) on ``flag_id`` to
    recover the ``(task_id, flag_type)`` signature for each resolved flag, then
    delegates to :func:`~fused_memory.reconciliation.flag_dedup.acknowledge_resolved_flags`
    with ``mode='tag'`` so the Channel-1 marker is tagged
    ``addressed_by=<run_id>`` rather than deleted outright (task-2029 scenario b).

    A record is a join candidate whenever it is a dict carrying a ``flag_id``
    key — the documented shape stamps ``action='flag_deleted'``, but the join
    itself does not require that key, only ``flag_id``. Unmatched records
    (no ``rendered_flags`` entry with that ``flag_id``, or a match missing
    ``task_id``/``flag_type``) are skipped.

    Best-effort and non-raising: a malformed/absent *flag_deleted_records* or
    *rendered_flags* (not a non-empty list) short-circuits to ``0`` with no
    I/O; any exception raised while acknowledging is caught and logged at
    WARNING, returning ``0``.

    Args:
        memory_service: Mem0 service forwarded to ``acknowledge_resolved_flags``.
        project_id: Project scope.
        run_id: Current reconciliation run identifier.
        flag_deleted_records: ``report.stats['flag_deleted_records']`` from
            this Stage 2 run.
        rendered_flags: The ``combined_flags`` Stage 2 rendered this run.

    Returns:
        Count of Channel-1 markers acknowledged (0 on no-op/failure).
    """
    if not isinstance(flag_deleted_records, list) or not flag_deleted_records:
        return 0
    if not isinstance(rendered_flags, list) or not rendered_flags:
        return 0

    by_flag_id = {
        f['flag_id']: f
        for f in rendered_flags
        if isinstance(f, dict) and f.get('flag_id') is not None
    }

    collected: list[dict] = []
    for record in flag_deleted_records:
        if not isinstance(record, dict):
            continue
        flag_id = record.get('flag_id')
        if flag_id is None:
            continue
        matched = by_flag_id.get(flag_id)
        if matched is None:
            continue
        if matched.get('task_id') is None or matched.get('flag_type') is None:
            continue
        collected.append(matched)

    if not collected:
        return 0

    try:
        return await acknowledge_resolved_flags(
            memory_service,
            project_id,
            run_id,
            collected,
            mode='tag',
        )
    except Exception:
        logger.warning(
            'reconciliation._acknowledge_resolved_stage1_markers: '
            'acknowledge_resolved_flags failed for run_id=%s',
            run_id,
            exc_info=True,
        )
        return 0


#: resolve_ticket statuses that constitute a confirmed task creation (task 3046).
_TASK_CREATED_SUCCESS_STATUSES: frozenset[str] = frozenset({'created', 'combined'})


def _count_valid_task_created_records(records: object) -> int:
    """Return the deduped count of confirmed task creations in *records* (task 3046).

    *records* is ``report.stats['task_created_records']`` — the action-shaped
    ground truth the '## Task-Creation Accounting' prompt section mandates
    Stage 2 append to at the moment each ``resolve_ticket`` call confirms a
    creation, modeled directly on ``flag_deleted_records``. A record counts
    only when its ``status`` (case/whitespace-insensitive) is ``created`` or
    ``combined`` AND it carries a non-empty ``task_id``; ``failed`` is NEVER
    counted regardless of whether a ``task_id`` is present, mirroring the
    '## Verifying Task Operations' confirmation rule so the prompt and this
    helper cannot drift on what counts.

    Deduplication is keyed on ``(project_id, str(task_id))``, NOT on
    ``task_id`` alone: Cross-Project Routing means the same numeric id can
    legitimately be filed under two different projects in the same cycle
    (Taskmaster ids are per-project), and that is two distinct tasks, not a
    duplicate report of one.

    Best-effort and non-raising throughout, mirroring
    ``_acknowledge_resolved_stage1_markers`` above: *records* must be a
    non-empty ``list`` or this returns ``0``; non-``dict`` entries and
    entries that raise while being inspected are silently skipped rather
    than aborting the count — a malformed record degrades to "not counted",
    never to an exception that would corrupt an otherwise-good stage report.
    """
    if not isinstance(records, list) or not records:
        return 0

    seen: set[tuple[str | None, str]] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            status = record.get('status')
            if (
                not isinstance(status, str)
                or status.strip().lower() not in _TASK_CREATED_SUCCESS_STATUSES
            ):
                continue
            task_id = record.get('task_id')
            if task_id is None:
                continue
            task_id_str = str(task_id).strip()
            if not task_id_str:
                continue
            project_id = record.get('project_id')
            project_id_str = str(project_id) if project_id is not None else None
        except Exception:
            continue
        seen.add((project_id_str, task_id_str))

    return len(seen)


def _marker_is_within_run_window(created_at: object, run_window_start: object) -> bool:
    """Return True iff *created_at* falls within the current run window.

    A marker is considered within the run window when *run_window_start* is a
    :class:`~datetime.datetime` instance AND *created_at* is a non-empty string
    that parses as an ISO-8601 datetime that is >= *run_window_start* minus
    :data:`_CLOCK_SKEW_GRACE` (30 s by default).

    The grace period absorbs inter-host clock skew between the orchestrator
    (which persists ``run_window_start`` via the journal) and the Mem0 server
    (which stamps ``created_at``).  Without it, a legitimate same-cycle marker
    written fractionally before the persisted start time would be swept.

    Naive datetimes are assumed UTC on **both** sides of the comparison via
    :func:`_assume_utc`:

    * A naive parsed *created_at* is normalised (existing behaviour, documented
      convention).
    * A naive *run_window_start* is also normalised (task-1383 hardening).
      Without this, the comparison
      ``parsed(tz-aware) >= run_window_start(naive) - _CLOCK_SKEW_GRACE`` raises
      ``TypeError``, which the ``except`` clause silently swallows, causing the
      guard to return ``False`` for *every* marker and disabling the run-window
      guard for the entire cycle (the task-1369 regression).  Assuming UTC is
      safe: the journal persists ``started_at`` via ``datetime.now(UTC)``.

    Any type mismatch or parse failure returns ``False`` (fail-open: falls back
    to the existing pure run_id partition behaviour).

    .. note::
        Only a lower-bound check is applied.  The absence of an upper bound is
        intentional: per-project run serialization (harness run lock) guarantees
        that at most one reconciliation run for a given project is active at any
        time, so a marker written by a *later* run cannot exist yet.  If that
        serialization assumption is ever relaxed, an upper bound of
        ``datetime.now(UTC)`` should be added here as defence-in-depth.
    """
    if not isinstance(run_window_start, datetime):
        return False
    run_window_start = _assume_utc(run_window_start)
    if not created_at or not isinstance(created_at, str):
        return False
    try:
        parsed = datetime.fromisoformat(created_at)
        return _assume_utc(parsed) >= run_window_start - _CLOCK_SKEW_GRACE
    except (ValueError, TypeError):
        return False


class Stage2FlagPartition(NamedTuple):
    """Partition result from :func:`_query_stage2_flags`.

    Using a NamedTuple makes all return values self-documenting at call sites
    and avoids positional-index surprises when the return shape evolves.

    Attributes:
        current: Full dict records whose ``metadata.run_id`` matches the active
            run, OR whose Mem0 ``created_at`` timestamp falls within the current
            run window (run-window guard, task-1369).  Rendered to the Stage 2
            LLM for FIX-C processing.
        stale_missing_run_id_ids: ``id`` strings for records whose
            ``metadata.run_id`` is absent or falsy (empty string) AND whose
            ``created_at`` is out of the run window (or unknown).  These
            indicate Stage 1 producer drift from a prior cycle — the LLM wrote
            a flag without the required ``run_id`` field (see
            prompts/stage1.py).  A WARNING is logged by
            :func:`_query_stage2_flags` when this bucket is non-empty.  These
            are excluded from what is rendered to the Stage 2 LLM; residue is
            reaped by the recon_ledger's TTL/terminal-task GC pass
            (:func:`_gc_recon_markers`) rather than an immediate per-cycle
            delete (task 2228 W5-κ).
        stale_mismatched_run_id_ids: ``id`` strings for records whose
            ``metadata.run_id`` is present and truthy but does not match the
            current ``run_id`` AND whose ``created_at`` is out of the run
            window.  These are normal prior-cycle residue, excluded from what
            is rendered to the Stage 2 LLM and reaped the same way (see
            ``stale_missing_run_id_ids`` above).
        rescued_ids: ``id`` strings for markers rescued by the run-window guard
            (a subset of ``current``).  Non-empty indicates Stage 1 producer
            drift within the CURRENT cycle — the LLM omitted or mis-stamped
            ``metadata.run_id`` on a flag it wrote during this run, but the
            marker was still surfaced to Stage 2 (not swept).  Populated
            exclusively by the two rescue branches inside
            :func:`_query_stage2_flags`; this is the single source of truth for
            the rescued count (task-1381).
    """

    current: list[dict]
    stale_missing_run_id_ids: list[str]
    stale_mismatched_run_id_ids: list[str]
    rescued_ids: list[str]

    @property
    def stale_all_ids(self) -> list[str]:
        """Combined stale IDs for sweeping (missing + mismatched)."""
        return self.stale_missing_run_id_ids + self.stale_mismatched_run_id_ids


async def _query_stage2_flags(
    memory_service,
    project_id: str,
    current_run_id: str,
    run_window_start: datetime | None = None,
) -> Stage2FlagPartition:
    """Query Mem0 for active Stage-2-destined flags and partition by run_id.

    Searches for memories with ``metadata.flag_for_stage2=true`` (the only
    supported convention — the ``stage1_flag_marker`` key was a never-shipped
    alias and is not checked here; see task-1139 reviewer note on dead code).
    Any other memories are discarded.

    Results are partitioned into four groups:

    * **current** — full dict records whose ``metadata.run_id`` is present,
      non-empty, and matches ``current_run_id`` after ``str()`` coercion.
      These are rendered to the Stage 2 LLM for FIX-C processing.  Markers
      with a stale or absent ``run_id`` that were written during the current
      run window (``created_at >= run_window_start``) are also routed here —
      the run-window guard rescues same-cycle Stage-1 writes whose
      ``metadata.run_id`` was omitted or mis-stamped by the LLM producer.
    * **stale_missing_run_id_ids** — ``id`` strings for records whose
      ``metadata.run_id`` is absent or falsy (empty string) AND whose
      ``created_at`` is either absent, unparseable, or before
      ``run_window_start``.  These indicate Stage 1 producer drift from a
      prior cycle.  A WARNING is logged when this bucket is non-empty.
    * **stale_mismatched_run_id_ids** — ``id`` strings for records whose
      ``metadata.run_id`` is present and truthy but does not match
      ``current_run_id`` AND whose ``created_at`` is out of the run window.
      These are normal prior-cycle residue.
    * **rescued_ids** — subset of **current** containing markers rescued by
      the run-window guard; see :attr:`Stage2FlagPartition.rescued_ids` for
      full semantics.  This is the single source of truth for the rescued
      count (task-1381).

    Both stale buckets are excluded from what is rendered to the Stage 2 LLM;
    residue is reaped by the in-cycle Mem0 age-GC sweep
    (:func:`_sweep_stale_mem0_flag_for_stage2_markers`, task 2966) rather
    than an immediate per-cycle delete by the caller. (Prior to task 2966
    this docstring pointed at the recon_ledger's TTL/terminal-task GC pass,
    :func:`_gc_recon_markers` — that pass STRUCTURALLY cannot reach these
    markers: ``flag_for_stage2`` is written ONLY to Mem0 and is never
    upserted into the recon_ledger, so ``ReconLedgerStore.gc()``'s
    ``record_kind IN (...)`` clause never matches it.  See
    ``recon_self_model.py``'s ``flag_for_stage2`` ``MARKER_LIFECYCLE`` entry
    for the full declared-vs-actual explanation.)

    The *run_window_start* parameter is optional and defaults to ``None``
    (backward-compatible: window guard dormant, pure run_id partition applies).
    When supplied a tz-aware :class:`~datetime.datetime` is preferred; a naive
    value is defensively normalized to UTC by :func:`_marker_is_within_run_window`
    (task-1383 hardening) so the guard remains active regardless.

    .. note::
        The partition check requires ``metadata.run_id`` to be **truthy**
        before comparing — an empty-string ``run_id`` is treated as absent and
        placed in the stale partition even when ``current_run_id`` is also
        empty.  This prevents a falsy ``_current_run_id`` from silently
        classifying all empty-string markers as "current".

    .. warning::
        This function uses semantic search with a ``limit=100`` top-N cutoff.
        In a busy Mem0 collection, flags with low embedding similarity to the
        query can be pushed off the bottom and silently dropped.  When the
        result count equals the limit a WARNING is logged to surface this
        condition.  FIX D's persistence tracking (``_track_flag_persistence``)
        uses a deterministic ``count_memories_by_metadata`` call to avoid this
        problem for staleness detection.  The active-query path here still
        carries the top-N risk — see follow-up task for a proper
        ``scroll_by_metadata`` API on Mem0Backend.

    Returns ``([], [], [], [])`` on search failure (best-effort; logs WARNING).
    """
    try:
        results = await memory_service.search(
            query='stage 1 flag for stage 2',
            project_id=project_id,
            categories=['observations_and_summaries'],
            limit=100,
        )
    except Exception:
        logger.warning(
            'reconciliation._query_stage2_flags: Mem0 search failed; '
            'skipping active-query path this cycle',
            extra={'project_id': project_id},
        )
        return Stage2FlagPartition([], [], [], [])

    if len(results) == 100:
        logger.warning(
            'reconciliation._query_stage2_flags: search returned limit=100 '
            'results — some markers may be beyond the top-N cutoff and will '
            'not be swept or rendered this cycle.  Follow-up: migrate to '
            'scroll_by_metadata for GC correctness.',
            extra={'project_id': project_id},
        )

    current_flags: list[dict] = []
    stale_missing_run_id_ids: list[str] = []
    stale_mismatched_run_id_ids: list[str] = []
    rescued_ids: list[str] = []

    def _rescue(flag: dict, rid: str) -> None:
        """Route a rescued marker to current and record its id.

        This is the ONLY place that should append to both current_flags and
        rescued_ids so the two lists stay in sync.  Adding a new rescue path
        means calling _rescue — forgetting rescued_ids is a compile-visible
        error rather than a silent counter undercount.
        """
        current_flags.append(flag)
        rescued_ids.append(rid)

    run_id_str = str(current_run_id)
    for r in results:
        meta = dict(r.metadata or {})
        if not meta.get('flag_for_stage2'):
            continue
        flag_dict = {
            'id': r.id,
            'content': r.content,
            'metadata': meta,
            'task_id': str(meta.get('task_id', '')),
        }
        # Require run_id to be truthy before comparing; empty-string run_id is
        # treated as absent and placed in the missing partition unconditionally.
        if meta.get('run_id') and str(meta['run_id']) == run_id_str:
            current_flags.append(flag_dict)
        elif meta.get('run_id') in (None, ''):
            # None or empty-string only; other non-string types (0, False, [], {}) fall through to
            # the mismatched bucket where the type-violation warning at line ~683 fires.
            # Producer drift = missing-bucket warning; protocol violation = mismatched-bucket type warning.
            # Run-window guard: if the marker was written during this run's window, rescue it to
            # current so Stage 2 can process it this cycle (task-1369 same-cycle Stage-1 write fix).
            _created_at_val = getattr(r, 'created_at', None)
            if _marker_is_within_run_window(_created_at_val, run_window_start):
                logger.info(
                    'reconciliation._query_stage2_flags: same-cycle Stage-1 marker rescued '
                    'by run-window guard (missing run_id, created_at=%s); routing to current '
                    'for Stage 2 — indicates Stage 1 producer drift within the current cycle',
                    _created_at_val,
                    extra={'project_id': project_id, 'current_run_id': run_id_str, 'marker_id': r.id},
                )
                _rescue(flag_dict, r.id)
            else:
                if run_window_start is not None and (
                    not _created_at_val or not isinstance(_created_at_val, str)
                ):
                    logger.debug(
                        'reconciliation._query_stage2_flags: run_window_start set but '
                        'created_at is missing/non-string for marker %s; window guard '
                        'dormant for this marker, routing to stale bucket',
                        r.id,
                        extra={'project_id': project_id, 'current_run_id': run_id_str},
                    )
                stale_missing_run_id_ids.append(r.id)
        else:
            # Present but does not match current run_id — prior-cycle residue (or unexpected type).
            # Log a warning when run_id is not a string: producer contract requires a non-empty
            # string, so any non-string type is a detectable protocol violation.
            _rid_val = meta.get('run_id')
            if not isinstance(_rid_val, str):
                logger.warning(
                    'reconciliation._query_stage2_flags: non-string run_id type %s=%r — '
                    'routing to mismatched bucket; producer contract requires a non-empty string',
                    type(_rid_val).__name__,
                    _rid_val,
                    extra={'project_id': project_id, 'current_run_id': run_id_str},
                )
            # Run-window guard: same-cycle marker with a mis-stamped run_id — rescue to current.
            _created_at_val = getattr(r, 'created_at', None)
            if _marker_is_within_run_window(_created_at_val, run_window_start):
                logger.info(
                    'reconciliation._query_stage2_flags: same-cycle Stage-1 marker rescued '
                    'by run-window guard (mismatched run_id=%r, created_at=%s); routing to '
                    'current for Stage 2 — indicates Stage 1 run_id mis-stamp this cycle',
                    _rid_val,
                    _created_at_val,
                    extra={'project_id': project_id, 'current_run_id': run_id_str, 'marker_id': r.id},
                )
                _rescue(flag_dict, r.id)
            else:
                if run_window_start is not None and (
                    not _created_at_val or not isinstance(_created_at_val, str)
                ):
                    logger.debug(
                        'reconciliation._query_stage2_flags: run_window_start set but '
                        'created_at is missing/non-string for marker %s; window guard '
                        'dormant for this marker, routing to stale bucket',
                        r.id,
                        extra={'project_id': project_id, 'current_run_id': run_id_str},
                    )
                stale_mismatched_run_id_ids.append(r.id)
    if stale_missing_run_id_ids:
        logger.warning(
            'reconciliation._query_stage2_flags: %d Stage 2 marker(s) missing '
            'metadata.run_id — Stage 1 producer drift (markers will be swept and '
            'never reach the LLM); see prompts/stage1.py',
            len(stale_missing_run_id_ids),
            extra={
                'project_id': project_id,
                'missing_run_id_count': len(stale_missing_run_id_ids),
                'current_run_id': run_id_str,
            },
        )
    return Stage2FlagPartition(current_flags, stale_missing_run_id_ids, stale_mismatched_run_id_ids, rescued_ids)


def _query_recon_report_findings(
    recon_report_state: Any,
    run_id: str,
    *,
    categories: frozenset[str] = frozenset({'systemic_pattern'}),
) -> list[dict]:
    """Poll the recon_report channel for *run_id*, filtered to *categories*.

    Independent second channel (task-1966 scope item 2): unlike the Mem0
    flagged-items channel (``_query_stage2_flags`` / Stage 1's
    ``items_flagged``), findings read via ``recon_report_state.get_findings_for_run``
    do NOT pass through ``filter_suppressed`` — a ``stage1_flag_suppression``
    record can never hide a ``systemic_pattern`` finding from Stage 2 through
    this path.

    Reached via duck-typed method call (no import of ``ReconReportState`` —
    mirrors ``base.py``'s ``_active_rrs.get_assembled_report`` usage, avoiding
    a server←reconciliation import).

    Best-effort, mirrors :func:`_query_stage2_flags`: returns ``[]`` when
    *recon_report_state* is ``None`` (recon_report channel not configured for
    this stage instance — no call is made) or when
    ``get_findings_for_run`` raises (logs a WARNING; never propagates).
    """
    if recon_report_state is None:
        return []
    try:
        findings = recon_report_state.get_findings_for_run(run_id)
    except Exception:
        logger.warning(
            'reconciliation._query_recon_report_findings: get_findings_for_run '
            'failed; skipping recon_report channel this cycle',
            extra={'run_id': run_id},
        )
        return []
    return [f for f in findings if f.get('category') in categories]


def _compute_stale_flags(
    persistence_counts: dict[str, int],
    threshold: int = STAGE2_FLAG_PERSISTENCE_THRESHOLD,
) -> list[str]:
    """Return sorted list of flag_ids whose persistence count is >= *threshold*.

    Args:
        persistence_counts: Mapping of flag_id → cycle count (from
            ``_track_flag_persistence``).
        threshold: Minimum count to be considered stale.  Defaults to
            ``STAGE2_FLAG_PERSISTENCE_THRESHOLD`` (3).

    Returns:
        Sorted list of flag_id strings that meet or exceed the threshold.
    """
    return sorted(fid for fid, count in persistence_counts.items() if count >= threshold)


# Persistence and escalation marker source tags — written into Mem0 metadata so
# they can be retrieved deterministically via Qdrant payload filters (NOT via
# semantic search, which can silently rank matches off the bottom of top-N).
_STAGE2_PERSISTENCE_MARKER_SOURCE = 'stage2_persistence_marker'
_STAGE2_ESCALATION_MARKER_SOURCE = 'stage2_escalation_marker'

# Stage 2 per-cycle summary pool cap and related constants.
# The deterministic Mem0 mirror write (write_cycle_summary, task 2229) tags
# recon_pool='stage2_cycle_summary' — _STAGE2_CYCLE_SUMMARY_RECON_POOL is
# imported above from the shared leaf module recon_pool_map.py (task 2140),
# not redefined here.  After each write, write_cycle_summary's internal
# enforce_summary_pool_cap call trims the pool to at most this many members by
# deleting the OLDEST entries — deterministically via Qdrant scroll, NOT semantic search.
STAGE2_CYCLE_SUMMARY_POOL_CAP: int = 2
_STAGE2_CYCLE_SUMMARY_TRIM_SOURCE = 'stage2_cycle_summary_trim'

# Audit tag for _write_task_count_snapshot's deterministic write (task 2325)
# — makes the Mem0 task_count_snapshot write structural (a plain Python
# add_memory call at the end of run()) instead of depending on the Stage-2
# LLM remembering the memory-stored Snapshot Discipline norm. Mirrors
# write_cycle_summary's internal-service idiom (task 2229).
_TASK_COUNT_SNAPSHOT_WRITE_SOURCE = 'stage2_task_count_snapshot_write'

# Audit tag for _prune_task_count_snapshots's prune-before-write delete pass
# (task 2429) — makes the recurring task_count_snapshot near-duplicate
# accumulation self-correcting: every kind='task_count_snapshot' Mem0 record
# is deleted immediately before _write_task_count_snapshot's add_memory call,
# so at most one canonical snapshot survives per project per cycle.
_TASK_COUNT_SNAPSHOT_PRUNE_SOURCE = 'stage2_task_count_snapshot_prune'

# Age-based GC for the stage2_persistence_marker Channel-3 persistence-counter
# family (task 2095), mirroring the stage1_flag_marker age-GC above (task 1944).
# _track_flag_persistence writes one stage2_persistence_marker per surviving
# flag per cycle with NO existing sweep. Unlike stage1_flag_marker (persistent
# dedup-state meant to persist indefinitely), this is a bounded failure-case
# counter: once a flag stops recurring (deleted by FIX C, or simply resolved)
# its markers are orphaned. 14 days matches the task-1944 default for
# operator consistency; a still-live flag rewrites a fresh marker every
# cycle (its newest marker is always <1 day old), so age-GC only trims aged
# surplus and never blocks the STAGE2_FLAG_PERSISTENCE_THRESHOLD escalation.
STAGE2_PERSISTENCE_MARKER_MAX_AGE_DAYS: int = 14
_STAGE2_PERSISTENCE_MARKER_GC_SWEEP_SOURCE = 'stage2_persistence_marker_gc_sweep'

# Age-based GC for the legacy Mem0 stage1_flag_marker pool (task 2853).
# Task 2406 retired the Mem0 stage1_flag_marker mirror write — markers now
# persist only to the recon_ledger SQLite table (see _gc_recon_markers /
# ReconLedgerStore.gc above, which reaps ledger rows only). Task 2228 W5-κ
# deleted the prior in-cycle Mem0 sweeps for this source (_sweep_stale_flag_
# markers, _sweep_terminal_task_flag_markers) on the assumption that the
# ledger gc() pass fully replaced them; it does not reach Mem0, so the
# pre-2406 Mem0 pool was left with no in-cycle collector for any project.
# Since the write path is fully retired, every remaining Mem0
# stage1_flag_marker record is dead weight; 14 days reuses the
# STAGE2_PERSISTENCE_MARKER_MAX_AGE_DAYS / task-1944 convention as a
# conservative, consistent aging cutoff rather than deleting immediately.
_STAGE1_FLAG_MARKER_MEM0_SOURCE = 'stage1_flag_marker'
STAGE1_FLAG_MARKER_MEM0_MAX_AGE_DAYS: int = 14
_STAGE1_FLAG_MARKER_GC_SWEEP_SOURCE = 'stage1_flag_marker_gc_sweep'

# Age-based GC for the Mem0-only flag_for_stage2 Stage-1 -> Stage-2 relay pool
# (task 2966). flag_for_stage2 markers are written ONLY to Mem0 by the
# Stage-1 flag_dedup/LLM add_memory path (metadata.flag_for_stage2=true); no
# code path upserts a flag_for_stage2 row into recon_ledger, so
# ReconLedgerStore.gc()'s record_kind IN (...) clause never matches it — the
# same declared-vs-actual gap already documented above for
# stage2_persistence_marker (task 2228 W5-κ). Unlike stage1_flag_marker /
# stage2_persistence_marker, these markers carry NO ``source`` metadata field
# at all (live verification, task 2966), so the pool cannot be identified by
# a {'source': ...} filter — it is enumerated by the boolean payload key
# {'flag_for_stage2': True} instead (Qdrant payload filters are
# type-sensitive; the stored value is boolean True, not the string 'true').
# A marker Stage 2 hasn't consumed in 14+ days can never be "current" per
# _query_stage2_flags' run_id/run-window semantics (run_ids are per-cycle;
# Stage 2 runs many times/day) — so it is definitionally unconsumed dead
# signal past that point; 14 days reuses the task-1944
# STAGE1_FLAG_MARKER_MEM0_MAX_AGE_DAYS convention for operator consistency.
_FLAG_FOR_STAGE2_MEM0_MAX_AGE_DAYS: int = 14
_FLAG_FOR_STAGE2_GC_SWEEP_SOURCE = 'flag_for_stage2_gc_sweep'
_FLAG_FOR_STAGE2_ENUM_FILTERS: dict = {'flag_for_stage2': True}

# Boolean/string type-drift probe (task 2966 amendment, reviewer finding).
# _query_stage2_flags' consumer-side check (``meta.get('flag_for_stage2')``,
# ~line 489) is a plain truthy check — it accepts BOTH the boolean True and
# the string 'true'. The GC filter above is type-EXACT (boolean True only),
# because Qdrant payload filters are type-sensitive. Nothing at the
# add_memory write boundary normalizes this key's type (it is an LLM
# tool-call argument, and only ``_normalize_task_id_metadata`` runs
# server-side) — so a producer writing the string 'true' instead of the
# boolean is not structurally prevented. Such a marker would be rendered
# and consumed normally by _query_stage2_flags but INVISIBLE to the GC
# filter above, and would accumulate forever, silently. This filter is used
# by _warn_on_flag_for_stage2_type_drift to probe for that condition and
# log loudly rather than let it grow unbounded and silent.
_FLAG_FOR_STAGE2_STRING_VARIANT_FILTERS: dict = {'flag_for_stage2': 'true'}

# --- Entity-standing-decision growth-freshness sweep (task 2899 ζ) -----------
# PRD plans/stage1-entity-standing-decision-prd.md §Staleness: a standing
# decision's decision-time edge-count snapshot (``edge_count_at_decision``) is
# corroborated against the LIVE graph at each Stage-2 tail. An ACTIVE row is
# flipped to expired/growth when the entity has grown materially since the
# decision:
#     live > edge_count_at_decision * FACTOR
#     OR live >= edge_count_at_decision + ABS_DELTA
# (whichever trips first). The two-armed rule keeps a small entity from expiring
# on a single-edge wobble (the ABS_DELTA floor) while still catching proportional
# growth on a large entity (the FACTOR). These are the PRD contract defaults; each
# is overridable PER RECORD via optional payload keys ``growth_factor`` /
# ``growth_abs_delta`` (α's row-schema override path) — a local sweep constant
# mirrors the file's existing convention (STAGE1_FLAG_MARKER_MEM0_MAX_AGE_DAYS).
STANDING_DECISION_GROWTH_FACTOR: float = 1.25
STANDING_DECISION_GROWTH_ABS_DELTA: int = 15

# Per-cycle Stage-2 stats recorded by the growth sweep (explicit-zero, like the
# marker-sweep stats): whether >=1 active row could not be verified this cycle
# (fail-safe: the row is left ACTIVE) and how many rows were flipped to
# expired/growth.
ENTITY_STANDING_DECISION_GROWTH_SWEEP_FAILED_STAT_KEY = (
    'entity_standing_decision_growth_sweep_failed'
)
ENTITY_STANDING_DECISION_GROWTH_EXPIRED_STAT_KEY = (
    'entity_standing_decision_growth_expired'
)

# Consecutive-failure streak (in full+completed cycles) at which the growth
# sweep files a recon escalation (INV-4). Mirrors the task-count-snapshot
# miss-streak threshold; recomputed from the journal each cycle (no stored
# counter), so any successful sweep resets it.
GROWTH_SWEEP_FAILURE_STREAK_THRESHOLD: int = 3

# Escalation category for the growth-sweep failure streak (INV-4). Free-form str
# on ``Escalation.category``; kept distinct from the Stage-1/Stage-2 recon
# categories (reconciliation_stale_human_operator / reconciliation_stale_flag /
# recon_stale_task_count_snapshot) so it can be filtered independently.
ENTITY_STANDING_DECISION_GROWTH_SWEEP_ESCALATION_CATEGORY = (
    'reconciliation_growth_sweep_failure'
)


async def _resolve_terminal_task_ids(
    taskmaster,
    scope: ProjectScope,
    run_id: str,
) -> list[str]:
    """Resolve the terminal-status task ids for *scope* via ONE bulk status read.

    Feeds :meth:`fused_memory.reconciliation.recon_ledger.ReconLedgerStore.gc`'s
    ``terminal_task_ids`` argument (task 2228 W5-κ). Replaces the per-marker
    ``taskmaster.get_task`` loop previously used by the now-deleted
    ``_sweep_terminal_task_flag_markers``: a single bulk
    ``taskmaster.get_statuses(project_root)`` call is strictly cheaper than N
    individual lookups, and the ledger's ``task_id IN (...)`` clause does the
    membership match itself.

    **Fail-safe direction**: degrades to ``[]`` when ``taskmaster`` is falsy,
    when ``get_statuses`` raises, or when its result is anything other than
    the documented ``dict[str, str]`` shape (e.g. an unconfigured test double)
    — never propagates. An empty result feeds ``gc()``'s expiry-only branch,
    which keeps every task-referenced marker, preserving the old sweeps'
    fail-safe KEEP direction (uncertain => keep, never delete on
    partial/failed information). This runs unconditionally on every
    ``TaskKnowledgeSync.run()`` call, so it must never be the reason a
    reconciliation cycle crashes.

    Args:
        taskmaster: Object with an async ``get_statuses(project_root) ->
            dict[str, str]`` method. Falsy => no-op (returns ``[]``).
        scope: ``ProjectScope`` supplying ``project_root`` for the bulk
            status read.
        run_id: Current reconciliation run identifier, logged on failure.

    Returns:
        Sorted list of task_id strings whose status is in
        ``TERMINAL_STATUSES`` (``{'done', 'cancelled'}``); ``[]`` on any
        failure or falsy taskmaster.
    """
    if not taskmaster:
        return []

    try:
        statuses = await taskmaster.get_statuses(scope.project_root)
        return sorted(str(tid) for tid, status in statuses.items() if status in TERMINAL_STATUSES)
    except Exception:
        logger.warning(
            'reconciliation._resolve_terminal_task_ids: get_statuses failed for project_id=%s',
            scope.project_id,
            extra={'project_id': scope.project_id, 'run_id': run_id},
        )
        return []


async def _gc_recon_markers(
    memory_service,
    taskmaster,
    scope: ProjectScope,
    run_id: str,
    *,
    now: datetime | None = None,
) -> int:
    """Garbage-collect ``recon_ledger`` marker rows for *scope* in ONE DELETE pass.

    Replaces three of the original four Mem0 sweeps — the now-deleted
    ``_sweep_stale_fixc_markers``, ``_sweep_stale_flag_markers``, and
    ``_sweep_terminal_task_flag_markers`` — with a single
    :meth:`~fused_memory.reconciliation.recon_ledger.ReconLedgerStore.gc` call
    (task 2228 W5-κ): the ledger's ``expires_at < now`` clause replaces the
    age-based sweeps and the fixc residue delete, and its
    ``record_kind IN MARKER_KINDS AND task_id IN terminal_task_ids`` clause
    replaces the terminal-task sweep. The fourth original sweep,
    :func:`_sweep_stale_persistence_markers`, was restored as a separate
    Mem0-resident pass — its writer (:func:`_track_flag_persistence`) was
    never migrated to the ledger, so this function's ``gc()`` call cannot
    collect its markers — and is NOT replaced here; see the GC block in
    ``TaskKnowledgeSync.run()`` for both passes running side by side.

    Degrades to ``0`` — never raises — when ``memory_service`` has no ledger
    wired (``recon_ledger`` is ``None``, e.g. mid write-both/read-new
    cutover) or when ``ledger.gc`` itself fails. ``terminal_task_ids`` is
    resolved via :func:`_resolve_terminal_task_ids`, which is itself
    fail-safe to ``[]`` — an empty list makes ``gc()`` run its expiry-only
    branch, preserving the old sweeps' fail-safe KEEP direction.

    **Bounding (task 2228 W5-κ amendment)**: before being forwarded to
    ``gc()``, a non-empty ``terminal_task_ids`` is intersected with
    :meth:`~fused_memory.reconciliation.recon_ledger.ReconLedgerStore.marker_task_ids`
    — the set of task ids that actually have a marker row in the ledger right
    now. A terminal id with no marker row can never match ``gc()``'s
    ``task_id IN (...)`` clause, so dropping it keeps the DELETE's
    bind-parameter count tied to ledger occupancy instead of the project's
    full (monotonically growing) terminal-task history, which could
    otherwise approach SQLite's ``SQLITE_MAX_VARIABLE_NUMBER`` on a large,
    mature project. A ``marker_task_ids`` failure is itself fail-safe: it
    degrades ``terminal_task_ids`` to ``[]`` (logged WARNING) rather than
    forwarding the unbounded list or aborting the whole GC pass, so ``gc()``
    still performs its expiry-only DELETE that cycle.

    **Comma-joined markers are TTL-only (task 2228 W5-κ, review finding
    robustness_regression)**: ``terminal_task_ids`` holds only individual ids
    (as returned by ``taskmaster.get_statuses``), and ``ledger.gc()``'s
    terminal-referenced match is an exact string comparison — so a marker
    whose stored ``task_id`` is a comma-joined multi-task list (e.g.
    ``'12,15'``) never matches even when every cited task is terminal. This is
    an intentional, fail-safe simplification: such markers are reaped via the
    ``expires_at`` TTL path instead of promptly on terminal transition. See
    :meth:`~fused_memory.reconciliation.recon_ledger.ReconLedgerStore.gc` for
    the full rationale.

    Args:
        memory_service: Service that may expose a ``recon_ledger``
            (:class:`~fused_memory.reconciliation.recon_ledger.ReconLedgerStore`)
            attribute. Missing/``None`` => no-op (returns ``0``).
        taskmaster: Forwarded to :func:`_resolve_terminal_task_ids`. Falsy => no-op there.
        scope: ``ProjectScope`` supplying ``project_id`` (ledger scope) and
            ``project_root`` (forwarded to ``taskmaster.get_statuses``).
        run_id: Current reconciliation run identifier, logged on failure.
        now: Reference "current time" for the ``expires_at`` comparison.
            Defaults to ``datetime.now(UTC)``; tests inject a fixed value.
            Normalized via :func:`_assume_utc` and rendered with
            ``.isoformat()`` to match the writer format used by
            ``flag_dedup._persist_flag_marker``, so the ledger's
            lexicographic TEXT comparison against stored ``expires_at``
            values is correct.

    Returns:
        Number of rows deleted by the ``gc()`` pass (``0`` on any failure or
        when no ledger is wired).
    """
    ledger = getattr(memory_service, 'recon_ledger', None)
    if ledger is None:
        return 0

    now_iso = _assume_utc(now or datetime.now(UTC)).isoformat()
    terminal_task_ids = await _resolve_terminal_task_ids(taskmaster, scope, run_id)

    if terminal_task_ids:
        try:
            resident_ids = await ledger.marker_task_ids(scope.project_id)
            terminal_task_ids = [tid for tid in terminal_task_ids if tid in resident_ids]
        except Exception:
            logger.warning(
                'reconciliation._gc_recon_markers: ledger.marker_task_ids failed for '
                'project_id=%s — bounding terminal_task_ids to [] (expiry-only gc this cycle)',
                scope.project_id,
                extra={'project_id': scope.project_id, 'run_id': run_id},
            )
            terminal_task_ids = []

    try:
        return await ledger.gc(scope.project_id, now_iso, terminal_task_ids)
    except Exception:
        logger.warning(
            'reconciliation._gc_recon_markers: ledger.gc failed for project_id=%s',
            scope.project_id,
            extra={'project_id': scope.project_id, 'run_id': run_id},
        )
        return 0


async def _sweep_stale_mem0_pool(
    memory_service,
    project_id: str,
    run_id: str,
    *,
    source: str,
    gc_sweep_source: str,
    max_age_days: int,
    log_name: str,
    now: datetime | None = None,
    scroll_limit: int = 1000,
    count_short_circuit: bool = False,
    enum_filters: dict | None = None,
) -> int:
    """Shared age-GC skeleton for a single-source Mem0 marker pool.

    Factored out of :func:`_sweep_stale_persistence_markers` and
    :func:`_sweep_stale_mem0_flag_markers` (task 2853 amendment, reviewer
    finding code-duplication): both were near-verbatim ~70-line copies of
    the same enumerate -> age-filter -> gather_collect-delete ->
    count-successes skeleton, differing only in the source filter, the
    delete ``_source`` audit tag, and the max-age constant. A future fix to
    the KEEP-on-uncertainty logic or the scroll-cap accounting now lands
    once for every caller instead of needing to be replicated per source.

    Enumerates deterministically via ``memory_service.get_memories_by_metadata``
    (Qdrant payload-filter scroll) — NEVER semantic search, which silently
    drops low-similarity rows and is unsuitable for exhaustive GC. Members
    with a missing or unparseable ``created_at`` are KEPT (never deleted) —
    a fail-safe KEEP-on-uncertainty posture shared with every other marker
    sweep in this module.

    **Protected-mirror invariant (task 3041): this skeleton NEVER deletes a
    ``kind='cycle_summary'`` / ``record_type='ledger_stamp'`` record**, no
    matter which pool filter selected it. Every enumerated member is tested
    against
    :func:`~fused_memory.reconciliation.mem0_tombstone.is_protected_mirror_record`
    BEFORE the age check; a match is skipped with a WARNING naming the
    memory_id, its kind/record_type and *log_name*, and is excluded from the
    returned count. An over-broad payload filter therefore degrades to a LOUD
    skip rather than collateral mirror loss.

    The guard lives HERE rather than in each caller's payload filter because
    filter-tightening cannot guarantee precision:
    :data:`_FLAG_FOR_STAGE2_ENUM_FILTERS` is ``{'flag_for_stage2': True}``
    with no ``kind``/``source``/``record_type`` discriminator at all, and
    ``flag_for_stage2`` is an LLM-supplied metadata key that nothing at the
    ``add_memory`` boundary stops a cycle_summary write from also carrying.
    Enforcing at this single choke point also means every future caller
    inherits the guard for free — the same reuse the task-2853 reviewer
    created this factoring for.

    Deletes are issued best-effort in parallel via ``gather_collect``:
    individual failures log WARNING and are excluded from the returned
    count.

    Every CONFIRMED-successful delete leaves a queryable tombstone via
    :func:`~fused_memory.reconciliation.mem0_tombstone.record_mem0_deletion_tombstones`
    (task 3041), naming *gc_sweep_source* as the deleter and *run_id* as the
    deleting run — so an auditor holding only the memory uuid can find out who
    reaped it and why. Written from the success branch ONLY: a tombstone must
    never claim a record that is still alive, and written for the whole sweep
    in ONE ledger transaction rather than one commit (one fsync) per victim.

    Args:
        memory_service: Service with ``get_memories_by_metadata`` and
            ``delete_memory`` (and, when ``count_short_circuit`` is set,
            ``count_memories_by_metadata``).
        project_id: Project scope for enumeration and delete calls.
        run_id: Current reconciliation run identifier used as
            ``causation_id`` in the audit journal.
        source: The ``metadata.source`` value identifying this pool. Also
            used as the default enumeration/count filter (``{'source':
            source}``) when ``enum_filters`` is not given, and always used
            as the human-readable log label (e.g. in the scroll-cap
            WARNING) regardless of which filter is actually applied.
        gc_sweep_source: The delete ``_source`` audit tag for this pool.
        max_age_days: Staleness cutoff in days.
        log_name: Public caller name, interpolated into log messages so
            WARNING lines stay attributable to the right sweep.
        now: Reference "current time" for the cutoff calculation. Defaults
            to ``datetime.now(UTC)``; tests inject a fixed value.
        scroll_limit: Max records to enumerate in one scroll (default 1000).
        count_short_circuit: When ``True``, probe
            ``memory_service.count_memories_by_metadata`` first and return
            ``0`` immediately on a confirmed exact-zero count — skipping the
            larger scroll entirely. Fails OPEN: an exception, or any result
            other than exactly ``0``, falls straight through to the normal
            scroll path unchanged — so this can only ever skip a scroll that
            would itself have found nothing. Intended for a pool whose write
            path is fully retired (task 2853 review, efficiency finding),
            where the scroll would otherwise run forever against an
            already-empty pool; deliberately NOT used for a still-active
            pool, where the count is almost never zero and the extra
            round-trip would be pure overhead.
        enum_filters: Overrides the enumeration/count filter dict when the
            pool cannot be identified by a ``{'source': source}`` payload
            filter (e.g. a marker with no ``source`` field at all). Defaults
            to ``None``, which preserves the ``{'source': source}`` filter
            used by every caller before task 2966. When given, it is applied
            verbatim to BOTH the ``count_short_circuit`` probe and the
            enumeration scroll — ``source`` itself still supplies the
            human-readable log label regardless.

    Returns:
        Number of memories successfully deleted (0 if nothing is stale, on
        enumeration failure, or on a confirmed-empty count short-circuit).
    """
    filters = enum_filters if enum_filters is not None else {'source': source}

    if count_short_circuit:
        try:
            count = await memory_service.count_memories_by_metadata(
                project_id=project_id,
                filters=filters,
            )
        except Exception:
            count = None
        if count == 0:
            return 0

    try:
        members = await memory_service.get_memories_by_metadata(
            project_id=project_id,
            filters=filters,
            limit=scroll_limit,
        )
    except Exception:
        logger.warning(
            'reconciliation.%s: get_memories_by_metadata failed for project_id=%s; skipping sweep',
            log_name, project_id,
            extra={'project_id': project_id, 'run_id': run_id},
        )
        return 0

    if len(members) >= scroll_limit:
        logger.warning(
            'reconciliation.%s: enumerated %d of scroll_limit=%d %s records — scroll cap '
            'reached; older stale markers may remain uncollected this cycle; re-run with a '
            'higher scroll_limit.',
            log_name, len(members), scroll_limit, source,
            extra={'project_id': project_id, 'run_id': run_id},
        )

    if not members:
        return 0

    cutoff = _assume_utc(now or datetime.now(UTC)) - timedelta(days=max_age_days)

    # Full member dicts, not bare ids: the tombstone write below needs the
    # victim's metadata/created_at at classification time (task 3041). Kept
    # as one list so the zip(..., strict=True) delete/result pairing below is
    # structurally unchanged.
    stale_members: list[dict] = []
    for member in members:
        mid = member.get('id')
        if not mid:
            continue

        # Protected-mirror exclusion (task 3041), checked BEFORE the age test
        # so an over-broad payload filter degrades to a loud skip rather than
        # collateral mirror loss. See this function's docstring for why the
        # guard lives here instead of in each caller's filter.
        member_metadata = member.get('metadata')
        if is_protected_mirror_record(member_metadata):
            metadata = member_metadata if isinstance(member_metadata, dict) else {}
            logger.warning(
                'reconciliation.%s: SKIPPING protected cycle_summary mirror '
                'memory_id=%s (kind=%s record_type=%s) — this pool filter matched a '
                'record it must never delete; the enumeration filter is over-broad '
                'for this pool and should be tightened (task 3041).',
                log_name, mid, metadata.get('kind'), metadata.get('record_type'),
                extra={
                    'project_id': project_id,
                    'memory_id': mid,
                    'run_id': run_id,
                    'log_name': log_name,
                },
            )
            continue

        raw = member.get('created_at')
        if raw is None:
            continue
        try:
            created_at = _assume_utc(datetime.fromisoformat(raw))
        except (ValueError, TypeError):
            continue

        if created_at < cutoff:
            stale_members.append(member)

    if not stale_members:
        return 0

    # Two-tier check via gather_collect (fused_memory.utils.async_utils).
    # Pass 1 (inside gather_collect): re-raises structured-cancellation
    # signals — this preserves the structured-cancellation contract and
    # prevents this delete sweep from silently converting a shutdown
    # signal into an under-counted deletion tally.
    # Pass 2 (below): per-item degrade-to-warning on ordinary Exceptions.
    results = await gather_collect(
        memory_service.delete_memory(
            memory_id=member['id'],
            store='mem0',
            project_id=project_id,
            causation_id=run_id,
            _source=gc_sweep_source,
        )
        for member in stale_members
    )

    success_count = 0
    tombstone_victims = []
    for member, result in zip(stale_members, results, strict=True):
        mid = member['id']
        if isinstance(result, Exception):
            logger.warning(
                'reconciliation.%s: delete failed for memory_id=%s; not counted',
                log_name, mid,
                extra={'project_id': project_id, 'memory_id': mid, 'run_id': run_id},
            )
        else:
            success_count += 1
            # Success branch ONLY (task 3041): a tombstone must never claim a
            # record that is still alive, so the failed-delete branch above is
            # deliberately left untouched.
            tombstone_victims.append(member)

    if tombstone_victims:
        # ONE ledger transaction for the whole sweep, not one per victim: each
        # upsert is its own commit — hence its own fsync, serialized on the
        # single aiosqlite worker thread the rest of the cycle shares — so a
        # per-victim loop made a backlog sweep cost N sequential fsyncs on the
        # cycle's critical path (reviewer finding efficiency, task 3041
        # amendment pass).
        #
        # record_mem0_deletion_tombstones is internally fail-safe (returns 0,
        # never raises); this try/except is a second belt so even a helper that
        # is patched/broken cannot raise out of, or alter the count of, this
        # sweep — while still saying so out loud.
        try:
            await record_mem0_deletion_tombstones(
                memory_service,
                project_id,
                tombstone_victims,
                deleter=gc_sweep_source,
                deleting_run_id=run_id,
            )
        except Exception:
            logger.warning(
                'reconciliation.%s: tombstone batch raised for %d deleted record(s); '
                'the deletes themselves succeeded and are counted',
                log_name, len(tombstone_victims),
                exc_info=True,
                extra={'project_id': project_id, 'run_id': run_id},
            )

    return success_count


async def _sweep_stale_persistence_markers(
    memory_service,
    project_id: str,
    run_id: str,
    *,
    max_age_days: int = STAGE2_PERSISTENCE_MARKER_MAX_AGE_DAYS,
    now: datetime | None = None,
    scroll_limit: int = 1000,
) -> int:
    """Age-GC orphaned ``stage2_persistence_marker`` Mem0 records (task 2095).

    ``stage2_persistence_marker`` records are written once per surviving flag
    per cycle by :func:`_track_flag_persistence` with no existing collector.
    A flag that stops recurring (deleted by FIX C, or simply resolved) leaves
    its markers orphaned; this helper closes that gap by enumerating the
    whole pool and deleting members whose ``created_at`` is strictly older
    than ``now - max_age_days``.

    This is an AGE-ONLY sweep (task 1944 precedent, since retired for
    ``stage1_flag_marker`` by the recon_ledger ``gc()`` pass — task 2228
    W5-κ): it implements no cross-cycle ``fp:`` content-fingerprint
    predicate. ``stage2_persistence_marker`` metadata is ``{source, flag_id,
    run_id}`` with no ``task_id``/``fp:`` key, so that predicate has no
    applicable input here.

    Delegates to :func:`_sweep_stale_mem0_pool` (task 2853 amendment) for the
    shared enumerate -> age-filter -> gather_collect-delete -> count
    skeleton — see that function's docstring for the fail-safe posture and
    for its protected-mirror invariant (task 3041): a ``kind='cycle_summary'``
    / ``record_type='ledger_stamp'`` record is never deleted by this sweep,
    whatever this pool's filter matches.
    ``count_short_circuit`` is deliberately left off here: unlike
    ``stage1_flag_marker`` (below), this pool's writer
    (:func:`_track_flag_persistence`) is still active, so a zero count would
    be a rare transient rather than the steady state, and probing for it on
    every cycle would be pure overhead.

    Args:
        memory_service: Service with ``get_memories_by_metadata`` and
            ``delete_memory``.
        project_id: Project scope for enumeration and delete calls.
        run_id: Current reconciliation run identifier used as ``causation_id``
            in the audit journal.
        max_age_days: Staleness cutoff in days (default
            ``STAGE2_PERSISTENCE_MARKER_MAX_AGE_DAYS`` == 14).
        now: Reference "current time" for the cutoff calculation. Defaults to
            ``datetime.now(UTC)``; tests inject a fixed value.
        scroll_limit: Max records to enumerate in one scroll (default 1000).

    Returns:
        Number of memories successfully deleted (0 if nothing is stale, or
        on enumeration failure).
    """
    return await _sweep_stale_mem0_pool(
        memory_service,
        project_id,
        run_id,
        source=_STAGE2_PERSISTENCE_MARKER_SOURCE,
        gc_sweep_source=_STAGE2_PERSISTENCE_MARKER_GC_SWEEP_SOURCE,
        max_age_days=max_age_days,
        log_name='_sweep_stale_persistence_markers',
        now=now,
        scroll_limit=scroll_limit,
    )


async def _sweep_stale_mem0_flag_markers(
    memory_service,
    project_id: str,
    run_id: str,
    *,
    max_age_days: int = STAGE1_FLAG_MARKER_MEM0_MAX_AGE_DAYS,
    now: datetime | None = None,
    scroll_limit: int = 1000,
) -> int:
    """Age-GC the legacy ``stage1_flag_marker`` Mem0 pool (task 2853).

    Task 2406 retired the Mem0 ``stage1_flag_marker`` mirror write — markers
    now persist only to the ``recon_ledger`` SQLite table, reaped by
    :func:`_gc_recon_markers` / :meth:`~fused_memory.reconciliation.recon_ledger.ReconLedgerStore.gc`.
    Task 2228 W5-κ deleted the prior in-cycle Mem0 sweeps for this source
    (the old ``_sweep_stale_flag_markers`` and
    ``_sweep_terminal_task_flag_markers``) on the assumption that the ledger
    ``gc()`` pass fully replaced them; ``gc()`` only reaps ledger rows, so
    the pre-2406 Mem0 pool was left with no in-cycle collector for any
    project — including projects other than ``dark_factory``, which is the
    only project the operational ``sweep_orphan_flag_markers.py`` systemd
    timer targets by default. This helper closes that gap per-project,
    in-cycle, so every project self-heals its own legacy pool.

    Delegates to :func:`_sweep_stale_mem0_pool` (task 2853 amendment) for the
    shared enumerate -> age-filter -> gather_collect-delete -> count
    skeleton, with a distinct source filter, delete ``_source`` tag, and
    max-age constant from :func:`_sweep_stale_persistence_markers` — see
    that function's docstring for the fail-safe posture and for its
    protected-mirror invariant (task 3041): a ``kind='cycle_summary'`` /
    ``record_type='ledger_stamp'`` record is never deleted by this sweep,
    whatever this pool's filter matches.

    Passes ``count_short_circuit=True`` (task 2853 review, efficiency
    finding): this pool's write path is fully retired, so once the legacy
    records age past the cutoff and are drained, every subsequent cycle
    would otherwise re-scroll an already-empty pool forever. A cheap
    ``count_memories_by_metadata`` probe short-circuits that steady state;
    see :func:`_sweep_stale_mem0_pool`'s docstring for why this is safe to
    fail open.

    Args:
        memory_service: Service with ``get_memories_by_metadata``,
            ``delete_memory``, and ``count_memories_by_metadata``.
        project_id: Project scope for enumeration and delete calls.
        run_id: Current reconciliation run identifier used as ``causation_id``
            in the audit journal.
        max_age_days: Staleness cutoff in days (default
            ``STAGE1_FLAG_MARKER_MEM0_MAX_AGE_DAYS`` == 14).
        now: Reference "current time" for the cutoff calculation. Defaults to
            ``datetime.now(UTC)``; tests inject a fixed value.
        scroll_limit: Max records to enumerate in one scroll (default 1000).

    Returns:
        Number of memories successfully deleted (0 if nothing is stale, on
        enumeration failure, or on a confirmed-empty count short-circuit).
    """
    return await _sweep_stale_mem0_pool(
        memory_service,
        project_id,
        run_id,
        source=_STAGE1_FLAG_MARKER_MEM0_SOURCE,
        gc_sweep_source=_STAGE1_FLAG_MARKER_GC_SWEEP_SOURCE,
        max_age_days=max_age_days,
        log_name='_sweep_stale_mem0_flag_markers',
        now=now,
        scroll_limit=scroll_limit,
        count_short_circuit=True,
    )


async def _warn_on_flag_for_stage2_type_drift(
    memory_service,
    project_id: str,
    run_id: str,
) -> None:
    """Best-effort probe for the boolean/string type-drift gap (task 2966 review).

    :func:`_sweep_stale_mem0_flag_for_stage2_markers`'s GC filter
    (``_FLAG_FOR_STAGE2_ENUM_FILTERS`` == ``{'flag_for_stage2': True}``) is a
    type-exact Qdrant payload match, but ``_query_stage2_flags``' consumer-side
    check (``meta.get('flag_for_stage2')``) is a plain truthy check that also
    accepts the string ``'true'``. Nothing at the ``add_memory`` write
    boundary normalizes this key's type, so a producer writing the string
    instead of the boolean would create a marker that Stage 2 renders and
    consumes normally but that the GC filter can never see — an unbounded,
    silent leak of exactly the shape this sweep exists to prevent.

    Issues one supplementary ``count_memories_by_metadata`` call for the
    string-typed variant (``_FLAG_FOR_STAGE2_STRING_VARIANT_FILTERS``) and
    logs a WARNING when it is a positive integer, so drift surfaces loudly
    instead of accumulating quietly (reviewer's stated minimum floor: "at
    minimum add a log/metric when ... a type drift surfaces loudly rather
    than growing unbounded and silent").

    Purely diagnostic and fail-safe: any exception, or any non-``int``/
    non-positive result (e.g. a falsy probe or an unexpected return shape),
    is treated as "nothing to report" — this must never raise and must never
    affect the caller's sweep count.
    """
    try:
        string_variant_count = await memory_service.count_memories_by_metadata(
            project_id=project_id,
            filters=_FLAG_FOR_STAGE2_STRING_VARIANT_FILTERS,
        )
    except Exception:
        logger.warning(
            'reconciliation._sweep_stale_mem0_flag_for_stage2_markers: '
            'flag_for_stage2 string-variant type-drift probe raised; skipping '
            'this diagnostic (fail-safe, does not affect the sweep count).',
            exc_info=True,
            extra={'project_id': project_id, 'run_id': run_id},
        )
        return
    if not isinstance(string_variant_count, int) or string_variant_count <= 0:
        return
    logger.warning(
        'reconciliation._sweep_stale_mem0_flag_for_stage2_markers: found %d '
        "flag_for_stage2 marker(s) stored as the string 'true' rather than "
        'boolean True — _query_stage2_flags renders/consumes these normally '
        "(truthy check) but this sweep's type-exact GC filter "
        '(%r) can never match them, so they will accumulate forever; '
        'investigate the producer (task 2966 review finding).',
        string_variant_count,
        _FLAG_FOR_STAGE2_ENUM_FILTERS,
        extra={'project_id': project_id, 'run_id': run_id},
    )


async def _sweep_stale_mem0_flag_for_stage2_markers(
    memory_service,
    project_id: str,
    run_id: str,
    *,
    max_age_days: int = _FLAG_FOR_STAGE2_MEM0_MAX_AGE_DAYS,
    now: datetime | None = None,
    scroll_limit: int = 1000,
) -> int:
    """Age-GC the Mem0-only ``flag_for_stage2`` relay pool (task 2966).

    ``flag_for_stage2`` markers are the Stage-1 -> Stage-2 relay channel:
    written ONLY to Mem0 by the Stage-1 flag_dedup/LLM ``add_memory`` path
    (``metadata.flag_for_stage2=true``), with no code path that ever upserts
    a ``flag_for_stage2`` row into the ``recon_ledger`` SQLite table. This
    means :meth:`~fused_memory.reconciliation.recon_ledger.ReconLedgerStore.gc`
    STRUCTURALLY cannot reap it — its ``record_kind IN (...)`` clause never
    matches — even though ``recon_self_model.MARKER_LIFECYCLE`` classifies
    ``flag_for_stage2`` as ``deleter=DELETER_GC`` (see that module for the
    declared-vs-actual explanation, mirroring the identical
    ``stage2_persistence_marker`` gap from task 2228 W5-κ). Absent this
    sweep, the pool accumulates forever. This is the live in-cycle collector.

    Delegates to :func:`_sweep_stale_mem0_pool` (task 2853 amendment) for the
    shared enumerate -> age-filter -> gather_collect-delete -> count
    skeleton, passing ``enum_filters=_FLAG_FOR_STAGE2_ENUM_FILTERS`` — the
    DISTINCT boolean payload filter ``{'flag_for_stage2': True}`` — because
    these markers carry no ``source`` metadata field to filter on (unlike
    ``stage1_flag_marker`` / ``stage2_persistence_marker`` above), and
    because the stored value is a boolean ``True``, not the string ``'true'``
    (Qdrant payload filters are type-sensitive; live verification against
    dark_factory Mem0 confirmed this shape). See
    :func:`_sweep_stale_mem0_pool`'s docstring for the fail-safe posture.

    That boolean-only filter has NO ``kind``/``source``/``record_type``
    discriminator, so on its own it matches any record an LLM writer happened
    to stamp ``flag_for_stage2=True`` on — including a cycle_summary mirror.
    This sweep is therefore the concrete motivating case for the skeleton's
    protected-mirror invariant (task 3041), which makes that over-breadth
    degrade to a loud skip instead of collateral mirror loss.

    Passes ``count_short_circuit=True``: unlike ``stage2_persistence_marker``
    (written nearly every cycle that has surviving flags), Stage-1 writes a
    ``flag_for_stage2`` marker only intermittently — whenever Stage 1 flags
    an item for Stage 2 — so an empty pool is a plausible steady state,
    especially on the many non-dark_factory projects this per-project
    every-cycle sweep also runs on. A cheap
    ``count_memories_by_metadata`` probe short-circuits that steady state;
    it fails OPEN (see :func:`_sweep_stale_mem0_pool`'s docstring), so this
    can only ever skip a scroll that would itself have found nothing.

    After the primary sweep, also runs
    :func:`_warn_on_flag_for_stage2_type_drift` (task 2966 amendment,
    reviewer finding) — a supplementary best-effort probe for markers stored
    with the string ``'true'`` instead of boolean ``True``, which the
    type-exact GC filter above can never see even though such markers are
    consumed normally by ``_query_stage2_flags``'s truthy check. Purely
    diagnostic: never affects this function's return value.

    Args:
        memory_service: Service with ``get_memories_by_metadata``,
            ``delete_memory``, and ``count_memories_by_metadata``.
        project_id: Project scope for enumeration and delete calls.
        run_id: Current reconciliation run identifier used as ``causation_id``
            in the audit journal.
        max_age_days: Staleness cutoff in days (default
            ``_FLAG_FOR_STAGE2_MEM0_MAX_AGE_DAYS`` == 14).
        now: Reference "current time" for the cutoff calculation. Defaults to
            ``datetime.now(UTC)``; tests inject a fixed value.
        scroll_limit: Max records to enumerate in one scroll (default 1000).

    Returns:
        Number of memories successfully deleted (0 if nothing is stale, on
        enumeration failure, or on a confirmed-empty count short-circuit).
    """
    swept = await _sweep_stale_mem0_pool(
        memory_service,
        project_id,
        run_id,
        source='flag_for_stage2',
        gc_sweep_source=_FLAG_FOR_STAGE2_GC_SWEEP_SOURCE,
        max_age_days=max_age_days,
        log_name='_sweep_stale_mem0_flag_for_stage2_markers',
        now=now,
        scroll_limit=scroll_limit,
        count_short_circuit=True,
        enum_filters=_FLAG_FOR_STAGE2_ENUM_FILTERS,
    )
    # Diagnostic-only; never affects the returned sweep count (task 2966
    # amendment, reviewer finding — see _warn_on_flag_for_stage2_type_drift).
    await _warn_on_flag_for_stage2_type_drift(memory_service, project_id, run_id)
    return swept


async def _sweep_entity_standing_decision_growth(
    memory_service,
    project_id: str,
    run_id: str,
) -> dict:
    """Corroborate each ACTIVE entity_standing_decision against the live graph (ζ).

    The growth-freshness sweep (task 2899), a sibling of the Stage-2-tail marker
    sweeps. For each ACTIVE ``entity_standing_decision`` row it fetches the
    entity's LIVE edge count via graphiti and EXPIRES the row (reason='growth',
    through the single-source :func:`expire_entity_standing_decision` flip helper)
    when the entity has grown materially past its decision-time snapshot:

        live > edge_count_at_decision * factor
        OR live >= edge_count_at_decision + abs_delta

    ``factor`` / ``abs_delta`` default to :data:`STANDING_DECISION_GROWTH_FACTOR`
    / :data:`STANDING_DECISION_GROWTH_ABS_DELTA` and are overridable per row via
    optional payload keys ``growth_factor`` / ``growth_abs_delta`` (a non-numeric
    override is ignored in favour of the default).

    **Fail-safe (never raises):**

    * Unwired ledger (``memory_service.recon_ledger`` is ``None``) ⇒
      ``{'checked': 0, 'expired': 0, 'failed': False}`` — the sweep did not run,
      which is NOT a failure (mirrors the writer's ledger-None convention).
    * A failure LISTING the active rows ⇒ ``failed=True`` (nothing checked).
    * A per-row error (unparseable/incomplete payload, live-edge fetch failure,
      or flip failure) ⇒ that row is LEFT ACTIVE (TTL-bounded; re-verified next
      cycle) and ``failed=True``; the sweep continues to the next row. Leaving a
      row active on unverified information is strictly safer than expiring it —
      the marker sweeps' uncertain⇒keep direction.

    ``failed=True`` is the signal the Stage-2 tail feeds to the consecutive-
    failure streak escalation (:func:`_maybe_escalate_growth_sweep_failures`).

    Returns ``{'checked': int, 'expired': int, 'failed': bool}``.
    """
    ledger = getattr(memory_service, 'recon_ledger', None)
    if ledger is None:
        return {'checked': 0, 'expired': 0, 'failed': False}

    log_extra = {'project_id': project_id, 'run_id': run_id}
    try:
        rows = await ledger.list_entity_standing_decisions(project_id, state=STATE_ACTIVE)
    except Exception:
        logger.warning(
            '_sweep_entity_standing_decision_growth: listing active standing '
            'decisions failed for project_id=%s (sweep did not run)',
            project_id, extra=log_extra, exc_info=True,
        )
        return {'checked': 0, 'expired': 0, 'failed': True}

    checked = 0
    expired = 0
    failed = False
    for row in rows:
        checked += 1

        # Reconstruct the decision-time snapshot + optional per-record overrides.
        try:
            payload = json.loads(row.payload_json)
            decision_count = payload['edge_count_at_decision']
        except Exception:
            failed = True
            logger.warning(
                '_sweep_entity_standing_decision_growth: unreadable payload/snapshot '
                'for entity_uuid=%s project_id=%s (left active)',
                row.entity_uuid, project_id, extra=log_extra, exc_info=True,
            )
            continue
        if not isinstance(decision_count, (int, float)):
            failed = True
            logger.warning(
                '_sweep_entity_standing_decision_growth: non-numeric '
                'edge_count_at_decision=%r for entity_uuid=%s project_id=%s (left active)',
                decision_count, row.entity_uuid, project_id, extra=log_extra,
            )
            continue
        factor = payload.get('growth_factor')
        if not isinstance(factor, (int, float)):
            factor = STANDING_DECISION_GROWTH_FACTOR
        abs_delta = payload.get('growth_abs_delta')
        if not isinstance(abs_delta, (int, float)):
            abs_delta = STANDING_DECISION_GROWTH_ABS_DELTA

        # Sample the live edge count — a fetch failure leaves the row ACTIVE.
        try:
            edges = await memory_service.graphiti.get_valid_edges_for_node(
                row.entity_uuid, group_id=project_id
            )
            live = len(edges)
        except Exception:
            failed = True
            logger.warning(
                '_sweep_entity_standing_decision_growth: live edge fetch failed for '
                'entity_uuid=%s project_id=%s (left active, TTL-bounded)',
                row.entity_uuid, project_id, extra=log_extra, exc_info=True,
            )
            continue

        if live > decision_count * factor or live >= decision_count + abs_delta:
            try:
                await expire_entity_standing_decision(
                    ledger, row, reason=EXPIRY_REASON_GROWTH
                )
                expired += 1
            except Exception:
                failed = True
                logger.warning(
                    '_sweep_entity_standing_decision_growth: flip to expired/growth '
                    'failed for entity_uuid=%s project_id=%s (left active)',
                    row.entity_uuid, project_id, extra=log_extra, exc_info=True,
                )

    return {'checked': checked, 'expired': expired, 'failed': failed}


# ── Growth-sweep failure-streak escalation (task 2899 ζ, INV-4) ──────────────
# Pure helpers mirroring task_count_snapshot_cadence's streak-on-miss shape,
# INVERTED to streak-on-failure: the counted flag is True=sweep-failed (a
# Graphiti error left >=1 active row unverified this cycle), not False=miss. The
# streak is recomputed from journalled prior-run stats each cycle (no persisted
# counter), so any successful sweep resets it and it survives a restart.


def _extract_growth_sweep_failed(stage_report: object) -> bool | None:
    """Read the growth-sweep failed flag off a Stage-2 report.

    Accepts a real ``StageReport`` (attribute access), a raw dict shape (a
    journal-reconstructed report or test double), or ``None``.

    Returns ``True`` when
    ``stats[ENTITY_STANDING_DECISION_GROWTH_SWEEP_FAILED_STAT_KEY] == 1``,
    ``False`` when ``== 0``, and ``None`` when the report is ``None``, its
    ``stats`` is absent, or the key itself is absent — "unknown", never
    miscounted as a confirmed failure or a confirmed success.
    """
    if stage_report is None:
        return None
    if isinstance(stage_report, dict):
        stats = stage_report.get('stats') or {}
    else:
        stats = getattr(stage_report, 'stats', None) or {}
    value = stats.get(ENTITY_STANDING_DECISION_GROWTH_SWEEP_FAILED_STAT_KEY)
    if value == 1:
        return True
    if value == 0:
        return False
    return None


def _compute_growth_sweep_failure_streak(recent_flags: list[bool | None]) -> int:
    """Count the leading run of consecutive failures in *recent_flags*.

    *recent_flags* is most-recent-first. Counts consecutive ``True`` entries
    from the start, stopping at the first ``False`` (a successful sweep resets
    the streak) or ``None`` (unknown — stop, fail-safe: an inconclusive cycle
    must never be counted as either a failure or a reset).
    """
    streak = 0
    for flag in recent_flags:
        if flag is True:
            streak += 1
        else:
            break
    return streak


def _evaluate_growth_sweep_escalation(
    current_failed: bool | None,
    prior_flags: list[bool | None],
    *,
    threshold: int = GROWTH_SWEEP_FAILURE_STREAK_THRESHOLD,
) -> dict:
    """Decide whether the current cycle's sweep failure should escalate.

    Fail-safe short-circuit (checked before the streak is computed): when
    *current_failed* is not ``True`` (i.e. ``False`` — the sweep succeeded this
    cycle — or ``None`` — it did not run / was inconclusive) the streak is ``0``
    and there is no escalation. Only a CONFIRMED current failure can trigger.

    Otherwise the streak is ``_compute_growth_sweep_failure_streak(prior_flags)
    + 1`` (the "+1" is the current confirmed failure) and ``escalate`` is
    ``streak >= threshold``.

    Returns ``{'streak': int, 'escalate': bool}``.
    """
    if current_failed is not True:
        return {'streak': 0, 'escalate': False}
    streak = _compute_growth_sweep_failure_streak(prior_flags) + 1
    return {'streak': streak, 'escalate': streak >= threshold}


async def _maybe_escalate_growth_sweep_failures(
    escalation_queue,
    journal,
    project_id: str,
    run_id: str,
    current_failed: bool | None,
    *,
    threshold: int = GROWTH_SWEEP_FAILURE_STREAK_THRESHOLD,
) -> bool:
    """File a level-1 recon escalation on a sustained growth-sweep failure streak.

    Mirrors ``stage1_stall_detector.maybe_escalate_stalled_tasks`` (a Stage may
    file a recon escalation directly via ``self._escalation_queue``) and the
    harness's ``_maybe_escalate_stale_task_count_snapshot`` journal-recompute:
    the consecutive-failure streak is recomputed each cycle from
    ``journal.get_recent_runs`` rather than a stored counter, so it resets on any
    successful sweep and survives a restart with no new schema.

    * Fail-safe short-circuit: only a CONFIRMED current failure
      (*current_failed* is ``True``) is eligible — a successful or inconclusive
      cycle returns ``False`` WITHOUT a journal read (cheap steady state).
    * Prior runs are filtered to full-cycle COMPLETED runs only, EXCLUDING the
      current *run_id* (a remediation/targeted or still-running run neither
      counts toward nor resets the streak — matching the snapshot cadence).
    * On reaching *threshold*, a single ``Escalation`` (level=1) is filed under
      a STABLE per-project synthetic key ``recon-esd-growth-sweep:<project_id>``
      and deduped via ``escalation_queue.has_open_l1`` so a persistent streak
      folds into one open escalation rather than re-filing every cycle.

    Fails open — never raises (the journal read is wrapped too), so a journal
    hiccup never aborts the Stage-2 tail. Returns whether a NEW escalation was
    filed. No-op (``False``) when the ``escalation`` package is unavailable or
    *escalation_queue* is ``None``.
    """
    if Escalation is None or escalation_queue is None:
        return False
    if current_failed is not True:
        return False
    try:
        recent = await journal.get_recent_runs(project_id, limit=max(20, threshold * 4))
        prior_runs = [
            r for r in recent
            if getattr(r, 'id', None) != run_id
            and str(getattr(r, 'run_type', '')) == 'full'
            and str(getattr(r, 'status', '')) == 'completed'
        ]
        # Defensive re-sort most-recent-first: get_recent_runs already orders by
        # started_at DESC, but _compute_growth_sweep_failure_streak depends on it.
        prior_runs.sort(
            key=lambda r: getattr(r, 'started_at', None) or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        prior_flags: list[bool | None] = []
        for r in prior_runs:
            stage_reports = getattr(r, 'stage_reports', None)
            report = stage_reports.get('task_knowledge_sync') if isinstance(stage_reports, dict) else None
            prior_flags.append(_extract_growth_sweep_failed(report))

        result = _evaluate_growth_sweep_escalation(current_failed, prior_flags, threshold=threshold)
        if not result['escalate']:
            return False

        key = f'recon-esd-growth-sweep:{project_id}'
        if escalation_queue.has_open_l1(key):
            logger.info(
                'reconciliation.growth_sweep_failure_escalation_suppressed: '
                'project_id=%s already has an open level-1 escalation',
                project_id, extra={'project_id': project_id, 'run_id': run_id},
            )
            return False

        streak = result['streak']
        esc = Escalation(
            id=escalation_queue.make_id(key),
            task_id=key,
            agent_role='reconciliation-stage2',
            severity='blocking',
            category=ENTITY_STANDING_DECISION_GROWTH_SWEEP_ESCALATION_CATEGORY,
            summary=(
                f'entity_standing_decision growth sweep failed for {streak} '
                f'consecutive full cycles (project {project_id})'
            ),
            detail=(
                'The reconciliation Stage-2 entity_standing_decision '
                'growth-freshness sweep left >=1 ACTIVE row unverified (a Graphiti '
                f'error) for {streak} consecutive completed full reconciliation '
                f'cycles for project {project_id!r} (latest run_id={run_id}). '
                'Active standing decisions may be growing stale without '
                'corroboration; investigate Graphiti reachability for this project.'
            ),
            level=1,
        )
        escalation_queue.submit(esc)
        return True
    except Exception as e:
        logger.warning(
            'reconciliation.growth_sweep_failure_escalation_failed',
            extra={'project_id': project_id, 'run_id': run_id, 'error': str(e)},
        )
        return False


async def _verify_task_count_snapshot_written(
    memory_service,
    project_id: str,
    run_window_start: datetime | None,
    *,
    scroll_limit: int = 1000,
) -> bool | None:
    """Best-effort freshness check: was a task_count_snapshot written this run window?

    Enumerates existing ``kind='task_count_snapshot'`` Mem0 records via
    ``get_memories_by_metadata`` (deterministic Qdrant scroll, NOT semantic
    search) and checks whether any record's ``created_at`` falls within the
    current run window via :func:`_marker_is_within_run_window`.

    The existing snapshot write does not reliably carry ``metadata.run_id``,
    so this checks the write's *timestamp* against the run window rather than
    matching on run_id (which would read 0 forever).

    Follows this module's never-raises, best-effort, WARNING-on-failure
    contract, including the crucial distinction between a
    CONFIRMED miss (``False``) and an inconclusive check (``None``).

    Unlike that helper, this one passes an explicit ``limit`` (mirroring
    :func:`_sweep_stale_persistence_markers`'s ``scroll_limit`` idiom —
    task_count_snapshot has no GC or pool-cap, so its record count grows
    unbounded over cycles) and treats a
    saturated page as INCONCLUSIVE rather than a confirmed miss: Qdrant's
    scroll orders by point id, not ``created_at``, so once matches exceed
    *scroll_limit* the freshest record can fall outside the returned page —
    misreading that as a confirmed miss would eventually fire a false
    stale-snapshot escalation even though a fresh snapshot exists. A ``True``
    result is unaffected by truncation: finding a match anywhere in a
    (possibly partial) page is still real evidence of a fresh write.

    Args:
        memory_service: Service with ``get_memories_by_metadata``.
        project_id: Project scope for the query.
        run_window_start: Start of the current run window, or ``None`` when
            unknown.
        scroll_limit: Max records to enumerate in one scroll (default 1000,
            matching the sibling GC helpers in this module).

    Returns:
        ``True`` when a record's ``created_at`` falls within the run window
        (confirmed fresh). ``False`` when the query succeeded, the page was
        NOT saturated (fewer than *scroll_limit* records), and no record
        falls within the window (confirmed miss). ``None`` when
        *run_window_start* is ``None`` (window unknown), the query itself
        raised (transient failure), or no match was found in a saturated
        (possibly-truncated) page — "unknown", never miscounted as a
        confirmed miss.
    """
    if run_window_start is None:
        return None
    try:
        members = await memory_service.get_memories_by_metadata(
            project_id=project_id,
            filters={'kind': TASK_COUNT_SNAPSHOT_KIND},
            limit=scroll_limit,
        )
    except Exception:
        logger.warning(
            'reconciliation._verify_task_count_snapshot_written: '
            'get_memories_by_metadata failed for project_id=%s; absence NOT '
            'confirmed (transient failure, not treated as a miss)',
            project_id,
            extra={'project_id': project_id},
        )
        return None
    found = any(
        _marker_is_within_run_window(member.get('created_at'), run_window_start)
        for member in members
    )
    if found:
        return True
    if len(members) >= scroll_limit:
        logger.warning(
            'reconciliation._verify_task_count_snapshot_written: enumerated '
            '%d of scroll_limit=%d task_count_snapshot records with none in '
            'the current run window — Qdrant scroll is point-id ordered, NOT '
            'created_at ordered, so a truncated page may exclude the '
            'freshest record; treating as unknown rather than a confirmed '
            'miss. Re-run with a higher scroll_limit.',
            len(members), scroll_limit,
            extra={'project_id': project_id},
        )
        return None
    return False


async def _prune_task_count_snapshots(
    memory_service,
    project_id: str,
    run_id: str,
    *,
    scroll_limit: int = 1000,
    stats: dict | None = None,
) -> int:
    """Delete every existing ``kind='task_count_snapshot'`` Mem0 record (task 2429).

    Called by :func:`_write_task_count_snapshot` immediately before its
    ``add_memory`` call, so the net effect of one cycle is exactly one
    canonical snapshot surviving per project — self-correcting the
    near-duplicate accumulation that manual/LLM-driven cleanups failed to
    keep pruned.

    Enumerates deterministically via ``memory_service.get_memories_by_metadata``
    (Qdrant payload-filter scroll) — NEVER semantic search, which silently
    drops low-similarity rows and is unsuitable for exhaustive GC. Unlike
    :func:`_sweep_stale_persistence_markers`, this applies NO age cutoff:
    every enumerated ``kind='task_count_snapshot'`` match is deleted —
    including LLM-authored duplicates — so the caller's subsequent
    ``add_memory`` leaves exactly one canonical snapshot.

    Multi-cycle convergence on a large backlog (task 2429 review): when the
    existing backlog exceeds *scroll_limit* records — precisely the
    accumulation this change targets — only the first *scroll_limit*
    matches are pruned in a given cycle (see the saturation WARNING
    below); the remainder drains at up to *scroll_limit* per subsequent
    cycle rather than being fully pruned in one run. This is an accepted
    trade-off, not a bug: it is loudly logged (no silent truncation) and
    strictly improves on the unbounded pre-2429 accumulation either way.

    Best-effort and never raises: an enumeration failure or a per-item
    delete failure degrades to a WARNING log and is excluded from the
    returned count, rather than aborting the caller's write.

    Args:
        memory_service: Service with ``get_memories_by_metadata`` and
            ``delete_memory``.
        project_id: Project scope for enumeration and delete calls.
        run_id: Current reconciliation run identifier used as ``causation_id``
            in the audit journal.
        scroll_limit: Max records to enumerate in one scroll (default 1000).
        stats: Optional dict to populate with this cycle's runtime-observability
            counts — see ``task_count_snapshot_cadence.SNAPSHOT_PRUNE_ENUMERATED_STAT_KEY``
            / ``SNAPSHOT_PRUNED_STAT_KEY`` / ``SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY``
            / ``SNAPSHOT_PRUNE_TRUNCATED_STAT_KEY`` (task 2646; the last one
            added in the amendment round). Left untouched when ``None`` (the
            default).

    Returns:
        Number of memories successfully deleted (0 if nothing matched, or
        on enumeration failure).
    """
    enumeration_ok = True
    try:
        members = await memory_service.get_memories_by_metadata(
            project_id=project_id,
            filters={'kind': TASK_COUNT_SNAPSHOT_KIND},
            limit=scroll_limit,
        )
    except Exception:
        logger.warning(
            'reconciliation._prune_task_count_snapshots: '
            'get_memories_by_metadata failed for project_id=%s; skipping prune',
            project_id,
            extra={'project_id': project_id, 'run_id': run_id},
        )
        enumeration_ok = False
        members = []

    # Silent-empty enumeration guard (task 2655, 5th recorded recurrence).
    # Mem0Backend.scroll_by_metadata swallows TimeoutError and returns []
    # (mem0_client.py:392-407), while its sibling count_by_metadata lets
    # timeouts propagate (mem0_client.py:296-339) -- so an empty,
    # NON-exceptional scroll page is ambiguous: it could be a genuine empty
    # pool, or a swallowed Qdrant read timeout that would otherwise let the
    # caller's canonical write proceed without ever pruning the prior
    # snapshot(s), growing the byte-identical duplicate pile. Cross-check
    # via count_memories_by_metadata, which propagates timeouts: a raised
    # cross-check, or a confirmed count > 0, is the swallowed-timeout
    # fingerprint. Only runs on an empty scroll -- a non-empty page skips
    # the extra count call entirely. Best-effort: never raises.
    #
    # Benign false-positive note (reviewer finding, amendment round): the
    # fingerprint assumes scroll and count agree for committed records. If a
    # task_count_snapshot was authored earlier in this SAME cycle by the
    # still-live Stage-2 LLM path (see the duplicate-write note in run()) and
    # Qdrant's scroll vs. count views momentarily disagree under eventual
    # consistency, this guard can flip enumeration_ok to 0 against an
    # otherwise healthy pool. That is an accepted, self-correcting trigger --
    # it costs one skipped write, recovered next cycle -- so not every
    # enumeration_ok == 0 log below is a confirmed Qdrant timeout.
    if enumeration_ok and not members:
        try:
            snapshot_count = await memory_service.count_memories_by_metadata(
                project_id=project_id,
                filters={'kind': TASK_COUNT_SNAPSHOT_KIND},
            )
            enumeration_confirmed_empty = snapshot_count <= 0
        except Exception:
            logger.warning(
                'reconciliation._prune_task_count_snapshots: '
                'count_memories_by_metadata cross-check failed for project_id=%s '
                'after an empty scroll; treating enumeration as not-ok rather '
                'than a confirmed empty pool (possible swallowed Qdrant read '
                'timeout, task 2655)',
                project_id,
                extra={'project_id': project_id, 'run_id': run_id},
            )
            enumeration_ok = False
        else:
            if not enumeration_confirmed_empty:
                logger.warning(
                    'reconciliation._prune_task_count_snapshots: '
                    'get_memories_by_metadata scroll returned 0 task_count_snapshot '
                    'records for project_id=%s but count_memories_by_metadata '
                    'reports %d existing -- swallowed Qdrant read timeout '
                    'fingerprint (task 2655); treating enumeration as not-ok '
                    'rather than a genuine empty pool',
                    project_id, snapshot_count,
                    extra={'project_id': project_id, 'run_id': run_id},
                )
                enumeration_ok = False

    if len(members) >= scroll_limit:
        logger.warning(
            'reconciliation._prune_task_count_snapshots: enumerated %d of scroll_limit=%d '
            'task_count_snapshot records — scroll cap reached; older stale snapshots may '
            'remain unpruned this cycle; re-run with a higher scroll_limit.',
            len(members), scroll_limit,
            extra={'project_id': project_id, 'run_id': run_id},
        )

    ids = [member['id'] for member in members if member.get('id')]

    success_count = 0
    if ids:
        # Two-tier check via gather_collect (fused_memory.utils.async_utils).
        # Pass 1 (inside gather_collect): re-raises structured-cancellation
        # signals — this preserves the structured-cancellation contract and
        # prevents this delete pass from silently converting a shutdown
        # signal into an under-counted deletion tally.
        # Pass 2 (below): per-item degrade-to-warning on ordinary Exceptions.
        results = await gather_collect(
            memory_service.delete_memory(
                memory_id=mid,
                store='mem0',
                project_id=project_id,
                causation_id=run_id,
                _source=_TASK_COUNT_SNAPSHOT_PRUNE_SOURCE,
            )
            for mid in ids
        )

        for mid, result in zip(ids, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(
                    'reconciliation._prune_task_count_snapshots: delete failed for memory_id=%s; not counted',
                    mid,
                    extra={'project_id': project_id, 'memory_id': mid, 'run_id': run_id},
                )
            else:
                success_count += 1

        if success_count:
            logger.info(
                'reconciliation._prune_task_count_snapshots: pruned %d stale task_count_snapshot '
                'record(s) for project_id=%s prior to canonical write',
                success_count, project_id,
                extra={'project_id': project_id, 'run_id': run_id},
            )

    if stats is not None:
        stats[SNAPSHOT_PRUNE_ENUMERATED_STAT_KEY] = len(members)
        stats[SNAPSHOT_PRUNED_STAT_KEY] = success_count
        stats[SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY] = 1 if enumeration_ok else 0
        stats[SNAPSHOT_PRUNE_TRUNCATED_STAT_KEY] = 1 if len(members) >= scroll_limit else 0

    return success_count


async def _write_task_count_snapshot(
    memory_service,
    taskmaster,
    project_root: str,
    project_id: str,
    run_id: str,
    run_window_start: datetime | None,
    *,
    stats: dict | None = None,
) -> bool | None:
    """Deterministically write this cycle's task_count_snapshot Mem0 record.

    Makes the Mem0 ``task_count_snapshot`` write structural (task 2325): a
    plain Python ``add_memory`` call performed unconditionally at the end of
    ``run()`` for non-blocked projects, rather than depending on the Stage-2
    LLM remembering the memory-stored "Snapshot Discipline" norm.

    Self-pruning (task 2429): immediately before ``add_memory``, calls
    :func:`_prune_task_count_snapshots` to delete every existing
    ``kind='task_count_snapshot'`` record for this project, so the net
    effect of one cycle is at most ONE canonical snapshot surviving —
    self-correcting the near-duplicate accumulation that manual/LLM-driven
    cleanups repeatedly failed to keep pruned. The prune runs after the
    ``taskmaster is None`` early-return and after ``content`` is derived, so
    a skipped/failed fetch never deletes existing snapshots without a
    replacement in hand; the prune helper is itself best-effort and never
    raises, so a prune failure cannot abort this write.

    Accepted zero-snapshot window (task 2429 review): if the prune
    succeeds but the subsequent ``add_memory`` call itself then fails
    (e.g. a transient mem0 outage), this cycle deletes the prior
    snapshot(s) without landing a replacement, leaving zero
    ``task_count_snapshot`` records until the next cycle's write
    succeeds. This is an accepted, self-correcting trade-off rather than
    an oversight: the alternative (write-then-prune-all-but-the-just-
    written-id) avoids the window but must exclude the just-written id
    under Qdrant scroll eventual-consistency, which is more complex and
    race-prone for no material safety gain (see the task 2429 plan's
    design decisions). The window is also no worse than today's
    escalation path — a failed write already makes
    ``_verify_task_count_snapshot_written`` return ``None``
    (inconclusive) rather than a confirmed miss, so it cannot spuriously
    grow the consecutive-miss streak that drives the stale-snapshot
    escalation. Pinned by
    ``test_add_memory_failure_after_successful_prune_returns_none``.

    Enumeration-failure write-gate (task 2655; supersedes task 2646's note
    that the write proceeds regardless of the prune's enumeration outcome):
    if :func:`_prune_task_count_snapshots` reports
    ``SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY == 0`` — including the silent-
    empty-scroll case caught by its count cross-check — this function logs a
    structured WARNING and returns ``None`` BEFORE calling ``add_memory``.
    Writing a fresh snapshot when the prior one(s) could not be confirmed
    enumerated/pruned is exactly what grows the byte-identical duplicate
    pile (the recurring incident this task exists to fix); skipping is
    self-correcting, since the next healthy cycle enumerates and prunes all
    accumulated duplicates before writing one. Pinned by
    ``TestWriteTaskCountSnapshotEnumerationGate``.

    Counts are derived by self-fetching via ``taskmaster.get_tasks`` and
    filtering with :func:`filter_task_tree` — mirroring
    ``assemble_payload``'s own self-fetch fallback idiom — rather than
    reusing any tree stashed by ``assemble_payload``, which can be
    short-circuited or skipped and would then be stale.

    Uses the internal ``memory_service.add_memory`` idiom directly, NOT the
    MCP layer, so the ``ReconSnapshotWriteRejected`` temporal_facts guard
    never applies here.

    Best-effort: never raises. Returns ``True`` (wrote) or ``None``
    (skipped — no taskmaster — or failed); NEVER ``False``, so a failed
    best-effort write stays "inconclusive" for the caller rather than being
    recorded as a confirmed miss (which would spuriously grow the
    consecutive-miss streak that drives the stale-snapshot escalation).

    Args:
        memory_service: Service with ``add_memory``.
        taskmaster: Service with ``get_tasks``, or ``None``.
        project_root: Project root for the ``get_tasks`` call.
        project_id: Project scope for the write.
        run_id: Current reconciliation run identifier — both the write's
            ``metadata.run_id`` and its ``causation_id``.
        run_window_start: Start of the current run window, or ``None`` when
            unknown — used only to derive the content's ``as_of`` date.
        stats: Optional dict forwarded to ``_prune_task_count_snapshots`` to
            populate with this cycle's prune runtime-observability counts
            (task 2646). Populated only when the prune is actually reached —
            left untouched if *taskmaster* is ``None`` or the fetch/filter
            step fails before the prune call. Callers reading these keys
            back out of ``report.stats`` MUST use ``.get(key, default)``
            rather than direct indexing — see the "Conditional presence"
            note on
            ``task_count_snapshot_cadence.SNAPSHOT_PRUNE_TRUNCATED_STAT_KEY``
            (amendment round, task 2646 review). Also read back internally
            (task 2655) immediately after the prune call to decide the
            enumeration-failure write-gate below — a throwaway local dict is
            used for this when the caller passes ``None``, so the gate reads
            correctly even when the caller doesn't want the stats.

    Returns:
        ``True`` on a successful write; ``None`` when *taskmaster* is
        ``None``, the prune could not confirm a clean enumeration (task
        2655 write-gate), or any step of the fetch/filter/write fails.
    """
    if taskmaster is None:
        return None
    try:
        tasks_data = await taskmaster.get_tasks(project_root=project_root)
        tree = filter_task_tree(tasks_data)
        as_of = (
            run_window_start.date().isoformat()
            if isinstance(run_window_start, datetime)
            else None
        )
        # Task 2738: SqliteTaskBackend.get_tasks auto-creates an empty
        # tasks.db and returns {'tasks': []} for ANY project_root (never
        # raising), so a zero-count tree at a non-git project_root (e.g. a
        # project whose repo was deleted) is a false census, not a
        # genuinely empty project -- indistinguishable from one at the data
        # layer. Gate ONLY on total_count == 0 so the common non-empty path
        # stays byte-identical and never pays the git-subprocess cost.
        # Note: resolve_main_checkout raises ValueError for two distinct
        # causes -- project_root not inside any git working tree, OR the
        # `git` executable being missing from the host entirely (it wraps
        # FileNotFoundError) -- and does not distinguish between them. On a
        # host with no git binary, EVERY zero-count project would take this
        # branch, including a genuinely empty git-backed one that would
        # otherwise get its legitimate numeric "0 total" record. This repo's
        # hosts always have git installed, so it is not a live bug here, and
        # "unavailable" is the fail-safe direction if it ever were (loud
        # sentinel over a silently-wrong zero census) -- but be aware the two
        # causes are conflated should resolve_main_checkout ever need to
        # expose them separately.
        snapshot_unavailable = False
        if tree.total_count == 0:
            try:
                resolve_main_checkout(project_root)
            except ValueError:
                snapshot_unavailable = True
        if snapshot_unavailable:
            content = build_task_count_snapshot_unavailable_content(
                project_id, project_root, as_of=as_of,
            )
        else:
            content = build_task_count_snapshot_content(
                project_id,
                total=tree.total_count,
                done=tree.done_count,
                cancelled=tree.cancelled_count,
                active=len(tree.active_tasks),
                other=tree.other_count,
                highest_task_id=tree.max_task_id,
                as_of=as_of,
            )
        # prune_stats is ALWAYS a real dict (never None) so the write-gate
        # below can read SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY back
        # regardless of whether the caller wanted the observability stats
        # (task 2655). When the caller passed a real `stats` dict (e.g.
        # run()'s report.stats), prune_stats IS that same object, so the
        # four prune stats still surface there unchanged.
        prune_stats = stats if stats is not None else {}
        await _prune_task_count_snapshots(
            memory_service, project_id, run_id, stats=prune_stats,
        )
        if prune_stats.get(SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY) == 0:
            logger.warning(
                'reconciliation._write_task_count_snapshot: '
                'prune enumeration failed for project_id=%s run_id=%s; '
                'skipping this cycle\'s snapshot write to avoid growing the '
                'duplicate pile (task 2655) -- the next healthy cycle will '
                'prune and write normally',
                project_id, run_id,
                extra={'project_id': project_id, 'run_id': run_id},
            )
            return None
        metadata = {
            'kind': TASK_COUNT_SNAPSHOT_KIND,
            'stage': 'task_knowledge_sync',
            'run_id': run_id,
        }
        if snapshot_unavailable:
            metadata['snapshot_status'] = 'unavailable'
        await memory_service.add_memory(
            content=content,
            category=TASK_COUNT_SNAPSHOT_CATEGORY,
            project_id=project_id,
            metadata=metadata,
            causation_id=run_id,
            _source=_TASK_COUNT_SNAPSHOT_WRITE_SOURCE,
        )
    except Exception:
        logger.warning(
            'reconciliation._write_task_count_snapshot: '
            'deterministic write failed for project_id=%s run_id=%s',
            project_id, run_id,
            extra={'project_id': project_id, 'run_id': run_id},
        )
        return None
    return True


async def _track_flag_persistence(
    memory_service,
    project_id: str,
    run_id: str,
    flag_ids: list[str],
) -> dict[str, int]:
    """Track how many Stage 2 cycles each flag has survived.

    For each *flag_id* in *flag_ids*:

    1. Counts prior ``stage2_persistence_marker`` memories via a metadata-filtered
       Qdrant count (``MemoryService.count_memories_by_metadata``) — a deterministic
       key-equality lookup, NOT semantic search.  The previous implementation used
       ``memory_service.search`` with ``query='stage2_persistence_marker flag_id=...'``
       and ``limit=100``, which silently dropped prior markers when unrelated
       higher-similarity hits pushed them off the bottom of the top-N — see
       reviewer note on FIX D, task 1139.
    2. Writes a fresh marker so the count accumulates across cycles.
    3. Returns ``{flag_id: prior_count + 1}`` where the "+1" accounts for the
       current cycle.

    Count reads are issued in parallel via ``asyncio.gather`` (no ordering
    constraint); marker writes are also parallelised.  Both phases use
    ``return_exceptions=True`` so a single Qdrant failure degrades gracefully
    rather than aborting the whole batch.

    On count failure: prior_count defaults to 0 so the cycle count is 1.
    On add_memory failure: the count is still returned (write is best-effort).
    Both failures log WARNING.

    Note: markers accumulate monotonically (same pattern as ``stage1_flag_marker``
    in ``flag_dedup.py``).  FIX C's prompt-driven deletion ensures healthy flags
    never reach threshold; the monotonic growth is bounded to failure cases.
    Automated age-based GC is performed each cycle by
    :func:`_sweep_stale_persistence_markers` (see ``TaskKnowledgeSync.run()``),
    which deletes markers older than ``STAGE2_PERSISTENCE_MARKER_MAX_AGE_DAYS``
    (task 2095).

    Note (task 1256, updated task 2228 W5-κ): this function receives only
    *surviving_ids* — flags whose ``metadata.run_id`` differs from the active
    run are partitioned out by :func:`_query_stage2_flags` upstream and never
    reach this call; that residue is reaped separately by the recon_ledger's
    single GC pass (:func:`_gc_recon_markers`, called from ``run()``) on its
    TTL/terminal-task schedule, not by an immediate per-cycle delete.  The
    counter therefore no longer observes Stage 2 delete failures from
    previous cycles; it only counts Stage 1 re-flags that survive within the
    current cycle's ``run_id``.
    """
    if not flag_ids:
        return {}

    # ── count phase (parallel) ───────────────────────────────────────────────
    # Two-tier check via gather_collect (fused_memory.utils.async_utils).
    # Pass 1 (inside gather_collect): re-raises structured-cancellation
    # signals — this preserves the structured-cancellation contract and
    # prevents the persistence counter from silently converting a shutdown
    # signal into a wrongly-defaulted prior_count of 0.
    # Pass 2 (below): per-item degrade-to-warning on ordinary Exceptions.
    count_results = await gather_collect(
        memory_service.count_memories_by_metadata(
            project_id=project_id,
            filters={'source': _STAGE2_PERSISTENCE_MARKER_SOURCE, 'flag_id': fid},
        )
        for fid in flag_ids
    )
    prior_counts: dict[str, int] = {}
    for fid, result in zip(flag_ids, count_results, strict=True):
        if isinstance(result, Exception):
            logger.warning(
                'reconciliation._track_flag_persistence: count failed for flag_id=%s; '
                'treating prior_count as 0',
                fid,
                extra={'project_id': project_id, 'flag_id': fid},
            )
            prior_counts[fid] = 0
        else:
            prior_counts[fid] = result

    # ── write phase (parallel) ───────────────────────────────────────────────
    # Two-tier check via gather_collect (fused_memory.utils.async_utils).
    # Pass 1 (inside gather_collect): re-raises structured-cancellation
    # signals — this preserves the structured-cancellation contract and
    # prevents the persistence marker write from silently converting a
    # shutdown signal into a silently-dropped marker.
    # Pass 2 (below): per-item degrade-to-warning on ordinary Exceptions.
    write_results = await gather_collect(
        memory_service.add_memory(
            content=f'Stage 2 flag-persistence marker: flag_id={fid} run={run_id}',
            category='observations_and_summaries',
            project_id=project_id,
            metadata={
                'source': _STAGE2_PERSISTENCE_MARKER_SOURCE,
                'flag_id': fid,
                'run_id': run_id,
            },
            causation_id=run_id,
            _source='stage2_flag_relay',
        )
        for fid in flag_ids
    )
    for fid, result in zip(flag_ids, write_results, strict=True):
        if isinstance(result, Exception):
            logger.warning(
                'reconciliation._track_flag_persistence: add_memory failed for flag_id=%s; '
                'count still returned',
                fid,
                extra={'project_id': project_id, 'flag_id': fid},
            )

    return {fid: prior_counts[fid] + 1 for fid in flag_ids}


async def _filter_already_escalated_flags(
    memory_service,
    project_id: str,
    flag_ids: list[str],
) -> tuple[list[str], list[str]]:
    """Split *flag_ids* into (newly_escalating, already_escalated).

    A flag is considered already-escalated iff a Mem0 memory exists with
    metadata ``{'source': 'stage2_escalation_marker', 'flag_id': <flag_id>}``.
    The check is a deterministic metadata-filtered count — same rationale as
    :func:`_track_flag_persistence` — to avoid escalating the same flag every
    cycle when FIX C deletion fails (reviewer note on FIX D, task 1139).

    Counts are issued in parallel via ``asyncio.gather``.  On count failure:
    the flag is treated as NOT already-escalated (i.e. it surfaces in
    *newly_escalating*).  This is fail-loud-via-escalation: a transient Qdrant
    glitch causes one extra escalation, not silent suppression of a real one.
    """
    if not flag_ids:
        return [], []

    # Two-tier check via gather_collect (fused_memory.utils.async_utils).
    # Pass 1 (inside gather_collect): re-raises structured-cancellation
    # signals — this preserves the structured-cancellation contract and
    # prevents this count phase from silently converting a shutdown signal
    # into a flag wrongly classified as newly-escalating.
    # Pass 2 (below): per-item degrade-to-warning on ordinary Exceptions.
    count_results = await gather_collect(
        memory_service.count_memories_by_metadata(
            project_id=project_id,
            filters={'source': _STAGE2_ESCALATION_MARKER_SOURCE, 'flag_id': fid},
        )
        for fid in flag_ids
    )
    newly: list[str] = []
    already: list[str] = []
    for fid, result in zip(flag_ids, count_results, strict=True):
        if isinstance(result, Exception):
            logger.warning(
                'reconciliation._filter_already_escalated_flags: count failed for '
                'flag_id=%s; treating as newly-escalating',
                fid,
                extra={'project_id': project_id, 'flag_id': fid},
            )
            newly.append(fid)
        elif result > 0:
            already.append(fid)
        else:
            newly.append(fid)
    return newly, already


async def _write_escalation_markers(
    memory_service,
    project_id: str,
    run_id: str,
    flag_ids: list[str],
) -> None:
    """Persist a ``stage2_escalation_marker`` for each newly-escalated flag.

    Subsequent cycles use :func:`_filter_already_escalated_flags` to look up
    these markers via a metadata-filtered Qdrant count, suppressing duplicate
    escalations even when the LLM fails to delete the underlying flag.

    Writes are issued in parallel via ``asyncio.gather``.  Best-effort: write
    failures log WARNING but do not raise — at worst this produces one duplicate
    escalation next cycle, never silent suppression.
    """
    if not flag_ids:
        return

    # Two-tier check via gather_collect (fused_memory.utils.async_utils).
    # Pass 1 (inside gather_collect): re-raises structured-cancellation
    # signals — this preserves the structured-cancellation contract and
    # prevents this marker write from silently converting a shutdown signal
    # into a silently-dropped marker.
    # Pass 2 (below): per-item degrade-to-warning on ordinary Exceptions.
    write_results = await gather_collect(
        memory_service.add_memory(
            content=(f'Stage 2 escalation marker: flag_id={fid} escalated in run={run_id}'),
            category='observations_and_summaries',
            project_id=project_id,
            metadata={
                'source': _STAGE2_ESCALATION_MARKER_SOURCE,
                'flag_id': fid,
                'run_id': run_id,
            },
            causation_id=run_id,
            _source='stage2_flag_relay',
        )
        for fid in flag_ids
    )
    for fid, result in zip(flag_ids, write_results, strict=True):
        if isinstance(result, Exception):
            logger.warning(
                'reconciliation._write_escalation_markers: add_memory failed for '
                'flag_id=%s; next cycle may re-escalate',
                fid,
                extra={'project_id': project_id, 'flag_id': fid},
            )


def _render_live_workflow_section(
    tasks: list[dict],
    project_root: ProjectRoot,
    *,
    now: datetime | None = None,
) -> str:
    """Render the '### Live-Workflow Signals' payload section for *tasks*.

    For each task in *tasks*, calls :func:`detect_live_workflow` and collects
    tasks whose :attr:`WorkflowLiveness.is_live` is True.  Returns an empty
    string when no task is live (keeps the payload tight).

    Each live task is listed with the firing signal names so the Stage 2 LLM
    can see which evidence contributed to the live designation:

    ```
    ### Live-Workflow Signals
    - task/4321: worktree, recent-commit
    ```

    Detector errors per task are swallowed and logged at WARNING level (the task
    is treated as not-live for that call — fail-safe, matching the harness gate).

    Each task's ``status`` is forwarded to the detector, so never-dispatched
    statuses (deferred/done/cancelled — see
    :data:`~fused_memory.services.live_workflow_detector.ORCH_LIVE_INELIGIBLE_STATUSES`)
    drop the project-wide orchestrator_live signal for that task while leaving
    its per-task worktree/commit signals unaffected.

    Each task's ``metadata.task_kind`` is also forwarded.  A BLOCKED
    deterministic task (``task_kind == 'deterministic'``) never acquires a
    worktree/branch of its own — it is routed to ``DeterministicRunner``
    instead — so the project-wide orchestrator_live signal is dropped for it
    too, the same way it is for never-dispatched statuses.  A normal blocked
    task (``task_kind`` absent or not ``'deterministic'``) keeps the signal
    ONLY when it carries genuine per-task evidence (a registered worktree or a
    recent commit), since it may legitimately auto-unblock mid-pipeline; a
    blocked normal task whose ONLY evidence is the bare project-wide
    orchestrator lock also has the signal dropped (task 2409 — closes the
    repeated re-deferral loop this caused for tasks 2335/2196).

    A PENDING deterministic PURE GATE — ``always_escalates`` truthy with no
    ``before_done``, classified by :func:`is_pure_gate_metadata` and forwarded
    as ``pure_gate`` — likewise has the project-wide signal dropped (task
    3751).  Its entire ``DeterministicRunner`` run is "file one born-at-L2
    escalation, stamp ``gate_escalated_at``, set status blocked": no script, no
    systemd, no ``git_ops``, and (like every deterministic task) no
    worktree/branch, so the bare lock can never be task-specific evidence for
    it.  A pending deterministic task carrying a ``before_done`` KEEPS the
    signal: that path runs a blocking deploy/predicate script while the status
    is still ``'pending'`` (``Harness._run_deterministic_slot`` never flips it
    to ``'in-progress'``) with no git evidence to reveal it.  Confirmed
    incident: task 3845 was listed here with ONLY the bare ``orchestrator``
    signal for 3+ consecutive reconciliation cycles, blocking its disposition.

    **In-progress corroboration gate (task 2963).** For an ``in-progress`` task
    whose only live signals are a lingering registered worktree and/or the
    project-wide orchestrator lock (no ``recent_commit``), a fleet redeploy that
    KILLED the workflow leaves both those signals falsely asserting liveness.
    This renderer therefore computes an explicit per-task corroboration verdict
    (:func:`corroboration_for_task`) for every in-progress task and passes it to
    the detector as ``corroborated``.  Corroboration requires at least one FRESH
    per-task signal, ANY sufficient: (1) a live claimant/heartbeat, (2) the
    task_id present in the scheduler's ``current_holders``/``parks`` snapshot, or
    (3) a ``routing.latest.decided_at`` newer than the orchestrator's start time
    (parsed from the lock).  When none corroborates, the detector downgrades the
    task to ``indeterminate`` (``is_live=False``) and the
    ``if not liveness.is_live: continue`` below drops it from the section — so a
    stranded post-redeploy task is no longer reported live, unblocking recon's
    stranded-remediation path.  The scheduler-state snapshot and the
    orchestrator start-time are hoisted once per render (like the orchestrator
    hoist); both are fail-safe → ``None``.  Non-in-progress tasks pass
    ``corroborated=None`` so the gate stays inert (behavior unchanged).

    Args:
        tasks: Task dicts from the active/proactive-sample pool.  Only tasks
            with a parseable ``id`` are inspected (non-int ids are skipped).
        project_root: Absolute path to the project root, forwarded to the
            detector and used to read the orchestrator lock + scheduler-state
            snapshot for the in-progress corroboration gate.
        now: Injectable reference time for deterministic tests.  Also the
            reference used for the claimant-heartbeat freshness check in the
            in-progress corroboration gate.

    Returns:
        A Markdown section string (e.g. ``'### Live-Workflow Signals\\n...\\n'``),
        or ``''`` when no tasks are live.
    """
    if not tasks:
        return ''

    # Hoist the project-level orchestrator check: it is constant for this
    # project_root (one lock file regardless of how many tasks are inspected).
    # Swallow any detector errors here — the per-task detect_live_workflow calls
    # will gracefully degrade on subsequent orchestrator checks.
    try:
        project_orch_live: bool | None = is_orchestrator_live_for(project_root)
    except Exception:
        project_orch_live = None  # let detect_live_workflow derive it per-task

    kwargs: dict = {} if now is None else {'now': now}
    if project_orch_live is not None:
        kwargs['_orchestrator_live'] = project_orch_live

    # Hoist the per-render corroboration inputs (task 2963), mirroring the
    # orchestrator hoist above: both the scheduler-state snapshot and the
    # orchestrator restart-boundary timestamp are constant for this
    # project_root, so read each once. Both are wrapped fail-safe → None on any
    # error (corroboration_for_task tolerates None inputs; a None simply means
    # that corroboration signal cannot fire — never a raise). now_eff is the
    # reference time threaded into the claimant-freshness check.
    now_eff = now or datetime.now(UTC)
    try:
        scheduler_state: dict | None = read_scheduler_state(Path(project_root))
    except Exception:
        scheduler_state = None
    try:
        orch_started: datetime | None = orchestrator_started_at(project_root)
    except Exception:
        orch_started = None

    live_lines: list[str] = []

    for task in tasks:
        raw_id = task.get('id')
        if raw_id is None:
            continue
        task_id = str(raw_id)
        raw_metadata = task.get('metadata')
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        task_kind = metadata.get('task_kind')
        # `metadata` is already the isinstance-guarded dict above, so a task
        # with absent/non-dict metadata yields pure_gate=False — fail-safe
        # toward live. See the docstring's pending-pure-gate paragraph.
        pure_gate = is_pure_gate_metadata(metadata)

        # Per-task corroboration gate (task 2963). For an IN-PROGRESS task,
        # compute an explicit corroboration verdict so the detector can downgrade
        # a killed-but-lingering task — whose only live signals are a stale
        # registered worktree and/or the project-wide orchestrator lock (no
        # recent_commit) — to indeterminate (is_live=False), which the
        # `if not liveness.is_live: continue` below then drops from the section.
        # A fresh per-task signal (live claimant/heartbeat, scheduler
        # holder/park, or a post-restart routing decision) keeps corroborated
        # True and the task listed. Non-in-progress tasks pass corroborated=None
        # so the gate stays inert (behavior unchanged). Fail-safe TOWARD live:
        # any assembler error leaves corroborated=None.
        corroborated: bool | None = None
        if task.get('status') == 'in-progress':
            try:
                corroborated = corroboration_for_task(
                    task, task_id, now=now_eff,
                    scheduler_state=scheduler_state,
                    orchestrator_started_at=orch_started,
                )
            except Exception:
                corroborated = None

        try:
            liveness = detect_live_workflow(
                task_id, project_root,
                status=task.get('status'), task_kind=task_kind,
                pure_gate=pure_gate,
                corroborated=corroborated, **kwargs
            )
        except Exception:
            logger.warning(
                'reconciliation._render_live_workflow_section: '
                'detector error for task_id=%s; treating as not-live',
                task_id,
            )
            continue

        if not liveness.is_live:
            continue

        # Collect which signals fired for human-readable display.
        signals: list[str] = []
        if liveness.worktree_registered:
            signals.append('worktree')
        if liveness.recent_commit:
            signals.append('recent-commit')
        if liveness.orchestrator_live:
            signals.append('orchestrator')
        signal_str = ', '.join(signals) if signals else 'live'
        live_lines.append(f'- {liveness.branch}: {signal_str}')

    if not live_lines:
        return ''

    return '### Live-Workflow Signals\n' + '\n'.join(live_lines) + '\n'


class TaskKnowledgeSync(BaseStage):
    """Stage 2: Reconcile tasks against memory, attach hints, fix inconsistencies."""

    # Remediation support — set by harness for second pass
    remediation_mode: bool = False

    # Active task tree — set by harness before run() (task 455)
    filtered_task_tree: FilteredTaskTree | None = None

    # Minimum number of tasks to proactively spot-check each run
    MIN_TASK_SAMPLE: int = 5

    # Defensive render cap for the since-boundary Done-Task Completion-Memory
    # Audit section.  The audit is meant to cover EVERY done task since the last
    # cycle, but a pathological gap between cycles (or a first-cycle None
    # boundary that admits the full done history) could otherwise grow the prompt
    # unboundedly.  When the audit list exceeds this cap the section renders the
    # most-recent MAX_DONE_AUDIT_RENDERED and appends an explicit overflow note
    # (never a silent truncation) plus a WARNING log — per the project's
    # no-silent-caps principle.
    MAX_DONE_AUDIT_RENDERED: int = 60

    # Current reconciliation run_id — set by run() so assemble_payload can use
    # it for stale-flag persistence markers (FIX D).
    # Sentinel None means run() has not yet been called; assemble_payload raises
    # loudly in that case so test authors are reminded to set this attribute.
    _current_run_id: str | None = None

    # Start of the current run's window (journal started_at) — stashed by
    # assemble_payload() (task-2047 Gap 2) so run() can forward it, after
    # super().run() returns, to _write_task_count_snapshot's freshness check.
    # Mirrors the _current_run_id stash pattern. Reset to None at the top of
    # run() so a prior run's value can never leak into a cycle whose
    # assemble_payload call is skipped or short-circuited.
    _run_window_start: datetime | None = None

    # Count of Stage 2 markers with absent/empty metadata.run_id that were routed
    # to the stale_missing_run_id bucket in the current assemble_payload() call
    # (task 1257).  Non-zero indicates Stage 1 producer drift from a PRIOR cycle —
    # the LLM omitted the required run_id field.  Reset to 0 at the top of run()
    # and injected into report.stats after super().run() returns (same per-run
    # counter reset-then-inject pattern used throughout this class).
    # Note (task-1369): same-cycle markers with absent run_id whose Mem0 created_at
    # is within the current run window are rescued to partition.current by
    # _query_stage2_flags and are NOT counted here.
    _stale_missing_run_id_markers: int = 0

    # Count of Stage 2 markers rescued to partition.current by the run-window guard
    # in _query_stage2_flags (task 1369).  Non-zero indicates Stage 1 producer drift
    # within the CURRENT cycle — the LLM omitted or mis-stamped metadata.run_id on a
    # flag it wrote during this run.  The marker was still surfaced to the Stage 2 LLM
    # (not swept), but the drift is recorded here for operator observability.  Reset
    # to 0 at the top of run() and injected into report.stats after super().run()
    # returns (same per-run counter reset-then-inject pattern used throughout this
    # class).
    _rescued_in_window_markers: int = 0

    # Count of systemic_pattern findings polled from the recon_report channel
    # (task-1966) that were newly APPENDED to combined_flags in the current
    # assemble_payload() call — i.e. survived compute_flag_signature dedup
    # against the Mem0/Stage-1 channel.  A finding whose signature already
    # exists in combined_flags is deduped and does NOT increment this counter.
    # Reset to 0 at the top of run() and injected into
    # report.stats['stage2_recon_report_systemic_polled'] after super().run()
    # returns (same per-run counter reset-then-inject pattern used throughout
    # this class).
    _recon_report_systemic_polled: int = 0

    # Combined Stage 2 flags (Stage 1 items + surviving Mem0 active-query flags)
    # assembled by assemble_payload() during the current run (task-2029 scenario
    # b).  Read via getattr(self, '_stage2_combined_flags', []) by the post-run
    # guard flow to join report.stats['flag_deleted_records'] on flag_id and
    # recover (task_id, flag_type) for Channel-1 tagging.  Reset to [] at the
    # top of run() (amendment round 2, reviewer finding: robustness) so that a
    # run whose assembly is skipped/short-circuited can never leave a PRIOR
    # run's flags visible to the join — the getattr default only covers the
    # never-set case, not the stale-from-a-prior-run case.
    _stage2_combined_flags: list[dict] | None = None

    def get_system_prompt(self) -> str:
        return build_stage2_system_prompt(self.project_id)

    def get_disallowed_tools(self) -> list[str]:
        return STAGE2_DISALLOWED

    async def run(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
        run_id: str,
        model: str | None = None,
        resume_session_id: str | None = None,
    ) -> StageReport:
        """Capture run_id, run briefing-refresh hook, delegate to BaseStage.run(), then
        apply the same-run Stage 1 human_operator_required dedup post-processor.

        Post-processing: after ``super().run()`` returns, any Stage 2 items whose
        ``(task_id, flag_type, resolution_status='human_operator_required')`` 3-tuple
        matches a Stage 1 item flagged ``human_operator_required`` in the same run are
        dropped from ``report.items_flagged``.  An INFO log
        ``reconciliation.stage2_suppressed_stage1_dup_flags`` is emitted whenever
        suppressions fire, and ``report.stats['stage2_stage1_dups_suppressed']`` records
        the suppressed count so downstream consumers can reconcile the count against the
        LLM's own ``flagged_count``.  The post-processor is a no-op when
        ``prior_reports`` is empty, ``prior_reports[0].stage`` is not
        ``memory_consolidator``, or Stage 1's ``items_flagged`` is empty.
        """
        self._current_run_id = run_id
        # Reset per-run counters so cross-invocation contamination is impossible
        # (mirrors _current_run_id overwrite pattern).
        self._stale_missing_run_id_markers = 0
        self._rescued_in_window_markers = 0
        self._recon_report_systemic_polled = 0
        # Reset BEFORE super().run() (task-2047 Gap 2) so a run whose
        # assemble_payload call is skipped or short-circuited can never forward
        # a PRIOR run's run_window_start into this cycle's sweep — falls back to
        # the age-only sweep instead (backward compatible).
        self._run_window_start = None
        # Amendment round 2 (reviewer finding: robustness): reset BEFORE
        # super().run() so a run whose assemble_payload call is skipped or
        # short-circuited can never leave a PRIOR run's combined_flags visible
        # to the post-flight guard's flag_deleted_records join.
        self._stage2_combined_flags = []
        await self._maybe_queue_briefing_refresh_tasks(run_id=run_id)
        report = await super().run(
            events, watermark, prior_reports, run_id, model=model,
            resume_session_id=resume_session_id,
        )

        # --- missing-run_id marker stat (task 1257) ---
        # Explicit zero is required so downstream consumers never need
        # .get(..., 0) fallbacks.
        report.stats['stale_missing_run_id_markers'] = self._stale_missing_run_id_markers
        # --- rescued-in-window marker stat (task-1369 amendment) ---
        # Non-zero when the run-window guard rescued same-cycle Stage-1 markers
        # whose run_id was omitted/mis-stamped.  They reached Stage 2 fine (not
        # swept), but the count is observable here for operator diagnostics.
        # Explicit zero so downstream consumers never need .get(..., 0) fallbacks.
        report.stats['rescued_in_window_markers'] = self._rescued_in_window_markers
        # --- recon_report systemic_pattern channel poll stat (task-1966) ---
        # Explicit zero so downstream consumers never need .get(..., 0) fallbacks.
        report.stats['stage2_recon_report_systemic_polled'] = self._recon_report_systemic_polled

        # --- same-run Stage 1 human_operator_required dedup (task 1154) ---
        # Guard on stage identity so a future reorder of prior_reports doesn't
        # accidentally dedup against the wrong stage (suggestion 3).
        if (
            prior_reports
            and prior_reports[0].stage == StageId.memory_consolidator
            and prior_reports[0].items_flagged
        ):
            kept, suppressed = _suppress_same_run_human_operator_dups(
                report.items_flagged,
                prior_reports[0].items_flagged,
            )
            if suppressed:
                logger.info(
                    'reconciliation.stage2_suppressed_stage1_dup_flags',
                    extra={
                        'run_id': run_id,
                        'project_id': self.project_id,
                        'suppressed_count': len(suppressed),
                    },
                )
                report.items_flagged = kept
                # Record the suppressed count in stats so downstream consumers
                # (Stage 3 prompt, observability) can reconcile items_flagged
                # length against the LLM's own flagged_count (suggestion 2).
                report.stats['stage2_stage1_dups_suppressed'] = len(suppressed)

        # --- post-flight guards (task 1137) ---
        await self._apply_post_flight_guards(report, prior_reports, run_id)

        # --- deterministic per-cycle summary write (task 2229 W5-λ) ---
        # Python writes the authoritative cycle_summary ledger row directly
        # from `report` — no LLM turn, no nonce, no verify/repair/reconstruct
        # self-heal (all retired by this task; see
        # summary_pool.write_cycle_summary).
        # Cross-reference (reviewer finding design-consistency): unlike Stage 1
        # (memory_consolidator.py), this call is unconditional — it also fires
        # on remediation passes, not just full cycles. That is intentional,
        # not a missed guard: it preserves the pre-refactor contract, where
        # Stage 2's prompt mandated a summary write on every pass, whereas
        # Stage 1's remediation payload never asked for one. Do not "fix" this
        # to mirror Stage 1's full-cycle-only gating.
        #
        # Data-fidelity note, run_id claim corrected (task 2995): the ledger
        # upsert's primary key is (project_id, 'cycle_summary', flag_type=stage,
        # run_id). Each remediation pass runs under its OWN fresh run_id — a
        # new uuid4() minted per pass in harness.py's _maybe_remediate (see
        # `run_id = str(uuid4())` there) — NOT the parent full cycle's run_id.
        # So in normal operation a remediation pass's Stage 2 write does not
        # collide with, and cannot overwrite, the parent full cycle's ledger
        # row: it upserts a brand-new row keyed on its own distinct run_id.
        # test_remediation_pass_overwrites_full_cycle_ledger_payload
        # (tests/test_stages.py) exercises write_cycle_summary's "last write
        # wins" upsert semantics for two calls that are deliberately made to
        # share one run_id (see ReconLedgerStore.upsert's own docstring for
        # the general semantics) — it documents what the upsert does WHEN a
        # run_id collides, not a claim that a remediation pass and its parent
        # cycle actually share a run_id in production; they don't.
        # remediation=self.remediation_mode (task 2652): lets Stage 3
        # distinguish a remediation pass's expected missing Stage 1
        # (memory_consolidator) cycle_summary — Stage 1 still runs a focused
        # turn and may emit findings on such a pass, but early-returns before
        # reaching its own summary write, by design — from a genuine Stage 1
        # write failure on a full cycle. See prompts/stage3.py's Remediation
        # Run Exception.
        ledger_written = await write_cycle_summary(
            self.memory,
            self.project_id,
            report,
            run_id,
            stage='task_knowledge_sync',
            recon_pool=_STAGE2_CYCLE_SUMMARY_RECON_POOL,
            trim_source=_STAGE2_CYCLE_SUMMARY_TRIM_SOURCE,
            cap=STAGE2_CYCLE_SUMMARY_POOL_CAP,
            remediation=self.remediation_mode,
        )
        # Named "..._ledger_written", not "..._written" (reviewer finding
        # observability, task 2229 amendment pass round 2): this reflects
        # ONLY the authoritative ReconLedgerStore upsert. write_cycle_summary
        # also attempts a best-effort Mem0 mirror write regardless of the
        # ledger outcome (see its docstring), so a deployment running with
        # recon_ledger_enabled=False can have this stat at 0 while the
        # mirror was in fact written — the "_ledger_" qualifier makes that
        # distinction explicit instead of implying "no summary at all".
        report.stats['stage2_cycle_summary_ledger_written'] = 1 if ledger_written else 0

        # --- task_count_snapshot deterministic write + freshness stat (task 2325,
        # follow-up to task 2278) ---
        # Makes the Mem0 task_count_snapshot write structural rather than
        # depending on the Stage-2 LLM remembering the memory-stored "Snapshot
        # Discipline" norm: for a non-blocked project we WRITE the snapshot
        # ourselves (deterministic Python, no LLM involved) and trust that
        # write directly rather than re-querying Mem0 for it — Qdrant's scroll
        # is point-id-ordered, so an immediate re-read can false-miss a write
        # we just performed. Blocked projects (SNAPSHOT_WRITE_BLOCKED_PROJECTS
        # — the per-project census is not in use there, so absence is
        # correct-by-design) and any write failure fall back to the
        # pre-existing best-effort freshness read below.
        # run_window_start forwards whatever assemble_payload stashed on
        # self._run_window_start this cycle.
        # written is None when blocked+unverified, the run window is unknown,
        # or the freshness query itself failed transiently, in which case the
        # stat key is left absent entirely so the harness
        # (_maybe_escalate_stale_task_count_snapshot) can distinguish a
        # CONFIRMED miss (0) from "unknown" (never miscounted as a miss).
        #
        # Duplicate-write note (reviewer finding, amendment round): a
        # memory-stored Stage-2 norm ("the corrected 2026-07-01 norm",
        # dark_factory procedural_knowledge) separately tells the Stage-2 LLM
        # to author this same kind='task_count_snapshot' observation as its
        # own final action each cycle. This task does not retire that norm
        # in-repo (it is runtime Mem0 content, not source under this task's
        # locks), so for a non-blocked project the LLM path and this
        # deterministic path can both fire in the same cycle. That is
        # tolerated, not silently assumed away: (a) task_count_snapshot has
        # no pool cap/GC (unlike e.g. stage1_flag_marker), so an extra record
        # is inert rather than a leak; (b) neither this stat nor the
        # harness's evaluate_snapshot_cadence()/build_stale_snapshot_finding()
        # (task_count_snapshot_cadence.py) look at record *count* — only
        # presence-within-window (verify path, skipped here on a successful
        # write) or the directly-trusted True below — so a duplicate cannot
        # change either outcome; (c) historically the LLM has fired only
        # sporadically (a handful of snapshots recorded over several weeks of
        # reify reconciliation cycles), so same-cycle collisions are expected
        # to stay rare even though the norm itself has not been retired.
        run_window_start = getattr(self, '_run_window_start', None)
        task_count_snapshot_written: bool | None = None
        if not is_snapshot_write_blocked(self.project_id):
            task_count_snapshot_written = await _write_task_count_snapshot(
                self.memory, self.taskmaster, self.project_root, self.project_id,
                run_id, run_window_start, stats=report.stats,
            )
        # Task 2655 step-6: if the prune couldn't enumerate (enumeration_ok
        # == 0 -- the swallowed-Qdrant-timeout fingerprint), _write_task_
        # count_snapshot already skipped its write above (step-4) rather than
        # risk another duplicate. Don't fall back to verify in that case
        # either: _verify_task_count_snapshot_written's scroll swallows
        # timeouts the same way and would mis-read the empty page as a
        # CONFIRMED miss (False), spuriously growing the harness's
        # consecutive-stale-snapshot streak. Leave the stat key absent
        # (inconclusive) instead. When the key is absent entirely (taskmaster
        # None, or write-blocked project -- the prune never ran) or
        # enumeration_ok == 1, behavior is unchanged: verify still runs.
        #
        # Accepted trade-off (reviewer finding, amendment round): a
        # SUSTAINED enumeration failure (e.g. a Qdrant read timeout that
        # persists across cycles) is now invisible to the harness's
        # consecutive-stale-snapshot escalation -- every affected cycle skips
        # both the write and the verify fallback, so 'task_count_snapshot_
        # written' stays absent (never a CONFIRMED miss) and the streak never
        # advances. The only signal is the per-cycle WARNING logged in
        # _prune_task_count_snapshots / _write_task_count_snapshot. Before
        # this change a sustained timeout still produced a fresh (duplicated)
        # snapshot each cycle, so freshness was maintained at the cost of the
        # duplicate pile this task exists to stop. A bounded backstop --
        # escalate once consecutive enumeration_ok == 0 cycles cross a
        # threshold, mirroring the harness's existing consecutive-miss streak
        # -- would close this blind spot; it needs a harness-side consumer of
        # the prune stats and is left as a possible follow-up rather than
        # folded into this task.
        if (
            task_count_snapshot_written is None
            and report.stats.get(SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY) != 0
        ):
            task_count_snapshot_written = await _verify_task_count_snapshot_written(
                self.memory, self.project_id, run_window_start,
            )
        if task_count_snapshot_written is not None:
            report.stats[SNAPSHOT_WRITTEN_STAT_KEY] = (
                1 if task_count_snapshot_written else 0
            )

        # --- recon_ledger marker GC: single DELETE pass (task 2228 W5-κ) ---
        # Collapses what were two separate stage1_flag_marker sweeps —
        # age-based + cross-cycle fp: GC (task 1944) and terminal-task-status
        # GC (task 2103) — into one ReconLedgerStore.gc() call: its
        # expires_at < now clause replaces the age-based sweep, and its
        # record_kind/task_id membership clause replaces the terminal-task
        # sweep. Runs unconditionally on both full and remediation paths so
        # the pool is bounded every cycle. Explicit zero so downstream
        # consumers never need a .get(..., 0) fallback.
        report.stats['recon_markers_gc_swept'] = await _gc_recon_markers(
            self.memory, self.taskmaster, self.scope, run_id,
        )

        # stage2_persistence_marker (task 2095) is GC'd separately from Mem0,
        # NOT folded into the ledger gc() pass above: its writer
        # (_track_flag_persistence) still writes to — and counts from — Mem0,
        # and no code path upserts a stage2_persistence_marker row into the
        # recon_ledger, so ReconLedgerStore.gc() can never collect it
        # (review finding regression-orphan-growth; migrating the writer to
        # the ledger is deferred to a follow-up). Runs unconditionally every
        # cycle alongside the ledger gc() pass; explicit zero for the same
        # reason as above.
        report.stats['stale_persistence_markers_gc_swept'] = await _sweep_stale_persistence_markers(
            self.memory, self.project_id, run_id,
        )

        # Legacy Mem0 stage1_flag_marker pool (task 2853): task 2406 retired
        # its mirror write (markers now persist only to the recon_ledger,
        # reaped above by _gc_recon_markers), and task 2228 W5-κ deleted the
        # prior in-cycle Mem0 sweep for this source on the assumption that
        # the ledger gc() pass replaced it — it does not reach Mem0, so the
        # pre-2406 pool was left uncollected for every project (the
        # operational sweep_orphan_flag_markers.py systemd timer only
        # targets dark_factory by default). Runs unconditionally every cycle,
        # per-project, so each project self-heals its own legacy pool;
        # explicit zero for the same reason as the two GC stats above.
        report.stats['stale_mem0_flag_markers_gc_swept'] = await _sweep_stale_mem0_flag_markers(
            self.memory, self.project_id, run_id,
        )

        # Mem0-only flag_for_stage2 relay pool (task 2966): its only writer is
        # the Stage-1 flag_dedup/LLM add_memory path, which writes solely to
        # Mem0 (metadata.flag_for_stage2=true) — no code path upserts a
        # flag_for_stage2 row into the recon_ledger, so ReconLedgerStore.gc()
        # structurally cannot reach this pool (the same declared-vs-actual
        # gap as stage2_persistence_marker above). Runs unconditionally every
        # cycle, per-project, alongside the three sibling GC passes; explicit
        # value so downstream consumers never need a .get(..., 0) fallback.
        report.stats['stale_mem0_flag_for_stage2_markers_gc_swept'] = (
            await _sweep_stale_mem0_flag_for_stage2_markers(
                self.memory, self.project_id, run_id,
            )
        )

        # Entity-standing-decision growth-freshness sweep (task 2899 ζ) — a
        # sibling of the marker sweeps above, run at the Stage-2 TAIL (Stage 3 is
        # read-only by design; PRD decision 9). Corroborates each ACTIVE standing
        # decision's decision-time edge-count snapshot against the live graph and
        # expires grown rows (reason='growth'); records explicit-zero stats and,
        # on a full non-remediation cycle, evaluates the failure-streak escalation.
        await self._run_entity_standing_decision_growth_sweep(report, run_id)

        return report

    async def _run_entity_standing_decision_growth_sweep(
        self,
        report: StageReport,
        run_id: str,
    ) -> None:
        """Run the ζ growth-freshness sweep and record its per-cycle stats (task 2899).

        Delegates the row work to :func:`_sweep_entity_standing_decision_growth`
        (best-effort, never raises), then records explicit-zero stats — the
        ``failed`` flag (``1``/``0``) that the streak escalation reads next cycle
        and the count of rows flipped to expired/growth — so downstream consumers
        never need a ``.get(..., default)`` fallback.

        The consecutive-failure streak escalation is evaluated ONLY on a full,
        non-remediation cycle with a wired escalation queue: a remediation/targeted
        pass must not inflate or reset the streak (mirroring the task-count-snapshot
        cadence), and an unwired queue has nothing to file to. The filer itself is
        journal-recompute + ``has_open_l1``-deduped and never raises.
        """
        result = await _sweep_entity_standing_decision_growth(
            self.memory, self.project_id, run_id,
        )
        report.stats[ENTITY_STANDING_DECISION_GROWTH_SWEEP_FAILED_STAT_KEY] = (
            1 if result['failed'] else 0
        )
        report.stats[ENTITY_STANDING_DECISION_GROWTH_EXPIRED_STAT_KEY] = result['expired']

        if not self.remediation_mode and self._escalation_queue is not None:
            await _maybe_escalate_growth_sweep_failures(
                self._escalation_queue,
                self.journal,
                self.project_id,
                run_id,
                result['failed'],
            )

    async def _apply_post_flight_guards(
        self,
        report: StageReport,
        prior_reports: list[StageReport],
        run_id: str,
    ) -> None:
        """Acknowledge resolved Stage 1 flag markers and normalize Stage 2 counters.

        ``report.stats['stage1_analytical_findings_processed']`` and
        ``report.stats['stage1_mem0_flags_processed']`` are Stage 2's
        self-reported flag-processing counters. They are no longer clamped
        against ground truth here: that reconciliation now happens via
        ``derive_stage_stats`` + ``stats_verifier`` (task 2229), which
        recompute canonical counters from the write journal.

        Terminal-state, stall-guard-freshness, post-action-mismatch, and
        live-workflow writes are also no longer reclassified here: that write
        class is now rejected pre-write, server-side, by ``ReconWritePolicy``
        (task 2224), so post-hoc detection is redundant.

        Args:
            report: The ``StageReport`` returned by ``super().run()``.
                Mutated in place.
            prior_reports: Stage reports for all earlier stages in this cycle.
                Unused now that the analytical flag-counter clamp has been
                removed; kept for signature compatibility with the sole call
                site.
            run_id: Current reconciliation run identifier.
        """
        # ── Stage-1 flag-marker acknowledgment (task-2029 scenario b) ──────────
        # flag_deleted_records names the flags Stage 2 resolved via FIX C
        # deletion this run.  Join those against the
        # combined_flags rendered by assemble_payload (stashed on self) to
        # recover each resolved flag's (task_id, flag_type), then tag the
        # originating Channel-1 stage1_flag_marker addressed_by=run_id so it
        # no longer needs manual per-cycle disambiguation.  Additive and
        # best-effort — never raises, never touches Channel-2 sweeps.
        report.stats['stage2_flag_markers_acknowledged'] = await _acknowledge_resolved_stage1_markers(
            self.memory,
            self.project_id,
            run_id,
            report.stats.get('flag_deleted_records', []),
            getattr(self, '_stage2_combined_flags', []) or [],
        )

        # Normalize: ensure both Stage 2 counters are always present in stats so
        # Stage 3's audit sees a deterministic, present pair via recon_report's
        # free-form stats passthrough.  setdefault preserves any agent-reported
        # value.
        report.stats.setdefault('stage1_analytical_findings_processed', 0)
        report.stats.setdefault('stage1_mem0_flags_processed', 0)

    async def _maybe_queue_briefing_refresh_tasks(self, run_id: str = '') -> None:
        """Best-effort: queue 'Refresh briefing' tasks for each briefing-known-gaps mismatch.

        Silently skips if taskmaster is absent, or if project_id is not in
        ``_BRIEFING_REFRESH_PROJECT_ALLOWLIST`` (reify-specific feature). Any
        exception is caught and logged as a WARNING so a broken script can
        never abort Stage 2.
        """
        if not self.taskmaster:
            return
        if self.project_id not in _BRIEFING_REFRESH_PROJECT_ALLOWLIST:
            return
        try:
            mismatches = await _run_briefing_known_gaps_script(self.scope.project_root)
            if not mismatches:
                return
            # Avoid a redundant get_tasks round-trip when the harness has
            # already injected the full task tree into self.filtered_task_tree.
            existing_tasks: list[dict] | None = None
            if self.filtered_task_tree is not None:
                existing_tasks = list(
                    itertools.chain(
                        self.filtered_task_tree.active_tasks,
                        self.filtered_task_tree.done_tasks,
                        self.filtered_task_tree.cancelled_tasks,
                    )
                )
            summary = await _queue_briefing_refresh_tasks(
                self.taskmaster,
                self.scope.project_root,
                mismatches,
                existing_tasks=existing_tasks,
                run_id=run_id,
            )
            # Extract values inside the try/except so a contract violation
            # (e.g. summary being None due to a future refactor) is caught
            # here rather than propagating out of this method.
            created_ids = summary.get('created', [])
            skipped_ids = summary.get('skipped', [])
            failed_ids = summary.get('failed', [])
        except Exception:
            logger.warning(
                'briefing_refresh_hook_failed',
                exc_info=True,
                extra={'project_root': self.project_root},
            )
            return
        # Logging is intentionally outside the try/except above so a logging
        # bug (e.g. a reserved-name collision in `extra`) surfaces as a real
        # error rather than being swallowed as a misleading
        # 'briefing_refresh_hook_failed' WARNING. Note: 'created' is a
        # reserved LogRecord attribute (the timestamp), so we use
        # 'created_ids'/'skipped_ids'/'failed_ids' here.
        logger.info(
            'briefing_refresh_tasks_queued',
            extra={
                'project_root': self.project_root,
                'created_ids': created_ids,
                'skipped_ids': skipped_ids,
                'failed_ids': failed_ids,
            },
        )

    async def assemble_payload(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
    ) -> str:
        # Guard: reject calls without a run_id before any I/O.  With an empty
        # run_id, _query_stage2_flags would classify ALL existing Mem0 markers
        # as stale (the partition treats an empty-string run_id as absent — see
        # its docstring), so every marker would be excluded from what's shown
        # to the Stage 2 LLM even though no flags are actually stale this
        # cycle.  Raising here short-circuits before any filter_task_tree /
        # Mem0 / Taskmaster I/O, ensuring _track_flag_persistence and
        # _write_escalation_markers never write records stamped with an empty
        # run_id.
        if not self._current_run_id:
            raise RuntimeError(
                'TaskKnowledgeSync.assemble_payload() called without a run_id: '
                '_current_run_id is not set.  In production this is set '
                'automatically by run().  In tests, assign '
                'stage._current_run_id = "test-run" before calling '
                'assemble_payload() directly.'
            )

        stage1_report = prior_reports[0] if prior_reports else None

        # Dual-path: use harness-injected tree or self-fetch (task 455)
        if self.filtered_task_tree is not None:
            # Harness path: tree already fetched and filtered before run()
            filtered = self.filtered_task_tree
        else:
            # Fallback path: self-fetch via taskmaster
            # filter_task_tree accepts `object`; either GetTasksResult or {} is fine.
            tasks_data: object = {}
            if self.taskmaster:
                try:
                    tasks_data = await self.taskmaster.get_tasks(project_root=self.project_root)
                except Exception:
                    tasks_data = {}
            filtered = filter_task_tree(tasks_data)

        # Defensive invariant check (task-782): see _check_filtered_tree_invariant.
        self._check_filtered_tree_invariant(filtered)

        # Render "Recently Completed Tasks" section.
        # Invariant: filter_task_tree() always appends to done_tasks when it increments
        # done_count (capped at MAX_DONE_TASKS_RETAINED=30), so done_tasks is guaranteed
        # non-empty whenever done_count > 0.  No fallback summary branch is needed.
        # _check_filtered_tree_invariant() warns for externally-constructed trees that
        # violate this.
        if filtered.done_tasks:
            recently_completed_text = format_task_list(filtered.done_tasks)
        else:
            recently_completed_text = format_task_list([])  # 'No tasks.'

        # Done-task provenance section — feeds verified evidence to the agent
        # so 'shipped via X' edges come from commit diffs instead of being
        # fabricated from metadata.modules. Empty string when no done tasks
        # carry done_provenance (legacy tree, warn-only rollout).
        provenance_section = await _render_done_provenance_section(
            filtered.done_tasks,
            self.scope.project_root,
        )

        remediation_note = ''
        if self.remediation_mode:
            remediation_note = (
                '### Remediation Mode\n'
                'This is a focused remediation run. Address remaining task-level issues '
                'from Stage 1. Do not perform general task-knowledge sync.\n\n'
            )

        proactive_sample_section = ''
        if not self.remediation_mode:
            # Pool intentionally excludes unknown-status tasks (dropped by
            # filter_task_tree) and caps done_tasks at MAX_DONE_TASKS_RETAINED.
            sample = _select_proactive_sample(
                itertools.chain(
                    filtered.active_tasks, filtered.done_tasks, filtered.cancelled_tasks
                ),
                self.MIN_TASK_SAMPLE,
            )
            proactive_sample_section = (
                f'\n### Proactive Task Sample ({len(sample)} tasks)\n{format_task_list(sample)}\n'
            )

        # Done-Task Completion-Memory Audit — enumerate EVERY task that
        # transitioned to `done` since the prior successful cycle boundary
        # (watermark.last_full_run_completed) via an updatedAt-window scan over
        # the UNCAPPED filtered.all_done_tasks.  This supersedes reliance on the
        # 5-item Proactive Task Sample for done-task completion-memory coverage,
        # which systematically missed done tasks as throughput grew (the sample
        # sorts done tasks last, so they rarely survive the top-5 cut).  Gated on
        # `not self.remediation_mode` — mirrors proactive_sample_section
        # (remediation is a focused second pass, not general sync).
        done_audit_section = ''
        if not self.remediation_mode:
            boundary = watermark.last_full_run_completed
            audit_tasks = select_done_since_boundary(filtered.all_done_tasks, boundary)
            total_audit = len(audit_tasks)
            boundary_label = (
                boundary.isoformat()
                if boundary is not None
                else 'no prior full-run boundary (first cycle — all done tasks in scope)'
            )
            # Defensive render cap: never a silent truncation.  select_done_since_boundary
            # sorts most-recent-first (and parse-failures to the front), so a clip
            # drops only the oldest tasks, and the note + WARNING log make the
            # clipped coverage explicit (no-silent-caps principle).
            rendered_audit = audit_tasks
            overflow_note = ''
            if total_audit > self.MAX_DONE_AUDIT_RENDERED:
                rendered_audit = audit_tasks[: self.MAX_DONE_AUDIT_RENDERED]
                omitted = total_audit - self.MAX_DONE_AUDIT_RENDERED
                overflow_note = (
                    f'\n_NOTE: {omitted} additional done task(s) since the boundary were '
                    f'omitted from this render by the MAX_DONE_AUDIT_RENDERED='
                    f'{self.MAX_DONE_AUDIT_RENDERED} cap. Coverage was clipped — NOT complete '
                    f'this cycle; the oldest since-boundary tasks were dropped first._'
                )
                logger.warning(
                    'reconciliation.done_task_audit_render_capped',
                    extra={
                        'project_id': self.project_id,
                        'run_id': self._current_run_id,
                        'total_since_boundary': total_audit,
                        'rendered': self.MAX_DONE_AUDIT_RENDERED,
                        'omitted': omitted,
                        'boundary': boundary_label,
                    },
                )
            done_audit_section = (
                f'\n### Done-Task Completion-Memory Audit '
                f'({total_audit} since last cycle)\n'
                f'Boundary (last full-run completed): {boundary_label}\n'
                f'{format_task_list(rendered_audit)}\n'
                f'{overflow_note}'
            )

        # Live-Workflow Signals section: check active tasks for live workflows so the
        # Stage 2 LLM can skip set_task_status / stranded-work escalation for those tasks.
        # Only active tasks are inspected (done/cancelled tasks cannot have live workflows).
        # Empty string when no active tasks are live (keeps the payload tight).
        live_workflow_section = ''
        if filtered.active_tasks:
            live_workflow_section = _render_live_workflow_section(
                filtered.active_tasks,
                self.scope.project_root,
            )

        # Call render_active_section once to get both the visible-task list (for
        # hint-attention slice-then-filter below) and the fully assembled Active
        # Task Tree string (for the prompt template slot) — single rendering pass.
        # Both the max_tasks=50 slice cap and the secondary max_chars=50_000 clamp
        # are applied once here, so the hint section and the tree slot always
        # reference the same set of tasks.
        visible_active, active_tree_text = render_active_section(filtered)

        # Compute the hint-attention section: active tasks whose memory_hints
        # need conversion from legacy list format (or are missing entirely).
        # Gated on `if not self.remediation_mode:` — mirrors proactive_sample_section
        # (above) because hint conversion is a general-sync activity, not a Stage 1
        # remediation task. Rendered conditionally within that gate — omitted on the
        # steady-state case where every active task already has valid dict-format hints.
        # Note: visible_active is only consumed inside the gate; active_tree_text is
        # always used for the prompt template slot regardless of remediation_mode.
        hint_conversion_section = ''
        if not self.remediation_mode:
            tasks_needing_hint_attention = [
                t for t in visible_active if _needs_hint_conversion(t)
            ]
            if tasks_needing_hint_attention:
                hint_conversion_section = (
                    '\n### Tasks Needing Memory Hint Attention\n'
                    + format_task_list(tasks_needing_hint_attention)
                    + '\n'
                )

        # FIX A — merge Mem0 active-query flags into the flagged section.
        # _query_stage2_flags is best-effort: search failures yield ([], [], [], []) internally.
        # Returns a Stage2FlagPartition with four fields:
        #   .current          — markers whose run_id matches the current run, OR whose
        #                       Mem0 created_at is within the run window (task-1369 guard)
        #   .stale_missing_run_id_ids  — prior-cycle markers with absent/empty run_id
        #   .stale_mismatched_run_id_ids — prior-cycle markers with wrong run_id
        #   .rescued_ids      — ids of markers rescued by the run-window guard (subset of
        #                       .current); single source of truth for the rescued count
        #                       (task-1381, replaces the re-derivation over active_flags)
        # Both stale buckets contain only genuine prior-cycle residue (in-window same-
        # cycle markers are rescued to .current by the run-window guard) and are simply
        # excluded from `current` below, so they are never rendered to the LLM. They are
        # NOT deleted here — that residue is reaped separately by the recon_ledger's
        # single GC pass (_gc_recon_markers, called from run()) on its TTL/terminal-task
        # schedule (task 2228 W5-κ), not by an immediate delete inside this method.
        #
        # Run-window guard (task-1369): fetch the run's started_at from the journal so
        # same-cycle Stage-1 markers whose run_id was omitted/mis-stamped by the LLM
        # producer are rescued (routed to current) rather than swept.  Best-effort:
        # any failure leaves run_window_start=None (window guard dormant this cycle).
        run_id_for_markers = self._current_run_id
        run_window_start: datetime | None = None
        try:
            _run = await self.journal.get_run(self._current_run_id)
            _sa = getattr(_run, 'started_at', None)
            if isinstance(_sa, datetime):
                if _sa.tzinfo is None:
                    logger.warning(
                        'reconciliation.assemble_payload: journal.get_run().started_at is a '
                        'naive datetime (tzinfo=None) — journal contract requires a tz-aware '
                        'UTC datetime; normalizing to UTC here and in _marker_is_within_run_window '
                        'as defence-in-depth so the run-window sweep guard remains active',
                        extra={'project_id': self.project_id, 'run_id': self._current_run_id},
                    )
                run_window_start = _assume_utc(_sa)
        except Exception:
            logger.warning(
                'reconciliation.assemble_payload: journal.get_run failed; '
                'run-window sweep guard disabled this cycle',
                extra={'project_id': self.project_id, 'run_id': self._current_run_id},
            )
        # Stash on the instance (task-2047 Gap 2) so run() can forward this cycle's
        # run_window_start into the task_count_snapshot write/verify helpers
        # (_write_task_count_snapshot / _verify_task_count_snapshot_written) after
        # super().run() returns — mirrors the _current_run_id stash pattern. Stays
        # None (set at the top of run()) when the journal lookup above failed or
        # returned a non-datetime started_at, which disables the run-window guard
        # and falls those helpers back to their window-agnostic behaviour.
        self._run_window_start = run_window_start
        partition = await _query_stage2_flags(
            self.memory, self.project_id, run_id_for_markers,
            run_window_start=run_window_start,
        )
        active_flags = partition.current
        self._stale_missing_run_id_markers = len(partition.stale_missing_run_id_ids)
        # Single source of truth for the rescued count: read directly from
        # partition.rescued_ids, which is populated exclusively by the run-window
        # guard branches inside _query_stage2_flags (task-1381).  Avoids re-deriving
        # the partition predicate here, so any future change to the guard logic
        # automatically flows through to this counter without drift.
        self._rescued_in_window_markers = len(partition.rescued_ids)

        surviving = active_flags  # active-query path is pass-through (no scope filter)

        # FIX D — stale-flag persistence tracking.
        # Track how many cycles each surviving flag has survived without being
        # deleted.  Best-effort: _track_flag_persistence degrades gracefully.
        # Note (task 1256, updated task 2228 W5-κ): :func:`_query_stage2_flags`
        # partitions flags by metadata.run_id, routing prior-cycle residue to
        # partition.stale_all_ids (not active_flags) so it is never rendered
        # to the LLM; that residue is reaped separately by the recon_ledger's
        # single GC pass (:func:`_gc_recon_markers`, called from run()) on its
        # TTL/terminal-task schedule rather than an immediate same-cycle
        # delete.  FIX D therefore only fires on Stage 1 re-flags surviving
        # within the current run_id.
        surviving_ids = [f['id'] for f in surviving]

        persistence_counts = await _track_flag_persistence(
            self.memory,
            self.project_id,
            run_id_for_markers,
            surviving_ids,
        )
        stale_ids = _compute_stale_flags(persistence_counts)

        # Escalation dedup: skip flags we already escalated in a prior cycle so
        # operators don't get the same alarm every run when FIX C deletion fails
        # (reviewer note on FIX D, task 1139).  Only the *newly* stale flags
        # render in the section and get a fresh escalation marker.
        newly_stale_ids, already_escalated_ids = await _filter_already_escalated_flags(
            self.memory,
            self.project_id,
            stale_ids,
        )
        if already_escalated_ids:
            logger.info(
                'reconciliation.stale_flag_escalation_suppressed',
                extra={
                    'project_id': self.project_id,
                    'run_id': run_id_for_markers,
                    'suppressed_flag_ids': already_escalated_ids,
                    'reason': 'already escalated in a prior cycle',
                },
            )
        stale_ids = newly_stale_ids

        # Build the combined flags list: Stage 1 structured-output first, then
        # surviving Mem0 active-query results.  Normalise each Stage 1 item via
        # _inject_flag_id so that Stage 2's FIX C deletion can always find
        # flag_id regardless of which field name Stage 1 used for the Mem0 id.
        combined_flags: list[dict] = [
            _inject_flag_id(item)
            for item in (stage1_report.items_flagged if stage1_report else [])
        ]
        for f in surviving:
            combined_flags.append(
                {
                    '_source': 'mem0_active_query',
                    'flag_id': f['id'],
                    'task_id': f['task_id'],
                    # flag_type surfaced from the Mem0 marker's own metadata
                    # (task-1966 amendment, reviewer finding dedup_gap):
                    # compute_flag_signature() requires BOTH task_id and
                    # flag_type to produce a signature, so without this a
                    # surviving mem0_active_query flag never enters
                    # existing_signatures below and the recon_report_systemic
                    # poll could double-render a finding that also survived
                    # this channel.  Stage 1 markers are documented (stage1.py
                    # "## Stage 2 Flag Relay (FIX B)") to carry flag_type
                    # alongside task_id; falls back to None (no signature,
                    # same as before) when a legacy marker omits it.
                    'flag_type': f.get('metadata', {}).get('flag_type'),
                    'content': f['content'],
                }
            )

        # recon_report systemic_pattern channel (task-1966 scope item 2): an
        # independent poll that does NOT pass through filter_suppressed, so a
        # stage1_flag_suppression record can never hide a systemic_pattern
        # finding from Stage 2 through this path.  Deduped against
        # combined_flags (Stage 1 + Mem0 channels) by compute_flag_signature so
        # a finding that also survived the primary channel is not double-rendered.
        #
        # Content-fingerprint fallback (task-2078): compute_flag_signature
        # requires BOTH task_id and flag_type to produce a signature. A
        # systemic_pattern finding frequently has task_id=None (no single
        # owning task), and a legacy flag_for_stage2 marker can omit
        # flag_type — in either case the signature is None on at least one
        # side, existing_signatures misses the match, and the SAME finding
        # renders twice: once as mem0_active_query (carries flag_id, drives
        # FIX C deletion + stage1_mem0_flags_processed) and once as a
        # flag_id-less recon_report_systemic duplicate processed as a
        # separate no-op — causing the per-cycle narrative summary and the
        # structured stats block to disagree (incident: flag 2d2ad790).
        # existing_content_fps closes this gap using the normalized-content
        # fingerprint as an additional, additive join key.
        existing_signatures = {
            sig for flag in combined_flags
            if (sig := compute_flag_signature(flag)) is not None
        }
        existing_content_fps = {
            fp for flag in combined_flags
            if (fp := _flag_content_fingerprint(flag)) is not None
        }
        polled_systemic_findings = _query_recon_report_findings(
            self._recon_report_state, self._current_run_id
        )
        appended = 0
        for finding in polled_systemic_findings:
            sig = compute_flag_signature(finding)
            # Findings carry only 'description' (never 'content'), so passing
            # finding directly is equivalent to the description-only lookup:
            # _flag_content_fingerprint's content-or-description fallback
            # resolves it the same way either way.
            fp = _flag_content_fingerprint(finding)
            if (sig is not None and sig in existing_signatures) or (
                fp is not None and fp in existing_content_fps
            ):
                continue
            combined_flags.append(
                {
                    '_source': 'recon_report_systemic',
                    'task_id': finding.get('task_id'),
                    'flag_type': finding.get('flag_type'),
                    'finding_id': finding.get('finding_id'),
                    'content': finding.get('description'),
                    # Hook B (task 2897 δ): carry the standing-decision
                    # annotation through this re-projection so it renders into
                    # the Stage 2 payload (_format_flagged json.dumps verbatim);
                    # None when the finding was not adjudicated.
                    'standing_decision_id': finding.get('standing_decision_id'),
                }
            )
            if sig is not None:
                existing_signatures.add(sig)
            if fp is not None:
                existing_content_fps.add(fp)
            appended += 1
        self._recon_report_systemic_polled = appended

        flagged_text = _format_flagged(combined_flags, run_stage='stage2')

        # Stash for the post-run guard flow (task-2029 scenario b): run()'s
        # _apply_post_flight_guards joins report.stats['flag_deleted_records']
        # against this list on flag_id to recover (task_id, flag_type) for
        # each FIX-C-deleted flag, so the originating Channel-1
        # stage1_flag_marker can be tagged addressed_by=<run_id>.
        self._stage2_combined_flags = combined_flags

        # Emit per-flag warnings and build the stale-flag section.
        stale_section = ''
        if stale_ids:
            # Build a lookup map so we can surface content + task_id in the section.
            surviving_map = {f['id']: f for f in surviving}
            stale_entries = []
            for fid in stale_ids:
                cycle_count = persistence_counts[fid]
                logger.warning(
                    'reconciliation.stale_flag_escalated',
                    extra={
                        'project_id': self.project_id,
                        'run_id': run_id_for_markers,
                        'flag_id': fid,
                        'cycle_count': cycle_count,
                    },
                )
                flag_info = surviving_map.get(fid, {})
                stale_entries.append(
                    f'- flag_id={fid!r} task_id={flag_info.get("task_id", "")!r} '
                    f'cycle_count={cycle_count} '
                    f'content={flag_info.get("content", "")!r}'
                )
            stale_section = (
                '\n### Stale Flags Requiring Escalation\n'
                'These flags have persisted for ≥ 3 cycles without being deleted.\n'
                'For each flag below, call `mcp__escalation__escalate_blocker`, '
                'then call `mcp__fused-memory__delete_memory` on the same flag_id '
                '(store="mem0") so the escalation is terminal — see prompt for '
                'details — and increment `stats.stale_flags_escalated`.\n\n'
                + '\n'.join(stale_entries)
                + '\n'
            )
            # Persist a per-flag escalation marker so the next cycle's
            # _filter_already_escalated_flags suppresses re-escalation if FIX C
            # deletion fails.  Best-effort: marker writes never raise.
            await _write_escalation_markers(
                self.memory,
                self.project_id,
                run_id_for_markers,
                stale_ids,
            )

        known_projects_section = self._format_known_projects_section()

        # Step 5 in the Your Task block below ("read-modify-write +
        # metadata_mode='replace' for hint conversion") is grounded in Mem0
        # memory 0b0eeb8d (old-wins semantics for list-format hints under
        # append=True).  A bare append=False RMW is no longer sanctioned — the
        # task-2180 metadata-wipe guard in _resolve_metadata_mode now rejects it
        # — so the reshape writes the COMPLETE blob back under the explicit
        # metadata_mode='replace' co-signal instead.  The memory id is kept here
        # rather than in the prompt string so the LLM is not burdened with an
        # opaque reference it cannot look up, and the traceability survives
        # prompt rewording.
        return f"""## Stage 2: Task-Knowledge Sync
## Project: {self.project_id}

{remediation_note}### Stage 1 Report Summary
{_format_report(stage1_report)}

### Stage 1 Flagged Items (Task-Relevant)
{flagged_text}
{stale_section}{known_projects_section}
{active_tree_text}

### Recently Completed Tasks
{recently_completed_text}
{provenance_section}{proactive_sample_section}{done_audit_section}{hint_conversion_section}{live_workflow_section}

## Your Task
Reconcile task state against memory:
1. For completed tasks: verify knowledge was captured. If sparse, search for related memories \
to check context, then write appropriate memories.
2. For tasks whose assumptions were invalidated by Stage 1 findings: modify, re-scope, or \
delete tasks. Update dependent tasks.
3. For AI-generated tasks: cross-reference against knowledge graph for factual consistency.
4. Attach memory_hints to tasks that would benefit from knowledge context at execution time. \
Use entity references + semantic queries, NOT inline content.
5. For tasks listed in **Tasks Needing Memory Hint Attention**: reshape legacy list-format \
memory_hints via read-modify-write — call `get_task` to read the FULL current metadata, convert \
the hints to the canonical `{{entities, queries}}` dict shape and merge them into that metadata \
locally, then write the COMPLETE metadata blob back with `metadata_mode='replace'`. Do NOT use a \
bare `append=False` (the task-2180 metadata-wipe guard now rejects it), and do NOT rely on \
Stage 2's default `append=True` merge — it silently discards legacy list-format hints under \
old-wins semantics.
6. Proactively review the **Proactive Task Sample** regardless of Stage 1 findings: check \
in-progress tasks for completion knowledge to capture, blocked tasks for unblock conditions \
that may now be met, and done tasks for missing knowledge capture. **For each done task, \
call count_memories_by_metadata(project_id, {{'task_id': str(task_id), 'stage2_suppress': True}}) \
FIRST** — if count > 0, skip the task entirely (no search, no write, no finding).
7. Check if any knowledge implies new tasks should be created or existing tasks unblocked.
8. Hints on completed tasks are static — don't update them.
9. When you have completed your work, produce your final structured report as your response.

{_STAGE2_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}
Use project_root="{self.project_root}" for tasks scoped to this project.
For cross-project routing see "Known Projects" above.
"""

    def _format_known_projects_section(self) -> str:
        """Render the cross-project routing context for the Stage 2 LLM.

        Emits a ``### Known Projects`` markdown section listing every
        configured project_id and its project_root, marking the current
        one.  Returns the empty string when fewer than two projects are
        known — there is no "cross-project" dimension to surface in that
        case, and the section would only add noise.
        """
        known = self.known_projects
        if len(known) < 2:
            return ''
        # Stable ordering: current project first, then alphabetical.
        ordered = [(self.project_id, known[self.project_id])] if self.project_id in known else []
        for pid in sorted(p for p in known if p != self.project_id):
            ordered.append((pid, known[pid]))
        # Pad the project_id column to a consistent width for readability.
        width = max(len(pid) for pid, _ in ordered)
        lines = []
        for pid, root in ordered:
            marker = '  (current)' if pid == self.project_id else ''
            lines.append(f'- {pid:<{width}}  → {root}{marker}')
        return '\n### Known Projects (for cross-project routing)\n' + '\n'.join(lines) + '\n'

    @staticmethod
    def _warn_if_count_tasks_mismatch(
        count: int,
        tasks: list,
        count_label: str,
        tasks_label: str,
        section_label: str,
        task_ref: str,
    ) -> None:
        """Emit a WARNING when a count>0/tasks-empty invariant is violated.

        Extracted to avoid repeating the same guard pattern for each
        count↔tasks pair (done, cancelled, and any future additions).
        """
        if count > 0 and not tasks:
            logger.warning(
                'FilteredTaskTree invariant violation: %s=%d but %s is '
                'empty. Externally-constructed tree bypassed filter_task_tree() guarantee. '
                '%s section will render as empty. (%s defensive check)',
                count_label,
                count,
                tasks_label,
                section_label,
                task_ref,
            )

    @staticmethod
    def _check_filtered_tree_invariant(filtered: FilteredTaskTree) -> None:
        """Emit a WARNING for each violated done/cancelled count↔tasks invariant.

        filter_task_tree() always appends to done_tasks when it increments done_count
        (capped at MAX_DONE_TASKS_RETAINED=30), and always appends to cancelled_tasks
        when it increments cancelled_count (capped at MAX_CANCELLED_TASKS_RETAINED=15),
        so both invariants are impossible to violate via the normal code path.
        Externally-constructed FilteredTaskTree instances that bypass filter_task_tree()
        could violate either; these checks catch them at the callsite rather than silently
        dropping data from the "Recently Completed" or "Recently Cancelled" sections.
        """
        TaskKnowledgeSync._warn_if_count_tasks_mismatch(
            filtered.done_count,
            filtered.done_tasks,
            'done_count',
            'done_tasks',
            'Recently Completed',
            'task-782',
        )
        TaskKnowledgeSync._warn_if_count_tasks_mismatch(
            filtered.cancelled_count,
            filtered.cancelled_tasks,
            'cancelled_count',
            'cancelled_tasks',
            'Recently Cancelled',
            'task-828',
        )


class IntegrityCheck(BaseStage):
    """Stage 3: Read-only cross-system consistency verification."""

    # Active task tree — injected by harness before run() (mirrors Stage 1/2 wiring).
    # Used by record_task_dump_spot_check to spot-check the cached task dump source.
    filtered_task_tree: FilteredTaskTree | None = None

    def get_system_prompt(self) -> str:
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

        return STAGE3_SYSTEM_PROMPT

    def get_disallowed_tools(self) -> list[str]:
        return STAGE3_DISALLOWED

    def get_report_schema(self) -> dict:
        return STAGE3_REPORT_SCHEMA

    async def run(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
        run_id: str,
        model: str | None = None,
        resume_session_id: str | None = None,
    ) -> StageReport:
        """Execute Stage 3 and post-process with task-dump spot-check.

        Calls super().run() first (LLM agent + report extraction), then:
        1. Applies filter_blocked_snapshot_findings() to suppress false-positive
           task-count snapshot findings for projects with blocked-by-design write
           paths (e.g. autopilot_video).  Records
           report.stats['blocked_snapshot_findings_dropped'] = before - after.
        2. Applies filter_false_phantom_task_creation_flags() to drop
           phantom-tasks_created findings corroborated by a task that was
           legitimately created in a different known project via cross-project
           routing.  Records
           report.stats['phantom_task_creation_findings_dropped'] = before - after.
        3. Applies filter_contamination_ceiling_findings() to suppress
           false-positive missing_knowledge/memory_stale findings asserting a
           retired Stage-1 task-ID contamination-ceiling guardrail memory is
           missing or stale, for projects whose ceiling is retired-by-design
           (e.g. autopilot_video; task 2818/2826).  Records
           report.stats['contamination_ceiling_findings_dropped'] = before - after.
        4. Calls record_task_dump_spot_check() to record a non-destructive
           observability stat when the cached task tree contains contamination
           signals.

        Mirrors MemoryConsolidator.run() override structure (Stage 1).
        """
        report = await super().run(
            events, watermark, prior_reports, run_id, model=model,
            resume_session_id=resume_session_id,
        )

        # Layer 3 (finding-side) gate for blocked-snapshot false positives.
        # Mirrors the filter_stale_count_snapshot_corrections idiom in
        # MemoryConsolidator.run() (stages/memory_consolidator.py:115-150).
        if report.items_flagged:
            _before = len(report.items_flagged)
            report.items_flagged = filter_blocked_snapshot_findings(
                report.items_flagged, project_id=self.project_id
            )
            report.stats['blocked_snapshot_findings_dropped'] = (
                _before - len(report.items_flagged)
            )
        else:
            report.stats['blocked_snapshot_findings_dropped'] = 0

        # Layer 3 (finding-side) gate for false phantom-tasks_created findings
        # (task-2525): drop a finding only when a cited task is positively
        # confirmed present in a different known project (cross-project routing).
        if report.items_flagged:
            _before = len(report.items_flagged)
            report.items_flagged = await filter_false_phantom_task_creation_flags(
                taskmaster=self.taskmaster,
                known_projects=self.known_projects,
                flags=report.items_flagged,
            )
            report.stats['phantom_task_creation_findings_dropped'] = (
                _before - len(report.items_flagged)
            )
        else:
            report.stats['phantom_task_creation_findings_dropped'] = 0

        # Layer 3 (finding-side) gate for retired-contamination-ceiling false
        # positives (task 2818/2826): for projects whose Stage-1 task-ID
        # contamination ceiling is retired-by-design, drop
        # missing_knowledge/memory_stale findings asserting the (correctly absent)
        # ceiling guardrail memory is missing or stale.  Mirrors the two gates
        # above.
        if report.items_flagged:
            _before = len(report.items_flagged)
            report.items_flagged = filter_contamination_ceiling_findings(
                report.items_flagged, project_id=self.project_id
            )
            report.stats['contamination_ceiling_findings_dropped'] = (
                _before - len(report.items_flagged)
            )
        else:
            report.stats['contamination_ceiling_findings_dropped'] = 0

        self.record_task_dump_spot_check(report)
        return report

    def record_task_dump_spot_check(self, report: StageReport) -> None:
        """Spot-check the harness-injected filtered_task_tree for contamination signals.

        If self.filtered_task_tree is set, builds a synthetic raw-dump dict from the
        tree's task lists, runs detect_task_dump_contamination against it, and — when
        contamination is detected — records report.stats['task_dump_spot_check'] and
        emits a WARNING for operator triage.

        Non-destructive: never empties the task tree or affects subsequent stages.
        Only records the stat key when contaminated=True (clean cycles leave stats
        untouched, matching the Stage 1 census-inconsistency pattern).
        """
        if self.filtered_task_tree is None:
            return

        tree = self.filtered_task_tree
        # Build a synthetic dump from the tree's task lists, stamping self.project_id so
        # the dump mirrors what get_tasks now returns (task-1661 envelope stamp).
        #
        # Note on project_mismatch signal: through this code path the dump is always
        # stamped with self.project_id and compared against the same value, so
        # project_mismatch can never be non-None here.  The stamped-project_id mismatch
        # signal is delegated to the Stage-3 LLM via the prompt's cross_project_routing
        # guard (stage3.py).  The title-plausibility heuristic (step_pattern_title_ids)
        # remains the active in-process signal.
        raw_tasks = list(tree.active_tasks) + list(tree.done_tasks) + list(tree.cancelled_tasks)
        dump: dict = {'project_id': self.project_id, 'tasks': raw_tasks}

        result = detect_task_dump_contamination(dump, expected_project_id=self.project_id)
        if not result['contaminated']:
            return

        stat_payload = {
            'contaminated': True,
            'project_mismatch': result.get('project_mismatch'),
            'step_pattern_title_ids': result.get('step_pattern_title_ids', []),
        }
        report.stats['task_dump_spot_check'] = stat_payload
        logger.warning(
            'reconciliation.task_dump_contamination',
            extra={
                'project_id': self.project_id,
                'project_mismatch': result.get('project_mismatch'),
                'step_pattern_title_ids': result.get('step_pattern_title_ids', []),
            },
        )

    async def assemble_payload(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
    ) -> str:
        stage1_report = prior_reports[0] if len(prior_reports) > 0 else None
        stage2_report = prior_reports[1] if len(prior_reports) > 1 else None

        flagged = []
        if stage1_report:
            flagged.extend(stage1_report.items_flagged)
        if stage2_report:
            flagged.extend(stage2_report.items_flagged)

        flagged_text = _format_flagged(flagged, run_stage='stage3')

        return f"""## Stage 3: Cross-System Integrity Check
## Project: {self.project_id}

### Stage 1 Report
{_format_report(stage1_report)}

### Stage 2 Report
{_format_report(stage2_report)}

### Items Flagged for Cross-System Verification ({len(flagged)})
{flagged_text}

## Your Task
Verify consistency across all three systems:
1. Spot-check: do recently modified tasks align with current memory state?
2. Spot-check: do recently written memories align with task state?
3. For flagged items: investigate and classify as consistent/inconsistent.
4. Report all findings. Inconsistencies found here will be addressed in the next cycle's \
Stage 1 and Stage 2.
5. When you have completed your work, produce your final structured report as your response.

{_STAGE3_PROJECT_ID_GUIDELINE.format(project_id=self.project_id)}
Use project_root="{self.project_root}" for tasks scoped to this project.
"""


def _format_report(report: StageReport | None) -> str:
    if report is None:
        return 'No report available.'
    duration = (report.completed_at - report.started_at).total_seconds()
    return (
        f'Stage: {report.stage.value}\n'
        f'Duration: {duration:.1f}s | LLM calls: {report.llm_calls} | '
        f'Tokens: {report.tokens_used}\n'
        f'Stats: {json.dumps(report.stats, default=str)}\n'
        f'Items flagged: {len(report.items_flagged)}'
    )


_FLAGGED_ITEMS_CHAR_BUDGET = 40_000


def _inject_flag_id(flag: dict) -> dict:
    """Promote a Mem0 id to the canonical ``flag_id`` key if absent.

    Stage 1's FIX B instructs the LLM to emit the confirmed canonical Mem0 id
    in each ``flagged_items`` entry under the field name ``flag_id``.  As a
    defensive normalisation, if the LLM instead used the alternative key
    ``memory_id`` (consistent with the prompt's prior wording), this function
    promotes it so Stage 2's FIX C deletion can always locate ``flag_id``.

    The active-query path already injects ``flag_id`` explicitly at
    ``assemble_payload``; this function only acts on analytical-findings-path
    items (those without a ``_source`` key set by Python).

    Returns:
        The original dict unchanged when ``flag_id`` is already set, or when
        no known id key is present.  A shallow copy with ``flag_id`` injected
        from ``memory_id`` when that key is present and ``flag_id`` is absent.
    """
    if 'flag_id' in flag:
        return flag
    if 'memory_id' in flag:
        return {**flag, 'flag_id': flag['memory_id']}
    return flag


def _flag_content_fingerprint(item: dict) -> str | None:
    """Return a normalized-content fingerprint for *item*, or ``None``.

    Cross-channel dedup key (task-2078) for the recon_report_systemic poll in
    :meth:`TaskKnowledgeSync.assemble_payload`. ``finding_id`` is never
    available on the Mem0 side (a flag_for_stage2 marker's metadata carries
    only task_id/flag_type/run_id), so content is the only reliable join key
    between a Stage 1 / mem0_active_query item (keyed on ``content``) and a
    polled recon_report_systemic finding (keyed on ``description``).

    Deliberately NOT a call site of
    :func:`~fused_memory.reconciliation.flag_dedup.compute_content_fingerprint_signature`:
    that helper is gated on ``task_id``/``cited_tasks`` being absent and
    returns a ``(fp, flag_type)`` tuple for marker write/match in
    ``dedup_flags``, whereas this helper is ungated (any item with non-blank
    text qualifies, regardless of task_id) and returns the bare ``fp`` for a
    same-cycle payload-assembly join. Both delegate to the same underlying
    :func:`~fused_memory.reconciliation.flag_dedup._content_fingerprint` /
    :func:`~fused_memory.reconciliation.flag_dedup._normalize_content_description`
    primitives — if that normalization contract ever changes, cross-check
    both call sites so the two dedup keys don't silently drift apart.

    Text is read as ``item.get('content') or item.get('description') or ''``
    so either field name resolves to the same fingerprint for identical text
    — the cross-channel equality the dedup relies on. Coerced to ``str`` in
    case a caller places a truthy non-string value (e.g. an int) under
    either key, so normalization never raises AttributeError. Returns
    ``None`` when the normalized text is blank (mirrors
    :func:`compute_content_fingerprint_signature`'s non-blank gate).

    Pure, sync, no I/O.
    """
    text = str(item.get('content') or item.get('description') or '')
    if not _normalize_content_description(text):
        return None
    return _content_fingerprint(text)


def _format_flagged(
    items: list[dict],
    *,
    budget_chars: int = _FLAGGED_ITEMS_CHAR_BUDGET,
    run_stage: str | None = None,
) -> str:
    """Render flagged items as a bullet list, capped by *budget_chars*.

    Returns the rendered bullet-list text.  The per-render breakdown
    (``rendered``, ``dropped``, ``first_item_fragmented``) is emitted in the
    structured warning's ``extra`` when truncation fires — callers that need
    telemetry should read it from the warning record, not the return value.

    When *run_stage* is provided it is embedded in the warning's ``extra`` dict
    so ops can correlate the drop to its call site without a separate
    stage-specific shortfall warning.

    Edge case: if the very first item's JSON alone exceeds *budget_chars*, a
    truncated fragment is always rendered (with a ``… [item truncated]`` marker)
    so the LLM receives at least some signal rather than an opaque footer-only
    body.  ``rendered`` stays 0 (the fragment is not a full render) and the
    warning's ``extra`` includes ``first_item_fragmented=True`` so
    callers/telemetry can distinguish fragmented-first-item from all-dropped.
    """
    if not items:
        return 'No flagged items.'
    lines: list[str] = []
    running_chars = 0
    first_item_fragmented = False
    for idx, item in enumerate(items):
        json_str = json.dumps(item, default=str)
        line = f'- {json_str}'
        # +1 for the '\n' separator between lines
        separator = 1 if lines else 0
        if running_chars + separator + len(line) > budget_chars:
            # Budget exceeded — stop and emit a truncation footer + warning.
            if idx == 0:
                # First item alone exceeds the budget.  Always show at least a
                # truncated fragment so the LLM has some signal rather than an
                # opaque footer-only body.
                marker = '… [item truncated]'
                available = budget_chars - len('- ') - len(marker)
                if available > 0:
                    lines.append(f'- {json_str[:available]}{marker}')
                    first_item_fragmented = True
            dropped = len(items) - idx
            # Footer shows only items that are completely absent (not the
            # fragmented first item, which already appears as a truncated line).
            completely_missing = dropped - (1 if first_item_fragmented else 0)
            if completely_missing > 0:
                lines.append(f'... and {completely_missing} more (truncated: char budget)')
            extra: dict = {
                'total': len(items),
                'rendered': idx,
                'dropped': dropped,
                'budget_chars': budget_chars,
                'first_item_fragmented': first_item_fragmented,
            }
            if run_stage is not None:
                extra['run_stage'] = run_stage
            logger.warning('reconciliation.flagged_items_truncated', extra=extra)
            return '\n'.join(lines)
        lines.append(line)
        running_chars += separator + len(line)
    return '\n'.join(lines)


async def _render_done_provenance_section(
    done_tasks: list[dict],
    project_root: ProjectRoot | None,
    *,
    max_files_per_task: int = 50,
    max_chars_per_task: int = 2000,
) -> str:
    """Render a '### Done-task Provenance' block from task metadata.done_provenance.

    For each done task:
    - With ``commit``: emits the resolved SHA + a bounded file list produced by
      ``git show --name-only --format=%H%n%ai%n%s <sha>``. Capped at
      ``max_files_per_task`` files and ``max_chars_per_task`` characters per task
      so a single runaway commit can't blow the prompt budget.
    - With ``note``: emits the note text verbatim (no git call).
    - Without either: emits ``provenance: unknown (legacy)``.

    Returns an empty string when ``done_tasks`` is empty — no section is injected
    in that case, keeping the prompt tight when no new completions exist.
    """
    if not done_tasks:
        return ''

    lines: list[str] = ['### Done-task Provenance']
    for task in done_tasks:
        if not isinstance(task, dict):
            continue
        tid = task.get('id', '?')
        title = task.get('title', '')
        metadata = task.get('metadata') if isinstance(task.get('metadata'), dict) else {}
        prov = metadata.get('done_provenance') if isinstance(metadata, dict) else None
        header = f'- [{tid}] {title}'

        if not isinstance(prov, dict):
            lines.append(f'{header} — provenance: unknown (legacy)')
            continue

        commit = prov.get('commit') if isinstance(prov.get('commit'), str) else None
        note = prov.get('note') if isinstance(prov.get('note'), str) else None

        if commit and project_root:
            diff_block = await _git_show_name_only(
                project_root,
                commit,
                max_files=max_files_per_task,
                max_chars=max_chars_per_task,
            )
            lines.append(f'{header}\n  commit: {commit}')
            if diff_block:
                indented = '\n'.join('    ' + ln for ln in diff_block.splitlines())
                lines.append(indented)
        if note:
            lines.append(f'  note: {note}')
        if not commit and not note:
            lines.append(f'{header} — provenance: unknown (legacy)')

    return '\n'.join(lines) + '\n'


async def _git_show_name_only(
    project_root: ProjectRoot,
    commit: str,
    *,
    max_files: int,
    max_chars: int,
) -> str:
    """Run ``git show --name-only --format=%H%n%ai%n%s <commit>`` and truncate.

    Returns a short text block:

        <sha>
        <iso date>
        <subject>
        files:
          path/to/file1
          path/to/file2
          ... (N more)

    Returns an empty string on subprocess failure — the caller still emits the
    commit SHA header, just without the file list. We deliberately don't raise
    so one broken ref doesn't abort the whole Stage-2 briefing.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            'git',
            '-C',
            project_root,
            'show',
            '--name-only',
            '--format=%H%n%ai%n%s',
            '--no-color',
            commit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except TimeoutError:
            proc.kill()
            return ''
    except FileNotFoundError:
        return ''
    except Exception as e:
        logger.warning('git show failed for %s in %s: %s', commit, project_root, e)
        return ''
    if proc.returncode != 0:
        return ''

    raw = stdout.decode('utf-8', errors='replace')
    lines = raw.splitlines()
    if len(lines) < 3:
        return raw[:max_chars]

    header = lines[:3]
    file_lines = [ln for ln in lines[3:] if ln.strip()]
    total = len(file_lines)
    shown = file_lines[:max_files]
    more = total - len(shown)

    block = '\n'.join(header) + '\nfiles:'
    for f in shown:
        block += f'\n  {f}'
    if more > 0:
        block += f'\n  ... ({more} more)'

    if len(block) > max_chars:
        block = block[:max_chars] + '\n  ... (truncated)'
    return block


def _select_proactive_sample(tasks: Iterable[dict], n: int) -> list[dict]:
    """Select the top-N tasks for proactive spot-checking.

    Sorted by status priority (in-progress > blocked > review > pending > done),
    then by task ID descending (proxy for recency — higher ID = more recently created).
    Returns at most n tasks; fewer if the task list is smaller than n.

    Input must contain only dict elements; callers should pass
    FilteredTaskTree.active_tasks/done_tasks/cancelled_tasks fields, which
    filter_task_tree already pre-validates to be dict-only.
    """
    # Import from task_filter — the single source of truth for status priority
    from fused_memory.reconciliation.task_filter import _STATUS_PRIORITY  # noqa: PLC0415

    def sort_key(t: dict) -> tuple[int, int]:
        status = t.get('status', 'pending')
        priority = _STATUS_PRIORITY.get(status, len(_STATUS_PRIORITY))
        return (priority, -id_key(t))

    return heapq.nsmallest(n, tasks, key=sort_key)


def _needs_hint_conversion(task: dict) -> bool:
    """Classify whether *task*'s ``metadata.memory_hints`` needs conversion to the
    canonical ``{entities: [...], queries: [...]}`` dict shape (task 1275).

    Three branches matching the pseudo-code in the task spec:

    1. ``isinstance(task_hints, list)`` → True (legacy list-of-dict format
       ``[{entity: ..., query: ...}, ...]`` — treat as conversion target).
    2. ``not task_hints`` (missing key or empty dict) → True (existing falsy path).
    3. otherwise (any truthy non-list value, including malformed scalars like strings
       or ints) → False (skip). Any truthy non-list value is treated as
       already-converted — narrowing to dict is a separable robustness change.

    Per Mem0 memory ``0b0eeb8d``: Stage 2's ``append=True`` merge silently discards
    list-format hints under old-wins semantics, so list-format must be re-classified
    as a conversion target so the LLM uses read-modify-write, writing the complete
    metadata blob back with ``metadata_mode='replace'`` (a bare ``append=False`` is
    now rejected by the task-2180 metadata-wipe guard).
    """
    metadata = task.get('metadata')
    task_hints = metadata.get('memory_hints') if isinstance(metadata, dict) else None
    if isinstance(task_hints, list):
        return True
    return not task_hints


async def _run_briefing_known_gaps_script(project_root: ProjectRoot) -> list[dict] | None:
    """Run reify's refresh_briefing_known_gaps.py in --json mode.

    Returns a list of mismatch dicts when mismatches are present, an empty list
    when none are found, or None when the script is absent (non-reify project),
    the briefing file is missing, or the subprocess fails.

    Exit codes:
    - 0: no mismatches → return []
    - 1: mismatches present → return parsed JSON list
    - 2+: script error → log WARNING, return None
    """
    script_path = Path(project_root) / 'scripts' / 'refresh_briefing_known_gaps.py'
    if not script_path.exists():
        return None

    briefing_path = Path(project_root) / 'review' / 'briefing.yaml'
    if not briefing_path.exists():
        return None

    tasks_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.json'

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script_path),
            '--briefing',
            str(briefing_path),
            '--tasks',
            str(tasks_path),
            '--json',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            logger.warning(
                'briefing_known_gaps_script_timeout',
                extra={'project_root': project_root},
            )
            return None
    except FileNotFoundError:
        logger.warning(
            'briefing_known_gaps_script_not_found',
            extra={'project_root': project_root},
        )
        return None
    except Exception as exc:
        logger.warning(
            'briefing_known_gaps_script_error',
            extra={'project_root': project_root, 'error': str(exc)},
        )
        return None

    if proc.returncode not in (0, 1):
        logger.warning(
            'briefing_known_gaps_script_failed',
            extra={
                'project_root': project_root,
                'returncode': proc.returncode,
                'stderr': stderr.decode('utf-8', errors='replace')[:500],
            },
        )
        return None

    try:
        return json.loads(stdout.decode('utf-8', errors='replace'))
    except json.JSONDecodeError as exc:
        logger.warning(
            'briefing_known_gaps_script_bad_json',
            extra={'project_root': project_root, 'error': str(exc)},
        )
        return None


async def _queue_briefing_refresh_tasks(
    taskmaster: TaskBackendProtocol,
    project_root: ProjectRoot,
    mismatches: list[dict],
    existing_tasks: list[dict] | None = None,
    run_id: str = '',
) -> dict:
    """Create 'Refresh briefing: remove task N from known_gaps' tasks for each mismatch.

    Skips creation when a task with status 'pending' and the same canonical title
    already exists (exact-title de-dup, case-sensitive string equality).

    ``existing_tasks`` may be pre-supplied by the caller (e.g. derived from
    the harness-injected ``filtered_task_tree``) to avoid a redundant
    ``get_tasks`` round-trip.  When ``None``, the function fetches the tree
    itself.

    ``run_id`` is written into task metadata as ``_causation_id`` so the
    created tasks are traceable back to the reconciliation run that filed them.

    Returns ``{"created": [task_ids], "skipped": [task_ids], "failed": [task_ids]}``.
    """
    if existing_tasks is None:
        existing_raw = await taskmaster.get_tasks(project_root=project_root)
        existing_tasks = existing_raw.get('tasks', []) if isinstance(existing_raw, dict) else []
    pending_titles = {t.get('title', '') for t in existing_tasks if t.get('status') == 'pending'}

    created: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    task_metadata: str | None = (
        json.dumps({'_causation_id': run_id, 'agent_id': 'recon-stage-task_knowledge_sync'})
        if run_id
        else None
    )

    for mismatch in mismatches:
        task_id = str(mismatch.get('task_id', ''))
        title = f'Refresh briefing: remove task {task_id} from known_gaps'

        if title in pending_titles:
            skipped.append(task_id)
            continue

        subproject = mismatch.get('subproject', '')
        task_title = mismatch.get('title', '')
        what = mismatch.get('what', '')
        description = f'Subproject: {subproject}\nTask title: {task_title}\nGap: {what}'
        try:
            result = await taskmaster.add_task(
                project_root=project_root,
                title=title,
                description=description,
                metadata=task_metadata,
            )
            id_val = result.get('id') if isinstance(result, dict) else None
            if isinstance(id_val, str) and id_val and id_val.strip() == id_val:
                created.append(id_val)
            else:
                logger.warning(
                    'briefing_refresh_add_task_unexpected_shape',
                    extra={
                        'project_root': project_root,
                        'task_id': task_id,
                        'result_type': type(result).__name__,
                    },
                )
                failed.append(task_id)
        except DuplicateCandidateKeyError as exc:
            # The store's index-independent dedup guard (or, when the
            # partial UNIQUE index is present, the index itself) rejected
            # this insert as a duplicate of an existing non-cancelled row.
            # PRD decision #3: a candidate_key collision is a successful
            # dedup, not a failure -- fold it into `skipped` rather than
            # `failed` so this direct add_task path (bypassing the curator
            # entirely) matches the interceptor's combined resolution.
            logger.info(
                'briefing_refresh_add_task_deduped',
                extra={
                    'project_root': project_root,
                    'task_id': task_id,
                    'existing_id': exc.existing_id,
                },
            )
            skipped.append(task_id)
        except Exception:
            logger.warning(
                'briefing_refresh_add_task_failed',
                exc_info=True,
                extra={'project_root': project_root, 'task_id': task_id},
            )
            failed.append(task_id)

    return {'created': created, 'skipped': skipped, 'failed': failed}
